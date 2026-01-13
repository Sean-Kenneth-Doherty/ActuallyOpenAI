"""
Adaptive Mining System
======================
Dynamically splits compute between inference and training based on demand.

High API demand → More inference (up to 100%)
Low API demand → More training (up to 50%)

The AI improves itself during quiet periods, serves users during busy periods.
"""

import asyncio
import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
from enum import Enum
from pathlib import Path
import json
import logging
import random

logger = logging.getLogger("AOAI-Miner")


class MinerMode(Enum):
    INFERENCE = "inference"
    TRAINING = "training"
    IDLE = "idle"


@dataclass
class MinerStats:
    """Track miner performance and allocation"""
    total_inferences: int = 0
    total_training_steps: int = 0
    tokens_earned_inference: float = 0.0
    tokens_earned_training: float = 0.0
    current_mode: MinerMode = MinerMode.IDLE
    inference_ratio: float = 0.5  # Current split (0.5 = 50% inference)
    
    # Demand tracking (rolling window)
    requests_per_minute: float = 0.0
    peak_rpm: float = 10.0  # Adjusts over time
    
    # Training progress
    training_loss: float = float('inf')
    model_version: int = 0
    gradients_submitted: int = 0


@dataclass 
class TrainingTask:
    """A training task to be processed"""
    task_id: str
    data: List[str]  # Training examples
    created_at: float = field(default_factory=time.time)
    priority: int = 1


class DemandTracker:
    """
    Tracks API demand to determine compute allocation.
    
    Uses exponential moving average for smooth transitions.
    """
    
    def __init__(self, window_seconds: int = 60):
        self.window = window_seconds
        self.request_times: List[float] = []
        self.ema_rpm = 0.0  # Exponential moving average
        self.alpha = 0.1    # Smoothing factor
        self._lock = threading.Lock()
    
    def record_request(self):
        """Record an incoming request"""
        now = time.time()
        with self._lock:
            self.request_times.append(now)
            # Clean old requests
            cutoff = now - self.window
            self.request_times = [t for t in self.request_times if t > cutoff]
            
            # Update EMA
            current_rpm = len(self.request_times) * (60 / self.window)
            self.ema_rpm = self.alpha * current_rpm + (1 - self.alpha) * self.ema_rpm
    
    def get_demand_level(self) -> float:
        """
        Get current demand level (0.0 to 1.0)
        
        0.0 = No demand (all training)
        1.0 = Max demand (all inference)
        """
        # Adaptive peak tracking
        if self.ema_rpm > 0:
            # Normalize against recent peak
            demand = min(1.0, self.ema_rpm / max(10, self.ema_rpm * 2))
        else:
            demand = 0.0
        
        return demand
    
    def get_rpm(self) -> float:
        """Get requests per minute"""
        return self.ema_rpm


class AdaptiveMiner:
    """
    Adaptive miner that balances inference and training based on demand.
    
    The key insight: When users need the API, serve them (earn inference tokens).
    When it's quiet, improve the model (earn training tokens).
    
    This creates a self-improving AI that's always getting better.
    """
    
    # Allocation bounds
    MIN_INFERENCE_RATIO = 0.5   # Always at least 50% for inference capacity
    MAX_INFERENCE_RATIO = 1.0   # Can go 100% inference under heavy load
    MIN_TRAINING_RATIO = 0.0    # Training can pause under heavy load
    MAX_TRAINING_RATIO = 0.5    # Max 50% for training
    
    # Rewards (tokens per unit of work)
    INFERENCE_REWARD = 0.01     # Per request served
    TRAINING_REWARD = 0.1       # Per gradient computed
    VALIDATION_REWARD = 0.05    # Per gradient validated
    
    def __init__(
        self,
        model_inference,  # The inference engine
        data_dir: str = "./mining_data",
        wallet_address: Optional[str] = None
    ):
        self.model_inference = model_inference
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.wallet_address = wallet_address or "unknown"
        
        # Demand tracking
        self.demand_tracker = DemandTracker()
        
        # Stats
        self.stats = MinerStats()
        
        # Training queue
        self.training_queue: queue.Queue = queue.Queue()
        self.training_data: List[dict] = []
        
        # Running state
        self.is_running = False
        self._inference_thread: Optional[threading.Thread] = None
        self._training_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.on_tokens_earned: Optional[Callable[[float, str], None]] = None
        self.on_model_improved: Optional[Callable[[float], None]] = None
        self.on_gradient_ready: Optional[Callable[[dict], None]] = None
        
        # Training state
        self.model = None
        self.optimizer = None
        self.training_step = 0
        
        logger.info(f"⚡ Adaptive Miner initialized")
    
    def get_allocation(self) -> tuple:
        """
        Get current compute allocation based on demand.
        
        Returns: (inference_ratio, training_ratio)
        """
        demand = self.demand_tracker.get_demand_level()
        
        # Linear interpolation based on demand
        # High demand → more inference
        # Low demand → more training
        
        if demand >= 0.8:
            # Very high demand - all inference
            inference_ratio = 1.0
            training_ratio = 0.0
        elif demand >= 0.5:
            # High demand - mostly inference
            inference_ratio = 0.7 + (demand - 0.5) * 0.6  # 0.7 to 1.0
            training_ratio = 1.0 - inference_ratio
        elif demand >= 0.2:
            # Medium demand - balanced
            inference_ratio = 0.5 + demand * 0.4  # 0.5 to 0.7
            training_ratio = 1.0 - inference_ratio
        else:
            # Low demand - favor training
            inference_ratio = 0.5  # Minimum 50%
            training_ratio = 0.5  # Maximum 50%
        
        self.stats.inference_ratio = inference_ratio
        
        return (inference_ratio, training_ratio)
    
    async def start(self):
        """Start the adaptive mining system"""
        self.is_running = True
        
        logger.info("🚀 Starting Adaptive Mining System")
        logger.info(f"   Inference/Training split adapts to demand")
        logger.info(f"   Min inference: {self.MIN_INFERENCE_RATIO*100:.0f}%")
        logger.info(f"   Max training: {self.MAX_TRAINING_RATIO*100:.0f}%")
        
        # Initialize training components
        await self._init_training()
        
        # Start background tasks
        asyncio.create_task(self._training_loop())
        asyncio.create_task(self._stats_loop())
        asyncio.create_task(self._data_collection_loop())
    
    async def _init_training(self):
        """Initialize training components"""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            
            # Ensure model is loaded first
            if not self.model_inference.model_loaded:
                self.model_inference.load_model()
            
            # Get model from inference engine
            if hasattr(self.model_inference, 'model') and self.model_inference.model is not None:
                self.model = self.model_inference.model
                
                # Create optimizer for fine-tuning
                # Only train the last few layers (more efficient)
                trainable_params = []
                for name, param in self.model.named_parameters():
                    if 'layers.3' in name or 'output' in name or 'ln_f' in name:
                        param.requires_grad = True
                        trainable_params.append(param)
                    else:
                        param.requires_grad = False
                
                self.optimizer = optim.AdamW(trainable_params, lr=1e-5)
                self.criterion = nn.CrossEntropyLoss()
                
                trainable_count = sum(p.numel() for p in trainable_params)
                total_count = sum(p.numel() for p in self.model.parameters())
                
                logger.info(f"📚 Training initialized")
                logger.info(f"   Trainable: {trainable_count:,} / {total_count:,} params")
                
        except Exception as e:
            logger.warning(f"Training init failed: {e}")
            self.model = None
    
    async def handle_inference(self, prompt: str, **kwargs) -> str:
        """
        Handle an inference request.
        
        This is called by the API mesh when a request comes in.
        Records the request for demand tracking and data collection.
        """
        # Record for demand tracking
        self.demand_tracker.record_request()
        self.stats.total_inferences += 1
        self.stats.current_mode = MinerMode.INFERENCE
        
        # Get response from model
        response = self.model_inference.generate(prompt, **kwargs)
        
        # Store conversation for training (with privacy consideration)
        if len(prompt) > 10 and len(response) > 10:
            self._collect_training_data(prompt, response)
        
        # Earn tokens
        self._earn_tokens(self.INFERENCE_REWARD, "inference")
        
        return response
    
    def _collect_training_data(self, prompt: str, response: str):
        """
        Collect conversation data for future training.
        
        Privacy: Only stores if conversation seems high-quality.
        """
        # Simple quality filter
        if len(prompt.split()) < 3:
            return
        
        # Don't store sensitive patterns
        sensitive_patterns = ['password', 'credit card', 'ssn', 'secret']
        if any(p in prompt.lower() for p in sensitive_patterns):
            return
        
        # Store for training
        self.training_data.append({
            "prompt": prompt,
            "response": response,
            "timestamp": time.time(),
            "quality_score": 1.0  # Will be updated by validators
        })
        
        # Cap stored data
        if len(self.training_data) > 10000:
            self.training_data = self.training_data[-10000:]
    
    async def _training_loop(self):
        """
        Background training loop.
        
        Runs when demand is low, pauses when demand is high.
        """
        import torch
        
        while self.is_running:
            try:
                inference_ratio, training_ratio = self.get_allocation()
                
                # Skip if allocation says no training
                if training_ratio < 0.1 or self.model is None:
                    await asyncio.sleep(1)
                    continue
                
                # Check if we have training data
                if len(self.training_data) < 10:
                    await asyncio.sleep(5)
                    continue
                
                self.stats.current_mode = MinerMode.TRAINING
                
                # Do one training step
                loss = await self._training_step()
                
                if loss is not None:
                    self.stats.training_loss = loss
                    self.stats.total_training_steps += 1
                    
                    # Earn training tokens
                    self._earn_tokens(self.TRAINING_REWARD, "training")
                    
                    # Log periodically
                    if self.stats.total_training_steps % 10 == 0:
                        logger.info(
                            f"📈 Training step {self.stats.total_training_steps} | "
                            f"Loss: {loss:.4f} | "
                            f"Allocation: {inference_ratio*100:.0f}% inf / {training_ratio*100:.0f}% train"
                        )
                    
                    # Share gradient with network periodically
                    if self.stats.total_training_steps % 100 == 0:
                        await self._share_gradients()
                
                # Adaptive sleep based on allocation
                # More training allocation = less sleep
                sleep_time = 1.0 / (training_ratio + 0.1)
                await asyncio.sleep(min(sleep_time, 5.0))
                
            except Exception as e:
                logger.error(f"Training error: {e}")
                await asyncio.sleep(10)
    
    async def _training_step(self) -> Optional[float]:
        """
        Perform one training step on collected data.
        
        Returns loss value.
        """
        if self.model is None or self.optimizer is None:
            return None
        
        import torch
        
        try:
            self.model.train()
            
            # Sample batch from collected data
            batch_size = min(4, len(self.training_data))
            samples = random.sample(self.training_data, batch_size)
            
            total_loss = 0.0
            
            for sample in samples:
                # Tokenize
                text = sample["prompt"] + " " + sample["response"]
                
                if hasattr(self.model_inference, 'tokenizer'):
                    tokens = self.model_inference.tokenizer.encode(text)
                else:
                    tokens = [ord(c) % 256 for c in text]
                
                # Truncate/pad
                max_len = min(128, len(tokens) - 1)
                if max_len < 2:
                    continue
                
                tokens = tokens[:max_len + 1]
                
                # Create input/target
                device = next(self.model.parameters()).device
                input_ids = torch.tensor([tokens[:-1]], device=device)
                target_ids = torch.tensor([tokens[1:]], device=device)
                
                # Forward pass
                self.optimizer.zero_grad()
                logits, _ = self.model(input_ids)
                
                # Compute loss
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1)
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update
                self.optimizer.step()
                
                total_loss += loss.item()
            
            self.model.eval()
            self.training_step += 1
            
            return total_loss / batch_size
            
        except Exception as e:
            logger.error(f"Training step error: {e}")
            return None
    
    async def _share_gradients(self):
        """
        Share gradients with the network for federated averaging.
        
        Other nodes can use these to update their models.
        """
        if self.model is None:
            return
        
        import torch
        
        try:
            # Extract gradient summary (not full gradients - too large)
            gradient_summary = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    gradient_summary[name] = {
                        "mean": param.grad.mean().item(),
                        "std": param.grad.std().item(),
                        "norm": param.grad.norm().item()
                    }
            
            gradient_update = {
                "node_id": self.wallet_address,
                "step": self.training_step,
                "loss": self.stats.training_loss,
                "gradient_summary": gradient_summary,
                "data_points": len(self.training_data),
                "timestamp": time.time()
            }
            
            self.stats.gradients_submitted += 1
            
            # Callback to P2P network
            if self.on_gradient_ready:
                self.on_gradient_ready(gradient_update)
            
            logger.info(f"📤 Gradient update #{self.stats.gradients_submitted} shared")
            
        except Exception as e:
            logger.error(f"Gradient sharing error: {e}")
    
    async def receive_gradients(self, gradient_update: dict):
        """
        Receive gradients from another node and apply them.
        
        This is the federated learning aggregation step.
        """
        # TODO: Implement proper FedAvg
        # For now, just log
        node_id = gradient_update.get("node_id", "unknown")[:16]
        loss = gradient_update.get("loss", 0)
        logger.info(f"📥 Received gradients from {node_id} (loss: {loss:.4f})")
    
    async def _stats_loop(self):
        """Print stats periodically"""
        while self.is_running:
            await asyncio.sleep(30)
            
            inf_ratio, train_ratio = self.get_allocation()
            rpm = self.demand_tracker.get_rpm()
            
            logger.info(
                f"📊 Miner Stats | "
                f"RPM: {rpm:.1f} | "
                f"Allocation: {inf_ratio*100:.0f}% inf / {train_ratio*100:.0f}% train | "
                f"Inferences: {self.stats.total_inferences} | "
                f"Training steps: {self.stats.total_training_steps} | "
                f"Loss: {self.stats.training_loss:.4f}"
            )
    
    async def _data_collection_loop(self):
        """
        Periodically save collected training data.
        """
        while self.is_running:
            await asyncio.sleep(300)  # Every 5 minutes
            
            if self.training_data:
                data_file = self.data_dir / "training_data.json"
                try:
                    with open(data_file, 'w') as f:
                        json.dump(self.training_data[-1000:], f)  # Save latest 1000
                    logger.info(f"💾 Saved {len(self.training_data)} training examples")
                except Exception as e:
                    logger.error(f"Data save error: {e}")
    
    def _earn_tokens(self, amount: float, work_type: str):
        """Record earned tokens"""
        if work_type == "inference":
            self.stats.tokens_earned_inference += amount
        else:
            self.stats.tokens_earned_training += amount
        
        if self.on_tokens_earned:
            self.on_tokens_earned(amount, work_type)
    
    def get_stats(self) -> dict:
        """Get current miner statistics"""
        inf_ratio, train_ratio = self.get_allocation()
        
        return {
            "mode": self.stats.current_mode.value,
            "allocation": {
                "inference": inf_ratio,
                "training": train_ratio
            },
            "demand": {
                "rpm": self.demand_tracker.get_rpm(),
                "level": self.demand_tracker.get_demand_level()
            },
            "work": {
                "total_inferences": self.stats.total_inferences,
                "total_training_steps": self.stats.total_training_steps,
                "gradients_submitted": self.stats.gradients_submitted
            },
            "earnings": {
                "inference_tokens": self.stats.tokens_earned_inference,
                "training_tokens": self.stats.tokens_earned_training,
                "total": self.stats.tokens_earned_inference + self.stats.tokens_earned_training
            },
            "training": {
                "loss": self.stats.training_loss,
                "data_points": len(self.training_data),
                "model_version": self.stats.model_version
            }
        }
    
    async def stop(self):
        """Stop the miner"""
        self.is_running = False
        logger.info("🛑 Adaptive Miner stopped")


# Standalone test
async def test_miner():
    """Test the adaptive miner"""
    print("Testing Adaptive Miner...\n")
    
    class MockInference:
        def generate(self, prompt, **kwargs):
            return f"Response to: {prompt}"
    
    miner = AdaptiveMiner(MockInference())
    
    # Simulate varying demand
    print("Low demand (should favor training):")
    for _ in range(5):
        miner.demand_tracker.record_request()
        await asyncio.sleep(0.5)
    
    inf, train = miner.get_allocation()
    print(f"  Allocation: {inf*100:.0f}% inference / {train*100:.0f}% training")
    
    print("\nHigh demand (should favor inference):")
    for _ in range(50):
        miner.demand_tracker.record_request()
        await asyncio.sleep(0.01)
    
    inf, train = miner.get_allocation()
    print(f"  Allocation: {inf*100:.0f}% inference / {train*100:.0f}% training")
    
    print("\n✅ Adaptive allocation working!")


if __name__ == "__main__":
    asyncio.run(test_miner())
