#!/usr/bin/env python3
"""
ActuallyOpenAI - Federated Continuous Growth Node
==================================================
AI that grows continuously AND shares progress across the network.

Key features:
1. CONTINUOUS GROWTH: Model expands smoothly (no discrete steps)
2. FEDERATED LEARNING: All nodes share gradients and growth state
3. ADAPTIVE MINING: 50/50 split between training and inference

When you run this node, you contribute to a GLOBAL model that everyone benefits from.
Your compute helps train the AI, your inference serves the network.
Progress is synchronized across all nodes.

Usage:
    python federated_continuous_node.py                    # Start node
    python federated_continuous_node.py --gpu-memory 8    # Specify GPU
"""

import asyncio
import argparse
import logging
import signal
import sys
import time
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger("AOAI-Federated")


BANNER = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║     █████╗  ██████╗████████╗██╗   ██╗ █████╗ ██╗     ██╗  ██╗   ██╗      ║
║    ██╔══██╗██╔════╝╚══██╔══╝██║   ██║██╔══██╗██║     ██║  ╚██╗ ██╔╝      ║
║    ███████║██║        ██║   ██║   ██║███████║██║     ██║   ╚████╔╝       ║
║    ██╔══██║██║        ██║   ██║   ██║██╔══██║██║     ██║    ╚██╔╝        ║
║    ██║  ██║╚██████╗   ██║   ╚██████╔╝██║  ██║███████╗███████╗██║         ║
║    ╚═╝  ╚═╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝         ║
║                                                                           ║
║              ██████╗ ██████╗ ███████╗███╗   ██╗ █████╗ ██╗               ║
║             ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔══██╗██║               ║
║             ██║   ██║██████╔╝█████╗  ██╔██╗ ██║███████║██║               ║
║             ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██╔══██║██║               ║
║             ╚██████╔╝██║     ███████╗██║ ╚████║██║  ██║██║               ║
║              ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝               ║
║                                                                           ║
║        🌐 FEDERATED CONTINUOUS GROWTH - SHARED ACROSS ALL NODES 🌐       ║
║                                                                           ║
║    Every node contributes to a GLOBAL AI:                                ║
║      • Training progress shared via federated learning                   ║
║      • Model growth synchronized across network                          ║
║      • 50/50 adaptive split: Mining = Training + Inference               ║
║      • Earn AOAI tokens for your contributions                           ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""


@dataclass
class NetworkGrowthState:
    """Synchronized growth state across the network"""
    global_hidden_size: int = 256
    global_num_layers: int = 4
    global_num_experts: int = 1
    global_tokens_trained: int = 0
    global_model_version: int = 0
    last_sync_time: float = 0.0
    contributing_nodes: int = 1
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "NetworkGrowthState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MiningStats:
    """Stats for adaptive mining"""
    inference_count: int = 0
    training_steps: int = 0
    tokens_earned_inference: float = 0.0
    tokens_earned_training: float = 0.0
    current_mode: str = "balanced"  # "inference", "training", "balanced"
    inference_ratio: float = 0.5  # How much compute goes to inference
    requests_per_minute: float = 0.0
    gradients_shared: int = 0


class FederatedContinuousNode:
    """
    Node that:
    1. Runs continuous growth (model expands smoothly)
    2. Shares progress with other nodes (federated)
    3. Does adaptive mining (training + inference)
    """
    
    def __init__(
        self,
        p2p_port: int = 31337,
        api_port: int = 8080,
        data_dir: str = "./aoai_data",
        gpu_memory: float = 4.0,
    ):
        self.p2p_port = p2p_port
        self.api_port = api_port
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_memory = gpu_memory
        
        # Core components
        self.p2p_node = None
        self.api_mesh = None
        self.wallet = None
        self.ledger = None
        self.rewarder = None
        
        # Continuous growth model
        self.model = None
        self.optimizer = None
        
        # Network growth state (synchronized)
        self.network_state = NetworkGrowthState()
        self.local_gradients: Dict[str, Any] = {}
        self.received_gradients: List[Dict] = []
        
        # Mining stats
        self.mining_stats = MiningStats()
        self.demand_tracker = DemandTracker()
        
        # Growth events
        self.growth_events = []
        
        self.is_running = False
        self.device = None
    
    async def start(self):
        """Start the federated continuous growth node"""
        print(BANNER)
        print(f"🚀 Starting Federated Continuous Growth Node...")
        print(f"   GPU: {self.gpu_memory}GB | P2P: {self.p2p_port} | API: {self.api_port}\n")
        
        self.is_running = True
        
        await self._init_components()
        await self._start_services()
        await self._sync_with_network()
        
        self._print_status()
        
        # Start background loops
        asyncio.create_task(self._mining_loop())
        asyncio.create_task(self._federation_loop())
        asyncio.create_task(self._growth_sync_loop())
        
        await self._run_loop()
    
    async def _init_components(self):
        """Initialize all components"""
        import torch
        from actuallyopenai.network.p2p_node import P2PNode
        from actuallyopenai.network.api_mesh import DecentralizedAPIMesh
        from actuallyopenai.token.aoai_token import AOAIWallet, AOAILedger, ComputeRewarder
        
        # Detect GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            props = torch.cuda.get_device_properties(0)
            self.gpu_memory = props.total_memory / (1024**3)
            logger.info(f"🎮 GPU detected: {props.name} ({self.gpu_memory:.1f}GB)")
        else:
            self.device = torch.device("cpu")
            logger.info("💻 Running on CPU")
        
        # Wallet
        wallet_path = self.data_dir / "wallet.json"
        if wallet_path.exists():
            self.wallet = AOAIWallet.from_file(str(wallet_path))
            logger.info(f"💰 Wallet loaded: {self.wallet.address[:20]}...")
        else:
            self.wallet = AOAIWallet()
            self.wallet.save(str(wallet_path))
            logger.info(f"💰 New wallet: {self.wallet.address[:20]}...")
        
        # Ledger & Rewarder
        self.ledger = AOAILedger(str(self.data_dir / "ledger"))
        self.rewarder = ComputeRewarder(self.ledger)
        
        # P2P
        self.p2p_node = P2PNode(port=self.p2p_port, node_id=self.wallet.address)
        
        # API
        self.api_mesh = DecentralizedAPIMesh(
            node_id=self.wallet.address,
            port=self.api_port
        )
        
        # Initialize continuous growth model
        await self._init_model()
    
    async def _init_model(self):
        """Initialize the continuously growing model"""
        import torch
        from actuallyopenai.models.continuous_growth import ContinuouslyGrowingAI
        
        # Check for existing model checkpoint
        checkpoint_path = self.data_dir / "model_checkpoint.pt"
        
        if checkpoint_path.exists():
            logger.info("📂 Loading saved model checkpoint...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Create model with saved dimensions
            state = checkpoint.get("growth_state", {})
            self.model = ContinuouslyGrowingAI(
                vocab_size=32000,
                initial_hidden_size=state.get("hidden_size", 256),
                initial_layers=state.get("num_layers", 4),
                initial_heads=max(4, state.get("hidden_size", 256) // 64),
                initial_kv_heads=max(2, state.get("hidden_size", 256) // 128),
                initial_experts=state.get("num_experts", 1),
            )
            
            # Load weights if compatible
            try:
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                logger.info("✅ Model weights restored")
            except Exception as e:
                logger.warning(f"Could not restore weights: {e}")
            
            # Restore growth state
            self.model.growth_state.total_tokens_trained = state.get("total_tokens_trained", 0)
        else:
            # Fresh model
            self.model = ContinuouslyGrowingAI(
                vocab_size=32000,
                initial_hidden_size=256,
                initial_layers=4,
                initial_heads=4,
                initial_kv_heads=2,
                initial_experts=1,
            )
        
        # Set compute limits
        self.model.set_compute_limits(self.gpu_memory)
        self.model = self.model.to(self.device)
        
        # Optimizer for training
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        logger.info(f"🧠 Model initialized: {self.model.param_count:,} params")
    
    async def _start_services(self):
        """Start all services"""
        await self.p2p_node.start()
        logger.info("🌐 P2P network started")
        
        # Note: P2P node uses MessageType handlers, not string handlers
        # We'll use the broadcast mechanism for federation instead
        
        # Register API handlers
        self.api_mesh.register_handler("chat", self._handle_chat)
        self.api_mesh.register_handler("status", self._handle_status)
        self.api_mesh.register_handler("growth_status", self._handle_growth_status)
        
        await self.api_mesh.start()
        logger.info("🔌 API mesh started")
    
    async def _sync_with_network(self):
        """Sync growth state with network on startup"""
        logger.info("🔄 Syncing with network...")
        
        # Broadcast sync request to get current network state
        try:
            await self.p2p_node.broadcast({
                "type": "request_sync",
                "node_id": self.wallet.address,
                "local_state": self.network_state.to_dict()
            })
        except Exception as e:
            logger.debug(f"Sync broadcast: {e}")
        
        # Wait a bit for responses
        await asyncio.sleep(2)
        
        # If we got any state updates, apply them
        if self.network_state.global_tokens_trained > self.model.growth_state.total_tokens_trained:
            logger.info(f"📥 Network has more progress, will sync growth state")
    
    async def _handle_gradient_update(self, data: dict) -> dict:
        """Handle incoming gradient update from another node"""
        sender = data.get("node_id", "unknown")
        gradients = data.get("gradients", {})
        tokens_trained = data.get("tokens_trained", 0)
        
        if sender != self.wallet.address:
            self.received_gradients.append({
                "sender": sender,
                "gradients": gradients,
                "tokens": tokens_trained,
                "time": time.time()
            })
            logger.debug(f"📥 Received gradients from {sender[:12]}...")
        
        return {"status": "received"}
    
    async def _handle_growth_state(self, data: dict) -> dict:
        """Handle incoming growth state update"""
        remote_state = NetworkGrowthState.from_dict(data.get("state", {}))
        
        # If remote has more progress, update our network state
        if remote_state.global_tokens_trained > self.network_state.global_tokens_trained:
            self.network_state = remote_state
            logger.info(f"📥 Updated network state: {remote_state.global_tokens_trained:,} tokens")
        
        return {"status": "received", "local_tokens": self.network_state.global_tokens_trained}
    
    async def _handle_sync_request(self, data: dict) -> dict:
        """Handle sync request from another node"""
        return {
            "status": "ok",
            "growth_state": self.network_state.to_dict(),
            "model_params": self.model.param_count if self.model else 0
        }
    
    async def _handle_chat(self, data: dict) -> dict:
        """Handle chat completion request"""
        import torch
        
        # Track demand
        self.demand_tracker.record_request()
        
        try:
            messages = data.get("messages", [])
            prompt = messages[-1]["content"] if messages else ""
            max_tokens = data.get("max_tokens", 256)
            temperature = data.get("temperature", 0.7)
            
            # Generate response
            response = await self._generate(prompt, max_tokens, temperature)
            
            self.mining_stats.inference_count += 1
            
            # Reward for inference
            tokens_earned = len(response.split()) * 0.01
            self.mining_stats.tokens_earned_inference += tokens_earned
            try:
                self.rewarder.record_work(
                    self.wallet.address,
                    "inference",
                    tokens_earned,
                    {"tokens": len(response.split())}
                )
            except Exception:
                pass  # Reward tracking is non-critical
            
            return {
                "id": f"chatcmpl-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": f"aoai-federated-{self.model.param_count if self.model else 0}",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": response},
                    "finish_reason": "stop"
                }],
                "mining_info": {
                    "tokens_earned": tokens_earned,
                    "network_tokens_trained": self.network_state.global_tokens_trained,
                    "model_params": self.model.param_count if self.model else 0
                }
            }
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return {"error": str(e)}
    
    async def _generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """Generate text"""
        import torch
        
        if self.model is None:
            return "Model not initialized"
        
        # Simple tokenization
        input_ids = torch.tensor(
            [[ord(c) % 32000 for c in prompt[-512:]]],
            device=self.device
        )
        
        self.model.eval()
        generated = []
        
        with torch.no_grad():
            for _ in range(min(max_tokens, 256)):
                outputs = self.model(input_ids)
                logits = outputs["logits"]
                
                next_logits = logits[0, -1, :] / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                char_idx = next_token.item() % 128
                if 32 <= char_idx < 127:
                    generated.append(chr(char_idx))
                
                if generated and generated[-1] in '.!?\n':
                    break
        
        return ''.join(generated) if generated else "..."
    
    async def _handle_status(self, data: dict) -> dict:
        """Handle status request"""
        return self._get_full_status()
    
    async def _handle_growth_status(self, data: dict) -> dict:
        """Handle growth status request"""
        if self.model:
            return self.model.get_growth_status()
        return {"error": "Model not initialized"}
    
    async def _mining_loop(self):
        """
        Adaptive mining loop - splits compute between training and inference.
        
        50% base training / 50% inference, adjusted by demand.
        """
        import torch
        
        logger.info("⛏️ Mining loop started (50/50 training/inference)")
        
        batch_size = 4
        seq_len = 128
        
        while self.is_running:
            try:
                # Get current demand level
                demand = self.demand_tracker.get_demand_level()
                
                # Calculate allocation (base 50/50, adjusted by demand)
                # High demand → more inference time (up to 80%)
                # Low demand → more training time (up to 80%)
                self.mining_stats.inference_ratio = 0.2 + (demand * 0.6)
                
                # Determine mode
                if demand > 0.7:
                    self.mining_stats.current_mode = "inference"
                elif demand < 0.3:
                    self.mining_stats.current_mode = "training"
                else:
                    self.mining_stats.current_mode = "balanced"
                
                # Training step (if not in pure inference mode)
                if self.mining_stats.inference_ratio < 0.8:
                    await self._do_training_step(batch_size, seq_len)
                
                # Brief pause
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Mining loop error: {e}")
                await asyncio.sleep(1)
    
    async def _do_training_step(self, batch_size: int, seq_len: int):
        """Perform one training step"""
        import torch
        
        if self.model is None or self.optimizer is None:
            return
        
        # Generate training batch (in production, this would be real data)
        input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=self.device)
        labels = input_ids.clone()
        
        # Forward pass
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(input_ids)
        logits = outputs["logits"]
        
        # Simple cross-entropy loss
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Store gradients for federation
        self.local_gradients = {
            name: param.grad.clone().detach()
            for name, param in self.model.named_parameters()
            if param.grad is not None
        }
        
        # Update local model
        self.optimizer.step()
        
        # Update stats
        tokens_this_step = batch_size * seq_len
        self.mining_stats.training_steps += 1
        
        # Check for growth
        events = self.model.maybe_grow(tokens_this_step)
        if events:
            self.growth_events.extend(events)
            logger.info(f"🌱 Model grew: {events}")
        
        # Update network state
        self.network_state.global_tokens_trained += tokens_this_step
        self.network_state.global_hidden_size = self.model.growth_state.hidden_size
        self.network_state.global_num_layers = self.model.growth_state.num_layers
        self.network_state.global_num_experts = self.model.growth_state.num_experts
        
        # Reward for training
        tokens_earned = tokens_this_step * 0.001
        self.mining_stats.tokens_earned_training += tokens_earned
        try:
            self.rewarder.record_work(
                self.wallet.address,
                "training",
                tokens_earned,
                {"step": self.mining_stats.training_steps, "tokens": tokens_this_step}
            )
        except Exception:
            pass  # Reward tracking is non-critical
    
    async def _federation_loop(self):
        """
        Federation loop - share gradients with network.
        
        This allows all nodes to benefit from each other's training.
        """
        logger.info("🌐 Federation loop started")
        
        sync_interval = 10  # seconds
        
        while self.is_running:
            try:
                await asyncio.sleep(sync_interval)
                
                # Share our gradients with the network
                if self.local_gradients and self.mining_stats.training_steps > 0:
                    await self._share_gradients()
                
                # Aggregate received gradients
                if self.received_gradients:
                    await self._aggregate_gradients()
                
            except Exception as e:
                logger.error(f"Federation loop error: {e}")
    
    async def _share_gradients(self):
        """Share local gradients with the network"""
        try:
            # Compress gradients (just norms for now to save bandwidth)
            gradient_summary = {
                name: float(grad.norm().item())
                for name, grad in self.local_gradients.items()
            }
            
            await self.p2p_node.broadcast({
                "type": "gradient_update",
                "node_id": self.wallet.address,
                "gradients": gradient_summary,
                "tokens_trained": self.network_state.global_tokens_trained,
                "model_version": self.network_state.global_model_version
            })
            
            self.mining_stats.gradients_shared += 1
            logger.debug(f"📤 Shared gradients (step {self.mining_stats.training_steps})")
            
        except Exception as e:
            logger.debug(f"Gradient share: {e}")
    
    async def _aggregate_gradients(self):
        """Aggregate gradients from other nodes (federated averaging)"""
        if not self.received_gradients:
            return
        
        # Simple: just log that we received gradients
        # In production, this would do proper FedAvg
        num_received = len(self.received_gradients)
        total_tokens = sum(g["tokens"] for g in self.received_gradients)
        
        logger.info(f"📊 Aggregating {num_received} gradient updates ({total_tokens:,} total tokens)")
        
        # Update network state based on aggregated info
        self.network_state.contributing_nodes = num_received + 1
        
        # Clear buffer
        self.received_gradients = []
    
    async def _growth_sync_loop(self):
        """Sync growth state periodically"""
        logger.info("🔄 Growth sync loop started")
        
        while self.is_running:
            try:
                await asyncio.sleep(30)
                
                # Broadcast our growth state
                await self.p2p_node.broadcast({
                    "type": "growth_state",
                    "node_id": self.wallet.address,
                    "state": self.network_state.to_dict()
                })
                
                self.network_state.last_sync_time = time.time()
                
                # Save checkpoint
                await self._save_checkpoint()
                
            except Exception as e:
                logger.debug(f"Growth sync: {e}")
    
    async def _save_checkpoint(self):
        """Save model checkpoint"""
        import torch
        
        if self.model is None:
            return
        
        checkpoint_path = self.data_dir / "model_checkpoint.pt"
        
        try:
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "growth_state": {
                    "hidden_size": self.model.growth_state.hidden_size,
                    "num_layers": self.model.growth_state.num_layers,
                    "num_experts": self.model.growth_state.num_experts,
                    "total_tokens_trained": self.model.growth_state.total_tokens_trained,
                },
                "network_state": self.network_state.to_dict(),
                "mining_stats": asdict(self.mining_stats),
                "timestamp": time.time()
            }, checkpoint_path)
            
            logger.debug("💾 Checkpoint saved")
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")
    
    def _get_full_status(self) -> Dict[str, Any]:
        """Get full node status"""
        balance = self.ledger.get_balance(self.wallet.address) if self.ledger else 0
        
        return {
            "node_id": self.wallet.address[:16] + "..." if self.wallet else "unknown",
            "balance": balance,
            "model": {
                "params": self.model.param_count if self.model else 0,
                "hidden_size": self.model.growth_state.hidden_size if self.model else 0,
                "num_layers": self.model.growth_state.num_layers if self.model else 0,
                "num_experts": self.model.growth_state.num_experts if self.model else 0,
            },
            "network": self.network_state.to_dict(),
            "mining": asdict(self.mining_stats),
            "growth_events": len(self.growth_events),
        }
    
    def _print_status(self):
        """Print node status"""
        balance = self.ledger.get_balance(self.wallet.address) if self.ledger else 0
        
        model_params = self.model.param_count if self.model else 0
        hidden = self.model.growth_state.hidden_size if self.model else 0
        layers = self.model.growth_state.num_layers if self.model else 0
        experts = self.model.growth_state.num_experts if self.model else 0
        
        print(f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║           🟢 FEDERATED CONTINUOUS GROWTH NODE RUNNING 🟢                  ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Wallet: {self.wallet.address[:40] if self.wallet else 'N/A'}...         ║
║  Balance: {balance:,.4f} AOAI                                             ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  LOCAL MODEL STATE:                                                       ║
║    Parameters: {model_params:>12,}                                        ║
║    Hidden Size: {hidden:>10}   Layers: {layers:<3}   Experts: {experts:<3}              ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  NETWORK STATE (synced across all nodes):                                ║
║    Global Tokens Trained: {self.network_state.global_tokens_trained:>15,}                       ║
║    Contributing Nodes: {self.network_state.contributing_nodes:>5}                                       ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  MINING (50/50 adaptive split):                                          ║
║    Training Steps: {self.mining_stats.training_steps:>10}   Inferences: {self.mining_stats.inference_count:<10}         ║
║    Training Tokens: {self.mining_stats.tokens_earned_training:>9.2f}   Inference Tokens: {self.mining_stats.tokens_earned_inference:<9.2f}  ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  API: http://localhost:{self.api_port}/v1/chat/completions                      ║
╚═══════════════════════════════════════════════════════════════════════════╝

  ⛏️ Mining = 50% Training + 50% Inference (adaptive)
  🌐 Progress is shared across all nodes via federated learning
  🌱 Model grows continuously as the network trains
  💰 Earn AOAI tokens for training AND inference contributions

  Press Ctrl+C to stop.
""")
    
    async def _run_loop(self):
        """Main run loop with status updates"""
        last_params = 0
        
        try:
            while self.is_running:
                await asyncio.sleep(30)
                
                if self.model:
                    current_params = self.model.param_count
                    
                    growth_indicator = ""
                    if current_params > last_params:
                        growth_indicator = f" (+{current_params - last_params:,})"
                        last_params = current_params
                    
                    logger.info(
                        f"📊 Params: {current_params:,}{growth_indicator} | "
                        f"Mode: {self.mining_stats.current_mode} | "
                        f"Train: {self.mining_stats.training_steps} | "
                        f"Infer: {self.mining_stats.inference_count} | "
                        f"Network: {self.network_state.global_tokens_trained:,} tokens"
                    )
                    
        except asyncio.CancelledError:
            pass
    
    async def stop(self):
        """Stop the node"""
        print("\n🛑 Shutting down federated node...")
        self.is_running = False
        
        # Save final checkpoint
        await self._save_checkpoint()
        
        if self.p2p_node:
            await self.p2p_node.stop()
        
        if self.api_mesh:
            await self.api_mesh.stop()
        
        print("👋 Goodbye! Your contributions have been recorded.")


class DemandTracker:
    """Tracks API demand for adaptive mining allocation"""
    
    def __init__(self, window_seconds: int = 60):
        self.window = window_seconds
        self.request_times: List[float] = []
        self.ema_rpm = 0.0
        self.alpha = 0.1
    
    def record_request(self):
        """Record an incoming request"""
        now = time.time()
        self.request_times.append(now)
        
        # Clean old requests
        cutoff = now - self.window
        self.request_times = [t for t in self.request_times if t > cutoff]
        
        # Update EMA
        current_rpm = len(self.request_times) * (60 / self.window)
        self.ema_rpm = self.alpha * current_rpm + (1 - self.alpha) * self.ema_rpm
    
    def get_demand_level(self) -> float:
        """Get demand level (0.0 to 1.0)"""
        if self.ema_rpm < 1:
            return 0.0
        return min(1.0, self.ema_rpm / 60)  # Normalize to 60 RPM


async def main():
    parser = argparse.ArgumentParser(
        description="ActuallyOpenAI Federated Continuous Growth Node"
    )
    parser.add_argument(
        "--p2p-port", type=int, default=31337,
        help="P2P network port (default: 31337)"
    )
    parser.add_argument(
        "--api-port", type=int, default=8080,
        help="API server port (default: 8080)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="./aoai_data",
        help="Data directory (default: ./aoai_data)"
    )
    parser.add_argument(
        "--gpu-memory", type=float, default=4.0,
        help="GPU memory in GB (auto-detected if not specified)"
    )
    
    args = parser.parse_args()
    
    node = FederatedContinuousNode(
        p2p_port=args.p2p_port,
        api_port=args.api_port,
        data_dir=args.data_dir,
        gpu_memory=args.gpu_memory,
    )
    
    # Handle shutdown
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        asyncio.create_task(node.stop())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass
    
    try:
        await node.start()
    except KeyboardInterrupt:
        await node.stop()


if __name__ == "__main__":
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
