"""
Distributed Pre-Training System
===============================
Coordinates large-scale pre-training across the decentralized network.

Key Features:
- Data parallel training across nodes
- Gradient accumulation and synchronization
- Mixed precision training (FP16/BF16)
- Checkpoint sharding for large models
- Progressive scaling (train small, scale up)
- Quality-weighted gradient aggregation

The system enables training models that no single node could train alone
by distributing the work across the entire network.
"""

import asyncio
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import json
import hashlib
import time
import logging
import struct
import zlib

logger = logging.getLogger("AOAI-DistTrain")


@dataclass
class TrainingConfig:
    """Configuration for distributed training"""
    # Model
    model_scale: str = "small"
    
    # Training hyperparameters
    batch_size: int = 32
    micro_batch_size: int = 4
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Optimizer
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Distributed
    gradient_accumulation_steps: int = 8
    sync_every_n_steps: int = 10
    min_nodes_for_sync: int = 3
    
    # Mixed precision
    use_amp: bool = True
    dtype: str = "bfloat16"  # or "float16"
    
    # Checkpointing
    save_every_n_steps: int = 1000
    checkpoint_dir: str = "./checkpoints"
    
    # Data
    sequence_length: int = 2048
    
    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps


@dataclass
class GradientPacket:
    """Compressed gradient packet for network transmission"""
    node_id: str
    step: int
    layer_gradients: Dict[str, bytes]  # Compressed gradients per layer
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    signature: str = ""
    
    def to_bytes(self) -> bytes:
        """Serialize for network transmission"""
        data = {
            "node_id": self.node_id,
            "step": self.step,
            "gradients": {k: v.hex() for k, v in self.layer_gradients.items()},
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }
        json_bytes = json.dumps(data).encode()
        return zlib.compress(json_bytes)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "GradientPacket":
        """Deserialize from network"""
        json_bytes = zlib.decompress(data)
        d = json.loads(json_bytes.decode())
        return cls(
            node_id=d["node_id"],
            step=d["step"],
            layer_gradients={k: bytes.fromhex(v) for k, v in d["gradients"].items()},
            metadata=d["metadata"],
            timestamp=d["timestamp"],
        )


class GradientCompressor:
    """
    Compress gradients for efficient network transmission.
    Uses quantization and sparsification.
    """
    
    @staticmethod
    def compress(tensor: torch.Tensor, bits: int = 8) -> bytes:
        """Compress a gradient tensor"""
        # Convert to float32 for processing
        t = tensor.float()
        
        # Get min/max for quantization
        t_min = t.min().item()
        t_max = t.max().item()
        
        # Quantize to uint8
        if t_max - t_min > 0:
            t_normalized = (t - t_min) / (t_max - t_min)
            t_quantized = (t_normalized * 255).to(torch.uint8)
        else:
            t_quantized = torch.zeros_like(t, dtype=torch.uint8)
        
        # Pack metadata + data
        shape = list(t.shape)
        header = struct.pack('f f i', t_min, t_max, len(shape))
        header += struct.pack(f'{len(shape)}i', *shape)
        data = t_quantized.numpy().tobytes()
        
        return zlib.compress(header + data)
    
    @staticmethod
    def decompress(data: bytes) -> torch.Tensor:
        """Decompress a gradient tensor"""
        raw = zlib.decompress(data)
        
        # Unpack header
        t_min, t_max, ndim = struct.unpack('f f i', raw[:12])
        shape = struct.unpack(f'{ndim}i', raw[12:12 + ndim * 4])
        
        # Unpack data
        tensor_data = raw[12 + ndim * 4:]
        t_quantized = torch.frombuffer(bytearray(tensor_data), dtype=torch.uint8)
        t_quantized = t_quantized.reshape(shape).float()
        
        # Dequantize
        if t_max - t_min > 0:
            t = t_quantized / 255.0 * (t_max - t_min) + t_min
        else:
            t = torch.zeros(shape)
        
        return t


class DistributedTrainer:
    """
    Coordinates distributed training across the network.
    
    Each node:
    1. Trains on local data
    2. Compresses and shares gradients
    3. Aggregates gradients from other nodes
    4. Updates model
    
    This enables training models larger than any single node could handle.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        node_id: str,
        data_dir: str = "./training_data"
    ):
        self.model = model
        self.config = config
        self.node_id = node_id
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.local_step = 0
        self.tokens_trained = 0
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        self.scaler = GradScaler() if config.use_amp else None
        
        # Learning rate schedule
        self.lr_scheduler = self._create_lr_scheduler()
        
        # Gradient buffers
        self.local_gradients: Dict[str, torch.Tensor] = {}
        self.received_gradients: List[GradientPacket] = []
        
        # Network callbacks
        self.on_gradient_ready: Optional[Callable[[GradientPacket], None]] = None
        self.on_model_updated: Optional[Callable[[int, float], None]] = None
        
        # Compressor
        self.compressor = GradientCompressor()
        
        # Metrics
        self.metrics = {
            "loss": [],
            "grad_norm": [],
            "lr": [],
            "tokens_per_second": [],
        }
        
        logger.info(f"Distributed trainer initialized for node {node_id[:16]}...")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay"""
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        return torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )
    
    def _create_lr_scheduler(self):
        """Create cosine learning rate scheduler with warmup"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            
            progress = (step - self.config.warmup_steps) / (
                self.config.max_steps - self.config.warmup_steps
            )
            return max(
                self.config.min_learning_rate / self.config.learning_rate,
                0.5 * (1 + math.cos(math.pi * progress))
            )
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Execute one training step.
        
        Returns metrics for this step.
        """
        self.model.train()
        
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids[:, 1:])
        
        # Forward pass with mixed precision
        with autocast(enabled=self.config.use_amp):
            outputs = self.model(input_ids)
            logits = outputs["logits"][:, :-1, :]
            
            # Compute loss
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            
            # Add auxiliary loss (MoE load balancing)
            if "aux_loss" in outputs:
                loss = loss + outputs["aux_loss"]
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        self.local_step += 1
        
        # Gradient accumulation
        if self.local_step % self.config.gradient_accumulation_steps == 0:
            # Clip gradients
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )
            
            # Store gradients for sharing
            self._store_local_gradients()
            
            # Check if we should sync with network
            if self.global_step % self.config.sync_every_n_steps == 0:
                self._prepare_gradient_packet()
            
            # Optimizer step
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.lr_scheduler.step()
            self.global_step += 1
            
            # Update metrics
            self.metrics["loss"].append(loss.item() * self.config.gradient_accumulation_steps)
            self.metrics["grad_norm"].append(grad_norm.item())
            self.metrics["lr"].append(self.optimizer.param_groups[0]["lr"])
        
        # Track tokens
        self.tokens_trained += input_ids.numel()
        
        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "step": self.global_step,
            "tokens": self.tokens_trained,
        }
    
    def _store_local_gradients(self):
        """Store current gradients for later aggregation"""
        self.local_gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.local_gradients[name] = param.grad.clone()
    
    def _prepare_gradient_packet(self):
        """Prepare gradients for network transmission"""
        if not self.local_gradients:
            return
        
        # Compress gradients
        compressed = {}
        for name, grad in self.local_gradients.items():
            compressed[name] = self.compressor.compress(grad)
        
        # Create packet
        packet = GradientPacket(
            node_id=self.node_id,
            step=self.global_step,
            layer_gradients=compressed,
            metadata={
                "loss": self.metrics["loss"][-1] if self.metrics["loss"] else 0,
                "tokens": self.tokens_trained,
                "model_scale": self.config.model_scale,
            }
        )
        
        # Sign packet
        packet.signature = self._sign_packet(packet)
        
        # Notify network
        if self.on_gradient_ready:
            self.on_gradient_ready(packet)
        
        logger.debug(f"Gradient packet ready: step {self.global_step}")
    
    def receive_gradients(self, packet: GradientPacket):
        """Receive gradients from another node"""
        # Verify signature
        if not self._verify_packet(packet):
            logger.warning(f"Invalid gradient packet from {packet.node_id[:16]}")
            return
        
        self.received_gradients.append(packet)
        logger.debug(f"Received gradients from {packet.node_id[:16]}, step {packet.step}")
    
    def aggregate_gradients(self):
        """
        Aggregate received gradients into the model.
        Uses quality-weighted averaging based on loss.
        """
        if len(self.received_gradients) < self.config.min_nodes_for_sync:
            return
        
        # Filter to recent gradients
        recent = [p for p in self.received_gradients if p.step >= self.global_step - 10]
        if not recent:
            return
        
        # Calculate weights based on loss (lower loss = higher weight)
        losses = [p.metadata.get("loss", float("inf")) for p in recent]
        min_loss = min(losses)
        weights = [1.0 / (loss - min_loss + 1.0) for loss in losses]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Aggregate gradients
        aggregated = {}
        for i, packet in enumerate(recent):
            for name, compressed in packet.layer_gradients.items():
                grad = self.compressor.decompress(compressed)
                
                if name not in aggregated:
                    aggregated[name] = torch.zeros_like(grad)
                
                aggregated[name] += weights[i] * grad
        
        # Apply aggregated gradients
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated:
                    # Blend with local gradients
                    if name in self.local_gradients:
                        blended = 0.7 * self.local_gradients[name] + 0.3 * aggregated[name]
                    else:
                        blended = aggregated[name]
                    
                    param.grad = blended
        
        logger.info(f"Aggregated gradients from {len(recent)} nodes")
        
        # Clear old gradients
        self.received_gradients = [p for p in self.received_gradients if p.step > self.global_step - 5]
    
    def _sign_packet(self, packet: GradientPacket) -> str:
        """Sign a gradient packet"""
        data = f"{packet.node_id}:{packet.step}:{packet.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _verify_packet(self, packet: GradientPacket) -> bool:
        """Verify a gradient packet signature"""
        expected = self._sign_packet(packet)
        return packet.signature == expected
    
    def save_checkpoint(self, path: Optional[str] = None):
        """Save training checkpoint"""
        if path is None:
            path = Path(self.config.checkpoint_dir) / f"step_{self.global_step}.pt"
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),
            "global_step": self.global_step,
            "tokens_trained": self.tokens_trained,
            "config": self.config.__dict__,
            "metrics": self.metrics,
        }
        
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location="cpu")
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.tokens_trained = checkpoint["tokens_trained"]
        self.metrics = checkpoint.get("metrics", self.metrics)
        
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        logger.info(f"Loaded checkpoint from {path}, step {self.global_step}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics"""
        return {
            "global_step": self.global_step,
            "local_step": self.local_step,
            "tokens_trained": self.tokens_trained,
            "current_lr": self.optimizer.param_groups[0]["lr"],
            "avg_loss": sum(self.metrics["loss"][-100:]) / max(len(self.metrics["loss"][-100:]), 1),
            "avg_grad_norm": sum(self.metrics["grad_norm"][-100:]) / max(len(self.metrics["grad_norm"][-100:]), 1),
            "received_gradients": len(self.received_gradients),
        }


# Import math for lr scheduler
import math


class ProgressiveScaler:
    """
    Manages progressive scaling of models.
    
    Strategy:
    1. Train small model to convergence
    2. Initialize larger model from small model
    3. Continue training larger model
    4. Repeat until target scale
    """
    
    SCALE_ORDER = ["tiny", "small", "medium", "large", "xlarge", "frontier"]
    
    def __init__(self, checkpoint_dir: str = "./progressive_checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.current_scale_idx = 0
    
    def get_current_scale(self) -> str:
        return self.SCALE_ORDER[self.current_scale_idx]
    
    def should_scale_up(self, trainer: DistributedTrainer) -> bool:
        """Check if model should scale up based on convergence"""
        if self.current_scale_idx >= len(self.SCALE_ORDER) - 1:
            return False
        
        # Check convergence criteria
        if len(trainer.metrics["loss"]) < 1000:
            return False
        
        recent_loss = trainer.metrics["loss"][-100:]
        older_loss = trainer.metrics["loss"][-500:-400]
        
        if not older_loss:
            return False
        
        # Loss improvement < 5% over 400 steps = converged
        improvement = (sum(older_loss) / len(older_loss) - sum(recent_loss) / len(recent_loss))
        improvement_pct = improvement / (sum(older_loss) / len(older_loss))
        
        return improvement_pct < 0.05
    
    def scale_up(self, old_model: nn.Module, new_config) -> nn.Module:
        """
        Initialize larger model from smaller model.
        Uses progressive layer initialization.
        """
        from .scalable_model import ScalableAOAI
        
        new_model = ScalableAOAI(new_config)
        
        # Copy compatible weights
        old_state = old_model.state_dict()
        new_state = new_model.state_dict()
        
        for name in new_state:
            if name in old_state and old_state[name].shape == new_state[name].shape:
                new_state[name] = old_state[name]
            elif name in old_state:
                # Partial copy for expanded dimensions
                old_shape = old_state[name].shape
                new_shape = new_state[name].shape
                
                slices = tuple(slice(0, min(o, n)) for o, n in zip(old_shape, new_shape))
                new_state[name][slices] = old_state[name][slices]
        
        new_model.load_state_dict(new_state)
        self.current_scale_idx += 1
        
        logger.info(f"Scaled up to {self.get_current_scale()}")
        return new_model


if __name__ == "__main__":
    # Test distributed trainer
    from .scalable_model import create_model
    
    model = create_model("tiny")
    config = TrainingConfig(model_scale="tiny", micro_batch_size=2)
    
    trainer = DistributedTrainer(model, config, node_id="test_node")
    
    # Simulate training step
    batch = {
        "input_ids": torch.randint(0, 1000, (2, 64)),
    }
    
    metrics = trainer.train_step(batch)
    print(f"Step metrics: {metrics}")
    print(f"Training stats: {trainer.get_training_stats()}")
