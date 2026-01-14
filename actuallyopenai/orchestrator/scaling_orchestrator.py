"""
Scaling Orchestrator
====================
The brain that coordinates progressive scaling across the network.

This orchestrates:
1. Model scale progression (tiny → small → medium → large → frontier)
2. Network resource allocation
3. Data pipeline management
4. Training job scheduling
5. Model version management
"""

import asyncio
import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging

logger = logging.getLogger("AOAI-ScalingOrchestrator")


class ScalePhase(Enum):
    """Training phases for progressive scaling"""
    TINY = "tiny"           # 10M params - any device
    SMALL = "small"         # 100M params - single GPU
    MEDIUM = "medium"       # 1B params - multi-GPU
    LARGE = "large"         # 7B params - distributed
    XLARGE = "xlarge"       # 70B params - large cluster
    FRONTIER = "frontier"   # 400B+ params - global network


@dataclass
class ScalePhaseConfig:
    """Configuration for each scale phase"""
    phase: ScalePhase
    min_params: int
    max_params: int
    min_nodes: int
    min_gpu_memory: float
    target_tokens: int
    expected_loss_threshold: float
    
    # Training hyperparams that change with scale
    learning_rate: float
    batch_size: int
    warmup_steps: int
    max_grad_norm: float


# Scale progression configuration
SCALE_CONFIGS = {
    ScalePhase.TINY: ScalePhaseConfig(
        phase=ScalePhase.TINY,
        min_params=1_000_000,
        max_params=50_000_000,
        min_nodes=1,
        min_gpu_memory=4.0,
        target_tokens=1_000_000_000,       # 1B tokens
        expected_loss_threshold=4.0,
        learning_rate=3e-4,
        batch_size=32,
        warmup_steps=1000,
        max_grad_norm=1.0,
    ),
    ScalePhase.SMALL: ScalePhaseConfig(
        phase=ScalePhase.SMALL,
        min_params=50_000_000,
        max_params=500_000_000,
        min_nodes=1,
        min_gpu_memory=8.0,
        target_tokens=10_000_000_000,      # 10B tokens
        expected_loss_threshold=3.5,
        learning_rate=2e-4,
        batch_size=64,
        warmup_steps=2000,
        max_grad_norm=1.0,
    ),
    ScalePhase.MEDIUM: ScalePhaseConfig(
        phase=ScalePhase.MEDIUM,
        min_params=500_000_000,
        max_params=5_000_000_000,
        min_nodes=4,
        min_gpu_memory=16.0,
        target_tokens=100_000_000_000,     # 100B tokens
        expected_loss_threshold=3.0,
        learning_rate=1.5e-4,
        batch_size=128,
        warmup_steps=5000,
        max_grad_norm=1.0,
    ),
    ScalePhase.LARGE: ScalePhaseConfig(
        phase=ScalePhase.LARGE,
        min_params=5_000_000_000,
        max_params=50_000_000_000,
        min_nodes=16,
        min_gpu_memory=24.0,
        target_tokens=500_000_000_000,     # 500B tokens
        expected_loss_threshold=2.5,
        learning_rate=1e-4,
        batch_size=256,
        warmup_steps=10000,
        max_grad_norm=0.5,
    ),
    ScalePhase.XLARGE: ScalePhaseConfig(
        phase=ScalePhase.XLARGE,
        min_params=50_000_000_000,
        max_params=200_000_000_000,
        min_nodes=64,
        min_gpu_memory=48.0,
        target_tokens=2_000_000_000_000,   # 2T tokens
        expected_loss_threshold=2.0,
        learning_rate=6e-5,
        batch_size=512,
        warmup_steps=20000,
        max_grad_norm=0.5,
    ),
    ScalePhase.FRONTIER: ScalePhaseConfig(
        phase=ScalePhase.FRONTIER,
        min_params=200_000_000_000,
        max_params=2_000_000_000_000,
        min_nodes=256,
        min_gpu_memory=80.0,
        target_tokens=15_000_000_000_000,  # 15T tokens
        expected_loss_threshold=1.5,
        learning_rate=3e-5,
        batch_size=1024,
        warmup_steps=50000,
        max_grad_norm=0.3,
    ),
}


@dataclass
class ScaleProgress:
    """Tracks progress through scaling phases"""
    current_phase: ScalePhase = ScalePhase.TINY
    phases_completed: List[str] = field(default_factory=list)
    
    # Current phase progress
    tokens_trained: int = 0
    current_loss: float = 10.0
    best_loss: float = 10.0
    
    # Version tracking
    model_version: str = "0.0.1"
    checkpoint_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timing
    phase_started_at: float = field(default_factory=time.time)
    total_training_time: float = 0


class ScalingOrchestrator:
    """
    Coordinates progressive scaling from tiny to frontier models.
    
    The key insight: you can't train a 400B model from scratch on a single GPU.
    But you CAN:
    1. Train tiny model → verify quality
    2. Scale up architecture → continue training
    3. Aggregate compute from network → train larger
    4. Repeat until frontier scale
    """
    
    def __init__(
        self,
        node_id: str,
        compute_aggregator=None,  # NetworkComputeAggregator
        model_factory=None,       # Callable to create ScalableAOAI
        data_pipeline=None,       # PretrainingDataPipeline
    ):
        self.node_id = node_id
        self.compute_aggregator = compute_aggregator
        self.model_factory = model_factory
        self.data_pipeline = data_pipeline
        
        self.progress = ScaleProgress()
        self.current_model = None
        self.current_config = SCALE_CONFIGS[ScalePhase.TINY]
        
        # Callbacks
        self.on_phase_complete: Optional[callable] = None
        self.on_scale_up: Optional[callable] = None
        self.on_checkpoint: Optional[callable] = None
        
        logger.info("Scaling orchestrator initialized")
    
    def get_network_capability(self) -> Tuple[int, float]:
        """Get total network nodes and compute"""
        if self.compute_aggregator is None:
            return 1, 0.0
        
        return (
            len(self.compute_aggregator.known_nodes),
            self.compute_aggregator.get_total_network_compute(),
        )
    
    def can_scale_to(self, phase: ScalePhase) -> Tuple[bool, str]:
        """Check if network can support a scale phase"""
        config = SCALE_CONFIGS[phase]
        num_nodes, total_compute = self.get_network_capability()
        
        if num_nodes < config.min_nodes:
            return False, f"Need {config.min_nodes} nodes, have {num_nodes}"
        
        # Check if any nodes have enough GPU memory
        if self.compute_aggregator:
            high_memory_nodes = sum(
                1 for n in self.compute_aggregator.known_nodes.values()
                if n.gpu_memory_gb >= config.min_gpu_memory
            )
            if high_memory_nodes < config.min_nodes:
                return False, f"Need {config.min_nodes} nodes with {config.min_gpu_memory}GB VRAM"
        
        return True, "Ready"
    
    def get_next_phase(self) -> Optional[ScalePhase]:
        """Get next scaling phase if ready"""
        phases = list(ScalePhase)
        current_idx = phases.index(self.progress.current_phase)
        
        if current_idx >= len(phases) - 1:
            return None  # Already at frontier
        
        return phases[current_idx + 1]
    
    def should_scale_up(self) -> Tuple[bool, str]:
        """Check if it's time to scale up"""
        config = self.current_config
        
        # Check if we've trained enough tokens
        if self.progress.tokens_trained < config.target_tokens:
            remaining = config.target_tokens - self.progress.tokens_trained
            return False, f"Need {remaining:,} more tokens"
        
        # Check if loss is good enough
        if self.progress.best_loss > config.expected_loss_threshold:
            return False, f"Loss {self.progress.best_loss:.2f} > threshold {config.expected_loss_threshold}"
        
        # Check if next phase is achievable
        next_phase = self.get_next_phase()
        if next_phase is None:
            return False, "Already at frontier scale"
        
        can_scale, reason = self.can_scale_to(next_phase)
        if not can_scale:
            return False, f"Can't scale to {next_phase.value}: {reason}"
        
        return True, f"Ready to scale to {next_phase.value}"
    
    async def initialize_model(self, phase: Optional[ScalePhase] = None):
        """Initialize model for a scale phase"""
        if phase is None:
            phase = self.progress.current_phase
        
        self.current_config = SCALE_CONFIGS[phase]
        
        if self.model_factory:
            self.current_model = self.model_factory(phase.value)
            model_params = sum(p.numel() for p in self.current_model.parameters())
            logger.info(f"Initialized {phase.value} model: {model_params:,} params")
        else:
            logger.warning("No model factory - using placeholder")
        
        self.progress.current_phase = phase
        self.progress.phase_started_at = time.time()
    
    async def scale_up(self):
        """Scale up to next phase"""
        next_phase = self.get_next_phase()
        if next_phase is None:
            logger.warning("Already at frontier - cannot scale up")
            return
        
        can_scale, reason = self.can_scale_to(next_phase)
        if not can_scale:
            logger.warning(f"Cannot scale up: {reason}")
            return
        
        # Save current checkpoint
        await self.save_checkpoint(f"pre_scale_{next_phase.value}")
        
        # Record completed phase
        self.progress.phases_completed.append(self.progress.current_phase.value)
        
        # Initialize new model
        prev_phase = self.progress.current_phase
        await self.initialize_model(next_phase)
        
        # Transfer knowledge if possible
        if self.current_model is not None:
            # In practice, this would involve:
            # 1. Loading smaller model weights
            # 2. Expanding to larger architecture
            # 3. Initializing new params intelligently
            logger.info(f"Knowledge transfer: {prev_phase.value} → {next_phase.value}")
        
        # Update version
        major, minor, patch = self.progress.model_version.split(".")
        self.progress.model_version = f"{int(major) + 1}.0.0"
        
        # Reset phase tracking
        self.progress.tokens_trained = 0
        self.progress.best_loss = 10.0
        
        logger.info(f"Scaled up to {next_phase.value} (v{self.progress.model_version})")
        
        if self.on_scale_up:
            self.on_scale_up(next_phase)
    
    async def training_step(self, loss: float, tokens: int):
        """Record a training step"""
        self.progress.tokens_trained += tokens
        self.progress.current_loss = loss
        
        if loss < self.progress.best_loss:
            self.progress.best_loss = loss
        
        # Check if should scale up
        should_scale, reason = self.should_scale_up()
        if should_scale:
            logger.info(f"Auto-scaling: {reason}")
            await self.scale_up()
        
        # Periodic checkpoint
        if self.progress.tokens_trained % 100_000_000 == 0:  # Every 100M tokens
            await self.save_checkpoint("periodic")
    
    async def save_checkpoint(self, reason: str = "periodic"):
        """Save a checkpoint"""
        checkpoint = {
            "phase": self.progress.current_phase.value,
            "version": self.progress.model_version,
            "tokens_trained": self.progress.tokens_trained,
            "loss": self.progress.best_loss,
            "timestamp": time.time(),
            "reason": reason,
        }
        
        # Generate checkpoint hash
        checkpoint["hash"] = hashlib.sha256(
            json.dumps(checkpoint, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        self.progress.checkpoint_history.append(checkpoint)
        
        logger.info(f"Checkpoint saved: {checkpoint['hash']} ({reason})")
        
        if self.on_checkpoint:
            self.on_checkpoint(checkpoint)
    
    def get_progress_report(self) -> Dict[str, Any]:
        """Get current scaling progress"""
        config = self.current_config
        
        # Calculate progress percentage
        token_progress = min(100, (self.progress.tokens_trained / config.target_tokens) * 100)
        loss_progress = min(100, (config.expected_loss_threshold / max(self.progress.best_loss, 0.1)) * 100)
        
        # Estimate time to next scale
        if self.progress.tokens_trained > 0:
            tokens_per_second = self.progress.tokens_trained / (
                time.time() - self.progress.phase_started_at
            )
            remaining_tokens = config.target_tokens - self.progress.tokens_trained
            eta_seconds = remaining_tokens / tokens_per_second if tokens_per_second > 0 else 0
        else:
            tokens_per_second = 0
            eta_seconds = 0
        
        return {
            "current_phase": self.progress.current_phase.value,
            "model_version": self.progress.model_version,
            "phases_completed": self.progress.phases_completed,
            "progress": {
                "tokens_trained": f"{self.progress.tokens_trained:,}",
                "target_tokens": f"{config.target_tokens:,}",
                "token_progress_pct": f"{token_progress:.1f}%",
                "current_loss": f"{self.progress.current_loss:.4f}",
                "best_loss": f"{self.progress.best_loss:.4f}",
                "loss_threshold": f"{config.expected_loss_threshold:.2f}",
                "loss_progress_pct": f"{loss_progress:.1f}%",
            },
            "performance": {
                "tokens_per_second": f"{tokens_per_second:.1f}",
                "eta_to_next_scale": f"{eta_seconds/3600:.1f} hours",
            },
            "next_phase": {
                "name": self.get_next_phase().value if self.get_next_phase() else "FRONTIER (max)",
                "ready": self.should_scale_up()[0],
                "reason": self.should_scale_up()[1],
            },
            "checkpoints": len(self.progress.checkpoint_history),
        }
    
    def get_scaling_roadmap(self) -> List[Dict[str, Any]]:
        """Get full scaling roadmap"""
        roadmap = []
        
        for phase in ScalePhase:
            config = SCALE_CONFIGS[phase]
            can_achieve, reason = self.can_scale_to(phase)
            
            status = "completed" if phase.value in self.progress.phases_completed else \
                     "current" if phase == self.progress.current_phase else \
                     "achievable" if can_achieve else "blocked"
            
            roadmap.append({
                "phase": phase.value,
                "params_range": f"{config.min_params/1e9:.1f}B - {config.max_params/1e9:.1f}B",
                "target_tokens": f"{config.target_tokens/1e12:.1f}T",
                "min_nodes": config.min_nodes,
                "min_gpu_memory": f"{config.min_gpu_memory}GB",
                "status": status,
                "blocker": None if can_achieve else reason,
            })
        
        return roadmap


class AutoScalingLoop:
    """
    Fully autonomous scaling loop that continuously improves the model.
    
    This is the "AI improving itself" part - it runs forever,
    training larger and larger models as resources allow.
    """
    
    def __init__(
        self,
        orchestrator: ScalingOrchestrator,
        data_loader,      # Iterator of training batches
        trainer,          # Training engine
    ):
        self.orchestrator = orchestrator
        self.data_loader = data_loader
        self.trainer = trainer
        
        self.running = False
        self.stats = {
            "total_steps": 0,
            "total_tokens": 0,
            "scale_ups": 0,
            "started_at": None,
        }
    
    async def start(self):
        """Start the autonomous scaling loop"""
        self.running = True
        self.stats["started_at"] = time.time()
        
        logger.info("=== AUTO-SCALING LOOP STARTED ===")
        logger.info("Target: Progressive scaling to frontier models")
        
        # Initialize at current phase
        await self.orchestrator.initialize_model()
        
        while self.running:
            try:
                # Get batch from data loader
                batch = next(self.data_loader, None)
                if batch is None:
                    logger.info("Data exhausted, restarting epoch")
                    # In practice, would reset data loader
                    await asyncio.sleep(1)
                    continue
                
                # Training step
                loss = await self.trainer.train_step(batch)
                tokens = len(batch) * 2048  # Approximate
                
                # Record progress
                await self.orchestrator.training_step(loss, tokens)
                
                self.stats["total_steps"] += 1
                self.stats["total_tokens"] += tokens
                
                # Log periodically
                if self.stats["total_steps"] % 1000 == 0:
                    report = self.orchestrator.get_progress_report()
                    logger.info(
                        f"Step {self.stats['total_steps']:,}: "
                        f"loss={loss:.4f}, "
                        f"phase={report['current_phase']}, "
                        f"progress={report['progress']['token_progress_pct']}"
                    )
                
                await asyncio.sleep(0)  # Yield to event loop
                
            except Exception as e:
                logger.error(f"Training error: {e}")
                await asyncio.sleep(5)
    
    async def stop(self):
        """Stop the scaling loop"""
        self.running = False
        
        # Save final checkpoint
        await self.orchestrator.save_checkpoint("shutdown")
        
        elapsed = time.time() - self.stats["started_at"] if self.stats["started_at"] else 0
        
        logger.info(f"=== AUTO-SCALING LOOP STOPPED ===")
        logger.info(f"Total steps: {self.stats['total_steps']:,}")
        logger.info(f"Total tokens: {self.stats['total_tokens']:,}")
        logger.info(f"Runtime: {elapsed/3600:.1f} hours")


if __name__ == "__main__":
    # Demo scaling orchestrator
    orchestrator = ScalingOrchestrator("demo_node")
    
    print("=== SCALING ROADMAP ===")
    for phase in orchestrator.get_scaling_roadmap():
        print(f"\n{phase['phase'].upper()}:")
        print(f"  Params: {phase['params_range']}")
        print(f"  Target: {phase['target_tokens']} tokens")
        print(f"  Requires: {phase['min_nodes']} nodes, {phase['min_gpu_memory']}")
        print(f"  Status: {phase['status']}")
        if phase['blocker']:
            print(f"  Blocker: {phase['blocker']}")
    
    print("\n" + "="*50)
    print("Network grows → Models scale → Intelligence increases")
    print("="*50)
