"""
Auto-Scaling Training Controller - Dynamically adjusts to available compute.

This system:
1. Monitors available compute resources in real-time
2. Automatically scales training parameters
3. Handles worker join/leave events gracefully
4. Optimizes resource utilization
5. Balances training speed vs model quality
"""

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import statistics

import structlog

logger = structlog.get_logger()


class ScalingMode(str, Enum):
    """Mode for auto-scaling behavior."""
    AGGRESSIVE = "aggressive"  # Maximize throughput
    BALANCED = "balanced"  # Balance speed and quality
    CONSERVATIVE = "conservative"  # Prioritize stability
    ADAPTIVE = "adaptive"  # Learn optimal parameters


class ResourceTier(str, Enum):
    """Tier of available compute resources."""
    MINIMAL = "minimal"  # < 10 compute units
    SMALL = "small"  # 10-50 compute units
    MEDIUM = "medium"  # 50-200 compute units
    LARGE = "large"  # 200-1000 compute units
    MASSIVE = "massive"  # > 1000 compute units


@dataclass
class ComputeSnapshot:
    """Snapshot of compute resources at a point in time."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Worker counts
    total_workers: int = 0
    active_workers: int = 0
    idle_workers: int = 0
    
    # Compute power
    total_compute_power: float = 0.0
    gpu_count: int = 0
    total_vram_gb: float = 0.0
    
    # Network
    avg_latency_ms: float = 0.0
    total_bandwidth_mbps: float = 0.0
    
    # Performance
    tokens_per_second: float = 0.0
    gradients_per_second: float = 0.0


@dataclass
class ScalingParameters:
    """Parameters controlled by auto-scaling."""
    # Batch size
    global_batch_size: int = 256
    micro_batch_size: int = 16
    gradient_accumulation_steps: int = 16
    
    # Learning rate
    base_learning_rate: float = 1e-4
    effective_learning_rate: float = 1e-4
    warmup_steps: int = 1000
    
    # Parallelism
    data_parallel_degree: int = 1
    pipeline_parallel_degree: int = 1
    tensor_parallel_degree: int = 1
    
    # Efficiency
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    
    # Communication
    gradient_compression: bool = False
    all_reduce_bucket_size_mb: int = 25


@dataclass
class ScalingDecision:
    """A decision made by the auto-scaler."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    reason: str = ""
    
    # What changed
    parameter_changed: str = ""
    old_value: Any = None
    new_value: Any = None
    
    # Context
    compute_power: float = 0.0
    worker_count: int = 0
    
    # Impact prediction
    expected_throughput_change: float = 0.0


class AutoScalingController:
    """
    Automatically scales training based on available compute.
    
    This is what allows the AI to seamlessly adapt as:
    - More workers join → Training speeds up
    - Workers leave → Training continues gracefully
    - Network conditions change → Parameters adjust
    """
    
    def __init__(
        self,
        mode: ScalingMode = ScalingMode.BALANCED,
        min_batch_size: int = 32,
        max_batch_size: int = 4096,
        target_tokens_per_step: int = 500_000,
        adjustment_interval_seconds: float = 30.0,
    ):
        self.mode = mode
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_tokens_per_step = target_tokens_per_step
        self.adjustment_interval_seconds = adjustment_interval_seconds
        
        # Current parameters
        self.params = ScalingParameters()
        
        # History for adaptive scaling
        self.compute_history: List[ComputeSnapshot] = []
        self.decision_history: List[ScalingDecision] = []
        self.throughput_history: List[float] = []
        
        # Worker tracking
        self.workers: Dict[str, Dict] = {}
        
        # Scaling state
        self.last_adjustment: datetime = datetime.utcnow()
        self.is_scaling: bool = False
        self.current_tier: ResourceTier = ResourceTier.MINIMAL
        
        # Adaptive learning (for ADAPTIVE mode)
        self.parameter_performance: Dict[str, List[float]] = {}
        
        # Callbacks
        self.on_scale_event: Optional[Callable] = None
        
        logger.info(
            "AutoScalingController initialized",
            mode=mode.value,
            target_tokens=target_tokens_per_step
        )
    
    async def take_snapshot(self, workers: Dict[str, Any]) -> ComputeSnapshot:
        """Take a snapshot of current compute resources."""
        active = [w for w in workers.values() if w.get("status") in ("online", "training", "idle")]
        
        snapshot = ComputeSnapshot(
            total_workers=len(workers),
            active_workers=len(active),
            idle_workers=len([w for w in active if w.get("status") == "idle"]),
            total_compute_power=sum(w.get("compute_score", 0) for w in active),
            gpu_count=sum(1 for w in active if w.get("has_gpu")),
            total_vram_gb=sum(w.get("vram_gb", 0) for w in active),
            avg_latency_ms=statistics.mean(
                [w.get("latency_ms", 100) for w in active]
            ) if active else 0,
        )
        
        self.compute_history.append(snapshot)
        
        # Keep only recent history
        cutoff = datetime.utcnow() - timedelta(hours=1)
        self.compute_history = [
            s for s in self.compute_history
            if s.timestamp > cutoff
        ]
        
        # Update resource tier
        self.current_tier = self._determine_tier(snapshot.total_compute_power)
        
        return snapshot
    
    def _determine_tier(self, compute_power: float) -> ResourceTier:
        """Determine the resource tier based on compute power."""
        if compute_power < 10:
            return ResourceTier.MINIMAL
        elif compute_power < 50:
            return ResourceTier.SMALL
        elif compute_power < 200:
            return ResourceTier.MEDIUM
        elif compute_power < 1000:
            return ResourceTier.LARGE
        else:
            return ResourceTier.MASSIVE
    
    async def adjust_parameters(
        self,
        workers: Dict[str, Any],
        current_throughput: float = 0.0
    ) -> List[ScalingDecision]:
        """
        Analyze resources and adjust training parameters.
        Returns list of scaling decisions made.
        """
        # Rate limit adjustments
        if (datetime.utcnow() - self.last_adjustment).total_seconds() < self.adjustment_interval_seconds:
            return []
        
        snapshot = await self.take_snapshot(workers)
        decisions = []
        
        # Store throughput for adaptive learning
        if current_throughput > 0:
            self.throughput_history.append(current_throughput)
        
        # Apply scaling strategy
        if self.mode == ScalingMode.AGGRESSIVE:
            decisions = await self._scale_aggressive(snapshot)
        elif self.mode == ScalingMode.CONSERVATIVE:
            decisions = await self._scale_conservative(snapshot)
        elif self.mode == ScalingMode.ADAPTIVE:
            decisions = await self._scale_adaptive(snapshot, current_throughput)
        else:  # BALANCED
            decisions = await self._scale_balanced(snapshot)
        
        if decisions:
            self.decision_history.extend(decisions)
            self.last_adjustment = datetime.utcnow()
            
            for decision in decisions:
                logger.info(
                    "Scaling decision",
                    parameter=decision.parameter_changed,
                    old=decision.old_value,
                    new=decision.new_value,
                    reason=decision.reason
                )
            
            if self.on_scale_event:
                await self.on_scale_event(decisions)
        
        return decisions
    
    # =========================================================================
    # Scaling Strategies
    # =========================================================================
    
    async def _scale_balanced(
        self, snapshot: ComputeSnapshot
    ) -> List[ScalingDecision]:
        """Balanced scaling - optimize both speed and quality."""
        decisions = []
        
        # Scale batch size based on compute
        target_batch = self._calculate_target_batch(snapshot)
        if abs(target_batch - self.params.global_batch_size) > self.params.global_batch_size * 0.2:
            decisions.append(self._adjust_batch_size(target_batch, snapshot, "Adapting to compute capacity"))
        
        # Scale learning rate (linear scaling rule)
        target_lr = self._calculate_scaled_lr(target_batch)
        if abs(target_lr - self.params.effective_learning_rate) > 1e-6:
            decisions.append(self._adjust_learning_rate(target_lr, snapshot, "Linear scaling with batch size"))
        
        # Enable optimizations based on resources
        decisions.extend(self._adjust_optimizations(snapshot))
        
        # Adjust parallelism
        decisions.extend(self._adjust_parallelism(snapshot))
        
        return [d for d in decisions if d is not None]
    
    async def _scale_aggressive(
        self, snapshot: ComputeSnapshot
    ) -> List[ScalingDecision]:
        """Aggressive scaling - maximize throughput."""
        decisions = []
        
        # Push batch size higher
        target_batch = self._calculate_target_batch(snapshot, multiplier=1.5)
        target_batch = min(target_batch, self.max_batch_size)
        
        if target_batch != self.params.global_batch_size:
            decisions.append(self._adjust_batch_size(target_batch, snapshot, "Maximizing throughput"))
        
        # Higher learning rate
        target_lr = self._calculate_scaled_lr(target_batch, base_multiplier=1.2)
        target_lr = min(target_lr, 5e-4)  # Cap for stability
        decisions.append(self._adjust_learning_rate(target_lr, snapshot, "Aggressive learning"))
        
        # Enable all optimizations
        if not self.params.mixed_precision:
            self.params.mixed_precision = True
            decisions.append(ScalingDecision(
                reason="Enable mixed precision for speed",
                parameter_changed="mixed_precision",
                old_value=False,
                new_value=True,
                compute_power=snapshot.total_compute_power,
                worker_count=snapshot.active_workers
            ))
        
        if not self.params.gradient_compression and snapshot.active_workers > 10:
            self.params.gradient_compression = True
            decisions.append(ScalingDecision(
                reason="Enable gradient compression for large clusters",
                parameter_changed="gradient_compression",
                old_value=False,
                new_value=True,
                compute_power=snapshot.total_compute_power,
                worker_count=snapshot.active_workers
            ))
        
        return [d for d in decisions if d is not None]
    
    async def _scale_conservative(
        self, snapshot: ComputeSnapshot
    ) -> List[ScalingDecision]:
        """Conservative scaling - prioritize stability."""
        decisions = []
        
        # Smaller batch size changes
        target_batch = self._calculate_target_batch(snapshot, multiplier=0.8)
        target_batch = max(target_batch, self.min_batch_size)
        
        # Only change if significant
        if abs(target_batch - self.params.global_batch_size) > self.params.global_batch_size * 0.3:
            decisions.append(self._adjust_batch_size(target_batch, snapshot, "Conservative scaling"))
        
        # Conservative learning rate
        target_lr = self._calculate_scaled_lr(target_batch, base_multiplier=0.8)
        target_lr = min(target_lr, 2e-4)  # Conservative cap
        decisions.append(self._adjust_learning_rate(target_lr, snapshot, "Stable learning rate"))
        
        # Enable gradient checkpointing for memory stability
        if snapshot.total_vram_gb < 24 and not self.params.gradient_checkpointing:
            self.params.gradient_checkpointing = True
            decisions.append(ScalingDecision(
                reason="Enable checkpointing for memory stability",
                parameter_changed="gradient_checkpointing",
                old_value=False,
                new_value=True,
                compute_power=snapshot.total_compute_power,
                worker_count=snapshot.active_workers
            ))
        
        return [d for d in decisions if d is not None]
    
    async def _scale_adaptive(
        self,
        snapshot: ComputeSnapshot,
        current_throughput: float
    ) -> List[ScalingDecision]:
        """Adaptive scaling - learn optimal parameters."""
        decisions = []
        
        # Need throughput history for adaptive scaling
        if len(self.throughput_history) < 10:
            return await self._scale_balanced(snapshot)
        
        # Calculate throughput trend
        recent = self.throughput_history[-5:]
        older = self.throughput_history[-10:-5]
        
        recent_avg = statistics.mean(recent)
        older_avg = statistics.mean(older)
        
        throughput_improving = recent_avg > older_avg * 1.05
        throughput_declining = recent_avg < older_avg * 0.95
        
        # Adaptive batch size
        if throughput_improving:
            # Things are good, try to push further
            target_batch = int(self.params.global_batch_size * 1.2)
            reason = "Throughput improving, increasing batch"
        elif throughput_declining:
            # Things are getting worse, pull back
            target_batch = int(self.params.global_batch_size * 0.85)
            reason = "Throughput declining, reducing batch"
        else:
            # Stable - make minor adjustments based on compute
            target_batch = self._calculate_target_batch(snapshot)
            reason = "Stable throughput, minor adjustment"
        
        target_batch = max(self.min_batch_size, min(self.max_batch_size, target_batch))
        
        if target_batch != self.params.global_batch_size:
            decisions.append(self._adjust_batch_size(target_batch, snapshot, reason))
        
        # Adaptive learning rate based on loss progress
        target_lr = self._calculate_scaled_lr(target_batch)
        decisions.append(self._adjust_learning_rate(target_lr, snapshot, "Adaptive LR"))
        
        return [d for d in decisions if d is not None]
    
    # =========================================================================
    # Parameter Calculation
    # =========================================================================
    
    def _calculate_target_batch(
        self,
        snapshot: ComputeSnapshot,
        multiplier: float = 1.0
    ) -> int:
        """Calculate target batch size based on compute power."""
        # Base calculation: more compute = larger batch
        compute_factor = math.sqrt(max(1, snapshot.total_compute_power))
        
        # Scale based on tier
        tier_multipliers = {
            ResourceTier.MINIMAL: 0.5,
            ResourceTier.SMALL: 0.75,
            ResourceTier.MEDIUM: 1.0,
            ResourceTier.LARGE: 1.5,
            ResourceTier.MASSIVE: 2.0
        }
        tier_mult = tier_multipliers.get(self.current_tier, 1.0)
        
        target = int(256 * compute_factor * tier_mult * multiplier / 10)
        
        # Round to nice numbers
        target = (target // 32) * 32
        
        return max(self.min_batch_size, min(self.max_batch_size, target))
    
    def _calculate_scaled_lr(
        self,
        batch_size: int,
        base_multiplier: float = 1.0
    ) -> float:
        """Calculate learning rate using linear scaling rule."""
        base_lr = 1e-4
        base_batch = 256
        
        # Linear scaling
        scaled_lr = base_lr * (batch_size / base_batch) * base_multiplier
        
        # Square root scaling for very large batches (more stable)
        if batch_size > 1024:
            excess = batch_size - 1024
            sqrt_scaling = math.sqrt(1 + excess / 1024)
            scaled_lr = base_lr * (1024 / base_batch) * sqrt_scaling * base_multiplier
        
        return min(scaled_lr, 1e-3)  # Cap at 1e-3
    
    def _adjust_batch_size(
        self,
        target: int,
        snapshot: ComputeSnapshot,
        reason: str
    ) -> Optional[ScalingDecision]:
        """Create a batch size adjustment decision."""
        if target == self.params.global_batch_size:
            return None
        
        old_value = self.params.global_batch_size
        self.params.global_batch_size = target
        
        # Adjust micro batch and accumulation
        self.params.micro_batch_size = min(32, target // 8)
        self.params.gradient_accumulation_steps = target // self.params.micro_batch_size
        
        return ScalingDecision(
            reason=reason,
            parameter_changed="global_batch_size",
            old_value=old_value,
            new_value=target,
            compute_power=snapshot.total_compute_power,
            worker_count=snapshot.active_workers,
            expected_throughput_change=(target - old_value) / old_value
        )
    
    def _adjust_learning_rate(
        self,
        target: float,
        snapshot: ComputeSnapshot,
        reason: str
    ) -> Optional[ScalingDecision]:
        """Create a learning rate adjustment decision."""
        if abs(target - self.params.effective_learning_rate) < 1e-7:
            return None
        
        old_value = self.params.effective_learning_rate
        self.params.effective_learning_rate = target
        
        return ScalingDecision(
            reason=reason,
            parameter_changed="effective_learning_rate",
            old_value=old_value,
            new_value=target,
            compute_power=snapshot.total_compute_power,
            worker_count=snapshot.active_workers
        )
    
    def _adjust_optimizations(
        self, snapshot: ComputeSnapshot
    ) -> List[ScalingDecision]:
        """Adjust optimization flags based on resources."""
        decisions = []
        
        # Mixed precision for GPUs
        if snapshot.gpu_count > 0 and not self.params.mixed_precision:
            self.params.mixed_precision = True
            decisions.append(ScalingDecision(
                reason="GPUs detected, enabling mixed precision",
                parameter_changed="mixed_precision",
                old_value=False,
                new_value=True,
                compute_power=snapshot.total_compute_power,
                worker_count=snapshot.active_workers
            ))
        
        # Gradient checkpointing for limited VRAM
        avg_vram = snapshot.total_vram_gb / max(1, snapshot.gpu_count)
        if avg_vram < 16 and not self.params.gradient_checkpointing:
            self.params.gradient_checkpointing = True
            decisions.append(ScalingDecision(
                reason="Limited VRAM, enabling gradient checkpointing",
                parameter_changed="gradient_checkpointing",
                old_value=False,
                new_value=True,
                compute_power=snapshot.total_compute_power,
                worker_count=snapshot.active_workers
            ))
        
        # Gradient compression for many workers
        if snapshot.active_workers > 16 and not self.params.gradient_compression:
            self.params.gradient_compression = True
            decisions.append(ScalingDecision(
                reason="Many workers, enabling gradient compression",
                parameter_changed="gradient_compression",
                old_value=False,
                new_value=True,
                compute_power=snapshot.total_compute_power,
                worker_count=snapshot.active_workers
            ))
        
        return decisions
    
    def _adjust_parallelism(
        self, snapshot: ComputeSnapshot
    ) -> List[ScalingDecision]:
        """Adjust parallelism settings based on resources."""
        decisions = []
        
        # Data parallel = number of active workers
        target_dp = max(1, snapshot.active_workers)
        if target_dp != self.params.data_parallel_degree:
            old_value = self.params.data_parallel_degree
            self.params.data_parallel_degree = target_dp
            decisions.append(ScalingDecision(
                reason="Adjusting data parallelism to worker count",
                parameter_changed="data_parallel_degree",
                old_value=old_value,
                new_value=target_dp,
                compute_power=snapshot.total_compute_power,
                worker_count=snapshot.active_workers
            ))
        
        return decisions
    
    # =========================================================================
    # Worker Events
    # =========================================================================
    
    async def on_worker_joined(self, worker_id: str, worker_info: Dict):
        """Handle a new worker joining."""
        self.workers[worker_id] = worker_info
        
        logger.info(
            "Worker joined, may trigger scaling",
            worker_id=worker_id,
            total_workers=len(self.workers)
        )
        
        # Immediate adjustment for significant change
        if len(self.workers) % 5 == 0:  # Every 5 workers
            await self.adjust_parameters(self.workers)
    
    async def on_worker_left(self, worker_id: str):
        """Handle a worker leaving."""
        if worker_id in self.workers:
            del self.workers[worker_id]
        
        logger.info(
            "Worker left, may trigger scaling",
            worker_id=worker_id,
            remaining_workers=len(self.workers)
        )
        
        # Immediate adjustment if significant loss
        await self.adjust_parameters(self.workers)
    
    # =========================================================================
    # Status & Metrics
    # =========================================================================
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current scaling parameters."""
        return {
            "global_batch_size": self.params.global_batch_size,
            "micro_batch_size": self.params.micro_batch_size,
            "gradient_accumulation_steps": self.params.gradient_accumulation_steps,
            "effective_learning_rate": self.params.effective_learning_rate,
            "data_parallel_degree": self.params.data_parallel_degree,
            "mixed_precision": self.params.mixed_precision,
            "gradient_checkpointing": self.params.gradient_checkpointing,
            "gradient_compression": self.params.gradient_compression
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get auto-scaler status."""
        return {
            "mode": self.mode.value,
            "current_tier": self.current_tier.value,
            "parameters": self.get_current_parameters(),
            "recent_decisions": len(self.decision_history[-10:]),
            "compute_snapshots": len(self.compute_history),
            "throughput_samples": len(self.throughput_history),
            "last_adjustment": self.last_adjustment.isoformat()
        }
    
    def get_recent_decisions(self, limit: int = 10) -> List[Dict]:
        """Get recent scaling decisions."""
        return [
            {
                "timestamp": d.timestamp.isoformat(),
                "parameter": d.parameter_changed,
                "old_value": d.old_value,
                "new_value": d.new_value,
                "reason": d.reason,
                "workers": d.worker_count
            }
            for d in self.decision_history[-limit:]
        ]
