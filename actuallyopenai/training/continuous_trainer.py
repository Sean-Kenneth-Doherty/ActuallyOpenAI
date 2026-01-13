"""
Continuous Trainer - The self-assembling AI training engine.

This is the core system that:
1. Never stops training - runs 24/7 as long as workers are contributing
2. Dynamically scales with available compute
3. Automatically improves the model over time
4. Checkpoints progress and tracks improvements
"""

import asyncio
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import uuid

import structlog
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from actuallyopenai.config import get_settings
from actuallyopenai.core.models import (
    TrainingTask, TrainingJob, TaskStatus, TaskType,
    Worker, WorkerStatus, ComputeContribution
)

logger = structlog.get_logger()


class TrainingPhase(str, Enum):
    """Current phase of continuous training."""
    INITIALIZING = "initializing"
    COLLECTING_WORKERS = "collecting_workers"
    DISTRIBUTING_TASKS = "distributing_tasks"
    AGGREGATING_GRADIENTS = "aggregating_gradients"
    UPDATING_MODEL = "updating_model"
    CHECKPOINTING = "checkpointing"
    EVALUATING = "evaluating"
    SCALING = "scaling"  # Adjusting to new compute capacity


@dataclass
class TrainingState:
    """Current state of the continuous training process."""
    phase: TrainingPhase = TrainingPhase.INITIALIZING
    global_step: int = 0
    epoch: int = 0
    total_tokens_processed: int = 0
    total_compute_hours: float = 0.0
    
    # Performance tracking
    current_loss: float = float('inf')
    best_loss: float = float('inf')
    loss_history: List[float] = field(default_factory=list)
    
    # Improvement tracking
    improvements: List[Dict] = field(default_factory=list)
    last_improvement_step: int = 0
    
    # Worker tracking
    active_workers: int = 0
    total_compute_power: float = 0.0
    
    # Timing
    started_at: Optional[datetime] = None
    last_checkpoint_at: Optional[datetime] = None
    last_evaluation_at: Optional[datetime] = None


@dataclass
class GradientPacket:
    """A packet of gradients from a worker."""
    worker_id: str
    task_id: str
    gradients: Dict[str, torch.Tensor]
    loss: float
    batch_size: int
    compute_time: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Verification
    gradient_hash: str = ""
    
    def __post_init__(self):
        if not self.gradient_hash:
            # Create hash for gradient verification
            hash_input = f"{self.worker_id}-{self.task_id}-{self.loss}"
            self.gradient_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]


class ContinuousTrainer:
    """
    The self-assembling, continuously improving AI training engine.
    
    Key Features:
    - Runs indefinitely, improving the model as long as compute is available
    - Dynamically scales training based on available workers
    - Aggregates gradients from distributed workers (federated learning)
    - Automatically checkpoints and tracks improvements
    - Adjusts hyperparameters based on training dynamics
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_id: str,
        checkpoint_dir: str = "./checkpoints",
        min_workers: int = 1,
        target_batch_size: int = 256,
        checkpoint_every_steps: int = 1000,
        eval_every_steps: int = 500,
    ):
        self.model = model
        self.model_id = model_id
        self.checkpoint_dir = checkpoint_dir
        self.min_workers = min_workers
        self.target_batch_size = target_batch_size
        self.checkpoint_every_steps = checkpoint_every_steps
        self.eval_every_steps = eval_every_steps
        
        # Training state
        self.state = TrainingState()
        self.is_running = False
        
        # Gradient accumulation
        self.gradient_buffer: List[GradientPacket] = []
        self.accumulated_batch_size = 0
        
        # Worker management
        self.workers: Dict[str, Worker] = {}
        self.worker_contributions: Dict[str, List[GradientPacket]] = defaultdict(list)
        
        # Task management
        self.pending_tasks: List[TrainingTask] = []
        self.active_tasks: Dict[str, TrainingTask] = {}
        
        # Optimizer and scheduler (will be initialized based on model)
        self.optimizer: Optional[AdamW] = None
        self.scheduler: Optional[CosineAnnealingWarmRestarts] = None
        
        # Callbacks for external systems
        self.on_improvement: Optional[Callable] = None
        self.on_checkpoint: Optional[Callable] = None
        self.on_worker_joined: Optional[Callable] = None
        
        # Metrics
        self.metrics_history: List[Dict] = []
        
        logger.info(
            "ContinuousTrainer initialized",
            model_id=model_id,
            target_batch_size=target_batch_size
        )
    
    async def initialize(self):
        """Initialize the trainer and prepare for continuous training."""
        self.state.phase = TrainingPhase.INITIALIZING
        self.state.started_at = datetime.utcnow()
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=1e-4,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        # Cosine annealing with warm restarts - allows indefinite training
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=1000,  # Restart every 1000 steps
            T_mult=2,  # Double the period after each restart
            eta_min=1e-6
        )
        
        # Try to load latest checkpoint
        loaded = await self._load_latest_checkpoint()
        if loaded:
            logger.info(
                "Resumed from checkpoint",
                step=self.state.global_step,
                loss=self.state.current_loss
            )
        
        self.state.phase = TrainingPhase.COLLECTING_WORKERS
        logger.info("Continuous trainer initialized, waiting for workers")
    
    async def run(self):
        """
        Main training loop - runs continuously until stopped.
        This is the heart of the self-improving AI system.
        """
        self.is_running = True
        await self.initialize()
        
        logger.info("🚀 Starting continuous training loop")
        
        while self.is_running:
            try:
                # Phase 1: Check worker availability
                await self._update_worker_status()
                
                if self.state.active_workers < self.min_workers:
                    self.state.phase = TrainingPhase.COLLECTING_WORKERS
                    logger.debug(
                        "Waiting for workers",
                        active=self.state.active_workers,
                        required=self.min_workers
                    )
                    await asyncio.sleep(5)
                    continue
                
                # Phase 2: Distribute training tasks
                self.state.phase = TrainingPhase.DISTRIBUTING_TASKS
                await self._distribute_tasks()
                
                # Phase 3: Collect and aggregate gradients
                self.state.phase = TrainingPhase.AGGREGATING_GRADIENTS
                ready_for_update = await self._collect_gradients()
                
                if ready_for_update:
                    # Phase 4: Update model with aggregated gradients
                    self.state.phase = TrainingPhase.UPDATING_MODEL
                    await self._update_model()
                    
                    # Phase 5: Check for checkpointing
                    if self._should_checkpoint():
                        self.state.phase = TrainingPhase.CHECKPOINTING
                        await self._checkpoint()
                    
                    # Phase 6: Evaluate and track improvement
                    if self._should_evaluate():
                        self.state.phase = TrainingPhase.EVALUATING
                        await self._evaluate_and_track()
                    
                    # Phase 7: Dynamic scaling based on compute
                    self.state.phase = TrainingPhase.SCALING
                    await self._adjust_training_dynamics()
                
                # Small delay to prevent tight loop
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error("Error in training loop", error=str(e))
                await asyncio.sleep(5)
        
        logger.info("Continuous training stopped")
    
    async def stop(self):
        """Gracefully stop the training loop."""
        logger.info("Stopping continuous trainer...")
        self.is_running = False
        
        # Final checkpoint
        await self._checkpoint()
    
    # =========================================================================
    # Worker Management
    # =========================================================================
    
    async def register_worker(self, worker: Worker):
        """Register a new worker for training."""
        self.workers[worker.id] = worker
        self.state.active_workers = len([
            w for w in self.workers.values() 
            if w.status in (WorkerStatus.ONLINE, WorkerStatus.IDLE, WorkerStatus.TRAINING)
        ])
        self.state.total_compute_power = sum(
            w.compute_score for w in self.workers.values()
            if w.status != WorkerStatus.OFFLINE
        )
        
        logger.info(
            "Worker registered for training",
            worker_id=worker.id,
            compute_score=worker.compute_score,
            total_workers=self.state.active_workers
        )
        
        if self.on_worker_joined:
            await self.on_worker_joined(worker)
    
    async def remove_worker(self, worker_id: str):
        """Remove a worker from training."""
        if worker_id in self.workers:
            del self.workers[worker_id]
            self._recalculate_compute_power()
            
            # Reassign any active tasks from this worker
            for task_id, task in list(self.active_tasks.items()):
                if task.assigned_worker_id == worker_id:
                    task.status = TaskStatus.PENDING
                    task.assigned_worker_id = None
                    self.pending_tasks.append(task)
                    del self.active_tasks[task_id]
    
    async def _update_worker_status(self):
        """Update the status of all workers."""
        cutoff = datetime.utcnow() - timedelta(seconds=60)
        
        for worker in list(self.workers.values()):
            if worker.last_heartbeat and worker.last_heartbeat < cutoff:
                worker.status = WorkerStatus.OFFLINE
                await self.remove_worker(worker.id)
        
        self._recalculate_compute_power()
    
    def _recalculate_compute_power(self):
        """Recalculate total compute power."""
        active = [
            w for w in self.workers.values()
            if w.status != WorkerStatus.OFFLINE
        ]
        self.state.active_workers = len(active)
        self.state.total_compute_power = sum(w.compute_score for w in active)
    
    # =========================================================================
    # Task Distribution
    # =========================================================================
    
    async def _distribute_tasks(self):
        """Distribute training tasks to available workers."""
        available_workers = [
            w for w in self.workers.values()
            if w.status in (WorkerStatus.ONLINE, WorkerStatus.IDLE)
        ]
        
        if not available_workers:
            return
        
        # Generate tasks if needed
        while len(self.pending_tasks) < len(available_workers) * 2:
            task = self._generate_training_task()
            self.pending_tasks.append(task)
        
        # Assign tasks to workers
        for worker in available_workers:
            if self.pending_tasks:
                task = self.pending_tasks.pop(0)
                task.assigned_worker_id = worker.id
                task.assigned_at = datetime.utcnow()
                task.status = TaskStatus.ASSIGNED
                
                self.active_tasks[task.id] = task
                worker.status = WorkerStatus.TRAINING
                
                logger.debug(
                    "Task assigned",
                    task_id=task.id,
                    worker_id=worker.id
                )
    
    def _generate_training_task(self) -> TrainingTask:
        """Generate a new training task."""
        # Calculate batch size based on available compute
        worker_count = max(1, self.state.active_workers)
        batch_per_worker = max(1, self.target_batch_size // worker_count)
        
        task = TrainingTask(
            job_id=f"continuous-{self.model_id}",
            task_type=TaskType.GRADIENT_COMPUTE,
            model_id=self.model_id,
            data_shard_url=f"ipfs://training-data/shard_{self.state.global_step}",
            batch_size=batch_per_worker,
            learning_rate=self.optimizer.param_groups[0]['lr'] if self.optimizer else 1e-4,
            gradient_accumulation=1
        )
        
        return task
    
    # =========================================================================
    # Gradient Collection & Aggregation
    # =========================================================================
    
    async def receive_gradients(self, packet: GradientPacket):
        """Receive gradients from a worker."""
        self.gradient_buffer.append(packet)
        self.accumulated_batch_size += packet.batch_size
        
        # Track worker contribution
        self.worker_contributions[packet.worker_id].append(packet)
        
        # Update task status
        if packet.task_id in self.active_tasks:
            task = self.active_tasks[packet.task_id]
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.loss = packet.loss
            task.compute_time_seconds = packet.compute_time
            
            # Free up the worker
            if task.assigned_worker_id and task.assigned_worker_id in self.workers:
                self.workers[task.assigned_worker_id].status = WorkerStatus.IDLE
            
            del self.active_tasks[packet.task_id]
        
        logger.debug(
            "Gradients received",
            worker_id=packet.worker_id,
            loss=packet.loss,
            accumulated_batch=self.accumulated_batch_size
        )
    
    async def _collect_gradients(self) -> bool:
        """
        Collect gradients and determine if ready for model update.
        Returns True if we have enough gradients for an update.
        """
        # Wait for enough gradients to accumulate
        if self.accumulated_batch_size < self.target_batch_size:
            # Don't wait forever - do update after timeout
            oldest_gradient = min(
                (p.timestamp for p in self.gradient_buffer),
                default=datetime.utcnow()
            )
            
            if datetime.utcnow() - oldest_gradient > timedelta(seconds=30):
                if self.gradient_buffer:
                    return True  # Timeout - update with what we have
            
            return False
        
        return True
    
    def _aggregate_gradients(self) -> Dict[str, torch.Tensor]:
        """
        Aggregate gradients from multiple workers using weighted averaging.
        This is the federated learning aggregation step.
        """
        if not self.gradient_buffer:
            return {}
        
        # Weighted average based on batch size
        total_samples = sum(p.batch_size for p in self.gradient_buffer)
        aggregated = {}
        
        for packet in self.gradient_buffer:
            weight = packet.batch_size / total_samples
            
            for name, grad in packet.gradients.items():
                if name not in aggregated:
                    aggregated[name] = torch.zeros_like(grad)
                aggregated[name] += grad * weight
        
        return aggregated
    
    # =========================================================================
    # Model Update
    # =========================================================================
    
    async def _update_model(self):
        """Update the model with aggregated gradients."""
        if not self.gradient_buffer:
            return
        
        # Aggregate gradients from all workers
        aggregated_gradients = self._aggregate_gradients()
        
        # Calculate average loss
        avg_loss = sum(p.loss for p in self.gradient_buffer) / len(self.gradient_buffer)
        total_compute_time = sum(p.compute_time for p in self.gradient_buffer)
        
        # Apply gradients to model
        self.optimizer.zero_grad()
        
        for name, param in self.model.named_parameters():
            if name in aggregated_gradients and param.requires_grad:
                param.grad = aggregated_gradients[name]
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        # Update training state
        self.state.global_step += 1
        self.state.current_loss = avg_loss
        self.state.loss_history.append(avg_loss)
        self.state.total_compute_hours += total_compute_time / 3600
        self.state.total_tokens_processed += self.accumulated_batch_size
        
        # Check for improvement
        if avg_loss < self.state.best_loss:
            improvement = (self.state.best_loss - avg_loss) / self.state.best_loss * 100
            self.state.best_loss = avg_loss
            self.state.last_improvement_step = self.state.global_step
            
            self.state.improvements.append({
                "step": self.state.global_step,
                "loss": avg_loss,
                "improvement_percent": improvement,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.info(
                "🎉 Model improved!",
                step=self.state.global_step,
                loss=avg_loss,
                improvement=f"{improvement:.2f}%"
            )
            
            if self.on_improvement:
                await self.on_improvement(self.state)
        
        # Log progress
        logger.info(
            "Model updated",
            step=self.state.global_step,
            loss=round(avg_loss, 6),
            best_loss=round(self.state.best_loss, 6),
            lr=self.optimizer.param_groups[0]['lr'],
            workers=self.state.active_workers,
            batch_size=self.accumulated_batch_size
        )
        
        # Clear gradient buffer
        self.gradient_buffer = []
        self.accumulated_batch_size = 0
        
        # Record metrics
        self.metrics_history.append({
            "step": self.state.global_step,
            "loss": avg_loss,
            "lr": self.optimizer.param_groups[0]['lr'],
            "workers": self.state.active_workers,
            "compute_power": self.state.total_compute_power,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    # =========================================================================
    # Checkpointing
    # =========================================================================
    
    def _should_checkpoint(self) -> bool:
        """Determine if we should save a checkpoint."""
        if self.state.last_checkpoint_at is None:
            return True
        
        steps_since_checkpoint = (
            self.state.global_step - 
            (self.state.last_checkpoint_at.timestamp() if isinstance(self.state.last_checkpoint_at, datetime) else 0)
        )
        
        return self.state.global_step % self.checkpoint_every_steps == 0
    
    async def _checkpoint(self):
        """Save a training checkpoint."""
        import os
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_step_{self.state.global_step}.pt"
        )
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_state": {
                "global_step": self.state.global_step,
                "epoch": self.state.epoch,
                "current_loss": self.state.current_loss,
                "best_loss": self.state.best_loss,
                "total_tokens_processed": self.state.total_tokens_processed,
                "total_compute_hours": self.state.total_compute_hours,
                "improvements": self.state.improvements[-100:],  # Keep last 100
            },
            "timestamp": datetime.utcnow().isoformat(),
            "model_id": self.model_id
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as "latest"
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        torch.save(checkpoint, latest_path)
        
        self.state.last_checkpoint_at = datetime.utcnow()
        
        logger.info(
            "Checkpoint saved",
            path=checkpoint_path,
            step=self.state.global_step,
            loss=self.state.current_loss
        )
        
        if self.on_checkpoint:
            await self.on_checkpoint(checkpoint_path, self.state)
    
    async def _load_latest_checkpoint(self) -> bool:
        """Load the latest checkpoint if it exists."""
        import os
        
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        
        if not os.path.exists(latest_path):
            return False
        
        try:
            checkpoint = torch.load(latest_path, map_location="cpu")
            
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            state_data = checkpoint["training_state"]
            self.state.global_step = state_data["global_step"]
            self.state.epoch = state_data["epoch"]
            self.state.current_loss = state_data["current_loss"]
            self.state.best_loss = state_data["best_loss"]
            self.state.total_tokens_processed = state_data["total_tokens_processed"]
            self.state.total_compute_hours = state_data["total_compute_hours"]
            self.state.improvements = state_data.get("improvements", [])
            
            return True
            
        except Exception as e:
            logger.error("Failed to load checkpoint", error=str(e))
            return False
    
    # =========================================================================
    # Evaluation & Tracking
    # =========================================================================
    
    def _should_evaluate(self) -> bool:
        """Determine if we should run evaluation."""
        return self.state.global_step % self.eval_every_steps == 0
    
    async def _evaluate_and_track(self):
        """Evaluate model and track improvement metrics."""
        self.state.last_evaluation_at = datetime.utcnow()
        
        # Calculate smoothed loss (exponential moving average)
        if len(self.state.loss_history) >= 100:
            recent_losses = self.state.loss_history[-100:]
            smoothed_loss = sum(recent_losses) / len(recent_losses)
        else:
            smoothed_loss = self.state.current_loss
        
        # Calculate improvement rate
        if len(self.state.improvements) >= 2:
            recent_improvements = self.state.improvements[-10:]
            improvement_rate = len(recent_improvements) / 10  # Improvements per 10 evals
        else:
            improvement_rate = 0
        
        # Training efficiency
        if self.state.total_compute_hours > 0:
            tokens_per_hour = self.state.total_tokens_processed / self.state.total_compute_hours
        else:
            tokens_per_hour = 0
        
        eval_metrics = {
            "step": self.state.global_step,
            "smoothed_loss": smoothed_loss,
            "best_loss": self.state.best_loss,
            "improvement_rate": improvement_rate,
            "tokens_per_hour": tokens_per_hour,
            "total_improvements": len(self.state.improvements),
            "active_workers": self.state.active_workers,
            "total_compute_hours": self.state.total_compute_hours
        }
        
        logger.info(
            "Evaluation complete",
            **eval_metrics
        )
        
        return eval_metrics
    
    # =========================================================================
    # Dynamic Scaling
    # =========================================================================
    
    async def _adjust_training_dynamics(self):
        """
        Dynamically adjust training based on available compute.
        This is what makes it truly self-assembling.
        """
        # Adjust learning rate based on effective batch size
        effective_batch_size = self.target_batch_size * max(1, self.state.active_workers)
        
        # Linear scaling rule for learning rate
        base_lr = 1e-4
        scaled_lr = base_lr * (effective_batch_size / 256)
        scaled_lr = min(scaled_lr, 1e-3)  # Cap at 1e-3
        
        # Only adjust if significant change
        current_lr = self.optimizer.param_groups[0]['lr']
        if abs(scaled_lr - current_lr) / current_lr > 0.1:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = scaled_lr
            
            logger.info(
                "Learning rate adjusted for scale",
                old_lr=current_lr,
                new_lr=scaled_lr,
                workers=self.state.active_workers
            )
        
        # Adjust target batch size based on compute power
        if self.state.total_compute_power > 100:
            self.target_batch_size = min(1024, int(256 * (self.state.total_compute_power / 50)))
        else:
            self.target_batch_size = 256
    
    # =========================================================================
    # Status & Metrics
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            "phase": self.state.phase.value,
            "is_running": self.is_running,
            "global_step": self.state.global_step,
            "current_loss": self.state.current_loss,
            "best_loss": self.state.best_loss,
            "total_improvements": len(self.state.improvements),
            "active_workers": self.state.active_workers,
            "total_compute_power": self.state.total_compute_power,
            "total_compute_hours": round(self.state.total_compute_hours, 2),
            "total_tokens_processed": self.state.total_tokens_processed,
            "learning_rate": self.optimizer.param_groups[0]['lr'] if self.optimizer else 0,
            "target_batch_size": self.target_batch_size,
            "started_at": self.state.started_at.isoformat() if self.state.started_at else None,
            "uptime_hours": (
                (datetime.utcnow() - self.state.started_at).total_seconds() / 3600
                if self.state.started_at else 0
            )
        }
    
    def get_improvement_history(self, limit: int = 50) -> List[Dict]:
        """Get recent improvements."""
        return self.state.improvements[-limit:]
