"""
Training Orchestrator - Integrates all training components.

This is the main entry point that:
1. Manages the continuous training loop
2. Coordinates workers via WebSocket
3. Handles gradient aggregation
4. Tracks model evolution
5. Auto-scales based on compute
6. Reports improvements
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

import structlog
import torch
import torch.nn as nn
from fastapi import WebSocket

from actuallyopenai.config import get_settings
from actuallyopenai.training.continuous_trainer import (
    ContinuousTrainer,
    GradientPacket,
    TrainingPhase,
)
from actuallyopenai.training.federated_aggregator import (
    FederatedAggregator,
    AggregationStrategy,
    WorkerUpdate,
)
from actuallyopenai.training.model_evolution import (
    ModelEvolution,
    EvolutionStrategy,
)
from actuallyopenai.training.auto_scaler import (
    AutoScalingController,
    ScalingMode,
)
from actuallyopenai.training.improvement_tracker import (
    ImprovementTracker,
    BenchmarkSuite,
)
from actuallyopenai.core.models import Worker, WorkerStatus

logger = structlog.get_logger()


class TrainingOrchestrator:
    """
    Central orchestrator for the self-improving AI system.
    
    This class integrates all training components and provides:
    - Unified interface for managing distributed training
    - WebSocket-based communication with workers
    - Real-time status updates
    - Automatic model evolution
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_id: str = "aoai-v1",
        checkpoint_dir: str = "./checkpoints",
        evolution_dir: str = "./evolution",
        tracking_dir: str = "./tracking",
    ):
        self.model = model
        self.model_id = model_id
        
        # Initialize all training components
        self.trainer = ContinuousTrainer(
            model=model,
            model_id=model_id,
            checkpoint_dir=checkpoint_dir,
        )
        
        self.aggregator = FederatedAggregator(
            strategy=AggregationStrategy.FEDAVG,
            min_workers=2,
        )
        
        self.evolution = ModelEvolution(
            model_family=model_id,
            evolution_dir=evolution_dir,
            strategy=EvolutionStrategy.LINEAR,
        )
        
        self.scaler = AutoScalingController(
            mode=ScalingMode.BALANCED,
        )
        
        self.tracker = ImprovementTracker(
            tracking_dir=tracking_dir,
            benchmark_suite=BenchmarkSuite.standard_llm_suite(),
        )
        
        # Worker connections
        self.worker_connections: Dict[str, WebSocket] = {}
        self.workers: Dict[str, Worker] = {}
        
        # Training state
        self.is_running = False
        self.current_generation_id: Optional[str] = None
        
        # Event loop for background tasks
        self.training_task: Optional[asyncio.Task] = None
        
        # Setup callbacks
        self._setup_callbacks()
        
        logger.info(
            "TrainingOrchestrator initialized",
            model_id=model_id
        )
    
    def _setup_callbacks(self):
        """Setup callbacks between components."""
        # When trainer improves, notify evolution
        self.trainer.on_improvement = self._on_training_improvement
        
        # When we checkpoint, update evolution
        self.trainer.on_checkpoint = self._on_checkpoint
        
        # When a worker joins, notify scaler
        self.trainer.on_worker_joined = self._on_worker_joined
        
        # When we get a new best, celebrate
        self.tracker.on_new_best = self._on_new_benchmark_best
        
        # When evolution finds new best, save it
        self.evolution.on_new_best = self._on_new_best_model
    
    async def start(self):
        """Start the training orchestrator."""
        self.is_running = True
        
        # Load any saved state
        await self.evolution.load_evolution_state()
        await self.tracker.load_state()
        
        # Start new generation
        gen = await self.evolution.start_new_generation(self.model)
        self.current_generation_id = gen.id
        
        logger.info(
            "🚀 Starting training orchestrator",
            generation_id=gen.id,
            generation_number=gen.generation_number
        )
        
        # Start continuous training in background
        self.training_task = asyncio.create_task(self._training_loop())
    
    async def stop(self):
        """Stop the training orchestrator."""
        logger.info("Stopping training orchestrator...")
        self.is_running = False
        
        if self.training_task:
            self.training_task.cancel()
            try:
                await self.training_task
            except asyncio.CancelledError:
                pass
        
        # Final checkpoint and save state
        await self.trainer.stop()
        await self.evolution.save_evolution_state()
        await self.tracker.save_state()
        
        logger.info("Training orchestrator stopped")
    
    async def _training_loop(self):
        """Main training loop."""
        await self.trainer.initialize()
        
        while self.is_running:
            try:
                # Check if we have workers
                if len(self.workers) == 0:
                    logger.debug("Waiting for workers...")
                    await asyncio.sleep(5)
                    continue
                
                # Get current compute status
                worker_dict = {
                    w.id: {
                        "status": w.status.value,
                        "compute_score": w.compute_score,
                        "has_gpu": w.hardware_info.get("gpu_count", 0) > 0 if w.hardware_info else False,
                        "vram_gb": w.hardware_info.get("vram_total_gb", 0) if w.hardware_info else 0,
                    }
                    for w in self.workers.values()
                }
                
                # Auto-scale parameters
                throughput = self._estimate_throughput()
                scaling_decisions = await self.scaler.adjust_parameters(
                    workers=worker_dict,
                    current_throughput=throughput
                )
                
                # Apply scaling decisions to trainer
                if scaling_decisions:
                    self._apply_scaling_decisions(scaling_decisions)
                
                # Distribute tasks to workers
                await self._distribute_training_tasks()
                
                # Check for gradients to aggregate
                if self.aggregator.ready_to_aggregate():
                    result = await self.aggregator.aggregate()
                    
                    if result:
                        # Apply aggregated gradients to model
                        await self._apply_aggregated_gradients(result)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error("Error in training loop", error=str(e))
                await asyncio.sleep(5)
    
    async def _distribute_training_tasks(self):
        """Distribute training tasks to idle workers."""
        idle_workers = [
            w for w in self.workers.values()
            if w.status in (WorkerStatus.ONLINE, WorkerStatus.IDLE)
        ]
        
        if not idle_workers:
            return
        
        # Get current scaling parameters
        params = self.scaler.get_current_parameters()
        batch_per_worker = params["micro_batch_size"]
        lr = params["effective_learning_rate"]
        
        for worker in idle_workers:
            task = {
                "type": "training_task",
                "task_id": f"task_{worker.id}_{datetime.utcnow().timestamp()}",
                "model_id": self.model_id,
                "generation_id": self.current_generation_id,
                "batch_size": batch_per_worker,
                "learning_rate": lr,
                "mixed_precision": params["mixed_precision"],
                "gradient_accumulation": params["gradient_accumulation_steps"],
            }
            
            if worker.id in self.worker_connections:
                ws = self.worker_connections[worker.id]
                try:
                    await ws.send_json(task)
                    worker.status = WorkerStatus.TRAINING
                    logger.debug(
                        "Task distributed",
                        worker_id=worker.id,
                        task_id=task["task_id"]
                    )
                except Exception as e:
                    logger.error("Failed to send task", worker_id=worker.id, error=str(e))
    
    async def _apply_aggregated_gradients(self, result):
        """Apply aggregated gradients to the model."""
        if not result.aggregated_gradients:
            return
        
        # Zero gradients
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()
        
        # Apply aggregated gradients
        for name, param in self.model.named_parameters():
            if name in result.aggregated_gradients:
                param.grad = result.aggregated_gradients[name]
        
        # This would normally be handled by the trainer's optimizer
        # For now, notify trainer of the update
        self.trainer.state.global_step += 1
        self.trainer.state.current_loss = result.weighted_loss
        
        # Update evolution tracking
        if self.current_generation_id:
            await self.evolution.update_generation_metrics(
                generation_id=self.current_generation_id,
                loss=result.weighted_loss,
                steps=self.trainer.state.global_step,
                tokens=self.trainer.state.total_tokens_processed,
                compute_hours=self.trainer.state.total_compute_hours
            )
        
        logger.info(
            "Gradients applied",
            step=self.trainer.state.global_step,
            loss=round(result.weighted_loss, 6),
            workers=result.num_workers
        )
    
    def _apply_scaling_decisions(self, decisions):
        """Apply scaling decisions to trainer."""
        for decision in decisions:
            if decision.parameter_changed == "global_batch_size":
                self.trainer.target_batch_size = decision.new_value
            elif decision.parameter_changed == "effective_learning_rate":
                if self.trainer.optimizer:
                    for param_group in self.trainer.optimizer.param_groups:
                        param_group['lr'] = decision.new_value
    
    def _estimate_throughput(self) -> float:
        """Estimate current training throughput."""
        if self.trainer.state.total_compute_hours > 0:
            return self.trainer.state.total_tokens_processed / self.trainer.state.total_compute_hours
        return 0.0
    
    # =========================================================================
    # Worker Management
    # =========================================================================
    
    async def register_worker(self, worker: Worker, websocket: WebSocket):
        """Register a new worker connection."""
        self.workers[worker.id] = worker
        self.worker_connections[worker.id] = websocket
        
        await self.trainer.register_worker(worker)
        await self.scaler.on_worker_joined(worker.id, {
            "status": worker.status.value,
            "compute_score": worker.compute_score,
        })
        
        # Notify worker of current state
        await websocket.send_json({
            "type": "registration_confirmed",
            "worker_id": worker.id,
            "generation_id": self.current_generation_id,
            "model_id": self.model_id,
            "training_params": self.scaler.get_current_parameters()
        })
        
        logger.info(
            "Worker registered",
            worker_id=worker.id,
            total_workers=len(self.workers)
        )
    
    async def unregister_worker(self, worker_id: str):
        """Unregister a worker."""
        if worker_id in self.workers:
            del self.workers[worker_id]
        if worker_id in self.worker_connections:
            del self.worker_connections[worker_id]
        
        await self.trainer.remove_worker(worker_id)
        await self.scaler.on_worker_left(worker_id)
        
        logger.info(
            "Worker unregistered",
            worker_id=worker_id,
            remaining_workers=len(self.workers)
        )
    
    async def receive_gradients(
        self,
        worker_id: str,
        gradients: Dict[str, torch.Tensor],
        loss: float,
        batch_size: int,
        compute_time: float
    ):
        """Receive gradients from a worker."""
        # Create gradient packet for trainer
        packet = GradientPacket(
            worker_id=worker_id,
            task_id=f"task_{worker_id}",
            gradients=gradients,
            loss=loss,
            batch_size=batch_size,
            compute_time=compute_time
        )
        await self.trainer.receive_gradients(packet)
        
        # Also send to aggregator
        update = WorkerUpdate(
            worker_id=worker_id,
            round_id=self.aggregator.current_round,
            gradients=gradients,
            num_samples=batch_size,
            local_loss=loss,
            compute_time=compute_time
        )
        await self.aggregator.receive_update(update)
        
        # Update worker status
        if worker_id in self.workers:
            self.workers[worker_id].status = WorkerStatus.IDLE
    
    # =========================================================================
    # Callbacks
    # =========================================================================
    
    async def _on_training_improvement(self, state):
        """Called when training shows improvement."""
        logger.info(
            "🎉 Training improvement detected!",
            step=state.global_step,
            loss=state.current_loss,
            best_loss=state.best_loss
        )
        
        # Run evaluation if significant improvement
        if len(state.improvements) % 5 == 0:
            await self._run_evaluation()
    
    async def _on_checkpoint(self, checkpoint_path, state):
        """Called when a checkpoint is saved."""
        logger.info(
            "💾 Checkpoint saved",
            path=checkpoint_path,
            step=state.global_step
        )
    
    async def _on_worker_joined(self, worker):
        """Called when a worker joins."""
        # Broadcast to all workers
        message = {
            "type": "worker_update",
            "total_workers": len(self.workers),
            "total_compute": sum(w.compute_score for w in self.workers.values())
        }
        await self._broadcast(message)
    
    async def _on_new_benchmark_best(self, benchmark_id, result):
        """Called when a new benchmark best is achieved."""
        logger.info(
            f"🏆 New best on {benchmark_id}!",
            score=result.score
        )
        
        # Broadcast achievement
        await self._broadcast({
            "type": "achievement",
            "benchmark": benchmark_id,
            "score": result.score,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def _on_new_best_model(self, generation):
        """Called when evolution finds a new best model."""
        logger.info(
            "🚀 New best model generation!",
            generation_id=generation.id,
            loss=generation.loss
        )
    
    async def _run_evaluation(self):
        """Run model evaluation."""
        if self.current_generation_id:
            report = await self.tracker.evaluate_generation(
                generation_id=self.current_generation_id,
                model=self.model,
                compute_hours=self.trainer.state.total_compute_hours,
                tokens_trained=self.trainer.state.total_tokens_processed
            )
            
            logger.info(
                "Evaluation complete",
                overall_score=round(report.overall_score, 4),
                rank=report.overall_rank
            )
    
    async def _broadcast(self, message: Dict):
        """Broadcast message to all connected workers."""
        for worker_id, ws in self.worker_connections.items():
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.debug(f"Failed to broadcast to worker {worker_id}: {e}")
    
    # =========================================================================
    # Status & Metrics
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the training system."""
        return {
            "is_running": self.is_running,
            "model_id": self.model_id,
            "current_generation_id": self.current_generation_id,
            "training": self.trainer.get_status(),
            "aggregation": self.aggregator.get_statistics(),
            "evolution": self.evolution.get_evolution_summary(),
            "scaling": self.scaler.get_status(),
            "improvement": self.tracker.get_improvement_summary(),
            "workers": {
                "total": len(self.workers),
                "online": len([w for w in self.workers.values() if w.status != WorkerStatus.OFFLINE]),
                "training": len([w for w in self.workers.values() if w.status == WorkerStatus.TRAINING])
            }
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display."""
        trainer_status = self.trainer.get_status()
        improvement = self.tracker.get_improvement_summary()
        
        return {
            "status": "running" if self.is_running else "stopped",
            "model": self.model_id,
            "generation": self.current_generation_id,
            
            # Key metrics
            "metrics": {
                "global_step": trainer_status["global_step"],
                "current_loss": round(trainer_status["current_loss"], 6),
                "best_loss": round(trainer_status["best_loss"], 6),
                "overall_score": round(improvement.get("best_scores", {}).get("perplexity", 0), 4),
            },
            
            # Resources
            "resources": {
                "workers": len(self.workers),
                "total_compute_power": round(trainer_status["total_compute_power"], 2),
                "total_compute_hours": round(trainer_status["total_compute_hours"], 2),
                "tokens_processed": trainer_status["total_tokens_processed"],
            },
            
            # Improvement
            "improvement": {
                "total_improvements": trainer_status["total_improvements"],
                "improvement_rate": improvement.get("improvement_per_compute_hour", 0),
                "leaderboard_rank": improvement.get("current_rank", 0),
            },
            
            # Training params
            "training_params": self.scaler.get_current_parameters(),
        }
