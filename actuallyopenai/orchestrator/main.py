"""
ActuallyOpenAI Training Orchestrator
====================================
Main entry point for the distributed training orchestrator.
Coordinates workers, manages training jobs, and handles model aggregation.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional, Set
import os

# aioredis is optional - only needed for production
try:
    import aioredis
except ImportError:
    aioredis = None

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("orchestrator")


# =============================================================================
# Configuration
# =============================================================================

class OrchestratorConfig:
    """Orchestrator configuration."""
    
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Training settings
    MIN_WORKERS: int = int(os.getenv("MIN_WORKERS", "1"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    GRADIENT_ACCUMULATION_STEPS: int = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4"))
    
    # Checkpoint settings
    CHECKPOINT_DIR: str = os.getenv("CHECKPOINT_DIR", "./checkpoints")
    CHECKPOINT_INTERVAL: int = int(os.getenv("CHECKPOINT_INTERVAL", "1000"))  # steps
    
    # Worker settings
    WORKER_TIMEOUT: int = int(os.getenv("WORKER_TIMEOUT", "300"))  # seconds
    HEARTBEAT_INTERVAL: int = int(os.getenv("HEARTBEAT_INTERVAL", "30"))  # seconds


config = OrchestratorConfig()


# =============================================================================
# Data Models
# =============================================================================

class WorkerInfo(BaseModel):
    """Worker information."""
    worker_id: str
    gpu_type: Optional[str] = None
    gpu_memory: Optional[int] = None  # MB
    cpu_cores: Optional[int] = None
    status: str = "idle"  # idle, training, syncing
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    tasks_completed: int = 0
    total_compute_time: float = 0.0  # seconds
    reputation_score: float = 1.0


class TrainingTask(BaseModel):
    """Training task for workers."""
    task_id: str
    batch_indices: List[int]
    model_version: str
    learning_rate: float = 1e-4
    created_at: datetime = Field(default_factory=datetime.utcnow)
    assigned_worker: Optional[str] = None
    status: str = "pending"  # pending, assigned, completed, failed


class GradientSubmission(BaseModel):
    """Gradient submission from worker."""
    task_id: str
    worker_id: str
    gradients: Dict[str, List[float]]  # Simplified for JSON
    loss: float
    compute_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TrainingStatus(BaseModel):
    """Current training status."""
    is_training: bool
    current_step: int
    total_steps: int
    current_loss: float
    best_loss: float
    active_workers: int
    total_workers: int
    tokens_processed: int
    model_version: str


# =============================================================================
# Orchestrator State
# =============================================================================

class OrchestratorState:
    """Global orchestrator state."""
    
    def __init__(self):
        self.workers: Dict[str, WorkerInfo] = {}
        self.pending_tasks: Dict[str, TrainingTask] = {}
        self.completed_tasks: Dict[str, TrainingTask] = {}
        self.gradient_buffer: List[GradientSubmission] = []
        
        self.is_training: bool = False
        self.current_step: int = 0
        self.total_steps: int = 100000
        self.current_loss: float = 999.0  # Start with high but JSON-serializable value
        self.best_loss: float = 999.0  # Start with high but JSON-serializable value
        self.tokens_processed: int = 0
        self.model_version: str = "v0.1.0"
        
        self.shutdown_event = asyncio.Event()


state = OrchestratorState()


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="ActuallyOpenAI Orchestrator",
    description="Distributed training orchestrator for ActuallyOpenAI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_workers": len([w for w in state.workers.values() if w.status != "offline"]),
        "is_training": state.is_training
    }


@app.get("/status", response_model=TrainingStatus)
async def get_status():
    """Get current training status."""
    active = len([w for w in state.workers.values() if w.status != "offline"])
    return TrainingStatus(
        is_training=state.is_training,
        current_step=state.current_step,
        total_steps=state.total_steps,
        current_loss=state.current_loss,
        best_loss=state.best_loss,
        active_workers=active,
        total_workers=len(state.workers),
        tokens_processed=state.tokens_processed,
        model_version=state.model_version
    )


@app.post("/workers/register")
async def register_worker(worker: WorkerInfo):
    """Register a new worker."""
    worker.last_heartbeat = datetime.utcnow()
    state.workers[worker.worker_id] = worker
    logger.info(f"Worker registered: {worker.worker_id} ({worker.gpu_type or 'CPU'})")
    return {"status": "registered", "worker_id": worker.worker_id}


@app.post("/workers/{worker_id}/heartbeat")
async def worker_heartbeat(worker_id: str):
    """Worker heartbeat to indicate it's still alive."""
    if worker_id not in state.workers:
        raise HTTPException(status_code=404, detail="Worker not found")
    
    state.workers[worker_id].last_heartbeat = datetime.utcnow()
    return {"status": "ok"}


@app.get("/workers")
async def list_workers():
    """List all registered workers."""
    return {
        "workers": list(state.workers.values()),
        "total": len(state.workers),
        "active": len([w for w in state.workers.values() if w.status != "offline"])
    }


@app.post("/workers/{worker_id}/unregister")
async def unregister_worker(worker_id: str):
    """Unregister a worker."""
    if worker_id in state.workers:
        del state.workers[worker_id]
        logger.info(f"Worker unregistered: {worker_id}")
    return {"status": "unregistered"}


@app.get("/tasks/next/{worker_id}")
async def get_next_task(worker_id: str):
    """Get the next training task for a worker."""
    if worker_id not in state.workers:
        raise HTTPException(status_code=404, detail="Worker not registered")
    
    if not state.is_training:
        return {"task": None, "message": "Training not active"}
    
    # Find pending task
    for task_id, task in state.pending_tasks.items():
        if task.status == "pending":
            task.status = "assigned"
            task.assigned_worker = worker_id
            state.workers[worker_id].status = "training"
            return {"task": task}
    
    return {"task": None, "message": "No pending tasks"}


@app.post("/tasks/submit")
async def submit_gradients(submission: GradientSubmission, background_tasks: BackgroundTasks):
    """Submit computed gradients from a worker."""
    if submission.worker_id not in state.workers:
        raise HTTPException(status_code=404, detail="Worker not registered")
    
    if submission.task_id not in state.pending_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Update task status
    task = state.pending_tasks.pop(submission.task_id)
    task.status = "completed"
    state.completed_tasks[submission.task_id] = task
    
    # Update worker stats
    worker = state.workers[submission.worker_id]
    worker.status = "idle"
    worker.tasks_completed += 1
    worker.total_compute_time += submission.compute_time
    
    # Add to gradient buffer
    state.gradient_buffer.append(submission)
    
    # Update metrics
    state.current_loss = submission.loss
    if submission.loss < state.best_loss:
        state.best_loss = submission.loss
    
    # Trigger aggregation if enough gradients
    if len(state.gradient_buffer) >= config.GRADIENT_ACCUMULATION_STEPS:
        background_tasks.add_task(aggregate_gradients)
    
    logger.info(f"Gradients submitted: task={submission.task_id}, loss={submission.loss:.4f}")
    
    return {"status": "accepted", "current_loss": state.current_loss}


@app.post("/training/start")
async def start_training(background_tasks: BackgroundTasks):
    """Start distributed training."""
    if state.is_training:
        return {"status": "already_running"}
    
    active_workers = len([w for w in state.workers.values() if w.status != "offline"])
    if active_workers < config.MIN_WORKERS:
        raise HTTPException(
            status_code=400, 
            detail=f"Need at least {config.MIN_WORKERS} workers, have {active_workers}"
        )
    
    state.is_training = True
    background_tasks.add_task(training_loop)
    logger.info("Training started")
    
    return {"status": "started", "active_workers": active_workers}


@app.post("/training/stop")
async def stop_training():
    """Stop distributed training."""
    state.is_training = False
    logger.info("Training stopped")
    return {"status": "stopped", "final_step": state.current_step}


# =============================================================================
# Background Tasks
# =============================================================================

async def aggregate_gradients():
    """Aggregate gradients from workers and update model."""
    if not state.gradient_buffer:
        return
    
    logger.info(f"Aggregating {len(state.gradient_buffer)} gradient submissions")
    
    # Average the losses
    avg_loss = sum(g.loss for g in state.gradient_buffer) / len(state.gradient_buffer)
    state.current_loss = avg_loss
    
    # Clear buffer
    state.gradient_buffer.clear()
    
    # Update step counter
    state.current_step += 1
    
    # Checkpoint if needed
    if state.current_step % config.CHECKPOINT_INTERVAL == 0:
        await save_checkpoint()
    
    logger.info(f"Step {state.current_step}: avg_loss={avg_loss:.4f}")


async def save_checkpoint():
    """Save training checkpoint."""
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    checkpoint_path = os.path.join(
        config.CHECKPOINT_DIR, 
        f"checkpoint_step_{state.current_step}.json"
    )
    
    # In a real implementation, this would save model weights
    import json
    checkpoint_data = {
        "step": state.current_step,
        "loss": state.current_loss,
        "best_loss": state.best_loss,
        "model_version": state.model_version,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f)
    
    logger.info(f"Checkpoint saved: {checkpoint_path}")


async def training_loop():
    """Main training loop - generates tasks for workers."""
    import uuid
    
    logger.info("Training loop started")
    
    while state.is_training and not state.shutdown_event.is_set():
        # Check for active workers
        active_workers = [
            w for w in state.workers.values() 
            if w.status == "idle"
        ]
        
        if not active_workers:
            await asyncio.sleep(1)
            continue
        
        # Generate tasks for idle workers
        for _ in active_workers:
            task = TrainingTask(
                task_id=str(uuid.uuid4()),
                batch_indices=list(range(config.BATCH_SIZE)),
                model_version=state.model_version
            )
            state.pending_tasks[task.task_id] = task
        
        # Wait before generating more tasks
        await asyncio.sleep(5)
    
    logger.info("Training loop ended")


async def worker_monitor():
    """Monitor worker health and mark inactive workers."""
    while not state.shutdown_event.is_set():
        now = datetime.utcnow()
        
        for worker_id, worker in list(state.workers.items()):
            time_since_heartbeat = (now - worker.last_heartbeat).total_seconds()
            
            if time_since_heartbeat > config.WORKER_TIMEOUT:
                if worker.status != "offline":
                    logger.warning(f"Worker {worker_id} timed out")
                    worker.status = "offline"
        
        await asyncio.sleep(config.HEARTBEAT_INTERVAL)


# =============================================================================
# Application Lifecycle
# =============================================================================

@app.on_event("startup")
async def startup():
    """Application startup."""
    logger.info("🚀 ActuallyOpenAI Orchestrator starting...")
    
    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Start background tasks
    asyncio.create_task(worker_monitor())
    
    logger.info("✅ Orchestrator ready")


@app.on_event("shutdown")
async def shutdown():
    """Application shutdown."""
    logger.info("👋 Orchestrator shutting down...")
    state.shutdown_event.set()
    state.is_training = False


# =============================================================================
# Main Entry Point
# =============================================================================

def handle_signals():
    """Handle shutdown signals."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        state.shutdown_event.set()
        state.is_training = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point."""
    handle_signals()
    
    logger.info(f"Starting orchestrator on {config.HOST}:{config.PORT}")
    
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()
