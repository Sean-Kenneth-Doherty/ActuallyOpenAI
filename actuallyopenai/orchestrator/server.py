"""
Orchestrator Server - Central coordination hub for ActuallyOpenAI distributed training.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
import structlog

from actuallyopenai.config import get_settings, Settings
from actuallyopenai.core.models import (
    Worker, WorkerRegistration, WorkerStatus,
    TrainingTask, TrainingJob, TaskStatus, TaskType,
    ComputeContribution, HardwareSpec
)
from actuallyopenai.core.database import (
    create_database_engine, create_tables, create_session_factory
)

logger = structlog.get_logger()


# =============================================================================
# Application State
# =============================================================================

class OrchestratorState:
    """Global state for the orchestrator."""
    
    def __init__(self):
        self.workers: Dict[str, Worker] = {}
        self.active_connections: Dict[str, WebSocket] = {}
        self.pending_tasks: List[TrainingTask] = []
        self.running_tasks: Dict[str, TrainingTask] = {}
        self.completed_tasks: List[TrainingTask] = []
        self.training_jobs: Dict[str, TrainingJob] = {}
        
    def get_available_workers(self) -> List[Worker]:
        """Get workers that are available for tasks."""
        return [
            w for w in self.workers.values()
            if w.status in (WorkerStatus.ONLINE, WorkerStatus.IDLE)
        ]
    
    def get_worker_by_wallet(self, wallet: str) -> Optional[Worker]:
        """Find worker by wallet address."""
        for worker in self.workers.values():
            if worker.wallet_address.lower() == wallet.lower():
                return worker
        return None


state = OrchestratorState()


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()
    
    # Initialize database
    logger.info("Initializing database connection...")
    engine = await create_database_engine(settings.database_url)
    await create_tables(engine)
    app.state.db_engine = engine
    app.state.session_factory = create_session_factory(engine)
    
    # Start background tasks
    logger.info("Starting background task scheduler...")
    task_scheduler = asyncio.create_task(task_distribution_loop())
    heartbeat_checker = asyncio.create_task(heartbeat_check_loop())
    
    logger.info("Orchestrator server started", port=settings.orchestrator_port)
    
    yield
    
    # Cleanup
    logger.info("Shutting down orchestrator...")
    task_scheduler.cancel()
    heartbeat_checker.cancel()
    await engine.dispose()


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="ActuallyOpenAI Orchestrator",
    description="Central coordination hub for distributed AI training",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Worker Management Endpoints
# =============================================================================

@app.post("/api/v1/workers/register", response_model=Worker)
async def register_worker(registration: WorkerRegistration):
    """
    Register a new worker node to contribute compute.
    Returns the worker with assigned ID.
    """
    # Check if wallet already registered
    existing = state.get_worker_by_wallet(registration.wallet_address)
    if existing:
        logger.info("Worker re-registering", wallet=registration.wallet_address)
        existing.status = WorkerStatus.ONLINE
        existing.hardware = registration.hardware
        existing.last_heartbeat = datetime.utcnow()
        return existing
    
    # Calculate compute score based on hardware
    compute_score = calculate_compute_score(registration.hardware)
    
    worker = Worker(
        wallet_address=registration.wallet_address,
        name=registration.name,
        region=registration.region,
        hardware=registration.hardware,
        compute_score=compute_score,
        status=WorkerStatus.ONLINE,
        last_heartbeat=datetime.utcnow(),
        connected_at=datetime.utcnow()
    )
    
    state.workers[worker.id] = worker
    
    logger.info(
        "Worker registered",
        worker_id=worker.id,
        wallet=registration.wallet_address,
        compute_score=compute_score
    )
    
    return worker


@app.get("/api/v1/workers", response_model=List[Worker])
async def list_workers():
    """List all registered workers."""
    return list(state.workers.values())


@app.get("/api/v1/workers/{worker_id}", response_model=Worker)
async def get_worker(worker_id: str):
    """Get details of a specific worker."""
    if worker_id not in state.workers:
        raise HTTPException(status_code=404, detail="Worker not found")
    return state.workers[worker_id]


@app.post("/api/v1/workers/{worker_id}/heartbeat")
async def worker_heartbeat(worker_id: str):
    """
    Heartbeat endpoint for workers to signal they're alive.
    Should be called every 30 seconds.
    """
    if worker_id not in state.workers:
        raise HTTPException(status_code=404, detail="Worker not found")
    
    worker = state.workers[worker_id]
    worker.last_heartbeat = datetime.utcnow()
    
    if worker.status == WorkerStatus.OFFLINE:
        worker.status = WorkerStatus.IDLE
    
    return {"status": "ok", "server_time": datetime.utcnow().isoformat()}


# =============================================================================
# Training Job Endpoints
# =============================================================================

@app.post("/api/v1/jobs", response_model=TrainingJob)
async def create_training_job(
    name: str,
    model_id: str,
    total_epochs: int = 10,
    total_steps: int = 10000,
    token_reward: float = 1000.0,
    description: str = ""
):
    """Create a new distributed training job."""
    from decimal import Decimal
    
    job = TrainingJob(
        name=name,
        description=description,
        model_id=model_id,
        total_epochs=total_epochs,
        total_steps=total_steps,
        total_token_reward=Decimal(str(token_reward))
    )
    
    state.training_jobs[job.id] = job
    
    logger.info(
        "Training job created",
        job_id=job.id,
        name=name,
        token_reward=token_reward
    )
    
    return job


@app.get("/api/v1/jobs", response_model=List[TrainingJob])
async def list_jobs():
    """List all training jobs."""
    return list(state.training_jobs.values())


@app.get("/api/v1/jobs/{job_id}", response_model=TrainingJob)
async def get_job(job_id: str):
    """Get details of a training job."""
    if job_id not in state.training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return state.training_jobs[job_id]


@app.post("/api/v1/jobs/{job_id}/start")
async def start_job(job_id: str):
    """Start a training job - begins distributing tasks to workers."""
    if job_id not in state.training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = state.training_jobs[job_id]
    
    if job.status != TaskStatus.PENDING:
        raise HTTPException(status_code=400, detail="Job already started")
    
    job.status = TaskStatus.RUNNING
    job.started_at = datetime.utcnow()
    
    # Generate initial batch of tasks
    tasks = generate_tasks_for_job(job, batch_count=10)
    state.pending_tasks.extend(tasks)
    
    logger.info("Training job started", job_id=job_id, initial_tasks=len(tasks))
    
    return {"status": "started", "tasks_queued": len(tasks)}


# =============================================================================
# Task Management Endpoints
# =============================================================================

@app.get("/api/v1/tasks/pending", response_model=List[TrainingTask])
async def get_pending_tasks():
    """Get all pending tasks waiting for assignment."""
    return state.pending_tasks


@app.post("/api/v1/tasks/{task_id}/complete")
async def complete_task(
    task_id: str,
    loss: float,
    compute_time_seconds: float,
    result_url: str,
    metrics: dict = None
):
    """
    Mark a task as completed and record the contribution.
    Called by worker nodes when they finish a task.
    """
    if task_id not in state.running_tasks:
        raise HTTPException(status_code=404, detail="Task not found in running tasks")
    
    task = state.running_tasks[task_id]
    task.status = TaskStatus.COMPLETED
    task.completed_at = datetime.utcnow()
    task.loss = loss
    task.compute_time_seconds = compute_time_seconds
    task.result_url = result_url
    task.metrics = metrics or {}
    
    # Calculate GPU hours
    if task.started_at:
        duration_hours = (task.completed_at - task.started_at).total_seconds() / 3600
        worker = state.workers.get(task.assigned_worker_id)
        if worker and worker.hardware.gpu_count > 0:
            task.gpu_hours = duration_hours * worker.hardware.gpu_count
    
    # Move to completed
    del state.running_tasks[task_id]
    state.completed_tasks.append(task)
    
    # Update worker status
    if task.assigned_worker_id in state.workers:
        worker = state.workers[task.assigned_worker_id]
        worker.status = WorkerStatus.IDLE
        worker.total_tasks_completed += 1
        worker.total_compute_hours += compute_time_seconds / 3600
    
    # Record contribution (tokens will be calculated by contribution tracker)
    contribution = await record_contribution(task)
    
    logger.info(
        "Task completed",
        task_id=task_id,
        loss=loss,
        compute_time=compute_time_seconds,
        tokens_earned=str(contribution.total_tokens)
    )
    
    return {
        "status": "completed",
        "tokens_earned": str(contribution.total_tokens)
    }


# =============================================================================
# WebSocket for Real-time Communication
# =============================================================================

@app.websocket("/ws/worker/{worker_id}")
async def worker_websocket(websocket: WebSocket, worker_id: str):
    """
    WebSocket connection for real-time task distribution and status updates.
    """
    await websocket.accept()
    
    if worker_id not in state.workers:
        await websocket.close(code=4004, reason="Worker not registered")
        return
    
    state.active_connections[worker_id] = websocket
    worker = state.workers[worker_id]
    worker.status = WorkerStatus.IDLE
    
    logger.info("Worker WebSocket connected", worker_id=worker_id)
    
    try:
        while True:
            # Receive messages from worker
            data = await websocket.receive_json()
            
            if data.get("type") == "heartbeat":
                worker.last_heartbeat = datetime.utcnow()
                await websocket.send_json({"type": "heartbeat_ack"})
                
            elif data.get("type") == "task_progress":
                # Handle progress updates
                task_id = data.get("task_id")
                progress = data.get("progress", 0)
                if task_id in state.running_tasks:
                    state.running_tasks[task_id].metrics["progress"] = progress
                    
            elif data.get("type") == "task_complete":
                # Handle task completion via WebSocket
                await handle_websocket_task_complete(data, worker_id)
                
    except WebSocketDisconnect:
        logger.info("Worker WebSocket disconnected", worker_id=worker_id)
        del state.active_connections[worker_id]
        worker.status = WorkerStatus.OFFLINE


# =============================================================================
# Network Statistics Endpoints
# =============================================================================

@app.get("/api/v1/stats")
async def get_network_stats():
    """Get overall network statistics."""
    total_workers = len(state.workers)
    online_workers = len([w for w in state.workers.values() if w.status != WorkerStatus.OFFLINE])
    total_compute_score = sum(w.compute_score for w in state.workers.values())
    
    total_tasks_completed = sum(w.total_tasks_completed for w in state.workers.values())
    total_compute_hours = sum(w.total_compute_hours for w in state.workers.values())
    total_tokens_distributed = sum(w.total_tokens_earned for w in state.workers.values())
    
    active_jobs = len([j for j in state.training_jobs.values() if j.status == TaskStatus.RUNNING])
    
    return {
        "network": {
            "total_workers": total_workers,
            "online_workers": online_workers,
            "total_compute_score": total_compute_score
        },
        "training": {
            "active_jobs": active_jobs,
            "pending_tasks": len(state.pending_tasks),
            "running_tasks": len(state.running_tasks),
            "completed_tasks": len(state.completed_tasks)
        },
        "rewards": {
            "total_tasks_completed": total_tasks_completed,
            "total_compute_hours": round(total_compute_hours, 2),
            "total_tokens_distributed": str(total_tokens_distributed)
        }
    }


@app.get("/api/v1/stats/leaderboard")
async def get_leaderboard(limit: int = 10):
    """Get top contributors by tokens earned."""
    sorted_workers = sorted(
        state.workers.values(),
        key=lambda w: w.total_tokens_earned,
        reverse=True
    )[:limit]
    
    return [
        {
            "rank": i + 1,
            "wallet": w.wallet_address,
            "name": w.name,
            "tokens_earned": str(w.total_tokens_earned),
            "tasks_completed": w.total_tasks_completed,
            "compute_hours": round(w.total_compute_hours, 2)
        }
        for i, w in enumerate(sorted_workers)
    ]


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_compute_score(hardware: HardwareSpec) -> float:
    """
    Calculate a normalized compute score based on hardware.
    Used for fair task distribution and reward calculation.
    """
    score = 0.0
    
    # CPU contribution
    score += hardware.cpu_cores * 0.5
    
    # RAM contribution
    score += hardware.ram_gb * 0.1
    
    # GPU contribution (major factor)
    if hardware.gpu_count > 0 and hardware.gpu_memory_gb:
        # Approximate TFLOPS based on GPU memory (rough heuristic)
        gpu_power = hardware.gpu_memory_gb * 2  # Rough estimate
        score += gpu_power * hardware.gpu_count * 10
    
    # Bandwidth factor
    score *= (1 + hardware.bandwidth_mbps / 1000 * 0.1)
    
    return round(score, 2)


def generate_tasks_for_job(job: TrainingJob, batch_count: int = 10) -> List[TrainingTask]:
    """Generate training tasks for a job."""
    tasks = []
    
    for i in range(batch_count):
        task = TrainingTask(
            job_id=job.id,
            task_type=TaskType.GRADIENT_COMPUTE,
            model_id=job.model_id,
            data_shard_url=f"ipfs://training-data/{job.id}/shard_{job.current_step + i}",
            batch_indices=list(range(i * 32, (i + 1) * 32)),
            learning_rate=1e-4,
            batch_size=32
        )
        tasks.append(task)
    
    return tasks


async def record_contribution(task: TrainingTask) -> ComputeContribution:
    """Record a compute contribution and calculate token reward."""
    from decimal import Decimal
    
    worker = state.workers.get(task.assigned_worker_id)
    if not worker:
        raise ValueError("Worker not found")
    
    # Base token calculation
    # Tokens = compute_time * gpu_multiplier * quality_score
    base_rate = Decimal("0.1")  # Base tokens per second of compute
    gpu_multiplier = Decimal(str(1 + (worker.hardware.gpu_count * 0.5)))
    
    base_tokens = Decimal(str(task.compute_time_seconds)) * base_rate * gpu_multiplier
    
    # Bonus for fast completion
    bonus_tokens = Decimal("0")
    if task.compute_time_seconds < 60:  # Fast task
        bonus_tokens = base_tokens * Decimal("0.1")
    
    total_tokens = base_tokens + bonus_tokens
    
    contribution = ComputeContribution(
        worker_id=task.assigned_worker_id,
        wallet_address=worker.wallet_address,
        task_id=task.id,
        job_id=task.job_id,
        compute_time_seconds=task.compute_time_seconds,
        gpu_hours=task.gpu_hours,
        base_tokens=base_tokens,
        bonus_tokens=bonus_tokens,
        total_tokens=total_tokens,
        task_success=True,
        verified=True  # Would be verified by gradient validation in production
    )
    
    # Update worker's total earnings
    worker.total_tokens_earned += total_tokens
    
    return contribution


async def handle_websocket_task_complete(data: dict, worker_id: str):
    """Handle task completion received via WebSocket."""
    task_id = data.get("task_id")
    if task_id and task_id in state.running_tasks:
        await complete_task(
            task_id=task_id,
            loss=data.get("loss", 0.0),
            compute_time_seconds=data.get("compute_time", 0.0),
            result_url=data.get("result_url", ""),
            metrics=data.get("metrics", {})
        )


# =============================================================================
# Background Tasks
# =============================================================================

async def task_distribution_loop():
    """Background loop to distribute tasks to available workers."""
    while True:
        try:
            available_workers = state.get_available_workers()
            
            while state.pending_tasks and available_workers:
                task = state.pending_tasks.pop(0)
                
                # Select best worker (highest compute score with good reliability)
                worker = max(
                    available_workers,
                    key=lambda w: w.compute_score * w.reliability_score
                )
                
                # Assign task
                task.status = TaskStatus.ASSIGNED
                task.assigned_worker_id = worker.id
                task.assigned_at = datetime.utcnow()
                
                state.running_tasks[task.id] = task
                worker.status = WorkerStatus.TRAINING
                
                # Send task to worker via WebSocket if connected
                if worker.id in state.active_connections:
                    ws = state.active_connections[worker.id]
                    await ws.send_json({
                        "type": "task_assigned",
                        "task": task.model_dump()
                    })
                
                available_workers.remove(worker)
                
                logger.debug(
                    "Task assigned",
                    task_id=task.id,
                    worker_id=worker.id
                )
            
        except Exception as e:
            logger.error("Error in task distribution", error=str(e))
        
        await asyncio.sleep(1)  # Check every second


async def heartbeat_check_loop():
    """Background loop to check worker heartbeats and mark offline."""
    while True:
        try:
            cutoff = datetime.utcnow() - timedelta(seconds=60)
            
            for worker in state.workers.values():
                if worker.last_heartbeat and worker.last_heartbeat < cutoff:
                    if worker.status != WorkerStatus.OFFLINE:
                        logger.warning(
                            "Worker went offline (missed heartbeat)",
                            worker_id=worker.id
                        )
                        worker.status = WorkerStatus.OFFLINE
                        
                        # Reassign any running tasks
                        for task_id, task in list(state.running_tasks.items()):
                            if task.assigned_worker_id == worker.id:
                                task.status = TaskStatus.PENDING
                                task.assigned_worker_id = None
                                state.pending_tasks.append(task)
                                del state.running_tasks[task_id]
                        
        except Exception as e:
            logger.error("Error in heartbeat check", error=str(e))
        
        await asyncio.sleep(10)  # Check every 10 seconds


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Run the orchestrator server."""
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "actuallyopenai.orchestrator.server:app",
        host=settings.orchestrator_host,
        port=settings.orchestrator_port,
        workers=settings.orchestrator_workers if not settings.debug else 1,
        reload=settings.debug
    )


if __name__ == "__main__":
    main()
