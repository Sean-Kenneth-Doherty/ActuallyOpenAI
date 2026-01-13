"""
Worker Node - Compute contribution client for ActuallyOpenAI distributed training.
Connects to the orchestrator and contributes GPU/CPU cycles.
"""

import asyncio
import hashlib
import platform
import signal
import sys
import time
from datetime import datetime
from typing import Optional, Dict, Any
import uuid

import httpx
import torch
import websockets
from websockets.client import WebSocketClientProtocol
import structlog
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel

from actuallyopenai.config import get_settings, Settings
from actuallyopenai.core.models import (
    Worker, WorkerRegistration, WorkerStatus, 
    TrainingTask, TaskStatus, HardwareSpec
)

logger = structlog.get_logger()
console = Console()


class WorkerNode:
    """
    Worker node that connects to the orchestrator and contributes compute.
    Earns AOAI tokens for completed training tasks.
    """
    
    def __init__(
        self,
        orchestrator_url: str,
        wallet_address: str,
        name: str = None,
        region: str = "unknown",
        use_gpu: bool = True
    ):
        self.orchestrator_url = orchestrator_url.rstrip("/")
        self.wallet_address = wallet_address
        self.name = name or f"worker-{uuid.uuid4().hex[:8]}"
        self.region = region
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # State
        self.worker_id: Optional[str] = None
        self.is_running = False
        self.current_task: Optional[TrainingTask] = None
        self.websocket: Optional[WebSocketClientProtocol] = None
        
        # Stats
        self.tasks_completed = 0
        self.tokens_earned = 0.0
        self.total_compute_time = 0.0
        self.start_time: Optional[datetime] = None
        
        # HTTP client
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Device
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
    
    def detect_hardware(self) -> HardwareSpec:
        """Detect local hardware specifications."""
        import psutil
        
        cpu_cores = psutil.cpu_count(logical=False) or 1
        cpu_model = platform.processor() or "Unknown CPU"
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        
        gpu_count = 0
        gpu_model = None
        gpu_memory_gb = None
        cuda_version = None
        
        if self.use_gpu and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_model = torch.cuda.get_device_name(0)
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            cuda_version = torch.version.cuda
        
        return HardwareSpec(
            cpu_cores=cpu_cores,
            cpu_model=cpu_model,
            ram_gb=round(ram_gb, 2),
            gpu_count=gpu_count,
            gpu_model=gpu_model,
            gpu_memory_gb=round(gpu_memory_gb, 2) if gpu_memory_gb else None,
            cuda_version=cuda_version,
            bandwidth_mbps=100  # Would need actual measurement
        )
    
    async def register(self) -> bool:
        """Register this worker with the orchestrator."""
        try:
            hardware = self.detect_hardware()
            
            registration = WorkerRegistration(
                wallet_address=self.wallet_address,
                name=self.name,
                region=self.region,
                hardware=hardware
            )
            
            response = await self.http_client.post(
                f"{self.orchestrator_url}/api/v1/workers/register",
                json=registration.model_dump()
            )
            response.raise_for_status()
            
            worker_data = response.json()
            self.worker_id = worker_data["id"]
            
            logger.info(
                "Worker registered successfully",
                worker_id=self.worker_id,
                wallet=self.wallet_address,
                compute_score=worker_data.get("compute_score", 0)
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to register worker", error=str(e))
            return False
    
    async def connect_websocket(self):
        """Establish WebSocket connection with orchestrator."""
        ws_url = self.orchestrator_url.replace("http", "ws")
        ws_url = f"{ws_url}/ws/worker/{self.worker_id}"
        
        try:
            self.websocket = await websockets.connect(ws_url)
            logger.info("WebSocket connected", url=ws_url)
            return True
        except Exception as e:
            logger.error("WebSocket connection failed", error=str(e))
            return False
    
    async def send_heartbeat(self):
        """Send heartbeat to orchestrator."""
        try:
            if self.websocket:
                await self.websocket.send('{"type": "heartbeat"}')
            else:
                await self.http_client.post(
                    f"{self.orchestrator_url}/api/v1/workers/{self.worker_id}/heartbeat"
                )
        except Exception as e:
            logger.warning("Heartbeat failed", error=str(e))
    
    async def process_task(self, task_data: dict) -> Dict[str, Any]:
        """
        Process a training task and return results.
        This is where the actual ML computation happens.
        """
        task = TrainingTask(**task_data)
        self.current_task = task
        
        logger.info(
            "Processing task",
            task_id=task.id,
            task_type=task.task_type.value
        )
        
        start_time = time.time()
        
        try:
            # Simulate training computation
            # In production, this would:
            # 1. Download model shard from IPFS
            # 2. Download data shard from IPFS
            # 3. Perform forward/backward pass
            # 4. Upload gradients to IPFS
            
            result = await self._execute_training_step(task)
            
            compute_time = time.time() - start_time
            
            return {
                "success": True,
                "task_id": task.id,
                "loss": result["loss"],
                "compute_time": compute_time,
                "result_url": result["gradient_url"],
                "metrics": result.get("metrics", {})
            }
            
        except Exception as e:
            logger.error("Task execution failed", task_id=task.id, error=str(e))
            return {
                "success": False,
                "task_id": task.id,
                "error": str(e),
                "compute_time": time.time() - start_time
            }
        finally:
            self.current_task = None
    
    async def _execute_training_step(self, task: TrainingTask) -> Dict[str, Any]:
        """
        Execute the actual training computation.
        This is a simplified simulation - real implementation would use actual models.
        """
        # Simulate computation based on batch size
        computation_iterations = task.batch_size * task.gradient_accumulation
        
        # Create dummy tensor operations to simulate GPU work
        if self.use_gpu:
            # GPU computation simulation
            tensor_size = 1024
            for i in range(computation_iterations):
                x = torch.randn(tensor_size, tensor_size, device=self.device)
                y = torch.randn(tensor_size, tensor_size, device=self.device)
                z = torch.matmul(x, y)
                loss = z.mean()
                loss.backward() if z.requires_grad else None
                
                # Small delay to simulate realistic training
                await asyncio.sleep(0.01)
                
                # Report progress
                if self.websocket and i % 10 == 0:
                    progress = (i + 1) / computation_iterations * 100
                    await self.websocket.send(f'{{"type": "task_progress", "task_id": "{task.id}", "progress": {progress}}}')
        else:
            # CPU computation simulation
            await asyncio.sleep(computation_iterations * 0.05)
        
        # Generate mock gradient hash (would be actual gradients in production)
        gradient_hash = hashlib.sha256(
            f"{task.id}-{time.time()}".encode()
        ).hexdigest()
        
        # Simulate uploading to IPFS
        gradient_url = f"ipfs://gradients/{gradient_hash}"
        
        return {
            "loss": 0.5 + (hash(task.id) % 100) / 1000,  # Simulated loss
            "gradient_url": gradient_url,
            "metrics": {
                "batch_size": task.batch_size,
                "iterations": computation_iterations
            }
        }
    
    async def report_task_completion(self, result: Dict[str, Any]):
        """Report task completion to orchestrator."""
        try:
            if result["success"]:
                if self.websocket:
                    await self.websocket.send(
                        f'{{"type": "task_complete", "task_id": "{result["task_id"]}", '
                        f'"loss": {result["loss"]}, "compute_time": {result["compute_time"]}, '
                        f'"result_url": "{result["result_url"]}"}}'
                    )
                else:
                    response = await self.http_client.post(
                        f"{self.orchestrator_url}/api/v1/tasks/{result['task_id']}/complete",
                        params={
                            "loss": result["loss"],
                            "compute_time_seconds": result["compute_time"],
                            "result_url": result["result_url"]
                        },
                        json=result.get("metrics", {})
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        tokens = float(data.get("tokens_earned", 0))
                        self.tokens_earned += tokens
                        
                self.tasks_completed += 1
                self.total_compute_time += result["compute_time"]
                
                logger.info(
                    "Task completed and reported",
                    task_id=result["task_id"],
                    loss=result["loss"],
                    compute_time=round(result["compute_time"], 2)
                )
            else:
                logger.error(
                    "Task failed",
                    task_id=result["task_id"],
                    error=result.get("error")
                )
                
        except Exception as e:
            logger.error("Failed to report task completion", error=str(e))
    
    async def listen_for_tasks(self):
        """Listen for incoming tasks via WebSocket."""
        while self.is_running and self.websocket:
            try:
                message = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=5.0
                )
                
                import json
                data = json.loads(message)
                
                if data.get("type") == "task_assigned":
                    task_data = data.get("task")
                    result = await self.process_task(task_data)
                    await self.report_task_completion(result)
                    
                elif data.get("type") == "heartbeat_ack":
                    pass  # Heartbeat acknowledged
                    
            except asyncio.TimeoutError:
                # No message received, send heartbeat
                await self.send_heartbeat()
            except websockets.ConnectionClosed:
                logger.warning("WebSocket connection closed, attempting reconnect...")
                if self.is_running:
                    await asyncio.sleep(5)
                    await self.connect_websocket()
            except Exception as e:
                logger.error("Error in task listener", error=str(e))
                await asyncio.sleep(1)
    
    def get_status_display(self) -> Panel:
        """Generate rich status display for terminal."""
        table = Table(show_header=False, box=None)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Worker ID", self.worker_id or "Not registered")
        table.add_row("Wallet", f"{self.wallet_address[:10]}...{self.wallet_address[-8:]}")
        table.add_row("Status", "🟢 Running" if self.is_running else "🔴 Stopped")
        table.add_row("Device", f"{'GPU' if self.use_gpu else 'CPU'} ({self.device})")
        table.add_row("", "")
        table.add_row("Tasks Completed", str(self.tasks_completed))
        table.add_row("Tokens Earned", f"{self.tokens_earned:.4f} AOAI")
        table.add_row("Compute Time", f"{self.total_compute_time:.1f}s")
        
        if self.current_task:
            table.add_row("", "")
            table.add_row("Current Task", self.current_task.id[:16] + "...")
        
        return Panel(
            table,
            title="[bold blue]ActuallyOpenAI Worker[/bold blue]",
            border_style="blue"
        )
    
    async def run(self):
        """Main worker loop."""
        console.print("[bold green]Starting ActuallyOpenAI Worker Node[/bold green]")
        console.print(f"Wallet: {self.wallet_address}")
        console.print(f"Device: {'GPU' if self.use_gpu else 'CPU'}")
        console.print()
        
        # Register with orchestrator
        if not await self.register():
            console.print("[bold red]Failed to register with orchestrator[/bold red]")
            return
        
        # Connect WebSocket
        if not await self.connect_websocket():
            console.print("[bold yellow]WebSocket connection failed, using HTTP polling[/bold yellow]")
        
        self.is_running = True
        self.start_time = datetime.utcnow()
        
        console.print("[bold green]Worker is now online and ready for tasks![/bold green]")
        console.print("Press Ctrl+C to stop\n")
        
        # Start listening for tasks
        try:
            with Live(self.get_status_display(), refresh_per_second=1) as live:
                while self.is_running:
                    await self.listen_for_tasks()
                    live.update(self.get_status_display())
        except KeyboardInterrupt:
            pass
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown."""
        console.print("\n[yellow]Shutting down worker...[/yellow]")
        self.is_running = False
        
        if self.websocket:
            await self.websocket.close()
        
        await self.http_client.aclose()
        
        console.print(f"[green]Session summary:[/green]")
        console.print(f"  Tasks completed: {self.tasks_completed}")
        console.print(f"  Tokens earned: {self.tokens_earned:.4f} AOAI")
        console.print(f"  Total compute time: {self.total_compute_time:.1f}s")


async def main():
    """Entry point for worker node."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ActuallyOpenAI Worker Node")
    parser.add_argument(
        "--orchestrator",
        default="http://localhost:8000",
        help="Orchestrator server URL"
    )
    parser.add_argument(
        "--wallet",
        required=True,
        help="Your wallet address for receiving AOAI tokens"
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Worker name (optional)"
    )
    parser.add_argument(
        "--region",
        default="unknown",
        help="Worker region/location"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU only (no GPU)"
    )
    
    args = parser.parse_args()
    
    worker = WorkerNode(
        orchestrator_url=args.orchestrator,
        wallet_address=args.wallet,
        name=args.name,
        region=args.region,
        use_gpu=not args.cpu
    )
    
    # Handle shutdown signals
    def signal_handler(sig, frame):
        worker.is_running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
