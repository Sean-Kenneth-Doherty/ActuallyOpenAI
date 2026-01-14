"""
Network Compute Aggregator
==========================
Aggregates compute from all nodes in the network for large-scale training.

This is the key to scaling beyond what any single machine can do.

Architecture:
- Coordinator nodes organize training runs
- Worker nodes contribute compute
- Gradient aggregation across network
- Fault tolerance and recovery
- Economic incentives for participation
"""

import asyncio
import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Callable, Any
from enum import Enum
import logging

logger = logging.getLogger("AOAI-Aggregator")


class NodeRole(Enum):
    COORDINATOR = "coordinator"
    WORKER = "worker"
    VALIDATOR = "validator"


class TrainingState(Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    TRAINING = "training"
    SYNCING = "syncing"
    CHECKPOINTING = "checkpointing"


@dataclass
class ComputeCapability:
    """Describes a node's compute capabilities"""
    node_id: str
    gpu_memory_gb: float = 0.0
    gpu_type: str = "unknown"
    gpu_count: int = 0
    cpu_cores: int = 4
    ram_gb: float = 16.0
    bandwidth_mbps: float = 100.0
    
    @property
    def compute_score(self) -> float:
        """Calculate compute score for work allocation"""
        gpu_score = self.gpu_memory_gb * self.gpu_count * 10
        cpu_score = self.cpu_cores * 0.5
        return gpu_score + cpu_score
    
    def can_train_model(self, model_params: int, batch_size: int) -> bool:
        """Check if node can train a model of given size"""
        # Rough estimate: 4 bytes per param * 4 (model + gradients + optimizer states)
        required_gb = (model_params * 16) / (1024 ** 3)
        return self.gpu_memory_gb >= required_gb


@dataclass
class TrainingJob:
    """Describes a distributed training job"""
    job_id: str
    model_scale: str
    target_steps: int
    current_step: int = 0
    state: TrainingState = TrainingState.IDLE
    coordinator_id: str = ""
    worker_ids: List[str] = field(default_factory=list)
    
    # Rewards
    total_tokens_processed: int = 0
    tokens_per_node: Dict[str, int] = field(default_factory=dict)
    
    # Checkpoints
    checkpoint_hashes: List[str] = field(default_factory=list)
    last_checkpoint_step: int = 0
    
    # Timing
    started_at: float = 0
    estimated_completion: float = 0


@dataclass
class GradientContribution:
    """A gradient contribution from a worker node"""
    node_id: str
    job_id: str
    step: int
    gradient_hash: str
    tokens_processed: int
    loss: float
    timestamp: float = field(default_factory=time.time)
    signature: str = ""


class NetworkComputeAggregator:
    """
    Coordinates compute across the entire network for large-scale training.
    
    The more nodes participate, the larger models we can train.
    """
    
    def __init__(
        self,
        node_id: str,
        role: NodeRole = NodeRole.WORKER,
    ):
        self.node_id = node_id
        self.role = role
        
        # Network state
        self.known_nodes: Dict[str, ComputeCapability] = {}
        self.active_jobs: Dict[str, TrainingJob] = {}
        
        # Local state
        self.current_job: Optional[TrainingJob] = None
        self.pending_gradients: List[GradientContribution] = []
        
        # Callbacks
        self.on_job_started: Optional[Callable[[TrainingJob], None]] = None
        self.on_gradient_received: Optional[Callable[[GradientContribution], None]] = None
        self.on_job_completed: Optional[Callable[[TrainingJob], None]] = None
        
        # Network interface (set by P2P layer)
        self.broadcast: Optional[Callable[[str, dict], None]] = None
        self.send_to_node: Optional[Callable[[str, str, dict], None]] = None
        
        logger.info(f"Compute aggregator initialized as {role.value}")
    
    def register_capabilities(self, capabilities: ComputeCapability):
        """Register node compute capabilities"""
        self.known_nodes[capabilities.node_id] = capabilities
        logger.info(f"Node {capabilities.node_id[:16]} registered: score={capabilities.compute_score:.1f}")
    
    def get_total_network_compute(self) -> float:
        """Get total compute score across network"""
        return sum(n.compute_score for n in self.known_nodes.values())
    
    def get_capable_nodes(self, model_params: int, batch_size: int) -> List[str]:
        """Get nodes capable of training a given model"""
        capable = []
        for node_id, capabilities in self.known_nodes.items():
            if capabilities.can_train_model(model_params, batch_size):
                capable.append(node_id)
        return capable
    
    def estimate_training_time(
        self,
        model_params: int,
        dataset_tokens: int,
        target_steps: int,
    ) -> float:
        """Estimate training time with current network"""
        # Get capable nodes
        capable = self.get_capable_nodes(model_params, batch_size=32)
        if not capable:
            return float('inf')
        
        # Estimate throughput (tokens/second)
        total_compute = sum(
            self.known_nodes[n].compute_score 
            for n in capable
        )
        
        # Rough estimate: 100 tokens/sec per compute unit
        tokens_per_second = total_compute * 100
        
        return dataset_tokens / tokens_per_second
    
    async def propose_training_job(
        self,
        model_scale: str,
        target_steps: int,
        min_nodes: int = 3,
    ) -> Optional[TrainingJob]:
        """Propose a new training job to the network"""
        if self.role != NodeRole.COORDINATOR:
            logger.warning("Only coordinators can propose jobs")
            return None
        
        # Get model params for this scale
        param_estimates = {
            "tiny": 10_000_000,
            "small": 100_000_000,
            "medium": 1_000_000_000,
            "large": 7_000_000_000,
            "xlarge": 70_000_000_000,
            "frontier": 400_000_000_000,
        }
        
        model_params = param_estimates.get(model_scale, 100_000_000)
        
        # Find capable nodes
        capable_nodes = self.get_capable_nodes(model_params, batch_size=32)
        
        if len(capable_nodes) < min_nodes:
            logger.warning(f"Not enough capable nodes: {len(capable_nodes)} < {min_nodes}")
            return None
        
        # Create job
        job = TrainingJob(
            job_id=hashlib.sha256(f"{time.time()}{model_scale}".encode()).hexdigest()[:16],
            model_scale=model_scale,
            target_steps=target_steps,
            coordinator_id=self.node_id,
            worker_ids=capable_nodes,
            started_at=time.time(),
        )
        
        self.active_jobs[job.job_id] = job
        
        # Broadcast job to network
        if self.broadcast:
            self.broadcast("training_job_proposed", {
                "job": job.__dict__,
            })
        
        logger.info(f"Proposed training job {job.job_id} with {len(capable_nodes)} nodes")
        return job
    
    async def join_training_job(self, job_id: str) -> bool:
        """Join an existing training job as a worker"""
        if job_id not in self.active_jobs:
            logger.warning(f"Unknown job: {job_id}")
            return False
        
        job = self.active_jobs[job_id]
        
        if self.node_id in job.worker_ids:
            logger.info(f"Already in job {job_id}")
            return True
        
        job.worker_ids.append(self.node_id)
        self.current_job = job
        
        # Notify coordinator
        if self.send_to_node:
            self.send_to_node(job.coordinator_id, "worker_joined", {
                "job_id": job_id,
                "node_id": self.node_id,
                "capabilities": self.known_nodes.get(self.node_id, {}).__dict__
                    if self.node_id in self.known_nodes else {},
            })
        
        logger.info(f"Joined training job {job_id}")
        return True
    
    def submit_gradient(
        self,
        job_id: str,
        step: int,
        gradient_hash: str,
        tokens_processed: int,
        loss: float,
    ) -> GradientContribution:
        """Submit a gradient contribution"""
        contribution = GradientContribution(
            node_id=self.node_id,
            job_id=job_id,
            step=step,
            gradient_hash=gradient_hash,
            tokens_processed=tokens_processed,
            loss=loss,
        )
        
        # Sign contribution
        contribution.signature = hashlib.sha256(
            f"{contribution.node_id}:{contribution.step}:{contribution.gradient_hash}".encode()
        ).hexdigest()[:16]
        
        # Update job stats
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.total_tokens_processed += tokens_processed
            job.tokens_per_node[self.node_id] = (
                job.tokens_per_node.get(self.node_id, 0) + tokens_processed
            )
            job.current_step = max(job.current_step, step)
        
        # Broadcast to network
        if self.broadcast:
            self.broadcast("gradient_submitted", contribution.__dict__)
        
        return contribution
    
    def receive_gradient(self, contribution: GradientContribution):
        """Receive a gradient contribution from the network"""
        # Verify signature
        expected_sig = hashlib.sha256(
            f"{contribution.node_id}:{contribution.step}:{contribution.gradient_hash}".encode()
        ).hexdigest()[:16]
        
        if contribution.signature != expected_sig:
            logger.warning(f"Invalid gradient signature from {contribution.node_id[:16]}")
            return
        
        self.pending_gradients.append(contribution)
        
        if self.on_gradient_received:
            self.on_gradient_received(contribution)
        
        logger.debug(f"Received gradient from {contribution.node_id[:16]}, step {contribution.step}")
    
    def get_aggregated_step(self, job_id: str, step: int) -> Dict[str, Any]:
        """Get aggregated gradients for a step"""
        contributions = [
            g for g in self.pending_gradients
            if g.job_id == job_id and g.step == step
        ]
        
        if not contributions:
            return {"status": "pending", "contributions": 0}
        
        # Calculate weighted average loss
        total_tokens = sum(c.tokens_processed for c in contributions)
        weighted_loss = sum(
            c.loss * c.tokens_processed for c in contributions
        ) / total_tokens if total_tokens > 0 else 0
        
        return {
            "status": "ready",
            "contributions": len(contributions),
            "total_tokens": total_tokens,
            "weighted_loss": weighted_loss,
            "gradient_hashes": [c.gradient_hash for c in contributions],
        }
    
    def calculate_rewards(self, job_id: str) -> Dict[str, float]:
        """Calculate token rewards for job participants"""
        if job_id not in self.active_jobs:
            return {}
        
        job = self.active_jobs[job_id]
        
        if job.total_tokens_processed == 0:
            return {}
        
        # Reward based on contribution proportion
        # Base rate: 1 token per 1M tokens processed
        base_reward = job.total_tokens_processed / 1_000_000
        
        rewards = {}
        for node_id, tokens in job.tokens_per_node.items():
            proportion = tokens / job.total_tokens_processed
            rewards[node_id] = base_reward * proportion
        
        return rewards
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get current network training status"""
        return {
            "total_nodes": len(self.known_nodes),
            "total_compute": self.get_total_network_compute(),
            "active_jobs": len(self.active_jobs),
            "pending_gradients": len(self.pending_gradients),
            "jobs": {
                job_id: {
                    "model_scale": job.model_scale,
                    "state": job.state.value,
                    "progress": f"{job.current_step}/{job.target_steps}",
                    "workers": len(job.worker_ids),
                    "tokens_processed": job.total_tokens_processed,
                }
                for job_id, job in self.active_jobs.items()
            }
        }


class TrainingCoordinator:
    """
    High-level coordinator for network-wide training.
    
    Manages the full lifecycle of distributed training:
    1. Data distribution
    2. Model initialization  
    3. Training orchestration
    4. Checkpoint management
    5. Model publishing
    """
    
    def __init__(
        self,
        aggregator: NetworkComputeAggregator,
        data_pipeline,  # PretrainingDataPipeline
        model_factory,  # Callable to create models
    ):
        self.aggregator = aggregator
        self.data_pipeline = data_pipeline
        self.model_factory = model_factory
        
        self.current_training = None
        self.model = None
        self.trainer = None
    
    async def start_training(
        self,
        model_scale: str,
        target_tokens: int,
        checkpoint_every: int = 10000,
    ):
        """Start a distributed training run"""
        logger.info(f"Starting {model_scale} training, target: {target_tokens:,} tokens")
        
        # Create model
        self.model = self.model_factory(model_scale)
        model_params = sum(p.numel() for p in self.model.parameters())
        
        # Propose job to network
        steps = target_tokens // (2048 * 32)  # sequence_length * batch_size
        job = await self.aggregator.propose_training_job(model_scale, steps)
        
        if job is None:
            logger.error("Failed to create training job")
            return
        
        self.current_training = job
        job.state = TrainingState.PREPARING
        
        # Wait for workers to join
        await asyncio.sleep(10)
        
        if len(job.worker_ids) < 2:
            logger.warning("Not enough workers joined")
            return
        
        # Start training
        job.state = TrainingState.TRAINING
        
        logger.info(f"Training started with {len(job.worker_ids)} workers")
    
    async def training_loop(self):
        """Main training loop"""
        if self.current_training is None:
            return
        
        job = self.current_training
        
        while job.current_step < job.target_steps:
            # Check for aggregated gradients
            step_data = self.aggregator.get_aggregated_step(job.job_id, job.current_step)
            
            if step_data["status"] == "ready":
                # Apply aggregated gradients
                # (In practice, this would involve actual gradient application)
                logger.info(f"Step {job.current_step}: loss={step_data['weighted_loss']:.4f}")
                job.current_step += 1
            
            # Checkpoint periodically
            if job.current_step % 10000 == 0 and job.current_step > job.last_checkpoint_step:
                await self.save_checkpoint()
                job.last_checkpoint_step = job.current_step
            
            await asyncio.sleep(0.1)
        
        # Training complete
        job.state = TrainingState.IDLE
        
        # Calculate and distribute rewards
        rewards = self.aggregator.calculate_rewards(job.job_id)
        logger.info(f"Training complete. Rewards: {rewards}")
        
        if self.aggregator.on_job_completed:
            self.aggregator.on_job_completed(job)
    
    async def save_checkpoint(self):
        """Save training checkpoint"""
        if self.model is None or self.current_training is None:
            return
        
        # In practice, checkpoint would be saved to IPFS and hash recorded
        checkpoint_hash = hashlib.sha256(
            f"{self.current_training.job_id}:{self.current_training.current_step}".encode()
        ).hexdigest()
        
        self.current_training.checkpoint_hashes.append(checkpoint_hash)
        
        logger.info(f"Checkpoint saved: {checkpoint_hash[:16]}...")


if __name__ == "__main__":
    # Test aggregator
    aggregator = NetworkComputeAggregator("test_node", NodeRole.COORDINATOR)
    
    # Register some nodes
    for i in range(5):
        cap = ComputeCapability(
            node_id=f"node_{i}",
            gpu_memory_gb=16 + i * 8,
            gpu_type="RTX 4090",
            gpu_count=1,
        )
        aggregator.register_capabilities(cap)
    
    print(f"Total network compute: {aggregator.get_total_network_compute():.1f}")
    
    # Check capable nodes
    capable = aggregator.get_capable_nodes(model_params=1_000_000_000, batch_size=32)
    print(f"Nodes capable of 1B params: {len(capable)}")
    
    # Estimate training time
    time_estimate = aggregator.estimate_training_time(
        model_params=1_000_000_000,
        dataset_tokens=100_000_000_000,
        target_steps=100000,
    )
    print(f"Estimated training time: {time_estimate/3600:.1f} hours")
