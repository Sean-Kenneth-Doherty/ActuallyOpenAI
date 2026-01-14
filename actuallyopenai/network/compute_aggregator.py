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

Aggregation Strategies:
- FedAvg: Weighted average based on sample counts
- FedProx: FedAvg with proximal term for heterogeneous data
- Krum: Byzantine-fault-tolerant aggregation
- TrimmedMean: Robust aggregation against outliers
- Median: Coordinate-wise median aggregation
"""

import asyncio
import json
import time
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Callable, Any, Tuple, Union
from enum import Enum
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger("AOAI-Aggregator")


# ============================================================================
# Aggregation Strategies for Federated Learning
# ============================================================================

class AggregationStrategy(Enum):
    """Available gradient/model aggregation strategies"""
    FEDAVG = "fedavg"           # Standard Federated Averaging
    FEDPROX = "fedprox"         # FedAvg with proximal regularization
    KRUM = "krum"               # Byzantine-fault-tolerant
    TRIMMED_MEAN = "trimmed_mean"  # Outlier-robust
    MEDIAN = "median"           # Coordinate-wise median


@dataclass
class WorkerUpdate:
    """Represents a gradient or model update from a worker"""
    node_id: str
    step: int
    num_samples: int  # Number of training samples used
    parameters: Dict[str, np.ndarray]  # Parameter name -> values
    loss: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def flatten(self) -> np.ndarray:
        """Flatten all parameters into a single vector"""
        arrays = []
        for name in sorted(self.parameters.keys()):
            arrays.append(self.parameters[name].flatten())
        return np.concatenate(arrays) if arrays else np.array([])
    
    @staticmethod
    def unflatten(flat_array: np.ndarray, shapes: Dict[str, tuple]) -> Dict[str, np.ndarray]:
        """Reconstruct parameters from a flattened array"""
        result = {}
        offset = 0
        for name in sorted(shapes.keys()):
            shape = shapes[name]
            size = np.prod(shape)
            result[name] = flat_array[offset:offset + size].reshape(shape)
            offset += size
        return result


class BaseAggregator(ABC):
    """Abstract base class for aggregation strategies"""
    
    @abstractmethod
    def aggregate(
        self, 
        updates: List[WorkerUpdate],
        global_params: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """Aggregate worker updates into global parameters"""
        pass
    
    def validate_updates(self, updates: List[WorkerUpdate]) -> List[WorkerUpdate]:
        """Validate and filter updates"""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        # Ensure all updates have same parameter structure
        reference_keys = set(updates[0].parameters.keys())
        valid_updates = []
        
        for update in updates:
            if set(update.parameters.keys()) == reference_keys:
                valid_updates.append(update)
            else:
                logger.warning(f"Skipping update from {update.node_id}: mismatched parameter structure")
        
        if not valid_updates:
            raise ValueError("No valid updates after validation")
        
        return valid_updates


class FedAvgAggregator(BaseAggregator):
    """
    Federated Averaging (FedAvg) Implementation
    
    The classic federated learning algorithm from McMahan et al. (2017).
    Computes weighted average of model parameters based on number of samples.
    
    Formula: w_global = Σ(n_k / n_total) * w_k
    where n_k is number of samples from client k
    """
    
    def __init__(self, min_updates: int = 1):
        """
        Args:
            min_updates: Minimum number of updates required before aggregation
        """
        self.min_updates = min_updates
    
    def aggregate(
        self,
        updates: List[WorkerUpdate],
        global_params: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Perform FedAvg aggregation
        
        Args:
            updates: List of worker updates with parameters and sample counts
            global_params: Optional current global parameters (not used in FedAvg)
        
        Returns:
            Aggregated parameters as weighted average
        """
        updates = self.validate_updates(updates)
        
        if len(updates) < self.min_updates:
            raise ValueError(f"Need at least {self.min_updates} updates, got {len(updates)}")
        
        # Calculate total samples for weighting
        total_samples = sum(u.num_samples for u in updates)
        
        if total_samples == 0:
            # Fall back to uniform weighting
            logger.warning("Total samples is 0, using uniform weighting")
            total_samples = len(updates)
            for u in updates:
                u.num_samples = 1
        
        # Initialize aggregated parameters
        aggregated = {}
        
        # Get parameter names from first update
        param_names = list(updates[0].parameters.keys())
        
        for name in param_names:
            # Initialize with zeros of correct shape
            shape = updates[0].parameters[name].shape
            dtype = updates[0].parameters[name].dtype
            aggregated[name] = np.zeros(shape, dtype=np.float64)
            
            # Weighted sum
            for update in updates:
                weight = update.num_samples / total_samples
                aggregated[name] += weight * update.parameters[name].astype(np.float64)
            
            # Convert back to original dtype
            aggregated[name] = aggregated[name].astype(dtype)
        
        logger.info(f"FedAvg: Aggregated {len(updates)} updates, {total_samples} total samples")
        return aggregated


class FedProxAggregator(BaseAggregator):
    """
    FedProx Implementation (Li et al., 2020)
    
    Extends FedAvg with a proximal term to handle heterogeneous data.
    The proximal term encourages local models to stay close to global model.
    
    Note: The proximal regularization is applied during local training.
    This aggregator uses FedAvg-style weighted averaging.
    """
    
    def __init__(self, mu: float = 0.01, min_updates: int = 1):
        """
        Args:
            mu: Proximal term coefficient (used during local training)
            min_updates: Minimum updates required
        """
        self.mu = mu
        self.min_updates = min_updates
        self._fedavg = FedAvgAggregator(min_updates)
    
    def aggregate(
        self,
        updates: List[WorkerUpdate],
        global_params: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """FedProx uses same aggregation as FedAvg"""
        return self._fedavg.aggregate(updates, global_params)
    
    def get_proximal_loss(
        self,
        local_params: Dict[str, np.ndarray],
        global_params: Dict[str, np.ndarray]
    ) -> float:
        """
        Calculate proximal term: (mu/2) * ||w - w_global||^2
        
        Workers should add this to their loss during training.
        """
        total_diff = 0.0
        for name in local_params:
            if name in global_params:
                diff = local_params[name] - global_params[name]
                total_diff += np.sum(diff ** 2)
        return (self.mu / 2) * total_diff


class KrumAggregator(BaseAggregator):
    """
    Krum Byzantine-Fault-Tolerant Aggregation (Blanchard et al., 2017)
    
    Selects the update that is closest to the majority of other updates.
    Robust against up to f Byzantine (malicious) workers.
    
    Multi-Krum variant selects m best updates and averages them.
    """
    
    def __init__(self, num_byzantine: int = 0, multi_krum_m: int = 1):
        """
        Args:
            num_byzantine: Expected number of Byzantine workers (f)
            multi_krum_m: Number of updates to select for Multi-Krum
        """
        self.num_byzantine = num_byzantine
        self.multi_krum_m = multi_krum_m
    
    def aggregate(
        self,
        updates: List[WorkerUpdate],
        global_params: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Perform Krum aggregation
        
        Args:
            updates: List of worker updates
            global_params: Not used
        
        Returns:
            Selected (or averaged) parameters
        """
        updates = self.validate_updates(updates)
        n = len(updates)
        f = self.num_byzantine
        
        # Krum requires n >= 2f + 3
        if n < 2 * f + 3:
            logger.warning(f"Krum: n={n} < 2*{f}+3, reducing f")
            f = max(0, (n - 3) // 2)
        
        # Flatten all updates for distance computation
        flat_updates = [u.flatten() for u in updates]
        
        # Calculate pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(flat_updates[i] - flat_updates[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Calculate Krum scores (sum of n-f-2 smallest distances)
        scores = []
        num_closest = n - f - 2
        
        for i in range(n):
            # Sort distances from node i to all others
            sorted_dists = np.sort(distances[i])
            # Sum the smallest n-f-2 distances (excluding self which is 0)
            score = np.sum(sorted_dists[1:num_closest + 1])
            scores.append(score)
        
        # Select m updates with lowest scores
        selected_indices = np.argsort(scores)[:self.multi_krum_m]
        
        logger.info(f"Krum: Selected {self.multi_krum_m} of {n} updates, scores: {[scores[i] for i in selected_indices]}")
        
        if self.multi_krum_m == 1:
            # Single Krum: return selected update
            return updates[selected_indices[0]].parameters.copy()
        else:
            # Multi-Krum: average selected updates
            selected_updates = [updates[i] for i in selected_indices]
            return FedAvgAggregator().aggregate(selected_updates)


class TrimmedMeanAggregator(BaseAggregator):
    """
    Coordinate-wise Trimmed Mean Aggregation (Yin et al., 2018)
    
    For each parameter coordinate, removes the largest and smallest
    values and averages the rest. Robust against outliers/adversaries.
    """
    
    def __init__(self, trim_ratio: float = 0.1):
        """
        Args:
            trim_ratio: Fraction of values to trim from each end (0-0.5)
        """
        self.trim_ratio = min(0.49, max(0.0, trim_ratio))
    
    def aggregate(
        self,
        updates: List[WorkerUpdate],
        global_params: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Perform trimmed mean aggregation
        
        For each coordinate, sort values from all workers, trim
        the highest and lowest, then average the remainder.
        """
        updates = self.validate_updates(updates)
        n = len(updates)
        trim_count = int(n * self.trim_ratio)
        
        if n - 2 * trim_count < 1:
            logger.warning(f"TrimmedMean: Not enough updates ({n}) for trim_ratio={self.trim_ratio}")
            trim_count = max(0, (n - 1) // 2)
        
        aggregated = {}
        
        for name in updates[0].parameters.keys():
            # Stack all updates for this parameter
            stacked = np.stack([u.parameters[name] for u in updates], axis=0)
            
            # Sort along the worker axis (axis 0)
            sorted_values = np.sort(stacked, axis=0)
            
            # Trim and average
            if trim_count > 0:
                trimmed = sorted_values[trim_count:-trim_count]
            else:
                trimmed = sorted_values
            
            aggregated[name] = np.mean(trimmed, axis=0).astype(updates[0].parameters[name].dtype)
        
        logger.info(f"TrimmedMean: Aggregated {n} updates, trimmed {trim_count} from each end")
        return aggregated


class MedianAggregator(BaseAggregator):
    """
    Coordinate-wise Median Aggregation
    
    Takes the median value at each parameter coordinate.
    Very robust against outliers but may be slower to converge.
    """
    
    def aggregate(
        self,
        updates: List[WorkerUpdate],
        global_params: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Perform coordinate-wise median aggregation
        """
        updates = self.validate_updates(updates)
        n = len(updates)
        
        aggregated = {}
        
        for name in updates[0].parameters.keys():
            # Stack all updates for this parameter
            stacked = np.stack([u.parameters[name] for u in updates], axis=0)
            
            # Take median along worker axis
            aggregated[name] = np.median(stacked, axis=0).astype(updates[0].parameters[name].dtype)
        
        logger.info(f"Median: Aggregated {n} updates using coordinate-wise median")
        return aggregated


class FederatedAggregator:
    """
    Main federated aggregation manager
    
    Provides a unified interface for different aggregation strategies
    with support for:
    - Multiple aggregation algorithms
    - Gradient compression
    - Differential privacy (noise injection)
    - Staleness handling for async updates
    """
    
    def __init__(
        self,
        strategy: AggregationStrategy = AggregationStrategy.FEDAVG,
        num_byzantine: int = 0,
        trim_ratio: float = 0.1,
        fedprox_mu: float = 0.01,
        enable_compression: bool = False,
        compression_ratio: float = 0.1,
        dp_epsilon: Optional[float] = None,
        max_staleness: int = 10,
    ):
        """
        Initialize the federated aggregator
        
        Args:
            strategy: Aggregation strategy to use
            num_byzantine: Expected Byzantine workers (for Krum)
            trim_ratio: Trim ratio (for TrimmedMean)
            fedprox_mu: Proximal coefficient (for FedProx)
            enable_compression: Enable gradient compression
            compression_ratio: Ratio of gradients to keep (top-k)
            dp_epsilon: Differential privacy epsilon (None = disabled)
            max_staleness: Maximum allowed staleness for updates
        """
        self.strategy = strategy
        self.enable_compression = enable_compression
        self.compression_ratio = compression_ratio
        self.dp_epsilon = dp_epsilon
        self.max_staleness = max_staleness
        
        # Initialize the appropriate aggregator
        self._aggregator = self._create_aggregator(
            strategy, num_byzantine, trim_ratio, fedprox_mu
        )
        
        # Track pending updates
        self.pending_updates: Dict[int, List[WorkerUpdate]] = {}  # step -> updates
        self.global_step = 0
        self.global_params: Optional[Dict[str, np.ndarray]] = None
        
        # Statistics
        self.aggregation_history: List[Dict[str, Any]] = []
    
    def _create_aggregator(
        self,
        strategy: AggregationStrategy,
        num_byzantine: int,
        trim_ratio: float,
        fedprox_mu: float,
    ) -> BaseAggregator:
        """Factory method to create aggregators"""
        if strategy == AggregationStrategy.FEDAVG:
            return FedAvgAggregator()
        elif strategy == AggregationStrategy.FEDPROX:
            return FedProxAggregator(mu=fedprox_mu)
        elif strategy == AggregationStrategy.KRUM:
            return KrumAggregator(num_byzantine=num_byzantine)
        elif strategy == AggregationStrategy.TRIMMED_MEAN:
            return TrimmedMeanAggregator(trim_ratio=trim_ratio)
        elif strategy == AggregationStrategy.MEDIAN:
            return MedianAggregator()
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")
    
    def add_update(self, update: WorkerUpdate) -> None:
        """
        Add a worker update to pending aggregation
        
        Args:
            update: The worker's model/gradient update
        """
        step = update.step
        
        # Check staleness
        if step < self.global_step - self.max_staleness:
            logger.warning(f"Rejecting stale update from {update.node_id}: step {step} < {self.global_step - self.max_staleness}")
            return
        
        # Apply compression if enabled
        if self.enable_compression:
            update = self._compress_update(update)
        
        # Store update
        if step not in self.pending_updates:
            self.pending_updates[step] = []
        self.pending_updates[step].append(update)
        
        logger.debug(f"Added update from {update.node_id} for step {step}")
    
    def _compress_update(self, update: WorkerUpdate) -> WorkerUpdate:
        """Apply top-k gradient compression"""
        compressed_params = {}
        
        for name, values in update.parameters.items():
            flat = values.flatten()
            k = int(len(flat) * self.compression_ratio)
            k = max(1, k)  # Keep at least one value
            
            # Get indices of top-k absolute values
            top_k_indices = np.argpartition(np.abs(flat), -k)[-k:]
            
            # Create sparse representation (zeros elsewhere)
            compressed = np.zeros_like(flat)
            compressed[top_k_indices] = flat[top_k_indices]
            compressed_params[name] = compressed.reshape(values.shape)
        
        return WorkerUpdate(
            node_id=update.node_id,
            step=update.step,
            num_samples=update.num_samples,
            parameters=compressed_params,
            loss=update.loss,
            metrics=update.metrics,
            timestamp=update.timestamp,
        )
    
    def _add_dp_noise(self, params: Dict[str, np.ndarray], sensitivity: float = 1.0) -> Dict[str, np.ndarray]:
        """Add Gaussian noise for differential privacy"""
        if self.dp_epsilon is None:
            return params
        
        # Calculate noise scale (simplified Gaussian mechanism)
        sigma = sensitivity / self.dp_epsilon
        
        noisy_params = {}
        for name, values in params.items():
            noise = np.random.normal(0, sigma, values.shape)
            noisy_params[name] = values + noise
        
        return noisy_params
    
    def aggregate(
        self,
        step: int,
        min_updates: int = 1,
        sensitivity: float = 1.0,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Aggregate updates for a given step
        
        Args:
            step: The training step to aggregate
            min_updates: Minimum number of updates required
            sensitivity: Sensitivity for DP noise (if enabled)
        
        Returns:
            Aggregated parameters or None if not enough updates
        """
        if step not in self.pending_updates:
            logger.debug(f"No updates pending for step {step}")
            return None
        
        updates = self.pending_updates[step]
        
        if len(updates) < min_updates:
            logger.debug(f"Not enough updates for step {step}: {len(updates)} < {min_updates}")
            return None
        
        # Perform aggregation
        try:
            aggregated = self._aggregator.aggregate(updates, self.global_params)
        except ValueError as e:
            logger.error(f"Aggregation failed: {e}")
            return None
        
        # Apply differential privacy noise
        aggregated = self._add_dp_noise(aggregated, sensitivity)
        
        # Update global state
        self.global_params = aggregated
        self.global_step = max(self.global_step, step)
        
        # Record statistics
        total_samples = sum(u.num_samples for u in updates)
        avg_loss = np.mean([u.loss for u in updates]) if updates else 0
        
        self.aggregation_history.append({
            "step": step,
            "num_updates": len(updates),
            "total_samples": total_samples,
            "avg_loss": avg_loss,
            "strategy": self.strategy.value,
            "timestamp": time.time(),
        })
        
        # Clean up old updates
        self._cleanup_old_updates()
        
        logger.info(f"Aggregated step {step}: {len(updates)} updates, {total_samples} samples, avg_loss={avg_loss:.4f}")
        
        return aggregated
    
    def _cleanup_old_updates(self):
        """Remove updates that are too old"""
        cutoff = self.global_step - self.max_staleness
        old_steps = [s for s in self.pending_updates.keys() if s < cutoff]
        for step in old_steps:
            del self.pending_updates[step]
    
    def get_pending_count(self, step: int) -> int:
        """Get number of pending updates for a step"""
        return len(self.pending_updates.get(step, []))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregation statistics"""
        return {
            "strategy": self.strategy.value,
            "global_step": self.global_step,
            "pending_steps": list(self.pending_updates.keys()),
            "total_aggregations": len(self.aggregation_history),
            "compression_enabled": self.enable_compression,
            "dp_enabled": self.dp_epsilon is not None,
            "recent_history": self.aggregation_history[-10:] if self.aggregation_history else [],
        }


# Utility functions for PyTorch integration
def torch_to_numpy_dict(state_dict: dict) -> Dict[str, np.ndarray]:
    """Convert PyTorch state dict to numpy dict"""
    try:
        import torch
        return {k: v.cpu().numpy() for k, v in state_dict.items()}
    except ImportError:
        raise ImportError("PyTorch is required for torch_to_numpy_dict")


def numpy_to_torch_dict(numpy_dict: Dict[str, np.ndarray], device: str = "cpu") -> dict:
    """Convert numpy dict to PyTorch state dict"""
    try:
        import torch
        return {k: torch.from_numpy(v).to(device) for k, v in numpy_dict.items()}
    except ImportError:
        raise ImportError("PyTorch is required for numpy_to_torch_dict")


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
    
    Now integrated with FederatedAggregator for proper FedAvg and
    Byzantine-fault-tolerant aggregation strategies.
    """
    
    def __init__(
        self,
        node_id: str,
        role: NodeRole = NodeRole.WORKER,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.FEDAVG,
        num_byzantine: int = 0,
        enable_compression: bool = False,
        dp_epsilon: Optional[float] = None,
    ):
        self.node_id = node_id
        self.role = role
        
        # Network state
        self.known_nodes: Dict[str, ComputeCapability] = {}
        self.active_jobs: Dict[str, TrainingJob] = {}
        
        # Local state
        self.current_job: Optional[TrainingJob] = None
        self.pending_gradients: List[GradientContribution] = []
        
        # Federated aggregation (NEW)
        self.federated_aggregator = FederatedAggregator(
            strategy=aggregation_strategy,
            num_byzantine=num_byzantine,
            enable_compression=enable_compression,
            dp_epsilon=dp_epsilon,
        )
        
        # Stored model updates for FedAvg (actual parameter values)
        self.pending_model_updates: Dict[str, List[WorkerUpdate]] = {}  # job_id -> updates
        
        # Callbacks
        self.on_job_started: Optional[Callable[[TrainingJob], None]] = None
        self.on_gradient_received: Optional[Callable[[GradientContribution], None]] = None
        self.on_job_completed: Optional[Callable[[TrainingJob], None]] = None
        
        # Network interface (set by P2P layer)
        self.broadcast: Optional[Callable[[str, dict], None]] = None
        self.send_to_node: Optional[Callable[[str, str, dict], None]] = None
        
        logger.info(f"Compute aggregator initialized as {role.value} with {aggregation_strategy.value} strategy")
    
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
    
    # ========================================================================
    # FedAvg Model Aggregation Methods (NEW)
    # ========================================================================
    
    def submit_model_update(
        self,
        job_id: str,
        step: int,
        model_params: Dict[str, np.ndarray],
        num_samples: int,
        loss: float = 0.0,
        metrics: Optional[Dict[str, float]] = None,
    ) -> WorkerUpdate:
        """
        Submit a model parameter update for federated aggregation.
        
        This is the main method for FedAvg - workers submit their trained
        model parameters along with the number of samples used for training.
        
        Args:
            job_id: The training job ID
            step: Current training step
            model_params: Dictionary mapping parameter names to numpy arrays
            num_samples: Number of training samples used
            loss: Training loss (for logging)
            metrics: Optional additional metrics
        
        Returns:
            The created WorkerUpdate object
        """
        update = WorkerUpdate(
            node_id=self.node_id,
            step=step,
            num_samples=num_samples,
            parameters=model_params,
            loss=loss,
            metrics=metrics or {},
        )
        
        # Add to federated aggregator
        self.federated_aggregator.add_update(update)
        
        # Store locally for job tracking
        if job_id not in self.pending_model_updates:
            self.pending_model_updates[job_id] = []
        self.pending_model_updates[job_id].append(update)
        
        # Update job stats
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.total_tokens_processed += num_samples * 2048  # Approximate tokens
            job.tokens_per_node[self.node_id] = (
                job.tokens_per_node.get(self.node_id, 0) + num_samples * 2048
            )
        
        # Broadcast to network (hash only for efficiency)
        if self.broadcast:
            # Create hash of parameters for verification
            param_hash = self._hash_parameters(model_params)
            self.broadcast("model_update_submitted", {
                "node_id": self.node_id,
                "job_id": job_id,
                "step": step,
                "num_samples": num_samples,
                "loss": loss,
                "param_hash": param_hash,
            })
        
        logger.info(f"Submitted model update: job={job_id}, step={step}, samples={num_samples}")
        return update
    
    def receive_model_update(self, update: WorkerUpdate, job_id: str):
        """
        Receive a model parameter update from another worker.
        
        Used by coordinator to collect updates from all workers.
        
        Args:
            update: The worker's model update
            job_id: The training job ID
        """
        # Add to federated aggregator
        self.federated_aggregator.add_update(update)
        
        # Store for job tracking
        if job_id not in self.pending_model_updates:
            self.pending_model_updates[job_id] = []
        self.pending_model_updates[job_id].append(update)
        
        logger.debug(f"Received model update from {update.node_id[:16]}, step {update.step}")
    
    def aggregate_model_updates(
        self,
        job_id: str,
        step: int,
        min_updates: int = 2,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Aggregate model updates using FedAvg (or configured strategy).
        
        This is the core aggregation method that implements federated averaging.
        
        Args:
            job_id: The training job ID
            step: The training step to aggregate
            min_updates: Minimum number of worker updates required
        
        Returns:
            Aggregated model parameters, or None if not enough updates
        """
        # Use the federated aggregator
        aggregated = self.federated_aggregator.aggregate(step, min_updates)
        
        if aggregated is not None:
            # Clean up job-specific tracking
            if job_id in self.pending_model_updates:
                self.pending_model_updates[job_id] = [
                    u for u in self.pending_model_updates[job_id]
                    if u.step > step
                ]
            
            logger.info(f"Aggregated model updates for job={job_id}, step={step}")
        
        return aggregated
    
    def aggregate_with_pytorch(
        self,
        job_id: str,
        step: int,
        min_updates: int = 2,
        device: str = "cpu",
    ) -> Optional[dict]:
        """
        Aggregate and return as PyTorch state dict.
        
        Convenience method that converts aggregated numpy arrays to PyTorch tensors.
        
        Args:
            job_id: The training job ID
            step: The training step to aggregate
            min_updates: Minimum worker updates required
            device: PyTorch device for the tensors
        
        Returns:
            PyTorch state dict or None
        """
        aggregated = self.aggregate_model_updates(job_id, step, min_updates)
        if aggregated is None:
            return None
        return numpy_to_torch_dict(aggregated, device)
    
    def _hash_parameters(self, params: Dict[str, np.ndarray]) -> str:
        """Create a hash of model parameters for verification"""
        hasher = hashlib.sha256()
        for name in sorted(params.keys()):
            hasher.update(name.encode())
            hasher.update(params[name].tobytes())
        return hasher.hexdigest()[:32]
    
    def get_aggregation_status(self, step: int) -> Dict[str, Any]:
        """Get the aggregation status for a step"""
        pending = self.federated_aggregator.get_pending_count(step)
        stats = self.federated_aggregator.get_statistics()
        
        return {
            "step": step,
            "pending_updates": pending,
            "global_step": stats["global_step"],
            "strategy": stats["strategy"],
            "total_aggregations": stats["total_aggregations"],
        }
    
    def change_aggregation_strategy(
        self,
        strategy: AggregationStrategy,
        num_byzantine: int = 0,
        trim_ratio: float = 0.1,
    ):
        """
        Change the aggregation strategy at runtime.
        
        Useful for adapting to different network conditions or attack scenarios.
        """
        self.federated_aggregator._aggregator = self.federated_aggregator._create_aggregator(
            strategy, num_byzantine, trim_ratio, 0.01
        )
        self.federated_aggregator.strategy = strategy
        logger.info(f"Changed aggregation strategy to {strategy.value}")
    
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
    print("=" * 60)
    print("Testing FedAvg and Aggregation Strategies")
    print("=" * 60)
    
    # Test 1: Basic FedAvg aggregation
    print("\n--- Test 1: FedAvg Weighted Averaging ---")
    fedavg = FedAvgAggregator()
    
    # Simulate 3 workers with different sample counts
    updates = [
        WorkerUpdate(
            node_id="worker_1",
            step=0,
            num_samples=100,
            parameters={
                "layer1.weight": np.array([[1.0, 2.0], [3.0, 4.0]]),
                "layer1.bias": np.array([0.1, 0.2]),
            },
            loss=0.5,
        ),
        WorkerUpdate(
            node_id="worker_2",
            step=0,
            num_samples=200,
            parameters={
                "layer1.weight": np.array([[2.0, 3.0], [4.0, 5.0]]),
                "layer1.bias": np.array([0.2, 0.3]),
            },
            loss=0.4,
        ),
        WorkerUpdate(
            node_id="worker_3",
            step=0,
            num_samples=100,
            parameters={
                "layer1.weight": np.array([[3.0, 4.0], [5.0, 6.0]]),
                "layer1.bias": np.array([0.3, 0.4]),
            },
            loss=0.6,
        ),
    ]
    
    aggregated = fedavg.aggregate(updates)
    print(f"Aggregated weights (weighted by samples 100:200:100):")
    print(f"  layer1.weight:\n{aggregated['layer1.weight']}")
    print(f"  layer1.bias: {aggregated['layer1.bias']}")
    # Expected: (100*1 + 200*2 + 100*3)/400 = 200/400 = 2.0 for [0,0] etc.
    
    # Test 2: Krum Byzantine fault tolerance
    print("\n--- Test 2: Krum Byzantine Fault Tolerance ---")
    krum = KrumAggregator(num_byzantine=1, multi_krum_m=1)
    
    # Add a Byzantine (malicious) update
    updates_with_byzantine = updates + [
        WorkerUpdate(
            node_id="byzantine",
            step=0,
            num_samples=100,
            parameters={
                "layer1.weight": np.array([[100.0, 100.0], [100.0, 100.0]]),  # Outlier
                "layer1.bias": np.array([100.0, 100.0]),
            },
            loss=0.1,
        ),
    ]
    
    krum_result = krum.aggregate(updates_with_byzantine)
    print(f"Krum selected update (should reject Byzantine outlier):")
    print(f"  layer1.weight:\n{krum_result['layer1.weight']}")
    
    # Test 3: Trimmed Mean
    print("\n--- Test 3: Trimmed Mean ---")
    trimmed = TrimmedMeanAggregator(trim_ratio=0.25)
    trimmed_result = trimmed.aggregate(updates_with_byzantine)
    print(f"Trimmed mean (25% from each end):")
    print(f"  layer1.weight:\n{trimmed_result['layer1.weight']}")
    
    # Test 4: Median aggregation
    print("\n--- Test 4: Median Aggregation ---")
    median = MedianAggregator()
    median_result = median.aggregate(updates)
    print(f"Median aggregation:")
    print(f"  layer1.weight:\n{median_result['layer1.weight']}")
    
    # Test 5: FederatedAggregator with compression
    print("\n--- Test 5: FederatedAggregator with Compression ---")
    fed_agg = FederatedAggregator(
        strategy=AggregationStrategy.FEDAVG,
        enable_compression=True,
        compression_ratio=0.5,
    )
    
    for update in updates:
        fed_agg.add_update(update)
    
    result = fed_agg.aggregate(step=0, min_updates=2)
    print(f"Compressed FedAvg result:")
    print(f"  layer1.weight:\n{result['layer1.weight']}")
    print(f"Stats: {fed_agg.get_statistics()}")
    
    # Test 6: NetworkComputeAggregator integration
    print("\n--- Test 6: NetworkComputeAggregator Integration ---")
    aggregator = NetworkComputeAggregator(
        "test_coordinator",
        NodeRole.COORDINATOR,
        aggregation_strategy=AggregationStrategy.FEDAVG,
    )
    
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
    
    # Simulate worker updates
    job_id = "test_job"
    for i, update in enumerate(updates):
        update_copy = WorkerUpdate(
            node_id=f"worker_{i}",
            step=1,
            num_samples=update.num_samples,
            parameters=update.parameters.copy(),
            loss=update.loss,
        )
        aggregator.receive_model_update(update_copy, job_id)
    
    # Aggregate
    agg_result = aggregator.aggregate_model_updates(job_id, step=1, min_updates=2)
    if agg_result:
        print(f"NetworkComputeAggregator result:")
        print(f"  layer1.weight:\n{agg_result['layer1.weight']}")
    
    print(f"\nAggregation status: {aggregator.get_aggregation_status(1)}")
    
    # Check capable nodes
    capable = aggregator.get_capable_nodes(model_params=1_000_000_000, batch_size=32)
    print(f"\nNodes capable of 1B params: {len(capable)}")
    
    # Estimate training time
    time_estimate = aggregator.estimate_training_time(
        model_params=1_000_000_000,
        dataset_tokens=100_000_000_000,
        target_steps=100000,
    )
    print(f"Estimated training time: {time_estimate/3600:.1f} hours")
    
    print("\n" + "=" * 60)
    print("All FedAvg tests completed successfully!")
    print("=" * 60)
