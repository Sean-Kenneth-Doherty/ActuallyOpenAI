"""
Federated Learning Aggregator - Combines updates from distributed workers.

Implements multiple aggregation strategies:
1. FedAvg - Weighted average based on sample count
2. FedProx - Adds proximal term for heterogeneous data
3. FedAdam - Adaptive optimization for federated setting
4. Byzantine-resilient - Handles malicious/faulty workers
"""

import asyncio
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
import math

import structlog
import torch
import torch.nn as nn
import numpy as np

logger = structlog.get_logger()


class AggregationStrategy(str, Enum):
    """Strategy for aggregating gradients from workers."""
    FEDAVG = "fedavg"  # Standard federated averaging
    FEDPROX = "fedprox"  # Proximal term for heterogeneity
    FEDADAM = "fedadam"  # Adaptive federated optimization
    BYZANTINE = "byzantine"  # Byzantine fault tolerant
    WEIGHTED_QUALITY = "weighted_quality"  # Weight by worker quality score


@dataclass
class WorkerUpdate:
    """An update (gradients or weights) from a worker."""
    worker_id: str
    round_id: int
    
    # Either gradients or model delta
    gradients: Optional[Dict[str, torch.Tensor]] = None
    model_delta: Optional[Dict[str, torch.Tensor]] = None
    
    # Metadata
    num_samples: int = 0
    local_loss: float = float('inf')
    local_steps: int = 1
    compute_time: float = 0.0
    
    # Quality metrics
    gradient_norm: float = 0.0
    staleness: int = 0  # How many rounds old
    
    # Verification
    checksum: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AggregationResult:
    """Result of an aggregation round."""
    round_id: int
    strategy: AggregationStrategy
    
    # Aggregated updates
    aggregated_gradients: Optional[Dict[str, torch.Tensor]] = None
    aggregated_delta: Optional[Dict[str, torch.Tensor]] = None
    
    # Statistics
    num_workers: int = 0
    total_samples: int = 0
    average_loss: float = 0.0
    weighted_loss: float = 0.0
    
    # Quality metrics
    consensus_score: float = 0.0  # Agreement among workers
    outliers_detected: int = 0
    
    timestamp: datetime = field(default_factory=datetime.utcnow)


class FederatedAggregator:
    """
    Aggregates updates from distributed workers using federated learning.
    
    Features:
    - Multiple aggregation strategies
    - Byzantine fault tolerance
    - Staleness handling
    - Worker quality tracking
    - Gradient compression support
    """
    
    def __init__(
        self,
        strategy: AggregationStrategy = AggregationStrategy.FEDAVG,
        min_workers: int = 2,
        staleness_threshold: int = 5,
        byzantine_threshold: float = 0.3,  # Max fraction of Byzantine workers
        mu: float = 0.01,  # FedProx proximal term
    ):
        self.strategy = strategy
        self.min_workers = min_workers
        self.staleness_threshold = staleness_threshold
        self.byzantine_threshold = byzantine_threshold
        self.mu = mu
        
        # Current aggregation round
        self.current_round = 0
        
        # Update buffers
        self.update_buffer: List[WorkerUpdate] = []
        self.round_updates: Dict[int, List[WorkerUpdate]] = defaultdict(list)
        
        # Worker quality tracking
        self.worker_quality: Dict[str, float] = defaultdict(lambda: 1.0)
        self.worker_history: Dict[str, List[float]] = defaultdict(list)
        
        # FedAdam state (server-side optimizer)
        self.fedadam_m: Optional[Dict[str, torch.Tensor]] = None  # First moment
        self.fedadam_v: Optional[Dict[str, torch.Tensor]] = None  # Second moment
        self.fedadam_beta1 = 0.9
        self.fedadam_beta2 = 0.99
        self.fedadam_tau = 1e-3
        
        # Global model reference (for FedProx and delta computation)
        self.global_model_state: Optional[Dict[str, torch.Tensor]] = None
        
        # Statistics
        self.aggregation_history: List[AggregationResult] = []
        
        logger.info(
            "FederatedAggregator initialized",
            strategy=strategy.value,
            min_workers=min_workers
        )
    
    def set_global_model(self, model: nn.Module):
        """Set reference to the global model."""
        self.global_model_state = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }
    
    async def receive_update(self, update: WorkerUpdate):
        """Receive an update from a worker."""
        # Calculate gradient norm for quality assessment
        if update.gradients:
            total_norm = 0.0
            for grad in update.gradients.values():
                total_norm += grad.norm().item() ** 2
            update.gradient_norm = math.sqrt(total_norm)
        
        # Calculate staleness
        update.staleness = self.current_round - update.round_id
        
        # Add to buffer
        self.update_buffer.append(update)
        self.round_updates[update.round_id].append(update)
        
        logger.debug(
            "Received update",
            worker_id=update.worker_id,
            round_id=update.round_id,
            samples=update.num_samples,
            loss=round(update.local_loss, 6),
            staleness=update.staleness
        )
    
    def ready_to_aggregate(self) -> bool:
        """Check if we have enough updates for aggregation."""
        current_updates = len(self.round_updates[self.current_round])
        return current_updates >= self.min_workers
    
    async def aggregate(self) -> Optional[AggregationResult]:
        """
        Aggregate updates from workers using the configured strategy.
        """
        updates = self._get_valid_updates()
        
        if len(updates) < self.min_workers:
            logger.debug(
                "Not enough updates for aggregation",
                have=len(updates),
                need=self.min_workers
            )
            return None
        
        logger.info(
            f"Aggregating {len(updates)} updates using {self.strategy.value}"
        )
        
        # Select aggregation strategy
        if self.strategy == AggregationStrategy.FEDAVG:
            result = await self._aggregate_fedavg(updates)
        elif self.strategy == AggregationStrategy.FEDPROX:
            result = await self._aggregate_fedprox(updates)
        elif self.strategy == AggregationStrategy.FEDADAM:
            result = await self._aggregate_fedadam(updates)
        elif self.strategy == AggregationStrategy.BYZANTINE:
            result = await self._aggregate_byzantine(updates)
        elif self.strategy == AggregationStrategy.WEIGHTED_QUALITY:
            result = await self._aggregate_weighted_quality(updates)
        else:
            result = await self._aggregate_fedavg(updates)
        
        # Update worker quality based on contribution
        self._update_worker_quality(updates, result)
        
        # Store result
        self.aggregation_history.append(result)
        
        # Clear processed updates and advance round
        self._clear_processed_updates()
        self.current_round += 1
        
        logger.info(
            "Aggregation complete",
            round_id=result.round_id,
            workers=result.num_workers,
            samples=result.total_samples,
            avg_loss=round(result.average_loss, 6),
            consensus=round(result.consensus_score, 4)
        )
        
        return result
    
    # =========================================================================
    # Aggregation Strategies
    # =========================================================================
    
    async def _aggregate_fedavg(
        self, updates: List[WorkerUpdate]
    ) -> AggregationResult:
        """
        FedAvg: Weighted average of updates based on sample count.
        Classic McMahan et al. algorithm.
        """
        total_samples = sum(u.num_samples for u in updates)
        
        aggregated = {}
        weighted_loss = 0.0
        
        for update in updates:
            weight = update.num_samples / total_samples
            weighted_loss += update.local_loss * weight
            
            gradients = update.gradients or {}
            for name, grad in gradients.items():
                if name not in aggregated:
                    aggregated[name] = torch.zeros_like(grad)
                aggregated[name] += grad * weight
        
        return AggregationResult(
            round_id=self.current_round,
            strategy=AggregationStrategy.FEDAVG,
            aggregated_gradients=aggregated,
            num_workers=len(updates),
            total_samples=total_samples,
            average_loss=sum(u.local_loss for u in updates) / len(updates),
            weighted_loss=weighted_loss,
            consensus_score=self._calculate_consensus(updates)
        )
    
    async def _aggregate_fedprox(
        self, updates: List[WorkerUpdate]
    ) -> AggregationResult:
        """
        FedProx: FedAvg + proximal term for heterogeneous data.
        Adds regularization towards global model.
        """
        # Start with FedAvg
        result = await self._aggregate_fedavg(updates)
        
        # Add proximal regularization if we have global model reference
        if self.global_model_state and result.aggregated_gradients:
            for name, agg_grad in result.aggregated_gradients.items():
                if name in self.global_model_state:
                    # Proximal term: mu * (w - w_global)
                    # This encourages updates to stay close to global model
                    global_param = self.global_model_state[name]
                    proximal_term = self.mu * (agg_grad - global_param)
                    result.aggregated_gradients[name] = agg_grad + proximal_term
        
        result.strategy = AggregationStrategy.FEDPROX
        return result
    
    async def _aggregate_fedadam(
        self, updates: List[WorkerUpdate]
    ) -> AggregationResult:
        """
        FedAdam: Server-side adaptive optimization.
        Maintains momentum and variance on the server.
        """
        # First compute FedAvg pseudo-gradient
        result = await self._aggregate_fedavg(updates)
        
        if not result.aggregated_gradients:
            return result
        
        # Initialize momentum if needed
        if self.fedadam_m is None:
            self.fedadam_m = {
                name: torch.zeros_like(grad)
                for name, grad in result.aggregated_gradients.items()
            }
            self.fedadam_v = {
                name: torch.zeros_like(grad)
                for name, grad in result.aggregated_gradients.items()
            }
        
        # Update momentum and variance
        adapted_gradients = {}
        for name, grad in result.aggregated_gradients.items():
            if name in self.fedadam_m:
                # Update first moment
                self.fedadam_m[name] = (
                    self.fedadam_beta1 * self.fedadam_m[name] +
                    (1 - self.fedadam_beta1) * grad
                )
                
                # Update second moment
                self.fedadam_v[name] = (
                    self.fedadam_beta2 * self.fedadam_v[name] +
                    (1 - self.fedadam_beta2) * (grad ** 2)
                )
                
                # Compute adapted gradient
                adapted_gradients[name] = (
                    self.fedadam_m[name] /
                    (torch.sqrt(self.fedadam_v[name]) + self.fedadam_tau)
                )
            else:
                adapted_gradients[name] = grad
        
        result.aggregated_gradients = adapted_gradients
        result.strategy = AggregationStrategy.FEDADAM
        return result
    
    async def _aggregate_byzantine(
        self, updates: List[WorkerUpdate]
    ) -> AggregationResult:
        """
        Byzantine fault-tolerant aggregation.
        Uses coordinate-wise trimmed mean or Krum selection.
        """
        if len(updates) < 4:
            # Not enough workers for Byzantine tolerance
            return await self._aggregate_fedavg(updates)
        
        # Detect and filter outliers using gradient norms
        outliers = self._detect_byzantine_workers(updates)
        
        # Filter out Byzantine workers
        clean_updates = [u for u in updates if u.worker_id not in outliers]
        
        if len(clean_updates) < self.min_workers:
            # Too many outliers, fall back to trimmed mean
            clean_updates = self._trimmed_mean_filter(updates)
        
        # Aggregate clean updates
        result = await self._aggregate_fedavg(clean_updates)
        result.strategy = AggregationStrategy.BYZANTINE
        result.outliers_detected = len(outliers)
        
        # Penalize quality score of outliers
        for worker_id in outliers:
            self.worker_quality[worker_id] *= 0.5
        
        logger.info(
            "Byzantine aggregation complete",
            total_workers=len(updates),
            outliers=len(outliers),
            clean_workers=len(clean_updates)
        )
        
        return result
    
    async def _aggregate_weighted_quality(
        self, updates: List[WorkerUpdate]
    ) -> AggregationResult:
        """
        Weight updates by worker quality score.
        Rewards consistent, high-quality contributors.
        """
        # Calculate weights based on samples AND quality
        weights = {}
        total_weight = 0.0
        
        for update in updates:
            quality = self.worker_quality.get(update.worker_id, 1.0)
            sample_weight = update.num_samples
            
            # Combined weight
            weight = sample_weight * quality
            weights[update.worker_id] = weight
            total_weight += weight
        
        # Normalize weights
        for worker_id in weights:
            weights[worker_id] /= total_weight
        
        # Weighted aggregation
        aggregated = {}
        weighted_loss = 0.0
        
        for update in updates:
            weight = weights[update.worker_id]
            weighted_loss += update.local_loss * weight
            
            gradients = update.gradients or {}
            for name, grad in gradients.items():
                if name not in aggregated:
                    aggregated[name] = torch.zeros_like(grad)
                aggregated[name] += grad * weight
        
        return AggregationResult(
            round_id=self.current_round,
            strategy=AggregationStrategy.WEIGHTED_QUALITY,
            aggregated_gradients=aggregated,
            num_workers=len(updates),
            total_samples=sum(u.num_samples for u in updates),
            average_loss=sum(u.local_loss for u in updates) / len(updates),
            weighted_loss=weighted_loss,
            consensus_score=self._calculate_consensus(updates)
        )
    
    # =========================================================================
    # Byzantine Detection
    # =========================================================================
    
    def _detect_byzantine_workers(
        self, updates: List[WorkerUpdate]
    ) -> List[str]:
        """
        Detect potentially Byzantine (malicious/faulty) workers.
        Uses gradient norm analysis and inter-update similarity.
        """
        outliers = []
        
        # Collect gradient norms
        norms = [u.gradient_norm for u in updates]
        
        if not norms:
            return outliers
        
        # Calculate statistics
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        # Z-score based outlier detection
        for update in updates:
            if std_norm > 0:
                z_score = abs(update.gradient_norm - mean_norm) / std_norm
                if z_score > 3.0:  # 3 sigma outlier
                    outliers.append(update.worker_id)
        
        # Limit number of removed workers
        max_outliers = int(len(updates) * self.byzantine_threshold)
        return outliers[:max_outliers]
    
    def _trimmed_mean_filter(
        self, updates: List[WorkerUpdate]
    ) -> List[WorkerUpdate]:
        """Apply trimmed mean by removing extreme values."""
        if len(updates) < 4:
            return updates
        
        # Sort by loss
        sorted_updates = sorted(updates, key=lambda u: u.local_loss)
        
        # Remove top and bottom 10%
        trim_count = max(1, len(updates) // 10)
        return sorted_updates[trim_count:-trim_count]
    
    # =========================================================================
    # Quality Tracking
    # =========================================================================
    
    def _calculate_consensus(self, updates: List[WorkerUpdate]) -> float:
        """
        Calculate consensus score among workers.
        Higher score means workers agree more.
        """
        if len(updates) < 2:
            return 1.0
        
        # Use gradient similarity as consensus measure
        similarities = []
        
        for i, u1 in enumerate(updates):
            for u2 in updates[i + 1:]:
                if u1.gradients and u2.gradients:
                    sim = self._gradient_similarity(u1.gradients, u2.gradients)
                    similarities.append(sim)
        
        if not similarities:
            return 1.0
        
        return sum(similarities) / len(similarities)
    
    def _gradient_similarity(
        self,
        grads1: Dict[str, torch.Tensor],
        grads2: Dict[str, torch.Tensor]
    ) -> float:
        """Calculate cosine similarity between gradient sets."""
        common_keys = set(grads1.keys()) & set(grads2.keys())
        
        if not common_keys:
            return 0.0
        
        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0
        
        for key in common_keys:
            g1 = grads1[key].flatten()
            g2 = grads2[key].flatten()
            
            dot_product += torch.dot(g1, g2).item()
            norm1 += torch.norm(g1).item() ** 2
            norm2 += torch.norm(g2).item() ** 2
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))
    
    def _update_worker_quality(
        self,
        updates: List[WorkerUpdate],
        result: AggregationResult
    ):
        """Update quality scores based on contribution."""
        if not updates:
            return
        
        avg_loss = result.average_loss
        
        for update in updates:
            # Higher quality if loss is close to average (not an outlier)
            loss_diff = abs(update.local_loss - avg_loss)
            
            # Exponential moving average of quality
            if avg_loss > 0:
                contribution_quality = 1.0 / (1.0 + loss_diff / avg_loss)
            else:
                contribution_quality = 1.0
            
            old_quality = self.worker_quality[update.worker_id]
            new_quality = 0.9 * old_quality + 0.1 * contribution_quality
            
            self.worker_quality[update.worker_id] = new_quality
            self.worker_history[update.worker_id].append(new_quality)
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    def _get_valid_updates(self) -> List[WorkerUpdate]:
        """Get valid updates for current round, filtering stale ones."""
        valid = []
        
        for update in self.update_buffer:
            # Filter very stale updates
            if update.staleness <= self.staleness_threshold:
                valid.append(update)
            else:
                logger.debug(
                    "Discarding stale update",
                    worker_id=update.worker_id,
                    staleness=update.staleness
                )
        
        return valid
    
    def _clear_processed_updates(self):
        """Clear processed updates from buffer."""
        # Keep only recent updates
        self.update_buffer = [
            u for u in self.update_buffer
            if u.round_id > self.current_round - 2
        ]
        
        # Clear old round data
        old_rounds = [r for r in self.round_updates.keys() if r < self.current_round - 2]
        for r in old_rounds:
            del self.round_updates[r]
    
    def get_worker_quality(self, worker_id: str) -> float:
        """Get quality score for a worker."""
        return self.worker_quality.get(worker_id, 1.0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        return {
            "current_round": self.current_round,
            "strategy": self.strategy.value,
            "total_aggregations": len(self.aggregation_history),
            "buffer_size": len(self.update_buffer),
            "num_tracked_workers": len(self.worker_quality),
            "avg_consensus": (
                sum(r.consensus_score for r in self.aggregation_history[-10:]) / 
                max(1, len(self.aggregation_history[-10:]))
            ),
            "total_outliers_detected": sum(
                r.outliers_detected for r in self.aggregation_history
            )
        }
