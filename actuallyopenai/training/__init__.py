"""
ActuallyOpenAI Training Module.

The self-assembling, continuously improving AI training system.

This module contains the core components that make the AI truly self-improving:

1. ContinuousTrainer - Never stops training, runs 24/7 as compute is available
2. FederatedAggregator - Aggregates gradients from distributed workers
3. ModelEvolution - Tracks model generations and improvement over time
4. AutoScalingController - Dynamically adjusts training to available compute
5. ImprovementTracker - Benchmarks and tracks improvement metrics
"""

from actuallyopenai.training.continuous_trainer import (
    ContinuousTrainer,
    TrainingState,
    TrainingPhase,
    GradientPacket,
)
from actuallyopenai.training.federated_aggregator import (
    FederatedAggregator,
    AggregationStrategy,
    WorkerUpdate,
    AggregationResult,
)
from actuallyopenai.training.model_evolution import (
    ModelEvolution,
    ModelGeneration,
    EvolutionStrategy,
    EvolutionBranch,
    ModelStatus,
)
from actuallyopenai.training.auto_scaler import (
    AutoScalingController,
    ScalingMode,
    ScalingParameters,
    ResourceTier,
)
from actuallyopenai.training.improvement_tracker import (
    ImprovementTracker,
    BenchmarkSuite,
    BenchmarkResult,
    QualityReport,
)
from actuallyopenai.training.training_orchestrator import TrainingOrchestrator

__all__ = [
    # Core trainer
    "ContinuousTrainer",
    "TrainingState",
    "TrainingPhase",
    "GradientPacket",
    # Federated learning
    "FederatedAggregator",
    "AggregationStrategy",
    "WorkerUpdate",
    "AggregationResult",
    # Model evolution
    "ModelEvolution",
    "ModelGeneration",
    "EvolutionStrategy",
    "EvolutionBranch",
    "ModelStatus",
    # Auto-scaling
    "AutoScalingController",
    "ScalingMode",
    "ScalingParameters",
    "ResourceTier",
    # Benchmarking
    "ImprovementTracker",
    "BenchmarkSuite",
    "BenchmarkResult",
    "QualityReport",
    # Orchestrator
    "TrainingOrchestrator",
]
