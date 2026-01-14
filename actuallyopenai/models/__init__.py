"""
ActuallyOpenAI Models module.

Contains the base model architectures for distributed training,
scalable models for discrete phase scaling, and continuously
growing models for smooth, step-free expansion.
"""

from actuallyopenai.models.base_model import (
    AOAIModel,
    ModelConfig,
    create_model,
)

# Scalable model for discrete phase scaling
try:
    from actuallyopenai.models.scalable_model import (
        ScalableAOAI,
        ScalableConfig,
    )
except ImportError:
    ScalableAOAI = None
    ScalableConfig = None

# Continuous growth model for smooth expansion
try:
    from actuallyopenai.models.continuous_growth import (
        ContinuouslyGrowingAI,
        ContinuousGrowthTrainer,
        GrowthState,
    )
except ImportError:
    ContinuouslyGrowingAI = None
    ContinuousGrowthTrainer = None
    GrowthState = None

__all__ = [
    "AOAIModel",
    "ModelConfig",
    "create_model",
    # Discrete scaling
    "ScalableAOAI",
    "ScalableConfig",
    # Continuous growth
    "ContinuouslyGrowingAI",
    "ContinuousGrowthTrainer",
    "GrowthState",
]
