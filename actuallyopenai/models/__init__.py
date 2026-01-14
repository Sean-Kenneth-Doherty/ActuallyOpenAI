"""
ActuallyOpenAI Models module.

Contains the base model architectures for distributed training
and scalable models for frontier-scale training.
"""

from actuallyopenai.models.base_model import (
    AOAIModel,
    ModelConfig,
    create_model,
)

# Scalable model for progressive scaling
try:
    from actuallyopenai.models.scalable_model import (
        ScalableAOAI,
        ScalableConfig,
    )
except ImportError:
    ScalableAOAI = None
    ScalableConfig = None

__all__ = [
    "AOAIModel",
    "ModelConfig",
    "create_model",
    "ScalableAOAI",
    "ScalableConfig",
]
