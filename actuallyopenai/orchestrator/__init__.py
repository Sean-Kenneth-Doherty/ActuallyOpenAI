"""
Orchestrator module - Central coordination for ActuallyOpenAI distributed training.

Includes:
- Main orchestration for distributed training
- Scaling orchestrator for progressive model scaling
"""

# Frontier scaling orchestrator
try:
    from .scaling_orchestrator import (
        ScalingOrchestrator,
        ScalePhase,
        ScalePhaseConfig,
        ScaleProgress,
        AutoScalingLoop,
    )
except ImportError:
    ScalingOrchestrator = None
    ScalePhase = None
    ScalePhaseConfig = None
    ScaleProgress = None
    AutoScalingLoop = None

__all__ = [
    "ScalingOrchestrator",
    "ScalePhase",
    "ScalePhaseConfig",
    "ScaleProgress",
    "AutoScalingLoop",
]
