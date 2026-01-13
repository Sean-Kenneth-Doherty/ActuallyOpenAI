"""
ActuallyOpenAI Mining Package
=============================
Adaptive mining system that splits compute between inference and training.

High demand → More inference
Low demand → More training

The AI grows smarter during quiet periods.
"""

from .adaptive_miner import AdaptiveMiner, MinerStats, MinerMode, DemandTracker

__all__ = [
    "AdaptiveMiner",
    "MinerStats", 
    "MinerMode",
    "DemandTracker"
]
