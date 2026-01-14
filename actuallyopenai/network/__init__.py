"""
ActuallyOpenAI Network Package
==============================
Decentralized infrastructure for the people's AI.

No central servers. No single point of failure.
The network IS the AI.
"""

from .p2p_node import P2PNode, Peer, Message, MessageType
from .ipfs_storage import IPFSModelStorage, ModelRegistry
from .api_mesh import DecentralizedAPIMesh, LoadBalancer

# Frontier scaling network components
try:
    from .compute_aggregator import (
        NetworkComputeAggregator,
        ComputeCapability,
        TrainingJob,
        NodeRole,
        # FedAvg and aggregation strategies
        AggregationStrategy,
        FederatedAggregator,
        FedAvgAggregator,
        FedProxAggregator,
        KrumAggregator,
        TrimmedMeanAggregator,
        MedianAggregator,
        WorkerUpdate,
        torch_to_numpy_dict,
        numpy_to_torch_dict,
    )
except ImportError:
    NetworkComputeAggregator = None
    ComputeCapability = None
    TrainingJob = None
    NodeRole = None
    AggregationStrategy = None
    FederatedAggregator = None
    FedAvgAggregator = None
    FedProxAggregator = None
    KrumAggregator = None
    TrimmedMeanAggregator = None
    MedianAggregator = None
    WorkerUpdate = None
    torch_to_numpy_dict = None
    numpy_to_torch_dict = None

__all__ = [
    "P2PNode",
    "Peer", 
    "Message",
    "MessageType",
    "IPFSModelStorage",
    "ModelRegistry",
    "DecentralizedAPIMesh",
    "LoadBalancer",
    # Frontier scaling
    "NetworkComputeAggregator",
    "ComputeCapability",
    "TrainingJob",
    "NodeRole",
    # FedAvg and aggregation
    "AggregationStrategy",
    "FederatedAggregator",
    "FedAvgAggregator",
    "FedProxAggregator",
    "KrumAggregator",
    "TrimmedMeanAggregator",
    "MedianAggregator",
    "WorkerUpdate",
    "torch_to_numpy_dict",
    "numpy_to_torch_dict",
]
