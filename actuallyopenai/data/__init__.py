"""
Data module for ActuallyOpenAI.

Fully open and decentralized data pipeline:
- Open source tokenizers (no proprietary dependencies)
- Open datasets (Wikipedia, The Pile, etc.)
- P2P data sharing between workers
"""

from .tokenizer import (
    SimpleByteTokenizer,
    BPETokenizer,
    OpenTokenizer
)

from .data_loader import (
    OPEN_DATASETS,
    HuggingFaceSource,
    IPFSSource,
    LocalFileSource,
    TokenizedDataset,
    DecentralizedDataLoader
)

from .p2p_sharing import (
    DataChunk,
    PeerInfo,
    ChunkStore,
    P2PDataSharing,
    DistributedDataset
)

# Frontier scaling data components
try:
    from .pretraining_pipeline import (
        PretrainingDataPipeline,
        QualityFilter,
        Deduplicator,
        DataMixer,
    )
except ImportError:
    PretrainingDataPipeline = None
    QualityFilter = None
    Deduplicator = None
    DataMixer = None

try:
    from .bpe_tokenizer import (
        BPETokenizer as AdvancedBPETokenizer,
        ChatFormatter,
    )
except ImportError:
    AdvancedBPETokenizer = None
    ChatFormatter = None

__all__ = [
    # Tokenizers
    "SimpleByteTokenizer",
    "BPETokenizer", 
    "OpenTokenizer",
    
    # Data sources
    "OPEN_DATASETS",
    "HuggingFaceSource",
    "IPFSSource",
    "LocalFileSource",
    "TokenizedDataset",
    "DecentralizedDataLoader",
    
    # P2P sharing
    "DataChunk",
    "PeerInfo",
    "ChunkStore",
    "P2PDataSharing",
    "DistributedDataset",
    
    # Frontier scaling
    "PretrainingDataPipeline",
    "QualityFilter",
    "Deduplicator",
    "DataMixer",
    "AdvancedBPETokenizer",
    "ChatFormatter",
]
