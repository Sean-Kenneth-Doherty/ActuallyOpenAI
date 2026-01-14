"""
High-Quality Data Pipeline for Pre-Training
============================================
You can't train a good model without good data.

Data Sources:
1. Public datasets (C4, The Pile, RedPajama, etc.)
2. Quality-filtered web crawls
3. Curated knowledge sources
4. Code repositories
5. Scientific papers
6. Books and documentation

Quality Filters:
- Perplexity filtering (remove low-quality text)
- Deduplication (exact and near-duplicate)
- Language detection
- Content filtering (safety, quality)
- Length filtering
"""

import os
import json
import hashlib
import asyncio
import aiohttp
import aiofiles
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterator, Any, AsyncIterator
from pathlib import Path
import logging
import random
import re
from collections import Counter

logger = logging.getLogger("AOAI-Data")


@dataclass
class DataSource:
    """Configuration for a data source"""
    name: str
    source_type: str  # "huggingface", "url", "local", "api"
    path: str
    weight: float = 1.0  # Sampling weight
    quality_score: float = 1.0  # Estimated quality
    
    # Optional filters
    min_length: int = 100
    max_length: int = 100000
    language: str = "en"


@dataclass
class DataSample:
    """A single training sample"""
    text: str
    source: str
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def token_estimate(self) -> int:
        """Rough token count (chars / 4)"""
        return len(self.text) // 4


class QualityFilter:
    """
    Filters data for quality.
    
    Bad data leads to bad models. This is critical.
    """
    
    # Patterns indicating low quality
    LOW_QUALITY_PATTERNS = [
        r'click here',
        r'subscribe now',
        r'buy now',
        r'limited time offer',
        r'[A-Z]{20,}',  # Long caps
        r'(.)\1{10,}',  # Repeated characters
        r'(.)( \1){10,}',  # Repeated words
        r'https?://\S{100,}',  # Very long URLs
        r'\b\d{10,}\b',  # Long numbers
    ]
    
    # Patterns indicating quality content
    QUALITY_INDICATORS = [
        r'\b(however|therefore|furthermore|moreover)\b',
        r'\b(according to|research shows|studies indicate)\b',
        r'\b(in conclusion|to summarize|in summary)\b',
        r'[.!?][\s]+[A-Z]',  # Proper sentence structure
    ]
    
    def __init__(self):
        self.low_quality_re = [re.compile(p, re.IGNORECASE) for p in self.LOW_QUALITY_PATTERNS]
        self.quality_re = [re.compile(p, re.IGNORECASE) for p in self.QUALITY_INDICATORS]
    
    def score(self, text: str) -> float:
        """Score text quality from 0 to 1"""
        if len(text) < 50:
            return 0.0
        
        score = 0.5  # Start neutral
        
        # Penalize low-quality patterns
        for pattern in self.low_quality_re:
            if pattern.search(text):
                score -= 0.1
        
        # Reward quality indicators
        for pattern in self.quality_re:
            if pattern.search(text):
                score += 0.1
        
        # Check sentence structure
        sentences = text.split('.')
        if len(sentences) > 3:
            avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
            if 10 < avg_sentence_len < 30:
                score += 0.1
        
        # Check vocabulary diversity
        words = text.lower().split()
        if len(words) > 50:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio > 0.5:
                score += 0.1
        
        # Check for excessive special characters
        special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_ratio > 0.3:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def filter(self, text: str, min_score: float = 0.3) -> Optional[str]:
        """Filter text, returning None if below threshold"""
        score = self.score(text)
        if score < min_score:
            return None
        return text


class Deduplicator:
    """
    Remove duplicate and near-duplicate content.
    
    Duplicates waste compute and can cause memorization.
    """
    
    def __init__(self, num_hashes: int = 128, bands: int = 16):
        self.num_hashes = num_hashes
        self.bands = bands
        self.rows_per_band = num_hashes // bands
        
        self.seen_hashes: set = set()
        self.seen_minhashes: Dict[int, set] = {i: set() for i in range(bands)}
    
    def _get_shingles(self, text: str, k: int = 5) -> set:
        """Get k-shingles from text"""
        words = text.lower().split()
        if len(words) < k:
            return set()
        return set(tuple(words[i:i+k]) for i in range(len(words) - k + 1))
    
    def _minhash(self, shingles: set) -> List[int]:
        """Compute MinHash signature"""
        if not shingles:
            return [0] * self.num_hashes
        
        signature = []
        for i in range(self.num_hashes):
            min_hash = float('inf')
            for shingle in shingles:
                h = hash((shingle, i)) & 0xFFFFFFFF
                min_hash = min(min_hash, h)
            signature.append(min_hash)
        
        return signature
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate"""
        # Exact duplicate check
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.seen_hashes:
            return True
        
        # Near-duplicate check using LSH
        shingles = self._get_shingles(text)
        if not shingles:
            return False
        
        signature = self._minhash(shingles)
        
        # Check each band
        for band_idx in range(self.bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_hash = hash(tuple(signature[start:end]))
            
            if band_hash in self.seen_minhashes[band_idx]:
                return True
        
        # Not a duplicate - add to index
        self.seen_hashes.add(text_hash)
        for band_idx in range(self.bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_hash = hash(tuple(signature[start:end]))
            self.seen_minhashes[band_idx].add(band_hash)
        
        return False


class DataMixer:
    """
    Mix data from multiple sources with configurable weights.
    
    The right data mix is crucial for good performance.
    """
    
    # Recommended data mix for general LLM
    DEFAULT_MIX = {
        "web": 0.50,      # Web crawl data
        "books": 0.15,    # Books and literature
        "code": 0.15,     # Code repositories
        "science": 0.10,  # Scientific papers
        "conversation": 0.05,  # Dialog data
        "knowledge": 0.05,     # Wikipedia, encyclopedias
    }
    
    def __init__(self, data_mix: Optional[Dict[str, float]] = None):
        self.data_mix = data_mix or self.DEFAULT_MIX
        self.source_iterators: Dict[str, Iterator] = {}
        self.source_counts: Counter = Counter()
    
    def add_source(self, category: str, iterator: Iterator[DataSample]):
        """Add a data source"""
        self.source_iterators[category] = iterator
    
    def sample(self) -> Optional[DataSample]:
        """Sample from mixed sources according to weights"""
        # Calculate target proportions
        total_samples = sum(self.source_counts.values()) or 1
        
        # Find most underrepresented source
        deficits = {}
        for category, target_weight in self.data_mix.items():
            if category not in self.source_iterators:
                continue
            
            current_ratio = self.source_counts[category] / total_samples
            deficit = target_weight - current_ratio
            deficits[category] = deficit
        
        if not deficits:
            return None
        
        # Sample from most underrepresented source
        category = max(deficits, key=deficits.get)
        
        try:
            sample = next(self.source_iterators[category])
            self.source_counts[category] += 1
            return sample
        except StopIteration:
            del self.source_iterators[category]
            return self.sample() if self.source_iterators else None


class PretrainingDataPipeline:
    """
    Complete data pipeline for pre-training.
    
    1. Load from sources
    2. Filter for quality
    3. Deduplicate
    4. Mix sources
    5. Tokenize
    6. Batch
    """
    
    def __init__(
        self,
        data_dir: str = "./pretraining_data",
        cache_dir: str = "./data_cache",
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.quality_filter = QualityFilter()
        self.deduplicator = Deduplicator()
        self.mixer = DataMixer()
        
        # Stats
        self.stats = {
            "total_loaded": 0,
            "filtered_quality": 0,
            "filtered_duplicate": 0,
            "total_tokens": 0,
        }
    
    async def download_dataset(self, source: DataSource) -> AsyncIterator[DataSample]:
        """Download and yield samples from a data source"""
        logger.info(f"Loading data from {source.name}...")
        
        if source.source_type == "huggingface":
            async for sample in self._load_huggingface(source):
                yield sample
        
        elif source.source_type == "local":
            async for sample in self._load_local(source):
                yield sample
        
        elif source.source_type == "url":
            async for sample in self._load_url(source):
                yield sample
    
    async def _load_huggingface(self, source: DataSource) -> AsyncIterator[DataSample]:
        """Load from Hugging Face datasets"""
        try:
            from datasets import load_dataset
            
            dataset = load_dataset(source.path, streaming=True, split="train")
            
            for item in dataset:
                text = item.get("text", item.get("content", ""))
                if text:
                    yield DataSample(
                        text=text,
                        source=source.name,
                        quality_score=source.quality_score,
                    )
        except ImportError:
            logger.warning("datasets library not available, skipping HF source")
    
    async def _load_local(self, source: DataSource) -> AsyncIterator[DataSample]:
        """Load from local files"""
        path = Path(source.path)
        
        if path.is_file():
            files = [path]
        else:
            files = list(path.glob("**/*.txt")) + list(path.glob("**/*.jsonl"))
        
        for file_path in files:
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                if file_path.suffix == '.jsonl':
                    async for line in f:
                        try:
                            data = json.loads(line)
                            text = data.get("text", data.get("content", ""))
                            if text:
                                yield DataSample(
                                    text=text,
                                    source=source.name,
                                    quality_score=source.quality_score,
                                    metadata=data.get("metadata", {}),
                                )
                        except json.JSONDecodeError:
                            continue
                else:
                    content = await f.read()
                    yield DataSample(
                        text=content,
                        source=source.name,
                        quality_score=source.quality_score,
                    )
    
    async def _load_url(self, source: DataSource) -> AsyncIterator[DataSample]:
        """Load from URL"""
        async with aiohttp.ClientSession() as session:
            async with session.get(source.path) as response:
                if response.status == 200:
                    content = await response.text()
                    yield DataSample(
                        text=content,
                        source=source.name,
                        quality_score=source.quality_score,
                    )
    
    def process_sample(self, sample: DataSample) -> Optional[DataSample]:
        """Process a single sample through the pipeline"""
        self.stats["total_loaded"] += 1
        
        # Length filter
        if len(sample.text) < 100 or len(sample.text) > 100000:
            return None
        
        # Quality filter
        filtered_text = self.quality_filter.filter(sample.text)
        if filtered_text is None:
            self.stats["filtered_quality"] += 1
            return None
        
        # Deduplication
        if self.deduplicator.is_duplicate(filtered_text):
            self.stats["filtered_duplicate"] += 1
            return None
        
        # Update quality score
        sample.text = filtered_text
        sample.quality_score *= self.quality_filter.score(filtered_text)
        
        self.stats["total_tokens"] += sample.token_estimate
        
        return sample
    
    async def create_batches(
        self,
        sources: List[DataSource],
        batch_size: int = 1024,
        sequence_length: int = 2048,
    ) -> AsyncIterator[List[str]]:
        """Create batches of text for training"""
        
        # Start loading from all sources
        async def load_all():
            for source in sources:
                async for sample in self.download_dataset(source):
                    processed = self.process_sample(sample)
                    if processed:
                        yield processed.text
        
        # Batch texts
        batch = []
        async for text in load_all():
            batch.append(text)
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        if batch:
            yield batch
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            **self.stats,
            "filter_rate_quality": self.stats["filtered_quality"] / max(self.stats["total_loaded"], 1),
            "filter_rate_duplicate": self.stats["filtered_duplicate"] / max(self.stats["total_loaded"], 1),
        }


# Public datasets for pre-training
PUBLIC_DATASETS = [
    DataSource(
        name="redpajama",
        source_type="huggingface",
        path="togethercomputer/RedPajama-Data-1T-Sample",
        weight=1.0,
        quality_score=0.9,
    ),
    DataSource(
        name="c4",
        source_type="huggingface", 
        path="c4",
        weight=0.8,
        quality_score=0.8,
    ),
    DataSource(
        name="wikipedia",
        source_type="huggingface",
        path="wikipedia",
        weight=0.5,
        quality_score=1.0,
    ),
    DataSource(
        name="code",
        source_type="huggingface",
        path="codeparrot/github-code",
        weight=0.3,
        quality_score=0.9,
    ),
]


if __name__ == "__main__":
    # Test quality filter
    qf = QualityFilter()
    
    good_text = """
    Machine learning is a subset of artificial intelligence that enables systems 
    to learn and improve from experience. However, the field has evolved significantly 
    since its inception. Furthermore, recent advances in deep learning have revolutionized 
    many applications. In conclusion, machine learning continues to transform industries.
    """
    
    bad_text = "CLICK HERE NOW!!! BUY BUY BUY $$$$$ limited time offer click subscribe"
    
    print(f"Good text score: {qf.score(good_text):.2f}")
    print(f"Bad text score: {qf.score(bad_text):.2f}")
    
    # Test deduplicator
    dd = Deduplicator()
    
    text1 = "The quick brown fox jumps over the lazy dog. This is a test sentence."
    text2 = "The quick brown fox jumps over the lazy dog. This is a test sentence."
    text3 = "A completely different sentence about something else entirely."
    
    print(f"\nText1 duplicate: {dd.is_duplicate(text1)}")  # False (first time)
    print(f"Text2 duplicate: {dd.is_duplicate(text2)}")  # True (exact duplicate)
    print(f"Text3 duplicate: {dd.is_duplicate(text3)}")  # False (different)
