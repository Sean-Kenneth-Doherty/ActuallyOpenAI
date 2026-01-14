"""
Continuous Improvement Engine
=============================
The engine that makes AOAI truly self-improving.

This ties everything together:
- Monitors model quality
- Triggers training when improvements found
- Manages knowledge accumulation
- Orchestrates the feedback loop

The goal: Every interaction makes the model better.
"""

import asyncio
import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import logging
import random

logger = logging.getLogger("AOAI-ImprovementEngine")


class ImprovementSource(Enum):
    """Sources of improvement data"""
    USER_FEEDBACK = "user_feedback"      # Thumbs up/down, corrections
    INFERENCE_DATA = "inference_data"    # Actual usage patterns
    SYNTHETIC = "synthetic"              # Generated training data
    EXTERNAL = "external"                # Crawled/imported data
    NETWORK = "network"                  # From other nodes


@dataclass
class ImprovementSignal:
    """A signal that the model could be improved"""
    signal_id: str
    source: ImprovementSource
    timestamp: float
    
    # The improvement data
    prompt: str
    current_response: Optional[str] = None
    better_response: Optional[str] = None
    
    # Quality metrics
    quality_score: float = 0.0  # 0-1, how good is the improvement
    confidence: float = 0.0     # 0-1, how confident are we
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImprovementBatch:
    """A batch of improvements ready for training"""
    batch_id: str
    signals: List[ImprovementSignal]
    created_at: float
    
    # Training info
    trained: bool = False
    trained_at: Optional[float] = None
    loss_before: Optional[float] = None
    loss_after: Optional[float] = None
    
    @property
    def size(self) -> int:
        return len(self.signals)
    
    @property
    def avg_quality(self) -> float:
        if not self.signals:
            return 0.0
        return sum(s.quality_score for s in self.signals) / len(self.signals)


class QualityMonitor:
    """
    Monitors model output quality and identifies improvement opportunities.
    """
    
    def __init__(self):
        self.response_history: List[Dict[str, Any]] = []
        self.quality_scores: List[float] = []
        self.improvement_signals: List[ImprovementSignal] = []
        
        # Quality thresholds
        self.min_acceptable_quality = 0.6
        self.excellence_threshold = 0.9
    
    def score_response(
        self,
        prompt: str,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Score a model response for quality"""
        scores = {}
        
        # Length appropriateness (not too short, not too long)
        ideal_length = len(prompt) * 2  # Rough heuristic
        length_ratio = len(response) / ideal_length if ideal_length > 0 else 0
        scores["length"] = min(1.0, 1.0 - abs(1.0 - length_ratio) * 0.5)
        
        # Coherence (basic: starts with capital, ends with punctuation)
        coherent = (
            response and 
            response[0].isupper() and 
            response[-1] in ".!?\"'"
        )
        scores["coherence"] = 1.0 if coherent else 0.5
        
        # Relevance (basic: shares words with prompt)
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words & response_words)
        scores["relevance"] = min(1.0, overlap / max(len(prompt_words), 1) * 2)
        
        # Diversity (not repetitive)
        words = response.split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        scores["diversity"] = unique_ratio
        
        # No obvious errors
        error_patterns = ["i don't know", "as an ai", "i cannot", "error"]
        has_errors = any(p in response.lower() for p in error_patterns)
        scores["no_errors"] = 0.3 if has_errors else 1.0
        
        # Weighted average
        weights = {
            "length": 0.1,
            "coherence": 0.2,
            "relevance": 0.3,
            "diversity": 0.2,
            "no_errors": 0.2,
        }
        
        total_score = sum(scores[k] * weights[k] for k in scores)
        
        return total_score, scores
    
    def record_response(
        self,
        prompt: str,
        response: str,
        user_feedback: Optional[float] = None,
    ) -> ImprovementSignal:
        """Record a response and check if it's an improvement opportunity"""
        quality_score, _ = self.score_response(prompt, response)
        
        # Record history
        self.response_history.append({
            "prompt": prompt,
            "response": response,
            "quality": quality_score,
            "feedback": user_feedback,
            "timestamp": time.time(),
        })
        
        self.quality_scores.append(quality_score)
        
        # Create improvement signal if quality is low
        if quality_score < self.min_acceptable_quality or (
            user_feedback is not None and user_feedback < 0.5
        ):
            signal = ImprovementSignal(
                signal_id=hashlib.sha256(f"{prompt}{time.time()}".encode()).hexdigest()[:16],
                source=ImprovementSource.USER_FEEDBACK if user_feedback else ImprovementSource.INFERENCE_DATA,
                timestamp=time.time(),
                prompt=prompt,
                current_response=response,
                quality_score=quality_score,
                confidence=0.8 if user_feedback else 0.5,
                metadata={"user_feedback": user_feedback},
            )
            self.improvement_signals.append(signal)
            return signal
        
        return None
    
    def get_quality_trend(self, window: int = 100) -> Dict[str, float]:
        """Get quality trend over recent responses"""
        if not self.quality_scores:
            return {"current": 0, "trend": 0, "min": 0, "max": 0}
        
        recent = self.quality_scores[-window:]
        
        if len(recent) < 2:
            return {
                "current": recent[-1] if recent else 0,
                "trend": 0,
                "min": min(recent) if recent else 0,
                "max": max(recent) if recent else 0,
            }
        
        # Calculate trend
        first_half = sum(recent[:len(recent)//2]) / (len(recent)//2)
        second_half = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
        trend = second_half - first_half
        
        return {
            "current": recent[-1],
            "average": sum(recent) / len(recent),
            "trend": trend,
            "min": min(recent),
            "max": max(recent),
        }


class SyntheticDataGenerator:
    """
    Generates synthetic training data for continuous improvement.
    
    Uses the model itself to create training examples,
    then filters for quality.
    """
    
    def __init__(self, model=None):
        self.model = model
        self.templates = self._load_templates()
    
    def _load_templates(self) -> List[Dict[str, str]]:
        """Load prompt templates for synthetic generation"""
        return [
            {
                "category": "instruction_following",
                "template": "Write a {length} explanation of {topic}.",
                "variables": {
                    "length": ["brief", "detailed", "comprehensive"],
                    "topic": ["machine learning", "neural networks", "python programming", 
                             "data science", "software engineering", "algorithms"],
                }
            },
            {
                "category": "question_answering",
                "template": "What is {concept} and why is it important?",
                "variables": {
                    "concept": ["gradient descent", "backpropagation", "attention mechanism",
                               "transformer architecture", "tokenization", "embeddings"],
                }
            },
            {
                "category": "code_generation",
                "template": "Write a Python function that {task}.",
                "variables": {
                    "task": ["sorts a list", "reads a file", "makes an API call",
                            "parses JSON", "handles errors", "processes data"],
                }
            },
            {
                "category": "reasoning",
                "template": "Compare and contrast {thing_a} with {thing_b}.",
                "variables": {
                    "thing_a": ["CNNs", "RNNs", "Transformers", "GPT", "BERT"],
                    "thing_b": ["LSTMs", "GRUs", "attention", "fine-tuning", "pre-training"],
                }
            },
        ]
    
    def generate_prompt(self) -> str:
        """Generate a random training prompt"""
        template = random.choice(self.templates)
        prompt = template["template"]
        
        for var_name, var_options in template["variables"].items():
            value = random.choice(var_options)
            prompt = prompt.replace(f"{{{var_name}}}", value)
        
        return prompt
    
    async def generate_example(self) -> Optional[ImprovementSignal]:
        """Generate a synthetic training example"""
        prompt = self.generate_prompt()
        
        # If we have a model, generate response
        if self.model:
            try:
                response = await self.model.generate(prompt)
                quality_score = 0.7  # Assume synthetic is decent quality
            except Exception as e:
                logger.warning(f"Generation failed: {e}")
                return None
        else:
            response = f"[Placeholder response for: {prompt}]"
            quality_score = 0.5
        
        return ImprovementSignal(
            signal_id=hashlib.sha256(f"synthetic:{prompt}".encode()).hexdigest()[:16],
            source=ImprovementSource.SYNTHETIC,
            timestamp=time.time(),
            prompt=prompt,
            better_response=response,
            quality_score=quality_score,
            confidence=0.6,
            metadata={"synthetic": True},
        )
    
    async def generate_batch(self, size: int = 100) -> List[ImprovementSignal]:
        """Generate a batch of synthetic examples"""
        examples = []
        
        for _ in range(size):
            example = await self.generate_example()
            if example:
                examples.append(example)
        
        return examples


class ContinuousImprovementEngine:
    """
    The main engine that drives continuous model improvement.
    
    Pipeline:
    1. Monitor model outputs for quality
    2. Collect improvement signals
    3. Generate synthetic data
    4. Batch improvements
    5. Train on batches
    6. Verify improvements
    7. Deploy if better
    """
    
    def __init__(
        self,
        model=None,
        trainer=None,
        min_batch_size: int = 100,
        max_batch_age: float = 3600,  # 1 hour
    ):
        self.model = model
        self.trainer = trainer
        self.min_batch_size = min_batch_size
        self.max_batch_age = max_batch_age
        
        # Components
        self.quality_monitor = QualityMonitor()
        self.synthetic_generator = SyntheticDataGenerator(model)
        
        # State
        self.pending_signals: List[ImprovementSignal] = []
        self.pending_batch_start: float = time.time()
        self.trained_batches: List[ImprovementBatch] = []
        
        # Metrics
        self.total_improvements = 0
        self.total_tokens_trained = 0
        self.quality_before = []
        self.quality_after = []
        
        # Running state
        self.running = False
        
        logger.info("Continuous improvement engine initialized")
    
    def add_improvement(self, signal: ImprovementSignal):
        """Add an improvement signal to the pending batch"""
        self.pending_signals.append(signal)
        logger.debug(f"Added improvement signal: {signal.signal_id}")
    
    def record_inference(
        self,
        prompt: str,
        response: str,
        user_feedback: Optional[float] = None,
    ):
        """Record an inference for quality monitoring"""
        signal = self.quality_monitor.record_response(prompt, response, user_feedback)
        if signal:
            self.add_improvement(signal)
    
    def should_train(self) -> Tuple[bool, str]:
        """Check if we should train on pending improvements"""
        if len(self.pending_signals) >= self.min_batch_size:
            return True, f"Batch size reached ({len(self.pending_signals)})"
        
        batch_age = time.time() - self.pending_batch_start
        if batch_age > self.max_batch_age and self.pending_signals:
            return True, f"Batch age exceeded ({batch_age:.0f}s)"
        
        return False, "Waiting for more data"
    
    async def create_training_batch(self) -> Optional[ImprovementBatch]:
        """Create a training batch from pending signals"""
        if not self.pending_signals:
            return None
        
        # Sort by quality (train on best improvements first)
        signals = sorted(
            self.pending_signals,
            key=lambda s: s.quality_score * s.confidence,
            reverse=True,
        )
        
        batch = ImprovementBatch(
            batch_id=hashlib.sha256(f"{time.time()}{len(signals)}".encode()).hexdigest()[:16],
            signals=signals[:self.min_batch_size * 2],  # Cap batch size
            created_at=time.time(),
        )
        
        # Clear pending
        self.pending_signals = self.pending_signals[len(batch.signals):]
        self.pending_batch_start = time.time()
        
        logger.info(f"Created training batch {batch.batch_id}: {batch.size} examples, avg quality {batch.avg_quality:.2f}")
        return batch
    
    async def train_batch(self, batch: ImprovementBatch):
        """Train on an improvement batch"""
        if self.trainer is None:
            logger.warning("No trainer configured")
            batch.trained = True
            batch.trained_at = time.time()
            return
        
        logger.info(f"Training on batch {batch.batch_id}...")
        
        # Evaluate before
        batch.loss_before = await self.trainer.evaluate()
        
        # Train
        for signal in batch.signals:
            if signal.better_response:
                await self.trainer.train_example(
                    signal.prompt,
                    signal.better_response,
                )
                self.total_tokens_trained += len(signal.prompt.split()) + len(signal.better_response.split())
        
        # Evaluate after
        batch.loss_after = await self.trainer.evaluate()
        
        batch.trained = True
        batch.trained_at = time.time()
        
        # Record quality change
        if batch.loss_before and batch.loss_after:
            self.quality_before.append(batch.loss_before)
            self.quality_after.append(batch.loss_after)
        
        self.trained_batches.append(batch)
        self.total_improvements += batch.size
        
        logger.info(
            f"Batch {batch.batch_id} trained. "
            f"Loss: {batch.loss_before:.4f} → {batch.loss_after:.4f}"
        )
    
    async def improvement_loop(self):
        """Main continuous improvement loop"""
        self.running = True
        logger.info("=== CONTINUOUS IMPROVEMENT LOOP STARTED ===")
        
        synthetic_interval = 300  # Generate synthetic every 5 min
        last_synthetic = 0
        
        while self.running:
            try:
                # Generate synthetic data periodically
                if time.time() - last_synthetic > synthetic_interval:
                    examples = await self.synthetic_generator.generate_batch(50)
                    for ex in examples:
                        self.add_improvement(ex)
                    last_synthetic = time.time()
                    logger.info(f"Generated {len(examples)} synthetic examples")
                
                # Check if we should train
                should_train, reason = self.should_train()
                
                if should_train:
                    logger.info(f"Training triggered: {reason}")
                    batch = await self.create_training_batch()
                    if batch:
                        await self.train_batch(batch)
                
                # Log status periodically
                await asyncio.sleep(60)
                
                quality_trend = self.quality_monitor.get_quality_trend()
                logger.info(
                    f"Quality: avg={quality_trend.get('average', 0):.2f}, "
                    f"trend={quality_trend.get('trend', 0):+.3f}, "
                    f"pending={len(self.pending_signals)}"
                )
                
            except Exception as e:
                logger.error(f"Improvement loop error: {e}")
                await asyncio.sleep(10)
    
    async def stop(self):
        """Stop the improvement loop"""
        self.running = False
        
        # Train any pending signals
        if self.pending_signals:
            batch = await self.create_training_batch()
            if batch:
                await self.train_batch(batch)
        
        logger.info("=== CONTINUOUS IMPROVEMENT LOOP STOPPED ===")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get improvement engine statistics"""
        return {
            "total_improvements": self.total_improvements,
            "total_tokens_trained": self.total_tokens_trained,
            "pending_signals": len(self.pending_signals),
            "batches_trained": len(self.trained_batches),
            "quality_trend": self.quality_monitor.get_quality_trend(),
            "improvement_rate": (
                (sum(self.quality_after) - sum(self.quality_before)) / len(self.quality_after)
                if self.quality_after else 0
            ),
        }


class FeedbackCollector:
    """
    Collects feedback from various sources to drive improvement.
    """
    
    def __init__(self, engine: ContinuousImprovementEngine):
        self.engine = engine
        self.feedback_queue: List[Dict[str, Any]] = []
    
    def thumbs_up(self, prompt: str, response: str):
        """Record positive feedback"""
        signal = ImprovementSignal(
            signal_id=hashlib.sha256(f"thumbs_up:{prompt}".encode()).hexdigest()[:16],
            source=ImprovementSource.USER_FEEDBACK,
            timestamp=time.time(),
            prompt=prompt,
            better_response=response,
            quality_score=0.9,
            confidence=0.95,
            metadata={"feedback_type": "thumbs_up"},
        )
        self.engine.add_improvement(signal)
    
    def thumbs_down(self, prompt: str, bad_response: str):
        """Record negative feedback - marks for improvement"""
        signal = ImprovementSignal(
            signal_id=hashlib.sha256(f"thumbs_down:{prompt}".encode()).hexdigest()[:16],
            source=ImprovementSource.USER_FEEDBACK,
            timestamp=time.time(),
            prompt=prompt,
            current_response=bad_response,
            quality_score=0.2,  # Low quality
            confidence=0.95,
            metadata={"feedback_type": "thumbs_down", "needs_improvement": True},
        )
        self.engine.add_improvement(signal)
    
    def correction(self, prompt: str, bad_response: str, corrected_response: str):
        """Record a user correction - very high value signal"""
        signal = ImprovementSignal(
            signal_id=hashlib.sha256(f"correction:{prompt}".encode()).hexdigest()[:16],
            source=ImprovementSource.USER_FEEDBACK,
            timestamp=time.time(),
            prompt=prompt,
            current_response=bad_response,
            better_response=corrected_response,
            quality_score=1.0,  # Human-verified
            confidence=1.0,
            metadata={"feedback_type": "correction"},
        )
        self.engine.add_improvement(signal)


if __name__ == "__main__":
    # Demo the improvement engine
    engine = ContinuousImprovementEngine(min_batch_size=10)
    
    # Simulate some inferences
    test_interactions = [
        ("What is machine learning?", "Machine learning is a subset of AI.", 0.8),
        ("Explain neural networks", "err", 0.2),  # Bad response
        ("Write hello world", "print('Hello, World!')", 0.9),
        ("What is Python?", "Python is a programming language.", 0.7),
        ("Explain transformers", "Transformers are...", 0.4),  # Cut off
    ]
    
    for prompt, response, feedback in test_interactions:
        engine.record_inference(prompt, response, feedback)
    
    print("Quality trend:", engine.quality_monitor.get_quality_trend())
    print("Should train:", engine.should_train())
    print("Pending signals:", len(engine.pending_signals))
    print("\nStats:", engine.get_stats())
