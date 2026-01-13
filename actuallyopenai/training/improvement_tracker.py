"""
Benchmark & Improvement Tracking System.

Tracks model quality over time across multiple dimensions:
1. Standard benchmarks (perplexity, accuracy)
2. Capability benchmarks (reasoning, coding, math)
3. Efficiency metrics (tokens/sec, cost per quality)
4. Improvement rate tracking
5. Comparison with previous generations
"""

import asyncio
import json
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple

import structlog
import torch
import torch.nn as nn

logger = structlog.get_logger()


class BenchmarkType(str, Enum):
    """Types of benchmarks."""
    PERPLEXITY = "perplexity"  # Language modeling
    ACCURACY = "accuracy"  # Classification tasks
    REASONING = "reasoning"  # Logic & reasoning
    CODING = "coding"  # Code generation
    MATH = "math"  # Mathematical reasoning
    KNOWLEDGE = "knowledge"  # Factual knowledge
    INSTRUCTION = "instruction"  # Following instructions
    CUSTOM = "custom"  # User-defined


@dataclass
class BenchmarkResult:
    """Result from running a benchmark."""
    benchmark_id: str
    benchmark_type: BenchmarkType
    name: str
    
    # Scores
    score: float = 0.0
    max_score: float = 100.0
    normalized_score: float = 0.0  # 0-1 scale
    
    # Details
    num_samples: int = 0
    correct: int = 0
    total_tokens: int = 0
    
    # Performance
    inference_time_seconds: float = 0.0
    tokens_per_second: float = 0.0
    
    # Metadata
    model_generation_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    raw_outputs: List[Dict] = field(default_factory=list)


@dataclass
class ImprovementMetric:
    """Tracks improvement over time."""
    metric_name: str
    
    # Current values
    current_value: float = 0.0
    previous_value: float = 0.0
    best_value: float = 0.0
    
    # Change
    absolute_change: float = 0.0
    percent_change: float = 0.0
    
    # Trend
    trend_direction: str = "stable"  # "improving", "declining", "stable"
    trend_strength: float = 0.0  # How strong the trend is
    
    # History
    history: List[Tuple[datetime, float]] = field(default_factory=list)


@dataclass
class QualityReport:
    """Comprehensive quality report for a model generation."""
    generation_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Overall scores
    overall_score: float = 0.0
    overall_rank: int = 0  # Rank among all generations
    
    # Benchmark results
    benchmarks: List[BenchmarkResult] = field(default_factory=list)
    
    # Improvement metrics
    improvements: Dict[str, ImprovementMetric] = field(default_factory=dict)
    
    # Comparison
    comparison_to_previous: Dict[str, float] = field(default_factory=dict)
    comparison_to_best: Dict[str, float] = field(default_factory=dict)
    
    # Resources used
    compute_hours_used: float = 0.0
    tokens_trained: int = 0
    
    # Efficiency
    improvement_per_compute_hour: float = 0.0
    cost_efficiency_score: float = 0.0


class BenchmarkSuite:
    """A suite of benchmarks for evaluating model quality."""
    
    def __init__(self, name: str, benchmarks: List[Dict]):
        self.name = name
        self.benchmarks = benchmarks
    
    @classmethod
    def standard_llm_suite(cls) -> "BenchmarkSuite":
        """Create standard LLM benchmark suite."""
        return cls("Standard LLM", [
            {
                "id": "perplexity",
                "type": BenchmarkType.PERPLEXITY,
                "name": "Language Model Perplexity",
                "weight": 1.0,
                "lower_is_better": True
            },
            {
                "id": "hellaswag",
                "type": BenchmarkType.REASONING,
                "name": "HellaSwag Commonsense",
                "weight": 1.0,
                "lower_is_better": False
            },
            {
                "id": "arc_easy",
                "type": BenchmarkType.KNOWLEDGE,
                "name": "ARC Easy",
                "weight": 0.5,
                "lower_is_better": False
            },
            {
                "id": "arc_challenge",
                "type": BenchmarkType.REASONING,
                "name": "ARC Challenge",
                "weight": 1.0,
                "lower_is_better": False
            },
            {
                "id": "gsm8k",
                "type": BenchmarkType.MATH,
                "name": "GSM8K Math",
                "weight": 1.0,
                "lower_is_better": False
            },
            {
                "id": "humaneval",
                "type": BenchmarkType.CODING,
                "name": "HumanEval Coding",
                "weight": 1.0,
                "lower_is_better": False
            }
        ])


class ImprovementTracker:
    """
    Comprehensive system for tracking model improvement over time.
    
    This is crucial for demonstrating that the AI is actually improving
    as more compute is contributed.
    """
    
    def __init__(
        self,
        tracking_dir: str = "./tracking",
        benchmark_suite: Optional[BenchmarkSuite] = None,
        eval_interval_steps: int = 1000,
    ):
        self.tracking_dir = tracking_dir
        self.benchmark_suite = benchmark_suite or BenchmarkSuite.standard_llm_suite()
        self.eval_interval_steps = eval_interval_steps
        
        # Results storage
        self.benchmark_results: Dict[str, List[BenchmarkResult]] = {}
        self.quality_reports: List[QualityReport] = []
        
        # Improvement tracking
        self.improvement_metrics: Dict[str, ImprovementMetric] = {}
        
        # Best values
        self.best_scores: Dict[str, float] = {}
        self.best_generation: Dict[str, str] = {}  # benchmark -> generation_id
        
        # Leaderboard
        self.generation_rankings: List[Tuple[str, float]] = []
        
        # Callbacks
        self.on_new_best: Optional[Callable] = None
        self.on_improvement: Optional[Callable] = None
        
        logger.info(
            "ImprovementTracker initialized",
            benchmarks=len(self.benchmark_suite.benchmarks)
        )
    
    async def evaluate_generation(
        self,
        generation_id: str,
        model: nn.Module,
        compute_hours: float = 0.0,
        tokens_trained: int = 0
    ) -> QualityReport:
        """
        Run full evaluation for a model generation.
        """
        logger.info(
            "Running evaluation",
            generation_id=generation_id
        )
        
        results = []
        
        # Run each benchmark
        for benchmark_config in self.benchmark_suite.benchmarks:
            result = await self._run_benchmark(
                model=model,
                generation_id=generation_id,
                benchmark_config=benchmark_config
            )
            results.append(result)
            
            # Store result
            bench_id = benchmark_config["id"]
            if bench_id not in self.benchmark_results:
                self.benchmark_results[bench_id] = []
            self.benchmark_results[bench_id].append(result)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(results)
        
        # Build improvement metrics
        improvements = self._calculate_improvements(results, generation_id)
        
        # Comparison to previous and best
        comparison_prev = self._compare_to_previous(results)
        comparison_best = self._compare_to_best(results)
        
        # Calculate efficiency
        improvement_per_hour = 0.0
        if compute_hours > 0 and comparison_prev:
            avg_improvement = statistics.mean(comparison_prev.values())
            improvement_per_hour = avg_improvement / compute_hours
        
        # Create report
        report = QualityReport(
            generation_id=generation_id,
            overall_score=overall_score,
            benchmarks=results,
            improvements=improvements,
            comparison_to_previous=comparison_prev,
            comparison_to_best=comparison_best,
            compute_hours_used=compute_hours,
            tokens_trained=tokens_trained,
            improvement_per_compute_hour=improvement_per_hour
        )
        
        # Update rankings
        self._update_rankings(generation_id, overall_score)
        report.overall_rank = self._get_rank(generation_id)
        
        # Store report
        self.quality_reports.append(report)
        
        # Check for new bests
        await self._check_for_new_bests(results, generation_id)
        
        # Callbacks
        if self.on_improvement and comparison_prev:
            avg_improvement = statistics.mean(comparison_prev.values())
            if avg_improvement > 0:
                await self.on_improvement(report)
        
        logger.info(
            "Evaluation complete",
            generation_id=generation_id,
            overall_score=round(overall_score, 4),
            rank=report.overall_rank,
            improvement_per_hour=round(improvement_per_hour, 6)
        )
        
        return report
    
    async def _run_benchmark(
        self,
        model: nn.Module,
        generation_id: str,
        benchmark_config: Dict
    ) -> BenchmarkResult:
        """Run a single benchmark."""
        bench_id = benchmark_config["id"]
        bench_type = benchmark_config["type"]
        
        logger.debug(f"Running benchmark: {bench_id}")
        
        # Dispatch to specific benchmark implementations
        if bench_type == BenchmarkType.PERPLEXITY:
            score = await self._evaluate_perplexity(model)
            # For perplexity, lower is better, so invert for normalized score
            normalized = max(0, 1 - (score / 100))  # Assume 100 is bad perplexity
        else:
            # For other benchmarks, simulate evaluation
            # In production, this would call actual benchmark datasets
            score = await self._evaluate_accuracy_benchmark(model, bench_id)
            normalized = score / 100
        
        return BenchmarkResult(
            benchmark_id=bench_id,
            benchmark_type=bench_type,
            name=benchmark_config["name"],
            score=score,
            normalized_score=normalized,
            model_generation_id=generation_id
        )
    
    async def _evaluate_perplexity(self, model: nn.Module) -> float:
        """Evaluate model perplexity on validation data."""
        # Placeholder - in production would use actual validation data
        model.eval()
        
        # Simulate perplexity calculation
        # Lower perplexity = better model
        with torch.no_grad():
            # Would normally compute cross-entropy loss and exp() it
            # For demo, return a reasonable value
            param_count = sum(p.numel() for p in model.parameters())
            # Rough heuristic: larger models tend to have lower perplexity
            base_perplexity = 50 / (1 + param_count / 1e8)
            return max(1.0, base_perplexity + torch.randn(1).item() * 2)
    
    async def _evaluate_accuracy_benchmark(
        self, model: nn.Module, benchmark_id: str
    ) -> float:
        """Evaluate accuracy on a benchmark."""
        # Placeholder - in production would use actual benchmark datasets
        model.eval()
        
        # Simulate benchmark evaluation
        # Score improves with more parameters (rough heuristic)
        param_count = sum(p.numel() for p in model.parameters())
        base_score = min(80, 30 + 50 * (param_count / 1e9))
        
        # Add some variance
        score = base_score + torch.randn(1).item() * 5
        return max(0, min(100, score))
    
    def _calculate_overall_score(
        self, results: List[BenchmarkResult]
    ) -> float:
        """Calculate weighted overall score."""
        if not results:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for result in results:
            # Find weight from config
            weight = 1.0
            for bench_config in self.benchmark_suite.benchmarks:
                if bench_config["id"] == result.benchmark_id:
                    weight = bench_config.get("weight", 1.0)
                    break
            
            weighted_sum += result.normalized_score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_improvements(
        self,
        results: List[BenchmarkResult],
        generation_id: str
    ) -> Dict[str, ImprovementMetric]:
        """Calculate improvement metrics."""
        improvements = {}
        
        for result in results:
            bench_id = result.benchmark_id
            
            # Get history
            history = self.benchmark_results.get(bench_id, [])
            
            metric = ImprovementMetric(metric_name=bench_id)
            metric.current_value = result.normalized_score
            
            if len(history) > 0:
                previous = history[-1]
                metric.previous_value = previous.normalized_score
                metric.absolute_change = metric.current_value - metric.previous_value
                
                if metric.previous_value > 0:
                    metric.percent_change = (
                        metric.absolute_change / metric.previous_value * 100
                    )
            
            # Best value
            metric.best_value = self.best_scores.get(bench_id, metric.current_value)
            
            # Calculate trend
            if len(history) >= 3:
                recent_values = [h.normalized_score for h in history[-3:]]
                recent_values.append(metric.current_value)
                
                # Simple trend calculation
                diffs = [recent_values[i+1] - recent_values[i] for i in range(len(recent_values)-1)]
                avg_diff = statistics.mean(diffs)
                
                if avg_diff > 0.01:
                    metric.trend_direction = "improving"
                    metric.trend_strength = min(1.0, avg_diff * 10)
                elif avg_diff < -0.01:
                    metric.trend_direction = "declining"
                    metric.trend_strength = min(1.0, abs(avg_diff) * 10)
                else:
                    metric.trend_direction = "stable"
                    metric.trend_strength = 0.0
            
            # Store history
            metric.history = [
                (h.timestamp, h.normalized_score) for h in history[-50:]
            ]
            metric.history.append((result.timestamp, result.normalized_score))
            
            improvements[bench_id] = metric
        
        return improvements
    
    def _compare_to_previous(
        self, results: List[BenchmarkResult]
    ) -> Dict[str, float]:
        """Compare results to previous generation."""
        comparison = {}
        
        for result in results:
            bench_id = result.benchmark_id
            history = self.benchmark_results.get(bench_id, [])
            
            if len(history) > 0:
                previous = history[-1]
                if previous.normalized_score > 0:
                    change = (
                        (result.normalized_score - previous.normalized_score) /
                        previous.normalized_score * 100
                    )
                    comparison[bench_id] = change
        
        return comparison
    
    def _compare_to_best(
        self, results: List[BenchmarkResult]
    ) -> Dict[str, float]:
        """Compare results to best ever."""
        comparison = {}
        
        for result in results:
            bench_id = result.benchmark_id
            best = self.best_scores.get(bench_id)
            
            if best is not None and best > 0:
                change = (result.normalized_score - best) / best * 100
                comparison[bench_id] = change
        
        return comparison
    
    async def _check_for_new_bests(
        self,
        results: List[BenchmarkResult],
        generation_id: str
    ):
        """Check if any results are new bests."""
        for result in results:
            bench_id = result.benchmark_id
            
            current_best = self.best_scores.get(bench_id)
            is_new_best = (
                current_best is None or
                result.normalized_score > current_best
            )
            
            if is_new_best:
                self.best_scores[bench_id] = result.normalized_score
                self.best_generation[bench_id] = generation_id
                
                logger.info(
                    f"🏆 New best on {bench_id}!",
                    score=round(result.normalized_score, 4),
                    generation_id=generation_id
                )
                
                if self.on_new_best:
                    await self.on_new_best(bench_id, result)
    
    def _update_rankings(self, generation_id: str, score: float):
        """Update generation rankings."""
        # Remove existing entry if present
        self.generation_rankings = [
            (gid, s) for gid, s in self.generation_rankings
            if gid != generation_id
        ]
        
        # Add new entry
        self.generation_rankings.append((generation_id, score))
        
        # Sort by score descending
        self.generation_rankings.sort(key=lambda x: x[1], reverse=True)
    
    def _get_rank(self, generation_id: str) -> int:
        """Get rank of a generation."""
        for i, (gid, _) in enumerate(self.generation_rankings):
            if gid == generation_id:
                return i + 1
        return len(self.generation_rankings)
    
    # =========================================================================
    # Analytics & Reporting
    # =========================================================================
    
    def get_improvement_summary(self) -> Dict[str, Any]:
        """Get summary of improvements over time."""
        if not self.quality_reports:
            return {"message": "No evaluations yet"}
        
        first_report = self.quality_reports[0]
        latest_report = self.quality_reports[-1]
        
        total_improvement = (
            latest_report.overall_score - first_report.overall_score
        ) / max(0.01, first_report.overall_score) * 100
        
        # Per-benchmark improvement
        benchmark_improvements = {}
        for bench_id in self.best_scores:
            history = self.benchmark_results.get(bench_id, [])
            if len(history) >= 2:
                first_score = history[0].normalized_score
                latest_score = history[-1].normalized_score
                if first_score > 0:
                    improvement = (latest_score - first_score) / first_score * 100
                    benchmark_improvements[bench_id] = round(improvement, 2)
        
        # Compute efficiency
        total_compute = sum(r.compute_hours_used for r in self.quality_reports)
        total_tokens = sum(r.tokens_trained for r in self.quality_reports)
        
        if total_compute > 0:
            improvement_per_hour = total_improvement / total_compute
        else:
            improvement_per_hour = 0
        
        return {
            "total_evaluations": len(self.quality_reports),
            "overall_improvement_percent": round(total_improvement, 2),
            "benchmark_improvements": benchmark_improvements,
            "best_scores": {k: round(v, 4) for k, v in self.best_scores.items()},
            "total_compute_hours": round(total_compute, 2),
            "total_tokens_trained": total_tokens,
            "improvement_per_compute_hour": round(improvement_per_hour, 4),
            "current_rank": 1 if self.generation_rankings else 0,
            "generations_evaluated": len(self.generation_rankings)
        }
    
    def get_leaderboard(self, limit: int = 10) -> List[Dict]:
        """Get generation leaderboard."""
        return [
            {
                "rank": i + 1,
                "generation_id": gid,
                "overall_score": round(score, 4)
            }
            for i, (gid, score) in enumerate(self.generation_rankings[:limit])
        ]
    
    def get_benchmark_history(
        self, benchmark_id: str, limit: int = 50
    ) -> List[Dict]:
        """Get score history for a benchmark."""
        history = self.benchmark_results.get(benchmark_id, [])
        
        return [
            {
                "generation_id": r.model_generation_id,
                "score": round(r.score, 4),
                "normalized_score": round(r.normalized_score, 4),
                "timestamp": r.timestamp.isoformat()
            }
            for r in history[-limit:]
        ]
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    async def save_state(self, path: Optional[str] = None):
        """Save tracking state."""
        path = path or os.path.join(self.tracking_dir, "tracking_state.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state = {
            "best_scores": self.best_scores,
            "best_generation": self.best_generation,
            "generation_rankings": self.generation_rankings,
            "benchmark_results": {
                bid: [
                    {
                        "benchmark_id": r.benchmark_id,
                        "score": r.score,
                        "normalized_score": r.normalized_score,
                        "generation_id": r.model_generation_id,
                        "timestamp": r.timestamp.isoformat()
                    }
                    for r in results[-100:]  # Keep last 100
                ]
                for bid, results in self.benchmark_results.items()
            },
            "saved_at": datetime.utcnow().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info("Tracking state saved", path=path)
    
    async def load_state(self, path: Optional[str] = None):
        """Load tracking state."""
        path = path or os.path.join(self.tracking_dir, "tracking_state.json")
        
        if not os.path.exists(path):
            return
        
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.best_scores = state.get("best_scores", {})
        self.best_generation = state.get("best_generation", {})
        self.generation_rankings = [
            tuple(item) for item in state.get("generation_rankings", [])
        ]
        
        # Reconstruct benchmark results
        for bid, results_data in state.get("benchmark_results", {}).items():
            self.benchmark_results[bid] = []
            for r_data in results_data:
                result = BenchmarkResult(
                    benchmark_id=r_data["benchmark_id"],
                    benchmark_type=BenchmarkType.CUSTOM,
                    name=bid,
                    score=r_data["score"],
                    normalized_score=r_data["normalized_score"],
                    model_generation_id=r_data["generation_id"]
                )
                result.timestamp = datetime.fromisoformat(r_data["timestamp"])
                self.benchmark_results[bid].append(result)
        
        logger.info("Tracking state loaded", path=path)
