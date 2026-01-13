"""
Model Evolution System - Tracks and manages model generations over time.

This is what makes the AI truly self-improving:
1. Tracks model versions and their performance
2. Manages model lineage (which model evolved from which)
3. Implements evolutionary strategies (keep best, prune worst)
4. Enables model comparison and selection
5. Supports branching for experimentation
"""

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import uuid

import structlog
import torch
import torch.nn as nn

logger = structlog.get_logger()


class EvolutionStrategy(str, Enum):
    """Strategy for evolving models."""
    LINEAR = "linear"  # Simple sequential improvement
    BRANCHING = "branching"  # Multiple experimental branches
    TOURNAMENT = "tournament"  # Keep best, prune worst
    ENSEMBLE = "ensemble"  # Combine multiple models


class ModelStatus(str, Enum):
    """Status of a model in the evolution tree."""
    TRAINING = "training"  # Currently being trained
    EVALUATING = "evaluating"  # Being evaluated
    ACTIVE = "active"  # Best current model
    ARCHIVED = "archived"  # Superseded but kept
    PRUNED = "pruned"  # Removed from active use


@dataclass
class ModelGeneration:
    """Represents a single generation/version of the model."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    generation_number: int = 0
    parent_id: Optional[str] = None
    
    # Model info
    model_hash: str = ""
    checkpoint_path: Optional[str] = None
    parameter_count: int = 0
    
    # Performance metrics
    loss: float = float('inf')
    perplexity: float = float('inf')
    accuracy: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Training info
    total_steps: int = 0
    total_tokens: int = 0
    total_compute_hours: float = 0.0
    num_contributors: int = 0
    
    # Status
    status: ModelStatus = ModelStatus.TRAINING
    created_at: datetime = field(default_factory=datetime.utcnow)
    finalized_at: Optional[datetime] = None
    
    # Improvement tracking
    improvement_over_parent: float = 0.0
    improvement_rate: float = 0.0  # Improvement per compute hour
    
    def compute_hash(self, model: nn.Module) -> str:
        """Compute a hash of the model parameters."""
        hash_input = ""
        for name, param in model.named_parameters():
            hash_input += f"{name}:{param.mean().item():.6f}"
        self.model_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        return self.model_hash


@dataclass
class EvolutionBranch:
    """A branch in the evolution tree for experimentation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    
    # Branch point
    parent_generation_id: str = ""
    
    # Generations in this branch
    generations: List[str] = field(default_factory=list)
    
    # Status
    is_active: bool = True
    is_merged: bool = False
    merged_to_main: bool = False
    
    # Performance
    best_loss: float = float('inf')
    best_generation_id: Optional[str] = None
    
    created_at: datetime = field(default_factory=datetime.utcnow)


class ModelEvolution:
    """
    Manages the evolution of AI models over time.
    
    This system tracks how the model improves as more compute is contributed,
    manages different model versions, and implements strategies for continuous
    improvement.
    """
    
    def __init__(
        self,
        model_family: str = "aoai-base",
        evolution_dir: str = "./evolution",
        strategy: EvolutionStrategy = EvolutionStrategy.LINEAR,
        max_generations_to_keep: int = 10,
        improvement_threshold: float = 0.01,  # 1% improvement to keep
    ):
        self.model_family = model_family
        self.evolution_dir = evolution_dir
        self.strategy = strategy
        self.max_generations_to_keep = max_generations_to_keep
        self.improvement_threshold = improvement_threshold
        
        # Generation tracking
        self.generations: Dict[str, ModelGeneration] = {}
        self.current_generation_id: Optional[str] = None
        self.best_generation_id: Optional[str] = None
        
        # Lineage (parent -> children)
        self.lineage: Dict[str, List[str]] = {}
        
        # Branches for experimentation
        self.branches: Dict[str, EvolutionBranch] = {}
        self.main_branch_id: str = "main"
        
        # Evolution history
        self.evolution_history: List[Dict] = []
        
        # Callbacks
        self.on_new_best: Optional[callable] = None
        self.on_generation_complete: Optional[callable] = None
        
        # Initialize main branch
        self._init_main_branch()
        
        logger.info(
            "ModelEvolution initialized",
            model_family=model_family,
            strategy=strategy.value
        )
    
    def _init_main_branch(self):
        """Initialize the main evolution branch."""
        main_branch = EvolutionBranch(
            id=self.main_branch_id,
            name="main",
            description="Main evolution branch"
        )
        self.branches[self.main_branch_id] = main_branch
    
    async def start_new_generation(
        self,
        model: nn.Module,
        parent_id: Optional[str] = None,
        branch_id: Optional[str] = None
    ) -> ModelGeneration:
        """
        Start a new model generation.
        Called when beginning training from a checkpoint.
        """
        branch = branch_id or self.main_branch_id
        
        # Determine parent
        if parent_id is None:
            parent_id = self.current_generation_id
        
        # Create new generation
        gen_number = len(self.generations) + 1
        generation = ModelGeneration(
            generation_number=gen_number,
            parent_id=parent_id,
            status=ModelStatus.TRAINING,
            parameter_count=sum(p.numel() for p in model.parameters())
        )
        generation.compute_hash(model)
        
        # Register
        self.generations[generation.id] = generation
        self.current_generation_id = generation.id
        
        # Update lineage
        if parent_id:
            if parent_id not in self.lineage:
                self.lineage[parent_id] = []
            self.lineage[parent_id].append(generation.id)
        
        # Add to branch
        if branch in self.branches:
            self.branches[branch].generations.append(generation.id)
        
        logger.info(
            "Started new generation",
            generation_id=generation.id,
            gen_number=gen_number,
            parent_id=parent_id,
            branch=branch
        )
        
        return generation
    
    async def update_generation_metrics(
        self,
        generation_id: str,
        loss: float,
        steps: int,
        tokens: int,
        compute_hours: float,
        custom_metrics: Optional[Dict[str, float]] = None
    ):
        """Update metrics for a generation during training."""
        if generation_id not in self.generations:
            return
        
        gen = self.generations[generation_id]
        gen.loss = loss
        gen.total_steps = steps
        gen.total_tokens = tokens
        gen.total_compute_hours = compute_hours
        
        if custom_metrics:
            gen.custom_metrics.update(custom_metrics)
        
        # Calculate perplexity from loss
        import math
        gen.perplexity = math.exp(min(loss, 20))  # Cap to prevent overflow
        
        # Calculate improvement over parent
        if gen.parent_id and gen.parent_id in self.generations:
            parent = self.generations[gen.parent_id]
            if parent.loss > 0:
                gen.improvement_over_parent = (parent.loss - loss) / parent.loss
        
        # Calculate improvement rate
        if compute_hours > 0:
            gen.improvement_rate = gen.improvement_over_parent / compute_hours
    
    async def finalize_generation(
        self,
        generation_id: str,
        model: nn.Module,
        checkpoint_path: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Finalize a generation after training.
        Determines if it's an improvement and updates the evolution tree.
        
        Returns: (is_improvement, message)
        """
        if generation_id not in self.generations:
            return False, "Generation not found"
        
        gen = self.generations[generation_id]
        gen.status = ModelStatus.EVALUATING
        gen.finalized_at = datetime.utcnow()
        gen.checkpoint_path = checkpoint_path
        gen.compute_hash(model)
        
        # Compare to best
        is_improvement = False
        message = ""
        
        if self.best_generation_id is None:
            # First generation
            is_improvement = True
            message = "First generation - setting as best"
        else:
            best_gen = self.generations[self.best_generation_id]
            improvement = (best_gen.loss - gen.loss) / best_gen.loss
            
            if improvement >= self.improvement_threshold:
                is_improvement = True
                message = f"Improvement of {improvement*100:.2f}% over previous best"
            else:
                message = f"No significant improvement ({improvement*100:.2f}%)"
        
        # Apply evolution strategy
        if is_improvement:
            gen.status = ModelStatus.ACTIVE
            
            # Archive old best
            if self.best_generation_id:
                old_best = self.generations[self.best_generation_id]
                old_best.status = ModelStatus.ARCHIVED
            
            self.best_generation_id = generation_id
            
            # Update branch best
            for branch in self.branches.values():
                if generation_id in branch.generations:
                    if gen.loss < branch.best_loss:
                        branch.best_loss = gen.loss
                        branch.best_generation_id = generation_id
            
            # Callback
            if self.on_new_best:
                await self.on_new_best(gen)
            
            logger.info(
                "🎉 New best model!",
                generation_id=generation_id,
                loss=gen.loss,
                improvement=message
            )
        else:
            gen.status = ModelStatus.ARCHIVED
            logger.info(
                "Generation archived",
                generation_id=generation_id,
                loss=gen.loss,
                reason=message
            )
        
        # Record evolution
        self.evolution_history.append({
            "generation_id": generation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "loss": gen.loss,
            "is_improvement": is_improvement,
            "message": message
        })
        
        # Prune old generations if needed
        await self._prune_generations()
        
        # Callback
        if self.on_generation_complete:
            await self.on_generation_complete(gen, is_improvement)
        
        return is_improvement, message
    
    async def _prune_generations(self):
        """Prune old generations to manage storage."""
        # Get archived generations sorted by age
        archived = [
            gen for gen in self.generations.values()
            if gen.status == ModelStatus.ARCHIVED
        ]
        archived.sort(key=lambda g: g.finalized_at or g.created_at)
        
        # Keep only the most recent
        while len(archived) > self.max_generations_to_keep:
            to_prune = archived.pop(0)
            to_prune.status = ModelStatus.PRUNED
            
            # Delete checkpoint if exists
            if to_prune.checkpoint_path and os.path.exists(to_prune.checkpoint_path):
                try:
                    os.remove(to_prune.checkpoint_path)
                    logger.debug(
                        "Deleted checkpoint",
                        path=to_prune.checkpoint_path
                    )
                except Exception as e:
                    logger.error("Failed to delete checkpoint", error=str(e))
    
    # =========================================================================
    # Branching for Experimentation
    # =========================================================================
    
    async def create_branch(
        self,
        name: str,
        description: str = "",
        branch_from: Optional[str] = None
    ) -> EvolutionBranch:
        """Create a new evolution branch for experimentation."""
        branch_from = branch_from or self.current_generation_id
        
        branch = EvolutionBranch(
            name=name,
            description=description,
            parent_generation_id=branch_from
        )
        
        self.branches[branch.id] = branch
        
        logger.info(
            "Created evolution branch",
            branch_id=branch.id,
            name=name,
            branched_from=branch_from
        )
        
        return branch
    
    async def merge_branch(
        self,
        branch_id: str,
        to_main: bool = True
    ) -> Optional[str]:
        """
        Merge a branch's best model to main if it's an improvement.
        Returns the merged generation ID if successful.
        """
        if branch_id not in self.branches:
            return None
        
        branch = self.branches[branch_id]
        
        if not branch.best_generation_id:
            logger.warning("Branch has no best generation", branch_id=branch_id)
            return None
        
        branch_best = self.generations[branch.best_generation_id]
        
        # Compare to main best
        if self.best_generation_id:
            main_best = self.generations[self.best_generation_id]
            
            if branch_best.loss >= main_best.loss:
                logger.info(
                    "Branch not better than main",
                    branch_loss=branch_best.loss,
                    main_loss=main_best.loss
                )
                branch.is_merged = False
                return None
        
        # Merge - the branch best becomes the new main best
        if to_main:
            self.branches[self.main_branch_id].generations.append(branch.best_generation_id)
            branch.merged_to_main = True
        
        branch.is_merged = True
        branch.is_active = False
        
        logger.info(
            "Branch merged",
            branch_id=branch_id,
            merged_generation=branch.best_generation_id
        )
        
        return branch.best_generation_id
    
    # =========================================================================
    # Tournament Selection (Alternative Strategy)
    # =========================================================================
    
    async def run_tournament(
        self,
        candidates: List[str],
        evaluation_fn: callable
    ) -> Optional[str]:
        """
        Run tournament selection among candidate generations.
        Useful when multiple branches have been training in parallel.
        """
        if len(candidates) < 2:
            return candidates[0] if candidates else None
        
        logger.info(
            "Running tournament selection",
            num_candidates=len(candidates)
        )
        
        # Evaluate all candidates
        scores = {}
        for gen_id in candidates:
            if gen_id in self.generations:
                gen = self.generations[gen_id]
                score = await evaluation_fn(gen)
                scores[gen_id] = score
                logger.debug(
                    "Tournament evaluation",
                    generation_id=gen_id,
                    score=score
                )
        
        # Select winner (lowest loss/score wins)
        winner_id = min(scores, key=scores.get)
        winner = self.generations[winner_id]
        winner.status = ModelStatus.ACTIVE
        
        # Archive losers
        for gen_id in candidates:
            if gen_id != winner_id and gen_id in self.generations:
                self.generations[gen_id].status = ModelStatus.ARCHIVED
        
        # Update best if winner is better
        if self.best_generation_id:
            best = self.generations[self.best_generation_id]
            if scores[winner_id] < best.loss:
                best.status = ModelStatus.ARCHIVED
                self.best_generation_id = winner_id
        else:
            self.best_generation_id = winner_id
        
        logger.info(
            "Tournament winner",
            winner_id=winner_id,
            score=scores[winner_id]
        )
        
        return winner_id
    
    # =========================================================================
    # Model Access
    # =========================================================================
    
    def get_best_model_path(self) -> Optional[str]:
        """Get the checkpoint path for the best model."""
        if self.best_generation_id and self.best_generation_id in self.generations:
            return self.generations[self.best_generation_id].checkpoint_path
        return None
    
    def get_generation(self, generation_id: str) -> Optional[ModelGeneration]:
        """Get a specific generation."""
        return self.generations.get(generation_id)
    
    def get_current_generation(self) -> Optional[ModelGeneration]:
        """Get the current generation being trained."""
        if self.current_generation_id:
            return self.generations.get(self.current_generation_id)
        return None
    
    def get_best_generation(self) -> Optional[ModelGeneration]:
        """Get the best performing generation."""
        if self.best_generation_id:
            return self.generations.get(self.best_generation_id)
        return None
    
    # =========================================================================
    # Evolution Analytics
    # =========================================================================
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get a summary of the evolution history."""
        total_gens = len(self.generations)
        improvements = len([h for h in self.evolution_history if h.get("is_improvement")])
        
        # Calculate total resources used
        total_compute = sum(g.total_compute_hours for g in self.generations.values())
        total_tokens = sum(g.total_tokens for g in self.generations.values())
        
        # Loss progression
        loss_history = []
        for gen_id in sorted(
            self.generations.keys(),
            key=lambda x: self.generations[x].generation_number
        ):
            gen = self.generations[gen_id]
            if gen.status != ModelStatus.TRAINING:
                loss_history.append({
                    "generation": gen.generation_number,
                    "loss": gen.loss,
                    "status": gen.status.value
                })
        
        return {
            "model_family": self.model_family,
            "strategy": self.strategy.value,
            "total_generations": total_gens,
            "improvements": improvements,
            "improvement_rate": improvements / max(1, total_gens),
            "current_generation_id": self.current_generation_id,
            "best_generation_id": self.best_generation_id,
            "best_loss": (
                self.generations[self.best_generation_id].loss
                if self.best_generation_id else None
            ),
            "total_compute_hours": round(total_compute, 2),
            "total_tokens_processed": total_tokens,
            "active_branches": len([b for b in self.branches.values() if b.is_active]),
            "loss_history": loss_history
        }
    
    def get_lineage(self, generation_id: str) -> List[str]:
        """Get the lineage (ancestors) of a generation."""
        lineage = []
        current_id = generation_id
        
        while current_id:
            lineage.append(current_id)
            gen = self.generations.get(current_id)
            if gen:
                current_id = gen.parent_id
            else:
                break
        
        return list(reversed(lineage))
    
    def get_descendants(self, generation_id: str) -> List[str]:
        """Get all descendants of a generation."""
        descendants = []
        
        def collect_descendants(gen_id: str):
            children = self.lineage.get(gen_id, [])
            for child_id in children:
                descendants.append(child_id)
                collect_descendants(child_id)
        
        collect_descendants(generation_id)
        return descendants
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    async def save_evolution_state(self, path: Optional[str] = None):
        """Save the evolution state to disk."""
        path = path or os.path.join(self.evolution_dir, "evolution_state.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state = {
            "model_family": self.model_family,
            "strategy": self.strategy.value,
            "current_generation_id": self.current_generation_id,
            "best_generation_id": self.best_generation_id,
            "generations": {
                gid: {
                    "id": g.id,
                    "generation_number": g.generation_number,
                    "parent_id": g.parent_id,
                    "model_hash": g.model_hash,
                    "checkpoint_path": g.checkpoint_path,
                    "loss": g.loss,
                    "perplexity": g.perplexity,
                    "total_steps": g.total_steps,
                    "total_tokens": g.total_tokens,
                    "total_compute_hours": g.total_compute_hours,
                    "status": g.status.value,
                    "created_at": g.created_at.isoformat(),
                    "improvement_over_parent": g.improvement_over_parent
                }
                for gid, g in self.generations.items()
            },
            "lineage": self.lineage,
            "branches": {
                bid: {
                    "id": b.id,
                    "name": b.name,
                    "generations": b.generations,
                    "best_loss": b.best_loss,
                    "is_active": b.is_active
                }
                for bid, b in self.branches.items()
            },
            "evolution_history": self.evolution_history[-100:]  # Keep last 100
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info("Evolution state saved", path=path)
    
    async def load_evolution_state(self, path: Optional[str] = None):
        """Load evolution state from disk."""
        path = path or os.path.join(self.evolution_dir, "evolution_state.json")
        
        if not os.path.exists(path):
            return
        
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.model_family = state.get("model_family", self.model_family)
        self.strategy = EvolutionStrategy(state.get("strategy", "linear"))
        self.current_generation_id = state.get("current_generation_id")
        self.best_generation_id = state.get("best_generation_id")
        self.lineage = state.get("lineage", {})
        self.evolution_history = state.get("evolution_history", [])
        
        # Reconstruct generations
        for gid, gdata in state.get("generations", {}).items():
            gen = ModelGeneration(
                id=gdata["id"],
                generation_number=gdata["generation_number"],
                parent_id=gdata.get("parent_id"),
                model_hash=gdata.get("model_hash", ""),
                checkpoint_path=gdata.get("checkpoint_path"),
                loss=gdata.get("loss", float('inf')),
                perplexity=gdata.get("perplexity", float('inf')),
                total_steps=gdata.get("total_steps", 0),
                total_tokens=gdata.get("total_tokens", 0),
                total_compute_hours=gdata.get("total_compute_hours", 0),
                status=ModelStatus(gdata.get("status", "archived")),
                improvement_over_parent=gdata.get("improvement_over_parent", 0)
            )
            if gdata.get("created_at"):
                gen.created_at = datetime.fromisoformat(gdata["created_at"])
            self.generations[gid] = gen
        
        logger.info(
            "Evolution state loaded",
            generations=len(self.generations),
            best_id=self.best_generation_id
        )
