"""
Worker Verification System for ActuallyOpenAI.

Ensures workers are actually contributing compute and not cheating.

Methods:
1. Proof of Work - Verifiable computation results
2. Spot Checks - Random verification of worker outputs
3. Gradient Verification - Check that gradients are valid
4. Hardware Attestation - Verify claimed hardware specs
5. Reputation System - Track worker reliability over time
"""

import asyncio
import hashlib
import hmac
import json
import random
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
import struct

import structlog

logger = structlog.get_logger()


# =============================================================================
# Verification Methods
# =============================================================================

class VerificationMethod(str, Enum):
    """Verification methods for worker contributions."""
    PROOF_OF_WORK = "proof_of_work"
    GRADIENT_CHECK = "gradient_check"
    SPOT_CHECK = "spot_check"
    HARDWARE_ATTESTATION = "hardware_attestation"
    CONSENSUS = "consensus"


class VerificationStatus(str, Enum):
    """Status of a verification."""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    SUSPICIOUS = "suspicious"


@dataclass
class VerificationChallenge:
    """A verification challenge for a worker."""
    id: str
    worker_id: str
    method: VerificationMethod
    challenge_data: Dict[str, Any]
    expected_result: Optional[str] = None
    issued_at: datetime = field(default_factory=datetime.utcnow)
    deadline: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(minutes=5))
    response: Optional[Dict[str, Any]] = None
    status: VerificationStatus = VerificationStatus.PENDING


@dataclass
class WorkerReputation:
    """Reputation score for a worker."""
    worker_id: str
    total_verifications: int = 0
    passed_verifications: int = 0
    failed_verifications: int = 0
    suspicious_count: int = 0
    reputation_score: float = 1.0  # 0.0 to 1.0
    last_verification: Optional[datetime] = None
    is_trusted: bool = False
    penalties: List[Dict[str, Any]] = field(default_factory=list)


# =============================================================================
# Proof of Work Verification
# =============================================================================

class ProofOfWorkVerifier:
    """
    Verify workers through proof of work.
    
    Workers must compute a hash that meets a difficulty target.
    This proves they have CPU/GPU cycles available.
    """
    
    def __init__(self, difficulty: int = 4):
        """
        Initialize verifier.
        
        Args:
            difficulty: Number of leading zeros required in hash
        """
        self.difficulty = difficulty
        self.target = "0" * difficulty
    
    def create_challenge(self, worker_id: str) -> VerificationChallenge:
        """Create a proof of work challenge."""
        challenge_data = {
            "nonce_seed": secrets.token_hex(32),
            "difficulty": self.difficulty,
            "timestamp": int(time.time())
        }
        
        return VerificationChallenge(
            id=secrets.token_hex(16),
            worker_id=worker_id,
            method=VerificationMethod.PROOF_OF_WORK,
            challenge_data=challenge_data
        )
    
    def solve_challenge(self, challenge: VerificationChallenge) -> Dict[str, Any]:
        """
        Solve a proof of work challenge (done by worker).
        
        Returns the nonce that produces a valid hash.
        """
        nonce_seed = challenge.challenge_data["nonce_seed"]
        nonce = 0
        
        while True:
            data = f"{nonce_seed}:{nonce}".encode()
            hash_result = hashlib.sha256(data).hexdigest()
            
            if hash_result.startswith(self.target):
                return {
                    "nonce": nonce,
                    "hash": hash_result,
                    "solve_time": time.time()
                }
            
            nonce += 1
            
            # Timeout protection
            if nonce > 10_000_000:
                raise TimeoutError("Could not solve PoW challenge")
    
    def verify_response(
        self,
        challenge: VerificationChallenge,
        response: Dict[str, Any]
    ) -> VerificationStatus:
        """Verify a worker's proof of work response."""
        try:
            nonce_seed = challenge.challenge_data["nonce_seed"]
            nonce = response["nonce"]
            claimed_hash = response["hash"]
            
            # Recompute hash
            data = f"{nonce_seed}:{nonce}".encode()
            actual_hash = hashlib.sha256(data).hexdigest()
            
            # Verify
            if actual_hash != claimed_hash:
                return VerificationStatus.FAILED
            
            if not actual_hash.startswith(self.target):
                return VerificationStatus.FAILED
            
            return VerificationStatus.PASSED
            
        except Exception as e:
            logger.error(f"PoW verification error: {e}")
            return VerificationStatus.FAILED


# =============================================================================
# Gradient Verification
# =============================================================================

class GradientVerifier:
    """
    Verify that workers are computing valid gradients.
    
    Send a known input/output pair and verify the gradient is correct.
    """
    
    def __init__(self, tolerance: float = 0.01):
        """
        Initialize verifier.
        
        Args:
            tolerance: Maximum allowed difference in gradient values
        """
        self.tolerance = tolerance
    
    def create_challenge(self, worker_id: str, model_config: Dict) -> VerificationChallenge:
        """Create a gradient verification challenge."""
        
        # Create deterministic test data
        seed = int(hashlib.sha256(worker_id.encode()).hexdigest()[:8], 16)
        random.seed(seed)
        
        # Generate test input
        seq_len = 32
        vocab_size = model_config.get("vocab_size", 260)
        test_input = [random.randint(0, vocab_size - 1) for _ in range(seq_len)]
        test_labels = test_input[1:] + [0]
        
        challenge_data = {
            "input_ids": test_input,
            "labels": test_labels,
            "seed": seed,
            "model_config": model_config
        }
        
        return VerificationChallenge(
            id=secrets.token_hex(16),
            worker_id=worker_id,
            method=VerificationMethod.GRADIENT_CHECK,
            challenge_data=challenge_data
        )
    
    def compute_expected_gradient_hash(
        self,
        challenge: VerificationChallenge,
        model
    ) -> str:
        """
        Compute expected gradient hash for verification.
        
        This is done by the orchestrator using a reference model.
        """
        import torch
        import torch.nn.functional as F
        
        # Prepare input
        input_ids = torch.tensor([challenge.challenge_data["input_ids"]])
        labels = torch.tensor([challenge.challenge_data["labels"]])
        
        # Forward pass
        model.zero_grad()
        output = model(input_ids)
        
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        
        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Hash gradients
        grad_values = []
        for param in model.parameters():
            if param.grad is not None:
                # Sample some gradient values
                flat_grad = param.grad.flatten()
                sample_indices = torch.linspace(0, len(flat_grad) - 1, min(100, len(flat_grad))).long()
                grad_values.extend(flat_grad[sample_indices].tolist())
        
        # Create hash
        grad_str = ",".join(f"{v:.6f}" for v in grad_values[:1000])
        return hashlib.sha256(grad_str.encode()).hexdigest()
    
    def verify_gradient_hash(
        self,
        expected_hash: str,
        reported_hash: str
    ) -> VerificationStatus:
        """Verify that gradient hash matches expected."""
        if expected_hash == reported_hash:
            return VerificationStatus.PASSED
        else:
            return VerificationStatus.FAILED


# =============================================================================
# Spot Check Verification
# =============================================================================

class SpotCheckVerifier:
    """
    Randomly verify worker outputs.
    
    Re-run a small portion of the work and compare results.
    """
    
    def __init__(self, check_rate: float = 0.05):
        """
        Initialize verifier.
        
        Args:
            check_rate: Fraction of tasks to spot check (0.0 to 1.0)
        """
        self.check_rate = check_rate
    
    def should_spot_check(self) -> bool:
        """Determine if this task should be spot checked."""
        return random.random() < self.check_rate
    
    def create_spot_check(
        self,
        worker_id: str,
        task_id: str,
        task_data: Dict[str, Any]
    ) -> VerificationChallenge:
        """Create a spot check challenge."""
        return VerificationChallenge(
            id=secrets.token_hex(16),
            worker_id=worker_id,
            method=VerificationMethod.SPOT_CHECK,
            challenge_data={
                "task_id": task_id,
                "task_data": task_data,
                "check_type": "output_comparison"
            }
        )
    
    def verify_outputs(
        self,
        worker_output: Dict[str, Any],
        reference_output: Dict[str, Any],
        tolerance: float = 0.01
    ) -> VerificationStatus:
        """
        Compare worker output with reference output.
        """
        try:
            # Compare loss values
            worker_loss = worker_output.get("loss", 0)
            reference_loss = reference_output.get("loss", 0)
            
            if abs(worker_loss - reference_loss) > tolerance:
                return VerificationStatus.SUSPICIOUS
            
            # Compare gradient norms
            worker_grad_norm = worker_output.get("grad_norm", 0)
            reference_grad_norm = reference_output.get("grad_norm", 0)
            
            if reference_grad_norm > 0:
                relative_diff = abs(worker_grad_norm - reference_grad_norm) / reference_grad_norm
                if relative_diff > tolerance:
                    return VerificationStatus.SUSPICIOUS
            
            return VerificationStatus.PASSED
            
        except Exception as e:
            logger.error(f"Spot check error: {e}")
            return VerificationStatus.FAILED


# =============================================================================
# Hardware Attestation
# =============================================================================

class HardwareAttestor:
    """
    Verify worker hardware claims.
    
    Uses benchmarks and system queries to verify hardware.
    """
    
    @staticmethod
    def create_hardware_challenge(worker_id: str) -> VerificationChallenge:
        """Create a hardware attestation challenge."""
        return VerificationChallenge(
            id=secrets.token_hex(16),
            worker_id=worker_id,
            method=VerificationMethod.HARDWARE_ATTESTATION,
            challenge_data={
                "benchmark_type": "matrix_multiply",
                "matrix_size": 2048,
                "iterations": 100,
                "expected_flops_range": {
                    "cpu_min": 1e9,
                    "cpu_max": 1e12,
                    "gpu_min": 1e12,
                    "gpu_max": 1e15
                }
            }
        )
    
    @staticmethod
    def run_benchmark(challenge: VerificationChallenge) -> Dict[str, Any]:
        """
        Run hardware benchmark (executed on worker).
        """
        import time
        
        try:
            import torch
            
            size = challenge.challenge_data["matrix_size"]
            iterations = challenge.challenge_data["iterations"]
            
            # Check for GPU
            has_gpu = torch.cuda.is_available()
            device = "cuda" if has_gpu else "cpu"
            
            # Create matrices
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            # Warmup
            for _ in range(10):
                c = torch.matmul(a, b)
            
            if has_gpu:
                torch.cuda.synchronize()
            
            # Benchmark
            start = time.time()
            for _ in range(iterations):
                c = torch.matmul(a, b)
            
            if has_gpu:
                torch.cuda.synchronize()
            
            elapsed = time.time() - start
            
            # Calculate FLOPS (2 * n^3 for matrix multiply)
            total_flops = 2 * (size ** 3) * iterations
            flops = total_flops / elapsed
            
            result = {
                "device": device,
                "matrix_size": size,
                "iterations": iterations,
                "elapsed_seconds": elapsed,
                "flops": flops,
                "teraflops": flops / 1e12
            }
            
            if has_gpu:
                result["gpu_name"] = torch.cuda.get_device_name()
                result["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def verify_hardware(
        claimed_hardware: Dict[str, Any],
        benchmark_result: Dict[str, Any]
    ) -> VerificationStatus:
        """Verify hardware claims match benchmark results."""
        
        if "error" in benchmark_result:
            return VerificationStatus.FAILED
        
        # Verify GPU claim
        claimed_gpu = claimed_hardware.get("has_gpu", False)
        actual_gpu = benchmark_result.get("device") == "cuda"
        
        if claimed_gpu and not actual_gpu:
            return VerificationStatus.FAILED
        
        # Verify performance is reasonable
        flops = benchmark_result.get("flops", 0)
        device = benchmark_result.get("device", "cpu")
        
        expected_range = {
            "cpu": (1e9, 1e12),
            "cuda": (1e12, 1e15)
        }
        
        min_flops, max_flops = expected_range.get(device, (0, float("inf")))
        
        if flops < min_flops or flops > max_flops:
            return VerificationStatus.SUSPICIOUS
        
        return VerificationStatus.PASSED


# =============================================================================
# Consensus Verification
# =============================================================================

class ConsensusVerifier:
    """
    Verify results through consensus among multiple workers.
    
    Same task is given to multiple workers, results must match.
    """
    
    def __init__(self, min_workers: int = 3, agreement_threshold: float = 0.66):
        """
        Initialize verifier.
        
        Args:
            min_workers: Minimum workers needed for consensus
            agreement_threshold: Fraction that must agree
        """
        self.min_workers = min_workers
        self.agreement_threshold = agreement_threshold
    
    def check_consensus(
        self,
        results: List[Dict[str, Any]],
        tolerance: float = 0.01
    ) -> Tuple[VerificationStatus, List[str]]:
        """
        Check if worker results reach consensus.
        
        Returns:
            (status, list of disagreeing worker IDs)
        """
        if len(results) < self.min_workers:
            return VerificationStatus.PENDING, []
        
        # Group similar results
        groups: List[List[Dict]] = []
        
        for result in results:
            loss = result.get("loss", 0)
            matched = False
            
            for group in groups:
                ref_loss = group[0].get("loss", 0)
                if abs(loss - ref_loss) <= tolerance:
                    group.append(result)
                    matched = True
                    break
            
            if not matched:
                groups.append([result])
        
        # Find largest group
        largest_group = max(groups, key=len)
        agreement = len(largest_group) / len(results)
        
        if agreement >= self.agreement_threshold:
            # Find disagreeing workers
            disagreeing = []
            for group in groups:
                if group != largest_group:
                    for result in group:
                        disagreeing.append(result.get("worker_id", "unknown"))
            
            return VerificationStatus.PASSED, disagreeing
        else:
            return VerificationStatus.SUSPICIOUS, []


# =============================================================================
# Reputation System
# =============================================================================

class ReputationManager:
    """
    Manage worker reputation scores.
    
    Workers with higher reputation get:
    - More tasks
    - Higher rewards
    - Less frequent verification
    """
    
    def __init__(self):
        self.reputations: Dict[str, WorkerReputation] = {}
        
        # Reputation thresholds
        self.TRUSTED_THRESHOLD = 0.95
        self.SUSPICIOUS_THRESHOLD = 0.5
        self.BANNED_THRESHOLD = 0.2
    
    def get_reputation(self, worker_id: str) -> WorkerReputation:
        """Get or create reputation for a worker."""
        if worker_id not in self.reputations:
            self.reputations[worker_id] = WorkerReputation(worker_id=worker_id)
        return self.reputations[worker_id]
    
    def record_verification(
        self,
        worker_id: str,
        status: VerificationStatus,
        method: VerificationMethod
    ):
        """Record a verification result."""
        rep = self.get_reputation(worker_id)
        rep.total_verifications += 1
        rep.last_verification = datetime.utcnow()
        
        if status == VerificationStatus.PASSED:
            rep.passed_verifications += 1
        elif status == VerificationStatus.FAILED:
            rep.failed_verifications += 1
        elif status == VerificationStatus.SUSPICIOUS:
            rep.suspicious_count += 1
        
        # Update score
        self._update_score(worker_id)
        
        logger.info(
            "Verification recorded",
            worker_id=worker_id,
            status=status.value,
            method=method.value,
            new_score=rep.reputation_score
        )
    
    def _update_score(self, worker_id: str):
        """Update reputation score using weighted average."""
        rep = self.reputations[worker_id]
        
        if rep.total_verifications == 0:
            return
        
        # Base score from pass rate
        pass_rate = rep.passed_verifications / rep.total_verifications
        
        # Penalty for suspicious activities
        suspicious_penalty = min(0.3, rep.suspicious_count * 0.05)
        
        # Time decay - older failed verifications matter less
        # (simplified - would use timestamps in production)
        
        # Calculate final score
        score = pass_rate - suspicious_penalty
        score = max(0.0, min(1.0, score))
        
        rep.reputation_score = score
        rep.is_trusted = score >= self.TRUSTED_THRESHOLD
    
    def apply_penalty(
        self,
        worker_id: str,
        reason: str,
        severity: float = 0.1
    ):
        """Apply a penalty to a worker's reputation."""
        rep = self.get_reputation(worker_id)
        
        penalty = {
            "timestamp": datetime.utcnow().isoformat(),
            "reason": reason,
            "severity": severity
        }
        rep.penalties.append(penalty)
        
        # Reduce score
        rep.reputation_score = max(0.0, rep.reputation_score - severity)
        
        logger.warning(
            "Penalty applied",
            worker_id=worker_id,
            reason=reason,
            severity=severity,
            new_score=rep.reputation_score
        )
    
    def is_banned(self, worker_id: str) -> bool:
        """Check if worker is banned."""
        rep = self.get_reputation(worker_id)
        return rep.reputation_score < self.BANNED_THRESHOLD
    
    def get_verification_rate(self, worker_id: str) -> float:
        """
        Get verification rate for a worker.
        
        Trusted workers are verified less often.
        """
        rep = self.get_reputation(worker_id)
        
        if rep.is_trusted:
            return 0.01  # 1% of tasks
        elif rep.reputation_score > self.SUSPICIOUS_THRESHOLD:
            return 0.05  # 5% of tasks
        else:
            return 0.20  # 20% of tasks


# =============================================================================
# Verification Coordinator
# =============================================================================

class VerificationCoordinator:
    """
    Coordinates all verification activities.
    
    Decides which verification methods to use and when.
    """
    
    def __init__(self):
        self.pow_verifier = ProofOfWorkVerifier(difficulty=4)
        self.gradient_verifier = GradientVerifier()
        self.spot_checker = SpotCheckVerifier(check_rate=0.05)
        self.hardware_attestor = HardwareAttestor()
        self.consensus_verifier = ConsensusVerifier()
        self.reputation_manager = ReputationManager()
        
        # Pending challenges
        self.pending_challenges: Dict[str, VerificationChallenge] = {}
    
    async def verify_worker(self, worker_id: str) -> bool:
        """
        Run verification checks on a worker.
        
        Returns True if worker passes all checks.
        """
        # Check if banned
        if self.reputation_manager.is_banned(worker_id):
            logger.warning(f"Worker {worker_id} is banned")
            return False
        
        # Determine verification rate
        rate = self.reputation_manager.get_verification_rate(worker_id)
        
        if random.random() > rate:
            # Skip verification this time
            return True
        
        # Run proof of work
        challenge = self.pow_verifier.create_challenge(worker_id)
        self.pending_challenges[challenge.id] = challenge
        
        # In production, this would be sent to the worker
        # and we'd wait for a response
        
        logger.info(
            "Verification challenge issued",
            worker_id=worker_id,
            method=VerificationMethod.PROOF_OF_WORK.value,
            challenge_id=challenge.id
        )
        
        return True
    
    async def handle_challenge_response(
        self,
        challenge_id: str,
        response: Dict[str, Any]
    ) -> VerificationStatus:
        """Handle a worker's response to a challenge."""
        challenge = self.pending_challenges.get(challenge_id)
        
        if not challenge:
            logger.warning(f"Unknown challenge: {challenge_id}")
            return VerificationStatus.FAILED
        
        # Verify based on method
        if challenge.method == VerificationMethod.PROOF_OF_WORK:
            status = self.pow_verifier.verify_response(challenge, response)
        elif challenge.method == VerificationMethod.HARDWARE_ATTESTATION:
            # Get claimed hardware from worker record
            status = self.hardware_attestor.verify_hardware({}, response)
        else:
            status = VerificationStatus.PENDING
        
        # Record result
        self.reputation_manager.record_verification(
            challenge.worker_id,
            status,
            challenge.method
        )
        
        # Clean up
        del self.pending_challenges[challenge_id]
        
        return status
    
    def get_worker_stats(self, worker_id: str) -> Dict[str, Any]:
        """Get verification statistics for a worker."""
        rep = self.reputation_manager.get_reputation(worker_id)
        
        return {
            "worker_id": worker_id,
            "reputation_score": rep.reputation_score,
            "is_trusted": rep.is_trusted,
            "total_verifications": rep.total_verifications,
            "pass_rate": rep.passed_verifications / max(1, rep.total_verifications),
            "suspicious_count": rep.suspicious_count,
            "is_banned": self.reputation_manager.is_banned(worker_id)
        }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Demo
    coordinator = VerificationCoordinator()
    
    # Test PoW verification
    worker_id = "test-worker-001"
    challenge = coordinator.pow_verifier.create_challenge(worker_id)
    
    print(f"Challenge created: {challenge.id}")
    print(f"Difficulty: {challenge.challenge_data['difficulty']}")
    
    # Solve challenge (simulating worker)
    response = coordinator.pow_verifier.solve_challenge(challenge)
    print(f"Solution found: nonce={response['nonce']}, hash={response['hash'][:20]}...")
    
    # Verify
    status = coordinator.pow_verifier.verify_response(challenge, response)
    print(f"Verification status: {status.value}")
    
    # Check reputation
    stats = coordinator.get_worker_stats(worker_id)
    print(f"Worker stats: {stats}")
