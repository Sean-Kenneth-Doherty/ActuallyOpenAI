"""
Core data models for ActuallyOpenAI.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import uuid


# =============================================================================
# Enums
# =============================================================================

class WorkerStatus(str, Enum):
    """Status of a worker node."""
    ONLINE = "online"
    OFFLINE = "offline"
    TRAINING = "training"
    IDLE = "idle"
    ERROR = "error"


class TaskStatus(str, Enum):
    """Status of a training task."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Type of training task."""
    FORWARD_PASS = "forward_pass"
    BACKWARD_PASS = "backward_pass"
    GRADIENT_COMPUTE = "gradient_compute"
    MODEL_UPDATE = "model_update"
    VALIDATION = "validation"


class PayoutStatus(str, Enum):
    """Status of a crypto payout."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Worker Models
# =============================================================================

class HardwareSpec(BaseModel):
    """Hardware specifications of a worker."""
    cpu_cores: int = Field(ge=1)
    cpu_model: str = ""
    ram_gb: float = Field(ge=1)
    gpu_count: int = Field(ge=0, default=0)
    gpu_model: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    bandwidth_mbps: float = Field(ge=1, default=100)


class Worker(BaseModel):
    """A compute contributor node."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    wallet_address: str
    name: str
    region: str
    hardware: HardwareSpec
    status: WorkerStatus = WorkerStatus.OFFLINE
    
    # Compute metrics
    compute_score: float = Field(default=0.0, ge=0)  # Normalized compute power
    reliability_score: float = Field(default=1.0, ge=0, le=1)  # Uptime reliability
    
    # Connection info
    last_heartbeat: Optional[datetime] = None
    connected_at: Optional[datetime] = None
    ip_address: Optional[str] = None
    
    # Lifetime stats
    total_tasks_completed: int = Field(default=0, ge=0)
    total_compute_hours: float = Field(default=0.0, ge=0)
    total_tokens_earned: Decimal = Field(default=Decimal("0"))
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class WorkerRegistration(BaseModel):
    """Request to register a new worker."""
    wallet_address: str
    name: str
    region: str
    hardware: HardwareSpec


# =============================================================================
# Task Models
# =============================================================================

class TrainingTask(BaseModel):
    """A unit of training work to be distributed."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str  # Parent training job
    task_type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    
    # Task specification
    model_id: str
    model_shard_url: Optional[str] = None  # IPFS URL for model weights
    data_shard_url: str  # IPFS URL for training data
    batch_indices: List[int] = Field(default_factory=list)
    
    # Hyperparameters for this task
    learning_rate: float = 1e-4
    batch_size: int = 32
    gradient_accumulation: int = 1
    
    # Assignment
    assigned_worker_id: Optional[str] = None
    assigned_at: Optional[datetime] = None
    
    # Results
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result_url: Optional[str] = None  # IPFS URL for gradients/results
    loss: Optional[float] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    
    # Compute tracking
    compute_time_seconds: float = Field(default=0.0, ge=0)
    gpu_hours: float = Field(default=0.0, ge=0)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TrainingJob(BaseModel):
    """A complete training job consisting of multiple tasks."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    model_id: str
    
    # Job configuration
    total_epochs: int = Field(ge=1)
    current_epoch: int = Field(default=0, ge=0)
    total_steps: int = Field(ge=1)
    current_step: int = Field(default=0, ge=0)
    
    # Status
    status: TaskStatus = TaskStatus.PENDING
    progress_percent: float = Field(default=0.0, ge=0, le=100)
    
    # Metrics
    current_loss: Optional[float] = None
    best_loss: Optional[float] = None
    metrics_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Token allocation for this job
    total_token_reward: Decimal = Field(default=Decimal("0"))


# =============================================================================
# Contribution & Token Models
# =============================================================================

class ComputeContribution(BaseModel):
    """Record of compute contribution from a worker."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    worker_id: str
    wallet_address: str
    task_id: str
    job_id: str
    
    # Compute metrics
    compute_time_seconds: float = Field(ge=0)
    gpu_hours: float = Field(default=0.0, ge=0)
    cpu_hours: float = Field(default=0.0, ge=0)
    
    # Quality metrics
    gradient_quality_score: float = Field(default=1.0, ge=0, le=1)
    task_success: bool = True
    
    # Token calculation
    base_tokens: Decimal = Field(default=Decimal("0"))
    bonus_tokens: Decimal = Field(default=Decimal("0"))  # For high quality/speed
    total_tokens: Decimal = Field(default=Decimal("0"))
    
    # Verification
    verified: bool = False
    verification_hash: Optional[str] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TokenBalance(BaseModel):
    """Token balance for a wallet."""
    wallet_address: str
    balance: Decimal = Field(default=Decimal("0"))
    pending_balance: Decimal = Field(default=Decimal("0"))  # Earned but not paid out
    total_earned: Decimal = Field(default=Decimal("0"))
    total_withdrawn: Decimal = Field(default=Decimal("0"))
    last_payout_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Payout(BaseModel):
    """A crypto payout to a contributor."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    wallet_address: str
    amount: Decimal
    status: PayoutStatus = PayoutStatus.PENDING
    
    # Blockchain transaction
    tx_hash: Optional[str] = None
    block_number: Optional[int] = None
    gas_used: Optional[int] = None
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None
    confirmed_at: Optional[datetime] = None
    
    # Error handling
    retry_count: int = Field(default=0, ge=0)
    error_message: Optional[str] = None


# =============================================================================
# Model Registry
# =============================================================================

class ModelInfo(BaseModel):
    """Information about a trained model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    model_type: str  # e.g., "transformer", "cnn", "diffusion"
    architecture: str  # e.g., "gpt2", "llama", "stable-diffusion"
    
    # Model specs
    parameter_count: int = Field(ge=0)
    file_size_bytes: int = Field(ge=0)
    
    # Storage
    ipfs_hash: Optional[str] = None
    checkpoint_url: Optional[str] = None
    
    # Training info
    training_job_id: Optional[str] = None
    training_steps: int = Field(default=0, ge=0)
    final_loss: Optional[float] = None
    
    # Versioning
    version: str = "1.0.0"
    parent_model_id: Optional[str] = None
    
    # Access control
    is_public: bool = True
    api_calls: int = Field(default=0, ge=0)
    
    # Revenue tracking
    total_revenue: Decimal = Field(default=Decimal("0"))
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# API Models
# =============================================================================

class APIKey(BaseModel):
    """API key for accessing trained models."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    key_hash: str  # Hashed API key
    name: str
    wallet_address: str  # Owner's wallet
    
    # Permissions
    allowed_models: List[str] = Field(default_factory=list)  # Empty = all models
    rate_limit_per_minute: int = Field(default=60, ge=1)
    
    # Usage tracking
    total_requests: int = Field(default=0, ge=0)
    total_tokens_used: int = Field(default=0, ge=0)
    
    # Status
    is_active: bool = True
    expires_at: Optional[datetime] = None
    
    created_at: datetime = Field(default_factory=datetime.utcnow)


class InferenceRequest(BaseModel):
    """Request for model inference."""
    model_id: str
    inputs: Dict[str, Any]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    stream: bool = False


class InferenceResponse(BaseModel):
    """Response from model inference."""
    request_id: str
    model_id: str
    outputs: Dict[str, Any]
    usage: Dict[str, int] = Field(default_factory=dict)
    processing_time_ms: float
