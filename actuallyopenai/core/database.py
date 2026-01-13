"""
Database models and ORM setup for ActuallyOpenAI.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, 
    Numeric, Text, JSON, ForeignKey, Index, Enum as SQLEnum
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from actuallyopenai.core.models import (
    WorkerStatus, TaskStatus, TaskType, PayoutStatus
)

Base = declarative_base()


# =============================================================================
# Worker Tables
# =============================================================================

class WorkerDB(Base):
    """Database model for worker nodes."""
    __tablename__ = "workers"
    
    id = Column(String(36), primary_key=True)
    wallet_address = Column(String(42), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    region = Column(String(100), nullable=False)
    
    # Hardware specs (JSON)
    hardware = Column(JSON, nullable=False)
    
    # Status
    status = Column(SQLEnum(WorkerStatus), default=WorkerStatus.OFFLINE)
    
    # Scores
    compute_score = Column(Float, default=0.0)
    reliability_score = Column(Float, default=1.0)
    
    # Connection
    last_heartbeat = Column(DateTime, nullable=True)
    connected_at = Column(DateTime, nullable=True)
    ip_address = Column(String(45), nullable=True)
    
    # Stats
    total_tasks_completed = Column(Integer, default=0)
    total_compute_hours = Column(Float, default=0.0)
    total_tokens_earned = Column(Numeric(36, 18), default=Decimal("0"))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    contributions = relationship("ComputeContributionDB", back_populates="worker")
    
    __table_args__ = (
        Index("idx_worker_wallet", "wallet_address"),
        Index("idx_worker_status", "status"),
    )


# =============================================================================
# Training Tables
# =============================================================================

class TrainingJobDB(Base):
    """Database model for training jobs."""
    __tablename__ = "training_jobs"
    
    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    model_id = Column(String(36), ForeignKey("models.id"), nullable=False)
    
    # Configuration
    total_epochs = Column(Integer, nullable=False)
    current_epoch = Column(Integer, default=0)
    total_steps = Column(Integer, nullable=False)
    current_step = Column(Integer, default=0)
    
    # Status
    status = Column(SQLEnum(TaskStatus), default=TaskStatus.PENDING)
    progress_percent = Column(Float, default=0.0)
    
    # Metrics
    current_loss = Column(Float, nullable=True)
    best_loss = Column(Float, nullable=True)
    metrics_history = Column(JSON, default=list)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Token allocation
    total_token_reward = Column(Numeric(36, 18), default=Decimal("0"))
    
    # Relationships
    tasks = relationship("TrainingTaskDB", back_populates="job")
    model = relationship("ModelDB", back_populates="training_jobs")


class TrainingTaskDB(Base):
    """Database model for training tasks."""
    __tablename__ = "training_tasks"
    
    id = Column(String(36), primary_key=True)
    job_id = Column(String(36), ForeignKey("training_jobs.id"), nullable=False)
    task_type = Column(SQLEnum(TaskType), nullable=False)
    status = Column(SQLEnum(TaskStatus), default=TaskStatus.PENDING)
    
    # Task spec
    model_id = Column(String(36), nullable=False)
    model_shard_url = Column(String(500), nullable=True)
    data_shard_url = Column(String(500), nullable=False)
    batch_indices = Column(JSON, default=list)
    
    # Hyperparameters
    learning_rate = Column(Float, default=1e-4)
    batch_size = Column(Integer, default=32)
    gradient_accumulation = Column(Integer, default=1)
    
    # Assignment
    assigned_worker_id = Column(String(36), ForeignKey("workers.id"), nullable=True)
    assigned_at = Column(DateTime, nullable=True)
    
    # Results
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    result_url = Column(String(500), nullable=True)
    loss = Column(Float, nullable=True)
    metrics = Column(JSON, default=dict)
    error_message = Column(Text, nullable=True)
    
    # Compute tracking
    compute_time_seconds = Column(Float, default=0.0)
    gpu_hours = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    job = relationship("TrainingJobDB", back_populates="tasks")
    
    __table_args__ = (
        Index("idx_task_status", "status"),
        Index("idx_task_worker", "assigned_worker_id"),
    )


# =============================================================================
# Contribution & Token Tables
# =============================================================================

class ComputeContributionDB(Base):
    """Database model for compute contributions."""
    __tablename__ = "compute_contributions"
    
    id = Column(String(36), primary_key=True)
    worker_id = Column(String(36), ForeignKey("workers.id"), nullable=False)
    wallet_address = Column(String(42), nullable=False, index=True)
    task_id = Column(String(36), ForeignKey("training_tasks.id"), nullable=False)
    job_id = Column(String(36), ForeignKey("training_jobs.id"), nullable=False)
    
    # Compute metrics
    compute_time_seconds = Column(Float, nullable=False)
    gpu_hours = Column(Float, default=0.0)
    cpu_hours = Column(Float, default=0.0)
    
    # Quality
    gradient_quality_score = Column(Float, default=1.0)
    task_success = Column(Boolean, default=True)
    
    # Tokens
    base_tokens = Column(Numeric(36, 18), default=Decimal("0"))
    bonus_tokens = Column(Numeric(36, 18), default=Decimal("0"))
    total_tokens = Column(Numeric(36, 18), default=Decimal("0"))
    
    # Verification
    verified = Column(Boolean, default=False)
    verification_hash = Column(String(66), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    worker = relationship("WorkerDB", back_populates="contributions")
    
    __table_args__ = (
        Index("idx_contribution_wallet", "wallet_address"),
        Index("idx_contribution_created", "created_at"),
    )


class TokenBalanceDB(Base):
    """Database model for token balances."""
    __tablename__ = "token_balances"
    
    wallet_address = Column(String(42), primary_key=True)
    balance = Column(Numeric(36, 18), default=Decimal("0"))
    pending_balance = Column(Numeric(36, 18), default=Decimal("0"))
    total_earned = Column(Numeric(36, 18), default=Decimal("0"))
    total_withdrawn = Column(Numeric(36, 18), default=Decimal("0"))
    last_payout_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PayoutDB(Base):
    """Database model for payouts."""
    __tablename__ = "payouts"
    
    id = Column(String(36), primary_key=True)
    wallet_address = Column(String(42), nullable=False, index=True)
    amount = Column(Numeric(36, 18), nullable=False)
    status = Column(SQLEnum(PayoutStatus), default=PayoutStatus.PENDING)
    
    # Transaction
    tx_hash = Column(String(66), nullable=True, unique=True)
    block_number = Column(Integer, nullable=True)
    gas_used = Column(Integer, nullable=True)
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    confirmed_at = Column(DateTime, nullable=True)
    
    # Error handling
    retry_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    
    __table_args__ = (
        Index("idx_payout_status", "status"),
        Index("idx_payout_wallet", "wallet_address"),
    )


# =============================================================================
# Model Tables
# =============================================================================

class ModelDB(Base):
    """Database model for AI models."""
    __tablename__ = "models"
    
    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    model_type = Column(String(100), nullable=False)
    architecture = Column(String(100), nullable=False)
    
    # Specs
    parameter_count = Column(Integer, default=0)
    file_size_bytes = Column(Integer, default=0)
    
    # Storage
    ipfs_hash = Column(String(100), nullable=True)
    checkpoint_url = Column(String(500), nullable=True)
    
    # Training
    training_job_id = Column(String(36), nullable=True)
    training_steps = Column(Integer, default=0)
    final_loss = Column(Float, nullable=True)
    
    # Versioning
    version = Column(String(50), default="1.0.0")
    parent_model_id = Column(String(36), nullable=True)
    
    # Access
    is_public = Column(Boolean, default=True)
    api_calls = Column(Integer, default=0)
    
    # Revenue
    total_revenue = Column(Numeric(36, 18), default=Decimal("0"))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    training_jobs = relationship("TrainingJobDB", back_populates="model")


class APIKeyDB(Base):
    """Database model for API keys."""
    __tablename__ = "api_keys"
    
    id = Column(String(36), primary_key=True)
    key_hash = Column(String(64), nullable=False, unique=True)
    name = Column(String(255), nullable=False)
    wallet_address = Column(String(42), nullable=False, index=True)
    
    # Permissions
    allowed_models = Column(JSON, default=list)
    rate_limit_per_minute = Column(Integer, default=60)
    
    # Usage
    total_requests = Column(Integer, default=0)
    total_tokens_used = Column(Integer, default=0)
    
    # Status
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)


# =============================================================================
# Database Session Management
# =============================================================================

async def create_database_engine(database_url: str):
    """Create async database engine."""
    engine = create_async_engine(
        database_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20
    )
    return engine


async def create_tables(engine):
    """Create all database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def create_session_factory(engine):
    """Create async session factory."""
    return sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
