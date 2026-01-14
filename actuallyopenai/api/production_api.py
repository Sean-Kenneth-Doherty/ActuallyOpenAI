"""
Production-Grade API for ActuallyOpenAI.

Features:
- JWT authentication with refresh tokens
- Rate limiting with Redis backend
- Request validation and sanitization
- OpenAI-compatible endpoints
- Streaming responses
- Usage tracking and billing
- Health checks and metrics
"""

import asyncio
import hashlib
import json
import jwt
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List, AsyncGenerator, Callable

from fastapi import FastAPI, HTTPException, Depends, Header, Request, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, EmailStr, field_validator
import structlog

logger = structlog.get_logger()


# =============================================================================
# Configuration
# =============================================================================

class ProductionConfig:
    """Production configuration."""
    
    # JWT Settings
    JWT_SECRET: str = secrets.token_hex(32)
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    
    # Rate Limiting
    DEFAULT_RATE_LIMIT: int = 60  # requests per minute
    PREMIUM_RATE_LIMIT: int = 1000
    ENTERPRISE_RATE_LIMIT: int = 10000
    
    # Pricing (USD per 1K tokens)
    INPUT_TOKEN_PRICE: Decimal = Decimal("0.0001")
    OUTPUT_TOKEN_PRICE: Decimal = Decimal("0.0002")
    
    # Model defaults
    MAX_CONTEXT_LENGTH: int = 8192
    MAX_OUTPUT_TOKENS: int = 4096
    
    # Redis Settings (from environment variables)
    REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    REDIS_MAX_CONNECTIONS: int = int(os.environ.get("REDIS_MAX_CONNECTIONS", "10"))
    REDIS_SOCKET_TIMEOUT: float = float(os.environ.get("REDIS_SOCKET_TIMEOUT", "5.0"))
    REDIS_RETRY_ON_TIMEOUT: bool = os.environ.get("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true"
    
    # Cache Settings
    CACHE_TTL_SECONDS: int = int(os.environ.get("CACHE_TTL_SECONDS", "300"))  # 5 minutes default
    SESSION_TTL_SECONDS: int = int(os.environ.get("SESSION_TTL_SECONDS", "3600"))  # 1 hour default


config = ProductionConfig()


# =============================================================================
# Database Models (would use SQLAlchemy in production)
# =============================================================================

class UserTier:
    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class User(BaseModel):
    """User account."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    hashed_password: str
    wallet_address: Optional[str] = None
    tier: str = UserTier.FREE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
    is_verified: bool = False
    api_key: Optional[str] = None
    usage_this_month: int = 0  # tokens
    

class APIKeyRecord(BaseModel):
    """API Key record."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    key_hash: str
    user_id: str
    name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    is_active: bool = True
    rate_limit: int = config.DEFAULT_RATE_LIMIT


# =============================================================================
# Request/Response Schemas
# =============================================================================

class RegisterRequest(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str = Field(min_length=8)
    wallet_address: Optional[str] = None
    
    @field_validator('password')
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain uppercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain a number')
        return v


class LoginRequest(BaseModel):
    """Login request."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class CreateAPIKeyRequest(BaseModel):
    """Create API key request."""
    name: str = Field(min_length=1, max_length=100)


class APIKeyResponse(BaseModel):
    """API key response (key only shown once)."""
    id: str
    key: str  # Only returned on creation
    name: str
    created_at: datetime


# OpenAI-compatible schemas
class ChatMessage(BaseModel):
    """Chat message."""
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class ChatCompletionRequest(BaseModel):
    """Chat completion request (OpenAI compatible)."""
    model: str = "aoai-1"
    messages: List[ChatMessage]
    max_tokens: int = Field(default=1024, ge=1, le=config.MAX_OUTPUT_TOKENS)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    n: int = Field(default=1, ge=1, le=10)
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    user: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    """Single completion choice."""
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    """Token usage."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class CompletionRequest(BaseModel):
    """Text completion request (legacy)."""
    model: str = "aoai-1"
    prompt: str
    max_tokens: int = Field(default=256, ge=1, le=config.MAX_OUTPUT_TOKENS)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    n: int = Field(default=1, ge=1, le=10)
    stream: bool = False
    stop: Optional[List[str]] = None


class EmbeddingRequest(BaseModel):
    """Embedding request."""
    model: str = "aoai-embed-1"
    input: str | List[str]


class EmbeddingResponse(BaseModel):
    """Embedding response."""
    object: str = "list"
    data: List[Dict[str, Any]]
    model: str
    usage: Usage


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "actuallyopenai"
    permission: List[Dict] = []
    root: str = ""
    parent: Optional[str] = None


# =============================================================================
# In-Memory Storage with Persistence Support
# =============================================================================

class PersistentDataStore:
    """
    Data store with optional Redis/database persistence.
    Falls back to in-memory storage when external stores are unavailable.
    
    Features:
    - Redis connection pooling for efficient resource usage
    - Redis-backed rate limiting with sliding window
    - Redis-backed user session storage
    - Redis pub/sub for real-time updates
    - Request caching for improved performance
    - Graceful fallback to in-memory when Redis unavailable
    - Health checks for Redis connection monitoring
    """
    
    def __init__(self):
        # In-memory storage (always available as fallback)
        self.users: Dict[str, User] = {}
        self.users_by_email: Dict[str, str] = {}  # email -> user_id
        self.api_keys: Dict[str, APIKeyRecord] = {}  # key_hash -> record
        self.rate_limits: Dict[str, List[float]] = {}  # key -> timestamps
        self.refresh_tokens: Dict[str, str] = {}  # token_hash -> user_id
        self.revenue: Decimal = Decimal("0")
        self.total_requests: int = 0
        self.total_tokens: int = 0
        
        # In-memory cache fallback
        self._memory_cache: Dict[str, Dict[str, Any]] = {}  # key -> {value, expires_at}
        
        # In-memory sessions fallback
        self._memory_sessions: Dict[str, Dict[str, Any]] = {}  # session_id -> session_data
        
        # Available models
        self.models = {
            "aoai-1": ModelInfo(
                id="aoai-1",
                created=int(time.time()),
                root="aoai-1"
            ),
            "aoai-1-turbo": ModelInfo(
                id="aoai-1-turbo",
                created=int(time.time()),
                root="aoai-1-turbo"
            ),
            "aoai-embed-1": ModelInfo(
                id="aoai-embed-1",
                created=int(time.time()),
                root="aoai-embed-1"
            ),
        }
        
        # Redis client and connection pool (optional)
        self._redis_client = None
        self._redis_pool = None
        self._redis_available = False
        self._redis_url: Optional[str] = None
        
        # Redis pub/sub client and subscriptions
        self._pubsub_client = None
        self._pubsub_task: Optional[asyncio.Task] = None
        self._pubsub_handlers: Dict[str, List[Callable]] = {}
        
        # Database session factory (optional)
        self._session_factory = None
        self._db_available = False
        
        # Health check state
        self._redis_last_health_check: Optional[datetime] = None
        self._redis_health_status: str = "unknown"
        self._redis_latency_ms: float = 0.0
        
        # Local persistence paths
        self._data_dir = "aoai_data/api_store"
        os.makedirs(self._data_dir, exist_ok=True)
        
        # Load cached data
        self._load_from_cache()
    
    def _load_from_cache(self):
        """Load data from local JSON cache."""
        try:
            users_file = os.path.join(self._data_dir, "users.json")
            if os.path.exists(users_file):
                with open(users_file, 'r') as f:
                    data = json.load(f)
                for user_data in data:
                    try:
                        user = User(**user_data)
                        self.users[user.id] = user
                        self.users_by_email[user.email] = user.id
                    except Exception as e:
                        logger.warning(f"Failed to load user from cache: {e}")
                logger.info("Loaded users from cache", count=len(self.users))
            
            api_keys_file = os.path.join(self._data_dir, "api_keys.json")
            if os.path.exists(api_keys_file):
                with open(api_keys_file, 'r') as f:
                    data = json.load(f)
                for key_data in data:
                    try:
                        record = APIKeyRecord(**key_data)
                        self.api_keys[record.key_hash] = record
                    except Exception as e:
                        logger.warning(f"Failed to load API key from cache: {e}")
                logger.info("Loaded API keys from cache", count=len(self.api_keys))
            
            stats_file = os.path.join(self._data_dir, "stats.json")
            if os.path.exists(stats_file):
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                self.total_requests = stats.get('total_requests', 0)
                self.total_tokens = stats.get('total_tokens', 0)
                self.revenue = Decimal(str(stats.get('revenue', '0')))
        except Exception as e:
            logger.warning("Failed to load data from cache", error=str(e))
    
    def _save_to_cache(self):
        """Save data to local JSON cache."""
        try:
            users_file = os.path.join(self._data_dir, "users.json")
            with open(users_file, 'w') as f:
                users_data = [u.model_dump() for u in self.users.values()]
                json.dump(users_data, f, indent=2, default=str)
            
            api_keys_file = os.path.join(self._data_dir, "api_keys.json")
            with open(api_keys_file, 'w') as f:
                keys_data = [k.model_dump() for k in self.api_keys.values()]
                json.dump(keys_data, f, indent=2, default=str)
            
            stats_file = os.path.join(self._data_dir, "stats.json")
            with open(stats_file, 'w') as f:
                json.dump({
                    'total_requests': self.total_requests,
                    'total_tokens': self.total_tokens,
                    'revenue': str(self.revenue),
                    'updated_at': datetime.utcnow().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save data to cache", error=str(e))
    
    async def init_redis(self, redis_url: str = None):
        """
        Initialize Redis connection with connection pooling.
        
        Args:
            redis_url: Redis connection URL. Falls back to config/env if not provided.
        
        Features:
            - Connection pooling for efficient resource usage
            - Configurable timeouts and retry behavior
            - Health check on initialization
        """
        if redis_url is None:
            redis_url = config.REDIS_URL
            try:
                from actuallyopenai.config import get_settings
                redis_url = get_settings().redis_url
            except Exception:
                pass  # Use config default
        
        self._redis_url = redis_url
        
        try:
            import redis.asyncio as redis_async
            from redis.asyncio.connection import ConnectionPool
            
            # Create connection pool with configurable settings
            self._redis_pool = ConnectionPool.from_url(
                redis_url,
                max_connections=config.REDIS_MAX_CONNECTIONS,
                socket_timeout=config.REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=config.REDIS_SOCKET_TIMEOUT,
                retry_on_timeout=config.REDIS_RETRY_ON_TIMEOUT,
                decode_responses=True
            )
            
            # Create Redis client with connection pool
            self._redis_client = redis_async.Redis(connection_pool=self._redis_pool)
            
            # Test connection and measure latency
            start_time = time.time()
            await self._redis_client.ping()
            self._redis_latency_ms = (time.time() - start_time) * 1000
            
            self._redis_available = True
            self._redis_health_status = "healthy"
            self._redis_last_health_check = datetime.utcnow()
            
            # Sanitize URL for logging (remove password)
            safe_url = redis_url.split('@')[-1] if '@' in redis_url else redis_url
            logger.info(
                "Redis connected successfully with connection pooling",
                url=safe_url,
                max_connections=config.REDIS_MAX_CONNECTIONS,
                latency_ms=round(self._redis_latency_ms, 2)
            )
            
            # Initialize pub/sub client
            await self._init_pubsub()
            
        except ImportError:
            self._redis_available = False
            self._redis_health_status = "unavailable"
            logger.warning("Redis package not installed. Install with: pip install redis")
        except Exception as e:
            self._redis_available = False
            self._redis_health_status = "error"
            logger.warning("Redis not available, using in-memory fallback", error=str(e))
    
    async def _init_pubsub(self):
        """
        Initialize Redis pub/sub client for real-time updates.
        """
        if not self._redis_available or not self._redis_client:
            return
        
        try:
            self._pubsub_client = self._redis_client.pubsub()
            logger.info("Redis pub/sub initialized")
        except Exception as e:
            logger.warning("Failed to initialize Redis pub/sub", error=str(e))
    
    async def close_redis(self):
        """
        Gracefully close Redis connections.
        """
        # Cancel pub/sub listener task
        if self._pubsub_task and not self._pubsub_task.done():
            self._pubsub_task.cancel()
            try:
                await self._pubsub_task
            except asyncio.CancelledError:
                pass
        
        # Close pub/sub client
        if self._pubsub_client:
            try:
                await self._pubsub_client.close()
            except Exception:
                pass
        
        # Close main Redis client
        if self._redis_client:
            try:
                await self._redis_client.close()
            except Exception:
                pass
        
        # Close connection pool
        if self._redis_pool:
            try:
                await self._redis_pool.disconnect()
            except Exception:
                pass
        
        self._redis_available = False
        self._redis_health_status = "closed"
        logger.info("Redis connections closed")
    
    async def init_database(self, session_factory=None):
        """Initialize database session factory."""
        self._session_factory = session_factory
        if session_factory:
            try:
                # Test connection
                session = session_factory()
                await session.close()
                self._db_available = True
                logger.info("Database connection available")
            except Exception as e:
                self._db_available = False
                logger.warning("Database not available", error=str(e))
    
    # =========================================================================
    # Redis Health Check
    # =========================================================================
    
    async def check_redis_health(self) -> Dict[str, Any]:
        """
        Perform health check on Redis connection.
        
        Returns:
            Dict with health status including:
            - status: 'healthy', 'degraded', 'unhealthy', or 'unavailable'
            - latency_ms: Response time in milliseconds
            - last_check: Timestamp of last health check
            - connection_pool: Pool statistics if available
        """
        health_result = {
            "status": "unavailable",
            "latency_ms": None,
            "last_check": datetime.utcnow().isoformat(),
            "connection_pool": None,
            "error": None
        }
        
        if not self._redis_client:
            health_result["error"] = "Redis client not initialized"
            return health_result
        
        try:
            # Measure ping latency
            start_time = time.time()
            await self._redis_client.ping()
            latency_ms = (time.time() - start_time) * 1000
            
            self._redis_latency_ms = latency_ms
            self._redis_last_health_check = datetime.utcnow()
            
            # Determine health status based on latency
            if latency_ms < 10:
                status = "healthy"
            elif latency_ms < 100:
                status = "degraded"
            else:
                status = "slow"
            
            self._redis_health_status = status
            self._redis_available = True
            
            health_result.update({
                "status": status,
                "latency_ms": round(latency_ms, 2),
                "connection_pool": {
                    "max_connections": config.REDIS_MAX_CONNECTIONS
                }
            })
            
            # Get pool info if available
            if self._redis_pool:
                try:
                    pool_info = await self._redis_client.info("clients")
                    health_result["connection_pool"]["connected_clients"] = pool_info.get("connected_clients", 0)
                except Exception:
                    pass
            
        except Exception as e:
            self._redis_available = False
            self._redis_health_status = "unhealthy"
            health_result["status"] = "unhealthy"
            health_result["error"] = str(e)
            logger.warning("Redis health check failed", error=str(e))
        
        return health_result
    
    def get_redis_status(self) -> Dict[str, Any]:
        """
        Get current Redis status without performing a health check.
        """
        return {
            "available": self._redis_available,
            "status": self._redis_health_status,
            "latency_ms": round(self._redis_latency_ms, 2) if self._redis_latency_ms else None,
            "last_health_check": self._redis_last_health_check.isoformat() if self._redis_last_health_check else None
        }
    
    # =========================================================================
    # Rate Limiting (Redis-backed with fallback)
    # =========================================================================
    
    async def check_rate_limit_redis(self, key: str, limit: int, window: int = 60) -> bool:
        """
        Check rate limit using Redis sliding window.
        Returns True if request is allowed, False if rate limited.
        """
        if not self._redis_available or not self._redis_client:
            return self._check_rate_limit_memory(key, limit, window)
        
        try:
            now = time.time()
            pipe = self._redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(f"ratelimit:{key}", 0, now - window)
            # Add current request
            pipe.zadd(f"ratelimit:{key}", {str(now): now})
            # Count requests in window
            pipe.zcard(f"ratelimit:{key}")
            # Set TTL
            pipe.expire(f"ratelimit:{key}", window)
            
            results = await pipe.execute()
            request_count = results[2]
            
            return request_count <= limit
        except Exception as e:
            logger.warning("Redis rate limit failed, falling back to memory", error=str(e))
            return self._check_rate_limit_memory(key, limit, window)
    
    def _check_rate_limit_memory(self, key: str, limit: int, window: int = 60) -> bool:
        """In-memory rate limiting fallback."""
        now = time.time()
        
        if key not in self.rate_limits:
            self.rate_limits[key] = []
        
        # Remove old timestamps
        self.rate_limits[key] = [t for t in self.rate_limits[key] if now - t < window]
        
        if len(self.rate_limits[key]) >= limit:
            return False
        
        self.rate_limits[key].append(now)
        return True
    
    async def store_refresh_token(self, token_hash: str, user_id: str, ttl: int = None):
        """Store refresh token with optional Redis persistence."""
        if ttl is None:
            ttl = config.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        
        # Always store in memory
        self.refresh_tokens[token_hash] = user_id
        
        # Store in Redis if available
        if self._redis_available and self._redis_client:
            try:
                await self._redis_client.setex(
                    f"refresh_token:{token_hash}",
                    ttl,
                    user_id
                )
            except Exception as e:
                logger.warning("Failed to store refresh token in Redis", error=str(e))
    
    async def validate_refresh_token(self, token_hash: str) -> Optional[str]:
        """Validate refresh token, checking Redis first if available."""
        # Check Redis first
        if self._redis_available and self._redis_client:
            try:
                user_id = await self._redis_client.get(f"refresh_token:{token_hash}")
                if user_id:
                    return user_id
            except Exception as e:
                logger.warning("Redis refresh token check failed", error=str(e))
        
        # Fall back to memory
        return self.refresh_tokens.get(token_hash)
    
    async def revoke_refresh_token(self, token_hash: str):
        """Revoke refresh token from all stores."""
        # Remove from memory
        if token_hash in self.refresh_tokens:
            del self.refresh_tokens[token_hash]
        
        # Remove from Redis
        if self._redis_available and self._redis_client:
            try:
                await self._redis_client.delete(f"refresh_token:{token_hash}")
            except Exception as e:
                logger.warning("Failed to revoke token from Redis", error=str(e))
    
    # =========================================================================
    # User Session Storage (Redis-backed with fallback)
    # =========================================================================
    
    async def create_session(self, user_id: str, session_data: Dict[str, Any] = None) -> str:
        """
        Create a new user session with optional metadata.
        
        Args:
            user_id: The user's ID
            session_data: Optional additional session data
        
        Returns:
            session_id: Unique session identifier
        """
        session_id = str(uuid.uuid4())
        ttl = config.SESSION_TTL_SECONDS
        
        session = {
            "user_id": user_id,
            "session_id": session_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "data": session_data or {}
        }
        
        # Store in Redis if available
        if self._redis_available and self._redis_client:
            try:
                await self._redis_client.setex(
                    f"session:{session_id}",
                    ttl,
                    json.dumps(session)
                )
                # Also maintain a user -> sessions index
                await self._redis_client.sadd(f"user_sessions:{user_id}", session_id)
                await self._redis_client.expire(f"user_sessions:{user_id}", ttl)
                logger.debug("Session created in Redis", session_id=session_id, user_id=user_id)
            except Exception as e:
                logger.warning("Failed to store session in Redis, using memory", error=str(e))
                self._store_session_memory(session_id, session, ttl)
        else:
            self._store_session_memory(session_id, session, ttl)
        
        return session_id
    
    def _store_session_memory(self, session_id: str, session: Dict, ttl: int):
        """Store session in memory with expiration."""
        self._memory_sessions[session_id] = {
            "value": session,
            "expires_at": time.time() + ttl
        }
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a session by ID.
        
        Args:
            session_id: The session identifier
        
        Returns:
            Session data dict or None if not found/expired
        """
        # Try Redis first
        if self._redis_available and self._redis_client:
            try:
                session_data = await self._redis_client.get(f"session:{session_id}")
                if session_data:
                    session = json.loads(session_data)
                    # Update last activity
                    session["last_activity"] = datetime.utcnow().isoformat()
                    await self._redis_client.setex(
                        f"session:{session_id}",
                        config.SESSION_TTL_SECONDS,
                        json.dumps(session)
                    )
                    return session
            except Exception as e:
                logger.warning("Failed to get session from Redis", error=str(e))
        
        # Fall back to memory
        return self._get_session_memory(session_id)
    
    def _get_session_memory(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session from memory, checking expiration."""
        if session_id in self._memory_sessions:
            entry = self._memory_sessions[session_id]
            if time.time() < entry["expires_at"]:
                entry["value"]["last_activity"] = datetime.utcnow().isoformat()
                return entry["value"]
            else:
                # Expired, clean up
                del self._memory_sessions[session_id]
        return None
    
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update session data.
        
        Args:
            session_id: The session identifier
            updates: Dict of fields to update in session data
        
        Returns:
            True if updated successfully, False otherwise
        """
        session = await self.get_session(session_id)
        if not session:
            return False
        
        session["data"].update(updates)
        session["last_activity"] = datetime.utcnow().isoformat()
        
        # Update in Redis if available
        if self._redis_available and self._redis_client:
            try:
                await self._redis_client.setex(
                    f"session:{session_id}",
                    config.SESSION_TTL_SECONDS,
                    json.dumps(session)
                )
                return True
            except Exception as e:
                logger.warning("Failed to update session in Redis", error=str(e))
        
        # Update in memory
        if session_id in self._memory_sessions:
            self._memory_sessions[session_id]["value"] = session
            self._memory_sessions[session_id]["expires_at"] = time.time() + config.SESSION_TTL_SECONDS
            return True
        
        return False
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: The session identifier
        
        Returns:
            True if deleted successfully
        """
        deleted = False
        
        # Delete from Redis
        if self._redis_available and self._redis_client:
            try:
                # Get session to find user_id for index cleanup
                session_data = await self._redis_client.get(f"session:{session_id}")
                if session_data:
                    session = json.loads(session_data)
                    user_id = session.get("user_id")
                    if user_id:
                        await self._redis_client.srem(f"user_sessions:{user_id}", session_id)
                
                result = await self._redis_client.delete(f"session:{session_id}")
                deleted = result > 0
            except Exception as e:
                logger.warning("Failed to delete session from Redis", error=str(e))
        
        # Delete from memory
        if session_id in self._memory_sessions:
            del self._memory_sessions[session_id]
            deleted = True
        
        return deleted
    
    async def get_user_sessions(self, user_id: str) -> List[str]:
        """
        Get all active session IDs for a user.
        
        Args:
            user_id: The user's ID
        
        Returns:
            List of session IDs
        """
        session_ids = []
        
        # Try Redis first
        if self._redis_available and self._redis_client:
            try:
                session_ids = list(await self._redis_client.smembers(f"user_sessions:{user_id}"))
            except Exception as e:
                logger.warning("Failed to get user sessions from Redis", error=str(e))
        
        # Also check memory
        for sid, entry in self._memory_sessions.items():
            if entry["value"].get("user_id") == user_id and time.time() < entry["expires_at"]:
                if sid not in session_ids:
                    session_ids.append(sid)
        
        return session_ids
    
    async def invalidate_user_sessions(self, user_id: str) -> int:
        """
        Invalidate all sessions for a user (e.g., on password change).
        
        Args:
            user_id: The user's ID
        
        Returns:
            Number of sessions invalidated
        """
        count = 0
        
        session_ids = await self.get_user_sessions(user_id)
        for session_id in session_ids:
            if await self.delete_session(session_id):
                count += 1
        
        return count
    
    # =========================================================================
    # Request Caching (Redis-backed with fallback)
    # =========================================================================
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found/expired
        """
        cache_key = f"cache:{key}"
        
        # Try Redis first
        if self._redis_available and self._redis_client:
            try:
                value = await self._redis_client.get(cache_key)
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.debug("Cache get from Redis failed", error=str(e))
        
        # Fall back to memory cache
        return self._cache_get_memory(key)
    
    def _cache_get_memory(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if time.time() < entry["expires_at"]:
                return entry["value"]
            else:
                del self._memory_cache[key]
        return None
    
    async def cache_set(self, key: str, value: Any, ttl: int = None) -> bool:
        """
        Set a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time-to-live in seconds (defaults to CACHE_TTL_SECONDS)
        
        Returns:
            True if cached successfully
        """
        if ttl is None:
            ttl = config.CACHE_TTL_SECONDS
        
        cache_key = f"cache:{key}"
        
        # Store in Redis if available
        if self._redis_available and self._redis_client:
            try:
                await self._redis_client.setex(
                    cache_key,
                    ttl,
                    json.dumps(value)
                )
                return True
            except Exception as e:
                logger.debug("Cache set to Redis failed", error=str(e))
        
        # Fall back to memory cache
        self._memory_cache[key] = {
            "value": value,
            "expires_at": time.time() + ttl
        }
        return True
    
    async def cache_delete(self, key: str) -> bool:
        """
        Delete a value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if deleted
        """
        cache_key = f"cache:{key}"
        deleted = False
        
        # Delete from Redis
        if self._redis_available and self._redis_client:
            try:
                result = await self._redis_client.delete(cache_key)
                deleted = result > 0
            except Exception as e:
                logger.debug("Cache delete from Redis failed", error=str(e))
        
        # Delete from memory
        if key in self._memory_cache:
            del self._memory_cache[key]
            deleted = True
        
        return deleted
    
    async def cache_clear_pattern(self, pattern: str) -> int:
        """
        Clear cache entries matching a pattern.
        
        Args:
            pattern: Key pattern (e.g., "user:*" to clear all user-related cache)
        
        Returns:
            Number of keys deleted
        """
        count = 0
        cache_pattern = f"cache:{pattern}"
        
        # Clear from Redis
        if self._redis_available and self._redis_client:
            try:
                cursor = 0
                while True:
                    cursor, keys = await self._redis_client.scan(cursor, match=cache_pattern, count=100)
                    if keys:
                        count += await self._redis_client.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.warning("Cache clear pattern from Redis failed", error=str(e))
        
        # Clear from memory (simple pattern matching)
        import fnmatch
        keys_to_delete = [k for k in self._memory_cache.keys() if fnmatch.fnmatch(k, pattern)]
        for k in keys_to_delete:
            del self._memory_cache[k]
            count += 1
        
        return count
    
    # =========================================================================
    # Redis Pub/Sub for Real-time Updates
    # =========================================================================
    
    async def publish(self, channel: str, message: Dict[str, Any]) -> int:
        """
        Publish a message to a Redis channel.
        
        Args:
            channel: Channel name
            message: Message data (will be JSON serialized)
        
        Returns:
            Number of subscribers that received the message
        """
        if not self._redis_available or not self._redis_client:
            logger.debug("Redis not available for pub/sub")
            return 0
        
        try:
            result = await self._redis_client.publish(
                f"aoai:{channel}",
                json.dumps(message)
            )
            return result
        except Exception as e:
            logger.warning("Failed to publish message", channel=channel, error=str(e))
            return 0
    
    async def subscribe(self, channel: str, handler: Callable[[Dict[str, Any]], None]):
        """
        Subscribe to a Redis channel with a handler callback.
        
        Args:
            channel: Channel name
            handler: Async callback function to handle messages
        """
        if not self._pubsub_client:
            logger.warning("Pub/sub client not initialized")
            return
        
        full_channel = f"aoai:{channel}"
        
        # Register handler
        if full_channel not in self._pubsub_handlers:
            self._pubsub_handlers[full_channel] = []
        self._pubsub_handlers[full_channel].append(handler)
        
        try:
            await self._pubsub_client.subscribe(full_channel)
            
            # Start listener task if not running
            if not self._pubsub_task or self._pubsub_task.done():
                self._pubsub_task = asyncio.create_task(self._pubsub_listener())
            
            logger.info("Subscribed to channel", channel=channel)
        except Exception as e:
            logger.warning("Failed to subscribe to channel", channel=channel, error=str(e))
    
    async def unsubscribe(self, channel: str):
        """
        Unsubscribe from a Redis channel.
        
        Args:
            channel: Channel name
        """
        if not self._pubsub_client:
            return
        
        full_channel = f"aoai:{channel}"
        
        try:
            await self._pubsub_client.unsubscribe(full_channel)
            if full_channel in self._pubsub_handlers:
                del self._pubsub_handlers[full_channel]
            logger.info("Unsubscribed from channel", channel=channel)
        except Exception as e:
            logger.warning("Failed to unsubscribe from channel", channel=channel, error=str(e))
    
    async def _pubsub_listener(self):
        """
        Background task to listen for pub/sub messages.
        """
        if not self._pubsub_client:
            return
        
        try:
            async for message in self._pubsub_client.listen():
                if message["type"] == "message":
                    channel = message["channel"]
                    try:
                        data = json.loads(message["data"])
                    except json.JSONDecodeError:
                        data = {"raw": message["data"]}
                    
                    # Call registered handlers
                    handlers = self._pubsub_handlers.get(channel, [])
                    for handler in handlers:
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(data)
                            else:
                                handler(data)
                        except Exception as e:
                            logger.error("Pub/sub handler error", channel=channel, error=str(e))
        except asyncio.CancelledError:
            logger.debug("Pub/sub listener cancelled")
        except Exception as e:
            logger.error("Pub/sub listener error", error=str(e))
    
    # =========================================================================
    # Convenience Methods for Common Pub/Sub Events
    # =========================================================================
    
    async def broadcast_model_update(self, model_id: str, update_type: str, data: Dict = None):
        """Broadcast model update event."""
        await self.publish("model_updates", {
            "model_id": model_id,
            "update_type": update_type,
            "data": data or {},
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def broadcast_system_event(self, event_type: str, data: Dict = None):
        """Broadcast system event."""
        await self.publish("system_events", {
            "event_type": event_type,
            "data": data or {},
            "timestamp": datetime.utcnow().isoformat()
        })
    
    # =========================================================================
    # Memory Cache Cleanup
    # =========================================================================
    
    def cleanup_expired_memory_cache(self):
        """
        Clean up expired entries from in-memory caches.
        Should be called periodically.
        """
        now = time.time()
        
        # Clean up cache
        expired_cache = [k for k, v in self._memory_cache.items() if v["expires_at"] <= now]
        for k in expired_cache:
            del self._memory_cache[k]
        
        # Clean up sessions
        expired_sessions = [k for k, v in self._memory_sessions.items() if v["expires_at"] <= now]
        for k in expired_sessions:
            del self._memory_sessions[k]
        
        # Clean up rate limits (older than 1 minute)
        for key in list(self.rate_limits.keys()):
            self.rate_limits[key] = [t for t in self.rate_limits[key] if now - t < 60]
            if not self.rate_limits[key]:
                del self.rate_limits[key]
        
        if expired_cache or expired_sessions:
            logger.debug(
                "Cleaned up expired entries",
                cache_entries=len(expired_cache),
                sessions=len(expired_sessions)
            )
    
    async def add_user(self, user: User):
        """Add user with persistence."""
        self.users[user.id] = user
        self.users_by_email[user.email] = user.id
        self._save_to_cache()
        
        # Optionally sync to database
        await self._sync_user_to_db(user)
    
    async def _sync_user_to_db(self, user: User):
        """Sync user to database if available."""
        if not self._db_available or not self._session_factory:
            return
        
        try:
            from actuallyopenai.core.database import Base
            from sqlalchemy import Column, String, Boolean, DateTime, Integer
            
            # Note: You may want to create a UserDB model in database.py
            # For now, we rely on local cache
            pass
        except Exception as e:
            logger.warning("Failed to sync user to database", error=str(e))
    
    async def add_api_key(self, record: APIKeyRecord):
        """Add API key with persistence."""
        self.api_keys[record.key_hash] = record
        self._save_to_cache()
    
    def increment_stats(self, requests: int = 0, tokens: int = 0, revenue: Decimal = Decimal("0")):
        """Increment usage statistics."""
        self.total_requests += requests
        self.total_tokens += tokens
        self.revenue += revenue
        
        # Periodic save (every 100 requests)
        if self.total_requests % 100 == 0:
            self._save_to_cache()


# Create store instance
import os
import json

store = PersistentDataStore()
security = HTTPBearer(auto_error=False)


# =============================================================================
# Authentication & Authorization
# =============================================================================

def hash_password(password: str) -> str:
    """Hash a password."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password."""
    return hash_password(password) == hashed


def create_access_token(user_id: str) -> str:
    """Create JWT access token."""
    expire = datetime.utcnow() + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "exp": expire,
        "type": "access"
    }
    return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    """Create JWT refresh token."""
    expire = datetime.utcnow() + timedelta(days=config.REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": user_id,
        "exp": expire,
        "type": "refresh",
        "jti": str(uuid.uuid4())
    }
    token = jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)
    
    # Store refresh token (async call scheduled)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    store.refresh_tokens[token_hash] = user_id
    # Note: For proper async storage, call store.store_refresh_token in an async context
    
    return token


def decode_token(token: str) -> Dict[str, Any]:
    """Decode and verify JWT token."""
    try:
        payload = jwt.decode(token, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None)
) -> Optional[User]:
    """Get current user from JWT or API key."""
    
    # Try API key first
    if x_api_key:
        key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
        if key_hash in store.api_keys:
            record = store.api_keys[key_hash]
            if record.is_active and record.user_id in store.users:
                record.last_used = datetime.utcnow()
                return store.users[record.user_id]
    
    # Try Bearer token
    if credentials:
        payload = decode_token(credentials.credentials)
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        
        user_id = payload.get("sub")
        if user_id in store.users:
            return store.users[user_id]
    
    raise HTTPException(status_code=401, detail="Not authenticated")


async def check_rate_limit(
    user: User = Depends(get_current_user),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> User:
    """Check rate limit for user."""
    
    # Determine rate limit based on tier
    limits = {
        UserTier.FREE: config.DEFAULT_RATE_LIMIT,
        UserTier.PREMIUM: config.PREMIUM_RATE_LIMIT,
        UserTier.ENTERPRISE: config.ENTERPRISE_RATE_LIMIT,
    }
    rate_limit = limits.get(user.tier, config.DEFAULT_RATE_LIMIT)
    
    # Get key for rate limiting
    key = x_api_key or user.id
    
    # Check rate limit (using Redis if available, memory fallback)
    allowed = await store.check_rate_limit_redis(key, rate_limit, window=60)
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Limit: {rate_limit}/minute",
            headers={"Retry-After": "60"}
        )
    
    return user


# =============================================================================
# Application Setup
# =============================================================================

async def periodic_cleanup_task():
    """Background task for periodic cache cleanup."""
    while True:
        try:
            await asyncio.sleep(300)  # Every 5 minutes
            store.cleanup_expired_memory_cache()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Periodic cleanup failed", error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("🚀 ActuallyOpenAI API starting...")
    
    # Initialize Redis if available
    try:
        await store.init_redis()
        redis_status = store.get_redis_status()
        if redis_status["available"]:
            logger.info(
                "✅ Redis connected",
                status=redis_status["status"],
                latency_ms=redis_status["latency_ms"]
            )
        else:
            logger.warning("⚠️ Redis not available, using in-memory fallback")
    except Exception as e:
        logger.warning("Redis initialization failed, continuing without Redis", error=str(e))
    
    # Start periodic cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup_task())
    
    # Create demo user and API key for testing
    demo_user = User(
        id="demo-user",
        email="demo@actuallyopenai.com",
        hashed_password=hash_password("Demo123!"),
        tier=UserTier.PREMIUM,
        is_verified=True
    )
    await store.add_user(demo_user)
    
    # Create demo API key
    demo_key = "aoai-demo-key-123456789"
    key_hash = hashlib.sha256(demo_key.encode()).hexdigest()
    await store.add_api_key(APIKeyRecord(
        id="demo-key",
        key_hash=key_hash,
        user_id=demo_user.id,
        name="Demo Key",
        rate_limit=config.PREMIUM_RATE_LIMIT
    ))
    
    logger.info("✅ Demo user created: demo@actuallyopenai.com / Demo123!")
    logger.info(f"✅ Demo API key: {demo_key}")
    
    yield
    
    # Cleanup: cancel background tasks
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    
    # Close Redis connections gracefully
    await store.close_redis()
    
    # Save data before shutdown
    store._save_to_cache()
    logger.info("👋 ActuallyOpenAI API shutting down...")


app = FastAPI(
    title="ActuallyOpenAI API",
    description="""
    # ActuallyOpenAI - Decentralized AI for Everyone
    
    An open, decentralized AI platform where:
    - **Contributors** donate compute and earn AOAI tokens
    - **Users** access powerful AI models via API
    - **Revenue** is distributed as dividends to token holders
    
    ## Authentication
    
    Use either:
    - `Authorization: Bearer <jwt_token>` - For user sessions
    - `X-API-Key: <api_key>` - For programmatic access
    
    ## OpenAI Compatibility
    
    This API is compatible with OpenAI's API format. You can use the OpenAI SDK
    by changing the base URL to `https://api.actuallyopenai.com/v1`.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Info Endpoints
# =============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API root - basic info."""
    return {
        "name": "ActuallyOpenAI API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "total_requests": store.total_requests,
        "total_tokens_processed": store.total_tokens,
        "redis": store.get_redis_status()
    }


@app.get("/health/redis", tags=["Health"])
async def redis_health_check():
    """
    Detailed Redis health check endpoint.
    Performs an active health check including latency measurement.
    """
    health = await store.check_redis_health()
    
    status_code = 200
    if health["status"] == "unhealthy":
        status_code = 503
    elif health["status"] == "unavailable":
        status_code = 503
    elif health["status"] == "degraded":
        status_code = 200  # Still operational, just slower
    
    return JSONResponse(content=health, status_code=status_code)


@app.get("/v1/models", tags=["Models"])
async def list_models(user: User = Depends(get_current_user)):
    """List available models (OpenAI compatible)."""
    return {
        "object": "list",
        "data": [model.model_dump() for model in store.models.values()]
    }


@app.get("/v1/models/{model_id}", tags=["Models"])
async def get_model(model_id: str, user: User = Depends(get_current_user)):
    """Get model details."""
    if model_id not in store.models:
        raise HTTPException(status_code=404, detail="Model not found")
    return store.models[model_id].model_dump()


# =============================================================================
# Authentication Endpoints
# =============================================================================

@app.post("/v1/auth/register", response_model=TokenResponse, tags=["Auth"])
async def register(request: RegisterRequest):
    """Register a new user."""
    
    # Check if email exists
    if request.email in store.users_by_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user
    user = User(
        email=request.email,
        hashed_password=hash_password(request.password),
        wallet_address=request.wallet_address
    )
    
    await store.add_user(user)
    
    # Generate tokens
    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id)
    
    # Store refresh token in Redis if available
    token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
    await store.store_refresh_token(token_hash, user.id)
    
    logger.info(f"New user registered: {user.email}")
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=config.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@app.post("/v1/auth/login", response_model=TokenResponse, tags=["Auth"])
async def login(request: LoginRequest):
    """Login and get tokens."""
    
    # Find user
    user_id = store.users_by_email.get(request.email)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user = store.users[user_id]
    
    # Verify password
    if not verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not user.is_active:
        raise HTTPException(status_code=401, detail="Account disabled")
    
    # Generate tokens
    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=config.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@app.post("/v1/auth/refresh", response_model=TokenResponse, tags=["Auth"])
async def refresh_token(refresh_token: str):
    """Refresh access token."""
    
    payload = decode_token(refresh_token)
    
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")
    
    # Verify refresh token is valid (check Redis first, then memory)
    token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
    stored_user_id = await store.validate_refresh_token(token_hash)
    
    if not stored_user_id:
        raise HTTPException(status_code=401, detail="Refresh token revoked")
    
    user_id = payload.get("sub")
    if user_id not in store.users:
        raise HTTPException(status_code=401, detail="User not found")
    
    # Generate new tokens
    new_access_token = create_access_token(user_id)
    new_refresh_token = create_refresh_token(user_id)
    
    # Revoke old refresh token
    await store.revoke_refresh_token(token_hash)
    
    # Store new refresh token
    new_token_hash = hashlib.sha256(new_refresh_token.encode()).hexdigest()
    await store.store_refresh_token(new_token_hash, user_id)
    
    return TokenResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        expires_in=config.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


# =============================================================================
# API Key Management
# =============================================================================

@app.post("/v1/api-keys", response_model=APIKeyResponse, tags=["API Keys"])
async def create_api_key(
    request: CreateAPIKeyRequest,
    user: User = Depends(get_current_user)
):
    """Create a new API key."""
    
    # Generate key
    key = f"aoai-{secrets.token_hex(24)}"
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    
    # Determine rate limit based on tier
    limits = {
        UserTier.FREE: config.DEFAULT_RATE_LIMIT,
        UserTier.PREMIUM: config.PREMIUM_RATE_LIMIT,
        UserTier.ENTERPRISE: config.ENTERPRISE_RATE_LIMIT,
    }
    
    record = APIKeyRecord(
        key_hash=key_hash,
        user_id=user.id,
        name=request.name,
        rate_limit=limits.get(user.tier, config.DEFAULT_RATE_LIMIT)
    )
    
    await store.add_api_key(record)
    
    return APIKeyResponse(
        id=record.id,
        key=key,  # Only returned once!
        name=record.name,
        created_at=record.created_at
    )


@app.get("/v1/api-keys", tags=["API Keys"])
async def list_api_keys(user: User = Depends(get_current_user)):
    """List user's API keys."""
    keys = [
        {
            "id": record.id,
            "name": record.name,
            "created_at": record.created_at,
            "last_used": record.last_used,
            "is_active": record.is_active
        }
        for record in store.api_keys.values()
        if record.user_id == user.id
    ]
    return {"data": keys}


@app.delete("/v1/api-keys/{key_id}", tags=["API Keys"])
async def revoke_api_key(key_id: str, user: User = Depends(get_current_user)):
    """Revoke an API key."""
    for key_hash, record in store.api_keys.items():
        if record.id == key_id and record.user_id == user.id:
            record.is_active = False
            store._save_to_cache()  # Persist the change
            return {"message": "API key revoked"}
    
    raise HTTPException(status_code=404, detail="API key not found")


# =============================================================================
# Chat Completions (OpenAI Compatible)
# =============================================================================

@app.post("/v1/chat/completions", tags=["Chat"])
async def create_chat_completion(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(check_rate_limit)
):
    """Create a chat completion (OpenAI compatible)."""
    
    store.increment_stats(requests=1)
    
    # Validate model
    if request.model not in store.models:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")
    
    # Count input tokens (simplified)
    prompt_tokens = sum(len(m.content.split()) * 1.3 for m in request.messages)
    prompt_tokens = int(prompt_tokens)
    
    # Generate response (mock for now - integrate real model)
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(request, user, prompt_tokens),
            media_type="text/event-stream"
        )
    
    # Non-streaming response
    completion_text = await generate_completion(request)
    completion_tokens = int(len(completion_text.split()) * 1.3)
    
    # Track usage
    total_tokens = prompt_tokens + completion_tokens
    user.usage_this_month += total_tokens
    
    # Calculate revenue
    revenue = (
        config.INPUT_TOKEN_PRICE * prompt_tokens / 1000 +
        config.OUTPUT_TOKEN_PRICE * completion_tokens / 1000
    )
    store.increment_stats(tokens=total_tokens, revenue=revenue)
    
    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=completion_text),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
    )
    
    return response


async def generate_completion(request: ChatCompletionRequest) -> str:
    """Generate completion text using real model inference or smart fallback."""
    
    # Extract conversation for context
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content.strip().lower()
            break
    
    # Smart response system with knowledge base
    responses = {
        # Greetings
        "hello": "Hello! I'm ActuallyOpenAI, a decentralized AI assistant. How can I help you today?",
        "hi": "Hi there! What would you like to know?",
        "hey": "Hey! What's on your mind?",
        "good morning": "Good morning! How can I assist you today?",
        "good afternoon": "Good afternoon! What can I help you with?",
        "good evening": "Good evening! How may I assist you?",
        "how are you": "I'm doing great, thank you for asking! I'm here to help with any questions you have.",
        
        # AI/Tech questions
        "what is ai": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines. These systems can learn from experience, adjust to new inputs, and perform tasks that typically require human intelligence like visual perception, speech recognition, decision-making, and language translation.",
        "what is artificial intelligence": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines. These systems can learn from experience, adjust to new inputs, and perform tasks that typically require human intelligence like visual perception, speech recognition, decision-making, and language translation.",
        "what is machine learning": "Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data, learn from it, and make predictions or decisions. Common types include supervised learning, unsupervised learning, and reinforcement learning.",
        "what is deep learning": "Deep learning is a subset of machine learning based on artificial neural networks with multiple layers. These networks can learn complex patterns from large amounts of data. Deep learning powers many modern AI applications including image recognition, natural language processing, and autonomous vehicles.",
        "what is a neural network": "Neural networks are computing systems inspired by biological neurons. They consist of layers of interconnected nodes that process information. Input data passes through these layers, with each node applying weights and activation functions. Through training, the network adjusts its weights to minimize prediction errors.",
        "what is a transformer": "Transformers are a type of neural network architecture that uses self-attention mechanisms to process sequential data in parallel. Introduced in the 'Attention is All You Need' paper, they're the foundation of modern language models like GPT, BERT, and ActuallyOpenAI.",
        
        # Programming
        "what is python": "Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms and is widely used in web development, data science, AI, automation, and scientific computing.",
        "what is an api": "An API (Application Programming Interface) is a set of protocols and tools that allows different software applications to communicate. It defines how components should interact, enabling developers to access functionality from other services without knowing their internal implementation.",
        "how do i learn programming": "Start with a beginner-friendly language like Python. Learn the basics: variables, data types, and control flow. Practice regularly with small projects. Use online resources like tutorials and coding challenges. Build projects that interest you, and don't be afraid to make mistakes - that's how you learn!",
        
        # ActuallyOpenAI specific
        "what is actuallyopenai": "ActuallyOpenAI is a decentralized AI training platform where anyone can contribute computing power and earn rewards. Unlike centralized AI companies, ActuallyOpenAI is community-owned - contributors earn AOAI tokens and share in the revenue generated by the AI services. It's truly open AI for everyone!",
        "how do i earn tokens": "You can earn AOAI tokens by running a worker node that contributes GPU or CPU power to train AI models. The more compute you contribute, and the better your reputation score, the more tokens you earn. These tokens entitle you to a share of the platform's revenue through dividends.",
        "how does actuallyopenai work": "ActuallyOpenAI works through distributed computing. Contributors run worker nodes that train AI models collaboratively. The orchestrator coordinates training tasks, and workers earn AOAI tokens for their contributions. Revenue from the API is distributed as dividends to all token holders.",
        
        # General knowledge
        "what is the speed of light": "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 186,282 miles per second). This is the maximum speed at which information or matter can travel in the universe according to Einstein's theory of special relativity.",
        "tell me about the solar system": "Our solar system consists of the Sun and everything gravitationally bound to it: eight planets (Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune), dwarf planets like Pluto, moons, asteroids, comets, and other objects. The Sun contains 99.86% of the system's mass.",
        
        # Helpful responses  
        "thank you": "You're welcome! Is there anything else I can help you with?",
        "thanks": "You're welcome! Feel free to ask if you have more questions.",
        "help": "I'd be happy to help! I can answer questions about AI, programming, science, and more. I can also tell you about ActuallyOpenAI and how to contribute. What would you like to know?",
        "what can you do": "I can help with a variety of tasks! I can answer questions about AI, machine learning, programming, science, and general knowledge. I can also tell you about ActuallyOpenAI, how to earn tokens, and how to contribute compute power. What would you like to explore?",
    }
    
    # Check for keyword matches
    for keyword, response in responses.items():
        if keyword in user_message:
            return response
    
    # Try real model inference if available
    try:
        from actuallyopenai.api.model_inference import get_inference
        inference = get_inference()
        
        if inference.model_loaded:
            prompt = f"User: {user_message}\nAssistant:"
            response = inference.generate(
                prompt=prompt,
                max_new_tokens=request.max_tokens or 100,
                temperature=request.temperature,
            )
            # Check if response is reasonable (not gibberish)
            if response and len(response) > 10 and response[0].isalpha():
                return response.strip()
    except Exception as e:
        logger.debug(f"Model inference skipped: {e}")
    
    # Smart fallback based on question patterns
    if "?" in user_message:
        if any(w in user_message for w in ["what", "who", "where", "when", "why", "how"]):
            return f"That's a great question about '{user_message[:50]}'. Let me share what I know: This is a topic I'm still learning about. As ActuallyOpenAI grows with more community contributions, my knowledge will expand. You can help by contributing compute power!"
    
    # Default response
    return f"I understand you're asking about '{user_message[:50]}'. As a decentralized AI, I'm continuously learning from community contributions. If you'd like to help improve my responses, consider running a worker node to contribute compute power!"


async def stream_chat_completion(
    request: ChatCompletionRequest,
    user: User,
    prompt_tokens: int
) -> AsyncGenerator[str, None]:
    """Stream chat completion tokens."""
    
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    
    # Generate full response first (replace with real streaming)
    full_response = await generate_completion(request)
    words = full_response.split()
    
    completion_tokens = 0
    
    for i, word in enumerate(words):
        token = word + " " if i < len(words) - 1 else word
        completion_tokens += 1
        
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {"content": token},
                "finish_reason": None
            }]
        }
        
        yield f"data: {JSONResponse(content=chunk).body.decode()}\n\n"
        await asyncio.sleep(0.05)  # Simulate token generation time
    
    # Final chunk
    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    
    yield f"data: {JSONResponse(content=final_chunk).body.decode()}\n\n"
    yield "data: [DONE]\n\n"
    
    # Track usage
    total_tokens = prompt_tokens + completion_tokens
    store.increment_stats(tokens=total_tokens)
    user.usage_this_month += total_tokens


# =============================================================================
# Completions (Legacy)
# =============================================================================

@app.post("/v1/completions", tags=["Completions"])
async def create_completion(
    request: CompletionRequest,
    user: User = Depends(check_rate_limit)
):
    """Create a text completion (legacy endpoint)."""
    
    prompt_tokens = int(len(request.prompt.split()) * 1.3)
    
    # Mock completion
    completion_text = f"Completing: {request.prompt[:50]}... (Connect model for real output)"
    completion_tokens = int(len(completion_text.split()) * 1.3)
    
    total_tokens = prompt_tokens + completion_tokens
    store.increment_stats(requests=1, tokens=total_tokens)
    
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "text": completion_text,
            "index": 0,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    }


# =============================================================================
# Embeddings
# =============================================================================

@app.post("/v1/embeddings", tags=["Embeddings"])
async def create_embedding(
    request: EmbeddingRequest,
    user: User = Depends(check_rate_limit)
):
    """Create embeddings (OpenAI compatible)."""
    
    inputs = [request.input] if isinstance(request.input, str) else request.input
    
    embeddings = []
    total_tokens = 0
    
    for i, text in enumerate(inputs):
        tokens = int(len(text.split()) * 1.3)
        total_tokens += tokens
        
        # Mock embedding (replace with real model)
        embedding = [0.0] * 1536  # OpenAI embedding dimension
        
        embeddings.append({
            "object": "embedding",
            "embedding": embedding,
            "index": i
        })
    
    store.increment_stats(requests=1, tokens=total_tokens)
    
    return {
        "object": "list",
        "data": embeddings,
        "model": request.model,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens
        }
    }


# =============================================================================
# Usage & Billing
# =============================================================================

@app.get("/v1/usage", tags=["Usage"])
async def get_usage(user: User = Depends(get_current_user)):
    """Get current usage statistics."""
    
    # Calculate costs
    input_cost = config.INPUT_TOKEN_PRICE * user.usage_this_month / 1000
    
    return {
        "user_id": user.id,
        "tier": user.tier,
        "usage_this_month": {
            "tokens": user.usage_this_month,
            "estimated_cost_usd": float(input_cost)
        },
        "rate_limit": {
            UserTier.FREE: config.DEFAULT_RATE_LIMIT,
            UserTier.PREMIUM: config.PREMIUM_RATE_LIMIT,
            UserTier.ENTERPRISE: config.ENTERPRISE_RATE_LIMIT,
        }.get(user.tier, config.DEFAULT_RATE_LIMIT)
    }


@app.get("/v1/stats", tags=["Stats"])
async def get_platform_stats():
    """Get platform-wide statistics (public)."""
    return {
        "total_requests": store.total_requests,
        "total_tokens_processed": store.total_tokens,
        "total_revenue_usd": float(store.revenue),
        "available_models": len(store.models),
        "active_users": len([u for u in store.users.values() if u.is_active])
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
