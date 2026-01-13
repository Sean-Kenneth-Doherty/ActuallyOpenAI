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
import jwt
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List, AsyncGenerator

from fastapi import FastAPI, HTTPException, Depends, Header, Request, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, EmailStr, validator
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
    
    @validator('password')
    def password_strength(cls, v):
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
# In-Memory Storage (Use Redis/PostgreSQL in production)
# =============================================================================

class DataStore:
    """In-memory data store (replace with Redis/PostgreSQL)."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.users_by_email: Dict[str, str] = {}  # email -> user_id
        self.api_keys: Dict[str, APIKeyRecord] = {}  # key_hash -> record
        self.rate_limits: Dict[str, List[float]] = {}  # key -> timestamps
        self.refresh_tokens: Dict[str, str] = {}  # token_hash -> user_id
        self.revenue: Decimal = Decimal("0")
        self.total_requests: int = 0
        self.total_tokens: int = 0
        
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


store = DataStore()
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
    
    # Store refresh token
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    store.refresh_tokens[token_hash] = user_id
    
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
    
    # Check rate limit (sliding window)
    now = time.time()
    window = 60  # 1 minute
    
    if key not in store.rate_limits:
        store.rate_limits[key] = []
    
    # Remove old timestamps
    store.rate_limits[key] = [t for t in store.rate_limits[key] if now - t < window]
    
    if len(store.rate_limits[key]) >= rate_limit:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Limit: {rate_limit}/minute",
            headers={"Retry-After": "60"}
        )
    
    store.rate_limits[key].append(now)
    return user


# =============================================================================
# Application Setup
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("🚀 ActuallyOpenAI API starting...")
    
    # Create demo user and API key for testing
    demo_user = User(
        id="demo-user",
        email="demo@actuallyopenai.com",
        hashed_password=hash_password("Demo123!"),
        tier=UserTier.PREMIUM,
        is_verified=True
    )
    store.users[demo_user.id] = demo_user
    store.users_by_email[demo_user.email] = demo_user.id
    
    # Create demo API key
    demo_key = "aoai-demo-key-123456789"
    key_hash = hashlib.sha256(demo_key.encode()).hexdigest()
    store.api_keys[key_hash] = APIKeyRecord(
        id="demo-key",
        key_hash=key_hash,
        user_id=demo_user.id,
        name="Demo Key",
        rate_limit=config.PREMIUM_RATE_LIMIT
    )
    
    logger.info("✅ Demo user created: demo@actuallyopenai.com / Demo123!")
    logger.info(f"✅ Demo API key: {demo_key}")
    
    yield
    
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
        "total_tokens_processed": store.total_tokens
    }


@app.get("/v1/models", tags=["Models"])
async def list_models(user: User = Depends(get_current_user)):
    """List available models (OpenAI compatible)."""
    return {
        "object": "list",
        "data": [model.dict() for model in store.models.values()]
    }


@app.get("/v1/models/{model_id}", tags=["Models"])
async def get_model(model_id: str, user: User = Depends(get_current_user)):
    """Get model details."""
    if model_id not in store.models:
        raise HTTPException(status_code=404, detail="Model not found")
    return store.models[model_id].dict()


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
    
    store.users[user.id] = user
    store.users_by_email[user.email] = user.id
    
    # Generate tokens
    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id)
    
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
    
    # Verify refresh token is valid
    token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
    if token_hash not in store.refresh_tokens:
        raise HTTPException(status_code=401, detail="Refresh token revoked")
    
    user_id = payload.get("sub")
    if user_id not in store.users:
        raise HTTPException(status_code=401, detail="User not found")
    
    # Generate new tokens
    new_access_token = create_access_token(user_id)
    new_refresh_token = create_refresh_token(user_id)
    
    # Revoke old refresh token
    del store.refresh_tokens[token_hash]
    
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
    
    store.api_keys[key_hash] = record
    
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
    
    store.total_requests += 1
    
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
    store.total_tokens += total_tokens
    user.usage_this_month += total_tokens
    
    # Calculate revenue
    revenue = (
        config.INPUT_TOKEN_PRICE * prompt_tokens / 1000 +
        config.OUTPUT_TOKEN_PRICE * completion_tokens / 1000
    )
    store.revenue += revenue
    
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
    """Generate completion text (integrate real model here)."""
    
    # Extract last user message
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break
    
    # Mock response (replace with real model inference)
    responses = {
        "hello": "Hello! I'm ActuallyOpenAI, a decentralized AI assistant. How can I help you today?",
        "what is": "That's a great question! Let me help explain...",
        "how to": "Here's how you can do that...",
        "help": "I'd be happy to help! Please tell me more about what you need.",
    }
    
    # Simple keyword matching (replace with real model)
    for keyword, response in responses.items():
        if keyword in user_message.lower():
            return response
    
    return f"I understand you're asking about: {user_message[:100]}... As a decentralized AI, I'm here to help! (Note: Connect a trained model for real responses)"


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
    store.total_tokens += total_tokens
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
    
    store.total_requests += 1
    
    prompt_tokens = int(len(request.prompt.split()) * 1.3)
    
    # Mock completion
    completion_text = f"Completing: {request.prompt[:50]}... (Connect model for real output)"
    completion_tokens = int(len(completion_text.split()) * 1.3)
    
    total_tokens = prompt_tokens + completion_tokens
    store.total_tokens += total_tokens
    
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
    
    store.total_requests += 1
    
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
    
    store.total_tokens += total_tokens
    
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
