"""
Model API - RESTful API for serving trained AI models and generating revenue.
Revenue is distributed as dividends to AOAI token holders.
"""

import asyncio
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Depends, Header, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import structlog

from actuallyopenai.config import get_settings
from actuallyopenai.core.models import (
    ModelInfo, APIKey, InferenceRequest, InferenceResponse
)
from actuallyopenai.core.model_registry import get_model_registry, ModelRegistry

logger = structlog.get_logger()


# =============================================================================
# Request/Response Models
# =============================================================================

class CreateAPIKeyRequest(BaseModel):
    """Request to create a new API key."""
    name: str
    wallet_address: str
    allowed_models: List[str] = Field(default_factory=list)
    rate_limit_per_minute: int = Field(default=60, ge=1, le=1000)


class APIKeyResponse(BaseModel):
    """Response containing API key details."""
    id: str
    key: str  # Only returned on creation
    name: str
    wallet_address: str
    rate_limit_per_minute: int
    created_at: datetime


class TextGenerationRequest(BaseModel):
    """Request for text generation."""
    model: str
    prompt: str
    max_tokens: int = Field(default=100, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, ge=0, le=1)
    stream: bool = False


class TextGenerationResponse(BaseModel):
    """Response from text generation."""
    id: str
    model: str
    created: int
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class ImageGenerationRequest(BaseModel):
    """Request for image generation."""
    model: str
    prompt: str
    size: str = "512x512"
    n: int = Field(default=1, ge=1, le=4)


class ModelListResponse(BaseModel):
    """Response listing available models."""
    data: List[Dict[str, Any]]


# =============================================================================
# API State
# =============================================================================

class APIState:
    """Global state for the API."""
    
    def __init__(self):
        self.api_keys: Dict[str, APIKey] = {}
        self.key_hash_to_id: Dict[str, str] = {}
        self.rate_limits: Dict[str, List[datetime]] = {}  # key_id -> request times
        self.revenue_collected: Decimal = Decimal("0")
        
    def hash_key(self, key: str) -> str:
        """Hash an API key for storage."""
        settings = get_settings()
        return hashlib.sha256(
            (key + settings.api_key_salt).encode()
        ).hexdigest()
    
    def generate_api_key(self) -> str:
        """Generate a new API key."""
        return f"aoai-{secrets.token_urlsafe(32)}"
    
    def check_rate_limit(self, key_id: str, limit: int) -> bool:
        """Check if request is within rate limit."""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old entries
        if key_id in self.rate_limits:
            self.rate_limits[key_id] = [
                t for t in self.rate_limits[key_id] if t > minute_ago
            ]
        else:
            self.rate_limits[key_id] = []
        
        # Check limit
        if len(self.rate_limits[key_id]) >= limit:
            return False
        
        # Record request
        self.rate_limits[key_id].append(now)
        return True


state = APIState()


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="ActuallyOpenAI Model API",
    description="API for accessing community-trained AI models. Revenue supports compute contributors.",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Authentication
# =============================================================================

async def get_api_key(authorization: str = Header(None)) -> APIKey:
    """Validate API key from Authorization header."""
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include 'Authorization: Bearer <key>' header."
        )
    
    # Extract key from "Bearer <key>" format
    if authorization.startswith("Bearer "):
        key = authorization[7:]
    else:
        key = authorization
    
    key_hash = state.hash_key(key)
    
    if key_hash not in state.key_hash_to_id:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    key_id = state.key_hash_to_id[key_hash]
    api_key = state.api_keys.get(key_id)
    
    if not api_key:
        raise HTTPException(status_code=401, detail="API key not found")
    
    if not api_key.is_active:
        raise HTTPException(status_code=401, detail="API key is disabled")
    
    if api_key.expires_at and api_key.expires_at < datetime.utcnow():
        raise HTTPException(status_code=401, detail="API key has expired")
    
    # Check rate limit
    if not state.check_rate_limit(api_key.id, api_key.rate_limit_per_minute):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {api_key.rate_limit_per_minute} requests per minute."
        )
    
    return api_key


# =============================================================================
# API Key Management
# =============================================================================

@app.post("/v1/api-keys", response_model=APIKeyResponse)
async def create_api_key(request: CreateAPIKeyRequest):
    """
    Create a new API key for accessing the model API.
    The key is only shown once - store it securely.
    """
    # Generate key
    raw_key = state.generate_api_key()
    key_hash = state.hash_key(raw_key)
    
    api_key = APIKey(
        key_hash=key_hash,
        name=request.name,
        wallet_address=request.wallet_address,
        allowed_models=request.allowed_models,
        rate_limit_per_minute=request.rate_limit_per_minute
    )
    
    state.api_keys[api_key.id] = api_key
    state.key_hash_to_id[key_hash] = api_key.id
    
    logger.info(
        "API key created",
        key_id=api_key.id,
        wallet=request.wallet_address
    )
    
    return APIKeyResponse(
        id=api_key.id,
        key=raw_key,  # Only returned on creation
        name=api_key.name,
        wallet_address=api_key.wallet_address,
        rate_limit_per_minute=api_key.rate_limit_per_minute,
        created_at=api_key.created_at
    )


@app.get("/v1/api-keys/me")
async def get_my_api_key(api_key: APIKey = Depends(get_api_key)):
    """Get information about the current API key."""
    return {
        "id": api_key.id,
        "name": api_key.name,
        "wallet_address": api_key.wallet_address,
        "rate_limit_per_minute": api_key.rate_limit_per_minute,
        "total_requests": api_key.total_requests,
        "total_tokens_used": api_key.total_tokens_used,
        "is_active": api_key.is_active,
        "created_at": api_key.created_at.isoformat()
    }


# =============================================================================
# Model Endpoints
# =============================================================================

@app.get("/v1/models", response_model=ModelListResponse)
async def list_models(api_key: APIKey = Depends(get_api_key)):
    """List all available models."""
    registry = get_model_registry()
    models = await registry.list_models(is_public=True)
    
    return ModelListResponse(
        data=[
            {
                "id": m.id,
                "name": m.name,
                "description": m.description,
                "model_type": m.model_type,
                "architecture": m.architecture,
                "parameter_count": m.parameter_count,
                "created": int(m.created_at.timestamp())
            }
            for m in models
        ]
    )


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str, api_key: APIKey = Depends(get_api_key)):
    """Get details of a specific model."""
    registry = get_model_registry()
    model = await registry.get_model(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if not model.is_public:
        if api_key.allowed_models and model_id not in api_key.allowed_models:
            raise HTTPException(status_code=403, detail="Access denied to this model")
    
    return {
        "id": model.id,
        "name": model.name,
        "description": model.description,
        "model_type": model.model_type,
        "architecture": model.architecture,
        "parameter_count": model.parameter_count,
        "version": model.version,
        "created": int(model.created_at.timestamp())
    }


# =============================================================================
# Inference Endpoints
# =============================================================================

@app.post("/v1/completions", response_model=TextGenerationResponse)
async def create_completion(
    request: TextGenerationRequest,
    background_tasks: BackgroundTasks,
    api_key: APIKey = Depends(get_api_key)
):
    """
    Generate text completion using a model.
    Charges based on token usage - revenue supports compute contributors.
    """
    registry = get_model_registry()
    
    # Find model
    model = await registry.get_model(request.model)
    if not model:
        model = await registry.get_model_by_name(request.model)
    
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found")
    
    # Check model access
    if api_key.allowed_models and model.id not in api_key.allowed_models:
        raise HTTPException(status_code=403, detail="Access denied to this model")
    
    start_time = time.time()
    
    # TODO: Actual model inference would happen here
    # For now, simulate response
    generated_text = f"[Simulated response from {model.name}] This is a placeholder for actual model inference."
    
    # Calculate tokens (rough estimate)
    prompt_tokens = len(request.prompt.split()) * 2
    completion_tokens = min(request.max_tokens, len(generated_text.split()) * 2)
    total_tokens = prompt_tokens + completion_tokens
    
    # Calculate cost and record revenue
    cost_per_1k_tokens = Decimal("0.001")  # $0.001 per 1K tokens
    revenue = cost_per_1k_tokens * Decimal(str(total_tokens)) / Decimal("1000")
    
    # Update stats
    api_key.total_requests += 1
    api_key.total_tokens_used += total_tokens
    state.revenue_collected += revenue
    
    # Record API call
    background_tasks.add_task(
        registry.record_api_call,
        model.id,
        revenue
    )
    
    processing_time = time.time() - start_time
    
    logger.info(
        "Completion generated",
        model=model.name,
        tokens=total_tokens,
        revenue=str(revenue),
        time_ms=round(processing_time * 1000, 2)
    )
    
    return TextGenerationResponse(
        id=f"cmpl-{secrets.token_hex(12)}",
        model=model.name,
        created=int(time.time()),
        choices=[
            {
                "index": 0,
                "text": generated_text,
                "finish_reason": "stop"
            }
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    )


@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    api_key: APIKey = Depends(get_api_key)
):
    """
    Generate chat completion (OpenAI-compatible endpoint).
    """
    model_name = request.get("model", "")
    messages = request.get("messages", [])
    max_tokens = request.get("max_tokens", 100)
    temperature = request.get("temperature", 0.7)
    stream = request.get("stream", False)
    
    registry = get_model_registry()
    
    # Find model
    model = await registry.get_model(model_name)
    if not model:
        model = await registry.get_model_by_name(model_name)
    
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    # TODO: Actual chat inference
    # Simulate response
    last_message = messages[-1]["content"] if messages else ""
    response_text = f"[{model.name}] I received your message about: {last_message[:50]}..."
    
    # Calculate usage
    prompt_tokens = sum(len(m.get("content", "").split()) * 2 for m in messages)
    completion_tokens = len(response_text.split()) * 2
    total_tokens = prompt_tokens + completion_tokens
    
    # Revenue
    revenue = Decimal("0.001") * Decimal(str(total_tokens)) / Decimal("1000")
    api_key.total_requests += 1
    api_key.total_tokens_used += total_tokens
    state.revenue_collected += revenue
    
    background_tasks.add_task(registry.record_api_call, model.id, revenue)
    
    if stream:
        async def generate():
            """Stream response chunks."""
            words = response_text.split()
            for i, word in enumerate(words):
                chunk = {
                    "id": f"chatcmpl-{secrets.token_hex(12)}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model.name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": word + " "},
                        "finish_reason": None if i < len(words) - 1 else "stop"
                    }]
                }
                yield f"data: {chunk}\n\n"
                await asyncio.sleep(0.05)
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
    
    return {
        "id": f"chatcmpl-{secrets.token_hex(12)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model.name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    }


# =============================================================================
# Revenue & Stats Endpoints
# =============================================================================

@app.get("/v1/stats")
async def get_api_stats():
    """Get public API statistics."""
    registry = get_model_registry()
    models = await registry.list_models()
    
    total_api_calls = sum(m.api_calls for m in models)
    total_revenue = sum(m.total_revenue for m in models)
    
    return {
        "total_models": len(models),
        "total_api_calls": total_api_calls,
        "total_revenue_usd": str(total_revenue),
        "revenue_for_dividend": str(state.revenue_collected)
    }


@app.get("/v1/usage")
async def get_usage(api_key: APIKey = Depends(get_api_key)):
    """Get usage statistics for the current API key."""
    return {
        "total_requests": api_key.total_requests,
        "total_tokens_used": api_key.total_tokens_used,
        "rate_limit_per_minute": api_key.rate_limit_per_minute
    }


# =============================================================================
# Health & Info
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/")
async def root():
    """API information."""
    return {
        "name": "ActuallyOpenAI Model API",
        "version": "0.1.0",
        "description": "Community-owned AI models. Revenue supports compute contributors.",
        "docs_url": "/docs"
    }


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Run the API server."""
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "actuallyopenai.api.model_api:app",
        host="0.0.0.0",
        port=8001,
        reload=settings.debug
    )


if __name__ == "__main__":
    main()
