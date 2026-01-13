"""
Prometheus Metrics and Monitoring for ActuallyOpenAI.

Provides real-time metrics for:
- API performance and usage
- Training progress
- Worker health
- Blockchain transactions
- Revenue tracking
"""

import time
from functools import wraps
from typing import Callable, Dict, Any
import asyncio

from prometheus_client import (
    Counter, Gauge, Histogram, Summary, Info,
    REGISTRY, generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry
)
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

logger = structlog.get_logger()


# =============================================================================
# Metric Definitions
# =============================================================================

# API Metrics
API_REQUESTS_TOTAL = Counter(
    "aoai_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status_code"]
)

API_REQUEST_DURATION = Histogram(
    "aoai_api_request_duration_seconds",
    "API request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

API_ACTIVE_REQUESTS = Gauge(
    "aoai_api_active_requests",
    "Number of active API requests"
)

API_RATE_LIMIT_HITS = Counter(
    "aoai_api_rate_limit_hits_total",
    "Total rate limit hits",
    ["user_tier"]
)

# Token Usage Metrics
TOKENS_PROCESSED_TOTAL = Counter(
    "aoai_tokens_processed_total",
    "Total tokens processed",
    ["model", "type"]  # type: input/output
)

TOKENS_PER_REQUEST = Histogram(
    "aoai_tokens_per_request",
    "Tokens per request",
    ["model"],
    buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
)

# Model Inference Metrics
INFERENCE_LATENCY = Histogram(
    "aoai_inference_latency_seconds",
    "Model inference latency",
    ["model"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
)

INFERENCE_QUEUE_SIZE = Gauge(
    "aoai_inference_queue_size",
    "Current inference queue size",
    ["model"]
)

# Training Metrics
TRAINING_STEPS_TOTAL = Counter(
    "aoai_training_steps_total",
    "Total training steps completed"
)

TRAINING_LOSS = Gauge(
    "aoai_training_loss",
    "Current training loss",
    ["model", "metric"]  # metric: loss, perplexity, accuracy
)

TRAINING_THROUGHPUT = Gauge(
    "aoai_training_throughput_tokens_per_second",
    "Training throughput in tokens per second"
)

TRAINING_EPOCH = Gauge(
    "aoai_training_epoch",
    "Current training epoch",
    ["model"]
)

# Worker Metrics
WORKERS_TOTAL = Gauge(
    "aoai_workers_total",
    "Total registered workers",
    ["status"]  # online, offline, training
)

WORKER_COMPUTE_HOURS = Counter(
    "aoai_worker_compute_hours_total",
    "Total compute hours contributed",
    ["worker_type"]  # gpu, cpu
)

WORKER_TASKS_COMPLETED = Counter(
    "aoai_worker_tasks_completed_total",
    "Total tasks completed by workers"
)

WORKER_GPU_UTILIZATION = Gauge(
    "aoai_worker_gpu_utilization_percent",
    "GPU utilization percentage",
    ["worker_id"]
)

WORKER_GPU_MEMORY = Gauge(
    "aoai_worker_gpu_memory_bytes",
    "GPU memory usage",
    ["worker_id", "type"]  # used, total
)

# Blockchain Metrics
BLOCKCHAIN_TRANSACTIONS = Counter(
    "aoai_blockchain_transactions_total",
    "Total blockchain transactions",
    ["type"]  # mint, transfer, dividend
)

TOKENS_MINTED = Counter(
    "aoai_tokens_minted_total",
    "Total AOAI tokens minted"
)

DIVIDENDS_DISTRIBUTED = Counter(
    "aoai_dividends_distributed_total",
    "Total dividends distributed (ETH)"
)

TOKEN_PRICE = Gauge(
    "aoai_token_price_usd",
    "Current AOAI token price in USD"
)

# Revenue Metrics
REVENUE_TOTAL = Counter(
    "aoai_revenue_total_usd",
    "Total revenue collected",
    ["source"]  # api, subscription
)

REVENUE_24H = Gauge(
    "aoai_revenue_24h_usd",
    "Revenue in last 24 hours"
)

# System Metrics
SYSTEM_INFO = Info(
    "aoai_system",
    "System information"
)

DATABASE_CONNECTIONS = Gauge(
    "aoai_database_connections",
    "Active database connections",
    ["database"]
)

CACHE_HITS = Counter(
    "aoai_cache_hits_total",
    "Cache hits",
    ["cache"]
)

CACHE_MISSES = Counter(
    "aoai_cache_misses_total",
    "Cache misses",
    ["cache"]
)


# =============================================================================
# Middleware for FastAPI
# =============================================================================

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect API metrics."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        method = request.method
        path = request.url.path
        
        # Normalize path (remove IDs)
        normalized_path = self._normalize_path(path)
        
        # Track active requests
        API_ACTIVE_REQUESTS.inc()
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise
        finally:
            duration = time.time() - start_time
            
            # Record metrics
            API_REQUESTS_TOTAL.labels(
                method=method,
                endpoint=normalized_path,
                status_code=status_code
            ).inc()
            
            API_REQUEST_DURATION.labels(
                method=method,
                endpoint=normalized_path
            ).observe(duration)
            
            API_ACTIVE_REQUESTS.dec()
        
        return response
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path by replacing IDs with placeholders."""
        parts = path.split("/")
        normalized = []
        
        for part in parts:
            # Replace UUIDs and numeric IDs
            if len(part) == 36 and "-" in part:  # UUID
                normalized.append("{id}")
            elif part.isdigit():
                normalized.append("{id}")
            else:
                normalized.append(part)
        
        return "/".join(normalized)


# =============================================================================
# Metrics Endpoint
# =============================================================================

def create_metrics_app() -> FastAPI:
    """Create a separate FastAPI app for metrics."""
    
    app = FastAPI(
        title="ActuallyOpenAI Metrics",
        description="Prometheus metrics endpoint"
    )
    
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST
        )
    
    @app.get("/health")
    async def health():
        """Health check."""
        return {"status": "healthy"}
    
    return app


# =============================================================================
# Metric Recording Helpers
# =============================================================================

class MetricsRecorder:
    """Helper class for recording metrics."""
    
    @staticmethod
    def record_api_request(method: str, endpoint: str, status_code: int, duration: float):
        """Record an API request."""
        API_REQUESTS_TOTAL.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()
        
        API_REQUEST_DURATION.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    @staticmethod
    def record_tokens(model: str, input_tokens: int, output_tokens: int):
        """Record token usage."""
        TOKENS_PROCESSED_TOTAL.labels(model=model, type="input").inc(input_tokens)
        TOKENS_PROCESSED_TOTAL.labels(model=model, type="output").inc(output_tokens)
        
        total_tokens = input_tokens + output_tokens
        TOKENS_PER_REQUEST.labels(model=model).observe(total_tokens)
    
    @staticmethod
    def record_inference(model: str, latency: float, queue_size: int = 0):
        """Record inference metrics."""
        INFERENCE_LATENCY.labels(model=model).observe(latency)
        INFERENCE_QUEUE_SIZE.labels(model=model).set(queue_size)
    
    @staticmethod
    def record_training_step(loss: float, perplexity: float, throughput: float):
        """Record a training step."""
        TRAINING_STEPS_TOTAL.inc()
        TRAINING_LOSS.labels(model="current", metric="loss").set(loss)
        TRAINING_LOSS.labels(model="current", metric="perplexity").set(perplexity)
        TRAINING_THROUGHPUT.set(throughput)
    
    @staticmethod
    def record_worker_status(online: int, offline: int, training: int):
        """Record worker status counts."""
        WORKERS_TOTAL.labels(status="online").set(online)
        WORKERS_TOTAL.labels(status="offline").set(offline)
        WORKERS_TOTAL.labels(status="training").set(training)
    
    @staticmethod
    def record_worker_gpu(worker_id: str, utilization: float, memory_used: int, memory_total: int):
        """Record worker GPU metrics."""
        WORKER_GPU_UTILIZATION.labels(worker_id=worker_id).set(utilization)
        WORKER_GPU_MEMORY.labels(worker_id=worker_id, type="used").set(memory_used)
        WORKER_GPU_MEMORY.labels(worker_id=worker_id, type="total").set(memory_total)
    
    @staticmethod
    def record_blockchain_tx(tx_type: str, amount: float = 0):
        """Record a blockchain transaction."""
        BLOCKCHAIN_TRANSACTIONS.labels(type=tx_type).inc()
        
        if tx_type == "mint":
            TOKENS_MINTED.inc(amount)
        elif tx_type == "dividend":
            DIVIDENDS_DISTRIBUTED.inc(amount)
    
    @staticmethod
    def record_revenue(source: str, amount: float):
        """Record revenue."""
        REVENUE_TOTAL.labels(source=source).inc(amount)
    
    @staticmethod
    def update_system_info(version: str, environment: str):
        """Update system info."""
        SYSTEM_INFO.info({
            "version": version,
            "environment": environment
        })


# =============================================================================
# Decorator for timing functions
# =============================================================================

def track_time(metric_name: str = None):
    """Decorator to track function execution time."""
    
    def decorator(func: Callable):
        name = metric_name or func.__name__
        
        # Create a histogram for this function if not exists
        histogram = Histogram(
            f"aoai_function_duration_seconds_{name}",
            f"Duration of {name}",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        )
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                histogram.observe(time.time() - start)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                histogram.observe(time.time() - start)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# =============================================================================
# Grafana Dashboard Configuration
# =============================================================================

GRAFANA_DASHBOARD = {
    "title": "ActuallyOpenAI Platform",
    "uid": "aoai-main",
    "panels": [
        {
            "title": "API Requests/sec",
            "type": "graph",
            "datasource": "Prometheus",
            "targets": [{
                "expr": "rate(aoai_api_requests_total[5m])",
                "legendFormat": "{{method}} {{endpoint}}"
            }]
        },
        {
            "title": "API Latency (p99)",
            "type": "graph",
            "datasource": "Prometheus",
            "targets": [{
                "expr": "histogram_quantile(0.99, rate(aoai_api_request_duration_seconds_bucket[5m]))",
                "legendFormat": "{{endpoint}}"
            }]
        },
        {
            "title": "Tokens Processed",
            "type": "counter",
            "datasource": "Prometheus",
            "targets": [{
                "expr": "sum(rate(aoai_tokens_processed_total[5m]))",
                "legendFormat": "Tokens/sec"
            }]
        },
        {
            "title": "Active Workers",
            "type": "gauge",
            "datasource": "Prometheus",
            "targets": [{
                "expr": "sum(aoai_workers_total{status='online'})",
                "legendFormat": "Online"
            }]
        },
        {
            "title": "Training Loss",
            "type": "graph",
            "datasource": "Prometheus",
            "targets": [{
                "expr": "aoai_training_loss{metric='loss'}",
                "legendFormat": "Loss"
            }]
        },
        {
            "title": "Revenue (24h)",
            "type": "stat",
            "datasource": "Prometheus",
            "targets": [{
                "expr": "aoai_revenue_24h_usd",
                "legendFormat": "USD"
            }]
        }
    ]
}


# =============================================================================
# Alerting Rules
# =============================================================================

ALERT_RULES = """
groups:
  - name: actuallyopenai
    rules:
      # API Health
      - alert: HighAPILatency
        expr: histogram_quantile(0.99, rate(aoai_api_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency detected"
          description: "P99 latency is above 2 seconds"
      
      - alert: HighErrorRate
        expr: sum(rate(aoai_api_requests_total{status_code=~"5.."}[5m])) / sum(rate(aoai_api_requests_total[5m])) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is above 1%"
      
      # Workers
      - alert: LowWorkerCount
        expr: sum(aoai_workers_total{status="online"}) < 5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low number of online workers"
          description: "Less than 5 workers are online"
      
      # Training
      - alert: TrainingStalled
        expr: increase(aoai_training_steps_total[30m]) == 0
        for: 30m
        labels:
          severity: critical
        annotations:
          summary: "Training has stalled"
          description: "No training steps in 30 minutes"
      
      - alert: HighTrainingLoss
        expr: aoai_training_loss{metric="loss"} > 5
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Training loss is high"
          description: "Training loss above 5 for 15 minutes"
      
      # Revenue
      - alert: LowRevenue
        expr: aoai_revenue_24h_usd < 100
        for: 1h
        labels:
          severity: info
        annotations:
          summary: "Low 24h revenue"
          description: "24h revenue below $100"
"""


# =============================================================================
# Initialization
# =============================================================================

def init_metrics(app: FastAPI, version: str = "1.0.0", environment: str = "development"):
    """Initialize metrics for a FastAPI app."""
    
    # Add middleware
    app.add_middleware(MetricsMiddleware)
    
    # Update system info
    MetricsRecorder.update_system_info(version, environment)
    
    logger.info("Metrics initialized", version=version, environment=environment)
    
    return MetricsRecorder()
