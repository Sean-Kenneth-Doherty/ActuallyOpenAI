# =============================================================================
# ActuallyOpenAI Orchestrator Dockerfile
# Distributed training orchestrator for coordinating workers
# =============================================================================

FROM python:3.10-slim AS builder

# Build arguments
ARG APP_VERSION=1.0.0

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir redis aioredis

# Production stage
FROM python:3.10-slim AS production

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash aoai && \
    mkdir -p /app /app/checkpoints /app/logs && \
    chown -R aoai:aoai /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=aoai:aoai actuallyopenai/ ./actuallyopenai/
COPY --chown=aoai:aoai scripts/ ./scripts/ 2>/dev/null || true

# Switch to non-root user
USER aoai

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    REDIS_URL=redis://redis:6379 \
    CHECKPOINT_DIR=/app/checkpoints \
    LOG_LEVEL=INFO

# Expose ports
EXPOSE 8000 50051

# Volumes for persistent data
VOLUME ["/app/checkpoints", "/app/logs"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start script
COPY --chown=aoai:aoai <<EOF /app/start.sh
#!/bin/bash
set -e

echo "🔄 Waiting for Redis..."
while ! nc -z redis 6379; do
    sleep 1
done
echo "✅ Redis is ready"

echo "🔄 Waiting for PostgreSQL..."
while ! nc -z postgres 5432; do
    sleep 1
done
echo "✅ PostgreSQL is ready"

echo "🚀 Starting ActuallyOpenAI Orchestrator..."
exec python -m actuallyopenai.orchestrator.main
EOF

RUN chmod +x /app/start.sh

# Run orchestrator
CMD ["/app/start.sh"]
