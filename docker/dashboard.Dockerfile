# =============================================================================
# ActuallyOpenAI Dashboard Dockerfile
# Production web dashboard for contributors
# =============================================================================

FROM python:3.10-slim AS builder

# Build arguments
ARG APP_VERSION=1.0.0

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim AS production

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash aoai && \
    mkdir -p /app && \
    chown -R aoai:aoai /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=aoai:aoai actuallyopenai/ ./actuallyopenai/
COPY --chown=aoai:aoai static/ ./static/ 2>/dev/null || true
COPY --chown=aoai:aoai templates/ ./templates/ 2>/dev/null || true

# Switch to non-root user
USER aoai

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8501

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run dashboard
CMD ["uvicorn", "actuallyopenai.web.dashboard:app", "--host", "0.0.0.0", "--port", "8501"]
