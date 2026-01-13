# =============================================================================
# ActuallyOpenAI - GPU Worker Dockerfile
# =============================================================================

FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04 as base

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-venv \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

# Create non-root user
RUN useradd --create-home --shell /bin/bash aoai

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Copy application
COPY actuallyopenai/ /app/actuallyopenai/

# Set environment
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    WORKER_TYPE=gpu \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Health check
HEALTHCHECK --interval=60s --timeout=30s --start-period=30s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Switch to non-root user
USER aoai

# Default command
CMD ["python", "-m", "actuallyopenai.worker.worker_node"]
