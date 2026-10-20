FROM python:3.11-slim

ARG DEVICE=cpu

# Install system runtime libraries needed by RDKit
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install PyTorch conditionally (CPU vs CUDA 12.1) + dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    if [ "$DEVICE" = "cuda" ]; then \
        echo "Installing PyTorch with CUDA 12.1 support..." && \
        pip install --no-cache-dir \
            --extra-index-url https://download.pytorch.org/whl/cu121 \
            torch==2.5.1+cu121; \
    else \
        echo "Installing lightweight CPU-optimized PyTorch..." && \
        pip install --no-cache-dir \
            --extra-index-url https://download.pytorch.org/whl/cpu \
            torch==2.5.1+cpu; \
    fi && \
    pip install --no-cache-dir \
        rdkit \
        "fastmcp>=0.1.0" \
        "pandas>=2.0.0" \
        "numpy>=1.24.0" \
        "scipy>=1.10.0" \
        "pydantic>=2.0" \
        "tqdm>=4.65.0"

# Copy source code, pyproject, and baked model/cache
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY checkpoints /app/checkpoints
COPY data /app/data

# Install flavri-mcp package
RUN pip install --no-cache-dir .

# Default environment configuration
ENV PYTHONUNBUFFERED=1
ENV FLAVRI_CACHE_PATH=/app/data/flavri_cache.pt
ENV FLAVRI_CHECKPOINT=/app/checkpoints/flavri-0.8.pt

EXPOSE 8000

# Default entrypoint runs flavri-mcp in stdio mode (can be overridden with --transport sse)
ENTRYPOINT ["flavri-mcp"]
