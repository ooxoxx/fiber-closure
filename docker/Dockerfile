# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.0-cudnn-runtime-ubuntu24.04 AS runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY inference/ inference/

# Install dependencies with uv
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install .

# Create directories for mounted volumes
RUN mkdir -p /app/weights /app/cfg /app/lib

# Environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV FIBER_DARKNET_LIB_PATH=/app/lib/libdarknet.so
ENV FIBER_MODEL_CFG=/app/cfg/fiber.cfg
ENV FIBER_MODEL_WEIGHTS=/app/weights/fiber_best.weights
ENV FIBER_MODEL_DATA=/app/cfg/fiber.data

# Expose port
EXPOSE 8000

# Copy entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "inference.main:app", "--host", "0.0.0.0", "--port", "8000"]
