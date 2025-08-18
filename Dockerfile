FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

# Install Python bawaan Ubuntu 22.04 (Python 3.10) + tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-distutils \
    git \
    libgl1 \
    ffmpeg \
    curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-docker-gpu.txt .

# Upgrade pip dan install dependencies
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker-gpu.txt

COPY . .

EXPOSE 8000

CMD ["python3", "api_server.py"]