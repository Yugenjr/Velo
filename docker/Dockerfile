# Reproducible Linux Build and Verification Environment for Velo
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 1. Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    pkg-config \
    libssl-dev \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# 3. Upgrade pip and install build dependencies
RUN python3.10 -m pip install --upgrade pip
RUN pip install maturin torch --extra-index-url https://download.pytorch.org/whl/cu121

# 4. Copy Velo source code
WORKDIR /workspace/velo
COPY . .

# 5. Build and install Velo
RUN pip install -e .

# 6. Verify native library import
CMD ["python3.10", "-c", "import torch; torch.cuda.init(); import velo; print('Velo compiled successfully on Linux!'); print(velo)"]
