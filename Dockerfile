FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

# Non-interactive apt-get commands
ARG DEBIAN_FRONTEND=noninteractive

# No GPUs visible during build
ARG CUDA_VISIBLE_DEVICES=none

# Specify CUDA architectures -> 8.9: L40S, 9.0: H100
ARG TORCH_CUDA_ARCH_LIST="8.9;9.0"

# Set the Python version
ARG PYTHON_VERSION=3.10.12

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev libffi-dev libncursesw5-dev \
    xz-utils tk-dev libxml2-dev libxmlsec1-dev liblzma-dev git vim libnuma-dev\
    && rm -rf /var/lib/apt/lists/*

# Install Python
RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
    tar -xzf Python-$PYTHON_VERSION.tgz && \
    cd Python-$PYTHON_VERSION && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-$PYTHON_VERSION.tgz Python-$PYTHON_VERSION

# Install pip and core Python tools
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py && \
    python3.10 -m pip install --upgrade pip setuptools wheel uv

# Set up project
WORKDIR /vec-inf
COPY . /vec-inf

# Install project dependencies with build requirements
RUN PIP_INDEX_URL="https://download.pytorch.org/whl/cu121" uv pip install --system -e .[dev]

# Final configuration
RUN mkdir -p /vec-inf/nccl && \
    mv /root/.config/vllm/nccl/cu12/libnccl.so.2.18.1 /vec-inf/nccl/libnccl.so.2.18.1

# Set the default command to start an interactive shell
CMD ["bash"]
