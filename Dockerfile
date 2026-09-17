# SSA-3DGS training image: CUDA 11.8 + PyTorch cu118 + gsplat (JIT-compiled).
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV CUDA_HOME="/usr/local/cuda"
# GPU architectures the gsplat CUDA kernels are JIT-compiled for. Trim to your
# GPU's compute capability (https://developer.nvidia.com/cuda-gpus) to speed up
# the first run, e.g. "8.6" for an RTX 3090.
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV PYTHONDONTWRITEBYTECODE=1

# System packages. COLMAP is only needed to pose your own captures.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    colmap \
    curl \
    ffmpeg \
    git \
    ninja-build \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    sudo \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Non-root user (uid 1000 matches the default host user for bind mounts),
# with passwordless sudo for ad-hoc installs inside the container.
RUN useradd -m -d /home/user -u 1000 -G sudo user && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER 1000
ENV PATH="${PATH}:/home/user/.local/bin"
SHELL ["/bin/bash", "-c"]
WORKDIR /home/user

# Python environment.
RUN python3.10 -m pip install --upgrade pip setuptools
RUN python3.10 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
COPY --chown=1000 requirements.txt /home/user/requirements.txt
RUN python3.10 -m pip install -r /home/user/requirements.txt

# Copy the code.
COPY --chown=1000 . /home/user/code
WORKDIR /home/user/code

# Viewer port.
EXPOSE 8080

CMD ["/bin/bash"]
