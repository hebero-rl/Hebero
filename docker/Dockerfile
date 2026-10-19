# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Container for Hebero DGPO training (LIBERO 40-task suite).
#
# Reproduces the reference training environment (pip-installed Isaac Sim 6.0.0.1,
# torch 2.10.0+cu128, rsl-rl-lib 5.0.1, Python 3.12.12) from the version lock
# in docker/requirements.txt, and bakes this repo in at /workspace/hebero.
# LIBERO USD assets and the demo dataset are volume-mounted at runtime — see
# docker/launch.sh.
#
# Build (from the repo root):
#   docker build -f docker/Dockerfile -t hebero:latest .

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

# System libraries required by Kit/Isaac Sim in headless mode, plus build tools
# for the few sdists in the lock.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libatomic1 \
    libegl1 \
    libgl1 \
    libgles2 \
    libglib2.0-0 \
    libglu1-mesa \
    libvulkan1 \
    libx11-6 \
    libxcursor1 \
    libxext6 \
    libxi6 \
    libxinerama1 \
    libxrandr2 \
    libxt6 \
    libxxf86vm1 \
    ncurses-term \
    vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

# uv + managed Python matching the local env
COPY --from=ghcr.io/astral-sh/uv:0.9.22 /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
RUN uv venv /opt/venv --python 3.12.12
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Install the frozen dependency set. unsafe-best-match + prerelease=allow match
# the reference resolver settings (numpy 2.5.0rc1, NVIDIA-hosted wheels).
# --no-deps: the lock is the complete closure of the working env, which contains
# metadata-inconsistent pins (e.g. lxml 4.9.4 vs dex-retargeting's >=5.2.2) that
# a resolver would reject; install exactly what the working env has.
COPY docker/requirements.txt /tmp/requirements.txt
RUN uv pip install \
    --no-deps \
    --index-strategy unsafe-best-match \
    --prerelease allow \
    --extra-index-url https://pypi.nvidia.com \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# The lock carries both cu12 and cu13 variants of nccl/cudnn/cusparselt/nvshmem,
# which install the SAME shared-library paths (nvidia/nccl/lib/libnccl.so.2, ...)
# and overwrite each other; the cu13 copies win the install race and then fail on
# nodes whose driver is < 580 ("CUDA driver version is insufficient", NCCL 2.28.9
# at the first distributed broadcast). torch 2.10.0+cu128 is a cu12 build —
# force-reinstall the cu12 variants last so their files end up on disk, matching
# the verified workstation env.
RUN uv pip install \
    --force-reinstall \
    --no-deps \
    --extra-index-url https://pypi.nvidia.com \
    nvidia-nccl-cu12==2.27.5 \
    nvidia-cudnn-cu12==9.10.2.21 \
    nvidia-cusparselt-cu12==0.7.1 \
    nvidia-nvshmem-cu12==3.4.5

# The repo itself (isaaclab packages are vendored under source/ and picked up
# via PYTHONPATH by run.sh — no pip install needed).
WORKDIR /workspace/hebero
COPY . /workspace/hebero

# Accept the Omniverse EULA (required on first isaacsim import) and expose the
# GPU for graphics + compute when run through the NVIDIA container runtime.
ENV OMNI_KIT_ACCEPT_EULA=YES \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

# Default data locations inside the container (bind-mount the host dirs here).
ENV LIBERO_ASSETS_DATA_DIR=/data/libero/USD \
    LIBERO_ASSEMBLED_DATASET_DIR=/data/libero/demos \
    LIBERO_CONFIG_DIR=/workspace/hebero/config \
    VIZ_ARGS=--headless

CMD ["./train.sh", "libero"]
