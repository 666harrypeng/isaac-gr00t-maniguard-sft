# syntax=docker/dockerfile:1
# GR00T N1.6 SFT image (ManiGuard) — x86_64. Self-contained: sets up the Isaac-GR00T
# n1d6 env (torch cu128 + flash-attn / torchcodec / deepspeed wheels via `uv sync`) and
# runs the ManiGuard SFT launcher. The env recipe mirrors upstream `docker/Dockerfile`,
# dropping the EGL/Vulkan rendering + aarch64 torchcodec-source steps (SFT does not need
# them). If upstream changes its env, re-sync this recipe from `docker/Dockerfile`.
#
# Build: docker build -f docker/sft.Dockerfile -t gr00t-maniguard-sft .
# Run:   docker run --gpus all --ipc=host -e HF_TOKEN -e WANDB_API_KEY \
#          -v "$PWD/outputs:/workspace/outputs" gr00t-maniguard-sft --family jar
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHON=/usr/bin/python \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# System deps: build tools, python 3.10, ffmpeg (torchcodec runtime), libaio-dev (deepspeed), uv.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git git-lfs curl \
        python3.10 python3.10-venv python3.10-dev python3-pip python-is-python3 \
        ffmpeg libaio-dev && \
    python -m pip install --upgrade pip setuptools wheel && \
    curl -LsSf https://astral.sh/uv/0.8.14/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Deps first (cached layer that only busts on lockfile change). x86_64 pulls
# flash-attn / torchcodec / deepspeed as wheels — no source build.
COPY pyproject.toml uv.lock ./
RUN UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 \
    uv sync --frozen --no-install-workspace --all-packages --extra dev --no-cache

# Project code (includes the vendored maniguard/ + tools/gr00t_sft/ + gr00t_stats/).
COPY . .
RUN uv pip install --python /workspace/.venv/bin/python -e . --no-deps
ENV PATH="/workspace/.venv/bin:${PATH}" \
    VIRTUAL_ENV="/workspace/.venv"

# Cap per-worker math threads (dataloader oversubscription; run_sft.sh also sets these).
ENV OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Entrypoint = the SFT launcher; args are run_all.sh's (e.g. `--family jar` or `--all`).
ENTRYPOINT ["bash", "tools/gr00t_sft/run_all.sh"]
