#!/usr/bin/env bash
# Build vLLM 0.18 from local source for GB10 (sm_121, aarch64) against the env's
# torch 2.9.1+cu130. See BUILD_NOTES_cuda.md for context.
#
# Key constraints discovered during A2:
#  - vllm 0.18 pyproject build-requires torch==2.10.0; env has 2.9.1. We run
#    `use_existing_torch.py` (already done) to strip the pin and build against
#    installed torch via --no-build-isolation.
#  - CUDA toolkit nvcc 13.0 at /usr/local/cuda (CUDA_HOME was unset).
#  - GB10 is sm_121; torch 2.9.1 only validates up to sm_120, so we target
#    "12.0+PTX" (PTX JITs forward to 12.1 at runtime).
set -euo pipefail

PY=/home/yubaifeng/e84381970/envs/drkernel310/bin/python3.10
VLLM_SRC=/home/yubaifeng/e84381970/experiment/verl-vllm/vllm
LOG=/tmp/vllm018_build.log

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"
export MAX_JOBS="${MAX_JOBS:-8}"
export NVCC_THREADS="${NVCC_THREADS:-2}"
# ccache makes the build resumable across runs: vllm wires it as the CUDA/C++
# compiler launcher, so a restart reuses already-compiled kernels. Critical here
# because the long compile can be interrupted (background-task time caps).
export CCACHE_DIR="${CCACHE_DIR:-/home/yubaifeng/e84381970/.ccache}"
ccache -M 60G >/dev/null 2>&1 || true
echo "ccache: $(command -v ccache)  dir=$CCACHE_DIR"

echo "CUDA_HOME=$CUDA_HOME  ARCH=$TORCH_CUDA_ARCH_LIST  MAX_JOBS=$MAX_JOBS"
nvcc --version | tail -2

cd "$VLLM_SRC"
# Build/install editable against the EXISTING torch (no isolation).
$PY -m pip install --no-build-isolation -e . 2>&1 | tee "$LOG"
