#!/usr/bin/env bash
#
# Convert a verl FSDP training checkpoint into a HuggingFace-format checkpoint.
#
# Usage:
#   bash scripts/merge_to_hf.sh <CKPT_DIR> <TARGET_DIR>
#   CKPT_DIR=... TARGET_DIR=... bash scripts/merge_to_hf.sh
#
# CKPT_DIR is the actor checkpoint directory written by verl, e.g.
#   checkpoints/<exp>/<run>/global_step_<N>/actor
# It must contain the per-rank shards and a `huggingface/` subdir with the
# tokenizer/processor and model config used during training.
#
# TARGET_DIR is where the merged HF model will be written.

set -euo pipefail

CKPT_DIR="${1:-${CKPT_DIR:-/data/nfs/ahmad/verl_logs/checkpoints/drkernel_async/8b-test3/global_step_20/actor}}"
TARGET_DIR="${2:-${TARGET_DIR:-/home/model/drkernel_8b_test3_step20_hf}}"

if [[ -z "${CKPT_DIR}" || -z "${TARGET_DIR}" ]]; then
    echo "Usage: $0 <CKPT_DIR> <TARGET_DIR>" >&2
    echo "   or: CKPT_DIR=... TARGET_DIR=... $0" >&2
    exit 1
fi

if [[ ! -d "${CKPT_DIR}" ]]; then
    echo "CKPT_DIR does not exist: ${CKPT_DIR}" >&2
    exit 1
fi

if [[ ! -d "${CKPT_DIR}/huggingface" ]]; then
    echo "Missing '${CKPT_DIR}/huggingface' (tokenizer/config dir from training)." >&2
    exit 1
fi

mkdir -p "${TARGET_DIR}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd)"

cd "${REPO_ROOT}"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${CKPT_DIR}" \
    --target_dir "${TARGET_DIR}" \
    --trust-remote-code

echo "Merged HF checkpoint written to: ${TARGET_DIR}"
