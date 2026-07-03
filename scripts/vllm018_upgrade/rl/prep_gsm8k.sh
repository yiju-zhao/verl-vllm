#!/usr/bin/env bash
set -euo pipefail
# Repo-relative: works on any machine/checkout (script lives at scripts/vllm018_upgrade/rl/).
cd "$(dirname "$0")/../../.."
# Interpreter: honor $PY; else the GB10 env if present; else plain python3.
PY="${PY:-}"
if [ -z "$PY" ]; then
  if [ -x /home/yubaifeng/e84381970/envs/drkernel310/bin/python3.10 ]; then
    PY=/home/yubaifeng/e84381970/envs/drkernel310/bin/python3.10
  else
    PY=python3
  fi
fi
$PY examples/data_preprocess/gsm8k.py --local_dir "$HOME/data/gsm8k"
ls -la "$HOME/data/gsm8k"/train.parquet "$HOME/data/gsm8k"/test.parquet
