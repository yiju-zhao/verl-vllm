#!/usr/bin/env bash
set -euo pipefail
PY="${PY:-/home/yubaifeng/e84381970/envs/drkernel310/bin/python3.10}"
cd /home/yubaifeng/e84381970/experiment/verl-vllm/drkernel-verl-port-drkernel
$PY examples/data_preprocess/gsm8k.py --local_dir "$HOME/data/gsm8k"
ls -la "$HOME/data/gsm8k"/{train,test}.parquet
