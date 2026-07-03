#!/usr/bin/env bash
# One-shot environment verification for a prebuilt NPU docker/env.
# Run INSIDE the container, from the root of THIS repo (after clone + pip install -e . --no-deps).
# Prints PASS/FAIL per item; exit 0 only if all hard requirements pass.
# Items 1-6 check the stack; 7-9 check this repo's integration; 10 checks data/model.
PY="${PY:-python3}"
fail=0
ck() { # ck <name> <expected-hint> <python-code>
  out=$($PY -c "$3" 2>&1 | tail -1)
  if [ $? -eq 0 ] && [ -n "$out" ] && ! echo "$out" | grep -qiE "error|traceback|No module"; then
    echo "PASS  $1: $out"
  else
    echo "FAIL  $1: $out   (期望: $2)"; fail=1
  fi
}

echo "== 1-6: 基础栈 =="
ck "npu 设备"        ">=1"                 "import torch,torch_npu; print(torch.npu.device_count(), 'NPUs')"
ck "torch/torch_npu" "2.9.x / 2.9.x"       "import torch,torch_npu; print(torch.__version__, '/', torch_npu.__version__)"
ck "npu 算力冒烟"    "matmul ok"           "import torch,torch_npu; a=torch.randn(64,64).npu(); print('matmul ok', float((a@a).sum())*0+1)"
ck "vllm 0.18"       "0.18.x"              "import vllm; assert vllm.__version__.startswith('0.18'), vllm.__version__; print(vllm.__version__)"
ck "vllm_ascend"     "0.18.x"              "import vllm_ascend; print(getattr(vllm_ascend,'__version__','installed'))"
ck "ray"             "2.4x"                "import ray; print(ray.__version__)"

echo "== 7-9: 本仓库集成 =="
ck "verl 指向本仓库" "本 repo 路径"        "import verl,os; print(os.path.dirname(verl.__file__))"
ck "GapA 修复在场"   "fp32 upcast present" "import inspect, verl.utils.torch_functional as tf; src=inspect.getsource(tf.logprobs_from_logits_torch_npu); assert 'DKV_FP32_LOGPROB' in src; print('fp32 upcast present')"
ck "OOV TP 修复在场" "global-coord mask"   "import inspect, verl.workers.rollout.vllm_rollout.utils as u; src=inspect.getsource(u.monkey_patch_compute_logits); assert 'get_tensor_model_parallel_rank' in src; print('global-coord mask present')"
ck "checkpoint 引擎" "含 nccl"             "import verl.checkpoint_engine; from verl.checkpoint_engine.base import CheckpointEngineRegistry as R; ks=sorted(R._registry.keys()); assert 'nccl' in ks, ks; print(ks)"

echo "== RL 路径 import 全检 =="
if $PY scripts/vllm018_upgrade/rl/check_rl_imports.py >/tmp/npu_imp.txt 2>&1; then
  echo "PASS  rl imports: $(grep -c '^OK' /tmp/npu_imp.txt) OK, trloo registered"
else
  echo "FAIL  rl imports (详见 /tmp/npu_imp.txt):"; grep '^FAIL' /tmp/npu_imp.txt | head -5; fail=1
fi

echo "== 10: 数据/模型(软性,缺了按指引补) =="
[ -f "$HOME/data/gsm8k/train.parquet" ] && echo "PASS  gsm8k data" || echo "WARN  gsm8k data 缺失 -> PY=$PY bash scripts/vllm018_upgrade/rl/prep_gsm8k.sh"
$PY -c "from huggingface_hub import try_to_load_from_cache; import os,glob; hits=glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B*')); print('PASS  Qwen3-0.6B cached' if hits else 'WARN  Qwen3-0.6B 未缓存 -> snapshot_download 或 SMOKE_MODEL=/abs/path')" 2>/dev/null | tail -1

echo ""
if [ $fail -eq 0 ]; then echo "== 环境体检通过:可直接从 Phase 1(推理 smoke)开始 =="; else echo "== 有 FAIL 项:先修再跑(对照 NPU_EXPERIMENT_PLAN.md Phase 0)=="; fi
exit $fail
