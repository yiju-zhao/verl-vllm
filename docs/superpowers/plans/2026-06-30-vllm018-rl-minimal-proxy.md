# verl RL minimal-proxy pipeline on vLLM 0.18 (Stage 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get a minimal verl RL loop (`trloo` + gsm8k + `Qwen3-0.6B`) running end-to-end on a single GB10 GPU under vLLM 0.18 — rollout → reward → advantage → actor update → weights back into vLLM — for a few clean steps with sane numerics.

**Architecture:** Use the **colocated** standard trainer `verl.trainer.main_ppo` (`RayPPOTrainer`), the simplest single-GPU path that exercises vLLM 0.18 inside the real RL loop (vLLM rollout + sleep-mode weight resync + FSDP actor + advantage + rule-based reward). This is Stage-1 of the goal doc. The **separation** architecture (`recipe.drkernel.main`, `update_weights_from_ipc`), **TP=2** on the two Sparks, and the **real 8B + KernelGYM** target are explicitly OUT OF SCOPE here — they get their own follow-on plans.

**Tech Stack:** verl (`RayPPOTrainer`), vLLM 0.18.1.dev0 (CUDA), torch 2.9.1+cu130, Ray 2.48, FSDP, Python 3.10, env `drkernel310`.

## Global Constraints
- Env python: `/home/yubaifeng/e84381970/envs/drkernel310/bin/python3.10` (call it `$PY`). Do not change torch.
- Run from the port root `/home/yubaifeng/e84381970/experiment/verl-vllm/drkernel-verl-port-drkernel` — **NOT** from `…/verl-vllm/` (a `vllm/` subdir there shadows the installed package).
- Single GPU, single node: `trainer.n_gpus_per_node=1 trainer.nnodes=1 actor_rollout_ref.rollout.tensor_model_parallel_size=1`.
- Algorithm: `algorithm.adv_estimator=trloo` (real drkernel estimator; `core_algos.py:640`).
- Model `Qwen/Qwen3-0.6B` (already in HF cache), data = gsm8k, `rollout.name=vllm`.
- vLLM-0.18 env knobs proven necessary in this env: `VLLM_HOST_IP=127.0.0.1` (host has a Tailscale 100.66.x IP that breaks vLLM's rendezvous) and `VLLM_WORKER_MULTIPROC_METHOD=spawn`. Set both for every run.
- Pre-existing dep skew (NOT from this work): `numpy 2.2.6` (verl wants `<2.0.0`), `tensordict 0.12.3`. Only address if a run actually fails on them.
- Empirical bring-up: fix each runtime breakage following superpowers:systematic-debugging; any verl-side fix must be minimal and, if version-specific, version-gated (don't regress older vLLM).
- Commit after every task. Trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Artifacts live under `scripts/vllm018_upgrade/rl/`.

---

### Task 1: Env readiness — data, model, RL-path imports

**Files:**
- Create: `scripts/vllm018_upgrade/rl/prep_gsm8k.sh`
- Create: `scripts/vllm018_upgrade/rl/check_rl_imports.py`

**Interfaces:**
- Produces: gsm8k parquet at `$HOME/data/gsm8k/{train,test}.parquet`; a check that `verl.trainer.main_ppo`, `RayPPOTrainer`, and the vLLM rollout worker import under 0.18.

- [ ] **Step 1: Write the gsm8k prep script**

```bash
# scripts/vllm018_upgrade/rl/prep_gsm8k.sh
#!/usr/bin/env bash
set -euo pipefail
PY=/home/yubaifeng/e84381970/envs/drkernel310/bin/python3.10
cd /home/yubaifeng/e84381970/experiment/verl-vllm/drkernel-verl-port-drkernel
$PY examples/data_preprocess/gsm8k.py --local_dir "$HOME/data/gsm8k"
ls -la "$HOME/data/gsm8k"/{train,test}.parquet
```

- [ ] **Step 2: Run it**

Run: `bash scripts/vllm018_upgrade/rl/prep_gsm8k.sh`
Expected: `train.parquet` and `test.parquet` exist under `~/data/gsm8k`. (If the script needs `datasets`/network, it uses the env's already-installed `datasets`.)

- [ ] **Step 3: Write the RL-path import check**

```python
# scripts/vllm018_upgrade/rl/check_rl_imports.py
import importlib, sys, vllm
print("vllm", vllm.__version__)
mods = [
    "verl.trainer.main_ppo",
    "verl.trainer.ppo.ray_trainer",
    "verl.trainer.ppo.core_algos",
    "verl.workers.fsdp_workers",
    "verl.workers.rollout.vllm_rollout.vllm_rollout",
]
bad = False
for m in mods:
    try:
        importlib.import_module(m); print("OK  ", m)
    except Exception as e:  # noqa: BLE001
        print("FAIL", m, type(e).__name__, e); bad = True
# trloo must be a registered estimator
from verl.trainer.ppo.core_algos import AdvantageEstimator
assert AdvantageEstimator.TRLOO.value == "trloo", "trloo missing"
print("trloo registered OK")
sys.exit(1 if bad else 0)
```

- [ ] **Step 4: Run the import check**

Run: `cd <port-root> && $PY scripts/vllm018_upgrade/rl/check_rl_imports.py`
Expected: `vllm 0.18…`, all `OK`, `trloo registered OK`, exit 0. Fix any `FAIL` by adding a 0.18 version-gate branch (same approach as the upgrade); most should already be clean (the upgrade oracle passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/vllm018_upgrade/rl/prep_gsm8k.sh scripts/vllm018_upgrade/rl/check_rl_imports.py
git commit -m "test(rl): gsm8k prep + RL-path import check under vllm 0.18"
```

---

### Task 2: Minimal `trloo` single step (the integration test)

**Files:**
- Create: `scripts/vllm018_upgrade/rl/run_trloo_qwen3_0.6b_gsm8k.sh`
- Create: `scripts/vllm018_upgrade/rl/RL_NOTES.md` (breakages + fixes log)

**Interfaces:**
- Consumes: gsm8k data + model from Task 1.
- Produces: proof that one full PPO/`trloo` step completes on 1 GPU under vLLM 0.18 (rollout → reward → advantage → actor update → weights reloaded into vLLM).

- [ ] **Step 1: Write the minimal run script** (adapted from `examples/cispo_trainer/run_cispo_qwen2_5_0_5b_gsm8k.sh`, shrunk to 1 GPU / tiny batch / `trloo` / 1 step)

```bash
# scripts/vllm018_upgrade/rl/run_trloo_qwen3_0.6b_gsm8k.sh
#!/usr/bin/env bash
set -x
PY=/home/yubaifeng/e84381970/envs/drkernel310/bin/python3.10
cd /home/yubaifeng/e84381970/experiment/verl-vllm/drkernel-verl-port-drkernel
export VLLM_HOST_IP=127.0.0.1 VLLM_WORKER_MULTIPROC_METHOD=spawn
train=$HOME/data/gsm8k/train.parquet ; test=$HOME/data/gsm8k/test.parquet
STEPS="${STEPS:-1}"
$PY -m verl.trainer.main_ppo \
  algorithm.adv_estimator=trloo \
  data.train_files="['$train']" data.val_files="['$test']" \
  data.train_batch_size=8 data.max_prompt_length=512 data.max_response_length=256 \
  data.filter_overlong_prompts=True data.truncation=error \
  actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.logger='["console"]' \
  trainer.project_name=vllm018_rl_smoke trainer.experiment_name=trloo_qwen3_0.6b \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.save_freq=-1 trainer.test_freq=-1 \
  trainer.total_training_steps="$STEPS" "$@"
```
Make executable: `chmod +x scripts/vllm018_upgrade/rl/run_trloo_qwen3_0.6b_gsm8k.sh`.

- [ ] **Step 2: Run one step in the background, capture the log**

Run:
```bash
bash scripts/vllm018_upgrade/rl/run_trloo_qwen3_0.6b_gsm8k.sh 2>&1 | tee /tmp/rl_trloo_1step.log
```
Expected (first attempt likely fails somewhere — that's the point). Success looks like: vLLM rollout generates, a reward/score is computed for gsm8k, advantages computed, one actor optimizer step runs, and updated weights are pushed back into the vLLM engine, then the process exits after step 1 (`total_training_steps=1`). Watch for the `step:1` / training-metrics console line.

- [ ] **Step 3: Triage & fix the first breakage**

Read the traceback in `/tmp/rl_trloo_1step.log`. Likely areas and how to handle:
- **vLLM engine init** (spawn/host-ip/attention backend): env knobs are already set; if a Tailscale-IP or fork-CUDA error appears, confirm `VLLM_HOST_IP`/`spawn` took effect. If first-gen JIT is slow but progressing, wait — single-GPU TP=1 autotune completed in ~1 min in the bare smoke.
- **Weight resync into vLLM** (the vLLM-0.18-fragile part — sleep/wake + load_weights): if it errors on a vLLM worker/`collective_rpc`/`load_weights` signature, add a 0.18-gated shim next to the existing gates (`verl/utils/vllm/*`), following systematic-debugging.
- **dep skew** (numpy2/tensordict): only if a concrete error points there.
Record each breakage + fix in `RL_NOTES.md`. Re-run Step 2 after each fix.

- [ ] **Step 4: Confirm one clean step**

Run: `grep -E "step:1|'step': 1|actor/.*loss|rollout" /tmp/rl_trloo_1step.log | tail`
Expected: evidence of a completed step 1 with training metrics (a loss value, an advantage/return, a reward/score), and a clean process exit (rc 0).

- [ ] **Step 5: Commit**

```bash
git add scripts/vllm018_upgrade/rl/run_trloo_qwen3_0.6b_gsm8k.sh scripts/vllm018_upgrade/rl/RL_NOTES.md verl/
git commit -m "feat(rl): minimal trloo+gsm8k single step on vllm 0.18 (Qwen3-0.6B, 1 GPU)"
```

---

### Task 3: Multi-step + numerics (Stage-1 done)

**Files:**
- Modify: `scripts/vllm018_upgrade/rl/RL_NOTES.md` (record the numeric observations)

**Interfaces:**
- Consumes: the working single step (Task 2).
- Produces: proof the loop is stable for several steps with sane train-vs-rollout logprob consistency (the `trloo` importance ratio is finite and not near-fully-masked).

- [ ] **Step 1: Run ~5 steps**

Run:
```bash
STEPS=5 bash scripts/vllm018_upgrade/rl/run_trloo_qwen3_0.6b_gsm8k.sh 2>&1 | tee /tmp/rl_trloo_5step.log
```
Expected: 5 steps complete without crashing.

- [ ] **Step 2: Check numeric sanity (logprob consistency / ratio)**

Run:
```bash
grep -iE "ratio|kl|importance|mask|actor/pg_loss|entropy|reward|score" /tmp/rl_trloo_5step.log | tail -30
```
Expected: the importance ratio (train-engine logprob vs rollout logprob) stays finite and near 1 on the first inner epoch; KL small; reward/score present and not NaN; response mask not ~100%. If ratios explode or masking is ~100% (the failure the other team hit at TP2), record it in `RL_NOTES.md` — that is the trigger to consider `DKV_FP32_LOGPROB` / rollout-sanitize in a follow-on (do NOT port pre-emptively).

- [ ] **Step 3: Record the verdict**

Append to `RL_NOTES.md`: steps completed, whether numerics were sane, and any anomaly. This is the Stage-1 definition-of-done record.

- [ ] **Step 4: Commit**

```bash
git add scripts/vllm018_upgrade/rl/RL_NOTES.md
git commit -m "test(rl): minimal trloo pipeline stable for 5 steps on vllm 0.18 + numerics note"
```

**Stage-1 done when:** `run_trloo_qwen3_0.6b_gsm8k.sh` completes ≥5 steps on 1 GPU under vLLM 0.18 with sane numerics, and the RL-path import check passes.

---

## Follow-on plans (out of scope here — noted so nothing is lost)
- **Plan RL-2 (separation + weight-IPC):** port the minimal loop to `recipe.drkernel.main` /
  the separation architecture, exercising `update_weights_from_ipc` + `vLLMColocateWorkerExtension`.
- **Plan RL-3 (TP=2 on the two Sparks):** run the loop with `rollout.tensor_model_parallel_size=2`
  over the 2-Spark cluster (per `scripts/vllm018_upgrade/TP2_CLUSTER_RUNBOOK.md`).
- **Plan RL-4 (real target):** `Qwen3-8B` + KernelGYM reward via `recipe.drkernel.main`.
