# Goal: verl RL full pipeline on vLLM 0.18 (CUDA)

- **Date:** 2026-06-30
- **Status:** Stage-1 (M0–M4, single-GPU minimal proxy) IMPLEMENTED & verified 2026-07-02 —
  5 clean `trloo` steps on vLLM 0.18, `ppo_kl=0` (train/rollout logprob consistent), no
  hardening needed at this scale (see `scripts/vllm018_upgrade/rl/RL_NOTES.md`). M5 (TP=2
  RL), M6 (8B+KernelGYM), separation arch = follow-on plans RL-2/3/4, not started.
- **Repo/branch:** `drkernel-verl-port-drkernel/` @ `vllm-0.18-upgrade`
- **Depends on:** the vLLM 0.18 CUDA upgrade (DONE — imports clean, generation smoke PASS)
  and the TP=2 two-Spark cluster (DONE — `scripts/vllm018_upgrade/TP2_CLUSTER_RUNBOOK.md`).

## 1. Objective
Get verl's RL training **full pipeline** running on vLLM 0.18, using the project's
real algorithm **`trloo`**, validated in two stages (minimal proxy → real). Frame it
as normal pipeline bring-up; only add correctness hardening (fp32 logprob / rollout
sanitize) **if we actually observe** a numeric problem — not pre-emptively.

## 2. Confirmed scope decisions
- **Backend:** CUDA. Single-GPU first for fast iteration; **TP=2 across the two Sparks**
  is now a first-class milestone (the cluster is up, and TP=2 is where the interesting
  correctness questions live — cf. the other team's TP2 investigation).
- **Validation:** two stages —
  1. **Minimal proxy:** `Qwen3-0.6B` + gsm8k (simple reward) + `trloo`, a few clean
     steps with sane numerics.
  2. **Real target:** `Qwen3-8B` + KernelGYM reward via `recipe.drkernel.main`, short
     run showing reward trend.
- **Algorithm:** `trloo` (`algorithm.adv_estimator=trloo`, `core_algos.py:640`) — matches
  the drkernel line (`recipe/drkernel/scripts/rl/drkernel_kernel_train_native_8b.sh`).

## 3. The pipeline & its 0.18-sensitive integration points
Server-based "separation" architecture: trainer (FSDP) ⇄ vLLM `AsyncLLM` rollout server
via `collective_rpc`; weight resync via `update_weights_from_ipc` + bucketed IPC +
`vLLMColocateWorkerExtension`.
1. **Rollout server** (`vllm_async_server.py`, colocate worker ext) — only bare `vllm.LLM`
   tested so far.
2. **Weight resync** (`vllm_rollout.py:162` `update_weights` → `update_weights_from_ipc`,
   `bucketed_weight_transfer.py`) — **untested**; most 0.18-fragile (vllm worker/
   `collective_rpc` internals).
3. **sleep/wake** (`collective_rpc("sleep")`, `VLLM_SLEEP_LEVEL`) — untested on 0.18.
4. **Training engine** (`experimental/separation/ray_trainer.py`, FSDP + Ray + data +
   reward) — untested in this env; watch dep skew (numpy 2.2.6 vs verl `<2.0.0`).
5. **logprob numerics** (`verl/utils/torch_functional.py`; `trloo` uses rollout logprobs) —
   untested; the conditional-hardening point.

## 4. Milestones
- **M0 — env readiness:** resolve/limit dep skew (numpy2/tensordict), confirm Ray+FSDP on
  GB10 sm_121; KernelGYM server reachable (for Stage 2).
- **M1 — rollout server path (TP=1):** bring up verl's `vllm_async_server` with Qwen3-0.6B
  single-GPU colocate; confirm the colocate worker extension attaches under 0.18 and
  generates.
- **M2 — weight resync:** exercise `update_weights_from_ipc` end-to-end (one train→infer
  push) + sleep/wake on 0.18.
- **M3 — minimal `trloo` single step (Stage-1a):** Qwen3-0.6B + gsm8k + `trloo`, 1 clean
  step (rollout → reward → advantage → update → weight resync).
- **M4 — minimal `trloo` multi-step + numerics (Stage-1b):** a few stable steps; verify
  train-engine vs rollout logprob consistency (`trloo` IS ratio sane). **= minimal-proxy
  "pipeline works".**
- **M5 — TP=2 rollout on the two Sparks:** run M1–M4 with `rollout.tensor_model_parallel_size=2`
  over the 2-Spark cluster (per the TP2 runbook: per-node `VLLM_HOST_IP`,
  `enable_flashinfer_autotune=False`, NCCL on ConnectX). This is the parity point with the
  other team's TP2 work.
- **M6 — real target (Stage-2):** Qwen3-8B + KernelGYM via `recipe.drkernel.main`, short run,
  reward trend. (Needs the multi-GPU separation deployment; scope/size per available HW.)
- **(conditional) M-fix:** only if M4/M5/M6 show logprob/ratio anomalies, evaluate porting
  `DKV_FP32_LOGPROB` / `DKV_SANITIZE_ROLLOUT`. Not assumed.

## 5. Definition of done
- **Stage 1:** minimal `trloo` (Qwen3-0.6B + gsm8k) runs ≥5 clean steps on CUDA with sane
  numerics (single-GPU **and** TP=2 across the two Sparks).
- **Stage 2:** real `trloo` (Qwen3-8B + KernelGYM) short run with reward moving sanely.

## 6. Key risks
1. Weight resync 0.18 API churn (highest) — `vLLMColocateWorkerExtension`/`collective_rpc`.
2. logprob consistency (the conditional-hardening trigger; the other team hit this at TP2).
3. Single-GPU colocate memory for 8B (M6 may need the multi-GPU cluster, not one Spark).
4. Dep skew (numpy 2.2.6) surfacing in the full trainer.
5. TP=2 perf bound by TCP-NCCL until RoCE is tuned (functional now; slow all-reduce).

## 7. Out of scope
NPU RL (separate track), performance tuning beyond "runs", upstreaming.
