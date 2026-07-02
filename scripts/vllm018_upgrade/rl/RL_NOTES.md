# RL smoke (trloo + gsm8k, Qwen3-0.6B, 1 GPU) on vLLM 0.18 — breakages & fixes

Goal: run `verl.trainer.main_ppo` (RayPPOTrainer, colocated FSDP+vLLM) for ONE `trloo`
step to prove the full RL loop works on the vLLM 0.18 CUDA build:
rollout (vLLM) -> gsm8k rule reward -> `trloo` advantage -> FSDP actor update ->
updated weights resynced into the vLLM engine.

Env: `/home/yubaifeng/e84381970/envs/drkernel310/bin/python3.10`, torch 2.9.1+cu130,
vllm `0.18.1.dev0+...cu130` (built from source), GB10 sm_121 aarch64, single GPU.
Launcher: `run_trloo_qwen3_0.6b_gsm8k.sh` (this dir).

Run/inspect:
```
bash scripts/vllm018_upgrade/rl/run_trloo_qwen3_0.6b_gsm8k.sh 2>&1 | tee /tmp/rl_trloo_1step.log
grep -E "step:1|actor/|reward|update_weights done" /tmp/rl_trloo_1step.log | tail
```

---

## Breakage 1 — Ray joins a dead/foreign cluster, or `ray.init` hangs on node-IP autodetect

Two intertwined machine-state problems, both fixed at the launcher level (no verl code change):

### 1a. Stale `ray_current_cluster` -> ConnectionError
First launch printed `Connecting to existing Ray cluster at address: 192.168.1.101:6379`
then died with:
```
ConnectionError: Could not read 'session_name' from GCS. Did GCS start successfully?
```
Cause: a leftover `/tmp/ray/ray_current_cluster` file (from the 2026-06-30 TP=2 two-Spark
session) still pointed `ray.init()` at the ConnectX head `192.168.1.101:6379`, whose GCS is
long gone. `RAY_ADDRESS` was unset; Ray falls back to that file.
Also note: a **root-owned** Ray GCS (pid from a python3.12 install) has been squatting on
port 6379 for hours — not ours, cannot kill; irrelevant once we stop auto-joining it.

Fix (environment cleanup, per the upgrade runbook): before each run,
`ray stop --force` and `rm -rf /tmp/ray`. `ray.init()` then starts a fresh *local* cluster
(GCS binds a random free port, not 6379).

### 1b. `ray.init()` / `ray start` hangs forever on node-IP autodetection
After clearing the stale file, a bare `ray.init()` (and `ray start --head`) hung
indefinitely. Isolated it: `ray.init(_node_ip_address='127.0.0.1')` returns in ~1s, while
letting Ray auto-detect the node IP never returns. Same root cause as the vLLM
`VLLM_HOST_IP=127.0.0.1` fix — this host has a Tailscale `100.66.x` IP that Ray's
autodetect latches onto, then GCS/raylet binding hangs. (Dashboard was ruled out: init with
the dashboard ON but node-IP pinned to loopback works fine.)

Fix (launcher): force Ray's node IP to loopback via a Hydra override on the verl command,
and disable the usage-stats path (this Ray build crashes in `usage_lib`):
```
export RAY_USAGE_STATS_ENABLED=0
... +ray_kwargs.ray_init._node_ip_address=127.0.0.1 ...
```
`main_ppo.run_ppo` calls `ray.init(**config.ray_kwargs.ray_init)`, so the override lands as
`ray.init(_node_ip_address='127.0.0.1')`. After this: `Started a local Ray instance` and the
TaskRunner comes up.

---

## Breakage 2 — `flash_attn` not installed -> FSDP actor/ref model load fails

During `WorkerDict.actor_rollout_ref_init_model()`:
```
ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the
following error: the package flash_attn seems to be not installed.
```
Cause: verl loads the FSDP actor/ref HF model with `attn_implementation="flash_attention_2"`
by default (`verl/workers/config/model.py:186`, `override_config.get("attn_implementation",
"flash_attention_2")`), and `use_remove_padding=True` monkey-patches
`_flash_attention_forward` to pack sequences and rely on flash-attn's varlen (cu_seqlens)
block-diagonal masking. But `flash_attn` is **not installed** in this env and has no prebuilt
wheel for sm_121 / aarch64 / cu130 (vLLM 0.18 itself uses flashinfer, not flash_attn); a
source build is heavy/risky and out of scope. (This only affects the **FSDP training**
model — vLLM's own rollout attention is independent and already smoke-passes.)

Fix (launcher, this env only): load the training model with PyTorch SDPA and disable the
padding-free path (sdpa can't do varlen block-diagonal masking across packed sequences, so
remove_padding must be off for correctness):
```
+actor_rollout_ref.model.override_config.attn_implementation=sdpa
actor_rollout_ref.model.use_remove_padding=False
```
CONCERN: this drops the remove-padding throughput optimization the brief's verbatim script
requested (`use_remove_padding=True`). It is the correct choice given no flash_attn here; to
restore it, install a flash-attn built for sm_121 and revert both overrides.

Progress after this fix: model loads, vLLM 0.18 rollout runs, and the pre-train validation
completes end-to-end (rollout + gsm8k rule reward):
`step:0 - val-core/openai/gsm8k/acc/mean@1: ~0.009`. Confirms the vLLM-0.18 rollout + reward
path works. The crash then moved into the FSDP training path (breakage 3).

---

## Breakage 3 — `flash_attn` again: hard import in the FSDP log-prob unpad path

Code change (minimal, guarded fallback — the only verl source edit).

After the sdpa fix, the training step crashed in `_compute_old_log_prob`:
```
ray_trainer.py:1170 _compute_old_log_prob
  -> verl/workers/utils/padding.py:53 left_right_2_no_padding
    -> verl/utils/attention_utils.py:96 unpad_input
      -> verl/utils/attention_utils.py:30 _get_attention_functions
        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
ModuleNotFoundError: No module named 'flash_attn'
```
Cause: the FSDP engine's forward-prep `left_right_2_no_padding` converts left/right-padded
batches to packed nested tensors and needs `unpad_input`/`index_first_axis`. On CUDA,
`_get_attention_functions` imports these from `flash_attn.bert_padding` **unconditionally** —
independent of `use_remove_padding` (so the breakage-2 fix does not avoid it). This runs on
every actor/ref log-prob and actor update, so training cannot start without it.

Key point: these four helpers (`index_first_axis`, `pad_input`, `unpad_input`, `rearrange`)
are **pure-torch** tensor gather/scatter ops — they do NOT invoke flash-attn CUDA kernels.
verl already vendors a device-agnostic copy at `verl/utils/npu_flash_attn_utils.py` (used on
NPU), whose `unpad_input` returns the same tuple the caller unpacks.

Fix (`verl/utils/attention_utils.py`, `_get_attention_functions`): wrap the CUDA
`from flash_attn.bert_padding import ...` in try/except ImportError and, on failure, fall
back to `verl.utils.npu_flash_attn_utils` (+ `einops.rearrange`). This only triggers when
flash_attn is genuinely absent, so it is a no-op / no-regression for envs that ship it.
No version gate needed (it is presence-gated, matching the existing NPU/CUDA branch style).

NOTE on correctness: the FSDP engine forward runs on `torch.nested` jagged tensors, which
encode per-sequence boundaries, so SDPA attends within each sequence correctly even without
flash-attn varlen packing. So sdpa + this fallback is numerically sound for training.

---

## RESULT — one clean `trloo` step, rc 0

After the three fixes above, a full run completes step 1 and exits `rc 0`
(`/tmp/rl_trloo_1step.log`, `EXITED rc=0`). The whole RL loop executed on vLLM 0.18:

```
step:1 - actor/entropy:1.459 - actor/pg_loss:0.0 - actor/kl_loss:0.0 - actor/loss:0.0
         actor/kl_coef:0.001 - actor/grad_norm:0.0 - actor/lr:1e-06 - training/global_step:1
         response_length/mean:256 - prompt_length/mean:98.6
timing_s/gen:3.67  reward:3.4e-05  old_log_prob:2.53  ref:4.72  adv:0.00099
timing_s/update_actor:6.21  update_weights:2.59  step:19.73  throughput:575 tok/s
```

Every stage of the loop ran and was timed:
- `gen` — vLLM 0.18 rollout (async server, `vLLMColocateWorkerExtension`, enforce_eager, TP=1)
- `reward` — gsm8k rule-based scorer (no external server)
- `old_log_prob` + `ref` — FSDP actor/ref log-prob (uses the breakage-3 unpad fallback)
- `adv` — `trloo` advantage on the driver
- `update_actor` — FSDP forward/backward + optimizer step (grad checkpointing, sdpa)
- `update_weights` — **the vLLM-0.18-fragile weight resync**: FSDP weights bucketed over ZMQ
  into the live vLLM engine (`update_weights_from_ipc` -> `load_weights` ->
  `process_weights_after_loading`). Ran in 2.59s with no error. The 0.18 weight-sync APIs
  (`vllm.model_executor.model_loader.utils.process_weights_after_loading(model, model_config,
  device)`, `model_runner.model.load_weights`) already match verl's calls — **no shim needed**
  in `verl/utils/vllm/`.

OBSERVATION (not a failure): all loss/advantage/reward metrics are `0.0`. Qwen3-0.6B (base,
untrained) gets 0 gsm8k reward and every response is truncated at 256 tokens
(`response_length/clip_ratio:1.0`), so rewards are uniform -> leave-one-out `trloo` advantage
is 0 -> zero policy gradient -> `grad_norm 0`, weights unchanged. The optimizer step and
weight resync still execute; this is the expected degenerate-but-valid result for a 1-step
smoke on a tiny batch, and it exercises the entire pipeline. Longer response length / more
steps would produce non-zero signal but are unnecessary to prove the loop.

## How to reproduce
```bash
# ensure no stale Ray state (foreign root GCS squats on :6379; TP=2 leftovers in /tmp/ray)
/home/yubaifeng/e84381970/envs/drkernel310/bin/ray stop --force; rm -rf /tmp/ray
bash scripts/vllm018_upgrade/rl/run_trloo_qwen3_0.6b_gsm8k.sh 2>&1 | tee /tmp/rl_trloo_1step.log
grep -E "step:1|actor/.*loss|update_weights done|EXITED rc=" /tmp/rl_trloo_1step.log | tail
```

## Files changed
- `scripts/vllm018_upgrade/rl/run_trloo_qwen3_0.6b_gsm8k.sh` (new launcher; carries the
  breakage-1/2 env + Hydra overrides)
- `scripts/vllm018_upgrade/rl/RL_NOTES.md` (this file)
- `verl/utils/attention_utils.py` (breakage-3: presence-gated flash_attn -> vendored-copy
  fallback; the only verl source edit)


---

## Stage-1 DoD — multi-step + numerics (Task 3)

Command (Task-2 launcher + longer responses so gsm8k answers aren't truncated → non-degenerate signal, per the Task-2 review's Minor finding):
```
STEPS=5 bash scripts/vllm018_upgrade/rl/run_trloo_qwen3_0.6b_gsm8k.sh \
    data.max_response_length=768 actor_rollout_ref.rollout.n=8 2>&1 | tee /tmp/rl_trloo_5step.log
```

Result: **PASS.** All 5 `trloo` steps completed, `EXITED rc=0`, weight resync every step
(`timing_s/update_weights` ~2.6s). **0 NaN/Inf** anywhere.

Per-step numerics (the load-bearing check = train-engine vs rollout logprob consistency):

| step | actor/ppo_kl | critic/score/mean | critic/advantages/mean | actor/pg_loss | actor/grad_norm |
|---|---|---|---|---|---|
| 1 | 0.0 | 0.031 | -0.0033 | 0.0033 | 1.93 |
| 2 | 0.0 | 0.219 | -0.0318 | 0.0318 | 3.41 |
| 3 | 0.0 | 0.219 | -0.0093 | 0.0093 | 2.50 |
| 4 | 0.0 | 0.156 | -0.0299 | 0.0299 | 1.27 |
| 5 | 0.0 | 0.219 | -0.0566 | 0.0566 | 2.64 |

- **`actor/ppo_kl == 0.0` on every step** → train-engine log-prob equals the rollout
  log-prob on the first inner epoch → importance ratio ≈ 1, NOT exploding and NOT
  near-fully-masked. This is the failure mode the other team reported at TP2; on this
  single-GPU CUDA path under vLLM 0.18 it is clean.
- Non-degenerate this time (vs the Task-2 1-step run): the 0.6B model scores non-zero
  gsm8k reward once responses fit in 768 tokens, so advantages, pg_loss and grad_norm
  are all real/non-zero — the optimizer + weight-resync run on a genuine gradient.
- `response_length/clip_ratio` ≈ 0.47 at step 5 (about half the responses hit the 768
  cap), `response/aborted_ratio` = 0.0.

**Conclusion:** Stage-1 (minimal-proxy RL pipeline on vLLM 0.18, single GPU) is DONE —
the full `trloo` loop runs stably for ≥5 steps with sane, consistent numerics. No
correctness hardening (fp32 logprob / rollout-sanitize) was needed at this scale.

> Scope note (from final review): the `attention_utils.py` flash_attn→vendored fallback
> only covers the no-remove-padding FSDP engine path used here. The remove-padding rmpad
> helpers (`verl/utils/torch_functional.py`) and the Megatron paths still `import flash_attn`
> directly, so they would still require flash_attn on a wheel-less CUDA build if
> `use_remove_padding=True` or Megatron were used. Not a regression (pre-existing); flash_attn
> is NOT globally optional in verl — only this Stage-1 FSDP path is.
