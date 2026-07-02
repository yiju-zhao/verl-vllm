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

---

## RL-3 — TP=2 across the two Sparks (goal-doc M5)

Launcher: `run_trloo_tp2.sh` (rollout TP=2 spanning gx10-090e + spark-bruce over 200G
ConnectX; existing Ray cluster on :6380; per-node VLLM_HOST_IP from ray-start env).

Breakages hit & fixed on the way:
1. **Ray memory monitor kills bruce workers** — GB10 unified memory makes vLLM's GPU
   preallocation look like huge process RSS; the node had 111 GB free. Fix:
   `RAY_memory_monitor_refresh_ms=0` in the ray-start env on both nodes.
2. **Foreign root-owned Ray on :6379** — our cluster moved to `--port=6380`.
3. **verl bug: boolean False dropped from vllm server CLI** —
   `build_cli_args_from_config` skipped False booleans, so
   `engine_kwargs.vllm.enable_flashinfer_autotune=False` never reached `vllm serve`,
   the platform default (True) applied, and the flashinfer autotuner hung over
   cross-node TCP-NCCL (same hang as the TP=2 inference bring-up). Fixed: False now
   emits `--no-<flag>` (vLLM bool args are argparse.BooleanOptionalAction; verified
   end-to-end through AsyncEngineArgs + FlexibleArgumentParser).

Result: **PASS.** STEPS=5, rc=0, cross-node placement confirmed (FSDP WorkerDict rank1,
vLLMHttpServer, AgentLoopWorkers on 192.168.1.106). ~81 s/step (gen ~30 s — TCP-NCCL
cross-node all-reduce; RoCE tuning would cut this), update_weights ~4.5 s/step.

| step | actor/ppo_kl | critic/score/mean | actor/pg_loss | actor/grad_norm |
|---|---|---|---|---|
| 1 | 0.0 | 0.0     | 0.0    | 0.0   |
| 2 | 0.0 | 0.141   | 0.0155 | 1.22  |
| 3 | 0.0 | 0.297   | 0.0279 | 1.86  |
| 4 | 0.0 | 0.219   | 0.0284 | 1.69  |
| 5 | 0.0 | 0.266   | 0.0379 | 2.52  |

0 NaN/Inf; `response/aborted_ratio` 0.0. (Step 1's batch scored 0 → legitimately zero
advantage/gradient; steps 2–5 non-degenerate.)

**Conclusion: vLLM 0.18 TP=2 RL training numerics are CLEAN on this stack** —
`ppo_kl == 0` every step means the train-engine log-probs match the cross-node TP=2
rollout log-probs exactly; no ratio explosion, no mass masking. The failure mode the
other team reported at TP2 (token-271 spike, ~97% RS masking) does NOT reproduce here.
No fp32-logprob / rollout-sanitize hardening needed at this scale on CUDA.

---

## CORRECTION — proper consistency diagnostics (pearson / RS-IS ratio)

**The earlier "numerics are CLEAN / ppo_kl=0" conclusions (Stage-1 and RL-3 above) used
the wrong instrument.** In our config `old_log_prob` is recomputed by the training engine
(`use_rollout_log_probs` not set) and batch==mini-batch → one update per step → `ppo_kl`
is computed against the engine's own logprobs *before* the update → **trivially 0**. It
never measured vllm-rollout vs train-engine consistency.

Proper instruments (per verl's rollout-correction tooling; same ones the other team used):
`actor_rollout_ref.rollout.calculate_log_probs=True` → `training/rollout_actor_probs_pearson_corr`
+ `training/rollout_probs_diff_*`; `algorithm.rollout_correction.rollout_is=token` →
`rollout_corr/*` IS-ratio stats.

TP=2 (two Sparks), Qwen3-0.6B bf16, sdpa training engine, 2 steps, no val:

| metric | step 1 | step 2 |
|---|---|---|
| pearson_corr (rollout vs actor logprobs) | 0.758 | 0.808 |
| probs_diff_mean / _max | 0.083 / ~1.0 | 0.056 / 1.0 |
| rollout_is_mean (trunc@2) | 0.896 | 0.936 |
| rollout_is fraction_low | 10.7% | 6.6% |
| rollout_is_min | 2e-9 | 2e-9 |
| eff_sample_size | 0.90 | 0.94 |

**Verdict: train/rollout mismatch is REAL and non-trivial on this stack** — far milder
than the other team's NPU/TP2 report (~97% RS masking vs our ~7-11% fraction_low), but
pearson 0.76-0.81 is well below the ~0.99 a clean setup shows, and isolated tokens
disagree completely (probs_diff_max≈1, is_min≈2e-9).

Why prior runs still trained sanely: without `use_rollout_log_probs=True` the gradient
uses engine-recomputed old_log_prob (self-consistent); the mismatch manifests as
off-policy drift, not a poisoned ratio. The real drkernel config DOES use rollout
logprobs, so this matters there.

Also fixed en route: a second engine crash was stale-process memory (crashed run left
~86 GB held on bruce; vllm init then saw 33.3/119.69 GiB free < 0.5 utilization —
"EngineDeadError" runs must be followed by a node cleanup). The earlier
`sample_tokens timed out` was slowness (1319-prompt validation + first logprobs pass
over TCP-NCCL exceeding VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300), not a deadlock.

**Open question (control experiment running): TP=1 with identical diagnostics** — if
TP=1 pearson ≈0.99, TP=2 is the mismatch source (echoes the other team); if TP=1 is
also ~0.8, it's an engine-pair difference (vllm bf16 vs FSDP+sdpa bf16) independent
of TP.

### Control result: TP=1 vs TP=2 — TP=2 is EXONERATED

Same diagnostics, same cluster, single GPU (TP=1): pearson 0.752/0.808,
diff_mean 0.084/0.055, fraction_low 10.6%/6.4%, is_min 2e-9 — **statistically
identical to TP=2** (0.758/0.808, 0.083/0.056, 10.7%/6.6%, 2e-9).

Conclusions:
1. **Cross-node TP=2 on vllm 0.18 adds ZERO measurable logprob mismatch** — the TP
   mechanism itself is numerically faithful on this stack. The other team's TP2-specific
   blowup does not reproduce on CUDA (their issue is elsewhere: NPU kernels, their TP
   impl, or their engine pair).
2. The mismatch (pearson ~0.75-0.81) is inherent to the ENGINE PAIR on this box:
   vllm-0.18 bf16 (FLASH_ATTN backend) rollout vs FSDP training engine forced to
   **sdpa** + `use_remove_padding=False` + pure-torch pad fallback (no flash_attn
   wheel for sm_121). Candidate contributors: attention-kernel numerics (sdpa vs FA2),
   bf16 accumulation-order differences, tiny-model (0.6B) amplification.
3. Extreme disagreements (probs_diff_max≈1, is_min≈2e-9) concentrate in isolated
   tokens — same *shape* as the other team's token-271 spike, far smaller magnitude.
4. Practical: token-IS truncation @2.0 already yields ESS 0.90-0.94 → verl's
   rollout-correction is a workable mitigation as-is; for the drkernel config
   (`use_rollout_log_probs=True`) either enable rollout_correction or align the
   engine pair before trusting gradients.

### ROOT CAUSE FOUND — token-level localization (final)

Per-token dump (`VERL_LOGPROB_DIAG_DUMP` hook in `verl/utils/debug/metrics.py`,
analyzer `analyze_logprob_diag.py`), one step, TP=1, 64 seqs x 768 tokens:

- Extreme disagreements (prob diff>0.5): **0% in deciles 0-7** of every sequence;
  16.6% in decile 8, **62% in decile 9**. Onset varies per sequence (~630-660) →
  content-dependent, not positional. 40/64 last tokens extreme, 0/64 first tokens.
- Offending tokens: '0', ' ', '1', ',', '.', ' the' — **degenerate numeric repetition
  loops** in sequences that hit the 768 cap (48/64 capped; clip_ratio 0.47).
- Direction: rollout-confident/actor-not on 3650/3655 extreme tokens.

**Pearson on healthy tokens: 0.9993; fraction_low: 0.00%.** The vllm-0.18 / FSDP
engine pair is numerically excellent everywhere except chaotic repetition tails,
where peaked distributions amplify any engine difference into full flips.

Final attribution table:
| hypothesis | verdict |
|---|---|
| TP=2 tensor-parallel numerics | innocent (identical to TP=1) |
| attention kernel (sdpa vs eager) | minor (~+0.02 pearson) |
| training-side bf16 forward | innocent (fp32 changed nothing) |
| sampling scaling (temp/top-p) | innocent (1.0/1.0/-1) |
| **degenerate truncated-tail tokens** | **ROOT CAUSE (100% of extremes)** |

Practical guidance for vllm-0.18 RL on this stack:
1. The engine pair needs NO numeric hardening (no fp32-logprob, no kernel work).
2. Handle degenerate tails with standard RL hygiene: overlong filtering/penalty,
   adequate max_response_length for the task, and/or rollout-correction RS/veto
   (token-IS trunc@2 gives ESS 0.90+; RS would mask exactly these tails).
3. Reinterpretation of the other team's ~97% RS masking: on our stack that magnitude
   would require nearly all tokens degenerate — their failure (token-271 spike at
   TP2, early positions) is a different phenomenon; ours shows ZERO early-position
   anomalies.

---

## RL-2 — fully-async separation (the drkernel production deployment shape) — PASS

`recipe.drkernel.main` is built on `verl.experimental.fully_async_policy`; we validated
that core end-to-end on vLLM 0.18 with the minimal proxy (trloo + gsm8k + Qwen3-0.6B),
scaled to the 2-Spark cluster: **TRAINER pool = 1 GPU (head), ROLLOUTER pool = 1 GPU
(bruce) — fully separated, cross-node**. Launcher: `run_trloo_fullyasync_1p1.sh`.

Result: 3/3 training steps (43/34/31 s/it), rc=0. **NCCL checkpoint-engine weight
sync working**: `_fit_update_weights timing_s/param_sync ≈ 2.0–2.3 s`, param_version
incrementing every step (`trigger_parameter_sync_step=1`) — this is the trainer→rollouter
weight push the production deployment uses. Non-degenerate training (score 0.03→0.25,
grad_norm 0.35→0.94). Rollouter streaming via FullyAsyncAgentLoopWorkers on bruce.

Breakages fixed en route (all environment, no verl code changes):
1. **cupy missing → "Checkpoint engine nccl not registered"** (conditional import
   silently skips). Fix: `cupy-cuda13x` 14.1.1 (aarch64 manylinux wheel exists) on BOTH
   nodes.
2. **`naive` checkpoint backend is incompatible with fully-async** (Ray async actor
   requires coroutine methods; naive has none). nccl backend is effectively required.
3. **torchdata missing on bruce** (env drift after the mirror) → Ray's misleading
   "actor does not have any coroutine functions" on FullyAsyncRollouter — actually an
   ImportError during class deserialization on the worker node. Lesson: after
   incremental pip installs on the head, re-diff site-packages dist-infos vs bruce.

Remaining delta to the FULL drkernel form (→ RL-4): `recipe.drkernel.main` swaps in
KernelAgentLoop + KernelAsyncRewardManager + KernelGYM server + kernel dataset + 8B
model. The architecture underneath is what was just validated.
