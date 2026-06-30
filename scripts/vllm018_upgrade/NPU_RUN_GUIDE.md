# Running the vLLM 0.18 NPU smoke test (Task B2)

This is the Ascend-NPU half of the upgrade. The verl code changes are already done
and committed on branch `vllm-0.18-upgrade`; this guide installs vLLM 0.18 +
vllm-ascend 0.18 on an Ascend box and runs a minimal rollout smoke test.

Everything below runs **on the Ascend machine** (it needs `torch_npu` + CANN; it
cannot run on the GB10/CUDA host).

## 0. Prerequisites on the Ascend box
- A working Ascend stack: CANN toolkit + matching `torch` + `torch_npu`.
- The driver/firmware for your SoC (A2/A3/etc.).
- Network egress to download `Qwen/Qwen3-0.6B` (≈1.2 GB), or pre-place it and set
  `SMOKE_MODEL=/abs/path/to/Qwen3-0.6B`.

## 1. Get the patched verl onto the box
Pick whichever is easiest:

**a) git bundle** (offline-friendly; produced at
`/home/yubaifeng/e84381970/experiment/verl-vllm/verl-vllm018.bundle` on the GB10 host —
`scp` it to the Ascend box):
```bash
# on the Ascend box, after copying the bundle over:
git clone verl-vllm018.bundle verl && cd verl
git checkout vllm-0.18-upgrade
```

**b) plain copy**: `rsync`/`scp` the whole `drkernel-verl-port-drkernel/` tree over.

## 2. Install vLLM 0.18 + vllm-ascend 0.18
vllm-ascend pins the vllm version it targets; install the matching pair. From source
(mirrors what we did on CUDA) or from your Ascend wheel index:

```bash
# vllm 0.18 (build against the box's existing torch/torch_npu — do NOT let it pull a
# different torch). vllm-ascend's own install docs are authoritative for CANN/torch
# matching; the key flags:
cd /path/to/vllm           # the 0.18.0 source
python use_existing_torch.py
pip install -r requirements/build.txt          # if present for the ascend target
VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .   # ascend builds the
                                                                 # device bits via vllm-ascend
cd /path/to/vllm-ascend    # releases/v0.18.0
pip install --no-build-isolation -e .
```
> Note: on Ascend, vLLM itself is usually built with `VLLM_TARGET_DEVICE=empty` and the
> device kernels come from `vllm-ascend`. Follow vllm-ascend 0.18's official install
> doc for the exact CANN/torch_npu versions — that pairing is the main gotcha.

## 3. Install the patched verl (editable, no dep churn)
```bash
cd /path/to/verl           # the patched port, on branch vllm-0.18-upgrade
pip install -e . --no-deps --no-build-isolation
```

## 4. Sanity: imports clean under 0.18 on NPU
```bash
python scripts/vllm018_upgrade/check_imports.py
```
Expect: header shows `vllm 0.18...`; all **non-`[omni]`** lines `OK`. In particular
`verl.utils.vllm.npu_vllm_patch` must be `OK` (its 0.18 branch is now active because
`torch_npu` is present). `[omni]` FAILs are fine (out of scope).

## 5. Run the smoke test
```bash
ASCEND_RT_VISIBLE_DEVICES=0 python scripts/vllm018_upgrade/smoke_rollout_npu.py
```
Expect: two `PROMPT=... -> '...'` lines with non-empty text and a final `SMOKE PASS`.

## 6. Two things to watch (flagged in NPU_API_AUDIT.md)
These are the only parts of the 0.18 NPU patch that couldn't be verified statically:
1. **Rotary patch** — the 0.13-style `ApplyRotaryEmb.__init__` replacement calls
   `super(ApplyRotaryEmb, self).__init__()` with no args, while vLLM 0.18's real
   `__init__` passes `enforce_enable=...` to super. If init raises a TypeError around
   `ApplyRotaryEmb`/`CustomOp`, that's this — tell me and I'll adjust the wrapper.
2. **`FusedMoE.weight_loader` wrapper** — only matters for MoE models; Qwen3-0.6B is
   dense so the smoke won't hit it. If you later run a MoE model and weight loading
   errors with a signature/arg mismatch in `weight_loader`, that's this.

## If something fails
Capture the full traceback and the output of step 4, and send it over — every verl-side
fix will be a version-gated change (won't regress the CUDA path) and I'll re-verify.

## What "done" looks like
`smoke_rollout_npu.py` prints `SMOKE PASS` on the Ascend box under vLLM 0.18 +
vllm-ascend 0.18, with step 4 clean. That closes Milestone B.
