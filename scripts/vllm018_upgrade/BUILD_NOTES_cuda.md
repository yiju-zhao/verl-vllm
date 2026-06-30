# CUDA build notes — vLLM 0.18 on GB10 (Task A2)

## Outcome
- Installed: `vllm 0.18.1.dev0+gbcf2be961.d20260630.cu130` (editable, from `../vllm`).
- `import vllm._C` loads — the pre-existing `libcudart.so.12` breakage (0.11.2 was a
  CUDA-12 wheel in a cu130 env) is resolved; built against the env's torch 2.9.1+cu130.
- 366 CUDA/C++ objects compiled (ccache populated, 0.7 GB).

## Environment
- GB10, sm_121 (cap 12.1), aarch64. CUDA toolkit `nvcc 13.0` at `/usr/local/cuda`.
- Env `drkernel310`: torch `2.9.1+cu130`, Python 3.10.

## Key decisions / steps
1. **torch pin mismatch:** vllm 0.18 build-requires `torch==2.10.0`; env has 2.9.1.
   Ran `python use_existing_torch.py` in `../vllm` to strip the torch pins, then built
   with `--no-build-isolation` so the build uses the installed torch 2.9.1. The
   2.9→2.10 C++ API gap caused **no** compile errors.
2. **Build deps:** with `--no-build-isolation`, pip does not provide build-system
   requires; installed them into the env: `setuptools-scm`, `packaging`, `setuptools`,
   `wheel`, `jinja2`, `cmake`, `ninja`.
3. **CUDA:** `CUDA_HOME=/usr/local/cuda` (was unset); `TORCH_CUDA_ARCH_LIST=12.0+PTX`
   (torch 2.9.1 validates only up to sm_120; PTX JITs forward to the GB10's sm_121).
4. **MAX_JOBS=8, NVCC_THREADS=2.**
5. **ccache (60 GB at `/home/yubaifeng/e84381970/.ccache`):** vllm wires it as the
   CUDA/C++ compiler launcher. Made the build resumable — an earlier attempt was
   killed at ~20 min (background-task runtime cap, not OOM/error), so the build runs
   under `build_until_done.sh` which retries until `import vllm` reports 0.18; ccache
   makes each retry resume.

## Reproduce
```bash
cd ../vllm && python use_existing_torch.py    # once; strips torch==2.10.0 pins
bash scripts/vllm018_upgrade/build_until_done.sh   # self-restarting; calls build_vllm_cuda.sh
```

## Pre-existing dep skew (NOT from this build; watch during smoke test A4)
- `numpy 2.2.6` (verl wants `<2.0.0`) — most likely to bite at verl runtime.
- `tensordict 0.12.3` (verl wants `<=0.10.0`), `protobuf 6.33.6` (wandb/grpc want `<6`).
These were already present in the env; address only if the smoke test hits them.

## Rollback
0.11.2 was a wheel (origin: plain `site-packages`, no special index recorded). To
restore: `pip install vllm==0.11.2` (note: that wheel is CUDA-12 and broken in this
cu130 env — the 0.18 source build is the working install).
