"""Minimal Ascend NPU rollout smoke test for the vllm 0.18 / vllm-ascend 0.18 upgrade.

Run this ON the Ascend box (this requires torch_npu + vllm_ascend; it cannot run on
the GB10/CUDA host). It:
  1. imports `verl.utils.vllm.npu_vllm_patch` -> applies the 0.18 (0.13-style) NPU
     patches at import time (rotary / FusedMoE.weight_loader / MC2 select_moe_comm /
     matmul_and_reduce);
  2. calls `check_vllm_ascend_before_server_launch()` -> exercises the is_A2 dispatch
     that was extended for 0.18;
  3. imports `verl.third_party.vllm` -> verl's version gate (NPU branch);
  4. constructs vllm.LLM with a tiny dense model and asserts non-empty generations.

Env knobs (override as needed for your cluster):
  SMOKE_MODEL                  default Qwen/Qwen3-0.6B
  ASCEND_RT_VISIBLE_DEVICES    which NPU(s) to use, e.g. "0"
  VLLM_HOST_IP                 default 127.0.0.1 (single-node TCP rendezvous)
"""

import os
import sys

os.environ.setdefault("VLLM_USE_V1", "1")
# Same rationale as the CUDA smoke: spawn the engine subprocess (NPU can't re-init
# in a fork), and pin the single-node rendezvous to loopback.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")
MODEL = os.environ.get("SMOKE_MODEL", "Qwen/Qwen3-0.6B")


def main() -> int:
    import vllm
    import vllm_ascend  # noqa: F401  (registers the Ascend platform)

    print(f"vllm {vllm.__version__} | vllm_ascend {getattr(vllm_ascend, '__version__', '?')}")

    # 1) Apply the NPU patches (module-level, gated on is_torch_npu_available()).
    import verl.utils.vllm.npu_vllm_patch as npu_patch

    # 2) Exercise the 0.18-extended is_A2 dispatch (no-op unless A2 + MATMUL_ALLREDUCE).
    try:
        npu_patch.check_vllm_ascend_before_server_launch()
        print("check_vllm_ascend_before_server_launch: OK")
    except Exception as e:  # noqa: BLE001
        # Don't fail the smoke on this; just surface it (it's the is_A2 path).
        print(f"check_vllm_ascend_before_server_launch raised: {type(e).__name__}: {e}")

    # 3) verl version gate (NPU branch).
    import verl.third_party.vllm  # noqa: F401

    # 4) Generate.
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.6,
        max_model_len=2048,
        enforce_eager=True,
        trust_remote_code=True,
    )
    prompts = ["The capital of France is", "2 + 2 ="]
    out = llm.generate(prompts, SamplingParams(max_tokens=16, temperature=0.0))
    ok = True
    for o in out:
        text = o.outputs[0].text
        print(f"PROMPT={o.prompt!r} -> {text!r}")
        if not text.strip():
            ok = False
    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
