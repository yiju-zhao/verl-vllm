"""Minimal CUDA rollout smoke test for the vllm 0.18 upgrade.

Drives generation through verl's vLLM integration with a tiny dense model and
asserts non-empty completions. Keep dependencies minimal (no Ray/trainer).

Importing `verl.third_party.vllm` first exercises verl's version gate against the
installed vllm before constructing the engine.
"""

import os
import sys

os.environ.setdefault("VLLM_USE_V1", "1")
# vllm V1 forks its EngineCore by default; since importing verl/vllm in the parent
# initializes CUDA, the fork can't re-init it. Force spawn (the __main__ guard below
# makes the script safe to re-import in the spawned child).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# This host has a Tailscale/overlay interface (100.66.x.x); vllm's IP auto-detect
# picks it for the single-node TCP rendezvous and the spawned EngineCore can't
# connect back. Pin the rendezvous to loopback.
os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
MODEL = os.environ.get("SMOKE_MODEL", "Qwen/Qwen3-0.6B")


def main() -> int:
    # Import through verl so the third_party gate + patches are exercised.
    import verl.third_party.vllm  # noqa: F401  (runs verl's version gate)
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
