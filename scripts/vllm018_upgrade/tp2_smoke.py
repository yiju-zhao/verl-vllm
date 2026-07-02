"""vLLM TP=2 cross-node smoke test (gx10-090e + spark-bruce) on vllm 0.18.

Runs on the Ray head. tensor_parallel_size=2 with the Ray backend places one TP
worker per node; the TP all-reduce goes over NCCL on the ConnectX link.
"""
import os
import sys

os.environ.setdefault("VLLM_HOST_IP", "192.168.1.101")
os.environ.setdefault("RAY_ADDRESS", "192.168.1.101:6379")
os.environ.setdefault("NCCL_SOCKET_IFNAME", "enp1s0f1np1")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "enp1s0f1np1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
MODEL = os.environ.get("SMOKE_MODEL", "Qwen/Qwen3-0.6B")


def main() -> int:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        tensor_parallel_size=2,
        distributed_executor_backend="ray",
        gpu_memory_utilization=0.6,
        max_model_len=2048,
        enforce_eager=True,
        trust_remote_code=True,
        enable_flashinfer_autotune=False,  # flashinfer autotune hangs over cross-node TCP-NCCL
    )
    out = llm.generate(
        ["The capital of France is", "2 + 2 ="],
        SamplingParams(max_tokens=16, temperature=0.0),
    )
    ok = True
    for o in out:
        text = o.outputs[0].text
        print(f"PROMPT={o.prompt!r} -> {text!r}", flush=True)
        if not text.strip():
            ok = False
    print("TP2_SMOKE", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
