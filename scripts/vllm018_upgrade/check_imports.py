"""Import-inventory regression oracle for the vLLM 0.18 upgrade.

Run under the *current* vllm to capture a baseline, then under vllm 0.18 to
detect regressions. `vllm_omni`-tagged entries are reported but never cause a
non-zero exit (out of scope per the plan).
"""

import importlib
import sys

import vllm

# (module path, is_omni)
VERL_MODULES = [
    ("verl.third_party.vllm", False),
    ("verl.utils.vllm.utils", False),
    ("verl.utils.vllm.patch", False),
    ("verl.utils.vllm.vllm_fp8_utils", False),
    ("verl.utils.vllm.npu_vllm_patch", False),
    ("verl.workers.rollout.vllm_rollout", False),
    ("verl.workers.rollout.vllm_rollout.vllm_rollout", False),
    ("verl.workers.rollout.vllm_rollout.vllm_async_server", False),
    ("verl.workers.rollout.vllm_rollout.utils", False),
    ("verl.utils.profiler.config", False),
    ("verl.workers.rollout.vllm_rollout.vllm_omni_async_server", True),
    ("verl.utils.vllm_omni.utils", True),
]

# Raw vllm symbols verl imports (sanity that the upstream API still exists).
VLLM_SYMBOLS = [
    ("vllm", "LLM"),
    ("vllm.outputs", "RequestOutput"),
    ("vllm.v1.engine.async_llm", "AsyncLLM"),
    ("vllm.v1.engine", "FinishReason"),
    ("vllm.distributed", "parallel_state"),
    ("vllm.distributed.utils", "StatelessProcessGroup"),
    ("vllm.lora.request", "LoRARequest"),
    ("vllm.platforms", "current_platform"),
    ("vllm.usage.usage_lib", "UsageContext"),
    ("vllm.model_executor.layers.quantization.utils.fp8_utils", None),
    ("vllm.model_executor.layers.quantization.utils.marlin_utils_fp4", None),
    ("vllm.model_executor.parameter", None),
]


def main() -> int:
    print(f"# vllm {vllm.__version__} @ {vllm.__file__}")
    hard_fail = False
    for mod, is_omni in VERL_MODULES:
        tag = " [omni]" if is_omni else ""
        try:
            importlib.import_module(mod)
            print(f"OK   {mod}{tag}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {mod}{tag}: {type(e).__name__}: {e}")
            if not is_omni:
                hard_fail = True
    for mod, attr in VLLM_SYMBOLS:
        name = f"{mod}.{attr}" if attr else mod
        try:
            m = importlib.import_module(mod)
            if attr and not hasattr(m, attr):
                raise AttributeError(f"missing attribute {attr}")
            print(f"OK   {name}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            hard_fail = True
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
