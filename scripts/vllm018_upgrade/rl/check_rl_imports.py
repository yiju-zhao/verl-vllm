import importlib, sys, vllm
print("vllm", vllm.__version__)
mods = [
    "verl.trainer.main_ppo",
    "verl.trainer.ppo.ray_trainer",
    "verl.trainer.ppo.core_algos",
    "verl.workers.engine_workers",
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
