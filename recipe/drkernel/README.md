# DR.Kernel — Triton kernel-generation RL on NPU

This recipe ports **DR.Kernel** ([arXiv:2602.05885](https://arxiv.org/abs/2602.05885)) onto
this verl fork and runs it on **NPU**. 

- **Entry point:** `recipe/drkernel/main.py` (`python -m recipe.drkernel.main`)
- **Reward server:** `recipe/NPU-kernelGym/` — a GPU/NPU-distributed
  kernel evaluation service. See its own [README](../NPU-kernelGym/README.md).

---

## 1. Installation

Follow the instruction in environment setup (base container and framework stack, verl) in this guide: **[`INSTALL.md`](INSTALL.md)**.

---

## 2. Running a training experiment

A training run has **two** pieces that come up in order:

1. the **KernelGYM reward server** (evaluates generated kernels, returns
   correctness + speedup), and
2. the **verl async-PPO trainer** (`recipe.drkernel.main`).

The tracked, canonical launcher
[`scripts/rl/drkernel_kernel_train_native_8b.sh`](scripts/rl/drkernel_kernel_train_native_8b.sh)
wires both together — an 8B Qwen3 run with MRS on the
2k-difficulty-filtered training set. Use it as the worked example below.


### 2.1 Prerequisites before launching

- **Model** at `MODEL_PATH` (default `/home/model/Qwen3-8B`).
- **Datasets** — train/val parquet files at `TRAIN_FILES` / `VAL_FILES`.
  Download them from the Hugging Face dataset
  [`AhNr/dr-kernel-RL`](https://huggingface.co/datasets/AhNr/dr-kernel-RL).
- **KernelGYM server reachable** at `KERNELGYM_SERVER_URL`
  (default `http://127.0.0.1:8002`). Point it at the host/port where
  `recipe/NPU-kernelGym` is serving — e.g. a remote node's IP for a multi-node setup.
- **NPUs** — defaults to a two nodes settings **10 rollout + 16 training + 6 KernelGYM**
  NPUs (`N_GPUS_ROLLOUT=10`, `N_GPUS_TRAINING=16`).

### 2.2 Two-node launch

Bring up the Ray cluster first (head node should be started with 10 cards (6 cards reserved for KernelGYM), the other nodes joining as
workers), make sure all nodes see the same synced code and the same
`KERNELGYM_SERVER_URL`, then launch from the head node with `NNODES` set:

```bash
bash recipe/drkernel/scripts/rl/drkernel_kernel_train_native_8b.sh
```

### 2.3 Key config knobs

| Group | Knob (env var) | Default | Meaning |
|---|---|---|---|
| Algorithm | `ALGORITHM` | `trloo` | advantage estimator |
| MRS | `ROLLOUT_RS` / `ROLLOUT_RS_THRESHOLD` | `seq_mean_k1` / `0.999001_1.001001` | paper geometric-mean rejection |
| MRS | `ROLLOUT_TOKEN_VETO_THRESHOLD` | `1e-4` | per-token catastrophic veto |
| MRS | `ROLLOUT_IS` / `ROLLOUT_IS_THRESHOLD` | `token` / `2.0` | token-level truncated IS |
| Rollout | `ROLLOUT_N` | `16` | samples per prompt |
| Rollout | `MAX_TURN` | `3` | multi-turn turns |
| Lengths | `MAX_PROMPT_LENGTH` / `MAX_RESPONSE_LENGTH` | `4096` / `24576` | token budgets |
| Reward | `REWARD_FUNC_NAME` | `calculate_reward_weighted` | correctness + speedup blend |
| Reward | `INIT_CORRECT_WEIGHT` / `INIT_PERFORMANCE_WEIGHT` | `1.0` / `1.0` | reward weights |
| Async | `STALENESS` / `SYNC_STEP` / `REQUIRE_BATCHES` | `0.1` / `1` / `16` | async-PPO pacing |
| Compute | `N_GPUS_ROLLOUT` / `N_GPUS_TRAINING` | `10` / `16` | NPU split |


### 2.4 Outputs

- **Training log:** `logs/${PROJECT_NAME}/run_${EXP_NAME}.log` (teed live).
- **Checkpoints:** `trainer.default_local_dir`
  (default `${REPO_ROOT}/checkpoints/${PROJECT_NAME}/${EXP_NAME}`; override with
  `DEFAULT_LOCAL_DIR`).
- **Rollout dumps:** `./rollout_dump/...` and `./rollout_validation_dump/...`.
- **Metrics:** console + TensorBoard (`trainer.logger=["console","tensorboard"]`).

---

## 3. Running validation only

To score a trained checkpoint without launching a full training run, use the
tracked validation launcher
[`scripts/validation/drkernel_validate_native_8b.sh`](scripts/validation/drkernel_validate_native_8b.sh).
It starts KernelGYM and runs `recipe.drkernel.main_validate` once
(rollout-only — no trainer pool, no parameter sync), reusing the same
multi-turn / kernel_async-reward path as training so the `val-core` / `val-aux`
metrics line up with the curves training logs at `test_freq`.

```bash
MODEL_PATH=/path/to/merged_hf_checkpoint \
VAL_FILES=/path/to/validation.parquet \
    bash recipe/drkernel/scripts/validation/drkernel_validate_native_8b.sh
```

- **`MODEL_PATH` must be a merged HF checkpoint** (e.g. produced from a training
  checkpoint with `scripts/merge_to_hf.sh`), not a raw FSDP checkpoint dir.
- **Pass@N:** set `VAL_N>1` (with `VAL_DO_SAMPLE=True`) to generate N stochastic
  rollouts per prompt and emit naive `Pass@N` metrics plus a per-prompt JSON
  dump. `VAL_N=1` (default) does a single rollout per prompt.
- **Resource pools** are rollout-only: the trainer pool is given
  `n_gpus_per_node=0`, so all NPUs go to rollout (`N_GPUS_ROLLOUT`, default `10`).
- **Outputs:** metrics JSON + TensorBoard under `outputs/${PROJECT_NAME}/${EXP_NAME}`,
  per-rollout dumps under `rollout_validation_dump/...`, and a teed log at
  `logs/${PROJECT_NAME}/val_${EXP_NAME}.log`.

