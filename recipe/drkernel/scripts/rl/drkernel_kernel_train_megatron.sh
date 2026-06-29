#!/bin/bash
# DR.Kernel kernel-RL training launcher — Megatron variant (recipe-side, async + NPU).
#
# Routes through `recipe.drkernel.main` with the Megatron-flavored kernel
# trainer config (`drkernel_kernel_megatron_trainer`). Mirrors the structure
# of the existing fully_async NPU shells with Megatron parallelism (e.g.
# `for_reference_geo3k_qwen3vl_30b_megatron_4_4_npu_async_gmpo_multiturn.sh`).
#
# Required env (must be set):
#   KERNELGYM_SERVER_URL   e.g. http://npu-4:10907
#   MODEL_PATH             path to the (SFT-warmed) checkpoint
#   TRAIN_FILES            list-form, e.g. [/data/train.parquet]
#   VAL_FILES              list-form, e.g. [/data/val.parquet]
#
# Optional env (defaults shown):
#   # Async / resource split (Megatron uses separate trainer/rollouter node counts)
#   NNODES_TRAIN=2            number of nodes dedicated to the trainer
#   NNODES_ROLLOUT=2          number of nodes dedicated to the rollouter
#   NGPUS_PER_NODE=16         total NPUs per node (Ascend)
#   N_GPUS_ROLLOUT=12         NPUs per node for rollouter
#                             (trainer gets NGPUS_PER_NODE - N_GPUS_ROLLOUT)
#   STALENESS=0.5             async_training.staleness_threshold
#   SYNC_STEP=4               async_training.trigger_parameter_sync_step
#   REQUIRE_BATCHES=1         async_training.require_batches
#   PARTIAL_ROLLOUT=True      async_training.partial_rollout
#   TOTAL_ROLLOUT_STEPS       default 64*160=10240
#   TEST_FREQ=1000
#
#   # Megatron parallelism (actor side)
#   TP=4                      tensor model parallel
#   PP=2                      pipeline model parallel
#   CP=1                      context parallel
#   EP=4                      expert model parallel  (set 1 for dense models)
#   ETP=1                     expert tensor parallel (set 1 for dense models)
#   GEN_TP=4                  rollout (vLLM) tensor parallel
#
#   # Algo / sampling
#   ALGORITHM=trloo           algorithm.adv_estimator
#   ROLLOUT_N=16              actor_rollout_ref.rollout.n
#   PPO_MINI_BATCH=16         actor_rollout_ref.actor.ppo_mini_batch_size
#   MAX_TURN=3                multi-turn cap (user + assistant)
#   ENABLE_MRS=False          algorithm.batch_filter.enable
#   ENABLE_BYPASS_MODE=True   algorithm.rollout_correction.bypass_mode
#   LOSS_MODE=geo_mean        actor_rollout_ref.actor.policy_loss.loss_mode
#   CLIP_RATIO=0.4            actor.clip_ratio_low + clip_ratio_high
#   USE_KL_LOSS=False         actor_rollout_ref.actor.use_kl_loss
#
#   # Lengths + memory
#   MAX_PROMPT_LENGTH=1024
#   MAX_RESPONSE_LENGTH=65536        (== 1024 * 64; matches the reference shell)
#   ROLLOUT_GPU_MEM_UTIL=0.7
#   ENABLE_CHUNKED_PREFILL=False
#
#   # Kernel evaluation tuning
#   REWARD_MAX_CONCURRENT=32
#   NUM_PERF_TRIALS=100
#   DETECT_DECOY_KERNEL=True
#
# Example (single 16-NPU node, dense Qwen3-8B):
#   KERNELGYM_SERVER_URL=http://npu-4:10907 \
#   MODEL_PATH=/data/models/qwen3-8b-sft \
#   TRAIN_FILES='[/data/drkernel-rl-data.parquet]' \
#   VAL_FILES='[/data/drkernel-validation-data.parquet]' \
#   NNODES_TRAIN=1 NNODES_ROLLOUT=1 NGPUS_PER_NODE=16 N_GPUS_ROLLOUT=12 \
#   TP=2 PP=2 EP=1 ETP=1 GEN_TP=2 \
#   bash recipe/drkernel/scripts/rl/drkernel_kernel_train_megatron.sh

set -x
# Note: `set -u` (nounset) is intentionally NOT enabled here. The Ascend
# toolkit's set_env.sh references $ZSH_VERSION (and similar shell-version
# vars) without defaulting them, which trips nounset. Mirrors the FSDP
# variant and the existing fully_async NPU shells.
set -eo pipefail

ENGINE=${1:-vllm}

# ---------- Ascend / vLLM env (matches existing NPU async shells) ----------
export CUDA_DEVICE_MAX_CONNECTIONS=1   # for megatron communication/computation overlapping
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export HYDRA_FULL_ERROR=1
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_USE_V1=1

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

# ---------- Required overrides (uncomment the asserts to enforce) ----------
# : "${KERNELGYM_SERVER_URL:?KERNELGYM_SERVER_URL must be set}"
# : "${MODEL_PATH:?MODEL_PATH must be set}"
# : "${TRAIN_FILES:?TRAIN_FILES must be set, e.g. [/data/train.parquet]}"
# : "${VAL_FILES:?VAL_FILES must be set, e.g. [/data/val.parquet]}"

# Convenience defaults for local testing — override via env or CLI.
TRAIN_FILES="${TRAIN_FILES:-/data/nfs/ahmad/dataset/thinking/training_data_thinking.parquet}"
VAL_FILES="${VAL_FILES:-/data/nfs/ahmad/dataset/thinking/validation_data_thinking.parquet}"
MODEL_PATH="${MODEL_PATH:-/data/nfs/model/Qwen3-4B}"
KERNELGYM_SERVER_URL="${KERNELGYM_SERVER_URL:-http://npu-4:10907}"

# ---------- Async / resource split ----------
NNODES_TRAIN="${NNODES_TRAIN:-1}"
NNODES_ROLLOUT="${NNODES_ROLLOUT:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-16}"
N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-12}"
N_GPUS_TRAINING=$((NGPUS_PER_NODE - N_GPUS_ROLLOUT))

STALENESS="${STALENESS:-0.5}"
SYNC_STEP="${SYNC_STEP:-4}"
REQUIRE_BATCHES="${REQUIRE_BATCHES:-1}"
PARTIAL_ROLLOUT="${PARTIAL_ROLLOUT:-True}"
TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-$((64 * 160))}"
TEST_FREQ="${TEST_FREQ:-1000}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-200}"
SAVE_FREQ="${SAVE_FREQ:--1}"

# ---------- Megatron parallelism ----------
TP="${TP:-1}"            # tensor parallel
PP="${PP:-2}"            # pipeline parallel
CP="${CP:-1}"            # context parallel
EP="${EP:-1}"            # expert parallel  (1 for dense models)
ETP="${ETP:-1}"          # expert tensor parallel
GEN_TP="${GEN_TP:-1}"    # rollout (vLLM) tensor parallel

# ---------- Algo / sampling ----------
ALGORITHM="${ALGORITHM:-trloo}"
ROLLOUT_N="${ROLLOUT_N:-16}"
PPO_MINI_BATCH="${PPO_MINI_BATCH:-16}"
MAX_TURN="${MAX_TURN:-3}"
ENABLE_MRS="${ENABLE_MRS:-False}"
ENABLE_BYPASS_MODE="${ENABLE_BYPASS_MODE:-True}"
LOSS_MODE="${LOSS_MODE:-geo_mean}"
CLIP_RATIO="${CLIP_RATIO:-0.4}"
USE_KL_LOSS="${USE_KL_LOSS:-False}"

# ---------- Lengths / memory ----------
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-$((1024 * 4))}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
LR_DECAY_STEPS="${LR_DECAY_STEPS:-51200}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.7}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-False}"
TRUNCATION="${TRUNCATION:-right}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"

# Megatron uses static (token-budget) micro-batches. Token caps are sized so
# (max_prompt + max_response) fits when divided by TP.
ACTOR_PPO_MAX_TOKEN_LEN="${ACTOR_PPO_MAX_TOKEN_LEN:-$(( (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) / TP ))}"
INFER_PPO_MAX_TOKEN_LEN="${INFER_PPO_MAX_TOKEN_LEN:-$(( (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) / TP ))}"

# ---------- Kernel evaluation tuning ----------
REWARD_MAX_CONCURRENT="${REWARD_MAX_CONCURRENT:-32}"
NUM_PERF_TRIALS="${NUM_PERF_TRIALS:-100}"
DETECT_DECOY_KERNEL="${DETECT_DECOY_KERNEL:-True}"

PROJECT_NAME="${PROJECT_NAME:-drkernel_async_megatron}"
EXP_NAME="${EXP_NAME:-drkernel_kernel_megatron}"

# ---------- Resolve repo root + PYTHONPATH ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

mkdir -p "logs/${PROJECT_NAME}"

# ---------- Launch ----------
python -m recipe.drkernel.main \
    --config-name=drkernel_kernel_megatron_trainer \
    \
    `# data` \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.return_raw_chat=True \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation="${TRUNCATION}" \
    \
    `# model + actor (Megatron — no fsdp_config)` \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
    actor_rollout_ref.actor.optim.lr_decay_steps="${LR_DECAY_STEPS}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN}" \
    actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.policy_loss.loss_mode="${LOSS_MODE}" \
    actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO}" \
    actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO}" \
    actor_rollout_ref.hybrid_engine=False \
    \
    `# Megatron parallelism (actor)` \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${TP}" \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${PP}" \
    actor_rollout_ref.actor.megatron.context_parallel_size="${CP}" \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size="${EP}" \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size="${ETP}" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.context_parallel_size="${CP}" \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    \
    `# Megatron transformer-config + optimizer overrides` \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8=False \
    \
    `# Reference model (Megatron, offloaded)` \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${INFER_PPO_MAX_TOKEN_LEN}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    \
    `# rollout (vLLM, async, multi-turn)` \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.enable_chunked_prefill="${ENABLE_CHUNKED_PREFILL}" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${INFER_PPO_MAX_TOKEN_LEN}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_TURN}" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_TURN}" \
    \
    `# algorithm` \
    algorithm.adv_estimator="${ALGORITHM}" \
    algorithm.use_kl_in_reward=False \
    algorithm.batch_filter.enable="${ENABLE_MRS}" \
    algorithm.rollout_correction.bypass_mode="${ENABLE_BYPASS_MODE}" \
    \
    `# reward (KernelGym via verl's custom_reward_function path) ` \
    `# - reward_manager stays "naive" (stock dispatcher set in the yaml).` \
    `# - The whole reward_model block is forwarded into our compute_score` \
    `#   via custom_reward_function.reward_kwargs.reward_config (yaml side).` \
    `# - server_url / max_concurrent / num_perf_trials / detect_decoy_kernel` \
    `#   are NOT in verl's stock RewardModelConfig schema, so the "+" prefix` \
    `#   tells Hydra "append rather than strict-validate".` \
    reward_model.enable=False \
    +reward_model.server_url="${KERNELGYM_SERVER_URL}" \
    +reward_model.max_concurrent="${REWARD_MAX_CONCURRENT}" \
    +reward_model.num_perf_trials="${NUM_PERF_TRIALS}" \
    +reward_model.detect_decoy_kernel="${DETECT_DECOY_KERNEL}" \
    \
    `# resource pools (fully_async splits trainer / rollouter; Megatron uses separate node counts)` \
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.n_gpus_per_node="${N_GPUS_TRAINING}" \
    rollout.nnodes="${NNODES_ROLLOUT}" \
    rollout.n_gpus_per_node="${N_GPUS_ROLLOUT}" \
    rollout.total_rollout_steps="${TOTAL_ROLLOUT_STEPS}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.device=npu \
    trainer.critic_warmup=0 \
    trainer.resume_mode=auto \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.logger='["console","tensorboard"]' \
    trainer.val_before_train=False \
    \
    `# async training tuning` \
    async_training.staleness_threshold="${STALENESS}" \
    async_training.trigger_parameter_sync_step="${SYNC_STEP}" \
    async_training.require_batches="${REQUIRE_BATCHES}" \
    async_training.partial_rollout="${PARTIAL_ROLLOUT}" \
    \
    "$@" 2>&1 | tee "logs/${PROJECT_NAME}/run_${EXP_NAME}.log"
