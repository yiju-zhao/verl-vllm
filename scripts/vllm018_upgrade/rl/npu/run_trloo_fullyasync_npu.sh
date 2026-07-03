#!/usr/bin/env bash
# NPU mirror of the validated fully-async separation run (the drkernel production
# deployment shape): TRAINER pool = 1 NPU, ROLLOUTER (vllm-ascend server) pool = 1 NPU,
# weight sync via the checkpoint engine (backend name "nccl" — on NPU the registry
# resolves it to the HCCL implementation; no cupy needed, deps are ray/zmq only).
# Run ON the Ascend box; needs >= 2 NPUs visible (single node is fine).
#
# Mirrors run_trloo_fullyasync_1p1.sh minus CUDA-specific knobs.
set -x
PY="${PY:-python3}"
cd "$(dirname "$0")/../../../.."   # port root
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn RAY_USAGE_STATS_ENABLED=0
ray stop --force >/dev/null 2>&1; rm -rf /tmp/ray 2>/dev/null
train=$HOME/data/gsm8k/train.parquet ; test=$HOME/data/gsm8k/test.parquet
TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-24}"
ATTN_ARGS=()
if [ "${ATTN_FALLBACK:-0}" = "1" ]; then
  ATTN_ARGS+=("+actor_rollout_ref.model.override_config.attn_implementation=eager"
              "actor_rollout_ref.model.use_remove_padding=False")
fi
$PY -m verl.experimental.fully_async_policy.fully_async_main \
  algorithm.adv_estimator=trloo \
  data.train_files="['$train']" data.val_files="['$test']" \
  data.prompt_key=prompt data.truncation=left data.return_raw_chat=True \
  data.max_prompt_length=512 data.max_response_length=768 \
  data.train_batch_size=0 data.gen_batch_size=1 \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
  critic.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2560 \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=3840 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=3840 \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  algorithm.use_kl_in_reward=False \
  trainer.logger='["console"]' \
  trainer.project_name=vllm018_rl_smoke_npu trainer.experiment_name=trloo_fullyasync_npu \
  trainer.val_before_train=False \
  trainer.save_freq=-1 trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.nnodes=1 trainer.n_gpus_per_node=1 \
  rollout.nnodes=1 rollout.n_gpus_per_node=1 \
  rollout.total_rollout_steps="$TOTAL_ROLLOUT_STEPS" \
  async_training.staleness_threshold=0.1 \
  async_training.trigger_parameter_sync_step=1 \
  async_training.require_batches=1 \
  async_training.partial_rollout=False \
  "${ATTN_ARGS[@]}" "$@"
