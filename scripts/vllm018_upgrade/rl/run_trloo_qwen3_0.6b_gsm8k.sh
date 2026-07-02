#!/usr/bin/env bash
set -x
PY=/home/yubaifeng/e84381970/envs/drkernel310/bin/python3.10
cd /home/yubaifeng/e84381970/experiment/verl-vllm/drkernel-verl-port-drkernel
export VLLM_HOST_IP=127.0.0.1 VLLM_WORKER_MULTIPROC_METHOD=spawn
# This host has a Tailscale 100.66.x IP that breaks Ray's node-IP autodetect (same
# root cause as VLLM_HOST_IP). Force Ray to loopback and disable the buggy usage-stats
# path (this Ray build crashes in usage_lib). See scripts/vllm018_upgrade/rl/RL_NOTES.md.
export RAY_USAGE_STATS_ENABLED=0
train=$HOME/data/gsm8k/train.parquet ; test=$HOME/data/gsm8k/test.parquet
STEPS="${STEPS:-1}"
$PY -m verl.trainer.main_ppo \
  +ray_kwargs.ray_init._node_ip_address=127.0.0.1 \
  algorithm.adv_estimator=trloo \
  data.train_files="['$train']" data.val_files="['$test']" \
  data.train_batch_size=8 data.max_prompt_length=512 data.max_response_length=256 \
  data.filter_overlong_prompts=True data.truncation=error \
  actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.logger='["console"]' \
  trainer.project_name=vllm018_rl_smoke trainer.experiment_name=trloo_qwen3_0.6b \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.save_freq=-1 trainer.test_freq=-1 \
  trainer.total_training_steps="$STEPS" "$@"
