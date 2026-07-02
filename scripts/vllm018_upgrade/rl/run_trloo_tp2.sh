#!/usr/bin/env bash
# TP=2 variant of the Stage-1 trloo launcher (plan RL-3 / goal-doc M5):
# rollout tensor-parallel across the TWO Sparks (gx10-090e + spark-bruce).
#
# Prereqs (see TP2_CLUSTER_RUNBOOK.md): the 2-node Ray cluster must already be up —
#   head  (this box):  ray start --head --node-ip-address=192.168.1.101 --port=6380 ... VLLM_HOST_IP=192.168.1.101
#   worker (bruce)  :  ray start --address=192.168.1.101:6380 ...            VLLM_HOST_IP=192.168.1.106
# Per-node VLLM_HOST_IP comes from the ray-start env; do NOT export it here.
set -x
PY=/home/yubaifeng/e84381970/envs/drkernel310/bin/python3.10
cd /home/yubaifeng/e84381970/experiment/verl-vllm/drkernel-verl-port-drkernel
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_USAGE_STATS_ENABLED=0
# Connect to the existing 2-node cluster (port 6380: a foreign root-owned Ray squats 6379).
export RAY_ADDRESS=192.168.1.101:6380
# NCCL on the 200G ConnectX (TCP; RoCE not tuned yet). vllm copies NCCL_* to its workers.
export NCCL_SOCKET_IFNAME=enp1s0f1np1 GLOO_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_DISABLE=1
train=$HOME/data/gsm8k/train.parquet ; test=$HOME/data/gsm8k/test.parquet
STEPS="${STEPS:-1}"
$PY -m verl.trainer.main_ppo \
  +ray_kwargs.ray_init._node_ip_address=192.168.1.101 \
  algorithm.adv_estimator=trloo \
  data.train_files="['$train']" data.val_files="['$test']" \
  data.train_batch_size=8 data.max_prompt_length=512 data.max_response_length=768 \
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
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.enforce_eager=True \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_flashinfer_autotune=False \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.logger='["console"]' \
  trainer.project_name=vllm018_rl_smoke trainer.experiment_name=trloo_tp2_qwen3_0.6b \
  trainer.n_gpus_per_node=1 trainer.nnodes=2 \
  trainer.save_freq=-1 trainer.test_freq=-1 \
  trainer.total_training_steps="$STEPS" "$@"
