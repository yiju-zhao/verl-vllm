#!/usr/bin/env bash
# RL-2: minimal fully-async separation run — the drkernel production deployment shape
# (recipe.drkernel.main is built on verl.experimental.fully_async_policy) scaled to
# our 2-Spark cluster: TRAINER pool = 1 GPU, ROLLOUTER (vllm server) pool = 1 GPU,
# weight sync via CheckpointEngineManager (bucketed IPC) — cross-node.
#
# Prereq: the 2-node Ray cluster on :6380 (see TP2_CLUSTER_RUNBOOK.md), both nodes
# started with per-node VLLM_HOST_IP + NCCL ifname + RAY_memory_monitor_refresh_ms=0.
set -x
PY=/home/yubaifeng/e84381970/envs/drkernel310/bin/python3.10
cd /home/yubaifeng/e84381970/experiment/verl-vllm/drkernel-verl-port-drkernel
export VLLM_WORKER_MULTIPROC_METHOD=spawn RAY_USAGE_STATS_ENABLED=0
export RAY_ADDRESS=192.168.1.101:6380
export NCCL_SOCKET_IFNAME=enp1s0f1np1 GLOO_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_DISABLE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
train=$HOME/data/gsm8k/train.parquet ; test=$HOME/data/gsm8k/test.parquet
TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-24}"   # prompt budget for the rollouter
$PY -m verl.experimental.fully_async_policy.fully_async_main \
  +ray_kwargs.ray_init._node_ip_address=192.168.1.101 \
  algorithm.adv_estimator=trloo \
  data.train_files="['$train']" data.val_files="['$test']" \
  data.prompt_key=prompt data.truncation=left data.return_raw_chat=True \
  data.max_prompt_length=512 data.max_response_length=768 \
  data.train_batch_size=0 data.gen_batch_size=1 \
  actor_rollout_ref.hybrid_engine=False \
  actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.use_remove_padding=False \
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
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_flashinfer_autotune=False \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  algorithm.use_kl_in_reward=False \
  trainer.logger='["console"]' \
  trainer.project_name=vllm018_rl_smoke trainer.experiment_name=trloo_fullyasync_1p1 \
  trainer.val_before_train=False \
  trainer.save_freq=-1 trainer.test_freq=-1 \
  trainer.total_epochs=1 \
  trainer.nnodes=1 trainer.n_gpus_per_node=1 \
  rollout.nnodes=1 rollout.n_gpus_per_node=1 \
  rollout.total_rollout_steps="$TOTAL_ROLLOUT_STEPS" \
  async_training.staleness_threshold=0.1 \
  async_training.trigger_parameter_sync_step=1 \
  async_training.require_batches=1 \
  async_training.partial_rollout=False "$@"
