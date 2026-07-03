#!/usr/bin/env bash
# NPU mirror of the validated CUDA Stage-1 run: colocated verl.trainer.main_ppo,
# trloo + gsm8k + Qwen3-0.6B, ONE Ascend NPU, with the consistency diagnostics
# (pearson / probs_diff / token-IS "RS ratio") enabled — the same instruments used
# to root-cause the CUDA-side mismatch. Run ON the Ascend box.
#
# Prereqs (see ../../NPU_RUN_GUIDE.md): vllm 0.18 + vllm-ascend 0.18 + this verl
# installed; gsm8k parquet prepared (prep_gsm8k.sh); Qwen3-0.6B available.
#
# NPU deltas vs the CUDA launcher (run_trloo_qwen3_0.6b_gsm8k.sh):
#  - no sdpa override / remove_padding stays True: verl has native NPU attention +
#    pad utils (npu_flash_attn_utils). If attention breaks, flip ATTN_FALLBACK=1.
#  - no flashinfer knob (CUDA-only concern), no Tailscale VLLM_HOST_IP hack.
#  - clean Ray state per run: stale /tmp/ray/ray_current_cluster from earlier
#    clusters makes ray.init join dead clusters (same failure as on CUDA).
set -x
PY="${PY:-python3}"
cd "$(dirname "$0")/../../../.."   # port root
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn RAY_USAGE_STATS_ENABLED=0
ray stop --force >/dev/null 2>&1; rm -rf /tmp/ray 2>/dev/null
train=$HOME/data/gsm8k/train.parquet ; test=$HOME/data/gsm8k/test.parquet
STEPS="${STEPS:-5}"
ATTN_ARGS=()
if [ "${ATTN_FALLBACK:-0}" = "1" ]; then
  ATTN_ARGS+=("+actor_rollout_ref.model.override_config.attn_implementation=eager"
              "actor_rollout_ref.model.use_remove_padding=False")
fi
$PY -m verl.trainer.main_ppo \
  algorithm.adv_estimator=trloo \
  data.train_files="['$train']" data.val_files="['$test']" \
  data.train_batch_size=8 data.max_prompt_length=512 data.max_response_length=768 \
  data.filter_overlong_prompts=True data.truncation=error \
  actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  algorithm.rollout_correction.rollout_is=token \
  algorithm.rollout_correction.rollout_is_threshold=2.0 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  trainer.logger='["console"]' \
  trainer.project_name=vllm018_rl_smoke_npu trainer.experiment_name=trloo_npu \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.save_freq=-1 trainer.test_freq=-1 \
  trainer.total_training_steps="$STEPS" \
  "${ATTN_ARGS[@]}" "$@"
