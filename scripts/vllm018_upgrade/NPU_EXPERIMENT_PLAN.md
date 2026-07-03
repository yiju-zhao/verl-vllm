# NPU 实验方法(B2 + Austin/token-271 差分验证)

在 Ascend 机器上按序执行。目标有两个:① 验证我们的 vllm 0.18 + verl 栈在 NPU 上端到端可用(B2);
② 用受控实验回答 Austin 的 token-271 问题(TP2 批变性假设)——我们的 CUDA 对照(TP1≡TP2,
pearson 0.9993)已排除 vllm core/verl 集成,NPU 上的矩阵实验将把根因钉死在 vllm-ascend kernel 层。

前置修复(已在本仓库,无需操作):
- **Gap A**:`logprobs_from_logits_torch_npu` 现在默认分块 fp32 upcast(`DKV_FP32_LOGPROB=1`,
  设 0 可 A/B)。
- **OOV mask TP 洞**:`monkey_patch_compute_logits` 现在按全局词表坐标 mask,TP 分片下不再失效。

---

## Phase 0 — 环境(一次性,~1-2 小时)

1. 拷贝 bundle 并检出(bundle 在 GB10 上:`~/e84381970/experiment/verl-vllm/verl-vllm018.bundle`):
   ```bash
   git clone verl-vllm018.bundle verl && cd verl && git checkout main
   ```
2. 按 `scripts/vllm018_upgrade/NPU_RUN_GUIDE.md` §2 安装 vllm 0.18 + vllm-ascend 0.18
   (CANN/torch_npu 配对是最大的坑;Austin 验证过的组合:torch/torch_npu 2.9 + CANN9 +
   triton-ascend 3.2.0 + `setuptools<81`,详见其 deploy-vllm018.md)。
3. `pip install -e . --no-deps` 装本 verl;跑 `python scripts/vllm018_upgrade/rl/check_rl_imports.py`
   → 期望 5 OK + trloo registered。
4. 数据:`bash scripts/vllm018_upgrade/rl/prep_gsm8k.sh`(改脚本里 `$PY` 为你的解释器)。
5. 检查 checkpoint 引擎:`python -c "import verl.checkpoint_engine; from
   verl.checkpoint_engine.base import CheckpointEngineRegistry as R; print(R._registry.keys())"`
   → 应含 `nccl`(NPU 上解析为 HCCL 实现)。

## Phase 1 — 推理 smoke(~10 分钟)

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python scripts/vllm018_upgrade/smoke_rollout_npu.py
```
期望 `SMOKE PASS`。这一步同时验证 npu_vllm_patch 0.18 分支和审计中两个只能实机确认的点
(rotary `super().__init__`、FusedMoE.weight_loader —— dense 0.6B 只触发前者)。

## Phase 2 — TP1 基线 + Gap A 度量(~30 分钟)

```bash
# 2a. TP1 + 诊断(fp32 logprob 默认开)
STEPS=5 bash scripts/vllm018_upgrade/rl/npu/run_trloo_npu.sh 2>&1 | tee /tmp/npu_tp1.log
# 2b. Gap A 的 A/B:关掉 fp32,量化你们 torch_npu 版本上的系统性偏差
STEPS=2 DKV_FP32_LOGPROB=0 bash scripts/vllm018_upgrade/rl/npu/run_trloo_npu.sh 2>&1 | tee /tmp/npu_tp1_bf16.log
```
读数(step 行里):`training/rollout_actor_probs_pearson_corr`、`rollout_corr/rollout_is_*`。
**判读**:
- 2a 期望 pearson ≥0.99(CUDA 基线 0.9993;Austin 的 NPU TP1 也是 0.9997)。若明显更低,先跑
  token 级定位(见 Phase 3 的 dump 方法)再前进。
- 2b vs 2a 的 `rollout_is_mean` 偏移 = 你们 torch_npu 版本的 Gap A 大小(2.10 上约 e^-0.06≈0.94,
  2.9 上应接近 1)。
- 若 attention/padding 报错:`ATTN_FALLBACK=1` 重试(CUDA 验证过的保守配置)。

## Phase 3 — TP2 矩阵:Austin 问题的判定实验(核心,~1 小时)

需要 2 张 NPU。三个受控 run,每个都带 token 级 dump:

```bash
DUMP="+ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGPROB_DIAG_DUMP=/tmp/lp_diag"
TP2="actor_rollout_ref.rollout.tensor_model_parallel_size=2"

# 3a. TP2 基线(裸)—— 若你们的栈有 271,这里会现形
STEPS=2 ASCEND_RT_VISIBLE_DEVICES=0,1 VERL_LOGPROB_DIAG_DUMP=/tmp/lp_diag \
  bash scripts/vllm018_upgrade/rl/npu/run_trloo_npu.sh $TP2 $DUMP 2>&1 | tee /tmp/npu_tp2.log
python scripts/vllm018_upgrade/rl/analyze_logprob_diag.py /tmp/lp_diag/*.pt | tee /tmp/npu_tp2_analysis.txt

# 3b. TP2 + NZ=0 —— Austin pending 的决定性实验(NZ 矩阵批变性假设)
STEPS=2 ASCEND_RT_VISIBLE_DEVICES=0,1 VLLM_ASCEND_ENABLE_NZ=0 VERL_LOGPROB_DIAG_DUMP=/tmp/lp_diag_nz0 \
  bash scripts/vllm018_upgrade/rl/npu/run_trloo_npu.sh $TP2 \
  "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGPROB_DIAG_DUMP=/tmp/lp_diag_nz0" \
  "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_ENABLE_NZ=0" 2>&1 | tee /tmp/npu_tp2_nz0.log
python scripts/vllm018_upgrade/rl/analyze_logprob_diag.py /tmp/lp_diag_nz0/*.pt

# 3c.(仅当 3b 仍脏)选择性 BI:只换 matmul、跳过 attention 覆盖 —— 见 Austin 病历 §8 阶梯
```

**判读矩阵**(用 analyzer 的十分位直方图 + top offending tokens):

| 观测 | 结论 |
|---|---|
| 3a 干净(pearson≈TP1,无早期位置异常) | 你们的栈没有 271 问题 —— Austin 的问题在其 0.20/2.10 组合或其 verl 分支,升级即解 |
| 3a 脏(早期位置尖峰、"\n\n"/271 类 token 领跑)+ **3b 干净** | **NZ matmul 批变性 = 根因坐实**(Austin 假设正确);生产方案 = TP2+NZ=0,量化其吞吐代价 |
| 3a 脏 + 3b 仍脏 | 批变性在 attention/HCCL 层 → 走 3c 选择性 BI 阶梯 |

与 CUDA 尾巴现象区分:我们的退化尾巴集中在**序列末 20%**且 onset 随内容漂移;271 集中在**固定
token id、结构边界**。analyzer 的输出能直接区分两者。

## Phase 4 — fully-async 生产形态(2 NPU,~30 分钟)

```bash
bash scripts/vllm018_upgrade/rl/npu/run_trloo_fullyasync_npu.sh 2>&1 | tee /tmp/npu_async.log
```
期望:Rollouter/Trainer 分池、`_fit_update_weights timing_s/param_sync` 每步出现(HCCL 推送)、
3/3 步 rc=0(CUDA 参照:~2s/sync)。

## 记录与回传

每个 phase 保留:完整 log、analyzer 输出、`pip freeze`、CANN/torch_npu 版本。失败时把
traceback + 对应 phase 的 log 发回来 —— verl 侧修复会保持版本门控,不回归 CUDA。

## 已知运维坑(CUDA 上踩过,NPU 同样适用)

- 每次 run 前脚本自带 `ray stop --force && rm -rf /tmp/ray`(残留 cluster 文件会让 ray.init 连死集群)。
- 崩溃的 run 会留占显存的进程:重跑前 `npu-smi info` 查、按 PID 杀(vllm init 见
  "Free memory ... less than desired" 即此因)。
- 长 validation + 首次 logprob 路径可能超 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`(默认 300s):
  实验脚本已跳过 val;如需 val,把该 env 提到 1200。
