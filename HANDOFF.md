# HANDOFF — verl + vLLM 0.18 升级与验证(接续文档)

> 给新会话的 Claude Code:这是完整上下文。仓库 = drkernel verl port(verl fork),
> 任务 = 把 vllm 支持从 0.11 升到 0.18 并验证 RL 全通路(CUDA + Ascend NPU)。
> GitHub: `github.com/yiju-zhao/verl-vllm`(main;本地 GB10 工作副本在
> `/home/yubaifeng/e84381970/experiment/verl-vllm/drkernel-verl-port-drkernel`)。
> 用户偏好:**给用户的 shell 命令一律单行**(多行反斜杠粘贴出过两次事故)。

## 一、总目标与分期

1. ✅ vLLM 0.18 CUDA 升级(编译/推理/TP=2)
2. ✅ CUDA RL 全通路(colocated 单卡 + 双机 TP=2 + fully-async 生产形态)+ 数值验证
3. 🔄 **B2:NPU 实机验证(进行中,当前在 Phase 2)** ← 现在的主线
4. ⏳ RL-4:真实 drkernel(8B + KernelGYM + kernel 数据集)——Austin 机器上物料齐备

## 二、已完成(全部入库并推送)

### CUDA(GB10 gx10-090e,sm_121,torch 2.9.1+cu130,env `drkernel310`)
- vllm 0.18.1.dev0 源码编译安装(`use_existing_torch.py` + `--no-build-isolation`,
  ccache 可断点续编);verl import 全绿,0 处 API 需改(现有版本门控已覆盖 0.18)。
- 推理 smoke PASS;**跨双机 TP=2 推理 PASS**(第二台 Spark=bruce/gx10-ca1e,
  200G ConnectX:本机 192.168.1.101 / bruce 192.168.1.106,Ray 集群端口 **6380**
  (6379 被外来 root Ray 占),关键 knob `enable_flashinfer_autotune=False`
  ——autotuner 在跨机 TCP-NCCL 上挂死)。runbook: `scripts/vllm018_upgrade/TP2_CLUSTER_RUNBOOK.md`。
- **RL Stage-1**(trloo+gsm8k+Qwen3-0.6B,colocated `verl.trainer.main_ppo`):
  单卡 5 步 + TP=2 双机 5 步全部 rc=0。launcher:
  `scripts/vllm018_upgrade/rl/run_trloo_qwen3_0.6b_gsm8k.sh` / `run_trloo_tp2.sh`。
- **RL-2 fully-async**(drkernel 生产形态,`verl.experimental.fully_async_policy`):
  1+1 GPU 池跨双机,NCCL checkpoint-engine 权重推送 ~2s/步,3/3 步。
  launcher: `scripts/vllm018_upgrade/rl/run_trloo_fullyasync_1p1.sh`。
  依赖:**cupy-cuda13x 两台都要装**(否则 "Checkpoint engine nccl not registered");
  naive 后端与 fully-async 不兼容(无协程)。

### 数值诊断(关键方法论,复用于 NPU)
- 仪器:`actor_rollout_ref.rollout.calculate_log_probs=True`(→ pearson/probs_diff)
  + `algorithm.rollout_correction.rollout_is=token` + `rollout_is_threshold=2.0`(→ IS/RS 指标);
  token 级定位:env `VERL_LOGPROB_DIAG_DUMP=/tmp/lp_diag`(dump 钩子在
  `verl/utils/debug/metrics.py`,须同时经 `+ray_kwargs.ray_init.runtime_env.env_vars.…` 传入 Ray)
  + 分析器 `scripts/vllm018_upgrade/rl/analyze_logprob_diag.py`。
- **CUDA 结论**:原始 pearson 0.75–0.83 全部由"被 768 截断序列的数字复读尾巴"造成;
  **健康 token pearson 0.9993、RS rate 0%**;TP=1≡TP=2(TP 无罪)、
  sdpa/eager 仅 +0.02、fp32 前向零变化、采样参数干净。引擎对无需任何数值加固。
- ⚠️ 教训:`ppo_kl=0` 是平凡零(engine 自算 old_log_prob + 单 mini-batch),
  不能当一致性证据 —— 用 pearson/RS。

### Austin/token-271 差分诊断(对面团队的 TP2 翻车)
- 病历:Kernel-Agent 仓库 `npu-vllm0.18-port` 分支 `recipe/drkernel/docs/tp271-investigation.md`
  + `deploy-vllm018.md`;真实代码在 `Sawyer117/verl @ drkernel-port-vllm018`。
- 他们:Gap A(torch_npu bf16 logsumexp 偏差,已修 DKV_FP32_LOGPROB)+
  Gap B(token-271="\n\n" 尖峰,TP2-only,离线不复现,BI 能修 →
  **NZ matmul 批变性假设,待 NZ=0 判定**)+ OOV-151669 -inf 泄漏(TP 分片下 mask 失效)。
- 我们的 CUDA 对照(TP1≡TP2)已排除 vllm core/verl 集成 → 根因锁定在 vllm-ascend kernel 层。
- **已把他们两个修复移植进本仓库**(commit 9b9a9f5):
  1. `verl/utils/torch_functional.py::logprobs_from_logits_torch_npu` 分块 fp32 upcast
     (`DKV_FP32_LOGPROB=1` 默认开,=0 可 A/B);
  2. `verl/workers/rollout/vllm_rollout/utils.py::monkey_patch_compute_logits`
     改为全局词表坐标 mask(修 TP 分片洞)。
- 附带修的 verl 真 bug:`build_cli_args_from_config` 吞布尔 False → 现在发 `--no-<flag>`
  (vllm bool 参数是 BooleanOptionalAction)。commit 84f66c7。

## 三、当前主线:B2 NPU 验证(Austin 的共享 8×910B 机器)

### 环境(已借用成功,体检全 PASS)
- 用户 `f00518697`,conda env **`dr-kernel-npu-018`**(Austin 团队的:CANN9 loader
  `900env_npu.sh` + torch/torch_npu 2.9.0 + vllm/vllm-ascend 0.18.0 editable)。
- 我们的仓库 clone 在 `/home/canada_group_folder/verl-vllm`,pip -e 装进该 env。
- 模型:`/home/canada_group_folder/ckpt/Qwen3-0.6B`(本地目录,绕开一切网络)。
- gsm8k parquet:已备好(launcher 默认读 `~/data/gsm8k/{train,test}.parquet`)。
- 卡位:**6,7 = KernelGYM 常驻服务,避开**;卡 1 常被占;体检时 0/2/3/4/5 空闲。
- 体检脚本:`bash scripts/vllm018_upgrade/npu_env_check.sh`(判定按退出码,
  Ascend ERR99999 日志噪音已过滤)。

### 网络坑(这台机器,已踩明白)
- 公司 proxy(80.254.72.208:6688)做 TLS 拦截:python/httpx/wget 全报自签证书。
- HF 的 Xet 协议过 proxy 必卡 0% → `HF_HUB_DISABLE_XET=1`。
- 能用的下载路:`wget -c --no-check-certificate` + sha256 校验;或抓 proxy CA
  (`openssl s_client -showcerts …`)后 export `REQUESTS_CA_BUNDLE/SSL_CERT_FILE/CURL_CA_BUNDLE`。
- 最稳:模型/数据用本地路径,训练命令一律不碰网。
- ⚠️ 曾试过的 docker(16 NPU 那个)CANN=8.5.2 不满足 vllm-ascend 0.18 的
  **CANN==9.0.0 硬性要求**,已弃用,改借 Austin 的 host env。

### 进度(NPU_EXPERIMENT_PLAN.md 的梯子)
- ✅ Phase 0 环境(借用)  ✅ Phase 1 推理 smoke:**SMOKE PASS**(npu_vllm_patch
  0.18 分支 + rotary 审计点真机验证通过)
- 🔄 **Phase 2(下一步,命令已给用户)**:
  - 2a: `STEPS=5 ASCEND_RT_VISIBLE_DEVICES=0 PY=python bash scripts/vllm018_upgrade/rl/npu/run_trloo_npu.sh actor_rollout_ref.model.path=/home/canada_group_folder/ckpt/Qwen3-0.6B 2>&1 | tee /tmp/npu_tp1.log`
  - 判定:5 步 rc=0,pearson ≥0.99(基线 CUDA 0.9993 / Austin TP1 0.9997),score 非零
  - 2b(Gap-A A/B): 同命令前加 `STEPS=2 DKV_FP32_LOGPROB=0`,对比 `rollout_is_mean`
- ⏳ Phase 3(**Austin 判定实验**):TP2 裸跑 vs `VLLM_ASCEND_ENABLE_NZ=0`,
  各带 token dump + analyzer;判定表在 NPU_EXPERIMENT_PLAN.md。
  区分特征:Austin 的 271 = 固定 token/早期位置;我们 CUDA 的尾巴 = 序列末 20%/内容漂移。
- ⏳ Phase 4:fully-async(2 NPU,`rl/npu/run_trloo_fullyasync_npu.sh`;
  checkpoint 后端名仍是 "nccl",NPU 上由 HCCL 实现解析,无 cupy 依赖)。
- 若 attention/padding 报错:加 `ATTN_FALLBACK=1`(eager + no-remove-padding)。

## 四、关键文件地图(全在 `scripts/vllm018_upgrade/`)

- `NPU_EXPERIMENT_PLAN.md` — NPU step-by-step 实验协议(主文档)
- `NPU_RUN_GUIDE.md` / `NPU_API_AUDIT.md` — 安装指引 / vllm-ascend 0.18 API 审计
- `npu_env_check.sh` — 环境体检
- `rl/RL_NOTES.md` — 全部 RL 实验记录/坑/数值(最完整的实验台账)
- `rl/npu/run_trloo_npu.sh`、`rl/npu/run_trloo_fullyasync_npu.sh` — NPU launcher
- `rl/analyze_logprob_diag.py` — token 级 mismatch 定位
- `TP2_CLUSTER_RUNBOOK.md`、`BUILD_NOTES_cuda.md` — CUDA 侧运维
- 设计/计划:`docs/superpowers/specs/*.md`、`docs/superpowers/plans/*.md`

## 五、GB10 侧现状(如果回到本机继续)

- 双 Spark Ray 集群没常驻,按 TP2_CLUSTER_RUNBOOK 随时可拉起(:6380,
  每节点 VLLM_HOST_IP=各自 ConnectX IP,`RAY_memory_monitor_refresh_ms=0`)。
- ConnectX IP 已 netplan 持久化;RoCE 存在但 NCCL-over-RoCE 未调通(TCP 可用)。
- 运维口诀:跑前 `ray stop --force && rm -rf /tmp/ray`;崩溃后查残留 GPU 进程再重跑;
  大 validation + 首次 logprob 会超 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300`(调 1200 或跳 val)。

## 六、下一步队列

1. **收 Phase 2 结果**(用户在 Austin 机器上跑,等数字:pearson/rollout_is/score)
2. Phase 3 TP2 矩阵 → 出 Austin 问题的最终判定(三行判定表在实验计划里)
3. Phase 4 fully-async
4. 全部通过 → 更新 RL_NOTES + 推送 + (可选)把差分诊断结论同步给 Austin
5. RL-4:在 Austin 机器上用真实件(8B SFT 权重 + KernelGYM:8002 + kernel parquet)
