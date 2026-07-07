# NPU 实验方法(B2 + Austin/token-271 差分验证)— step-by-step

在 Ascend 机器上从上往下逐条执行,每步都给出**字面命令**和**期望输出**。目标:
① 验证我们的 vllm 0.18 + verl 栈在 NPU 上端到端可用(B2);
② 用 TP2 受控矩阵回答 Austin 的 token-271 问题(NZ 批变性假设)。
我们的 CUDA 对照(TP1≡TP2,pearson 0.9993)已排除 vllm core/verl 集成 —— NPU 矩阵将把根因
钉死在 vllm-ascend kernel 层。

本仓库已内置的前置修复(无需操作):Gap A fp32 logprob(`DKV_FP32_LOGPROB=1` 默认开)、
OOV mask 的 TP 分片修复(全局词表坐标)。

约定:下文 `$WORK` = 你在 Ascend 机上的工作目录;所有命令在激活的 conda env 里跑。
安装序列以 Austin 在同一目标栈上验证过的顺序为准(其 deploy-vllm018.md),已适配到本仓库。

---

## Phase 0 — 环境安装(一次性,~1-2 小时)

- [ ] **0.1 CANN 环境**(用机器上现成的 CANN 9 loader;不要用系统 /usr/local/Ascend 的旧版本):
  ```bash
  source /path/to/your/900env_npu.sh     # 或你们机器上等价的 CANN 环境脚本
  ```
- [ ] **0.2 新建 conda env**(隔离,不动既有环境):
  ```bash
  conda create -n verl-vllm018 python=3.11 -y && conda activate verl-vllm018
  ```
- [ ] **0.3 torch/torch_npu 2.9 栈 + 构建依赖**(从你们的 Ascend wheel 源):
  ```bash
  pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0
  pip install torch-npu==2.9.0 triton-ascend==3.2.0
  # CANN TBE 编译器依赖 + torchair 需要的 pkg_resources(否则 vllm-ascend 构建/初始化失败):
  pip install "numpy<2.0.0" scipy attrs cloudpickle decorator ml-dtypes psutil tornado absl-py "setuptools<81"
  python -c "import te.platform, pkg_resources; print('te + pkg_resources ok')"
  ```
  期望:`te + pkg_resources ok`
- [ ] **0.4 GATE-1:torch_npu 对 CANN 冒烟**:
  ```bash
  python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.device_count())"
  ```
  期望:`2.9.0 2.9.0 <卡数>`,无 `undefined symbol`/camem 错误。
- [ ] **0.5 vLLM + vllm-ascend 0.18.0(源码 editable)**:
  ```bash
  mkdir -p $WORK/installation && cd $WORK/installation
  git clone https://github.com/vllm-project/vllm.git        && (cd vllm && git checkout v0.18.0)
  git clone https://github.com/vllm-project/vllm-ascend.git && (cd vllm-ascend && git checkout v0.18.0)
  cd $WORK/installation/vllm        && VLLM_TARGET_DEVICE=empty pip install -e . --no-build-isolation
  cd $WORK/installation/vllm-ascend && pip install -e . --no-build-isolation
  pip show vllm | head -2 ; pip show vllm-ascend | head -2
  ```
  期望:两者 `Version: 0.18.0`。若 vllm-ascend 构建报 `No module named 'scipy'`:0.3 的 TBE
  依赖没装全,补装后 `rm -rf csrc/build` 重试。
  (vLLM 会把 transformers 拉到 4.57.x —— 对我们的 Qwen3-0.6B 阶梯足够;只有以后上
  Qwen3.5/3.6 才需要强制 `transformers==5.5.3`,见 Austin deploy 文档 step 4b。)
- [ ] **0.6 本仓库(bundle 拷入后)**:
  ```bash
  cd $WORK && git clone /path/to/verl-vllm018.bundle verl && cd verl && git checkout main
  pip install -e . --no-deps --no-build-isolation
  pip install "ray[default]==2.48.0" tensordict codetiming hydra-core datasets pylatexenc torchdata mathruler pybind11
  ```
  (缺什么补什么,直到 0.7 的检查全绿;以上是 CUDA 侧实测的最小集。)
- [ ] **0.7 import 全检 + checkpoint 引擎**:
  ```bash
  python scripts/vllm018_upgrade/rl/check_rl_imports.py
  python -c "import verl.checkpoint_engine; from verl.checkpoint_engine.base import CheckpointEngineRegistry as R; print(sorted(R._registry.keys()))"
  ```
  期望:5 个 `OK` + `trloo registered OK` + exit 0;registry 含 `'nccl'`(NPU 上由 HCCL 实现)。
- [ ] **0.8 数据 + 模型**:
  ```bash
  PY=python bash scripts/vllm018_upgrade/rl/prep_gsm8k.sh          # 产出 ~/data/gsm8k/{train,test}.parquet
  python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-0.6B'))"
  # 离线机器:预先拷模型目录,后续命令用 SMOKE_MODEL=/abs/path 覆盖
  ```

## Phase 1 — 推理 smoke(~10 分钟)

- [ ] ```bash
  ASCEND_RT_VISIBLE_DEVICES=0 python scripts/vllm018_upgrade/smoke_rollout_npu.py
  ```
  期望:两行非空 `PROMPT=... -> '...'` + `SMOKE PASS`。
  (顺带验证 npu_vllm_patch 0.18 分支 + 审计遗留的 rotary 点。GEN_OK 后若尾随一条
  "EngineCore proc died" 是 teardown 噪音,可忽略 —— Austin 同样观察到。)

## Phase 2 — TP1 基线 + Gap A 度量(~30 分钟,1 NPU)

- [x] **2a. TP1 + 诊断**(fp32 logprob 默认开)— **PASS 2026-07-07**:pearson 0.9991、
  rollout_is_mean 0.9999、fraction_low 0.0、score/mean→0.25。engine 对如 CUDA 般干净。
  (先修:torch.compile 在 CANN9 上编译失败,已默认 `TORCHDYNAMO_DISABLE=1`,见下方运维坑。)
  ```bash
  STEPS=5 PY=python bash scripts/vllm018_upgrade/rl/npu/run_trloo_npu.sh 2>&1 | tee /tmp/npu_tp1.log
  grep -oE "pearson_corr[^ ]+|rollout_is_mean[^ ]+|rollout_is_ratio_fraction_low[^ ]+" /tmp/npu_tp1.log | tail -6
  ```
  期望:5 步 rc=0;**pearson ≥ 0.99**(参照:CUDA 0.9993,Austin 的 NPU TP1 0.9997)。
- [ ] **2b. Gap A 的 A/B**(量化你们 torch_npu 版本的 bf16 偏差):
  ```bash
  STEPS=2 DKV_FP32_LOGPROB=0 PY=python bash scripts/vllm018_upgrade/rl/npu/run_trloo_npu.sh 2>&1 | tee /tmp/npu_tp1_bf16.log
  ```
  判读:2b 相对 2a 的 `rollout_is_mean` 偏移即 Gap A(torch_npu 2.10 上 ≈×0.94;2.9 上应≈1)。
  **PASS 2026-07-07**:2b rollout_is_mean 1.00006 vs 2a 0.9999(Δ≈7e-5,统计零)→ **torch_npu 2.9
  无 Gap-A**;DKV_FP32_LOGPROB 在这台机器是无害的空保险(2.10 才 ×0.94)。保留默认开。
- [ ] 若 attention/padding 报错:加 `ATTN_FALLBACK=1` 重试(eager + no-remove-padding,CUDA 验证过)。

## Phase 3 — TP2 矩阵:Austin 判定实验(核心,~1 小时,2 NPU)

> ⚠️ **单节点 TP2 必须把 worker group 撑到 2 卡**:base launcher 硬编码
> `trainer.n_gpus_per_node=1`(world size 1),TP2 rollout 要 world≥2,否则 verl 直接
> 报 `world_size < tensor_model_parallel_size`。CUDA 侧靠 `nnodes=2` 拿到 world 2;
> 单机 NPU 必须追加 `trainer.n_gpus_per_node=2`(Hydra 后者覆盖前者,已验证)。
> Austin checkout 未打 24b1aa4 前,命令前缀再加 `TORCHDYNAMO_DISABLE=1`。
> ⚠️ 经 `+ray_kwargs...env_vars` 传数值型 env(如 NZ)**必须加引号** `='0'`,否则 Hydra 解析成
> int 0,Ray 的 `env_vars` 只收 `Dict[str,str]` → `TypeError: value 0 is of type int`。

- [ ] **3a. TP2 裸跑 + token 级 dump**(`0,1` 换成你实际空闲的两张卡):
  ```bash
  STEPS=2 PY=python TORCHDYNAMO_DISABLE=1 VERL_LOGPROB_DIAG_DUMP=/tmp/lp_diag ASCEND_RT_VISIBLE_DEVICES=0,1 bash scripts/vllm018_upgrade/rl/npu/run_trloo_npu.sh actor_rollout_ref.model.path=/home/canada_group_folder/ckpt/Qwen3-0.6B actor_rollout_ref.rollout.tensor_model_parallel_size=2 trainer.n_gpus_per_node=2 "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGPROB_DIAG_DUMP=/tmp/lp_diag" 2>&1 | tee /tmp/npu_tp2.log
  python scripts/vllm018_upgrade/rl/analyze_logprob_diag.py "/tmp/lp_diag/*.pt" | tee /tmp/npu_tp2_analysis.txt
  ```
  看 analyzer 的:十分位直方图(早期位置有无异常)、top offending tokens(有无 "\n\n"/271 类)、
  首 token 异常数。
  **CLEAN 2026-07-07(cards 2,3)**:pearson 0.9940、rollout_is_mean 0.9984、veto 0/0。45890
  token 里仅 4 个 extreme(0.01%),token id 11/13/438/2704 各 1 次(**非固定 token,非 271**)。
  → Austin 的 271 在本栈**不复现**;其问题限于 0.20/torch_npu 2.10。
- [x] **3b. TP2 + NZ=0(决定性实验)— 2026-07-07(cards 2,3)**:pearson 0.9955、
  rollout_is_mean 0.9982、veto 0/0,与 3a 同一干净带。48/46102 extreme(0.10%,散布全十分位、
  无固定 token,top id=11 逗号)。**NZ 开关都不复现 271;extreme 数 4↔48 是不同生成序列的抽样
  方差(未固定 seed),非 NZ 效应。→ NZ 不是驱动因素。**
  ```bash
  STEPS=2 PY=python TORCHDYNAMO_DISABLE=1 VLLM_ASCEND_ENABLE_NZ=0 VERL_LOGPROB_DIAG_DUMP=/tmp/lp_diag_nz0 ASCEND_RT_VISIBLE_DEVICES=0,1 bash scripts/vllm018_upgrade/rl/npu/run_trloo_npu.sh actor_rollout_ref.model.path=/home/canada_group_folder/ckpt/Qwen3-0.6B actor_rollout_ref.rollout.tensor_model_parallel_size=2 trainer.n_gpus_per_node=2 "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGPROB_DIAG_DUMP=/tmp/lp_diag_nz0" "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_ENABLE_NZ='0'" 2>&1 | tee /tmp/npu_tp2_nz0.log
  python scripts/vllm018_upgrade/rl/analyze_logprob_diag.py "/tmp/lp_diag_nz0/*.pt"
  ```
- [ ] **判定矩阵**:

  **→ 命中 ROW 1(3a 干净)。最终判定见下。**

  | 观测 | 结论 |
  |---|---|
  | ✅ **3a 干净(pearson≈TP1,无早期异常)← 本次结果** | 你们的栈没有 271 —— Austin 的问题在其 0.20/torch_npu 2.10 组合或其分支,升级即解 |
  | 3a 脏 + **3b 干净** | **NZ matmul 批变性 = 根因坐实**;生产方案 = TP2 + `VLLM_ASCEND_ENABLE_NZ=0`,顺带记录吞吐差 |
  | 3a 脏 + 3b 仍脏 | 批变性在 attention/HCCL 层 → 按 Austin 病历 §8 走选择性 BI(只换 matmul、跳过 attention 覆盖,并保留 npu 采样算子避开 triton 崩溃) |

  区分我们 CUDA 的尾巴现象:退化尾巴集中在**序列末 20%、onset 随内容漂移**;271 集中在
  **固定 token id、结构边界** —— analyzer 输出可直接分辨。

## Phase 4 — fully-async 生产形态(~30 分钟,2 NPU)

- [x] **PASS 2026-07-07(cards 4,5,gpu_mem_util=0.3)**:
  ```bash
  PY=python TORCHDYNAMO_DISABLE=1 TOTAL_ROLLOUT_STEPS=8 ASCEND_RT_VISIBLE_DEVICES=4,5 bash scripts/vllm018_upgrade/rl/npu/run_trloo_fullyasync_npu.sh actor_rollout_ref.model.path=/home/canada_group_folder/ckpt/Qwen3-0.6B actor_rollout_ref.rollout.gpu_memory_utilization=0.3 2>&1 | tee /tmp/npu_async.log
  grep -E "param_sync|param_version|[0-9]+/[0-9]+ \[" /tmp/npu_async.log | tail -12
  ```
  结果:Rollouter(pid 46871)/Trainer(pid 45734)分池在两张 NPU(两个独立 Ray resource pool,
  Ray 自动错开物理卡)、`_fit_update_weights timing_s/param_sync` 3.51s(v0)→2.57s(v1)、
  `param_version` 0→1 递增(HCCL 权重推送成功;CUDA 参照 ~2s/步)、干净退出。**NPU 无需 cupy**
  (checkpoint 后端 "nccl" 解析为 HCCL 实现)。
  - ⚠️ 坑1:上一轮 TP2 的 `VLLMWorker_TP` 进程会残留占 ~30GB/卡且 `ray stop` 杀不掉 → 下一轮
    rollout 报 `Free memory 30.46/60.96 < desired 0.5`。`npu-smi info` 查、按 pid 杀自己的
    `VLLMWorker*`(共享机勿 blanket-pkill,卡 1 有别人的 `VLLMEngineCor`),或换干净卡。
  - ⚠️ 坑2:0.6B 用 `gpu_memory_utilization=0.5` 是浪费(要 ~30GB KV),降到 0.3(~18GB)即可。

## 记录与回传

每个 Phase 保留:完整 log、analyzer 输出、`pip freeze`、CANN/torch_npu 版本。任何一步失败:
把 traceback + 该 Phase 的 log 发回来;verl 侧修复保持版本门控,不回归 CUDA。

## 运维坑速查(CUDA 上踩过,NPU 同样适用)

- 每次 run 前:`ray stop --force && rm -rf /tmp/ray`(NPU 启动器已内置)——残留
  `ray_current_cluster` 会让 ray.init 连上死集群。
- 崩溃的 run 会留占显存的进程:`npu-smi info` 查、按 PID 杀;vllm init 报
  `Free memory ... less than desired gpu_memory_utilization` 即此因。
- 大批量 validation + 首次 logprob 路径可能超 RPC 超时:实验脚本已跳过 val;需要 val 时
  设 `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200`。
- 多次实验之间共享卡:确保上一轮(含别人的 0.20 栈)已停干净再开跑。
- **torch.compile 在这台机器上不可用**(2026-07-07 Phase 2a 踩到):entropy 路径
  (`compute_entropy_from_logits`)走 torch._inductor → triton-ascend,后者引用
  `RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE`,而 CANN 9.0.0 已把它改名为 `*_SIMT_DVG_WARP_STACK_SIZE`
  → `MLIRCompilationError` / `NoTritonConfigsError`,Ray 里表现为 "cannot pickle 'frame' object"。
  修复:launcher 已默认 `export TORCHDYNAMO_DISABLE=1`(dynamo 关闭 → 全部 torch.compile 退回
  eager,数值无损,rollout 本就 enforce_eager)。要 A/B 编译时传 `TORCHDYNAMO_DISABLE=0`。
  这是 triton-ascend 与 CANN 头文件的版本错配,非 verl 问题。
