# vLLM 0.18 TP=2 across two DGX Sparks — runbook

Goal: run vLLM `tensor_parallel_size=2` across two single-GPU GB10 Sparks
(this machine `gx10-090e` + `spark-bruce`/`gx10-ca1e`). **Validated: `TP2_SMOKE PASS`
with Qwen3-0.6B on vllm 0.18.1.dev0.**

## Topology
- gx10-090e: 1×GB10, ConnectX `enp1s0f1np1` = **192.168.1.101/24**
- spark-bruce (gx10-ca1e): 1×GB10, ConnectX `enp1s0f1np1` = **192.168.1.106/24**
- 200G ConnectX is the data path (all-reduce). Tailscale (100.111.48.70) = mgmt/SSH only.
- RoCE/RDMA devices exist (`rocep1s0f1`, GID idx 3 = RoCEv2) but NCCL-over-RoCE hangs
  (fabric needs PFC/tuning) — we use **TCP NCCL over the 200G link** (`NCCL_IB_DISABLE=1`),
  functionally fine.

## One-time setup
1. **ConnectX IPs** (per node, `enp1s0f1np1`): `.101` here, `.106` on bruce
   (`sudo ip addr add 192.168.1.106/24 dev enp1s0f1np1 && sudo ip link set ... up`).
2. **bruce env = mirror of `drkernel310`** at the *identical* path
   `/home/yubaifeng/e84381970/envs/drkernel310` (rsync over ConnectX; the env is a
   self-contained py3.10 standalone, so same-path mirroring makes editable installs +
   shebangs resolve — no rebuild; bruce's GPU/CUDA/torch are ABI-identical). The vllm +
   verl source must also sit at the identical `/home/yubaifeng/...` paths.
   - bruce's own `kernelgym` env is py3.12 and is NOT used (Ray/vllm need matching py3.10).

## Bring up the cluster (per session)
NCCL pinned to ConnectX; **`VLLM_HOST_IP` must be each node's ConnectX IP** (else vllm
picks the WiFi IP and dies with "nodes have non-unique IP"):

```bash
ENV=/home/yubaifeng/e84381970/envs/drkernel310
N="NCCL_SOCKET_IFNAME=enp1s0f1np1 GLOO_SOCKET_IFNAME=enp1s0f1np1"
# head (this machine)
env $N VLLM_HOST_IP=192.168.1.101 $ENV/bin/ray start --head \
    --node-ip-address=192.168.1.101 --port=6379 --num-gpus=1 --disable-usage-stats
# worker (bruce, over ssh)
ssh -i ~/.ssh/id_ed25519_spark bruce@192.168.1.106 \
  "env $N VLLM_HOST_IP=192.168.1.106 $ENV/bin/ray start \
   --address=192.168.1.101:6379 --node-ip-address=192.168.1.106 --num-gpus=1"
$ENV/bin/ray status   # expect 2 nodes, 0.0/2.0 GPU
```

## Run vLLM TP=2
```bash
cd .../scripts/vllm018_upgrade   # NOT a dir containing a `vllm/` subdir (namespace shadowing!)
env NCCL_SOCKET_IFNAME=enp1s0f1np1 GLOO_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_DISABLE=1 \
    VLLM_HOST_IP=192.168.1.101 RAY_ADDRESS=192.168.1.101:6379 VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  $ENV/bin/python3.10 tp2_smoke.py
```
`tp2_smoke.py` uses `LLM(..., tensor_parallel_size=2, distributed_executor_backend="ray",
enable_flashinfer_autotune=False, enforce_eager=True)`.

## Gotchas hit (and fixes)
1. **`enable_flashinfer_autotune=False`** is essential: the flashinfer JIT autotuner runs
   a cross-node collective per trial; over TCP-NCCL it effectively hangs (>70 min, GPU 96%,
   no progress). Disabling it (per-invocation engine arg) → warmup logs "Skipping FlashInfer
   autotune" and generation proceeds. (RoCE would also fix the speed but isn't configured.)
2. **Per-node `VLLM_HOST_IP`** = ConnectX IP (see above).
3. **Run dir must not contain a `vllm/` subdir** (else `import vllm` resolves to the source
   tree as a namespace package: "cannot import name 'LLM' from 'vllm'").
4. **Stale placement groups**: killing a vllm run leaks its Ray PG (next run sees "no GPU
   available"). Always `ray stop --force` + restart between runs.
5. NCCL bootstrap: `NCCL_SOCKET_IFNAME=enp1s0f1np1` so it uses ConnectX, not WiFi/Tailscale.

## Status / next
- TP=2 cross-node **inference** works. For throughput, get RoCE/RDMA NCCL working
  (PFC/lossless fabric) so the per-layer all-reduce isn't TCP-bound.
- This unblocks TP=2 for the verl RL pipeline (the RL goal doc's TP=2 milestone).
