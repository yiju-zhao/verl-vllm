# Installation Guide (NPU)

Run these steps once to build the training environment, then reuse the image/env
across nodes. Commands assume an `aarch64` host — adjust paths for `x86_64`.

## Step 0: Pull and Enter the Base Docker Image

```bash
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.3.rc1-a3-ubuntu22.04-py3.11
docker run -it --name verl-env \
    swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.3.rc1-a3-ubuntu22.04-py3.11 /bin/bash
```

## Step 1: Install System Dependencies

```bash
apt-get update -y && apt-get install -y --no-install-recommends \
    gcc g++ cmake libnuma-dev wget git curl jq vim build-essential
pip install --upgrade pip packaging setuptools==80.10.2
```

## Step 2: Clone Required Repositories

```bash
git clone --depth 1 --branch v0.11.0 https://github.com/vllm-project/vllm.git
git clone --depth 1 --branch v0.11.0 https://github.com/vllm-project/vllm-ascend.git
git clone https://gitcode.com/Ascend/MindSpeed.git
git clone --depth 1 --branch core_v0.14.0 https://github.com/NVIDIA/Megatron-LM.git
```

## Step 3: Set Up NPU Environment Variables

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/8.3.RC1/aarch64-linux/devlib/linux/aarch64:$LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

## Step 4: Install PyTorch and NPU Support

```bash
pip install torch==2.7.1 torch_npu==2.7.1 torchvision==0.22.1 transformers==4.57.6
```

## Step 5: Install vLLM and vLLM-Ascend

```bash
cd vllm        && VLLM_TARGET_DEVICE=empty pip install -v -e . && cd ..
cd vllm-ascend && pip install -v -e .                          && cd ..
```

## Step 6: Install Megatron-LM and MindSpeed

```bash
cd Megatron-LM && pip install -v -e .                          && cd ..
cd MindSpeed   && git checkout core_r0.14.0 && pip install -e . && cd ..
```

## Step 7: Clean Up Conflicting Packages

```bash
pip uninstall -y triton triton-ascend
```

## Step 8: Install mbridge

```bash
pip install git+https://github.com/ISEEKYAN/mbridge.git@4389fcc450c5f90f0cf22e9c77e3d49e2c643e24
```

## Step 9: Install verl

Install this fork (not the upstream clone) so the DR.Kernel recipe is on the
PYTHONPATH. Run from the repo root:

```bash
pip install -r requirements-npu.txt
pip install -v -e .
```

## Step 10: Install Apex

```bash
git clone -b master https://gitcode.com/Ascend/apex.git
cd apex && bash scripts/build.sh --python=3.11
pip install apex/dist/apex-0.1+ascend-cp311-cp311-linux_aarch64.whl
```

## Step 11: Install the KernelGYM Reward Server

The DR.Kernel reward server ships in-repo at `recipe/NPU-kernelGym/`. Install
its dependencies (python deps + Redis) from the repo root:

```bash
cd recipe/NPU-kernelGym && bash setup.sh && cd ../..
```

See [`recipe/NPU-kernelGym/README.md`](../NPU-kernelGym/README.md) for deployment
details.

