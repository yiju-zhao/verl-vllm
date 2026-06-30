"""Minimal 2-node NCCL all-reduce test over ConnectX.
rank0 on gx10-090e, rank1 on bruce. Expect allreduce of (rank+1) == 3 for world_size=2.
"""
import os
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
r, ws = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(0)
t = torch.ones(1, device="cuda") * (r + 1)
dist.all_reduce(t)
expect = sum(range(1, ws + 1))
ok = abs(t.item() - expect) < 1e-6
print(f"[rank {r}/{ws}] allreduce={t.item():.1f} expect={expect} {'OK' if ok else 'MISMATCH'}", flush=True)
dist.destroy_process_group()
