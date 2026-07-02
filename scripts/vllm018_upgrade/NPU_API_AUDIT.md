# NPU API Audit — vllm-ascend 0.18 vs `npu_vllm_patch.py` (Task B0)

Source checked: `../vllm-ascend` @ `v0.18.0-37-g36e15a2f`, `../vllm` @ `v0.18.0`.
Goal: determine what the 0.18 branch of `verl/utils/vllm/npu_vllm_patch.py` must wire,
given the patch currently handles only 0.11 and 0.13/0.14.

## Symbol map

| Symbol used by the patch | 0.13/0.14 location | 0.18 location | Verdict |
|---|---|---|---|
| `vllm_ascend.ops.linear_op.SequenceRowParallelOp` | linear_op.py | `linear_op.py:476` | **unchanged** |
| `SequenceRowParallelOp.matmul_and_reduce` | method | `linear_op.py:500` | **unchanged** |
| `vllm_ascend.ascend_forward_context.select_moe_comm_method` | module fn | `ascend_forward_context.py:203` sig `(num_tokens, vllm_config, is_draft_model=False)` | **present** (wrapper is `*args,**kwargs` → signature-agnostic) |
| `vllm_ascend.ascend_forward_context.MoECommType` | Enum | `ascend_forward_context.py:26` | **unchanged** |
| `vllm_ascend.utils.AscendDeviceType` | Enum A2/A3/_310P/A5 | `utils.py:703` (A2=0,A3=1,_310P=2,A5=3) | **unchanged** |
| `vllm_ascend.utils.get_ascend_device_type` | fn | `utils.py:743` | **present** (new clean API) |
| `vllm.model_executor.layers.rotary_embedding.common.ApplyRotaryEmb` | class, `__init__(enforce_enable, is_neox_style, enable_fp32_compute)` | `common.py:123` — **same `__init__` signature** | **unchanged** |
| `vllm.model_executor.layers.fused_moe.FusedMoE` / `.weight_loader` | class/method | `fused_moe/layer.py:276` / `:1039` | **present** (weight_loader now `@overload`-typed; impl exists) |
| `vllm_ascend.worker.model_runner_v1.NPUModelRunner._select_moe_comm_method` (0.11 branch only) | method | **GONE** — 0.18 uses module-level `select_moe_comm_method` | 0.11-style not applicable to 0.18 |

## SoC / A2 detection

`AscendDeviceType` members are identical (A2/A3/_310P/A5) and
`torch_npu.npu.get_soc_version()` mapping is unchanged (utils.py
`check_ascend_device_type`: 220–225→A2, 250–255→A3, 200–205→_310P, 260→A5).
So the patch's existing `_is_ascend_soc_version_A2_v013_local()` works verbatim on
0.18. 0.18 also exposes `get_ascend_device_type()` returning an `AscendDeviceType`,
which is the cleaner way to test for A2.

The patch's `is_A2` dispatch currently matches exact strings
`vllm.__version__ == "0.11.0"` / `"0.13.0"`; 0.18 reports e.g. `0.18.0` →
falls through to `is_A2 = False`. Must add a 0.18 case.

## Conclusion (drives B1)

The 0.18 NPU patch path is **the same as the existing 0.13/0.14 path**:
- module-level `select_moe_comm_method` wrapper (`vllm_ascend_v013_select_moe_comm_method_wrapper`),
- `SequenceRowParallelOp.matmul_and_reduce` wrapper (`vllm_ascend_v013_matmul_and_reduce_wrapper`),
- rotary `patch_vllm013_rotary_emb()` (signature matches),
- `FusedMoE.weight_loader` wrapper (`vllm_v013_weight_loader_method_wrapper`).

**B1 plan:** broaden the two version gates (the `is_A2` dispatch and the
`VERL_NPU_ENABLE_A2_PATCH_VLLM_ASCEND_MC2` block) so vLLM 0.18 takes the 0.13-style
wiring, keeping the 0.11 and 0.13/0.14 branches intact. Reuse the v013 helpers
(no new wrapper bodies needed) → DRY.

**Verify on remote Ascend (B2), cannot check statically here:**
1. Rotary patch: the 0.13 replacement `__init__` calls
   `super(ApplyRotaryEmb, self).__init__()` with no args, while 0.18's real
   `__init__` calls `super().__init__(enforce_enable=enforce_enable)`. Confirm the
   no-arg super call is still valid for `CustomOp` in 0.18 (likely fine; flag if it
   raises).
2. `FusedMoE.weight_loader` wrapper signature
   `(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success=False)`
   vs the 0.18 impl — confirm it still matches at call time.
