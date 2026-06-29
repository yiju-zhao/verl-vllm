"""KernelBench correctness helpers (toolkit layer)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from kernelgym.toolkit.kernelbench.exec_types import (
    KernelExecResult,
    get_error_name,
    set_seed,
)

def get_limit(data_type):
    """根据数据类型获取精度阈值"""
    import torch
    if data_type == torch.float16:
        return 0.004
    elif data_type == torch.bfloat16:
        return 0.03
    elif data_type == torch.int8:
        return 0.01
    else:
        return 0.02

def compare(fw_out, impl_out, limit, data_type):
    """对比框架输出和实现输出"""
    import torch
    fw_flat = fw_out.flatten().detach().cpu()
    impl_flat = impl_out.flatten()
    if isinstance(impl_flat, torch.Tensor):
        impl_flat = impl_flat.detach().cpu()
    else:
        impl_flat = torch.tensor(impl_flat, dtype=fw_flat.dtype)

    size = fw_flat.numel()

    if fw_flat.shape != impl_flat.shape:
        raise AssertionError(
            f"验证失败，输出形状不一致: framework={fw_flat.shape}, impl={impl_flat.shape}"
        )

    fw_nan_mask = torch.isnan(fw_flat)
    impl_nan_mask = torch.isnan(impl_flat)
    if not torch.equal(fw_nan_mask, impl_nan_mask):
        fw_nan_count = fw_nan_mask.sum().item()
        impl_nan_count = impl_nan_mask.sum().item()
        raise AssertionError(
            f"验证失败，NaN 位置不匹配: Framework={fw_nan_count}/{size}, "
            f"Implementation={impl_nan_count}/{size}"
        )

    fw_inf_mask = torch.isinf(fw_flat)
    impl_inf_mask = torch.isinf(impl_flat)
    if not torch.equal(fw_inf_mask, impl_inf_mask):
        fw_inf_count = fw_inf_mask.sum().item()
        impl_inf_count = impl_inf_mask.sum().item()
        raise AssertionError(
            f"验证失败，Inf 位置不匹配: Framework={fw_inf_count}/{size}, "
            f"Implementation={impl_inf_count}/{size}"
        )
    if fw_inf_mask.any():
        if not torch.equal(
            torch.sign(fw_flat[fw_inf_mask]),
            torch.sign(impl_flat[impl_inf_mask]),
        ):
            raise AssertionError("验证失败，Inf 符号不匹配")

    finite_mask = torch.isfinite(fw_flat) & torch.isfinite(impl_flat)
    finite_count = finite_mask.sum().item()
    if finite_count == 0:
        print("警告: 所有值都是非有限值，跳过精度检查")
        return

    fw_finite = fw_flat[finite_mask]
    impl_finite = impl_flat[finite_mask]

    if fw_finite.dtype == torch.bool:
        if not torch.equal(fw_finite, impl_finite):
            raise AssertionError(f"验证失败，布尔值不匹配: dtype={data_type}")
        return

    if impl_finite.dtype != fw_finite.dtype:
        impl_finite = impl_finite.to(fw_finite.dtype)

    abs_diff = torch.abs(fw_finite.float() - impl_finite.float())
    abs_ref = torch.abs(fw_finite.float())
    eps = 1e-8
    relative_error = torch.where(abs_ref > eps, abs_diff / abs_ref, abs_diff)

    err_cnt = (relative_error > limit).sum().item()
    limit_cnt = int(finite_count * limit)

    if err_cnt > limit_cnt:
        max_error = relative_error.max().item()
        mean_error = relative_error.mean().item()
        mismatch_mask = relative_error > limit
        mismatch_indices = torch.where(mismatch_mask)[0]
        num_to_show = min(10, len(mismatch_indices))

        error_msg = (
            f"验证失败，输出不一致(误差数/最大容忍误差数): "
            f"err_cnt={err_cnt} / {limit_cnt}, dtype={data_type}, limit={limit}\n"
        )
        error_msg += f"最大相对误差: {max_error:.6e}, 平均相对误差: {mean_error:.6e}\n"
        error_msg += f"前 {num_to_show} 个不一致的值:\n"
        for i in range(num_to_show):
            idx = mismatch_indices[i].item()
            error_msg += (
                f"  位置[{idx}]: framework={fw_finite[idx]:.6e}, "
                f"impl={impl_finite[idx]:.6e}, "
                f"相对误差={relative_error[idx]:.6e}\n"
            )
        raise AssertionError(error_msg)

def register_and_format_exception(
    exception_type: str,
    exception_msg: Exception | str,
    metadata: dict,
    verbose: bool = False,
    truncate: bool = False,
    max_length: int = 200,
):
    if verbose:
        print(f"[Exception {exception_type}] {str(exception_msg)} ")

    metadata[exception_type] = exception_msg
    return metadata


def run_and_check_correctness(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    num_correct_trials: int,
    verbose: bool = False,
    seed: int = 42,
    device: Any = None,
) -> KernelExecResult:
    pass_count = 0

    torch.manual_seed(seed)
    correctness_trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)
    ]

    with torch.no_grad():
        for trial in range(num_correct_trials):
            trial_seed = correctness_trial_seeds[trial]
            if verbose:
                print(f"[Eval] Generating Random Input with seed {trial_seed}")

            set_seed(trial_seed)
            inputs = get_inputs_fn()
            inputs = [
                x.npu(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]

            set_seed(trial_seed)
            model = original_model_instance.npu(device=device)

            set_seed(trial_seed)
            model_new = new_model_instance.npu(device=device)
            if verbose:
                print(f"device: {device}")
                print(f"inputs: {inputs}")

            output = model(*inputs)
            torch.npu.synchronize(device=device)

            try:
                output_new = model_new(*inputs)
                torch.npu.synchronize(device=device)
                if output.shape != output_new.shape:
                    metadata = register_and_format_exception(
                        "correctness_issue",
                        f"Output shape mismatch: Expected {output.shape}, got {output_new.shape}",
                        metadata,
                    )
                    metadata["correctness_issue_name"] = "correctness_issue"
                    if verbose:
                        print(
                            f"[FAIL] trial {trial}: Output shape mismatch: Expected {output.shape}, got {output_new.shape}"
                        )
                    return KernelExecResult(
                        compiled=True, correctness=False, metadata=metadata
                    )

                try:
                    data_type = output.dtype
                    limit = get_limit(data_type)
                    compare(output, output_new, limit, data_type)
                    pass_count += 1
                    if verbose:
                        print(f"[PASS] trial {trial}: New Model matches Model")
                except AssertionError as e:
                    max_diff = torch.max(torch.abs(output - output_new)).item()
                    avg_diff = torch.mean(torch.abs(output - output_new)).item()
                    metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
                    metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
                    metadata["correctness_issue"] = "Output mismatch"
                    if verbose:
                        print(f"[FAIL] trial {trial}: Output mismatch")

            except Exception as e:
                print("[Error] Exception happens during correctness check")
                print(f"Error in launching kernel for ModelNew: {e}")

                metadata = register_and_format_exception(
                    "runtime_error", e, metadata, truncate=False
                )
                metadata["runtime_error_name"] = get_error_name(e)
                return KernelExecResult(
                    compiled=True, correctness=False, metadata=metadata
                )

    if verbose:
        print(
            f"[Eval] Pass count: {pass_count}, num_correct_trials: {num_correct_trials}"
        )

    metadata["correctness_trials"] = f"({pass_count} / {num_correct_trials})"

    if pass_count == num_correct_trials:
        return KernelExecResult(compiled=True, correctness=True, metadata=metadata)
    return KernelExecResult(compiled=True, correctness=False, metadata=metadata)
