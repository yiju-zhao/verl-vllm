"""KernelBench profiling helpers (toolkit layer)."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch

from kernelgym.config import settings

logger = logging.getLogger("kernelgym.toolkit.kernelbench.profiling")


def compute_triton_kernel_coverage(matched_triton_kernels: List[str], profilling_result: Dict[str, Any]):
    """Compute the coverage of the matched triton kernels in the profiling result."""

    def _matches_profiler_name(captured: str, profiler_name: str) -> bool:
        cap = captured.lower()
        prof = profiler_name.lower()
        if cap == prof:
            return True
        if cap in prof or prof in cap:
            return True
        return False

    kernels = matched_triton_kernels
    num_custom_kernels = 0
    kernel_names = [kernel.split(" ")[0] for kernel in kernels]

    kernels_in_profiling = profilling_result["kernels"]

    total_time = 0.0
    matched_cuda_time = 0.0
    triton_kernels_in_profiling = []

    for prof_kernel in kernels_in_profiling:
        prof_name = prof_kernel["name"]
        cuda_time = float(prof_kernel["device_time_us"])
        cpu_time = float(prof_kernel["cpu_time_us"])
        total_time += cuda_time + cpu_time

        if any(_matches_profiler_name(kernel_name, prof_name) for kernel_name in kernel_names):
            triton_kernels_in_profiling.append(prof_name)
            num_custom_kernels += 1
            matched_cuda_time += cuda_time

    triton_kernels_not_in_profiling = [
        kernel_name
        for kernel_name in kernel_names
        if not any(_matches_profiler_name(kernel_name, prof_name) for prof_name in triton_kernels_in_profiling)
    ]

    return {
        "num_custom_kernels": num_custom_kernels,
        "num_total_kernels": len(kernels_in_profiling),
        "total_kernel_run_time_in_profiling_us": total_time,
        "custom_kernel_cuda_time_in_profiling_us": matched_cuda_time,
        "triton_kernels_not_in_profiling": triton_kernels_not_in_profiling,
        "triton_kernels_in_profiling": triton_kernels_in_profiling,
    }


@contextmanager
def profiling_context(enabled: bool = True, num_warmup:int = 3, profile_path: str = "./"):
    if not enabled:
        yield None
        return

    try:
        import torch_npu
        import torch_npu.profiler as profiler

        activities = []
        if "cpu" in settings.profiling_activities:
            activities.append(profiler.ProfilerActivity.CPU)
        if "npu" in settings.profiling_activities:
            activities.append(profiler.ProfilerActivity.NPU)

        print(f"[Profiler] Initializing with activities: {[str(a) for a in activities]}")

        if not activities:
            print("[Profiler] No activities configured, profiler will return no data")
            yield None
            return

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=[
                torch_npu.profiler.ExportType.Text
                ],
            profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
            mstx=False,    # 原参数名msprof_tx改为mstx，新版本依旧兼容原参数名msprof_tx
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            l2_cache=False,
            op_attr=True,
            data_simplification=False,
            record_op_args=False,
            gc_detect_threshold=None,
            host_sys=[],
            sys_io=False,
            sys_interconnection=False
        )

        prof = profiler.profile(
            activities=activities,
            record_shapes=settings.profiling_record_shapes,
            profile_memory=settings.profiling_profile_memory,
            with_stack=settings.profiling_with_stack,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_path),
            experimental_config=experimental_config,
            # schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=1),
        )

        prof.__enter__()
        try:
            print("[Profiler] Profiler started successfully")
            npu_available = torch.npu.is_available()
            npu_visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
            device_info = "npu:unavailable"
            if npu_available:
                try:
                    current_device = torch.npu.current_device()
                    device_name = torch.npu.get_device_name(current_device)
                    device_info = f"npu:{current_device} ({device_name})"
                except Exception as e:
                    device_info = f"npu:unknown (error={e})"
            print(
                "[Profiler] Context pid=%s npu_available=%s device=%s ASCEND_RT_VISIBLE_DEVICES=%s",
                os.getpid(),
                npu_available,
                device_info,
                npu_visible,
            )
            if npu_available:
                try:
                    test = torch.ones((1024,), device="npu")
                    _ = test.sum()
                    torch.npu.synchronize()
                    print("[Profiler] Self-test NPU op executed")
                except Exception as e:
                    print(f"[Profiler] Self-test failed: {e}")
            yield prof
        finally:
            try:
                prof.__exit__(None, None, None)
                print("[Profiler] Profiler stopped successfully")
            except Exception as e:
                print(f"[Profiler] Error during profiler cleanup: {e}")

    except Exception as e:
        logger.warning(f"[Profiler] Failed to initialize profiler: {e}. Continuing without profiling.")
        yield None


def extract_profiling_metrics(base_dir: str) -> Dict[str, Any]:
    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")

    try:
        import pandas as pd 
        op_time_file_path = None
        op_memory_file_path = None
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file == 'operator_details.csv':
                    op_time_file_path = os.path.join(root, file) 
                elif file == 'memory_record.csv':
                    op_memory_file_path = os.path.join(root, file) 
                else:
                    continue

        try:
            op_time_df = pd.read_csv(op_time_file_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError) as e:
            print(f"Failed to read {op_time_file_path}: {e}")

        try:
            op_mem_df = pd.read_csv(op_memory_file_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError) as e:
            print(f"Failed to read {op_memory_file_path}: {e}")
        

        op_time_df['Input Shapes'] = op_time_df['Input Shapes'].fillna("")
        op_time_agg = op_time_df.groupby(['Name','Input Shapes']).mean()
        op_time_agg['count'] = op_time_df.groupby(['Name','Input Shapes']).size()
        op_time_agg = op_time_agg.reset_index()

        events = op_time_agg.to_dict('records')
        total_events = len(events)
        npu_device_event_count = 0
        npu_time_event_count = 0
        self_npu_time_event_count = 0

        logger.debug(f"[Profiler] Captured {total_events} total events")

        def _safe_metric(evt: Any, names: Tuple[str, ...], default: float = 0.0) -> float:
            for name in names:
                # ===================== 修复点 =====================
                # 1. 如果是字典
                if isinstance(evt, dict):
                    if name in evt:
                        value = evt[name]
                    else:
                        continue
                # 2. 如果是对象（torch profiler event）
                else:
                    if hasattr(evt, name):
                        value = getattr(evt)
                    else:
                        continue
                # ==================================================

                if callable(value):
                    try:
                        value = value()
                    except Exception:
                        continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
            return default

        # int 版本也一起修复
        def _safe_int_metric(evt: Any, names: Tuple[str, ...], default: int = 0) -> int:
            for name in names:
                if isinstance(evt, dict):
                    if name in evt:
                        value = evt[name]
                    else:
                        continue
                else:
                    if hasattr(evt, name):
                        value = getattr(evt)
                    else:
                        continue

                if callable(value):
                    try:
                        value = value()
                    except Exception:
                        continue
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
            return default

        npu_kernels = []
        total_cpu_time = 0.0
        total_self_cuda_time = 0.0
        for evt in events:
            cpu_time_us = _safe_metric(evt, ("Host Total Duration(us)", "cpu_time_total", "cpu_time"), 0.0)
            total_cpu_time += cpu_time_us
            device_time_us = _safe_metric(
                evt,
                ("Device Total Duration(us)", "device_time_total", "device_time", "cuda_time_total", "cuda_time"),
                0.0,
            )
            self_device_time_us = _safe_metric(
                evt,
                ("Device Self Duration(us)", "self_cuda_time_total", "self_cuda_time"),
                0.0,
            )
            if self_device_time_us > 0.0:
                self_npu_time_event_count += 1
                total_self_cuda_time += self_device_time_us
            if device_time_us <= 0.0:
                continue

            device_type = evt.get("Name", None)
            if "aclnn" in device_type:
                npu_device_event_count += 1                
            npu_time_event_count += 1

            kernel_entry = {
                "name": evt.get("Name", "unknown"),
                "device_time_us": device_time_us,
                "cpu_time_us": cpu_time_us,
                "count": _safe_int_metric(evt, ("count",), 0),
            }
            print(kernel_entry)
            memory_usage = _safe_metric(evt, ("cuda_memory_usage",), 0.0)
            if memory_usage > 0.0:
                kernel_entry["cuda_memory_usage"] = memory_usage
            npu_kernels.append(kernel_entry)
        npu_kernels.sort(key=lambda x: x["device_time_us"], reverse=True)

        logger.debug(
            f"[Profiler] Filtered to {len(npu_kernels)} CUDA kernels (from {len(events)} total)"
        )
        if len(npu_kernels) == 0 and len(events) > 0:
            logger.warning(
                f"[Profiler] Captured events but no CUDA kernels! Event types: {[getattr(evt, 'device_type', 'unknown') for evt in list(events)[:5]]}"
            )

        memory_stats = {}
        try:
            if torch.npu.is_available():
                device = torch.npu.current_device()
                memory_stats = {
                    "allocated_mb": torch.npu.memory_allocated(device) / (1024 * 1024),
                    "reserved_mb": torch.npu.memory_reserved(device) / (1024 * 1024),
                    "max_allocated_mb": torch.npu.max_memory_allocated(device) / (1024 * 1024),
                    "max_reserved_mb": torch.npu.max_memory_reserved(device) / (1024 * 1024),
                }
        except Exception as e:
            logger.warning(f"[Profiler] Failed to collect memory stats: {e}")

        profiling_metrics = {
            "kernels": npu_kernels,
            "kernel_count": len(npu_kernels),
            "total_cpu_time_us": total_cpu_time,
            "total_device_time_us": sum(k["device_time_us"] for k in npu_kernels),
            "total_self_device_time_us": total_self_cuda_time,
            "npu_device_event_count": npu_device_event_count,
            "npu_time_event_count": npu_time_event_count,
            "self_npu_time_event_count": self_npu_time_event_count,
            "memory_stats": memory_stats,
        }

        if len(npu_kernels) == 0:
            profiling_metrics["profiling_warning"] = (
                "Profiler captured no CUDA kernels. This may indicate a profiler failure."
            )
        return profiling_metrics

    except Exception as e:
        logger.warning(f"[Profiler] Failed to extract profiling metrics: {e}")
        return {"profiling_error": str(e)}
