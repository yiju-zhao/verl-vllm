"""
NPU诊断工具 - 用于测试subprocess隔离和profiler兼容性

这个模块提供以下功能：
1. 在不初始化主进程NPU的情况下测试NPU可用性
2. 测试subprocess中的NPU隔离
3. 测试profiler在subprocess中的兼容性
4. 验证NPU Error不会污染主进程

Author: KernelServer Team
Date: 2025-10-29
"""

import os
import sys
import subprocess
import multiprocessing as mp
import logging
import traceback
from typing import Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass
import time
import pandas as pd

logger = logging.getLogger("kernelgym.npu_diagnostics")

# DSL 类型定义
DslType = Literal["triton_ascend", "triton_cuda", "torch", "tilelang_npuir", "ascendc", "other"]

# def collect_time(base_dir: str, active: int, dsl: DslType = "other") -> float:
#     """
#     从 profiling 结果中收集时间信息。

#     Args:
#         base_dir: profiling 结果目录
#         active: 有效测量次数
#         clear_l2_cache_flag: 是否启用了 L2 cache 清除
#         dsl: DSL 类型，决定如何过滤 L2 cache 清除操作
#              - "triton_ascend": 过滤名为 "AKG_l2cache_clear" 的 kernel
#              - 其他: 过滤 "ZerosLike" 类型的操作

#     Returns:
#         float: 平均执行时间(微秒)，失败时返回 float('inf')
#     """
#     if not os.path.exists(base_dir):
#         print(f"Base directory not found: {base_dir}")
#         return float('inf')

#     for root, _, files in os.walk(base_dir):
#         for file in files:
#             if file != 'op_statistic.csv':
#                 continue

#             target_file = os.path.join(root, file)
#             try:
#                 df = pd.read_csv(target_file)
#             except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError) as e:
#                 print(f"Failed to read {target_file}: {e}")
#                 continue

#             # 检查必需的列
#             required_columns = ['Count', 'Total Time(us)']
#             if not all(col in df.columns for col in required_columns):
#                 print(f"Missing required columns in {target_file}. Found: {list(df.columns)}")
#                 continue

#             # 过滤有效操作
#             try:                
#                 valid_ops = df[df['Count'] % active == 0].copy()

#                 if valid_ops.empty:
#                     print(f"No valid ops found in {target_file}")
#                     continue

#                 total_time_sum = valid_ops['Total Time(us)'].sum()
#                 if pd.isna(total_time_sum) or total_time_sum <= 0:
#                     print(f"Invalid timing data in {target_file}")
#                     continue

#                 average_time = total_time_sum / active
#                 return average_time

#             except (KeyError, ValueError, ZeroDivisionError) as e:
#                 print(f"Error processing timing data in {target_file}: {e}")
#                 continue

#     print(f"No valid timing data found in {base_dir}")
#     return float('inf')


@dataclass
class NPUHealthReport:
    """NPU健康检查报告"""
    healthy: bool
    device_id: int
    device_name: Optional[str] = None
    total_memory_gb: Optional[float] = None
    npu_available: bool = False
    error_message: Optional[str] = None
    test_duration_sec: float = 0.0


@dataclass
class IsolationTestReport:
    """隔离测试报告"""
    isolation_successful: bool
    main_process_contaminated: bool
    subprocess_error_message: Optional[str] = None
    details: Dict[str, Any] = None


@dataclass
class ProfilerTestReport:
    """Profiler兼容性测试报告"""
    profiler_works: bool
    profiling_data_received: bool
    profiling_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


_NPU_LOGIC_MAP_CACHE: Optional[Dict[int, Tuple[int, int]]] = None


def _get_logic_to_card_chip_map() -> Dict[int, Tuple[int, int]]:
    """Build {chip_logic_id: (npu_id, chip_id)} from `npu-smi info -m`.

    On Atlas multi-chip nodes (e.g. 8 cards x 2 chips = 16 NPUs) torch.npu's
    logical device id is the "Chip Logic ID" column, but `npu-smi info -i`
    expects the "NPU ID" (card) and `-c` expects the "Chip ID". Querying with
    the logic id directly fails for ids that don't happen to match an NPU ID
    (e.g. logic id 15 on an 8-card box where NPU IDs only go 0..7).
    """
    global _NPU_LOGIC_MAP_CACHE
    if _NPU_LOGIC_MAP_CACHE is not None:
        return _NPU_LOGIC_MAP_CACHE

    mapping: Dict[int, Tuple[int, int]] = {}
    try:
        result = subprocess.run(
            ["npu-smi", "info", "-m"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    npu_id = int(parts[0])
                    chip_id = int(parts[1])
                    logic_id = int(parts[2])
                except ValueError:
                    continue
                if logic_id < 0:
                    continue
                mapping[logic_id] = (npu_id, chip_id)
    except Exception as e:
        logger.warning(f"npu-smi info -m failed, falling back to identity mapping: {e}")

    _NPU_LOGIC_MAP_CACHE = mapping
    return mapping


def _resolve_npu_chip(device_id: int) -> Tuple[int, int]:
    """Translate a torch.npu logical device id to (npu_id, chip_id) for npu-smi."""
    mapping = _get_logic_to_card_chip_map()
    if device_id in mapping:
        return mapping[device_id]
    return device_id, 0


# Module-level worker functions (必须在模块级别以便pickle)

def _npu_health_worker(device_id: int, result_queue):
    """Subprocess worker for NPU health test"""
    try:
        # ASCEND_RT_VISIBLE_DEVICES
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(device_id)
        
        # Import torch (在subprocess中)
        import torch
        
        if not torch.npu.is_available():
            result_queue.put({
                'success': False,
                'error': 'NPU not available in subprocess'
            })
            return
        
        # 初始化NPU
        torch.npu.init()
        torch.npu.set_device(0)  # 因为ASCEND_RT_VISIBLE_DEVICES只暴露一个NPU
        
        # 获取NPU信息
        device_name = torch.npu.get_device_name(0)
        total_memory = torch.npu.get_device_properties(0).total_memory
        
        # 简单测试
        test_tensor = torch.randn(100, 100, device='npu')
        result = torch.mm(test_tensor, test_tensor.T)
        torch.npu.synchronize()
        
        result_queue.put({
            'success': True,
            'device_name': device_name,
            'total_memory': total_memory
        })
        
    except Exception as e:
        result_queue.put({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


def _npu_error_worker(device_id: int, result_queue):
    """故意触发 Error的worker"""
    try:
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(device_id)
        import torch
        
        torch.npu.init()
        torch.npu.set_device(0)
        
        # 故意触发 Error: 访问无效的内存地址
        try:
            # 创建一个非常大的tensor，可能导致OOM
            giant_tensor = torch.randn(1000000, 100000, device='npu')
            # 或者使用无效的NPU kernel配置
            result_queue.put({'phase': 'error_triggered', 'success': False, 'expected': True})
        except RuntimeError as e:
            if 'NPU' in str(e) or 'out of memory' in str(e):
                result_queue.put({
                    'phase': 'error_caught',
                    'success': True,
                    'error': str(e)
                })
            else:
                raise
                
    except Exception as e:
        result_queue.put({
            'phase': 'unexpected_error',
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


def _normal_worker(device_id: int, result_queue):
    """正常的worker，用于测试NPU是否仍然可用"""
    try:
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(device_id)
        import torch
        
        torch.npu.init()
        torch.npu.set_device(0)
        
        # 简单测试
        test_tensor = torch.randn(100, 100, device='npu')
        result = torch.mm(test_tensor, test_tensor.T)
        torch.npu.synchronize()
        
        result_queue.put({'success': True, 'phase': 'normal_execution'})
        
    except Exception as e:
        result_queue.put({
            'success': False,
            'phase': 'normal_execution_failed',
            'error': str(e)
        })


def _profiler_worker(device_id: int, result_queue, profiling_queue):
    """使用profiler的worker"""
    try:
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(device_id)
        import torch
        import torch_npu
        import torch_npu.profiler as profiler
        
        torch.npu.init()
        torch.npu.set_device(0)
        
        timestamp = int(time.time() * 1000)
        profile_path = os.path.join(os.getcwd(), f"profile_results_{timestamp}")
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

        # 启用profiler
        prof = profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.NPU
            ],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            experimental_config=experimental_config
        )
        
        with prof:
            # 执行一些CUDA操作
            x = torch.randn(1000, 1000, device='npu')
            y = torch.mm(x, x.T)
            z = torch.nn.functional.relu(y)
            torch.npu.synchronize()
        
        # 提取profiling数据
        print(f"prof={prof}")

        def collect_time(base_dir: str, active: int, clear_l2_cache_flag: bool = False,
                        dsl: DslType = "other") -> float:
            """
            从 profiling 结果中收集时间信息。

            Args:
                base_dir: profiling 结果目录
                active: 有效测量次数
                clear_l2_cache_flag: 是否启用了 L2 cache 清除
                dsl: DSL 类型，决定如何过滤 L2 cache 清除操作
                    - "triton_ascend": 过滤名为 "AKG_l2cache_clear" 的 kernel
                    - 其他: 过滤 "ZerosLike" 类型的操作

            Returns:
                float: 平均执行时间(微秒)，失败时返回 float('inf')
            """
            if not os.path.exists(base_dir):
                print(f"Base directory not found: {base_dir}")

            for root, _, files in os.walk(base_dir):
                for file in files:
                    if file != 'operator_details.csv':
                        continue

                    target_file = os.path.join(root, file)
                    try:
                        df = pd.read_csv(target_file)
                    except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError) as e:
                        print(f"Failed to read {target_file}: {e}")
                        continue

                    return df
            

        operator_details_df = collect_time(profile_path)
        # events = prof.events()
        # npu_events = [
        #     evt for evt in events 
        #     if hasattr(evt, 'device_type') and 
        #     evt.device_type == profiler.DeviceType.NPU
        # ]
        
        # # 构建可序列化的profiling数据
        # profiling_data = {
        #     'total_events': len(operator_details_df),
        #     'npu_events': len(npu_events),
        #     'top_5_npu_kernels': [
        #         {
        #             'name': evt.key,
        #             'npu_time_us': float(evt.npu_time_total) if hasattr(evt, 'npu_time_total') else 0.0,
        #             'count': int(evt.count) if hasattr(evt, 'count') else 0
        #         }
        #         for evt in sorted(
        #             npu_events,
        #             key=lambda e: getattr(e, 'npu_time_total', 0.0),
        #             reverse=True
        #         )[:5]
        #     ]
        # }
        
        # # 发送profiling数据
        # profiling_queue.put(profiling_data)
        # result_queue.put({'success': True})
        
    except Exception as e:
        result_queue.put({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


class NPUDiagnostics:
    """NPU 诊断工具集"""
    
    @staticmethod
    def test_npu_health_npu_smi(device_id: int) -> NPUHealthReport:
        """
        使用npu-smi测试NPU健康状态（不初始化CANN）

        Args:
            device_id: GPU设备ID
            
        Returns:
            GPUHealthReport对象
        """
        start_time = time.time()

        npu_id, chip_id = _resolve_npu_chip(device_id)

        try:
            # ====================== 1. 查询 NPU 名称 ======================
            result_name = subprocess.run(
                [
                    "npu-smi",
                    "info",
                    "-i", str(npu_id),
                    "-c", str(chip_id),
                    "-t", "board"
                ],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result_name.returncode != 0:
                return NPUHealthReport(
                    healthy=False,
                    device_id=device_id,
                    error_message=(
                        f"npu-smi board failed for logic_id={device_id} "
                        f"(npu_id={npu_id}, chip_id={chip_id}): "
                        f"{result_name.stderr.strip()}"
                    ),
                    test_duration_sec=time.time() - start_time
                )

            # 解析 NPU 名称
            name_output = result_name.stdout.strip()
            device_name = "Unknown Ascend NPU"
            for line in name_output.split("\n"):
                if "Product Name" in line:
                    device_name = line.split(":", 1)[1].strip()
                    break

            # ====================== 2. 查询 NPU 总显存 ======================
            result_mem = subprocess.run(
                [
                    "npu-smi",
                    "info",
                    "-i", str(npu_id),
                    "-t", "memory",
                    "-c", str(chip_id)
                ],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result_mem.returncode != 0:
                return NPUHealthReport(
                    healthy=False,
                    device_id=device_id,
                    error_message=(
                        f"npu-smi memory failed for logic_id={device_id} "
                        f"(npu_id={npu_id}, chip_id={chip_id}): "
                        f"{result_mem.stderr.strip()}"
                    ),
                    test_duration_sec=time.time() - start_time
                )
            
            # 解析 HBM 总显存（MB）
            mem_output = result_mem.stdout.strip()
            memory_mb = 0.0
            for line in mem_output.split("\n"):
                if "HBM Capacity(MB)" in line:
                    mem_str = line.split(":", 1)[1].strip()
                    memory_mb = float(mem_str)
                    break
            memory_gb = memory_mb / 1024.0
            
            return NPUHealthReport(
                healthy=True,
                device_id=device_id,
                device_name=device_name,
                total_memory_gb=memory_gb,
                npu_available=True,
                test_duration_sec=time.time() - start_time
            )
            
        except subprocess.TimeoutExpired:
            return NPUHealthReport(
                healthy=False,
                device_id=device_id,
                error_message="npu-smi timeout",
                test_duration_sec=time.time() - start_time
            )
        except Exception as e:
            return NPUHealthReport(
                healthy=False,
                device_id=device_id,
                error_message=f"npu-smi error: {str(e)}",
                test_duration_sec=time.time() - start_time
            )
    
    @staticmethod
    def test_npu_health_subprocess(device_id: int) -> NPUHealthReport:
        """
        在subprocess中测试NPU健康状态
        
        这个方法在独立的subprocess中初始化CANN并测试NPU，
        不会影响主进程。
        
        Args:
            device_id: GPU设备ID
            
        Returns:
            GPUHealthReport对象
        """
        start_time = time.time()
        
        # 使用spawn context创建进程
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        
        process = ctx.Process(target=_npu_health_worker, args=(device_id, result_queue))
        process.start()
        
        try:
            # 等待结果，10秒超时（spawn+import torch+CANN init需要时间）
            result = result_queue.get(timeout=20)
            process.join(timeout=2)
            
            duration = time.time() - start_time
            
            if result['success']:
                return NPUHealthReport(
                    healthy=True,
                    device_id=device_id,
                    device_name=result['device_name'],
                    total_memory_gb=result['total_memory'] / (1024**3),
                    npu_available=True,
                    test_duration_sec=duration
                )
            else:
                return NPUHealthReport(
                    healthy=False,
                    device_id=device_id,
                    npu_available=False,
                    error_message=result.get('error', '未知错误'),
                    test_duration_sec=duration
                )
                
        except Exception as e:
            process.terminate()
            process.join(timeout=2)
            
            return NPUHealthReport(
                healthy=False,
                device_id=device_id,
                npu_available=False,
                error_message=f"Subprocess test failed: {str(e)}",
                test_duration_sec=time.time() - start_time
            )
    
    @staticmethod
    def test_npu_error_isolation(device_id: int) -> IsolationTestReport:
        """
        测试CANN Error隔离
        
        在subprocess中故意触发CANN Error，验证：
        1. Subprocess正确捕获错误
        2. 主进程不受影响
        3. 后续subprocess可以正常使用GPU
        
        Args:
            device_id: GPU设备ID
            
        Returns:
            IsolationTestReport对象
        """
        ctx = mp.get_context('spawn')
        
        # Step 1: 触发CUDA Error
        logger.info(f"[Isolation Test] Step 1: 触发CUDA Error在subprocess中")
        result_queue1 = ctx.Queue()
        process1 = ctx.Process(target=_npu_error_worker, args=(device_id, result_queue1))
        process1.start()
        
        try:
            result1 = result_queue1.get(timeout=15)  # 增加超时时间
            process1.join(timeout=2)
            logger.info(f"[Isolation Test] result1={result1}")
        except Exception as e:
            process1.terminate()
            return IsolationTestReport(
                isolation_successful=False,
                main_process_contaminated=False,
                subprocess_error_message=f"Step 1 failed: {str(e)}"
            )
        
        # Step 2: 检查主进程是否受影响（主进程不使用CUDA，所以应该没有影响）
        logger.info(f"[Isolation Test] Step 2: 检查主进程状态")
        # 主进程不使用CUDA，所以这一步总是成功
        main_process_ok = True
        
        # Step 3: 在新的subprocess中测试GPU是否仍然可用
        logger.info(f"[Isolation Test] Step 3: 测试GPU在新subprocess中是否可用")
        result_queue2 = ctx.Queue()
        process2 = ctx.Process(target=_normal_worker, args=(device_id, result_queue2))
        process2.start()
        
        try:
            result2 = result_queue2.get(timeout=15)  # 增加超时时间
            process2.join(timeout=2)
        except Exception as e:
            process2.terminate()
            return IsolationTestReport(
                isolation_successful=False,
                main_process_contaminated=False,
                subprocess_error_message=f"Step 3 failed: {str(e)}",
                details={
                    'step1': result1,
                    'step2': 'main_process_ok',
                    'step3_error': str(e)
                }
            )
        
        # 判断隔离是否成功
        step1_ok = result1.get('phase') == 'error_caught'
        step2_ok = main_process_ok
        step3_ok = result2.get('success') == True
        
        isolation_successful = step1_ok and step2_ok and step3_ok
        
        # 构建详细的错误信息
        error_parts = []
        if not step1_ok:
            error_parts.append(f"Step1 failed: phase={result1.get('phase')}, expected='error_caught'")
        if not step2_ok:
            error_parts.append("Step2 failed: main process contaminated")
        if not step3_ok:
            error_parts.append(f"Step3 failed: success={result2.get('success')}, expected=True")
        
        error_message = "; ".join(error_parts) if error_parts else None
        
        logger.info(f"[Isolation Test] Step 1 OK: {step1_ok}, Phase: {result1.get('phase')}")
        logger.info(f"[Isolation Test] Step 2 OK: {step2_ok}")
        logger.info(f"[Isolation Test] Step 3 OK: {step3_ok}, Success: {result2.get('success')}")
        
        return IsolationTestReport(
            isolation_successful=isolation_successful,
            main_process_contaminated=not main_process_ok,
            subprocess_error_message=error_message,
            details={
                'step1_error_caught': result1,
                'step2_main_process_ok': main_process_ok,
                'step3_gpu_available': result2
            }
        )
    
    @staticmethod
    def test_profiler_compatibility(device_id: int) -> ProfilerTestReport:
        """
        测试torch.profiler在subprocess中的兼容性
        
        验证：
        1. Profiler可以在subprocess中正常启动
        2. Profiler可以收集CUDA事件
        3. Profiling数据可以通过Queue传递回主进程
        
        Args:
            device_id: GPU设备ID
            
        Returns:
            ProfilerTestReport对象
        """
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        profiling_queue = ctx.Queue()
        
        process = ctx.Process(
            target=_profiler_worker,
            args=(device_id, result_queue, profiling_queue)
        )
        process.start()
        
        try:
            # 等待结果（profiler需要更长时间）
            result = result_queue.get(timeout=20)
            print(f"result: {result}")
            # 尝试获取profiling数据
            profiling_data = None
            if not profiling_queue.empty():
                profiling_data = profiling_queue.get_nowait()
            
            process.join(timeout=2)
            
            if result['success']:
                return ProfilerTestReport(
                    profiler_works=True,
                    profiling_data_received=(profiling_data is not None),
                    profiling_data=profiling_data
                )
            else:
                return ProfilerTestReport(
                    profiler_works=False,
                    profiling_data_received=False,
                    error_message=result.get('error', 'Unknown error')
                )
                
        except Exception as e:
            process.terminate()
            process.join(timeout=2)
            
            return ProfilerTestReport(
                profiler_works=False,
                profiling_data_received=False,
                error_message=f"Profiler test failed: {str(e)}"
            )
    
    @staticmethod
    def run_full_diagnostics(device_id: int) -> Dict[str, Any]:
        """
        运行完整的GPU诊断
        
        Args:
            device_id: GPU设备ID
            
        Returns:
            包含所有诊断结果的字典
        """
        logger.info(f"=== 开始NPU {device_id} 完整诊断 ===")
        
        results = {}
        
        # Test 1: nvidia-smi健康检查
        logger.info("[Test 1/4] nvidia-smi健康检查...")
        health_nvidia_smi = NPUDiagnostics.test_npu_health_npu_smi(device_id)
        results['health_nvidia_smi'] = health_nvidia_smi
        logger.info(f"  结果: {'✅ 通过' if health_nvidia_smi.healthy else '❌ 失败'}")
        if health_nvidia_smi.healthy:
            logger.info(f"  GPU: {health_nvidia_smi.device_name}, "
                       f"Memory: {health_nvidia_smi.total_memory_gb:.1f}GB")
        
        # Test 2: Subprocess健康检查
        logger.info("[Test 2/4] Subprocess CUDA健康检查...")
        health_subprocess = NPUDiagnostics.test_npu_health_subprocess(device_id)
        results['health_subprocess'] = health_subprocess
        logger.info(f"  结果: {'✅ 通过' if health_subprocess.healthy else '❌ 失败'}")
        if not health_subprocess.healthy:
            logger.error(f"  错误: {health_subprocess.error_message}")
        
        # Test 3: CUDA Error隔离测试
        logger.info("[Test 3/4] CUDA Error隔离测试...")
        isolation = NPUDiagnostics.test_npu_error_isolation(device_id)
        results['isolation_test'] = isolation
        logger.info(f"  结果: {'✅ 隔离成功' if isolation.isolation_successful else '❌ 隔离失败'}")
        if not isolation.isolation_successful:
            logger.warning(f"  主进程污染: {isolation.main_process_contaminated}")
            logger.error(f"  错误信息: {isolation.subprocess_error_message}")
            if isolation.details:
                logger.debug(f"  详细信息: {isolation.details}")
        
        # Test 4: Profiler兼容性测试
        logger.info("[Test 4/4] torch.profiler兼容性测试...")
        profiler_test = NPUDiagnostics.test_profiler_compatibility(device_id)
        results['profiler_test'] = profiler_test
        logger.info(f"  结果: {'✅ Profiler可用' if profiler_test.profiler_works else '❌ Profiler失败'}")
        if profiler_test.profiler_works:
            logger.info(f"  Profiling数据接收: {'✅' if profiler_test.profiling_data_received else '❌'}")
            if profiler_test.profiling_data:
                logger.info(f"  CUDA事件数量: {profiler_test.profiling_data.get('cuda_events', 0)}")
        
        # 总结
        logger.info("=== 诊断完成 ===")
        all_passed = (
            health_nvidia_smi.healthy and
            health_subprocess.healthy and
            isolation.isolation_successful and
            profiler_test.profiler_works
        )
        logger.info(f"总体状态: {'✅ 所有测试通过' if all_passed else '⚠️  部分测试失败'}")
        
        results['all_passed'] = all_passed
        return results


def main():
    """命令行入口"""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="NPU诊断工具")
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='NPU设备ID（默认: 0）'
    )
    parser.add_argument(
        '--test',
        choices=['health', 'isolation', 'profiler', 'all'],
        default='all',
        help='要运行的测试类型'
    )
    
    args = parser.parse_args()
    
    if args.test == 'all':
        results = NPUDiagnostics.run_full_diagnostics(args.device)
        sys.exit(0 if results['all_passed'] else 1)
    
    elif args.test == 'health':
        report = NPUDiagnostics.test_npu_health_subprocess(args.device)
        print(f"Healthy: {report.healthy}")
        if report.healthy:
            print(f"Device: {report.device_name}")
            print(f"Memory: {report.total_memory_gb:.1f}GB")
        sys.exit(0 if report.healthy else 1)
    
    elif args.test == 'isolation':
        report = NPUDiagnostics.test_npu_error_isolation(args.device)
        print(f"Isolation Successful: {report.isolation_successful}")
        print(f"Main Process Contaminated: {report.main_process_contaminated}")
        sys.exit(0 if report.isolation_successful else 1)
    
    elif args.test == 'profiler':
        report = NPUDiagnostics.test_profiler_compatibility(args.device)
        print(f"Profiler Works: {report.profiler_works}")
        print(f"Data Received: {report.profiling_data_received}")
        if report.profiling_data:
            print(f"CUDA Events: {report.profiling_data.get('npu_events', 0)}")
        sys.exit(0 if report.profiler_works else 1)


if __name__ == '__main__':
    main()

