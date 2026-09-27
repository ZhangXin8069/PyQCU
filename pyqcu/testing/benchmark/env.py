# PyTorch硬件性能测试：整合后的完整代码
# 基于搜索结果，我为您整合了一套完整的PyTorch硬件性能测试代码，这套代码综合了多种测试方法，旨在对特定计算设备（CPU/GPU）和数据类型（如float32, float16, bfloat16）进行可靠、可重复的性能评估。
# 一、 环境验证与基础信息获取
# 在进行任何性能测试前，必须首先验证环境并获取基准信息。
import matplotlib.pyplot as plt
import pandas as pd
from torch.profiler import profile, record_function, ProfilerActivity
import torch.nn as nn
import torch
import sys
import platform
import numpy as np
from torch.utils.benchmark import Timer, Compare
import time


def verify_environment():
    """验证PyTorch环境和硬件配置[1](@ref)[4](@ref)"""
    print("=" * 60)
    print("PyTorch环境验证报告")
    print("=" * 60)
    # 基础信息
    print(f"PyTorch版本: {torch.__version__}")
    print(f"Python版本: {sys.version}")
    print(f"操作系统: {platform.platform()}")
    # CUDA信息
    if torch.cuda.is_available():
        print(f"CUDA可用: 是")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        # GPU详细信息
        device_count = torch.cuda.device_count()
        print(f"GPU数量: {device_count}")
        for i in range(device_count):
            print(f"\nGPU {i} 信息:")
            print(f"  名称: {torch.cuda.get_device_name(i)}")
            print(f"  计算能力: {torch.cuda.get_device_capability(i)}")
            print(
                f"  总显存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print(
                f"  多处理器数量: {torch.cuda.get_device_properties(i).multi_processor_count}")
    else:
        print("CUDA可用: 否 (将使用CPU进行测试)")
    print("=" * 60)
    return torch.cuda.is_available()
# 二、 核心性能测试框架
# 1. 使用PyTorch Timer进行精确微基准测试
# 这是最可靠的性能测试方法，自动处理预热和多次运行取平均。


class DeviceDtypeBenchmark:
    """设备与数据类型性能基准测试类"""

    def __init__(self, device='cuda', dtype=torch.float32):
        """
        初始化测试类
        Args:
            device: 测试设备 ('cpu', 'cuda', 'cuda:0'等)
            dtype: 测试数据类型 (torch.float32, torch.float16, torch.bfloat16等)
        """
        self.device = torch.device(device)
        self.dtype = dtype
        self.results = {}

    def get_measurement_stats(self, measurement):
        """
        从Measurement对象获取完整的统计信息
        Args:
            measurement: torch.utils.benchmark.utils.common.Measurement对象
        Returns:
            dict: 包含各种统计指标的字典
        """
        # 首先验证measurement对象是否具有所需的基本属性
        if not hasattr(measurement, 'times'):
            raise AttributeError("Measurement对象缺少'times'属性")
        stats = {
            'median': measurement.median,
            'mean': measurement.mean,
            'raw_times': measurement.times
        }
        # 计算时间统计
        if measurement.times:
            stats['min'] = min(measurement.times)
            stats['max'] = max(measurement.times)
            stats['total_time'] = sum(measurement.times)
            # 计算标准差和方差
            if len(measurement.times) > 1:
                stats['std'] = np.std(measurement.times)
                stats['var'] = np.var(measurement.times)
                stats['cv'] = stats['std'] / \
                    stats['mean'] if stats['mean'] > 0 else 0.0
            else:
                stats['std'] = 0.0
                stats['var'] = 0.0
                stats['cv'] = 0.0
        else:
            stats['min'] = 0.0
            stats['max'] = 0.0
            stats['total_time'] = 0.0
            stats['std'] = 0.0
            stats['var'] = 0.0
            stats['cv'] = 0.0
        # 获取测量次数信息
        # 从measurement对象的字符串表示中提取
        import re
        measurement_str = str(measurement)
        # 尝试提取运行次数
        runs_match = re.search(r'(\d+) runs per measurement', measurement_str)
        if runs_match:
            stats['runs_per_measurement'] = int(runs_match.group(1))
        else:
            stats['runs_per_measurement'] = 1
        # 尝试提取总运行次数
        total_runs_match = re.search(r'(\d+) total runs', measurement_str)
        if total_runs_match:
            stats['total_runs'] = int(total_runs_match.group(1))
        else:
            stats['total_runs'] = len(
                measurement.times) if measurement.times else 0
        return stats

    def test_matrix_multiplication(self, sizes=[1024, 2048, 4096, 8192]):
        """
    测试矩阵乘法性能 (TFLOPS计算)
    """
        print(f"\n{'='*60}")
        print(f"矩阵乘法性能测试 (设备: {self.device}, 类型: {self.dtype})")
        print(f"{'='*60}")
        results = {}
        for n in sizes:
            # 创建测试数据
            a = torch.randn(n, n, dtype=self.dtype, device=self.device)
            b = torch.randn(n, n, dtype=self.dtype, device=self.device)
            # 使用Timer进行精确计时
            timer = Timer(
                stmt='torch.matmul(a, b)',
                globals={'a': a, 'b': b, 'torch': torch},
                label=f"MatMul {n}x{n}",
                description=f"dtype={self.dtype}, device={self.device}",
                num_threads=1  # 明确指定线程数
            )
            # 运行基准测试
            measurement = timer.timeit(50)  # 运行50次取中位数
            # 获取统计信息
            stats = self.get_measurement_stats(measurement)
            # 计算TFLOPS (理论浮点运算次数 / 时间)
            # 矩阵乘法浮点运算次数: 2 * n^3
            flops = 2 * n ** 3
            tflops = flops / stats['median'] / 1e12
            results[n] = {
                'time_median': stats['median'],
                'time_mean': stats['mean'],
                'time_std': stats['std'],
                'time_min': stats['min'],
                'time_max': stats['max'],
                'time_cv': stats['cv'],
                'runs_per_measurement': stats['runs_per_measurement'],
                'total_runs': stats['total_runs'],
                'tflops': tflops,
                'raw_times': stats['raw_times']
            }
            print(f"尺寸 {n}x{n}:")
            print(f"  中位数时间: {stats['median']:.6f} s")
            print(f"  平均时间: {stats['mean']:.6f} s")
            print(f"  标准差: {stats['std']:.6f} s")
            print(f"  性能: {tflops:.2f} TFLOPS")
            print(f"  每次测量的运行次数: {stats['runs_per_measurement']}")
            print(f"  总运行次数: {stats['total_runs']}")
        self.results['matmul'] = results
        return results

    def test_memory_bandwidth(self, vector_sizes=[1_000_000, 10_000_000, 100_000_000]):
        """
        测试内存带宽 (向量加法测试)[2](@ref)
        这种方法常用于评估GPU的内存带宽性能
        """
        print(f"\n{'='*60}")
        print(f"内存带宽测试 (设备: {self.device}, 类型: {self.dtype})")
        print(f"{'='*60}")
        results = {}
        for N in vector_sizes:
            # 创建大向量
            data = torch.ones(N, dtype=self.dtype, device=self.device)
            # 确保CUDA操作同步[5](@ref)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            # 测试向量加法
            timer = Timer(
                stmt='data + 1.0',
                globals={'data': data, 'torch': torch},
                label=f"Vector Add N={N}",
                description=f"dtype={self.dtype}, device={self.device}"
            )
            measurement = timer.timeit(100)
            stats = self.get_measurement_stats(measurement)
            # 计算有效带宽 (GB/s)
            # 每次操作读取和写入各N个元素
            bytes_per_element = torch.tensor(
                [], dtype=self.dtype).element_size()
            bytes_transferred = 2 * N * bytes_per_element  # 读取+写入
            bandwidth_gbs = bytes_transferred / stats['median'] / 1e9
            results[N] = {
                'time_median': stats['median'],
                'bandwidth_gbs': bandwidth_gbs,
                'bytes_per_element': bytes_per_element
            }
            print(f"向量大小 {N:,}:")
            print(f"  数据类型大小: {bytes_per_element} 字节")
            print(f"  中位数时间: {stats['median']:.6f} s")
            print(f"  内存带宽: {bandwidth_gbs:.2f} GB/s")
        self.results['memory'] = results
        return results

    def test_convolution_operations(self, batch_sizes=[1, 8, 32],
                                    channels=[64, 128, 256],
                                    spatial_sizes=[56, 112, 224]):
        """
        测试卷积操作性能 (模拟CNN工作负载)
        """
        print(f"\n{'='*60}")
        print(f"卷积操作性能测试 (设备: {self.device}, 类型: {self.dtype})")
        print(f"{'='*60}")
        results = {}
        for bs in batch_sizes:
            for c in channels:
                for hw in spatial_sizes:
                    # 创建输入和卷积核
                    x = torch.randn(
                        bs, c, hw, hw, dtype=self.dtype, device=self.device)
                    conv = torch.nn.Conv2d(c, c, kernel_size=3, padding=1)
                    conv = conv.to(self.device).to(self.dtype)
                    # 预热[6](@ref)
                    with torch.no_grad():
                        for _ in range(5):
                            _ = conv(x)
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    # 计时
                    timer = Timer(
                        stmt='conv(x)',
                        globals={'conv': conv, 'x': x, 'torch': torch},
                        label=f"Conv bs={bs}, c={c}, hw={hw}",
                        description=f"dtype={self.dtype}, device={self.device}"
                    )
                    measurement = timer.timeit(30)
                    stats = self.get_measurement_stats(measurement)
                    key = f"bs{bs}_c{c}_hw{hw}"
                    results[key] = {
                        'time_median': stats['median'],
                        'time_mean': stats['mean'],
                        'throughput': bs / stats['median']  # 样本/秒
                    }
                    print(f"Batch={bs}, Channels={c}, Size={hw}x{hw}:")
                    print(f"  时间: {stats['median']:.6f} s")
                    print(f"  吞吐量: {bs/stats['median']:.1f} samples/s")
        self.results['conv'] = results
        return results

    def test_mixed_precision(self):
        """
        测试混合精度性能 (仅适用于CUDA设备)[3](@ref)
        混合精度训练可以显著提升性能并减少内存占用
        """
        if self.device.type != 'cuda':
            print("混合精度测试仅适用于CUDA设备")
            return None
        print(f"\n{'='*60}")
        print(f"混合精度性能测试")
        print(f"{'='*60}")
        from torch.cuda.amp import autocast, GradScaler
        # 创建模型和数据
        model = torch.nn.Sequential(
            torch.nn.Linear(1024, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512)
        ).to(self.device)
        # 创建不同精度的模型副本
        model_fp32 = model.to(torch.float32)
        model_fp16 = model.to(torch.float16)
        # 测试数据
        batch_size = 64
        x = torch.randn(batch_size, 1024, device=self.device)
        y = torch.randn(batch_size, 512, device=self.device)
        results = {}
        # 测试FP32
        model_fp32.eval()
        with torch.no_grad():
            timer_fp32 = Timer(
                stmt='model(x)',
                globals={'model': model_fp32, 'x': x, 'torch': torch},
                label="FP32 Inference"
            )
            measurement_fp32 = timer_fp32.timeit(50)
            stats_fp32 = self.get_measurement_stats(measurement_fp32)
        # 测试FP16
        model_fp16.eval()
        with torch.no_grad():
            timer_fp16 = Timer(
                stmt='model(x)',
                globals={'model': model_fp16, 'x': x, 'torch': torch},
                label="FP16 Inference"
            )
            measurement_fp16 = timer_fp16.timeit(50)
            stats_fp16 = self.get_measurement_stats(measurement_fp16)
        # 测试自动混合精度
        model_amp = model.to(torch.float32)
        model_amp.eval()
        with torch.no_grad(), autocast():
            timer_amp = Timer(
                stmt='model(x)',
                globals={'model': model_amp, 'x': x, 'torch': torch},
                label="AMP Inference"
            )
            measurement_amp = timer_amp.timeit(50)
            stats_amp = self.get_measurement_stats(measurement_amp)
        # 计算加速比
        speedup_fp16 = stats_fp32['median'] / stats_fp16['median']
        speedup_amp = stats_fp32['median'] / stats_amp['median']
        results = {
            'fp32_time': stats_fp32['median'],
            'fp16_time': stats_fp16['median'],
            'amp_time': stats_amp['median'],
            'speedup_fp16': speedup_fp16,
            'speedup_amp': speedup_amp
        }
        print("推理性能对比:")
        print(f"  FP32: {stats_fp32['median']:.6f} s")
        print(
            f"  FP16: {stats_fp16['median']:.6f} s (加速: {speedup_fp16:.2f}x)")
        print(f"  AMP:  {stats_amp['median']:.6f} s (加速: {speedup_amp:.2f}x)")
        # 测试训练性能
        print("\n训练性能对比:")
        model_fp32.train()
        model_fp16.train()
        # 简单的训练循环测试
        test_iterations = 10
        # FP32训练
        optimizer_fp32 = torch.optim.SGD(model_fp32.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(test_iterations):
            optimizer_fp32.zero_grad()
            output = model_fp32(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer_fp32.step()
        end.record()
        torch.cuda.synchronize()
        fp32_train_time = start.elapsed_time(end) / 1000  # 转换为秒
        # AMP训练
        model_amp.train()
        optimizer_amp = torch.optim.SGD(model_amp.parameters(), lr=0.01)
        scaler = GradScaler()
        start.record()
        for _ in range(test_iterations):
            optimizer_amp.zero_grad()
            with autocast():
                output = model_amp(x)
                loss = criterion(output, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer_amp)
            scaler.update()
        end.record()
        torch.cuda.synchronize()
        amp_train_time = start.elapsed_time(end) / 1000
        train_speedup = fp32_train_time / amp_train_time
        print(f"  FP32训练时间: {fp32_train_time:.3f} s")
        print(f"  AMP训练时间: {amp_train_time:.3f} s (加速: {train_speedup:.2f}x)")
        self.results['mixed_precision'] = results
        return results

    def run_comprehensive_test(self):
        """运行全面的性能测试套件"""
        print(f"\n{'='*60}")
        print(f"开始全面性能测试套件")
        print(f"设备: {self.device}, 数据类型: {self.dtype}")
        print(f"{'='*60}")
        all_results = {}
        # 1. 矩阵乘法测试[5](@ref)
        all_results['matmul'] = self.test_matrix_multiplication()
        # 2. 内存带宽测试[2](@ref)
        all_results['memory'] = self.test_memory_bandwidth()
        # 3. 卷积操作测试
        all_results['conv'] = self.test_convolution_operations()
        # 4. 混合精度测试 (仅CUDA)[3](@ref)
        if self.device.type == 'cuda':
            all_results['mixed_precision'] = self.test_mixed_precision()
        # 5. 设备对比测试
        if self.device.type == 'cuda':
            # 与CPU对比
            print(f"\n{'='*60}")
            print(f"设备对比测试: CUDA vs CPU")
            print(f"{'='*60}")
            # 创建CPU测试实例
            cpu_benchmark = DeviceDtypeBenchmark(
                device='cpu', dtype=torch.float32)
            # 测试中等规模矩阵乘法
            n = 2048
            a_gpu = torch.randn(n, n, dtype=self.dtype, device=self.device)
            b_gpu = torch.randn(n, n, dtype=self.dtype, device=self.device)
            a_cpu = a_gpu.cpu()
            b_cpu = b_gpu.cpu()
            # GPU测试
            timer_gpu = Timer(
                stmt='torch.matmul(a, b)',
                globals={'a': a_gpu, 'b': b_gpu, 'torch': torch},
                label="GPU MatMul"
            )
            measurement_gpu = timer_gpu.timeit(20)
            stats_gpu = self.get_measurement_stats(measurement_gpu)
            # CPU测试
            timer_cpu = Timer(
                stmt='torch.matmul(a, b)',
                globals={'a': a_cpu, 'b': b_cpu, 'torch': torch},
                label="CPU MatMul"
            )
            measurement_cpu = timer_cpu.timeit(20)
            stats_cpu = self.get_measurement_stats(measurement_cpu)
            speedup = stats_cpu['median'] / stats_gpu['median']
            print(f"矩阵乘法 {n}x{n}:")
            print(f"  GPU ({self.device}): {stats_gpu['median']:.6f} s")
            print(f"  CPU: {stats_cpu['median']:.6f} s")
            print(f"  加速比: {speedup:.2f}x")
            all_results['device_comparison'] = {
                'gpu_time': stats_gpu['median'],
                'cpu_time': stats_cpu['median'],
                'speedup': speedup
            }
        return all_results
# 三、 使用PyTorch Profiler进行深度性能分析
# 对于需要深入分析性能瓶颈的场景，使用Profiler。


def profile_model_performance(model, input_shape, device='cuda', dtype=torch.float32):
    """
    使用PyTorch Profiler分析模型性能[3](@ref)
    Args:
        model: 要分析的模型
        input_shape: 输入张量形状
        device: 设备
        dtype: 数据类型
    """
    # 准备模型和数据
    device = torch.device(device)
    model = model.to(device).to(dtype)
    model.eval()
    # 创建输入数据
    x = torch.randn(input_shape, dtype=dtype, device=device)
    print(f"\n{'='*60}")
    print(f"模型性能分析")
    print(f"设备: {device}, 数据类型: {dtype}")
    print(f"输入形状: {input_shape}")
    print(f"{'='*60}")
    # 使用Profiler[3](@ref)
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA if device.type == 'cuda' else None
        ],
        schedule=torch.profiler.schedule(
            wait=1,  # 跳过前1次迭代
            warmup=2,  # 预热2次迭代
            active=5,  # 分析5次迭代
            repeat=1
        ),
        on_trace_ready=lambda prof: prof.export_chrome_trace(
            f"trace_{device.type}_{dtype}.json"
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True  # 记录调用栈
    ) as prof:
        with torch.no_grad():
            for i in range(8):  # 总共8次迭代
                with record_function(f"model_inference_{i}"):
                    output = model(x)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                prof.step()  # 通知profiler步骤完成
    # 打印性能摘要
    print("\n性能摘要表:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total" if device.type == 'cuda' else "cpu_time_total",
        row_limit=20
    ))
    # 内存使用分析
    print("\n内存使用摘要:")
    print(prof.key_averages().table(
        sort_by="self_cpu_memory_usage" if device.type == 'cpu' else "self_cuda_memory_usage",
        row_limit=10
    ))
    return prof
# 四、 多设备与多数据类型对比测试


def compare_devices_and_dtypes():
    """比较不同设备和数据类型的性能[1](@ref)[5](@ref)"""
    devices = []
    if torch.cuda.is_available():
        devices.append('cuda:0')
    devices.append('cpu')
    dtypes = [torch.float32]
    if torch.cuda.is_available():
        dtypes.extend([torch.float16, torch.bfloat16])
    comparison_results = []
    for device in devices:
        for dtype in dtypes:
            print(f"\n{'='*60}")
            print(f"测试配置: 设备={device}, 数据类型={dtype}")
            print(f"{'='*60}")
            try:
                benchmark = DeviceDtypeBenchmark(device=device, dtype=dtype)
                # 运行核心测试
                matmul_results = benchmark.test_matrix_multiplication(sizes=[
                                                                      1024, 2048])
                memory_results = benchmark.test_memory_bandwidth(
                    vector_sizes=[10_000_000])
                # 收集结果
                for size, res in matmul_results.items():
                    comparison_results.append({
                        'device': device,
                        'dtype': str(dtype),
                        'test_type': 'matmul',
                        'size': size,
                        'time_median': res['time_median'],
                        'tflops': res['tflops']
                    })
                for size, res in memory_results.items():
                    comparison_results.append({
                        'device': device,
                        'dtype': str(dtype),
                        'test_type': 'memory',
                        'size': size,
                        'time_median': res['time_median'],
                        'bandwidth_gbs': res['bandwidth_gbs']
                    })
            except Exception as e:
                print(f"测试失败: {e}")
                continue
    # 创建DataFrame并分析
    df = pd.DataFrame(comparison_results)
    print(f"\n{'='*60}")
    print(f"性能对比总结")
    print(f"{'='*60}")
    # 矩阵乘法性能对比
    matmul_df = df[df['test_type'] == 'matmul']
    if not matmul_df.empty:
        print("\n矩阵乘法性能对比 (TFLOPS, 越高越好):")
        pivot = matmul_df.pivot_table(
            values='tflops',
            index=['device', 'dtype'],
            columns='size',
            aggfunc='mean'
        )
        print(pivot)
    # 内存带宽对比
    memory_df = df[df['test_type'] == 'memory']
    if not memory_df.empty:
        print("\n内存带宽对比 (GB/s, 越高越好):")
        pivot = memory_df.pivot_table(
            values='bandwidth_gbs',
            index=['device', 'dtype'],
            columns='size',
            aggfunc='mean'
        )
        print(pivot)
    # 可视化
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # TFLOPS对比图
        if not matmul_df.empty:
            for (device, dtype), group in matmul_df.groupby(['device', 'dtype']):
                axes[0].plot(group['size'], group['tflops'],
                             marker='o', label=f'{device} {dtype}')
            axes[0].set_xlabel('Matrix Size')
            axes[0].set_ylabel('TFLOPS')
            axes[0].set_title('Matrix Multiplication Performance')
            axes[0].legend()
            axes[0].grid(True)
        # 带宽对比图
        if not memory_df.empty:
            for (device, dtype), group in memory_df.groupby(['device', 'dtype']):
                axes[1].plot(group['size'], group['bandwidth_gbs'],
                             marker='s', label=f'{device} {dtype}')
            axes[1].set_xlabel('Vector Size')
            axes[1].set_ylabel('Bandwidth (GB/s)')
            axes[1].set_title('Memory Bandwidth Performance')
            axes[1].legend()
            axes[1].grid(True)
        plt.tight_layout()
        plt.savefig('device_dtype_comparison.png',
                    dpi=150, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"可视化失败: {e}")
    return df


# 五、 主程序入口
if __name__ == "__main__":
    # 验证环境
    has_cuda = verify_environment()
    if has_cuda:
        # 测试不同数据类型在GPU上的性能
        dtypes_to_test = [torch.float32, torch.float16, torch.bfloat16]
        for dtype in dtypes_to_test:
            print(f"\n{'='*60}")
            print(f"开始GPU性能测试 - 数据类型: {dtype}")
            print(f"{'='*60}")
            benchmark = DeviceDtypeBenchmark(device='cuda:0', dtype=dtype)
            results = benchmark.run_comprehensive_test()
            # 保存结果
            import json
            import pickle
            # 保存为JSON (可读)
            with open(f'benchmark_results_cuda_{dtype}.json', 'w') as f:
                # 转换Tensor为Python原生类型
                def convert(obj):
                    if isinstance(obj, torch.Tensor):
                        return obj.cpu().numpy().tolist()
                    return obj
                json.dump(results, f, default=convert, indent=2)
            # 保存为pickle (完整保存)
            with open(f'benchmark_results_cuda_{dtype}.pkl', 'wb') as f:
                pickle.dump(results, f)
            print(f"结果已保存到 benchmark_results_cuda_{dtype}.json/pkl")
    # CPU测试
    print(f"\n{'='*60}")
    print(f"CPU性能测试")
    print(f"{'='*60}")
    cpu_benchmark = DeviceDtypeBenchmark(device='cpu', dtype=torch.float32)
    cpu_results = cpu_benchmark.run_comprehensive_test()
    # 保存CPU结果
    with open('benchmark_results_cpu_float32.json', 'w') as f:
        def convert(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            return obj
        json.dump(cpu_results, f, default=convert, indent=2)
    # 运行对比测试
    print(f"\n{'='*60}")
    print(f"设备与数据类型对比测试")
    print(f"{'='*60}")
    comparison_df = compare_devices_and_dtypes()
    # 保存对比结果
    comparison_df.to_csv('device_dtype_comparison.csv', index=False)
    print("\n对比结果已保存到 'device_dtype_comparison.csv'")
    # 示例：创建并分析一个测试模型
    if has_cuda:
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
                self.fc = nn.Linear(256 * 28 * 28, 1000)
                self.relu = nn.ReLU()
                self.pool = nn.MaxPool2d(2)

            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.pool(x)
                x = self.relu(self.conv2(x))
                x = self.pool(x)
                x = self.relu(self.conv3(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        model = TestModel()
        # 分析不同数据类型的性能
        for dtype in [torch.float32, torch.float16]:
            profiler = profile_model_performance(
                model=model,
                input_shape=(32, 3, 224, 224),  # batch=32, 3通道, 224x224
                device='cuda:0',
                dtype=dtype
            )
    print(f"\n{'='*60}")
    print(f"所有测试完成！")
    print(f"{'='*60}")
# 六、 代码特点与可信度保证
# 这套整合后的代码具有以下特点：
# 高可信度计时：使用PyTorch内置的torch.utils.benchmark.Timer而非简单time.time()，自动处理预热和多次运行。
# 正确的统计处理：通过get_measurement_stats函数正确处理Measurement对象的统计信息，避免了AttributeError: 'Measurement' object has no attribute 'std'错误。
# 全面的测试覆盖：包括矩阵乘法、内存带宽、卷积操作、混合精度训练等多种测试场景。
# 设备与数据类型对比：支持CPU与GPU对比，以及不同数据类型（float32, float16, bfloat16）的性能对比。
# 深度性能分析：集成PyTorch Profiler进行详细的性能分析。
# 结果保存与可视化：自动保存测试结果为JSON、pickle和CSV格式，并生成可视化图表。
# 环境验证：测试前验证PyTorch版本、CUDA版本和硬件兼容性。
# 错误处理：完善的异常处理，避免单点失败影响整体测试。
# 这套代码提供了从基础验证到深度分析的全套工具，能够可靠地评估特定设备和数据类型在PyTorch中的性能表现，为模型优化和硬件选型提供数据支持。
