"""
TileLang与PyTorch复数einsum性能对比
基于TileLang官方文档示例语法实现
支持：
1. 可配置矩阵尺寸(M,N,K)
2. 可配置数据类型(complex64/complex128)
3. 可配置后端(cuda/cpu)
4. 完整正确性验证
5. 性能测量与加速比计算
"""

import torch
import time
import numpy as np
import tilelang as T
from typing import Optional

def create_complex_tensors(device: str, dtype: str, M: int, N: int, K: int):
    """创建复数张量"""
    torch.manual_seed(42)
    if dtype == 'complex64':
        torch_dtype = torch.complex64
        tile_dtype = T.float32
    elif dtype == 'complex128':
        torch_dtype = torch.complex128
        tile_dtype = T.float64
    else:
        raise ValueError(f"不支持的数据类型: {dtype}")
    
    # 创建PyTorch复数张量
    A_torch = torch.randn(M, K, device=device, dtype=torch_dtype)
    B_torch = torch.randn(K, N, device=device, dtype=torch_dtype)
    
    # 分解为实部和虚部（TileLang需要）
    A_real = A_torch.real
    A_imag = A_torch.imag
    B_real = B_torch.real
    B_imag = B_torch.imag
    
    return A_torch, B_torch, A_real, A_imag, B_real, B_imag, torch_dtype, tile_dtype

def complex_einsum_torch(A: torch.Tensor, B: torch.Tensor, pattern: str = 'ik,kj->ij'):
    """PyTorch baseline: 复数矩阵乘法"""
    return torch.einsum(pattern, A, B)

# 方法1: 使用官方Quick Start风格（推荐）
@T.jit
def complex_matmul_tilelang_method1(
    A,
    B,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 64,
    dtype: T.dtype = T.float32,
    accum_dtype: T.dtype = T.float32,
):
    """
    TileLang实现复数矩阵乘法 - 方法1 (Quick Start风格)
    复数乘法: (A_real + i*A_imag) * (B_real + i*B_imag)
    """
    # 声明编译时常量
    M, N, K = T.const('M, N, K')
    
    # 类型注解 (如文档所示)
    A_real: T.Tensor[[M, K], dtype]
    A_imag: T.Tensor[[M, K], dtype]
    B_real: T.Tensor[[K, N], dtype]
    B_imag: T.Tensor[[K, N], dtype]
    
    # 分配输出张量
    C_real = T.empty([M, N], dtype)
    C_imag = T.empty([M, N], dtype)
    
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        # 共享内存分配（按文档图16实现）
        A_real_shared = T.alloc_shared((block_M, block_K), dtype)
        A_imag_shared = T.alloc_shared((block_M, block_K), dtype)
        B_real_shared = T.alloc_shared((block_K, block_N), dtype)
        B_imag_shared = T.alloc_shared((block_K, block_N), dtype)
        
        # 寄存器累加器（四个实数部分）
        acc_rr = T.alloc_fragment((block_M, block_N), accum_dtype)  # A_real*B_real
        acc_ri = T.alloc_fragment((block_M, block_N), accum_dtype)  # A_real*B_imag
        acc_ir = T.alloc_fragment((block_M, block_N), accum_dtype)  # A_imag*B_real
        acc_ii = T.alloc_fragment((block_M, block_N), accum_dtype)  # A_imag*B_imag
        
        # 初始化累加器（按文档B.1示例）
        T.clear(acc_rr)
        T.clear(acc_ri)
        T.clear(acc_ir)
        T.clear(acc_ii)
        
        # 主循环，使用流水线（按文档5.2节说明）
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            # 从全局内存复制到共享内存
            T.copy(A_real[by * block_M, ko * block_K], A_real_shared)
            T.copy(A_imag[by * block_M, ko * block_K], A_imag_shared)
            T.copy(B_real[ko * block_K, bx * block_N], B_real_shared)
            T.copy(B_imag[ko * block_K, bx * block_N], B_imag_shared)
            
            # 执行四个实数GEMM（形成复数乘法）
            T.gemm(A_real_shared, B_real_shared, acc_rr)  # A_real * B_real
            T.gemm(A_real_shared, B_imag_shared, acc_ri)  # A_real * B_imag
            T.gemm(A_imag_shared, B_real_shared, acc_ir)  # A_imag * B_real
            T.gemm(A_imag_shared, B_imag_shared, acc_ii)  # A_imag * B_imag
        
        # 组合复数结果（实部和虚部）
        C_real_local = T.alloc_fragment((block_M, block_N), dtype)
        C_imag_local = T.alloc_fragment((block_M, block_N), dtype)
        
        # 并行计算结果（使用T.Parallel）
        for i, j in T.Parallel(block_M, block_N):
            # C_real = acc_rr - acc_ii
            C_real_local[i, j] = T.cast(acc_rr[i, j] - acc_ii[i, j], dtype)
            # C_imag = acc_ri + acc_ir
            C_imag_local[i, j] = T.cast(acc_ri[i, j] + acc_ir[i, j], dtype)
        
        # 写回全局内存（按文档示例）
        T.copy(C_real_local, C_real[by * block_M, bx * block_N])
        T.copy(C_imag_local, C_imag[by * block_M, bx * block_N])
    
    return C_real, C_imag

# 方法2: 使用附录B.1的更简洁风格
@T.jit
def complex_matmul_tilelang_method2(
    A_real: T.Tensor,
    A_imag: T.Tensor,
    B_real: T.Tensor,
    B_imag: T.Tensor,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 64,
):
    """
    TileLang实现复数矩阵乘法 - 方法2 (附录B.1风格)
    更简洁的接口，输入四个实数张量
    """
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        # 获取数据类型
        dtype = A_real.dtype
        
        # 共享内存分配
        A_real_shared = T.alloc_shared((block_M, block_K), dtype)
        A_imag_shared = T.alloc_shared((block_M, block_K), dtype)
        B_real_shared = T.alloc_shared((block_K, block_N), dtype)
        B_imag_shared = T.alloc_shared((block_K, block_N), dtype)
        
        # 寄存器累加器
        acc_rr = T.alloc_fragment((block_M, block_N), dtype)
        acc_ri = T.alloc_fragment((block_M, block_N), dtype)
        acc_ir = T.alloc_fragment((block_M, block_N), dtype)
        acc_ii = T.alloc_fragment((block_M, block_N), dtype)
        
        T.clear(acc_rr)
        T.clear(acc_ri)
        T.clear(acc_ir)
        T.clear(acc_ii)
        
        # 流水线循环
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
            T.copy(A_real[by * block_M, ko * block_K], A_real_shared)
            T.copy(A_imag[by * block_M, ko * block_K], A_imag_shared)
            T.copy(B_real[ko * block_K, bx * block_N], B_real_shared)
            T.copy(B_imag[ko * block_K, bx * block_N], B_imag_shared)
            
            T.gemm(A_real_shared, B_real_shared, acc_rr)
            T.gemm(A_real_shared, B_imag_shared, acc_ri)
            T.gemm(A_imag_shared, B_real_shared, acc_ir)
            T.gemm(A_imag_shared, B_imag_shared, acc_ii)
        
        # 分配输出片段
        C_real_local = T.alloc_fragment((block_M, block_N), dtype)
        C_imag_local = T.alloc_fragment((block_M, block_N), dtype)
        
        # 计算结果
        for i, j in T.Parallel(block_M, block_N):
            C_real_local[i, j] = acc_rr[i, j] - acc_ii[i, j]
            C_imag_local[i, j] = acc_ri[i, j] + acc_ir[i, j]
        
        # 创建输出张量
        C_real = T.empty([M, N], dtype)
        C_imag = T.empty([M, N], dtype)
        
        T.copy(C_real_local, C_real[by * block_M, bx * block_N])
        T.copy(C_imag_local, C_imag[by * block_M, bx * block_N])
    
    return C_real, C_imag

def benchmark_complex_einsum(
    M: int = 1024,
    N: int = 1024,
    K: int = 1024,
    dtype: str = 'complex64',
    backend: str = 'cuda',
    method: int = 1,
    num_iterations: int = 50,
    verify: bool = True,
    verbose: bool = False,
):
    """
    主基准测试函数
    
    参数:
        M, N, K: 矩阵维度
        dtype: complex64/complex128
        backend: cuda/cpu
        method: 1 (QuickStart风格) 或 2 (附录B.1风格)
        num_iterations: 迭代次数
        verify: 是否验证正确性
        verbose: 是否输出详细信息
    """
    print("\n" + "="*70)
    print("TileLang vs PyTorch 复数Einsum性能对比")
    print("="*70)
    print(f"配置:")
    print(f"  矩阵维度: A[{M}, {K}] * B[{K}, {N}] = C[{M}, {N}]")
    print(f"  数据类型: {dtype}")
    print(f"  后端: {backend}")
    print(f"  TileLang方法: {method}")
    print(f"  迭代次数: {num_iterations}")
    
    # 检查后端可用性
    if backend == 'cuda':
        if not torch.cuda.is_available():
            print("警告: CUDA不可用，自动切换到CPU")
            backend = 'cpu'
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # 创建数据
    A_torch, B_torch, A_real, A_imag, B_real, B_imag, torch_dtype, tile_dtype = \
        create_complex_tensors(str(device), dtype, M, N, K)
    
    # 1. PyTorch基线测试
    print("\n" + "-"*40)
    print("1. PyTorch einsum基线测试")
    print("-"*40)
    
    # 预热
    _ = complex_einsum_torch(A_torch, B_torch)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 性能测试
    torch_times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        C_torch = complex_einsum_torch(A_torch, B_torch, 'ik,kj->ij')
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        torch_times.append((end - start) * 1000)  # ms
    
    torch_time_avg = np.mean(torch_times)
    torch_time_std = np.std(torch_times)
    flops = 2 * M * N * K * 4  # 复数乘加：实部4次乘加，虚部2次加法
    
    print(f"平均延迟: {torch_time_avg:.3f} ± {torch_time_std:.3f} ms")
    print(f"理论FLOPS: {flops / 1e12:.2f} T")
    print(f"实测吞吐: {(flops / (torch_time_avg / 1000)) / 1e12:.2f} TFLOPS")
    
    # 2. TileLang测试
    print("\n" + "-"*40)
    print("2. TileLang实现测试")
    print("-"*40)
    
    # 选择TileLang实现方法
    if method == 1:
        tilelang_func = complex_matmul_tilelang_method1
        # 准备输入参数（方法1使用单个复数参数）
        # 需要将实部和虚部组合成复合参数
        # 由于TileLang支持Python对象，我们可以传递元组
        # 或者使用不同的调用方式
        print("使用方法1: QuickStart风格")
        
        # 编译内核（按文档示例）
        try:
            if backend == 'cuda':
                target = "cuda"
            else:
                target = "cpu"
            
            # 按文档示例编译：使用占位符T.Tensor
            compile_args = {
                'M': M, 'N': N, 'K': K,
                'block_M': 128 if M >= 2048 else 64,
                'block_N': 128 if N >= 2048 else 64,
                'block_K': 64,
                'dtype': tile_dtype,
                'accum_dtype': T.float32,
            }
            
            if verbose:
                print(f"编译参数: {compile_args}")
            
            kernel = tilelang_func.compile(
                T.Tensor((M, K), tile_dtype),
                T.Tensor((M, K), tile_dtype),
                **{k: v for k, v in compile_args.items() if k not in ['M', 'N', 'K']}
            )
            
            # 创建复合输入（将实部和虚部分开传递）
            # 注意：这里需要根据实际的API调整
            print("注意: 复合输入处理可能需要调整...")
            
            # 预热
            C_real_tl, C_imag_tl = kernel(A_real, A_imag, B_real, B_imag)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # 性能测试
            tilelang_times = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                C_real_tl, C_imag_tl = kernel(A_real, A_imag, B_real, B_imag)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                tilelang_times.append((end - start) * 1000)
            
        except Exception as e:
            print(f"方法1编译错误: {e}")
            print("尝试使用方法2...")
            method = 2
    
    if method == 2:
        print("使用方法2: 附录B.1风格 (四个独立实数张量)")
        tilelang_func = complex_matmul_tilelang_method2
        
        try:
            # 编译内核
            compile_args = {
                'block_M': 128 if M >= 2048 else 64,
                'block_N': 128 if N >= 2048 else 64,
                'block_K': 64,
            }
            
            kernel = tilelang_func.compile(
                T.Tensor((M, K), tile_dtype),
                T.Tensor((M, K), tile_dtype),
                T.Tensor((K, N), tile_dtype),
                T.Tensor((K, N), tile_dtype),
                **compile_args
            )
            
            # 预热
            C_real_tl, C_imag_tl = kernel(A_real, A_imag, B_real, B_imag)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # 性能测试
            tilelang_times = []
            for _ in range(num_iterations):
                start = time.perf_counter()
                C_real_tl, C_imag_tl = kernel(A_real, A_imag, B_real, B_imag)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.perf_counter()
                tilelang_times.append((end - start) * 1000)
            
        except Exception as e:
            print(f"TileLang编译/执行错误: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # 计算TileLang性能指标
    tilelang_time_avg = np.mean(tilelang_times)
    tilelang_time_std = np.std(tilelang_times)
    
    print(f"平均延迟: {tilelang_time_avg:.3f} ± {tilelang_time_std:.3f} ms")
    print(f"实测吞吐: {(flops / (tilelang_time_avg / 1000)) / 1e12:.2f} TFLOPS")
    
    # 重构复数结果
    C_tilelang = torch.complex(C_real_tl, C_imag_tl)
    
    # 3. 性能分析
    print("\n" + "-"*40)
    print("3. 性能对比分析")
    print("-"*40)
    
    speedup = torch_time_avg / tilelang_time_avg
    efficiency = speedup
    
    print(f"加速比 (TileLang/PyTorch): {speedup:.2f}x")
    print(f"TileLang相对性能: {efficiency*100:.1f}%")
    
    # 4. 正确性验证
    if verify:
        print("\n" + "-"*40)
        print("4. 数值正确性验证")
        print("-"*40)
        
        # 计算误差指标
        abs_diff = torch.abs(C_torch - C_tilelang)
        rel_diff = abs_diff / (torch.abs(C_torch) + 1e-8)
        
        max_abs = torch.max(abs_diff).item()
        max_rel = torch.max(rel_diff).item()
        mean_abs = torch.mean(abs_diff).item()
        mean_rel = torch.mean(rel_diff).item()
        
        print(f"绝对误差 - 最大值: {max_abs:.2e}, 平均值: {mean_abs:.2e}")
        print(f"相对误差 - 最大值: {max_rel:.2e}, 平均值: {mean_rel:.2e}")
        
        # 设置误差阈值
        abs_threshold = 1e-3 if dtype == 'complex64' else 1e-6
        rel_threshold = 1e-3
        
        if max_abs < abs_threshold and max_rel < rel_threshold:
            print(f"✅ 正确性验证通过")
        else:
            print(f"⚠️  数值存在差异")
            if verbose:
                # 找出差异最大的位置
                bad_mask = abs_diff > abs_threshold
                bad_count = torch.sum(bad_mask).item()
                if bad_count > 0:
                    print(f"超过阈值{abs_threshold:.1e}的元素: {bad_count}个")
                    
                    # 显示前几个差异较大的值
                    bad_indices = torch.nonzero(bad_mask)
                    for i in range(min(5, len(bad_indices))):
                        idx = bad_indices[i]
                        print(f"位置{idx}: PyTorch={C_torch[idx[0], idx[1]]:.4f}, "
                              f"TileLang={C_tilelang[idx[0], idx[1]]:.4f}, "
                              f"差异={abs_diff[idx[0], idx[1]]:.2e}")
    
    print("\n" + "="*70)

def run_scaling_study():
    """运行尺寸缩放研究"""
    print("\n执行尺寸缩放研究...")
    
    sizes = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]
    
    for M, N, K in sizes:
        print(f"\n尺寸: {M}x{K} * {K}x{N}")
        try:
            benchmark_complex_einsum(
                M=M, N=N, K=K,
                dtype='complex64',
                backend='cuda',
                method=2,
                num_iterations=20,
                verify=False,
                verbose=False
            )
        except Exception as e:
            print(f"尺寸{M}测试失败: {e}")
            continue

def run_dtype_study():
    """运行数据类型研究"""
    print("\n执行数据类型研究...")
    
    for dtype in ['complex64', 'complex128']:
        print(f"\n数据类型: {dtype}")
        try:
            benchmark_complex_einsum(
                M=1024, N=1024, K=1024,
                dtype=dtype,
                backend='cuda',
                method=2,
                num_iterations=20,
                verify=True,
                verbose=False
            )
        except Exception as e:
            print(f"数据类型{dtype}测试失败: {e}")
            continue

if __name__ == "__main__":
    # 基本测试
    print("复数矩阵乘法性能对比测试")
    
    # 示例1: 基本测试 (CUDA, complex64)
    benchmark_complex_einsum(
        M=1024, N=1024, K=1024,
        dtype='complex64',
        backend='cuda',
        method=2,
        num_iterations=30,
        verify=True,
        verbose=True
    )
    
    # 示例2: CPU测试 (如果有TileLang CPU支持)
    # benchmark_complex_einsum(
    #     M=512, N=512, K=512,
    #     dtype='complex64',
    #     backend='cpu',
    #     method=2,
    #     num_iterations=10,
    #     verify=True,
    #     verbose=True
    # )
    
    # 示例3: 大尺寸测试
    # benchmark_complex_einsum(
    #     M=2048, N=2048, K=2048,
    #     dtype='complex64',
    #     backend='cuda',
    #     method=2,
    #     num_iterations=10,
    #     verify=False,
    #     verbose=True
    # )
    
    # 示例4: 运行完整研究 (可选)
    # run_scaling_study()
    # run_dtype_study()
    
    print("\n测试完成！")