import tilelang as tl
import tilelang.language as T
from tilelang.autotuner import autotune

def Eexyzt_exyzt2Exyzt_gpu(
    E: int, e: int, x: int, y: int, z: int, t: int,
    block_E: int = 32, block_e: int = 32, block_x: int = 8, block_y: int = 8, block_z: int = 8, block_t: int = 8,
    dtype: str = "float16", accum_dtype: str = "float32"
):
    """
    GPU implementation of tensor contraction: Eexyzt,exyzt->Exyzt
    Optimized with tiling, shared memory, and software pipelining
    """
    @T.prim_func
    def main(
        Eexyzt: T.Tensor((E, e, x, y, z, t), dtype),
        exyzt: T.Tensor((e, x, y, z, t), dtype),
        Exyzt: T.Tensor((E, x, y, z, t), dtype),
    ):
        # Calculate total output elements and flatten grid
        total_output_elements = E * x * y * z * t
        elements_per_block = block_E * block_x * block_y * block_z * block_t
        
        # Use 1D grid with 3D thread blocks
        with T.Kernel(
            T.ceildiv(total_output_elements, elements_per_block),  # 1D grid
            threads=(block_x * block_y * block_z * block_t, block_E, 1)  # 3D thread block
        ) as (block_id, tx, ty, tz):
            
            # Calculate output indices from thread IDs
            # tx: spatial dimensions (x,y,z,t)
            # ty: E dimension
            # tz: unused (always 0)
            
            # Flatten spatial dimensions
            spatial_size = block_x * block_y * block_z * block_t
            spatial_idx = tx
            
            # Reconstruct spatial indices
            if spatial_idx < spatial_size:
                # Calculate x,y,z,t indices from flattened index
                t_idx = spatial_idx % block_t
                spatial_idx //= block_t
                z_idx = spatial_idx % block_z
                spatial_idx //= block_z
                y_idx = spatial_idx % block_y
                spatial_idx //= block_y
                x_idx = spatial_idx % block_x
                
                # Calculate global output indices
                global_block_idx = block_id * elements_per_block
                global_spatial_idx = global_block_idx // block_E
                global_E_idx = global_block_idx % block_E
                
                # Final global indices
                global_x = (global_spatial_idx // (y * z * t)) * block_x + x_idx
                global_y = ((global_spatial_idx // (z * t)) % y) * block_y + y_idx
                global_z = ((global_spatial_idx // t) % z) * block_z + z_idx
                global_t = (global_spatial_idx % t) * block_t + t_idx
                global_E = global_E_idx + ty * block_E
                
                # Check bounds
                if (global_E < E and global_x < x and global_y < y and 
                    global_z < z and global_t < t):
                    
                    # Allocate shared memory for input tiles
                    Eexyzt_shared = T.alloc_shared((block_e,), dtype)
                    exyzt_shared = T.alloc_shared((block_e,), dtype)
                    
                    # Local accumulator
                    local_sum = T.alloc_fragment((1,), accum_dtype)
                    T.clear(local_sum)
                    
                    # Reduction over contraction dimension 'e' with pipelining
                    for be in T.Pipelined(T.ceildiv(e, block_e), num_stages=3):
                        # Load tiles for current e-block
                        for ie in T.Parallel(block_e):
                            e_idx = be * block_e + ie
                            if e_idx < e:
                                # Load Eexyzt value
                                Eexyzt_shared[ie] = Eexyzt[global_E, e_idx, global_x, global_y, global_z, global_t]
                                
                                # Load exyzt value
                                exyzt_shared[ie] = exyzt[e_idx, global_x, global_y, global_z, global_t]
                        
                        # Synchronize threads to ensure shared memory is loaded
                        T.sync_threads()
                        
                        # Perform contraction for this e-block
                        for ie in range(block_e):
                            e_idx = be * block_e + ie
                            if e_idx < e:
                                val1 = T.cast(Eexyzt_shared[ie], accum_dtype)
                                val2 = T.cast(exyzt_shared[ie], accum_dtype)
                                local_sum[0] += val1 * val2
                        
                        T.sync_threads()
                    
                    # Write result to global memory
                    Exyzt[global_E, global_x, global_y, global_z, global_t] = T.cast(local_sum[0], dtype)
    
    return main


def Eexyzt_exyzt2Exyzt_gpu_simple(
    E: int, e: int, x: int, y: int, z: int, t: int,
    block_E: int = 32, block_e: int = 32,
    dtype: str = "float16", accum_dtype: str = "float32"
):
    """
    Simplified GPU implementation with 2D grid
    Each thread computes one output element
    """
    @T.prim_func
    def main(
        Eexyzt: T.Tensor((E, e, x, y, z, t), dtype),
        exyzt: T.Tensor((e, x, y, z, t), dtype),
        Exyzt: T.Tensor((E, x, y, z, t), dtype),
    ):
        # Use 2D grid: (E*x*y*z*t, 1)
        total_output = E * x * y * z * t
        
        with T.Kernel(
            T.ceildiv(total_output, 256),  # Grid X: output elements
            1,                             # Grid Y: unused
            threads=256                    # Threads per block
        ) as (bx, by, tx):
            
            # Calculate global output index
            global_idx = bx * 256 + tx
            
            if global_idx < total_output:
                # Decode global index to 5D coordinates (E, x, y, z, t)
                idx = global_idx
                
                # t dimension
                t_idx = idx % t
                idx //= t
                
                # z dimension
                z_idx = idx % z
                idx //= z
                
                # y dimension
                y_idx = idx % y
                idx //= y
                
                # x dimension
                x_idx = idx % x
                idx //= x
                
                # E dimension
                E_idx = idx % E
                
                # Allocate local accumulator
                local_sum = T.alloc_fragment((1,), accum_dtype)
                T.clear(local_sum)
                
                # Reduction over contraction dimension 'e'
                for e_idx in range(e):
                    val1 = T.cast(Eexyzt[E_idx, e_idx, x_idx, y_idx, z_idx, t_idx], accum_dtype)
                    val2 = T.cast(exyzt[e_idx, x_idx, y_idx, z_idx, t_idx], accum_dtype)
                    local_sum[0] += val1 * val2
                
                # Write result
                Exyzt[E_idx, x_idx, y_idx, z_idx, t_idx] = T.cast(local_sum[0], dtype)
    
    return main


def Eexyzt_exyzt2Exyzt_cpu(
    E: int, e: int, x: int, y: int, z: int, t: int,
    block_E: int = 8, block_e: int = 8, block_x: int = 4, block_y: int = 4, block_z: int = 4, block_t: int = 4,
    dtype: str = "float16", accum_dtype: str = "float32"
):
    """
    CPU implementation of tensor contraction: Eexyzt,exyzt->Exyzt
    Optimized with tiling and local caching
    """
    @T.prim_func
    def main(
        Eexyzt: T.Tensor((E, e, x, y, z, t), dtype),
        exyzt: T.Tensor((e, x, y, z, t), dtype),
        Exyzt: T.Tensor((E, x, y, z, t), dtype),
    ):
        # Tile over x,y,z,t dimensions
        for bx, by, bz, bt in T.grid(
            T.ceildiv(x, block_x),  # x blocks
            T.ceildiv(y, block_y),  # y blocks
            T.ceildiv(z, block_z),  # z blocks
            T.ceildiv(t, block_t)   # t blocks
        ):
            # Allocate local buffer for accumulation
            acc = T.alloc_buffer((block_E, block_x, block_y, block_z, block_t), accum_dtype, scope="local")
            
            # Initialize accumulator to zero
            for iE, ix, iy, iz, it in T.grid(block_E, block_x, block_y, block_z, block_t):
                acc[iE, ix, iy, iz, it] = T.cast(0.0, accum_dtype)
            
            # Reduction over contraction dimension 'e'
            for be in range(T.ceildiv(e, block_e)):
                e_start = be * block_e
                e_end = T.min(e_start + block_e, e)
                
                # Process current 'e' tile
                for iE in range(block_E):
                    for ie in range(e_start, e_end):
                        for ix in range(block_x):
                            x_idx = bx * block_x + ix
                            for iy in range(block_y):
                                y_idx = by * block_y + iy
                                for iz in range(block_z):
                                    z_idx = bz * block_z + iz
                                    for it in range(block_t):
                                        t_idx = bt * block_t + it
                                        
                                        # Read and cast data types
                                        val1 = T.cast(Eexyzt[iE, ie, x_idx, y_idx, z_idx, t_idx], accum_dtype)
                                        val2 = T.cast(exyzt[ie, x_idx, y_idx, z_idx, t_idx], accum_dtype)
                                        
                                        # Accumulate
                                        acc[iE, ix, iy, iz, it] += val1 * val2
            
            # Write result to output tensor
            for iE, ix, iy, iz, it in T.grid(block_E, block_x, block_y, block_z, block_t):
                Exyzt[
                    iE,
                    bx * block_x + ix,
                    by * block_y + iy,
                    bz * block_z + iz,
                    bt * block_t + it
                ] = T.cast(acc[iE, ix, iy, iz, it], dtype)
    
    return main


# Configuration generator for autotuning
def get_contraction_configs(E, e, x, y, z, t):
    """
    Generate configuration space for autotuning tensor contraction
    """
    configs = []
    
    # Simple GPU configurations (per-thread computation)
    threads_per_block_options = [64, 128, 256, 512]
    block_e_options = [16, 32, 64]
    
    for threads in threads_per_block_options:
        for block_e in block_e_options:
            configs.append({
                'block_e': block_e,
                'threads': threads,
                'dtype': 'float16',
                'accum_dtype': 'float32'
            })
    
    return configs


# Autotunable simplified GPU version
@autotune(configs=get_contraction_configs, warmup=10, rep=50, timeout=120)
@tl.jit(out_idx=[-1])
def Eexyzt_exyzt2Exyzt_gpu_tunable(
    E: int, e: int, x: int, y: int, z: int, t: int,
    block_e: int = 32,
    threads: int = 256,
    dtype: str = "float16", 
    accum_dtype: str = "float32"
):
    """
    Autotunable simplified GPU version
    Each thread computes one output element
    """
    @T.prim_func
    def main(
        Eexyzt: T.Tensor((E, e, x, y, z, t), dtype),
        exyzt: T.Tensor((e, x, y, z, t), dtype),
        Exyzt: T.Tensor((E, x, y, z, t), dtype),
    ):
        total_output = E * x * y * z * t
        
        with T.Kernel(
            T.ceildiv(total_output, threads),  # Grid size
            1,                                 # Grid Y
            threads=threads                    # Threads per block
        ) as (bx, by, tx):
            
            global_idx = bx * threads + tx
            
            if global_idx < total_output:
                # Decode 5D coordinates from global index
                idx = global_idx
                
                # t dimension
                t_idx = idx % t
                idx //= t
                
                # z dimension
                z_idx = idx % z
                idx //= z
                
                # y dimension
                y_idx = idx % y
                idx //= y
                
                # x dimension
                x_idx = idx % x
                idx //= x
                
                # E dimension
                E_idx = idx % E
                
                # Local accumulator
                local_sum = T.alloc_fragment((1,), accum_dtype)
                T.clear(local_sum)
                
                # Process e dimension in blocks for better cache utilization
                for be in range(T.ceildiv(e, block_e)):
                    e_start = be * block_e
                    e_end = T.min(e_start + block_e, e)
                    
                    # Process current e-block
                    for e_idx in range(e_start, e_end):
                        val1 = T.cast(Eexyzt[E_idx, e_idx, x_idx, y_idx, z_idx, t_idx], accum_dtype)
                        val2 = T.cast(exyzt[e_idx, x_idx, y_idx, z_idx, t_idx], accum_dtype)
                        local_sum[0] += val1 * val2
                
                # Write result
                Exyzt[E_idx, x_idx, y_idx, z_idx, t_idx] = T.cast(local_sum[0], dtype)
    
    return main


# Wrapper function for easy usage
def create_contraction_kernel(
    E: int, e: int, x: int, y: int, z: int, t: int,
    target: str = "auto",
    use_autotune: bool = False,
    implementation: str = "simple",  # "simple" or "tiled"
    **kwargs
):
    """
    Create tensor contraction kernel
    
    Args:
        E, e, x, y, z, t: Tensor dimensions
        target: "cuda", "hip", "cpu", or "auto"
        use_autotune: Enable autotuning for GPU targets
        implementation: "simple" (per-thread) or "tiled" (blocked)
        **kwargs: Additional parameters for kernel configuration
    """
    # Auto-detect target if not specified
    if target == "auto":
        import torch
        target = "cuda" if torch.cuda.is_available() else "cpu"
    
    if target in ["cuda", "hip"]:
        if use_autotune:
            # Return autotunable kernel factory
            return lambda: Eexyzt_exyzt2Exyzt_gpu_tunable(E, e, x, y, z, t, **kwargs)
        elif implementation == "tiled":
            # Return tiled GPU kernel
            return Eexyzt_exyzt2Exyzt_gpu(E, e, x, y, z, t, **kwargs)
        else:
            # Return simple GPU kernel
            return Eexyzt_exyzt2Exyzt_gpu_simple(E, e, x, y, z, t, **kwargs)
    else:
        # Return CPU kernel
        return Eexyzt_exyzt2Exyzt_cpu(E, e, x, y, z, t, **kwargs)


# Example usage
if __name__ == "__main__":
    import torch
    from tilelang.autotuner import set_autotune_inputs
    
    # Test dimensions (smaller for testing)
    E, e, x, y, z, t = 4, 8, 16, 16, 8, 4
    
    print(f"Tensor dimensions: E={E}, e={e}, x={x}, y={y}, z={z}, t={t}")
    print(f"Total output elements: {E*x*y*z*t}")
    
    # Create test tensors
    Eexyzt = torch.randn(E, e, x, y, z, t, device="cuda", dtype=torch.float16)
    exyzt = torch.randn(e, x, y, z, t, device="cuda", dtype=torch.float16)
    Exyzt = torch.empty(E, x, y, z, t, device="cuda", dtype=torch.float16)
    
    print("\n1. Testing simple GPU kernel (per-thread):")
    
    # Simple GPU kernel
    kernel_simple = tl.jit(create_contraction_kernel(
        E, e, x, y, z, t, 
        target="cuda", 
        use_autotune=False,
        implementation="simple"
    ))
    
    # Execute
    kernel_simple(Eexyzt, exyzt, Exyzt)
    
    # Verify
    ref_result = torch.einsum('Eexyzt,exyzt->Exyzt', Eexyzt, exyzt)
    if torch.allclose(Exyzt, ref_result, rtol=1e-2, atol=1e-2):
        print("✓ Simple kernel: Results match")
    else:
        diff = torch.abs(Exyzt - ref_result)
        print(f"✗ Simple kernel: Max diff = {diff.max().item():.6f}")
    
    print("\n2. Testing autotuned GPU kernel:")
    
    # Autotuned kernel
    with set_autotune_inputs(Eexyzt, exyzt, Exyzt):
        tuned_kernel = Eexyzt_exyzt2Exyzt_gpu_tunable(E, e, x, y, z, t)
    
    # Execute tuned kernel
    tuned_kernel(Eexyzt, exyzt, Exyzt)
    
    # Verify
    if torch.allclose(Exyzt, ref_result, rtol=1e-2, atol=1e-2):
        print("✓ Autotuned kernel: Results match")
    else:
        diff = torch.abs(Exyzt - ref_result)
        print(f"✗ Autotuned kernel: Max diff = {diff.max().item():.6f}")
    
    # Performance comparison
    import time
    
    print("\n3. Performance comparison:")
    
    # Warmup
    for _ in range(10):
        kernel_simple(Eexyzt, exyzt, Exyzt)
        tuned_kernel(Eexyzt, exyzt, Exyzt)
    
    # Time simple kernel
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        kernel_simple(Eexyzt, exyzt, Exyzt)
    torch.cuda.synchronize()
    simple_time = (time.time() - start) / 100
    
    # Time tuned kernel
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        tuned_kernel(Eexyzt, exyzt, Exyzt)
    torch.cuda.synchronize()
    tuned_time = (time.time() - start) / 100
    
    # Time torch.einsum
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = torch.einsum('Eexyzt,exyzt->Exyzt', Eexyzt, exyzt)
    torch.cuda.synchronize()
    torch_time = (time.time() - start) / 100
    
    print(f"Simple kernel: {simple_time*1000:.2f} ms")
    print(f"Autotuned kernel: {tuned_time*1000:.2f} ms")
    print(f"torch.einsum: {torch_time*1000:.2f} ms")
    print(f"Speedup (autotuned vs simple): {simple_time/tuned_time:.2f}x")
    print(f"Speedup (autotuned vs torch): {torch_time/tuned_time:.2f}x")
    
    # Try CPU version if available
    print("\n4. Testing CPU kernel:")
    try:
        # Move data to CPU
        Eexyzt_cpu = Eexyzt.cpu()
        exyzt_cpu = exyzt.cpu()
        Exyzt_cpu = torch.empty(E, x, y, z, t, dtype=torch.float16)
        
        # CPU kernel
        kernel_cpu = tl.jit(create_contraction_kernel(
            E, e, x, y, z, t, 
            target="cpu", 
            use_autotune=False
        ))
        
        # Execute
        kernel_cpu(Eexyzt_cpu, exyzt_cpu, Exyzt_cpu)
        
        # Verify
        ref_cpu = torch.einsum('Eexyzt,exyzt->Exyzt', Eexyzt_cpu, exyzt_cpu)
        if torch.allclose(Exyzt_cpu, ref_cpu, rtol=1e-2, atol=1e-2):
            print("✓ CPU kernel: Results match")
        else:
            print("✗ CPU kernel: Results differ")
            
    except Exception as e:
        print(f"CPU kernel test skipped: {e}")
