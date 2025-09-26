import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from typing import Union


def find_free_port():
    """Find a free port for distributed training"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def setup(rank, world_size, use_cuda=True):
    """Initialize distributed environment with proper backend selection"""
    # Determine device first
    if use_cuda and torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(device)
        backend = "nccl"  # Try NCCL first for CUDA
    else:
        device = torch.device("cpu")
        backend = "gloo"  # Always use gloo for CPU

    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')

    # Try to find a free port if default port is taken
    if 'MASTER_PORT' not in os.environ:
        try:
            # Try default port first
            test_port = '29500'
            os.environ['MASTER_PORT'] = test_port
        except:
            # If default fails, find a free port
            free_port = find_free_port()
            os.environ['MASTER_PORT'] = str(free_port)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                rank=rank,
                world_size=world_size,
                timeout=torch.distributed.default_pg_timeout
            )
            if rank == 0:
                print(
                    f"[Rank {rank}] Distributed initialized with backend={backend}, device={device}")
                print(f"[Rank {rank}] Using port: {os.environ['MASTER_PORT']}")
            break
        except RuntimeError as e:
            if "EADDRINUSE" in str(e) or "address already in use" in str(e):
                # Port is in use, try a different port
                if attempt < max_retries - 1:
                    free_port = find_free_port()
                    os.environ['MASTER_PORT'] = str(free_port)
                    if rank == 0:
                        print(
                            f"[Rank {rank}] Port in use, trying port {free_port}...")
                    continue
                else:
                    raise RuntimeError(
                        f"Failed to find free port after {max_retries} attempts")
            elif backend == "nccl" and ("No backend type" in str(e) or "NCCL" in str(e)):
                # NCCL failed, fallback to gloo
                if rank == 0:
                    print(
                        f"[Rank {rank}] NCCL failed, falling back to gloo backend")
                backend = "gloo"
                continue
            else:
                raise e
        except Exception as e:
            if attempt < max_retries - 1 and backend == "nccl":
                # Try gloo as fallback
                if rank == 0:
                    print(
                        f"[Rank {rank}] Backend {backend} failed, trying gloo...")
                backend = "gloo"
                continue
            else:
                raise e

    return backend, device


def cleanup():
    """Clean up distributed environment"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_dtype_info(dtype):
    """Get information about the data type"""
    dtype_map = {
        torch.complex64: ("complex64", "single precision complex"),
        torch.complex128: ("complex128", "double precision complex"),
        torch.float16: ("float16", "half precision"),
        torch.bfloat16: ("bfloat16", "brain float"),
        torch.float32: ("float32", "single precision"),
        torch.float64: ("float64", "double precision"),
    }
    return dtype_map.get(dtype, (str(dtype), "unknown"))


def get_real_dtype(complex_dtype):
    """Get the corresponding real dtype for a complex dtype"""
    if complex_dtype == torch.complex64:
        return torch.float32
    elif complex_dtype == torch.complex128:
        return torch.float64
    else:
        return complex_dtype  # For non-complex types, return as is


def broadcast_complex(tensor: torch.Tensor, src: int = 0):
    """
    Broadcast a tensor that may be complex. If tensor is complex, broadcast its
    real-view (shape (...,2)) and then reconstruct complex tensor.
    """
    if not torch.is_complex(tensor):
        dist.broadcast(tensor, src=src)
        return tensor

    real_view = torch.view_as_real(tensor).contiguous()
    dist.broadcast(real_view, src=src)
    tensor.copy_(torch.view_as_complex(real_view))
    return tensor


def all_gather_complex(local: torch.Tensor):
    """
    All-gather a local tensor (possibly complex) across all ranks and return
    the concatenated full tensor along dim=0.
    """
    world_size = dist.get_world_size()
    if not torch.is_complex(local):
        gather_list = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gather_list, local)
        return torch.cat(gather_list, dim=0)

    real_local = torch.view_as_real(local).contiguous()
    gather_list = [torch.empty_like(real_local) for _ in range(world_size)]
    dist.all_gather(gather_list, real_local)
    real_full = torch.cat(gather_list, dim=0)
    return torch.view_as_complex(real_full)


def distributed_complex_matvec(A_local, x):
    """
    Optimized distributed complex matrix-vector multiplication
    Each rank holds a portion of matrix A and computes partial result
    """
    y_local = torch.matmul(A_local, x)
    return all_gather_complex(y_local)


def create_complex_test_matrix(N, dtype, device, condition_target=1e4, seed=42):
    """
    Create a well-conditioned Hermitian positive definite complex test matrix
    """
    torch.manual_seed(seed)

    # Get the corresponding real dtype
    real_dtype = get_real_dtype(dtype)

    # Create complex random matrix
    if condition_target < 1e2:
        # For well-conditioned matrices
        M_real = torch.randn(N, N, dtype=real_dtype, device=device)
        M_imag = torch.randn(N, N, dtype=real_dtype, device=device)
        M = torch.complex(M_real, M_imag)
        # Make it Hermitian positive definite: A = M^H M + I
        A = M.conj().T @ M + torch.eye(N, dtype=dtype, device=device) * 1.0
    elif condition_target < 1e6:
        # For moderately conditioned matrices
        M_real = torch.randn(N, N, dtype=real_dtype, device=device)
        M_imag = torch.randn(N, N, dtype=real_dtype, device=device)
        M = torch.complex(M_real, M_imag)
        A = M.conj().T @ M + torch.eye(N, dtype=dtype, device=device) * 0.1
    else:
        # For ill-conditioned matrices
        # Create complex random matrix
        M_real = torch.randn(N, N, dtype=real_dtype, device=device)
        M_imag = torch.randn(N, N, dtype=real_dtype, device=device)
        M = torch.complex(M_real, M_imag)

        # SVD to control condition number
        try:
            U, s, Vh = torch.linalg.svd(M)
            # Create singular values with desired condition number
            smax = 1.0
            smin = smax / condition_target
            s_controlled = torch.logspace(
                torch.log10(torch.tensor(smin, device=device)),
                torch.log10(torch.tensor(smax, device=device)),
                N, dtype=real_dtype, device=device
            )
            # Reconstruct matrix with controlled singular values
            A = U @ torch.diag(s_controlled) @ Vh
        except:
            # Fallback: use simpler method if SVD fails
            A = M.conj().T @ M

        # Make it Hermitian positive definite
        A = A.conj().T @ A + torch.eye(N, dtype=dtype, device=device) * 0.01

    return A


def complex_bicgstab_optimized(rank, world_size, N=4096, max_iter=1000, tol=1e-6,
                               use_cuda=True, dtype=torch.complex128, condition_target=1e4,
                               print_interval=50):
    """
    Optimized distributed BiCGStab solver for complex linear systems Ax = b

    Args:
        rank: Process rank
        world_size: Total number of processes
        N: Matrix size
        max_iter: Maximum iterations
        tol: Convergence tolerance
        use_cuda: Whether to use CUDA
        dtype: Complex data type (torch.complex64, torch.complex128)
        condition_target: Target condition number for test matrix
        print_interval: Interval for printing progress
    """
    N = int(N)
    max_iter = int(max_iter)
    backend, device = setup(rank, world_size, use_cuda=use_cuda)

    # Get data type information
    dtype_name, dtype_desc = get_dtype_info(dtype)
    if rank == 0:
        print(f"Using data type: {dtype_name} ({dtype_desc})")

    # Get the corresponding real dtype for epsilon calculation
    real_dtype = get_real_dtype(dtype)

    # Calculate matrix distribution
    rows_per_rank = (N + world_size - 1) // world_size
    start = rank * rows_per_rank
    end = min(start + rows_per_rank, N)

    start_time = time.time()

    # Initialize matrix and vectors on rank 0
    if rank == 0:
        print(
            f"Creating complex test matrix with target condition number: {condition_target:.1e}")
        A_full = create_complex_test_matrix(N, dtype, device, condition_target)
        # Create complex RHS vector
        b_real = torch.randn(N, 1, dtype=real_dtype, device=device)
        b_imag = torch.randn(N, 1, dtype=real_dtype, device=device)
        b_full = torch.complex(b_real, b_imag)

        # Compute actual condition number (convert to real for condition number)
        try:
            # Convert complex matrix to real representation for condition number
            A_real = torch.view_as_real(A_full).reshape(N, N * 2)
            actual_cond = torch.linalg.cond(A_real).item()
            print(f"Actual matrix condition number: {actual_cond:.2e}")
        except Exception as e:
            print(f"Could not compute condition number: {e}")
    else:
        A_full = torch.zeros(N, N, dtype=dtype, device=device)
        b_full = torch.zeros(N, 1, dtype=dtype, device=device)

    # Broadcast matrix and RHS vector to all ranks using complex-safe broadcast
    broadcast_complex(A_full, src=0)
    broadcast_complex(b_full, src=0)

    # Extract local portion of matrix
    A_local = A_full[start:end, :]

    # Pre-allocate all vectors to avoid memory allocation during iteration
    x = torch.zeros((N, 1), device=device, dtype=dtype)
    r = torch.empty_like(b_full)
    r_hat = torch.empty_like(b_full)
    v = torch.empty_like(b_full)
    p = torch.empty_like(b_full)
    s = torch.empty_like(b_full)
    t = torch.empty_like(b_full)

    # Initial residual: r0 = b - Ax0
    r.copy_(b_full)  # r = b (since x0 = 0)
    r_hat.copy_(r)   # Shadow residual

    # Algorithm variables (use appropriate epsilon for data type)
    # Adaptive epsilon based on precision
    eps = torch.finfo(real_dtype).eps * 1000

    rho_old = torch.tensor(1.0, device=device, dtype=dtype)
    alpha = torch.tensor(1.0, device=device, dtype=dtype)
    omega = torch.tensor(1.0, device=device, dtype=dtype)

    if rank == 0:
        setup_time = time.time() - start_time
        print(f"Setup time: {setup_time:.2f}s")
        print("Starting complex BiCGStab iterations...")
        initial_residual_norm = torch.norm(r).item()
        print(f"Initial residual norm: {initial_residual_norm:.6e}")
        print(f"Breakdown epsilon: {eps:.2e}")

    iter_start_time = time.time()

    for iteration in range(1, max_iter + 1):
        # Compute rho_new = <r_hat, r> (complex dot product with conjugate on first argument)
        rho_new = torch.vdot(r_hat.flatten(), r.flatten())

        # Check for breakdown
        if rho_new.abs() < eps:
            if rank == 0:
                print(
                    f"Complex BiCGStab breakdown: rho = {rho_new.abs().item():.2e}")
            break

        if iteration == 1:
            p.copy_(r)
        else:
            beta = (rho_new / rho_old) * (alpha / omega)
            # p = r + beta * (p - omega * v)
            p.copy_(r + beta * (p - omega * v))

        # Broadcast p to all ranks for matrix-vector multiplication
        broadcast_complex(p, src=0)

        # Compute v = A * p (distributed complex matvec)
        v = distributed_complex_matvec(A_local, p)

        # Compute alpha (complex division)
        denominator = torch.vdot(r_hat.flatten(), v.flatten())
        if denominator.abs() < eps:
            if rank == 0:
                print(
                    f"Complex BiCGStab breakdown: denominator = {denominator.abs().item():.2e}")
            break

        alpha = rho_new / denominator

        # s = r - alpha * v
        s.copy_(r - alpha * v)

        # Check convergence after first half-step
        s_norm = torch.norm(s)
        if s_norm < tol:
            x.add_(alpha * p)
            if rank == 0:
                print(f"Converged at iteration {iteration} (first half-step)")
            break

        # Broadcast s to all ranks for matrix-vector multiplication
        broadcast_complex(s, src=0)

        # Compute t = A * s (distributed complex matvec)
        t = distributed_complex_matvec(A_local, s)

        # Compute omega (complex division)
        numerator = torch.vdot(t.flatten(), s.flatten())
        denominator_t = torch.vdot(t.flatten(), t.flatten())

        if denominator_t.abs() < eps:
            if rank == 0:
                print(
                    f"Complex BiCGStab breakdown: ||t||^2 = {denominator_t.abs().item():.2e}")
            break

        omega = numerator / denominator_t

        # Update solution and residual
        # x = x + alpha * p + omega * s
        x.add_(alpha * p + omega * s)
        # r = s - omega * t
        r.copy_(s - omega * t)

        # Check convergence
        r_norm = torch.norm(r)
        if rank == 0 and iteration % print_interval == 0:
            iter_time = time.time() - iter_start_time
            rate = iteration / iter_time if iter_time > 0 else 0
            print(
                f"[Iter {iteration:4d}] Residual: {r_norm.item():.6e} | Rate: {rate:.1f} iter/s")

        if r_norm < tol:
            if rank == 0:
                print(f"Converged at iteration {iteration}")
            break

        # Check for breakdown
        if omega.abs() < eps:
            if rank == 0:
                print(
                    f"Complex BiCGStab breakdown: omega = {omega.abs().item():.2e}")
            break

        rho_old = rho_new

    total_time = time.time() - start_time
    iter_time = time.time() - iter_start_time

    # Verification on rank 0
    if rank == 0:
        # Move to CPU for verification if needed
        if device.type == 'cuda':
            x_final = x.cpu()
            A_cpu = A_full.cpu()
            b_cpu = b_full.cpu()
        else:
            x_final = x
            A_cpu = A_full
            b_cpu = b_full

        # Compute final residual
        Ax = A_cpu @ x_final
        final_residual_norm = torch.norm(Ax - b_cpu).item()

        # Compare with direct solve for accuracy
        try:
            solve_start = time.time()
            x_direct = torch.linalg.solve(A_cpu, b_cpu)
            solve_time = time.time() - solve_start
            solution_error = torch.norm(x_final - x_direct).item()
            relative_error = solution_error / torch.norm(x_direct).item()
        except Exception as e:
            solve_time = float('nan')
            solution_error = float('nan')
            relative_error = float('nan')
            print(f"Direct solve failed: {e}")

        print("\n" + "="*60)
        print("Complex BiCGStab Results Summary:")
        print(f"Data type               : {dtype_name} ({dtype_desc})")
        print(f"Matrix size             : {N}×{N}")
        print(f"Number of processes     : {world_size}")
        print(f"Iterations completed    : {iteration}")
        print(f"Final residual norm     : {final_residual_norm:.6e}")
        print(f"Target tolerance        : {tol:.6e}")
        print(f"Solution error norm     : {solution_error:.6e}")
        print(f"Relative solution error : {relative_error:.6e}")
        print("-" * 60)
        print("Performance Metrics:")
        print(f"Total time              : {total_time:.2f}s")
        print(f"Iteration time          : {iter_time:.2f}s")
        print(f"Setup time              : {total_time - iter_time:.2f}s")
        print(f"Average iter rate       : {iteration/iter_time:.1f} iter/s")
        print(f"Direct solve time       : {solve_time:.2f}s")
        if not torch.isnan(torch.tensor(solve_time)) and solve_time > 0:
            speedup = solve_time / iter_time
            print(f"Speedup vs direct solve : {speedup:.2f}x")
        print("="*60)

    cleanup()


def run_complex_demo(world_size=None, N=4096, max_iter=1000, tol=1e-6, use_cuda=True,
                     dtype="complex128", condition_target=1e4, print_interval=50):
    """
    Run distributed complex BiCGStab demo

    Args:
        world_size: Number of processes (None for auto-detection)
        N: Matrix size
        max_iter: Maximum iterations  
        tol: Convergence tolerance
        use_cuda: Whether to use CUDA
        dtype: Complex data type ("complex64", "complex128")
        condition_target: Target condition number for test matrix
        print_interval: Progress print interval
    """
    # Convert string dtype to torch dtype
    dtype_map = {
        "complex64": torch.complex64,
        "complex128": torch.complex128,
        "float32": torch.complex64,  # Allow aliases
        "float64": torch.complex128,
    }

    if isinstance(dtype, str):
        dtype = dtype_map.get(dtype.lower(), torch.complex128)

    if world_size is None:
        if use_cuda and torch.cuda.is_available():
            world_size = min(torch.cuda.device_count(),
                             8)  # Limit to 8 GPUs max
        else:
            world_size = min(mp.cpu_count(), 8)  # Limit to 8 CPU processes max

    dtype_name, dtype_desc = get_dtype_info(dtype)

    print(f"Running optimized distributed Complex BiCGStab")
    print(f"Processes    : {world_size}")
    print(f"Matrix size  : {N}×{N}")
    print(f"Data type    : {dtype_name} ({dtype_desc})")
    print(f"Max iters    : {max_iter}")
    print(f"Tolerance    : {tol:.2e}")
    print(f"Condition #  : {condition_target:.1e}")
    print(f"Using CUDA   : {use_cuda and torch.cuda.is_available()}")
    print("-" * 50)

    start_time = time.time()

    try:
        mp.spawn(
            complex_bicgstab_optimized,
            args=(world_size, N, max_iter, tol, use_cuda,
                  dtype, condition_target, print_interval),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        print(f"Error during execution: {e}")
        raise

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f}s")


if __name__ == "__main__":
    # Configuration options
    use_cuda = torch.cuda.is_available()
    use_cuda = False  # Uncomment to force CPU

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using CUDA: {use_cuda}")
    print()

    # Example runs with different configurations - start with smaller problems
    configs = [
        # (world_size, N, dtype, condition_target, description)
        (2, 256, "complex64", 1e2, "Small well-conditioned complex (single precision)"),
        (2, 512, "complex64", 1e3, "Medium well-conditioned complex (single precision)"),
        (4, 1024, "complex128", 1e4,
         "Large moderately-conditioned complex (double precision)"),
        # (8, 1024, "complex128", 1e5, "Large moderately-conditioned complex (double precision)"),
        # (16, 1024, "complex128", 1e6, "Large moderately-conditioned complex (double precision)"),
    ]

    for world_size, N, dtype, condition_target, description in configs:
        print(f"\n{'='*60}")
        print(f"Complex Test: {description}")
        print(f"{'='*60}")

        run_complex_demo(
            world_size=world_size,
            N=N,
            max_iter=1000,  # Reduced for testing
            tol=1e-6,      # Looser tolerance for testing
            use_cuda=use_cuda,
            dtype=dtype,
            condition_target=condition_target,
            print_interval=25
        )

        print("\nTest completed. Waiting before next test...")
        time.sleep(2)
