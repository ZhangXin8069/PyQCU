import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size, use_cuda=True):
    """初始化分布式环境"""
    if use_cuda and torch.cuda.is_available():
        backend = "nccl"
        torch.cuda.set_device(rank)
    else:
        backend = "gloo"

    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29500')

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size
    )
    if rank == 0:
        print(f"[Rank {rank}] Distributed initialized with backend={backend}")

def cleanup():
    dist.destroy_process_group()

def distributed_matvec(A_local, x, world_size, rank):
    """分布式矩阵向量乘法"""
    x_list = [x.clone() if r == rank else torch.zeros_like(x) for r in range(world_size)]
    dist.broadcast_object_list(x_list, src=0)
    x = x_list[rank]

    y_local = A_local @ x
    y_list = [torch.zeros_like(y_local) for _ in range(world_size)]
    dist.all_gather(y_list, y_local)
    y_full = torch.cat(y_list, dim=0)
    return y_full

def bicgstab(rank, world_size, N=4096, max_iter=100, tol=1e-6, use_cuda=True):
    N = int(N)
    max_iter = int(max_iter)
    setup(rank, world_size, use_cuda=use_cuda)

    device = torch.device(f"cuda:{rank}" if use_cuda and torch.cuda.is_available() else "cpu")
    rows_per_rank = N // world_size
    start = rank * rows_per_rank
    end = start + rows_per_rank

    # rank 0 初始化矩阵和向量
    if rank == 0:
        torch.manual_seed(0)
        M = torch.randn(N, N)
        A_full = M.T @ M + torch.eye(N) * 1e-3
        b_full = torch.randn(N, 1)
    else:
        A_full = torch.zeros(N, N)
        b_full = torch.zeros(N, 1)

    # 广播 A_full / b_full
    A_list = [A_full]
    b_list = [b_full]
    dist.broadcast_object_list(A_list, src=0)
    dist.broadcast_object_list(b_list, src=0)
    A_full = A_list[0].to(device)
    b_full = b_list[0].to(device)
    A_local = A_full[start:end, :]

    # BiCGStab 初始化
    x = torch.zeros((N, 1), device=device)
    r = b_full - distributed_matvec(A_local, x, world_size, rank)
    r_hat = r.clone()
    rho_old = alpha = omega = torch.tensor(1.0, device=device)
    v = torch.zeros_like(r)
    p = torch.zeros_like(r)

    for it in range(1, max_iter + 1):
        rho_new = torch.dot(r_hat.flatten(), r.flatten())
        if rho_new.abs() < 1e-15:
            break

        if it == 1:
            p = r.clone()
        else:
            beta = (rho_new / rho_old) * (alpha / omega)
            p = r + beta * (p - omega * v)

        v = distributed_matvec(A_local, p, world_size, rank)
        alpha = rho_new / torch.dot(r_hat.flatten(), v.flatten())
        s = r - alpha * v
        if torch.norm(s) < tol:
            x = x + alpha * p
            break

        t = distributed_matvec(A_local, s, world_size, rank)
        omega = torch.dot(t.flatten(), s.flatten()) / torch.dot(t.flatten(), t.flatten())
        x = x + alpha * p + omega * s
        r = s - omega * t

        # 打印每个 rank 的进度
        if it % 10 == 0:
            print(f"[Rank {rank}] Iter {it}, Residual norm = {torch.norm(r).item():.6e}")

        if torch.norm(r) < tol or omega.abs() < 1e-15:
            break

        rho_old = rho_new

    # rank 0 验证解
    if rank == 0:
        x_bicg = x.cpu()
        Ax = A_full.cpu() @ x_bicg
        res_norm = torch.norm(Ax - b_full.cpu()).item()
        x_ref = torch.linalg.solve(A_full.cpu(), b_full.cpu())
        err_norm = torch.norm(x_bicg - x_ref).item()
        rel_err = err_norm / torch.norm(x_ref).item()
        print("BiCGStab finished.")
        print(f"Final residual norm = {res_norm:.6e}")
        print(f"Solution error norm = {err_norm:.6e}")
        print(f"Relative error      = {rel_err:.6e}")

    cleanup()

def run_demo(world_size, use_cuda=True):
    mp.spawn(bicgstab, args=(world_size, 4096, 1e-6, use_cuda), nprocs=world_size, join=True)

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    world_size = torch.cuda.device_count() if use_cuda else 4  # CPU 多进程
    run_demo(world_size, use_cuda=use_cuda)
