# mpi_torch_hello.py
from mpi4py import MPI
import torch

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 每个进程创建一个张量
tensor = torch.tensor([rank], dtype=torch.float32).cpu().contiguous().numpy()

# rank 0 收集所有进程的张量
if rank == 0:
    recv_buf = torch.empty(size, dtype=torch.float32).cpu().contiguous().numpy()
else:
    recv_buf = None

comm.Gather(sendbuf=tensor, recvbuf=None if rank != 0 else recv_buf, root=0)

if rank == 0:
    print(f"[Rank {rank}] 收集到的结果: {recv_buf}")
