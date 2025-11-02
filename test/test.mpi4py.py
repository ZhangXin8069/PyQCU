from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

send = np.arange(10, dtype=np.int32).copy(order='C') + rank*100
recv = np.empty_like(send, order='C')

print(f"[Rank {rank}] before: send={send} addr={hex(send.ctypes.data)} recv_addr={hex(recv.ctypes.data)}")
comm.Barrier()
comm.Sendrecv([send, MPI.INT], dest=(rank+1) % 2, sendtag=rank,
              recvbuf=[recv, MPI.INT], source=(rank-1) % 2, recvtag=(rank-1) % 2)
print(f"[Rank {rank}] after recv={recv}")
comm.Barrier()
