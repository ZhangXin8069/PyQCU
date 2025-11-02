import mpi4py.MPI as MPI
import numpy as np
# import torch # dcu:bug here!
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
rank_plus = (rank+1) % size
rank_minus = (size+rank-1) % size
send = np.arange(10, dtype=np.int32)*2**rank
recv = np.zeros_like(send)
comm.Barrier()
comm.Sendrecv(sendbuf=send, dest=rank_plus, sendtag=rank,
              recvbuf=recv, source=rank_minus, recvtag=rank_minus)
comm.Barrier()