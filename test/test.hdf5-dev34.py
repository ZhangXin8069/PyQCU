from mpi4py import MPI
import h5py
import numpy as np
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# 每个进程生成一些数据
data = np.array([rank * 10 + i for i in range(10)], dtype=np.int64)
print("rank:", rank, "data:", data)
# 以并行模式打开HDF5文件
with h5py.File('parallel_example.h5', 'w', driver='mpio', comm=comm) as f:
    # 创建数据集
    dset = f.create_dataset('my_dataset', shape=(size, 10), dtype=data.dtype)
    # 每个进程将自己的数据写入数据集
    dset[rank, :] = data
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# 以并行模式打开HDF5文件
with h5py.File('parallel_example.h5', 'r', driver='mpio', comm=comm) as f:
    # 读取数据集
    dset = f['my_dataset']
    # 每个进程读取自己的数据块
    local_data = dset[rank, :]
    print(f"进程 {rank} 读取的数据: {local_data}")
    print(f"进程 {rank} 读取的数据类型: {local_data.dtype}")
