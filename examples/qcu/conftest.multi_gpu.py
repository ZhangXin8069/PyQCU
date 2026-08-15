from pyqcu.testing import test_multi_gpu_multigrid

# 本机 3 卡：torch 视角 device 0=V100-32GB, 1/2=P100-16GB
test_multi_gpu_multigrid(
    nthreads=3, lat_size=[8, 8, 8, 16], mass=0.05, atol=1e-6,
    num_levels=2, tol=1e-5, verbose=True)
# 独立问题模式（每线程不同 seed 吞吐并行；小格子快速验证）
test_multi_gpu_multigrid(
    nthreads=3, lat_size=[4, 4, 4, 8], mass=0.05, atol=1e-6,
    num_levels=2, tol=1e-5, verbose=False, independent_problems=True)
