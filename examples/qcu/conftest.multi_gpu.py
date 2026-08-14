from pyqcu.testing import test_multi_gpu_multigrid

test_multi_gpu_multigrid(
    nthreads=2, lat_size=[8, 8, 8, 16], mass=0.05, atol=1e-6,
    num_levels=2, tol=1e-5, verbose=True)
