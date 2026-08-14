from pyqcu.testing import test_multigrid_multithread

test_multigrid_multithread(nthreads=2, lat_size=[8, 8, 8, 8], mass=0.05, tol=1e-5)
