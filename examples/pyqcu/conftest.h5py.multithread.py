from pyqcu.testing import test_h5py_multithread
import torch

test_h5py_multithread(nthreads=4, dtype=torch.complex64, lat_size=[4, 4, 4, 8])
