from pyqcu.testing import *
import torch
import torch_npu
test_dslash_clover(with_data=False, device=torch.device('npu'), dtype=torch.complex64)
