from pyqcu.testing import *
import torch
import torch_npu
test_smear_stout(device=torch.device('npu'), dtype=torch.complex64)
