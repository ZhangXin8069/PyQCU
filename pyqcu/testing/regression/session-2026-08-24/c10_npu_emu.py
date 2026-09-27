import torch
import pyqcu.cann as cann
cann.force_use_npu = True   # CPU 上强制 NPU 复数分解路径(无硬件模拟, bug36 后补)
import pyqcu.testing as T
T.test_npu_emulation()
