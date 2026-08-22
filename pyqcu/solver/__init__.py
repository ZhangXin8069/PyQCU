from ._bistabcg import bistabcg as bistabcg
from ._bistabcg import bistabcg_history as bistabcg_history
from ._gmres import fgmres as fgmres
from ._mr import mr as mr
from ._multigrid import multigrid as multigrid
from argparse import Namespace
Namespace.__module__ = "pyqcu.solver"

# 多线程多卡 C++ Multigrid 驱动（一线程一卡）：延迟导入避免循环依赖
# （pyqcu.tools._multigrid 顶层 from pyqcu import solver, tools）。
_MULTI_GPU_CACHE = {}


def __getattr__(name):
    if name in ("MultiGpuMultigrid", "verify_multi_gpu_mg"):
        if name not in _MULTI_GPU_CACHE:
            from pyqcu.cuda._multi_gpu import MultiGpuMultigrid as _MG
            from pyqcu.cuda._multi_gpu import verify_multi_gpu_mg as _VM
            _MULTI_GPU_CACHE["MultiGpuMultigrid"] = _MG
            _MULTI_GPU_CACHE["verify_multi_gpu_mg"] = _VM
        return _MULTI_GPU_CACHE[name]
    raise AttributeError(f"module 'pyqcu.solver' has no attribute '{name}'")
