from ._bistabcg import bistabcg as bistabcg
from ._bistabcg import bistabcg_history as bistabcg_history
from ._gmres import fgmres as fgmres
from ._mr import mr as mr
from .multigrid import (LegacyMultigrid as LegacyMultigrid,
                        Multigrid as Multigrid,
                        StrictMultigrid as StrictMultigrid)
from .multigrid.strict_multigrid import (
    Checkerboard as Checkerboard,
    CompactParityLayout as CompactParityLayout,
    CompactParityOperator as CompactParityOperator,
    ParitySchurOperator as ParitySchurOperator,
    QcuStrictAssetBinding as QcuStrictAssetBinding,
    StrictCoarseOperator as StrictCoarseOperator,
    StrictMatPCOperator as StrictMatPCOperator,
    StrictTransfer as StrictTransfer,
)
from argparse import Namespace
Namespace.__module__ = "pyqcu.solver"

# 测试专用：单进程多线程多 GPU 的 Legacy Multigrid 驱动（一线程一卡）。
# 只有显式访问 MultiGpuMultigrid/verify_multi_gpu_mg 时才延迟导入；生产
# CUDA 拓扑固定为一 Python 进程 = 一 MPI rank = 一 GPU/DCU/GPGPU。
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
from ._multishift_cg import multishift_cg as multishift_cg
from ._multishift_cg import multishift_cg_true_residuals as multishift_cg_true_residuals
from ._cacg import cacg as cacg
from ._lanczos import tr_lanczos as tr_lanczos
