"""Torch/CANN solver implementations."""

from ._bistabcg import bistabcg as bistabcg
from ._bistabcg import bistabcg_history as bistabcg_history
from ._cacg import cacg as cacg
from ._gmres import fgmres as fgmres
from ._krylov_drivers import TorchBiCGStabDriver as TorchBiCGStabDriver
from ._krylov_drivers import TorchCGDriver as TorchCGDriver
from ._lanczos import tr_lanczos as tr_lanczos
from ._mr import mr as mr
from ._multishift_cg import multishift_cg as multishift_cg
from ._multishift_cg import multishift_cg_true_residuals as multishift_cg_true_residuals
from .multigrid import LegacyMultigrid as LegacyMultigrid
from .multigrid import StrictMultigrid as StrictMultigrid
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

Namespace.__module__ = "pyqcu.torch.solver"

_MULTI_GPU_CACHE = {}


def __getattr__(name):
    if name in ("MultiGpuMultigrid", "verify_multi_gpu_mg"):
        if name not in _MULTI_GPU_CACHE:
            from pyqcu.cuda._multi_gpu import MultiGpuMultigrid as mg_type
            from pyqcu.cuda._multi_gpu import verify_multi_gpu_mg as verify

            _MULTI_GPU_CACHE["MultiGpuMultigrid"] = mg_type
            _MULTI_GPU_CACHE["verify_multi_gpu_mg"] = verify
        return _MULTI_GPU_CACHE[name]
    raise AttributeError(f"module 'pyqcu.torch.solver' has no attribute {name!r}")


__all__ = [
    "Checkerboard",
    "CompactParityLayout",
    "CompactParityOperator",
    "LegacyMultigrid",
    "MultiGpuMultigrid",
    "ParitySchurOperator",
    "QcuStrictAssetBinding",
    "StrictCoarseOperator",
    "StrictMatPCOperator",
    "StrictMultigrid",
    "StrictTransfer",
    "TorchBiCGStabDriver",
    "TorchCGDriver",
    "bistabcg",
    "bistabcg_history",
    "cacg",
    "fgmres",
    "mr",
    "multishift_cg",
    "multishift_cg_true_residuals",
    "tr_lanczos",
    "verify_multi_gpu_mg",
]
