"""Stable public solver facade.

Torch implementations live in :mod:`pyqcu.torch.solver`. Backend-neutral
multigrid and Krylov selectors remain here so all three backends share stable
public entry points.
"""

from pyqcu.torch.solver import Checkerboard as Checkerboard
from pyqcu.torch.solver import CompactParityLayout as CompactParityLayout
from pyqcu.torch.solver import CompactParityOperator as CompactParityOperator
from pyqcu.torch.solver import LegacyMultigrid as LegacyMultigrid
from pyqcu.torch.solver import ParitySchurOperator as ParitySchurOperator
from pyqcu.torch.solver import QcuStrictAssetBinding as QcuStrictAssetBinding
from pyqcu.torch.solver import StrictCoarseOperator as StrictCoarseOperator
from pyqcu.torch.solver import StrictMatPCOperator as StrictMatPCOperator
from pyqcu.torch.solver import StrictMultigrid as StrictMultigrid
from pyqcu.torch.solver import StrictTransfer as StrictTransfer
from pyqcu.torch.solver import bistabcg as bistabcg
from pyqcu.torch.solver import bistabcg_history as bistabcg_history
from pyqcu.torch.solver import cacg as cacg
from pyqcu.torch.solver import fgmres as fgmres
from pyqcu.torch.solver import mr as mr
from pyqcu.torch.solver import multishift_cg as multishift_cg
from pyqcu.torch.solver import multishift_cg_true_residuals as multishift_cg_true_residuals
from pyqcu.torch.solver import tr_lanczos as tr_lanczos
from .krylov import Krylov as Krylov
from .multigrid import Multigrid as Multigrid

from argparse import Namespace

Namespace.__module__ = "pyqcu.solver"

_MULTI_GPU_CACHE = {}


def __getattr__(name):
    if name in ("MultiGpuMultigrid", "verify_multi_gpu_mg"):
        if name not in _MULTI_GPU_CACHE:
            from pyqcu.cuda._multi_gpu import MultiGpuMultigrid as mg_type
            from pyqcu.cuda._multi_gpu import verify_multi_gpu_mg as verify

            _MULTI_GPU_CACHE["MultiGpuMultigrid"] = mg_type
            _MULTI_GPU_CACHE["verify_multi_gpu_mg"] = verify
        return _MULTI_GPU_CACHE[name]
    raise AttributeError(f"module 'pyqcu.solver' has no attribute {name!r}")

__all__ = [
    "Checkerboard",
    "CompactParityLayout",
    "CompactParityOperator",
    "LegacyMultigrid",
    "Krylov",
    "MultiGpuMultigrid",
    "Multigrid",
    "ParitySchurOperator",
    "QcuStrictAssetBinding",
    "StrictCoarseOperator",
    "StrictMatPCOperator",
    "StrictMultigrid",
    "StrictTransfer",
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
