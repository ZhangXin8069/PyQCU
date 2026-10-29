"""Torch/CANN-only numerical utilities."""

import sys

from ._linalg import norm as norm
from ._linalg import norm_many as norm_many
from ._linalg import vdot as vdot
from ._linalg import vdot_many as vdot_many
from ._multigrid import PAIRS as PAIRS
from ._multigrid import SIGN as SIGN
from ._multigrid import apply_stencil as apply_stencil
from ._multigrid import build_stencil as build_stencil
from ._multigrid import build_stencil_mt as build_stencil_mt
from ._multigrid import give_null_vecs as give_null_vecs
from ._multigrid import give_null_vecs_mt as give_null_vecs_mt
from ._multigrid import local_orthogonalize as local_orthogonalize
from ._multigrid import prolong as prolong
from ._multigrid import restrict as restrict
from ._hierarchical import HierarchicalCache as HierarchicalCache
from ._hierarchical import HierarchicalTensor as HierarchicalTensor
from ._hierarchical import offload_to_disk as offload_to_disk
from ._hierarchical import offload_to_ram as offload_to_ram
from ._hierarchical import reload_to_vram as reload_to_vram

try:
    from ._einsum import Eexyzt_exyzt2Exyzt as Eexyzt_exyzt2Exyzt
    from ._matul import matmul_cpu as matmul_cpu
    from ._matul import matmul_gpu as matmul_gpu
except Exception as exc:  # pragma: no cover - optional TileLang environment
    print(f"Error:{exc}", file=sys.stderr)

__all__ = [
    "Eexyzt_exyzt2Exyzt",
    "HierarchicalCache",
    "HierarchicalTensor",
    "PAIRS",
    "SIGN",
    "apply_stencil",
    "build_stencil",
    "build_stencil_mt",
    "give_null_vecs",
    "give_null_vecs_mt",
    "local_orthogonalize",
    "matmul_cpu",
    "matmul_gpu",
    "norm",
    "norm_many",
    "offload_to_disk",
    "offload_to_ram",
    "prolong",
    "reload_to_vram",
    "restrict",
    "vdot",
    "vdot_many",
]
