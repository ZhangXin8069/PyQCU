"""Type stub for pyqcu.cuda.qcu — the Cython bridge to libqcu.so.

Provides full type annotations and docstrings for IDE support.
Created 2026-07-28 R3.
"""

import torch
from pyqcu.cuda import define
from typing import Optional


def applyInitQcu(
    set_ptrs: torch.Tensor,
    params: torch.Tensor,
    argv: torch.Tensor,
) -> None:
    """Allocate CUDA scratch buffers from the C++ backend.

    Args:
        set_ptrs: int64 tensor, shape [100]. Scratch pointer table.
        params: int32 tensor, shape [define._PARAMS_SIZE_=54]. Lattice dimensions,
                grid sizes, data types, iteration counts, plan selection, MG config.
        argv: real-valued tensor, shape [define._ARGV_SIZE_=7]. Physical params:
              mass (idx 0), atol (idx 1), sigma (idx 2), MG level tolerances (3-6).
    """
    ...

def applyEndQcu(
    set_ptrs: torch.Tensor,
    params: torch.Tensor,
) -> None:
    """Free CUDA scratch buffers and delete LatticeSet.

    Must be called after applyInitQcu to release GPU resources.
    """
    ...

# --- Wilson dslash ---
def applyWilsonDslashQcu(
    fermion_out: torch.Tensor, fermion_in: torch.Tensor,
    gauge: torch.Tensor, set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None: ...

def testWilsonDslashQcu(
    fermion_out: torch.Tensor, fermion_in: torch.Tensor,
    gauge: torch.Tensor, set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None: ...

# --- Clover dslash ---
def applyCloverDslashQcu(
    fermion_out: torch.Tensor, fermion_in: torch.Tensor,
    gauge: torch.Tensor, set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None: ...

def testCloverDslashQcu(
    fermion_out: torch.Tensor, fermion_in: torch.Tensor,
    gauge: torch.Tensor, set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None: ...

def applyCloverQcu(
    clover: torch.Tensor, gauge: torch.Tensor,
    set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None: ...

def applyCloversQcu(
    clover: torch.Tensor, clover_inv: torch.Tensor, gauge: torch.Tensor,
    set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None: ...

def applyDslashQcu(
    fermion_out: torch.Tensor, fermion_in: torch.Tensor, gauge: torch.Tensor,
    clover: torch.Tensor, set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None: ...

# --- Wilson BiCGStab ---
def applyWilsonBistabCgQcu(
    fermion_out: torch.Tensor, fermion_in: torch.Tensor,
    gauge: torch.Tensor, set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None: ...

def applyWilsonBistabCgDslashQcu(
    fermion_out: torch.Tensor, fermion_in: torch.Tensor,
    gauge: torch.Tensor, set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None: ...

# --- Wilson CG ---
def applyWilsonCgQcu(
    fermion_out: torch.Tensor, fermion_in: torch.Tensor,
    gauge: torch.Tensor, set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None: ...

def applyWilsonCgDslashQcu(
    fermion_out: torch.Tensor, fermion_in: torch.Tensor,
    gauge: torch.Tensor, set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None: ...

# --- Clover BiCGStab ---
def applyCloverBistabCgQcu(
    fermion_out: torch.Tensor, fermion_in: torch.Tensor, gauge: torch.Tensor,
    clover_ee: torch.Tensor, clover_oo: torch.Tensor,
    clover_ee_inv: torch.Tensor, clover_oo_inv: torch.Tensor,
    set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None: ...

def applyCloverBistabCgDslashQcu(
    fermion_out: torch.Tensor, fermion_in: torch.Tensor, gauge: torch.Tensor,
    clover_ee: torch.Tensor, clover_oo: torch.Tensor,
    clover_ee_inv: torch.Tensor, clover_oo_inv: torch.Tensor,
    set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None: ...

# --- Laplacian ---
def applyLaplacianQcu(
    laplacian_out: torch.Tensor, laplacian_in: torch.Tensor,
    gauge: torch.Tensor, set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None: ...

# --- Gauss gauge ---
def applyGaussGaugeQcu(
    gauge: torch.Tensor, set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None: ...

# --- Multigrid restrict / prolong / coarse dslash ---
# BUGFIX 2026-07-28 R3: these 4 functions were previously missing from the stub.

def applyMultigridRestrictQcu(
    coarse_out: torch.Tensor, fine_in: torch.Tensor,
    null_vecs: torch.Tensor, set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None:
    """CUDA-accelerated MG restriction: coarse = P^T * fine."""
    ...

def applyMultigridProLongQcu(
    fine_out: torch.Tensor, coarse_in: torch.Tensor,
    null_vecs: torch.Tensor, set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None:
    """CUDA-accelerated MG prolongation: fine = P * coarse."""
    ...

def applyMultigridCoarseDslashQcu(
    fermion_out: torch.Tensor, fermion_in: torch.Tensor,
    hopping: torch.Tensor, sitting: torch.Tensor,
    set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None:
    """CUDA-accelerated coarse-grid Dirac operator."""
    ...

def applyCloverMultigridQcu(
    fermion_out: torch.Tensor, fermion_in: torch.Tensor, gauge: torch.Tensor,
    clover_ee: torch.Tensor, clover_oo: torch.Tensor,
    clover_ee_inv: torch.Tensor, clover_oo_inv: torch.Tensor,
    set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None:
    """CUDA-accelerated Clover multigrid solver."""
    ...
