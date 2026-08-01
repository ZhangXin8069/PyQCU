"""Type stub for pyqcu.cuda.qcu — the Cython bridge to libqcu.so.

Provides full type annotations and docstrings for IDE support.
Updated 2026-08-01: added MG coarse operator documentation, fixed param notes.
"""

import torch
from typing import Optional


def applyInitQcu(
    set_ptrs: torch.Tensor,
    params: torch.Tensor,
    argv: torch.Tensor,
) -> None:
    """Allocate CUDA scratch buffers from the C++ backend.

    Args:
        set_ptrs: int64 tensor, shape [100]. Scratch pointer table.
        params: int32 tensor, shape [54]. Lattice dimensions,
                grid sizes, data types, iteration counts, plan selection, MG config.
        argv: real-valued tensor, shape [7]. Physical params:
              mass (idx 0), atol (idx 1), sigma (idx 2), MG level tolerances (3-6).
    """
    ...

def applyEndQcu(
    set_ptrs: torch.Tensor,
    params: torch.Tensor,
) -> None:
    """Free CUDA scratch buffers and delete LatticeSet."""
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

# --- Multigrid ---

def applyMultigridRestrictQcu(
    coarse_out: torch.Tensor, fine_in: torch.Tensor,
    null_vecs: torch.Tensor, set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None:
    """CUDA-accelerated MG restriction: coarse = P^T * fine.

    Tensor layouts:
      coarse_out: [E, Xc, Yc, Zc, Tc]  (E = coarse DOF)
      fine_in:    [e, Xf, Yf, Zf, Tf]  (e = fine DOF, always _LAT_SC_ = 12)
      null_vecs:  [E, e, Xf, Yf, Zf, Tf]  (local orthogonalized, FLAT format)

    Params used:
      _MG_LEVEL1_E_ = E (coarse DOF)
      _MG_LEVEL1_X_, _Y_, _Z_, _T_ = coarse lattice
      _LAT_X_, _LAT_Y_, _LAT_Z_, _LAT_T_ = fine lattice

    IMPORTANT: Set _LAT_T_ to the FULL t-dimension (not halved) before calling.
    The fine DOF (e) is always _LAT_SC_ = 12, read from a compile-time constant.
    """
    ...

def applyMultigridProLongQcu(
    fine_out: torch.Tensor, coarse_in: torch.Tensor,
    null_vecs: torch.Tensor, set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None:
    """CUDA-accelerated MG prolongation: fine = P * coarse.

    Tensor layouts:
      fine_out:   [e, Xf, Yf, Zf, Tf]
      coarse_in:  [E, Xc, Yc, Zc, Tc]
      null_vecs:  [E, e, Xf, Yf, Zf, Tf]  (FLAT format)

    Params used: same convention as applyMultigridRestrictQcu.
    """
    ...

def applyMultigridCoarseDslashQcu(
    fermion_out: torch.Tensor, fermion_in: torch.Tensor,
    hopping: torch.Tensor, sitting: torch.Tensor,
    set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None:
    """CUDA-accelerated coarse-grid Dirac operator D_c = sitting + hopping_plus + hopping_minus.

    Tensor layouts:
      fermion_out/in:  [E, X, Y, Z, Lt]  (coarse DOF × coarse lattice)
      hopping:         [2, 4, E, E, X, Y, Z, Lt]
                       dim0: 0=plus, 1=minus; dim1: ward (0=X,1=Y,2=Z,3=T)
      sitting:         [E, E, X, Y, Z, Lt]

    Params used:
      _MG_LEVEL1_E_ = E (coarse DOF)
      _MG_LEVEL1_X_, _Y_, _Z_, _T_ = coarse lattice
    """
    ...

def applyCloverMultigridQcu(
    fermion_out: torch.Tensor, fermion_in: torch.Tensor, gauge: torch.Tensor,
    clover_ee: torch.Tensor, clover_oo: torch.Tensor,
    clover_ee_inv: torch.Tensor, clover_oo_inv: torch.Tensor,
    set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None:
    """Full Clover multigrid V-cycle solver.

    Performs BiStabCG iterations at the finest level with coarse-grid
    V-cycle corrections applied every num_restart iterations.

    All tensor layouts follow parity-split (even-odd) convention:
      fermion_out/in:    [2, 4, 3, X, Y, Z, Lt/2]  (parity × spin × color × XYZT)
      gauge:             [2, 3, 3, 4, X, Y, Z, Lt/2]
      clover_ee/oo/inv:  [4, 3, 4, 3, X, Y, Z, Lt/2]

    Coarse-grid operators are passed via set_ptrs:
      set_ptrs[10 + 3*fl + 0] = null_vecs (LONV) pointer  [E_{fl+1}, e_fl, X_fl, Y_fl, Z_fl, T_fl]
      set_ptrs[10 + 3*fl + 1] = hop_packed pointer         [2, 4, E_{fl+1}, E_{fl+1}, X_{fl+1}, Y_{fl+1}, Z_{fl+1}, T_{fl+1}]
      set_ptrs[10 + 3*fl + 2] = sit_packed pointer         [E_{fl+1}, E_{fl+1}, X_{fl+1}, Y_{fl+1}, Z_{fl+1}, T_{fl+1}]

    Params used:
      _MG_NUM_LEVEL_ = number of MG levels
      _MG_LEVEL1_E_, _X_, _Y_, _Z_, _T_ = coarse level 1 config
      _MG_LEVEL1_MAX_ITER_ = max iterations for coarse BiStabCG
      _MG_LEVEL1_NUM_RESTART_ = V-cycle interval (every N fine iterations)
      _MAX_ITER_ = max fine-level iterations
      _VERBOSE_ = 0 or 1 for logging

    argv:
      _ATOL_ = fine-level convergence tolerance
      _MG_LEVEL1_ATOL_ = initial coarse-level tolerance (may be overridden by relative tol)
    """
    ...
