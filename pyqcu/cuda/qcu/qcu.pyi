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
        params: int32 tensor, shape [58]. Lattice dimensions,
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

def applyCloverBistabCgPrepareQcu(
    compact_rhs: torch.Tensor, full_rhs: torch.Tensor, gauge: torch.Tensor,
    clover_ee: torch.Tensor, clover_oo: torch.Tensor,
    clover_ee_inv: torch.Tensor, clover_oo_inv: torch.Tensor,
    set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None:
    """Prepare the odd symmetric-Schur right-hand side."""
    ...

def applyCloverBistabCgReconstructQcu(
    full_out: torch.Tensor, full_rhs: torch.Tensor,
    target_odd: torch.Tensor, gauge: torch.Tensor,
    clover_ee: torch.Tensor, clover_oo: torch.Tensor,
    clover_ee_inv: torch.Tensor, clover_oo_inv: torch.Tensor,
    set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None:
    """Reconstruct the even field from an odd Schur solution."""
    ...

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

def applyMultigridCoarseDslashWideQcu(
    fermion_out: torch.Tensor, fermion_in: torch.Tensor,
    sitting: torch.Tensor, hop_nn: torch.Tensor, hop_diag: torch.Tensor,
    set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None:
    """CUDA-accelerated wide-stencil coarse-grid Schur operator A_c = P^T S P.

    Wide 33-tensor stencil (on-site + 8 nearest + 24 diagonal couplings),
    Schur-consistent coarse operator used for coarse-level null-vector
    generation and stencil probing (arbitrary DOF E).

    Tensor layouts:
      fermion_out/in:  [E, X, Y, Z, Lt]  (coarse DOF × coarse odd lattice)
      sitting:         [E, E, X, Y, Z, Lt]
      hop_nn:          [2, 4, E, E, X, Y, Z, Lt]   (pm × ward)
      hop_diag:        [2, 2, 6, E, E, X, Y, Z, Lt] (s1 × s2 × pair)

    Params used:
      _MG_LEVEL1_E_ = E (coarse DOF)
      _MG_LEVEL1_X_, _Y_, _Z_, _T_ = coarse odd lattice
    """
    ...

def applyMultigridStrictCoarseQcu(
    fermion_out: torch.Tensor, fermion_in: torch.Tensor,
    links: torch.Tensor, onsite_pair: torch.Tensor,
    set_ptrs: torch.Tensor, params: torch.Tensor,
    onsite_index: int = -1,
) -> None:
    """Apply strict QUDA-stored links on ``[E,X,Y,Z,T]``.

    ``links`` is ``[2,4,E,E,X,Y,Z,T]``.  The backward half is stored at
    ``q-mu`` and adjointed during the gather.  ``onsite_index=-1`` applies
    hopping only; 0 and 1 select ``X`` and ``X^-1`` from ``onsite_pair``.
    """
    ...

def applyMultigridStrictMatPCQcu(
    fermion_out: torch.Tensor, fermion_in: torch.Tensor,
    preconditioned_links: torch.Tensor, scratch: torch.Tensor,
    set_ptrs: torch.Tensor, params: torch.Tensor, parity: int,
) -> None:
    """Apply ``I-Hhat_pq Hhat_qp`` on ``[E,X,Y,Z,T/2]``."""
    ...

def applyMultigridStrictPrepareQcu(
    fermion_out: torch.Tensor, full_rhs: torch.Tensor,
    preconditioned_links: torch.Tensor, onsite_pair: torch.Tensor,
    scratch: torch.Tensor, set_ptrs: torch.Tensor,
    params: torch.Tensor, parity: int,
) -> None:
    """Prepare the compact symmetric-preconditioned rhs from a full rhs."""
    ...

def applyMultigridStrictReconstructQcu(
    full_out: torch.Tensor, full_rhs: torch.Tensor,
    target_solution: torch.Tensor, preconditioned_links: torch.Tensor,
    onsite_pair: torch.Tensor, scratch: torch.Tensor,
    set_ptrs: torch.Tensor, params: torch.Tensor, parity: int,
) -> None:
    """Reconstruct the eliminated parity into a full coarse solution."""
    ...

def applyMultigridStrictRestrictQcu(
    coarse_out: torch.Tensor, fine_in: torch.Tensor,
    null_vectors: torch.Tensor, set_ptrs: torch.Tensor,
    params: torch.Tensor, parity: int,
) -> None:
    """Restrict one compact fine parity to a full coarse field."""
    ...

def applyMultigridStrictProLongQcu(
    fine_out: torch.Tensor, coarse_in: torch.Tensor,
    null_vectors: torch.Tensor, set_ptrs: torch.Tensor,
    params: torch.Tensor, parity: int,
) -> None:
    """Prolong a full coarse field to one compact fine parity."""
    ...

def applyMultigridStrictVCycleQcu(
    full_out: torch.Tensor, full_rhs: torch.Tensor,
    set_ptrs: torch.Tensor, params: torch.Tensor,
    start_level: int = 1,
) -> int:
    """Run the recursive strict coarse V-cycle.

    The return value is the exact transient allocation in bytes for the
    arena-backed hierarchy.  Caller-owned resident transfer/operator assets
    are not included.
    """
    ...

def applyMultigridStrictInitQcu(
    set_ptrs: torch.Tensor, params: torch.Tensor, start_level: int = 1,
) -> int:
    """Allocate a reusable strict coarse hierarchy and return its bytes."""
    ...

def applyMultigridStrictEndQcu(
    set_ptrs: torch.Tensor, params: torch.Tensor,
) -> None:
    """Release the reusable strict hierarchy before ``applyEndQcu``."""
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
      set_ptrs[30 + 4*fl + 0] = null_vecs pointer           [E_{fl+1}, e_fl, X_fl, Y_fl, Z_fl, T_fl]
      set_ptrs[30 + 4*fl + 1] = hop_nn pointer              [2, 4, E_{fl+1}, E_{fl+1}, X_{fl+1}, Y_{fl+1}, Z_{fl+1}, T_{fl+1}]
      set_ptrs[30 + 4*fl + 2] = hop_diag pointer            [2, 2, 6, E_{fl+1}, E_{fl+1}, X_{fl+1}, Y_{fl+1}, Z_{fl+1}, T_{fl+1}]
      set_ptrs[30 + 4*fl + 3] = sit_packed pointer          [E_{fl+1}, E_{fl+1}, X_{fl+1}, Y_{fl+1}, Z_{fl+1}, T_{fl+1}]

    Params used:
      _MG_NUM_LEVEL_ = number of MG levels
      _MG_LEVEL1_E_, _X_, _Y_, _Z_, _T_ = coarse level 1 config
      _MG_LEVEL1_MAX_ITER_ = max iterations for coarse BiStabCG
      _MG_LEVEL1_NUM_RESTART_ = V-cycle interval (every N fine iterations)
      _MAX_ITER_ = max fine-level iterations
      _MG_USE_GCR_ = mode bit mask selecting GCR/MR/Chebyshev/CA-GCR,
                     W/F/K-cycle, or BiCGStabL variants
      _VERBOSE_ = 0 or 1 for logging

    argv:
      _ATOL_ = fine-level convergence tolerance
      _MG_LEVEL1_ATOL_ = initial coarse-level tolerance (may be overridden by relative tol)
    """
    ...

def verifyCloverMultigridQcu(
    fermion_out: torch.Tensor, fermion_in: torch.Tensor, gauge: torch.Tensor,
    clover_ee: torch.Tensor, clover_oo: torch.Tensor,
    clover_ee_inv: torch.Tensor, clover_oo_inv: torch.Tensor,
    set_ptrs: torch.Tensor, params: torch.Tensor,
) -> int:
    """Run all five C++ multigrid diagnostics.

    Returns 0 for PASS, 1 for a diagnostic FAIL, and 2 for a bridge/runtime
    error.  The caller still owns the outer ``applyEndQcu`` lifecycle.
    """
    ...
