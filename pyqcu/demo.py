import cupy as cp
import numpy as np
from pyqcu import qcu, define, gauge, linalg, io
def give(params, argv=None, set_ptrs=None, sigma=0.1, seed=12138):
    if argv is None:
        from pyqcu.set import argv
    if set_ptrs is None:
        from pyqcu.set import set_ptrs
    print('My rank is ', define.rank)
    print("Parameters:", params)
    src = cp.ones(params[define._LAT_XYZT_]*define._LAT_SC_,
                  dtype=define.dtype(params[define._DATA_TYPE_]))
    src = io.fermion2psctzyx(src, params)
    print("Src data:", src.data)
    print("Src shape:", src.shape)
    argv = argv.astype(src.real.dtype)
    print("Arguments:", argv)
    print("Arguments data:", argv.data)
    print("Arguments dtype:", argv.dtype)
    print("Demo is running...")
    print("Set pointers:", set_ptrs)
    print("Set pointers data:", set_ptrs.data)
    U = gauge.give_gauss_SU3(sigma=sigma, seed=seed,
                             dtype=src.dtype, size=params[define._LAT_XYZT_]*define._LAT_D_)
    U = io.gauge2dptzyxcc(U, params)
    U = io.dptzyxcc2ccdptzyx(U)
    print("U data:", U.data)
    print("U shape:", U.shape)
    wilson_cg_params = params.copy()
    wilson_cg_params[define._SET_INDEX_] = 0
    wilson_cg_params[define._SET_PLAN_] = define._SET_PLAN1_
    qcu.applyInitQcu(set_ptrs, wilson_cg_params, argv)
    wilson_dslash_eo_params = params.copy()
    wilson_dslash_eo_params[define._SET_INDEX_] = 1
    wilson_dslash_eo_params[define._SET_PLAN_] = define._SET_PLAN0_
    wilson_dslash_eo_params[define._PARITY_] = define._EVEN_
    wilson_dslash_eo_params[define._DAGGER_] = define._NO_USE_
    qcu.applyInitQcu(set_ptrs, wilson_dslash_eo_params, argv)
    wilson_dslash_eo_dag_params = params.copy()
    wilson_dslash_eo_dag_params[define._SET_INDEX_] = 2
    wilson_dslash_eo_dag_params[define._SET_PLAN_] = define._SET_PLAN0_
    wilson_dslash_eo_dag_params[define._PARITY_] = define._EVEN_
    wilson_dslash_eo_dag_params[define._DAGGER_] = define._USE_
    qcu.applyInitQcu(set_ptrs, wilson_dslash_eo_dag_params, argv)
    wilson_dslash_oe_params = params.copy()
    wilson_dslash_oe_params[define._SET_INDEX_] = 3
    wilson_dslash_oe_params[define._SET_PLAN_] = define._SET_PLAN0_
    wilson_dslash_oe_params[define._PARITY_] = define._ODD_
    wilson_dslash_oe_params[define._DAGGER_] = define._NO_USE_
    qcu.applyInitQcu(set_ptrs, wilson_dslash_oe_params, argv)
    wilson_dslash_oe_dag_params = params.copy()
    wilson_dslash_oe_dag_params[define._SET_INDEX_] = 4
    wilson_dslash_oe_dag_params[define._SET_PLAN_] = define._SET_PLAN0_
    wilson_dslash_oe_dag_params[define._PARITY_] = define._ODD_
    wilson_dslash_oe_dag_params[define._DAGGER_] = define._USE_
    qcu.applyInitQcu(set_ptrs, wilson_dslash_oe_dag_params, argv)
    dest = cp.zeros_like(src)
    qcu.applyWilsonBistabCgQcu(dest, src,
                               U, set_ptrs, wilson_cg_params)
    print("Dest data:", dest.data)
    print("Dest shape:", dest.shape)
    return U, src, dest, set_ptrs, wilson_cg_params, wilson_dslash_eo_params, wilson_dslash_oe_params, wilson_dslash_eo_dag_params, wilson_dslash_oe_dag_params
def end(set_ptrs, params):
    print("Set pointers:", set_ptrs)
    print("Set pointers data:", set_ptrs.data)
    _params = params.copy()
    for i in range(len(set_ptrs)):
        if set_ptrs[i] != 0:
            _params[define._SET_INDEX_] = i
            print(f"Set pointers[{i}] data: {set_ptrs[i]} has been freed")
            qcu.applyEndQcu(set_ptrs, _params)
