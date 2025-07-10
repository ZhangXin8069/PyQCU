import torch
from pyqcu.ascend import dslash

# Example usage
if __name__ == "__main__":
    # Lattice parameters
    # latt_size = (16, 8, 8, 8)
    # latt_size = (1, 1, 1, 1)
    # latt_size = (2, 2, 2, 2)
    # latt_size = (8, 4, 4, 4)
    latt_size = (8, 4, 4, 4)
    # latt_size = (8, 4, 4, 8)
    kappa = 0.125
    # dtype = torch.complex128
    dtype = torch.complex64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    verbose = True
    print(f"Using device: {device}")
    # Initialize lattice gauge theory
    wilson = dslash.wilson(
        latt_size=latt_size,
        kappa=kappa,
        dtype=dtype,
        device=device,
        verbose=verbose
    )
    clover = dslash.clover(
        latt_size=latt_size,
        kappa=kappa,
        dtype=dtype,
        device=device,
        verbose=verbose
    )
    # Generate random gauge field
    U = wilson.generate_gauge_field(sigma=0.1, seed=42)
    # U = torch.eye(3, 3, dtype=dtype, device=device).repeat(
    #     4, latt_size[-1], latt_size[-2], latt_size[-3], latt_size[-4], 1, 1).permute(5, 6, 0, 1, 2, 3, 4)
    # U = torch.ones_like(U)
    # U = torch.zeros_like(U)
    # U=U*1.0j+U
    # U = torch.tensor(data=[0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=dtype, device=device).reshape(3, 3).repeat(
    #     4, latt_size[-1], latt_size[-2], latt_size[-3], latt_size[-4], 1, 1).permute(5, 6, 0, 1, 2, 3, 4)
    # Generate random source field [s, c, t, z, y, x]
    src = torch.randn(4, 3, latt_size[3], latt_size[2], latt_size[1], latt_size[0],
                      dtype=dtype, device=device)
    # print(
    #     f" torch.tensor(data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=dtype, device=device).reshape(4, 3){ torch.tensor(data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=dtype, device=device).reshape(4, 3)}")
    # src = torch.tensor(data=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=dtype, device=device).reshape(4, 3).repeat(
    #     latt_size[-1], latt_size[-2], latt_size[-3], latt_size[-4], 1, 1).permute(4, 5,  0, 1, 2, 3)
    # src = torch.ones_like(src)
    # Apply Wilson-Dirac operator
    dest = wilson.apply_dirac_operator(src, U)
    # Verify properties
    print("\nDests:")
    print(f"Dest shape: {dest.shape}")
    print(f"Max abs value: {torch.max(torch.abs(dest)).item()}")
    print(f"Dest norm: {torch.norm(dest).item()}")
    print(f"Dest dtype: {dest.dtype}")
    print(f"U value:{U}")
    # print(f"Src value:{src}")
    # print(f"Dest value:{dest}")
    import warnings
    from pyqcu.cuda.set import *
    from pyqcu.cuda import io, gauge, cg, bistabcg, define, qcu, linalg, eigen
    import cupy as cp
    import numpy as np
    np.Inf = np.inf
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    params[define._LAT_X_] = latt_size[define._LAT_X_]
    params[define._LAT_Y_] = latt_size[define._LAT_Y_]
    params[define._LAT_Z_] = latt_size[define._LAT_Z_]
    params[define._LAT_T_] = latt_size[define._LAT_T_]
    params[define._LAT_XYZT_] = params[define._LAT_X_] * \
        params[define._LAT_Y_] * \
        params[define._LAT_Z_] * params[define._LAT_T_]
    params[define._NODE_RANK_] = define.rank
    params[define._NODE_SIZE_] = define.size
    # params[define._DATA_TYPE_] = define._LAT_C128_
    params[define._DATA_TYPE_] = define._LAT_C64_
    argv = argv.astype(define.dtype_half(params[define._DATA_TYPE_]))
    argv[define._MASS_] = 0.0
    argv[define._TOL_] = 1e-8
    kappa = 1 / (2 * argv[define._MASS_] + 8)
    print('My rank is ', define.rank)
    print("Parameters:", params)
    print("Args:", argv)
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
    clover_dslash_eo_params = params.copy()
    clover_dslash_eo_params[define._SET_INDEX_] = 5
    clover_dslash_eo_params[define._SET_PLAN_] = define._SET_PLAN2_
    clover_dslash_eo_params[define._PARITY_] = define._EVEN_
    clover_dslash_eo_params[define._DAGGER_] = define._NO_USE_
    qcu.applyInitQcu(set_ptrs, clover_dslash_eo_params, argv)
    clover_dslash_oe_params = params.copy()
    clover_dslash_oe_params[define._SET_INDEX_] = 6
    clover_dslash_oe_params[define._SET_PLAN_] = define._SET_PLAN2_
    clover_dslash_oe_params[define._PARITY_] = define._ODD_
    clover_dslash_oe_params[define._DAGGER_] = define._NO_USE_
    qcu.applyInitQcu(set_ptrs, clover_dslash_oe_params, argv)
    print("Set pointers:", set_ptrs)
    print("Set pointers data:", set_ptrs.data)
    U_eo = io.xxxtzyx2pxxxtzyx(cp.array(U.cpu().numpy()))
    U_eo = io.pccdtzyx2ccdptzyx(U_eo)
    U_eo = U_eo.copy()  # DEBUG!!!
    print(f"U_eo:{U_eo}")
    _clover_even = cp.zeros((define._LAT_S_, define._LAT_C_, define._LAT_S_, define._LAT_C_,
                            params[define._LAT_T_], params[define._LAT_Z_], params[define._LAT_Y_], int(params[define._LAT_X_]/define._LAT_P_),), dtype=U_eo.dtype)
    _clover_odd = cp.zeros((define._LAT_S_, define._LAT_C_, define._LAT_S_, define._LAT_C_,
                           params[define._LAT_T_], params[define._LAT_Z_], params[define._LAT_Y_], int(params[define._LAT_X_]/define._LAT_P_),), dtype=U_eo.dtype)
    qcu.applyCloverQcu(_clover_even, U_eo, set_ptrs, clover_dslash_eo_params)
    qcu.applyCloverQcu(_clover_odd, U_eo, set_ptrs, clover_dslash_oe_params)
    _inverse_clover_term = cp.array([_clover_even.get(), _clover_odd.get()])
    _inverse_clover_term = io.pxxxtzyx2xxxtzyx(_inverse_clover_term)
    print(f"_inverse_clover_term.shape:{_inverse_clover_term.shape}")

    def _dslash_eo(src, U):
        dest = cp.zeros_like(src)
        qcu.applyWilsonDslashQcu(
            dest, src, U, set_ptrs, wilson_dslash_eo_params)
        return dest

    def _dslash_oe(src, U):
        dest = cp.zeros_like(src)
        qcu.applyWilsonDslashQcu(
            dest, src, U, set_ptrs, wilson_dslash_oe_params)
        return dest

    def _dslash(src, U):
        print(
            f"src.type:{type(src)},src.dtype:{src.dtype},src.shape:{src.shape}")
        print(f"U.type:{type(U)},U.dtype:{U.dtype},U.shape:{U.shape}")
        U_eo = io.xxxtzyx2pxxxtzyx(U)
        U_eo = io.pccdtzyx2ccdptzyx(U_eo)
        print(
            f"U_eo type:{type(U_eo)},U_eo.dtype:{U_eo.dtype},U_eo.shape:{U_eo.shape}")
        # print(f"U_eo value:{U_eo}")
        src_eo = io.xxxtzyx2pxxxtzyx(src)
        src_eo = src_eo.copy()  # DEBUG!!!
        src_e = src_eo[define._EVEN_].copy()  # DEBUG!!!
        src_o = src_eo[define._ODD_].copy()  # DEBUG!!!
        dest = cp.zeros_like(src_eo)
        U_eo = U_eo.copy()  # DEBUG!!!
        dest[define._EVEN_] = src_e - kappa*_dslash_eo(src_o, U_eo)
        dest[define._ODD_] = src_o-kappa * _dslash_oe(src_e, U_eo)
        return io.pxxxtzyx2xxxtzyx(dest)
    # gauge.test_su3(U[:, :, -1, -1, -1, -1, -1])
    _dest = _dslash(cp.array(src.cpu().numpy()), cp.array(U.cpu().numpy()))
    _dest = torch.tensor(
        data=_dest.get(), device=dest.device, dtype=dest.dtype)
    # print(f"dest value:{dest}")
    print(f"dest norm value:{torch.linalg.norm(dest)}")
    # print(f"_dest value:{_dest}")
    print(f"_dest norm value:{torch.linalg.norm(_dest)}")
    print(
        f"torch.linalg.norm(dest-_dest)/torch.linalg.norm(dest):{torch.linalg.norm(dest-_dest)/torch.linalg.norm(dest)}")
    # print(f"dest - _dest value:{dest-_dest}")
    clover_term = clover.make_clover(U=U)
    inverse_clover_term = clover.add_eye(clover=clover_term)
    inverse_clover_term = clover.inverse(clover=clover_term)
    _inverse_clover_term = torch.tensor(
        data=_inverse_clover_term.get(), device=inverse_clover_term.device, dtype=inverse_clover_term.dtype)
    print(f"inverse_clover_term:{inverse_clover_term}")
    print(f"_inverse_clover_term:{_inverse_clover_term}")
    print(
        f"inverse_clover_term-_inverse_clover_term:{inverse_clover_term-_inverse_clover_term}")
    print(
        f"torch.linalg.norm(inverse_clover_term-_inverse_clover_term)/torch.linalg.norm(inverse_clover_term):{torch.linalg.norm(inverse_clover_term-_inverse_clover_term)/torch.linalg.norm(inverse_clover_term)}")
