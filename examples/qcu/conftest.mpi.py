import torch
from pyqcu import tools, dslash, lattice
from pyqcu.cuda import qcu, define
from pyqcu.cuda.define import params, argv, set_ptrs
params[define._LAT_X_] = 4*4
params[define._LAT_Y_] = 4*4
params[define._LAT_Z_] = 4*4
params[define._LAT_T_] = 8*4
params[define._LAT_XYZT_] = params[define._LAT_X_] * \
    params[define._LAT_Y_]*params[define._LAT_Z_]*params[define._LAT_T_]
params[define._GRID_X_], params[define._GRID_Y_], params[define._GRID_Z_], params[
    define._GRID_T_] = tools.give_grid_size()
params[define._PARITY_] = 0
params[define._NODE_RANK_] = define.rank
params[define._NODE_SIZE_] = define.size
params[define._DAGGER_] = 0
params[define._MAX_ITER_] = 1000
params[define._DATA_TYPE_] = define._LAT_C64_
# params[define._DATA_TYPE_] = define._LAT_C128_
params[define._SET_INDEX_] = 0
params[define._SET_PLAN_] = 1
params[define._MG_X_] = 1
params[define._MG_Y_] = 1
params[define._MG_Z_] = 1
params[define._MG_T_] = 1
params[define._LAT_E_] = 24
params[define._VERBOSE_] = 1
params[define._SEED_] = 42
argv = argv.to(dtype=define.dtype(params[define._DATA_TYPE_]).to_real())
argv[define._MASS_] = 0.05
argv[define._TOL_] = 1e-9
argv[define._SIGMA_] = 0.1
print(params)
print(argv)
print(set_ptrs)
gauge_eo = torch.zeros(size=[2, 3, 3, 4]+[params[define._LAT_X_], params[define._LAT_Y_], params[define._LAT_Z_],
                       params[define._LAT_T_]//define._LAT_P_]).to(dtype=define.dtype(params[define._DATA_TYPE_]), device=torch.device('cuda'))
fermion_in_eo = torch.zeros(size=[2, 4, 3]+[params[define._LAT_X_], params[define._LAT_Y_], params[define._LAT_Z_],
                                            params[define._LAT_T_]//define._LAT_P_]).to(dtype=define.dtype(params[define._DATA_TYPE_]), device=torch.device('cuda'))
fermion_in_eo = torch.rand_like(fermion_in_eo)
fermion_out_eo = torch.zeros(size=[2, 4, 3]+[params[define._LAT_X_], params[define._LAT_Y_], params[define._LAT_Z_],
                             params[define._LAT_T_]//define._LAT_P_]).to(dtype=define.dtype(params[define._DATA_TYPE_]), device=torch.device('cuda'))
params[define._VERBOSE_] = 1
params[define._SET_INDEX_] = 0
params[define._SET_PLAN_] = -1
params[define._PARITY_] = 0
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyGaussGaugeQcu(gauge_eo, set_ptrs, params)
print(lattice.check_su3(U=gauge_eo[0]))
print(lattice.check_su3(U=gauge_eo[1]))
print(set_ptrs)
params[define._VERBOSE_] = 1
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 1
params[define._PARITY_] = 0
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyWilsonBistabCgQcu(
    fermion_out_eo, fermion_in_eo, gauge_eo, set_ptrs, params)
print(set_ptrs)
qcu_dest = tools.poooxyzt2oooxyzt(input_array=fermion_out_eo)
qcu_U = tools.poooxyzt2oooxyzt(input_array=gauge_eo)
qcu_src = tools.poooxyzt2oooxyzt(input_array=fermion_in_eo)
refer_src = dslash.give_wilson(
    src=qcu_dest, U=qcu_U, kappa=1 / (2 * argv[define._MASS_] + 8), with_I=True)
print('qcu_src:', qcu_src.flatten()[:100])
print('refer_src:', refer_src.flatten()[:100])
print('Difference:', tools.norm(refer_src-qcu_src)/tools.norm(qcu_src))
clover_ee = torch.zeros(size=[4, 3, 4, 3]+[params[define._LAT_X_], params[define._LAT_Y_], params[define._LAT_Z_],
                                           params[define._LAT_T_]//define._LAT_P_]).to(dtype=define.dtype(params[define._DATA_TYPE_]), device=torch.device('cuda'))
clover_ee_inv = torch.zeros(size=[4, 3, 4, 3]+[params[define._LAT_X_], params[define._LAT_Y_], params[define._LAT_Z_],
                                               params[define._LAT_T_]//define._LAT_P_]).to(dtype=define.dtype(params[define._DATA_TYPE_]), device=torch.device('cuda'))
clover_oo = torch.zeros(size=[4, 3, 4, 3]+[params[define._LAT_X_], params[define._LAT_Y_], params[define._LAT_Z_],
                                           params[define._LAT_T_]//define._LAT_P_]).to(dtype=define.dtype(params[define._DATA_TYPE_]), device=torch.device('cuda'))
clover_oo_inv = torch.zeros(size=[4, 3, 4, 3]+[params[define._LAT_X_], params[define._LAT_Y_], params[define._LAT_Z_],
                                               params[define._LAT_T_]//define._LAT_P_]).to(dtype=define.dtype(params[define._DATA_TYPE_]), device=torch.device('cuda'))
# gauge_eo = torch.ones_like(gauge_eo)*2
params[define._VERBOSE_] = 1
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 2
params[define._PARITY_] = 0
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyCloversQcu(clover_ee, clover_ee_inv, gauge_eo, set_ptrs, params)
print(set_ptrs)
params[define._VERBOSE_] = 1
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 2
params[define._PARITY_] = 1
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyCloversQcu(clover_oo, clover_oo_inv, gauge_eo, set_ptrs, params)
print(set_ptrs)
qcu_U = tools.poooxyzt2oooxyzt(input_array=gauge_eo)
qcu_src = tools.poooxyzt2oooxyzt(input_array=fermion_in_eo)
refer_clover_term = dslash.make_clover(
    U=qcu_U, kappa=1 / (2 * argv[define._MASS_] + 8))
clover_eeoo = torch.zeros(size=[2, 4, 3, 4, 3]+[params[define._LAT_X_], params[define._LAT_Y_], params[define._LAT_Z_],
                                                params[define._LAT_T_]//define._LAT_P_]).to(dtype=define.dtype(params[define._DATA_TYPE_]), device=torch.device('cuda'))
clover_eeoo[0] = clover_ee
clover_eeoo[1] = clover_oo
qcu_clover_term = tools.poooxyzt2oooxyzt(
    input_array=clover_eeoo)
qcu_clover_term = dslash.cut_I(clover_term=qcu_clover_term)
qcu_clover_term_eo = tools.oooxyzt2poooxyzt(
    input_array=qcu_clover_term)
refer_clover_term_eo = tools.oooxyzt2poooxyzt(
    input_array=refer_clover_term)
print('qcu_clover_term:', qcu_clover_term.flatten()[:100])
print('refer_clover_term:', refer_clover_term.flatten()[:100])
print('Difference:', tools.norm(refer_clover_term -
      qcu_clover_term)/tools.norm(qcu_clover_term))
params[define._VERBOSE_] = 1
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 2
params[define._PARITY_] = 0
qcu.applyInitQcu(set_ptrs, params, argv)
for i in range(10):
    fermion_out_eo = torch.zeros_like(fermion_out_eo)
    qcu.applyCloverBistabCgQcu(fermion_out_eo, fermion_in_eo, gauge_eo,
                               clover_ee, clover_oo, clover_ee_inv, clover_oo_inv,  set_ptrs, params)
    print(set_ptrs)
    qcu_dest = tools.poooxyzt2oooxyzt(input_array=fermion_out_eo)
    refer_src = dslash.give_wilson(
        src=qcu_dest, U=qcu_U, kappa=1 / (2 * argv[define._MASS_] + 8), with_I=True)+dslash.give_clover(src=qcu_dest, clover_term=refer_clover_term)
    print('qcu_src:', qcu_src.flatten()[:100])
    print('refer_src:', refer_src.flatten()[:100])
    print('Difference:', tools.norm(refer_src-qcu_src)/tools.norm(qcu_src))

print("gauge_eo.is_contiguous():", gauge_eo.is_contiguous())
print("fermion_in_eo.is_contiguous():", fermion_in_eo.is_contiguous())
print("fermion_in_out.is_contiguous():", fermion_out_eo.is_contiguous())
print("qcu_src.is_contiguous():", qcu_src.is_contiguous())
print("qcu_dest.is_contiguous():", qcu_dest.is_contiguous())
