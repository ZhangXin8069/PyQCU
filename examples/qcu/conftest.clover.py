import torch
from pyqcu import tools, dslash, lattice
from pyqcu.cuda import qcu, define
from pyqcu.cuda.define import params, argv, set_ptrs
params[define._LAT_X_] = 4
params[define._LAT_Y_] = 4
params[define._LAT_Z_] = 4
params[define._LAT_T_] = 8
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





params[define._VERBOSE_] = 1
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 1
params[define._PARITY_] = 0
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyWilsonBistabCgQcu(
    fermion_out_eo, fermion_in_eo, gauge_eo, set_ptrs, params)





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
clover_oo = torch.zeros(size=[4, 3, 4, 3]+[params[define._LAT_X_], params[define._LAT_Y_], params[define._LAT_Z_],
                                           params[define._LAT_T_]//define._LAT_P_]).to(dtype=define.dtype(params[define._DATA_TYPE_]), device=torch.device('cuda'))
qcu_clover_term_eo = torch.zeros(size=[2, 4, 3, 4, 3]+[params[define._LAT_X_], params[define._LAT_Y_], params[define._LAT_Z_],
                                                params[define._LAT_T_]//define._LAT_P_]).to(dtype=define.dtype(params[define._DATA_TYPE_]), device=torch.device('cuda'))

# _gauge_eo = gauge_eo.flatten().clone()
# print('_gauge_eo.shape', _gauge_eo.shape)
# for p in range(2):
#     for c0 in range(3):
#         for c1 in range(3):
#             for d in range(4):
#                 for x in range(params[define._LAT_X_]):
#                     for y in range(params[define._LAT_Y_]):
#                         for z in range(params[define._LAT_Z_]):
#                             for t in range(params[define._LAT_T_]//2):
#                                 _gauge_eo[(((((((((((p)*3)+c0)*3)+c1)*4)+d)*params[define._LAT_X_])+x)*params[define._LAT_Y_]+y)*params[define._LAT_Z_]+z)
#                                        * params[define._LAT_T_]//2+t].real = (((((((60+p)*10)+c0)*10)+c1)*10)+d)
#                                 _gauge_eo[(((((((((((p)*3)+c0)*3)+c1)*4)+d)*params[define._LAT_X_])+x)*params[define._LAT_Y_]+y)
#                                         * params[define._LAT_Z_]+z)*params[define._LAT_T_]//2+t].imag = ((((90+x)*10+y)*10+z)*10+t)
# gauge_eo = _gauge_eo.reshape(gauge_eo.shape)


params[define._VERBOSE_] = 1
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 2
params[define._PARITY_] = 0
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyCloverQcu(clover_ee, gauge_eo, set_ptrs, params)


params[define._VERBOSE_] = 1
params[define._SET_INDEX_] += 1
params[define._SET_PLAN_] = 2
params[define._PARITY_] = 1
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyCloverQcu(clover_oo, gauge_eo, set_ptrs, params)


refer_clover_term = dslash.make_clover(
    U=qcu_U, kappa=1 / (2 * argv[define._MASS_] + 8))
refer_clover_term_eo = tools.oooxyzt2poooxyzt(
    input_array=refer_clover_term)

qcu_clover_term_eo[0] = clover_ee.clone()
qcu_clover_term_eo[1] = clover_oo.clone()
qcu_clover_term_eo = dslash.cut_I(clover_term=qcu_clover_term_eo)
qcu_clover_term = tools.poooxyzt2oooxyzt(
    input_array=qcu_clover_term_eo)

print('qcu_clover_term_eo[0]:', qcu_clover_term_eo[0].flatten()[:100])
print('refer_clover_term_eo[0]:', refer_clover_term_eo[0].flatten()[:100])
print('Difference:', tools.norm(refer_clover_term_eo[0] -
      qcu_clover_term_eo[0])/tools.norm(qcu_clover_term_eo[0]))

print('qcu_clover_term_eo[1]:', qcu_clover_term_eo[1].flatten()[:100])
print('refer_clover_term_eo[1]:', refer_clover_term_eo[1].flatten()[:100])
print('Difference:', tools.norm(refer_clover_term_eo[1] -
      qcu_clover_term_eo[1])/tools.norm(qcu_clover_term_eo[1]))

print('qcu_clover_term:', qcu_clover_term.flatten()[:100])
print('refer_clover_term:', refer_clover_term.flatten()[:100])
print('Difference:', tools.norm(refer_clover_term -
      qcu_clover_term)/tools.norm(qcu_clover_term))


print("gauge_eo.is_contiguous():", gauge_eo.is_contiguous())
print("fermion_in_eo.is_contiguous():", fermion_in_eo.is_contiguous())
print("fermion_in_out.is_contiguous():", fermion_out_eo.is_contiguous())
print("qcu_src.is_contiguous():", qcu_src.is_contiguous())
print("qcu_dest.is_contiguous():", qcu_dest.is_contiguous())
