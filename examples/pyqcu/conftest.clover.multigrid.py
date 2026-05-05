import torch
from pyqcu import tools, dslash, lattice, solver
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
params[define._VERBOSE_] = 1
params[define._SEED_] = 42
argv = argv.to(dtype=define.dtype(params[define._DATA_TYPE_]).to_real())
argv[define._MASS_] = 0.05
argv[define._ATOL_] = 1e-9
argv[define._SIGMA_] = 0.1
print(params)
print(argv)
print(set_ptrs)
gauge_eo = torch.zeros(size=[2, 3, 3, 4]+define.lat_shape(params)).to(
    dtype=define.dtype(params[define._DATA_TYPE_]), device=torch.device('cuda'))
fermion_in_eo = torch.zeros(size=[2, 4, 3]+define.lat_shape(params)).to(
    dtype=define.dtype(params[define._DATA_TYPE_]), device=torch.device('cuda'))
fermion_in_eo = torch.rand_like(fermion_in_eo)
fermion_out_eo = torch.zeros(size=[2, 4, 3]+define.lat_shape(params)).to(
    dtype=define.dtype(params[define._DATA_TYPE_]), device=torch.device('cuda'))
params[define._VERBOSE_] = 1
params[define._SET_INDEX_] = 0
params[define._SET_PLAN_] = -1
params[define._PARITY_] = 0
qcu.applyInitQcu(set_ptrs, params, argv)
qcu.applyGaussGaugeQcu(gauge_eo, set_ptrs, params)
print(lattice.check_su3(U=gauge_eo[0]))
print(lattice.check_su3(U=gauge_eo[1]))
print(set_ptrs)
clover_ee = torch.zeros(size=[4, 3, 4, 3]+define.lat_shape(params)).to(
    dtype=define.dtype(params[define._DATA_TYPE_]), device=torch.device('cuda'))
clover_ee_inv = torch.zeros(size=[4, 3, 4, 3]+define.lat_shape(params)).to(
    dtype=define.dtype(params[define._DATA_TYPE_]), device=torch.device('cuda'))
clover_oo = torch.zeros(size=[4, 3, 4, 3]+define.lat_shape(params)).to(
    dtype=define.dtype(params[define._DATA_TYPE_]), device=torch.device('cuda'))
clover_oo_inv = torch.zeros(size=[4, 3, 4, 3]+define.lat_shape(params)).to(
    dtype=define.dtype(params[define._DATA_TYPE_]), device=torch.device('cuda'))
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
pyqcu_U = tools.poooxyzt2oooxyzt(input_array=gauge_eo)
pyqcu_src = tools.poooxyzt2oooxyzt(input_array=fermion_in_eo)
refer_clover_term = dslash.make_clover(
    U=pyqcu_U, kappa=1 / (2 * argv[define._MASS_] + 8))
clover_eeoo = torch.zeros(size=[2, 4, 3, 4, 3]+define.lat_shape(params)).to(
    dtype=define.dtype(params[define._DATA_TYPE_]), device=torch.device('cuda'))
clover_eeoo[0] = clover_ee
clover_eeoo[1] = clover_oo
pyqcu_clover_term = tools.poooxyzt2oooxyzt(
    input_array=clover_eeoo)
pyqcu_clover_term = dslash.cut_I(clover_term=pyqcu_clover_term)
print('pyqcu_clover_term:', pyqcu_clover_term.flatten()[:100])
print('refer_clover_term:', refer_clover_term.flatten()[:100])
print('Difference:', tools.norm(refer_clover_term -
      pyqcu_clover_term)/tools.norm(pyqcu_clover_term))
# params[define._VERBOSE_] = 1
# params[define._SET_INDEX_] += 1
# params[define._SET_PLAN_] = 1
# params[define._PARITY_] = 0
# qcu.applyInitQcu(set_ptrs, params, argv)
# for i in range(10):
#     fermion_out_eo = torch.zeros_like(fermion_out_eo)
#     qcu.applyCloverBistabCgQcu(fermion_out_eo, fermion_in_eo, gauge_eo,
#                                clover_ee, clover_oo, clover_ee_inv, clover_oo_inv,  set_ptrs, params)
#     print(set_ptrs)
#     pyqcu_dest = tools.poooxyzt2oooxyzt(input_array=fermion_out_eo)
#     refer_src = dslash.give_wilson(
#         src=pyqcu_dest, U=pyqcu_U, kappa=1 / (2 * argv[define._MASS_] + 8), with_I=True)+dslash.give_clover(src=pyqcu_dest, clover_term=refer_clover_term)
#     print('pyqcu_src:', pyqcu_src.flatten()[:100])
#     print('refer_src:', refer_src.flatten()[:100])
#     print('Difference:', tools.norm(refer_src-pyqcu_src)/tools.norm(pyqcu_src))
# print("gauge_eo.is_contiguous():", gauge_eo.is_contiguous())
# print("fermion_in_eo.is_contiguous():", fermion_in_eo.is_contiguous())
# print("fermion_in_out.is_contiguous():", fermion_out_eo.is_contiguous())
# print("pyqcu_src.is_contiguous():", pyqcu_src.is_contiguous())
# print("pyqcu_dest.is_contiguous():", pyqcu_dest.is_contiguous())
# for i in range(10):
#     fermion_out_eo = torch.zeros_like(fermion_out_eo)
#     fermion_in_e = fermion_in_eo[0]
#     fermion_in_o = fermion_in_eo[1]
#     fermion_out_e = fermion_out_eo[0]
#     fermion_out_o = fermion_out_eo[1]
#     dest_o = fermion_out_eo[1].clone()
#     operator = dslash.operator(
#         U=pyqcu_U, clover_term=refer_clover_term, kappa=1 / (2 * argv[define._MASS_] + 8), verbose=True, support_parity=True)

#     def matvec(src_o):
#         qcu.applyCloverBistabCgDslashQcu(dest_o, src_o, gauge_eo,
#                                          clover_ee, clover_oo, clover_ee_inv, clover_oo_inv,  set_ptrs, params)
#         return dest_o
#     fermion_out_o = solver.bistabcg(b=operator.give_b_parity4fermion(
#         fermion_in_e=fermion_in_e, fermion_in_o=fermion_in_o), matvec=matvec)
#     fermion_out_e = operator.give_x_e4fermion(
#         fermion_in_e=fermion_in_e, fermion_out_o=fermion_out_o)
#     fermion_out_eo[0] = fermion_out_e
#     fermion_out_eo[1] = fermion_out_o
#     pyqcu_dest = tools.poooxyzt2oooxyzt(input_array=fermion_out_eo)
#     refer_src = dslash.give_wilson(
#         src=pyqcu_dest, U=pyqcu_U, kappa=1 / (2 * argv[define._MASS_] + 8), with_I=True)+dslash.give_clover(src=pyqcu_dest, clover_term=refer_clover_term)
#     print('pyqcu_src:', pyqcu_src.flatten()[:100])
#     print('refer_src:', refer_src.flatten()[:100])
#     print('Difference:', tools.norm(refer_src-pyqcu_src)/tools.norm(pyqcu_src))
# print("gauge_eo.is_contiguous():", gauge_eo.is_contiguous())
# print("fermion_in_eo.is_contiguous():", fermion_in_eo.is_contiguous())
# print("fermion_in_out.is_contiguous():", fermion_out_eo.is_contiguous())
# print("pyqcu_src.is_contiguous():", pyqcu_src.is_contiguous())
# print("pyqcu_dest.is_contiguous():", pyqcu_dest.is_contiguous())

mg = solver.multigrid(dtype_list=[pyqcu_U.dtype]*10, device_list=[pyqcu_U.device]*10, U=pyqcu_U,
                      clover_term=refer_clover_term, kappa=1 / (2 * argv[define._MASS_] + 8), clover_ee_inv=clover_ee_inv, clover_oo_inv=clover_oo_inv, tol=1e-6, max_iter=1000, max_level=3, num_restart=3, support_parity=True, verbose=True)
mg.init()
for i in range(1):
    pyqcu_dest = mg.solve(b=pyqcu_src)
    refer_src = dslash.give_wilson(
        src=pyqcu_dest, U=pyqcu_U, kappa=1 / (2 * argv[define._MASS_] + 8), with_I=True)+dslash.give_clover(src=pyqcu_dest, clover_term=refer_clover_term)
    print('pyqcu_src:', pyqcu_src.flatten()[:100])
    print('refer_src:', refer_src.flatten()[:100])
    print('Difference:', tools.norm(refer_src-pyqcu_src)/tools.norm(pyqcu_src))
mg.plot()
print("gauge_eo.is_contiguous():", gauge_eo.is_contiguous())
print("fermion_in_eo.is_contiguous():", fermion_in_eo.is_contiguous())
print("fermion_in_out.is_contiguous():", fermion_out_eo.is_contiguous())
print("pyqcu_src.is_contiguous():", pyqcu_src.is_contiguous())
print("pyqcu_dest.is_contiguous():", pyqcu_dest.is_contiguous())
exit()
