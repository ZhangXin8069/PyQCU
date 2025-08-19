import lattice
import mg
import numpy as cp
import bicgstab
import time
nx = 64
ny = nx
nc = 2

# 生成随即组态
U = lattice.load_or_generate_U(nx, ny, nc, "./U_data/")
print("生成的矩阵形状：", U.shape)

# 确定格点
fine_op = lattice.operator_para(nx, ny, nc, U=U, if_fine=1)


V = cp.random.rand(nx, ny, nc*2).view(cp.complex128)
x0 = cp.random.rand(nx, ny, nc*2).view(cp.complex128)


Vout = lattice.apply_mat(V, fine_op)

start_time = time.perf_counter()
V0 = bicgstab.bicgstab(Vout, x0, op=fine_op, if_info=1, tol=1e-8)
end_time = time.perf_counter()
print(f"Execution time: {end_time - start_time} seconds")

start_time = time.perf_counter()
my_mg = mg.mg(fine_op, 2)

# V1 = my_mg.mg_bicgstab(Vout, if_info=1, tol=1e-8)
end_time = time.perf_counter()
print(f"Execution time: {end_time - start_time} seconds")
# print(V[0,0,0] - V1[0,0,0])
print(V[0, 0, 0] - V0[0, 0, 0])
