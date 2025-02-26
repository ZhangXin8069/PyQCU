# %% [markdown]
# # Init for pyqcu.

# %%
import cupy as cp
import numpy as np
from pyqcu import define
from pyqcu import io
from pyqcu import qcu
from pyqcu import eigen, cg
from opt_einsum import contract
from pyqcu.set import params, argv, set_ptrs
params[define._NODE_RANK_] = define.rank
params[define._NODE_SIZE_] = define.size
kappa = 1 / (2 * argv[define._MASS_] + 8)
print('My rank is ', define.rank)
gauge_filename = f"quda_wilson-bistabcg-gauge_-{params[define._LAT_X_]}-{params[define._LAT_Y_]}-{params  [define._LAT_Z_]}-{params[define._LAT_T_]}-{params[define._LAT_XYZT_]}-{params[define._GRID_X_]}-{params[define._GRID_Y_]}-{params[define._GRID_Z_]}-{params[define._GRID_T_]}-{params[define._PARITY_]}-{params[define._NODE_RANK_]}-{params[define._NODE_SIZE_]}-{params[define._DAGGER_]}-f.h5"
print("Parameters:", params)

# %%
wilson_cg_params = params.copy()
wilson_cg_params[define._SET_INDEX_] = 0
wilson_cg_params[define._SET_PLAN_] = define._SET_PLAN1_
qcu.applyInitQcu(set_ptrs, wilson_cg_params, argv)

# %%
wilson_dslash_eo_params = params.copy()
wilson_dslash_eo_params[define._SET_INDEX_] = 1
wilson_dslash_eo_params[define._SET_PLAN_] = define._SET_PLAN0_
wilson_dslash_eo_params[define._PARITY_] = define._EVEN_
wilson_dslash_eo_params[define._DAGGER_] = define._NO_USE_
qcu.applyInitQcu(set_ptrs, wilson_dslash_eo_params, argv)

# %%
wilson_dslash_eo_dag_params = params.copy()
wilson_dslash_eo_dag_params[define._SET_INDEX_] = 2
wilson_dslash_eo_dag_params[define._SET_PLAN_] = define._SET_PLAN0_
wilson_dslash_eo_dag_params[define._PARITY_] = define._EVEN_
wilson_dslash_eo_dag_params[define._DAGGER_] = define._USE_
qcu.applyInitQcu(set_ptrs, wilson_dslash_eo_dag_params, argv)

# %%
wilson_dslash_oe_params = params.copy()
wilson_dslash_oe_params[define._SET_INDEX_] = 3
wilson_dslash_oe_params[define._SET_PLAN_] = define._SET_PLAN0_
wilson_dslash_oe_params[define._PARITY_] = define._ODD_
wilson_dslash_oe_params[define._DAGGER_] = define._NO_USE_
qcu.applyInitQcu(set_ptrs, wilson_dslash_oe_params, argv)

# %%
wilson_dslash_oe_dag_params = params.copy()
wilson_dslash_oe_dag_params[define._SET_INDEX_] = 4
wilson_dslash_oe_dag_params[define._SET_PLAN_] = define._SET_PLAN0_
wilson_dslash_oe_dag_params[define._PARITY_] = define._ODD_
wilson_dslash_oe_dag_params[define._DAGGER_] = define._USE_
qcu.applyInitQcu(set_ptrs, wilson_dslash_oe_dag_params, argv)

# %%
print("Set pointers:", set_ptrs)
print("Set pointers data:", set_ptrs.data)

# %% [markdown]
# # Read from hdf5 files.

# %%
print("Gauge filename:", gauge_filename)
gauge = io.hdf5_xxxtzyx2grid_xxxtzyx(params, gauge_filename)
fermion_in_filename = gauge_filename.replace("gauge", "fermion-in")
print("Fermion in filename:", fermion_in_filename)
fermion_in = io.hdf5_xxxtzyx2grid_xxxtzyx(
    params, fermion_in_filename)
fermion_out_filename = gauge_filename.replace("gauge", "fermion-out")
print("Fermion out filename:", fermion_out_filename)
quda_fermion_out = io.hdf5_xxxtzyx2grid_xxxtzyx(
    params, fermion_out_filename)
fermion_out = cp.zeros_like(fermion_in)
print("Fermion out data:", fermion_out.data)
print("Fermion out shape:", fermion_out.shape)
eigenvalues_filename = gauge_filename.replace("gauge", "eigenvalues")
print("Eigenvalues filename:", eigenvalues_filename)
eigenvalues = io.hdf5_xxx2xxx(file_name=eigenvalues_filename)
print("Eigenvalues data:", eigenvalues.data)
print("Eigenvalues shape:", eigenvalues.shape)
eigenvectors_filename = gauge_filename.replace("gauge", "eigenvectors")
print("Eigenvectors filename:", eigenvectors_filename)
eigenvectors = io.eigenvectors2esctzyx(
    params=params, eigenvectors=io.hdf5_xxx2xxx(file_name=eigenvectors_filename))
print("Eigenvectors data:", eigenvectors.data)
print("Eigenvectors shape:", eigenvectors.shape)

# %% [markdown]
# # Run wilson bistabcg from pyqcu test.

# %%
qcu.applyWilsonBistabCgQcu(fermion_out, fermion_in,
                           gauge, set_ptrs, wilson_cg_params)
# qcu.applyWilsonCgQcu(fermion_out, fermion_in,
#                            gauge, set_ptrs, wilson_cg_params)
print("Fermion out data:", fermion_out.data)
print("Fermion out shape:", fermion_out.shape)
print("QUDA Fermion out data:", quda_fermion_out.data)
print("QUDA Fermion out shape:", quda_fermion_out.shape)
print("Difference:", cp.linalg.norm(fermion_out -
      quda_fermion_out)/cp.linalg.norm(quda_fermion_out))

# %% [markdown]
# # Give CG Dslash
# > src_o-set_ptr->kappa()**2*dslash_oe(dslash_eo(src_o))

# %%
def cg_dslash_no_dag(src):
    tmp0 = cp.zeros_like(src)
    tmp1 = cp.zeros_like(src)
    qcu.applyWilsonDslashQcu(
        tmp0, src, gauge, set_ptrs, wilson_dslash_eo_params)
    qcu.applyWilsonDslashQcu(
        tmp1, tmp0, gauge, set_ptrs, wilson_dslash_oe_params)
    return src-kappa**2*tmp1


def cg_dslash_dag(src):
    tmp0 = cp.zeros_like(src)
    tmp1 = cp.zeros_like(src)
    qcu.applyWilsonDslashQcu(
        tmp0, src, gauge, set_ptrs, wilson_dslash_eo_dag_params)
    qcu.applyWilsonDslashQcu(
        tmp1, tmp0, gauge, set_ptrs, wilson_dslash_oe_dag_params)
    return src-kappa**2*tmp1


def cg_dslash(src):
    return cg_dslash_dag(cg_dslash_no_dag(src))


def matvec(src):
    return cg_dslash(src)

# %% [markdown]
# # Run matvec(eigenvector[.]) ?= eigenvalue[.]*eigenvector[.] for eigen test

# %%
for i, ev in enumerate(eigenvalues):
    print(f"λ_{i} = {ev:.2e}")
    # Verify eigenvector
    v = eigenvectors[i]
    w = cp.zeros_like(v)
    w = matvec(v)
    error = cp.linalg.norm(w - ev * v) / cp.linalg.norm(w)
    print(f"Relative error: {error:.2e}")
    j = i+1
    if j == len(eigenvalues):
        j = 0
    print(
        f"Diff between λ_{i} and λ_{j}: {cp.linalg.norm(eigenvectors[i] - eigenvectors[j])/cp.linalg.norm(eigenvectors[i]):.2e}")

# %% [markdown]
# # Give guage's eigenvalues and eigenvectors to hdf5 files. (pass, don't run this)

# %%
# eigen_solver = eigen.solver(
#     n=params[define._LAT_XYZT_] * define._LAT_HALF_SC_, k=define._LAT_Ne_,matvec=matvec,dtype=gauge.dtype)
# eigenvalues, eigenvectors = eigen_solver.run()
# io.xxx2hdf5_xxx(
#     eigenvalues, params, gauge_filename.replace("gauge", "eigenvalues"))
# io.xxx2hdf5_xxx(
#     eigenvectors, params, gauge_filename.replace("gauge", "eigenvectors"))

# %% [markdown]
# # Origin CG

# %%
b_e = fermion_in[define._EVEN_].flatten()
b_o = fermion_in[define._ODD_].flatten()
b__o = cp.zeros_like(b_o)
tmp = cp.zeros_like(b_o)
# b__o=b_o+kappa*D_oe(b_e)
qcu.applyWilsonDslashQcu(tmp, b_e, gauge, set_ptrs, wilson_dslash_oe_params)
b__o = b_o+kappa*tmp
# b__o -> Dslash^dag b__o
b__o = cg_dslash_dag(b__o)
# Dslash(x_o)=b__o
cg_solver = cg.slover(b=b__o, matvec=matvec, tol=1e-10, max_iter=1000000)
x_o = cg_solver.run()
# x_e  =b_e+kappa*D_eo(x_o)
qcu.applyWilsonDslashQcu(tmp, x_o, gauge, set_ptrs, wilson_dslash_eo_params)
x_e = b_e+kappa*tmp
# give qcu_fermion_out
qcu_fermion_out = cp.zeros_like(quda_fermion_out)
qcu_fermion_out[define._EVEN_] = x_e.reshape(
    quda_fermion_out[define._EVEN_].shape)
qcu_fermion_out[define._ODD_] = x_o.reshape(
    quda_fermion_out[define._ODD_].shape)

# %%
np.linalg.norm(qcu_fermion_out-quda_fermion_out) / \
    np.linalg.norm(quda_fermion_out)

# %% [markdown]
# # MultiGrid - give grids

# %%
_eigenvectors = io.xxxtzyx2mg_xxxtzyx(input_array=eigenvectors, params=params)
_eigenvectors.shape  # escTtZzYyXx

# %%
def orthogonalize(eigenvectors):
    _eigenvectors = eigenvectors.copy()
    size_e, size_s, size_c, size_T, size_t, size_Z, size_z, size_Y, size_y, size_X, size_x = eigenvectors.shape
    print(size_e, size_s, size_c, size_T, size_t,
          size_Z, size_z, size_Y, size_y, size_X, size_x)
    for T in range(size_T):
        for Z in range(size_Z):
            for Y in range(size_Y):
                for X in range(size_X):
                    origin_matrix = eigenvectors[:,
                                                 :, :, T, :, Z, :, Y, :, X, :]
                    _shape = origin_matrix.shape
                    _origin_matrix = origin_matrix.reshape(size_e, -1)
                    condition_number = np.linalg.cond(_origin_matrix.get())
                    print(f"矩阵条件数: {condition_number}")
                    a = _origin_matrix[:, 0]
                    b = _origin_matrix[:, -1]
                    print(cp.dot(a.conj(), b))
                    Q = cp.linalg.qr(_origin_matrix.T)[0]
                    condition_number = np.linalg.cond(Q.get())
                    print(f"矩阵条件数: {condition_number}")
                    a = Q[:, 0]
                    b = Q[:, -1]
                    print(cp.dot(a.conj(), b))
                    _eigenvectors[:, :, :, T, :, Z, :, Y, :, X, :] = Q.T.reshape(
                        _shape)
    return _eigenvectors


orth_eigenvectors = orthogonalize(_eigenvectors)

# %%
orth_eigenvectors.shape

# %%
testvectors = orth_eigenvectors
_src = io.xxxtzyx2mg_xxxtzyx(
    input_array=fermion_out[define._EVEN_], params=params)

# %% [markdown]
# # MultiGrid - R*vector
# ![](./image0-dev40.png)

# %%
r_src = _src


def r_vec(src):
    return contract("escTtZzYyXx,scTtZzYyXx->eTZYX", testvectors, src)


r_dest = r_vec(r_src)

# %%
r_dest.shape

# %% [markdown]
# # MultiGrid - P*vector
# ![](./image1-dev40.png)
# 

# %%
p_src = r_dest


def p_vec(src):
    return contract("escTtZzYyXx,eTZYX->scTtZzYyXx", cp.conj(testvectors), src)


p_dest = p_vec(p_src)

# %%
p_dest.shape

# %% [markdown]
# # MultiGrid - verify above
# ![](./image2-dev40.png)

# %%
print(cp.linalg.norm(r_src))
print(cp.linalg.norm(p_dest))

# %%
print(cp.linalg.norm(r_src-p_dest)/cp.linalg.norm(r_src))

# %%
print(cp.linalg.norm(r_src-p_vec(r_vec(r_src)))/cp.linalg.norm(r_src))

# %%
r_src.flatten()[:50]

# %%
p_dest.flatten()[:50]

# %%
cp.linalg.norm(r_src-p_dest)/cp.linalg.norm(r_src)

# %%
_mat = contract("escTtZzYyXx,escTtZzYyXx->scTtZzYyXx",
                testvectors, cp.conj(testvectors)).flatten()
print(cp.linalg.norm(_mat))
print(_mat[:100])

# %%


# %% [markdown]
# # End for pyqcu. (pass, don't run this)

# %%
# qcu.applyEndQcu(set_ptrs, params)
# qcu.applyEndQcu(set_ptrs, wilson_dslash_eo_params)
# qcu.applyEndQcu(set_ptrs, wilson_dslash_oe_params)
# qcu.applyEndQcu(set_ptrs, wilson_dslash_eo_dag_params)
# qcu.applyEndQcu(set_ptrs, wilson_dslash_oe_dag_params)


