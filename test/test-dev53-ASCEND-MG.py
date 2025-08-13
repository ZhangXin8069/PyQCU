import torch
from pyqcu.ascend import dslash
from pyqcu.ascend.include import *
from pyqcu.ascend import inverse
dof = 12
# latt_size = (16, 16, 16, 16)
latt_size = (4, 8, 8, 8)
# latt_size = (8, 8, 8, 8)
# latt_size = (4, 4, 4, 4)
kappa = 0.125
# dtype = torch.complex128
dtype = torch.complex64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Initialize lattice gauge theory
wilson = dslash.wilson_parity(
    latt_size=latt_size,
    kappa=kappa,
    dtype=dtype,
    device=device,
    verbose=False
)
clover = dslash.clover_parity(
    latt_size=latt_size,
    kappa=kappa,
    dtype=dtype,
    device=device,
    verbose=False
)
U = wilson.generate_gauge_field(sigma=0.1, seed=42)
null_vecs = torch.randn(dof, 4, 3, latt_size[3], latt_size[2], latt_size[1], latt_size[0],
                        dtype=dtype, device=device)
clover_term = clover.make_clover(U=U)


# def matvec(src: torch.Tensor, U: torch.Tensor = U) -> torch.Tensor:
#     # return wilson.give_wilson(src, U)
#     return wilson.give_wilson(src, U)+clover.give_clover(clover=clover_term, src=src)


# null_vecs = inverse.give_null_vecs(
#     null_vecs=null_vecs,
#     matvec=matvec,
#     verbose=False
# )

# local_ortho_null_vecs = inverse.local_orthogonalize(
#     null_vecs=null_vecs, verbose=False)

# fine_vec = torch.randn_like(null_vecs[0])
# print(f"fine_vec.shape:{fine_vec.shape}")
# print(f"fine_vec.flatten()[:100]:{fine_vec.flatten()[:100]}")
# coarse_vec = inverse.restrict(
#     local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=fine_vec)
# print(f"coarse_vec.shape:{coarse_vec.shape}")
# print(f"coarse_vec.flatten()[:100]:{coarse_vec.flatten()[:100]}")
# _fine_vec = inverse.prolong(
#     local_ortho_null_vecs=local_ortho_null_vecs, coarse_vec=coarse_vec).reshape(fine_vec.shape)
# print(f"_fine_vec.shape:{_fine_vec.shape}")
# print(f"_fine_vec.flatten()[:100]:{_fine_vec.flatten()[:100]}")
# print(
#     f"(fine_vec/_fine_vec).flatten()[:100]:{(fine_vec/_fine_vec).flatten()[:100]}")
# _coarse_vec = inverse.restrict(
#     local_ortho_null_vecs=local_ortho_null_vecs, fine_vec=_fine_vec).reshape(coarse_vec.shape)
# print(f"_coarse_vec.shape:{_coarse_vec.shape}")
# print(f"_coarse_vec.flatten()[:100]:{_coarse_vec.flatten()[:100]}")
# print(
#     f"(coarse_vec/_coarse_vec).flatten()[:100]:{(coarse_vec/_coarse_vec).flatten()[:100]}")
# inverse.demo()
b = torch.randn_like(null_vecs[0])
U_eo = xxxtzyx2pxxxtzyx(input_array=U)
clover_eo = xxxtzyx2pxxxtzyx(input_array=clover_term)
mg = inverse.mg(b=b, wilson=wilson, U_eo=U_eo,
                clover=clover, clover_eo=clover_eo)
