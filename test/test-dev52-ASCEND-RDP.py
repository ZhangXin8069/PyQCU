import torch
from pyqcu.ascend import dslash
from pyqcu.ascend import inverse
dof = 12
# latt_size = (16, 16, 16, 16)
# latt_size = (8, 8, 8, 8)
latt_size = (4, 4, 4, 4)
latt_size = (4, 4, 4, 8)
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


def matvec(src: torch.Tensor, U: torch.Tensor = U) -> torch.Tensor:
    return wilson.give_wilson(src, U)+clover.give_clover(clover=clover_term, src=src)
    return wilson.give_wilson(src, U)


null_vecs = inverse.give_null_vecs(
    null_vecs=null_vecs,
    matvec=matvec,
    verbose=False,
    normalize=True
)

# for i in range(dof):
#     print(f"A*v/v check again:")
#     Av = matvec(null_vecs[i])
#     print(f"  Vector {i}: ||A*v|| = {torch.norm(Av).item():.6e}")
#     print(
#         f"  Vector {i}: v = {null_vecs[i]}")
#     print(
#         f"  Vector {i}: A*v = {Av}")
#     print(
#         f"  Vector {i}: A*v/v = {Av/null_vecs[i]}")
#     print(
#         f"torch.norm(null_vecs[{i}]).item():.6e:{torch.norm(null_vecs[i]).item():.6e}")
#     # orthogonalization
#     for k in range(0, i+1):
#         print(
#             f"torch.vdot(null_vecs[{i}].flatten(), null_vecs[{k}].flatten()):{torch.vdot(null_vecs[i].flatten(), null_vecs[k].flatten())}")

local_ortho_null_vecs=inverse.local_orthogonalize(null_vecs=null_vecs,normalize=True)