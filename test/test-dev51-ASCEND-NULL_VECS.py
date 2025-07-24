import torch
from pyqcu.ascend import dslash_parity
from pyqcu.ascend import inverse
dof = 12
latt_size = (8, 8, 8, 8)
latt_size = (4, 4, 4, 4)
kappa = 0.125
# dtype = torch.complex128
dtype = torch.complex64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Initialize lattice gauge theory
wilson = dslash_parity.wilson_parity(
    latt_size=latt_size,
    kappa=kappa,
    dtype=dtype,
    device=device,
    verbose=False
)
clover = dslash_parity.clover_parity(
    latt_size=latt_size,
    kappa=kappa,
    dtype=dtype,
    device=device,
    verbose=False
)
U = wilson.generate_gauge_field(sigma=0.1, seed=42)
null_vectors = torch.randn(dof, 4, 3, latt_size[3], latt_size[2], latt_size[1], latt_size[0],
                           dtype=dtype, device=device)
clover_term = clover.make_clover(U=U)


def matvec(src: torch.Tensor, U: torch.Tensor = U) -> torch.Tensor:
    return wilson.give_wilson(src, U)+clover.give_clover(clover=clover_term, src=src)
    # return wilson.give_wilson(src, U)

result = inverse.give_null_vecs(
    null_vecs=null_vectors,
    matvec=matvec,
)