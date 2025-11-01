
from pyqcu.ascend import inverse
import torch
from pyqcu.ascend import dslash
from pyqcu.ascend.define import *
# latt_size = (32, 32, 32, 32)
# latt_size = (32, 32, 16, 16)
# latt_size = (16, 16, 16, 32)
# latt_size = (16, 16, 16, 16)
latt_size = (16, 16, 16, 8)
# latt_size = (16, 16, 8, 8)
# latt_size = (16, 8, 8, 8)
# latt_size = (8, 8, 8, 8)
# latt_size = (8, 16, 16, 32)
# latt_size = (16, 16, 16, 32)
# latt_size = (32, 32, 32, 32)
# latt_size = (32, 32, 32, 64)
# latt_size = (4, 8, 8, 8)
# latt_size = (8, 8, 8, 4)
# latt_size = (16, 8, 8, 8)
# latt_size = (8, 8, 8, 16)
# latt_size = (8, 8, 8, 8)
# latt_size = (4, 4, 4, 4)
# latt_size = (2, 2, 2, 2)
# mass = -3.5
# mass = -0.8
# mass = -0.5
# mass = 0.05
# mass = 0.0
mass = -0.05
# kappa = 0.4
# kappa = 0.125
# kappa = 0.5
kappa = 1 / (2 * mass + 8)
# dtype = torch.complex128
dtype = torch.complex64
# dtype = torch.complex32
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print(f"Using device: {device}")
# Initialize lattice gauge theory
wilson = dslash.wilson_mg(
    latt_size=latt_size,
    kappa=kappa,
    dtype=dtype,
    device=device,
    verbose=False
)
clover = dslash.clover(
    latt_size=latt_size,
    kappa=kappa,
    dtype=dtype,
    device=device,
    verbose=False
)
U = wilson.generate_gauge_field(sigma=0.1, seed=42)

wilson.check_su3(U)
clover_term = clover.make_clover(U=U)
# clover_term = torch.zeros_like(clover_term) # just for test, just wilson

b = torch.randn(4, 3, latt_size[3], latt_size[2], latt_size[1], latt_size[0],
                dtype=dtype, device=device)
verbose = True
tol = 1e-9
# mg = inverse.mg(lat_size=latt_size,dtype=dtype,device=device, wilson=wilson, U=U,
#                 clover=clover, clover_term=clover_term, tol=tol, verbose=verbose)
mg = inverse.mg(lat_size=latt_size, dtype_list=[dtype, torch.complex64, torch.complex64, torch.complex64, torch.complex32], device_list=[device, torch.device('cuda'), torch.device('cpu'), torch.device('cpu'), torch.device('cpu')], wilson=wilson, U=U,
                clover=clover, clover_term=clover_term, tol=tol, verbose=verbose)
mg.init()

for op in mg.op_list:
    print(f"op.sitting.M.shape: {op.sitting.M.shape}")
    print(f"op.sitting.M.dtype: {op.sitting.M.dtype}")
    print(f"op.sitting.M.device: {op.sitting.M.device}")

def matvec(src: torch.Tensor, U: torch.Tensor = U, clover_term: torch.Tensor = clover_term) -> torch.Tensor:
    return wilson.give_wilson(src, U)+clover.give_clover(clover_term=clover_term, src=src)

def _matvec(src: torch.Tensor) -> torch.Tensor:
    return mg.op_list[0].matvec(src=src)

Ab = matvec(b)
_Ab = _matvec(b)
print(
    f"torch.norm(Ab-_Ab).item()/torch.norm(_Ab).item(): {torch.norm(Ab-_Ab).item()/torch.norm(_Ab).item()}")
# _x = inverse.cg(b=b, matvec=matvec, tol=tol, verbose=verbose)
_x = inverse.bicgstab(b=b, matvec=_matvec, tol=tol, verbose=verbose)
# _x = inverse.bicgstab(b=b, matvec=mg.op_list[0].matvec, tol=tol, verbose=verbose)

x = mg.solve(b=b)
mg.plot()
