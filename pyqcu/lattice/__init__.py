from typing import Optional
import mpi4py.MPI as MPI
from typing import Optional
from pyqcu import _torch, tools
import torch
from argparse import Namespace
from math import sqrt
Namespace.__module__ = "pyqcu.lattice"
Namespace.__module__ = "pyqcu"
wards = dict()
wards['x'] = 0 - 4
wards['y'] = 1 - 4
wards['z'] = 2 - 4
wards['t'] = 3 - 4
wards['t_p'] = 3 - 4
wards['xy'] = 0 - 6
wards['xz'] = 1 - 6
wards['xt'] = 2 - 6
wards['yz'] = 3 - 6
wards['yt'] = 4 - 6
wards['zt'] = 5 - 6
wards['xyz'] = 0 - 4
wards['xyt'] = 1 - 4
wards['xzt'] = 2 - 4
wards['yzt'] = 3 - 4
ward_keys = ['x', 'y', 'z', 't']
ward_p_keys = ['x', 'y', 'z', 't_p']
ward_wards = dict()
ward_wards['xy'] = {'mu': wards['x'], 'nu': wards['y'], 'ward': wards['xy']}
ward_wards['xz'] = {'mu': wards['x'], 'nu': wards['z'], 'ward': wards['xz']}
ward_wards['xt'] = {'mu': wards['x'], 'nu': wards['t'], 'ward': wards['xt']}
ward_wards['yz'] = {'mu': wards['y'], 'nu': wards['z'], 'ward': wards['yz']}
ward_wards['yt'] = {'mu': wards['y'], 'nu': wards['t'], 'ward': wards['yt']}
ward_wards['zt'] = {'mu': wards['z'], 'nu': wards['t'], 'ward': wards['zt']}
ward_ward_keys = ['xy', 'xz', 'xt', 'yz', 'yt', 'zt']


def give_support_multi() -> bool:
    comm = MPI.COMM_WORLD
    return comm.Get_size() > 1


I = _torch.eye(4, dtype=torch.complex64, device=torch.device('cpu'))
minus_I = -1 * I
gamma = torch.zeros(4, 4, 4, dtype=torch.complex64,
                    device=torch.device('cpu'))
# gamma_0 (x-direction)
gamma[wards['x']] = torch.tensor([
    [0, 0, 0, 1j],
    [0, 0, 1j, 0],
    [0, -1j, 0, 0],
    [-1j, 0, 0, 0]
])
# gamma_1 (y-direction)
gamma[wards['y']] = torch.tensor([
    [0, 0, 0, -1],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [-1, 0, 0, 0]
])
# gamma_2 (z-direction)
gamma[wards['z']] = torch.tensor([
    [0, 0, 1j, 0],
    [0, 0, 0, -1j],
    [-1j, 0, 0, 0],
    [0, 1j, 0, 0]
])
# gamma_3 (t-direction)
gamma[wards['t']] = torch.tensor([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])
gamma_5 = torch.matmul(gamma[wards['x']], torch.matmul(
    gamma[wards['y']], torch.matmul(gamma[wards['z']], gamma[wards['t']])))
gamma_gamma = torch.zeros(
    6, 4, 4, dtype=torch.complex64,  device=torch.device('cpu'))
# gamma_gamma0 xy-direction)
gamma_gamma[wards['xy']] = torch.einsum(
    'ab,bc->ac', gamma[wards['x']], gamma[wards['y']])
# gamma_gamma1 xz-direction)
gamma_gamma[wards['xz']] = torch.einsum(
    'ab,bc->ac', gamma[wards['x']], gamma[wards['z']])
# gamma_gamma2 xt-direction)
gamma_gamma[wards['xt']] = torch.einsum(
    'ab,bc->ac', gamma[wards['x']], gamma[wards['t']])
# gamma_gamma3 yz-direction)
gamma_gamma[wards['yz']] = torch.einsum(
    'ab,bc->ac', gamma[wards['y']], gamma[wards['z']])
# gamma_gamma4 yt-direction)
gamma_gamma[wards['yt']] = torch.einsum(
    'ab,bc->ac', gamma[wards['y']], gamma[wards['t']])
# gamma_gamma5 zt-direction)
gamma_gamma[wards['zt']] = torch.einsum(
    'ab,bc->ac', gamma[wards['z']], gamma[wards['t']])
gell_mann = torch.zeros(8, 3, 3, dtype=torch.complex64,
                        device=torch.device('cpu'))
gell_mann[0] = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
# Will multiply by 1j later
gell_mann[1] = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
gell_mann[2] = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
gell_mann[3] = torch.tensor([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
# Will multiply by 1j later
gell_mann[4] = torch.tensor([[0, 0, -1], [0, 0, 0], [1, 0, 0]])
gell_mann[5] = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
# Will multiply by 1j later
gell_mann[6] = torch.tensor([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
gell_mann[7] = torch.tensor(
    [[1/sqrt(3), 0, 0], [0, 1/sqrt(3), 0], [0, 0, -2/sqrt(3)]])
# Apply imaginary factors where needed
gell_mann[1] = gell_mann[1] * 1j
gell_mann[4] = gell_mann[4] * 1j
gell_mann[6] = gell_mann[6] * 1j


def check_su3(U: torch.Tensor, tol: float = 1e-6, verbose: bool = False) -> bool:
    U_mat = U.permute(*range(2, U.ndim), 0,
                      1).reshape(-1, 3, 3).clone()  # N x 3 x 3
    N = U_mat.shape[0]
    # Precompute the identity matrix for unitary check
    eye = _torch.eye(3, dtype=U_mat.dtype,
                     device=U_mat.device).expand(N, -1, -1)
    # 1 Unitarity check: Uᴴ U ≈ I
    UH_U = _torch.matmul(U_mat.conj().transpose(-1, -2), U_mat)
    unitary_ok = _torch.allclose(UH_U, eye, atol=tol)
    # 2 Determinant check: det(U) ≈ 1
    det_U = torch.linalg.det(U_mat)
    det_ok = _torch.allclose(det_U, torch.ones_like(det_U), atol=tol)
    # 3 Minor identities check
    # Flatten matrices to shape (N, 9) for easy indexing
    Uf = U_mat.reshape(N, 9)
    c6 = (Uf[:, 1] * Uf[:, 5] - Uf[:, 2] * Uf[:, 4]).conj()
    c7 = (Uf[:, 2] * Uf[:, 3] - Uf[:, 0] * Uf[:, 5]).conj()
    c8 = (Uf[:, 0] * Uf[:, 4] - Uf[:, 1] * Uf[:, 3]).conj()
    minors_ok = (_torch.allclose(Uf[:, 6], c6, atol=tol) and
                 _torch.allclose(Uf[:, 7], c7, atol=tol) and
                 _torch.allclose(Uf[:, 8], c8, atol=tol))
    # --- Optional verbose output ---
    if verbose:
        print(
            f"PYQCU::TESTING::LATTICE:\n [check_su3] Total matrices checked: {N}")
        print(f"PYQCU::TESTING::LATTICE:\n Unitary check   : {unitary_ok}")
        print(f"PYQCU::TESTING::LATTICE:\n Determinant=1   : {det_ok}")
        print(f"PYQCU::TESTING::LATTICE:\n Minor identities: {minors_ok}")
        if not unitary_ok:
            max_err = (UH_U - eye).abs().max().item()
            print(
                f"PYQCU::TESTING::LATTICE:\n Max unitary deviation: {max_err:e}")
        if not det_ok:
            max_det_err = (det_U - 1).abs().max().item()
            print(
                f"PYQCU::TESTING::LATTICE:\n Max det deviation: {max_det_err:e}")
    return unitary_ok and det_ok and minors_ok


def generate_gauge_field(U: torch.Tensor, sigma: float = 0.1, seed: Optional[int] = None, verbose: bool = False) -> torch.Tensor:
    if verbose:
        print(
            f"PYQCU::TESTING::LATTICE:\n Generating gauge field with sigma={sigma}")
    # Random seed
    if seed is not None:
        if verbose:
            print(f"PYQCU::TESTING::LATTICE:\n Setting random seed: {seed}")
        torch.manual_seed(seed)
        if U.device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)
    # Random Gaussian coefficients: shape [4, Lx, Ly, Lz, Lt, 8]
    a = torch.normal(
        0.0, 1.0,
        size=(4, U.shape[-4], U.shape[-3], U.shape[-2], U.shape[-1], 8),
        dtype=U.dtype.to_real(),
        device=U.device
    )
    if verbose:
        print(
            f"PYQCU::TESTING::LATTICE:\n Coefficient tensor shape: {a.shape}")
        print(f"PYQCU::TESTING::LATTICE:\n Coefficient dtype: {a.dtype}")
    # Expand gell_mann basis for broadcasting
    # gell_mann: (8, 3, 3) -> (1, 1, 1, 1, 1, 8, 3, 3)
    gell_mann_expanded = gell_mann.view(
        1, 1, 1, 1, 1, 8, 3, 3).to(U.dtype).to(U.device)
    # Compute all Hermitian matrices H in one go: shape [4, Lx, Ly, Lz, Lt, 3, 3]
    H = _torch.einsum('...i,...ijk->...jk',
                      a.to(U.dtype), gell_mann_expanded)
    # Apply exponential map: shape stays [4, Lx, Ly, Lz, Lt, 3, 3]
    U_all = torch.matrix_exp(1j * sigma * H)
    # Rearrange to [3, 3, 4, Lx, Ly, Lz, Lt]
    U[:] = U_all.permute(5, 6, 0, 1, 2, 3, 4).contiguous()
    if verbose:
        print("PYQCU::TESTING::LATTICE:\n Gauge field generation complete")
        print(
            f"PYQCU::TESTING::LATTICE:\n Gauge field norm: {_torch.norm(U).item()}")
    return U
