import torch
import pyqcu.cann as _torch
import pyqcu.tools as tools
import pyqcu.lattice as lattice
from pyqcu.smear import wuppertal_smear
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
root = 0
CUDA = torch.device('cuda')
DT = torch.complex64
LAT = [8, 8, 8, 16]

if rank == root:
    U = _torch.zeros(size=[3, 3, 4] + LAT, dtype=DT, device=CUDA)
    lattice.generate_gauge_field(U, seed=42, sigma=0.1, verbose=False)
    src = _torch.randn(size=[4, 3] + LAT, dtype=DT, device=CUDA)
    ref = wuppertal_smear(src, U, rho=4.0, nstep=5, support_parallel=False)
    torch.cuda.synchronize()
    print(f"[G] r={rank} serial ref done", flush=True)
    whole_U = U.cpu()
    whole_src = src.cpu()
else:
    whole_U = None
    whole_src = None
    ref = None

local_U = tools.whole_xyzt2local_xyzt(
    whole_array=whole_U, whole_shape=[3, 3, 4] + LAT, root=root,
    dtype=DT, device=CUDA)
local_src = tools.whole_xyzt2local_xyzt(
    whole_array=whole_src, whole_shape=[4, 3] + LAT, root=root,
    dtype=DT, device=CUDA)
print(f"[G] r={rank} scatter done shape={tuple(local_src.shape)}", flush=True)

out_local = wuppertal_smear(local_src, local_U, rho=4.0,
                            nstep=5, support_parallel=True)
torch.cuda.synchronize()
print(f"[G] r={rank} parallel smear done", flush=True)

print(f"[G] r={rank} pre-gather barrier", flush=True)
import mpi4py.MPI as _M
_M.COMM_WORLD.Barrier()
print(f"[G] r={rank} post-barrier -> gather", flush=True)
gathered = tools.local_xyzt2whole_xyzt(local_array=out_local, root=root)
print(f"[G] r={rank} gather done", flush=True)
# tools.norm 内含 comm.Allreduce —— 必须全体 rank 对称参与(非 root 以零场占位)
num = tools.norm((gathered.to(CUDA) - ref) if rank == root else torch.zeros(1, dtype=DT, device=CUDA))
den = tools.norm(ref.to(CUDA) if rank == root and ref is not None else torch.zeros(1, dtype=DT, device=CUDA))
if rank == root:
    diff = num / den   # tools.norm 已返回 float(勿再 .item())
    verdict = "PASS" if diff < 1e-5 else "FAIL"
    print(f"[C6][GOLD] np={comm.Get_size()} grid={tools.give_grid_size()} "
          f"rel_vs_serial_ref={diff:.2e} -> {verdict}", flush=True)
comm.Barrier()
