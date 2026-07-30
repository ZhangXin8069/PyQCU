# CLAUDE.md — pyqcu.tools

Utility modules for MPI grid management, I/O, linear algebra, tensor operations, and multigrid transfers.

## Files

| File | Purpose |
|------|---------|
| `_define.py` | MPI grid size factorization, rank neighbors, parity splitting, dimension reordering, dtype conversion, constants |
| `_io.py` | HDF5 I/O with MPI parallel I/O (`driver='mpio'`) and serial gather/scatter fallback |
| `_einsum.py` | TileLang JIT-compiled einsum kernels for CUDA (e.g., `Eexyzt_exyzt2Exyzt`) |
| `_matul.py` | TileLang-based matrix multiply kernels for GPU and CPU |
| `_linalg.py` | Vector dot product (`vdot`) and norm (`norm`) |
| `_multigrid.py` | Null vector generation, local orthogonalization, restrict/prolong operators |
| `_roll.py` | Tensor rolling utilities |

## MPI Grid

The 4D process grid is auto-factored from `MPI.COMM_WORLD` size via prime factorization (`give_grid_size()`). Neighbor ranks are computed by `give_rank_plus`/`give_rank_minus` (and their double-hop variants `_plus_plus`, `_plus_minus`, etc.).

## Dimension Reordering

HDF5 I/O uses dimension order `zyxt` (fastest to slowest: t, z, y, x) internally. Conversion functions:
- `ccdxyzt2ccdptzyx` / `ccdptzyx2ccdxyzt` — gauge field (color,color,direction,parity,t,z,y,x)
- `scxyzt2psctzyx` / `psctzyx2scxyzt` — fermion field (parity,spin,color,t,z,y,x)

## Parity Splitting

- `oooxyzt2poooxyzt` — standard layout → parity-split `[p=2, ...]` (even/odd sites)
- `poooxyzt2oooxyzt` — reverse: parity-split → standard layout

## TileLang Integration

Optional — import fails silently if TileLang is not available. Kernels use `warp_size=128` from `_define` for GPU launch configuration.

## Multigrid Utilities

- `give_null_vecs(src, U, kappa, num_null_vecs, atol, ...)` — inverse iteration for near-null-space vectors
- `local_orthogonalize(vecs)` — Gram-Schmidt orthogonalization
- `restrict(coarse, fine, null_vecs)` / `prolong(fine, coarse, null_vecs)` — inter-grid transfers
