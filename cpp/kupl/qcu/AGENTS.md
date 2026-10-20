# KUPL QCU Backend

- Independent C++17 MPI+OpenMP implementation. Never include or link
  `cpp/cuda/qcu`.
- Keep the same backend layering as CUDA QCU: `include/kupl/` declarations,
  `src/apply_*.cpp` thin C ABI adapters, `src/lattice_*.cpp` numerical
  domains, and `backend.cpp` as the `LatticeSet` facade.
- Public ABI: `params int32[58]`, real `argv[8]`, `set_ptrs int64[100]`.
  `argv[7]` is `csw`; keep indices synchronized with `include/kupl/define.h`.
- Canonical storage is C-order with parity-compressed time:
  gauge `[2,3,3,4,X,Y,Z,T/2]`,
  fermion `[2,4,3,X,Y,Z,T/2]`,
  onsite `[4,3,4,3,X,Y,Z,T/2]`.
- Fermion external indices are `(parity*12+component)*half_sites+site`;
  Clover indices are `(row*12+column)*half_sites+site`. Halo buffers may be
  face-site-major but must be converted through a vector view.
- Use MPI for four-dimensional rank halos and OpenMP only inside a rank.
  Global reductions must use `MPI_Allreduce`; do not introduce rank-dependent
  collectives.
- Clover onsite coefficient is `-kappa*csw/8`, with
  `kappa=1/(2*mass+8)`. The public Clover ABI returns `A=I+T`, never bare
  `T`; `applyCloversQcu` returns `A^-1`, and `csw=0` must produce identity.
- Wilson solver semantics are parity Schur based:
  `S_W=I-kappa^2 D_bare,pq D_bare,qp`. BiStabCG solves `S_W`; CG solves
  `S_W^dagger S_W`. They solve the same full Wilson matrix but do not share an
  iteration trajectory.
- New solve paths check a reconstructed full-field true residual. The c64
  relative floor is `max(atol,2048*epsilon)` and the c128 floor is
  `max(atol,128*epsilon)`.
- The standalone Laplacian uses `gauge[3,3,3,X,Y,Z]`,
  `field[3,X,Y,Z]`, periodic three-dimensional MPI halos, and ignores `T` and
  `grid_t`. Never route it through the parity-compressed Wilson geometry.
- Clover prepare/reconstruct returns `0` on success and nonzero on failure;
  do not turn a failed Schur setup into a plausible output buffer.
- Gaussian gauge, recursive legacy Clover-MG, strict coarse/NULL/MATPC
  primitives, persistent strict V-cycle/FGMRES and MG verify are implemented.
  Keep homogeneous c64/c128 precision and fail closed on unsupported mixed
  precision or CUDA-only mode bits.
- Build and validate all changes with:

  ```bash
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j
  ctest --test-dir build --output-on-failure
  ```

- `qcu.map` exposes only the 37 CUDA-compatible `*Qcu` symbols. Keep
  `pyqcu/testing/kupl/check_abi.py` in the CTest suite when changing ABI
  declarations; it compares both headers and the exported dynamic surface.
- `KUPL_NUMA_BIND=1` owns CPU placement; launch MPI with binding disabled
  (`--bind-to none` for OpenMPI). If the scheduler already provides a
  NUMA-local cpuset per rank, keep `KUPL_NUMA_BIND=0`.

- Do not add GPU, LIME/HDF5 I/O, or PyQCU Python changes to this backend
  without a separate interface and validation task.
