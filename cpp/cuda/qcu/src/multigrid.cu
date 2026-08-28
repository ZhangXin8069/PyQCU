#include "../include/qcu.h"
#include <cooperative_groups.h>
#pragma optimize(5)
namespace qcu {
namespace cg = cooperative_groups;

template <typename T>
__device__ inline bool multigrid_cg_bad(const LatticeComplex<T> &z) {
  return !((z.real() == z.real()) && (z.imag() == z.imag()) &&
           fabs(z.real()) != INFINITY && fabs(z.imag()) != INFINITY);
}

// local_orthogonalize stores null vectors in the blocked layout
// [E,e,Xc,bx,Yc,by,Zc,bz,Tc,bt].  This is not the same memory order as the
// flattened [E,e,Xf,Yf,Zf,Tf] vector used by the solver: block-local
// coordinates are interleaved with coarse coordinates.  Keep the address
// calculation shared by homogeneous and mixed-precision transfer kernels.
__device__ inline size_t multigrid_null_index(
    int E_idx, int e_idx, int ix_f, int iy_f, int iz_f, int it_f, int e,
    int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc) {
  const int bx = Xf / Xc;
  const int by = Yf / Yc;
  const int bz = Zf / Zc;
  const int bt = Tf / Tc;
  const int ix_c = ix_f / bx;
  const int iy_c = iy_f / by;
  const int iz_c = iz_f / bz;
  const int it_c = it_f / bt;
  const int dx = ix_f - ix_c * bx;
  const int dy = iy_f - iy_c * by;
  const int dz = iz_f - iz_c * bz;
  const int dt = it_f - it_c * bt;

  size_t idx = static_cast<size_t>(E_idx);
  idx = idx * static_cast<size_t>(e) + static_cast<size_t>(e_idx);
  idx = idx * static_cast<size_t>(Xc) + static_cast<size_t>(ix_c);
  idx = idx * static_cast<size_t>(bx) + static_cast<size_t>(dx);
  idx = idx * static_cast<size_t>(Yc) + static_cast<size_t>(iy_c);
  idx = idx * static_cast<size_t>(by) + static_cast<size_t>(dy);
  idx = idx * static_cast<size_t>(Zc) + static_cast<size_t>(iz_c);
  idx = idx * static_cast<size_t>(bz) + static_cast<size_t>(dz);
  idx = idx * static_cast<size_t>(Tc) + static_cast<size_t>(it_c);
  idx = idx * static_cast<size_t>(bt) + static_cast<size_t>(dt);
  return idx;
}

template <typename T>
__device__ inline T multigrid_cg_abs1(const LatticeComplex<T> &z) {
  return fabs(z.real()) + fabs(z.imag());
}

template <typename T>
__device__ inline bool multigrid_cg_near_zero(const LatticeComplex<T> &z,
                                              T scale) {
  return multigrid_cg_bad(z) ||
         multigrid_cg_abs1(z) <= (T)1e-13 * scale;
}

// Every thread in the cooperative grid calls this helper.  A single global
// flag records the first invalid recurrence scalar, and grid.sync() makes the
// decision uniform before any thread can leave the iteration loop.  This is
// essential: an early return by only one block would deadlock later grid.sync
// calls, while allowing the recurrence to continue would poison x with NaN.
template <typename T>
__device__ inline bool multigrid_cg_abort(cg::grid_group &grid, int tid,
                                          int *breakdown, bool bad) {
  if (bad && tid == 0) atomicExch(breakdown, 1);
  grid.sync();
  return *breakdown != 0;
}

template <typename T>
__global__ void multigrid_restrict(void *coarse_out, void *fine_in,
                                   void *null_vecs, int E, int e, int Xf, int Yf,
                                   int Zf, int Tf, int Xc, int Yc, int Zc,
                                   int Tc) {
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int coarse_vol = Xc * Yc * Zc * Tc;
  int total_output = E * coarse_vol;
  if (global_idx >= total_output)
    return;

  LatticeComplex<T> *out = static_cast<LatticeComplex<T> *>(coarse_out);
  LatticeComplex<T> *in = static_cast<LatticeComplex<T> *>(fine_in);
  LatticeComplex<T> *nv = static_cast<LatticeComplex<T> *>(null_vecs);

  // Decompose global_idx into (E_idx, coarse_site)
  int E_idx = global_idx / coarse_vol;
  int rest = global_idx - E_idx * coarse_vol;
  int ix_c = rest / (Yc * Zc * Tc);
  rest -= ix_c * (Yc * Zc * Tc);
  int iy_c = rest / (Zc * Tc);
  rest -= iy_c * (Zc * Tc);
  int iz_c = rest / Tc;
  int it_c = rest - iz_c * Tc;

  // Coarsening factors
  int x = Xf / Xc;
  int y = Yf / Yc;
  int z = Zf / Zc;
  int t = Tf / Tc;

  // Pre-compute strides
  int fine_vol = Xf * Yf * Zf * Tf;
  int stride_YfZfTf = Yf * Zf * Tf;
  int stride_ZfTf = Zf * Tf;

  LatticeComplex<T> sum(0.0, 0.0);
  int fine_start = ix_c * x * stride_YfZfTf + iy_c * y * stride_ZfTf +
                   iz_c * z * Tf + it_c * t;

  for (int dx = 0; dx < x; dx++) {
    int ix_f_offset = dx * stride_YfZfTf;
    for (int dy = 0; dy < y; dy++) {
      int iy_f_offset = dy * stride_ZfTf;
      for (int dz = 0; dz < z; dz++) {
        int iz_f_offset = dz * Tf;
        for (int dt = 0; dt < t; dt++) {
          int fine_site = fine_start + ix_f_offset + iy_f_offset +
                          iz_f_offset + dt;
          int ix_f = fine_site / stride_YfZfTf;
          int rem = fine_site - ix_f * stride_YfZfTf;
          int iy_f = rem / stride_ZfTf;
          rem -= iy_f * stride_ZfTf;
          int iz_f = rem / Tf;
          int it_f = rem - iz_f * Tf;
          for (int e_idx = 0; e_idx < e; e_idx++) {
            int fine_idx = e_idx * fine_vol + fine_site;
            size_t nv_idx = multigrid_null_index(
                E_idx, e_idx, ix_f, iy_f, iz_f, it_f, e, Xf, Yf, Zf, Tf,
                Xc, Yc, Zc, Tc);
            sum += nv[nv_idx].conj() * in[fine_idx];
          }
        }
      }
    }
  }
  out[global_idx] = sum;
}
template <typename T>
__global__ void multigrid_prolong(void *fine_out, void *coarse_in,
                                  void *null_vecs, int E, int e, int Xf, int Yf,
                                  int Zf, int Tf, int Xc, int Yc, int Zc,
                                  int Tc) {
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int fine_vol = Xf * Yf * Zf * Tf;
  int total_output = e * fine_vol;
  if (global_idx >= total_output)
    return;

  LatticeComplex<T> *out = static_cast<LatticeComplex<T> *>(fine_out);
  LatticeComplex<T> *cin = static_cast<LatticeComplex<T> *>(coarse_in);
  LatticeComplex<T> *nv = static_cast<LatticeComplex<T> *>(null_vecs);

  // Decompose global_idx into (e_idx, fine_site)
  int e_idx = global_idx / fine_vol;
  int fine_site = global_idx - e_idx * fine_vol;

  // Compute fine coordinates
  int stride_YfZfTf = Yf * Zf * Tf;
  int stride_ZfTf = Zf * Tf;
  int ix_f = fine_site / stride_YfZfTf;
  int rest = fine_site - ix_f * stride_YfZfTf;
  int iy_f = rest / stride_ZfTf;
  rest -= iy_f * stride_ZfTf;
  int iz_f = rest / Tf;
  int it_f = rest - iz_f * Tf;

  // Coarse coordinates
  int x = Xf / Xc;
  int y = Yf / Yc;
  int z = Zf / Zc;
  int t = Tf / Tc;
  int ix_c = ix_f / x;
  int iy_c = iy_f / y;
  int iz_c = iz_f / z;
  int it_c = it_f / t;

  // The coarse vector is C-order [E,Xc,Yc,Zc,Tc].  Null vectors use the
  // blocked 10-D layout; their physical fine-site index is not contiguous.
  int coarse_stride_E = Xc * Yc * Zc * Tc;
  int coarse_stride_YZT = Yc * Zc * Tc;
  int coarse_stride_ZT = Zc * Tc;
  // C-order coarse-site index (t fastest):
  int coarse_site = ix_c * coarse_stride_YZT + iy_c * coarse_stride_ZT +
                    iz_c * Tc + it_c;

  LatticeComplex<T> sum(0.0, 0.0);
  for (int E_idx = 0; E_idx < E; E_idx++) {
    size_t nv_idx = multigrid_null_index(
        E_idx, e_idx, ix_f, iy_f, iz_f, it_f, e, Xf, Yf, Zf, Tf, Xc, Yc,
        Zc, Tc);
    int coarse_idx = E_idx * coarse_stride_E + coarse_site;
    sum += nv[nv_idx] * cin[coarse_idx];
  }
  out[global_idx] = sum;
}

// ---------------------------------------------------------------------------
// Explicit cross-precision transfer kernels.
//
// A transition owns its null vectors in the child precision.  The old
// same-type kernels are intentionally kept for the homogeneous fast path;
// these variants make the conversion part of the memory access rather than
// relying on a reinterpret_cast<void*> at the call site.
// ---------------------------------------------------------------------------
template <typename Out, typename In>
__device__ inline LatticeComplex<Out>
multigrid_cast_complex(const LatticeComplex<In> &value) {
  return LatticeComplex<Out>((Out)value.real(), (Out)value.imag());
}

template <typename Out, typename In>
__global__ void multigrid_restrict_cast(void *coarse_out, void *fine_in,
                                        void *null_vecs, int E, int e, int Xf,
                                        int Yf, int Zf, int Tf, int Xc, int Yc,
                                        int Zc, int Tc) {
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int coarse_vol = Xc * Yc * Zc * Tc;
  int total_output = E * coarse_vol;
  if (global_idx >= total_output) return;

  LatticeComplex<Out> *out = static_cast<LatticeComplex<Out> *>(coarse_out);
  const LatticeComplex<In> *in =
      static_cast<const LatticeComplex<In> *>(fine_in);
  // Restriction writes the child/coarse precision (Out), so the null vectors
  // used for the local projection are in that same child precision.  ``In``
  // is only the parent/fine input type.
  const LatticeComplex<Out> *nv =
      static_cast<const LatticeComplex<Out> *>(null_vecs);

  int E_idx = global_idx / coarse_vol;
  int rest = global_idx - E_idx * coarse_vol;
  int ix_c = rest / (Yc * Zc * Tc);
  rest -= ix_c * (Yc * Zc * Tc);
  int iy_c = rest / (Zc * Tc);
  rest -= iy_c * (Zc * Tc);
  int iz_c = rest / Tc;
  int it_c = rest - iz_c * Tc;

  int x = Xf / Xc;
  int y = Yf / Yc;
  int z = Zf / Zc;
  int t = Tf / Tc;
  int fine_vol = Xf * Yf * Zf * Tf;
  int stride_YfZfTf = Yf * Zf * Tf;
  int stride_ZfTf = Zf * Tf;
  int fine_start = ix_c * x * stride_YfZfTf + iy_c * y * stride_ZfTf +
                   iz_c * z * Tf + it_c * t;

  LatticeComplex<Out> sum((Out)0, (Out)0);
  for (int dx = 0; dx < x; ++dx) {
    for (int dy = 0; dy < y; ++dy) {
      for (int dz = 0; dz < z; ++dz) {
        for (int dt = 0; dt < t; ++dt) {
          int fine_site = fine_start + dx * stride_YfZfTf +
                          dy * stride_ZfTf + dz * Tf + dt;
          int ix_f = fine_site / stride_YfZfTf;
          int rem = fine_site - ix_f * stride_YfZfTf;
          int iy_f = rem / stride_ZfTf;
          rem -= iy_f * stride_ZfTf;
          int iz_f = rem / Tf;
          int it_f = rem - iz_f * Tf;
          for (int e_idx = 0; e_idx < e; ++e_idx) {
            int fine_idx = e_idx * fine_vol + fine_site;
            size_t nv_idx = multigrid_null_index(
                E_idx, e_idx, ix_f, iy_f, iz_f, it_f, e, Xf, Yf, Zf, Tf,
                Xc, Yc, Zc, Tc);
            sum += nv[nv_idx].conj() *
                   multigrid_cast_complex<Out>(in[fine_idx]);
          }
        }
      }
    }
  }
  out[global_idx] = sum;
}

template <typename Out, typename In>
__global__ void multigrid_prolong_cast(void *fine_out, void *coarse_in,
                                       void *null_vecs, int E, int e, int Xf,
                                       int Yf, int Zf, int Tf, int Xc, int Yc,
                                       int Zc, int Tc) {
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int fine_vol = Xf * Yf * Zf * Tf;
  int total_output = e * fine_vol;
  if (global_idx >= total_output) return;

  LatticeComplex<Out> *out = static_cast<LatticeComplex<Out> *>(fine_out);
  const LatticeComplex<In> *cin =
      static_cast<const LatticeComplex<In> *>(coarse_in);
  // Prolongation consumes the child/coarse vector (In) and writes the parent
  // vector (Out); null vectors are stored with the child level as well.
  const LatticeComplex<In> *nv =
      static_cast<const LatticeComplex<In> *>(null_vecs);

  int e_idx = global_idx / fine_vol;
  int fine_site = global_idx - e_idx * fine_vol;
  int stride_YfZfTf = Yf * Zf * Tf;
  int stride_ZfTf = Zf * Tf;
  int ix_f = fine_site / stride_YfZfTf;
  int rest = fine_site - ix_f * stride_YfZfTf;
  int iy_f = rest / stride_ZfTf;
  rest -= iy_f * stride_ZfTf;
  int iz_f = rest / Tf;
  int it_f = rest - iz_f * Tf;

  int x = Xf / Xc;
  int y = Yf / Yc;
  int z = Zf / Zc;
  int t = Tf / Tc;
  int ix_c = ix_f / x;
  int iy_c = iy_f / y;
  int iz_c = iz_f / z;
  int it_c = it_f / t;
  int coarse_vol = Xc * Yc * Zc * Tc;
  int coarse_stride_YZT = Yc * Zc * Tc;
  int coarse_stride_ZT = Zc * Tc;
  int coarse_site = ix_c * coarse_stride_YZT + iy_c * coarse_stride_ZT +
                    iz_c * Tc + it_c;

  LatticeComplex<Out> sum((Out)0, (Out)0);
  for (int E_idx = 0; E_idx < E; ++E_idx) {
    size_t nv_idx = multigrid_null_index(
        E_idx, e_idx, ix_f, iy_f, iz_f, it_f, e, Xf, Yf, Zf, Tf, Xc, Yc,
        Zc, Tc);
    int coarse_idx = E_idx * coarse_vol + coarse_site;
    sum += multigrid_cast_complex<Out>(nv[nv_idx]) *
           multigrid_cast_complex<Out>(cin[coarse_idx]);
  }
  out[global_idx] = sum;
}

// ---------------------------------------------------------------------------
// Distributed wide-stencil kernel.
//
// The ordinary wide kernel uses periodic modulo indexing because its input is
// a complete coarse lattice.  In an MPI run each rank owns only a local block.
// The host-side MG driver fills ``halo`` with the 32 remote neighbour blocks;
// this kernel reads the local vector for an interior neighbour and the padded
// halo for a coordinate outside the local block.  The halo layout is
// [E, X+2, Y+2, Z+2, Lt+2], with one layer on every side.
// ---------------------------------------------------------------------------
template <typename T>
__device__ inline LatticeComplex<T> multigrid_halo_read(
    const LatticeComplex<T> *in, const LatticeComplex<T> *halo, int E,
    int X, int Y, int Z, int Lt, int e, int x, int y, int z, int t,
    int dx, int dy, int dz, int dt) {
  int nx = x + dx, ny = y + dy, nz = z + dz, nt = t + dt;
  if (nx >= 0 && nx < X && ny >= 0 && ny < Y && nz >= 0 && nz < Z &&
      nt >= 0 && nt < Lt) {
    int vol = X * Y * Z * Lt;
    int site = ((nx * Y + ny) * Z + nz) * Lt + nt;
    return in[e * vol + site];
  }
  int hvol = (X + 2) * (Y + 2) * (Z + 2) * (Lt + 2);
  int hsite = ((((nx + 1) * (Y + 2) + (ny + 1)) * (Z + 2) +
                (nz + 1)) * (Lt + 2) + (nt + 1));
  return halo[e * hvol + hsite];
}

template <typename T>
__global__ void multigrid_coarse_dslash_wide_halo(
    void *fermion_out, void *fermion_in, void *halo, void *sitting,
    void *hop_nn, void *hop_diag, int E, int X, int Y, int Z, int Lt) {
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int vol = X * Y * Z * Lt;
  int total_output = E * vol;
  if (global_idx >= total_output || E <= 0 || vol <= 0) return;

  LatticeComplex<T> *out = static_cast<LatticeComplex<T> *>(fermion_out);
  const LatticeComplex<T> *in =
      static_cast<const LatticeComplex<T> *>(fermion_in);
  const LatticeComplex<T> *h = static_cast<const LatticeComplex<T> *>(halo);
  const LatticeComplex<T> *sit =
      static_cast<const LatticeComplex<T> *>(sitting);
  const LatticeComplex<T> *nn =
      static_cast<const LatticeComplex<T> *>(hop_nn);
  const LatticeComplex<T> *dg =
      static_cast<const LatticeComplex<T> *>(hop_diag);

  int E_out = global_idx / vol;
  int site = global_idx - E_out * vol;
  int stride_YZT = Y * Z * Lt;
  int stride_ZT = Z * Lt;
  int x = site / stride_YZT;
  int rest = site - x * stride_YZT;
  int y = rest / stride_ZT;
  rest -= y * stride_ZT;
  int z = rest / Lt;
  int t = rest - z * Lt;

  int str_Ein = vol;
  int str_Eout = E * str_Ein;
  int str_dir = E * str_Eout;
  int str_pm = 4 * str_dir;
  int dg_str_s2 = 6 * E * str_Eout;
  int dg_str_s1 = 2 * dg_str_s2;
  int dg_str_pair = E * str_Eout;
  LatticeComplex<T> sum(0.0, 0.0);

  int sit_base = E_out * str_Eout + site;
  for (int e = 0; e < E; ++e)
    sum += sit[sit_base + e * str_Ein] * in[e * vol + site];

  int dir_dx[4] = {1, 0, 0, 0};
  int dir_dy[4] = {0, 1, 0, 0};
  int dir_dz[4] = {0, 0, 1, 0};
  int dir_dt[4] = {0, 0, 0, 1};
  for (int d = 0; d < 4; ++d) {
    int sx = dir_dx[d], sy = dir_dy[d], sz = dir_dz[d], st = dir_dt[d];
    int plus_base = d * str_dir + E_out * str_Eout + site;
    int minus_base = str_pm + d * str_dir + E_out * str_Eout + site;
    for (int e = 0; e < E; ++e) {
      sum += nn[plus_base + e * str_Ein] *
             multigrid_halo_read(in, h, E, X, Y, Z, Lt, e, x, y, z, t,
                                 sx, sy, sz, st);
      sum += nn[minus_base + e * str_Ein] *
             multigrid_halo_read(in, h, E, X, Y, Z, Lt, e, x, y, z, t,
                                 -sx, -sy, -sz, -st);
    }
  }

  int pair_d1[6] = {0, 0, 0, 1, 1, 2};
  int pair_d2[6] = {1, 2, 3, 2, 3, 3};
  int coords[4] = {x, y, z, t};
  for (int pi = 0; pi < 6; ++pi) {
    int d1 = pair_d1[pi], d2 = pair_d2[pi];
    for (int s1i = 0; s1i < 2; ++s1i) {
      for (int s2i = 0; s2i < 2; ++s2i) {
        int s1 = s1i == 0 ? 1 : -1;
        int s2 = s2i == 0 ? 1 : -1;
        int dx = 0, dy = 0, dz = 0, dt = 0;
        if (d1 == 0) dx = s1;
        else if (d1 == 1) dy = s1;
        else if (d1 == 2) dz = s1;
        else dt = s1;
        if (d2 == 0) dx = s2;
        else if (d2 == 1) dy = s2;
        else if (d2 == 2) dz = s2;
        else dt = s2;
        int dg_base = s1i * dg_str_s1 + s2i * dg_str_s2 +
                      pi * dg_str_pair + E_out * str_Eout + site;
        for (int e = 0; e < E; ++e)
          sum += dg[dg_base + e * str_Ein] *
                 multigrid_halo_read(in, h, E, X, Y, Z, Lt, e, x, y, z, t,
                                     dx, dy, dz, dt);
      }
    }
  }
  out[global_idx] = sum;
}

template <typename T>
__global__ void multigrid_coarse_dslash(void *fermion_out, void *fermion_in,
                                         void *hopping, void *sitting,
                                         int E, int X, int Y, int Z, int Lt) {
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int vol = X * Y * Z * Lt;
  int total_output = E * vol;
  if (global_idx >= total_output)
    return;
  // Early exit for invalid parameters
  if (E <= 0 || vol <= 0)
    return;

  LatticeComplex<T> *out = static_cast<LatticeComplex<T> *>(fermion_out);
  LatticeComplex<T> *in = static_cast<LatticeComplex<T> *>(fermion_in);
  LatticeComplex<T> *hop = static_cast<LatticeComplex<T> *>(hopping);
  LatticeComplex<T> *sit = static_cast<LatticeComplex<T> *>(sitting);

  // Decompose global_idx into (E_out, site)
  int E_out = global_idx / vol;
  int site = global_idx - E_out * vol;

  // Decompose site into (x, y, z, t) — row-major (C-order) layout
  int stride_YZT = Y * Z * Lt;
  int stride_ZT = Z * Lt;
  int x = site / stride_YZT;
  int rest = site - x * stride_YZT;
  int y = rest / stride_ZT;
  rest -= y * stride_ZT;
  int z = rest / Lt;
  int t = rest - z * Lt;

  // Strides for hopping: shape [2, 4, E, E, X, Y, Z, T] in C-order
  // dim order: pm(2) × dir(4) × Eout(E) × Ein(E) × X × Y × Z × T
  int hop_vol = vol;
  int hop_stride_Ein = hop_vol;
  int hop_stride_Eout = E * hop_stride_Ein;
  int hop_stride_dir = E * hop_stride_Eout;
  int hop_stride_pm = 4 * hop_stride_dir;

  // Strides for sitting: shape [E, E, X, Y, Z, T] in C-order
  int sit_stride_Ein = hop_vol;
  int sit_stride_Eout = E * sit_stride_Ein;

  // Strides for fermion: shape [E, X, Y, Z, T] in C-order
  int ferm_stride_E = hop_vol;

  LatticeComplex<T> sum(0.0, 0.0);

  // --- Sitting term ---
  // out[E_out, x,y,z,t] = sum_e sitting[E_out, e, x,y,z,t] * in[e, x,y,z,t]
  int sit_base = E_out * sit_stride_Eout + site;
  for (int e = 0; e < E; e++) {
    sum += sit[sit_base + e * sit_stride_Ein] * in[e * ferm_stride_E + site];
  }

  // --- Hopping term: 4 directions × plus/minus ---
  // Direction data: offset (stride in flattened index), dim size, coordinate
  int dir_offsets[4] = {stride_YZT, stride_ZT, Lt, 1};
  int dir_dims[4] = {X, Y, Z, Lt};
  int dir_coords[4] = {x, y, z, t};

  for (int d = 0; d < 4; d++) {
    int offset = dir_offsets[d];
    int dim = dir_dims[d];
    int coord = dir_coords[d];

    // Forward neighbor: site + e_d (periodic)
    int fwd_coord = (coord + 1) % dim;
    int fwd_site = site - coord * offset + fwd_coord * offset;

    // Backward neighbor: site - e_d (periodic)
    int bwd_coord = (coord - 1 + dim) % dim;
    int bwd_site = site - coord * offset + bwd_coord * offset;

    // Base offsets for plus/minus hopping at this site, direction, E_out
    int hop_plus_base = 0 * hop_stride_pm + d * hop_stride_dir +
                        E_out * hop_stride_Eout + site;
    int hop_minus_base = 1 * hop_stride_pm + d * hop_stride_dir +
                         E_out * hop_stride_Eout + site;

    for (int e = 0; e < E; e++) {
      int e_offset = e * hop_stride_Ein;
      sum += hop[hop_plus_base + e_offset] * in[e * ferm_stride_E + fwd_site];
      sum += hop[hop_minus_base + e_offset] * in[e * ferm_stride_E + bwd_site];
    }
  }

  out[global_idx] = sum;
}
/**
 * @brief Wide-stencil coarse-grid dslash for the SCHUR-consistent coarse operator
 *        A_c = P^T S P (on-site + nearest-neighbour + diagonal couplings).
 *
 * The Schur operator S = D_oo - k^2 H_oe D_ee^{-1} H_eo couples odd sites x to
 * x, x±2μ (nearest in the coarse grid) AND x±μ±ν (diagonal), so its Galerkin
 * projection A_c = P^T S P has a 33-tensor stencil:
 *   sit      [E, E, X, Y, Z, T]            on-site
 *   hop_nn   [2, 4, E, E, X, Y, Z, T]      nearest (pm × dir)
 *   hop_diag [2, 2, 6, E, E, X, Y, Z, T]   diagonal (s1 × s2 × pair)
 *      pair: 0=(x,y) 1=(x,z) 2=(x,t) 3=(y,z) 4=(y,t) 5=(z,t); sign 0=+1 1=-1
 *
 * Kernel convention:
 *   out[j,c] += sit[j,e,c]·in[e,c]
 *             + hop_nn[pm,d,j,e,c]·in[e, c + pm?(+1):(-1) e_d]
 *             + hop_diag[s1,s2,pair,j,e,c]·in[e, c + s1 e_d1 + s2 e_d2]
 *
 * Layout is C-order (t fastest) everywhere, matching multigrid_coarse_dslash.
 */
template <typename T>
__global__ void multigrid_coarse_dslash_wide(void *fermion_out, void *fermion_in,
                                             void *sitting, void *hop_nn,
                                             void *hop_diag, int E, int X, int Y,
                                             int Z, int Lt) {
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int vol = X * Y * Z * Lt;
  int total_output = E * vol;
  if (global_idx >= total_output) return;
  if (E <= 0 || vol <= 0) return;

  LatticeComplex<T> *out = static_cast<LatticeComplex<T>*>(fermion_out);
  LatticeComplex<T> *in  = static_cast<LatticeComplex<T>*>(fermion_in);
  LatticeComplex<T> *sit = static_cast<LatticeComplex<T>*>(sitting);
  LatticeComplex<T> *nn  = static_cast<LatticeComplex<T>*>(hop_nn);
  LatticeComplex<T> *dg  = static_cast<LatticeComplex<T>*>(hop_diag);

  int E_out = global_idx / vol;
  int site  = global_idx - E_out * vol;

  // Decompose site into (x, y, z, t) — C-order layout
  int stride_YZT = Y * Z * Lt;
  int stride_ZT  = Z * Lt;
  int x = site / stride_YZT;
  int rest = site - x * stride_YZT;
  int y = rest / stride_ZT;
  rest -= y * stride_ZT;
  int z = rest / Lt;
  int t = rest - z * Lt;

  // Strides: [.., E, E, X, Y, Z, T] C-order
  int hop_vol = vol;
  int str_Ein  = hop_vol;
  int str_Eout = E * str_Ein;
  int str_dir  = E * str_Eout;
  int str_pm   = 4 * str_dir;
  int str_s2   = 6 * str_Eout;          // hop_diag [s1,s2,pair,..] after s1? see below
  // hop_diag dims: [2(s1), 2(s2), 6(pair), E, E, X, Y, Z, T]
  int dg_str_s2   = 6 * E * str_Eout;   // s2 stride
  int dg_str_s1   = 2 * dg_str_s2;      // s1 stride
  // (pair stride is E*str_Eout)
  int dg_str_pair = E * str_Eout;

  int sit_str_Ein  = hop_vol;
  int sit_str_Eout = E * sit_str_Ein;
  int ferm_str_E   = hop_vol;

  LatticeComplex<T> sum(0.0, 0.0);

  // --- Sitting (on-site) ---
  int sit_base = E_out * sit_str_Eout + site;
  for (int e = 0; e < E; e++) {
    sum += sit[sit_base + e * sit_str_Ein] * in[e * ferm_str_E + site];
  }

  // --- Nearest neighbours: 4 directions × plus/minus ---
  int dir_offsets[4] = {stride_YZT, stride_ZT, Lt, 1};
  int dir_dims[4]    = {X, Y, Z, Lt};
  int dir_coords[4]  = {x, y, z, t};
  for (int d = 0; d < 4; d++) {
    int offset = dir_offsets[d];
    int dim    = dir_dims[d];
    int coord  = dir_coords[d];
    int fwd_coord = (coord + 1) % dim;
    int bwd_coord = (coord - 1 + dim) % dim;
    int fwd_site  = site - coord * offset + fwd_coord * offset;
    int bwd_site  = site - coord * offset + bwd_coord * offset;
    int nn_plus_base  = 0 * str_pm + d * str_dir + E_out * str_Eout + site;
    int nn_minus_base = 1 * str_pm + d * str_dir + E_out * str_Eout + site;
    for (int e = 0; e < E; e++) {
      int e_off = e * str_Ein;
      sum += nn[nn_plus_base  + e_off] * in[e * ferm_str_E + fwd_site];
      sum += nn[nn_minus_base + e_off] * in[e * ferm_str_E + bwd_site];
    }
  }

  // --- Diagonal couplings: 6 pairs × 4 sign combos ---
  // hop_diag [s1,s2,pair,E,E,X,Y,Z,T]: neighbour = c + s1*e_d1 + s2*e_d2
  int pair_d1[6] = {0, 0, 0, 1, 1, 2};  // (x,y),(x,z),(x,t),(y,z),(y,t),(z,t)
  int pair_d2[6] = {1, 2, 3, 2, 3, 3};
  for (int pi = 0; pi < 6; pi++) {
    int d1 = pair_d1[pi], d2 = pair_d2[pi];
    for (int s1i = 0; s1i < 2; s1i++) {      // 0 → +1, 1 → -1
      for (int s2i = 0; s2i < 2; s2i++) {
        int s1 = (s1i == 0) ? 1 : -1;
        int s2 = (s2i == 0) ? 1 : -1;
        // neighbour coordinate along d1, d2
        int n1 = (dir_coords[d1] + s1 + dir_dims[d1]) % dir_dims[d1];
        int n2 = (dir_coords[d2] + s2 + dir_dims[d2]) % dir_dims[d2];
        int ns = site;
        if (d1 == 0)      ns = ns - x * stride_YZT + n1 * stride_YZT;
        else if (d1 == 1) ns = ns - y * stride_ZT + n1 * stride_ZT;
        else if (d1 == 2) ns = ns - z * Lt + n1 * Lt;
        else              ns = ns - t + n1;      // d1 == 3
        // apply d2 on the (possibly d1-shifted) coordinates
        int cur = (d2 == 0) ? (ns / stride_YZT) :
                  (d2 == 1) ? ((ns % stride_YZT) / stride_ZT) :
                  (d2 == 2) ? ((ns % stride_ZT) / Lt) :
                              (ns % Lt);
        int delta = n2 - cur;
        ns += delta * dir_offsets[d2];
        int dg_base = s1i * dg_str_s1 + s2i * dg_str_s2 + pi * dg_str_pair +
                      E_out * str_Eout + site;
        for (int e = 0; e < E; e++) {
          sum += dg[dg_base + e * str_Ein] * in[e * ferm_str_E + ns];
        }
      }
    }
  }
  out[global_idx] = sum;
}
/**
 * @brief FUSED coarse-level BiStabCG solver — the ENTIRE coarse solve runs in
 *        ONE kernel launch.
 *
 * Motivation: the GPU runs at idle clock (210 MHz) for the many tiny kernels of
 * a per-iteration coarse solve (~13 launches/iter × 30-90 iters ≈ 400-1200
 * launches per V-cycle at ~107 µs each).  This single kernel does all BiStabCG
 * iterations internally with block reductions, costing ONE launch per solve.
 *
 * Coarse vectors are [E, X, Y, Z, Lt] C-order (n = E·X·Y·Z·Lt elements).  Each
 * of 256 threads owns elements at stride 256.  The 33-tensor coarse operator
 * A_c = P^T S P (sit + hop_nn + hop_diag) is applied in-kernel.  All vectors
 * live in global memory (the coarse grid is tiny, ~49 KB, so L2 serves the
 * neighbour reads).
 *
 * @param x,r_tilde,r,p,v,s,t  BiStabCG vectors [n] (x and p/v/s/t zeroed here)
 * @param rhs    RHS [n]
 * @param sitting,hop_nn,hop_diag  33-tensor coarse operator
 * @param E,X,Y,Z,Lt  coarse dims
 * @param max_iter  max BiStabCG iterations
 * @param tol    relative convergence target (||r|| < tol·||rhs||)
 */
#define _CS_REDUCE(sum)                                                     \
  sred[tid] = sum; __syncthreads();                                         \
  for (int k = NT/2; k > 0; k >>= 1) {                                      \
    if (tid < k) sred[tid] += sred[tid + k]; __syncthreads();               \
  }

template <typename T>
__global__ void multigrid_coarse_solve(void *x, void *rhs, void *r_tilde,
                                       void *r, void *p, void *v, void *s,
                                       void *t, void *sitting, void *hop_nn,
                                       void *hop_diag, int E, int X, int Y,
                                       int Z, int Lt, int max_iter, T tol) {
  const int NT = 256;
  int n = E * X * Y * Z * Lt;
  int tid = threadIdx.x;
  __shared__ LatticeComplex<T> sred[NT];
  __shared__ LatticeComplex<T> s_rho, s_rtv, s_ts, s_tt, s_norm2;
  __shared__ LatticeComplex<T> s_beta, s_alpha, s_omega;
  __shared__ T s_r0;

  LatticeComplex<T> *xr = static_cast<LatticeComplex<T>*>(x);
  LatticeComplex<T> *br = static_cast<LatticeComplex<T>*>(rhs);
  LatticeComplex<T> *rtr = static_cast<LatticeComplex<T>*>(r_tilde);
  LatticeComplex<T> *rr = static_cast<LatticeComplex<T>*>(r);
  LatticeComplex<T> *pr = static_cast<LatticeComplex<T>*>(p);
  LatticeComplex<T> *vr = static_cast<LatticeComplex<T>*>(v);
  LatticeComplex<T> *sr = static_cast<LatticeComplex<T>*>(s);
  LatticeComplex<T> *tr = static_cast<LatticeComplex<T>*>(t);
  const LatticeComplex<T> *sit = static_cast<const LatticeComplex<T>*>(sitting);
  const LatticeComplex<T> *nn  = static_cast<const LatticeComplex<T>*>(hop_nn);
  const LatticeComplex<T> *dg  = static_cast<const LatticeComplex<T>*>(hop_diag);

  // ---- Init ----
  for (int i = tid; i < n; i += NT) {
    xr[i] = LatticeComplex<T>(0,0);
    rr[i] = br[i];
    rtr[i] = br[i];
    pr[i] = LatticeComplex<T>(0,0);
    vr[i] = LatticeComplex<T>(0,0);
    sr[i] = LatticeComplex<T>(0,0);
    tr[i] = LatticeComplex<T>(0,0);
  }
  __syncthreads();

  // ---- ||rhs|| ----
  {
    LatticeComplex<T> sum(0,0);
    for (int i = tid; i < n; i += NT) sum += br[i].conj() * br[i];
    _CS_REDUCE(sum);
    if (tid == 0) s_r0 = sqrt(sred[0].real() > 0 ? sred[0].real() : 0);
    __syncthreads();
    if (s_r0 < (T)1e-30) return;
  }

  // ---- Wide 33-stencil coarse dslash (out = A_c·in) ----
  // Thread computes output elements tid, tid+NT, ...
  auto dslash = [&](LatticeComplex<T> *out, const LatticeComplex<T> *in) {
    int vol = X*Y*Z*Lt;
    int stride_YZT = Y*Z*Lt, stride_ZT = Z*Lt;
    int str_Ein = vol, str_Eout = E*vol, str_dir = E*str_Eout, str_pm = 4*str_dir;
    int dg_str_s1 = 2*6*E*str_Eout, dg_str_s2 = 6*E*str_Eout, dg_str_pair = E*str_Eout;
    int d1s[6] = {0,0,0,1,1,2}, d2s[6] = {1,2,3,2,3,3};
    int offs[4] = {stride_YZT, stride_ZT, Lt, 1}, dims[4] = {X,Y,Z,Lt};
    for (int idx = tid; idx < n; idx += NT) {
      int E_out = idx / (X*Y*Z*Lt);
      int site  = idx - E_out * (X*Y*Z*Lt);
      int xc = site / stride_YZT; int rem = site % stride_YZT;
      int yc = rem / stride_ZT;   rem %= stride_ZT;
      int zc = rem / Lt;          int tc = rem % Lt;
      int coords[4] = {xc,yc,zc,tc};
      LatticeComplex<T> sum(0,0);
      int sb = E_out*str_Eout + site;
      for (int e=0; e<E; e++) sum += sit[sb + e*str_Ein] * in[e*str_Ein + site];
      for (int d=0; d<4; d++) {
        int fwd=(coords[d]+1)%dims[d], bwd=(coords[d]-1+dims[d])%dims[d];
        int fs = site - coords[d]*offs[d] + fwd*offs[d];
        int bs = site - coords[d]*offs[d] + bwd*offs[d];
        int nb = E_out*str_Eout + site;
        for (int e=0; e<E; e++) {
          sum += nn[0*str_pm + d*str_dir + nb + e*str_Ein] * in[e*str_Ein + fs];
          sum += nn[1*str_pm + d*str_dir + nb + e*str_Ein] * in[e*str_Ein + bs];
        }
      }
      for (int pi=0; pi<6; pi++) {
        int d1=d1s[pi], d2=d2s[pi];
        for (int s1i=0; s1i<2; s1i++) for (int s2i=0; s2i<2; s2i++) {
          int sgn1 = s1i==0?1:-1, sgn2 = s2i==0?1:-1;
          int n1 = (coords[d1]+sgn1+dims[d1])%dims[d1];
          int n2 = (coords[d2]+sgn2+dims[d2])%dims[d2];
          int ns = site;
          if (d1==0) ns = ns - xc*stride_YZT + n1*stride_YZT;
          else if (d1==1) ns = ns - yc*stride_ZT + n1*stride_ZT;
          else if (d1==2) ns = ns - zc*Lt + n1*Lt;
          else ns = ns - tc + n1;
          int cur = (d2==0)?(ns/stride_YZT):(d2==1)?((ns%stride_YZT)/stride_ZT):(d2==2)?((ns%stride_ZT)/Lt):(ns%Lt);
          ns += (n2-cur)*offs[d2];
          int db = s1i*dg_str_s1 + s2i*dg_str_s2 + pi*dg_str_pair + E_out*str_Eout + site;
          for (int e=0; e<E; e++) sum += dg[db + e*str_Ein] * in[e*str_Ein + ns];
        }
      }
      out[idx] = sum;
    }
  };

  LatticeComplex<T> rho(1,0), rho_prev(1,0), alpha(1,0), omega(1,0), rtv, ts, tt;
  T target = tol * s_r0;

  for (int it = 0; it < max_iter; ++it) {
    // 1. rho = <r_tilde, r>
    {
      LatticeComplex<T> sum(0,0);
      for (int i = tid; i < n; i += NT) sum += rtr[i].conj() * rr[i];
      _CS_REDUCE(sum);
      if (tid == 0) { s_rho = sred[0]; rho = sred[0]; }
      __syncthreads();
    }
    // 2. beta = (rho/rho_prev)*(alpha/omega); rho_prev = rho
    if (tid == 0) { s_beta = (rho / rho_prev) * (alpha / omega); rho_prev = rho; }
    __syncthreads();
    // 3. p = r + beta*(p - omega*v)
    for (int i = tid; i < n; i += NT) pr[i] = rr[i] + s_beta * (pr[i] - omega * vr[i]);
    __syncthreads();
    // 4. v = A_c·p
    dslash(vr, pr);
    __syncthreads();
    // 5. rtv = <r_tilde, v>; alpha = rho/rtv
    {
      LatticeComplex<T> sum(0,0);
      for (int i = tid; i < n; i += NT) sum += rtr[i].conj() * vr[i];
      _CS_REDUCE(sum);
      if (tid == 0) { rtv = sred[0]; s_alpha = rho / rtv; alpha = s_alpha; }
      __syncthreads();
    }
    // 6. s = r - alpha*v
    for (int i = tid; i < n; i += NT) sr[i] = rr[i] - alpha * vr[i];
    __syncthreads();
    // 7. t = A_c·s
    dslash(tr, sr);
    __syncthreads();
    // 8. ts = <t,s>, tt = <t,t>; omega = ts/tt
    {
      LatticeComplex<T> sum(0,0), sum2(0,0);
      for (int i = tid; i < n; i += NT) {
        sum  += tr[i].conj() * sr[i];
        sum2 += tr[i].conj() * tr[i];
      }
      sred[tid] = sum; __syncthreads();
      for (int k = NT/2; k > 0; k >>= 1) { if (tid < k) sred[tid] += sred[tid+k]; __syncthreads(); }
      if (tid == 0) ts = sred[0];
      __syncthreads();
      sred[tid] = sum2; __syncthreads();
      for (int k = NT/2; k > 0; k >>= 1) { if (tid < k) sred[tid] += sred[tid+k]; __syncthreads(); }
      if (tid == 0) { tt = sred[0]; s_omega = ts / tt; omega = s_omega; }
      __syncthreads();
    }
    // 9. r = s - omega*t ; x = x + alpha*p + omega*s
    for (int i = tid; i < n; i += NT) {
      rr[i] = sr[i] - omega * tr[i];
      xr[i] = xr[i] + alpha * pr[i] + omega * sr[i];
    }
    __syncthreads();
    // 10. ||r||² and convergence
    {
      LatticeComplex<T> sum(0,0);
      for (int i = tid; i < n; i += NT) sum += rr[i].conj() * rr[i];
      _CS_REDUCE(sum);
      if (tid == 0) { s_norm2 = sred[0]; }
      __syncthreads();
      if (s_norm2.real() < target * target) break;
    }
  }
}
#undef _CS_REDUCE

// ====================================================================
// COOPERATIVE-GROUPS PARALLEL FUSED COARSE BiStabCG SOLVE (2026-08-02)
// --------------------------------------------------------------------
// Replaces the per-iteration coarse path (bistabcg_iter_coarse: ~14 tiny
// kernel launches + 1 host sync per iteration, ~1.3 ms/iter on this
// WSL2/V100 box — the launch/execution overhead, NOT compute) with ONE
// cooperative kernel that runs the ENTIRE BiStabCG solve in-kernel.
//
// grid.sync() is the only "barrier" (~2-5 us each); per iteration there
// are 4 cross-block reductions (rho, rtv, [ts,tt], norm2) = 8 grid.syncs,
// plus the wide 33-tensor dslash, all grid-strided across n elements.
//
// Launch: cudaLaunchCooperativeKernel with grid = ceil(n/NT) blocks.
// Requires the grid to be co-resident (n <= ~50k elements for NT=256 on a
// V100) — always true for the coarsest level of our lattices.
//
// @param partials  scratch [gridDim.x] LatticeComplex — per-block dot sums
// @param target    convergence target (||r|| < target), target = tol*||rhs||
// ====================================================================
template <typename T, int NT>
__global__ void multigrid_coarse_solve_cg(
    void *x, void *rhs, void *r_tilde, void *r, void *p, void *v, void *s,
    void *t, void *sitting, void *hop_nn, void *hop_diag,
    int E, int X, int Y, int Z, int Lt, int max_iter, T tol,
    void *partials, void *breakdown_ptr) {
  cg::grid_group grid = cg::this_grid();
  const int n = E * X * Y * Z * Lt;
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int nblocks = gridDim.x;
  const int NT_local = NT;
  __shared__ LatticeComplex<T> sred[NT];

  LatticeComplex<T> *xr  = static_cast<LatticeComplex<T>*>(x);
  LatticeComplex<T> *br  = static_cast<LatticeComplex<T>*>(rhs);
  LatticeComplex<T> *rtr = static_cast<LatticeComplex<T>*>(r_tilde);
  LatticeComplex<T> *rr  = static_cast<LatticeComplex<T>*>(r);
  LatticeComplex<T> *pr  = static_cast<LatticeComplex<T>*>(p);
  LatticeComplex<T> *vr  = static_cast<LatticeComplex<T>*>(v);
  LatticeComplex<T> *sr  = static_cast<LatticeComplex<T>*>(s);
  LatticeComplex<T> *tr  = static_cast<LatticeComplex<T>*>(t);
  const LatticeComplex<T> *sit = static_cast<const LatticeComplex<T>*>(sitting);
  const LatticeComplex<T> *nn  = static_cast<const LatticeComplex<T>*>(hop_nn);
  const LatticeComplex<T> *dg  = static_cast<const LatticeComplex<T>*>(hop_diag);
  LatticeComplex<T> *prt = static_cast<LatticeComplex<T>*>(partials);
  int *breakdown = static_cast<int*>(breakdown_ptr);

  // ---- Init vectors (grid-stride) ----
  for (int i = bid * NT_local + tid; i < n; i += nblocks * NT_local) {
    xr[i] = LatticeComplex<T>(0,0);
    rr[i] = br[i];
    rtr[i] = br[i];
    pr[i] = LatticeComplex<T>(0,0);
    vr[i] = LatticeComplex<T>(0,0);
    sr[i] = LatticeComplex<T>(0,0);
    tr[i] = LatticeComplex<T>(0,0);
  }
  grid.sync();

  // ---- ||rhs|| (r0) via block reduction ----
  {
    LatticeComplex<T> sum(0,0);
    for (int i = bid * NT_local + tid; i < n; i += nblocks * NT_local)
      sum += br[i].conj() * br[i];
    sred[tid] = sum; __syncthreads();
    for (int k = NT_local/2; k > 0; k >>= 1) {
      if (tid < k) sred[tid] += sred[tid+k];
      __syncthreads();
    }
    if (tid == 0) prt[bid] = sred[0];
    grid.sync();
    // every block sums the partials locally
    LatticeComplex<T> tot(0,0);
    for (int b = 0; b < nblocks; b++) tot += prt[b];
    bool bad_norm = multigrid_cg_bad(tot) || tot.real() < (T)0;
    if (multigrid_cg_abort<T>(grid, tid, breakdown, bad_norm)) return;
    T r0 = sqrt(tot.real() > 0 ? tot.real() : 0);
    T target = tol * r0;
    bool bad_target = (tol != tol) || fabs(tol) == INFINITY || tol < (T)0 ||
                      (target != target) || fabs(target) == INFINITY;
    if (multigrid_cg_abort<T>(grid, tid, breakdown, bad_target)) return;
    grid.sync();  // before next block-partial overwrites prt[]

    if (r0 < (T)1e-4) return;
    const T scale = tot.real();

    // ---- Wide 33-tensor coarse dslash (out = A_c·in), grid-stride ----
    auto dslash = [&](LatticeComplex<T> *out, const LatticeComplex<T> *in) {
      int vol = X*Y*Z*Lt;
      int stride_YZT = Y*Z*Lt, stride_ZT = Z*Lt;
      int str_Ein = vol, str_Eout = E*vol, str_dir = E*str_Eout, str_pm = 4*str_dir;
      int dg_str_s1 = 2*6*E*str_Eout, dg_str_s2 = 6*E*str_Eout, dg_str_pair = E*str_Eout;
      int d1s[6] = {0,0,0,1,1,2}, d2s[6] = {1,2,3,2,3,3};
      int offs[4] = {stride_YZT, stride_ZT, Lt, 1}, dims[4] = {X,Y,Z,Lt};
      for (int idx = bid*NT_local + tid; idx < n; idx += nblocks*NT_local) {
        int E_out = idx / (X*Y*Z*Lt);
        int site  = idx - E_out * (X*Y*Z*Lt);
        int xc = site / stride_YZT; int rem = site % stride_YZT;
        int yc = rem / stride_ZT;   rem %= stride_ZT;
        int zc = rem / Lt;          int tc = rem % Lt;
        int coords[4] = {xc,yc,zc,tc};
        LatticeComplex<T> sum(0,0);
        int sb = E_out*str_Eout + site;
        for (int e=0; e<E; e++) sum += sit[sb + e*str_Ein] * in[e*str_Ein + site];
        for (int d=0; d<4; d++) {
          int fwd=(coords[d]+1)%dims[d], bwd=(coords[d]-1+dims[d])%dims[d];
          int fs = site - coords[d]*offs[d] + fwd*offs[d];
          int bs = site - coords[d]*offs[d] + bwd*offs[d];
          int nb = E_out*str_Eout + site;
          for (int e=0; e<E; e++) {
            sum += nn[0*str_pm + d*str_dir + nb + e*str_Ein] * in[e*str_Ein + fs];
            sum += nn[1*str_pm + d*str_dir + nb + e*str_Ein] * in[e*str_Ein + bs];
          }
        }
        for (int pi=0; pi<6; pi++) {
          int d1=d1s[pi], d2=d2s[pi];
          for (int s1i=0; s1i<2; s1i++) for (int s2i=0; s2i<2; s2i++) {
            int sgn1 = s1i==0?1:-1, sgn2 = s2i==0?1:-1;
            int n1 = (coords[d1]+sgn1+dims[d1])%dims[d1];
            int n2 = (coords[d2]+sgn2+dims[d2])%dims[d2];
            int ns = site;
            if (d1==0) ns = ns - xc*stride_YZT + n1*stride_YZT;
            else if (d1==1) ns = ns - yc*stride_ZT + n1*stride_ZT;
            else if (d1==2) ns = ns - zc*Lt + n1*Lt;
            else ns = ns - tc + n1;
            int cur = (d2==0)?(ns/stride_YZT):(d2==1)?((ns%stride_YZT)/stride_ZT):(d2==2)?((ns%stride_ZT)/Lt):(ns%Lt);
            ns += (n2-cur)*offs[d2];
            int db = s1i*dg_str_s1 + s2i*dg_str_s2 + pi*dg_str_pair + E_out*str_Eout + site;
            for (int e=0; e<E; e++) sum += dg[db + e*str_Ein] * in[e*str_Ein + ns];
          }
        }
        out[idx] = sum;
      }
    };

    // ---- Block dot: computes <a,b> partial into prt[bid] ----
    auto block_dot = [&](const LatticeComplex<T>* a, const LatticeComplex<T>* b) {
      LatticeComplex<T> sum(0,0);
      for (int i = bid*NT_local + tid; i < n; i += nblocks*NT_local)
        sum += a[i].conj() * b[i];
      sred[tid] = sum; __syncthreads();
      for (int k = NT_local/2; k > 0; k >>= 1) {
        if (tid < k) sred[tid] += sred[tid+k];
        __syncthreads();
      }
      if (tid == 0) prt[bid] = sred[0];
      grid.sync();
      LatticeComplex<T> tot(0,0);
      for (int b = 0; b < nblocks; b++) tot += prt[b];
      grid.sync();  // protect prt[] before next block_dot overwrites
      return tot;
    };

    LatticeComplex<T> rho(1,0), rho_prev(1,0), alpha(1,0), omega(1,0);
    for (int it = 0; it < max_iter; ++it) {
      // 1. rho = <rt, r>
      rho = block_dot(rtr, rr);
      if (multigrid_cg_abort<T>(
              grid, tid, breakdown,
              multigrid_cg_bad(rho) ||
                  multigrid_cg_near_zero(rho, scale)))
        break;
      // 2. beta; rho_prev = rho
      bool bad_recurrence = multigrid_cg_bad(rho_prev) ||
                            multigrid_cg_bad(alpha) ||
                            multigrid_cg_bad(omega) ||
                            multigrid_cg_abs1(omega) <= (T)1e-13 ||
                            (it > 0 && multigrid_cg_near_zero(rho_prev, scale));
      if (multigrid_cg_abort<T>(grid, tid, breakdown, bad_recurrence))
        break;
      LatticeComplex<T> beta = (rho / rho_prev) * (alpha / omega);
      if (multigrid_cg_abort<T>(grid, tid, breakdown,
                                multigrid_cg_bad(beta)))
        break;
      rho_prev = rho;
      // 3. p = r + beta*(p - omega*v)
      for (int i = bid*NT_local + tid; i < n; i += nblocks*NT_local)
        pr[i] = rr[i] + beta * (pr[i] - omega * vr[i]);
      __syncthreads();
      // 4. v = A_c·p
      dslash(vr, pr);
      __syncthreads();
      // 5. rtv; alpha = rho/rtv
      LatticeComplex<T> rtv = block_dot(rtr, vr);
      if (multigrid_cg_abort<T>(
              grid, tid, breakdown,
              multigrid_cg_bad(rtv) ||
                  multigrid_cg_near_zero(rtv, scale)))
        break;
      alpha = rho / rtv;
      if (multigrid_cg_abort<T>(grid, tid, breakdown,
                                multigrid_cg_bad(alpha)))
        break;
      // 6. s = r - alpha*v
      for (int i = bid*NT_local + tid; i < n; i += nblocks*NT_local)
        sr[i] = rr[i] - alpha * vr[i];
      __syncthreads();
      // 7. t = A_c·s
      dslash(tr, sr);
      __syncthreads();
      // 8. ts = <t,s>, tt = <t,t>  (one pass)
      {
        LatticeComplex<T> s1(0,0), s2(0,0);
        for (int i = bid*NT_local + tid; i < n; i += nblocks*NT_local) {
          s1 += tr[i].conj() * sr[i];
          s2 += tr[i].conj() * tr[i];
        }
        sred[tid] = s1; __syncthreads();
        for (int k = NT_local/2; k > 0; k >>= 1) {
          if (tid < k) sred[tid] += sred[tid+k];
          __syncthreads();
        }
        if (tid == 0) prt[bid] = sred[0];
        grid.sync();
        LatticeComplex<T> ts(0,0);
        for (int b = 0; b < nblocks; b++) ts += prt[b];
        grid.sync();
        sred[tid] = s2; __syncthreads();
        for (int k = NT_local/2; k > 0; k >>= 1) {
          if (tid < k) sred[tid] += sred[tid+k];
          __syncthreads();
        }
        if (tid == 0) prt[bid] = sred[0];
        grid.sync();
        LatticeComplex<T> tt(0,0);
        for (int b = 0; b < nblocks; b++) tt += prt[b];
        grid.sync();
        if (multigrid_cg_abort<T>(
                grid, tid, breakdown,
                multigrid_cg_bad(ts) || multigrid_cg_bad(tt) ||
                    multigrid_cg_near_zero(tt, scale)))
          break;
        omega = ts / tt;
        if (multigrid_cg_abort<T>(
                grid, tid, breakdown,
                multigrid_cg_bad(omega) ||
                    multigrid_cg_abs1(omega) <= (T)1e-13))
          break;
      }
      // 9. r = s - omega*t ; x += alpha*p + omega*s
      for (int i = bid*NT_local + tid; i < n; i += nblocks*NT_local) {
        rr[i] = sr[i] - omega * tr[i];
        xr[i] = xr[i] + alpha * pr[i] + omega * sr[i];
      }
      __syncthreads();
      // 10. ||r||^2 and convergence
      LatticeComplex<T> nr = block_dot(rr, rr);
      if (multigrid_cg_abort<T>(
              grid, tid, breakdown,
              multigrid_cg_bad(nr) || nr.real() < (T)0))
        break;
      if (nr.real() < target * target) break;
    }
  }
}

// Explicit template instantiations
template __global__ void multigrid_restrict<float>(
    void *coarse_out, void *fine_in, void *null_vecs, int E, int e, int Xf,
    int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc);
template __global__ void multigrid_restrict<double>(
    void *coarse_out, void *fine_in, void *null_vecs, int E, int e, int Xf,
    int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc);
template __global__ void multigrid_prolong<float>(
    void *fine_out, void *coarse_in, void *null_vecs, int E, int e, int Xf,
    int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc);
template __global__ void multigrid_prolong<double>(
    void *fine_out, void *coarse_in, void *null_vecs, int E, int e, int Xf,
    int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc);
template __global__ void multigrid_restrict_cast<float, float>(
    void *coarse_out, void *fine_in, void *null_vecs, int E, int e, int Xf,
    int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc);
template __global__ void multigrid_restrict_cast<float, double>(
    void *coarse_out, void *fine_in, void *null_vecs, int E, int e, int Xf,
    int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc);
template __global__ void multigrid_restrict_cast<double, float>(
    void *coarse_out, void *fine_in, void *null_vecs, int E, int e, int Xf,
    int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc);
template __global__ void multigrid_restrict_cast<double, double>(
    void *coarse_out, void *fine_in, void *null_vecs, int E, int e, int Xf,
    int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc);
template __global__ void multigrid_prolong_cast<float, float>(
    void *fine_out, void *coarse_in, void *null_vecs, int E, int e, int Xf,
    int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc);
template __global__ void multigrid_prolong_cast<float, double>(
    void *fine_out, void *coarse_in, void *null_vecs, int E, int e, int Xf,
    int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc);
template __global__ void multigrid_prolong_cast<double, float>(
    void *fine_out, void *coarse_in, void *null_vecs, int E, int e, int Xf,
    int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc);
template __global__ void multigrid_prolong_cast<double, double>(
    void *fine_out, void *coarse_in, void *null_vecs, int E, int e, int Xf,
    int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc);
template __global__ void multigrid_coarse_dslash<float>(
    void *fermion_out, void *fermion_in, void *hopping, void *sitting,
    int E, int X, int Y, int Z, int Lt);
template __global__ void multigrid_coarse_dslash<double>(
    void *fermion_out, void *fermion_in, void *hopping, void *sitting,
    int E, int X, int Y, int Z, int Lt);
template __global__ void multigrid_coarse_dslash_wide<float>(
    void *fermion_out, void *fermion_in, void *sitting, void *hop_nn,
    void *hop_diag, int E, int X, int Y, int Z, int Lt);
template __global__ void multigrid_coarse_dslash_wide<double>(
    void *fermion_out, void *fermion_in, void *sitting, void *hop_nn,
    void *hop_diag, int E, int X, int Y, int Z, int Lt);
template __global__ void multigrid_coarse_dslash_wide_halo<float>(
    void *fermion_out, void *fermion_in, void *halo, void *sitting,
    void *hop_nn, void *hop_diag, int E, int X, int Y, int Z, int Lt);
template __global__ void multigrid_coarse_dslash_wide_halo<double>(
    void *fermion_out, void *fermion_in, void *halo, void *sitting,
    void *hop_nn, void *hop_diag, int E, int X, int Y, int Z, int Lt);
template __global__ void multigrid_coarse_solve<float>(
    void*, void*, void*, void*, void*, void*, void*, void*, void*, void*,
    void*, int, int, int, int, int, int, float);
template __global__ void multigrid_coarse_solve<double>(
    void*, void*, void*, void*, void*, void*, void*, void*, void*, void*,
    void*, int, int, int, int, int, int, double);
template __global__ void multigrid_coarse_solve_cg<float, 256>(
    void*, void*, void*, void*, void*, void*, void*, void*, void*, void*,
    void*, int, int, int, int, int, int, float, void*, void*);
template __global__ void multigrid_coarse_solve_cg<double, 256>(
    void*, void*, void*, void*, void*, void*, void*, void*, void*, void*,
    void*, int, int, int, int, int, int, double, void*, void*);
// ---- Parity-split ↔ full-site layout conversion kernels ----
// Layouts:
//   Full-site:   [sc, X, Y, Z, Lt]       — all sites, contiguous t
//   Parity-split:[2, sc, X, Y, Z, Lt/2]  — channel 0 = even, channel 1 = odd
//
// CRITICAL — parity convention (checkerboard, matches tools.oooxyzt2poooxyzt):
//   A full site (x,y,z,t) belongs to parity p = (x+y+z+t) % 2.
//   Let eo = (x+y+z) % 2 (spatial parity).  For a given spatial site:
//     * even parity (p=0): t_full = 2*t_half + eo
//     * odd  parity (p=1): t_full = 2*t_half + (1 - eo)
//   So t_half is NOT simply "even/odd t"; the t-offset depends on (x+y+z)%2.
//   (Historical bug: these kernels used t_full = 2*t_half+1 unconditionally,
//   which is only correct when (x+y+z) is even.  For (x+y+z) odd the odd
//   channel lives on EVEN t-slices, so the mapping was scrambled and every
//   V-cycle correction blew up the residual.)
template <typename T>
__global__ void multigrid_odd_to_full(void *full_out, void *odd_in,
                                       int sc, int X, int Y, int Z, int Lt_full) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int Lt_half = Lt_full / 2;
  int vol_half = X * Y * Z * Lt_half;
  int total_output = sc * vol_half;
  if (idx >= total_output) return;

  // --- Decompose idx into (sc, x, y, z, t_half) for parity-split layout ---
  int sc_idx = idx / vol_half;
  int site = idx - sc_idx * vol_half;
  int stride_YZT_half = Y * Z * Lt_half;
  int stride_ZT_half = Z * Lt_half;
  int x = site / stride_YZT_half;
  int rem = site - x * stride_YZT_half;
  int y = rem / stride_ZT_half;
  rem -= y * stride_ZT_half;
  int z = rem / Lt_half;
  int t_half = rem - z * Lt_half;

  // --- Map odd-channel → full-site t: t_full = 2*t_half + (1 - eo) ---
  int eo = (x + y + z) & 1;        // spatial parity
  int t_full = 2 * t_half + (1 - eo);
  int vol_full = X * Y * Z * Lt_full;
  int stride_YZT_full = Y * Z * Lt_full;
  int stride_ZT_full = Z * Lt_full;
  int dest_idx = sc_idx * vol_full
               + x * stride_YZT_full
               + y * stride_ZT_full
               + z * Lt_full
               + t_full;

  LatticeComplex<T> *d = static_cast<LatticeComplex<T>*>(full_out);
  LatticeComplex<T> *s = static_cast<LatticeComplex<T>*>(odd_in);
  d[dest_idx] = s[idx];
}

template <typename T>
__global__ void multigrid_even_to_full(void *full_out, void *even_in,
                                       int sc, int X, int Y, int Z, int Lt_full) {
  // Even channel → full-site.  Same checkerboard convention as the odd kernel:
  //   even parity (p=0): t_full = 2*t_half + eo,  eo = (x+y+z)%2.
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int Lt_half = Lt_full / 2;
  int vol_half = X * Y * Z * Lt_half;
  int total_output = sc * vol_half;
  if (idx >= total_output) return;

  int sc_idx = idx / vol_half;
  int site = idx - sc_idx * vol_half;
  int stride_YZT_half = Y * Z * Lt_half;
  int stride_ZT_half = Z * Lt_half;
  int x = site / stride_YZT_half;
  int rem = site - x * stride_YZT_half;
  int y = rem / stride_ZT_half;
  rem -= y * stride_ZT_half;
  int z = rem / Lt_half;
  int t_half = rem - z * Lt_half;

  int eo = (x + y + z) & 1;
  int t_full = 2 * t_half + eo;
  int vol_full = X * Y * Z * Lt_full;
  int stride_YZT_full = Y * Z * Lt_full;
  int stride_ZT_full = Z * Lt_full;
  int dest_idx = sc_idx * vol_full
               + x * stride_YZT_full
               + y * stride_ZT_full
               + z * Lt_full
               + t_full;

  LatticeComplex<T> *d = static_cast<LatticeComplex<T>*>(full_out);
  LatticeComplex<T> *s = static_cast<LatticeComplex<T>*>(even_in);
  d[dest_idx] = s[idx];
}

template <typename T>
__global__ void multigrid_full_to_even(void *even_out, void *full_in,
                                       int sc, int X, int Y, int Z, int Lt_full) {
  // Full-site even-parity site → even channel.
  // even channel: t_full = 2*t_half + eo  ⇒  t_half = (t_full - eo)/2.
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int Lt_half = Lt_full / 2;
  int vol_half = X * Y * Z * Lt_half;
  int total_output = sc * vol_half;
  if (idx >= total_output) return;

  int sc_idx = idx / vol_half;
  int site = idx - sc_idx * vol_half;
  int stride_YZT_half = Y * Z * Lt_half;
  int stride_ZT_half = Z * Lt_half;
  int x = site / stride_YZT_half;
  int rem = site - x * stride_YZT_half;
  int y = rem / stride_ZT_half;
  rem -= y * stride_ZT_half;
  int z = rem / Lt_half;
  int t_half = rem - z * Lt_half;

  int eo = (x + y + z) & 1;
  int t_full = 2 * t_half + eo;
  int vol_full = X * Y * Z * Lt_full;
  int stride_YZT_full = Y * Z * Lt_full;
  int stride_ZT_full = Z * Lt_full;
  int src_idx = sc_idx * vol_full
              + x * stride_YZT_full
              + y * stride_ZT_full
              + z * Lt_full
              + t_full;

  LatticeComplex<T> *d = static_cast<LatticeComplex<T>*>(even_out);
  LatticeComplex<T> *s = static_cast<LatticeComplex<T>*>(full_in);
  d[idx] = s[src_idx];
}

template <typename T>
__global__ void multigrid_full_to_odd(void *odd_out, void *full_in,
                                       int sc, int X, int Y, int Z, int Lt_full) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int Lt_half = Lt_full / 2;
  int vol_half = X * Y * Z * Lt_half;
  int total_output = sc * vol_half;
  if (idx >= total_output) return;

  // --- Decompose idx into (sc, x, y, z, t_half) for parity-split layout ---
  int sc_idx = idx / vol_half;
  int site = idx - sc_idx * vol_half;
  int stride_YZT_half = Y * Z * Lt_half;
  int stride_ZT_half = Z * Lt_half;
  int x = site / stride_YZT_half;
  int rem = site - x * stride_YZT_half;
  int y = rem / stride_ZT_half;
  rem -= y * stride_ZT_half;
  int z = rem / Lt_half;
  int t_half = rem - z * Lt_half;

  // --- Map full-site odd-parity site → odd-channel t_half ---
  // odd channel: t_full = 2*t_half + (1-eo)  ⇒  t_half = (t_full + eo - 1)/2
  int eo = (x + y + z) & 1;        // spatial parity
  int t_full = 2 * t_half + (1 - eo);
  int vol_full = X * Y * Z * Lt_full;
  int stride_YZT_full = Y * Z * Lt_full;
  int stride_ZT_full = Z * Lt_full;
  int src_idx = sc_idx * vol_full
              + x * stride_YZT_full
              + y * stride_ZT_full
              + z * Lt_full
              + t_full;

  LatticeComplex<T> *d = static_cast<LatticeComplex<T>*>(odd_out);
  LatticeComplex<T> *s = static_cast<LatticeComplex<T>*>(full_in);
  d[idx] = s[src_idx];
}
// Template instantiations for conversion kernels
template __global__ void multigrid_odd_to_full<float>(
    void*, void*, int, int, int, int, int);
template __global__ void multigrid_odd_to_full<double>(
    void*, void*, int, int, int, int, int);
template __global__ void multigrid_even_to_full<float>(
    void*, void*, int, int, int, int, int);
template __global__ void multigrid_even_to_full<double>(
    void*, void*, int, int, int, int, int);
template __global__ void multigrid_full_to_odd<float>(
    void*, void*, int, int, int, int, int);
template __global__ void multigrid_full_to_odd<double>(
    void*, void*, int, int, int, int, int);
template __global__ void multigrid_full_to_even<float>(
    void*, void*, int, int, int, int, int);
template __global__ void multigrid_full_to_even<double>(
    void*, void*, int, int, int, int, int);
} // namespace qcu
