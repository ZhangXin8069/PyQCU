#include "../include/qcu.h"
#pragma optimize(5)
namespace qcu {
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
  int nv_stride_E = e * fine_vol;

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
          for (int e_idx = 0; e_idx < e; e_idx++) {
            int fine_idx = e_idx * fine_vol + fine_site;
            int nv_idx = E_idx * nv_stride_E + fine_idx;
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

  // Strides for null vectors and coarse vector.
  // The null-vector and coarse/fine vectors are C-order tensors
  // [E, e, X, Y, Z, T] and [E, X, Y, Z, T] respectively, i.e. the LAST
  // (t) dimension is contiguous (stride 1).  The coarse-site index must
  // therefore use t-fastest strides:  x*(Yc*Zc*Tc) + y*(Zc*Tc) + z*Tc + t.
  // (Historical bug: this used x + Xc*y + Xc*Yc*z + Xc*Yc*Zc*t — the
  // transpose/X-fastest convention — which mismatches the tensor layout
  // for any coarse site with t>0 or mixed coordinates.)
  int nv_stride_E = e * fine_vol;
  int coarse_stride_E = Xc * Yc * Zc * Tc;
  int coarse_stride_YZT = Yc * Zc * Tc;
  int coarse_stride_ZT = Zc * Tc;
  // C-order coarse-site index (t fastest):
  int coarse_site = ix_c * coarse_stride_YZT + iy_c * coarse_stride_ZT +
                    iz_c * Tc + it_c;

  LatticeComplex<T> sum(0.0, 0.0);
  int fine_idx = global_idx;

  for (int E_idx = 0; E_idx < E; E_idx++) {
    int nv_idx = E_idx * nv_stride_E + fine_idx;
    int coarse_idx = E_idx * coarse_stride_E + coarse_site;
    sum += nv[nv_idx] * cin[coarse_idx];
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
