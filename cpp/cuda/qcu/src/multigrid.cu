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

  // Strides for null vectors and coarse vector
  int nv_stride_E = e * fine_vol;
  int coarse_stride_E = Xc * Yc * Zc * Tc;

  LatticeComplex<T> sum(0.0, 0.0);
  int fine_idx = global_idx;

  for (int E_idx = 0; E_idx < E; E_idx++) {
    int nv_idx = E_idx * nv_stride_E + fine_idx;
    int coarse_idx =
        E_idx * coarse_stride_E +
        (ix_c + Xc * (iy_c + Yc * (iz_c + Zc * it_c)));
    sum += nv[nv_idx] * cin[coarse_idx];
  }
  out[global_idx] = sum;
}
template <typename T>
__global__ void multigrid_coarse_dslash(void *fermion_out, void *fermion_in,
                                         void *hopping, void *sitting,
                                         int E, int X, int Y, int Z, int T) {
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int vol = X * Y * Z * T;
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
  int stride_YZT = Y * Z * T;
  int stride_ZT = Z * T;
  int x = site / stride_YZT;
  int rest = site - x * stride_YZT;
  int y = rest / stride_ZT;
  rest -= y * stride_ZT;
  int z = rest / T;
  int t = rest - z * T;

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
  int dir_offsets[4] = {stride_YZT, stride_ZT, T, 1};
  int dir_dims[4] = {X, Y, Z, T};
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
    int E, int X, int Y, int Z, int T);
template __global__ void multigrid_coarse_dslash<double>(
    void *fermion_out, void *fermion_in, void *hopping, void *sitting,
    int E, int X, int Y, int Z, int T);
} // namespace qcu
