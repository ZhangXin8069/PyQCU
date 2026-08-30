#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include "../include/qcu.h"
#include "../python/pyqcu.h"

namespace qcu {
namespace {

template <typename T>
__device__ inline size_t strict_null_index(
    int coarse_dof, int fine_dof, int coarse_component, int fine_component,
    int xf, int yf, int zf, int tf, int Xf, int Yf, int Zf, int Tf,
    int Xc, int Yc, int Zc, int Tc) {
  const int bx = Xf / Xc;
  const int by = Yf / Yc;
  const int bz = Zf / Zc;
  const int bt = Tf / Tc;
  const int xc = xf / bx;
  const int yc = yf / by;
  const int zc = zf / bz;
  const int tc = tf / bt;

  size_t index = static_cast<size_t>(coarse_component);
  index = index * static_cast<size_t>(fine_dof) + fine_component;
  index = index * static_cast<size_t>(Xc) + xc;
  index = index * static_cast<size_t>(bx) + (xf - xc * bx);
  index = index * static_cast<size_t>(Yc) + yc;
  index = index * static_cast<size_t>(by) + (yf - yc * by);
  index = index * static_cast<size_t>(Zc) + zc;
  index = index * static_cast<size_t>(bz) + (zf - zc * bz);
  index = index * static_cast<size_t>(Tc) + tc;
  index = index * static_cast<size_t>(bt) + (tf - tc * bt);
  return index;
}

__device__ inline int strict_full_site(int x, int y, int z, int t,
                                       int Y, int Z, int T) {
  return ((x * Y + y) * Z + z) * T + t;
}

__device__ inline int strict_half_site(int x, int y, int z, int t,
                                       int Y, int Z, int T) {
  return ((x * Y + y) * Z + z) * (T / 2) + t / 2;
}

__device__ inline void strict_decode_half_site(
    int site, int parity, int X, int Y, int Z, int T,
    int &x, int &y, int &z, int &t) {
  const int Th = T / 2;
  const int stride_yzt = Y * Z * Th;
  const int stride_zt = Z * Th;
  x = site / stride_yzt;
  int rest = site - x * stride_yzt;
  y = rest / stride_zt;
  rest -= y * stride_zt;
  z = rest / Th;
  const int th = rest - z * Th;
  const int spatial_parity = (x + y + z) & 1;
  t = 2 * th + (parity ^ spatial_parity);
}

template <typename T>
__global__ void strict_coarse_apply_kernel(
    void *out_ptr, const void *in_ptr, const void *links_ptr,
    const void *onsite_pair_ptr, int E, int X, int Y, int Z, int Lt,
    int onsite_index) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int volume = X * Y * Z * Lt;
  if (index >= E * volume) return;

  LatticeComplex<T> *out = static_cast<LatticeComplex<T> *>(out_ptr);
  const LatticeComplex<T> *in =
      static_cast<const LatticeComplex<T> *>(in_ptr);
  const LatticeComplex<T> *links =
      static_cast<const LatticeComplex<T> *>(links_ptr);
  const LatticeComplex<T> *onsite =
      static_cast<const LatticeComplex<T> *>(onsite_pair_ptr);

  const int row = index / volume;
  const int site = index - row * volume;
  const int stride_yzt = Y * Z * Lt;
  const int stride_zt = Z * Lt;
  const int x = site / stride_yzt;
  int rest = site - x * stride_yzt;
  const int y = rest / stride_zt;
  rest -= y * stride_zt;
  const int z = rest / Lt;
  const int t = rest - z * Lt;

  LatticeComplex<T> sum((T)0, (T)0);
  if (onsite_index >= 0) {
    for (int col = 0; col < E; ++col) {
      const size_t matrix_index =
          (((static_cast<size_t>(onsite_index) * E + row) * E + col) *
               static_cast<size_t>(volume) +
           site);
      sum += onsite[matrix_index] * in[col * volume + site];
    }
  }

  const int coords[4] = {x, y, z, t};
  const int extents[4] = {X, Y, Z, Lt};
  const int offsets[4] = {stride_yzt, stride_zt, Lt, 1};
  for (int dim = 0; dim < 4; ++dim) {
    const int forward_coord = (coords[dim] + 1) % extents[dim];
    const int backward_coord =
        (coords[dim] + extents[dim] - 1) % extents[dim];
    const int forward_site =
        site + (forward_coord - coords[dim]) * offsets[dim];
    const int backward_site =
        site + (backward_coord - coords[dim]) * offsets[dim];
    for (int col = 0; col < E; ++col) {
      const size_t forward_link =
          ((((static_cast<size_t>(0) * 4 + dim) * E + row) * E + col) *
               static_cast<size_t>(volume) +
           site);
      // QUDA stores the backward link at q-mu.  The action at q therefore
      // reads [col,row,q-mu] and conjugates it (matrix adjoint).
      const size_t backward_link =
          ((((static_cast<size_t>(1) * 4 + dim) * E + col) * E + row) *
               static_cast<size_t>(volume) +
           backward_site);
      sum += links[forward_link] * in[col * volume + forward_site];
      sum += links[backward_link].conj() *
             in[col * volume + backward_site];
    }
  }
  out[index] = sum;
}

template <typename T>
__global__ void strict_hopping_parity_kernel(
    void *out_ptr, const void *in_ptr, const void *links_ptr,
    const void *base_ptr, int E, int X, int Y, int Z, int Lt,
    int target_parity) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int half_volume = X * Y * Z * (Lt / 2);
  if (index >= E * half_volume) return;

  LatticeComplex<T> *out = static_cast<LatticeComplex<T> *>(out_ptr);
  const LatticeComplex<T> *in =
      static_cast<const LatticeComplex<T> *>(in_ptr);
  const LatticeComplex<T> *links =
      static_cast<const LatticeComplex<T> *>(links_ptr);
  const LatticeComplex<T> *base =
      static_cast<const LatticeComplex<T> *>(base_ptr);

  const int row = index / half_volume;
  const int half_site = index - row * half_volume;
  int x, y, z, t;
  strict_decode_half_site(
      half_site, target_parity, X, Y, Z, Lt, x, y, z, t);
  const int target_site = strict_full_site(x, y, z, t, Y, Z, Lt);
  const int volume = 2 * half_volume;
  const int coords[4] = {x, y, z, t};
  const int extents[4] = {X, Y, Z, Lt};

  LatticeComplex<T> sum((T)0, (T)0);
  for (int dim = 0; dim < 4; ++dim) {
    int forward[4] = {x, y, z, t};
    int backward[4] = {x, y, z, t};
    forward[dim] = (coords[dim] + 1) % extents[dim];
    backward[dim] = (coords[dim] + extents[dim] - 1) % extents[dim];
    const int forward_site = strict_full_site(
        forward[0], forward[1], forward[2], forward[3], Y, Z, Lt);
    const int backward_site = strict_full_site(
        backward[0], backward[1], backward[2], backward[3], Y, Z, Lt);
    const int forward_half = strict_half_site(
        forward[0], forward[1], forward[2], forward[3], Y, Z, Lt);
    const int backward_half = strict_half_site(
        backward[0], backward[1], backward[2], backward[3], Y, Z, Lt);
    (void)forward_site;
    for (int col = 0; col < E; ++col) {
      const size_t forward_link =
          ((((static_cast<size_t>(0) * 4 + dim) * E + row) * E + col) *
               static_cast<size_t>(volume) +
           target_site);
      const size_t backward_link =
          ((((static_cast<size_t>(1) * 4 + dim) * E + col) * E + row) *
               static_cast<size_t>(volume) +
           backward_site);
      sum += links[forward_link] * in[col * half_volume + forward_half];
      sum += links[backward_link].conj() *
             in[col * half_volume + backward_half];
    }
  }
  out[index] = base == nullptr ? sum : base[index] - sum;
}

template <typename T>
__global__ void strict_onsite_full_to_parity_kernel(
    void *compact_out_ptr, const void *full_in_ptr,
    const void *onsite_pair_ptr, int E, int X, int Y, int Z, int Lt,
    int parity, int onsite_index) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int half_volume = X * Y * Z * (Lt / 2);
  if (index >= E * half_volume) return;
  LatticeComplex<T> *compact_out =
      static_cast<LatticeComplex<T> *>(compact_out_ptr);
  const LatticeComplex<T> *full_in =
      static_cast<const LatticeComplex<T> *>(full_in_ptr);
  const LatticeComplex<T> *onsite =
      static_cast<const LatticeComplex<T> *>(onsite_pair_ptr);

  const int row = index / half_volume;
  const int half_site = index - row * half_volume;
  int x, y, z, t;
  strict_decode_half_site(half_site, parity, X, Y, Z, Lt, x, y, z, t);
  const int full_site = strict_full_site(x, y, z, t, Y, Z, Lt);
  const int volume = 2 * half_volume;
  LatticeComplex<T> sum((T)0, (T)0);
  for (int col = 0; col < E; ++col) {
    const size_t matrix_index =
        (((static_cast<size_t>(onsite_index) * E + row) * E + col) *
             static_cast<size_t>(volume) +
         full_site);
    sum += onsite[matrix_index] * full_in[col * volume + full_site];
  }
  compact_out[index] = sum;
}

template <typename T>
__global__ void strict_join_parities_kernel(
    void *full_out_ptr, const void *target_ptr, const void *other_ptr,
    int E, int X, int Y, int Z, int Lt, int target_parity) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int volume = X * Y * Z * Lt;
  if (index >= E * volume) return;
  LatticeComplex<T> *full_out =
      static_cast<LatticeComplex<T> *>(full_out_ptr);
  const LatticeComplex<T> *target =
      static_cast<const LatticeComplex<T> *>(target_ptr);
  const LatticeComplex<T> *other =
      static_cast<const LatticeComplex<T> *>(other_ptr);

  const int component = index / volume;
  const int site = index - component * volume;
  const int x = site / (Y * Z * Lt);
  int rest = site - x * Y * Z * Lt;
  const int y = rest / (Z * Lt);
  rest -= y * Z * Lt;
  const int z = rest / Lt;
  const int t = rest - z * Lt;
  const int parity = (x + y + z + t) & 1;
  const int half_volume = volume / 2;
  const int half_site = strict_half_site(x, y, z, t, Y, Z, Lt);
  const int compact_index = component * half_volume + half_site;
  full_out[index] = parity == target_parity
                        ? target[compact_index]
                        : other[compact_index];
}

template <typename T>
__global__ void strict_subtract_kernel(
    void *out_ptr, const void *left_ptr, const void *right_ptr, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) return;
  LatticeComplex<T> *out = static_cast<LatticeComplex<T> *>(out_ptr);
  const LatticeComplex<T> *left =
      static_cast<const LatticeComplex<T> *>(left_ptr);
  const LatticeComplex<T> *right =
      static_cast<const LatticeComplex<T> *>(right_ptr);
  out[index] = left[index] - right[index];
}

template <typename T>
__global__ void strict_add_kernel(
    void *out_ptr, const void *correction_ptr, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) return;
  LatticeComplex<T> *out = static_cast<LatticeComplex<T> *>(out_ptr);
  const LatticeComplex<T> *correction =
      static_cast<const LatticeComplex<T> *>(correction_ptr);
  out[index] += correction[index];
}

template <typename T>
__global__ void strict_mr_update_kernel(
    void *x_ptr, void *r_ptr, const void *Ar_ptr,
    LatticeComplex<T> alpha, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) return;
  LatticeComplex<T> *x = static_cast<LatticeComplex<T> *>(x_ptr);
  LatticeComplex<T> *r = static_cast<LatticeComplex<T> *>(r_ptr);
  const LatticeComplex<T> *Ar =
      static_cast<const LatticeComplex<T> *>(Ar_ptr);
  const LatticeComplex<T> old_r = r[index];
  x[index] += alpha * old_r;
  r[index] = old_r - alpha * Ar[index];
}

template <typename T>
__global__ void strict_bicg_p_kernel(
    void *p_ptr, const void *r_ptr, const void *v_ptr,
    LatticeComplex<T> beta, LatticeComplex<T> omega, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) return;
  LatticeComplex<T> *p = static_cast<LatticeComplex<T> *>(p_ptr);
  const LatticeComplex<T> *r =
      static_cast<const LatticeComplex<T> *>(r_ptr);
  const LatticeComplex<T> *v =
      static_cast<const LatticeComplex<T> *>(v_ptr);
  p[index] = r[index] + beta * (p[index] - omega * v[index]);
}

template <typename T>
__global__ void strict_bicg_s_kernel(
    void *s_ptr, const void *r_ptr, const void *v_ptr,
    LatticeComplex<T> alpha, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) return;
  LatticeComplex<T> *s = static_cast<LatticeComplex<T> *>(s_ptr);
  const LatticeComplex<T> *r =
      static_cast<const LatticeComplex<T> *>(r_ptr);
  const LatticeComplex<T> *v =
      static_cast<const LatticeComplex<T> *>(v_ptr);
  s[index] = r[index] - alpha * v[index];
}

template <typename T>
__global__ void strict_bicg_short_update_kernel(
    void *x_ptr, const void *p_ptr, LatticeComplex<T> alpha, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) return;
  LatticeComplex<T> *x = static_cast<LatticeComplex<T> *>(x_ptr);
  const LatticeComplex<T> *p =
      static_cast<const LatticeComplex<T> *>(p_ptr);
  x[index] += alpha * p[index];
}

template <typename T>
__global__ void strict_bicg_update_kernel(
    void *x_ptr, void *r_ptr, const void *p_ptr, const void *s_ptr,
    const void *t_ptr, LatticeComplex<T> alpha,
    LatticeComplex<T> omega, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) return;
  LatticeComplex<T> *x = static_cast<LatticeComplex<T> *>(x_ptr);
  LatticeComplex<T> *r = static_cast<LatticeComplex<T> *>(r_ptr);
  const LatticeComplex<T> *p =
      static_cast<const LatticeComplex<T> *>(p_ptr);
  const LatticeComplex<T> *s =
      static_cast<const LatticeComplex<T> *>(s_ptr);
  const LatticeComplex<T> *t =
      static_cast<const LatticeComplex<T> *>(t_ptr);
  x[index] += alpha * p[index] + omega * s[index];
  r[index] = s[index] - omega * t[index];
}

template <typename T>
__global__ void strict_restrict_parity_kernel(
    void *coarse_out_ptr, const void *fine_in_ptr, const void *null_ptr,
    int E, int e, int Xf, int Yf, int Zf, int Tf,
    int Xc, int Yc, int Zc, int Tc, int parity) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int coarse_volume = Xc * Yc * Zc * Tc;
  if (index >= E * coarse_volume) return;

  LatticeComplex<T> *coarse_out =
      static_cast<LatticeComplex<T> *>(coarse_out_ptr);
  const LatticeComplex<T> *fine_in =
      static_cast<const LatticeComplex<T> *>(fine_in_ptr);
  const LatticeComplex<T> *null_vectors =
      static_cast<const LatticeComplex<T> *>(null_ptr);

  const int coarse_component = index / coarse_volume;
  int coarse_site = index - coarse_component * coarse_volume;
  const int xc = coarse_site / (Yc * Zc * Tc);
  coarse_site -= xc * Yc * Zc * Tc;
  const int yc = coarse_site / (Zc * Tc);
  coarse_site -= yc * Zc * Tc;
  const int zc = coarse_site / Tc;
  const int tc = coarse_site - zc * Tc;
  const int bx = Xf / Xc;
  const int by = Yf / Yc;
  const int bz = Zf / Zc;
  const int bt = Tf / Tc;
  const int half_volume = Xf * Yf * Zf * (Tf / 2);

  LatticeComplex<T> sum((T)0, (T)0);
  for (int dx = 0; dx < bx; ++dx) {
    const int xf = xc * bx + dx;
    for (int dy = 0; dy < by; ++dy) {
      const int yf = yc * by + dy;
      for (int dz = 0; dz < bz; ++dz) {
        const int zf = zc * bz + dz;
        for (int dt = 0; dt < bt; ++dt) {
          const int tf = tc * bt + dt;
          if (((xf + yf + zf + tf) & 1) != parity) continue;
          const int fine_site = strict_half_site(xf, yf, zf, tf, Yf, Zf, Tf);
          for (int fine_component = 0; fine_component < e; ++fine_component) {
            const size_t null_index = strict_null_index<T>(
                E, e, coarse_component, fine_component, xf, yf, zf, tf,
                Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
            sum += null_vectors[null_index].conj() *
                   fine_in[fine_component * half_volume + fine_site];
          }
        }
      }
    }
  }
  coarse_out[index] = sum;
}

template <typename T>
__global__ void strict_prolong_parity_kernel(
    void *fine_out_ptr, const void *coarse_in_ptr, const void *null_ptr,
    int E, int e, int Xf, int Yf, int Zf, int Tf,
    int Xc, int Yc, int Zc, int Tc, int parity) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int half_volume = Xf * Yf * Zf * (Tf / 2);
  if (index >= e * half_volume) return;

  LatticeComplex<T> *fine_out =
      static_cast<LatticeComplex<T> *>(fine_out_ptr);
  const LatticeComplex<T> *coarse_in =
      static_cast<const LatticeComplex<T> *>(coarse_in_ptr);
  const LatticeComplex<T> *null_vectors =
      static_cast<const LatticeComplex<T> *>(null_ptr);
  const int fine_component = index / half_volume;
  const int half_site = index - fine_component * half_volume;
  int xf, yf, zf, tf;
  strict_decode_half_site(
      half_site, parity, Xf, Yf, Zf, Tf, xf, yf, zf, tf);

  const int bx = Xf / Xc;
  const int by = Yf / Yc;
  const int bz = Zf / Zc;
  const int bt = Tf / Tc;
  const int xc = xf / bx;
  const int yc = yf / by;
  const int zc = zf / bz;
  const int tc = tf / bt;
  const int coarse_site = ((xc * Yc + yc) * Zc + zc) * Tc + tc;
  const int coarse_volume = Xc * Yc * Zc * Tc;

  LatticeComplex<T> sum((T)0, (T)0);
  for (int coarse_component = 0; coarse_component < E; ++coarse_component) {
    const size_t null_index = strict_null_index<T>(
        E, e, coarse_component, fine_component, xf, yf, zf, tf,
        Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
    sum += null_vectors[null_index] *
           coarse_in[coarse_component * coarse_volume + coarse_site];
  }
  fine_out[index] = sum;
}

inline void strict_validate_parity_geometry(int X, int Y, int Z, int T) {
  if (X <= 0 || Y <= 0 || Z <= 0 || T <= 0 ||
      (X & 1) || (Y & 1) || (Z & 1) || (T & 1))
    throw std::invalid_argument(
        "strict checkerboard requires positive even X/Y/Z/T");
}

inline void strict_validate_transfer_geometry(
    int E, int e, int Xf, int Yf, int Zf, int Tf,
    int Xc, int Yc, int Zc, int Tc, int parity) {
  if (E <= 0 || e <= 0 || parity < 0 || parity > 1)
    throw std::invalid_argument("invalid strict transfer dof/parity");
  strict_validate_parity_geometry(Xf, Yf, Zf, Tf);
  if (Xc <= 0 || Yc <= 0 || Zc <= 0 || Tc <= 0 ||
      Xf % Xc || Yf % Yc || Zf % Zc || Tf % Tc)
    throw std::invalid_argument("strict transfer geometry is not divisible");
}

template <typename T>
LatticeSet<T> *strict_get_set(void *set_ptrs, int *params) {
  const int index = params[_SET_INDEX_];
  if (index < 0 || index >= 100)
    throw std::out_of_range("strict multigrid set index is outside int64[100]");
  const long long address = static_cast<long long *>(set_ptrs)[index];
  if (address == 0)
    throw std::invalid_argument("strict multigrid lattice set is null");
  return static_cast<LatticeSet<T> *>(reinterpret_cast<void *>(address));
}

inline void strict_check_cuda(cudaError_t status, const char *where) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "PYQCU::SOLVER::STRICT_MG::CUDA: %s: %s\n",
                 where, cudaGetErrorString(status));
    throw std::runtime_error("strict multigrid CUDA failure");
  }
}

template <typename T>
void strict_launch_coarse(
    void *out, const void *in, const void *links, const void *onsite_pair,
    void *set_ptrs, int *params, int E, int X, int Y, int Z, int Tdim,
    int onsite_index) {
  if (E <= 0 || X <= 0 || Y <= 0 || Z <= 0 || Tdim <= 0 ||
      onsite_index < -1 || onsite_index > 1)
    throw std::invalid_argument("invalid strict coarse operator descriptor");
  LatticeSet<T> *set = strict_get_set<T>(set_ptrs, params);
  const int total = E * X * Y * Z * Tdim;
  const int blocks = (total + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_;
  strict_coarse_apply_kernel<T><<<blocks, _BLOCK_SIZE_, 0, set->stream>>>(
      out, in, links, onsite_pair, E, X, Y, Z, Tdim, onsite_index);
  strict_check_cuda(cudaGetLastError(), "coarse apply launch");
  strict_check_cuda(cudaStreamSynchronize(set->stream), "coarse apply sync");
}

template <typename T>
void strict_launch_matpc(
    void *out, const void *in, const void *links, void *scratch,
    void *set_ptrs, int *params, int E, int X, int Y, int Z, int Tdim,
    int parity) {
  if (E <= 0 || parity < 0 || parity > 1)
    throw std::invalid_argument("invalid strict MATPC descriptor");
  strict_validate_parity_geometry(X, Y, Z, Tdim);
  LatticeSet<T> *set = strict_get_set<T>(set_ptrs, params);
  const int total = E * X * Y * Z * (Tdim / 2);
  const int blocks = (total + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_;
  strict_hopping_parity_kernel<T><<<blocks, _BLOCK_SIZE_, 0, set->stream>>>(
      scratch, in, links, nullptr, E, X, Y, Z, Tdim, 1 - parity);
  strict_hopping_parity_kernel<T><<<blocks, _BLOCK_SIZE_, 0, set->stream>>>(
      out, scratch, links, in, E, X, Y, Z, Tdim, parity);
  strict_check_cuda(cudaGetLastError(), "MATPC launch");
  strict_check_cuda(cudaStreamSynchronize(set->stream), "MATPC sync");
}

template <typename T>
void strict_launch_prepare(
    void *out, const void *full_rhs, const void *links,
    const void *onsite_pair, void *scratch, void *set_ptrs, int *params,
    int E, int X, int Y, int Z, int Lt, int parity) {
  if (E <= 0 || parity < 0 || parity > 1)
    throw std::invalid_argument("invalid strict prepare descriptor");
  strict_validate_parity_geometry(X, Y, Z, Lt);
  LatticeSet<T> *set = strict_get_set<T>(set_ptrs, params);
  const int total = E * X * Y * Z * (Lt / 2);
  const int blocks = (total + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_;
  strict_onsite_full_to_parity_kernel<T>
      <<<blocks, _BLOCK_SIZE_, 0, set->stream>>>(
          out, full_rhs, onsite_pair, E, X, Y, Z, Lt, parity, 1);
  strict_onsite_full_to_parity_kernel<T>
      <<<blocks, _BLOCK_SIZE_, 0, set->stream>>>(
          scratch, full_rhs, onsite_pair, E, X, Y, Z, Lt, 1 - parity, 1);
  // out = X_p^-1 b_p - Hhat_pq X_q^-1 b_q.
  strict_hopping_parity_kernel<T><<<blocks, _BLOCK_SIZE_, 0, set->stream>>>(
      out, scratch, links, out, E, X, Y, Z, Lt, parity);
  strict_check_cuda(cudaGetLastError(), "prepare launch");
  strict_check_cuda(cudaStreamSynchronize(set->stream), "prepare sync");
}

template <typename T>
void strict_launch_reconstruct(
    void *full_out, const void *full_rhs, const void *target_solution,
    const void *links, const void *onsite_pair, void *scratch,
    void *set_ptrs, int *params, int E, int X, int Y, int Z, int Lt,
    int parity) {
  if (E <= 0 || parity < 0 || parity > 1)
    throw std::invalid_argument("invalid strict reconstruct descriptor");
  strict_validate_parity_geometry(X, Y, Z, Lt);
  LatticeSet<T> *set = strict_get_set<T>(set_ptrs, params);
  const int half_total = E * X * Y * Z * (Lt / 2);
  const int half_blocks =
      (half_total + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_;
  strict_onsite_full_to_parity_kernel<T>
      <<<half_blocks, _BLOCK_SIZE_, 0, set->stream>>>(
          scratch, full_rhs, onsite_pair, E, X, Y, Z, Lt, 1 - parity, 1);
  // x_q = X_q^-1 b_q - Hhat_qp x_p.
  strict_hopping_parity_kernel<T>
      <<<half_blocks, _BLOCK_SIZE_, 0, set->stream>>>(
          scratch, target_solution, links, scratch,
          E, X, Y, Z, Lt, 1 - parity);
  const int full_total = 2 * half_total;
  const int full_blocks =
      (full_total + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_;
  strict_join_parities_kernel<T>
      <<<full_blocks, _BLOCK_SIZE_, 0, set->stream>>>(
          full_out, target_solution, scratch, E, X, Y, Z, Lt, parity);
  strict_check_cuda(cudaGetLastError(), "reconstruct launch");
  strict_check_cuda(cudaStreamSynchronize(set->stream), "reconstruct sync");
}

template <typename T>
void strict_launch_restrict(
    void *coarse_out, const void *fine_in, const void *null_vectors,
    void *set_ptrs, int *params, int E, int e,
    int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc,
    int parity) {
  strict_validate_transfer_geometry(
      E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc, parity);
  LatticeSet<T> *set = strict_get_set<T>(set_ptrs, params);
  const int total = E * Xc * Yc * Zc * Tc;
  const int blocks = (total + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_;
  strict_restrict_parity_kernel<T><<<blocks, _BLOCK_SIZE_, 0, set->stream>>>(
      coarse_out, fine_in, null_vectors, E, e, Xf, Yf, Zf, Tf,
      Xc, Yc, Zc, Tc, parity);
  strict_check_cuda(cudaGetLastError(), "parity restrict launch");
  strict_check_cuda(cudaStreamSynchronize(set->stream), "parity restrict sync");
}

template <typename T>
void strict_launch_prolong(
    void *fine_out, const void *coarse_in, const void *null_vectors,
    void *set_ptrs, int *params, int E, int e,
    int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc,
    int parity) {
  strict_validate_transfer_geometry(
      E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc, parity);
  LatticeSet<T> *set = strict_get_set<T>(set_ptrs, params);
  const int total = e * Xf * Yf * Zf * (Tf / 2);
  const int blocks = (total + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_;
  strict_prolong_parity_kernel<T><<<blocks, _BLOCK_SIZE_, 0, set->stream>>>(
      fine_out, coarse_in, null_vectors, E, e, Xf, Yf, Zf, Tf,
      Xc, Yc, Zc, Tc, parity);
  strict_check_cuda(cudaGetLastError(), "parity prolong launch");
  strict_check_cuda(cudaStreamSynchronize(set->stream), "parity prolong sync");
}

struct StrictLevelGeometry {
  int E = 0, X = 0, Y = 0, Z = 0, Lt = 0;
  int max_iter = 0;
  size_t full_n = 0, compact_n = 0;
};

inline StrictLevelGeometry strict_read_level(int *params, int level) {
  if (level < 1 || level > 4)
    throw std::out_of_range("strict coarse level must be in [1,4]");
  const int base = _MG_LEVEL1_E_ + (level - 1) * _MG_PARAMS_SIZE_;
  StrictLevelGeometry geometry;
  geometry.E = params[base + 0];
  geometry.X = params[base + 1];
  geometry.Y = params[base + 2];
  geometry.Z = params[base + 3];
  geometry.Lt = params[base + 4];
  geometry.max_iter = params[base + 5];
  strict_validate_parity_geometry(
      geometry.X, geometry.Y, geometry.Z, geometry.Lt);
  if (geometry.E <= 0)
    throw std::invalid_argument("strict coarse level has non-positive dof");
  geometry.full_n = static_cast<size_t>(geometry.E) * geometry.X *
                    geometry.Y * geometry.Z * geometry.Lt;
  geometry.compact_n = geometry.full_n / 2;
  return geometry;
}

template <typename T> struct StrictPersistentLevel {
  StrictLevelGeometry geometry;
  void *full_rhs = nullptr;
  void *pc_rhs = nullptr;
  void *x = nullptr;
};

template <typename T> struct StrictWorkspaceArena {
  void *storage = nullptr;
  void *r = nullptr, *v = nullptr, *tmp = nullptr;
  void *rhat = nullptr, *p = nullptr, *s = nullptr, *t = nullptr;
  void *correction_full = nullptr;
  size_t bytes = 0;

  static size_t align_elements(size_t elements) {
    const size_t alignment = std::max(
        static_cast<size_t>(1),
        static_cast<size_t>(256) / sizeof(LatticeComplex<T>));
    return (elements + alignment - 1) / alignment * alignment;
  }

  void allocate(size_t max_compact, size_t coarsest_compact,
                size_t max_child_full, cudaStream_t stream) {
    size_t cursor = 0;
    auto reserve = [&](size_t count) -> size_t {
      const size_t offset = cursor;
      cursor += align_elements(count);
      return offset;
    };
    const size_t r_offset = reserve(max_compact);
    const size_t v_offset = reserve(max_compact);
    const size_t tmp_offset = reserve(max_compact);
    const size_t rhat_offset = reserve(coarsest_compact);
    const size_t p_offset = reserve(coarsest_compact);
    const size_t s_offset = reserve(coarsest_compact);
    const size_t t_offset = reserve(coarsest_compact);
    const size_t correction_offset = reserve(max_child_full);
    bytes = cursor * sizeof(LatticeComplex<T>);
    if (bytes == 0) return;
    strict_check_cuda(cudaMallocAsync(&storage, bytes, stream),
                      "strict workspace allocation");
    LatticeComplex<T> *base = static_cast<LatticeComplex<T> *>(storage);
    r = base + r_offset;
    v = base + v_offset;
    tmp = base + tmp_offset;
    rhat = base + rhat_offset;
    p = base + p_offset;
    s = base + s_offset;
    t = base + t_offset;
    correction_full = max_child_full == 0 ? nullptr : base + correction_offset;
  }

  void release(cudaStream_t stream) {
    if (storage != nullptr) {
      strict_check_cuda(cudaFreeAsync(storage, stream),
                        "strict workspace release");
      storage = nullptr;
    }
  }
};

template <typename T> class StrictCoarseHierarchy {
 public:
  StrictCoarseHierarchy(LatticeSet<T> *set, void *set_ptrs, int *params,
                        int start_level)
      : set_(set), set_ptrs_(static_cast<long long *>(set_ptrs)),
        params_(params), start_(start_level),
        num_levels_(params[_MG_NUM_LEVEL_]),
        parity_(params[_PARITY_]),
        smoother_steps_(params[_MG_MU_PRE_] > 0 ? params[_MG_MU_PRE_] : 2),
                        persistent_storage_(nullptr), persistent_bytes_(0) {
    try {
    if (num_levels_ < 2 || num_levels_ > 5 || start_ < 1 ||
        start_ >= num_levels_)
      throw std::invalid_argument(
          "strict hierarchy requires 2..5 levels and a valid coarse start");
    if (parity_ != 0 && parity_ != 1)
      throw std::invalid_argument("strict hierarchy parity must be 0 or 1");
    if (params_[_GRID_X_] != 1 || params_[_GRID_Y_] != 1 ||
        params_[_GRID_Z_] != 1 || params_[_GRID_T_] != 1)
      throw std::invalid_argument(
          "strict recursive hierarchy currently supports single rank only");

    levels_ = new StrictPersistentLevel<T>[num_levels_];
    size_t persistent_elements = 0;
    size_t max_compact = 0;
    size_t max_child_full = 0;
    for (int level = start_; level < num_levels_; ++level) {
      levels_[level].geometry = strict_read_level(params_, level);
      const int base = _MG_LEVEL1_E_ + (level - 1) * _MG_PARAMS_SIZE_;
      if (params_[base + 6] != params_[_DATA_TYPE_])
        throw std::invalid_argument(
            "strict recursive hierarchy mixed precision is not enabled yet");
      max_compact = std::max(
          max_compact, levels_[level].geometry.compact_n);
      if (level > start_)
        max_child_full = std::max(
            max_child_full, levels_[level].geometry.full_n);
      persistent_elements +=
          2 * StrictWorkspaceArena<T>::align_elements(
                  levels_[level].geometry.compact_n);
      if (level > start_)
        persistent_elements += StrictWorkspaceArena<T>::align_elements(
            levels_[level].geometry.full_n);
      validate_assets(level);
    }

    persistent_bytes_ = persistent_elements * sizeof(LatticeComplex<T>);
    strict_check_cuda(
        cudaMallocAsync(&persistent_storage_, persistent_bytes_, set_->stream),
        "strict persistent allocation");
    LatticeComplex<T> *base =
        static_cast<LatticeComplex<T> *>(persistent_storage_);
    size_t cursor = 0;
    auto take = [&](size_t count) -> void * {
      LatticeComplex<T> *result = base + cursor;
      cursor += StrictWorkspaceArena<T>::align_elements(count);
      return result;
    };
    for (int level = start_; level < num_levels_; ++level) {
      StrictPersistentLevel<T> &state = levels_[level];
      state.pc_rhs = take(state.geometry.compact_n);
      state.x = take(state.geometry.compact_n);
      if (level > start_) state.full_rhs = take(state.geometry.full_n);
    }
    arena_.allocate(
        max_compact, levels_[num_levels_ - 1].geometry.compact_n,
        max_child_full, set_->stream);
    if (params_[_VERBOSE_] && params_[_NODE_RANK_] == 0) {
      const double persistent_mib =
          static_cast<double>(persistent_bytes_) / (1024.0 * 1024.0);
      const double arena_mib =
          static_cast<double>(arena_.bytes) / (1024.0 * 1024.0);
      std::printf(
          "PYQCU::SOLVER::STRICT_MG::MEMORY:\n "
          "persistent=%.3f MiB arena=%.3f MiB total=%.3f MiB\n",
          persistent_mib, arena_mib, persistent_mib + arena_mib);
    }
    } catch (...) {
      release_noexcept();
      throw;
    }
  }

  ~StrictCoarseHierarchy() { release_noexcept(); }

  void run(void *full_out, const void *full_rhs) {
    cublasPointerMode_t saved_pointer_mode = CUBLAS_POINTER_MODE_HOST;
    if (cublasGetPointerMode(set_->cublasH, &saved_pointer_mode) !=
        CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cannot query cublas pointer mode");
    if (cublasSetPointerMode(set_->cublasH, CUBLAS_POINTER_MODE_HOST) !=
        CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cannot select host cublas pointer mode");
    try {
      solve_level(start_, full_rhs, full_out);
      strict_check_cuda(cudaGetLastError(), "strict recursive V-cycle launch");
      strict_check_cuda(cudaStreamSynchronize(set_->stream),
                        "strict recursive V-cycle sync");
    } catch (...) {
      (void)cublasSetPointerMode(set_->cublasH, saved_pointer_mode);
      throw;
    }
    if (cublasSetPointerMode(set_->cublasH, saved_pointer_mode) !=
        CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cannot restore cublas pointer mode");
  }

  size_t allocated_bytes() const {
    return persistent_bytes_ + arena_.bytes;
  }

  int start_level() const { return start_; }

 private:
  LatticeSet<T> *set_;
  long long *set_ptrs_;
  int *params_;
  int start_, num_levels_, parity_, smoother_steps_;
  StrictPersistentLevel<T> *levels_ = nullptr;
  StrictWorkspaceArena<T> arena_;
  void *persistent_storage_;
  size_t persistent_bytes_;

  void release_noexcept() noexcept {
    if (set_ != nullptr) {
      if (arena_.storage != nullptr) {
        (void)cudaFreeAsync(arena_.storage, set_->stream);
        arena_.storage = nullptr;
      }
      if (persistent_storage_ != nullptr) {
        (void)cudaFreeAsync(persistent_storage_, set_->stream);
        persistent_storage_ = nullptr;
      }
      (void)cudaStreamSynchronize(set_->stream);
    }
    arena_.r = arena_.v = arena_.tmp = nullptr;
    arena_.rhat = arena_.p = arena_.s = arena_.t = nullptr;
    arena_.correction_full = nullptr;
    arena_.bytes = 0;
    persistent_bytes_ = 0;
    delete[] levels_;
    levels_ = nullptr;
  }

  void *asset(int transition, int slot) const {
    const int index = _SET_PTRS_STRICT_COARSE_BASE_ +
                      transition * _SET_PTRS_STRICT_STRIDE_ + slot;
    if (index < 0 || index >= 100 || set_ptrs_[index] == 0)
      throw std::invalid_argument("strict hierarchy asset slot is null");
    return reinterpret_cast<void *>(set_ptrs_[index]);
  }

  void validate_assets(int level) const {
    const int transition = level - 1;
    (void)asset(transition, _SET_PTRS_STRICT_PRECONDITIONED_LINKS_);
    (void)asset(transition, _SET_PTRS_STRICT_ONSITE_PAIR_);
    // raw Y is intentionally optional at solve time: retaining only Yhat and
    // X^-1 after setup saves the largest duplicated operator allocation.
    if (level + 1 < num_levels_)
      (void)asset(level, _SET_PTRS_STRICT_NULL_);
  }

  int blocks(size_t n) const {
    return static_cast<int>((n + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
  }

  LatticeComplex<T> dot(const void *left, const void *right, size_t n) {
    if (n > static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("strict dot vector exceeds cublas int range");
    LatticeComplex<T> result((T)0, (T)0);
    const cublasStatus_t status = _cublasDot<T>(
        set_->cublasH, static_cast<int>(n), left, 1, right, 1, &result);
    if (status != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("strict cublas dot failed");
    strict_check_cuda(cudaStreamSynchronize(set_->stream), "strict dot sync");
    return result;
  }

  static T abs2(const LatticeComplex<T> &value) {
    return value.real() * value.real() + value.imag() * value.imag();
  }

  static bool finite(const LatticeComplex<T> &value) {
    return std::isfinite(static_cast<double>(value.real())) &&
           std::isfinite(static_cast<double>(value.imag()));
  }

  void apply_matpc(int level, void *out, const void *in, void *scratch) {
    const StrictLevelGeometry &g = levels_[level].geometry;
    strict_hopping_parity_kernel<T>
        <<<blocks(g.compact_n), _BLOCK_SIZE_, 0, set_->stream>>>(
            scratch, in,
            asset(level - 1, _SET_PTRS_STRICT_PRECONDITIONED_LINKS_),
            nullptr, g.E, g.X, g.Y, g.Z, g.Lt, 1 - parity_);
    strict_hopping_parity_kernel<T>
        <<<blocks(g.compact_n), _BLOCK_SIZE_, 0, set_->stream>>>(
            out, scratch,
            asset(level - 1, _SET_PTRS_STRICT_PRECONDITIONED_LINKS_),
            in, g.E, g.X, g.Y, g.Z, g.Lt, parity_);
  }

  void prepare(int level, void *out, const void *full_rhs, void *scratch) {
    const StrictLevelGeometry &g = levels_[level].geometry;
    void *links = asset(
        level - 1, _SET_PTRS_STRICT_PRECONDITIONED_LINKS_);
    void *onsite = asset(level - 1, _SET_PTRS_STRICT_ONSITE_PAIR_);
    strict_onsite_full_to_parity_kernel<T>
        <<<blocks(g.compact_n), _BLOCK_SIZE_, 0, set_->stream>>>(
            out, full_rhs, onsite, g.E, g.X, g.Y, g.Z, g.Lt, parity_, 1);
    strict_onsite_full_to_parity_kernel<T>
        <<<blocks(g.compact_n), _BLOCK_SIZE_, 0, set_->stream>>>(
            scratch, full_rhs, onsite, g.E, g.X, g.Y, g.Z, g.Lt,
            1 - parity_, 1);
    strict_hopping_parity_kernel<T>
        <<<blocks(g.compact_n), _BLOCK_SIZE_, 0, set_->stream>>>(
            out, scratch, links, out,
            g.E, g.X, g.Y, g.Z, g.Lt, parity_);
  }

  void reconstruct(int level, void *full_out, const void *full_rhs,
                   const void *target_solution, void *scratch) {
    const StrictLevelGeometry &g = levels_[level].geometry;
    void *links = asset(
        level - 1, _SET_PTRS_STRICT_PRECONDITIONED_LINKS_);
    void *onsite = asset(level - 1, _SET_PTRS_STRICT_ONSITE_PAIR_);
    strict_onsite_full_to_parity_kernel<T>
        <<<blocks(g.compact_n), _BLOCK_SIZE_, 0, set_->stream>>>(
            scratch, full_rhs, onsite, g.E, g.X, g.Y, g.Z, g.Lt,
            1 - parity_, 1);
    strict_hopping_parity_kernel<T>
        <<<blocks(g.compact_n), _BLOCK_SIZE_, 0, set_->stream>>>(
            scratch, target_solution, links, scratch,
            g.E, g.X, g.Y, g.Z, g.Lt, 1 - parity_);
    strict_join_parities_kernel<T>
        <<<blocks(g.full_n), _BLOCK_SIZE_, 0, set_->stream>>>(
            full_out, target_solution, scratch,
            g.E, g.X, g.Y, g.Z, g.Lt, parity_);
  }

  void mr_smooth(int level, int count) {
    StrictPersistentLevel<T> &state = levels_[level];
    const size_t n = state.geometry.compact_n;
    const T floor = std::is_same<T, float>::value ? (T)1e-20 : (T)1e-40;
    for (int iteration = 0; iteration < count; ++iteration) {
      apply_matpc(level, arena_.v, arena_.r, arena_.tmp);
      const LatticeComplex<T> numerator = dot(arena_.v, arena_.r, n);
      const LatticeComplex<T> denominator = dot(arena_.v, arena_.v, n);
      if (!finite(numerator) || !finite(denominator) ||
          abs2(denominator) <= floor)
        break;
      const LatticeComplex<T> alpha = numerator / denominator;
      strict_mr_update_kernel<T>
          <<<blocks(n), _BLOCK_SIZE_, 0, set_->stream>>>(
              state.x, arena_.r, arena_.v, alpha, static_cast<int>(n));
    }
  }

  bool coarsest_bicgstab(int level) {
    StrictPersistentLevel<T> &state = levels_[level];
    const StrictLevelGeometry &g = state.geometry;
    const size_t n = g.compact_n;
    void *r = arena_.r;
    void *v = arena_.v;
    void *tmp = arena_.tmp;
    void *rhat = arena_.rhat;
    void *p = arena_.p;
    void *s = arena_.s;
    void *t = arena_.t;
    const size_t bytes = n * sizeof(LatticeComplex<T>);
    strict_check_cuda(cudaMemsetAsync(state.x, 0, bytes, set_->stream),
                      "strict coarsest x zero");
    strict_check_cuda(cudaMemcpyAsync(
                          r, state.pc_rhs, bytes, cudaMemcpyDeviceToDevice,
                          set_->stream),
                      "strict coarsest rhs copy");
    strict_check_cuda(cudaMemcpyAsync(
                          rhat, state.pc_rhs, bytes, cudaMemcpyDeviceToDevice,
                          set_->stream),
                      "strict coarsest shadow copy");
    strict_check_cuda(cudaMemsetAsync(p, 0, bytes, set_->stream),
                      "strict coarsest p zero");
    strict_check_cuda(cudaMemsetAsync(v, 0, bytes, set_->stream),
                      "strict coarsest v zero");

    const T rhs_norm2 = std::max((T)0, dot(state.pc_rhs, state.pc_rhs, n).real());
    if (rhs_norm2 == (T)0) return true;
    T tolerance = set_->host_argv[_MG_LEVEL1_ATOL_ + level - 1];
    if (!(tolerance > (T)0 && tolerance < (T)1))
      tolerance = std::is_same<T, float>::value ? (T)1e-6 : (T)1e-12;
    const T target = tolerance * tolerance * rhs_norm2;
    const int max_iter = g.max_iter > 0 ? g.max_iter : 100;
    const T floor = std::is_same<T, float>::value ? (T)1e-20 : (T)1e-40;
    LatticeComplex<T> rho_old((T)1, (T)0);
    LatticeComplex<T> alpha((T)1, (T)0);
    LatticeComplex<T> omega((T)1, (T)0);

    for (int iteration = 0; iteration < max_iter; ++iteration) {
      const LatticeComplex<T> rho = dot(rhat, r, n);
      if (!finite(rho) || abs2(rho) <= floor || abs2(omega) <= floor)
        return false;
      const LatticeComplex<T> beta = (rho / rho_old) * (alpha / omega);
      strict_bicg_p_kernel<T>
          <<<blocks(n), _BLOCK_SIZE_, 0, set_->stream>>>(
              p, r, v, beta, omega, static_cast<int>(n));
      apply_matpc(level, v, p, tmp);
      const LatticeComplex<T> denominator = dot(rhat, v, n);
      if (!finite(denominator) || abs2(denominator) <= floor) return false;
      alpha = rho / denominator;
      strict_bicg_s_kernel<T>
          <<<blocks(n), _BLOCK_SIZE_, 0, set_->stream>>>(
              s, r, v, alpha, static_cast<int>(n));
      const T s_norm2 = std::max((T)0, dot(s, s, n).real());
      if (s_norm2 <= target) {
        strict_bicg_short_update_kernel<T>
            <<<blocks(n), _BLOCK_SIZE_, 0, set_->stream>>>(
                state.x, p, alpha, static_cast<int>(n));
        return true;
      }
      apply_matpc(level, t, s, tmp);
      const LatticeComplex<T> tt = dot(t, t, n);
      if (!finite(tt) || abs2(tt) <= floor) return false;
      omega = dot(t, s, n) / tt;
      if (!finite(omega) || abs2(omega) <= floor) return false;
      strict_bicg_update_kernel<T>
          <<<blocks(n), _BLOCK_SIZE_, 0, set_->stream>>>(
              state.x, r, p, s, t, alpha, omega, static_cast<int>(n));
      const T residual_norm2 = std::max((T)0, dot(r, r, n).real());
      if (residual_norm2 <= target) return true;
      rho_old = rho;
    }
    return false;
  }

  void restrict_to_child(int level, const void *fine_compact,
                         void *child_full_rhs) {
    const StrictLevelGeometry &fine = levels_[level].geometry;
    const StrictLevelGeometry &coarse = levels_[level + 1].geometry;
    strict_restrict_parity_kernel<T>
        <<<blocks(coarse.full_n), _BLOCK_SIZE_, 0, set_->stream>>>(
            child_full_rhs, fine_compact,
            asset(level, _SET_PTRS_STRICT_NULL_),
            coarse.E, fine.E,
            fine.X, fine.Y, fine.Z, fine.Lt,
            coarse.X, coarse.Y, coarse.Z, coarse.Lt, parity_);
  }

  void prolong_from_child(int level, const void *child_full,
                          void *fine_compact) {
    const StrictLevelGeometry &fine = levels_[level].geometry;
    const StrictLevelGeometry &coarse = levels_[level + 1].geometry;
    strict_prolong_parity_kernel<T>
        <<<blocks(fine.compact_n), _BLOCK_SIZE_, 0, set_->stream>>>(
            fine_compact, child_full,
            asset(level, _SET_PTRS_STRICT_NULL_),
            coarse.E, fine.E,
            fine.X, fine.Y, fine.Z, fine.Lt,
            coarse.X, coarse.Y, coarse.Z, coarse.Lt, parity_);
  }

  void solve_level(int level, const void *full_rhs, void *full_out) {
    StrictPersistentLevel<T> &state = levels_[level];
    const StrictLevelGeometry &g = state.geometry;
    const size_t compact_bytes = g.compact_n * sizeof(LatticeComplex<T>);
    prepare(level, state.pc_rhs, full_rhs, arena_.tmp);

    if (level == num_levels_ - 1) {
      const bool converged = coarsest_bicgstab(level);
      if (!converged && params_[_VERBOSE_] && params_[_NODE_RANK_] == 0)
        std::printf(
            "PYQCU::SOLVER::STRICT_MG::COARSE:\n "
            "BiCGStab reached breakdown/max_iter; returning finite iterate\n");
    } else {
      strict_check_cuda(cudaMemsetAsync(
                            state.x, 0, compact_bytes, set_->stream),
                        "strict smoother x zero");
      strict_check_cuda(cudaMemcpyAsync(
                            arena_.r, state.pc_rhs, compact_bytes,
                            cudaMemcpyDeviceToDevice, set_->stream),
                        "strict smoother rhs copy");
      mr_smooth(level, smoother_steps_);

      StrictPersistentLevel<T> &child = levels_[level + 1];
      restrict_to_child(level, arena_.r, child.full_rhs);
      solve_level(level + 1, child.full_rhs, arena_.correction_full);
      prolong_from_child(level, arena_.correction_full, arena_.tmp);
      strict_add_kernel<T>
          <<<blocks(g.compact_n), _BLOCK_SIZE_, 0, set_->stream>>>(
              state.x, arena_.tmp, static_cast<int>(g.compact_n));

      // Child recursion intentionally reuses r/v/tmp.  Recompute the parent
      // residual instead of preserving three parent-sized vectors per level.
      apply_matpc(level, arena_.v, state.x, arena_.tmp);
      strict_subtract_kernel<T>
          <<<blocks(g.compact_n), _BLOCK_SIZE_, 0, set_->stream>>>(
              arena_.r, state.pc_rhs, arena_.v,
              static_cast<int>(g.compact_n));
      mr_smooth(level, smoother_steps_);
    }
    reconstruct(level, full_out, full_rhs, state.x, arena_.tmp);
  }
};

template <typename T>
size_t strict_launch_vcycle(
    void *full_out, const void *full_rhs, void *set_ptrs, int *params,
    int start_level) {
  if (full_out == nullptr || full_rhs == nullptr)
    throw std::invalid_argument("strict V-cycle fields must be non-null");
  long long *table = static_cast<long long *>(set_ptrs);
  if (table[_SET_PTRS_STRICT_HIERARCHY_] != 0) {
    StrictCoarseHierarchy<T> *hierarchy =
        reinterpret_cast<StrictCoarseHierarchy<T> *>(
            table[_SET_PTRS_STRICT_HIERARCHY_]);
    if (hierarchy->start_level() != start_level)
      throw std::invalid_argument(
          "persistent strict hierarchy start_level mismatch");
    hierarchy->run(full_out, full_rhs);
    return hierarchy->allocated_bytes();
  }
  LatticeSet<T> *set = strict_get_set<T>(set_ptrs, params);
  StrictCoarseHierarchy<T> hierarchy(set, set_ptrs, params, start_level);
  const size_t allocated_bytes = hierarchy.allocated_bytes();
  hierarchy.run(full_out, full_rhs);
  return allocated_bytes;
}

template <typename T>
size_t strict_init_hierarchy(void *set_ptrs, int *params, int start_level) {
  long long *table = static_cast<long long *>(set_ptrs);
  if (table[_SET_PTRS_STRICT_HIERARCHY_] != 0)
    throw std::invalid_argument("persistent strict hierarchy already exists");
  LatticeSet<T> *set = strict_get_set<T>(set_ptrs, params);
  StrictCoarseHierarchy<T> *hierarchy =
      new StrictCoarseHierarchy<T>(set, set_ptrs, params, start_level);
  table[_SET_PTRS_STRICT_HIERARCHY_] =
      reinterpret_cast<long long>(hierarchy);
  return hierarchy->allocated_bytes();
}

template <typename T>
void strict_end_hierarchy(void *set_ptrs) {
  long long *table = static_cast<long long *>(set_ptrs);
  if (table[_SET_PTRS_STRICT_HIERARCHY_] == 0) return;
  StrictCoarseHierarchy<T> *hierarchy =
      reinterpret_cast<StrictCoarseHierarchy<T> *>(
          table[_SET_PTRS_STRICT_HIERARCHY_]);
  table[_SET_PTRS_STRICT_HIERARCHY_] = 0;
  delete hierarchy;
}

}  // namespace
}  // namespace qcu

extern "C" int applyMultigridStrictVCycleQcu(
    long long full_out, long long full_rhs, long long set_ptrs,
    long long params, int start_level,
    unsigned long long *allocated_bytes) {
  try {
    if (allocated_bytes == nullptr)
      throw std::invalid_argument("strict V-cycle byte output is null");
    *allocated_bytes = 0;
    qcu::strict_check_cuda(
        cudaDeviceSynchronize(), "strict V-cycle input sync");
    int *host_params = reinterpret_cast<int *>(params);
    size_t bytes = 0;
    if (host_params[_DATA_TYPE_] == _LAT_C64_)
      bytes = qcu::strict_launch_vcycle<float>(
          reinterpret_cast<void *>(full_out),
          reinterpret_cast<void *>(full_rhs),
          reinterpret_cast<void *>(set_ptrs), host_params, start_level);
    else if (host_params[_DATA_TYPE_] == _LAT_C128_)
      bytes = qcu::strict_launch_vcycle<double>(
          reinterpret_cast<void *>(full_out),
          reinterpret_cast<void *>(full_rhs),
          reinterpret_cast<void *>(set_ptrs), host_params, start_level);
    else
      throw std::invalid_argument(
          "strict V-cycle supports complex64/complex128");
    *allocated_bytes = static_cast<unsigned long long>(bytes);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "PYQCU::SOLVER::STRICT_MG::VCYCLE: %s\n",
                 error.what());
    return 1;
  }
}

extern "C" int applyMultigridStrictInitQcu(
    long long set_ptrs, long long params, int start_level,
    unsigned long long *allocated_bytes) {
  try {
    if (allocated_bytes == nullptr)
      throw std::invalid_argument("strict hierarchy byte output is null");
    *allocated_bytes = 0;
    qcu::strict_check_cuda(
        cudaDeviceSynchronize(), "strict hierarchy init input sync");
    int *host_params = reinterpret_cast<int *>(params);
    size_t bytes = 0;
    if (host_params[_DATA_TYPE_] == _LAT_C64_)
      bytes = qcu::strict_init_hierarchy<float>(
          reinterpret_cast<void *>(set_ptrs), host_params, start_level);
    else if (host_params[_DATA_TYPE_] == _LAT_C128_)
      bytes = qcu::strict_init_hierarchy<double>(
          reinterpret_cast<void *>(set_ptrs), host_params, start_level);
    else
      throw std::invalid_argument(
          "strict hierarchy supports complex64/complex128");
    *allocated_bytes = static_cast<unsigned long long>(bytes);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "PYQCU::SOLVER::STRICT_MG::INIT: %s\n",
                 error.what());
    return 1;
  }
}

extern "C" int applyMultigridStrictEndQcu(
    long long set_ptrs, long long params) {
  try {
    int *host_params = reinterpret_cast<int *>(params);
    if (host_params[_DATA_TYPE_] == _LAT_C64_)
      qcu::strict_end_hierarchy<float>(reinterpret_cast<void *>(set_ptrs));
    else if (host_params[_DATA_TYPE_] == _LAT_C128_)
      qcu::strict_end_hierarchy<double>(reinterpret_cast<void *>(set_ptrs));
    else
      throw std::invalid_argument(
          "strict hierarchy supports complex64/complex128");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "PYQCU::SOLVER::STRICT_MG::END: %s\n",
                 error.what());
    return 1;
  }
}

extern "C" int applyMultigridStrictCoarseQcu(
    long long out, long long in, long long links, long long onsite_pair,
    long long set_ptrs, long long params, int E, int X, int Y, int Z, int T,
    int onsite_index) {
  try {
    qcu::strict_check_cuda(cudaDeviceSynchronize(), "coarse input sync");
    int *host_params = reinterpret_cast<int *>(params);
    if (host_params[_DATA_TYPE_] == _LAT_C64_)
      qcu::strict_launch_coarse<float>(
          reinterpret_cast<void *>(out), reinterpret_cast<void *>(in),
          reinterpret_cast<void *>(links), reinterpret_cast<void *>(onsite_pair),
          reinterpret_cast<void *>(set_ptrs), host_params,
          E, X, Y, Z, T, onsite_index);
    else if (host_params[_DATA_TYPE_] == _LAT_C128_)
      qcu::strict_launch_coarse<double>(
          reinterpret_cast<void *>(out), reinterpret_cast<void *>(in),
          reinterpret_cast<void *>(links), reinterpret_cast<void *>(onsite_pair),
          reinterpret_cast<void *>(set_ptrs), host_params,
          E, X, Y, Z, T, onsite_index);
    else
      throw std::invalid_argument("strict coarse supports complex64/complex128");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "PYQCU::SOLVER::STRICT_MG::COARSE: %s\n", error.what());
    return 1;
  }
}

extern "C" int applyMultigridStrictMatPCQcu(
    long long out, long long in, long long links, long long scratch,
    long long set_ptrs, long long params, int E, int X, int Y, int Z, int T,
    int parity) {
  try {
    qcu::strict_check_cuda(cudaDeviceSynchronize(), "MATPC input sync");
    int *host_params = reinterpret_cast<int *>(params);
    if (host_params[_DATA_TYPE_] == _LAT_C64_)
      qcu::strict_launch_matpc<float>(
          reinterpret_cast<void *>(out), reinterpret_cast<void *>(in),
          reinterpret_cast<void *>(links), reinterpret_cast<void *>(scratch),
          reinterpret_cast<void *>(set_ptrs), host_params,
          E, X, Y, Z, T, parity);
    else if (host_params[_DATA_TYPE_] == _LAT_C128_)
      qcu::strict_launch_matpc<double>(
          reinterpret_cast<void *>(out), reinterpret_cast<void *>(in),
          reinterpret_cast<void *>(links), reinterpret_cast<void *>(scratch),
          reinterpret_cast<void *>(set_ptrs), host_params,
          E, X, Y, Z, T, parity);
    else
      throw std::invalid_argument("strict MATPC supports complex64/complex128");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "PYQCU::SOLVER::STRICT_MG::MATPC: %s\n", error.what());
    return 1;
  }
}

extern "C" int applyMultigridStrictPrepareQcu(
    long long out, long long full_rhs, long long links, long long onsite_pair,
    long long scratch, long long set_ptrs, long long params,
    int E, int X, int Y, int Z, int T, int parity) {
  try {
    qcu::strict_check_cuda(cudaDeviceSynchronize(), "prepare input sync");
    int *host_params = reinterpret_cast<int *>(params);
    if (host_params[_DATA_TYPE_] == _LAT_C64_)
      qcu::strict_launch_prepare<float>(
          reinterpret_cast<void *>(out), reinterpret_cast<void *>(full_rhs),
          reinterpret_cast<void *>(links), reinterpret_cast<void *>(onsite_pair),
          reinterpret_cast<void *>(scratch), reinterpret_cast<void *>(set_ptrs),
          host_params, E, X, Y, Z, T, parity);
    else if (host_params[_DATA_TYPE_] == _LAT_C128_)
      qcu::strict_launch_prepare<double>(
          reinterpret_cast<void *>(out), reinterpret_cast<void *>(full_rhs),
          reinterpret_cast<void *>(links), reinterpret_cast<void *>(onsite_pair),
          reinterpret_cast<void *>(scratch), reinterpret_cast<void *>(set_ptrs),
          host_params, E, X, Y, Z, T, parity);
    else
      throw std::invalid_argument("strict prepare supports complex64/complex128");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "PYQCU::SOLVER::STRICT_MG::PREPARE: %s\n", error.what());
    return 1;
  }
}

extern "C" int applyMultigridStrictReconstructQcu(
    long long full_out, long long full_rhs, long long target_solution,
    long long links, long long onsite_pair, long long scratch,
    long long set_ptrs, long long params,
    int E, int X, int Y, int Z, int T, int parity) {
  try {
    qcu::strict_check_cuda(cudaDeviceSynchronize(), "reconstruct input sync");
    int *host_params = reinterpret_cast<int *>(params);
    if (host_params[_DATA_TYPE_] == _LAT_C64_)
      qcu::strict_launch_reconstruct<float>(
          reinterpret_cast<void *>(full_out), reinterpret_cast<void *>(full_rhs),
          reinterpret_cast<void *>(target_solution), reinterpret_cast<void *>(links),
          reinterpret_cast<void *>(onsite_pair), reinterpret_cast<void *>(scratch),
          reinterpret_cast<void *>(set_ptrs), host_params,
          E, X, Y, Z, T, parity);
    else if (host_params[_DATA_TYPE_] == _LAT_C128_)
      qcu::strict_launch_reconstruct<double>(
          reinterpret_cast<void *>(full_out), reinterpret_cast<void *>(full_rhs),
          reinterpret_cast<void *>(target_solution), reinterpret_cast<void *>(links),
          reinterpret_cast<void *>(onsite_pair), reinterpret_cast<void *>(scratch),
          reinterpret_cast<void *>(set_ptrs), host_params,
          E, X, Y, Z, T, parity);
    else
      throw std::invalid_argument(
          "strict reconstruct supports complex64/complex128");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "PYQCU::SOLVER::STRICT_MG::RECONSTRUCT: %s\n",
                 error.what());
    return 1;
  }
}

extern "C" int applyMultigridStrictRestrictQcu(
    long long coarse_out, long long fine_in, long long null_vectors,
    long long set_ptrs, long long params, int E, int e,
    int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc,
    int parity) {
  try {
    qcu::strict_check_cuda(cudaDeviceSynchronize(), "restrict input sync");
    int *host_params = reinterpret_cast<int *>(params);
    if (host_params[_DATA_TYPE_] == _LAT_C64_)
      qcu::strict_launch_restrict<float>(
          reinterpret_cast<void *>(coarse_out), reinterpret_cast<void *>(fine_in),
          reinterpret_cast<void *>(null_vectors), reinterpret_cast<void *>(set_ptrs),
          host_params, E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc, parity);
    else if (host_params[_DATA_TYPE_] == _LAT_C128_)
      qcu::strict_launch_restrict<double>(
          reinterpret_cast<void *>(coarse_out), reinterpret_cast<void *>(fine_in),
          reinterpret_cast<void *>(null_vectors), reinterpret_cast<void *>(set_ptrs),
          host_params, E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc, parity);
    else
      throw std::invalid_argument("strict restrict supports complex64/complex128");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "PYQCU::SOLVER::STRICT_MG::RESTRICT: %s\n", error.what());
    return 1;
  }
}

extern "C" int applyMultigridStrictProLongQcu(
    long long fine_out, long long coarse_in, long long null_vectors,
    long long set_ptrs, long long params, int E, int e,
    int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc,
    int parity) {
  try {
    qcu::strict_check_cuda(cudaDeviceSynchronize(), "prolong input sync");
    int *host_params = reinterpret_cast<int *>(params);
    if (host_params[_DATA_TYPE_] == _LAT_C64_)
      qcu::strict_launch_prolong<float>(
          reinterpret_cast<void *>(fine_out), reinterpret_cast<void *>(coarse_in),
          reinterpret_cast<void *>(null_vectors), reinterpret_cast<void *>(set_ptrs),
          host_params, E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc, parity);
    else if (host_params[_DATA_TYPE_] == _LAT_C128_)
      qcu::strict_launch_prolong<double>(
          reinterpret_cast<void *>(fine_out), reinterpret_cast<void *>(coarse_in),
          reinterpret_cast<void *>(null_vectors), reinterpret_cast<void *>(set_ptrs),
          host_params, E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc, parity);
    else
      throw std::invalid_argument("strict prolong supports complex64/complex128");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "PYQCU::SOLVER::STRICT_MG::PROLONG: %s\n", error.what());
    return 1;
  }
}
