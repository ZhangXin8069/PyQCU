#include "../include/qcu.h"
#include "../python/pyqcu.h"
#include <chrono>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#pragma optimize(5)
namespace qcu {
namespace cg = cooperative_groups;

template <typename T>
__device__ inline bool multigrid_cg_bad(const LatticeComplex<T> &z) {
  return !((z.real() == z.real()) && (z.imag() == z.imag()) &&
           fabs(z.real()) != INFINITY && fabs(z.imag()) != INFINITY);
}

template <typename T>
__global__ void multigrid_vector_subtract(void *destination,
                                          const void *source,
                                          const void *subtract, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  auto *out = static_cast<LatticeComplex<T> *>(destination);
  const auto *in = static_cast<const LatticeComplex<T> *>(source);
  const auto *correction = static_cast<const LatticeComplex<T> *>(subtract);
  out[index] = in[index] - correction[index];
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
__device__ inline LatticeComplex<T> multigrid_block_reduce_sum_complex(
    T *real_scratch, T *imag_scratch, LatticeComplex<T> value) {
  const int tid = threadIdx.x;
  real_scratch[tid] = value.real();
  imag_scratch[tid] = value.imag();
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      real_scratch[tid] += real_scratch[tid + stride];
      imag_scratch[tid] += imag_scratch[tid + stride];
    }
    __syncthreads();
  }
  return LatticeComplex<T>(real_scratch[0], imag_scratch[0]);
}

template <typename T>
__device__ inline T multigrid_block_reduce_sum_real(T *scratch, T value) {
  const int tid = threadIdx.x;
  scratch[tid] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) scratch[tid] += scratch[tid + stride];
    __syncthreads();
  }
  return scratch[0];
}

template <typename T>
__global__ void multigrid_local_orthogonalize_cgs2(
    const void *flat_in, void *blocked_out, int E, int e, int Xf, int Yf,
    int Zf, int Tf, int Xc, int Yc, int Zc, int Tc, int *status) {
  extern __shared__ unsigned char dynamic_shared[];
  __shared__ T dot_real_scratch[256];
  __shared__ T dot_imag_scratch[256];
  __shared__ T norm_scratch[256];
  __shared__ int block_failure;

  LatticeComplex<T> *work =
      reinterpret_cast<LatticeComplex<T> *>(dynamic_shared);
  const LatticeComplex<T> *input =
      static_cast<const LatticeComplex<T> *>(flat_in);
  LatticeComplex<T> *output = static_cast<LatticeComplex<T> *>(blocked_out);

  const int tid = threadIdx.x;
  int coarse_site = blockIdx.x;
  const int it_c = coarse_site % Tc;
  coarse_site /= Tc;
  const int iz_c = coarse_site % Zc;
  coarse_site /= Zc;
  const int iy_c = coarse_site % Yc;
  const int ix_c = coarse_site / Yc;

  const int bx = Xf / Xc;
  const int by = Yf / Yc;
  const int bz = Zf / Zc;
  const int bt = Tf / Tc;
  const int local_dim = e * bx * by * bz * bt;
  if (tid == 0) block_failure = 0;
  __syncthreads();

  for (int column = 0; column < E; ++column) {
    for (int local = tid; local < local_dim; local += blockDim.x) {
      int rem = local;
      const int dt = rem % bt;
      rem /= bt;
      const int dz = rem % bz;
      rem /= bz;
      const int dy = rem % by;
      rem /= by;
      const int dx = rem % bx;
      const int e_idx = rem / bx;
      const int ix_f = ix_c * bx + dx;
      const int iy_f = iy_c * by + dy;
      const int iz_f = iz_c * bz + dz;
      const int it_f = it_c * bt + dt;
      size_t in_idx = static_cast<size_t>(column);
      in_idx = in_idx * e + e_idx;
      in_idx = in_idx * Xf + ix_f;
      in_idx = in_idx * Yf + iy_f;
      in_idx = in_idx * Zf + iz_f;
      in_idx = in_idx * Tf + it_f;
      work[local] = input[in_idx];
    }
    __syncthreads();

    for (int pass = 0; pass < 2; ++pass) {
      for (int previous_index = 0; previous_index < column;
           ++previous_index) {
        LatticeComplex<T> dot(0, 0);
        for (int local = tid; local < local_dim; local += blockDim.x) {
          int rem = local;
          const int dt = rem % bt;
          rem /= bt;
          const int dz = rem % bz;
          rem /= bz;
          const int dy = rem % by;
          rem /= by;
          const int dx = rem % bx;
          const int e_idx = rem / bx;
          const int ix_f = ix_c * bx + dx;
          const int iy_f = iy_c * by + dy;
          const int iz_f = iz_c * bz + dz;
          const int it_f = it_c * bt + dt;
          const size_t out_idx = multigrid_null_index(
              previous_index, e_idx, ix_f, iy_f, iz_f, it_f, e,
              Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
          dot += output[out_idx].conj() * work[local];
        }
        dot = multigrid_block_reduce_sum_complex(
            dot_real_scratch, dot_imag_scratch, dot);
        const LatticeComplex<T> coefficient = dot;
        for (int local = tid; local < local_dim; local += blockDim.x) {
          int rem = local;
          const int dt = rem % bt;
          rem /= bt;
          const int dz = rem % bz;
          rem /= bz;
          const int dy = rem % by;
          rem /= by;
          const int dx = rem % bx;
          const int e_idx = rem / bx;
          const int ix_f = ix_c * bx + dx;
          const int iy_f = iy_c * by + dy;
          const int iz_f = iz_c * bz + dz;
          const int it_f = it_c * bt + dt;
          const size_t out_idx = multigrid_null_index(
              previous_index, e_idx, ix_f, iy_f, iz_f, it_f, e,
              Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
          work[local] -= coefficient * output[out_idx];
        }
        __syncthreads();
      }
    }

    T norm2 = 0;
    for (int local = tid; local < local_dim; local += blockDim.x) {
      const T real = work[local].real();
      const T imag = work[local].imag();
      norm2 += real * real + imag * imag;
    }
    norm2 = multigrid_block_reduce_sum_real(norm_scratch, norm2);
    if (tid == 0) {
      const T threshold = std::numeric_limits<T>::epsilon() *
                          static_cast<T>(local_dim > E ? local_dim : E);
      if (!isfinite(norm2) || norm2 <= threshold * threshold) {
        block_failure = 1;
        atomicExch(status, 1);
        norm_scratch[0] = 0;
      } else {
        norm_scratch[0] = sqrt(norm2);
      }
    }
    __syncthreads();
    const T column_norm = norm_scratch[0];
    for (int local = tid; local < local_dim; local += blockDim.x) {
      int rem = local;
      const int dt = rem % bt;
      rem /= bt;
      const int dz = rem % bz;
      rem /= bz;
      const int dy = rem % by;
      rem /= by;
      const int dx = rem % bx;
      const int e_idx = rem / bx;
      const int ix_f = ix_c * bx + dx;
      const int iy_f = iy_c * by + dy;
      const int iz_f = iz_c * bz + dz;
      const int it_f = it_c * bt + dt;
      const size_t out_idx = multigrid_null_index(
          column, e_idx, ix_f, iy_f, iz_f, it_f, e,
          Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
      output[out_idx] = block_failure
                            ? LatticeComplex<T>(0, 0)
                            : work[local] / column_norm;
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void multigrid_local_gather_gauge(
    const void *global_gauge, void *local_gauge, const int *centers,
    int batch, int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc,
    int Tc, int W0, int W1, int W2, int W3, int bx, int by, int bz,
    int bt) {
  const int64_t local_sites =
      static_cast<int64_t>(W0) * W1 * W2 * W3;
  const int64_t stride = 2 * 3 * 3 * 4 * local_sites;
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<int64_t>(batch) * stride) return;
  int64_t value = index;
  const int site = value % local_sites;
  value /= local_sites;
  const int direction = value % 4;
  value /= 4;
  const int column = value % 3;
  value /= 3;
  const int row = value % 3;
  value /= 3;
  const int parity = value % 2;
  const int example = value / 2;
  const int t2 = site % W3;
  int rest = site / W3;
  const int z = rest % W2;
  rest /= W2;
  const int y = rest % W1;
  const int x = rest / W1;
  const int start[4] = {
      centers[example * 4] * bx - bx - 2,
      centers[example * 4 + 1] * by - by - 2,
      centers[example * 4 + 2] * bz - bz - 2,
      centers[example * 4 + 3] * bt - bt - 1};
  const int gx = (start[0] + x + Xf) % Xf;
  const int gy = (start[1] + y + Yf) % Yf;
  const int gz = (start[2] + z + Zf) % Zf;
  const int gt = (start[3] + t2 + Tf) % Tf;
  int64_t source = parity;
  source = source * 3 + row;
  source = source * 3 + column;
  source = source * 4 + direction;
  source = source * Xf + gx;
  source = source * Yf + gy;
  source = source * Zf + gz;
  source = source * Tf + gt;
  static_cast<LatticeComplex<T> *>(local_gauge)[index] =
      static_cast<const LatticeComplex<T> *>(global_gauge)[source];
}

template <typename T>
__global__ void multigrid_local_fill_source(
    void *local_source, const void *null_vectors, const int *centers,
    int E, int e, int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc,
    int Tc, int W0, int W1, int W2, int W3, int bx, int by, int bz,
    int bt, int parity, int probe, int batch) {
  const int64_t local_sites =
      static_cast<int64_t>(W0) * W1 * W2 * W3;
  const int64_t block_sites =
      static_cast<int64_t>(bx) * by * bz * bt;
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<int64_t>(batch) * block_sites * e) return;
  int64_t value = index;
  const int component = value % e;
  value /= e;
  const int local = value % block_sites;
  const int example = value / block_sites;
  const int dt = local % bt;
  int rem = local / bt;
  const int dz = rem % bz;
  rem /= bz;
  const int dy = rem % by;
  const int dx = rem / by;
  const int x = centers[example * 4] * bx + dx;
  const int y = centers[example * 4 + 1] * by + dy;
  const int z = centers[example * 4 + 2] * bz + dz;
  const int t = centers[example * 4 + 3] * bt + dt;
  const int lx = (bx + 2 + dx) % W0;
  const int ly = (by + 2 + dy) % W1;
  const int lz = (bz + 2 + dz) % W2;
  const int lt = (bt + 1 + dt) % W3;
  const int64_t local_site = ((static_cast<int64_t>(lx) * W1 + ly) * W2 + lz) * W3 + lt;
  const std::size_t null_index = multigrid_null_index(
      probe, component, x, y, z, t, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
  const int64_t destination =
      (static_cast<int64_t>(example) * 2 + parity) * e * local_sites +
      static_cast<int64_t>(component) * local_sites + local_site;
  static_cast<LatticeComplex<T> *>(local_source)[destination] =
      static_cast<const LatticeComplex<T> *>(null_vectors)[null_index];
}

template <typename T>
__global__ void multigrid_local_onsite_inverse(
    const void *input, void *output, const void *inverse, const int *centers,
    int parity, int Xf, int Yf, int Zf, int Tf, int W0, int W1, int W2,
    int W3, int bx, int by, int bz, int bt, int batch) {
  const int64_t local_sites =
      static_cast<int64_t>(W0) * W1 * W2 * W3;
  const int64_t global_sites =
      static_cast<int64_t>(Xf) * Yf * Zf * Tf;
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<int64_t>(batch) * local_sites * 12) return;
  int64_t value = index;
  const int row = value % 12;
  value /= 12;
  const int local_site = value % local_sites;
  const int example = value / local_sites;
  int rest = local_site;
  const int lt = rest % W3;
  rest /= W3;
  const int lz = rest % W2;
  rest /= W2;
  const int ly = rest % W1;
  const int lx = rest / W1;
  const int start_x = centers[example * 4] * bx - bx - 2;
  const int start_y = centers[example * 4 + 1] * by - by - 2;
  const int start_z = centers[example * 4 + 2] * bz - bz - 2;
  const int start_t = centers[example * 4 + 3] * bt - bt - 1;
  const int gx = (start_x + lx + Xf) % Xf;
  const int gy = (start_y + ly + Yf) % Yf;
  const int gz = (start_z + lz + Zf) % Zf;
  const int gt = (start_t + lt + Tf) % Tf;
  const int64_t global_site = ((static_cast<int64_t>(gx) * Yf + gy) * Zf + gz) * Tf + gt;
  LatticeComplex<T> sum(0, 0);
  for (int column = 0; column < 12; ++column) {
    const int64_t matrix_index =
        (static_cast<int64_t>(row) * 12 + column) * global_sites + global_site;
    const int64_t input_index =
        (static_cast<int64_t>(example) * 2 + parity) * 12 * local_sites +
        static_cast<int64_t>(column) * local_sites + local_site;
    sum += static_cast<const LatticeComplex<T> *>(inverse)[matrix_index] *
           static_cast<const LatticeComplex<T> *>(input)[input_index];
  }
  const int64_t output_index =
      (static_cast<int64_t>(example) * 2 + parity) * 12 * local_sites +
      static_cast<int64_t>(row) * local_sites + local_site;
  static_cast<LatticeComplex<T> *>(output)[output_index] = sum;
}

template <typename T>
__global__ void multigrid_local_onsite_subtract(
    const void *source, const void *odd_term, void *schur, const int *centers,
    int parity, T kappa2, int Xf, int Yf, int Zf, int Tf, int W0, int W1,
    int W2, int W3, int bx, int by, int bz, int bt, int batch) {
  const int64_t local_sites =
      static_cast<int64_t>(W0) * W1 * W2 * W3;
  const int64_t global_sites =
      static_cast<int64_t>(Xf) * Yf * Zf * Tf;
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<int64_t>(batch) * local_sites * 12) return;
  int64_t value = index;
  const int row = value % 12;
  value /= 12;
  const int local_site = value % local_sites;
  const int example = value / local_sites;
  int rest = local_site;
  const int lt = rest % W3;
  rest /= W3;
  const int lz = rest % W2;
  rest /= W2;
  const int ly = rest % W1;
  const int lx = rest / W1;
  const int start_x = centers[example * 4] * bx - bx - 2;
  const int start_y = centers[example * 4 + 1] * by - by - 2;
  const int start_z = centers[example * 4 + 2] * bz - bz - 2;
  const int start_t = centers[example * 4 + 3] * bt - bt - 1;
  const int gx = (start_x + lx + Xf) % Xf;
  const int gy = (start_y + ly + Yf) % Yf;
  const int gz = (start_z + lz + Zf) % Zf;
  const int gt = (start_t + lt + Tf) % Tf;
  const int64_t global_site = ((static_cast<int64_t>(gx) * Yf + gy) * Zf + gz) * Tf + gt;
  LatticeComplex<T> onsite(0, 0);
  for (int column = 0; column < 12; ++column) {
    const int64_t matrix_index =
        (static_cast<int64_t>(row) * 12 + column) * global_sites + global_site;
    const int64_t source_index =
        (static_cast<int64_t>(example) * 2 + parity) * 12 * local_sites +
        static_cast<int64_t>(column) * local_sites + local_site;
    onsite += static_cast<const LatticeComplex<T> *>(odd_term)[matrix_index] *
              static_cast<const LatticeComplex<T> *>(source)[source_index];
  }
  auto *output = static_cast<LatticeComplex<T> *>(schur);
  const int64_t output_index =
      (static_cast<int64_t>(example) * 2 + parity) * 12 * local_sites +
      static_cast<int64_t>(row) * local_sites + local_site;
  output[output_index] =
      onsite - LatticeComplex<T>(kappa2, 0) * output[output_index];
}

__device__ __constant__ int multigrid_local_stencil_shifts[33][4] = {
    {0, 0, 0, 0},
    {1, 0, 0, 0}, {-1, 0, 0, 0}, {0, 1, 0, 0}, {0, -1, 0, 0},
    {0, 0, 1, 0}, {0, 0, -1, 0}, {0, 0, 0, 1}, {0, 0, 0, -1},
    {1, 1, 0, 0}, {-1, -1, 0, 0}, {1, -1, 0, 0}, {-1, 1, 0, 0},
    {1, 0, 1, 0}, {-1, 0, -1, 0}, {1, 0, -1, 0}, {-1, 0, 1, 0},
    {1, 0, 0, 1}, {-1, 0, 0, -1}, {1, 0, 0, -1}, {-1, 0, 0, 1},
    {0, 1, 1, 0}, {0, -1, -1, 0}, {0, 1, -1, 0}, {0, -1, 1, 0},
    {0, 1, 0, 1}, {0, -1, 0, -1}, {0, 1, 0, -1}, {0, -1, 0, 1},
    {0, 0, 1, 1}, {0, 0, -1, -1}, {0, 0, 1, -1}, {0, 0, -1, 1}};

template <typename T>
__global__ void multigrid_local_restrict_stencil(
    void *sitting, void *hop_nn, void *hop_diag, const void *schur,
    const void *null_vectors, const int *centers, int E, int e, int Xf,
    int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc, int W0, int W1,
    int W2, int W3, int bx, int by, int bz, int bt, int parity, int probe) {
  const int example = blockIdx.x;
  const int coarse_dims[4] = {Xc, Yc, Zc, Tc};
  const int64_t coarse_volume =
      static_cast<int64_t>(Xc) * Yc * Zc * Tc;
  const int64_t local_sites =
      static_cast<int64_t>(W0) * W1 * W2 * W3;
  const int64_t block_sites =
      static_cast<int64_t>(bx) * by * bz * bt;
  for (int task = threadIdx.x; task < 33 * E; task += blockDim.x) {
    const int slot = task / E;
    const int out = task % E;
    int target[4];
    for (int dim = 0; dim < 4; ++dim) {
      target[dim] = (centers[example * 4 + dim] +
                     multigrid_local_stencil_shifts[slot][dim] +
                     coarse_dims[dim]) % coarse_dims[dim];
    }
    int multiplicity = 0;
    for (int other = 0; other < 33; ++other) {
      int other_target[4];
      bool same = true;
      for (int dim = 0; dim < 4; ++dim) {
        other_target[dim] =
            (centers[example * 4 + dim] +
             multigrid_local_stencil_shifts[other][dim] + coarse_dims[dim]) %
            coarse_dims[dim];
        same = same && other_target[dim] == target[dim];
      }
      multiplicity += same ? 1 : 0;
    }
    LatticeComplex<T> sum(0, 0);
    for (int dt = 0; dt < bt; ++dt) {
      for (int dz = 0; dz < bz; ++dz) {
        for (int dy = 0; dy < by; ++dy) {
          for (int dx = 0; dx < bx; ++dx) {
            const int gx = target[0] * bx + dx;
            const int gy = target[1] * by + dy;
            const int gz = target[2] * bz + dz;
            const int gt = target[3] * bt + dt;
            const int lx = (target[0] * bx + dx -
                            (centers[example * 4] * bx - bx - 2) + Xf) % Xf;
            const int ly = (target[1] * by + dy -
                            (centers[example * 4 + 1] * by - by - 2) + Yf) % Yf;
            const int lz = (target[2] * bz + dz -
                            (centers[example * 4 + 2] * bz - bz - 2) + Zf) % Zf;
            const int lt = (target[3] * bt + dt -
                            (centers[example * 4 + 3] * bt - bt - 1) + Tf) % Tf;
            const int64_t local_site = ((static_cast<int64_t>(lx) * W1 + ly) * W2 + lz) * W3 + lt;
            for (int component = 0; component < e; ++component) {
              const std::size_t null_index = multigrid_null_index(
                  out, component, gx, gy, gz, gt, e, Xf, Yf, Zf, Tf, Xc, Yc,
                  Zc, Tc);
              const int64_t field_index =
                  (static_cast<int64_t>(example) * 2 + parity) * e *
                      local_sites +
                  static_cast<int64_t>(component) * local_sites + local_site;
              sum += static_cast<const LatticeComplex<T> *>(null_vectors)
                         [null_index].conj() *
                     static_cast<const LatticeComplex<T> *>(schur)[field_index];
            }
          }
        }
      }
    }
    sum = sum / static_cast<T>(multiplicity);
    const int64_t coarse_site =
        ((static_cast<int64_t>(target[0]) * Yc + target[1]) * Zc + target[2]) *
            Tc +
        target[3];
    if (slot == 0) {
      static_cast<LatticeComplex<T> *>(sitting)
          [(static_cast<int64_t>(out) * E + probe) * coarse_volume +
           coarse_site] = sum;
    } else if (slot <= 8) {
      // The coefficient at target=n is consumed by the coarse kernel as
      // input[n + encoded_shift].  Store n=center+shift under the opposite
      // encoded sign so that it reads input[center].
      const int raw = slot - 1;
      const int dim = raw / 2;
      const int shift_is_negative = raw % 2;
      const int direction = (1 - shift_is_negative) * 4 + dim;
      static_cast<LatticeComplex<T> *>(hop_nn)
          [((static_cast<int64_t>(direction) * E + out) * E + probe) *
               coarse_volume +
           coarse_site] = sum;
    } else {
      // hop_diag is [sign1,sign2,pair].  The shift table is pair-major, so
      // decode the target shift, invert both signs, then flatten in the
      // kernel's sign-major layout.
      const int raw = slot - 9;
      const int pair = raw / 4;
      const int shift_encoding = raw % 4;
      // The table order is (++), (--), (+-), (-+).  Flipping both
      // signs therefore swaps 0<->1 and 2<->3; a numeric 3-x map
      // would leave (++/--) unchanged and would only swap mixed signs.
      const int opposite_encoding = shift_encoding ^ 1;
      // Table order is (++), (--), (+-), (-+), while the kernel's
      // combination order is (++), (+-), (-+), (--).  The two orders are
      // not the same two-bit encoding.
      const int kernel_combination =
          opposite_encoding == 0 ? 0 :
          opposite_encoding == 1 ? 3 :
          opposite_encoding == 2 ? 1 : 2;
      const int direction = kernel_combination * 6 + pair;
      static_cast<LatticeComplex<T> *>(hop_diag)
          [((static_cast<int64_t>(direction) * E + out) * E + probe) *
               coarse_volume +
           coarse_site] = sum;
    }
  }
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
// Device-side coarse halo packing/unpacking.
// ---------------------------------------------------------------------------
// The 16 canonical shifts are
//   x, y, z, t, x+y, x-y, x+z, x-z, x+t, x-t, y+z, y-z, y+t, y-t, z+t, z-t
// followed by their opposites.  Keeping this table in a small device helper
// avoids a per-level device allocation for metadata and gives both transport
// paths exactly the same wire order.
__device__ inline int multigrid_coarse_shift_component(int direction,
                                                       int component) {
  int pair = direction >> 1;
  int value = 0;
  if (pair == 0 && component == 0) value = 1;
  if (pair == 1 && component == 1) value = 1;
  if (pair == 2 && component == 2) value = 1;
  if (pair == 3 && component == 3) value = 1;
  if (pair == 4 && (component == 0 || component == 1)) value = 1;
  if (pair == 5 && (component == 0 || component == 1))
    value = component == 0 ? 1 : -1;
  if (pair == 6 && (component == 0 || component == 2)) value = 1;
  if (pair == 7 && (component == 0 || component == 2))
    value = component == 0 ? 1 : -1;
  if (pair == 8 && (component == 0 || component == 3)) value = 1;
  if (pair == 9 && (component == 0 || component == 3))
    value = component == 0 ? 1 : -1;
  if (pair == 10 && (component == 1 || component == 2)) value = 1;
  if (pair == 11 && (component == 1 || component == 2))
    value = component == 1 ? 1 : -1;
  if (pair == 12 && (component == 1 || component == 3)) value = 1;
  if (pair == 13 && (component == 1 || component == 3))
    value = component == 1 ? 1 : -1;
  if (pair == 14 && (component == 2 || component == 3)) value = 1;
  if (pair == 15 && (component == 2 || component == 3))
    value = component == 2 ? 1 : -1;
  return (direction & 1) == 0 ? value : -value;
}

__device__ inline int multigrid_coarse_face_sites(int direction, int X, int Y,
                                                  int Z, int Lt, int grid_x,
                                                  int grid_y, int grid_z,
                                                  int grid_t) {
  int dims[4] = {X, Y, Z, Lt};
  int grid[4] = {grid_x, grid_y, grid_z, grid_t};
  bool distributed = false;
  for (int d = 0; d < 4; ++d) {
    if (multigrid_coarse_shift_component(direction, d) != 0 &&
        grid[d] > 1)
      distributed = true;
  }
  if (!distributed) return 0;
  int result = 1;
  for (int d = 0; d < 4; ++d)
    if (multigrid_coarse_shift_component(direction, d) == 0 ||
        grid[d] == 1)
      result *= dims[d];
  return result;
}

template <typename T>
__global__ void multigrid_coarse_pack_halo(
    void *packed, const void *fermion_in, int E, int X, int Y, int Z, int Lt,
    int grid_x, int grid_y, int grid_z, int grid_t, int max_face) {
  const int vol = X * Y * Z * Lt;
  const long long direction_stride = (long long)E * max_face;
  const long long total = 32LL * direction_stride;
  const long long global_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (global_idx >= total || E <= 0 || vol <= 0 || max_face <= 0) return;

  const int direction = (int)(global_idx / direction_stride);
  const long long local = global_idx - (long long)direction * direction_stride;
  const int face_sites = multigrid_coarse_face_sites(
      direction, X, Y, Z, Lt, grid_x, grid_y, grid_z, grid_t);
  const int e = (int)(local / max_face);
  const int k = (int)(local - (long long)e * max_face);
  if (e >= E || k >= face_sites) return;

  int dims[4] = {X, Y, Z, Lt};
  int grid[4] = {grid_x, grid_y, grid_z, grid_t};
  int free_dims[4];
  int nfree = 0;
  for (int d = 0; d < 4; ++d)
    if (multigrid_coarse_shift_component(direction, d) == 0 || grid[d] == 1)
      free_dims[nfree++] = d;

  int coordinate[4] = {0, 0, 0, 0};
  int q = k;
  for (int i = nfree - 1; i >= 0; --i) {
    const int d = free_dims[i];
    coordinate[d] = q % dims[d];
    q /= dims[d];
  }
  for (int d = 0; d < 4; ++d) {
    const int shift = multigrid_coarse_shift_component(direction, d);
    if (shift != 0 && grid[d] > 1)
      coordinate[d] = shift > 0 ? dims[d] - 1 : 0;
  }

  const int site = ((coordinate[0] * Y + coordinate[1]) * Z +
                    coordinate[2]) * Lt + coordinate[3];
  const LatticeComplex<T> *in =
      static_cast<const LatticeComplex<T> *>(fermion_in);
  LatticeComplex<T> *out = static_cast<LatticeComplex<T> *>(packed);
  out[(long long)direction * direction_stride + (long long)e * max_face + k] =
      in[(long long)e * vol + site];
}

template <typename T>
__global__ void multigrid_coarse_unpack_halo(
    void *halo, const void *packed, int E, int X, int Y, int Z, int Lt,
    int grid_x, int grid_y, int grid_z, int grid_t, int max_face) {
  const long long hvol = (long long)(X + 2) * (Y + 2) * (Z + 2) * (Lt + 2);
  const long long direction_stride = (long long)E * max_face;
  const long long total = 32LL * direction_stride;
  const long long global_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (global_idx >= total || E <= 0 || max_face <= 0) return;

  const int direction = (int)(global_idx / direction_stride);
  const long long local = global_idx - (long long)direction * direction_stride;
  const int face_sites = multigrid_coarse_face_sites(
      direction, X, Y, Z, Lt, grid_x, grid_y, grid_z, grid_t);
  const int e = (int)(local / max_face);
  const int k = (int)(local - (long long)e * max_face);
  if (e >= E || k >= face_sites) return;

  int dims[4] = {X, Y, Z, Lt};
  int grid[4] = {grid_x, grid_y, grid_z, grid_t};
  int free_dims[4];
  int nfree = 0;
  for (int d = 0; d < 4; ++d)
    if (multigrid_coarse_shift_component(direction, d) == 0 || grid[d] == 1)
      free_dims[nfree++] = d;

  int coordinate[4] = {1, 1, 1, 1};
  int q = k;
  for (int i = nfree - 1; i >= 0; --i) {
    const int d = free_dims[i];
    coordinate[d] = q % dims[d] + 1;
    q /= dims[d];
  }
  for (int d = 0; d < 4; ++d) {
    const int shift = multigrid_coarse_shift_component(direction, d);
    if (shift != 0 && grid[d] > 1)
      coordinate[d] = shift > 0 ? dims[d] + 1 : 0;
  }
  const int hsite = (((coordinate[0] * (Y + 2) + coordinate[1]) * (Z + 2) +
                      coordinate[2]) * (Lt + 2) + coordinate[3]);
  const LatticeComplex<T> *in = static_cast<const LatticeComplex<T> *>(packed);
  LatticeComplex<T> *out = static_cast<LatticeComplex<T> *>(halo);
  out[(long long)e * hvol + hsite] =
      in[(long long)direction * direction_stride + (long long)e * max_face + k];
}

template <typename T>
__global__ void multigrid_coarse_pack_axis_halo(
    void *packed, const void *fermion_in, int E, int X, int Y, int Z, int Lt,
    int axis) {
  const int dims[4] = {X, Y, Z, Lt};
  if (E <= 0 || axis < 0 || axis >= 4) return;
  int plane_sites = 1;
  for (int d = 0; d < 4; ++d)
    if (d != axis) plane_sites *= dims[d];
  const long long stride = static_cast<long long>(E) * plane_sites;
  const long long total = 2 * stride;
  const long long global_idx =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (global_idx >= total || plane_sites <= 0) return;

  const int side = static_cast<int>(global_idx / stride);
  const long long local = global_idx - side * stride;
  const int e = static_cast<int>(local / plane_sites);
  int q = static_cast<int>(local - static_cast<long long>(e) * plane_sites);
  int coordinate[4] = {0, 0, 0, 0};
  for (int d = 3; d >= 0; --d) {
    if (d == axis) continue;
    coordinate[d] = q % dims[d];
    q /= dims[d];
  }
  coordinate[axis] = side == 0 ? 0 : dims[axis] - 1;
  const int vol = X * Y * Z * Lt;
  const int site = ((coordinate[0] * Y + coordinate[1]) * Z +
                    coordinate[2]) * Lt + coordinate[3];
  static_cast<LatticeComplex<T> *>(packed)[global_idx] =
      static_cast<const LatticeComplex<T> *>(fermion_in)[e * vol + site];
}

template <typename T>
__global__ void multigrid_coarse_unpack_axis_halo(
    void *halo, const void *packed, int E, int X, int Y, int Z, int Lt,
    int axis) {
  const int dims[4] = {X, Y, Z, Lt};
  if (E <= 0 || axis < 0 || axis >= 4) return;
  int plane_sites = 1;
  for (int d = 0; d < 4; ++d)
    if (d != axis) plane_sites *= dims[d];
  const long long stride = static_cast<long long>(E) * plane_sites;
  const long long total = 2 * stride;
  const long long global_idx =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (global_idx >= total || plane_sites <= 0) return;

  const int side = static_cast<int>(global_idx / stride);
  const long long local = global_idx - side * stride;
  const int e = static_cast<int>(local / plane_sites);
  int q = static_cast<int>(local - static_cast<long long>(e) * plane_sites);
  int coordinate[4] = {1, 1, 1, 1};
  for (int d = 3; d >= 0; --d) {
    if (d == axis) continue;
    coordinate[d] = q % dims[d] + 1;
    q /= dims[d];
  }
  coordinate[axis] = side == 0 ? 0 : dims[axis] + 1;
  const long long hvol = static_cast<long long>(X + 2) * (Y + 2) *
                         (Z + 2) * (Lt + 2);
  const int hsite = ((coordinate[0] * (Y + 2) + coordinate[1]) * (Z + 2) +
                     coordinate[2]) * (Lt + 2) + coordinate[3];
  static_cast<LatticeComplex<T> *>(halo)[e * hvol + hsite] =
      static_cast<const LatticeComplex<T> *>(packed)[global_idx];
}

template <typename T>
__global__ void multigrid_fill_test_vector(void *out, int n,
                                           unsigned long seed) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || n <= 0) return;
  // Deterministic, non-degenerate complex data.  The irrational-looking
  // phase prevents a periodic zero pattern for common lattice volumes while
  // keeping verification independent of curand state and global RNG state.
  const T phase = (T)(0.17320508075688773) * (T)(i + 1) +
                  (T)(0.011131) * (T)(seed & 0xffffUL);
  LatticeComplex<T> *values = static_cast<LatticeComplex<T> *>(out);
  values[i] = LatticeComplex<T>(sin(phase) + (T)0.25 * cos((T)0.37 * phase),
                               cos(phase) - (T)0.15 * sin((T)0.61 * phase));
}

template <typename T>
__global__ void multigrid_difference(void *out, const void *a, const void *b,
                                      int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || n <= 0) return;
  const LatticeComplex<T> *av = static_cast<const LatticeComplex<T> *>(a);
  const LatticeComplex<T> *bv = static_cast<const LatticeComplex<T> *>(b);
  LatticeComplex<T> *ov = static_cast<LatticeComplex<T> *>(out);
  ov[i] = av[i] - bv[i];
}

template <typename T>
__global__ void multigrid_extract_null_vector(
    void *out, const void *null_vecs, int vector_index, int E, int e, int Xf,
    int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc) {
  const int fine_vol = Xf * Yf * Zf * Tf;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = e * fine_vol;
  if (idx >= total || vector_index < 0 || vector_index >= E || e <= 0 ||
      fine_vol <= 0)
    return;

  const int e_idx = idx / fine_vol;
  const int site = idx - e_idx * fine_vol;
  const int stride_yzt = Yf * Zf * Tf;
  const int stride_zt = Zf * Tf;
  const int x = site / stride_yzt;
  int rest = site - x * stride_yzt;
  const int y = rest / stride_zt;
  rest -= y * stride_zt;
  const int z = rest / Tf;
  const int t = rest - z * Tf;
  const size_t nv_idx = multigrid_null_index(
      vector_index, e_idx, x, y, z, t, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
  const LatticeComplex<T> *nv =
      static_cast<const LatticeComplex<T> *>(null_vecs);
  LatticeComplex<T> *dst = static_cast<LatticeComplex<T> *>(out);
  dst[idx] = nv[nv_idx];
}

#if defined(QCU_HAVE_NVSHMEM)
__device__ inline int multigrid_nvshmem_rank_shift(
    int direction, int grid_x, int grid_y, int grid_z, int grid_t,
    int coord_x, int coord_y, int coord_z, int coord_t) {
  const int dims[4] = {grid_x, grid_y, grid_z, grid_t};
  int coord[4] = {coord_x, coord_y, coord_z, coord_t};
  for (int d = 0; d < 4; ++d) {
    const int shift = multigrid_coarse_shift_component(direction, d);
    if (dims[d] > 1) {
      coord[d] += shift;
      if (coord[d] < 0) coord[d] += dims[d];
      if (coord[d] >= dims[d]) coord[d] -= dims[d];
    }
  }
  return ((coord[0] * dims[1] + coord[1]) * dims[2] + coord[2]) * dims[3] +
         coord[3];
}

template <typename T>
__global__ void multigrid_nvshmem_put_halo(
    void *device_recv, const void *device_send, int E, int X, int Y, int Z,
    int Lt, int max_face, int grid_x, int grid_y, int grid_z, int grid_t,
    int coord_x, int coord_y, int coord_z, int coord_t, int rank) {
  const int direction = blockIdx.x * blockDim.x + threadIdx.x;
  if (direction >= 32 || E <= 0 || max_face <= 0) return;
  const int target = multigrid_nvshmem_rank_shift(
      direction, grid_x, grid_y, grid_z, grid_t, coord_x, coord_y, coord_z,
      coord_t);
  if (target != rank) {
    const size_t face_sites = static_cast<size_t>(
        multigrid_coarse_face_sites(direction, X, Y, Z, Lt, grid_x, grid_y,
                                    grid_z, grid_t));
    if (face_sites > 0) {
      // device_send/device_recv use a fixed [E,max_face] slot.  Transfer the
      // complete slot so every E channel keeps the same offset on the remote
      // PE when face_sites is smaller than max_face.
      const size_t bytes = static_cast<size_t>(2) * static_cast<size_t>(E) *
                           static_cast<size_t>(max_face) * sizeof(T);
      const size_t stride_bytes = static_cast<size_t>(E) *
                                  static_cast<size_t>(max_face) *
                                  sizeof(LatticeComplex<T>);
      const int receive_direction = direction ^ 1;
      char *remote_receive = static_cast<char *>(device_recv) +
                             static_cast<size_t>(receive_direction) *
                                 stride_bytes;
      const char *local_send = static_cast<const char *>(device_send) +
                               static_cast<size_t>(direction) * stride_bytes;
      nvshmem_putmem_nbi(remote_receive, local_send, bytes, target);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) nvshmem_quiet();
}
#endif

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
    int dx, int dy, int dz, int dt, int grid_x, int grid_y, int grid_z,
    int grid_t) {
  int nx = x + dx, ny = y + dy, nz = z + dz, nt = t + dt;
  int coordinates[4] = {nx, ny, nz, nt};
  int dims[4] = {X, Y, Z, Lt};
  int grid[4] = {grid_x, grid_y, grid_z, grid_t};
  bool remote = false;
  for (int d = 0; d < 4; ++d) {
    if (coordinates[d] >= 0 && coordinates[d] < dims[d]) continue;
    if (grid[d] == 1) {
      // A non-distributed axis wraps inside this rank.  Keeping it in the
      // local coordinate range is what lets an x+y remote hop use the x
      // face while applying y periodicity locally.
      coordinates[d] %= dims[d];
      if (coordinates[d] < 0) coordinates[d] += dims[d];
    } else {
      remote = true;
    }
  }
  if (!remote) {
    int vol = X * Y * Z * Lt;
    int site = ((coordinates[0] * Y + coordinates[1]) * Z +
                coordinates[2]) * Lt + coordinates[3];
    return in[e * vol + site];
  }
  int hvol = (X + 2) * (Y + 2) * (Z + 2) * (Lt + 2);
  int hsite = ((((coordinates[0] + 1) * (Y + 2) +
                 (coordinates[1] + 1)) * (Z + 2) +
                (coordinates[2] + 1)) * (Lt + 2) +
               (coordinates[3] + 1));
  return halo[e * hvol + hsite];
}

template <typename T>
__device__ inline void multigrid_coarse_dslash_wide_halo_impl(
    void *fermion_out, void *fermion_in, void *halo, void *sitting,
    void *hop_nn, void *hop_diag, int E, int X, int Y, int Z, int Lt,
    int grid_x, int grid_y, int grid_z, int grid_t, int boundary_only,
    int global_idx) {
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

  const bool is_boundary =
      (grid_x > 1 && (x == 0 || x == X - 1)) ||
      (grid_y > 1 && (y == 0 || y == Y - 1)) ||
      (grid_z > 1 && (z == 0 || z == Z - 1)) ||
      (grid_t > 1 && (t == 0 || t == Lt - 1));
  if (boundary_only >= 0 && (is_boundary ? 1 : 0) != boundary_only) return;

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
                                 sx, sy, sz, st, grid_x, grid_y, grid_z,
                                 grid_t);
      sum += nn[minus_base + e * str_Ein] *
             multigrid_halo_read(in, h, E, X, Y, Z, Lt, e, x, y, z, t,
                                 -sx, -sy, -sz, -st, grid_x, grid_y, grid_z,
                                 grid_t);
    }
  }

  int pair_d1[6] = {0, 0, 0, 1, 1, 2};
  int pair_d2[6] = {1, 2, 3, 2, 3, 3};
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
                                     dx, dy, dz, dt, grid_x, grid_y, grid_z,
                                     grid_t);
      }
    }
  }
  out[global_idx] = sum;
}

template <typename T>
__global__ void multigrid_coarse_dslash_wide_halo(
    void *fermion_out, void *fermion_in, void *halo, void *sitting,
    void *hop_nn, void *hop_diag, int E, int X, int Y, int Z, int Lt,
    int grid_x, int grid_y, int grid_z, int grid_t) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  multigrid_coarse_dslash_wide_halo_impl<T>(
      fermion_out, fermion_in, halo, sitting, hop_nn, hop_diag, E, X, Y, Z,
      Lt, grid_x, grid_y, grid_z, grid_t, -1, global_idx);
}

template <typename T>
__global__ void multigrid_coarse_dslash_wide_halo_region(
    void *fermion_out, void *fermion_in, void *halo, void *sitting,
    void *hop_nn, void *hop_diag, int E, int X, int Y, int Z, int Lt,
    int grid_x, int grid_y, int grid_z, int grid_t, int boundary_only) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  multigrid_coarse_dslash_wide_halo_impl<T>(
      fermion_out, fermion_in, halo, sitting, hop_nn, hop_diag, E, X, Y, Z,
      Lt, grid_x, grid_y, grid_z, grid_t, boundary_only, global_idx);
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
  sred[tid].x = (sum).real(); sred[tid].y = (sum).imag(); __syncthreads();   \
  for (int k = NT/2; k > 0; k >>= 1) {                                      \
    if (tid < k) {                                                           \
      sred[tid].x += sred[tid + k].x;                                        \
      sred[tid].y += sred[tid + k].y;                                        \
    }                                                                        \
    __syncthreads();                                                         \
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
  using complex_data = typename LatticeComplex<T>::_data_type;
  __shared__ complex_data sred[NT];
  __shared__ complex_data s_rho, s_rtv, s_ts, s_tt, s_norm2;
  __shared__ complex_data s_beta, s_alpha, s_omega;
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
    if (tid == 0) s_r0 = sqrt(sred[0].x > 0 ? sred[0].x : 0);
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
      if (tid == 0) {
        s_rho = sred[0];
        rho = LatticeComplex<T>(sred[0].x, sred[0].y);
      }
      __syncthreads();
    }
    // 2. beta = (rho/rho_prev)*(alpha/omega); rho_prev = rho
    if (tid == 0) {
      LatticeComplex<T> beta = (rho / rho_prev) * (alpha / omega);
      s_beta.x = beta.real();
      s_beta.y = beta.imag();
      rho_prev = rho;
    }
    __syncthreads();
    // 3. p = r + beta*(p - omega*v)
    LatticeComplex<T> beta(s_beta.x, s_beta.y);
    for (int i = tid; i < n; i += NT) pr[i] = rr[i] + beta * (pr[i] - omega * vr[i]);
    __syncthreads();
    // 4. v = A_c·p
    dslash(vr, pr);
    __syncthreads();
    // 5. rtv = <r_tilde, v>; alpha = rho/rtv
    {
      LatticeComplex<T> sum(0,0);
      for (int i = tid; i < n; i += NT) sum += rtr[i].conj() * vr[i];
      _CS_REDUCE(sum);
      if (tid == 0) {
        rtv = LatticeComplex<T>(sred[0].x, sred[0].y);
        LatticeComplex<T> alpha_value = rho / rtv;
        s_alpha.x = alpha_value.real();
        s_alpha.y = alpha_value.imag();
        alpha = alpha_value;
      }
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
      sred[tid].x = sum.real();
      sred[tid].y = sum.imag();
      __syncthreads();
      for (int k = NT/2; k > 0; k >>= 1) {
        if (tid < k) {
          sred[tid].x += sred[tid+k].x;
          sred[tid].y += sred[tid+k].y;
        }
        __syncthreads();
      }
      if (tid == 0)
        ts = LatticeComplex<T>(sred[0].x, sred[0].y);
      __syncthreads();
      sred[tid].x = sum2.real();
      sred[tid].y = sum2.imag();
      __syncthreads();
      for (int k = NT/2; k > 0; k >>= 1) {
        if (tid < k) {
          sred[tid].x += sred[tid+k].x;
          sred[tid].y += sred[tid+k].y;
        }
        __syncthreads();
      }
      if (tid == 0) {
        tt = LatticeComplex<T>(sred[0].x, sred[0].y);
        LatticeComplex<T> omega_value = ts / tt;
        s_omega.x = omega_value.real();
        s_omega.y = omega_value.imag();
        omega = omega_value;
      }
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
      if (tid == 0) {
        s_norm2 = sred[0];
      }
      __syncthreads();
      if (s_norm2.x < target * target) break;
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
  using complex_data = typename LatticeComplex<T>::_data_type;
  __shared__ complex_data sred[NT];

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
    sred[tid].x = sum.real();
    sred[tid].y = sum.imag();
    __syncthreads();
    for (int k = NT_local/2; k > 0; k >>= 1) {
      if (tid < k) {
        sred[tid].x += sred[tid+k].x;
        sred[tid].y += sred[tid+k].y;
      }
      __syncthreads();
    }
    if (tid == 0)
      prt[bid] = LatticeComplex<T>(sred[0].x, sred[0].y);
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
      sred[tid].x = sum.real();
      sred[tid].y = sum.imag();
      __syncthreads();
      for (int k = NT_local/2; k > 0; k >>= 1) {
        if (tid < k) {
          sred[tid].x += sred[tid+k].x;
          sred[tid].y += sred[tid+k].y;
        }
        __syncthreads();
      }
      if (tid == 0)
        prt[bid] = LatticeComplex<T>(sred[0].x, sred[0].y);
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
        sred[tid].x = s1.real();
        sred[tid].y = s1.imag();
        __syncthreads();
        for (int k = NT_local/2; k > 0; k >>= 1) {
          if (tid < k) {
            sred[tid].x += sred[tid+k].x;
            sred[tid].y += sred[tid+k].y;
          }
          __syncthreads();
        }
        if (tid == 0)
          prt[bid] = LatticeComplex<T>(sred[0].x, sred[0].y);
        grid.sync();
        LatticeComplex<T> ts(0,0);
        for (int b = 0; b < nblocks; b++) ts += prt[b];
        grid.sync();
        sred[tid].x = s2.real();
        sred[tid].y = s2.imag();
        __syncthreads();
        for (int k = NT_local/2; k > 0; k >>= 1) {
          if (tid < k) {
            sred[tid].x += sred[tid+k].x;
            sred[tid].y += sred[tid+k].y;
          }
          __syncthreads();
        }
        if (tid == 0)
          prt[bid] = LatticeComplex<T>(sred[0].x, sred[0].y);
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
    void *hop_nn, void *hop_diag, int E, int X, int Y, int Z, int Lt,
    int grid_x, int grid_y, int grid_z, int grid_t);
template __global__ void multigrid_coarse_dslash_wide_halo<double>(
    void *fermion_out, void *fermion_in, void *halo, void *sitting,
    void *hop_nn, void *hop_diag, int E, int X, int Y, int Z, int Lt,
    int grid_x, int grid_y, int grid_z, int grid_t);
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
// Template instantiations for coarse communication and verification kernels.
template __global__ void multigrid_coarse_pack_halo<float>(
    void*, const void*, int, int, int, int, int, int, int, int, int, int);
template __global__ void multigrid_coarse_pack_halo<double>(
    void*, const void*, int, int, int, int, int, int, int, int, int, int);
template __global__ void multigrid_coarse_unpack_halo<float>(
    void*, const void*, int, int, int, int, int, int, int, int, int, int);
template __global__ void multigrid_coarse_unpack_halo<double>(
    void*, const void*, int, int, int, int, int, int, int, int, int, int);
template __global__ void multigrid_coarse_pack_axis_halo<float>(
    void*, const void*, int, int, int, int, int, int);
template __global__ void multigrid_coarse_pack_axis_halo<double>(
    void*, const void*, int, int, int, int, int, int);
template __global__ void multigrid_coarse_unpack_axis_halo<float>(
    void*, const void*, int, int, int, int, int, int);
template __global__ void multigrid_coarse_unpack_axis_halo<double>(
    void*, const void*, int, int, int, int, int, int);
template __global__ void multigrid_coarse_dslash_wide_halo_region<float>(
    void*, void*, void*, void*, void*, void*, int, int, int, int, int, int,
    int, int, int, int);
template __global__ void multigrid_coarse_dslash_wide_halo_region<double>(
    void*, void*, void*, void*, void*, void*, int, int, int, int, int, int,
    int, int, int, int);
template __global__ void multigrid_fill_test_vector<float>(
    void*, int, unsigned long);
template __global__ void multigrid_fill_test_vector<double>(
    void*, int, unsigned long);
template __global__ void multigrid_vector_subtract<float>(
    void*, const void*, const void*, int);
template __global__ void multigrid_vector_subtract<double>(
    void*, const void*, const void*, int);
template __global__ void multigrid_difference<float>(
    void*, const void*, const void*, int);
template __global__ void multigrid_difference<double>(
    void*, const void*, const void*, int);
template __global__ void multigrid_extract_null_vector<float>(
    void*, const void*, int, int, int, int, int, int, int, int, int, int, int);
template __global__ void multigrid_extract_null_vector<double>(
    void*, const void*, int, int, int, int, int, int, int, int, int, int, int);

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
#if defined(QCU_HAVE_NVSHMEM)
template __global__ void multigrid_nvshmem_put_halo<float>(
    void*, const void*, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int);
template __global__ void multigrid_nvshmem_put_halo<double>(
    void*, const void*, int, int, int, int, int, int, int, int, int, int, int,
    int, int, int, int, int);
#endif

template <typename T>
int launch_multigrid_local_orthogonalize_cgs2(
    void *flat_in, void *blocked_out, int E, int e, int Xf, int Yf, int Zf,
    int Tf, int Xc, int Yc, int Zc, int Tc) {
  if (flat_in == nullptr || blocked_out == nullptr || E <= 0 || e <= 0 ||
      Xf <= 0 || Yf <= 0 || Zf <= 0 || Tf <= 0 || Xc <= 0 || Yc <= 0 ||
      Zc <= 0 || Tc <= 0 || Xf % Xc || Yf % Yc || Zf % Zc || Tf % Tc) {
    return 1;
  }
  const int bx = Xf / Xc;
  const int by = Yf / Yc;
  const int bz = Zf / Zc;
  const int bt = Tf / Tc;
  const int local_dim = e * bx * by * bz * bt;
  if (E > local_dim) return 3;
  const size_t shared_bytes =
      static_cast<size_t>(local_dim) * sizeof(LatticeComplex<T>);
  int device = 0;
  cudaGetDevice(&device);
  cudaDeviceProp properties{};
  cudaGetDeviceProperties(&properties, device);
  if (shared_bytes > properties.sharedMemPerBlock) return 4;
  int threads = 32;
  while (threads < local_dim && threads < 256) threads <<= 1;

  int *device_status = nullptr;
  if (cudaMalloc(&device_status, sizeof(int)) != cudaSuccess) return 2;
  if (cudaMemset(device_status, 0, sizeof(int)) != cudaSuccess) {
    cudaFree(device_status);
    return 2;
  }
  const int coarse_volume = Xc * Yc * Zc * Tc;
  multigrid_local_orthogonalize_cgs2<T>
      <<<coarse_volume, threads, shared_bytes>>>(
          flat_in, blocked_out, E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc,
          device_status);
  if (cudaGetLastError() != cudaSuccess ||
      cudaDeviceSynchronize() != cudaSuccess) {
    cudaFree(device_status);
    return 2;
  }
  int status = 0;
  if (cudaMemcpy(&status, device_status, sizeof(int), cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    status = 2;
  }
  cudaFree(device_status);
  return status;
}

template <typename T>
int launch_multigrid_build_null_vectors(
    void *blocked_out, long long gauge_ptr, long long clover_even_ptr,
    long long clover_odd_ptr, long long clover_even_inv_ptr,
    long long clover_odd_inv_ptr, long long set_ptrs_ptr,
    long long params_ptr, int E, int e, int Xf, int Yf, int Zf, int Tf,
    int Xc, int Yc, int Zc, int Tc, int nv_iters, unsigned long long seed,
    double nv_tol) {
  if (blocked_out == nullptr || gauge_ptr == 0 || clover_even_ptr == 0 ||
      clover_odd_ptr == 0 || clover_even_inv_ptr == 0 ||
      clover_odd_inv_ptr == 0 || set_ptrs_ptr == 0 || params_ptr == 0 ||
      E <= 0 || e != _LAT_SC_ || Xf <= 0 || Yf <= 0 || Zf <= 0 || Tf <= 0 ||
      Xc <= 0 || Yc <= 0 || Zc <= 0 || Tc <= 0 || Xf % Xc || Yf % Yc ||
      Zf % Zc || Tf % Tc || nv_iters < 0 || !(nv_tol > 0.0) ||
      !std::isfinite(nv_tol)) {
    return 1;
  }
  auto *params = reinterpret_cast<int *>(params_ptr);
  auto *set_ptrs = reinterpret_cast<long long *>(set_ptrs_ptr);
  auto *set = reinterpret_cast<LatticeSet<T> *>(set_ptrs[params[_SET_INDEX_]]);
  if (set == nullptr || set->host_params[_LAT_X_] != Xf ||
      set->host_params[_LAT_Y_] != Yf || set->host_params[_LAT_Z_] != Zf ||
      set->host_params[_LAT_T_] != Tf) {
    return 1;
  }
  const int count = set->lat_4dim_SC;
  const size_t bytes = static_cast<size_t>(count) * sizeof(LatticeComplex<T>);
  const size_t full_bytes = 2 * bytes;
  const bool progress = std::getenv("PYQCU_MG_BUILD_PROGRESS") != nullptr;
  const auto progress_start = std::chrono::steady_clock::now();
  void *flat = nullptr;
  void *vector = nullptr;
  void *schur = nullptr;
  void *full_rhs = nullptr;
  void *full_solution = nullptr;
  const T saved_atol = set->host_argv[_ATOL_];
  const int saved_guess = set->host_params[_MG_USE_INIT_GUESS_];
  const auto restore_solver_controls = [&]() {
    set->host_argv[_ATOL_] = saved_atol;
    set->host_params[_MG_USE_INIT_GUESS_] = saved_guess;
  };
  const auto release_buffers = [&]() noexcept {
    if (full_solution != nullptr) cudaFreeAsync(full_solution, set->stream);
    if (full_rhs != nullptr) cudaFreeAsync(full_rhs, set->stream);
    if (schur != nullptr) cudaFreeAsync(schur, set->stream);
    if (vector != nullptr) cudaFreeAsync(vector, set->stream);
    if (flat != nullptr) cudaFreeAsync(flat, set->stream);
  };
  int status = 2;
  try {
    checkCudaErrors(cudaMallocAsync(&flat, static_cast<size_t>(E) * bytes,
                                    set->stream));
    checkCudaErrors(cudaMallocAsync(&vector, bytes, set->stream));
    checkCudaErrors(cudaMallocAsync(&schur, bytes, set->stream));
    checkCudaErrors(cudaMallocAsync(&full_rhs, full_bytes, set->stream));
    checkCudaErrors(cudaMallocAsync(&full_solution, full_bytes, set->stream));
    set->host_argv[_ATOL_] = static_cast<T>(nv_tol);
    set->host_params[_MG_USE_INIT_GUESS_] = 0;
    LatticeComplex<T> scalar;
    status = 0;
    for (int column = 0; column < E && status == 0; ++column) {
      if (progress) {
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - progress_start).count();
        std::fprintf(stderr,
                     "PYQCU::CUDA::MG_SETUP: null column=%d/%d elapsed=%.3fs\n",
                     column + 1, E, elapsed);
        std::fflush(stderr);
      }
      give_random_vals<T><<<set->gridDim, set->blockDim, 0, set->stream>>>(
          vector, static_cast<unsigned long>(seed + 10007ULL * column));
      for (int iteration = 0; iteration < nv_iters; ++iteration) {
        applyCloverBistabCgDslashQcu(
            reinterpret_cast<long long>(schur),
            reinterpret_cast<long long>(vector), gauge_ptr, clover_even_ptr,
            clover_odd_ptr, clover_even_inv_ptr, clover_odd_inv_ptr,
            set_ptrs_ptr, params_ptr);
        checkCudaErrors(cudaMemsetAsync(full_rhs, 0, full_bytes, set->stream));
        checkCudaErrors(cudaMemcpyAsync(
            static_cast<unsigned char *>(full_rhs) + bytes, schur, bytes,
            cudaMemcpyDeviceToDevice, set->stream));
        applyCloverBistabCgQcu(
            reinterpret_cast<long long>(full_solution),
            reinterpret_cast<long long>(full_rhs), gauge_ptr, clover_even_ptr,
            clover_odd_ptr, clover_even_inv_ptr, clover_odd_inv_ptr,
            set_ptrs_ptr, params_ptr);
        const int blocks = (count + set->blockDim.x - 1) / set->blockDim.x;
        multigrid_vector_subtract<T><<<blocks, set->blockDim, 0, set->stream>>>(
            vector, vector,
            static_cast<unsigned char *>(full_solution) + bytes, count);
        checkCudaErrors(cudaGetLastError());
      }
      auto *device_scalar = static_cast<LatticeComplex<T> *>(set->device_vals) +
                            _send_tmp_;
      CUBLAS_CHECK(_cublasDot<T>(set->cublasH, count, vector, 1, vector, 1,
                                 device_scalar));
      checkCudaErrors(cudaMemcpyAsync(&scalar, device_scalar,
                                      sizeof(LatticeComplex<T>),
                                      cudaMemcpyDeviceToHost, set->stream));
      checkCudaErrors(cudaStreamSynchronize(set->stream));
      T norm2 = scalar.real();
      MPI_Allreduce(MPI_IN_PLACE, &norm2, 1, mpi_real_type<T>(), MPI_SUM,
                    MPI_COMM_WORLD);
      if (!std::isfinite(norm2) || norm2 <= std::numeric_limits<T>::min()) {
        status = 3;
        break;
      }
      const LatticeComplex<T> inverse_norm(
          static_cast<T>(1) / std::sqrt(norm2), 0);
      CUBLAS_CHECK(_cublasScal<T>(set->cublasH, count, &inverse_norm, vector,
                                 1));
      checkCudaErrors(cudaMemcpyAsync(
          static_cast<unsigned char *>(flat) +
              static_cast<size_t>(column) * bytes,
          vector, bytes, cudaMemcpyDeviceToDevice, set->stream));
    }
    if (status == 0) {
      checkCudaErrors(cudaStreamSynchronize(set->stream));
      status = launch_multigrid_local_orthogonalize_cgs2<T>(
          flat, blocked_out, E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
    }
  } catch (...) {
    status = 2;
  }
  restore_solver_controls();
  release_buffers();
  return status;
}

template <typename T>
int launch_multigrid_build_stencil_local(
    void *sitting, void *hop_nn, void *hop_diag, long long null_vectors_ptr,
    long long gauge_ptr, long long clover_even_ptr,
    long long clover_odd_ptr, long long clover_even_inv_ptr,
    long long clover_odd_inv_ptr, long long set_ptrs_ptr,
    long long params_ptr, int E, int e, int Xf, int Yf, int Zf, int Tf,
    int Xc, int Yc, int Zc, int Tc, int window, int batch_size) {
  if (sitting == nullptr || hop_nn == nullptr || hop_diag == nullptr ||
      null_vectors_ptr == 0 || gauge_ptr == 0 || clover_even_ptr == 0 ||
      clover_odd_ptr == 0 || clover_even_inv_ptr == 0 ||
      clover_odd_inv_ptr == 0 || set_ptrs_ptr == 0 || params_ptr == 0 ||
      E <= 0 || e != _LAT_SC_ || Xf <= 0 || Yf <= 0 || Zf <= 0 || Tf <= 0 ||
      Xc <= 0 || Yc <= 0 || Zc <= 0 || Tc <= 0 || Xf % Xc || Yf % Yc ||
      Zf % Zc || Tf % Tc || batch_size <= 0 || batch_size > 64) {
    return 1;
  }
  const int bx = Xf / Xc;
  const int by = Yf / Yc;
  const int bz = Zf / Zc;
  const int bt = Tf / Tc;
  // Legacy MG uses parity-compressed time, so a valid production hierarchy
  // commonly has bt == 1 while the spatial block extents are still two.
  if (bx < 1 || by < 1 || bz < 1 || bt < 1) return 1;
  const int required_window = std::max(
      std::max(3 * bx + 2, 3 * by + 2),
      std::max(3 * bz + 2, 3 * bt + 2));
  if (window < required_window) return 1;
  if (3 * bx + 2 > Xf || 3 * by + 2 > Yf || 3 * bz + 2 > Zf ||
      3 * bt + 2 > Tf) {
    return 1;
  }
  const int W0 = std::min(Xf, 3 * bx + 3);
  const int W1 = std::min(Yf, 3 * by + 3);
  const int W2 = std::min(Zf, 3 * bz + 3);
  const int W3 = Tf;
  const int64_t local_sites =
      static_cast<int64_t>(W0) * W1 * W2 * W3;
  const int64_t field_stride = 2 * e * local_sites;
  const int64_t gauge_stride = 2 * 3 * 3 * 4 * local_sites;
  const int64_t coarse_count = static_cast<int64_t>(Xc) * Yc * Zc * Tc;
  const bool progress = std::getenv("PYQCU_MG_BUILD_PROGRESS") != nullptr;
  const auto progress_start = std::chrono::steady_clock::now();
  auto last_progress = progress_start;
  auto *params = reinterpret_cast<int *>(params_ptr);
  auto *set_ptrs = reinterpret_cast<long long *>(set_ptrs_ptr);
  auto *set = reinterpret_cast<LatticeSet<T> *>(set_ptrs[params[_SET_INDEX_]]);
  // The local stencil operates on an explicitly gathered periodic patch and
  // never enters a distributed halo path. A process-grid of 1x1x1x1 is the
  // actual safety condition; MPI_COMM_WORLD may still contain multiple ranks
  // when each rank owns an independent patch.
  if (set == nullptr || set->host_params[_GRID_X_] != 1 ||
      set->host_params[_GRID_Y_] != 1 ||
      set->host_params[_GRID_Z_] != 1 || set->host_params[_GRID_T_] != 1 ||
      set->host_params[_LAT_X_] != Xf || set->host_params[_LAT_Y_] != Yf ||
      set->host_params[_LAT_Z_] != Zf || set->host_params[_LAT_T_] != Tf) {
    return 6;
  }

  int local_params[_PARAMS_SIZE_];
  std::copy(set->host_params, set->host_params + _PARAMS_SIZE_, local_params);
  local_params[_LAT_X_] = W0;
  local_params[_LAT_Y_] = W1;
  local_params[_LAT_Z_] = W2;
  // wilson_dslash consumes the already parity-compressed time extent in
  // LatticeSet::host_params; the local window W3 is that extent, not a full
  // physical time dimension that should be halved again.
  local_params[_LAT_T_] = W3;
  local_params[_LAT_XYZT_] = W0 * W1 * W2 * W3;
  local_params[_GRID_X_] = local_params[_GRID_Y_] = 1;
  local_params[_GRID_Z_] = local_params[_GRID_T_] = 1;
  local_params[_NODE_RANK_] = 0;
  local_params[_NODE_SIZE_] = 1;
  local_params[_DAGGER_] = _NO_USE_;
  constexpr int source_parity = _EVEN_;
  constexpr int target_parity = _ODD_;
  const int64_t parity_field = static_cast<int64_t>(e) * local_sites;
  int parity_params[_PARAMS_SIZE_];
  std::copy(local_params, local_params + _PARAMS_SIZE_, parity_params);
  // LatticeCloverBistabCg::dslash performs the first compact hop with the
  // even (0) Wilson parameters and the second with odd (1) parameters.
  parity_params[_PARITY_] = source_parity;
  local_params[_PARITY_] = target_parity;
  local_params[_PARITY_] = target_parity;

  void *local_gauge = nullptr;
  void *source = nullptr;
  void *even_dslash = nullptr;
  void *even_inverse = nullptr;
  void *schur = nullptr;
  void *device_centers = nullptr;
  int *device_params_other = nullptr;
  int *device_params_target = nullptr;
  const int max_batch = batch_size;
  checkCudaErrors(cudaMallocAsync(
      &local_gauge, static_cast<size_t>(max_batch) * gauge_stride *
                         sizeof(LatticeComplex<T>), set->stream));
  checkCudaErrors(cudaMallocAsync(
      &source, static_cast<size_t>(max_batch) * field_stride *
                   sizeof(LatticeComplex<T>), set->stream));
  checkCudaErrors(cudaMallocAsync(
      &even_dslash, static_cast<size_t>(max_batch) * field_stride *
                        sizeof(LatticeComplex<T>), set->stream));
  checkCudaErrors(cudaMallocAsync(
      &even_inverse, static_cast<size_t>(max_batch) * field_stride *
                         sizeof(LatticeComplex<T>), set->stream));
  checkCudaErrors(cudaMallocAsync(
      &schur, static_cast<size_t>(max_batch) * field_stride *
                  sizeof(LatticeComplex<T>), set->stream));
  checkCudaErrors(cudaMallocAsync(
      &device_centers, max_batch * 4 * sizeof(int), set->stream));
  checkCudaErrors(cudaMallocAsync(
      &device_params_other, _PARAMS_SIZE_ * sizeof(int), set->stream));
  checkCudaErrors(cudaMallocAsync(
      &device_params_target, _PARAMS_SIZE_ * sizeof(int), set->stream));
  checkCudaErrors(cudaMemcpyAsync(device_params_other, parity_params,
                                  _PARAMS_SIZE_ * sizeof(int),
                                  cudaMemcpyHostToDevice, set->stream));
  checkCudaErrors(cudaMemcpyAsync(device_params_target, local_params,
                                  _PARAMS_SIZE_ * sizeof(int),
                                  cudaMemcpyHostToDevice, set->stream));

  const int threads = set->blockDim.x;
  const int dslash_blocks =
      static_cast<int>((local_sites + threads - 1) / threads);
  const int64_t block_sites =
      static_cast<int64_t>(bx) * by * bz * bt;
  const T kappa2 = set->kappa() * set->kappa();
  for (int64_t base = 0; base < coarse_count; base += max_batch) {
    if (progress) {
      const auto now = std::chrono::steady_clock::now();
      if (base == 0 || now - last_progress >= std::chrono::seconds(30)) {
        const double elapsed = std::chrono::duration<double>(
            now - progress_start).count();
        std::fprintf(
            stderr,
            "PYQCU::CUDA::MG_SETUP: stencil coarse=%lld/%lld elapsed=%.3fs "
            "batch=%d\n",
            static_cast<long long>(base), static_cast<long long>(coarse_count),
            elapsed, batch_size);
        std::fflush(stderr);
        last_progress = now;
      }
    }
    const int batch = static_cast<int>(std::min<int64_t>(
        max_batch, coarse_count - base));
    std::vector<int> host_centers(static_cast<size_t>(batch) * 4);
    for (int example = 0; example < batch; ++example) {
      int64_t value = base + example;
      host_centers[example * 4 + 3] = value % Tc;
      value /= Tc;
      host_centers[example * 4 + 2] = value % Zc;
      value /= Zc;
      host_centers[example * 4 + 1] = value % Yc;
      host_centers[example * 4] = value / Yc;
    }
    checkCudaErrors(cudaMemcpyAsync(
        device_centers, host_centers.data(), host_centers.size() * sizeof(int),
        cudaMemcpyHostToDevice, set->stream));
    const int64_t gauge_values = static_cast<int64_t>(batch) * gauge_stride;
    multigrid_local_gather_gauge<T>
        <<<(gauge_values + threads - 1) / threads, threads, 0,
            set->stream>>>(
            reinterpret_cast<void *>(gauge_ptr), local_gauge,
            static_cast<int *>(device_centers), batch, Xf, Yf, Zf, Tf, Xc, Yc,
            Zc, Tc, W0, W1, W2, W3, bx, by, bz, bt);
    for (int probe = 0; probe < E; ++probe) {
      checkCudaErrors(cudaMemsetAsync(
          source, 0,
          static_cast<size_t>(batch) * field_stride *
              sizeof(LatticeComplex<T>),
          set->stream));
      const int64_t source_values =
          static_cast<int64_t>(batch) * block_sites * e;
      multigrid_local_fill_source<T>
          <<<(source_values + threads - 1) / threads, threads, 0,
              set->stream>>>(
              source, reinterpret_cast<void *>(null_vectors_ptr),
              static_cast<int *>(device_centers), E, e, Xf, Yf, Zf, Tf, Xc,
              Yc, Zc, Tc, W0, W1, W2, W3, bx, by, bz, bt,
              source_parity, probe, batch);
      for (int example = 0; example < batch; ++example) {
        const int64_t field_offset = static_cast<int64_t>(example) * field_stride;
        const int64_t gauge_offset = static_cast<int64_t>(example) * gauge_stride;
        // The Clover inverse is a global parity field.  Do not index it with
        // window-local coordinates: a unit-gauge smoke cannot expose that
        // aliasing because its Clover term is site independent.  Apply the
        // mapped global Clover inverse explicitly after the Wilson hop.
        wilson_dslash<T, false><<<dslash_blocks, threads, 0, set->stream>>>(
            static_cast<LatticeComplex<T> *>(local_gauge) + gauge_offset,
            static_cast<LatticeComplex<T> *>(source) + field_offset +
                source_parity * parity_field,
            static_cast<LatticeComplex<T> *>(even_dslash) + field_offset +
                source_parity * parity_field,
            device_params_other, nullptr, false);
      }
      const int64_t onsite_values =
          static_cast<int64_t>(batch) * local_sites * 12;
      multigrid_local_onsite_inverse<T>
          <<<(onsite_values + threads - 1) / threads, threads, 0,
              set->stream>>>(
              even_dslash, even_inverse,
              reinterpret_cast<void *>(clover_even_inv_ptr),
              static_cast<int *>(device_centers),
              source_parity, Xf, Yf, Zf, Tf, W0, W1, W2,
              W3, bx, by, bz, bt, batch);
      for (int example = 0; example < batch; ++example) {
        const int64_t field_offset = static_cast<int64_t>(example) * field_stride;
        const int64_t gauge_offset = static_cast<int64_t>(example) * gauge_stride;
        wilson_dslash<T, false><<<dslash_blocks, threads, 0, set->stream>>>(
            static_cast<LatticeComplex<T> *>(local_gauge) + gauge_offset,
            static_cast<LatticeComplex<T> *>(even_inverse) + field_offset +
                source_parity * parity_field,
            static_cast<LatticeComplex<T> *>(schur) + field_offset +
                source_parity * parity_field,
            device_params_target, nullptr, false);
      }
      multigrid_local_onsite_subtract<T>
          <<<(onsite_values + threads - 1) / threads, threads, 0,
              set->stream>>>(
              source, reinterpret_cast<void *>(clover_odd_ptr), schur,
              static_cast<int *>(device_centers), source_parity,
              kappa2, Xf, Yf, Zf, Tf, W0, W1, W2, W3, bx, by, bz, bt, batch);
      multigrid_local_restrict_stencil<T>
          <<<batch, threads, 0, set->stream>>>(
              sitting, hop_nn, hop_diag, schur,
              reinterpret_cast<void *>(null_vectors_ptr),
              static_cast<int *>(device_centers), E, e, Xf, Yf, Zf, Tf, Xc,
              Yc, Zc, Tc, W0, W1, W2, W3, bx, by, bz, bt,
              source_parity, probe);
      checkCudaErrors(cudaGetLastError());
    }
  }
  checkCudaErrors(cudaStreamSynchronize(set->stream));
  checkCudaErrors(cudaFreeAsync(device_params_target, set->stream));
  checkCudaErrors(cudaFreeAsync(device_params_other, set->stream));
  checkCudaErrors(cudaFreeAsync(device_centers, set->stream));
  checkCudaErrors(cudaFreeAsync(schur, set->stream));
  checkCudaErrors(cudaFreeAsync(even_inverse, set->stream));
  checkCudaErrors(cudaFreeAsync(even_dslash, set->stream));
  checkCudaErrors(cudaFreeAsync(source, set->stream));
  checkCudaErrors(cudaFreeAsync(local_gauge, set->stream));
  return 0;
}
} // namespace qcu

extern "C" int applyMultigridLocalOrthogonalizeQcu(
    long long _flat_in, long long _blocked_out, int E, int e, int Xf, int Yf,
    int Zf, int Tf, int Xc, int Yc, int Zc, int Tc, int data_type) {
  try {
    cudaDeviceSynchronize();
    int status = 5;
    if (data_type == _LAT_C64_) {
      status = qcu::launch_multigrid_local_orthogonalize_cgs2<float>(
          reinterpret_cast<void *>(_flat_in),
          reinterpret_cast<void *>(_blocked_out), E, e, Xf, Yf, Zf, Tf,
          Xc, Yc, Zc, Tc);
    } else if (data_type == _LAT_C128_) {
      status = qcu::launch_multigrid_local_orthogonalize_cgs2<double>(
          reinterpret_cast<void *>(_flat_in),
          reinterpret_cast<void *>(_blocked_out), E, e, Xf, Yf, Zf, Tf,
          Xc, Yc, Zc, Tc);
    }
    cudaDeviceSynchronize();
    return status;
  } catch (...) {
    cudaDeviceSynchronize();
    return 2;
  }
}

extern "C" int applyMultigridBuildNullVectorsQcu(
    long long _blocked_out, long long _gauge, long long _clover_even,
    long long _clover_odd, long long _clover_even_inv,
    long long _clover_odd_inv, long long _set_ptrs, long long _params, int E,
    int e, int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc,
    int nv_iters, unsigned long long seed, double nv_tol) {
  try {
    cudaDeviceSynchronize();
    const int *params = reinterpret_cast<const int *>(_params);
    if (params == nullptr) return 1;
    int status = 5;
    if (params[_DATA_TYPE_] == _LAT_C64_) {
      status = qcu::launch_multigrid_build_null_vectors<float>(
          reinterpret_cast<void *>(_blocked_out), _gauge, _clover_even,
          _clover_odd, _clover_even_inv, _clover_odd_inv, _set_ptrs, _params,
          E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc, nv_iters, seed, nv_tol);
    } else if (params[_DATA_TYPE_] == _LAT_C128_) {
      status = qcu::launch_multigrid_build_null_vectors<double>(
          reinterpret_cast<void *>(_blocked_out), _gauge, _clover_even,
          _clover_odd, _clover_even_inv, _clover_odd_inv, _set_ptrs, _params,
          E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc, nv_iters, seed, nv_tol);
    }
    cudaDeviceSynchronize();
    return status;
  } catch (...) {
    cudaDeviceSynchronize();
    return 2;
  }
}

extern "C" int applyMultigridBuildStencilLocalQcu(
    long long _sitting, long long _hop_nn, long long _hop_diag,
    long long _null_vectors, long long _gauge, long long _clover_even,
    long long _clover_odd, long long _clover_even_inv,
    long long _clover_odd_inv, long long _set_ptrs, long long _params, int E,
    int e, int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc,
    int window, int batch_size) {
  try {
    cudaDeviceSynchronize();
    const int *params = reinterpret_cast<const int *>(_params);
    if (params == nullptr) return 1;
    int status = 5;
    if (params[_DATA_TYPE_] == _LAT_C64_) {
      status = qcu::launch_multigrid_build_stencil_local<float>(
          reinterpret_cast<void *>(_sitting), reinterpret_cast<void *>(_hop_nn),
          reinterpret_cast<void *>(_hop_diag), _null_vectors, _gauge,
          _clover_even, _clover_odd, _clover_even_inv, _clover_odd_inv,
          _set_ptrs, _params, E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc, window,
          batch_size);
    } else if (params[_DATA_TYPE_] == _LAT_C128_) {
      status = qcu::launch_multigrid_build_stencil_local<double>(
          reinterpret_cast<void *>(_sitting), reinterpret_cast<void *>(_hop_nn),
          reinterpret_cast<void *>(_hop_diag), _null_vectors, _gauge,
          _clover_even, _clover_odd, _clover_even_inv, _clover_odd_inv,
          _set_ptrs, _params, E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc, window,
          batch_size);
    }
    cudaDeviceSynchronize();
    return status;
  } catch (...) {
    cudaDeviceSynchronize();
    return 2;
  }
}
