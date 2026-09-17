#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "../include/qcu.h"
#include "../python/pyqcu.h"

namespace qcu {
namespace {

namespace cg = cooperative_groups;

inline void strict_check_cuda(cudaError_t status, const char *where);

// Optional, diagnostic-only trace for the fused outer FGMRES.  The default
// path never constructs this file and therefore keeps the production solver
// free of per-iteration host I/O.  The trace deliberately distinguishes the
// cheap Arnoldi/Givens residual estimate from the true residual recomputed at
// every restart boundary.
template <typename T> class StrictFgmresTrace {
 public:
  StrictFgmresTrace() {
    const char *path = std::getenv("PYQCU_STRICT_TRACE_FILE");
    if (path == nullptr || *path == '\0') return;
    file_.open(path, std::ios::out | std::ios::app);
    if (!file_) {
      throw std::runtime_error(
          "cannot open PYQCU_STRICT_TRACE_FILE for append");
    }
    enabled_ = true;
    file_ << std::setprecision(17);
    // v3 adds `iteration_count` records (per-level smoother/restriction/
    // prolongation/recursive-solve counters) on top of the v2 stage/residual
    // layout.  Parsers must accept 1, 2 and 3.
    file_ << "trace_version\t3\n";
  }

  void begin(T rhs_norm) {
    if (!enabled_) return;
    rhs_norm_ = rhs_norm;
    started_ = std::chrono::steady_clock::now();
    file_ << "solve_begin\t" << static_cast<double>(rhs_norm_) << "\n";
  }

  void initial(T residual) {
    if (!enabled_) return;
    file_ << "initial_residual\t0\t" << static_cast<double>(residual)
           << "\t" << relative(residual) << "\t" << elapsed() << "\n";
  }

  void iteration(int global_iteration, int cycle_iteration, T estimate,
                 T next_norm) {
    if (!enabled_) return;
    file_ << "iteration\t" << global_iteration << "\t" << cycle_iteration
           << "\t" << static_cast<double>(estimate) << "\t"
           << relative(estimate) << "\t" << static_cast<double>(next_norm)
           << "\t" << elapsed() << "\n";
  }

  void restart(int global_iteration, T residual) {
    if (!enabled_) return;
    file_ << "restart_residual\t" << global_iteration << "\t"
           << static_cast<double>(residual) << "\t" << relative(residual)
           << "\t" << elapsed() << "\n";
  }

  void end(int iterations, bool converged, T residual) {
    if (!enabled_) return;
    file_ << "solve_end\t" << iterations << "\t" << (converged ? 1 : 0)
           << "\t" << static_cast<double>(residual) << "\t"
           << relative(residual) << "\t" << elapsed() << "\n"
           << std::flush;
  }

  bool enabled() const { return enabled_; }

  void stage(int outer_iteration, int level, const char *name,
             double seconds) {
    if (!enabled_) return;
    file_ << "stage\t" << outer_iteration << "\t" << level << "\t"
           << name << "\t" << seconds << "\t" << elapsed() << "\n";
  }

  void residual(int outer_iteration, int level, const char *name,
                T value, T rhs_norm) {
    if (!enabled_) return;
    const double absolute = static_cast<double>(value);
    const double relative =
        rhs_norm > (T)0 ? static_cast<double>(value / rhs_norm) : 0.0;
    file_ << "residual\t" << outer_iteration << "\t" << level << "\t"
          << name << "\t" << absolute << "\t" << relative << "\t"
          << elapsed() << "\n";
  }

  void counter(int outer_iteration, int level, const char *name,
               unsigned long long value) {
    if (!enabled_) return;
    file_ << "iteration_count\t" << outer_iteration << "\t" << level
          << "\t" << name << "\t" << value << "\t" << elapsed() << "\n";
  }

 private:
  std::ofstream file_;
  bool enabled_ = false;
  T rhs_norm_ = (T)1;
  std::chrono::steady_clock::time_point started_;

  double elapsed() const {
    if (!enabled_) return 0.0;
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now() - started_)
        .count();
  }

  double relative(T value) const {
    return rhs_norm_ > (T)0 ? static_cast<double>(value / rhs_norm_) : 0.0;
  }
};

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

__host__ __device__ inline size_t strict_face_count(
    int X, int Y, int Z, int Lt, int dim) {
  switch (dim) {
    case 0: return static_cast<size_t>(Y) * Z * Lt;
    case 1: return static_cast<size_t>(X) * Z * Lt;
    case 2: return static_cast<size_t>(X) * Y * Lt;
    case 3: return static_cast<size_t>(X) * Y * Z;
    default: return 0;
  }
}

__device__ inline int strict_face_index_excluding(
    int x, int y, int z, int t, int dim,
    int X, int Y, int Z, int Lt) {
  const int coords[4] = {x, y, z, t};
  const int extents[4] = {X, Y, Z, Lt};
  int index = 0;
  for (int axis = 0; axis < 4; ++axis) {
    if (axis == dim) continue;
    index = index * extents[axis] + coords[axis];
  }
  return index;
}

__device__ inline bool strict_neighbor_is_remote(
    int coordinate, int extent, int process_extent, bool forward) {
  return process_extent > 1 &&
         (forward ? coordinate == extent - 1 : coordinate == 0);
}

template <typename T, bool COMPACT>
__device__ inline LatticeComplex<T> strict_load_neighbor(
    const LatticeComplex<T> *input, const LatticeComplex<T> *ghost,
    size_t ghost_slot_stride, int components, int X, int Y, int Z, int Lt,
    int grid_x, int grid_y, int grid_z, int grid_t, int component,
    int x, int y, int z, int t, int dim, bool forward) {
  int neighbor[4] = {x, y, z, t};
  const int extents[4] = {X, Y, Z, Lt};
  const int process_extents[4] = {grid_x, grid_y, grid_z, grid_t};
  const int coordinate = neighbor[dim];
  if (forward) {
    neighbor[dim] = (coordinate + 1) % extents[dim];
  } else {
    neighbor[dim] = (coordinate + extents[dim] - 1) % extents[dim];
  }

  if (strict_neighbor_is_remote(
          coordinate, extents[dim], process_extents[dim], forward)) {
    const int slot = 2 * dim + (forward ? 0 : 1);
    const int face = strict_face_index_excluding(
        neighbor[0], neighbor[1], neighbor[2], neighbor[3],
        dim, X, Y, Z, Lt);
    const size_t face_stride =
        static_cast<size_t>(components) * ghost_slot_stride;
    const size_t slot_base =
        static_cast<size_t>(slot) * face_stride;
    const size_t component_base =
        static_cast<size_t>(component) * ghost_slot_stride;
    return ghost[slot_base + component_base + face];
  }

  if (COMPACT) {
    const int half_volume = X * Y * Z * (Lt / 2);
    const int site = strict_half_site(
        neighbor[0], neighbor[1], neighbor[2], neighbor[3], Y, Z, Lt);
    return input[static_cast<size_t>(component) * half_volume + site];
  }
  const int volume = X * Y * Z * Lt;
  const int site = strict_full_site(
      neighbor[0], neighbor[1], neighbor[2], neighbor[3], Y, Z, Lt);
  return input[static_cast<size_t>(component) * volume + site];
}

template <typename T, bool COMPACT>
__global__ void strict_pack_face_kernel(
    void *packed_ptr, const void *input_ptr, int components,
    int X, int Y, int Z, int Lt, int dim, int side, int parity,
    int component_stride) {
  const int face_count = static_cast<int>(
      strict_face_count(X, Y, Z, Lt, dim));
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= components * face_count) return;

  LatticeComplex<T> *packed =
      static_cast<LatticeComplex<T> *>(packed_ptr);
  const LatticeComplex<T> *input =
      static_cast<const LatticeComplex<T> *>(input_ptr);
  const int component = index / face_count;
  const int face = index - component * face_count;
  int remaining = face;
  const int extents[4] = {X, Y, Z, Lt};
  int coords[4] = {0, 0, 0, 0};
  for (int axis = 3; axis >= 0; --axis) {
    if (axis == dim) continue;
    coords[axis] = remaining % extents[axis];
    remaining /= extents[axis];
  }
  coords[dim] = side == 0 ? 0 : extents[dim] - 1;

  if (COMPACT &&
      (((coords[0] + coords[1] + coords[2] + coords[3]) & 1) != parity)) {
    packed[static_cast<size_t>(component) * component_stride + face] =
        LatticeComplex<T>((T)0, (T)0);
    return;
  }
  if (COMPACT) {
    const int half_volume = X * Y * Z * (Lt / 2);
    const int site = strict_half_site(
        coords[0], coords[1], coords[2], coords[3], Y, Z, Lt);
    packed[static_cast<size_t>(component) * component_stride + face] =
        input[static_cast<size_t>(component) * half_volume + site];
  } else {
    const int volume = X * Y * Z * Lt;
    const int site = strict_full_site(
        coords[0], coords[1], coords[2], coords[3], Y, Z, Lt);
    packed[static_cast<size_t>(component) * component_stride + face] =
        input[static_cast<size_t>(component) * volume + site];
  }
}

template <typename T>
inline void strict_launch_pack_face(
    bool compact, void *packed, const void *input, int components,
    int X, int Y, int Z, int Lt, int dim, int side, int parity,
    cudaStream_t stream, int component_stride = -1) {
  const size_t face_count = strict_face_count(X, Y, Z, Lt, dim);
  const size_t count = static_cast<size_t>(components) * face_count;
  const int blocks =
      static_cast<int>((count + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
  const int stride = component_stride < 0
                         ? static_cast<int>(face_count)
                         : component_stride;
  if (compact) {
    strict_pack_face_kernel<T, true><<<blocks, _BLOCK_SIZE_, 0, stream>>>(
        packed, input, components, X, Y, Z, Lt, dim, side, parity, stride);
  } else {
    strict_pack_face_kernel<T, false><<<blocks, _BLOCK_SIZE_, 0, stream>>>(
        packed, input, components, X, Y, Z, Lt, dim, side, parity, stride);
  }
}

template <typename T>
__global__ void strict_pack_compact_faces_kernel(
    void *packed_ptr, const void *input_ptr, int components,
    int X, int Y, int Z, int Lt, int dim, int side, int parity,
    int component_stride) {
  const int half_volume = X * Y * Z * (Lt / 2);
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= components * half_volume) return;

  LatticeComplex<T> *packed =
      static_cast<LatticeComplex<T> *>(packed_ptr);
  const LatticeComplex<T> *input =
      static_cast<const LatticeComplex<T> *>(input_ptr);
  const int component = index / half_volume;
  const int half_site = index - component * half_volume;
  int x, y, z, t;
  strict_decode_half_site(
      half_site, parity, X, Y, Z, Lt, x, y, z, t);
  const int coordinates[4] = {x, y, z, t};
  const int extents[4] = {X, Y, Z, Lt};
  const int face_count = static_cast<int>(
      strict_face_count(X, Y, Z, Lt, dim));
  const int face = strict_face_index_excluding(
      x, y, z, t, dim, X, Y, Z, Lt);
  const LatticeComplex<T> value = input[index];
  if ((side == 0 && coordinates[dim] == 0) ||
      (side == 1 && coordinates[dim] == extents[dim] - 1))
    packed[static_cast<size_t>(component) * component_stride + face] = value;
}

template <typename T>
inline void strict_launch_pack_compact_faces(
    void *packed, const void *input, int components,
    int X, int Y, int Z, int Lt, int dim, int side, int parity,
    cudaStream_t stream, int component_stride = -1) {
  const size_t count =
      static_cast<size_t>(components) * X * Y * Z * (Lt / 2);
  const int blocks =
      static_cast<int>((count + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
  strict_pack_compact_faces_kernel<T><<<blocks, _BLOCK_SIZE_, 0, stream>>>(
      packed, input, components, X, Y, Z, Lt, dim, side, parity,
      component_stride < 0
          ? static_cast<int>(strict_face_count(X, Y, Z, Lt, dim))
          : component_stride);
}

template <typename T>
__global__ void strict_coarse_apply_kernel(
    void *out_ptr, const void *in_ptr, const void *links_ptr,
    const void *onsite_pair_ptr, const void *input_ghost_ptr,
    size_t input_ghost_slot_stride, int E, int X, int Y, int Z, int Lt,
    int grid_x, int grid_y, int grid_z, int grid_t, int onsite_index) {
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
  const LatticeComplex<T> *input_ghost =
      static_cast<const LatticeComplex<T> *>(input_ghost_ptr);

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

  for (int dim = 0; dim < 4; ++dim) {
    const int coords[4] = {x, y, z, t};
    const int extents[4] = {X, Y, Z, Lt};
    const bool backward_remote = strict_neighbor_is_remote(
        coords[dim], extents[dim],
        dim == 0 ? grid_x : dim == 1 ? grid_y : dim == 2 ? grid_z : grid_t,
        false);
    int backward[4] = {x, y, z, t};
    backward[dim] = (coords[dim] + extents[dim] - 1) % extents[dim];
    const int backward_site = strict_full_site(
        backward[0], backward[1], backward[2], backward[3], Y, Z, Lt);
    for (int col = 0; col < E; ++col) {
      const size_t forward_link =
          ((((static_cast<size_t>(0) * 4 + dim) * E + row) * E + col) *
               static_cast<size_t>(volume) +
           site);
      const LatticeComplex<T> forward_value =
          strict_load_neighbor<T, false>(
              in, input_ghost, input_ghost_slot_stride, E, X, Y, Z, Lt,
              grid_x, grid_y, grid_z, grid_t, col, x, y, z, t, dim, true);
      // QUDA stores the backward link at q-mu.  The action at q therefore
      // reads [col,row,q-mu] and conjugates it (matrix adjoint).
      const size_t backward_link =
          ((((static_cast<size_t>(1) * 4 + dim) * E + col) * E + row) *
               static_cast<size_t>(volume) +
           backward_site);
      sum += links[forward_link] * forward_value;
      if (!backward_remote) {
        const LatticeComplex<T> backward_value =
            strict_load_neighbor<T, false>(
                in, input_ghost, input_ghost_slot_stride, E, X, Y, Z, Lt,
                grid_x, grid_y, grid_z, grid_t, col, x, y, z, t, dim, false);
        sum += links[backward_link].conj() * backward_value;
      }
    }
  }
  out[index] = sum;
}

template <typename T>
__global__ void strict_coarse_apply_backward_correction_kernel(
    void *out_ptr, const void *input_ghost_ptr,
    const void *link_ghost_ptr, int E, int X, int Y, int Z, int Lt,
    int dim) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int volume = X * Y * Z * Lt;
  if (index >= E * volume) return;

  LatticeComplex<T> *out = static_cast<LatticeComplex<T> *>(out_ptr);
  const LatticeComplex<T> *input_ghost =
      static_cast<const LatticeComplex<T> *>(input_ghost_ptr);
  const LatticeComplex<T> *link_ghost =
      static_cast<const LatticeComplex<T> *>(link_ghost_ptr);

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
  const int coordinates[4] = {x, y, z, t};
  if (coordinates[dim] != 0) return;

  int neighbor[4] = {x, y, z, t};
  const int extents[4] = {X, Y, Z, Lt};
  neighbor[dim] = extents[dim] - 1;
  const int face_count = static_cast<int>(
      strict_face_count(X, Y, Z, Lt, dim));
  const int face = strict_face_index_excluding(
      neighbor[0], neighbor[1], neighbor[2], neighbor[3],
      dim, X, Y, Z, Lt);
  LatticeComplex<T> correction((T)0, (T)0);
  for (int col = 0; col < E; ++col) {
    const size_t link_index =
        (static_cast<size_t>(col) * E + row) * face_count + face;
    const size_t input_index =
        static_cast<size_t>(col) * face_count + face;
    correction += link_ghost[link_index].conj() * input_ghost[input_index];
  }
  out[index] += correction;
}

template <typename T>
__global__ void strict_hopping_parity_base_kernel(
    void *out_ptr, const void *in_ptr, const void *links_ptr,
    const void *base_ptr, const void *input_ghost_ptr,
    size_t input_ghost_slot_stride, int E, int X, int Y, int Z, int Lt,
    int grid_x, int grid_y, int grid_z, int grid_t, int target_parity) {
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
  const LatticeComplex<T> *input_ghost =
      static_cast<const LatticeComplex<T> *>(input_ghost_ptr);

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
    const int process_extent =
        dim == 0 ? grid_x : dim == 1 ? grid_y :
        dim == 2 ? grid_z : grid_t;
    const bool backward_remote = strict_neighbor_is_remote(
        coords[dim], extents[dim], process_extent, false);
    int backward[4] = {x, y, z, t};
    backward[dim] = (coords[dim] + extents[dim] - 1) % extents[dim];
    const int backward_site = strict_full_site(
        backward[0], backward[1], backward[2], backward[3], Y, Z, Lt);
    for (int col = 0; col < E; ++col) {
      const size_t forward_link =
          ((((static_cast<size_t>(0) * 4 + dim) * E + row) * E + col) *
               static_cast<size_t>(volume) +
           target_site);
      const LatticeComplex<T> forward_value =
          strict_load_neighbor<T, true>(
              in, input_ghost, input_ghost_slot_stride, E,
              X, Y, Z, Lt, grid_x, grid_y, grid_z, grid_t,
              col, x, y, z, t, dim, true);
      const size_t backward_link =
          ((((static_cast<size_t>(1) * 4 + dim) * E + col) * E + row) *
               static_cast<size_t>(volume) +
           backward_site);
      sum += links[forward_link] * forward_value;
      if (!backward_remote) {
        const LatticeComplex<T> backward_value =
            strict_load_neighbor<T, true>(
                in, input_ghost, input_ghost_slot_stride, E,
                X, Y, Z, Lt, grid_x, grid_y, grid_z, grid_t,
                col, x, y, z, t, dim, false);
        sum += links[backward_link].conj() * backward_value;
      }
    }
  }
  out[index] = base == nullptr ? sum : base[index] - sum;
}

template <typename T>
__global__ void strict_hopping_parity_backward_correction_kernel(
    void *out_ptr, const void *input_ghost_ptr,
    const void *link_ghost_ptr, int E, int X, int Y, int Z, int Lt,
    int target_parity, int dim, int base_present) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int half_volume = X * Y * Z * (Lt / 2);
  if (index >= E * half_volume) return;

  LatticeComplex<T> *out = static_cast<LatticeComplex<T> *>(out_ptr);
  const LatticeComplex<T> *input_ghost =
      static_cast<const LatticeComplex<T> *>(input_ghost_ptr);
  const LatticeComplex<T> *link_ghost =
      static_cast<const LatticeComplex<T> *>(link_ghost_ptr);

  const int row = index / half_volume;
  const int half_site = index - row * half_volume;
  int x, y, z, t;
  strict_decode_half_site(
      half_site, target_parity, X, Y, Z, Lt, x, y, z, t);
  const int coordinates[4] = {x, y, z, t};
  if (coordinates[dim] != 0) return;

  int neighbor[4] = {x, y, z, t};
  const int extents[4] = {X, Y, Z, Lt};
  neighbor[dim] = extents[dim] - 1;
  const int face_count = static_cast<int>(
      strict_face_count(X, Y, Z, Lt, dim));
  const int face = strict_face_index_excluding(
      neighbor[0], neighbor[1], neighbor[2], neighbor[3],
      dim, X, Y, Z, Lt);
  LatticeComplex<T> correction((T)0, (T)0);
  for (int col = 0; col < E; ++col) {
    const size_t link_index =
        (static_cast<size_t>(col) * E + row) * face_count + face;
    const size_t input_index =
        static_cast<size_t>(col) * face_count + face;
    correction += link_ghost[link_index].conj() * input_ghost[input_index];
  }
  // The parity base kernel skips the backward term when the source lives on a
  // neighbouring rank.  Whether the remote term enters with a plus or minus
  // sign depends on the base kernel's own accumulation: hop(out=H in) adds it,
  // while hop(out=base-H in) subtracts it.
  out[index] += base_present ? -correction : correction;
}

template <typename T>
__device__ inline bool strict_fused_scalar_bad(
    const LatticeComplex<T> &value) {
  return !((value.real() == value.real()) &&
           (value.imag() == value.imag()) &&
           fabs(value.real()) != INFINITY &&
           fabs(value.imag()) != INFINITY);
}

template <typename T>
__device__ inline T strict_fused_abs2(const LatticeComplex<T> &value) {
  return value.real() * value.real() + value.imag() * value.imag();
}

template <typename T>
__device__ inline bool strict_fused_abort(
    cg::grid_group &grid, int block_thread, int block_index,
    int *status, bool bad) {
  if (bad && block_thread == 0)
    atomicExch(status, 1);
  grid.sync();
  return *status != 0;
}

template <typename T, int NT>
__device__ inline void strict_fused_hop(
    LatticeComplex<T> *out, const LatticeComplex<T> *in,
    const LatticeComplex<T> *links, const LatticeComplex<T> *base,
    int E, int X, int Y, int Z, int Lt, int target_parity,
    int nblocks, int block_index, int block_thread) {
  const int half_volume = X * Y * Z * (Lt / 2);
  const int total = E * half_volume;
  const int volume = 2 * half_volume;
  for (int index = block_index * NT + block_thread; index < total;
       index += nblocks * NT) {
    const int row = index / half_volume;
    const int half_site = index - row * half_volume;
    int x, y, z, t;
    strict_decode_half_site(
        half_site, target_parity, X, Y, Z, Lt, x, y, z, t);
    const int target_site = strict_full_site(x, y, z, t, Y, Z, Lt);
    const int coords[4] = {x, y, z, t};
    const int extents[4] = {X, Y, Z, Lt};

    LatticeComplex<T> sum((T)0, (T)0);
    for (int dim = 0; dim < 4; ++dim) {
      int forward[4] = {x, y, z, t};
      int backward[4] = {x, y, z, t};
      forward[dim] = (coords[dim] + 1) % extents[dim];
      backward[dim] = (coords[dim] + extents[dim] - 1) % extents[dim];
      const int backward_site = strict_full_site(
          backward[0], backward[1], backward[2], backward[3], Y, Z, Lt);
      const int forward_half = strict_half_site(
          forward[0], forward[1], forward[2], forward[3], Y, Z, Lt);
      const int backward_half = strict_half_site(
          backward[0], backward[1], backward[2], backward[3], Y, Z, Lt);
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
}

template <typename T, int NT>
__device__ inline LatticeComplex<T> strict_fused_dot(
    const LatticeComplex<T> *left, const LatticeComplex<T> *right,
    int n, LatticeComplex<T> *partials, int nblocks,
    typename LatticeComplex<T>::_data_type *shared,
    cg::grid_group &grid, int block_index, int block_thread) {
  LatticeComplex<T> sum((T)0, (T)0);
  for (int index = block_index * NT + block_thread; index < n;
       index += nblocks * NT)
    sum += left[index].conj() * right[index];
  shared[block_thread].x = sum.real();
  shared[block_thread].y = sum.imag();
  __syncthreads();
  for (int width = NT / 2; width > 0; width >>= 1) {
    if (block_thread < width) {
      shared[block_thread].x += shared[block_thread + width].x;
      shared[block_thread].y += shared[block_thread + width].y;
    }
    __syncthreads();
  }
  if (block_thread == 0)
    partials[block_index] =
        LatticeComplex<T>(shared[0].x, shared[0].y);
  grid.sync();
  if (block_index == 0) {
    LatticeComplex<T> total((T)0, (T)0);
    for (int index = 0; index < nblocks; ++index)
      total += partials[index];
    partials[0] = total;
  }
  grid.sync();
  return partials[0];
}

template <typename T, int NT>
__device__ inline void strict_fused_dot_pair(
    const LatticeComplex<T> *left0, const LatticeComplex<T> *right0,
    const LatticeComplex<T> *left1, const LatticeComplex<T> *right1,
    int n, LatticeComplex<T> *partials, int nblocks,
    typename LatticeComplex<T>::_data_type *shared0,
    typename LatticeComplex<T>::_data_type *shared1,
    cg::grid_group &grid, int block_index, int block_thread,
    LatticeComplex<T> &result0, LatticeComplex<T> &result1) {
  LatticeComplex<T> sum0((T)0, (T)0);
  LatticeComplex<T> sum1((T)0, (T)0);
  for (int index = block_index * NT + block_thread; index < n;
       index += nblocks * NT) {
    sum0 += left0[index].conj() * right0[index];
    sum1 += left1[index].conj() * right1[index];
  }
  shared0[block_thread].x = sum0.real();
  shared0[block_thread].y = sum0.imag();
  shared1[block_thread].x = sum1.real();
  shared1[block_thread].y = sum1.imag();
  __syncthreads();
  for (int width = NT / 2; width > 0; width >>= 1) {
    if (block_thread < width) {
      shared0[block_thread].x += shared0[block_thread + width].x;
      shared0[block_thread].y += shared0[block_thread + width].y;
      shared1[block_thread].x += shared1[block_thread + width].x;
      shared1[block_thread].y += shared1[block_thread + width].y;
    }
    __syncthreads();
  }
  if (block_thread == 0) {
    partials[block_index] =
        LatticeComplex<T>(shared0[0].x, shared0[0].y);
    partials[nblocks + block_index] =
        LatticeComplex<T>(shared1[0].x, shared1[0].y);
  }
  grid.sync();
  if (block_index == 0) {
    LatticeComplex<T> total0((T)0, (T)0);
    LatticeComplex<T> total1((T)0, (T)0);
    for (int index = 0; index < nblocks; ++index) {
      total0 += partials[index];
      total1 += partials[nblocks + index];
    }
    partials[0] = total0;
    partials[nblocks] = total1;
  }
  grid.sync();
  result0 = partials[0];
  result1 = partials[nblocks];
}

template <typename T, int NT>
__global__ void strict_coarse_bicgstab_fused_kernel(
    void *x_ptr, const void *rhs_ptr, void *rhat_ptr, void *r_ptr,
    void *p_ptr, void *v_ptr, void *s_ptr, void *t_ptr, void *tmp_ptr,
    const void *links_ptr, int E, int X, int Y, int Z, int Lt,
    int parity, int max_iter, T tolerance, void *partials_ptr,
    int nblocks, int *status_ptr) {
  cg::grid_group grid = cg::this_grid();
  const int block_index = static_cast<int>(blockIdx.x);
  const int block_thread = static_cast<int>(threadIdx.x);
  using complex_data = typename LatticeComplex<T>::_data_type;
  __shared__ complex_data shared0[NT];
  __shared__ complex_data shared1[NT];

  LatticeComplex<T> *x =
      static_cast<LatticeComplex<T> *>(x_ptr);
  const LatticeComplex<T> *rhs =
      static_cast<const LatticeComplex<T> *>(rhs_ptr);
  LatticeComplex<T> *rhat =
      static_cast<LatticeComplex<T> *>(rhat_ptr);
  LatticeComplex<T> *r =
      static_cast<LatticeComplex<T> *>(r_ptr);
  LatticeComplex<T> *p =
      static_cast<LatticeComplex<T> *>(p_ptr);
  LatticeComplex<T> *v =
      static_cast<LatticeComplex<T> *>(v_ptr);
  LatticeComplex<T> *s =
      static_cast<LatticeComplex<T> *>(s_ptr);
  LatticeComplex<T> *t =
      static_cast<LatticeComplex<T> *>(t_ptr);
  LatticeComplex<T> *tmp =
      static_cast<LatticeComplex<T> *>(tmp_ptr);
  const LatticeComplex<T> *links =
      static_cast<const LatticeComplex<T> *>(links_ptr);
  LatticeComplex<T> *partials =
      static_cast<LatticeComplex<T> *>(partials_ptr);
  const int n = E * X * Y * Z * (Lt / 2);
  const T floor = std::is_same<T, float>::value ? (T)1e-20 : (T)1e-40;

  for (int index = block_index * NT + block_thread; index < n;
       index += nblocks * NT) {
    x[index] = LatticeComplex<T>((T)0, (T)0);
    r[index] = rhs[index];
    rhat[index] = rhs[index];
    p[index] = LatticeComplex<T>((T)0, (T)0);
    v[index] = LatticeComplex<T>((T)0, (T)0);
    s[index] = LatticeComplex<T>((T)0, (T)0);
    t[index] = LatticeComplex<T>((T)0, (T)0);
  }
  grid.sync();

  const LatticeComplex<T> rhs_norm = strict_fused_dot<T, NT>(
      rhs, rhs, n, partials, nblocks, shared0, grid,
      block_index, block_thread);
  const bool bad_rhs = strict_fused_scalar_bad(rhs_norm) ||
                       rhs_norm.real() < (T)0 ||
                       !((tolerance == tolerance) &&
                         fabs(tolerance) != INFINITY) ||
                       tolerance <= (T)0;
  if (strict_fused_abort<T>(
          grid, block_thread, block_index, status_ptr, bad_rhs))
    return;
  const T rhs_norm2 = rhs_norm.real() > (T)0 ? rhs_norm.real() : (T)0;
  const T target2 = tolerance * tolerance * rhs_norm2;
  if (target2 == (T)0) {
    if (block_index == 0 && block_thread == 0)
      status_ptr[1] = 1;
    grid.sync();
    return;
  }

  LatticeComplex<T> rho_prev((T)1, (T)0);
  LatticeComplex<T> alpha((T)1, (T)0);
  LatticeComplex<T> omega((T)1, (T)0);
  for (int iteration = 0; iteration < max_iter; ++iteration) {
    if (block_index == 0 && block_thread == 0)
      status_ptr[2] = iteration + 1;
    grid.sync();
    const LatticeComplex<T> rho = strict_fused_dot<T, NT>(
        rhat, r, n, partials, nblocks, shared0, grid,
        block_index, block_thread);
    if (strict_fused_abort<T>(
            grid, block_thread, block_index, status_ptr,
            strict_fused_scalar_bad(rho) ||
                strict_fused_abs2(rho) <= floor ||
                strict_fused_abs2(omega) <= floor))
      return;

    const LatticeComplex<T> beta =
        (rho / rho_prev) * (alpha / omega);
    if (strict_fused_abort<T>(
            grid, block_thread, block_index, status_ptr,
            strict_fused_scalar_bad(beta)))
      return;
    rho_prev = rho;
    for (int index = block_index * NT + block_thread; index < n;
         index += nblocks * NT)
      p[index] = r[index] + beta * (p[index] - omega * v[index]);
    grid.sync();

    strict_fused_hop<T, NT>(
        tmp, p, links, nullptr, E, X, Y, Z, Lt, 1 - parity,
        nblocks, block_index, block_thread);
    grid.sync();
    strict_fused_hop<T, NT>(
        v, tmp, links, p, E, X, Y, Z, Lt, parity,
        nblocks, block_index, block_thread);
    grid.sync();

    const LatticeComplex<T> denominator = strict_fused_dot<T, NT>(
        rhat, v, n, partials, nblocks, shared0, grid,
        block_index, block_thread);
    if (strict_fused_abort<T>(
            grid, block_thread, block_index, status_ptr,
            strict_fused_scalar_bad(denominator) ||
                strict_fused_abs2(denominator) <= floor))
      return;
    alpha = rho / denominator;
    if (strict_fused_abort<T>(
            grid, block_thread, block_index, status_ptr,
            strict_fused_scalar_bad(alpha)))
      return;

    for (int index = block_index * NT + block_thread; index < n;
         index += nblocks * NT)
      s[index] = r[index] - alpha * v[index];
    grid.sync();

    const LatticeComplex<T> s_norm = strict_fused_dot<T, NT>(
        s, s, n, partials, nblocks, shared0, grid,
        block_index, block_thread);
    if (strict_fused_abort<T>(
            grid, block_thread, block_index, status_ptr,
            strict_fused_scalar_bad(s_norm) || s_norm.real() < (T)0))
      return;
    if (s_norm.real() <= target2) {
      for (int index = block_index * NT + block_thread; index < n;
           index += nblocks * NT)
        x[index] += alpha * p[index];
      grid.sync();
      if (block_index == 0 && block_thread == 0)
        status_ptr[1] = 1;
      grid.sync();
      return;
    }

    strict_fused_hop<T, NT>(
        tmp, s, links, nullptr, E, X, Y, Z, Lt, 1 - parity,
        nblocks, block_index, block_thread);
    grid.sync();
    strict_fused_hop<T, NT>(
        t, tmp, links, s, E, X, Y, Z, Lt, parity,
        nblocks, block_index, block_thread);
    grid.sync();

    LatticeComplex<T> ts;
    LatticeComplex<T> tt;
    strict_fused_dot_pair<T, NT>(
        t, s, t, t, n, partials, nblocks, shared0, shared1,
        grid, block_index, block_thread, ts, tt);
    if (strict_fused_abort<T>(
            grid, block_thread, block_index, status_ptr,
            strict_fused_scalar_bad(ts) ||
                strict_fused_scalar_bad(tt) ||
                strict_fused_abs2(tt) <= floor))
      return;
    omega = ts / tt;
    if (strict_fused_abort<T>(
            grid, block_thread, block_index, status_ptr,
            strict_fused_scalar_bad(omega) ||
                strict_fused_abs2(omega) <= floor))
      return;

    for (int index = block_index * NT + block_thread; index < n;
         index += nblocks * NT) {
      x[index] += alpha * p[index] + omega * s[index];
      r[index] = s[index] - omega * t[index];
    }
    grid.sync();

    const LatticeComplex<T> residual = strict_fused_dot<T, NT>(
        r, r, n, partials, nblocks, shared0, grid,
        block_index, block_thread);
    if (strict_fused_abort<T>(
            grid, block_thread, block_index, status_ptr,
            strict_fused_scalar_bad(residual) ||
                residual.real() < (T)0))
      return;
    if (residual.real() <= target2) {
      if (block_index == 0 && block_thread == 0)
        status_ptr[1] = 1;
      grid.sync();
      return;
    }
  }
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
__global__ void strict_fine_matpc_update_kernel(
    void *out_ptr, const void *identity_ptr, const void *cross_ptr,
    T kappa_squared, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) return;
  LatticeComplex<T> *out = static_cast<LatticeComplex<T> *>(out_ptr);
  const LatticeComplex<T> *identity =
      static_cast<const LatticeComplex<T> *>(identity_ptr);
  const LatticeComplex<T> *cross =
      static_cast<const LatticeComplex<T> *>(cross_ptr);
  out[index] = identity[index] - cross[index] * kappa_squared;
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

// The strict solver is deliberately single-rank, so its reductions do not
// need the MPI staging used by the legacy solver.  Keeping all inner products
// for one Arnoldi column in one launch is important on WSL2, where the host
// round trip costs considerably more than the reduction itself.
template <typename T, int NT>
__global__ void strict_dot_many_kernel(
    const LatticeComplex<T> *basis, size_t basis_stride,
    const LatticeComplex<T> *right, int n, int count,
    LatticeComplex<T> *out) {
  using complex_data = typename LatticeComplex<T>::_data_type;
  const int column = static_cast<int>(blockIdx.x);
  if (column >= count) return;

  __shared__ complex_data partial[NT];
  const LatticeComplex<T> *left = basis +
      static_cast<size_t>(column) * basis_stride;
  LatticeComplex<T> sum((T)0, (T)0);
  for (int index = threadIdx.x; index < n; index += NT)
    sum += left[index].conj() * right[index];
  partial[threadIdx.x].x = sum.real();
  partial[threadIdx.x].y = sum.imag();
  __syncthreads();
  for (int width = NT / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) {
      partial[threadIdx.x].x += partial[threadIdx.x + width].x;
      partial[threadIdx.x].y += partial[threadIdx.x + width].y;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    out[column] = LatticeComplex<T>(partial[0].x, partial[0].y);
}

// Two products with the same vector length are reduced together.  This is
// used by MR and BiCGStab, where both scalars are needed before the next
// update.  The output is written into the existing device_vals allocation;
// no additional arena bytes are required.
template <typename T, int NT>
__global__ void strict_dot_pair_kernel(
    const LatticeComplex<T> *left0, const LatticeComplex<T> *right0,
    const LatticeComplex<T> *left1, const LatticeComplex<T> *right1,
    int n, LatticeComplex<T> *out) {
  using complex_data = typename LatticeComplex<T>::_data_type;
  __shared__ complex_data partial0[NT];
  __shared__ complex_data partial1[NT];
  LatticeComplex<T> sum0((T)0, (T)0);
  for (int index = threadIdx.x; index < n; index += NT)
    sum0 += left0[index].conj() * right0[index];
  partial0[threadIdx.x].x = sum0.real();
  partial0[threadIdx.x].y = sum0.imag();

  LatticeComplex<T> sum1((T)0, (T)0);
  for (int index = threadIdx.x; index < n; index += NT)
    sum1 += left1[index].conj() * right1[index];
  partial1[threadIdx.x].x = sum1.real();
  partial1[threadIdx.x].y = sum1.imag();
  __syncthreads();
  for (int width = NT / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) {
      partial0[threadIdx.x].x += partial0[threadIdx.x + width].x;
      partial0[threadIdx.x].y += partial0[threadIdx.x + width].y;
      partial1[threadIdx.x].x += partial1[threadIdx.x + width].x;
      partial1[threadIdx.x].y += partial1[threadIdx.x + width].y;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[0] = LatticeComplex<T>(partial0[0].x, partial0[0].y);
    out[1] = LatticeComplex<T>(partial1[0].x, partial1[0].y);
  }
}

// The original strict dot kernels intentionally used one block so the first
// implementation had no reduction workspace.  That is a poor trade-off for
// fine-grid vectors: with a 2M-element vector, every one of 256 threads had
// to perform roughly 8K serial products.  Keep the same accumulation type and
// conjugation convention, but split the vector into grid-stride block slices.
// The second reduction is tiny and remains on the same stream, so the host
// synchronization/API contract is unchanged.
template <typename T, int NT>
__global__ void strict_dot_pair_partial_kernel(
    const LatticeComplex<T> *left0, const LatticeComplex<T> *right0,
    const LatticeComplex<T> *left1, const LatticeComplex<T> *right1,
    int n, LatticeComplex<T> *partials) {
  using complex_data = typename LatticeComplex<T>::_data_type;
  __shared__ complex_data partial0[NT];
  __shared__ complex_data partial1[NT];
  const int block = blockIdx.x;
  const int index = block * NT + threadIdx.x;
  const int stride = gridDim.x * NT;
  LatticeComplex<T> sum0((T)0, (T)0);
  LatticeComplex<T> sum1((T)0, (T)0);
  for (int i = index; i < n; i += stride) {
    sum0 += left0[i].conj() * right0[i];
    sum1 += left1[i].conj() * right1[i];
  }
  partial0[threadIdx.x].x = sum0.real();
  partial0[threadIdx.x].y = sum0.imag();
  partial1[threadIdx.x].x = sum1.real();
  partial1[threadIdx.x].y = sum1.imag();
  __syncthreads();
  for (int width = NT / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) {
      partial0[threadIdx.x].x += partial0[threadIdx.x + width].x;
      partial0[threadIdx.x].y += partial0[threadIdx.x + width].y;
      partial1[threadIdx.x].x += partial1[threadIdx.x + width].x;
      partial1[threadIdx.x].y += partial1[threadIdx.x + width].y;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partials[block] = LatticeComplex<T>(partial0[0].x, partial0[0].y);
    partials[gridDim.x + block] =
        LatticeComplex<T>(partial1[0].x, partial1[0].y);
  }
}

template <typename T, int NT>
__global__ void strict_dot_pair_reduce_kernel(
    const LatticeComplex<T> *partials, int count,
    LatticeComplex<T> *out) {
  using complex_data = typename LatticeComplex<T>::_data_type;
  __shared__ complex_data partial0[NT];
  __shared__ complex_data partial1[NT];
  LatticeComplex<T> sum0((T)0, (T)0);
  LatticeComplex<T> sum1((T)0, (T)0);
  for (int i = threadIdx.x; i < count; i += NT) {
    sum0 += partials[i];
    sum1 += partials[count + i];
  }
  partial0[threadIdx.x].x = sum0.real();
  partial0[threadIdx.x].y = sum0.imag();
  partial1[threadIdx.x].x = sum1.real();
  partial1[threadIdx.x].y = sum1.imag();
  __syncthreads();
  for (int width = NT / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) {
      partial0[threadIdx.x].x += partial0[threadIdx.x + width].x;
      partial0[threadIdx.x].y += partial0[threadIdx.x + width].y;
      partial1[threadIdx.x].x += partial1[threadIdx.x + width].x;
      partial1[threadIdx.x].y += partial1[threadIdx.x + width].y;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out[0] = LatticeComplex<T>(partial0[0].x, partial0[0].y);
    out[1] = LatticeComplex<T>(partial1[0].x, partial1[0].y);
  }
}

template <typename T>
__device__ inline bool strict_scalar_bad(T real, T imag) {
  return !((real == real) && (imag == imag) &&
           fabs(real) != INFINITY && fabs(imag) != INFINITY);
}

template <typename T>
__global__ void strict_mr_give_alpha_kernel(
    void *device_vals, T floor, bool squared_denominator_norm) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  LatticeComplex<T> *values =
      static_cast<LatticeComplex<T> *>(device_vals);
  const LatticeComplex<T> numerator = values[_tmp0_];
  const LatticeComplex<T> denominator = values[_tmp1_];
  const T denominator_abs2 =
      denominator.real() * denominator.real() +
      denominator.imag() * denominator.imag();
  const T denominator_norm =
      squared_denominator_norm ? denominator_abs2 : sqrt(denominator_abs2);
  if (strict_scalar_bad<T>(numerator.real(), numerator.imag()) ||
      strict_scalar_bad<T>(denominator.real(), denominator.imag()) ||
      !(denominator_norm > floor)) {
    values[_alpha_] = LatticeComplex<T>((T)0, (T)0);
    return;
  }
  const LatticeComplex<T> alpha = numerator / denominator;
  values[_alpha_] = strict_scalar_bad<T>(alpha.real(), alpha.imag())
                        ? LatticeComplex<T>((T)0, (T)0)
                        : alpha;
}

template <typename T>
__global__ void strict_mr_update_device_kernel(
    void *x_ptr, void *r_ptr, const void *Ar_ptr,
    const void *device_vals, int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) return;
  const LatticeComplex<T> alpha =
      static_cast<const LatticeComplex<T> *>(device_vals)[_alpha_];
  LatticeComplex<T> *x = static_cast<LatticeComplex<T> *>(x_ptr);
  LatticeComplex<T> *r = static_cast<LatticeComplex<T> *>(r_ptr);
  const LatticeComplex<T> *Ar =
      static_cast<const LatticeComplex<T> *>(Ar_ptr);
  const LatticeComplex<T> old_r = r[index];
  x[index] += alpha * old_r;
  r[index] = old_r - alpha * Ar[index];
}

template <typename T, int NT>
__global__ void strict_dot_many_partial_kernel(
    const LatticeComplex<T> *basis, size_t basis_stride,
    const LatticeComplex<T> *right, int n, int count,
    int partial_count, LatticeComplex<T> *partials) {
  using complex_data = typename LatticeComplex<T>::_data_type;
  const int column = static_cast<int>(blockIdx.x);
  const int block = static_cast<int>(blockIdx.y);
  if (column >= count || block >= partial_count) return;
  __shared__ complex_data partial[NT];
  const LatticeComplex<T> *left = basis +
      static_cast<size_t>(column) * basis_stride;
  const int index = block * NT + threadIdx.x;
  const int stride = partial_count * NT;
  LatticeComplex<T> sum((T)0, (T)0);
  for (int i = index; i < n; i += stride)
    sum += left[i].conj() * right[i];
  partial[threadIdx.x].x = sum.real();
  partial[threadIdx.x].y = sum.imag();
  __syncthreads();
  for (int width = NT / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) {
      partial[threadIdx.x].x += partial[threadIdx.x + width].x;
      partial[threadIdx.x].y += partial[threadIdx.x + width].y;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    partials[column * partial_count + block] =
        LatticeComplex<T>(partial[0].x, partial[0].y);
}

template <typename T, int NT>
__global__ void strict_dot_many_reduce_kernel(
    const LatticeComplex<T> *partials, int partial_count, int count,
    LatticeComplex<T> *out) {
  using complex_data = typename LatticeComplex<T>::_data_type;
  const int column = static_cast<int>(blockIdx.x);
  if (column >= count) return;
  __shared__ complex_data partial[NT];
  LatticeComplex<T> sum((T)0, (T)0);
  const LatticeComplex<T> *input =
      partials + static_cast<size_t>(column) * partial_count;
  for (int i = threadIdx.x; i < partial_count; i += NT)
    sum += input[i];
  partial[threadIdx.x].x = sum.real();
  partial[threadIdx.x].y = sum.imag();
  __syncthreads();
  for (int width = NT / 2; width > 0; width >>= 1) {
    if (threadIdx.x < width) {
      partial[threadIdx.x].x += partial[threadIdx.x + width].x;
      partial[threadIdx.x].y += partial[threadIdx.x + width].y;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    out[column] = LatticeComplex<T>(partial[0].x, partial[0].y);
}

// Apply one Arnoldi orthogonalisation column in a single pass.  The dot
// coefficients are already resident in the first `count` entries of the
// temporary V[dot_count] slice, so this removes one cuBLAS launch per basis
// vector without adding storage or a host-to-device scalar copy.  The loop
// order matches the former sequential axpy order: w <- w - h_i V_i.
template <typename T>
__global__ void strict_orthogonalize_kernel(
    LatticeComplex<T> *vector, const LatticeComplex<T> *basis,
    size_t basis_stride, const LatticeComplex<T> *coefficients, int count,
    int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) return;
  LatticeComplex<T> value = vector[index];
  for (int column = 0; column < count; ++column)
    value -= coefficients[column] *
             basis[static_cast<size_t>(column) * basis_stride + index];
  vector[index] = value;
}

template <int NT = 256>
inline int strict_reduction_blocks(size_t n, int cap = 1024) {
  if (n == 0) return 1;
  if (cap < 1) cap = 1;
  const size_t items_per_block = static_cast<size_t>(NT) * 8;
  size_t blocks = (n + items_per_block - 1) / items_per_block;
  blocks = std::max<size_t>(1, std::min<size_t>(blocks, cap));
  if (blocks > static_cast<size_t>(std::numeric_limits<int>::max()))
    return std::numeric_limits<int>::max();
  return static_cast<int>(blocks);
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

inline int strict_world_size() {
  int initialized = 0;
  if (MPI_Initialized(&initialized) != MPI_SUCCESS)
    throw std::runtime_error("strict MPI cannot query initialization");
  if (!initialized) return 1;
  int finalized = 0;
  if (MPI_Finalized(&finalized) != MPI_SUCCESS)
    throw std::runtime_error("strict MPI cannot query finalization");
  if (finalized)
    throw std::runtime_error("strict MPI is already finalized");
  int size = 1;
  if (MPI_Comm_size(MPI_COMM_WORLD, &size) != MPI_SUCCESS)
    throw std::runtime_error("strict MPI cannot query communicator size");
  return size;
}

template <typename T>
inline int strict_dim_extent(const LatticeSet<T> *set, int dim) {
  switch (dim) {
    case 0: return set->host_params[_GRID_X_];
    case 1: return set->host_params[_GRID_Y_];
    case 2: return set->host_params[_GRID_Z_];
    case 3: return set->host_params[_GRID_T_];
    default: throw std::out_of_range("strict MPI dimension is outside [0,4)");
  }
}

template <typename T>
inline int strict_coordinate(const LatticeSet<T> *set, int dim) {
  switch (dim) {
    case 0: return set->grid_index_1dim[_X_];
    case 1: return set->grid_index_1dim[_Y_];
    case 2: return set->grid_index_1dim[_Z_];
    case 3: return set->grid_index_1dim[_T_];
    default: throw std::out_of_range("strict MPI dimension is outside [0,4)");
  }
}

template <typename T>
inline int strict_peer(
    const LatticeSet<T> *set, int dim, int displacement) {
  const int size = strict_dim_extent(set, dim);
  if (size <= 1) return set->host_params[_NODE_RANK_];
  int coordinate = strict_coordinate(set, dim) + displacement;
  if (coordinate < 0) coordinate += size;
  if (coordinate >= size) coordinate -= size;
  const int grid[4] = {set->host_params[_GRID_X_],
                       set->host_params[_GRID_Y_],
                       set->host_params[_GRID_Z_],
                       set->host_params[_GRID_T_]};
  int rank_coordinate[4] = {
      set->grid_index_1dim[_X_], set->grid_index_1dim[_Y_],
      set->grid_index_1dim[_Z_], set->grid_index_1dim[_T_]};
  rank_coordinate[dim] = coordinate;
  return ((rank_coordinate[0] * grid[1] + rank_coordinate[1]) * grid[2] +
          rank_coordinate[2]) * grid[3] + rank_coordinate[3];
}

inline size_t strict_max_face(
    int X, int Y, int Z, int Lt) {
  size_t result = 0;
  for (int dim = 0; dim < 4; ++dim)
    result = std::max(result, strict_face_count(X, Y, Z, Lt, dim));
  return std::max<size_t>(result, 1);
}

inline size_t strict_checked_real_count(
    size_t complex_count, const char *where) {
  if (complex_count >
      static_cast<size_t>(std::numeric_limits<int>::max()) / 2)
    throw std::overflow_error(where);
  return 2 * complex_count;
}

template <typename T> class StrictVectorHalo {
 public:
  StrictVectorHalo(LatticeSet<T> *set, int components,
                   int X, int Y, int Z, int Lt, bool compact,
                   int tag_base)
      : set_(set), components_(components), X_(X), Y_(Y), Z_(Z), Lt_(Lt),
        compact_(compact), tag_base_(tag_base),
        max_face_(strict_max_face(X, Y, Z, Lt)),
        slot_elements_(static_cast<size_t>(components) * max_face_),
        bytes_(8 * slot_elements_ * sizeof(LatticeComplex<T>)) {
    if (set_ == nullptr || components_ <= 0)
      throw std::invalid_argument("strict vector halo descriptor is invalid");
    distributed_ = false;
    for (int dim = 0; dim < 4; ++dim)
      distributed_ = distributed_ || strict_dim_extent(set_, dim) > 1;
    if (!distributed_) return;
    strict_check_cuda(cudaMalloc(&device_send_, bytes_),
                      "strict vector halo send allocation");
    strict_check_cuda(cudaMalloc(&device_recv_, bytes_),
                      "strict vector halo receive allocation");
    strict_check_cuda(cudaMallocHost(&host_send_, bytes_),
                      "strict vector halo pinned send allocation");
    strict_check_cuda(cudaMallocHost(&host_recv_, bytes_),
                      "strict vector halo pinned receive allocation");
  }

  ~StrictVectorHalo() { release(); }
  StrictVectorHalo(const StrictVectorHalo &) = delete;
  StrictVectorHalo &operator=(const StrictVectorHalo &) = delete;

  bool distributed() const { return distributed_; }
  const void *device_ghost() const { return device_recv_; }
  size_t slot_stride() const { return max_face_; }
  size_t bytes() const { return distributed_ ? 4 * bytes_ : 0; }

  void exchange(const void *input, int parity) {
    if (!distributed_) return;
    if (input == nullptr)
      throw std::invalid_argument("strict vector halo input is null");
    strict_check_cuda(cudaStreamSynchronize(set_->stream),
                      "strict vector halo input sync");
    (void)cudaGetLastError();
    for (int dim = 0; dim < 4; ++dim) {
      if (strict_dim_extent(set_, dim) <= 1) continue;
      for (int side = 0; side < 2; ++side) {
        void *slot =
            static_cast<char *>(device_send_) +
            (2 * dim + side) * slot_elements_ * sizeof(LatticeComplex<T>);
        if (compact_) {
          strict_check_cuda(
              cudaMemsetAsync(
                  slot, 0, slot_elements_ * sizeof(LatticeComplex<T>),
                  set_->stream),
              "strict compact vector halo slot zero");
            strict_launch_pack_compact_faces<T>(
                slot, input, components_, X_, Y_, Z_, Lt_,
                dim, side, parity, set_->stream,
                static_cast<int>(slot_stride()));
        } else {
          strict_launch_pack_face<T>(
              false, slot, input, components_, X_, Y_, Z_, Lt_,
              dim, side, parity, set_->stream,
              static_cast<int>(slot_stride()));
        }
      }
    }
    strict_check_cuda(cudaGetLastError(), "strict vector halo pack launch");
    const bool use_device_mpi =
        qcu_mpi_can_use_buffer<T>(device_send_);
    if (!use_device_mpi) {
      strict_check_cuda(
          cudaMemcpyAsync(host_send_, device_send_, bytes_,
                          cudaMemcpyDeviceToHost, set_->stream),
          "strict vector halo send staging");
    }
    strict_check_cuda(cudaStreamSynchronize(set_->stream),
                      "strict vector halo pack sync");

    for (int dim = 0; dim < 4; ++dim) {
      if (strict_dim_extent(set_, dim) <= 1) continue;
      // Slots are padded to the largest face so that a single ghost layout
      // serves every dimension; both peers therefore exchange the padded
      // span, not just the live ``components * face_count`` prefix.
      const size_t complex_count =
          static_cast<size_t>(components_) * slot_stride();
      const int count = static_cast<int>(
          strict_checked_real_count(complex_count,
                                    "strict vector halo message is too large"));
      const int backward = strict_peer(set_, dim, -1);
      const int forward = strict_peer(set_, dim, 1);
      void *send = use_device_mpi ? device_send_ : host_send_;
      void *recv = use_device_mpi ? device_recv_ : host_recv_;
      const size_t min_offset =
          static_cast<size_t>(2 * dim) * slot_elements_;
      const size_t max_offset =
          static_cast<size_t>(2 * dim + 1) * slot_elements_;
      checkMpiErrors(_MPI_Sendrecv<T>(
          static_cast<const LatticeComplex<T> *>(send) + max_offset, count,
          forward, tag_base_ + 2 * dim,
          static_cast<LatticeComplex<T> *>(recv) + max_offset, count,
          backward, tag_base_ + 2 * dim,
          MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      checkMpiErrors(_MPI_Sendrecv<T>(
          static_cast<const LatticeComplex<T> *>(send) + min_offset, count,
          backward, tag_base_ + 2 * dim + 1,
          static_cast<LatticeComplex<T> *>(recv) + min_offset, count,
          forward, tag_base_ + 2 * dim + 1,
          MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }
    if (!use_device_mpi) {
      strict_check_cuda(
          cudaMemcpyAsync(device_recv_, host_recv_, bytes_,
                          cudaMemcpyHostToDevice, set_->stream),
          "strict vector halo receive staging");
    }
    strict_check_cuda(cudaStreamSynchronize(set_->stream),
                      "strict vector halo exchange sync");
  }

 private:
  void release() noexcept {
    if (device_send_ != nullptr) {
      (void)cudaFree(device_send_);
      device_send_ = nullptr;
    }
    if (device_recv_ != nullptr) {
      (void)cudaFree(device_recv_);
      device_recv_ = nullptr;
    }
    if (host_send_ != nullptr) {
      (void)cudaFreeHost(host_send_);
      host_send_ = nullptr;
    }
    if (host_recv_ != nullptr) {
      (void)cudaFreeHost(host_recv_);
      host_recv_ = nullptr;
    }
  }

  LatticeSet<T> *set_ = nullptr;
  int components_, X_, Y_, Z_, Lt_, tag_base_;
  bool compact_ = false;
  bool distributed_ = false;
  size_t max_face_ = 1;
  size_t slot_elements_ = 0;
  size_t bytes_ = 0;
  void *device_send_ = nullptr;
  void *device_recv_ = nullptr;
  void *host_send_ = nullptr;
  void *host_recv_ = nullptr;
};

template <typename T> class StrictAxisLinkHalo {
 public:
  StrictAxisLinkHalo(LatticeSet<T> *set, int components,
                     int X, int Y, int Z, int Lt, bool compact,
                     int tag_base)
      : set_(set), components_(components), X_(X), Y_(Y), Z_(Z), Lt_(Lt),
        compact_(compact), tag_base_(tag_base),
        max_face_(strict_max_face(X, Y, Z, Lt)),
        elements_(static_cast<size_t>(components) * max_face_),
        bytes_(elements_ * sizeof(LatticeComplex<T>)) {
    if (set_ == nullptr || components_ <= 0)
      throw std::invalid_argument("strict link halo descriptor is invalid");
    distributed_ = false;
    for (int dim = 0; dim < 4; ++dim)
      distributed_ = distributed_ || strict_dim_extent(set_, dim) > 1;
    if (!distributed_) return;
    strict_check_cuda(cudaMalloc(&device_send_, bytes_),
                      "strict link halo send allocation");
    strict_check_cuda(cudaMalloc(&device_recv_, bytes_),
                      "strict link halo receive allocation");
    strict_check_cuda(cudaMallocHost(&host_send_, bytes_),
                      "strict link halo pinned send allocation");
    strict_check_cuda(cudaMallocHost(&host_recv_, bytes_),
                      "strict link halo pinned receive allocation");
  }

  ~StrictAxisLinkHalo() { release(); }
  StrictAxisLinkHalo(const StrictAxisLinkHalo &) = delete;
  StrictAxisLinkHalo &operator=(const StrictAxisLinkHalo &) = delete;

  bool distributed() const { return distributed_; }
  const void *device_ghost() const { return device_recv_; }
  size_t bytes() const { return distributed_ ? 4 * bytes_ : 0; }

  void exchange(const void *input, int parity, int dim) {
    if (!distributed_ || strict_dim_extent(set_, dim) <= 1) return;
    if (input == nullptr)
      throw std::invalid_argument("strict link halo input is null");
    const size_t face_count = strict_face_count(X_, Y_, Z_, Lt_, dim);
    const size_t complex_count =
        static_cast<size_t>(components_) * face_count;
    strict_check_cuda(cudaStreamSynchronize(set_->stream),
                      "strict link halo input sync");
    (void)cudaGetLastError();
    if (compact_) {
      strict_check_cuda(
          cudaMemsetAsync(device_send_, 0, bytes_, set_->stream),
          "strict compact link halo zero");
      strict_launch_pack_compact_faces<T>(
          device_send_, input, components_, X_, Y_, Z_, Lt_,
          dim, 1, parity, set_->stream, static_cast<int>(face_count));
    } else {
      strict_launch_pack_face<T>(
          false, device_send_, input, components_, X_, Y_, Z_, Lt_,
          dim, 1, parity, set_->stream, static_cast<int>(face_count));
    }
    strict_check_cuda(cudaGetLastError(), "strict link halo pack launch");
    const bool use_device_mpi =
        qcu_mpi_can_use_buffer<T>(device_send_);
    if (!use_device_mpi) {
      strict_check_cuda(
          cudaMemcpyAsync(host_send_, device_send_, bytes_,
                          cudaMemcpyDeviceToHost, set_->stream),
          "strict link halo send staging");
    }
    strict_check_cuda(cudaStreamSynchronize(set_->stream),
                      "strict link halo pack sync");

    const int count = static_cast<int>(
        strict_checked_real_count(complex_count,
                                  "strict link halo message is too large"));
    const int backward = strict_peer(set_, dim, -1);
    const int forward = strict_peer(set_, dim, 1);
    void *send = use_device_mpi ? device_send_ : host_send_;
    void *recv = use_device_mpi ? device_recv_ : host_recv_;
    checkMpiErrors(_MPI_Sendrecv<T>(
        send, count, forward, tag_base_ + dim,
        recv, count, backward, tag_base_ + dim,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    if (!use_device_mpi) {
      strict_check_cuda(
          cudaMemcpyAsync(device_recv_, host_recv_, bytes_,
                          cudaMemcpyHostToDevice, set_->stream),
          "strict link halo receive staging");
    }
    strict_check_cuda(cudaStreamSynchronize(set_->stream),
                      "strict link halo exchange sync");
  }

 private:
  void release() noexcept {
    if (device_send_ != nullptr) {
      (void)cudaFree(device_send_);
      device_send_ = nullptr;
    }
    if (device_recv_ != nullptr) {
      (void)cudaFree(device_recv_);
      device_recv_ = nullptr;
    }
    if (host_send_ != nullptr) {
      (void)cudaFreeHost(host_send_);
      host_send_ = nullptr;
    }
    if (host_recv_ != nullptr) {
      (void)cudaFreeHost(host_recv_);
      host_recv_ = nullptr;
    }
  }

  LatticeSet<T> *set_ = nullptr;
  int components_, X_, Y_, Z_, Lt_, tag_base_;
  bool compact_ = false;
  bool distributed_ = false;
  size_t max_face_ = 1;
  size_t elements_ = 0;
  size_t bytes_ = 0;
  void *device_send_ = nullptr;
  void *device_recv_ = nullptr;
  void *host_send_ = nullptr;
  void *host_recv_ = nullptr;
};

template <typename T>
__global__ void strict_local_to_global_kernel(
    void *global_ptr, const void *blocks_ptr, int components,
    int local_X, int local_Y, int local_Z, int local_T,
    int global_X, int global_Y, int global_Z, int global_T,
    int grid_x, int grid_y, int grid_z, int grid_t,
    int parity, size_t local_sites, size_t global_sites, int world) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t block_elements =
      static_cast<size_t>(components) * local_sites;
  if (index >= block_elements * world) return;
  LatticeComplex<T> *global = static_cast<LatticeComplex<T> *>(global_ptr);
  const LatticeComplex<T> *blocks =
      static_cast<const LatticeComplex<T> *>(blocks_ptr);
  const int block_rank = static_cast<int>(index / block_elements);
  const size_t block_index = index - static_cast<size_t>(block_rank) *
                                         block_elements;
  const int component = static_cast<int>(block_index / local_sites);
  const int site = static_cast<int>(block_index -
                                    static_cast<size_t>(component) *
                                        local_sites);
  int block_coordinate = block_rank;
  const int block_x = block_coordinate / (grid_y * grid_z * grid_t);
  block_coordinate -= block_x * grid_y * grid_z * grid_t;
  const int block_y = block_coordinate / (grid_z * grid_t);
  block_coordinate -= block_y * grid_z * grid_t;
  const int block_z = block_coordinate / grid_t;
  const int block_t = block_coordinate - block_z * grid_t;
  const int origin_x = block_x * local_X;
  const int origin_y = block_y * local_Y;
  const int origin_z = block_z * local_Z;
  const int origin_t = block_t * local_T;
  int x, y, z, t;
  strict_decode_half_site(
      site, parity, local_X, local_Y, local_Z, local_T, x, y, z, t);
  const int global_x = origin_x + x;
  const int global_y = origin_y + y;
  const int global_z = origin_z + z;
  const int global_t = origin_t + t;
  const int global_site =
      ((global_x * global_Y + global_y) * global_Z + global_z) *
          (global_T / 2) +
      global_t / 2;
  global[static_cast<size_t>(component) * global_sites + global_site] =
      blocks[static_cast<size_t>(block_rank) * block_elements + block_index];
}

template <typename T>
__global__ void strict_global_to_local_kernel(
    void *local_ptr, const void *global_ptr, int components,
    int local_X, int local_Y, int local_Z, int local_T,
    int global_X, int global_Y, int global_Z, int global_T,
    int origin_x, int origin_y, int origin_z, int origin_t,
    int parity, size_t local_sites, size_t global_sites) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= static_cast<size_t>(components) * local_sites) return;
  LatticeComplex<T> *local = static_cast<LatticeComplex<T> *>(local_ptr);
  const LatticeComplex<T> *global =
      static_cast<const LatticeComplex<T> *>(global_ptr);
  const int component = static_cast<int>(index / local_sites);
  const int site = static_cast<int>(index - component * local_sites);
  int x, y, z, t;
  strict_decode_half_site(
      site, parity, local_X, local_Y, local_Z, local_T, x, y, z, t);
  const int global_x = origin_x + x;
  const int global_y = origin_y + y;
  const int global_z = origin_z + z;
  const int global_t = origin_t + t;
  const int global_site =
      ((global_x * global_Y + global_y) * global_Z + global_z) *
          (global_T / 2) +
      global_t / 2;
  local[index] =
      global[static_cast<size_t>(component) * global_sites + global_site];
}

__global__ void strict_gauge_block_to_global_kernel(
    void *global_ptr, const void *blocks_ptr,
    int local_X, int local_Y, int local_Z, int local_T,
    int global_X, int global_Y, int global_Z, int global_T,
    int grid_x, int grid_y, int grid_z, int grid_t,
    size_t local_sites, size_t global_sites, int world, int data_type) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t block_elements = static_cast<size_t>(72) * local_sites;
  const size_t total = block_elements * world;
  if (index >= total) return;
  const int block_rank = static_cast<int>(index / block_elements);
  const size_t block_index = index - static_cast<size_t>(block_rank) *
                                         block_elements;
  const int component = static_cast<int>(block_index / local_sites);
  const int site = static_cast<int>(block_index -
                                    static_cast<size_t>(component) *
                                        local_sites);
  int block_coordinate = block_rank;
  const int block_x = block_coordinate / (grid_y * grid_z * grid_t);
  block_coordinate -= block_x * grid_y * grid_z * grid_t;
  const int block_y = block_coordinate / (grid_z * grid_t);
  block_coordinate -= block_y * grid_z * grid_t;
  const int block_z = block_coordinate / grid_t;
  const int block_t = block_coordinate - block_z * grid_t;
  const int origin_x = block_x * local_X;
  const int origin_y = block_y * local_Y;
  const int origin_z = block_z * local_Z;
  const int origin_t = block_t * local_T;
  const int parity = component / 36;
  int x, y, z, t;
  if (data_type == _LAT_C64_) {
    float *global = static_cast<float *>(global_ptr);
    const float *blocks = static_cast<const float *>(blocks_ptr);
    int xx, yy, zz, tt;
    strict_decode_half_site(
        site, parity, local_X, local_Y, local_Z, local_T, xx, yy, zz, tt);
    const int global_x = origin_x + xx;
    const int global_y = origin_y + yy;
    const int global_z = origin_z + zz;
    const int global_t = origin_t + tt;
    const int global_site =
        ((global_x * global_Y + global_y) * global_Z + global_z) *
            (global_T / 2) +
        global_t / 2;
    const size_t global_offset =
        (static_cast<size_t>(component) * global_sites + global_site) * 2;
    const size_t block_offset =
        static_cast<size_t>(block_rank) * block_elements + block_index;
    global[global_offset] = blocks[block_offset * 2];
    global[global_offset + 1] = blocks[block_offset * 2 + 1];
  } else if (data_type == _LAT_C128_) {
    double *global = static_cast<double *>(global_ptr);
    const double *blocks = static_cast<const double *>(blocks_ptr);
    int xx, yy, zz, tt;
    strict_decode_half_site(
        site, parity, local_X, local_Y, local_Z, local_T, xx, yy, zz, tt);
    const int global_x = origin_x + xx;
    const int global_y = origin_y + yy;
    const int global_z = origin_z + zz;
    const int global_t = origin_t + tt;
    const int global_site =
        ((global_x * global_Y + global_y) * global_Z + global_z) *
            (global_T / 2) +
        global_t / 2;
    const size_t global_offset =
        (static_cast<size_t>(component) * global_sites + global_site) * 2;
    const size_t block_offset =
        static_cast<size_t>(block_rank) * block_elements + block_index;
    global[global_offset] = blocks[block_offset * 2];
    global[global_offset + 1] = blocks[block_offset * 2 + 1];
  }
}

template <typename T>
class StrictFineGlobalDslash {
 public:
  StrictFineGlobalDslash(
      LatticeSet<T> *set, void *local_gauge,
      int local_X, int local_Y, int local_Z, int local_T)
      : set_(set), local_X_(local_X), local_Y_(local_Y), local_Z_(local_Z),
        local_T_(local_T),
        local_sites_(static_cast<size_t>(local_X) * local_Y * local_Z *
                     (local_T / 2)),
        global_X_(local_X * strict_dim_extent(set, 0)),
        global_Y_(local_Y * strict_dim_extent(set, 1)),
        global_Z_(local_Z * strict_dim_extent(set, 2)),
        global_T_(local_T * strict_dim_extent(set, 3)),
        global_sites_(static_cast<size_t>(global_X_) * global_Y_ * global_Z_ *
                      (global_T_ / 2)),
        world_(strict_world_size()) {
    if (world_ < 1 || local_gauge == nullptr)
      throw std::invalid_argument(
          "strict fine global dslash requires a valid communicator");
    const size_t field_local_bytes =
        static_cast<size_t>(12) * local_sites_ * sizeof(LatticeComplex<T>);
    const size_t gauge_local_bytes =
        static_cast<size_t>(72) * local_sites_ * sizeof(LatticeComplex<T>);
    const size_t field_all_bytes = field_local_bytes * world_;
    const size_t gauge_all_bytes = gauge_local_bytes * world_;
    strict_check_cuda(cudaMallocAsync(
                          &global_gauge_,
                          72 * global_sites_ * sizeof(LatticeComplex<T>),
                          set_->stream),
                      "strict fine global gauge allocation");
    strict_check_cuda(cudaMallocAsync(
                          &global_field_in_,
                          12 * global_sites_ * sizeof(LatticeComplex<T>),
                          set_->stream),
                      "strict fine global input allocation");
    strict_check_cuda(cudaMallocAsync(
                          &global_field_out_,
                          12 * global_sites_ * sizeof(LatticeComplex<T>),
                          set_->stream),
                      "strict fine global output allocation");
    strict_check_cuda(cudaMallocHost(&host_field_local_, field_all_bytes),
                      "strict fine global field host allocation");
    strict_check_cuda(cudaMallocHost(&host_field_send_, field_local_bytes),
                      "strict fine global field send allocation");
    strict_check_cuda(cudaMallocHost(&host_gauge_local_, gauge_all_bytes),
                      "strict fine global gauge host allocation");
    strict_check_cuda(cudaMallocHost(&host_gauge_send_, gauge_local_bytes),
                      "strict fine global gauge send allocation");
    std::vector<int> params(_PARAMS_SIZE_, 0);
    std::copy(set_->host_params, set_->host_params + _PARAMS_SIZE_,
              params.begin());
    params[_LAT_X_] = global_X_;
    params[_LAT_Y_] = global_Y_;
    params[_LAT_Z_] = global_Z_;
    params[_LAT_T_] = global_T_ / 2;
    params[_LAT_XYZT_] = static_cast<int>(global_X_ * global_Y_ *
                                          global_Z_ * (global_T_ / 2));
    params[_DAGGER_] = _NO_USE_;
    strict_check_cuda(cudaMallocAsync(&device_params_,
                                      _PARAMS_SIZE_ * sizeof(int),
                                      set_->stream),
                      "strict fine global params allocation");
    strict_check_cuda(cudaMemcpyAsync(
                          device_params_, params.data(),
                          _PARAMS_SIZE_ * sizeof(int), cudaMemcpyHostToDevice,
                          set_->stream),
                      "strict fine global params copy");
    // Gauge is static for the hierarchy lifetime.  Gather raw local blocks,
    // then scatter them into the global parity-split layout.
    strict_check_cuda(cudaMemcpyAsync(
                          host_gauge_send_, local_gauge, gauge_local_bytes,
                          cudaMemcpyDeviceToHost, set_->stream),
                      "strict fine global gauge staging");
    strict_check_cuda(cudaStreamSynchronize(set_->stream),
                      "strict fine global gauge sync");
    const int real_count = static_cast<int>(
        strict_checked_real_count(72 * local_sites_,
                                  "strict fine global gauge gather overflow"));
    checkMpiErrors(MPI_Allgather(
        host_gauge_send_, real_count,
        std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE,
        host_gauge_local_, real_count,
        std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE,
        MPI_COMM_WORLD));
    const int origin_x = set_->grid_index_1dim[_X_] * local_X_;
    const int origin_y = set_->grid_index_1dim[_Y_] * local_Y_;
    const int origin_z = set_->grid_index_1dim[_Z_] * local_Z_;
    const int origin_t = set_->grid_index_1dim[_T_] * local_T_;
    const size_t gauge_elements = 72 * local_sites_ * world_;
    strict_gauge_block_to_global_kernel<<<
        static_cast<int>((gauge_elements + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_),
        _BLOCK_SIZE_, 0, set_->stream>>>(
        global_gauge_, host_gauge_local_, local_X_, local_Y_, local_Z_,
        local_T_, global_X_, global_Y_, global_Z_, global_T_,
        strict_dim_extent(set_, 0), strict_dim_extent(set_, 1),
        strict_dim_extent(set_, 2), strict_dim_extent(set_, 3),
        local_sites_, global_sites_,
        world_, set_->host_params[_DATA_TYPE_]);
    strict_check_cuda(cudaGetLastError(),
                      "strict fine global gauge scatter launch");
    strict_check_cuda(cudaStreamSynchronize(set_->stream),
                      "strict fine global gauge scatter sync");
    last_gauge_ = local_gauge;
  }

  ~StrictFineGlobalDslash() { release(); }
  StrictFineGlobalDslash(const StrictFineGlobalDslash &) = delete;
  StrictFineGlobalDslash &operator=(const StrictFineGlobalDslash &) = delete;

  bool matches(void *local_gauge) const { return local_gauge == last_gauge_; }

  void apply(void *out, const void *in, int target_parity) {
    const int source_parity = 1 - target_parity;
    const size_t field_bytes =
        static_cast<size_t>(12) * local_sites_ * sizeof(LatticeComplex<T>);
    const size_t all_bytes = field_bytes * world_;
    strict_check_cuda(cudaMemcpyAsync(
                          host_field_send_, in, field_bytes,
                          cudaMemcpyDeviceToHost, set_->stream),
                      "strict fine global field staging");
    strict_check_cuda(cudaStreamSynchronize(set_->stream),
                      "strict fine global field staging sync");
    const int real_count = static_cast<int>(
        strict_checked_real_count(12 * local_sites_,
                                  "strict fine global field gather overflow"));
    checkMpiErrors(MPI_Allgather(
        host_field_send_, real_count,
        std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE,
        host_field_local_, real_count,
        std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE,
        MPI_COMM_WORLD));
    const int origin_x = set_->grid_index_1dim[_X_] * local_X_;
    const int origin_y = set_->grid_index_1dim[_Y_] * local_Y_;
    const int origin_z = set_->grid_index_1dim[_Z_] * local_Z_;
    const int origin_t = set_->grid_index_1dim[_T_] * local_T_;
    const size_t local_elements = 12 * local_sites_ * world_;
    strict_local_to_global_kernel<T><<<
        static_cast<int>((local_elements + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_),
        _BLOCK_SIZE_, 0, set_->stream>>>(
        global_field_in_, host_field_local_, 12,
        local_X_, local_Y_, local_Z_, local_T_,
        global_X_, global_Y_, global_Z_, global_T_,
        strict_dim_extent(set_, 0), strict_dim_extent(set_, 1),
        strict_dim_extent(set_, 2), strict_dim_extent(set_, 3),
        source_parity,
        local_sites_, global_sites_, world_);
    strict_check_cuda(cudaGetLastError(),
                      "strict fine global input scatter launch");
    int parity = target_parity;
    strict_check_cuda(cudaMemcpyAsync(
                          static_cast<char *>(device_params_) +
                              _PARITY_ * sizeof(int),
                          &parity, sizeof(int), cudaMemcpyHostToDevice,
                          set_->stream),
                      "strict fine global parity copy");
    const int total = static_cast<int>(12 * global_sites_);
    const int blocks = (total + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_;
    wilson_dslash<T><<<blocks, _BLOCK_SIZE_, 0, set_->stream>>>(
        global_gauge_, global_field_in_, global_field_out_, device_params_);
    strict_check_cuda(cudaGetLastError(),
                      "strict fine global dslash launch");
    const size_t local_out_elements = 12 * local_sites_;
    strict_global_to_local_kernel<T><<<
        static_cast<int>((local_out_elements + _BLOCK_SIZE_ - 1) /
                         _BLOCK_SIZE_),
        _BLOCK_SIZE_, 0, set_->stream>>>(
        out, global_field_out_, 12,
        local_X_, local_Y_, local_Z_, local_T_,
        global_X_, global_Y_, global_Z_, global_T_,
        origin_x, origin_y, origin_z, origin_t, target_parity,
        local_sites_, global_sites_);
    strict_check_cuda(cudaGetLastError(),
                      "strict fine global output slice launch");
    strict_check_cuda(cudaStreamSynchronize(set_->stream),
                      "strict fine global dslash sync");
    (void)all_bytes;
  }

 private:
  void release() noexcept {
    if (set_ != nullptr) {
      if (global_gauge_ != nullptr) (void)cudaFreeAsync(global_gauge_, set_->stream);
      if (global_field_in_ != nullptr) (void)cudaFreeAsync(global_field_in_, set_->stream);
      if (global_field_out_ != nullptr) (void)cudaFreeAsync(global_field_out_, set_->stream);
      if (device_params_ != nullptr) (void)cudaFreeAsync(device_params_, set_->stream);
      (void)cudaStreamSynchronize(set_->stream);
    }
    if (host_field_local_ != nullptr) (void)cudaFreeHost(host_field_local_);
    if (host_field_send_ != nullptr) (void)cudaFreeHost(host_field_send_);
    if (host_gauge_local_ != nullptr) (void)cudaFreeHost(host_gauge_local_);
    if (host_gauge_send_ != nullptr) (void)cudaFreeHost(host_gauge_send_);
    global_gauge_ = global_field_in_ = global_field_out_ = nullptr;
    device_params_ = nullptr;
    host_field_local_ = host_gauge_local_ = nullptr;
    host_field_send_ = host_gauge_send_ = nullptr;
  }

  LatticeSet<T> *set_;
  int local_X_, local_Y_, local_Z_, local_T_;
  size_t local_sites_;
  int global_X_, global_Y_, global_Z_, global_T_;
  size_t global_sites_;
  int world_;
  void *global_gauge_ = nullptr;
  void *global_field_in_ = nullptr;
  void *global_field_out_ = nullptr;
  void *device_params_ = nullptr;
  void *host_field_local_ = nullptr;
  void *host_field_send_ = nullptr;
  void *host_gauge_local_ = nullptr;
  void *host_gauge_send_ = nullptr;
  void *last_gauge_ = nullptr;
};

// The strict coarse hopping kernel is the dominant kernel in the recursive
// V-cycle.  Keep its tuning local to this new path: legacy QCU kernels retain
// the project-wide production block size, so an A/B result cannot silently
// change unrelated Wilson/Clover operators.
constexpr int kStrictHoppingBlockSize = 256;

inline int strict_hopping_blocks(size_t n) {
  return static_cast<int>(
      (n + static_cast<size_t>(kStrictHoppingBlockSize) - 1) /
      static_cast<size_t>(kStrictHoppingBlockSize));
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

inline void strict_validate_mpi_backend(const int *params) {
  if (params == nullptr)
    throw std::invalid_argument("strict MPI gate received null params");

  int mpi_initialized = 0;
  if (MPI_Initialized(&mpi_initialized) != MPI_SUCCESS)
    throw std::runtime_error("strict MPI gate cannot query MPI state");
  int mpi_finalized = 0;
  if (mpi_initialized &&
      MPI_Finalized(&mpi_finalized) != MPI_SUCCESS)
    throw std::runtime_error("strict MPI gate cannot query MPI finalization");

  int world_size = 1;
  int world_rank = 0;
  if (mpi_initialized && !mpi_finalized) {
    if (MPI_Comm_size(MPI_COMM_WORLD, &world_size) != MPI_SUCCESS ||
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank) != MPI_SUCCESS)
      throw std::runtime_error("strict MPI gate cannot query MPI_COMM_WORLD");
  }

  if (mpi_finalized)
    throw std::invalid_argument(
        "strict MPI fail-closed: MPI_COMM_WORLD is already finalized");
  const int grid_x = params[_GRID_X_];
  const int grid_y = params[_GRID_Y_];
  const int grid_z = params[_GRID_Z_];
  const int grid_t = params[_GRID_T_];
  if (grid_x <= 0 || grid_y <= 0 || grid_z <= 0 || grid_t <= 0)
    throw std::invalid_argument(
        "strict MPI fail-closed: process-grid extents must be positive");
  const long long grid_size =
      static_cast<long long>(grid_x) * grid_y * grid_z * grid_t;
  if (grid_size != world_size || params[_NODE_SIZE_] != world_size ||
      params[_NODE_RANK_] != world_rank)
    throw std::invalid_argument(
        "strict MPI fail-closed: params NODE_RANK/NODE_SIZE/GRID do not "
        "match MPI_COMM_WORLD");
}

template <typename T>
LatticeComplex<T> strict_global_sum_complex(
    const LatticeComplex<T> &local, int *collective_calls = nullptr) {
  static_assert(std::is_same<T, float>::value ||
                    std::is_same<T, double>::value,
                "strict global reduction supports float/double only");

  int mpi_initialized = 0;
  if (MPI_Initialized(&mpi_initialized) != MPI_SUCCESS)
    throw std::runtime_error(
        "strict global reduction cannot query MPI state");
  if (!mpi_initialized)
    return local;

  int mpi_finalized = 0;
  if (MPI_Finalized(&mpi_finalized) != MPI_SUCCESS)
    throw std::runtime_error(
        "strict global reduction cannot query MPI finalization");
  if (mpi_finalized)
    throw std::runtime_error(
        "strict global reduction cannot run after MPI finalization");

  int world_size = 1;
  if (MPI_Comm_size(MPI_COMM_WORLD, &world_size) != MPI_SUCCESS)
    throw std::runtime_error(
        "strict global reduction cannot query MPI_COMM_WORLD");
  if (world_size == 1)
    return local;
  if (world_size < 1)
    throw std::runtime_error(
        "strict global reduction received invalid MPI_COMM_WORLD size");

  T values[2] = {local.real(), local.imag()};
  const MPI_Datatype scalar_type =
      std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE;
  if (MPI_Allreduce(MPI_IN_PLACE, values, 2, scalar_type, MPI_SUM,
                    MPI_COMM_WORLD) != MPI_SUCCESS)
    throw std::runtime_error("strict global reduction MPI_Allreduce failed");
  if (collective_calls != nullptr)
    ++(*collective_calls);
  return LatticeComplex<T>(values[0], values[1]);
}

template <typename T>
void strict_launch_coarse_with_halos(
    void *out, const void *in, const void *links, const void *onsite_pair,
    StrictVectorHalo<T> &vector_halo, StrictAxisLinkHalo<T> &link_halo,
    LatticeSet<T> *set, int E, int X, int Y, int Z, int Tdim,
    int onsite_index) {
  vector_halo.exchange(in, 0);
  const int total = E * X * Y * Z * Tdim;
  const int blocks = strict_hopping_blocks(total);
  strict_coarse_apply_kernel<T><<<blocks, kStrictHoppingBlockSize, 0,
                                  set->stream>>>(
      out, in, links, onsite_pair, vector_halo.device_ghost(),
      vector_halo.slot_stride(), E, X, Y, Z, Tdim,
      strict_dim_extent(set, 0), strict_dim_extent(set, 1),
      strict_dim_extent(set, 2), strict_dim_extent(set, 3), onsite_index);
  strict_check_cuda(cudaGetLastError(), "coarse apply base launch");
  const size_t volume =
      static_cast<size_t>(X) * Y * Z * Tdim;
  for (int dim = 0; dim < 4; ++dim) {
    if (strict_dim_extent(set, dim) <= 1) continue;
    const LatticeComplex<T> *backward_links =
        static_cast<const LatticeComplex<T> *>(links) +
        (static_cast<size_t>(1 * 4 + dim) * E * E) * volume;
    link_halo.exchange(backward_links, 0, dim);
    const int correction_blocks =
        (static_cast<int>(E * volume) + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_;
    const size_t ghost_slot_stride =
        vector_halo.slot_stride();
    const LatticeComplex<T> *input_ghost =
        static_cast<const LatticeComplex<T> *>(vector_halo.device_ghost()) +
        static_cast<size_t>(2 * dim + 1) * E * ghost_slot_stride;
    strict_coarse_apply_backward_correction_kernel<T>
        <<<correction_blocks, _BLOCK_SIZE_, 0, set->stream>>>(
            out, input_ghost, link_halo.device_ghost(),
            E, X, Y, Z, Tdim, dim);
    strict_check_cuda(cudaGetLastError(),
                      "coarse apply backward correction launch");
  }
}

template <typename T>
void strict_launch_hopping_with_halos(
    void *out, const void *in, const void *links, const void *base,
    StrictVectorHalo<T> &vector_halo, StrictAxisLinkHalo<T> &link_halo,
    LatticeSet<T> *set, int E, int X, int Y, int Z, int Lt,
    int target_parity) {
  const int input_parity = 1 - target_parity;
  vector_halo.exchange(in, input_parity);
  const int half_volume = E * X * Y * Z * (Lt / 2);
  const int blocks = strict_hopping_blocks(half_volume);
  strict_hopping_parity_base_kernel<T>
      <<<blocks, kStrictHoppingBlockSize, 0, set->stream>>>(
          out, in, links, base, vector_halo.device_ghost(),
          vector_halo.slot_stride(), E, X, Y, Z, Lt,
          strict_dim_extent(set, 0), strict_dim_extent(set, 1),
          strict_dim_extent(set, 2), strict_dim_extent(set, 3),
          target_parity);
  strict_check_cuda(cudaGetLastError(), "hopping parity base launch");
  const size_t volume =
      static_cast<size_t>(X) * Y * Z * Lt;
  for (int dim = 0; dim < 4; ++dim) {
    if (strict_dim_extent(set, dim) <= 1) continue;
    const LatticeComplex<T> *backward_links =
        static_cast<const LatticeComplex<T> *>(links) +
        (static_cast<size_t>(1 * 4 + dim) * E * E) * volume;
    link_halo.exchange(backward_links, input_parity, dim);
    const size_t ghost_slot_stride =
        vector_halo.slot_stride();
    const LatticeComplex<T> *input_ghost =
        static_cast<const LatticeComplex<T> *>(vector_halo.device_ghost()) +
        static_cast<size_t>(2 * dim + 1) * E * ghost_slot_stride;
    const int correction_blocks =
        (half_volume + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_;
    strict_hopping_parity_backward_correction_kernel<T>
        <<<correction_blocks, _BLOCK_SIZE_, 0, set->stream>>>(
            out, input_ghost, link_halo.device_ghost(),
            E, X, Y, Z, Lt, target_parity, dim, base == nullptr ? 0 : 1);
    strict_check_cuda(cudaGetLastError(),
                      "hopping parity backward correction launch");
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
  StrictVectorHalo<T> vector_halo(
      set, E, X, Y, Z, Tdim, false, 1100);
  StrictAxisLinkHalo<T> link_halo(
      set, E * E, X, Y, Z, Tdim, false, 1200);
  strict_launch_coarse_with_halos(
      out, in, links, onsite_pair, vector_halo, link_halo, set,
      E, X, Y, Z, Tdim, onsite_index);
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
  StrictVectorHalo<T> vector_halo(
      set, E, X, Y, Z, Tdim, true, 1300);
  // The preconditioned coarse links are stored on the full-site grid even
  // when the vectors are checkerboard packed: strict_hopping_parity_*_kernel
  // indexes ``links`` with strict_full_site().  Packing the link halo with the
  // compact decoder would therefore read only the first half of the array
  // with the wrong index map and silently corrupt the remote backward term.
  StrictAxisLinkHalo<T> link_halo(
      set, E * E, X, Y, Z, Tdim, false, 1400);
  strict_launch_hopping_with_halos(
      scratch, in, links, nullptr, vector_halo, link_halo, set,
      E, X, Y, Z, Tdim, 1 - parity);
  strict_launch_hopping_with_halos(
      out, scratch, links, in, vector_halo, link_halo, set,
      E, X, Y, Z, Tdim, parity);
  strict_check_cuda(cudaGetLastError(), "MATPC launch");
  strict_check_cuda(cudaStreamSynchronize(set->stream), "MATPC sync");
}

template <typename T>
void strict_launch_fine_matpc(
    void *out, const void *in, void *gauge, void *clover_ee,
    void *clover_oo, void *clover_ee_inv, void *clover_oo_inv,
    void *set_ptrs, int *params, int parity) {
  if (out == nullptr || in == nullptr || gauge == nullptr ||
      clover_ee == nullptr || clover_oo == nullptr ||
      clover_ee_inv == nullptr || clover_oo_inv == nullptr ||
      parity < 0 || parity > 1)
    throw std::invalid_argument("invalid strict fine MATPC descriptor");

  LatticeSet<T> *set = strict_get_set<T>(set_ptrs, params);
  const int n = set->lat_4dim_SC;
  if (n <= 0)
    throw std::invalid_argument("strict fine MATPC has empty compact field");

  LatticeCloverBistabCg<T> fine;
  fine.give(set);
  fine.init(gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv);
  std::unique_ptr<StrictFineGlobalDslash<T>> global_dslash;
  // Production default is the face-halo path: LatticeWilsonDslash::run_mpi
  // already exchanges the six rank-local faces of the compact 12-component
  // field and of the clover gauge, which keeps the communication volume
  // O(surface) instead of O(volume).  The all-gather replica dslash is kept
  // as a diagnostic A/B reference and must be requested explicitly.
  if (qcu_parse_bool_env("PYQCU_STRICT_GLOBAL_DSLASH", false))
    global_dslash.reset(new StrictFineGlobalDslash<T>(
        set, gauge, params[_LAT_X_], params[_LAT_Y_], params[_LAT_Z_],
        params[_LAT_T_]));

  // H_{q p}: source parity p -> eliminated parity q.  The Wilson helper
  // names its kernels by destination/source, so run_oe is 0 -> 1 and run_eo
  // is 1 -> 0.  Keep this mapping identical to the fused strict solver.
  if (global_dslash) {
    global_dslash->apply(set->device_vec0, in, 1 - parity);
  } else if (parity == 0) {
    fine.wilson_dslash.run_oe(set->device_vec0,
                              const_cast<void *>(in), fine.gauge);
  } else {
    fine.wilson_dslash.run_eo(set->device_vec0,
                              const_cast<void *>(in), fine.gauge);
  }
  if (parity == 0)
    fine.clover_dslash_oo_inv.give(set->device_vec0);
  else
    fine.clover_dslash_ee_inv.give(set->device_vec0);

  // H_{p q} A_q^{-1} H_{q p}; the second hopping returns to the target
  // parity, after which A_p^{-1} completes the symmetric MATPC action.
  if (global_dslash) {
    global_dslash->apply(set->device_vec1, set->device_vec0, parity);
  } else if (parity == 0) {
    fine.wilson_dslash.run_eo(set->device_vec1, set->device_vec0,
                              fine.gauge);
  } else {
    fine.wilson_dslash.run_oe(set->device_vec1, set->device_vec0,
                              fine.gauge);
  }
  if (parity == 0)
    fine.clover_dslash_ee_inv.give(set->device_vec1);
  else
    fine.clover_dslash_oo_inv.give(set->device_vec1);

  const int blocks = (n + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_;
  strict_check_cuda(cudaMemcpyAsync(
                        out, in,
                        static_cast<size_t>(n) * sizeof(LatticeComplex<T>),
                        cudaMemcpyDeviceToDevice, set->stream),
                    "fine MATPC identity copy");
  strict_fine_matpc_update_kernel<T>
      <<<blocks, _BLOCK_SIZE_, 0, set->stream>>>(
          out, out, set->device_vec1, set->kappa() * set->kappa(), n);
  strict_check_cuda(cudaGetLastError(), "fine MATPC launch");
  strict_check_cuda(cudaStreamSynchronize(set->stream), "fine MATPC sync");
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
  StrictVectorHalo<T> vector_halo(
      set, E, X, Y, Z, Lt, true, 1500);
  StrictAxisLinkHalo<T> link_halo(
      set, E * E, X, Y, Z, Lt, true, 1600);
  strict_launch_hopping_with_halos(
      out, scratch, links, out, vector_halo, link_halo, set,
      E, X, Y, Z, Lt, parity);
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
  const int half_blocks = (half_total + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_;
  const int hopping_blocks = strict_hopping_blocks(half_total);
  strict_onsite_full_to_parity_kernel<T>
      <<<half_blocks, _BLOCK_SIZE_, 0, set->stream>>>(
          scratch, full_rhs, onsite_pair, E, X, Y, Z, Lt, 1 - parity, 1);
  // x_q = X_q^-1 b_q - Hhat_qp x_p.
  StrictVectorHalo<T> vector_halo(
      set, E, X, Y, Z, Lt, true, 1700);
  StrictAxisLinkHalo<T> link_halo(
      set, E * E, X, Y, Z, Lt, true, 1800);
  strict_launch_hopping_with_halos(
      scratch, target_solution, links, scratch, vector_halo, link_halo,
      set, E, X, Y, Z, Lt, 1 - parity);
  (void)hopping_blocks;
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
  std::unique_ptr<StrictVectorHalo<T>> compact_vector_halo;
  std::unique_ptr<StrictAxisLinkHalo<T>> compact_link_halo;
  std::unique_ptr<StrictVectorHalo<T>> full_vector_halo;
  std::unique_ptr<StrictAxisLinkHalo<T>> full_link_halo;
};

template <typename T> struct StrictWorkspaceArena {
  void *storage = nullptr;
  void *r = nullptr, *v = nullptr, *tmp = nullptr;
  void *rhat = nullptr, *p = nullptr, *s = nullptr, *t = nullptr;
  void *correction_full = nullptr;
  void *dot_storage = nullptr;
  size_t dot_elements = 0;
  size_t bytes = 0;

  static size_t align_elements(size_t elements) {
    const size_t alignment = std::max(
        static_cast<size_t>(1),
        static_cast<size_t>(256) / sizeof(LatticeComplex<T>));
    return (elements + alignment - 1) / alignment * alignment;
  }

  void allocate(size_t max_compact, size_t coarsest_compact,
                size_t max_child_full, size_t max_reduction_n,
                cudaStream_t stream) {
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
    const size_t dot_blocks = static_cast<size_t>(
        strict_reduction_blocks(max_reduction_n));
    const size_t dot_offset = reserve(2 * dot_blocks);
    dot_elements = 2 * dot_blocks;
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
    dot_storage = base + dot_offset;
  }

  void release(cudaStream_t stream) {
    if (storage != nullptr) {
      strict_check_cuda(cudaFreeAsync(storage, stream),
                        "strict workspace release");
      storage = nullptr;
    }
  }
};

// Fine-grid restarted FGMRES storage.  The whole device arena is attached to
// the persistent slot-80 hierarchy and is reused by subsequent solves with
// the same geometry/restart.  Its exact device footprint is
//
//   (2 * restart + 5) * fine_compact + 2 * first_coarse_full
//
// complex numbers: V[m+1], Z[m], b/r/w/x and the two coarse transfer fields.
// Hessenberg/Givens data are small host-side vectors and never enter the
// per-iteration CUDA allocation path.
template <typename T> struct StrictOuterWorkspace {
  void *storage = nullptr;
  std::vector<void *> V;
  std::vector<void *> Z;
  void *b = nullptr, *r = nullptr, *w = nullptr, *x = nullptr;
  void *coarse_rhs = nullptr, *coarse_out = nullptr;
  std::vector<std::complex<T>> H, cs, sn, g, y;
  std::vector<LatticeComplex<T>> dot_values;
  size_t fine_n = 0, coarse_n = 0, bytes = 0;
  int restart = 0;

  static size_t checked_elements(size_t fine_count, size_t coarse_count,
                                 int restart_count) {
    if (restart_count <= 0)
      throw std::invalid_argument("strict FGMRES restart must be positive");
    const size_t vectors = static_cast<size_t>(2) * restart_count + 5;
    const size_t limit = std::numeric_limits<size_t>::max();
    if (fine_count > limit / vectors)
      throw std::overflow_error("strict FGMRES fine arena size overflow");
    const size_t fine_elements = vectors * fine_count;
    if (coarse_count > (limit - fine_elements) / 2)
      throw std::overflow_error("strict FGMRES coarse arena size overflow");
    return fine_elements + 2 * coarse_count;
  }

  void clear_views() {
    V.clear();
    Z.clear();
    b = r = w = x = nullptr;
    coarse_rhs = coarse_out = nullptr;
    H.clear();
    cs.clear();
    sn.clear();
    g.clear();
    y.clear();
    dot_values.clear();
    fine_n = coarse_n = bytes = 0;
    restart = 0;
  }

  void release(cudaStream_t stream) {
    if (storage != nullptr) {
      strict_check_cuda(cudaFreeAsync(storage, stream),
                        "strict FGMRES workspace release");
      storage = nullptr;
    }
    clear_views();
  }

  void release_noexcept(cudaStream_t stream) noexcept {
    if (storage != nullptr) {
      (void)cudaFreeAsync(storage, stream);
      storage = nullptr;
    }
    clear_views();
  }

  void configure(size_t fine_count, size_t coarse_count, int restart_count,
                 size_t budget_bytes, cudaStream_t stream) {
    if (fine_count == 0 || coarse_count == 0)
      throw std::invalid_argument("strict FGMRES workspace has zero extent");
    const size_t elements =
        checked_elements(fine_count, coarse_count, restart_count);
    if (elements > std::numeric_limits<size_t>::max() /
                       sizeof(LatticeComplex<T>))
      throw std::overflow_error("strict FGMRES workspace byte overflow");
    const size_t requested_bytes = elements * sizeof(LatticeComplex<T>);
    if (requested_bytes > budget_bytes)
      throw std::invalid_argument("strict FGMRES workspace exceeds budget");
    if (storage != nullptr && fine_n == fine_count &&
        coarse_n == coarse_count && restart == restart_count) {
      if (bytes != requested_bytes)
        throw std::runtime_error("strict FGMRES workspace accounting drift");
      return;
    }

    release(stream);
    try {
      V.resize(static_cast<size_t>(restart_count) + 1);
      Z.resize(static_cast<size_t>(restart_count));
      H.resize((static_cast<size_t>(restart_count) + 1) * restart_count);
      cs.resize(static_cast<size_t>(restart_count));
      sn.resize(static_cast<size_t>(restart_count));
      g.resize(static_cast<size_t>(restart_count) + 1);
      y.resize(static_cast<size_t>(restart_count));
      dot_values.resize(static_cast<size_t>(restart_count) + 1);
      strict_check_cuda(cudaMallocAsync(&storage, requested_bytes, stream),
                        "strict FGMRES workspace allocation");

      LatticeComplex<T> *base =
          static_cast<LatticeComplex<T> *>(storage);
      size_t cursor = 0;
      for (int i = 0; i <= restart_count; ++i) {
        V[static_cast<size_t>(i)] = base + cursor;
        cursor += fine_count;
      }
      for (int i = 0; i < restart_count; ++i) {
        Z[static_cast<size_t>(i)] = base + cursor;
        cursor += fine_count;
      }
      b = base + cursor;
      cursor += fine_count;
      r = base + cursor;
      cursor += fine_count;
      w = base + cursor;
      cursor += fine_count;
      x = base + cursor;
      cursor += fine_count;
      coarse_rhs = base + cursor;
      cursor += coarse_count;
      coarse_out = base + cursor;
      cursor += coarse_count;
      if (cursor != elements)
        throw std::runtime_error("strict FGMRES workspace partition drift");
      fine_n = fine_count;
      coarse_n = coarse_count;
      restart = restart_count;
      bytes = requested_bytes;
    } catch (...) {
      release_noexcept(stream);
      throw;
    }
  }
};

template <typename T> class StrictCoarseHierarchy {
 public:
  struct StrictCycleCounts {
    unsigned long long pre_smoother = 0;
    unsigned long long post_smoother = 0;
    unsigned long long restrictions = 0;
    unsigned long long prolongations = 0;
    unsigned long long recursive_solves = 0;
    unsigned long long coarse_iterations = 0;
    unsigned long long coarse_calls = 0;
  };

  StrictCoarseHierarchy(LatticeSet<T> *set, void *set_ptrs, int *params,
                        int start_level)
      : set_(set), set_ptrs_(static_cast<long long *>(set_ptrs)),
        params_(params), start_(start_level),
        num_levels_(params[_MG_NUM_LEVEL_]),
        parity_(params[_PARITY_]),
        smoother_steps_(params[_MG_MU_PRE_] > 0 ? params[_MG_MU_PRE_] : 2),
                        persistent_storage_(nullptr), persistent_bytes_(0) {
    try {
    strict_validate_mpi_backend(params_);
    if (num_levels_ < 2 || num_levels_ > 5 || start_ < 1 ||
        start_ >= num_levels_)
      throw std::invalid_argument(
          "strict hierarchy requires 2..5 levels and a valid coarse start");
    if (parity_ != 0 && parity_ != 1)
      throw std::invalid_argument("strict hierarchy parity must be 0 or 1");
    levels_ = new StrictPersistentLevel<T>[num_levels_];
    cycle_counts_.assign(static_cast<size_t>(num_levels_),
                         StrictCycleCounts{});
    size_t persistent_elements = 0;
    size_t max_compact = 0;
    size_t max_child_full = 0;
    size_t max_reduction_n = static_cast<size_t>(set_->lat_4dim_SC);
    for (int level = start_; level < num_levels_; ++level) {
      levels_[level].geometry = strict_read_level(params_, level);
      const int base = _MG_LEVEL1_E_ + (level - 1) * _MG_PARAMS_SIZE_;
      if (params_[base + 6] != params_[_DATA_TYPE_])
        throw std::invalid_argument(
            "strict recursive hierarchy mixed precision is not enabled yet");
      max_compact = std::max(
          max_compact, levels_[level].geometry.compact_n);
      max_reduction_n = std::max(
          max_reduction_n, levels_[level].geometry.compact_n);
      if (level > start_)
        max_child_full = std::max(
            max_child_full, levels_[level].geometry.full_n);
      persistent_elements +=
          2 * StrictWorkspaceArena<T>::align_elements(
                  levels_[level].geometry.compact_n);
      if (level > start_)
        persistent_elements += StrictWorkspaceArena<T>::align_elements(
            levels_[level].geometry.full_n);
      const int tag_base = 2000 + level * 200;
      levels_[level].compact_vector_halo =
          std::unique_ptr<StrictVectorHalo<T>>(new StrictVectorHalo<T>(
              set_, levels_[level].geometry.E,
              levels_[level].geometry.X, levels_[level].geometry.Y,
              levels_[level].geometry.Z, levels_[level].geometry.Lt,
              true, tag_base));
      levels_[level].compact_link_halo =
          std::unique_ptr<StrictAxisLinkHalo<T>>(new StrictAxisLinkHalo<T>(
              set_, levels_[level].geometry.E * levels_[level].geometry.E,
              levels_[level].geometry.X, levels_[level].geometry.Y,
              levels_[level].geometry.Z, levels_[level].geometry.Lt,
              false, tag_base + 50));
      levels_[level].full_vector_halo =
          std::unique_ptr<StrictVectorHalo<T>>(new StrictVectorHalo<T>(
              set_, levels_[level].geometry.E,
              levels_[level].geometry.X, levels_[level].geometry.Y,
              levels_[level].geometry.Z, levels_[level].geometry.Lt,
              false, tag_base + 100));
      levels_[level].full_link_halo =
          std::unique_ptr<StrictAxisLinkHalo<T>>(new StrictAxisLinkHalo<T>(
              set_, levels_[level].geometry.E * levels_[level].geometry.E,
              levels_[level].geometry.X, levels_[level].geometry.Y,
              levels_[level].geometry.Z, levels_[level].geometry.Lt,
              false, tag_base + 150));
      validate_assets(level);
    }

    size_t halo_bytes = 0;
    for (int level = start_; level < num_levels_; ++level) {
      halo_bytes += levels_[level].compact_vector_halo->bytes();
      halo_bytes += levels_[level].compact_link_halo->bytes();
      halo_bytes += levels_[level].full_vector_halo->bytes();
      halo_bytes += levels_[level].full_link_halo->bytes();
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
        max_child_full, max_reduction_n, set_->stream);
    strict_check_cuda(
        cudaMallocAsync(&coarse_fused_status_, 3 * sizeof(int),
                        set_->stream),
        "strict coarse fused status allocation");
    coarse_fused_status_bytes_ = 3 * sizeof(int);
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

  size_t run_fgmres(
      void *full_out, const void *full_rhs, void *gauge, void *clover_ee,
      void *clover_oo, void *clover_ee_inv, void *clover_oo_inv,
      const void *fine_null_vectors, int fine_E, int fine_X, int fine_Y,
      int fine_Z, int fine_T, int coarse_E, int coarse_X, int coarse_Y,
      int coarse_Z, int coarse_T, int restart, int max_iter, T tolerance,
      int nu_pre, int nu_post, size_t max_workspace_bytes,
      int &iterations, bool &converged, T &final_true_residual) {
    validate_fgmres_descriptor(
        full_out, full_rhs, gauge, clover_ee, clover_oo, clover_ee_inv,
        clover_oo_inv, fine_null_vectors, fine_E, fine_X, fine_Y, fine_Z,
        fine_T, coarse_E, coarse_X, coarse_Y, coarse_Z, coarse_T, restart,
        max_iter, tolerance, nu_pre, nu_post, max_workspace_bytes);

    const StrictLevelGeometry &coarse = levels_[start_].geometry;
    const size_t fine_n = static_cast<size_t>(fine_E) * fine_X * fine_Y *
                          fine_Z * (fine_T / 2);
    outer_.configure(fine_n, coarse.full_n, restart, max_workspace_bytes,
                     set_->stream);

    cublasPointerMode_t saved_pointer_mode = CUBLAS_POINTER_MODE_HOST;
    if (cublasGetPointerMode(set_->cublasH, &saved_pointer_mode) !=
        CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cannot query strict FGMRES pointer mode");
    if (cublasSetPointerMode(set_->cublasH, CUBLAS_POINTER_MODE_HOST) !=
        CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cannot select strict FGMRES host scalars");

    try {
      LatticeCloverBistabCg<T> fine;
      fine.give(set_);
      fine.init(gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv);
      if (qcu_parse_bool_env("PYQCU_STRICT_GLOBAL_DSLASH", false) &&
          (!fine_global_ || !fine_global_->matches(gauge))) {
        fine_global_.reset(new StrictFineGlobalDslash<T>(
            set_, gauge, fine_X, fine_Y, fine_Z, fine_T));
      }
      // Every producer/consumer below uses set_->stream.  Suppress the
      // single-rank Wilson endpoint syncs and synchronize only at the fused
      // C entry boundary (coarse dot reductions may still synchronize).
      fine.wilson_dslash.skip_final_sync_ = true;
      fgmres_impl(
          fine, full_out, full_rhs, fine_null_vectors, fine_E, fine_X,
          fine_Y, fine_Z, fine_T, restart, max_iter, tolerance, nu_pre,
          nu_post, iterations, converged, final_true_residual);
      strict_check_cuda(cudaGetLastError(), "strict FGMRES launch");
      strict_check_cuda(cudaStreamSynchronize(set_->stream),
                        "strict FGMRES output sync");
    } catch (...) {
      (void)cublasSetPointerMode(set_->cublasH, saved_pointer_mode);
      throw;
    }
    if (cublasSetPointerMode(set_->cublasH, saved_pointer_mode) !=
        CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cannot restore strict FGMRES pointer mode");
    return outer_.bytes;
  }

  size_t allocated_bytes() const {
    size_t halo_bytes = 0;
    for (int level = start_; level < num_levels_; ++level) {
      halo_bytes += levels_[level].compact_vector_halo->bytes();
      halo_bytes += levels_[level].compact_link_halo->bytes();
      halo_bytes += levels_[level].full_vector_halo->bytes();
      halo_bytes += levels_[level].full_link_halo->bytes();
    }
    return persistent_bytes_ + arena_.bytes + coarse_fused_status_bytes_ +
           halo_bytes;
  }

  int start_level() const { return start_; }

 private:
  LatticeSet<T> *set_;
  long long *set_ptrs_;
  int *params_;
  int start_, num_levels_, parity_, smoother_steps_;
  StrictPersistentLevel<T> *levels_ = nullptr;
  std::vector<StrictCycleCounts> cycle_counts_;
  StrictWorkspaceArena<T> arena_;
  StrictOuterWorkspace<T> outer_;
  std::unique_ptr<StrictFineGlobalDslash<T>> fine_global_;
  void *persistent_storage_;
  size_t persistent_bytes_;
  void *coarse_fused_status_ = nullptr;
  size_t coarse_fused_status_bytes_ = 0;
  StrictFgmresTrace<T> *trace_ = nullptr;
  int trace_iteration_ = -1;
  void *trace_work_ = nullptr;
  void *trace_scratch_ = nullptr;
  size_t trace_work_elements_ = 0;

  class TraceBinding {
   public:
    TraceBinding(StrictCoarseHierarchy &owner, StrictFgmresTrace<T> &trace)
        : owner_(owner) {
      owner_.trace_ = &trace;
    }

    ~TraceBinding() { owner_.trace_ = nullptr; }

   private:
    StrictCoarseHierarchy &owner_;
  };

  void release_noexcept() noexcept {
    if (set_ != nullptr) {
      fine_global_.reset();
      outer_.release_noexcept(set_->stream);
      if (arena_.storage != nullptr) {
        (void)cudaFreeAsync(arena_.storage, set_->stream);
        arena_.storage = nullptr;
      }
      if (persistent_storage_ != nullptr) {
        (void)cudaFreeAsync(persistent_storage_, set_->stream);
        persistent_storage_ = nullptr;
      }
      if (coarse_fused_status_ != nullptr) {
        (void)cudaFreeAsync(coarse_fused_status_, set_->stream);
        coarse_fused_status_ = nullptr;
      }
      if (trace_work_ != nullptr) {
        (void)cudaFreeAsync(trace_work_, set_->stream);
        trace_work_ = nullptr;
      }
      if (trace_scratch_ != nullptr) {
        (void)cudaFreeAsync(trace_scratch_, set_->stream);
        trace_scratch_ = nullptr;
      }
      (void)cudaStreamSynchronize(set_->stream);
    }
    arena_.r = arena_.v = arena_.tmp = nullptr;
    arena_.rhat = arena_.p = arena_.s = arena_.t = nullptr;
    arena_.correction_full = nullptr;
    arena_.dot_storage = nullptr;
    arena_.dot_elements = 0;
    arena_.bytes = 0;
    persistent_bytes_ = 0;
    coarse_fused_status_bytes_ = 0;
    trace_work_elements_ = 0;
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

  using TraceClock = std::chrono::steady_clock;

  TraceClock::time_point trace_stage_start() {
    if (trace_ == nullptr || !trace_->enabled()) return TraceClock::time_point();
    strict_check_cuda(cudaStreamSynchronize(set_->stream),
                      "strict trace stage begin sync");
    return TraceClock::now();
  }

  void trace_stage_end(int outer_iteration, int level, const char *name,
                       TraceClock::time_point started) {
    if (started == TraceClock::time_point()) return;
    strict_check_cuda(cudaStreamSynchronize(set_->stream),
                      "strict trace stage end sync");
    const double seconds = std::chrono::duration<double>(
                               TraceClock::now() - started)
                               .count();
    trace_->stage(outer_iteration, level, name, seconds);
  }

  template <typename Function>
  void trace_stage(int outer_iteration, int level, const char *name,
                   Function function) {
    if (trace_ == nullptr || !trace_->enabled()) {
      function();
      return;
    }
    const TraceClock::time_point started = trace_stage_start();
    function();
    trace_stage_end(outer_iteration, level, name, started);
  }

  void ensure_trace_workspace(size_t n) {
    if (trace_ == nullptr || !trace_->enabled() || n == 0) return;
    if (trace_work_ != nullptr && trace_work_elements_ >= n) return;
    if (trace_work_ != nullptr) {
      strict_check_cuda(cudaFreeAsync(trace_work_, set_->stream),
                        "strict trace workspace release");
      trace_work_ = nullptr;
    }
    if (trace_scratch_ != nullptr) {
      strict_check_cuda(cudaFreeAsync(trace_scratch_, set_->stream),
                        "strict trace scratch release");
      trace_scratch_ = nullptr;
    }
    const size_t bytes = n * sizeof(LatticeComplex<T>);
    strict_check_cuda(cudaMallocAsync(&trace_work_, bytes, set_->stream),
                      "strict trace workspace allocation");
    strict_check_cuda(cudaMallocAsync(&trace_scratch_, bytes, set_->stream),
                      "strict trace scratch allocation");
    trace_work_elements_ = n;
  }

  void trace_residual(int level, const char *name, const void *solution,
                      const void *rhs, size_t n) {
    if (trace_ == nullptr || !trace_->enabled()) return;
    ensure_trace_workspace(n);
    const T rhs_norm = norm(rhs, n);
    apply_matpc(level, trace_work_, solution, trace_scratch_);
    strict_subtract_kernel<T>
        <<<blocks(n), _BLOCK_SIZE_, 0, set_->stream>>>(
            trace_work_, rhs, trace_work_, static_cast<int>(n));
    const T residual = norm(trace_work_, n);
    trace_->residual(trace_iteration_, level, name, residual, rhs_norm);
  }

  void trace_counter(
      int outer_iteration, int level, const char *name,
      unsigned long long value) {
    if (trace_ == nullptr || !trace_->enabled()) return;
    trace_->counter(outer_iteration, level, name, value);
  }

  void emit_cycle_counters(int outer_iteration) {
    if (trace_ == nullptr || !trace_->enabled()) return;
    for (int level = 0; level < num_levels_; ++level) {
      const StrictCycleCounts &counts =
          cycle_counts_[static_cast<size_t>(level)];
      trace_counter(outer_iteration, level, "pre_smoother_iterations",
                    counts.pre_smoother);
      trace_counter(outer_iteration, level, "post_smoother_iterations",
                    counts.post_smoother);
      trace_counter(outer_iteration, level, "restriction_calls",
                    counts.restrictions);
      trace_counter(outer_iteration, level, "prolongation_calls",
                    counts.prolongations);
      trace_counter(outer_iteration, level, "recursive_solve_calls",
                    counts.recursive_solves);
      trace_counter(outer_iteration, level, "coarse_solver_iterations",
                    counts.coarse_iterations);
      trace_counter(outer_iteration, level, "coarse_solver_calls",
                    counts.coarse_calls);
    }
  }

  void reset_cycle_counters() {
    std::fill(
        cycle_counts_.begin(), cycle_counts_.end(), StrictCycleCounts{});
  }

  void trace_fine_residual(LatticeCloverBistabCg<T> &fine,
                           const char *name, const void *solution,
                           const void *rhs, size_t n) {
    if (trace_ == nullptr || !trace_->enabled()) return;
    const T rhs_norm = norm(rhs, n);
    fine_matpc(fine, outer_.w, solution);
    strict_subtract_kernel<T>
        <<<blocks(n), _BLOCK_SIZE_, 0, set_->stream>>>(
            outer_.w, rhs, outer_.w, static_cast<int>(n));
    const T residual = norm(outer_.w, n);
    trace_->residual(trace_iteration_, 0, name, residual, rhs_norm);
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
    return strict_global_sum_complex(result);
  }

  void dot_pair_device(const void *left0, const void *right0,
                       const void *left1, const void *right1, size_t n) {
    if (n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("strict dot pair vector exceeds cublas int range");
    LatticeComplex<T> *device_results =
        static_cast<LatticeComplex<T> *>(set_->device_vals);
    if (device_results == nullptr || arena_.dot_storage == nullptr)
      throw std::runtime_error("strict dot pair device scratch is null");
    const int reduction_count = strict_reduction_blocks(n);
    if (2 * static_cast<size_t>(reduction_count) > arena_.dot_elements)
      throw std::runtime_error("strict dot pair reduction scratch is too small");
    strict_dot_pair_partial_kernel<T, 256>
        <<<reduction_count, 256, 0, set_->stream>>>(
            static_cast<const LatticeComplex<T> *>(left0),
            static_cast<const LatticeComplex<T> *>(right0),
            static_cast<const LatticeComplex<T> *>(left1),
            static_cast<const LatticeComplex<T> *>(right1),
            static_cast<int>(n),
            static_cast<LatticeComplex<T> *>(arena_.dot_storage));
    strict_check_cuda(cudaGetLastError(), "strict dot pair partial launch");
    strict_dot_pair_reduce_kernel<T, 256>
        <<<1, 256, 0, set_->stream>>>(
            static_cast<const LatticeComplex<T> *>(arena_.dot_storage),
            reduction_count, device_results);
    strict_check_cuda(cudaGetLastError(), "strict dot pair reduce launch");
  }

  void dot_pair(const void *left0, const void *right0,
                const void *left1, const void *right1, size_t n,
                LatticeComplex<T> &result0,
                LatticeComplex<T> &result1) {
    dot_pair_device(left0, right0, left1, right1, n);
    LatticeComplex<T> *device_results =
        static_cast<LatticeComplex<T> *>(set_->device_vals);
    LatticeComplex<T> host_results[2];
    strict_check_cuda(cudaMemcpyAsync(
                          host_results, device_results,
                          2 * sizeof(LatticeComplex<T>),
                          cudaMemcpyDeviceToHost, set_->stream),
                      "strict dot pair copy");
    strict_check_cuda(cudaStreamSynchronize(set_->stream),
                      "strict dot pair sync");
    result0 = strict_global_sum_complex(host_results[0]);
    result1 = strict_global_sum_complex(host_results[1]);
  }

  void dot_pair_global_device(
      const void *left0, const void *right0,
      const void *left1, const void *right1, size_t n) {
    dot_pair_device(left0, right0, left1, right1, n);
    if (strict_world_size() == 1) return;
    LatticeComplex<T> *device_results =
        static_cast<LatticeComplex<T> *>(set_->device_vals);
    LatticeComplex<T> results[2];
    strict_check_cuda(cudaMemcpy(
                          results, device_results,
                          2 * sizeof(LatticeComplex<T>),
                          cudaMemcpyDeviceToHost),
                      "strict dot pair global copy to host");
    T values[4] = {
        results[0].real(), results[0].imag(),
        results[1].real(), results[1].imag()};
    const MPI_Datatype scalar_type =
        std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE;
    checkMpiErrors(MPI_Allreduce(
        MPI_IN_PLACE, values, 4, scalar_type, MPI_SUM, MPI_COMM_WORLD));
    results[0] = LatticeComplex<T>(values[0], values[1]);
    results[1] = LatticeComplex<T>(values[2], values[3]);
    strict_check_cuda(cudaMemcpy(
                          device_results, results,
                          2 * sizeof(LatticeComplex<T>),
                          cudaMemcpyHostToDevice),
                      "strict dot pair global copy to device");
  }

  void dot_many(const void *basis, size_t basis_stride, const void *right,
                int count, void *device_output) {
    if (count <= 0 || count > static_cast<int>(outer_.dot_values.size()))
      throw std::invalid_argument("strict dot-many count is outside workspace");
    if (outer_.fine_n == 0 ||
        outer_.fine_n > static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("strict dot-many vector exceeds int range");
    const int reduction_count = std::min(
        strict_reduction_blocks(outer_.fine_n),
        static_cast<int>(outer_.fine_n / static_cast<size_t>(count)));
    if (reduction_count <= 0)
      throw std::invalid_argument(
          "strict dot-many requires at least one result slot per partial block");
    strict_dot_many_partial_kernel<T, 256>
        <<<dim3(static_cast<unsigned int>(count),
                static_cast<unsigned int>(reduction_count), 1),
            256, 0, set_->stream>>>(
            static_cast<const LatticeComplex<T> *>(basis), basis_stride,
            static_cast<const LatticeComplex<T> *>(right),
            static_cast<int>(outer_.fine_n), count, reduction_count,
            static_cast<LatticeComplex<T> *>(device_output));
    strict_check_cuda(cudaGetLastError(), "strict dot-many partial launch");
    strict_dot_many_reduce_kernel<T, 256>
        <<<count, 256, 0, set_->stream>>>(
            static_cast<const LatticeComplex<T> *>(device_output),
            reduction_count, count,
            static_cast<LatticeComplex<T> *>(outer_.coarse_out));
    strict_check_cuda(cudaGetLastError(), "strict dot-many reduce launch");
    strict_check_cuda(cudaMemcpyAsync(
                          device_output, outer_.coarse_out,
                          static_cast<size_t>(count) * sizeof(LatticeComplex<T>),
                          cudaMemcpyDeviceToDevice, set_->stream),
                      "strict dot-many compact copy");
    strict_check_cuda(cudaMemcpyAsync(
                          outer_.dot_values.data(), outer_.coarse_out,
                          static_cast<size_t>(count) * sizeof(LatticeComplex<T>),
                          cudaMemcpyDeviceToHost, set_->stream),
                      "strict dot-many copy");
    strict_check_cuda(cudaStreamSynchronize(set_->stream),
                      "strict dot-many sync");
    if (strict_world_size() > 1) {
      std::vector<T> values(static_cast<size_t>(2) * count);
      for (int column = 0; column < count; ++column) {
        values[static_cast<size_t>(2) * column] =
            outer_.dot_values[static_cast<size_t>(column)].real();
        values[static_cast<size_t>(2) * column + 1] =
            outer_.dot_values[static_cast<size_t>(column)].imag();
      }
      const MPI_Datatype scalar_type =
          std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE;
      checkMpiErrors(MPI_Allreduce(
          MPI_IN_PLACE, values.data(), 2 * count, scalar_type, MPI_SUM,
          MPI_COMM_WORLD));
      for (int column = 0; column < count; ++column) {
        outer_.dot_values[static_cast<size_t>(column)] =
            LatticeComplex<T>(
                values[static_cast<size_t>(2) * column],
                values[static_cast<size_t>(2) * column + 1]);
      }
      // The device copy consumed by strict_orthogonalize_kernel() is still
      // holding rank-local partial coefficients; refresh it with the global
      // sums so the Arnoldi basis and the reported Hessenberg matrix agree.
      strict_check_cuda(cudaMemcpyAsync(
                            device_output, outer_.dot_values.data(),
                            static_cast<size_t>(count) *
                                sizeof(LatticeComplex<T>),
                            cudaMemcpyHostToDevice, set_->stream),
                        "strict dot-many global copy to device");
    }
  }

  void cublas_check(cublasStatus_t status, const char *where) const {
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::fprintf(stderr, "PYQCU::SOLVER::STRICT_MG::CUBLAS: %s (%d)\n",
                   where, static_cast<int>(status));
      throw std::runtime_error("strict FGMRES cuBLAS failure");
    }
  }

  void copy_vector(void *out, const void *in, size_t n,
                   const char *where) const {
    strict_check_cuda(cudaMemcpyAsync(
                          out, in, n * sizeof(LatticeComplex<T>),
                          cudaMemcpyDeviceToDevice, set_->stream),
                      where);
  }

  void zero_vector(void *out, size_t n, const char *where) const {
    strict_check_cuda(
        cudaMemsetAsync(out, 0, n * sizeof(LatticeComplex<T>), set_->stream),
        where);
  }

  void axpy(void *out, const void *in, size_t n,
            const std::complex<T> &alpha) const {
    if (n > static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("strict FGMRES axpy exceeds cublas int range");
    const LatticeComplex<T> value(alpha.real(), alpha.imag());
    cublas_check(_cublasAxpy<T>(set_->cublasH, static_cast<int>(n), &value,
                                in, 1, out, 1),
                 "strict FGMRES axpy");
  }

  void scale(void *vector, size_t n, T value) const {
    if (n > static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("strict FGMRES scale exceeds cublas int range");
    const LatticeComplex<T> alpha(value, (T)0);
    cublas_check(_cublasScal<T>(set_->cublasH, static_cast<int>(n), &alpha,
                                vector, 1),
                 "strict FGMRES scale");
  }

  T norm(const void *vector, size_t n) {
    const LatticeComplex<T> value = dot(vector, vector, n);
    const T norm2 = value.real();
    if (!std::isfinite(static_cast<double>(norm2)))
      throw std::runtime_error("strict FGMRES norm is non-finite");
    return std::sqrt(std::max((T)0, norm2));
  }

  static bool finite_complex(const std::complex<T> &value) {
    return std::isfinite(static_cast<double>(value.real())) &&
           std::isfinite(static_cast<double>(value.imag()));
  }

  void validate_fgmres_descriptor(
      const void *full_out, const void *full_rhs, const void *gauge,
      const void *clover_ee, const void *clover_oo,
      const void *clover_ee_inv, const void *clover_oo_inv,
      const void *fine_null_vectors, int fine_E, int fine_X, int fine_Y,
      int fine_Z, int fine_T, int coarse_E, int coarse_X, int coarse_Y,
      int coarse_Z, int coarse_T, int restart, int max_iter, T tolerance,
      int nu_pre, int nu_post, size_t max_workspace_bytes) const {
    if (full_out == nullptr || full_rhs == nullptr || gauge == nullptr ||
        clover_ee == nullptr || clover_oo == nullptr ||
        clover_ee_inv == nullptr || clover_oo_inv == nullptr ||
        fine_null_vectors == nullptr)
      throw std::invalid_argument("strict FGMRES field pointer is null");
    if (start_ != 1 || (parity_ != 0 && parity_ != 1))
      throw std::invalid_argument(
          "strict fine FGMRES requires start_level=1 and parity in {0,1}");
    if (params_[_SET_PLAN_] != _SET_PLAN1_)
      throw std::invalid_argument("strict fine FGMRES requires QCU plan 1");
    if (params_[_MG_USE_INIT_GUESS_] != 0 &&
        params_[_MG_USE_INIT_GUESS_] != 1)
      throw std::invalid_argument("strict FGMRES warm-start flag must be 0/1");
    strict_validate_mpi_backend(params_);
    if (fine_E != _LAT_SC_)
      throw std::invalid_argument("strict fine FGMRES requires 12 fine dof");
    strict_validate_parity_geometry(
        fine_X, fine_Y, fine_Z, fine_T);
    if (fine_X != params_[_LAT_X_] || fine_Y != params_[_LAT_Y_] ||
        fine_Z != params_[_LAT_Z_] || fine_T != params_[_LAT_T_] ||
        fine_X != set_->host_params[_LAT_X_] ||
        fine_Y != set_->host_params[_LAT_Y_] ||
        fine_Z != set_->host_params[_LAT_Z_] ||
        fine_T / 2 != set_->host_params[_LAT_T_])
      throw std::invalid_argument("strict FGMRES fine geometry mismatch");
    const StrictLevelGeometry &coarse = levels_[start_].geometry;
    if (coarse_E != coarse.E || coarse_X != coarse.X ||
        coarse_Y != coarse.Y || coarse_Z != coarse.Z ||
        coarse_T != coarse.Lt)
      throw std::invalid_argument("strict FGMRES coarse geometry mismatch");
    strict_validate_transfer_geometry(
        coarse_E, fine_E, fine_X, fine_Y, fine_Z, fine_T,
        coarse_X, coarse_Y, coarse_Z, coarse_T, parity_);
    const size_t fine_n = static_cast<size_t>(fine_E) * fine_X * fine_Y *
                          fine_Z * (fine_T / 2);
    if (fine_n != static_cast<size_t>(set_->lat_4dim_SC) ||
        fine_n > static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("strict FGMRES compact shape mismatch");
    if (coarse.full_n >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        coarse.compact_n >
            static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("strict FGMRES coarse field is too large");
    if (restart <= 0 || max_iter <= 0 || restart > max_iter)
      throw std::invalid_argument(
          "strict FGMRES requires 1 <= restart <= max_iter");
    if (!(tolerance > (T)0 && tolerance < (T)1) ||
        !std::isfinite(static_cast<double>(tolerance)))
      throw std::invalid_argument("strict FGMRES tolerance must be in (0,1)");
    if (nu_pre < 0 || nu_post < 0)
      throw std::invalid_argument("strict FGMRES MR counts must be non-negative");
    const size_t elements = StrictOuterWorkspace<T>::checked_elements(
        fine_n, coarse.full_n, restart);
    if (elements > std::numeric_limits<size_t>::max() /
                       sizeof(LatticeComplex<T>) ||
        elements * sizeof(LatticeComplex<T>) > max_workspace_bytes)
      throw std::invalid_argument("strict FGMRES workspace exceeds budget");
  }

  void fine_hopping(LatticeCloverBistabCg<T> &fine, void *out,
                    const void *in, int source_parity) {
    if (fine_global_) {
      fine_global_->apply(out, in, 1 - source_parity);
      return;
    }
    // LatticeWilsonDslash names the block by its destination/source pair:
    // run_eo is odd -> even and run_oe is even -> odd.  Keeping this mapping
    // in one helper prevents the target-parity branch from being silently
    // reversed in prepare, action, or reconstruction.
    if (source_parity == 0)
      fine.wilson_dslash.run_oe(out, const_cast<void *>(in), fine.gauge);
    else if (source_parity == 1)
      fine.wilson_dslash.run_eo(out, const_cast<void *>(in), fine.gauge);
    else
      throw std::invalid_argument("strict fine hopping parity must be 0/1");
  }

  void fine_diagonal_inverse(LatticeCloverBistabCg<T> &fine, void *field,
                             int parity) {
    if (parity == 0)
      fine.clover_dslash_ee_inv.give(field);
    else if (parity == 1)
      fine.clover_dslash_oo_inv.give(field);
    else
      throw std::invalid_argument(
          "strict fine diagonal-inverse parity must be 0/1");
  }

  void fine_matpc(LatticeCloverBistabCg<T> &fine, void *out,
                  const void *in) {
    const size_t n = static_cast<size_t>(set_->lat_4dim_SC);
    // M_p = A_p^{-1} (A_p - kappa^2 H_pq A_q^{-1} H_qp).
    // The two shared lattice scratch buffers hold the hopping/inverse chain.
    // Fuse the identity copy and final scaled subtraction: this is on the
    // innermost FGMRES/MR path, so a separate D2D copy plus cuBLAS AXPY would
    // add two launches for every fine MATPC application without changing the
    // mathematical operation.
    fine_hopping(fine, set_->device_vec0, in, parity_);
    fine_diagonal_inverse(fine, set_->device_vec0, 1 - parity_);
    fine_hopping(fine, set_->device_vec1, set_->device_vec0, 1 - parity_);
    fine_diagonal_inverse(fine, set_->device_vec1, parity_);
    const int thread_blocks = blocks(n);
    strict_fine_matpc_update_kernel<T>
        <<<thread_blocks, _BLOCK_SIZE_, 0, set_->stream>>>(
            out, in, set_->device_vec1,
            set_->kappa() * set_->kappa(), static_cast<int>(n));
    strict_check_cuda(cudaGetLastError(),
                      "strict fused fine MATPC update launch");
  }

  void fine_prepare_in_stream(LatticeCloverBistabCg<T> &fine,
                              void *compact_rhs,
                              const void *full_rhs) {
    const size_t n = static_cast<size_t>(set_->lat_4dim_SC);
    const LatticeComplex<T> *rhs =
        static_cast<const LatticeComplex<T> *>(full_rhs);
    const int other_parity = 1 - parity_;
    const LatticeComplex<T> *target_rhs = rhs +
        static_cast<size_t>(parity_) * n;
    const LatticeComplex<T> *other_rhs = rhs +
        static_cast<size_t>(other_parity) * n;

    // b_p^pc = A_p^{-1}(b_p + kappa H_pq A_q^{-1} b_q).
    // Build the parenthesized sum first.  Applying A_p^{-1} only to the
    // target-parity RHS before the hopping update would instead produce
    // A_p^{-1} b_p + kappa H_pq A_q^{-1} b_q, which is not the symmetric
    // Clover MATPC RHS used by QUDA.
    strict_check_cuda(cudaMemcpyAsync(
                          compact_rhs, target_rhs,
                          n * sizeof(LatticeComplex<T>),
                          cudaMemcpyDeviceToDevice, set_->stream),
                      "strict fine prepare target copy");
    strict_check_cuda(cudaMemcpyAsync(
                          set_->device_vec2, other_rhs,
                          n * sizeof(LatticeComplex<T>),
                          cudaMemcpyDeviceToDevice, set_->stream),
                      "strict fine prepare other copy");
    fine_diagonal_inverse(fine, set_->device_vec2, other_parity);
    fine_hopping(fine, set_->device_vec0, set_->device_vec2, other_parity);
    const LatticeComplex<T> kappa(set_->kappa(), (T)0);
    cublas_check(_cublasAxpy<T>(
                     set_->cublasH, static_cast<int>(n), &kappa,
                     set_->device_vec0, 1, compact_rhs, 1),
                 "strict fine prepare hopping update");
    fine_diagonal_inverse(fine, compact_rhs, parity_);
  }

  void fine_reconstruct_in_stream(LatticeCloverBistabCg<T> &fine,
                                  void *full_out, const void *full_rhs,
                                  const void *target_solution) {
    const size_t n = static_cast<size_t>(set_->lat_4dim_SC);
    LatticeComplex<T> *out = static_cast<LatticeComplex<T> *>(full_out);
    const LatticeComplex<T> *rhs =
        static_cast<const LatticeComplex<T> *>(full_rhs);
    const int other_parity = 1 - parity_;
    const LatticeComplex<T> *other_rhs = rhs +
        static_cast<size_t>(other_parity) * n;
    copy_vector(out + static_cast<size_t>(parity_) * n, target_solution, n,
                "strict fine reconstruct target copy");
    copy_vector(set_->device_vec0, other_rhs, n,
                "strict fine reconstruct other rhs copy");
    fine_hopping(fine, set_->device_vec1, target_solution, parity_);
    const LatticeComplex<T> kappa(set_->kappa(), (T)0);
    cublas_check(_cublasAxpy<T>(
                     set_->cublasH, static_cast<int>(n), &kappa,
                     set_->device_vec1, 1, set_->device_vec0, 1),
                 "strict fine reconstruct axpy");
    fine_diagonal_inverse(fine, set_->device_vec0, other_parity);
    copy_vector(out + static_cast<size_t>(other_parity) * n,
                set_->device_vec0, n,
                "strict fine reconstruct other copy");
  }

  void fine_smooth(LatticeCloverBistabCg<T> &fine, void *solution,
                   void *residual, void *image, size_t n, int count) {
    // Match the Python fine-MR guard exactly: it thresholds |<Ar,Ar>|,
    // rather than its squared magnitude.
    const T floor = (T)1e-20;
    for (int iteration = 0; iteration < count; ++iteration) {
      fine_matpc(fine, image, residual);
      dot_pair_global_device(image, residual, image, image, n);
      strict_mr_give_alpha_kernel<T>
          <<<1, 1, 0, set_->stream>>>(
              set_->device_vals, floor, false);
      strict_check_cuda(cudaGetLastError(), "strict fine MR alpha launch");
      strict_mr_update_device_kernel<T>
          <<<blocks(n), _BLOCK_SIZE_, 0, set_->stream>>>(
              solution, residual, image, set_->device_vals,
              static_cast<int>(n));
      strict_check_cuda(cudaGetLastError(), "strict fine MR update launch");
    }
  }

  void precondition_fine(LatticeCloverBistabCg<T> &fine, void *out,
                         const void *source, const void *fine_null_vectors,
                         int fine_E, int fine_X, int fine_Y, int fine_Z,
                         int fine_T, int nu_pre, int nu_post) {
    const size_t n = outer_.fine_n;
    const int outer_iteration = trace_iteration_;
    trace_stage(outer_iteration, 0, "fine_pre_smoother", [&] {
      zero_vector(out, n, "strict fine preconditioner zero");
      copy_vector(outer_.r, source, n,
                  "strict fine preconditioner residual copy");
      fine_smooth(fine, out, outer_.r, outer_.w, n, nu_pre);
    });
    cycle_counts_[0].pre_smoother +=
        static_cast<unsigned long long>(std::max(0, nu_pre));
    trace_fine_residual(fine, "after_fine_pre_smoother", out, source, n);

    const StrictLevelGeometry &coarse = levels_[start_].geometry;
    trace_stage(outer_iteration, 0, "fine_restriction", [&] {
      strict_restrict_parity_kernel<T>
          <<<blocks(coarse.full_n), _BLOCK_SIZE_, 0, set_->stream>>>(
              outer_.coarse_rhs, outer_.r, fine_null_vectors,
              coarse.E, fine_E, fine_X, fine_Y, fine_Z, fine_T,
              coarse.X, coarse.Y, coarse.Z, coarse.Lt, parity_);
    });
    trace_stage(outer_iteration, start_, "coarse_vcycle", [&] {
      // The recursive hierarchy and its arena are persistent.  Avoid the
      // public V-cycle API's entry/exit synchronizations; all kernels remain on
      // the same stream and coarse dot reductions retain their existing syncs.
      solve_level(start_, outer_.coarse_rhs, outer_.coarse_out);
    });
    trace_stage(outer_iteration, 0, "fine_prolongation", [&] {
      strict_prolong_parity_kernel<T>
          <<<blocks(n), _BLOCK_SIZE_, 0, set_->stream>>>(
              outer_.w, outer_.coarse_out, fine_null_vectors,
              coarse.E, fine_E, fine_X, fine_Y, fine_Z, fine_T,
              coarse.X, coarse.Y, coarse.Z, coarse.Lt, parity_);
      strict_add_kernel<T><<<blocks(n), _BLOCK_SIZE_, 0, set_->stream>>>(
          out, outer_.w, static_cast<int>(n));
    });
    trace_fine_residual(fine, "after_fine_prolongation", out, source, n);

    trace_stage(outer_iteration, 0, "fine_correction_residual", [&] {
      fine_matpc(fine, outer_.w, out);
      strict_subtract_kernel<T>
          <<<blocks(n), _BLOCK_SIZE_, 0, set_->stream>>>(
              outer_.r, source, outer_.w, static_cast<int>(n));
    });
    trace_stage(outer_iteration, 0, "fine_post_smoother", [&] {
      fine_smooth(fine, out, outer_.r, outer_.w, n, nu_post);
    });
    cycle_counts_[0].post_smoother +=
        static_cast<unsigned long long>(std::max(0, nu_post));
    trace_fine_residual(fine, "after_fine_post_smoother", out, source, n);
  }

  void reset_outer_scalars(int restart) {
    std::fill(outer_.H.begin(), outer_.H.end(), std::complex<T>((T)0, (T)0));
    std::fill(outer_.cs.begin(), outer_.cs.end(), std::complex<T>((T)0, (T)0));
    std::fill(outer_.sn.begin(), outer_.sn.end(), std::complex<T>((T)0, (T)0));
    std::fill(outer_.g.begin(), outer_.g.end(), std::complex<T>((T)0, (T)0));
    std::fill(outer_.y.begin(), outer_.y.end(), std::complex<T>((T)0, (T)0));
    if (static_cast<int>(outer_.cs.size()) != restart)
      throw std::runtime_error("strict FGMRES host scalar size drift");
  }

  void apply_givens(int column, int restart) {
    auto Hidx = [restart](int row, int col) {
      return static_cast<size_t>(row) * restart + col;
    };
    for (int i = 0; i < column; ++i) {
      const std::complex<T> upper = outer_.H[Hidx(i, column)];
      const std::complex<T> lower = outer_.H[Hidx(i + 1, column)];
      outer_.H[Hidx(i, column)] =
          std::conj(outer_.cs[i]) * upper +
          std::conj(outer_.sn[i]) * lower;
      outer_.H[Hidx(i + 1, column)] =
          -outer_.sn[i] * upper + outer_.cs[i] * lower;
    }

    const std::complex<T> diagonal = outer_.H[Hidx(column, column)];
    const std::complex<T> next = outer_.H[Hidx(column + 1, column)];
    // Complex Givens rotations must preserve the two-vector norm.  Using
    // sqrt(abs(a*a+b*b)) is phase-sensitive and can spuriously vanish, e.g.
    // for a=i and b=1.  Match QUDA FlexArnoldiProcedure's
    // sqrt(norm(a)+norm(b)) exactly.
    const T denominator = std::sqrt(
        std::norm(diagonal) + std::norm(next));
    if (!(denominator > (T)0) ||
        !std::isfinite(static_cast<double>(denominator))) {
      outer_.sn[column] = std::complex<T>((T)0, (T)0);
      outer_.cs[column] = std::complex<T>((T)1, (T)0);
      outer_.H[Hidx(column + 1, column)] =
          std::complex<T>((T)0, (T)0);
    } else {
      outer_.sn[column] = next / denominator;
      outer_.cs[column] = diagonal / denominator;
      outer_.H[Hidx(column, column)] =
          std::conj(outer_.cs[column]) * diagonal +
          std::conj(outer_.sn[column]) * next;
      outer_.H[Hidx(column + 1, column)] =
          std::complex<T>((T)0, (T)0);
    }
    const std::complex<T> old_g = outer_.g[column];
    outer_.g[column + 1] = -outer_.sn[column] * old_g;
    outer_.g[column] = std::conj(outer_.cs[column]) * old_g;
  }

  void update_solution_from_hessenberg(int inner, int restart) {
    auto Hidx = [restart](int row, int col) {
      return static_cast<size_t>(row) * restart + col;
    };
    for (int i = inner - 1; i >= 0; --i) {
      std::complex<T> value = outer_.g[i];
      for (int j = i + 1; j < inner; ++j)
        value -= outer_.H[Hidx(i, j)] * outer_.y[j];
      const std::complex<T> diagonal = outer_.H[Hidx(i, i)];
      if (!finite_complex(value) || !finite_complex(diagonal))
        throw std::runtime_error("strict FGMRES triangular solve is non-finite");
      // Preserve _solve_upper_triangular(): only an exact zero diagonal is
      // suppressed; small finite pivots are allowed to update the iterate,
      // after which the mandatory true-residual refresh judges the result.
      outer_.y[i] = std::abs(diagonal) == (T)0
                        ? std::complex<T>((T)0, (T)0)
                        : value / diagonal;
    }
    for (int i = 0; i < inner; ++i)
      axpy(outer_.x, outer_.Z[i], outer_.fine_n, outer_.y[i]);
  }

  void fgmres_impl(
      LatticeCloverBistabCg<T> &fine, void *full_out, const void *full_rhs,
      const void *fine_null_vectors, int fine_E, int fine_X, int fine_Y,
      int fine_Z, int fine_T, int restart, int max_iter, T tolerance,
      int nu_pre, int nu_post, int &iterations, bool &converged,
      T &final_true_residual) {
    StrictFgmresTrace<T> trace;
    TraceBinding trace_binding(*this, trace);
    const size_t n = outer_.fine_n;
    fine_prepare_in_stream(fine, outer_.b, full_rhs);
    if (params_[_MG_USE_INIT_GUESS_] != 0) {
      const LatticeComplex<T> *initial =
          static_cast<const LatticeComplex<T> *>(full_out) +
          static_cast<size_t>(parity_) * n;
      copy_vector(outer_.x, initial, n, "strict FGMRES warm x0 copy");
    } else {
      zero_vector(outer_.x, n, "strict FGMRES cold x0 zero");
    }

    const T b_norm = norm(outer_.b, n);
    trace.begin(b_norm);
    if (b_norm == (T)0) {
      zero_vector(outer_.x, n, "strict FGMRES zero rhs solution");
      zero_vector(outer_.r, n, "strict FGMRES zero rhs residual");
      iterations = 0;
      converged = true;
      final_true_residual = (T)0;
      trace.initial((T)0);
      trace.end(iterations, converged, final_true_residual);
      fine_reconstruct_in_stream(fine, full_out, full_rhs, outer_.x);
      return;
    }

    fine_matpc(fine, outer_.w, outer_.x);
    strict_subtract_kernel<T><<<blocks(n), _BLOCK_SIZE_, 0, set_->stream>>>(
        outer_.r, outer_.b, outer_.w, static_cast<int>(n));
    T residual = norm(outer_.r, n);
    trace.initial(residual);
    const T threshold = tolerance * b_norm;
    iterations = 0;
    converged = residual <= threshold;
    const T basis_floor =
        std::is_same<T, float>::value ? (T)1e-20 : (T)1e-30;

    while (iterations < max_iter && !converged) {
      const T beta = residual;
      if (!(beta > (T)0) || !std::isfinite(static_cast<double>(beta)))
        throw std::runtime_error("strict FGMRES residual is non-finite");
      copy_vector(outer_.V[0], outer_.r, n,
                  "strict FGMRES basis seed copy");
      scale(outer_.V[0], n, (T)1 / beta);
      reset_outer_scalars(restart);
      outer_.g[0] = std::complex<T>(beta, (T)0);
      const int cycle = std::min(restart, max_iter - iterations);
      int inner = 0;
      auto Hidx = [restart](int row, int col) {
        return static_cast<size_t>(row) * restart + col;
      };

      for (int column = 0; column < cycle; ++column) {
        const int current_iteration = iterations + 1;
        trace_iteration_ = current_iteration;
        T estimate = (T)0;
        T next_norm = (T)0;
        bool stop_after_iteration = false;
        reset_cycle_counters();
        trace_stage(current_iteration, 0, "outer_iteration", [&] {
          precondition_fine(
              fine, outer_.Z[column], outer_.V[column], fine_null_vectors,
              fine_E, fine_X, fine_Y, fine_Z, fine_T, nu_pre, nu_post);
          emit_cycle_counters(current_iteration);
          trace_stage(current_iteration, 0, "fine_matpc", [&] {
            fine_matpc(fine, outer_.w, outer_.Z[column]);
          });

          trace_stage(current_iteration, 0, "arnoldi", [&] {
            const int dot_count = column + 1;
            // V[dot_count] is not a live basis vector until the next norm has
            // been computed, so its first ``dot_count`` elements provide a tiny
            // device result area without changing the advertised arena size.
            dot_many(outer_.V[0], n, outer_.w, dot_count,
                     outer_.V[dot_count]);
            for (int row = 0; row < dot_count; ++row) {
              const LatticeComplex<T> coefficient = outer_.dot_values[row];
              const std::complex<T> h(coefficient.real(), coefficient.imag());
              if (!finite_complex(h))
                throw std::runtime_error(
                    "strict FGMRES Arnoldi dot is non-finite");
              outer_.H[Hidx(row, column)] = h;
            }
            // dot_many() leaves the same coefficients on device in
            // V[dot_count].  Consume them directly, avoiding one host-scalar
            // cuBLAS axpy launch for every previously computed Arnoldi vector.
            strict_orthogonalize_kernel<T>
                <<<blocks(n), _BLOCK_SIZE_, 0, set_->stream>>>(
                    static_cast<LatticeComplex<T> *>(outer_.w),
                    static_cast<const LatticeComplex<T> *>(outer_.V[0]), n,
                    static_cast<const LatticeComplex<T> *>(outer_.V[dot_count]),
                    dot_count, static_cast<int>(n));
            strict_check_cuda(cudaGetLastError(),
                              "strict FGMRES orthogonalization launch");
            next_norm = norm(outer_.w, n);
            outer_.H[Hidx(column + 1, column)] =
                std::complex<T>(next_norm, (T)0);
            if (next_norm > basis_floor) {
              copy_vector(outer_.V[column + 1], outer_.w, n,
                          "strict FGMRES next basis copy");
              scale(outer_.V[column + 1], n, (T)1 / next_norm);
            } else {
              zero_vector(outer_.V[column + 1], n,
                          "strict FGMRES collapsed basis zero");
            }
            apply_givens(column, restart);
          });
          ++iterations;
          inner = column + 1;
          estimate = std::abs(outer_.g[column + 1]);
          if (!std::isfinite(static_cast<double>(estimate)))
            throw std::runtime_error("strict FGMRES estimate is non-finite");
          stop_after_iteration =
              estimate <= threshold || next_norm <= basis_floor;
        });
        trace.iteration(iterations, column + 1, estimate, next_norm);
        if (stop_after_iteration) break;
      }

      if (inner <= 0)
        throw std::runtime_error("strict FGMRES produced an empty cycle");
      trace_stage(iterations, 0, "fgmres_solution_update", [&] {
        update_solution_from_hessenberg(inner, restart);
      });

      // True residual refresh is mandatory at every restart boundary,
      // including an estimated-convergence or happy-breakdown early exit.
      trace_stage(iterations, 0, "true_residual_refresh", [&] {
        fine_matpc(fine, outer_.w, outer_.x);
        strict_subtract_kernel<T>
            <<<blocks(n), _BLOCK_SIZE_, 0, set_->stream>>>(
                outer_.r, outer_.b, outer_.w, static_cast<int>(n));
        residual = norm(outer_.r, n);
      });
      trace.restart(iterations, residual);
      converged = residual <= threshold;
    }

    final_true_residual = residual;
    trace.end(iterations, converged, final_true_residual);
    fine_reconstruct_in_stream(fine, full_out, full_rhs, outer_.x);
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
    StrictPersistentLevel<T> &state = levels_[level];
    void *links = asset(
        level - 1, _SET_PTRS_STRICT_PRECONDITIONED_LINKS_);
    strict_launch_hopping_with_halos(
        scratch, in, links, nullptr, *state.compact_vector_halo,
        *state.compact_link_halo, set_, g.E, g.X, g.Y, g.Z, g.Lt,
        1 - parity_);
    strict_launch_hopping_with_halos(
        out, scratch, links, in, *state.compact_vector_halo,
        *state.compact_link_halo, set_, g.E, g.X, g.Y, g.Z, g.Lt, parity_);
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
    StrictPersistentLevel<T> &state = levels_[level];
    strict_launch_hopping_with_halos(
        out, scratch, links, out, *state.compact_vector_halo,
        *state.compact_link_halo, set_, g.E, g.X, g.Y, g.Z, g.Lt, parity_);
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
    StrictPersistentLevel<T> &state = levels_[level];
    strict_launch_hopping_with_halos(
        scratch, target_solution, links, scratch,
        *state.compact_vector_halo, *state.compact_link_halo,
        set_, g.E, g.X, g.Y, g.Z, g.Lt, 1 - parity_);
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
      dot_pair_global_device(arena_.v, arena_.r, arena_.v, arena_.v, n);
      strict_mr_give_alpha_kernel<T>
          <<<1, 1, 0, set_->stream>>>(
              set_->device_vals, floor, true);
      strict_check_cuda(cudaGetLastError(), "strict coarse MR alpha launch");
      strict_mr_update_device_kernel<T>
          <<<blocks(n), _BLOCK_SIZE_, 0, set_->stream>>>(
              state.x, arena_.r, arena_.v, set_->device_vals,
              static_cast<int>(n));
      strict_check_cuda(cudaGetLastError(), "strict coarse MR update launch");
    }
  }

  bool coarsest_bicgstab_fused(int level, bool &used, int &iterations) {
    used = false;
    iterations = 0;
    // The cooperative kernel has no MPI collectives and therefore cannot
    // provide a distributed coarse solve.
    if (strict_world_size() > 1) return false;
    const char *disable = std::getenv("PYQCU_STRICT_FUSED_COARSE");
    if (disable != nullptr &&
        (disable[0] == '0' || disable[0] == 'n' || disable[0] == 'N'))
      return false;
    if (coarse_fused_status_ == nullptr || arena_.dot_storage == nullptr)
      return false;

    int device = 0;
    strict_check_cuda(cudaGetDevice(&device),
                      "strict fused coarse get device");
    int cooperative_launch = 0;
    strict_check_cuda(cudaDeviceGetAttribute(
                          &cooperative_launch,
                          cudaDevAttrCooperativeLaunch, device),
                      "strict fused coarse cooperative attribute");
    if (cooperative_launch == 0) return false;

    constexpr int kBlockSize = 256;
    int active_blocks_per_sm = 0;
    const cudaError_t occupancy_status =
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm,
            (const void *)strict_coarse_bicgstab_fused_kernel<T, kBlockSize>,
            kBlockSize, 0);
    if (occupancy_status != cudaSuccess || active_blocks_per_sm <= 0) {
      (void)cudaGetLastError();
      return false;
    }

    cudaDeviceProp properties;
    strict_check_cuda(cudaGetDeviceProperties(&properties, device),
                      "strict fused coarse device properties");
    const StrictLevelGeometry &g = levels_[level].geometry;
    const size_t n = g.compact_n;
    if (n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max()))
      return false;
    const int reduction_capacity =
        static_cast<int>(arena_.dot_elements / 2);
    if (reduction_capacity <= 0) return false;
    int grid_size = static_cast<int>(
        (n + static_cast<size_t>(kBlockSize) - 1) /
        static_cast<size_t>(kBlockSize));
    const int resident_limit = std::min(
        active_blocks_per_sm * properties.multiProcessorCount,
        reduction_capacity);
    if (resident_limit <= 0) return false;
    grid_size = std::min(grid_size, resident_limit);
    if (grid_size <= 0) return false;

    StrictPersistentLevel<T> &state = levels_[level];
    const void *links = asset(
        level - 1, _SET_PTRS_STRICT_PRECONDITIONED_LINKS_);
    const int E = g.E;
    const int X = g.X;
    const int Y = g.Y;
    const int Z = g.Z;
    const int Lt = g.Lt;
    const int max_iter = g.max_iter > 0 ? g.max_iter : 100;
    T tolerance = set_->host_argv[_MG_LEVEL1_ATOL_ + level - 1];
    if (!(tolerance > (T)0 && tolerance < (T)1))
      tolerance = std::is_same<T, float>::value ? (T)1e-6 : (T)1e-12;
    void *rhat = arena_.rhat;
    void *r = arena_.r;
    void *p = arena_.p;
    void *v = arena_.v;
    void *s = arena_.s;
    void *t = arena_.t;
    void *tmp = arena_.tmp;
    void *partials = arena_.dot_storage;
    void *status = coarse_fused_status_;
    void *args[] = {
        (void *)&state.x, (void *)&state.pc_rhs, (void *)&rhat,
        (void *)&r, (void *)&p, (void *)&v, (void *)&s, (void *)&t,
        (void *)&tmp, (void *)&links, (void *)&E, (void *)&X, (void *)&Y,
        (void *)&Z, (void *)&Lt, (void *)&parity_, (void *)&max_iter,
        (void *)&tolerance, (void *)&partials, (void *)&grid_size,
        (void *)&status};

    strict_check_cuda(cudaMemsetAsync(
                          coarse_fused_status_, 0, 3 * sizeof(int),
                          set_->stream),
                      "strict fused coarse status reset");
    const cudaError_t launch_status = cudaLaunchCooperativeKernel(
        (const void *)strict_coarse_bicgstab_fused_kernel<T, kBlockSize>,
        dim3(static_cast<unsigned int>(grid_size)), dim3(kBlockSize), args,
        0, set_->stream);
    if (launch_status == cudaErrorNotSupported ||
        launch_status == cudaErrorCooperativeLaunchTooLarge) {
      (void)cudaGetLastError();
      return false;
    }
    strict_check_cuda(launch_status, "strict fused coarse launch");
    used = true;
    strict_check_cuda(cudaStreamSynchronize(set_->stream),
                      "strict fused coarse sync");
    int host_status[3] = {0, 0, 0};
    strict_check_cuda(cudaMemcpy(
                          host_status, coarse_fused_status_,
                          sizeof(host_status), cudaMemcpyDeviceToHost),
                      "strict fused coarse status copy");
    iterations = host_status[2];
    if (host_status[0] != 0 && params_[_VERBOSE_] &&
        params_[_NODE_RANK_] == 0)
      std::printf(
          "PYQCU::SOLVER::STRICT_MG::COARSE:\n "
          "fused BiCGStab breakdown/max_iter guard activated\n");
    return host_status[1] != 0;
  }

  bool coarsest_bicgstab_host(int level, int &iterations) {
    iterations = 0;
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
      iterations = iteration + 1;
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
      LatticeComplex<T> ts;
      LatticeComplex<T> tt;
      dot_pair(t, s, t, t, n, ts, tt);
      if (!finite(tt) || abs2(tt) <= floor) return false;
      omega = ts / tt;
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

  bool coarsest_bicgstab(int level, int &iterations) {
    bool fused = false;
    iterations = 0;
    const bool fused_result =
        coarsest_bicgstab_fused(level, fused, iterations);
    if (fused) return fused_result;
    return coarsest_bicgstab_host(level, iterations);
  }

  void restrict_to_child(int level, const void *fine_compact,
                         void *child_full_rhs) {
    ++cycle_counts_[static_cast<size_t>(level)].restrictions;
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
    ++cycle_counts_[static_cast<size_t>(level)].prolongations;
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
    trace_stage(trace_iteration_, level, "level_prepare", [&] {
      prepare(level, state.pc_rhs, full_rhs, arena_.tmp);
    });

    if (level == num_levels_ - 1) {
      bool converged = false;
      int iterations = 0;
      trace_stage(trace_iteration_, level, "coarsest_bicgstab", [&] {
        converged = coarsest_bicgstab(level, iterations);
      });
      cycle_counts_[static_cast<size_t>(level)].coarse_iterations +=
          static_cast<unsigned long long>(std::max(0, iterations));
      ++cycle_counts_[static_cast<size_t>(level)].coarse_calls;
      trace_residual(level, "after_coarsest_bicgstab", state.x,
                     state.pc_rhs, g.compact_n);
      if (!converged && params_[_VERBOSE_] && params_[_NODE_RANK_] == 0)
        std::printf(
            "PYQCU::SOLVER::STRICT_MG::COARSE:\n "
            "BiCGStab reached breakdown/max_iter; returning finite iterate\n");
    } else {
      cycle_counts_[static_cast<size_t>(level)].pre_smoother +=
          static_cast<unsigned long long>(smoother_steps_);
      trace_stage(trace_iteration_, level, "level_pre_smoother", [&] {
        strict_check_cuda(cudaMemsetAsync(
                              state.x, 0, compact_bytes, set_->stream),
                          "strict smoother x zero");
        strict_check_cuda(cudaMemcpyAsync(
                              arena_.r, state.pc_rhs, compact_bytes,
                              cudaMemcpyDeviceToDevice, set_->stream),
                          "strict smoother rhs copy");
        mr_smooth(level, smoother_steps_);
      });
      trace_residual(level, "after_level_pre_smoother", state.x,
                     state.pc_rhs, g.compact_n);

      StrictPersistentLevel<T> &child = levels_[level + 1];
      ++cycle_counts_[static_cast<size_t>(level + 1)].recursive_solves;
      trace_stage(trace_iteration_, level, "level_restriction", [&] {
        restrict_to_child(level, arena_.r, child.full_rhs);
      });
      trace_stage(trace_iteration_, level + 1, "level_recursive_solve", [&] {
        solve_level(level + 1, child.full_rhs, arena_.correction_full);
      });
      trace_residual(level + 1, "after_level_recursive_solve", child.x,
                     child.pc_rhs, child.geometry.compact_n);
      trace_stage(trace_iteration_, level, "level_prolongation", [&] {
        prolong_from_child(level, arena_.correction_full, arena_.tmp);
        strict_add_kernel<T>
            <<<blocks(g.compact_n), _BLOCK_SIZE_, 0, set_->stream>>>(
                state.x, arena_.tmp, static_cast<int>(g.compact_n));
      });

      // Child recursion intentionally reuses r/v/tmp.  Recompute the parent
      // residual instead of preserving three parent-sized vectors per level.
      trace_stage(trace_iteration_, level, "level_residual_recompute", [&] {
        apply_matpc(level, arena_.v, state.x, arena_.tmp);
        strict_subtract_kernel<T>
            <<<blocks(g.compact_n), _BLOCK_SIZE_, 0, set_->stream>>>(
                arena_.r, state.pc_rhs, arena_.v,
                static_cast<int>(g.compact_n));
      });
      trace_residual(level, "after_level_residual_recompute", state.x,
                     state.pc_rhs, g.compact_n);
      trace_stage(trace_iteration_, level, "level_post_smoother", [&] {
        mr_smooth(level, smoother_steps_);
      });
      cycle_counts_[static_cast<size_t>(level)].post_smoother +=
          static_cast<unsigned long long>(smoother_steps_);
      trace_residual(level, "after_level_post_smoother", state.x,
                     state.pc_rhs, g.compact_n);
    }
    trace_stage(trace_iteration_, level, "level_reconstruct", [&] {
      reconstruct(level, full_out, full_rhs, state.x, arena_.tmp);
    });
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

template <typename T>
size_t strict_launch_fgmres(
    void *full_out, const void *full_rhs, void *gauge, void *clover_ee,
    void *clover_oo, void *clover_ee_inv, void *clover_oo_inv,
    const void *fine_null_vectors, void *set_ptrs, int *params,
    int fine_E, int fine_X, int fine_Y, int fine_Z, int fine_T,
    int coarse_E, int coarse_X, int coarse_Y, int coarse_Z, int coarse_T,
    int restart, int max_iter, T tolerance, int nu_pre, int nu_post,
    size_t max_workspace_bytes, int &iterations, bool &converged,
    T &final_true_residual) {
  long long *table = static_cast<long long *>(set_ptrs);
  if (table[_SET_PTRS_STRICT_HIERARCHY_] == 0)
    throw std::invalid_argument(
        "strict FGMRES requires an initialized slot-80 hierarchy");
  StrictCoarseHierarchy<T> *hierarchy =
      reinterpret_cast<StrictCoarseHierarchy<T> *>(
          table[_SET_PTRS_STRICT_HIERARCHY_]);
  return hierarchy->run_fgmres(
      full_out, full_rhs, gauge, clover_ee, clover_oo, clover_ee_inv,
      clover_oo_inv, fine_null_vectors, fine_E, fine_X, fine_Y, fine_Z,
      fine_T, coarse_E, coarse_X, coarse_Y, coarse_Z, coarse_T, restart,
      max_iter, tolerance, nu_pre, nu_post, max_workspace_bytes, iterations,
      converged, final_true_residual);
}

template <typename T>
bool strict_test_global_reduction_input_valid(
    double local_real, double local_imag, double local_norm2,
    double threshold) {
  const T typed_real = static_cast<T>(local_real);
  const T typed_imag = static_cast<T>(local_imag);
  const T typed_norm2 = static_cast<T>(local_norm2);
  const T typed_threshold = static_cast<T>(threshold);
  return std::isfinite(static_cast<double>(typed_real)) &&
         std::isfinite(static_cast<double>(typed_imag)) &&
         std::isfinite(static_cast<double>(typed_norm2)) &&
         typed_norm2 >= (T)0 &&
         std::isfinite(static_cast<double>(typed_threshold)) &&
         typed_threshold >= (T)0;
}

int strict_test_global_reduction_preflight(
    int data_type, bool local_input_valid, double threshold,
    int &collective_calls) {
  constexpr unsigned int invalid_input = 1u << 0;
  constexpr unsigned int c64_type = 1u << 1;
  constexpr unsigned int c128_type = 1u << 2;
  constexpr unsigned int invalid_type = 1u << 3;

  unsigned int local_flags = local_input_valid ? 0u : invalid_input;
  if (data_type == _LAT_C64_)
    local_flags |= c64_type;
  else if (data_type == _LAT_C128_)
    local_flags |= c128_type;
  else
    local_flags |= invalid_type;

  int mpi_initialized = 0;
  if (MPI_Initialized(&mpi_initialized) != MPI_SUCCESS)
    throw std::runtime_error(
        "strict global reduction preflight cannot query MPI state");
  int world_size = 1;
  if (mpi_initialized) {
    int mpi_finalized = 0;
    if (MPI_Finalized(&mpi_finalized) != MPI_SUCCESS)
      throw std::runtime_error(
          "strict global reduction preflight cannot query MPI finalization");
    if (mpi_finalized)
      throw std::runtime_error(
          "strict global reduction preflight cannot run after MPI finalization");
    if (MPI_Comm_size(MPI_COMM_WORLD, &world_size) != MPI_SUCCESS ||
        world_size < 1)
      throw std::runtime_error(
          "strict global reduction preflight cannot query MPI_COMM_WORLD");
  }

  unsigned int global_flags = local_flags;
  if (world_size > 1) {
    if (MPI_Allreduce(&local_flags, &global_flags, 1, MPI_UNSIGNED, MPI_BOR,
                      MPI_COMM_WORLD) != MPI_SUCCESS)
      throw std::runtime_error(
          "strict global reduction preflight flag Allreduce failed");
    ++collective_calls;
  }

  if ((global_flags & invalid_type) != 0u)
    throw std::invalid_argument(
        "strict global reduction test supports complex64/complex128");
  const unsigned int type_flags = global_flags & (c64_type | c128_type);
  if (type_flags != c64_type && type_flags != c128_type)
    throw std::invalid_argument(
        "strict global reduction test data_type differs between MPI ranks");
  if ((global_flags & invalid_input) != 0u)
    throw std::invalid_argument(
        "strict global reduction test input is invalid on at least one rank");

  if (world_size > 1) {
    std::vector<double> thresholds(static_cast<size_t>(world_size));
    if (MPI_Allgather(&threshold, 1, MPI_DOUBLE, thresholds.data(), 1,
                      MPI_DOUBLE, MPI_COMM_WORLD) != MPI_SUCCESS)
      throw std::runtime_error(
          "strict global reduction preflight threshold Allgather failed");
    ++collective_calls;
    for (int rank = 1; rank < world_size; ++rank) {
      if (thresholds[rank] != thresholds[0])
        throw std::invalid_argument(
            "strict global reduction test threshold differs between MPI ranks");
    }
  }
  return type_flags == c64_type ? _LAT_C64_ : _LAT_C128_;
}

template <typename T>
void strict_test_global_reduction(
    double local_real, double local_imag, double local_norm2,
    double threshold, double &global_real, double &global_imag,
    double &global_norm, int &converged, int &collective_calls) {
  const LatticeComplex<T> local_dot(
      static_cast<T>(local_real), static_cast<T>(local_imag));
  const T typed_norm2 = static_cast<T>(local_norm2);
  const T typed_threshold = static_cast<T>(threshold);
  const LatticeComplex<T> dot =
      strict_global_sum_complex(local_dot, &collective_calls);
  const LatticeComplex<T> norm2 = strict_global_sum_complex(
      LatticeComplex<T>(typed_norm2, (T)0), &collective_calls);
  if (!std::isfinite(static_cast<double>(dot.real())) ||
      !std::isfinite(static_cast<double>(dot.imag())) ||
      !std::isfinite(static_cast<double>(norm2.real())) ||
      norm2.real() < (T)0)
    throw std::runtime_error(
        "strict global reduction test output is invalid");

  const T norm = std::sqrt(norm2.real());
  global_real = static_cast<double>(dot.real());
  global_imag = static_cast<double>(dot.imag());
  global_norm = static_cast<double>(norm);
  converged = norm <= typed_threshold ? 1 : 0;
}

}  // namespace
}  // namespace qcu

extern "C" int testMultigridStrictGlobalReductionQcu(
    int data_type, double local_real, double local_imag,
    double local_norm2, double threshold, double *global_real,
    double *global_imag, double *global_norm, int *converged,
    int *collective_calls) {
  int observed_collective_calls = 0;
  const bool output_pointers_valid =
      global_real != nullptr && global_imag != nullptr &&
      global_norm != nullptr && converged != nullptr &&
      collective_calls != nullptr;
  if (global_real != nullptr)
    *global_real = std::numeric_limits<double>::quiet_NaN();
  if (global_imag != nullptr)
    *global_imag = std::numeric_limits<double>::quiet_NaN();
  if (global_norm != nullptr)
    *global_norm = std::numeric_limits<double>::quiet_NaN();
  if (converged != nullptr)
    *converged = 0;
  if (collective_calls != nullptr)
    *collective_calls = 0;

  try {
    bool local_input_valid = output_pointers_valid;
    if (data_type == _LAT_C64_)
      local_input_valid = local_input_valid &&
          qcu::strict_test_global_reduction_input_valid<float>(
              local_real, local_imag, local_norm2, threshold);
    else if (data_type == _LAT_C128_)
      local_input_valid = local_input_valid &&
          qcu::strict_test_global_reduction_input_valid<double>(
              local_real, local_imag, local_norm2, threshold);
    const int agreed_data_type =
        qcu::strict_test_global_reduction_preflight(
            data_type, local_input_valid, threshold,
            observed_collective_calls);
    if (!output_pointers_valid)
      throw std::invalid_argument(
          "strict global reduction test output pointer is null");

    if (agreed_data_type == _LAT_C64_)
      qcu::strict_test_global_reduction<float>(
          local_real, local_imag, local_norm2, threshold, *global_real,
          *global_imag, *global_norm, *converged,
          observed_collective_calls);
    else
      qcu::strict_test_global_reduction<double>(
          local_real, local_imag, local_norm2, threshold, *global_real,
          *global_imag, *global_norm, *converged,
          observed_collective_calls);
    *collective_calls = observed_collective_calls;
    return 0;
  } catch (const std::exception &error) {
    if (collective_calls != nullptr)
      *collective_calls = observed_collective_calls;
    std::fprintf(stderr,
                 "PYQCU::SOLVER::STRICT_MG::GLOBAL_REDUCTION_TEST: %s\n",
                 error.what());
    return 1;
  }
}

extern "C" int applyMultigridStrictVCycleQcu(
    long long full_out, long long full_rhs, long long set_ptrs,
    long long params, int start_level,
    unsigned long long *allocated_bytes) {
  try {
    int *host_params = reinterpret_cast<int *>(params);
    qcu::strict_validate_mpi_backend(host_params);
    if (allocated_bytes == nullptr)
      throw std::invalid_argument("strict V-cycle byte output is null");
    *allocated_bytes = 0;
    qcu::strict_check_cuda(
        cudaDeviceSynchronize(), "strict V-cycle input sync");
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
    int *host_params = reinterpret_cast<int *>(params);
    qcu::strict_validate_mpi_backend(host_params);
    if (allocated_bytes == nullptr)
      throw std::invalid_argument("strict hierarchy byte output is null");
    *allocated_bytes = 0;
    qcu::strict_check_cuda(
        cudaDeviceSynchronize(), "strict hierarchy init input sync");
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
    qcu::strict_validate_mpi_backend(host_params);
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

extern "C" int applyMultigridStrictFgmresQcu(
    long long full_out, long long full_rhs, long long gauge,
    long long clover_ee, long long clover_oo, long long clover_ee_inv,
    long long clover_oo_inv, long long fine_null_vectors,
    long long set_ptrs, long long params, int fine_E, int fine_X,
    int fine_Y, int fine_Z, int fine_T, int coarse_E, int coarse_X,
    int coarse_Y, int coarse_Z, int coarse_T, int element_bytes,
    int restart, int max_iter, double tolerance, int nu_pre, int nu_post,
    unsigned long long max_workspace_bytes, int *iterations,
    int *converged, double *final_true_residual,
    unsigned long long *allocated_bytes) {
  try {
    int *host_params = reinterpret_cast<int *>(params);
    qcu::strict_validate_mpi_backend(host_params);
    if (iterations == nullptr || converged == nullptr ||
        final_true_residual == nullptr || allocated_bytes == nullptr)
      throw std::invalid_argument("strict FGMRES result pointer is null");
    *iterations = 0;
    *converged = 0;
    *final_true_residual = std::numeric_limits<double>::infinity();
    *allocated_bytes = 0;
    if (max_workspace_bytes > static_cast<unsigned long long>(
                                  std::numeric_limits<size_t>::max()))
      throw std::overflow_error("strict FGMRES budget exceeds size_t");
    qcu::strict_check_cuda(
        cudaDeviceSynchronize(), "strict FGMRES input sync");
    size_t bytes = 0;
    bool did_converge = false;
    if (host_params[_DATA_TYPE_] == _LAT_C64_) {
      if (element_bytes != static_cast<int>(sizeof(qcu::LatticeComplex<float>)))
        throw std::invalid_argument("strict FGMRES complex64 dtype mismatch");
      float residual = std::numeric_limits<float>::infinity();
      bytes = qcu::strict_launch_fgmres<float>(
          reinterpret_cast<void *>(full_out),
          reinterpret_cast<const void *>(full_rhs),
          reinterpret_cast<void *>(gauge), reinterpret_cast<void *>(clover_ee),
          reinterpret_cast<void *>(clover_oo),
          reinterpret_cast<void *>(clover_ee_inv),
          reinterpret_cast<void *>(clover_oo_inv),
          reinterpret_cast<const void *>(fine_null_vectors),
          reinterpret_cast<void *>(set_ptrs), host_params,
          fine_E, fine_X, fine_Y, fine_Z, fine_T,
          coarse_E, coarse_X, coarse_Y, coarse_Z, coarse_T,
          restart, max_iter, static_cast<float>(tolerance), nu_pre, nu_post,
          static_cast<size_t>(max_workspace_bytes), *iterations,
          did_converge, residual);
      *final_true_residual = static_cast<double>(residual);
    } else if (host_params[_DATA_TYPE_] == _LAT_C128_) {
      if (element_bytes != static_cast<int>(sizeof(qcu::LatticeComplex<double>)))
        throw std::invalid_argument("strict FGMRES complex128 dtype mismatch");
      double residual = std::numeric_limits<double>::infinity();
      bytes = qcu::strict_launch_fgmres<double>(
          reinterpret_cast<void *>(full_out),
          reinterpret_cast<const void *>(full_rhs),
          reinterpret_cast<void *>(gauge), reinterpret_cast<void *>(clover_ee),
          reinterpret_cast<void *>(clover_oo),
          reinterpret_cast<void *>(clover_ee_inv),
          reinterpret_cast<void *>(clover_oo_inv),
          reinterpret_cast<const void *>(fine_null_vectors),
          reinterpret_cast<void *>(set_ptrs), host_params,
          fine_E, fine_X, fine_Y, fine_Z, fine_T,
          coarse_E, coarse_X, coarse_Y, coarse_Z, coarse_T,
          restart, max_iter, tolerance, nu_pre, nu_post,
          static_cast<size_t>(max_workspace_bytes), *iterations,
          did_converge, residual);
      *final_true_residual = residual;
    } else {
      throw std::invalid_argument(
          "strict FGMRES supports complex64/complex128");
    }
    *converged = did_converge ? 1 : 0;
    *allocated_bytes = static_cast<unsigned long long>(bytes);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "PYQCU::SOLVER::STRICT_MG::FGMRES: %s\n",
                 error.what());
    return 1;
  }
}

extern "C" int applyMultigridStrictCoarseQcu(
    long long out, long long in, long long links, long long onsite_pair,
    long long set_ptrs, long long params, int E, int X, int Y, int Z, int T,
    int onsite_index) {
  try {
    int *host_params = reinterpret_cast<int *>(params);
    qcu::strict_validate_mpi_backend(host_params);
    qcu::strict_check_cuda(cudaDeviceSynchronize(), "coarse input sync");
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
    int *host_params = reinterpret_cast<int *>(params);
    qcu::strict_validate_mpi_backend(host_params);
    qcu::strict_check_cuda(cudaDeviceSynchronize(), "MATPC input sync");
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

extern "C" int applyMultigridStrictFineMatPCQcu(
    long long out, long long in, long long gauge, long long clover_ee,
    long long clover_oo, long long clover_ee_inv, long long clover_oo_inv,
    long long set_ptrs, long long params, int parity) {
  try {
    int *host_params = reinterpret_cast<int *>(params);
    qcu::strict_validate_mpi_backend(host_params);
    qcu::strict_check_cuda(cudaDeviceSynchronize(),
                           "fine MATPC input sync");
    if (host_params[_DATA_TYPE_] == _LAT_C64_)
      qcu::strict_launch_fine_matpc<float>(
          reinterpret_cast<void *>(out), reinterpret_cast<const void *>(in),
          reinterpret_cast<void *>(gauge), reinterpret_cast<void *>(clover_ee),
          reinterpret_cast<void *>(clover_oo),
          reinterpret_cast<void *>(clover_ee_inv),
          reinterpret_cast<void *>(clover_oo_inv),
          reinterpret_cast<void *>(set_ptrs), host_params, parity);
    else if (host_params[_DATA_TYPE_] == _LAT_C128_)
      qcu::strict_launch_fine_matpc<double>(
          reinterpret_cast<void *>(out), reinterpret_cast<const void *>(in),
          reinterpret_cast<void *>(gauge), reinterpret_cast<void *>(clover_ee),
          reinterpret_cast<void *>(clover_oo),
          reinterpret_cast<void *>(clover_ee_inv),
          reinterpret_cast<void *>(clover_oo_inv),
          reinterpret_cast<void *>(set_ptrs), host_params, parity);
    else
      throw std::invalid_argument(
          "strict fine MATPC supports complex64/complex128");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "PYQCU::SOLVER::STRICT_MG::FINE_MATPC: %s\n",
                 error.what());
    return 1;
  }
}

extern "C" int applyMultigridStrictPrepareQcu(
    long long out, long long full_rhs, long long links, long long onsite_pair,
    long long scratch, long long set_ptrs, long long params,
    int E, int X, int Y, int Z, int T, int parity) {
  try {
    int *host_params = reinterpret_cast<int *>(params);
    qcu::strict_validate_mpi_backend(host_params);
    qcu::strict_check_cuda(cudaDeviceSynchronize(), "prepare input sync");
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
    int *host_params = reinterpret_cast<int *>(params);
    qcu::strict_validate_mpi_backend(host_params);
    qcu::strict_check_cuda(cudaDeviceSynchronize(), "reconstruct input sync");
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
    int *host_params = reinterpret_cast<int *>(params);
    qcu::strict_validate_mpi_backend(host_params);
    qcu::strict_check_cuda(cudaDeviceSynchronize(), "restrict input sync");
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
    int *host_params = reinterpret_cast<int *>(params);
    qcu::strict_validate_mpi_backend(host_params);
    qcu::strict_check_cuda(cudaDeviceSynchronize(), "prolong input sync");
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
