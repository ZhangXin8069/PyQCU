#include "backend.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#include <omp.h>

namespace kupl::qcu {
namespace {

constexpr int kDim = 4;
constexpr int kSpin = 4;
constexpr int kColor = 3;
constexpr int kVector = kSpin * kColor;
constexpr int kOnsite = kVector * kVector;
constexpr int kGauge = kColor * kColor;

template <typename Real> struct Parameters {
  std::array<int32_t, kParamsSize> ints{};
  std::array<Real, kArgvSize> reals{};

  int get(int index) const { return static_cast<int>(ints[index]); }
};

template <typename Real> struct Geometry {
  std::array<int, kDim> local{};
  std::array<int, kDim> grid{};
  std::array<int, kDim> coord{};
  bool laplacian = false;
  int rank = 0;
  int size = 1;
  int64_t half_sites = 0;
  int64_t parity_vectors = 0;
  std::array<int64_t, kDim> face_sites{};
  std::array<std::array<int64_t, kDim>, kDim> corner_sites{};
  std::array<int, kDim> neighbor_plus{};
  std::array<int, kDim> neighbor_minus{};
  MPI_Comm comm = MPI_COMM_NULL;

  Geometry() = default;
  Geometry(const Geometry &) = delete;
  Geometry &operator=(const Geometry &) = delete;

  Geometry(Geometry &&other) noexcept { moveFrom(other); }
  Geometry &operator=(Geometry &&other) noexcept {
    if (this != &other) {
      release();
      moveFrom(other);
    }
    return *this;
  }

  ~Geometry() { release(); }

  void moveFrom(Geometry &other) {
    local = other.local;
    grid = other.grid;
    coord = other.coord;
    laplacian = other.laplacian;
    rank = other.rank;
    size = other.size;
    half_sites = other.half_sites;
    parity_vectors = other.parity_vectors;
    face_sites = other.face_sites;
    corner_sites = other.corner_sites;
    neighbor_plus = other.neighbor_plus;
    neighbor_minus = other.neighbor_minus;
    comm = other.comm;
    other.comm = MPI_COMM_NULL;
  }

  void release() {
    if (comm != MPI_COMM_NULL) {
      MPI_Comm_free(&comm);
      comm = MPI_COMM_NULL;
    }
  }

  int64_t siteLinear(int x, int y, int z, int t2) const {
    if (laplacian) {
      return (static_cast<int64_t>(x) * local[1] + y) * local[2] + z;
    }
    return ((static_cast<int64_t>(x) * local[1] + y) * local[2] + z) *
               (local[3] / 2) +
           t2;
  }

  int64_t faceIndex(int dim, int x, int y, int z, int t2) const {
    const int coordinate[kDim] = {x, y, z, t2};
    const int extent[kDim] = {
        local[0], local[1], local[2],
        laplacian ? local[3] : local[3] / 2};
    int64_t value = 0;
    int64_t stride = 1;
    for (int axis = kDim - 1; axis >= 0; --axis) {
      if (axis == dim) {
        continue;
      }
      value += coordinate[axis] * stride;
      stride *= extent[axis];
    }
    return value;
  }

  int64_t cornerIndex(int dim0, int dim1, int x, int y, int z,
                      int t2) const {
    const int coordinate[kDim] = {x, y, z, t2};
    const int extent[kDim] = {
        local[0], local[1], local[2],
        laplacian ? local[3] : local[3] / 2};
    int64_t value = 0;
    int64_t stride = 1;
    for (int axis = kDim - 1; axis >= 0; --axis) {
      if (axis == dim0 || axis == dim1) {
        continue;
      }
      value += coordinate[axis] * stride;
      stride *= extent[axis];
    }
    return value;
  }

  void decodeHalf(int64_t index, int &x, int &y, int &z, int &t2) const {
    if (laplacian) {
      t2 = 0;
      z = static_cast<int>(index % local[2]);
      index /= local[2];
      y = static_cast<int>(index % local[1]);
      index /= local[1];
      x = static_cast<int>(index);
      return;
    }
    const int ht = local[3] / 2;
    t2 = static_cast<int>(index % ht);
    index /= ht;
    z = static_cast<int>(index % local[2]);
    index /= local[2];
    y = static_cast<int>(index % local[1]);
    index /= local[1];
    x = static_cast<int>(index);
  }

  int parityAt(int x, int y, int z, int t) const {
    return (x + y + z + t) & 1;
  }

  int timeForParity(int x, int y, int z, int t2, int parity) const {
    return 2 * t2 + (((x + y + z) & 1) ^ parity);
  }

  int globalCoordinate(int dim, int local_coordinate) const {
    return coord[dim] * local[dim] + local_coordinate;
  }

  int lastHalfCoordinate(int dim) const {
    if (dim == 3) {
      return (laplacian ? local[3] : local[3] / 2) - 1;
    }
    return local[dim] - 1;
  }
};

template <typename Real> struct GaugeHalo {
  const Complex<Real> *source = nullptr;
  std::array<std::vector<Complex<Real>>, 8> faces;
  std::array<std::array<std::vector<Complex<Real>>, 4>, 6> corners;
  bool ready = false;
};

template <typename Real> struct GaugeView {
  const Complex<Real> *data = nullptr;
  int64_t matrix_stride = 1;
};

template <typename Real> struct VectorView {
  const Complex<Real> *base = nullptr;
  int64_t stride = 1;

  Complex<Real> operator[](int component) const {
    return base[static_cast<int64_t>(component) * stride];
  }
};

std::mutex &registryMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<const void *, std::shared_ptr<BackendBase>> &registry() {
  static std::unordered_map<const void *, std::shared_ptr<BackendBase>> value;
  return value;
}

int largestPowerOfTwo(int value) {
  if (value <= 1) {
    return 1;
  }
  unsigned int v = static_cast<unsigned int>(value);
  v |= v >> 1U;
  v |= v >> 2U;
  v |= v >> 4U;
  v |= v >> 8U;
  v |= v >> 16U;
  return static_cast<int>(v - (v >> 1U));
}

#if defined(__linux__)
const std::vector<int> &initialAffinityCpus() {
  static const std::vector<int> cpus = [] {
    std::vector<int> result;
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
      for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &set)) {
          result.push_back(cpu);
        }
      }
    }
    return result;
  }();
  return cpus;
}
#endif

int configuredThreads() {
#if defined(__linux__)
  const auto &affinity = initialAffinityCpus();
#endif
  int requested = omp_get_max_threads();
  const char *text = std::getenv("KUPL_OMP_THREADS");
  const bool explicit_request = text != nullptr;
  if (explicit_request) {
    requested = std::max(1, std::atoi(text));
  }
  requested = largestPowerOfTwo(requested);
#if defined(__linux__)
  if (!explicit_request && !affinity.empty()) {
    requested = std::min(
        requested, largestPowerOfTwo(static_cast<int>(affinity.size())));
  }
#endif
  omp_set_dynamic(0);
  omp_set_num_threads(std::max(1, requested));
  return std::max(1, requested);
}

void bindLogicalNuma() {
  const char *enabled = std::getenv("KUPL_NUMA_BIND");
  if (enabled == nullptr || std::atoi(enabled) == 0) {
    return;
  }
#if defined(__linux__)
  const char *omp_bind = std::getenv("OMP_PROC_BIND");
  if (omp_bind != nullptr && omp_bind[0] != '\0') {
    throw std::runtime_error(
        "KUPL_NUMA_BIND=1 owns CPU placement; unset OMP_PROC_BIND "
        "(use mpirun --bind-to none)");
  }
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    return;
  }
  MPI_Comm local_comm = MPI_COMM_NULL;
  if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                          MPI_INFO_NULL, &local_comm) != MPI_SUCCESS) {
    return;
  }
  int local_rank = 0;
  int local_size = 1;
  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_size(local_comm, &local_size);
  MPI_Comm_free(&local_comm);

  const std::vector<int> &cpus = initialAffinityCpus();
  if (cpus.empty()) {
    return;
  }
  int per_rank = std::max(1, static_cast<int>(cpus.size()) / local_size);
  per_rank = std::min(per_rank, configuredThreads());
  per_rank = largestPowerOfTwo(per_rank);
  const int begin = std::min(static_cast<int>(cpus.size()),
                             local_rank * per_rank);
  const int end = std::min(static_cast<int>(cpus.size()), begin + per_rank);
  cpu_set_t bound;
  CPU_ZERO(&bound);
  for (int index = begin; index < end; ++index) {
    CPU_SET(cpus[static_cast<std::size_t>(index)], &bound);
  }
  if (begin < end) {
    if (sched_setaffinity(0, sizeof(bound), &bound) != 0) {
      throw std::runtime_error("KUPL_NUMA_BIND failed to set CPU affinity");
    }
  }
#endif
}

template <typename Real>
Geometry<Real> makeGeometry(const int32_t *params) {
  Geometry<Real> geometry;
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    throw std::runtime_error("MPI must be initialized before applyInitQcu");
  }
  MPI_Comm_size(MPI_COMM_WORLD, &geometry.size);
  MPI_Comm_rank(MPI_COMM_WORLD, &geometry.rank);
  geometry.laplacian = params[kSetPlan] == kSetPlanLaplacian;

  for (int dim = 0; dim < kDim; ++dim) {
    geometry.local[dim] = params[kLatX + dim];
    geometry.grid[dim] = params[kGridX + dim];
    if (geometry.local[dim] <= 0) {
      throw std::runtime_error("lattice dimensions must be positive");
    }
    if (!geometry.laplacian && (geometry.local[dim] & 1) != 0) {
      throw std::runtime_error(
          "all local lattice dimensions must be even");
    }
  }
  if (geometry.laplacian) {
    geometry.local[3] = 1;
    geometry.grid[3] = 1;
  }
  int64_t grid_product = 1;
  const int grid_dims = geometry.laplacian ? 3 : kDim;
  for (int dim = 0; dim < grid_dims; ++dim) {
    const int value = geometry.grid[dim];
    if (value <= 0) {
      grid_product = -1;
      break;
    }
    grid_product *= value;
  }
  if (grid_product != geometry.size) {
    throw std::runtime_error(
        "process-grid product does not equal MPI_COMM_WORLD size");
  }

  const int periods[kDim] = {1, 1, 1, 1};
  int mpi_error =
      MPI_Cart_create(MPI_COMM_WORLD, kDim, geometry.grid.data(),
                      periods, 0, &geometry.comm);
  if (mpi_error != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Cart_create failed");
  }
  MPI_Comm_rank(geometry.comm, &geometry.rank);
  MPI_Cart_coords(geometry.comm, geometry.rank, kDim, geometry.coord.data());

  const int64_t time_extent =
      geometry.laplacian ? 1 : geometry.local[3] / 2;
  geometry.half_sites = static_cast<int64_t>(geometry.local[0]) *
                        geometry.local[1] * geometry.local[2] * time_extent;
  geometry.parity_vectors = geometry.half_sites * kVector;

  for (int dim = 0; dim < kDim; ++dim) {
    const int64_t extent = dim == 3
                               ? (geometry.laplacian ? 1
                                                     : geometry.local[3] / 2)
                               : geometry.local[dim];
    int64_t count = geometry.half_sites / extent;
    geometry.face_sites[dim] = count;
    MPI_Cart_shift(geometry.comm, dim, 1,
                   &geometry.neighbor_minus[dim],
                   &geometry.neighbor_plus[dim]);
  }
  for (int dim0 = 0; dim0 < kDim; ++dim0) {
    for (int dim1 = dim0 + 1; dim1 < kDim; ++dim1) {
      const int64_t extent0 =
          dim0 == 3
              ? (geometry.laplacian ? 1 : geometry.local[3] / 2)
              : geometry.local[dim0];
      const int64_t extent1 =
          dim1 == 3
              ? (geometry.laplacian ? 1 : geometry.local[3] / 2)
              : geometry.local[dim1];
      geometry.corner_sites[dim0][dim1] =
          geometry.half_sites / (extent0 * extent1);
    }
  }
  return geometry;
}

int planeIndex(int dim0, int dim1) {
  static constexpr int planes[kDim][kDim] = {
      {-1, 0, 1, 2},
      {-1, -1, 3, 4},
      {-1, -1, -1, 5},
      {-1, -1, -1, -1},
  };
  return planes[dim0][dim1];
}

int signIndex(int sign) { return sign > 0 ? 1 : 0; }

int faceOffset(int dim, int sign) { return dim * 2 + signIndex(sign); }

int cornerTag(int dim0, int dim1, int sign0, int sign1) {
  return 3000 + planeIndex(dim0, dim1) * 100 +
         (sign0 > 0 ? 10 : 0) + (sign1 > 0 ? 1 : 0);
}

int faceTag(int dim, int boundary_sign) {
  return 2000 + dim * 10 + (boundary_sign > 0 ? 2 : 1);
}

int fermionFaceTag(int dim, int boundary_sign) {
  return 4000 + dim * 10 + (boundary_sign > 0 ? 2 : 1);
}

int laplacianFaceTag(int kind, int dim, int boundary_sign) {
  return 5000 + kind * 100 + dim * 10 + (boundary_sign > 0 ? 2 : 1);
}

template <typename Real>
Complex<Real> gammaEntry(int direction, int row, int column) {
  using C = Complex<Real>;
  const C i(0, 1);
  if (direction == 0) {
    static const C matrix[16] = {
        C(0), C(0), C(0), i, C(0), C(0), i, C(0),
        C(0), -i, C(0), C(0), -i, C(0), C(0), C(0),
    };
    return matrix[row * 4 + column];
  }
  if (direction == 1) {
    static const C matrix[16] = {
        C(0), C(0), C(0), C(-1), C(0), C(0), C(1), C(0),
        C(0), C(1), C(0), C(0), C(-1), C(0), C(0), C(0),
    };
    return matrix[row * 4 + column];
  }
  if (direction == 2) {
    static const C matrix[16] = {
        C(0), C(0), i, C(0), C(0), C(0), C(0), -i,
        -i, C(0), C(0), C(0), C(0), i, C(0), C(0),
    };
    return matrix[row * 4 + column];
  }
  static const C matrix[16] = {
      C(0), C(0), C(1), C(0), C(0), C(0), C(0), C(1),
      C(1), C(0), C(0), C(0), C(0), C(1), C(0), C(0),
  };
  return matrix[row * 4 + column];
}

template <typename Real>
void applyGammaSpin(int direction, const Complex<Real> *input,
                    Complex<Real> *output) {
  for (int row = 0; row < 4; ++row) {
    output[row] = Complex<Real>(0);
    for (int column = 0; column < 4; ++column) {
      output[row] += gammaEntry<Real>(direction, row, column) *
                     input[column];
    }
  }
}

template <typename Real>
void multiplyGaugeSpin(const Complex<Real> *matrix,
                       const Complex<Real> in[kVector],
                       Complex<Real> out[kVector], bool dagger) {
  for (int spin = 0; spin < kSpin; ++spin) {
    for (int row = 0; row < kColor; ++row) {
      Complex<Real> value(0);
      for (int column = 0; column < kColor; ++column) {
        const Complex<Real> entry =
            dagger ? std::conj(matrix[column * kColor + row])
                   : matrix[row * kColor + column];
        value += entry * in[spin * kColor + column];
      }
      out[spin * kColor + row] = value;
    }
  }
}

template <typename Real>
void applyOneMinusGamma(int direction, const Complex<Real> in[kVector],
                        Complex<Real> out[kVector], int gamma_sign) {
  using C = Complex<Real>;
  for (int spin = 0; spin < kSpin; ++spin) {
    for (int color = 0; color < kColor; ++color) {
      out[spin * kColor + color] = in[spin * kColor + color];
    }
  }
  C gamma[4];
  for (int color = 0; color < kColor; ++color) {
    C column[4];
    for (int spin = 0; spin < kSpin; ++spin) {
      column[spin] = in[spin * kColor + color];
    }
    applyGammaSpin(direction, column, gamma);
    for (int spin = 0; spin < kSpin; ++spin) {
      out[spin * kColor + color] -=
          static_cast<Real>(gamma_sign) * gamma[spin];
    }
  }
}

template <typename Real>
void multiply3(const Complex<Real> *left, const Complex<Real> *right,
               Complex<Real> *out) {
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      Complex<Real> value(0);
      for (int inner = 0; inner < 3; ++inner) {
        value += left[row * 3 + inner] * right[inner * 3 + column];
      }
      out[row * 3 + column] = value;
    }
  }
}

inline uint64_t mix64(uint64_t value) {
  value += UINT64_C(0x9e3779b97f4a7c15);
  value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31U);
}

template <typename Real>
Real deterministicUniform(uint64_t seed, uint64_t global_site, int parity,
                          int direction, int channel) {
  uint64_t key = mix64(seed ^ global_site);
  key ^= mix64((static_cast<uint64_t>(parity) * kDim + direction) * 8U +
               static_cast<uint64_t>(channel));
  const uint64_t bits = mix64(key) >> 11U;
  return static_cast<Real>(
      (static_cast<long double>(bits) + 0.5L) /
      9007199254740992.0L);
}

template <typename Real>
Real deterministicNormal(uint64_t seed, uint64_t global_site, int parity,
                         int direction, int channel) {
  const Real first = std::max(
      deterministicUniform<Real>(
          seed, global_site, parity, direction, 2 * channel),
      std::numeric_limits<Real>::min());
  const Real second = deterministicUniform<Real>(
      seed, global_site, parity, direction, 2 * channel + 1);
  constexpr Real two_pi = Real(6.2831853071795864769);
  return std::sqrt(Real(-2) * std::log(first)) *
         std::cos(two_pi * second);
}

template <typename Real>
Real matrixInfinityNorm(const Complex<Real> matrix[9]) {
  Real norm = 0;
  for (int row = 0; row < 3; ++row) {
    Real row_sum = 0;
    for (int column = 0; column < 3; ++column) {
      row_sum += std::abs(matrix[row * 3 + column]);
    }
    norm = std::max(norm, row_sum);
  }
  return norm;
}

template <typename Real>
void su3Exponential(const Complex<Real> input[9], Complex<Real> output[9]) {
  using C = Complex<Real>;
  C scaled[9];
  std::copy(input, input + 9, scaled);
  int squarings = 0;
  while (matrixInfinityNorm(scaled) > Real(0.25) && squarings < 32) {
    for (int index = 0; index < 9; ++index) {
      scaled[index] *= Real(0.5);
    }
    ++squarings;
  }

  C term[9] = {C(1), C(0), C(0), C(0), C(1), C(0), C(0), C(0), C(1)};
  std::copy(term, term + 9, output);
  C next[9];
  for (int order = 1; order <= 14; ++order) {
    multiply3(term, scaled, next);
    const Real inverse_order = Real(1) / static_cast<Real>(order);
    for (int index = 0; index < 9; ++index) {
      term[index] = next[index] * inverse_order;
      output[index] += term[index];
    }
  }
  for (int square = 0; square < squarings; ++square) {
    multiply3(output, output, next);
    std::copy(next, next + 9, output);
  }
}

template <typename Real>
void makeRandomSu3(uint64_t seed, uint64_t global_site, int parity,
                   int direction, Real sigma, Complex<Real> output[9]) {
  using C = Complex<Real>;
  static constexpr Real gellMann[8][9] = {
      {0, 1, 0, 1, 0, 0, 0, 0, 0},
      {0, -1, 0, 1, 0, 0, 0, 0, 0},
      {1, 0, 0, 0, -1, 0, 0, 0, 0},
      {0, 0, 1, 0, 0, 0, 1, 0, 0},
      {0, 0, -1, 0, 0, 0, 1, 0, 0},
      {0, 0, 0, 0, 0, 1, 0, 1, 0},
      {0, 0, 0, 0, 0, -1, 0, 1, 0},
      {Real(0.5773502691896257), 0, 0, 0,
       Real(0.5773502691896257), 0, 0, 0, Real(-1.1547005383792515)},
  };
  C hermitian[9] = {C(0)};
  for (int generator = 0; generator < 8; ++generator) {
    const Real sample = deterministicNormal<Real>(
        seed, global_site, parity, direction, generator);
    const bool imaginary = generator == 1 || generator == 4 || generator == 6;
    for (int entry = 0; entry < 9; ++entry) {
      const Real coefficient = sample * gellMann[generator][entry];
      if (imaginary) {
        hermitian[entry] += C(0, coefficient);
      } else {
        hermitian[entry] += C(coefficient, 0);
      }
    }
  }
  C generator[9];
  for (int entry = 0; entry < 9; ++entry) {
    generator[entry] =
        C(-sigma * hermitian[entry].imag(),
          sigma * hermitian[entry].real());
  }
  su3Exponential(generator, output);
}

constexpr int kCoarseDirections = 32;

int coarseShiftComponent(int direction, int dim) {
  static constexpr int shifts[kCoarseDirections][kDim] = {
      {1, 0, 0, 0},  {-1, 0, 0, 0}, {0, 1, 0, 0},  {0, -1, 0, 0},
      {0, 0, 1, 0},  {0, 0, -1, 0}, {0, 0, 0, 1},  {0, 0, 0, -1},
      {1, 1, 0, 0},  {-1, -1, 0, 0}, {1, -1, 0, 0}, {-1, 1, 0, 0},
      {1, 0, 1, 0},  {-1, 0, -1, 0}, {1, 0, -1, 0}, {-1, 0, 1, 0},
      {1, 0, 0, 1},  {-1, 0, 0, -1}, {1, 0, 0, -1}, {-1, 0, 0, 1},
      {0, 1, 1, 0},  {0, -1, -1, 0}, {0, 1, -1, 0}, {0, -1, 1, 0},
      {0, 1, 0, 1},  {0, -1, 0, -1}, {0, 1, 0, -1}, {0, -1, 0, 1},
      {0, 0, 1, 1},  {0, 0, -1, -1}, {0, 0, 1, -1}, {0, 0, -1, 1}};
  return shifts[direction][dim];
}

int coarseDirectionForShift(const int shift[kDim]) {
  for (int direction = 0; direction < kCoarseDirections; ++direction) {
    bool matches = true;
    for (int dim = 0; dim < kDim; ++dim) {
      if (coarseShiftComponent(direction, dim) != shift[dim]) {
        matches = false;
        break;
      }
    }
    if (matches) {
      return direction;
    }
  }
  return -1;
}

template <typename Real>
int64_t ompChunk(int64_t count) {
  const int threads = std::max(1, omp_get_max_threads());
  const int64_t alignment = std::max<int64_t>(1, 64 / sizeof(Complex<Real>));
  int64_t chunk = std::max<int64_t>(alignment, count / threads);
  chunk = ((chunk + alignment - 1) / alignment) * alignment;
  return std::min(count, std::max<int64_t>(1, chunk));
}

template <typename Real> class Backend final : public BackendBase {
public:
  Backend(const int32_t *params, const Real *argv)
      : geometry_(makeGeometry<Real>(params)),
        threads_(configuredThreads()) {
    std::copy(params, params + kParamsSize, parameters_.ints.begin());
    std::copy(argv, argv + kArgvSize, parameters_.reals.begin());
    data_type_ = parameters_.get(kDataType);
    if (data_type_ != kLatC64 && data_type_ != kLatC128) {
      throw std::runtime_error("KUPL supports only c64 and c128");
    }
    allocateScratch();
    log("INIT", "rank=%d size=%d grid=%d,%d,%d,%d local=%d,%d,%d,%d "
                "threads=%d csw=%g",
        geometry_.rank, geometry_.size, geometry_.grid[0], geometry_.grid[1],
        geometry_.grid[2], geometry_.grid[3], geometry_.local[0],
        geometry_.local[1], geometry_.local[2], geometry_.local[3], threads_,
        static_cast<double>(csw()));
  }

  void updateParams(const int32_t *params) override {
    std::copy(params, params + kParamsSize, parameters_.ints.begin());
    validateParity();
  }

  void wilsonDslash(const void *out, const void *in,
                    const void *gauge) override {
    requireWilsonMode();
    const int target = targetParity();
    ensureGaugeHalo(static_cast<const Complex<Real> *>(gauge));
    bareDslash(static_cast<Complex<Real> *>(const_cast<void *>(out)),
               static_cast<const Complex<Real> *>(in), target);
  }

  void wilsonBistabCg(void *out, const void *in, const void *gauge) override {
    requireSolverMode();
    ensureGaugeHalo(static_cast<const Complex<Real> *>(gauge));
    solveWilsonBistabCg(static_cast<Complex<Real> *>(out),
                        static_cast<const Complex<Real> *>(in));
  }

  void wilsonBistabCgDslash(void *out, const void *in,
                            const void *gauge) override {
    requireSolverMode();
    requireWilsonGauge(static_cast<const Complex<Real> *>(gauge));
    applyWilsonSchur(static_cast<const Complex<Real> *>(in),
                     static_cast<Complex<Real> *>(out), targetParity(),
                     work_a_.data(), work_b_.data());
  }

  void wilsonCg(void *out, const void *in, const void *gauge) override {
    requireSolverMode();
    ensureGaugeHalo(static_cast<const Complex<Real> *>(gauge));
    solveWilsonCg(static_cast<Complex<Real> *>(out),
                  static_cast<const Complex<Real> *>(in));
  }

  void wilsonCgDslash(void *out, const void *in, const void *gauge) override {
    requireSolverMode();
    requireWilsonGauge(static_cast<const Complex<Real> *>(gauge));
    applyWilsonCgDslash(static_cast<const Complex<Real> *>(in),
                        static_cast<Complex<Real> *>(out), targetParity());
  }

  void clover(void *clover_out, const void *gauge) override {
    requireWilsonMode();
    ensureGaugeHalo(static_cast<const Complex<Real> *>(gauge));
    computeClover(static_cast<Complex<Real> *>(clover_out), targetParity());
  }

  void clovers(void *clover_out, void *clover_inv,
               const void *gauge) override {
    requireWilsonMode();
    ensureGaugeHalo(static_cast<const Complex<Real> *>(gauge));
    const int parity = targetParity();
    auto *term = static_cast<Complex<Real> *>(clover_out);
    computeClover(term, parity);
    invertClover(term, static_cast<Complex<Real> *>(clover_inv));
  }

  void cloverDslash(void *out, const void *in, const void *gauge) override {
    requireWilsonMode();
    const int target = targetParity();
    ensureGaugeHalo(static_cast<const Complex<Real> *>(gauge));
    ensureCachedClover(static_cast<const Complex<Real> *>(gauge));
    bareDslash(static_cast<Complex<Real> *>(out),
               static_cast<const Complex<Real> *>(in), target);
    applyOnsite(static_cast<const Complex<Real> *>(out),
                cached_clover_inv_[static_cast<std::size_t>(target)].data(),
                static_cast<Complex<Real> *>(out));
  }

  void suppliedCloverDslash(void *out, const void *in, const void *gauge,
                            const void *clover) override {
    requireWilsonMode();
    const int target = targetParity();
    ensureGaugeHalo(static_cast<const Complex<Real> *>(gauge));
    bareDslash(static_cast<Complex<Real> *>(out),
               static_cast<const Complex<Real> *>(in), target);
    applyOnsite(static_cast<const Complex<Real> *>(out),
                static_cast<const Complex<Real> *>(clover),
                static_cast<Complex<Real> *>(out));
  }

  void cloverBistabCg(
      void *out, const void *in, const void *gauge, const void *clover_even,
      const void *clover_odd, const void *clover_even_inv,
      const void *clover_odd_inv) override {
    requireSolverMode();
    const int parity = targetParity();
    ensureGaugeHalo(static_cast<const Complex<Real> *>(gauge));
    const std::array<const Complex<Real> *, 2> clover = {
        static_cast<const Complex<Real> *>(clover_even),
        static_cast<const Complex<Real> *>(clover_odd)};
    const std::array<const Complex<Real> *, 2> inverse = {
        static_cast<const Complex<Real> *>(clover_even_inv),
        static_cast<const Complex<Real> *>(clover_odd_inv)};
    solveClover(static_cast<Complex<Real> *>(out),
                static_cast<const Complex<Real> *>(in), parity, clover,
                inverse);
  }

  void cloverBistabCgDslash(
      void *out, const void *in, const void *gauge, const void *clover_even,
      const void *clover_odd, const void *clover_even_inv,
      const void *clover_odd_inv) override {
    requireSolverMode();
    const int parity = targetParity();
    ensureGaugeHalo(static_cast<const Complex<Real> *>(gauge));
    const std::array<const Complex<Real> *, 2> clover = {
        static_cast<const Complex<Real> *>(clover_even),
        static_cast<const Complex<Real> *>(clover_odd)};
    const std::array<const Complex<Real> *, 2> inverse = {
        static_cast<const Complex<Real> *>(clover_even_inv),
        static_cast<const Complex<Real> *>(clover_odd_inv)};
    applySchur(static_cast<const Complex<Real> *>(in),
               static_cast<Complex<Real> *>(out), parity, clover, inverse,
               tmp_a_.data(), tmp_b_.data());
  }

  int cloverBistabCgPrepare(
      void *compact_rhs, const void *full_rhs, const void *gauge,
      const void *clover_even, const void *clover_odd,
      const void *clover_even_inv, const void *clover_odd_inv) override {
    requireSolverMode();
    if (compact_rhs == nullptr || full_rhs == nullptr || gauge == nullptr ||
        clover_even == nullptr || clover_odd == nullptr ||
        clover_even_inv == nullptr || clover_odd_inv == nullptr) {
      throw std::runtime_error("Clover prepare received a null pointer");
    }
    ensureGaugeHalo(static_cast<const Complex<Real> *>(gauge));
    const std::array<const Complex<Real> *, 2> inverse = {
        static_cast<const Complex<Real> *>(clover_even_inv),
        static_cast<const Complex<Real> *>(clover_odd_inv)};
    prepareCloverSchur(static_cast<Complex<Real> *>(compact_rhs),
                       static_cast<const Complex<Real> *>(full_rhs),
                       targetParity(), inverse);
    return 0;
  }

  int cloverBistabCgReconstruct(
      void *full_out, const void *full_rhs, const void *target_solution,
      const void *gauge, const void *clover_even, const void *clover_odd,
      const void *clover_even_inv, const void *clover_odd_inv) override {
    requireSolverMode();
    if (full_out == nullptr || full_rhs == nullptr ||
        target_solution == nullptr || gauge == nullptr ||
        clover_even == nullptr || clover_odd == nullptr ||
        clover_even_inv == nullptr || clover_odd_inv == nullptr) {
      throw std::runtime_error("Clover reconstruct received a null pointer");
    }
    ensureGaugeHalo(static_cast<const Complex<Real> *>(gauge));
    const std::array<const Complex<Real> *, 2> inverse = {
        static_cast<const Complex<Real> *>(clover_even_inv),
        static_cast<const Complex<Real> *>(clover_odd_inv)};
    reconstructCloverSchur(
        static_cast<Complex<Real> *>(full_out),
        static_cast<const Complex<Real> *>(full_rhs),
        static_cast<const Complex<Real> *>(target_solution), targetParity(),
        inverse);
    return 0;
  }

  void gaussGauge(void *gauge_out) override {
    requireGaugeMode();
    if (gauge_out == nullptr) {
      throw std::runtime_error("Gaussian gauge received a null output pointer");
    }
    const Real sigma = parameters_.reals[kSigma];
    if (!std::isfinite(sigma) || sigma < Real(0)) {
      throw std::runtime_error("Gaussian gauge sigma must be finite and nonnegative");
    }
    auto *gauge = static_cast<Complex<Real> *>(gauge_out);
    const uint64_t seed = static_cast<uint64_t>(parameters_.get(kSeed));
    const int64_t task_count = geometry_.half_sites * 2 * kDim;
    const int64_t chunk = ompChunk<Real>(task_count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t task = 0; task < task_count; ++task) {
      const int64_t site = task % geometry_.half_sites;
      const int direction =
          static_cast<int>((task / geometry_.half_sites) % kDim);
      const int parity =
          static_cast<int>(task / (geometry_.half_sites * kDim));
      int x = 0;
      int y = 0;
      int z = 0;
      int t2 = 0;
      geometry_.decodeHalf(site, x, y, z, t2);
      const int local_t = geometry_.timeForParity(x, y, z, t2, parity);
      const int global_x = geometry_.coord[0] * geometry_.local[0] + x;
      const int global_y = geometry_.coord[1] * geometry_.local[1] + y;
      const int global_z = geometry_.coord[2] * geometry_.local[2] + z;
      const int global_t = geometry_.coord[3] * geometry_.local[3] + local_t;
      const uint64_t global_y_extent =
          static_cast<uint64_t>(geometry_.grid[1] * geometry_.local[1]);
      const uint64_t global_z_extent =
          static_cast<uint64_t>(geometry_.grid[2] * geometry_.local[2]);
      const uint64_t global_t_extent =
          static_cast<uint64_t>(geometry_.grid[3] * geometry_.local[3]);
      const uint64_t global_site =
          (((static_cast<uint64_t>(global_x) * global_y_extent +
             static_cast<uint64_t>(global_y)) *
                global_z_extent +
            static_cast<uint64_t>(global_z)) *
               global_t_extent +
           static_cast<uint64_t>(global_t));
      Complex<Real> matrix[9];
      makeRandomSu3(seed, global_site, parity, direction, sigma, matrix);
      for (int row = 0; row < kColor; ++row) {
        for (int column = 0; column < kColor; ++column) {
          gauge[gaugeIndex(parity, row, column, direction, x, y, z, t2)] =
              matrix[row * kColor + column];
        }
      }
    }
  }

  void multigridRestrict(void *coarse_out, const void *fine_in,
                         const void *null_vectors, int E, int e, int Xf,
                         int Yf, int Zf, int Tf, int Xc, int Yc, int Zc,
                         int Tc) override {
    requireWilsonMode();
    applyMultigridRestrict(
        static_cast<Complex<Real> *>(coarse_out),
        static_cast<const Complex<Real> *>(fine_in),
        static_cast<const Complex<Real> *>(null_vectors), E, e, Xf, Yf, Zf,
        Tf, Xc, Yc, Zc, Tc);
  }

  void multigridProlong(void *fine_out, const void *coarse_in,
                        const void *null_vectors, int E, int e, int Xf,
                        int Yf, int Zf, int Tf, int Xc, int Yc, int Zc,
                        int Tc) override {
    requireWilsonMode();
    applyMultigridProlong(
        static_cast<Complex<Real> *>(fine_out),
        static_cast<const Complex<Real> *>(coarse_in),
        static_cast<const Complex<Real> *>(null_vectors), E, e, Xf, Yf, Zf,
        Tf, Xc, Yc, Zc, Tc);
  }

  void multigridCoarseDslash(void *out, const void *in, const void *hopping,
                             const void *sitting, int E, int X, int Y, int Z,
                             int T) override {
    requireWilsonMode();
    applyMultigridCoarseDslash(
        static_cast<Complex<Real> *>(out),
        static_cast<const Complex<Real> *>(in),
        static_cast<const Complex<Real> *>(hopping),
        static_cast<const Complex<Real> *>(sitting), E, X, Y, Z, T);
  }

  void multigridCoarseDslashWide(
      void *out, const void *in, const void *sitting, const void *hop_nn,
      const void *hop_diag, int E, int X, int Y, int Z, int T) override {
    requireWilsonMode();
    applyMultigridCoarseDslashWide(
        static_cast<Complex<Real> *>(out),
        static_cast<const Complex<Real> *>(in),
        static_cast<const Complex<Real> *>(sitting),
        static_cast<const Complex<Real> *>(hop_nn),
        static_cast<const Complex<Real> *>(hop_diag), E, X, Y, Z, T);
  }

  void laplacian(void *out, const void *in, const void *gauge) override {
    requireLaplacianMode();
    applyLaplacian(static_cast<Complex<Real> *>(out),
                   static_cast<const Complex<Real> *>(in),
                   static_cast<const Complex<Real> *>(gauge));
  }

  int iterations() const override { return last_iterations_; }

private:
  Geometry<Real> geometry_;
  Parameters<Real> parameters_{};
  int threads_ = 1;
  int data_type_ = kLatC64;
  int last_iterations_ = 0;
  GaugeHalo<Real> gauge_halo_;
  std::array<std::vector<Complex<Real>>, 8> fermion_send_;
  std::array<std::vector<Complex<Real>>, 8> fermion_recv_;
  std::vector<MPI_Request> fermion_requests_;
  std::array<std::vector<Complex<Real>>, 8> laplacian_source_send_;
  std::array<std::vector<Complex<Real>>, 8> laplacian_source_recv_;
  std::array<std::vector<Complex<Real>>, 8> laplacian_gauge_send_;
  std::array<std::vector<Complex<Real>>, 8> laplacian_gauge_recv_;
  std::vector<MPI_Request> laplacian_requests_;
  std::array<std::vector<Complex<Real>>, 2> cached_clover_;
  std::array<std::vector<Complex<Real>>, 2> cached_clover_inv_;
  const Complex<Real> *cached_gauge_ = nullptr;
  std::vector<Complex<Real>> r_;
  std::vector<Complex<Real>> rhat_;
  std::vector<Complex<Real>> p_;
  std::vector<Complex<Real>> v_;
  std::vector<Complex<Real>> s_;
  std::vector<Complex<Real>> t_;
  std::vector<Complex<Real>> ph_;
  std::vector<Complex<Real>> sh_;
  std::vector<Complex<Real>> bp_;
  std::vector<Complex<Real>> tmp_a_;
  std::vector<Complex<Real>> tmp_b_;
  std::vector<Complex<Real>> work_a_;
  std::vector<Complex<Real>> work_b_;
  std::vector<Complex<Real>> work_c_;
  int coarse_halo_E_ = 0;
  std::array<int, kDim> coarse_halo_dims_{};
  int coarse_halo_face_size_ = 0;
  std::array<std::vector<Complex<Real>>, kCoarseDirections>
      coarse_halo_send_;
  std::array<std::vector<Complex<Real>>, kCoarseDirections>
      coarse_halo_recv_;
  std::array<int, kCoarseDirections> coarse_halo_offsets_{};

  void allocateScratch() {
    if (geometry_.laplacian) {
      for (int dim = 0; dim < 3; ++dim) {
        const std::size_t source_face =
            static_cast<std::size_t>(geometry_.face_sites[dim] * kColor);
        const std::size_t gauge_face =
            static_cast<std::size_t>(geometry_.face_sites[dim] * kGauge);
        for (int sign : {-1, 1}) {
          const int offset = faceOffset(dim, sign);
          laplacian_source_send_[offset].resize(source_face);
          laplacian_source_recv_[offset].resize(source_face);
          laplacian_gauge_send_[offset].resize(gauge_face);
          laplacian_gauge_recv_[offset].resize(gauge_face);
        }
      }
      return;
    }
    const std::size_t vectors =
        static_cast<std::size_t>(geometry_.parity_vectors * 2);
    for (int dim = 0; dim < kDim; ++dim) {
      const std::size_t face =
          static_cast<std::size_t>(geometry_.face_sites[dim] * kVector);
      fermion_send_[faceOffset(dim, -1)].resize(face);
      fermion_send_[faceOffset(dim, 1)].resize(face);
      fermion_recv_[faceOffset(dim, -1)].resize(face);
      fermion_recv_[faceOffset(dim, 1)].resize(face);
      const std::size_t gauge_face =
          static_cast<std::size_t>(geometry_.face_sites[dim] * 2 * kGauge *
                                   kDim);
      gauge_halo_.faces[faceOffset(dim, -1)].resize(gauge_face);
      gauge_halo_.faces[faceOffset(dim, 1)].resize(gauge_face);
    }
    for (int dim0 = 0; dim0 < kDim; ++dim0) {
      for (int dim1 = dim0 + 1; dim1 < kDim; ++dim1) {
        for (int sign0 : {-1, 1}) {
          for (int sign1 : {-1, 1}) {
            auto &buffer = gauge_halo_
                               .corners[static_cast<std::size_t>(
                                   planeIndex(dim0, dim1))]
                               [static_cast<std::size_t>(
                                   signIndex(sign0) * 2 + signIndex(sign1))];
            buffer.resize(static_cast<std::size_t>(
                geometry_.corner_sites[dim0][dim1] * 2 * kGauge * kDim));
          }
        }
      }
    }
    for (auto *field :
         {&r_, &rhat_, &p_, &v_, &s_, &t_, &ph_, &sh_, &bp_, &tmp_a_,
          &tmp_b_}) {
      field->resize(vectors);
    }
    for (auto *field : {&work_a_, &work_b_, &work_c_}) {
      field->resize(static_cast<std::size_t>(geometry_.parity_vectors));
    }
    for (int parity = 0; parity < 2; ++parity) {
      cached_clover_[static_cast<std::size_t>(parity)].resize(
          static_cast<std::size_t>(geometry_.half_sites * kOnsite));
      cached_clover_inv_[static_cast<std::size_t>(parity)].resize(
          static_cast<std::size_t>(geometry_.half_sites * kOnsite));
    }
  }

  Real mass() const { return parameters_.reals[kMass]; }
  Real kappa() const { return Real(1) / (Real(2) * mass() + Real(8)); }
  Real csw() const { return parameters_.reals[kCsw]; }
  Real atol() const { return parameters_.reals[kAtol]; }
  Real residualFloor() const {
    const Real multiplier = sizeof(Real) == sizeof(float) ? Real(2048)
                                                         : Real(128);
    return std::numeric_limits<Real>::epsilon() * multiplier;
  }

  void requireWilsonMode() const {
    if (geometry_.laplacian) {
      throw std::runtime_error(
          "Wilson/Clover operation cannot run in Laplacian mode");
    }
  }

  void requireGaugeMode() const {
    if (geometry_.laplacian ||
        parameters_.get(kSetPlan) != kSetPlanGauge) {
      throw std::runtime_error(
          "Gaussian gauge requires params[_SET_PLAN_]=-1");
    }
  }

  void requireSolverMode() const {
    requireWilsonMode();
    if (parameters_.get(kSetPlan) != kSetPlanSolver) {
      throw std::runtime_error(
          "Wilson/Clover solver operation requires params[_SET_PLAN_]=1");
    }
  }

  void requireLaplacianMode() const {
    if (!geometry_.laplacian) {
      throw std::runtime_error(
          "Laplacian operation requires applyInitQcu with plan -2");
    }
    if (parameters_.get(kSetPlan) != kSetPlanLaplacian) {
      throw std::runtime_error(
          "Laplacian operation requires params[_SET_PLAN_]=-2");
    }
  }

  void requireWilsonGauge(const Complex<Real> *gauge) {
    if (gauge == nullptr) {
      throw std::runtime_error("null gauge pointer");
    }
    ensureGaugeHalo(gauge);
  }

  void validateParity() const {
    const int parity = parameters_.get(kParity);
    if (parity != 0 && parity != 1) {
      throw std::runtime_error("params[_PARITY_] must be 0 or 1");
    }
  }

  int targetParity() const {
    validateParity();
    return parameters_.get(kParity);
  }

  template <typename... Args>
  void log(const char *submodule, const char *format, Args... args) const {
    if (parameters_.get(kVerbose) == 0) {
      return;
    }
    std::fprintf(stderr, "PYQCU::KUPL::%s:\n ", submodule);
    std::fprintf(stderr, format, args...);
    std::fprintf(stderr, "\n");
  }

  int64_t siteLinear(int x, int y, int z, int t2) const {
    return geometry_.siteLinear(x, y, z, t2);
  }

  int surfaceExtent(int dim) const {
    if (dim != 3) {
      return geometry_.local[dim];
    }
    return geometry_.laplacian ? 1 : geometry_.local[3] / 2;
  }

  void decodeFace(int64_t face, int fixed_dim, int fixed_value,
                  int &x, int &y, int &z, int &t2) const {
    std::array<int, kDim> coordinate{};
    for (int dim = kDim - 1; dim >= 0; --dim) {
      if (dim == fixed_dim) {
        coordinate[dim] = fixed_value;
      } else {
        coordinate[dim] = static_cast<int>(face % surfaceExtent(dim));
        face /= surfaceExtent(dim);
      }
    }
    x = coordinate[0];
    y = coordinate[1];
    z = coordinate[2];
    t2 = coordinate[3];
  }

  void decodeCorner(int64_t corner, int fixed_dim0, int fixed_value0,
                    int fixed_dim1, int fixed_value1, int &x, int &y,
                    int &z, int &t2) const {
    std::array<int, kDim> coordinate{};
    for (int dim = kDim - 1; dim >= 0; --dim) {
      if (dim == fixed_dim0) {
        coordinate[dim] = fixed_value0;
      } else if (dim == fixed_dim1) {
        coordinate[dim] = fixed_value1;
      } else {
        coordinate[dim] = static_cast<int>(corner % surfaceExtent(dim));
        corner /= surfaceExtent(dim);
      }
    }
    x = coordinate[0];
    y = coordinate[1];
    z = coordinate[2];
    t2 = coordinate[3];
  }

  int64_t canonicalVectorIndex(int parity, int component,
                               int64_t site) const {
    return (static_cast<int64_t>(parity) * kVector + component) *
               geometry_.half_sites +
           site;
  }

  VectorView<Real> compactView(const Complex<Real> *field,
                               int64_t site) const {
    return {field + site, geometry_.half_sites};
  }

  VectorView<Real> haloView(const std::vector<Complex<Real>> &halo,
                            int64_t face_site) const {
    return {halo.data() + face_site * kVector, 1};
  }

  void gatherVector(const VectorView<Real> &view,
                    Complex<Real> local[kVector]) const {
    for (int component = 0; component < kVector; ++component) {
      local[component] = view[component];
    }
  }

  void scatterVector(const Complex<Real> local[kVector],
                     Complex<Real> *field, int64_t site) const {
    for (int component = 0; component < kVector; ++component) {
      field[static_cast<int64_t>(component) * geometry_.half_sites + site] =
          local[component];
    }
  }

  std::size_t gaugeIndex(int parity, int row, int column, int direction, int x,
                         int y, int z, int t2) const {
    int64_t index = parity;
    index = index * kColor + row;
    index = index * kColor + column;
    index = index * kDim + direction;
    index = index * geometry_.local[0] + x;
    index = index * geometry_.local[1] + y;
    index = index * geometry_.local[2] + z;
    index = index * (geometry_.local[3] / 2) + t2;
    return static_cast<std::size_t>(index);
  }

  GaugeView<Real> localGauge(int parity, int direction, int x, int y, int z,
                             int t2) const {
    return {gauge_halo_.source +
                gaugeIndex(parity, 0, 0, direction, x, y, z, t2),
            static_cast<int64_t>(kDim) * geometry_.half_sites};
  }

  template <typename Callback>
  void forEachGaugeBoundary(int dim, int boundary, Callback callback) {
    const int64_t count = geometry_.half_sites;
    const int64_t chunk = ompChunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t linear = 0; linear < count; ++linear) {
      int x = 0;
      int y = 0;
      int z = 0;
      int t2 = 0;
      geometry_.decodeHalf(linear, x, y, z, t2);
      int coordinate[kDim] = {x, y, z, t2};
      if (coordinate[dim] != boundary) {
        continue;
      }
      callback(linear, x, y, z, t2);
    }
  }

  void packGaugeFace(int dim, int boundary, Complex<Real> *buffer) {
    const int64_t count =
        geometry_.half_sites / surfaceExtent(dim);
    const int64_t chunk = ompChunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t face_site = 0; face_site < count; ++face_site) {
      int x = 0;
      int y = 0;
      int z = 0;
      int t2 = 0;
      decodeFace(face_site, dim, boundary, x, y, z, t2);
      const int64_t target_index =
          face_site * 2 * kGauge * kDim;
      for (int parity = 0; parity < 2; ++parity) {
        for (int row = 0; row < kColor; ++row) {
          for (int column = 0; column < kColor; ++column) {
            for (int direction = 0; direction < kDim; ++direction) {
              const int64_t offset =
                  ((((parity * kColor + row) * kColor + column) * kDim +
                    direction));
              buffer[target_index + offset] =
                  gauge_halo_.source[gaugeIndex(parity, row, column,
                                                direction, x, y, z, t2)];
            }
          }
        }
      }
    }
  }

  void packGaugeCorner(int dim0, int dim1, int boundary0, int boundary1,
                       Complex<Real> *buffer) {
    const int64_t count =
        geometry_.half_sites / (surfaceExtent(dim0) * surfaceExtent(dim1));
    const int64_t chunk = ompChunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t corner_site = 0; corner_site < count; ++corner_site) {
      int x = 0;
      int y = 0;
      int z = 0;
      int t2 = 0;
      decodeCorner(corner_site, dim0, boundary0, dim1, boundary1,
                   x, y, z, t2);
      const int64_t target_index =
          corner_site * 2 * kGauge * kDim;
      for (int parity = 0; parity < 2; ++parity) {
        for (int row = 0; row < kColor; ++row) {
          for (int column = 0; column < kColor; ++column) {
            for (int direction = 0; direction < kDim; ++direction) {
              const int64_t offset =
                  ((((parity * kColor + row) * kColor + column) * kDim +
                    direction));
              buffer[target_index + offset] =
                  gauge_halo_.source[gaugeIndex(parity, row, column,
                                                direction, x, y, z, t2)];
            }
          }
        }
      }
    }
  }

  void ensureGaugeHalo(const Complex<Real> *gauge) {
    if (gauge == nullptr) {
      throw std::runtime_error("null gauge pointer");
    }
    if (gauge_halo_.ready && gauge_halo_.source == gauge) {
      return;
    }
    gauge_halo_.source = gauge;
    gauge_halo_.ready = false;

    std::vector<MPI_Request> requests;
    requests.reserve(64);
    auto post = [&requests](MPI_Request request) {
      if (request != MPI_REQUEST_NULL) {
        requests.push_back(request);
      }
    };

    for (int dim = 0; dim < kDim; ++dim) {
      if (geometry_.grid[dim] == 1) {
        continue;
      }
      auto &send_lower = gauge_halo_.faces[faceOffset(dim, -1)];
      auto &send_upper = gauge_halo_.faces[faceOffset(dim, 1)];
      packGaugeFace(dim, 0, send_lower.data());
      packGaugeFace(dim, geometry_.lastHalfCoordinate(dim),
                    send_upper.data());

      // The lower face is sent to rank-minus and receives rank-plus' lower
      // face in return. The upper face is sent to rank-plus and receives
      // rank-minus' upper face.
      std::vector<Complex<Real>> recv_minus_storage(send_lower.size());
      std::vector<Complex<Real>> recv_plus_storage(send_upper.size());
      MPI_Request request = MPI_REQUEST_NULL;
      MPI_Irecv(recv_plus_storage.data(),
                static_cast<int>(recv_plus_storage.size() *
                                 sizeof(Complex<Real>)),
                MPI_BYTE, geometry_.neighbor_plus[dim],
                faceTag(dim, -1), geometry_.comm, &request);
      post(request);
      MPI_Irecv(recv_minus_storage.data(),
                static_cast<int>(recv_minus_storage.size() *
                                 sizeof(Complex<Real>)),
                MPI_BYTE, geometry_.neighbor_minus[dim],
                faceTag(dim, 1), geometry_.comm, &request);
      post(request);
      MPI_Isend(send_upper.data(),
                static_cast<int>(send_upper.size() * sizeof(Complex<Real>)),
                MPI_BYTE, geometry_.neighbor_plus[dim], faceTag(dim, 1),
                geometry_.comm, &request);
      post(request);
      MPI_Isend(send_lower.data(),
                static_cast<int>(send_lower.size() * sizeof(Complex<Real>)),
                MPI_BYTE, geometry_.neighbor_minus[dim], faceTag(dim, -1),
                geometry_.comm, &request);
      post(request);

      // Wait immediately so the two temporary receive buffers can be moved
      // into the persistent face cache without increasing peak memory.
      if (!requests.empty()) {
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                    MPI_STATUSES_IGNORE);
        requests.clear();
      }
      gauge_halo_.faces[faceOffset(dim, 1)] = std::move(recv_plus_storage);
      gauge_halo_.faces[faceOffset(dim, -1)] =
          std::move(recv_minus_storage);
    }

    std::array<std::array<std::vector<Complex<Real>>, 4>, 6>
        corner_recv_storage;
    std::vector<MPI_Request> corner_requests;
    corner_requests.reserve(48);
    for (int dim0 = 0; dim0 < kDim; ++dim0) {
      for (int dim1 = dim0 + 1; dim1 < kDim; ++dim1) {
        if (geometry_.grid[dim0] == 1 || geometry_.grid[dim1] == 1) {
          continue;
        }
        for (int sign0 : {-1, 1}) {
          for (int sign1 : {-1, 1}) {
            int diagonal = MPI_PROC_NULL;
            std::array<int, kDim> coordinates = geometry_.coord;
            coordinates[dim0] =
                (coordinates[dim0] + sign0 + geometry_.grid[dim0]) %
                geometry_.grid[dim0];
            coordinates[dim1] =
                (coordinates[dim1] + sign1 + geometry_.grid[dim1]) %
                geometry_.grid[dim1];
            MPI_Cart_rank(geometry_.comm, coordinates.data(), &diagonal);

            const int boundary0 =
                sign0 > 0 ? geometry_.lastHalfCoordinate(dim0) : 0;
            const int boundary1 =
                sign1 > 0 ? geometry_.lastHalfCoordinate(dim1) : 0;
            auto &send = gauge_halo_
                             .corners[static_cast<std::size_t>(
                                 planeIndex(dim0, dim1))]
                             [static_cast<std::size_t>(
                                 signIndex(sign0) * 2 + signIndex(sign1))];
            packGaugeCorner(dim0, dim1, boundary0, boundary1, send.data());
            const int plane = planeIndex(dim0, dim1);
            const int slot =
                signIndex(sign0) * 2 + signIndex(sign1);
            auto &recv = corner_recv_storage[static_cast<std::size_t>(plane)]
                                            [static_cast<std::size_t>(slot)];
            recv.resize(send.size());
            const int send_tag = cornerTag(dim0, dim1, sign0, sign1);
            const int recv_tag =
                cornerTag(dim0, dim1, -sign0, -sign1);
            MPI_Request request = MPI_REQUEST_NULL;
            MPI_Irecv(recv.data(),
                      static_cast<int>(recv.size() * sizeof(Complex<Real>)),
                      MPI_BYTE, diagonal, recv_tag, geometry_.comm, &request);
            corner_requests.push_back(request);
            MPI_Isend(send.data(),
                      static_cast<int>(send.size() * sizeof(Complex<Real>)),
                      MPI_BYTE, diagonal, send_tag, geometry_.comm, &request);
            corner_requests.push_back(request);
          }
        }
      }
    }
    if (!corner_requests.empty()) {
      MPI_Waitall(static_cast<int>(corner_requests.size()),
                  corner_requests.data(), MPI_STATUSES_IGNORE);
    }
    for (int dim0 = 0; dim0 < kDim; ++dim0) {
      for (int dim1 = dim0 + 1; dim1 < kDim; ++dim1) {
        if (geometry_.grid[dim0] == 1 || geometry_.grid[dim1] == 1) {
          continue;
        }
        const int plane = planeIndex(dim0, dim1);
        for (int slot = 0; slot < 4; ++slot) {
          gauge_halo_.corners[static_cast<std::size_t>(plane)]
                             [static_cast<std::size_t>(slot)] =
              std::move(
                  corner_recv_storage[static_cast<std::size_t>(plane)]
                                     [static_cast<std::size_t>(slot)]);
        }
      }
    }
    gauge_halo_.ready = true;
  }

  GaugeView<Real> gaugeAtDelta(int x, int y, int z, int t2, int parity,
                               int direction,
                               const int delta[kDim]) const {
    int full_t = geometry_.timeForParity(x, y, z, t2, parity);
    int full[kDim] = {x, y, z, full_t};
    bool remote[kDim] = {false, false, false, false};
    int offsets[kDim] = {delta[0], delta[1], delta[2], delta[3]};
    for (int dim = 0; dim < kDim; ++dim) {
      if (offsets[dim] == 0) {
        continue;
      }
      full[dim] += offsets[dim];
      if (full[dim] < 0) {
        if (geometry_.grid[dim] == 1) {
          full[dim] = geometry_.local[dim] - 1;
        } else {
          full[dim] = geometry_.local[dim] - 1;
          remote[dim] = true;
        }
      } else if (full[dim] >= geometry_.local[dim]) {
        if (geometry_.grid[dim] == 1) {
          full[dim] = 0;
        } else {
          full[dim] = 0;
          remote[dim] = true;
        }
      }
    }
    const int moved_parity =
        geometry_.parityAt(full[0], full[1], full[2], full[3]);
    const int moved_t2 = full[3] / 2;
    const int remote_dims =
        static_cast<int>(remote[0]) + static_cast<int>(remote[1]) +
        static_cast<int>(remote[2]) + static_cast<int>(remote[3]);
    if (remote_dims == 0) {
      return localGauge(moved_parity, direction, full[0], full[1], full[2],
                        moved_t2);
    }
    int remote_dim0 = -1;
    int remote_dim1 = -1;
    int sign0 = 0;
    int sign1 = 0;
    for (int dim = 0; dim < kDim; ++dim) {
      if (!remote[dim]) {
        continue;
      }
      if (remote_dim0 < 0) {
        remote_dim0 = dim;
        sign0 = offsets[dim];
      } else {
        remote_dim1 = dim;
        sign1 = offsets[dim];
      }
    }
    if (remote_dim1 < 0) {
      const auto &face = gauge_halo_.faces[faceOffset(remote_dim0, sign0)];
      const int64_t index =
          geometry_.faceIndex(remote_dim0, full[0], full[1], full[2],
                              moved_t2);
      const int64_t offset =
          index * 2 * kGauge * kDim + direction;
      const int64_t parity_base =
          static_cast<int64_t>(moved_parity) * kGauge * kDim;
      return {face.data() + parity_base + offset, kDim};
    }
    if (remote_dim0 > remote_dim1) {
      std::swap(remote_dim0, remote_dim1);
      std::swap(sign0, sign1);
    }
    const int plane = planeIndex(remote_dim0, remote_dim1);
    const int slot = signIndex(sign0) * 2 + signIndex(sign1);
    const auto &corner =
        gauge_halo_.corners[static_cast<std::size_t>(plane)]
                           [static_cast<std::size_t>(slot)];
    const int64_t index =
        geometry_.cornerIndex(remote_dim0, remote_dim1, full[0], full[1],
                              full[2], moved_t2);
    const int64_t offset =
        index * 2 * kGauge * kDim + direction;
    const int64_t parity_base =
        static_cast<int64_t>(moved_parity) * kGauge * kDim;
    return {corner.data() + parity_base + offset, kDim};
  }

  void loadGaugeMatrix(const GaugeView<Real> &source,
                       Complex<Real> matrix[9]) const {
    for (int row = 0; row < kColor; ++row) {
      for (int column = 0; column < kColor; ++column) {
        matrix[row * 3 + column] =
            source.data[(row * kColor + column) * source.matrix_stride];
      }
    }
  }

  void computePlaquetteSum(int x, int y, int z, int t2, int mu, int nu,
                           int parity, Complex<Real> sum[9]) const {
    using C = Complex<Real>;
    for (int index = 0; index < 9; ++index) {
      sum[index] = C(0);
    }
    C m0[9];
    C m1[9];
    C m2[9];
    C m3[9];
    auto load_term = [&](int direction, int dx, int dy, int dz, int dt,
                         bool adjoint, C output[9]) {
      const int delta[kDim] = {dx, dy, dz, dt};
      loadGaugeMatrix(
          gaugeAtDelta(x, y, z, t2, parity, direction, delta), output);
      if (adjoint) {
        C adjoint_matrix[9];
        for (int row = 0; row < kColor; ++row) {
          for (int column = 0; column < kColor; ++column) {
            adjoint_matrix[row * kColor + column] =
                std::conj(output[column * kColor + row]);
          }
        }
        std::copy(std::begin(adjoint_matrix), std::end(adjoint_matrix),
                  output);
      }
    };
    auto add_product = [&]() {
      C temp[9];
      C product[9];
      multiply3(m0, m1, temp);
      multiply3(temp, m2, product);
      multiply3(product, m3, temp);
      for (int index = 0; index < 9; ++index) {
        sum[index] += temp[index];
      }
    };

    // term1 = U_mu(x) U_nu(x+mu) U_mu(x+nu)^dag U_nu(x)^dag
    load_term(mu, 0, 0, 0, 0, false, m0);
    load_term(nu, mu == 0 ? 1 : 0, mu == 1 ? 1 : 0, mu == 2 ? 1 : 0,
              mu == 3 ? 1 : 0, false, m1);
    load_term(mu, nu == 0 ? 1 : 0, nu == 1 ? 1 : 0, nu == 2 ? 1 : 0,
              nu == 3 ? 1 : 0, true, m2);
    load_term(nu, 0, 0, 0, 0, true, m3);
    add_product();

    // term2 = U_nu(x) U_mu(x-mu+nu)^dag U_nu(x-mu)^dag U_mu(x-mu)
    load_term(nu, 0, 0, 0, 0, false, m0);
    load_term(mu, (mu == 0 ? -1 : 0) + (nu == 0 ? 1 : 0),
              (mu == 1 ? -1 : 0) + (nu == 1 ? 1 : 0),
              (mu == 2 ? -1 : 0) + (nu == 2 ? 1 : 0),
              (mu == 3 ? -1 : 0) + (nu == 3 ? 1 : 0), true, m1);
    load_term(nu, mu == 0 ? -1 : 0, mu == 1 ? -1 : 0,
              mu == 2 ? -1 : 0, mu == 3 ? -1 : 0, true, m2);
    load_term(mu, mu == 0 ? -1 : 0, mu == 1 ? -1 : 0,
              mu == 2 ? -1 : 0, mu == 3 ? -1 : 0, false, m3);
    add_product();

    // term3 = U_mu(x-mu)^dag U_nu(x-mu-nu)^dag
    //         U_mu(x-mu-nu) U_nu(x-nu)
    load_term(mu, mu == 0 ? -1 : 0, mu == 1 ? -1 : 0,
              mu == 2 ? -1 : 0, mu == 3 ? -1 : 0, true, m0);
    load_term(nu, (mu == 0 ? -1 : 0) + (nu == 0 ? -1 : 0),
              (mu == 1 ? -1 : 0) + (nu == 1 ? -1 : 0),
              (mu == 2 ? -1 : 0) + (nu == 2 ? -1 : 0),
              (mu == 3 ? -1 : 0) + (nu == 3 ? -1 : 0), true, m1);
    load_term(mu, (mu == 0 ? -1 : 0) + (nu == 0 ? -1 : 0),
              (mu == 1 ? -1 : 0) + (nu == 1 ? -1 : 0),
              (mu == 2 ? -1 : 0) + (nu == 2 ? -1 : 0),
              (mu == 3 ? -1 : 0) + (nu == 3 ? -1 : 0), false, m2);
    load_term(nu, nu == 0 ? -1 : 0, nu == 1 ? -1 : 0,
              nu == 2 ? -1 : 0, nu == 3 ? -1 : 0, false, m3);
    add_product();

    // term4 = U_nu(x-nu)^dag U_mu(x-nu)
    //         U_nu(x-nu+mu) U_mu(x)^dag
    load_term(nu, nu == 0 ? -1 : 0, nu == 1 ? -1 : 0,
              nu == 2 ? -1 : 0, nu == 3 ? -1 : 0, true, m0);
    load_term(mu, nu == 0 ? -1 : 0, nu == 1 ? -1 : 0,
              nu == 2 ? -1 : 0, nu == 3 ? -1 : 0, false, m1);
    load_term(nu, (nu == 0 ? -1 : 0) + (mu == 0 ? 1 : 0),
              (nu == 1 ? -1 : 0) + (mu == 1 ? 1 : 0),
              (nu == 2 ? -1 : 0) + (mu == 2 ? 1 : 0),
              (nu == 3 ? -1 : 0) + (mu == 3 ? 1 : 0), false, m2);
    load_term(mu, 0, 0, 0, 0, true, m3);
    add_product();
  }

  void computeClover(Complex<Real> *output, int parity) {
    using C = Complex<Real>;
    const Real coefficient = -Real(0.125) * kappa() * csw();
    const int64_t count = geometry_.half_sites;
    const int64_t chunk = ompChunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t linear = 0; linear < count; ++linear) {
      int x = 0;
      int y = 0;
      int z = 0;
      int t2 = 0;
      geometry_.decodeHalf(linear, x, y, z, t2);
      C matrix[kOnsite];
      std::fill(std::begin(matrix), std::end(matrix), C(0));
      for (int component = 0; component < kVector; ++component) {
        matrix[component * kVector + component] = C(1);
      }
      for (int mu = 0; mu < kDim; ++mu) {
        for (int nu = mu + 1; nu < kDim; ++nu) {
          C plaquette[9];
          computePlaquetteSum(x, y, z, t2, mu, nu, parity, plaquette);
          C anti_hermitian[9];
          for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
              anti_hermitian[row * 3 + column] =
                  plaquette[row * 3 + column] -
                  std::conj(plaquette[column * 3 + row]);
            }
          }
          C sigma[16];
          for (int spin_out = 0; spin_out < 4; ++spin_out) {
            for (int spin_in = 0; spin_in < 4; ++spin_in) {
              C value(0);
              for (int inner = 0; inner < 4; ++inner) {
                value += gammaEntry<Real>(mu, spin_out, inner) *
                         gammaEntry<Real>(nu, inner, spin_in);
              }
              sigma[spin_out * 4 + spin_in] = value;
            }
          }
          for (int spin_out = 0; spin_out < 4; ++spin_out) {
            for (int color_out = 0; color_out < 3; ++color_out) {
              const int row = spin_out * 3 + color_out;
              for (int spin_in = 0; spin_in < 4; ++spin_in) {
                const C spin_factor = sigma[spin_out * 4 + spin_in];
                if (std::abs(spin_factor) == Real(0)) {
                  continue;
                }
                for (int color_in = 0; color_in < 3; ++color_in) {
                  const int column = spin_in * 3 + color_in;
                  matrix[row * kVector + column] +=
                      coefficient * spin_factor *
                      anti_hermitian[color_out * 3 + color_in];
                }
              }
            }
          }
        }
      }
      for (int component = 0; component < kOnsite; ++component) {
        output[static_cast<int64_t>(component) * geometry_.half_sites +
               linear] = matrix[component];
      }
    }
  }

  void invertClover(const Complex<Real> *term, Complex<Real> *inverse) {
    const int64_t count = geometry_.half_sites;
    const int64_t chunk = ompChunk<Real>(count);
    std::atomic<bool> singular{false};
    std::atomic<int64_t> singular_site{-1};
#pragma omp parallel for schedule(static, chunk)
    for (int64_t site = 0; site < count; ++site) {
      Complex<Real> augmented[kVector][2 * kVector];
      for (int row = 0; row < kVector; ++row) {
        for (int column = 0; column < 2 * kVector; ++column) {
          augmented[row][column] = Complex<Real>(0);
        }
        for (int column = 0; column < kVector; ++column) {
          augmented[row][column] =
              term[static_cast<int64_t>(row * kVector + column) *
                       geometry_.half_sites +
                   site];
        }
        augmented[row][kVector + row] = Complex<Real>(1);
      }
      const Real tolerance =
          std::numeric_limits<Real>::epsilon() * Real(256);
      for (int column = 0; column < kVector; ++column) {
        int pivot = column;
        Real pivot_norm = std::norm(augmented[pivot][column]);
        for (int row = column + 1; row < kVector; ++row) {
          const Real candidate = std::norm(augmented[row][column]);
          if (candidate > pivot_norm) {
            pivot = row;
            pivot_norm = candidate;
          }
        }
        if (std::sqrt(pivot_norm) <= tolerance) {
          singular.store(true, std::memory_order_relaxed);
          int64_t expected = -1;
          singular_site.compare_exchange_strong(
              expected, site, std::memory_order_relaxed);
          break;
        }
        if (pivot != column) {
          for (int entry = 0; entry < 2 * kVector; ++entry) {
            std::swap(augmented[pivot][entry],
                      augmented[column][entry]);
          }
        }
        const Complex<Real> diagonal = augmented[column][column];
        for (int entry = 0; entry < 2 * kVector; ++entry) {
          augmented[column][entry] /= diagonal;
        }
        for (int row = 0; row < kVector; ++row) {
          if (row == column) {
            continue;
          }
          const Complex<Real> factor = augmented[row][column];
          if (std::abs(factor) == Real(0)) {
            continue;
          }
          for (int entry = 0; entry < 2 * kVector; ++entry) {
            augmented[row][entry] -=
                factor * augmented[column][entry];
          }
        }
      }
      if (singular.load(std::memory_order_relaxed)) {
        continue;
      }
      for (int row = 0; row < kVector; ++row) {
        for (int column = 0; column < kVector; ++column) {
          inverse[static_cast<int64_t>(row * kVector + column) *
                      geometry_.half_sites +
                  site] =
              augmented[row][kVector + column];
        }
      }
    }
    if (singular.load(std::memory_order_relaxed)) {
      const int64_t site =
          singular_site.load(std::memory_order_relaxed);
      const Complex<Real> diagonal =
          term[static_cast<std::size_t>(site)];
      throw std::runtime_error(
          "singular clover onsite matrix at local half-site " +
          std::to_string(site) + " diagonal=(" +
          std::to_string(static_cast<double>(diagonal.real())) + "," +
          std::to_string(static_cast<double>(diagonal.imag())) + ")");
    }
  }

  void ensureCachedClover(const Complex<Real> *gauge) {
    if (cached_gauge_ == gauge) {
      return;
    }
    ensureGaugeHalo(gauge);
    for (int parity = 0; parity < 2; ++parity) {
      auto &term = cached_clover_[static_cast<std::size_t>(parity)];
      auto &inverse =
          cached_clover_inv_[static_cast<std::size_t>(parity)];
      computeClover(term.data(), parity);
      invertClover(term.data(), inverse.data());
    }
    cached_gauge_ = gauge;
  }

  void applyOnsite(const Complex<Real> *input, const Complex<Real> *matrix,
                   Complex<Real> *output) const {
    const int64_t count = geometry_.half_sites;
    const int64_t chunk = ompChunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t site = 0; site < count; ++site) {
      Complex<Real> source[kVector];
      Complex<Real> local[kVector];
      gatherVector(compactView(input, site), source);
      for (int row = 0; row < kVector; ++row) {
        Complex<Real> value(0);
        for (int column = 0; column < kVector; ++column) {
          value += matrix[static_cast<int64_t>(row * kVector + column) *
                              geometry_.half_sites +
                          site] *
                   source[column];
        }
        local[row] = value;
      }
      scatterVector(local, output, site);
    }
  }

  void packFermionFace(const Complex<Real> *source, int parity, int dim,
                       int boundary, Complex<Real> *buffer) {
    const int64_t count =
        geometry_.half_sites / surfaceExtent(dim);
    const int64_t chunk = ompChunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t face_site = 0; face_site < count; ++face_site) {
      int x = 0;
      int y = 0;
      int z = 0;
      int t2 = 0;
      decodeFace(face_site, dim, boundary, x, y, z, t2);
      const int64_t target = face_site * kVector;
      const int64_t site = siteLinear(x, y, z, t2);
      for (int component = 0; component < kVector; ++component) {
        buffer[target + component] =
            source[static_cast<int64_t>(component) *
                       geometry_.half_sites +
                   site];
      }
    }
    (void)parity;
  }

  void postFermionHalo(const Complex<Real> *source, int parity) {
    fermion_requests_.clear();
    fermion_requests_.reserve(16);
    for (int dim = 0; dim < kDim; ++dim) {
      if (geometry_.grid[dim] == 1) {
        continue;
      }
      auto &send_minus = fermion_send_[faceOffset(dim, -1)];
      auto &send_plus = fermion_send_[faceOffset(dim, 1)];
      auto &recv_minus = fermion_recv_[faceOffset(dim, -1)];
      auto &recv_plus = fermion_recv_[faceOffset(dim, 1)];
      packFermionFace(source, parity, dim, 0, send_minus.data());
      packFermionFace(source, parity, dim,
                      geometry_.lastHalfCoordinate(dim),
                      send_plus.data());
      MPI_Request request = MPI_REQUEST_NULL;
      MPI_Irecv(recv_plus.data(),
                static_cast<int>(recv_plus.size() * sizeof(Complex<Real>)),
                MPI_BYTE, geometry_.neighbor_plus[dim],
                fermionFaceTag(dim, -1), geometry_.comm, &request);
      fermion_requests_.push_back(request);
      MPI_Irecv(recv_minus.data(),
                static_cast<int>(recv_minus.size() * sizeof(Complex<Real>)),
                MPI_BYTE, geometry_.neighbor_minus[dim],
                fermionFaceTag(dim, 1), geometry_.comm, &request);
      fermion_requests_.push_back(request);
      MPI_Isend(send_plus.data(),
                static_cast<int>(send_plus.size() * sizeof(Complex<Real>)),
                MPI_BYTE, geometry_.neighbor_plus[dim],
                fermionFaceTag(dim, 1), geometry_.comm, &request);
      fermion_requests_.push_back(request);
      MPI_Isend(send_minus.data(),
                static_cast<int>(send_minus.size() * sizeof(Complex<Real>)),
                MPI_BYTE, geometry_.neighbor_minus[dim],
                fermionFaceTag(dim, -1), geometry_.comm, &request);
      fermion_requests_.push_back(request);
    }
  }

  VectorView<Real> sourceAtDelta(const Complex<Real> *local_source, int x,
                                 int y, int z, int t2, int dim, int sign,
                                 int target) const {
    int coordinate[kDim] = {x, y, z, t2};
    bool remote = false;
    if (dim == 3) {
      const int target_time = geometry_.timeForParity(x, y, z, t2, target);
      int moved_time = target_time + sign;
      if (moved_time < 0 || moved_time >= geometry_.local[3]) {
        if (geometry_.grid[3] == 1) {
          moved_time = moved_time < 0 ? geometry_.local[3] - 1 : 0;
        } else {
          remote = true;
        }
      }
      if (!remote) {
        coordinate[3] = moved_time / 2;
      }
    } else {
      coordinate[dim] += sign;
      if (coordinate[dim] < 0 || coordinate[dim] >= geometry_.local[dim]) {
        if (geometry_.grid[dim] == 1) {
          coordinate[dim] =
              coordinate[dim] < 0 ? geometry_.local[dim] - 1 : 0;
        } else {
          remote = true;
        }
      }
    }
    if (!remote) {
      return compactView(
          local_source,
          siteLinear(coordinate[0], coordinate[1], coordinate[2],
                     coordinate[3]));
    }
    const auto &halo = fermion_recv_[faceOffset(dim, sign)];
    const int64_t index =
        geometry_.faceIndex(dim, x, y, z, t2);
    return haloView(halo, index);
  }

  void computeDslashInterior(const Complex<Real> *source,
                             Complex<Real> *output, int target) {
    const int64_t count = geometry_.half_sites;
    const int64_t chunk = ompChunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t linear = 0; linear < count; ++linear) {
      int x = 0;
      int y = 0;
      int z = 0;
      int t2 = 0;
      geometry_.decodeHalf(linear, x, y, z, t2);
      if (x == 0 || y == 0 || z == 0 || t2 == 0 ||
          x == geometry_.local[0] - 1 ||
          y == geometry_.local[1] - 1 ||
          z == geometry_.local[2] - 1 ||
          t2 == geometry_.local[3] / 2 - 1) {
        continue;
      }
      computeDslashSite(source, output, x, y, z, t2, target);
    }
  }

  void computeDslashBoundary(const Complex<Real> *local_source,
                             Complex<Real> *output, int target) {
    const int64_t count = geometry_.half_sites;
    const int64_t chunk = ompChunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t linear = 0; linear < count; ++linear) {
      int x = 0;
      int y = 0;
      int z = 0;
      int t2 = 0;
      geometry_.decodeHalf(linear, x, y, z, t2);
      const bool boundary =
          x == 0 || y == 0 || z == 0 || t2 == 0 ||
          x == geometry_.local[0] - 1 ||
          y == geometry_.local[1] - 1 ||
          z == geometry_.local[2] - 1 ||
          t2 == geometry_.local[3] / 2 - 1;
      if (!boundary) {
        continue;
      }
      computeDslashSite(local_source, output, x, y, z, t2, target);
    }
  }

  void computeDslashSite(const Complex<Real> *local_source,
                         Complex<Real> *output, int x, int y, int z,
                         int t2, int target) {
    using C = Complex<Real>;
    C result[kVector];
    for (int index = 0; index < kVector; ++index) {
      result[index] = C(0);
    }
    for (int direction = 0; direction < kDim; ++direction) {
      int delta_plus[kDim] = {0, 0, 0, 0};
      int delta_minus[kDim] = {0, 0, 0, 0};
      delta_plus[direction] = 1;
      delta_minus[direction] = -1;
      const VectorView<Real> forward_view =
          sourceAtDelta(local_source, x, y, z, t2, direction, 1, target);
      C forward_source[kVector];
      gatherVector(forward_view, forward_source);
      C gauge_matrix[9];
      loadGaugeMatrix(localGauge(target, direction, x, y, z, t2),
                      gauge_matrix);
      C gauge_source[kVector];
      C gamma_source[kVector];
      multiplyGaugeSpin(gauge_matrix, forward_source, gauge_source, false);
      applyOneMinusGamma(direction, gauge_source, gamma_source, 1);
      for (int index = 0; index < kVector; ++index) {
        result[index] += gamma_source[index];
      }

      const VectorView<Real> backward_view =
          sourceAtDelta(local_source, x, y, z, t2, direction, -1, target);
      C backward_source[kVector];
      gatherVector(backward_view, backward_source);
      C backward_coordinate[kDim] = {C(0), C(0), C(0), C(0)};
      (void)backward_coordinate;
      C backward_gauge[9];
      loadGaugeMatrix(backwardGauge(x, y, z, t2, direction, target),
                      backward_gauge);
      multiplyGaugeSpin(backward_gauge, backward_source, gauge_source, true);
      applyOneMinusGamma(direction, gauge_source, gamma_source, -1);
      for (int index = 0; index < kVector; ++index) {
        result[index] += gamma_source[index];
      }
      (void)delta_plus;
      (void)delta_minus;
    }
    scatterVector(result, output, siteLinear(x, y, z, t2));
  }

  void computeDslashDaggerInterior(const Complex<Real> *source,
                                   Complex<Real> *output, int target) {
    const int64_t count = geometry_.half_sites;
    const int64_t chunk = ompChunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t linear = 0; linear < count; ++linear) {
      int x = 0;
      int y = 0;
      int z = 0;
      int t2 = 0;
      geometry_.decodeHalf(linear, x, y, z, t2);
      if (x == 0 || y == 0 || z == 0 || t2 == 0 ||
          x == geometry_.local[0] - 1 ||
          y == geometry_.local[1] - 1 ||
          z == geometry_.local[2] - 1 ||
          t2 == geometry_.local[3] / 2 - 1) {
        continue;
      }
      computeDslashDaggerSite(source, output, x, y, z, t2, target);
    }
  }

  void computeDslashDaggerBoundary(const Complex<Real> *local_source,
                                   Complex<Real> *output, int target) {
    const int64_t count = geometry_.half_sites;
    const int64_t chunk = ompChunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t linear = 0; linear < count; ++linear) {
      int x = 0;
      int y = 0;
      int z = 0;
      int t2 = 0;
      geometry_.decodeHalf(linear, x, y, z, t2);
      const bool boundary =
          x == 0 || y == 0 || z == 0 || t2 == 0 ||
          x == geometry_.local[0] - 1 ||
          y == geometry_.local[1] - 1 ||
          z == geometry_.local[2] - 1 ||
          t2 == geometry_.local[3] / 2 - 1;
      if (!boundary) {
        continue;
      }
      computeDslashDaggerSite(local_source, output, x, y, z, t2, target);
    }
  }

  void computeDslashDaggerSite(const Complex<Real> *local_source,
                               Complex<Real> *output, int x, int y, int z,
                               int t2, int target) {
    using C = Complex<Real>;
    C result[kVector];
    for (int index = 0; index < kVector; ++index) {
      result[index] = C(0);
    }
    for (int direction = 0; direction < kDim; ++direction) {
      C transported[kVector];
      C gamma_source[kVector];

      // Adjoint of the forward hop: U_mu(x-mu)^dagger (1-gamma_mu).
      const VectorView<Real> previous_view =
          sourceAtDelta(local_source, x, y, z, t2, direction, -1, target);
      C previous_source[kVector];
      gatherVector(previous_view, previous_source);
      C previous_gauge[9];
      loadGaugeMatrix(backwardGauge(x, y, z, t2, direction, target),
                      previous_gauge);
      multiplyGaugeSpin(previous_gauge, previous_source, transported, true);
      applyOneMinusGamma(direction, transported, gamma_source, 1);
      for (int index = 0; index < kVector; ++index) {
        result[index] += gamma_source[index];
      }

      // Adjoint of the backward hop: U_mu(x) (1+gamma_mu).
      const VectorView<Real> next_view =
          sourceAtDelta(local_source, x, y, z, t2, direction, 1, target);
      C next_source[kVector];
      gatherVector(next_view, next_source);
      C current_gauge[9];
      loadGaugeMatrix(localGauge(target, direction, x, y, z, t2),
                      current_gauge);
      multiplyGaugeSpin(current_gauge, next_source, transported, false);
      applyOneMinusGamma(direction, transported, gamma_source, -1);
      for (int index = 0; index < kVector; ++index) {
        result[index] += gamma_source[index];
      }
    }
    scatterVector(result, output, siteLinear(x, y, z, t2));
  }

  GaugeView<Real> backwardGauge(int x, int y, int z, int t2,
                                int direction, int target) const {
    int delta[kDim] = {0, 0, 0, 0};
    delta[direction] = -1;
    return gaugeAtDelta(x, y, z, t2, target, direction, delta);
  }

  void bareDslash(Complex<Real> *output, const Complex<Real> *input,
                  int target) {
    const int source_parity = 1 - target;
    const int64_t count = geometry_.parity_vectors;
    const int64_t chunk = ompChunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < count; ++index) {
      output[index] = Complex<Real>(0);
    }
    postFermionHalo(input, source_parity);
    computeDslashInterior(input, output, target);
    if (!fermion_requests_.empty()) {
      MPI_Waitall(static_cast<int>(fermion_requests_.size()),
                  fermion_requests_.data(), MPI_STATUSES_IGNORE);
      fermion_requests_.clear();
    }
    computeDslashBoundary(input, output, target);
  }

  void bareDslashDagger(Complex<Real> *output, const Complex<Real> *input,
                        int target) {
    const int source_parity = 1 - target;
    const int64_t count = geometry_.parity_vectors;
    const int64_t chunk = ompChunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < count; ++index) {
      output[index] = Complex<Real>(0);
    }
    postFermionHalo(input, source_parity);
    computeDslashDaggerInterior(input, output, target);
    if (!fermion_requests_.empty()) {
      MPI_Waitall(static_cast<int>(fermion_requests_.size()),
                  fermion_requests_.data(), MPI_STATUSES_IGNORE);
      fermion_requests_.clear();
    }
    computeDslashDaggerBoundary(input, output, target);
  }

  void applySchur(const Complex<Real> *input, Complex<Real> *output,
                  int parity,
                  const std::array<const Complex<Real> *, 2> &clover,
                  const std::array<const Complex<Real> *, 2> &inverse,
                  Complex<Real> *scratch_a, Complex<Real> *scratch_b) {
    const int other = 1 - parity;
    bareDslash(scratch_a, input, other);
    applyOnsite(scratch_a, inverse[static_cast<std::size_t>(other)],
                scratch_b);
    bareDslash(scratch_a, scratch_b, parity);
    const Real k = kappa();
    const int64_t vectors = geometry_.parity_vectors;
    const int64_t chunk = ompChunk<Real>(vectors);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < vectors; ++index) {
      scratch_a[index] *= k * k;
    }
    applyOnsite(input, clover[static_cast<std::size_t>(parity)], output);
    for (int64_t index = 0; index < vectors; ++index) {
      output[index] -= scratch_a[index];
    }
  }

  void applyWilsonSchur(const Complex<Real> *input, Complex<Real> *output,
                        int parity, Complex<Real> *scratch_a,
                        Complex<Real> *scratch_b) {
    const int other = 1 - parity;
    bareDslash(scratch_a, input, other);
    bareDslash(scratch_b, scratch_a, parity);
    const Real coefficient = kappa() * kappa();
    const int64_t vectors = geometry_.parity_vectors;
    const int64_t chunk = ompChunk<Real>(vectors);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < vectors; ++index) {
      output[index] = input[index] - coefficient * scratch_b[index];
    }
  }

  void applyWilsonSchurDagger(const Complex<Real> *input,
                              Complex<Real> *output, int parity,
                              Complex<Real> *scratch_a,
                              Complex<Real> *scratch_b) {
    const int other = 1 - parity;
    bareDslashDagger(scratch_a, input, other);
    bareDslashDagger(scratch_b, scratch_a, parity);
    const Real coefficient = kappa() * kappa();
    const int64_t vectors = geometry_.parity_vectors;
    const int64_t chunk = ompChunk<Real>(vectors);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < vectors; ++index) {
      output[index] = input[index] - coefficient * scratch_b[index];
    }
  }

  void applyWilsonCgDslash(const Complex<Real> *input,
                           Complex<Real> *output, int parity) {
    applyWilsonSchur(input, work_a_.data(), parity, work_b_.data(),
                     work_c_.data());
    applyWilsonSchurDagger(work_a_.data(), output, parity, work_b_.data(),
                           work_c_.data());
  }

  Complex<Real> globalDot(const Complex<Real> *left,
                          const Complex<Real> *right,
                          int64_t count) const {
    Real real_sum = 0;
    Real imag_sum = 0;
    const int64_t chunk = ompChunk<Real>(count);
#pragma omp parallel for schedule(static, chunk) reduction(+ : real_sum, imag_sum)
    for (int64_t index = 0; index < count; ++index) {
      const Complex<Real> value = std::conj(left[index]) * right[index];
      real_sum += value.real();
      imag_sum += value.imag();
    }
    const Real local[2] = {real_sum, imag_sum};
    Real global[2] = {0, 0};
    const MPI_Datatype type =
        sizeof(Real) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Allreduce(local, global, 2, type, MPI_SUM, geometry_.comm);
    return Complex<Real>(global[0], global[1]);
  }

  Real globalNorm2(const Complex<Real> *field, int64_t count) const {
    Real local = 0;
    const int64_t chunk = ompChunk<Real>(count);
#pragma omp parallel for schedule(static, chunk) reduction(+ : local)
    for (int64_t index = 0; index < count; ++index) {
      local += std::norm(field[index]);
    }
    Real global = 0;
    const MPI_Datatype type =
        sizeof(Real) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Allreduce(&local, &global, 1, type, MPI_SUM, geometry_.comm);
    return global;
  }

  void applyFullOperator(
      const Complex<Real> *input, Complex<Real> *output,
      const std::array<const Complex<Real> *, 2> &clover,
      Complex<Real> *scratch) {
    const int64_t vectors = geometry_.parity_vectors;
    for (int parity = 0; parity < 2; ++parity) {
      const int other = 1 - parity;
      const Complex<Real> *source =
          input + static_cast<std::size_t>(other) * vectors;
      Complex<Real> *destination =
          output + static_cast<std::size_t>(parity) * vectors;
      bareDslash(scratch, source, parity);
      const Complex<Real> *onsite =
          input + static_cast<std::size_t>(parity) * vectors;
      applyOnsite(onsite, clover[static_cast<std::size_t>(parity)],
                  destination);
      const Real k = kappa();
      const int64_t chunk = ompChunk<Real>(vectors);
#pragma omp parallel for schedule(static, chunk)
      for (int64_t index = 0; index < vectors; ++index) {
        destination[index] -= k * scratch[index];
      }
      if (parity == 0 &&
          std::getenv("KUPL_DEBUG_OPERATOR") != nullptr) {
        log("OPERATOR", "source=(%g,%g) dslash=(%g,%g) result=(%g,%g)",
            static_cast<double>(source[0].real()),
            static_cast<double>(source[0].imag()),
            static_cast<double>(scratch[0].real()),
            static_cast<double>(scratch[0].imag()),
            static_cast<double>(destination[0].real()),
            static_cast<double>(destination[0].imag()));
      }
    }
  }

  void applyBlockInverse(
      const Complex<Real> *input, Complex<Real> *output,
      const std::array<const Complex<Real> *, 2> &inverse) {
    const int64_t vectors = geometry_.parity_vectors;
    for (int parity = 0; parity < 2; ++parity) {
      applyOnsite(
          input + static_cast<std::size_t>(parity) * vectors,
          inverse[static_cast<std::size_t>(parity)],
          output + static_cast<std::size_t>(parity) * vectors);
    }
  }

  int64_t laplacianSite(int x, int y, int z) const {
    return geometry_.siteLinear(x, y, z, 0);
  }

  std::size_t laplacianGaugeIndex(int row, int column, int direction, int x,
                                  int y, int z) const {
    int64_t index = row;
    index = index * kColor + column;
    index = index * 3 + direction;
    index = index * geometry_.local[0] + x;
    index = index * geometry_.local[1] + y;
    index = index * geometry_.local[2] + z;
    return static_cast<std::size_t>(index);
  }

  void decodeLaplacianSite(int64_t index, int &x, int &y, int &z) const {
    z = static_cast<int>(index % geometry_.local[2]);
    index /= geometry_.local[2];
    y = static_cast<int>(index % geometry_.local[1]);
    index /= geometry_.local[1];
    x = static_cast<int>(index);
  }

  void packLaplacianFace(int dim, int boundary, const Complex<Real> *source,
                         const Complex<Real> *gauge,
                         Complex<Real> *source_buffer,
                         Complex<Real> *gauge_buffer) {
    const int64_t count =
        geometry_.half_sites / surfaceExtent(dim);
    const int64_t chunk = ompChunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t face_site = 0; face_site < count; ++face_site) {
      int x = 0;
      int y = 0;
      int z = 0;
      int t2 = 0;
      decodeFace(face_site, dim, boundary, x, y, z, t2);
      const int64_t linear = laplacianSite(x, y, z);
      const int64_t face = face_site;
      for (int color = 0; color < kColor; ++color) {
        source_buffer[face * kColor + color] =
            source[static_cast<int64_t>(color) * geometry_.half_sites +
                   linear];
      }
      for (int row = 0; row < kColor; ++row) {
        for (int column = 0; column < kColor; ++column) {
          gauge_buffer[face * kGauge + row * kColor + column] =
              gauge[laplacianGaugeIndex(row, column, dim, x, y, z)];
        }
      }
    }
  }

  void postLaplacianHalo(const Complex<Real> *source,
                         const Complex<Real> *gauge) {
    laplacian_requests_.clear();
    laplacian_requests_.reserve(24);
    for (int dim = 0; dim < 3; ++dim) {
      if (geometry_.grid[dim] == 1) {
        continue;
      }
      auto &source_send_minus =
          laplacian_source_send_[faceOffset(dim, -1)];
      auto &source_send_plus =
          laplacian_source_send_[faceOffset(dim, 1)];
      auto &source_recv_minus =
          laplacian_source_recv_[faceOffset(dim, -1)];
      auto &source_recv_plus =
          laplacian_source_recv_[faceOffset(dim, 1)];
      auto &gauge_send_minus =
          laplacian_gauge_send_[faceOffset(dim, -1)];
      auto &gauge_send_plus =
          laplacian_gauge_send_[faceOffset(dim, 1)];
      auto &gauge_recv_minus =
          laplacian_gauge_recv_[faceOffset(dim, -1)];
      auto &gauge_recv_plus =
          laplacian_gauge_recv_[faceOffset(dim, 1)];

      packLaplacianFace(dim, 0, source, gauge, source_send_minus.data(),
                        gauge_send_minus.data());
      packLaplacianFace(dim, geometry_.local[dim] - 1, source, gauge,
                        source_send_plus.data(), gauge_send_plus.data());
      MPI_Request request = MPI_REQUEST_NULL;
      MPI_Irecv(source_recv_plus.data(),
                static_cast<int>(source_recv_plus.size() *
                                 sizeof(Complex<Real>)),
                MPI_BYTE, geometry_.neighbor_plus[dim],
                laplacianFaceTag(1, dim, -1), geometry_.comm, &request);
      laplacian_requests_.push_back(request);
      MPI_Irecv(source_recv_minus.data(),
                static_cast<int>(source_recv_minus.size() *
                                 sizeof(Complex<Real>)),
                MPI_BYTE, geometry_.neighbor_minus[dim],
                laplacianFaceTag(1, dim, 1), geometry_.comm, &request);
      laplacian_requests_.push_back(request);
      MPI_Irecv(gauge_recv_plus.data(),
                static_cast<int>(gauge_recv_plus.size() *
                                 sizeof(Complex<Real>)),
                MPI_BYTE, geometry_.neighbor_plus[dim],
                laplacianFaceTag(2, dim, -1), geometry_.comm, &request);
      laplacian_requests_.push_back(request);
      MPI_Irecv(gauge_recv_minus.data(),
                static_cast<int>(gauge_recv_minus.size() *
                                 sizeof(Complex<Real>)),
                MPI_BYTE, geometry_.neighbor_minus[dim],
                laplacianFaceTag(2, dim, 1), geometry_.comm, &request);
      laplacian_requests_.push_back(request);
      MPI_Isend(source_send_plus.data(),
                static_cast<int>(source_send_plus.size() *
                                 sizeof(Complex<Real>)),
                MPI_BYTE, geometry_.neighbor_plus[dim],
                laplacianFaceTag(1, dim, 1), geometry_.comm, &request);
      laplacian_requests_.push_back(request);
      MPI_Isend(source_send_minus.data(),
                static_cast<int>(source_send_minus.size() *
                                 sizeof(Complex<Real>)),
                MPI_BYTE, geometry_.neighbor_minus[dim],
                laplacianFaceTag(1, dim, -1), geometry_.comm, &request);
      laplacian_requests_.push_back(request);
      MPI_Isend(gauge_send_plus.data(),
                static_cast<int>(gauge_send_plus.size() *
                                 sizeof(Complex<Real>)),
                MPI_BYTE, geometry_.neighbor_plus[dim],
                laplacianFaceTag(2, dim, 1), geometry_.comm, &request);
      laplacian_requests_.push_back(request);
      MPI_Isend(gauge_send_minus.data(),
                static_cast<int>(gauge_send_minus.size() *
                                 sizeof(Complex<Real>)),
                MPI_BYTE, geometry_.neighbor_minus[dim],
                laplacianFaceTag(2, dim, -1), geometry_.comm, &request);
      laplacian_requests_.push_back(request);
    }
  }

  void computeLaplacianLocal(const Complex<Real> *source,
                             const Complex<Real> *gauge,
                             Complex<Real> *output) {
    const int64_t count = geometry_.half_sites;
    const int64_t chunk = ompChunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t site = 0; site < count; ++site) {
      int x = 0;
      int y = 0;
      int z = 0;
      decodeLaplacianSite(site, x, y, z);
      int coordinate[3] = {x, y, z};
      Complex<Real> result[kColor];
      for (int color = 0; color < kColor; ++color) {
        result[color] =
            Real(6) * source[static_cast<int64_t>(color) * count + site];
      }
      for (int dim = 0; dim < 3; ++dim) {
        const bool defer_plus =
            geometry_.grid[dim] > 1 &&
            coordinate[dim] == geometry_.local[dim] - 1;
        if (!defer_plus) {
          int plus[3] = {x, y, z};
          plus[dim] = (plus[dim] + 1) % geometry_.local[dim];
          const int64_t plus_site = laplacianSite(plus[0], plus[1], plus[2]);
          for (int row = 0; row < kColor; ++row) {
            Complex<Real> value(0);
            for (int column = 0; column < kColor; ++column) {
              const Complex<Real> link =
                  gauge[laplacianGaugeIndex(row, column, dim, x, y, z)];
              value +=
                  link *
                  source[static_cast<int64_t>(column) * count + plus_site];
            }
            result[row] -= value;
          }
        }

        const bool defer_minus =
            geometry_.grid[dim] > 1 && coordinate[dim] == 0;
        if (!defer_minus) {
          int minus[3] = {x, y, z};
          minus[dim] =
              (minus[dim] - 1 + geometry_.local[dim]) % geometry_.local[dim];
          const int64_t minus_site =
              laplacianSite(minus[0], minus[1], minus[2]);
          for (int row = 0; row < kColor; ++row) {
            Complex<Real> value(0);
            for (int column = 0; column < kColor; ++column) {
              const Complex<Real> link = std::conj(
                  gauge[laplacianGaugeIndex(column, row, dim, minus[0],
                                            minus[1], minus[2])]);
              value +=
                  link *
                  source[static_cast<int64_t>(column) * count + minus_site];
            }
            result[row] -= value;
          }
        }
      }
      for (int color = 0; color < kColor; ++color) {
        output[static_cast<int64_t>(color) * count + site] = result[color];
      }
    }
  }

  void subtractLaplacianBoundary(const Complex<Real> *gauge,
                                 Complex<Real> *output) {
    const int64_t count = geometry_.half_sites;
    const int64_t chunk = ompChunk<Real>(count);
    for (int dim = 0; dim < 3; ++dim) {
      if (geometry_.grid[dim] == 1) {
        continue;
      }
      for (int sign : {-1, 1}) {
        const auto &source_buffer =
            laplacian_source_recv_[faceOffset(dim, sign)];
        const auto &gauge_buffer =
            laplacian_gauge_recv_[faceOffset(dim, sign)];
#pragma omp parallel for schedule(static, chunk)
        for (int64_t site = 0; site < count; ++site) {
          int x = 0;
          int y = 0;
          int z = 0;
          decodeLaplacianSite(site, x, y, z);
          const int coordinate[3] = {x, y, z};
          const int boundary = sign > 0 ? geometry_.local[dim] - 1 : 0;
          if (coordinate[dim] != boundary) {
            continue;
          }
          const int64_t face =
              geometry_.faceIndex(dim, x, y, z, 0);
          const Complex<Real> *source =
              source_buffer.data() + face * kColor;
          const Complex<Real> *link =
              gauge_buffer.data() + face * kGauge;
          for (int row = 0; row < kColor; ++row) {
            Complex<Real> value(0);
            for (int column = 0; column < kColor; ++column) {
              Complex<Real> factor;
              if (sign > 0) {
                factor =
                    gauge[laplacianGaugeIndex(row, column, dim, x, y, z)];
              } else {
                factor = std::conj(link[column * kColor + row]);
              }
              value += factor * source[column];
            }
            output[static_cast<int64_t>(row) * count + site] -= value;
          }
        }
      }
    }
  }

  void applyLaplacian(Complex<Real> *output, const Complex<Real> *input,
                      const Complex<Real> *gauge) {
    if (output == nullptr || input == nullptr || gauge == nullptr) {
      throw std::runtime_error("Laplacian received a null pointer");
    }
    postLaplacianHalo(input, gauge);
    computeLaplacianLocal(input, gauge, output);
    if (!laplacian_requests_.empty()) {
      MPI_Waitall(static_cast<int>(laplacian_requests_.size()),
                  laplacian_requests_.data(), MPI_STATUSES_IGNORE);
      laplacian_requests_.clear();
    }
    subtractLaplacianBoundary(gauge, output);
  }

  int64_t coarseVolume(const std::array<int, kDim> &dims) const {
    int64_t volume = 1;
    for (int dim = 0; dim < kDim; ++dim) {
      volume *= dims[dim];
    }
    return volume;
  }

  int64_t coarseSiteIndex(const std::array<int, kDim> &dims, int x, int y,
                          int z, int t) const {
    return ((static_cast<int64_t>(x) * dims[1] + y) * dims[2] + z) *
               dims[3] +
           t;
  }

  void decodeCoarseSite(int64_t site, const std::array<int, kDim> &dims,
                        int &x, int &y, int &z, int &t) const {
    t = static_cast<int>(site % dims[3]);
    site /= dims[3];
    z = static_cast<int>(site % dims[2]);
    site /= dims[2];
    y = static_cast<int>(site % dims[1]);
    site /= dims[1];
    x = static_cast<int>(site);
  }

  int64_t coarseFaceSites(int direction,
                          const std::array<int, kDim> &dims) const {
    int64_t count = 1;
    for (int dim = 0; dim < kDim; ++dim) {
      if (coarseShiftComponent(direction, dim) == 0) {
        count *= dims[dim];
      }
    }
    return count;
  }

  bool coarseDirectionDistributed(int direction) const {
    for (int dim = 0; dim < kDim; ++dim) {
      if (coarseShiftComponent(direction, dim) != 0 &&
          geometry_.grid[dim] > 1) {
        return true;
      }
    }
    return false;
  }

  void decodeCoarseFace(
      int direction, int64_t face, const std::array<int, kDim> &dims,
      std::array<int, kDim> &coordinate) const {
    for (int dim = kDim - 1; dim >= 0; --dim) {
      const int shift = coarseShiftComponent(direction, dim);
      if (shift == 0) {
        coordinate[dim] = static_cast<int>(face % dims[dim]);
        face /= dims[dim];
      } else {
        coordinate[dim] = shift > 0 ? dims[dim] - 1 : 0;
      }
    }
  }

  int64_t coarseFaceIndex(
      int direction, const std::array<int, kDim> &coordinate,
      const std::array<int, kDim> &dims) const {
    int64_t index = 0;
    int64_t stride = 1;
    for (int dim = kDim - 1; dim >= 0; --dim) {
      if (coarseShiftComponent(direction, dim) == 0) {
        index += static_cast<int64_t>(coordinate[dim]) * stride;
        stride *= dims[dim];
      }
    }
    return index;
  }

  void ensureCoarseHalo(int E, const std::array<int, kDim> &dims) {
    if (E <= 0) {
      throw std::runtime_error("coarse halo E must be positive");
    }
    for (int dim = 0; dim < kDim; ++dim) {
      if (dims[dim] <= 0) {
        throw std::runtime_error(
            "coarse halo dimensions must be positive");
      }
    }
    if (coarse_halo_E_ == E && coarse_halo_dims_ == dims) {
      return;
    }
    int64_t max_face = 0;
    for (int direction = 0; direction < kCoarseDirections; ++direction) {
      max_face =
          std::max(max_face, coarseFaceSites(direction, dims));
    }
    const int64_t elements = static_cast<int64_t>(E) * max_face;
    if (elements <= 0 ||
        elements > std::numeric_limits<int>::max() /
                       static_cast<int64_t>(sizeof(Complex<Real>))) {
      throw std::runtime_error("coarse halo message is too large");
    }
    coarse_halo_E_ = E;
    coarse_halo_dims_ = dims;
    coarse_halo_face_size_ = static_cast<int>(max_face);
    for (int direction = 0; direction < kCoarseDirections; ++direction) {
      coarse_halo_offsets_[direction] = 0;
      coarse_halo_send_[direction].resize(static_cast<std::size_t>(elements));
      coarse_halo_recv_[direction].resize(static_cast<std::size_t>(elements));
    }
  }

  void packCoarseDirection(const Complex<Real> *input, int E,
                           const std::array<int, kDim> &dims,
                           int direction) {
    const int64_t face_sites = coarseFaceSites(direction, dims);
    const int64_t volume = coarseVolume(dims);
    auto &buffer = coarse_halo_send_[direction];
    const int offset = coarse_halo_offsets_[direction];
    const int64_t chunk = ompChunk<Real>(face_sites);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t face = 0; face < face_sites; ++face) {
      std::array<int, kDim> coordinate{};
      decodeCoarseFace(direction, face, dims, coordinate);
      const int64_t site = coarseSiteIndex(
          dims, coordinate[0], coordinate[1], coordinate[2], coordinate[3]);
      for (int e = 0; e < E; ++e) {
        buffer[static_cast<std::size_t>(offset + face * E + e)] =
            input[static_cast<int64_t>(e) * volume + site];
      }
    }
  }

  int coarsePeer(int direction) const {
    std::array<int, kDim> coordinate = geometry_.coord;
    for (int dim = 0; dim < kDim; ++dim) {
      const int shift = coarseShiftComponent(direction, dim);
      if (shift != 0 && geometry_.grid[dim] > 1) {
        coordinate[dim] =
            (coordinate[dim] + shift + geometry_.grid[dim]) %
            geometry_.grid[dim];
      }
    }
    int peer = geometry_.rank;
    MPI_Cart_rank(geometry_.comm, coordinate.data(), &peer);
    return peer;
  }

  void exchangeCoarseHalo(const Complex<Real> *input, int E,
                          const std::array<int, kDim> &dims) {
    ensureCoarseHalo(E, dims);
    std::vector<MPI_Request> requests;
    requests.reserve(2 * kCoarseDirections);
    for (int direction = 0; direction < kCoarseDirections; ++direction) {
      if (!coarseDirectionDistributed(direction)) {
        continue;
      }
      packCoarseDirection(input, E, dims, direction);
      const int peer = coarsePeer(direction);
      auto &receive = coarse_halo_recv_[direction];
      if (peer == geometry_.rank) {
        continue;
      }
      MPI_Request request = MPI_REQUEST_NULL;
      MPI_Irecv(
          receive.data(),
          static_cast<int>(receive.size() * sizeof(Complex<Real>)),
          MPI_BYTE, peer, 7100 + (direction ^ 1), geometry_.comm, &request);
      requests.push_back(request);
    }
    for (int direction = 0; direction < kCoarseDirections; ++direction) {
      if (!coarseDirectionDistributed(direction)) {
        continue;
      }
      const int peer = coarsePeer(direction);
      auto &send = coarse_halo_send_[direction];
      if (peer == geometry_.rank) {
        coarse_halo_recv_[direction] = send;
        continue;
      }
      MPI_Request request = MPI_REQUEST_NULL;
      MPI_Isend(
          send.data(),
          static_cast<int>(send.size() * sizeof(Complex<Real>)),
          MPI_BYTE, peer, 7100 + direction, geometry_.comm, &request);
      requests.push_back(request);
    }
    if (!requests.empty()) {
      MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                  MPI_STATUSES_IGNORE);
    }
  }

  Complex<Real> coarseNeighbor(
      const Complex<Real> *input, int E, const std::array<int, kDim> &dims,
      const std::array<int, kDim> &coordinate, const int shift[kDim],
      int e) const {
    std::array<int, kDim> target = coordinate;
    int crossing[kDim] = {0, 0, 0, 0};
    for (int dim = 0; dim < kDim; ++dim) {
      target[dim] += shift[dim];
      if (target[dim] < 0) {
        if (geometry_.grid[dim] > 1) {
          crossing[dim] = shift[dim];
        }
        target[dim] = dims[dim] - 1;
      } else if (target[dim] >= dims[dim]) {
        if (geometry_.grid[dim] > 1) {
          crossing[dim] = shift[dim];
        }
        target[dim] = 0;
      }
    }
    const int direction = coarseDirectionForShift(crossing);
    if (direction >= 0) {
      const int64_t face =
          coarseFaceIndex(direction, target, dims);
      return coarse_halo_recv_[direction][
          static_cast<std::size_t>(
              coarse_halo_offsets_[direction] + face * E + e)];
    }
    const int64_t volume = coarseVolume(dims);
    const int64_t site = coarseSiteIndex(
        dims, target[0], target[1], target[2], target[3]);
    return input[static_cast<int64_t>(e) * volume + site];
  }

  std::size_t multigridNullIndex(
      int E_idx, int e_idx, int x, int y, int z, int t, int e, int Xf,
      int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc) const {
    const int bx = Xf / Xc;
    const int by = Yf / Yc;
    const int bz = Zf / Zc;
    const int bt = Tf / Tc;
    const int xc = x / bx;
    const int yc = y / by;
    const int zc = z / bz;
    const int tc = t / bt;
    const int dx = x - xc * bx;
    const int dy = y - yc * by;
    const int dz = z - zc * bz;
    const int dt = t - tc * bt;
    std::size_t index = static_cast<std::size_t>(E_idx);
    index = index * static_cast<std::size_t>(e) +
            static_cast<std::size_t>(e_idx);
    index = index * static_cast<std::size_t>(Xc) +
            static_cast<std::size_t>(xc);
    index = index * static_cast<std::size_t>(bx) +
            static_cast<std::size_t>(dx);
    index = index * static_cast<std::size_t>(Yc) +
            static_cast<std::size_t>(yc);
    index = index * static_cast<std::size_t>(by) +
            static_cast<std::size_t>(dy);
    index = index * static_cast<std::size_t>(Zc) +
            static_cast<std::size_t>(zc);
    index = index * static_cast<std::size_t>(bz) +
            static_cast<std::size_t>(dz);
    index = index * static_cast<std::size_t>(Tc) +
            static_cast<std::size_t>(tc);
    index = index * static_cast<std::size_t>(bt) +
            static_cast<std::size_t>(dt);
    return index;
  }

  void validateMultigridTransfer(int E, int e, int Xf, int Yf, int Zf,
                                 int Tf, int Xc, int Yc, int Zc,
                                 int Tc) const {
    if (E <= 0 || e <= 0 || Xf <= 0 || Yf <= 0 || Zf <= 0 || Tf <= 0 ||
        Xc <= 0 || Yc <= 0 || Zc <= 0 || Tc <= 0) {
      throw std::runtime_error("multigrid transfer dimensions must be positive");
    }
    if (Xf % Xc != 0 || Yf % Yc != 0 || Zf % Zc != 0 || Tf % Tc != 0) {
      throw std::runtime_error(
          "fine multigrid dimensions must be divisible by coarse dimensions");
    }
  }

  void applyMultigridRestrict(
      Complex<Real> *coarse_out, const Complex<Real> *fine_in,
      const Complex<Real> *null_vectors, int E, int e, int Xf, int Yf,
      int Zf, int Tf, int Xc, int Yc, int Zc, int Tc) {
    if (coarse_out == nullptr || fine_in == nullptr ||
        null_vectors == nullptr) {
      throw std::runtime_error("multigrid restrict received a null pointer");
    }
    validateMultigridTransfer(E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
    const int bx = Xf / Xc;
    const int by = Yf / Yc;
    const int bz = Zf / Zc;
    const int bt = Tf / Tc;
    const int64_t fine_volume =
        static_cast<int64_t>(Xf) * Yf * Zf * Tf;
    const int64_t coarse_volume =
        static_cast<int64_t>(Xc) * Yc * Zc * Tc;
    const int64_t output_count = static_cast<int64_t>(E) * coarse_volume;
    const int64_t chunk = ompChunk<Real>(output_count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t output = 0; output < output_count; ++output) {
      const int E_idx = static_cast<int>(output / coarse_volume);
      const int64_t coarse_site = output % coarse_volume;
      int xc = 0;
      int yc = 0;
      int zc = 0;
      int tc = 0;
      decodeCoarseSite(coarse_site,
                       {Xc, Yc, Zc, Tc}, xc, yc, zc, tc);
      Complex<Real> sum(0);
      for (int dx = 0; dx < bx; ++dx) {
        for (int dy = 0; dy < by; ++dy) {
          for (int dz = 0; dz < bz; ++dz) {
            for (int dt = 0; dt < bt; ++dt) {
              const int x = xc * bx + dx;
              const int y = yc * by + dy;
              const int z = zc * bz + dz;
              const int t = tc * bt + dt;
              const int64_t fine_site =
                  ((static_cast<int64_t>(x) * Yf + y) * Zf + z) * Tf + t;
              for (int e_idx = 0; e_idx < e; ++e_idx) {
                const int64_t fine_index =
                    static_cast<int64_t>(e_idx) * fine_volume + fine_site;
                const std::size_t null_index = multigridNullIndex(
                    E_idx, e_idx, x, y, z, t, e, Xf, Yf, Zf, Tf, Xc, Yc,
                    Zc, Tc);
                sum += std::conj(null_vectors[null_index]) *
                       fine_in[fine_index];
              }
            }
          }
        }
      }
      coarse_out[output] = sum;
    }
  }

  void applyMultigridProlong(
      Complex<Real> *fine_out, const Complex<Real> *coarse_in,
      const Complex<Real> *null_vectors, int E, int e, int Xf, int Yf,
      int Zf, int Tf, int Xc, int Yc, int Zc, int Tc) {
    if (fine_out == nullptr || coarse_in == nullptr ||
        null_vectors == nullptr) {
      throw std::runtime_error("multigrid prolong received a null pointer");
    }
    validateMultigridTransfer(E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
    const int bx = Xf / Xc;
    const int by = Yf / Yc;
    const int bz = Zf / Zc;
    const int bt = Tf / Tc;
    const int64_t fine_volume =
        static_cast<int64_t>(Xf) * Yf * Zf * Tf;
    const int64_t coarse_volume =
        static_cast<int64_t>(Xc) * Yc * Zc * Tc;
    const int64_t output_count = static_cast<int64_t>(e) * fine_volume;
    const int64_t chunk = ompChunk<Real>(output_count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t output = 0; output < output_count; ++output) {
      const int e_idx = static_cast<int>(output / fine_volume);
      const int64_t fine_site = output % fine_volume;
      int x = 0;
      int y = 0;
      int z = 0;
      int t = 0;
      decodeCoarseSite(fine_site,
                       {Xf, Yf, Zf, Tf}, x, y, z, t);
      const int xc = x / bx;
      const int yc = y / by;
      const int zc = z / bz;
      const int tc = t / bt;
      const int64_t coarse_site =
          ((static_cast<int64_t>(xc) * Yc + yc) * Zc + zc) * Tc + tc;
      Complex<Real> sum(0);
      for (int E_idx = 0; E_idx < E; ++E_idx) {
        const std::size_t null_index = multigridNullIndex(
            E_idx, e_idx, x, y, z, t, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
        sum += null_vectors[null_index] *
               coarse_in[static_cast<int64_t>(E_idx) * coarse_volume +
                         coarse_site];
      }
      fine_out[output] = sum;
    }
  }

  void validateCoarseOperator(const void *out, const void *in, int E, int X,
                              int Y, int Z, int T) const {
    if (out == nullptr || in == nullptr) {
      throw std::runtime_error("coarse dslash received a null pointer");
    }
    if (E <= 0 || X <= 0 || Y <= 0 || Z <= 0 || T <= 0) {
      throw std::runtime_error(
          "coarse dslash dimensions must be positive");
    }
  }

  void applyMultigridCoarseDslash(
      Complex<Real> *out, const Complex<Real> *in,
      const Complex<Real> *hopping, const Complex<Real> *sitting, int E,
      int X, int Y, int Z, int T) {
    validateCoarseOperator(out, in, E, X, Y, Z, T);
    if (hopping == nullptr || sitting == nullptr) {
      throw std::runtime_error("coarse dslash received a null stencil");
    }
    const std::array<int, kDim> dims = {X, Y, Z, T};
    exchangeCoarseHalo(in, E, dims);
    const int64_t volume = coarseVolume(dims);
    const int64_t output_count = static_cast<int64_t>(E) * volume;
    const int64_t chunk = ompChunk<Real>(output_count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t output = 0; output < output_count; ++output) {
      const int E_out = static_cast<int>(output / volume);
      const int64_t site = output % volume;
      std::array<int, kDim> coordinate{};
      decodeCoarseSite(site, dims, coordinate[0], coordinate[1],
                       coordinate[2], coordinate[3]);
      Complex<Real> sum(0);
      for (int e = 0; e < E; ++e) {
        const int64_t on_site =
            (static_cast<int64_t>(E_out) * E + e) * volume + site;
        sum += sitting[on_site] * in[static_cast<int64_t>(e) * volume + site];
      }
      for (int dim = 0; dim < kDim; ++dim) {
        for (int sign_index = 0; sign_index < 2; ++sign_index) {
          const int sign = sign_index == 0 ? 1 : -1;
          int shift[kDim] = {0, 0, 0, 0};
          shift[dim] = sign;
          for (int e = 0; e < E; ++e) {
            const int64_t hop =
                (static_cast<int64_t>(sign_index * kDim + dim) * E * E +
                 E_out * E + e) *
                    volume +
                site;
            sum += hopping[hop] *
                   coarseNeighbor(in, E, dims, coordinate, shift, e);
          }
        }
      }
      out[output] = sum;
    }
  }

  void applyMultigridCoarseDslashWide(
      Complex<Real> *out, const Complex<Real> *in,
      const Complex<Real> *sitting, const Complex<Real> *hop_nn,
      const Complex<Real> *hop_diag, int E, int X, int Y, int Z, int T) {
    validateCoarseOperator(out, in, E, X, Y, Z, T);
    if (sitting == nullptr || hop_nn == nullptr || hop_diag == nullptr) {
      throw std::runtime_error("wide coarse dslash received a null stencil");
    }
    const std::array<int, kDim> dims = {X, Y, Z, T};
    exchangeCoarseHalo(in, E, dims);
    const int64_t volume = coarseVolume(dims);
    const int64_t output_count = static_cast<int64_t>(E) * volume;
    const int64_t chunk = ompChunk<Real>(output_count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t output = 0; output < output_count; ++output) {
      const int E_out = static_cast<int>(output / volume);
      const int64_t site = output % volume;
      std::array<int, kDim> coordinate{};
      decodeCoarseSite(site, dims, coordinate[0], coordinate[1],
                       coordinate[2], coordinate[3]);
      Complex<Real> sum(0);
      for (int e = 0; e < E; ++e) {
        const int64_t on_site =
            (static_cast<int64_t>(E_out) * E + e) * volume + site;
        sum += sitting[on_site] * in[static_cast<int64_t>(e) * volume + site];
      }
      for (int dim = 0; dim < kDim; ++dim) {
        for (int sign_index = 0; sign_index < 2; ++sign_index) {
          const int sign = sign_index == 0 ? 1 : -1;
          int shift[kDim] = {0, 0, 0, 0};
          shift[dim] = sign;
          for (int e = 0; e < E; ++e) {
            const int64_t hop =
                (static_cast<int64_t>(sign_index * kDim + dim) * E * E +
                 E_out * E + e) *
                    volume +
                site;
            sum += hop_nn[hop] *
                   coarseNeighbor(in, E, dims, coordinate, shift, e);
          }
        }
      }
      static constexpr int pair_first[6] = {0, 0, 0, 1, 1, 2};
      static constexpr int pair_second[6] = {1, 2, 3, 2, 3, 3};
      for (int pair = 0; pair < 6; ++pair) {
        for (int first_sign_index = 0; first_sign_index < 2;
             ++first_sign_index) {
          for (int second_sign_index = 0; second_sign_index < 2;
               ++second_sign_index) {
            int shift[kDim] = {0, 0, 0, 0};
            shift[pair_first[pair]] =
                first_sign_index == 0 ? 1 : -1;
            shift[pair_second[pair]] =
                second_sign_index == 0 ? 1 : -1;
            for (int e = 0; e < E; ++e) {
              const int64_t diagonal =
                  ((static_cast<int64_t>(first_sign_index * 2 +
                                         second_sign_index) *
                        6 +
                    pair) *
                       E * E +
                   E_out * E + e) *
                      volume +
                  site;
              sum += hop_diag[diagonal] *
                     coarseNeighbor(in, E, dims, coordinate, shift, e);
            }
          }
        }
      }
      out[output] = sum;
    }
  }

  void applyFullWilsonOperator(const Complex<Real> *input,
                               Complex<Real> *output,
                               Complex<Real> *scratch) {
    const int64_t vectors = geometry_.parity_vectors;
    const int64_t chunk = ompChunk<Real>(vectors);
    for (int parity = 0; parity < 2; ++parity) {
      const int other = 1 - parity;
      const Complex<Real> *source =
          input + static_cast<std::size_t>(other) * vectors;
      Complex<Real> *destination =
          output + static_cast<std::size_t>(parity) * vectors;
      bareDslash(scratch, source, parity);
#pragma omp parallel for schedule(static, chunk)
      for (int64_t index = 0; index < vectors; ++index) {
        destination[index] =
            input[static_cast<std::size_t>(parity) * vectors + index] -
            kappa() * scratch[index];
      }
    }
  }

  template <typename Operator>
  void solveCompactBiStabCg(Complex<Real> *solution,
                            const Complex<Real> *rhs, int64_t count,
                            Operator &&apply_operator, const char *label,
                            Real relative_tolerance = Real(-1)) {
    using C = Complex<Real>;
    const int64_t chunk = ompChunk<Real>(count);
    const Real solve_tolerance =
        relative_tolerance > Real(0) ? relative_tolerance : atol();
    const Real rhs_norm = std::sqrt(globalNorm2(rhs, count));
    const Real recurrence_tolerance =
        std::max(solve_tolerance,
                 std::numeric_limits<Real>::epsilon() * Real(4)) *
        rhs_norm;
    const Real acceptance_tolerance =
        std::max(solve_tolerance, residualFloor()) * rhs_norm;
    if (rhs_norm == Real(0)) {
#pragma omp parallel for schedule(static, chunk)
      for (int64_t index = 0; index < count; ++index) {
        solution[index] = C(0);
      }
      last_iterations_ = 0;
      return;
    }
    if (solve_tolerance <= Real(0) || parameters_.get(kMaxIter) <= 0) {
      throw std::runtime_error(std::string(label) +
                               ": invalid atol/max_iter configuration");
    }

    apply_operator(solution, v_.data());
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < count; ++index) {
      r_[static_cast<std::size_t>(index)] =
          rhs[index] - v_[static_cast<std::size_t>(index)];
      rhat_[static_cast<std::size_t>(index)] =
          r_[static_cast<std::size_t>(index)];
      p_[static_cast<std::size_t>(index)] = C(0);
      v_[static_cast<std::size_t>(index)] = C(0);
    }

    C rho_old(1);
    C alpha(1);
    C omega(1);
    last_iterations_ = 0;
    const int max_iterations = parameters_.get(kMaxIter);
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
      const Real residual = std::sqrt(globalNorm2(r_.data(), count));
      if (residual <= recurrence_tolerance) {
        last_iterations_ = iteration;
        break;
      }
      const C rho = globalDot(rhat_.data(), r_.data(), count);
      if (std::abs(rho) <= std::numeric_limits<Real>::min() ||
          std::abs(omega) <= std::numeric_limits<Real>::min()) {
        throw std::runtime_error(std::string(label) +
                                 ": BiStabCG scalar breakdown");
      }
      const C beta = (rho / rho_old) * (alpha / omega);
#pragma omp parallel for schedule(static, chunk)
      for (int64_t index = 0; index < count; ++index) {
        p_[static_cast<std::size_t>(index)] =
            r_[static_cast<std::size_t>(index)] +
            beta * (p_[static_cast<std::size_t>(index)] -
                    omega * v_[static_cast<std::size_t>(index)]);
      }

      apply_operator(p_.data(), v_.data());
      const C rhat_v = globalDot(rhat_.data(), v_.data(), count);
      if (std::abs(rhat_v) <= std::numeric_limits<Real>::min()) {
        throw std::runtime_error(std::string(label) +
                                 ": BiStabCG alpha breakdown");
      }
      alpha = rho / rhat_v;
#pragma omp parallel for schedule(static, chunk)
      for (int64_t index = 0; index < count; ++index) {
        s_[static_cast<std::size_t>(index)] =
            r_[static_cast<std::size_t>(index)] -
            alpha * v_[static_cast<std::size_t>(index)];
      }
      if (std::sqrt(globalNorm2(s_.data(), count)) <=
          recurrence_tolerance) {
#pragma omp parallel for schedule(static, chunk)
        for (int64_t index = 0; index < count; ++index) {
          solution[index] += alpha * p_[static_cast<std::size_t>(index)];
          r_[static_cast<std::size_t>(index)] =
              s_[static_cast<std::size_t>(index)];
        }
        last_iterations_ = iteration + 1;
        break;
      }

      apply_operator(s_.data(), t_.data());
      const C tt = globalDot(t_.data(), t_.data(), count);
      if (std::abs(tt) <= std::numeric_limits<Real>::min()) {
        throw std::runtime_error(std::string(label) +
                                 ": BiStabCG omega breakdown");
      }
      omega = globalDot(t_.data(), s_.data(), count) / tt;
#pragma omp parallel for schedule(static, chunk)
      for (int64_t index = 0; index < count; ++index) {
        solution[index] +=
            alpha * p_[static_cast<std::size_t>(index)] +
            omega * s_[static_cast<std::size_t>(index)];
        r_[static_cast<std::size_t>(index)] =
            s_[static_cast<std::size_t>(index)] -
            omega * t_[static_cast<std::size_t>(index)];
      }
      rho_old = rho;
      last_iterations_ = iteration + 1;
    }

    apply_operator(solution, v_.data());
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < count; ++index) {
      r_[static_cast<std::size_t>(index)] =
          rhs[index] - v_[static_cast<std::size_t>(index)];
    }
    const Real true_residual =
        std::sqrt(globalNorm2(r_.data(), count));
    log(label,
        "iterations=%d residual=%g recurrence_tolerance=%g "
        "acceptance_tolerance=%g relative=%g",
        last_iterations_, static_cast<double>(true_residual),
        static_cast<double>(recurrence_tolerance),
        static_cast<double>(acceptance_tolerance),
        static_cast<double>(true_residual / rhs_norm));
    if (!std::isfinite(true_residual) ||
        true_residual > acceptance_tolerance) {
      throw std::runtime_error(std::string(label) +
                               ": non-finite or out-of-tolerance true residual");
    }
  }

  template <typename Operator>
  void solveCompactCg(Complex<Real> *solution, const Complex<Real> *rhs,
                      int64_t count, Operator &&apply_operator,
                      const char *label, Real relative_tolerance) {
    using C = Complex<Real>;
    const int64_t chunk = ompChunk<Real>(count);
    const Real rhs_norm = std::sqrt(globalNorm2(rhs, count));
    const Real recurrence_tolerance =
        std::max(relative_tolerance,
                 std::numeric_limits<Real>::epsilon() * Real(4)) *
        rhs_norm;
    const Real acceptance_tolerance =
        std::max(relative_tolerance, residualFloor()) * rhs_norm;
    if (rhs_norm == Real(0)) {
#pragma omp parallel for schedule(static, chunk)
      for (int64_t index = 0; index < count; ++index) {
        solution[index] = C(0);
      }
      last_iterations_ = 0;
      return;
    }
    if (atol() <= Real(0) || parameters_.get(kMaxIter) <= 0) {
      throw std::runtime_error(std::string(label) +
                               ": invalid atol/max_iter configuration");
    }

    apply_operator(solution, v_.data());
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < count; ++index) {
      r_[static_cast<std::size_t>(index)] =
          rhs[index] - v_[static_cast<std::size_t>(index)];
      p_[static_cast<std::size_t>(index)] =
          r_[static_cast<std::size_t>(index)];
    }
    Real rho = globalNorm2(r_.data(), count);
    last_iterations_ = 0;
    const int max_iterations = parameters_.get(kMaxIter);
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
      if (std::sqrt(rho) <= recurrence_tolerance) {
        last_iterations_ = iteration;
        break;
      }
      apply_operator(p_.data(), v_.data());
      const C denominator = globalDot(p_.data(), v_.data(), count);
      if (std::abs(denominator) <= std::numeric_limits<Real>::min()) {
        throw std::runtime_error(std::string(label) +
                                 ": CG alpha breakdown");
      }
      const C alpha = C(rho, 0) / denominator;
#pragma omp parallel for schedule(static, chunk)
      for (int64_t index = 0; index < count; ++index) {
        solution[index] += alpha * p_[static_cast<std::size_t>(index)];
        r_[static_cast<std::size_t>(index)] -=
            alpha * v_[static_cast<std::size_t>(index)];
      }
      const Real rho_next = globalNorm2(r_.data(), count);
      if (rho <= std::numeric_limits<Real>::min()) {
        throw std::runtime_error(std::string(label) +
                                 ": CG beta breakdown");
      }
      const Real beta = rho_next / rho;
#pragma omp parallel for schedule(static, chunk)
      for (int64_t index = 0; index < count; ++index) {
        p_[static_cast<std::size_t>(index)] =
            r_[static_cast<std::size_t>(index)] +
            beta * p_[static_cast<std::size_t>(index)];
      }
      rho = rho_next;
      last_iterations_ = iteration + 1;
    }

    apply_operator(solution, v_.data());
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < count; ++index) {
      r_[static_cast<std::size_t>(index)] =
          rhs[index] - v_[static_cast<std::size_t>(index)];
    }
    const Real true_residual =
        std::sqrt(globalNorm2(r_.data(), count));
    log(label,
        "iterations=%d residual=%g recurrence_tolerance=%g "
        "acceptance_tolerance=%g relative=%g",
        last_iterations_, static_cast<double>(true_residual),
        static_cast<double>(recurrence_tolerance),
        static_cast<double>(acceptance_tolerance),
        static_cast<double>(true_residual / rhs_norm));
    if (!std::isfinite(true_residual) ||
        true_residual > acceptance_tolerance) {
      throw std::runtime_error(std::string(label) +
                               ": non-finite or out-of-tolerance true residual");
    }
  }

  void prepareCloverSchur(
      Complex<Real> *compact_rhs, const Complex<Real> *full_rhs, int parity,
      const std::array<const Complex<Real> *, 2> &inverse) {
    const int other = 1 - parity;
    const int64_t vectors = geometry_.parity_vectors;
    const int64_t chunk = ompChunk<Real>(vectors);
    const Complex<Real> *rhs_other =
        full_rhs + static_cast<std::size_t>(other) * vectors;
    const Complex<Real> *rhs_target =
        full_rhs + static_cast<std::size_t>(parity) * vectors;
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < vectors; ++index) {
      work_a_[static_cast<std::size_t>(index)] = rhs_other[index];
    }
    applyOnsite(work_a_.data(), inverse[static_cast<std::size_t>(other)],
                work_b_.data());
    bareDslash(work_c_.data(), work_b_.data(), parity);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < vectors; ++index) {
      compact_rhs[index] =
          rhs_target[index] + kappa() * work_c_[static_cast<std::size_t>(index)];
    }
  }

  void reconstructCloverSchur(
      Complex<Real> *full_out, const Complex<Real> *full_rhs,
      const Complex<Real> *target_solution, int parity,
      const std::array<const Complex<Real> *, 2> &inverse) {
    const int other = 1 - parity;
    const int64_t vectors = geometry_.parity_vectors;
    const int64_t chunk = ompChunk<Real>(vectors);
    Complex<Real> *out_target =
        full_out + static_cast<std::size_t>(parity) * vectors;
    Complex<Real> *out_other =
        full_out + static_cast<std::size_t>(other) * vectors;
    const Complex<Real> *rhs_other =
        full_rhs + static_cast<std::size_t>(other) * vectors;
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < vectors; ++index) {
      out_target[index] = target_solution[index];
    }
    bareDslash(work_a_.data(), target_solution, other);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < vectors; ++index) {
      work_b_[static_cast<std::size_t>(index)] =
          rhs_other[index] + kappa() * work_a_[static_cast<std::size_t>(index)];
    }
    applyOnsite(work_b_.data(), inverse[static_cast<std::size_t>(other)],
                out_other);
  }

  void reconstructWilson(Complex<Real> *full_output,
                         const Complex<Real> *full_input, int parity) {
    const int other = 1 - parity;
    const int64_t vectors = geometry_.parity_vectors;
    const int64_t chunk = ompChunk<Real>(vectors);
    Complex<Real> *target =
        full_output + static_cast<std::size_t>(parity) * vectors;
    Complex<Real> *other_output =
        full_output + static_cast<std::size_t>(other) * vectors;
    const Complex<Real> *other_rhs =
        full_input + static_cast<std::size_t>(other) * vectors;
    bareDslash(work_a_.data(), target, other);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < vectors; ++index) {
      other_output[index] =
          other_rhs[index] + kappa() * work_a_[static_cast<std::size_t>(index)];
    }
  }

  Real checkWilsonFullResidual(Complex<Real> *full_output,
                               const Complex<Real> *full_input,
                               const char *label) {
    const int64_t full_vectors = geometry_.parity_vectors * 2;
    const int64_t chunk = ompChunk<Real>(full_vectors);
    applyFullWilsonOperator(full_output, tmp_b_.data(), tmp_a_.data());
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < full_vectors; ++index) {
      bp_[static_cast<std::size_t>(index)] =
          full_input[index] - tmp_b_[static_cast<std::size_t>(index)];
    }
    const Real rhs_norm = std::sqrt(globalNorm2(full_input, full_vectors));
    const Real residual = std::sqrt(globalNorm2(bp_.data(), full_vectors));
    const Real tolerance = std::max(atol(), residualFloor()) * rhs_norm;
    log(label, "full_true_residual=%g rhs_norm=%g relative=%g",
        static_cast<double>(residual), static_cast<double>(rhs_norm),
        static_cast<double>(residual / std::max(rhs_norm, Real(1))));
    if (residual > tolerance) {
      throw std::runtime_error(std::string(label) +
                               ": full true residual exceeds tolerance");
    }
    return residual;
  }

  void solveWilsonBistabCg(Complex<Real> *full_output,
                           const Complex<Real> *full_input) {
    const int parity = targetParity();
    const int other = 1 - parity;
    const int64_t vectors = geometry_.parity_vectors;
    const int64_t chunk = ompChunk<Real>(vectors);
    const Complex<Real> *rhs_target =
        full_input + static_cast<std::size_t>(parity) * vectors;
    const Complex<Real> *rhs_other =
        full_input + static_cast<std::size_t>(other) * vectors;
    Complex<Real> *target =
        full_output + static_cast<std::size_t>(parity) * vectors;
    if (parameters_.get(kUseInitialGuess) == 0) {
#pragma omp parallel for schedule(static, chunk)
      for (int64_t index = 0; index < vectors; ++index) {
        target[index] = Complex<Real>(0);
      }
    }

#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < vectors; ++index) {
      work_a_[static_cast<std::size_t>(index)] = rhs_other[index];
    }
    bareDslash(work_b_.data(), work_a_.data(), parity);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < vectors; ++index) {
      bp_[static_cast<std::size_t>(index)] =
          rhs_target[index] + kappa() * work_b_[static_cast<std::size_t>(index)];
    }
    solveCompactBiStabCg(
        target, bp_.data(), vectors,
        [this, parity](const Complex<Real> *input, Complex<Real> *output) {
          applyWilsonSchur(input, output, parity, work_b_.data(),
                           work_c_.data());
        },
        "WILSON_BISTABCG");
    reconstructWilson(full_output, full_input, parity);
    (void)checkWilsonFullResidual(full_output, full_input, "WILSON_BISTABCG");
  }

  void solveWilsonCg(Complex<Real> *full_output,
                     const Complex<Real> *full_input) {
    const int parity = targetParity();
    const int other = 1 - parity;
    const int64_t vectors = geometry_.parity_vectors;
    const int64_t chunk = ompChunk<Real>(vectors);
    const Complex<Real> *rhs_target =
        full_input + static_cast<std::size_t>(parity) * vectors;
    const Complex<Real> *rhs_other =
        full_input + static_cast<std::size_t>(other) * vectors;
    Complex<Real> *target =
        full_output + static_cast<std::size_t>(parity) * vectors;
    if (parameters_.get(kUseInitialGuess) == 0) {
#pragma omp parallel for schedule(static, chunk)
      for (int64_t index = 0; index < vectors; ++index) {
        target[index] = Complex<Real>(0);
      }
    }

#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < vectors; ++index) {
      work_a_[static_cast<std::size_t>(index)] = rhs_other[index];
    }
    bareDslash(work_b_.data(), work_a_.data(), parity);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < vectors; ++index) {
      bp_[static_cast<std::size_t>(index)] =
          rhs_target[index] + kappa() * work_b_[static_cast<std::size_t>(index)];
    }
    applyWilsonSchurDagger(bp_.data(), tmp_a_.data(), parity, work_b_.data(),
                           work_c_.data());
    solveCompactCg(
        target, tmp_a_.data(), vectors,
        [this, parity](const Complex<Real> *input, Complex<Real> *output) {
          applyWilsonCgDslash(input, output, parity);
        },
        "WILSON_CG_NORMAL", Real(0.01) * atol());
    reconstructWilson(full_output, full_input, parity);
    (void)checkWilsonFullResidual(full_output, full_input, "WILSON_CG");
  }

  void solveClover(Complex<Real> *full_output, const Complex<Real> *full_input,
                   int parity,
                   const std::array<const Complex<Real> *, 2> &clover,
                   const std::array<const Complex<Real> *, 2> &inverse) {
    using C = Complex<Real>;
    const int64_t vectors = geometry_.parity_vectors;
    const int64_t full_vectors = vectors * 2;
    const bool warm = parameters_.get(kUseInitialGuess) != 0;
    Complex<Real> *target =
        full_output + static_cast<std::size_t>(parity) * vectors;
    if (!warm) {
      const int64_t chunk = ompChunk<Real>(vectors);
#pragma omp parallel for schedule(static, chunk)
      for (int64_t index = 0; index < vectors; ++index) {
        target[index] = C(0);
      }
    }

    prepareCloverSchur(bp_.data(), full_input, parity, inverse);
    solveCompactBiStabCg(
        target, bp_.data(), vectors,
        [this, parity, &clover, &inverse](
            const Complex<Real> *input, Complex<Real> *output) {
          applySchur(input, output, parity, clover, inverse, tmp_a_.data(),
                     tmp_b_.data());
        },
        "CLOVER_BISTABCG_SCHUR", Real(0.1) * atol());
    reconstructCloverSchur(full_output, full_input, target, parity, inverse);

    applyFullOperator(full_output, tmp_b_.data(), clover, tmp_a_.data());
    const int64_t chunk = ompChunk<Real>(full_vectors);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < full_vectors; ++index) {
      bp_[static_cast<std::size_t>(index)] =
          full_input[index] - tmp_b_[static_cast<std::size_t>(index)];
    }
    const Real rhs_norm = std::sqrt(globalNorm2(full_input, full_vectors));
    const Real final_residual =
        std::sqrt(globalNorm2(bp_.data(), full_vectors));
    const Real tolerance =
        std::max(atol(), residualFloor()) * std::max(rhs_norm, Real(1));
    log("CLOVER_BISTABCG_SCHUR",
        "recomputed_residual=%g rhs_norm=%g tolerance=%g relative=%g",
        static_cast<double>(final_residual), static_cast<double>(rhs_norm),
        static_cast<double>(tolerance),
        static_cast<double>(final_residual /
                            std::max(rhs_norm, Real(1))));
    if (!std::isfinite(final_residual) || final_residual > tolerance) {
      throw std::runtime_error(
          "Clover BiStabCG: full true residual exceeds tolerance");
    }
  }
};

template <typename Real>
std::shared_ptr<BackendBase> createBackend(const int32_t *params,
                                           const void *argv) {
  return std::make_shared<Backend<Real>>(
      params, static_cast<const Real *>(argv));
}

} // namespace

void initializeMpi() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    int provided = 0;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
      throw std::runtime_error("MPI_THREAD_FUNNELED is unavailable");
    }
  }
  bindLogicalNuma();
  configuredThreads();
}

std::shared_ptr<BackendBase> makeBackend(const int32_t *params,
                                         const void *argv) {
  if (params == nullptr || argv == nullptr) {
    throw std::runtime_error("null params or argv");
  }
  const int type = params[kDataType];
  if (type == kLatC64) {
    return createBackend<float>(params, argv);
  }
  if (type == kLatC128) {
    return createBackend<double>(params, argv);
  }
  throw std::runtime_error("unsupported data type");
}

std::shared_ptr<BackendBase> registerBackend(const void *set_ptrs,
                                             const int32_t *params,
                                             const void *argv) {
  auto backend = makeBackend(params, argv);
  std::lock_guard<std::mutex> lock(registryMutex());
  registry()[set_ptrs] = backend;
  return backend;
}

std::shared_ptr<BackendBase> findBackend(const void *set_ptrs) {
  std::lock_guard<std::mutex> lock(registryMutex());
  const auto iterator = registry().find(set_ptrs);
  if (iterator == registry().end()) {
    throw std::runtime_error("applyInitQcu has not initialized this set_ptrs");
  }
  return iterator->second;
}

void eraseBackend(const void *set_ptrs) {
  std::lock_guard<std::mutex> lock(registryMutex());
  registry().erase(set_ptrs);
}

[[noreturn]] void abortWithMessage(const char *submodule,
                                   const char *message) {
  std::fprintf(stderr, "PYQCU::KUPL::%s:\n %s\n", submodule, message);
  std::fflush(stderr);
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
  }
  std::abort();
}

} // namespace kupl::qcu
