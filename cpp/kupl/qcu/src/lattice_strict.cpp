#include "kupl/lattice_set.h"
#include "kupl/lattice_strict.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace kupl::qcu {
namespace {

template <typename Real>
using CloverArrays =
    std::array<std::array<const Complex<Real> *, 2>, 2>;

template <typename Real>
CloverArrays<Real> clover_arrays(const void *even, const void *odd,
                                 const void *even_inv,
                                 const void *odd_inv) {
  return {{
      {static_cast<const Complex<Real> *>(even),
       static_cast<const Complex<Real> *>(odd)},
      {static_cast<const Complex<Real> *>(even_inv),
       static_cast<const Complex<Real> *>(odd_inv)},
  }};
}

} // namespace

template <typename Real>
int LatticeSet<Real>::strict_fine_matpc(
    void *out, const void *in, const void *gauge, const void *clover_even,
    const void *clover_odd, const void *clover_even_inv,
    const void *clover_odd_inv, int parity) {
  if (out == nullptr || in == nullptr || gauge == nullptr ||
      clover_even == nullptr || clover_odd == nullptr ||
      clover_even_inv == nullptr || clover_odd_inv == nullptr) {
    throw std::runtime_error("strict fine MATPC received a null pointer");
  }
  if (parity < 0 || parity > 1) {
    throw std::runtime_error("strict fine MATPC parity must be 0 or 1");
  }
  require_wilson_mode();
  ensure_gauge_halo(static_cast<const Complex<Real> *>(gauge));
  const auto clover = clover_arrays<Real>(
      clover_even, clover_odd, clover_even_inv, clover_odd_inv);
  auto *output = static_cast<Complex<Real> *>(out);
  const auto *source = static_cast<const Complex<Real> *>(in);
  if (output == source) {
    std::copy(source, source + geometry_.parity_vectors, bp_.begin());
    source = bp_.data();
  }
  const int other = 1 - parity;
  bare_dslash(work_a_.data(), source, other);
  apply_onsite(work_a_.data(), clover[1][static_cast<std::size_t>(other)],
               work_b_.data());
  bare_dslash(work_a_.data(), work_b_.data(), parity);
  apply_onsite(work_a_.data(), clover[1][static_cast<std::size_t>(parity)],
               work_a_.data());
  const Real coefficient = kappa() * kappa();
  const int64_t count = geometry_.parity_vectors;
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < count; ++index) {
    output[index] = source[index] - coefficient * work_a_[index];
  }
  return 0;
}

template <typename Real>
int LatticeSet<Real>::strict_fine_prepare(
    void *compact_rhs, const void *full_rhs, const void *gauge,
    const void *clover_even, const void *clover_odd,
    const void *clover_even_inv, const void *clover_odd_inv, int parity) {
  if (compact_rhs == nullptr || full_rhs == nullptr || gauge == nullptr ||
      clover_even == nullptr || clover_odd == nullptr ||
      clover_even_inv == nullptr || clover_odd_inv == nullptr) {
    throw std::runtime_error("strict fine prepare received a null pointer");
  }
  if (parity < 0 || parity > 1) {
    throw std::runtime_error("strict fine prepare parity must be 0 or 1");
  }
  require_wilson_mode();
  ensure_gauge_halo(static_cast<const Complex<Real> *>(gauge));
  const auto clover = clover_arrays<Real>(
      clover_even, clover_odd, clover_even_inv, clover_odd_inv);
  const int other = 1 - parity;
  const int64_t vectors = geometry_.parity_vectors;
  const auto *rhs = static_cast<const Complex<Real> *>(full_rhs);
  const Complex<Real> *rhs_target =
      rhs + static_cast<std::size_t>(parity) * vectors;
  const Complex<Real> *rhs_other =
      rhs + static_cast<std::size_t>(other) * vectors;
  auto *output = static_cast<Complex<Real> *>(compact_rhs);

  apply_onsite(rhs_other, clover[1][static_cast<std::size_t>(other)],
               tmp_a_.data());
  bare_dslash(work_b_.data(), tmp_a_.data(), parity);
  const int64_t chunk = omp_chunk<Real>(vectors);
  const Real k = kappa();
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < vectors; ++index) {
    output[index] = rhs_target[index] + k * work_b_[index];
  }
  apply_onsite(output, clover[1][static_cast<std::size_t>(parity)], output);
  return 0;
}

template <typename Real>
int LatticeSet<Real>::strict_fine_reconstruct(
    void *full_out, const void *full_rhs, const void *target_solution,
    const void *gauge, const void *clover_even, const void *clover_odd,
    const void *clover_even_inv, const void *clover_odd_inv, int parity) {
  if (full_out == nullptr || full_rhs == nullptr ||
      target_solution == nullptr || gauge == nullptr ||
      clover_even == nullptr || clover_odd == nullptr ||
      clover_even_inv == nullptr || clover_odd_inv == nullptr) {
    throw std::runtime_error("strict fine reconstruct received a null pointer");
  }
  if (parity < 0 || parity > 1) {
    throw std::runtime_error("strict fine reconstruct parity must be 0 or 1");
  }
  require_wilson_mode();
  ensure_gauge_halo(static_cast<const Complex<Real> *>(gauge));
  const auto clover = clover_arrays<Real>(
      clover_even, clover_odd, clover_even_inv, clover_odd_inv);
  const int other = 1 - parity;
  const int64_t vectors = geometry_.parity_vectors;
  auto *output = static_cast<Complex<Real> *>(full_out);
  const auto *rhs = static_cast<const Complex<Real> *>(full_rhs);
  const auto *target =
      static_cast<const Complex<Real> *>(target_solution);
  Complex<Real> *out_target =
      output + static_cast<std::size_t>(parity) * vectors;
  Complex<Real> *out_other =
      output + static_cast<std::size_t>(other) * vectors;
  const Complex<Real> *rhs_other =
      rhs + static_cast<std::size_t>(other) * vectors;

  std::copy(target, target + vectors, out_target);
  bare_dslash(tmp_a_.data(), target, other);
  const int64_t chunk = omp_chunk<Real>(vectors);
  const Real k = kappa();
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < vectors; ++index) {
    work_a_[static_cast<std::size_t>(index)] =
        rhs_other[index] + k * tmp_a_[index];
  }
  apply_onsite(work_a_.data(), clover[1][static_cast<std::size_t>(other)],
               out_other);
  return 0;
}

namespace {

struct StrictGeometry {
  int E = 0;
  int X = 0;
  int Y = 0;
  int Z = 0;
  int T = 0;
  int64_t volume = 0;
  int64_t half_volume = 0;
};

struct StrictTopology {
  std::array<int, kDim> grid{};
  std::array<int, kDim> coord{};
  int rank = 0;
  int size = 1;
  std::array<int, kDim> neighbor_minus{};
  std::array<int, kDim> neighbor_plus{};
};

inline void strict_validate_parity_geometry(int X, int Y, int Z, int T) {
  if (X <= 0 || Y <= 0 || Z <= 0 || T <= 0 ||
      (X & 1) != 0 || (Y & 1) != 0 || (Z & 1) != 0 ||
      (T & 1) != 0) {
    throw std::runtime_error(
        "strict geometry requires positive even X/Y/Z/T");
  }
}

template <typename Real>
StrictGeometry strict_geometry(int E, int X, int Y, int Z, int T,
                               bool require_even = false) {
  if (E <= 0 || X <= 0 || Y <= 0 || Z <= 0 || T <= 0) {
    throw std::runtime_error(
        "strict geometry dimensions must be positive");
  }
  if (require_even) {
    strict_validate_parity_geometry(X, Y, Z, T);
  }
  const uint64_t volume =
      static_cast<uint64_t>(X) * Y * Z * T;
  if (volume > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      static_cast<uint64_t>(E) >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) /
              std::max<uint64_t>(volume, 1)) {
    throw std::runtime_error("strict geometry element count overflow");
  }
  StrictGeometry result;
  result.E = E;
  result.X = X;
  result.Y = Y;
  result.Z = Z;
  result.T = T;
  result.volume = static_cast<int64_t>(volume);
  result.half_volume =
      require_even ? result.volume / 2 : result.volume;
  (void)sizeof(Real);
  return result;
}

inline StrictTopology strict_topology(const int32_t *params) {
  if (params == nullptr) {
    throw std::runtime_error("strict topology params pointer is null");
  }
  StrictTopology topology;
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    throw std::runtime_error("strict topology requires MPI initialization");
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &topology.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &topology.size);
  int product = 1;
  for (int dim = 0; dim < kDim; ++dim) {
    topology.grid[dim] = params[kGridX + dim];
    if (topology.grid[dim] <= 0) {
      throw std::runtime_error("strict process grid must be positive");
    }
    product *= topology.grid[dim];
  }
  if (product != topology.size) {
    throw std::runtime_error(
        "strict process grid product does not match MPI_COMM_WORLD");
  }
  int periods[kDim] = {1, 1, 1, 1};
  MPI_Comm comm = MPI_COMM_NULL;
  if (MPI_Cart_create(MPI_COMM_WORLD, kDim, topology.grid.data(), periods, 0,
                      &comm) != MPI_SUCCESS) {
    throw std::runtime_error("strict MPI_Cart_create failed");
  }
  MPI_Cart_coords(comm, topology.rank, kDim, topology.coord.data());
  for (int dim = 0; dim < kDim; ++dim) {
    MPI_Cart_shift(comm, dim, 1, &topology.neighbor_minus[dim],
                   &topology.neighbor_plus[dim]);
  }
  MPI_Comm_free(&comm);
  return topology;
}

inline int64_t strict_full_index(const StrictGeometry &geometry, int x, int y,
                                 int z, int t) {
  return ((static_cast<int64_t>(x) * geometry.Y + y) * geometry.Z + z) *
             geometry.T +
         t;
}

inline int64_t strict_half_index(const StrictGeometry &geometry, int x, int y,
                                 int z, int t) {
  return ((static_cast<int64_t>(x) * geometry.Y + y) * geometry.Z + z) *
             (geometry.T / 2) +
         t / 2;
}

inline void strict_decode_half(const StrictGeometry &geometry, int64_t index,
                               int parity, int &x, int &y, int &z, int &t) {
  const int half_t = geometry.T / 2;
  int xyz = static_cast<int>(index / half_t);
  const int t2 = static_cast<int>(index - static_cast<int64_t>(xyz) * half_t);
  z = xyz % geometry.Z;
  xyz /= geometry.Z;
  y = xyz % geometry.Y;
  x = xyz / geometry.Y;
  t = 2 * t2 + (((x + y + z) & 1) ^ parity);
}

inline int strict_local_extent(const StrictGeometry &geometry, int dim) {
  return dim == 0 ? geometry.X
                  : dim == 1 ? geometry.Y
                             : dim == 2 ? geometry.Z : geometry.T;
}

inline int64_t strict_face_sites(const StrictGeometry &geometry, int dim) {
  return geometry.volume / strict_local_extent(geometry, dim);
}

inline void strict_decode_face(const StrictGeometry &geometry, int dim,
                               int64_t face, int boundary,
                               std::array<int, kDim> &coord) {
  const int extent[kDim] = {geometry.X, geometry.Y, geometry.Z, geometry.T};
  for (int axis = kDim - 1; axis >= 0; --axis) {
    if (axis == dim) {
      coord[axis] = boundary;
    } else {
      coord[axis] = static_cast<int>(face % extent[axis]);
      face /= extent[axis];
    }
  }
}

inline int64_t strict_face_index(const StrictGeometry &geometry, int dim,
                                 const std::array<int, kDim> &coord) {
  const int extent[kDim] = {geometry.X, geometry.Y, geometry.Z, geometry.T};
  int64_t index = 0;
  int64_t stride = 1;
  for (int axis = kDim - 1; axis >= 0; --axis) {
    if (axis == dim) {
      continue;
    }
    index += static_cast<int64_t>(coord[axis]) * stride;
    stride *= extent[axis];
  }
  return index;
}

template <typename Real>
std::vector<Complex<Real>> strict_pack_full_face(
    const Complex<Real> *field, const StrictGeometry &geometry, int components,
    int dim, int boundary) {
  const int64_t face_sites = strict_face_sites(geometry, dim);
  std::vector<Complex<Real>> packed(
      static_cast<std::size_t>(components * face_sites));
  const int64_t count = components * face_sites;
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < count; ++index) {
    const int component = static_cast<int>(index / face_sites);
    const int64_t face = index % face_sites;
    std::array<int, kDim> coord{};
    strict_decode_face(geometry, dim, face, boundary, coord);
    const int64_t site = strict_full_index(
        geometry, coord[0], coord[1], coord[2], coord[3]);
    packed[static_cast<std::size_t>(index)] =
        field[static_cast<int64_t>(component) * geometry.volume + site];
  }
  return packed;
}

template <typename Real>
std::vector<Complex<Real>> strict_pack_compact_face(
    const Complex<Real> *field, const StrictGeometry &geometry, int components,
    int parity, int dim, int boundary) {
  const int64_t face_sites = strict_face_sites(geometry, dim);
  std::vector<Complex<Real>> packed(
      static_cast<std::size_t>(components * face_sites));
  const int64_t count = components * geometry.half_volume;
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < count; ++index) {
    const int component = static_cast<int>(index / geometry.half_volume);
    const int64_t half_site = index % geometry.half_volume;
    int x = 0;
    int y = 0;
    int z = 0;
    int t = 0;
    strict_decode_half(geometry, half_site, parity, x, y, z, t);
    const int coordinate[kDim] = {x, y, z, t};
    if (coordinate[dim] != boundary) {
      continue;
    }
    std::array<int, kDim> full_coord = {x, y, z, t};
    const int64_t face = strict_face_index(geometry, dim, full_coord);
    packed[static_cast<std::size_t>(
        static_cast<int64_t>(component) * face_sites + face)] =
        field[static_cast<int64_t>(component) * geometry.half_volume +
              half_site];
  }
  return packed;
}

template <typename Real>
std::vector<Complex<Real>> strict_pack_backward_links(
    const Complex<Real> *links, const StrictGeometry &geometry, int dim) {
  const int64_t face_sites = strict_face_sites(geometry, dim);
  const int64_t matrices = static_cast<int64_t>(geometry.E) * geometry.E;
  std::vector<Complex<Real>> packed(
      static_cast<std::size_t>(matrices * face_sites));
  const int64_t count = matrices * face_sites;
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < count; ++index) {
    const int column = static_cast<int>(index / (geometry.E * face_sites));
    const int row =
        static_cast<int>((index / face_sites) % geometry.E);
    const int64_t face = index % face_sites;
    std::array<int, kDim> coord{};
    strict_decode_face(geometry, dim, face,
                       strict_local_extent(geometry, dim) - 1, coord);
    const int64_t site = strict_full_index(
        geometry, coord[0], coord[1], coord[2], coord[3]);
    const int64_t link =
        (((static_cast<int64_t>(1 * kDim + dim) * geometry.E + column) *
              geometry.E +
          row) *
             geometry.volume +
         site);
    packed[static_cast<std::size_t>(index)] = links[link];
  }
  return packed;
}

template <typename Real>
void strict_exchange_vector(const Complex<Real> *field,
                            const StrictGeometry &geometry, int components,
                            const StrictTopology &topology, int dim,
                            bool compact, int parity,
                            std::vector<Complex<Real>> &recv_lower,
                            std::vector<Complex<Real>> &recv_upper,
                            int tag_base) {
  if (topology.grid[dim] <= 1) {
    return;
  }
  const int last = strict_local_extent(geometry, dim) - 1;
  auto send_lower =
      compact ? strict_pack_compact_face(field, geometry, components, parity,
                                         dim, 0)
              : strict_pack_full_face(field, geometry, components, dim, 0);
  auto send_upper =
      compact ? strict_pack_compact_face(field, geometry, components, parity,
                                         dim, last)
              : strict_pack_full_face(field, geometry, components, dim, last);
  recv_lower.assign(send_lower.size(), Complex<Real>(0));
  recv_upper.assign(send_upper.size(), Complex<Real>(0));
  if (send_lower.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()) /
          sizeof(Complex<Real>)) {
    throw std::runtime_error("strict vector halo message is too large");
  }
  const int count =
      static_cast<int>(send_lower.size() * sizeof(Complex<Real>));
  MPI_Sendrecv(send_upper.data(), count, MPI_BYTE,
               topology.neighbor_plus[dim], tag_base + dim * 4,
               recv_lower.data(), count, MPI_BYTE,
               topology.neighbor_minus[dim], tag_base + dim * 4,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Sendrecv(send_lower.data(), count, MPI_BYTE,
               topology.neighbor_minus[dim], tag_base + dim * 4 + 1,
               recv_upper.data(), count, MPI_BYTE,
               topology.neighbor_plus[dim], tag_base + dim * 4 + 1,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

template <typename Real>
std::vector<Complex<Real>> strict_exchange_backward_links(
    const Complex<Real> *links, const StrictGeometry &geometry,
    const StrictTopology &topology, int dim, int tag_base) {
  if (topology.grid[dim] <= 1) {
    return {};
  }
  auto send = strict_pack_backward_links(links, geometry, dim);
  std::vector<Complex<Real>> receive(send.size(), Complex<Real>(0));
  if (send.size() >
      static_cast<std::size_t>(std::numeric_limits<int>::max()) /
          sizeof(Complex<Real>)) {
    throw std::runtime_error("strict link halo message is too large");
  }
  const int count =
      static_cast<int>(send.size() * sizeof(Complex<Real>));
  MPI_Sendrecv(send.data(), count, MPI_BYTE,
               topology.neighbor_plus[dim], tag_base + dim * 4 + 2,
               receive.data(), count, MPI_BYTE,
               topology.neighbor_minus[dim], tag_base + dim * 4 + 2,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  return receive;
}

template <typename Real>
void strict_hopping_full(
    Complex<Real> *out, const Complex<Real> *in, const Complex<Real> *links,
    const Complex<Real> *base, const StrictGeometry &geometry,
    const StrictTopology &topology, int tag_base) {
  if (out == nullptr || in == nullptr || links == nullptr) {
    throw std::runtime_error("strict full hopping received a null pointer");
  }
  const int64_t output_count = static_cast<int64_t>(geometry.E) *
                               geometry.volume;
  const int64_t chunk = omp_chunk<Real>(output_count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < output_count; ++index) {
    out[index] =
        base == nullptr ? Complex<Real>(0) : base[index];
  }

  for (int dim = 0; dim < kDim; ++dim) {
    std::vector<Complex<Real>> recv_lower;
    std::vector<Complex<Real>> recv_upper;
    strict_exchange_vector(in, geometry, geometry.E, topology, dim, false, 0,
                           recv_lower, recv_upper, tag_base);
    auto link_ghost = strict_exchange_backward_links(
        links, geometry, topology, dim, tag_base + 100);
    const int64_t face_sites = strict_face_sites(geometry, dim);
    const int extent = strict_local_extent(geometry, dim);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < output_count; ++index) {
      const int row = static_cast<int>(index / geometry.volume);
      const int64_t site = index % geometry.volume;
      int xyz = static_cast<int>(site / geometry.T);
      const int t = static_cast<int>(site - static_cast<int64_t>(xyz) *
                                                  geometry.T);
      int z = xyz % geometry.Z;
      xyz /= geometry.Z;
      int y = xyz % geometry.Y;
      const int x = xyz / geometry.Y;
      const int coord[kDim] = {x, y, z, t};
      int forward[kDim] = {x, y, z, t};
      int backward_coord[kDim] = {x, y, z, t};
      const bool forward_remote =
          topology.grid[dim] > 1 && coord[dim] == extent - 1;
      const bool backward_remote =
          topology.grid[dim] > 1 && coord[dim] == 0;
      forward[dim] = (coord[dim] + 1) % extent;
      backward_coord[dim] = (coord[dim] + extent - 1) % extent;
      const int64_t forward_site = strict_full_index(
          geometry, forward[0], forward[1], forward[2], forward[3]);
      const int64_t backward_site = strict_full_index(
          geometry, backward_coord[0], backward_coord[1], backward_coord[2],
          backward_coord[3]);
      std::array<int, kDim> target_coord = {x, y, z, t};
      const int64_t face =
          strict_face_index(geometry, dim, target_coord);
      Complex<Real> sum(0);
      for (int column = 0; column < geometry.E; ++column) {
        const int64_t forward_link =
            (((static_cast<int64_t>(0 * kDim + dim) * geometry.E + row) *
                  geometry.E +
              column) *
                 geometry.volume +
             site);
        const int64_t backward_link =
            (((static_cast<int64_t>(1 * kDim + dim) * geometry.E + column) *
                  geometry.E +
              row) *
                 geometry.volume +
             backward_site);
        const Complex<Real> forward_value =
            forward_remote
                ? recv_upper[static_cast<std::size_t>(
                      static_cast<int64_t>(column) * face_sites + face)]
                : in[static_cast<int64_t>(column) * geometry.volume +
                     forward_site];
        const Complex<Real> backward_value =
            backward_remote
                ? recv_lower[static_cast<std::size_t>(
                      static_cast<int64_t>(column) * face_sites + face)]
                : in[static_cast<int64_t>(column) * geometry.volume +
                     backward_site];
        const Complex<Real> backward_link_value =
            backward_remote
                ? link_ghost[static_cast<std::size_t>(
                      (static_cast<int64_t>(column) * geometry.E + row) *
                          face_sites +
                      face)]
                : links[backward_link];
        sum += links[forward_link] * forward_value;
        sum += std::conj(backward_link_value) * backward_value;
      }
      if (base == nullptr) {
        out[index] += sum;
      } else {
        out[index] -= sum;
      }
    }
  }
}

template <typename Real>
void strict_hopping_compact(
    Complex<Real> *out, const Complex<Real> *in, const Complex<Real> *links,
    const Complex<Real> *base, const StrictGeometry &geometry,
    const StrictTopology &topology, int parity, int tag_base) {
  if (out == nullptr || in == nullptr || links == nullptr) {
    throw std::runtime_error("strict compact hopping received a null pointer");
  }
  const int source_parity = 1 - parity;
  const int64_t output_count =
      static_cast<int64_t>(geometry.E) * geometry.half_volume;
  const int64_t chunk = omp_chunk<Real>(output_count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < output_count; ++index) {
    out[index] =
        base == nullptr ? Complex<Real>(0) : base[index];
  }

  for (int dim = 0; dim < kDim; ++dim) {
    std::vector<Complex<Real>> recv_lower;
    std::vector<Complex<Real>> recv_upper;
    strict_exchange_vector(in, geometry, geometry.E, topology, dim, true,
                           source_parity, recv_lower, recv_upper,
                           tag_base);
    auto link_ghost = strict_exchange_backward_links(
        links, geometry, topology, dim, tag_base + 100);
    const int64_t face_sites = strict_face_sites(geometry, dim);
    const int extent = strict_local_extent(geometry, dim);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < output_count; ++index) {
      const int row = static_cast<int>(index / geometry.half_volume);
      const int64_t half_site = index % geometry.half_volume;
      int x = 0;
      int y = 0;
      int z = 0;
      int t = 0;
      strict_decode_half(geometry, half_site, parity, x, y, z, t);
      const int coord[kDim] = {x, y, z, t};
      int forward[kDim] = {x, y, z, t};
      int backward_coord[kDim] = {x, y, z, t};
      const bool forward_remote =
          topology.grid[dim] > 1 && coord[dim] == extent - 1;
      const bool backward_remote =
          topology.grid[dim] > 1 && coord[dim] == 0;
      forward[dim] = (coord[dim] + 1) % extent;
      backward_coord[dim] = (coord[dim] + extent - 1) % extent;
      const int64_t forward_half = strict_half_index(
          geometry, forward[0], forward[1], forward[2], forward[3]);
      const int64_t backward_half = strict_half_index(
          geometry, backward_coord[0], backward_coord[1],
          backward_coord[2], backward_coord[3]);
      const int64_t site = strict_full_index(geometry, x, y, z, t);
      const int64_t backward_site = strict_full_index(
          geometry, backward_coord[0], backward_coord[1],
          backward_coord[2], backward_coord[3]);
      std::array<int, kDim> target_coord = {x, y, z, t};
      const int64_t face =
          strict_face_index(geometry, dim, target_coord);
      Complex<Real> sum(0);
      for (int column = 0; column < geometry.E; ++column) {
        const int64_t forward_link =
            (((static_cast<int64_t>(0 * kDim + dim) * geometry.E + row) *
                  geometry.E +
              column) *
                 geometry.volume +
             site);
        const int64_t backward_link =
            (((static_cast<int64_t>(1 * kDim + dim) * geometry.E + column) *
                  geometry.E +
              row) *
                 geometry.volume +
             backward_site);
        const Complex<Real> forward_value =
            forward_remote
                ? recv_upper[static_cast<std::size_t>(
                      static_cast<int64_t>(column) * face_sites + face)]
                : in[static_cast<int64_t>(column) * geometry.half_volume +
                     forward_half];
        const Complex<Real> backward_value =
            backward_remote
                ? recv_lower[static_cast<std::size_t>(
                      static_cast<int64_t>(column) * face_sites + face)]
                : in[static_cast<int64_t>(column) * geometry.half_volume +
                     backward_half];
        const Complex<Real> backward_link_value =
            backward_remote
                ? link_ghost[static_cast<std::size_t>(
                      (static_cast<int64_t>(column) * geometry.E + row) *
                          face_sites +
                      face)]
                : links[backward_link];
        sum += links[forward_link] * forward_value;
        sum += std::conj(backward_link_value) * backward_value;
      }
      if (base == nullptr) {
        out[index] += sum;
      } else {
        out[index] -= sum;
      }
    }
  }
}

template <typename Real>
void strict_onsite_full_to_parity(Complex<Real> *out,
                                  const Complex<Real> *full_rhs,
                                  const Complex<Real> *onsite_pair,
                                  const StrictGeometry &geometry, int parity,
                                  int onsite_index) {
  const int64_t count = static_cast<int64_t>(geometry.E) *
                        geometry.half_volume;
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < count; ++index) {
    const int row = static_cast<int>(index / geometry.half_volume);
    const int64_t half_site = index % geometry.half_volume;
    int x = 0;
    int y = 0;
    int z = 0;
    int t = 0;
    strict_decode_half(geometry, half_site, parity, x, y, z, t);
    const int64_t site = strict_full_index(geometry, x, y, z, t);
    Complex<Real> sum(0);
    for (int column = 0; column < geometry.E; ++column) {
      const int64_t matrix =
          (((static_cast<int64_t>(onsite_index) * geometry.E + row) *
                geometry.E +
            column) *
               geometry.volume +
           site);
      sum += onsite_pair[matrix] *
             full_rhs[static_cast<int64_t>(column) * geometry.volume +
                      site];
    }
    out[index] = sum;
  }
}

template <typename Real>
void strict_join_parities(Complex<Real> *full_out,
                          const Complex<Real> *target_solution,
                          const Complex<Real> *other_solution,
                          const StrictGeometry &geometry, int parity) {
  const int64_t count =
      static_cast<int64_t>(geometry.E) * geometry.volume;
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < count; ++index) {
    const int component = static_cast<int>(index / geometry.volume);
    const int64_t site = index % geometry.volume;
    int xyz = static_cast<int>(site / geometry.T);
    const int t = static_cast<int>(site - static_cast<int64_t>(xyz) *
                                                geometry.T);
    int z = xyz % geometry.Z;
    xyz /= geometry.Z;
    int y = xyz % geometry.Y;
    const int x = xyz / geometry.Y;
    const int site_parity = (x + y + z + t) & 1;
    const int64_t half_site = strict_half_index(geometry, x, y, z, t);
    const int64_t compact_index =
        static_cast<int64_t>(component) * geometry.half_volume + half_site;
    full_out[index] =
        site_parity == parity ? target_solution[compact_index]
                              : other_solution[compact_index];
  }
}

inline std::size_t strict_null_index(
    int E_idx, int e_idx, int x, int y, int z, int t, int e, int Xf, int Yf,
    int Zf, int Tf, int Xc, int Yc, int Zc, int Tc) {
  const int bx = Xf / Xc;
  const int by = Yf / Yc;
  const int bz = Zf / Zc;
  const int bt = Tf / Tc;
  const int xc = x / bx;
  const int yc = y / by;
  const int zc = z / bz;
  const int tc = t / bt;
  std::size_t index = static_cast<std::size_t>(E_idx);
  index = index * static_cast<std::size_t>(e) +
          static_cast<std::size_t>(e_idx);
  index = index * static_cast<std::size_t>(Xc) +
          static_cast<std::size_t>(xc);
  index = index * static_cast<std::size_t>(bx) +
          static_cast<std::size_t>(x - xc * bx);
  index = index * static_cast<std::size_t>(Yc) +
          static_cast<std::size_t>(yc);
  index = index * static_cast<std::size_t>(by) +
          static_cast<std::size_t>(y - yc * by);
  index = index * static_cast<std::size_t>(Zc) +
          static_cast<std::size_t>(zc);
  index = index * static_cast<std::size_t>(bz) +
          static_cast<std::size_t>(z - zc * bz);
  index = index * static_cast<std::size_t>(Tc) +
          static_cast<std::size_t>(tc);
  index = index * static_cast<std::size_t>(bt) +
          static_cast<std::size_t>(t - tc * bt);
  return index;
}

template <typename Real>
Complex<Real> strict_global_dot(const Complex<Real> *left,
                                const Complex<Real> *right, int64_t count) {
  Real real_sum = 0;
  Real imag_sum = 0;
  const int64_t chunk = omp_chunk<Real>(count);
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
  MPI_Allreduce(local, global, 2, type, MPI_SUM, MPI_COMM_WORLD);
  return Complex<Real>(global[0], global[1]);
}

template <typename Real>
Complex<Real> strict_global_dot(const std::vector<Complex<Real>> &left,
                                const std::vector<Complex<Real>> &right) {
  if (left.size() != right.size()) {
    throw std::runtime_error("strict dot vector size mismatch");
  }
  return strict_global_dot(left.data(), right.data(),
                           static_cast<int64_t>(left.size()));
}

template <typename Real>
Real strict_global_norm2(const Complex<Real> *field, int64_t count) {
  Real local = 0;
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk) reduction(+ : local)
  for (int64_t index = 0; index < count; ++index) {
    local += std::norm(field[index]);
  }
  Real global = 0;
  const MPI_Datatype type =
      sizeof(Real) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE;
  MPI_Allreduce(&local, &global, 1, type, MPI_SUM, MPI_COMM_WORLD);
  return global;
}

template <typename Real>
Real strict_global_norm2(const std::vector<Complex<Real>> &field) {
  return strict_global_norm2(field.data(),
                             static_cast<int64_t>(field.size()));
}

template <typename Real> struct StrictHostLevel {
  StrictGeometry geometry;
  const Complex<Real> *links = nullptr;
  const Complex<Real> *onsite_pair = nullptr;
  const Complex<Real> *null_vectors = nullptr;
  int max_iter = 100;
  Real tolerance = Real(0);
  std::vector<Complex<Real>> full_rhs;
  std::vector<Complex<Real>> pc_rhs;
  std::vector<Complex<Real>> x;
  std::vector<Complex<Real>> r;
  std::vector<Complex<Real>> v;
  std::vector<Complex<Real>> tmp;
  std::vector<Complex<Real>> correction_full;
};

template <typename Real> class StrictHostHierarchy {
public:
  static constexpr uint64_t kMagic = UINT64_C(0x4b55504c53545243);
  static constexpr int kMaxLevels = 5;

  StrictHostHierarchy(void *set_ptrs, const int32_t *params, int start_level)
      : start_level_(start_level),
        num_levels_(0),
        parity_(0),
        smoother_steps_(2) {
    if (set_ptrs == nullptr || params == nullptr) {
      throw std::runtime_error("strict hierarchy requires set_ptrs/params");
    }
    params_ = params;
    num_levels_ = params[kMgNumLevel];
    parity_ = params[kParity];
    smoother_steps_ = params[kMgMuPre] > 0 ? params[kMgMuPre] : 2;
    header_.magic = kMagic;
    header_.data_type = params[kDataType];
    header_.start_level = start_level;
    header_.num_levels = num_levels_;
    if (num_levels_ < 2 || num_levels_ > kMaxLevels ||
        start_level_ < 1 || start_level_ >= num_levels_) {
      throw std::runtime_error("strict hierarchy requires 2..5 valid levels");
    }
    if (parity_ < 0 || parity_ > 1) {
      throw std::runtime_error("strict hierarchy parity must be 0 or 1");
    }
    topology_ = strict_topology(params);
    table_ = static_cast<long long *>(set_ptrs);
    const int set_index = params[kSetIndex];
    if (set_index < 0 || set_index >= kSetPtrsSize) {
      throw std::runtime_error("strict hierarchy set index is out of range");
    }
    const auto owner = find_lattice_set(table_ + set_index);
    for (int level = start_level_; level < num_levels_; ++level) {
      const int param_base =
          kMgLevel1E + (level - 1) * kMgParamsSize;
      const int asset_base =
          kSetPtrsStrictCoarseBase +
          (level - 1) * kSetPtrsStrictStride;
      StrictHostLevel<Real> &state = levels_[level];
      state.geometry = strict_geometry<Real>(
          params[param_base], params[param_base + 1],
          params[param_base + 2], params[param_base + 3],
          params[param_base + 4], true);
      if (params[param_base + 6] != params[kDataType]) {
        throw std::runtime_error(
            "strict hierarchy mixed precision is not supported");
      }
      state.links = reinterpret_cast<const Complex<Real> *>(
          table_[asset_base + kSetPtrsStrictPreconditionedLinks]);
      state.onsite_pair = reinterpret_cast<const Complex<Real> *>(
          table_[asset_base + kSetPtrsStrictOnsitePair]);
      if (state.links == nullptr || state.onsite_pair == nullptr) {
        throw std::runtime_error("strict hierarchy coarse assets are missing");
      }
      if (level < num_levels_ - 1) {
        state.null_vectors = reinterpret_cast<const Complex<Real> *>(
            table_[asset_base + kSetPtrsStrictNull]);
        if (state.null_vectors == nullptr) {
          throw std::runtime_error(
              "strict hierarchy transition null vectors are missing");
        }
      }
      state.max_iter = params[param_base + 5] > 0
                           ? params[param_base + 5]
                           : 100;
      if (level == num_levels_ - 1) {
        const double level_tolerance =
            owner->real_parameter(kMgLevel1Atol + level - 1);
        if (!std::isfinite(level_tolerance) || level_tolerance <= 0) {
          throw std::runtime_error(
              "strict coarsest tolerance must be finite and positive");
        }
        state.tolerance = static_cast<Real>(level_tolerance);
      }
      const std::size_t compact = static_cast<std::size_t>(
          state.geometry.E * state.geometry.half_volume);
      const std::size_t full = static_cast<std::size_t>(
          state.geometry.E * state.geometry.volume);
      state.pc_rhs.resize(compact);
      state.x.resize(compact);
      state.r.resize(compact);
      state.v.resize(compact);
      state.tmp.resize(compact);
      if (level > start_level_) {
        state.full_rhs.resize(full);
      }
      if (level < num_levels_ - 1) {
        const StrictGeometry child_geometry = strict_geometry<Real>(
            params[kMgLevel1E + level * kMgParamsSize],
            params[kMgLevel1E + level * kMgParamsSize + 1],
            params[kMgLevel1E + level * kMgParamsSize + 2],
            params[kMgLevel1E + level * kMgParamsSize + 3],
            params[kMgLevel1E + level * kMgParamsSize + 4], true);
        state.correction_full.resize(static_cast<std::size_t>(
            child_geometry.E * child_geometry.volume));
      }
      bytes_ += sizeof(Complex<Real>) *
                (state.pc_rhs.size() + state.x.size() + state.r.size() +
                 state.v.size() + state.tmp.size() + state.full_rhs.size() +
                 state.correction_full.size());
    }
  }

  uint64_t magic() const { return header_.magic; }
  int data_type() const { return header_.data_type; }
  int start_level() const { return header_.start_level; }
  int num_levels() const { return header_.num_levels; }
  std::size_t bytes() const { return bytes_; }

  void run(void *full_out, const void *full_rhs) {
    if (full_out == nullptr || full_rhs == nullptr) {
      throw std::runtime_error("strict V-cycle fields must be non-null");
    }
    solve_level(start_level_, static_cast<const Complex<Real> *>(full_rhs),
                static_cast<Complex<Real> *>(full_out));
  }

  std::size_t run_fgmres(
      LatticeSetBase *fine, void *full_out, const void *full_rhs,
      const void *gauge, const void *clover_even, const void *clover_odd,
      const void *clover_even_inv, const void *clover_odd_inv,
      const void *fine_null_vectors, int fine_E, int fine_X, int fine_Y,
      int fine_Z, int fine_T, int coarse_E, int coarse_X, int coarse_Y,
      int coarse_Z, int coarse_T, int restart, int max_iter,
      Real tolerance, int nu_pre, int nu_post,
      std::size_t max_workspace_bytes, int &iterations, bool &converged,
      Real &final_true_residual) {
    using C = Complex<Real>;
    if (fine == nullptr || full_out == nullptr || full_rhs == nullptr ||
        gauge == nullptr || clover_even == nullptr ||
        clover_odd == nullptr || clover_even_inv == nullptr ||
        clover_odd_inv == nullptr || fine_null_vectors == nullptr) {
      throw std::runtime_error("strict FGMRES received a null pointer");
    }
    if (start_level_ != 1 || restart <= 0 || max_iter <= 0 ||
        !(tolerance > Real(0) && tolerance < Real(1)) || nu_pre < 0 ||
        nu_post < 0) {
      throw std::runtime_error("invalid strict FGMRES descriptor");
    }
    const StrictHostLevel<Real> &coarse = levels_[start_level_];
    if (coarse.geometry.E != coarse_E || coarse.geometry.X != coarse_X ||
        coarse.geometry.Y != coarse_Y || coarse.geometry.Z != coarse_Z ||
        coarse.geometry.T != coarse_T) {
      throw std::runtime_error(
          "strict FGMRES first-coarse geometry mismatch");
    }
    const int64_t fine_n =
        static_cast<int64_t>(fine_E) * fine_X * fine_Y * fine_Z *
        (fine_T / 2);
    const int64_t coarse_n = coarse.geometry.E * coarse.geometry.volume;
    const uint64_t fine_vectors = 2ULL * restart + 5;
    const uint64_t elements =
        fine_vectors * static_cast<uint64_t>(fine_n) +
        2ULL * static_cast<uint64_t>(coarse_n);
    if (elements >
        std::numeric_limits<std::size_t>::max() / sizeof(C)) {
      throw std::runtime_error("strict FGMRES workspace size overflow");
    }
    const std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(C);
    if (bytes > max_workspace_bytes) {
      throw std::runtime_error("strict FGMRES workspace exceeds budget");
    }

    std::vector<std::vector<C>> V(
        static_cast<std::size_t>(restart) + 1,
        std::vector<C>(static_cast<std::size_t>(fine_n), C(0)));
    std::vector<std::vector<C>> Z(
        static_cast<std::size_t>(restart),
        std::vector<C>(static_cast<std::size_t>(fine_n), C(0)));
    std::vector<C> r(static_cast<std::size_t>(fine_n), C(0));
    std::vector<C> w(static_cast<std::size_t>(fine_n), C(0));
    std::vector<C> x(static_cast<std::size_t>(fine_n), C(0));
    std::vector<C> coarse_rhs(static_cast<std::size_t>(coarse_n), C(0));
    std::vector<C> coarse_out(static_cast<std::size_t>(coarse_n), C(0));
    std::vector<C> smooth_image(static_cast<std::size_t>(fine_n), C(0));

    auto fine_matpc = [&](C *out, const C *in) {
      const int status = fine->strict_fine_matpc(
          out, in, gauge, clover_even, clover_odd, clover_even_inv,
          clover_odd_inv, parity_);
      if (status != 0) {
        throw std::runtime_error("strict fine MATPC failed");
      }
    };
    auto fine_smooth = [&](C *solution, C *residual, int steps) {
      for (int step = 0; step < steps; ++step) {
        fine_matpc(smooth_image.data(), residual);
        const C numerator = strict_global_dot(
            smooth_image.data(), residual, fine_n);
        const Real denominator =
            strict_global_norm2(smooth_image.data(), fine_n);
        if (denominator <= std::numeric_limits<Real>::min()) {
          break;
        }
        const C alpha = numerator / denominator;
        for (int64_t i = 0; i < fine_n; ++i) {
          solution[i] += alpha * residual[i];
          residual[i] -= alpha * smooth_image[i];
        }
      }
    };
    auto precondition = [&](C *out, const C *source) {
      std::fill(out, out + fine_n, C(0));
      std::copy(source, source + fine_n, r.begin());
      fine_smooth(out, r.data(), nu_pre);
      strict_restrict<Real>(
          coarse_rhs.data(), r.data(), fine_null_vectors, coarse.geometry.E,
          fine_E, fine_X, fine_Y, fine_Z, fine_T, coarse.geometry.X,
          coarse.geometry.Y, coarse.geometry.Z, coarse.geometry.T, parity_);
      solve_level(start_level_, coarse_rhs.data(), coarse_out.data());
      strict_prolong<Real>(
          smooth_image.data(), coarse_out.data(), fine_null_vectors,
          coarse.geometry.E, fine_E, fine_X, fine_Y, fine_Z, fine_T,
          coarse.geometry.X, coarse.geometry.Y, coarse.geometry.Z,
          coarse.geometry.T, parity_);
      for (int64_t i = 0; i < fine_n; ++i) {
        out[i] += smooth_image[i];
      }
      fine_matpc(w.data(), out);
      for (std::size_t i = 0; i < r.size(); ++i) {
        r[i] = source[i] - w[i];
      }
      fine_smooth(out, r.data(), nu_post);
    };

    if (fine->strict_fine_prepare(
            r.data(), full_rhs, gauge, clover_even, clover_odd,
            clover_even_inv, clover_odd_inv, parity_) != 0) {
      throw std::runtime_error("strict fine prepare failed");
    }
    const Real b_norm = std::sqrt(strict_global_norm2(r));
    if (params_[kUseInitialGuess] != 0) {
      const auto *initial = static_cast<const C *>(full_out) +
                            static_cast<std::size_t>(parity_) * fine_n;
      std::copy(initial, initial + fine_n, x.begin());
    }
    fine_matpc(w.data(), x.data());
    for (std::size_t i = 0; i < r.size(); ++i) {
      r[i] -= w[i];
    }
    Real residual = std::sqrt(strict_global_norm2(r));
    if (b_norm == Real(0)) {
      std::fill(x.begin(), x.end(), C(0));
      std::fill(r.begin(), r.end(), C(0));
      residual = Real(0);
      iterations = 0;
      converged = true;
      if (fine->strict_fine_reconstruct(
              full_out, full_rhs, x.data(), gauge, clover_even, clover_odd,
              clover_even_inv, clover_odd_inv, parity_) != 0) {
        throw std::runtime_error("strict fine reconstruct failed");
      }
      final_true_residual = residual;
      return bytes;
    }
    const Real target = tolerance * b_norm;
    std::vector<std::complex<Real>> H(
        static_cast<std::size_t>(restart + 1) * restart,
        std::complex<Real>(0));
    std::vector<std::complex<Real>> cs(restart, {1, 0});
    std::vector<std::complex<Real>> sn(restart, {0, 0});
    std::vector<std::complex<Real>> g(restart + 1, {0, 0});
    std::vector<std::complex<Real>> y(restart, {0, 0});
    auto hidx = [restart](int row, int column) {
      return static_cast<std::size_t>(row * restart + column);
    };
    iterations = 0;
    converged = residual <= target;
    while (iterations < max_iter && !converged) {
      const Real beta = residual;
      if (!(beta > Real(0)) || !std::isfinite(static_cast<double>(beta))) {
        throw std::runtime_error("strict FGMRES residual is non-finite");
      }
      for (std::size_t i = 0; i < r.size(); ++i) {
        V[0][i] = r[i] / beta;
      }
      std::fill(H.begin(), H.end(), std::complex<Real>(0));
      std::fill(cs.begin(), cs.end(), std::complex<Real>(1, 0));
      std::fill(sn.begin(), sn.end(), std::complex<Real>(0, 0));
      std::fill(g.begin(), g.end(), std::complex<Real>(0, 0));
      std::fill(y.begin(), y.end(), std::complex<Real>(0, 0));
      g[0] = std::complex<Real>(beta, 0);
      const int cycle = std::min(restart, max_iter - iterations);
      int inner = 0;
      for (int column = 0; column < cycle; ++column) {
        precondition(Z[static_cast<std::size_t>(column)].data(),
                     V[static_cast<std::size_t>(column)].data());
        fine_matpc(w.data(), Z[static_cast<std::size_t>(column)].data());
        for (int row = 0; row <= column; ++row) {
          const C coefficient =
              strict_global_dot(V[static_cast<std::size_t>(row)], w);
          H[hidx(row, column)] =
              std::complex<Real>(coefficient.real(), coefficient.imag());
          for (std::size_t i = 0; i < w.size(); ++i) {
            w[i] -= coefficient * V[static_cast<std::size_t>(row)][i];
          }
        }
        const Real next_norm = std::sqrt(strict_global_norm2(w));
        H[hidx(column + 1, column)] =
            std::complex<Real>(next_norm, 0);
        if (next_norm > (sizeof(Real) == sizeof(float) ? Real(1e-20)
                                                       : Real(1e-30))) {
          for (std::size_t i = 0; i < w.size(); ++i) {
            V[static_cast<std::size_t>(column + 1)][i] =
                w[i] / next_norm;
          }
        } else {
          std::fill(V[static_cast<std::size_t>(column + 1)].begin(),
                    V[static_cast<std::size_t>(column + 1)].end(), C(0));
        }
        for (int row = 0; row < column; ++row) {
          const auto upper = H[hidx(row, column)];
          const auto lower = H[hidx(row + 1, column)];
          H[hidx(row, column)] =
              std::conj(cs[static_cast<std::size_t>(row)]) * upper +
              std::conj(sn[static_cast<std::size_t>(row)]) * lower;
          H[hidx(row + 1, column)] =
              -sn[static_cast<std::size_t>(row)] * upper +
              cs[static_cast<std::size_t>(row)] * lower;
        }
        const auto diagonal = H[hidx(column, column)];
        const auto next = H[hidx(column + 1, column)];
        const Real denominator =
            std::sqrt(std::norm(diagonal) + std::norm(next));
        if (denominator > Real(0)) {
          sn[static_cast<std::size_t>(column)] = next / denominator;
          cs[static_cast<std::size_t>(column)] = diagonal / denominator;
          H[hidx(column, column)] =
              std::conj(cs[static_cast<std::size_t>(column)]) * diagonal +
              std::conj(sn[static_cast<std::size_t>(column)]) * next;
          H[hidx(column + 1, column)] = 0;
        }
        const auto old_g = g[static_cast<std::size_t>(column)];
        g[static_cast<std::size_t>(column + 1)] =
            -sn[static_cast<std::size_t>(column)] * old_g;
        g[static_cast<std::size_t>(column)] =
            std::conj(cs[static_cast<std::size_t>(column)]) * old_g;
        ++iterations;
        inner = column + 1;
        if (std::abs(g[static_cast<std::size_t>(column + 1)]) <= target ||
            next_norm <= (sizeof(Real) == sizeof(float) ? Real(1e-20)
                                                        : Real(1e-30))) {
          break;
        }
      }
      if (inner <= 0) {
        throw std::runtime_error("strict FGMRES produced an empty cycle");
      }
      for (int row = inner - 1; row >= 0; --row) {
        std::complex<Real> value = g[static_cast<std::size_t>(row)];
        for (int column = row + 1; column < inner; ++column) {
          value -= H[hidx(row, column)] *
                   y[static_cast<std::size_t>(column)];
        }
        const auto diagonal = H[hidx(row, row)];
        y[static_cast<std::size_t>(row)] =
            std::abs(diagonal) == Real(0)
                ? std::complex<Real>(0)
                : value / diagonal;
      }
      for (int column = 0; column < inner; ++column) {
        const C coefficient(y[static_cast<std::size_t>(column)].real(),
                            y[static_cast<std::size_t>(column)].imag());
        for (std::size_t i = 0; i < x.size(); ++i) {
          x[i] += coefficient * Z[static_cast<std::size_t>(column)][i];
        }
      }
      fine_matpc(w.data(), x.data());
      if (fine->strict_fine_prepare(
              r.data(), full_rhs, gauge, clover_even, clover_odd,
              clover_even_inv, clover_odd_inv, parity_) != 0) {
        throw std::runtime_error("strict fine prepare failed");
      }
      for (std::size_t i = 0; i < r.size(); ++i) {
        r[i] -= w[i];
      }
      residual = std::sqrt(strict_global_norm2(r));
      converged = residual <= target;
    }
    final_true_residual = residual;
    if (fine->strict_fine_reconstruct(
            full_out, full_rhs, x.data(), gauge, clover_even, clover_odd,
            clover_even_inv, clover_odd_inv, parity_) != 0) {
      throw std::runtime_error("strict fine reconstruct failed");
    }
    return bytes;
  }

private:
  struct Header {
    uint64_t magic = 0;
    int data_type = 0;
    int start_level = 0;
    int num_levels = 0;
  };

  bool solve_coarsest(StrictHostLevel<Real> &state) {
    using C = Complex<Real>;
    const std::size_t count = state.x.size();
    std::fill(state.x.begin(), state.x.end(), C(0));
    state.r = state.pc_rhs;
    state.v.assign(count, C(0));
    state.tmp.assign(count, C(0));
    const Real rhs_norm = std::sqrt(strict_global_norm2(state.pc_rhs));
    if (rhs_norm == Real(0)) {
      return true;
    }
    const Real target = state.tolerance * rhs_norm;
    C rho_old(1);
    C alpha(1);
    C omega(1);
    std::vector<C> rhat = state.pc_rhs;
    std::vector<C> p(count, C(0));
    for (int iteration = 0; iteration < state.max_iter; ++iteration) {
      if (std::sqrt(strict_global_norm2(state.r)) <= target) {
        return true;
      }
      const C rho = strict_global_dot(rhat, state.r);
      if (std::abs(rho) <= std::numeric_limits<Real>::min() ||
          std::abs(omega) <= std::numeric_limits<Real>::min()) {
        return false;
      }
      const C beta = (rho / rho_old) * (alpha / omega);
      for (std::size_t i = 0; i < count; ++i) {
        p[i] = state.r[i] + beta * (p[i] - omega * state.v[i]);
      }
      strict_matpc<Real>(
          state.v.data(), p.data(), state.links, state.tmp.data(),
          reinterpret_cast<const int32_t *>(params_), state.geometry.E,
          state.geometry.X, state.geometry.Y, state.geometry.Z,
          state.geometry.T, parity_);
      const C denominator = strict_global_dot(rhat, state.v);
      if (std::abs(denominator) <= std::numeric_limits<Real>::min()) {
        return false;
      }
      alpha = rho / denominator;
      for (std::size_t i = 0; i < count; ++i) {
        state.tmp[i] = state.r[i] - alpha * state.v[i];
      }
      if (std::sqrt(strict_global_norm2(state.tmp)) <= target) {
        for (std::size_t i = 0; i < count; ++i) {
          state.x[i] += alpha * p[i];
        }
        return true;
      }
      strict_matpc<Real>(
          state.v.data(), state.tmp.data(), state.links, state.r.data(),
          reinterpret_cast<const int32_t *>(params_), state.geometry.E,
          state.geometry.X, state.geometry.Y, state.geometry.Z,
          state.geometry.T, parity_);
      const C tt = strict_global_dot(state.v, state.v);
      if (std::abs(tt) <= std::numeric_limits<Real>::min()) {
        return false;
      }
      omega = strict_global_dot(state.v, state.tmp) / tt;
      if (std::abs(omega) <= std::numeric_limits<Real>::min()) {
        return false;
      }
      for (std::size_t i = 0; i < count; ++i) {
        state.x[i] += alpha * p[i] + omega * state.tmp[i];
        state.r[i] = state.tmp[i] - omega * state.v[i];
      }
      rho_old = rho;
    }
    return std::sqrt(strict_global_norm2(state.r)) <= target;
  }

  void smooth(StrictHostLevel<Real> &state, int steps) {
    using C = Complex<Real>;
    for (int step = 0; step < steps; ++step) {
      strict_matpc<Real>(
          state.v.data(), state.r.data(), state.links, state.tmp.data(),
          reinterpret_cast<const int32_t *>(params_), state.geometry.E,
          state.geometry.X, state.geometry.Y, state.geometry.Z,
          state.geometry.T, parity_);
      const C numerator = strict_global_dot(state.v, state.r);
      const Real denominator = strict_global_norm2(state.v);
      if (denominator <= std::numeric_limits<Real>::min()) {
        return;
      }
      const C alpha = numerator / denominator;
      for (std::size_t i = 0; i < state.x.size(); ++i) {
        state.x[i] += alpha * state.r[i];
        state.r[i] -= alpha * state.v[i];
      }
    }
  }

  void solve_level(int level, const Complex<Real> *full_rhs,
                   Complex<Real> *full_out) {
    StrictHostLevel<Real> &state = levels_[level];
    strict_prepare<Real>(
        state.pc_rhs.data(), full_rhs, state.links, state.onsite_pair,
        state.tmp.data(), reinterpret_cast<const int32_t *>(params_),
        state.geometry.E, state.geometry.X, state.geometry.Y,
        state.geometry.Z, state.geometry.T, parity_);
    std::fill(state.x.begin(), state.x.end(), Complex<Real>(0));
    if (level == num_levels_ - 1) {
      if (!solve_coarsest(state)) {
        throw std::runtime_error("strict coarsest BiCGStab did not converge");
      }
    } else {
      state.r = state.pc_rhs;
      smooth(state, smoother_steps_);
      StrictHostLevel<Real> &child = levels_[level + 1];
      strict_restrict<Real>(
          child.full_rhs.data(), state.r.data(), state.null_vectors,
          child.geometry.E, state.geometry.E, state.geometry.X,
          state.geometry.Y, state.geometry.Z, state.geometry.T,
          child.geometry.X, child.geometry.Y, child.geometry.Z,
          child.geometry.T, parity_);
      solve_level(level + 1, child.full_rhs.data(),
                  state.correction_full.data());
      strict_prolong<Real>(
          state.tmp.data(), state.correction_full.data(), state.null_vectors,
          child.geometry.E, state.geometry.E, state.geometry.X,
          state.geometry.Y, state.geometry.Z, state.geometry.T,
          child.geometry.X, child.geometry.Y, child.geometry.Z,
          child.geometry.T, parity_);
      for (std::size_t i = 0; i < state.x.size(); ++i) {
        state.x[i] += state.tmp[i];
      }
      strict_matpc<Real>(
          state.v.data(), state.x.data(), state.links, state.tmp.data(),
          reinterpret_cast<const int32_t *>(params_), state.geometry.E,
          state.geometry.X, state.geometry.Y, state.geometry.Z,
          state.geometry.T, parity_);
      for (std::size_t i = 0; i < state.r.size(); ++i) {
        state.r[i] = state.pc_rhs[i] - state.v[i];
      }
      smooth(state, smoother_steps_);
    }
    strict_reconstruct<Real>(
        full_out, full_rhs, state.x.data(), state.links, state.onsite_pair,
        state.tmp.data(), reinterpret_cast<const int32_t *>(params_),
        state.geometry.E, state.geometry.X, state.geometry.Y,
        state.geometry.Z, state.geometry.T, parity_);
  }

  Header header_;
  int start_level_ = 1;
  int num_levels_ = 0;
  int parity_ = 0;
  int smoother_steps_ = 2;
  long long *table_ = nullptr;
  const int32_t *params_ = nullptr;
  StrictTopology topology_;
  std::array<StrictHostLevel<Real>, kMaxLevels> levels_{};
  std::size_t bytes_ = 0;
};

} // namespace

template <typename Real>
int strict_coarse(void *out, const void *in, const void *links,
                  const void *onsite_pair, const int32_t *params, int E,
                  int X, int Y, int Z, int T, int onsite_index) {
  if (out == nullptr || in == nullptr || links == nullptr ||
      onsite_pair == nullptr || onsite_index < -1 || onsite_index > 1) {
    throw std::runtime_error("invalid strict coarse operator descriptor");
  }
  const StrictGeometry geometry = strict_geometry<Real>(E, X, Y, Z, T);
  const StrictTopology topology = strict_topology(params);
  auto *output = static_cast<Complex<Real> *>(out);
  const auto *input = static_cast<const Complex<Real> *>(in);
  const auto *link_data = static_cast<const Complex<Real> *>(links);
  const auto *onsite = static_cast<const Complex<Real> *>(onsite_pair);
  strict_hopping_full(
      output, input, link_data,
      static_cast<const Complex<Real> *>(nullptr), geometry, topology, 10000);
  if (onsite_index >= 0) {
    const int64_t count = static_cast<int64_t>(E) * geometry.volume;
    const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < count; ++index) {
      const int row = static_cast<int>(index / geometry.volume);
      const int64_t site = index % geometry.volume;
      Complex<Real> sum(0);
      for (int column = 0; column < E; ++column) {
        const int64_t matrix =
            (((static_cast<int64_t>(onsite_index) * E + row) * E + column) *
                 geometry.volume +
             site);
        sum += onsite[matrix] *
               input[static_cast<int64_t>(column) * geometry.volume + site];
      }
      output[index] += sum;
    }
  }
  return 0;
}

template <typename Real>
int strict_matpc(void *out, const void *in, const void *links, void *scratch,
                 const int32_t *params, int E, int X, int Y, int Z, int T,
                 int parity) {
  if (out == nullptr || in == nullptr || links == nullptr ||
      scratch == nullptr || parity < 0 || parity > 1) {
    throw std::runtime_error("invalid strict MATPC descriptor");
  }
  const StrictGeometry geometry =
      strict_geometry<Real>(E, X, Y, Z, T, true);
  const StrictTopology topology = strict_topology(params);
  auto *output = static_cast<Complex<Real> *>(out);
  auto *work = static_cast<Complex<Real> *>(scratch);
  const auto *input = static_cast<const Complex<Real> *>(in);
  const auto *link_data = static_cast<const Complex<Real> *>(links);
  strict_hopping_compact(
      work, input, link_data, static_cast<const Complex<Real> *>(nullptr),
      geometry, topology, 1 - parity, 11000);
  strict_hopping_compact(output, work, link_data, input, geometry, topology,
                         parity, 12000);
  return 0;
}

template <typename Real>
int strict_prepare(void *out, const void *full_rhs, const void *links,
                   const void *onsite_pair, void *scratch,
                   const int32_t *params, int E, int X, int Y, int Z, int T,
                   int parity) {
  if (out == nullptr || full_rhs == nullptr || links == nullptr ||
      onsite_pair == nullptr || scratch == nullptr || parity < 0 ||
      parity > 1) {
    throw std::runtime_error("invalid strict prepare descriptor");
  }
  const StrictGeometry geometry =
      strict_geometry<Real>(E, X, Y, Z, T, true);
  const StrictTopology topology = strict_topology(params);
  auto *output = static_cast<Complex<Real> *>(out);
  auto *work = static_cast<Complex<Real> *>(scratch);
  const auto *rhs = static_cast<const Complex<Real> *>(full_rhs);
  const auto *link_data = static_cast<const Complex<Real> *>(links);
  const auto *onsite = static_cast<const Complex<Real> *>(onsite_pair);
  const int other = 1 - parity;
  strict_onsite_full_to_parity(output, rhs, onsite, geometry, parity, 1);
  strict_onsite_full_to_parity(work, rhs, onsite, geometry, other, 1);
  std::vector<Complex<Real>> correction(
      static_cast<std::size_t>(E * geometry.half_volume));
  strict_hopping_compact(
      correction.data(), work, link_data,
      static_cast<const Complex<Real> *>(nullptr), geometry, topology, parity,
      13000);
  const int64_t count = static_cast<int64_t>(E) * geometry.half_volume;
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < count; ++index) {
    output[index] -= correction[static_cast<std::size_t>(index)];
  }
  return 0;
}

template <typename Real>
int strict_reconstruct(void *full_out, const void *full_rhs,
                       const void *target_solution, const void *links,
                       const void *onsite_pair, void *scratch,
                       const int32_t *params, int E, int X, int Y, int Z,
                       int T, int parity) {
  if (full_out == nullptr || full_rhs == nullptr ||
      target_solution == nullptr || links == nullptr ||
      onsite_pair == nullptr || scratch == nullptr || parity < 0 ||
      parity > 1) {
    throw std::runtime_error("invalid strict reconstruct descriptor");
  }
  const StrictGeometry geometry =
      strict_geometry<Real>(E, X, Y, Z, T, true);
  const StrictTopology topology = strict_topology(params);
  auto *output = static_cast<Complex<Real> *>(full_out);
  auto *other_solution = static_cast<Complex<Real> *>(scratch);
  const auto *rhs = static_cast<const Complex<Real> *>(full_rhs);
  const auto *target =
      static_cast<const Complex<Real> *>(target_solution);
  const auto *link_data = static_cast<const Complex<Real> *>(links);
  const auto *onsite = static_cast<const Complex<Real> *>(onsite_pair);
  const int other = 1 - parity;
  strict_onsite_full_to_parity(other_solution, rhs, onsite, geometry, other,
                               1);
  strict_hopping_compact(other_solution, target, link_data, other_solution,
                         geometry, topology, other, 14000);
  strict_join_parities(output, target, other_solution, geometry, parity);
  return 0;
}

template <typename Real>
int strict_restrict(void *coarse_out, const void *fine_in,
                    const void *null_vectors, int E, int e, int Xf, int Yf,
                    int Zf, int Tf, int Xc, int Yc, int Zc, int Tc,
                    int parity) {
  if (coarse_out == nullptr || fine_in == nullptr ||
      null_vectors == nullptr || parity < 0 || parity > 1) {
    throw std::runtime_error("invalid strict restrict descriptor");
  }
  strict_validate_parity_geometry(Xf, Yf, Zf, Tf);
  if (E <= 0 || e <= 0 || Xc <= 0 || Yc <= 0 || Zc <= 0 || Tc <= 0 ||
      Xf % Xc != 0 || Yf % Yc != 0 || Zf % Zc != 0 || Tf % Tc != 0) {
    throw std::runtime_error("invalid strict restrict geometry");
  }
  const int64_t coarse_volume =
      static_cast<int64_t>(Xc) * Yc * Zc * Tc;
  const int64_t fine_volume =
      static_cast<int64_t>(Xf) * Yf * Zf * Tf;
  const int64_t output_count = static_cast<int64_t>(E) * coarse_volume;
  const int block_x = Xf / Xc;
  const int block_y = Yf / Yc;
  const int block_z = Zf / Zc;
  const int block_t = Tf / Tc;
  const int64_t chunk = omp_chunk<Real>(output_count);
  auto *output = static_cast<Complex<Real> *>(coarse_out);
  const auto *input = static_cast<const Complex<Real> *>(fine_in);
  const auto *null_vectors_data =
      static_cast<const Complex<Real> *>(null_vectors);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < output_count; ++index) {
    const int coarse_component = static_cast<int>(index / coarse_volume);
    const int64_t coarse_site = index % coarse_volume;
    int tc = static_cast<int>(coarse_site % Tc);
    int xyz = static_cast<int>(coarse_site / Tc);
    int zc = xyz % Zc;
    xyz /= Zc;
    int yc = xyz % Yc;
    const int xc = xyz / Yc;
    (void)tc;
    Complex<Real> sum(0);
    for (int dx = 0; dx < block_x; ++dx) {
      const int x = xc * block_x + dx;
      for (int dy = 0; dy < block_y; ++dy) {
        const int y = yc * block_y + dy;
        for (int dz = 0; dz < block_z; ++dz) {
          const int z = zc * block_z + dz;
          for (int dt = 0; dt < block_t; ++dt) {
            const int t = tc * block_t + dt;
            if (((x + y + z + t) & 1) != parity) {
              continue;
            }
            const int64_t fine_half =
                ((static_cast<int64_t>(x) * Yf + y) * Zf + z) * (Tf / 2) +
                t / 2;
            for (int fine_component = 0; fine_component < e;
                 ++fine_component) {
              const std::size_t null_index = strict_null_index(
                  coarse_component, fine_component, x, y, z, t, e, Xf, Yf,
                  Zf, Tf, Xc, Yc, Zc, Tc);
              sum += std::conj(null_vectors_data[null_index]) *
                     input[static_cast<int64_t>(fine_component) *
                               (fine_volume / 2) +
                           fine_half];
            }
          }
        }
      }
    }
    output[index] = sum;
  }
  return 0;
}

template <typename Real>
int strict_prolong(void *fine_out, const void *coarse_in,
                   const void *null_vectors, int E, int e, int Xf, int Yf,
                   int Zf, int Tf, int Xc, int Yc, int Zc, int Tc,
                   int parity) {
  if (fine_out == nullptr || coarse_in == nullptr ||
      null_vectors == nullptr || parity < 0 || parity > 1) {
    throw std::runtime_error("invalid strict prolong descriptor");
  }
  strict_validate_parity_geometry(Xf, Yf, Zf, Tf);
  if (E <= 0 || e <= 0 || Xc <= 0 || Yc <= 0 || Zc <= 0 || Tc <= 0 ||
      Xf % Xc != 0 || Yf % Yc != 0 || Zf % Zc != 0 || Tf % Tc != 0) {
    throw std::runtime_error("invalid strict prolong geometry");
  }
  const int64_t fine_half_volume =
      static_cast<int64_t>(Xf) * Yf * Zf * (Tf / 2);
  const int64_t coarse_volume =
      static_cast<int64_t>(Xc) * Yc * Zc * Tc;
  const int64_t output_count = static_cast<int64_t>(e) * fine_half_volume;
  const int block_x = Xf / Xc;
  const int block_y = Yf / Yc;
  const int block_z = Zf / Zc;
  const int block_t = Tf / Tc;
  const int64_t chunk = omp_chunk<Real>(output_count);
  auto *output = static_cast<Complex<Real> *>(fine_out);
  const auto *input = static_cast<const Complex<Real> *>(coarse_in);
  const auto *null_vectors_data =
      static_cast<const Complex<Real> *>(null_vectors);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < output_count; ++index) {
    const int fine_component =
        static_cast<int>(index / fine_half_volume);
    const int64_t half_site = index % fine_half_volume;
    int xt = 0;
    const int t2 = static_cast<int>(half_site % (Tf / 2));
    xt = static_cast<int>(half_site / (Tf / 2));
    int z = xt % Zf;
    xt /= Zf;
    int y = xt % Yf;
    const int x = xt / Yf;
    const int t = 2 * t2 + (((x + y + z) & 1) ^ parity);
    const int xc = x / block_x;
    const int yc = y / block_y;
    const int zc = z / block_z;
    const int tc = t / block_t;
    const int64_t coarse_site =
        ((static_cast<int64_t>(xc) * Yc + yc) * Zc + zc) * Tc + tc;
    Complex<Real> sum(0);
    for (int coarse_component = 0; coarse_component < E;
         ++coarse_component) {
      const std::size_t null_index = strict_null_index(
          coarse_component, fine_component, x, y, z, t, e, Xf, Yf, Zf, Tf,
          Xc, Yc, Zc, Tc);
      sum += null_vectors_data[null_index] *
             input[static_cast<int64_t>(coarse_component) * coarse_volume +
                   coarse_site];
    }
    output[index] = sum;
  }
  return 0;
}

template <typename Real>
std::size_t strict_init_hierarchy(void *set_ptrs, const int32_t *params,
                                  int start_level) {
  if (set_ptrs == nullptr || params == nullptr) {
    throw std::runtime_error("strict hierarchy init received null controls");
  }
  auto *table = static_cast<long long *>(set_ptrs);
  if (table[kSetPtrsStrictHierarchy] != 0) {
    throw std::runtime_error("strict hierarchy is already initialized");
  }
  auto *hierarchy =
      new StrictHostHierarchy<Real>(set_ptrs, params, start_level);
  table[kSetPtrsStrictHierarchy] =
      reinterpret_cast<long long>(hierarchy);
  return hierarchy->bytes();
}

template <typename Real>
std::size_t strict_vcycle(void *full_out, const void *full_rhs,
                          void *set_ptrs, const int32_t *params,
                          int start_level) {
  if (set_ptrs == nullptr || params == nullptr) {
    throw std::runtime_error("strict V-cycle received null controls");
  }
  auto *table = static_cast<long long *>(set_ptrs);
  auto *hierarchy = reinterpret_cast<StrictHostHierarchy<Real> *>(
      table[kSetPtrsStrictHierarchy]);
  if (hierarchy != nullptr) {
    if (hierarchy->magic() != StrictHostHierarchy<Real>::kMagic ||
        hierarchy->data_type() != params[kDataType]) {
      throw std::runtime_error("strict hierarchy type mismatch");
    }
    if (hierarchy->start_level() != start_level) {
      throw std::runtime_error(
          "strict hierarchy start_level mismatch");
    }
    hierarchy->run(full_out, full_rhs);
    return hierarchy->bytes();
  }
  StrictHostHierarchy<Real> temporary(set_ptrs, params, start_level);
  temporary.run(full_out, full_rhs);
  return temporary.bytes();
}

template <typename Real>
int strict_end_hierarchy(void *set_ptrs) {
  if (set_ptrs == nullptr) {
    throw std::runtime_error("strict hierarchy end received null set_ptrs");
  }
  auto *table = static_cast<long long *>(set_ptrs);
  auto *hierarchy = reinterpret_cast<StrictHostHierarchy<Real> *>(
      table[kSetPtrsStrictHierarchy]);
  if (hierarchy == nullptr) {
    return 0;
  }
  if (hierarchy->magic() != StrictHostHierarchy<Real>::kMagic) {
    throw std::runtime_error("strict hierarchy end type mismatch");
  }
  table[kSetPtrsStrictHierarchy] = 0;
  delete hierarchy;
  return 0;
}

template <typename Real>
std::size_t strict_fgmres(
    void *full_out, const void *full_rhs, const void *gauge,
    const void *clover_even, const void *clover_odd,
    const void *clover_even_inv, const void *clover_odd_inv,
    const void *fine_null_vectors, void *set_ptrs, const int32_t *params,
    int fine_E, int fine_X, int fine_Y, int fine_Z, int fine_T, int coarse_E,
    int coarse_X, int coarse_Y, int coarse_Z, int coarse_T, int element_bytes,
    int restart, int max_iter, double tolerance, int nu_pre, int nu_post,
    std::size_t max_workspace_bytes, int &iterations, bool &converged,
    Real &final_true_residual) {
  if (set_ptrs == nullptr || params == nullptr ||
      element_bytes != static_cast<int>(sizeof(Complex<Real>))) {
    throw std::runtime_error("invalid strict FGMRES ABI descriptor");
  }
  auto *table = static_cast<long long *>(set_ptrs);
  auto *hierarchy = reinterpret_cast<StrictHostHierarchy<Real> *>(
      table[kSetPtrsStrictHierarchy]);
  if (hierarchy == nullptr) {
    throw std::runtime_error(
        "strict FGMRES requires an initialized hierarchy");
  }
  if (hierarchy->magic() != StrictHostHierarchy<Real>::kMagic ||
      hierarchy->data_type() != params[kDataType]) {
    throw std::runtime_error("strict FGMRES hierarchy type mismatch");
  }
  const int index = params[kSetIndex];
  if (index < 0 || index >= kSetPtrsSize) {
    throw std::runtime_error("strict FGMRES set index is out of range");
  }
  const auto lattice_set = find_lattice_set(table + index);
  return hierarchy->run_fgmres(
      lattice_set.get(), full_out, full_rhs, gauge, clover_even, clover_odd,
      clover_even_inv, clover_odd_inv, fine_null_vectors, fine_E, fine_X,
      fine_Y, fine_Z, fine_T, coarse_E, coarse_X, coarse_Y, coarse_Z,
      coarse_T, restart, max_iter, static_cast<Real>(tolerance), nu_pre,
      nu_post, max_workspace_bytes, iterations, converged,
      final_true_residual);
}

template int strict_coarse<float>(void *, const void *, const void *,
                                  const void *, const int32_t *, int, int, int,
                                  int, int, int);
template int strict_coarse<double>(void *, const void *, const void *,
                                   const void *, const int32_t *, int, int, int,
                                   int, int, int);
template int strict_matpc<float>(void *, const void *, const void *, void *,
                                 const int32_t *, int, int, int, int, int, int);
template int strict_matpc<double>(void *, const void *, const void *, void *,
                                  const int32_t *, int, int, int, int, int,
                                  int);
template int strict_prepare<float>(void *, const void *, const void *,
                                   const void *, void *, const int32_t *, int,
                                   int, int, int, int, int);
template int strict_prepare<double>(void *, const void *, const void *,
                                    const void *, void *, const int32_t *, int,
                                    int, int, int, int, int);
template int strict_reconstruct<float>(
    void *, const void *, const void *, const void *, const void *, void *,
    const int32_t *, int, int, int, int, int, int);
template int strict_reconstruct<double>(
    void *, const void *, const void *, const void *, const void *, void *,
    const int32_t *, int, int, int, int, int, int);
template int strict_restrict<float>(void *, const void *, const void *, int,
                                    int, int, int, int, int, int, int, int,
                                    int, int);
template int strict_restrict<double>(void *, const void *, const void *, int,
                                     int, int, int, int, int, int, int, int,
                                     int, int);
template int strict_prolong<float>(void *, const void *, const void *, int,
                                   int, int, int, int, int, int, int, int, int,
                                   int);
template int strict_prolong<double>(void *, const void *, const void *, int,
                                    int, int, int, int, int, int, int, int,
                                    int, int);
template std::size_t strict_init_hierarchy<float>(void *, const int32_t *,
                                                  int);
template std::size_t strict_init_hierarchy<double>(void *, const int32_t *,
                                                   int);
template std::size_t strict_vcycle<float>(void *, const void *, void *,
                                          const int32_t *, int);
template std::size_t strict_vcycle<double>(void *, const void *, void *,
                                           const int32_t *, int);
template int strict_end_hierarchy<float>(void *);
template int strict_end_hierarchy<double>(void *);
template std::size_t strict_fgmres<float>(
    void *, const void *, const void *, const void *, const void *,
    const void *, const void *, const void *, void *, const int32_t *, int,
    int, int, int, int, int, int, int, int, int, int, int, int, double, int,
    int, std::size_t, int &, bool &, float &);
template std::size_t strict_fgmres<double>(
    void *, const void *, const void *, const void *, const void *,
    const void *, const void *, const void *, void *, const int32_t *, int,
    int, int, int, int, int, int, int, int, int, int, int, int, double, int,
    int, std::size_t, int &, bool &, double &);

template class LatticeSet<float>;
template class LatticeSet<double>;

} // namespace kupl::qcu
