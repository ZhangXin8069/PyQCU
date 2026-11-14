#include "kupl/lattice_multigrid.h"
#include "kupl/lattice_linalg.h"
#include "kupl/lattice_set.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <omp.h>

namespace kupl::qcu {

int coarse_shift_component(int direction, int dim) {
  static constexpr int shifts[kCoarseDirections][kDim] = {
      {1, 0, 0, 0},   {-1, 0, 0, 0},  {0, 1, 0, 0},   {0, -1, 0, 0},
      {0, 0, 1, 0},   {0, 0, -1, 0},  {0, 0, 0, 1},   {0, 0, 0, -1},
      {1, 1, 0, 0},   {-1, -1, 0, 0}, {1, -1, 0, 0},  {-1, 1, 0, 0},
      {1, 0, 1, 0},   {-1, 0, -1, 0}, {1, 0, -1, 0},  {-1, 0, 1, 0},
      {1, 0, 0, 1},   {-1, 0, 0, -1}, {1, 0, 0, -1},  {-1, 0, 0, 1},
      {0, 1, 1, 0},   {0, -1, -1, 0}, {0, 1, -1, 0},  {0, -1, 1, 0},
      {0, 1, 0, 1},   {0, -1, 0, -1}, {0, 1, 0, -1},  {0, -1, 0, 1},
      {0, 0, 1, 1},   {0, 0, -1, -1}, {0, 0, 1, -1},  {0, 0, -1, 1}};
  return shifts[direction][dim];
}

int coarse_direction_for_shift(const int shift[kDim]) {
  for (int direction = 0; direction < kCoarseDirections; ++direction) {
    bool matches = true;
    for (int dim = 0; dim < kDim; ++dim) {
      if (coarse_shift_component(direction, dim) != shift[dim]) {
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
int64_t LatticeSet<Real>::coarse_volume(
    const std::array<int, kDim> &dims) const {
  int64_t volume = 1;
  for (int dim = 0; dim < kDim; ++dim) {
    volume *= dims[dim];
  }
  return volume;
}

template <typename Real>
int64_t LatticeSet<Real>::coarse_site_index(
    const std::array<int, kDim> &dims, int x, int y, int z, int t) const {
  return ((static_cast<int64_t>(x) * dims[1] + y) * dims[2] + z) *
             dims[3] +
         t;
}

template <typename Real>
void LatticeSet<Real>::decode_coarse_site(
    int64_t site, const std::array<int, kDim> &dims, int &x, int &y, int &z,
    int &t) const {
  t = static_cast<int>(site % dims[3]);
  site /= dims[3];
  z = static_cast<int>(site % dims[2]);
  site /= dims[2];
  y = static_cast<int>(site % dims[1]);
  site /= dims[1];
  x = static_cast<int>(site);
}

template <typename Real>
int64_t LatticeSet<Real>::coarse_face_sites(
    int direction, const std::array<int, kDim> &dims) const {
  int64_t count = 1;
  for (int dim = 0; dim < kDim; ++dim) {
    if (coarse_shift_component(direction, dim) == 0) {
      count *= dims[dim];
    }
  }
  return count;
}

template <typename Real>
bool LatticeSet<Real>::coarse_direction_distributed(int direction) const {
  for (int dim = 0; dim < kDim; ++dim) {
    if (coarse_shift_component(direction, dim) != 0 &&
        geometry_.grid[dim] > 1) {
      return true;
    }
  }
  return false;
}

template <typename Real>
void LatticeSet<Real>::decode_coarse_face(
    int direction, int64_t face, const std::array<int, kDim> &dims,
    std::array<int, kDim> &coordinate) const {
  for (int dim = kDim - 1; dim >= 0; --dim) {
    const int shift = coarse_shift_component(direction, dim);
    if (shift == 0) {
      coordinate[dim] = static_cast<int>(face % dims[dim]);
      face /= dims[dim];
    } else {
      coordinate[dim] = shift > 0 ? dims[dim] - 1 : 0;
    }
  }
}

template <typename Real>
int64_t LatticeSet<Real>::coarse_face_index(
    int direction, const std::array<int, kDim> &coordinate,
    const std::array<int, kDim> &dims) const {
  int64_t index = 0;
  int64_t stride = 1;
  for (int dim = kDim - 1; dim >= 0; --dim) {
    if (coarse_shift_component(direction, dim) == 0) {
      index += static_cast<int64_t>(coordinate[dim]) * stride;
      stride *= dims[dim];
    }
  }
  return index;
}

template <typename Real>
void LatticeSet<Real>::ensure_coarse_halo(
    int E, const std::array<int, kDim> &dims) {
  if (E <= 0) {
    throw std::runtime_error("coarse halo E must be positive");
  }
  for (int dim = 0; dim < kDim; ++dim) {
    if (dims[dim] <= 0) {
      throw std::runtime_error("coarse halo dimensions must be positive");
    }
  }
  if (coarse_halo_E_ == E && coarse_halo_dims_ == dims) {
    return;
  }
  int64_t max_face = 0;
  for (int direction = 0; direction < kCoarseDirections; ++direction) {
    max_face = std::max(max_face, coarse_face_sites(direction, dims));
  }
  const int64_t elements = static_cast<int64_t>(E) * max_face;
  if (elements <= 0 ||
      elements > std::numeric_limits<int>::max() /
                     static_cast<int64_t>(sizeof(Complex<Real>))) {
    throw std::runtime_error("coarse halo message is too large");
  }
  coarse_halo_E_ = E;
  coarse_halo_dims_ = dims;
  for (int direction = 0; direction < kCoarseDirections; ++direction) {
    coarse_halo_offsets_[direction] = 0;
    coarse_halo_send_[direction].resize(static_cast<std::size_t>(elements));
    coarse_halo_recv_[direction].resize(static_cast<std::size_t>(elements));
  }
}

template <typename Real>
void LatticeSet<Real>::pack_coarse_direction(
    const Complex<Real> *input, int E, const std::array<int, kDim> &dims,
    int direction) {
  const int64_t face_sites = coarse_face_sites(direction, dims);
  const int64_t volume = coarse_volume(dims);
  auto &buffer = coarse_halo_send_[direction];
  const int offset = coarse_halo_offsets_[direction];
  const int64_t chunk = omp_chunk<Real>(face_sites);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t face = 0; face < face_sites; ++face) {
    std::array<int, kDim> coordinate{};
    decode_coarse_face(direction, face, dims, coordinate);
    const int64_t site = coarse_site_index(
        dims, coordinate[0], coordinate[1], coordinate[2], coordinate[3]);
    for (int e = 0; e < E; ++e) {
      buffer[static_cast<std::size_t>(offset + face * E + e)] =
          input[static_cast<int64_t>(e) * volume + site];
    }
  }
}

template <typename Real>
int LatticeSet<Real>::coarse_peer(int direction) const {
  std::array<int, kDim> coordinate = geometry_.coord;
  for (int dim = 0; dim < kDim; ++dim) {
    const int shift = coarse_shift_component(direction, dim);
    if (shift != 0 && geometry_.grid[dim] > 1) {
      coordinate[dim] =
          (coordinate[dim] + shift + geometry_.grid[dim]) %
          geometry_.grid[dim];
    }
  }
  return geometry_.cartesian_rank(coordinate);
}

template <typename Real>
void LatticeSet<Real>::exchange_coarse_halo(
    const Complex<Real> *input, int E, const std::array<int, kDim> &dims) {
  ensure_coarse_halo(E, dims);
  std::vector<MPI_Request> requests;
  requests.reserve(2 * kCoarseDirections);
  for (int direction = 0; direction < kCoarseDirections; ++direction) {
    if (!coarse_direction_distributed(direction)) {
      continue;
    }
    pack_coarse_direction(input, E, dims, direction);
    const int peer = coarse_peer(direction);
    auto &receive = coarse_halo_recv_[direction];
    if (peer == geometry_.rank) {
      continue;
    }
    MPI_Request request = MPI_REQUEST_NULL;
    MPI_Irecv(receive.data(),
              static_cast<int>(receive.size() * sizeof(Complex<Real>)),
              MPI_BYTE, peer, 7100 + (direction ^ 1), geometry_.comm,
              &request);
    requests.push_back(request);
  }
  for (int direction = 0; direction < kCoarseDirections; ++direction) {
    if (!coarse_direction_distributed(direction)) {
      continue;
    }
    const int peer = coarse_peer(direction);
    auto &send = coarse_halo_send_[direction];
    if (peer == geometry_.rank) {
      coarse_halo_recv_[direction] = send;
      continue;
    }
    MPI_Request request = MPI_REQUEST_NULL;
    MPI_Isend(send.data(),
              static_cast<int>(send.size() * sizeof(Complex<Real>)),
              MPI_BYTE, peer, 7100 + direction, geometry_.comm, &request);
    requests.push_back(request);
  }
  if (!requests.empty()) {
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                MPI_STATUSES_IGNORE);
  }
}

template <typename Real>
Complex<Real> LatticeSet<Real>::coarse_neighbor(
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
  const int direction = coarse_direction_for_shift(crossing);
  if (direction >= 0) {
    const int64_t face = coarse_face_index(direction, target, dims);
    return coarse_halo_recv_[direction][static_cast<std::size_t>(
        coarse_halo_offsets_[direction] + face * E + e)];
  }
  const int64_t volume = coarse_volume(dims);
  const int64_t site = coarse_site_index(
      dims, target[0], target[1], target[2], target[3]);
  return input[static_cast<int64_t>(e) * volume + site];
}

template <typename Real>
std::size_t LatticeSet<Real>::multigrid_null_index(
    int E_idx, int e_idx, int x, int y, int z, int t, int e, int Xf, int Yf,
    int Zf, int Tf, int Xc, int Yc, int Zc, int Tc) const {
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

template <typename Real>
void LatticeSet<Real>::validate_multigrid_transfer(
    int E, int e, int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc,
    int Tc) const {
  if (E <= 0 || e <= 0 || Xf <= 0 || Yf <= 0 || Zf <= 0 || Tf <= 0 ||
      Xc <= 0 || Yc <= 0 || Zc <= 0 || Tc <= 0) {
    throw std::runtime_error(
        "multigrid transfer dimensions must be positive");
  }
  if (Xf % Xc != 0 || Yf % Yc != 0 || Zf % Zc != 0 || Tf % Tc != 0) {
    throw std::runtime_error(
        "fine multigrid dimensions must be divisible by coarse dimensions");
  }
}

template <typename Real>
void LatticeSet<Real>::apply_multigrid_restrict(
    Complex<Real> *coarse_out, const Complex<Real> *fine_in,
    const Complex<Real> *null_vectors, int E, int e, int Xf, int Yf, int Zf,
    int Tf, int Xc, int Yc, int Zc, int Tc) {
  if (coarse_out == nullptr || fine_in == nullptr ||
      null_vectors == nullptr) {
    throw std::runtime_error("multigrid restrict received a null pointer");
  }
  validate_multigrid_transfer(E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
  const int bx = Xf / Xc;
  const int by = Yf / Yc;
  const int bz = Zf / Zc;
  const int bt = Tf / Tc;
  const int64_t fine_volume =
      static_cast<int64_t>(Xf) * Yf * Zf * Tf;
  const int64_t coarse_volume =
      static_cast<int64_t>(Xc) * Yc * Zc * Tc;
  const int64_t output_count = static_cast<int64_t>(E) * coarse_volume;
  const int64_t chunk = omp_chunk<Real>(output_count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t output = 0; output < output_count; ++output) {
    const int E_idx = static_cast<int>(output / coarse_volume);
    const int64_t coarse_site = output % coarse_volume;
    int xc = 0;
    int yc = 0;
    int zc = 0;
    int tc = 0;
    decode_coarse_site(coarse_site, {Xc, Yc, Zc, Tc}, xc, yc, zc, tc);
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
              const std::size_t null_index = multigrid_null_index(
                  E_idx, e_idx, x, y, z, t, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc,
                  Tc);
              const Complex<Real> &projector = null_vectors[null_index];
              const Complex<Real> &vector = fine_in[fine_index];
              sum += Complex<Real>(
                  projector.real() * vector.real() +
                      projector.imag() * vector.imag(),
                  projector.real() * vector.imag() -
                      projector.imag() * vector.real());
            }
          }
        }
      }
    }
    coarse_out[output] = sum;
  }
}

template <typename Real>
void LatticeSet<Real>::apply_multigrid_prolong(
    Complex<Real> *fine_out, const Complex<Real> *coarse_in,
    const Complex<Real> *null_vectors, int E, int e, int Xf, int Yf, int Zf,
    int Tf, int Xc, int Yc, int Zc, int Tc) {
  if (fine_out == nullptr || coarse_in == nullptr ||
      null_vectors == nullptr) {
    throw std::runtime_error("multigrid prolong received a null pointer");
  }
  validate_multigrid_transfer(E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
  const int bx = Xf / Xc;
  const int by = Yf / Yc;
  const int bz = Zf / Zc;
  const int bt = Tf / Tc;
  const int64_t fine_volume =
      static_cast<int64_t>(Xf) * Yf * Zf * Tf;
  const int64_t null_count = static_cast<int64_t>(E) * e * fine_volume;
  const int64_t coarse_volume =
      static_cast<int64_t>(Xc) * Yc * Zc * Tc;
  const int64_t output_count = static_cast<int64_t>(e) * fine_volume;
  const int64_t chunk = omp_chunk<Real>(output_count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t output = 0; output < output_count; ++output) {
    const int e_idx = static_cast<int>(output / fine_volume);
    const int64_t fine_site = output % fine_volume;
    int x = 0;
    int y = 0;
    int z = 0;
    int t = 0;
    decode_coarse_site(fine_site, {Xf, Yf, Zf, Tf}, x, y, z, t);
    const int xc = x / bx;
    const int yc = y / by;
    const int zc = z / bz;
    const int tc = t / bt;
    const int64_t coarse_site =
        ((static_cast<int64_t>(xc) * Yc + yc) * Zc + zc) * Tc + tc;
    Complex<Real> sum(0);
    for (int E_idx = 0; E_idx < E; ++E_idx) {
      const std::size_t null_index = multigrid_null_index(
          E_idx, e_idx, x, y, z, t, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
      if (null_index >= static_cast<std::size_t>(null_count)) {
        throw std::runtime_error("multigrid prolong null-vector index overflow");
      }
      sum += multiply_complex(
          null_vectors[null_index],
          coarse_in[static_cast<int64_t>(E_idx) * coarse_volume + coarse_site]);
    }
    fine_out[output] = sum;
  }
}

template <typename Real>
void LatticeSet<Real>::validate_coarse_operator(
    const void *out, const void *in, int E, int X, int Y, int Z, int T) const {
  if (out == nullptr || in == nullptr) {
    throw std::runtime_error("coarse dslash received a null pointer");
  }
  if (E <= 0 || X <= 0 || Y <= 0 || Z <= 0 || T <= 0) {
    throw std::runtime_error(
        "coarse dslash dimensions must be positive");
  }
}

template <typename Real>
void LatticeSet<Real>::apply_multigrid_coarse_dslash(
    Complex<Real> *out, const Complex<Real> *in,
    const Complex<Real> *hopping, const Complex<Real> *sitting, int E, int X,
    int Y, int Z, int T) {
  validate_coarse_operator(out, in, E, X, Y, Z, T);
  if (hopping == nullptr || sitting == nullptr) {
    throw std::runtime_error("coarse dslash received a null stencil");
  }
  const std::array<int, kDim> dims = {X, Y, Z, T};
  exchange_coarse_halo(in, E, dims);
  const int64_t volume = coarse_volume(dims);
  const int64_t output_count = static_cast<int64_t>(E) * volume;
  const int64_t chunk = omp_chunk<Real>(output_count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t output = 0; output < output_count; ++output) {
    const int E_out = static_cast<int>(output / volume);
    const int64_t site = output % volume;
    std::array<int, kDim> coordinate{};
    decode_coarse_site(site, dims, coordinate[0], coordinate[1],
                       coordinate[2], coordinate[3]);
    Complex<Real> sum(0);
    for (int e = 0; e < E; ++e) {
      const int64_t on_site =
          (static_cast<int64_t>(E_out) * E + e) * volume + site;
      sum += multiply_complex(
          sitting[on_site], in[static_cast<int64_t>(e) * volume + site]);
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
          sum += multiply_complex(
              hopping[hop],
              coarse_neighbor(in, E, dims, coordinate, shift, e));
        }
      }
    }
    out[output] = sum;
  }
}

template <typename Real>
void LatticeSet<Real>::apply_multigrid_coarse_dslash_wide(
    Complex<Real> *out, const Complex<Real> *in,
    const Complex<Real> *sitting, const Complex<Real> *hop_nn,
    const Complex<Real> *hop_diag, int E, int X, int Y, int Z, int T) {
  validate_coarse_operator(out, in, E, X, Y, Z, T);
  if (sitting == nullptr || hop_nn == nullptr || hop_diag == nullptr) {
    throw std::runtime_error("wide coarse dslash received a null stencil");
  }
  const std::array<int, kDim> dims = {X, Y, Z, T};
  exchange_coarse_halo(in, E, dims);
  const int64_t volume = coarse_volume(dims);
  // Process a tile of consecutive sites at a time.  The coefficient arrays
  // are site-major within every stencil stream, so streaming a whole tile
  // through each stream keeps the coefficient pages hot instead of walking
  // 4608 widely separated streams once per site.  That address pattern was
  // limited by TLB/page misses and dominated legacy-MG coarse applications.
  constexpr int kCoarseTile = 512;

  struct NeighborView {
    const Complex<Real> *data = nullptr;
    int64_t base = 0;
    int64_t stride = 0;
  };
  const auto halo_direction = [](const int crossing[kDim]) {
    int dimensions[2] = {-1, -1};
    int signs[2] = {0, 0};
    int count = 0;
    for (int dim = 0; dim < kDim; ++dim) {
      if (crossing[dim] != 0) {
        if (count == 2) {
          return -1;
        }
        dimensions[count] = dim;
        signs[count] = crossing[dim];
        ++count;
      }
    }
    if (count == 0) {
      return -1;
    }
    if (count == 1) {
      return 2 * dimensions[0] + (signs[0] < 0 ? 1 : 0);
    }
    static constexpr int pair_index[kDim][kDim] = {
        {-1, 0, 1, 2}, {-1, -1, 3, 4},
        {-1, -1, -1, 5}, {-1, -1, -1, -1}};
    const int pair = pair_index[dimensions[0]][dimensions[1]];
    int combination = 0;
    if (signs[0] < 0 && signs[1] < 0) {
      combination = 1;
    } else if (signs[0] > 0 && signs[1] < 0) {
      combination = 2;
    } else if (signs[0] < 0 && signs[1] > 0) {
      combination = 3;
    }
    return 8 + pair * 4 + combination;
  };
  const auto neighbor_view = [&](const std::array<int, kDim> &coordinate,
                                 const int shift[kDim]) {
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
    const int direction = halo_direction(crossing);
    if (direction >= 0) {
      const int64_t face = coarse_face_index(direction, target, dims);
      return NeighborView{
          coarse_halo_recv_[direction].data(),
          coarse_halo_offsets_[direction] + face * E, 1};
    }
    return NeighborView{
        in,
        coarse_site_index(dims, target[0], target[1], target[2], target[3]),
        volume};
  };

  std::array<std::array<int, kDim>, 8> nearest_shift{};
  for (int dim = 0; dim < kDim; ++dim) {
    for (int sign_index = 0; sign_index < 2; ++sign_index) {
      nearest_shift[sign_index * kDim + dim][dim] =
          sign_index == 0 ? 1 : -1;
    }
  }
  std::array<std::array<int, kDim>, 24> diagonal_shift{};
  static constexpr int pair_first[6] = {0, 0, 0, 1, 1, 2};
  static constexpr int pair_second[6] = {1, 2, 3, 2, 3, 3};
  for (int pair = 0; pair < 6; ++pair) {
    for (int first_sign_index = 0; first_sign_index < 2;
         ++first_sign_index) {
      for (int second_sign_index = 0; second_sign_index < 2;
           ++second_sign_index) {
        std::array<int, kDim> shift{};
        shift[pair_first[pair]] =
            first_sign_index == 0 ? 1 : -1;
        shift[pair_second[pair]] =
            second_sign_index == 0 ? 1 : -1;
        const int combination = first_sign_index * 2 + second_sign_index;
        diagonal_shift[combination * 6 + pair] = shift;
      }
    }
  }

#pragma omp parallel for schedule(static)
  for (int64_t block = 0; block < volume; block += kCoarseTile) {
    const int count =
        static_cast<int>(std::min<int64_t>(kCoarseTile, volume - block));
    const Complex<Real> *neighbor_data[32][kCoarseTile];
    int64_t neighbor_step[32][kCoarseTile];
    std::array<int, kDim> coordinate{};
    decode_coarse_site(block, dims, coordinate[0], coordinate[1],
                       coordinate[2], coordinate[3]);
    for (int site = 0; site < count; ++site) {
      for (int direction = 0; direction < 8; ++direction) {
        const NeighborView view =
            neighbor_view(coordinate, nearest_shift[direction].data());
        neighbor_data[direction][site] = view.data + view.base;
        neighbor_step[direction][site] = view.stride;
      }
      for (int direction = 0; direction < 24; ++direction) {
        const NeighborView view =
            neighbor_view(coordinate, diagonal_shift[direction].data());
        neighbor_data[8 + direction][site] = view.data + view.base;
        neighbor_step[8 + direction][site] = view.stride;
      }
      if (++coordinate[3] == dims[3]) {
        coordinate[3] = 0;
        if (++coordinate[2] == dims[2]) {
          coordinate[2] = 0;
          if (++coordinate[1] == dims[1]) {
            coordinate[1] = 0;
            ++coordinate[0];
          }
        }
      }
    }

    for (int E_out = 0; E_out < E; ++E_out) {
      Complex<Real> sum[kCoarseTile];
      for (int site = 0; site < count; ++site) {
        sum[site] = Complex<Real>(0);
      }
      for (int e = 0; e < E; ++e) {
        const Complex<Real> *coeff =
            sitting + (static_cast<int64_t>(E_out) * E + e) * volume + block;
        const Complex<Real> *value = in + static_cast<int64_t>(e) * volume +
                                     block;
        for (int site = 0; site < count; ++site) {
          sum[site] += multiply_complex(coeff[site], value[site]);
        }
      }
      for (int direction = 0; direction < 8; ++direction) {
        for (int e = 0; e < E; ++e) {
          const Complex<Real> *coeff =
              hop_nn + (static_cast<int64_t>(direction) * E * E +
                        static_cast<int64_t>(E_out) * E + e) *
                           volume +
              block;
          for (int site = 0; site < count; ++site) {
            sum[site] += multiply_complex(
                coeff[site],
                neighbor_data[direction][site]
                              [e * neighbor_step[direction][site]]);
          }
        }
      }
      for (int direction = 0; direction < 24; ++direction) {
        for (int e = 0; e < E; ++e) {
          const Complex<Real> *coeff =
              hop_diag + (static_cast<int64_t>(direction) * E * E +
                          static_cast<int64_t>(E_out) * E + e) *
                             volume +
              block;
          for (int site = 0; site < count; ++site) {
            sum[site] += multiply_complex(
                coeff[site],
                neighbor_data[8 + direction][site]
                              [e * neighbor_step[8 + direction][site]]);
          }
        }
      }
      Complex<Real> *destination =
          out + static_cast<int64_t>(E_out) * volume + block;
      for (int site = 0; site < count; ++site) {
        destination[site] = sum[site];
      }
    }
  }
}

namespace {

// ---------------------------------------------------------------------------
// Distributed Legacy-MG local-stencil support.
//
// The single-rank builder probes every local coarse block m with the block's
// own null-vector columns and stores the resulting coefficients in the rows it
// owns.  A rank of a 4D process grid owns only a sub-box of coarse blocks, so
// the same probe also produces coefficients for rows that belong to a
// neighbour (`n = m + s` with `s` a unit coarse step).  Those rows are pushed
// to their owners once per build, and the probe needs fine-level data beyond
// the local sub-domain only where the probe fields can be non-zero:
//
//   source (the neighbour's blocked null vectors)  block m              -> 1 hop
//   even = clover_inv * D(source)                  block m +- 1 hop     -> 2 hops
//   schur = clover_odd * source - kappa^2 D(even)  block m +- 2 hops    -> 2 hops
//
// so a two-site ghost layer is enough for the null vectors and the gauge links
// (the backward link of a depth-2 ghost site is at depth 3) and a one-site
// layer for the Clover inverse.  That is a superset of every site a dot product
// with a neighbouring block can touch, so a distributed build reproduces the
// single-rank stencil value for value (the only differences are the ones the
// ghost copy itself cannot avoid, i.e. none).  Face, edge and corner ghost
// regions are all exchanged, and the MPI tags are a pure function of the rank
// offset, hence identical on every rank.
// ---------------------------------------------------------------------------
constexpr int kStencilNullGhost = 2;
constexpr int kStencilGaugeGhost = 3;
constexpr int kStencilCloverGhost = 1;
constexpr int kStencilPushGhost = 1;

int stencil_mod(int value, int extent) {
  const int wrapped = value % extent;
  return wrapped < 0 ? wrapped + extent : wrapped;
}

int stencil_floor_half(int value) {
  return value >= 0 ? value / 2 : -((1 - value) / 2);
}

// Extended fine frame: `ghost[dim] == 0` keeps the periodic wrap of the
// single-rank path (that dimension is not distributed), otherwise the
// coordinate is a ghost slot offset by `ghost[dim]`.
struct StencilFrame {
  std::array<int, kDim> local{};
  std::array<int, kDim> ghost{};
  std::array<int, kDim> extent{};
  std::array<int64_t, kDim> stride{};
  int64_t sites = 0;

  void reset(const std::array<int, kDim> &local_dims,
             const std::array<int, kDim> &ghost_depth) {
    local = local_dims;
    ghost = ghost_depth;
    sites = 1;
    for (int dim = 0; dim < kDim; ++dim) {
      extent[dim] = local[dim] + 2 * ghost[dim];
      if (local[dim] <= 0 || ghost[dim] < 0 || ghost[dim] > local[dim]) {
        throw std::runtime_error(
            "invalid distributed stencil ghost geometry");
      }
      sites *= extent[dim];
    }
    stride[kDim - 1] = 1;
    for (int dim = kDim - 2; dim >= 0; --dim) {
      stride[dim] = stride[dim + 1] * extent[dim + 1];
    }
  }

  int64_t index(int x, int y, int z, int t2) const {
    const int coordinate[kDim] = {x, y, z, t2};
    int64_t value = 0;
    for (int dim = 0; dim < kDim; ++dim) {
      int slot = coordinate[dim];
      if (ghost[dim] > 0) {
        slot += ghost[dim];
      } else {
        slot = stencil_mod(slot, local[dim]);
      }
      value += static_cast<int64_t>(slot) * stride[dim];
    }
    return value;
  }
};

// Mirrors lattice_wilson_dslash.cpp's transform_gamma operation for operation.
template <typename Real, int Direction, int GammaSign>
inline void stencil_transform_gamma(const Complex<Real> &v0,
                                    const Complex<Real> &v1,
                                    const Complex<Real> &v2,
                                    const Complex<Real> &v3,
                                    Complex<Real> &h0, Complex<Real> &h1,
                                    Complex<Real> &h2, Complex<Real> &h3) {
  using C = Complex<Real>;
  if constexpr (Direction == 0) {
    h0 = C(v0.real() + GammaSign * v3.imag(), v0.imag() - GammaSign * v3.real());
    h1 = C(v1.real() + GammaSign * v2.imag(), v1.imag() - GammaSign * v2.real());
    h2 = C(v2.real() - GammaSign * v1.imag(), v2.imag() + GammaSign * v1.real());
    h3 = C(v3.real() - GammaSign * v0.imag(), v3.imag() + GammaSign * v0.real());
  } else if constexpr (Direction == 1) {
    h0 = C(v0.real() + GammaSign * v3.real(), v0.imag() + GammaSign * v3.imag());
    h1 = C(v1.real() - GammaSign * v2.real(), v1.imag() - GammaSign * v2.imag());
    h2 = C(v2.real() - GammaSign * v1.real(), v2.imag() - GammaSign * v1.imag());
    h3 = C(v3.real() + GammaSign * v0.real(), v3.imag() + GammaSign * v0.imag());
  } else if constexpr (Direction == 2) {
    h0 = C(v0.real() + GammaSign * v2.imag(), v0.imag() - GammaSign * v2.real());
    h1 = C(v1.real() - GammaSign * v3.imag(), v1.imag() + GammaSign * v3.real());
    h2 = C(v2.real() - GammaSign * v0.imag(), v2.imag() + GammaSign * v0.real());
    h3 = C(v3.real() + GammaSign * v1.imag(), v3.imag() - GammaSign * v1.real());
  } else {
    h0 = C(v0.real() - GammaSign * v2.real(), v0.imag() - GammaSign * v2.imag());
    h1 = C(v1.real() - GammaSign * v3.real(), v1.imag() - GammaSign * v3.imag());
    h2 = C(v2.real() - GammaSign * v0.real(), v2.imag() - GammaSign * v0.imag());
    h3 = C(v3.real() - GammaSign * v1.real(), v3.imag() - GammaSign * v1.imag());
  }
}

// Mirrors lattice_wilson_dslash.cpp's accumulate_hopping: the accumulation
// order (spin, row, colour) and the real/imaginary splitting are identical, so
// a distributed build reproduces the single-rank arithmetic bit for bit.
template <typename Real, int Direction, bool Dagger, int GammaSign>
void stencil_accumulate_hop(const VectorView<Real> &source,
                            const GaugeView<Real> &gauge,
                            Complex<Real> result[kVector]) {
  using C = Complex<Real>;
  C transported[kVector];
#define KUPL_STENCIL_ACC(S, R, CO)                                          \
  {                                                                         \
    const C &e = gauge.data[static_cast<int64_t>(                           \
        Dagger ? (CO) * kColor + (R) : (R) * kColor + (CO)) *               \
        gauge.matrix_stride];                                               \
    const C &v = source[(S) * kColor + (CO)];                               \
    const Real ei = Dagger ? -e.imag() : e.imag();                          \
    r##S##R += e.real() * v.real() - ei * v.imag();                         \
    i##S##R += e.real() * v.imag() + ei * v.real();                         \
  }
  Real r00 = 0, r01 = 0, r02 = 0, r10 = 0, r11 = 0, r12 = 0;
  Real r20 = 0, r21 = 0, r22 = 0, r30 = 0, r31 = 0, r32 = 0;
  Real i00 = 0, i01 = 0, i02 = 0, i10 = 0, i11 = 0, i12 = 0;
  Real i20 = 0, i21 = 0, i22 = 0, i30 = 0, i31 = 0, i32 = 0;
  KUPL_STENCIL_ACC(0, 0, 0) KUPL_STENCIL_ACC(0, 0, 1) KUPL_STENCIL_ACC(0, 0, 2)
  KUPL_STENCIL_ACC(0, 1, 0) KUPL_STENCIL_ACC(0, 1, 1) KUPL_STENCIL_ACC(0, 1, 2)
  KUPL_STENCIL_ACC(0, 2, 0) KUPL_STENCIL_ACC(0, 2, 1) KUPL_STENCIL_ACC(0, 2, 2)
  KUPL_STENCIL_ACC(1, 0, 0) KUPL_STENCIL_ACC(1, 0, 1) KUPL_STENCIL_ACC(1, 0, 2)
  KUPL_STENCIL_ACC(1, 1, 0) KUPL_STENCIL_ACC(1, 1, 1) KUPL_STENCIL_ACC(1, 1, 2)
  KUPL_STENCIL_ACC(1, 2, 0) KUPL_STENCIL_ACC(1, 2, 1) KUPL_STENCIL_ACC(1, 2, 2)
  KUPL_STENCIL_ACC(2, 0, 0) KUPL_STENCIL_ACC(2, 0, 1) KUPL_STENCIL_ACC(2, 0, 2)
  KUPL_STENCIL_ACC(2, 1, 0) KUPL_STENCIL_ACC(2, 1, 1) KUPL_STENCIL_ACC(2, 1, 2)
  KUPL_STENCIL_ACC(2, 2, 0) KUPL_STENCIL_ACC(2, 2, 1) KUPL_STENCIL_ACC(2, 2, 2)
  KUPL_STENCIL_ACC(3, 0, 0) KUPL_STENCIL_ACC(3, 0, 1) KUPL_STENCIL_ACC(3, 0, 2)
  KUPL_STENCIL_ACC(3, 1, 0) KUPL_STENCIL_ACC(3, 1, 1) KUPL_STENCIL_ACC(3, 1, 2)
  KUPL_STENCIL_ACC(3, 2, 0) KUPL_STENCIL_ACC(3, 2, 1) KUPL_STENCIL_ACC(3, 2, 2)
#undef KUPL_STENCIL_ACC
  transported[0] = C(r00, i00);
  transported[1] = C(r01, i01);
  transported[2] = C(r02, i02);
  transported[3] = C(r10, i10);
  transported[4] = C(r11, i11);
  transported[5] = C(r12, i12);
  transported[6] = C(r20, i20);
  transported[7] = C(r21, i21);
  transported[8] = C(r22, i22);
  transported[9] = C(r30, i30);
  transported[10] = C(r31, i31);
  transported[11] = C(r32, i32);
  for (int color = 0; color < kColor; ++color) {
    const C v0 = transported[color];
    const C v1 = transported[kColor + color];
    const C v2 = transported[2 * kColor + color];
    const C v3 = transported[3 * kColor + color];
    C h0;
    C h1;
    C h2;
    C h3;
    stencil_transform_gamma<Real, Direction, GammaSign>(v0, v1, v2, v3, h0,
                                                        h1, h2, h3);
    result[color] += h0;
    result[kColor + color] += h1;
    result[2 * kColor + color] += h2;
    result[3 * kColor + color] += h3;
  }
}

template <typename Real, int Direction>
void stencil_accumulate_direction(const VectorView<Real> &forward_source,
                                  const GaugeView<Real> &forward_gauge,
                                  const VectorView<Real> &backward_source,
                                  const GaugeView<Real> &backward_gauge,
                                  Complex<Real> result[kVector]) {
  stencil_accumulate_hop<Real, Direction, false, 1>(forward_source,
                                                    forward_gauge, result);
  stencil_accumulate_hop<Real, Direction, true, -1>(backward_source,
                                                    backward_gauge, result);
}

// Rank offset enumerated by the base-3 code of the ghost direction.  The code
// is a pure function of the offset, so both sides of a message agree on the
// tag without any extra negotiation.
inline void stencil_decode_offset(int code, std::array<int, kDim> &offset) {
  int digits = code;
  for (int dim = 0; dim < kDim; ++dim) {
    offset[dim] = digits % 3 - 1;
    digits /= 3;
  }
}

// Face/edge/corner halo exchange.  `ghost_to_ghost == false` ships the local
// boundary slabs into the neighbours' ghost layers (fine fields);
// `ghost_to_ghost == true` ships ghost slots into the neighbours' opposite
// ghost slots (the coarse coefficient push).  Every rank walks the same
// base-3 offset enumeration, posts all receives before any send and waits once
// at the end, so the exchange is deadlock free for every process-grid shape,
// including grids where the +dim and the -dim neighbour are the same rank.  The
// tag of a message is a pure function of the rank offset and its mirror, hence
// identical on every rank.
template <typename Real>
void stencil_exchange_halo(const StencilFrame &frame,
                           const LatticeGeometry &geometry, int tag_base,
                           int payload, Complex<Real> *data,
                           bool ghost_to_ghost) {
  using C = Complex<Real>;
  struct Task {
    std::vector<C> send;
    std::vector<C> receive;
    std::array<int, kDim> extent{};
    std::array<int, kDim> send_lower{};
    std::array<int, kDim> receive_lower{};
  };
  std::vector<Task> tasks;
  std::vector<MPI_Request> requests;
  tasks.reserve(81);
  for (int code = 1; code < 81; ++code) {
    std::array<int, kDim> offset{};
    stencil_decode_offset(code, offset);
    bool usable = true;
    for (int dim = 0; dim < kDim; ++dim) {
      if (offset[dim] != 0 && frame.ghost[dim] == 0) {
        usable = false;
      }
    }
    if (!usable) {
      continue;
    }
    Task task;
    std::array<int, kDim> peer_coordinate{};
    int64_t count = 1;
    for (int dim = 0; dim < kDim; ++dim) {
      peer_coordinate[dim] =
          stencil_mod(geometry.coord[dim] + offset[dim] + geometry.grid[dim],
                      geometry.grid[dim]);
      task.extent[dim] =
          offset[dim] == 0 ? frame.local[dim] : frame.ghost[dim];
      count *= task.extent[dim];
      if (offset[dim] == 0) {
        task.send_lower[dim] = 0;
        task.receive_lower[dim] = 0;
      } else if (ghost_to_ghost) {
        task.send_lower[dim] =
            offset[dim] > 0 ? frame.local[dim] : -frame.ghost[dim];
        task.receive_lower[dim] =
            offset[dim] > 0 ? -frame.ghost[dim] : frame.local[dim];
      } else if (offset[dim] < 0) {
        task.send_lower[dim] = frame.local[dim] - frame.ghost[dim];
        task.receive_lower[dim] = -frame.ghost[dim];
      } else {
        task.send_lower[dim] = 0;
        task.receive_lower[dim] = frame.local[dim];
      }
    }
    if (count <= 0) {
      continue;
    }
    const int peer = geometry.cartesian_rank(peer_coordinate);
    const std::size_t elements =
        static_cast<std::size_t>(count) * static_cast<std::size_t>(payload);
    task.send.resize(elements);
    task.receive.resize(elements);
    int64_t flat = 0;
    for (int a = 0; a < task.extent[0]; ++a) {
      for (int b = 0; b < task.extent[1]; ++b) {
        for (int c = 0; c < task.extent[2]; ++c) {
          for (int d = 0; d < task.extent[3]; ++d) {
            const int x = task.send_lower[0] + a;
            const int y = task.send_lower[1] + b;
            const int z = task.send_lower[2] + c;
            const int t = task.send_lower[3] + d;
            const int64_t site = frame.index(x, y, z, t);
            std::copy_n(data + site * payload, payload,
                        task.send.data() + flat * payload);
            ++flat;
          }
        }
      }
    }
    const int bytes = static_cast<int>(elements * sizeof(C));
    MPI_Request receive_request = MPI_REQUEST_NULL;
    MPI_Irecv(task.receive.data(), bytes, MPI_BYTE, peer,
              tag_base + (80 - code), geometry.comm, &receive_request);
    requests.push_back(receive_request);
    MPI_Request send_request = MPI_REQUEST_NULL;
    MPI_Isend(task.send.data(), bytes, MPI_BYTE, peer, tag_base + code,
              geometry.comm, &send_request);
    requests.push_back(send_request);
    tasks.push_back(std::move(task));
  }
  if (!requests.empty()) {
    MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                MPI_STATUSES_IGNORE);
  }
  for (Task &task : tasks) {
    int64_t flat = 0;
    for (int a = 0; a < task.extent[0]; ++a) {
      for (int b = 0; b < task.extent[1]; ++b) {
        for (int c = 0; c < task.extent[2]; ++c) {
          for (int d = 0; d < task.extent[3]; ++d) {
            const int x = task.receive_lower[0] + a;
            const int y = task.receive_lower[1] + b;
            const int z = task.receive_lower[2] + c;
            const int t = task.receive_lower[3] + d;
            const int64_t site = frame.index(x, y, z, t);
            std::copy_n(task.receive.data() + flat * payload, payload,
                        data + site * payload);
            ++flat;
          }
        }
      }
    }
  }
}

} // namespace

template <typename Real>
void LatticeSet<Real>::build_local_stencil_serial(
    void *sitting, void *hop_nn, void *hop_diag, const void *null_vectors,
    const void *gauge, const void *clover_even, const void *clover_odd,
    const void *clover_even_inv, const void *clover_odd_inv, int E, int e,
    int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc,
    int window) {
  using C = Complex<Real>;
  if (sitting == nullptr || hop_nn == nullptr || hop_diag == nullptr ||
      null_vectors == nullptr || gauge == nullptr || clover_even == nullptr ||
      clover_odd == nullptr || clover_even_inv == nullptr ||
      clover_odd_inv == nullptr) {
    throw std::runtime_error("local stencil build received a null pointer");
  }
  if (E <= 0 || e != kVectorSize || Xf <= 0 || Yf <= 0 || Zf <= 0 ||
      Tf <= 0 || Xc <= 0 || Yc <= 0 || Zc <= 0 || Tc <= 0 ||
      Xf != geometry_.local[0] || Yf != geometry_.local[1] ||
      Zf != geometry_.local[2] || Tf != geometry_.local[3] / 2 ||
      Xf % Xc || Yf % Yc || Zf % Zc || Tf % Tc) {
    throw std::invalid_argument("invalid local stencil setup geometry");
  }
  const std::array<int, kDim> fine = {Xf, Yf, Zf, Tf};
  const std::array<int, kDim> coarse = {Xc, Yc, Zc, Tc};
  const std::array<int, kDim> block = {
      Xf / Xc, Yf / Yc, Zf / Zc, Tf / Tc};
  int required_window = 0;
  for (int dim = 0; dim < kDim; ++dim) {
    // Parity-compressed time normally leaves block[3] == 1.  The local
    // window construction remains valid for any positive block extent.
    if (block[dim] < 1) {
      throw std::invalid_argument(
          "local stencil setup requires positive block extents");
    }
    // The expanded window is periodic and is clamped to the local extent
    // below.  This keeps overlap patches with a short compressed time axis
    // valid while retaining the full-window requirement for larger axes.
    required_window = std::max(
        required_window, std::min(fine[dim], 3 * block[dim] + 2));
  }
  if (window < required_window) {
    throw std::invalid_argument(
        "local stencil window is too small: need at least " +
        std::to_string(required_window));
  }

  require_wilson_mode();
  ensure_gauge_halo(static_cast<const C *>(gauge));
  const int parity = target_parity();
  const int other = 1 - parity;
  const int64_t half_sites = geometry_.half_sites;
  const int64_t coarse_count = coarse_volume(coarse);
  const int64_t fine_vectors = geometry_.parity_vectors;
  const auto *lonv = static_cast<const C *>(null_vectors);
  const auto *odd_term = static_cast<const C *>(clover_odd);
  const std::array<const C *, 2> inverse = {
      static_cast<const C *>(clover_even_inv),
      static_cast<const C *>(clover_odd_inv)};
  auto *sit = static_cast<C *>(sitting);
  auto *hnn = static_cast<C *>(hop_nn);
  auto *hdg = static_cast<C *>(hop_diag);
  const auto env_enabled = [](const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "no") != 0 &&
           std::strcmp(value, "off") != 0;
  };
  // Parallel stencil construction is the default; set
  // KUPL_ENABLE_MG_STENCIL_PARALLEL=0/false/no/off to force the serial path.
  // Each coarse center writes a disjoint output slice, so the parallel result
  // is bitwise identical to the serial one.
  const char *parallel_text =
      std::getenv("KUPL_ENABLE_MG_STENCIL_PARALLEL");
  const bool parallel_stencil =
      parallel_text == nullptr ||
      env_enabled("KUPL_ENABLE_MG_STENCIL_PARALLEL");
  int stencil_threads = parallel_stencil ? omp_get_max_threads() : 1;
  if (const char *text = std::getenv("KUPL_MG_STENCIL_THREADS");
      parallel_stencil && text != nullptr && *text != '\0') {
    char *end = nullptr;
    const long requested = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || requested < 1) {
      throw std::invalid_argument(
          "KUPL_MG_STENCIL_THREADS must be a positive integer");
    }
    stencil_threads = std::min<int>(omp_get_max_threads(), requested);
  }
  std::vector<std::vector<C>> source_buffers(
      static_cast<std::size_t>(stencil_threads));
  std::vector<std::vector<C>> even_buffers(
      static_cast<std::size_t>(stencil_threads));
  std::vector<std::vector<C>> schur_buffers(
      static_cast<std::size_t>(stencil_threads));
  for (int thread = 0; thread < stencil_threads; ++thread) {
    source_buffers[static_cast<std::size_t>(thread)].resize(
        static_cast<std::size_t>(fine_vectors));
    even_buffers[static_cast<std::size_t>(thread)].resize(
        static_cast<std::size_t>(fine_vectors));
    schur_buffers[static_cast<std::size_t>(thread)].resize(
        static_cast<std::size_t>(fine_vectors));
    // The probe loop only writes the current coarse block and clears it again
    // at the end of each probe, so the buffers must start zeroed exactly once.
    std::fill(source_buffers[static_cast<std::size_t>(thread)].begin(),
              source_buffers[static_cast<std::size_t>(thread)].end(), C(0));
  }

  const auto multiply_onsite_site = [](const C *input, const C *matrix,
                                       int64_t site, int64_t half_sites,
                                       C result[kVector]) {
    for (int row = 0; row < kVector; ++row) {
      C sum(0);
      for (int column = 0; column < kVector; ++column) {
        sum += matrix[(static_cast<int64_t>(row) * kVector + column) *
                          half_sites +
                      site] *
               input[static_cast<int64_t>(column) * half_sites + site];
      }
      result[row] = sum;
    }
  };
  const auto apply_onsite_field_site = [&](const C *input, const C *matrix,
                                           C *output, int64_t site) {
    C result[kVector];
    multiply_onsite_site(input, matrix, site, half_sites, result);
    for (int component = 0; component < kVector; ++component) {
      output[static_cast<int64_t>(component) * half_sites + site] =
          result[component];
    }
  };
  const auto apply_onsite_vector_site = [&](const C *input, const C *matrix,
                                            int64_t site, C output[kVector]) {
    multiply_onsite_site(input, matrix, site, half_sites, output);
  };
  const auto coarse_site = [](const std::array<int, kDim> &coord,
                              const std::array<int, kDim> &dims) {
    return ((static_cast<int64_t>(coord[0]) * dims[1] + coord[1]) * dims[2] +
            coord[2]) *
               dims[3] +
           coord[3];
  };
  const auto unique_points =
      [](std::vector<std::array<int, kDim>> points) {
        std::vector<std::array<int, kDim>> unique;
        for (const auto &point : points) {
          if (std::find(unique.begin(), unique.end(), point) == unique.end()) {
            unique.push_back(point);
          }
        }
        return unique;
      };
  static constexpr int pair_first[6] = {0, 0, 0, 1, 1, 2};
  static constexpr int pair_second[6] = {1, 2, 3, 2, 3, 3};
  const auto progress_start = std::chrono::steady_clock::now();
  const int64_t progress_step = std::max<int64_t>(1, coarse_count / 20);
  std::atomic<int64_t> next_progress{0};
  std::atomic<int64_t> completed_sites{0};
  const bool progress =
      std::getenv("PYQCU_MG_BUILD_PROGRESS") != nullptr;
  // Temporary section profiler (off by default).
  const bool section_profile = env_enabled("KUPL_MG_STENCIL_PROFILE");
  int64_t stencil_limit = coarse_count;
  if (const char *text = std::getenv("KUPL_MG_STENCIL_LIMIT");
      text != nullptr && *text != '\0') {
    stencil_limit = std::min<int64_t>(coarse_count, std::atoll(text));
  }
  std::vector<std::array<double, 9>> section_seconds(
      static_cast<std::size_t>(stencil_threads));

  #pragma omp parallel for schedule(dynamic, 1) num_threads(stencil_threads)
  for (int64_t c_linear = 0; c_linear < stencil_limit; ++c_linear) {
    const int thread = omp_get_thread_num();
    std::vector<C> &source =
        source_buffers[static_cast<std::size_t>(thread)];
    std::vector<C> &even =
        even_buffers[static_cast<std::size_t>(thread)];
    std::vector<C> &schur =
        schur_buffers[static_cast<std::size_t>(thread)];
    double *section =
        section_profile
            ? section_seconds[static_cast<std::size_t>(thread)].data()
            : nullptr;
    auto section_last = std::chrono::steady_clock::now();
    const auto tick = [&](int slot) {
      if (section == nullptr) {
        return;
      }
      const auto now = std::chrono::steady_clock::now();
      section[slot] += std::chrono::duration<double>(now - section_last).count();
      section_last = now;
    };
    std::array<int, kDim> center{};
    decode_coarse_site(c_linear, coarse, center[0], center[1], center[2],
                       center[3]);
    std::vector<std::array<int, kDim>> points;
    points.push_back(center);
    for (int dim = 0; dim < kDim; ++dim) {
      for (int sign : {1, -1}) {
        auto point = center;
        point[dim] = (point[dim] + sign + coarse[dim]) % coarse[dim];
        points.push_back(point);
      }
    }
    for (int pair = 0; pair < 6; ++pair) {
      for (int sign0 : {1, -1}) {
        for (int sign1 : {1, -1}) {
          auto point = center;
          point[pair_first[pair]] =
              (point[pair_first[pair]] + sign0 + coarse[pair_first[pair]]) %
              coarse[pair_first[pair]];
          point[pair_second[pair]] =
              (point[pair_second[pair]] + sign1 + coarse[pair_second[pair]]) %
              coarse[pair_second[pair]];
          points.push_back(point);
        }
      }
    }
    points = unique_points(std::move(points));
    std::vector<std::vector<C>> coefficient(
        points.size(), std::vector<C>(static_cast<std::size_t>(E) * E));
    tick(0);

    std::array<int, kDim> expanded_start{};
    std::array<int, kDim> expanded_extent{};
    for (int dim = 0; dim < kDim; ++dim) {
      // Match the CUDA local-stencil window: spatial dimensions are padded
      // by two sites on the low side, while parity-compressed time is padded
      // by one.  The extra high-side site matters once the fine dimension is
      // larger than the minimum 3*block+2 window.
      const int low_padding = dim == 3 ? 1 : 2;
      expanded_start[dim] =
          (center[dim] * block[dim] - block[dim] - low_padding + fine[dim]) %
          fine[dim];
      // The second dslash only reads the 3*block+3 neighbourhood of the
      // center block, including the parity-compressed time axis.  Using the
      // full fine time extent here made every probe scan the whole time layer
      // (L24 1-rank: 9*9*9*32 instead of 9*9*9*6, ~5.3x redundant work).
      expanded_extent[dim] = std::min(fine[dim], 3 * block[dim] + 3);
    }

    // D(source) is non-zero only on the coarse block and its one-hop
    // neighbours, because the probe source is a delta on that block.  Build
    // that small target set once per center instead of dslash-ing the whole
    // expanded window per probe.
    std::vector<std::array<int, 4>> dslash_targets;
    dslash_targets.reserve(64);
    const auto add_target = [&](int x, int y, int z, int t) {
      for (const auto &entry : dslash_targets) {
        if (entry[0] == x && entry[1] == y && entry[2] == z && entry[3] == t) {
          return;
        }
      }
      dslash_targets.push_back({x, y, z, t});
    };
    for (int dx = 0; dx < block[0]; ++dx) {
      for (int dy = 0; dy < block[1]; ++dy) {
        for (int dz = 0; dz < block[2]; ++dz) {
          for (int dt = 0; dt < block[3]; ++dt) {
            add_target(center[0] * block[0] + dx,
                       center[1] * block[1] + dy,
                       center[2] * block[2] + dz,
                       center[3] * block[3] + dt);
          }
        }
      }
    }
    for (int dx = 0; dx < block[0]; ++dx) {
      for (int dy = 0; dy < block[1]; ++dy) {
        for (int dz = 0; dz < block[2]; ++dz) {
          for (int dt = 0; dt < block[3]; ++dt) {
            const int bx = center[0] * block[0] + dx;
            const int by = center[1] * block[1] + dy;
            const int bz = center[2] * block[2] + dz;
            const int bt = center[3] * block[3] + dt;
            add_target((bx + 1) % Xf, by, bz, bt);
            add_target((bx - 1 + Xf) % Xf, by, bz, bt);
            add_target(bx, (by + 1) % Yf, bz, bt);
            add_target(bx, (by - 1 + Yf) % Yf, bz, bt);
            add_target(bx, by, (bz + 1) % Zf, bt);
            add_target(bx, by, (bz - 1 + Zf) % Zf, bt);
            add_target(bx, by, bz, (bt + 1) % Tf);
            add_target(bx, by, bz, (bt - 1 + Tf) % Tf);
          }
        }
      }
    }
    tick(1);

    for (int probe = 0; probe < E; ++probe) {
      // The previous probe (or the previous center) already cleared its block
      // at the end of its own iteration, so the buffer is zero outside the
      // current block.  Do not clear the full fine field here: that is
      // O(fine_vectors) per probe and dominated the entire build time.
      for (int dx = 0; dx < block[0]; ++dx) {
        for (int dy = 0; dy < block[1]; ++dy) {
          for (int dz = 0; dz < block[2]; ++dz) {
            for (int dt = 0; dt < block[3]; ++dt) {
              const int x = center[0] * block[0] + dx;
              const int y = center[1] * block[1] + dy;
              const int z = center[2] * block[2] + dz;
              const int t = center[3] * block[3] + dt;
              const int64_t site = site_linear(x, y, z, t);
              for (int component = 0; component < e; ++component) {
                source[static_cast<int64_t>(component) * half_sites + site] =
                    lonv[multigrid_null_index(
                        probe, component, x, y, z, t, e, Xf, Yf, Zf, Tf, Xc,
                        Yc, Zc, Tc)];
              }
            }
          }
        }
      }

      // Clear the expanded window of `even` so stale values from the previous
      // probe/center are not read by the second dslash; only the delta-block
      // neighbourhood is written below.
      tick(2);
      for (int ox = 0; ox < expanded_extent[0]; ++ox) {
        for (int oy = 0; oy < expanded_extent[1]; ++oy) {
          for (int oz = 0; oz < expanded_extent[2]; ++oz) {
            for (int ot = 0; ot < expanded_extent[3]; ++ot) {
              const int x = (expanded_start[0] + ox) % Xf;
              const int y = (expanded_start[1] + oy) % Yf;
              const int z = (expanded_start[2] + oz) % Zf;
              const int t = (expanded_start[3] + ot) % Tf;
              const int64_t site = site_linear(x, y, z, t);
              for (int component = 0; component < kVector; ++component) {
                even[static_cast<int64_t>(component) * half_sites + site] =
                    C(0);
              }
            }
          }
        }
      }
      for (const auto &target : dslash_targets) {
        compute_dslash_site(source.data(), even.data(), target[0], target[1],
                            target[2], target[3], other);
        const int64_t site =
            site_linear(target[0], target[1], target[2], target[3]);
        apply_onsite_field_site(
            even.data(), inverse[static_cast<std::size_t>(other)], even.data(),
            site);
      }
      tick(3);

      for (const auto &point : points) {
        for (int dx = 0; dx < block[0]; ++dx) {
          for (int dy = 0; dy < block[1]; ++dy) {
            for (int dz = 0; dz < block[2]; ++dz) {
              for (int dt = 0; dt < block[3]; ++dt) {
                const int x = point[0] * block[0] + dx;
                const int y = point[1] * block[1] + dy;
                const int z = point[2] * block[2] + dz;
                const int t = point[3] * block[3] + dt;
                const int64_t site = site_linear(x, y, z, t);
                compute_dslash_site(even.data(), schur.data(), x, y, z, t,
                                    parity);
                C onsite[kVector];
                apply_onsite_vector_site(source.data(), odd_term, site,
                                         onsite);
                const Real kappa2 = kappa() * kappa();
#pragma omp simd
                for (int component = 0; component < kVector; ++component) {
                  schur[static_cast<int64_t>(component) * half_sites + site] =
                      onsite[component] -
                      kappa2 *
                          schur[static_cast<int64_t>(component) * half_sites +
                                site];
                }
              }
            }
          }
        }
      }
      tick(4);

      for (std::size_t pi = 0; pi < points.size(); ++pi) {
        const auto &point = points[pi];
        for (int out = 0; out < E; ++out) {
          C sum(0);
          for (int dx = 0; dx < block[0]; ++dx) {
            for (int dy = 0; dy < block[1]; ++dy) {
              for (int dz = 0; dz < block[2]; ++dz) {
                for (int dt = 0; dt < block[3]; ++dt) {
                  const int x = point[0] * block[0] + dx;
                  const int y = point[1] * block[1] + dy;
                  const int z = point[2] * block[2] + dz;
                  const int t = point[3] * block[3] + dt;
                  const int64_t site = site_linear(x, y, z, t);
                  for (int component = 0; component < e; ++component) {
                    const C projector = lonv[multigrid_null_index(
                        out, component, x, y, z, t, e, Xf, Yf, Zf, Tf, Xc,
                        Yc, Zc, Tc)];
                    const C value =
                        schur[static_cast<int64_t>(component) * half_sites +
                              site];
                    sum += C(projector.real() * value.real() +
                                 projector.imag() * value.imag(),
                             projector.real() * value.imag() -
                                 projector.imag() * value.real());
                  }
                }
              }
            }
          }
          coefficient[pi][static_cast<std::size_t>(out) * E + probe] = sum;
        }
      }
      tick(5);

      const auto coefficient_for = [&](const std::array<int, kDim> &point) {
        const auto iterator =
            std::find(points.begin(), points.end(), point);
        if (iterator == points.end()) {
          throw std::runtime_error("local stencil lost a neighbor point");
        }
        return static_cast<std::size_t>(iterator - points.begin());
      };
      for (int out = 0; out < E; ++out) {
        for (int in = 0; in < E; ++in) {
          sit[(static_cast<int64_t>(out) * E + in) * coarse_count +
              c_linear] =
              coefficient[coefficient_for(center)]
                         [static_cast<std::size_t>(out) * E + in];
        }
      }
      for (int dim = 0; dim < kDim; ++dim) {
        auto backward = center;
        auto forward = center;
        backward[dim] =
            (backward[dim] - 1 + coarse[dim]) % coarse[dim];
        forward[dim] = (forward[dim] + 1) % coarse[dim];
        const bool degenerate = backward == forward;
        for (int out = 0; out < E; ++out) {
          for (int in = 0; in < E; ++in) {
            const auto write = [&](int sign_index,
                                   const std::array<int, kDim> &point,
                                   Real scale) {
              const int64_t site = coarse_site(point, coarse);
              const int64_t index =
                  (((static_cast<int64_t>(sign_index) * kDim + dim) * E + out) *
                       E +
                   in) *
                      coarse_count +
                  site;
              hnn[index] =
                  scale * coefficient[coefficient_for(point)]
                                     [static_cast<std::size_t>(out) * E + in];
            };
            // Store at target=n and encode the opposite displacement, since
            // the coarse kernel reads input[n + encoded_shift].
            write(0, backward, 1);
            write(1, forward, 1);
            if (degenerate) {
              const int64_t site = coarse_site(forward, coarse);
              for (int sign_index = 0; sign_index < 2; ++sign_index) {
                const int64_t index =
                    (((static_cast<int64_t>(sign_index) * kDim + dim) * E +
                      out) *
                         E +
                     in) *
                        coarse_count +
                    site;
                hnn[index] =
                    Real(0.5) *
                    coefficient[coefficient_for(forward)]
                               [static_cast<std::size_t>(out) * E + in];
              }
            }
          }
        }
      }
      for (int pair = 0; pair < 6; ++pair) {
        for (int sign0_index = 0; sign0_index < 2; ++sign0_index) {
          const int sign0 = sign0_index == 0 ? 1 : -1;
          for (int sign1_index = 0; sign1_index < 2; ++sign1_index) {
            const int sign1 = sign1_index == 0 ? 1 : -1;
            auto point = center;
            point[pair_first[pair]] =
                (point[pair_first[pair]] - sign0 +
                 coarse[pair_first[pair]]) %
                coarse[pair_first[pair]];
            point[pair_second[pair]] =
                (point[pair_second[pair]] - sign1 +
                 coarse[pair_second[pair]]) %
                coarse[pair_second[pair]];
            int combinations = 0;
            for (int other0_index = 0; other0_index < 2; ++other0_index) {
              const int other0 = other0_index == 0 ? 1 : -1;
              for (int other1_index = 0; other1_index < 2; ++other1_index) {
                const int other1 = other1_index == 0 ? 1 : -1;
                auto other_point = center;
                other_point[pair_first[pair]] =
                    (other_point[pair_first[pair]] - other0 +
                     coarse[pair_first[pair]]) %
                    coarse[pair_first[pair]];
                other_point[pair_second[pair]] =
                    (other_point[pair_second[pair]] - other1 +
                     coarse[pair_second[pair]]) %
                    coarse[pair_second[pair]];
                combinations += other_point == point ? 1 : 0;
              }
            }
            const int64_t site = coarse_site(point, coarse);
            for (int out = 0; out < E; ++out) {
              for (int in = 0; in < E; ++in) {
                const int64_t index =
                    ((((static_cast<int64_t>(sign0_index) * 2 +
                        sign1_index) *
                           6 +
                       pair) *
                          E +
                      out) *
                         E +
                     in) *
                        coarse_count +
                    site;
                hdg[index] =
                    coefficient[coefficient_for(point)]
                               [static_cast<std::size_t>(out) * E + in] /
                    static_cast<Real>(combinations);
              }
            }
          }
        }
      }
      tick(6);

      for (int dx = 0; dx < block[0]; ++dx) {
        for (int dy = 0; dy < block[1]; ++dy) {
          for (int dz = 0; dz < block[2]; ++dz) {
            for (int dt = 0; dt < block[3]; ++dt) {
              const int x = center[0] * block[0] + dx;
              const int y = center[1] * block[1] + dy;
              const int z = center[2] * block[2] + dz;
              const int t = center[3] * block[3] + dt;
              const int64_t site = site_linear(x, y, z, t);
              for (int component = 0; component < e; ++component) {
                source[static_cast<int64_t>(component) * half_sites + site] =
                    C(0);
              }
            }
          }
        }
      }
      tick(7);
    }

    if (progress) {
      const int64_t completed =
          completed_sites.fetch_add(1, std::memory_order_relaxed) + 1;
      int64_t observed = next_progress.load(std::memory_order_relaxed);
      while (completed >= observed) {
        const int64_t next =
            std::min(coarse_count, completed + progress_step);
        if (next_progress.compare_exchange_weak(
                observed, next, std::memory_order_relaxed)) {
          const double elapsed = std::chrono::duration<double>(
              std::chrono::steady_clock::now() - progress_start).count();
          #pragma omp critical(kupl_mg_build_progress)
          {
            std::fprintf(
                stderr,
                "PYQCU::KUPL::MG_BUILD:\n "
                "stencil coarse=%lld/%lld elapsed=%.3fs window=%d\n",
                static_cast<long long>(completed),
                static_cast<long long>(coarse_count), elapsed, window);
            std::fflush(stderr);
          }
          break;
        }
      }
    }
  }

  if (section_profile) {
    static const char *labels[9] = {
        "0 points+coefficient", "1 dslash_targets", "2 source_write",
        "3 even_window_clear", "4 step1_dslash+inverse", "5 step2_dslash",
        "6 project_coefficient", "7 sit_hnn_hdg_write", "8 source_clear"};
    std::array<double, 9> total{};
    for (const auto &entry : section_seconds) {
      for (int slot = 0; slot < 9; ++slot) {
        total[static_cast<std::size_t>(slot)] +=
            entry[static_cast<std::size_t>(slot)];
      }
    }
    std::fprintf(stderr, "PYQCU::KUPL::MG_STENCIL_PROFILE limit=%lld\n",
                 static_cast<long long>(stencil_limit));
    for (int slot = 0; slot < 9; ++slot) {
      std::fprintf(stderr, "PYQCU::KUPL::MG_STENCIL_PROFILE %s = %.3f s\n",
                   labels[slot], total[static_cast<std::size_t>(slot)]);
    }
    std::fflush(stderr);
  }
}

template <typename Real>
void LatticeSet<Real>::multigrid_build_stencil_local(
    void *sitting, void *hop_nn, void *hop_diag, const void *null_vectors,
    const void *gauge, const void *clover_even, const void *clover_odd,
    const void *clover_even_inv, const void *clover_odd_inv, int E, int e,
    int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc,
    int window) {
  if (sitting == nullptr || hop_nn == nullptr || hop_diag == nullptr ||
      null_vectors == nullptr || gauge == nullptr || clover_even == nullptr ||
      clover_odd == nullptr || clover_even_inv == nullptr ||
      clover_odd_inv == nullptr) {
    throw std::runtime_error("local stencil build received a null pointer");
  }
  if (E <= 0 || E > 64 || e != kVectorSize || Xf <= 0 || Yf <= 0 || Zf <= 0 ||
      Tf <= 0 || Xc <= 0 || Yc <= 0 || Zc <= 0 || Tc <= 0 ||
      Xf != geometry_.local[0] || Yf != geometry_.local[1] ||
      Zf != geometry_.local[2] || Tf != geometry_.local[3] / 2 ||
      Xf % Xc || Yf % Yc || Zf % Zc || Tf % Tc) {
    throw std::invalid_argument("invalid local stencil setup geometry");
  }
  // A general 4D process grid is supported: every rank owns one block-aligned
  // sub-domain, the local extents must stay even (parity-compressed layout) and
  // grid_t must divide T/2 so the compressed time axis stays integral.
  int64_t grid_product = 1;
  for (int dim = 0; dim < kDim; ++dim) {
    if ((geometry_.local[dim] & 1) != 0) {
      throw std::invalid_argument(
          "local stencil setup requires even local lattice extents");
    }
    if (geometry_.grid[dim] <= 0) {
      throw std::invalid_argument(
          "local stencil setup requires a positive 4D process grid");
    }
    grid_product *= geometry_.grid[dim];
  }
  if (grid_product != geometry_.size) {
    throw std::invalid_argument(
        "local stencil setup requires a 4D process grid whose product equals "
        "the MPI size");
  }
  const int64_t global_time =
      static_cast<int64_t>(geometry_.local[3]) * geometry_.grid[3];
  if (global_time % 2 != 0 || (global_time / 2) % geometry_.grid[3] != 0) {
    throw std::invalid_argument(
        "local stencil setup requires grid_t to divide T/2");
  }
  const std::array<int, kDim> fine = {Xf, Yf, Zf, Tf};
  const std::array<int, kDim> coarse = {Xc, Yc, Zc, Tc};
  int required_window = 0;
  for (int dim = 0; dim < kDim; ++dim) {
    const int extent = fine[dim] / coarse[dim];
    if (extent < 1) {
      throw std::invalid_argument(
          "local stencil setup requires positive block extents");
    }
    required_window = std::max(
        required_window, std::min(fine[dim], 3 * extent + 2));
  }
  if (window < required_window) {
    throw std::invalid_argument(
        "local stencil window is too small: need at least " +
        std::to_string(required_window));
  }
  // KUPL_MG_STENCIL_FORCE_DISTRIBUTED=1 routes a single-rank build through the
  // distributed implementation (no ghost layers exist then), which is the
  // reference check for the shared arithmetic and index translation.
  const char *force_text = std::getenv("KUPL_MG_STENCIL_FORCE_DISTRIBUTED");
  const bool force_distributed =
      force_text != nullptr && *force_text != '\0' &&
      std::strcmp(force_text, "0") != 0;
  if (geometry_.size == 1 && !force_distributed) {
    build_local_stencil_serial(
        sitting, hop_nn, hop_diag, null_vectors, gauge, clover_even,
        clover_odd, clover_even_inv, clover_odd_inv, E, e, Xf, Yf, Zf, Tf, Xc,
        Yc, Zc, Tc, window);
    return;
  }
  build_local_stencil_distributed(
      sitting, hop_nn, hop_diag, null_vectors, gauge, clover_even, clover_odd,
      clover_even_inv, clover_odd_inv, E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc,
      window);
}

template <typename Real>
void LatticeSet<Real>::build_local_stencil_distributed(
    void *sitting, void *hop_nn, void *hop_diag, const void *null_vectors,
    const void *gauge, const void *clover_even, const void *clover_odd,
    const void *clover_even_inv, const void *clover_odd_inv, int E, int e,
    int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc,
    int window) {
  using C = Complex<Real>;
  (void)clover_even;
  (void)window;
  const std::array<int, kDim> fine = {Xf, Yf, Zf, Tf};
  const std::array<int, kDim> coarse = {Xc, Yc, Zc, Tc};
  std::array<int, kDim> block{};
  std::array<int, kDim> global_coarse{};
  std::array<int, kDim> null_ghost{};
  std::array<int, kDim> gauge_ghost{};
  std::array<int, kDim> clover_ghost{};
  std::array<int, kDim> push_ghost{};
  int64_t block_volume = 1;
  int64_t coarse_count = 1;
  for (int dim = 0; dim < kDim; ++dim) {
    block[dim] = fine[dim] / coarse[dim];
    global_coarse[dim] = coarse[dim] * geometry_.grid[dim];
    const bool distributed = geometry_.grid[dim] > 1;
    if (distributed && fine[dim] < 2) {
      throw std::runtime_error(
          "distributed local stencil needs at least two fine sites along "
          "every distributed dimension");
    }
    null_ghost[dim] =
        distributed ? std::min(kStencilNullGhost, fine[dim]) : 0;
    gauge_ghost[dim] =
        distributed ? std::min(kStencilGaugeGhost, fine[dim]) : 0;
    clover_ghost[dim] =
        distributed ? std::min(kStencilCloverGhost, fine[dim]) : 0;
    push_ghost[dim] = distributed ? kStencilPushGhost : 0;
    block_volume *= block[dim];
    coarse_count *= coarse[dim];
  }
  StencilFrame null_frame;
  null_frame.reset(fine, null_ghost);
  StencilFrame gauge_frame;
  gauge_frame.reset(fine, gauge_ghost);
  StencilFrame clover_frame;
  clover_frame.reset(fine, clover_ghost);
  StencilFrame push_frame;
  push_frame.reset(coarse, push_ghost);

  require_wilson_mode();
  ensure_gauge_halo(static_cast<const C *>(gauge));
  const int parity = target_parity();
  const int other = 1 - parity;
  const int64_t half_sites = geometry_.half_sites;
  const auto *lonv = static_cast<const C *>(null_vectors);
  const auto *odd_term = static_cast<const C *>(clover_odd);
  const std::array<const C *, 2> inverse = {
      static_cast<const C *>(clover_even_inv),
      static_cast<const C *>(clover_odd_inv)};
  auto *sit = static_cast<C *>(sitting);
  auto *hnn = static_cast<C *>(hop_nn);
  auto *hdg = static_cast<C *>(hop_diag);

  const bool stencil_trace = std::getenv("KUPL_MG_STENCIL_TRACE") != nullptr;
  const auto trace_stage = [&](const char *stage) {
    if (!stencil_trace) {
      return;
    }
    std::fprintf(stderr,
                 "PYQCU::KUPL::MG_TRACE:\n rank=%d/%d stage=%s "
                 "grid=%d,%d,%d,%d ghost=%d,%d,%d,%d\n",
                 geometry_.rank, geometry_.size, stage, geometry_.grid[0],
                 geometry_.grid[1], geometry_.grid[2], geometry_.grid[3],
                 null_ghost[0], null_ghost[1], null_ghost[2], null_ghost[3]);
    std::fflush(stderr);
  };
  trace_stage("frames");
  constexpr int gauge_payload = 2 * kGauge * kDim;
  constexpr int clover_payload = kVector * kVector;
  const int null_payload = E * e;
  const int64_t push_slot = static_cast<int64_t>(E) * E;
  const int64_t push_payload = 33 * push_slot;

  // ---- extended gauge links (both parities) ----
  std::vector<C> gauge_ext(
      static_cast<std::size_t>(gauge_frame.sites) * gauge_payload);
  {
    const int64_t chunk = omp_chunk<Real>(half_sites);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t site = 0; site < half_sites; ++site) {
      int x = 0;
      int y = 0;
      int z = 0;
      int t2 = 0;
      geometry_.decode_half(site, x, y, z, t2);
      C *destination =
          gauge_ext.data() + gauge_frame.index(x, y, z, t2) * gauge_payload;
      for (int field = 0; field < 2; ++field) {
        for (int row = 0; row < kColor; ++row) {
          for (int column = 0; column < kColor; ++column) {
            for (int direction = 0; direction < kDim; ++direction) {
              destination[(((field * kColor) + row) * kColor + column) * kDim +
                          direction] =
                  gauge_halo_.source[gauge_index(field, row, column,
                                                 direction, x, y, z, t2)];
            }
          }
        }
      }
    }
  }
  trace_stage("gauge-filled");
  stencil_exchange_halo(gauge_frame, geometry_, 8000, gauge_payload,
                        gauge_ext.data(), false);
  trace_stage("gauge-exchanged");

  // ---- extended Clover inverse of the probe (`other`) parity ----
  std::vector<C> clover_ext(
      static_cast<std::size_t>(clover_frame.sites) * clover_payload);
  {
    const C *source = inverse[static_cast<std::size_t>(other)];
    const int64_t chunk = omp_chunk<Real>(half_sites);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t site = 0; site < half_sites; ++site) {
      int x = 0;
      int y = 0;
      int z = 0;
      int t2 = 0;
      geometry_.decode_half(site, x, y, z, t2);
      C *destination =
          clover_ext.data() + clover_frame.index(x, y, z, t2) * clover_payload;
      for (int row = 0; row < kVector; ++row) {
        for (int column = 0; column < kVector; ++column) {
          destination[row * kVector + column] =
              source[(static_cast<int64_t>(row) * kVector + column) *
                         half_sites +
                     site];
        }
      }
    }
  }
  trace_stage("clover-filled");
  stencil_exchange_halo(clover_frame, geometry_, 8100, clover_payload,
                        clover_ext.data(), false);
  trace_stage("clover-exchanged");

  // ---- extended blocked null vectors ----
  std::vector<C> null_ext(
      static_cast<std::size_t>(null_frame.sites) * null_payload);
  {
    int64_t blocked = 0;
    for (int xc = 0; xc < coarse[0]; ++xc) {
      for (int dx = 0; dx < block[0]; ++dx) {
        for (int yc = 0; yc < coarse[1]; ++yc) {
          for (int dy = 0; dy < block[1]; ++dy) {
            for (int zc = 0; zc < coarse[2]; ++zc) {
              for (int dz = 0; dz < block[2]; ++dz) {
                for (int tc = 0; tc < coarse[3]; ++tc) {
                  for (int dt = 0; dt < block[3]; ++dt) {
                    const int x = xc * block[0] + dx;
                    const int y = yc * block[1] + dy;
                    const int z = zc * block[2] + dz;
                    const int t2 = tc * block[3] + dt;
                    C *destination = null_ext.data() +
                                     null_frame.index(x, y, z, t2) *
                                         null_payload;
                    for (int component = 0; component < null_payload;
                         ++component) {
                      destination[component] =
                          lonv[static_cast<int64_t>(component) * half_sites +
                               blocked];
                    }
                    ++blocked;
                  }
                }
              }
            }
          }
        }
      }
    }
    if (blocked != half_sites) {
      throw std::runtime_error("distributed stencil null-vector layout mismatch");
    }
  }
  trace_stage("null-filled");
  stencil_exchange_halo(null_frame, geometry_, 8200, null_payload,
                        null_ext.data(), false);
  trace_stage("null-exchanged");

  // ---- coarse coefficient push buffer (ghost block layer only) ----
  std::vector<C> push(
      static_cast<std::size_t>(push_frame.sites) * push_payload, C(0));

  // ---- stencil column table ----
  struct ColumnSpec {
    int kind = 0; // 0 = sitting, 1 = nearest, 2 = diagonal
    int dim0 = 0;
    int dim1 = 0;
    int sign0 = 0;
    int sign1 = 0;
    int64_t slot = -1;
    int64_t reverse_slot = -1;
    Real scale = Real(1);
  };
  static constexpr int pair_first[6] = {0, 0, 0, 1, 1, 2};
  static constexpr int pair_second[6] = {1, 2, 3, 2, 3, 3};
  std::vector<ColumnSpec> columns;
  columns.reserve(33);
  {
    ColumnSpec sitting_spec;
    sitting_spec.kind = 0;
    columns.push_back(sitting_spec);
  }
  for (int dim = 0; dim < kDim; ++dim) {
    const Real scale = global_coarse[dim] <= 2 ? Real(0.5) : Real(1);
    for (int sign_index = 0; sign_index < 2; ++sign_index) {
      ColumnSpec spec;
      spec.kind = 1;
      spec.dim0 = dim;
      spec.sign0 = sign_index == 0 ? 1 : -1;
      spec.slot = sign_index * kDim + dim;
      spec.reverse_slot = (1 - sign_index) * kDim + dim;
      spec.scale = scale;
      columns.push_back(spec);
    }
  }
  for (int pair = 0; pair < 6; ++pair) {
    const int factors[2] = {global_coarse[pair_first[pair]] <= 2 ? 2 : 1,
                            global_coarse[pair_second[pair]] <= 2 ? 2 : 1};
    const Real scale = Real(1) /
                       static_cast<Real>(factors[0] * factors[1]);
    for (int sign0_index = 0; sign0_index < 2; ++sign0_index) {
      for (int sign1_index = 0; sign1_index < 2; ++sign1_index) {
        ColumnSpec spec;
        spec.kind = 2;
        spec.dim0 = pair_first[pair];
        spec.dim1 = pair_second[pair];
        spec.sign0 = sign0_index == 0 ? 1 : -1;
        spec.sign1 = sign1_index == 0 ? 1 : -1;
        spec.slot = (sign0_index * 2 + sign1_index) * 6 + pair;
        spec.reverse_slot = ((1 - sign0_index) * 2 + (1 - sign1_index)) * 6 +
                            pair;
        spec.scale = scale;
        columns.push_back(spec);
      }
    }
  }

  // ---- per-thread probe work space ----
  const auto env_enabled = [](const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "no") != 0 &&
           std::strcmp(value, "off") != 0;
  };
  const char *parallel_text = std::getenv("KUPL_ENABLE_MG_STENCIL_PARALLEL");
  const bool parallel_stencil =
      parallel_text == nullptr ||
      env_enabled("KUPL_ENABLE_MG_STENCIL_PARALLEL");
  int stencil_threads = parallel_stencil ? omp_get_max_threads() : 1;
  if (const char *text = std::getenv("KUPL_MG_STENCIL_THREADS");
      parallel_stencil && text != nullptr && *text != '\0') {
    char *end = nullptr;
    const long requested = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || requested < 1) {
      throw std::invalid_argument(
          "KUPL_MG_STENCIL_THREADS must be a positive integer");
    }
    stencil_threads = std::min<int>(omp_get_max_threads(), requested);
  }
  std::array<int, kDim> expanded{};
  int64_t expanded_volume = 1;
  for (int dim = 0; dim < kDim; ++dim) {
    expanded[dim] = block[dim] + 2;
    expanded_volume *= expanded[dim];
  }
  std::vector<std::vector<C>> even_buffers(
      static_cast<std::size_t>(stencil_threads));
  std::vector<std::vector<C>> schur_buffers(
      static_cast<std::size_t>(stencil_threads));
  std::vector<std::vector<C>> coefficient_buffers(
      static_cast<std::size_t>(stencil_threads));
  for (int thread = 0; thread < stencil_threads; ++thread) {
    even_buffers[static_cast<std::size_t>(thread)].resize(
        static_cast<std::size_t>(expanded_volume) * e);
    schur_buffers[static_cast<std::size_t>(thread)].resize(
        static_cast<std::size_t>(block_volume) * e);
    coefficient_buffers[static_cast<std::size_t>(thread)].resize(
        static_cast<std::size_t>(E));
  }
  const Real kappa2 = kappa() * kappa();

  const auto progress = std::getenv("PYQCU_MG_BUILD_PROGRESS") != nullptr;
  const auto progress_start = std::chrono::steady_clock::now();
  const int64_t progress_step = std::max<int64_t>(1, coarse_count / 20);
  std::atomic<int64_t> next_progress{0};
  std::atomic<int64_t> completed{0};

  trace_stage("probe-start");
  #pragma omp parallel for schedule(dynamic, 1) num_threads(stencil_threads)
  for (int64_t center = 0; center < coarse_count; ++center) {
    const int thread = omp_get_thread_num();
    C *even = even_buffers[static_cast<std::size_t>(thread)].data();
    C *schur = schur_buffers[static_cast<std::size_t>(thread)].data();
    C zero_vector[kVector];
    for (int component = 0; component < kVector; ++component) {
      zero_vector[component] = C(0);
    }
    std::array<int, kDim> mc{};
    decode_coarse_site(center, coarse, mc[0], mc[1], mc[2], mc[3]);
    std::array<int, kDim> base{};
    for (int dim = 0; dim < kDim; ++dim) {
      base[dim] = mc[dim] * block[dim];
    }
    const auto inside_probe_block = [&](int x, int y, int z, int t2) {
      const int coordinate[kDim] = {x, y, z, t2};
      for (int dim = 0; dim < kDim; ++dim) {
        int value = coordinate[dim];
        if (null_ghost[dim] == 0) {
          value = stencil_mod(value, fine[dim]);
        }
        if (value < base[dim] || value >= base[dim] + block[dim]) {
          return false;
        }
      }
      return true;
    };
    const auto gauge_view = [&](const std::array<int, kDim> &point, int field,
                                int direction,
                                const std::array<int, kDim> &delta) {
      const int full_t = 2 * point[3] +
                         (((point[0] + point[1] + point[2]) & 1) ^ field);
      const int fx = point[0] + delta[0];
      const int fy = point[1] + delta[1];
      const int fz = point[2] + delta[2];
      const int ft = full_t + delta[3];
      const int moved_field = (fx + fy + fz + ft) & 1;
      const int moved_t2 = stencil_floor_half(ft);
      const int64_t site = gauge_frame.index(fx, fy, fz, moved_t2);
      return GaugeView<Real>{
          gauge_ext.data() + site * gauge_payload +
              moved_field * (kGauge * kDim) + direction,
          kDim};
    };
    std::array<int, kDim> unit{};
    const auto dslash_at =
        [&](const std::array<int, kDim> &point, int field,
            const auto &forward_source, const auto &backward_source,
            C result[kVector]) {
          VectorView<Real> forward[kDim];
          VectorView<Real> backward[kDim];
          GaugeView<Real> forward_gauge[kDim];
          GaugeView<Real> backward_gauge[kDim];
          for (int dim = 0; dim < kDim; ++dim) {
            std::array<int, kDim> plus = point;
            std::array<int, kDim> minus = point;
            std::array<int, kDim> delta{};
            plus[dim] += 1;
            minus[dim] -= 1;
            delta[dim] = -1;
            forward[dim] = forward_source(plus);
            backward[dim] = backward_source(minus);
            forward_gauge[dim] = gauge_view(point, field, dim, unit);
            backward_gauge[dim] = gauge_view(point, field, dim, delta);
          }
          for (int component = 0; component < kVector; ++component) {
            result[component] = C(0);
          }
          stencil_accumulate_direction<Real, 0>(forward[0], forward_gauge[0],
                                               backward[0], backward_gauge[0],
                                               result);
          stencil_accumulate_direction<Real, 1>(forward[1], forward_gauge[1],
                                               backward[1], backward_gauge[1],
                                               result);
          stencil_accumulate_direction<Real, 2>(forward[2], forward_gauge[2],
                                               backward[2], backward_gauge[2],
                                               result);
          stencil_accumulate_direction<Real, 3>(forward[3], forward_gauge[3],
                                               backward[3], backward_gauge[3],
                                               result);
        };
    const auto apply_matrix = [&](const C *matrix, const C input[kVector],
                                  C output[kVector]) {
      for (int row = 0; row < kVector; ++row) {
        C sum(0);
        for (int column = 0; column < kVector; ++column) {
          sum += matrix[row * kVector + column] * input[column];
        }
        output[row] = sum;
      }
    };

    for (int probe = 0; probe < E; ++probe) {
      const auto probe_source = [&](const std::array<int, kDim> &point) {
        if (!inside_probe_block(point[0], point[1], point[2], point[3])) {
          return VectorView<Real>{zero_vector, 1};
        }
        const int64_t site =
            null_frame.index(point[0], point[1], point[2], point[3]);
        return VectorView<Real>{
            null_ext.data() + site * null_payload + probe * e, 1};
      };
      // even = clover_inv[other] * D(probe) on the compact support of the
      // probe, i.e. the probe block plus one hop (the single-rank builder
      // computes exactly these sites through its sparse target list).
      for (int a = -1; a <= block[0]; ++a) {
        for (int b = -1; b <= block[1]; ++b) {
          for (int c = -1; c <= block[2]; ++c) {
            for (int d = -1; d <= block[3]; ++d) {
              std::array<int, kDim> point = {base[0] + a, base[1] + b,
                                             base[2] + c, base[3] + d};
              C result[kVector];
              dslash_at(point, other, probe_source, probe_source, result);
              const C *matrix =
                  clover_ext.data() +
                  clover_frame.index(point[0], point[1], point[2], point[3]) *
                      clover_payload;
              const int64_t box =
                  ((static_cast<int64_t>(a + 1) * expanded[1] + (b + 1)) *
                       expanded[2] +
                   (c + 1)) *
                      expanded[3] +
                  (d + 1);
              apply_matrix(matrix, result, even + box * e);
            }
          }
        }
      }
      const auto even_source = [&](const std::array<int, kDim> &point) {
        std::array<int, kDim> offset{};
        for (int dim = 0; dim < kDim; ++dim) {
          offset[dim] = point[dim] - base[dim];
          if (offset[dim] < -1 || offset[dim] > block[dim]) {
            return VectorView<Real>{zero_vector, 1};
          }
        }
        const int64_t box =
            ((static_cast<int64_t>(offset[0] + 1) * expanded[1] +
              (offset[1] + 1)) *
                 expanded[2] +
             (offset[2] + 1)) *
                expanded[3] +
            (offset[3] + 1);
        return VectorView<Real>{even + box * e, 1};
      };

      for (const ColumnSpec &spec : columns) {
        std::array<int, kDim> point = mc;
        if (spec.kind == 1) {
          point[spec.dim0] += spec.sign0;
        } else if (spec.kind == 2) {
          point[spec.dim0] += spec.sign0;
          point[spec.dim1] += spec.sign1;
        }
        std::array<int, kDim> point_coarse{};
        std::array<int, kDim> point_base{};
        bool point_local = true;
        for (int dim = 0; dim < kDim; ++dim) {
          int value = point[dim];
          if (null_ghost[dim] == 0) {
            value = stencil_mod(value, coarse[dim]);
          } else if (value < 0 || value >= coarse[dim]) {
            point_local = false;
          }
          point_coarse[dim] = value;
          point_base[dim] = value * block[dim];
        }
        std::array<int, kDim> low{};
        std::array<int, kDim> high{};
        for (int dim = 0; dim < kDim; ++dim) {
          if (null_ghost[dim] == 0) {
            // Periodically wrapped axis: the point block can be adjacent to
            // the probe block through the wrap, so keep the full block.
            low[dim] = point_base[dim];
            high[dim] = point_base[dim] + block[dim];
          } else {
            // The second dslash only reaches two sites beyond the probe block,
            // so every other site of the point block contributes an exact zero
            // to the projection and is skipped.
            low[dim] = std::max(point_base[dim], base[dim] - 2);
            high[dim] = std::min(point_base[dim] + block[dim],
                                 base[dim] + block[dim] + 2);
          }
        }
        int64_t cursor = 0;
        for (int x = low[0]; x < high[0]; ++x) {
          for (int y = low[1]; y < high[1]; ++y) {
            for (int z = low[2]; z < high[2]; ++z) {
              for (int t = low[3]; t < high[3]; ++t) {
                std::array<int, kDim> site = {x, y, z, t};
                C result[kVector];
                dslash_at(site, parity, even_source, even_source, result);
                C onsite[kVector];
                if (inside_probe_block(x, y, z, t)) {
                  int sx = x;
                  int sy = y;
                  int sz = z;
                  int st = t;
                  for (int dim = 0; dim < kDim; ++dim) {
                    if (null_ghost[dim] != 0) {
                      continue;
                    }
                    int &slot = dim == 0 ? sx : (dim == 1 ? sy
                                             : dim == 2   ? sz
                                                          : st);
                    slot = stencil_mod(slot, fine[dim]);
                  }
                  const int64_t local_site = site_linear(sx, sy, sz, st);
                  const C *matrix = odd_term + local_site;
                  C source_vector[kVector];
                  const int64_t probe_site =
                      null_frame.index(x, y, z, t) * null_payload + probe * e;
                  for (int component = 0; component < kVector; ++component) {
                    source_vector[component] =
                        null_ext[probe_site + component];
                  }
                  for (int row = 0; row < kVector; ++row) {
                    C sum(0);
                    for (int column = 0; column < kVector; ++column) {
                      sum += matrix[(static_cast<int64_t>(row) * kVector +
                                     column) *
                                        half_sites] *
                             source_vector[column];
                    }
                    onsite[row] = sum;
                  }
                } else {
                  for (int component = 0; component < kVector; ++component) {
                    onsite[component] = C(0);
                  }
                }
                for (int component = 0; component < kVector; ++component) {
                  schur[cursor * e + component] =
                      onsite[component] - kappa2 * result[component];
                }
                ++cursor;
              }
            }
          }
        }
        // Projection onto the point block's null-vector columns.
        C *coefficient = coefficient_buffers[static_cast<std::size_t>(thread)].data();
        for (int out = 0; out < E; ++out) {
          C sum(0);
          int64_t slot_index = 0;
          for (int x = low[0]; x < high[0]; ++x) {
            for (int y = low[1]; y < high[1]; ++y) {
              for (int z = low[2]; z < high[2]; ++z) {
                for (int t = low[3]; t < high[3]; ++t) {
                  const C *projector =
                      null_ext.data() +
                      null_frame.index(x, y, z, t) * null_payload + out * e;
                  for (int component = 0; component < e; ++component) {
                    const C &value = schur[slot_index * e + component];
                    const C &project = projector[component];
                    sum += C(project.real() * value.real() +
                                 project.imag() * value.imag(),
                             project.real() * value.imag() -
                                 project.imag() * value.real());
                  }
                  ++slot_index;
                }
              }
            }
          }
          coefficient[out] = sum;
        }
        if (spec.kind == 0) {
          for (int out = 0; out < E; ++out) {
            sit[(static_cast<int64_t>(out) * E + probe) * coarse_count +
                center] = coefficient[out];
          }
        } else if (point_local) {
          const int64_t row =
              ((static_cast<int64_t>(point_coarse[0]) * coarse[1] +
                point_coarse[1]) *
                   coarse[2] +
               point_coarse[2]) *
                  coarse[3] +
              point_coarse[3];
          C *destination = spec.kind == 1 ? hnn : hdg;
          for (int out = 0; out < E; ++out) {
            // The coefficient belongs to M[point, center]; the coarse kernel
            // reads in[row + shift(slot)], so row=point needs the reverse
            // displacement (center - point = -sign0).  The remote push path
            // already stores/reads with reverse_slot for the same reason.
            destination[((spec.reverse_slot * E + out) * E + probe) *
                            coarse_count +
                        row] = spec.scale * coefficient[out];
          }
        } else {
          const int64_t block_index =
              push_frame.index(point[0], point[1], point[2], point[3]);
          C *destination = push.data() + block_index * push_payload +
                           spec.reverse_slot * push_slot;
          for (int out = 0; out < E; ++out) {
            destination[out * E + probe] = spec.scale * coefficient[out];
          }
        }
      }
    }
    if (progress) {
      const int64_t done =
          completed.fetch_add(1, std::memory_order_relaxed) + 1;
      int64_t observed = next_progress.load(std::memory_order_relaxed);
      while (done >= observed) {
        const int64_t next = std::min(coarse_count, done + progress_step);
        if (next_progress.compare_exchange_weak(observed, next,
                                                std::memory_order_relaxed)) {
          const double elapsed = std::chrono::duration<double>(
              std::chrono::steady_clock::now() - progress_start).count();
          #pragma omp critical(kupl_mg_build_progress)
          {
            std::fprintf(
                stderr,
                "PYQCU::KUPL::MG_BUILD:\n "
                "stencil coarse=%lld/%lld elapsed=%.3fs ghost=%d/%d/%d "
                "grid=%d,%d,%d,%d\n",
                static_cast<long long>(done),
                static_cast<long long>(coarse_count), elapsed,
                kStencilNullGhost, kStencilGaugeGhost, kStencilCloverGhost,
                geometry_.grid[0], geometry_.grid[1], geometry_.grid[2],
                geometry_.grid[3]);
            std::fflush(stderr);
          }
          break;
        }
      }
    }
  }

  // ---- deliver the coefficients of neighbour rows ----
  trace_stage("probe-done");
  stencil_exchange_halo(push_frame, geometry_, 8300,
                        static_cast<int>(push_payload), push.data(), true);
  trace_stage("push-exchanged");
  {
    const int64_t chunk = omp_chunk<Real>(coarse_count);
#pragma omp parallel for schedule(static, chunk) num_threads(stencil_threads)
    for (int64_t row = 0; row < coarse_count; ++row) {
      std::array<int, kDim> rc{};
      decode_coarse_site(row, coarse, rc[0], rc[1], rc[2], rc[3]);
      for (const ColumnSpec &spec : columns) {
        if (spec.kind == 0) {
          continue;
        }
        std::array<int, kDim> point = rc;
        if (spec.kind == 1) {
          point[spec.dim0] += spec.sign0;
        } else {
          point[spec.dim0] += spec.sign0;
          point[spec.dim1] += spec.sign1;
        }
        bool local = true;
        for (int dim = 0; dim < kDim; ++dim) {
          if (null_ghost[dim] != 0 &&
              (point[dim] < 0 || point[dim] >= coarse[dim])) {
            local = false;
          }
        }
        if (local) {
          continue;
        }
        const int64_t block_index =
            push_frame.index(point[0], point[1], point[2], point[3]);
        const C *source = push.data() + block_index * push_payload +
                          spec.slot * push_slot;
        C *destination = spec.kind == 1 ? hnn : hdg;
        for (int out = 0; out < E; ++out) {
          for (int in = 0; in < E; ++in) {
            destination[((spec.slot * E + out) * E + in) * coarse_count +
                        row] = source[out * E + in];
          }
        }
      }
    }
  }
  trace_stage("placed");
}

template class LatticeSet<float>;
template class LatticeSet<double>;

} // namespace kupl::qcu
