#include "kupl/lattice_multigrid.h"
#include "kupl/lattice_linalg.h"
#include "kupl/lattice_set.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

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
  int peer = geometry_.rank;
  MPI_Cart_rank(geometry_.comm, coordinate.data(), &peer);
  return peer;
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

template class LatticeSet<float>;
template class LatticeSet<double>;

} // namespace kupl::qcu
