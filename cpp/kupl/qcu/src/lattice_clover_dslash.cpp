#include "kupl/lattice_halo.h"
#include "kupl/lattice_linalg.h"
#include "kupl/lattice_set.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kupl::qcu {

template <typename Real>
void LatticeSet<Real>::pack_gauge_face(int dim, int boundary,
                                       Complex<Real> *buffer) {
  const int64_t count = geometry_.half_sites / surface_extent(dim);
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t face_site = 0; face_site < count; ++face_site) {
    int x = 0;
    int y = 0;
    int z = 0;
    int t2 = 0;
    decode_face(face_site, dim, boundary, x, y, z, t2);
    const int64_t target_index = face_site * 2 * kGauge * kDim;
    for (int parity = 0; parity < 2; ++parity) {
      for (int row = 0; row < kColor; ++row) {
        for (int column = 0; column < kColor; ++column) {
          for (int direction = 0; direction < kDim; ++direction) {
            const int64_t offset =
                (((parity * kColor + row) * kColor + column) * kDim +
                 direction);
            buffer[target_index + offset] =
                gauge_halo_.source[gauge_index(parity, row, column, direction,
                                               x, y, z, t2)];
          }
        }
      }
    }
  }
}

template <typename Real>
void LatticeSet<Real>::pack_gauge_corner(int dim0, int dim1, int boundary0,
                                         int boundary1,
                                         Complex<Real> *buffer) {
  const int64_t count =
      geometry_.half_sites /
      (surface_extent(dim0) * surface_extent(dim1));
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t corner_site = 0; corner_site < count; ++corner_site) {
    int x = 0;
    int y = 0;
    int z = 0;
    int t2 = 0;
    decode_corner(corner_site, dim0, boundary0, dim1, boundary1, x, y, z,
                  t2);
    const int64_t target_index = corner_site * 2 * kGauge * kDim;
    for (int parity = 0; parity < 2; ++parity) {
      for (int row = 0; row < kColor; ++row) {
        for (int column = 0; column < kColor; ++column) {
          for (int direction = 0; direction < kDim; ++direction) {
            const int64_t offset =
                (((parity * kColor + row) * kColor + column) * kDim +
                 direction);
            buffer[target_index + offset] =
                gauge_halo_.source[gauge_index(parity, row, column, direction,
                                               x, y, z, t2)];
          }
        }
      }
    }
  }
}

template <typename Real>
void LatticeSet<Real>::ensure_gauge_halo(const Complex<Real> *gauge) {
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
    auto &send_lower = gauge_halo_.faces[face_offset(dim, -1)];
    auto &send_upper = gauge_halo_.faces[face_offset(dim, 1)];
    pack_gauge_face(dim, 0, send_lower.data());
    pack_gauge_face(dim, geometry_.last_half_coordinate(dim),
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
              gauge_face_tag(dim, -1), geometry_.comm, &request);
    post(request);
    MPI_Irecv(recv_minus_storage.data(),
              static_cast<int>(recv_minus_storage.size() *
                               sizeof(Complex<Real>)),
              MPI_BYTE, geometry_.neighbor_minus[dim],
              gauge_face_tag(dim, 1), geometry_.comm, &request);
    post(request);
    MPI_Isend(send_upper.data(),
              static_cast<int>(send_upper.size() * sizeof(Complex<Real>)),
              MPI_BYTE, geometry_.neighbor_plus[dim],
              gauge_face_tag(dim, 1), geometry_.comm, &request);
    post(request);
    MPI_Isend(send_lower.data(),
              static_cast<int>(send_lower.size() * sizeof(Complex<Real>)),
              MPI_BYTE, geometry_.neighbor_minus[dim],
              gauge_face_tag(dim, -1), geometry_.comm, &request);
    post(request);

    // Wait immediately so the two temporary receive buffers can be moved
    // into the persistent face cache without increasing peak memory.
    if (!requests.empty()) {
      MPI_Waitall(static_cast<int>(requests.size()), requests.data(),
                  MPI_STATUSES_IGNORE);
      requests.clear();
    }
    gauge_halo_.faces[face_offset(dim, 1)] = std::move(recv_plus_storage);
    gauge_halo_.faces[face_offset(dim, -1)] = std::move(recv_minus_storage);
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
              sign0 > 0 ? geometry_.last_half_coordinate(dim0) : 0;
          const int boundary1 =
              sign1 > 0 ? geometry_.last_half_coordinate(dim1) : 0;
          auto &send =
              gauge_halo_
                  .corners[static_cast<std::size_t>(
                      plane_index(dim0, dim1))]
                  [static_cast<std::size_t>(sign_index(sign0) * 2 +
                                            sign_index(sign1))];
          pack_gauge_corner(dim0, dim1, boundary0, boundary1, send.data());
          const int plane = plane_index(dim0, dim1);
          const int slot = sign_index(sign0) * 2 + sign_index(sign1);
          auto &recv = corner_recv_storage[static_cast<std::size_t>(plane)]
                                          [static_cast<std::size_t>(slot)];
          recv.resize(send.size());
          MPI_Request request = MPI_REQUEST_NULL;
          MPI_Irecv(recv.data(),
                    static_cast<int>(recv.size() * sizeof(Complex<Real>)),
                    MPI_BYTE, diagonal,
                    corner_tag(dim0, dim1, -sign0, -sign1), geometry_.comm,
                    &request);
          corner_requests.push_back(request);
          MPI_Isend(send.data(),
                    static_cast<int>(send.size() * sizeof(Complex<Real>)),
                    MPI_BYTE, diagonal,
                    corner_tag(dim0, dim1, sign0, sign1), geometry_.comm,
                    &request);
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
      const int plane = plane_index(dim0, dim1);
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

template <typename Real>
GaugeView<Real> LatticeSet<Real>::gauge_at_delta(
    int x, int y, int z, int t2, int parity, int direction,
    const int delta[kDim]) const {
  int full_t = geometry_.time_for_parity(x, y, z, t2, parity);
  int full[kDim] = {x, y, z, full_t};
  bool remote[kDim] = {false, false, false, false};
  int offsets[kDim] = {delta[0], delta[1], delta[2], delta[3]};
  for (int dim = 0; dim < kDim; ++dim) {
    if (offsets[dim] == 0) {
      continue;
    }
    full[dim] += offsets[dim];
    if (full[dim] < 0) {
      full[dim] = geometry_.local[dim] - 1;
      remote[dim] = geometry_.grid[dim] > 1;
    } else if (full[dim] >= geometry_.local[dim]) {
      full[dim] = 0;
      remote[dim] = geometry_.grid[dim] > 1;
    }
  }
  const int moved_parity =
      geometry_.parity_at(full[0], full[1], full[2], full[3]);
  const int moved_t2 = full[3] / 2;
  const int remote_dims =
      static_cast<int>(remote[0]) + static_cast<int>(remote[1]) +
      static_cast<int>(remote[2]) + static_cast<int>(remote[3]);
  if (remote_dims == 0) {
    return local_gauge(moved_parity, direction, full[0], full[1], full[2],
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
    const auto &face =
        gauge_halo_.faces[face_offset(remote_dim0, sign0)];
    const int64_t index =
        geometry_.face_index(remote_dim0, full[0], full[1], full[2],
                             moved_t2);
    const int64_t offset = index * 2 * kGauge * kDim + direction;
    const int64_t parity_base =
        static_cast<int64_t>(moved_parity) * kGauge * kDim;
    return {face.data() + parity_base + offset, kDim};
  }
  if (remote_dim0 > remote_dim1) {
    std::swap(remote_dim0, remote_dim1);
    std::swap(sign0, sign1);
  }
  const int plane = plane_index(remote_dim0, remote_dim1);
  const int slot = sign_index(sign0) * 2 + sign_index(sign1);
  const auto &corner =
      gauge_halo_.corners[static_cast<std::size_t>(plane)]
                         [static_cast<std::size_t>(slot)];
  const int64_t index =
      geometry_.corner_index(remote_dim0, remote_dim1, full[0], full[1],
                             full[2], moved_t2);
  const int64_t offset = index * 2 * kGauge * kDim + direction;
  const int64_t parity_base =
      static_cast<int64_t>(moved_parity) * kGauge * kDim;
  return {corner.data() + parity_base + offset, kDim};
}

template <typename Real>
GaugeView<Real> LatticeSet<Real>::backward_gauge(int x, int y, int z, int t2,
                                                 int direction,
                                                 int target) const {
  int delta[kDim] = {0, 0, 0, 0};
  delta[direction] = -1;
  return gauge_at_delta(x, y, z, t2, target, direction, delta);
}

template <typename Real>
void LatticeSet<Real>::compute_plaquette_sum(
    int x, int y, int z, int t2, int mu, int nu, int parity,
    Complex<Real> sum[9]) const {
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
    load_gauge_matrix(
        gauge_at_delta(x, y, z, t2, parity, direction, delta), output);
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
    multiply_su3(m0, m1, temp);
    multiply_su3(temp, m2, product);
    multiply_su3(product, m3, temp);
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
  load_term(nu, mu == 0 ? -1 : 0, mu == 1 ? -1 : 0, mu == 2 ? -1 : 0,
            mu == 3 ? -1 : 0, true, m2);
  load_term(mu, mu == 0 ? -1 : 0, mu == 1 ? -1 : 0, mu == 2 ? -1 : 0,
            mu == 3 ? -1 : 0, false, m3);
  add_product();

  // term3 = U_mu(x-mu)^dag U_nu(x-mu-nu)^dag
  //         U_mu(x-mu-nu) U_nu(x-nu)
  load_term(mu, mu == 0 ? -1 : 0, mu == 1 ? -1 : 0, mu == 2 ? -1 : 0,
            mu == 3 ? -1 : 0, true, m0);
  load_term(nu, (mu == 0 ? -1 : 0) + (nu == 0 ? -1 : 0),
            (mu == 1 ? -1 : 0) + (nu == 1 ? -1 : 0),
            (mu == 2 ? -1 : 0) + (nu == 2 ? -1 : 0),
            (mu == 3 ? -1 : 0) + (nu == 3 ? -1 : 0), true, m1);
  load_term(mu, (mu == 0 ? -1 : 0) + (nu == 0 ? -1 : 0),
            (mu == 1 ? -1 : 0) + (nu == 1 ? -1 : 0),
            (mu == 2 ? -1 : 0) + (nu == 2 ? -1 : 0),
            (mu == 3 ? -1 : 0) + (nu == 3 ? -1 : 0), false, m2);
  load_term(nu, nu == 0 ? -1 : 0, nu == 1 ? -1 : 0, nu == 2 ? -1 : 0,
            nu == 3 ? -1 : 0, false, m3);
  add_product();

  // term4 = U_nu(x-nu)^dag U_mu(x-nu)
  //         U_nu(x-nu+mu) U_mu(x)^dag
  load_term(nu, nu == 0 ? -1 : 0, nu == 1 ? -1 : 0, nu == 2 ? -1 : 0,
            nu == 3 ? -1 : 0, true, m0);
  load_term(mu, nu == 0 ? -1 : 0, nu == 1 ? -1 : 0, nu == 2 ? -1 : 0,
            nu == 3 ? -1 : 0, false, m1);
  load_term(nu, (nu == 0 ? -1 : 0) + (mu == 0 ? 1 : 0),
            (nu == 1 ? -1 : 0) + (mu == 1 ? 1 : 0),
            (nu == 2 ? -1 : 0) + (mu == 2 ? 1 : 0),
            (nu == 3 ? -1 : 0) + (mu == 3 ? 1 : 0), false, m2);
  load_term(mu, 0, 0, 0, 0, true, m3);
  add_product();
}

template <typename Real>
void LatticeSet<Real>::compute_clover(Complex<Real> *output, int parity) {
  using C = Complex<Real>;
  const Real coefficient = -Real(0.125) * kappa() * csw();
  const int64_t count = geometry_.half_sites;
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t linear = 0; linear < count; ++linear) {
    int x = 0;
    int y = 0;
    int z = 0;
    int t2 = 0;
    geometry_.decode_half(linear, x, y, z, t2);
    C matrix[kOnsite];
    std::fill(std::begin(matrix), std::end(matrix), C(0));
    for (int component = 0; component < kVector; ++component) {
      matrix[component * kVector + component] = C(1);
    }
    for (int mu = 0; mu < kDim; ++mu) {
      for (int nu = mu + 1; nu < kDim; ++nu) {
        C plaquette[9];
        compute_plaquette_sum(x, y, z, t2, mu, nu, parity, plaquette);
        C anti_hermitian[9];
        for (int row = 0; row < kColor; ++row) {
          for (int column = 0; column < kColor; ++column) {
            anti_hermitian[row * kColor + column] =
                plaquette[row * kColor + column] -
                std::conj(plaquette[column * kColor + row]);
          }
        }
        C sigma[kSpin * kSpin];
        for (int spin_out = 0; spin_out < kSpin; ++spin_out) {
          for (int spin_in = 0; spin_in < kSpin; ++spin_in) {
            C value(0);
            for (int inner = 0; inner < kSpin; ++inner) {
              value += gamma_entry<Real>(mu, spin_out, inner) *
                       gamma_entry<Real>(nu, inner, spin_in);
            }
            sigma[spin_out * kSpin + spin_in] = value;
          }
        }
        for (int spin_out = 0; spin_out < kSpin; ++spin_out) {
          for (int color_out = 0; color_out < kColor; ++color_out) {
            const int row = spin_out * kColor + color_out;
            for (int spin_in = 0; spin_in < kSpin; ++spin_in) {
              const C spin_factor = sigma[spin_out * kSpin + spin_in];
              if (std::abs(spin_factor) == Real(0)) {
                continue;
              }
              for (int color_in = 0; color_in < kColor; ++color_in) {
                const int column = spin_in * kColor + color_in;
                matrix[row * kVector + column] +=
                    coefficient * spin_factor *
                    anti_hermitian[color_out * kColor + color_in];
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

template <typename Real>
void LatticeSet<Real>::invert_clover(const Complex<Real> *term,
                                     Complex<Real> *inverse) {
  const int64_t count = geometry_.half_sites;
  const int64_t chunk = omp_chunk<Real>(count);
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
    const Real tolerance = std::numeric_limits<Real>::epsilon() * Real(256);
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
        singular_site.compare_exchange_strong(expected, site,
                                              std::memory_order_relaxed);
        break;
      }
      if (pivot != column) {
        for (int entry = 0; entry < 2 * kVector; ++entry) {
          std::swap(augmented[pivot][entry], augmented[column][entry]);
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
          augmented[row][entry] -= factor * augmented[column][entry];
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
                site] = augmented[row][kVector + column];
      }
    }
  }
  if (singular.load(std::memory_order_relaxed)) {
    const int64_t site = singular_site.load(std::memory_order_relaxed);
    const Complex<Real> diagonal = term[static_cast<std::size_t>(site)];
    throw std::runtime_error(
        "singular clover onsite matrix at local half-site " +
        std::to_string(site) + " diagonal=(" +
        std::to_string(static_cast<double>(diagonal.real())) + "," +
        std::to_string(static_cast<double>(diagonal.imag())) + ")");
  }
}

template <typename Real>
void LatticeSet<Real>::ensure_cached_clover(const Complex<Real> *gauge) {
  if (cached_gauge_ == gauge) {
    return;
  }
  ensure_gauge_halo(gauge);
  for (int parity = 0; parity < 2; ++parity) {
    auto &term = cached_clover_[static_cast<std::size_t>(parity)];
    auto &inverse = cached_clover_inv_[static_cast<std::size_t>(parity)];
    compute_clover(term.data(), parity);
    invert_clover(term.data(), inverse.data());
  }
  cached_gauge_ = gauge;
}

template class LatticeSet<float>;
template class LatticeSet<double>;

} // namespace kupl::qcu
