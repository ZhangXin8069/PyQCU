#include "kupl/lattice_halo.h"
#include "kupl/lattice_linalg.h"
#include "kupl/lattice_set.h"

#include <algorithm>
#include <array>
#include <vector>

namespace kupl::qcu {

template <typename Real>
void LatticeSet<Real>::pack_fermion_face(const Complex<Real> *source,
                                         int parity, int dim, int boundary,
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
    const int64_t target = face_site * kVector;
    const int64_t site = site_linear(x, y, z, t2);
    for (int component = 0; component < kVector; ++component) {
      buffer[target + component] =
          source[static_cast<int64_t>(component) * geometry_.half_sites +
                 site];
    }
  }
  (void)parity;
}

template <typename Real>
void LatticeSet<Real>::post_fermion_halo(const Complex<Real> *source,
                                         int parity) {
  fermion_requests_.clear();
  fermion_requests_.reserve(16);
  for (int dim = 0; dim < kDim; ++dim) {
    if (geometry_.grid[dim] == 1) {
      continue;
    }
    auto &send_minus = fermion_send_[face_offset(dim, -1)];
    auto &send_plus = fermion_send_[face_offset(dim, 1)];
    auto &recv_minus = fermion_recv_[face_offset(dim, -1)];
    auto &recv_plus = fermion_recv_[face_offset(dim, 1)];
    pack_fermion_face(source, parity, dim, 0, send_minus.data());
    pack_fermion_face(source, parity, dim,
                      geometry_.last_half_coordinate(dim),
                      send_plus.data());
    MPI_Request request = MPI_REQUEST_NULL;
    MPI_Irecv(recv_plus.data(),
              static_cast<int>(recv_plus.size() * sizeof(Complex<Real>)),
              MPI_BYTE, geometry_.neighbor_plus[dim],
              fermion_face_tag(dim, -1), geometry_.comm, &request);
    fermion_requests_.push_back(request);
    MPI_Irecv(recv_minus.data(),
              static_cast<int>(recv_minus.size() * sizeof(Complex<Real>)),
              MPI_BYTE, geometry_.neighbor_minus[dim],
              fermion_face_tag(dim, 1), geometry_.comm, &request);
    fermion_requests_.push_back(request);
    MPI_Isend(send_plus.data(),
              static_cast<int>(send_plus.size() * sizeof(Complex<Real>)),
              MPI_BYTE, geometry_.neighbor_plus[dim],
              fermion_face_tag(dim, 1), geometry_.comm, &request);
    fermion_requests_.push_back(request);
    MPI_Isend(send_minus.data(),
              static_cast<int>(send_minus.size() * sizeof(Complex<Real>)),
              MPI_BYTE, geometry_.neighbor_minus[dim],
              fermion_face_tag(dim, -1), geometry_.comm, &request);
    fermion_requests_.push_back(request);
  }
}

template <typename Real>
VectorView<Real> LatticeSet<Real>::source_at_delta(
    const Complex<Real> *local_source, int x, int y, int z, int t2, int dim,
    int sign, int target) const {
  int coordinate[kDim] = {x, y, z, t2};
  bool remote = false;
  if (dim == 3) {
    const int target_time =
        geometry_.time_for_parity(x, y, z, t2, target);
    int moved_time = target_time + sign;
    if (moved_time < 0 || moved_time >= geometry_.local[3]) {
      if (geometry_.grid[3] == 1) {
        moved_time =
            moved_time < 0 ? geometry_.local[3] - 1 : 0;
      } else {
        remote = true;
      }
    }
    if (!remote) {
      coordinate[3] = moved_time / 2;
    }
  } else {
    coordinate[dim] += sign;
    if (coordinate[dim] < 0 ||
        coordinate[dim] >= geometry_.local[dim]) {
      if (geometry_.grid[dim] == 1) {
        coordinate[dim] =
            coordinate[dim] < 0 ? geometry_.local[dim] - 1 : 0;
      } else {
        remote = true;
      }
    }
  }
  if (!remote) {
    return compact_view(
        local_source,
        site_linear(coordinate[0], coordinate[1], coordinate[2],
                    coordinate[3]));
  }
  const auto &halo = fermion_recv_[face_offset(dim, sign)];
  const int64_t index = geometry_.face_index(dim, x, y, z, t2);
  return halo_view(halo, index);
}

template <typename Real>
void LatticeSet<Real>::compute_dslash_interior(const Complex<Real> *source,
                                               Complex<Real> *output,
                                               int target) {
  const int64_t count = geometry_.half_sites;
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t linear = 0; linear < count; ++linear) {
    int x = 0;
    int y = 0;
    int z = 0;
    int t2 = 0;
    geometry_.decode_half(linear, x, y, z, t2);
    if (x == 0 || y == 0 || z == 0 || t2 == 0 ||
        x == geometry_.local[0] - 1 ||
        y == geometry_.local[1] - 1 ||
        z == geometry_.local[2] - 1 ||
        t2 == geometry_.local[3] / 2 - 1) {
      continue;
    }
    compute_dslash_site(source, output, x, y, z, t2, target);
  }
}

template <typename Real>
void LatticeSet<Real>::compute_dslash_boundary(
    const Complex<Real> *local_source, Complex<Real> *output, int target) {
  const int64_t count = geometry_.half_sites;
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t linear = 0; linear < count; ++linear) {
    int x = 0;
    int y = 0;
    int z = 0;
    int t2 = 0;
    geometry_.decode_half(linear, x, y, z, t2);
    const bool boundary =
        x == 0 || y == 0 || z == 0 || t2 == 0 ||
        x == geometry_.local[0] - 1 ||
        y == geometry_.local[1] - 1 ||
        z == geometry_.local[2] - 1 ||
        t2 == geometry_.local[3] / 2 - 1;
    if (!boundary) {
      continue;
    }
    compute_dslash_site(local_source, output, x, y, z, t2, target);
  }
}

template <typename Real>
void LatticeSet<Real>::compute_dslash_site(
    const Complex<Real> *local_source, Complex<Real> *output, int x, int y,
    int z, int t2, int target) {
  using C = Complex<Real>;
  C result[kVector];
  std::fill(std::begin(result), std::end(result), C(0));
  for (int direction = 0; direction < kDim; ++direction) {
    const VectorView<Real> forward_view =
        source_at_delta(local_source, x, y, z, t2, direction, 1, target);
    C forward_source[kVector];
    gather_vector(forward_view, forward_source);
    C gauge_matrix[9];
    load_gauge_matrix(local_gauge(target, direction, x, y, z, t2),
                      gauge_matrix);
    C gauge_source[kVector];
    C gamma_source[kVector];
    multiply_gauge_spin(gauge_matrix, forward_source, gauge_source, false);
    apply_one_minus_gamma(direction, gauge_source, gamma_source, 1);
    for (int index = 0; index < kVector; ++index) {
      result[index] += gamma_source[index];
    }

    const VectorView<Real> backward_view =
        source_at_delta(local_source, x, y, z, t2, direction, -1, target);
    C backward_source[kVector];
    gather_vector(backward_view, backward_source);
    C backward_gauge_matrix[9];
    load_gauge_matrix(
        backward_gauge(x, y, z, t2, direction, target),
        backward_gauge_matrix);
    multiply_gauge_spin(backward_gauge_matrix, backward_source, gauge_source,
                        true);
    apply_one_minus_gamma(direction, gauge_source, gamma_source, -1);
    for (int index = 0; index < kVector; ++index) {
      result[index] += gamma_source[index];
    }
  }
  scatter_vector(result, output, site_linear(x, y, z, t2));
}

template <typename Real>
void LatticeSet<Real>::compute_dslash_dagger_interior(
    const Complex<Real> *source, Complex<Real> *output, int target) {
  const int64_t count = geometry_.half_sites;
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t linear = 0; linear < count; ++linear) {
    int x = 0;
    int y = 0;
    int z = 0;
    int t2 = 0;
    geometry_.decode_half(linear, x, y, z, t2);
    if (x == 0 || y == 0 || z == 0 || t2 == 0 ||
        x == geometry_.local[0] - 1 ||
        y == geometry_.local[1] - 1 ||
        z == geometry_.local[2] - 1 ||
        t2 == geometry_.local[3] / 2 - 1) {
      continue;
    }
    compute_dslash_dagger_site(source, output, x, y, z, t2, target);
  }
}

template <typename Real>
void LatticeSet<Real>::compute_dslash_dagger_boundary(
    const Complex<Real> *local_source, Complex<Real> *output, int target) {
  const int64_t count = geometry_.half_sites;
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t linear = 0; linear < count; ++linear) {
    int x = 0;
    int y = 0;
    int z = 0;
    int t2 = 0;
    geometry_.decode_half(linear, x, y, z, t2);
    const bool boundary =
        x == 0 || y == 0 || z == 0 || t2 == 0 ||
        x == geometry_.local[0] - 1 ||
        y == geometry_.local[1] - 1 ||
        z == geometry_.local[2] - 1 ||
        t2 == geometry_.local[3] / 2 - 1;
    if (!boundary) {
      continue;
    }
    compute_dslash_dagger_site(local_source, output, x, y, z, t2, target);
  }
}

template <typename Real>
void LatticeSet<Real>::compute_dslash_dagger_site(
    const Complex<Real> *local_source, Complex<Real> *output, int x, int y,
    int z, int t2, int target) {
  using C = Complex<Real>;
  C result[kVector];
  std::fill(std::begin(result), std::end(result), C(0));
  for (int direction = 0; direction < kDim; ++direction) {
    C transported[kVector];
    C gamma_source[kVector];

    // Adjoint of the forward hop: U_mu(x-mu)^dagger (1-gamma_mu).
    const VectorView<Real> previous_view =
        source_at_delta(local_source, x, y, z, t2, direction, -1, target);
    C previous_source[kVector];
    gather_vector(previous_view, previous_source);
    C previous_gauge[9];
    load_gauge_matrix(backward_gauge(x, y, z, t2, direction, target),
                      previous_gauge);
    multiply_gauge_spin(previous_gauge, previous_source, transported, true);
    apply_one_minus_gamma(direction, transported, gamma_source, 1);
    for (int index = 0; index < kVector; ++index) {
      result[index] += gamma_source[index];
    }

    // Adjoint of the backward hop: U_mu(x) (1+gamma_mu).
    const VectorView<Real> next_view =
        source_at_delta(local_source, x, y, z, t2, direction, 1, target);
    C next_source[kVector];
    gather_vector(next_view, next_source);
    C current_gauge[9];
    load_gauge_matrix(local_gauge(target, direction, x, y, z, t2),
                      current_gauge);
    multiply_gauge_spin(current_gauge, next_source, transported, false);
    apply_one_minus_gamma(direction, transported, gamma_source, -1);
    for (int index = 0; index < kVector; ++index) {
      result[index] += gamma_source[index];
    }
  }
  scatter_vector(result, output, site_linear(x, y, z, t2));
}

template <typename Real>
void LatticeSet<Real>::bare_dslash(Complex<Real> *output,
                                   const Complex<Real> *input, int target) {
  const int source_parity = 1 - target;
  const int64_t count = geometry_.parity_vectors;
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < count; ++index) {
    output[index] = Complex<Real>(0);
  }
  post_fermion_halo(input, source_parity);
  compute_dslash_interior(input, output, target);
  if (!fermion_requests_.empty()) {
    MPI_Waitall(static_cast<int>(fermion_requests_.size()),
                fermion_requests_.data(), MPI_STATUSES_IGNORE);
    fermion_requests_.clear();
  }
  compute_dslash_boundary(input, output, target);
}

template <typename Real>
void LatticeSet<Real>::bare_dslash_dagger(Complex<Real> *output,
                                          const Complex<Real> *input,
                                          int target) {
  const int source_parity = 1 - target;
  const int64_t count = geometry_.parity_vectors;
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < count; ++index) {
    output[index] = Complex<Real>(0);
  }
  post_fermion_halo(input, source_parity);
  compute_dslash_dagger_interior(input, output, target);
  if (!fermion_requests_.empty()) {
    MPI_Waitall(static_cast<int>(fermion_requests_.size()),
                fermion_requests_.data(), MPI_STATUSES_IGNORE);
    fermion_requests_.clear();
  }
  compute_dslash_dagger_boundary(input, output, target);
}

template <typename Real>
void LatticeSet<Real>::apply_schur(
    const Complex<Real> *input, Complex<Real> *output, int parity,
    const std::array<const Complex<Real> *, 2> &clover,
    const std::array<const Complex<Real> *, 2> &inverse,
    Complex<Real> *scratch_a, Complex<Real> *scratch_b) {
  const int other = 1 - parity;
  bare_dslash(scratch_a, input, other);
  apply_onsite(scratch_a, inverse[static_cast<std::size_t>(other)],
               scratch_b);
  bare_dslash(scratch_a, scratch_b, parity);
  const Real k = kappa();
  const int64_t vectors = geometry_.parity_vectors;
  const int64_t chunk = omp_chunk<Real>(vectors);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < vectors; ++index) {
    scratch_a[index] *= k * k;
  }
  apply_onsite(input, clover[static_cast<std::size_t>(parity)], output);
  for (int64_t index = 0; index < vectors; ++index) {
    output[index] -= scratch_a[index];
  }
}

template <typename Real>
void LatticeSet<Real>::apply_wilson_schur(const Complex<Real> *input,
                                          Complex<Real> *output, int parity,
                                          Complex<Real> *scratch_a,
                                          Complex<Real> *scratch_b) {
  const int other = 1 - parity;
  bare_dslash(scratch_a, input, other);
  bare_dslash(scratch_b, scratch_a, parity);
  const Real coefficient = kappa() * kappa();
  const int64_t vectors = geometry_.parity_vectors;
  const int64_t chunk = omp_chunk<Real>(vectors);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < vectors; ++index) {
    output[index] = input[index] - coefficient * scratch_b[index];
  }
}

template <typename Real>
void LatticeSet<Real>::apply_wilson_schur_dagger(
    const Complex<Real> *input, Complex<Real> *output, int parity,
    Complex<Real> *scratch_a, Complex<Real> *scratch_b) {
  const int other = 1 - parity;
  bare_dslash_dagger(scratch_a, input, other);
  bare_dslash_dagger(scratch_b, scratch_a, parity);
  const Real coefficient = kappa() * kappa();
  const int64_t vectors = geometry_.parity_vectors;
  const int64_t chunk = omp_chunk<Real>(vectors);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < vectors; ++index) {
    output[index] = input[index] - coefficient * scratch_b[index];
  }
}

template <typename Real>
void LatticeSet<Real>::apply_wilson_cg_dslash(
    const Complex<Real> *input, Complex<Real> *output, int parity) {
  apply_wilson_schur(input, work_a_.data(), parity, work_b_.data(),
                     work_c_.data());
  apply_wilson_schur_dagger(work_a_.data(), output, parity, work_b_.data(),
                            work_c_.data());
}

template <typename Real>
void LatticeSet<Real>::apply_full_wilson_operator(
    const Complex<Real> *input, Complex<Real> *output,
    Complex<Real> *scratch) {
  const int64_t vectors = geometry_.parity_vectors;
  const int64_t chunk = omp_chunk<Real>(vectors);
  for (int parity = 0; parity < 2; ++parity) {
    const int other = 1 - parity;
    const Complex<Real> *source =
        input + static_cast<std::size_t>(other) * vectors;
    Complex<Real> *destination =
        output + static_cast<std::size_t>(parity) * vectors;
    bare_dslash(scratch, source, parity);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < vectors; ++index) {
      destination[index] =
          input[static_cast<std::size_t>(parity) * vectors + index] -
          kappa() * scratch[index];
    }
  }
}

template class LatticeSet<float>;
template class LatticeSet<double>;

} // namespace kupl::qcu
