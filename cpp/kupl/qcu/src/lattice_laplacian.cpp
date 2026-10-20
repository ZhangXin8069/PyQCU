#include "kupl/lattice_halo.h"
#include "kupl/lattice_set.h"

#include <complex>
#include <stdexcept>
#include <vector>

namespace kupl::qcu {

template <typename Real>
int64_t LatticeSet<Real>::laplacian_site(int x, int y, int z) const {
  return geometry_.site_linear(x, y, z, 0);
}

template <typename Real>
std::size_t LatticeSet<Real>::laplacian_gauge_index(
    int row, int column, int direction, int x, int y, int z) const {
  int64_t index = row;
  index = index * kColor + column;
  index = index * 3 + direction;
  index = index * geometry_.local[0] + x;
  index = index * geometry_.local[1] + y;
  index = index * geometry_.local[2] + z;
  return static_cast<std::size_t>(index);
}

template <typename Real>
void LatticeSet<Real>::decode_laplacian_site(int64_t index, int &x, int &y,
                                             int &z) const {
  z = static_cast<int>(index % geometry_.local[2]);
  index /= geometry_.local[2];
  y = static_cast<int>(index % geometry_.local[1]);
  index /= geometry_.local[1];
  x = static_cast<int>(index);
}

template <typename Real>
void LatticeSet<Real>::pack_laplacian_face(
    int dim, int boundary, const Complex<Real> *source,
    const Complex<Real> *gauge, Complex<Real> *source_buffer,
    Complex<Real> *gauge_buffer) {
  const int64_t count = geometry_.half_sites / surface_extent(dim);
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t face_site = 0; face_site < count; ++face_site) {
    int x = 0;
    int y = 0;
    int z = 0;
    int t2 = 0;
    decode_face(face_site, dim, boundary, x, y, z, t2);
    const int64_t linear = laplacian_site(x, y, z);
    for (int color = 0; color < kColor; ++color) {
      source_buffer[face_site * kColor + color] =
          source[static_cast<int64_t>(color) * geometry_.half_sites +
                 linear];
    }
    for (int row = 0; row < kColor; ++row) {
      for (int column = 0; column < kColor; ++column) {
        gauge_buffer[face_site * kGauge + row * kColor + column] =
            gauge[laplacian_gauge_index(row, column, dim, x, y, z)];
      }
    }
  }
}

template <typename Real>
void LatticeSet<Real>::post_laplacian_halo(const Complex<Real> *source,
                                           const Complex<Real> *gauge) {
  laplacian_requests_.clear();
  laplacian_requests_.reserve(24);
  for (int dim = 0; dim < 3; ++dim) {
    if (geometry_.grid[dim] == 1) {
      continue;
    }
    auto &source_send_minus =
        laplacian_source_send_[face_offset(dim, -1)];
    auto &source_send_plus = laplacian_source_send_[face_offset(dim, 1)];
    auto &source_recv_minus =
        laplacian_source_recv_[face_offset(dim, -1)];
    auto &source_recv_plus = laplacian_source_recv_[face_offset(dim, 1)];
    auto &gauge_send_minus = laplacian_gauge_send_[face_offset(dim, -1)];
    auto &gauge_send_plus = laplacian_gauge_send_[face_offset(dim, 1)];
    auto &gauge_recv_minus = laplacian_gauge_recv_[face_offset(dim, -1)];
    auto &gauge_recv_plus = laplacian_gauge_recv_[face_offset(dim, 1)];

    pack_laplacian_face(dim, 0, source, gauge, source_send_minus.data(),
                        gauge_send_minus.data());
    pack_laplacian_face(dim, geometry_.local[dim] - 1, source, gauge,
                        source_send_plus.data(), gauge_send_plus.data());
    MPI_Request request = MPI_REQUEST_NULL;
    MPI_Irecv(source_recv_plus.data(),
              static_cast<int>(source_recv_plus.size() *
                               sizeof(Complex<Real>)),
              MPI_BYTE, geometry_.neighbor_plus[dim],
              laplacian_face_tag(1, dim, -1), geometry_.comm, &request);
    laplacian_requests_.push_back(request);
    MPI_Irecv(source_recv_minus.data(),
              static_cast<int>(source_recv_minus.size() *
                               sizeof(Complex<Real>)),
              MPI_BYTE, geometry_.neighbor_minus[dim],
              laplacian_face_tag(1, dim, 1), geometry_.comm, &request);
    laplacian_requests_.push_back(request);
    MPI_Irecv(gauge_recv_plus.data(),
              static_cast<int>(gauge_recv_plus.size() *
                               sizeof(Complex<Real>)),
              MPI_BYTE, geometry_.neighbor_plus[dim],
              laplacian_face_tag(2, dim, -1), geometry_.comm, &request);
    laplacian_requests_.push_back(request);
    MPI_Irecv(gauge_recv_minus.data(),
              static_cast<int>(gauge_recv_minus.size() *
                               sizeof(Complex<Real>)),
              MPI_BYTE, geometry_.neighbor_minus[dim],
              laplacian_face_tag(2, dim, 1), geometry_.comm, &request);
    laplacian_requests_.push_back(request);
    MPI_Isend(source_send_plus.data(),
              static_cast<int>(source_send_plus.size() *
                               sizeof(Complex<Real>)),
              MPI_BYTE, geometry_.neighbor_plus[dim],
              laplacian_face_tag(1, dim, 1), geometry_.comm, &request);
    laplacian_requests_.push_back(request);
    MPI_Isend(source_send_minus.data(),
              static_cast<int>(source_send_minus.size() *
                               sizeof(Complex<Real>)),
              MPI_BYTE, geometry_.neighbor_minus[dim],
              laplacian_face_tag(1, dim, -1), geometry_.comm, &request);
    laplacian_requests_.push_back(request);
    MPI_Isend(gauge_send_plus.data(),
              static_cast<int>(gauge_send_plus.size() *
                               sizeof(Complex<Real>)),
              MPI_BYTE, geometry_.neighbor_plus[dim],
              laplacian_face_tag(2, dim, 1), geometry_.comm, &request);
    laplacian_requests_.push_back(request);
    MPI_Isend(gauge_send_minus.data(),
              static_cast<int>(gauge_send_minus.size() *
                               sizeof(Complex<Real>)),
              MPI_BYTE, geometry_.neighbor_minus[dim],
              laplacian_face_tag(2, dim, -1), geometry_.comm, &request);
    laplacian_requests_.push_back(request);
  }
}

template <typename Real>
void LatticeSet<Real>::compute_laplacian_local(
    const Complex<Real> *source, const Complex<Real> *gauge,
    Complex<Real> *output) {
  const int64_t count = geometry_.half_sites;
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t site = 0; site < count; ++site) {
    int x = 0;
    int y = 0;
    int z = 0;
    decode_laplacian_site(site, x, y, z);
    const int coordinate[3] = {x, y, z};
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
        const int64_t plus_site = laplacian_site(plus[0], plus[1], plus[2]);
        for (int row = 0; row < kColor; ++row) {
          Complex<Real> value(0);
          for (int column = 0; column < kColor; ++column) {
            const Complex<Real> link =
                gauge[laplacian_gauge_index(row, column, dim, x, y, z)];
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
            (minus[dim] - 1 + geometry_.local[dim]) %
            geometry_.local[dim];
        const int64_t minus_site =
            laplacian_site(minus[0], minus[1], minus[2]);
        for (int row = 0; row < kColor; ++row) {
          Complex<Real> value(0);
          for (int column = 0; column < kColor; ++column) {
            const Complex<Real> link = std::conj(
                gauge[laplacian_gauge_index(column, row, dim, minus[0],
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

template <typename Real>
void LatticeSet<Real>::subtract_laplacian_boundary(
    const Complex<Real> *gauge, Complex<Real> *output) {
  const int64_t count = geometry_.half_sites;
  const int64_t chunk = omp_chunk<Real>(count);
  for (int dim = 0; dim < 3; ++dim) {
    if (geometry_.grid[dim] == 1) {
      continue;
    }
    for (int sign : {-1, 1}) {
      const auto &source_buffer =
          laplacian_source_recv_[face_offset(dim, sign)];
      const auto &gauge_buffer =
          laplacian_gauge_recv_[face_offset(dim, sign)];
#pragma omp parallel for schedule(static, chunk)
      for (int64_t site = 0; site < count; ++site) {
        int x = 0;
        int y = 0;
        int z = 0;
        decode_laplacian_site(site, x, y, z);
        const int coordinate[3] = {x, y, z};
        const int boundary =
            sign > 0 ? geometry_.local[dim] - 1 : 0;
        if (coordinate[dim] != boundary) {
          continue;
        }
        const int64_t face = geometry_.face_index(dim, x, y, z, 0);
        const Complex<Real> *source =
            source_buffer.data() + face * kColor;
        const Complex<Real> *link =
            gauge_buffer.data() + face * kGauge;
        for (int row = 0; row < kColor; ++row) {
          Complex<Real> value(0);
          for (int column = 0; column < kColor; ++column) {
            const Complex<Real> factor =
                sign > 0
                    ? gauge[laplacian_gauge_index(row, column, dim, x, y, z)]
                    : std::conj(link[column * kColor + row]);
            value += factor * source[column];
          }
          output[static_cast<int64_t>(row) * count + site] -= value;
        }
      }
    }
  }
}

template <typename Real>
void LatticeSet<Real>::apply_laplacian(Complex<Real> *output,
                                       const Complex<Real> *input,
                                       const Complex<Real> *gauge) {
  if (output == nullptr || input == nullptr || gauge == nullptr) {
    throw std::runtime_error("Laplacian received a null pointer");
  }
  post_laplacian_halo(input, gauge);
  compute_laplacian_local(input, gauge, output);
  if (!laplacian_requests_.empty()) {
    MPI_Waitall(static_cast<int>(laplacian_requests_.size()),
                laplacian_requests_.data(), MPI_STATUSES_IGNORE);
    laplacian_requests_.clear();
  }
  subtract_laplacian_boundary(gauge, output);
}

template class LatticeSet<float>;
template class LatticeSet<double>;

} // namespace kupl::qcu
