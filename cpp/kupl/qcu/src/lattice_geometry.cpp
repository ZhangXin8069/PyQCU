#include "kupl/lattice_geometry.h"
#include "kupl/lattice_set.h"

#include <stdexcept>
#include <utility>

namespace kupl::qcu {

LatticeGeometry make_geometry(const int32_t *params) {
  LatticeGeometry geometry;
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
      throw std::runtime_error("all local lattice dimensions must be even");
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
  if (MPI_Cart_create(MPI_COMM_WORLD, kDim, geometry.grid.data(), periods, 0,
                      &geometry.comm) != MPI_SUCCESS) {
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
    const int64_t extent =
        dim == 3 ? (geometry.laplacian ? 1 : geometry.local[3] / 2)
                 : geometry.local[dim];
    geometry.face_sites[dim] = geometry.half_sites / extent;
    MPI_Cart_shift(geometry.comm, dim, 1, &geometry.neighbor_minus[dim],
                   &geometry.neighbor_plus[dim]);
  }
  for (int dim0 = 0; dim0 < kDim; ++dim0) {
    for (int dim1 = dim0 + 1; dim1 < kDim; ++dim1) {
      const int64_t extent0 =
          dim0 == 3 ? (geometry.laplacian ? 1 : geometry.local[3] / 2)
                    : geometry.local[dim0];
      const int64_t extent1 =
          dim1 == 3 ? (geometry.laplacian ? 1 : geometry.local[3] / 2)
                    : geometry.local[dim1];
      geometry.corner_sites[dim0][dim1] =
          geometry.half_sites / (extent0 * extent1);
    }
  }
  return geometry;
}

template <typename Real>
int64_t LatticeSet<Real>::site_linear(int x, int y, int z, int t2) const {
  return geometry_.site_linear(x, y, z, t2);
}

template <typename Real> int LatticeSet<Real>::surface_extent(int dim) const {
  if (dim != 3) {
    return geometry_.local[dim];
  }
  return geometry_.laplacian ? 1 : geometry_.local[3] / 2;
}

template <typename Real>
void LatticeSet<Real>::decode_face(int64_t face, int fixed_dim,
                                   int fixed_value, int &x, int &y, int &z,
                                   int &t2) const {
  std::array<int, kDim> coordinate{};
  for (int dim = kDim - 1; dim >= 0; --dim) {
    if (dim == fixed_dim) {
      coordinate[dim] = fixed_value;
    } else {
      coordinate[dim] = static_cast<int>(face % surface_extent(dim));
      face /= surface_extent(dim);
    }
  }
  x = coordinate[0];
  y = coordinate[1];
  z = coordinate[2];
  t2 = coordinate[3];
}

template <typename Real>
void LatticeSet<Real>::decode_corner(int64_t corner, int fixed_dim0,
                                     int fixed_value0, int fixed_dim1,
                                     int fixed_value1, int &x, int &y, int &z,
                                     int &t2) const {
  std::array<int, kDim> coordinate{};
  for (int dim = kDim - 1; dim >= 0; --dim) {
    if (dim == fixed_dim0) {
      coordinate[dim] = fixed_value0;
    } else if (dim == fixed_dim1) {
      coordinate[dim] = fixed_value1;
    } else {
      coordinate[dim] = static_cast<int>(corner % surface_extent(dim));
      corner /= surface_extent(dim);
    }
  }
  x = coordinate[0];
  y = coordinate[1];
  z = coordinate[2];
  t2 = coordinate[3];
}

template <typename Real>
VectorView<Real>
LatticeSet<Real>::compact_view(const Complex<Real> *field,
                               int64_t site) const {
  return {field + site, geometry_.half_sites};
}

template <typename Real>
VectorView<Real>
LatticeSet<Real>::halo_view(const std::vector<Complex<Real>> &halo,
                            int64_t face_site) const {
  return {halo.data() + face_site * kVector, 1};
}

template <typename Real>
void LatticeSet<Real>::gather_vector(const VectorView<Real> &view,
                                     Complex<Real> local[kVector]) const {
  for (int component = 0; component < kVector; ++component) {
    local[component] = view[component];
  }
}

template <typename Real>
void LatticeSet<Real>::scatter_vector(const Complex<Real> local[kVector],
                                      Complex<Real> *field,
                                      int64_t site) const {
  for (int component = 0; component < kVector; ++component) {
    field[static_cast<int64_t>(component) * geometry_.half_sites + site] =
        local[component];
  }
}

template <typename Real>
std::size_t LatticeSet<Real>::gauge_index(
    int parity, int row, int column, int direction, int x, int y, int z,
    int t2) const {
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

template <typename Real>
GaugeView<Real> LatticeSet<Real>::local_gauge(int parity, int direction,
                                              int x, int y, int z,
                                              int t2) const {
  return {gauge_halo_.source +
              gauge_index(parity, 0, 0, direction, x, y, z, t2),
          static_cast<int64_t>(kDim) * geometry_.half_sites};
}

template class LatticeSet<float>;
template class LatticeSet<double>;

} // namespace kupl::qcu
