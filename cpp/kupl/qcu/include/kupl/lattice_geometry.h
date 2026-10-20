#ifndef KUPL_QCU_LATTICE_GEOMETRY_H
#define KUPL_QCU_LATTICE_GEOMETRY_H

#include "kupl/define.h"
#include "kupl/lattice_complex.h"

#include <array>
#include <cstdint>
#include <vector>

#include <mpi.h>

namespace kupl::qcu {

template <typename Real> struct RuntimeParameters {
  std::array<int32_t, kParamsSize> ints{};
  std::array<Real, kArgvSize> reals{};

  int get(int index) const { return static_cast<int>(ints[index]); }
};

struct LatticeGeometry {
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

  LatticeGeometry() = default;
  LatticeGeometry(const LatticeGeometry &) = delete;
  LatticeGeometry &operator=(const LatticeGeometry &) = delete;

  LatticeGeometry(LatticeGeometry &&other) noexcept { move_from(other); }
  LatticeGeometry &operator=(LatticeGeometry &&other) noexcept {
    if (this != &other) {
      release();
      move_from(other);
    }
    return *this;
  }

  ~LatticeGeometry() { release(); }

  void move_from(LatticeGeometry &other) {
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

  int64_t site_linear(int x, int y, int z, int t2) const {
    if (laplacian) {
      return (static_cast<int64_t>(x) * local[1] + y) * local[2] + z;
    }
    return ((static_cast<int64_t>(x) * local[1] + y) * local[2] + z) *
               (local[3] / 2) +
           t2;
  }

  int64_t face_index(int dim, int x, int y, int z, int t2) const {
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

  int64_t corner_index(int dim0, int dim1, int x, int y, int z,
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

  void decode_half(int64_t index, int &x, int &y, int &z, int &t2) const {
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

  int parity_at(int x, int y, int z, int t) const {
    return (x + y + z + t) & 1;
  }

  int time_for_parity(int x, int y, int z, int t2, int parity) const {
    return 2 * t2 + (((x + y + z) & 1) ^ parity);
  }

  int global_coordinate(int dim, int local_coordinate) const {
    return coord[dim] * local[dim] + local_coordinate;
  }

  int last_half_coordinate(int dim) const {
    if (dim == 3) {
      return (laplacian ? local[3] : local[3] / 2) - 1;
    }
    return local[dim] - 1;
  }
};

template <typename Real> struct GaugeHaloBuffer {
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

LatticeGeometry make_geometry(const int32_t *params);

} // namespace kupl::qcu

#endif
