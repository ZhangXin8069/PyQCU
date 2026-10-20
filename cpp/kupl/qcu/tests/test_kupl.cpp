#include "kupl/define.h"
#include "kupl/qcu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <mpi.h>
#include <omp.h>

namespace {

template <typename Real> using Complex = std::complex<Real>;

constexpr int kDim = 4;
constexpr int kSpin = 4;
constexpr int kColor = 3;
constexpr int kVector = kSpin * kColor;
constexpr int kOnsite = kVector * kVector;

int64_t canonicalFermionIndex(int parity, int component, int64_t site,
                              int64_t half_sites) {
  return (static_cast<int64_t>(parity) * kVector + component) * half_sites +
         site;
}

int64_t canonicalCompactFermionIndex(int component, int64_t site,
                                     int64_t half_sites) {
  return static_cast<int64_t>(component) * half_sites + site;
}

int64_t canonicalCloverIndex(int row, int column, int64_t site,
                             int64_t half_sites) {
  return static_cast<int64_t>(row * kVector + column) * half_sites + site;
}

template <typename Real> struct Grid {
  std::array<int, kDim> local{};
  std::array<int, kDim> grid{};
  std::array<int, kDim> global{};
  std::array<int, kDim> coord{};
  int rank = 0;
  int size = 1;
  int64_t local_volume = 0;
  int64_t global_volume = 0;
  int64_t half_sites = 0;
  int64_t parity_vectors = 0;
  MPI_Comm comm = MPI_COMM_NULL;

  Grid(const std::array<int, kDim> &local_dims,
       const std::array<int, kDim> &grid_dims) {
    local = local_dims;
    grid = grid_dims;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int product = 1;
    for (int dim = 0; dim < kDim; ++dim) {
      product *= grid[dim];
      global[dim] = local[dim] * grid[dim];
    }
    if (product != size) {
      throw std::runtime_error("test grid product must equal MPI size");
    }
    int periods[kDim] = {1, 1, 1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, kDim, grid.data(), periods, 0, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, kDim, coord.data());
    local_volume = static_cast<int64_t>(local[0]) * local[1] * local[2] *
                   local[3];
    global_volume = static_cast<int64_t>(global[0]) * global[1] * global[2] *
                    global[3];
    half_sites = local_volume / 2;
    parity_vectors = half_sites * kVector;
  }

  Grid(const Grid &) = delete;
  Grid &operator=(const Grid &) = delete;
  ~Grid() {
    if (comm != MPI_COMM_NULL) {
      MPI_Comm_free(&comm);
    }
  }

  int globalCoordinate(int dim, int local_coordinate) const {
    return coord[dim] * local[dim] + local_coordinate;
  }

  int positiveModulo(int value, int extent) const {
    int result = value % extent;
    return result < 0 ? result + extent : result;
  }

  int64_t fullGlobalIndex(int x, int y, int z, int t) const {
    return (((static_cast<int64_t>(x) * global[1] + y) * global[2] + z) *
                global[3] +
            t);
  }
};

template <typename Real>
Complex<Real> globalSource(int x, int y, int z, int t, int spin, int color) {
  const Real phase = Real(0.071) *
                         static_cast<Real>(x + 3 * y + 5 * z + 7 * t +
                                           11 * spin + 13 * color) +
                     Real(0.037) * static_cast<Real>(spin * color + 1);
  return Complex<Real>(std::cos(phase), std::sin(Real(1.7) * phase));
}

template <typename Real>
std::array<Complex<Real>, 9> gaugeMatrix(int x, int y, int z, int t,
                                         int direction) {
  using C = Complex<Real>;
  if (const char *identity = std::getenv("KUPL_TEST_IDENTITY_GAUGE");
      identity != nullptr && std::atoi(identity) != 0) {
    return {C(1), C(0), C(0), C(0), C(1), C(0), C(0), C(0), C(1)};
  }
  const Real theta =
      Real(0.31) * std::sin(Real(0.17) * static_cast<Real>(x + 2 * y + 3 * z +
                                                          5 * t + 7 * direction)) +
      Real(0.11);
  const Real phi =
      Real(0.43) * static_cast<Real>(x - y + 2 * z - 3 * t + direction);
  const Real psi =
      Real(0.29) * static_cast<Real>(2 * x + y - z + 4 * t - direction);
  const Real cosine = std::cos(theta);
  const Real sine = std::sin(theta);
  const C first_phase(std::cos(phi), std::sin(phi));
  const C second_phase(std::cos(psi), std::sin(psi));
  return {
      cosine, -sine * first_phase, C(0),
      sine * first_phase, cosine, C(0),
      C(0), C(0), second_phase,
  };
}

template <typename Real>
Complex<Real> gammaEntry(int direction, int row, int column) {
  using C = Complex<Real>;
  const C i(0, 1);
  static const C g0[16] = {
      C(0), C(0), C(0), i, C(0), C(0), i, C(0),
      C(0), -i, C(0), C(0), -i, C(0), C(0), C(0),
  };
  static const C g1[16] = {
      C(0), C(0), C(0), C(-1), C(0), C(0), C(1), C(0),
      C(0), C(1), C(0), C(0), C(-1), C(0), C(0), C(0),
  };
  static const C g2[16] = {
      C(0), C(0), i, C(0), C(0), C(0), C(0), -i,
      -i, C(0), C(0), C(0), C(0), i, C(0), C(0),
  };
  static const C g3[16] = {
      C(0), C(0), C(1), C(0), C(0), C(0), C(0), C(1),
      C(1), C(0), C(0), C(0), C(0), C(1), C(0), C(0),
  };
  if (direction == 0) {
    return g0[row * 4 + column];
  }
  if (direction == 1) {
    return g1[row * 4 + column];
  }
  if (direction == 2) {
    return g2[row * 4 + column];
  }
  return g3[row * 4 + column];
}

template <typename Real>
void multiplyGauge(const std::array<Complex<Real>, 9> &matrix,
                   const Complex<Real> input[kVector],
                   Complex<Real> output[kVector], bool dagger) {
  for (int spin = 0; spin < kSpin; ++spin) {
    for (int row = 0; row < kColor; ++row) {
      Complex<Real> value(0);
      for (int column = 0; column < kColor; ++column) {
        const auto entry =
            dagger ? std::conj(matrix[column * kColor + row])
                   : matrix[row * kColor + column];
        value += entry * input[spin * kColor + column];
      }
      output[spin * kColor + row] = value;
    }
  }
}

template <typename Real>
void multiplyOnsite(const Complex<Real> matrix[144],
                    const Complex<Real> input[kVector],
                    Complex<Real> output[kVector]) {
  for (int row = 0; row < kVector; ++row) {
    Complex<Real> value(0);
    for (int column = 0; column < kVector; ++column) {
      value += matrix[row * kVector + column] * input[column];
    }
    output[row] = value;
  }
}

template <typename Real>
void multiply3(const std::array<Complex<Real>, 9> &left,
               const std::array<Complex<Real>, 9> &right,
               std::array<Complex<Real>, 9> &output) {
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      Complex<Real> value(0);
      for (int inner = 0; inner < 3; ++inner) {
        value += left[row * 3 + inner] * right[inner * 3 + column];
      }
      output[row * 3 + column] = value;
    }
  }
}

template <typename Real>
std::array<Complex<Real>, 9> dagger(const std::array<Complex<Real>, 9> &input) {
  std::array<Complex<Real>, 9> output{};
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      output[row * 3 + column] =
          std::conj(input[column * 3 + row]);
    }
  }
  return output;
}

template <typename Real>
std::array<Complex<Real>, 9> referencePlaquetteSum(
    const Grid<Real> &grid, int x, int y, int z, int t, int mu, int nu) {
  using C = Complex<Real>;
  std::array<C, 9> sum{};
  std::array<C, 9> m0{};
  std::array<C, 9> m1{};
  std::array<C, 9> m2{};
  std::array<C, 9> m3{};
  auto load_term = [&](int direction, int dx, int dy, int dz, int dt,
                       bool adjoint, std::array<C, 9> &output) {
    int coordinate[kDim] = {x + dx, y + dy, z + dz, t + dt};
    for (int dim = 0; dim < kDim; ++dim) {
      coordinate[dim] =
          grid.positiveModulo(coordinate[dim], grid.global[dim]);
    }
    output = gaugeMatrix<Real>(coordinate[0], coordinate[1], coordinate[2],
                               coordinate[3], direction);
    if (adjoint) {
      output = dagger(output);
    }
  };
  auto add_product = [&]() {
    std::array<C, 9> temp{};
    std::array<C, 9> product{};
    multiply3(m0, m1, temp);
    multiply3(temp, m2, product);
    multiply3(product, m3, temp);
    for (int index = 0; index < 9; ++index) {
      sum[index] += temp[index];
    }
  };

  load_term(mu, 0, 0, 0, 0, false, m0);
  load_term(nu, mu == 0 ? 1 : 0, mu == 1 ? 1 : 0, mu == 2 ? 1 : 0,
            mu == 3 ? 1 : 0, false, m1);
  load_term(mu, nu == 0 ? 1 : 0, nu == 1 ? 1 : 0, nu == 2 ? 1 : 0,
            nu == 3 ? 1 : 0, true, m2);
  load_term(nu, 0, 0, 0, 0, true, m3);
  add_product();

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
  return sum;
}

template <typename Real>
void referenceClover(const Grid<Real> &grid, int x, int y, int z, int t,
                     Real mass, Real csw, Complex<Real> output[144]) {
  using C = Complex<Real>;
  const Real kappa = Real(1) / (Real(2) * mass + Real(8));
  const Real coefficient = -Real(0.125) * kappa * csw;
  std::fill(output, output + 144, C(0));
  for (int component = 0; component < kVector; ++component) {
    output[component * kVector + component] = C(1);
  }
  for (int mu = 0; mu < kDim; ++mu) {
    for (int nu = mu + 1; nu < kDim; ++nu) {
      const auto plaquette_sum =
          referencePlaquetteSum(grid, x, y, z, t, mu, nu);
      std::array<C, 9> field{};
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          field[row * 3 + column] =
              plaquette_sum[row * 3 + column] -
              std::conj(plaquette_sum[column * 3 + row]);
        }
      }
      for (int spin_out = 0; spin_out < 4; ++spin_out) {
        for (int spin_in = 0; spin_in < 4; ++spin_in) {
          C sigma(0);
          for (int inner = 0; inner < 4; ++inner) {
            sigma += gammaEntry<Real>(mu, spin_out, inner) *
                     gammaEntry<Real>(nu, inner, spin_in);
          }
          for (int color_out = 0; color_out < 3; ++color_out) {
            for (int color_in = 0; color_in < 3; ++color_in) {
              output[(spin_out * 3 + color_out) * kVector +
                     spin_in * 3 + color_in] +=
                  coefficient * sigma * field[color_out * 3 + color_in];
            }
          }
        }
      }
    }
  }
}

template <typename Real>
void writeCanonicalClover(const Grid<Real> &grid, int x, int y, int z, int t,
                          int64_t site, Real mass, Real csw,
                          Complex<Real> *output) {
  Complex<Real> local[kOnsite];
  referenceClover(grid, x, y, z, t, mass, csw, local);
  for (int component = 0; component < kOnsite; ++component) {
    output[static_cast<int64_t>(component) * grid.half_sites + site] =
        local[component];
  }
}

template <typename Real>
void referenceBareDslash(
    const Grid<Real> &grid, int x, int y, int z, int t,
    const std::vector<Complex<Real>> &source, Complex<Real> output[kVector]) {
  using C = Complex<Real>;
  for (int index = 0; index < kVector; ++index) {
    output[index] = C(0);
  }
  for (int direction = 0; direction < kDim; ++direction) {
    int forward[kDim] = {x, y, z, t};
    int backward[kDim] = {x, y, z, t};
    forward[direction] =
        grid.positiveModulo(forward[direction] + 1, grid.global[direction]);
    backward[direction] =
        grid.positiveModulo(backward[direction] - 1, grid.global[direction]);
    const int64_t forward_base =
        grid.fullGlobalIndex(forward[0], forward[1], forward[2], forward[3]) *
        kVector;
    const int64_t backward_base =
        grid.fullGlobalIndex(backward[0], backward[1], backward[2],
                             backward[3]) *
        kVector;
    C forward_source[kVector];
    C backward_source[kVector];
    std::copy(source.begin() + forward_base,
              source.begin() + forward_base + kVector, forward_source);
    std::copy(source.begin() + backward_base,
              source.begin() + backward_base + kVector, backward_source);

    C gauge_forward[kVector];
    C gauge_backward[kVector];
    multiplyGauge(gaugeMatrix<Real>(x, y, z, t, direction), forward_source,
                  gauge_forward, false);
    multiplyGauge(gaugeMatrix<Real>(backward[0], backward[1], backward[2],
                                    backward[3], direction),
                  backward_source, gauge_backward, true);
    for (int spin = 0; spin < 4; ++spin) {
      for (int color = 0; color < 3; ++color) {
        C gamma_forward(0);
        C gamma_backward(0);
        for (int inner = 0; inner < 4; ++inner) {
          gamma_forward +=
              gammaEntry<Real>(direction, spin, inner) *
              gauge_forward[inner * 3 + color];
          gamma_backward +=
              gammaEntry<Real>(direction, spin, inner) *
              gauge_backward[inner * 3 + color];
        }
        output[spin * 3 + color] +=
            gauge_forward[spin * 3 + color] - gamma_forward +
            gauge_backward[spin * 3 + color] + gamma_backward;
      }
    }
  }
}

template <typename Real>
bool invertOnsite(const Complex<Real> term[144],
                  Complex<Real> inverse[144]) {
  Complex<Real> augmented[kVector][2 * kVector];
  for (int row = 0; row < kVector; ++row) {
    for (int column = 0; column < 2 * kVector; ++column) {
      augmented[row][column] = Complex<Real>(0);
    }
    for (int column = 0; column < kVector; ++column) {
      augmented[row][column] = term[row * kVector + column];
    }
    augmented[row][kVector + row] = Complex<Real>(1);
  }
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
    if (pivot_norm <= std::numeric_limits<Real>::epsilon()) {
      return false;
    }
    if (pivot != column) {
      for (int entry = 0; entry < 2 * kVector; ++entry) {
        std::swap(augmented[pivot][entry], augmented[column][entry]);
      }
    }
    const auto diagonal = augmented[column][column];
    for (int entry = 0; entry < 2 * kVector; ++entry) {
      augmented[column][entry] /= diagonal;
    }
    for (int row = 0; row < kVector; ++row) {
      if (row == column) {
        continue;
      }
      const auto factor = augmented[row][column];
      for (int entry = 0; entry < 2 * kVector; ++entry) {
        augmented[row][entry] -= factor * augmented[column][entry];
      }
    }
  }
  for (int row = 0; row < kVector; ++row) {
    for (int column = 0; column < kVector; ++column) {
      inverse[row * kVector + column] = augmented[row][kVector + column];
    }
  }
  return true;
}

template <typename Real>
void decodeHalf(const Grid<Real> &grid, int64_t index, int &x, int &y, int &z,
                int &t2) {
  const int ht = grid.local[3] / 2;
  t2 = static_cast<int>(index % ht);
  index /= ht;
  z = static_cast<int>(index % grid.local[2]);
  index /= grid.local[2];
  y = static_cast<int>(index % grid.local[1]);
  index /= grid.local[1];
  x = static_cast<int>(index);
}

template <typename Real>
int localTime(int x, int y, int z, int t2, int parity) {
  return 2 * t2 + (((x + y + z) & 1) ^ parity);
}

template <typename Real>
std::vector<Complex<Real>> makeLocalGauge(const Grid<Real> &grid) {
  std::vector<Complex<Real>> gauge(
      static_cast<std::size_t>(2 * grid.half_sites * 9 * kDim));
  for (int parity = 0; parity < 2; ++parity) {
    for (int64_t site = 0; site < grid.half_sites; ++site) {
      int x = 0;
      int y = 0;
      int z = 0;
      int t2 = 0;
      decodeHalf(grid, site, x, y, z, t2);
      const int t = localTime<Real>(x, y, z, t2, parity);
      const int gx = grid.globalCoordinate(0, x);
      const int gy = grid.globalCoordinate(1, y);
      const int gz = grid.globalCoordinate(2, z);
      const int gt = grid.globalCoordinate(3, t);
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          for (int direction = 0; direction < kDim; ++direction) {
            int64_t index = parity;
            index = index * 3 + row;
            index = index * 3 + column;
            index = index * 4 + direction;
            index = index * grid.local[0] + x;
            index = index * grid.local[1] + y;
            index = index * grid.local[2] + z;
            index = index * (grid.local[3] / 2) + t2;
            gauge[static_cast<std::size_t>(index)] =
                gaugeMatrix<Real>(gx, gy, gz, gt, direction)
                    [row * 3 + column];
          }
        }
      }
    }
  }
  return gauge;
}

template <typename Real>
std::vector<Complex<Real>> makeLocalFullField(const Grid<Real> &grid) {
  std::vector<Complex<Real>> field(
      static_cast<std::size_t>(grid.parity_vectors * 2));
  for (int parity = 0; parity < 2; ++parity) {
    for (int64_t site = 0; site < grid.half_sites; ++site) {
      int x = 0;
      int y = 0;
      int z = 0;
      int t2 = 0;
      decodeHalf(grid, site, x, y, z, t2);
      const int t = localTime<Real>(x, y, z, t2, parity);
      const int gx = grid.globalCoordinate(0, x);
      const int gy = grid.globalCoordinate(1, y);
      const int gz = grid.globalCoordinate(2, z);
      const int gt = grid.globalCoordinate(3, t);
      for (int spin = 0; spin < 4; ++spin) {
        for (int color = 0; color < 3; ++color) {
          const int component = spin * 3 + color;
          field[static_cast<std::size_t>(canonicalFermionIndex(
              parity, component, site, grid.half_sites))] =
              globalSource<Real>(gx, gy, gz, gt, spin, color);
        }
      }
    }
  }
  return field;
}

template <typename Real>
std::vector<Complex<Real>> makeGlobalField(const Grid<Real> &grid,
                                           const std::vector<Complex<Real>> &local) {
  std::vector<Complex<Real>> global(
      static_cast<std::size_t>(grid.global_volume * kVector),
      Complex<Real>(0));
  for (int parity = 0; parity < 2; ++parity) {
    for (int64_t site = 0; site < grid.half_sites; ++site) {
      int x = 0;
      int y = 0;
      int z = 0;
      int t2 = 0;
      decodeHalf(grid, site, x, y, z, t2);
      const int t = localTime<Real>(x, y, z, t2, parity);
      const int gx = grid.globalCoordinate(0, x);
      const int gy = grid.globalCoordinate(1, y);
      const int gz = grid.globalCoordinate(2, z);
      const int gt = grid.globalCoordinate(3, t);
      const int64_t destination =
          grid.fullGlobalIndex(gx, gy, gz, gt) * kVector;
      for (int component = 0; component < kVector; ++component) {
        global[static_cast<std::size_t>(destination + component)] =
            local[static_cast<std::size_t>(canonicalFermionIndex(
                parity, component, site, grid.half_sites))];
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, global.data(),
                static_cast<int>(global.size() * 2),
                sizeof(Real) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD);
  return global;
}

template <typename Real>
std::vector<Complex<Real>> makeGlobalParityField(
    const Grid<Real> &grid, const std::vector<Complex<Real>> &local,
    int parity) {
  std::vector<Complex<Real>> global(
      static_cast<std::size_t>(grid.global_volume * kVector),
      Complex<Real>(0));
  for (int64_t site = 0; site < grid.half_sites; ++site) {
    int x = 0;
    int y = 0;
    int z = 0;
    int t2 = 0;
    decodeHalf(grid, site, x, y, z, t2);
    const int t = localTime<Real>(x, y, z, t2, parity);
    const int64_t destination =
        grid.fullGlobalIndex(grid.globalCoordinate(0, x),
                             grid.globalCoordinate(1, y),
                             grid.globalCoordinate(2, z),
                             grid.globalCoordinate(3, t)) *
        kVector;
    for (int component = 0; component < kVector; ++component) {
      global[static_cast<std::size_t>(destination + component)] =
          local[static_cast<std::size_t>(canonicalCompactFermionIndex(
              component, site, grid.half_sites))];
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, global.data(),
                static_cast<int>(global.size() * 2),
                sizeof(Real) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD);
  return global;
}

template <typename Real>
std::vector<Complex<Real>> referenceWilsonSchur(
    const Grid<Real> &grid, const std::vector<Complex<Real>> &input,
    int target, Real mass) {
  using C = Complex<Real>;
  const int other = 1 - target;
  const Real kappa = Real(1) / (Real(2) * mass + Real(8));
  const auto global_input = makeGlobalParityField(grid, input, target);
  std::vector<C> first_hop(static_cast<std::size_t>(grid.parity_vectors));
  for (int64_t site = 0; site < grid.half_sites; ++site) {
    int x = 0;
    int y = 0;
    int z = 0;
    int t2 = 0;
    decodeHalf(grid, site, x, y, z, t2);
    const int t = localTime<Real>(x, y, z, t2, other);
    C result[kVector];
    referenceBareDslash(
        grid, grid.globalCoordinate(0, x), grid.globalCoordinate(1, y),
        grid.globalCoordinate(2, z), grid.globalCoordinate(3, t),
        global_input, result);
    for (int component = 0; component < kVector; ++component) {
      first_hop[static_cast<std::size_t>(canonicalCompactFermionIndex(
          component, site, grid.half_sites))] = result[component];
    }
  }
  const auto global_first = makeGlobalParityField(grid, first_hop, other);
  std::vector<C> expected(static_cast<std::size_t>(grid.parity_vectors));
  for (int64_t site = 0; site < grid.half_sites; ++site) {
    int x = 0;
    int y = 0;
    int z = 0;
    int t2 = 0;
    decodeHalf(grid, site, x, y, z, t2);
    const int t = localTime<Real>(x, y, z, t2, target);
    C second_hop[kVector];
    referenceBareDslash(
        grid, grid.globalCoordinate(0, x), grid.globalCoordinate(1, y),
        grid.globalCoordinate(2, z), grid.globalCoordinate(3, t),
        global_first, second_hop);
    for (int component = 0; component < kVector; ++component) {
      const std::size_t index = static_cast<std::size_t>(
          canonicalCompactFermionIndex(component, site, grid.half_sites));
      expected[index] = input[index] - kappa * kappa * second_hop[component];
    }
  }
  return expected;
}

template <typename Real>
Complex<Real> globalCompactDot(const std::vector<Complex<Real>> &left,
                               const std::vector<Complex<Real>> &right) {
  Real real_sum = 0;
  Real imag_sum = 0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    const Complex<Real> value = std::conj(left[index]) * right[index];
    real_sum += value.real();
    imag_sum += value.imag();
  }
  const Real local[2] = {real_sum, imag_sum};
  Real global[2] = {0, 0};
  MPI_Allreduce(local, global, 2,
                sizeof(Real) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD);
  return Complex<Real>(global[0], global[1]);
}

template <typename Real>
Real localRelativeError(const std::vector<Complex<Real>> &actual,
                        const std::vector<Complex<Real>> &expected) {
  Real difference = 0;
  Real norm = 0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    difference += std::norm(actual[index] - expected[index]);
    norm += std::norm(expected[index]);
  }
  Real global_difference = 0;
  Real global_norm = 0;
  const MPI_Datatype type =
      sizeof(Real) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE;
  MPI_Allreduce(&difference, &global_difference, 1, type, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&norm, &global_norm, 1, type, MPI_SUM, MPI_COMM_WORLD);
  return std::sqrt(global_difference / std::max(global_norm, Real(1e-30)));
}

template <typename Real>
Real trueResidual(const Grid<Real> &grid, Real mass, Real csw,
                  const std::vector<Complex<Real>> &solution_global,
                  const std::vector<Complex<Real>> &rhs_local) {
  using C = Complex<Real>;
  const Real kappa = Real(1) / (Real(2) * mass + Real(8));
  Real local_difference = 0;
  Real local_rhs = 0;
  for (int parity = 0; parity < 2; ++parity) {
    for (int64_t site = 0; site < grid.half_sites; ++site) {
      int x = 0;
      int y = 0;
      int z = 0;
      int t2 = 0;
      decodeHalf(grid, site, x, y, z, t2);
      const int t = localTime<Real>(x, y, z, t2, parity);
      const int gx = grid.globalCoordinate(0, x);
      const int gy = grid.globalCoordinate(1, y);
      const int gz = grid.globalCoordinate(2, z);
      const int gt = grid.globalCoordinate(3, t);
      C onsite[144];
      referenceClover(grid, gx, gy, gz, gt, mass, csw, onsite);
      C local[kVector];
      C hopping[kVector];
      const int64_t solution_base =
          grid.fullGlobalIndex(gx, gy, gz, gt) * kVector;
      std::copy(solution_global.begin() + solution_base,
                solution_global.begin() + solution_base + kVector, local);
      referenceBareDslash(grid, gx, gy, gz, gt, solution_global, hopping);
      C onsite_result[kVector];
      multiplyOnsite(onsite, local, onsite_result);
      for (int component = 0; component < kVector; ++component) {
        const std::size_t rhs_index = static_cast<std::size_t>(
            canonicalFermionIndex(parity, component, site, grid.half_sites));
        const C residual =
            onsite_result[component] - kappa * hopping[component] -
            rhs_local[rhs_index];
        local_difference += std::norm(residual);
        local_rhs += std::norm(rhs_local[rhs_index]);
      }
    }
  }
  Real global_difference = 0;
  Real global_rhs = 0;
  const MPI_Datatype type =
      sizeof(Real) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE;
  MPI_Allreduce(&local_difference, &global_difference, 1, type, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&local_rhs, &global_rhs, 1, type, MPI_SUM,
                MPI_COMM_WORLD);
  return std::sqrt(global_difference / std::max(global_rhs, Real(1e-30)));
}

template <typename Real>
int64_t laplacianLocalSite(const Grid<Real> &grid, int x, int y, int z) {
  return (static_cast<int64_t>(x) * grid.local[1] + y) * grid.local[2] + z;
}

template <typename Real>
std::vector<Complex<Real>> makeLocalLaplacianGauge(const Grid<Real> &grid) {
  std::vector<Complex<Real>> gauge(static_cast<std::size_t>(
      kColor * kColor * 3 * grid.local_volume));
  for (int x = 0; x < grid.local[0]; ++x) {
    for (int y = 0; y < grid.local[1]; ++y) {
      for (int z = 0; z < grid.local[2]; ++z) {
        const int64_t site = laplacianLocalSite(grid, x, y, z);
        const int gx = grid.globalCoordinate(0, x);
        const int gy = grid.globalCoordinate(1, y);
        const int gz = grid.globalCoordinate(2, z);
        for (int row = 0; row < kColor; ++row) {
          for (int column = 0; column < kColor; ++column) {
            for (int direction = 0; direction < 3; ++direction) {
              const int64_t index =
                  ((row * kColor + column) * 3 + direction) *
                      grid.local_volume +
                  site;
              gauge[static_cast<std::size_t>(index)] =
                  gaugeMatrix<Real>(gx, gy, gz, 0, direction)
                      [row * kColor + column];
            }
          }
        }
      }
    }
  }
  return gauge;
}

template <typename Real>
std::vector<Complex<Real>> makeLocalLaplacianField(const Grid<Real> &grid) {
  std::vector<Complex<Real>> field(
      static_cast<std::size_t>(kColor * grid.local_volume));
  for (int x = 0; x < grid.local[0]; ++x) {
    for (int y = 0; y < grid.local[1]; ++y) {
      for (int z = 0; z < grid.local[2]; ++z) {
        const int64_t site = laplacianLocalSite(grid, x, y, z);
        const int gx = grid.globalCoordinate(0, x);
        const int gy = grid.globalCoordinate(1, y);
        const int gz = grid.globalCoordinate(2, z);
        for (int color = 0; color < kColor; ++color) {
          field[static_cast<std::size_t>(color * grid.local_volume + site)] =
              globalSource<Real>(gx, gy, gz, 0, 0, color);
        }
      }
    }
  }
  return field;
}

template <typename Real>
std::vector<Complex<Real>> makeGlobalLaplacianField(
    const Grid<Real> &grid,
    const std::vector<Complex<Real>> &local_field) {
  std::vector<Complex<Real>> global(
      static_cast<std::size_t>(kColor * grid.global_volume),
      Complex<Real>(0));
  for (int x = 0; x < grid.local[0]; ++x) {
    for (int y = 0; y < grid.local[1]; ++y) {
      for (int z = 0; z < grid.local[2]; ++z) {
        const int64_t local_site = laplacianLocalSite(grid, x, y, z);
        const int64_t global_site = grid.fullGlobalIndex(
            grid.globalCoordinate(0, x), grid.globalCoordinate(1, y),
            grid.globalCoordinate(2, z), 0);
        for (int color = 0; color < kColor; ++color) {
          global[static_cast<std::size_t>(color * grid.global_volume +
                                          global_site)] =
              local_field[static_cast<std::size_t>(
                  color * grid.local_volume + local_site)];
        }
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, global.data(),
                static_cast<int>(global.size() * 2),
                sizeof(Real) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE,
                MPI_SUM, MPI_COMM_WORLD);
  return global;
}

template <typename Real>
std::vector<Complex<Real>> referenceLaplacian(
    const Grid<Real> &grid,
    const std::vector<Complex<Real>> &global_field) {
  using C = Complex<Real>;
  std::vector<C> expected(
      static_cast<std::size_t>(kColor * grid.local_volume));
  for (int x = 0; x < grid.local[0]; ++x) {
    for (int y = 0; y < grid.local[1]; ++y) {
      for (int z = 0; z < grid.local[2]; ++z) {
        const int coordinate[3] = {grid.globalCoordinate(0, x),
                                   grid.globalCoordinate(1, y),
                                   grid.globalCoordinate(2, z)};
        const int64_t site =
            grid.fullGlobalIndex(coordinate[0], coordinate[1], coordinate[2],
                                 0);
        const int64_t local_site = laplacianLocalSite(grid, x, y, z);
        C result[kColor];
        for (int color = 0; color < kColor; ++color) {
          result[color] =
              Real(6) *
              global_field[static_cast<std::size_t>(color *
                                                    grid.global_volume +
                                                    site)];
        }
        for (int direction = 0; direction < 3; ++direction) {
          int plus[3] = {coordinate[0], coordinate[1], coordinate[2]};
          int minus[3] = {coordinate[0], coordinate[1], coordinate[2]};
          plus[direction] =
              grid.positiveModulo(plus[direction] + 1, grid.global[direction]);
          minus[direction] =
              grid.positiveModulo(minus[direction] - 1, grid.global[direction]);
          const int64_t plus_site =
              grid.fullGlobalIndex(plus[0], plus[1], plus[2], 0);
          const int64_t minus_site =
              grid.fullGlobalIndex(minus[0], minus[1], minus[2], 0);
          const auto plus_link = gaugeMatrix<Real>(
              coordinate[0], coordinate[1], coordinate[2], 0, direction);
          const auto minus_link = gaugeMatrix<Real>(
              minus[0], minus[1], minus[2], 0, direction);
          for (int row = 0; row < kColor; ++row) {
            C forward(0);
            C backward(0);
            for (int column = 0; column < kColor; ++column) {
              forward +=
                  plus_link[row * kColor + column] *
                  global_field[static_cast<std::size_t>(
                      column * grid.global_volume + plus_site)];
              backward +=
                  std::conj(minus_link[column * kColor + row]) *
                  global_field[static_cast<std::size_t>(
                      column * grid.global_volume + minus_site)];
            }
            result[row] -= forward + backward;
          }
        }
        for (int color = 0; color < kColor; ++color) {
          expected[static_cast<std::size_t>(color * grid.local_volume +
                                            local_site)] = result[color];
        }
      }
    }
  }
  return expected;
}

template <typename Real>
bool runLaplacianTests(const Grid<Real> &grid) {
  using C = Complex<Real>;
  if (grid.local[3] != 1 || grid.grid[3] != 1) {
    throw std::runtime_error("Laplacian tests require T=grid_t=1");
  }
  std::array<int32_t, kupl::qcu::kParamsSize> params{};
  std::array<Real, kupl::qcu::kArgvSize> argv{};
  std::array<int64_t, kupl::qcu::kSetPtrsSize> set_ptrs{};
  params[kupl::qcu::kLatX] = grid.local[0];
  params[kupl::qcu::kLatY] = grid.local[1];
  params[kupl::qcu::kLatZ] = grid.local[2];
  params[kupl::qcu::kLatT] = 1;
  params[kupl::qcu::kLatXyzt] = static_cast<int32_t>(grid.local_volume);
  params[kupl::qcu::kGridX] = grid.grid[0];
  params[kupl::qcu::kGridY] = grid.grid[1];
  params[kupl::qcu::kGridZ] = grid.grid[2];
  params[kupl::qcu::kGridT] = 1;
  params[kupl::qcu::kParity] = 0;
  params[kupl::qcu::kNodeRank] = grid.rank;
  params[kupl::qcu::kNodeSize] = grid.size;
  params[kupl::qcu::kDataType] =
      sizeof(Real) == sizeof(float) ? kupl::qcu::kLatC64
                                    : kupl::qcu::kLatC128;
  params[kupl::qcu::kSetPlan] = kupl::qcu::kSetPlanLaplacian;
  params[kupl::qcu::kVerbose] = 0;
  argv[kupl::qcu::kMass] = Real(0.1);
  argv[kupl::qcu::kAtol] =
      sizeof(Real) == sizeof(float) ? Real(1e-6) : Real(1e-13);
  argv[kupl::qcu::kCsw] = Real(0);

  auto gauge = makeLocalLaplacianGauge(grid);
  auto input = makeLocalLaplacianField(grid);
  auto global_input = makeGlobalLaplacianField(grid, input);
  auto expected = referenceLaplacian(grid, global_input);
  std::vector<C> output(input.size(), C(0));
  applyInitQcu(reinterpret_cast<long long>(set_ptrs.data()),
               reinterpret_cast<long long>(params.data()),
               reinterpret_cast<long long>(argv.data()));
  applyLaplacianQcu(reinterpret_cast<long long>(output.data()),
                    reinterpret_cast<long long>(input.data()),
                    reinterpret_cast<long long>(gauge.data()),
                    reinterpret_cast<long long>(set_ptrs.data()),
                    reinterpret_cast<long long>(params.data()));
  applyEndQcu(reinterpret_cast<long long>(set_ptrs.data()),
              reinterpret_cast<long long>(params.data()));
  const Real error = localRelativeError(output, expected);
  const Real tolerance =
      sizeof(Real) == sizeof(float) ? Real(2e-5) : Real(2e-12);
  const bool pass = error < tolerance;
  if (grid.rank == 0) {
    std::printf(
        "laplacian type=%s ranks=%d grid=%d,%d,%d,%d local=%d,%d,%d,1 "
        "relative_error=%.9e status=%s\n",
        sizeof(Real) == sizeof(float) ? "c64" : "c128", grid.size,
        grid.grid[0], grid.grid[1], grid.grid[2], grid.grid[3],
        grid.local[0], grid.local[1], grid.local[2],
        static_cast<double>(error), pass ? "PASS" : "FAIL");
  }
  int local_pass = pass ? 1 : 0;
  int global_pass = 0;
  MPI_Allreduce(&local_pass, &global_pass, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  return global_pass != 0;
}

template <typename Real>
bool runTests(const Grid<Real> &grid) {
  using C = Complex<Real>;
  const Real mass = Real(0.10);
  const Real csw = Real(1.0);
  const Real atol = sizeof(Real) == sizeof(float) ? Real(1e-7) : Real(1e-12);

  std::array<int32_t, kupl::qcu::kParamsSize> params{};
  std::array<Real, kupl::qcu::kArgvSize> argv{};
  std::array<int64_t, kupl::qcu::kSetPtrsSize> set_ptrs{};
  params[kupl::qcu::kLatX] = grid.local[0];
  params[kupl::qcu::kLatY] = grid.local[1];
  params[kupl::qcu::kLatZ] = grid.local[2];
  params[kupl::qcu::kLatT] = grid.local[3];
  params[kupl::qcu::kLatXyzt] = static_cast<int32_t>(grid.local_volume);
  params[kupl::qcu::kGridX] = grid.grid[0];
  params[kupl::qcu::kGridY] = grid.grid[1];
  params[kupl::qcu::kGridZ] = grid.grid[2];
  params[kupl::qcu::kGridT] = grid.grid[3];
  params[kupl::qcu::kParity] = 0;
  params[kupl::qcu::kNodeRank] = grid.rank;
  params[kupl::qcu::kNodeSize] = grid.size;
  params[kupl::qcu::kMaxIter] =
      std::getenv("KUPL_TEST_MAX_ITER") == nullptr
          ? 300
          : std::atoi(std::getenv("KUPL_TEST_MAX_ITER"));
  params[kupl::qcu::kDataType] =
      sizeof(Real) == sizeof(float) ? kupl::qcu::kLatC64
                                    : kupl::qcu::kLatC128;
  params[kupl::qcu::kSetPlan] = kupl::qcu::kSetPlanClover;
  params[kupl::qcu::kVerbose] =
      std::getenv("KUPL_TEST_VERBOSE") == nullptr
          ? 0
          : (std::atoi(std::getenv("KUPL_TEST_VERBOSE")) != 0 ? 1 : 0);
  argv[kupl::qcu::kMass] = mass;
  argv[kupl::qcu::kAtol] = atol;
  argv[kupl::qcu::kSigma] = Real(0.1);
  argv[kupl::qcu::kCsw] = csw;

  auto gauge = makeLocalGauge(grid);
  auto full_source = makeLocalFullField(grid);
  auto global_source = makeGlobalField(grid, full_source);
  const int target =
      std::getenv("KUPL_TEST_TARGET_PARITY") == nullptr
          ? 0
          : std::atoi(std::getenv("KUPL_TEST_TARGET_PARITY"));
  const int other = 1 - target;
  const std::size_t block = static_cast<std::size_t>(grid.parity_vectors);
  std::vector<C> compact_input(
      full_source.begin() + static_cast<std::ptrdiff_t>(other * block),
      full_source.begin() + static_cast<std::ptrdiff_t>((other + 1) * block));
  std::vector<C> compact_output(block);
  std::vector<C> clover_even(static_cast<std::size_t>(grid.half_sites * kOnsite));
  std::vector<C> clover_odd(clover_even.size());
  std::vector<C> clover_single(clover_even.size());
  std::vector<C> inverse_even(clover_even.size());
  std::vector<C> inverse_odd(clover_even.size());
  std::vector<C> inverse_single(clover_even.size());

  applyInitQcu(reinterpret_cast<long long>(set_ptrs.data()),
               reinterpret_cast<long long>(params.data()),
               reinterpret_cast<long long>(argv.data()));

  params[kupl::qcu::kSetPlan] = kupl::qcu::kSetPlanWilson;
  params[kupl::qcu::kParity] = target;
  applyWilsonDslashQcu(reinterpret_cast<long long>(compact_output.data()),
                        reinterpret_cast<long long>(compact_input.data()),
                        reinterpret_cast<long long>(gauge.data()),
                        reinterpret_cast<long long>(set_ptrs.data()),
                        reinterpret_cast<long long>(params.data()));

  std::vector<C> expected(block);
  for (int64_t site = 0; site < grid.half_sites; ++site) {
    int x = 0;
    int y = 0;
    int z = 0;
    int t2 = 0;
    decodeHalf(grid, site, x, y, z, t2);
    const int t = localTime<Real>(x, y, z, t2, target);
    C site_result[kVector];
    referenceBareDslash(grid, grid.globalCoordinate(0, x),
                        grid.globalCoordinate(1, y),
                        grid.globalCoordinate(2, z),
                        grid.globalCoordinate(3, t), global_source, site_result);
    for (int component = 0; component < kVector; ++component) {
      expected[static_cast<std::size_t>(canonicalCompactFermionIndex(
          component, site, grid.half_sites))] = site_result[component];
    }
  }
  const Real wilson_error = localRelativeError(compact_output, expected);

  params[kupl::qcu::kSetPlan] = kupl::qcu::kSetPlanSolver;
  params[kupl::qcu::kParity] = target;
  std::vector<C> wilson_schur_input(block);
  for (int component = 0; component < kVector; ++component) {
    for (int64_t site = 0; site < grid.half_sites; ++site) {
      const std::size_t index = static_cast<std::size_t>(
          canonicalCompactFermionIndex(component, site, grid.half_sites));
      wilson_schur_input[index] =
          full_source[static_cast<std::size_t>(canonicalFermionIndex(
              target, component, site, grid.half_sites))];
    }
  }
  std::vector<C> wilson_schur_output(block);
  applyWilsonBistabCgDslashQcu(
      reinterpret_cast<long long>(wilson_schur_output.data()),
      reinterpret_cast<long long>(wilson_schur_input.data()),
      reinterpret_cast<long long>(gauge.data()),
      reinterpret_cast<long long>(set_ptrs.data()),
      reinterpret_cast<long long>(params.data()));
  const auto expected_wilson_schur =
      referenceWilsonSchur(grid, wilson_schur_input, target, mass);
  const Real wilson_schur_error =
      localRelativeError(wilson_schur_output, expected_wilson_schur);

  std::vector<C> wilson_probe(block);
  for (std::size_t index = 0; index < block; ++index) {
    const Real phase = Real(0.013) * static_cast<Real>(index + 1);
    wilson_probe[index] =
        C(std::cos(phase), std::sin(Real(0.7) * phase));
  }
  std::vector<C> probe_schur(block);
  applyWilsonBistabCgDslashQcu(
      reinterpret_cast<long long>(probe_schur.data()),
      reinterpret_cast<long long>(wilson_probe.data()),
      reinterpret_cast<long long>(gauge.data()),
      reinterpret_cast<long long>(set_ptrs.data()),
      reinterpret_cast<long long>(params.data()));
  std::vector<C> normal_output(block);
  applyWilsonCgDslashQcu(
      reinterpret_cast<long long>(normal_output.data()),
      reinterpret_cast<long long>(wilson_schur_input.data()),
      reinterpret_cast<long long>(gauge.data()),
      reinterpret_cast<long long>(set_ptrs.data()),
      reinterpret_cast<long long>(params.data()));
  const C normal_left =
      globalCompactDot(probe_schur, wilson_schur_output);
  const C normal_right =
      globalCompactDot(wilson_probe, normal_output);
  const Real normal_adjoint_error =
      std::abs(normal_left - normal_right) /
      std::max(std::abs(normal_left), Real(1e-30));

  params[kupl::qcu::kSetPlan] = kupl::qcu::kSetPlanClover;
  params[kupl::qcu::kParity] = 0;
  applyCloverQcu(reinterpret_cast<long long>(clover_single.data()),
                  reinterpret_cast<long long>(gauge.data()),
                  reinterpret_cast<long long>(set_ptrs.data()),
                  reinterpret_cast<long long>(params.data()));
  applyCloversQcu(reinterpret_cast<long long>(clover_even.data()),
                   reinterpret_cast<long long>(inverse_even.data()),
                   reinterpret_cast<long long>(gauge.data()),
                   reinterpret_cast<long long>(set_ptrs.data()),
                   reinterpret_cast<long long>(params.data()));
  params[kupl::qcu::kParity] = 1;
  applyCloversQcu(reinterpret_cast<long long>(clover_odd.data()),
                   reinterpret_cast<long long>(inverse_odd.data()),
                   reinterpret_cast<long long>(gauge.data()),
                   reinterpret_cast<long long>(set_ptrs.data()),
                   reinterpret_cast<long long>(params.data()));

  std::vector<C> expected_clover(block * kOnsite);
  for (int64_t site = 0; site < grid.half_sites; ++site) {
    int x = 0;
    int y = 0;
    int z = 0;
    int t2 = 0;
    decodeHalf(grid, site, x, y, z, t2);
    const int t = localTime<Real>(x, y, z, t2, target);
    writeCanonicalClover(grid, grid.globalCoordinate(0, x),
                         grid.globalCoordinate(1, y),
                         grid.globalCoordinate(2, z),
                         grid.globalCoordinate(3, t), site, mass, csw,
                         expected_clover.data());
  }
  const Real clover_error = localRelativeError(clover_even, expected_clover);
  const Real single_clover_error =
      localRelativeError(clover_single, expected_clover);

  std::vector<C> identity(static_cast<std::size_t>(grid.half_sites * kOnsite),
                          C(0));
  for (int64_t site = 0; site < grid.half_sites; ++site) {
    for (int row = 0; row < kVector; ++row) {
      identity[static_cast<std::size_t>(canonicalCloverIndex(
          row, row, site, grid.half_sites))] = C(1);
    }
  }
  std::vector<C> product(identity.size());
  Real inverse_difference = 0;
  Real inverse_norm = 0;
  for (int64_t site = 0; site < grid.half_sites; ++site) {
    for (int row = 0; row < kVector; ++row) {
      for (int column = 0; column < kVector; ++column) {
        C value(0);
        for (int inner = 0; inner < kVector; ++inner) {
          const C a = clover_even[static_cast<std::size_t>(
              canonicalCloverIndex(row, inner, site, grid.half_sites))];
          const C inverse_entry = inverse_even[static_cast<std::size_t>(
              canonicalCloverIndex(inner, column, site, grid.half_sites))];
          value += a * inverse_entry;
        }
        product[static_cast<std::size_t>(canonicalCloverIndex(
            row, column, site, grid.half_sites))] = value;
      }
    }
  }
  for (std::size_t index = 0; index < product.size(); ++index) {
    inverse_difference += std::norm(product[index] - identity[index]);
    inverse_norm += std::norm(identity[index]);
  }
  Real global_inverse_difference = 0;
  Real global_inverse_norm = 0;
  const MPI_Datatype mpi_real =
      sizeof(Real) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE;
  MPI_Allreduce(&inverse_difference, &global_inverse_difference, 1, mpi_real,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&inverse_norm, &global_inverse_norm, 1, mpi_real, MPI_SUM,
                MPI_COMM_WORLD);
  const Real inverse_error =
      std::sqrt(global_inverse_difference /
                std::max(global_inverse_norm, Real(1e-30)));

  params[kupl::qcu::kSetPlan] = kupl::qcu::kSetPlanClover;
  params[kupl::qcu::kParity] = target;
  applyCloverDslashQcu(reinterpret_cast<long long>(compact_output.data()),
                        reinterpret_cast<long long>(compact_input.data()),
                        reinterpret_cast<long long>(gauge.data()),
                        reinterpret_cast<long long>(set_ptrs.data()),
                        reinterpret_cast<long long>(params.data()));
  std::vector<C> expected_clover_dslash(block);
  for (int64_t site = 0; site < grid.half_sites; ++site) {
    C canonical_onsite[kOnsite];
    for (int row = 0; row < kVector; ++row) {
      for (int column = 0; column < kVector; ++column) {
        canonical_onsite[row * kVector + column] =
            expected_clover[static_cast<std::size_t>(canonicalCloverIndex(
                row, column, site, grid.half_sites))];
      }
    }
    C inverse_matrix[144];
    if (!invertOnsite(canonical_onsite, inverse_matrix)) {
      throw std::runtime_error("reference onsite matrix is singular");
    }
    C local_hopping[kVector];
    C local_result[kVector];
    for (int component = 0; component < kVector; ++component) {
      local_hopping[component] =
          expected[static_cast<std::size_t>(canonicalCompactFermionIndex(
              component, site, grid.half_sites))];
    }
    multiplyOnsite(inverse_matrix, local_hopping, local_result);
    for (int component = 0; component < kVector; ++component) {
      expected_clover_dslash[static_cast<std::size_t>(
          canonicalCompactFermionIndex(component, site,
                                       grid.half_sites))] =
          local_result[component];
    }
  }
  const Real clover_dslash_error =
      localRelativeError(compact_output, expected_clover_dslash);

  const C *target_inverse =
      target == 0 ? inverse_even.data() : inverse_odd.data();
  applyDslashQcu(reinterpret_cast<long long>(compact_output.data()),
                 reinterpret_cast<long long>(compact_input.data()),
                 reinterpret_cast<long long>(gauge.data()),
                 reinterpret_cast<long long>(target_inverse),
                 reinterpret_cast<long long>(set_ptrs.data()),
                 reinterpret_cast<long long>(params.data()));
  const Real supplied_dslash_error =
      localRelativeError(compact_output, expected_clover_dslash);

  std::vector<C> schur_input(block);
  for (int component = 0; component < kVector; ++component) {
    for (int64_t site = 0; site < grid.half_sites; ++site) {
      const std::size_t compact_index = static_cast<std::size_t>(
          canonicalCompactFermionIndex(component, site, grid.half_sites));
      schur_input[compact_index] =
          full_source[static_cast<std::size_t>(canonicalFermionIndex(
              target, component, site, grid.half_sites))];
    }
  }
  const auto global_schur_input =
      makeGlobalParityField(grid, schur_input, target);
  std::vector<C> local_d_qp(block);
  for (int64_t site = 0; site < grid.half_sites; ++site) {
    int x = 0;
    int y = 0;
    int z = 0;
    int t2 = 0;
    decodeHalf(grid, site, x, y, z, t2);
    const int t = localTime<Real>(x, y, z, t2, other);
    C hopping[kVector];
    referenceBareDslash(
        grid, grid.globalCoordinate(0, x), grid.globalCoordinate(1, y),
        grid.globalCoordinate(2, z), grid.globalCoordinate(3, t),
        global_schur_input, hopping);
    for (int component = 0; component < kVector; ++component) {
      local_d_qp[static_cast<std::size_t>(canonicalCompactFermionIndex(
          component, site, grid.half_sites))] = hopping[component];
    }
  }
  const auto global_d_qp =
      makeGlobalParityField(grid, local_d_qp, other);
  std::vector<C> local_preconditioned_q(block);
  for (int64_t site = 0; site < grid.half_sites; ++site) {
    int x = 0;
    int y = 0;
    int z = 0;
    int t2 = 0;
    decodeHalf(grid, site, x, y, z, t2);
    const int t = localTime<Real>(x, y, z, t2, other);
    const int gx = grid.globalCoordinate(0, x);
    const int gy = grid.globalCoordinate(1, y);
    const int gz = grid.globalCoordinate(2, z);
    const int gt = grid.globalCoordinate(3, t);
    C onsite[kOnsite];
    C inverse_onsite[kOnsite];
    referenceClover(grid, gx, gy, gz, gt, mass, csw, onsite);
    if (!invertOnsite(onsite, inverse_onsite)) {
      throw std::runtime_error("reference Schur onsite matrix is singular");
    }
    const int64_t source_base =
        grid.fullGlobalIndex(gx, gy, gz, gt) * kVector;
    C source[kVector];
    std::copy(global_d_qp.begin() + source_base,
              global_d_qp.begin() + source_base + kVector, source);
    C result[kVector];
    multiplyOnsite(inverse_onsite, source, result);
    for (int component = 0; component < kVector; ++component) {
      local_preconditioned_q[static_cast<std::size_t>(
          canonicalCompactFermionIndex(component, site,
                                       grid.half_sites))] =
          result[component];
    }
  }
  const auto global_preconditioned_q =
      makeGlobalParityField(grid, local_preconditioned_q, other);
  std::vector<C> expected_schur(block);
  const Real kappa_value = Real(1) / (Real(2) * mass + Real(8));
  for (int64_t site = 0; site < grid.half_sites; ++site) {
    int x = 0;
    int y = 0;
    int z = 0;
    int t2 = 0;
    decodeHalf(grid, site, x, y, z, t2);
    const int t = localTime<Real>(x, y, z, t2, target);
    const int gx = grid.globalCoordinate(0, x);
    const int gy = grid.globalCoordinate(1, y);
    const int gz = grid.globalCoordinate(2, z);
    const int gt = grid.globalCoordinate(3, t);
    C onsite[kOnsite];
    referenceClover(grid, gx, gy, gz, gt, mass, csw, onsite);
    C local_input[kVector];
    for (int component = 0; component < kVector; ++component) {
      local_input[component] =
          schur_input[static_cast<std::size_t>(canonicalCompactFermionIndex(
              component, site, grid.half_sites))];
    }
    C onsite_result[kVector];
    C hopping[kVector];
    multiplyOnsite(onsite, local_input, onsite_result);
    referenceBareDslash(grid, gx, gy, gz, gt, global_preconditioned_q,
                        hopping);
    for (int component = 0; component < kVector; ++component) {
      expected_schur[static_cast<std::size_t>(canonicalCompactFermionIndex(
          component, site, grid.half_sites))] =
          onsite_result[component] - kappa_value * kappa_value *
                                         hopping[component];
    }
  }
  params[kupl::qcu::kSetPlan] = kupl::qcu::kSetPlanSolver;
  params[kupl::qcu::kParity] = target;
  std::vector<C> schur_output(block);
  applyCloverBistabCgDslashQcu(
      reinterpret_cast<long long>(schur_output.data()),
      reinterpret_cast<long long>(schur_input.data()),
      reinterpret_cast<long long>(gauge.data()),
      reinterpret_cast<long long>(clover_even.data()),
      reinterpret_cast<long long>(clover_odd.data()),
      reinterpret_cast<long long>(inverse_even.data()),
      reinterpret_cast<long long>(inverse_odd.data()),
      reinterpret_cast<long long>(set_ptrs.data()),
      reinterpret_cast<long long>(params.data()));
  const Real schur_error =
      localRelativeError(schur_output, expected_schur);

  params[kupl::qcu::kSetPlan] = kupl::qcu::kSetPlanSolver;
  params[kupl::qcu::kParity] = target;
  params[kupl::qcu::kUseInitialGuess] = 0;
  std::vector<C> solution(full_source.size(), C(0));
  applyCloverBistabCgQcu(
      reinterpret_cast<long long>(solution.data()),
      reinterpret_cast<long long>(full_source.data()),
      reinterpret_cast<long long>(gauge.data()),
      reinterpret_cast<long long>(clover_even.data()),
      reinterpret_cast<long long>(clover_odd.data()),
      reinterpret_cast<long long>(inverse_even.data()),
      reinterpret_cast<long long>(inverse_odd.data()),
      reinterpret_cast<long long>(set_ptrs.data()),
      reinterpret_cast<long long>(params.data()));
  const int iterations =
      getCloverBistabCgIterationsQcu(reinterpret_cast<long long>(set_ptrs.data()),
                                     reinterpret_cast<long long>(params.data()));
  auto global_solution = makeGlobalField(grid, solution);
  const Real residual =
      trueResidual(grid, mass, csw, global_solution, full_source);

  params[kupl::qcu::kUseInitialGuess] = 0;
  std::vector<C> wilson_bistab_solution(full_source.size(), C(0));
  applyWilsonBistabCgQcu(
      reinterpret_cast<long long>(wilson_bistab_solution.data()),
      reinterpret_cast<long long>(full_source.data()),
      reinterpret_cast<long long>(gauge.data()),
      reinterpret_cast<long long>(set_ptrs.data()),
      reinterpret_cast<long long>(params.data()));
  const auto global_wilson_bistab_solution =
      makeGlobalField(grid, wilson_bistab_solution);
  const Real wilson_bistab_residual =
      trueResidual(grid, mass, Real(0), global_wilson_bistab_solution,
                   full_source);

  std::vector<C> wilson_cg_solution(full_source.size(), C(0));
  applyWilsonCgQcu(
      reinterpret_cast<long long>(wilson_cg_solution.data()),
      reinterpret_cast<long long>(full_source.data()),
      reinterpret_cast<long long>(gauge.data()),
      reinterpret_cast<long long>(set_ptrs.data()),
      reinterpret_cast<long long>(params.data()));
  const auto global_wilson_cg_solution =
      makeGlobalField(grid, wilson_cg_solution);
  const Real wilson_cg_residual =
      trueResidual(grid, mass, Real(0), global_wilson_cg_solution,
                   full_source);

  std::vector<C> compact_rhs(block, C(0));
  const int prepare_status = applyCloverBistabCgPrepareQcu(
      reinterpret_cast<long long>(compact_rhs.data()),
      reinterpret_cast<long long>(full_source.data()),
      reinterpret_cast<long long>(gauge.data()),
      reinterpret_cast<long long>(clover_even.data()),
      reinterpret_cast<long long>(clover_odd.data()),
      reinterpret_cast<long long>(inverse_even.data()),
      reinterpret_cast<long long>(inverse_odd.data()),
      reinterpret_cast<long long>(set_ptrs.data()),
      reinterpret_cast<long long>(params.data()));
  std::vector<C> reconstructed(full_source.size(), C(0));
  const C *solution_target =
      solution.data() + static_cast<std::ptrdiff_t>(target * block);
  const int reconstruct_status = applyCloverBistabCgReconstructQcu(
      reinterpret_cast<long long>(reconstructed.data()),
      reinterpret_cast<long long>(full_source.data()),
      reinterpret_cast<long long>(solution_target),
      reinterpret_cast<long long>(gauge.data()),
      reinterpret_cast<long long>(clover_even.data()),
      reinterpret_cast<long long>(clover_odd.data()),
      reinterpret_cast<long long>(inverse_even.data()),
      reinterpret_cast<long long>(inverse_odd.data()),
      reinterpret_cast<long long>(set_ptrs.data()),
      reinterpret_cast<long long>(params.data()));
  const int null_prepare_status = applyCloverBistabCgPrepareQcu(
      reinterpret_cast<long long>(compact_rhs.data()),
      reinterpret_cast<long long>(full_source.data()), 0,
      reinterpret_cast<long long>(clover_even.data()),
      reinterpret_cast<long long>(clover_odd.data()),
      reinterpret_cast<long long>(inverse_even.data()),
      reinterpret_cast<long long>(inverse_odd.data()),
      reinterpret_cast<long long>(set_ptrs.data()),
      reinterpret_cast<long long>(params.data()));
  const Real reconstruct_error =
      localRelativeError(reconstructed, solution);

  applyEndQcu(reinterpret_cast<long long>(set_ptrs.data()),
              reinterpret_cast<long long>(params.data()));

  argv[kupl::qcu::kCsw] = Real(0);
  params[kupl::qcu::kSetPlan] = kupl::qcu::kSetPlanClover;
  params[kupl::qcu::kParity] = 0;
  applyInitQcu(reinterpret_cast<long long>(set_ptrs.data()),
               reinterpret_cast<long long>(params.data()),
               reinterpret_cast<long long>(argv.data()));
  applyCloversQcu(reinterpret_cast<long long>(clover_single.data()),
                   reinterpret_cast<long long>(inverse_single.data()),
                   reinterpret_cast<long long>(gauge.data()),
                   reinterpret_cast<long long>(set_ptrs.data()),
                   reinterpret_cast<long long>(params.data()));
  const Real csw0_identity_error =
      localRelativeError(clover_single, identity);
  const Real csw0_inverse_identity_error =
      localRelativeError(inverse_single, identity);
  applyEndQcu(reinterpret_cast<long long>(set_ptrs.data()),
              reinterpret_cast<long long>(params.data()));

  const Real wilson_tolerance =
      sizeof(Real) == sizeof(float) ? Real(3e-5) : Real(2e-11);
  const Real clover_tolerance =
      sizeof(Real) == sizeof(float) ? Real(2e-4) : Real(2e-10);
  const Real inverse_tolerance =
      sizeof(Real) == sizeof(float) ? Real(2e-4) : Real(2e-10);
  const Real residual_tolerance =
      sizeof(Real) == sizeof(float) ? Real(2e-4) : Real(2e-9);
  const Real dslash_supplied_tolerance =
      sizeof(Real) == sizeof(float) ? Real(2e-4) : Real(2e-10);
  const Real schur_tolerance =
      sizeof(Real) == sizeof(float) ? Real(3e-4) : Real(3e-9);
  const Real wilson_normal_tolerance =
      sizeof(Real) == sizeof(float) ? Real(2e-3) : Real(2e-11);
  const bool pass = wilson_error < wilson_tolerance &&
                    clover_error < clover_tolerance &&
                    single_clover_error < clover_tolerance &&
                    inverse_error < inverse_tolerance &&
                    clover_dslash_error < clover_tolerance &&
                    supplied_dslash_error < dslash_supplied_tolerance &&
                    schur_error < schur_tolerance &&
                    wilson_schur_error < schur_tolerance &&
                    normal_adjoint_error < wilson_normal_tolerance &&
                    csw0_identity_error < inverse_tolerance &&
                    csw0_inverse_identity_error < inverse_tolerance &&
                    residual < residual_tolerance &&
                    wilson_bistab_residual < residual_tolerance &&
                    wilson_cg_residual < residual_tolerance &&
                    reconstruct_error < schur_tolerance &&
                    prepare_status == 0 && reconstruct_status == 0 &&
                    null_prepare_status != 0 &&
                    iterations > 0 &&
                    iterations < params[kupl::qcu::kMaxIter];

  if (grid.rank == 0) {
    std::printf(
        "type=%s ranks=%d grid=%d,%d,%d,%d local=%d,%d,%d,%d "
        "wilson_rel=%.9e clover_rel=%.9e inverse_rel=%.9e "
        "clover_single_rel=%.9e clover_dslash_rel=%.9e "
        "supplied_dslash_rel=%.9e schur_rel=%.9e "
        "wilson_schur_rel=%.9e wilson_normal_adjoint=%.9e "
        "csw0_identity_rel=%.9e csw0_inverse_identity_rel=%.9e "
        "clover_true_res=%.9e wilson_bistab_true_res=%.9e "
        "wilson_cg_true_res=%.9e prepare_status=%d reconstruct_status=%d "
        "null_prepare_status=%d reconstruct_rel=%.9e iterations=%d "
        "status=%s\n",
        sizeof(Real) == sizeof(float) ? "c64" : "c128", grid.size,
        grid.grid[0], grid.grid[1], grid.grid[2], grid.grid[3],
        grid.local[0], grid.local[1], grid.local[2], grid.local[3],
        static_cast<double>(wilson_error), static_cast<double>(clover_error),
        static_cast<double>(inverse_error),
        static_cast<double>(single_clover_error),
        static_cast<double>(clover_dslash_error),
        static_cast<double>(supplied_dslash_error),
        static_cast<double>(schur_error),
        static_cast<double>(wilson_schur_error),
        static_cast<double>(normal_adjoint_error),
        static_cast<double>(csw0_identity_error),
        static_cast<double>(csw0_inverse_identity_error),
        static_cast<double>(residual),
        static_cast<double>(wilson_bistab_residual),
        static_cast<double>(wilson_cg_residual), prepare_status,
        reconstruct_status, null_prepare_status,
        static_cast<double>(reconstruct_error),
        iterations, pass ? "PASS" : "FAIL");
  }
  int local_pass = pass ? 1 : 0;
  int global_pass = 0;
  MPI_Allreduce(&local_pass, &global_pass, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  return global_pass != 0;
}

std::array<int, kDim> parseVector(int argc, char **argv, const char *flag,
                                  const std::array<int, kDim> &fallback) {
  std::array<int, kDim> result = fallback;
  for (int index = 1; index + 4 < argc; ++index) {
    if (std::strcmp(argv[index], flag) == 0) {
      for (int dim = 0; dim < kDim; ++dim) {
        result[dim] = std::atoi(argv[index + 1 + dim]);
      }
      break;
    }
  }
  return result;
}

bool hasFlag(int argc, char **argv, const char *flag) {
  for (int index = 1; index < argc; ++index) {
    if (std::strcmp(argv[index], flag) == 0) {
      return true;
    }
  }
  return false;
}

} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int status = 1;
  try {
    const auto local =
        parseVector(argc, argv, "--lattice", {4, 2, 2, 4});
    const auto process_grid =
        parseVector(argc, argv, "--grid", {1, 1, 1, 1});
    bool pass = true;
    if (hasFlag(argc, argv, "--laplacian")) {
      {
        Grid<float> grid(local, process_grid);
        pass = runLaplacianTests(grid);
      }
      {
        Grid<double> grid(local, process_grid);
        pass = runLaplacianTests(grid) && pass;
      }
    } else {
      {
        Grid<float> grid(local, process_grid);
        pass = runTests(grid);
      }
      {
        Grid<double> grid(local, process_grid);
        pass = runTests(grid) && pass;
      }
    }
    status = pass ? 0 : 1;
    if (rank == 0) {
      std::printf("overall_status=%s\n", pass ? "PASS" : "FAIL");
    }
  } catch (const std::exception &error) {
    std::fprintf(stderr, "test exception: %s\n", error.what());
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  MPI_Finalize();
  return status;
}
