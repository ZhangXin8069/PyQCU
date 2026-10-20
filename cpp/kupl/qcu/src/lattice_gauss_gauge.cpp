#include "kupl/lattice_linalg.h"
#include "kupl/lattice_set.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace kupl::qcu {
namespace {

inline uint64_t mix64(uint64_t value) {
  value += UINT64_C(0x9e3779b97f4a7c15);
  value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31U);
}

template <typename Real>
Real deterministic_uniform(uint64_t seed, uint64_t global_site, int parity,
                           int direction, int channel) {
  uint64_t key = mix64(seed ^ global_site);
  key ^= mix64((static_cast<uint64_t>(parity) * kDim + direction) * 8U +
               static_cast<uint64_t>(channel));
  const uint64_t bits = mix64(key) >> 11U;
  return static_cast<Real>(
      (static_cast<long double>(bits) + 0.5L) / 9007199254740992.0L);
}

template <typename Real>
Real deterministic_normal(uint64_t seed, uint64_t global_site, int parity,
                          int direction, int channel) {
  const Real first = std::max(
      deterministic_uniform<Real>(seed, global_site, parity, direction,
                                  2 * channel),
      std::numeric_limits<Real>::min());
  const Real second = deterministic_uniform<Real>(
      seed, global_site, parity, direction, 2 * channel + 1);
  constexpr Real two_pi = Real(6.2831853071795864769);
  return std::sqrt(Real(-2) * std::log(first)) *
         std::cos(two_pi * second);
}

template <typename Real>
Real matrix_infinity_norm(const Complex<Real> matrix[9]) {
  Real norm = 0;
  for (int row = 0; row < kColor; ++row) {
    Real row_sum = 0;
    for (int column = 0; column < kColor; ++column) {
      row_sum += std::abs(matrix[row * kColor + column]);
    }
    norm = std::max(norm, row_sum);
  }
  return norm;
}

template <typename Real>
void su3_exponential(const Complex<Real> input[9],
                     Complex<Real> output[9]) {
  using C = Complex<Real>;
  C scaled[9];
  std::copy(input, input + 9, scaled);
  int squarings = 0;
  while (matrix_infinity_norm(scaled) > Real(0.25) && squarings < 32) {
    for (int index = 0; index < 9; ++index) {
      scaled[index] *= Real(0.5);
    }
    ++squarings;
  }

  C term[9] = {C(1), C(0), C(0), C(0), C(1), C(0), C(0), C(0), C(1)};
  std::copy(term, term + 9, output);
  C next[9];
  for (int order = 1; order <= 14; ++order) {
    multiply_su3(term, scaled, next);
    const Real inverse_order = Real(1) / static_cast<Real>(order);
    for (int index = 0; index < 9; ++index) {
      term[index] = next[index] * inverse_order;
      output[index] += term[index];
    }
  }
  for (int square = 0; square < squarings; ++square) {
    multiply_su3(output, output, next);
    std::copy(next, next + 9, output);
  }
}

template <typename Real>
void make_random_su3(uint64_t seed, uint64_t global_site, int parity,
                     int direction, Real sigma, Complex<Real> output[9]) {
  using C = Complex<Real>;
  static constexpr Real gell_mann[8][9] = {
      {0, 1, 0, 1, 0, 0, 0, 0, 0},
      {0, -1, 0, 1, 0, 0, 0, 0, 0},
      {1, 0, 0, 0, -1, 0, 0, 0, 0},
      {0, 0, 1, 0, 0, 0, 1, 0, 0},
      {0, 0, -1, 0, 0, 0, 1, 0, 0},
      {0, 0, 0, 0, 0, 1, 0, 1, 0},
      {0, 0, 0, 0, 0, -1, 0, 1, 0},
      {Real(0.5773502691896257), 0, 0, 0, Real(0.5773502691896257), 0, 0,
       0, Real(-1.1547005383792515)},
  };
  C hermitian[9] = {C(0)};
  for (int generator = 0; generator < 8; ++generator) {
    const Real sample = deterministic_normal<Real>(
        seed, global_site, parity, direction, generator);
    const bool imaginary =
        generator == 1 || generator == 4 || generator == 6;
    for (int entry = 0; entry < 9; ++entry) {
      const Real coefficient = sample * gell_mann[generator][entry];
      hermitian[entry] +=
          imaginary ? C(0, coefficient) : C(coefficient, 0);
    }
  }
  C generator[9];
  for (int entry = 0; entry < 9; ++entry) {
    generator[entry] =
        C(-sigma * hermitian[entry].imag(),
          sigma * hermitian[entry].real());
  }
  su3_exponential(generator, output);
}

} // namespace

template <typename Real>
void LatticeSet<Real>::gauss_gauge(void *gauge_out) {
  require_gauge_mode();
  if (gauge_out == nullptr) {
    throw std::runtime_error("Gaussian gauge received a null output pointer");
  }
  const Real sigma = parameters_.reals[kSigma];
  if (!std::isfinite(sigma) || sigma < Real(0)) {
    throw std::runtime_error(
        "Gaussian gauge sigma must be finite and nonnegative");
  }
  auto *gauge = static_cast<Complex<Real> *>(gauge_out);
  const uint64_t seed = static_cast<uint64_t>(parameters_.get(kSeed));
  const int64_t task_count = geometry_.half_sites * 2 * kDim;
  const int64_t chunk = omp_chunk<Real>(task_count);
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
    geometry_.decode_half(site, x, y, z, t2);
    const int local_t = geometry_.time_for_parity(x, y, z, t2, parity);
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
    make_random_su3(seed, global_site, parity, direction, sigma, matrix);
    for (int row = 0; row < kColor; ++row) {
      for (int column = 0; column < kColor; ++column) {
        gauge[gauge_index(parity, row, column, direction, x, y, z, t2)] =
            matrix[row * kColor + column];
      }
    }
  }
}

template class LatticeSet<float>;
template class LatticeSet<double>;

} // namespace kupl::qcu
