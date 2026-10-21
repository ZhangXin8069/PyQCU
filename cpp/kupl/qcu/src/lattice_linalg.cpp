#include "kupl/lattice_set.h"

namespace kupl::qcu {

template <typename Real>
void LatticeSet<Real>::load_gauge_matrix(const GaugeView<Real> &source,
                                         Complex<Real> matrix[9]) const {
  for (int row = 0; row < kColor; ++row) {
    for (int column = 0; column < kColor; ++column) {
      matrix[row * kColor + column] =
          source.data[(row * kColor + column) * source.matrix_stride];
    }
  }
}

template <typename Real>
void LatticeSet<Real>::apply_onsite(const Complex<Real> *input,
                                    const Complex<Real> *matrix,
                                    Complex<Real> *output,
                                    const Complex<Real> *subtract,
                                    Real subtract_scale) const {
  using C = Complex<Real>;
  const int64_t count = geometry_.half_sites;
  // Keep each site tile at 128 bytes: 16 c64 or 8 c128 sites.
  constexpr int kSiteLanes = 64 / sizeof(Real);
  const int64_t blocks = (count + kSiteLanes - 1) / kSiteLanes;
#pragma omp parallel for schedule(static)
  for (int64_t block = 0; block < blocks; ++block) {
    const int64_t site = block * kSiteLanes;
    const int lanes = static_cast<int>(
        std::min<int64_t>(kSiteLanes, count - site));
    C source[kVector][kSiteLanes];

    // Copy the whole site block first so input and output may alias.
    for (int column = 0; column < kVector; ++column) {
      const int64_t base = static_cast<int64_t>(column) * count + site;
#pragma omp simd
      for (int lane = 0; lane < lanes; ++lane) {
        source[column][lane] = input[base + lane];
      }
    }

    for (int row = 0; row < kVector; ++row) {
      Real real[kSiteLanes] = {};
      Real imag[kSiteLanes] = {};
      for (int column = 0; column < kVector; ++column) {
        const int64_t base =
            static_cast<int64_t>(row * kVector + column) * count + site;
#pragma omp simd
        for (int lane = 0; lane < lanes; ++lane) {
          const C &entry = matrix[base + lane];
          const C &vector = source[column][lane];
          real[lane] += entry.real() * vector.real() -
                        entry.imag() * vector.imag();
          imag[lane] += entry.real() * vector.imag() +
                        entry.imag() * vector.real();
        }
      }
      const int64_t base = static_cast<int64_t>(row) * count + site;
#pragma omp simd
      for (int lane = 0; lane < lanes; ++lane) {
        C value(real[lane], imag[lane]);
        if (subtract != nullptr) {
          value -= subtract_scale * subtract[base + lane];
        }
        output[base + lane] = value;
      }
    }
  }
}

template <typename Real>
void LatticeSet<Real>::apply_clover_onsite(
    const Complex<Real> *input, const Complex<Real> *matrix,
    Complex<Real> *output, const Complex<Real> *subtract,
    Real subtract_scale) const {
  using C = Complex<Real>;
  const int64_t count = geometry_.half_sites;
  constexpr int kSiteLanes = 64 / sizeof(Real);
  const int64_t blocks = (count + kSiteLanes - 1) / kSiteLanes;
#pragma omp parallel for schedule(static)
  for (int64_t block = 0; block < blocks; ++block) {
    const int64_t site = block * kSiteLanes;
    const int lanes = static_cast<int>(
        std::min<int64_t>(kSiteLanes, count - site));
    C source[kVector][kSiteLanes];
    for (int column = 0; column < kVector; ++column) {
      const int64_t base = static_cast<int64_t>(column) * count + site;
#pragma omp simd
      for (int lane = 0; lane < lanes; ++lane) {
        source[column][lane] = input[base + lane];
      }
    }

    // In this gamma basis the Wilson-Clover onsite matrix and its inverse are
    // block diagonal for spins (0,1) and (2,3).  Apply two dense 6x6 blocks
    // instead of reading the 72 structural zeros in the generic 12x12 path.
    for (int spin_block = 0; spin_block < 2; ++spin_block) {
      const int spins[2] = {spin_block == 0 ? 0 : 2,
                            spin_block == 0 ? 1 : 3};
      for (int local_row = 0; local_row < 6; ++local_row) {
        const int spin_row = spins[local_row / kColor];
        const int color_row = local_row % kColor;
        const int row = spin_row * kColor + color_row;
        Real real[kSiteLanes] = {};
        Real imag[kSiteLanes] = {};
        for (int local_column = 0; local_column < 6; ++local_column) {
          const int spin_column = spins[local_column / kColor];
          const int color_column = local_column % kColor;
          const int column = spin_column * kColor + color_column;
          const int64_t base =
              static_cast<int64_t>(row * kVector + column) * count + site;
#pragma omp simd
          for (int lane = 0; lane < lanes; ++lane) {
            const C &entry = matrix[base + lane];
            const C &vector = source[column][lane];
            real[lane] += entry.real() * vector.real() -
                          entry.imag() * vector.imag();
            imag[lane] += entry.real() * vector.imag() +
                          entry.imag() * vector.real();
          }
        }
        const int64_t base = static_cast<int64_t>(row) * count + site;
#pragma omp simd
        for (int lane = 0; lane < lanes; ++lane) {
          C value(real[lane], imag[lane]);
          if (subtract != nullptr) {
            value -= subtract_scale * subtract[base + lane];
          }
          output[base + lane] = value;
        }
      }
    }
  }
}

template class LatticeSet<float>;
template class LatticeSet<double>;

} // namespace kupl::qcu
