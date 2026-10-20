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
  constexpr int kSiteLanes = 32 / sizeof(Real);
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

template class LatticeSet<float>;
template class LatticeSet<double>;

} // namespace kupl::qcu
