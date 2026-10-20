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
                                    Complex<Real> *output) const {
  const int64_t count = geometry_.half_sites;
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t site = 0; site < count; ++site) {
    Complex<Real> source[kVector];
    Complex<Real> local[kVector];
    gather_vector(compact_view(input, site), source);
    for (int row = 0; row < kVector; ++row) {
      Complex<Real> value(0);
      for (int column = 0; column < kVector; ++column) {
        value += matrix[static_cast<int64_t>(row * kVector + column) *
                            geometry_.half_sites +
                        site] *
                 source[column];
      }
      local[row] = value;
    }
    scatter_vector(local, output, site);
  }
}

template class LatticeSet<float>;
template class LatticeSet<double>;

} // namespace kupl::qcu
