#ifndef KUPL_QCU_LATTICE_COMPLEX_H
#define KUPL_QCU_LATTICE_COMPLEX_H

#include <complex>

namespace kupl::qcu {

template <typename Real> using Complex = std::complex<Real>;

inline constexpr int kDim = 4;
inline constexpr int kSpin = 4;
inline constexpr int kColor = 3;
inline constexpr int kVector = kSpin * kColor;
inline constexpr int kOnsite = kVector * kVector;
inline constexpr int kGauge = kColor * kColor;

} // namespace kupl::qcu

#endif
