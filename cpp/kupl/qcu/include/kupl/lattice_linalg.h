#ifndef KUPL_QCU_LATTICE_LINALG_H
#define KUPL_QCU_LATTICE_LINALG_H

#include "kupl/lattice_complex.h"

#include <algorithm>

namespace kupl::qcu {

template <typename Real>
inline Complex<Real> gamma_entry(int direction, int row, int column) {
  using C = Complex<Real>;
  const C i(0, 1);
  if (direction == 0) {
    static const C matrix[16] = {
        C(0), C(0), C(0), i, C(0), C(0), i, C(0),
        C(0), -i, C(0), C(0), -i, C(0), C(0), C(0),
    };
    return matrix[row * 4 + column];
  }
  if (direction == 1) {
    static const C matrix[16] = {
        C(0), C(0), C(0), C(-1), C(0), C(0), C(1), C(0),
        C(0), C(1), C(0), C(0), C(-1), C(0), C(0), C(0),
    };
    return matrix[row * 4 + column];
  }
  if (direction == 2) {
    static const C matrix[16] = {
        C(0), C(0), i, C(0), C(0), C(0), C(0), -i,
        -i, C(0), C(0), C(0), C(0), i, C(0), C(0),
    };
    return matrix[row * 4 + column];
  }
  static const C matrix[16] = {
      C(0), C(0), C(1), C(0), C(0), C(0), C(0), C(1),
      C(1), C(0), C(0), C(0), C(0), C(1), C(0), C(0),
  };
  return matrix[row * 4 + column];
}

template <typename Real>
inline Complex<Real> multiply_complex(const Complex<Real> &left,
                                      const Complex<Real> &right) {
  return Complex<Real>(
      left.real() * right.real() - left.imag() * right.imag(),
      left.real() * right.imag() + left.imag() * right.real());
}

template <typename Real>
inline void multiply_su3(const Complex<Real> *left,
                         const Complex<Real> *right, Complex<Real> *out) {
  for (int row = 0; row < kColor; ++row) {
    for (int column = 0; column < kColor; ++column) {
      Real real = Real(0);
      Real imag = Real(0);
      for (int inner = 0; inner < kColor; ++inner) {
        const Complex<Real> &a = left[row * kColor + inner];
        const Complex<Real> &b = right[inner * kColor + column];
        real += a.real() * b.real() - a.imag() * b.imag();
        imag += a.real() * b.imag() + a.imag() * b.real();
      }
      out[row * kColor + column] = Complex<Real>(real, imag);
    }
  }
}

} // namespace kupl::qcu

#endif
