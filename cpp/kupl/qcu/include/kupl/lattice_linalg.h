#ifndef KUPL_QCU_LATTICE_LINALG_H
#define KUPL_QCU_LATTICE_LINALG_H

#include "kupl/lattice_complex.h"

#include <algorithm>

namespace kupl::qcu {

template <typename Real>
Complex<Real> gamma_entry(int direction, int row, int column) {
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
void apply_gamma_spin(int direction, const Complex<Real> *input,
                      Complex<Real> *output) {
  for (int row = 0; row < kSpin; ++row) {
    output[row] = Complex<Real>(0);
    for (int column = 0; column < kSpin; ++column) {
      output[row] +=
          gamma_entry<Real>(direction, row, column) * input[column];
    }
  }
}

template <typename Real>
void multiply_gauge_spin(const Complex<Real> *matrix,
                         const Complex<Real> in[kVector],
                         Complex<Real> out[kVector], bool dagger) {
  for (int spin = 0; spin < kSpin; ++spin) {
    for (int row = 0; row < kColor; ++row) {
      Complex<Real> value(0);
      for (int column = 0; column < kColor; ++column) {
        const Complex<Real> entry =
            dagger ? std::conj(matrix[column * kColor + row])
                   : matrix[row * kColor + column];
        value += entry * in[spin * kColor + column];
      }
      out[spin * kColor + row] = value;
    }
  }
}

template <typename Real>
void apply_one_minus_gamma(int direction,
                           const Complex<Real> in[kVector],
                           Complex<Real> out[kVector], int gamma_sign) {
  using C = Complex<Real>;
  std::copy(in, in + kVector, out);
  C gamma[kSpin];
  for (int color = 0; color < kColor; ++color) {
    C column[kSpin];
    for (int spin = 0; spin < kSpin; ++spin) {
      column[spin] = in[spin * kColor + color];
    }
    apply_gamma_spin<Real>(direction, column, gamma);
    for (int spin = 0; spin < kSpin; ++spin) {
      out[spin * kColor + color] -= static_cast<Real>(gamma_sign) * gamma[spin];
    }
  }
}

template <typename Real>
void multiply_su3(const Complex<Real> *left, const Complex<Real> *right,
                  Complex<Real> *out) {
  for (int row = 0; row < kColor; ++row) {
    for (int column = 0; column < kColor; ++column) {
      Complex<Real> value(0);
      for (int inner = 0; inner < kColor; ++inner) {
        value += left[row * kColor + inner] *
                 right[inner * kColor + column];
      }
      out[row * kColor + column] = value;
    }
  }
}

} // namespace kupl::qcu

#endif
