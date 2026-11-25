#ifndef PYQCU_KUPL_MG_ARITHMETIC_HPP
#define PYQCU_KUPL_MG_ARITHMETIC_HPP

// The three pieces of Legacy-MG arithmetic that are wrong in a way no solve-level
// test can see, factored out so a unit test can pin them.
//
// Why they need their own test rather than a residual comparison:
//   - a non-unitary Givens rotation still annihilates the subdiagonal (the next
//     line writes 0 there), so the solve converges anyway; the error scales with
//     Im(a), which measured 1e-4 on this problem, i.e. below the 1e-5..1e-4
//     tolerance band;
//   - the contraction gate is inert on every operator measured so far
//     (contraction ~0.40 against an 0.8812 trip), so swapping its statistic
//     changes no result that has ever been run;
//   - mu_pre_steps only changes the answer for --mg-nu-pre >= 9, which no
//     production run uses.
// In all three cases the code can be silently reverted and the 497-test suite
// stays green.  tests/test_mg_arithmetic.cpp is the guard.

#include "kupl/lattice_complex.h"

#include <algorithm>
#include <cmath>

namespace kupl::qcu {

// The preconditioner gate: sample one residual estimate per iteration, and after
// every `kMgGateWindow` samples compare the geometric mean of the contraction
// against `kMgGateThreshold`; two consecutive failing windows switch the
// preconditioner off for the rest of the solve.
//
// The statistic is (r_6/r_1)^(1/6), i.e. five ratios raised to 1/6, compared
// against 0.9.  Expressed per iteration that is a contraction of
// 0.9^(6/5) = 0.8812, which is where the "0.8812" in the review notes comes from:
// the same number, measured per step instead of per window.  The earlier code
// divided six ratios by six, which at the same 0.9 threshold tolerates a per-step
// contraction of 0.900 - a 2% difference in where a permanent latch fires.
// Measured problems contract at ~0.40, so neither form fires on them, which is
// exactly why this needs a unit test rather than a residual comparison.
inline constexpr int kMgGateWindow = 6;
inline constexpr double kMgGateThreshold = 0.9;
inline constexpr double kMgGatePerStepThreshold = 0.8812; // 0.9^(6/5)

// One plane rotation of the Givens QR used by the FGMRES outer solver.
//
// The rotation must be unitary for the residual estimate |g_{j+1}| to mean what
// it claims.  Writing c = conj(a)/r is only unitary when a is real: the second
// row becomes b(conj(a)-a)/r, which is zero only if a is real, and the code used
// to overwrite that entry with 0 so the defect never showed.  CUDA takes
// c = |a|/r real, s = (a/|a|)conj(b)/r, and keeps r*a/|a| on the diagonal
// (cpp/cuda/qcu/include/lattice_clover_multigrid.h:5862-5866).
template <typename Real>
struct MgGivensRotation {
  Real cosine;             // real, so that c^2 is a real squared magnitude
  Complex<Real> sine;
  Complex<Real> diagonal;  // radius * a/|a|
  Real radius;
};

// Written through an out-parameter rather than returned by value: returning a
// struct that holds a std::complex makes GCC emit a psabi note on every
// translation unit that calls it, which buries the real diagnostics in a build
// log.
template <typename Real>
void mg_givens_rotation(const Complex<Real> &a, const Complex<Real> &b,
                        MgGivensRotation<Real> &rotation) {
  const Real radius = std::sqrt(std::norm(a) + std::norm(b));
  const Real magnitude = std::abs(a);
  const Complex<Real> phase = magnitude > Real(0) ? a / magnitude : Complex<Real>(1);
  rotation.radius = radius;
  rotation.cosine = radius > Real(0) ? magnitude / radius : Real(1);
  rotation.sine =
      radius > Real(0) ? phase * std::conj(b) / radius : Complex<Real>(0);
  rotation.diagonal = phase * Complex<Real>(radius, 0);
}

// The contraction statistic the preconditioner gate compares against its
// threshold: (r_n/r_1)^(1/window), i.e. the geometric mean of the per-iteration
// contraction over one window, anchored on the window's first sample.
//
// Two forms have been in this file.  CUDA's (lattice_clover_multigrid.h:5886) is
// this one: five ratios raised to 1/6, so the statistic is the geometric mean
// per iteration over the window, not the average over all six samples.  The
// earlier form divided six ratios by six, which is a slightly smaller number
// for the same sequence and therefore trips slightly later.
template <typename Real>
Real mg_contraction_mean(const Real estimate, const Real anchor, const int window) {
  if (window <= 0 || !(anchor > Real(0)) || !(estimate > Real(0))) {
    return Real(1);
  }
  return std::exp((std::log(estimate) - std::log(anchor)) / Real(window));
}

// Number of MR smoothing steps one preconditioner application performs.
// Zero means "unset" and takes CUDA's default of four; the MR smoother is only
// defined for a small number of steps, so anything above eight is clamped rather
// than silently costing more than the envelope allows.
inline int mg_mu_pre_steps(const int configured) {
  if (configured <= 0) {
    return 4;
  }
  return std::min(configured, 8);
}

// Safe pivot division for the coarse solve: a diagonal smaller than 1e-30 in
// magnitude produces a zero correction rather than an infinity.  CUDA floors the
// same quantity at 1e-30.  std::numeric_limits<Real>::min() would be ~2e-308 for
// double, i.e. no floor at all in the range that matters.
template <typename Real>
Complex<Real> mg_guarded_divide(const Complex<Real> &numerator,
                                const Complex<Real> &diagonal) {
  return std::abs(diagonal) > Real(1e-30) ? numerator / diagonal
                                          : Complex<Real>(0);
}

} // namespace kupl::qcu

#endif // PYQCU_KUPL_MG_ARITHMETIC_HPP
