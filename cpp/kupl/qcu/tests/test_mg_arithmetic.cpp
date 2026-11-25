// Unit test for the three pieces of Legacy-MG arithmetic that no solve-level
// test can observe: the Givens rotation must be unitary, the contraction gate
// statistic must be CUDA's (r_n/r_1)^(1/window), and mu_pre must map 0 -> 4 and
// clamp at 8.  Each assertion below fails on the version of the code that was
// shipping before 2026-11-25.
//
// Built and run by ctest as kupl_qcu_mg_arithmetic; it needs no MPI, no lattice
// and no GPU, so it runs anywhere the CPU backend compiles.

#include "kupl/mg_arithmetic.hpp"

#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <random>

using kupl::qcu::Complex;
using kupl::qcu::kMgGatePerStepThreshold;
using kupl::qcu::kMgGateThreshold;
using kupl::qcu::kMgGateWindow;
using kupl::qcu::mg_contraction_mean;
using kupl::qcu::mg_givens_rotation;
using kupl::qcu::mg_guarded_divide;
using kupl::qcu::mg_mu_pre_steps;

namespace {

int failures = 0;

void check(const bool condition, const char *what, const double detail = 0.0) {
  if (!condition) {
    std::fprintf(stderr, "FAIL %s (detail %.17g)\n", what, detail);
    ++failures;
  }
}

// The second row of the rotation must annihilate b: -conj(s)a + c b == 0.
// With c = conj(a)/r this is b(conj(a)-a)/r, which is zero only for real a;
// the old code hid that by overwriting the entry.
template <typename Real>
void check_rotation(const Complex<Real> &a, const Complex<Real> &b,
                    const char *label) {
  kupl::qcu::MgGivensRotation<Real> rotation;
  mg_givens_rotation(a, b, rotation);
  const Real residual =
      std::abs(-std::conj(rotation.sine) * a + rotation.cosine * b);
  const Real scale = std::sqrt(std::norm(a) + std::norm(b));
  check(residual <= Real(64) * std::numeric_limits<Real>::epsilon() * scale,
        label, static_cast<double>(residual / (scale > Real(0) ? scale : Real(1))));

  // unitarity: |c|^2 + |s|^2 == 1 (c is real, so this is also c^2 + |s|^2)
  const Real norm = rotation.cosine * rotation.cosine + std::norm(rotation.sine);
  check(std::abs(norm - Real(1)) <= Real(64) * std::numeric_limits<Real>::epsilon(),
        "rotation is unitary", static_cast<double>(norm));

  // c must be real: a complex c is exactly the defect
  check(std::abs(std::imag(std::complex<Real>(rotation.cosine))) == Real(0),
        "cosine is real", static_cast<double>(std::imag(std::complex<Real>(rotation.cosine))));

  // the diagonal keeps the phase of a: diag/|diag| == a/|a|
  const Real radius = rotation.radius;
  check(std::abs(radius - scale) <= Real(64) * std::numeric_limits<Real>::epsilon() * scale,
        "radius is |a,b|", static_cast<double>(radius));
  if (std::abs(a) > Real(0)) {
    const Complex<Real> phase_expected = a / std::abs(a);
    const Complex<Real> phase_actual = rotation.diagonal / std::abs(rotation.diagonal);
    check(std::abs(phase_actual - phase_expected) <= Real(64) * std::numeric_limits<Real>::epsilon(),
          "diagonal keeps the phase of a",
          static_cast<double>(std::abs(phase_actual - phase_expected)));
  }
}

template <typename Real>
void run(const char *name) {
  // deterministic sweep over complex diagonals, including the ones that broke it
  check_rotation<Real>(Complex<Real>(Real(1.0155), Real(-0.0001)),
                       Complex<Real>(Real(0.0890), Real(0)), "h_jj almost real");
  check_rotation<Real>(Complex<Real>(Real(0), Real(1)), Complex<Real>(Real(1), Real(0)),
                       "pure imaginary diagonal");
  check_rotation<Real>(Complex<Real>(Real(0.3), Real(-0.7)),
                       Complex<Real>(Real(0.2), Real(0.5)), "generic complex");
  check_rotation<Real>(Complex<Real>(Real(1), Real(0)), Complex<Real>(Real(0), Real(0)),
                       "b is zero");
  check_rotation<Real>(Complex<Real>(Real(0), Real(0)), Complex<Real>(Real(0), Real(1)),
                       "a is zero");
  std::mt19937 generator(20261125u);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  for (int i = 0; i < 2000; ++i) {
    const Complex<Real> a(Real(uniform(generator)), Real(uniform(generator)));
    const Complex<Real> b(Real(uniform(generator)), Real(uniform(generator)));
    check_rotation<Real>(a, b, "random rotation");
  }

  // The gate statistic.  With a CONSTANT per-iteration contraction g, five
  // ratios over a six-sample window give (g^5)^(1/6) = g^(5/6), not g.  This is
  // the whole difference between the two forms this file has contained:
  //   current: g^(5/6) compared against 0.9  -> trips at g >= 0.9^(6/5) = 0.8812
  //   old:     g      compared against 0.9  -> trips only at g >= 0.9
  // so g = 0.89 is the case that separates them.
  {
    const Real g = Real(0.89);
    Real g5 = Real(1);
    for (int i = 0; i < 5; ++i) {
      g5 *= g;
    }
    const Real statistic = mg_contraction_mean(g5, Real(1), kMgGateWindow);
    const Real wanted = std::pow(g, Real(5) / Real(6));
    check(std::abs(statistic - wanted) <= Real(1e-12), "statistic is g^(5/6)",
          static_cast<double>(std::abs(statistic - wanted)));
    check(statistic >= Real(kMgGateThreshold), "g = 0.89 trips the 0.9 window threshold",
          static_cast<double>(statistic));
    // the old form, six ratios over six, is g itself and does NOT trip
    check(g < Real(kMgGateThreshold), "the old form leaves g = 0.89 alone",
          static_cast<double>(g));
    // the two thresholds describe the same point
    const Real per_step = std::pow(Real(kMgGateThreshold), Real(6) / Real(5));
    check(std::abs(per_step - Real(kMgGatePerStepThreshold)) < Real(1e-4),
          "0.9 per window is 0.8812 per step", static_cast<double>(per_step));
  }
  // a flat window stays at 1 regardless of the exponent, so the anchor case is safe
  check(mg_contraction_mean(Real(1), Real(1), kMgGateWindow) == Real(1),
        "mean of a flat window");
  check(mg_contraction_mean(Real(0), Real(1), 6) == Real(1), "zero estimate is neutral");
  check(mg_contraction_mean(Real(1), Real(0), 6) == Real(1), "zero anchor is neutral");

  // mu_pre mapping: 0 is "unset" and takes CUDA's default of four.
  check(mg_mu_pre_steps(0) == 4, "mu_pre 0 -> 4");
  check(mg_mu_pre_steps(-3) == 4, "mu_pre negative -> 4");
  for (int steps = 1; steps <= 8; ++steps) {
    check(mg_mu_pre_steps(steps) == steps, "mu_pre 1..8 is the value itself");
  }
  check(mg_mu_pre_steps(9) == 8, "mu_pre 9 is clamped to 8");
  check(mg_mu_pre_steps(100) == 8, "mu_pre 100 is clamped to 8");

  // The guarded divide must turn a numerical zero into a zero correction rather
  // than an infinity; 1e-40 is below the 1e-30 floor.
  {
    const Complex<Real> quotient =
        mg_guarded_divide(Complex<Real>(Real(1), Real(1)), Complex<Real>(Real(1e-40), Real(0)));
    check(quotient == Complex<Real>(0), "guarded divide of a 1e-40 pivot is zero");
    const Complex<Real> normal =
        mg_guarded_divide(Complex<Real>(Real(2), Real(0)), Complex<Real>(Real(4), Real(0)));
    check(std::abs(normal - Complex<Real>(Real(0.5), Real(0))) < Real(1e-15),
          "guarded divide keeps a normal quotient");
  }

  std::printf("mg_arithmetic %s: %s\n", name, failures == 0 ? "PASS" : "FAIL");
}

} // namespace

int main() {
  run<float>("float");
  run<double>("double");
  return failures == 0 ? 0 : 1;
}
