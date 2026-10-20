#include "kupl/lattice_set.h"
#include "kupl/lattice_linalg.h"

#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace kupl::qcu {

template <typename Real>
Complex<Real> LatticeSet<Real>::global_dot(const Complex<Real> *left,
                                           const Complex<Real> *right,
                                           int64_t count) const {
  Real real_sum = 0;
  Real imag_sum = 0;
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk) reduction(+ : real_sum, imag_sum)
  for (int64_t index = 0; index < count; ++index) {
    const Complex<Real> &a = left[index];
    const Complex<Real> &b = right[index];
    real_sum += a.real() * b.real() + a.imag() * b.imag();
    imag_sum += a.real() * b.imag() - a.imag() * b.real();
  }
  const Real local[2] = {real_sum, imag_sum};
  Real global[2] = {0, 0};
  const MPI_Datatype type =
      sizeof(Real) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE;
  MPI_Allreduce(local, global, 2, type, MPI_SUM, geometry_.comm);
  return Complex<Real>(global[0], global[1]);
}

template <typename Real>
Real LatticeSet<Real>::global_norm2(const Complex<Real> *field,
                                    int64_t count) const {
  Real local = 0;
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk) reduction(+ : local)
  for (int64_t index = 0; index < count; ++index) {
    local += std::norm(field[index]);
  }
  Real global = 0;
  const MPI_Datatype type =
      sizeof(Real) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE;
  MPI_Allreduce(&local, &global, 1, type, MPI_SUM, geometry_.comm);
  return global;
}

template <typename Real>
void LatticeSet<Real>::apply_full_operator(
    const Complex<Real> *input, Complex<Real> *output,
    const std::array<const Complex<Real> *, 2> &clover,
    Complex<Real> *scratch) {
  const int64_t vectors = geometry_.parity_vectors;
  for (int parity = 0; parity < 2; ++parity) {
    const int other = 1 - parity;
    const Complex<Real> *source =
        input + static_cast<std::size_t>(other) * vectors;
    Complex<Real> *destination =
        output + static_cast<std::size_t>(parity) * vectors;
    bare_dslash(scratch, source, parity);
    const Complex<Real> *onsite =
        input + static_cast<std::size_t>(parity) * vectors;
    apply_onsite(onsite, clover[static_cast<std::size_t>(parity)],
                 destination, scratch, kappa());
    if (parity == 0 && std::getenv("KUPL_DEBUG_OPERATOR") != nullptr) {
      log("OPERATOR", "source=(%g,%g) dslash=(%g,%g) result=(%g,%g)",
          static_cast<double>(source[0].real()),
          static_cast<double>(source[0].imag()),
          static_cast<double>(scratch[0].real()),
          static_cast<double>(scratch[0].imag()),
          static_cast<double>(destination[0].real()),
          static_cast<double>(destination[0].imag()));
    }
  }
}

template <typename Real>
template <typename Operator>
void LatticeSet<Real>::solve_compact_bistabcg(
    Complex<Real> *solution, const Complex<Real> *rhs, int64_t count,
    Operator &&apply_operator, const char *label,
    Real relative_tolerance) {
  using C = Complex<Real>;
  const int64_t chunk = omp_chunk<Real>(count);
  const Real solve_tolerance =
      relative_tolerance > Real(0) ? relative_tolerance : atol();
  const Real rhs_norm = std::sqrt(global_norm2(rhs, count));
  const Real recurrence_tolerance =
      std::max(solve_tolerance,
               std::numeric_limits<Real>::epsilon() * Real(4)) *
      rhs_norm;
  const Real acceptance_tolerance =
      std::max(solve_tolerance, residual_floor()) * rhs_norm;
  if (rhs_norm == Real(0)) {
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < count; ++index) {
      solution[index] = C(0);
    }
    last_iterations_ = 0;
    return;
  }
  if (solve_tolerance <= Real(0) || parameters_.get(kMaxIter) <= 0) {
    throw std::runtime_error(std::string(label) +
                             ": invalid atol/max_iter configuration");
  }

  apply_operator(solution, v_.data());
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < count; ++index) {
    r_[static_cast<std::size_t>(index)] =
        rhs[index] - v_[static_cast<std::size_t>(index)];
    rhat_[static_cast<std::size_t>(index)] =
        r_[static_cast<std::size_t>(index)];
    p_[static_cast<std::size_t>(index)] = C(0);
    v_[static_cast<std::size_t>(index)] = C(0);
  }

  C rho_old(1);
  C alpha(1);
  C omega(1);
  last_iterations_ = 0;
  const int max_iterations = parameters_.get(kMaxIter);
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    const Real residual = std::sqrt(global_norm2(r_.data(), count));
    if (residual <= recurrence_tolerance) {
      last_iterations_ = iteration;
      break;
    }
    const C rho = global_dot(rhat_.data(), r_.data(), count);
    if (std::abs(rho) <= std::numeric_limits<Real>::min() ||
        std::abs(omega) <= std::numeric_limits<Real>::min()) {
      throw std::runtime_error(std::string(label) +
                               ": BiStabCG scalar breakdown");
    }
    const C beta = (rho / rho_old) * (alpha / omega);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < count; ++index) {
      p_[static_cast<std::size_t>(index)] =
          r_[static_cast<std::size_t>(index)] +
          multiply_complex(
              beta, p_[static_cast<std::size_t>(index)] -
                        multiply_complex(
                            omega, v_[static_cast<std::size_t>(index)]));
    }

    apply_operator(p_.data(), v_.data());
    const C rhat_v = global_dot(rhat_.data(), v_.data(), count);
    if (std::abs(rhat_v) <= std::numeric_limits<Real>::min()) {
      throw std::runtime_error(std::string(label) +
                               ": BiStabCG alpha breakdown");
    }
    alpha = rho / rhat_v;
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < count; ++index) {
      s_[static_cast<std::size_t>(index)] =
          r_[static_cast<std::size_t>(index)] -
          multiply_complex(alpha,
                           v_[static_cast<std::size_t>(index)]);
    }
    if (std::sqrt(global_norm2(s_.data(), count)) <=
        recurrence_tolerance) {
#pragma omp parallel for schedule(static, chunk)
      for (int64_t index = 0; index < count; ++index) {
        solution[index] += multiply_complex(
            alpha, p_[static_cast<std::size_t>(index)]);
        r_[static_cast<std::size_t>(index)] =
            s_[static_cast<std::size_t>(index)];
      }
      last_iterations_ = iteration + 1;
      break;
    }

    apply_operator(s_.data(), t_.data());
    const C tt = global_dot(t_.data(), t_.data(), count);
    if (std::abs(tt) <= std::numeric_limits<Real>::min()) {
      throw std::runtime_error(std::string(label) +
                               ": BiStabCG omega breakdown");
    }
    omega = global_dot(t_.data(), s_.data(), count) / tt;
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < count; ++index) {
      solution[index] +=
          multiply_complex(alpha,
                           p_[static_cast<std::size_t>(index)]) +
          multiply_complex(omega,
                           s_[static_cast<std::size_t>(index)]);
      r_[static_cast<std::size_t>(index)] =
          s_[static_cast<std::size_t>(index)] -
          multiply_complex(omega,
                           t_[static_cast<std::size_t>(index)]);
    }
    rho_old = rho;
    last_iterations_ = iteration + 1;
  }

  apply_operator(solution, v_.data());
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < count; ++index) {
    r_[static_cast<std::size_t>(index)] =
        rhs[index] - v_[static_cast<std::size_t>(index)];
  }
  const Real true_residual =
      std::sqrt(global_norm2(r_.data(), count));
  log(label,
      "iterations=%d residual=%g recurrence_tolerance=%g "
      "acceptance_tolerance=%g relative=%g",
      last_iterations_, static_cast<double>(true_residual),
      static_cast<double>(recurrence_tolerance),
      static_cast<double>(acceptance_tolerance),
      static_cast<double>(true_residual / rhs_norm));
  if (!std::isfinite(true_residual) ||
      true_residual > acceptance_tolerance) {
    throw std::runtime_error(std::string(label) +
                             ": non-finite or out-of-tolerance true residual");
  }
}

template <typename Real>
template <typename Operator>
void LatticeSet<Real>::solve_compact_cg(
    Complex<Real> *solution, const Complex<Real> *rhs, int64_t count,
    Operator &&apply_operator, const char *label, Real relative_tolerance) {
  using C = Complex<Real>;
  const int64_t chunk = omp_chunk<Real>(count);
  const Real rhs_norm = std::sqrt(global_norm2(rhs, count));
  const Real recurrence_tolerance =
      std::max(relative_tolerance,
               std::numeric_limits<Real>::epsilon() * Real(4)) *
      rhs_norm;
  const Real acceptance_tolerance =
      std::max(relative_tolerance, residual_floor()) * rhs_norm;
  if (rhs_norm == Real(0)) {
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < count; ++index) {
      solution[index] = C(0);
    }
    last_iterations_ = 0;
    return;
  }
  if (atol() <= Real(0) || parameters_.get(kMaxIter) <= 0) {
    throw std::runtime_error(std::string(label) +
                             ": invalid atol/max_iter configuration");
  }

  apply_operator(solution, v_.data());
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < count; ++index) {
    r_[static_cast<std::size_t>(index)] =
        rhs[index] - v_[static_cast<std::size_t>(index)];
    p_[static_cast<std::size_t>(index)] =
        r_[static_cast<std::size_t>(index)];
  }
  Real rho = global_norm2(r_.data(), count);
  last_iterations_ = 0;
  const int max_iterations = parameters_.get(kMaxIter);
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    if (std::sqrt(rho) <= recurrence_tolerance) {
      last_iterations_ = iteration;
      break;
    }
    apply_operator(p_.data(), v_.data());
    const C denominator = global_dot(p_.data(), v_.data(), count);
    if (std::abs(denominator) <= std::numeric_limits<Real>::min()) {
      throw std::runtime_error(std::string(label) +
                               ": CG alpha breakdown");
    }
    const C alpha = C(rho, 0) / denominator;
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < count; ++index) {
      solution[index] += multiply_complex(
          alpha, p_[static_cast<std::size_t>(index)]);
      r_[static_cast<std::size_t>(index)] -=
          multiply_complex(alpha,
                           v_[static_cast<std::size_t>(index)]);
    }
    const Real rho_next = global_norm2(r_.data(), count);
    if (rho <= std::numeric_limits<Real>::min()) {
      throw std::runtime_error(std::string(label) +
                               ": CG beta breakdown");
    }
    const Real beta = rho_next / rho;
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < count; ++index) {
      p_[static_cast<std::size_t>(index)] =
          r_[static_cast<std::size_t>(index)] +
          beta * p_[static_cast<std::size_t>(index)];
    }
    rho = rho_next;
    last_iterations_ = iteration + 1;
  }

  apply_operator(solution, v_.data());
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < count; ++index) {
    r_[static_cast<std::size_t>(index)] =
        rhs[index] - v_[static_cast<std::size_t>(index)];
  }
  const Real true_residual =
      std::sqrt(global_norm2(r_.data(), count));
  log(label,
      "iterations=%d residual=%g recurrence_tolerance=%g "
      "acceptance_tolerance=%g relative=%g",
      last_iterations_, static_cast<double>(true_residual),
      static_cast<double>(recurrence_tolerance),
      static_cast<double>(acceptance_tolerance),
      static_cast<double>(true_residual / rhs_norm));
  if (!std::isfinite(true_residual) ||
      true_residual > acceptance_tolerance) {
    throw std::runtime_error(std::string(label) +
                             ": non-finite or out-of-tolerance true residual");
  }
}

template <typename Real>
void LatticeSet<Real>::prepare_clover_schur(
    Complex<Real> *compact_rhs, const Complex<Real> *full_rhs, int parity,
    const std::array<const Complex<Real> *, 2> &inverse) {
  const int other = 1 - parity;
  const int64_t vectors = geometry_.parity_vectors;
  const int64_t chunk = omp_chunk<Real>(vectors);
  const Complex<Real> *rhs_other =
      full_rhs + static_cast<std::size_t>(other) * vectors;
  const Complex<Real> *rhs_target =
      full_rhs + static_cast<std::size_t>(parity) * vectors;
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < vectors; ++index) {
    work_a_[static_cast<std::size_t>(index)] = rhs_other[index];
  }
  apply_onsite(work_a_.data(), inverse[static_cast<std::size_t>(other)],
               work_b_.data());
  bare_dslash(work_c_.data(), work_b_.data(), parity);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < vectors; ++index) {
    compact_rhs[index] =
        rhs_target[index] + kappa() * work_c_[static_cast<std::size_t>(index)];
  }
}

template <typename Real>
void LatticeSet<Real>::reconstruct_clover_schur(
    Complex<Real> *full_out, const Complex<Real> *full_rhs,
    const Complex<Real> *target_solution, int parity,
    const std::array<const Complex<Real> *, 2> &inverse) {
  const int other = 1 - parity;
  const int64_t vectors = geometry_.parity_vectors;
  const int64_t chunk = omp_chunk<Real>(vectors);
  Complex<Real> *out_target =
      full_out + static_cast<std::size_t>(parity) * vectors;
  Complex<Real> *out_other =
      full_out + static_cast<std::size_t>(other) * vectors;
  const Complex<Real> *rhs_other =
      full_rhs + static_cast<std::size_t>(other) * vectors;
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < vectors; ++index) {
    out_target[index] = target_solution[index];
  }
  bare_dslash(work_a_.data(), target_solution, other);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < vectors; ++index) {
    work_b_[static_cast<std::size_t>(index)] =
        rhs_other[index] + kappa() * work_a_[static_cast<std::size_t>(index)];
  }
  apply_onsite(work_b_.data(), inverse[static_cast<std::size_t>(other)],
               out_other);
}

template <typename Real>
void LatticeSet<Real>::reconstruct_wilson(Complex<Real> *full_output,
                                          const Complex<Real> *full_input,
                                          int parity) {
  const int other = 1 - parity;
  const int64_t vectors = geometry_.parity_vectors;
  const int64_t chunk = omp_chunk<Real>(vectors);
  Complex<Real> *target =
      full_output + static_cast<std::size_t>(parity) * vectors;
  Complex<Real> *other_output =
      full_output + static_cast<std::size_t>(other) * vectors;
  const Complex<Real> *other_rhs =
      full_input + static_cast<std::size_t>(other) * vectors;
  bare_dslash(work_a_.data(), target, other);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < vectors; ++index) {
    other_output[index] =
        other_rhs[index] + kappa() * work_a_[static_cast<std::size_t>(index)];
  }
}

template <typename Real>
Real LatticeSet<Real>::check_wilson_full_residual(
    Complex<Real> *full_output, const Complex<Real> *full_input,
    const char *label) {
  const int64_t full_vectors = geometry_.parity_vectors * 2;
  const int64_t chunk = omp_chunk<Real>(full_vectors);
  apply_full_wilson_operator(full_output, tmp_b_.data(), tmp_a_.data());
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < full_vectors; ++index) {
    bp_[static_cast<std::size_t>(index)] =
        full_input[index] - tmp_b_[static_cast<std::size_t>(index)];
  }
  const Real rhs_norm = std::sqrt(global_norm2(full_input, full_vectors));
  const Real residual = std::sqrt(global_norm2(bp_.data(), full_vectors));
  const Real tolerance =
      std::max(atol(), residual_floor()) * rhs_norm;
  log(label, "full_true_residual=%g rhs_norm=%g relative=%g",
      static_cast<double>(residual), static_cast<double>(rhs_norm),
      static_cast<double>(residual / std::max(rhs_norm, Real(1))));
  if (residual > tolerance) {
    throw std::runtime_error(std::string(label) +
                             ": full true residual exceeds tolerance");
  }
  return residual;
}

template <typename Real>
void LatticeSet<Real>::solve_wilson_bistabcg(
    Complex<Real> *full_output, const Complex<Real> *full_input) {
  const int parity = target_parity();
  const int other = 1 - parity;
  const int64_t vectors = geometry_.parity_vectors;
  const int64_t chunk = omp_chunk<Real>(vectors);
  const Complex<Real> *rhs_target =
      full_input + static_cast<std::size_t>(parity) * vectors;
  const Complex<Real> *rhs_other =
      full_input + static_cast<std::size_t>(other) * vectors;
  Complex<Real> *target =
      full_output + static_cast<std::size_t>(parity) * vectors;
  if (parameters_.get(kUseInitialGuess) == 0) {
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < vectors; ++index) {
      target[index] = Complex<Real>(0);
    }
  }

#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < vectors; ++index) {
    work_a_[static_cast<std::size_t>(index)] = rhs_other[index];
  }
  bare_dslash(work_b_.data(), work_a_.data(), parity);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < vectors; ++index) {
    bp_[static_cast<std::size_t>(index)] =
        rhs_target[index] + kappa() * work_b_[static_cast<std::size_t>(index)];
  }
  solve_compact_bistabcg(
      target, bp_.data(), vectors,
      [this, parity](const Complex<Real> *input, Complex<Real> *output) {
        apply_wilson_schur(input, output, parity, work_b_.data(),
                           work_c_.data());
      },
      "WILSON_BISTABCG", Real(0.1) * atol());
  reconstruct_wilson(full_output, full_input, parity);
  (void)check_wilson_full_residual(full_output, full_input,
                                   "WILSON_BISTABCG");
}

template <typename Real>
void LatticeSet<Real>::solve_wilson_cg(Complex<Real> *full_output,
                                       const Complex<Real> *full_input) {
  const int parity = target_parity();
  const int other = 1 - parity;
  const int64_t vectors = geometry_.parity_vectors;
  const int64_t chunk = omp_chunk<Real>(vectors);
  const Complex<Real> *rhs_target =
      full_input + static_cast<std::size_t>(parity) * vectors;
  const Complex<Real> *rhs_other =
      full_input + static_cast<std::size_t>(other) * vectors;
  Complex<Real> *target =
      full_output + static_cast<std::size_t>(parity) * vectors;
  if (parameters_.get(kUseInitialGuess) == 0) {
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < vectors; ++index) {
      target[index] = Complex<Real>(0);
    }
  }

#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < vectors; ++index) {
    work_a_[static_cast<std::size_t>(index)] = rhs_other[index];
  }
  bare_dslash(work_b_.data(), work_a_.data(), parity);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < vectors; ++index) {
    bp_[static_cast<std::size_t>(index)] =
        rhs_target[index] + kappa() * work_b_[static_cast<std::size_t>(index)];
  }
  apply_wilson_schur_dagger(bp_.data(), tmp_a_.data(), parity,
                            work_b_.data(), work_c_.data());
  solve_compact_cg(
      target, tmp_a_.data(), vectors,
      [this, parity](const Complex<Real> *input, Complex<Real> *output) {
        apply_wilson_cg_dslash(input, output, parity);
      },
      "WILSON_CG_NORMAL", Real(0.01) * atol());
  reconstruct_wilson(full_output, full_input, parity);
  (void)check_wilson_full_residual(full_output, full_input, "WILSON_CG");
}

template <typename Real>
void LatticeSet<Real>::solve_clover(
    Complex<Real> *full_output, const Complex<Real> *full_input, int parity,
    const std::array<const Complex<Real> *, 2> &clover,
    const std::array<const Complex<Real> *, 2> &inverse) {
  using C = Complex<Real>;
  const int64_t vectors = geometry_.parity_vectors;
  const int64_t full_vectors = vectors * 2;
  const bool warm = parameters_.get(kUseInitialGuess) != 0;
  Complex<Real> *target =
      full_output + static_cast<std::size_t>(parity) * vectors;
  if (!warm) {
    const int64_t chunk = omp_chunk<Real>(vectors);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < vectors; ++index) {
      target[index] = C(0);
    }
  }

  prepare_clover_schur(bp_.data(), full_input, parity, inverse);
  solve_compact_bistabcg(
      target, bp_.data(), vectors,
      [this, parity, &clover, &inverse](const Complex<Real> *input,
                                        Complex<Real> *output) {
        apply_schur(input, output, parity, clover, inverse, tmp_a_.data(),
                    tmp_b_.data());
      },
      "CLOVER_BISTABCG_SCHUR", Real(0.1) * atol());
  reconstruct_clover_schur(full_output, full_input, target, parity, inverse);

  apply_full_operator(full_output, tmp_b_.data(), clover, tmp_a_.data());
  const int64_t chunk = omp_chunk<Real>(full_vectors);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < full_vectors; ++index) {
    bp_[static_cast<std::size_t>(index)] =
        full_input[index] - tmp_b_[static_cast<std::size_t>(index)];
  }
  const Real rhs_norm = std::sqrt(global_norm2(full_input, full_vectors));
  const Real final_residual =
      std::sqrt(global_norm2(bp_.data(), full_vectors));
  const Real tolerance =
      std::max(atol(), residual_floor()) * std::max(rhs_norm, Real(1));
  log("CLOVER_BISTABCG_SCHUR",
      "recomputed_residual=%g rhs_norm=%g tolerance=%g relative=%g",
      static_cast<double>(final_residual), static_cast<double>(rhs_norm),
      static_cast<double>(tolerance),
      static_cast<double>(final_residual / std::max(rhs_norm, Real(1))));
  if (!std::isfinite(final_residual) || final_residual > tolerance) {
    throw std::runtime_error(
        "Clover BiStabCG: full true residual exceeds tolerance");
  }
}

template <typename Real>
void LatticeSet<Real>::clover_full_apply(
    void *out, const void *in, const void *gauge, const void *clover_even,
    const void *clover_odd, const void *clover_even_inv,
    const void *clover_odd_inv) {
  (void)clover_even_inv;
  (void)clover_odd_inv;
  if (out == nullptr || in == nullptr || gauge == nullptr ||
      clover_even == nullptr || clover_odd == nullptr) {
    throw std::runtime_error("full Clover apply received a null pointer");
  }
  require_solver_mode();
  ensure_gauge_halo(static_cast<const Complex<Real> *>(gauge));
  const std::array<const Complex<Real> *, 2> clover = {
      static_cast<const Complex<Real> *>(clover_even),
      static_cast<const Complex<Real> *>(clover_odd)};
  apply_full_operator(static_cast<const Complex<Real> *>(in),
                      static_cast<Complex<Real> *>(out), clover,
                      tmp_a_.data());
}

template class LatticeSet<float>;
template class LatticeSet<double>;

} // namespace kupl::qcu
