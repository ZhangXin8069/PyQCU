#include "kupl/qcu.h"
#include "kupl/lattice_strict.h"

#include "api_common.hpp"

#include <limits>

extern "C" {

int applyMultigridStrictCoarseQcu(long long _out, long long _in,
                                  long long _links, long long _onsite_pair,
                                  long long _set_ptrs, long long _params, int E,
                                  int X, int Y, int Z, int T,
                                  int onsite_index) {
  (void)_set_ptrs;
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_STRICT_COARSE", [&] {
        const auto *params = kupl::qcu::api::params_pointer(_params);
        if (params[kupl::qcu::kDataType] == kupl::qcu::kLatC64) {
          return kupl::qcu::strict_coarse<float>(
              kupl::qcu::api::mutable_pointer(_out),
              kupl::qcu::api::pointer(_in), kupl::qcu::api::pointer(_links),
              kupl::qcu::api::pointer(_onsite_pair), params, E, X, Y, Z, T,
              onsite_index);
        }
        if (params[kupl::qcu::kDataType] == kupl::qcu::kLatC128) {
          return kupl::qcu::strict_coarse<double>(
              kupl::qcu::api::mutable_pointer(_out),
              kupl::qcu::api::pointer(_in), kupl::qcu::api::pointer(_links),
              kupl::qcu::api::pointer(_onsite_pair), params, E, X, Y, Z, T,
              onsite_index);
        }
        throw std::runtime_error("strict coarse supports c64/c128");
      });
}

int applyMultigridStrictMatPCQcu(long long _out, long long _in,
                                 long long _links, long long _scratch,
                                 long long _set_ptrs, long long _params, int E,
                                 int X, int Y, int Z, int T, int parity) {
  (void)_set_ptrs;
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_STRICT_MATPC", [&] {
        const auto *params = kupl::qcu::api::params_pointer(_params);
        if (params[kupl::qcu::kDataType] == kupl::qcu::kLatC64) {
          return kupl::qcu::strict_matpc<float>(
              kupl::qcu::api::mutable_pointer(_out),
              kupl::qcu::api::pointer(_in), kupl::qcu::api::pointer(_links),
              kupl::qcu::api::mutable_pointer(_scratch), params, E, X, Y, Z, T,
              parity);
        }
        if (params[kupl::qcu::kDataType] == kupl::qcu::kLatC128) {
          return kupl::qcu::strict_matpc<double>(
              kupl::qcu::api::mutable_pointer(_out),
              kupl::qcu::api::pointer(_in), kupl::qcu::api::pointer(_links),
              kupl::qcu::api::mutable_pointer(_scratch), params, E, X, Y, Z, T,
              parity);
        }
        throw std::runtime_error("strict MATPC supports c64/c128");
      });
}

int applyMultigridStrictFineMatPCQcu(
    long long _out, long long _in, long long _gauge, long long _clover_even,
    long long _clover_odd, long long _clover_even_inv,
    long long _clover_odd_inv, long long _set_ptrs, long long _params,
    int parity) {
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_STRICT_FINE_MATPC", [&] {
        return kupl::qcu::api::lattice_set(_set_ptrs, _params)
            ->strict_fine_matpc(
                kupl::qcu::api::mutable_pointer(_out),
                kupl::qcu::api::pointer(_in), kupl::qcu::api::pointer(_gauge),
                kupl::qcu::api::pointer(_clover_even),
                kupl::qcu::api::pointer(_clover_odd),
                kupl::qcu::api::pointer(_clover_even_inv),
                kupl::qcu::api::pointer(_clover_odd_inv), parity);
      });
}

int applyMultigridStrictPrepareQcu(
    long long _out, long long _full_rhs, long long _links,
    long long _onsite_pair, long long _scratch, long long _set_ptrs,
    long long _params, int E, int X, int Y, int Z, int T, int parity) {
  (void)_set_ptrs;
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_STRICT_PREPARE", [&] {
        const auto *params = kupl::qcu::api::params_pointer(_params);
        if (params[kupl::qcu::kDataType] == kupl::qcu::kLatC64) {
          return kupl::qcu::strict_prepare<float>(
              kupl::qcu::api::mutable_pointer(_out),
              kupl::qcu::api::pointer(_full_rhs),
              kupl::qcu::api::pointer(_links),
              kupl::qcu::api::pointer(_onsite_pair),
              kupl::qcu::api::mutable_pointer(_scratch), params, E, X, Y, Z, T,
              parity);
        }
        if (params[kupl::qcu::kDataType] == kupl::qcu::kLatC128) {
          return kupl::qcu::strict_prepare<double>(
              kupl::qcu::api::mutable_pointer(_out),
              kupl::qcu::api::pointer(_full_rhs),
              kupl::qcu::api::pointer(_links),
              kupl::qcu::api::pointer(_onsite_pair),
              kupl::qcu::api::mutable_pointer(_scratch), params, E, X, Y, Z, T,
              parity);
        }
        throw std::runtime_error("strict prepare supports c64/c128");
      });
}

int applyMultigridStrictReconstructQcu(
    long long _full_out, long long _full_rhs, long long _target_solution,
    long long _links, long long _onsite_pair, long long _scratch,
    long long _set_ptrs, long long _params, int E, int X, int Y, int Z, int T,
    int parity) {
  (void)_set_ptrs;
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_STRICT_RECONSTRUCT", [&] {
        const auto *params = kupl::qcu::api::params_pointer(_params);
        if (params[kupl::qcu::kDataType] == kupl::qcu::kLatC64) {
          return kupl::qcu::strict_reconstruct<float>(
              kupl::qcu::api::mutable_pointer(_full_out),
              kupl::qcu::api::pointer(_full_rhs),
              kupl::qcu::api::pointer(_target_solution),
              kupl::qcu::api::pointer(_links),
              kupl::qcu::api::pointer(_onsite_pair),
              kupl::qcu::api::mutable_pointer(_scratch), params, E, X, Y, Z, T,
              parity);
        }
        if (params[kupl::qcu::kDataType] == kupl::qcu::kLatC128) {
          return kupl::qcu::strict_reconstruct<double>(
              kupl::qcu::api::mutable_pointer(_full_out),
              kupl::qcu::api::pointer(_full_rhs),
              kupl::qcu::api::pointer(_target_solution),
              kupl::qcu::api::pointer(_links),
              kupl::qcu::api::pointer(_onsite_pair),
              kupl::qcu::api::mutable_pointer(_scratch), params, E, X, Y, Z, T,
              parity);
        }
        throw std::runtime_error("strict reconstruct supports c64/c128");
      });
}

int applyMultigridStrictRestrictQcu(
    long long _coarse_out, long long _fine_in, long long _null_vectors,
    long long _set_ptrs, long long _params, int E, int e, int Xf, int Yf,
    int Zf, int Tf, int Xc, int Yc, int Zc, int Tc, int parity) {
  (void)_set_ptrs;
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_STRICT_RESTRICT", [&] {
        const auto *params = kupl::qcu::api::params_pointer(_params);
        if (params[kupl::qcu::kDataType] == kupl::qcu::kLatC64) {
          return kupl::qcu::strict_restrict<float>(
              kupl::qcu::api::mutable_pointer(_coarse_out),
              kupl::qcu::api::pointer(_fine_in),
              kupl::qcu::api::pointer(_null_vectors), E, e, Xf, Yf, Zf, Tf,
              Xc, Yc, Zc, Tc, parity);
        }
        if (params[kupl::qcu::kDataType] == kupl::qcu::kLatC128) {
          return kupl::qcu::strict_restrict<double>(
              kupl::qcu::api::mutable_pointer(_coarse_out),
              kupl::qcu::api::pointer(_fine_in),
              kupl::qcu::api::pointer(_null_vectors), E, e, Xf, Yf, Zf, Tf,
              Xc, Yc, Zc, Tc, parity);
        }
        throw std::runtime_error("strict restrict supports c64/c128");
      });
}

int applyMultigridStrictProLongQcu(
    long long _fine_out, long long _coarse_in, long long _null_vectors,
    long long _set_ptrs, long long _params, int E, int e, int Xf, int Yf,
    int Zf, int Tf, int Xc, int Yc, int Zc, int Tc, int parity) {
  (void)_set_ptrs;
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_STRICT_PROLONG", [&] {
        const auto *params = kupl::qcu::api::params_pointer(_params);
        if (params[kupl::qcu::kDataType] == kupl::qcu::kLatC64) {
          return kupl::qcu::strict_prolong<float>(
              kupl::qcu::api::mutable_pointer(_fine_out),
              kupl::qcu::api::pointer(_coarse_in),
              kupl::qcu::api::pointer(_null_vectors), E, e, Xf, Yf, Zf, Tf,
              Xc, Yc, Zc, Tc, parity);
        }
        if (params[kupl::qcu::kDataType] == kupl::qcu::kLatC128) {
          return kupl::qcu::strict_prolong<double>(
              kupl::qcu::api::mutable_pointer(_fine_out),
              kupl::qcu::api::pointer(_coarse_in),
              kupl::qcu::api::pointer(_null_vectors), E, e, Xf, Yf, Zf, Tf,
              Xc, Yc, Zc, Tc, parity);
        }
        throw std::runtime_error("strict prolong supports c64/c128");
      });
}

int applyMultigridStrictVCycleQcu(
    long long _full_out, long long _full_rhs, long long _set_ptrs,
    long long _params, int start_level, unsigned long long *allocated_bytes) {
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_STRICT_VCYCLE", [&] {
        if (allocated_bytes == nullptr) {
          throw std::runtime_error("strict V-cycle byte output is null");
        }
        *allocated_bytes = 0;
        const auto *params = kupl::qcu::api::params_pointer(_params);
        std::size_t bytes = 0;
        if (params[kupl::qcu::kDataType] == kupl::qcu::kLatC64) {
          bytes = kupl::qcu::strict_vcycle<float>(
              kupl::qcu::api::mutable_pointer(_full_out),
              kupl::qcu::api::pointer(_full_rhs),
              kupl::qcu::api::mutable_pointer(_set_ptrs), params, start_level);
        } else if (params[kupl::qcu::kDataType] == kupl::qcu::kLatC128) {
          bytes = kupl::qcu::strict_vcycle<double>(
              kupl::qcu::api::mutable_pointer(_full_out),
              kupl::qcu::api::pointer(_full_rhs),
              kupl::qcu::api::mutable_pointer(_set_ptrs), params, start_level);
        } else {
          throw std::runtime_error("strict V-cycle supports c64/c128");
        }
        *allocated_bytes = static_cast<unsigned long long>(bytes);
        return 0;
      });
}

int applyMultigridStrictInitQcu(long long _set_ptrs, long long _params,
                                int start_level,
                                unsigned long long *allocated_bytes) {
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_STRICT_INIT", [&] {
        if (allocated_bytes == nullptr) {
          throw std::runtime_error("strict hierarchy byte output is null");
        }
        *allocated_bytes = 0;
        const auto *params = kupl::qcu::api::params_pointer(_params);
        std::size_t bytes = 0;
        if (params[kupl::qcu::kDataType] == kupl::qcu::kLatC64) {
          bytes = kupl::qcu::strict_init_hierarchy<float>(
              kupl::qcu::api::mutable_pointer(_set_ptrs), params, start_level);
        } else if (params[kupl::qcu::kDataType] == kupl::qcu::kLatC128) {
          bytes = kupl::qcu::strict_init_hierarchy<double>(
              kupl::qcu::api::mutable_pointer(_set_ptrs), params, start_level);
        } else {
          throw std::runtime_error("strict hierarchy supports c64/c128");
        }
        *allocated_bytes = static_cast<unsigned long long>(bytes);
        return 0;
      });
}

int applyMultigridStrictEndQcu(long long _set_ptrs, long long _params) {
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_STRICT_END", [&] {
        const auto *params = kupl::qcu::api::params_pointer(_params);
        if (params[kupl::qcu::kDataType] == kupl::qcu::kLatC64) {
          return kupl::qcu::strict_end_hierarchy<float>(
              kupl::qcu::api::mutable_pointer(_set_ptrs));
        }
        if (params[kupl::qcu::kDataType] == kupl::qcu::kLatC128) {
          return kupl::qcu::strict_end_hierarchy<double>(
              kupl::qcu::api::mutable_pointer(_set_ptrs));
        }
        throw std::runtime_error("strict hierarchy supports c64/c128");
      });
}

int applyMultigridStrictFgmresQcu(
    long long _full_out, long long _full_rhs, long long _gauge,
    long long _clover_even, long long _clover_odd,
    long long _clover_even_inv, long long _clover_odd_inv,
    long long _fine_null_vectors, long long _set_ptrs, long long _params,
    int fine_E, int fine_X, int fine_Y, int fine_Z, int fine_T, int coarse_E,
    int coarse_X, int coarse_Y, int coarse_Z, int coarse_T, int element_bytes,
    int restart, int max_iter, double tolerance, int nu_pre, int nu_post,
    unsigned long long max_workspace_bytes, int *iterations, int *converged,
    double *final_true_residual, unsigned long long *allocated_bytes) {
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_STRICT_FGMRES", [&] {
        if (iterations == nullptr || converged == nullptr ||
            final_true_residual == nullptr || allocated_bytes == nullptr) {
          throw std::runtime_error("strict FGMRES result pointer is null");
        }
        if (max_workspace_bytes >
            static_cast<unsigned long long>(
                std::numeric_limits<std::size_t>::max())) {
          throw std::overflow_error("strict FGMRES workspace exceeds size_t");
        }
        *iterations = 0;
        *converged = 0;
        *final_true_residual = std::numeric_limits<double>::infinity();
        *allocated_bytes = 0;
        const auto *params = kupl::qcu::api::params_pointer(_params);
        std::size_t bytes = 0;
        if (params[kupl::qcu::kDataType] == kupl::qcu::kLatC64) {
          bool did_converge = false;
          float residual = std::numeric_limits<float>::infinity();
          bytes = kupl::qcu::strict_fgmres<float>(
              kupl::qcu::api::mutable_pointer(_full_out),
              kupl::qcu::api::pointer(_full_rhs),
              kupl::qcu::api::pointer(_gauge),
              kupl::qcu::api::pointer(_clover_even),
              kupl::qcu::api::pointer(_clover_odd),
              kupl::qcu::api::pointer(_clover_even_inv),
              kupl::qcu::api::pointer(_clover_odd_inv),
              kupl::qcu::api::pointer(_fine_null_vectors),
              kupl::qcu::api::mutable_pointer(_set_ptrs), params, fine_E,
              fine_X, fine_Y, fine_Z, fine_T, coarse_E, coarse_X, coarse_Y,
              coarse_Z, coarse_T, element_bytes, restart, max_iter, tolerance,
              nu_pre, nu_post,
              static_cast<std::size_t>(max_workspace_bytes), *iterations,
              did_converge, residual);
          *converged = did_converge ? 1 : 0;
          *final_true_residual = static_cast<double>(residual);
        } else if (params[kupl::qcu::kDataType] == kupl::qcu::kLatC128) {
          bool did_converge = false;
          double residual = std::numeric_limits<double>::infinity();
          bytes = kupl::qcu::strict_fgmres<double>(
              kupl::qcu::api::mutable_pointer(_full_out),
              kupl::qcu::api::pointer(_full_rhs),
              kupl::qcu::api::pointer(_gauge),
              kupl::qcu::api::pointer(_clover_even),
              kupl::qcu::api::pointer(_clover_odd),
              kupl::qcu::api::pointer(_clover_even_inv),
              kupl::qcu::api::pointer(_clover_odd_inv),
              kupl::qcu::api::pointer(_fine_null_vectors),
              kupl::qcu::api::mutable_pointer(_set_ptrs), params, fine_E,
              fine_X, fine_Y, fine_Z, fine_T, coarse_E, coarse_X, coarse_Y,
              coarse_Z, coarse_T, element_bytes, restart, max_iter, tolerance,
              nu_pre, nu_post,
              static_cast<std::size_t>(max_workspace_bytes), *iterations,
              did_converge, residual);
          *converged = did_converge ? 1 : 0;
          *final_true_residual = residual;
        } else {
          throw std::runtime_error("strict FGMRES supports c64/c128");
        }
        *allocated_bytes = static_cast<unsigned long long>(bytes);
        return 0;
      });
}

} // extern "C"
