#include "kupl/lattice_clover_multigrid.h"
#include "kupl/lattice_linalg.h"
#include "kupl/lattice_set.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace kupl::qcu {
namespace {

using MgProfileClock = std::chrono::steady_clock;

bool mg_env_enabled(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && *value != '\0' &&
         std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
         std::strcmp(value, "no") != 0 && std::strcmp(value, "off") != 0;
}

// The Legacy-MG solves with the SAME structure as CUDA, so the two backends can
// be compared: a right-preconditioned FGMRES(m) whose preconditioner is one
// application of the hierarchy.
//
//   M^-1 v = nu_pre MR smoothing steps on v, then one coarse cycle on the
//            residual that remains, then the prolongation of that correction
//            added with weight 1.  No post-smoothing: CUDA omits it and leaves
//            the remaining high frequencies to the outer Krylov
//            (cpp/cuda/qcu/include/lattice_clover_multigrid.h:5377).
//
//   outer:  z_j = M^-1 v_j; w = A z_j; Arnoldi with j+1 projections in one
//           collective; x += Z y; restart at m = 10 on a true residual
//           (lattice_clover_multigrid.h:5755-5900).
//
//   coarse: cycle() is CUDA's V-cycle body - two BiCGStab steps of pre-smoothing,
//           restrict, recurse, prolong, two steps of post-smoothing - and the
//           coarsest level solves to its tolerance.  At two levels the entry
//           level IS the coarsest, so the body does not run; that is a property
//           of the hierarchy depth, identical on both backends.
//
// This replaced an in-loop corrector that applied one damped coarse correction to
// a running BiCGStab residual on a fixed fine-iteration schedule.  That was a
// different method, so a CUDA-vs-KUPL measurement could not attribute a
// difference to the hardware.  It is deleted rather than kept behind a switch:
// two solvers selected by an environment variable is exactly how an A/B ends up
// measuring the same arm twice.
//
// `num_levels_ == 1` or _MG_USE_DEFLATE_ == 0 has no coarse space at all and runs
// set_.clover_bistabcg directly - that is the underlying solver, not a second
// multigrid.

struct MgProfileTimer {
  MgProfileTimer(bool enabled, double &target)
      : enabled_(enabled), target_(target),
        start_(enabled ? MgProfileClock::now() : MgProfileClock::time_point{}) {}

  ~MgProfileTimer() {
    if (enabled_) {
      target_ += std::chrono::duration<double>(MgProfileClock::now() - start_)
                     .count();
    }
  }

  bool enabled_;
  double &target_;
  MgProfileClock::time_point start_;
};

template <typename Real>
Complex<Real> mg_dot(const std::vector<Complex<Real>> &left,
                     const std::vector<Complex<Real>> &right,
                     bool distributed) {
  if (left.size() != right.size()) {
    throw std::runtime_error("legacy MG dot size mismatch");
  }
  Real real_sum = 0;
  Real imag_sum = 0;
  const int64_t count = static_cast<int64_t>(left.size());
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk) reduction(+ : real_sum, imag_sum)
  for (int64_t index = 0; index < count; ++index) {
    const Complex<Real> &a = left[static_cast<std::size_t>(index)];
    const Complex<Real> &b = right[static_cast<std::size_t>(index)];
    real_sum += a.real() * b.real() + a.imag() * b.imag();
    imag_sum += a.real() * b.imag() - a.imag() * b.real();
  }
  const Real local[2] = {real_sum, imag_sum};
  if (!distributed) {
    return Complex<Real>(local[0], local[1]);
  }
  Real global[2] = {0, 0};
  const MPI_Datatype type =
      sizeof(Real) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE;
  MPI_Allreduce(local, global, 2, type, MPI_SUM, MPI_COMM_WORLD);
  return Complex<Real>(global[0], global[1]);
}

template <typename Real>
std::array<Complex<Real>, 2> mg_dot_pair(
    const std::vector<Complex<Real>> &left,
    const std::vector<Complex<Real>> &right, bool distributed) {
  if (left.size() != right.size()) {
    throw std::runtime_error("legacy MG paired dot size mismatch");
  }
  Real left_real_sum = 0;
  Real left_imag_sum = 0;
  Real mixed_real_sum = 0;
  Real mixed_imag_sum = 0;
  const int64_t count = static_cast<int64_t>(left.size());
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk) reduction(+ : left_real_sum, left_imag_sum, mixed_real_sum, mixed_imag_sum)
  for (int64_t index = 0; index < count; ++index) {
    const Complex<Real> &a = left[static_cast<std::size_t>(index)];
    const Complex<Real> &b = right[static_cast<std::size_t>(index)];
    left_real_sum += a.real() * a.real() + a.imag() * a.imag();
    left_imag_sum += a.real() * a.imag() - a.imag() * a.real();
    mixed_real_sum += a.real() * b.real() + a.imag() * b.imag();
    mixed_imag_sum += a.real() * b.imag() - a.imag() * b.real();
  }
  const Real local[4] = {
      left_real_sum, left_imag_sum, mixed_real_sum, mixed_imag_sum};
  if (!distributed) {
    return {Complex<Real>(local[0], local[1]),
            Complex<Real>(local[2], local[3])};
  }
  Real global[4] = {0, 0, 0, 0};
  const MPI_Datatype type =
      sizeof(Real) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE;
  MPI_Allreduce(local, global, 4, type, MPI_SUM, MPI_COMM_WORLD);
  return {Complex<Real>(global[0], global[1]),
          Complex<Real>(global[2], global[3])};
}

// Batched inner products sharing the left vector, in one collective.
//
// FGMRES needs j+1 projections at Arnoldi step j.  One at a time they cost j+1
// blocking MPI_Allreduce, and at the measured 0.45-1.02 ms per reduction on eight
// ranks the recurrence would dominate the solve.  One reduction of 2n doubles
// replaces them, which is this backend's equivalent of CUDA keeping every dot
// device-side (cpp/cuda/qcu/include/lattice_clover_multigrid.h:5830-5845).
template <typename Real>
void mg_dot_many(const std::vector<std::vector<Complex<Real>>> &lefts,
                 const std::vector<Complex<Real>> &right, int n,
                 bool distributed, std::vector<Complex<Real>> &out) {
  out.assign(static_cast<std::size_t>(n), Complex<Real>(0));
  if (n <= 0) {
    return;
  }
  const int64_t count = static_cast<int64_t>(right.size());
  const int64_t chunk = omp_chunk<Real>(count);
  std::vector<Real> local(static_cast<std::size_t>(2 * n), Real(0));
  for (int slot = 0; slot < n; ++slot) {
    const std::vector<Complex<Real>> &other =
        lefts[static_cast<std::size_t>(slot)];
    Real real_part = 0;
    Real imag_part = 0;
    // out[slot] = lefts[slot]^H right, the same convention as mg_dot (which
    // conjugates its LEFT argument).  Arnoldi needs h_ij = <v_i, w> = v_i^H w, so
    // the basis goes on the left; passing the image on the left instead yields
    // conj(h_ij), and a conjugated Hessenberg makes the least-squares update
    // wrong while still producing plausible-looking numbers - which is exactly
    // how the first FGMRES run failed to converge.
#pragma omp parallel for schedule(static, chunk) reduction(+ : real_part, imag_part)
    for (int64_t index = 0; index < count; ++index) {
      const Complex<Real> product =
          std::conj(other[static_cast<std::size_t>(index)]) *
          right[static_cast<std::size_t>(index)];
      real_part += product.real();
      imag_part += product.imag();
    }
    local[static_cast<std::size_t>(2 * slot)] = real_part;
    local[static_cast<std::size_t>(2 * slot + 1)] = imag_part;
  }
  if (distributed) {
    std::vector<Real> global(static_cast<std::size_t>(2 * n), Real(0));
    const MPI_Datatype type =
        sizeof(Real) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Allreduce(local.data(), global.data(), 2 * n, type, MPI_SUM,
                  MPI_COMM_WORLD);
    local.swap(global);
  }
  for (int slot = 0; slot < n; ++slot) {
    out[static_cast<std::size_t>(slot)] =
        Complex<Real>(local[static_cast<std::size_t>(2 * slot)],
                      local[static_cast<std::size_t>(2 * slot + 1)]);
  }
}

template <typename Real>
Real mg_norm2(const std::vector<Complex<Real>> &field, bool distributed) {
  Real local = 0;
  const int64_t count = static_cast<int64_t>(field.size());
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk) reduction(+ : local)
  for (int64_t index = 0; index < count; ++index) {
    local += std::norm(field[static_cast<std::size_t>(index)]);
  }
  if (!distributed) {
    return local;
  }
  Real global = 0;
  const MPI_Datatype type =
      sizeof(Real) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE;
  MPI_Allreduce(&local, &global, 1, type, MPI_SUM, MPI_COMM_WORLD);
  return global;
}

template <typename Real>
Real mg_norm2(const Complex<Real> *field, int64_t count, bool distributed) {
  Real local = 0;
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk) reduction(+ : local)
  for (int64_t index = 0; index < count; ++index) {
    local += std::norm(field[index]);
  }
  if (!distributed) {
    return local;
  }
  Real global = 0;
  const MPI_Datatype type =
      sizeof(Real) == sizeof(float) ? MPI_FLOAT : MPI_DOUBLE;
  MPI_Allreduce(&local, &global, 1, type, MPI_SUM, MPI_COMM_WORLD);
  return global;
}

inline int64_t mg_volume(const std::array<int, kDim> &dims) {
  return static_cast<int64_t>(dims[0]) * dims[1] * dims[2] * dims[3];
}

template <typename Real> struct LegacyMgLevel {
  int E = 0;
  std::array<int, kDim> dims{};
  int max_iter = 100;
  // BiCGStab restart length for this level, in iterations; `<= 0` means one
  // unbroken BiCGStab.  Only the coarse levels use it: the fine level's outer
  // solver is FGMRES, which restarts on its own m = 10.
  int restart = 0;
  // Iterations since this level's last BiCGStab restart.  It lives in the level,
  // not in the chunk, so a chunk boundary is not a restart boundary unless this
  // counter says so.
  int since_restart = 0;
  Real tolerance = Real(0);
  Real target = Real(0);
  const Complex<Real> *null_vectors = nullptr;
  const Complex<Real> *hop_nn = nullptr;
  const Complex<Real> *hop_diag = nullptr;
  const Complex<Real> *sitting = nullptr;
  std::vector<Complex<Real>> x;
  std::vector<Complex<Real>> rhs;
  std::vector<Complex<Real>> r;
  std::vector<Complex<Real>> rhat;
  std::vector<Complex<Real>> correction;
  std::vector<Complex<Real>> p;
  std::vector<Complex<Real>> v;
  std::vector<Complex<Real>> s;
  std::vector<Complex<Real>> t;
};

template <typename Real> class LegacyCloverMultigrid {
public:
  LegacyCloverMultigrid(
      LatticeSetBase &set, const void *gauge, const void *clover_even,
      const void *clover_odd, const void *clover_even_inv,
      const void *clover_odd_inv, const void *fermion_in, void *set_ptrs,
      const int32_t *params)
      : set_(set), gauge_(gauge), clover_even_(clover_even),
        clover_odd_(clover_odd), clover_even_inv_(clover_even_inv),
        clover_odd_inv_(clover_odd_inv), fermion_in_(fermion_in),
        set_ptrs_(static_cast<long long *>(set_ptrs)), params_(params) {
    if (gauge_ == nullptr || clover_even_ == nullptr ||
        clover_odd_ == nullptr || clover_even_inv_ == nullptr ||
        clover_odd_inv_ == nullptr || fermion_in_ == nullptr ||
        set_ptrs_ == nullptr || params_ == nullptr) {
      throw std::runtime_error("legacy Clover-MG received a null pointer");
    }
    // The solver cannot bind itself and the caller's environment dominates
    // every timing it produces: the identical L48 c64 16-rank solve measured
    // 196.00 s without KUPL_NUMA_BIND=1 KUPL_MEMORY_NODE=local+16
    // KUPL_HUGEPAGE=1 and 2.06 s with them (jobs 2236451 vs 2236485, 95x).
    // The fine operator streams the gauge slab plus four Clover terms, so the
    // OpenMP threads and first-touch pages scatter over 8 NUMA domains unbound.
    num_levels_ = params_[kMgNumLevel];
    distributed_ = params_[kNodeSize] > 1;
    // 5 is an ABI-layout limit, not an algorithmic one (define.h:41,49,61,76):
    // per-level params at 19 + 8*(level-1) (E,X,Y,Z,T,max_iter,dtype,restart),
    // per-level set_ptrs quartets at 30 + 4*(level-1) (lonv,hnn,hdg,sit) and
    // per-level argv tolerances at 3 + (level-1).  Raising the cap needs a
    // layout change on both the C++ and the Python side, not just this check.
    if (num_levels_ < 1 || num_levels_ > 5) {
      throw std::runtime_error("legacy Clover-MG level count must be 1..5");
    }
    if (params_[kMgUseGcr] != 0) {
      throw std::runtime_error(
          "KUPL legacy MG rejects CUDA-only GCR/cycle mode bits");
    }
    deflate_ = params_[kMgUseDeflate] != 0;
    compact_fine_ =
        set_ptrs_[kSetPtrsLegacyLayout] == kLegacyLayoutOddSchur;
    levels_[0].E = kVector;
    levels_[0].dims = {
        params_[kLatX], params_[kLatY], params_[kLatZ], params_[kLatT]};
    if (compact_fine_) {
      if (levels_[0].dims[3] % 2 != 0) {
        throw std::runtime_error(
            "odd-Schur Legacy MG requires an even time extent");
      }
      levels_[0].dims[3] /= 2;
    }
    levels_[0].max_iter = params_[kMaxIter];
    if (levels_[0].max_iter <= 0) {
      throw std::runtime_error(
          "legacy Clover-MG max iterations must be positive");
    }
    const double fine_tolerance = set_.real_parameter(kAtol);
    if (!std::isfinite(fine_tolerance) || fine_tolerance <= 0) {
      throw std::runtime_error(
          "legacy Clover-MG fine tolerance must be finite and positive");
    }
    levels_[0].tolerance = static_cast<Real>(fine_tolerance);
    allocate_level(levels_[0]);
    for (int level = 1; level < num_levels_; ++level) {
      const int base = kMgLevel1E + (level - 1) * kMgParamsSize;
      const int slot = kSetPtrsCoarseBase + 4 * (level - 1);
      levels_[level].E = params_[base];
      levels_[level].dims = {
          params_[base + 1], params_[base + 2], params_[base + 3],
          params_[base + 4]};
      for (int dim = 0; dim < kDim; ++dim) {
        if (levels_[level].dims[dim] <= 0 ||
            levels_[level - 1].dims[dim] % levels_[level].dims[dim] != 0) {
          throw std::runtime_error(
              "legacy Clover-MG coarse dimensions must divide the parent");
        }
      }
      levels_[level].max_iter =
          params_[base + 5] > 0 ? params_[base + 5] : 50;
      const int arg_index = kMgLevel1Atol + level - 1;
      const double level_tolerance = set_.real_parameter(arg_index);
      levels_[level].tolerance = static_cast<Real>(
          level_tolerance > 0
              ? level_tolerance
              : (sizeof(Real) == sizeof(float) ? 1e-5 : 1e-11));
      levels_[level - 1].null_vectors =
          reinterpret_cast<const Complex<Real> *>(set_ptrs_[slot + 0]);
      levels_[level].hop_nn =
          reinterpret_cast<const Complex<Real> *>(set_ptrs_[slot + 1]);
      levels_[level].hop_diag =
          reinterpret_cast<const Complex<Real> *>(set_ptrs_[slot + 2]);
      levels_[level].sitting =
          reinterpret_cast<const Complex<Real> *>(set_ptrs_[slot + 3]);
      if (params_[base + 6] != params_[kDataType]) {
        throw std::runtime_error(
            "KUPL legacy MG mixed precision is not supported");
      }
      if (levels_[level - 1].null_vectors == nullptr ||
          levels_[level].hop_nn == nullptr ||
          levels_[level].hop_diag == nullptr ||
          levels_[level].sitting == nullptr) {
        throw std::runtime_error(
            "legacy Clover-MG coarse transition assets are incomplete");
      }
      allocate_level(levels_[level]);
    }
    if (num_levels_ > 1 && params_[kNodeRank] == 0) {
      const auto &coarse = levels_[num_levels_ - 1].dims;
      const auto &parent = levels_[num_levels_ - 2].dims;
      int block[kDim] = {1, 1, 1, 1};
      int next[kDim] = {1, 1, 1, 1};
      long long volume = 1;
      bool too_small = false;
      bool no_next = false;
      for (int dim = 0; dim < kDim; ++dim) {
        block[dim] = std::max(1, parent[dim] / coarse[dim]);
        next[dim] = std::max(1, coarse[dim] / block[dim]);
        volume *= coarse[dim];
        too_small = too_small || coarse[dim] < 4;
        no_next = no_next || next[dim] < 4;
      }
      too_small = too_small || volume < 1024;
      if (num_levels_ > 2 || too_small || no_next) {
        std::fprintf(
            stderr,
            "PYQCU::KUPL::MULTIGRID::WARNING:\n"
            " Legacy-MultiGrid levels=%d, coarsest=[%d,%d,%d,%d], "
            "next=[%d,%d,%d,%d], volume=%lld. Legacy defaults to 2 levels; "
            "deep or small coarse hierarchies are experimental.\n",
            num_levels_, coarse[0], coarse[1], coarse[2], coarse[3],
            next[0], next[1], next[2], next[3], volume);
      }
    }
    canonical_input_.resize(
        static_cast<std::size_t>(2 * kVector * params_[kLatX] *
                                 params_[kLatY] * params_[kLatZ] *
                                 (params_[kLatT] / 2)));
    canonical_output_.resize(canonical_input_.size());
    profile_ = mg_env_enabled("PYQCU_KUPL_MG_PROFILE");
    // KUPL_MG_DEBUG is accepted and ignored: the per-correction trace it used to
    // enable belonged to the deleted corrector, and the FGMRES path prints one
    // MG_FGMRES_TRACE line per restart unconditionally (rank 0, to stderr) because
    // the one time that diagnostic was needed it was behind a flag nobody set.
    // Saying so here is better than a knob that silently does nothing.
    require_correction_ = mg_env_enabled("KUPL_REQUIRE_MG_CORRECTION");
    level_iterations_.assign(static_cast<std::size_t>(num_levels_), 0);
  }

  void solve(void *fermion_out, bool keep_initial) {
    const auto profile_start = MgProfileClock::now();
    if (fermion_out == nullptr) {
      throw std::runtime_error("legacy Clover-MG output is null");
    }
    if (compact_fine_) {
      {
        MgProfileTimer timer(profile_, prof_prepare_seconds_);
        prepare_target_rhs(levels_[0].rhs, fermion_in_);
      }
      if (keep_initial) {
        load_target_solution(
            levels_[0].x, static_cast<const Complex<Real> *>(fermion_out));
      } else {
        std::fill(levels_[0].x.begin(), levels_[0].x.end(), Complex<Real>(0));
      }
    } else {
      canonical_to_full(levels_[0].rhs,
                        static_cast<const Complex<Real> *>(fermion_in_));
      if (keep_initial) {
        canonical_to_full(levels_[0].x,
                          static_cast<const Complex<Real> *>(fermion_out));
      } else {
        std::fill(levels_[0].x.begin(), levels_[0].x.end(), Complex<Real>(0));
      }
    }
    apply_level(0, levels_[0].v, levels_[0].x);
    for (std::size_t i = 0; i < levels_[0].r.size(); ++i) {
      levels_[0].r[i] = levels_[0].rhs[i] - levels_[0].v[i];
    }
    const Real rhs_norm = std::sqrt(mg_norm2(levels_[0].rhs, distributed_));
    Real residual = std::sqrt(mg_norm2(levels_[0].r, distributed_));
    if (rhs_norm == Real(0)) {
      std::fill(levels_[0].x.begin(), levels_[0].x.end(), Complex<Real>(0));
      store_solution(fermion_out);
      last_total_iterations_ = 0;
      last_accepted_corrections_ = 0;
      print_level_iterations();
      print_profile(profile_start);
      return;
    }
    const Real target = levels_[0].tolerance * rhs_norm;
    levels_[0].target = target;
    int total_iterations = 0;
    int accepted_corrections = 0;
    if (num_levels_ <= 1 || !deflate_) {
      // No coarse space.  This is the underlying parity-Schur BiCGStab, not a
      // second multigrid, and it writes the solution itself, so it returns here.
      set_.clover_bistabcg(
          fermion_out, fermion_in_, gauge_, clover_even_, clover_odd_,
          clover_even_inv_, clover_odd_inv_);
      total_iterations = set_.iterations();
      last_total_iterations_ = total_iterations;
      last_accepted_corrections_ = 0;
      if (require_correction_) {
        throw std::runtime_error(
            "KUPL_REQUIRE_MG_CORRECTION=1 needs a coarse space, but levels=" +
            std::to_string(num_levels_) +
            " and _MG_USE_DEFLATE_=" + std::to_string(deflate_ ? 1 : 0));
      }
      if (params_[kVerbose] != 0) {
        std::fprintf(stdout,
                     "PYQCU::KUPL::CLOVER_MULTIGRID_DEBUG:\n "
                     "no coarse space: iterations=%d\n",
                     total_iterations);
      }
      print_level_iterations();
      print_profile(profile_start);
      return;
    }
    fgmres_solve(total_iterations, accepted_corrections, residual, target);
    if (!std::isfinite(static_cast<double>(residual)) ||
        residual > target) {
      throw std::runtime_error(
          "legacy Clover-MG did not reach the full true residual");
    }
    last_total_iterations_ = total_iterations;
    last_accepted_corrections_ = accepted_corrections;
    if (require_correction_ && accepted_corrections == 0) {
      throw std::runtime_error(
          "KUPL_REQUIRE_MG_CORRECTION=1 completed without an accepted coarse "
          "correction");
    }
    {
      MgProfileTimer timer(profile_, prof_store_seconds_);
      store_solution(fermion_out);
    }
    if (compact_fine_) {
      Real full_residual = 0;
      {
        MgProfileTimer timer(profile_, prof_full_residual_seconds_);
        full_residual = checked_full_residual(fermion_out);
      }
      const Real full_rhs_norm = std::sqrt(mg_norm2(
          static_cast<const Complex<Real> *>(fermion_in_),
          static_cast<int64_t>(canonical_input_.size()), distributed_));
      const Real full_target = levels_[0].tolerance * full_rhs_norm;
      if (params_[kVerbose] != 0) {
        std::fprintf(
            stdout,
            "PYQCU::KUPL::CLOVER_MULTIGRID_DEBUG:\n "
            "full true residual=%g target=%g relative=%g\n",
            static_cast<double>(full_residual),
            static_cast<double>(full_target),
            static_cast<double>(full_residual /
                                std::max(full_rhs_norm, Real(1))));
      }
      if (!std::isfinite(static_cast<double>(full_residual)) ||
          full_residual > full_target) {
        throw std::runtime_error(
            "legacy Clover-MG full true residual exceeds tolerance");
      }
    }
    print_level_iterations();
    print_profile(profile_start);
  }

  bool verify() {
    const Real tolerance =
        sizeof(Real) == sizeof(float) ? Real(2e-3) : Real(1e-8);
    for (int parent = 0; parent + 1 < num_levels_; ++parent) {
      LegacyMgLevel<Real> &fine = levels_[parent];
      LegacyMgLevel<Real> &coarse = levels_[parent + 1];
      // The dof-level mapping audit used to be reached only from the corrector's
      // diagnostic.  It is a gate on the transfer pair, not part of any solver, so
      // it runs here instead - once per level, guarded inside.
      audit_transfer_mapping(parent,
                             std::max(1, fine.dims[0] / coarse.dims[0]),
                             std::max(1, fine.dims[1] / coarse.dims[1]),
                             std::max(1, fine.dims[2] / coarse.dims[2]),
                             std::max(1, fine.dims[3] / coarse.dims[3]));
      std::vector<Complex<Real>> coarse_seed(coarse.x.size());
      std::vector<Complex<Real>> fine_image(fine.x.size());
      std::vector<Complex<Real>> coarse_back(coarse.x.size());
      std::vector<Complex<Real>> parent_image(fine.x.size());
      std::vector<Complex<Real>> coarse_direct(coarse.x.size());
      std::vector<Complex<Real>> coarse_galerkin(coarse.x.size());
      for (std::size_t i = 0; i < coarse_seed.size(); ++i) {
        const Real phase =
            Real(0.013) * static_cast<Real>((i * 17 + parent * 31) % 101);
        coarse_seed[i] =
            Complex<Real>(std::cos(phase), Real(0.7) * std::sin(phase));
      }
      prolong(parent, fine_image, coarse_seed);
      restrict(parent, coarse_back, fine_image);
      apply_level(parent, parent_image, fine_image);
      restrict(parent, coarse_galerkin, parent_image);
      apply_level(parent + 1, coarse_direct, coarse_seed);
      const Real seed_norm = std::sqrt(mg_norm2(coarse_seed, distributed_));
      std::vector<Complex<Real>> projection_difference(coarse.x.size());
      for (std::size_t i = 0; i < projection_difference.size(); ++i) {
        projection_difference[i] = coarse_back[i] - coarse_seed[i];
      }
      const Real projection_error =
          std::sqrt(mg_norm2(projection_difference, distributed_)) / seed_norm;
      std::vector<Complex<Real>> difference(coarse.x.size());
      for (std::size_t i = 0; i < difference.size(); ++i) {
        difference[i] = coarse_galerkin[i] - coarse_direct[i];
      }
      const Real galerkin_error =
          std::sqrt(mg_norm2(difference, distributed_)) /
          std::max(std::sqrt(mg_norm2(coarse_direct, distributed_)), Real(1));
      if (projection_error > tolerance || galerkin_error > tolerance) {
        const auto nonzero_count = [](const auto &field) {
          std::size_t count = 0;
          for (const auto &value : field) {
            if (std::abs(value) != 0) {
              ++count;
            }
          }
          return count;
        };
        std::fprintf(
            stderr,
            "PYQCU::KUPL::CLOVER_MULTIGRID_VERIFY:\n "
            "transition=%d projection=%g galerkin=%g tolerance=%g "
            "fine_nonzero=%zu coarse_back_nonzero=%zu\n",
            parent, static_cast<double>(projection_error),
            static_cast<double>(galerkin_error),
            static_cast<double>(tolerance), nonzero_count(fine_image),
            nonzero_count(coarse_back));
        return false;
      }
    }
    return true;
  }

private:
  void load_target_solution(std::vector<Complex<Real>> &compact,
                            const Complex<Real> *canonical) const {
    if (canonical == nullptr) {
      throw std::runtime_error("legacy Clover-MG received a null field");
    }
    const int parity = params_[kParity];
    if (parity != 0 && parity != 1) {
      throw std::runtime_error("legacy Clover-MG parity must be 0 or 1");
    }
    const int64_t vectors = static_cast<int64_t>(compact.size());
    const Complex<Real> *source =
        canonical + static_cast<int64_t>(parity) * vectors;
    std::copy(source, source + vectors, compact.begin());
  }

  void prepare_target_rhs(std::vector<Complex<Real>> &compact,
                          const void *full_rhs) {
    set_.clover_bistabcg_prepare(
        compact.data(), full_rhs, gauge_, clover_even_, clover_odd_,
        clover_even_inv_, clover_odd_inv_);
  }

  void store_solution(void *fermion_out) {
    if (!compact_fine_) {
      full_to_canonical(static_cast<Complex<Real> *>(fermion_out),
                        levels_[0].x);
      return;
    }
    set_.clover_bistabcg_reconstruct(
        fermion_out, fermion_in_, levels_[0].x.data(), gauge_, clover_even_,
        clover_odd_, clover_even_inv_, clover_odd_inv_);
  }

  Real checked_full_residual(void *fermion_out) {
    set_.clover_full_apply(canonical_output_.data(), fermion_out, gauge_,
                          clover_even_, clover_odd_, clover_even_inv_,
                          clover_odd_inv_);
    const auto *rhs = static_cast<const Complex<Real> *>(fermion_in_);
    const int64_t count = static_cast<int64_t>(canonical_input_.size());
    const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < count; ++index) {
      canonical_input_[static_cast<std::size_t>(index)] =
          rhs[index] - canonical_output_[static_cast<std::size_t>(index)];
    }
    return std::sqrt(mg_norm2(canonical_input_, distributed_));
  }

  void allocate_level(LegacyMgLevel<Real> &level) {
    if (level.E <= 0 ||
        std::any_of(level.dims.begin(), level.dims.end(),
                    [](int value) { return value <= 0; })) {
      throw std::runtime_error("legacy Clover-MG level geometry is invalid");
    }
    const std::size_t size =
        static_cast<std::size_t>(level.E) * mg_volume(level.dims);
    for (auto *field :
         {&level.x, &level.rhs, &level.r, &level.rhat, &level.correction,
          &level.p, &level.v, &level.s, &level.t}) {
      field->assign(size, Complex<Real>(0));
    }
  }

  void canonical_to_full(std::vector<Complex<Real>> &full,
                         const Complex<Real> *canonical) const {
    const int X = params_[kLatX];
    const int Y = params_[kLatY];
    const int Z = params_[kLatZ];
    const int T = params_[kLatT];
    const int half_time = T / 2;
    const int64_t half_sites =
        static_cast<int64_t>(X) * Y * Z * half_time;
    const int64_t plane = static_cast<int64_t>(X) * Y * Z * T;
    const int64_t chunk = omp_chunk<Real>(half_sites);
    // Walk canonical half-sites once and emit both time parities.  The
    // component-major full layout used to require a 4-D division for every
    // component of every site on each legacy-MG fine application.
#pragma omp parallel for schedule(static, chunk)
    for (int64_t half_site = 0; half_site < half_sites; ++half_site) {
      int64_t remainder = half_site;
      const int t2 = static_cast<int>(remainder % half_time);
      remainder /= half_time;
      const int z = static_cast<int>(remainder % Z);
      remainder /= Z;
      const int y = static_cast<int>(remainder % Y);
      const int x = static_cast<int>(remainder / Y);
      const int even_parity = (x + y + z) & 1;
      const int64_t site =
          ((static_cast<int64_t>(x) * Y + y) * Z + z) * T;
      for (int component = 0; component < kVector; ++component) {
        const int64_t source =
            static_cast<int64_t>(component) * plane + site;
        full[static_cast<std::size_t>(source + 2 * t2 + even_parity)] =
            canonical[(static_cast<int64_t>(0) * kVector + component) *
                          half_sites +
                      half_site];
        full[static_cast<std::size_t>(
            source + 2 * t2 + (even_parity ^ 1))] =
            canonical[(static_cast<int64_t>(1) * kVector + component) *
                          half_sites +
                      half_site];
      }
    }
  }

  void full_to_canonical(Complex<Real> *canonical,
                         const std::vector<Complex<Real>> &full) const {
    const int X = params_[kLatX];
    const int Y = params_[kLatY];
    const int Z = params_[kLatZ];
    const int T = params_[kLatT];
    const int half_time = T / 2;
    const int64_t half_sites =
        static_cast<int64_t>(X) * Y * Z * half_time;
    const int64_t plane = static_cast<int64_t>(X) * Y * Z * T;
    const int64_t chunk = omp_chunk<Real>(half_sites);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t half_site = 0; half_site < half_sites; ++half_site) {
      int64_t remainder = half_site;
      const int t2 = static_cast<int>(remainder % half_time);
      remainder /= half_time;
      const int z = static_cast<int>(remainder % Z);
      remainder /= Z;
      const int y = static_cast<int>(remainder % Y);
      const int x = static_cast<int>(remainder / Y);
      const int even_parity = (x + y + z) & 1;
      const int64_t site =
          ((static_cast<int64_t>(x) * Y + y) * Z + z) * T;
      for (int component = 0; component < kVector; ++component) {
        const int64_t source =
            static_cast<int64_t>(component) * plane + site;
        canonical[(static_cast<int64_t>(0) * kVector + component) *
                      half_sites +
                  half_site] =
            full[static_cast<std::size_t>(source + 2 * t2 + even_parity)];
        canonical[(static_cast<int64_t>(1) * kVector + component) *
                      half_sites +
                  half_site] =
            full[static_cast<std::size_t>(
                source + 2 * t2 + (even_parity ^ 1))];
      }
    }
  }

  void apply_level(int level, std::vector<Complex<Real>> &out,
                   const std::vector<Complex<Real>> &in) {
    MgProfileTimer timer(profile_, level == 0 ? prof_fine_apply_seconds_
                                              : prof_coarse_apply_seconds_);
    ++(level == 0 ? prof_fine_apply_calls_ : prof_coarse_apply_calls_);
    if (level == 0) {
      if (compact_fine_) {
        set_.clover_bistabcg_dslash(
            out.data(), in.data(), gauge_, clover_even_, clover_odd_,
            clover_even_inv_, clover_odd_inv_);
        return;
      }
      // Legacy MG stores full-site vectors; the QCU full operator consumes
      // parity-compressed canonical vectors.
      full_to_canonical(canonical_input_.data(), in);
      set_.clover_full_apply(
          canonical_output_.data(), canonical_input_.data(), gauge_, clover_even_,
          clover_odd_, clover_even_inv_, clover_odd_inv_);
      canonical_to_full(out, canonical_output_.data());
      return;
    }
    LegacyMgLevel<Real> &state = levels_[level];
    set_.multigrid_coarse_dslash_wide(
        out.data(), in.data(), state.sitting, state.hop_nn, state.hop_diag,
        state.E, state.dims[0], state.dims[1], state.dims[2], state.dims[3]);
  }

  void restrict(int parent, std::vector<Complex<Real>> &coarse,
                const std::vector<Complex<Real>> &fine) {
    MgProfileTimer timer(profile_, prof_restrict_seconds_);
    ++prof_restrict_calls_;
    LegacyMgLevel<Real> &p = levels_[parent];
    LegacyMgLevel<Real> &c = levels_[parent + 1];
    set_.multigrid_restrict(
        coarse.data(), fine.data(), p.null_vectors, c.E, p.E, p.dims[0],
        p.dims[1], p.dims[2], p.dims[3], c.dims[0], c.dims[1], c.dims[2],
        c.dims[3]);
  }

  void prolong(int parent, std::vector<Complex<Real>> &fine,
               const std::vector<Complex<Real>> &coarse) {
    MgProfileTimer timer(profile_, prof_prolong_seconds_);
    ++prof_prolong_calls_;
    LegacyMgLevel<Real> &p = levels_[parent];
    LegacyMgLevel<Real> &c = levels_[parent + 1];
    set_.multigrid_prolong(
        fine.data(), coarse.data(), p.null_vectors, c.E, p.E, p.dims[0],
        p.dims[1], p.dims[2], p.dims[3], c.dims[0], c.dims[1], c.dims[2],
        c.dims[3]);
  }

  // One Krylov chunk: up to `iterations` BiCGStab iterations on `level`.
  //
  // A chunk is NOT a BiCGStab run.  `state.restart` (> 0 on level 0, the
  // --restart value) is the restart length: the recurrence restarts every
  // `restart` fine iterations, counted in absolute iterations through
  // `state.since_restart`, so one chunk can span several restarts and one
  // restart can span several coarse corrections.  A restart re-initialises
  // rhat = r, p = v = 0, rho_old = alpha = omega = 1 and touches nothing else:
  // x and the true residual r are continuous across it.
  //
  // `state.restart <= 0` keeps the pre-decoupling behaviour (one unbroken
  // BiCGStab for the whole chunk).  The coarse levels use it: their solves are
  // 2-iteration smoothers plus one coarsest-level solve, and changing them would
  // move level1 counts for no measurement in hand.
  // ---------------------------------------------------------------------
  // CUDA-parity outer solver.
  //
  // CUDA runs the MG as a RIGHT PRECONDITIONER inside FGMRES(m=10) / CA-GCR
  // (cpp/cuda/qcu/include/lattice_clover_multigrid.h:5749-5825): z_j = M^-1 v_j,
  // w = A z_j, Arnoldi, then x += Z y.  M^-1 is one fine pre-smoothing pass
  // (nu_pre steps of MR, CG or Chebyshev, :5292-5345) followed by one coarse
  // cycle and a prolongation added with weight 1; post-smoothing is omitted on
  // purpose (:5378 "nu_post 平滑省略 (v1): 外层 FGMRES 承担剩余高频误差").
  //
  // This backend used to run the OPPOSITE structure - the MG corrected a running
  // BiCGStab residual in place, with analytic optimal damping - which is why no
  // measurement could attribute a difference to the hardware.  That path is gone.
  // ---------------------------------------------------------------------

  // nu_pre steps, same parameter and same default as CUDA
  // (_MG_MU_PRE_, default 4 at lattice_clover_multigrid.h:5293).
  int mu_pre_steps() const {
    // CUDA's parameter semantics: _MG_MU_PRE_ > 0 is the step count and 0 means
    // the default 4.  A zero-step (coarse-only) preconditioner was tried behind an
    // environment override and REMOVED: three compute-node jobs died with SIGTERM
    // ("force to exit immediately") inside that arm while the same batch without
    // the override completed, so the path is unstable.  A parameter that can kill
    // a job must not ship.
    const int configured = params_[kMgMuPre];
    const int steps = configured > 0 ? configured : 4;
    return std::min(steps, 8);
  }

  // One MR smoothing pass on level's (x, r), matching CUDA's MR branch
  // (:5319-5336): alpha = <A r, r> / <A r, A r>, x += alpha r, r -= alpha A r.
  // mg_dot_pair returns {<a,a>, <a,b>}, so both scalars arrive in ONE collective
  // - that is what makes four steps affordable on this backend.
  void mr_smooth(int level, int steps) {
    using C = Complex<Real>;
    LegacyMgLevel<Real> &state = levels_[level];
    const int64_t count = static_cast<int64_t>(state.x.size());
    const int64_t chunk = omp_chunk<Real>(count);
    for (int step = 0; step < steps; ++step) {
      apply_level(level, state.v, state.r);
      const auto pair = mg_dot_pair(state.v, state.r, distributed_);
      const Real denominator = pair[0].real();
      if (!std::isfinite(static_cast<double>(denominator)) ||
          denominator <= Real(0)) {
        // A vanishing image means this direction is exhausted; leaving the
        // smoother early is what CUDA's "zero or invalid denominator is a no-op"
        // does (:206-210).
        return;
      }
      const C alpha = pair[1] / pair[0];
#pragma omp parallel for schedule(static, chunk)
      for (int64_t index = 0; index < count; ++index) {
        const auto i = static_cast<std::size_t>(index);
        state.x[i] += alpha * state.r[i];
        state.r[i] -= alpha * state.v[i];
      }
    }
  }

  // M^-1 v: exactly CUDA's apply_mg_prec.  Zero, nu_pre MR steps on the incoming
  // vector, one coarse cycle on the residual that remains, then prolong and add
  // with weight 1 (CUDA takes t=1; the analytic damping belongs to the corrector
  // path and is deliberately not used here, because the outer FGMRES absorbs a
  // badly scaled direction).
  void apply_mg_preconditioner(std::vector<Complex<Real>> &out,
                               const std::vector<Complex<Real>> &in) {
    using C = Complex<Real>;
    MgProfileTimer timer(profile_, prof_coarse_correction_seconds_);
    ++prof_coarse_correction_calls_;
    LegacyMgLevel<Real> &fine = levels_[0];
    // levels_[0].x IS the FGMRES solution vector.  The preconditioner needs a
    // zeroed buffer of its own, so save and restore instead of borrowing it:
    // zeroing it here silently discarded every update FGMRES had accumulated,
    // which is exactly why the true residual grew across restarts (0.497 ->
    // 5.74) while the Givens estimate kept falling to 1e-3.  The basis probe
    // excluded the other explanations first: z_j were not parallel (overlap with
    // z_0 was 0.14-0.17), h_{j+1,j} was not degenerate (0.09-0.41) and h_jj was
    // ~1.0, i.e. M^-1 ~ A^-1, so one iteration should almost solve it.
    const std::vector<C> saved_solution = fine.x;
    std::fill(fine.x.begin(), fine.x.end(), C(0));
    fine.r = in;
    mr_smooth(0, mu_pre_steps());
    if (num_levels_ > 1) {
      LegacyMgLevel<Real> &child = levels_[1];
      restrict(0, child.rhs, fine.r);
      std::vector<C> coarse_correction(child.x.size(), C(0));
      cycle(1, child.rhs, coarse_correction);
      prolong(0, fine.correction, coarse_correction);
      const int64_t count = static_cast<int64_t>(fine.x.size());
      const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
      for (int64_t index = 0; index < count; ++index) {
        const auto i = static_cast<std::size_t>(index);
        fine.x[i] += fine.correction[i];
      }
    }
    out = fine.x;
    fine.x = saved_solution;
  }

  // Right-preconditioned FGMRES(m), structured after
  // cpp/cuda/qcu/include/lattice_clover_multigrid.h:5755-5900.
  void fgmres_solve(int &total_iterations, int &applications, Real &residual,
                    Real target) {
    using C = Complex<Real>;
    LegacyMgLevel<Real> &fine = levels_[0];
    const int m = 10;  // CUDA hardcodes 10 (lattice_clover_multigrid.h:5749)
    const int64_t count = static_cast<int64_t>(fine.x.size());
    const int64_t chunk = omp_chunk<Real>(count);
    std::vector<std::vector<C>> V(
        static_cast<std::size_t>(m + 1), std::vector<C>(static_cast<std::size_t>(count), C(0)));
    std::vector<std::vector<C>> Z(
        static_cast<std::size_t>(m), std::vector<C>(static_cast<std::size_t>(count), C(0)));
    std::vector<C> image(static_cast<std::size_t>(count), C(0));
    std::vector<C> hessenberg(static_cast<std::size_t>((m + 1) * m), C(0));
    std::vector<C> cosine(static_cast<std::size_t>(m), C(0));
    std::vector<C> sine(static_cast<std::size_t>(m), C(0));
    std::vector<C> givens(static_cast<std::size_t>(m + 1), C(0));
    std::vector<C> y(static_cast<std::size_t>(m), C(0));
    const auto hindex = [m](int row, int column) {
      return static_cast<std::size_t>(row * m + column);
    };

    // CUDA's preconditioner gate (:5767-5772): a six-step window, geometric mean
    // of the residual ratios, and two consecutive windows at or above 0.9 turn
    // the preconditioner off for the rest of the solve so a weak coarse space
    // stops burning cycles.  Z[j] = V[j] is exactly its "no preconditioner"
    // branch.
    // The window is CUDA's: six samples, and the statistic is the geometric mean of
    // the FIVE ratios between them, i.e. (r_6/r_1)^(1/6) anchored on the first
    // sample.  Dividing six ratios by six instead - which this did at first -
    // measures (r_6/r_0)^(1/6) and, at the same 0.9 threshold, tolerates a per-step
    // contraction of 0.900 instead of CUDA's 0.9^(6/5) = 0.8812.  Both are
    // permanent latches that change the whole Krylov basis, so a 2% difference in
    // where they fire is a real behavioural difference, not a rounding one.
    // The window state deliberately spans restarts, as CUDA's does.
    const int gate_window = 6;
    const Real gate_threshold = Real(0.9);
    int gate_count = 0;
    int gate_failures = 0;
    Real gate_log_sum = Real(0);
    Real gate_anchor_log = Real(0);
    bool precondition = num_levels_ > 1;
    Real previous_estimate = 0;

    total_iterations = 0;
    applications = 0;
    const int restarts = (levels_[0].max_iter + m - 1) / m;
    for (int restart = 0; restart < restarts; ++restart) {
      if (restart > 0) {
        residual = true_residual(0);
      }
      if (!std::isfinite(static_cast<double>(residual)) || residual <= target) {
        break;
      }
      const Real beta = residual;
      if (!(beta > Real(0))) {
        break;
      }
      const Real inverse_beta = Real(1) / beta;
#pragma omp parallel for schedule(static, chunk)
      for (int64_t index = 0; index < count; ++index) {
        V[0][static_cast<std::size_t>(index)] =
            fine.r[static_cast<std::size_t>(index)] * inverse_beta;
      }
      std::fill(hessenberg.begin(), hessenberg.end(), C(0));
      std::fill(givens.begin(), givens.end(), C(0));
      givens[0] = C(beta, 0);
      previous_estimate = beta;
      int used = 0;
      for (int j = 0; j < m; ++j) {
        if (precondition) {
          apply_mg_preconditioner(Z[static_cast<std::size_t>(j)],
                                  V[static_cast<std::size_t>(j)]);
          ++applications;
        } else {
          Z[static_cast<std::size_t>(j)] = V[static_cast<std::size_t>(j)];
        }
        apply_level(0, image, Z[static_cast<std::size_t>(j)]);
        std::vector<C> projections;
        mg_dot_many(V, image, j + 1, distributed_, projections);
        for (int i = 0; i <= j; ++i) {
          const C h = projections[static_cast<std::size_t>(i)];
          hessenberg[hindex(i, j)] = h;
#pragma omp parallel for schedule(static, chunk)
          for (int64_t index = 0; index < count; ++index) {
            const auto k = static_cast<std::size_t>(index);
            image[k] -= h * V[static_cast<std::size_t>(i)][k];
          }
        }
        const Real next = std::sqrt(mg_norm2(image, distributed_));
        hessenberg[hindex(j + 1, j)] = C(next, 0);
        // First-restart probe: if the preconditioned directions are nearly
        // parallel the Hessenberg collapses, y explodes and x walks away from the
        // solution while the Givens estimate still looks converged.  Print the
        // norm of z_j, the new Arnoldi length and the overlap with z_0 so that
        // "degenerate basis" can be confirmed or excluded from the data.
        if (restart == 0 && j < 3) {
          // The three reductions run on EVERY rank and only the print is guarded:
          // mg_norm2/mg_dot are MPI_Allreduce, so a rank-subset guard on the call
          // deadlocks every multi-rank run.  That is the same defect this file
          // already records for the old diagnose_correction (jobs 2247200 and
          // 2247255 hung right after the banner); it hung this probe too.
          const Real z_norm = std::sqrt(mg_norm2(Z[static_cast<std::size_t>(j)],
                                                 distributed_));
          const C z0_dot =
              mg_dot(Z[0], Z[static_cast<std::size_t>(j)], distributed_);
          const Real z0_norm = std::sqrt(mg_norm2(Z[0], distributed_));
          if (params_[kNodeRank] == 0) {
            std::fprintf(stderr,
                         "PYQCU::KUPL::MG_FGMRES_BASIS:\n "
                         "j=%d z_norm=%g next=%g h_diag=(%g,%g) "
                         "overlap_z0=%g\n",
                         j, static_cast<double>(z_norm),
                         static_cast<double>(next),
                         static_cast<double>(
                             hessenberg[hindex(j, j)].real()),
                         static_cast<double>(
                             hessenberg[hindex(j, j)].imag()),
                         static_cast<double>(z0_norm > Real(0)
                                                 ? std::abs(z0_dot) /
                                                       (z0_norm * z_norm)
                                                 : Real(0)));
            std::fflush(stderr);
          }
        }
        used = j + 1;
        if (next > Real(0)) {
          const Real inverse = Real(1) / next;
#pragma omp parallel for schedule(static, chunk)
          for (int64_t index = 0; index < count; ++index) {
            const auto k = static_cast<std::size_t>(index);
            V[static_cast<std::size_t>(j + 1)][k] = image[k] * inverse;
          }
        }
        for (int i = 0; i < j; ++i) {
          const C a = hessenberg[hindex(i, j)];
          const C b = hessenberg[hindex(i + 1, j)];
          hessenberg[hindex(i, j)] = cosine[static_cast<std::size_t>(i)] * a +
                                     sine[static_cast<std::size_t>(i)] * b;
          hessenberg[hindex(i + 1, j)] =
              -std::conj(sine[static_cast<std::size_t>(i)]) * a +
              cosine[static_cast<std::size_t>(i)] * b;
        }
        const C a = hessenberg[hindex(j, j)];
        const C b = hessenberg[hindex(j + 1, j)];
        const Real radius = std::sqrt(std::norm(a) + std::norm(b));
        if (!(radius > Real(0))) {
          break;
        }
        // The rotation [[c, s], [-conj(s), c]] is unitary only when c is REAL.
        // With c = conj(a)/radius the second row does not annihilate b -
        // -conj(s)a + c b = b(conj(a) - a)/radius - it only looked annihilated
        // because the next line overwrites that entry with 0.  CUDA gets this
        // right (lattice_clover_multigrid.h:5862-5866): c = |a|/radius is real,
        // s = (a/|a|)conj(b)/radius, and the diagonal keeps r*a/|a|.  The error
        // here scaled with Im(a); h_jj measured (1.0155, -0.0001), i.e. almost
        // real, which is why the solve still converged to 4e-5 and why this was
        // invisible in the results while remaining wrong for a complex diagonal.
        const Real magnitude = std::abs(a);
        const C phase = magnitude > Real(0) ? a / magnitude : C(1);
        cosine[static_cast<std::size_t>(j)] = C(magnitude / radius, 0);
        sine[static_cast<std::size_t>(j)] = phase * std::conj(b) / radius;
        hessenberg[hindex(j, j)] = phase * C(radius, 0);
        hessenberg[hindex(j + 1, j)] = C(0, 0);
        const C g0 = givens[static_cast<std::size_t>(j)];
        const C g1 = givens[static_cast<std::size_t>(j + 1)];
        givens[static_cast<std::size_t>(j)] =
            cosine[static_cast<std::size_t>(j)] * g0 +
            sine[static_cast<std::size_t>(j)] * g1;
        givens[static_cast<std::size_t>(j + 1)] =
            -std::conj(sine[static_cast<std::size_t>(j)]) * g0 +
            cosine[static_cast<std::size_t>(j)] * g1;
        ++total_iterations;
        const Real estimate = std::abs(givens[static_cast<std::size_t>(j + 1)]);
        if (precondition && estimate > Real(0)) {
          if (gate_count == 0) {
            // first sample of a window: it is the anchor, not a ratio
            gate_anchor_log = std::log(estimate);
          } else if (previous_estimate > Real(0)) {
            gate_log_sum += std::log(estimate / previous_estimate);
          }
          ++gate_count;
          if (gate_count >= gate_window) {
            // (r_6/r_1)^(1/6): five ratios, divided by the window length, exactly
            // as CUDA's gfac at lattice_clover_multigrid.h:5886.
            const Real mean =
                std::exp((std::log(estimate) - gate_anchor_log) /
                         Real(gate_window));
            gate_log_sum = Real(0);
            gate_count = 0;
            if (mean >= gate_threshold) {
              ++gate_failures;
              if (gate_failures >= 2) {
                precondition = false;
                if (params_[kVerbose] != 0) {
                  std::fprintf(stdout,
                               "PYQCU::KUPL::CLOVER_MULTIGRID_DEBUG:\n "
                               "MG preconditioner disabled by the contraction "
                               "gate at iteration %d (window mean %g)\n",
                               total_iterations, static_cast<double>(mean));
                }
              }
            } else {
              gate_failures = 0;
            }
          }
        }
        previous_estimate = estimate;
        if (estimate <= target) {
          break;
        }
      }
      if (used <= 0) {
        break;
      }
      // Back substitution on the triangular factor, then x += Z y.
      std::fill(y.begin(), y.end(), C(0));
      for (int i = used - 1; i >= 0; --i) {
        C sum = givens[static_cast<std::size_t>(i)];
        for (int k = i + 1; k < used; ++k) {
          sum -= hessenberg[hindex(i, k)] * y[static_cast<std::size_t>(k)];
        }
        const C diagonal = hessenberg[hindex(i, i)];
        // 1e-30, not numeric_limits<Real>::min(): the latter is ~2e-308, so a
        // tiny-but-representable pivot would still divide and y could overflow.
        // CUDA floors the same quantity at 1e-30 (:5880 for its window factor).
        y[static_cast<std::size_t>(i)] =
            std::abs(diagonal) > Real(1e-30) ? sum / diagonal : C(0);
      }
      for (int i = 0; i < used; ++i) {
        const C coefficient = y[static_cast<std::size_t>(i)];
        if (std::abs(coefficient) == Real(0)) {
          continue;
        }
#pragma omp parallel for schedule(static, chunk)
        for (int64_t index = 0; index < count; ++index) {
          const auto k = static_cast<std::size_t>(index);
          fine.x[k] += coefficient * Z[static_cast<std::size_t>(i)][k];
        }
      }
      // Unconditional, rank 0 only: when FGMRES misses its target the sequence of
      // restart residuals is the only thing that separates "stagnating", "slowly
      // converging" and "diverging", and a diagnostic behind kVerbose is exactly
      // the kind that is absent when it is needed.
      if (params_[kNodeRank] == 0) {
        std::fprintf(stderr,
                     "PYQCU::KUPL::MG_FGMRES_TRACE:\n "
                     "restart=%d iterations=%d before=%g estimate=%g "
                     "applications=%d preconditioner=%d\n",
                     restart, total_iterations, static_cast<double>(beta),
                     static_cast<double>(std::abs(
                         givens[static_cast<std::size_t>(used)])),
                     applications, precondition ? 1 : 0);
        std::fflush(stderr);
      }
    }
    residual = true_residual(0);
    // The fine level's count belongs to this solver: FGMRES does not go through
    // bicgstab_chunk(0, ...), so without this the row would report level0=0 while
    // level1/level2 showed real work - the most misleading possible output.
    level_iterations_[0] += static_cast<long long>(total_iterations);
    if (params_[kVerbose] != 0) {
      std::fprintf(stdout,
                   "PYQCU::KUPL::CLOVER_MULTIGRID_DEBUG:\n "
                   "fgmres final iterations=%d residual=%g target=%g "
                   "applications=%d preconditioner=%d\n",
                   total_iterations, static_cast<double>(residual),
                   static_cast<double>(target), applications,
                   precondition ? 1 : 0);
    }
  }

  bool bicgstab_chunk(int level, int iterations, bool force_convergence) {
    using C = Complex<Real>;
    LegacyMgLevel<Real> &state = levels_[level];
    last_chunk_iterations_ = 0;
    last_chunk_converged_ = false;
    struct IterationRecorder {
      std::vector<long long> &counts;
      int level;
      int &iterations;
      ~IterationRecorder() {
        if (level >= 0 && static_cast<std::size_t>(level) < counts.size()) {
          counts[static_cast<std::size_t>(level)] += iterations;
        }
      }
    } iteration_recorder{level_iterations_, level, last_chunk_iterations_};
    const int64_t count = static_cast<int64_t>(state.x.size());
    const int64_t chunk = omp_chunk<Real>(count);
    const C zero(0);
    const Real target = state.target;
    const int restart = state.restart;
    int performed = 0;
    while (performed < iterations) {
      // Length of this Krylov segment: to the chunk end, or to the next restart
      // boundary, whichever comes first.  `state.since_restart` is carried
      // across chunks, so a segment boundary falls inside a chunk when the
      // correction interval is shorter than the restart length - which is the
      // case the decoupling exists for.
      const int segment =
          restart > 0
              ? std::min(iterations - performed,
                         std::max(1, restart - state.since_restart))
              : iterations - performed;
      // The restart: rhat = r, p = v = 0, rho_old = alpha = omega = 1.  x and r
      // are deliberately untouched.
#pragma omp parallel for schedule(static, chunk)
      for (int64_t i = 0; i < count; ++i) {
        state.rhat[static_cast<std::size_t>(i)] =
            state.r[static_cast<std::size_t>(i)];
        state.p[static_cast<std::size_t>(i)] = zero;
        state.v[static_cast<std::size_t>(i)] = zero;
      }
      C rho_old(1);
      C alpha(1);
      C omega(1);
      for (int iteration = 0; iteration < segment; ++iteration) {
        if (std::sqrt(mg_norm2(state.r, distributed_)) <= target) {
          last_chunk_iterations_ = performed + iteration;
          last_chunk_converged_ = true;
          return true;
        }
        const C rho = mg_dot(state.rhat, state.r, distributed_);
        if (std::abs(rho) <= std::numeric_limits<Real>::min() ||
            std::abs(omega) <= std::numeric_limits<Real>::min()) {
          if (params_[kVerbose] != 0) {
            std::fprintf(stderr,
                         "PYQCU::KUPL::CLOVER_MULTIGRID_DEBUG:\n "
                         "level=%d iteration=%d rho-breakdown rho=(%g,%g) "
                         "omega=(%g,%g)\n",
                         level, iteration,
                         static_cast<double>(rho.real()),
                         static_cast<double>(rho.imag()),
                         static_cast<double>(omega.real()),
                         static_cast<double>(omega.imag()));
          }
          last_chunk_iterations_ = performed + iteration;
          return false;
        }
        const C beta = (rho / rho_old) * (alpha / omega);
#pragma omp parallel for schedule(static, chunk)
        for (int64_t i = 0; i < count; ++i) {
          state.p[static_cast<std::size_t>(i)] =
              state.r[static_cast<std::size_t>(i)] +
              multiply_complex(
                  beta, state.p[static_cast<std::size_t>(i)] -
                            multiply_complex(
                                omega, state.v[static_cast<std::size_t>(i)]));
        }
        apply_level(level, state.v, state.p);
        const C denominator = mg_dot(state.rhat, state.v, distributed_);
        if (std::abs(denominator) <= std::numeric_limits<Real>::min()) {
          if (params_[kVerbose] != 0) {
            std::fprintf(stderr,
                         "PYQCU::KUPL::CLOVER_MULTIGRID_DEBUG:\n "
                         "level=%d iteration=%d alpha-breakdown\n",
                         level, iteration);
          }
          last_chunk_iterations_ = performed + iteration;
          return false;
        }
        alpha = rho / denominator;
#pragma omp parallel for schedule(static, chunk)
        for (int64_t i = 0; i < count; ++i) {
          state.s[static_cast<std::size_t>(i)] =
              state.r[static_cast<std::size_t>(i)] -
              multiply_complex(
                  alpha, state.v[static_cast<std::size_t>(i)]);
        }
        if (std::sqrt(mg_norm2(state.s, distributed_)) <= target) {
#pragma omp parallel for schedule(static, chunk)
          for (int64_t i = 0; i < count; ++i) {
            state.x[static_cast<std::size_t>(i)] +=
                multiply_complex(alpha, state.p[static_cast<std::size_t>(i)]);
            state.r[static_cast<std::size_t>(i)] =
                state.s[static_cast<std::size_t>(i)];
          }
          last_chunk_iterations_ = performed + iteration + 1;
          last_chunk_converged_ = true;
          return true;
        }
        apply_level(level, state.v, state.s);
        C tt(0);
        C v_s(0);
        if (fused_bicg_dots_enabled()) {
          const auto pair = mg_dot_pair(state.v, state.s, distributed_);
          tt = pair[0];
          v_s = pair[1];
        } else {
          tt = mg_dot(state.v, state.v, distributed_);
          v_s = mg_dot(state.v, state.s, distributed_);
        }
        if (std::abs(tt) <= std::numeric_limits<Real>::min()) {
          if (params_[kVerbose] != 0) {
            std::fprintf(stderr,
                         "PYQCU::KUPL::CLOVER_MULTIGRID_DEBUG:\n "
                         "level=%d iteration=%d tt-breakdown\n",
                         level, iteration);
          }
          last_chunk_iterations_ = performed + iteration;
          return false;
        }
        omega = v_s / tt;
        if (std::abs(omega) <= std::numeric_limits<Real>::min()) {
          if (params_[kVerbose] != 0) {
            std::fprintf(stderr,
                         "PYQCU::KUPL::CLOVER_MULTIGRID_DEBUG:\n "
                         "level=%d iteration=%d omega-breakdown\n",
                         level, iteration);
          }
          last_chunk_iterations_ = performed + iteration;
          return false;
        }
#pragma omp parallel for schedule(static, chunk)
        for (int64_t i = 0; i < count; ++i) {
          state.x[static_cast<std::size_t>(i)] +=
              multiply_complex(alpha, state.p[static_cast<std::size_t>(i)]) +
              multiply_complex(omega,
                               state.s[static_cast<std::size_t>(i)]);
          state.r[static_cast<std::size_t>(i)] =
              state.s[static_cast<std::size_t>(i)] -
              multiply_complex(omega,
                               state.v[static_cast<std::size_t>(i)]);
        }
        rho_old = rho;
      }
      performed += segment;
      if (restart > 0) {
        state.since_restart += segment;
        if (state.since_restart >= restart) {
          // A restart is due at the next segment start; keep the counter in
          // [0, restart) and let the loop re-initialise the recurrence.
          state.since_restart = 0;
        }
      }
    }
    const bool converged = std::sqrt(mg_norm2(state.r, distributed_)) <= target;
    last_chunk_iterations_ = iterations;
    last_chunk_converged_ = converged;
    if (force_convergence && !converged) {
      return false;
    }
    return true;
  }

  void cycle(int level, const std::vector<Complex<Real>> &rhs,
             std::vector<Complex<Real>> &out) {
    LegacyMgLevel<Real> &state = levels_[level];
    std::fill(state.x.begin(), state.x.end(), Complex<Real>(0));
    state.rhs = rhs;
    state.r = rhs;
    state.target = state.tolerance *
        std::sqrt(mg_norm2(state.rhs, distributed_));
    if (level == num_levels_ - 1) {
      const bool converged = bicgstab_chunk(level, state.max_iter, true);
      // The reduction is computed unconditionally: a collective behind a flag
      // that a caller could set per rank is the deadlock shape this file already
      // records twice.
      const Real coarsest_residual_norm =
          std::sqrt(mg_norm2(state.r, distributed_));
      if (params_[kVerbose] != 0) {
        const Real residual_norm = coarsest_residual_norm;
        std::fprintf(
            stdout,
            "PYQCU::KUPL::CLOVER_MULTIGRID_DEBUG:\n "
            "coarsest level=%d iterations=%d residual=%g target=%g\n",
            level, last_chunk_iterations_, static_cast<double>(residual_norm),
            static_cast<double>(state.target));
      }
      if (!converged) {
        const Real rhs_norm = std::sqrt(mg_norm2(state.rhs, distributed_));
        const Real residual_norm = std::sqrt(mg_norm2(state.r, distributed_));
        throw std::runtime_error(
            "legacy Clover-MG coarsest BiCGStab failed: rhs=" +
            std::to_string(static_cast<double>(rhs_norm)) +
            " residual=" +
            std::to_string(static_cast<double>(residual_norm)));
      }
    } else {
      bicgstab_chunk(level, 2, false);
      state.r = state.rhs;
      apply_level(level, state.v, state.x);
      for (std::size_t i = 0; i < state.r.size(); ++i) {
        state.r[i] = state.rhs[i] - state.v[i];
      }
      LegacyMgLevel<Real> &child = levels_[level + 1];
      restrict(level, child.rhs, state.r);
      std::vector<Complex<Real>> coarse_correction(child.x.size());
      cycle(level + 1, child.rhs, coarse_correction);
      prolong(level, state.correction, coarse_correction);
      for (std::size_t i = 0; i < state.x.size(); ++i) {
        state.x[i] += state.correction[i];
      }
      apply_level(level, state.v, state.x);
      for (std::size_t i = 0; i < state.r.size(); ++i) {
        state.r[i] = state.rhs[i] - state.v[i];
      }
      bicgstab_chunk(level, 2, false);
    }
    out = state.x;
  }

  // KUPL_MG_DEBUG dof-level audit of the coarse<->fine mapping, run once per
  // transition.  It answers the only question a mapping bug can hide behind:
  // for a chosen coarse site, which fine dofs does restrict() gather and which
  // fine dofs does prolong() scatter to, and are they the same set?
  //
  // prolong() of a unit coarse dof (E_idx, c) must be supported on exactly
  //   { (e_idx, x, y, z, t) : e_idx in [0,e), x in [bx*xc, bx*xc+bx),
  //     y in [by*yc, ...), z in [bz*zc, ...), t in [bt*tc, ...) }
  // and restrict() of that image must return the same unit coarse dof and
  // nothing else.  A permuted axis, an off-by-one block origin, a swapped
  // (x<->t) fastest index, a prolong-only parity shift or a wrong component
  // stride all break one of the two.
  //
  // restrict()/prolong() are pure local loops (no MPI, no halo), so this runs
  // on every rank without a collective; only rank 0 prints.
  void audit_transfer_mapping(int level, int bx, int by, int bz, int bt) {
    // Once per (level, block), not once per process: pyqcu.test runs rows
    // in-process, so a function-local static made the audit silent for every
    // rank's second and later solve while still looking "already done".
    if (level < 0 ||
        static_cast<std::size_t>(level) >= audited_levels_.size()) {
      return;
    }
    if (audited_levels_[static_cast<std::size_t>(level)]) {
      return;
    }
    audited_levels_[static_cast<std::size_t>(level)] = true;
    // A debug-only audit must never be able to take a production solve down.
    try {
      audit_transfer_mapping_body(level, bx, by, bz, bt);
    } catch (const std::exception &error) {
      if (params_[kNodeRank] == 0) {
        std::fprintf(stderr,
                     "PYQCU::KUPL::MG_MAP_AUDIT:\n level=%d audit failed: %s\n",
                     level, error.what());
        std::fflush(stderr);
      }
    }
  }

  void audit_transfer_mapping_body(int level, int bx, int by, int bz, int bt) {
    LegacyMgLevel<Real> &p = levels_[level];
    LegacyMgLevel<Real> &c = levels_[level + 1];
    const int64_t fine_volume = mg_volume(p.dims);
    const int64_t coarse_volume = mg_volume(c.dims);
    const int64_t block_sites =
        static_cast<int64_t>(bx) * by * bz * bt;
    const auto coarse_site = [&](int xc, int yc, int zc, int tc) {
      return ((static_cast<int64_t>(xc) * c.dims[1] + yc) * c.dims[2] + zc) *
                 c.dims[3] +
             tc;
    };
    // Two centres: the local origin block (exercises the wrap of the first
    // block) and a block whose global fine indices are strictly interior.
    std::array<std::array<int, kDim>, 2> centres{};
    centres[0] = {0, 0, 0, 0};
    for (int dim = 0; dim < kDim; ++dim) {
      centres[1][dim] = c.dims[dim] > 2 ? c.dims[dim] / 2 : 0;
    }
    std::vector<Complex<Real>> seed(c.x.size());
    std::vector<Complex<Real>> image(p.x.size());
    std::vector<Complex<Real>> back(c.x.size());
    Real max_roundtrip = 0;
    Real max_outside = 0;
    int mismatches = 0;
    for (const auto &centre : centres) {
      const int64_t site = coarse_site(centre[0], centre[1], centre[2],
                                       centre[3]);
      if (params_[kNodeRank] == 0) {
        std::fprintf(
            stdout,
            "PYQCU::KUPL::MG_MAP_AUDIT:\n "
            "level=%d coarse_site=(%d,%d,%d,%d) linear=%lld block=(%d,%d,%d,%d) "
            "block_sites=%lld gathers=%lld dofs\n",
            level, centre[0], centre[1], centre[2], centre[3],
            static_cast<long long>(site), bx, by, bz, bt,
            static_cast<long long>(block_sites),
            static_cast<long long>(block_sites * p.E));
      }
      for (int E_idx = 0; E_idx < c.E; ++E_idx) {
        std::fill(seed.begin(), seed.end(), Complex<Real>(0));
        seed[static_cast<std::size_t>(
            static_cast<int64_t>(E_idx) * coarse_volume + site)] =
            Complex<Real>(1);
        std::fill(image.begin(), image.end(), Complex<Real>(0));
        prolong(level, image, seed);
        // support scan: which (e_idx, x, y, z, t) are non-zero
        int64_t support = 0;
        int64_t in_block = 0;
        int64_t outside = 0;
        int first_component = -1;
        int last_component = -1;
        int64_t first_site = -1;
        int64_t last_site = -1;
        for (int e_idx = 0; e_idx < p.E; ++e_idx) {
          const Complex<Real> *plane =
              image.data() + static_cast<int64_t>(e_idx) * fine_volume;
          for (int64_t s = 0; s < fine_volume; ++s) {
            if (std::abs(plane[s]) == Real(0)) {
              continue;
            }
            ++support;
            if (first_component < 0) {
              first_component = e_idx;
              first_site = s;
            }
            last_component = e_idx;
            last_site = s;
            // decode the fine site (t fastest, then z, y, x)
            int64_t remainder = s;
            const int t = static_cast<int>(remainder % p.dims[3]);
            remainder /= p.dims[3];
            const int z = static_cast<int>(remainder % p.dims[2]);
            remainder /= p.dims[2];
            const int y = static_cast<int>(remainder % p.dims[1]);
            const int x = static_cast<int>(remainder / p.dims[1]);
            const bool inside =
                x / bx == centre[0] && y / by == centre[1] &&
                z / bz == centre[2] && t / bt == centre[3];
            if (inside) {
              ++in_block;
            } else {
              ++outside;
            }
          }
        }
        std::fill(back.begin(), back.end(), Complex<Real>(0));
        restrict(level, back, image);
        Real roundtrip = 0;
        Real leak = 0;
        for (int64_t i = 0; i < coarse_volume * c.E; ++i) {
          const Real expected =
              i == static_cast<int64_t>(E_idx) * coarse_volume + site
                  ? Real(1)
                  : Real(0);
          const Real error = std::abs(back[static_cast<std::size_t>(i)] -
                                      Complex<Real>(expected));
          if (expected == Real(1)) {
            roundtrip = std::max(roundtrip, error);
          } else {
            leak = std::max(leak, error);
          }
        }
        max_roundtrip = std::max(max_roundtrip, roundtrip);
        max_outside = std::max(max_outside, leak);
        // A correct transfer must pass all three checks, and each is stated in
        // units that a correct transfer can actually satisfy:
        //  (1) the image is supported on exactly the block: every one of the
        //      block_sites * p.E (component, site) pairs and nothing else;
        //  (2) no mass outside the block (fp32 round-off is tolerated);
        //  (3) restrict(prolong(unit)) returns the unit - the pair is adjoint.
        // `first_component/last_component` are printed for diagnosis only: a
        // non-diagonal null-vector basis legitimately spreads one coarse dof
        // over all p.E components, so requiring a single component would fail
        // on every correct implementation (that was the old assertion's defect).
        const bool ok = support == block_sites * p.E && outside == 0 &&
                        roundtrip < Real(1e-4) && leak < Real(1e-4);
        if (!ok) {
          ++mismatches;
        }
        if (params_[kNodeRank] == 0) {
          std::fprintf(
              stdout,
              "PYQCU::KUPL::MG_MAP_AUDIT:\n "
              "level=%d E_idx=%d scatter=%lld in_block=%lld outside_block=%lld "
              "components=[%d,%d] first_fine=%lld last_fine=%lld "
              "roundtrip_err=%g leak=%g ok=%d\n",
              level, E_idx, static_cast<long long>(support),
              static_cast<long long>(in_block),
              static_cast<long long>(outside), first_component, last_component,
              static_cast<long long>(first_site),
              static_cast<long long>(last_site),
              static_cast<double>(roundtrip), static_cast<double>(leak),
              ok ? 1 : 0);
        }
      }
    }
    if (params_[kNodeRank] == 0) {
      std::fprintf(
          stdout,
          "PYQCU::KUPL::MG_MAP_AUDIT:\n "
          "level=%d checked_coarse_sites=%zu block_sites=%lld "
          "max_roundtrip_err=%g max_leak=%g mismatches=%d "
          "gather_set_equals_scatter_set=%d\n",
          level, centres.size(), static_cast<long long>(block_sites),
          static_cast<double>(max_roundtrip), static_cast<double>(max_outside),
          mismatches, mismatches == 0 ? 1 : 0);
      std::fflush(stdout);
    }
  }

  Real true_residual(int level) {
    LegacyMgLevel<Real> &state = levels_[level];
    apply_level(level, state.v, state.x);
    const int64_t count = static_cast<int64_t>(state.r.size());
    const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t i = 0; i < count; ++i) {
      const auto index = static_cast<std::size_t>(i);
      state.r[index] = state.rhs[index] - state.v[index];
    }
    return std::sqrt(mg_norm2(state.r, distributed_));
  }

  void print_level_iterations() const {
    if (params_[kNodeRank] != 0) {
      return;
    }
    std::fprintf(stderr, "PYQCU::KUPL::MG_LEVEL_ITERS:\n");
    for (std::size_t level = 0; level < level_iterations_.size(); ++level) {
      std::fprintf(stderr, " level%zu=%lld", level,
                   static_cast<long long>(level_iterations_[level]));
    }
    std::fprintf(stderr, " total=%lld accepted_corrections=%d\n",
                 static_cast<long long>(last_total_iterations_),
                 last_accepted_corrections_);
  }

  void print_profile(MgProfileClock::time_point start) const {
    if (!profile_ || params_[kNodeRank] != 0) {
      return;
    }
    const double total =
        std::chrono::duration<double>(MgProfileClock::now() - start).count();
    std::fprintf(
        stderr,
        "PYQCU::KUPL::MG_PROFILE:\n total=%.6f fine_apply=%.6f "
        "fine_calls=%lld coarse_apply=%.6f coarse_calls=%lld "
        "restrict=%.6f restrict_calls=%lld prolong=%.6f prolong_calls=%lld "
        "coarse_correction=%.6f coarse_correction_calls=%lld "
        "prepare=%.6f store=%.6f full_residual=%.6f\n",
        total, prof_fine_apply_seconds_,
        static_cast<long long>(prof_fine_apply_calls_),
        prof_coarse_apply_seconds_,
        static_cast<long long>(prof_coarse_apply_calls_),
        prof_restrict_seconds_, static_cast<long long>(prof_restrict_calls_),
        prof_prolong_seconds_, static_cast<long long>(prof_prolong_calls_),
        prof_coarse_correction_seconds_,
        static_cast<long long>(prof_coarse_correction_calls_),
        prof_prepare_seconds_, prof_store_seconds_,
        prof_full_residual_seconds_);
  }

  LatticeSetBase &set_;
  int last_chunk_iterations_ = 0;
  bool last_chunk_converged_ = false;
  std::vector<long long> level_iterations_;
  long long last_total_iterations_ = 0;
  int last_accepted_corrections_ = 0;
  const void *gauge_ = nullptr;
  const void *clover_even_ = nullptr;
  const void *clover_odd_ = nullptr;
  const void *clover_even_inv_ = nullptr;
  const void *clover_odd_inv_ = nullptr;
  const void *fermion_in_ = nullptr;
  long long *set_ptrs_ = nullptr;
  const int32_t *params_ = nullptr;
  int num_levels_ = 1;
  bool distributed_ = false;
  bool deflate_ = false;
  bool compact_fine_ = false;
  bool profile_ = false;
  bool require_correction_ = false;
  double prof_fine_apply_seconds_ = 0;
  double prof_coarse_apply_seconds_ = 0;
  double prof_restrict_seconds_ = 0;
  double prof_prolong_seconds_ = 0;
  double prof_coarse_correction_seconds_ = 0;
  double prof_prepare_seconds_ = 0;
  double prof_store_seconds_ = 0;
  double prof_full_residual_seconds_ = 0;
  int64_t prof_fine_apply_calls_ = 0;
  int64_t prof_coarse_apply_calls_ = 0;
  int64_t prof_restrict_calls_ = 0;
  int64_t prof_prolong_calls_ = 0;
  int64_t prof_coarse_correction_calls_ = 0;
  std::array<LegacyMgLevel<Real>, 5> levels_{};
  //: Which levels have already had the dof-level transfer audit printed.
  std::array<bool, 5> audited_levels_{};
  std::vector<Complex<Real>> canonical_input_;
  std::vector<Complex<Real>> canonical_output_;
};

} // namespace

template <typename Real>
void clover_multigrid_solve(
    LatticeSetBase &set, void *fermion_out, const void *fermion_in,
    const void *gauge, const void *clover_even, const void *clover_odd,
    const void *clover_even_inv, const void *clover_odd_inv, void *set_ptrs,
    const int32_t *params) {
  if (gauge == nullptr || clover_even == nullptr ||
      clover_odd == nullptr || clover_even_inv == nullptr ||
      clover_odd_inv == nullptr || fermion_in == nullptr ||
      set_ptrs == nullptr || params == nullptr) {
    throw std::runtime_error("legacy Clover-MG received a null pointer");
  }
  if (params[kMgUseGcr] != 0) {
    throw std::runtime_error(
        "KUPL legacy MG rejects CUDA-only GCR/cycle mode bits");
  }
  if (mg_env_enabled("KUPL_REQUIRE_MG_CORRECTION") &&
      (params[kMgNumLevel] <= 1 || params[kMgLevel1NumRestart] == 0)) {
    throw std::runtime_error(
        "KUPL_REQUIRE_MG_CORRECTION=1 requires levels>1 and restart>0");
  }
  // A disabled hierarchy must use the optimized parity-Schur solver rather
  // than the much slower unpreconditioned full-site Krylov loop.
  if (params[kMgNumLevel] <= 1 || params[kMgLevel1NumRestart] == 0) {
    if (params[kVerbose] != 0) {
      std::fprintf(stdout,
                   "PYQCU::KUPL::CLOVER_MULTIGRID:\n "
                   "hierarchy disabled; using parity-Schur BiCGStab\n");
    }
    set.clover_bistabcg(fermion_out, fermion_in, gauge, clover_even,
                        clover_odd, clover_even_inv, clover_odd_inv);
    return;
  }
  LegacyCloverMultigrid<Real> solver(
      set, gauge, clover_even, clover_odd, clover_even_inv, clover_odd_inv,
      fermion_in, set_ptrs, params);
  solver.solve(fermion_out, params[kUseInitialGuess] != 0);
}

template <typename Real>
int clover_multigrid_verify(
    LatticeSetBase &set, void *fermion_out, const void *fermion_in,
    const void *gauge, const void *clover_even, const void *clover_odd,
    const void *clover_even_inv, const void *clover_odd_inv, void *set_ptrs,
    const int32_t *params) {
  (void)fermion_out;
  (void)fermion_in;
  if (set_ptrs == nullptr || params == nullptr) {
    return 2;
  }
  auto *table = static_cast<long long *>(set_ptrs);
  // _MG_NUM_LEVEL_ is only validated inside LegacyCloverMultigrid, but the
  // coarse-asset probe below reads table[kSetPtrsCoarseBase + 4*transition + 0
  // ..3] before that constructor runs.  Clamp the level count to what the
  // 100-entry set_ptrs table can describe first (same rule as the CUDA
  // verifier in apply_clover_multigrid_verify.cu): _MG_NUM_LEVEL_ = 21 used to
  // read table[109], nine slots past the ABI table, before failing.
  const int num_levels = params[kMgNumLevel];
  if (num_levels < 1 ||
      num_levels > 1 + (kSetPtrsSize - kSetPtrsCoarseBase) / 4) {
    return 2;
  }
  for (int transition = 0; transition + 1 < num_levels; ++transition) {
    const int base = kSetPtrsCoarseBase + 4 * transition;
    if (table[base + 0] == 0 || table[base + 1] == 0 ||
        table[base + 2] == 0 || table[base + 3] == 0) {
      return 1;
    }
  }
  LegacyCloverMultigrid<Real> verifier(
      set, gauge, clover_even, clover_odd, clover_even_inv, clover_odd_inv,
      fermion_in, set_ptrs, params);
  return verifier.verify() ? 0 : 1;
}

template void clover_multigrid_solve<float>(
    LatticeSetBase &, void *, const void *, const void *, const void *,
    const void *, const void *, const void *, void *, const int32_t *);
template void clover_multigrid_solve<double>(
    LatticeSetBase &, void *, const void *, const void *, const void *,
    const void *, const void *, const void *, void *, const int32_t *);
template int clover_multigrid_verify<float>(
    LatticeSetBase &, void *, const void *, const void *, const void *,
    const void *, const void *, const void *, void *, const int32_t *);
template int clover_multigrid_verify<double>(
    LatticeSetBase &, void *, const void *, const void *, const void *,
    const void *, const void *, const void *, void *, const int32_t *);

} // namespace kupl::qcu
