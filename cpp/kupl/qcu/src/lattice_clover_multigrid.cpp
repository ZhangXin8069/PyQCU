#include "kupl/lattice_clover_multigrid.h"
#include "kupl/lattice_linalg.h"
#include "kupl/lattice_set.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace kupl::qcu {
namespace {

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

inline int64_t mg_volume(const std::array<int, kDim> &dims) {
  return static_cast<int64_t>(dims[0]) * dims[1] * dims[2] * dims[3];
}

template <typename Real> struct LegacyMgLevel {
  int E = 0;
  std::array<int, kDim> dims{};
  int max_iter = 100;
  int correction_interval = 5;
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
    num_levels_ = params_[kMgNumLevel];
    distributed_ = params_[kNodeSize] > 1;
    if (num_levels_ < 1 || num_levels_ > 5) {
      throw std::runtime_error("legacy Clover-MG level count must be 1..5");
    }
    if (params_[kMgUseGcr] != 0) {
      throw std::runtime_error(
          "KUPL legacy MG rejects CUDA-only GCR/cycle mode bits");
    }
    deflate_ = params_[kMgUseDeflate] != 0;
    levels_[0].E = kVector;
    levels_[0].dims = {
        params_[kLatX], params_[kLatY], params_[kLatZ], params_[kLatT]};
    levels_[0].max_iter = params_[kMaxIter];
    if (levels_[0].max_iter <= 0) {
      throw std::runtime_error(
          "legacy Clover-MG max iterations must be positive");
    }
    levels_[0].correction_interval =
        params_[kMgLevel1NumRestart] > 0
            ? params_[kMgLevel1NumRestart]
            : 3;
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
      levels_[level].correction_interval =
          params_[base + 7] > 0 ? params_[base + 7] : 3;
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
    canonical_input_.resize(
        static_cast<std::size_t>(2 * kVector * params_[kLatX] *
                                 params_[kLatY] * params_[kLatZ] *
                                 (params_[kLatT] / 2)));
    canonical_output_.resize(canonical_input_.size());
  }

  void solve(void *fermion_out, bool keep_initial) {
    if (fermion_out == nullptr) {
      throw std::runtime_error("legacy Clover-MG output is null");
    }
    canonical_to_full(levels_[0].rhs,
                      static_cast<const Complex<Real> *>(fermion_in_));
    if (keep_initial) {
      canonical_to_full(levels_[0].x,
                        static_cast<const Complex<Real> *>(fermion_out));
    } else {
      std::fill(levels_[0].x.begin(), levels_[0].x.end(), Complex<Real>(0));
    }
    apply_level(0, levels_[0].v, levels_[0].x);
    for (std::size_t i = 0; i < levels_[0].r.size(); ++i) {
      levels_[0].r[i] = levels_[0].rhs[i] - levels_[0].v[i];
    }
    const Real rhs_norm = std::sqrt(mg_norm2(levels_[0].rhs, distributed_));
    Real residual = std::sqrt(mg_norm2(levels_[0].r, distributed_));
    if (rhs_norm == Real(0)) {
      std::fill(levels_[0].x.begin(), levels_[0].x.end(), Complex<Real>(0));
      full_to_canonical(static_cast<Complex<Real> *>(fermion_out),
                        levels_[0].x);
      return;
    }
    const Real target = levels_[0].tolerance * rhs_norm;
    levels_[0].target = target;
    int total_iterations = 0;
    bool corrections_enabled =
        num_levels_ > 1 && levels_[0].correction_interval > 0;
    const auto reject_correction = [&](int level) {
      LegacyMgLevel<Real> &state = levels_[level];
      for (std::size_t i = 0; i < state.x.size(); ++i) {
        state.x[i] -= state.correction[i];
      }
      return true_residual(level);
    };
    int since_correction = 0;
    if (corrections_enabled && deflate_ && residual > target) {
      const Real before = residual;
      const Real candidate = apply_coarse_correction(0);
      if (!std::isfinite(static_cast<double>(candidate)) ||
          candidate >= before) {
        residual = reject_correction(0);
        corrections_enabled = false;
      } else {
        residual = candidate;
      }
      if (params_[kVerbose] != 0) {
        std::fprintf(stdout,
                     "PYQCU::KUPL::CLOVER_MULTIGRID_DEBUG:\n "
                     "deflation residual=%g target=%g corrections=%d\n",
                     static_cast<double>(residual),
                     static_cast<double>(target),
                     corrections_enabled ? 1 : 0);
      }
    }
    while (total_iterations < levels_[0].max_iter && residual > target) {
      const int remaining = levels_[0].max_iter - total_iterations;
      const int chunk =
          corrections_enabled
              ? std::min(remaining,
                         levels_[0].correction_interval - since_correction)
              : remaining;
      bicgstab_chunk(0, chunk, false);
      const int completed = last_chunk_iterations_;
      total_iterations += completed;
      since_correction += completed;
      residual = true_residual(0);
      if (params_[kVerbose] != 0 &&
          (total_iterations < 20 || total_iterations % 50 == 0)) {
        std::fprintf(stdout,
                     "PYQCU::KUPL::CLOVER_MULTIGRID_DEBUG:\n "
                     "iterations=%d residual=%g target=%g\n",
                     total_iterations, static_cast<double>(residual),
                     static_cast<double>(target));
      }
      if (!std::isfinite(static_cast<double>(residual)) ||
          residual <= target) {
        break;
      }
      if (corrections_enabled &&
          since_correction >= levels_[0].correction_interval) {
        const Real before = residual;
        const Real candidate = apply_coarse_correction(0);
        since_correction = 0;
        if (!std::isfinite(static_cast<double>(candidate)) ||
            candidate >= before) {
          residual = reject_correction(0);
          corrections_enabled = false;
        } else {
          residual = candidate;
        }
        if (params_[kVerbose] != 0) {
          std::fprintf(stdout,
                       "PYQCU::KUPL::CLOVER_MULTIGRID_DEBUG:\n "
                       "coarse-correction iterations=%d residual=%g "
                       "corrections=%d\n",
                       total_iterations, static_cast<double>(residual),
                       corrections_enabled ? 1 : 0);
        }
      }
    }
    if (params_[kVerbose] != 0) {
      std::fprintf(
          stdout,
          "PYQCU::KUPL::CLOVER_MULTIGRID_DEBUG:\n "
          "final iterations=%d residual=%g target=%g relative=%g\n",
          total_iterations, static_cast<double>(residual),
          static_cast<double>(target),
          static_cast<double>(residual / rhs_norm));
    }
    if (!std::isfinite(static_cast<double>(residual)) ||
        residual > target) {
      throw std::runtime_error(
          "legacy Clover-MG did not reach the full true residual");
    }
    full_to_canonical(static_cast<Complex<Real> *>(fermion_out),
                      levels_[0].x);
  }

  bool verify() {
    const Real tolerance =
        sizeof(Real) == sizeof(float) ? Real(2e-3) : Real(1e-8);
    for (int parent = 0; parent + 1 < num_levels_; ++parent) {
      LegacyMgLevel<Real> &fine = levels_[parent];
      LegacyMgLevel<Real> &coarse = levels_[parent + 1];
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
    if (level == 0) {
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
    LegacyMgLevel<Real> &p = levels_[parent];
    LegacyMgLevel<Real> &c = levels_[parent + 1];
    set_.multigrid_restrict(
        coarse.data(), fine.data(), p.null_vectors, c.E, p.E, p.dims[0],
        p.dims[1], p.dims[2], p.dims[3], c.dims[0], c.dims[1], c.dims[2],
        c.dims[3]);
  }

  void prolong(int parent, std::vector<Complex<Real>> &fine,
               const std::vector<Complex<Real>> &coarse) {
    LegacyMgLevel<Real> &p = levels_[parent];
    LegacyMgLevel<Real> &c = levels_[parent + 1];
    set_.multigrid_prolong(
        fine.data(), coarse.data(), p.null_vectors, c.E, p.E, p.dims[0],
        p.dims[1], p.dims[2], p.dims[3], c.dims[0], c.dims[1], c.dims[2],
        c.dims[3]);
  }

  bool bicgstab_chunk(int level, int iterations, bool force_convergence) {
    using C = Complex<Real>;
    LegacyMgLevel<Real> &state = levels_[level];
    last_chunk_iterations_ = 0;
    last_chunk_converged_ = false;
    const int64_t count = static_cast<int64_t>(state.x.size());
    const int64_t chunk = omp_chunk<Real>(count);
    const C zero(0);
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
    const Real target = state.target;
    for (int iteration = 0; iteration < iterations; ++iteration) {
      if (std::sqrt(mg_norm2(state.r, distributed_)) <= target) {
        last_chunk_iterations_ = iteration;
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
        last_chunk_iterations_ = iteration;
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
        last_chunk_iterations_ = iteration;
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
        last_chunk_iterations_ = iteration + 1;
        last_chunk_converged_ = true;
        return true;
      }
      apply_level(level, state.v, state.s);
      const C tt = mg_dot(state.v, state.v, distributed_);
      if (std::abs(tt) <= std::numeric_limits<Real>::min()) {
        if (params_[kVerbose] != 0) {
          std::fprintf(stderr,
                       "PYQCU::KUPL::CLOVER_MULTIGRID_DEBUG:\n "
                       "level=%d iteration=%d tt-breakdown\n",
                       level, iteration);
        }
        last_chunk_iterations_ = iteration;
        return false;
      }
      omega = mg_dot(state.v, state.s, distributed_) / tt;
      if (std::abs(omega) <= std::numeric_limits<Real>::min()) {
        if (params_[kVerbose] != 0) {
          std::fprintf(stderr,
                       "PYQCU::KUPL::CLOVER_MULTIGRID_DEBUG:\n "
                       "level=%d iteration=%d omega-breakdown\n",
                       level, iteration);
        }
        last_chunk_iterations_ = iteration;
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
      if (!bicgstab_chunk(level, state.max_iter, true)) {
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
      cycle(level + 1, child.rhs, state.correction);
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

  Real apply_coarse_correction(int level) {
    if (level + 1 >= num_levels_) {
      return Real(0);
    }
    LegacyMgLevel<Real> &parent = levels_[level];
    LegacyMgLevel<Real> &child = levels_[level + 1];
    for (const Complex<Real> &value : parent.r) {
      if (!std::isfinite(static_cast<double>(value.real())) ||
          !std::isfinite(static_cast<double>(value.imag()))) {
        throw std::runtime_error(
            "legacy Clover-MG parent residual is non-finite");
      }
    }
    restrict(level, child.rhs, parent.r);
    for (const Complex<Real> &value : child.rhs) {
      if (!std::isfinite(static_cast<double>(value.real())) ||
          !std::isfinite(static_cast<double>(value.imag()))) {
        throw std::runtime_error(
            "legacy Clover-MG restriction produced a non-finite coarse RHS");
      }
    }
    cycle(level + 1, child.rhs, parent.correction);
    for (std::size_t i = 0; i < parent.x.size(); ++i) {
      parent.x[i] += parent.correction[i];
    }
    apply_level(level, parent.v, parent.x);
    for (std::size_t i = 0; i < parent.r.size(); ++i) {
      parent.r[i] = parent.rhs[i] - parent.v[i];
    }
    // The caller used to immediately recompute this same fine-level true
    // residual with true_residual(); return it from the freshly formed r.
    return std::sqrt(mg_norm2(parent.r, distributed_));
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

  LatticeSetBase &set_;
  int last_chunk_iterations_ = 0;
  bool last_chunk_converged_ = false;
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
  std::array<LegacyMgLevel<Real>, 5> levels_{};
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
  for (int transition = 0; transition + 1 < params[kMgNumLevel];
       ++transition) {
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
