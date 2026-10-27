#include "kupl/lattice_set.h"
#include "kupl/lattice_halo.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace kupl::qcu {

template <typename Real>
LatticeSet<Real>::LatticeSet(const int32_t *params, const Real *argv)
    : geometry_(make_geometry(params)), threads_(configured_threads()) {
  std::copy(params, params + kParamsSize, parameters_.ints.begin());
  std::copy(argv, argv + kArgvSize, parameters_.reals.begin());
  const int data_type = parameters_.get(kDataType);
  if (data_type != kLatC64 && data_type != kLatC128) {
    throw std::runtime_error("KUPL supports only c64 and c128");
  }
  allocate_scratch();
  log("INIT",
      "rank=%d size=%d grid=%d,%d,%d,%d local=%d,%d,%d,%d "
      "threads=%d csw=%g",
      geometry_.rank, geometry_.size, geometry_.grid[0],
      geometry_.grid[1], geometry_.grid[2], geometry_.grid[3],
      geometry_.local[0], geometry_.local[1], geometry_.local[2],
      geometry_.local[3], threads_, static_cast<double>(csw()));
}

template <typename Real>
void LatticeSet<Real>::update_params(const int32_t *params) {
  std::copy(params, params + kParamsSize, parameters_.ints.begin());
  validate_parity();
}

template <typename Real>
void LatticeSet<Real>::wilson_dslash(const void *out, const void *in,
                                     const void *gauge) {
  require_wilson_mode();
  const int target = target_parity();
  ensure_gauge_halo(static_cast<const Complex<Real> *>(gauge));
  bare_dslash(static_cast<Complex<Real> *>(const_cast<void *>(out)),
              static_cast<const Complex<Real> *>(in), target);
}

template <typename Real>
void LatticeSet<Real>::wilson_bistabcg(void *out, const void *in,
                                       const void *gauge) {
  require_solver_mode();
  ensure_solver_scratch();
  ensure_gauge_halo(static_cast<const Complex<Real> *>(gauge));
  solve_wilson_bistabcg(static_cast<Complex<Real> *>(out),
                        static_cast<const Complex<Real> *>(in));
}

template <typename Real>
void LatticeSet<Real>::wilson_bistabcg_dslash(void *out, const void *in,
                                              const void *gauge) {
  require_solver_mode();
  ensure_solver_scratch();
  require_wilson_gauge(static_cast<const Complex<Real> *>(gauge));
  apply_wilson_schur(static_cast<const Complex<Real> *>(in),
                     static_cast<Complex<Real> *>(out), target_parity(),
                     work_a_.data(), work_b_.data());
}

template <typename Real>
void LatticeSet<Real>::wilson_cg(void *out, const void *in,
                                 const void *gauge) {
  require_solver_mode();
  ensure_solver_scratch();
  ensure_gauge_halo(static_cast<const Complex<Real> *>(gauge));
  solve_wilson_cg(static_cast<Complex<Real> *>(out),
                  static_cast<const Complex<Real> *>(in));
}

template <typename Real>
void LatticeSet<Real>::wilson_cg_dslash(void *out, const void *in,
                                        const void *gauge) {
  require_solver_mode();
  ensure_solver_scratch();
  require_wilson_gauge(static_cast<const Complex<Real> *>(gauge));
  apply_wilson_cg_dslash(static_cast<const Complex<Real> *>(in),
                         static_cast<Complex<Real> *>(out), target_parity());
}

template <typename Real>
void LatticeSet<Real>::clover(void *clover_out, const void *gauge) {
  require_wilson_mode();
  ensure_gauge_halo(static_cast<const Complex<Real> *>(gauge));
  compute_clover(static_cast<Complex<Real> *>(clover_out), target_parity());
}

template <typename Real>
void LatticeSet<Real>::clovers(void *clover_out, void *clover_inv,
                               const void *gauge) {
  require_wilson_mode();
  ensure_gauge_halo(static_cast<const Complex<Real> *>(gauge));
  const int parity = target_parity();
  auto *term = static_cast<Complex<Real> *>(clover_out);
  compute_clover(term, parity);
  invert_clover(term, static_cast<Complex<Real> *>(clover_inv));
}

template <typename Real>
void LatticeSet<Real>::clover_dslash(void *out, const void *in,
                                     const void *gauge) {
  require_wilson_mode();
  const int target = target_parity();
  ensure_gauge_halo(static_cast<const Complex<Real> *>(gauge));
  ensure_cached_clover(static_cast<const Complex<Real> *>(gauge));
  bare_dslash(static_cast<Complex<Real> *>(out),
              static_cast<const Complex<Real> *>(in), target);
  apply_clover_onsite(
      static_cast<const Complex<Real> *>(out),
      cached_clover_inv_[static_cast<std::size_t>(target)].data(),
      static_cast<Complex<Real> *>(out));
}

template <typename Real>
void LatticeSet<Real>::supplied_clover_dslash(void *out, const void *in,
                                              const void *gauge,
                                              const void *clover) {
  require_wilson_mode();
  const int target = target_parity();
  ensure_gauge_halo(static_cast<const Complex<Real> *>(gauge));
  bare_dslash(static_cast<Complex<Real> *>(out),
              static_cast<const Complex<Real> *>(in), target);
  apply_onsite(static_cast<const Complex<Real> *>(out),
               static_cast<const Complex<Real> *>(clover),
               static_cast<Complex<Real> *>(out));
}

template <typename Real>
void LatticeSet<Real>::clover_bistabcg(
    void *out, const void *in, const void *gauge, const void *clover_even,
    const void *clover_odd, const void *clover_even_inv,
    const void *clover_odd_inv) {
  require_solver_mode();
  ensure_solver_scratch();
  const int parity = target_parity();
  ensure_gauge_halo(static_cast<const Complex<Real> *>(gauge));
  const std::array<const Complex<Real> *, 2> clover = {
      static_cast<const Complex<Real> *>(clover_even),
      static_cast<const Complex<Real> *>(clover_odd)};
  const std::array<const Complex<Real> *, 2> inverse = {
      static_cast<const Complex<Real> *>(clover_even_inv),
      static_cast<const Complex<Real> *>(clover_odd_inv)};
  solve_clover(static_cast<Complex<Real> *>(out),
               static_cast<const Complex<Real> *>(in), parity, clover,
               inverse);
}

template <typename Real>
void LatticeSet<Real>::clover_bistabcg_dslash(
    void *out, const void *in, const void *gauge, const void *clover_even,
    const void *clover_odd, const void *clover_even_inv,
    const void *clover_odd_inv) {
  require_solver_mode();
  ensure_solver_scratch();
  const int parity = target_parity();
  ensure_gauge_halo(static_cast<const Complex<Real> *>(gauge));
  const std::array<const Complex<Real> *, 2> clover = {
      static_cast<const Complex<Real> *>(clover_even),
      static_cast<const Complex<Real> *>(clover_odd)};
  const std::array<const Complex<Real> *, 2> inverse = {
      static_cast<const Complex<Real> *>(clover_even_inv),
      static_cast<const Complex<Real> *>(clover_odd_inv)};
  apply_schur(static_cast<const Complex<Real> *>(in),
              static_cast<Complex<Real> *>(out), parity, clover, inverse,
              tmp_a_.data(), tmp_b_.data());
}

template <typename Real>
int LatticeSet<Real>::clover_bistabcg_prepare(
    void *compact_rhs, const void *full_rhs, const void *gauge,
    const void *clover_even, const void *clover_odd,
    const void *clover_even_inv, const void *clover_odd_inv) {
  require_solver_mode();
  ensure_solver_scratch();
  if (compact_rhs == nullptr || full_rhs == nullptr || gauge == nullptr ||
      clover_even == nullptr || clover_odd == nullptr ||
      clover_even_inv == nullptr || clover_odd_inv == nullptr) {
    throw std::runtime_error("Clover prepare received a null pointer");
  }
  ensure_gauge_halo(static_cast<const Complex<Real> *>(gauge));
  const std::array<const Complex<Real> *, 2> inverse = {
      static_cast<const Complex<Real> *>(clover_even_inv),
      static_cast<const Complex<Real> *>(clover_odd_inv)};
  prepare_clover_schur(static_cast<Complex<Real> *>(compact_rhs),
                       static_cast<const Complex<Real> *>(full_rhs),
                       target_parity(), inverse);
  return 0;
}

template <typename Real>
int LatticeSet<Real>::clover_bistabcg_reconstruct(
    void *full_out, const void *full_rhs, const void *target_solution,
    const void *gauge, const void *clover_even, const void *clover_odd,
    const void *clover_even_inv, const void *clover_odd_inv) {
  require_solver_mode();
  ensure_solver_scratch();
  if (full_out == nullptr || full_rhs == nullptr ||
      target_solution == nullptr || gauge == nullptr ||
      clover_even == nullptr || clover_odd == nullptr ||
      clover_even_inv == nullptr || clover_odd_inv == nullptr) {
    throw std::runtime_error("Clover reconstruct received a null pointer");
  }
  ensure_gauge_halo(static_cast<const Complex<Real> *>(gauge));
  const std::array<const Complex<Real> *, 2> inverse = {
      static_cast<const Complex<Real> *>(clover_even_inv),
      static_cast<const Complex<Real> *>(clover_odd_inv)};
  reconstruct_clover_schur(
      static_cast<Complex<Real> *>(full_out),
      static_cast<const Complex<Real> *>(full_rhs),
      static_cast<const Complex<Real> *>(target_solution), target_parity(),
      inverse);
  return 0;
}

template <typename Real>
void LatticeSet<Real>::multigrid_restrict(
    void *coarse_out, const void *fine_in, const void *null_vectors, int E,
    int e, int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc) {
  require_wilson_mode();
  apply_multigrid_restrict(
      static_cast<Complex<Real> *>(coarse_out),
      static_cast<const Complex<Real> *>(fine_in),
      static_cast<const Complex<Real> *>(null_vectors), E, e, Xf, Yf, Zf,
      Tf, Xc, Yc, Zc, Tc);
}

template <typename Real>
void LatticeSet<Real>::multigrid_prolong(
    void *fine_out, const void *coarse_in, const void *null_vectors, int E,
    int e, int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc) {
  require_wilson_mode();
  apply_multigrid_prolong(
      static_cast<Complex<Real> *>(fine_out),
      static_cast<const Complex<Real> *>(coarse_in),
      static_cast<const Complex<Real> *>(null_vectors), E, e, Xf, Yf, Zf,
      Tf, Xc, Yc, Zc, Tc);
}

template <typename Real>
void LatticeSet<Real>::multigrid_coarse_dslash(
    void *out, const void *in, const void *hopping, const void *sitting,
    int E, int X, int Y, int Z, int T) {
  require_wilson_mode();
  apply_multigrid_coarse_dslash(
      static_cast<Complex<Real> *>(out),
      static_cast<const Complex<Real> *>(in),
      static_cast<const Complex<Real> *>(hopping),
      static_cast<const Complex<Real> *>(sitting), E, X, Y, Z, T);
}

template <typename Real>
void LatticeSet<Real>::multigrid_coarse_dslash_wide(
    void *out, const void *in, const void *sitting, const void *hop_nn,
    const void *hop_diag, int E, int X, int Y, int Z, int T) {
  require_wilson_mode();
  apply_multigrid_coarse_dslash_wide(
      static_cast<Complex<Real> *>(out),
      static_cast<const Complex<Real> *>(in),
      static_cast<const Complex<Real> *>(sitting),
      static_cast<const Complex<Real> *>(hop_nn),
      static_cast<const Complex<Real> *>(hop_diag), E, X, Y, Z, T);
}

template <typename Real>
void LatticeSet<Real>::laplacian(void *out, const void *in,
                                 const void *gauge) {
  require_laplacian_mode();
  apply_laplacian(static_cast<Complex<Real> *>(out),
                  static_cast<const Complex<Real> *>(in),
                  static_cast<const Complex<Real> *>(gauge));
}

template <typename Real> void LatticeSet<Real>::allocate_scratch() {
  const auto advise = [](auto &field) {
    advise_large_pages(field.data(), field.size() * sizeof(field[0]));
  };
  if (geometry_.laplacian) {
    for (int dim = 0; dim < 3; ++dim) {
      const std::size_t source_face =
          static_cast<std::size_t>(geometry_.face_sites[dim] * kColor);
      const std::size_t gauge_face =
          static_cast<std::size_t>(geometry_.face_sites[dim] * kGauge);
      for (int sign : {-1, 1}) {
        const int offset = face_offset(dim, sign);
        laplacian_source_send_[offset].resize(source_face);
        laplacian_source_recv_[offset].resize(source_face);
        laplacian_gauge_send_[offset].resize(gauge_face);
        laplacian_gauge_recv_[offset].resize(gauge_face);
      }
    }
    for (int dim = 0; dim < 3; ++dim) {
      for (int sign : {-1, 1}) {
        const int offset = face_offset(dim, sign);
        advise(laplacian_source_send_[offset]);
        advise(laplacian_source_recv_[offset]);
        advise(laplacian_gauge_send_[offset]);
        advise(laplacian_gauge_recv_[offset]);
      }
    }
    return;
  }
  for (int dim = 0; dim < kDim; ++dim) {
    if (geometry_.grid[dim] == 1) {
      continue;
    }
    const std::size_t face =
        static_cast<std::size_t>(geometry_.face_sites[dim] * kVector);
    fermion_send_[face_offset(dim, -1)].resize(face);
    fermion_send_[face_offset(dim, 1)].resize(face);
    fermion_recv_[face_offset(dim, -1)].resize(face);
    fermion_recv_[face_offset(dim, 1)].resize(face);
    const std::size_t gauge_face = static_cast<std::size_t>(
        geometry_.face_sites[dim] * 2 * kGauge * kDim);
    gauge_halo_.faces[face_offset(dim, -1)].resize(gauge_face);
    gauge_halo_.faces[face_offset(dim, 1)].resize(gauge_face);
    gauge_halo_.recv_faces[face_offset(dim, -1)].resize(gauge_face);
    gauge_halo_.recv_faces[face_offset(dim, 1)].resize(gauge_face);
  }
  for (int dim0 = 0; dim0 < kDim; ++dim0) {
    for (int dim1 = dim0 + 1; dim1 < kDim; ++dim1) {
      if (geometry_.grid[dim0] == 1 || geometry_.grid[dim1] == 1) {
        continue;
      }
      for (int sign0 : {-1, 1}) {
        for (int sign1 : {-1, 1}) {
          auto &buffer =
              gauge_halo_
                  .corners[static_cast<std::size_t>(
                      plane_index(dim0, dim1))]
                  [static_cast<std::size_t>(sign_index(sign0) * 2 +
                                            sign_index(sign1))];
          auto &recv_buffer =
              gauge_halo_
                  .recv_corners[static_cast<std::size_t>(
                      plane_index(dim0, dim1))]
                  [static_cast<std::size_t>(sign_index(sign0) * 2 +
                                            sign_index(sign1))];
          const std::size_t size = static_cast<std::size_t>(
              geometry_.corner_sites[dim0][dim1] * 2 * kGauge * kDim);
          buffer.resize(size);
          recv_buffer.resize(size);
        }
      }
    }
  }
  for (int dim = 0; dim < kDim; ++dim) {
    if (geometry_.grid[dim] == 1) {
      continue;
    }
    for (int sign : {-1, 1}) {
      const int offset = face_offset(dim, sign);
      advise(fermion_send_[offset]);
      advise(fermion_recv_[offset]);
      advise(gauge_halo_.faces[offset]);
      advise(gauge_halo_.recv_faces[offset]);
    }
  }
  for (int dim0 = 0; dim0 < kDim; ++dim0) {
    for (int dim1 = dim0 + 1; dim1 < kDim; ++dim1) {
      if (geometry_.grid[dim0] == 1 || geometry_.grid[dim1] == 1) {
        continue;
      }
      for (int sign0 : {-1, 1}) {
        for (int sign1 : {-1, 1}) {
          const std::size_t plane =
              static_cast<std::size_t>(plane_index(dim0, dim1));
          const std::size_t corner = static_cast<std::size_t>(
              sign_index(sign0) * 2 + sign_index(sign1));
          advise(gauge_halo_.corners[plane][corner]);
          advise(gauge_halo_.recv_corners[plane][corner]);
        }
      }
    }
  }
  boundary_sites_.clear();
  boundary_sites_.reserve(static_cast<std::size_t>(geometry_.half_sites / 4));
  for (int64_t linear = 0; linear < geometry_.half_sites; ++linear) {
    int x = 0;
    int y = 0;
    int z = 0;
    int t2 = 0;
    geometry_.decode_half(linear, x, y, z, t2);
    if (x == 0 || y == 0 || z == 0 || t2 == 0 ||
        x == geometry_.local[0] - 1 || y == geometry_.local[1] - 1 ||
        z == geometry_.local[2] - 1 ||
        t2 == geometry_.local[3] / 2 - 1) {
      boundary_sites_.push_back(linear);
    }
  }
}

template <typename Real> void LatticeSet<Real>::ensure_solver_scratch() {
  const std::size_t vectors =
      static_cast<std::size_t>(geometry_.parity_vectors * 2);
  for (auto *field :
       {&r_, &rhat_, &p_, &v_, &s_, &t_, &bp_, &tmp_a_, &tmp_b_}) {
    if (!field->empty()) {
      continue;
    }
    field->resize(vectors);
    advise_large_pages(field->data(), field->size() * sizeof(Complex<Real>));
  }
  for (auto *field : {&work_a_, &work_b_, &work_c_}) {
    if (!field->empty()) {
      continue;
    }
    field->resize(static_cast<std::size_t>(geometry_.parity_vectors));
    advise_large_pages(field->data(), field->size() * sizeof(Complex<Real>));
  }
}

template <typename Real> Real LatticeSet<Real>::mass() const {
  return parameters_.reals[kMass];
}

template <typename Real> Real LatticeSet<Real>::kappa() const {
  return Real(1) / (Real(2) * mass() + Real(8));
}

template <typename Real> Real LatticeSet<Real>::csw() const {
  return parameters_.reals[kCsw];
}

template <typename Real> Real LatticeSet<Real>::atol() const {
  return parameters_.reals[kAtol];
}

template <typename Real> Real LatticeSet<Real>::residual_floor() const {
  const Real multiplier =
      sizeof(Real) == sizeof(float) ? Real(2048) : Real(128);
  return std::numeric_limits<Real>::epsilon() * multiplier;
}

template <typename Real>
void LatticeSet<Real>::require_wilson_mode() const {
  if (geometry_.laplacian) {
    throw std::runtime_error(
        "Wilson/Clover operation cannot run in Laplacian mode");
  }
}

template <typename Real>
void LatticeSet<Real>::require_gauge_mode() const {
  if (geometry_.laplacian || parameters_.get(kSetPlan) != kSetPlanGauge) {
    throw std::runtime_error(
        "Gaussian gauge requires params[_SET_PLAN_]=-1");
  }
}

template <typename Real>
void LatticeSet<Real>::require_solver_mode() const {
  require_wilson_mode();
  if (parameters_.get(kSetPlan) != kSetPlanSolver) {
    throw std::runtime_error(
        "Wilson/Clover solver operation requires params[_SET_PLAN_]=1");
  }
}

template <typename Real>
void LatticeSet<Real>::require_laplacian_mode() const {
  if (!geometry_.laplacian) {
    throw std::runtime_error(
        "Laplacian operation requires applyInitQcu with plan -2");
  }
  if (parameters_.get(kSetPlan) != kSetPlanLaplacian) {
    throw std::runtime_error(
        "Laplacian operation requires params[_SET_PLAN_]=-2");
  }
}

template <typename Real>
void LatticeSet<Real>::require_wilson_gauge(const Complex<Real> *gauge) {
  if (gauge == nullptr) {
    throw std::runtime_error("null gauge pointer");
  }
  ensure_gauge_halo(gauge);
}

template <typename Real> void LatticeSet<Real>::validate_parity() const {
  const int parity = parameters_.get(kParity);
  if (parity != 0 && parity != 1) {
    throw std::runtime_error("params[_PARITY_] must be 0 or 1");
  }
}

template <typename Real> int LatticeSet<Real>::target_parity() const {
  validate_parity();
  return parameters_.get(kParity);
}

template class LatticeSet<float>;
template class LatticeSet<double>;

} // namespace kupl::qcu
