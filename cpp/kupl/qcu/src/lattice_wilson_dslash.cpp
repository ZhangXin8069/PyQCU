#include "kupl/lattice_halo.h"
#include "kupl/lattice_linalg.h"
#include "kupl/lattice_set.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

namespace kupl::qcu {
namespace {

inline bool use_block_dslash(int64_t half_sites) {
  static const bool allowed =
      std::getenv("KUPL_DISABLE_BLOCK_DSLASH") == nullptr;
  static const int64_t minimum_half_sites = [] {
    const char *text = std::getenv("KUPL_BLOCK_DSLASH_MIN_HALF_SITES");
    if (text == nullptr || *text == '\0') {
      return int64_t{65536};
    }
    char *end = nullptr;
    const long long value = std::strtoll(text, &end, 10);
    return end != text && *end == '\0' && value >= 0
               ? static_cast<int64_t>(value)
               : int64_t{65536};
  }();
  return allowed && half_sites >= minimum_half_sites;
}

inline bool use_t2_zblock_dslash() {
  static const bool enabled = [] {
    const char *text = std::getenv("KUPL_ENABLE_T2_ZBLOCK_DSLASH");
    return text != nullptr && *text != '\0' && std::strcmp(text, "0") != 0 &&
           std::strcmp(text, "false") != 0 &&
           std::strcmp(text, "no") != 0 &&
           std::strcmp(text, "off") != 0;
  }();
  return enabled;
}

inline bool use_ht2_tboundary_split() {
  static const bool enabled = [] {
    const char *text = std::getenv("KUPL_ENABLE_HT2_TBOUNDARY_SPLIT");
    return text != nullptr && *text != '\0' && std::strcmp(text, "0") != 0 &&
           std::strcmp(text, "false") != 0 &&
           std::strcmp(text, "no") != 0 &&
           std::strcmp(text, "off") != 0;
  }();
  return enabled;
}

// Opt-in cross-site lane kernel for the boundary sites. Disabled by default so
// the scalar boundary path stays byte-for-byte unchanged.
inline bool use_boundary_lanes() {
  static const bool enabled = [] {
    const char *text = std::getenv("KUPL_ENABLE_BOUNDARY_LANES");
    return text != nullptr && *text != '\0' && std::strcmp(text, "0") != 0 &&
           std::strcmp(text, "false") != 0 &&
           std::strcmp(text, "no") != 0 &&
           std::strcmp(text, "off") != 0;
  }();
  return enabled;
}

template <typename Real, int Direction, int GammaSign>
inline void transform_gamma(const Complex<Real> &v0,
                            const Complex<Real> &v1,
                            const Complex<Real> &v2,
                            const Complex<Real> &v3,
                            Complex<Real> &h0, Complex<Real> &h1,
                            Complex<Real> &h2, Complex<Real> &h3) {
  using C = Complex<Real>;
  if constexpr (Direction == 0) {
    h0 = C(v0.real() + GammaSign * v3.imag(),
           v0.imag() - GammaSign * v3.real());
    h1 = C(v1.real() + GammaSign * v2.imag(),
           v1.imag() - GammaSign * v2.real());
    h2 = C(v2.real() - GammaSign * v1.imag(),
           v2.imag() + GammaSign * v1.real());
    h3 = C(v3.real() - GammaSign * v0.imag(),
           v3.imag() + GammaSign * v0.real());
  } else if constexpr (Direction == 1) {
    h0 = C(v0.real() + GammaSign * v3.real(),
           v0.imag() + GammaSign * v3.imag());
    h1 = C(v1.real() - GammaSign * v2.real(),
           v1.imag() - GammaSign * v2.imag());
    h2 = C(v2.real() - GammaSign * v1.real(),
           v2.imag() - GammaSign * v1.imag());
    h3 = C(v3.real() + GammaSign * v0.real(),
           v3.imag() + GammaSign * v0.imag());
  } else if constexpr (Direction == 2) {
    h0 = C(v0.real() + GammaSign * v2.imag(),
           v0.imag() - GammaSign * v2.real());
    h1 = C(v1.real() - GammaSign * v3.imag(),
           v1.imag() + GammaSign * v3.real());
    h2 = C(v2.real() - GammaSign * v0.imag(),
           v2.imag() + GammaSign * v0.real());
    h3 = C(v3.real() + GammaSign * v1.imag(),
           v3.imag() - GammaSign * v1.real());
  } else {
    h0 = C(v0.real() - GammaSign * v2.real(),
           v0.imag() - GammaSign * v2.imag());
    h1 = C(v1.real() - GammaSign * v3.real(),
           v1.imag() - GammaSign * v3.imag());
    h2 = C(v2.real() - GammaSign * v0.real(),
           v2.imag() - GammaSign * v0.imag());
    h3 = C(v3.real() - GammaSign * v1.real(),
           v3.imag() - GammaSign * v1.imag());
  }
}

// The halo pack needs to know whether the current operator is the daggered one,
// because the receiver's gamma sign (and hence the half-spinor projection the
// sender must transmit) flips for the daggered hop.  post_fermion_halo has no
// distinct signature for the two operators, so bare_dslash/bare_dslash_dagger
// set this per-thread flag before posting.
thread_local bool g_halo_pack_dagger = false;

// Forward projection used by KUPL_HALO_PACK6.  Only the two independent spin
// components of transform_gamma's rank-2 output are kept; the receiver feeds
// them to the same transform_gamma with spins 2/3 zero, which reproduces all
// four output spins exactly (h3/h2 are the known multiples of h0/h1).  `half`
// is laid out as plane = spin * kColor + color, matching the plane-major halo
// buffer.
template <typename Real, int Direction, int GammaSign>
inline void project_half_spinor(const Complex<Real> source[kVector],
                                Complex<Real> half[2 * kColor]) {
  using C = Complex<Real>;
  for (int color = 0; color < kColor; ++color) {
    C h0;
    C h1;
    C h2;
    C h3;
    transform_gamma<Real, Direction, GammaSign>(
        source[0 * kColor + color], source[1 * kColor + color],
        source[2 * kColor + color], source[3 * kColor + color], h0, h1, h2,
        h3);
    half[0 * kColor + color] = h0;
    half[1 * kColor + color] = h1;
  }
}

template <typename Real>
inline void project_half_spinor_dispatch(const Complex<Real> source[kVector],
                                         Complex<Real> half[2 * kColor],
                                         int dim, int gamma_sign) {
  if (dim == 0) {
    if (gamma_sign > 0) {
      project_half_spinor<Real, 0, 1>(source, half);
    } else {
      project_half_spinor<Real, 0, -1>(source, half);
    }
  } else if (dim == 1) {
    if (gamma_sign > 0) {
      project_half_spinor<Real, 1, 1>(source, half);
    } else {
      project_half_spinor<Real, 1, -1>(source, half);
    }
  } else if (dim == 2) {
    if (gamma_sign > 0) {
      project_half_spinor<Real, 2, 1>(source, half);
    } else {
      project_half_spinor<Real, 2, -1>(source, half);
    }
  } else {
    if (gamma_sign > 0) {
      project_half_spinor<Real, 3, 1>(source, half);
    } else {
      project_half_spinor<Real, 3, -1>(source, half);
    }
  }
}

template <typename Real, int Direction, bool Dagger, int GammaSign>
void accumulate_hopping(const VectorView<Real> &source,
                        const GaugeView<Real> &gauge,
                        Complex<Real> result[kVector]) {
  using C = Complex<Real>;
  C transported[kVector];
#define KUPL_ACC(S, R, CO)                                                  \
  {                                                                         \
    const C &e = gauge.data[static_cast<int64_t>(                           \
        Dagger ? (CO) * kColor + (R) : (R) * kColor + (CO)) *               \
        gauge.matrix_stride];                                               \
    const C &v = source[(S) * kColor + (CO)];                               \
    const Real ei = Dagger ? -e.imag() : e.imag();                          \
    r##S##R += e.real() * v.real() - ei * v.imag();                         \
    i##S##R += e.real() * v.imag() + ei * v.real();                         \
  }
  Real r00 = 0, r01 = 0, r02 = 0, r10 = 0, r11 = 0, r12 = 0;
  Real r20 = 0, r21 = 0, r22 = 0, r30 = 0, r31 = 0, r32 = 0;
  Real i00 = 0, i01 = 0, i02 = 0, i10 = 0, i11 = 0, i12 = 0;
  Real i20 = 0, i21 = 0, i22 = 0, i30 = 0, i31 = 0, i32 = 0;
  KUPL_ACC(0, 0, 0) KUPL_ACC(0, 0, 1) KUPL_ACC(0, 0, 2)
  KUPL_ACC(0, 1, 0) KUPL_ACC(0, 1, 1) KUPL_ACC(0, 1, 2)
  KUPL_ACC(0, 2, 0) KUPL_ACC(0, 2, 1) KUPL_ACC(0, 2, 2)
  KUPL_ACC(1, 0, 0) KUPL_ACC(1, 0, 1) KUPL_ACC(1, 0, 2)
  KUPL_ACC(1, 1, 0) KUPL_ACC(1, 1, 1) KUPL_ACC(1, 1, 2)
  KUPL_ACC(1, 2, 0) KUPL_ACC(1, 2, 1) KUPL_ACC(1, 2, 2)
  KUPL_ACC(2, 0, 0) KUPL_ACC(2, 0, 1) KUPL_ACC(2, 0, 2)
  KUPL_ACC(2, 1, 0) KUPL_ACC(2, 1, 1) KUPL_ACC(2, 1, 2)
  KUPL_ACC(2, 2, 0) KUPL_ACC(2, 2, 1) KUPL_ACC(2, 2, 2)
  KUPL_ACC(3, 0, 0) KUPL_ACC(3, 0, 1) KUPL_ACC(3, 0, 2)
  KUPL_ACC(3, 1, 0) KUPL_ACC(3, 1, 1) KUPL_ACC(3, 1, 2)
  KUPL_ACC(3, 2, 0) KUPL_ACC(3, 2, 1) KUPL_ACC(3, 2, 2)
#undef KUPL_ACC
  transported[0] = C(r00, i00);
  transported[1] = C(r01, i01);
  transported[2] = C(r02, i02);
  transported[3] = C(r10, i10);
  transported[4] = C(r11, i11);
  transported[5] = C(r12, i12);
  transported[6] = C(r20, i20);
  transported[7] = C(r21, i21);
  transported[8] = C(r22, i22);
  transported[9] = C(r30, i30);
  transported[10] = C(r31, i31);
  transported[11] = C(r32, i32);

  for (int color = 0; color < kColor; ++color) {
    const C v0 = transported[color];
    const C v1 = transported[kColor + color];
    const C v2 = transported[2 * kColor + color];
    const C v3 = transported[3 * kColor + color];
    C h0;
    C h1;
    C h2;
    C h3;
    transform_gamma<Real, Direction, GammaSign>(
        v0, v1, v2, v3, h0, h1, h2, h3);
    result[color] += h0;
    result[kColor + color] += h1;
    result[2 * kColor + color] += h2;
    result[3 * kColor + color] += h3;
  }
}

template <typename Real, int Direction, bool Dagger, int GammaSign>
void accumulate_hopping_block(const Complex<Real> *source,
                              const Complex<Real> *gauge,
                              int64_t source_stride,
                              int64_t gauge_stride, int lanes,
                              Complex<Real> result[kVector][64 / sizeof(Real)]) {
  using C = Complex<Real>;
  constexpr int kSiteLanes = 64 / sizeof(Real);
  C transported[kVector][kSiteLanes];
  for (int spin = 0; spin < kSpin; ++spin) {
    for (int row = 0; row < kColor; ++row) {
      Real real[kSiteLanes] = {};
      Real imag[kSiteLanes] = {};
      for (int column = 0; column < kColor; ++column) {
        const int gauge_component =
            Dagger ? column * kColor + row : row * kColor + column;
        const C *entries = gauge + gauge_component * gauge_stride;
        const C *vectors =
            source + static_cast<int64_t>(spin * kColor + column) *
                         source_stride;
#pragma omp simd
        for (int lane = 0; lane < lanes; ++lane) {
          const Real entry_imag = Dagger ? -entries[lane].imag()
                                         : entries[lane].imag();
          real[lane] += entries[lane].real() * vectors[lane].real() -
                        entry_imag * vectors[lane].imag();
          imag[lane] += entries[lane].real() * vectors[lane].imag() +
                        entry_imag * vectors[lane].real();
        }
      }
      for (int lane = 0; lane < lanes; ++lane) {
        transported[spin * kColor + row][lane] =
            C(real[lane], imag[lane]);
      }
    }
  }

  for (int color = 0; color < kColor; ++color) {
    C h0[kSiteLanes];
    C h1[kSiteLanes];
    C h2[kSiteLanes];
    C h3[kSiteLanes];
#pragma omp simd
    for (int lane = 0; lane < lanes; ++lane) {
      transform_gamma<Real, Direction, GammaSign>(
          transported[color][lane], transported[kColor + color][lane],
          transported[2 * kColor + color][lane],
          transported[3 * kColor + color][lane], h0[lane], h1[lane],
          h2[lane], h3[lane]);
      result[color][lane] += h0[lane];
      result[kColor + color][lane] += h1[lane];
      result[2 * kColor + color][lane] += h2[lane];
      result[3 * kColor + color][lane] += h3[lane];
    }
  }
}

template <typename Real, int Direction>
void accumulate_direction_block(
    const Complex<Real> *forward_source, const Complex<Real> *forward_gauge,
    const Complex<Real> *backward_source, const Complex<Real> *backward_gauge,
    int64_t source_stride, int64_t gauge_stride, int lanes,
    Complex<Real> result[kVector][64 / sizeof(Real)]) {
  accumulate_hopping_block<Real, Direction, false, 1>(
      forward_source, forward_gauge, source_stride, gauge_stride, lanes,
      result);
  accumulate_hopping_block<Real, Direction, true, -1>(
      backward_source, backward_gauge, source_stride, gauge_stride, lanes,
      result);
}

template <typename Real, int Direction>
void accumulate_direction(const VectorView<Real> &forward_source,
                          const GaugeView<Real> &forward_gauge,
                          const VectorView<Real> &backward_source,
                          const GaugeView<Real> &backward_gauge,
                          Complex<Real> result[kVector]) {
  accumulate_hopping<Real, Direction, false, 1>(forward_source,
                                                forward_gauge, result);
  accumulate_hopping<Real, Direction, true, -1>(backward_source,
                                                backward_gauge, result);
}

template <typename Real, int Direction, bool Adjoint>
void accumulate_site_block_direction(
    const Complex<Real> *local_source, const Complex<Real> *gauge,
    int64_t site, int64_t forward_offset, int64_t backward_offset,
    int target, int source_parity, int64_t half_sites, int lanes,
    Complex<Real> result[kVector][64 / sizeof(Real)]) {
  const int64_t source_stride = half_sites;
  const int64_t gauge_stride = static_cast<int64_t>(kDim) * half_sites;
  const Complex<Real> *forward_source = local_source + site + forward_offset;
  const Complex<Real> *backward_source =
      local_source + site - backward_offset;
  const Complex<Real> *forward_gauge =
      gauge + (static_cast<int64_t>(target) * kColor * kColor * kDim +
               Direction) * half_sites + site;
  const Complex<Real> *backward_gauge =
      gauge + (static_cast<int64_t>(source_parity) * kColor * kColor * kDim +
               Direction) * half_sites + site - backward_offset;
  if constexpr (Adjoint) {
    accumulate_hopping_block<Real, Direction, true, 1>(
        backward_source, backward_gauge, source_stride, gauge_stride, lanes,
        result);
    accumulate_hopping_block<Real, Direction, false, -1>(
        forward_source, forward_gauge, source_stride, gauge_stride, lanes,
        result);
  } else {
    accumulate_hopping_block<Real, Direction, false, 1>(
        forward_source, forward_gauge, source_stride, gauge_stride, lanes,
        result);
    accumulate_hopping_block<Real, Direction, true, -1>(
        backward_source, backward_gauge, source_stride, gauge_stride, lanes,
        result);
  }
}

template <typename Real, int Direction, bool Adjoint>
void accumulate_explicit_direction_block(
    const Complex<Real> *local_source, const Complex<Real> *gauge,
    int64_t site, int64_t forward_site, int64_t backward_site, int target,
    int source_parity, int64_t half_sites, int lanes,
    Complex<Real> result[kVector][64 / sizeof(Real)]) {
  const int64_t source_stride = half_sites;
  const int64_t gauge_stride = static_cast<int64_t>(kDim) * half_sites;
  const Complex<Real> *forward_source = local_source + forward_site;
  const Complex<Real> *backward_source = local_source + backward_site;
  const Complex<Real> *forward_gauge =
      gauge + (static_cast<int64_t>(target) * kColor * kColor * kDim +
               Direction) * half_sites + site;
  const Complex<Real> *backward_gauge =
      gauge + (static_cast<int64_t>(source_parity) * kColor * kColor * kDim +
               Direction) * half_sites + backward_site;
  if constexpr (Adjoint) {
    accumulate_hopping_block<Real, Direction, true, 1>(
        backward_source, backward_gauge, source_stride, gauge_stride, lanes,
        result);
    accumulate_hopping_block<Real, Direction, false, -1>(
        forward_source, forward_gauge, source_stride, gauge_stride, lanes,
        result);
  } else {
    accumulate_hopping_block<Real, Direction, false, 1>(
        forward_source, forward_gauge, source_stride, gauge_stride, lanes,
        result);
    accumulate_hopping_block<Real, Direction, true, -1>(
        backward_source, backward_gauge, source_stride, gauge_stride, lanes,
        result);
  }
}

template <typename Real, bool Adjoint>
void compute_interior_blocks(const Complex<Real> *local_source,
                             Complex<Real> *output,
                             const Complex<Real> *gauge,
                             const std::array<int, kDim> &local, int target) {
  using C = Complex<Real>;
  constexpr int kSiteLanes = 64 / sizeof(Real);
  const int x_dim = local[0];
  const int y_dim = local[1];
  const int z_dim = local[2];
  const int ht = local[3] / 2;
  if (x_dim < 3 || y_dim < 3 || z_dim < 3 || ht < 3) {
    return;
  }
  const int64_t half_sites = static_cast<int64_t>(x_dim) * y_dim * z_dim * ht;
  const int64_t x_stride = static_cast<int64_t>(y_dim) * z_dim * ht;
  const int64_t y_stride = static_cast<int64_t>(z_dim) * ht;
  const int source_parity = 1 - target;

#pragma omp parallel for collapse(3) schedule(static)
  for (int x = 1; x < x_dim - 1; ++x) {
    for (int y = 1; y < y_dim - 1; ++y) {
      for (int z = 1; z < z_dim - 1; ++z) {
        for (int t2 = 1; t2 < ht - 1; t2 += kSiteLanes) {
          const int lanes = std::min(kSiteLanes, ht - 1 - t2);
          const int64_t site = ((static_cast<int64_t>(x) * y_dim + y) *
                                    z_dim + z) * ht + t2;
          C result[kVector][kSiteLanes];
          for (int component = 0; component < kVector; ++component) {
#pragma omp simd
            for (int lane = 0; lane < kSiteLanes; ++lane) {
              result[component][lane] = C(0);
            }
          }
          accumulate_site_block_direction<Real, 0, Adjoint>(
              local_source, gauge, site, x_stride, x_stride, target,
              source_parity, half_sites, lanes, result);
          accumulate_site_block_direction<Real, 1, Adjoint>(
              local_source, gauge, site, y_stride, y_stride, target,
              source_parity, half_sites, lanes, result);
          accumulate_site_block_direction<Real, 2, Adjoint>(
              local_source, gauge, site, ht, ht, target, source_parity,
              half_sites, lanes, result);
          accumulate_site_block_direction<Real, 3, Adjoint>(
              local_source, gauge, site,
              ((x + y + z) & 1) ^ target,
              1 - (((x + y + z) & 1) ^ target), target, source_parity,
              half_sites, lanes, result);
          for (int component = 0; component < kVector; ++component) {
            C *destination = output + static_cast<int64_t>(component) *
                                         half_sites + site;
#pragma omp simd
            for (int lane = 0; lane < lanes; ++lane) {
              destination[lane] = result[component][lane];
            }
          }
        }
      }
    }
  }
}

template <typename Real, bool Adjoint>
void compute_periodic_face_blocks(
    const Complex<Real> *local_source, Complex<Real> *output,
    const Complex<Real> *gauge, const std::array<int, kDim> &local,
    const std::array<int, kDim> &grid, int target) {
  using C = Complex<Real>;
  constexpr int kSiteLanes = 64 / sizeof(Real);
  const int x_dim = local[0];
  const int y_dim = local[1];
  const int z_dim = local[2];
  const int ht = local[3] / 2;
  if (x_dim < 3 || y_dim < 3 || z_dim < 3 || ht < 3) {
    return;
  }
  const int64_t half_sites = static_cast<int64_t>(x_dim) * y_dim * z_dim * ht;
  const int source_parity = 1 - target;
  const int64_t x_face_sites = static_cast<int64_t>(y_dim - 2) * (z_dim - 2);
  const int64_t y_face_sites = static_cast<int64_t>(x_dim - 2) * (z_dim - 2);
  const int64_t z_face_sites = static_cast<int64_t>(x_dim - 2) * (y_dim - 2);
  const int64_t face_tasks = 2 * (x_face_sites + y_face_sites + z_face_sites);

  auto decode_task = [&](int64_t task, int64_t &face, int &x, int &y,
                         int &z) {
    int64_t offset = task;
    int64_t face_sites = 2 * x_face_sites;
    face = 0;
    if (offset >= face_sites) {
      offset -= face_sites;
      face = 1;
      face_sites = 2 * y_face_sites;
    }
    if (offset >= face_sites) {
      offset -= face_sites;
      face = 2;
      face_sites = 2 * z_face_sites;
    }
    face_sites /= 2;
    const int side = offset >= face_sites ? 1 : 0;
    offset %= face_sites;
    if (face == 0) {
      x = side ? x_dim - 1 : 0;
      y = 1 + static_cast<int>(offset / (z_dim - 2));
      z = 1 + static_cast<int>(offset % (z_dim - 2));
    } else if (face == 1) {
      y = side ? y_dim - 1 : 0;
      x = 1 + static_cast<int>(offset / (z_dim - 2));
      z = 1 + static_cast<int>(offset % (z_dim - 2));
    } else {
      z = side ? z_dim - 1 : 0;
      x = 1 + static_cast<int>(offset / (y_dim - 2));
      y = 1 + static_cast<int>(offset % (y_dim - 2));
    }
  };

#pragma omp parallel for schedule(static)
  for (int64_t task = 0; task < face_tasks; ++task) {
    int64_t face = 0;
    int x = 0;
    int y = 0;
    int z = 0;
    decode_task(task, face, x, y, z);
    if (grid[face] != 1) {
      continue;
    }
    const int extent = local[face];
    const int coordinate[kDim] = {x, y, z, 0};
    int forward_coordinate[kDim] = {x, y, z, 0};
    int backward_coordinate[kDim] = {x, y, z, 0};
    forward_coordinate[face] =
        (coordinate[face] + 1 + extent) % extent;
    backward_coordinate[face] =
        (coordinate[face] - 1 + extent) % extent;
    const int64_t spatial_site =
        (static_cast<int64_t>(x) * y_dim + y) * z_dim + z;
    const int64_t forward_spatial =
        (static_cast<int64_t>(forward_coordinate[0]) * y_dim +
         forward_coordinate[1]) * z_dim + forward_coordinate[2];
    const int64_t backward_spatial =
        (static_cast<int64_t>(backward_coordinate[0]) * y_dim +
         backward_coordinate[1]) * z_dim + backward_coordinate[2];

    for (int t2 = 1; t2 < ht - 1; t2 += kSiteLanes) {
      const int lanes = std::min(kSiteLanes, ht - 1 - t2);
      const int64_t site = spatial_site * ht + t2;
      const int64_t forward_site = forward_spatial * ht + t2;
      const int64_t backward_site = backward_spatial * ht + t2;
      C result[kVector][kSiteLanes];
      for (int component = 0; component < kVector; ++component) {
#pragma omp simd
        for (int lane = 0; lane < kSiteLanes; ++lane) {
          result[component][lane] = C(0);
        }
      }
      const int64_t offsets[kDim] = {
          static_cast<int64_t>(y_dim) * z_dim * ht,
          static_cast<int64_t>(z_dim) * ht, ht, 1};
      const int64_t forward_time = ((x + y + z) & 1) ^ target;
      const int64_t backward_time = 1 - forward_time;
      auto accumulate = [&](auto direction_constant) {
        constexpr int Direction = decltype(direction_constant)::value;
        if (Direction == face) {
          accumulate_explicit_direction_block<Real, Direction, Adjoint>(
              local_source, gauge, site, forward_site, backward_site, target,
              source_parity, half_sites, lanes, result);
        } else {
          const int64_t forward_offset =
              Direction == 3 ? forward_time : offsets[Direction];
          const int64_t backward_offset =
              Direction == 3 ? backward_time : offsets[Direction];
          accumulate_site_block_direction<Real, Direction, Adjoint>(
              local_source, gauge, site, forward_offset, backward_offset,
              target, source_parity, half_sites, lanes, result);
        }
      };
      accumulate(std::integral_constant<int, 0>{});
      accumulate(std::integral_constant<int, 1>{});
      accumulate(std::integral_constant<int, 2>{});
      accumulate(std::integral_constant<int, 3>{});
      for (int component = 0; component < kVector; ++component) {
        C *destination = output + static_cast<int64_t>(component) *
                                     half_sites + site;
#pragma omp simd
        for (int lane = 0; lane < lanes; ++lane) {
          destination[lane] = result[component][lane];
        }
      }
    }
  }
}

inline bool is_periodic_face_site(const std::array<int, kDim> &local,
                                  const std::array<int, kDim> &grid, int x,
                                  int y, int z, int t2) {
  const bool yz_interior = y > 0 && y < local[1] - 1 && z > 0 &&
                           z < local[2] - 1 && t2 > 0 &&
                           t2 < local[3] / 2 - 1;
  const bool xz_interior = x > 0 && x < local[0] - 1 && z > 0 &&
                           z < local[2] - 1 && t2 > 0 &&
                           t2 < local[3] / 2 - 1;
  const bool xy_interior = x > 0 && x < local[0] - 1 && y > 0 &&
                           y < local[1] - 1 && t2 > 0 &&
                           t2 < local[3] / 2 - 1;
  return (grid[0] == 1 &&
          (x == 0 || x == local[0] - 1) && yz_interior) ||
         (grid[1] == 1 &&
          (y == 0 || y == local[1] - 1) && xz_interior) ||
         (grid[2] == 1 &&
          (z == 0 || z == local[2] - 1) && xy_interior);
}

template <typename Real>
inline Complex<Real> rotate_plus_i(const Complex<Real> &value) {
  return Complex<Real>(-value.imag(), value.real());
}

template <typename Real>
inline Complex<Real> rotate_minus_i(const Complex<Real> &value) {
  return Complex<Real>(value.imag(), -value.real());
}

// Fold the gamma matrix into two spin projections, then apply SU(3) to the two
// projected vectors.  This mirrors the CUDA kernel (wilson_dslash.cu): two
// 3x3 complex multiplies (18 MACs) instead of 12 components times 3 columns
// (36 MACs), followed by the cheap +/-1 / +/-i post-combination that the gamma
// transform reduces to.  The lanes are consecutive boundary sites.
template <typename Real, int Direction, bool Dagger, int GammaSign>
void accumulate_hop_lanes(const Complex<Real> source[kVector][64 / sizeof(Real)],
                          const Complex<Real> gauge[kGauge][64 / sizeof(Real)],
                          Complex<Real> result[kVector][64 / sizeof(Real)]) {
  using C = Complex<Real>;
  constexpr int kSiteLanes = 64 / sizeof(Real);
  constexpr bool kNegative = GammaSign < 0;

  C projected0[kColor][kSiteLanes];
  C projected1[kColor][kSiteLanes];
#pragma omp simd
  for (int lane = 0; lane < kSiteLanes; ++lane) {
    for (int color = 0; color < kColor; ++color) {
      const C &v0 = source[0 * kColor + color][lane];
      const C &v1 = source[1 * kColor + color][lane];
      const C &v2 = source[2 * kColor + color][lane];
      const C &v3 = source[3 * kColor + color][lane];
      if constexpr (Direction == 0) {
        const C minus_ig2 =
            kNegative ? rotate_plus_i(v2) : rotate_minus_i(v2);
        const C minus_ig3 =
            kNegative ? rotate_plus_i(v3) : rotate_minus_i(v3);
        projected0[color][lane] = v0 + minus_ig3;
        projected1[color][lane] = v1 + minus_ig2;
      } else if constexpr (Direction == 1) {
        projected0[color][lane] = kNegative ? v0 - v3 : v0 + v3;
        projected1[color][lane] = kNegative ? v1 + v2 : v1 - v2;
      } else if constexpr (Direction == 2) {
        const C minus_ig2 =
            kNegative ? rotate_plus_i(v2) : rotate_minus_i(v2);
        const C ig3 = kNegative ? rotate_minus_i(v3) : rotate_plus_i(v3);
        projected0[color][lane] = v0 + minus_ig2;
        projected1[color][lane] = v1 + ig3;
      } else {
        projected0[color][lane] = kNegative ? v0 + v2 : v0 - v2;
        projected1[color][lane] = kNegative ? v1 + v3 : v1 - v3;
      }
    }
  }

  for (int row = 0; row < kColor; ++row) {
    C transported0[kSiteLanes];
    C transported1[kSiteLanes];
    // Explicit real/imaginary arithmetic so the lane loop vectorises; a
    // std::complex multiply here defeats the OpenMP simd pragma.
#pragma omp simd
    for (int lane = 0; lane < kSiteLanes; ++lane) {
      Real sum0_real = 0;
      Real sum0_imag = 0;
      Real sum1_real = 0;
      Real sum1_imag = 0;
      for (int column = 0; column < kColor; ++column) {
        const int entry =
            Dagger ? column * kColor + row : row * kColor + column;
        const C &u = gauge[entry][lane];
        const Real entry_real = u.real();
        const Real entry_imag = Dagger ? -u.imag() : u.imag();
        const C &p0 = projected0[column][lane];
        const C &p1 = projected1[column][lane];
        sum0_real += p0.real() * entry_real - p0.imag() * entry_imag;
        sum0_imag += p0.real() * entry_imag + p0.imag() * entry_real;
        sum1_real += p1.real() * entry_real - p1.imag() * entry_imag;
        sum1_imag += p1.real() * entry_imag + p1.imag() * entry_real;
      }
      transported0[lane] = C(sum0_real, sum0_imag);
      transported1[lane] = C(sum1_real, sum1_imag);
    }
#pragma omp simd
    for (int lane = 0; lane < kSiteLanes; ++lane) {
      const C &t0 = transported0[lane];
      const C &t1 = transported1[lane];
      result[0 * kColor + row][lane] += t0;
      result[1 * kColor + row][lane] += t1;
      if constexpr (Direction == 0) {
        result[2 * kColor + row][lane] +=
            kNegative ? rotate_minus_i(t1) : rotate_plus_i(t1);
        result[3 * kColor + row][lane] +=
            kNegative ? rotate_minus_i(t0) : rotate_plus_i(t0);
      } else if constexpr (Direction == 1) {
        result[2 * kColor + row][lane] += kNegative ? t1 : -t1;
        result[3 * kColor + row][lane] += kNegative ? -t0 : t0;
      } else if constexpr (Direction == 2) {
        result[2 * kColor + row][lane] +=
            kNegative ? rotate_minus_i(t0) : rotate_plus_i(t0);
        result[3 * kColor + row][lane] +=
            kNegative ? rotate_plus_i(t1) : rotate_minus_i(t1);
      } else {
        result[2 * kColor + row][lane] += kNegative ? t0 : -t0;
        result[3 * kColor + row][lane] += kNegative ? t1 : -t1;
      }
    }
  }
}

// Lane kernel for the boundary sites.  Every lane resolves its source / gauge
// through the same accessors the scalar site kernel uses, so the parity and
// t2-boundary (including time wrap and remote halo) logic is identical.  The
// sites are grouped in boundary_sites_ order, which is linear ascending; at
// ht == 1 (the production configuration) each group is a run of consecutive
// linear sites.
template <typename Real, bool Dagger, int Part, typename SourceAtDeltaFn,
          typename LocalGaugeFn, typename BackwardGaugeFn>
void compute_dslash_boundary_lanes(
    const int64_t *sites, int64_t count, const LatticeGeometry &geometry,
    bool block_dslash, SourceAtDeltaFn source_at_delta,
    LocalGaugeFn local_gauge, BackwardGaugeFn backward_gauge,
    Complex<Real> *output) {
  using C = Complex<Real>;
  constexpr int kSiteLanes = 64 / sizeof(Real);
  const int64_t half_sites = geometry.half_sites;
  const int64_t groups = (count + kSiteLanes - 1) / kSiteLanes;

#pragma omp parallel for schedule(static)
  for (int64_t group = 0; group < groups; ++group) {
    const int64_t base = group * kSiteLanes;
    const int lanes = static_cast<int>(
        std::min<int64_t>(kSiteLanes, count - base));
    C result[kVector][kSiteLanes];
    for (int component = 0; component < kVector; ++component) {
#pragma omp simd
      for (int lane = 0; lane < kSiteLanes; ++lane) {
        result[component][lane] = C(0);
      }
    }

    bool active[kSiteLanes];
    int coordinates[kSiteLanes][kDim];
    for (int lane = 0; lane < lanes; ++lane) {
      int x = 0;
      int y = 0;
      int z = 0;
      int t2 = 0;
      geometry.decode_half(sites[base + lane], x, y, z, t2);
      coordinates[lane][0] = x;
      coordinates[lane][1] = y;
      coordinates[lane][2] = z;
      coordinates[lane][3] = t2;
      active[lane] =
          !(block_dslash && is_periodic_face_site(geometry.local, geometry.grid,
                                                  x, y, z, t2));
    }

    auto run_direction = [&](auto direction_tag) {
      constexpr int Direction = decltype(direction_tag)::value;
      if constexpr ((Part == 1 && Direction == 3) ||
                    (Part == 2 && Direction != 3)) {
        return;
      }
      C forward_source[kVector][kSiteLanes];
      C backward_source[kVector][kSiteLanes];
      C forward_gauge_lanes[kGauge][kSiteLanes];
      C backward_gauge_lanes[kGauge][kSiteLanes];
      // Zero-pad the tail lanes so the fixed-width SIMD loops only ever read
      // initialised data; the tail results are discarded after the scatter.
      for (int component = 0; component < kVector; ++component) {
#pragma omp simd
        for (int lane = 0; lane < kSiteLanes; ++lane) {
          forward_source[component][lane] = C(0);
          backward_source[component][lane] = C(0);
        }
      }
      for (int entry = 0; entry < kGauge; ++entry) {
#pragma omp simd
        for (int lane = 0; lane < kSiteLanes; ++lane) {
          forward_gauge_lanes[entry][lane] = C(0);
          backward_gauge_lanes[entry][lane] = C(0);
        }
      }
      for (int lane = 0; lane < lanes; ++lane) {
        const int x = coordinates[lane][0];
        const int y = coordinates[lane][1];
        const int z = coordinates[lane][2];
        const int t2 = coordinates[lane][3];
        const VectorView<Real> forward = source_at_delta(
            Direction, x, y, z, t2, 1);
        const VectorView<Real> backward = source_at_delta(
            Direction, x, y, z, t2, -1);
        const GaugeView<Real> gauge_forward =
            local_gauge(Direction, x, y, z, t2);
        const GaugeView<Real> gauge_backward =
            backward_gauge(Direction, x, y, z, t2);
        for (int component = 0; component < kVector; ++component) {
          forward_source[component][lane] = forward[component];
          backward_source[component][lane] = backward[component];
        }
        for (int entry = 0; entry < kGauge; ++entry) {
          forward_gauge_lanes[entry][lane] =
              gauge_forward.data[entry * gauge_forward.matrix_stride];
          backward_gauge_lanes[entry][lane] =
              gauge_backward.data[entry * gauge_backward.matrix_stride];
        }
      }
      if constexpr (Dagger) {
        accumulate_hop_lanes<Real, Direction, true, 1>(
            backward_source, backward_gauge_lanes, result);
        accumulate_hop_lanes<Real, Direction, false, -1>(
            forward_source, forward_gauge_lanes, result);
      } else {
        accumulate_hop_lanes<Real, Direction, false, 1>(
            forward_source, forward_gauge_lanes, result);
        accumulate_hop_lanes<Real, Direction, true, -1>(
            backward_source, backward_gauge_lanes, result);
      }
    };
    run_direction(std::integral_constant<int, 0>{});
    run_direction(std::integral_constant<int, 1>{});
    run_direction(std::integral_constant<int, 2>{});
    run_direction(std::integral_constant<int, 3>{});

    for (int lane = 0; lane < lanes; ++lane) {
      if (!active[lane]) {
        continue;
      }
      const int64_t site = sites[base + lane];
      for (int component = 0; component < kVector; ++component) {
        C *destination =
            output + static_cast<int64_t>(component) * half_sites + site;
        if constexpr (Part == 2) {
          destination[0] += result[component][lane];
        } else {
          destination[0] = result[component][lane];
        }
      }
    }
  }
}

} // namespace

template <typename Real>
void LatticeSet<Real>::pack_fermion_face(const Complex<Real> *source,
                                         int parity, int dim, int boundary,
                                         Complex<Real> *buffer) {
  const int64_t count = geometry_.half_sites / surface_extent(dim);
  if (!halo_pack6_enabled()) {
    const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t face_site = 0; face_site < count; ++face_site) {
      int x = 0;
      int y = 0;
      int z = 0;
      int t2 = 0;
      decode_face(face_site, dim, boundary, x, y, z, t2);
      const int64_t target = face_site * kVector;
      const int64_t site = site_linear(x, y, z, t2);
      for (int component = 0; component < kVector; ++component) {
        buffer[target + component] =
            source[static_cast<int64_t>(component) * geometry_.half_sites +
                   site];
      }
    }
    (void)parity;
    return;
  }

  // KUPL_HALO_PACK6.  `buffer` identifies which face is being packed: the
  // plus buffer feeds the +dim neighbour, which consumes it in its backward
  // hop, and vice versa.  The gamma sign follows the consumer's hop sign and
  // flips for the daggered operator.
  const bool plus_face = buffer == fermion_send_[face_offset(dim, 1)].data();
  const int receiver_sign = plus_face ? -1 : 1;
  const int gamma_sign = g_halo_pack_dagger ? -receiver_sign : receiver_sign;
  const bool half_face =
      halo_pack6_half_face(dim, geometry_.local[0], geometry_.local[1],
                           geometry_.local[2]);
  const int64_t face_count = half_face ? count / 2 : count;
  // The t face is read only on its even-odd half; because the compact field is
  // already one parity, that half is selected by the (x + y + z) parity.
  const int select_parity = plus_face ? (1 - parity) : parity;
  const int64_t half_sites = geometry_.half_sites;
  const int ly = geometry_.local[1];
  const int lz = geometry_.local[2];
  const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t face_site = 0; face_site < count; ++face_site) {
    int x = 0;
    int y = 0;
    int z = 0;
    int t2 = 0;
    decode_face(face_site, dim, boundary, x, y, z, t2);
    if (half_face && (((x + y + z) & 1) != select_parity)) {
      continue;
    }
    const int64_t rank = half_face ? halo_t_half_rank(x, y, z, select_parity,
                                                      ly, lz)
                                   : face_site;
    const int64_t site = site_linear(x, y, z, t2);
    Complex<Real> vector[kVector];
    for (int component = 0; component < kVector; ++component) {
      vector[component] =
          source[static_cast<int64_t>(component) * half_sites + site];
    }
    Complex<Real> half[2 * kColor];
    project_half_spinor_dispatch<Real>(vector, half, dim, gamma_sign);
    for (int component = 0; component < 2 * kColor; ++component) {
      buffer[static_cast<int64_t>(component) * face_count + rank] =
          half[component];
    }
  }
}

template <typename Real>
void LatticeSet<Real>::post_fermion_halo(const Complex<Real> *source,
                                         int parity) {
  fermion_requests_.clear();
  fermion_requests_.reserve(16);
  for (int dim = 0; dim < kDim; ++dim) {
    if (geometry_.grid[dim] == 1) {
      continue;
    }
    auto &send_minus = fermion_send_[face_offset(dim, -1)];
    auto &send_plus = fermion_send_[face_offset(dim, 1)];
    auto &recv_minus = fermion_recv_[face_offset(dim, -1)];
    auto &recv_plus = fermion_recv_[face_offset(dim, 1)];
    const bool pack6 = halo_pack6_enabled();
    const int64_t face_count =
        pack6 ? halo_pack6_face_count(
                    dim, geometry_.face_sites[dim], geometry_.local[0],
                    geometry_.local[1], geometry_.local[2])
              : geometry_.face_sites[dim];
    const std::size_t wire_components =
        pack6 ? static_cast<std::size_t>(6 * face_count)
              : static_cast<std::size_t>(face_count * kVector);
    const int wire_bytes =
        static_cast<int>(wire_components * sizeof(Complex<Real>));
    // ht == 1 makes the two t faces byte-identical (both are t2 == 0).  A full
    // 12-component pack may then be shared by both Isend calls.  PACK6 cannot
    // share: the two neighbours consume opposite gamma projections.
    const bool dedup_pack =
        !pack6 && halo_dedup_pack_enabled() &&
        geometry_.last_half_coordinate(dim) == 0;
    pack_fermion_face(source, parity, dim, 0, send_minus.data());
    if (!dedup_pack) {
      pack_fermion_face(source, parity, dim,
                        geometry_.last_half_coordinate(dim),
                        send_plus.data());
    }
    const Complex<Real> *plus_source =
        dedup_pack ? send_minus.data() : send_plus.data();
    MPI_Request request = MPI_REQUEST_NULL;
    MPI_Irecv(recv_plus.data(), wire_bytes, MPI_BYTE,
              geometry_.neighbor_plus[dim], fermion_face_tag(dim, -1),
              geometry_.comm, &request);
    fermion_requests_.push_back(request);
    MPI_Irecv(recv_minus.data(), wire_bytes, MPI_BYTE,
              geometry_.neighbor_minus[dim], fermion_face_tag(dim, 1),
              geometry_.comm, &request);
    fermion_requests_.push_back(request);
    MPI_Isend(plus_source, wire_bytes, MPI_BYTE,
              geometry_.neighbor_plus[dim], fermion_face_tag(dim, 1),
              geometry_.comm, &request);
    fermion_requests_.push_back(request);
    MPI_Isend(send_minus.data(), wire_bytes, MPI_BYTE,
              geometry_.neighbor_minus[dim], fermion_face_tag(dim, -1),
              geometry_.comm, &request);
    fermion_requests_.push_back(request);
  }
}

template <typename Real>
VectorView<Real> LatticeSet<Real>::source_at_delta(
    const Complex<Real> *local_source, int x, int y, int z, int t2, int dim,
    int sign, int target) const {
  int coordinate[kDim] = {x, y, z, t2};
  bool remote = false;
  if (dim == 3) {
    const int target_time =
        geometry_.time_for_parity(x, y, z, t2, target);
    int moved_time = target_time + sign;
    if (moved_time < 0 || moved_time >= geometry_.local[3]) {
      if (geometry_.grid[3] == 1) {
        moved_time =
            moved_time < 0 ? geometry_.local[3] - 1 : 0;
      } else {
        remote = true;
      }
    }
    if (!remote) {
      coordinate[3] = moved_time / 2;
    }
  } else {
    coordinate[dim] += sign;
    if (coordinate[dim] < 0 ||
        coordinate[dim] >= geometry_.local[dim]) {
      if (geometry_.grid[dim] == 1) {
        coordinate[dim] =
            coordinate[dim] < 0 ? geometry_.local[dim] - 1 : 0;
      } else {
        remote = true;
      }
    }
  }
  if (!remote) {
    return compact_view(
        local_source,
        site_linear(coordinate[0], coordinate[1], coordinate[2],
                    coordinate[3]));
  }
  const auto &halo = fermion_recv_[face_offset(dim, sign)];
  if (halo_pack6_enabled()) {
    const bool half_face =
        halo_pack6_half_face(dim, geometry_.local[0], geometry_.local[1],
                             geometry_.local[2]);
    // A +sign hop off the high t face reaches the upper neighbour's backward
    // hop, which reads the (x + y + z) == 1 - target half; a -sign hop off the
    // low face reaches the lower neighbour's forward hop (== target half).
    const int read_parity = sign > 0 ? (1 - target) : target;
    const int64_t face_count =
        halo_pack6_face_count(dim, geometry_.face_sites[dim],
                              geometry_.local[0], geometry_.local[1],
                              geometry_.local[2]);
    const int64_t rank =
        half_face ? halo_t_half_rank(x, y, z, read_parity,
                                     geometry_.local[1], geometry_.local[2])
                  : geometry_.face_index(dim, x, y, z, t2);
    return {halo.data() + rank, face_count};
  }
  const int64_t index = geometry_.face_index(dim, x, y, z, t2);
  return halo_view(halo, index);
}

template <typename Real>
void LatticeSet<Real>::compute_dslash_interior(const Complex<Real> *source,
                                               Complex<Real> *output,
                                               int target) {
  if (!use_block_dslash(geometry_.half_sites)) {
    const int64_t count = geometry_.half_sites;
    const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t linear = 0; linear < count; ++linear) {
      int x = 0;
      int y = 0;
      int z = 0;
      int t2 = 0;
      geometry_.decode_half(linear, x, y, z, t2);
      if (x == 0 || y == 0 || z == 0 || t2 == 0 ||
          x == geometry_.local[0] - 1 || y == geometry_.local[1] - 1 ||
          z == geometry_.local[2] - 1 ||
          t2 == geometry_.local[3] / 2 - 1) {
        continue;
      }
      compute_dslash_site(source, output, x, y, z, t2, target);
    }
    return;
  }
  compute_interior_blocks<Real, false>(
      source, output, gauge_halo_.source, geometry_.local, target);
  compute_periodic_face_blocks<Real, false>(
      source, output, gauge_halo_.source, geometry_.local, geometry_.grid,
      target);
}

template <typename Real>
void LatticeSet<Real>::compute_dslash_boundary(
    const Complex<Real> *local_source, Complex<Real> *output, int target) {
  const int64_t count = static_cast<int64_t>(boundary_sites_.size());
  const int64_t chunk = omp_chunk<Real>(count);
  const bool block_dslash = use_block_dslash(geometry_.half_sites);
  if (use_boundary_lanes()) {
    auto source_fn = [&](int direction, int x, int y, int z, int t2,
                         int sign) {
      return source_at_delta(local_source, x, y, z, t2, direction, sign,
                             target);
    };
    auto gauge_fn = [&](int direction, int x, int y, int z, int t2) {
      return local_gauge(target, direction, x, y, z, t2);
    };
    auto backward_gauge_fn = [&](int direction, int x, int y, int z,
                                 int t2) {
      return backward_gauge(x, y, z, t2, direction, target);
    };
    compute_dslash_boundary_lanes<Real, false, 0>(
        boundary_sites_.data(), count, geometry_, block_dslash, source_fn,
        gauge_fn, backward_gauge_fn, output);
    return;
  }
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < count; ++index) {
    const int64_t linear = boundary_sites_[static_cast<std::size_t>(index)];
    int x = 0;
    int y = 0;
    int z = 0;
    int t2 = 0;
    geometry_.decode_half(linear, x, y, z, t2);
    if (block_dslash &&
        is_periodic_face_site(geometry_.local, geometry_.grid, x, y, z,
                              t2)) {
      continue;
    }
    compute_dslash_site(local_source, output, x, y, z, t2, target);
  }
}

template <typename Real>
void LatticeSet<Real>::compute_dslash_boundary_part(
    const Complex<Real> *local_source, Complex<Real> *output, int target,
    int part) {
  const int64_t count = static_cast<int64_t>(boundary_sites_.size());
  const int64_t chunk = omp_chunk<Real>(count);
  const bool block_dslash = use_block_dslash(geometry_.half_sites);
  if (use_boundary_lanes()) {
    auto source_fn = [&](int direction, int x, int y, int z, int t2,
                         int sign) {
      return source_at_delta(local_source, x, y, z, t2, direction, sign,
                             target);
    };
    auto gauge_fn = [&](int direction, int x, int y, int z, int t2) {
      return local_gauge(target, direction, x, y, z, t2);
    };
    auto backward_gauge_fn = [&](int direction, int x, int y, int z,
                                 int t2) {
      return backward_gauge(x, y, z, t2, direction, target);
    };
    if (part == 2) {
      compute_dslash_boundary_lanes<Real, false, 2>(
          boundary_sites_.data(), count, geometry_, block_dslash, source_fn,
          gauge_fn, backward_gauge_fn, output);
    } else if (part == 1) {
      compute_dslash_boundary_lanes<Real, false, 1>(
          boundary_sites_.data(), count, geometry_, block_dslash, source_fn,
          gauge_fn, backward_gauge_fn, output);
    } else {
      compute_dslash_boundary_lanes<Real, false, 0>(
          boundary_sites_.data(), count, geometry_, block_dslash, source_fn,
          gauge_fn, backward_gauge_fn, output);
    }
    return;
  }
  // KUPL_T_LOCAL_IN_PART1: with local_t == 2 exactly one t hop per site stays
  // on-rank.  part 1 computes directions 0..2 plus that local t hop, so part 2
  // only has to add the remote t hop after the halo wait.  The scalar
  // site-kernel boundary path (block dslash disables the split) keeps the same
  // per-direction accumulation order; only the association of the t pair moves.
  const bool t_split = t_local_part1_enabled() &&
                       geometry_.local[3] == 2 && !block_dslash &&
                       (part == 1 || part == 2);
  auto compute_t_split = [&](int x, int y, int z, int t2) {
    using C = Complex<Real>;
    const int64_t site = site_linear(x, y, z, t2);
    const int target_time =
        geometry_.time_for_parity(x, y, z, t2, target);
    C result[kVector];
    std::fill(std::begin(result), std::end(result), C(0));
    if (part == 1) {
      accumulate_direction<Real, 0>(
          source_at_delta(local_source, x, y, z, t2, 0, 1, target),
          local_gauge(target, 0, x, y, z, t2),
          source_at_delta(local_source, x, y, z, t2, 0, -1, target),
          backward_gauge(x, y, z, t2, 0, target), result);
      accumulate_direction<Real, 1>(
          source_at_delta(local_source, x, y, z, t2, 1, 1, target),
          local_gauge(target, 1, x, y, z, t2),
          source_at_delta(local_source, x, y, z, t2, 1, -1, target),
          backward_gauge(x, y, z, t2, 1, target), result);
      accumulate_direction<Real, 2>(
          source_at_delta(local_source, x, y, z, t2, 2, 1, target),
          local_gauge(target, 2, x, y, z, t2),
          source_at_delta(local_source, x, y, z, t2, 2, -1, target),
          backward_gauge(x, y, z, t2, 2, target), result);
      if (target_time == 0) {
        accumulate_hopping<Real, 3, false, 1>(
            source_at_delta(local_source, x, y, z, t2, 3, 1, target),
            local_gauge(target, 3, x, y, z, t2), result);
      } else {
        accumulate_hopping<Real, 3, true, -1>(
            source_at_delta(local_source, x, y, z, t2, 3, -1, target),
            backward_gauge(x, y, z, t2, 3, target), result);
      }
      scatter_vector(result, output, site);
    } else {
      if (target_time == 0) {
        accumulate_hopping<Real, 3, true, -1>(
            source_at_delta(local_source, x, y, z, t2, 3, -1, target),
            backward_gauge(x, y, z, t2, 3, target), result);
      } else {
        accumulate_hopping<Real, 3, false, 1>(
            source_at_delta(local_source, x, y, z, t2, 3, 1, target),
            local_gauge(target, 3, x, y, z, t2), result);
      }
      C existing[kVector];
      gather_vector(compact_view(output, site), existing);
      for (int i = 0; i < kVector; ++i) {
        existing[i] += result[i];
      }
      scatter_vector(existing, output, site);
    }
  };
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < count; ++index) {
    const int64_t linear = boundary_sites_[static_cast<std::size_t>(index)];
    int x = 0;
    int y = 0;
    int z = 0;
    int t2 = 0;
    geometry_.decode_half(linear, x, y, z, t2);
    if (block_dslash &&
        is_periodic_face_site(geometry_.local, geometry_.grid, x, y, z,
                              t2)) {
      continue;
    }
    if (t_split) {
      compute_t_split(x, y, z, t2);
    } else {
      compute_dslash_site(local_source, output, x, y, z, t2, target, part);
    }
  }
}

template <typename Real>
void LatticeSet<Real>::compute_dslash_t2_spatial_zblocks(
    const Complex<Real> *local_source, Complex<Real> *output, int target) {
  using C = Complex<Real>;
  constexpr int kSiteLanes = 64 / sizeof(Real);
  const int x_dim = geometry_.local[0];
  const int y_dim = geometry_.local[1];
  const int z_dim = geometry_.local[2];
  const int source_parity = 1 - target;
  const int64_t half_sites = geometry_.half_sites;
  const int64_t gauge_stride = static_cast<int64_t>(kDim) * half_sites;

#pragma omp parallel for collapse(2) schedule(static)
  for (int x = 0; x < x_dim; ++x) {
    for (int y = 0; y < y_dim; ++y) {
      const int x_forward = x + 1 < x_dim ? x + 1 : 0;
      const int x_backward = x > 0 ? x - 1 : x_dim - 1;
      const int y_forward = y + 1 < y_dim ? y + 1 : 0;
      const int y_backward = y > 0 ? y - 1 : y_dim - 1;
      for (int z0 = 0; z0 < z_dim; z0 += kSiteLanes) {
        const int lanes = std::min(kSiteLanes, z_dim - z0);
        const int64_t site =
            (static_cast<int64_t>(x) * y_dim + y) * z_dim + z0;
        C result[kVector][kSiteLanes];
        for (int component = 0; component < kVector; ++component) {
#pragma omp simd
          for (int lane = 0; lane < kSiteLanes; ++lane) {
            result[component][lane] = C(0);
          }
        }

        const int64_t x_forward_site =
            (static_cast<int64_t>(x_forward) * y_dim + y) * z_dim + z0;
        const int64_t x_backward_site =
            (static_cast<int64_t>(x_backward) * y_dim + y) * z_dim + z0;
        const int64_t y_forward_site =
            (static_cast<int64_t>(x) * y_dim + y_forward) * z_dim + z0;
        const int64_t y_backward_site =
            (static_cast<int64_t>(x) * y_dim + y_backward) * z_dim + z0;
        accumulate_hopping_block<Real, 0, false, 1>(
            local_source + x_forward_site,
            local_gauge(target, 0, x, y, z0, 0).data, half_sites,
            gauge_stride, lanes, result);
        accumulate_hopping_block<Real, 0, true, -1>(
            local_source + x_backward_site,
            local_gauge(source_parity, 0, x_backward, y, z0, 0).data,
            half_sites, gauge_stride, lanes, result);
        accumulate_hopping_block<Real, 1, false, 1>(
            local_source + y_forward_site,
            local_gauge(target, 1, x, y, z0, 0).data, half_sites,
            gauge_stride, lanes, result);
        accumulate_hopping_block<Real, 1, true, -1>(
            local_source + y_backward_site,
            local_gauge(source_parity, 1, x, y_backward, z0, 0).data,
            half_sites, gauge_stride, lanes, result);

        const int z_interior_begin = std::max(z0, 1);
        const int z_interior_end = std::min(z0 + lanes, z_dim - 1);
        if (z_interior_begin < z_interior_end) {
          const int interior_lanes = z_interior_end - z_interior_begin;
          const int64_t interior_site =
              site + (z_interior_begin - z0);
          accumulate_hopping_block<Real, 2, false, 1>(
              local_source + interior_site + 1,
              local_gauge(target, 2, x, y, z_interior_begin, 0).data,
              half_sites, gauge_stride, interior_lanes, result);
          accumulate_hopping_block<Real, 2, true, -1>(
              local_source + interior_site - 1,
              local_gauge(source_parity, 2, x, y,
                          z_interior_begin - 1, 0).data,
              half_sites, gauge_stride, interior_lanes, result);
        }

        if (z0 == 0) {
          const int64_t lower_site = site;
          accumulate_hopping_block<Real, 2, false, 1>(
              local_source + lower_site + 1,
              local_gauge(target, 2, x, y, 0, 0).data, half_sites,
              gauge_stride, 1, result);
          const int64_t upper_site = site + z_dim - 1;
          accumulate_hopping_block<Real, 2, true, -1>(
              local_source + upper_site,
              local_gauge(source_parity, 2, x, y, z_dim - 1, 0).data,
              half_sites, gauge_stride, 1, result);
        }
        if (z0 + lanes == z_dim) {
          const int64_t upper_site = site + lanes - 1;
          const int64_t wrapped_site = site;
          accumulate_hopping_block<Real, 2, false, 1>(
              local_source + wrapped_site,
              local_gauge(target, 2, x, y, z_dim - 1, 0).data,
              half_sites, gauge_stride, 1, result);
          accumulate_hopping_block<Real, 2, true, -1>(
              local_source + upper_site - 1,
              local_gauge(source_parity, 2, x, y, z_dim - 2, 0).data,
              half_sites, gauge_stride, 1, result);
        }

        for (int component = 0; component < kVector; ++component) {
          C *destination = output + static_cast<int64_t>(component) *
                                       half_sites + site;
#pragma omp simd
          for (int lane = 0; lane < lanes; ++lane) {
            destination[lane] = result[component][lane];
          }
        }
      }
    }
  }
}

template <typename Real>
void LatticeSet<Real>::compute_dslash_t2_time_zblocks(
    const Complex<Real> *local_source, Complex<Real> *output, int target) {
  using C = Complex<Real>;
  constexpr int kSiteLanes = 64 / sizeof(Real);
  const int x_dim = geometry_.local[0];
  const int y_dim = geometry_.local[1];
  const int z_dim = geometry_.local[2];
  const int source_parity = 1 - target;
  const int64_t half_sites = geometry_.half_sites;
  const int64_t gauge_stride = static_cast<int64_t>(kDim) * half_sites;
  const bool pack6 = halo_pack6_enabled();
  const bool half_face = halo_pack6_half_face(3, x_dim, y_dim, z_dim);
  const int64_t t_face_count = halo_pack6_face_count(
      3, geometry_.face_sites[3], x_dim, y_dim, z_dim);

#pragma omp parallel for collapse(2) schedule(static)
  for (int x = 0; x < x_dim; ++x) {
    for (int y = 0; y < y_dim; ++y) {
      for (int z0 = 0; z0 < z_dim; z0 += kSiteLanes) {
        const int lanes = std::min(kSiteLanes, z_dim - z0);
        const int64_t site =
            (static_cast<int64_t>(x) * y_dim + y) * z_dim + z0;
        C result[kVector][kSiteLanes];
        C forward_source[kVector][kSiteLanes];
        C backward_source[kVector][kSiteLanes];
        C backward_gauge[kGauge][kSiteLanes];
        for (int component = 0; component < kVector; ++component) {
#pragma omp simd
          for (int lane = 0; lane < kSiteLanes; ++lane) {
            result[component][lane] = C(0);
            forward_source[component][lane] = C(0);
            backward_source[component][lane] = C(0);
          }
        }
        for (int component = 0; component < kGauge; ++component) {
#pragma omp simd
          for (int lane = 0; lane < kSiteLanes; ++lane) {
            backward_gauge[component][lane] = C(0);
          }
        }

        const auto &forward_halo =
            fermion_recv_[face_offset(3, 1)];
        const auto &backward_halo =
            fermion_recv_[face_offset(3, -1)];
        const auto &backward_gauge_halo =
            gauge_halo_.faces[face_offset(3, -1)];
        const C *local_backward_gauge =
            local_gauge(source_parity, 3, x, y, z0, 0).data;
        for (int lane = 0; lane < lanes; ++lane) {
          const int z = z0 + lane;
          const int64_t lane_site = site + lane;
          const int target_time = ((x + y + z) & 1) ^ target;
          const int64_t face_site =
              (static_cast<int64_t>(x) * y_dim + y) * z_dim + z;
          const bool forward_remote = target_time != 0;
          int64_t forward_rank = face_site;
          int64_t backward_rank = face_site;
          if (pack6 && half_face) {
            // +t reads the (x+y+z) == 1 - target half (upper neighbour's
            // backward hop); -t reads the == target half.
            forward_rank =
                halo_t_half_rank(x, y, z, 1 - target, y_dim, z_dim);
            backward_rank =
                halo_t_half_rank(x, y, z, target, y_dim, z_dim);
          }
          for (int component = 0; component < kVector; ++component) {
            const int64_t local_index =
                static_cast<int64_t>(component) * half_sites + lane_site;
            const int64_t halo_index =
                pack6 ? static_cast<int64_t>(component) * t_face_count +
                            (forward_remote ? forward_rank : backward_rank)
                      : face_site * kVector + component;
            forward_source[component][lane] =
                forward_remote ? forward_halo[halo_index]
                               : local_source[local_index];
            backward_source[component][lane] =
                forward_remote
                    ? local_source[local_index]
                    : backward_halo[halo_index];
          }
          if (forward_remote) {
            const int64_t face_base =
                face_site * 2 * kGauge * kDim;
            const int64_t parity_base =
                static_cast<int64_t>(source_parity) * kGauge * kDim;
            for (int row = 0; row < kColor; ++row) {
              for (int column = 0; column < kColor; ++column) {
                const int component = row * kColor + column;
                backward_gauge[component][lane] =
                    backward_gauge_halo[face_base + parity_base +
                                        component * kDim + 3];
              }
            }
          } else {
            for (int row = 0; row < kColor; ++row) {
              for (int column = 0; column < kColor; ++column) {
                const int component = row * kColor + column;
                backward_gauge[component][lane] =
                    local_backward_gauge[
                        static_cast<int64_t>(component) * gauge_stride +
                        lane];
              }
            }
          }
        }

        accumulate_hopping_block<Real, 3, false, 1>(
            &forward_source[0][0],
            local_gauge(target, 3, x, y, z0, 0).data, kSiteLanes,
            gauge_stride, lanes, result);
        accumulate_hopping_block<Real, 3, true, -1>(
            &backward_source[0][0], &backward_gauge[0][0], kSiteLanes,
            kSiteLanes, lanes, result);
        for (int component = 0; component < kVector; ++component) {
          C *destination = output + static_cast<int64_t>(component) *
                                       half_sites + site;
#pragma omp simd
          for (int lane = 0; lane < lanes; ++lane) {
            destination[lane] += result[component][lane];
          }
        }
      }
    }
  }
}

template <typename Real>
void LatticeSet<Real>::compute_dslash_site(
    const Complex<Real> *local_source, Complex<Real> *output, int x, int y,
    int z, int t2, int target, int part) {
  using C = Complex<Real>;
  C result[kVector];
  std::fill(std::begin(result), std::end(result), C(0));
  if (part != 2) {
  accumulate_direction<Real, 0>(
      source_at_delta(local_source, x, y, z, t2, 0, 1, target),
      local_gauge(target, 0, x, y, z, t2),
      source_at_delta(local_source, x, y, z, t2, 0, -1, target),
      backward_gauge(x, y, z, t2, 0, target), result);
  accumulate_direction<Real, 1>(
      source_at_delta(local_source, x, y, z, t2, 1, 1, target),
      local_gauge(target, 1, x, y, z, t2),
      source_at_delta(local_source, x, y, z, t2, 1, -1, target),
      backward_gauge(x, y, z, t2, 1, target), result);
  accumulate_direction<Real, 2>(
      source_at_delta(local_source, x, y, z, t2, 2, 1, target),
      local_gauge(target, 2, x, y, z, t2),
      source_at_delta(local_source, x, y, z, t2, 2, -1, target),
      backward_gauge(x, y, z, t2, 2, target), result);
  }
  if (part != 1) {
  accumulate_direction<Real, 3>(
      source_at_delta(local_source, x, y, z, t2, 3, 1, target),
      local_gauge(target, 3, x, y, z, t2),
      source_at_delta(local_source, x, y, z, t2, 3, -1, target),
      backward_gauge(x, y, z, t2, 3, target), result);
  }
  const int64_t site = site_linear(x, y, z, t2);
  if (part == 2) {
    C existing[kVector];
    gather_vector(compact_view(output, site), existing);
    for (int i = 0; i < kVector; ++i) {
      existing[i] += result[i];
    }
    scatter_vector(existing, output, site);
  } else {
    scatter_vector(result, output, site);
  }
}

template <typename Real>
void LatticeSet<Real>::compute_dslash_dagger_interior(
    const Complex<Real> *source, Complex<Real> *output, int target) {
  if (!use_block_dslash(geometry_.half_sites)) {
    const int64_t count = geometry_.half_sites;
    const int64_t chunk = omp_chunk<Real>(count);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t linear = 0; linear < count; ++linear) {
      int x = 0;
      int y = 0;
      int z = 0;
      int t2 = 0;
      geometry_.decode_half(linear, x, y, z, t2);
      if (x == 0 || y == 0 || z == 0 || t2 == 0 ||
          x == geometry_.local[0] - 1 || y == geometry_.local[1] - 1 ||
          z == geometry_.local[2] - 1 ||
          t2 == geometry_.local[3] / 2 - 1) {
        continue;
      }
      compute_dslash_dagger_site(source, output, x, y, z, t2, target);
    }
    return;
  }
  compute_interior_blocks<Real, true>(
      source, output, gauge_halo_.source, geometry_.local, target);
  compute_periodic_face_blocks<Real, true>(
      source, output, gauge_halo_.source, geometry_.local, geometry_.grid,
      target);
}

template <typename Real>
void LatticeSet<Real>::compute_dslash_dagger_boundary(
    const Complex<Real> *local_source, Complex<Real> *output, int target) {
  const int64_t count = static_cast<int64_t>(boundary_sites_.size());
  const int64_t chunk = omp_chunk<Real>(count);
  const bool block_dslash = use_block_dslash(geometry_.half_sites);
  if (use_boundary_lanes()) {
    auto source_fn = [&](int direction, int x, int y, int z, int t2,
                         int sign) {
      return source_at_delta(local_source, x, y, z, t2, direction, sign,
                             target);
    };
    auto gauge_fn = [&](int direction, int x, int y, int z, int t2) {
      return local_gauge(target, direction, x, y, z, t2);
    };
    auto backward_gauge_fn = [&](int direction, int x, int y, int z,
                                 int t2) {
      return backward_gauge(x, y, z, t2, direction, target);
    };
    compute_dslash_boundary_lanes<Real, true, 0>(
        boundary_sites_.data(), count, geometry_, block_dslash, source_fn,
        gauge_fn, backward_gauge_fn, output);
    return;
  }
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < count; ++index) {
    const int64_t linear = boundary_sites_[static_cast<std::size_t>(index)];
    int x = 0;
    int y = 0;
    int z = 0;
    int t2 = 0;
    geometry_.decode_half(linear, x, y, z, t2);
    if (block_dslash &&
        is_periodic_face_site(geometry_.local, geometry_.grid, x, y, z,
                              t2)) {
      continue;
    }
    compute_dslash_dagger_site(local_source, output, x, y, z, t2, target);
  }
}

template <typename Real>
void LatticeSet<Real>::compute_dslash_dagger_boundary_part(
    const Complex<Real> *local_source, Complex<Real> *output, int target,
    int part) {
  const int64_t count = static_cast<int64_t>(boundary_sites_.size());
  const int64_t chunk = omp_chunk<Real>(count);
  const bool block_dslash = use_block_dslash(geometry_.half_sites);
  if (use_boundary_lanes()) {
    auto source_fn = [&](int direction, int x, int y, int z, int t2,
                         int sign) {
      return source_at_delta(local_source, x, y, z, t2, direction, sign,
                             target);
    };
    auto gauge_fn = [&](int direction, int x, int y, int z, int t2) {
      return local_gauge(target, direction, x, y, z, t2);
    };
    auto backward_gauge_fn = [&](int direction, int x, int y, int z,
                                 int t2) {
      return backward_gauge(x, y, z, t2, direction, target);
    };
    if (part == 2) {
      compute_dslash_boundary_lanes<Real, true, 2>(
          boundary_sites_.data(), count, geometry_, block_dslash, source_fn,
          gauge_fn, backward_gauge_fn, output);
    } else if (part == 1) {
      compute_dslash_boundary_lanes<Real, true, 1>(
          boundary_sites_.data(), count, geometry_, block_dslash, source_fn,
          gauge_fn, backward_gauge_fn, output);
    } else {
      compute_dslash_boundary_lanes<Real, true, 0>(
          boundary_sites_.data(), count, geometry_, block_dslash, source_fn,
          gauge_fn, backward_gauge_fn, output);
    }
    return;
  }
  // KUPL_T_LOCAL_IN_PART1 for the daggered operator (see the non-daggered
  // compute_dslash_boundary_part for the rationale).
  const bool t_split = t_local_part1_enabled() &&
                       geometry_.local[3] == 2 && !block_dslash &&
                       (part == 1 || part == 2);
  auto compute_t_split = [&](int x, int y, int z, int t2) {
    using C = Complex<Real>;
    const int64_t site = site_linear(x, y, z, t2);
    const int target_time =
        geometry_.time_for_parity(x, y, z, t2, target);
    C result[kVector];
    std::fill(std::begin(result), std::end(result), C(0));
    if (part == 1) {
      accumulate_hopping<Real, 0, true, 1>(
          source_at_delta(local_source, x, y, z, t2, 0, -1, target),
          backward_gauge(x, y, z, t2, 0, target), result);
      accumulate_hopping<Real, 0, false, -1>(
          source_at_delta(local_source, x, y, z, t2, 0, 1, target),
          local_gauge(target, 0, x, y, z, t2), result);
      accumulate_hopping<Real, 1, true, 1>(
          source_at_delta(local_source, x, y, z, t2, 1, -1, target),
          backward_gauge(x, y, z, t2, 1, target), result);
      accumulate_hopping<Real, 1, false, -1>(
          source_at_delta(local_source, x, y, z, t2, 1, 1, target),
          local_gauge(target, 1, x, y, z, t2), result);
      accumulate_hopping<Real, 2, true, 1>(
          source_at_delta(local_source, x, y, z, t2, 2, -1, target),
          backward_gauge(x, y, z, t2, 2, target), result);
      accumulate_hopping<Real, 2, false, -1>(
          source_at_delta(local_source, x, y, z, t2, 2, 1, target),
          local_gauge(target, 2, x, y, z, t2), result);
      if (target_time == 0) {
        accumulate_hopping<Real, 3, false, -1>(
            source_at_delta(local_source, x, y, z, t2, 3, 1, target),
            local_gauge(target, 3, x, y, z, t2), result);
      } else {
        accumulate_hopping<Real, 3, true, 1>(
            source_at_delta(local_source, x, y, z, t2, 3, -1, target),
            backward_gauge(x, y, z, t2, 3, target), result);
      }
      scatter_vector(result, output, site);
    } else {
      if (target_time == 0) {
        accumulate_hopping<Real, 3, true, 1>(
            source_at_delta(local_source, x, y, z, t2, 3, -1, target),
            backward_gauge(x, y, z, t2, 3, target), result);
      } else {
        accumulate_hopping<Real, 3, false, -1>(
            source_at_delta(local_source, x, y, z, t2, 3, 1, target),
            local_gauge(target, 3, x, y, z, t2), result);
      }
      C existing[kVector];
      gather_vector(compact_view(output, site), existing);
      for (int i = 0; i < kVector; ++i) {
        existing[i] += result[i];
      }
      scatter_vector(existing, output, site);
    }
  };
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < count; ++index) {
    const int64_t linear = boundary_sites_[static_cast<std::size_t>(index)];
    int x = 0;
    int y = 0;
    int z = 0;
    int t2 = 0;
    geometry_.decode_half(linear, x, y, z, t2);
    if (block_dslash &&
        is_periodic_face_site(geometry_.local, geometry_.grid, x, y, z,
                              t2)) {
      continue;
    }
    if (t_split) {
      compute_t_split(x, y, z, t2);
    } else {
      compute_dslash_dagger_site(local_source, output, x, y, z, t2, target,
                                 part);
    }
  }
}

template <typename Real>
void LatticeSet<Real>::compute_dslash_dagger_site(
    const Complex<Real> *local_source, Complex<Real> *output, int x, int y,
    int z, int t2, int target, int part) {
  using C = Complex<Real>;
  C result[kVector];
  std::fill(std::begin(result), std::end(result), C(0));
  if (part != 2) {
  accumulate_hopping<Real, 0, true, 1>(
      source_at_delta(local_source, x, y, z, t2, 0, -1, target),
      backward_gauge(x, y, z, t2, 0, target), result);
  accumulate_hopping<Real, 0, false, -1>(
      source_at_delta(local_source, x, y, z, t2, 0, 1, target),
      local_gauge(target, 0, x, y, z, t2), result);
  accumulate_hopping<Real, 1, true, 1>(
      source_at_delta(local_source, x, y, z, t2, 1, -1, target),
      backward_gauge(x, y, z, t2, 1, target), result);
  accumulate_hopping<Real, 1, false, -1>(
      source_at_delta(local_source, x, y, z, t2, 1, 1, target),
      local_gauge(target, 1, x, y, z, t2), result);
  accumulate_hopping<Real, 2, true, 1>(
      source_at_delta(local_source, x, y, z, t2, 2, -1, target),
      backward_gauge(x, y, z, t2, 2, target), result);
  accumulate_hopping<Real, 2, false, -1>(
      source_at_delta(local_source, x, y, z, t2, 2, 1, target),
      local_gauge(target, 2, x, y, z, t2), result);
  }
  if (part != 1) {
  accumulate_hopping<Real, 3, true, 1>(
      source_at_delta(local_source, x, y, z, t2, 3, -1, target),
      backward_gauge(x, y, z, t2, 3, target), result);
  accumulate_hopping<Real, 3, false, -1>(
      source_at_delta(local_source, x, y, z, t2, 3, 1, target),
      local_gauge(target, 3, x, y, z, t2), result);
  }
  const int64_t site = site_linear(x, y, z, t2);
  if (part == 2) {
    C existing[kVector];
    gather_vector(compact_view(output, site), existing);
    for (int i = 0; i < kVector; ++i) {
      existing[i] += result[i];
    }
    scatter_vector(existing, output, site);
  } else {
    scatter_vector(result, output, site);
  }
}

template <typename Real>
void LatticeSet<Real>::bare_dslash(Complex<Real> *output,
                                   const Complex<Real> *input, int target) {
  g_halo_pack_dagger = false;
  static const bool profile =
      std::getenv("KUPL_PROFILE_DSLASH") != nullptr;
  const int source_parity = 1 - target;
  if (!profile && use_t2_zblock_dslash() &&
      geometry_.grid[0] == 1 && geometry_.grid[1] == 1 &&
      geometry_.grid[2] == 1 && geometry_.grid[3] > 1 &&
      geometry_.local[3] == 2) {
    post_fermion_halo(input, source_parity);
    compute_dslash_t2_spatial_zblocks(input, output, target);
    if (!fermion_requests_.empty()) {
      MPI_Waitall(static_cast<int>(fermion_requests_.size()),
                  fermion_requests_.data(), MPI_STATUSES_IGNORE);
      fermion_requests_.clear();
    }
    compute_dslash_t2_time_zblocks(input, output, target);
    return;
  }
  if (!profile) {
    const bool split_t_boundary =
        geometry_.grid[0] == 1 && geometry_.grid[1] == 1 &&
        geometry_.grid[2] == 1 && !use_block_dslash(geometry_.half_sites) &&
        (geometry_.local[3] == 2 ||
         (geometry_.local[3] == 4 && use_ht2_tboundary_split()));
    if (split_t_boundary) {
      post_fermion_halo(input, source_parity);
      compute_dslash_boundary_part(input, output, target, 1);
      if (!fermion_requests_.empty()) {
        MPI_Waitall(static_cast<int>(fermion_requests_.size()),
                    fermion_requests_.data(), MPI_STATUSES_IGNORE);
        fermion_requests_.clear();
      }
      compute_dslash_boundary_part(input, output, target, 2);
      return;
    }
    post_fermion_halo(input, source_parity);
    compute_dslash_interior(input, output, target);
    if (!fermion_requests_.empty()) {
      MPI_Waitall(static_cast<int>(fermion_requests_.size()),
                  fermion_requests_.data(), MPI_STATUSES_IGNORE);
      fermion_requests_.clear();
    }
    compute_dslash_boundary(input, output, target);
    return;
  }
  const auto start = std::chrono::steady_clock::now();
  post_fermion_halo(input, source_parity);
  const auto halo_done = std::chrono::steady_clock::now();
  compute_dslash_interior(input, output, target);
  const auto interior_done = std::chrono::steady_clock::now();
  if (!fermion_requests_.empty()) {
    MPI_Waitall(static_cast<int>(fermion_requests_.size()),
                fermion_requests_.data(), MPI_STATUSES_IGNORE);
    fermion_requests_.clear();
  }
  const auto wait_done = std::chrono::steady_clock::now();
  compute_dslash_boundary(input, output, target);
  const auto boundary_done = std::chrono::steady_clock::now();
  const auto elapsed = [](const auto &begin, const auto &end) {
    return std::chrono::duration<double>(end - begin).count();
  };
  prof_dslash_halo_seconds_ += elapsed(start, halo_done);
  prof_dslash_interior_seconds_ += elapsed(halo_done, interior_done);
  prof_dslash_wait_seconds_ += elapsed(interior_done, wait_done);
  prof_dslash_boundary_seconds_ += elapsed(wait_done, boundary_done);
  ++prof_dslash_calls_;
}

template <typename Real>
void LatticeSet<Real>::bare_dslash_dagger(Complex<Real> *output,
                                          const Complex<Real> *input,
                                          int target) {
  g_halo_pack_dagger = true;
  const int source_parity = 1 - target;
  if (geometry_.grid[0] == 1 && geometry_.grid[1] == 1 &&
      geometry_.grid[2] == 1 && !use_block_dslash(geometry_.half_sites) &&
      (geometry_.local[3] == 2 ||
       (geometry_.local[3] == 4 && use_ht2_tboundary_split()))) {
    post_fermion_halo(input, source_parity);
    compute_dslash_dagger_boundary_part(input, output, target, 1);
    if (!fermion_requests_.empty()) {
      MPI_Waitall(static_cast<int>(fermion_requests_.size()),
                  fermion_requests_.data(), MPI_STATUSES_IGNORE);
      fermion_requests_.clear();
    }
    compute_dslash_dagger_boundary_part(input, output, target, 2);
    return;
  }
  post_fermion_halo(input, source_parity);
  compute_dslash_dagger_interior(input, output, target);
  if (!fermion_requests_.empty()) {
    MPI_Waitall(static_cast<int>(fermion_requests_.size()),
                fermion_requests_.data(), MPI_STATUSES_IGNORE);
    fermion_requests_.clear();
  }
  compute_dslash_dagger_boundary(input, output, target);
}

template <typename Real>
void LatticeSet<Real>::apply_schur(
    const Complex<Real> *input, Complex<Real> *output, int parity,
    const std::array<const Complex<Real> *, 2> &clover,
    const std::array<const Complex<Real> *, 2> &inverse,
    Complex<Real> *scratch_a, Complex<Real> *scratch_b) {
  const int other = 1 - parity;
  bare_dslash(scratch_a, input, other);
  apply_clover_onsite(
      scratch_a, inverse[static_cast<std::size_t>(other)], scratch_b);
  bare_dslash(scratch_a, scratch_b, parity);
  const Real k = kappa();
  apply_clover_onsite(
      input, clover[static_cast<std::size_t>(parity)], output, scratch_a,
      k * k);
}

template <typename Real>
void LatticeSet<Real>::apply_wilson_schur(const Complex<Real> *input,
                                          Complex<Real> *output, int parity,
                                          Complex<Real> *scratch_a,
                                          Complex<Real> *scratch_b) {
  const int other = 1 - parity;
  bare_dslash(scratch_a, input, other);
  bare_dslash(scratch_b, scratch_a, parity);
  const Real coefficient = kappa() * kappa();
  const int64_t vectors = geometry_.parity_vectors;
  const int64_t chunk = omp_chunk<Real>(vectors);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < vectors; ++index) {
    output[index] = input[index] - coefficient * scratch_b[index];
  }
}

template <typename Real>
void LatticeSet<Real>::apply_wilson_schur_dagger(
    const Complex<Real> *input, Complex<Real> *output, int parity,
    Complex<Real> *scratch_a, Complex<Real> *scratch_b) {
  const int other = 1 - parity;
  bare_dslash_dagger(scratch_a, input, other);
  bare_dslash_dagger(scratch_b, scratch_a, parity);
  const Real coefficient = kappa() * kappa();
  const int64_t vectors = geometry_.parity_vectors;
  const int64_t chunk = omp_chunk<Real>(vectors);
#pragma omp parallel for schedule(static, chunk)
  for (int64_t index = 0; index < vectors; ++index) {
    output[index] = input[index] - coefficient * scratch_b[index];
  }
}

template <typename Real>
void LatticeSet<Real>::apply_wilson_cg_dslash(
    const Complex<Real> *input, Complex<Real> *output, int parity) {
  apply_wilson_schur(input, work_a_.data(), parity, work_b_.data(),
                     work_c_.data());
  apply_wilson_schur_dagger(work_a_.data(), output, parity, work_b_.data(),
                            work_c_.data());
}

template <typename Real>
void LatticeSet<Real>::apply_full_wilson_operator(
    const Complex<Real> *input, Complex<Real> *output,
    Complex<Real> *scratch) {
  const int64_t vectors = geometry_.parity_vectors;
  const int64_t chunk = omp_chunk<Real>(vectors);
  for (int parity = 0; parity < 2; ++parity) {
    const int other = 1 - parity;
    const Complex<Real> *source =
        input + static_cast<std::size_t>(other) * vectors;
    Complex<Real> *destination =
        output + static_cast<std::size_t>(parity) * vectors;
    bare_dslash(scratch, source, parity);
#pragma omp parallel for schedule(static, chunk)
    for (int64_t index = 0; index < vectors; ++index) {
      destination[index] =
          input[static_cast<std::size_t>(parity) * vectors + index] -
          kappa() * scratch[index];
    }
  }
}

template class LatticeSet<float>;
template class LatticeSet<double>;

} // namespace kupl::qcu
