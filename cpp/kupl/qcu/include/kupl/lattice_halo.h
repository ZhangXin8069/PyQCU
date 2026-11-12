#ifndef KUPL_QCU_LATTICE_HALO_H
#define KUPL_QCU_LATTICE_HALO_H

#include "kupl/lattice_complex.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace kupl::qcu {

// Environment gate used by both the halo scratch allocator (backend.cpp) and
// the Wilson-dslash halo pack/unpack (lattice_wilson_dslash.cpp).
inline bool halo_env_flag(const char *name) {
  const char *text = std::getenv(name);
  return text != nullptr && *text != '\0' && std::strcmp(text, "0") != 0 &&
         std::strcmp(text, "false") != 0 && std::strcmp(text, "no") != 0 &&
         std::strcmp(text, "off") != 0;
}

// KUPL_HALO_PACK6: transmit only the 2 spin x 3 color half-spinor projection
// that a given direction's hop actually consumes, and for the t direction only
// the even-odd half of the face that the neighbour reads.
inline bool halo_pack6_enabled() {
  static const bool enabled = halo_env_flag("KUPL_HALO_PACK6");
  return enabled;
}

// KUPL_HALO_DEDUP_PACK: at ht == 1 the low and high t faces hold identical
// 12-component data, so pack once and post the same buffer twice.
inline bool halo_dedup_pack_enabled() {
  static const bool enabled = halo_env_flag("KUPL_HALO_DEDUP_PACK");
  return enabled;
}

// KUPL_T_LOCAL_IN_PART1: with local_t == 2 fold the local t hop into part 1.
inline bool t_local_part1_enabled() {
  static const bool enabled = halo_env_flag("KUPL_T_LOCAL_IN_PART1");
  return enabled;
}

// The even-odd half-face reduction is exact only when the t face is indexed by
// a spatial parity, which needs even x/y/z extents (the production 24^3 case).
inline bool halo_pack6_half_face(int dim, int lx, int ly, int lz) {
  return dim == 3 && ((lx | ly | lz) & 1) == 0;
}

inline int64_t halo_pack6_face_count(int dim, int64_t face_sites, int lx,
                                     int ly, int lz) {
  return halo_pack6_half_face(dim, lx, ly, lz) ? face_sites / 2 : face_sites;
}

// Rank of (x, y, z) among the sites whose (x + y + z) parity equals `parity`,
// in the x-major lexicographic order used by LatticeGeometry::face_index for
// dim == 3.  Requires even y/z extents (guaranteed by halo_pack6_half_face).
inline int64_t halo_t_half_rank(int x, int y, int z, int parity, int ly,
                                int lz) {
  const int q = (parity ^ (x & 1) ^ (y & 1)) & 1;
  const int64_t z_rank = q == 0 ? (z + 1) / 2 : z / 2;
  return static_cast<int64_t>(x) * ((static_cast<int64_t>(ly) * lz) / 2) +
         static_cast<int64_t>(y) * (lz / 2) + z_rank;
}

inline int plane_index(int dim0, int dim1) {
  static constexpr int planes[kDim][kDim] = {
      {-1, 0, 1, 2},
      {-1, -1, 3, 4},
      {-1, -1, -1, 5},
      {-1, -1, -1, -1},
  };
  return planes[dim0][dim1];
}

inline int sign_index(int sign) { return sign > 0 ? 1 : 0; }

inline int face_offset(int dim, int sign) {
  return dim * 2 + sign_index(sign);
}

inline int corner_tag(int dim0, int dim1, int sign0, int sign1) {
  return 3000 + plane_index(dim0, dim1) * 100 +
         (sign0 > 0 ? 10 : 0) + (sign1 > 0 ? 1 : 0);
}

inline int gauge_face_tag(int dim, int boundary_sign) {
  return 2000 + dim * 10 + (boundary_sign > 0 ? 2 : 1);
}

inline int fermion_face_tag(int dim, int boundary_sign) {
  return 4000 + dim * 10 + (boundary_sign > 0 ? 2 : 1);
}

inline int laplacian_face_tag(int kind, int dim, int boundary_sign) {
  return 5000 + kind * 100 + dim * 10 + (boundary_sign > 0 ? 2 : 1);
}

} // namespace kupl::qcu

#endif
