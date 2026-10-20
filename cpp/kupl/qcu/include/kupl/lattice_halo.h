#ifndef KUPL_QCU_LATTICE_HALO_H
#define KUPL_QCU_LATTICE_HALO_H

#include "kupl/lattice_complex.h"

namespace kupl::qcu {

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
