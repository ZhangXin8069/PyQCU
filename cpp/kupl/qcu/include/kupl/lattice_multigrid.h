#ifndef KUPL_QCU_LATTICE_MULTIGRID_H
#define KUPL_QCU_LATTICE_MULTIGRID_H

#include "kupl/lattice_complex.h"

namespace kupl::qcu {

inline constexpr int kCoarseDirections = 32;

int coarse_shift_component(int direction, int dim);
int coarse_direction_for_shift(const int shift[kDim]);

} // namespace kupl::qcu

#endif
