#ifndef KUPL_QCU_LATTICE_CLOVER_MULTIGRID_H
#define KUPL_QCU_LATTICE_CLOVER_MULTIGRID_H

#include <cstdint>

namespace kupl::qcu {

class LatticeSetBase;

template <typename Real>
void clover_multigrid_solve(
    LatticeSetBase &set, void *fermion_out, const void *fermion_in,
    const void *gauge, const void *clover_even, const void *clover_odd,
    const void *clover_even_inv, const void *clover_odd_inv, void *set_ptrs,
    const int32_t *params);

template <typename Real>
int clover_multigrid_verify(
    LatticeSetBase &set, void *fermion_out, const void *fermion_in,
    const void *gauge, const void *clover_even, const void *clover_odd,
    const void *clover_even_inv, const void *clover_odd_inv, void *set_ptrs,
    const int32_t *params);

} // namespace kupl::qcu

#endif
