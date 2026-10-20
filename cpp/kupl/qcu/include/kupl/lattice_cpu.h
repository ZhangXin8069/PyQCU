#ifndef KUPL_QCU_LATTICE_CPU_H
#define KUPL_QCU_LATTICE_CPU_H

#include "kupl/lattice_complex.h"

#include <algorithm>
#include <cstdint>

#include <omp.h>

namespace kupl::qcu {

void initialize_mpi();
int configured_threads();
void bind_logical_numa();

template <typename Real> int64_t omp_chunk(int64_t count) {
  const int threads = std::max(1, omp_get_max_threads());
  const int64_t alignment = std::max<int64_t>(1, 64 / sizeof(Complex<Real>));
  int64_t chunk = std::max<int64_t>(alignment, count / threads);
  chunk = ((chunk + alignment - 1) / alignment) * alignment;
  return std::min(count, std::max<int64_t>(1, chunk));
}

} // namespace kupl::qcu

#endif
