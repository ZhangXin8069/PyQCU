#ifndef _MULTIGRID_H
#define _MULTIGRID_H
#include "./lattice_complex.h"
namespace qcu {
template <typename T>
__global__ void multigrid_restrict(void *coarse_out, void *fine_in,
                                   void *null_vecs, int E, int e, int Xf, int Yf,
                                   int Zf, int Tf, int Xc, int Yc, int Zc,
                                   int Tc);
template <typename T>
__global__ void multigrid_prolong(void *fine_out, void *coarse_in,
                                  void *null_vecs, int E, int e, int Xf, int Yf,
                                  int Zf, int Tf, int Xc, int Yc, int Zc,
                                  int Tc);
template <typename T>
__global__ void multigrid_coarse_dslash(void *fermion_out, void *fermion_in,
                                         void *hopping, void *sitting,
                                         int E, int X, int Y, int Z, int T);
} // namespace qcu
#endif
