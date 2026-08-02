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
                                         int E, int X, int Y, int Z, int Lt);
/**
 * @brief Convert parity-split odd-site data to full-site layout (odd t-slices only).
 *
 * Converts a parity-split odd buffer [sc, X, Y, Z, Lt/2] to a full-site buffer
 * [sc, X, Y, Z, Lt] by interleaving: odd t-slices (t=1,3,5,...,Lt-1) in the
 * full-site output are filled from the parity-split source. Even t-slices (0,2,4,...)
 * are NOT written by this kernel — the caller must zero-initialize the destination
 * buffer before calling this kernel.
 *
 * @param full_out Full-site output buffer [sc, X, Y, Z, Lt] (must be pre-zeroed)
 * @param odd_in   Parity-split odd-source [sc, X, Y, Z, Lt/2]
 * @param sc       Spin×color degrees of freedom (typically 12 for 4×3)
 * @param X,Y,Z    Spatial lattice dimensions (same for both layouts)
 * @param Lt_full  Full t-dimension (Lt in full-site; parity-split has Lt/2)
 */
template <typename T>
__global__ void multigrid_odd_to_full(void *full_out, void *odd_in, int sc,
                                       int X, int Y, int Z, int Lt_full);
template <typename T>
__global__ void multigrid_even_to_full(void *full_out, void *even_in, int sc,
                                       int X, int Y, int Z, int Lt_full);
/**
 * @brief Extract parity-split odd-site data from a full-site buffer.
 *
 * Reads the odd t-slices (t=1,3,5,...,Lt-1) from a full-site buffer
 * [sc, X, Y, Z, Lt] and writes them contiguously to a parity-split odd buffer
 * [sc, X, Y, Z, Lt/2].
 *
 * @param odd_out  Parity-split odd output [sc, X, Y, Z, Lt/2]
 * @param full_in  Full-site input buffer [sc, X, Y, Z, Lt]
 * @param sc       Spin×color degrees of freedom
 * @param X,Y,Z    Spatial lattice dimensions
 * @param Lt_full  Full t-dimension
 */
template <typename T>
__global__ void multigrid_full_to_odd(void *odd_out, void *full_in, int sc,
                                       int X, int Y, int Z, int Lt_full);
template <typename T>
__global__ void multigrid_full_to_even(void *even_out, void *full_in, int sc,
                                       int X, int Y, int Z, int Lt_full);
} // namespace qcu
#endif
