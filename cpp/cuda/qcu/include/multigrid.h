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
/**
 * @brief Mixed-precision restriction/prolongation.
 *
 * The null vectors live in the child-level precision.  Restriction casts the
 * parent input into that precision before accumulating; prolongation casts
 * the child result back to the parent precision.  Keeping the casts in the
 * kernel prevents a void* coarse buffer from ever being interpreted with the
 * wrong element size.
 */
template <typename Out, typename In>
__global__ void multigrid_restrict_cast(void *coarse_out, void *fine_in,
                                        void *null_vecs, int E, int e, int Xf,
                                        int Yf, int Zf, int Tf, int Xc, int Yc,
                                        int Zc, int Tc);
template <typename Out, typename In>
__global__ void multigrid_prolong_cast(void *fine_out, void *coarse_in,
                                       void *null_vecs, int E, int e, int Xf,
                                       int Yf, int Zf, int Tf, int Xc, int Yc,
                                       int Zc, int Tc);
template <typename T>
__global__ void multigrid_coarse_dslash(void *fermion_out, void *fermion_in,
                                         void *hopping, void *sitting,
                                         int E, int X, int Y, int Z, int Lt);
/**
 * @brief Wide-stencil coarse dslash for the Schur-consistent coarse operator
 *        A_c = P^T S P (on-site + nearest-neighbour + diagonal couplings).
 * @see multigrid.cu for the full stencil documentation.
 */
template <typename T>
__global__ void multigrid_coarse_dslash_wide(void *fermion_out, void *fermion_in,
                                             void *sitting, void *hop_nn,
                                             void *hop_diag, int E, int X, int Y,
                                             int Z, int Lt);
/**
 * @brief Wide stencil dslash using a padded local coarse-grid halo.
 *
 * ``halo`` has layout [E, X+2, Y+2, Z+2, T+2].  Interior neighbours are read
 * from ``fermion_in``; only coordinates outside the local block are read from
 * the halo.  This is the distributed counterpart of
 * multigrid_coarse_dslash_wide.
 */
template <typename T>
__global__ void multigrid_coarse_dslash_wide_halo(
    void *fermion_out, void *fermion_in, void *halo, void *sitting,
    void *hop_nn, void *hop_diag, int E, int X, int Y, int Z, int Lt);

/**
 * @brief Pack all 32 one-hop coarse-grid halo faces/edges/corners on device.
 *
 * ``packed`` is laid out as [direction, E, free-face-site] with a fixed
 * per-direction stride ``max_face``.  The direction table is the 16 canonical
 * axial/two-axis shifts and their opposites.  MPI sees two real scalars per
 * LatticeComplex element, while this kernel works in complex elements.
 */
template <typename T>
__global__ void multigrid_coarse_pack_halo(
    void *packed, const void *fermion_in, int E, int X, int Y, int Z, int Lt,
    int max_face);

/**
 * @brief Unpack the received 32 direction buffers into the padded halo.
 */
template <typename T>
__global__ void multigrid_coarse_unpack_halo(
    void *halo, const void *packed, int E, int X, int Y, int Z, int Lt,
    int max_face);

/**
 * @brief Apply the distributed wide stencil to either the interior or the
 *        boundary.  ``boundary_only=0`` computes sites whose complete stencil
 *        is local; ``boundary_only=1`` computes the complement.
 */
template <typename T>
__global__ void multigrid_coarse_dslash_wide_halo_region(
    void *fermion_out, void *fermion_in, void *halo, void *sitting,
    void *hop_nn, void *hop_diag, int E, int X, int Y, int Z, int Lt,
    int boundary_only);

#if defined(QCU_HAVE_NVSHMEM)
/**
 * @brief GPU-initiated put of the 32 packed coarse-grid halo directions.
 *
 * ``device_recv`` is a symmetric NVSHMEM allocation.  Direction h is written
 * to the receive slot on PE peer(h^1), so that the remote unpack kernel can
 * use the same direction-indexed layout as the MPI path.
 */
template <typename T>
__global__ void multigrid_nvshmem_put_halo(
    void *device_recv, const void *device_send, int E, int X, int Y, int Z,
    int Lt, int max_face, int grid_x, int grid_y, int grid_z, int grid_t,
    int coord_x, int coord_y, int coord_z, int coord_t, int rank);
#endif

/**
 * @brief Deterministic generic test-vector and subtraction kernels used by
 *        the C++ multigrid verification path.
 */
template <typename T>
__global__ void multigrid_fill_test_vector(void *out, int n, unsigned long seed);
template <typename T>
__global__ void multigrid_difference(void *out, const void *a, const void *b,
                                     int n);
template <typename T>
__global__ void multigrid_extract_null_vector(void *out, const void *null_vecs,
                                              int vector_index, int E, int e,
                                              int Xf, int Yf, int Zf, int Tf,
                                              int Xc, int Yc, int Zc, int Tc);
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
/**
 * @brief FUSED coarse-level BiStabCG solver (single kernel launch).
 * @see multigrid.cu for full documentation.
 */
template <typename T>
__global__ void multigrid_coarse_solve(void *x, void *rhs, void *r_tilde,
                                       void *r, void *p, void *v, void *s,
                                       void *t, void *sitting, void *hop_nn,
                                       void *hop_diag, int E, int X, int Y,
                                       int Z, int Lt, int max_iter, T tol);
/**
 * @brief Cooperative-groups PARALLEL fused coarse BiStabCG solve.
 * @see multigrid.cu for full documentation.
 */
template <typename T, int NT>
__global__ void multigrid_coarse_solve_cg(void *x, void *rhs, void *r_tilde,
                                          void *r, void *p, void *v, void *s,
                                          void *t, void *sitting, void *hop_nn,
                                          void *hop_diag, int E, int X, int Y,
                                          int Z, int Lt, int max_iter, T tol,
                                          void *partials, void *breakdown);
} // namespace qcu
#endif
