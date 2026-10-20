#ifndef KUPL_QCU_LATTICE_STRICT_H
#define KUPL_QCU_LATTICE_STRICT_H

#include <cstddef>
#include <cstdint>

namespace kupl::qcu {

class LatticeSetBase;

template <typename Real>
int strict_coarse(void *out, const void *in, const void *links,
                  const void *onsite_pair, const int32_t *params, int E,
                  int X, int Y, int Z, int T, int onsite_index);

template <typename Real>
int strict_matpc(void *out, const void *in, const void *links, void *scratch,
                 const int32_t *params, int E, int X, int Y, int Z, int T,
                 int parity);

template <typename Real>
int strict_prepare(void *out, const void *full_rhs, const void *links,
                   const void *onsite_pair, void *scratch,
                   const int32_t *params, int E, int X, int Y, int Z, int T,
                   int parity);

template <typename Real>
int strict_reconstruct(void *full_out, const void *full_rhs,
                       const void *target_solution, const void *links,
                       const void *onsite_pair, void *scratch,
                       const int32_t *params, int E, int X, int Y, int Z,
                       int T, int parity);

template <typename Real>
int strict_restrict(void *coarse_out, const void *fine_in,
                    const void *null_vectors, int E, int e, int Xf, int Yf,
                    int Zf, int Tf, int Xc, int Yc, int Zc, int Tc,
                    int parity);

template <typename Real>
int strict_prolong(void *fine_out, const void *coarse_in,
                   const void *null_vectors, int E, int e, int Xf, int Yf,
                   int Zf, int Tf, int Xc, int Yc, int Zc, int Tc,
                   int parity);

template <typename Real>
std::size_t strict_init_hierarchy(void *set_ptrs, const int32_t *params,
                                  int start_level);

template <typename Real>
std::size_t strict_vcycle(void *full_out, const void *full_rhs,
                          void *set_ptrs, const int32_t *params,
                          int start_level);

template <typename Real>
int strict_end_hierarchy(void *set_ptrs);

template <typename Real>
std::size_t strict_fgmres(
    void *full_out, const void *full_rhs, const void *gauge,
    const void *clover_even, const void *clover_odd,
    const void *clover_even_inv, const void *clover_odd_inv,
    const void *fine_null_vectors, void *set_ptrs, const int32_t *params,
    int fine_E, int fine_X, int fine_Y, int fine_Z, int fine_T, int coarse_E,
    int coarse_X, int coarse_Y, int coarse_Z, int coarse_T, int element_bytes,
    int restart, int max_iter, double tolerance, int nu_pre, int nu_post,
    std::size_t max_workspace_bytes, int &iterations, bool &converged,
    Real &final_true_residual);

} // namespace kupl::qcu

#endif
