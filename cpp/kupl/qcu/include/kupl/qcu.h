#ifndef KUPL_QCU_H
#define KUPL_QCU_H

#if defined(_WIN32)
#define KUPL_QCU_API __declspec(dllexport)
#else
#define KUPL_QCU_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

KUPL_QCU_API void applyInitQcu(long long _set_ptrs, long long _params,
                               long long _argv);
KUPL_QCU_API void applyEndQcu(long long _set_ptrs, long long _params);

KUPL_QCU_API void applyWilsonDslashQcu(long long _fermion_out,
                                       long long _fermion_in, long long _gauge,
                                       long long _set_ptrs, long long _params);
KUPL_QCU_API void applyWilsonBistabCgQcu(
    long long _fermion_out, long long _fermion_in, long long _gauge,
    long long _set_ptrs, long long _params);
KUPL_QCU_API void applyWilsonBistabCgDslashQcu(
    long long _fermion_out, long long _fermion_in, long long _gauge,
    long long _set_ptrs, long long _params);
KUPL_QCU_API void applyWilsonCgQcu(long long _fermion_out,
                                  long long _fermion_in, long long _gauge,
                                  long long _set_ptrs, long long _params);
KUPL_QCU_API void applyWilsonCgDslashQcu(long long _fermion_out,
                                        long long _fermion_in, long long _gauge,
                                        long long _set_ptrs, long long _params);
KUPL_QCU_API void testWilsonDslashQcu(long long _fermion_out,
                                      long long _fermion_in, long long _gauge,
                                      long long _set_ptrs, long long _params);
KUPL_QCU_API void applyCloverDslashQcu(long long _fermion_out,
                                       long long _fermion_in, long long _gauge,
                                       long long _set_ptrs, long long _params);
KUPL_QCU_API void testCloverDslashQcu(long long _fermion_out,
                                      long long _fermion_in, long long _gauge,
                                      long long _set_ptrs, long long _params);

KUPL_QCU_API void applyCloverQcu(long long _clover, long long _gauge,
                                 long long _set_ptrs, long long _params);
KUPL_QCU_API void applyCloversQcu(long long _clover, long long _clover_inv,
                                  long long _gauge, long long _set_ptrs,
                                  long long _params);
KUPL_QCU_API void applyDslashQcu(long long _fermion_out, long long _fermion_in,
                                 long long _gauge, long long _clover,
                                 long long _set_ptrs, long long _params);
KUPL_QCU_API void applyLaplacianQcu(long long _laplacian_out,
                                   long long _laplacian_in, long long _gauge,
                                   long long _set_ptrs, long long _params);

KUPL_QCU_API void applyCloverBistabCgQcu(
    long long _fermion_out, long long _fermion_in, long long _gauge,
    long long _clover_even, long long _clover_odd,
    long long _clover_even_inv, long long _clover_odd_inv, long long _set_ptrs,
    long long _params);
KUPL_QCU_API int getCloverBistabCgIterationsQcu(long long _set_ptrs,
                                                long long _params);
KUPL_QCU_API void applyCloverBistabCgDslashQcu(
    long long _fermion_out, long long _fermion_in, long long _gauge,
    long long _clover_even, long long _clover_odd,
    long long _clover_even_inv, long long _clover_odd_inv, long long _set_ptrs,
    long long _params);
KUPL_QCU_API int applyCloverBistabCgPrepareQcu(
    long long _compact_rhs, long long _full_rhs, long long _gauge,
    long long _clover_even, long long _clover_odd,
    long long _clover_even_inv, long long _clover_odd_inv,
    long long _set_ptrs, long long _params);
KUPL_QCU_API int applyCloverBistabCgReconstructQcu(
    long long _full_out, long long _full_rhs, long long _target_solution,
    long long _gauge, long long _clover_even, long long _clover_odd,
    long long _clover_even_inv, long long _clover_odd_inv,
    long long _set_ptrs, long long _params);

KUPL_QCU_API void applyGaussGaugeQcu(long long _gauge, long long _set_ptrs,
                                    long long _params);
KUPL_QCU_API void applyMultigridRestrictQcu(
    long long _coarse_out, long long _fine_in, long long _null_vecs,
    long long _set_ptrs, long long _params);
KUPL_QCU_API void applyMultigridProLongQcu(
    long long _fine_out, long long _coarse_in, long long _null_vecs,
    long long _set_ptrs, long long _params);
KUPL_QCU_API void applyMultigridCoarseDslashQcu(
    long long _fermion_out, long long _fermion_in, long long _hopping,
    long long _sitting, long long _set_ptrs, long long _params);
KUPL_QCU_API void applyMultigridCoarseDslashWideQcu(
    long long _fermion_out, long long _fermion_in, long long _sitting,
    long long _hop_nn, long long _hop_diag, long long _set_ptrs,
    long long _params);
KUPL_QCU_API int applyMultigridLocalOrthogonalizeQcu(
    long long _flat_in, long long _blocked_out, int E, int e, int Xf, int Yf,
    int Zf, int Tf, int Xc, int Yc, int Zc, int Tc, int data_type);
KUPL_QCU_API int applyMultigridBuildNullVectorsQcu(
    long long _blocked_out, long long _gauge, long long _clover_even,
    long long _clover_odd, long long _clover_even_inv,
    long long _clover_odd_inv, long long _set_ptrs, long long _params, int E,
    int e, int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc,
    int nv_iters, unsigned long long seed, double nv_tol);
KUPL_QCU_API int applyMultigridBuildStencilLocalQcu(
    long long _sitting, long long _hop_nn, long long _hop_diag,
    long long _null_vectors, long long _gauge, long long _clover_even,
    long long _clover_odd, long long _clover_even_inv,
    long long _clover_odd_inv, long long _set_ptrs, long long _params, int E,
    int e, int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc,
    int window, int batch_size);
KUPL_QCU_API int applyMultigridStrictCoarseQcu(
    long long _fermion_out, long long _fermion_in, long long _links,
    long long _onsite_pair, long long _set_ptrs, long long _params,
    int E, int X, int Y, int Z, int T, int onsite_index);
KUPL_QCU_API int applyMultigridStrictMatPCQcu(
    long long _fermion_out, long long _fermion_in, long long _links,
    long long _scratch, long long _set_ptrs, long long _params,
    int E, int X, int Y, int Z, int T, int parity);
KUPL_QCU_API int applyMultigridStrictFineMatPCQcu(
    long long _fermion_out, long long _fermion_in, long long _gauge,
    long long _clover_even, long long _clover_odd,
    long long _clover_even_inv, long long _clover_odd_inv,
    long long _set_ptrs, long long _params, int parity);
KUPL_QCU_API int applyMultigridStrictPrepareQcu(
    long long _fermion_out, long long _full_rhs, long long _links,
    long long _onsite_pair, long long _scratch, long long _set_ptrs,
    long long _params, int E, int X, int Y, int Z, int T, int parity);
KUPL_QCU_API int applyMultigridStrictReconstructQcu(
    long long _full_out, long long _full_rhs, long long _target_solution,
    long long _links, long long _onsite_pair, long long _scratch,
    long long _set_ptrs, long long _params,
    int E, int X, int Y, int Z, int T, int parity);
KUPL_QCU_API int applyMultigridStrictRestrictQcu(
    long long _coarse_out, long long _fine_in, long long _null_vectors,
    long long _set_ptrs, long long _params, int E, int e,
    int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc,
    int parity);
KUPL_QCU_API int applyMultigridStrictProLongQcu(
    long long _fine_out, long long _coarse_in, long long _null_vectors,
    long long _set_ptrs, long long _params, int E, int e,
    int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc,
    int parity);
KUPL_QCU_API int applyMultigridStrictVCycleQcu(
    long long _full_out, long long _full_rhs, long long _set_ptrs,
    long long _params, int start_level,
    unsigned long long *_allocated_bytes);
KUPL_QCU_API int applyMultigridStrictInitQcu(
    long long _set_ptrs, long long _params, int start_level,
    unsigned long long *_allocated_bytes);
KUPL_QCU_API int applyMultigridStrictEndQcu(long long _set_ptrs,
                                            long long _params);
KUPL_QCU_API int applyMultigridStrictFgmresQcu(
    long long _full_out, long long _full_rhs, long long _gauge,
    long long _clover_even, long long _clover_odd,
    long long _clover_even_inv, long long _clover_odd_inv,
    long long _fine_null_vectors, long long _set_ptrs, long long _params,
    int fine_E, int fine_X, int fine_Y, int fine_Z, int fine_T,
    int coarse_E, int coarse_X, int coarse_Y, int coarse_Z, int coarse_T,
    int element_bytes, int restart, int max_iter, double tolerance,
    int nu_pre, int nu_post, unsigned long long max_workspace_bytes,
    int *_iterations, int *_converged, double *_final_true_residual,
    unsigned long long *_allocated_bytes);
KUPL_QCU_API void applyCloverMultigridQcu(
    long long _fermion_out, long long _fermion_in, long long _gauge,
    long long _clover_even, long long _clover_odd,
    long long _clover_even_inv, long long _clover_odd_inv,
    long long _set_ptrs, long long _params);
KUPL_QCU_API int verifyCloverMultigridQcu(
    long long _fermion_out, long long _fermion_in, long long _gauge,
    long long _clover_even, long long _clover_odd,
    long long _clover_even_inv, long long _clover_odd_inv,
    long long _set_ptrs, long long _params);

#ifdef __cplusplus
}
#endif

#endif
