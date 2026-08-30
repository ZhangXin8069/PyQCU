#ifndef _PYQCU_H
#define _PYQCU_H
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
void applyInitQcu(long long _set_ptrs, long long _params, long long _argv);
void applyEndQcu(long long _set_ptrs, long long _params);
void testWilsonDslashQcu(long long _fermion_out, long long _fermion_in,
                         long long _gauge, long long _set_ptrs,
                         long long _params);
void applyWilsonDslashQcu(long long _fermion_out, long long _fermion_in,
                          long long _gauge, long long _set_ptrs,
                          long long _params);
void testCloverDslashQcu(long long _fermion_out, long long _fermion_in,
                         long long _gauge, long long _set_ptrs,
                         long long _params);
void applyCloverDslashQcu(long long _fermion_out, long long _fermion_in,
                          long long _gauge, long long _set_ptrs,
                          long long _params);
void applyWilsonBistabCgQcu(long long _fermion_out, long long _fermion_in,
                            long long _gauge, long long _set_ptrs,
                            long long _params);
void applyWilsonBistabCgDslashQcu(long long _fermion_out, long long _fermion_in,
                                  long long _gauge, long long _set_ptrs,
                                  long long _params);
void applyWilsonCgQcu(long long _fermion_out, long long _fermion_in,
                      long long _gauge, long long _set_ptrs, long long _params);
void applyWilsonCgDslashQcu(long long _fermion_out, long long _fermion_in,
                            long long _gauge, long long _set_ptrs,
                            long long _params);
void applyLaplacianQcu(long long _laplacian_out, long long _laplacian_in,
                       long long _gauge, long long _set_ptrs,
                       long long _params);
void applyCloverQcu(long long _clover, long long _gauge, long long _set_ptrs,
                    long long _params);
void applyCloversQcu(long long _clover, long long _clover_inv, long long _gauge,
                     long long _set_ptrs, long long _params);
void applyDslashQcu(long long _fermion_out, long long _fermion_in,
                    long long _gauge, long long _clover, long long _set_ptrs,
                    long long _params);
void applyGaussGaugeQcu(long long _gauge, long long _set_ptrs,
                        long long _params);
void applyCloverBistabCgQcu(long long _fermion_out, long long _fermion_in,
                            long long _gauge, long long _clover_ee,
                            long long _clover_oo, long long _clover_ee_inv,
                            long long _clover_oo_inv, long long _set_ptrs,
                            long long _params);
void applyCloverBistabCgDslashQcu(long long _fermion_out, long long _fermion_in,
                                  long long _gauge, long long _clover_ee,
                                  long long _clover_oo,
                                  long long _clover_ee_inv,
                                  long long _clover_oo_inv, long long _set_ptrs,
                                  long long _params);
int applyCloverBistabCgPrepareQcu(
    long long _compact_rhs, long long _full_rhs, long long _gauge,
    long long _clover_ee, long long _clover_oo, long long _clover_ee_inv,
    long long _clover_oo_inv, long long _set_ptrs, long long _params);
int applyCloverBistabCgReconstructQcu(
    long long _full_out, long long _full_rhs, long long _target_odd,
    long long _gauge, long long _clover_ee, long long _clover_oo,
    long long _clover_ee_inv, long long _clover_oo_inv,
    long long _set_ptrs, long long _params);
void applyMultigridRestrictQcu(long long _coarse_out, long long _fine_in,
                                long long _null_vecs, long long _set_ptrs,
                                long long _params);
void applyMultigridProLongQcu(long long _fine_out, long long _coarse_in,
                               long long _null_vecs, long long _set_ptrs,
                               long long _params);
void applyMultigridCoarseDslashQcu(long long _fermion_out, long long _fermion_in,
                                    long long _hopping, long long _sitting,
                                    long long _set_ptrs, long long _params);
void applyMultigridCoarseDslashWideQcu(long long _fermion_out,
                                       long long _fermion_in,
                                       long long _sitting, long long _hop_nn,
                                       long long _hop_diag, long long _set_ptrs,
                                       long long _params);
int applyMultigridStrictCoarseQcu(
    long long _fermion_out, long long _fermion_in, long long _links,
    long long _onsite_pair, long long _set_ptrs, long long _params,
    int E, int X, int Y, int Z, int T, int onsite_index);
int applyMultigridStrictMatPCQcu(
    long long _fermion_out, long long _fermion_in, long long _links,
    long long _scratch, long long _set_ptrs, long long _params,
    int E, int X, int Y, int Z, int T, int parity);
int applyMultigridStrictPrepareQcu(
    long long _fermion_out, long long _full_rhs, long long _links,
    long long _onsite_pair, long long _scratch, long long _set_ptrs,
    long long _params, int E, int X, int Y, int Z, int T, int parity);
int applyMultigridStrictReconstructQcu(
    long long _full_out, long long _full_rhs, long long _target_solution,
    long long _links, long long _onsite_pair, long long _scratch,
    long long _set_ptrs, long long _params,
    int E, int X, int Y, int Z, int T, int parity);
int applyMultigridStrictRestrictQcu(
    long long _coarse_out, long long _fine_in, long long _null_vectors,
    long long _set_ptrs, long long _params, int E, int e,
    int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc,
    int parity);
int applyMultigridStrictProLongQcu(
    long long _fine_out, long long _coarse_in, long long _null_vectors,
    long long _set_ptrs, long long _params, int E, int e,
    int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc,
    int parity);
int applyMultigridStrictVCycleQcu(
    long long _full_out, long long _full_rhs, long long _set_ptrs,
    long long _params, int start_level,
    unsigned long long *_allocated_bytes);
int applyMultigridStrictInitQcu(
    long long _set_ptrs, long long _params, int start_level,
    unsigned long long *_allocated_bytes);
int applyMultigridStrictEndQcu(long long _set_ptrs, long long _params);
void applyCloverMultigridQcu(long long _fermion_out, long long _fermion_in,
                              long long _gauge, long long _clover_ee,
                              long long _clover_oo,
                              long long _clover_ee_inv,
                              long long _clover_oo_inv,
                              long long _set_ptrs,
                              long long _params);
int verifyCloverMultigridQcu(
    long long _fermion_out, long long _fermion_in, long long _gauge,
    long long _clover_ee, long long _clover_oo, long long _clover_ee_inv,
    long long _clover_oo_inv, long long _set_ptrs, long long _params);
#ifdef __cplusplus
}
#endif
#endif
