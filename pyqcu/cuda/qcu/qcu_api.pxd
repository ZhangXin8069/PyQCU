cdef extern from "pyqcu.h":
    void applyInitQcu(long long _set_ptrs, long long _params, long long _argv) nogil
    void applyEndQcu(long long _set_ptrs, long long _params) nogil
    void testWilsonDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _set_ptrs, long long _params) nogil
    void applyWilsonDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _set_ptrs, long long _params) nogil
    void testCloverDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _set_ptrs, long long _params) nogil
    void applyCloverDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _set_ptrs, long long _params) nogil
    void applyWilsonBistabCgQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _set_ptrs, long long _params) nogil
    void applyWilsonBistabCgDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _set_ptrs, long long _params) nogil
    void applyWilsonCgQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _set_ptrs, long long _params) nogil
    void applyWilsonCgDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _set_ptrs, long long _params) nogil
    void applyLaplacianQcu(long long _laplacian_out, long long _laplacian_in, long long _gauge, long long _set_ptrs, long long _params) nogil
    void applyCloverQcu(long long _clover, long long _gauge, long long _set_ptrs, long long _params) nogil
    void applyCloversQcu(long long _clover, long long _clover_inv, long long _gauge, long long _set_ptrs, long long _params) nogil
    void applyDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _clover, long long _set_ptrs, long long _params) nogil
    void applyGaussGaugeQcu(long long _gauge, long long _set_ptrs, long long _params) nogil
    void applyCloverBistabCgQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _clover_ee, long long _clover_oo, long long _clover_ee_inv, long long _clover_oo_inv, long long _set_ptrs, long long _params) nogil
    void applyCloverBistabCgDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _clover_ee, long long _clover_oo, long long _clover_ee_inv, long long _clover_oo_inv, long long _set_ptrs, long long _params) nogil
    int applyCloverBistabCgPrepareQcu(long long _compact_rhs, long long _full_rhs, long long _gauge, long long _clover_ee, long long _clover_oo, long long _clover_ee_inv, long long _clover_oo_inv, long long _set_ptrs, long long _params) nogil
    int applyCloverBistabCgReconstructQcu(long long _full_out, long long _full_rhs, long long _target_odd, long long _gauge, long long _clover_ee, long long _clover_oo, long long _clover_ee_inv, long long _clover_oo_inv, long long _set_ptrs, long long _params) nogil
    void applyMultigridRestrictQcu(long long _coarse_out, long long _fine_in, long long _null_vecs, long long _set_ptrs, long long _params) nogil
    void applyMultigridProLongQcu(long long _fine_out, long long _coarse_in, long long _null_vecs, long long _set_ptrs, long long _params) nogil
    void applyMultigridCoarseDslashQcu(long long _fermion_out, long long _fermion_in, long long _hopping, long long _sitting, long long _set_ptrs, long long _params) nogil
    void applyMultigridCoarseDslashWideQcu(long long _fermion_out, long long _fermion_in, long long _sitting, long long _hop_nn, long long _hop_diag, long long _set_ptrs, long long _params) nogil
    int applyMultigridStrictCoarseQcu(long long _fermion_out, long long _fermion_in, long long _links, long long _onsite_pair, long long _set_ptrs, long long _params, int E, int X, int Y, int Z, int T, int onsite_index) nogil
    int applyMultigridStrictMatPCQcu(long long _fermion_out, long long _fermion_in, long long _links, long long _scratch, long long _set_ptrs, long long _params, int E, int X, int Y, int Z, int T, int parity) nogil
    int applyMultigridStrictFineMatPCQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _clover_ee, long long _clover_oo, long long _clover_ee_inv, long long _clover_oo_inv, long long _set_ptrs, long long _params, int parity) nogil
    int applyMultigridStrictPrepareQcu(long long _fermion_out, long long _full_rhs, long long _links, long long _onsite_pair, long long _scratch, long long _set_ptrs, long long _params, int E, int X, int Y, int Z, int T, int parity) nogil
    int applyMultigridStrictReconstructQcu(long long _full_out, long long _full_rhs, long long _target_solution, long long _links, long long _onsite_pair, long long _scratch, long long _set_ptrs, long long _params, int E, int X, int Y, int Z, int T, int parity) nogil
    int applyMultigridStrictRestrictQcu(long long _coarse_out, long long _fine_in, long long _null_vectors, long long _set_ptrs, long long _params, int E, int e, int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc, int parity) nogil
    int applyMultigridStrictProLongQcu(long long _fine_out, long long _coarse_in, long long _null_vectors, long long _set_ptrs, long long _params, int E, int e, int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc, int parity) nogil
    int applyMultigridStrictVCycleQcu(long long _full_out, long long _full_rhs, long long _set_ptrs, long long _params, int start_level, unsigned long long *_allocated_bytes) nogil
    int applyMultigridStrictInitQcu(long long _set_ptrs, long long _params, int start_level, unsigned long long *_allocated_bytes) nogil
    int applyMultigridStrictEndQcu(long long _set_ptrs, long long _params) nogil
    int applyMultigridStrictFgmresQcu(long long _full_out, long long _full_rhs, long long _gauge, long long _clover_ee, long long _clover_oo, long long _clover_ee_inv, long long _clover_oo_inv, long long _fine_null_vectors, long long _set_ptrs, long long _params, int fine_E, int fine_X, int fine_Y, int fine_Z, int fine_T, int coarse_E, int coarse_X, int coarse_Y, int coarse_Z, int coarse_T, int element_bytes, int restart, int max_iter, double tolerance, int nu_pre, int nu_post, unsigned long long max_workspace_bytes, int *_iterations, int *_converged, double *_final_true_residual, unsigned long long *_allocated_bytes) nogil
    void applyCloverMultigridQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _clover_ee, long long _clover_oo, long long _clover_ee_inv, long long _clover_oo_inv, long long _set_ptrs, long long _params) nogil
    int verifyCloverMultigridQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _clover_ee, long long _clover_oo, long long _clover_ee_inv, long long _clover_oo_inv, long long _set_ptrs, long long _params) nogil
