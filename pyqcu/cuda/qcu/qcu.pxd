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
    void applyMultigridRestrictQcu(long long _coarse_out, long long _fine_in, long long _null_vecs, long long _set_ptrs, long long _params) nogil
    void applyMultigridProLongQcu(long long _fine_out, long long _coarse_in, long long _null_vecs, long long _set_ptrs, long long _params) nogil
    void applyMultigridCoarseDslashQcu(long long _fermion_out, long long _fermion_in, long long _hopping, long long _sitting, long long _set_ptrs, long long _params) nogil
    void applyCloverMultigridQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _clover_ee, long long _clover_oo, long long _clover_ee_inv, long long _clover_oo_inv, long long _set_ptrs, long long _params) nogil