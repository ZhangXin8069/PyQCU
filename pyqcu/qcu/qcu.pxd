cdef extern from "pyqcu.h":
    void testWilsonDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _params, long long _argv)
    void applyWilsonDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _params, long long _argv)
    void testCloverDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _params, long long _argv)
    void applyCloverDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _params, long long _argv)
    void applyBistabCgQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _params, long long _argv)
    void applyCgQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _params, long long _argv)
    void applyGmresIrQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _params, long long _argv);