cdef extern from "pyqcu.h":
    void testWilsonDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _params, long long _argv)
    void applyWilsonDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _params, long long _argv)
    void testCloverDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _params, long long _argv)
    void applyCloverDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _params, long long _argv)
    void applyWilsonBistabCgQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _params, long long _argv)
    void applyWilsonBistabCgDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _params, long long _argv)
    void applyWilsonCgQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _params, long long _argv)
    void applyWilsonGmresIrQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _params, long long _argv);