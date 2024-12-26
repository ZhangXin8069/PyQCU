cdef extern from "pyqcu.h":
    void testWilsonDslashQcu(void *fermion_out, void *fermion_in, void *gauge, void *params, void *argv)
    void applyWilsonDslashQcu(void *fermion_out, void *fermion_in, void *gauge, void *params, void *argv)
    void testCloverDslashQcu(void *fermion_out, void *fermion_in, void *gauge, void *params, void *argv)
    void applyCloverDslashQcu(void *fermion_out, void *fermion_in, void *gauge, void *params, void *argv)
    void applyBistabCgQcu(void *fermion_out, void *fermion_in, void *gauge, void *params, void *argv)
    void applyCgQcu(void *fermion_out, void *fermion_in, void *gauge, void *params, void *argv)
    void applyGmresIrQcu(void *fermion_out, void *fermion_in, void *gauge, void *params, void *argv);