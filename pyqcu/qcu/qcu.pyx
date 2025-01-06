cimport qcu
import numpy
cimport numpy
import ctypes
cdef long long set_ptrs
cdef long long fermion_out
cdef long long fermion_in
cdef long long gauge
cdef long long params
cdef long long argv
def applyInitQcu(_set_ptrs, _params, _argv):
    set_ptrs = _set_ptrs.ctypes.data
    params = _params.ctypes.data
    argv = _argv.ctypes.data
    qcu.applyInitQcu(set_ptrs, params, argv);
def applyEndQcu(_set_ptrs, _params):
    set_ptrs = _set_ptrs.ctypes.data
    params = _params.ctypes.data
    qcu.applyEndQcu(set_ptrs, params);
def testWilsonDslashQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    fermion_out = _fermion_out.data.ptr
    fermion_in = _fermion_in.data.ptr
    gauge = _gauge.data.ptr
    set_ptrs = _set_ptrs.ctypes.data
    params = _params.ctypes.data
    qcu.testWilsonDslashQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def applyWilsonDslashQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    fermion_out = _fermion_out.data.ptr
    fermion_in = _fermion_in.data.ptr
    gauge = _gauge.data.ptr
    set_ptrs = _set_ptrs.ctypes.data
    params = _params.ctypes.data
    qcu.applyWilsonDslashQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def testCloverDslashQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    fermion_out = _fermion_out.data.ptr
    fermion_in = _fermion_in.data.ptr
    gauge = _gauge.data.ptr
    set_ptrs = _set_ptrs.ctypes.data
    params = _params.ctypes.data
    qcu.testCloverDslashQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def applyCloverDslashQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    fermion_out = _fermion_out.data.ptr
    fermion_in = _fermion_in.data.ptr
    gauge = _gauge.data.ptr
    set_ptrs = _set_ptrs.ctypes.data
    params = _params.ctypes.data
    qcu.applyCloverDslashQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def applyWilsonBistabCgQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    fermion_out = _fermion_out.data.ptr
    fermion_in = _fermion_in.data.ptr
    gauge = _gauge.data.ptr
    set_ptrs = _set_ptrs.ctypes.data
    params = _params.ctypes.data
    qcu.applyWilsonBistabCgQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def applyWilsonBistabCgDslashQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    fermion_out = _fermion_out.data.ptr
    fermion_in = _fermion_in.data.ptr
    gauge = _gauge.data.ptr
    set_ptrs = _set_ptrs.ctypes.data
    params = _params.ctypes.data
    qcu.applyWilsonBistabCgDslashQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def applyWilsonCgQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    fermion_out = _fermion_out.data.ptr
    fermion_in = _fermion_in.data.ptr
    gauge = _gauge.data.ptr
    set_ptrs = _set_ptrs.ctypes.data
    params = _params.ctypes.data
    qcu.applyWilsonCgQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def applyWilsonCgDslashQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    fermion_out = _fermion_out.data.ptr
    fermion_in = _fermion_in.data.ptr
    gauge = _gauge.data.ptr
    set_ptrs = _set_ptrs.ctypes.data
    params = _params.ctypes.data
    qcu.applyWilsonCgDslashQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def applyWilsonGmresIrQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    fermion_out = _fermion_out.data.ptr
    fermion_in = _fermion_in.data.ptr
    gauge = _gauge.data.ptr
    set_ptrs = _set_ptrs.ctypes.data
    params = _params.ctypes.data
    qcu.applyWilsonGmresIrQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def applyLaplacianQcu(_laplacian_out, _laplacian_in, _gauge, _set_ptrs, _params):
    laplacian_out = _laplacian_out.data.ptr
    laplacian_in = _laplacian_in.data.ptr
    gauge = _gauge.data.ptr
    set_ptrs = _set_ptrs.ctypes.data
    params = _params.ctypes.data
    qcu.applyLaplacianQcu(laplacian_out, laplacian_in, gauge, set_ptrs, params)