cimport qcu
import numpy
cimport numpy
import ctypes
cdef long long fermion_out
cdef long long fermion_in
cdef long long gauge
cdef long long params
cdef long long argv
def testWilsonDslashQcu(_fermion_out, _fermion_in, _gauge, _params, _argv):
    fermion_out = _fermion_out.data.ptr
    fermion_in = _fermion_in.data.ptr
    gauge = _gauge.data.ptr
    params = _params.ctypes.data
    argv = _argv.ctypes.data
    qcu.testWilsonDslashQcu(fermion_out, fermion_in, gauge, params, argv)
def applyWilsonDslashQcu(_fermion_out, _fermion_in, _gauge, _params, _argv):
    fermion_out = _fermion_out.data.ptr
    fermion_in = _fermion_in.data.ptr
    gauge = _gauge.data.ptr
    params = _params.ctypes.data
    argv = _argv.ctypes.data
    qcu.applyWilsonDslashQcu(fermion_out, fermion_in, gauge, params, argv)
def testCloverDslashQcu(_fermion_out, _fermion_in, _gauge, _params, _argv):
    fermion_out = _fermion_out.data.ptr
    fermion_in = _fermion_in.data.ptr
    gauge = _gauge.data.ptr
    params = _params.ctypes.data
    argv = _argv.ctypes.data
    qcu.testCloverDslashQcu(fermion_out, fermion_in, gauge, params, argv)
def applyCloverDslashQcu(_fermion_out, _fermion_in, _gauge, _params, _argv):
    fermion_out = _fermion_out.data.ptr
    fermion_in = _fermion_in.data.ptr
    gauge = _gauge.data.ptr
    params = _params.ctypes.data
    argv = _argv.ctypes.data
    qcu.applyCloverDslashQcu(fermion_out, fermion_in, gauge, params, argv)
def applyWilsonBistabCgQcu(_fermion_out, _fermion_in, _gauge, _params, _argv):
    fermion_out = _fermion_out.data.ptr
    fermion_in = _fermion_in.data.ptr
    gauge = _gauge.data.ptr
    params = _params.ctypes.data
    argv = _argv.ctypes.data
    qcu.applyWilsonBistabCgQcu(fermion_out, fermion_in, gauge, params, argv)
def applyWilsonBistabCgDslashQcu(_fermion_out, _fermion_in, _gauge, _params, _argv):
    fermion_out = _fermion_out.data.ptr
    fermion_in = _fermion_in.data.ptr
    gauge = _gauge.data.ptr
    params = _params.ctypes.data
    argv = _argv.ctypes.data
    qcu.applyWilsonBistabCgDslashQcu(fermion_out, fermion_in, gauge, params, argv)
def applyWilsonCgQcu(_fermion_out, _fermion_in, _gauge, _params, _argv):
    fermion_out = _fermion_out.data.ptr
    fermion_in = _fermion_in.data.ptr
    gauge = _gauge.data.ptr
    params = _params.ctypes.data
    argv = _argv.ctypes.data
    qcu.applyWilsonCgQcu(fermion_out, fermion_in, gauge, params, argv)
def applyWilsonGmresIrQcu(_fermion_out, _fermion_in, _gauge, _params, _argv):
    fermion_out = _fermion_out.data.ptr
    fermion_in = _fermion_in.data.ptr
    gauge = _gauge.data.ptr
    params = _params.ctypes.data
    argv = _argv.ctypes.data
    qcu.applyWilsonGmresIrQcu(fermion_out, fermion_in, gauge, params, argv)