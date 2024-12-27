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
    qcu.testWilsonDslashQcu(fermion_out,fermion_in,gauge,params,argv)
def applyWilsonDslashQcu(_fermion_out, _fermion_in, _gauge, _params, _argv):
    fermion_out = _fermion_out.data.ptr
    fermion_in = _fermion_in.data.ptr
    gauge = _gauge.data.ptr
    params = _params.ctypes.data
    argv = _argv.ctypes.data
    qcu.applyWilsonDslashQcu(fermion_out,fermion_in,gauge,params,argv)
def testCloverDslashQcu(_fermion_out, _fermion_in, _gauge, _params, _argv):
    fermion_out = _fermion_out.data.ptr
    fermion_in = _fermion_in.data.ptr
    gauge = _gauge.data.ptr
    params = _params.ctypes.data
    argv = _argv.ctypes.data
    qcu.testCloverDslashQcu(fermion_out,fermion_in,gauge,params,argv)
def applyCloverDslashQcu(_fermion_out, _fermion_in, _gauge, _params, _argv):
    fermion_out = _fermion_out.data.ptr
    fermion_in = _fermion_in.data.ptr
    gauge = _gauge.data.ptr
    params = _params.ctypes.data
    argv = _argv.ctypes.data
    qcu.applyCloverDslashQcu(fermion_out,fermion_in,gauge,params,argv)
def applyBistabCgQcu(_fermion_out, _fermion_in, _gauge, _params, _argv):
    print((_fermion_out.data.ptr))
    print((_fermion_in.data.ptr))
    print((_gauge.data.ptr))
    print((_params.ctypes.data))
    print((_argv.ctypes.data))
    fermion_out = _fermion_out.data.ptr
    fermion_in = _fermion_in.data.ptr
    gauge = _gauge.data.ptr
    params = _params.ctypes.data
    argv = _argv.ctypes.data
    qcu.applyBistabCgQcu(fermion_out,fermion_in,gauge,params,argv)
def applyCgQcu(_fermion_out, _fermion_in, _gauge, _params, _argv):
    fermion_out = _fermion_out.data.ptr
    fermion_in = _fermion_in.data.ptr
    gauge = _gauge.data.ptr
    params = _params.ctypes.data
    argv = _argv.ctypes.data
    qcu.applyCgQcu(fermion_out,fermion_in,gauge,params,argv)
def applyGmresIrQcu(_fermion_out, _fermion_in, _gauge, _params, _argv):
    fermion_out = _fermion_out.data.ptr
    fermion_in = _fermion_in.data.ptr
    gauge = _gauge.data.ptr
    params = _params.ctypes.data
    argv = _argv.ctypes.data
    qcu.applyGmresIrQcu(fermion_out,fermion_in,gauge,params,argv)