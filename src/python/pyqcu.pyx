cimport pyqcu
import numpy
cimport numpy
import ctypes
cdef class Pointer:
    cdef void *ptr
    def __cinit__(self, *args):
        self.ptr = NULL
    cdef set_ptr(self, void *ptr):
        self.ptr = ptr
def testWilsonDslashQcu(Pointer _fermion_out, Pointer _fermion_in, Pointer _gauge, Pointer _params, Pointer _argv):
    pyqcu.testWilsonDslashQcu(_fermion_out.ptr, _fermion_in.ptr, _gauge.ptr, _params.ptr, _argv.ptr)
def applyWilsonDslashQcu(Pointer _fermion_out, Pointer _fermion_in, Pointer _gauge, Pointer _params, Pointer _argv):
    pyqcu.applyWilsonDslashQcu(_fermion_out.ptr, _fermion_in.ptr, _gauge.ptr, _params.ptr, _argv.ptr)
def testCloverDslashQcu(Pointer _fermion_out, Pointer _fermion_in, Pointer _gauge, Pointer _params, Pointer _argv):
    pyqcu.testCloverDslashQcu(_fermion_out.ptr, _fermion_in.ptr, _gauge.ptr, _params.ptr, _argv.ptr)
def applyCloverDslashQcu(Pointer _fermion_out, Pointer _fermion_in, Pointer _gauge, Pointer _params, Pointer _argv):
    pyqcu.applyCloverDslashQcu(_fermion_out.ptr, _fermion_in.ptr, _gauge.ptr, _params.ptr, _argv.ptr)
def applyBistabCgQcu(Pointer _fermion_out, Pointer _fermion_in, Pointer _gauge, Pointer _params, Pointer _argv):
    pyqcu.applyBistabCgQcu(_fermion_out.ptr, _fermion_in.ptr, _gauge.ptr, _params.ptr, _argv.ptr)
def applyCgQcu(Pointer _fermion_out, Pointer _fermion_in, Pointer _gauge, Pointer _params, Pointer _argv):
    pyqcu.applyCgQcu(_fermion_out.ptr, _fermion_in.ptr, _gauge.ptr, _params.ptr, _argv.ptr)
def applyGmresIrQcu(Pointer _fermion_out, Pointer _fermion_in, Pointer _gauge, Pointer _params, Pointer _argv):
    pyqcu.applyGmresIrQcu(_fermion_out.ptr, _fermion_in.ptr, _gauge.ptr, _params.ptr, _argv.ptr)