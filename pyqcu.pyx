cimport qcu
from pyqcu.pointer cimport Pointer, Pointers, Pointerss
cdef class QcuParam:
    cdef qcu.QcuParam param
    def __init__(self):
        pass
    @property
    def lattice_size(self):
        return self.param.lattice_size
    @lattice_size.setter
    def lattice_size(self, value):
        self.param.lattice_size = value
def testDslashQcu(Pointer fermion_out, Pointer fermion_in, Pointer gauge, QcuParam param, int parity):
    qcu.testDslashQcu(fermion_out.ptr, fermion_in.ptr, gauge.ptr, &param.param, parity)
def applyDslashQcu(Pointer fermion_out, Pointer fermion_in, Pointer gauge, QcuParam param, int parity, QcuParam grid):
    qcu.applyDslashQcu(fermion_out.ptr, fermion_in.ptr, gauge.ptr, &param.param, parity, &grid.param)
def testCloverDslashQcu(Pointer fermion_out, Pointer fermion_in, Pointer gauge, QcuParam param, int parity):
    qcu.testCloverDslashQcu(fermion_out.ptr, fermion_in.ptr, gauge.ptr, &param.param, parity)
def applyCloverDslashQcu(Pointer fermion_out, Pointer fermion_in, Pointer gauge, QcuParam param, int parity, QcuParam grid):
    qcu.applyCloverDslashQcu(fermion_out.ptr, fermion_in.ptr, gauge.ptr, &param.param, parity, &grid.param)
def applyBistabCgQcu(Pointer fermion_out, Pointer fermion_in, Pointer gauge, QcuParam param, QcuParam grid):
    qcu.applyBistabCgQcu(fermion_out.ptr, fermion_in.ptr, gauge.ptr, &param.param, &grid.param)
def applyCgQcu(Pointer fermion_out, Pointer fermion_in, Pointer gauge, QcuParam param, QcuParam grid):
    qcu.applyCgQcu(fermion_out.ptr, fermion_in.ptr, gauge.ptr, &param.param, &grid.param)
def applyGmresIrQcu(Pointer fermion_out, Pointer fermion_in, Pointer gauge, QcuParam param, QcuParam grid):
    qcu.applyGmresIrQcu(fermion_out.ptr, fermion_in.ptr, gauge.ptr, &param.param, &grid.param)