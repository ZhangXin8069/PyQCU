import numpy as np
import cupy as cp 
cimport qcu

def testWilsonDslashQcu(void* fermion_out, void* fermion_in, void* gauge, void* param):
    qcu.testWilsonDslashQcu(fermion_out, fermion_in, gauge, param)
def applyWilsonDslashQcu(void* fermion_out, void* fermion_in, void* gauge, void* param, void* param):
    qcu.applyWilsonDslashQcu(fermion_out, fermion_in, gauge, param)
def testCloverDslashQcu(void* fermion_out, void* fermion_in, void* gauge, void* param):
    qcu.testCloverDslashQcu(fermion_out, fermion_in, gauge, param)
def applyCloverDslashQcu(void* fermion_out, void* fermion_in, void* gauge, void* param, void* param):
    qcu.applyCloverDslashQcu(fermion_out, fermion_in, gauge, param)
def applyBistabCgQcu(void* fermion_out, void* fermion_in, void* gauge, QcuParam param, void* param):
    qcu.applyBistabCgQcu(fermion_out, fermion_in, gauge, param)
def applyCgQcu(void* fermion_out, void* fermion_in, void* gauge, QcuParam param, void* param):
    qcu.applyCgQcu(fermion_out, fermion_in, gauge, param)
def applyGmresIrQcu(void* fermion_out, void* fermion_in, void* gauge, QcuParam param, void* param):
    qcu.applyGmresIrQcu(fermion_out, fermion_in, gauge, param)