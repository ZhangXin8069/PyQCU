import pyqcu.cuda.define as define
from pyqcu.cuda.linalg import rayleigh_quotient, orthogonalize_against_vectors
import pyqcu.cuda.bistabcg as bistabcg
import numpy as np
import cupy as cp
def setup(n, k, matvec, dtype, bsi=20, cl=0.95, mi=5, tol=1e-4):
    if cl >= 1.0:
        raise ValueError("cl must be less than 1.0")
    testvectors = []
    for i in range(k):
        testvector_current = cp.random.randn(n).astype(dtype)
        testvector_current /= cp.linalg.norm(testvector_current)
        for j in range(mi):
            if j == 0:
                rayleigh_quotient_current = rayleigh_quotient(
                    testvector_current, matvec)
            else:
                if i == 0:
                    pass
                else:
                    testvector_current = orthogonalize_against_vectors(
                        testvector_current, testvectors)
            testvector_next = bistabcg.slover(
                testvector_current, matvec, max_iter=bsi, tol=tol, x0=cp.zeros_like(testvector_current))
            rayleigh_quotient_next = rayleigh_quotient(
                testvector_next, matvec)
            print("(given) loop-", j, "[", i, "]rayleigh_quotient_current:",
                  rayleigh_quotient_current)
            print("(given) loop-", j,
                  "[", i, "]rayleigh_quotient_next:", rayleigh_quotient_next)
            if rayleigh_quotient_next >= cl*rayleigh_quotient_current:
                break
            testvector_current = testvector_next
            rayleigh_quotient_current = rayleigh_quotient_next
        testvectors = cp.append(testvectors, testvector_next/cp.linalg.norm(testvector_next)).astype(dtype).reshape(
            i+1, n)
    return testvectors
