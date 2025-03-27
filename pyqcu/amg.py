import pyqcu.define as define
import pyqcu.bistabcg as bistabcg
import numpy as np
import cupy as cp


def rayleigh_quotient(x, matvec):
    return cp.dot(x.conj(), matvec(x)).real / cp.dot(x.conj(), x).real


def setup(n, k, matvec, dtype, bsi=50, cl=0.5, mi=50, tol=1e-2):
    if cl >= 1.0:
        raise ValueError("cl must be less than 1.0")
    testvectors = []
    for i in range(k):
        testvector_current = cp.random.randn(n) + 1j*cp.random.randn(n)
        print("(given) norm of testvector_current:",
              cp.linalg.norm(testvector_current))
        for j in range(mi):
            if j == 0:
                rayleigh_quotient_current = rayleigh_quotient(
                    testvector_current, matvec)
                print("(given) rayleigh_quotient_current:",
                      rayleigh_quotient_current)
            else:
                if i == 0:
                    pass
                else:
                    Q = cp.linalg.qr(cp.array(testvectors[:i]).T)[0]
                    testvector_current -= Q @ (Q.conj().T @ testvector_current)
                    print("(given) loop-", j, ":cp.array(testvectors[:", i, "]).conj() @ testvector_current:",
                          cp.array(testvectors[:i]).conj() @ testvector_current)
            testvector_next = bistabcg.slover(
                testvector_current, matvec, max_iter=bsi, tol=tol)
            rayleigh_quotient_next = rayleigh_quotient(
                testvector_next, matvec)
            print("(given) rayleigh_quotient_next:", rayleigh_quotient_next)
            if rayleigh_quotient_next >= cl*rayleigh_quotient_current:
                break
            testvector_current = testvector_next.copy()
            rayleigh_quotient_current = rayleigh_quotient_next
        testvectors.append(testvector_next)
        print("(given) norm of testvectors[", i,
              "]:", cp.linalg.norm(testvectors[i]))
    testvectors=cp.array(testvectors, dtype=dtype)
    for i in range(k):
        testvectors[i] /= cp.linalg.norm(testvectors[i])
    return testvectors