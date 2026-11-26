"""Capture fraction rho of a coarse basis for a real BiCGStab residual.

Redone on a NON-SINGULAR operator.  The earlier run of this measurement used
mass=-0.277, where a plain BiCGStab on this very gauge diverges and goes NaN at
iteration 229 (logs/v20261123.txt section 180), so its 6.1x figure is void.  Here
mass=+0.1 -> kappa=0.121951 < kappa_c=1/8 and the same solve converges to the c64
floor.

Problem: lattice [8,8,8,16] c64, mass=+0.1, csw=1.160920226, mg_grid [2,2,2,2],
dof [12,2].  The two caches were built by the C++ builder on a compute node, one
per null-vector builder, both status=PASS:

    sb_inverse  nv_builder=inverse-iteration/2
    sb_legacy   nv_builder=legacy-residual/1

Layout (proved from the code and re-verified here):
  full field        [..., 4, 3, x, y, z, t]                  (t fastest)
  odd parity block  [..., 4, 3, x, y, z, t//2], odd t only
  lonv              [E, e, Xc,bx, Yc,by, Zc,bz, Tc,bt]        c = i//2, b = i%2

rho is the fraction of the residual that lies in the coarse basis's range:
    rho = sum_c g_c^H G_c^+ g_c / ||r||^2,  g = V^H r,  G = V^H V
with V the block's (e*block^4, E) basis matrix.  Two independent routes are
computed (Gram pseudo-inverse and the SVD frame) and must agree.

Gates that must hold before any rho is reported:
  * a single basis vector as r must give rho = 1.000000 for BOTH bases
  * a dense random odd field must give rho within 10% of capacity = E/(e*block^4)
  (the gate was 5% at first and is 10% in the code: 256 blocks give a few percent
  of sampling noise on this statistic)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

# Both input directories are build products and are not in Git, so they come from
# the command line: SMALL holds the gauge and the point source, CACHES the two
# per-builder coarse bases.  --json writes the measured table out.
SMALL: Path | None = None
CACHES: Path | None = None

LATTICE = (8, 8, 8, 16)
MASS = 0.1
CSW = 1.160920226
KAPPA = 1.0 / (2.0 * MASS + 8.0)
E_COARSE, E_DOF, BLOCK = 2, 12, 2
FINE_ODD = (8, 8, 8, 8)
COARSE = (4, 4, 4, 4)
CAPACITY = E_COARSE / (E_DOF * BLOCK ** 4)
WANT = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]


def operator():
    from pyqcu.torch import dslash

    with h5py.File(SMALL / "gauge_8x8x8x16_c64.h5", "r") as handle:
        gauge = torch.from_numpy(handle["data"][()].astype(np.complex64))
    kappa = torch.tensor([KAPPA], dtype=torch.float32)
    clover = dslash.make_clover(gauge, kappa=kappa, csw=CSW, verbose=False)

    def apply(x):
        return (dslash.give_wilson(x, gauge, kappa, with_I=True)
                + dslash.give_clover(x, clover))

    return apply


def full_to_odd(full: np.ndarray) -> np.ndarray:
    """(4,3,x,y,z,t) or (12,x,y,z,t) -> (12,8,8,8,8), the odd-Schur block."""
    if full.shape[-6:-4] == (4, 3):
        full = full.reshape((E_DOF,) + LATTICE)
    lead = full.shape[:-4]
    out = np.empty(lead + FINE_ODD, dtype=full.dtype)
    for t2 in range(FINE_ODD[3]):
        x, y, z = np.indices(FINE_ODD[:3])
        t = 2 * t2 + (1 - ((x + y + z) & 1))
        out[..., :, :, :, t2] = full[..., :, x, y, z, t]
    return out


def residuals(matvec, source, want):
    """Return {k: true residual r_k = b - A x_k} for a plain BiCGStab."""
    b = source
    x = torch.zeros_like(b)
    r = b.clone()
    r_tilde = r.clone()
    p = torch.zeros_like(b)
    v = torch.zeros_like(b)
    one = torch.tensor(1.0 + 0j, dtype=r.dtype)
    rho_prev = alpha = omega = one
    keep = {0: r.clone()}
    k = 0
    for step in range(1, max(want) + 1):
        rho = torch.vdot(r_tilde.flatten(), r.flatten())
        if step == 1:
            p = r.clone()
        else:
            beta = (rho / rho_prev) * (alpha / omega)
            p = r + beta * (p - omega * v)
        v = matvec(p)
        alpha = rho / torch.vdot(r_tilde.flatten(), v.flatten())
        s = r - alpha * v
        t = matvec(s)
        omega = (torch.vdot(t.flatten(), s.flatten())
                 / torch.vdot(t.flatten(), t.flatten()))
        x = x + alpha * p + omega * s
        r = s - omega * t
        rho_prev = rho
        k = step
        if step in want:
            keep[step] = r.clone()
        if float(torch.linalg.vector_norm(r.flatten())) / float(
                torch.linalg.vector_norm(b.flatten())) < 1e-8:
            break
    return keep, k


def roughness(field: np.ndarray) -> float:
    f = field.reshape(E_DOF, *LATTICE)
    num = 0.0
    for axis in range(4):
        d = np.roll(f, -1, axis=1 + axis) - f
        num += float(np.vdot(d.ravel(), d.ravel()).real)
    den = float(np.vdot(f.ravel(), f.ravel()).real)
    return num / den if den else float("nan")


def capture(field: np.ndarray, lonv: np.ndarray) -> dict:
    r = np.asarray(field, dtype=np.complex128)
    basis = np.asarray(lonv, dtype=np.complex128)
    total = total_svd = 0.0
    ranks = []
    for cx in range(COARSE[0]):
        for cy in range(COARSE[1]):
            for cz in range(COARSE[2]):
                for ct in range(COARSE[3]):
                    A = np.ascontiguousarray(
                        basis[:, :, cx, :, cy, :, cz, :, ct, :]
                    ).reshape(E_COARSE, E_DOF * BLOCK ** 4)
                    rc = np.ascontiguousarray(
                        r[:, 2 * cx:2 * cx + 2, 2 * cy:2 * cy + 2,
                          2 * cz:2 * cz + 2, 2 * ct:2 * ct + 2]).reshape(-1)
                    g = A.conj() @ rc
                    G = A.conj() @ A.T
                    sv = np.linalg.svd(A, compute_uv=False)
                    keep = sv > 1e-10 * sv.max()
                    ranks.append(int(keep.sum()))
                    total += float(np.real(g.conj() @ np.linalg.pinv(
                        G, rcond=1e-10, hermitian=True) @ g))
                    U, s, Vh = np.linalg.svd(A, full_matrices=False)
                    proj = Vh[keep].conj() @ rc
                    total_svd += float(np.vdot(proj, proj).real)
    norm2 = float(np.vdot(r.ravel(), r.ravel()).real)
    if norm2 == 0.0:
        # k=0 with a point source at the origin: the source is even-parity, so its
        # odd-Schur block is identically zero and rho is undefined, not zero.
        return {"rho": None, "rho_svd": None,
                "rank_min": min(ranks), "rank_max": max(ranks)}
    return {"rho": total / norm2, "rho_svd": total_svd / norm2,
            "rank_min": min(ranks), "rank_max": max(ranks)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture fraction of a coarse basis for a real BiCGStab "
                    "residual.  Both inputs are build products that are not in "
                    "Git, so both must be given.")
    parser.add_argument("--small", required=True, type=Path,
                        help="directory holding gauge_8x8x8x16_c64.h5 and "
                             "source_point_8x8x8x16_c64.h5")
    parser.add_argument("--caches", required=True, type=Path,
                        help="directory holding sb_inverse_lonv.npy and "
                             "sb_legacy_lonv.npy, the two per-builder caches")
    parser.add_argument("--json", type=Path, default=None,
                        help="write the measured rows here (default: stdout only)")
    arguments = parser.parse_args(argv)
    global SMALL, CACHES
    SMALL, CACHES = arguments.small, arguments.caches

    bases = {}
    for name in ("inverse", "legacy"):
        path = CACHES / ("sb_%s_lonv.npy" % name)
        if not path.is_file():
            print("missing", path)
            return 2
        bases[name] = np.load(path)
    print("operator: lattice %s mass %+.3f kappa %.7f (kappa_c = 0.125)"
          % (LATTICE, MASS, KAPPA))
    print("capacity E/(e*block^4) = %.6f   sqrt = %.6f"
          % (CAPACITY, math.sqrt(CAPACITY)))

    print("\n== gates")
    for name, lonv in bases.items():
        unit = np.zeros((E_DOF,) + FINE_ODD, dtype=np.complex128)
        # the WHOLE basis vector 1 at coarse site 0, i.e. all 16 block positions
        unit[:, 0:2, 0:2, 0:2, 0:2] = lonv[1, :, 0, :, 0, :, 0, :, 0, :]
        gate1 = capture(unit, lonv)["rho"]
        rng = np.random.default_rng(5)
        dense = (rng.standard_normal((E_DOF,) + FINE_ODD)
                 + 1j * rng.standard_normal((E_DOF,) + FINE_ODD))
        gate2 = capture(dense, lonv)["rho"]
        print("  %-8s single-vector rho %.6f (==1)  dense rho %.6f (%.1f%% of "
              "capacity)  %s"
              % (name, gate1, gate2, 100.0 * gate2 / CAPACITY,
                 "PASS" if abs(gate1 - 1.0) < 1e-4
                 and abs(gate2 / CAPACITY - 1.0) < 0.10 else "FAIL"))

    print("\n== residuals and capture")
    with h5py.File(SMALL / "source_point_8x8x8x16_c64.h5", "r") as handle:
        source = torch.from_numpy(handle["data"][()].astype(np.complex64))
    matvec = operator()
    keep, reached = residuals(matvec, source, WANT)
    b_norm = float(torch.linalg.vector_norm(source.flatten()))
    rows = []
    for k in sorted(keep):
        r_full = keep[k].numpy()
        r_odd = full_to_odd(r_full)
        if float(np.linalg.norm(r_odd)) == 0.0:
            print("  k=%-3d odd block is exactly zero (even-parity source at k=0);"
                  " rho undefined" % k)
            continue
        row = {"k": k,
               "relative": float(np.linalg.norm(r_full)) / b_norm,
               "roughness": roughness(r_full)}
        for name, lonv in bases.items():
            row[name] = capture(r_odd, lonv)["rho"]
        row["ratio_leg_over_inv"] = row["legacy"] / row["inverse"]
        rows.append(row)
        print("  k=%-3d rel=%-11.3e rough=%6.3f  rho_inv=%.6f rho_leg=%.6f  "
              "leg/inv=%6.3f"
              % (k, row["relative"], row["roughness"], row["inverse"],
                 row["legacy"], row["ratio_leg_over_inv"]))
    print("\nBiCGStab reached k=%d" % reached)
    document = {"mass": MASS, "kappa": KAPPA, "capacity": CAPACITY, "rows": rows}
    if arguments.json is not None:
        arguments.json.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
