---
name: smear
description: pyqcu.smear 目录的完整生成 skill：stout smearing（Morningstar-Peardon SU(3) 投影），含数值稳定性处理与 MPI 支持。
---
# CLAUDE.md — pyqcu.smear

Gauge field smearing — spatial smoothing of gauge links to reduce UV noise.

## Files

| File | Purpose |
|------|---------|
| `_stout.py` | Stout smearing algorithm (copied/adapted from EasyDistillation's elemental generator) |

## Exported API

### `stout_smear(U, nstep=1, rho=0.12, support_parallel=False) → torch.Tensor`

Apply nstep iterations of stout smearing with parameter rho.

**Algorithm (per step):**

1. **Compute Q_μ = staple sum** for each direction μ: sum over ν≠μ of two 3-link staples (U_ν U_μ U^†_ν forward + U^†_ν U_μ U_ν backward)
2. **Project to su(3) algebra:** Q ← ρ · Q · U^†, then anti-hermitize: Q ← i/2 · (Q^† − Q) − (1/3) Tr(Q) · I
3. **Compute SU(3) projection coefficients f₀, f₁, f₂** via the Morningstar-Peardon method:
   - c₀ = Re(Tr(Q³))/3, c₁ = Re(Tr(Q²))/2
   - θ = arccos(c₀ / (2(c₁/3)^(3/2)))
   - u = √(c₁/3) · cos(θ/3), w = √c₁ · sin(θ/3)
   - f₀, f₁, f₂ expressed in terms of e^{iu}, e^{2iu}, cos(w), sinc(w)
4. **Parity handling** (when c₀ < 0): f₀ → f₀^*, f₁ → −f₁^*, f₂ → f₂^* (standard path); NPU path uses real/imag decomposition
5. **Update U:** U_new = (f₀·I + f₁·Q + f₂·Q²) · U

**Numerical stability:**
- c₁ clamped to min 1e-15 (prevents c₀_max = 0)
- ratio clamped to [−1+1e-15, 1−1e-15] for arccos domain
- sinc(w) uses Taylor expansion for |w| ≤ 0.05, sin(w)/w otherwise
- Denominator 9u² − w² has 1e-15 epsilon to prevent division by zero

**MPI support:** When `support_parallel=True`, MPI boundary data (U_head, U_tail, U_head_tail) is recomputed each step since U changes with each smearing step.

## Key Anti-Pattern (Fixed)

The `nstep>1` loop previously did not update `U` between steps — the loop variable was properly rebound but the MPI boundary data was computed outside the loop. Fixed by moving MPI exchange inside the step loop.

## Data Layout

Gauge field: `[3, 3, 4, Lx, Ly, Lz, Lt]` = `[color, color, direction, x, y, z, t]`

Returned tensor has the same shape.

## NPU Support

Has per-module `force_use_npu` flag. On NPU, the parity sign convention for f₀/f₁/f₂ uses explicit real/imag decomposition:
- f₀: imag = −imag (conj)
- f₁: real = −real, imag unchanged (conj + leading minus cancel)
- f₂: real = −real, imag unchanged (same as f₁)
