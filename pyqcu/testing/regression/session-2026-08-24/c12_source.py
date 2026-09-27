"""lattice/_source.py 五种源场构造冒烟（物理性质级验证）。"""
import traceback
import torch
from pyqcu.lattice._source import (point_source, wall_source, volume_source,
                                   z2_source, momentum_source)

LAT = [4, 4, 4, 8]
DT = torch.complex64
results = []


def run(name, fn):
    try:
        msg = fn() or ""
        results.append((name, 'PASS', ''))
        print(f"[SRC][PASS] {name} {msg}", flush=True)
    except Exception as e:
        results.append((name, 'FAIL', f"{type(e).__name__}: {e}"))
        print(f"[SRC][FAIL] {name}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


def t_point():
    s = point_source(LAT, t_srce=[1, 2, 3, 5], spin=0, color=0,
                     dtype=DT, device=torch.device('cpu'))
    nz = torch.nonzero(s.abs() > 0)
    assert nz.shape[0] == 1, f"point source must have exactly 1 nonzero, got {nz.shape[0]}"
    assert tuple(nz[0].tolist()[1:]) == (1, 2, 3, 5) or tuple(nz[0].tolist())[-4:] == (1, 2, 3, 5), nz
    return f"single-site at [*,*,1,2,3,5]"


def t_wall():
    t_srce = 3
    s = wall_source(LAT, t_srce=t_srce, dtype=DT, device=torch.device('cpu'))
    mask_t = [i == t_srce for i in range(LAT[3])]
    for ti in range(LAT[3]):
        sl = s[..., ti]
        n = int((sl.abs() > 0).sum())
        if mask_t[ti]:
            assert n == sl.numel(), f"wall slice t={ti} not full: {n}/{sl.numel()}"
        else:
            assert n == 0, f"non-wall slice t={ti} has {n} nonzeros"
    return "single t-slice fully populated"


def t_volume():
    s = volume_source(LAT, dtype=DT, device=torch.device('cpu'))
    assert bool(torch.isfinite(s.real).all())
    nz = int((s.abs() > 0).sum())
    assert nz == s.numel(), f"volume source must be dense: {nz}/{s.numel()}"
    return f"dense {tuple(s.shape)}"


def t_z2():
    a = z2_source(LAT, seed=42, dtype=torch.float32, device=torch.device('cpu'))
    vals = torch.unique(a[a != 0])
    assert set(vals.tolist()).issubset({1.0, -1.0}), f"Z2 values {vals}"
    b = z2_source(LAT, seed=42, dtype=torch.float32, device=torch.device('cpu'))
    assert torch.equal(a, b), "z2 not deterministic for same seed"
    return f"values={{±1}}, deterministic"


def t_momentum():
    mode = [1, 0, 0, 0]
    t_srce = None
    s = momentum_source(LAT, mode=mode, t_srce=t_srce, dtype=DT, device=torch.device('cpu'))
    # 动量源 |值| 恒定(平面波), 相位沿 x 以 2π/Lx 旋转
    amp = s.abs()
    assert torch.allclose(amp, amp[0, 0, 0, 0].expand_as(amp)), \
        "momentum source amplitude must be constant"
    x_phase = s[0, 0, :, :, :, :].reshape(LAT[0], -1)[:, 0]
    phase_x = torch.angle(x_phase)
    expected = 2 * 3.14159265 * torch.arange(LAT[0]) * mode[0] / LAT[0]
    dph = (phase_x - expected + 3.14159265) % (2 * 3.14159265) - 3.14159265
    assert float(dph.abs().max()) < 1e-3, \
        f"x phase mismatch max={float(dph.abs().max()):.3e}"
    return f"plane-wave e^{{ipx}} verified mode={mode}"


run("point_source", t_point)
run("wall_source", t_wall)
run("volume_source", t_volume)
run("z2_source", t_z2)
run("momentum_source", t_momentum)

n_fail = sum(1 for _, s, _ in results if s == 'FAIL')
print(f"\n[SRC][SUMMARY] {len(results)-n_fail}/{len(results)} PASS")
