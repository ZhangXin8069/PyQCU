"""_budget.py 预算模型 + _env.py 环境快照冒烟。
注意: vram_model/vram_model_warm/rss_model 返回 **MB**; disk_cache_bytes 返回 bytes。
"""
import torch
from pyqcu.tools import _budget as B, _env as E

for v in (1e4, 1e6):
    print(f"v={v:.0e}: disk={B.disk_cache_bytes(v)/1e9:.1f}GB "
          f"vram={B.vram_model(v):.0f}MB warm={B.vram_model_warm(v):.0f}MB "
          f"rss={B.rss_model(v):.0f}MB")
assert B.vram_model(2e6) > B.vram_model(1e6) > 0
tbl = B.budget_table(mode="cluster", vram_gb=32)
assert isinstance(tbl, list) and all(isinstance(r, dict) and "pred_vram_mb" in r for r in tbl)
print(f"budget_table rows={[r['lattice'] for r in tbl]}")

gs = E.git_snapshot(); assert isinstance(gs, dict) and gs.get("head")
p = "/tmp/opencode/env_dump.h5"
E.dump_env_h5(path=p)
from pyqcu.tools._io import load_dict_h5
back = load_dict_h5(p); assert isinstance(back, dict) and back
print("git_snapshot/dump_env_h5 roundtrip OK:", sorted(back.keys())[:4])
print("[BUDGET_ENV] ALL PASS")
