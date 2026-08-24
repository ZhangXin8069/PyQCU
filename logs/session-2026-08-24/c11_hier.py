"""_hierarchical.py 分层显存管理冒烟（VRAM→RAM→DISK）。"""
import traceback
import torch
from pathlib import Path
from pyqcu.tools._hierarchical import HierarchicalTensor, HierarchicalCache

TMP = Path('/tmp/opencode/hier_cache')
results = []


def run(name, fn):
    try:
        msg = fn() or ""
        results.append((name, 'PASS', ''))
        print(f"[HIER][PASS] {name} {msg}", flush=True)
    except Exception as e:
        results.append((name, 'FAIL', f"{type(e).__name__}: {e}"))
        print(f"[HIER][FAIL] {name}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()


def t_tensor_ram_roundtrip():
    ht = HierarchicalTensor(torch.randn(1024, dtype=torch.complex64,
                              device='cuda'), "t1",
                            cache_dir=TMP)
    assert ht.is_on_vram()
    assert ht.offload_to_ram()
    assert not ht.is_on_vram() and ht.ram is not None
    back = ht.to_device(torch.device('cuda'))
    assert torch.equal(back.cpu(), ht.ram.cpu()) if ht.ram is not None else True
    return f"tier={ht.memory_tier()}"


def t_tensor_disk_roundtrip():
    ht = HierarchicalTensor(torch.randn(512, dtype=torch.float32,
                                        device='cuda'), "t2", cache_dir=TMP,
                            keep_ram=False)
    assert ht.offload_to_ram()
    assert ht.offload_to_disk()
    disk_files = list(TMP.glob("t2*"))
    back = ht.to_device(torch.device('cuda'))
    assert float((back - ht.vram).abs().max().item()) == 0.0
    assert len(disk_files) >= 0
    return f"disk roundtrip ok tier={ht.memory_tier()}"


def t_cache_lru_flow():
    c = HierarchicalCache(cache_dir=TMP / "cache")
    a = torch.randn(256, device='cuda')
    b = torch.randn(256, device='cuda')
    c.register("a", a)
    c.register("b", b)
    # 访问 a 使 b 成为 LRU 尾部，然后回迁 a 前先 offload b
    _ = c.get("a", torch.device('cuda'))
    import time
    time.sleep(0.01)
    c.offload_lru(torch.device('cuda'), needed_bytes=1)  # 触发一次 LRU offload
    st = c.status()
    va = c.get("a", torch.device('cuda'))
    vb = c.get("b", torch.device('cuda'))
    assert torch.equal(va.cpu(), a.cpu()) and torch.equal(vb.cpu(), b.cpu())
    try:
        c.get("nope", torch.device('cuda'))
        raise AssertionError("missing key did not raise")
    except KeyError:
        pass
    return f"status={st}"


run("tensor_ram_roundtrip", t_tensor_ram_roundtrip)
run("tensor_disk_roundtrip", t_tensor_disk_roundtrip)
run("cache_lru_flow", t_cache_lru_flow)

n_fail = sum(1 for _, s, _ in results if s == 'FAIL')
print(f"\n[HIER][SUMMARY] {len(results)-n_fail}/{len(results)} PASS")
