"""
分层显存管理：VRAM → RAM → 硬盘（data/）自动溢出

当显存不足时，将非必需张量转存至内存；内存不足时转存至硬盘（h5py），
需要时再按需回迁。默认硬盘路径为 ${HOME}/PyQCU/data（与任务22一致）。

设计：
- HierarchicalTensor：单张量的三级存储抽象（VRAM/RAM/DISK），LRU 友好
- HierarchicalCache：多张量 LRU 缓存，按需 offload/reload，线程安全
- 工具函数：offload_to_ram / offload_to_disk / reload_to_vram

与现有 pyqcu/tools/_io.py 的 save_tensor_h5 / load_tensor_h5 复用（h5py 多线程安全，每调用独立 File 句柄）。

sm 兼容说明：本模块不涉及内核编译，仅数据搬运；sm 兼容由 cpp/cuda/qcu/CMakeLists-nv.txt 统一 sm60 实现。
"""
import os
import time
import threading
from pathlib import Path
from typing import Dict, Optional

import torch
import psutil

try:
    from pyqcu.tools._io import save_tensor_h5, load_tensor_h5
except ImportError:
    from _io import save_tensor_h5, load_tensor_h5  # fallback

DEFAULT_CACHE_DIR = Path.home() / "PyQCU" / "data"
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _norm_device(device) -> torch.device:
    """规范设备对象: 无索引 'cuda' 展开为当前卡 'cuda:N'。

    torch.device('cuda') != torch.device('cuda:0')(相等性含索引),
    否则 to_device 对无索引输入误判"不在目标卡"而坠入回迁分支。
    """
    d = torch.device(device)
    if d.type == 'cuda' and d.index is None:
        d = torch.device('cuda', torch.cuda.current_device())
    return d


def _ram_available_bytes() -> int:
    try:
        return psutil.virtual_memory().available
    except Exception:
        # 保守估计 4GB
        return 4 * 1024**3

def _vram_available_bytes(device: torch.device) -> int:
    try:
        if not torch.cuda.is_available():
            return 0
        # torch.cuda.mem_get_info 在 2.0+ 可用
        free, total = torch.cuda.mem_get_info(device)
        return free
    except Exception:
        try:
            # 回退：total - allocated
            props = torch.cuda.get_device_properties(device)
            total = props.total_memory
            allocated = torch.cuda.memory_allocated(device)
            return max(0, total - allocated)
        except Exception:
            return 0

class HierarchicalTensor:
    """
    单张量的三级存储。

    - vram: torch.Tensor 在 GPU（若 None 则已 offload）
    - ram: torch.Tensor 在 CPU（pin_memory 可选）
    - disk_path: Path 在硬盘（h5py）

    迁移策略：
    - offload_to_ram(): vram -> ram (cpu), 释放 vram
    - offload_to_disk(): ram/vram -> disk (h5), 释放 ram/vram
    - to_device(device): 确保在指定 device 的 VRAM 上（按需从 ram/disk 回迁）
    """
    def __init__(self, tensor: torch.Tensor, name: str, cache_dir: Path = DEFAULT_CACHE_DIR, keep_ram: bool = True):
        self.name = name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.disk_path = self.cache_dir / f"hier_{name}.h5"
        self.shape = tuple(tensor.shape)
        self.dtype = tensor.dtype
        self.keep_ram = keep_ram
        # 初始在 VRAM
        self.vram: Optional[torch.Tensor] = tensor
        self.ram: Optional[torch.Tensor] = None
        self.on_disk = False
        self.lock = threading.Lock()
        self.last_access = time.time()

    def offload_to_ram(self) -> bool:
        with self.lock:
            if self.vram is None:
                return False
            try:
                # 尝试转存到 RAM，若 RAM 不足则直接到硬盘
                needed = self.vram.numel() * self.vram.element_size()
                if _ram_available_bytes() < needed * 1.2:
                    return self.offload_to_disk()
                self.ram = self.vram.detach().cpu()
                # 保留 vram 的 shape/dtype 供后续 reload，但释放显存
                del self.vram
                self.vram = None
                torch.cuda.empty_cache()
                self.last_access = time.time()
                return True
            except Exception as e:
                print(f"[Hierarchical] offload_to_ram {self.name} failed {e}, try disk")
                return self.offload_to_disk()

    def offload_to_disk(self) -> bool:
        with self.lock:
            # 优先从 vram，其次 ram
            src = self.vram if self.vram is not None else self.ram
            if src is None and not self.on_disk:
                return False
            try:
                if src is not None:
                    # 保存到硬盘
                    save_tensor_h5(src, str(self.disk_path), dataset="data", verbose=False)
                    self.on_disk = True
                    # 释放
                    if self.vram is not None:
                        del self.vram
                        self.vram = None
                    if self.ram is not None:
                        del self.ram
                        self.ram = None
                    torch.cuda.empty_cache()
                self.last_access = time.time()
                return True
            except Exception as e:
                print(f"[Hierarchical] offload_to_disk {self.name} failed {e}")
                return False

    def to_device(self, device: torch.device) -> torch.Tensor:
        """确保在指定 device 的 VRAM 上，必要时从 RAM/DISK 回迁（带 VRAM 空间检查）。"""
        with self.lock:
            device = _norm_device(device)
            if self.vram is not None and self.vram.device == device:
                self.last_access = time.time()
                return self.vram
            # 需要回迁：检查 VRAM 空间，若不足则让调用方先 offload 其他张量（由 HierarchicalCache 统一调度）
            # 此处直接尝试分配，若 OOM 则抛出由上层捕获并触发 LRU offload
            if self.on_disk:
                # 从硬盘加载
                t = load_tensor_h5(str(self.disk_path), dataset="data", device=device, verbose=False)
                self.vram = t
                self.last_access = time.time()
                return self.vram
            if self.ram is not None:
                # 从 RAM 搬运
                # 检查 VRAM 空间
                needed = self.ram.numel() * self.ram.element_size()
                if _vram_available_bytes(device) < needed * 1.2:
                    raise torch.cuda.OutOfMemoryError(f"VRAM insufficient for {self.name} {needed/1e9:.2f}GB")
                self.vram = self.ram.to(device, non_blocking=True)
                if not self.keep_ram:
                    del self.ram
                    self.ram = None
                self.last_access = time.time()
                return self.vram
            raise RuntimeError(f"[Hierarchical] {self.name} has no data in any tier")

    def is_on_vram(self) -> bool:
        return self.vram is not None

    def memory_tier(self) -> str:
        if self.vram is not None:
            return "VRAM"
        if self.ram is not None:
            return "RAM"
        if self.on_disk:
            return "DISK"
        return "NONE"

class HierarchicalCache:
    """
    多张量 LRU 分层缓存。

    - register(name, tensor): 注册张量（初始在 VRAM）
    - ensure_fit(device, needed_bytes): 确保 device 上有足够 VRAM，不足时按 LRU 将非必需张量 offload 到 RAM/DISK
    - get(name, device): 获取张量在指定 device 的 VRAM 上（自动回迁）
    - offload_lru(device, needed_bytes): 按 LRU 顺序 offload 直到满足 needed_bytes
    """
    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tensors: Dict[str, HierarchicalTensor] = {}
        self.lock = threading.Lock()

    def register(self, name: str, tensor: torch.Tensor, keep_ram: bool = True) -> HierarchicalTensor:
        with self.lock:
            ht = HierarchicalTensor(tensor, name, cache_dir=self.cache_dir, keep_ram=keep_ram)
            self.tensors[name] = ht
            return ht

    def get(self, name: str, device: torch.device) -> torch.Tensor:
        ht = self.tensors.get(name)
        if ht is None:
            raise KeyError(f"[HierarchicalCache] {name} not found")
        # 尝试直接获取，若 OOM 则触发 LRU offload 重试
        try:
            return ht.to_device(device)
        except torch.cuda.OutOfMemoryError:
            # 需要 offload 其他张量
            needed = ht.ram.numel()*ht.ram.element_size() if ht.ram is not None else 1024**3
            self.offload_lru(device, needed)
            return ht.to_device(device)

    def offload_lru(self, device: torch.device, needed_bytes: int):
        """按 LRU 将非活跃张量 offload，直到 VRAM 满足 needed_bytes 或无可 offload。"""
        with self.lock:
            device = _norm_device(device)
            # 按 last_access 排序，最久未访问的先 offload
            candidates = sorted(self.tensors.values(), key=lambda x: x.last_access)
            for ht in candidates:
                if _vram_available_bytes(device) >= needed_bytes * 1.2:
                    break
                if ht.is_on_vram() and ht.vram.device == device:
                    # 避免 offload 正在请求的张量本身（由调用方保证）
                    # 此处按 LRU，跳过最近访问的（可能为当前请求）
                    if ht.name == getattr(ht, '_exclude', None):
                        continue
                    # 优先 RAM，若 RAM 不足则 DISK
                    if _ram_available_bytes() > ht.vram.numel()*ht.vram.element_size()*1.2:
                        ht.offload_to_ram()
                    else:
                        ht.offload_to_disk()

    def status(self) -> Dict[str, str]:
        return {k: v.memory_tier() for k, v in self.tensors.items()}

# 便捷函数（无状态）
def offload_to_ram(tensor: torch.Tensor) -> torch.Tensor:
    """将张量从 VRAM 转存到 RAM，释放显存，返回 CPU 张量。"""
    cpu = tensor.detach().cpu()
    del tensor
    torch.cuda.empty_cache()
    return cpu

def offload_to_disk(tensor: torch.Tensor, path: str, dataset: str = "data") -> None:
    """将张量转存到硬盘 h5，释放显存与内存。"""
    save_tensor_h5(tensor, path, dataset=dataset, verbose=False)
    del tensor
    torch.cuda.empty_cache()

def reload_to_vram(path: str, dataset: str, device: torch.device) -> torch.Tensor:
    """从硬盘或 RAM 重新加载到 VRAM。"""
    return load_tensor_h5(path, dataset=dataset, device=device, verbose=False)
