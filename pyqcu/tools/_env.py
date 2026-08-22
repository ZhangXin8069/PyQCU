"""环境与资源快照工具。

整合自 logs/dev78_2/main.py::_git_snapshot/_gpu_snapshot/_gpu_used_mb/
rss_kb/cache_disk_mb/dump_env_h5 与 examples/qcu/dev74/mg_dev74_bench.py。
用途：基准测试环境快照（env.h5，跨环境比对）与运行期资源统计。
"""
import os
import resource
import socket
import subprocess
import sys
from typing import Dict, Optional

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def git_snapshot() -> Dict[str, str]:
    """git 分支与最新提交（子进程失败时返回 '?' 占位）。"""
    try:
        br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                            cwd=_REPO, capture_output=True, text=True, timeout=10)
        hd = subprocess.run(["git", "log", "-1", "--oneline"],
                            cwd=_REPO, capture_output=True, text=True, timeout=10)
        return {"branch": br.stdout.strip() or "?", "head": hd.stdout.strip() or "?"}
    except Exception:
        return {"branch": "?", "head": "?"}


def gpu_snapshot() -> str:
    """nvidia-smi GPU 静态信息（index,name,memory.total,driver_version；不可用返回 '?'）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "?"
    except Exception:
        return "?"


def gpu_used_mb() -> str:
    """当前各卡显存占用采样（nvidia-smi，MB；不可用返回 '?'）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip().replace("\n", "; ") or "?"
    except Exception:
        return "?"


def rss_kb() -> int:
    """本进程峰值 RSS（KB，getrusage ru_maxrss）。"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def nullvec_cache_dir() -> str:
    """null 向量/粗算子共享缓存目录（PYQCU_NULLVEC_CACHE 可覆盖）。"""
    return os.environ.get("PYQCU_NULLVEC_CACHE",
                          os.path.join(_REPO, "logs", "nullvec_cache"))


def cache_disk_mb(directory: Optional[str] = None) -> float:
    """目录磁盘占用（du -s，MB；目录不存在或 du 失败返回 0.0）。"""
    target = directory or nullvec_cache_dir()
    try:
        out = subprocess.run(["du", "-s", target], capture_output=True,
                             text=True, timeout=10)
        if out.returncode != 0 or not out.stdout.strip():
            return 0.0
        return int(out.stdout.split()[0]) / 1024.0
    except Exception:
        return 0.0


def dump_env_h5(path: Optional[str] = None,
                cmdline: Optional[str] = None) -> Dict:
    """环境快照写入 .h5（经 tools.save_dict_h5），返回快照 dict。

    快照内容：git branch/head、GPU 静态信息、torch 版本/CUDA 设备数、
    主机名、命令行。默认写到 ./env.h5（当前工作目录）。
    """
    import torch
    from pyqcu.tools._io import save_dict_h5
    g = git_snapshot()
    env = {"branch": g["branch"], "head": g["head"],
           "gpu": gpu_snapshot(),
           "torch": torch.__version__ if torch.cuda.is_available() else "cpu",
           "cuda_devices": torch.cuda.device_count(),
           "host": socket.gethostname(),
           "cmdline": " ".join(sys.argv) if cmdline is None else cmdline}
    save_dict_h5(path or os.path.join(os.getcwd(), "env.h5"), {"env": env})
    return env
