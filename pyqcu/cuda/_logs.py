"""C++ CUDA 后端收敛日志解析（clover_multigrid.log）。

整合自 logs/dev78_2/main.py::parse_mg_log/_parse_conv_histories 与
examples/qcu/dev73/mg_dev73_5_bench.py::parse_mg_log（4 处内联重复的库级收敛）。

C++ 端（lattice_clover_multigrid.h）无条件写入：
  - CONVERGENCE_HISTORY: [r0,r1,...]   逐迭代残差（一次求解一行）
  - Residual(norm2):(v,...)            逐 V-cycle 残差点
  - PROF_SECTIONS: k=v(ms) ...         分段耗时剖析
  - Total iterations: N                总迭代数
"""
import os
import re
from typing import Dict, List, Optional, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_LOG_PATH = os.path.join(_REPO, 'logs', 'clover_multigrid.log')


def parse_mg_log(path: Optional[str] = None) -> Tuple[List[float], Dict[str, float], Optional[int]]:
    """解析整份日志 → (残差列表, PROF_SECTIONS 字典(ms), 总迭代数)。

    残差列表合并两种来源：CONVERGENCE_HISTORY 行取整段（最后一次覆盖），
    Residual(norm2) 行逐条追加；文件不存在时返回 ([], {}, None)。
    """
    path = DEFAULT_LOG_PATH if path is None else path
    conv: List[float] = []
    prof: Dict[str, float] = {}
    n_iter: Optional[int] = None
    if not os.path.exists(path):
        return conv, prof, n_iter
    with open(path) as f:
        for line in f:
            m = re.search(r'CONVERGENCE_HISTORY:\s*\[([^\]]*)\]', line)
            if m:
                conv = [float(x) for x in m.group(1).split(',') if x.strip()]
                continue
            if "Residual(norm2)" in line:
                mm = re.search(r"Residual\(norm2\):\(([^,]+),", line)
                if mm:
                    conv.append(float(mm.group(1)))
                continue
            if "PROF_SECTIONS" in line:
                for tok in line.split("PROF_SECTIONS:")[1].split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        prof[k] = float(v.rstrip("ms"))
                continue
            m = re.search(r'Total iterations:\s*(\d+)', line)
            if m:
                n_iter = int(m.group(1))
    return conv, prof, n_iter


def parse_convergence_histories(path: Optional[str] = None, offset: int = 0) -> Tuple[List[List[float]], int]:
    """偏移 offset 之后全部 CONVERGENCE_HISTORY → (histories, 新偏移)。

    增量解析：每次调用从上次返回的新偏移继续，实现逐次求解的残差历史
    一一对应收集（多线程时每次求解各产生一条）。offset 超过当前文件大小
    （日志被截断/轮转）时自动从 0 重读。
    """
    path = DEFAULT_LOG_PATH if path is None else path
    histories: List[List[float]] = []
    if not os.path.exists(path):
        return histories, offset
    size = os.path.getsize(path)
    start = offset if 0 <= offset <= size else 0
    with open(path) as f:
        f.seek(start)
        data = f.read()
    for line in data.splitlines():
        m = re.search(r'CONVERGENCE_HISTORY:\s*\[([^\]]*)\]', line)
        if m:
            histories.append([float(x) for x in m.group(1).split(',') if x.strip()])
    return histories, size
