#!/usr/bin/env python3
"""dev74 —— CudaSchurOp：C++ CUDA Schur 奇偶算子封装（多线程版本核心）。

applyCloverBistabCgDslashQcu（cpp/cuda/qcu/python/pyqcu.h:49）实现
Schur 奇偶算子  S = A_oo - k^2 * D_oe * A_ee^-1 * D_eo，
与 Python 端 dslash.operator.matvec_parity 完全等价
（实测 8x8x8x16 c64 相对误差 ~1e-7，单次调用快 ~10x）。

接口约定（经 mg_dev74_layout_test.py 实测确定）：
  * 输入/输出均为 [12, X, Y, Z, T/2]（spin×color 展平、奇子格）连续张量
  * 依赖 applyInitQcu(plan=1) 初始化的 LatticeSet（scratch: device_vec0/1/2 等）

多线程并发约定：
  * 每个 CudaSchurOp 实例持独立 params 副本（int32[54]），其中
    _SET_INDEX_ 独占一个槽位；共享 set_ptrs（槽位互不重叠）。
  * Cython 桥 applyInitQcu/applyCloverBistabCgDslashQcu 均接受自定义
    _params/_set_ptrs 张量 → 各线程调用互不干扰，无共享写竞争。
  * 实例构造必须在单线程完成；每实例显存开销 ≈ LatticeSet scratch
    （见 mg_dev74_budget.py 预算模型）。
"""
import sys, os
import torch
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params, set_ptrs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_GLOBAL_SET_COUNTER = [0]  # 槽位分配器（单线程使用）


def _next_set_index():
    _GLOBAL_SET_COUNTER[0] += 1
    return _GLOBAL_SET_COUNTER[0]


class CudaSchurOp(object):
    """C++ CUDA 实现的 Schur 奇偶算子（matvec_parity 等价物）。

    matvec(x_o) -> y_o：x_o/y_o 为 [12,X,Y,Z,T/2] 奇子格场。
    构造即分配独立 LatticeSet（scratch 缓冲）；release() 释放。
    """

    def __init__(self, av, g, ce, coo, cei, coi):
        self.params = params.clone()
        self.params[define._SET_INDEX_] = _next_set_index()
        self.set_index = int(self.params[define._SET_INDEX_])
        self.params[define._SET_PLAN_] = 1
        self.params[define._VERBOSE_] = 0
        self._g, self._ce, self._coo, self._cei, self._coi = g, ce, coo, cei, coi
        qcu.applyInitQcu(set_ptrs, self.params, av)

    def matvec(self, x_o):
        y_o = torch.empty_like(x_o)
        qcu.applyCloverBistabCgDslashQcu(
            y_o, x_o, self._g, self._ce, self._coo, self._cei, self._coi,
            set_ptrs, self.params)
        return y_o

    def release(self):
        """释放本实例 scratch：用本实例 params 调 applyEndQcu。

        注意 applyEndQcu 释放 set_ptrs 中该槽位对应的 LatticeSet。
        """
        qcu.applyEndQcu(set_ptrs, self.params)
        self.set_index = None


def make_cuda_schur_ops(av, g, ce, coo, cei, coi, n=1, verbose=False):
    """创建 n 个互不冲突的 CudaSchurOp 实例（多线程各持一个）。单线程调用。"""
    ops = [CudaSchurOp(av, g, ce, coo, cei, coi) for _ in range(n)]
    if verbose:
        print(f"[dev74] created {n} CudaSchurOp set_index="
              f"{[o.set_index for o in ops]}")
    return ops
