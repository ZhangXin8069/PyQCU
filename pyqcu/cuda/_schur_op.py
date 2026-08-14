"""C++ CUDA Schur 奇偶算子封装（多线程安全，一线程一卡模式的基础算子）。

applyCloverBistabCgDslashQcu（cpp/cuda/qcu/python/pyqcu.h）实现
Schur 奇偶算子  S = A_oo - k^2 * D_oe * A_ee^-1 * D_eo，
与 Python 端 dslash.operator.matvec_parity 等价
（实测 8x8x8x16 c64 相对误差 ~1e-7，单次调用快 ~10x）。

接口约定（经 mg_dev74_layout_test 实测确定）：
  * 输入/输出均为 [12, X, Y, Z, T/2]（spin×color 展平、奇子格）连续张量
  * 依赖 applyInitQcu(plan=1) 初始化的 LatticeSet（scratch: device_vec0/1/2 等）

多线程并发约定（一线程一卡）：
  * 每个 CudaSchurOp 实例持独立 params 副本（int32[54]）与独立 set_ptrs 副本，
    其中 _SET_INDEX_ 独占一个槽位 —— 各线程调用互不干扰，无共享写竞争。
  * 实例构造必须在单线程完成；每实例显存开销 ≈ LatticeSet scratch。
  * 设备绑定：构造时可指定 device；线程内首次调用前须 torch.cuda.set_device。
"""

import threading
import torch
from pyqcu.cuda import qcu
import pyqcu.cuda.define as define
from pyqcu.cuda.define import params as _module_params
from pyqcu.cuda.define import set_ptrs as _module_set_ptrs


class _SetIndexAllocator:
    """全局 set_index 槽位分配器（线程安全）。

    每个 CudaSchurOp 实例在共享 set_ptrs 中独占一个槽位
    （_SET_INDEX_ 对应 set_ptrs[_SET_INDEX_] = LatticeSet*）。
    """

    def __init__(self):
        self._counter = 0
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter


_GLOBAL_SET_ALLOC = _SetIndexAllocator()


class CudaSchurOp(object):
    """C++ CUDA 实现的 Schur 奇偶算子（matvec_parity 等价物）。

    matvec(x_o) -> y_o：x_o/y_o 为 [12,X,Y,Z,T/2] 奇子格场。
    构造即分配独立 LatticeSet（scratch 缓冲）；release() 释放。

    参数：
        av: argv 张量（float，real dtype，长度 7）
        g:  gauge 场 [2,3,3,4,X,Y,Z,T/2]（奇偶拆分）
        ce/coo: Clover 偶偶/奇奇项
        cei/coi: Clover 偶偶/奇奇逆
        device: 绑定的 torch 设备（多线程多卡模式下每线程一个 device）
    """

    def __init__(self, av, g, ce, coo, cei, coi, device=None):
        self.device = device if device is not None else torch.device('cuda')
        self.params = _module_params.clone()
        self.params[define._SET_INDEX_] = _GLOBAL_SET_ALLOC.next()
        self.set_index = int(self.params[define._SET_INDEX_])
        self.params[define._SET_PLAN_] = 1
        self.params[define._VERBOSE_] = 0
        # 独立 set_ptrs 副本：多线程各持一份，互不干扰
        self.set_ptrs = _module_set_ptrs.clone()
        self._g, self._ce, self._coo, self._cei, self._coi = g, ce, coo, cei, coi
        qcu.applyInitQcu(self.set_ptrs, self.params, av)

    def matvec(self, x_o):
        y_o = torch.empty_like(x_o)
        qcu.applyCloverBistabCgDslashQcu(
            y_o, x_o, self._g, self._ce, self._coo, self._cei, self._coi,
            self.set_ptrs, self.params)
        return y_o

    def release(self):
        """释放本实例 scratch：用本实例 params/set_ptrs 调 applyEndQcu。

        注意 applyEndQcu 释放 set_ptrs 中该槽位对应的 LatticeSet。
        """
        qcu.applyEndQcu(self.set_ptrs, self.params)
        self.set_index = None

    def __del__(self):
        try:
            if self.set_index is not None:
                self.release()
        except Exception:
            pass


def make_cuda_schur_ops(av, g, ce, coo, cei, coi, n=1, device=None, verbose=False):
    """创建 n 个互不冲突的 CudaSchurOp 实例（多线程各持一个）。单线程调用。"""
    ops = [CudaSchurOp(av, g, ce, coo, cei, coi, device=device) for _ in range(n)]
    if verbose:
        print(f"PYQCU::CUDA::SCHUR_OP:\n created {n} CudaSchurOp set_index="
              f"{[o.set_index for o in ops]}")
    return ops
