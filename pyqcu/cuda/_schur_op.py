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

    def __init__(self, av, g, ce, coo, cei, coi, device=None, params=None):
        self.device = device if device is not None else torch.device('cuda')
        if params is not None:
            self.params = params.clone()
        else:
            self.params = _module_params.clone()
        self.params[define._SET_INDEX_] = _GLOBAL_SET_ALLOC.next()
        self.set_index = int(self.params[define._SET_INDEX_])
        self.params[define._SET_PLAN_] = 1
        self.params[define._VERBOSE_] = 0
        # 独立 set_ptrs 副本：多线程各持一份，互不干扰
        self.set_ptrs = _module_set_ptrs.clone()
        self._g, self._ce, self._coo, self._cei, self._coi = g, ce, coo, cei, coi
        self._y_buf = None
        self._y_buf_shape = None
        qcu.applyInitQcu(self.set_ptrs, self.params, av)

    def matvec(self, x_o):
        # 输入/输出必须连续：桥函数 .contiguous() 会拷贝，若输出非连续则
        # C++ 写入副本、原张量内容不变（垃圾）导致求解错误。
        x_c = x_o.contiguous()
        # C++ 输出写预分配固定缓冲（不经过 torch 分配器，避免池复用与
        # C++ 私有流之间的深层竞争——实测 torch 池复用时偶发结果错误）；
        # 设备级同步保证 C++ 私有流完成，再 clone 到新张量（默认流语义）。
        if self._y_buf is None or self._y_buf_shape != tuple(x_c.shape):
            self._y_buf = torch.empty_like(x_c)
            self._y_buf_shape = tuple(x_c.shape)
        torch.cuda.current_stream().synchronize()
        qcu.applyCloverBistabCgDslashQcu(
            self._y_buf, x_c, self._g, self._ce, self._coo, self._cei, self._coi,
            self.set_ptrs, self.params)
        torch.cuda.synchronize()
        return self._y_buf.clone()

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


def make_cuda_schur_ops(av, g, ce, coo, cei, coi, n=1, device=None, params=None,
                        verbose=False):
    """创建 n 个互不冲突的 CudaSchurOp 实例（多线程各持一个）。单线程调用。

    params: 可选，与 gauge/clover 格点维度一致的 params 模板（克隆后分配
            独立 _SET_INDEX_ 槽位）；缺省用模块级 params（默认 32³ 格点，
            与 gauge/clover 维度不符会导致内核越界，务必传入）。
    """
    ops = [CudaSchurOp(av, g, ce, coo, cei, coi, device=device, params=params)
           for _ in range(n)]
    if verbose:
        print(f"PYQCU::CUDA::SCHUR_OP:\n created {n} CudaSchurOp set_index="
              f"{[o.set_index for o in ops]}")
    return ops


class CudaCoarseSchurOp(object):
    """C++ CUDA 宽版 33-tensor 粗层 Schur 算子（任意 DOF E）。

    matvec(x_c) -> y_c：x_c/y_c 为 [E, Xc, Yc, Zc, Tc] 粗层奇子格场，
    E 为粗层自由度（如 48）。内核 multigrid_coarse_dslash_wide（33-tensor：
    sit + hop_nn(±ward) + hop_diag(对角)），与 Python apply_stencil 等价
    （A_c = P^T S P 的 Schur 一致粗算子）。

    用途：build_schur_levels 的 lvl>=2 粗算子构建（null 向量生成与
    stencil 探测的 matvec）——把单线程 Python matvec 换成多线程 C++，
    加速 10-30 倍。

    构造约定（同 CudaSchurOp）：
      * 每实例独立 params 副本（int32[54]）与独立 set_ptrs 副本，
        _SET_INDEX_ 由全局分配器独占一个槽位。
      * 构造必须在单线程完成；调用前须 torch.cuda.set_device(绑定设备)。
      * 几何经 params 的 _MG_LEVEL1_* 槽位传递（C++ 从该处读取
        E/X/Y/Z/T）；stencil 张量须已在绑定设备上（.to(dev)）。
      * 构造即分配独立 LatticeSet；release() 释放。

    参数：
        av: argv 张量（float，real dtype，长度 7）
        E: 粗层自由度
        geo: [Xc, Yc, Zc, Tc] 粗层奇子格几何
        stencil: (sit, hop_nn, hop_diag) 33-tensor 三件套（绑定设备上）
        params: 可选 params 模板（克隆后写几何与 _SET_INDEX_）
    """

    def __init__(self, av, E, geo, stencil, device=None, params=None):
        self.device = device if device is not None else torch.device('cuda')
        if params is not None:
            self.params = params.clone()
        else:
            self.params = _module_params.clone()
        self.params[define._SET_INDEX_] = _GLOBAL_SET_ALLOC.next()
        self.set_index = int(self.params[define._SET_INDEX_])
        self.params[define._SET_PLAN_] = 1
        self.params[define._VERBOSE_] = 0
        # 几何（C++ 端从 _MG_LEVEL1_* 读取）
        self.params[define._MG_LEVEL1_E_] = int(E)
        self.params[define._MG_LEVEL1_X_] = int(geo[0])
        self.params[define._MG_LEVEL1_Y_] = int(geo[1])
        self.params[define._MG_LEVEL1_Z_] = int(geo[2])
        self.params[define._MG_LEVEL1_T_] = int(geo[3])
        # 独立 set_ptrs 副本：多线程各持一份，互不干扰
        self.set_ptrs = _module_set_ptrs.clone()
        self._sit, self._hop_nn, self._hop_diag = stencil
        self._y_buf = None
        self._y_buf_shape = None
        qcu.applyInitQcu(self.set_ptrs, self.params, av)

    def matvec(self, x_c):
        x_c = x_c.contiguous()
        if self._y_buf is None or self._y_buf_shape != tuple(x_c.shape):
            self._y_buf = torch.empty_like(x_c)
            self._y_buf_shape = tuple(x_c.shape)
        torch.cuda.current_stream().synchronize()
        qcu.applyMultigridCoarseDslashWideQcu(
            self._y_buf, x_c, self._sit, self._hop_nn, self._hop_diag,
            self.set_ptrs, self.params)
        torch.cuda.synchronize()
        return self._y_buf.clone()

    def release(self):
        qcu.applyEndQcu(self.set_ptrs, self.params)
        self.set_index = None

    def __del__(self):
        try:
            if self.set_index is not None:
                self.release()
        except Exception:
            pass
