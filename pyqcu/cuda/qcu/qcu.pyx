from qcu_api cimport applyInitQcu as _c_applyInitQcu, applyEndQcu as _c_applyEndQcu, testWilsonDslashQcu as _c_testWilsonDslashQcu, applyWilsonDslashQcu as _c_applyWilsonDslashQcu, testCloverDslashQcu as _c_testCloverDslashQcu, applyCloverDslashQcu as _c_applyCloverDslashQcu, applyWilsonBistabCgQcu as _c_applyWilsonBistabCgQcu, applyWilsonBistabCgDslashQcu as _c_applyWilsonBistabCgDslashQcu, applyWilsonCgQcu as _c_applyWilsonCgQcu, applyWilsonCgDslashQcu as _c_applyWilsonCgDslashQcu, applyLaplacianQcu as _c_applyLaplacianQcu, applyCloverQcu as _c_applyCloverQcu, applyCloversQcu as _c_applyCloversQcu, applyDslashQcu as _c_applyDslashQcu, applyGaussGaugeQcu as _c_applyGaussGaugeQcu, applyCloverBistabCgQcu as _c_applyCloverBistabCgQcu, applyCloverBistabCgDslashQcu as _c_applyCloverBistabCgDslashQcu, applyMultigridRestrictQcu as _c_applyMultigridRestrictQcu, applyMultigridProLongQcu as _c_applyMultigridProLongQcu, applyMultigridCoarseDslashQcu as _c_applyMultigridCoarseDslashQcu, applyMultigridCoarseDslashWideQcu as _c_applyMultigridCoarseDslashWideQcu, applyCloverMultigridQcu as _c_applyCloverMultigridQcu, verifyCloverMultigridQcu as _c_verifyCloverMultigridQcu
from qcu_api cimport applyMultigridStrictCoarseQcu as _c_applyMultigridStrictCoarseQcu, applyMultigridStrictMatPCQcu as _c_applyMultigridStrictMatPCQcu, applyMultigridStrictPrepareQcu as _c_applyMultigridStrictPrepareQcu, applyMultigridStrictReconstructQcu as _c_applyMultigridStrictReconstructQcu, applyMultigridStrictRestrictQcu as _c_applyMultigridStrictRestrictQcu, applyMultigridStrictProLongQcu as _c_applyMultigridStrictProLongQcu, applyMultigridStrictVCycleQcu as _c_applyMultigridStrictVCycleQcu, applyMultigridStrictInitQcu as _c_applyMultigridStrictInitQcu, applyMultigridStrictEndQcu as _c_applyMultigridStrictEndQcu
from qcu_api cimport applyCloverBistabCgPrepareQcu as _c_applyCloverBistabCgPrepareQcu, applyCloverBistabCgReconstructQcu as _c_applyCloverBistabCgReconstructQcu
# 多线程多卡（一线程一卡）约定：
#   * 所有桥函数在 GIL 段提取张量指针（.contiguous().data_ptr()），
#     随后 with nogil 释放 GIL 调用 C++ 后端 —— 多线程可真正并行进入
#     libqcu.so，各线程在各自绑定的 CUDA 设备上运行。
#   * 指针均声明为函数内局部 cdef 变量（栈变量，线程安全），
#     不再使用模块级共享 cdef 变量（曾依赖 GIL 串行化，见 2026-07-28 注释）。
#   * 调用者须保证：每线程独立 params/argv/set_ptrs 副本与设备绑定
#     （见 pyqcu/cuda/_multi_gpu.py、pyqcu/cuda/_schur_op.py）。
def applyInitQcu(_set_ptrs, _params, _argv):
    cdef long long set_ptrs, params, argv
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    argv = _argv.contiguous().data_ptr()
    with nogil:
        _c_applyInitQcu(set_ptrs, params, argv)
def applyEndQcu(_set_ptrs, _params):
    cdef long long set_ptrs, params
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    with nogil:
        _c_applyEndQcu(set_ptrs, params)
def testWilsonDslashQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    cdef long long fermion_out, fermion_in, gauge, set_ptrs, params
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    with nogil:
        _c_testWilsonDslashQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def applyWilsonDslashQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    cdef long long fermion_out, fermion_in, gauge, set_ptrs, params
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    with nogil:
        _c_applyWilsonDslashQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def testCloverDslashQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    cdef long long fermion_out, fermion_in, gauge, set_ptrs, params
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    with nogil:
        _c_testCloverDslashQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def applyCloverDslashQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    cdef long long fermion_out, fermion_in, gauge, set_ptrs, params
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    with nogil:
        _c_applyCloverDslashQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def applyWilsonBistabCgQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    cdef long long fermion_out, fermion_in, gauge, set_ptrs, params
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    with nogil:
        _c_applyWilsonBistabCgQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def applyWilsonBistabCgDslashQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    cdef long long fermion_out, fermion_in, gauge, set_ptrs, params
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    with nogil:
        _c_applyWilsonBistabCgDslashQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def applyWilsonCgQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    cdef long long fermion_out, fermion_in, gauge, set_ptrs, params
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    with nogil:
        _c_applyWilsonCgQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def applyWilsonCgDslashQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    cdef long long fermion_out, fermion_in, gauge, set_ptrs, params
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    with nogil:
        _c_applyWilsonCgDslashQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def applyLaplacianQcu(_laplacian_out, _laplacian_in, _gauge, _set_ptrs, _params):
    cdef long long laplacian_out, laplacian_in, gauge, set_ptrs, params
    laplacian_out = _laplacian_out.contiguous().data_ptr()
    laplacian_in = _laplacian_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    with nogil:
        _c_applyLaplacianQcu(laplacian_out, laplacian_in, gauge, set_ptrs, params)
def applyCloverQcu(_clover, _gauge, _set_ptrs, _params):
    cdef long long clover, gauge, set_ptrs, params
    clover = _clover.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    with nogil:
        _c_applyCloverQcu(clover, gauge, set_ptrs, params)
def applyCloversQcu(_clover, _clover_inv, _gauge, _set_ptrs, _params):
    cdef long long clover, clover_inv, gauge, set_ptrs, params
    clover = _clover.contiguous().data_ptr()
    clover_inv = _clover_inv.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    with nogil:
        _c_applyCloversQcu(clover, clover_inv, gauge, set_ptrs, params)
def applyDslashQcu(_fermion_out, _fermion_in, _gauge, _clover, _set_ptrs, _params):
    cdef long long fermion_out, fermion_in, clover, gauge, set_ptrs, params
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    clover = _clover.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    with nogil:
        _c_applyDslashQcu(fermion_out, fermion_in, gauge, clover, set_ptrs, params)
def applyGaussGaugeQcu(_gauge, _set_ptrs, _params):
    cdef long long gauge, set_ptrs, params
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    with nogil:
        _c_applyGaussGaugeQcu(gauge, set_ptrs, params)
def applyCloverBistabCgQcu(_fermion_out, _fermion_in, _gauge, _clover_ee, _clover_oo, _clover_ee_inv, _clover_oo_inv, _set_ptrs, _params):
    cdef long long fermion_out, fermion_in, gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv, set_ptrs, params
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    clover_ee = _clover_ee.contiguous().data_ptr()
    clover_oo = _clover_oo.contiguous().data_ptr()
    clover_ee_inv = _clover_ee_inv.contiguous().data_ptr()
    clover_oo_inv = _clover_oo_inv.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    with nogil:
        _c_applyCloverBistabCgQcu(fermion_out, fermion_in, gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv, set_ptrs, params)
def applyCloverBistabCgDslashQcu(_fermion_out, _fermion_in, _gauge, _clover_ee, _clover_oo, _clover_ee_inv, _clover_oo_inv, _set_ptrs, _params):
    cdef long long fermion_out, fermion_in, gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv, set_ptrs, params
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    clover_ee = _clover_ee.contiguous().data_ptr()
    clover_oo = _clover_oo.contiguous().data_ptr()
    clover_ee_inv = _clover_ee_inv.contiguous().data_ptr()
    clover_oo_inv = _clover_oo_inv.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    with nogil:
        _c_applyCloverBistabCgDslashQcu(fermion_out, fermion_in, gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv, set_ptrs, params)
def applyCloverBistabCgPrepareQcu(_compact_rhs, _full_rhs, _gauge, _clover_ee, _clover_oo, _clover_ee_inv, _clover_oo_inv, _set_ptrs, _params):
    """Prepare the odd symmetric-Schur rhs from a parity-split full rhs."""
    cdef long long compact_rhs, full_rhs, gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv, set_ptrs, params
    cdef int status
    if int(_compact_rhs.numel()) * 2 != int(_full_rhs.numel()):
        raise ValueError("compact Schur rhs must contain half of full_rhs")
    if not all(value.is_contiguous() for value in
               (_compact_rhs, _full_rhs, _gauge, _clover_ee, _clover_oo,
                _clover_ee_inv, _clover_oo_inv, _set_ptrs, _params)):
        raise ValueError("Clover Schur prepare tensors must be contiguous")
    compact_rhs = _compact_rhs.data_ptr()
    full_rhs = _full_rhs.data_ptr()
    gauge = _gauge.data_ptr()
    clover_ee = _clover_ee.data_ptr()
    clover_oo = _clover_oo.data_ptr()
    clover_ee_inv = _clover_ee_inv.data_ptr()
    clover_oo_inv = _clover_oo_inv.data_ptr()
    set_ptrs = _set_ptrs.data_ptr()
    params = _params.data_ptr()
    with nogil:
        status = _c_applyCloverBistabCgPrepareQcu(
            compact_rhs, full_rhs, gauge, clover_ee, clover_oo,
            clover_ee_inv, clover_oo_inv, set_ptrs, params)
    if status != 0:
        raise RuntimeError("applyCloverBistabCgPrepareQcu failed")
def applyCloverBistabCgReconstructQcu(_full_out, _full_rhs, _target_odd, _gauge, _clover_ee, _clover_oo, _clover_ee_inv, _clover_oo_inv, _set_ptrs, _params):
    """Reconstruct the even field from an odd Schur solution."""
    cdef long long full_out, full_rhs, target_odd, gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv, set_ptrs, params
    cdef int status
    if tuple(_full_out.shape) != tuple(_full_rhs.shape):
        raise ValueError("Clover reconstruct full fields must share shape")
    if int(_target_odd.numel()) * 2 != int(_full_rhs.numel()):
        raise ValueError("target_odd must contain half of full_rhs")
    if not all(value.is_contiguous() for value in
               (_full_out, _full_rhs, _target_odd, _gauge, _clover_ee,
                _clover_oo, _clover_ee_inv, _clover_oo_inv,
                _set_ptrs, _params)):
        raise ValueError("Clover reconstruct tensors must be contiguous")
    full_out = _full_out.data_ptr()
    full_rhs = _full_rhs.data_ptr()
    target_odd = _target_odd.data_ptr()
    gauge = _gauge.data_ptr()
    clover_ee = _clover_ee.data_ptr()
    clover_oo = _clover_oo.data_ptr()
    clover_ee_inv = _clover_ee_inv.data_ptr()
    clover_oo_inv = _clover_oo_inv.data_ptr()
    set_ptrs = _set_ptrs.data_ptr()
    params = _params.data_ptr()
    with nogil:
        status = _c_applyCloverBistabCgReconstructQcu(
            full_out, full_rhs, target_odd, gauge, clover_ee, clover_oo,
            clover_ee_inv, clover_oo_inv, set_ptrs, params)
    if status != 0:
        raise RuntimeError("applyCloverBistabCgReconstructQcu failed")
def applyMultigridRestrictQcu(_coarse_out, _fine_in, _null_vecs, _set_ptrs, _params):
    cdef long long coarse_out, fine_in, null_vecs, set_ptrs, params
    coarse_out = _coarse_out.contiguous().data_ptr()
    fine_in = _fine_in.contiguous().data_ptr()
    null_vecs = _null_vecs.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    with nogil:
        _c_applyMultigridRestrictQcu(coarse_out, fine_in, null_vecs, set_ptrs, params)
def applyMultigridProLongQcu(_fine_out, _coarse_in, _null_vecs, _set_ptrs, _params):
    cdef long long fine_out, coarse_in, null_vecs, set_ptrs, params
    fine_out = _fine_out.contiguous().data_ptr()
    coarse_in = _coarse_in.contiguous().data_ptr()
    null_vecs = _null_vecs.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    with nogil:
        _c_applyMultigridProLongQcu(fine_out, coarse_in, null_vecs, set_ptrs, params)
def applyMultigridCoarseDslashQcu(_fermion_out, _fermion_in, _hopping, _sitting, _set_ptrs, _params):
    cdef long long fermion_out, fermion_in, hopping, sitting, set_ptrs, params
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    hopping = _hopping.contiguous().data_ptr()
    sitting = _sitting.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    with nogil:
        _c_applyMultigridCoarseDslashQcu(fermion_out, fermion_in, hopping, sitting, set_ptrs, params)
def applyMultigridCoarseDslashWideQcu(_fermion_out, _fermion_in, _sitting, _hop_nn, _hop_diag, _set_ptrs, _params):
    cdef long long fermion_out, fermion_in, sitting, hop_nn, hop_diag, set_ptrs, params
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    sitting = _sitting.contiguous().data_ptr()
    hop_nn = _hop_nn.contiguous().data_ptr()
    hop_diag = _hop_diag.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    with nogil:
        _c_applyMultigridCoarseDslashWideQcu(fermion_out, fermion_in, sitting, hop_nn, hop_diag, set_ptrs, params)
def applyMultigridStrictCoarseQcu(_fermion_out, _fermion_in, _links, _onsite_pair, _set_ptrs, _params, int onsite_index=-1):
    """Apply strict QUDA-stored ``Y``/``Yhat`` links on a full coarse field."""
    cdef long long fermion_out, fermion_in, links, onsite_pair, set_ptrs, params
    cdef int E, X, Y, Z, T, status
    if _fermion_in.ndim != 5 or tuple(_fermion_out.shape) != tuple(_fermion_in.shape):
        raise ValueError("strict coarse fields must share shape [E,X,Y,Z,T]")
    E, X, Y, Z, T = [int(value) for value in _fermion_in.shape]
    if tuple(_links.shape) != (2, 4, E, E, X, Y, Z, T):
        raise ValueError("strict links must have shape [2,4,E,E,X,Y,Z,T]")
    if tuple(_onsite_pair.shape) != (2, E, E, X, Y, Z, T):
        raise ValueError("strict onsite_pair must have shape [2,E,E,X,Y,Z,T]")
    if not all(value.is_contiguous() for value in
               (_fermion_out, _fermion_in, _links, _onsite_pair, _set_ptrs, _params)):
        raise ValueError("strict coarse tensors must be contiguous")
    fermion_out = _fermion_out.data_ptr()
    fermion_in = _fermion_in.data_ptr()
    links = _links.data_ptr()
    onsite_pair = _onsite_pair.data_ptr()
    set_ptrs = _set_ptrs.data_ptr()
    params = _params.data_ptr()
    with nogil:
        status = _c_applyMultigridStrictCoarseQcu(fermion_out, fermion_in, links, onsite_pair, set_ptrs, params, E, X, Y, Z, T, onsite_index)
    if status != 0:
        raise RuntimeError("applyMultigridStrictCoarseQcu failed")
def applyMultigridStrictMatPCQcu(_fermion_out, _fermion_in, _links, _scratch, _set_ptrs, _params, int parity):
    """Apply ``I-Hhat_pq Hhat_qp`` in compact checkerboard layout."""
    cdef long long fermion_out, fermion_in, links, scratch, set_ptrs, params
    cdef int E, X, Y, Z, T, status
    if _fermion_in.ndim != 5 or tuple(_fermion_out.shape) != tuple(_fermion_in.shape) or tuple(_scratch.shape) != tuple(_fermion_in.shape):
        raise ValueError("strict MATPC fields must share shape [E,X,Y,Z,T/2]")
    E = int(_fermion_in.shape[0])
    X, Y, Z = [int(value) for value in _fermion_in.shape[1:4]]
    T = 2 * int(_fermion_in.shape[4])
    if tuple(_links.shape) != (2, 4, E, E, X, Y, Z, T):
        raise ValueError("strict MATPC links must have full-lattice shape")
    if not all(value.is_contiguous() for value in
               (_fermion_out, _fermion_in, _links, _scratch, _set_ptrs, _params)):
        raise ValueError("strict MATPC tensors must be contiguous")
    fermion_out = _fermion_out.data_ptr()
    fermion_in = _fermion_in.data_ptr()
    links = _links.data_ptr()
    scratch = _scratch.data_ptr()
    set_ptrs = _set_ptrs.data_ptr()
    params = _params.data_ptr()
    with nogil:
        status = _c_applyMultigridStrictMatPCQcu(fermion_out, fermion_in, links, scratch, set_ptrs, params, E, X, Y, Z, T, parity)
    if status != 0:
        raise RuntimeError("applyMultigridStrictMatPCQcu failed")
def applyMultigridStrictPrepareQcu(_fermion_out, _full_rhs, _links, _onsite_pair, _scratch, _set_ptrs, _params, int parity):
    """Prepare ``X_p^-1(b_p-H_pq X_q^-1 b_q)`` from a full rhs."""
    cdef long long fermion_out, full_rhs, links, onsite_pair, scratch, set_ptrs, params
    cdef int E, X, Y, Z, T, status
    if _full_rhs.ndim != 5:
        raise ValueError("strict prepare rhs must have shape [E,X,Y,Z,T]")
    E, X, Y, Z, T = [int(value) for value in _full_rhs.shape]
    expected_compact = (E, X, Y, Z, T // 2)
    if tuple(_fermion_out.shape) != expected_compact or tuple(_scratch.shape) != expected_compact:
        raise ValueError("strict prepare output/scratch must use compact parity shape")
    if tuple(_links.shape) != (2, 4, E, E, X, Y, Z, T):
        raise ValueError("strict prepare links must have full-lattice shape")
    if tuple(_onsite_pair.shape) != (2, E, E, X, Y, Z, T):
        raise ValueError("strict prepare onsite_pair has invalid shape")
    if not all(value.is_contiguous() for value in
               (_fermion_out, _full_rhs, _links, _onsite_pair, _scratch, _set_ptrs, _params)):
        raise ValueError("strict prepare tensors must be contiguous")
    fermion_out = _fermion_out.data_ptr()
    full_rhs = _full_rhs.data_ptr()
    links = _links.data_ptr()
    onsite_pair = _onsite_pair.data_ptr()
    scratch = _scratch.data_ptr()
    set_ptrs = _set_ptrs.data_ptr()
    params = _params.data_ptr()
    with nogil:
        status = _c_applyMultigridStrictPrepareQcu(fermion_out, full_rhs, links, onsite_pair, scratch, set_ptrs, params, E, X, Y, Z, T, parity)
    if status != 0:
        raise RuntimeError("applyMultigridStrictPrepareQcu failed")
def applyMultigridStrictReconstructQcu(_full_out, _full_rhs, _target_solution, _links, _onsite_pair, _scratch, _set_ptrs, _params, int parity):
    """Reconstruct the eliminated parity from a compact target solution."""
    cdef long long full_out, full_rhs, target_solution, links, onsite_pair, scratch, set_ptrs, params
    cdef int E, X, Y, Z, T, status
    if _full_rhs.ndim != 5 or tuple(_full_out.shape) != tuple(_full_rhs.shape):
        raise ValueError("strict reconstruct full fields must share [E,X,Y,Z,T]")
    E, X, Y, Z, T = [int(value) for value in _full_rhs.shape]
    expected_compact = (E, X, Y, Z, T // 2)
    if tuple(_target_solution.shape) != expected_compact or tuple(_scratch.shape) != expected_compact:
        raise ValueError("strict reconstruct target/scratch must use compact parity shape")
    if tuple(_links.shape) != (2, 4, E, E, X, Y, Z, T):
        raise ValueError("strict reconstruct links must have full-lattice shape")
    if tuple(_onsite_pair.shape) != (2, E, E, X, Y, Z, T):
        raise ValueError("strict reconstruct onsite_pair has invalid shape")
    if not all(value.is_contiguous() for value in
               (_full_out, _full_rhs, _target_solution, _links, _onsite_pair,
                _scratch, _set_ptrs, _params)):
        raise ValueError("strict reconstruct tensors must be contiguous")
    full_out = _full_out.data_ptr()
    full_rhs = _full_rhs.data_ptr()
    target_solution = _target_solution.data_ptr()
    links = _links.data_ptr()
    onsite_pair = _onsite_pair.data_ptr()
    scratch = _scratch.data_ptr()
    set_ptrs = _set_ptrs.data_ptr()
    params = _params.data_ptr()
    with nogil:
        status = _c_applyMultigridStrictReconstructQcu(full_out, full_rhs, target_solution, links, onsite_pair, scratch, set_ptrs, params, E, X, Y, Z, T, parity)
    if status != 0:
        raise RuntimeError("applyMultigridStrictReconstructQcu failed")
def applyMultigridStrictRestrictQcu(_coarse_out, _fine_in, _null_vectors, _set_ptrs, _params, int parity):
    """Restrict one compact fine parity to a full coarse field."""
    cdef long long coarse_out, fine_in, null_vectors, set_ptrs, params
    cdef int E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc, status
    if _null_vectors.ndim != 10:
        raise ValueError("strict null_vectors must use 10-D blocked layout")
    E, e = int(_null_vectors.shape[0]), int(_null_vectors.shape[1])
    Xc, Yc, Zc, Tc = [int(_null_vectors.shape[index]) for index in (2, 4, 6, 8)]
    Xf = Xc * int(_null_vectors.shape[3])
    Yf = Yc * int(_null_vectors.shape[5])
    Zf = Zc * int(_null_vectors.shape[7])
    Tf = Tc * int(_null_vectors.shape[9])
    if tuple(_fine_in.shape) != (e, Xf, Yf, Zf, Tf // 2):
        raise ValueError("strict fine input must have shape [e,Xf,Yf,Zf,Tf/2]")
    if tuple(_coarse_out.shape) != (E, Xc, Yc, Zc, Tc):
        raise ValueError("strict coarse output must have full coarse shape")
    if not all(value.is_contiguous() for value in
               (_coarse_out, _fine_in, _null_vectors, _set_ptrs, _params)):
        raise ValueError("strict restrict tensors must be contiguous")
    coarse_out = _coarse_out.data_ptr()
    fine_in = _fine_in.data_ptr()
    null_vectors = _null_vectors.data_ptr()
    set_ptrs = _set_ptrs.data_ptr()
    params = _params.data_ptr()
    with nogil:
        status = _c_applyMultigridStrictRestrictQcu(coarse_out, fine_in, null_vectors, set_ptrs, params, E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc, parity)
    if status != 0:
        raise RuntimeError("applyMultigridStrictRestrictQcu failed")
def applyMultigridStrictProLongQcu(_fine_out, _coarse_in, _null_vectors, _set_ptrs, _params, int parity):
    """Prolong a full coarse field to one compact fine parity."""
    cdef long long fine_out, coarse_in, null_vectors, set_ptrs, params
    cdef int E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc, status
    if _null_vectors.ndim != 10:
        raise ValueError("strict null_vectors must use 10-D blocked layout")
    E, e = int(_null_vectors.shape[0]), int(_null_vectors.shape[1])
    Xc, Yc, Zc, Tc = [int(_null_vectors.shape[index]) for index in (2, 4, 6, 8)]
    Xf = Xc * int(_null_vectors.shape[3])
    Yf = Yc * int(_null_vectors.shape[5])
    Zf = Zc * int(_null_vectors.shape[7])
    Tf = Tc * int(_null_vectors.shape[9])
    if tuple(_fine_out.shape) != (e, Xf, Yf, Zf, Tf // 2):
        raise ValueError("strict fine output must have shape [e,Xf,Yf,Zf,Tf/2]")
    if tuple(_coarse_in.shape) != (E, Xc, Yc, Zc, Tc):
        raise ValueError("strict coarse input must have full coarse shape")
    if not all(value.is_contiguous() for value in
               (_fine_out, _coarse_in, _null_vectors, _set_ptrs, _params)):
        raise ValueError("strict prolong tensors must be contiguous")
    fine_out = _fine_out.data_ptr()
    coarse_in = _coarse_in.data_ptr()
    null_vectors = _null_vectors.data_ptr()
    set_ptrs = _set_ptrs.data_ptr()
    params = _params.data_ptr()
    with nogil:
        status = _c_applyMultigridStrictProLongQcu(fine_out, coarse_in, null_vectors, set_ptrs, params, E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc, parity)
    if status != 0:
        raise RuntimeError("applyMultigridStrictProLongQcu failed")
def applyMultigridStrictVCycleQcu(_full_out, _full_rhs, _set_ptrs, _params, int start_level=1):
    """Run an arena-backed recursive strict coarse V-cycle.

    Returns the exact number of bytes allocated for transient hierarchy state;
    resident ``V/Yhat/(X,Xinv)`` assets are owned by the caller and excluded.
    """
    cdef long long full_out, full_rhs, set_ptrs, params
    cdef unsigned long long allocated_bytes = 0
    cdef int status
    if (_full_rhs.ndim != 5 or
            tuple(_full_out.shape) != tuple(_full_rhs.shape)):
        raise ValueError(
            "strict V-cycle fields must share full shape [E,X,Y,Z,T]")
    if start_level < 1 or start_level > 4:
        raise ValueError("strict V-cycle start_level must be in [1,4]")
    if not all(value.is_contiguous() for value in
               (_full_out, _full_rhs, _set_ptrs, _params)):
        raise ValueError("strict V-cycle tensors must be contiguous")
    full_out = _full_out.data_ptr()
    full_rhs = _full_rhs.data_ptr()
    set_ptrs = _set_ptrs.data_ptr()
    params = _params.data_ptr()
    with nogil:
        status = _c_applyMultigridStrictVCycleQcu(
            full_out, full_rhs, set_ptrs, params, start_level,
            &allocated_bytes)
    if status != 0:
        raise RuntimeError("applyMultigridStrictVCycleQcu failed")
    return int(allocated_bytes)
def applyMultigridStrictInitQcu(_set_ptrs, _params, int start_level=1):
    """Allocate one reusable strict coarse hierarchy and return its bytes."""
    cdef long long set_ptrs, params
    cdef unsigned long long allocated_bytes = 0
    cdef int status
    if start_level < 1 or start_level > 4:
        raise ValueError("strict hierarchy start_level must be in [1,4]")
    if not _set_ptrs.is_contiguous() or not _params.is_contiguous():
        raise ValueError("strict hierarchy control tensors must be contiguous")
    set_ptrs = _set_ptrs.data_ptr()
    params = _params.data_ptr()
    with nogil:
        status = _c_applyMultigridStrictInitQcu(
            set_ptrs, params, start_level, &allocated_bytes)
    if status != 0:
        raise RuntimeError("applyMultigridStrictInitQcu failed")
    return int(allocated_bytes)
def applyMultigridStrictEndQcu(_set_ptrs, _params):
    """Release a reusable strict hierarchy; safe when already released."""
    cdef long long set_ptrs, params
    cdef int status
    if not _set_ptrs.is_contiguous() or not _params.is_contiguous():
        raise ValueError("strict hierarchy control tensors must be contiguous")
    set_ptrs = _set_ptrs.data_ptr()
    params = _params.data_ptr()
    with nogil:
        status = _c_applyMultigridStrictEndQcu(set_ptrs, params)
    if status != 0:
        raise RuntimeError("applyMultigridStrictEndQcu failed")
def applyCloverMultigridQcu(_fermion_out, _fermion_in, _gauge, _clover_ee, _clover_oo, _clover_ee_inv, _clover_oo_inv, _set_ptrs, _params):
    cdef long long fermion_out, fermion_in, gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv, set_ptrs, params
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    clover_ee = _clover_ee.contiguous().data_ptr()
    clover_oo = _clover_oo.contiguous().data_ptr()
    clover_ee_inv = _clover_ee_inv.contiguous().data_ptr()
    clover_oo_inv = _clover_oo_inv.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    with nogil:
        _c_applyCloverMultigridQcu(fermion_out, fermion_in, gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv, set_ptrs, params)
def verifyCloverMultigridQcu(_fermion_out, _fermion_in, _gauge, _clover_ee, _clover_oo, _clover_ee_inv, _clover_oo_inv, _set_ptrs, _params):
    """Run the five C++ multigrid consistency diagnostics.

    Returns 0 when all diagnostics pass, 1 when a diagnostic fails, and 2
    when the C++ bridge rejects the call or catches an exception.  The
    caller must still invoke :func:`applyEndQcu` for the lattice set.
    """
    cdef long long fermion_out, fermion_in, gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv, set_ptrs, params
    cdef int status
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    clover_ee = _clover_ee.contiguous().data_ptr()
    clover_oo = _clover_oo.contiguous().data_ptr()
    clover_ee_inv = _clover_ee_inv.contiguous().data_ptr()
    clover_oo_inv = _clover_oo_inv.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    with nogil:
        status = _c_verifyCloverMultigridQcu(fermion_out, fermion_in, gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv, set_ptrs, params)
    return status
