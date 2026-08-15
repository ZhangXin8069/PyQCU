from qcu_api cimport applyInitQcu as _c_applyInitQcu, applyEndQcu as _c_applyEndQcu, testWilsonDslashQcu as _c_testWilsonDslashQcu, applyWilsonDslashQcu as _c_applyWilsonDslashQcu, testCloverDslashQcu as _c_testCloverDslashQcu, applyCloverDslashQcu as _c_applyCloverDslashQcu, applyWilsonBistabCgQcu as _c_applyWilsonBistabCgQcu, applyWilsonBistabCgDslashQcu as _c_applyWilsonBistabCgDslashQcu, applyWilsonCgQcu as _c_applyWilsonCgQcu, applyWilsonCgDslashQcu as _c_applyWilsonCgDslashQcu, applyLaplacianQcu as _c_applyLaplacianQcu, applyCloverQcu as _c_applyCloverQcu, applyCloversQcu as _c_applyCloversQcu, applyDslashQcu as _c_applyDslashQcu, applyGaussGaugeQcu as _c_applyGaussGaugeQcu, applyCloverBistabCgQcu as _c_applyCloverBistabCgQcu, applyCloverBistabCgDslashQcu as _c_applyCloverBistabCgDslashQcu, applyMultigridRestrictQcu as _c_applyMultigridRestrictQcu, applyMultigridProLongQcu as _c_applyMultigridProLongQcu, applyMultigridCoarseDslashQcu as _c_applyMultigridCoarseDslashQcu, applyMultigridCoarseDslashWideQcu as _c_applyMultigridCoarseDslashWideQcu, applyCloverMultigridQcu as _c_applyCloverMultigridQcu
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
