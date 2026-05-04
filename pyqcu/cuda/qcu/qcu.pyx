cimport qcu
cdef long long set_ptrs
cdef long long fermion_out
cdef long long fermion_in
cdef long long clover
cdef long long clover_inv
cdef long long clover_ee
cdef long long clover_oo
cdef long long clover_ee_inv
cdef long long clover_oo_inv
cdef long long gauge
cdef long long params
cdef long long argv
def applyInitQcu(_set_ptrs, _params, _argv):
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    argv = _argv.contiguous().data_ptr()
    qcu.applyInitQcu(set_ptrs, params, argv);
def applyEndQcu(_set_ptrs, _params):
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    qcu.applyEndQcu(set_ptrs, params);
def testWilsonDslashQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    qcu.testWilsonDslashQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def applyWilsonDslashQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    qcu.applyWilsonDslashQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def testCloverDslashQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    qcu.testCloverDslashQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def applyCloverDslashQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    qcu.applyCloverDslashQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def applyWilsonBistabCgQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    qcu.applyWilsonBistabCgQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def applyWilsonBistabCgDslashQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    qcu.applyWilsonBistabCgDslashQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def applyWilsonCgQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    qcu.applyWilsonCgQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def applyWilsonCgDslashQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs, _params):
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    qcu.applyWilsonCgDslashQcu(fermion_out, fermion_in, gauge, set_ptrs, params)
def applyLaplacianQcu(_laplacian_out, _laplacian_in, _gauge, _set_ptrs, _params):
    laplacian_out = _laplacian_out.contiguous().data_ptr()
    laplacian_in = _laplacian_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    qcu.applyLaplacianQcu(laplacian_out, laplacian_in, gauge, set_ptrs, params)
def applyCloverQcu(_clover, _gauge, _set_ptrs, _params):
    clover = _clover.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    qcu.applyCloverQcu(clover, gauge, set_ptrs, params)
def applyCloversQcu(_clover, _clover_inv, _gauge, _set_ptrs, _params):
    clover = _clover.contiguous().data_ptr()
    clover_inv = _clover_inv.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    qcu.applyCloversQcu(clover, clover_inv, gauge, set_ptrs, params)
def applyDslashQcu(_fermion_out, _fermion_in, _gauge, _clover, _set_ptrs, _params):
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    clover = _clover.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    qcu.applyDslashQcu(fermion_out, fermion_in, gauge, clover, set_ptrs, params)
def applyGaussGaugeQcu(_gauge, _set_ptrs, _params):
    gauge = _gauge.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    qcu.applyGaussGaugeQcu(gauge, set_ptrs, params)
def applyCloverBistabCgQcu(_fermion_out, _fermion_in, _gauge, _clover_ee, _clover_oo, _clover_ee_inv, _clover_oo_inv, _set_ptrs, _params):
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    clover_ee = _clover_ee.contiguous().data_ptr()
    clover_oo = _clover_oo.contiguous().data_ptr()
    clover_ee_inv = _clover_ee_inv.contiguous().data_ptr()
    clover_oo_inv = _clover_oo_inv.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    qcu.applyCloverBistabCgQcu(fermion_out, fermion_in, gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv, set_ptrs, params)
def applyCloverBistabCgDslashQcu(_fermion_out, _fermion_in, _gauge, _clover_ee, _clover_oo, _clover_ee_inv, _clover_oo_inv, _set_ptrs, _params):
    fermion_out = _fermion_out.contiguous().data_ptr()
    fermion_in = _fermion_in.contiguous().data_ptr()
    gauge = _gauge.contiguous().data_ptr()
    clover_ee = _clover_ee.contiguous().data_ptr()
    clover_oo = _clover_oo.contiguous().data_ptr()
    clover_ee_inv = _clover_ee_inv.contiguous().data_ptr()
    clover_oo_inv = _clover_oo_inv.contiguous().data_ptr()
    set_ptrs = _set_ptrs.contiguous().data_ptr()
    params = _params.contiguous().data_ptr()
    qcu.applyCloverBistabCgQcu(fermion_out, fermion_in, gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv, set_ptrs, params)