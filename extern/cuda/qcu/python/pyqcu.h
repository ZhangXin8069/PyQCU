#ifndef _PYQCU_H
#define _PYQCU_H
#pragma once
#ifdef __cplusplus
extern "C"
{
#endif
    void applyInitQcu(long long _set_ptrs, long long _params, long long _argv);
    void applyEndQcu(long long _set_ptrs, long long _params);
    void testWilsonDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _set_ptrs, long long _params);
    void applyWilsonDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _set_ptrs, long long _params);
    void testCloverDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _set_ptrs, long long _params);
    void applyCloverDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _set_ptrs, long long _params);
    void applyWilsonBistabCgQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _set_ptrs, long long _params);
    void applyWilsonBistabCgDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _set_ptrs, long long _params);
    void applyWilsonCgQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _set_ptrs, long long _params);
    void applyWilsonCgDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _set_ptrs, long long _params);
    void applyLaplacianQcu(long long _laplacian_out, long long _laplacian_in, long long _gauge, long long _set_ptrs, long long _params);
    void applyCloverQcu(long long _clover, long long _gauge, long long _set_ptrs, long long _params);
    void applyDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _clover, long long _set_ptrs, long long _params);
    void applyGaussGaugeQcu(long long _gauge, long long _set_ptrs, long long _params);
    void applyCloverBistabCgQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _clover, long long _set_ptrs, long long _params);
#ifdef __cplusplus
}
#endif
#endif