#include "../python/pyqcu.h"
#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
using T = float;
void applyWilsonBistabCgDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _params, long long _argv)
{
    void *fermion_out = (void *)_fermion_out;
    void *fermion_in = (void *)_fermion_in;
    void *gauge = (void *)_gauge;
    void *argv = (void *)_argv;
    void *params = (void *)_params;
    // define for apply_wilson_dslash
    LatticeSet<T> _set;
    _set.give(params, argv);
    _set.init();
    // dptzyxcc2ccdptzyx<T>(gauge, &_set);
    // tzyxsc2sctzyx<T>(fermion_in, &_set);
    // tzyxsc2sctzyx<T>(fermion_out, &_set);
    LatticeWilsonDslash<T> _wilson_dslash;
    _wilson_dslash.give(&_set);
    // { // test
    //     printf("fermion_out: %p\n", fermion_out);
    //     printf("fermion_in: %p\n", fermion_in);
    //     printf("gauge: %p\n", gauge);
    // }
    {
        void *device_vec0, *device_vec1, *device_vals;
        checkCudaErrors(cudaStreamSynchronize(_set.stream));
        checkCudaErrors(cudaMallocAsync(
            &device_vec0, _set.lat_4dim_SC * sizeof(LatticeComplex<T>),
            _set.stream));
        checkCudaErrors(cudaMallocAsync(
            &device_vec1, _set.lat_4dim_SC * sizeof(LatticeComplex<T>),
            _set.stream));
        checkCudaErrors(cudaMallocAsync(
            &device_vals, _vals_size_ * sizeof(LatticeComplex<T>), _set.stream));
        give_1custom<T><<<1, 1, 0, _set.stream>>>(
            device_vals, _lat_4dim_, T(_set.lat_4dim), 0.0);
        checkCudaErrors(cudaStreamSynchronize(_set.stream));
        // src_o-_set.kappa()**2*dslash_oe(dslash_eo(src_o))
        _wilson_dslash.run_eo(device_vec0, fermion_in, gauge);
        _wilson_dslash.run_oe(device_vec1, device_vec0, gauge);
        checkCudaErrors(cudaStreamSynchronize(_set.stream));
        bistabcg_give_dest_o<T><<<_set.gridDim, _set.blockDim, 0,
                                  _set.stream>>>(
            fermion_out, fermion_in, device_vec1, _set.kappa(), device_vals);
        checkCudaErrors(cudaFreeAsync(device_vec0, _set.stream));
        checkCudaErrors(cudaFreeAsync(device_vec1, _set.stream));
        checkCudaErrors(cudaFreeAsync(device_vals, _set.stream));
        checkCudaErrors(cudaStreamSynchronize(_set.stream));
    }
    // ccdptzyx2dptzyxcc<T>(gauge, &_set);
    // sctzyx2tzyxsc<T>(fermion_in, &_set);
    // sctzyx2tzyxsc<T>(fermion_out, &_set);
    _set.end();
}