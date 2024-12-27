#include "../python/pyqcu.h"
#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
using T = float;
void testWilsonDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _params, long long _argv)
{
    void *fermion_out = (void *)_fermion_out;
    void *fermion_in = (void *)_fermion_in;
    void *gauge = (void *)_gauge;
    void *argv = (void *)_argv;
    void *params = (void *)_params;
    // define for test_wilson_dslash
    LatticeSet<T> _set;
    _set.give(params, argv);
    _set.init();
    dptzyxcc2ccdptzyx<T>(gauge, &_set);
    tzyxsc2sctzyx<T>(fermion_in, &_set);
    tzyxsc2sctzyx<T>(fermion_out, &_set);
    auto start = std::chrono::high_resolution_clock::now();
    wilson_dslash<T><<<_set.gridDim, _set.blockDim>>>(gauge, fermion_in, fermion_out,
                                                      _set.device_params);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    cudaError_t err = cudaGetLastError();
    checkCudaErrors(err);
    printf("wilson dslash total time: (without malloc free memcpy) :%.9lf "
           "sec\n",
           double(duration) / 1e9);
    ccdptzyx2dptzyxcc<T>(gauge, &_set);
    sctzyx2tzyxsc<T>(fermion_in, &_set);
    sctzyx2tzyxsc<T>(fermion_out, &_set);
    _set.end();
}
void testCloverDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _params, long long _argv)
{
    void *fermion_out = (void *)_fermion_out;
    void *fermion_in = (void *)_fermion_in;
    void *gauge = (void *)_gauge;
    void *argv = (void *)_argv;
    void *params = (void *)_params;
    // define for test_clover_dslash
    LatticeSet<T> _set;
    _set.give(params, argv);
    _set.init();
    dptzyxcc2ccdptzyx<T>(gauge, &_set);
    tzyxsc2sctzyx<T>(fermion_in, &_set);
    tzyxsc2sctzyx<T>(fermion_out, &_set);
    LatticeWilsonDslash<T> _wilson_dslash;
    _wilson_dslash.give(&_set);
    void* clover;
    checkCudaErrors(cudaMallocAsync(
        &clover, (_set.lat_4dim * _LAT_SCSC_) * sizeof(LatticeComplex<T>),
        _set.stream));
    cudaError_t err;
    {
        // wilson dslash
        _wilson_dslash.run_test(fermion_out, fermion_in, gauge);
    }
    {
        // make clover
        checkCudaErrors(cudaStreamSynchronize(_set.stream));
        auto start = std::chrono::high_resolution_clock::now();
        make_clover<T><<<_set.gridDim, _set.blockDim, 0, _set.stream>>>(
            gauge, clover, _set.device_params);
        checkCudaErrors(cudaStreamSynchronize(_set.stream));
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count();
        err = cudaGetLastError();
        checkCudaErrors(err);
        printf("make clover total time: (without malloc free memcpy) :%.9lf sec\n ",
               double(duration) / 1e9);
    }
    {
        // inverse clover
        checkCudaErrors(cudaStreamSynchronize(_set.stream));
        auto start = std::chrono::high_resolution_clock::now();
        inverse_clover<T><<<_set.gridDim, _set.blockDim, 0, _set.stream>>>(
            clover, _set.device_params);
        checkCudaErrors(cudaStreamSynchronize(_set.stream));
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count();
        err = cudaGetLastError();
        checkCudaErrors(err);
        printf(
            "inverse clover total time: (without malloc free memcpy) :%.9lf sec\n ",
            double(duration) / 1e9);
    }
    {
        // give clover
        checkCudaErrors(cudaStreamSynchronize(_set.stream));
        auto start = std::chrono::high_resolution_clock::now();
        give_clover<T><<<_set.gridDim, _set.blockDim, 0, _set.stream>>>(
            clover, fermion_out, _set.device_params);
        checkCudaErrors(cudaStreamSynchronize(_set.stream));
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                .count();
        err = cudaGetLastError();
        checkCudaErrors(err);
        printf("give clover total time: (without malloc free memcpy) :%.9lf sec\n ",
               double(duration) / 1e9);
    }
    ccdptzyx2dptzyxcc<T>(gauge, &_set);
    sctzyx2tzyxsc<T>(fermion_in, &_set);
    sctzyx2tzyxsc<T>(fermion_out, &_set);
    // free
    checkCudaErrors(cudaFreeAsync(clover, _set.stream));
    checkCudaErrors(cudaStreamSynchronize(_set.stream));
    _set.end();
}
