#include "../python/pyqcu.h"
#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
using T = float;
void applyWilsonBistabCgDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _set_ptrs, long long _params)
{
    cudaDeviceSynchronize();
    void *fermion_out = (void *)_fermion_out;
    void *fermion_in = (void *)_fermion_in;
    void *gauge = (void *)_gauge;
    void *set_ptrs = (void *)_set_ptrs;
    void *params = (void *)_params;
    int set_index = static_cast<int *>(params)[_SET_INDEX_];
    int data_type = static_cast<int *>(params)[_DATA_TYPE_];
    if (data_type == _LAT_C64_)
    {
        using T = float;
    }
    else if (data_type == _LAT_C128_)
    {
        using T = double;
    }
    // define for apply_wilson_dslash
    LatticeSet<T> *set_ptr = static_cast<LatticeSet<T> *>((void *)(static_cast<long long *>(set_ptrs)[set_index]));
    // dptzyxcc2ccdptzyx<T>(gauge, &_set);
    // tzyxsc2sctzyx<T>(fermion_in, &_set);
    // tzyxsc2sctzyx<T>(fermion_out, &_set);
    LatticeWilsonDslash<T> _wilson_dslash;
    _wilson_dslash.give(set_ptr);
    // { // test
    //     printf("fermion_out: %p\n", fermion_out);
    //     printf("fermion_in: %p\n", fermion_in);
    //     printf("gauge: %p\n", gauge);
    // }
    {
        void *device_vec0, *device_vec1, *device_vals;
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
        checkCudaErrors(cudaMallocAsync(
            &device_vec0, set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>),
            set_ptr->stream));
        checkCudaErrors(cudaMallocAsync(
            &device_vec1, set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>),
            set_ptr->stream));
        checkCudaErrors(cudaMallocAsync(
            &device_vals, _vals_size_ * sizeof(LatticeComplex<T>), set_ptr->stream));
        give_1custom<T><<<1, 1, 0, set_ptr->stream>>>(
            device_vals, _lat_4dim_, T(set_ptr->lat_4dim), 0.0);
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
        // src_o-set_ptr->kappa()**2*dslash_oe(dslash_eo(src_o))
        _wilson_dslash.run_eo(device_vec0, fermion_in, gauge);
        _wilson_dslash.run_oe(device_vec1, device_vec0, gauge);
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
        bistabcg_give_dest_o<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                                  set_ptr->stream>>>(
            fermion_out, fermion_in, device_vec1, set_ptr->kappa(), device_vals);
        checkCudaErrors(cudaFreeAsync(device_vec0, set_ptr->stream));
        checkCudaErrors(cudaFreeAsync(device_vec1, set_ptr->stream));
        checkCudaErrors(cudaFreeAsync(device_vals, set_ptr->stream));
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    }
    // ccdptzyx2dptzyxcc<T>(gauge, &_set);
    // sctzyx2tzyxsc<T>(fermion_in, &_set);
    // sctzyx2tzyxsc<T>(fermion_out, &_set);
    cudaDeviceSynchronize();
}