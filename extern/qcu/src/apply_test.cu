#include "../python/pyqcu.h"
#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
void testWilsonDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _set_ptrs, long long _params)
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
        // define for test_wilson_dslash
        LatticeSet<float> *set_ptr = static_cast<LatticeSet<float> *>((void *)(static_cast<long long *>(set_ptrs)[set_index]));
        // dptzyxcc2ccdptzyx<float>(gauge, &_set);
        // tzyxsc2sctzyx<float>(fermion_in, &_set);
        // tzyxsc2sctzyx<float>(fermion_out, &_set);
        auto start = std::chrono::high_resolution_clock::now();
        wilson_dslash<float><<<set_ptr->gridDim, set_ptr->blockDim>>>(gauge, fermion_in, fermion_out,
                                                                      set_ptr->device_params);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        cudaError_t err = cudaGetLastError();
        checkCudaErrors(err);
        printf("wilson dslash total time: (without malloc free memcpy) :%.9lf "
               "sec\n",
               double(duration) / 1e9);
        // ccdptzyx2dptzyxcc<float>(gauge, &_set);
        // sctzyx2tzyxsc<float>(fermion_in, &_set);
        // sctzyx2tzyxsc<float>(fermion_out, &_set);
    }
    else if (data_type == _LAT_C128_)
    {
        // define for test_wilson_dslash
        LatticeSet<double> *set_ptr = static_cast<LatticeSet<double> *>((void *)(static_cast<long long *>(set_ptrs)[set_index]));
        // dptzyxcc2ccdptzyx<double>(gauge, &_set);
        // tzyxsc2sctzyx<double>(fermion_in, &_set);
        // tzyxsc2sctzyx<double>(fermion_out, &_set);
        auto start = std::chrono::high_resolution_clock::now();
        wilson_dslash<double><<<set_ptr->gridDim, set_ptr->blockDim>>>(gauge, fermion_in, fermion_out,
                                                                       set_ptr->device_params);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        cudaError_t err = cudaGetLastError();
        checkCudaErrors(err);
        printf("wilson dslash total time: (without malloc free memcpy) :%.9lf "
               "sec\n",
               double(duration) / 1e9);
        // ccdptzyx2dptzyxcc<double>(gauge, &_set);
        // sctzyx2tzyxsc<double>(fermion_in, &_set);
        // sctzyx2tzyxsc<double>(fermion_out, &_set);
    }
    else
    {
        printf("data_type error\n");
    }
    cudaDeviceSynchronize();
}
void testCloverDslashQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _set_ptrs, long long _params)
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
        // define for test_clover_dslash
        LatticeSet<float> *set_ptr = static_cast<LatticeSet<float> *>((void *)(static_cast<long long *>(set_ptrs)[set_index]));
        // dptzyxcc2ccdptzyx<float>(gauge, &_set);
        // tzyxsc2sctzyx<float>(fermion_in, &_set);
        // tzyxsc2sctzyx<float>(fermion_out, &_set);
        LatticeWilsonDslash<float> _wilson_dslash;
        _wilson_dslash.give(set_ptr);
        void *clover;
        checkCudaErrors(cudaMallocAsync(
            &clover, (set_ptr->lat_4dim * _LAT_SCSC_) * sizeof(LatticeComplex<float>),
            set_ptr->stream));
        cudaError_t err;
        {
            // wilson dslash
            _wilson_dslash.run_test(fermion_out, fermion_in, gauge);
        }
        {
            // make clover
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
            auto start = std::chrono::high_resolution_clock::now();
            make_clover<float><<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
                gauge, clover, set_ptr->device_params);
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
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
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
            auto start = std::chrono::high_resolution_clock::now();
            inverse_clover<float><<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
                clover, set_ptr->device_params);
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
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
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
            auto start = std::chrono::high_resolution_clock::now();
            give_clover<float><<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
                clover, fermion_out, set_ptr->device_params);
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
            auto end = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                    .count();
            err = cudaGetLastError();
            checkCudaErrors(err);
            printf("give clover total time: (without malloc free memcpy) :%.9lf sec\n ",
                   double(duration) / 1e9);
        }
        // ccdptzyx2dptzyxcc<float>(gauge, &_set);
        // sctzyx2tzyxsc<float>(fermion_in, &_set);
        // sctzyx2tzyxsc<float>(fermion_out, &_set);
        // free
        checkCudaErrors(cudaFreeAsync(clover, set_ptr->stream));
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    }
    else if (data_type == _LAT_C128_)
    {
        // define for test_clover_dslash
        LatticeSet<double> *set_ptr = static_cast<LatticeSet<double> *>((void *)(static_cast<long long *>(set_ptrs)[set_index]));
        // dptzyxcc2ccdptzyx<double>(gauge, &_set);
        // tzyxsc2sctzyx<double>(fermion_in, &_set);
        // tzyxsc2sctzyx<double>(fermion_out, &_set);
        LatticeWilsonDslash<double> _wilson_dslash;
        _wilson_dslash.give(set_ptr);
        void *clover;
        checkCudaErrors(cudaMallocAsync(
            &clover, (set_ptr->lat_4dim * _LAT_SCSC_) * sizeof(LatticeComplex<double>),
            set_ptr->stream));
        cudaError_t err;
        {
            // wilson dslash
            _wilson_dslash.run_test(fermion_out, fermion_in, gauge);
        }
        {
            // make clover
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
            auto start = std::chrono::high_resolution_clock::now();
            make_clover<double><<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
                gauge, clover, set_ptr->device_params);
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
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
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
            auto start = std::chrono::high_resolution_clock::now();
            inverse_clover<double><<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
                clover, set_ptr->device_params);
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
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
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
            auto start = std::chrono::high_resolution_clock::now();
            give_clover<double><<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
                clover, fermion_out, set_ptr->device_params);
            checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
            auto end = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                    .count();
            err = cudaGetLastError();
            checkCudaErrors(err);
            printf("give clover total time: (without malloc free memcpy) :%.9lf sec\n ",
                   double(duration) / 1e9);
        }
        // ccdptzyx2dptzyxcc<double>(gauge, &_set);
        // sctzyx2tzyxsc<double>(fermion_in, &_set);
        // sctzyx2tzyxsc<double>(fermion_out, &_set);
        // free
        checkCudaErrors(cudaFreeAsync(clover, set_ptr->stream));
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    }
    else
    {
        printf("data_type error\n");
    }
    cudaDeviceSynchronize();
}
