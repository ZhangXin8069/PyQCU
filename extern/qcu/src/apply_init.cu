#include "../python/pyqcu.h"
#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
using T = float;
void applyInitQcu(long long _set_ptrs, long long _params, long long _argv)
{
    cudaDeviceSynchronize();
    void *set_ptrs = (void *)_set_ptrs;
    void *argv = (void *)_argv;
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
    // init for lattice_set
    LatticeSet<T> *set_ptr = new LatticeSet<T>();
    auto start = std::chrono::high_resolution_clock::now();
    set_ptr->give(params, argv);
    set_ptr->init();
    set_ptr->_print();
    printf("set_ptr:%p\n", set_ptr);
    printf("long long set_ptr:%lld\n", (long long)set_ptr);
    static_cast<long long *>(set_ptrs)[set_index] = (long long)set_ptr;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    cudaError_t err = cudaGetLastError();
    checkCudaErrors(err);
    printf("lattice set init total time:%.9lf "
           "sec\n",
           double(duration) / 1e9);
    cudaDeviceSynchronize();
}