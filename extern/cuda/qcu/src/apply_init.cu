#include "../python/pyqcu.h"
#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
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
        // init for lattice_set
        LatticeSet<float> *set_ptr = new LatticeSet<float>();
        set_ptr->give(params, argv);
        if (set_ptr->host_params[_VERBOSE_])
        {
            printf("set_ptr:%p\n", set_ptr);
            printf("long long set_ptr:%lld\n", (long long)set_ptr);
            auto start = std::chrono::high_resolution_clock::now();
            set_ptr->init();
            set_ptr->_print();
            static_cast<long long *>(set_ptrs)[set_index] = (long long)set_ptr;
            auto end = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            cudaError_t err = cudaGetLastError();
            checkCudaErrors(err);
            printf("lattice set init total time:%.9lf "
                   "sec\n",
                   double(duration) / 1e9);
        }
        else
        {
            set_ptr->give(params, argv);
            set_ptr->init();
        }
    }
    else if (data_type == _LAT_C128_)
    {
        // init for lattice_set
        LatticeSet<double> *set_ptr = new LatticeSet<double>();
        set_ptr->give(params, argv);
        if (set_ptr->host_params[_VERBOSE_])
        {
            printf("set_ptr:%p\n", set_ptr);
            printf("long long set_ptr:%lld\n", (long long)set_ptr);
            auto start = std::chrono::high_resolution_clock::now();
            set_ptr->init();
            set_ptr->_print();
            static_cast<long long *>(set_ptrs)[set_index] = (long long)set_ptr;
            auto end = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            cudaError_t err = cudaGetLastError();
            checkCudaErrors(err);
            printf("lattice set init total time:%.9lf "
                   "sec\n",
                   double(duration) / 1e9);
        }
        else
        {
            set_ptr->give(params, argv);
            set_ptr->init();
        }
    }
    else
    {
        printf("data_type error\n");
    }
    cudaDeviceSynchronize();
}