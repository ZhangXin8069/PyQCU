#include "../python/pyqcu.h"
#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
void applyEndQcu(long long _set_ptrs, long long _params)
{
    cudaDeviceSynchronize();
    void *set_ptrs = (void *)_set_ptrs;
    void *params = (void *)_params;
    int set_index = static_cast<int *>(params)[_SET_INDEX_];
    int data_type = static_cast<int *>(params)[_DATA_TYPE_];
    if (data_type == _LAT_C64_)
    {
        // end for lattice_set
        LatticeSet<float> *set_ptr = static_cast<LatticeSet<float> *>((void *)(static_cast<long long *>(set_ptrs)[set_index]));
        if (set_ptr->host_params[_VERBOSE_])
        {
            printf("set_ptr:%p\n", set_ptr);
            printf("long long set_ptr:%lld\n", (long long)set_ptr);
            auto start = std::chrono::high_resolution_clock::now();
            set_ptr->_print();
            set_ptr->end();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            cudaError_t err = cudaGetLastError();
            checkCudaErrors(err);
            printf("lattice set end total time:%.9lf "
                   "sec\n",
                   double(duration) / 1e9);
        }
        else
        {
            set_ptr->end();
        }
    }
    else if (data_type == _LAT_C128_)
    {
        // end for lattice_set
        LatticeSet<double> *set_ptr = static_cast<LatticeSet<double> *>((void *)(static_cast<long long *>(set_ptrs)[set_index]));
        if (set_ptr->host_params[_VERBOSE_])
        {
            printf("set_ptr:%p\n", set_ptr);
            printf("long long set_ptr:%lld\n", (long long)set_ptr);
            auto start = std::chrono::high_resolution_clock::now();
            set_ptr->_print();
            set_ptr->end();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            cudaError_t err = cudaGetLastError();
            checkCudaErrors(err);
            printf("lattice set end total time:%.9lf "
                   "sec\n",
                   double(duration) / 1e9);
        }
        else
        {
            set_ptr->end();
        }
    }
    else
    {
        printf("data_type error\n");
    }
    cudaDeviceSynchronize();
}