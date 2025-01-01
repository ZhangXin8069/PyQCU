#include "../python/pyqcu.h"
#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
using T = float;
void applyEndQcu(long long _set_ptrs, long long _params)
{
    void *set_ptrs = (void *)_set_ptrs;
    void *params = (void *)_params;
    int set_index = static_cast<int *>(params)[_SET_INDEX_];
    // end for lattice_set
    LatticeSet<T> *set_ptr = static_cast<LatticeSet<T> *>((void *)(static_cast<long long *>(set_ptrs)[set_index]));
    auto start = std::chrono::high_resolution_clock::now();
    set_ptr->_print();
    set_ptr->end();
    printf("set_ptr:%p\n", set_ptr);
    printf("long long set_ptr:%lld\n", (long long)set_ptr);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    cudaError_t err = cudaGetLastError();
    checkCudaErrors(err);
    printf("lattice set end total time:%.9lf "
           "sec\n",
           double(duration) / 1e9);
}