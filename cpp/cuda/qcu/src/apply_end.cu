#include "../include/qcu.h"
#include "../python/pyqcu.h"
#pragma optimize(5)
using namespace qcu;
void applyEndQcu(long long _set_ptrs, long long _params) {
  cudaDeviceSynchronize();
  void *set_ptrs = (void *)_set_ptrs;
  void *params = (void *)_params;
  int set_index = static_cast<int *>(params)[_SET_INDEX_];
  int data_type = static_cast<int *>(params)[_DATA_TYPE_];
  long long *table = static_cast<long long *>(set_ptrs);
  if (table[set_index] == 0) return;
  if (data_type == _LAT_C64_) {
    // end for lattice_set
    LatticeSet<float> *set_ptr = static_cast<LatticeSet<float> *>(
        (void *)(static_cast<long long *>(set_ptrs)[set_index]));
    if (set_ptr->host_params[_VERBOSE_]) {
      printf("set_ptr:%p\n", set_ptr);
      printf("set_ptrs:%p\n", set_ptrs);
      printf("long long set_ptr:%lld\n", (long long)set_ptr);
      auto start = std::chrono::high_resolution_clock::now();
      set_ptr->_print();
      set_ptr->end();
      delete set_ptr;
      auto end = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();
      cudaError_t err = cudaGetLastError();
      checkCudaErrors(err);
      printf("lattice set end total time:%.9lf "
             "sec\n",
             double(duration) / 1e9);
    } else {
      set_ptr->end();
      // BUGFIX 2026-07-28: delete host LatticeSet to prevent memory leak.
      // applyInitQcu allocates with 'new'; end() only frees GPU resources.
      delete set_ptr;
    }
  } else if (data_type == _LAT_C128_) {
    // end for lattice_set
    LatticeSet<double> *set_ptr = static_cast<LatticeSet<double> *>(
        (void *)(static_cast<long long *>(set_ptrs)[set_index]));
    if (set_ptr->host_params[_VERBOSE_]) {
      printf("set_ptr:%p\n", set_ptr);
      printf("set_ptrs:%p\n", set_ptrs);
      printf("long long set_ptr:%lld\n", (long long)set_ptr);
      auto start = std::chrono::high_resolution_clock::now();
      set_ptr->_print();
      set_ptr->end();
      delete set_ptr;
      auto end = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();
      cudaError_t err = cudaGetLastError();
      checkCudaErrors(err);
      printf("lattice set end total time:%.9lf "
             "sec\n",
             double(duration) / 1e9);
    } else {
      set_ptr->end();
      // BUGFIX 2026-07-28: delete host LatticeSet to prevent memory leak.
      // applyInitQcu allocates with 'new'; end() only frees GPU resources.
      delete set_ptr;
    }
  } else {
    printf("data_type error\n");
  }
  table[set_index] = 0;
  cudaDeviceSynchronize();
}
