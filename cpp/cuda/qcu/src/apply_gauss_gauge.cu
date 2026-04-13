#include "../python/pyqcu.h"
#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
void applyGaussGaugeQcu(long long _gauge, long long _set_ptrs, long long _params)
{
  cudaDeviceSynchronize();
  void *gauge = (void *)_gauge;
  void *set_ptrs = (void *)_set_ptrs;
  void *params = (void *)_params;
  int set_index = static_cast<int *>(params)[_SET_INDEX_];
  int data_type = static_cast<int *>(params)[_DATA_TYPE_];
  if (data_type == _LAT_C64_)
  {
    void *set_ptr = (void *)(static_cast<long long *>(set_ptrs)[set_index]); // define for apply_gauss_gauge
    make_gauss_gauge<float>(gauge, set_ptr);
  }
  else if (data_type == _LAT_C128_)
  {
    void *set_ptr = (void *)(static_cast<long long *>(set_ptrs)[set_index]); // define for apply_gauss_gauge
    make_gauss_gauge<double>(gauge, set_ptr);
  }
  else
  {
    printf("data_type error\n");
  }
  cudaDeviceSynchronize();
}