#include "../include/qcu.h"
#pragma optimize(5)
namespace qcu
{
  template <typename T>
  __global__ void give_param(void *device_param, int vals_index, int val)
  {
    int *param = static_cast<int *>(device_param);
    param[vals_index] = val;
  }
  //@@@CUDA_TEMPLATE_FOR_DEVICE@@@
  template __global__ void give_param<double>(void *device_param, int vals_index, int val);
  template __global__ void give_param<float>(void *device_param, int vals_index, int val);
}