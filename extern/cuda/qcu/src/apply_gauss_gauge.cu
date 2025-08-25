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
    LatticeSet<float> *set_ptr = static_cast<LatticeSet<float> *>((void *)(static_cast<long long *>(set_ptrs)[set_index])); // define for apply_gauss_gauge
    // dptzyxcc2ccdptzyx<float>(gauge, &_set);
    make_gauss_gauge<float>(gauge, params, set_ptr->sigma());
    // ccdptzyx2dptzyxcc<float>(gauge, &_set);
    _gauss_gauge.end();
  }
  else if (data_type == _LAT_C128_)
  {
    LatticeSet<double> *set_ptr = static_cast<LatticeSet<double> *>((void *)(static_cast<long long *>(set_ptrs)[set_index])); // define for apply_gauss_gauge
    // dptzyxcc2ccdptzyx<double>(gauge, &_set);
    make_gauss_gauge<float>(gauge, params, set_ptr->sigma());
    // ccdptzyx2dptzyxcc<float>(gauge, &_set);
    _gauss_gauge.end();
  }
  else
  {
    printf("data_type error\n");
  }
  cudaDeviceSynchronize();
}