#include "../python/pyqcu.h"
#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
void applyLaplacianQcu(long long _laplacian_out, long long _laplacian_in, long long _gauge, long long _set_ptrs, long long _params)
{
  cudaDeviceSynchronize();
  void *laplacian_out = (void *)_laplacian_out;
  void *laplacian_in = (void *)_laplacian_in;
  void *gauge = (void *)_gauge;
  void *set_ptrs = (void *)_set_ptrs;
  void *params = (void *)_params;
  int set_index = static_cast<int *>(params)[_SET_INDEX_];
  int data_type = static_cast<int *>(params)[_DATA_TYPE_];
  if (data_type == _LAT_C64_)
  {
    LatticeSet<float> *set_ptr = static_cast<LatticeSet<float> *>((void *)(static_cast<long long *>(set_ptrs)[set_index])); // define for apply_laplacian
    // dptzyxcc2ccdptzyx<float>(gauge, &_set);
    // tzyxsc2sctzyx<float>(laplacian_in, &_set);
    // tzyxsc2sctzyx<float>(laplacian_out, &_set);
    LatticeLaplacian<float> _laplacian;
    _laplacian.give(set_ptr);
    _laplacian.run_test(laplacian_out, laplacian_in, gauge);
    // ccdptzyx2dptzyxcc<float>(gauge, &_set);
    // sctzyx2tzyxsc<float>(laplacian_in, &_set);
    // sctzyx2tzyxsc<float>(laplacian_out, &_set);
  }
  else if (data_type == _LAT_C128_)
  {
    LatticeSet<double> *set_ptr = static_cast<LatticeSet<double> *>((void *)(static_cast<long long *>(set_ptrs)[set_index])); // define for apply_laplacian
    // dptzyxcc2ccdptzyx<double>(gauge, &_set);
    // tzyxsc2sctzyx<double>(laplacian_in, &_set);
    // tzyxsc2sctzyx<double>(laplacian_out, &_set);
    LatticeLaplacian<double> _laplacian;
    _laplacian.give(set_ptr);
    _laplacian.run_test(laplacian_out, laplacian_in, gauge);
    // ccdptzyx2dptzyxcc<double>(gauge, &_set);
    // sctzyx2tzyxsc<double>(laplacian_in, &_set);
    // sctzyx2tzyxsc<double>(laplacian_out, &_set);
  }
  else
  {
    printf("data_type error\n");
  }
  cudaDeviceSynchronize();
}