#include "../python/pyqcu.h"
#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
using T = float;
void applyLaplacianQcu(long long _laplacian_out, long long _laplacian_in, long long _gauge, long long _set_ptrs, long long _params)
{
  cudaDeviceSynchronize();
  void *laplacian_out = (void *)_laplacian_out;
  void *laplacian_in = (void *)_laplacian_in;
  void *gauge = (void *)_gauge;
  void *set_ptrs = (void *)_set_ptrs;
  void *params = (void *)_params;
  int set_index = static_cast<int *>(params)[_SET_INDEX_];
  LatticeSet<T> *set_ptr = static_cast<LatticeSet<T> *>((void *)(static_cast<long long *>(set_ptrs)[set_index])); // define for apply_laplacian
  // dptzyxcc2ccdptzyx<T>(gauge, &_set);
  // tzyxsc2sctzyx<T>(laplacian_in, &_set);
  // tzyxsc2sctzyx<T>(laplacian_out, &_set);
  LatticeLaplacian<T> _laplacian;
  _laplacian.give(set_ptr);
  _laplacian.run_test(laplacian_out, laplacian_in, gauge);
  // ccdptzyx2dptzyxcc<T>(gauge, &_set);
  // sctzyx2tzyxsc<T>(laplacian_in, &_set);
  // sctzyx2tzyxsc<T>(laplacian_out, &_set);
  cudaDeviceSynchronize();
}