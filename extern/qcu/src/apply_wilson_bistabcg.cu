#include "../python/pyqcu.h"
#include "../include/qcu.h"
#pragma optimize(5)
using namespace qcu;
using T = float;
void applyWilsonBistabCgQcu(long long _fermion_out, long long _fermion_in, long long _gauge, long long _set_ptrs, long long _params)
{
  cudaDeviceSynchronize();
  void *fermion_out = (void *)_fermion_out;
  void *fermion_in = (void *)_fermion_in;
  void *gauge = (void *)_gauge;
  void *set_ptrs = (void *)_set_ptrs;
  void *params = (void *)_params;
  int set_index = static_cast<int *>(params)[_SET_INDEX_];
  LatticeSet<T> *set_ptr = static_cast<LatticeSet<T> *>((void *)(static_cast<long long *>(set_ptrs)[set_index])); // define for apply_wilson_bistabcg
  // dptzyxcc2ccdptzyx<T>(gauge, &_set);
  // ptzyxsc2psctzyx<T>(fermion_in, &_set);
  // ptzyxsc2psctzyx<T>(fermion_out, &_set);
  LatticeBistabCg<T> _bistabcg;
  _bistabcg.give(set_ptr);
  _bistabcg.init(fermion_out, fermion_in, gauge);
  _bistabcg.run_test();
  _bistabcg.end();
  // ccdptzyx2dptzyxcc<T>(gauge, &_set);
  // psctzyx2ptzyxsc<T>(fermion_in, &_set);
  // psctzyx2ptzyxsc<T>(fermion_out, &_set);
  cudaDeviceSynchronize();
}