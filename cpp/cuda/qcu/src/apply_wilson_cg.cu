#include "../include/qcu.h"
#include "../python/pyqcu.h"
#pragma optimize(5)
using namespace qcu;
void applyWilsonCgQcu(long long _fermion_out, long long _fermion_in,
                      long long _gauge, long long _set_ptrs,
                      long long _params) {
  cudaDeviceSynchronize();
  void *fermion_out = (void *)_fermion_out;
  void *fermion_in = (void *)_fermion_in;
  void *gauge = (void *)_gauge;
  void *set_ptrs = (void *)_set_ptrs;
  void *params = (void *)_params;
  int set_index = static_cast<int *>(params)[_SET_INDEX_];
  int data_type = static_cast<int *>(params)[_DATA_TYPE_];
  if (data_type == _LAT_C64_) {
    LatticeSet<float> *set_ptr =
        static_cast<LatticeSet<float> *>((void *)(static_cast<long long *>(
            set_ptrs)[set_index])); // define for apply_wilson_cg
    LatticeWilsonCg<float> _cg;
    _cg.give(set_ptr);
    _cg.init(fermion_out, fermion_in, gauge);
    if (set_ptr->host_params[_VERBOSE_]) {
      _cg.run_test();
    } else {
      _cg.run();
    }
    _cg.end();
  } else if (data_type == _LAT_C128_) {
    LatticeSet<double> *set_ptr =
        static_cast<LatticeSet<double> *>((void *)(static_cast<long long *>(
            set_ptrs)[set_index])); // define for apply_wilson_cg
    LatticeWilsonCg<double> _cg;
    _cg.give(set_ptr);
    _cg.init(fermion_out, fermion_in, gauge);
    if (set_ptr->host_params[_VERBOSE_]) {
      _cg.run_test();
    } else {
      _cg.run();
    }
    _cg.end();
  } else {
    printf("data_type error\n");
  }
  cudaDeviceSynchronize();
}