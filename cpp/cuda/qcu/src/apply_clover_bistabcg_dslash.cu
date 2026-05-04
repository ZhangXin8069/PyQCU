#include "../include/qcu.h"
#include "../python/pyqcu.h"
#pragma optimize(5)
using namespace qcu;
void applyCloverBistabCgDslashQcu(long long _fermion_out, long long _fermion_in,
                                  long long _gauge, long long _clover_ee,
                                  long long _clover_oo,
                                  long long _clover_ee_inv,
                                  long long _clover_oo_inv, long long _set_ptrs,
                                  long long _params) {
  cudaDeviceSynchronize();
  void *fermion_out = (void *)_fermion_out;
  void *fermion_in = (void *)_fermion_in;
  void *gauge = (void *)_gauge;
  void *clover_ee = (void *)_clover_ee;
  void *clover_oo = (void *)_clover_oo;
  void *clover_ee_inv = (void *)_clover_ee_inv;
  void *clover_oo_inv = (void *)_clover_oo_inv;
  void *set_ptrs = (void *)_set_ptrs;
  void *params = (void *)_params;
  int set_index = static_cast<int *>(params)[_SET_INDEX_];
  int data_type = static_cast<int *>(params)[_DATA_TYPE_];
  if (data_type == _LAT_C64_) {
    LatticeSet<float> *set_ptr =
        static_cast<LatticeSet<float> *>((void *)(static_cast<long long *>(
            set_ptrs)[set_index])); // define for apply_clover_bistabcg
    LatticeCloverBistabCg<float> _bistabcg;
    _bistabcg.give(set_ptr);
    _bistabcg.init(gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv);
    _bistabcg.dslash(fermion_out, fermion_in);

  } else if (data_type == _LAT_C128_) {
    LatticeSet<double> *set_ptr =
        static_cast<LatticeSet<double> *>((void *)(static_cast<long long *>(
            set_ptrs)[set_index])); // define for apply_clover_bistabcg
    LatticeCloverBistabCg<double> _bistabcg;
    _bistabcg.give(set_ptr);
    _bistabcg.init(gauge, clover_ee, clover_oo, clover_ee_inv, clover_oo_inv);
    _bistabcg.dslash(fermion_out, fermion_in);
  } else {
    printf("data_type error\n");
  }
  cudaDeviceSynchronize();
}