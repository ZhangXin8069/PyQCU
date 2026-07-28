#include "../include/qcu.h"
#include "../include/lattice_clover_multigrid.h"
#include "../python/pyqcu.h"
#pragma optimize(5)
using namespace qcu;

void applyCloverMultigridQcu(long long _fermion_out, long long _fermion_in,
                              long long _gauge, long long _clover_ee,
                              long long _clover_oo,
                              long long _clover_ee_inv,
                              long long _clover_oo_inv,
                              long long _set_ptrs,
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
            set_ptrs)[set_index]));
    LatticeCloverMultigrid<float> _multigrid;
    _multigrid.give(set_ptr);
    _multigrid.init(fermion_out, fermion_in, gauge, clover_ee, clover_oo,
                    clover_ee_inv, clover_oo_inv);
    if (set_ptr->host_params[_VERBOSE_]) {
      printf("=== MULTIGRID VERBOSE MODE ===\n");
      printf("fermion_out: %lld\n", (long long)fermion_out);
      printf("fermion_in: %lld\n", (long long)fermion_in);
      printf("gauge: %lld\n", (long long)gauge);
      printf("clover_ee: %lld\n", (long long)clover_ee);
      printf("clover_oo: %lld\n", (long long)clover_oo);
      printf("clover_ee_inv: %lld\n", (long long)clover_ee_inv);
      printf("clover_oo_inv: %lld\n", (long long)clover_oo_inv);
      _multigrid.run_test();
    } else {
      _multigrid.run();
    }
    _multigrid.end();
  } else if (data_type == _LAT_C128_) {
    LatticeSet<double> *set_ptr =
        static_cast<LatticeSet<double> *>((void *)(static_cast<long long *>(
            set_ptrs)[set_index]));
    LatticeCloverMultigrid<double> _multigrid;
    _multigrid.give(set_ptr);
    _multigrid.init(fermion_out, fermion_in, gauge, clover_ee, clover_oo,
                    clover_ee_inv, clover_oo_inv);
    if (set_ptr->host_params[_VERBOSE_]) {
      printf("=== MULTIGRID VERBOSE MODE (double) ===\n");
      printf("fermion_out: %lld\n", (long long)fermion_out);
      printf("fermion_in: %lld\n", (long long)fermion_in);
      printf("gauge: %lld\n", (long long)gauge);
      _multigrid.run_test();
    } else {
      _multigrid.run();
    }
    _multigrid.end();
  } else {
    printf("applyCloverMultigridQcu: unsupported data_type=%d\n", data_type);
  }
  cudaDeviceSynchronize();
}
