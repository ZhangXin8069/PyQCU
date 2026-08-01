#include "../include/qcu.h"
#include "../include/lattice_clover_multigrid.h"
#include "../python/pyqcu.h"
#pragma optimize(5)
using namespace qcu;

/**
 * @brief C++ CUDA Clover Multigrid solver entry point.
 *
 * The set_ptrs array (int64[100]) is used to pass coarse-grid operator data
 * using the following convention:
 *   set_ptrs[_SET_PTRS_COARSE_BASE_ + 3*fl + 0] = null_vecs for fine level fl→fl+1
 *   set_ptrs[_SET_PTRS_COARSE_BASE_ + 3*fl + 1] = hop_packed for coarse level fl+1
 *   set_ptrs[_SET_PTRS_COARSE_BASE_ + 3*fl + 2] = sit_packed for coarse level fl+1
 *
 * where _SET_PTRS_COARSE_BASE_ = 10 (first free slot in set_ptrs after scratch ptrs).
 * For 3 levels (fl=0,1), entries 10-15 are used.
 */
#define _SET_PTRS_COARSE_BASE_ 10

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
  long long *sp = static_cast<long long *>(set_ptrs);

  if (data_type == _LAT_C64_) {
    LatticeSet<float> *set_ptr =
        static_cast<LatticeSet<float> *>((void *)(sp[set_index]));
    LatticeCloverMultigrid<float> _multigrid;
    _multigrid.give(set_ptr);
    _multigrid.init(fermion_out, fermion_in, gauge, clover_ee, clover_oo,
                    clover_ee_inv, clover_oo_inv);

    // ---- Wire up coarse-grid operators from set_ptrs ----
    // For each fine level fl, read null_vecs, hop_packed, sit_packed.
    // These must be set by the Python caller before calling this function.
    int num_levels = set_ptr->host_params[_MG_NUM_LEVEL_];
    for (int fl = 0; fl < num_levels - 1; fl++) {
      int base = _SET_PTRS_COARSE_BASE_ + 3 * fl;
      void *nv = (void *)sp[base + 0];
      void *hp = (void *)sp[base + 1];
      void *sp_sit = (void *)sp[base + 2];
      if (nv && hp && sp_sit) {
        _multigrid.set_coarse_ops(fl, nv, hp, sp_sit);
        if (set_ptr->host_params[_VERBOSE_] && set_ptr->host_params[_NODE_RANK_] == 0) {
          printf("MG: set coarse ops level %d→%d: nv=%p, hp=%p, sit=%p\n",
                 fl, fl+1, nv, hp, sp_sit);
        }
      }
    }

    if (set_ptr->host_params[_VERBOSE_]) {
      printf("=== MULTIGRID VERBOSE MODE ===\n");
      printf("num_levels: %d\n", num_levels);
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
        static_cast<LatticeSet<double> *>((void *)(sp[set_index]));
    LatticeCloverMultigrid<double> _multigrid;
    _multigrid.give(set_ptr);
    _multigrid.init(fermion_out, fermion_in, gauge, clover_ee, clover_oo,
                    clover_ee_inv, clover_oo_inv);

    // ---- Wire up coarse-grid operators from set_ptrs ----
    int num_levels = set_ptr->host_params[_MG_NUM_LEVEL_];
    for (int fl = 0; fl < num_levels - 1; fl++) {
      int base = _SET_PTRS_COARSE_BASE_ + 3 * fl;
      void *nv = (void *)sp[base + 0];
      void *hp = (void *)sp[base + 1];
      void *sp_sit = (void *)sp[base + 2];
      if (nv && hp && sp_sit) {
        _multigrid.set_coarse_ops(fl, nv, hp, sp_sit);
        if (set_ptr->host_params[_VERBOSE_] && set_ptr->host_params[_NODE_RANK_] == 0) {
          printf("MG: set coarse ops level %d→%d (double)\n", fl, fl+1);
        }
      }
    }

    if (set_ptr->host_params[_VERBOSE_]) {
      printf("=== MULTIGRID VERBOSE MODE (double) ===\n");
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
