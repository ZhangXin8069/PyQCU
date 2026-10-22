#include "../include/qcu.h"
#include "../include/lattice_clover_multigrid.h"
#include "../python/pyqcu.h"
#include <chrono>
#pragma optimize(5)
using namespace qcu;

template <typename T>
static void warn_legacy_multigrid_depth(LatticeSet<T> *set_ptr) {
  if (set_ptr == nullptr ||
      set_ptr->host_params[_NODE_RANK_] != 0) {
    return;
  }
  const int num_levels = set_ptr->host_params[_MG_NUM_LEVEL_];
  if (num_levels <= 1) return;
  const int base = _MG_LEVEL1_E_ + 8 * (num_levels - 2);
  const int coarse[4] = {
      set_ptr->host_params[base + 1], set_ptr->host_params[base + 2],
      set_ptr->host_params[base + 3], set_ptr->host_params[base + 4]};
  const int next[4] = {
      coarse[0] / 2 > 0 ? coarse[0] / 2 : 1,
      coarse[1] / 2 > 0 ? coarse[1] / 2 : 1,
      coarse[2] / 2 > 0 ? coarse[2] / 2 : 1,
      coarse[3] / 2 > 0 ? coarse[3] / 2 : 1};
  const long long volume = static_cast<long long>(coarse[0]) * coarse[1] *
                           coarse[2] * coarse[3];
  const bool too_deep = num_levels > 2;
  const bool too_small = coarse[0] < 4 || coarse[1] < 4 ||
                         coarse[2] < 4 || coarse[3] < 4 ||
                         volume < 1024;
  const bool no_next = next[0] < 4 || next[1] < 4 ||
                       next[2] < 4 || next[3] < 4;
  if (too_deep || too_small || no_next) {
    fprintf(stderr,
            "PYQCU::QCU::MULTIGRID::WARNING:\n"
            " Legacy-MultiGrid levels=%d, coarsest=[%d,%d,%d,%d], "
            "next=[%d,%d,%d,%d], volume=%lld. Legacy defaults to 2 levels; "
            "deep or small coarse hierarchies are experimental.\n",
            num_levels, coarse[0], coarse[1], coarse[2], coarse[3],
            next[0], next[1], next[2], next[3], volume);
  }
}

/**
 * @brief C++ CUDA Clover Multigrid solver entry point.
 *
 * The set_ptrs array (int64[100]) is used to pass coarse-grid operator data
 * using the following convention (SCHUR-consistent 33-tensor stencil):
 *   set_ptrs[_SET_PTRS_COARSE_BASE_ + 4*fl + 0] = null_vecs  for fine level fl→fl+1
 *   set_ptrs[_SET_PTRS_COARSE_BASE_ + 4*fl + 1] = hop_nn     for coarse level fl+1
 *   set_ptrs[_SET_PTRS_COARSE_BASE_ + 4*fl + 2] = hop_diag   for coarse level fl+1
 *   set_ptrs[_SET_PTRS_COARSE_BASE_ + 4*fl + 3] = sit_packed for coarse level fl+1
 *
 * where _SET_PTRS_COARSE_BASE_ = 30 (well past the LatticeSet slots; _SET_INDEX_ grows with each applyInitQcu, so the coarse-op slots must not collide with it) in set_ptrs after scratch ptrs).
 */

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
    auto ta = std::chrono::high_resolution_clock::now();
    _multigrid.init(fermion_out, fermion_in, gauge, clover_ee, clover_oo,
                    clover_ee_inv, clover_oo_inv);
    auto tb = std::chrono::high_resolution_clock::now();
    double init_ms = std::chrono::duration<double,std::milli>(tb-ta).count();

    // ---- Wire up coarse-grid operators from set_ptrs ----
    // For each fine level fl, read null_vecs, hop_packed, sit_packed.
    // These must be set by the Python caller before calling this function.
    int num_levels = set_ptr->host_params[_MG_NUM_LEVEL_];
    warn_legacy_multigrid_depth(set_ptr);
    for (int fl = 0; fl < num_levels - 1; fl++) {
      int base = _SET_PTRS_COARSE_BASE_ + 4 * fl;
      void *nv = (void *)sp[base + 0];
      void *hnn = (void *)sp[base + 1];
      void *hdg = (void *)sp[base + 2];
      void *sp_sit = (void *)sp[base + 3];
      if (nv && hnn && hdg && sp_sit) {
        _multigrid.set_coarse_ops(fl, nv, hnn, hdg, sp_sit);
        if (set_ptr->host_params[_VERBOSE_] && set_ptr->host_params[_NODE_RANK_] == 0) {
          printf("MG: set coarse ops level %d→%d: nv=%p, hop_nn=%p, hop_diag=%p, sit=%p\n",
                 fl, fl+1, nv, hnn, hdg, sp_sit);
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
    auto tc = std::chrono::high_resolution_clock::now();
    _multigrid.end();
    auto td = std::chrono::high_resolution_clock::now();
    double end_ms = std::chrono::duration<double,std::milli>(td-tc).count();
    printf("MG init=%.1fms end=%.1fms solve=%.1fms\n", init_ms, end_ms, _multigrid.solve_time_ms);
  } else if (data_type == _LAT_C128_) {
    LatticeSet<double> *set_ptr =
        static_cast<LatticeSet<double> *>((void *)(sp[set_index]));
    LatticeCloverMultigrid<double> _multigrid;
    _multigrid.give(set_ptr);
    _multigrid.init(fermion_out, fermion_in, gauge, clover_ee, clover_oo,
                    clover_ee_inv, clover_oo_inv);

    // ---- Wire up coarse-grid operators from set_ptrs ----
    int num_levels = set_ptr->host_params[_MG_NUM_LEVEL_];
    warn_legacy_multigrid_depth(set_ptr);
    for (int fl = 0; fl < num_levels - 1; fl++) {
      int base = _SET_PTRS_COARSE_BASE_ + 4 * fl;
      void *nv = (void *)sp[base + 0];
      void *hnn = (void *)sp[base + 1];
      void *hdg = (void *)sp[base + 2];
      void *sp_sit = (void *)sp[base + 3];
      if (nv && hnn && hdg && sp_sit) {
        _multigrid.set_coarse_ops(fl, nv, hnn, hdg, sp_sit);
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
