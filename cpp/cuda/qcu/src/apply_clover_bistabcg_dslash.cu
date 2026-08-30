#include "../include/qcu.h"
#include "../python/pyqcu.h"
#pragma optimize(5)
using namespace qcu;
// dev74 MT: removed the two cudaDeviceSynchronize() calls (entry + exit).
// dslash() already synchronizes its own stream(s) before returning
// (single-rank fast path: cudaStreamSynchronize(set_ptr->stream); multi-rank:
// sync_if_multi on every dim stream).  A global device sync serialized
// concurrent CudaSchurOp instances (each owns a private non-blocking stream),
// which destroyed multi-threaded stencil-build parallelism.  Semantics for a
// single caller are unchanged (output is ready on return).
void applyCloverBistabCgDslashQcu(long long _fermion_out, long long _fermion_in,
                                  long long _gauge, long long _clover_ee,
                                  long long _clover_oo,
                                  long long _clover_ee_inv,
                                  long long _clover_oo_inv, long long _set_ptrs,
                                  long long _params) {
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
}

int applyCloverBistabCgPrepareQcu(
    long long _compact_rhs, long long _full_rhs, long long _gauge,
    long long _clover_ee, long long _clover_oo, long long _clover_ee_inv,
    long long _clover_oo_inv, long long _set_ptrs, long long _params) {
  try {
    checkCudaErrors(cudaDeviceSynchronize());
    void *set_ptrs = reinterpret_cast<void *>(_set_ptrs);
    int *params = reinterpret_cast<int *>(_params);
    const int set_index = params[_SET_INDEX_];
    if (params[_DATA_TYPE_] == _LAT_C64_) {
      auto *set = reinterpret_cast<LatticeSet<float> *>(
          reinterpret_cast<long long *>(set_ptrs)[set_index]);
      LatticeCloverBistabCg<float> solver;
      solver.give(set);
      solver.init(reinterpret_cast<void *>(_gauge),
                  reinterpret_cast<void *>(_clover_ee),
                  reinterpret_cast<void *>(_clover_oo),
                  reinterpret_cast<void *>(_clover_ee_inv),
                  reinterpret_cast<void *>(_clover_oo_inv));
      solver.prepare(reinterpret_cast<void *>(_compact_rhs),
                     reinterpret_cast<void *>(_full_rhs));
    } else if (params[_DATA_TYPE_] == _LAT_C128_) {
      auto *set = reinterpret_cast<LatticeSet<double> *>(
          reinterpret_cast<long long *>(set_ptrs)[set_index]);
      LatticeCloverBistabCg<double> solver;
      solver.give(set);
      solver.init(reinterpret_cast<void *>(_gauge),
                  reinterpret_cast<void *>(_clover_ee),
                  reinterpret_cast<void *>(_clover_oo),
                  reinterpret_cast<void *>(_clover_ee_inv),
                  reinterpret_cast<void *>(_clover_oo_inv));
      solver.prepare(reinterpret_cast<void *>(_compact_rhs),
                     reinterpret_cast<void *>(_full_rhs));
    } else {
      throw std::invalid_argument(
          "Clover Schur prepare supports complex64/complex128");
    }
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "PYQCU::SOLVER::STRICT_MG::FINE_PREPARE: %s\n",
                 error.what());
    return 1;
  }
}

int applyCloverBistabCgReconstructQcu(
    long long _full_out, long long _full_rhs, long long _target_odd,
    long long _gauge, long long _clover_ee, long long _clover_oo,
    long long _clover_ee_inv, long long _clover_oo_inv,
    long long _set_ptrs, long long _params) {
  try {
    checkCudaErrors(cudaDeviceSynchronize());
    void *set_ptrs = reinterpret_cast<void *>(_set_ptrs);
    int *params = reinterpret_cast<int *>(_params);
    const int set_index = params[_SET_INDEX_];
    if (params[_DATA_TYPE_] == _LAT_C64_) {
      auto *set = reinterpret_cast<LatticeSet<float> *>(
          reinterpret_cast<long long *>(set_ptrs)[set_index]);
      LatticeCloverBistabCg<float> solver;
      solver.give(set);
      solver.init(reinterpret_cast<void *>(_gauge),
                  reinterpret_cast<void *>(_clover_ee),
                  reinterpret_cast<void *>(_clover_oo),
                  reinterpret_cast<void *>(_clover_ee_inv),
                  reinterpret_cast<void *>(_clover_oo_inv));
      solver.reconstruct(reinterpret_cast<void *>(_full_out),
                         reinterpret_cast<void *>(_full_rhs),
                         reinterpret_cast<void *>(_target_odd));
    } else if (params[_DATA_TYPE_] == _LAT_C128_) {
      auto *set = reinterpret_cast<LatticeSet<double> *>(
          reinterpret_cast<long long *>(set_ptrs)[set_index]);
      LatticeCloverBistabCg<double> solver;
      solver.give(set);
      solver.init(reinterpret_cast<void *>(_gauge),
                  reinterpret_cast<void *>(_clover_ee),
                  reinterpret_cast<void *>(_clover_oo),
                  reinterpret_cast<void *>(_clover_ee_inv),
                  reinterpret_cast<void *>(_clover_oo_inv));
      solver.reconstruct(reinterpret_cast<void *>(_full_out),
                         reinterpret_cast<void *>(_full_rhs),
                         reinterpret_cast<void *>(_target_odd));
    } else {
      throw std::invalid_argument(
          "Clover Schur reconstruct supports complex64/complex128");
    }
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "PYQCU::SOLVER::STRICT_MG::FINE_RECONSTRUCT: %s\n",
                 error.what());
    return 1;
  }
}
