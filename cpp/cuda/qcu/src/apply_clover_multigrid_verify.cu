#include "../include/qcu.h"
#include "../python/pyqcu.h"

using namespace qcu;

namespace {

// Keep this value in sync with apply_clover_multigrid.cu and the public
// set_ptrs protocol.  Slots 0..29 belong to the LatticeSet scratch ABI.
constexpr int kCoarseOpsBase = 30;
constexpr int kSetPtrsSize = 100;

template <typename T>
int run_clover_multigrid_verify(
    void *fermion_out, void *fermion_in, void *gauge, void *clover_ee,
    void *clover_oo, void *clover_ee_inv, void *clover_oo_inv,
    long long *set_ptr_slots, LatticeSet<T> *set_ptr) {
  LatticeCloverMultigrid<T> multigrid;
  bool initialized = false;
  int result = 2;  // 2 means bridge/runtime error.

  try {
    multigrid.give(set_ptr);
    multigrid.init(fermion_out, fermion_in, gauge, clover_ee, clover_oo,
                   clover_ee_inv, clover_oo_inv);
    initialized = true;

    const int num_levels = set_ptr->host_params[_MG_NUM_LEVEL_];
    if (num_levels < 1 || num_levels > 1 + (kSetPtrsSize - kCoarseOpsBase) / 4)
      throw std::invalid_argument("verify: invalid multigrid level count");

    // The verifier deliberately binds the same four pointers as the solver:
    // null vectors, nearest-neighbour hopping, diagonal hopping, and the
    // packed on-site block.  A missing/partial transition is left visible to
    // verify(), which reports a FAIL instead of dereferencing a fabricated
    // operator.
    for (int fl = 0; fl < num_levels - 1; ++fl) {
      const int base = kCoarseOpsBase + 4 * fl;
      multigrid.set_coarse_ops(
          fl, reinterpret_cast<void *>(set_ptr_slots[base + 0]),
          reinterpret_cast<void *>(set_ptr_slots[base + 1]),
          reinterpret_cast<void *>(set_ptr_slots[base + 2]),
          reinterpret_cast<void *>(set_ptr_slots[base + 3]));
    }

    result = multigrid.verify() ? 0 : 1;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "PYQCU::VERIFY::MULTIGRID:: %s\n", error.what());
    result = 2;
  } catch (...) {
    std::fprintf(stderr,
                 "PYQCU::VERIFY::MULTIGRID:: unknown C++ exception\n");
    result = 2;
  }

  // applyEndQcu owns the LatticeSet lifetime.  This end() only releases the
  // temporary solver hierarchy and is required for both PASS and FAIL.
  if (initialized) {
    try {
      multigrid.end();
    } catch (const std::exception &error) {
      std::fprintf(stderr,
                   "PYQCU::VERIFY::MULTIGRID:: cleanup failed: %s\n",
                   error.what());
      result = 2;
    } catch (...) {
      std::fprintf(stderr,
                   "PYQCU::VERIFY::MULTIGRID:: cleanup failed: unknown exception\n");
      result = 2;
    }
  }
  return result;
}

}  // namespace

int verifyCloverMultigridQcu(
    long long _fermion_out, long long _fermion_in, long long _gauge,
    long long _clover_ee, long long _clover_oo, long long _clover_ee_inv,
    long long _clover_oo_inv, long long _set_ptrs, long long _params) {
  try {
    if (_set_ptrs == 0 || _params == 0)
      throw std::invalid_argument("verify: set_ptrs and params are required");

    const cudaError_t sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
      std::fprintf(stderr,
                   "PYQCU::VERIFY::MULTIGRID:: CUDA synchronize failed: %s\n",
                   cudaGetErrorString(sync_status));
      return 2;
    }

    int *params = reinterpret_cast<int *>(_params);
    const int set_index = params[_SET_INDEX_];
    if (set_index < 0 || set_index >= kSetPtrsSize)
      throw std::invalid_argument("verify: _SET_INDEX_ is outside set_ptrs");

    long long *set_ptr_slots = reinterpret_cast<long long *>(_set_ptrs);
    if (set_ptr_slots[set_index] == 0)
      throw std::invalid_argument("verify: LatticeSet is not initialized");

    const int data_type = params[_DATA_TYPE_];
    if (data_type == _LAT_C64_) {
      LatticeSet<float> *set_ptr = reinterpret_cast<LatticeSet<float> *>(
          set_ptr_slots[set_index]);
      return run_clover_multigrid_verify<float>(
          reinterpret_cast<void *>(_fermion_out),
          reinterpret_cast<void *>(_fermion_in), reinterpret_cast<void *>(_gauge),
          reinterpret_cast<void *>(_clover_ee),
          reinterpret_cast<void *>(_clover_oo),
          reinterpret_cast<void *>(_clover_ee_inv),
          reinterpret_cast<void *>(_clover_oo_inv), set_ptr_slots, set_ptr);
    }
    if (data_type == _LAT_C128_) {
      LatticeSet<double> *set_ptr = reinterpret_cast<LatticeSet<double> *>(
          set_ptr_slots[set_index]);
      return run_clover_multigrid_verify<double>(
          reinterpret_cast<void *>(_fermion_out),
          reinterpret_cast<void *>(_fermion_in), reinterpret_cast<void *>(_gauge),
          reinterpret_cast<void *>(_clover_ee),
          reinterpret_cast<void *>(_clover_oo),
          reinterpret_cast<void *>(_clover_ee_inv),
          reinterpret_cast<void *>(_clover_oo_inv), set_ptr_slots, set_ptr);
    }
    throw std::invalid_argument("verify: unsupported data type");
  } catch (const std::exception &error) {
    std::fprintf(stderr, "PYQCU::VERIFY::MULTIGRID:: %s\n", error.what());
    return 2;
  } catch (...) {
    std::fprintf(stderr,
                 "PYQCU::VERIFY::MULTIGRID:: unknown C++ exception\n");
    return 2;
  }
}

