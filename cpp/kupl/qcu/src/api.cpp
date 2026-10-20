#include "kupl/qcu.h"

#include "backend.hpp"

#include <cstdio>
#include <cstdint>
#include <exception>
#include <stdexcept>

namespace {

template <typename Function>
void guarded(const char *submodule, Function &&function) {
  try {
    function();
  } catch (const std::exception &error) {
    kupl::qcu::abortWithMessage(submodule, error.what());
  } catch (...) {
    kupl::qcu::abortWithMessage(submodule, "unknown exception");
  }
}

template <typename Function>
int statusGuarded(const char *submodule, Function &&function) {
  try {
    return function();
  } catch (const std::exception &error) {
    std::fprintf(stderr, "PYQCU::KUPL::%s:\n %s\n", submodule, error.what());
    return 1;
  } catch (...) {
    std::fprintf(stderr, "PYQCU::KUPL::%s:\n unknown exception\n", submodule);
    return 1;
  }
}

const int32_t *paramsPointer(long long pointer) {
  return reinterpret_cast<const int32_t *>(pointer);
}

const void *pointer(long long pointer) {
  return reinterpret_cast<const void *>(pointer);
}

void *mutablePointer(long long pointer) {
  return reinterpret_cast<void *>(pointer);
}

constexpr long long kKuplSlotToken = 0x4b55504cLL;

long long *setPtrTable(long long pointer) {
  if (pointer == 0) {
    throw std::runtime_error("set_ptrs pointer is null");
  }
  return reinterpret_cast<long long *>(pointer);
}

int setIndex(const int32_t *params) {
  if (params == nullptr) {
    throw std::runtime_error("params pointer is null");
  }
  const int index = params[kupl::qcu::kSetIndex];
  if (index < 0 || index >= kupl::qcu::kSetPtrsSize) {
    throw std::runtime_error("params[_SET_INDEX_] is outside set_ptrs");
  }
  return index;
}

const void *slotKey(long long set_ptrs, const int32_t *params) {
  return static_cast<const void *>(
      setPtrTable(set_ptrs) + setIndex(params));
}

std::shared_ptr<kupl::qcu::BackendBase> backend(long long set_ptrs,
                                                long long params) {
  const auto *host_params = paramsPointer(params);
  auto value = kupl::qcu::findBackend(slotKey(set_ptrs, host_params));
  value->updateParams(host_params);
  return value;
}

[[noreturn]] void unavailable(const char *submodule) {
  throw std::runtime_error(std::string("KUPL QCU entry point is not "
                                       "implemented: ") +
                           submodule);
}

[[noreturn]] int unavailableStatus(const char *submodule) {
  unavailable(submodule);
}

} // namespace

extern "C" {

void applyInitQcu(long long _set_ptrs, long long _params, long long _argv) {
  guarded("INIT", [&] {
    kupl::qcu::initializeMpi();
    long long *table = setPtrTable(_set_ptrs);
    const auto *host_params = paramsPointer(_params);
    const int index = setIndex(host_params);
    if (table[index] != 0) {
      throw std::runtime_error("QCU set_ptrs slot is already initialized");
    }
    kupl::qcu::registerBackend(slotKey(_set_ptrs, host_params), host_params,
                               pointer(_argv));
    table[index] = kKuplSlotToken;
  });
}

void applyEndQcu(long long _set_ptrs, long long _params) {
  guarded("END", [&] {
    long long *table = setPtrTable(_set_ptrs);
    const int index = setIndex(paramsPointer(_params));
    kupl::qcu::eraseBackend(slotKey(_set_ptrs, paramsPointer(_params)));
    table[index] = 0;
  });
}

void applyWilsonDslashQcu(long long _fermion_out, long long _fermion_in,
                          long long _gauge, long long _set_ptrs,
                          long long _params) {
  guarded("WILSON_DSLASH", [&] {
    backend(_set_ptrs, _params)->wilsonDslash(
        mutablePointer(_fermion_out), pointer(_fermion_in), pointer(_gauge));
  });
}

void applyWilsonBistabCgQcu(long long _fermion_out, long long _fermion_in,
                            long long _gauge, long long _set_ptrs,
                            long long _params) {
  guarded("WILSON_BISTABCG", [&] {
    backend(_set_ptrs, _params)->wilsonBistabCg(
        mutablePointer(_fermion_out), pointer(_fermion_in), pointer(_gauge));
  });
}

void applyWilsonBistabCgDslashQcu(long long _fermion_out,
                                  long long _fermion_in, long long _gauge,
                                  long long _set_ptrs, long long _params) {
  guarded("WILSON_BISTABCG_DSLASH", [&] {
    backend(_set_ptrs, _params)->wilsonBistabCgDslash(
        mutablePointer(_fermion_out), pointer(_fermion_in), pointer(_gauge));
  });
}

void applyWilsonCgQcu(long long _fermion_out, long long _fermion_in,
                      long long _gauge, long long _set_ptrs,
                      long long _params) {
  guarded("WILSON_CG", [&] {
    backend(_set_ptrs, _params)->wilsonCg(
        mutablePointer(_fermion_out), pointer(_fermion_in), pointer(_gauge));
  });
}

void applyWilsonCgDslashQcu(long long _fermion_out, long long _fermion_in,
                            long long _gauge, long long _set_ptrs,
                            long long _params) {
  guarded("WILSON_CG_DSLASH", [&] {
    backend(_set_ptrs, _params)->wilsonCgDslash(
        mutablePointer(_fermion_out), pointer(_fermion_in), pointer(_gauge));
  });
}

void testWilsonDslashQcu(long long _fermion_out, long long _fermion_in,
                         long long _gauge, long long _set_ptrs,
                         long long _params) {
  applyWilsonDslashQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs,
                       _params);
}

void applyCloverDslashQcu(long long _fermion_out, long long _fermion_in,
                          long long _gauge, long long _set_ptrs,
                          long long _params) {
  guarded("CLOVER_DSLASH", [&] {
    backend(_set_ptrs, _params)->cloverDslash(
        mutablePointer(_fermion_out), pointer(_fermion_in), pointer(_gauge));
  });
}

void testCloverDslashQcu(long long _fermion_out, long long _fermion_in,
                         long long _gauge, long long _set_ptrs,
                         long long _params) {
  applyCloverDslashQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs,
                       _params);
}

void applyCloverQcu(long long _clover, long long _gauge, long long _set_ptrs,
                    long long _params) {
  guarded("CLOVER", [&] {
    backend(_set_ptrs, _params)->clover(mutablePointer(_clover),
                                        pointer(_gauge));
  });
}

void applyCloversQcu(long long _clover, long long _clover_inv,
                     long long _gauge, long long _set_ptrs,
                     long long _params) {
  guarded("CLOVERS", [&] {
    backend(_set_ptrs, _params)->clovers(
        mutablePointer(_clover), mutablePointer(_clover_inv),
        pointer(_gauge));
  });
}

void applyDslashQcu(long long _fermion_out, long long _fermion_in,
                    long long _gauge, long long _clover, long long _set_ptrs,
                    long long _params) {
  guarded("DSLASH", [&] {
    backend(_set_ptrs, _params)->suppliedCloverDslash(
        mutablePointer(_fermion_out), pointer(_fermion_in), pointer(_gauge),
        pointer(_clover));
  });
}

void applyLaplacianQcu(long long _laplacian_out, long long _laplacian_in,
                       long long _gauge, long long _set_ptrs,
                       long long _params) {
  guarded("LAPLACIAN", [&] {
    backend(_set_ptrs, _params)->laplacian(
        mutablePointer(_laplacian_out), pointer(_laplacian_in),
        pointer(_gauge));
  });
}

void applyCloverBistabCgQcu(
    long long _fermion_out, long long _fermion_in, long long _gauge,
    long long _clover_even, long long _clover_odd,
    long long _clover_even_inv, long long _clover_odd_inv, long long _set_ptrs,
    long long _params) {
  guarded("CLOVER_BISTABCG", [&] {
    backend(_set_ptrs, _params)->cloverBistabCg(
        mutablePointer(_fermion_out), pointer(_fermion_in), pointer(_gauge),
        pointer(_clover_even), pointer(_clover_odd),
        pointer(_clover_even_inv), pointer(_clover_odd_inv));
  });
}

int getCloverBistabCgIterationsQcu(long long _set_ptrs, long long _params) {
  int value = 0;
  guarded("BISTABCG_ITERATIONS", [&] {
    value = backend(_set_ptrs, _params)->iterations();
  });
  return value;
}

void applyCloverBistabCgDslashQcu(
    long long _fermion_out, long long _fermion_in, long long _gauge,
    long long _clover_even, long long _clover_odd,
    long long _clover_even_inv, long long _clover_odd_inv, long long _set_ptrs,
    long long _params) {
  guarded("CLOVER_BISTABCG_DSLASH", [&] {
    backend(_set_ptrs, _params)->cloverBistabCgDslash(
        mutablePointer(_fermion_out), pointer(_fermion_in), pointer(_gauge),
        pointer(_clover_even), pointer(_clover_odd),
        pointer(_clover_even_inv), pointer(_clover_odd_inv));
  });
}

int applyCloverBistabCgPrepareQcu(
    long long _compact_rhs, long long _full_rhs, long long _gauge,
    long long _clover_even, long long _clover_odd,
    long long _clover_even_inv, long long _clover_odd_inv, long long _set_ptrs,
    long long _params) {
  return statusGuarded("CLOVER_BISTABCG_PREPARE", [&] {
    return backend(_set_ptrs, _params)->cloverBistabCgPrepare(
        mutablePointer(_compact_rhs), pointer(_full_rhs), pointer(_gauge),
        pointer(_clover_even), pointer(_clover_odd),
        pointer(_clover_even_inv), pointer(_clover_odd_inv));
  });
}

int applyCloverBistabCgReconstructQcu(
    long long _full_out, long long _full_rhs, long long _target_solution,
    long long _gauge, long long _clover_even, long long _clover_odd,
    long long _clover_even_inv, long long _clover_odd_inv,
    long long _set_ptrs, long long _params) {
  return statusGuarded("CLOVER_BISTABCG_RECONSTRUCT", [&] {
    return backend(_set_ptrs, _params)->cloverBistabCgReconstruct(
        mutablePointer(_full_out), pointer(_full_rhs),
        pointer(_target_solution), pointer(_gauge), pointer(_clover_even),
        pointer(_clover_odd), pointer(_clover_even_inv),
        pointer(_clover_odd_inv));
  });
}

void applyGaussGaugeQcu(long long _gauge, long long _set_ptrs,
                        long long _params) {
  guarded("GAUSS_GAUGE", [&] {
    backend(_set_ptrs, _params)->gaussGauge(mutablePointer(_gauge));
  });
}

void applyMultigridRestrictQcu(long long _coarse_out, long long _fine_in,
                               long long _null_vectors, long long _set_ptrs,
                               long long _params) {
  guarded("MULTIGRID_RESTRICT", [&] {
    const auto params = paramsPointer(_params);
    backend(_set_ptrs, _params)->multigridRestrict(
        mutablePointer(_coarse_out), pointer(_fine_in),
        pointer(_null_vectors), params[kupl::qcu::kMgLevel1E],
        kupl::qcu::kVectorSize, params[kupl::qcu::kLatX],
        params[kupl::qcu::kLatY], params[kupl::qcu::kLatZ],
        params[kupl::qcu::kLatT], params[kupl::qcu::kMgLevel1X],
        params[kupl::qcu::kMgLevel1Y], params[kupl::qcu::kMgLevel1Z],
        params[kupl::qcu::kMgLevel1T]);
  });
}

void applyMultigridProLongQcu(long long _fine_out, long long _coarse_in,
                              long long _null_vectors, long long _set_ptrs,
                              long long _params) {
  guarded("MULTIGRID_PROLONG", [&] {
    const auto params = paramsPointer(_params);
    backend(_set_ptrs, _params)->multigridProlong(
        mutablePointer(_fine_out), pointer(_coarse_in),
        pointer(_null_vectors), params[kupl::qcu::kMgLevel1E],
        kupl::qcu::kVectorSize, params[kupl::qcu::kLatX],
        params[kupl::qcu::kLatY], params[kupl::qcu::kLatZ],
        params[kupl::qcu::kLatT], params[kupl::qcu::kMgLevel1X],
        params[kupl::qcu::kMgLevel1Y], params[kupl::qcu::kMgLevel1Z],
        params[kupl::qcu::kMgLevel1T]);
  });
}

void applyMultigridCoarseDslashQcu(long long _out, long long _in,
                                   long long _hopping, long long _sitting,
                                   long long _set_ptrs, long long _params) {
  guarded("MULTIGRID_COARSE_DSLASH", [&] {
    const auto params = paramsPointer(_params);
    backend(_set_ptrs, _params)->multigridCoarseDslash(
        mutablePointer(_out), pointer(_in), pointer(_hopping),
        pointer(_sitting), params[kupl::qcu::kMgLevel1E],
        params[kupl::qcu::kMgLevel1X], params[kupl::qcu::kMgLevel1Y],
        params[kupl::qcu::kMgLevel1Z], params[kupl::qcu::kMgLevel1T]);
  });
}

void applyMultigridCoarseDslashWideQcu(
    long long _out, long long _in, long long _sitting, long long _hop_nn,
    long long _hop_diag, long long _set_ptrs, long long _params) {
  guarded("MULTIGRID_COARSE_DSLASH_WIDE", [&] {
    const auto params = paramsPointer(_params);
    backend(_set_ptrs, _params)->multigridCoarseDslashWide(
        mutablePointer(_out), pointer(_in), pointer(_sitting),
        pointer(_hop_nn), pointer(_hop_diag),
        params[kupl::qcu::kMgLevel1E], params[kupl::qcu::kMgLevel1X],
        params[kupl::qcu::kMgLevel1Y], params[kupl::qcu::kMgLevel1Z],
        params[kupl::qcu::kMgLevel1T]);
  });
}

int applyMultigridStrictCoarseQcu(long long, long long, long long, long long,
                                  long long, long long, int, int, int, int,
                                  int, int) {
  return statusGuarded(
      "MULTIGRID_STRICT_COARSE",
      [] { return unavailableStatus("applyMultigridStrictCoarseQcu"); });
}

int applyMultigridStrictMatPCQcu(long long, long long, long long, long long,
                                 long long, long long, int, int, int, int,
                                 int, int) {
  return statusGuarded(
      "MULTIGRID_STRICT_MATPC",
      [] { return unavailableStatus("applyMultigridStrictMatPCQcu"); });
}

int applyMultigridStrictFineMatPCQcu(long long, long long, long long,
                                     long long, long long, long long,
                                     long long, long long, long long, int) {
  return statusGuarded(
      "MULTIGRID_STRICT_FINE_MATPC",
      [] { return unavailableStatus("applyMultigridStrictFineMatPCQcu"); });
}

int applyMultigridStrictPrepareQcu(long long, long long, long long, long long,
                                   long long, long long, long long, int, int,
                                   int, int, int, int) {
  return statusGuarded(
      "MULTIGRID_STRICT_PREPARE",
      [] { return unavailableStatus("applyMultigridStrictPrepareQcu"); });
}

int applyMultigridStrictReconstructQcu(long long, long long, long long,
                                       long long, long long, long long,
                                       long long, long long, int, int, int,
                                       int, int, int) {
  return statusGuarded(
      "MULTIGRID_STRICT_RECONSTRUCT",
      [] { return unavailableStatus("applyMultigridStrictReconstructQcu"); });
}

int applyMultigridStrictRestrictQcu(long long, long long, long long,
                                    long long, long long, int, int, int, int,
                                    int, int, int, int, int, int, int) {
  return statusGuarded(
      "MULTIGRID_STRICT_RESTRICT",
      [] { return unavailableStatus("applyMultigridStrictRestrictQcu"); });
}

int applyMultigridStrictProLongQcu(long long, long long, long long, long long,
                                   long long, int, int, int, int, int, int,
                                   int, int, int, int, int) {
  return statusGuarded(
      "MULTIGRID_STRICT_PROLONG",
      [] { return unavailableStatus("applyMultigridStrictProLongQcu"); });
}

int applyMultigridStrictVCycleQcu(long long, long long, long long, long long,
                                  int, unsigned long long *) {
  return statusGuarded(
      "MULTIGRID_STRICT_VCYCLE",
      [] { return unavailableStatus("applyMultigridStrictVCycleQcu"); });
}

int applyMultigridStrictInitQcu(long long, long long, int,
                                unsigned long long *) {
  return statusGuarded(
      "MULTIGRID_STRICT_INIT",
      [] { return unavailableStatus("applyMultigridStrictInitQcu"); });
}

int applyMultigridStrictEndQcu(long long, long long) {
  return statusGuarded(
      "MULTIGRID_STRICT_END",
      [] { return unavailableStatus("applyMultigridStrictEndQcu"); });
}

int applyMultigridStrictFgmresQcu(
    long long, long long, long long, long long, long long, long long,
    long long, long long, long long, long long, int, int, int, int, int,
    int, int, int, int, int, int, int, int, double, int, int,
    unsigned long long, int *, int *, double *, unsigned long long *) {
  return statusGuarded(
      "MULTIGRID_STRICT_FGMRES",
      [] { return unavailableStatus("applyMultigridStrictFgmresQcu"); });
}

void applyCloverMultigridQcu(long long, long long, long long, long long,
                             long long, long long, long long, long long,
                             long long) {
  guarded("CLOVER_MULTIGRID",
          [] { unavailable("applyCloverMultigridQcu"); });
}

int verifyCloverMultigridQcu(long long, long long, long long, long long,
                             long long, long long, long long, long long,
                             long long) {
  return statusGuarded(
      "CLOVER_MULTIGRID_VERIFY",
      [] { return unavailableStatus("verifyCloverMultigridQcu"); });
}

} // extern "C"
