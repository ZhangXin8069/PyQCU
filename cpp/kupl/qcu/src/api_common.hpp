#ifndef KUPL_QCU_API_COMMON_HPP
#define KUPL_QCU_API_COMMON_HPP

#include "kupl/lattice_set.h"

#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

namespace kupl::qcu::api {

inline constexpr long long kSlotToken = 0x4b55504cLL;

template <typename Function>
void guarded(const char *submodule, Function &&function) {
  try {
    function();
  } catch (const std::exception &error) {
    abort_with_message(submodule, error.what());
  } catch (...) {
    abort_with_message(submodule, "unknown exception");
  }
}

template <typename Function>
int status_guarded(const char *submodule, Function &&function) {
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

inline const int32_t *params_pointer(long long pointer) {
  return reinterpret_cast<const int32_t *>(pointer);
}

inline const void *pointer(long long pointer) {
  return reinterpret_cast<const void *>(pointer);
}

inline void *mutable_pointer(long long pointer) {
  return reinterpret_cast<void *>(pointer);
}

inline long long *set_ptr_table(long long pointer) {
  if (pointer == 0) {
    throw std::runtime_error("set_ptrs pointer is null");
  }
  return reinterpret_cast<long long *>(pointer);
}

inline int set_index(const int32_t *params) {
  if (params == nullptr) {
    throw std::runtime_error("params pointer is null");
  }
  const int index = params[kSetIndex];
  if (index < 0 || index >= kSetPtrsSize) {
    throw std::runtime_error("params[_SET_INDEX_] is outside set_ptrs");
  }
  return index;
}

inline const void *slot_key(long long set_ptrs, const int32_t *params) {
  return static_cast<const void *>(set_ptr_table(set_ptrs) +
                                   set_index(params));
}

inline std::shared_ptr<LatticeSetBase> lattice_set(long long set_ptrs,
                                                   long long params) {
  const auto *host_params = params_pointer(params);
  auto value = find_lattice_set(slot_key(set_ptrs, host_params));
  value->update_params(host_params);
  return value;
}

[[noreturn]] inline void unavailable(const char *submodule) {
  throw std::runtime_error(
      std::string("KUPL QCU entry point is not implemented: ") + submodule);
}

[[noreturn]] inline int unavailable_status(const char *submodule) {
  unavailable(submodule);
}

} // namespace kupl::qcu::api

#endif
