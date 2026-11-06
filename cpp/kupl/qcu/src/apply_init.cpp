#include "kupl/qcu.h"
#include "kupl/lattice_cpu.h"

#include "api_common.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>

extern "C" {

void applyInitQcu(long long _set_ptrs, long long _params, long long _argv) {
  kupl::qcu::api::guarded("INIT", [&] {
    const bool profile = kupl::qcu::init_profile_enabled();
    const auto started = std::chrono::steady_clock::now();
    kupl::qcu::initialize_mpi();
    if (profile) {
      const double elapsed = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started).count();
      std::fprintf(stderr,
                   "PYQCU::KUPL::INIT_APPLY: stage=pre-register elapsed=%.6f\n",
                   elapsed);
    }
    long long *table = kupl::qcu::api::set_ptr_table(_set_ptrs);
    const auto *host_params = kupl::qcu::api::params_pointer(_params);
    const int index = kupl::qcu::api::set_index(host_params);
    if (table[index] != 0) {
      throw std::runtime_error("QCU set_ptrs slot is already initialized");
    }
    kupl::qcu::register_lattice_set(
        kupl::qcu::api::slot_key(_set_ptrs, host_params), host_params,
        kupl::qcu::api::pointer(_argv));
    table[index] = kupl::qcu::api::kSlotToken;
    if (profile) {
      const double elapsed = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started).count();
      std::fprintf(stderr,
                   "PYQCU::KUPL::INIT_APPLY: stage=register elapsed=%.6f\n",
                   elapsed);
    }
  });
}

} // extern "C"
