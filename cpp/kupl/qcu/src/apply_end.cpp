#include "kupl/qcu.h"
#include "kupl/lattice_cpu.h"

#include "api_common.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>

extern "C" {

void applyEndQcu(long long _set_ptrs, long long _params) {
  kupl::qcu::api::guarded("END", [&] {
    const bool profile = kupl::qcu::init_profile_enabled();
    const auto started = std::chrono::steady_clock::now();
    long long *table = kupl::qcu::api::set_ptr_table(_set_ptrs);
    const auto *params = kupl::qcu::api::params_pointer(_params);
    const int index = kupl::qcu::api::set_index(params);
    kupl::qcu::erase_lattice_set(kupl::qcu::api::slot_key(_set_ptrs, params));
    table[index] = 0;
    if (profile) {
      const double elapsed = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - started).count();
      std::fprintf(stderr,
                   "PYQCU::KUPL::INIT_APPLY: stage=erase elapsed=%.6f\n",
                   elapsed);
    }
  });
}

} // extern "C"
