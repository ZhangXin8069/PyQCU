#include "kupl/qcu.h"

#include "api_common.hpp"

extern "C" {

void applyInitQcu(long long _set_ptrs, long long _params, long long _argv) {
  kupl::qcu::api::guarded("INIT", [&] {
    kupl::qcu::initialize_mpi();
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
  });
}

} // extern "C"
