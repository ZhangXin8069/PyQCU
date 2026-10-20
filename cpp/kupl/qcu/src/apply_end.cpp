#include "kupl/qcu.h"

#include "api_common.hpp"

extern "C" {

void applyEndQcu(long long _set_ptrs, long long _params) {
  kupl::qcu::api::guarded("END", [&] {
    long long *table = kupl::qcu::api::set_ptr_table(_set_ptrs);
    const auto *params = kupl::qcu::api::params_pointer(_params);
    const int index = kupl::qcu::api::set_index(params);
    kupl::qcu::erase_lattice_set(kupl::qcu::api::slot_key(_set_ptrs, params));
    table[index] = 0;
  });
}

} // extern "C"
