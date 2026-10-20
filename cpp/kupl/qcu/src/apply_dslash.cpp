#include "kupl/qcu.h"

#include "api_common.hpp"

extern "C" {

void applyDslashQcu(long long _fermion_out, long long _fermion_in,
                    long long _gauge, long long _clover, long long _set_ptrs,
                    long long _params) {
  kupl::qcu::api::guarded("DSLASH", [&] {
    kupl::qcu::api::lattice_set(_set_ptrs, _params)
        ->supplied_clover_dslash(
            kupl::qcu::api::mutable_pointer(_fermion_out),
            kupl::qcu::api::pointer(_fermion_in),
            kupl::qcu::api::pointer(_gauge),
            kupl::qcu::api::pointer(_clover));
  });
}

} // extern "C"
