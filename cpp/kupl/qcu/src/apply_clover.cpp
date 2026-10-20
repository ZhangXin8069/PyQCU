#include "kupl/qcu.h"

#include "api_common.hpp"

extern "C" {

void applyCloverQcu(long long _clover, long long _gauge, long long _set_ptrs,
                    long long _params) {
  kupl::qcu::api::guarded("CLOVER", [&] {
    kupl::qcu::api::lattice_set(_set_ptrs, _params)
        ->clover(kupl::qcu::api::mutable_pointer(_clover),
                 kupl::qcu::api::pointer(_gauge));
  });
}

} // extern "C"
