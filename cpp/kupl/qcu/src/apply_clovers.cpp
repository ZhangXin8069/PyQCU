#include "kupl/qcu.h"

#include "api_common.hpp"

extern "C" {

void applyCloversQcu(long long _clover, long long _clover_inv,
                     long long _gauge, long long _set_ptrs,
                     long long _params) {
  kupl::qcu::api::guarded("CLOVERS", [&] {
    kupl::qcu::api::lattice_set(_set_ptrs, _params)
        ->clovers(kupl::qcu::api::mutable_pointer(_clover),
                  kupl::qcu::api::mutable_pointer(_clover_inv),
                  kupl::qcu::api::pointer(_gauge));
  });
}

} // extern "C"
