#include "kupl/qcu.h"

#include "api_common.hpp"

extern "C" {

void applyGaussGaugeQcu(long long _gauge, long long _set_ptrs,
                        long long _params) {
  kupl::qcu::api::guarded("GAUSS_GAUGE", [&] {
    kupl::qcu::api::lattice_set(_set_ptrs, _params)
        ->gauss_gauge(kupl::qcu::api::mutable_pointer(_gauge));
  });
}

} // extern "C"
