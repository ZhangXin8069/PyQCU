#include "kupl/qcu.h"

#include "api_common.hpp"

extern "C" {

void applyCloverBistabCgQcu(
    long long _fermion_out, long long _fermion_in, long long _gauge,
    long long _clover_even, long long _clover_odd,
    long long _clover_even_inv, long long _clover_odd_inv,
    long long _set_ptrs, long long _params) {
  kupl::qcu::api::guarded("CLOVER_BISTABCG", [&] {
    kupl::qcu::api::lattice_set(_set_ptrs, _params)
        ->clover_bistabcg(
            kupl::qcu::api::mutable_pointer(_fermion_out),
            kupl::qcu::api::pointer(_fermion_in),
            kupl::qcu::api::pointer(_gauge),
            kupl::qcu::api::pointer(_clover_even),
            kupl::qcu::api::pointer(_clover_odd),
            kupl::qcu::api::pointer(_clover_even_inv),
            kupl::qcu::api::pointer(_clover_odd_inv));
  });
}

int getCloverBistabCgIterationsQcu(long long _set_ptrs, long long _params) {
  int value = 0;
  kupl::qcu::api::guarded("BISTABCG_ITERATIONS", [&] {
    value = kupl::qcu::api::lattice_set(_set_ptrs, _params)->iterations();
  });
  return value;
}

} // extern "C"
