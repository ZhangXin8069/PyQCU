#include "kupl/qcu.h"

#include "api_common.hpp"

extern "C" {

void applyCloverBistabCgDslashQcu(
    long long _fermion_out, long long _fermion_in, long long _gauge,
    long long _clover_even, long long _clover_odd,
    long long _clover_even_inv, long long _clover_odd_inv,
    long long _set_ptrs, long long _params) {
  kupl::qcu::api::guarded("CLOVER_BISTABCG_DSLASH", [&] {
    kupl::qcu::api::lattice_set(_set_ptrs, _params)
        ->clover_bistabcg_dslash(
            kupl::qcu::api::mutable_pointer(_fermion_out),
            kupl::qcu::api::pointer(_fermion_in),
            kupl::qcu::api::pointer(_gauge),
            kupl::qcu::api::pointer(_clover_even),
            kupl::qcu::api::pointer(_clover_odd),
            kupl::qcu::api::pointer(_clover_even_inv),
            kupl::qcu::api::pointer(_clover_odd_inv));
  });
}

int applyCloverBistabCgPrepareQcu(
    long long _compact_rhs, long long _full_rhs, long long _gauge,
    long long _clover_even, long long _clover_odd,
    long long _clover_even_inv, long long _clover_odd_inv,
    long long _set_ptrs, long long _params) {
  return kupl::qcu::api::status_guarded("CLOVER_BISTABCG_PREPARE", [&] {
    return kupl::qcu::api::lattice_set(_set_ptrs, _params)
        ->clover_bistabcg_prepare(
            kupl::qcu::api::mutable_pointer(_compact_rhs),
            kupl::qcu::api::pointer(_full_rhs),
            kupl::qcu::api::pointer(_gauge),
            kupl::qcu::api::pointer(_clover_even),
            kupl::qcu::api::pointer(_clover_odd),
            kupl::qcu::api::pointer(_clover_even_inv),
            kupl::qcu::api::pointer(_clover_odd_inv));
  });
}

int applyCloverBistabCgReconstructQcu(
    long long _full_out, long long _full_rhs, long long _target_solution,
    long long _gauge, long long _clover_even, long long _clover_odd,
    long long _clover_even_inv, long long _clover_odd_inv,
    long long _set_ptrs, long long _params) {
  return kupl::qcu::api::status_guarded("CLOVER_BISTABCG_RECONSTRUCT", [&] {
    return kupl::qcu::api::lattice_set(_set_ptrs, _params)
        ->clover_bistabcg_reconstruct(
            kupl::qcu::api::mutable_pointer(_full_out),
            kupl::qcu::api::pointer(_full_rhs),
            kupl::qcu::api::pointer(_target_solution),
            kupl::qcu::api::pointer(_gauge),
            kupl::qcu::api::pointer(_clover_even),
            kupl::qcu::api::pointer(_clover_odd),
            kupl::qcu::api::pointer(_clover_even_inv),
            kupl::qcu::api::pointer(_clover_odd_inv));
  });
}

} // extern "C"
