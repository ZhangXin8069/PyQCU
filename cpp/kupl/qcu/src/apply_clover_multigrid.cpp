#include "kupl/qcu.h"
#include "kupl/lattice_clover_multigrid.h"

#include "api_common.hpp"

extern "C" {

void applyCloverMultigridQcu(
    long long _fermion_out, long long _fermion_in, long long _gauge,
    long long _clover_even, long long _clover_odd,
    long long _clover_even_inv, long long _clover_odd_inv,
    long long _set_ptrs, long long _params) {
  kupl::qcu::api::guarded("CLOVER_MULTIGRID", [&] {
    const auto *params = kupl::qcu::api::params_pointer(_params);
    auto set = kupl::qcu::api::lattice_set(_set_ptrs, _params);
    if (params[kupl::qcu::kDataType] == kupl::qcu::kLatC64) {
      kupl::qcu::clover_multigrid_solve<float>(
          *set, kupl::qcu::api::mutable_pointer(_fermion_out),
          kupl::qcu::api::pointer(_fermion_in),
          kupl::qcu::api::pointer(_gauge),
          kupl::qcu::api::pointer(_clover_even),
          kupl::qcu::api::pointer(_clover_odd),
          kupl::qcu::api::pointer(_clover_even_inv),
          kupl::qcu::api::pointer(_clover_odd_inv),
          kupl::qcu::api::mutable_pointer(_set_ptrs), params);
      return;
    }
    if (params[kupl::qcu::kDataType] == kupl::qcu::kLatC128) {
      kupl::qcu::clover_multigrid_solve<double>(
          *set, kupl::qcu::api::mutable_pointer(_fermion_out),
          kupl::qcu::api::pointer(_fermion_in),
          kupl::qcu::api::pointer(_gauge),
          kupl::qcu::api::pointer(_clover_even),
          kupl::qcu::api::pointer(_clover_odd),
          kupl::qcu::api::pointer(_clover_even_inv),
          kupl::qcu::api::pointer(_clover_odd_inv),
          kupl::qcu::api::mutable_pointer(_set_ptrs), params);
      return;
    }
    throw std::runtime_error("legacy Clover-MG supports c64/c128");
  });
}

} // extern "C"
