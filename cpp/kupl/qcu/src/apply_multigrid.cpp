#include "kupl/qcu.h"

#include "api_common.hpp"

extern "C" {

void applyMultigridRestrictQcu(long long _coarse_out, long long _fine_in,
                               long long _null_vectors, long long _set_ptrs,
                               long long _params) {
  kupl::qcu::api::guarded("MULTIGRID_RESTRICT", [&] {
    const auto *params = kupl::qcu::api::params_pointer(_params);
    kupl::qcu::api::lattice_set(_set_ptrs, _params)
        ->multigrid_restrict(
            kupl::qcu::api::mutable_pointer(_coarse_out),
            kupl::qcu::api::pointer(_fine_in),
            kupl::qcu::api::pointer(_null_vectors),
            params[kupl::qcu::kMgLevel1E], kupl::qcu::kVectorSize,
            params[kupl::qcu::kLatX], params[kupl::qcu::kLatY],
            params[kupl::qcu::kLatZ], params[kupl::qcu::kLatT],
            params[kupl::qcu::kMgLevel1X], params[kupl::qcu::kMgLevel1Y],
            params[kupl::qcu::kMgLevel1Z], params[kupl::qcu::kMgLevel1T]);
  });
}

void applyMultigridProLongQcu(long long _fine_out, long long _coarse_in,
                              long long _null_vectors, long long _set_ptrs,
                              long long _params) {
  kupl::qcu::api::guarded("MULTIGRID_PROLONG", [&] {
    const auto *params = kupl::qcu::api::params_pointer(_params);
    kupl::qcu::api::lattice_set(_set_ptrs, _params)
        ->multigrid_prolong(
            kupl::qcu::api::mutable_pointer(_fine_out),
            kupl::qcu::api::pointer(_coarse_in),
            kupl::qcu::api::pointer(_null_vectors),
            params[kupl::qcu::kMgLevel1E], kupl::qcu::kVectorSize,
            params[kupl::qcu::kLatX], params[kupl::qcu::kLatY],
            params[kupl::qcu::kLatZ], params[kupl::qcu::kLatT],
            params[kupl::qcu::kMgLevel1X], params[kupl::qcu::kMgLevel1Y],
            params[kupl::qcu::kMgLevel1Z], params[kupl::qcu::kMgLevel1T]);
  });
}

void applyMultigridCoarseDslashQcu(long long _out, long long _in,
                                   long long _hopping, long long _sitting,
                                   long long _set_ptrs, long long _params) {
  kupl::qcu::api::guarded("MULTIGRID_COARSE_DSLASH", [&] {
    const auto *params = kupl::qcu::api::params_pointer(_params);
    kupl::qcu::api::lattice_set(_set_ptrs, _params)
        ->multigrid_coarse_dslash(
            kupl::qcu::api::mutable_pointer(_out),
            kupl::qcu::api::pointer(_in),
            kupl::qcu::api::pointer(_hopping),
            kupl::qcu::api::pointer(_sitting),
            params[kupl::qcu::kMgLevel1E], params[kupl::qcu::kMgLevel1X],
            params[kupl::qcu::kMgLevel1Y], params[kupl::qcu::kMgLevel1Z],
            params[kupl::qcu::kMgLevel1T]);
  });
}

void applyMultigridCoarseDslashWideQcu(
    long long _out, long long _in, long long _sitting, long long _hop_nn,
    long long _hop_diag, long long _set_ptrs, long long _params) {
  kupl::qcu::api::guarded("MULTIGRID_COARSE_DSLASH_WIDE", [&] {
    const auto *params = kupl::qcu::api::params_pointer(_params);
    kupl::qcu::api::lattice_set(_set_ptrs, _params)
        ->multigrid_coarse_dslash_wide(
            kupl::qcu::api::mutable_pointer(_out),
            kupl::qcu::api::pointer(_in),
            kupl::qcu::api::pointer(_sitting),
            kupl::qcu::api::pointer(_hop_nn),
            kupl::qcu::api::pointer(_hop_diag),
            params[kupl::qcu::kMgLevel1E], params[kupl::qcu::kMgLevel1X],
            params[kupl::qcu::kMgLevel1Y], params[kupl::qcu::kMgLevel1Z],
            params[kupl::qcu::kMgLevel1T]);
  });
}

} // extern "C"
