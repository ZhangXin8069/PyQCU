#include "kupl/qcu.h"

#include "api_common.hpp"

extern "C" {

void applyLaplacianQcu(long long _laplacian_out, long long _laplacian_in,
                       long long _gauge, long long _set_ptrs,
                       long long _params) {
  kupl::qcu::api::guarded("LAPLACIAN", [&] {
    kupl::qcu::api::lattice_set(_set_ptrs, _params)
        ->laplacian(kupl::qcu::api::mutable_pointer(_laplacian_out),
                    kupl::qcu::api::pointer(_laplacian_in),
                    kupl::qcu::api::pointer(_gauge));
  });
}

} // extern "C"
