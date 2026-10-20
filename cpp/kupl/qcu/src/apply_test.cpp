#include "kupl/qcu.h"

extern "C" {

void testWilsonDslashQcu(long long _fermion_out, long long _fermion_in,
                         long long _gauge, long long _set_ptrs,
                         long long _params) {
  applyWilsonDslashQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs,
                       _params);
}

void testCloverDslashQcu(long long _fermion_out, long long _fermion_in,
                         long long _gauge, long long _set_ptrs,
                         long long _params) {
  applyCloverDslashQcu(_fermion_out, _fermion_in, _gauge, _set_ptrs,
                       _params);
}

} // extern "C"
