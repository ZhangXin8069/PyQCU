#include "kupl/qcu.h"

#include "api_common.hpp"

extern "C" {

void applyCloverMultigridQcu(long long, long long, long long, long long,
                             long long, long long, long long, long long,
                             long long) {
  kupl::qcu::api::guarded(
      "CLOVER_MULTIGRID",
      [] { kupl::qcu::api::unavailable("applyCloverMultigridQcu"); });
}

} // extern "C"
