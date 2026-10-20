#include "kupl/qcu.h"

#include "api_common.hpp"

extern "C" {

int verifyCloverMultigridQcu(long long, long long, long long, long long,
                             long long, long long, long long, long long,
                             long long) {
  return kupl::qcu::api::status_guarded(
      "CLOVER_MULTIGRID_VERIFY",
      [] {
        return kupl::qcu::api::unavailable_status(
            "verifyCloverMultigridQcu");
      });
}

} // extern "C"
