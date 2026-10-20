#include "kupl/qcu.h"

#include "api_common.hpp"

extern "C" {

int applyMultigridStrictCoarseQcu(long long, long long, long long, long long,
                                  long long, long long, int, int, int, int,
                                  int, int) {
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_STRICT_COARSE",
      [] {
        return kupl::qcu::api::unavailable_status(
            "applyMultigridStrictCoarseQcu");
      });
}

int applyMultigridStrictMatPCQcu(long long, long long, long long, long long,
                                 long long, long long, int, int, int, int,
                                 int, int) {
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_STRICT_MATPC",
      [] {
        return kupl::qcu::api::unavailable_status(
            "applyMultigridStrictMatPCQcu");
      });
}

int applyMultigridStrictFineMatPCQcu(long long, long long, long long,
                                     long long, long long, long long,
                                     long long, long long, long long, int) {
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_STRICT_FINE_MATPC",
      [] {
        return kupl::qcu::api::unavailable_status(
            "applyMultigridStrictFineMatPCQcu");
      });
}

int applyMultigridStrictPrepareQcu(long long, long long, long long, long long,
                                   long long, long long, long long, int, int,
                                   int, int, int, int) {
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_STRICT_PREPARE",
      [] {
        return kupl::qcu::api::unavailable_status(
            "applyMultigridStrictPrepareQcu");
      });
}

int applyMultigridStrictReconstructQcu(long long, long long, long long,
                                       long long, long long, long long,
                                       long long, long long, int, int, int,
                                       int, int, int) {
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_STRICT_RECONSTRUCT",
      [] {
        return kupl::qcu::api::unavailable_status(
            "applyMultigridStrictReconstructQcu");
      });
}

int applyMultigridStrictRestrictQcu(long long, long long, long long,
                                    long long, long long, int, int, int, int,
                                    int, int, int, int, int, int, int) {
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_STRICT_RESTRICT",
      [] {
        return kupl::qcu::api::unavailable_status(
            "applyMultigridStrictRestrictQcu");
      });
}

int applyMultigridStrictProLongQcu(long long, long long, long long, long long,
                                   long long, int, int, int, int, int, int,
                                   int, int, int, int, int) {
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_STRICT_PROLONG",
      [] {
        return kupl::qcu::api::unavailable_status(
            "applyMultigridStrictProLongQcu");
      });
}

int applyMultigridStrictVCycleQcu(long long, long long, long long, long long,
                                  int, unsigned long long *) {
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_STRICT_VCYCLE",
      [] {
        return kupl::qcu::api::unavailable_status(
            "applyMultigridStrictVCycleQcu");
      });
}

int applyMultigridStrictInitQcu(long long, long long, int,
                                unsigned long long *) {
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_STRICT_INIT",
      [] {
        return kupl::qcu::api::unavailable_status(
            "applyMultigridStrictInitQcu");
      });
}

int applyMultigridStrictEndQcu(long long, long long) {
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_STRICT_END",
      [] {
        return kupl::qcu::api::unavailable_status(
            "applyMultigridStrictEndQcu");
      });
}

int applyMultigridStrictFgmresQcu(
    long long, long long, long long, long long, long long, long long,
    long long, long long, long long, long long, int, int, int, int, int,
    int, int, int, int, int, int, int, int, double, int, int,
    unsigned long long, int *, int *, double *, unsigned long long *) {
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_STRICT_FGMRES",
      [] {
        return kupl::qcu::api::unavailable_status(
            "applyMultigridStrictFgmresQcu");
      });
}

} // extern "C"
