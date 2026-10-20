#ifndef KUPL_QCU_BACKEND_HPP
#define KUPL_QCU_BACKEND_HPP

#include "kupl/define.h"

#include <array>
#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

#include <mpi.h>

namespace kupl::qcu {

template <typename Real> using Complex = std::complex<Real>;

class BackendBase {
public:
  virtual ~BackendBase() = default;
  virtual void updateParams(const int32_t *params) = 0;
  virtual void wilsonDslash(const void *out, const void *in,
                            const void *gauge) = 0;
  virtual void wilsonBistabCg(void *out, const void *in,
                              const void *gauge) = 0;
  virtual void wilsonBistabCgDslash(void *out, const void *in,
                                    const void *gauge) = 0;
  virtual void wilsonCg(void *out, const void *in, const void *gauge) = 0;
  virtual void wilsonCgDslash(void *out, const void *in,
                              const void *gauge) = 0;
  virtual void clover(void *clover_out, const void *gauge) = 0;
  virtual void clovers(void *clover_out, void *clover_inv,
                       const void *gauge) = 0;
  virtual void cloverDslash(void *out, const void *in,
                            const void *gauge) = 0;
  virtual void suppliedCloverDslash(void *out, const void *in,
                                    const void *gauge, const void *clover) = 0;
  virtual void cloverBistabCg(
      void *out, const void *in, const void *gauge, const void *clover_even,
      const void *clover_odd, const void *clover_even_inv,
      const void *clover_odd_inv) = 0;
  virtual void cloverBistabCgDslash(
      void *out, const void *in, const void *gauge, const void *clover_even,
      const void *clover_odd, const void *clover_even_inv,
      const void *clover_odd_inv) = 0;
  virtual int cloverBistabCgPrepare(
      void *compact_rhs, const void *full_rhs, const void *gauge,
      const void *clover_even, const void *clover_odd,
      const void *clover_even_inv, const void *clover_odd_inv) = 0;
  virtual int cloverBistabCgReconstruct(
      void *full_out, const void *full_rhs, const void *target_solution,
      const void *gauge, const void *clover_even, const void *clover_odd,
      const void *clover_even_inv, const void *clover_odd_inv) = 0;
  virtual void gaussGauge(void *gauge) = 0;
  virtual void multigridRestrict(
      void *coarse_out, const void *fine_in, const void *null_vectors,
      int E, int e, int Xf, int Yf, int Zf, int Tf,
      int Xc, int Yc, int Zc, int Tc) = 0;
  virtual void multigridProlong(
      void *fine_out, const void *coarse_in, const void *null_vectors,
      int E, int e, int Xf, int Yf, int Zf, int Tf,
      int Xc, int Yc, int Zc, int Tc) = 0;
  virtual void multigridCoarseDslash(
      void *out, const void *in, const void *hopping, const void *sitting,
      int E, int X, int Y, int Z, int T) = 0;
  virtual void multigridCoarseDslashWide(
      void *out, const void *in, const void *sitting, const void *hop_nn,
      const void *hop_diag, int E, int X, int Y, int Z, int T) = 0;
  virtual void laplacian(void *out, const void *in, const void *gauge) = 0;
  virtual int iterations() const = 0;
};

void initializeMpi();
std::shared_ptr<BackendBase> makeBackend(const int32_t *params,
                                         const void *argv);

std::shared_ptr<BackendBase> registerBackend(const void *set_ptrs,
                                             const int32_t *params,
                                             const void *argv);
std::shared_ptr<BackendBase> findBackend(const void *set_ptrs);
void eraseBackend(const void *set_ptrs);

[[noreturn]] void abortWithMessage(const char *submodule,
                                   const char *message);

} // namespace kupl::qcu

#endif
