#ifndef KUPL_QCU_LATTICE_SET_H
#define KUPL_QCU_LATTICE_SET_H

#include "kupl/lattice_complex.h"
#include "kupl/lattice_cpu.h"
#include "kupl/lattice_geometry.h"
#include "kupl/lattice_multigrid.h"

#include <array>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <vector>

namespace kupl::qcu {

class LatticeSetBase {
public:
  virtual ~LatticeSetBase() = default;
  virtual void update_params(const int32_t *params) = 0;
  virtual void wilson_dslash(const void *out, const void *in,
                             const void *gauge) = 0;
  virtual void wilson_bistabcg(void *out, const void *in,
                               const void *gauge) = 0;
  virtual void wilson_bistabcg_dslash(void *out, const void *in,
                                      const void *gauge) = 0;
  virtual void wilson_cg(void *out, const void *in, const void *gauge) = 0;
  virtual void wilson_cg_dslash(void *out, const void *in,
                                const void *gauge) = 0;
  virtual void clover(void *clover_out, const void *gauge) = 0;
  virtual void clovers(void *clover_out, void *clover_inv,
                       const void *gauge) = 0;
  virtual void clover_dslash(void *out, const void *in,
                             const void *gauge) = 0;
  virtual void supplied_clover_dslash(void *out, const void *in,
                                      const void *gauge,
                                      const void *clover) = 0;
  virtual void clover_bistabcg(
      void *out, const void *in, const void *gauge, const void *clover_even,
      const void *clover_odd, const void *clover_even_inv,
      const void *clover_odd_inv) = 0;
  virtual void clover_bistabcg_dslash(
      void *out, const void *in, const void *gauge, const void *clover_even,
      const void *clover_odd, const void *clover_even_inv,
      const void *clover_odd_inv) = 0;
  virtual int clover_bistabcg_prepare(
      void *compact_rhs, const void *full_rhs, const void *gauge,
      const void *clover_even, const void *clover_odd,
      const void *clover_even_inv, const void *clover_odd_inv) = 0;
  virtual int clover_bistabcg_reconstruct(
      void *full_out, const void *full_rhs, const void *target_solution,
      const void *gauge, const void *clover_even, const void *clover_odd,
      const void *clover_even_inv, const void *clover_odd_inv) = 0;
  virtual void gauss_gauge(void *gauge) = 0;
  virtual void multigrid_restrict(
      void *coarse_out, const void *fine_in, const void *null_vectors,
      int E, int e, int Xf, int Yf, int Zf, int Tf,
      int Xc, int Yc, int Zc, int Tc) = 0;
  virtual void multigrid_prolong(
      void *fine_out, const void *coarse_in, const void *null_vectors,
      int E, int e, int Xf, int Yf, int Zf, int Tf,
      int Xc, int Yc, int Zc, int Tc) = 0;
  virtual void multigrid_coarse_dslash(
      void *out, const void *in, const void *hopping, const void *sitting,
      int E, int X, int Y, int Z, int T) = 0;
  virtual void multigrid_coarse_dslash_wide(
      void *out, const void *in, const void *sitting, const void *hop_nn,
      const void *hop_diag, int E, int X, int Y, int Z, int T) = 0;
  virtual void laplacian(void *out, const void *in, const void *gauge) = 0;
  virtual int strict_fine_matpc(
      void *out, const void *in, const void *gauge, const void *clover_even,
      const void *clover_odd, const void *clover_even_inv,
      const void *clover_odd_inv, int parity) = 0;
  virtual int strict_fine_prepare(
      void *compact_rhs, const void *full_rhs, const void *gauge,
      const void *clover_even, const void *clover_odd,
      const void *clover_even_inv, const void *clover_odd_inv,
      int parity) = 0;
  virtual int strict_fine_reconstruct(
      void *full_out, const void *full_rhs, const void *target_solution,
      const void *gauge, const void *clover_even, const void *clover_odd,
      const void *clover_even_inv, const void *clover_odd_inv,
      int parity) = 0;
  virtual void clover_full_apply(
      void *out, const void *in, const void *gauge, const void *clover_even,
      const void *clover_odd, const void *clover_even_inv,
      const void *clover_odd_inv) = 0;
  virtual double real_parameter(int index) const = 0;
  virtual int iterations() const = 0;
};

template <typename Real>
class LatticeSet final : public LatticeSetBase {
public:
  LatticeSet(const int32_t *params, const Real *argv);

  void update_params(const int32_t *params) override;
  void wilson_dslash(const void *out, const void *in,
                     const void *gauge) override;
  void wilson_bistabcg(void *out, const void *in, const void *gauge) override;
  void wilson_bistabcg_dslash(void *out, const void *in,
                              const void *gauge) override;
  void wilson_cg(void *out, const void *in, const void *gauge) override;
  void wilson_cg_dslash(void *out, const void *in,
                        const void *gauge) override;
  void clover(void *clover_out, const void *gauge) override;
  void clovers(void *clover_out, void *clover_inv,
               const void *gauge) override;
  void clover_dslash(void *out, const void *in, const void *gauge) override;
  void supplied_clover_dslash(void *out, const void *in, const void *gauge,
                              const void *clover) override;
  void clover_bistabcg(
      void *out, const void *in, const void *gauge, const void *clover_even,
      const void *clover_odd, const void *clover_even_inv,
      const void *clover_odd_inv) override;
  void clover_bistabcg_dslash(
      void *out, const void *in, const void *gauge, const void *clover_even,
      const void *clover_odd, const void *clover_even_inv,
      const void *clover_odd_inv) override;
  int clover_bistabcg_prepare(
      void *compact_rhs, const void *full_rhs, const void *gauge,
      const void *clover_even, const void *clover_odd,
      const void *clover_even_inv, const void *clover_odd_inv) override;
  int clover_bistabcg_reconstruct(
      void *full_out, const void *full_rhs, const void *target_solution,
      const void *gauge, const void *clover_even, const void *clover_odd,
      const void *clover_even_inv, const void *clover_odd_inv) override;
  void gauss_gauge(void *gauge_out) override;
  void multigrid_restrict(
      void *coarse_out, const void *fine_in, const void *null_vectors,
      int E, int e, int Xf, int Yf, int Zf, int Tf,
      int Xc, int Yc, int Zc, int Tc) override;
  void multigrid_prolong(
      void *fine_out, const void *coarse_in, const void *null_vectors,
      int E, int e, int Xf, int Yf, int Zf, int Tf,
      int Xc, int Yc, int Zc, int Tc) override;
  void multigrid_coarse_dslash(
      void *out, const void *in, const void *hopping, const void *sitting,
      int E, int X, int Y, int Z, int T) override;
  void multigrid_coarse_dslash_wide(
      void *out, const void *in, const void *sitting, const void *hop_nn,
      const void *hop_diag, int E, int X, int Y, int Z, int T) override;
  void laplacian(void *out, const void *in, const void *gauge) override;
  int strict_fine_matpc(
      void *out, const void *in, const void *gauge, const void *clover_even,
      const void *clover_odd, const void *clover_even_inv,
      const void *clover_odd_inv, int parity) override;
  int strict_fine_prepare(
      void *compact_rhs, const void *full_rhs, const void *gauge,
      const void *clover_even, const void *clover_odd,
      const void *clover_even_inv, const void *clover_odd_inv,
      int parity) override;
  int strict_fine_reconstruct(
      void *full_out, const void *full_rhs, const void *target_solution,
      const void *gauge, const void *clover_even, const void *clover_odd,
      const void *clover_even_inv, const void *clover_odd_inv,
      int parity) override;
  void clover_full_apply(
      void *out, const void *in, const void *gauge, const void *clover_even,
      const void *clover_odd, const void *clover_even_inv,
      const void *clover_odd_inv) override;
  double real_parameter(int index) const override {
    return static_cast<double>(parameters_.reals[index]);
  }
  int iterations() const override { return last_iterations_; }

private:
  LatticeGeometry geometry_;
  RuntimeParameters<Real> parameters_{};
  int threads_ = 1;
  int last_iterations_ = 0;
  GaugeHaloBuffer<Real> gauge_halo_;
  std::array<std::vector<Complex<Real>>, 8> fermion_send_;
  std::array<std::vector<Complex<Real>>, 8> fermion_recv_;
  std::vector<MPI_Request> fermion_requests_;
  std::array<std::vector<Complex<Real>>, 8> laplacian_source_send_;
  std::array<std::vector<Complex<Real>>, 8> laplacian_source_recv_;
  std::array<std::vector<Complex<Real>>, 8> laplacian_gauge_send_;
  std::array<std::vector<Complex<Real>>, 8> laplacian_gauge_recv_;
  std::vector<MPI_Request> laplacian_requests_;
  std::array<std::vector<Complex<Real>>, 2> cached_clover_inv_;
  std::array<const Complex<Real> *, 2> cached_gauge_{nullptr, nullptr};
  std::vector<Complex<Real>> r_;
  std::vector<Complex<Real>> rhat_;
  std::vector<Complex<Real>> p_;
  std::vector<Complex<Real>> v_;
  std::vector<Complex<Real>> s_;
  std::vector<Complex<Real>> t_;
  std::vector<Complex<Real>> bp_;
  std::vector<Complex<Real>> tmp_a_;
  std::vector<Complex<Real>> tmp_b_;
  std::vector<Complex<Real>> work_a_;
  std::vector<Complex<Real>> work_b_;
  std::vector<Complex<Real>> work_c_;
  int coarse_halo_E_ = 0;
  std::array<int, kDim> coarse_halo_dims_{};
  std::array<std::vector<Complex<Real>>, kCoarseDirections>
      coarse_halo_send_;
  std::array<std::vector<Complex<Real>>, kCoarseDirections>
      coarse_halo_recv_;
  std::array<int, kCoarseDirections> coarse_halo_offsets_{};
  std::vector<int64_t> boundary_sites_;
  double prof_dslash_halo_seconds_ = 0;
  double prof_dslash_interior_seconds_ = 0;
  double prof_dslash_wait_seconds_ = 0;
  double prof_dslash_boundary_seconds_ = 0;
  int64_t prof_dslash_calls_ = 0;

  void allocate_scratch();
  void ensure_solver_scratch();
  Real mass() const;
  Real kappa() const;
  Real csw() const;
  Real atol() const;
  Real residual_floor() const;

  void require_wilson_mode() const;
  void require_gauge_mode() const;
  void require_solver_mode() const;
  void require_laplacian_mode() const;
  void require_wilson_gauge(const Complex<Real> *gauge);
  void validate_parity() const;
  int target_parity() const;

  template <typename... Args>
  void log(const char *submodule, const char *format, Args... args) const {
    if (parameters_.get(kVerbose) == 0) {
      return;
    }
    std::fprintf(stderr, "PYQCU::KUPL::%s:\n ", submodule);
    std::fprintf(stderr, format, args...);
    std::fprintf(stderr, "\n");
  }

  int64_t site_linear(int x, int y, int z, int t2) const;
  int surface_extent(int dim) const;
  void decode_face(int64_t face, int fixed_dim, int fixed_value,
                   int &x, int &y, int &z, int &t2) const;
  void decode_corner(int64_t corner, int fixed_dim0, int fixed_value0,
                     int fixed_dim1, int fixed_value1, int &x, int &y,
                     int &z, int &t2) const;
  VectorView<Real> compact_view(const Complex<Real> *field,
                                int64_t site) const;
  VectorView<Real> halo_view(const std::vector<Complex<Real>> &halo,
                             int64_t face_site) const;
  void gather_vector(const VectorView<Real> &view,
                     Complex<Real> local[kVector]) const;
  void scatter_vector(const Complex<Real> local[kVector],
                      Complex<Real> *field, int64_t site) const;
  std::size_t gauge_index(int parity, int row, int column, int direction,
                          int x, int y, int z, int t2) const;
  GaugeView<Real> local_gauge(int parity, int direction, int x, int y, int z,
                              int t2) const;

  void pack_gauge_face(int dim, int boundary, Complex<Real> *buffer);
  void pack_gauge_corner(int dim0, int dim1, int boundary0, int boundary1,
                         Complex<Real> *buffer);
  void ensure_gauge_halo(const Complex<Real> *gauge);
  GaugeView<Real> gauge_at_delta(int x, int y, int z, int t2, int parity,
                                 int direction, const int delta[kDim]) const;
  GaugeView<Real> backward_gauge(int x, int y, int z, int t2, int direction,
                                 int target) const;
  void load_gauge_matrix(const GaugeView<Real> &source,
                         Complex<Real> matrix[9]) const;

  void compute_plaquette_sum(int x, int y, int z, int t2, int mu, int nu,
                             int parity, Complex<Real> sum[9]) const;
  void compute_clover(Complex<Real> *output, int parity);
  void invert_clover(const Complex<Real> *term, Complex<Real> *inverse);
  void ensure_cached_clover(const Complex<Real> *gauge);
  void apply_onsite(const Complex<Real> *input, const Complex<Real> *matrix,
                    Complex<Real> *output,
                    const Complex<Real> *subtract = nullptr,
                    Real subtract_scale = Real(0)) const;
  void apply_clover_onsite(
      const Complex<Real> *input, const Complex<Real> *matrix,
      Complex<Real> *output, const Complex<Real> *subtract = nullptr,
      Real subtract_scale = Real(0)) const;

  void pack_fermion_face(const Complex<Real> *source, int parity, int dim,
                         int boundary, Complex<Real> *buffer);
  void post_fermion_halo(const Complex<Real> *source, int parity);
  VectorView<Real> source_at_delta(const Complex<Real> *local_source, int x,
                                   int y, int z, int t2, int dim, int sign,
                                   int target) const;
  void compute_dslash_interior(const Complex<Real> *source,
                               Complex<Real> *output, int target);
  void compute_dslash_boundary(const Complex<Real> *local_source,
                               Complex<Real> *output, int target);
  void compute_dslash_site(const Complex<Real> *local_source,
                           Complex<Real> *output, int x, int y, int z,
                           int t2, int target);
  void compute_dslash_dagger_interior(const Complex<Real> *source,
                                      Complex<Real> *output, int target);
  void compute_dslash_dagger_boundary(const Complex<Real> *local_source,
                                      Complex<Real> *output, int target);
  void compute_dslash_dagger_site(const Complex<Real> *local_source,
                                  Complex<Real> *output, int x, int y, int z,
                                  int t2, int target);
  void bare_dslash(Complex<Real> *output, const Complex<Real> *input,
                   int target);
  void bare_dslash_dagger(Complex<Real> *output, const Complex<Real> *input,
                          int target);
  void apply_schur(
      const Complex<Real> *input, Complex<Real> *output, int parity,
      const std::array<const Complex<Real> *, 2> &clover,
      const std::array<const Complex<Real> *, 2> &inverse,
      Complex<Real> *scratch_a, Complex<Real> *scratch_b);
  void apply_wilson_schur(const Complex<Real> *input, Complex<Real> *output,
                          int parity, Complex<Real> *scratch_a,
                          Complex<Real> *scratch_b);
  void apply_wilson_schur_dagger(const Complex<Real> *input,
                                 Complex<Real> *output, int parity,
                                 Complex<Real> *scratch_a,
                                 Complex<Real> *scratch_b);
  void apply_wilson_cg_dslash(const Complex<Real> *input,
                              Complex<Real> *output, int parity);
  void apply_full_wilson_operator(const Complex<Real> *input,
                                  Complex<Real> *output,
                                  Complex<Real> *scratch);

  Complex<Real> global_dot(const Complex<Real> *left,
                           const Complex<Real> *right, int64_t count) const;
  Real global_norm2(const Complex<Real> *field, int64_t count) const;
  void apply_full_operator(
      const Complex<Real> *input, Complex<Real> *output,
      const std::array<const Complex<Real> *, 2> &clover,
      Complex<Real> *scratch);

  int64_t laplacian_site(int x, int y, int z) const;
  std::size_t laplacian_gauge_index(int row, int column, int direction, int x,
                                    int y, int z) const;
  void decode_laplacian_site(int64_t index, int &x, int &y, int &z) const;
  void pack_laplacian_face(int dim, int boundary,
                           const Complex<Real> *source,
                           const Complex<Real> *gauge,
                           Complex<Real> *source_buffer,
                           Complex<Real> *gauge_buffer);
  void post_laplacian_halo(const Complex<Real> *source,
                           const Complex<Real> *gauge);
  void compute_laplacian_local(const Complex<Real> *source,
                               const Complex<Real> *gauge,
                               Complex<Real> *output);
  void subtract_laplacian_boundary(const Complex<Real> *gauge,
                                   Complex<Real> *output);
  void apply_laplacian(Complex<Real> *output, const Complex<Real> *input,
                       const Complex<Real> *gauge);

  int64_t coarse_volume(const std::array<int, kDim> &dims) const;
  int64_t coarse_site_index(const std::array<int, kDim> &dims, int x, int y,
                            int z, int t) const;
  void decode_coarse_site(int64_t site, const std::array<int, kDim> &dims,
                          int &x, int &y, int &z, int &t) const;
  int64_t coarse_face_sites(int direction,
                            const std::array<int, kDim> &dims) const;
  bool coarse_direction_distributed(int direction) const;
  void decode_coarse_face(int direction, int64_t face,
                          const std::array<int, kDim> &dims,
                          std::array<int, kDim> &coordinate) const;
  int64_t coarse_face_index(
      int direction, const std::array<int, kDim> &coordinate,
      const std::array<int, kDim> &dims) const;
  void ensure_coarse_halo(int E, const std::array<int, kDim> &dims);
  void pack_coarse_direction(const Complex<Real> *input, int E,
                             const std::array<int, kDim> &dims,
                             int direction);
  int coarse_peer(int direction) const;
  void exchange_coarse_halo(const Complex<Real> *input, int E,
                            const std::array<int, kDim> &dims);
  Complex<Real> coarse_neighbor(
      const Complex<Real> *input, int E, const std::array<int, kDim> &dims,
      const std::array<int, kDim> &coordinate, const int shift[kDim],
      int e) const;
  std::size_t multigrid_null_index(
      int E_idx, int e_idx, int x, int y, int z, int t, int e,
      int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc) const;
  void validate_multigrid_transfer(int E, int e, int Xf, int Yf, int Zf,
                                   int Tf, int Xc, int Yc, int Zc,
                                   int Tc) const;
  void apply_multigrid_restrict(
      Complex<Real> *coarse_out, const Complex<Real> *fine_in,
      const Complex<Real> *null_vectors, int E, int e, int Xf, int Yf,
      int Zf, int Tf, int Xc, int Yc, int Zc, int Tc);
  void apply_multigrid_prolong(
      Complex<Real> *fine_out, const Complex<Real> *coarse_in,
      const Complex<Real> *null_vectors, int E, int e, int Xf, int Yf,
      int Zf, int Tf, int Xc, int Yc, int Zc, int Tc);
  void validate_coarse_operator(const void *out, const void *in, int E, int X,
                                int Y, int Z, int T) const;
  void apply_multigrid_coarse_dslash(
      Complex<Real> *out, const Complex<Real> *in,
      const Complex<Real> *hopping, const Complex<Real> *sitting, int E,
      int X, int Y, int Z, int T);
  void apply_multigrid_coarse_dslash_wide(
      Complex<Real> *out, const Complex<Real> *in,
      const Complex<Real> *sitting, const Complex<Real> *hop_nn,
      const Complex<Real> *hop_diag, int E, int X, int Y, int Z, int T);

  template <typename Operator>
  void solve_compact_bistabcg(Complex<Real> *solution,
                              const Complex<Real> *rhs, int64_t count,
                              Operator &&apply_operator, const char *label,
                              Real relative_tolerance = Real(-1));
  template <typename Operator>
  void solve_compact_cg(Complex<Real> *solution, const Complex<Real> *rhs,
                        int64_t count, Operator &&apply_operator,
                        const char *label, Real relative_tolerance);
  void prepare_clover_schur(
      Complex<Real> *compact_rhs, const Complex<Real> *full_rhs, int parity,
      const std::array<const Complex<Real> *, 2> &inverse);
  void reconstruct_clover_schur(
      Complex<Real> *full_out, const Complex<Real> *full_rhs,
      const Complex<Real> *target_solution, int parity,
      const std::array<const Complex<Real> *, 2> &inverse);
  void reconstruct_wilson(Complex<Real> *full_output,
                          const Complex<Real> *full_input, int parity);
  Real check_wilson_full_residual(Complex<Real> *full_output,
                                  const Complex<Real> *full_input,
                                  const char *label);
  void solve_wilson_bistabcg(Complex<Real> *full_output,
                             const Complex<Real> *full_input);
  void solve_wilson_cg(Complex<Real> *full_output,
                       const Complex<Real> *full_input);
  void solve_clover(Complex<Real> *full_output,
                    const Complex<Real> *full_input, int parity,
                    const std::array<const Complex<Real> *, 2> &clover,
                    const std::array<const Complex<Real> *, 2> &inverse);
};

std::shared_ptr<LatticeSetBase> make_lattice_set(const int32_t *params,
                                                 const void *argv);
std::shared_ptr<LatticeSetBase> register_lattice_set(const void *set_ptrs,
                                                     const int32_t *params,
                                                     const void *argv);
std::shared_ptr<LatticeSetBase> find_lattice_set(const void *set_ptrs);
void erase_lattice_set(const void *set_ptrs);

[[noreturn]] void abort_with_message(const char *submodule,
                                     const char *message);

} // namespace kupl::qcu

#endif
