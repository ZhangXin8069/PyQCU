#ifndef _LATTICE_CLOVER_MULTIGRID_H
#define _LATTICE_CLOVER_MULTIGRID_H
/**
 * @file lattice_clover_multigrid.h
 * @brief Multi-threaded, multi-precision CUDA C++ Multigrid solver with BiStabCG
 *        smoothing and V-cycle coarse-grid correction.
 *
 * Algorithm: mirrors pyqcu/solver/_multigrid.py.
 * Target API:  applyCloverBistabCgDslashQcu (parity-preconditioned Clover dslash).
 *
 * KEY FIXES (2026-08-01):
 *   1. V-cycle restriction now uses FULL-SITE residual (even=0, odd=r_o_full)
 *      computed by reconstructing x_e and applying the full Dirac operator components.
 *   2. Prolongation result is converted from full-site to parity-split odd before
 *      adding to x_o (matching the Python code's e_fine = e_fine_eo[1]).
 *   3. BiStabCG state (p, v, s, t, rho_prev, alpha, omega) is fully reset after
 *      each V-cycle correction, preventing stale-state divergence.
 *   4. num_restart is read from params[_MG_LEVEL1_NUM_RESTART_] instead of hardcoded.
 *   5. Coarse-level smoothing uses tolerance-based convergence instead of fixed
 *      iteration counts.
 *   6. Full 5-stream synchronization at the bottom of each BiStabCG iteration.
 *   7. Parity-split ↔ full-site layout conversion kernels for MG inter-grid transfers.
 *
 * Sync pattern: matches LatticeCloverBistabCg::_run() exactly for maximum performance.
 * Stream architecture (5 streams, same as reference):
 *   main (strm):   dslash operations
 *   _a_:           dot(r_tilde,r) → give_1beta → give_p → give_s → give_r
 *   _b_:           give_1rho_prev → give_x_o
 *   _c_:           dot(t,s), dot(r,r) convergence check
 *   _d_:           dot(r_tilde,v) → give_1alpha → dot(t,t) → give_1omega
 */
#include "./bistabcg.h"
#include "./define.h"
#include "./lattice_clover_dslash.h"
#include "./lattice_cuda.h"
#include "./lattice_mpi.h"
#include "./lattice_wilson_dslash.h"
#include "./multigrid.h"
#include "./lattice_sap.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <complex>

namespace qcu {

/**
 * @brief Single-block reduction dot product for tiny coarse-level vectors.
 * Computes <a,b> = Σ conj(a[i])·b[i] into *out (device_vals[vals_idx]).
 * Coarse vectors are a few thousand complex elements, so this avoids the
 * cublasDot launch + D2H/H2D memcpy overhead that dominated coarse solves
 * (416 ms of a 1.9 s solve before this kernel was introduced).
 *
 * 2026-08-15 dev76: large-lattice coarse levels (16x16x16x32 lv1 has
 * E=48 × 8x8x8x8 = 196608 elements) made the single-block version the
 * dominant coarse-solve cost (~768 serial adds per thread).  The new
 * multi-block path coarse_dot_kernel_multi + coarse_dot_reduce_kernel
 * (grid-stride, then a 1-block second reduction) restores parallelism.
 */
template <typename T, int NT>
__global__ void coarse_dot_kernel(const LatticeComplex<T> *a,
                                  const LatticeComplex<T> *b, int n,
                                  LatticeComplex<T> *out) {
  __shared__ LatticeComplex<T> sdata[NT];
  int idx = threadIdx.x;
  LatticeComplex<T> sum(0, 0);
  for (int i = idx; i < n; i += NT) sum += a[i].conj() * b[i];
  sdata[idx] = sum;
  __syncthreads();
  for (int s = NT / 2; s > 0; s >>= 1) {
    if (idx < s) sdata[idx] += sdata[idx + s];
    __syncthreads();
  }
  if (idx == 0) out[0] = sdata[0];
}

// Multi-block grid-stride reduction: partials[bid] = block-local <a,b> slice.
template <typename T, int NT>
__global__ void coarse_dot_kernel_multi(const LatticeComplex<T> *a,
                                        const LatticeComplex<T> *b, int n,
                                        LatticeComplex<T> *partials) {
  __shared__ LatticeComplex<T> sdata[NT];
  int idx = threadIdx.x, bid = blockIdx.x, nblk = gridDim.x;
  int stride = nblk * NT;
  LatticeComplex<T> sum(0, 0);
  for (int i = bid * NT + idx; i < n; i += stride) sum += a[i].conj() * b[i];
  sdata[idx] = sum;
  __syncthreads();
  for (int s = NT / 2; s > 0; s >>= 1) {
    if (idx < s) sdata[idx] += sdata[idx + s];
    __syncthreads();
  }
  if (idx == 0) partials[bid] = sdata[0];
}

// Second reduction over nblk partials (1 block).
template <typename T, int NT>
__global__ void coarse_dot_reduce_kernel(const LatticeComplex<T> *partials,
                                         int nblk, LatticeComplex<T> *out) {
  __shared__ LatticeComplex<T> sdata[NT];
  int idx = threadIdx.x;
  LatticeComplex<T> sum(0, 0);
  for (int i = idx; i < nblk; i += NT) sum += partials[i];
  sdata[idx] = sum;
  __syncthreads();
  for (int s = NT / 2; s > 0; s >>= 1) {
    if (idx < s) sdata[idx] += sdata[idx + s];
    __syncthreads();
  }
  if (idx == 0) out[0] = sdata[0];
}

// ====================================================================
// dev84: DEVICE-SIDE CG scalars for the MG smoother (apply_mg_prec).
// --------------------------------------------------------------------
// The previous smoother read rr/pv to the host once per CG step (2 host
// syncs × μ_pre steps per V-cycle ≈ several ms of pure latency).  These
// kernels keep α/β/rr entirely in device_vals — the smoother loop then
// runs with ZERO host syncs (fixed step count, quda Nsteps semantics).
//   _tmp0_     : rr        (<r,r>, real)
//   _tmp1_     : pv        (<p,Ap>, real)
//   _rho_prev_ : rr_prev   (for β, free in the smoother context)
//   _alpha_    : α = rr/pv
//   _beta_     : β = rr_new/rr
// ====================================================================
template <typename T>
__global__ void mg_cg_give_alpha(LatticeComplex<T> *dv) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    T pv = dv[_tmp1_].real();
    if (pv < (T)0) pv = -pv;
    if (pv < (T)1e-30) { LatticeComplex<T> z((T)0, (T)0); dv[_alpha_] = z; }
    else {
      LatticeComplex<T> a(dv[_tmp0_].real() / pv, (T)0);
      dv[_alpha_] = a;
    }
  }
}

template <typename T>
__global__ void mg_cg_update_xr(LatticeComplex<T> *x, const LatticeComplex<T> *p,
                                LatticeComplex<T> *r, const LatticeComplex<T> *ap,
                                const LatticeComplex<T> *dv, int n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= (size_t)n) return;
  LatticeComplex<T> al = dv[_alpha_];
  x[i] = x[i] + al * p[i];
  r[i] = r[i] - al * ap[i];
}

template <typename T>
__global__ void mg_cg_give_beta(LatticeComplex<T> *dv) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    LatticeComplex<T> rrn = dv[_tmp0_], rrp = dv[_rho_prev_];
    T denom = rrp.real();
    LatticeComplex<T> b((T)0, (T)0);
    if (denom > (T)1e-30) b = LatticeComplex<T>(rrn.real() / denom, (T)0);
    dv[_beta_] = b;
    dv[_rho_prev_] = rrn;   // rr_prev <- rr_new
  }
}

template <typename T>
__global__ void mg_cg_update_p(LatticeComplex<T> *p, const LatticeComplex<T> *r,
                               const LatticeComplex<T> *dv, int n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= (size_t)n) return;
  p[i] = r[i] + dv[_beta_] * p[i];
}

// ====================================================================
// dev84: GUARDED BiCGStab scalar updates for the SYNC-FREE coarse solve.
// --------------------------------------------------------------------
// The coarse Schur operator A_c = Pᵀ S P of a Clover-Wilson Dirac is only
// γ5-Hermitian (S† = γ5 S γ5), NOT Hermitian — measured on 16x32x32x48:
// plain CG DIVERGES there (residual 6084 after 200 steps vs ‖rhs‖≈878),
// so BiCGStab remains the right algorithm class.  But a FIXED-STEP
// sync-free BiCGStab iterates far past convergence where ρ→0 breaks down
// (β = ρ/ρ_prev · α/ω → 0/0 NaN poisoning x_c).  These variants clamp
// every scalar against the stored problem scale (‖b‖² stashed in
// _diff2_tmp_) and against NaN/Inf, turning post-convergence iterations
// into harmless no-ops — the host never reads anything mid-solve.
//   scale units: dots entering these kernels are all ⟨·,·⟩ ~ ‖b‖².
//   thresholds : 1e-13·scale (double) — well below useful signal, far
//                above the fp64 denormal floor.
// ====================================================================
template <typename T> __device__ inline bool mg_bad(T re, T im) {
  return !((re == re) && (im == im) && fabs(re) != INFINITY &&
           fabs(im) != INFINITY);
}

template <typename T>
__global__ void mg_give_1beta(LatticeComplex<T> *vals) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    LatticeComplex<T> scale = vals[_diff2_tmp_];
    T sc = scale.real();
    LatticeComplex<T> rho = vals[_rho_], rp = vals[_rho_prev_];
    LatticeComplex<T> al = vals[_alpha_], om = vals[_omega_];
    bool bad = sc <= (T)0 ||
               mg_bad<T>(rho.real(), rho.imag()) ||
               mg_bad<T>(rp.real(), rp.imag()) ||
               mg_bad<T>(al.real(), al.imag()) ||
               mg_bad<T>(om.real(), om.imag());
    T mag_rho = fabs(rho.real()) + fabs(rho.imag());
    if (bad || mag_rho < (T)1e-13 * sc) {
      vals[_beta_] = LatticeComplex<T>((T)0, (T)0);
      vals[_rho_prev_] = LatticeComplex<T>((T)1, (T)0);
    } else {
      vals[_beta_] = (rho / rp) * (al / om);
    }
  }
}

template <typename T>
__global__ void mg_give_1alpha(LatticeComplex<T> *vals) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    LatticeComplex<T> scale = vals[_diff2_tmp_];
    T sc = scale.real();
    LatticeComplex<T> d = vals[_tmp0_];   // <r_tilde, v>
    bool bad = sc <= (T)0 || mg_bad<T>(d.real(), d.imag());
    if (bad || fabs(d.real()) + fabs(d.imag()) < (T)1e-13 * sc) {
      vals[_alpha_] = LatticeComplex<T>((T)0, (T)0);
    } else {
      vals[_alpha_] = vals[_rho_] / d;
    }
  }
}

template <typename T>
__global__ void mg_give_1omega(LatticeComplex<T> *vals) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    LatticeComplex<T> scale = vals[_diff2_tmp_];
    T sc = scale.real();
    LatticeComplex<T> num = vals[_tmp0_];   // <t,s>
    LatticeComplex<T> den = vals[_tmp1_];   // <t,t>
    bool bad = sc <= (T)0 || mg_bad<T>(num.real(), num.imag()) ||
               mg_bad<T>(den.real(), den.imag());
    if (bad || fabs(den.real()) + fabs(den.imag()) < (T)1e-13 * sc) {
      vals[_omega_] = LatticeComplex<T>((T)0, (T)0);
    } else {
      vals[_omega_] = num / den;
    }
  }
}

// dev84 kernel-count diet: β-update + ρ_prev←ρ fused (was 2 launches).
template <typename T>
__global__ void mg_give_1beta_rp(LatticeComplex<T> *vals) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    LatticeComplex<T> scale = vals[_diff2_tmp_];
    T sc = scale.real();
    LatticeComplex<T> rho = vals[_rho_], rp = vals[_rho_prev_];
    LatticeComplex<T> al = vals[_alpha_], om = vals[_omega_];
    bool bad = sc <= (T)0 ||
               mg_bad<T>(rho.real(), rho.imag()) ||
               mg_bad<T>(rp.real(), rp.imag()) ||
               mg_bad<T>(al.real(), al.imag()) ||
               mg_bad<T>(om.real(), om.imag());
    T mag_rho = fabs(rho.real()) + fabs(rho.imag());
    if (bad || mag_rho < (T)1e-13 * sc) {
      vals[_beta_] = LatticeComplex<T>((T)0, (T)0);
      vals[_rho_prev_] = LatticeComplex<T>((T)1, (T)0);
    } else {
      vals[_beta_] = (rho / rp) * (al / om);
      vals[_rho_prev_] = rho;
    }
  }
}

// dev84 kernel-count diet: r = s − ω·t  and  x += α·p + ω·s  fused (was 2
// launches over the same index space).
template <typename T>
__global__ void mg_give_rx(LatticeComplex<T> *r, const LatticeComplex<T> *s,
                           const LatticeComplex<T> *t, LatticeComplex<T> *x,
                           const LatticeComplex<T> *p,
                           const LatticeComplex<T> *dv, int n4) {
  // n4 counts _LAT_SC_-element groups (site-grid convention of give_r/give_x_o)
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n4) return;
  LatticeComplex<T> om = dv[_omega_], al = dv[_alpha_];
  size_t base = (size_t)i * _LAT_SC_;
  for (int k = 0; k < _LAT_SC_; k++) {
    size_t j = base + k;
    r[j] = s[j] - om * t[j];
    x[j] = x[j] + al * p[j] + om * s[j];
  }
}

// ---- Logging infrastructure ----
inline void ensure_log_dir() {
  struct stat st;
  const char* qdir = std::getenv("QCU_LOG_DIR");
  const char* dir = qdir ? qdir : "logs";
  if (stat(dir, &st) != 0) mkdir(dir, 0755);
}

template <typename T>
inline void log_write(const std::string &msg, int rank, bool to_stdout = true) {
  ensure_log_dir();
  const char* qdir = std::getenv("QCU_LOG_DIR");
  std::string log_path = std::string(qdir ? qdir : "logs") + "/clover_multigrid.log";
  std::ofstream f(log_path, std::ios_base::app);
  if (f.is_open()) {
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    f << std::put_time(std::localtime(&tt), "%Y-%m-%d %H:%M:%S")
      << " | " << msg << std::endl;
    f.close();
  }
  if (to_stdout && rank == 0) printf("%s\n", msg.c_str());
}

template <typename T> inline MPI_Datatype mpi_real_type() { return MPI_FLOAT; }
template <> inline MPI_Datatype mpi_real_type<double>() { return MPI_DOUBLE; }

// ---- Per-level state container ----
template <typename T> struct MgLevelState {
  void *x, *rhs, *r, *r_tilde, *p, *v, *s, *t;
  int dof, X, Y, Z, Lt, vol;
  size_t vec_sz;
  bool owned;        // true = buffers allocated by us, false = external (level 0)
  bool is_fullsite;  // true = this level operates on FULL-site vectors (no parity)
  bool has_solution; // true = x holds a usable previous solution (warm start)
  T r0_ref = 0;      // dev84: ||r|| of the FIRST solve at this level — the
                     // absolute anchor for the relative tolerance (warm-start
                     // cycles would otherwise chase an ever-shrinking target)
  int max_iter;      // per-level max iterations
  T tol;             // per-level tolerance
  int num_restart;   // per-level restart interval for coarse correction
  void alloc(int _dof, int _X, int _Y, int _Z, int _Lt, cudaStream_t stream) {
    dof=_dof; X=_X; Y=_Y; Z=_Z; Lt=_Lt; vol=X*Y*Z*Lt; vec_sz=(size_t)dof*vol;
    is_fullsite=false;
    size_t nbytes = vec_sz*sizeof(LatticeComplex<T>);
    checkCudaErrors(cudaMallocAsync(&x,      nbytes, stream));
    checkCudaErrors(cudaMallocAsync(&rhs,    nbytes, stream));
    checkCudaErrors(cudaMallocAsync(&r,      nbytes, stream));
    checkCudaErrors(cudaMallocAsync(&r_tilde,nbytes, stream));
    checkCudaErrors(cudaMallocAsync(&p,      nbytes, stream));
    checkCudaErrors(cudaMallocAsync(&v,      nbytes, stream));
    checkCudaErrors(cudaMallocAsync(&s,      nbytes, stream));
    checkCudaErrors(cudaMallocAsync(&t,      nbytes, stream));
    owned=true;
    checkCudaErrors(cudaMemsetAsync(x,  0, nbytes, stream));
    checkCudaErrors(cudaMemsetAsync(rhs,0, nbytes, stream));
    has_solution=false;
  }
  void free_all(cudaStream_t stream) {
    if(!owned) return;
    auto F=[&](void*&p){if(p){cudaFreeAsync(p,stream);p=nullptr;}};
    F(x);F(rhs);F(r);F(r_tilde);F(p);F(v);F(s);F(t); owned=false;
  }
};

// ====================================================================
// Main solver class
// ====================================================================
template <typename T> struct LatticeCloverMultigrid {
  LatticeSet<T> *set_ptr;
  LatticeWilsonDslash<T> wilson_dslash;
  LatticeCloverDslash<T> clover_dslash_ee, clover_dslash_oo;
  LatticeCloverDslash<T> clover_dslash_ee_inv, clover_dslash_oo_inv;

  void *gauge, *clover_ee, *clover_oo, *clover_ee_inv, *clover_oo_inv;
  void *fermion_out_eo, *fermion_in_eo;
  void *b_e, *b_o, *x_o;
  void *b__o, *r0, *rt0, *p0, *v0, *s0, *t0;
  LatticeSap<T> sap;

  // ---- Multigrid hierarchy ----
  int num_levels, mg_grid_size[4];
  MgLevelState<T> *levels;
  // SCHUR-consistent coarse operators (33-tensor stencil):
  //   null_vecs[fl]  — odd-lattice null vectors [E_{l+1}, 12, X_l, Y_l, Z_l, T_l/2]
  //   hop_nn[fl]     — nearest-neighbour hopping [2, 4, E, E, X_c, Y_c, Z_c, T_c]
  //   hop_diag[fl]   — diagonal hopping [2, 2, 6, E, E, X_c, Y_c, Z_c, T_c]
  //   sit_packed[fl] — on-site block [E, E, X_c, Y_c, Z_c, T_c]
  void **null_vecs, **hop_nn, **hop_diag, **sit_packed;

  // ---- Solver parameters ----
  int max_iter;
  T atol;
  int num_restart, rank;
  bool verbose;
  T kappa_val;

  // ---- Full-site residual buffer for V-cycle ----
  // Parity-split buffers have size lat_4dim_SC = _LAT_SC_ * X * Y * Z * (Lt/2).
  // Full-site buffers have size 2*lat_4dim_SC = _LAT_SC_ * X * Y * Z * Lt_full.
  void *r_full;           // full-site residual [sc, X, Y, Z, Lt_full]
  int Lt_full;            // un-halved t-dimension (2 × levels[0].Lt)

  // ---- Dedicated buffer for the parity-split ODD correction ----
  // CRITICAL: the coarse-grid correction e_odd must be preserved across the
  // r_before / r_after computations because fine_dslash_op() uses
  // device_vec2 as internal scratch (give_copy_vals + clover_dslash_oo.give
  // overwrite it).  Storing e_odd in its own buffer avoids this aliasing bug.
  void *e_odd_buf;        // parity-split odd correction [sc, X, Y, Z, Lt/2]

  // ---- Full-site level-0 solve buffers ----
  // The level-0 solve is performed on the FULL (non-preconditioned) operator,
  // matching pyqcu/solver/_multigrid.py with support_parity=False, which is
  // what gives the real multigrid speed-up (the parity-preconditioned Schur
  // complement's low modes are not captured by the coarse space).
  // All full-site vectors have size sc*X*Y*Z*T = 2*lat_4dim_SC.
  void *full_x, *full_rhs, *full_r, *full_rt, *full_p, *full_v, *full_s, *full_t;
  void *parity_tmp;       // [2, sc, X, Y, Z, T/2] split scratch (both channels)
  void *parity_dst;       // [2, sc, X, Y, Z, T/2] dslash result scratch
  void *corr_scratch;     // full-site scratch for the guarded V-cycle correction
                          // (holds D·e_fine so the correction can be tested and
                          //  reverted without disturbing the solution)
  void *coarse_partials;  // [1024] LatticeComplex scratch for cooperative
                          // fused coarse-solve cross-block dot reductions

  // ---- Host mirror for convergence check ----
  LatticeComplex<T> host_vals[_vals_size_];
  std::vector<T> conv_history;
  double level_times[8];
  double solve_time_ms;

  // ---- Multi-rank (multi-GPU) support ----
  // The backend runs the redundant-global model: every rank holds the FULL
  // (parity-halved) lattice, keeps it consistent via halo exchange, and
  // synchronises scalars with MPI_Allreduce.  Coarse dslash needs no inter-rank
  // data (periodic wrap on the full coarse grid is exact), but coarse dots must
  // be Allreduced so every rank sees the global value.  Single-rank runs never
  // touch the MPI paths.
  bool mg_multi = false;          // true when the 4D process grid != 1x1x1x1

  // ---- Section timing (2026-08-02) ----
  double prof_fine_iter_ms = 0;   // total fine-level iteration time
  double prof_vcycle_ms = 0;      // total V-cycle correction overhead
  double prof_coarse_solve_ms = 0;// total coarse-solve time inside v_cycle
  int    prof_n_vcycles = 0;
  // Coarse-iteration cost breakdown (accumulated for profiling)
  double prof_coarse_dslash_ms = 0;   // wide coarse dslash kernels
  double prof_coarse_dot_ms = 0;      // coarse dot products
  double prof_coarse_vec_ms = 0;      // coarse vector kernels

  // dev84: cached fine-site value of dv[_lat_4dim_] so v_cycle() need not do
  // a BLOCKING D2H memcpy on every entry (the patched value is always the
  // same level-0 volume within one solve).
  bool lat4_cache_valid = false;
  double lat4_cached = 0;
  // dev84 CUDA-GRAPH segment replay: on this WSL2 box a kernel launch costs
  // ~75 µs host-side and a stream sync flushes the WHOLE queued backlog
  // (nvprof: sync cost grew from 38 ms to 531 ms as queue depth grew).  We
  // capture SEG=8 coarse BiCGStab iterations into a graph once per level and
  // REPLAY it between host residual checks — launch overhead collapses ~100×
  // and each check sees a shallow queue.  Single-rank only (multi-rank dslash
  // contains blocking MPI, not capturable).
  static const int GRAPH_SEG = 8;
  bool graph_ready[8] = {};
  cudaGraph_t graph[8] = {};
  cudaGraphExec_t graph_exec[8] = {};

  void coarse_graph_ensure(int lev) {
    if (mg_multi || graph_ready[lev]) return;
    cudaStream_t S = set_ptr->stream;
    checkCudaErrors(cudaStreamBeginCapture(S, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < GRAPH_SEG; i++) bistabcg_iter_coarse(lev, false);
    cudaGraph_t g = nullptr;
    checkCudaErrors(cudaStreamEndCapture(S, &g));
    checkCudaErrors(cudaGraphInstantiate(&graph_exec[lev], g, nullptr, nullptr, 0));
    graph[lev] = g;
    graph_ready[lev] = true;
  }

  void coarse_graph_run(int lev, int n_iter) {
    if (mg_multi || !graph_ready[lev]) {
      for (int i = 0; i < n_iter; i++) bistabcg_iter_coarse(lev, false);
      return;
    }
    for (int r = 0; r < n_iter / GRAPH_SEG; r++)
      checkCudaErrors(cudaGraphLaunch(graph_exec[lev], set_ptr->stream));
    for (int r = 0; r < n_iter % GRAPH_SEG; r++) bistabcg_iter_coarse(lev, false);
  }

  int prof_n_coarse_iters = 0;   // dev84: total coarse BiStabCG iterations
  double prof_check_ms = 0;      // dev84: time inside block residual checks
  int prof_n_checks = 0;         // dev84: number of block residual checks
  double prof_ck_kernel_ms = 0;  // dev84: check breakdown — dot-kernel launch
  double prof_ck_d2h_ms = 0;     // dev84: check breakdown — memcpy enqueue
  double prof_ck_sync_ms = 0;    // dev84: check breakdown — stream sync
  long mg_cycle_counter = 0;     // dev84: V-cycle counter for safety-check cadence
  // dev84 zero-copy staging: dot results land directly in host-visible pinned
  // memory (cudaHostAllocMapped) — no cudaMemcpyAsync D2H anywhere in the
  // coarse path.  [0]=residual norm²  [1]=cold-solve ‖rhs‖²
  LatticeComplex<T> *check_host = nullptr;   // host-visible
  void *check_dev = nullptr;                 // device alias of the same pages


  void give(LatticeSet<T> *_s) {
    set_ptr=_s; wilson_dslash.give(_s);
    clover_dslash_ee.give(_s); clover_dslash_oo.give(_s);
    clover_dslash_ee_inv.give(_s); clover_dslash_oo_inv.give(_s);
    rank=set_ptr->host_params[_NODE_RANK_];
    verbose=(set_ptr->host_params[_VERBOSE_]!=0);
  }

  // ==================================================================
  // Dslash operators
  // ==================================================================

  /**
   * @brief Fine-level even-odd preconditioned Clover dslash.
   *
   * Computes out = D_oo*in - κ² * H_oe * D_ee^{-1} * H_eo * in
   * where H_eo, H_oe are the Wilson hopping terms (no kappa factor).
   * The κ² factor in bistabcg_give_dest_o is correct for the even-odd
   * preconditioned Clover-Wilson system.
   *
   * Both input and output are parity-split ODD-site vectors
   * (size = lat_4dim_SC = _LAT_SC_ * X * Y * Z * Lt_half).
   */
  void fine_dslash_op(void *out, void *in) {
    wilson_dslash.run_eo(set_ptr->device_vec0, in, gauge);
    give_copy_vals<T><<<set_ptr->gridDim,set_ptr->blockDim,0,set_ptr->stream>>>(
        set_ptr->device_vec2,set_ptr->device_vec0);
    clover_dslash_ee_inv.give(set_ptr->device_vec2);
    wilson_dslash.run_oe(set_ptr->device_vec1,set_ptr->device_vec2,gauge);
    give_copy_vals<T><<<set_ptr->gridDim,set_ptr->blockDim,0,set_ptr->stream>>>(
        set_ptr->device_vec2,in);
    clover_dslash_oo.give(set_ptr->device_vec2);
    bistabcg_give_dest_o<T><<<set_ptr->gridDim,set_ptr->blockDim,0,set_ptr->stream>>>(
        out,set_ptr->device_vec2,set_ptr->device_vec1,kappa_val,set_ptr->device_vals);
  }

  /**
   * @brief Coarse-grid Schur-consistent dslash (wide 33-tensor stencil).
   *
   * Applies A_c = P^T S P for the SCHUR-consistent coarse operator.  The Schur
   * operator S = D_oo - k^2 H_oe D_ee^{-1} H_eo couples odd sites x to x±2μ
   * (nearest in the coarse grid) AND x±μ±ν (diagonal), so the Galerkin coarse
   * operator needs the WIDE stencil (on-site + 8 nearest + 24 diagonal).
   * Operates on coarse-lattice vectors [E, Xc, Yc, Zc, Tc].
   */
  void coarse_dslash_op(void *out, void *in, int lev) {
    int E=levels[lev].dof, Xc=levels[lev].X, Yc=levels[lev].Y,
        Zc=levels[lev].Z, Ltc=levels[lev].Lt;
    int t=E*Xc*Yc*Zc*Ltc;
    dim3 g((t+_BLOCK_SIZE_-1)/_BLOCK_SIZE_);
    auto p0=std::chrono::high_resolution_clock::now();
    multigrid_coarse_dslash_wide<T><<<g,_BLOCK_SIZE_,0,set_ptr->stream>>>(
        out,in,sit_packed[lev-1],hop_nn[lev-1],hop_diag[lev-1],E,Xc,Yc,Zc,Ltc);
    auto p1=std::chrono::high_resolution_clock::now();
    prof_coarse_dslash_ms += std::chrono::duration<double,std::milli>(p1-p0).count();
  }

  /**
   * @brief FUSED single-kernel coarse solve for the COARSEST level.
   *
   * Replaces the per-iteration coarse BiStabCG loop with ONE kernel launch.
   * This is critical on this GPU: the coarse solve's many tiny kernels ran at
   * the idle clock (210 MHz, ~107 µs per kernel — the GPU never boosts for such
   * short bursts), so ~30 iterations × 13 launches cost ~0.5 s.  The fused
   * kernel does all BiStabCG iterations internally with block reductions and
   * costs ONE launch.
   */
  void coarse_solve_fused(int lev) {
    auto &st=levels[lev];
    int E=st.dof, Xc=st.X, Yc=st.Y, Zc=st.Z, Ltc=st.Lt;
    T tol_factor = levels[lev].tol;
    if (tol_factor <= 0) tol_factor = (T)1e-3;
    auto q0=std::chrono::high_resolution_clock::now();
    const int NT = 256;
    const int n = E * Xc * Yc * Zc * Ltc;
    int grid_sz = (n + NT - 1) / NT;
    // Cap the grid to what can be co-resident on this device (cooperative
    // launch requires the whole grid resident at once).
    // BUGFIX 2026-08-14 (P100/sm_60 多线程多卡): 必须按**当前线程绑定设备**
    // 查询 SM 数与 occupancy —— 硬编码 device 0（V100, 80 SM）会让 P100
    // 线程（56 SM）的 grid 超限；且 static 缓存须按设备分槽，否则多线程
    // 共用一份值互相污染。
    static int max_blocks_per_sm[16], sm_count[16];
    static bool occ_init[16] = {false};
    int cur_dev = 0;
    cudaGetDevice(&cur_dev);
    if (cur_dev < 0 || cur_dev >= 16) cur_dev = 0;
    if (!occ_init[cur_dev]) {
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &max_blocks_per_sm[cur_dev],
          (const void*)multigrid_coarse_solve_cg<T,NT>, NT, 0);
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, cur_dev);
      sm_count[cur_dev] = prop.multiProcessorCount;
      occ_init[cur_dev] = true;
    }
    int max_blocks = max_blocks_per_sm[cur_dev] * sm_count[cur_dev];
    if (grid_sz > max_blocks) grid_sz = max_blocks;
    if (rank==0 && verbose)
      printf("MG: fused grid=%d (cap=%d, sm=%d, occ=%d/block) n=%d\n",
             grid_sz, max_blocks, sm_count[cur_dev], max_blocks_per_sm[cur_dev], n);
    void *args[] = {
        (void*)&st.x, (void*)&st.rhs, (void*)&st.r_tilde, (void*)&st.r,
        (void*)&st.p, (void*)&st.v, (void*)&st.s, (void*)&st.t,
        (void*)&sit_packed[lev-1], (void*)&hop_nn[lev-1],
        (void*)&hop_diag[lev-1],
        (void*)&E, (void*)&Xc, (void*)&Yc, (void*)&Zc, (void*)&Ltc,
        (void*)&st.max_iter, (void*)&tol_factor, (void*)&coarse_partials};
    checkCudaErrors(cudaLaunchCooperativeKernel(
        (const void*)multigrid_coarse_solve_cg<T,NT>,
        dim3(grid_sz), dim3(NT), args, 0, set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    auto q1=std::chrono::high_resolution_clock::now();
    if (rank==0)
      log_write<T>("PYQCU::SOLVER::MULTIGRID::\n "+std::to_string(lev)+
        ": FUSED-PARALLEL coarse solve ("+std::to_string(E)+" dof, "+
        std::to_string(Xc)+"x"+std::to_string(Yc)+"x"+std::to_string(Zc)+
        "x"+std::to_string(Ltc)+", grid="+std::to_string(grid_sz)+") in "+
        std::to_string(std::chrono::duration<double,std::milli>(q1-q0).count())+" ms",rank,true);
  }

  // ==================================================================
  // FULL-site fine-level dslash (used for the level-0 solve).
  //
  // The level-0 solve matches pyqcu/solver/_multigrid.py with
  // support_parity=False: we solve the FULL Clover-Wilson operator
  //   D = [D_ee, -κ·H_eo]
  //       [-κ·H_oe, D_oo]
  // on full-site vectors [sc, X, Y, Z, T], NOT the even-odd preconditioned
  // Schur complement.  This is what makes the V-cycle coarse correction
  // effective (the coarse space captures the low modes of the full operator).
  // ==================================================================

  /**
   * @brief Convert a full-site vector to parity-split [2, sc, X, Y, Z, T/2].
   * dst must be a scratch buffer of size 2*lat_4dim_SC.
   */
  void full_to_parity(void *dst, void *src) {
    cudaStream_t S=set_ptr->stream;
    int sc=_LAT_SC_, total=(int)levels[0].vol;
    int X=levels[0].X, Y=levels[0].Y, Z=levels[0].Z;
    int Lt_full=2*levels[0].Lt;
    dim3 g((sc*total+_BLOCK_SIZE_-1)/_BLOCK_SIZE_);
    multigrid_full_to_even<T><<<g,_BLOCK_SIZE_,0,S>>>(dst, src, sc, X, Y, Z, Lt_full);
    // NOTE: cast to LatticeComplex<T>* so the offset is in COMPLEX elements.
    // Using (T*)src + N (T=float) would advance N FLOATS = half the channel.
    multigrid_full_to_odd<T><<<g,_BLOCK_SIZE_,0,S>>>(
        (LatticeComplex<T>*)dst+(size_t)(_LAT_SC_*levels[0].vol), src, sc, X, Y, Z, Lt_full);
  }

  /**
   * @brief Convert parity-split [2, sc, X, Y, Z, T/2] to a full-site vector.
   * dst has size 2*lat_4dim_SC.
   */
  void parity_to_full(void *dst, void *src) {
    cudaStream_t S=set_ptr->stream;
    size_t nbytes=2*(size_t)(_LAT_SC_*levels[0].vol)*sizeof(LatticeComplex<T>);
    checkCudaErrors(cudaMemsetAsync(dst, 0, nbytes, S));
    int sc=_LAT_SC_, total=(int)levels[0].vol;
    int X=levels[0].X, Y=levels[0].Y, Z=levels[0].Z;
    int Lt_full=2*levels[0].Lt;
    dim3 g((sc*total+_BLOCK_SIZE_-1)/_BLOCK_SIZE_);
    multigrid_even_to_full<T><<<g,_BLOCK_SIZE_,0,S>>>(dst, src, sc, X, Y, Z, Lt_full);
    // NOTE: cast to LatticeComplex<T>* so the offset is in COMPLEX elements.
    multigrid_odd_to_full<T><<<g,_BLOCK_SIZE_,0,S>>>(
        dst,(LatticeComplex<T>*)src+(size_t)(_LAT_SC_*levels[0].vol), sc, X, Y, Z, Lt_full);
  }

  /**
   * @brief Full-site Clover-Wilson dslash: out = D·in.
   *
   * Decomposes in into even/odd parity, applies the block operator
   *   (D·in)_e = D_ee·in_e - κ·H_eo·in_o
   *   (D·in)_o = D_oo·in_o - κ·H_oe·in_e
   * using the C++ parity-split dslash components, then recombines.
   * Both in/out are full-site vectors [sc, X, Y, Z, T] (2*lat_4dim_SC).
   */
  void fine_full_dslash_op(void *out, void *in) {
    cudaStream_t S=set_ptr->stream;
    dim3 gv=set_ptr->gridDim, bv=set_ptr->blockDim;
    size_t ch=(size_t)(_LAT_SC_*levels[0].vol);   // one parity channel size (complex elems)

    // Split input into parity channels: even at parity_tmp[0], odd at +ch.
    full_to_parity(parity_tmp, in);
    checkCudaErrors(cudaStreamSynchronize(S));
    void *xe=parity_tmp;
    void *xo=(LatticeComplex<T>*)parity_tmp+ch;

    // (D·in)_e = D_ee·x_e - κ·H_eo·x_o
    give_copy_vals<T><<<gv,bv,0,S>>>(set_ptr->device_vec0, xe);
    clover_dslash_ee.give(set_ptr->device_vec0);              // vec0 = D_ee·x_e
    wilson_dslash.run_eo(set_ptr->device_vec1, xo, gauge);    // vec1 = H_eo·x_o
    checkCudaErrors(cudaStreamSynchronize(S));
    LatticeComplex<T> neg_kap(-kappa_val,0);
    CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH,(int)ch,&neg_kap,
        set_ptr->device_vec1,1,set_ptr->device_vec0,1));      // vec0 -= κ·vec1
    checkCudaErrors(cudaStreamSynchronize(S));
    give_copy_vals<T><<<gv,bv,0,S>>>(parity_dst, set_ptr->device_vec0);

    // (D·in)_o = D_oo·x_o - κ·H_oe·x_e
    give_copy_vals<T><<<gv,bv,0,S>>>(set_ptr->device_vec2, xo);
    clover_dslash_oo.give(set_ptr->device_vec2);              // vec2 = D_oo·x_o
    wilson_dslash.run_oe(set_ptr->device_vec0, xe, gauge);    // vec0 = H_oe·x_e
    checkCudaErrors(cudaStreamSynchronize(S));
    CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH,(int)ch,&neg_kap,
        set_ptr->device_vec0,1,set_ptr->device_vec2,1));      // vec2 -= κ·vec0
    checkCudaErrors(cudaStreamSynchronize(S));
    give_copy_vals<T><<<gv,bv,0,S>>>((LatticeComplex<T>*)parity_dst+ch, set_ptr->device_vec2);

    // Combine parity → full-site output
    parity_to_full(out, parity_dst);
    checkCudaErrors(cudaStreamSynchronize(S));
  }

  /**
   * @brief Restrict fine full-site → coarse full-site.
   * Both vectors are full-site (non-parity-split).
   */
  void restrict_op(void *co, void *fi, int fl) {
    int l=fl+1, E=levels[l].dof, e=levels[fl].dof;
    int Xf=levels[fl].X,Yf=levels[fl].Y,Zf=levels[fl].Z,Ltf=levels[fl].Lt;
    int Xc=levels[l].X,Yc=levels[l].Y,Zc=levels[l].Z,Ltc=levels[l].Lt;
    int t=E*Xc*Yc*Zc*Ltc; dim3 g((t+_BLOCK_SIZE_-1)/_BLOCK_SIZE_);
    multigrid_restrict<T><<<g,_BLOCK_SIZE_,0,set_ptr->stream>>>(
        co,fi,null_vecs[fl],E,e,Xf,Yf,Zf,Ltf,Xc,Yc,Zc,Ltc);
  }

  /**
   * @brief Prolong coarse full-site → fine full-site.
   */
  void prolong_op(void *fo, void *ci, int fl) {
    int l=fl+1, E=levels[l].dof, e=levels[fl].dof;
    int Xf=levels[fl].X,Yf=levels[fl].Y,Zf=levels[fl].Z,Ltf=levels[fl].Lt;
    int Xc=levels[l].X,Yc=levels[l].Y,Zc=levels[l].Z,Ltc=levels[l].Lt;
    int t=e*Xf*Yf*Zf*Ltf; dim3 g((t+_BLOCK_SIZE_-1)/_BLOCK_SIZE_);
    multigrid_prolong<T><<<g,_BLOCK_SIZE_,0,set_ptr->stream>>>(
        fo,ci,null_vecs[fl],E,e,Xf,Yf,Zf,Ltf,Xc,Yc,Zc,Ltc);
  }

  // ==================================================================
  // Dot products — matching LatticeCloverBistabCg::_dot_mpi EXACTLY
  // ==================================================================

  /**
   * @brief MPI-parallel dot product for fine-level (level 0) vectors.
   *
   * Computes dot = sum_i conj(a[i]) * b[i] across all MPI ranks.
   * Writes cublasDot result to device_vals[_send_tmp_], copies D→H,
   * runs MPI_Allreduce, then copies H→D to device_vals[vals_idx].
   *
   * CRITICAL: The result is in BOTH host_vals[vals_idx] AND
   * device_vals[vals_idx] after this call returns.
   */
  void dot_mpi(void *a, void *b, int vals_idx, int si) {
    LatticeComplex<T> *dv=static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
    CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasHs[si],set_ptr->lat_4dim_SC,
        a,1,b,1,&dv[_send_tmp_]));
    checkCudaErrors(cudaMemcpyAsync(&host_vals[_send_tmp_],&dv[_send_tmp_],
        sizeof(LatticeComplex<T>),cudaMemcpyDeviceToHost,set_ptr->streams[si]));
    MPI_Barrier(MPI_COMM_WORLD);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[si]));
    _MPI_Allreduce<T>(&host_vals[_send_tmp_],&host_vals[vals_idx],_REAL_IMAG_,
        MPI_SUM,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    checkCudaErrors(cudaMemcpyAsync(&dv[vals_idx],&host_vals[vals_idx],
        sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,set_ptr->streams[si]));
  }

  /**
   * @brief MPI-parallel dot product for FULL-site level-0 vectors.
   * Same protocol as dot_mpi but sums over 2*lat_4dim_SC elements
   * (the full-site level-0 vectors have size [sc, X, Y, Z, T]).
   */
  void dot_full_mpi(void *a, void *b, int vals_idx, int si) {
    LatticeComplex<T> *dv=static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
    int n = 2 * set_ptr->lat_4dim_SC;
    CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasHs[si],n,
        a,1,b,1,&dv[_send_tmp_]));
    checkCudaErrors(cudaMemcpyAsync(&host_vals[_send_tmp_],&dv[_send_tmp_],
        sizeof(LatticeComplex<T>),cudaMemcpyDeviceToHost,set_ptr->streams[si]));
    MPI_Barrier(MPI_COMM_WORLD);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[si]));
    _MPI_Allreduce<T>(&host_vals[_send_tmp_],&host_vals[vals_idx],_REAL_IMAG_,
        MPI_SUM,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    checkCudaErrors(cudaMemcpyAsync(&dv[vals_idx],&host_vals[vals_idx],
        sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,set_ptr->streams[si]));
  }

  /**
   * @brief Dot product for coarse-level vectors (single GPU, no MPI).
   *
   * Uses the single-block reduction kernel coarse_dot_kernel (see above)
   * instead of cublasDot — profiling showed the cublas launch + memcpy
   * overhead dominated coarse solves.  Writes device_vals[vals_idx] on-device
   * and mirrors to host_vals only for the convergence check.
   *
   * Multi-rank: the local partial sum is reduced across ranks with
   * MPI_Allreduce (real/imag separately) so every rank sees the global dot.
   */
  void dot_coarse(void *a, void *b, int lv, int vals_idx, int si) {
    auto d0=std::chrono::high_resolution_clock::now();
    LatticeComplex<T> *dv=static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
    // 2026-08-15 dev76: large coarse vectors (>= 64K elements) use the
    // multi-block grid-stride reduction (single-block serialized ~768 adds
    // per thread at 196608 elements dominated the coarse solve).
    int n = (int)levels[lv].vec_sz;
    if (n >= 65536) {
      int nblk = (n + 255) / 256; if (nblk > 256) nblk = 256;
      coarse_dot_kernel_multi<T, 256><<<nblk, 256, 0, set_ptr->streams[si]>>>(
          static_cast<const LatticeComplex<T>*>(a),
          static_cast<const LatticeComplex<T>*>(b), n,
          static_cast<LatticeComplex<T>*>(coarse_partials));
      coarse_dot_reduce_kernel<T, 256><<<1, 256, 0, set_ptr->streams[si]>>>(
          static_cast<const LatticeComplex<T>*>(coarse_partials), nblk, &dv[vals_idx]);
    } else {
      coarse_dot_kernel<T, 256><<<1, 256, 0, set_ptr->streams[si]>>>(
          static_cast<const LatticeComplex<T>*>(a), static_cast<const LatticeComplex<T>*>(b),
          n, &dv[vals_idx]);
    }
    // Mirror to host ONLY for the convergence check (host_vals[_norm2_tmp_]).
    checkCudaErrors(cudaMemcpyAsync(&host_vals[vals_idx], &dv[vals_idx],
        sizeof(LatticeComplex<T>), cudaMemcpyDeviceToHost, set_ptr->streams[si]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[si]));
    if (mg_multi) {
      T gx = host_vals[vals_idx].real(), gy = host_vals[vals_idx].imag();
      MPI_Allreduce(MPI_IN_PLACE, &gx, 1, mpi_real_type<T>(), MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &gy, 1, mpi_real_type<T>(), MPI_SUM,
                    MPI_COMM_WORLD);
      host_vals[vals_idx] = LatticeComplex<T>(gx, gy);
      // Mirror the reduced value back to device_vals — the BiStabCG kernels
      // read the scalars from device memory.
      checkCudaErrors(cudaMemcpyAsync(&dv[vals_idx], &host_vals[vals_idx],
          sizeof(LatticeComplex<T>), cudaMemcpyHostToDevice,
          set_ptr->streams[si]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[si]));
    }
    auto d1=std::chrono::high_resolution_clock::now();
    prof_coarse_dot_ms += std::chrono::duration<double,std::milli>(d1-d0).count();
  }

  // ---- Vector helpers ----
  void zero_c(void *v,int l) {
    checkCudaErrors(cudaMemsetAsync(v,0,levels[l].vec_sz*sizeof(LatticeComplex<T>),set_ptr->stream));
  }
  // Helper: grid dimensions for site-processing kernels (each thread = 1 site × all DOF).
  // The BiStabCG kernels (give_copy_vals, bistabcg_give_*) process _LAT_SC_ elements
  // per thread, so the number of threads must be (effective vec_sz)/_LAT_SC_
  // (NOT levels[lev].vol, which is the site count for 12-DOF vectors only).  This makes
  // the infrastructure correct for coarse levels with E != 12 (e.g. dof_list=[12,24,...])
  // and for the full-site level 0 (whose vectors are 2× the per-channel vec_sz).
  dim3 site_grid(int lev) {
    size_t eff = levels[lev].is_fullsite ? 2*levels[lev].vec_sz : levels[lev].vec_sz;
    int t=(int)(eff / _LAT_SC_);
    return dim3((t+_BLOCK_SIZE_-1)/_BLOCK_SIZE_);
  }
  void copy_c(void *d,void *s,int l) {
    give_copy_vals<T><<<site_grid(l),_BLOCK_SIZE_,0,set_ptr->stream>>>(d,s);
  }

  // ==================================================================
  // dev84: coarse-level dot into a device_vals slot — NO host sync.
  // Shared by bistabcg_iter_coarse() and coarse_resid_norm().  Multi-block
  // path for n ≥ 65536 (dev76), single-block otherwise.  In the redundant-
  // global multi-rank model all ranks compute bitwise-identical partials
  // (replicated data, deterministic kernels), so no MPI Allreduce is needed
  // here — matching the existing iteration semantics.
  // ==================================================================
  void coarse_dot_slot(int lev, const LatticeComplex<T>* a,
                       const LatticeComplex<T>* b, int slot) {
    coarse_dot_dest(lev, a, b,
        &static_cast<LatticeComplex<T>*>(set_ptr->device_vals)[slot]);
  }

  // dev84: same dot with an EXPLICIT destination — used to write results into
  // zero-copy mapped host memory (bypasses cudaMemcpyAsync D2H entirely; on
  // this WSL2 box each D2H memcpyAsync enqueue costs ~40 ms!).
  void coarse_dot_dest(int lev, const LatticeComplex<T>* a,
                       const LatticeComplex<T>* b, LatticeComplex<T>* dest) {
    cudaStream_t S = set_ptr->stream;
    const int n = (int)levels[lev].vec_sz;
    // dev84: SINGLE-BLOCK dot for everything below ~1M elements.  On this box
    // each kernel execution costs ~300 µs GPU-side regardless of work, so the
    // 2-kernel multi-block path (partials + reduce) always loses to one
    // 1-block grid-stride kernel (~125 µs of memory traffic at n=294912).
    if (n >= 1048576) {
      int nblk = (n + 255) / 256; if (nblk > 256) nblk = 256;
      coarse_dot_kernel_multi<T, 256><<<nblk, 256, 0, S>>>(
          a, b, n, static_cast<LatticeComplex<T>*>(coarse_partials));
      coarse_dot_reduce_kernel<T, 256><<<1, 256, 0, S>>>(
          static_cast<const LatticeComplex<T>*>(coarse_partials), nblk, dest);
    } else {
      coarse_dot_kernel<T, 256><<<1, 256, 0, S>>>(a, b, n, dest);
    }
  }

  // Host-visible ||r|| at a coarse level: ONE dot + ONE D2H + ONE sync.
  // Used by the block-checked coarse solve loop (every check_every iters).
  T coarse_resid_norm(int lev) {
    auto &st = levels[lev];
    cudaStream_t S = set_ptr->stream;
    auto k0=std::chrono::high_resolution_clock::now();
    coarse_dot_dest(lev, static_cast<const LatticeComplex<T>*>(st.r),
                    static_cast<const LatticeComplex<T>*>(st.r),
                    static_cast<LatticeComplex<T>*>(check_dev));
    auto k1=std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaStreamSynchronize(S));
    auto k2=std::chrono::high_resolution_clock::now();
    prof_ck_kernel_ms += std::chrono::duration<double,std::milli>(k1-k0).count();
    prof_ck_sync_ms += std::chrono::duration<double,std::milli>(k2-k1).count();
    T g = check_host[0].real();
    return sqrt(g < (T)0 ? (T)0 : g);
  }


  // ==================================================================
  // BiStabCG iteration — EXACT sync pattern of BistabCg::_run()
  //
  // Convergence residual (||r||²) is written to host_vals[_norm2_tmp_]
  // during step 3.5 (before dslash). The caller checks it after the
  // iteration to decide convergence. This is intentionally lagged by
  // one iteration, matching the reference BiStabCG behavior.
  //
  // FIX: Full stream sync at end of iteration to ensure all device_vals
  // updates are visible before the next iteration reads them.
  // ==================================================================
  /**
   * @brief Single-stream, overhead-light BiStabCG iteration for COARSE levels.
   *
   * Coarse vectors are tiny (a few thousand complex elements), so the 5-stream
   * parallel dot architecture used at the fine level is pure overhead: it
   * performs ~4 cublasDot + 4 memcpy + ~10 stream syncs per iteration, which
   * dominated the coarse solve (profiling: dots = 416 ms, vector kernels =
   * ~445 ms of a 2 s solve).  This version:
   *   - uses a single stream (serialised kernels — tiny anyway),
   *   - computes every dot with the single-block reduction kernel writing
   *     directly into device_vals (no cublas, no per-dot memcpy),
   *   - mirrors the convergence norm to host only ONCE per iteration.
   * Semantics match the standard BiStabCG exactly (r_tilde, p/v/s/t, scalars).
   *
   * dev84: with_norm=false skips step 10 entirely (no ||r||² dot, no D2H, no
   * sync) — the caller then checks the residual every `check_every` iterations
   * via coarse_resid_norm() instead of every iteration.  On 16x32x32x48 the
   * coarse solve is ~100 iterations × ~1.3 ms of pure host-sync latency; block
   * checking removes ≥7/8 of that sync cost while keeping the breakdown guard.
   */
  void bistabcg_iter_coarse(int lev, bool with_norm = true) {
    auto &st=levels[lev]; cudaStream_t S=set_ptr->stream;
    LatticeComplex<T> *dv=static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
    dim3 gv=site_grid(lev), bv=dim3(_BLOCK_SIZE_);
    const LatticeComplex<T>* rt = static_cast<const LatticeComplex<T>*>(st.r_tilde);
    const LatticeComplex<T>* r  = static_cast<const LatticeComplex<T>*>(st.r);
    auto tc0=std::chrono::high_resolution_clock::now();
    // 2026-08-15 dev76: multi-block dot for large coarse vectors
    // (single-block serial reduction dominated the coarse solve on
    //  16x16x16x32 lv1 = 196608 elements).
    // dev84: shared by coarse_resid_norm() via coarse_dot_slot().
    auto cdot = [&](const LatticeComplex<T>* a, const LatticeComplex<T>* b,
                    int slot) { coarse_dot_slot(lev, a, b, slot); };

    // 1. rho = <r_tilde, r>
    cdot(rt, r, _rho_);
    // 2. beta = (rho/rho_prev)*(alpha/omega);  rho_prev = rho
    //    dev84: GUARDED variants — post-convergence ρ→0 breakdown clamped to
    //    β=0 instead of 0/0 NaN (fixed-step sync-free loop never checks host).
    //    dev84 kernel-count diet: β+ρ_prev fused into one launch.
    mg_give_1beta_rp<T><<<1,1,0,S>>>(dv);
    // 3. p = r + beta*(p - omega*v)
    bistabcg_give_p<T><<<gv,bv,0,S>>>(st.p, st.r, st.v, dv);
    // 4. v = A_c·p
    coarse_dslash_op(st.v, st.p, lev);
    // 5. alpha = rho / <r_tilde, v>
    cdot(rt, static_cast<const LatticeComplex<T>*>(st.v), _tmp0_);
    mg_give_1alpha<T><<<1,1,0,S>>>(dv);
    // 6. s = r - alpha*v
    bistabcg_give_s<T><<<gv,bv,0,S>>>(st.s, st.r, st.v, dv);
    // 7. t = A_c·s
    coarse_dslash_op(st.t, st.s, lev);
    // 8. omega = <t,s>/<t,t>
    cdot(static_cast<const LatticeComplex<T>*>(st.t),
         static_cast<const LatticeComplex<T>*>(st.s), _tmp0_);
    cdot(static_cast<const LatticeComplex<T>*>(st.t),
         static_cast<const LatticeComplex<T>*>(st.t), _tmp1_);
    mg_give_1omega<T><<<1,1,0,S>>>(dv);
    // 9. r = s - omega*t;  x = x + alpha*p + omega*s
    //    dev84 kernel-count diet: both elementwise updates fused (one launch).
    {
      size_t eff = st.vec_sz;   // coarse levels: vec_sz/_LAT_SC_ site groups
      int t4 = (int)(eff / _LAT_SC_);
      mg_give_rx<T><<<(t4+_BLOCK_SIZE_-1)/_BLOCK_SIZE_,_BLOCK_SIZE_,0,S>>>(
          static_cast<LatticeComplex<T>*>(st.r),
          static_cast<const LatticeComplex<T>*>(st.s),
          static_cast<const LatticeComplex<T>*>(st.t),
          static_cast<LatticeComplex<T>*>(st.x),
          static_cast<const LatticeComplex<T>*>(st.p), dv, t4);
    }
    // 10. convergence norm ||r||² -> host (once per iteration, or never when
    //     with_norm=false — the caller block-checks via coarse_resid_norm())
    if (with_norm) {
      cdot(static_cast<const LatticeComplex<T>*>(st.r),
           static_cast<const LatticeComplex<T>*>(st.r), _norm2_tmp_);
      checkCudaErrors(cudaMemcpyAsync(&host_vals[_norm2_tmp_], &dv[_norm2_tmp_],
          sizeof(LatticeComplex<T>), cudaMemcpyDeviceToHost, S));
      checkCudaErrors(cudaStreamSynchronize(S));
    }
    // NOTE: the per-iteration cost here is dominated by the host sync above
    // (~170 us) plus the ~14 tiny kernel launches — NOT by the wide dslash
    // (prof_coarse_dslash_ms stays < 1 ms across the whole solve).
    auto tc1=std::chrono::high_resolution_clock::now();
    prof_coarse_vec_ms += std::chrono::duration<double,std::milli>(tc1-tc0).count();
  }

  void bistabcg_iter(int lev) {
    // Coarse levels use the single-stream overhead-light path.
    if (lev >= 1) { bistabcg_iter_coarse(lev); return; }
    // Single-rank fine level: use the sync-light single-stream path
    // (cublasDot directly into device_vals on the main stream, ONE sync per
    // iteration instead of ~20).  Multi-rank keeps the 5-stream MPI path.
    if (set_ptr->host_params[_GRID_X_] == 1 &&
        set_ptr->host_params[_GRID_Y_] == 1 &&
        set_ptr->host_params[_GRID_Z_] == 1 &&
        set_ptr->host_params[_GRID_T_] == 1 &&
        !_WILSON_AND_LAPLACIAN_TEST_SINGLE_IN_MULTI_) {
      bistabcg_iter_fine_fast();
      return;
    }
    auto &st=levels[lev];
    bool fine=(lev==0); cudaStream_t S=set_ptr->stream;
    // fullsite: level 0 solved on FULL-site vectors (support_parity=False style).
    // In this mode the vectors have size 2*lat_4dim_SC and _lat_4dim_ must be
    // patched (by the caller) to the full-site volume 2*st.vol.
    bool fullsite=(fine && st.is_fullsite);
    dim3 gv,bv;
    // Grid dimension: number of SITES (vol), not total elements (vec_sz = dof*vol).
    // Each thread processes one site × all DOF components.
    // Using vec_sz would launch vol*dof threads, causing OOB writes.
    if(fine && !fullsite){gv=set_ptr->gridDim;bv=set_ptr->blockDim;}
    else{
      // Number of threads = (effective vec_sz)/_LAT_SC_ so that the kernels
      // (which process _LAT_SC_ elements per thread) cover the ENTIRE vector.
      // The full-site level 0 has 2*st.vec_sz elements; coarse levels may have
      // E != 12 (e.g. dof_list=[12,24,...]) so their vec_sz/12 is used.
      size_t eff = fullsite ? 2*st.vec_sz : st.vec_sz;
      int t=(int)(eff / _LAT_SC_);
      gv=dim3((t+_BLOCK_SIZE_-1)/_BLOCK_SIZE_);bv=dim3(_BLOCK_SIZE_);
    }

    // Step 1: ρ = (r_tilde, r)           [stream _a_]
    auto tp0=std::chrono::high_resolution_clock::now();
    if(fine){
      if(fullsite) dot_full_mpi(st.r_tilde,st.r,_rho_,_a_);
      else         dot_mpi(st.r_tilde,st.r,_rho_,_a_);
    }
    else dot_coarse(st.r_tilde,st.r,lev,_rho_,_a_);
    auto tp1=std::chrono::high_resolution_clock::now();
    double dt_dot1=std::chrono::duration<double,std::micro>(tp1-tp0).count();

    // Step 2: β=(ρ/ρ_prev)*(α/ω)          [_a_];  ρ_prev←ρ [_b_]
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    bistabcg_give_1beta<T><<<1,1,0,set_ptr->streams[_a_]>>>(set_ptr->device_vals);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    bistabcg_give_1rho_prev<T><<<1,1,0,set_ptr->streams[_b_]>>>(set_ptr->device_vals);

    // Step 3: p = r + β·(p−ω·v)          [_a_]
    bistabcg_give_p<T><<<gv,bv,0,set_ptr->streams[_a_]>>>(st.p,st.r,st.v,set_ptr->device_vals);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));

    // Step 3.5: convergence check — ||r||² → host_vals[_norm2_tmp_]  [_c_]
    // Placed here (before dslash) matching the reference BiStabCG exactly.
    if(fine){ if(fullsite) dot_full_mpi(st.r,st.r,_norm2_tmp_,_c_);
              else         dot_mpi(st.r,st.r,_norm2_tmp_,_c_); }
    else     dot_coarse(st.r,st.r,lev,_norm2_tmp_,_c_);

    // Step 4: v = A·p                      [main stream]
    checkCudaErrors(cudaStreamSynchronize(S));
    if(fine){ if(fullsite) fine_full_dslash_op(st.v,st.p); else fine_dslash_op(st.v,st.p); }
    else     coarse_dslash_op(st.v,st.p,lev);
    checkCudaErrors(cudaStreamSynchronize(S));

    // Step 5: τ₀=(r_tilde,v); α=ρ/τ₀     [_d_]
    if(fine){ if(fullsite) dot_full_mpi(st.r_tilde,st.v,_tmp0_,_d_);
              else         dot_mpi(st.r_tilde,st.v,_tmp0_,_d_); }
    else     dot_coarse(st.r_tilde,st.v,lev,_tmp0_,_d_);
    bistabcg_give_1alpha<T><<<1,1,0,set_ptr->streams[_d_]>>>(set_ptr->device_vals);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

    // Step 6: s = r − α·v                  [_a_]
    bistabcg_give_s<T><<<gv,bv,0,set_ptr->streams[_a_]>>>(st.s,st.r,st.v,set_ptr->device_vals);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));

    // Step 7: t = A·s                      [main stream]
    checkCudaErrors(cudaStreamSynchronize(S));
    if(fine){ if(fullsite) fine_full_dslash_op(st.t,st.s); else fine_dslash_op(st.t,st.s); }
    else     coarse_dslash_op(st.t,st.s,lev);
    checkCudaErrors(cudaStreamSynchronize(S));

    // Step 8: τ₀=(t,s); τ₁=(t,t)          [_c_],[_d_]
    if(fine){ if(fullsite){dot_full_mpi(st.t,st.s,_tmp0_,_c_);dot_full_mpi(st.t,st.t,_tmp1_,_d_);}
              else        {dot_mpi(st.t,st.s,_tmp0_,_c_);dot_mpi(st.t,st.t,_tmp1_,_d_);} }
    else    {dot_coarse(st.t,st.s,lev,_tmp0_,_c_);dot_coarse(st.t,st.t,lev,_tmp1_,_d_);}
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));

    // Step 9: ω = τ₀/τ₁                   [_d_]
    bistabcg_give_1omega<T><<<1,1,0,set_ptr->streams[_d_]>>>(set_ptr->device_vals);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

    // Step 10: r=s−ω·t [_a_];  x=x+α·p+ω·s [_b_]
    bistabcg_give_r<T><<<gv,bv,0,set_ptr->streams[_a_]>>>(st.r,st.s,st.t,set_ptr->device_vals);
    bistabcg_give_x_o<T><<<gv,bv,0,set_ptr->streams[_b_]>>>(st.x,st.p,st.s,set_ptr->device_vals);

    // Full 5-stream sync at bottom of iteration — REQUIRED for deterministic
    // convergence.  Reducing to only _a_/_c_ kept a race (iteration count
    // 70, 70, 78 across runs vs a stable 63-68 with the full sync).
    checkCudaErrors(cudaStreamSynchronize(S));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
  }

  // ==================================================================
  // Single-rank, single-stream, sync-light BiStabCG iteration (level 0).
  //
  // WHY (2026-08-02): the reference 5-stream iteration (bistabcg_iter)
  // costs ~6 ms/iter on this WSL2/V100 box because every dot performs
  //   cublasDot -> D2H memcpy -> MPI_Barrier -> cudaStreamSynchronize
  //   (~170 us each) -> MPI_Allreduce -> H2D memcpy,
  // and there are ~20 such syncs per iteration.  For a 1x1x1x1 process
  // grid MPI_Allreduce is the identity, so we write each dot straight into
  // device_vals on the MAIN stream (cublasH is bound to set_ptr->stream).
  // In-stream ordering makes every consumer kernel see the result; we sync
  // ONCE at the end so host_vals[_norm2_tmp_] holds the convergence
  // residual.  Mathematically identical to bistabcg_iter, ~20x fewer syncs.
  //
  // This is only valid for single-rank runs (no MPI reduction needed).
  // ==================================================================
  void bistabcg_iter_fine_fast() {
    auto &st=levels[0]; cudaStream_t S=set_ptr->stream;
    LatticeComplex<T>* dv=static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
    dim3 gv=set_ptr->gridDim, bv=set_ptr->blockDim;
    const int n=(int)st.vec_sz;  // = lat_4dim_SC elements (odd-site vector)
    // The dslash outputs (st.v / st.t) are consumed by the NEXT kernel on the
    // same main stream, so run_mpi's final sync is redundant — save ~340 us/iter.
    wilson_dslash.skip_final_sync_ = true;
    // 1. rho = <r_tilde, r>                       (on-device, no host round-trip)
    CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasH, n, st.r_tilde,1,st.r,1,&dv[_rho_]));
    // 2. beta = (rho/rho_prev)*(alpha/omega); rho_prev = rho
    bistabcg_give_1beta<T><<<1,1,0,S>>>(dv);
    bistabcg_give_1rho_prev<T><<<1,1,0,S>>>(dv);
    // 3. p = r + beta*(p - omega*v)
    bistabcg_give_p<T><<<gv,bv,0,S>>>(st.p, st.r, st.v, dv);
    // 3.5 convergence residual ||r||^2 -> ZERO-COPY mapped page (the ONLY
    //     host read this iter).  dev84: cublasDot 写入映射宿主页别名,
    //     免去每次迭代的 D2H memcpyAsync (WSL2 每次 thunk ~0.6ms)。
    CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasH, n, st.r,1,st.r,1,
        static_cast<LatticeComplex<T>*>(check_dev) + 1));
    // 4. v = S·p
    fine_dslash_op(st.v, st.p);
    // 5. alpha = rho / <r_tilde, v>
    CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasH, n, st.r_tilde,1,st.v,1,&dv[_tmp0_]));
    bistabcg_give_1alpha<T><<<1,1,0,S>>>(dv);
    // 6. s = r - alpha*v
    bistabcg_give_s<T><<<gv,bv,0,S>>>(st.s, st.r, st.v, dv);
    // 7. t = S·s
    fine_dslash_op(st.t, st.s);
    // 8. omega = <t,s> / <t,t>
    CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasH, n, st.t,1,st.s,1,&dv[_tmp0_]));
    CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasH, n, st.t,1,st.t,1,&dv[_tmp1_]));
    bistabcg_give_1omega<T><<<1,1,0,S>>>(dv);
    // 9. r = s - omega*t ;  x = x + alpha*p + omega*s
    //    dev84 kernel-count diet: 双元素级更新融合为一次发射 (mg_give_rx)
    {
      int n4=(int)(n/_LAT_SC_);
      mg_give_rx<T><<<gv,bv,0,S>>>(
          static_cast<LatticeComplex<T>*>(st.r),
          static_cast<const LatticeComplex<T>*>(st.s),
          static_cast<const LatticeComplex<T>*>(st.t),
          static_cast<LatticeComplex<T>*>(st.x),
          static_cast<const LatticeComplex<T>*>(st.p), dv, n4);
    }
    // 10. single sync so check_host[1] (mapped) is valid for the caller
    checkCudaErrors(cudaStreamSynchronize(S));
  }

  // ==================================================================
  // Parity helper: b__o = b_o + κ · H_oe · D_ee^{-1} · b_e
  // (Schur complement RHS for the even-odd preconditioned system)
  // ==================================================================
  void setup_b__o() {
    give_copy_vals<T><<<set_ptr->gridDim,set_ptr->blockDim,0,set_ptr->stream>>>(
        set_ptr->device_vec2,b_e);
    clover_dslash_ee_inv.give(set_ptr->device_vec2);
    wilson_dslash.run_oe(set_ptr->device_vec0,set_ptr->device_vec2,gauge);
    bistabcg_give_b__o<T><<<set_ptr->gridDim,set_ptr->blockDim,0,set_ptr->stream>>>(
        b__o,b_o,set_ptr->device_vec0,kappa_val,set_ptr->device_vals);
  }

  // ==================================================================
  // Parity helper: x_e = D_ee^{-1} · (b_e + κ · H_eo · x_o)
  //
  // Reconstructs the even-site solution from the odd-site solution
  // using the even-odd Schur complement relationship.
  // Writes result to fermion_out_eo (the even part of the output buffer).
  // ==================================================================
  void recover_x_e() {
    CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH,set_ptr->lat_4dim_SC*_REAL_IMAG_,
        (T*)b_e,1,(T*)set_ptr->device_vec0,1));
    wilson_dslash.run_eo(set_ptr->device_vec1,x_o,gauge);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    LatticeComplex<T> kap(kappa_val,0.0);
    CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH,set_ptr->lat_4dim_SC,&kap,
        set_ptr->device_vec1,1,set_ptr->device_vec0,1));
    clover_dslash_ee_inv.give(set_ptr->device_vec0);
    CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH,set_ptr->lat_4dim_SC*_REAL_IMAG_,
        (T*)set_ptr->device_vec0,1,(T*)fermion_out_eo,1));
  }

  // ==================================================================
  // Full-site residual computation for V-cycle correction.
  //
  // Computes the FULL (non-preconditioned) residual r_full = b - D*x
  // where D is the full Clover-Wilson Dirac operator:
  //   D = [D_ee        , -κ·H_eo]
  //       [-κ·H_oe     ,  D_oo   ]
  //
  // Steps:
  //   1. Reconstruct x_e = D_ee^{-1} · (b_e + κ · H_eo · x_o)
  //      By construction, this makes r_e = 0 always.
  //   2. Compute r_o_full = b_o + κ · H_oe · x_e - D_oo · x_o
  //      This is the odd-site full residual (in parity-split layout).
  //   3. Convert r_o_full from parity-split odd to full-site odd t-slices,
  //      leaving even t-slices as zero.
  //
  // Output: r_full — full-site residual [sc, X, Y, Z, Lt_full]
  //         with r_even_sites = 0, r_odd_sites = r_o_full.
  // ==================================================================
  void compute_full_residual() {
    cudaStream_t S=set_ptr->stream;
    dim3 gf=set_ptr->gridDim, bf=set_ptr->blockDim;

    // Step 1: Reconstruct x_e into vec0
    //   vec0 = b_e + κ · H_eo · x_o → D_ee^{-1} → x_e
    checkCudaErrors(cudaStreamSynchronize(S));
    CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH,set_ptr->lat_4dim_SC*_REAL_IMAG_,
        (T*)b_e,1,(T*)set_ptr->device_vec0,1));
    wilson_dslash.run_eo(set_ptr->device_vec1,x_o,gauge);
    checkCudaErrors(cudaStreamSynchronize(S));
    LatticeComplex<T> kap(kappa_val,0.0);
    CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH,set_ptr->lat_4dim_SC,&kap,
        set_ptr->device_vec1,1,set_ptr->device_vec0,1));
    clover_dslash_ee_inv.give(set_ptr->device_vec0);
    checkCudaErrors(cudaStreamSynchronize(S));
    // device_vec0 now holds x_e (parity-split even layout)

    // Step 2: Compute r_o_full = b_o + κ · H_oe · x_e - D_oo · x_o
    //   vec1 = D_oo · x_o
    give_copy_vals<T><<<gf,bf,0,S>>>(set_ptr->device_vec1, x_o);
    clover_dslash_oo.give(set_ptr->device_vec1);   // vec1 = D_oo · x_o
    checkCudaErrors(cudaStreamSynchronize(S));

    //   vec2 = H_oe · x_e (result on odd sites)
    wilson_dslash.run_oe(set_ptr->device_vec2, set_ptr->device_vec0, gauge);
    checkCudaErrors(cudaStreamSynchronize(S));

    //   vec1 = D_oo·x_o - κ·H_oe·x_e = vec1 - κ · vec2
    LatticeComplex<T> neg_kap(-kappa_val,0.0);
    CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH,set_ptr->lat_4dim_SC,
        &neg_kap,set_ptr->device_vec2,1,set_ptr->device_vec1,1));
    checkCudaErrors(cudaStreamSynchronize(S));

    //   vec2 = b_o - vec1 = b_o - D_oo·x_o + κ·H_oe·x_e = r_o_full
    //   (vec2 now holds r_o_full in parity-split odd layout)
    bistabcg_give_diff2<T><<<gf,bf,0,S>>>(b_o,set_ptr->device_vec1,
        set_ptr->device_vec2,set_ptr->device_vals);
    checkCudaErrors(cudaStreamSynchronize(S));

    // Step 3: Convert r_o_full (parity-split odd) → full-site r_full
    //   Zero the full-site buffer first, then fill odd t-slices.
    size_t r_full_size = (size_t)_LAT_SC_ * set_ptr->lat_4dim * 2 * sizeof(LatticeComplex<T>);
    // Note: levels[0].Lt is HALVED (parity-split), so Lt_full = 2 * levels[0].Lt
    // And levels[0].vol = X * Y * Z * Lt_half, so full vol = 2 * levels[0].vol
    checkCudaErrors(cudaMemsetAsync(r_full, 0, r_full_size, S));
    int Lt_full_local = 2 * levels[0].Lt; // un-halved t-dimension
    int total_odd_sites = _LAT_SC_ * levels[0].X * levels[0].Y * levels[0].Z * levels[0].Lt;
    dim3 g_conv((total_odd_sites + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
    multigrid_odd_to_full<T><<<g_conv, _BLOCK_SIZE_, 0, S>>>(
        r_full, set_ptr->device_vec2,
        _LAT_SC_, levels[0].X, levels[0].Y, levels[0].Z, Lt_full_local);
    checkCudaErrors(cudaStreamSynchronize(S));
    // r_full now has the full-site residual: even sites = 0, odd sites = r_o_full
  }

  // ==================================================================
  // Extract odd-site correction from full-site prolonged vector.
  //
  // Takes a full-site prolonged correction and converts it to
  // parity-split odd layout for adding to x_o.
  //
  // Input:  full_in — full-site vector [sc, X, Y, Z, Lt_full]
  // Output: odd_out — parity-split odd vector [sc, X, Y, Z, Lt_half]
  // ==================================================================
  void extract_odd_from_full(void *odd_out, void *full_in) {
    cudaStream_t S=set_ptr->stream;
    int Lt_full_local = 2 * levels[0].Lt;
    int total_odd_sites = _LAT_SC_ * levels[0].X * levels[0].Y * levels[0].Z * levels[0].Lt;
    dim3 g_conv((total_odd_sites + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
    multigrid_full_to_odd<T><<<g_conv, _BLOCK_SIZE_, 0, S>>>(
        odd_out, full_in,
        _LAT_SC_, levels[0].X, levels[0].Y, levels[0].Z, Lt_full_local);
    checkCudaErrors(cudaStreamSynchronize(S));
  }

  // ==================================================================
  // BiStabCG state reset for level 0.
  //
  // After a V-cycle correction, x_o has changed, so all BiStabCG state
  // (p, v, s, t, rho_prev, alpha, omega) must be reinitialized.
  // Without this, the next BiStabCG iteration would use stale values
  // from before the correction, causing convergence degradation.
  // ==================================================================
  void reset_bistabcg_state_l0() {
    cudaStream_t S=set_ptr->stream;
    dim3 gf=set_ptr->gridDim, bf=set_ptr->blockDim;
    LatticeComplex<T> *dv=static_cast<LatticeComplex<T>*>(set_ptr->device_vals);

    // Zero search direction vectors
    checkCudaErrors(cudaMemsetAsync(p0, 0, set_ptr->lat_4dim_SC*sizeof(LatticeComplex<T>), S));
    checkCudaErrors(cudaMemsetAsync(v0, 0, set_ptr->lat_4dim_SC*sizeof(LatticeComplex<T>), S));
    checkCudaErrors(cudaMemsetAsync(s0, 0, set_ptr->lat_4dim_SC*sizeof(LatticeComplex<T>), S));
    checkCudaErrors(cudaMemsetAsync(t0, 0, set_ptr->lat_4dim_SC*sizeof(LatticeComplex<T>), S));

    // Reset scalar state for first iteration
    LatticeComplex<T> one(1,0), z(0,0);
    checkCudaErrors(cudaMemcpyAsync(&dv[_rho_],     &z, sizeof(LatticeComplex<T>), cudaMemcpyHostToDevice, S));
    checkCudaErrors(cudaMemcpyAsync(&dv[_rho_prev_],&one,sizeof(LatticeComplex<T>), cudaMemcpyHostToDevice, S));
    checkCudaErrors(cudaMemcpyAsync(&dv[_alpha_],   &one,sizeof(LatticeComplex<T>), cudaMemcpyHostToDevice, S));
    checkCudaErrors(cudaMemcpyAsync(&dv[_omega_],   &one,sizeof(LatticeComplex<T>), cudaMemcpyHostToDevice, S));
    checkCudaErrors(cudaStreamSynchronize(S));
  }

  // ==================================================================
  // BiStabCG state reset for the FULL-site level-0 solve.
  // (p/v/s/t are the full-site buffers; scalars live in device_vals.)
  // ==================================================================
  void reset_bistabcg_state_full() {
    cudaStream_t S=set_ptr->stream;
    int full_vol = 2 * levels[0].vol;
    size_t full_n = (size_t)_LAT_SC_ * full_vol;
    size_t nb = full_n * sizeof(LatticeComplex<T>);
    LatticeComplex<T> *dv=static_cast<LatticeComplex<T>*>(set_ptr->device_vals);

    // Zero the full-site search direction vectors
    checkCudaErrors(cudaMemsetAsync(full_p, 0, nb, S));
    checkCudaErrors(cudaMemsetAsync(full_v, 0, nb, S));
    checkCudaErrors(cudaMemsetAsync(full_s, 0, nb, S));
    checkCudaErrors(cudaMemsetAsync(full_t, 0, nb, S));

    // Reset scalar state for first iteration
    LatticeComplex<T> one(1,0), z(0,0);
    checkCudaErrors(cudaMemcpyAsync(&dv[_rho_],     &z, sizeof(LatticeComplex<T>), cudaMemcpyHostToDevice, S));
    checkCudaErrors(cudaMemcpyAsync(&dv[_rho_prev_],&one,sizeof(LatticeComplex<T>), cudaMemcpyHostToDevice, S));
    checkCudaErrors(cudaMemcpyAsync(&dv[_alpha_],   &one,sizeof(LatticeComplex<T>), cudaMemcpyHostToDevice, S));
    checkCudaErrors(cudaMemcpyAsync(&dv[_omega_],   &one,sizeof(LatticeComplex<T>), cudaMemcpyHostToDevice, S));
    checkCudaErrors(cudaStreamSynchronize(S));
  }

  // ==================================================================
  // BiStabCG state reset for coarse levels.
  // ==================================================================
  void reset_bistabcg_state(int lev) {
    auto &st=levels[lev];
    cudaStream_t S=set_ptr->stream;
    LatticeComplex<T> *dv=static_cast<LatticeComplex<T>*>(set_ptr->device_vals);

    zero_c(st.p, lev); zero_c(st.v, lev);
    zero_c(st.s, lev); zero_c(st.t, lev);

    LatticeComplex<T> one(1,0), z(0,0);
    checkCudaErrors(cudaMemcpyAsync(&dv[_rho_],     &z, sizeof(LatticeComplex<T>), cudaMemcpyHostToDevice, S));
    checkCudaErrors(cudaMemcpyAsync(&dv[_rho_prev_],&one,sizeof(LatticeComplex<T>), cudaMemcpyHostToDevice, S));
    checkCudaErrors(cudaMemcpyAsync(&dv[_alpha_],   &one,sizeof(LatticeComplex<T>), cudaMemcpyHostToDevice, S));
    checkCudaErrors(cudaMemcpyAsync(&dv[_omega_],   &one,sizeof(LatticeComplex<T>), cudaMemcpyHostToDevice, S));
    checkCudaErrors(cudaStreamSynchronize(S));
  }

  // ==================================================================
  // Params parsing — reads MG configuration from host_params.
  // ==================================================================
  void parse_params() {
    num_levels=set_ptr->host_params[_MG_NUM_LEVEL_];
    if(num_levels<1)num_levels=1; if(num_levels>8)num_levels=8;
    levels=new MgLevelState<T>[num_levels];

    // Level 0: parity-split layout (Lt is halved by LatticeSet::give)
    levels[0].dof=_LAT_SC_;
    levels[0].X=set_ptr->host_params[_LAT_X_];
    levels[0].Y=set_ptr->host_params[_LAT_Y_];
    levels[0].Z=set_ptr->host_params[_LAT_Z_];
    levels[0].Lt=set_ptr->host_params[_LAT_T_]; // already halved (parity-split)
    levels[0].vol=levels[0].X*levels[0].Y*levels[0].Z*levels[0].Lt;
    levels[0].vec_sz=(size_t)levels[0].dof*levels[0].vol;

    // Full-site Lt (un-halved) for restrict/prolong compatibility
    Lt_full = 2 * levels[0].Lt;

    // Parse coarse levels from params
    // param layout: [E, X, Y, Z, T, max_iter, data_type, num_restart] per level
    static const int oE=_MG_LEVEL1_E_,oX=_MG_LEVEL1_X_,oY=_MG_LEVEL1_Y_;
    static const int oZ=_MG_LEVEL1_Z_,oL=_MG_LEVEL1_T_;
    static const int oMI=_MG_LEVEL1_MAX_ITER_;
    static const int oNR=_MG_LEVEL1_NUM_RESTART_;
    for(int i=1;i<num_levels;i++){
      int b=(i-1)*_MG_PARAMS_SIZE_;
      levels[i].dof=set_ptr->host_params[oE+b];
      levels[i].X=set_ptr->host_params[oX+b];
      levels[i].Y=set_ptr->host_params[oY+b];
      levels[i].Z=set_ptr->host_params[oZ+b];
      levels[i].Lt=set_ptr->host_params[oL+b];
      if(levels[i].dof<=0)levels[i].dof=24;
      if(levels[i].X<=0)  levels[i].X=levels[i-1].X/2;
      if(levels[i].Y<=0)  levels[i].Y=levels[i-1].Y/2;
      if(levels[i].Z<=0)  levels[i].Z=levels[i-1].Z/2;
      // SCHUR mode: level 0 Lt is the HALVED (odd-lattice) T = T_full/2;
      // the coarse level is the coarsened odd-lattice, so Lt is coarsened
      // again by 2 (T_full/4).  (For the full-site mode this used the same Lt.)
      if(levels[i].Lt<=0) levels[i].Lt=levels[i-1].Lt/2;
      levels[i].vol=levels[i].X*levels[i].Y*levels[i].Z*levels[i].Lt;
      levels[i].vec_sz=(size_t)levels[i].dof*levels[i].vol;

      // Read per-level max_iter from params; num_restart for coarse level i
      // comes from the NEXT level's restart slot (_MG_LEVEL(i+1)_NUM_RESTART_)
      // because _MG_LEVEL1_NUM_RESTART_ is used for the FINE level-0 frequency.
      levels[i].max_iter=set_ptr->host_params[oMI+b];
      int nr_slot = oNR + b + _MG_PARAMS_SIZE_;   // next level's restart slot
      levels[i].num_restart = (nr_slot < _PARAMS_SIZE_) ? set_ptr->host_params[nr_slot] : 0;
      if(levels[i].max_iter<=0) levels[i].max_iter=50;
      if(levels[i].num_restart<=0) levels[i].num_restart=3;

      // Per-level tolerance from argv
      int tol_idx = _MG_LEVEL1_ATOL_ + (i-1);
      if(tol_idx < _ARGV_SIZE_) levels[i].tol = set_ptr->host_argv[tol_idx];
      else levels[i].tol = atol * (T)0.1; // default: 0.1 × fine tolerance

      levels[i].alloc(levels[i].dof,levels[i].X,levels[i].Y,levels[i].Z,levels[i].Lt,set_ptr->stream);
    }

    max_iter=set_ptr->host_params[_MAX_ITER_];
    atol=set_ptr->host_argv[_ATOL_];
    kappa_val=set_ptr->kappa();

    // Read num_restart from params (FIX: was hardcoded to 3)
    num_restart=set_ptr->host_params[_MG_LEVEL1_NUM_RESTART_];
    if(num_restart<=0) num_restart=3;

    for(int d=0;d<4;d++)mg_grid_size[d]=2;
    if(num_levels>1){
      if(levels[0].X>0&&levels[1].X>0)mg_grid_size[0]=levels[0].X/levels[1].X;
      if(levels[0].Y>0&&levels[1].Y>0)mg_grid_size[1]=levels[0].Y/levels[1].Y;
      if(levels[0].Z>0&&levels[1].Z>0)mg_grid_size[2]=levels[0].Z/levels[1].Z;
      // For coarse level, use full Lt. Level 0 Lt is halved, level 1 Lt is full.
      // Coarsening factor in t: if level 1 has same Lt as level 0 (which is halved),
      // then the effective coarsening in full-site is 2 (parity restore).
      if(levels[0].Lt>0&&levels[1].Lt>0)mg_grid_size[3]=levels[0].Lt/levels[1].Lt;
    }
    null_vecs=new void*[num_levels]; hop_nn=new void*[num_levels];
    hop_diag=new void*[num_levels]; sit_packed=new void*[num_levels];
    for(int i=0;i<num_levels;i++){null_vecs[i]=hop_nn[i]=hop_diag[i]=sit_packed[i]=nullptr;}

    if(rank==0){
      std::ostringstream oss;
      oss<<"PYQCU::SOLVER::MULTIGRID::\n self.dof_list:[";
      for(int i=0;i<num_levels;i++){if(i>0)oss<<", ";oss<<levels[i].dof;}
      oss<<"]\n self.lat_size_list:[";
      for(int i=0;i<num_levels;i++){
        if(i>0)oss<<", ";
        oss<<"["<<levels[i].X<<", "<<levels[i].Y<<", "<<levels[i].Z<<", "<<levels[i].Lt<<"]";
      }
      oss<<"]\n num_restart:"<<num_restart<<"\n tol:"<<std::scientific<<atol
         <<"\n max_iter:"<<max_iter<<"\n Lt_full:"<<Lt_full
         <<"\n mg_grid_size:["<<mg_grid_size[0]<<","<<mg_grid_size[1]<<","
         <<mg_grid_size[2]<<","<<mg_grid_size[3]<<"]";
      log_write<T>(oss.str(),rank,true);
    }
    solve_time_ms=0; for(int i=0;i<8;i++)level_times[i]=0;
  }

  /**
   * @brief Set coarse-grid operators for a given fine level transition.
   *
   * Called from the C API bridge after init() but before run().
   * @param fl   Fine level index (0 = level 0→1, 1 = level 1→2, etc.)
   * @param nv   Null vectors for restrict/prolong [E_{l+1}, 12, X_l, Y_l, Z_l, T_l/2]
   * @param hnn  Nearest-neighbour hopping [2, 4, E, E, X_c, Y_c, Z_c, T_c]
   * @param hdg  Diagonal hopping [2, 2, 6, E, E, X_c, Y_c, Z_c, T_c]
   * @param sp   On-site block [E, E, X_c, Y_c, Z_c, T_c]
   */
  void set_coarse_ops(int fl,void*nv,void*hnn,void*hdg,void*sp){
    if(fl>=0&&fl<num_levels-1){null_vecs[fl]=nv;hop_nn[fl]=hnn;hop_diag[fl]=hdg;sit_packed[fl]=sp;}
  }

  // ==================================================================
  // V-cycle for coarse levels (level ≥ 1).
  //
  // Uses fixed iteration counts (ns pre-smoothing steps + np post-smoothing
  // steps) rather than trying to converge. This is the standard multigrid
  // pattern: the smoother eliminates high-frequency error, and the coarse
  // correction handles the low-frequency components.
  //
  // For coarse levels: ns=5 pre + np=3 post smoothing steps.
  // For coarsest level: ns=10 steps (no coarse correction).
  // ==================================================================
  // ==================================================================
  // Recursive V-cycle for the SCHUR-consistent multigrid.
  //
  // For levels >= 1 (coarse), solve A_lev·x = rhs by BiStabCG converging to
  // a RELATIVE tolerance (matching pyqcu/solver/_multigrid.py cycle()):
  //   * coarsest level:   0.1 × ||rhs||
  //   * inner levels:     0.5 × ||rhs||
  // with a coarse-grid correction (recursive V-cycle) every num_restart
  // iterations.  The coarse operator is the 33-tensor Schur Galerkin
  // A_c = P^T S P (on-site + nearest + diagonal couplings).
  // ==================================================================
  T v_cycle(int lev) {
    auto &st=levels[lev]; cudaStream_t S=set_ptr->stream;
    T rn = 0;
    LatticeComplex<T>*dv=static_cast<LatticeComplex<T>*>(set_ptr->device_vals);

    // ====================================================================
    // COARSEST LEVEL: COOPERATIVE PARALLEL FUSED BiStabCG SOLVE (2026-08-02)
    // --------------------------------------------------------------------
    // The per-iteration coarse path below costs ~1.3 ms/iter on this
    // WSL2/V100 box (~14 tiny kernel launches + 1 host sync per iteration;
    // the wide dslash is only ~2 us).  multigrid_coarse_solve_cg() fuses the
    // ENTIRE BiStabCG solve into ONE cooperative kernel with grid.sync()
    // barriers — no host syncs inside.  Valid at the coarsest level only.
    //
    // 2026-08-15 dev76: for LARGE coarsest levels (vec_sz >= 64K, e.g.
    // 16x16x16x16 2L coarse [8,8,8,4] = 98304 elements) the fused kernel
    // is NOT faster: its per-iteration grid.sync() barriers × many
    // iterations cost more than the ordinary multi-block path (measured
    // 92 ms vs 7 ms per V-cycle).  Fall back to the ordinary iterative
    // path below (which now uses the multi-block reduction kernels).
    // ====================================================================
    // dev84 EXPERIMENT RESULT: fused cooperative solve at vec_sz=294912 ran
    // 85–112 ms/coarse-solve HERE (grid.sync() × ~100 internal iterations are
    // as expensive as any other sync on this box) — SLOWER than graph-replay
    // segments (~50 ms).  Threshold restored to 262144; graphs win at sizes
    // beyond the fused sweet spot.
    if (lev == num_levels - 1 && st.vec_sz < 262144) {
      // Fused single-launch coarse solve.  In the redundant-global multi-rank
      // model every rank holds the full coarse grid and the fused kernel
      // needs no inter-rank data, so the fused path is valid for ALL ranks.
      coarse_solve_fused(lev);
      st.has_solution = true;   // dev84: 供后续 V-cycle 热启动 (fused 内部仍从零解)
      return rn;
    }

    // ---- Save and patch _lat_4dim_ for this level ----
    // The BiStabCG kernels use device_vals[_lat_4dim_] as the site-count
    // stride.  For coarse levels the volume differs, so patch it.
    // dev84: the saved fine value is cached on first use — the blocking D2H
    // below used to cost a full device sync on EVERY v_cycle entry.
    if (!lat4_cache_valid) {
      LatticeComplex<T> saved_lat4;
      checkCudaErrors(cudaMemcpy(&saved_lat4, &dv[_lat_4dim_],
          sizeof(LatticeComplex<T>), cudaMemcpyDeviceToHost));
      lat4_cached = (double)saved_lat4.real();
      lat4_cache_valid = true;
    }
    T saved_lat_4dim = (T)lat4_cached;
    LatticeComplex<T> coarse_vol_val((T)(st.vec_sz / _LAT_SC_), 0.0);
    // dev84 SYNC DIET: async H2D on S orders before all later S work — no sync
    checkCudaErrors(cudaMemcpyAsync(&dv[_lat_4dim_], &coarse_vol_val,
        sizeof(LatticeComplex<T>), cudaMemcpyHostToDevice, S));

    // dev84: capture the SEG-iteration graph BEFORE any other ops this entry
    // (capture scope must contain only the iteration kernels).
    coarse_graph_ensure(lev);

    // ---- Init ----
    // 2026-08-22 dev84 WARM START: 中间/粗层在连续 V-cycle 间复用上次解作为初值
    // (同一 A_c, 不同 RHS)。粗残差在相邻校正间变化缓慢, 从零解起每轮要 ~100 次
    // 粗迭代 (dev84 实测 19 V-cycle 共 2665ms 粗迭代开销); 热启动把每轮粗解成本
    // 压到数十次以内。首轮 (has_solution=false) 保持 x=0 原语义。
    // dev84 SYNC DIET: 全部发射在 S 上按序执行, 中途无需同步。
    bool first_solve = !(st.r0_ref > (T)0);

    // Initial residual norm — dev84: 只在本层首次求解时计算 (host 需要它定标
    // target=r0_ref·tol 与守卫量表 _diff2_tmp_)；热启动周期直接复用 r0_ref,
    // 省掉一次 D2H + MPI_Barrier×2 (~3 次同步/V-cycle)。WSL2 上每次同步 ~1ms。
    // 此处对 rhs 取范数 (冷启动 x=0 ⇒ r=rhs)。
    T r0 = 0;
    if (first_solve || !st.has_solution) {
      LatticeComplex<T> hc0;
      CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasHs[_a_],(int)st.vec_sz,
          st.rhs,1,st.rhs,1,&dv[_send_tmp_]));
      checkCudaErrors(cudaMemcpyAsync(&hc0,&dv[_send_tmp_],sizeof(LatticeComplex<T>),
          cudaMemcpyDeviceToHost,set_ptr->streams[_a_]));
      MPI_Barrier(MPI_COMM_WORLD);checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
      T g0=hc0.real();MPI_Allreduce(MPI_IN_PLACE,&g0,1,mpi_real_type<T>(),MPI_SUM,MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD); r0 = sqrt(g0<0?0:g0);
    } else {
      r0 = st.r0_ref;
    }

    // Skip the coarse solve when the coarse RHS is already tiny: in fp32 the
    // target tol*r0 then falls below the achievable precision, the BiStabCG
    // scalars hit 0/0 and the solve returns NaN, poisoning the fine residual
    // (observed with coarse max_iter=200 at fine residual ~6e-6).  The fine
    // BiStabCG tail converges by itself from ~1e-5 down to atol.
    if (r0 < (T)1e-4) {
      LatticeComplex<T> restore_vol(saved_lat_4dim, 0.0);
      checkCudaErrors(cudaMemcpyAsync(&dv[_lat_4dim_], &restore_vol,
          sizeof(LatticeComplex<T>), cudaMemcpyHostToDevice, S));
      checkCudaErrors(cudaStreamSynchronize(S));
      return rn;
    }

    if(rank==0&&verbose){
      log_write<T>("PYQCU::SOLVER::MULTIGRID::\n "+std::to_string(lev)+":Norm of b:"+std::to_string(r0),rank,true);
      log_write<T>("PYQCU::SOLVER::MULTIGRID::\n "+std::to_string(lev)+":Norm of r:"+std::to_string(r0),rank,true);
      log_write<T>("PYQCU::SOLVER::MULTIGRID::\n "+std::to_string(lev)+":Norm of x0:0.000000",rank,true);
      log_write<T>("PYQCU::SOLVER::MULTIGRID::\n "+std::to_string(lev)+":Starting Iterations",rank,true);
    }

    // ---- ONE-TIME TOP sync ----
    // dev84 SYNC DIET: 粗层所有工作发射在主流 S (cublasH 绑定 S, cdot 内核也在
    // S)。r0 的 D2H 已在其分支内同步 streams[_a_]。这里只保留一次 S 同步,
    // 替换原先的 5 流全同步 (WSL2 每次 ~1ms)。
    checkCudaErrors(cudaStreamSynchronize(S));
    bool is_coarsest = (lev == num_levels-1);
    T tol_factor = levels[lev].tol;
    if (tol_factor <= 0) tol_factor = is_coarsest ? (T)1e-3 : (T)1e-2;
    // ==================================================================
    // dev84 ABSOLUTE-REFERENCED TARGET: with warm starts the incoming ||r||
    // shrinks cycle over cycle, so a purely relative target chases an
    // ever-smaller bar and nearly every solve ran to max_iter (measured:
    // ~200/200 coarse iters on most V-cycles even after warm starting).
    // Anchor the bar to the FIRST solve's ||r||: later (warm) cycles exit as
    // soon as they are as accurate as the first one was required to be.
    // ==================================================================
    if (!(st.r0_ref > (T)0)) st.r0_ref = r0;
    T target = tol_factor * st.r0_ref;
    int idx = 0;
    LatticeComplex<T> one(1,0);
    // rn tracks the PREVIOUS check's residual for the explosion guard.
    rn = r0;

    // ==================================================================
    // dev84 SYNC-FREE GUARDED-BiCGStab COARSE SOLVE: nvprof on this box shows
    // EVERY host↔device round-trip costs ~0.5–40 ms (WSL2 hypervisor thunk;
    // 4248 cudaStreamSynchronize = 6.1 s, 3266 cudaMemcpyAsync = 2.1 s in one
    // solve), while a coarse iteration is only ~75 µs of launches.  We run a
    // FIXED step count with ZERO host reads, relying on
    //   a) guarded scalar kernels (mg_give_1beta/1alpha/1omega): the coarse
    //      operator is γ5-Hermitian — plain CG DIVERGES there (measured:
    //      residual 6084 vs ‖rhs‖≈878 after 200 steps) — BiCGStab stays, but
    //      its ρ→0 breakdown past convergence must be clamped to β=0/α=0/ω=0
    //      instead of NaN,
    //   b) the outer fine-level true-residual restart guard,
    //   c) ONE safety check every SAFETY_EVERY V-cycles (+ cold solves).
    // ==================================================================
    {
      // problem-scale stash for the guards: ‖b‖² (units of every ⟨·,·⟩)
      LatticeComplex<T> sc(st.r0_ref * st.r0_ref, 0.0);
      checkCudaErrors(cudaMemcpyAsync(&dv[_diff2_tmp_], &sc,
          sizeof(LatticeComplex<T>), cudaMemcpyHostToDevice, S));
    }
    // warm start: r = rhs − A_c·x with CURRENT x; cold: x=0 ⇒ r=rhs
    if (!st.has_solution) {
      zero_c(st.x, lev);
      copy_c(st.r, st.rhs, lev);
    } else {
      copy_c(st.r, st.rhs, lev);
      coarse_dslash_op(set_ptr->device_vec0, st.x, lev);
      LatticeComplex<T> mone_ws(-1.0, 0.0);
      CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH, (int)st.vec_sz, &mone_ws,
          set_ptr->device_vec0, 1, st.r, 1));
    }
    copy_c(st.r_tilde, st.r, lev);
    zero_c(st.p, lev); zero_c(st.v, lev); zero_c(st.s, lev); zero_c(st.t, lev);
    {
      LatticeComplex<T> z(0,0);
      checkCudaErrors(cudaMemcpyAsync(&dv[_rho_],&z,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
      checkCudaErrors(cudaMemcpyAsync(&dv[_rho_prev_],&one,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
      checkCudaErrors(cudaMemcpyAsync(&dv[_alpha_],&one,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
      checkCudaErrors(cudaMemcpyAsync(&dv[_omega_],&one,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
    }

    const int SAFETY_EVERY = 6;
    ++mg_cycle_counter;
    const bool warm = st.has_solution;
    const bool do_safety = (!warm) || (mg_cycle_counter % SAFETY_EVERY == 0);

    auto ci0=std::chrono::high_resolution_clock::now();
    if (is_coarsest) {
      // ---- segment-replay solve: GRAPH_SEG iterations per host check ----
      // dev84: 每段一次检查 — 图回放把 8 次迭代压成 1 次 launch, 队列浅,
      // 检查同步在本箱上从 ~40-500ms 掉回毫秒级; 收敛语义与 r5 一致
      // (target=r0_ref·tol, 爆炸即丢弃 x_c)。
      int done = 0;
      while (done < st.max_iter) {
        coarse_graph_run(lev, GRAPH_SEG);
        done += GRAPH_SEG; idx = done;
        auto ck0=std::chrono::high_resolution_clock::now();
        T rn_new = coarse_resid_norm(lev);
        auto ck1=std::chrono::high_resolution_clock::now();
        prof_check_ms += std::chrono::duration<double,std::milli>(ck1-ck0).count();
        prof_n_checks++;
        if (!std::isfinite(rn_new) || rn_new > (T)1e4 * rn) {
          rn = rn_new; st.has_solution = false; zero_c(st.x, lev); break;
        }
        rn = rn_new;
        if (rn < target) break;
      }
      if(rank==0&&verbose){
        std::ostringstream bm;
        bm<<"PYQCU::SOLVER::MULTIGRID::\n B-"<<lev<<"-bicg it "<<idx
          <<": Residual = "<<std::scientific<<rn<<" (target "<<target<<")";
        log_write<T>(bm.str(),rank,true);
      }
    } else {
      // ---- Inner level: pre-smooth → coarse correction → post-smooth ----
      // dev84: 原实现每 num_restart 次迭代校正一次并全流同步；固定步数语义下
      // 改为 DDalphaAMG pre/coarse/post 结构。第二段前重算真残差并复位状态。
      const int steps = warm ? 64 : st.max_iter;
      const int pre = steps/2, post = steps - pre;
      idx = steps;
      coarse_graph_run(lev, pre);
      restrict_op(levels[lev+1].rhs, st.r, lev);
      zero_c(levels[lev+1].x, lev+1);
      v_cycle(lev+1);
      prolong_op(set_ptr->device_vec1, levels[lev+1].x, lev);
      CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH, (int)st.vec_sz, &one,
          set_ptr->device_vec1, 1, st.x, 1));
      coarse_dslash_op(set_ptr->device_vec0, st.x, lev);
      bistabcg_give_diff2<T><<<site_grid(lev),_BLOCK_SIZE_,0,S>>>(
          st.rhs, set_ptr->device_vec0, st.r, dv);
      copy_c(st.r_tilde, st.r, lev);
      reset_bistabcg_state(lev);
      coarse_graph_run(lev, post);
    }
    auto ci1=std::chrono::high_resolution_clock::now();
    prof_coarse_solve_ms += std::chrono::duration<double,std::milli>(ci1-ci0).count();

    // dev84: 记录本层已有解, 下一 V-cycle 热启动
    st.has_solution = true;
    prof_n_coarse_iters += idx;

    // ---- Restore _lat_4dim_ ----
    // dev84 SYNC DIET: async H2D on S — 后续内核同流有序, 无需同步
    LatticeComplex<T> restore_vol(saved_lat_4dim, 0.0);
    checkCudaErrors(cudaMemcpyAsync(&dv[_lat_4dim_], &restore_vol,
        sizeof(LatticeComplex<T>), cudaMemcpyHostToDevice, S));

    return rn;
  }

  // ==================================================================
  // Initialize solver with fine-level data.
  // ==================================================================
  void init(void*_fo,void*_fi,void*_g,void*_ce,void*_co,void*_cei,void*_coi){
    fermion_out_eo=_fo;fermion_in_eo=_fi;gauge=_g;
    clover_ee=_ce;clover_oo=_co;clover_ee_inv=_cei;clover_oo_inv=_coi;
    clover_dslash_ee.init(clover_ee);clover_dslash_oo.init(clover_oo);
    clover_dslash_ee_inv.init(clover_ee_inv);clover_dslash_oo_inv.init(clover_oo_inv);
    parse_params();
    sap.give(set_ptr);

    // Multi-rank (multi-GPU) mode: 4D process grid != 1x1x1x1.  Coarse dslash
    // and coarse dots then dispatch to the MPI paths (Allgather + Allreduce).
    mg_multi = !(set_ptr->host_params[_GRID_X_] == 1 &&
                 set_ptr->host_params[_GRID_Y_] == 1 &&
                 set_ptr->host_params[_GRID_Z_] == 1 &&
                 set_ptr->host_params[_GRID_T_] == 1);
    if (mg_multi && rank == 0 && verbose) {
      log_write<T>("PYQCU::QCU::MULTIGRID::\n MG_MULTI_RANK: process grid ["+
        std::to_string(set_ptr->host_params[_GRID_X_])+","+
        std::to_string(set_ptr->host_params[_GRID_Y_])+","+
        std::to_string(set_ptr->host_params[_GRID_Z_])+","+
        std::to_string(set_ptr->host_params[_GRID_T_])+"] — coarse MPI paths active",
        rank, true);
    }

    // Set up parity-split pointers (even/odd parts of fermion buffers)
    b_e=fermion_in_eo;
    b_o=static_cast<LatticeComplex<T>*>(fermion_in_eo)+set_ptr->lat_4dim_SC;
    x_o=static_cast<LatticeComplex<T>*>(fermion_out_eo)+set_ptr->lat_4dim_SC;
    checkCudaErrors(cudaMemsetAsync(x_o,0,set_ptr->lat_4dim_SC*sizeof(LatticeComplex<T>),set_ptr->stream));

    // Allocate level-0 BiStabCG working vectors (parity-split layout)
    size_t sc=set_ptr->lat_4dim_SC*sizeof(LatticeComplex<T>);
    checkCudaErrors(cudaMallocAsync(&b__o,sc,set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&r0,sc,set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&rt0,sc,set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&p0,sc,set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&v0,sc,set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&s0,sc,set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&t0,sc,set_ptr->stream));

    // dev84: zero-copy staging for scalar host reads (see coarse_resid_norm)
    checkCudaErrors(cudaHostAlloc(&check_host, 4*sizeof(LatticeComplex<T>),
        cudaHostAllocMapped));
    checkCudaErrors(cudaHostGetDevicePointer(&check_dev, check_host, 0));

    // Allocate full-site residual buffer for V-cycle correction
    // Full-site size = _LAT_SC_ * X * Y * Z * Lt_full where Lt_full = 2 * levels[0].Lt
    size_t r_full_sc = (size_t)_LAT_SC_ * levels[0].X * levels[0].Y * levels[0].Z * Lt_full;
    size_t r_full_bytes = r_full_sc * sizeof(LatticeComplex<T>);
    checkCudaErrors(cudaMallocAsync(&r_full, r_full_bytes, set_ptr->stream));
    checkCudaErrors(cudaMemsetAsync(r_full, 0, r_full_bytes, set_ptr->stream));

    // Dedicated odd-correction buffer (one parity channel, like x_o).
    size_t e_odd_bytes = (size_t)_LAT_SC_ * levels[0].vol * sizeof(LatticeComplex<T>);
    checkCudaErrors(cudaMallocAsync(&e_odd_buf, e_odd_bytes, set_ptr->stream));
    checkCudaErrors(cudaMemsetAsync(e_odd_buf, 0, e_odd_bytes, set_ptr->stream));

    // ---- Full-site level-0 solve buffers ----
    // The level-0 solve uses FULL-site vectors [sc, X, Y, Z, T]
    // (size 2*lat_4dim_SC) matching pyqcu/solver/_multigrid.py with
    // support_parity=False.  We do NOT use the parity-preconditioned Schur
    // complement for level 0 because its low modes are not captured by the
    // coarse space (that made the V-cycle correction ineffective).
    size_t full_bytes = 2 * sc;   // full-site vector = 2 parity channels
    checkCudaErrors(cudaMallocAsync(&full_x,   full_bytes, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&full_rhs, full_bytes, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&full_r,   full_bytes, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&full_rt,  full_bytes, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&full_p,   full_bytes, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&full_v,   full_bytes, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&full_s,   full_bytes, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&full_t,   full_bytes, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&parity_tmp, full_bytes, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&parity_dst, full_bytes, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&corr_scratch, full_bytes, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&coarse_partials,
        1024 * sizeof(LatticeComplex<T>), set_ptr->stream));

    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));

    // Wire level 0 to the ODD-site (Schur) working vectors — this matches
    // pyqcu/solver/_multigrid.py with support_parity=True and is consistent
    // with applyCloverBistabCgDslashQcu (the parity-preconditioned Clover dslash
    // S·v = D_oo·v - k^2·H_oe·D_ee^{-1}·H_eo·v).
    // x_o is the odd part of the output buffer (fermion_out_eo + lat_4dim_SC).
    levels[0].x=x_o; levels[0].rhs=b__o; levels[0].r=r0; levels[0].r_tilde=rt0;
    levels[0].p=p0; levels[0].v=v0; levels[0].s=s0; levels[0].t=t0;
    levels[0].owned=false;
    levels[0].is_fullsite=false;
    // Level 0 Lt is HALVED in parse_params (parity odd-lattice T/2).
    // Coarse levels are the coarsened odd-lattice (is_fullsite=false).

    if(rank==0){
      std::ostringstream oss;
      oss<<"PYQCU::QCU::MULTIGRID::\n MG_INIT_COMPLETE: "<<num_levels
         <<" levels, Lt_full="<<Lt_full
         <<", num_restart="<<num_restart;
      log_write<T>(oss.str(),rank,true);
    }
  }

  // ==================================================================
  // Main solve — FULL-operator BiStabCG at level 0 with V-cycle corrections.
  //
  // This matches pyqcu/solver/_multigrid.py with support_parity=False:
  //   * level 0 solves the FULL Clover-Wilson operator D·x = b on full-site
  //     vectors [sc, X, Y, Z, T]  (NOT the even-odd preconditioned Schur
  //     complement — the coarse space captures the low modes of D much better
  //     than those of the Schur-complement operator, so the V-cycle correction
  //     actually reduces the iteration count).
  //   * input/output is parity-split (consistent with applyCloverBistabCgQcu):
  //     b_eo is first combined into a full-site RHS, and the final full-site
  //     solution is split back into parity-split output.
  // ==================================================================
  // ==================================================================
  // Main solve — SCHUR (parity-preconditioned) level-0 BiStabCG with
  // recursive V-cycle coarse corrections.
  //
  // Solves  S·x_o = b__o  where  S = D_oo - k^2·H_oe·D_ee^{-1}·H_eo
  // and     b__o = b_o + k·H_oe·D_ee^{-1}·b_e.
  // The level-0 dslash (fine_dslash_op) is exactly applyCloverBistabCgDslashQcu,
  // so the solver is consistent with the Clover BiStabCG reference.
  //
  // The coarse space is built from the Schur operator's own null vectors
  // (capturing S's low modes, which differ from the full operator D's low
  // modes — this is why the previous full-D-based coarse space was ineffective).
  // V-cycle: restrict the SCHUR residual directly, solve the coarse operator
  // A_c = P^T S P (33-tensor stencil), prolong, add to x_o, recompute the
  // Schur residual, and reset the BiStabCG state.
  // ==================================================================
  void run() {
    if (set_ptr->host_params[_MG_USE_GCR_] != 0) { run_gcr(); return; }
    auto t0=std::chrono::high_resolution_clock::now();
    auto &st=levels[0]; cudaStream_t S=set_ptr->stream;
    LatticeComplex<T>*dv=static_cast<LatticeComplex<T>*>(set_ptr->device_vals);

    // 1. Build the Schur RHS: b__o = b_o + k·H_oe·D_ee^{-1}·b_e
    setup_b__o();
    checkCudaErrors(cudaStreamSynchronize(S));

    // 2. Initialise BiStabCG state on the ODD vectors (x=0, r=b__o, ...)
    checkCudaErrors(cudaMemsetAsync(st.x, 0, st.vec_sz*sizeof(LatticeComplex<T>), S));
    copy_c(st.r, st.rhs, 0);
    // ---- dev84 R5 DEFLATED START ----------------------------------------
    // quda/DDalphaAMG 粗空间的主要价值在"低模消除"。把一次 V-cycle 校正作为
    // 初始猜测 (x₀ = P·A_c⁻¹·Pᵀ·b)：近零模被粗解吸收后剩余谱聚于高频带，
    // Krylov 收敛大幅加快。与循环内校正不同：只做一次、**不触碰任何
    // BiCGStab 状态**（r̂₀ 直接取收缩后的真残差）。循环内校正仍由
    // num_restart 独立控制（rs=0 ⇒ 纯 deflated BiCGStab）。
    if(num_levels>1 && set_ptr->host_params[_MG_USE_DEFLATE_]!=0){
      auto df_t0=std::chrono::high_resolution_clock::now();
      prof_n_vcycles++;
      restrict_op(levels[1].rhs, st.r, 0);
      zero_c(levels[1].x, 1);
      checkCudaErrors(cudaStreamSynchronize(S));
      v_cycle(1);
      prolong_op(e_odd_buf, levels[1].x, 0);
      checkCudaErrors(cudaStreamSynchronize(S));
      {
        LatticeComplex<T> one_df(1,0);
        CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH,(int)set_ptr->lat_4dim_SC,
            &one_df, e_odd_buf, 1, x_o, 1));
      }
      fine_dslash_op(set_ptr->device_vec0, x_o);
      checkCudaErrors(cudaStreamSynchronize(S));
      bistabcg_give_diff2<T><<<set_ptr->gridDim,set_ptr->blockDim,0,S>>>(
          st.rhs, set_ptr->device_vec0, st.r, dv);
      checkCudaErrors(cudaStreamSynchronize(S));
      prof_vcycle_ms += std::chrono::duration<double,std::milli>(
          std::chrono::high_resolution_clock::now()-df_t0).count();
      if(rank==0&&verbose)
        log_write<T>("PYQCU::SOLVER::MULTIGRID::\n DEFLATE start applied",rank,true);
    }
    copy_c(st.r_tilde, st.r, 0);
    zero_c(st.p, 0); zero_c(st.v, 0); zero_c(st.s, 0); zero_c(st.t, 0);
    {
      LatticeComplex<T> one(1,0), z(0,0);
      checkCudaErrors(cudaMemcpyAsync(&dv[_rho_],&z,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
      checkCudaErrors(cudaMemcpyAsync(&dv[_rho_prev_],&one,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
      checkCudaErrors(cudaMemcpyAsync(&dv[_alpha_],&one,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
      checkCudaErrors(cudaMemcpyAsync(&dv[_omega_],&one,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
    }
    checkCudaErrors(cudaStreamSynchronize(S));

    // 3. Log the initial Schur residual norm ||b__o||
    if(rank==0){
      LatticeComplex<T> ht;
      CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasHs[_a_], set_ptr->lat_4dim_SC,
          st.rhs,1,st.rhs,1, &dv[_send_tmp_]));
      checkCudaErrors(cudaMemcpyAsync(&ht,&dv[_send_tmp_],sizeof(LatticeComplex<T>),
          cudaMemcpyDeviceToHost,set_ptr->streams[_a_]));
      MPI_Barrier(MPI_COMM_WORLD); checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
      T g=ht.real(); MPI_Allreduce(MPI_IN_PLACE,&g,1,mpi_real_type<T>(),MPI_SUM,MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      T nb=sqrt(g<0?0:g);
      log_write<T>("PYQCU::SOLVER::MULTIGRID::\n 0:Norm of b:"+std::to_string(nb),rank,true);
      log_write<T>("PYQCU::SOLVER::MULTIGRID::\n 0:Norm of r:"+std::to_string(nb),rank,true);
      log_write<T>("PYQCU::SOLVER::MULTIGRID::\n 0:Norm of x0:0.000000",rank,true);
      log_write<T>("PYQCU::SOLVER::MULTIGRID::\n 0:Starting Iterations",rank,true);
    }

    // ---- ONE-TIME initial sync ----
    checkCudaErrors(cudaStreamSynchronize(S));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

    // 4. Main BiStabCG loop (level-0 Schur)
    T atol2=atol*atol;
    int total=0; double tti=0;
    int count_restart=0;
    LatticeComplex<T> one(1,0);

    // ==================================================================
    // dev84 ADAPTIVE CORRECTION GATE: 实测(报告§3)在连续谱算子上 V-cycle
    // 校正对迭代数无净收益(138→138)而每次校正 ~50ms —— 多层恒劣于 L1。
    // 门控策略：先跑 mg_calib_iters 次纯 BiCGStab 标定收敛斜率 s0；
    // 之后允许校正, 但每个观测窗(W=12 迭代)测斜率 s1, 若连续 2 窗
    // s1 < 1.15·s0 (加速不足) 则永久停用校正 → 多层性能退回并稳定在
    // L1 水平, 保证 MG 永不劣于基线。粗空间有效时门控自动保持开启。
    // ==================================================================
    const int mg_calib_iters = 30;
    const int mg_gate_window = 12;
    const double mg_gate_factor = 1.15;
    double mg_ln_rn0 = 0, mg_slope0 = 0;
    int mg_calib_done = 0;
    int mg_win_iters = 0; double mg_win_rn = 0; int mg_fail_wins = 0;
    bool mg_corr_off = false;

    for(int it=0; it<max_iter; it++){
      auto ti0=std::chrono::high_resolution_clock::now();
      bistabcg_iter(0);
      auto ti1=std::chrono::high_resolution_clock::now();
      double sec=std::chrono::duration<double>(ti1-ti0).count(); tti+=sec; total++;
      prof_fine_iter_ms += std::chrono::duration<double,std::milli>(ti1-ti0).count();
      count_restart++;

      T rn2=check_host[1].real();   // dev84: zero-copy fine norm (mapped)
      T rn=sqrt(rn2<0?0:rn2);
      conv_history.push_back(rn);

      // ---- dev84 自适应校正门控: 标定与窗口斜率监测 ----
      if(num_levels>1 && num_restart>0){
        if(!mg_calib_done){
          if(total==1){ mg_ln_rn0 = std::log((double)rn); }
          if(total>=mg_calib_iters){
            double ln_now=std::log((double)std::max(rn,(T)1e-30));
            mg_slope0=(ln_now-mg_ln_rn0)/mg_calib_iters;
            mg_calib_done=1; mg_win_iters=0; mg_win_rn=std::log((double)std::max(rn,(T)1e-30));
            if(rank==0&&verbose)
              log_write<T>("PYQCU::SOLVER::MULTIGRID::\n GATE: calibrated slope0="
                +std::to_string(mg_slope0),rank,true);
          }
        } else if(!mg_corr_off){
          mg_win_iters++;
          if(mg_win_iters>=mg_gate_window){
            double ln_now=std::log((double)std::max(rn,(T)1e-30));
            double s1=(ln_now-mg_win_rn)/mg_win_iters;
            if(s1 < mg_gate_factor*mg_slope0){
              if(++mg_fail_wins>=2){
                mg_corr_off=true;
                if(rank==0&&verbose)
                  log_write<T>("PYQCU::SOLVER::MULTIGRID::\n GATE: corrections disabled"
                    " (s1="+std::to_string(s1)+" < "+std::to_string(mg_gate_factor)
                    +"*s0="+std::to_string(mg_slope0)+")",rank,true);
              }
            } else { mg_fail_wins=0; }
            mg_win_iters=0; mg_win_rn=ln_now;
          }
        }
      }

      if(rank==0&&verbose){
        std::ostringstream bm,fm;
        bm<<"PYQCU::SOLVER::MULTIGRID::\n B-0-bistabcg-Iteration "<<it
          <<": Residual = "<<std::scientific<<rn;
        log_write<T>(bm.str(),rank,true);
        fm<<"PYQCU::SOLVER::MULTIGRID::\n F-0-bistabcg-Iteration "<<it
          <<": Residual = "<<std::scientific<<rn<<", Time = "<<std::fixed<<std::setprecision(6)<<sec<<" s";
        log_write<T>(fm.str(),rank,true);
      }

      // Divergence safeguard — BiStabCG can break down (rho≈0) on the
      // ill-conditioned Schur operator; restart from the CURRENT x_o.
      if(!std::isfinite(rn)||rn>(T)1e10){
        if(rank==0&&verbose)log_write<T>("PYQCU::SOLVER::MULTIGRID::\n Restart at "+std::to_string(it),rank,true);
        fine_dslash_op(set_ptr->device_vec0, x_o);          // S·x_o
        checkCudaErrors(cudaStreamSynchronize(S));
        bistabcg_give_diff2<T><<<set_ptr->gridDim,set_ptr->blockDim,0,S>>>(
            st.rhs, set_ptr->device_vec0, st.r, dv);        // st.r = b__o - S·x_o
        copy_c(st.r_tilde, st.r, 0);
        reset_bistabcg_state_l0();
        count_restart=0;
        continue;
      }

      // Convergence check (Schur residual)
      if(rn2<atol2){
        if(rank==0&&verbose)
          log_write<T>("PYQCU::SOLVER::MULTIGRID::\n Converged at iteration "+
            std::to_string(it)+" with residual "+std::to_string(rn),rank,true);
        break;
      }

      // ---- V-cycle coarse correction ----
      // Skip once the fine Schur residual is small: the coarse solve cannot
      // deliver corrections below its fp32 precision floor (~1e-5·||P^T·r||),
      // and an empty/noisy correction only resets the BiStabCG Krylov space,
      // bouncing the residual back to ~1e-5 (observed: 500 full iterations).
      if(num_levels>1 && num_restart>0 && !mg_corr_off && mg_calib_done
         && count_restart>=num_restart && rn > (T)100 * atol){

        auto vc_t0=std::chrono::high_resolution_clock::now();
        prof_n_vcycles++;
        if(rank==0&&verbose)
          log_write<T>("PYQCU::SOLVER::MULTIGRID::\n V-cycle correction at iteration "+std::to_string(it),rank,true);

        // 0. SAP pre-smoothing: 16-color 1h coding tested 0.746× (1.688→2.261, 6vc534ms vs 159ms)
        //    16×0.12ms=1.9ms + 12ms dslash =14ms per VC, 6×14=84ms extra, but 534-159=375ms extra (4×), 137 vs 147 iters (-10) not enough, 0.746× <0.88× baseline, revert to disabled.
        //    True 16-color MINRES (5-step block) diverged 0.12× (1000it 10), left for next 1h FGMRES精修.
        // 1. Restrict the SCHUR residual (odd-site) -> coarse RHS
        // dev84 SYNC DIET: 全部工作发射在主流 S 上按序执行, 中途同步全部去除
        // (WSL2 每次 host↔device 往返 ~1-40ms, 见 v_cycle 内注释)。
        restrict_op(levels[1].rhs, st.r, 0);
        zero_c(levels[1].x, 1);

        // 2. Solve the coarse problem A_c·e_c = P^T·r (recursive V-cycle)
        v_cycle(1);

        // 3. Prolong the coarse correction -> odd-site e_fine
        prolong_op(e_odd_buf, levels[1].x, 0);

        // 4. x_o += e_fine
        CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH, set_ptr->lat_4dim_SC, &one,
            e_odd_buf, 1, x_o, 1));

        // 5. Recompute the Schur residual: st.r = b__o - S·x_o
        fine_dslash_op(set_ptr->device_vec0, x_o);
        bistabcg_give_diff2<T><<<set_ptr->gridDim,set_ptr->blockDim,0,S>>>(
            st.rhs, set_ptr->device_vec0, st.r, dv);

        // 6. Reset the BiStabCG state (r_tilde=r, p=v=s=t=0, scalars)
        copy_c(st.r_tilde, st.r, 0);
        reset_bistabcg_state_l0();
        count_restart=0;

        auto vc_t1=std::chrono::high_resolution_clock::now();
        prof_vcycle_ms += std::chrono::duration<double,std::milli>(vc_t1-vc_t0).count();
      }
    }

    // ---- Final sync ----
    checkCudaErrors(cudaStreamSynchronize(S));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

    // 5. Recover the even part of the solution and write the parity-split output.
    //    recover_x_e() writes x_e = D_ee^{-1}(b_e + k·H_eo·x_o) to the EVEN part
    //    of fermion_out_eo; x_o is already in the ODD part.  This matches the
    //    applyCloverBistabCgQcu output layout exactly.
    recover_x_e();
    checkCudaErrors(cudaStreamSynchronize(S));

    auto t1=std::chrono::high_resolution_clock::now();
    solve_time_ms=std::chrono::duration<double,std::milli>(t1-t0).count();

    // ---- Performance report ----
    if(rank==0){
      double avg=total>0?tti/total:0;
      T fn=conv_history.empty()?0:conv_history.back();
      log_write<T>("PYQCU::SOLVER::MULTIGRID::\n Performance Statistics:",rank,true);
      log_write<T>("PYQCU::SOLVER::MULTIGRID::\n Total iterations: "+std::to_string(total),rank,true);
      std::ostringstream tm;tm<<"PYQCU::SOLVER::MULTIGRID::\n Total time: "<<std::fixed<<std::setprecision(6)<<(solve_time_ms/1000.0)<<" seconds";
      log_write<T>(tm.str(),rank,true);
      std::ostringstream am;am<<"PYQCU::SOLVER::MULTIGRID::\n Average time per iteration: "<<std::fixed<<std::setprecision(6)<<avg<<" s";
      log_write<T>(am.str(),rank,true);
      std::ostringstream fm;fm<<"PYQCU::SOLVER::MULTIGRID::\n Final residual: "<<std::scientific<<fn;
      log_write<T>(fm.str(),rank,true);
      std::ostringstream ch;ch<<"CONVERGENCE_HISTORY: [";
      for(size_t j=0;j<conv_history.size();j++){if(j>0)ch<<",";ch<<std::scientific<<conv_history[j];}
      ch<<"]";log_write<T>(ch.str(),rank,false);
      std::ostringstream prof;
      prof<<"PROF_COARSE: dslash="<<std::fixed<<std::setprecision(1)<<prof_coarse_dslash_ms
          <<"ms dot="<<prof_coarse_dot_ms<<"ms";
      log_write<T>(prof.str(),rank,false);
      // ---- Section timing report (2026-08-02) ----
      std::ostringstream sect;
      sect<<"PROF_SECTIONS: fine_iter="<<std::fixed<<std::setprecision(1)
          <<prof_fine_iter_ms<<"ms vcycle="<<prof_vcycle_ms
          <<"ms n_vcycles="<<prof_n_vcycles
          <<" coarse_solve="<<prof_coarse_solve_ms
          <<"ms coarse_vec="<<prof_coarse_vec_ms
          <<"ms coarse_dslash="<<prof_coarse_dslash_ms
          <<"ms coarse_iters="<<prof_n_coarse_iters
          <<" checks="<<prof_n_checks<<" check_ms="<<prof_check_ms
          <<" ck(k/d/s)=" <<prof_ck_kernel_ms<<"/"<<prof_ck_d2h_ms
          <<"/"<<prof_ck_sync_ms;
      log_write<T>(sect.str(),rank,true);
    }
  }

  // ==================================================================
  // MG preconditioner: one V-cycle (restrict + coarse solve + prolong)
  // Input:  in  (fine residual, odd)
  // Output: out (preconditioned vector, odd)
  //
  // 2026-08-22 dev84 — quda MG::operator() (lib/multigrid.cpp:1131) 对齐改造:
  //   pre-smoothing (μ_pre 固定步 CG) → R → coarse → P。
  //   此前的"纯粗校正"预条件子有两个致命问题:
  //   a) 无平滑 → 高频误差全留给外层 FGMRES, ρ≈0.5+ 无加速;
  //   b) 粗 RHS < 1e-4 时 v_cycle 跳过解 → 返回零向量 → Arnoldi 崩溃
  //      (实测 dev84 GCR 轮: 1100 iters 残差卡在 0.77)。
  //   平滑器选固定步数 CG: Schur 补 S = D_oo − κ²H_oe D_ee⁻¹ H_eo 是
  //   Hermitian 正定 (D_ee⁻†=D_ee⁻¹, H_oe†=H_eo), CG 等效 Chebyshev,
  //   每步恰 1 次 matvec、无收敛分支 (quda smoother_tol/Nsteps 语义)。
  //   复用 levels[0] 的 BiCGStab 空闲缓冲 rt0(r_s)/p0(p_s)/device_vec1(Ap_s)
  //   —— GCR 模式下 run() 主循环不在运行, 无冲突。
  // ==================================================================
  void apply_mg_prec(void *out, void *in) {
    cudaStream_t S=set_ptr->stream;
    if (num_levels <= 1) {
      // No coarse level: identity preconditioner
      checkCudaErrors(cudaMemcpyAsync(out, in, set_ptr->lat_4dim_SC*sizeof(LatticeComplex<T>), cudaMemcpyDeviceToDevice, S));
      checkCudaErrors(cudaStreamSynchronize(S));
      return;
    }
    LatticeComplex<T> *dv=static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
    const int n=(int)set_ptr->lat_4dim_SC;
    int mu_pre = set_ptr->host_params[_MG_MU_PRE_];
    if (mu_pre <= 0) mu_pre = 4;   // dev84 默认 4 步（quda smoother Nsteps 语义）

    // ---- init: out=0, r_s=rt0←in, p_s=p0←r, rr=<r,r>, ω槽清零 ----
    // dev84: rr/rr_prev 驻留设备端 (_tmp0_/_rho_prev_)，循环内零 host 同步。
    checkCudaErrors(cudaMemsetAsync(out, 0, n*sizeof(LatticeComplex<T>), S));
    CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH,n*_REAL_IMAG_,(T*)in,1,(T*)rt0,1));
    CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH,n*_REAL_IMAG_,(T*)rt0,1,(T*)p0,1));
    {
      LatticeComplex<T> z(0,0);
      checkCudaErrors(cudaMemcpyAsync(&dv[_omega_],&z,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
    }
    CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasH,n,rt0,1,rt0,1,&dv[_tmp0_]));
    checkCudaErrors(cudaMemcpyAsync(&dv[_rho_prev_],&dv[_tmp0_],
        sizeof(LatticeComplex<T>),cudaMemcpyDeviceToDevice,S));

    const int cg_blocks = (n + 255) / 256;

    // ---- μ_pre 步固定 CG (无收敛检查；α/β 设备端计算, quda Nsteps 语义) ----
    // 每步: Ap=S·p → pv=<p,Ap> → α=rr/pv → x+=αp, r-=αAp → rr=<r,r>
    //       → β=rr_new/rr_prev → p=r+βp。全程单流 S, 无任何同步/D2H。
    for (int k=0;k<mu_pre;k++) {
      fine_dslash_op(set_ptr->device_vec1, p0);            // Ap = S·p
      CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasH,n,p0,1,set_ptr->device_vec1,1,&dv[_tmp1_]));
      mg_cg_give_alpha<T><<<1,1,0,S>>>(dv);
      mg_cg_update_xr<T><<<cg_blocks,256,0,S>>>(
          static_cast<LatticeComplex<T>*>(out),
          static_cast<const LatticeComplex<T>*>(p0),
          static_cast<LatticeComplex<T>*>(rt0),
          static_cast<const LatticeComplex<T>*>(set_ptr->device_vec1), dv, n);
      CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasH,n,rt0,1,rt0,1,&dv[_tmp0_]));
      mg_cg_give_beta<T><<<1,1,0,S>>>(dv);
      mg_cg_update_p<T><<<cg_blocks,256,0,S>>>(
          static_cast<LatticeComplex<T>*>(p0),
          static_cast<const LatticeComplex<T>*>(rt0), dv, n);
    }
    checkCudaErrors(cudaStreamSynchronize(S));

    // ---- R: 当前残差限制到粗层 ----
    restrict_op(levels[1].rhs, rt0, 0);
    // dev84: 不再清零 levels[1].x —— coarse_solve_cg 普通路径配合 has_solution
    // 热启动可跨 V-cycle 复用粗解；fused 路径内核从零起算并整体覆写 x，
    // 两条路径语义均正确。
    checkCudaErrors(cudaStreamSynchronize(S));
    // ---- coarse solve ----
    v_cycle(1);
    // ---- P: 校正延拓回细层并累加 ----
    prolong_op(set_ptr->device_vec1, levels[1].x, 0);
    checkCudaErrors(cudaStreamSynchronize(S));
    LatticeComplex<T> one(1,0);
    CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH, n, &one,
        set_ptr->device_vec1, 1, out, 1));
    checkCudaErrors(cudaStreamSynchronize(S));
    // ν_post 平滑省略 (v1): 外层 FGMRES 承担剩余高频误差
  }

  // ==================================================================
  // FGMRES(m) with MG preconditioning — DDalphaAMG C7 / QUDA FGMRES(10)
  // Replaces GCR when params[_MG_USE_GCR_]!=0 (reused flag for FGMRES).
  // FGMRES is more stable for variable MG preconditioning than GCR/BiStabCG.
  // m=10, max_iter from params, restart outer loop, Givens QR on host.
  // ==================================================================
  void run_gcr() {
    auto t0=std::chrono::high_resolution_clock::now();
    auto &st=levels[0]; cudaStream_t S=set_ptr->stream;
    LatticeComplex<T>*dv=static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
    setup_b__o();
    checkCudaErrors(cudaStreamSynchronize(S));
    checkCudaErrors(cudaMemsetAsync(st.x, 0, st.vec_sz*sizeof(LatticeComplex<T>), S));
    copy_c(st.r, st.rhs, 0);
    checkCudaErrors(cudaStreamSynchronize(S));
    const int m = 10;
    std::vector<void*> V(m+1), Z(m);
    for(int i=0;i<=m;i++){ checkCudaErrors(cudaMallocAsync(&V[i], st.vec_sz*sizeof(LatticeComplex<T>), S)); checkCudaErrors(cudaMemsetAsync(V[i],0,st.vec_sz*sizeof(LatticeComplex<T>),S)); }
    for(int i=0;i<m;i++){ checkCudaErrors(cudaMallocAsync(&Z[i], st.vec_sz*sizeof(LatticeComplex<T>), S)); checkCudaErrors(cudaMemsetAsync(Z[i],0,st.vec_sz*sizeof(LatticeComplex<T>),S)); }
    void *w; checkCudaErrors(cudaMallocAsync(&w, st.vec_sz*sizeof(LatticeComplex<T>), S));
    checkCudaErrors(cudaStreamSynchronize(S));
    // Host Hessenberg and Givens
    std::vector<std::complex<T>> H((m+1)*m, 0), cs(m,0), sn(m,0), s(m+1,0), y(m,0);
    // Precompute b norm
    T b_norm2=0;
    {
      CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasH, set_ptr->lat_4dim_SC, st.rhs,1,st.rhs,1,&dv[_send_tmp_]));
      checkCudaErrors(cudaMemcpyAsync(&host_vals[_send_tmp_],&dv[_send_tmp_],sizeof(LatticeComplex<T>),cudaMemcpyDeviceToHost,S));
      checkCudaErrors(cudaStreamSynchronize(S));
      b_norm2 = host_vals[_send_tmp_].real();
      MPI_Allreduce(MPI_IN_PLACE,&b_norm2,1,mpi_real_type<T>(),MPI_SUM,MPI_COMM_WORLD);
    }
    T b_norm = sqrt(b_norm2<0?0:b_norm2);
    if(b_norm < 1e-30) b_norm=1;
    int total=0;
    double tti=0;
    // Outer restart loop
    // dev84 GATE: 预条件子自适应门控 —— 窗口(W=6 步)几何平均残差因子 g,
    // g>=0.9 连续 2 窗即永久停用 MG 预条件子(退化为无预条件 FGMRES),
    // 防止弱预条件子纯烧 V-cycle。粗空间有效时门控永不触发。
    const int gW = 6; const double gTH = 0.9;
    int g_win_n=0, g_fail=0; double g_win_ln=0; bool g_prec_off=false;
    for(int restart=0; restart < (max_iter + m -1)/m; restart++){
      // Compute r = b - A x (true residual for restart)
      if(restart>0){
        fine_dslash_op(w, st.x);
        checkCudaErrors(cudaStreamSynchronize(S));
        bistabcg_give_diff2<T><<<set_ptr->gridDim,set_ptr->blockDim,0,S>>>(st.rhs, w, st.r, dv);
        checkCudaErrors(cudaStreamSynchronize(S));
      }
      // beta = ||r||, v0 = r / beta
      T r_norm2=0;
      {
        CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasH, set_ptr->lat_4dim_SC, st.r,1,st.r,1,&dv[_send_tmp_]));
        checkCudaErrors(cudaMemcpyAsync(&host_vals[_send_tmp_],&dv[_send_tmp_],sizeof(LatticeComplex<T>),cudaMemcpyDeviceToHost,S));
        checkCudaErrors(cudaStreamSynchronize(S));
        r_norm2 = host_vals[_send_tmp_].real();
        MPI_Allreduce(MPI_IN_PLACE,&r_norm2,1,mpi_real_type<T>(),MPI_SUM,MPI_COMM_WORLD);
      }
      T r_norm = sqrt(r_norm2<0?0:r_norm2);
      conv_history.push_back(r_norm);
      if(r_norm / b_norm < atol) break;
      T beta = r_norm;
      // v0 = r / beta
      LatticeComplex<T> inv_beta( (beta>1e-30)? (T)1/beta : 0, 0);
      checkCudaErrors(cudaMemcpyAsync(V[0], st.r, st.vec_sz*sizeof(LatticeComplex<T>), cudaMemcpyDeviceToDevice, S));
      CUBLAS_CHECK(_cublasScal<T>(set_ptr->cublasH, set_ptr->lat_4dim_SC, &inv_beta, V[0],1));
      checkCudaErrors(cudaStreamSynchronize(S));
      // s[0]=beta, s[1..]=0
      for(int i=0;i<=m;i++) s[i]=0;
      s[0]=std::complex<T>(beta,0);
      for(int i=0;i<m;i++) for(int j=0;j<m;j++) H[i*m+j]=0; // actually H is (m+1)*m, H[i*m+j] is row i col j
      // Use H as H[row*m + col] where row 0..m, col 0..m-1
      auto Hidx = [&](int row,int col){ return row*m + col; };
      for(int j=0;j<m;j++){
        auto j_t0=std::chrono::high_resolution_clock::now();
        // z_j = M^{-1} v_j   (dev84 GATE: 停用时 M=identity)
        auto prec_t0=std::chrono::high_resolution_clock::now();
        if(!g_prec_off) apply_mg_prec(Z[j], V[j]);
        else checkCudaErrors(cudaMemcpyAsync(Z[j], V[j],
                 st.vec_sz*sizeof(LatticeComplex<T>), cudaMemcpyDeviceToDevice, S));
        auto prec_t1=std::chrono::high_resolution_clock::now();
        prof_vcycle_ms += std::chrono::duration<double,std::milli>(prec_t1-prec_t0).count();
        prof_n_vcycles++;
        // w = A z_j
        fine_dslash_op(w, Z[j]);
        checkCudaErrors(cudaStreamSynchronize(S));
        // Arnoldi: for i=0..j, h_ij = (w, v_i), w -= h_ij * v_i
        for(int i=0;i<=j;i++){
          dot_mpi(w, V[i], _tmp0_, _a_);
          checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
          std::complex<T> h_ij(host_vals[_tmp0_].real(), host_vals[_tmp0_].imag());
          H[Hidx(i,j)] = h_ij;
          LatticeComplex<T> neg_h(-h_ij.real(), -h_ij.imag());
          CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH, set_ptr->lat_4dim_SC, &neg_h, V[i],1,w,1));
          checkCudaErrors(cudaStreamSynchronize(S));
        }
        // h_{j+1,j} = ||w||
        T w_norm2=0;
        {
          CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasH, set_ptr->lat_4dim_SC, w,1,w,1,&dv[_send_tmp_]));
          checkCudaErrors(cudaMemcpyAsync(&host_vals[_send_tmp_],&dv[_send_tmp_],sizeof(LatticeComplex<T>),cudaMemcpyDeviceToHost,S));
          checkCudaErrors(cudaStreamSynchronize(S));
          w_norm2 = host_vals[_send_tmp_].real();
          MPI_Allreduce(MPI_IN_PLACE,&w_norm2,1,mpi_real_type<T>(),MPI_SUM,MPI_COMM_WORLD);
        }
        T w_norm = sqrt(w_norm2<0?0:w_norm2);
        H[Hidx(j+1,j)] = std::complex<T>(w_norm,0);
        if(w_norm > 1e-30){
          LatticeComplex<T> inv_w(w_norm>1e-30? (T)1/w_norm:0,0);
          checkCudaErrors(cudaMemcpyAsync(V[j+1], w, st.vec_sz*sizeof(LatticeComplex<T>), cudaMemcpyDeviceToDevice, S));
          CUBLAS_CHECK(_cublasScal<T>(set_ptr->cublasH, set_ptr->lat_4dim_SC, &inv_w, V[j+1],1));
          checkCudaErrors(cudaStreamSynchronize(S));
        } else {
          checkCudaErrors(cudaMemsetAsync(V[j+1],0,st.vec_sz*sizeof(LatticeComplex<T>),S));
        }
        // Apply previous Givens to new column
        for(int i=0;i<j;i++){
          std::complex<T> temp = cs[i]*H[Hidx(i,j)] + sn[i]*H[Hidx(i+1,j)];
          H[Hidx(i+1,j)] = -std::conj(sn[i])*H[Hidx(i,j)] + cs[i]*H[Hidx(i+1,j)];
          H[Hidx(i,j)] = temp;
        }
        // New Givens for H[j][j] and H[j+1][j]
        T h_jj_abs = std::abs(H[Hidx(j,j)]);
        T h_next_abs = std::abs(H[Hidx(j+1,j)]);
        T r_giv = std::sqrt(h_jj_abs*h_jj_abs + h_next_abs*h_next_abs);
        T cs_j=1; std::complex<T> sn_j(0,0);
        if(r_giv > 1e-30){
          cs_j = h_jj_abs / r_giv;
          std::complex<T> h_jj_norm = (h_jj_abs>1e-30)? H[Hidx(j,j)]/h_jj_abs : std::complex<T>(1,0);
          sn_j = std::conj(H[Hidx(j+1,j)]) * h_jj_norm / r_giv;
          H[Hidx(j,j)] = r_giv * h_jj_norm;
          H[Hidx(j+1,j)] = 0;
        } else {
          cs_j=1; sn_j=0;
        }
        cs[j]=cs_j; sn[j]=sn_j;
        // Apply to s
        std::complex<T> s_j = s[j];
        s[j] = cs_j*s_j + sn_j*s[j+1];
        s[j+1] = -std::conj(sn_j)*s_j + cs_j*s[j+1];
        // Check convergence: |s_{j+1}|/beta < tol ?
        T s_next_abs = std::abs(s[j+1]);
        conv_history.push_back(s_next_abs);
        // dev84 GATE 窗口评估
        {
          double ln_r=std::log((double)std::max(s_next_abs,(T)1e-30));
          if(g_win_n==0){ g_win_ln=ln_r; g_win_n=1; }
          else {
            g_win_n++;
            if(g_win_n>=gW){
              double gfac=std::exp((ln_r-g_win_ln)/g_win_n);
              if(gfac>=gTH){ if(++g_fail>=2 && !g_prec_off){
                  g_prec_off=true;
                  if(rank==0&&verbose)
                    log_write<T>("PYQCU::SOLVER::MULTIGRID::\n GCR-GATE: preconditioner disabled"
                      " (window factor "+std::to_string(gfac)+" >= "+std::to_string(gTH)+")",rank,true);
                } }
              else g_fail=0;
              g_win_n=0;
            }
          }
        }
        auto j_t1=std::chrono::high_resolution_clock::now();
        tti += std::chrono::duration<double>(j_t1-j_t0).count();
        prof_fine_iter_ms += std::chrono::duration<double,std::milli>(j_t1-j_t0).count();
        if(s_next_abs / b_norm < atol){
          // Solve H*y = s (upper triangular, j+1 cols)
          for(int k=j;k>=0;k--){
            std::complex<T> sum_c = s[k];
            for(int l=k+1;l<=j;l++) sum_c -= H[Hidx(k,l)] * y[l];
            y[k] = sum_c / H[Hidx(k,k)];
          }
          // x += Z*y
          for(int k=0;k<=j;k++){
            LatticeComplex<T> yk(y[k].real(), y[k].imag());
            CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH, set_ptr->lat_4dim_SC, &yk, Z[k],1,st.x,1));
          }
          checkCudaErrors(cudaStreamSynchronize(S));
          total += j+1;
          goto fgmres_done;
        }
        if(j==m-1){
          // Restart: solve and update x, break to outer restart
          for(int k=m-1;k>=0;k--){
            std::complex<T> sum_c = s[k];
            for(int l=k+1;l<m;l++) sum_c -= H[Hidx(k,l)] * y[l];
            y[k] = sum_c / H[Hidx(k,k)];
          }
          for(int k=0;k<m;k++){
            LatticeComplex<T> yk(y[k].real(), y[k].imag());
            CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH, set_ptr->lat_4dim_SC, &yk, Z[k],1,st.x,1));
          }
          checkCudaErrors(cudaStreamSynchronize(S));
          total += m;
          break;
        }
      }
    }
    fgmres_done:
    // Cleanup
    for(int i=0;i<=m;i++) cudaFreeAsync(V[i],S);
    for(int i=0;i<m;i++) cudaFreeAsync(Z[i],S);
    cudaFreeAsync(w,S);
    checkCudaErrors(cudaStreamSynchronize(S));
    recover_x_e();
    checkCudaErrors(cudaStreamSynchronize(S));
    auto t1=std::chrono::high_resolution_clock::now();
    solve_time_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
    if(rank==0){
      double avg=total>0?tti/total:0;
      T fn=conv_history.empty()?0:conv_history.back();
      log_write<T>("PYQCU::SOLVER::MULTIGRID::\n FGMRES Performance:",rank,true);
      log_write<T>("PYQCU::SOLVER::MULTIGRID::\n Total FGMRES iters: "+std::to_string(total),rank,true);
      std::ostringstream tm;tm<<"PYQCU::SOLVER::MULTIGRID::\n Total time: "<<std::fixed<<std::setprecision(6)<<(solve_time_ms/1000.0)<<" s";
      log_write<T>(tm.str(),rank,true);
      std::ostringstream am;am<<"PYQCU::SOLVER::MULTIGRID::\n Avg per iter: "<<std::fixed<<std::setprecision(6)<<avg<<" s";
      log_write<T>(am.str(),rank,true);
      std::ostringstream fm;fm<<"PYQCU::SOLVER::MULTIGRID::\n Final res: "<<std::scientific<<fn;
      log_write<T>(fm.str(),rank,true);
      std::ostringstream ch;ch<<"CONVERGENCE_HISTORY: ["; for(size_t j=0;j<conv_history.size();j++){if(j>0)ch<<",";ch<<std::scientific<<conv_history[j];} ch<<"]"; log_write<T>(ch.str(),rank,false);
      std::ostringstream sect; sect<<"PROF_SECTIONS: fine_iter="<<std::fixed<<std::setprecision(1)<<prof_fine_iter_ms<<"ms vcycle="<<prof_vcycle_ms<<"ms n_vcycles="<<prof_n_vcycles; log_write<T>(sect.str(),rank,true);
    }
  }



  // ==================================================================
  // Test wrapper — same as run() but also validates the FULL residual.
  // Rebuilds the full-site solution from the parity-split output and
  // computes |b - D·x|/|b| using the full-site fine dslash.
  // ==================================================================
  void run_test() {
    auto t0=std::chrono::high_resolution_clock::now();run();
    auto t1=std::chrono::high_resolution_clock::now();
    double tm=std::chrono::duration<double,std::milli>(t1-t0).count();
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    cudaStream_t S=set_ptr->stream;
    int full_vol=2*levels[0].vol; size_t full_n=(size_t)_LAT_SC_*full_vol;
    dim3 gf=dim3((full_vol+_BLOCK_SIZE_-1)/_BLOCK_SIZE_), bf=dim3(_BLOCK_SIZE_);
    LatticeComplex<T>*dv=static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
    // Patch _lat_4dim_ (in case a coarse solve left it small)
    LatticeComplex<T> fv((T)full_vol,0.0);
    checkCudaErrors(cudaMemcpyAsync(&dv[_lat_4dim_],&fv,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
    checkCudaErrors(cudaStreamSynchronize(S));
    // ---- Full-site residual on the MASKED [12,X,Y,Z,T] layout ----
    // b_e/b_o are checkerboard-masked channels (size lat_4dim_SC): b_e holds
    // the even sites with odd sites zero and vice versa, so b_full = b_e+b_o.
    // The old parity_to_full/full_to_parity kernels assumed a [..., T/2]-
    // compressed channel layout, silently dropped the even channel
    // (|b_full| measured 313 vs the true |b| = 444, giving |D*x-b|/|b|~1.16),
    // so we assemble b and D·x from the masked components directly.
    {
      cudaStream_t S=set_ptr->stream;
      const int n=(int)set_ptr->lat_4dim_SC;               // one masked channel
      LatticeComplex<T> *xe=static_cast<LatticeComplex<T>*>(fermion_out_eo);
      LatticeComplex<T> *xo=static_cast<LatticeComplex<T>*>(fermion_out_eo)+n;
      // vec0 = D_ee·x_e - κ·H_eo·x_o = (D·x)_e
      CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH,n*_REAL_IMAG_,
          (T*)xe,1,(T*)set_ptr->device_vec0,1));
      clover_dslash_ee.give(set_ptr->device_vec0);
      wilson_dslash.run_eo(set_ptr->device_vec1,xo,gauge);
      checkCudaErrors(cudaStreamSynchronize(S));
      LatticeComplex<T> neg_kap(-kappa_val,0.0);
      CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH,n,&neg_kap,
          set_ptr->device_vec1,1,set_ptr->device_vec0,1));
      checkCudaErrors(cudaStreamSynchronize(S));
      // vec2 = D_oo·x_o - κ·H_oe·x_e = (D·x)_o
      CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH,n*_REAL_IMAG_,
          (T*)xo,1,(T*)set_ptr->device_vec2,1));
      clover_dslash_oo.give(set_ptr->device_vec2);
      wilson_dslash.run_oe(set_ptr->device_vec1,xe,gauge);
      checkCudaErrors(cudaStreamSynchronize(S));
      CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH,n,&neg_kap,
          set_ptr->device_vec1,1,set_ptr->device_vec2,1));
      checkCudaErrors(cudaStreamSynchronize(S));
      // parity_dst = b_e + b_o - (D·x)_e - (D·x)_o = b - D·x (masked full-site)
      LatticeComplex<T> one(1,0), mone(-1,0);
      CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH,n*_REAL_IMAG_,
          (T*)b_e,1,(T*)parity_dst,1));
      CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH,n,&one,(T*)b_o,1,(T*)parity_dst,1));
      CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH,n,&mone,
          (T*)set_ptr->device_vec0,1,(T*)parity_dst,1));
      CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH,n,&mone,
          (T*)set_ptr->device_vec2,1,(T*)parity_dst,1));
      checkCudaErrors(cudaStreamSynchronize(S));
      // |b|² = |b_e|² + |b_o|² (masked channels are disjoint site sets)
      LatticeComplex<T> ht;
      CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasHs[_a_],n,parity_dst,1,parity_dst,1,&dv[_send_tmp_]));
      checkCudaErrors(cudaMemcpyAsync(&ht,&dv[_send_tmp_],sizeof(LatticeComplex<T>),
          cudaMemcpyDeviceToHost,set_ptr->streams[_a_]));
      MPI_Barrier(MPI_COMM_WORLD);checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
      T gn=ht.real();MPI_Allreduce(MPI_IN_PLACE,&gn,1,mpi_real_type<T>(),MPI_SUM,MPI_COMM_WORLD);MPI_Barrier(MPI_COMM_WORLD);
      T dn=sqrt(gn<0?0:gn);
      CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasHs[_a_],n,b_e,1,b_e,1,&dv[_send_tmp_]));
      checkCudaErrors(cudaMemcpyAsync(&ht,&dv[_send_tmp_],sizeof(LatticeComplex<T>),
          cudaMemcpyDeviceToHost,set_ptr->streams[_a_]));
      MPI_Barrier(MPI_COMM_WORLD);checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
      T gbe=ht.real();MPI_Allreduce(MPI_IN_PLACE,&gbe,1,mpi_real_type<T>(),MPI_SUM,MPI_COMM_WORLD);MPI_Barrier(MPI_COMM_WORLD);
      CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasHs[_a_],n,b_o,1,b_o,1,&dv[_send_tmp_]));
      checkCudaErrors(cudaMemcpyAsync(&ht,&dv[_send_tmp_],sizeof(LatticeComplex<T>),
          cudaMemcpyDeviceToHost,set_ptr->streams[_a_]));
      MPI_Barrier(MPI_COMM_WORLD);checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
      T gbo=ht.real();MPI_Allreduce(MPI_IN_PLACE,&gbo,1,mpi_real_type<T>(),MPI_SUM,MPI_COMM_WORLD);MPI_Barrier(MPI_COMM_WORLD);
      T nb=sqrt(gbe+gbo);
      T rd=(nb>(T)1e-30)?dn/nb:dn;
      if(rank==0){
        printf("=== MULTIGRID SOLVER REPORT ===\nTotal time: %.3f ms (%.3f s)\n",tm,tm/1000.);
        printf("Solve time: %.3f ms\n",solve_time_ms);
        printf("Levels: %d, Restart: %d\n",num_levels,num_restart);
        printf("Convergence history entries: %zu\n",conv_history.size());
        if(!conv_history.empty()){printf("Initial residual: %.6e\n",conv_history[0]);
          printf("Final residual:   %.6e\n",conv_history.back());}
        printf("Relative residual |D*x - b|/|b|: %.6e\n",rd);
        printf("VERIFY: |b_full|=%.6e |Dx-b|=%.6e (|b_e|^2=%.4e |b_o|^2=%.4e)\n",nb,dn,gbe,gbo);
      }
    }
    set_ptr->err=cudaGetLastError();checkCudaErrors(set_ptr->err);
  }

  // ==================================================================
  // Cleanup
  // ==================================================================
  void end() {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

    auto F=[&](void*&p){if(p){cudaFreeAsync(p,set_ptr->stream);p=nullptr;}};
    F(b__o);F(r0);F(rt0);F(p0);F(v0);F(s0);F(t0);
    F(r_full);F(e_odd_buf);
    F(full_x);F(full_rhs);F(full_r);F(full_rt);
    F(full_p);F(full_v);F(full_s);F(full_t);
    F(parity_tmp);F(parity_dst);F(corr_scratch);F(coarse_partials);
    if(check_host!=nullptr){cudaFreeHost(check_host);check_host=nullptr;check_dev=nullptr;}
    for(int i=1;i<num_levels;i++)levels[i].free_all(set_ptr->stream);
    // dev84: free captured coarse-solve graphs
    for(int l=0;l<8;l++){
      if(graph_exec[l]!=nullptr){cudaGraphExecDestroy(graph_exec[l]);graph_exec[l]=nullptr;}
      if(graph[l]!=nullptr){cudaGraphDestroy(graph[l]);graph[l]=nullptr;}
      graph_ready[l]=false;
    }
    delete[] levels;levels=nullptr;delete[] null_vecs;null_vecs=nullptr;
    delete[] hop_nn;hop_nn=nullptr;delete[] hop_diag;hop_diag=nullptr;
    delete[] sit_packed;sit_packed=nullptr;
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
};

} // namespace qcu
#endif
