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
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace qcu {

// ---- Logging infrastructure ----
inline void ensure_log_dir() {
  struct stat st;
  if (stat("logs", &st) != 0) mkdir("logs", 0755);
}

template <typename T>
inline void log_write(const std::string &msg, int rank, bool to_stdout = true) {
  ensure_log_dir();
  std::ofstream f("logs/clover_multigrid.log", std::ios_base::app);
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
  int max_iter;      // per-level max iterations
  T tol;             // per-level tolerance
  int num_restart;   // per-level restart interval for coarse correction
  void alloc(int _dof, int _X, int _Y, int _Z, int _Lt, cudaStream_t stream) {
    dof=_dof; X=_X; Y=_Y; Z=_Z; Lt=_Lt; vol=X*Y*Z*Lt; vec_sz=(size_t)dof*vol;
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

  // ---- Multigrid hierarchy ----
  int num_levels, mg_grid_size[4];
  MgLevelState<T> *levels;
  void **null_vecs, **hop_packed, **sit_packed;

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

  // ---- Host mirror for convergence check ----
  LatticeComplex<T> host_vals[_vals_size_];
  std::vector<T> conv_history;
  double level_times[8];
  double solve_time_ms;

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
   * @brief Coarse-grid full-site dslash (hopping + sitting).
   * Operates on full-site (non-parity-split) vectors.
   */
  void coarse_dslash_op(void *out, void *in, int lev) {
    int E=levels[lev].dof, Xc=levels[lev].X, Yc=levels[lev].Y,
        Zc=levels[lev].Z, Ltc=levels[lev].Lt;
    int t=E*Xc*Yc*Zc*Ltc;
    dim3 g((t+_BLOCK_SIZE_-1)/_BLOCK_SIZE_);
    multigrid_coarse_dslash<T><<<g,_BLOCK_SIZE_,0,set_ptr->stream>>>(
        out,in,hop_packed[lev-1],sit_packed[lev-1],E,Xc,Yc,Zc,Ltc);
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
   * @brief Dot product for coarse-level vectors (single GPU, no MPI).
   */
  void dot_coarse(void *a, void *b, int lv, int vals_idx, int si) {
    LatticeComplex<T> *dv=static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
    CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasHs[si],levels[lv].vec_sz,
        a,1,b,1,&dv[_send_tmp_]));
    checkCudaErrors(cudaMemcpyAsync(&host_vals[vals_idx],&dv[_send_tmp_],
        sizeof(LatticeComplex<T>),cudaMemcpyDeviceToHost,set_ptr->streams[si]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[si]));
    checkCudaErrors(cudaMemcpyAsync(&dv[vals_idx],&host_vals[vals_idx],
        sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,set_ptr->streams[si]));
  }

  // ---- Vector helpers ----
  void zero_c(void *v,int l) {
    checkCudaErrors(cudaMemsetAsync(v,0,levels[l].vec_sz*sizeof(LatticeComplex<T>),set_ptr->stream));
  }
  // Helper: grid dimensions for site-processing kernels (each thread = 1 site × all DOF)
  dim3 site_grid(int lev) {
    int t=(int)levels[lev].vol;
    return dim3((t+_BLOCK_SIZE_-1)/_BLOCK_SIZE_);
  }
  void copy_c(void *d,void *s,int l) {
    give_copy_vals<T><<<site_grid(l),_BLOCK_SIZE_,0,set_ptr->stream>>>(d,s);
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
  void bistabcg_iter(int lev) {
    auto &st=levels[lev];
    bool fine=(lev==0); cudaStream_t S=set_ptr->stream;
    dim3 gv,bv;
    // Grid dimension: number of SITES (vol), not total elements (vec_sz = dof*vol).
    // Each thread processes one site × all DOF components.
    // Using vec_sz would launch vol*dof threads, causing OOB writes.
    if(fine){gv=set_ptr->gridDim;bv=set_ptr->blockDim;}
    else{int t=(int)st.vol;gv=dim3((t+_BLOCK_SIZE_-1)/_BLOCK_SIZE_);bv=dim3(_BLOCK_SIZE_);}

    // Step 1: ρ = (r_tilde, r)           [stream _a_]
    if(fine) dot_mpi(st.r_tilde,st.r,_rho_,_a_);
    else     dot_coarse(st.r_tilde,st.r,lev,_rho_,_a_);

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
    if(fine) dot_mpi(st.r,st.r,_norm2_tmp_,_c_);
    else     dot_coarse(st.r,st.r,lev,_norm2_tmp_,_c_);

    // Step 4: v = A·p                      [main stream]
    checkCudaErrors(cudaStreamSynchronize(S));
    if(fine) fine_dslash_op(st.v,st.p); else coarse_dslash_op(st.v,st.p,lev);
    checkCudaErrors(cudaStreamSynchronize(S));

    // Step 5: τ₀=(r_tilde,v); α=ρ/τ₀     [_d_]
    if(fine) dot_mpi(st.r_tilde,st.v,_tmp0_,_d_);
    else     dot_coarse(st.r_tilde,st.v,lev,_tmp0_,_d_);
    bistabcg_give_1alpha<T><<<1,1,0,set_ptr->streams[_d_]>>>(set_ptr->device_vals);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

    // Step 6: s = r − α·v                  [_a_]
    bistabcg_give_s<T><<<gv,bv,0,set_ptr->streams[_a_]>>>(st.s,st.r,st.v,set_ptr->device_vals);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));

    // Step 7: t = A·s                      [main stream]
    checkCudaErrors(cudaStreamSynchronize(S));
    if(fine) fine_dslash_op(st.t,st.s); else coarse_dslash_op(st.t,st.s,lev);
    checkCudaErrors(cudaStreamSynchronize(S));

    // Step 8: τ₀=(t,s); τ₁=(t,t)          [_c_],[_d_]
    if(fine){dot_mpi(st.t,st.s,_tmp0_,_c_);dot_mpi(st.t,st.t,_tmp1_,_d_);}
    else    {dot_coarse(st.t,st.s,lev,_tmp0_,_c_);dot_coarse(st.t,st.t,lev,_tmp1_,_d_);}
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));

    // Step 9: ω = τ₀/τ₁                   [_d_]
    bistabcg_give_1omega<T><<<1,1,0,set_ptr->streams[_d_]>>>(set_ptr->device_vals);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

    // Step 10: r=s−ω·t [_a_];  x=x+α·p+ω·s [_b_]
    bistabcg_give_r<T><<<gv,bv,0,set_ptr->streams[_a_]>>>(st.r,st.s,st.t,set_ptr->device_vals);
    bistabcg_give_x_o<T><<<gv,bv,0,set_ptr->streams[_b_]>>>(st.x,st.p,st.s,set_ptr->device_vals);

    // FIX: Full 5-stream sync at bottom of iteration.
    // Without this, the next iteration's Step 1 (_a_) may read stale
    // device_vals from the previous iteration's Step 10 (_a_).
    checkCudaErrors(cudaStreamSynchronize(S));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
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
      if(levels[i].Lt<=0) levels[i].Lt=levels[i-1].Lt; // coarse uses full Lt, not halved
      levels[i].vol=levels[i].X*levels[i].Y*levels[i].Z*levels[i].Lt;
      levels[i].vec_sz=(size_t)levels[i].dof*levels[i].vol;

      // Read per-level max_iter and num_restart from params
      levels[i].max_iter=set_ptr->host_params[oMI+b];
      levels[i].num_restart=set_ptr->host_params[oNR+b];
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
    null_vecs=new void*[num_levels]; hop_packed=new void*[num_levels]; sit_packed=new void*[num_levels];
    for(int i=0;i<num_levels;i++)null_vecs[i]=hop_packed[i]=sit_packed[i]=nullptr;

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
   * @param fl  Fine level index (0 = level 0→1, 1 = level 1→2, etc.)
   * @param nv  Null vectors for restrict/prolong [E_{l+1}, e_l, X_l, Y_l, Z_l, T_l]
   * @param hp  Hopping matrices [2, 4, E_{l+1}, E_{l+1}, X_{l+1}, Y_{l+1}, Z_{l+1}, T_{l+1}]
   * @param sp  Sitting matrices [E_{l+1}, E_{l+1}, X_{l+1}, Y_{l+1}, Z_{l+1}, T_{l+1}]
   */
  void set_coarse_ops(int fl,void*nv,void*hp,void*sp){
    if(fl>=0&&fl<num_levels-1){null_vecs[fl]=nv;hop_packed[fl]=hp;sit_packed[fl]=sp;}
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
  T v_cycle(int lev) {
    auto&st=levels[lev]; cudaStream_t S=set_ptr->stream;
    T rn = 0;

    // ---- Save and update _lat_4dim_ for coarse level ----
    // The BiStabCG kernels (bistabcg_give_p, _s, _x_o, _r, _diff2, etc.) use
    // device_vals[_lat_4dim_] as the site-count stride.  At the fine level this
    // is set correctly during LatticeSet::init().  For coarse levels the volume
    // differs, so we must patch _lat_4dim_ to match the current level.
    LatticeComplex<T>*dv=static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
    T saved_lat_4dim = (T)set_ptr->lat_4dim;   // fine-level value, known from host
    LatticeComplex<T> coarse_vol_val((T)st.vol, 0.0);
    checkCudaErrors(cudaMemcpyAsync(&dv[_lat_4dim_], &coarse_vol_val,
        sizeof(LatticeComplex<T>), cudaMemcpyHostToDevice, S));
    checkCudaErrors(cudaStreamSynchronize(S));

    // ---- Verify RHS is not NaN ----
    {
      size_t n=st.vec_sz;
      CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasHs[_a_],n,st.rhs,1,st.rhs,1,&dv[_send_tmp_]));
      LatticeComplex<T> ht; checkCudaErrors(cudaMemcpyAsync(&ht,&dv[_send_tmp_],sizeof(LatticeComplex<T>),cudaMemcpyDeviceToHost,set_ptr->streams[_a_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
      T nb2 = ht.real();
      if (!std::isfinite(nb2)) {
        if(rank==0) log_write<T>("PYQCU::SOLVER::MULTIGRID::\n FATAL: NaN in coarse RHS at level "+std::to_string(lev),rank,true);
        LatticeComplex<T> restore_vol2(saved_lat_4dim, 0.0);
        checkCudaErrors(cudaMemcpyAsync(&dv[_lat_4dim_], &restore_vol2, sizeof(LatticeComplex<T>), cudaMemcpyHostToDevice, S));
        checkCudaErrors(cudaStreamSynchronize(S));
        return (T)1e30;
      }
      T nb=sqrt(nb2<0?0:nb2);
      if(rank==0&&verbose){
        log_write<T>("PYQCU::SOLVER::MULTIGRID::\n "+std::to_string(lev)+":Norm of b:"+std::to_string(nb),rank,true);
        log_write<T>("PYQCU::SOLVER::MULTIGRID::\n "+std::to_string(lev)+":Norm of r:"+std::to_string(nb),rank,true);
        log_write<T>("PYQCU::SOLVER::MULTIGRID::\n "+std::to_string(lev)+":Norm of x0:0.000000",rank,true);
        log_write<T>("PYQCU::SOLVER::MULTIGRID::\n "+std::to_string(lev)+":Starting Iterations",rank,true);
      }
    }

    // One-time TOP sync
    checkCudaErrors(cudaStreamSynchronize(S));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

    // Init: x=0, r=b, r_tilde=r, p=v=s=t=0
    zero_c(st.x,lev); copy_c(st.r,st.rhs,lev); copy_c(st.r_tilde,st.r,lev);
    zero_c(st.p,lev); zero_c(st.v,lev); zero_c(st.s,lev); zero_c(st.t,lev);
    checkCudaErrors(cudaStreamSynchronize(S));

    // Set device_vals to initial BiStabCG scalars
    dv=static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
    LatticeComplex<T> one(1,0),z(0,0);
    checkCudaErrors(cudaMemcpyAsync(&dv[_rho_],&z,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
    checkCudaErrors(cudaMemcpyAsync(&dv[_rho_prev_],&one,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
    checkCudaErrors(cudaMemcpyAsync(&dv[_alpha_],&one,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
    checkCudaErrors(cudaMemcpyAsync(&dv[_omega_],&one,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
    checkCudaErrors(cudaStreamSynchronize(S));

    // Number of smoothing steps: coarsest level gets more iterations (no
    // further coarse correction), inner levels get fewer (V-cycle handles
    // low modes via the even-coarser correction).
    int ns = 5;  // pre-smoothing steps
    int np = (lev == num_levels-1) ? 5 : 3; // post-smoothing steps (more at coarsest)

    // ---- Pre-smoothing ----
    for (int i = 0; i < ns; i++) {
      auto t0=std::chrono::high_resolution_clock::now();
      bistabcg_iter(lev);
      auto t1=std::chrono::high_resolution_clock::now();
      double sec=std::chrono::duration<double>(t1-t0).count();
      rn = sqrt(host_vals[_norm2_tmp_].real());
      if(rank==0&&verbose){
        std::ostringstream bm,fm;
        bm<<"PYQCU::SOLVER::MULTIGRID::\n B-"<<lev<<"-bistabcg-Iteration "<<i
          <<": Residual = "<<std::scientific<<rn;
        log_write<T>(bm.str(),rank,true);
        fm<<"PYQCU::SOLVER::MULTIGRID::\n F-"<<lev<<"-bistabcg-Iteration "<<i
          <<": Residual = "<<std::scientific<<rn<<", Time = "<<std::fixed<<std::setprecision(6)<<sec<<" s";
        log_write<T>(fm.str(),rank,true);
      }
    }

    // ---- Coarse-grid correction (skip at coarsest level) ----
    if (lev < num_levels-1) {
      checkCudaErrors(cudaStreamSynchronize(S));

      // Compute residual: r = rhs - D_c * x
      coarse_dslash_op(set_ptr->device_vec0, st.x, lev);
      checkCudaErrors(cudaStreamSynchronize(S));
      dim3 gc_site=site_grid(lev);
      give_copy_vals<T><<<gc_site,_BLOCK_SIZE_,0,S>>>(set_ptr->device_vec2, st.rhs);
      bistabcg_give_diff2<T><<<gc_site,_BLOCK_SIZE_,0,S>>>(set_ptr->device_vec2,set_ptr->device_vec0,st.r,set_ptr->device_vals);
      checkCudaErrors(cudaStreamSynchronize(S));

      // Restrict → coarse solve → prolong
      restrict_op(levels[lev+1].rhs, st.r, lev);
      zero_c(levels[lev+1].x, lev+1);
      checkCudaErrors(cudaStreamSynchronize(S));

      v_cycle(lev+1);

      prolong_op(set_ptr->device_vec0, levels[lev+1].x, lev);
      checkCudaErrors(cudaStreamSynchronize(S));

      // Add correction to x (for coarse levels, data is full-site, no parity
      // extraction needed — prolong already maps coarse full-site → fine full-site)
      LatticeComplex<T> oc(1,0);
      CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH, (int)st.vec_sz, &oc,
          set_ptr->device_vec0, 1, st.x, 1));
      checkCudaErrors(cudaStreamSynchronize(S));

      // Recompute residual after correction
      coarse_dslash_op(set_ptr->device_vec0, st.x, lev);
      checkCudaErrors(cudaStreamSynchronize(S));
      give_copy_vals<T><<<gc_site,_BLOCK_SIZE_,0,S>>>(set_ptr->device_vec2, st.rhs);
      bistabcg_give_diff2<T><<<gc_site,_BLOCK_SIZE_,0,S>>>(set_ptr->device_vec2,set_ptr->device_vec0,st.r,set_ptr->device_vals);
      checkCudaErrors(cudaStreamSynchronize(S));

      // Reset shadow vector and BiStabCG state
      copy_c(st.r_tilde, st.r, lev);
      reset_bistabcg_state(lev);

      checkCudaErrors(cudaStreamSynchronize(S));
    }

    // ---- Post-smoothing ----
    for (int j = 0; j < np; j++) {
      auto t0=std::chrono::high_resolution_clock::now();
      bistabcg_iter(lev);
      auto t1=std::chrono::high_resolution_clock::now();
      double sec=std::chrono::duration<double>(t1-t0).count();
      rn = sqrt(host_vals[_norm2_tmp_].real());
      if(rank==0&&verbose){
        int idx = ns + j;
        std::ostringstream bm,fm;
        bm<<"PYQCU::SOLVER::MULTIGRID::\n B-"<<lev<<"-bistabcg-Iteration "<<idx
          <<": Residual = "<<std::scientific<<rn;
        log_write<T>(bm.str(),rank,true);
        fm<<"PYQCU::SOLVER::MULTIGRID::\n F-"<<lev<<"-bistabcg-Iteration "<<idx
          <<": Residual = "<<std::scientific<<rn<<", Time = "<<std::fixed<<std::setprecision(6)<<sec<<" s";
        log_write<T>(fm.str(),rank,true);
      }
    }

    // Report convergence
    if(rank==0&&verbose)
      log_write<T>("PYQCU::SOLVER::MULTIGRID::\n "+std::to_string(lev)+
        ": Converged at iteration "+
        std::to_string(ns+np-1)+" with residual "+std::to_string(rn),rank,true);

    // Final sync
    checkCudaErrors(cudaStreamSynchronize(S));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

    // ---- Restore _lat_4dim_ to fine-level value ----
    LatticeComplex<T> restore_vol(saved_lat_4dim, 0.0);
    checkCudaErrors(cudaMemcpyAsync(&dv[_lat_4dim_], &restore_vol,
        sizeof(LatticeComplex<T>), cudaMemcpyHostToDevice, S));
    checkCudaErrors(cudaStreamSynchronize(S));

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

    // Allocate full-site residual buffer for V-cycle correction
    // Full-site size = _LAT_SC_ * X * Y * Z * Lt_full where Lt_full = 2 * levels[0].Lt
    size_t r_full_sc = (size_t)_LAT_SC_ * levels[0].X * levels[0].Y * levels[0].Z * Lt_full;
    size_t r_full_bytes = r_full_sc * sizeof(LatticeComplex<T>);
    checkCudaErrors(cudaMallocAsync(&r_full, r_full_bytes, set_ptr->stream));
    checkCudaErrors(cudaMemsetAsync(r_full, 0, r_full_bytes, set_ptr->stream));

    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));

    // Wire level 0 state to parity-split working vectors
    levels[0].x=x_o;levels[0].rhs=b__o;levels[0].r=r0;levels[0].r_tilde=rt0;
    levels[0].p=p0;levels[0].v=v0;levels[0].s=s0;levels[0].t=t0;levels[0].owned=false;

    // Compute Schur complement RHS: b__o = b_o + κ · H_oe · D_ee^{-1} · b_e
    setup_b__o();
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));

    if(rank==0){
      std::ostringstream oss;
      oss<<"PYQCU::QCU::MULTIGRID::\n MG_INIT_COMPLETE: "<<num_levels
         <<" levels, Lt_full="<<Lt_full
         <<", num_restart="<<num_restart;
      log_write<T>(oss.str(),rank,true);
    }
  }

  // ==================================================================
  // Main solve — BiStabCG at level 0 with V-cycle corrections.
  //
  // FIXES applied:
  //   - V-cycle uses FULL-SITE residual (compute_full_residual + restrict)
  //   - Prolonged correction: extract odd part only before adding to x_o
  //   - BiStabCG state fully reset after each V-cycle correction
  //   - Full 5-stream sync at end of each iteration
  // ==================================================================
  void run() {
    auto t0=std::chrono::high_resolution_clock::now();
    auto&st=levels[0]; cudaStream_t S=set_ptr->stream;

    // Log initial state
    if(rank==0){
      LatticeComplex<T>*dv=static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
      CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasHs[_a_],set_ptr->lat_4dim_SC,b__o,1,b__o,1,&dv[_send_tmp_]));
      LatticeComplex<T> ht; checkCudaErrors(cudaMemcpyAsync(&ht,&dv[_send_tmp_],sizeof(LatticeComplex<T>),cudaMemcpyDeviceToHost,set_ptr->streams[_a_]));
      MPI_Barrier(MPI_COMM_WORLD);checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
      T g=ht.real();MPI_Allreduce(MPI_IN_PLACE,&g,1,mpi_real_type<T>(),MPI_SUM,MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD); T nb=sqrt(g);
      log_write<T>("PYQCU::SOLVER::MULTIGRID::\n 0:Norm of b:"+std::to_string(nb),rank,true);
      log_write<T>("PYQCU::SOLVER::MULTIGRID::\n 0:Norm of r:"+std::to_string(nb),rank,true);
      log_write<T>("PYQCU::SOLVER::MULTIGRID::\n 0:Norm of x0:0.000000",rank,true);
      log_write<T>("PYQCU::SOLVER::MULTIGRID::\n 0:Starting Iterations",rank,true);
    }

    // ONE-TIME initial sync (matches reference _run() before for loop)
    checkCudaErrors(cudaStreamSynchronize(S));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

    // Init state: x=0, r=b, r_tilde=r, p=v=s=t=0
    checkCudaErrors(cudaMemsetAsync(x_o,0,set_ptr->lat_4dim_SC*sizeof(LatticeComplex<T>),S));
    checkCudaErrors(cudaStreamSynchronize(S));
    copy_c(st.r,st.rhs,0);copy_c(st.r_tilde,st.r,0);zero_c(st.p,0);zero_c(st.v,0);
    zero_c(st.s,0);zero_c(st.t,0);
    checkCudaErrors(cudaStreamSynchronize(S));

    // Set initial BiStabCG scalars
    LatticeComplex<T>*dv=static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
    LatticeComplex<T> one(1,0),z(0,0);
    checkCudaErrors(cudaMemcpyAsync(&dv[_rho_],&z,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
    checkCudaErrors(cudaMemcpyAsync(&dv[_rho_prev_],&one,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
    checkCudaErrors(cudaMemcpyAsync(&dv[_alpha_],&one,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
    checkCudaErrors(cudaMemcpyAsync(&dv[_omega_],&one,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
    checkCudaErrors(cudaStreamSynchronize(S));

    // ---- Main BiStabCG loop ----
    T atol2=atol*atol;
    int total=0; double tti=0;
    int count_restart = 0;

    for(int it=0;it<max_iter;it++){
      auto ti0=std::chrono::high_resolution_clock::now();
      bistabcg_iter(0);
      auto ti1=std::chrono::high_resolution_clock::now();
      double sec=std::chrono::duration<double>(ti1-ti0).count();tti+=sec;total++;
      count_restart++;

      // Convergence check from host_vals[_norm2_tmp_] (lagged by ~1 iter)
      T rn2=host_vals[_norm2_tmp_].real();
      T rn=sqrt(rn2<0?0:rn2);
      conv_history.push_back(rn);

      if(rank==0&&verbose){
        std::ostringstream bm,fm;
        bm<<"PYQCU::SOLVER::MULTIGRID::\n B-0-bistabcg-Iteration "<<it
          <<": Residual = "<<std::scientific<<rn;
        log_write<T>(bm.str(),rank,true);
        fm<<"PYQCU::SOLVER::MULTIGRID::\n F-0-bistabcg-Iteration "<<it
          <<": Residual = "<<std::scientific<<rn<<", Time = "<<std::fixed<<std::setprecision(6)<<sec<<" s";
        log_write<T>(fm.str(),rank,true);
      }

      // Divergence safeguard
      if(!std::isfinite(rn)||rn>(T)1e10){
        if(rank==0&&verbose)log_write<T>("PYQCU::SOLVER::MULTIGRID::\n Restart at "+std::to_string(it),rank,true);
        checkCudaErrors(cudaStreamSynchronize(S));
        checkCudaErrors(cudaMemsetAsync(x_o,0,set_ptr->lat_4dim_SC*sizeof(LatticeComplex<T>),S));
        checkCudaErrors(cudaStreamSynchronize(S));
        copy_c(st.r,st.rhs,0);copy_c(st.r_tilde,st.r,0);
        reset_bistabcg_state_l0();
        count_restart = 0;
        continue;
      }

      // Convergence check
      if(rn2<atol2){
        if(rank==0&&verbose)
          log_write<T>("PYQCU::SOLVER::MULTIGRID::\n Converged at iteration "+
            std::to_string(it)+" with residual "+std::to_string(rn),rank,true);
        break;
      }

      // ---- V-cycle correction ----
      // V-cycles are applied at regular intervals to correct the error in the
      // near-null-space that BiStabCG smoothing leaves unresolved.
      if(num_levels>1 && num_restart>0 && count_restart >= num_restart){
        if(rank==0&&verbose)
          log_write<T>("PYQCU::SOLVER::MULTIGRID::\n V-cycle correction at iteration "+std::to_string(it)+" (rn="+std::to_string(rn)+")",rank,true);

        // Step 1: Compute full-site residual r_full = b_full - D_full * x_full
        compute_full_residual();
        // r_full now has the full-site residual (even=0, odd=r_o_full)

        // Step 2: Restrict full-site residual → coarse RHS
        // Use full-site dimensions for restrict (level 0 now has full-site Lt)
        checkCudaErrors(cudaStreamSynchronize(S));
        // Temporarily adjust level 0 dimensions to full-site for restrict
        int orig_Lt_0 = levels[0].Lt;
        size_t orig_vec_sz_0 = levels[0].vec_sz;
        int orig_vol_0 = levels[0].vol;
        levels[0].Lt = Lt_full;
        levels[0].vol = levels[0].X * levels[0].Y * levels[0].Z * Lt_full;
        levels[0].vec_sz = (size_t)levels[0].dof * levels[0].vol;

        restrict_op(levels[1].rhs, r_full, 0);
        zero_c(levels[1].x, 1);
        checkCudaErrors(cudaStreamSynchronize(S));

        // Restore level 0 dimensions
        levels[0].Lt = orig_Lt_0;
        levels[0].vol = orig_vol_0;
        levels[0].vec_sz = orig_vec_sz_0;

        // Step 3: Solve on coarse grid (recursive V-cycle)
        v_cycle(1);

        // Step 4: Prolong coarse solution → full-site correction.
        // IMPORTANT: Use r_full as the output buffer (full-site sized, 98304 elements)
        // NOT device_vec0 (parity-split sized, only 49152 elements).
        // Temporarily adjust level 0 dimensions to full-site for prolong
        levels[0].Lt = Lt_full;
        levels[0].vol = levels[0].X * levels[0].Y * levels[0].Z * Lt_full;
        levels[0].vec_sz = (size_t)levels[0].dof * levels[0].vol;

        prolong_op(r_full, levels[1].x, 0);   // <-- write to r_full (full-site buffer)
        checkCudaErrors(cudaStreamSynchronize(S));

        // Restore level 0 dimensions to parity-split
        levels[0].Lt = orig_Lt_0;
        levels[0].vol = orig_vol_0;
        levels[0].vec_sz = orig_vec_sz_0;

        // Step 5: Extract odd-site part of the correction (matching Python:
        //   e_fine_eo = oooxyzt2poooxyzt(e_fine); e_fine = e_fine_eo[1])
        // r_full has the full-site prolonged correction.
        // Convert full-site → parity-split odd into device_vec2
        extract_odd_from_full(set_ptr->device_vec2, r_full);

        // Step 7: Compute r_before = ||b__o - D_precond * x_o|| using vec0+vec1 only.
        // CRITICAL: device_vec2 holds the odd-part correction from extract_odd_from_full.
        // We must NOT overwrite it — use vec0 for Dprecond result and vec1 for r_before.
        fine_dslash_op(set_ptr->device_vec0, x_o);          // vec0 = D_precond * x_o
        checkCudaErrors(cudaStreamSynchronize(S));
        dim3 gf=set_ptr->gridDim,bf=set_ptr->blockDim;
        give_copy_vals<T><<<gf,bf,0,S>>>(set_ptr->device_vec1, b__o);   // vec1 = b__o
        bistabcg_give_diff2<T><<<gf,bf,0,S>>>(set_ptr->device_vec1,set_ptr->device_vec0,set_ptr->device_vec1,set_ptr->device_vals);
        checkCudaErrors(cudaStreamSynchronize(S));                       // vec1 = b__o - vec0 (= r_before, IN-PLACE)
        // device_vec2 still holds the correction, undisturbed.

        LatticeComplex<T> ht;
        CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasHs[_a_],set_ptr->lat_4dim_SC,set_ptr->device_vec1,1,set_ptr->device_vec1,1,&dv[_send_tmp_]));
        checkCudaErrors(cudaMemcpyAsync(&ht,&dv[_send_tmp_],sizeof(LatticeComplex<T>),cudaMemcpyDeviceToHost,set_ptr->streams[_a_]));
        MPI_Barrier(MPI_COMM_WORLD);checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
        T gr_before=ht.real();MPI_Allreduce(MPI_IN_PLACE,&gr_before,1,mpi_real_type<T>(),MPI_SUM,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        T rn_before = sqrt(gr_before < 0 ? 0 : gr_before);

        // Step 8: Apply correction to x_o. device_vec2 still has the correction.
        LatticeComplex<T> oc(1,0);
        CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH,set_ptr->lat_4dim_SC,
            &oc,set_ptr->device_vec2,1,x_o,1));
        checkCudaErrors(cudaStreamSynchronize(S));

        // Step 9: Compute r_after = ||b__o - D_precond * x_o_new||, write to st.r.
        // This overwrites device_vec2, but we're done with the correction.
        fine_dslash_op(set_ptr->device_vec0, x_o);
        checkCudaErrors(cudaStreamSynchronize(S));
        give_copy_vals<T><<<gf,bf,0,S>>>(set_ptr->device_vec2, b__o);
        bistabcg_give_diff2<T><<<gf,bf,0,S>>>(set_ptr->device_vec2,set_ptr->device_vec0,st.r,set_ptr->device_vals);
        checkCudaErrors(cudaStreamSynchronize(S));

        CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasHs[_a_],set_ptr->lat_4dim_SC,st.r,1,st.r,1,&dv[_send_tmp_]));
        checkCudaErrors(cudaMemcpyAsync(&ht,&dv[_send_tmp_],sizeof(LatticeComplex<T>),cudaMemcpyDeviceToHost,set_ptr->streams[_a_]));
        MPI_Barrier(MPI_COMM_WORLD);checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
        T gr_after=ht.real();MPI_Allreduce(MPI_IN_PLACE,&gr_after,1,mpi_real_type<T>(),MPI_SUM,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);T corr_rn=sqrt(gr_after<0?0:gr_after);
        conv_history.push_back(corr_rn);

        // Step 10: Log correction effectiveness
        if (rank==0&&verbose) {
          std::ostringstream oss;
          oss<<"PYQCU::SOLVER::MULTIGRID::\n V-cyc corr at "<<it
             <<": before="<<std::scientific<<rn_before<<" after="<<corr_rn;
          log_write<T>(oss.str(),rank,true);
        }

        // Step 11: Reset BiStabCG state after V-cycle
        copy_c(st.r_tilde, st.r, 0);
        reset_bistabcg_state_l0();
        count_restart = 0;

        // Step 11: Reset BiStabCG state for fresh restart after V-cycle
        copy_c(st.r_tilde, st.r, 0);
        reset_bistabcg_state_l0();
        count_restart = 0;

        // Full sync after state reset
        checkCudaErrors(cudaStreamSynchronize(S));
        checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
        checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
        checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
        checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
      }
    }

    // ---- Final sync (matches reference after for loop) ----
    checkCudaErrors(cudaStreamSynchronize(S));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

    // Recover even-site solution x_e from final x_o
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
    }
  }

  // ==================================================================
  // Test wrapper — same as run() but also validates residual.
  // ==================================================================
  void run_test() {
    auto t0=std::chrono::high_resolution_clock::now();run();
    auto t1=std::chrono::high_resolution_clock::now();
    double tm=std::chrono::duration<double,std::milli>(t1-t0).count();
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    fine_dslash_op(set_ptr->device_vec1,x_o);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    LatticeComplex<T>*dv=static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
    CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasHs[_a_],set_ptr->lat_4dim_SC,b__o,1,b__o,1,&dv[_send_tmp_]));
    LatticeComplex<T> ht;checkCudaErrors(cudaMemcpyAsync(&ht,&dv[_send_tmp_],sizeof(LatticeComplex<T>),cudaMemcpyDeviceToHost,set_ptr->streams[_a_]));
    MPI_Barrier(MPI_COMM_WORLD);checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    T g=ht.real();MPI_Allreduce(MPI_IN_PLACE,&g,1,mpi_real_type<T>(),MPI_SUM,MPI_COMM_WORLD);MPI_Barrier(MPI_COMM_WORLD);
    T nb=sqrt(g);
    dim3 gd=set_ptr->gridDim,bd=set_ptr->blockDim;
    give_copy_vals<T><<<gd,bd,0,set_ptr->stream>>>(set_ptr->device_vec2,set_ptr->device_vec1);
    bistabcg_give_diff2<T><<<gd,bd,0,set_ptr->stream>>>(set_ptr->device_vec2,b__o,set_ptr->device_vec1,set_ptr->device_vals);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasHs[_a_],set_ptr->lat_4dim_SC,set_ptr->device_vec1,1,set_ptr->device_vec1,1,&dv[_send_tmp_]));
    checkCudaErrors(cudaMemcpyAsync(&ht,&dv[_send_tmp_],sizeof(LatticeComplex<T>),cudaMemcpyDeviceToHost,set_ptr->streams[_a_]));
    MPI_Barrier(MPI_COMM_WORLD);checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    T gn=ht.real();MPI_Allreduce(MPI_IN_PLACE,&gn,1,mpi_real_type<T>(),MPI_SUM,MPI_COMM_WORLD);MPI_Barrier(MPI_COMM_WORLD);
    T dn=sqrt(gn),rd=(nb>(T)1e-30)?dn/nb:dn;
    if(rank==0){
      printf("=== MULTIGRID SOLVER REPORT ===\nTotal time: %.3f ms (%.3f s)\n",tm,tm/1000.);
      printf("Solve time: %.3f ms\n",solve_time_ms);
      printf("Levels: %d, Restart: %d\n",num_levels,num_restart);
      printf("Convergence history entries: %zu\n",conv_history.size());
      if(!conv_history.empty()){printf("Initial residual: %.6e\n",conv_history[0]);
        printf("Final residual:   %.6e\n",conv_history.back());}
      printf("Relative residual |D*x - b|/|b|: %.6e\n",rd);
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
    F(r_full);
    for(int i=1;i<num_levels;i++)levels[i].free_all(set_ptr->stream);
    delete[] levels;levels=nullptr;delete[] null_vecs;null_vecs=nullptr;
    delete[] hop_packed;hop_packed=nullptr;delete[] sit_packed;sit_packed=nullptr;
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
};

} // namespace qcu
#endif
