#ifndef _LATTICE_CLOVER_MULTIGRID_H
#define _LATTICE_CLOVER_MULTIGRID_H
#include "./bistabcg.h"
#include "./define.h"
#include "./lattice_clover_dslash.h"
#include "./lattice_cuda.h"
#include "./lattice_mpi.h"
#include "./lattice_wilson_dslash.h"
#include "./multigrid.h"
#include <fstream>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>

namespace qcu {

// Logging helper
inline void ensure_log_dir() {
  struct stat st;
  if (stat("logs", &st) != 0) {
    mkdir("logs", 0755);
  }
}

template <typename T>
void log_to_file(const std::string &filename, const std::string &message) {
  ensure_log_dir();
  std::ofstream logfile;
  logfile.open("logs/" + filename, std::ios_base::app);
  if (logfile.is_open()) {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    logfile << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S")
            << " | " << message << std::endl;
    logfile.close();
  }
}

template <typename T> struct LatticeCloverMultigrid {
  LatticeSet<T> *set_ptr;
  cudaError_t err;

  // Fine-level operator components
  LatticeWilsonDslash<T> wilson_dslash;
  LatticeCloverDslash<T> clover_dslash_ee;
  LatticeCloverDslash<T> clover_dslash_oo;
  LatticeCloverDslash<T> clover_dslash_ee_inv;
  LatticeCloverDslash<T> clover_dslash_oo_inv;

  // Multigrid data
  int num_levels;
  int mg_grid_size[4]; // coarsening factor per direction
  int max_iter;
  int num_restart;
  T atol;

  // Per-level parameters (from params array)
  int level_E[8];      // DOF per level (max 8 levels)
  int level_X[8], level_Y[8], level_Z[8], level_T[8];
  int level_max_iter[8];
  int level_num_restart[8];

  // Pointers to external data
  void *gauge;           // [2, 3, 3, 4, X, Y, Z, T/2] parity-split gauge
  void *clover_ee;       // [4,3,4,3, X,Y,Z,T/2]
  void *clover_oo;
  void *clover_ee_inv;
  void *clover_oo_inv;

  // Fermion pointers (parity-split layout from Python)
  void *fermion_out_eo;  // [2, 4, 3, X, Y, Z, T/2]
  void *fermion_in_eo;

  // Fine-level solver state (odd-site vectors only)
  void *x_o, *b_e, *b_o, *b__o; // external (from fermion_in/out)
  void *r, *r_tilde, *p, *v, *s, *t; // internal allocated

  // Coarse-grid external data
  void **coarse_null_vecs;   // [num_levels-1] null vectors
  void **coarse_hoppings;    // [num_levels-1] packed hopping [2,4,E,E,Xc,Yc,Zc,Tc]
  void **coarse_sittings;    // [num_levels-1] sitting [E,E,Xc,Yc,Zc,Tc]

  // Coarse-grid solver state (allocated per level)
  void **coarse_b;       // [num_levels] RHS vectors
  void **coarse_x;       // [num_levels] solution vectors
  void **coarse_r;       // [num_levels] residual
  void **coarse_r_tilde;
  void **coarse_p;
  void **coarse_v;
  void **coarse_s;
  void **coarse_t;

  // Host-side vals for BiStabCG scalars
  LatticeComplex<T> host_vals[_vals_size_];
  LatticeComplex<T> *host_level_vals; // per-level scalars [num_levels * _vals_size_]

  // Convergence history
  T *host_convergence_history;
  int convergence_history_len;
  int convergence_history_capacity;
  int *host_level_iters;
  T *host_level_final_res;

  // Timing
  double init_time_ms;
  double solve_time_ms;
  double *level_times_ms;

  // Logging
  std::string log_prefix;
  int rank;
  bool verbose;

  void give(LatticeSet<T> *_set_ptr) {
    set_ptr = _set_ptr;
    wilson_dslash.give(set_ptr);
    clover_dslash_ee.give(set_ptr);
    clover_dslash_oo.give(set_ptr);
    clover_dslash_ee_inv.give(set_ptr);
    clover_dslash_oo_inv.give(set_ptr);
    rank = set_ptr->host_params[_NODE_RANK_];
    verbose = (set_ptr->host_params[_VERBOSE_] != 0);
  }

  // --- Memory helpers ---
  void _malloc_internal() {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    size_t sc_size = set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>);
    checkCudaErrors(cudaMallocAsync(&b__o, sc_size, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&r, sc_size, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&r_tilde, sc_size, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&p, sc_size, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&v, sc_size, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&s, sc_size, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&t, sc_size, set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }

  void _malloc_coarse_level(int lev) {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    int E = level_E[lev];
    int Xc = level_X[lev], Yc = level_Y[lev], Zc = level_Z[lev], Lt = level_T[lev];
    size_t vol = (size_t)Xc * Yc * Zc * Lt;
    size_t vec_size = vol * sizeof(LatticeComplex<T>);
    size_t sc_size = E * vec_size;

    checkCudaErrors(cudaMallocAsync(&coarse_b[lev], sc_size, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&coarse_x[lev], sc_size, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&coarse_r[lev], sc_size, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&coarse_r_tilde[lev], sc_size, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&coarse_p[lev], sc_size, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&coarse_v[lev], sc_size, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&coarse_s[lev], sc_size, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&coarse_t[lev], sc_size, set_ptr->stream));

    // Zero-initialize x and b
    checkCudaErrors(cudaMemsetAsync(coarse_x[lev], 0, sc_size, set_ptr->stream));
    checkCudaErrors(cudaMemsetAsync(coarse_b[lev], 0, sc_size, set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }

  void _free_internal() {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    if (b__o) { checkCudaErrors(cudaFreeAsync(b__o, set_ptr->stream)); b__o = nullptr; }
    if (r) { checkCudaErrors(cudaFreeAsync(r, set_ptr->stream)); r = nullptr; }
    if (r_tilde) { checkCudaErrors(cudaFreeAsync(r_tilde, set_ptr->stream)); r_tilde = nullptr; }
    if (p) { checkCudaErrors(cudaFreeAsync(p, set_ptr->stream)); p = nullptr; }
    if (v) { checkCudaErrors(cudaFreeAsync(v, set_ptr->stream)); v = nullptr; }
    if (s) { checkCudaErrors(cudaFreeAsync(s, set_ptr->stream)); s = nullptr; }
    if (t) { checkCudaErrors(cudaFreeAsync(t, set_ptr->stream)); t = nullptr; }
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }

  void _free_all_coarse() {
    if (!coarse_b) return;
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    for (int lev = 0; lev < num_levels; lev++) {
      if (coarse_b[lev]) { checkCudaErrors(cudaFreeAsync(coarse_b[lev], set_ptr->stream)); }
      if (coarse_x[lev]) { checkCudaErrors(cudaFreeAsync(coarse_x[lev], set_ptr->stream)); }
      if (coarse_r[lev]) { checkCudaErrors(cudaFreeAsync(coarse_r[lev], set_ptr->stream)); }
      if (coarse_r_tilde[lev]) { checkCudaErrors(cudaFreeAsync(coarse_r_tilde[lev], set_ptr->stream)); }
      if (coarse_p[lev]) { checkCudaErrors(cudaFreeAsync(coarse_p[lev], set_ptr->stream)); }
      if (coarse_v[lev]) { checkCudaErrors(cudaFreeAsync(coarse_v[lev], set_ptr->stream)); }
      if (coarse_s[lev]) { checkCudaErrors(cudaFreeAsync(coarse_s[lev], set_ptr->stream)); }
      if (coarse_t[lev]) { checkCudaErrors(cudaFreeAsync(coarse_t[lev], set_ptr->stream)); }
    }
    delete[] coarse_b;
    delete[] coarse_x;
    delete[] coarse_r;
    delete[] coarse_r_tilde;
    delete[] coarse_p;
    delete[] coarse_v;
    delete[] coarse_s;
    delete[] coarse_t;
    coarse_b = nullptr;
  }

  // --- Parse multigrid params from the params array ---
  void _parse_mg_params() {
    num_levels = set_ptr->host_params[_MG_NUM_LEVEL_];
    if (num_levels < 1) num_levels = 1;
    if (num_levels > 8) num_levels = 8;

    // Level 0 = fine grid (implicit from _LAT_*)
    level_E[0] = _LAT_SC_; // 12 for Clover (spin*color)
    level_X[0] = set_ptr->host_params[_LAT_X_];
    level_Y[0] = set_ptr->host_params[_LAT_Y_];
    level_Z[0] = set_ptr->host_params[_LAT_Z_];
    level_T[0] = set_ptr->host_params[_LAT_T_];
    level_max_iter[0] = set_ptr->host_params[_MAX_ITER_];

    // Parse from MG params for levels 1+
    struct { int e, x, y, z, t, mi, dt, nr; } mgp[8];
    mgp[1] = {_MG_LEVEL1_E_, _MG_LEVEL1_X_, _MG_LEVEL1_Y_, _MG_LEVEL1_Z_, _MG_LEVEL1_T_,
              _MG_LEVEL1_MAX_ITER_, _MG_LEVEL1_DATA_TYPE_, _MG_LEVEL1_NUM_RESTART_};
    mgp[2] = {_MG_LEVEL2_E_, _MG_LEVEL2_X_, _MG_LEVEL2_Y_, _MG_LEVEL2_Z_, _MG_LEVEL2_T_,
              _MG_LEVEL2_MAX_ITER_, _MG_LEVEL2_DATA_TYPE_, _MG_LEVEL2_NUM_RESTART_};
    mgp[3] = {_MG_LEVEL3_E_, _MG_LEVEL3_X_, _MG_LEVEL3_Y_, _MG_LEVEL3_Z_, _MG_LEVEL3_T_,
              _MG_LEVEL3_MAX_ITER_, _MG_LEVEL3_DATA_TYPE_, _MG_LEVEL3_NUM_RESTART_};
    mgp[4] = {_MG_LEVEL4_E_, _MG_LEVEL4_X_, _MG_LEVEL4_Y_, _MG_LEVEL4_Z_, _MG_LEVEL4_T_,
              _MG_LEVEL4_MAX_ITER_, _MG_LEVEL4_DATA_TYPE_, _MG_LEVEL4_NUM_RESTART_};

    for (int i = 1; i < num_levels; i++) {
      level_E[i] = set_ptr->host_params[mgp[i].e];
      level_X[i] = set_ptr->host_params[mgp[i].x];
      level_Y[i] = set_ptr->host_params[mgp[i].y];
      level_Z[i] = set_ptr->host_params[mgp[i].z];
      level_T[i] = set_ptr->host_params[mgp[i].t];
      level_max_iter[i] = set_ptr->host_params[mgp[i].mi];
      if (level_max_iter[i] <= 0) level_max_iter[i] = 100;
    }
    max_iter = level_max_iter[0];
    num_restart = 5;
    for (int i = 1; i < num_levels && i < 5; i++) {
      int nr = set_ptr->host_params[mgp[i].nr];
      if (nr > 0) num_restart = nr;
    }
    atol = set_ptr->host_argv[_ATOL_];

    // Coarsening factors
    for (int d = 0; d < 4; d++) {
      mg_grid_size[d] = 2; // default
      if (num_levels > 1) {
        if (level_X[0] > 0 && level_X[1] > 0)
          mg_grid_size[0] = level_X[0] / level_X[1];
        if (level_Y[0] > 0 && level_Y[1] > 0)
          mg_grid_size[1] = level_Y[0] / level_Y[1];
        if (level_Z[0] > 0 && level_Z[1] > 0)
          mg_grid_size[2] = level_Z[0] / level_Z[1];
        if (level_T[0] > 0 && level_T[1] > 0)
          mg_grid_size[3] = level_T[0] / level_T[1];
      }
    }

    // Allocate coarse pointer arrays
    coarse_null_vecs = new void*[num_levels];
    coarse_hoppings = new void*[num_levels];
    coarse_sittings = new void*[num_levels];
    for (int i = 0; i < num_levels; i++) {
      coarse_null_vecs[i] = nullptr;
      coarse_hoppings[i] = nullptr;
      coarse_sittings[i] = nullptr;
    }

    // Allocate per-level state arrays
    coarse_b = new void*[num_levels];
    coarse_x = new void*[num_levels];
    coarse_r = new void*[num_levels];
    coarse_r_tilde = new void*[num_levels];
    coarse_p = new void*[num_levels];
    coarse_v = new void*[num_levels];
    coarse_s = new void*[num_levels];
    coarse_t = new void*[num_levels];
    for (int i = 0; i < num_levels; i++) {
      coarse_b[i] = nullptr; coarse_x[i] = nullptr;
      coarse_r[i] = nullptr; coarse_r_tilde[i] = nullptr;
      coarse_p[i] = nullptr; coarse_v[i] = nullptr;
      coarse_s[i] = nullptr; coarse_t[i] = nullptr;
    }

    // Per-level host scalars
    host_level_vals = new LatticeComplex<T>[num_levels * _vals_size_];
    memset(host_level_vals, 0, num_levels * _vals_size_ * sizeof(LatticeComplex<T>));

    // Convergence history
    convergence_history_capacity = max_iter * 4 + 100;
    host_convergence_history = new T[convergence_history_capacity];
    convergence_history_len = 0;

    host_level_iters = new int[num_levels];
    host_level_final_res = new T[num_levels];
    level_times_ms = new double[num_levels];
    for (int i = 0; i < num_levels; i++) {
      host_level_iters[i] = 0;
      host_level_final_res[i] = 0.0;
      level_times_ms[i] = 0.0;
    }

    // Initialize log
    if (rank == 0) {
      std::ostringstream oss;
      oss << "MG_INIT: num_levels=" << num_levels
          << " max_iter=" << max_iter << " atol=" << atol
          << " num_restart=" << num_restart;
      log_to_file<T>("clover_multigrid.log", oss.str());
      for (int i = 0; i < num_levels; i++) {
        std::ostringstream lvl;
        lvl << "MG_LEVEL[" << i << "]: E=" << level_E[i]
            << " dims=[" << level_X[i] << "," << level_Y[i]
            << "," << level_Z[i] << "," << level_T[i] << "]"
            << " max_iter=" << level_max_iter[i];
        log_to_file<T>("clover_multigrid.log", lvl.str());
      }
    }
  }

  // --- Fine-level Dslash: CloverBiStabCG even-odd preconditioned operator ---
  void fine_dslash(void *fermion_out, void *fermion_in) {
    // A_oo * src_o - kappa^2 * D_oe * A_ee^-1 * (D_eo * src_o)
    wilson_dslash.run_eo(set_ptr->device_vec0, fermion_in, gauge);
    give_copy_vals<T><<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
        set_ptr->device_vec2, set_ptr->device_vec0);
    clover_dslash_ee_inv.give(set_ptr->device_vec2);
    wilson_dslash.run_oe(set_ptr->device_vec1, set_ptr->device_vec2, gauge);
    give_copy_vals<T><<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
        set_ptr->device_vec2, fermion_in);
    clover_dslash_oo.give(set_ptr->device_vec2);
    bistabcg_give_dest_o<T><<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
        fermion_out, set_ptr->device_vec2, set_ptr->device_vec1,
        set_ptr->kappa(), set_ptr->device_vals);
  }

  // --- Coarse-level Dslash using pre-built operators ---
  void coarse_dslash(void *fermion_out, void *fermion_in, int level) {
    int E = level_E[level];
    int Xc = level_X[level], Yc = level_Y[level], Zc = level_Z[level], Lt = level_T[level];
    int vol = Xc * Yc * Zc * Lt;
    int total = E * vol;
    dim3 grid((total + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
    dim3 block(_BLOCK_SIZE_);
    multigrid_coarse_dslash<T><<<grid, block, 0, set_ptr->stream>>>(
        fermion_out, fermion_in, coarse_hoppings[level], coarse_sittings[level],
        E, Xc, Yc, Zc, Lt);
  }

  // --- Restrict fine -> coarse ---
  void restrict_op(void *coarse_out, void *fine_in, int level) {
    // level is the FINE level index, restrict to level+1
    int E = level_E[level + 1];
    int e = level_E[level];
    int Xf = level_X[level], Yf = level_Y[level], Zf = level_Z[level], Tf = level_T[level];
    int Xc = level_X[level+1], Yc = level_Y[level+1], Zc = level_Z[level+1], Tc = level_T[level+1];
    int coarse_vol = Xc * Yc * Zc * Tc;
    int total = E * coarse_vol;
    dim3 grid((total + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
    dim3 block(_BLOCK_SIZE_);
    multigrid_restrict<T><<<grid, block, 0, set_ptr->stream>>>(
        coarse_out, fine_in, coarse_null_vecs[level], E, e,
        Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
  }

  // --- Prolong coarse -> fine ---
  void prolong_op(void *fine_out, void *coarse_in, int level) {
    // level is the FINE level index, prolong from level+1
    int E = level_E[level + 1];
    int e = level_E[level];
    int Xf = level_X[level], Yf = level_Y[level], Zf = level_Z[level], Tf = level_T[level];
    int Xc = level_X[level+1], Yc = level_Y[level+1], Zc = level_Z[level+1], Tc = level_T[level+1];
    int fine_vol = Xf * Yf * Zf * Tf;
    int total = e * fine_vol;
    dim3 grid((total + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
    dim3 block(_BLOCK_SIZE_);
    multigrid_prolong<T><<<grid, block, 0, set_ptr->stream>>>(
        fine_out, coarse_in, coarse_null_vecs[level], E, e,
        Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
  }

  // --- Dot product with MPI allreduce ---
  void _dot_mpi(void *vec0, void *vec1, const int vals_index, const int stream_index) {
    int n_elements = set_ptr->lat_4dim_SC;
    // Use cublas dot
    CUBLAS_CHECK(_cublasDot<T>(
        set_ptr->cublasHs[stream_index], n_elements, vec0, 1, vec1, 1,
        ((static_cast<LatticeComplex<T> *>(set_ptr->device_vals)) + _send_tmp_)));
    checkCudaErrors(cudaMemcpyAsync(
        ((static_cast<LatticeComplex<T> *>(host_vals)) + _send_tmp_),
        ((static_cast<LatticeComplex<T> *>(set_ptr->device_vals)) + _send_tmp_),
        sizeof(LatticeComplex<T>), cudaMemcpyDeviceToHost,
        set_ptr->streams[stream_index]));
    MPI_Barrier(MPI_COMM_WORLD);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[stream_index]));
    _MPI_Allreduce<T>(
        ((static_cast<LatticeComplex<T> *>(host_vals)) + _send_tmp_),
        ((static_cast<LatticeComplex<T> *>(host_vals)) + vals_index), 2,
        MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    checkCudaErrors(cudaMemcpyAsync(
        ((static_cast<LatticeComplex<T> *>(set_ptr->device_vals)) + vals_index),
        ((static_cast<LatticeComplex<T> *>(host_vals)) + vals_index),
        sizeof(LatticeComplex<T>), cudaMemcpyHostToDevice,
        set_ptr->streams[stream_index]));
  }

  // --- Dot product for coarse levels (no MPI needed if single-GPU coarse grid) ---
  void _dot_coarse(void *vec0, void *vec1, int level, const int vals_index,
                   const int stream_index) {
    int E = level_E[level];
    int Xc = level_X[level], Yc = level_Y[level], Zc = level_Z[level], Lt = level_T[level];
    size_t n_elements = (size_t)E * Xc * Yc * Zc * Lt;
    CUBLAS_CHECK(_cublasDot<T>(
        set_ptr->cublasHs[stream_index], n_elements, vec0, 1, vec1, 1,
        ((static_cast<LatticeComplex<T> *>(set_ptr->device_vals)) + _send_tmp_)));
    checkCudaErrors(cudaMemcpyAsync(
        ((static_cast<LatticeComplex<T> *>(host_vals)) + vals_index),
        ((static_cast<LatticeComplex<T> *>(set_ptr->device_vals)) + _send_tmp_),
        sizeof(LatticeComplex<T>), cudaMemcpyDeviceToHost,
        set_ptr->streams[stream_index]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[stream_index]));
  }

  // --- Give vector norm ---
  T _vec_norm2_coarse(void *vec, int level) {
    int E = level_E[level];
    int Xc = level_X[level], Yc = level_Y[level], Zc = level_Z[level], Lt = level_T[level];
    size_t n_elements = (size_t)E * Xc * Yc * Zc * Lt;
    CUBLAS_CHECK(_cublasDot<T>(
        set_ptr->cublasHs[_a_], n_elements, vec, 1, vec, 1,
        ((static_cast<LatticeComplex<T> *>(set_ptr->device_vals)) + _send_tmp_)));
    checkCudaErrors(cudaMemcpyAsync(
        ((static_cast<LatticeComplex<T> *>(host_vals)) + _send_tmp_),
        ((static_cast<LatticeComplex<T> *>(set_ptr->device_vals)) + _send_tmp_),
        sizeof(LatticeComplex<T>), cudaMemcpyDeviceToHost,
        set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    return sqrt(host_vals[_send_tmp_].real());
  }

  // --- BiStabCG smoother at a given level ---
  // Returns number of iterations performed
  int bistabcg_smooth(int level, void *x, void *b, void *_r, void *_r_tilde,
                       void *_p, void *_v, void *_s, void *_t,
                       T tol, int max_smooth_iter, bool is_coarsest) {

    int E = level_E[level];
    int Xc = level_X[level], Yc = level_Y[level], Zc = level_Z[level], Lt = level_T[level];
    size_t sc = (level == 0) ? set_ptr->lat_4dim_SC : (size_t)E * Xc * Yc * Zc * Lt;

    // Initialize scalars
    give_1zero<T><<<1, 1, 0, set_ptr->stream>>>(set_ptr->device_vals, _tmp0_);
    give_1zero<T><<<1, 1, 0, set_ptr->stream>>>(set_ptr->device_vals, _tmp1_);
    give_1one<T><<<1, 1, 0, set_ptr->stream>>>(set_ptr->device_vals, _rho_prev_);
    give_1zero<T><<<1, 1, 0, set_ptr->stream>>>(set_ptr->device_vals, _rho_);
    give_1one<T><<<1, 1, 0, set_ptr->stream>>>(set_ptr->device_vals, _alpha_);
    give_1one<T><<<1, 1, 0, set_ptr->stream>>>(set_ptr->device_vals, _omega_);
    give_1zero<T><<<1, 1, 0, set_ptr->stream>>>(set_ptr->device_vals, _send_tmp_);
    give_1zero<T><<<1, 1, 0, set_ptr->stream>>>(set_ptr->device_vals, _norm2_tmp_);
    give_1zero<T><<<1, 1, 0, set_ptr->stream>>>(set_ptr->device_vals, _diff2_tmp_);

    // Compute r = b - D*x
    if (level == 0) {
      // r = b - D*x using fine dslash
      // Step 1: D*x -> device_vec0
      fine_dslash(set_ptr->device_vec0, x);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      // Step 2: b copy -> device_vec2
      dim3 g = set_ptr->gridDim;
      dim3 blk = set_ptr->blockDim;
      give_copy_vals<T><<<g, blk, 0, set_ptr->stream>>>(set_ptr->device_vec2, b);
      // Step 3: r = b - D*x  (diff2 kernel: vec = x - ans, output in 3rd arg)
      bistabcg_give_diff2<T><<<g, blk, 0, set_ptr->stream>>>(
          set_ptr->device_vec2, set_ptr->device_vec0, _r, set_ptr->device_vals);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    } else {
      // r = b - D*x (with coarse dslash)
      coarse_dslash(set_ptr->device_vec0, x, level);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      // r = b - D*x
      int total = E * Xc * Yc * Zc * Lt;
      dim3 g((total + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
      dim3 blk(_BLOCK_SIZE_);
      bistabcg_give_diff2<T><<<g, blk, 0, set_ptr->stream>>>(
          _r, set_ptr->device_vec0, set_ptr->device_vec0, set_ptr->device_vals);
    }

    // r_tilde = r
    dim3 g_copy = (level == 0) ? set_ptr->gridDim :
        dim3(((size_t)E * Xc * Yc * Zc * Lt + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
    dim3 blk_copy(_BLOCK_SIZE_);
    give_copy_vals<T><<<g_copy, blk_copy, 0, set_ptr->stream>>>(_r_tilde, _r);

    // Zero p, v
    give_custom_vals<T><<<g_copy, blk_copy, 0, set_ptr->stream>>>(_p, 0.0, 0.0);
    give_custom_vals<T><<<g_copy, blk_copy, 0, set_ptr->stream>>>(_v, 0.0, 0.0);

    int iter;
    for (iter = 0; iter < max_smooth_iter; iter++) {
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

      // rho = dot(r_tilde, r)
      if (level == 0) {
        _dot_mpi(_r_tilde, _r, _rho_, _a_);
      } else {
        _dot_coarse(_r_tilde, _r, level, _rho_, _a_);
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));

      // beta = (rho / rho_prev) * (alpha / omega)
      bistabcg_give_1beta<T><<<1, 1, 0, set_ptr->streams[_a_]>>>(set_ptr->device_vals);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));

      // rho_prev = rho
      bistabcg_give_1rho_prev<T><<<1, 1, 0, set_ptr->streams[_b_]>>>(set_ptr->device_vals);

      // p = r + beta*(p - omega*v)
      bistabcg_give_p<T><<<g_copy, blk_copy, 0, set_ptr->streams[_a_]>>>(
          _p, _r, _v, set_ptr->device_vals);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));

      // v = D * p
      if (level == 0) {
        fine_dslash(_v, _p);
      } else {
        coarse_dslash(_v, _p, level);
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));

      // tmp0 = dot(r_tilde, v)
      if (level == 0) {
        _dot_mpi(_r_tilde, _v, _tmp0_, _d_);
      } else {
        _dot_coarse(_r_tilde, _v, level, _tmp0_, _d_);
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

      // alpha = rho / tmp0
      bistabcg_give_1alpha<T><<<1, 1, 0, set_ptr->streams[_d_]>>>(set_ptr->device_vals);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

      // s = r - alpha*v
      bistabcg_give_s<T><<<g_copy, blk_copy, 0, set_ptr->streams[_a_]>>>(
          _s, _r, _v, set_ptr->device_vals);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));

      // t = D * s
      if (level == 0) {
        fine_dslash(_t, _s);
      } else {
        coarse_dslash(_t, _s, level);
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));

      // tmp0 = dot(t, s), tmp1 = dot(t, t)
      if (level == 0) {
        _dot_mpi(_t, _s, _tmp0_, _c_);
        _dot_mpi(_t, _t, _tmp1_, _d_);
      } else {
        _dot_coarse(_t, _s, level, _tmp0_, _c_);
        _dot_coarse(_t, _t, level, _tmp1_, _d_);
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

      // omega = tmp0 / tmp1
      bistabcg_give_1omega<T><<<1, 1, 0, set_ptr->streams[_d_]>>>(set_ptr->device_vals);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

      // x = x + alpha*p + omega*s
      bistabcg_give_x_o<T><<<g_copy, blk_copy, 0, set_ptr->streams[_b_]>>>(
          x, _p, _s, set_ptr->device_vals);

      // r = s - omega*t
      bistabcg_give_r<T><<<g_copy, blk_copy, 0, set_ptr->streams[_a_]>>>(
          _r, _s, _t, set_ptr->device_vals);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));

      // Check residual norm
      if (level == 0) {
        _dot_mpi(_r, _r, _norm2_tmp_, _c_);
        checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
        T r_norm = sqrt(host_vals[_norm2_tmp_].real());
        if (rank == 0 && verbose) {
          std::ostringstream oss;
          oss << "MG_SMOOTH[" << level << "] iter=" << iter
              << " res=" << std::scientific << r_norm;
          log_to_file<T>("clover_multigrid.log", oss.str());
        }
        // Record convergence history for level 0
        if (convergence_history_len < convergence_history_capacity) {
          host_convergence_history[convergence_history_len++] = r_norm;
        }
        if (r_norm < tol) break;
      } else {
        T r_norm = _vec_norm2_coarse(_r, level);
        if (rank == 0 && verbose) {
          std::ostringstream oss;
          oss << "MG_SMOOTH[" << level << "] iter=" << iter
              << " res=" << std::scientific << r_norm;
          log_to_file<T>("clover_multigrid.log", oss.str());
        }
        if (r_norm < tol) break;
      }
    }

    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

    return iter + 1;
  }

  // --- Recursive V-cycle ---
  void v_cycle(int level) {
    auto t_start = std::chrono::high_resolution_clock::now();

    int E = level_E[level];
    int Xc = level_X[level], Yc = level_Y[level], Zc = level_Z[level], Lt = level_T[level];
    int vol = Xc * Yc * Zc * Lt;

    void *lv_b, *lv_x, *lv_r, *lv_rt, *lv_p, *lv_v, *lv_s, *lv_t;
    if (level == 0) {
      lv_b = b__o;
      lv_x = x_o;
      lv_r = r;
      lv_rt = r_tilde;
      lv_p = p;
      lv_v = v;
      lv_s = s;
      lv_t = t;
    } else {
      lv_b = coarse_b[level];
      lv_x = coarse_x[level];
      lv_r = coarse_r[level];
      lv_rt = coarse_r_tilde[level];
      lv_p = coarse_p[level];
      lv_v = coarse_v[level];
      lv_s = coarse_s[level];
      lv_t = coarse_t[level];
    }

    // Smoothing tolerance: relative to initial residual
    T smooth_tol = atol * static_cast<T>(0.1);
    if (level == num_levels - 1) smooth_tol = atol * static_cast<T>(0.01);

    // Pre-smoothing
    int pre_iters = bistabcg_smooth(level, lv_x, lv_b, lv_r, lv_rt,
                                     lv_p, lv_v, lv_s, lv_t,
                                     smooth_tol, level_max_iter[level], false);

    // If not coarsest, do coarse-grid correction
    if (level < num_levels - 1) {
      // Compute residual r = b - D*x
      if (level == 0) {
        fine_dslash(set_ptr->device_vec0, lv_x);
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
        bistabcg_give_diff2<T><<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
            lv_r, set_ptr->device_vec0, set_ptr->device_vec0, set_ptr->device_vals);
      } else {
        coarse_dslash(set_ptr->device_vec0, lv_x, level);
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
        int total = E * vol;
        dim3 g((total + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
        dim3 blk(_BLOCK_SIZE_);
        bistabcg_give_diff2<T><<<g, blk, 0, set_ptr->stream>>>(
            lv_r, set_ptr->device_vec0, set_ptr->device_vec0, set_ptr->device_vals);
      }

      // Restrict residual to coarse grid
      restrict_op(coarse_b[level + 1], lv_r, level);

      // Zero coarse solution
      int Ec = level_E[level + 1];
      int Xc2 = level_X[level+1], Yc2 = level_Y[level+1], Zc2 = level_Z[level+1], Tc2 = level_T[level+1];
      size_t c_size = (size_t)Ec * Xc2 * Yc2 * Zc2 * Tc2 * sizeof(LatticeComplex<T>);
      checkCudaErrors(cudaMemsetAsync(coarse_x[level + 1], 0, c_size, set_ptr->stream));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));

      // Recursive V-cycle
      v_cycle(level + 1);

      // Prolong correction back
      prolong_op(set_ptr->device_vec0, coarse_x[level + 1], level);

      // x = x + correction (using cublasAxpy for correct accumulation)
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      {
        int total_elements = (level == 0) ? set_ptr->lat_4dim_SC
                                          : E * vol;
        dim3 g((total_elements + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
        dim3 blk(_BLOCK_SIZE_);
        // Copy x to device_vec1, then device_vec1 += correction, then copy back
        give_copy_vals<T><<<g, blk, 0, set_ptr->stream>>>(set_ptr->device_vec1, lv_x);
        LatticeComplex<T> one(1.0, 0.0);
        CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH, total_elements,
                                     &one, set_ptr->device_vec0, 1, set_ptr->device_vec1, 1));
        give_copy_vals<T><<<g, blk, 0, set_ptr->stream>>>(lv_x, set_ptr->device_vec1);
      }

      // Post-smoothing
      bistabcg_smooth(level, lv_x, lv_b, lv_r, lv_rt,
                       lv_p, lv_v, lv_s, lv_t,
                       smooth_tol, level_max_iter[level] / 2, false);
    }

    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    auto t_end = std::chrono::high_resolution_clock::now();
    double t_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    level_times_ms[level] += t_ms;
  }

  // --- Initialize: set up b__o from fermion input ---
  void _setup_b() {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    // b__o = b_o + kappa * D_oe(A_ee^-1 * b_e)
    // We already have b_e = fermion_in_eo[0], b_o = fermion_in_eo[1]
    give_copy_vals<T><<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
        set_ptr->device_vec2, b_e);
    clover_dslash_ee_inv.give(set_ptr->device_vec2);
    wilson_dslash.run_oe(set_ptr->device_vec0, set_ptr->device_vec2, gauge);
    bistabcg_give_b__o<T><<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
        b__o, b_o, set_ptr->device_vec0, set_ptr->kappa(), set_ptr->device_vals);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }

  // --- Compute x_e from x_o ---
  void _recover_x_e() {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    // x_e = A_ee^-1 * (b_e + kappa * D_eo(x_o))
    CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH, set_ptr->lat_4dim_SC * _REAL_IMAG_,
                                 (T *)b_e, 1, (T *)set_ptr->device_vec0, 1));
    wilson_dslash.run_eo(set_ptr->device_vec1, x_o, gauge);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    LatticeComplex<T> kap(set_ptr->kappa(), 0.0);
    CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH, set_ptr->lat_4dim_SC, &kap,
                                 set_ptr->device_vec1, 1, set_ptr->device_vec0, 1));
    clover_dslash_ee_inv.give(set_ptr->device_vec0);
    // Copy to x_e (first half of fermion_out_eo)
    CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH, set_ptr->lat_4dim_SC * _REAL_IMAG_,
                                 (T *)set_ptr->device_vec0, 1, (T *)fermion_out_eo, 1));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }

  // --- Main init: accept external data ---
  void init(void *_fermion_out, void *_fermion_in,
            void *_gauge, void *_clover_ee, void *_clover_oo,
            void *_clover_ee_inv, void *_clover_oo_inv) {
    // Parse parameters
    _parse_mg_params();

    // Set external pointers
    fermion_out_eo = _fermion_out;
    fermion_in_eo = _fermion_in;
    gauge = _gauge;
    clover_ee = _clover_ee;
    clover_oo = _clover_oo;
    clover_ee_inv = _clover_ee_inv;
    clover_oo_inv = _clover_oo_inv;

    // Initialize clover dslash components
    clover_dslash_ee.init(clover_ee);
    clover_dslash_oo.init(clover_oo);
    clover_dslash_ee_inv.init(clover_ee_inv);
    clover_dslash_oo_inv.init(clover_oo_inv);

    // Set sub-pointers for parity-split layout
    b_e = fermion_in_eo;
    b_o = ((static_cast<LatticeComplex<T> *>(fermion_in_eo)) + set_ptr->lat_4dim_SC);
    x_o = ((static_cast<LatticeComplex<T> *>(fermion_out_eo)) + set_ptr->lat_4dim_SC);

    // Allocate internal vectors
    _malloc_internal();
    // Initialize x_o to 0
    checkCudaErrors(cudaMemsetAsync(x_o, 0,
        set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>), set_ptr->stream));

    // Setup b__o (preconditioned RHS)
    _setup_b();

    // Allocate coarse-level vectors
    for (int lev = 1; lev < num_levels; lev++) {
      _malloc_coarse_level(lev);
    }

    if (rank == 0) {
      log_to_file<T>("clover_multigrid.log", "MG_INIT_COMPLETE: Solver ready");
    }
  }

  // --- Set coarse-grid operators from external ---
  void set_coarse_operators(int level, void *null_vecs, void *hopping, void *sitting) {
    if (level >= 0 && level < num_levels) {
      coarse_null_vecs[level] = null_vecs;
      coarse_hoppings[level] = hopping;
      coarse_sittings[level] = sitting;
    }
  }

  // --- Main solve ---
  void run() {
    auto solve_start = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
      log_to_file<T>("clover_multigrid.log", "MG_SOLVE_START: Beginning V-cycles");
    }

    // Execute V-cycles with adaptive convergence
    int total_cycles = 0;
    int max_cycles = (max_iter / (num_restart > 0 ? num_restart : 5)) + 1;
    if (max_cycles < 1) max_cycles = 1;
    if (max_cycles > 20) max_cycles = 20;  // cap at 20 V-cycles

    // b__o is already set up in init(), don't recompute
    for (int cycle = 0; cycle < max_cycles; cycle++) {
      if (rank == 0 && verbose) {
        std::ostringstream oss;
        oss << "MG_CYCLE[" << cycle << "]: Starting V-cycle";
        log_to_file<T>("clover_multigrid.log", oss.str());
      }

      // Run one V-cycle
      v_cycle(0);

      // Check convergence: compute |D*x_o - b__o|
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      fine_dslash(set_ptr->device_vec0, x_o);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      // r = D*x_o - b__o (diff2: vec = x - ans)
      bistabcg_give_diff2<T><<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
          set_ptr->device_vec0, b__o, r, set_ptr->device_vals);
      _dot_mpi(r, r, _norm2_tmp_, _c_);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));

      T r_norm = sqrt(host_vals[_norm2_tmp_].real());
      if (rank == 0) {
        std::ostringstream oss;
        oss << "MG_CYCLE[" << cycle << "]: residual=" << std::scientific << r_norm;
        log_to_file<T>("clover_multigrid.log", oss.str());
      }
      if (convergence_history_len < convergence_history_capacity) {
        host_convergence_history[convergence_history_len++] = r_norm;
      }

      total_cycles++;
      if (r_norm < atol) {
        if (rank == 0) {
          log_to_file<T>("clover_multigrid.log", "MG_CONVERGED: Tolerance reached");
        }
        break;
      }
    }

    // Recover x_e from x_o
    _recover_x_e();

    auto solve_end = std::chrono::high_resolution_clock::now();
    solve_time_ms = std::chrono::duration<double, std::milli>(solve_end - solve_start).count();

    if (rank == 0) {
      std::ostringstream oss;
      oss << "MG_SOLVE_END: cycles=" << total_cycles
          << " time_ms=" << solve_time_ms;
      log_to_file<T>("clover_multigrid.log", oss.str());

      // Save convergence history
      std::ostringstream conv_oss;
      conv_oss << "CONVERGENCE_HISTORY: [";
      for (int i = 0; i < convergence_history_len; i++) {
        if (i > 0) conv_oss << ",";
        conv_oss << std::scientific << host_convergence_history[i];
      }
      conv_oss << "]";
      log_to_file<T>("clover_multigrid.log", conv_oss.str());
    }
  }

  // --- Verify solution (test mode) ---
  void run_test() {
    auto start = std::chrono::high_resolution_clock::now();
    run();
    auto end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Compute residual
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    fine_dslash(set_ptr->device_vec1, x_o);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    _dot_mpi(set_ptr->device_vec1, set_ptr->device_vec1, _norm2_tmp_, _c_);
    _dot_mpi(b__o, b__o, _diff2_tmp_, _d_);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

    T norm_dest = sqrt(host_vals[_norm2_tmp_].real());
    T norm_b = sqrt(host_vals[_diff2_tmp_].real());

    // diff = dest - b__o
    bistabcg_give_diff2<T><<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
        set_ptr->device_vec1, b__o, set_ptr->device_vec0, set_ptr->device_vals);
    _dot_mpi(set_ptr->device_vec0, set_ptr->device_vec0, _diff2_tmp_, _c_);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    T diff_norm = sqrt(host_vals[_diff2_tmp_].real());
    T rel_diff = (norm_b > 1e-30) ? diff_norm / norm_b : diff_norm;

    if (rank == 0) {
      printf("=== MULTIGRID SOLVER REPORT ===\n");
      printf("Total time: %.6f ms (%.6f s)\n", total_ms, total_ms / 1000.0);
      printf("Solve time: %.6f ms\n", solve_time_ms);
      printf("Convergence history entries: %d\n", convergence_history_len);
      if (convergence_history_len > 0) {
        printf("Initial residual: %.6e\n", host_convergence_history[0]);
        printf("Final residual:   %.6e\n", host_convergence_history[convergence_history_len - 1]);
      }
      printf("Relative residual |D*x - b|/|b|: %.6e\n", rel_diff);

      std::ostringstream oss;
      oss << "=== REPORT === total_ms=" << total_ms
          << " solve_ms=" << solve_time_ms
          << " init_res=" << (convergence_history_len > 0 ? host_convergence_history[0] : 0)
          << " final_res=" << (convergence_history_len > 0 ? host_convergence_history[convergence_history_len-1] : 0)
          << " rel_diff=" << rel_diff;
      log_to_file<T>("clover_multigrid_report.log", oss.str());
    }
  }

  // --- Cleanup ---
  void end() {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

    _free_internal();
    _free_all_coarse();

    if (host_convergence_history) { delete[] host_convergence_history; host_convergence_history = nullptr; }
    if (host_level_iters) { delete[] host_level_iters; host_level_iters = nullptr; }
    if (host_level_final_res) { delete[] host_level_final_res; host_level_final_res = nullptr; }
    if (level_times_ms) { delete[] level_times_ms; level_times_ms = nullptr; }
    if (host_level_vals) { delete[] host_level_vals; host_level_vals = nullptr; }

    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
};

} // namespace qcu
#endif
