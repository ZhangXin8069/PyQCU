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
#include <vector>
#include <cmath>

namespace qcu {

inline void ensure_log_dir() {
  struct stat st;
  if (stat("logs", &st) != 0) mkdir("logs", 0755);
}

template <typename T>
inline void log_printf(const std::string &msg, int rank, bool to_stdout = true) {
  ensure_log_dir();
  std::ofstream f("logs/clover_multigrid.log", std::ios_base::app);
  if (f.is_open()) {
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    f << std::put_time(std::localtime(&tt), "%Y-%m-%d %H:%M:%S") << " | " << msg << std::endl;
    f.close();
  }
  if (to_stdout && rank == 0)
    printf("%s\n", msg.c_str());
}

// Pack 4 scalars for batched transfer to device_vals
template <typename T> struct BiStabCGScalars {
  LatticeComplex<T> rho, rho_prev, alpha, omega, tmp0, tmp1;
};

// Per-level state
template <typename T> struct MgLevelState {
  void *x, *b, *r, *r_tilde, *p, *v, *s, *t;
  T rho, rho_prev, alpha, omega;
  int dof, X, Y, Z, Lt, vol;
  size_t vec_sz;
  bool owned;

  void alloc(int _dof, int _X, int _Y, int _Z, int _Lt, cudaStream_t stream) {
    dof = _dof; X = _X; Y = _Y; Z = _Z; Lt = _Lt;
    vol = X * Y * Z * Lt;
    vec_sz = (size_t)dof * vol;
    size_t bytes = vec_sz * sizeof(LatticeComplex<T>);
    checkCudaErrors(cudaMallocAsync(&x, bytes, stream));
    checkCudaErrors(cudaMallocAsync(&b, bytes, stream));
    checkCudaErrors(cudaMallocAsync(&r, bytes, stream));
    checkCudaErrors(cudaMallocAsync(&r_tilde, bytes, stream));
    checkCudaErrors(cudaMallocAsync(&p, bytes, stream));
    checkCudaErrors(cudaMallocAsync(&v, bytes, stream));
    checkCudaErrors(cudaMallocAsync(&s, bytes, stream));
    checkCudaErrors(cudaMallocAsync(&t, bytes, stream));
    owned = true;
    checkCudaErrors(cudaMemsetAsync(x, 0, bytes, stream));
    checkCudaErrors(cudaMemsetAsync(b, 0, bytes, stream));
  }

  void free_all(cudaStream_t stream) {
    if (!owned) return;
    auto fre = [&](void *&p) { if (p) { cudaFreeAsync(p, stream); p = nullptr; } };
    fre(x); fre(b); fre(r); fre(r_tilde); fre(p); fre(v); fre(s); fre(t);
    owned = false;
  }
};

// Get MPI datatype for T
template <typename T> inline MPI_Datatype mpitype() { return MPI_FLOAT; }
template <> inline MPI_Datatype mpitype<double>() { return MPI_DOUBLE; }

template <typename T> struct LatticeCloverMultigrid {
  LatticeSet<T> *set_ptr;
  LatticeWilsonDslash<T> wilson_dslash;
  LatticeCloverDslash<T> clover_dslash_ee, clover_dslash_oo;
  LatticeCloverDslash<T> clover_dslash_ee_inv, clover_dslash_oo_inv;

  void *gauge, *clover_ee, *clover_oo, *clover_ee_inv, *clover_oo_inv;
  void *fermion_out_eo, *fermion_in_eo;
  void *b_e, *b_o, *x_o;
  void *b__o;
  void *r0, *rt0, *p0, *v0, *s0, *t0;

  int num_levels;
  MgLevelState<T> *levels;
  void **null_vecs, **hop_packed, **sit_packed;

  int max_iter;
  T atol;
  int num_restart;
  int rank;
  bool verbose;
  T kappa_val;

  std::vector<T> conv_history;
  int *level_iters;
  double *level_times;
  double solve_time_ms;

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

  // ---- fine-level dslash (even-odd preconditioned Clover) ----
  void fine_dslash_op(void *out, void *in) {
    // A_oo*src_o - kappa^2*D_oe*A_ee^-1*(D_eo(src_o))
    wilson_dslash.run_eo(set_ptr->device_vec0, in, gauge);
    give_copy_vals<T><<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
        set_ptr->device_vec2, set_ptr->device_vec0);
    clover_dslash_ee_inv.give(set_ptr->device_vec2);
    wilson_dslash.run_oe(set_ptr->device_vec1, set_ptr->device_vec2, gauge);
    give_copy_vals<T><<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
        set_ptr->device_vec2, in);
    clover_dslash_oo.give(set_ptr->device_vec2);
    bistabcg_give_dest_o<T><<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
        out, set_ptr->device_vec2, set_ptr->device_vec1, kappa_val, set_ptr->device_vals);
  }

  // ---- coarse-level dslash ----
  void coarse_dslash_op(void *out, void *in, int lev) {
    int E = levels[lev].dof;
    int Xc = levels[lev].X, Yc = levels[lev].Y, Zc = levels[lev].Z, Lt = levels[lev].Lt;
    int total = E * Xc * Yc * Zc * Lt;
    dim3 g((total + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
    multigrid_coarse_dslash<T><<<g, _BLOCK_SIZE_, 0, set_ptr->stream>>>(
        out, in, hop_packed[lev-1], sit_packed[lev-1], E, Xc, Yc, Zc, Lt);
  }

  void restrict_op(void *coarse_out, void *fine_in, int fine_lev) {
    int lev = fine_lev + 1;
    int E = levels[lev].dof, e = levels[fine_lev].dof;
    int Xf = levels[fine_lev].X, Yf = levels[fine_lev].Y, Zf = levels[fine_lev].Z, Ltf = levels[fine_lev].Lt;
    int Xc = levels[lev].X, Yc = levels[lev].Y, Zc = levels[lev].Z, Ltc = levels[lev].Lt;
    int total = E * Xc * Yc * Zc * Ltc;
    dim3 g((total + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
    multigrid_restrict<T><<<g, _BLOCK_SIZE_, 0, set_ptr->stream>>>(
        coarse_out, fine_in, null_vecs[fine_lev], E, e, Xf, Yf, Zf, Ltf, Xc, Yc, Zc, Ltc);
  }

  void prolong_op(void *fine_out, void *coarse_in, int fine_lev) {
    int lev = fine_lev + 1;
    int E = levels[lev].dof, e = levels[fine_lev].dof;
    int Xf = levels[fine_lev].X, Yf = levels[fine_lev].Y, Zf = levels[fine_lev].Z, Ltf = levels[fine_lev].Lt;
    int Xc = levels[lev].X, Yc = levels[lev].Y, Zc = levels[lev].Z, Ltc = levels[lev].Lt;
    int total = e * Xf * Yf * Zf * Ltf;
    dim3 g((total + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
    multigrid_prolong<T><<<g, _BLOCK_SIZE_, 0, set_ptr->stream>>>(
        fine_out, coarse_in, null_vecs[fine_lev], E, e, Xf, Yf, Zf, Ltf, Xc, Yc, Zc, Ltc);
  }

  // ---- MPI dot product with correct MPI type ----
  void dot_mpi(void *a, void *b, T &result, int stream_idx) {
    LatticeComplex<T> *vals = static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
    CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasHs[stream_idx], set_ptr->lat_4dim_SC,
        a, 1, b, 1, &vals[_send_tmp_]));
    LatticeComplex<T> host_tmp;
    checkCudaErrors(cudaMemcpyAsync(&host_tmp, &vals[_send_tmp_],
        sizeof(LatticeComplex<T>), cudaMemcpyDeviceToHost, set_ptr->streams[stream_idx]));
    MPI_Barrier(MPI_COMM_WORLD);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[stream_idx]));
    T g = host_tmp.real();
    MPI_Allreduce(MPI_IN_PLACE, &g, 1, mpitype<T>(), MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    result = g;
  }

  void dot_coarse(void *a, void *b, int lev, T &result, int stream_idx) {
    size_t n = levels[lev].vec_sz;
    LatticeComplex<T> *vals = static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
    CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasHs[stream_idx], n, a, 1, b, 1, &vals[_send_tmp_]));
    LatticeComplex<T> host_tmp;
    checkCudaErrors(cudaMemcpyAsync(&host_tmp, &vals[_send_tmp_],
        sizeof(LatticeComplex<T>), cudaMemcpyDeviceToHost, set_ptr->streams[stream_idx]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[stream_idx]));
    result = host_tmp.real();
  }

  T norm2_coarse(void *v, int lev) {
    size_t n = levels[lev].vec_sz;
    LatticeComplex<T> *vals = static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
    CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasHs[_a_], n, v, 1, v, 1, &vals[_send_tmp_]));
    LatticeComplex<T> host_tmp;
    checkCudaErrors(cudaMemcpyAsync(&host_tmp, &vals[_send_tmp_],
        sizeof(LatticeComplex<T>), cudaMemcpyDeviceToHost, set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    return host_tmp.real();
  }

  void zero_coarse(void *v, int lev) {
    checkCudaErrors(cudaMemsetAsync(v, 0, levels[lev].vec_sz * sizeof(LatticeComplex<T>), set_ptr->stream));
  }

  void copy_coarse(void *dst, void *src, int lev) {
    int total = (int)levels[lev].vec_sz;
    dim3 g((total + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
    give_copy_vals<T><<<g, _BLOCK_SIZE_, 0, set_ptr->stream>>>(dst, src);
  }

  void axpy_coarse(void *y, T a, void *x, int lev) {
    LatticeComplex<T> alpha(a, 0.0);
    CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH, levels[lev].vec_sz, &alpha, x, 1, y, 1));
  }

  // ---- Batched scalar upload to device_vals ----
  void upload_scalars(MgLevelState<T> &st, cudaStream_t stream) {
    BiStabCGScalars<T> sc;
    sc.rho = LatticeComplex<T>(st.rho, 0.0);
    sc.rho_prev = LatticeComplex<T>(st.rho_prev, 0.0);
    sc.alpha = LatticeComplex<T>(st.alpha, 0.0);
    sc.omega = LatticeComplex<T>(st.omega, 0.0);
    sc.tmp0 = LatticeComplex<T>(0.0, 0.0);
    sc.tmp1 = LatticeComplex<T>(0.0, 0.0);
    LatticeComplex<T> *vals = static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
    // Single batched copy for rho, rho_prev, alpha, omega
    checkCudaErrors(cudaMemcpyAsync(&vals[_rho_], &sc.rho, 4 * sizeof(LatticeComplex<T>),
        cudaMemcpyHostToDevice, stream));
  }

  void upload_tmp0(T val, cudaStream_t stream) {
    LatticeComplex<T> tv(val, 0.0);
    LatticeComplex<T> *vals = static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
    checkCudaErrors(cudaMemcpyAsync(&vals[_tmp0_], &tv, sizeof(LatticeComplex<T>),
        cudaMemcpyHostToDevice, stream));
  }

  void upload_tmp01(T t0, T t1, cudaStream_t stream) {
    LatticeComplex<T> tv[2] = { LatticeComplex<T>(t0, 0.0), LatticeComplex<T>(t1, 0.0) };
    LatticeComplex<T> *vals = static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
    checkCudaErrors(cudaMemcpyAsync(&vals[_tmp0_], tv, 2 * sizeof(LatticeComplex<T>),
        cudaMemcpyHostToDevice, stream));
  }

  // ---- Single BiStabCG iteration (optimized sync path) ----
  T bistabcg_iter(int lev) {
    auto &st = levels[lev];
    bool is_fine = (lev == 0);
    cudaStream_t strm = set_ptr->stream;

    dim3 g, blk;
    if (is_fine) { g = set_ptr->gridDim; blk = set_ptr->blockDim; }
    else { int t = (int)st.vec_sz; g = dim3((t + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_); blk = dim3(_BLOCK_SIZE_); }

    // Sync only before reusing shared scratch buffers
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

    // 1. rho = dot(r_tilde, r)
    T dot_tmp;
    if (is_fine) dot_mpi(st.r_tilde, st.r, dot_tmp, _a_);
    else dot_coarse(st.r_tilde, st.r, lev, dot_tmp, _a_);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    st.rho = dot_tmp;

    // 2. beta = (rho/rho_prev)*(alpha/omega)  — upload scalars (batched)
    upload_scalars(st, strm);
    checkCudaErrors(cudaStreamSynchronize(strm));
    bistabcg_give_1beta<T><<<1, 1, 0, set_ptr->streams[_a_]>>>(set_ptr->device_vals);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    bistabcg_give_1rho_prev<T><<<1, 1, 0, set_ptr->streams[_b_]>>>(set_ptr->device_vals);
    // No sync needed — kernel on stream _b_ doesn't use vals read by next kernel on _a_

    // 3. p = r + beta*(p - omega*v)
    bistabcg_give_p<T><<<g, blk, 0, set_ptr->streams[_a_]>>>(st.p, st.r, st.v, set_ptr->device_vals);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));

    // 4. v = A * p
    checkCudaErrors(cudaStreamSynchronize(strm)); // ensure scratch buffers free
    if (is_fine) fine_dslash_op(st.v, st.p);
    else coarse_dslash_op(st.v, st.p, lev);
    // fine_dslash_op has internal syncs via dslash routines

    // 5. tmp0 = dot(r_tilde, v)
    if (is_fine) dot_mpi(st.r_tilde, st.v, dot_tmp, _d_);
    else dot_coarse(st.r_tilde, st.v, lev, dot_tmp, _d_);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
    upload_tmp0(dot_tmp, strm);
    checkCudaErrors(cudaStreamSynchronize(strm));

    // 6. alpha = rho / tmp0
    bistabcg_give_1alpha<T><<<1, 1, 0, set_ptr->streams[_d_]>>>(set_ptr->device_vals);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

    // 7. s = r - alpha*v
    bistabcg_give_s<T><<<g, blk, 0, set_ptr->streams[_a_]>>>(st.s, st.r, st.v, set_ptr->device_vals);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));

    // 8. t = A * s
    checkCudaErrors(cudaStreamSynchronize(strm));
    if (is_fine) fine_dslash_op(st.t, st.s);
    else coarse_dslash_op(st.t, st.s, lev);

    // 9. tmp0 = dot(t, s), tmp1 = dot(t, t)
    T ts, tt;
    if (is_fine) { dot_mpi(st.t, st.s, ts, _c_); dot_mpi(st.t, st.t, tt, _d_); }
    else { dot_coarse(st.t, st.s, lev, ts, _c_); dot_coarse(st.t, st.t, lev, tt, _d_); }
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
    upload_tmp01(ts, tt, strm);
    checkCudaErrors(cudaStreamSynchronize(strm));

    // 10. omega = tmp0 / tmp1
    bistabcg_give_1omega<T><<<1, 1, 0, set_ptr->streams[_d_]>>>(set_ptr->device_vals);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

    // 11. x = x + alpha*p + omega*s
    bistabcg_give_x_o<T><<<g, blk, 0, set_ptr->streams[_b_]>>>(st.x, st.p, st.s, set_ptr->device_vals);

    // 12. r = s - omega*t
    bistabcg_give_r<T><<<g, blk, 0, set_ptr->streams[_a_]>>>(st.r, st.s, st.t, set_ptr->device_vals);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));

    // Read back alpha, omega for next iteration
    st.rho_prev = st.rho;
    LatticeComplex<T> *vals = static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
    LatticeComplex<T> hao[2];
    checkCudaErrors(cudaMemcpyAsync(hao, &vals[_alpha_], 2 * sizeof(LatticeComplex<T>),
        cudaMemcpyDeviceToHost, strm));
    checkCudaErrors(cudaStreamSynchronize(strm));
    st.alpha = hao[0].real();
    st.omega = hao[1].real();

    T rn = sqrt(norm2_coarse(st.r, lev));
    return rn;
  }

  // ---- Setup: b__o = b_o + kappa * D_oe(A_ee^-1 * b_e) ----
  void setup_b__o() {
    give_copy_vals<T><<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
        set_ptr->device_vec2, b_e);
    clover_dslash_ee_inv.give(set_ptr->device_vec2);
    wilson_dslash.run_oe(set_ptr->device_vec0, set_ptr->device_vec2, gauge);
    bistabcg_give_b__o<T><<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
        b__o, b_o, set_ptr->device_vec0, kappa_val, set_ptr->device_vals);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }

  void recover_x_e() {
    CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH, set_ptr->lat_4dim_SC * _REAL_IMAG_,
        (T*)b_e, 1, (T*)set_ptr->device_vec0, 1));
    wilson_dslash.run_eo(set_ptr->device_vec1, x_o, gauge);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    LatticeComplex<T> kap(kappa_val, 0.0);
    CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH, set_ptr->lat_4dim_SC, &kap,
        set_ptr->device_vec1, 1, set_ptr->device_vec0, 1));
    clover_dslash_ee_inv.give(set_ptr->device_vec0);
    CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH, set_ptr->lat_4dim_SC * _REAL_IMAG_,
        (T*)set_ptr->device_vec0, 1, (T*)fermion_out_eo, 1));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }

  T fine_norm2(void *v) {
    LatticeComplex<T> *vals = static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
    CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasHs[_a_], set_ptr->lat_4dim_SC,
        v, 1, v, 1, &vals[_send_tmp_]));
    LatticeComplex<T> host_tmp;
    checkCudaErrors(cudaMemcpyAsync(&host_tmp, &vals[_send_tmp_],
        sizeof(LatticeComplex<T>), cudaMemcpyDeviceToHost, set_ptr->streams[_a_]));
    MPI_Barrier(MPI_COMM_WORLD);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    T g = host_tmp.real();
    MPI_Allreduce(MPI_IN_PLACE, &g, 1, mpitype<T>(), MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    return g;
  }

  void parse_params() {
    num_levels = set_ptr->host_params[_MG_NUM_LEVEL_];
    if (num_levels < 1) num_levels = 1;
    if (num_levels > 8) num_levels = 8;

    levels = new MgLevelState<T>[num_levels];
    level_iters = new int[num_levels]();
    level_times = new double[num_levels]();

    levels[0].dof = _LAT_SC_;
    levels[0].X = set_ptr->host_params[_LAT_X_];
    levels[0].Y = set_ptr->host_params[_LAT_Y_];
    levels[0].Z = set_ptr->host_params[_LAT_Z_];
    levels[0].Lt = set_ptr->host_params[_LAT_T_];
    levels[0].vol = levels[0].X * levels[0].Y * levels[0].Z * levels[0].Lt;
    levels[0].vec_sz = (size_t)levels[0].dof * levels[0].vol;

    int offsets[] = { _MG_LEVEL1_E_, _MG_LEVEL1_X_, _MG_LEVEL1_Y_, _MG_LEVEL1_Z_, _MG_LEVEL1_T_,
                      _MG_LEVEL1_MAX_ITER_ };
    for (int i = 1; i < num_levels; i++) {
      int base = (i - 1) * _MG_PARAMS_SIZE_;
      levels[i].dof = set_ptr->host_params[offsets[0] + base];
      levels[i].X   = set_ptr->host_params[offsets[1] + base];
      levels[i].Y   = set_ptr->host_params[offsets[2] + base];
      levels[i].Z   = set_ptr->host_params[offsets[3] + base];
      levels[i].Lt  = set_ptr->host_params[offsets[4] + base];
      if (levels[i].dof <= 0) levels[i].dof = 24;
      if (levels[i].X <= 0) levels[i].X = levels[i-1].X / 2;
      if (levels[i].Y <= 0) levels[i].Y = levels[i-1].Y / 2;
      if (levels[i].Z <= 0) levels[i].Z = levels[i-1].Z / 2;
      if (levels[i].Lt <= 0) levels[i].Lt = levels[i-1].Lt / 2;
      levels[i].vol = levels[i].X * levels[i].Y * levels[i].Z * levels[i].Lt;
      levels[i].vec_sz = (size_t)levels[i].dof * levels[i].vol;
      levels[i].alloc(levels[i].dof, levels[i].X, levels[i].Y, levels[i].Z, levels[i].Lt,
                      set_ptr->stream);
    }

    max_iter = set_ptr->host_params[_MAX_ITER_];
    atol = set_ptr->host_argv[_ATOL_];
    kappa_val = set_ptr->kappa();
    num_restart = 3;

    for (int d = 0; d < 4; d++) mg_grid[d] = 2;

    null_vecs = new void*[num_levels];
    hop_packed = new void*[num_levels];
    sit_packed = new void*[num_levels];
    for (int i = 0; i < num_levels; i++)
      null_vecs[i] = hop_packed[i] = sit_packed[i] = nullptr;

    if (rank == 0) {
      std::ostringstream oss;
      oss << "PYQCU::SOLVER::MULTIGRID::\n self.dof_list:[";
      for (int i = 0; i < num_levels; i++) {
        if (i > 0) oss << ", "; oss << levels[i].dof;
      }
      oss << "]\n self.lat_size_list:[";
      for (int i = 0; i < num_levels; i++) {
        if (i > 0) oss << ", ";
        oss << "[" << levels[i].X << ", " << levels[i].Y << ", " << levels[i].Z << ", " << levels[i].Lt << "]";
      }
      oss << "]\n num_restart:" << num_restart << "\n tol:" << std::scientific << atol
          << "\n max_iter:" << max_iter;
      log_printf<T>(oss.str(), rank, true);
    }
    solve_time_ms = 0;
  }

  void set_coarse_ops(int fine_lev, void *nv, void *hp, void *sp) {
    if (fine_lev >= 0 && fine_lev < num_levels - 1) {
      null_vecs[fine_lev] = nv;
      hop_packed[fine_lev] = hp;
      sit_packed[fine_lev] = sp;
    }
  }

  // ---- V-cycle for coarse levels ----
  T v_cycle(int lev) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto &st = levels[lev];

    if (rank == 0 && verbose) {
      log_printf<T>("PYQCU::SOLVER::MULTIGRID::\n " + std::to_string(lev) +
          ":Norm of b:" + std::to_string(sqrt(norm2_coarse(st.b, lev))), rank, true);
      log_printf<T>("PYQCU::SOLVER::MULTIGRID::\n " + std::to_string(lev) +
          ":Norm of r:" + std::to_string(sqrt(norm2_coarse(st.b, lev))), rank, true);
      log_printf<T>("PYQCU::SOLVER::MULTIGRID::\n " + std::to_string(lev) +
          ":Norm of x0:0.000000", rank, true);
      log_printf<T>("PYQCU::SOLVER::MULTIGRID::\n " + std::to_string(lev) +
          ":Starting Iterations", rank, true);
    }

    // Initialize: x=0, r=b, r_tilde=r, p=v=0
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    zero_coarse(st.x, lev);
    copy_coarse(st.r, st.b, lev);
    copy_coarse(st.r_tilde, st.r, lev);
    zero_coarse(st.p, lev);
    zero_coarse(st.v, lev);
    st.rho = st.rho_prev = (T)1.0;
    st.alpha = st.omega = (T)1.0;

    T r_norm = 0;
    int max_smooth = 4;
    if (lev == num_levels - 1) max_smooth = 8;

    // Pre-smoothing
    int i;
    for (i = 0; i < max_smooth; i++) {
      auto ti0 = std::chrono::high_resolution_clock::now();
      r_norm = bistabcg_iter(lev);
      auto ti1 = std::chrono::high_resolution_clock::now();
      double sec = std::chrono::duration<double>(ti1 - ti0).count();
      if (rank == 0 && verbose) {
        std::ostringstream bmsg, fmsg;
        bmsg << "PYQCU::SOLVER::MULTIGRID::\n B-" << lev << "-bistabcg-Iteration " << i
             << ": Residual = " << std::scientific << r_norm;
        log_printf<T>(bmsg.str(), rank, true);
        fmsg << "PYQCU::SOLVER::MULTIGRID::\n F-" << lev << "-bistabcg-Iteration " << i
             << ": Residual = " << std::scientific << r_norm << ", Time = " << std::fixed
             << std::setprecision(6) << sec << " s";
        log_printf<T>(fmsg.str(), rank, true);
      }
    }

    // Coarse correction
    if (lev < num_levels - 1) {
      coarse_dslash_op(set_ptr->device_vec0, st.x, lev);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      int total = (int)st.vec_sz;
      dim3 gco((total + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
      give_copy_vals<T><<<gco, _BLOCK_SIZE_, 0, set_ptr->stream>>>(set_ptr->device_vec2, st.b);
      bistabcg_give_diff2<T><<<gco, _BLOCK_SIZE_, 0, set_ptr->stream>>>(
          set_ptr->device_vec2, set_ptr->device_vec0, st.r, set_ptr->device_vals);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));

      restrict_op(levels[lev+1].b, st.r, lev);
      zero_coarse(levels[lev+1].x, lev + 1);
      v_cycle(lev + 1);

      prolong_op(set_ptr->device_vec0, levels[lev+1].x, lev);
      axpy_coarse(st.x, (T)1.0, set_ptr->device_vec0, lev);

      // Reset BiStabCG state for post-smoothing
      coarse_dslash_op(set_ptr->device_vec0, st.x, lev);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      give_copy_vals<T><<<gco, _BLOCK_SIZE_, 0, set_ptr->stream>>>(set_ptr->device_vec2, st.b);
      bistabcg_give_diff2<T><<<gco, _BLOCK_SIZE_, 0, set_ptr->stream>>>(
          set_ptr->device_vec2, set_ptr->device_vec0, st.r, set_ptr->device_vals);
      copy_coarse(st.r_tilde, st.r, lev);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    }

    // Post-smoothing
    int post_smooth = max_smooth / 2;
    if (post_smooth < 1) post_smooth = 1;
    for (int j = 0; j < post_smooth; j++) {
      auto ti0 = std::chrono::high_resolution_clock::now();
      r_norm = bistabcg_iter(lev);
      auto ti1 = std::chrono::high_resolution_clock::now();
      double sec = std::chrono::duration<double>(ti1 - ti0).count();
      if (rank == 0 && verbose) {
        int idx = max_smooth + j;
        std::ostringstream bmsg, fmsg;
        bmsg << "PYQCU::SOLVER::MULTIGRID::\n B-" << lev << "-bistabcg-Iteration " << idx
             << ": Residual = " << std::scientific << r_norm;
        log_printf<T>(bmsg.str(), rank, true);
        fmsg << "PYQCU::SOLVER::MULTIGRID::\n F-" << lev << "-bistabcg-Iteration " << idx
             << ": Residual = " << std::scientific << r_norm << ", Time = " << std::fixed
             << std::setprecision(6) << sec << " s";
        log_printf<T>(fmsg.str(), rank, true);
      }
    }

    if (rank == 0 && verbose) {
      log_printf<T>("PYQCU::SOLVER::MULTIGRID::\n Converged at iteration " +
          std::to_string(max_smooth + post_smooth - 1) + " with residual " +
          std::to_string(r_norm), rank, true);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    level_times[lev] += std::chrono::duration<double, std::milli>(t1 - t0).count();
    return r_norm;
  }

  // ---- Initialize ----
  void init(void *_fo, void *_fi, void *_g, void *_ce, void *_co,
            void *_cei, void *_coi) {
    fermion_out_eo = _fo; fermion_in_eo = _fi;
    gauge = _g; clover_ee = _ce; clover_oo = _co;
    clover_ee_inv = _cei; clover_oo_inv = _coi;
    clover_dslash_ee.init(clover_ee);
    clover_dslash_oo.init(clover_oo);
    clover_dslash_ee_inv.init(clover_ee_inv);
    clover_dslash_oo_inv.init(clover_oo_inv);
    parse_params();

    b_e = fermion_in_eo;
    b_o = static_cast<LatticeComplex<T>*>(fermion_in_eo) + set_ptr->lat_4dim_SC;
    x_o = static_cast<LatticeComplex<T>*>(fermion_out_eo) + set_ptr->lat_4dim_SC;
    checkCudaErrors(cudaMemsetAsync(x_o, 0, set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>),
        set_ptr->stream));

    size_t sc = set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>);
    checkCudaErrors(cudaMallocAsync(&b__o, sc, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&r0, sc, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&rt0, sc, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&p0, sc, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&v0, sc, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&s0, sc, set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&t0, sc, set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));

    levels[0].x = x_o;   levels[0].b = b__o;
    levels[0].r = r0;    levels[0].r_tilde = rt0;
    levels[0].p = p0;    levels[0].v = v0;
    levels[0].s = s0;    levels[0].t = t0;
    levels[0].owned = false;

    setup_b__o();

    if (rank == 0)
      log_printf<T>("PYQCU::QCU::MULTIGRID::\n MG_INIT_COMPLETE: Solver ready", rank, true);
  }

  // ---- Main solve: BiStabCG with V-cycle corrections ----
  void run() {
    auto solve_t0 = std::chrono::high_resolution_clock::now();
    auto &st = levels[0];

    if (rank == 0) {
      log_printf<T>("PYQCU::SOLVER::MULTIGRID::\n 0:Norm of b:" +
          std::to_string(sqrt(fine_norm2(b__o))), rank, true);
      log_printf<T>("PYQCU::SOLVER::MULTIGRID::\n 0:Norm of r:" +
          std::to_string(sqrt(fine_norm2(b__o))), rank, true);
      log_printf<T>("PYQCU::SOLVER::MULTIGRID::\n 0:Norm of x0:0.000000", rank, true);
      log_printf<T>("PYQCU::SOLVER::MULTIGRID::\n 0:Starting Iterations", rank, true);
    }

    // One-time init: x=0, r=b, r_tilde=r, p=v=0
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(cudaMemsetAsync(x_o, 0, set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>), set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    copy_coarse(st.r, st.b, 0);
    copy_coarse(st.r_tilde, st.r, 0);
    zero_coarse(st.p, 0);
    zero_coarse(st.v, 0);
    st.rho = st.rho_prev = (T)1.0;
    st.alpha = st.omega = (T)1.0;

    T r_norm = 0;
    int total_iters = 0;
    double total_time = 0;

    for (int iter = 0; iter < max_iter; iter++) {
      auto ti0 = std::chrono::high_resolution_clock::now();
      r_norm = bistabcg_iter(0);
      auto ti1 = std::chrono::high_resolution_clock::now();
      double sec = std::chrono::duration<double>(ti1 - ti0).count();
      total_time += sec; total_iters++;

      if (rank == 0 && verbose) {
        std::ostringstream bmsg, fmsg;
        bmsg << "PYQCU::SOLVER::MULTIGRID::\n B-0-bistabcg-Iteration " << iter
             << ": Residual = " << std::scientific << r_norm;
        log_printf<T>(bmsg.str(), rank, true);
        fmsg << "PYQCU::SOLVER::MULTIGRID::\n F-0-bistabcg-Iteration " << iter
             << ": Residual = " << std::scientific << r_norm << ", Time = " << std::fixed
             << std::setprecision(6) << sec << " s";
        log_printf<T>(fmsg.str(), rank, true);
      }

      conv_history.push_back(r_norm);

      // Periodic coarse correction
      if (num_levels > 1 && (iter + 1) % num_restart == 0) {
        fine_dslash_op(set_ptr->device_vec0, st.x);
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
        dim3 gf = set_ptr->gridDim, blkf = set_ptr->blockDim;
        give_copy_vals<T><<<gf, blkf, 0, set_ptr->stream>>>(set_ptr->device_vec2, st.b);
        bistabcg_give_diff2<T><<<gf, blkf, 0, set_ptr->stream>>>(
            set_ptr->device_vec2, set_ptr->device_vec0, st.r, set_ptr->device_vals);
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));

        restrict_op(levels[1].b, st.r, 0);
        zero_coarse(levels[1].x, 1);
        v_cycle(1);

        prolong_op(set_ptr->device_vec0, levels[1].x, 0);
        give_copy_vals<T><<<gf, blkf, 0, set_ptr->stream>>>(set_ptr->device_vec1, st.x);
        LatticeComplex<T> one(1.0, 0.0);
        CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH, set_ptr->lat_4dim_SC,
            &one, set_ptr->device_vec0, 1, set_ptr->device_vec1, 1));
        give_copy_vals<T><<<gf, blkf, 0, set_ptr->stream>>>(st.x, set_ptr->device_vec1);

        // Reset BiStabCG state
        fine_dslash_op(set_ptr->device_vec0, st.x);
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
        give_copy_vals<T><<<gf, blkf, 0, set_ptr->stream>>>(set_ptr->device_vec2, st.b);
        bistabcg_give_diff2<T><<<gf, blkf, 0, set_ptr->stream>>>(
            set_ptr->device_vec2, set_ptr->device_vec0, st.r, set_ptr->device_vals);
        copy_coarse(st.r_tilde, st.r, 0);
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));

        T rn_post = sqrt(fine_norm2(st.r));
        conv_history.push_back(rn_post);
      }

      if (r_norm < atol) {
        if (rank == 0 && verbose) {
          log_printf<T>("PYQCU::SOLVER::MULTIGRID::\n Converged at iteration " +
              std::to_string(iter) + " with residual " + std::to_string(r_norm), rank, true);
        }
        break;
      }
    }

    recover_x_e();

    auto solve_t1 = std::chrono::high_resolution_clock::now();
    solve_time_ms = std::chrono::duration<double, std::milli>(solve_t1 - solve_t0).count();

    if (rank == 0) {
      double avg_time = total_iters > 0 ? total_time / total_iters : 0;
      std::ostringstream perf;
      perf << "PYQCU::SOLVER::MULTIGRID::\n Performance Statistics:";
      log_printf<T>(perf.str(), rank, true);
      log_printf<T>("PYQCU::SOLVER::MULTIGRID::\n Total iterations: " +
          std::to_string(total_iters), rank, true);
      std::ostringstream tmsg;
      tmsg << "PYQCU::SOLVER::MULTIGRID::\n Total time: " << std::fixed
           << std::setprecision(6) << (solve_time_ms / 1000.0) << " seconds";
      log_printf<T>(tmsg.str(), rank, true);
      std::ostringstream amsg;
      amsg << "PYQCU::SOLVER::MULTIGRID::\n Average time per iteration: " << std::fixed
           << std::setprecision(6) << avg_time << " s";
      log_printf<T>(amsg.str(), rank, true);
      std::ostringstream fmsg;
      fmsg << "PYQCU::SOLVER::MULTIGRID::\n Final residual: " << std::scientific << r_norm;
      log_printf<T>(fmsg.str(), rank, true);

      std::ostringstream ch;
      ch << "CONVERGENCE_HISTORY: [";
      for (size_t j = 0; j < conv_history.size(); j++) {
        if (j > 0) ch << ",";
        ch << std::scientific << conv_history[j];
      }
      ch << "]";
      log_printf<T>(ch.str(), rank, false); // don't print to stdout (huge)
    }
  }

  void run_test() {
    auto t0 = std::chrono::high_resolution_clock::now();
    run();
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    fine_dslash_op(set_ptr->device_vec1, x_o);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    T norm_b = sqrt(fine_norm2(b__o));

    dim3 g = set_ptr->gridDim, blk = set_ptr->blockDim;
    give_copy_vals<T><<<g, blk, 0, set_ptr->stream>>>(set_ptr->device_vec2, set_ptr->device_vec1);
    bistabcg_give_diff2<T><<<g, blk, 0, set_ptr->stream>>>(
        set_ptr->device_vec2, b__o, set_ptr->device_vec1, set_ptr->device_vals);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    T diff_norm = sqrt(fine_norm2(set_ptr->device_vec1));
    T rel_diff = (norm_b > (T)1e-30) ? diff_norm / norm_b : diff_norm;

    if (rank == 0) {
      printf("=== MULTIGRID SOLVER REPORT ===\n");
      printf("Total time: %.3f ms (%.3f s)\n", total_ms, total_ms / 1000.0);
      printf("Solve time: %.3f ms\n", solve_time_ms);
      printf("Convergence history entries: %zu\n", conv_history.size());
      if (!conv_history.empty()) {
        printf("Initial residual: %.6e\n", conv_history[0]);
        printf("Final residual:   %.6e\n", conv_history.back());
      }
      printf("Relative residual |D*x - b|/|b|: %.6e\n", rel_diff);
    }

    set_ptr->err = cudaGetLastError();
    checkCudaErrors(set_ptr->err);
  }

  void end() {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

    auto fre = [&](void *&p) { if (p) { cudaFreeAsync(p, set_ptr->stream); p = nullptr; } };
    fre(b__o); fre(r0); fre(rt0); fre(p0); fre(v0); fre(s0); fre(t0);

    for (int i = 1; i < num_levels; i++)
      levels[i].free_all(set_ptr->stream);

    delete[] levels; levels = nullptr;
    delete[] level_iters; level_iters = nullptr;
    delete[] level_times; level_times = nullptr;
    delete[] null_vecs; null_vecs = nullptr;
    delete[] hop_packed; hop_packed = nullptr;
    delete[] sit_packed; sit_packed = nullptr;

    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
};

} // namespace qcu
#endif
