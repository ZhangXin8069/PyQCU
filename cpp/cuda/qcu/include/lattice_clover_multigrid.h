#ifndef _LATTICE_CLOVER_MULTIGRID_H
#define _LATTICE_CLOVER_MULTIGRID_H
/**
 * @file lattice_clover_multigrid.h
 * @brief Multi-threaded, multi-precision CUDA C++ Multigrid solver with BiStabCG smoothing.
 * Algorithm: pyqcu/solver/_multigrid.py.  Target API: applyCloverBistabCgDslashQcu.
 * Sync pattern matches LatticeCloverBistabCg::_run() exactly for maximum performance.
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

template <typename T> struct MgLevelState {
  void *x, *rhs, *r, *r_tilde, *p, *v, *s, *t;
  int dof, X, Y, Z, Lt, vol;
  size_t vec_sz;
  bool owned;
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

template <typename T> struct LatticeCloverMultigrid {
  LatticeSet<T> *set_ptr;
  LatticeWilsonDslash<T> wilson_dslash;
  LatticeCloverDslash<T> clover_dslash_ee, clover_dslash_oo;
  LatticeCloverDslash<T> clover_dslash_ee_inv, clover_dslash_oo_inv;

  void *gauge, *clover_ee, *clover_oo, *clover_ee_inv, *clover_oo_inv;
  void *fermion_out_eo, *fermion_in_eo;
  void *b_e, *b_o, *x_o;
  void *b__o, *r0, *rt0, *p0, *v0, *s0, *t0;

  int num_levels, mg_grid_size[4];
  MgLevelState<T> *levels;
  void **null_vecs, **hop_packed, **sit_packed;

  int max_iter;
  T atol;
  int num_restart, rank;
  bool verbose;
  T kappa_val;

  // Host mirror of device_vals for convergence check (matches reference pattern)
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

  // ---- dslash ops (no internal syncs — iteration-level sync handles it) ----
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

  void coarse_dslash_op(void *out, void *in, int lev) {
    int E=levels[lev].dof, Xc=levels[lev].X, Yc=levels[lev].Y, Zc=levels[lev].Z, Lt=levels[lev].Lt;
    int t=E*Xc*Yc*Zc*Lt; dim3 g((t+_BLOCK_SIZE_-1)/_BLOCK_SIZE_);
    multigrid_coarse_dslash<T><<<g,_BLOCK_SIZE_,0,set_ptr->stream>>>(
        out,in,hop_packed[lev-1],sit_packed[lev-1],E,Xc,Yc,Zc,Lt);
  }

  void restrict_op(void *co, void *fi, int fl) {
    int l=fl+1, E=levels[l].dof, e=levels[fl].dof;
    int Xf=levels[fl].X,Yf=levels[fl].Y,Zf=levels[fl].Z,Ltf=levels[fl].Lt;
    int Xc=levels[l].X,Yc=levels[l].Y,Zc=levels[l].Z,Ltc=levels[l].Lt;
    int t=E*Xc*Yc*Zc*Ltc; dim3 g((t+_BLOCK_SIZE_-1)/_BLOCK_SIZE_);
    multigrid_restrict<T><<<g,_BLOCK_SIZE_,0,set_ptr->stream>>>(
        co,fi,null_vecs[fl],E,e,Xf,Yf,Zf,Ltf,Xc,Yc,Zc,Ltc);
  }

  void prolong_op(void *fo, void *ci, int fl) {
    int l=fl+1, E=levels[l].dof, e=levels[fl].dof;
    int Xf=levels[fl].X,Yf=levels[fl].Y,Zf=levels[fl].Z,Ltf=levels[fl].Lt;
    int Xc=levels[l].X,Yc=levels[l].Y,Zc=levels[l].Z,Ltc=levels[l].Lt;
    int t=e*Xf*Yf*Zf*Ltf; dim3 g((t+_BLOCK_SIZE_-1)/_BLOCK_SIZE_);
    multigrid_prolong<T><<<g,_BLOCK_SIZE_,0,set_ptr->stream>>>(
        fo,ci,null_vecs[fl],E,e,Xf,Yf,Zf,Ltf,Xc,Yc,Zc,Ltc);
  }

  // ====================================================================
  // Dot products — matching LatticeCloverBistabCg::_dot_mpi EXACTLY
  // ====================================================================

  /** MPI dot: _cublasDot→_send_tmp_ → D→H to host_vals[_send_tmp_] →
   *  sync → _MPI_Allreduce(_send_tmp_→vals_idx, 2 elem) →
   *  H→D to device_vals.  Result is in BOTH host_vals[vals_idx] AND
   *  device_vals[vals_idx]. */
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

  /** Coarse dot: same pattern, no MPI. Result in host_vals[vals_idx]. */
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

  // Vec helpers
  void zero_c(void *v,int l) {checkCudaErrors(cudaMemsetAsync(v,0,levels[l].vec_sz*sizeof(LatticeComplex<T>),set_ptr->stream));}
  void copy_c(void *d,void *s,int l) {int t=(int)levels[l].vec_sz; dim3 g((t+_BLOCK_SIZE_-1)/_BLOCK_SIZE_); give_copy_vals<T><<<g,_BLOCK_SIZE_,0,set_ptr->stream>>>(d,s);}
  void axpy_c(void *y,T a,void *x,int l) {LatticeComplex<T> al(a,0.0); CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH,levels[l].vec_sz,&al,x,1,y,1));}

  // ====================================================================
  // BiStabCG iteration — EXACT sync pattern of BistabCg::_run()
  //   In the reference, convergence is checked via host_vals[_norm2_tmp_]
  //   which was set by _dot(r,r,_norm2_tmp_,_c_) between give_p and dslash.
  //   We replicate that here: compute ||r||^2 after give_p, store in
  //   host_vals[_norm2_tmp_], caller checks it after iteration returns.
  // ====================================================================
  void bistabcg_iter(int lev) {
    auto &st=levels[lev];
    bool fine=(lev==0); cudaStream_t S=set_ptr->stream;
    dim3 gv,bv;
    if(fine){gv=set_ptr->gridDim;bv=set_ptr->blockDim;}
    else{int t=(int)st.vec_sz;gv=dim3((t+_BLOCK_SIZE_-1)/_BLOCK_SIZE_);bv=dim3(_BLOCK_SIZE_);}

    // Step 1: ρ = (r_tilde, r)           [stream _a_]
    if(fine) dot_mpi(st.r_tilde,st.r,_rho_,_a_);
    else     dot_coarse(st.r_tilde,st.r,lev,_rho_,_a_);

    // Step 2: β=(ρ/ρ_prev)*(α/ω)         [_a_];  ρ_prev←ρ [_b_]
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    bistabcg_give_1beta<T><<<1,1,0,set_ptr->streams[_a_]>>>(set_ptr->device_vals);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    bistabcg_give_1rho_prev<T><<<1,1,0,set_ptr->streams[_b_]>>>(set_ptr->device_vals);

    // Step 3: p = r + β·(p−ω·v)          [_a_]
    bistabcg_give_p<T><<<gv,bv,0,set_ptr->streams[_a_]>>>(st.p,st.r,st.v,set_ptr->device_vals);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));

    // Step 3.5: convergence check — ||r||² → host_vals[_norm2_tmp_]  [_c_]
    // This is placed HERE (before dslash) matching the reference exactly.
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
    // NO bottom sync here — the NEXT iteration's first sync(_b_) handles it.
    // The exception is the FINAL iteration, which synced in run() after the loop.
  }

  // ====================================================================
  // Setup / recover
  // ====================================================================
  void setup_b__o() {
    give_copy_vals<T><<<set_ptr->gridDim,set_ptr->blockDim,0,set_ptr->stream>>>(
        set_ptr->device_vec2,b_e);
    clover_dslash_ee_inv.give(set_ptr->device_vec2);
    wilson_dslash.run_oe(set_ptr->device_vec0,set_ptr->device_vec2,gauge);
    bistabcg_give_b__o<T><<<set_ptr->gridDim,set_ptr->blockDim,0,set_ptr->stream>>>(
        b__o,b_o,set_ptr->device_vec0,kappa_val,set_ptr->device_vals);
  }

  void recover_x_e() {
    CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH,set_ptr->lat_4dim_SC*_REAL_IMAG_,(T*)b_e,1,(T*)set_ptr->device_vec0,1));
    wilson_dslash.run_eo(set_ptr->device_vec1,x_o,gauge);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    LatticeComplex<T> kap(kappa_val,0.0);
    CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH,set_ptr->lat_4dim_SC,&kap,set_ptr->device_vec1,1,set_ptr->device_vec0,1));
    clover_dslash_ee_inv.give(set_ptr->device_vec0);
    CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH,set_ptr->lat_4dim_SC*_REAL_IMAG_,(T*)set_ptr->device_vec0,1,(T*)fermion_out_eo,1));
  }

  // ====================================================================
  // Params parsing
  // ====================================================================
  void parse_params() {
    num_levels=set_ptr->host_params[_MG_NUM_LEVEL_];
    if(num_levels<1)num_levels=1; if(num_levels>8)num_levels=8;
    levels=new MgLevelState<T>[num_levels];

    levels[0].dof=_LAT_SC_;
    levels[0].X=set_ptr->host_params[_LAT_X_]; levels[0].Y=set_ptr->host_params[_LAT_Y_];
    levels[0].Z=set_ptr->host_params[_LAT_Z_]; levels[0].Lt=set_ptr->host_params[_LAT_T_];
    levels[0].vol=levels[0].X*levels[0].Y*levels[0].Z*levels[0].Lt;
    levels[0].vec_sz=(size_t)levels[0].dof*levels[0].vol;

    static const int oE=_MG_LEVEL1_E_,oX=_MG_LEVEL1_X_,oY=_MG_LEVEL1_Y_;
    static const int oZ=_MG_LEVEL1_Z_,oL=_MG_LEVEL1_T_;
    for(int i=1;i<num_levels;i++){
      int b=(i-1)*_MG_PARAMS_SIZE_;
      levels[i].dof=set_ptr->host_params[oE+b]; levels[i].X=set_ptr->host_params[oX+b];
      levels[i].Y=set_ptr->host_params[oY+b]; levels[i].Z=set_ptr->host_params[oZ+b];
      levels[i].Lt=set_ptr->host_params[oL+b];
      if(levels[i].dof<=0)levels[i].dof=24;
      if(levels[i].X<=0)  levels[i].X=levels[i-1].X/2;
      if(levels[i].Y<=0)  levels[i].Y=levels[i-1].Y/2;
      if(levels[i].Z<=0)  levels[i].Z=levels[i-1].Z/2;
      if(levels[i].Lt<=0) levels[i].Lt=levels[i-1].Lt/2;
      levels[i].vol=levels[i].X*levels[i].Y*levels[i].Z*levels[i].Lt;
      levels[i].vec_sz=(size_t)levels[i].dof*levels[i].vol;
      levels[i].alloc(levels[i].dof,levels[i].X,levels[i].Y,levels[i].Z,levels[i].Lt,set_ptr->stream);
    }

    max_iter=set_ptr->host_params[_MAX_ITER_];
    atol=set_ptr->host_argv[_ATOL_];
    kappa_val=set_ptr->kappa();
    num_restart=3;
    for(int d=0;d<4;d++)mg_grid_size[d]=2;
    if(num_levels>1){
      if(levels[0].X>0&&levels[1].X>0)mg_grid_size[0]=levels[0].X/levels[1].X;
      if(levels[0].Y>0&&levels[1].Y>0)mg_grid_size[1]=levels[0].Y/levels[1].Y;
      if(levels[0].Z>0&&levels[1].Z>0)mg_grid_size[2]=levels[0].Z/levels[1].Z;
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
         <<"\n max_iter:"<<max_iter;
      log_write<T>(oss.str(),rank,true);
    }
    solve_time_ms=0; for(int i=0;i<8;i++)level_times[i]=0;
  }

  void set_coarse_ops(int fl,void*nv,void*hp,void*sp){
    if(fl>=0&&fl<num_levels-1){null_vecs[fl]=nv;hop_packed[fl]=hp;sit_packed[fl]=sp;}
  }

  // ====================================================================
  // V-cycle for coarse levels
  // ====================================================================
  T v_cycle(int lev) {
    auto&st=levels[lev]; cudaStream_t S=set_ptr->stream;
    if(rank==0&&verbose){
      size_t n=st.vec_sz; LatticeComplex<T>*dv=static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
      CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasHs[_a_],n,st.rhs,1,st.rhs,1,&dv[_send_tmp_]));
      LatticeComplex<T> ht; checkCudaErrors(cudaMemcpyAsync(&ht,&dv[_send_tmp_],sizeof(LatticeComplex<T>),cudaMemcpyDeviceToHost,set_ptr->streams[_a_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_])); T nb=sqrt(ht.real());
      log_write<T>("PYQCU::SOLVER::MULTIGRID::\n "+std::to_string(lev)+":Norm of b:"+std::to_string(nb),rank,true);
      log_write<T>("PYQCU::SOLVER::MULTIGRID::\n "+std::to_string(lev)+":Norm of r:"+std::to_string(nb),rank,true);
      log_write<T>("PYQCU::SOLVER::MULTIGRID::\n "+std::to_string(lev)+":Norm of x0:0.000000",rank,true);
      log_write<T>("PYQCU::SOLVER::MULTIGRID::\n "+std::to_string(lev)+":Starting Iterations",rank,true);
    }

    // One-time TOP sync (matches reference _run() before the for loop)
    checkCudaErrors(cudaStreamSynchronize(S));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

    // Init: x=0, r=b, r_tilde=r, p=v=0
    zero_c(st.x,lev); copy_c(st.r,st.rhs,lev); copy_c(st.r_tilde,st.r,lev);
    zero_c(st.p,lev); zero_c(st.v,lev);
    checkCudaErrors(cudaStreamSynchronize(S));
    // Set device_vals to initial BiStabCG scalars
    LatticeComplex<T>*dv=static_cast<LatticeComplex<T>*>(set_ptr->device_vals);
    LatticeComplex<T> one(1,0),z(0,0);
    checkCudaErrors(cudaMemcpyAsync(&dv[_rho_],&z,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
    checkCudaErrors(cudaMemcpyAsync(&dv[_rho_prev_],&one,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
    checkCudaErrors(cudaMemcpyAsync(&dv[_alpha_],&one,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
    checkCudaErrors(cudaMemcpyAsync(&dv[_omega_],&one,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
    checkCudaErrors(cudaStreamSynchronize(S));

    int ns=4; if(lev==num_levels-1)ns=8; T rn=0;
    // Pre-smoothing
    for(int i=0;i<ns;i++){
      auto t0=std::chrono::high_resolution_clock::now();
      bistabcg_iter(lev);
      auto t1=std::chrono::high_resolution_clock::now();
      double sec=std::chrono::duration<double>(t1-t0).count();
      rn=sqrt(host_vals[_norm2_tmp_].real());
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

    // Coarse correction
    if(lev<num_levels-1){
      checkCudaErrors(cudaStreamSynchronize(S));
      coarse_dslash_op(set_ptr->device_vec0,st.x,lev);
      checkCudaErrors(cudaStreamSynchronize(S));
      int nt=(int)st.vec_sz; dim3 gc((nt+_BLOCK_SIZE_-1)/_BLOCK_SIZE_);
      give_copy_vals<T><<<gc,_BLOCK_SIZE_,0,S>>>(set_ptr->device_vec2,st.rhs);
      bistabcg_give_diff2<T><<<gc,_BLOCK_SIZE_,0,S>>>(set_ptr->device_vec2,set_ptr->device_vec0,st.r,set_ptr->device_vals);
      checkCudaErrors(cudaStreamSynchronize(S));
      restrict_op(levels[lev+1].rhs,st.r,lev); zero_c(levels[lev+1].x,lev+1);
      checkCudaErrors(cudaStreamSynchronize(S));
      v_cycle(lev+1);
      prolong_op(set_ptr->device_vec0,levels[lev+1].x,lev);
      checkCudaErrors(cudaStreamSynchronize(S));
      axpy_c(st.x,(T)1.0,set_ptr->device_vec0,lev);
      checkCudaErrors(cudaStreamSynchronize(S));
      // Reset state: r = b - D*x + shadow
      coarse_dslash_op(set_ptr->device_vec0,st.x,lev);
      checkCudaErrors(cudaStreamSynchronize(S));
      give_copy_vals<T><<<gc,_BLOCK_SIZE_,0,S>>>(set_ptr->device_vec2,st.rhs);
      bistabcg_give_diff2<T><<<gc,_BLOCK_SIZE_,0,S>>>(set_ptr->device_vec2,set_ptr->device_vec0,st.r,set_ptr->device_vals);
      copy_c(st.r_tilde,st.r,lev); checkCudaErrors(cudaStreamSynchronize(S));
    }

    // Post-smoothing
    int np=ns/2; if(np<1)np=1;
    for(int j=0;j<np;j++){
      auto t0=std::chrono::high_resolution_clock::now();
      bistabcg_iter(lev);
      auto t1=std::chrono::high_resolution_clock::now();
      double sec=std::chrono::duration<double>(t1-t0).count();
      rn=sqrt(host_vals[_norm2_tmp_].real());
      if(rank==0&&verbose){
        int idx=ns+j;
        std::ostringstream bm,fm;
        bm<<"PYQCU::SOLVER::MULTIGRID::\n B-"<<lev<<"-bistabcg-Iteration "<<idx
          <<": Residual = "<<std::scientific<<rn;
        log_write<T>(bm.str(),rank,true);
        fm<<"PYQCU::SOLVER::MULTIGRID::\n F-"<<lev<<"-bistabcg-Iteration "<<idx
          <<": Residual = "<<std::scientific<<rn<<", Time = "<<std::fixed<<std::setprecision(6)<<sec<<" s";
        log_write<T>(fm.str(),rank,true);
      }
    }
    if(rank==0&&verbose)
      log_write<T>("PYQCU::SOLVER::MULTIGRID::\n Converged at iteration "+
        std::to_string(ns+np-1)+" with residual "+std::to_string(rn),rank,true);
    return rn;
  }

  // ====================================================================
  // Init
  // ====================================================================
  void init(void*_fo,void*_fi,void*_g,void*_ce,void*_co,void*_cei,void*_coi){
    fermion_out_eo=_fo;fermion_in_eo=_fi;gauge=_g;
    clover_ee=_ce;clover_oo=_co;clover_ee_inv=_cei;clover_oo_inv=_coi;
    clover_dslash_ee.init(clover_ee);clover_dslash_oo.init(clover_oo);
    clover_dslash_ee_inv.init(clover_ee_inv);clover_dslash_oo_inv.init(clover_oo_inv);
    parse_params();
    b_e=fermion_in_eo;b_o=static_cast<LatticeComplex<T>*>(fermion_in_eo)+set_ptr->lat_4dim_SC;
    x_o=static_cast<LatticeComplex<T>*>(fermion_out_eo)+set_ptr->lat_4dim_SC;
    checkCudaErrors(cudaMemsetAsync(x_o,0,set_ptr->lat_4dim_SC*sizeof(LatticeComplex<T>),set_ptr->stream));
    size_t sc=set_ptr->lat_4dim_SC*sizeof(LatticeComplex<T>);
    checkCudaErrors(cudaMallocAsync(&b__o,sc,set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&r0,sc,set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&rt0,sc,set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&p0,sc,set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&v0,sc,set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&s0,sc,set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&t0,sc,set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    levels[0].x=x_o;levels[0].rhs=b__o;levels[0].r=r0;levels[0].r_tilde=rt0;
    levels[0].p=p0;levels[0].v=v0;levels[0].s=s0;levels[0].t=t0;levels[0].owned=false;
    setup_b__o();checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    if(rank==0)log_write<T>("PYQCU::QCU::MULTIGRID::\n MG_INIT_COMPLETE: Solver ready",rank,true);
  }

  // ====================================================================
  // Main solve — tight BiStabCG loop matching reference sync pattern
  // ====================================================================
  void run() {
    auto t0=std::chrono::high_resolution_clock::now();
    auto&st=levels[0]; cudaStream_t S=set_ptr->stream;

    if(rank==0){
      // Compute ||b|| for logging
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

    // ---- ONE-TIME initial sync (matches reference _run() before for loop) ----
    checkCudaErrors(cudaStreamSynchronize(S));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

    // ---- Init state: x=0, r=b, r_tilde=r, p=v=0 ----
    checkCudaErrors(cudaMemsetAsync(x_o,0,set_ptr->lat_4dim_SC*sizeof(LatticeComplex<T>),S));
    checkCudaErrors(cudaStreamSynchronize(S));
    copy_c(st.r,st.rhs,0);copy_c(st.r_tilde,st.r,0);zero_c(st.p,0);zero_c(st.v,0);
    checkCudaErrors(cudaStreamSynchronize(S));
    // Set device_vals for first iteration
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
    for(int it=0;it<max_iter;it++){
      auto ti0=std::chrono::high_resolution_clock::now();
      bistabcg_iter(0);
      auto ti1=std::chrono::high_resolution_clock::now();
      double sec=std::chrono::duration<double>(ti1-ti0).count();tti+=sec;total++;

      // Convergence check from host_vals[_norm2_tmp_] (lazy, lagged by 1 iter)
      // This reads the ||r||² that was computed INSIDE bistabcg_iter at step 3.5.
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
        copy_c(st.r,st.rhs,0);copy_c(st.r_tilde,st.r,0);zero_c(st.p,0);zero_c(st.v,0);
        checkCudaErrors(cudaMemcpyAsync(&dv[_rho_],&z,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
        checkCudaErrors(cudaMemcpyAsync(&dv[_rho_prev_],&one,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
        checkCudaErrors(cudaMemcpyAsync(&dv[_alpha_],&one,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
        checkCudaErrors(cudaMemcpyAsync(&dv[_omega_],&one,sizeof(LatticeComplex<T>),cudaMemcpyHostToDevice,S));
        checkCudaErrors(cudaStreamSynchronize(S));
        continue;
      }

      // Convergence check
      if(rn2<atol2){
        if(rank==0&&verbose)
          log_write<T>("PYQCU::SOLVER::MULTIGRID::\n Converged at iteration "+
            std::to_string(it)+" with residual "+std::to_string(rn),rank,true);
        break;
      }

      // V-cycle correction
      if(num_levels>1&&num_restart>0&&(it+1)%num_restart==0){
        checkCudaErrors(cudaStreamSynchronize(S));
        fine_dslash_op(set_ptr->device_vec0,st.x);
        checkCudaErrors(cudaStreamSynchronize(S));
        dim3 gf=set_ptr->gridDim,bf=set_ptr->blockDim;
        give_copy_vals<T><<<gf,bf,0,S>>>(set_ptr->device_vec2,st.rhs);
        bistabcg_give_diff2<T><<<gf,bf,0,S>>>(set_ptr->device_vec2,set_ptr->device_vec0,st.r,set_ptr->device_vals);
        checkCudaErrors(cudaStreamSynchronize(S));
        restrict_op(levels[1].rhs,st.r,0);zero_c(levels[1].x,1);
        checkCudaErrors(cudaStreamSynchronize(S));
        v_cycle(1);
        prolong_op(set_ptr->device_vec0,levels[1].x,0);
        checkCudaErrors(cudaStreamSynchronize(S));
        give_copy_vals<T><<<gf,bf,0,S>>>(set_ptr->device_vec1,st.x);
        LatticeComplex<T> oc(1,0);CUBLAS_CHECK(_cublasAxpy<T>(set_ptr->cublasH,set_ptr->lat_4dim_SC,&oc,set_ptr->device_vec0,1,set_ptr->device_vec1,1));
        give_copy_vals<T><<<gf,bf,0,S>>>(st.x,set_ptr->device_vec1);
        checkCudaErrors(cudaStreamSynchronize(S));
        fine_dslash_op(set_ptr->device_vec0,st.x);
        checkCudaErrors(cudaStreamSynchronize(S));
        give_copy_vals<T><<<gf,bf,0,S>>>(set_ptr->device_vec2,st.rhs);
        bistabcg_give_diff2<T><<<gf,bf,0,S>>>(set_ptr->device_vec2,set_ptr->device_vec0,st.r,set_ptr->device_vals);
        copy_c(st.r_tilde,st.r,0);checkCudaErrors(cudaStreamSynchronize(S));
        // Record post-correction residual
        CUBLAS_CHECK(_cublasDot<T>(set_ptr->cublasHs[_a_],set_ptr->lat_4dim_SC,st.r,1,st.r,1,&dv[_send_tmp_]));
        LatticeComplex<T> ht;
        checkCudaErrors(cudaMemcpyAsync(&ht,&dv[_send_tmp_],sizeof(LatticeComplex<T>),cudaMemcpyDeviceToHost,set_ptr->streams[_a_]));
        MPI_Barrier(MPI_COMM_WORLD);checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
        T gr=ht.real();MPI_Allreduce(MPI_IN_PLACE,&gr,1,mpi_real_type<T>(),MPI_SUM,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);conv_history.push_back(sqrt(gr));
        if(rank==0&&verbose)
          log_write<T>("PYQCU::SOLVER::MULTIGRID::\n V-cyc corr at "+std::to_string(it),rank,true);
      }
    }

    // ---- Final sync (matches reference after for loop) ----
    checkCudaErrors(cudaStreamSynchronize(S));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));

    recover_x_e();checkCudaErrors(cudaStreamSynchronize(S));
    auto t1=std::chrono::high_resolution_clock::now();
    solve_time_ms=std::chrono::duration<double,std::milli>(t1-t0).count();

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
      printf("Convergence history entries: %zu\n",conv_history.size());
      if(!conv_history.empty()){printf("Initial residual: %.6e\n",conv_history[0]);
        printf("Final residual:   %.6e\n",conv_history.back());}
      printf("Relative residual |D*x - b|/|b|: %.6e\n",rd);
    }
    set_ptr->err=cudaGetLastError();checkCudaErrors(set_ptr->err);
  }

  void end() {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
    auto F=[&](void*&p){if(p){cudaFreeAsync(p,set_ptr->stream);p=nullptr;}};
    F(b__o);F(r0);F(rt0);F(p0);F(v0);F(s0);F(t0);
    for(int i=1;i<num_levels;i++)levels[i].free_all(set_ptr->stream);
    delete[] levels;levels=nullptr;delete[] null_vecs;null_vecs=nullptr;
    delete[] hop_packed;hop_packed=nullptr;delete[] sit_packed;sit_packed=nullptr;
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
};

} // namespace qcu
#endif
