#ifndef _LATTICE_CLOVER_BISTABCG_H
#define _LATTICE_CLOVER_BISTABCG_H
#include "./bistabcg.h"
#include "./lattice_mpi.h"
#include "./lattice_cuda.h"
#include "./lattice_clover_dslash.h"
#include "./lattice_clover_dslash.h"
namespace qcu
{
  // #define PRINT_MULTI_GPU_WILSON_BISTABCG
  template <typename T>
  struct LatticeCloverBistabCg
  {
    LatticeSet<T> *set_ptr;
    cudaError_t err;
    LatticeWilsonDslash<T> wilson_dslash;
    LatticeCloverDslash<T> clover_dslash_ee;
    LatticeCloverDslash<T> clover_dslash_oo;
    LatticeCloverDslash<T> clover_dslash_ee_inv;
    LatticeCloverDslash<T> clover_dslash_oo_inv;
    LatticeComplex<T> tmp0;
    LatticeComplex<T> tmp1;
    LatticeComplex<T> rho_prev;
    LatticeComplex<T> rho;
    LatticeComplex<T> alpha;
    LatticeComplex<T> beta;
    LatticeComplex<T> omega;
    void *gauge, *clover_ee, *clover_oo, *clover_ee_inv, *clover_oo_inv, *ans_e, *ans_o, *x_e, *x_o, *b_e, *b_o, *b__o, *r, *r_tilde, *p,
        *v, *s, *t, *device_vec0, *device_vec1, *device_vec2, *device_vals;
    LatticeComplex<T> host_vals[_vals_size_];
    int if_input, if_test;
    void give(LatticeSet<T> *_set_ptr)
    {
      set_ptr = _set_ptr;
      wilson_dslash.give(set_ptr);
      clover_dslash_ee.give(set_ptr);
      clover_dslash_oo.give(set_ptr);
      clover_dslash_ee_inv.give(set_ptr);
      clover_dslash_oo_inv.give(set_ptr);
    }
    void _init()
    {
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      checkCudaErrors(
          cudaMallocAsync(&b__o, set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>),
                          set_ptr->stream));
      checkCudaErrors(cudaMallocAsync(
          &r, set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>), set_ptr->stream));
      checkCudaErrors(cudaMallocAsync(
          &r_tilde, set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>),
          set_ptr->stream));
      checkCudaErrors(cudaMallocAsync(
          &p, set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>), set_ptr->stream));
      checkCudaErrors(cudaMallocAsync(
          &v, set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>), set_ptr->stream));
      checkCudaErrors(cudaMallocAsync(
          &s, set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>), set_ptr->stream));
      checkCudaErrors(cudaMallocAsync(
          &t, set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>), set_ptr->stream));
      checkCudaErrors(cudaMallocAsync(
          &device_vec0, set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>),
          set_ptr->stream));
      checkCudaErrors(cudaMallocAsync(
          &device_vec1, set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>),
          set_ptr->stream));
      checkCudaErrors(cudaMallocAsync(
          &device_vec2, set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>),
          set_ptr->stream));
      checkCudaErrors(cudaMallocAsync(
          &device_vals, _vals_size_ * sizeof(LatticeComplex<T>), set_ptr->stream));
      give_1custom<T><<<1, 1, 0, set_ptr->stream>>>(
          device_vals, _lat_4dim_, T(set_ptr->lat_4dim), 0.0);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    }
    void __init()
    {
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      if (if_input == 0)
      {
        checkCudaErrors(
            cudaMallocAsync(&x_o, set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>),
                            set_ptr->stream));
        checkCudaErrors(
            cudaMallocAsync(&ans_e, set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>),
                            set_ptr->stream));
        checkCudaErrors(
            cudaMallocAsync(&ans_o, set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>),
                            set_ptr->stream));
        give_random_vals<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                              set_ptr->stream>>>(ans_e, 12138);
        give_random_vals<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                              set_ptr->stream>>>(ans_o, 83121);
        checkCudaErrors(
            cudaMallocAsync(&b_e, set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>),
                            set_ptr->stream));
        checkCudaErrors(
            cudaMallocAsync(&b_o, set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>),
                            set_ptr->stream));
        wilson_dslash.run_eo(device_vec0, ans_o, gauge);
        give_copy_vals<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                            set_ptr->stream>>>(device_vec2, ans_e);
        clover_dslash_ee.give(device_vec2);
        bistabcg_give_b_e<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                               set_ptr->stream>>>(b_e, device_vec2, device_vec0, set_ptr->kappa(),
                                                  device_vals);
        wilson_dslash.run_oe(device_vec1, ans_e, gauge);
        bistabcg_give_b_o<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                               set_ptr->stream>>>(b_o, ans_o, device_vec1, set_ptr->kappa(),
                                                  device_vals);
      }
      { // give b__o, x_o, rr
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
        give_copy_vals<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                            set_ptr->stream>>>(device_vec2, b_e);
        printf("@@@@@@@@@@@@@@@DEBUGING!!!");
        clover_dslash_ee_inv.give(device_vec2);
        printf("###############DEBUGING!!!");
        wilson_dslash.run_oe(device_vec0, device_vec2, gauge);
        bistabcg_give_b__o<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                                set_ptr->stream>>>(b__o, b_o, device_vec0, set_ptr->kappa(),
                                                   device_vals);
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      }
      if (if_input == 0)
      {
        checkCudaErrors(cudaFreeAsync(b_e, set_ptr->stream));
        checkCudaErrors(cudaFreeAsync(b_o, set_ptr->stream));
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    }
    void _clover_dslash(void *fermion_out, void *fermion_in, void *gauge)
    {
      // A_oo*src_o-set_ptr->kappa()**2*dslash_oe*A_ee^-1*(dslash_eo(src_o))
      wilson_dslash.run_eo(device_vec0, fermion_in, gauge);
      give_copy_vals<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                          set_ptr->stream>>>(device_vec2, device_vec0);
      clover_dslash_ee_inv.give(device_vec2);
      wilson_dslash.run_oe(device_vec1, device_vec2, gauge);
      give_copy_vals<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                          set_ptr->stream>>>(device_vec2, fermion_in);
      clover_dslash_oo.give(device_vec2);
      bistabcg_give_dest_o<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                                set_ptr->stream>>>(
          fermion_out, device_vec2, device_vec1, set_ptr->kappa(), device_vals);
    }
    void _run_init()
    {
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      give_1zero<T><<<1, 1, 0, set_ptr->stream>>>(device_vals, _tmp0_);
      give_1zero<T><<<1, 1, 0, set_ptr->stream>>>(device_vals, _tmp1_);
      give_1one<T><<<1, 1, 0, set_ptr->stream>>>(device_vals, _rho_prev_);
      give_1zero<T><<<1, 1, 0, set_ptr->stream>>>(device_vals, _rho_);
      give_1one<T><<<1, 1, 0, set_ptr->stream>>>(device_vals, _alpha_);
      give_1one<T><<<1, 1, 0, set_ptr->stream>>>(device_vals, _omega_);
      give_1zero<T><<<1, 1, 0, set_ptr->stream>>>(device_vals, _send_tmp_);
      give_1zero<T><<<1, 1, 0, set_ptr->stream>>>(device_vals, _norm2_tmp_);
      give_1zero<T><<<1, 1, 0, set_ptr->stream>>>(device_vals, _diff_tmp_);
      give_custom_vals<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                            set_ptr->stream>>>(v, 0.0, 0.0);
      give_custom_vals<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                            set_ptr->stream>>>(p, 0.0, 0.0);
      give_random_vals<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                            set_ptr->stream>>>(x_o, 23333);
      _clover_dslash(r, x_o, gauge);
      bistabcg_give_rr<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                            set_ptr->stream>>>(r, b__o, r_tilde, device_vals);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    }
    void init(void *_x, void *_b, void *_gauge, void *_clover_ee, void *_clover_oo, void *_clover_ee_inv, void *_clover_oo_inv)
    {
      _init();
      if_input = 1;
      gauge = _gauge;
      clover_ee = _clover_ee;
      clover_oo = _clover_oo;
      clover_ee_inv = _clover_ee_inv;
      clover_oo_inv = _clover_oo_inv;
      clover_dslash_ee.init(clover_ee);
      clover_dslash_oo.init(clover_oo);
      clover_dslash_ee_inv.init(clover_ee_inv);
      clover_dslash_oo_inv.init(clover_oo_inv);
      x_e = _x;
      x_o = ((static_cast<LatticeComplex<T> *>(_x)) + set_ptr->lat_4dim_SC);
      b_e = _b;
      b_o = ((static_cast<LatticeComplex<T> *>(_b)) + set_ptr->lat_4dim_SC);
      __init();
      _run_init();
    }
    void init(void *_gauge)
    {
      _init();
      if_input = 0;
      gauge = _gauge;
      __init();
      _run_init();
    }
    void _dot_mpi(void *vec0, void *vec1, const int vals_index,
                  const int stream_index)
    {
      // dest(val) = _dot(A,B)
      CUBLAS_CHECK(_cublasDot<T>(
          set_ptr->cublasHs[stream_index], set_ptr->lat_4dim_SC, vec0,
          1, vec1,
          1,
          ((static_cast<LatticeComplex<T> *>(device_vals)) + _send_tmp_)));
      checkCudaErrors(cudaMemcpyAsync(
          ((static_cast<LatticeComplex<T> *>(host_vals)) + _send_tmp_),
          ((static_cast<LatticeComplex<T> *>(device_vals)) + _send_tmp_),
          sizeof(LatticeComplex<T>), cudaMemcpyDeviceToHost,
          set_ptr->streams[stream_index]));
      MPI_Barrier(MPI_COMM_WORLD);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[stream_index]));
      _MPI_Allreduce<T>(((static_cast<LatticeComplex<T> *>(host_vals)) + _send_tmp_), ((static_cast<LatticeComplex<T> *>(host_vals)) + vals_index), 2, MPI_SUM, MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
      checkCudaErrors(cudaMemcpyAsync(
          ((static_cast<LatticeComplex<T> *>(device_vals)) + vals_index),
          ((static_cast<LatticeComplex<T> *>(host_vals)) + vals_index),
          sizeof(LatticeComplex<T>), cudaMemcpyHostToDevice,
          set_ptr->streams[stream_index]));
    }
    void _dot(void *vec0, void *vec1, const int vals_index,
              const int stream_index)
    {
      _dot_mpi(vec0, vec1, vals_index, stream_index);
    }
    void _diff(void *x, void *ans)
    { // there is a bug
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
      _dot(ans, ans, _norm2_tmp_, _a_);
      bistabcg_give_diff<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                              set_ptr->streams[_a_]>>>(x, ans, device_vec0,
                                                       device_vals);
      _dot(device_vec0, device_vec0, _diff_tmp_, _a_);
      bistabcg_give_1diff<T><<<1, 1, 0, set_ptr->streams[_a_]>>>(device_vals);
      print_vals(0);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
    }
    void print_vals(int loop = 0)
    {
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
      checkCudaErrors(
          cudaMemcpyAsync((static_cast<LatticeComplex<T> *>(host_vals)),
                          (static_cast<LatticeComplex<T> *>(device_vals)),
                          _vals_size_ * sizeof(LatticeComplex<T>),
                          cudaMemcpyDeviceToHost, set_ptr->stream));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      std::cout << "######TIME  :" << set_ptr->get_time() << "######" << std::endl
                << "##RANK      :" << set_ptr->host_params[_NODE_RANK_] << std::endl
                << "##LOOP      :" << loop << std::endl
                << "##tmp0      :" << host_vals[_tmp0_] << std::endl
                << "##tmp1      :" << host_vals[_tmp1_] << std::endl
                << "##rho_prev  :" << host_vals[_rho_prev_] << std::endl
                << "##rho       :" << host_vals[_rho_] << std::endl
                << "##alpha     :" << host_vals[_alpha_] << std::endl
                << "##beta      :" << host_vals[_beta_] << std::endl
                << "##omega     :" << host_vals[_omega_] << std::endl
                << "##send_tmp  :" << host_vals[_send_tmp_] << std::endl
                << "##norm2_tmp :" << host_vals[_norm2_tmp_] << std::endl
                << "##diff_tmp  :" << host_vals[_diff_tmp_] << std::endl
                << "##lat_4dim  :" << host_vals[_lat_4dim_] << std::endl;
      // exit(1);
    }
    void _run()
    {
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
      for (int loop = 0; loop < set_ptr->max_iter(); loop++)
      {
        _dot(r_tilde, r, _rho_, _a_);
        checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
        {
          // beta = (rho / rho_prev) * (alpha / omega);
          bistabcg_give_1beta<T><<<1, 1, 0, set_ptr->streams[_a_]>>>(device_vals);
        }
        checkCudaErrors(cudaStreamSynchronize(
            set_ptr->streams[_a_])); // needed, but don't know why.
        {
          // rho_prev = rho;
          bistabcg_give_1rho_prev<T><<<1, 1, 0, set_ptr->streams[_b_]>>>(
              device_vals);
        }
        {
          // p[i] = r[i] + (p[i] - v[i] * omega) * beta;
          bistabcg_give_p<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                               set_ptr->streams[_a_]>>>(p, r, v, device_vals);
        }
        checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
        _dot(r, r, _norm2_tmp_, _c_);
        {
          // v = A * p;
          _clover_dslash(v, p, gauge);
        }
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
        _dot(r_tilde, v, _tmp0_, _d_);
        {
          // alpha = rho / tmp0;
          bistabcg_give_1alpha<T><<<1, 1, 0, set_ptr->streams[_d_]>>>(device_vals);
        }
        checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
        {
          // s[i] = r[i] - v[i] * alpha;
          bistabcg_give_s<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                               set_ptr->streams[_a_]>>>(s, r, v, device_vals);
        }
        checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
        {
          // t = A * s;
          _clover_dslash(t, s, gauge);
        }
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
        _dot(t, s, _tmp0_, _c_);
        _dot(t, t, _tmp1_, _d_);
        checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
        {
          // omega = tmp0 / tmp1;
          bistabcg_give_1omega<T><<<1, 1, 0, set_ptr->streams[_d_]>>>(device_vals);
        }
        checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
        {
          // r[i] = s[i] - t[i] * omega;
          bistabcg_give_r<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                               set_ptr->streams[_a_]>>>(r, s, t, device_vals);
        }
        {
          // x_o[i] = x_o[i] + p[i] * alpha + s[i] * omega;
          bistabcg_give_x_o<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                                 set_ptr->streams[_b_]>>>(x_o, p, s, device_vals);
        }
        {
#ifdef PRINT_MULTI_GPU_WILSON_BISTABCG
          std::cout << "##RANK:" << set_ptr->host_params[_NODE_RANK_] << "##LOOP:" << loop
                    << "##Residual:" << host_vals[_norm2_tmp_]._data.x
                    << std::endl;
#endif
          if ((host_vals[_norm2_tmp_]._data.x < set_ptr->tol() ||
               loop == set_ptr->max_iter() - 1))
          {
            std::cout << "##RANK:" << set_ptr->host_params[_NODE_RANK_] << "##LOOP:" << loop
                      << "##Residual:" << host_vals[_norm2_tmp_] << std::endl;
            break;
          }
        }
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
    }
    void run()
    {
      _run();
      if (if_input)
      {
        // get $x_{e}$ by $A_ee^-1*(b_{e}+\kappa D_{eo}x_{o})$
        CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH,
                                    set_ptr->lat_4dim_SC * _REAL_IMAG_,
                                    (T *)b_e, 1, (T *)device_vec0, 1));
        wilson_dslash.run_eo(device_vec1, x_o, gauge);
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
        LatticeComplex<T> _(set_ptr->kappa(), 0.0);
        // dest(B) = B + alpha*A
        CUBLAS_CHECK(
            _cublasAxpy<T>(set_ptr->cublasH, set_ptr->lat_4dim_SC, &_,
                           device_vec1,
                           1, device_vec0,
                           1));
        clover_dslash_ee_inv.give(device_vec0);
        CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH,
                                    set_ptr->lat_4dim_SC * _REAL_IMAG_,
                                    (T *)device_vec0, 1, (T *)x_e, 1));
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
        checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
        checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
        checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
        checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
      }
      set_ptr->err = cudaGetLastError();
      checkCudaErrors(set_ptr->err);
    }
    void run_test()
    {
      auto start = std::chrono::high_resolution_clock::now();
      run();
      auto end = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();
      printf(
          "multi-gpu clover bistabcg total time: (without malloc free memcpy) :%.9lf "
          "sec\n",
          T(duration) / 1e9);
      if (if_input == 0)
      {
        _diff(x_o, ans_o);
      }
      else
      {
        _clover_dslash(device_vec1, x_o, gauge);
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
        _diff(device_vec1, b__o);
      }
    }
    void end()
    {
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
      if (if_input == 0)
      {
        checkCudaErrors(cudaFreeAsync(ans_e, set_ptr->stream));
        checkCudaErrors(cudaFreeAsync(ans_o, set_ptr->stream));
        checkCudaErrors(cudaFreeAsync(x_o, set_ptr->stream));
      }
      checkCudaErrors(cudaFreeAsync(b__o, set_ptr->stream));
      checkCudaErrors(cudaFreeAsync(r, set_ptr->stream));
      checkCudaErrors(cudaFreeAsync(r_tilde, set_ptr->stream));
      checkCudaErrors(cudaFreeAsync(p, set_ptr->stream));
      checkCudaErrors(cudaFreeAsync(v, set_ptr->stream));
      checkCudaErrors(cudaFreeAsync(s, set_ptr->stream));
      checkCudaErrors(cudaFreeAsync(t, set_ptr->stream));
      checkCudaErrors(cudaFreeAsync(device_vec0, set_ptr->stream));
      checkCudaErrors(cudaFreeAsync(device_vec1, set_ptr->stream));
      checkCudaErrors(cudaFreeAsync(device_vals, set_ptr->stream));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_a_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_b_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_c_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->streams[_d_]));
    }
  };
}
#endif
