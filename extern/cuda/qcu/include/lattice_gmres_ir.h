#ifndef _LATTICE_GMERS_IR_H
#define _LATTICE_GMERS_IR_H
#include "./gmres_ir.h"
#include "./lattice_bistabcg.h"
namespace qcu
{
  template <typename T>
  struct LatticeGmresIr
  {
    void *r, *b, *x, *e;
    LatticeBistabCg<T> gmres;
    void give(LatticeSet<T> *_set_ptr)
    {
      gmres.give(_set_ptr);
    }
    void init(void *_x, void *_b, void *_gauge)
    {
      checkCudaErrors(cudaStreamSynchronize(gmres.set_ptr->stream));
      gmres.init(_x, _b, _gauge); // r_tilde = r = b__o (real b) - A * x_o, r_tilde and b__o doesn't change after init.
      checkCudaErrors(cudaStreamSynchronize(gmres.set_ptr->stream));
      checkCudaErrors(
          cudaMallocAsync(&x, gmres.set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>),
                          gmres.set_ptr->stream));
      CUBLAS_CHECK(_cublasCopy<T>(gmres.set_ptr->cublasH,
                                  gmres.set_ptr->lat_4dim_SC * _REAL_IMAG_,
                                  (T *)gmres.x_o, 1, (T *)x, 1)); // x_o (x_0) -> x
      checkCudaErrors(cudaStreamSynchronize(gmres.set_ptr->stream));
      checkCudaErrors(
          cudaMallocAsync(&b, gmres.set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>),
                          gmres.set_ptr->stream));
      CUBLAS_CHECK(_cublasCopy<T>(gmres.set_ptr->cublasH,
                                  gmres.set_ptr->lat_4dim_SC * _REAL_IMAG_,
                                  (T *)gmres.b__o, 1, (T *)b, 1)); // b__o (real b) -> b
      checkCudaErrors(cudaStreamSynchronize(gmres.set_ptr->stream));
      r = gmres.b__o; // r <-> b__o
      CUBLAS_CHECK(_cublasCopy<T>(gmres.set_ptr->cublasH,
                                  gmres.set_ptr->lat_4dim_SC * _REAL_IMAG_,
                                  (T *)gmres.r_tilde, 1, (T *)r, 1)); // r_tilde (b__o - A * x_o) [b - A * x_0] -> r (r_0)
      e = gmres.x_o;                                                  // e <-> x_o
      checkCudaErrors(cudaStreamSynchronize(gmres.set_ptr->stream));
    }
    void _run()
    {
      checkCudaErrors(cudaStreamSynchronize(gmres.set_ptr->stream));
      gmres._run_init(); // give init with new r (b__o)
      gmres._run();      // give e (x_o) [A * e = r]
      gmres_ir_give_x<T><<<gmres.set_ptr->gridDim, gmres.set_ptr->blockDim, 0,
                           gmres.set_ptr->stream>>>(x, e, gmres.device_vals); // x = x + e
      gmres._wilson_dslash(r, x, gmres.gauge);                                // r (tmp use) = A * x
      gmres_ir_give_r<T><<<gmres.set_ptr->gridDim, gmres.set_ptr->blockDim, 0,
                           gmres.set_ptr->stream>>>(r, b, gmres.device_vals); // r = b -r (b - A *x)
      checkCudaErrors(cudaStreamSynchronize(gmres.set_ptr->stream));
    }
    void run()
    {
      checkCudaErrors(cudaStreamSynchronize(gmres.set_ptr->stream));
      checkCudaErrors(cudaStreamSynchronize(gmres.set_ptr->streams[_a_]));
      for (int loop = 0; loop < gmres.set_ptr->max_iter(); loop++)
      {
        _run();
        checkCudaErrors(cudaStreamSynchronize(gmres.set_ptr->streams[_a_]));
        gmres._dot(r, r, _norm2_tmp_, _a_);
        checkCudaErrors(cudaStreamSynchronize(gmres.set_ptr->streams[_a_]));
        std::cout << "##RANK:" << gmres.set_ptr->host_params[_NODE_RANK_] << "##LOOP:" << loop
                  << "##Residual:" << gmres.host_vals[_norm2_tmp_] << std::endl;
        if ((gmres.host_vals[_norm2_tmp_]._data.x < gmres.set_ptr->tol() / 10 ||
             loop == gmres.set_ptr->max_iter() - 1)) // just for test, wait for multi-precision
        {
          std::cout << "##RANK:" << gmres.set_ptr->host_params[_NODE_RANK_] << "##LOOP:" << loop
                    << "##Residual:" << gmres.host_vals[_norm2_tmp_] << std::endl;
          break;
        }
      }
      checkCudaErrors(cudaStreamSynchronize(gmres.set_ptr->stream));
      CUBLAS_CHECK(_cublasCopy<T>(gmres.set_ptr->cublasH,
                                  gmres.set_ptr->lat_4dim_SC * _REAL_IMAG_,
                                  (T *)x, 1, (T *)gmres.x_o, 1)); // x -> x_o
      checkCudaErrors(cudaStreamSynchronize(gmres.set_ptr->stream));
      gmres.run(); // give x_e
      checkCudaErrors(cudaStreamSynchronize(gmres.set_ptr->stream));
      checkCudaErrors(cudaStreamSynchronize(gmres.set_ptr->streams[_a_]));
    }
    void run_test()
    {
      auto start = std::chrono::high_resolution_clock::now();
      run();
      auto end = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();
      gmres.set_ptr->err = cudaGetLastError();
      checkCudaErrors(gmres.set_ptr->err);
      printf(
          "multi-gpu wilson gmres_ir total time: (without malloc free memcpy) :%.9lf "
          "sec\n",
          T(duration) / 1e9);
      gmres._wilson_dslash(gmres.device_vec1, gmres.x_o, gmres.gauge);
      checkCudaErrors(cudaStreamSynchronize(gmres.set_ptr->stream));
      gmres._diff(gmres.device_vec1, gmres.b__o);
    }
    void end()
    {
      gmres.end();
    }
  };
}
#endif
