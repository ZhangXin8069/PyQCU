#ifndef _LATTICE_MULTIGRID_H
#define _LATTICE_MULTIGRID_H
#include "./define.h"
#include "./lattice_set.h"
#include "./multigrid.h"
namespace qcu {
template <typename T> struct LatticeMultigridRestrict {
  LatticeSet<T> *set_ptr;
  cudaError_t err;
  void give(LatticeSet<T> *_set_ptr) { set_ptr = _set_ptr; }
  void run(void *coarse_out, void *fine_in, void *null_vecs, int E, int e,
           int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc) {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    int total_output = E * Xc * Yc * Zc * Tc;
    dim3 gridDim((total_output + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
    dim3 blockDim(_BLOCK_SIZE_);
    multigrid_restrict<T><<<gridDim, blockDim, 0, set_ptr->stream>>>(
        coarse_out, fine_in, null_vecs, E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
};
template <typename T> struct LatticeMultigridProLong {
  LatticeSet<T> *set_ptr;
  cudaError_t err;
  void give(LatticeSet<T> *_set_ptr) { set_ptr = _set_ptr; }
  void run(void *fine_out, void *coarse_in, void *null_vecs, int E, int e,
           int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc) {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    int total_output = e * Xf * Yf * Zf * Tf;
    dim3 gridDim((total_output + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
    dim3 blockDim(_BLOCK_SIZE_);
    multigrid_prolong<T><<<gridDim, blockDim, 0, set_ptr->stream>>>(
        fine_out, coarse_in, null_vecs, E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
};
template <typename T> struct LatticeMultigridCoarseDslash {
  LatticeSet<T> *set_ptr;
  cudaError_t err;
  void give(LatticeSet<T> *_set_ptr) { set_ptr = _set_ptr; }
  void run(void *fermion_out, void *fermion_in, void *hopping, void *sitting,
           int E, int X, int Y, int Z, int T) {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    int total_output = E * X * Y * Z * T;
    dim3 gridDim((total_output + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
    dim3 blockDim(_BLOCK_SIZE_);
    multigrid_coarse_dslash<T><<<gridDim, blockDim, 0, set_ptr->stream>>>(
        fermion_out, fermion_in, hopping, sitting, E, X, Y, Z, T);
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
};
} // namespace qcu
#endif
