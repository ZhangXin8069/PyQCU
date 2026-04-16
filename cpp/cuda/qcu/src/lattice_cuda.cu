#include "../include/qcu.h"
#pragma optimize(5)
namespace qcu {
template <>
cublasStatus_t _cublasCopy<double>(cublasHandle_t handle, int n,
                                   const double *x, int incx, double *y,
                                   int incy) {
  return cublasDcopy(handle, n, x, incx, y, incy);
}
template <>
cublasStatus_t _cublasCopy<float>(cublasHandle_t handle, int n, const float *x,
                                  int incx, float *y, int incy) {
  return cublasScopy(handle, n, x, incx, y, incy);
}
template <>
cublasStatus_t _cublasAxpy<double>(cublasHandle_t handle, int n,
                                   const void *alpha, const void *x, int incx,
                                   void *y, int incy) {
  return cublasAxpyEx(handle, n, alpha, CUDA_C_64F, x, CUDA_C_64F, incx, y,
                      CUDA_C_64F, incy, CUDA_C_64F);
}
template <>
cublasStatus_t _cublasAxpy<float>(cublasHandle_t handle, int n,
                                  const void *alpha, const void *x, int incx,
                                  void *y, int incy) {
  return cublasAxpyEx(handle, n, alpha, CUDA_C_32F, x, CUDA_C_32F, incx, y,
                      CUDA_C_32F, incy, CUDA_C_32F);
}
template <>
cublasStatus_t _cublasDot<double>(cublasHandle_t handle, int n, const void *x,
                                  int incx, const void *y, int incy,
                                  void *result) {
  return cublasDotcEx(handle, n, x, CUDA_C_64F, incx, y, CUDA_C_64F, incy,
                      result, CUDA_C_64F, CUDA_C_64F);
}
template <>
cublasStatus_t _cublasDot<float>(cublasHandle_t handle, int n, const void *x,
                                 int incx, const void *y, int incy,
                                 void *result) {
  return cublasDotcEx(handle, n, x, CUDA_C_32F, incx, y, CUDA_C_32F, incy,
                      result, CUDA_C_32F, CUDA_C_32F);
}
template <typename T>
__global__ void give_copy_vals(void *device_dest, void *device_src) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex<T> *dest = static_cast<LatticeComplex<T> *>(device_dest);
  LatticeComplex<T> *src = static_cast<LatticeComplex<T> *>(device_src);
  for (int i = 0; i < _LAT_SC_; ++i) {
    dest[idx * _LAT_SC_ + i]._data.x = src[idx * _LAT_SC_ + i]._data.x;
    dest[idx * _LAT_SC_ + i]._data.y = src[idx * _LAT_SC_ + i]._data.y;
  }
}
template <typename T>
__global__ void give_random_vals(void *device_random_vals, unsigned long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex<T> *random_vals =
      static_cast<LatticeComplex<T> *>(device_random_vals);
  curandState state_real, state_imag;
  curand_init(seed, idx, 0, &state_real);
  curand_init(seed, idx, 1, &state_imag);
  for (int i = 0; i < _LAT_SC_; ++i) {
    random_vals[idx * _LAT_SC_ + i]._data.x = curand_uniform(&state_real);
    random_vals[idx * _LAT_SC_ + i]._data.y = curand_uniform(&state_imag);
  }
}
template <typename T>
__global__ void give_custom_vals(void *device_custom_vals, T real, T imag) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex<T> *custom_vals =
      static_cast<LatticeComplex<T> *>(device_custom_vals);
  for (int i = 0; i < _LAT_SC_; ++i) {
    custom_vals[idx * _LAT_SC_ + i]._data.x = real;
    custom_vals[idx * _LAT_SC_ + i]._data.y = imag;
  }
}
template <typename T>
__global__ void give_1zero(void *device_vals, const int vals_index) {
  LatticeComplex<T> *origin_vals =
      static_cast<LatticeComplex<T> *>(device_vals);
  LatticeComplex<T> _(0.0, 0.0);
  origin_vals[vals_index] = _;
}
template <typename T>
__global__ void give_1one(void *device_vals, const int vals_index) {
  LatticeComplex<T> *origin_vals =
      static_cast<LatticeComplex<T> *>(device_vals);
  LatticeComplex<T> _(1.0, 0.0);
  origin_vals[vals_index] = _;
}
template <typename T>
__global__ void give_1custom(void *device_vals, const int vals_index, T real,
                             T imag) {
  LatticeComplex<T> *origin_vals =
      static_cast<LatticeComplex<T> *>(device_vals);
  LatticeComplex<T> _(real, imag);
  origin_vals[vals_index] = _;
}
//@@@CUDA_TEMPLATE_FOR_DEVICE@@@
template __global__ void give_copy_vals<double>(void *device_dest,
                                                void *device_src);
template __global__ void give_random_vals<double>(void *device_random_vals,
                                                  unsigned long seed);
template __global__ void give_custom_vals<double>(void *device_custom_vals,
                                                  double real, double imag);
template __global__ void give_1zero<double>(void *device_vals,
                                            const int vals_index);
template __global__ void give_1one<double>(void *device_vals,
                                           const int vals_index);
template __global__ void give_1custom<double>(void *device_vals,
                                              const int vals_index, double real,
                                              double imag);
//@@@CUDA_TEMPLATE_FOR_DEVICE@@@
template __global__ void give_copy_vals<float>(void *device_dest,
                                               void *device_src);
template __global__ void give_random_vals<float>(void *device_random_vals,
                                                 unsigned long seed);
template __global__ void give_custom_vals<float>(void *device_custom_vals,
                                                 float real, float imag);
template __global__ void give_1zero<float>(void *device_vals,
                                           const int vals_index);
template __global__ void give_1one<float>(void *device_vals,
                                          const int vals_index);
template __global__ void give_1custom<float>(void *device_vals,
                                             const int vals_index, float real,
                                             float imag);
} // namespace qcu