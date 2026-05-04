#ifndef _LATTICE_CUDA_H
#define _LATTICE_CUDA_H
#include "./include.h"
namespace qcu {
template <typename T>
void device_save(void *d_array, const int size, const std::string &filename);
template <typename T>
void host_save(void *h_array, const int size, const std::string &filename);
template <typename T>
void device_load(void *d_array, const int size, const std::string &filename);
template <typename T>
void host_load(void *h_array, const int size, const std::string &filename);
template <typename T>
cublasStatus_t _cublasCopy(cublasHandle_t handle, int n, const T *x, int incx,
                           T *y, int incy);
template <typename T>
cublasStatus_t _cublasAxpy(cublasHandle_t handle, int n, const void *alpha,
                           const void *x, int incx, void *y, int incy);
template <typename T>
cublasStatus_t _cublasDot(cublasHandle_t handle, int n, const void *x, int incx,
                          const void *y, int incy, void *result);
template <typename T>
__global__ void give_copy_vals(void *device_dest, void *device_src);
template <typename T>
__global__ void give_random_vals(void *device_random_vals, unsigned long seed);
template <typename T>
__global__ void give_custom_vals(void *device_custom_vals, T real, T imag);
template <typename T>
__global__ void give_1zero(void *device_vals, const int vals_index);
template <typename T>
__global__ void give_1one(void *device_vals, const int vals_index);
template <typename T>
__global__ void give_1custom(void *device_vals, const int vals_index, T real,
                             T imag);
} // namespace qcu
#endif