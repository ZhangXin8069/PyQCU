#include "../include/qcu.h"
#pragma optimize(5)
namespace qcu
{
  template <typename T>
  void device_save(void *d_array, const int size, const std::string &filename)
  {
    T *h_array;
    h_array = new T[size];
    cudaDeviceSynchronize();
    cudaMemcpy(h_array, d_array, size * sizeof(T), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::ofstream outFile(filename, std::ios::binary);
    if (outFile.is_open())
    {
      outFile.write(reinterpret_cast<char *>(h_array), size * sizeof(T));
      outFile.close();
      std::cout << "save to " << filename << std::endl;
    }
    else
    {
      std::cerr << "unable to save to " << filename << std::endl;
    }
    delete[] h_array;
  }
  template <typename T>
  void host_save(void *h_array, const int size, const std::string &filename)
  {
    std::ofstream outFile(filename, std::ios::binary);
    if (outFile.is_open())
    {
      outFile.write(reinterpret_cast<char *>(h_array), size * sizeof(T));
      outFile.close();
      std::cout << "save to " << filename << std::endl;
    }
    else
    {
      std::cerr << "unable to save to " << filename << std::endl;
    }
  }
  template <typename T>
  void device_load(void *d_array, const int size, const std::string &filename)
  {
    T *h_array;
    h_array = new T[size];
    std::ifstream inFile(filename, std::ios::binary);
    if (inFile.is_open())
    {
      inFile.read(reinterpret_cast<char *>(h_array), size * sizeof(T));
      inFile.close();
      std::cout << "load from " << filename << std::endl;
    }
    else
    {
      std::cerr << "unable to load from " << filename << std::endl;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(d_array, h_array, size * sizeof(T), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    delete[] h_array;
  }
  template <typename T>
  void host_load(void *h_array, const int size, const std::string &filename)
  {
    std::ifstream inFile(filename, std::ios::binary);
    if (inFile.is_open())
    {
      inFile.read(reinterpret_cast<char *>(h_array), size * sizeof(T));
      inFile.close();
      std::cout << "load from " << filename << std::endl;
    }
    else
    {
      std::cerr << "unable to load from " << filename << std::endl;
    }
  }
  template <>
  cublasStatus_t _cublasCopy<double>(cublasHandle_t handle, int n, const double *x, int incx, double *y, int incy)
  {
    return cublasDcopy(handle, n, x, incx, y, incy);
  }
  template <>
  cublasStatus_t _cublasCopy<float>(cublasHandle_t handle, int n, const float *x, int incx, float *y, int incy)
  {
    return cublasScopy(handle, n, x, incx, y, incy);
  }
  template <>
  cublasStatus_t _cublasAxpy<double>(cublasHandle_t handle,
                                     int n,
                                     const void *alpha,
                                     const void *x,
                                     int incx,
                                     void *y,
                                     int incy)
  {
    return cublasAxpyEx(handle,
                        n,
                        alpha,
                        CUDA_C_64F,
                        x,
                        CUDA_C_64F,
                        incx,
                        y,
                        CUDA_C_64F,
                        incy,
                        CUDA_C_64F);
  }
  template <>
  cublasStatus_t _cublasAxpy<float>(cublasHandle_t handle,
                                    int n,
                                    const void *alpha,
                                    const void *x,
                                    int incx,
                                    void *y,
                                    int incy)
  {
    return cublasAxpyEx(handle,
                        n,
                        alpha,
                        CUDA_C_32F,
                        x,
                        CUDA_C_32F,
                        incx,
                        y,
                        CUDA_C_32F,
                        incy,
                        CUDA_C_32F);
  }
  template <>
  cublasStatus_t _cublasDot<double>(cublasHandle_t handle,
                                    int n,
                                    const void *x,
                                    int incx,
                                    const void *y,
                                    int incy,
                                    void *result)
  {
    return cublasDotcEx(handle,
                        n,
                        x,
                        CUDA_C_64F,
                        incx,
                        y,
                        CUDA_C_64F,
                        incy,
                        result,
                        CUDA_C_64F,
                        CUDA_C_64F);
  }
  template <>
  cublasStatus_t _cublasDot<float>(cublasHandle_t handle,
                                   int n,
                                   const void *x,
                                   int incx,
                                   const void *y,
                                   int incy,
                                   void *result)
  {
    return cublasDotcEx(handle,
                        n,
                        x,
                        CUDA_C_32F,
                        incx,
                        y,
                        CUDA_C_32F,
                        incy,
                        result,
                        CUDA_C_32F,
                        CUDA_C_32F);
  }
  template <typename T>
  __global__ void give_copy_vals(void *device_dest, void *device_src);
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *dest =
        static_cast<LatticeComplex<T> *>(device_dest);
    LatticeComplex<T> *src =
        static_cast<LatticeComplex<T> *>(device_src);
    for (int i = 0; i < _LAT_SC_; ++i)
    {
      dest[idx * _LAT_SC_ + i]._data.x = src[idx * _LAT_SC_ + i]._data.x;
      dest[idx * _LAT_SC_ + i]._data.y = src[idx * _LAT_SC_ + i]._data.y;
    }
  }
  template <typename T>
  __global__ void give_random_vals(void *device_random_vals, unsigned long seed)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *random_vals =
        static_cast<LatticeComplex<T> *>(device_random_vals);
    curandState state_real, state_imag;
    curand_init(seed, idx, 0, &state_real);
    curand_init(seed, idx, 1, &state_imag);
    for (int i = 0; i < _LAT_SC_; ++i)
    {
      random_vals[idx * _LAT_SC_ + i]._data.x = curand_uniform(&state_real);
      random_vals[idx * _LAT_SC_ + i]._data.y = curand_uniform(&state_imag);
    }
  }
  template <typename T>
  __global__ void give_custom_vals(void *device_custom_vals, T real,
                                   T imag)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *custom_vals =
        static_cast<LatticeComplex<T> *>(device_custom_vals);
    for (int i = 0; i < _LAT_SC_; ++i)
    {
      custom_vals[idx * _LAT_SC_ + i]._data.x = real;
      custom_vals[idx * _LAT_SC_ + i]._data.y = imag;
    }
  }
  template <typename T>
  __global__ void give_1zero(void *device_vals, const int vals_index)
  {
    LatticeComplex<T> *origin_vals = static_cast<LatticeComplex<T> *>(device_vals);
    LatticeComplex<T> _(0.0, 0.0);
    origin_vals[vals_index] = _;
  }
  template <typename T>
  __global__ void give_1one(void *device_vals, const int vals_index)
  {
    LatticeComplex<T> *origin_vals = static_cast<LatticeComplex<T> *>(device_vals);
    LatticeComplex<T> _(1.0, 0.0);
    origin_vals[vals_index] = _;
  }
  template <typename T>
  __global__ void give_1custom(void *device_vals, const int vals_index,
                               T real, T imag)
  {
    LatticeComplex<T> *origin_vals = static_cast<LatticeComplex<T> *>(device_vals);
    LatticeComplex<T> _(real, imag);
    origin_vals[vals_index] = _;
  }
  template <typename T>
  __global__ void _tzyxsc2sctzyx(void *device_fermi, void *device__fermi,
                                 int lat_4dim)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *fermion =
        ((static_cast<LatticeComplex<T> *>(device_fermi)) + idx * _LAT_SC_);
    LatticeComplex<T> *_fermion =
        ((static_cast<LatticeComplex<T> *>(device__fermi)) + idx);
    for (int i = 0; i < _LAT_SC_; i++)
    {
      _fermion[i * lat_4dim] = fermion[i];
    }
  }
  template <typename T>
  __global__ void _sctzyx2tzyxsc(void *device_fermi, void *device__fermi,
                                 int lat_4dim)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *fermion =
        ((static_cast<LatticeComplex<T> *>(device_fermi)) + idx);
    LatticeComplex<T> *_fermion =
        ((static_cast<LatticeComplex<T> *>(device__fermi)) + idx * _LAT_SC_);
    // printf("fermion[0]._data.x:%f,fermion[0]._data.y:%f\n", fermion[0]._data.x, fermion[0]._data.y); // test - bug10 refer to .log-bug10.txt
    for (int i = 0; i < _LAT_SC_; i++)
    {
      _fermion[i] = fermion[i * lat_4dim];
    }
  }
  template <typename T>
  void tzyxsc2sctzyx(void *fermion, LatticeSet<T> *set_ptr)
  {
    void *_fermion;
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&_fermion,
                                    set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>),
                                    set_ptr->stream));
    _tzyxsc2sctzyx<T><<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
        fermion, _fermion, set_ptr->lat_4dim);
    CUBLAS_CHECK(
        _cublasCopy<T>(set_ptr->cublasH,
                       set_ptr->lat_4dim_SC * _REAL_IMAG_,
                       (T *)_fermion, 1, (T *)fermion, 1));
    checkCudaErrors(cudaFreeAsync(_fermion, set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
  template <typename T>
  void sctzyx2tzyxsc(void *fermion, LatticeSet<T> *set_ptr)
  {
    void *_fermion;
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(&_fermion,
                                    set_ptr->lat_4dim_SC * sizeof(LatticeComplex<T>),
                                    set_ptr->stream));
    _sctzyx2tzyxsc<T><<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
        fermion, _fermion, set_ptr->lat_4dim);
    CUBLAS_CHECK(
        _cublasCopy<T>(set_ptr->cublasH,
                       set_ptr->lat_4dim_SC * _REAL_IMAG_,
                       (T *)_fermion, 1, (T *)fermion, 1));
    checkCudaErrors(cudaFreeAsync(_fermion, set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
  template <typename T>
  __global__ void _dptzyxcc2ccdptzyx(void *device_gauge, void *device__gauge,
                                     int lat_4dim)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *gauge =
        ((static_cast<LatticeComplex<T> *>(device_gauge)) + idx * _LAT_CC_);
    LatticeComplex<T> *_gauge =
        ((static_cast<LatticeComplex<T> *>(device__gauge)) + idx);
    for (int p = 0; p < _EVEN_ODD_; p++)
    {
      for (int d = 0; d < _LAT_D_; d++)
      {
        for (int cc = 0; cc < _LAT_CC_; cc++)
        {
          _gauge[((cc * _LAT_D_ + d) * _EVEN_ODD_ + p) * lat_4dim] =
              gauge[(d * _EVEN_ODD_ + p) * _LAT_CC_ * lat_4dim + cc];
        }
      }
    }
  }
  template <typename T>
  __global__ void _ccdptzyx2dptzyxcc(void *device_gauge, void *device__gauge,
                                     int lat_4dim)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *gauge = ((static_cast<LatticeComplex<T> *>(device_gauge)) + idx);
    LatticeComplex<T> *_gauge =
        ((static_cast<LatticeComplex<T> *>(device__gauge)) + idx * _LAT_CC_);
    for (int p = 0; p < _EVEN_ODD_; p++)
    {
      for (int d = 0; d < _LAT_D_; d++)
      {
        for (int cc = 0; cc < _LAT_CC_; cc++)
        {
          _gauge[(d * _EVEN_ODD_ + p) * _LAT_CC_ * lat_4dim + cc] =
              gauge[((cc * _LAT_D_ + d) * _EVEN_ODD_ + p) * lat_4dim];
        }
      }
    }
  }
  template <typename T>
  void dptzyxcc2ccdptzyx(void *gauge, LatticeSet<T> *set_ptr)
  {
    void *_gauge;
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(
        &_gauge, set_ptr->lat_4dim_DCC * _EVEN_ODD_ * sizeof(LatticeComplex<T>),
        set_ptr->stream));
    _dptzyxcc2ccdptzyx<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                            set_ptr->stream>>>(gauge, _gauge, set_ptr->lat_4dim);
    CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH,
                                set_ptr->lat_4dim_DCC * _EVEN_ODD_ *
                                    _REAL_IMAG_,
                                (T *)_gauge, 1, (T *)gauge, 1));
    checkCudaErrors(cudaFreeAsync(_gauge, set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
  template <typename T>
  void ccdptzyx2dptzyxcc(void *gauge, LatticeSet<T> *set_ptr)
  {
    void *_gauge;
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(
        &_gauge, set_ptr->lat_4dim_DCC * _EVEN_ODD_ * sizeof(LatticeComplex<T>),
        set_ptr->stream));
    _ccdptzyx2dptzyxcc<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                            set_ptr->stream>>>(gauge, _gauge, set_ptr->lat_4dim);
    CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH,
                                set_ptr->lat_4dim_DCC * _EVEN_ODD_ *
                                    _REAL_IMAG_,
                                (T *)_gauge, 1, (T *)gauge, 1));
    checkCudaErrors(cudaFreeAsync(_gauge, set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
  template <typename T>
  __global__ void _ptzyxsc2psctzyx(void *device_fermi, void *device__fermi,
                                   int lat_4dim)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *fermion =
        ((static_cast<LatticeComplex<T> *>(device_fermi)) + idx * _LAT_SC_);
    LatticeComplex<T> *_fermion =
        ((static_cast<LatticeComplex<T> *>(device__fermi)) + idx);
    for (int p = 0; p < _EVEN_ODD_; p++)
    {
      for (int i = 0; i < _LAT_SC_; i++)
      {
        _fermion[(p * _LAT_SC_ + i) * lat_4dim] =
            fermion[p * _LAT_SC_ * lat_4dim + i];
      }
    }
  }
  template <typename T>
  __global__ void _psctzyx2ptzyxsc(void *device_fermi, void *device__fermi,
                                   int lat_4dim)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    LatticeComplex<T> *fermion =
        ((static_cast<LatticeComplex<T> *>(device_fermi)) + idx);
    LatticeComplex<T> *_fermion =
        ((static_cast<LatticeComplex<T> *>(device__fermi)) + idx * _LAT_SC_);
    for (int p = 0; p < _EVEN_ODD_; p++)
    {
      for (int i = 0; i < _LAT_SC_; i++)
      {
        _fermion[p * _LAT_SC_ * lat_4dim + i] =
            fermion[(p * _LAT_SC_ + i) * lat_4dim];
      }
    }
  }
  template <typename T>
  void ptzyxsc2psctzyx(void *fermion, LatticeSet<T> *set_ptr)
  {
    void *_fermion;
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(
        &_fermion, set_ptr->lat_4dim_SC * _EVEN_ODD_ * sizeof(LatticeComplex<T>),
        set_ptr->stream));
    _ptzyxsc2psctzyx<T><<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
        fermion, _fermion, set_ptr->lat_4dim);
    CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH,
                                set_ptr->lat_4dim_SC * _EVEN_ODD_ *
                                    _REAL_IMAG_,
                                (T *)_fermion, 1, (T *)fermion, 1));
    checkCudaErrors(cudaFreeAsync(_fermion, set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
  template <typename T>
  void psctzyx2ptzyxsc(void *fermion, LatticeSet<T> *set_ptr)
  {
    void *_fermion;
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    checkCudaErrors(cudaMallocAsync(
        &_fermion, set_ptr->lat_4dim_SC * _EVEN_ODD_ * sizeof(LatticeComplex<T>),
        set_ptr->stream));
    _psctzyx2ptzyxsc<T><<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
        fermion, _fermion, set_ptr->lat_4dim);
    CUBLAS_CHECK(_cublasCopy<T>(set_ptr->cublasH,
                                set_ptr->lat_4dim_SC * _EVEN_ODD_ *
                                    _REAL_IMAG_,
                                (T *)_fermion, 1, (T *)fermion, 1));
    checkCudaErrors(cudaFreeAsync(_fermion, set_ptr->stream));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
  template <typename T>
  __global__ void give_debug_u(void *device_U, void *device_params)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int parity = idx;
    int *params = static_cast<int *>(device_params);
    int lat_x = params[_LAT_X_];
    int lat_y = params[_LAT_Y_];
    int lat_z = params[_LAT_Z_];
    int lat_t = params[_LAT_T_];
    int lat_tzyx = params[_LAT_XYZT_];
    int move0;
    move0 = lat_x * lat_y * lat_z;
    int t = parity / move0;
    parity -= t * move0;
    move0 = lat_x * lat_y;
    int z = parity / move0;
    parity -= z * move0;
    int y = parity / lat_x;
    int x = parity - y * lat_x;
    LatticeComplex<T> *origin_U = static_cast<LatticeComplex<T> *>(device_U);
    LatticeComplex<T> *tmp_U;
    parity = params[_PARITY_];
    tmp_U = (origin_U +
             ((((parity * lat_t + t) * lat_z + z) * lat_y + y) * lat_x + x));
    for (int i = 0; i < _LAT_DCC_; i++)
    {
      tmp_U[i * _EVEN_ODD_ * lat_tzyx]._data.x =
          T((((((i * _EVEN_ODD_ + parity) * lat_t + t) * lat_z + z) * lat_y +
              y) *
                 lat_x +
             x)) /
          lat_tzyx;
      tmp_U[i * _EVEN_ODD_ * lat_tzyx]._data.y = T(params[_NODE_RANK_]);
    }
  }
  //@@@CUDA_TEMPLATE_FOR_DEVICE@@@
  template void device_save<double>(void *d_array, const int size, const std::string &filename);
  template void host_save<double>(void *h_array, const int size, const std::string &filename);
  template void device_load<double>(void *d_array, const int size, const std::string &filename);
  template void host_load<double>(void *h_array, const int size, const std::string &filename);
  template __global__ void give_random_vals<double>(void *device_random_vals, unsigned long seed);
  template __global__ void give_custom_vals<double>(void *device_custom_vals, double real,
                                                    double imag);
  template __global__ void give_1zero<double>(void *device_vals, const int vals_index);
  template __global__ void give_1one<double>(void *device_vals, const int vals_index);
  template __global__ void give_1custom<double>(void *device_vals, const int vals_index,
                                                double real, double imag);
  template __global__ void _tzyxsc2sctzyx<double>(void *device_fermi, void *device__fermi,
                                                  int lat_4dim);
  template __global__ void _sctzyx2tzyxsc<double>(void *device_fermi, void *device__fermi,
                                                  int lat_4dim);
  template void tzyxsc2sctzyx<double>(void *fermion, LatticeSet<double> *set_ptr);
  template void sctzyx2tzyxsc<double>(void *fermion, LatticeSet<double> *set_ptr);
  template __global__ void _dptzyxcc2ccdptzyx<double>(void *device_gauge, void *device__gauge,
                                                      int lat_4dim);
  template __global__ void _ccdptzyx2dptzyxcc<double>(void *device_gauge, void *device__gauge,
                                                      int lat_4dim);
  template void dptzyxcc2ccdptzyx<double>(void *gauge, LatticeSet<double> *set_ptr);
  template void ccdptzyx2dptzyxcc<double>(void *gauge, LatticeSet<double> *set_ptr);
  template __global__ void _ptzyxsc2psctzyx<double>(void *device_fermi, void *device__fermi,
                                                    int lat_4dim);
  template __global__ void _psctzyx2ptzyxsc<double>(void *device_fermi, void *device__fermi,
                                                    int lat_4dim);
  template void ptzyxsc2psctzyx<double>(void *fermion, LatticeSet<double> *set_ptr);
  template void psctzyx2ptzyxsc<double>(void *fermion, LatticeSet<double> *set_ptr);
  template __global__ void give_debug_u<double>(void *device_U, void *device_params);
  //@@@CUDA_TEMPLATE_FOR_DEVICE@@@
  template void device_save<float>(void *d_array, const int size, const std::string &filename);
  template void host_save<float>(void *h_array, const int size, const std::string &filename);
  template void device_load<float>(void *d_array, const int size, const std::string &filename);
  template void host_load<float>(void *h_array, const int size, const std::string &filename);
  template __global__ void give_random_vals<float>(void *device_random_vals, unsigned long seed);
  template __global__ void give_custom_vals<float>(void *device_custom_vals, float real,
                                                   float imag);
  template __global__ void give_1zero<float>(void *device_vals, const int vals_index);
  template __global__ void give_1one<float>(void *device_vals, const int vals_index);
  template __global__ void give_1custom<float>(void *device_vals, const int vals_index,
                                               float real, float imag);
  template __global__ void _tzyxsc2sctzyx<float>(void *device_fermi, void *device__fermi,
                                                 int lat_4dim);
  template __global__ void _sctzyx2tzyxsc<float>(void *device_fermi, void *device__fermi,
                                                 int lat_4dim);
  template void tzyxsc2sctzyx<float>(void *fermion, LatticeSet<float> *set_ptr);
  template void sctzyx2tzyxsc<float>(void *fermion, LatticeSet<float> *set_ptr);
  template __global__ void _dptzyxcc2ccdptzyx<float>(void *device_gauge, void *device__gauge,
                                                     int lat_4dim);
  template __global__ void _ccdptzyx2dptzyxcc<float>(void *device_gauge, void *device__gauge,
                                                     int lat_4dim);
  template void dptzyxcc2ccdptzyx<float>(void *gauge, LatticeSet<float> *set_ptr);
  template void ccdptzyx2dptzyxcc<float>(void *gauge, LatticeSet<float> *set_ptr);
  template __global__ void _ptzyxsc2psctzyx<float>(void *device_fermi, void *device__fermi,
                                                   int lat_4dim);
  template __global__ void _psctzyx2ptzyxsc<float>(void *device_fermi, void *device__fermi,
                                                   int lat_4dim);
  template void ptzyxsc2psctzyx<float>(void *fermion, LatticeSet<float> *set_ptr);
  template void psctzyx2ptzyxsc<float>(void *fermion, LatticeSet<float> *set_ptr);
  template __global__ void give_debug_u<float>(void *device_U, void *device_params);
}