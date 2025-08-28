#include "../include/qcu.h"
#pragma optimize(5)
namespace qcu
{
    template <typename T>
    __global__ void give_random_8dtzyx(void *device_random_8dtzyx, void *device_params, unsigned long seed)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int *params = static_cast<int *>(device_params);
        int lat_tzyx = params[_LAT_XYZT_];
        LatticeComplex<T> *random_8dtzyx =
            (static_cast<LatticeComplex<T> *>(device_random_8dtzyx) + idx);
        curandState state_real, state_imag;
        curand_init(seed, idx, 0, &state_real);
        curand_init(seed, idx, 1, &state_imag);
        for (int d = 0; d < _LAT_D_; ++d)
        {
            for (int cc = 0; cc < (_LAT_CC_ - 1); ++cc)
            {
                random_8dtzyx[(cc * _LAT_D_ + d) * lat_tzyx]._data.x = curand_uniform(&state_real);
                random_8dtzyx[(cc * _LAT_D_ + d) * lat_tzyx]._data.y = curand_uniform(&state_real);
            }
        }
    }
    // 3x3 matrix exponential (Taylor expansion up to 12th order)
    template <typename T>
    __device__ void su3_matrix_exponential(const LatticeComplex<T> A[_LAT_CC_],
                                           LatticeComplex<T> R[_LAT_CC_])
    {
        // Identity matrix
        const LatticeComplex<T> I[_LAT_CC_] = {
            {1, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}};
        // Initialize result = I + A
        for (int i = 0; i < _LAT_CC_; i++)
        {
            R[i]._data.x = I[i]._data.x + A[i]._data.x;
            R[i]._data.y = I[i]._data.y + A[i]._data.y;
        }
        // Term = A
        LatticeComplex<T> Term[_LAT_CC_];
        for (int i = 0; i < _LAT_CC_; i++)
            Term[i] = A[i];
        // Precomputed inverse factorials (0! ... 12!)
        constexpr T inv_fac[13] = {
            T(1.0),              // 0!
            T(1.0),              // 1!
            T(1.0 / 2.0),        // 2!
            T(1.0 / 6.0),        // 3!
            T(1.0 / 24.0),       // 4!
            T(1.0 / 120.0),      // 5!
            T(1.0 / 720.0),      // 6!
            T(1.0 / 5040.0),     // 7!
            T(1.0 / 40320.0),    // 8!
            T(1.0 / 362880.0),   // 9!
            T(1.0 / 3628800.0),  // 10!
            T(1.0 / 39916800.0), // 11!
            T(1.0 / 479001600.0) // 12!
        };
        // Taylor expansion up to 12th order
        LatticeComplex<T> New[_LAT_CC_];
        for (int n = 2; n <= 12; n++)
        {
            for (int i = 0; i < _LAT_C_; i++)
            {
                for (int j = 0; j < _LAT_C_; j++)
                {
                    T real = 0.0;
                    T imag = 0.0;
                    for (int k = 0; k < _LAT_C_; k++)
                    {
                        int idx_a = i * _LAT_C_ + k;
                        int idx_b = k * _LAT_C_ + j;
                        real += A[idx_a]._data.x * Term[idx_b]._data.x - A[idx_a]._data.y * Term[idx_b]._data.y;
                        imag += A[idx_a]._data.x * Term[idx_b]._data.y + A[idx_a]._data.y * Term[idx_b]._data.x;
                    }
                    New[i * _LAT_C_ + j]._data.x = real;
                    New[i * _LAT_C_ + j]._data.y = imag;
                }
            }
            for (int i = 0; i < _LAT_CC_; i++)
            {
                Term[i]._data.x = New[i]._data.x * inv_fac[n];
                Term[i]._data.y = New[i]._data.y * inv_fac[n];
                R[i]._data.x += Term[i]._data.x;
                R[i]._data.y += Term[i]._data.y;
            }
        }
    }
    template <typename T>
    __global__ void _make_gauss_gauge(void *device_U, void *device_random_8dtzyx, void *device_params, T sigma)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int *params = static_cast<int *>(device_params);
        int lat_tzyx = params[_LAT_XYZT_];
        LatticeComplex<T> *random_8dtzyx = (static_cast<LatticeComplex<T> *>(device_random_8dtzyx) + idx);
        LatticeComplex<T> *origin_U = (static_cast<LatticeComplex<T> *>(device_U) + idx);
        LatticeComplex<T> U[_LAT_CC_] = {
            {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
        LatticeComplex<T> H[_LAT_CC_] = {
            {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
        LatticeComplex<T> A[_LAT_CC_];
        T gell_mann[8][9] = GELL_MANN;
        T a[_LAT_CC_ - 1];
        // Generate one SU(3) matrix for each direction
        for (int p = 0; p < _LAT_P_; p++)
        {
            for (int d = 0; d < _LAT_D_; d++)
            {
                // Generate 8 Gaussian random numbers
                for (int i = 0; i < _LAT_CC_ - 1; i++)
                {
                    if (p == 0)
                    {
                        a[i] = random_8dtzyx[(i * _LAT_D_ + d) * lat_tzyx]._data.x;
                    }
                    else
                    {
                        a[i] = random_8dtzyx[(i * _LAT_D_ + d) * lat_tzyx]._data.y;
                    }
                }
                // Construct Hermitian matrix H
                for (int i = 0; i < _LAT_CC_ - 1; i++)
                {
                    for (int j = 0; j < _LAT_CC_; j++)
                    {
                        int row = j / _LAT_C_;
                        int col = j % _LAT_C_;
                        int idx_h = row * _LAT_C_ + col;
                        if (i == 1 || i == 4 || i == 6)
                        {
                            // Handle matrices with imaginary unit
                            H[idx_h]._data.y += a[i] * gell_mann[i][j];
                        }
                        else
                        {
                            H[idx_h]._data.x += a[i] * gell_mann[i][j];
                        }
                    }
                }
                // Compute A = i * sigma * H
                for (int i = 0; i < _LAT_CC_; i++)
                {
                    A[i]._data.x = -sigma * H[i]._data.y; // imaginary part becomes real with negative sign
                    A[i]._data.y = sigma * H[i]._data.x;  // real part becomes imaginary
                }
                // Compute U = exp(A)
                su3_matrix_exponential(A, U);
                give_U(p, d, origin_U, U, lat_tzyx);
            }
        }
    }
    template <typename T>
    void make_gauss_gauge(void *device_U, void *set_ptr)
    {
        void *device_random_8dtzyx;
        LatticeSet<T> *_set_ptr = static_cast<LatticeSet<T> *>(set_ptr);
        if (_set_ptr->host_params[_VERBOSE_])
        {
            auto start = std::chrono::high_resolution_clock::now();
            checkCudaErrors(cudaStreamSynchronize(_set_ptr->stream));
            checkCudaErrors(
                cudaMallocAsync(&device_random_8dtzyx, _set_ptr->lat_4dim * _LAT_D_ * (_LAT_CC_ - 1) * sizeof(LatticeComplex<T>), _set_ptr->stream));
            give_random_8dtzyx<T><<<_set_ptr->gridDim, _set_ptr->blockDim, 0, _set_ptr->stream>>>(device_random_8dtzyx, _set_ptr->device_params, _set_ptr->host_params[_SEED_]);
            _make_gauss_gauge<T><<<_set_ptr->gridDim, _set_ptr->blockDim, 0, _set_ptr->stream>>>(device_U, device_random_8dtzyx, _set_ptr->device_params, _set_ptr->sigma());
            checkCudaErrors(cudaStreamSynchronize(_set_ptr->stream));
            auto end = std::chrono::high_resolution_clock::now();
            auto duration =
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                    .count();
            printf("multi-gpu make gauss gauge total time: (with malloc free memcpy) :%.9lf "
                   "sec\n",
                   double(duration) / 1e9);
        }
        else
        {
            checkCudaErrors(cudaStreamSynchronize(_set_ptr->stream));
            checkCudaErrors(
                cudaMallocAsync(&device_random_8dtzyx, _set_ptr->lat_4dim * _LAT_S_ * sizeof(LatticeComplex<T>), _set_ptr->stream));
            give_random_8dtzyx<T><<<_set_ptr->gridDim, _set_ptr->blockDim, 0, _set_ptr->stream>>>(device_random_8dtzyx, _set_ptr->device_params, _set_ptr->host_params[_SEED_]);
            _make_gauss_gauge<T><<<_set_ptr->gridDim, _set_ptr->blockDim, 0, _set_ptr->stream>>>(device_U, device_random_8dtzyx, _set_ptr->device_params, _set_ptr->sigma());
            checkCudaErrors(cudaStreamSynchronize(_set_ptr->stream));
        }
    }
    //@@@CUDA_TEMPLATE_FOR_DEVICE@@@
    template void make_gauss_gauge<double>(void *device_U, void *set_ptr);
    //@@@CUDA_TEMPLATE_FOR_DEVICE@@@
    template void make_gauss_gauge<float>(void *device_U, void *set_ptr);
}