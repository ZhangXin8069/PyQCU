#include "../include/qcu.h"
#pragma optimize(5)
namespace qcu
{
    template <typename T>
    __global__ void give_random_stzyx(void *device_random_stzyx, unsigned long seed)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        LatticeComplex<T> *random_stzyx =
            static_cast<LatticeComplex<T> *>(device_random_stzyx);
        curandState state_real, state_imag;
        curand_init(seed, idx, 0, &state_real);
        curand_init(seed, idx, 1, &state_imag);
        for (int i = 0; i < _LAT_S_; ++i)
        {
            random_stzyx[idx * _LAT_S_ + i]._data.x = curand_uniform(&state_real);
            random_stzyx[idx * _LAT_S_ + i]._data.y = curand_uniform(&state_real);
        }
    }
    // 3x3 complex matrix multiplication
    template <typename T>
    __device__ void su3_matrix_multiply(LatticeComplex<T> a[_LAT_CC_], LatticeComplex<T> b[_LAT_CC_], LatticeComplex<T> result[_LAT_CC_])
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
                    real += a[idx_a]._data.x * b[idx_b]._data.x - a[idx_a]._data.y * b[idx_b]._data.y;
                    imag += a[idx_a]._data.x * b[idx_b]._data.y + a[idx_a]._data.y * b[idx_b]._data.x;
                }
                result[i * _LAT_C_ + j]._data.x = real;
                result[i * _LAT_C_ + j]._data.y = imag;
            }
        }
    }
    // 3x3 matrix exponential (Taylor expansion up to 6th order)
    template <typename T>
    __device__ void su3_matrix_exponential(LatticeComplex<T> a[_LAT_CC_], LatticeComplex<T> result[_LAT_CC_])
    {
        LatticeComplex<T> identity[_LAT_CC_] = {
            {1, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}};
        // Copy a to current
        LatticeComplex<T> current[_LAT_CC_];
        for (int i = 0; i < _LAT_CC_; i++)
        {
            current[i] = a[i];
            result[i] = identity[i]; // initialize as identity matrix
        }
        LatticeComplex<T> term[_LAT_CC_];
        for (int i = 0; i < _LAT_CC_; i++)
        {
            term[i] = current[i];
            result[i]._data.x += term[i]._data.x;
            result[i]._data.y += term[i]._data.y;
        }
        // Precomputed reciprocal factorials up to 6!
        const T inv_factorials[7] = {
            1.0,         // 0! = 1
            1.0,         // 1! = 1
            1.0 / 2.0,   // 2! = 2
            1.0 / 6.0,   // 3! = 6
            1.0 / 24.0,  // 4! = 24
            1.0 / 120.0, // 5! = 120
            1.0 / 720.0  // 6! = 720
        };
        // Taylor expansion up to 6th order
        for (int n = 2; n <= 6; n++)
        {
            LatticeComplex<T> new_term[_LAT_CC_];
            su3_matrix_multiply(current, term, new_term);
            T factor = inv_factorials[n];
            for (int i = 0; i < _LAT_CC_; i++)
            {
                term[i]._data.x = new_term[i]._data.x * factor;
                term[i]._data.y = new_term[i]._data.y * factor;
                result[i]._data.x += term[i]._data.x;
                result[i]._data.y += term[i]._data.y;
                current[i] = new_term[i]; // update current for the next multiplication
            }
        }
    }
    template <typename T>
    __global__ void _make_gauss_gauge(void *device_U, void *device_random_stzyx, void *device_params, T sigma)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int *params = static_cast<int *>(device_params);
        int lat_tzyx = params[_LAT_XYZT_];
        LatticeComplex<T> *random_stzyx = static_cast<LatticeComplex<T> *>(device_random_stzyx);
        LatticeComplex<T> *origin_U = ((static_cast<LatticeComplex<T> *>(device_U)) + idx);
        LatticeComplex<T> U[_LAT_CC_] = {
            {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
        LatticeComplex<T> H[_LAT_CC_] = {
            {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
        // Generate one SU(3) matrix for each direction
        for (int p = 0; p < _LAT_P_; p++)
        {
            for (int d = 0; d < _LAT_D_; d++)
            {
                // Generate 8 Gaussian random numbers
                T a[(_LAT_CC_ - 1)];
                for (int i = 0; i < (_LAT_CC_ - 1); i++)
                {
                    // Use Box-Muller transform to generate Gaussian-distributed random numbers
                    if (p == 0)
                    {
                        T u1 = random_stzyx[d * (_LAT_CC_ - 1) + i * 2]._data.x;
                        T u2 = random_stzyx[d * (_LAT_CC_ - 1) + i * 2 + 1]._data.x;
                        T r = sqrt(-2.0 * log(u1));
                        a[i] = r * cos(2.0 * M_PI * u2);
                    }
                    else
                    {
                        T u1 = random_stzyx[d * (_LAT_CC_ - 1) + i * 2]._data.y;
                        T u2 = random_stzyx[d * (_LAT_CC_ - 1) + i * 2 + 1]._data.y;
                        T r = sqrt(-2.0 * log(u1));
                        a[i] = r * cos(2.0 * M_PI * u2);
                    }
                }
                // Construct Hermitian matrix H
                    for (int i = 0; i < (_LAT_CC_ - 1); i++)
                    {
                        for (int j = 0; j < _LAT_CC_; j++)
                        {
                            int row = j / _LAT_C_;
                            int col = j % _LAT_C_;
                            int idx_h = row * _LAT_C_ + col;
                            if (i == 1 || i == 4 || i == 6)
                            {
                                // Handle matrices with imaginary unit
                                H[idx_h]._data.y += a[i] * d_gell_mann[i][j];
                            }
                            else
                            {
                                H[idx_h]._data.x += a[i] * d_gell_mann[i][j];
                            }
                        }
                    }
                }
                else
                {
                    for (int i = 0; i < (_LAT_CC_ - 1); i++)
                    {
                        for (int j = 0; j < _LAT_CC_; j++)
                        {
                            int row = j / _LAT_C_;
                            int col = j % _LAT_C_;
                            int idx_h = row * _LAT_C_ + col;
                            if (i == 1 || i == 4 || i == 6)
                            {
                                H[idx_h]._data.y += a[i] * f_gell_mann[i][j];
                            }
                            else
                            {
                                H[idx_h]._data.x += a[i] * f_gell_mann[i][j];
                            }
                        }
                    }
                }
                // Compute A = i * sigma * H
                LatticeComplex<T> A[_LAT_CC_];
                for (int i = 0; i < _LAT_CC_; i++)
                {
                    A[i]._data.x = -sigma * H[i]._data.y; // imaginary part becomes real with negative sign
                    A[i]._data.y = sigma * H[i]._data.x;  // real part becomes imaginary
                }
                // Compute U = exp(A)
                su3_matrix_exponential(A, U);
                give_U(p, origin_U, U, lat_tzyx);
            }
        }
    }
    template <typename T>
    void make_gauss_gauge(void *device_U, void *set_ptr)
    {
        void *device_random_stzyx;
        LatticeSet<T> *_set_ptr = static_cast<LatticeSet<T> *>(set_ptr);
        if (_set_ptr->host_params[_VERBOSE_])
        {
            auto start = std::chrono::high_resolution_clock::now();
            checkCudaErrors(cudaStreamSynchronize(_set_ptr->stream));
            checkCudaErrors(
                cudaMallocAsync(&device_random_stzyx, _set_ptr->lat_4dim * _LAT_S_ * sizeof(LatticeComplex<T>), _set_ptr->stream));
            give_random_stzyx<T><<<_set_ptr->gridDim, _set_ptr->blockDim, 0, _set_ptr->stream>>>(device_random_stzyx, _set_ptr->host_params[_SEED_]);
            _make_gauss_gauge<T><<<_set_ptr->gridDim, _set_ptr->blockDim, 0, _set_ptr->stream>>>(device_U, device_random_stzyx, _set_ptr->device_params, _set_ptr->sigma());
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
                cudaMallocAsync(&device_random_stzyx, _set_ptr->lat_4dim * _LAT_S_ * sizeof(LatticeComplex<T>), _set_ptr->stream));
            give_random_stzyx<T><<<_set_ptr->gridDim, _set_ptr->blockDim, 0, _set_ptr->stream>>>(device_random_stzyx, _set_ptr->host_params[_SEED_]);
            _make_gauss_gauge<T><<<_set_ptr->gridDim, _set_ptr->blockDim, 0, _set_ptr->stream>>>(device_U, device_random_stzyx, _set_ptr->device_params, _set_ptr->sigma());
            checkCudaErrors(cudaStreamSynchronize(_set_ptr->stream));
        }
    }
    //@@@CUDA_TEMPLATE_FOR_DEVICE@@@
    template void make_gauss_gauge<double>(void *device_U, void *set_ptr);
    template __global__ void _make_gauss_gauge<double>(void *device_U, void *device_random_stzyx, void *device_params, double sigma);
    //@@@CUDA_TEMPLATE_FOR_DEVICE@@@
    template void make_gauss_gauge<float>(void *device_U, void *set_ptr);
    template __global__ void _make_gauss_gauge<float>(void *device_U, void *device_random_stzyx, void *device_params, float sigma);
}
