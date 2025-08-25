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
            random_stzyx[idx * _LAT_S_ + i]._data.y = 0.0;
        }
    }

    // 定义Gell-Mann矩阵常量
    __constant__ double d_gell_mann[8][9] = {
        {0, 1, 0, 1, 0, 0, 0, 0, 0},                               // lambda1
        {0, -1, 0, 1, 0, 0, 0, 0, 0},                              // lambda2 (虚部单位i在计算时处理)
        {1, 0, 0, 0, -1, 0, 0, 0, 0},                              // lambda3
        {0, 0, 1, 0, 0, 0, 1, 0, 0},                               // lambda4
        {0, 0, -1, 0, 0, 0, 1, 0, 0},                              // lambda5 (虚部单位i在计算时处理)
        {0, 0, 0, 0, 0, 1, 0, 1, 0},                               // lambda6
        {0, 0, 0, 0, 0, -1, 0, 1, 0},                              // lambda7 (虚部单位i在计算时处理)
        {1 / sqrt(3), 0, 0, 0, 1 / sqrt(3), 0, 0, 0, -2 / sqrt(3)} // lambda8
    };

    __constant__ float f_gell_mann[8][9] = {
        {0, 1, 0, 1, 0, 0, 0, 0, 0},
        {0, -1, 0, 1, 0, 0, 0, 0, 0},
        {1, 0, 0, 0, -1, 0, 0, 0, 0},
        {0, 0, 1, 0, 0, 0, 1, 0, 0},
        {0, 0, -1, 0, 0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0, 1, 0, 1, 0},
        {0, 0, 0, 0, 0, -1, 0, 1, 0},
        {1 / sqrtf(3), 0, 0, 0, 1 / sqrtf(3), 0, 0, 0, -2 / sqrtf(3)}};

    // 3x3复数矩阵乘法
    template <typename T>
    __device__ void su3_matrix_multiply(LatticeComplex<T> a[9], LatticeComplex<T> b[9], LatticeComplex<T> result[9])
    {
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                T real = 0.0;
                T imag = 0.0;
                for (int k = 0; k < 3; k++)
                {
                    int idx_a = i * 3 + k;
                    int idx_b = k * 3 + j;
                    real += a[idx_a]._data.x * b[idx_b]._data.x - a[idx_a]._data.y * b[idx_b]._data.y;
                    imag += a[idx_a]._data.x * b[idx_b]._data.y + a[idx_a]._data.y * b[idx_b]._data.x;
                }
                result[i * 3 + j]._data.x = real;
                result[i * 3 + j]._data.y = imag;
            }
        }
    }

    // 3x3矩阵指数计算（使用泰勒展开到6阶）
    template <typename T>
    __device__ void su3_matrix_exponential(LatticeComplex<T> a[9], LatticeComplex<T> result[9])
    {
        LatticeComplex<T> identity[9] = {
            {1, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}, {0, 0}, {0, 0}, {0, 0}, {1, 0}};

        // 复制a到current
        LatticeComplex<T> current[9];
        for (int i = 0; i < 9; i++)
        {
            current[i] = a[i];
            result[i] = identity[i]; // 初始化为单位矩阵
        }

        LatticeComplex<T> term[9];
        for (int i = 0; i < 9; i++)
        {
            term[i] = current[i];
            result[i]._data.x += term[i]._data.x;
            result[i]._data.y += term[i]._data.y;
        }

        // 泰勒展开到6阶
        for (int n = 2; n <= 6; n++)
        {
            LatticeComplex<T> new_term[9];
            su3_matrix_multiply(current, term, new_term);

            T factor = 1.0 / tgamma(n + 1);
            for (int i = 0; i < 9; i++)
            {
                term[i]._data.x = new_term[i]._data.x * factor;
                term[i]._data.y = new_term[i]._data.y * factor;
                result[i]._data.x += term[i]._data.x;
                result[i]._data.y += term[i]._data.y;

                current[i] = new_term[i]; // 更新current为下一次乘法准备
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
        LatticeComplex<T> *U = static_cast<LatticeComplex<T> *>(device_U);
        LatticeComplex<T> H[_LAT_CC_];

        LatticeComplex<T> *random_stzyx = static_cast<LatticeComplex<T> *>(device_random_stzyx);
        LatticeComplex<T> *U = static_cast<LatticeComplex<T> *>(device_U);

        // 获取当前时空点的随机数
        LatticeComplex<T> *current_randoms = &random_stzyx[idx * _LAT_S_];

        // 为每个方向生成一个SU(3)矩阵
        for (int d = 0; d < _LAT_D_; d++)
        {
            // 生成8个高斯随机数
            T a[8];
            for (int i = 0; i < 8; i++)
            {
                // 使用Box-Muller变换生成高斯分布随机数
                T u1 = current_randoms[d * 8 + i * 2]._data.x;
                T u2 = current_randoms[d * 8 + i * 2 + 1]._data.x;
                T r = sqrt(-2.0 * log(u1));
                a[i] = r * cos(2.0 * M_PI * u2);
            }

            // 构建厄米特矩阵H
            LatticeComplex<T> H[9] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};

            if constexpr (std::is_same_v<T, double>)
            {
                for (int i = 0; i < 8; i++)
                {
                    for (int j = 0; j < 9; j++)
                    {
                        int row = j / 3;
                        int col = j % 3;
                        int idx_h = row * 3 + col;

                        if (i == 1 || i == 4 || i == 6)
                        {
                            // 处理带有虚数单位的矩阵
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
                for (int i = 0; i < 8; i++)
                {
                    for (int j = 0; j < 9; j++)
                    {
                        int row = j / 3;
                        int col = j % 3;
                        int idx_h = row * 3 + col;

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

            // 计算A = i * sigma * H
            LatticeComplex<T> A[9];
            for (int i = 0; i < 9; i++)
            {
                A[i]._data.x = -sigma * H[i]._data.y; // i * H的虚部变为实部，符号为负
                A[i]._data.y = sigma * H[i]._data.x;  // i * H的实部变为虚部
            }

            // 计算U = exp(A)
            LatticeComplex<T> U_dir[9];
            su3_matrix_exponential(A, U_dir);

            // 存储结果到device_U
            int offset = idx * _LAT_DCC_ + d * _LAT_CC_;
            for (int cc = 0; cc < _LAT_CC_; cc++)
            {
                U[offset + cc] = U_dir[cc];
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
    template __global__ void make_gauss_gauge<double>(void *device_U, void *device_params, void *set_ptr);
    template void _make_gauss_gauge<double>(void *device_U, void *device_random_stzyx, void *device_params, double sigma);
    //@@@CUDA_TEMPLATE_FOR_DEVICE@@@
    template __global__ void make_gauss_gauge<float>(void *device_U, void *device_params, void *set_ptr);
    template void _make_gauss_gauge<float>(void *device_U, void *device_random_stzyx, void *device_params, float sigma);
}
