#include "../include/qcu.h"
#pragma optimize(5)
namespace qcu
{
    template <typename T>
    __global__ void make_gauss_gauge(void *device_x, void *device_e,
                                     void *device_vals)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        LatticeComplex<T> *x = (static_cast<LatticeComplex<T> *>(device_x) + idx);
        LatticeComplex<T> *e = (static_cast<LatticeComplex<T> *>(device_e) + idx);
        int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
        for (int i = 0; i < _LAT_SC_ * _; i += _)
        {
            x[i] = x[i] + e[i];
        }
    }
    //@@@CUDA_TEMPLATE_FOR_DEVICE@@@
    template __global__ void make_gauss_gauge<double>(void *device_x, void *device_p,
                                                      void *device_vals);
    //@@@CUDA_TEMPLATE_FOR_DEVICE@@@
    template __global__ void make_gauss_gauge<float>(void *device_x, void *device_p,
                                                     void *device_vals);
}
