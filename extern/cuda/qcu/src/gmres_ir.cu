#include "../include/qcu.h"
#pragma optimize(5)
namespace qcu
{
    template <typename T>
    __global__ void gmres_ir_give_x(void *device_x, void *device_e,
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
    template <typename T>
    __global__ void gmres_ir_give_r(void *device_r, void *device_b,
                                    void *device_vals)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        LatticeComplex<T> *r = (static_cast<LatticeComplex<T> *>(device_r) + idx);
        LatticeComplex<T> *b = (static_cast<LatticeComplex<T> *>(device_b) + idx);
        int _ = int(((LatticeComplex<T> *)device_vals)[_lat_4dim_]._data.x);
        for (int i = 0; i < _LAT_SC_ * _; i += _)
        {
            r[i] = b[i] - r[i];
        }
    }
    //@@@CUDA_TEMPLATE_FOR_DEVICE@@@
    template __global__ void gmres_ir_give_x<double>(void *device_x, void *device_p,
                                                     void *device_vals);
    template __global__ void gmres_ir_give_r<double>(void *device_r, void *device_v,
                                                     void *device_vals);
    //@@@CUDA_TEMPLATE_FOR_DEVICE@@@
    template __global__ void gmres_ir_give_x<float>(void *device_x, void *device_p,
                                                    void *device_vals);
    template __global__ void gmres_ir_give_r<float>(void *device_r, void *device_v,
                                                    void *device_vals);
}
