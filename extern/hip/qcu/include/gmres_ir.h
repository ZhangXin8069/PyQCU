#ifndef _GMRES_IR_H
#define _GMRES_IR_H
#include "./lattice_complex.h"
namespace qcu
{
    template <typename T>
    __global__ void gmres_ir_give_x(void *device_x, void *device_e,
                                    void *device_vals);
    template <typename T>
    __global__ void gmres_ir_give_r(void *device_r, void *device_b,
                                    void *device_vals);
}
#endif