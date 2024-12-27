#ifndef _CG_H
#define _CG_H
#include "./lattice_complex.h"
namespace qcu
{
  template <typename T>
  __global__ void cg_give_b_e(void *device_b_e, void *device_ans_e,
                              void *device_vec0, T kappa, void *device_vals);
  template <typename T>
  __global__ void cg_give_b_o(void *device_b_o, void *device_ans_o,
                              void *device_vec1, T kappa, void *device_vals);
  template <typename T>
  __global__ void cg_give_b__o(void *device_b__o, void *device_b_o,
                               void *device_vec0, T kappa,
                               void *device_vals);
  template <typename T>
  __global__ void cg_give_r(void *device_r, void *device_b__o, void *device_vec1,
                            void *device_vals);
  template <typename T>
  __global__ void cg_give_dest_o(void *device_dest_o, void *device_src_o,
                                 void *device_vec1, T kappa,
                                 void *device_vals);
  template <typename T>
  __global__ void cg_give_1diff(void *device_vals);
  template <typename T>
  __global__ void cg_give_1beta(void *device_vals);
  template <typename T>
  __global__ void cg_give_1rho_prev(void *device_vals);
  template <typename T>
  __global__ void cg_give_1alpha(void *device_vals);
  template <typename T>
  __global__ void cg_give_p(void *device_p, void *device_r_tilde,
                            void *device_vals);
  template <typename T>
  __global__ void cg_give_x_o(void *device_x_o, void *device_p,
                              void *device_vals);
  template <typename T>
  __global__ void cg_give_r_tilde(void *device_r, void *device_v,
                                  void *device_vals);
  template <typename T>
  __global__ void cg_give_diff(void *device_x, void *device_ans, void *device_vec,
                               void *device_vals);
}
#endif