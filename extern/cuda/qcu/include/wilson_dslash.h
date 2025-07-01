#ifndef _WILSON_DSLASH_H
#define _WILSON_DSLASH_H
namespace qcu
{
  template <typename T>
  __global__ void wilson_dslash(void *device_U, void *device_src,
                                void *device_dest, void *device_params);
  template <typename T>
  __global__ void wilson_dslash_inside(void *device_U, void *device_src,
                                       void *device_dest, void *device_params);
  template <typename T>
  __global__ void wilson_dslash_x_send(void *device_U, void *device_src,
                                       void *device_params,
                                       void *device_b_x_send_vec,
                                       void *device_f_x_send_vec);
  template <typename T>
  __global__ void wilson_dslash_x_recv(void *device_U, void *device_dest,
                                       void *device_params,
                                       void *device_b_x_recv_vec,
                                       void *device_f_x_recv_vec);
  template <typename T>
  __global__ void wilson_dslash_y_send(void *device_U, void *device_src,
                                       void *device_params,
                                       void *device_b_y_send_vec,
                                       void *device_f_y_send_vec);
  template <typename T>
  __global__ void wilson_dslash_y_recv(void *device_U, void *device_dest,
                                       void *device_params,
                                       void *device_b_y_recv_vec,
                                       void *device_f_y_recv_vec);
  template <typename T>
  __global__ void wilson_dslash_z_send(void *device_U, void *device_src,
                                       void *device_params,
                                       void *device_b_z_send_vec,
                                       void *device_f_z_send_vec);
  template <typename T>
  __global__ void wilson_dslash_z_recv(void *device_U, void *device_dest,
                                       void *device_params,
                                       void *device_b_z_recv_vec,
                                       void *device_f_z_recv_vec);
  template <typename T>
  __global__ void wilson_dslash_t_send(void *device_U, void *device_src,
                                       void *device_params,
                                       void *device_b_t_send_vec,
                                       void *device_f_t_send_vec);
  template <typename T>
  __global__ void wilson_dslash_t_recv(void *device_U, void *device_dest,
                                       void *device_params,
                                       void *device_b_t_recv_vec,
                                       void *device_f_t_recv_vec);
}
#endif