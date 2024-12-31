#ifndef _laplacian_H
#define _laplacian_H
namespace qcu
{
  template <typename T>
  __global__ void laplacian_inside(void *device_U, void *device_src,
                                   void *device_dest, void *device_params);
  template <typename T>
  __global__ void laplacian_x_send(void *device_U, void *device_src,
                                   void *device_params,
                                   void *device_b_x_send_vec,
                                   void *device_f_x_send_vec);
  template <typename T>
  __global__ void laplacian_x_recv(void *device_dest,
                                   void *device_params,
                                   void *device_b_x_recv_vec,
                                   void *device_f_x_recv_vec);
  template <typename T>
  __global__ void laplacian_y_send(void *device_U, void *device_src,
                                   void *device_params,
                                   void *device_b_y_send_vec,
                                   void *device_f_y_send_vec);
  template <typename T>
  __global__ void laplacian_y_recv(void *device_dest,
                                   void *device_params,
                                   void *device_b_y_recv_vec,
                                   void *device_f_y_recv_vec);
  template <typename T>
  __global__ void laplacian_z_send(void *device_U, void *device_src,
                                   void *device_params,
                                   void *device_b_z_send_vec,
                                   void *device_f_z_send_vec);
  template <typename T>
  __global__ void laplacian_z_recv(void *device_dest,
                                   void *device_params,
                                   void *device_b_z_recv_vec,
                                   void *device_f_z_recv_vec);
}
#endif