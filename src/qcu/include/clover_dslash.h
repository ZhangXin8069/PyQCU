#ifndef _CLOVER_DSLASH_H
#define _CLOVER_DSLASH_H
namespace qcu
{
  template <typename T>
  __global__ void make_clover(void *device_U, void *device_clover,
                              void *device_params);
  template <typename T>
  __global__ void inverse_clover(void *device_clover, void *device_params);
  template <typename T>
  __global__ void give_clover(void *device_clover, void *device_dest,
                              void *device_params);
  template <typename T>
  __global__ void pick_up_u_x(void *device_U, void *device_params,
                              void *device_u_b_x_send_vec,
                              void *device_u_f_x_send_vec);
  template <typename T>
  __global__ void pick_up_u_y(void *device_U, void *device_params,
                              void *device_u_b_y_send_vec,
                              void *device_u_f_y_send_vec);
  template <typename T>
  __global__ void pick_up_u_z(void *device_U, void *device_params,
                              void *device_u_b_z_send_vec,
                              void *device_u_f_z_send_vec);
  template <typename T>
  __global__ void pick_up_u_t(void *device_U, void *device_params,
                              void *device_u_b_t_send_vec,
                              void *device_u_f_t_send_vec);
  template <typename T>
  __global__ void pick_up_u_xy(void *device_U, void *device_params,
                               void *device_u_b_x_b_y_send_vec,
                               void *device_u_f_x_b_y_send_vec,
                               void *device_u_b_x_f_y_send_vec,
                               void *device_u_f_x_f_y_send_vec);
  template <typename T>
  __global__ void pick_up_u_xz(void *device_U, void *device_params,
                               void *device_u_b_x_b_z_send_vec,
                               void *device_u_f_x_b_z_send_vec,
                               void *device_u_b_x_f_z_send_vec,
                               void *device_u_f_x_f_z_send_vec);
  template <typename T>
  __global__ void pick_up_u_xt(void *device_U, void *device_params,
                               void *device_u_b_x_b_t_send_vec,
                               void *device_u_f_x_b_t_send_vec,
                               void *device_u_b_x_f_t_send_vec,
                               void *device_u_f_x_f_t_send_vec);
  template <typename T>
  __global__ void pick_up_u_yz(void *device_U, void *device_params,
                               void *device_u_b_y_b_z_send_vec,
                               void *device_u_f_y_b_z_send_vec,
                               void *device_u_b_y_f_z_send_vec,
                               void *device_u_f_y_f_z_send_vec);
  template <typename T>
  __global__ void pick_up_u_yt(void *device_U, void *device_params,
                               void *device_u_b_y_b_t_send_vec,
                               void *device_u_f_y_b_t_send_vec,
                               void *device_u_b_y_f_t_send_vec,
                               void *device_u_f_y_f_t_send_vec);
  template <typename T>
  __global__ void pick_up_u_zt(void *device_U, void *device_params,
                               void *device_u_b_z_b_t_send_vec,
                               void *device_u_f_z_b_t_send_vec,
                               void *device_u_b_z_f_t_send_vec,
                               void *device_u_f_z_f_t_send_vec);
  template <typename T>
  __global__ void make_clover_all(
      void *device_U, void *device_clover, void *device_params,
      void *device_u_b_x_recv_vec, void *device_u_f_x_recv_vec,
      void *device_u_b_y_recv_vec, void *device_u_f_y_recv_vec,
      void *device_u_b_z_recv_vec, void *device_u_f_z_recv_vec,
      void *device_u_b_t_recv_vec, void *device_u_f_t_recv_vec,
      void *device_u_b_x_b_y_recv_vec, void *device_u_f_x_b_y_recv_vec,
      void *device_u_b_x_f_y_recv_vec, void *device_u_f_x_f_y_recv_vec,
      void *device_u_b_x_b_z_recv_vec, void *device_u_f_x_b_z_recv_vec,
      void *device_u_b_x_f_z_recv_vec, void *device_u_f_x_f_z_recv_vec,
      void *device_u_b_x_b_t_recv_vec, void *device_u_f_x_b_t_recv_vec,
      void *device_u_b_x_f_t_recv_vec, void *device_u_f_x_f_t_recv_vec,
      void *device_u_b_y_b_z_recv_vec, void *device_u_f_y_b_z_recv_vec,
      void *device_u_b_y_f_z_recv_vec, void *device_u_f_y_f_z_recv_vec,
      void *device_u_b_y_b_t_recv_vec, void *device_u_f_y_b_t_recv_vec,
      void *device_u_b_y_f_t_recv_vec, void *device_u_f_y_f_t_recv_vec,
      void *device_u_b_z_b_t_recv_vec, void *device_u_f_z_b_t_recv_vec,
      void *device_u_b_z_f_t_recv_vec, void *device_u_f_z_f_t_recv_vec);
}
#endif