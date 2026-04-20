#include "../include/qcu.h"
#pragma optimize(5)
namespace qcu {
template <typename T>
__global__ void pick_up_u_x(void *device_U, void *device_params,
                            void *device_u_b_x_send_vec,
                            void *device_u_f_x_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tmp0 = idx;
  int *params = static_cast<int *>(device_params);
  int lat_x = 1;
  int lat_y = params[_LAT_Y_];
  int lat_z = params[_LAT_Z_];
  int lat_t = params[_LAT_T_];
  int lat_xyzt = params[_LAT_XYZT_];
  int tmp1;
  tmp1 = lat_y * lat_z * lat_t;
  int x = tmp0 / tmp1;
  tmp0 -= x * tmp1;
  tmp1 = lat_z * lat_t;
  int y = tmp0 / tmp1;
  tmp0 -= y * tmp1;
  int z = tmp0 / lat_t;
  int t = tmp0 - z * lat_t;
  lat_x = params[_LAT_X_];
  LatticeComplex<T> *origin_U = static_cast<LatticeComplex<T> *>(device_U);
  LatticeComplex<T> *tmp_U;
  // u_1dim_send_vec
  //// x
  LatticeComplex<T> *u_b_x_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_b_x_send_vec) + idx);
  LatticeComplex<T> *u_f_x_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_f_x_send_vec) + idx);
  // b_x
  tmp_U = (origin_U + ((((0) * lat_y + y) * lat_z + z) * lat_t + t));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_b_x_send_vec[i * lat_xyzt / lat_x] = tmp_U[i * lat_xyzt];
    printf("pick up b_x:idx,x,y,z,t:%d,%d,%d,%d,%d,tmp_U[i * lat_xyzt]._data.x:%e\n", idx,
           x, y, z, t, tmp_U[i * lat_xyzt]._data.x);
    printf("pick up b_x:idx,x,y,z,t:%d,%d,%d,%d,%d,tmp_U[i * lat_xyzt]._data.x:%e\n", idx,
           x, y, z, t, tmp_U[i * lat_xyzt]._data.y);
  }
  // for (int p = 0; p < _LAT_P_; p++) {
  //   for (int i = 0; i < _LAT_CCD_; i++) {
  //     u_b_x_send_vec[(p * _LAT_CCD_ + i) * lat_xyzt / lat_x]._data.x =
  //         -(p * 10000 + x * 1000 + y * 100 + z * 10 + t);
  //     u_b_x_send_vec[(p * _LAT_CCD_ + i) * lat_xyzt / lat_x]._data.y = 1.0;
  //   }
  // }
  // f_x
  tmp_U = (origin_U + ((((lat_x - 1) * lat_y + y) * lat_z + z) * lat_t + t));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_f_x_send_vec[i * lat_xyzt / lat_x] = tmp_U[i * lat_xyzt];
    printf("pick up f_x:idx,x,y,z,t:%d,%d,%d,%d,%d,tmp_U[i * lat_xyzt]._data.x:%e\n", idx,
           x, y, z, t, tmp_U[i * lat_xyzt]._data.x);
    printf("pick up f_x:idx,x,y,z,t:%d,%d,%d,%d,%d,tmp_U[i * lat_xyzt]._data.x:%e\n", idx,
           x, y, z, t, tmp_U[i * lat_xyzt]._data.y);
  }
  // for (int p = 0; p < _LAT_P_; p++) {
  //   for (int i = 0; i < _LAT_CCD_; i++) {
  //     u_f_x_send_vec[(p * _LAT_CCD_ + i) * lat_xyzt / lat_x]._data.x =
  //         (p * 10000 + x * 1000 + y * 100 + z * 10 + t);
  //     u_f_x_send_vec[(p * _LAT_CCD_ + i) * lat_xyzt / lat_x]._data.y = 1.0;
  //   }
  // }
}
template <typename T>
__global__ void pick_up_u_y(void *device_U, void *device_params,
                            void *device_u_b_y_send_vec,
                            void *device_u_f_y_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tmp0 = idx;
  int *params = static_cast<int *>(device_params);
  // int lat_x = params[_LAT_X_];
  int lat_y = 1;
  int lat_z = params[_LAT_Z_];
  int lat_t = params[_LAT_T_];
  int lat_xyzt = params[_LAT_XYZT_];
  int tmp1;
  tmp1 = lat_y * lat_z * lat_t;
  int x = tmp0 / tmp1;
  tmp0 -= x * tmp1;
  tmp1 = lat_z * lat_t;
  int y = tmp0 / tmp1;
  tmp0 -= y * tmp1;
  int z = tmp0 / lat_t;
  int t = tmp0 - z * lat_t;
  lat_y = params[_LAT_Y_];
  LatticeComplex<T> *origin_U = static_cast<LatticeComplex<T> *>(device_U);
  LatticeComplex<T> *tmp_U;
  // u_1dim_send_vec
  //// y
  LatticeComplex<T> *u_b_y_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_b_y_send_vec) + idx);
  LatticeComplex<T> *u_f_y_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_f_y_send_vec) + idx);
  // b_y
  tmp_U = (origin_U + ((((x)*lat_y + 0) * lat_z + z) * lat_t + t));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_b_y_send_vec[i * lat_xyzt / lat_y] = tmp_U[i * lat_xyzt];
  }
  // f_y
  tmp_U = (origin_U + ((((x)*lat_y + lat_y - 1) * lat_z + z) * lat_t + t));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_f_y_send_vec[i * lat_xyzt / lat_y] = tmp_U[i * lat_xyzt];
  }
}
template <typename T>
__global__ void pick_up_u_z(void *device_U, void *device_params,
                            void *device_u_b_z_send_vec,
                            void *device_u_f_z_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tmp0 = idx;
  int *params = static_cast<int *>(device_params);
  // int lat_x = params[_LAT_X_];
  int lat_y = params[_LAT_Y_];
  int lat_z = 1;
  int lat_t = params[_LAT_T_];
  int lat_xyzt = params[_LAT_XYZT_];
  int tmp1;
  tmp1 = lat_y * lat_z * lat_t;
  int x = tmp0 / tmp1;
  tmp0 -= x * tmp1;
  tmp1 = lat_z * lat_t;
  int y = tmp0 / tmp1;
  tmp0 -= y * tmp1;
  int z = tmp0 / lat_t;
  int t = tmp0 - z * lat_t;
  lat_z = params[_LAT_Z_];
  LatticeComplex<T> *origin_U = static_cast<LatticeComplex<T> *>(device_U);
  LatticeComplex<T> *tmp_U;
  // u_1dim_send_vec
  //// z
  LatticeComplex<T> *u_b_z_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_b_z_send_vec) + idx);
  LatticeComplex<T> *u_f_z_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_f_z_send_vec) + idx);
  // b_z
  tmp_U = (origin_U + ((((x)*lat_y + y) * lat_z + 0) * lat_t + t));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_b_z_send_vec[i * lat_xyzt / lat_z] = tmp_U[i * lat_xyzt];
  }
  // f_z
  tmp_U = (origin_U + ((((x)*lat_y + y) * lat_z + lat_z - 1) * lat_t + t));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_f_z_send_vec[i * lat_xyzt / lat_z] = tmp_U[i * lat_xyzt];
  }
}
template <typename T>
__global__ void pick_up_u_t(void *device_U, void *device_params,
                            void *device_u_b_t_send_vec,
                            void *device_u_f_t_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tmp0 = idx;
  int *params = static_cast<int *>(device_params);
  // int lat_x = params[_LAT_X_];
  int lat_y = params[_LAT_Y_];
  int lat_z = params[_LAT_Z_];
  int lat_t = 1;
  int lat_xyzt = params[_LAT_XYZT_];
  int tmp1;
  tmp1 = lat_y * lat_z * lat_t;
  int x = tmp0 / tmp1;
  tmp0 -= x * tmp1;
  tmp1 = lat_z * lat_t;
  int y = tmp0 / tmp1;
  tmp0 -= y * tmp1;
  int z = tmp0 / lat_t;
  // int t = tmp0 - z * lat_t;
  lat_t = params[_LAT_T_];
  LatticeComplex<T> *origin_U = static_cast<LatticeComplex<T> *>(device_U);
  LatticeComplex<T> *tmp_U;
  // u_1dim_send_vec
  //// t
  LatticeComplex<T> *u_b_t_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_b_t_send_vec) + idx);
  LatticeComplex<T> *u_f_t_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_f_t_send_vec) + idx);
  // b_t
  tmp_U = (origin_U + ((((x)*lat_y + y) * lat_z + z) * lat_t + 0));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_b_t_send_vec[i * lat_xyzt / lat_t] = tmp_U[i * lat_xyzt];
  }
  // f_t
  tmp_U = (origin_U + ((((x)*lat_y + y) * lat_z + z) * lat_t + lat_t - 1));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_f_t_send_vec[i * lat_xyzt / lat_t] = tmp_U[i * lat_xyzt];
  }
}
template <typename T>
__global__ void
pick_up_u_xy(void *device_U, void *device_params,
             void *device_u_b_x_b_y_send_vec, void *device_u_f_x_b_y_send_vec,
             void *device_u_b_x_f_y_send_vec, void *device_u_f_x_f_y_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tmp0 = idx;
  int *params = static_cast<int *>(device_params);
  int lat_x = 1;
  int lat_y = 1;
  int lat_z = params[_LAT_Z_];
  int lat_t = params[_LAT_T_];
  int lat_xyzt = params[_LAT_XYZT_];
  int tmp1;
  tmp1 = lat_y * lat_z * lat_t;
  int x = tmp0 / tmp1;
  tmp0 -= x * tmp1;
  tmp1 = lat_z * lat_t;
  int y = tmp0 / tmp1;
  tmp0 -= y * tmp1;
  int z = tmp0 / lat_t;
  int t = tmp0 - z * lat_t;
  lat_x = params[_LAT_X_];
  lat_y = params[_LAT_Y_];
  LatticeComplex<T> *origin_U = static_cast<LatticeComplex<T> *>(device_U);
  LatticeComplex<T> *tmp_U;
  // u_2dim_send_vec
  //// xy
  LatticeComplex<T> *u_b_x_b_y_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_b_x_b_y_send_vec) + idx);
  LatticeComplex<T> *u_f_x_b_y_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_f_x_b_y_send_vec) + idx);
  LatticeComplex<T> *u_b_x_f_y_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_b_x_f_y_send_vec) + idx);
  LatticeComplex<T> *u_f_x_f_y_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_f_x_f_y_send_vec) + idx);
  // b_x_b_y
  tmp_U = (origin_U + ((((0) * lat_y + 0) * lat_z + z) * lat_t + t));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_b_x_b_y_send_vec[i * lat_xyzt / lat_x / lat_y] = tmp_U[i * lat_xyzt];
  }
  // f_x_b_y
  tmp_U = (origin_U + ((((lat_x - 1) * lat_y + 0) * lat_z + z) * lat_t + t));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_f_x_b_y_send_vec[i * lat_xyzt / lat_x / lat_y] = tmp_U[i * lat_xyzt];
  }
  // b_x_f_y
  tmp_U = (origin_U + ((((0) * lat_y + lat_y - 1) * lat_z + z) * lat_t + t));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_b_x_f_y_send_vec[i * lat_xyzt / lat_x / lat_y] = tmp_U[i * lat_xyzt];
  }
  // f_x_f_y
  tmp_U = (origin_U +
           ((((lat_x - 1) * lat_y + lat_y - 1) * lat_z + z) * lat_t + t));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_f_x_f_y_send_vec[i * lat_xyzt / lat_x / lat_y] = tmp_U[i * lat_xyzt];
  }
}
template <typename T>
__global__ void
pick_up_u_xz(void *device_U, void *device_params,
             void *device_u_b_x_b_z_send_vec, void *device_u_f_x_b_z_send_vec,
             void *device_u_b_x_f_z_send_vec, void *device_u_f_x_f_z_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tmp0 = idx;
  int *params = static_cast<int *>(device_params);
  int lat_x = 1;
  int lat_y = params[_LAT_Y_];
  int lat_z = 1;
  int lat_t = params[_LAT_T_];
  int lat_xyzt = params[_LAT_XYZT_];
  int tmp1;
  tmp1 = lat_y * lat_z * lat_t;
  int x = tmp0 / tmp1;
  tmp0 -= x * tmp1;
  tmp1 = lat_z * lat_t;
  int y = tmp0 / tmp1;
  tmp0 -= y * tmp1;
  int z = tmp0 / lat_t;
  int t = tmp0 - z * lat_t;
  lat_x = params[_LAT_X_];
  lat_z = params[_LAT_Z_];
  LatticeComplex<T> *origin_U = static_cast<LatticeComplex<T> *>(device_U);
  LatticeComplex<T> *tmp_U;
  // u_2dim_send_vec
  // xz
  LatticeComplex<T> *u_b_x_b_z_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_b_x_b_z_send_vec) + idx);
  LatticeComplex<T> *u_f_x_b_z_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_f_x_b_z_send_vec) + idx);
  LatticeComplex<T> *u_b_x_f_z_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_b_x_f_z_send_vec) + idx);
  LatticeComplex<T> *u_f_x_f_z_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_f_x_f_z_send_vec) + idx);
  // b_x_b_z
  tmp_U = (origin_U + ((((0) * lat_y + y) * lat_z + 0) * lat_t + t));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_b_x_b_z_send_vec[i * lat_xyzt / lat_x / lat_z] = tmp_U[i * lat_xyzt];
  }
  // f_x_b_z
  tmp_U = (origin_U + ((((lat_x - 1) * lat_y + y) * lat_z + 0) * lat_t + t));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_f_x_b_z_send_vec[i * lat_xyzt / lat_x / lat_z] = tmp_U[i * lat_xyzt];
  }
  // b_x_f_z
  tmp_U = (origin_U + ((((0) * lat_y + y) * lat_z + lat_z - 1) * lat_t + t));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_b_x_f_z_send_vec[i * lat_xyzt / lat_x / lat_z] = tmp_U[i * lat_xyzt];
  }
  // f_x_f_z
  tmp_U = (origin_U +
           ((((lat_x - 1) * lat_y + y) * lat_z + lat_z - 1) * lat_t + t));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_f_x_f_z_send_vec[i * lat_xyzt / lat_x / lat_z] = tmp_U[i * lat_xyzt];
  }
}
template <typename T>
__global__ void
pick_up_u_xt(void *device_U, void *device_params,
             void *device_u_b_x_b_t_send_vec, void *device_u_f_x_b_t_send_vec,
             void *device_u_b_x_f_t_send_vec, void *device_u_f_x_f_t_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tmp0 = idx;
  int *params = static_cast<int *>(device_params);
  int lat_x = 1;
  int lat_y = params[_LAT_Y_];
  int lat_z = params[_LAT_Z_];
  int lat_t = 1;
  int lat_xyzt = params[_LAT_XYZT_];
  int tmp1;
  tmp1 = lat_y * lat_z * lat_t;
  int x = tmp0 / tmp1;
  tmp0 -= x * tmp1;
  tmp1 = lat_z * lat_t;
  int y = tmp0 / tmp1;
  tmp0 -= y * tmp1;
  int z = tmp0 / lat_t;
  // int t = tmp0 - z * lat_t;
  lat_x = params[_LAT_X_];
  lat_t = params[_LAT_T_];
  LatticeComplex<T> *origin_U = static_cast<LatticeComplex<T> *>(device_U);
  LatticeComplex<T> *tmp_U;
  // u_2dim_send_vec
  // xt
  LatticeComplex<T> *u_b_x_b_t_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_b_x_b_t_send_vec) + idx);
  LatticeComplex<T> *u_f_x_b_t_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_f_x_b_t_send_vec) + idx);
  LatticeComplex<T> *u_b_x_f_t_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_b_x_f_t_send_vec) + idx);
  LatticeComplex<T> *u_f_x_f_t_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_f_x_f_t_send_vec) + idx);
  // b_x_b_t
  tmp_U = (origin_U + ((((0) * lat_y + y) * lat_z + z) * lat_t + 0));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_b_x_b_t_send_vec[i * lat_xyzt / lat_x / lat_t] = tmp_U[i * lat_xyzt];
  }
  // f_x_b_t
  tmp_U = (origin_U + ((((lat_x - 1) * lat_y + y) * lat_z + z) * lat_t + 0));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_f_x_b_t_send_vec[i * lat_xyzt / lat_x / lat_t] = tmp_U[i * lat_xyzt];
  }
  // b_x_f_t
  tmp_U = (origin_U + ((((0) * lat_y + y) * lat_z + z) * lat_t + lat_t - 1));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_b_x_f_t_send_vec[i * lat_xyzt / lat_x / lat_t] = tmp_U[i * lat_xyzt];
  }
  // f_x_f_t
  tmp_U = (origin_U +
           ((((lat_x - 1) * lat_y + y) * lat_z + z) * lat_t + lat_t - 1));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_f_x_f_t_send_vec[i * lat_xyzt / lat_x / lat_t] = tmp_U[i * lat_xyzt];
  }
}
template <typename T>
__global__ void
pick_up_u_yz(void *device_U, void *device_params,
             void *device_u_b_y_b_z_send_vec, void *device_u_f_y_b_z_send_vec,
             void *device_u_b_y_f_z_send_vec, void *device_u_f_y_f_z_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tmp0 = idx;
  int *params = static_cast<int *>(device_params);
  // int lat_x = params[_LAT_X_];
  int lat_y = 1;
  int lat_z = 1;
  int lat_t = params[_LAT_T_];
  int lat_xyzt = params[_LAT_XYZT_];
  int tmp1;
  tmp1 = lat_y * lat_z * lat_t;
  int x = tmp0 / tmp1;
  tmp0 -= x * tmp1;
  tmp1 = lat_z * lat_t;
  int y = tmp0 / tmp1;
  tmp0 -= y * tmp1;
  int z = tmp0 / lat_t;
  int t = tmp0 - z * lat_t;
  lat_y = params[_LAT_Y_];
  lat_z = params[_LAT_Z_];
  LatticeComplex<T> *origin_U = static_cast<LatticeComplex<T> *>(device_U);
  LatticeComplex<T> *tmp_U;
  // u_2dim_send_vec
  // yz
  LatticeComplex<T> *u_b_y_b_z_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_b_y_b_z_send_vec) + idx);
  LatticeComplex<T> *u_f_y_b_z_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_f_y_b_z_send_vec) + idx);
  LatticeComplex<T> *u_b_y_f_z_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_b_y_f_z_send_vec) + idx);
  LatticeComplex<T> *u_f_y_f_z_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_f_y_f_z_send_vec) + idx);
  // b_y_b_z
  tmp_U = (origin_U + ((((x)*lat_y + 0) * lat_z + 0) * lat_t + t));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_b_y_b_z_send_vec[i * lat_xyzt / lat_y / lat_z] = tmp_U[i * lat_xyzt];
  }
  // f_y_b_z
  tmp_U = (origin_U + ((((x)*lat_y + lat_y - 1) * lat_z + 0) * lat_t + t));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_f_y_b_z_send_vec[i * lat_xyzt / lat_y / lat_z] = tmp_U[i * lat_xyzt];
  }
  // b_y_f_z
  tmp_U = (origin_U + ((((x)*lat_y + 0) * lat_z + lat_z - 1) * lat_t + t));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_b_y_f_z_send_vec[i * lat_xyzt / lat_y / lat_z] = tmp_U[i * lat_xyzt];
  }
  // f_y_f_z
  tmp_U =
      (origin_U + ((((x)*lat_y + lat_y - 1) * lat_z + lat_z - 1) * lat_t + t));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_f_y_f_z_send_vec[i * lat_xyzt / lat_y / lat_z] = tmp_U[i * lat_xyzt];
  }
}
template <typename T>
__global__ void
pick_up_u_yt(void *device_U, void *device_params,
             void *device_u_b_y_b_t_send_vec, void *device_u_f_y_b_t_send_vec,
             void *device_u_b_y_f_t_send_vec, void *device_u_f_y_f_t_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tmp0 = idx;
  int *params = static_cast<int *>(device_params);
  // int lat_x = params[_LAT_X_];
  int lat_y = 1;
  int lat_z = params[_LAT_Z_];
  int lat_t = 1;
  int lat_xyzt = params[_LAT_XYZT_];
  int tmp1;
  tmp1 = lat_y * lat_z * lat_t;
  int x = tmp0 / tmp1;
  tmp0 -= x * tmp1;
  tmp1 = lat_z * lat_t;
  int y = tmp0 / tmp1;
  tmp0 -= y * tmp1;
  int z = tmp0 / lat_t;
  // int t = tmp0 - z * lat_t;
  lat_y = params[_LAT_Y_];
  lat_t = params[_LAT_T_];
  LatticeComplex<T> *origin_U = static_cast<LatticeComplex<T> *>(device_U);
  LatticeComplex<T> *tmp_U;
  // u_2dim_send_vec
  // yt
  LatticeComplex<T> *u_b_y_b_t_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_b_y_b_t_send_vec) + idx);
  LatticeComplex<T> *u_f_y_b_t_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_f_y_b_t_send_vec) + idx);
  LatticeComplex<T> *u_b_y_f_t_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_b_y_f_t_send_vec) + idx);
  LatticeComplex<T> *u_f_y_f_t_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_f_y_f_t_send_vec) + idx);
  // b_y_b_t
  tmp_U = (origin_U + ((((x)*lat_y + 0) * lat_z + z) * lat_t + 0));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_b_y_b_t_send_vec[i * lat_xyzt / lat_y / lat_t] = tmp_U[i * lat_xyzt];
  }
  // f_y_b_t
  tmp_U = (origin_U + ((((x)*lat_y + lat_y - 1) * lat_z + z) * lat_t + 0));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_f_y_b_t_send_vec[i * lat_xyzt / lat_y / lat_t] = tmp_U[i * lat_xyzt];
  }
  // b_y_f_t
  tmp_U = (origin_U + ((((x)*lat_y + 0) * lat_z + z) * lat_t + lat_t - 1));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_b_y_f_t_send_vec[i * lat_xyzt / lat_y / lat_t] = tmp_U[i * lat_xyzt];
  }
  // f_y_f_t
  tmp_U =
      (origin_U + ((((x)*lat_y + lat_y - 1) * lat_z + z) * lat_t + lat_t - 1));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_f_y_f_t_send_vec[i * lat_xyzt / lat_y / lat_t] = tmp_U[i * lat_xyzt];
  }
}
template <typename T>
__global__ void
pick_up_u_zt(void *device_U, void *device_params,
             void *device_u_b_z_b_t_send_vec, void *device_u_f_z_b_t_send_vec,
             void *device_u_b_z_f_t_send_vec, void *device_u_f_z_f_t_send_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tmp0 = idx;
  int *params = static_cast<int *>(device_params);
  // int lat_x = params[_LAT_X_];
  int lat_y = params[_LAT_Y_];
  int lat_z = 1;
  int lat_t = 1;
  int lat_xyzt = params[_LAT_XYZT_];
  int tmp1;
  tmp1 = lat_y * lat_z * lat_t;
  int x = tmp0 / tmp1;
  tmp0 -= x * tmp1;
  tmp1 = lat_z * lat_t;
  int y = tmp0 / tmp1;
  tmp0 -= y * tmp1;
  // int z = tmp0 / lat_t;
  // int t = tmp0 - z * lat_t;
  lat_z = params[_LAT_Z_];
  lat_t = params[_LAT_T_];
  LatticeComplex<T> *origin_U = static_cast<LatticeComplex<T> *>(device_U);
  LatticeComplex<T> *tmp_U;
  // u_2dim_send_vec
  LatticeComplex<T> *u_b_z_b_t_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_b_z_b_t_send_vec) + idx);
  LatticeComplex<T> *u_f_z_b_t_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_f_z_b_t_send_vec) + idx);
  LatticeComplex<T> *u_b_z_f_t_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_b_z_f_t_send_vec) + idx);
  LatticeComplex<T> *u_f_z_f_t_send_vec =
      (static_cast<LatticeComplex<T> *>(device_u_f_z_f_t_send_vec) + idx);
  // b_z_b_t
  tmp_U = (origin_U + ((((x)*lat_y + y) * lat_z + 0) * lat_t + 0));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_b_z_b_t_send_vec[i * lat_xyzt / lat_z / lat_t] = tmp_U[i * lat_xyzt];
  }
  // f_z_b_t
  tmp_U = (origin_U + ((((x)*lat_y + y) * lat_z + lat_z - 1) * lat_t + 0));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_f_z_b_t_send_vec[i * lat_xyzt / lat_z / lat_t] = tmp_U[i * lat_xyzt];
  }
  // b_z_f_t
  tmp_U = (origin_U + ((((x)*lat_y + y) * lat_z + 0) * lat_t + lat_t - 1));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_b_z_f_t_send_vec[i * lat_xyzt / lat_z / lat_t] = tmp_U[i * lat_xyzt];
  }
  // f_z_f_t
  tmp_U =
      (origin_U + ((((x)*lat_y + y) * lat_z + lat_z - 1) * lat_t + lat_t - 1));
  for (int i = 0; i < _LAT_PCCD_; i++) {
    u_f_z_f_t_send_vec[i * lat_xyzt / lat_z / lat_t] = tmp_U[i * lat_xyzt];
  }
}
//@@@CUDA_TEMPLATE_FOR_DEVICE@@@
template __global__ void pick_up_u_x<double>(void *device_U,
                                             void *device_params,
                                             void *device_u_b_x_send_vec,
                                             void *device_u_f_x_send_vec);
template __global__ void pick_up_u_y<double>(void *device_U,
                                             void *device_params,
                                             void *device_u_b_y_send_vec,
                                             void *device_u_f_y_send_vec);
template __global__ void pick_up_u_z<double>(void *device_U,
                                             void *device_params,
                                             void *device_u_b_z_send_vec,
                                             void *device_u_f_z_send_vec);
template __global__ void pick_up_u_t<double>(void *device_U,
                                             void *device_params,
                                             void *device_u_b_t_send_vec,
                                             void *device_u_f_t_send_vec);
template __global__ void pick_up_u_xy<double>(void *device_U,
                                              void *device_params,
                                              void *device_u_b_x_b_y_send_vec,
                                              void *device_u_f_x_b_y_send_vec,
                                              void *device_u_b_x_f_y_send_vec,
                                              void *device_u_f_x_f_y_send_vec);
template __global__ void pick_up_u_xz<double>(void *device_U,
                                              void *device_params,
                                              void *device_u_b_x_b_z_send_vec,
                                              void *device_u_f_x_b_z_send_vec,
                                              void *device_u_b_x_f_z_send_vec,
                                              void *device_u_f_x_f_z_send_vec);
template __global__ void pick_up_u_xt<double>(void *device_U,
                                              void *device_params,
                                              void *device_u_b_x_b_t_send_vec,
                                              void *device_u_f_x_b_t_send_vec,
                                              void *device_u_b_x_f_t_send_vec,
                                              void *device_u_f_x_f_t_send_vec);
template __global__ void pick_up_u_yz<double>(void *device_U,
                                              void *device_params,
                                              void *device_u_b_y_b_z_send_vec,
                                              void *device_u_f_y_b_z_send_vec,
                                              void *device_u_b_y_f_z_send_vec,
                                              void *device_u_f_y_f_z_send_vec);
template __global__ void pick_up_u_yt<double>(void *device_U,
                                              void *device_params,
                                              void *device_u_b_y_b_t_send_vec,
                                              void *device_u_f_y_b_t_send_vec,
                                              void *device_u_b_y_f_t_send_vec,
                                              void *device_u_f_y_f_t_send_vec);
template __global__ void pick_up_u_zt<double>(void *device_U,
                                              void *device_params,
                                              void *device_u_b_z_b_t_send_vec,
                                              void *device_u_f_z_b_t_send_vec,
                                              void *device_u_b_z_f_t_send_vec,
                                              void *device_u_f_z_f_t_send_vec);
//@@@CUDA_TEMPLATE_FOR_DEVICE@@@
template __global__ void pick_up_u_x<float>(void *device_U, void *device_params,
                                            void *device_u_b_x_send_vec,
                                            void *device_u_f_x_send_vec);
template __global__ void pick_up_u_y<float>(void *device_U, void *device_params,
                                            void *device_u_b_y_send_vec,
                                            void *device_u_f_y_send_vec);
template __global__ void pick_up_u_z<float>(void *device_U, void *device_params,
                                            void *device_u_b_z_send_vec,
                                            void *device_u_f_z_send_vec);
template __global__ void pick_up_u_t<float>(void *device_U, void *device_params,
                                            void *device_u_b_t_send_vec,
                                            void *device_u_f_t_send_vec);
template __global__ void pick_up_u_xy<float>(void *device_U,
                                             void *device_params,
                                             void *device_u_b_x_b_y_send_vec,
                                             void *device_u_f_x_b_y_send_vec,
                                             void *device_u_b_x_f_y_send_vec,
                                             void *device_u_f_x_f_y_send_vec);
template __global__ void pick_up_u_xz<float>(void *device_U,
                                             void *device_params,
                                             void *device_u_b_x_b_z_send_vec,
                                             void *device_u_f_x_b_z_send_vec,
                                             void *device_u_b_x_f_z_send_vec,
                                             void *device_u_f_x_f_z_send_vec);
template __global__ void pick_up_u_xt<float>(void *device_U,
                                             void *device_params,
                                             void *device_u_b_x_b_t_send_vec,
                                             void *device_u_f_x_b_t_send_vec,
                                             void *device_u_b_x_f_t_send_vec,
                                             void *device_u_f_x_f_t_send_vec);
template __global__ void pick_up_u_yz<float>(void *device_U,
                                             void *device_params,
                                             void *device_u_b_y_b_z_send_vec,
                                             void *device_u_f_y_b_z_send_vec,
                                             void *device_u_b_y_f_z_send_vec,
                                             void *device_u_f_y_f_z_send_vec);
template __global__ void pick_up_u_yt<float>(void *device_U,
                                             void *device_params,
                                             void *device_u_b_y_b_t_send_vec,
                                             void *device_u_f_y_b_t_send_vec,
                                             void *device_u_b_y_f_t_send_vec,
                                             void *device_u_f_y_f_t_send_vec);
template __global__ void pick_up_u_zt<float>(void *device_U,
                                             void *device_params,
                                             void *device_u_b_z_b_t_send_vec,
                                             void *device_u_f_z_b_t_send_vec,
                                             void *device_u_b_z_f_t_send_vec,
                                             void *device_u_f_z_f_t_send_vec);
} // namespace qcu