#include "../include/qcu.h"
#include <cstdio>
#pragma optimize(5)
namespace qcu {
template <typename T>
__global__ void make_clover_all(
    void *device_U, void *device_clover, void *device_params, T kappa,
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
    void *device_u_b_z_f_t_recv_vec, void *device_u_f_z_f_t_recv_vec) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int parity = idx;
  int *params = static_cast<int *>(device_params);
  int lat_x = params[_LAT_X_];
  int lat_y = params[_LAT_Y_];
  int lat_z = params[_LAT_Z_];
  int lat_t = params[_LAT_T_];
  int lat_xyzt = params[_LAT_XYZT_];
  int move0;
  int move1;
  move0 = lat_y * lat_z * lat_t;
  int x = parity / move0;
  parity -= x * move0;
  move0 = lat_z * lat_t;
  int y = parity / move0;
  parity -= y * move0;
  int z = parity / lat_t;
  int t = parity - z * lat_t;
  int eo = (x + y + z) & 0x01; //(x+y+z)%2
  parity = params[_PARITY_];
  int move_wards[_WARDS_];
  move_backward(move_wards[_B_X_], x, lat_x);
  move_backward(move_wards[_B_Y_], y, lat_y);
  move_backward(move_wards[_B_Z_], z, lat_z);
  move_backward_t(move_wards[_B_T_], t, lat_t, eo, parity);
  move_forward(move_wards[_F_X_], x, lat_x);
  move_forward(move_wards[_F_Y_], y, lat_y);
  move_forward(move_wards[_F_Z_], z, lat_z);
  move_forward_t(move_wards[_F_T_], t, lat_t, eo, parity);
  //  LatticeComplex<T> I(0.0, 1.0);
  LatticeComplex<T> zero(0.0, 0.0);
  LatticeComplex<T> tmp0(0.0, 0.0);
  LatticeComplex<T> *origin_U =
      ((static_cast<LatticeComplex<T> *>(device_U)) + idx);
  LatticeComplex<T> *origin_clover =
      ((static_cast<LatticeComplex<T> *>(device_clover)) + idx);
  LatticeComplex<T> *tmp_U;
  LatticeComplex<T> tmp1[_LAT_CC_];
  LatticeComplex<T> tmp2[_LAT_CC_];
  LatticeComplex<T> tmp3[_LAT_CC_];
  LatticeComplex<T> tmp4[_LAT_CC_]; // just for test
  LatticeComplex<T> U[_LAT_CC_];
  LatticeComplex<T> clover[_LAT_SCSC_];
  // just all
  int if_b_x = (move_wards[_B_X_] == lat_x - 1);
  int if_b_y = (move_wards[_B_Y_] == lat_y - 1);
  int if_b_z = (move_wards[_B_Z_] == lat_z - 1);
  int if_b_t = (move_wards[_B_T_] == lat_t - 1);
  int if_f_x = (move_wards[_F_X_] == 1 - lat_x);
  int if_f_y = (move_wards[_F_Y_] == 1 - lat_y);
  int if_f_z = (move_wards[_F_Z_] == 1 - lat_z);
  int if_f_t = (move_wards[_F_T_] == 1 - lat_t);
  if (1) {
    // if_b_x = 0;
    // if_b_y = 0;
    // if_b_z = 0;
    // if_b_t = 0;
    // if_f_x = 0;
    // if_f_y = 0;
    // if_f_z = 0;
    // if_f_t = 0;
  }
  int if_b_x_b_y =
      (move_wards[_B_X_] == lat_x - 1) * (move_wards[_B_Y_] == lat_y - 1);
  int if_f_x_b_y =
      (move_wards[_F_X_] == 1 - lat_x) * (move_wards[_B_Y_] == lat_y - 1);
  int if_b_x_f_y =
      (move_wards[_B_X_] == lat_x - 1) * (move_wards[_F_Y_] == 1 - lat_y);
  // // int if_f_x_f_y=
  //(move_wards[_F_X_]==1-lat_x)*(move_wards[_F_Y_]==1-lat_y);
  int if_b_x_b_z =
      (move_wards[_B_X_] == lat_x - 1) * (move_wards[_B_Z_] == lat_z - 1);
  int if_f_x_b_z =
      (move_wards[_F_X_] == 1 - lat_x) * (move_wards[_B_Z_] == lat_z - 1);
  int if_b_x_f_z =
      (move_wards[_B_X_] == lat_x - 1) * (move_wards[_F_Z_] == 1 - lat_z);
  // // int if_f_x_f_z=
  //(move_wards[_F_X_]==1-lat_x)*(move_wards[_F_Z_]==1-lat_z);
  int if_b_x_b_t =
      (move_wards[_B_X_] == lat_x - 1) * (move_wards[_B_T_] == lat_t - 1);
  int if_f_x_b_t =
      (move_wards[_F_X_] == 1 - lat_x) * (move_wards[_B_T_] == lat_t - 1);
  int if_b_x_f_t =
      (move_wards[_B_X_] == lat_x - 1) * (move_wards[_F_T_] == 1 - lat_t);
  // // int if_f_x_f_t=
  //(move_wards[_F_X_]==1-lat_x)*(move_wards[_F_T_]==1-lat_t);
  int if_b_y_b_z =
      (move_wards[_B_Y_] == lat_y - 1) * (move_wards[_B_Z_] == lat_z - 1);
  int if_f_y_b_z =
      (move_wards[_F_Y_] == 1 - lat_y) * (move_wards[_B_Z_] == lat_z - 1);
  int if_b_y_f_z =
      (move_wards[_B_Y_] == lat_y - 1) * (move_wards[_F_Z_] == 1 - lat_z);
  // // int if_f_y_f_z=
  //(move_wards[_F_Y_]==1-lat_y)*(move_wards[_F_Z_]==1-lat_z);
  int if_b_y_b_t =
      (move_wards[_B_Y_] == lat_y - 1) * (move_wards[_B_T_] == lat_t - 1);
  int if_f_y_b_t =
      (move_wards[_F_Y_] == 1 - lat_y) * (move_wards[_B_T_] == lat_t - 1);
  int if_b_y_f_t =
      (move_wards[_B_Y_] == lat_y - 1) * (move_wards[_F_T_] == 1 - lat_t);
  // // int if_f_y_f_t=
  //(move_wards[_F_Y_]==1-lat_y)*(move_wards[_F_T_]==1-lat_t);
  int if_b_z_b_t =
      (move_wards[_B_Z_] == lat_z - 1) * (move_wards[_B_T_] == lat_t - 1);
  int if_f_z_b_t =
      (move_wards[_F_Z_] == 1 - lat_z) * (move_wards[_B_T_] == lat_t - 1);
  int if_b_z_f_t =
      (move_wards[_B_Z_] == lat_z - 1) * (move_wards[_F_T_] == 1 - lat_t);
  // // int if_f_z_f_t=
  //(move_wards[_F_Z_]==1-lat_z)*(move_wards[_F_T_]==1-lat_t);
  if (1) {
    if_b_x = 0;
    if_b_y = 0;
    if_b_z = 0;
    if_b_t = 0;
    // if_f_x = 0;
    if_f_y = 0;
    if_f_z = 0;
    if_f_t = 0;
    // if_b_x_b_y = 0;
    // if_b_x_b_z = 0;
    // if_b_x_b_t = 0;
    // if_b_y_b_z = 0;
    // if_b_y_b_t = 0;
    // if_b_z_b_t = 0;
    // if_f_x_b_y = 0;
    // if_f_x_b_z = 0;
    // if_f_x_b_t = 0;
    // if_f_y_b_z = 0;
    // if_f_y_b_t = 0;
    // if_f_z_b_t = 0;
    // if_b_x_f_y = 0;
    // if_b_x_f_z = 0;
    // if_b_x_f_t = 0;
    // if_b_y_f_z = 0;
    // if_b_y_f_t = 0;
    // if_b_z_f_t = 0;
  }
  if (idx == 0) { // just for test
    tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_x_recv_vec) +
             ((((0) * lat_y + y) * lat_z + z) * lat_t + t));
    for (int i = 0; i < _LAT_PCCD_; i++) {
      int p = i / _LAT_CCD_;
      int tmp = i - p * _LAT_CCD_;
      int c0 = tmp / _LAT_CD_;
      tmp -= c0 * _LAT_CD_;
      int c1 = tmp / _LAT_D_;
      tmp -= c1 * _LAT_D_;
      int d = tmp;
      printf("check out "
             "f_x_recv_vec:idx,i,p,c0,c1,d,x,y,z,t:%d,%d,%d,%d,%d,%d,%d,%d,%d,%"
             "d,u_b_x_"
             "device_u_f_x_recv_vec[i * lat_xyzt / lat_x]._data.x:%e\n",
             idx, i, p, c0, c1, d, x, y, z, t,
             tmp_U[i * lat_xyzt / lat_x]._data.x);
      printf("check out "
             "f_x_recv_vec:idx,i,p,c0,c1,d,x,y,z,t:%d,%d,%d,%d,%d,%d,%d,%d,%d,%"
             "d,u_b_x_"
             "device_u_f_x_recv_vec[i * lat_xyzt / lat_x]._data.y:%e\n",
             idx, i, p, c0, c1, d, x, y, z, t,
             tmp_U[i * lat_xyzt / lat_x]._data.y);
    }
  }
  // sigmaF
  {
    get_vals(clover, zero, _LAT_SCSC_);
    // get_vals(origin_clover,zero,_LAT_SCSC_);//BUG!!!!!!
    get_vals(tmp1, zero, _LAT_CC_);
    get_vals(tmp2, zero, _LAT_CC_);
  }
  // XY
  get_vals(U, zero, _LAT_CC_);
  {
    //// x,y,z,t;x
    tmp_U = (origin_U + (_X_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x+1,y,z,t;y
    if (if_f_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_x_recv_vec) +
               ((((_Y_ * 1 + 0) * lat_y + y) * lat_z + z) * lat_t + t));
      get_u_comm((1 - parity), tmp2, tmp_U, lat_xyzt / lat_x);
      // get_u_comm(1, tmp2, tmp_U, lat_xyzt / lat_x);
      printf("if_f_x:p,x,y,z,t:%d,%d,%d,%d,%d,tmp2[0]._data.x:%e\n", parity, x,
             y, z, t, tmp2[0]._data.x);
      printf("if_f_x:p,x,y,z,t:%d,%d,%d,%d,%d,tmp2[0]._data.y:%e\n", parity, x,
             y, z, t, tmp2[0]._data.y);
      printf("if_f_x:p,x,y,z,t:%d,%d,%d,%d,%d,tmp2[5]._data.x:%e\n", parity, x,
             y, z, t, tmp2[5]._data.x);
      printf("if_f_x:p,x,y,z,t:%d,%d,%d,%d,%d,tmp2[5]._data.y:%e\n", parity, x,
             y, z, t, tmp2[5]._data.y);
      move0 = move_wards[_F_X_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
               (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp4, tmp_U, lat_xyzt);
      printf("if_f_x:p,x,y,z,t:%d,%d,%d,%d,%d,tmp4[0]._data.x:%e\n", parity, x,
             y, z, t, tmp4[0]._data.x);
      printf("if_f_x:p,x,y,z,t:%d,%d,%d,%d,%d,tmp4[0]._data.y:%e\n", parity, x,
             y, z, t, tmp4[0]._data.y);
      printf("if_f_x:p,x,y,z,t:%d,%d,%d,%d,%d,tmp4[5]._data.x:%e\n", parity, x,
             y, z, t, tmp4[5]._data.x);
      printf("if_f_x:p,x,y,z,t:%d,%d,%d,%d,%d,tmp4[5]._data.y:%e\n", parity, x,
             y, z, t, tmp4[5]._data.y);
    } else {
      move0 = move_wards[_F_X_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
               (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z,t;x;dag
    if (if_f_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_y_recv_vec) +
               ((((_X_ * lat_x + x) * 1 + 0) * lat_z + z) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_y);
      printf("if_f_y:p,x,y,z,t:%d,%d,%d,%d,%d,tmp1[0]._data.x:%e\n", parity, x,
             y, z, t, tmp1[0]._data.x);
      printf("if_f_y:p,x,y,z,t:%d,%d,%d,%d,%d,tmp1[0]._data.y:%e\n", parity, x,
             y, z, t, tmp1[0]._data.y);
      printf("if_f_y:p,x,y,z,t:%d,%d,%d,%d,%d,tmp1[5]._data.x:%e\n", parity, x,
             y, z, t, tmp1[5]._data.x);
      printf("if_f_y:p,x,y,z,t:%d,%d,%d,%d,%d,tmp1[5]._data.y:%e\n", parity, x,
             y, z, t, tmp1[5]._data.y);
      move0 = move_wards[_F_Y_];
      tmp_U = (origin_U + move0 * lat_z * lat_t +
               (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp4, tmp_U, lat_xyzt);
      printf("if_f_y:p,x,y,z,t:%d,%d,%d,%d,%d,tmp4[0]._data.x:%e\n", parity, x,
             y, z, t, tmp4[0]._data.x);
      printf("if_f_y:p,x,y,z,t:%d,%d,%d,%d,%d,tmp4[0]._data.y:%e\n", parity, x,
             y, z, t, tmp4[0]._data.y);
      printf("if_f_y:p,x,y,z,t:%d,%d,%d,%d,%d,tmp4[5]._data.x:%e\n", parity, x,
             y, z, t, tmp4[5]._data.x);
      printf("if_f_y:p,x,y,z,t:%d,%d,%d,%d,%d,tmp4[5]._data.y:%e\n", parity, x,
             y, z, t, tmp4[5]._data.y);
    } else {
      move0 = move_wards[_F_Y_];
      tmp_U = (origin_U + move0 * lat_z * lat_t +
               (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;y;dag
    tmp_U = (origin_U + (_Y_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;y
    tmp_U = (origin_U + (_Y_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x-1,y+1,z,t;x;dag
    if (if_b_x_f_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_f_y_recv_vec) +
               ((((_X_ * 1 + 0) * 1 + 0) * lat_z + z) * lat_t + t));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_x / lat_y);
    } else if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_recv_vec) +
               ((((_X_ * 1 + 0) * lat_y + y + 1) * lat_z + z) * lat_t + t));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_x);
    } else if (if_f_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_y_recv_vec) +
               ((((_X_ * lat_x + x - 1) * 1 + 0) * lat_z + z) * lat_t + t));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_y);
    } else {
      move0 = move_wards[_B_X_];
      move1 = move_wards[_F_Y_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
               move1 * lat_z * lat_t + (_X_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;y;dag
    if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_recv_vec) +
               ((((_Y_ * 1 + 0) * lat_y + y) * lat_z + z) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_x);
      printf("if_b_x:p,x,y,z,t:%d,%d,%d,%d,%d,tmp1[0]._data.x:%e\n", parity, x,
             y, z, t, tmp1[0]._data.x);
      printf("if_b_x:p,x,y,z,t:%d,%d,%d,%d,%d,tmp1[0]._data.y:%e\n", parity, x,
             y, z, t, tmp1[0]._data.y);
      printf("if_b_x:p,x,y,z,t:%d,%d,%d,%d,%d,tmp1[5]._data.x:%e\n", parity, x,
             y, z, t, tmp1[5]._data.x);
      printf("if_b_x:p,x,y,z,t:%d,%d,%d,%d,%d,tmp1[5]._data.y:%e\n", parity, x,
             y, z, t, tmp1[5]._data.y);
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
               (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp4, tmp_U, lat_xyzt);
      printf("if_b_x:p,x,y,z,t:%d,%d,%d,%d,%d,tmp4[0]._data.x:%e\n", parity, x,
             y, z, t, tmp4[0]._data.x);
      printf("if_b_x:p,x,y,z,t:%d,%d,%d,%d,%d,tmp4[0]._data.y:%e\n", parity, x,
             y, z, t, tmp4[0]._data.y);
      printf("if_b_x:p,x,y,z,t:%d,%d,%d,%d,%d,tmp4[5]._data.x:%e\n", parity, x,
             y, z, t, tmp4[5]._data.x);
      printf("if_b_x:p,x,y,z,t:%d,%d,%d,%d,%d,tmp4[5]._data.y:%e\n", parity, x,
             y, z, t, tmp4[5]._data.y);
    } else {
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
               (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_recv_vec) +
               ((((_X_ * 1 + 0) * lat_y + y) * lat_z + z) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_x);
    } else {
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
               (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x-1,y,z,t;x;dag
    if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_recv_vec) +
               ((((_X_ * 1 + 0) * lat_y + y) * lat_z + z) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_x);
    } else {
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
               (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    //// x-1,y-1,z,t;y;dag
    if (if_b_x_b_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_b_y_recv_vec) +
               ((((_Y_ * 1 + 0) * 1 + 0) * lat_z + z) * lat_t + t));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_x / lat_y);
    } else if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_recv_vec) +
               ((((_Y_ * 1 + 0) * lat_y + y - 1) * lat_z + z) * lat_t + t));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_x);
    } else if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_recv_vec) +
               ((((_Y_ * lat_x + x - 1) * 1 + 0) * lat_z + z) * lat_t + t));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_y);
    } else {
      move0 = move_wards[_B_X_];
      move1 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
               move1 * lat_z * lat_t + (_Y_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y-1,z,t;x
    if (if_b_x_b_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_b_y_recv_vec) +
               ((((_X_ * 1 + 0) * 1 + 0) * lat_z + z) * lat_t + t));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_x / lat_y);
    } else if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_recv_vec) +
               ((((_X_ * 1 + 0) * lat_y + y - 1) * lat_z + z) * lat_t + t));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_x);
    } else if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_recv_vec) +
               ((((_X_ * lat_x + x - 1) * 1 + 0) * lat_z + z) * lat_t + t));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_y);
    } else {
      move0 = move_wards[_B_X_];
      move1 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
               move1 * lat_z * lat_t + (_X_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_recv_vec) +
               ((((_Y_ * lat_x + x) * 1 + 0) * lat_z + z) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_y);
    } else {
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_z * lat_t +
               (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y-1,z,t;y;dag
    if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_recv_vec) +
               ((((_Y_ * lat_x + x) * 1 + 0) * lat_z + z) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_y);
    } else {
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_z * lat_t +
               (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    //// x,y-1,z,t;x
    if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_recv_vec) +
               ((((_X_ * lat_x + x) * 1 + 0) * lat_z + z) * lat_t + t));
      get_u_comm(1 - parity, tmp2, tmp_U, lat_xyzt / lat_y);
    } else {
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_z * lat_t +
               (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y-1,z,t;y
    if (if_f_x_b_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_x_b_y_recv_vec) +
               ((((_Y_ * 1 + 0) * 1 + 0) * lat_z + z) * lat_t + t));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_x / lat_y);
    } else if (if_f_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_x_recv_vec) +
               ((((_Y_ * 1 + 0) * lat_y + y - 1) * lat_z + z) * lat_t + t));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_x);
    } else if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_recv_vec) +
               ((((_Y_ * lat_x + x + 1) * 1 + 0) * lat_z + z) * lat_t + t));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_y);
    } else {
      move0 = move_wards[_F_X_];
      move1 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
               move1 * lat_z * lat_t + (_Y_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;x;dag
    tmp_U = (origin_U + (_X_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        clover[c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj())
                .multi_minus_i();
        clover[39 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_i();
        clover[78 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj())
                .multi_minus_i();
        clover[117 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_i();
      }
    }
  }
  // XZ
  get_vals(U, zero, _LAT_CC_);
  {
    //// x,y,z,t;x
    tmp_U = (origin_U + (_X_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x+1,y,z,t;z
    if (if_f_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_x_recv_vec) +
               ((((_Z_ * 1 + 0) * lat_y + y) * lat_z + z) * lat_t + t));
      get_u_comm(1 - parity, tmp2, tmp_U, lat_xyzt / lat_x);
    } else {
      move0 = move_wards[_F_X_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
               (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t;x;dag
    if (if_f_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_z_recv_vec) +
               ((((_X_ * lat_x + x) * lat_y + y) * 1 + 0) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_z);
    } else {
      move0 = move_wards[_F_Z_];
      tmp_U = (origin_U + move0 * lat_t +
               (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;z;dag
    tmp_U = (origin_U + (_Z_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;z
    tmp_U = (origin_U + (_Z_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x-1,y,z+1,t;x;dag
    if (if_b_x_f_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_f_z_recv_vec) +
               ((((_X_ * 1 + 0) * lat_y + y) * 1 + 0) * lat_t + t));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_x / lat_z);
    } else if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_recv_vec) +
               ((((_X_ * 1 + 0) * lat_y + y) * lat_z + z + 1) * lat_t + t));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_x);
    } else if (if_f_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_z_recv_vec) +
               ((((_X_ * lat_x + x - 1) * lat_y + y) * 1 + 0) * lat_t + t));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_z);
    } else {
      move0 = move_wards[_B_X_];
      move1 = move_wards[_F_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t + move1 * lat_t +
               (_X_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;z;dag
    if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_recv_vec) +
               ((((_Z_ * 1 + 0) * lat_y + y) * lat_z + z) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_x);
    } else {
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
               (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_recv_vec) +
               ((((_X_ * 1 + 0) * lat_y + y) * lat_z + z) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_x);
    } else {
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
               (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x-1,y,z,t;x;dag
    if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_recv_vec) +
               ((((_X_ * 1 + 0) * lat_y + y) * lat_z + z) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_x);
    } else {
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
               (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    //// x-1,y,z-1,t;z;dag
    if (if_b_x_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_b_z_recv_vec) +
               ((((_Z_ * 1 + 0) * lat_y + y) * 1 + 0) * lat_t + t));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_x / lat_z);
    } else if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_recv_vec) +
               ((((_Z_ * 1 + 0) * lat_y + y) * lat_z + z - 1) * lat_t + t));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_x);
    } else if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_z_recv_vec) +
               ((((_Z_ * lat_x + x - 1) * lat_y + y) * 1 + 0) * lat_t + t));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_z);
    } else {
      move0 = move_wards[_B_X_];
      move1 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t + move1 * lat_t +
               (_Z_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z-1,t;x
    if (if_b_x_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_b_z_recv_vec) +
               ((((_X_ * 1 + 0) * lat_y + y) * 1 + 0) * lat_t + t));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_x / lat_z);
    } else if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_recv_vec) +
               ((((_X_ * 1 + 0) * lat_y + y) * lat_z + z - 1) * lat_t + t));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_x);
    } else if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_z_recv_vec) +
               ((((_X_ * lat_x + x - 1) * lat_y + y) * 1 + 0) * lat_t + t));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_z);
    } else {
      move0 = move_wards[_B_X_];
      move1 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t + move1 * lat_t +
               (_X_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_z_recv_vec) +
               ((((_Z_ * lat_x + x) * lat_y + y) * 1 + 0) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_z);
    } else {
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_t +
               (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z-1,t;z;dag
    if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_z_recv_vec) +
               ((((_Z_ * lat_x + x) * lat_y + y) * 1 + 0) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_z);
    } else {
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_t +
               (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    //// x,y,z-1,t;x
    if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_z_recv_vec) +
               ((((_X_ * lat_x + x) * lat_y + y) * 1 + 0) * lat_t + t));
      get_u_comm(1 - parity, tmp2, tmp_U, lat_xyzt / lat_z);
    } else {
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_t +
               (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y,z-1,t;z
    if (if_f_x_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_x_b_z_recv_vec) +
               ((((_Z_ * 1 + 0) * lat_y + y) * 1 + 0) * lat_t + t));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_x / lat_z);
    } else if (if_f_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_x_recv_vec) +
               ((((_Z_ * 1 + 0) * lat_y + y) * lat_z + z - 1) * lat_t + t));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_x);
    } else if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_z_recv_vec) +
               ((((_Z_ * lat_x + x + 1) * lat_y + y) * 1 + 0) * lat_t + t));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_z);
    } else {
      move0 = move_wards[_F_X_];
      move1 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t + move1 * lat_t +
               (_Z_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;x;dag
    tmp_U = (origin_U + (_X_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        clover[_LAT_C_ + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_minus();
        clover[36 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj());
        clover[81 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_minus();
        clover[114 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj());
      }
    }
  }
  // XT
  get_vals(U, zero, _LAT_CC_);
  {
    //// x,y,z,t;x
    tmp_U = (origin_U + (_X_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x+1,y,z,t;t
    if (if_f_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_x_recv_vec) +
               ((((_T_ * 1 + 0) * lat_y + y) * lat_z + z) * lat_t + t));
      get_u_comm(1 - parity, tmp2, tmp_U, lat_xyzt / lat_x);
    } else {
      move0 = move_wards[_F_X_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
               (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;x;dag
    if (if_f_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_t_recv_vec) +
               ((((_X_ * lat_x + x) * lat_y + y) * lat_z + z) * 1 + 0));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_F_T_];
      tmp_U = (origin_U + move0 + (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;t;dag
    tmp_U = (origin_U + (_T_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;t
    tmp_U = (origin_U + (_T_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x-1,y,z,t+1;x;dag
    if (if_b_x_f_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_f_t_recv_vec) +
               ((((_X_ * 1 + 0) * lat_y + y) * lat_z + z) * 1 + 0));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_x / lat_t);
    } else if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_recv_vec) +
               ((((_X_ * 1 + 0) * lat_y + y) * lat_z + z) * lat_t + t +
                move_wards[_F_T_]));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_x);
    } else if (if_f_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_t_recv_vec) +
               ((((_X_ * lat_x + x - 1) * lat_y + y) * lat_z + z) * 1 + 0));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_B_X_];
      move1 = move_wards[_F_T_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t + move1 +
               (_X_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;t;dag
    if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_recv_vec) +
               ((((_T_ * 1 + 0) * lat_y + y) * lat_z + z) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_x);
    } else {
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
               (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_recv_vec) +
               ((((_X_ * 1 + 0) * lat_y + y) * lat_z + z) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_x);
    } else {
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
               (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x-1,y,z,t;x;dag
    if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_recv_vec) +
               ((((_X_ * 1 + 0) * lat_y + y) * lat_z + z) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_x);
    } else {
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
               (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    //// x-1,y,z,t-1;t;dag
    if (if_b_x_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_b_t_recv_vec) +
               ((((_T_ * 1 + 0) * lat_y + y) * lat_z + z) * 1 + 0));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_x / lat_t);
    } else if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_recv_vec) +
               ((((_T_ * 1 + 0) * lat_y + y) * lat_z + z) * lat_t + t +
                move_wards[_B_T_]));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_x);
    } else if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_t_recv_vec) +
               ((((_T_ * lat_x + x - 1) * lat_y + y) * lat_z + z) * 1 + 0));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_B_X_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t + move1 +
               (_T_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t-1;x
    if (if_b_x_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_b_t_recv_vec) +
               ((((_X_ * 1 + 0) * lat_y + y) * lat_z + z) * 1 + 0));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_x / lat_t);
    } else if (if_b_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_x_recv_vec) +
               ((((_X_ * 1 + 0) * lat_y + y) * lat_z + z) * lat_t + t +
                move_wards[_B_T_]));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_x);
    } else if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_t_recv_vec) +
               ((((_X_ * lat_x + x - 1) * lat_y + y) * lat_z + z) * 1 + 0));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_B_X_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t + move1 +
               (_X_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_t_recv_vec) +
               ((((_T_ * lat_x + x) * lat_y + y) * lat_z + z) * 1 + 0));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 + (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t-1;t;dag
    if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_t_recv_vec) +
               ((((_T_ * lat_x + x) * lat_y + y) * lat_z + z) * 1 + 0));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 + (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    //// x,y,z,t-1;x
    if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_t_recv_vec) +
               ((((_X_ * lat_x + x) * lat_y + y) * lat_z + z) * 1 + 0));
      get_u_comm(1 - parity, tmp2, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 + (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y,z,t-1;t
    if (if_f_x_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_x_b_t_recv_vec) +
               ((((_T_ * 1 + 0) * lat_y + y) * lat_z + z) * 1 + 0));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_x / lat_t);
    } else if (if_f_x) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_x_recv_vec) +
               ((((_T_ * 1 + 0) * lat_y + y) * lat_z + z) * lat_t + t +
                move_wards[_B_T_]));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_x);
    } else if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_t_recv_vec) +
               ((((_T_ * lat_x + x + 1) * lat_y + y) * lat_z + z) * 1 + 0));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_F_X_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t + move1 +
               (_T_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;x;dag
    tmp_U = (origin_U + (_X_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        clover[_LAT_C_ + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_i();
        clover[36 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_i();
        clover[81 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj())
                .multi_minus_i();
        clover[114 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj())
                .multi_minus_i();
      }
    }
  }
  // YZ
  get_vals(U, zero, _LAT_CC_);
  {
    //// x,y,z,t;y
    tmp_U = (origin_U + (_Y_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x,y+1,z,t;z
    if (if_f_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_y_recv_vec) +
               ((((_Z_ * lat_x + x) * 1 + 0) * lat_z + z) * lat_t + t));
      get_u_comm(1 - parity, tmp2, tmp_U, lat_xyzt / lat_y);
    } else {
      move0 = move_wards[_F_Y_];
      tmp_U = (origin_U + move0 * lat_z * lat_t +
               (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t;y;dag
    if (if_f_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_z_recv_vec) +
               ((((_Y_ * lat_x + x) * lat_y + y) * 1 + 0) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_z);
    } else {
      move0 = move_wards[_F_Z_];
      tmp_U = (origin_U + move0 * lat_t +
               (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;z;dag
    tmp_U = (origin_U + (_Z_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;z
    tmp_U = (origin_U + (_Z_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x,y-1,z+1,t;y;dag
    if (if_b_y_f_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_f_z_recv_vec) +
               ((((_Y_ * lat_x + x) * 1 + 0) * 1 + 0) * lat_t + t));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_y / lat_z);
    } else if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_recv_vec) +
               ((((_Y_ * lat_x + x) * 1 + 0) * lat_z + z + 1) * lat_t + t));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_y);
    } else if (if_f_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_z_recv_vec) +
               ((((_Y_ * lat_x + x) * lat_y + y - 1) * 1 + 0) * lat_t + t));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_z);
    } else {
      move0 = move_wards[_B_Y_];
      move1 = move_wards[_F_Z_];
      tmp_U = (origin_U + move0 * lat_z * lat_t + move1 * lat_t +
               (_Y_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t;z;dag
    if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_recv_vec) +
               ((((_Z_ * lat_x + x) * 1 + 0) * lat_z + z) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_y);
    } else {
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_z * lat_t +
               (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_recv_vec) +
               ((((_Y_ * lat_x + x) * 1 + 0) * lat_z + z) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_y);
    } else {
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_z * lat_t +
               (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y-1,z,t;y;dag
    if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_recv_vec) +
               ((((_Y_ * lat_x + x) * 1 + 0) * lat_z + z) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_y);
    } else {
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_z * lat_t +
               (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    //// x,y-1,z-1,t;z;dag
    if (if_b_y_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_b_z_recv_vec) +
               ((((_Z_ * lat_x + x) * 1 + 0) * 1 + 0) * lat_t + t));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_y / lat_z);
    } else if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_recv_vec) +
               ((((_Z_ * lat_x + x) * 1 + 0) * lat_z + z - 1) * lat_t + t));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_y);
    } else if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_z_recv_vec) +
               ((((_Z_ * lat_x + x) * lat_y + y - 1) * 1 + 0) * lat_t + t));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_z);
    } else {
      move0 = move_wards[_B_Y_];
      move1 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_z * lat_t + move1 * lat_t +
               (_Z_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z-1,t;y
    if (if_b_y_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_b_z_recv_vec) +
               ((((_Y_ * lat_x + x) * 1 + 0) * 1 + 0) * lat_t + t));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_y / lat_z);
    } else if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_recv_vec) +
               ((((_Y_ * lat_x + x) * 1 + 0) * lat_z + z - 1) * lat_t + t));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_y);
    } else if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_z_recv_vec) +
               ((((_Y_ * lat_x + x) * lat_y + y - 1) * 1 + 0) * lat_t + t));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_z);
    } else {
      move0 = move_wards[_B_Y_];
      move1 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_z * lat_t + move1 * lat_t +
               (_Y_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_z_recv_vec) +
               ((((_Z_ * lat_x + x) * lat_y + y) * 1 + 0) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_z);
    } else {
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_t +
               (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z-1,t;z;dag
    if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_z_recv_vec) +
               ((((_Z_ * lat_x + x) * lat_y + y) * 1 + 0) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_z);
    } else {
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_t +
               (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    //// x,y,z-1,t;y
    if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_z_recv_vec) +
               ((((_Y_ * lat_x + x) * lat_y + y) * 1 + 0) * lat_t + t));
      get_u_comm(1 - parity, tmp2, tmp_U, lat_xyzt / lat_z);
    } else {
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_t +
               (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z-1,t;z
    if (if_f_y_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_y_b_z_recv_vec) +
               ((((_Z_ * lat_x + x) * 1 + 0) * 1 + 0) * lat_t + t));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_y / lat_z);
    } else if (if_f_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_y_recv_vec) +
               ((((_Z_ * lat_x + x) * 1 + 0) * lat_z + z - 1) * lat_t + t));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_y);
    } else if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_z_recv_vec) +
               ((((_Z_ * lat_x + x) * lat_y + y + 1) * 1 + 0) * lat_t + t));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_z);
    } else {
      move0 = move_wards[_F_Y_];
      move1 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_z * lat_t + move1 * lat_t +
               (_Z_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;y;dag
    tmp_U = (origin_U + (_Y_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        clover[_LAT_C_ + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj())
                .multi_minus_i();
        clover[36 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj())
                .multi_minus_i();
        clover[81 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj())
                .multi_minus_i();
        clover[114 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj())
                .multi_minus_i();
      }
    }
  }
  // YT
  get_vals(U, zero, _LAT_CC_);
  {
    //// x,y,z,t;y
    tmp_U = (origin_U + (_Y_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x,y+1,z,t;t
    if (if_f_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_y_recv_vec) +
               ((((_T_ * lat_x + x) * 1 + 0) * lat_z + z) * lat_t + t));
      get_u_comm(1 - parity, tmp2, tmp_U, lat_xyzt / lat_y);
    } else {
      move0 = move_wards[_F_Y_];
      tmp_U = (origin_U + move0 * lat_z * lat_t +
               (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;y;dag
    if (if_f_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_t_recv_vec) +
               ((((_Y_ * lat_x + x) * lat_y + y) * lat_z + z) * 1 + 0));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_F_T_];
      tmp_U = (origin_U + move0 + (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;t;dag
    tmp_U = (origin_U + (_T_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;t
    tmp_U = (origin_U + (_T_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x,y-1,z,t+1;y;dag
    if (if_b_y_f_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_f_t_recv_vec) +
               ((((_Y_ * lat_x + x) * 1 + 0) * lat_z + z) * 1 + 0));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_y / lat_t);
    } else if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_recv_vec) +
               ((((_Y_ * lat_x + x) * 1 + 0) * lat_z + z) * lat_t + t +
                move_wards[_F_T_]));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_y);
    } else if (if_f_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_t_recv_vec) +
               ((((_Y_ * lat_x + x) * lat_y + y - 1) * lat_z + z) * 1 + 0));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_B_Y_];
      move1 = move_wards[_F_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_t + move1 +
               (_Y_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t;t;dag
    if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_recv_vec) +
               ((((_T_ * lat_x + x) * 1 + 0) * lat_z + z) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_y);
    } else {
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_z * lat_t +
               (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_recv_vec) +
               ((((_Y_ * lat_x + x) * 1 + 0) * lat_z + z) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_y);
    } else {
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_z * lat_t +
               (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y-1,z,t;y;dag
    if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_recv_vec) +
               ((((_Y_ * lat_x + x) * 1 + 0) * lat_z + z) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_y);
    } else {
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_z * lat_t +
               (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    //// x,y-1,z,t-1;t;dag
    if (if_b_y_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_b_t_recv_vec) +
               ((((_T_ * lat_x + x) * 1 + 0) * lat_z + z) * 1 + 0));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_y / lat_t);
    } else if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_recv_vec) +
               ((((_T_ * lat_x + x) * 1 + 0) * lat_z + z) * lat_t + t +
                move_wards[_B_T_]));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_y);
    } else if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_t_recv_vec) +
               ((((_T_ * lat_x + x) * lat_y + y - 1) * lat_z + z) * 1 + 0));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_B_Y_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_t + move1 +
               (_T_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t-1;y
    if (if_b_y_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_b_t_recv_vec) +
               ((((_Y_ * lat_x + x) * 1 + 0) * lat_z + z) * 1 + 0));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_y / lat_t);
    } else if (if_b_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_y_recv_vec) +
               ((((_Y_ * lat_x + x) * 1 + 0) * lat_z + z) * lat_t + t +
                move_wards[_B_T_]));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_y);
    } else if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_t_recv_vec) +
               ((((_Y_ * lat_x + x) * lat_y + y - 1) * lat_z + z) * 1 + 0));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_B_Y_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_t + move1 +
               (_Y_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_t_recv_vec) +
               ((((_T_ * lat_x + x) * lat_y + y) * lat_z + z) * 1 + 0));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 + (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t-1;t;dag
    if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_t_recv_vec) +
               ((((_T_ * lat_x + x) * lat_y + y) * lat_z + z) * 1 + 0));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 + (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    //// x,y,z,t-1;y
    if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_t_recv_vec) +
               ((((_Y_ * lat_x + x) * lat_y + y) * lat_z + z) * 1 + 0));
      get_u_comm(1 - parity, tmp2, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 + (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z,t-1;t
    if (if_f_y_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_y_b_t_recv_vec) +
               ((((_T_ * lat_x + x) * 1 + 0) * lat_z + z) * 1 + 0));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_y / lat_t);
    } else if (if_f_y) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_y_recv_vec) +
               ((((_T_ * lat_x + x) * 1 + 0) * lat_z + z) * lat_t + t +
                move_wards[_B_T_]));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_y);
    } else if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_t_recv_vec) +
               ((((_T_ * lat_x + x) * lat_y + y + 1) * lat_z + z) * 1 + 0));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_F_Y_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_t + move1 +
               (_T_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;y;dag
    tmp_U = (origin_U + (_Y_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        clover[_LAT_C_ + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_minus();
        clover[36 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj());
        clover[81 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj());
        clover[114 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_minus();
      }
    }
  }
  // ZT
  get_vals(U, zero, _LAT_CC_);
  {
    //// x,y,z,t;z
    tmp_U = (origin_U + (_Z_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x,y,z+1,t;t
    if (if_f_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_z_recv_vec) +
               ((((_T_ * lat_x + x) * lat_y + y) * 1 + 0) * lat_t + t));
      get_u_comm(1 - parity, tmp2, tmp_U, lat_xyzt / lat_z);
    } else {
      move0 = move_wards[_F_Z_];
      tmp_U = (origin_U + move0 * lat_t +
               (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;z;dag
    if (if_f_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_t_recv_vec) +
               ((((_Z_ * lat_x + x) * lat_y + y) * lat_z + z) * 1 + 0));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_F_T_];
      tmp_U = (origin_U + move0 + (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;t;dag
    tmp_U = (origin_U + (_T_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;t
    tmp_U = (origin_U + (_T_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x,y,z-1,t+1;z;dag
    if (if_b_z_f_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_z_f_t_recv_vec) +
               ((((_Z_ * lat_x + x) * lat_y + y) * 1 + 0) * 1 + 0));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_z / lat_t);
    } else if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_z_recv_vec) +
               ((((_Z_ * lat_x + x) * lat_y + y) * 1 + 0) * lat_t + t +
                move_wards[_F_T_]));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_z);
    } else if (if_f_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_t_recv_vec) +
               ((((_Z_ * lat_x + x) * lat_y + y) * lat_z + z - 1) * 1 + 0));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_B_Z_];
      move1 = move_wards[_F_T_];
      tmp_U = (origin_U + move0 * lat_t + move1 +
               (_Z_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z-1,t;t;dag
    if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_z_recv_vec) +
               ((((_T_ * lat_x + x) * lat_y + y) * 1 + 0) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_z);
    } else {
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_t +
               (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_z_recv_vec) +
               ((((_Z_ * lat_x + x) * lat_y + y) * 1 + 0) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_z);
    } else {
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_t +
               (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z-1,t;z;dag
    if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_z_recv_vec) +
               ((((_Z_ * lat_x + x) * lat_y + y) * 1 + 0) * lat_t + t));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_z);
    } else {
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_t +
               (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    //// x,y,z-1,t-1;t;dag
    if (if_b_z_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_z_b_t_recv_vec) +
               ((((_T_ * lat_x + x) * lat_y + y) * 1 + 0) * 1 + 0));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_z / lat_t);
    } else if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_z_recv_vec) +
               ((((_T_ * lat_x + x) * lat_y + y) * 1 + 0) * lat_t + t +
                move_wards[_B_T_]));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_z);
    } else if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_t_recv_vec) +
               ((((_T_ * lat_x + x) * lat_y + y) * lat_z + z - 1) * 1 + 0));
      get_u_comm(parity, tmp2, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_B_Z_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_t + move1 +
               (_T_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z-1,t-1;z
    if (if_b_z_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_z_b_t_recv_vec) +
               ((((_Z_ * lat_x + x) * lat_y + y) * 1 + 0) * 1 + 0));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_z / lat_t);
    } else if (if_b_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_z_recv_vec) +
               ((((_Z_ * lat_x + x) * lat_y + y) * 1 + 0) * lat_t + t +
                move_wards[_B_T_]));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_z);
    } else if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_t_recv_vec) +
               ((((_Z_ * lat_x + x) * lat_y + y) * lat_z + z - 1) * 1 + 0));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_B_Z_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_t + move1 +
               (_Z_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_t_recv_vec) +
               ((((_T_ * lat_x + x) * lat_y + y) * lat_z + z) * 1 + 0));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 + (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t-1;t;dag
    if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_t_recv_vec) +
               ((((_T_ * lat_x + x) * lat_y + y) * lat_z + z) * 1 + 0));
      get_u_comm(1 - parity, tmp1, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 + (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    //// x,y,z,t-1;z
    if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_t_recv_vec) +
               ((((_Z_ * lat_x + x) * lat_y + y) * lat_z + z) * 1 + 0));
      get_u_comm(1 - parity, tmp2, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 + (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
      get_u(tmp2, tmp_U, lat_xyzt);
    }
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t-1;t
    if (if_f_z_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_z_b_t_recv_vec) +
               ((((_T_ * lat_x + x) * lat_y + y) * 1 + 0) * 1 + 0));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_z / lat_t);
    } else if (if_f_z) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_f_z_recv_vec) +
               ((((_T_ * lat_x + x) * lat_y + y) * 1 + 0) * lat_t + t +
                move_wards[_B_T_]));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_z);
    } else if (if_b_t) {
      tmp_U = (static_cast<LatticeComplex<T> *>(device_u_b_t_recv_vec) +
               ((((_T_ * lat_x + x) * lat_y + y) * lat_z + z + 1) * 1 + 0));
      get_u_comm(parity, tmp1, tmp_U, lat_xyzt / lat_t);
    } else {
      move0 = move_wards[_F_Z_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_t + move1 +
               (_T_ + parity * _LAT_CCD_) * lat_xyzt);
      get_u(tmp1, tmp_U, lat_xyzt);
    }
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;z;dag
    tmp_U = (origin_U + (_Z_ + parity * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      for (int c1 = 0; c1 < _LAT_C_; c1++) {
        clover[c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_i();
        clover[39 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj())
                .multi_minus_i();
        clover[78 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj())
                .multi_minus_i();
        clover[117 + c0 * _LAT_SC_ + c1] +=
            (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_i();
      }
    }
  }
  {
    // A=1+T(or A=1-kappa*T)
    LatticeComplex<T> one(1.0, 0);
    for (int i = 0; i < _LAT_SCSC_; i++) {
      clover[i] *= -kappa * 0.125; //-kappa*(1/8)
    }
    for (int i = 0; i < _LAT_SC_; i++) {
      clover[i * 13] += one;
    }
  }
  give_clr(origin_clover, clover, lat_xyzt);
}
//@@@CUDA_TEMPLATE_FOR_DEVICE@@@
template __global__ void make_clover_all<double>(
    void *device_U, void *device_clover, void *device_params, double kappa,
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
//@@@CUDA_TEMPLATE_FOR_DEVICE@@@
template __global__ void make_clover_all<float>(
    void *device_U, void *device_clover, void *device_params, float kappa,
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
} // namespace qcu