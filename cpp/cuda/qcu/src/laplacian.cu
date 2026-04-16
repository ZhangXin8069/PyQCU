#include "../include/qcu.h"
#pragma optimize(5)
namespace qcu {
#define __X__
#define __Y__
#define __Z__
template <typename T>
__global__ void laplacian_inside(void *device_U, void *device_src,
                                 void *device_dest, void *device_params) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int parity = idx;
  int *params = static_cast<int *>(device_params);
  int lat_x = params[_LAT_X_];
  int lat_y = params[_LAT_Y_];
  int lat_z = params[_LAT_Z_];
  int lat_t = params[_LAT_T_]; // in laplacian, lat_t = 1
  int lat_xyzt = params[_LAT_XYZT_];
  int move;
  move = lat_y * lat_z * lat_t;
  int x = parity / move;
  parity -= x * move;
  move = lat_z * lat_t;
  int y = parity / move;
  parity -= y * move;
  int z = parity / lat_t;
  // int t = parity - z * lat_t;
  LatticeComplex<T> zero(0.0, 0.0);
  LatticeComplex<T> *origin_U =
      ((static_cast<LatticeComplex<T> *>(device_U)) + idx);
  LatticeComplex<T> *origin_src =
      ((static_cast<LatticeComplex<T> *>(device_src)) + idx);
  LatticeComplex<T> *origin_dest =
      ((static_cast<LatticeComplex<T> *>(device_dest)) + idx);
  LatticeComplex<T> *tmp_U;
  LatticeComplex<T> *tmp_src;
  LatticeComplex<T> tmp0(0.0, 0.0);
  LatticeComplex<T> U[_LAT_CC_];
  LatticeComplex<T> src[_LAT_C_];
  LatticeComplex<T> dest[_LAT_C_];
  for (int i = 0; i < _LAT_C_; i++) {
    dest[i] = zero;
  }
  // just wilson(Sum part)
#ifdef __X__
  {   // x part
    { // x-1
      move_backward(move, x, lat_x);
      tmp_U = (origin_U + move * lat_y * lat_z * lat_t + _X_ * lat_xyzt);
      give_u_laplacian(U, tmp_U, lat_xyzt);
      tmp_src = (origin_src + move * lat_y * lat_z * lat_t);
      get_src_laplacian(src, tmp_src, lat_xyzt);
    }
    {
      for (int c0 = 0; c0 < _LAT_C_ * (move == -1); c0++) { // just inside
        tmp0 = zero;
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          tmp0 += src[c1] * U[c1 * _LAT_C_ + c0].conj();
        }
        dest[c0] += tmp0;
      }
    }
    {
      // x+1
      move_forward(move, x, lat_x);
      tmp_U = (origin_U + _X_ * lat_xyzt);
      give_u_laplacian(U, tmp_U, lat_xyzt);
      tmp_src = (origin_src + move * lat_y * lat_z * lat_t);
      get_src_laplacian(src, tmp_src, lat_xyzt);
    }
    {
      for (int c0 = 0; c0 < _LAT_C_ * (move == 1); c0++) { // just inside
        tmp0 = zero;
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          tmp0 += src[c1] * U[c0 * _LAT_C_ + c1];
        }
        dest[c0] += tmp0;
      }
    }
  }
#endif
#ifdef __Y__
  {   // y part
    { // y-1
      move_backward(move, y, lat_y);
      tmp_U = (origin_U + move * lat_z * lat_t + _Y_ * lat_xyzt);
      give_u_laplacian(U, tmp_U, lat_xyzt);
      tmp_src = (origin_src + move * lat_z * lat_t);
      get_src_laplacian(src, tmp_src, lat_xyzt);
    }
    {
      for (int c0 = 0; c0 < _LAT_C_ * (move == -1); c0++) { // just inside
        tmp0 = zero;
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          tmp0 += src[c1] * U[c1 * _LAT_C_ + c0].conj();
        }
        dest[c0] += tmp0;
      }
    }
    {
      // y+1
      move_forward(move, y, lat_y);
      tmp_U = (origin_U + _Y_ * lat_xyzt);
      give_u_laplacian(U, tmp_U, lat_xyzt);
      tmp_src = (origin_src + move * lat_z * lat_t);
      get_src_laplacian(src, tmp_src, lat_xyzt);
    }
    {
      for (int c0 = 0; c0 < _LAT_C_ * (move == 1); c0++) { // just inside
        tmp0 = zero;
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          tmp0 += src[c1] * U[c0 * _LAT_C_ + c1];
        }
        dest[c0] += tmp0;
      }
    }
  }
#endif
#ifdef __Z__
  {   // z part
    { // z-1
      move_backward(move, z, lat_z);
      tmp_U = (origin_U + move * lat_t + _Z_ * lat_xyzt);
      give_u_laplacian(U, tmp_U, lat_xyzt);
      tmp_src = (origin_src + move * lat_t);
      get_src_laplacian(src, tmp_src, lat_xyzt);
    }
    {
      for (int c0 = 0; c0 < _LAT_C_ * (move == -1); c0++) { // just inside
        tmp0 = zero;
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          tmp0 += src[c1] * U[c1 * _LAT_C_ + c0].conj();
        }
        dest[c0] += tmp0;
      }
    }
    {
      // z+1
      move_forward(move, z, lat_z);
      tmp_U = (origin_U + _Z_ * lat_xyzt);
      give_u_laplacian(U, tmp_U, lat_xyzt);
      tmp_src = (origin_src + move * lat_t);
      get_src_laplacian(src, tmp_src, lat_xyzt);
    }
    {
      for (int c0 = 0; c0 < _LAT_C_ * (move == 1); c0++) { // just inside
        tmp0 = zero;
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          tmp0 += src[c1] * U[c0 * _LAT_C_ + c1];
        }
        dest[c0] += tmp0;
      }
    }
  }
#endif
  give_dest_laplacian(origin_dest, dest, lat_xyzt);
}
template <typename T>
__global__ void laplacian_x_send(void *device_U, void *device_src,
                                 void *device_params, void *device_b_x_send_vec,
                                 void *device_f_x_send_vec) {
#ifdef __X__
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
  int *params = static_cast<int *>(device_params);
  // int lat_x = params[_LAT_X_];
  int lat_x = 1; // so let x=0 first, then x = lat_x -1
  int lat_y = params[_LAT_Y_];
  int lat_z = params[_LAT_Z_];
  int lat_t = params[_LAT_T_]; // in laplacian, lat_t = 1
  int lat_xyzt = params[_LAT_XYZT_];
  int move;
  move = lat_y * lat_z * lat_t;
  int x = parity / move;
  parity -= x * move;
  move = lat_z * lat_t;
  int y = parity / move;
  parity -= y * move;
  int z = parity / lat_t;
  int t = parity - z * lat_t;
  LatticeComplex<T> zero(0.0, 0.0);
  LatticeComplex<T> *tmp_U;
  LatticeComplex<T> tmp0(0.0, 0.0);
  LatticeComplex<T> U[_LAT_CC_];
  LatticeComplex<T> src[_LAT_C_];
  LatticeComplex<T> dest[_LAT_C_];
  for (int i = 0; i < _LAT_C_; i++) {
    dest[i] = zero;
  }
  LatticeComplex<T> b_x_send_vec[_LAT_C_];
  LatticeComplex<T> f_x_send_vec[_LAT_C_];
  LatticeComplex<T> *origin_U;
  LatticeComplex<T> *origin_src;
  LatticeComplex<T> *origin_b_x_send_vec;
  LatticeComplex<T> *origin_f_x_send_vec;
  {
    lat_x = params[_LAT_X_]; // give lat_size back
    x = 0;                   // b_x
    origin_src = ((static_cast<LatticeComplex<T> *>(device_src)) +
                  (((x * lat_y + y) * lat_z + z) * lat_t + t));
    origin_b_x_send_vec =
        ((static_cast<LatticeComplex<T> *>(device_b_x_send_vec)) +
         ((y * lat_z + z) * lat_t + t));
  }
  { // x-1
    move_backward(move, x, lat_x);
    // send in x+1 way
    get_src_laplacian(src, origin_src, lat_xyzt);
    { // just src
      for (int c0 = 0; c0 < _LAT_C_; c0++) {
        b_x_send_vec[c0] = src[c0];
      }
      give_send_laplacian(origin_b_x_send_vec, b_x_send_vec, lat_xyzt / lat_x);
    }
  }
  {
    x = lat_x - 1; // f_x
    origin_U = ((static_cast<LatticeComplex<T> *>(device_U)) +
                (((x * lat_y + y) * lat_z + z) * lat_t + t));
    origin_src = ((static_cast<LatticeComplex<T> *>(device_src)) +
                  (((x * lat_y + y) * lat_z + z) * lat_t + t));
    origin_f_x_send_vec =
        ((static_cast<LatticeComplex<T> *>(device_f_x_send_vec)) +
         ((y * lat_z + z) * lat_t + t));
  }
  { // x+1
    move_forward(move, x, lat_x);
    // send in x-1 way
    tmp_U = (origin_U + _X_ * lat_xyzt);
    give_u_laplacian(U, tmp_U, lat_xyzt);
    get_src_laplacian(src, origin_src, lat_xyzt);
    { // just tmp
      for (int c0 = 0; c0 < _LAT_C_; c0++) {
        tmp0 = zero;
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          tmp0 += src[c1] * U[c1 * _LAT_C_ + c0].conj();
        }
        f_x_send_vec[c0] = tmp0;
      }
      give_send_laplacian(origin_f_x_send_vec, f_x_send_vec, lat_xyzt / lat_x);
    }
  }
#endif
}
template <typename T>
__global__ void laplacian_x_recv(void *device_U, void *device_dest,
                                 void *device_params, void *device_b_x_recv_vec,
                                 void *device_f_x_recv_vec) {
#ifdef __X__
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
  int *params = static_cast<int *>(device_params);
  // int lat_x = params[_LAT_X_];
  int lat_x = 1; // so let x=0 first, then x = lat_x -1
  int lat_y = params[_LAT_Y_];
  int lat_z = params[_LAT_Z_];
  int lat_t = params[_LAT_T_]; // in laplacian, lat_t = 1
  int lat_xyzt = params[_LAT_XYZT_];
  int move;
  move = lat_y * lat_z * lat_t;
  int x = parity / move;
  parity -= x * move;
  move = lat_z * lat_t;
  int y = parity / move;
  parity -= y * move;
  int z = parity / lat_t;
  int t = parity - z * lat_t;
  LatticeComplex<T> zero(0.0, 0.0);
  LatticeComplex<T> *tmp_U;
  LatticeComplex<T> tmp0(0.0, 0.0);
  LatticeComplex<T> U[_LAT_CC_];
  LatticeComplex<T> dest[_LAT_C_];
  for (int i = 0; i < _LAT_C_; i++) {
    dest[i] = zero;
  }
  LatticeComplex<T> b_x_recv_vec[_LAT_C_];
  LatticeComplex<T> f_x_recv_vec[_LAT_C_]; // needed
  LatticeComplex<T> *origin_U;
  LatticeComplex<T> *origin_dest;
  LatticeComplex<T> *origin_b_x_recv_vec;
  LatticeComplex<T> *origin_f_x_recv_vec;
  {
    lat_x = params[_LAT_X_]; // give lat_size back
    x = 0;                   // b_x
    origin_dest = ((static_cast<LatticeComplex<T> *>(device_dest)) +
                   (((x * lat_y + y) * lat_z + z) * lat_t + t));
    origin_b_x_recv_vec =
        ((static_cast<LatticeComplex<T> *>(device_b_x_recv_vec)) +
         ((y * lat_z + z) * lat_t + t));
  }
  { // x-1
    move_backward(move, x, lat_x);
    // recv in x-1 way
    get_recv_laplacian(b_x_recv_vec, origin_b_x_recv_vec, lat_xyzt / lat_x);
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      dest[c0] += b_x_recv_vec[c0];
    }
  }
  // just add
  add_dest_laplacian(origin_dest, dest, lat_xyzt);
  for (int i = 0; i < _LAT_C_; i++) {
    dest[i] = zero;
  }
  {
    x = lat_x - 1; // f_x
    origin_U = ((static_cast<LatticeComplex<T> *>(device_U)) +
                (((x * lat_y + y) * lat_z + z) * lat_t + t));
    origin_dest = ((static_cast<LatticeComplex<T> *>(device_dest)) +
                   (((x * lat_y + y) * lat_z + z) * lat_t + t));
    origin_f_x_recv_vec =
        ((static_cast<LatticeComplex<T> *>(device_f_x_recv_vec)) +
         ((y * lat_z + z) * lat_t + t));
  }
  { // x+1
    move_forward(move, x, lat_x);
    // recv in x+1 way
    get_recv_laplacian(f_x_recv_vec, origin_f_x_recv_vec, lat_xyzt / lat_x);
    tmp_U = (origin_U + _X_ * lat_xyzt);
    give_u_laplacian(U, tmp_U, lat_xyzt);
    {
      for (int c0 = 0; c0 < _LAT_C_; c0++) {
        tmp0 = zero;
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          tmp0 += f_x_recv_vec[c1] * U[c0 * _LAT_C_ + c1];
        }
        dest[c0] += tmp0;
      }
    }
  } // just add
  add_dest_laplacian(origin_dest, dest, lat_xyzt);
#endif
}
template <typename T>
__global__ void laplacian_y_send(void *device_U, void *device_src,
                                 void *device_params, void *device_b_y_send_vec,
                                 void *device_f_y_send_vec) {
#ifdef __Y__
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
  int *params = static_cast<int *>(device_params);
  // int lat_x = params[_LAT_X_];
  // int lat_y = yyztsc[_y_];
  int lat_y = 1; // so let y=0 first, then y = lat_y -1
  int lat_z = params[_LAT_Z_];
  int lat_t = params[_LAT_T_]; // in laplacian, lat_t = 1
  int lat_xyzt = params[_LAT_XYZT_];
  int move;
  move = lat_y * lat_z * lat_t;
  int x = parity / move;
  parity -= x * move;
  move = lat_z * lat_t;
  int y = parity / move;
  parity -= y * move;
  int z = parity / lat_t;
  int t = parity - z * lat_t;
  //  LatticeComplex<T> I(0.0, 1.0);
  LatticeComplex<T> zero(0.0, 0.0);
  LatticeComplex<T> *tmp_U;
  LatticeComplex<T> tmp0(0.0, 0.0);
  LatticeComplex<T> U[_LAT_CC_];
  LatticeComplex<T> src[_LAT_C_];
  LatticeComplex<T> dest[_LAT_C_];
  for (int i = 0; i < _LAT_C_; i++) {
    dest[i] = zero;
  }
  LatticeComplex<T> b_y_send_vec[_LAT_C_];
  LatticeComplex<T> f_y_send_vec[_LAT_C_];
  LatticeComplex<T> *origin_U;
  LatticeComplex<T> *origin_src;
  LatticeComplex<T> *origin_b_y_send_vec;
  LatticeComplex<T> *origin_f_y_send_vec;
  {
    lat_y = params[_LAT_Y_]; // give lat_size back
    y = 0;                   // b_y
    origin_src = ((static_cast<LatticeComplex<T> *>(device_src)) +
                  (((x * lat_y + y) * lat_z + z) * lat_t + t));
    origin_b_y_send_vec =
        ((static_cast<LatticeComplex<T> *>(device_b_y_send_vec)) +
         (((x * lat_z + z)) * lat_t + t));
  }
  { // y-1
    // move_backward(move, y, lat_y);
    // send in y+1 way
    get_src_laplacian(src, origin_src, lat_xyzt);
    { // just src
      for (int c0 = 0; c0 < _LAT_C_; c0++) {
        b_y_send_vec[c0] = src[c0];
      }
      give_send_laplacian(origin_b_y_send_vec, b_y_send_vec, lat_xyzt / lat_y);
    }
  }
  {
    y = lat_y - 1; // f_y
    origin_U = ((static_cast<LatticeComplex<T> *>(device_U)) +
                (((x * lat_y + y) * lat_z + z) * lat_t + t));
    origin_src = ((static_cast<LatticeComplex<T> *>(device_src)) +
                  (((x * lat_y + y) * lat_z + z) * lat_t + t));
    origin_f_y_send_vec =
        ((static_cast<LatticeComplex<T> *>(device_f_y_send_vec)) +
         (((x * lat_z + z)) * lat_t + t));
  }
  { // y+1
    // move_forward(move, y, lat_y);
    // send in y-1 way
    tmp_U = (origin_U + _Y_ * lat_xyzt);
    give_u_laplacian(U, tmp_U, lat_xyzt);
    get_src_laplacian(src, origin_src, lat_xyzt);
    { // just tmp
      for (int c0 = 0; c0 < _LAT_C_; c0++) {
        tmp0 = zero;
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          tmp0 += src[c1] * U[c1 * _LAT_C_ + c0].conj();
        }
        f_y_send_vec[c0] = tmp0;
      }
      give_send_laplacian(origin_f_y_send_vec, f_y_send_vec, lat_xyzt / lat_y);
    }
  }
#endif
}
template <typename T>
__global__ void laplacian_y_recv(void *device_U, void *device_dest,
                                 void *device_params, void *device_b_y_recv_vec,
                                 void *device_f_y_recv_vec) {
#ifdef __Y__
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
  int *params = static_cast<int *>(device_params);
  // int lat_x = params[_LAT_X_];
  // int lat_y = yyztsc[_y_];
  int lat_y = 1; // so let y=0 first, then y = lat_y -1
  int lat_z = params[_LAT_Z_];
  int lat_t = params[_LAT_T_]; // in laplacian, lat_t = 1
  int lat_xyzt = params[_LAT_XYZT_];
  int move;
  move = lat_y * lat_z * lat_t;
  int x = parity / move;
  parity -= x * move;
  move = lat_z * lat_t;
  int y = parity / move;
  parity -= y * move;
  int z = parity / lat_t;
  int t = parity - z * lat_t;
  //  LatticeComplex<T> I(0.0, 1.0);
  LatticeComplex<T> zero(0.0, 0.0);
  LatticeComplex<T> *tmp_U;
  LatticeComplex<T> tmp0(0.0, 0.0);
  LatticeComplex<T> U[_LAT_CC_];
  LatticeComplex<T> dest[_LAT_C_];
  for (int i = 0; i < _LAT_C_; i++) {
    dest[i] = zero;
  }
  LatticeComplex<T> b_y_recv_vec[_LAT_C_];
  LatticeComplex<T> f_y_recv_vec[_LAT_C_]; // needed
  LatticeComplex<T> *origin_U;
  LatticeComplex<T> *origin_dest;
  LatticeComplex<T> *origin_b_y_recv_vec;
  LatticeComplex<T> *origin_f_y_recv_vec;
  {
    lat_y = params[_LAT_Y_]; // give lat_size back
    y = 0;                   // b_y
    origin_dest = ((static_cast<LatticeComplex<T> *>(device_dest)) +
                   (((x * lat_y + y) * lat_z + z) * lat_t + t));
    origin_b_y_recv_vec =
        ((static_cast<LatticeComplex<T> *>(device_b_y_recv_vec)) +
         (((x * lat_z + z)) * lat_t + t));
  }
  { // y-1
    move_backward(move, y, lat_y);
    // recv in y-1 way
    get_recv_laplacian(b_y_recv_vec, origin_b_y_recv_vec, lat_xyzt / lat_y);
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      dest[c0] += b_y_recv_vec[c0];
    }
  }
  // just add
  add_dest_laplacian(origin_dest, dest, lat_xyzt);
  for (int i = 0; i < _LAT_C_; i++) {
    dest[i] = zero;
  }
  {
    y = lat_y - 1; // f_y
    origin_U = ((static_cast<LatticeComplex<T> *>(device_U)) +
                (((x * lat_y + y) * lat_z + z) * lat_t + t));
    origin_dest = ((static_cast<LatticeComplex<T> *>(device_dest)) +
                   (((x * lat_y + y) * lat_z + z) * lat_t + t));
    origin_f_y_recv_vec =
        ((static_cast<LatticeComplex<T> *>(device_f_y_recv_vec)) +
         (((x * lat_z + z)) * lat_t + t));
  }
  { // y+1
    // move_forward(move, y, lat_y);
    // recv in y+1 way
    get_recv_laplacian(f_y_recv_vec, origin_f_y_recv_vec, lat_xyzt / lat_y);
    tmp_U = (origin_U + _Y_ * lat_xyzt);
    give_u_laplacian(U, tmp_U, lat_xyzt);
    {
      for (int c0 = 0; c0 < _LAT_C_; c0++) {
        tmp0 = zero;
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          tmp0 += f_y_recv_vec[c1] * U[c0 * _LAT_C_ + c1];
        }
        dest[c0] += tmp0;
      }
    }
  } // just add
  add_dest_laplacian(origin_dest, dest, lat_xyzt);
#endif
}
template <typename T>
__global__ void laplacian_z_send(void *device_U, void *device_src,
                                 void *device_params, void *device_b_z_send_vec,
                                 void *device_f_z_send_vec) {
#ifdef __Z__
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
  int *params = static_cast<int *>(device_params);
  // int lat_x = params[_LAT_X_];
  int lat_y = params[_LAT_Y_];
  // int lat_z = zzztsc[_z_];
  int lat_z = 1;               // so let z=0 first, then z = lat_z -1
  int lat_t = params[_LAT_T_]; // in laplacian, lat_t = 1
  int lat_xyzt = params[_LAT_XYZT_];
  int move;
  move = lat_y * lat_z * lat_t;
  int x = parity / move;
  parity -= x * move;
  move = lat_z * lat_t;
  int y = parity / move;
  parity -= y * move;
  int z = parity / lat_t;
  int t = parity - z * lat_t;
  //  LatticeComplex<T> I(0.0, 1.0);
  LatticeComplex<T> zero(0.0, 0.0);
  LatticeComplex<T> *tmp_U;
  LatticeComplex<T> tmp0(0.0, 0.0);
  LatticeComplex<T> U[_LAT_CC_];
  LatticeComplex<T> src[_LAT_C_];
  LatticeComplex<T> dest[_LAT_C_];
  for (int i = 0; i < _LAT_C_; i++) {
    dest[i] = zero;
  }
  LatticeComplex<T> b_z_send_vec[_LAT_C_];
  LatticeComplex<T> f_z_send_vec[_LAT_C_];
  LatticeComplex<T> *origin_U;
  LatticeComplex<T> *origin_src;
  LatticeComplex<T> *origin_b_z_send_vec;
  LatticeComplex<T> *origin_f_z_send_vec;
  {
    lat_z = params[_LAT_Z_]; // give lat_size back
    z = 0;                   // b_z
    origin_src = ((static_cast<LatticeComplex<T> *>(device_src)) +
                  (((x * lat_y + y) * lat_z + z) * lat_t + t));
    origin_b_z_send_vec =
        ((static_cast<LatticeComplex<T> *>(device_b_z_send_vec)) +
         (((x)*lat_y + y) * lat_t + t));
  }
  { // z-1
    // move_backward(move, z, lat_z);
    // send in z+1 way
    get_src_laplacian(src, origin_src, lat_xyzt);
    { // just src
      for (int c0 = 0; c0 < _LAT_C_; c0++) {
        b_z_send_vec[c0] = src[c0];
      }
      give_send_laplacian(origin_b_z_send_vec, b_z_send_vec, lat_xyzt / lat_z);
    }
  }
  {
    z = lat_z - 1; // f_z
    origin_U = ((static_cast<LatticeComplex<T> *>(device_U)) +
                (((x * lat_y + y) * lat_z + z) * lat_t + t));
    origin_src = ((static_cast<LatticeComplex<T> *>(device_src)) +
                  (((x * lat_y + y) * lat_z + z) * lat_t + t));
    origin_f_z_send_vec =
        ((static_cast<LatticeComplex<T> *>(device_f_z_send_vec)) +
         (((x)*lat_y + y) * lat_t + t));
  }
  { // z+1
    // move_forward(move, z, lat_z);
    // send in z-1 way
    tmp_U = (origin_U + _Z_ * lat_xyzt);
    give_u_laplacian(U, tmp_U, lat_xyzt);
    get_src_laplacian(src, origin_src, lat_xyzt);
    { // just tmp
      for (int c0 = 0; c0 < _LAT_C_; c0++) {
        tmp0 = zero;
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          tmp0 += src[c1] * U[c1 * _LAT_C_ + c0].conj();
        }
        f_z_send_vec[c0] = tmp0;
      }
      give_send_laplacian(origin_f_z_send_vec, f_z_send_vec, lat_xyzt / lat_z);
    }
  }
#endif
}
template <typename T>
__global__ void laplacian_z_recv(void *device_U, void *device_dest,
                                 void *device_params, void *device_b_z_recv_vec,
                                 void *device_f_z_recv_vec) {
#ifdef __Z__
  int parity = blockIdx.x * blockDim.x + threadIdx.x;
  int *params = static_cast<int *>(device_params);
  // int lat_x = params[_LAT_X_];
  int lat_y = params[_LAT_Y_];
  // int lat_z = zzztsc[_z_];
  int lat_z = 1;               // so let z=0 first, then z = lat_z -1
  int lat_t = params[_LAT_T_]; // in laplacian, lat_t = 1
  int lat_xyzt = params[_LAT_XYZT_];
  int move;
  move = lat_y * lat_z * lat_t;
  int x = parity / move;
  parity -= x * move;
  move = lat_z * lat_t;
  int y = parity / move;
  parity -= y * move;
  int z = parity / lat_t;
  int t = parity - z * lat_t;
  //  LatticeComplex<T> I(0.0, 1.0);
  LatticeComplex<T> zero(0.0, 0.0);
  LatticeComplex<T> *tmp_U;
  LatticeComplex<T> tmp0(0.0, 0.0);
  LatticeComplex<T> U[_LAT_CC_];
  LatticeComplex<T> dest[_LAT_C_];
  for (int i = 0; i < _LAT_C_; i++) {
    dest[i] = zero;
  }
  LatticeComplex<T> b_z_recv_vec[_LAT_C_];
  LatticeComplex<T> f_z_recv_vec[_LAT_C_]; // needed
  LatticeComplex<T> *origin_U;
  LatticeComplex<T> *origin_dest;
  LatticeComplex<T> *origin_b_z_recv_vec;
  LatticeComplex<T> *origin_f_z_recv_vec;
  {
    lat_z = params[_LAT_Z_]; // give lat_size back
    z = 0;                   // b_z
    origin_dest = ((static_cast<LatticeComplex<T> *>(device_dest)) +
                   (((x * lat_y + y) * lat_z + z) * lat_t + t));
    origin_b_z_recv_vec =
        ((static_cast<LatticeComplex<T> *>(device_b_z_recv_vec)) +
         (((x)*lat_y + y) * lat_t + t));
  }
  { // z-1
    // move_backward(move, z, lat_z);
    // recv in z-1 way
    get_recv_laplacian(b_z_recv_vec, origin_b_z_recv_vec, lat_xyzt / lat_z);
    for (int c0 = 0; c0 < _LAT_C_; c0++) {
      dest[c0] += b_z_recv_vec[c0];
    }
  }
  // just add
  add_dest_laplacian(origin_dest, dest, lat_xyzt);
  for (int i = 0; i < _LAT_C_; i++) {
    dest[i] = zero;
  }
  {
    z = lat_z - 1; // f_z
    origin_U = ((static_cast<LatticeComplex<T> *>(device_U)) +
                (((x * lat_y + y) * lat_z + z) * lat_t + t));
    origin_dest = ((static_cast<LatticeComplex<T> *>(device_dest)) +
                   (((x * lat_y + y) * lat_z + z) * lat_t + t));
    origin_f_z_recv_vec =
        ((static_cast<LatticeComplex<T> *>(device_f_z_recv_vec)) +
         (((x)*lat_y + y) * lat_t + t));
  }
  { // z+1
    // move_forward(move, z, lat_z);
    // recv in z+1 way
    get_recv_laplacian(f_z_recv_vec, origin_f_z_recv_vec, lat_xyzt / lat_z);
    tmp_U = (origin_U + _Z_ * lat_xyzt);
    give_u_laplacian(U, tmp_U, lat_xyzt);
    {
      for (int c0 = 0; c0 < _LAT_C_; c0++) {
        tmp0 = zero;
        for (int c1 = 0; c1 < _LAT_C_; c1++) {
          tmp0 += f_z_recv_vec[c1] * U[c0 * _LAT_C_ + c1];
        }
        dest[c0] += tmp0;
      }
    }
  } // just add
  add_dest_laplacian(origin_dest, dest, lat_xyzt);
#endif
}
template <typename T>
__global__ void laplacian_give_complete(void *device_dest, void *device_src,
                                        void *device_params) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  LatticeComplex<T> *dest =
      (static_cast<LatticeComplex<T> *>(device_dest) + idx);
  LatticeComplex<T> *src = (static_cast<LatticeComplex<T> *>(device_src) + idx);
  int _ = static_cast<int *>(device_params)[_LAT_XYZT_];
  for (int i = 0; i < _LAT_C_ * _; i += _) {
    dest[i] = src[i] * 6.0 - dest[i];
  }
}
//@@@CUDA_TEMPLATE_FOR_DEVICE@@@
template __global__ void laplacian_inside<double>(void *device_U,
                                                  void *device_src,
                                                  void *device_dest,
                                                  void *device_params);
template __global__ void
laplacian_x_send<double>(void *device_U, void *device_src, void *device_params,
                         void *device_b_x_send_vec, void *device_f_x_send_vec);
template __global__ void
laplacian_x_recv<double>(void *device_U, void *device_dest, void *device_params,
                         void *device_b_x_recv_vec, void *device_f_x_recv_vec);
template __global__ void
laplacian_y_send<double>(void *device_U, void *device_src, void *device_params,
                         void *device_b_y_send_vec, void *device_f_y_send_vec);
template __global__ void
laplacian_y_recv<double>(void *device_U, void *device_dest, void *device_params,
                         void *device_b_y_recv_vec, void *device_f_y_recv_vec);
template __global__ void
laplacian_z_send<double>(void *device_U, void *device_src, void *device_params,
                         void *device_b_z_send_vec, void *device_f_z_send_vec);
template __global__ void
laplacian_z_recv<double>(void *device_U, void *device_dest, void *device_params,
                         void *device_b_z_recv_vec, void *device_f_z_recv_vec);
template __global__ void laplacian_give_complete<double>(void *device_dest,
                                                         void *device_src,
                                                         void *device_params);
//@@@CUDA_TEMPLATE_FOR_DEVICE@@@
template __global__ void laplacian_inside<float>(void *device_U,
                                                 void *device_src,
                                                 void *device_dest,
                                                 void *device_params);
template __global__ void
laplacian_x_send<float>(void *device_U, void *device_src, void *device_params,
                        void *device_b_x_send_vec, void *device_f_x_send_vec);
template __global__ void
laplacian_x_recv<float>(void *device_U, void *device_dest, void *device_params,
                        void *device_b_x_recv_vec, void *device_f_x_recv_vec);
template __global__ void
laplacian_y_send<float>(void *device_U, void *device_src, void *device_params,
                        void *device_b_y_send_vec, void *device_f_y_send_vec);
template __global__ void
laplacian_y_recv<float>(void *device_U, void *device_dest, void *device_params,
                        void *device_b_y_recv_vec, void *device_f_y_recv_vec);
template __global__ void
laplacian_z_send<float>(void *device_U, void *device_src, void *device_params,
                        void *device_b_z_send_vec, void *device_f_z_send_vec);
template __global__ void
laplacian_z_recv<float>(void *device_U, void *device_dest, void *device_params,
                        void *device_b_z_recv_vec, void *device_f_z_recv_vec);
template __global__ void laplacian_give_complete<float>(void *device_dest,
                                                        void *device_src,
                                                        void *device_params);
} // namespace qcu