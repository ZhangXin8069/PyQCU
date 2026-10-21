#include "../include/qcu.h"
#pragma optimize(5)
namespace qcu {
template <typename T>
__global__ void make_clover(void *device_U, void *device_clover,
                            void *device_params, T clover_coefficient) {
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
  int eo = (x + y + z) & 0x01; // (x+y+z)%2
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
  LatticeComplex<T> U[_LAT_CC_];
  LatticeComplex<T> clover[_LAT_SCSC_];
  // sigmaF
  {
    get_vals(clover, zero, _LAT_SCSC_);
    get_vals(tmp1, zero, _LAT_CC_);
    get_vals(tmp2, zero, _LAT_CC_);
  }
  // XY
  get_vals(U, zero, _LAT_CC_);
  {
    //// x,y,z,t;x
    tmp_U = (origin_U + (_X_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x+1,y,z,t;y
    move0 = move_wards[_F_X_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
             (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z,t;x;dag
    move0 = move_wards[_F_Y_];
    tmp_U = (origin_U + move0 * lat_z * lat_t +
             (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;y;dag
    tmp_U = (origin_U + (_Y_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;y
    tmp_U = (origin_U + (_Y_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x-1,y+1,z,t;x;dag
    move0 = move_wards[_B_X_];
    move1 = move_wards[_F_Y_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t + move1 * lat_z * lat_t +
             (_X_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;y;dag
    move0 = move_wards[_B_X_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
             (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move0 = move_wards[_B_X_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
             (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x-1,y,z,t;x;dag
    move0 = move_wards[_B_X_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
             (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x-1,y-1,z,t;y;dag
    move0 = move_wards[_B_X_];
    move1 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t + move1 * lat_z * lat_t +
             (_Y_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y-1,z,t;x
    move0 = move_wards[_B_X_];
    move1 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t + move1 * lat_z * lat_t +
             (_X_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    move0 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 * lat_z * lat_t +
             (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y-1,z,t;y;dag
    move0 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 * lat_z * lat_t +
             (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x,y-1,z,t;x
    move0 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 * lat_z * lat_t +
             (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y-1,z,t;y
    move0 = move_wards[_F_X_];
    move1 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t + move1 * lat_z * lat_t +
             (_Y_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;x;dag
    tmp_U = (origin_U + (_X_ + (parity)*_LAT_CCD_) * lat_xyzt);
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
    tmp_U = (origin_U + (_X_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x+1,y,z,t;z
    move0 = move_wards[_F_X_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
             (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t;x;dag
    move0 = move_wards[_F_Z_];
    tmp_U = (origin_U + move0 * lat_t +
             (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;z;dag
    tmp_U = (origin_U + (_Z_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;z
    tmp_U = (origin_U + (_Z_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x-1,y,z+1,t;x;dag
    move0 = move_wards[_B_X_];
    move1 = move_wards[_F_Z_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t + move1 * lat_t +
             (_X_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;z;dag
    move0 = move_wards[_B_X_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
             (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move0 = move_wards[_B_X_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
             (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x-1,y,z,t;x;dag
    move0 = move_wards[_B_X_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
             (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x-1,y,z-1,t;z;dag
    move0 = move_wards[_B_X_];
    move1 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t + move1 * lat_t +
             (_Z_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z-1,t;x
    move0 = move_wards[_B_X_];
    move1 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t + move1 * lat_t +
             (_X_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    move0 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_t +
             (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z-1,t;z;dag
    move0 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_t +
             (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x,y,z-1,t;x
    move0 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_t +
             (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y,z-1,t;z
    move0 = move_wards[_F_X_];
    move1 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t + move1 * lat_t +
             (_Z_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;x;dag
    tmp_U = (origin_U + (_X_ + (parity)*_LAT_CCD_) * lat_xyzt);
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
    tmp_U = (origin_U + (_X_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x+1,y,z,t;t
    move0 = move_wards[_F_X_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
             (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;x;dag
    move0 = move_wards[_F_T_];
    tmp_U = (origin_U + move0 + (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;t;dag
    tmp_U = (origin_U + (_T_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;t
    tmp_U = (origin_U + (_T_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x-1,y,z,t+1;x;dag
    move0 = move_wards[_B_X_];
    move1 = move_wards[_F_T_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t + move1 +
             (_X_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t;t;dag
    move0 = move_wards[_B_X_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
             (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x-1,y,z,t;x
    move0 = move_wards[_B_X_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
             (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x-1,y,z,t;x;dag
    move0 = move_wards[_B_X_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t +
             (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x-1,y,z,t-1;t;dag
    move0 = move_wards[_B_X_];
    move1 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t + move1 +
             (_T_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x-1,y,z,t-1;x
    move0 = move_wards[_B_X_];
    move1 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t + move1 +
             (_X_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    move0 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 + (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t-1;t;dag
    move0 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 + (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x,y,z,t-1;x
    move0 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 + (_X_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x+1,y,z,t-1;t
    move0 = move_wards[_F_X_];
    move1 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_y * lat_z * lat_t + move1 +
             (_T_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;x;dag
    tmp_U = (origin_U + (_X_ + (parity)*_LAT_CCD_) * lat_xyzt);
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
    tmp_U = (origin_U + (_Y_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x,y+1,z,t;z
    move0 = move_wards[_F_Y_];
    tmp_U = (origin_U + move0 * lat_z * lat_t +
             (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t;y;dag
    move0 = move_wards[_F_Z_];
    tmp_U = (origin_U + move0 * lat_t +
             (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;z;dag
    tmp_U = (origin_U + (_Z_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;z
    tmp_U = (origin_U + (_Z_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x,y-1,z+1,t;y;dag
    move0 = move_wards[_B_Y_];
    move1 = move_wards[_F_Z_];
    tmp_U = (origin_U + move0 * lat_z * lat_t + move1 * lat_t +
             (_Y_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t;z;dag
    move0 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 * lat_z * lat_t +
             (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    move0 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 * lat_z * lat_t +
             (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y-1,z,t;y;dag
    move0 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 * lat_z * lat_t +
             (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x,y-1,z-1,t;z;dag
    move0 = move_wards[_B_Y_];
    move1 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_z * lat_t + move1 * lat_t +
             (_Z_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z-1,t;y
    move0 = move_wards[_B_Y_];
    move1 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_z * lat_t + move1 * lat_t +
             (_Y_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    move0 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_t +
             (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z-1,t;z;dag
    move0 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_t +
             (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x,y,z-1,t;y
    move0 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_t +
             (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z-1,t;z
    move0 = move_wards[_F_Y_];
    move1 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_z * lat_t + move1 * lat_t +
             (_Z_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;y;dag
    tmp_U = (origin_U + (_Y_ + (parity)*_LAT_CCD_) * lat_xyzt);
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
    tmp_U = (origin_U + (_Y_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x,y+1,z,t;t
    move0 = move_wards[_F_Y_];
    tmp_U = (origin_U + move0 * lat_z * lat_t +
             (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;y;dag
    move0 = move_wards[_F_T_];
    tmp_U = (origin_U + move0 + (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;t;dag
    tmp_U = (origin_U + (_T_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;t
    tmp_U = (origin_U + (_T_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x,y-1,z,t+1;y;dag
    move0 = move_wards[_B_Y_];
    move1 = move_wards[_F_T_];
    tmp_U = (origin_U + move0 * lat_z * lat_t + move1 +
             (_Y_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t;t;dag
    move0 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 * lat_z * lat_t +
             (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y-1,z,t;y
    move0 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 * lat_z * lat_t +
             (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y-1,z,t;y;dag
    move0 = move_wards[_B_Y_];
    tmp_U = (origin_U + move0 * lat_z * lat_t +
             (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x,y-1,z,t-1;t;dag
    move0 = move_wards[_B_Y_];
    move1 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_z * lat_t + move1 +
             (_T_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y-1,z,t-1;y
    move0 = move_wards[_B_Y_];
    move1 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_z * lat_t + move1 +
             (_Y_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    move0 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 + (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t-1;t;dag
    move0 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 + (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x,y,z,t-1;y
    move0 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 + (_Y_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y+1,z,t-1;t
    move0 = move_wards[_F_Y_];
    move1 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_z * lat_t + move1 +
             (_T_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;y;dag
    tmp_U = (origin_U + (_Y_ + (parity)*_LAT_CCD_) * lat_xyzt);
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
    tmp_U = (origin_U + (_Z_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x,y,z+1,t;t
    move0 = move_wards[_F_Z_];
    tmp_U = (origin_U + move0 * lat_t +
             (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z,t+1;z;dag
    move0 = move_wards[_F_T_];
    tmp_U = (origin_U + move0 + (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;t;dag
    tmp_U = (origin_U + (_T_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t;t
    tmp_U = (origin_U + (_T_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x,y,z-1,t+1;z;dag
    move0 = move_wards[_B_Z_];
    move1 = move_wards[_F_T_];
    tmp_U = (origin_U + move0 * lat_t + move1 +
             (_Z_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z-1,t;t;dag
    move0 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_t +
             (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z-1,t;z
    move0 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_t +
             (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z-1,t;z;dag
    move0 = move_wards[_B_Z_];
    tmp_U = (origin_U + move0 * lat_t +
             (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x,y,z-1,t-1;t;dag
    move0 = move_wards[_B_Z_];
    move1 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_t + move1 +
             (_T_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z-1,t-1;z
    move0 = move_wards[_B_Z_];
    move1 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_t + move1 +
             (_Z_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t-1;t
    move0 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 + (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
  }
  add_vals(U, tmp3, _LAT_CC_);
  {
    //// x,y,z,t-1;t;dag
    move0 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 + (_T_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    //// x,y,z,t-1;z
    move0 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 + (_Z_ + (1 - parity) * _LAT_CCD_) * lat_xyzt);
    get_u(tmp2, tmp_U, lat_xyzt);
    mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
  }
  {
    //// x,y,z+1,t-1;t
    move0 = move_wards[_F_Z_];
    move1 = move_wards[_B_T_];
    tmp_U = (origin_U + move0 * lat_t + move1 +
             (_T_ + (parity)*_LAT_CCD_) * lat_xyzt);
    get_u(tmp1, tmp_U, lat_xyzt);
    mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
  }
  {
    //// x,y,z,t;z;dag
    tmp_U = (origin_U + (_Z_ + (parity)*_LAT_CCD_) * lat_xyzt);
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
    // A = 1 + T, with T proportional to kappa * csw.
    LatticeComplex<T> one(1.0, 0);
    for (int i = 0; i < _LAT_SCSC_; i++) {
      clover[i] *= -clover_coefficient * 0.125;
    }
    for (int i = 0; i < _LAT_SC_; i++) {
      clover[i * 13] += one;
    }
  }
  give_clr(origin_clover, clover, lat_xyzt);
}
template <typename T>
__global__ void inverse_clover(void *device_clover, void *device_params) {
  LatticeComplex<T> *origin_clover;
  int lat_xyzt = static_cast<int *>(device_params)[_LAT_XYZT_];
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    origin_clover = ((static_cast<LatticeComplex<T> *>(device_clover)) + idx);
  }
  {
    LatticeComplex<T> pivot;
    LatticeComplex<T> factor;
    LatticeComplex<T> clover[_LAT_SCSC_];
    LatticeComplex<T> augmented_clover[_LAT_SCSC_ * _BF_];
    get_clr(clover, origin_clover, lat_xyzt);
    _inverse(clover, clover, augmented_clover, pivot, factor, _LAT_SC_);
    give_clr(origin_clover, clover, lat_xyzt);
  }
}
template <typename T>
__global__ void check_clover_block_diagonal(const void *device_clover,
                                            void *device_params, int *flag) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int *params = static_cast<const int *>(device_params);
  const int lat_xyzt = params[_LAT_XYZT_];
  const int half_sites = lat_xyzt / 2;
  if (idx >= half_sites) return;
  const LatticeComplex<T> *clover =
      static_cast<const LatticeComplex<T> *>(device_clover) + idx;
  bool off_diagonal_nonzero = false;
  for (int row = 0; row < 6 && !off_diagonal_nonzero; ++row) {
    for (int column = 6; column < _LAT_SC_; ++column) {
      const LatticeComplex<T> value =
          clover[(row * _LAT_SC_ + column) * lat_xyzt];
      if (value.real() != T(0) || value.imag() != T(0)) {
        off_diagonal_nonzero = true;
        break;
      }
    }
  }
  for (int row = 6; row < _LAT_SC_ && !off_diagonal_nonzero; ++row) {
    for (int column = 0; column < 6; ++column) {
      const LatticeComplex<T> value =
          clover[(row * _LAT_SC_ + column) * lat_xyzt];
      if (value.real() != T(0) || value.imag() != T(0)) {
        off_diagonal_nonzero = true;
        break;
      }
    }
  }
  if (off_diagonal_nonzero) atomicExch(flag, 1);
}
template <typename T>
__global__ void give_clover(void *device_clover, void *device_dest,
                            void *device_params) {
  LatticeComplex<T> *origin_clover;
  LatticeComplex<T> *origin_dest;
  int lat_xyzt = static_cast<int *>(device_params)[_LAT_XYZT_];
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    origin_clover = ((static_cast<LatticeComplex<T> *>(device_clover)) + idx);
    origin_dest = ((static_cast<LatticeComplex<T> *>(device_dest)) + idx);
  }
  {
    LatticeComplex<T> dest[_LAT_SC_];
    get_src(dest, origin_dest, lat_xyzt);
    for (int row = 0; row < _LAT_SC_; ++row) {
      LatticeComplex<T> value(0.0, 0.0);
      for (int column = 0; column < _LAT_SC_; ++column) {
        value += origin_clover[(row * _LAT_SC_ + column) * lat_xyzt] *
                 dest[column];
      }
      origin_dest[row * lat_xyzt] = value;
    }
  }
}
template <typename T>
__global__ void give_clover_block(void *device_clover, void *device_dest,
                                  void *device_params) {
  LatticeComplex<T> *origin_clover;
  LatticeComplex<T> *origin_dest;
  int lat_xyzt = static_cast<int *>(device_params)[_LAT_XYZT_];
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    origin_clover = ((static_cast<LatticeComplex<T> *>(device_clover)) + idx);
    origin_dest = ((static_cast<LatticeComplex<T> *>(device_dest)) + idx);
  }
  {
    LatticeComplex<T> dest[_LAT_SC_];
    get_src(dest, origin_dest, lat_xyzt);
    for (int row = 0; row < _LAT_SC_; ++row) {
      const int column_base = (row / 6) * 6;
      LatticeComplex<T> value(0.0, 0.0);
      for (int column = column_base; column < column_base + 6; ++column) {
        value += origin_clover[(row * _LAT_SC_ + column) * lat_xyzt] *
                 dest[column];
      }
      origin_dest[row * lat_xyzt] = value;
    }
  }
}
template <typename T>
__global__ void give_clover_to(void *device_clover, const void *device_src,
                               void *device_dest, void *device_params) {
  LatticeComplex<T> *origin_clover;
  const LatticeComplex<T> *origin_src;
  LatticeComplex<T> *origin_dest;
  int lat_xyzt = static_cast<int *>(device_params)[_LAT_XYZT_];
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    origin_clover = ((static_cast<LatticeComplex<T> *>(device_clover)) + idx);
    origin_src =
        ((static_cast<const LatticeComplex<T> *>(device_src)) + idx);
    origin_dest = ((static_cast<LatticeComplex<T> *>(device_dest)) + idx);
  }
  {
    LatticeComplex<T> src[_LAT_SC_];
    get_src(src, origin_src, lat_xyzt);
    for (int row = 0; row < _LAT_SC_; ++row) {
      LatticeComplex<T> value(0.0, 0.0);
      for (int column = 0; column < _LAT_SC_; ++column) {
        value += origin_clover[(row * _LAT_SC_ + column) * lat_xyzt] *
                 src[column];
      }
      origin_dest[row * lat_xyzt] = value;
    }
  }
}
template <typename T>
__global__ void give_clover_to_block(void *device_clover,
                                     const void *device_src,
                                     void *device_dest, void *device_params) {
  LatticeComplex<T> *origin_clover;
  const LatticeComplex<T> *origin_src;
  LatticeComplex<T> *origin_dest;
  int lat_xyzt = static_cast<int *>(device_params)[_LAT_XYZT_];
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    origin_clover = ((static_cast<LatticeComplex<T> *>(device_clover)) + idx);
    origin_src =
        ((static_cast<const LatticeComplex<T> *>(device_src)) + idx);
    origin_dest = ((static_cast<LatticeComplex<T> *>(device_dest)) + idx);
  }
  {
    LatticeComplex<T> src[_LAT_SC_];
    get_src(src, origin_src, lat_xyzt);
    for (int row = 0; row < _LAT_SC_; ++row) {
      const int column_base = (row / 6) * 6;
      LatticeComplex<T> value(0.0, 0.0);
      for (int column = column_base; column < column_base + 6; ++column) {
        value += origin_clover[(row * _LAT_SC_ + column) * lat_xyzt] *
                 src[column];
      }
      origin_dest[row * lat_xyzt] = value;
    }
  }
}
template <typename T>
__global__ void give_clover_subtract(
    void *device_clover, const void *device_src, const void *device_subtract,
    void *device_dest, T scale, void *device_params) {
  LatticeComplex<T> *origin_clover;
  const LatticeComplex<T> *origin_src;
  const LatticeComplex<T> *origin_subtract;
  LatticeComplex<T> *origin_dest;
  int lat_xyzt = static_cast<int *>(device_params)[_LAT_XYZT_];
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    origin_clover = ((static_cast<LatticeComplex<T> *>(device_clover)) + idx);
    origin_src =
        ((static_cast<const LatticeComplex<T> *>(device_src)) + idx);
    origin_subtract =
        ((static_cast<const LatticeComplex<T> *>(device_subtract)) + idx);
    origin_dest = ((static_cast<LatticeComplex<T> *>(device_dest)) + idx);
  }
  {
    LatticeComplex<T> src[_LAT_SC_];
    get_src(src, origin_src, lat_xyzt);
    for (int row = 0; row < _LAT_SC_; ++row) {
      LatticeComplex<T> value(0.0, 0.0);
      for (int column = 0; column < _LAT_SC_; ++column) {
        value += origin_clover[(row * _LAT_SC_ + column) * lat_xyzt] *
                 src[column];
      }
      origin_dest[row * lat_xyzt] =
          value - origin_subtract[row * lat_xyzt] * scale;
    }
  }
}
template <typename T>
__global__ void give_clover_subtract_block(
    void *device_clover, const void *device_src, const void *device_subtract,
    void *device_dest, T scale, void *device_params) {
  LatticeComplex<T> *origin_clover;
  const LatticeComplex<T> *origin_src;
  const LatticeComplex<T> *origin_subtract;
  LatticeComplex<T> *origin_dest;
  int lat_xyzt = static_cast<int *>(device_params)[_LAT_XYZT_];
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    origin_clover = ((static_cast<LatticeComplex<T> *>(device_clover)) + idx);
    origin_src =
        ((static_cast<const LatticeComplex<T> *>(device_src)) + idx);
    origin_subtract =
        ((static_cast<const LatticeComplex<T> *>(device_subtract)) + idx);
    origin_dest = ((static_cast<LatticeComplex<T> *>(device_dest)) + idx);
  }
  {
    LatticeComplex<T> src[_LAT_SC_];
    get_src(src, origin_src, lat_xyzt);
    for (int row = 0; row < _LAT_SC_; ++row) {
      const int column_base = (row / 6) * 6;
      LatticeComplex<T> value(0.0, 0.0);
      for (int column = column_base; column < column_base + 6; ++column) {
        value += origin_clover[(row * _LAT_SC_ + column) * lat_xyzt] *
                 src[column];
      }
      origin_dest[row * lat_xyzt] =
          value - origin_subtract[row * lat_xyzt] * scale;
    }
  }
}
//@@@CUDA_TEMPLATE_FOR_DEVICE@@@
template __global__ void make_clover<double>(void *device_U,
                                             void *device_clover,
                                             void *device_params, double T);
template __global__ void inverse_clover<double>(void *device_clover,
                                                void *device_params);
template __global__ void check_clover_block_diagonal<double>(
    const void *device_clover, void *device_params, int *flag);
template __global__ void give_clover<double>(void *device_clover,
                                             void *device_dest,
                                             void *device_params);
template __global__ void give_clover_block<double>(void *device_clover,
                                                   void *device_dest,
                                                   void *device_params);
template __global__ void give_clover_to<double>(
    void *device_clover, const void *device_src, void *device_dest,
    void *device_params);
template __global__ void give_clover_to_block<double>(
    void *device_clover, const void *device_src, void *device_dest,
    void *device_params);
template __global__ void give_clover_subtract<double>(
    void *device_clover, const void *device_src, const void *device_subtract,
    void *device_dest, double scale, void *device_params);
template __global__ void give_clover_subtract_block<double>(
    void *device_clover, const void *device_src, const void *device_subtract,
    void *device_dest, double scale, void *device_params);
//@@@CUDA_TEMPLATE_FOR_DEVICE@@@
template __global__ void make_clover<float>(void *device_U, void *device_clover,
                                            void *device_params, float T);
template __global__ void inverse_clover<float>(void *device_clover,
                                               void *device_params);
template __global__ void check_clover_block_diagonal<float>(
    const void *device_clover, void *device_params, int *flag);
template __global__ void
give_clover<float>(void *device_clover, void *device_dest, void *device_params);
template __global__ void give_clover_block<float>(void *device_clover,
                                                  void *device_dest,
                                                  void *device_params);
template __global__ void give_clover_to<float>(
    void *device_clover, const void *device_src, void *device_dest,
    void *device_params);
template __global__ void give_clover_to_block<float>(
    void *device_clover, const void *device_src, void *device_dest,
    void *device_params);
template __global__ void give_clover_subtract<float>(
    void *device_clover, const void *device_src, const void *device_subtract,
    void *device_dest, float scale, void *device_params);
template __global__ void give_clover_subtract_block<float>(
    void *device_clover, const void *device_src, const void *device_subtract,
    void *device_dest, float scale, void *device_params);
} // namespace qcu
