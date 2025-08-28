#include "../include/qcu.h"
#pragma optimize(5)
namespace qcu
{
  template <typename T>
  __global__ void make_clover(void *device_U, void *device_clover,
                              void *device_params, T kappa)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int parity = idx;
    int *params = static_cast<int *>(device_params);
    int lat_x = params[_LAT_X_];
    int lat_y = params[_LAT_Y_];
    int lat_z = params[_LAT_Z_];
    int lat_t = params[_LAT_T_];
    int lat_tzyx = params[_LAT_XYZT_];
    int move0;
    int move1;
    move0 = lat_x * lat_y * lat_z;
    int t = parity / move0;
    parity -= t * move0;
    move0 = lat_x * lat_y;
    int z = parity / move0;
    parity -= z * move0;
    int y = parity / lat_x;
    int x = parity - y * lat_x;
    int eo = (y + z + t) & 0x01; // (y+z+t)%2
    parity = params[_PARITY_];
    int move_wards[_WARDS_];
    move_backward_x(move_wards[_B_X_], x, lat_x, eo, parity);
    move_backward(move_wards[_B_Y_], y, lat_y);
    move_backward(move_wards[_B_Z_], z, lat_z);
    move_backward(move_wards[_B_T_], t, lat_t);
    move_forward_x(move_wards[_F_X_], x, lat_x, eo, parity);
    move_forward(move_wards[_F_Y_], y, lat_y);
    move_forward(move_wards[_F_Z_], z, lat_z);
    move_forward(move_wards[_F_T_], t, lat_t);
    //  LatticeComplex<T> I(0.0, 1.0);
    LatticeComplex<T> zero(0.0, 0.0);
    LatticeComplex<T> tmp0(0.0, 0.0);
    LatticeComplex<T> *origin_U = ((static_cast<LatticeComplex<T> *>(device_U)) + idx);
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
      give_vals(clover, zero, _LAT_SCSC_);
      give_vals(tmp1, zero, _LAT_CC_);
      give_vals(tmp2, zero, _LAT_CC_);
    }
    // XY
    give_vals(U, zero, _LAT_CC_);
    {
      //// x,y,z,t;x
      tmp_U = (origin_U + (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x+1,y,z,t;y
      move0 = move_wards[_F_X_];
      tmp_U = (origin_U + move0 + (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x,y+1,z,t;x;dag
      move0 = move_wards[_F_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x,y,z,t;y;dag
      tmp_U = (origin_U + (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      //// x,y,z,t;y
      tmp_U = (origin_U + (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x-1,y+1,z,t;x;dag
      move0 = move_wards[_B_X_];
      move1 = move_wards[_F_Y_];
      tmp_U = (origin_U + move0 + move1 * lat_x +
               (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x-1,y,z,t;y;dag
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 + (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x-1,y,z,t;x
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      //// x-1,y,z,t;x;dag
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x-1,y-1,z,t;y;dag
      move0 = move_wards[_B_X_];
      move1 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 + move1 * lat_x +
               (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x-1,y-1,z,t;x
      move0 = move_wards[_B_X_];
      move1 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 + move1 * lat_x +
               (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x,y-1,z,t;y
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      //// x,y-1,z,t;y;dag
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x,y-1,z,t;x
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x+1,y-1,z,t;y
      move0 = move_wards[_F_X_];
      move1 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 + move1 * lat_x +
               (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x,y,z,t;x;dag
      tmp_U = (origin_U + (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      for (int c0 = 0; c0 < _LAT_C_; c0++)
      {
        for (int c1 = 0; c1 < _LAT_C_; c1++)
        {
          clover[c0 * _LAT_SC_ + c1] +=
              (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_minus_i();
          clover[39 + c0 * _LAT_SC_ + c1] +=
              (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_i();
          clover[78 + c0 * _LAT_SC_ + c1] +=
              (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_minus_i();
          clover[117 + c0 * _LAT_SC_ + c1] +=
              (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_i();
        }
      }
    }
    // XZ
    give_vals(U, zero, _LAT_CC_);
    {
      //// x,y,z,t;x
      tmp_U = (origin_U + (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x+1,y,z,t;z
      move0 = move_wards[_F_X_];
      tmp_U = (origin_U + move0 + (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x,y,z+1,t;x;dag
      move0 = move_wards[_F_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x,y,z,t;z;dag
      tmp_U = (origin_U + (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      //// x,y,z,t;z
      tmp_U = (origin_U + (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x-1,y,z+1,t;x;dag
      move0 = move_wards[_B_X_];
      move1 = move_wards[_F_Z_];
      tmp_U = (origin_U + move0 + move1 * lat_y * lat_x +
               (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x-1,y,z,t;z;dag
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 + (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x-1,y,z,t;x
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      //// x-1,y,z,t;x;dag
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x-1,y,z-1,t;z;dag
      move0 = move_wards[_B_X_];
      move1 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 + move1 * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x-1,y,z-1,t;x
      move0 = move_wards[_B_X_];
      move1 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 + move1 * lat_y * lat_x +
               (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x,y,z-1,t;z
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      //// x,y,z-1,t;z;dag
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x,y,z-1,t;x
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x+1,y,z-1,t;z
      move0 = move_wards[_F_X_];
      move1 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 + move1 * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x,y,z,t;x;dag
      tmp_U = (origin_U + (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      for (int c0 = 0; c0 < _LAT_C_; c0++)
      {
        for (int c1 = 0; c1 < _LAT_C_; c1++)
        {
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
    give_vals(U, zero, _LAT_CC_);
    {
      //// x,y,z,t;x
      tmp_U = (origin_U + (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x+1,y,z,t;t
      move0 = move_wards[_F_X_];
      tmp_U = (origin_U + move0 + (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x,y,z,t+1;x;dag
      move0 = move_wards[_F_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x,y,z,t;t;dag
      tmp_U = (origin_U + (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      //// x,y,z,t;t
      tmp_U = (origin_U + (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x-1,y,z,t+1;x;dag
      move0 = move_wards[_B_X_];
      move1 = move_wards[_F_T_];
      tmp_U = (origin_U + move0 + move1 * lat_z * lat_y * lat_x +
               (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x-1,y,z,t;t;dag
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 + (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x-1,y,z,t;x
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      //// x-1,y,z,t;x;dag
      move0 = move_wards[_B_X_];
      tmp_U = (origin_U + move0 + (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x-1,y,z,t-1;t;dag
      move0 = move_wards[_B_X_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 + move1 * lat_z * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x-1,y,z,t-1;x
      move0 = move_wards[_B_X_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 + move1 * lat_z * lat_y * lat_x +
               (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x,y,z,t-1;t
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      //// x,y,z,t-1;t;dag
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x,y,z,t-1;x
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_X_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x+1,y,z,t-1;t
      move0 = move_wards[_F_X_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 + move1 * lat_z * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x,y,z,t;x;dag
      tmp_U = (origin_U + (_X_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      for (int c0 = 0; c0 < _LAT_C_; c0++)
      {
        for (int c1 = 0; c1 < _LAT_C_; c1++)
        {
          clover[_LAT_C_ + c0 * _LAT_SC_ + c1] +=
              (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_i();
          clover[36 + c0 * _LAT_SC_ + c1] +=
              (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_i();
          clover[81 + c0 * _LAT_SC_ + c1] +=
              (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_minus_i();
          clover[114 + c0 * _LAT_SC_ + c1] +=
              (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_minus_i();
        }
      }
    }
    // YZ
    give_vals(U, zero, _LAT_CC_);
    {
      //// x,y,z,t;y
      tmp_U = (origin_U + (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x,y+1,z,t;z
      move0 = move_wards[_F_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x,y,z+1,t;y;dag
      move0 = move_wards[_F_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x,y,z,t;z;dag
      tmp_U = (origin_U + (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      //// x,y,z,t;z
      tmp_U = (origin_U + (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x,y-1,z+1,t;y;dag
      move0 = move_wards[_B_Y_];
      move1 = move_wards[_F_Z_];
      tmp_U = (origin_U + move0 * lat_x + move1 * lat_y * lat_x +
               (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x,y-1,z,t;z;dag
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x,y-1,z,t;y
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      //// x,y-1,z,t;y;dag
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x,y-1,z-1,t;z;dag
      move0 = move_wards[_B_Y_];
      move1 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_x + move1 * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x,y-1,z-1,t;y
      move0 = move_wards[_B_Y_];
      move1 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_x + move1 * lat_y * lat_x +
               (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x,y,z-1,t;z
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      //// x,y,z-1,t;z;dag
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x,y,z-1,t;y
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x,y+1,z-1,t;z
      move0 = move_wards[_F_Y_];
      move1 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_x + move1 * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x,y,z,t;y;dag
      tmp_U = (origin_U + (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      for (int c0 = 0; c0 < _LAT_C_; c0++)
      {
        for (int c1 = 0; c1 < _LAT_C_; c1++)
        {
          clover[_LAT_C_ + c0 * _LAT_SC_ + c1] +=
              (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_minus_i();
          clover[36 + c0 * _LAT_SC_ + c1] +=
              (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_minus_i();
          clover[81 + c0 * _LAT_SC_ + c1] +=
              (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_minus_i();
          clover[114 + c0 * _LAT_SC_ + c1] +=
              (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_minus_i();
        }
      }
    }
    // YT
    give_vals(U, zero, _LAT_CC_);
    {
      //// x,y,z,t;y
      tmp_U = (origin_U + (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x,y+1,z,t;t
      move0 = move_wards[_F_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x,y,z,t+1;y;dag
      move0 = move_wards[_F_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x,y,z,t;t;dag
      tmp_U = (origin_U + (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      //// x,y,z,t;t
      tmp_U = (origin_U + (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x,y-1,z,t+1;y;dag
      move0 = move_wards[_B_Y_];
      move1 = move_wards[_F_T_];
      tmp_U = (origin_U + move0 * lat_x + move1 * lat_z * lat_y * lat_x +
               (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x,y-1,z,t;t;dag
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x,y-1,z,t;y
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      //// x,y-1,z,t;y;dag
      move0 = move_wards[_B_Y_];
      tmp_U = (origin_U + move0 * lat_x +
               (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x,y-1,z,t-1;t;dag
      move0 = move_wards[_B_Y_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_x + move1 * lat_z * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x,y-1,z,t-1;y
      move0 = move_wards[_B_Y_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_x + move1 * lat_z * lat_y * lat_x +
               (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x,y,z,t-1;t
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      //// x,y,z,t-1;t;dag
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x,y,z,t-1;y
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_Y_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x,y+1,z,t-1;t
      move0 = move_wards[_F_Y_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_x + move1 * lat_z * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x,y,z,t;y;dag
      tmp_U = (origin_U + (_Y_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      for (int c0 = 0; c0 < _LAT_C_; c0++)
      {
        for (int c1 = 0; c1 < _LAT_C_; c1++)
        {
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
    give_vals(U, zero, _LAT_CC_);
    {
      //// x,y,z,t;z
      tmp_U = (origin_U + (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x,y,z+1,t;t
      move0 = move_wards[_F_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x,y,z,t+1;z;dag
      move0 = move_wards[_F_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x,y,z,t;t;dag
      tmp_U = (origin_U + (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      //// x,y,z,t;t
      tmp_U = (origin_U + (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x,y,z-1,t+1;z;dag
      move0 = move_wards[_B_Z_];
      move1 = move_wards[_F_T_];
      tmp_U = (origin_U + move0 * lat_y * lat_x + move1 * lat_z * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x,y,z-1,t;t;dag
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x,y,z-1,t;z
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      //// x,y,z-1,t;z;dag
      move0 = move_wards[_B_Z_];
      tmp_U = (origin_U + move0 * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x,y,z-1,t-1;t;dag
      move0 = move_wards[_B_Z_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_y * lat_x + move1 * lat_z * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_dag_dag(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x,y,z-1,t-1;z
      move0 = move_wards[_B_Z_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_y * lat_x + move1 * lat_z * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x,y,z,t-1;t
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      //// x,y,z,t-1;t;dag
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      //// x,y,z,t-1;z
      move0 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_z * lat_y * lat_x +
               (_Z_ * _EVEN_ODD_ + (1 - parity)) * lat_tzyx);
      give_u(tmp2, tmp_U, lat_tzyx);
      mult_u_dag_none(tmp0, tmp1, tmp2, tmp3, zero);
    }
    {
      //// x,y,z+1,t-1;t
      move0 = move_wards[_F_Z_];
      move1 = move_wards[_B_T_];
      tmp_U = (origin_U + move0 * lat_y * lat_x + move1 * lat_z * lat_y * lat_x +
               (_T_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_none(tmp0, tmp3, tmp1, tmp2, zero);
    }
    {
      //// x,y,z,t;z;dag
      tmp_U = (origin_U + (_Z_ * _EVEN_ODD_ + parity) * lat_tzyx);
      give_u(tmp1, tmp_U, lat_tzyx);
      mult_u_none_dag(tmp0, tmp2, tmp1, tmp3, zero);
    }
    add_vals(U, tmp3, _LAT_CC_);
    {
      for (int c0 = 0; c0 < _LAT_C_; c0++)
      {
        for (int c1 = 0; c1 < _LAT_C_; c1++)
        {
          clover[c0 * _LAT_SC_ + c1] +=
              (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_i();
          clover[39 + c0 * _LAT_SC_ + c1] +=
              (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_minus_i();
          clover[78 + c0 * _LAT_SC_ + c1] +=
              (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_minus_i();
          clover[117 + c0 * _LAT_SC_ + c1] +=
              (U[c0 * _LAT_C_ + c1] - U[c1 * _LAT_C_ + c0].conj()).multi_i();
        }
      }
    }
    {
      // A=1+T(or A=1-kappa*T)
      LatticeComplex<T> one(1.0, 0);
      for (int i = 0; i < _LAT_SCSC_; i++)
      {
        clover[i] *= -kappa * 0.125; //-kappa*(1/8)
      }
      for (int i = 0; i < _LAT_SC_; i++)
      {
        clover[i * 13] += one;
      }
    }
    give_clr(origin_clover, clover, lat_tzyx);
  }
  template <typename T>
  __global__ void inverse_clover(void *device_clover, void *device_params)
  {
    LatticeComplex<T> *origin_clover;
    int lat_tzyx = static_cast<int *>(device_params)[_LAT_XYZT_];
    {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      origin_clover = ((static_cast<LatticeComplex<T> *>(device_clover)) + idx);
    }
    {
      LatticeComplex<T> pivot;
      LatticeComplex<T> factor;
      LatticeComplex<T> clover[_LAT_SCSC_];
      LatticeComplex<T> augmented_clover[_LAT_SCSC_ * _BF_];
      get_clr(clover, origin_clover, lat_tzyx);
      _inverse(clover, clover, augmented_clover, pivot, factor, _LAT_SC_);
      give_clr(origin_clover, clover, lat_tzyx);
    }
  }
  template <typename T>
  __global__ void give_clover(void *device_clover, void *device_dest,
                              void *device_params)
  {
    LatticeComplex<T> *origin_clover;
    LatticeComplex<T> *origin_dest;
    int lat_tzyx = static_cast<int *>(device_params)[_LAT_XYZT_];
    {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      origin_clover = ((static_cast<LatticeComplex<T> *>(device_clover)) + idx);
      origin_dest = ((static_cast<LatticeComplex<T> *>(device_dest)) + idx);
    }
    {
      LatticeComplex<T> clover[_LAT_SCSC_];
      LatticeComplex<T> dest[_LAT_SC_];
      LatticeComplex<T> tmp_dest[_LAT_SC_];
      LatticeComplex<T> zero(0.0, 0.0);
      give_vals(tmp_dest, zero, _LAT_SC_);
      get_src(dest, origin_dest, lat_tzyx);
      get_clr(clover, origin_clover, lat_tzyx);
      for (int sc0 = 0; sc0 < _LAT_SC_; sc0++)
      {
        for (int sc1 = 0; sc1 < _LAT_SC_; sc1++)
        {
          tmp_dest[sc0] += clover[sc0 * _LAT_SC_ + sc1] * dest[sc1];
        }
      }
      give_dest(origin_dest, tmp_dest, lat_tzyx);
    }
  }
  //@@@CUDA_TEMPLATE_FOR_DEVICE@@@
  template __global__ void make_clover<double>(void *device_U, void *device_clover,
                                               void *device_params, double T);
  template __global__ void inverse_clover<double>(void *device_clover, void *device_params);
  template __global__ void give_clover<double>(void *device_clover, void *device_dest,
                                               void *device_params);
  //@@@CUDA_TEMPLATE_FOR_DEVICE@@@
  template __global__ void make_clover<float>(void *device_U, void *device_clover,
                                              void *device_params, float T);
  template __global__ void inverse_clover<float>(void *device_clover, void *device_params);
  template __global__ void give_clover<float>(void *device_clover, void *device_dest,
                                              void *device_params);
}