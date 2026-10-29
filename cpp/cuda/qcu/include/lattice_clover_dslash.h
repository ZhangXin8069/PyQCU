#ifndef _LATTICE_CLOVER_DSLASH_H
#define _LATTICE_CLOVER_DSLASH_H
#include "./clover_dslash.h"
#include "./lattice_mpi.h"
#include "./lattice_set.h"
#include "define.h"
namespace qcu {
template <typename T> struct LatticeCloverDslash {
  LatticeSet<T> *set_ptr;
  cudaError_t err;
  void *clover;
  bool if_input;
  bool block_diagonal_ = false;
  void give(LatticeSet<T> *_set_ptr) { set_ptr = _set_ptr; }
  void init() {
    if_input = false;
    checkCudaErrors(cudaMallocAsync(
        &clover, (set_ptr->lat_4dim * _LAT_SCSC_) * sizeof(LatticeComplex<T>),
        set_ptr->stream));
  }
  void init(void *_clover) {
    if_input = true;
    clover = (void *)_clover;
  }
  void _make_mpi(void *gauge) {
    if (_CLOVER_TEST_MULTI_IN_SINGLE_) {
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
      make_clover<T>
          <<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
              gauge, clover, set_ptr->device_params,
              set_ptr->clover_coefficient());
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
    } else {
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
      // edge send part
      {
        // u_1dim_send
        pick_up_u_x<T><<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim, 0,
                         set_ptr->stream_dims[_X_]>>>(
            gauge, set_ptr->device_params,
            set_ptr->device_u_1dim_send_vec[_B_X_],
            set_ptr->device_u_1dim_send_vec[_F_X_]);
        checkCudaErrors(cudaGetLastError());
        pick_up_u_y<T><<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim, 0,
                         set_ptr->stream_dims[_Y_]>>>(
            gauge, set_ptr->device_params,
            set_ptr->device_u_1dim_send_vec[_B_Y_],
            set_ptr->device_u_1dim_send_vec[_F_Y_]);
        checkCudaErrors(cudaGetLastError());
        pick_up_u_z<T><<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim, 0,
                         set_ptr->stream_dims[_Z_]>>>(
            gauge, set_ptr->device_params,
            set_ptr->device_u_1dim_send_vec[_B_Z_],
            set_ptr->device_u_1dim_send_vec[_F_Z_]);
        checkCudaErrors(cudaGetLastError());
        pick_up_u_t<T><<<set_ptr->gridDim_3dim[_T_], set_ptr->blockDim, 0,
                         set_ptr->stream_dims[_T_]>>>(
            gauge, set_ptr->device_params,
            set_ptr->device_u_1dim_send_vec[_B_T_],
            set_ptr->device_u_1dim_send_vec[_F_T_]);
        checkCudaErrors(cudaGetLastError());
      }
      {
        // u_2dim_send
        pick_up_u_xy<T>
            <<<set_ptr->gridDim_2dim[_2DIM_ - 1 - _XY_], set_ptr->blockDim, 0,
               set_ptr->stream>>>(gauge, set_ptr->device_params,
                                  set_ptr->device_u_2dim_send_vec[_B_X_B_Y_],
                                  set_ptr->device_u_2dim_send_vec[_F_X_B_Y_],
                                  set_ptr->device_u_2dim_send_vec[_B_X_F_Y_],
                                  set_ptr->device_u_2dim_send_vec[_F_X_F_Y_]);
        checkCudaErrors(cudaGetLastError());
        pick_up_u_xz<T>
            <<<set_ptr->gridDim_2dim[_2DIM_ - 1 - _XZ_], set_ptr->blockDim, 0,
               set_ptr->stream>>>(gauge, set_ptr->device_params,
                                  set_ptr->device_u_2dim_send_vec[_B_X_B_Z_],
                                  set_ptr->device_u_2dim_send_vec[_F_X_B_Z_],
                                  set_ptr->device_u_2dim_send_vec[_B_X_F_Z_],
                                  set_ptr->device_u_2dim_send_vec[_F_X_F_Z_]);
        checkCudaErrors(cudaGetLastError());
        pick_up_u_xt<T>
            <<<set_ptr->gridDim_2dim[_2DIM_ - 1 - _XT_], set_ptr->blockDim, 0,
               set_ptr->stream>>>(gauge, set_ptr->device_params,
                                  set_ptr->device_u_2dim_send_vec[_B_X_B_T_],
                                  set_ptr->device_u_2dim_send_vec[_F_X_B_T_],
                                  set_ptr->device_u_2dim_send_vec[_B_X_F_T_],
                                  set_ptr->device_u_2dim_send_vec[_F_X_F_T_]);
        checkCudaErrors(cudaGetLastError());
        pick_up_u_yz<T>
            <<<set_ptr->gridDim_2dim[_2DIM_ - 1 - _YZ_], set_ptr->blockDim, 0,
               set_ptr->stream>>>(gauge, set_ptr->device_params,
                                  set_ptr->device_u_2dim_send_vec[_B_Y_B_Z_],
                                  set_ptr->device_u_2dim_send_vec[_F_Y_B_Z_],
                                  set_ptr->device_u_2dim_send_vec[_B_Y_F_Z_],
                                  set_ptr->device_u_2dim_send_vec[_F_Y_F_Z_]);
        checkCudaErrors(cudaGetLastError());
        pick_up_u_yt<T>
            <<<set_ptr->gridDim_2dim[_2DIM_ - 1 - _YT_], set_ptr->blockDim, 0,
               set_ptr->stream>>>(gauge, set_ptr->device_params,
                                  set_ptr->device_u_2dim_send_vec[_B_Y_B_T_],
                                  set_ptr->device_u_2dim_send_vec[_F_Y_B_T_],
                                  set_ptr->device_u_2dim_send_vec[_B_Y_F_T_],
                                  set_ptr->device_u_2dim_send_vec[_F_Y_F_T_]);
        checkCudaErrors(cudaGetLastError());
        pick_up_u_zt<T>
            <<<set_ptr->gridDim_2dim[_2DIM_ - 1 - _ZT_], set_ptr->blockDim, 0,
               set_ptr->stream>>>(gauge, set_ptr->device_params,
                                  set_ptr->device_u_2dim_send_vec[_B_Z_B_T_],
                                  set_ptr->device_u_2dim_send_vec[_F_Z_B_T_],
                                  set_ptr->device_u_2dim_send_vec[_B_Z_F_T_],
                                  set_ptr->device_u_2dim_send_vec[_F_Z_F_T_]);
        checkCudaErrors(cudaGetLastError());
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
      // edge comm part
      {
        { // d2h
          for (int i = 0; i < _DIM_; i++) {
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->host_u_1dim_send_vec[i * _BF_],
                set_ptr->device_u_1dim_send_vec[i * _BF_],
                set_ptr->lat_3dim[i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>),
                cudaMemcpyDeviceToHost, set_ptr->stream));
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->host_u_1dim_send_vec[i * _BF_ + 1],
                set_ptr->device_u_1dim_send_vec[i * _BF_ + 1],
                set_ptr->lat_3dim[i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>),
                cudaMemcpyDeviceToHost, set_ptr->stream));
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->host_u_1dim_recv_vec[i * _BF_],
                set_ptr->device_u_1dim_recv_vec[i * _BF_],
                set_ptr->lat_3dim[i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>),
                cudaMemcpyDeviceToHost, set_ptr->stream));
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->host_u_1dim_recv_vec[i * _BF_ + 1],
                set_ptr->device_u_1dim_recv_vec[i * _BF_ + 1],
                set_ptr->lat_3dim[i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>),
                cudaMemcpyDeviceToHost, set_ptr->stream));
          }
          for (int i = 0; i < _2DIM_; i++) {
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->host_u_2dim_send_vec[i * _BF_ * _BF_ + 0],
                set_ptr->device_u_2dim_send_vec[i * _BF_ * _BF_ + 0],
                set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                    sizeof(LatticeComplex<T>),
                cudaMemcpyDeviceToHost, set_ptr->stream));
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->host_u_2dim_recv_vec[i * _BF_ * _BF_ + 0],
                set_ptr->device_u_2dim_recv_vec[i * _BF_ * _BF_ + 0],
                set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                    sizeof(LatticeComplex<T>),
                cudaMemcpyDeviceToHost, set_ptr->stream));
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->host_u_2dim_send_vec[i * _BF_ * _BF_ + 1],
                set_ptr->device_u_2dim_send_vec[i * _BF_ * _BF_ + 1],
                set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                    sizeof(LatticeComplex<T>),
                cudaMemcpyDeviceToHost, set_ptr->stream));
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->host_u_2dim_recv_vec[i * _BF_ * _BF_ + 1],
                set_ptr->device_u_2dim_recv_vec[i * _BF_ * _BF_ + 1],
                set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                    sizeof(LatticeComplex<T>),
                cudaMemcpyDeviceToHost, set_ptr->stream));
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->host_u_2dim_send_vec[i * _BF_ * _BF_ + 2],
                set_ptr->device_u_2dim_send_vec[i * _BF_ * _BF_ + 2],
                set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                    sizeof(LatticeComplex<T>),
                cudaMemcpyDeviceToHost, set_ptr->stream));
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->host_u_2dim_recv_vec[i * _BF_ * _BF_ + 2],
                set_ptr->device_u_2dim_recv_vec[i * _BF_ * _BF_ + 2],
                set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                    sizeof(LatticeComplex<T>),
                cudaMemcpyDeviceToHost, set_ptr->stream));
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->host_u_2dim_send_vec[i * _BF_ * _BF_ + 3],
                set_ptr->device_u_2dim_send_vec[i * _BF_ * _BF_ + 3],
                set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                    sizeof(LatticeComplex<T>),
                cudaMemcpyDeviceToHost, set_ptr->stream));
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->host_u_2dim_recv_vec[i * _BF_ * _BF_ + 3],
                set_ptr->device_u_2dim_recv_vec[i * _BF_ * _BF_ + 3],
                set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                    sizeof(LatticeComplex<T>),
                cudaMemcpyDeviceToHost, set_ptr->stream));
          }
        }
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
      {
        {
          // x edge part comm
          _MPI_Sendrecv<T>(set_ptr->host_u_1dim_send_vec[_B_X_],
                           set_ptr->lat_3dim[_X_] * _LAT_PCCD_ * _REAL_IMAG_,
                           set_ptr->move_wards[_B_X_], _B_X_,
                           set_ptr->host_u_1dim_recv_vec[_F_X_],
                           set_ptr->lat_3dim[_X_] * _LAT_PCCD_ * _REAL_IMAG_,
                           set_ptr->move_wards[_F_X_], _B_X_, set_ptr->comm,
                           MPI_STATUS_IGNORE);
          _MPI_Sendrecv<T>(set_ptr->host_u_1dim_send_vec[_F_X_],
                           set_ptr->lat_3dim[_X_] * _LAT_PCCD_ * _REAL_IMAG_,
                           set_ptr->move_wards[_F_X_], _F_X_,
                           set_ptr->host_u_1dim_recv_vec[_B_X_],
                           set_ptr->lat_3dim[_X_] * _LAT_PCCD_ * _REAL_IMAG_,
                           set_ptr->move_wards[_B_X_], _F_X_, set_ptr->comm,
                           MPI_STATUS_IGNORE);
        }
        {
          //   y edge part comm
          _MPI_Sendrecv<T>(set_ptr->host_u_1dim_send_vec[_B_Y_],
                           set_ptr->lat_3dim[_Y_] * _LAT_PCCD_ * _REAL_IMAG_,
                           set_ptr->move_wards[_B_Y_], _B_Y_,
                           set_ptr->host_u_1dim_recv_vec[_F_Y_],
                           set_ptr->lat_3dim[_Y_] * _LAT_PCCD_ * _REAL_IMAG_,
                           set_ptr->move_wards[_F_Y_], _B_Y_, set_ptr->comm,
                           MPI_STATUS_IGNORE);
          _MPI_Sendrecv<T>(set_ptr->host_u_1dim_send_vec[_F_Y_],
                           set_ptr->lat_3dim[_Y_] * _LAT_PCCD_ * _REAL_IMAG_,
                           set_ptr->move_wards[_F_Y_], _F_Y_,
                           set_ptr->host_u_1dim_recv_vec[_B_Y_],
                           set_ptr->lat_3dim[_Y_] * _LAT_PCCD_ * _REAL_IMAG_,
                           set_ptr->move_wards[_B_Y_], _F_Y_, set_ptr->comm,
                           MPI_STATUS_IGNORE);
        }
        {
          // z edge part comm
          _MPI_Sendrecv<T>(set_ptr->host_u_1dim_send_vec[_B_Z_],
                           set_ptr->lat_3dim[_Z_] * _LAT_PCCD_ * _REAL_IMAG_,
                           set_ptr->move_wards[_B_Z_], _B_Z_,
                           set_ptr->host_u_1dim_recv_vec[_F_Z_],
                           set_ptr->lat_3dim[_Z_] * _LAT_PCCD_ * _REAL_IMAG_,
                           set_ptr->move_wards[_F_Z_], _B_Z_, set_ptr->comm,
                           MPI_STATUS_IGNORE);
          _MPI_Sendrecv<T>(set_ptr->host_u_1dim_send_vec[_F_Z_],
                           set_ptr->lat_3dim[_Z_] * _LAT_PCCD_ * _REAL_IMAG_,
                           set_ptr->move_wards[_F_Z_], _F_Z_,
                           set_ptr->host_u_1dim_recv_vec[_B_Z_],
                           set_ptr->lat_3dim[_Z_] * _LAT_PCCD_ * _REAL_IMAG_,
                           set_ptr->move_wards[_B_Z_], _F_Z_, set_ptr->comm,
                           MPI_STATUS_IGNORE);
        }
        {
          // t edge part comm
          _MPI_Sendrecv<T>(set_ptr->host_u_1dim_send_vec[_B_T_],
                           set_ptr->lat_3dim[_T_] * _LAT_PCCD_ * _REAL_IMAG_,
                           set_ptr->move_wards[_B_T_], _B_T_,
                           set_ptr->host_u_1dim_recv_vec[_F_T_],
                           set_ptr->lat_3dim[_T_] * _LAT_PCCD_ * _REAL_IMAG_,
                           set_ptr->move_wards[_F_T_], _B_T_, set_ptr->comm,
                           MPI_STATUS_IGNORE);
          _MPI_Sendrecv<T>(set_ptr->host_u_1dim_send_vec[_F_T_],
                           set_ptr->lat_3dim[_T_] * _LAT_PCCD_ * _REAL_IMAG_,
                           set_ptr->move_wards[_F_T_], _F_T_,
                           set_ptr->host_u_1dim_recv_vec[_B_T_],
                           set_ptr->lat_3dim[_T_] * _LAT_PCCD_ * _REAL_IMAG_,
                           set_ptr->move_wards[_B_T_], _F_T_, set_ptr->comm,
                           MPI_STATUS_IGNORE);
        }
        {
          // u_2dim_comm
          {
            // xy edge part comm
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_B_X_B_Y_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BX_BY_], _B_X_B_Y_,
                set_ptr->host_u_2dim_recv_vec[_F_X_F_Y_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FX_FY_], _B_X_B_Y_, set_ptr->comm,
                MPI_STATUS_IGNORE);
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_F_X_B_Y_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FX_BY_], _F_X_B_Y_,
                set_ptr->host_u_2dim_recv_vec[_B_X_F_Y_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BX_FY_], _F_X_B_Y_, set_ptr->comm,
                MPI_STATUS_IGNORE);
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_B_X_F_Y_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BX_FY_], _B_X_F_Y_,
                set_ptr->host_u_2dim_recv_vec[_F_X_B_Y_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FX_BY_], _B_X_F_Y_, set_ptr->comm,
                MPI_STATUS_IGNORE);
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_F_X_F_Y_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FX_FY_], _F_X_F_Y_,
                set_ptr->host_u_2dim_recv_vec[_B_X_B_Y_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XY_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BX_BY_], _F_X_F_Y_, set_ptr->comm,
                MPI_STATUS_IGNORE);
          }
          {
            // xz edge part comm
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_B_X_B_Z_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BX_BZ_], _B_X_B_Z_,
                set_ptr->host_u_2dim_recv_vec[_F_X_F_Z_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FX_FZ_], _B_X_B_Z_, set_ptr->comm,
                MPI_STATUS_IGNORE);
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_F_X_B_Z_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FX_BZ_], _F_X_B_Z_,
                set_ptr->host_u_2dim_recv_vec[_B_X_F_Z_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BX_FZ_], _F_X_B_Z_, set_ptr->comm,
                MPI_STATUS_IGNORE);
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_B_X_F_Z_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BX_FZ_], _B_X_F_Z_,
                set_ptr->host_u_2dim_recv_vec[_F_X_B_Z_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FX_BZ_], _B_X_F_Z_, set_ptr->comm,
                MPI_STATUS_IGNORE);
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_F_X_F_Z_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FX_FZ_], _F_X_F_Z_,
                set_ptr->host_u_2dim_recv_vec[_B_X_B_Z_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XZ_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BX_BZ_], _F_X_F_Z_, set_ptr->comm,
                MPI_STATUS_IGNORE);
          }
          {
            // xt edge part comm
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_B_X_B_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BX_BT_], _B_X_B_T_,
                set_ptr->host_u_2dim_recv_vec[_F_X_F_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FX_FT_], _B_X_B_T_, set_ptr->comm,
                MPI_STATUS_IGNORE);
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_F_X_B_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FX_BT_], _F_X_B_T_,
                set_ptr->host_u_2dim_recv_vec[_B_X_F_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BX_FT_], _F_X_B_T_, set_ptr->comm,
                MPI_STATUS_IGNORE);
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_B_X_F_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BX_FT_], _B_X_F_T_,
                set_ptr->host_u_2dim_recv_vec[_F_X_B_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FX_BT_], _B_X_F_T_, set_ptr->comm,
                MPI_STATUS_IGNORE);
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_F_X_F_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FX_FT_], _F_X_F_T_,
                set_ptr->host_u_2dim_recv_vec[_B_X_B_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _XT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BX_BT_], _F_X_F_T_, set_ptr->comm,
                MPI_STATUS_IGNORE);
          }
          {
            // yz edge part comm
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_B_Y_B_Z_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BY_BZ_], _B_Y_B_Z_,
                set_ptr->host_u_2dim_recv_vec[_F_Y_F_Z_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FY_FZ_], _B_Y_B_Z_, set_ptr->comm,
                MPI_STATUS_IGNORE);
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_F_Y_B_Z_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FY_BZ_], _F_Y_B_Z_,
                set_ptr->host_u_2dim_recv_vec[_B_Y_F_Z_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BY_FZ_], _F_Y_B_Z_, set_ptr->comm,
                MPI_STATUS_IGNORE);
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_B_Y_F_Z_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BY_FZ_], _B_Y_F_Z_,
                set_ptr->host_u_2dim_recv_vec[_F_Y_B_Z_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FY_BZ_], _B_Y_F_Z_, set_ptr->comm,
                MPI_STATUS_IGNORE);
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_F_Y_F_Z_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FY_FZ_], _F_Y_F_Z_,
                set_ptr->host_u_2dim_recv_vec[_B_Y_B_Z_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _YZ_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BY_BZ_], _F_Y_F_Z_, set_ptr->comm,
                MPI_STATUS_IGNORE);
          }
          {
            // yt edge part comm
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_B_Y_B_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BY_BT_], _B_Y_B_T_,
                set_ptr->host_u_2dim_recv_vec[_F_Y_F_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FY_FT_], _B_Y_B_T_, set_ptr->comm,
                MPI_STATUS_IGNORE);
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_F_Y_B_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FY_BT_], _F_Y_B_T_,
                set_ptr->host_u_2dim_recv_vec[_B_Y_F_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BY_FT_], _F_Y_B_T_, set_ptr->comm,
                MPI_STATUS_IGNORE);
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_B_Y_F_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BY_FT_], _B_Y_F_T_,
                set_ptr->host_u_2dim_recv_vec[_F_Y_B_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FY_BT_], _B_Y_F_T_, set_ptr->comm,
                MPI_STATUS_IGNORE);
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_F_Y_F_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FY_FT_], _F_Y_F_T_,
                set_ptr->host_u_2dim_recv_vec[_B_Y_B_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _YT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BY_BT_], _F_Y_F_T_, set_ptr->comm,
                MPI_STATUS_IGNORE);
          }
          {
            // zt edge part comm
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_B_Z_B_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BZ_BT_], _B_Z_B_T_,
                set_ptr->host_u_2dim_recv_vec[_F_Z_F_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FZ_FT_], _B_Z_B_T_, set_ptr->comm,
                MPI_STATUS_IGNORE);
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_F_Z_B_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FZ_BT_], _F_Z_B_T_,
                set_ptr->host_u_2dim_recv_vec[_B_Z_F_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BZ_FT_], _F_Z_B_T_, set_ptr->comm,
                MPI_STATUS_IGNORE);
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_B_Z_F_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BZ_FT_], _B_Z_F_T_,
                set_ptr->host_u_2dim_recv_vec[_F_Z_B_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FZ_BT_], _B_Z_F_T_, set_ptr->comm,
                MPI_STATUS_IGNORE);
            _MPI_Sendrecv<T>(
                set_ptr->host_u_2dim_send_vec[_F_Z_F_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_FZ_FT_], _F_Z_F_T_,
                set_ptr->host_u_2dim_recv_vec[_B_Z_B_T_],
                set_ptr->lat_2dim[_2DIM_ - 1 - _ZT_] * _LAT_PCCD_ * _REAL_IMAG_,
                set_ptr->move_wards[_BZ_BT_], _F_Z_F_T_, set_ptr->comm,
                MPI_STATUS_IGNORE);
          }
        }
      }
      {
        { // h2d
          for (int i = 0; i < _DIM_; i++) {
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->device_u_1dim_send_vec[i * _BF_],
                set_ptr->host_u_1dim_send_vec[i * _BF_],
                set_ptr->lat_3dim[i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>),
                cudaMemcpyHostToDevice, set_ptr->stream));
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->device_u_1dim_send_vec[i * _BF_ + 1],
                set_ptr->host_u_1dim_send_vec[i * _BF_ + 1],
                set_ptr->lat_3dim[i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>),
                cudaMemcpyHostToDevice, set_ptr->stream));
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->device_u_1dim_recv_vec[i * _BF_],
                set_ptr->host_u_1dim_recv_vec[i * _BF_],
                set_ptr->lat_3dim[i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>),
                cudaMemcpyHostToDevice, set_ptr->stream));
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->device_u_1dim_recv_vec[i * _BF_ + 1],
                set_ptr->host_u_1dim_recv_vec[i * _BF_ + 1],
                set_ptr->lat_3dim[i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>),
                cudaMemcpyHostToDevice, set_ptr->stream));
          }
          for (int i = 0; i < _2DIM_; i++) {
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->device_u_2dim_send_vec[i * _BF_ * _BF_ + 0],
                set_ptr->host_u_2dim_send_vec[i * _BF_ * _BF_ + 0],
                set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                    sizeof(LatticeComplex<T>),
                cudaMemcpyHostToDevice, set_ptr->stream));
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->device_u_2dim_recv_vec[i * _BF_ * _BF_ + 0],
                set_ptr->host_u_2dim_recv_vec[i * _BF_ * _BF_ + 0],
                set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                    sizeof(LatticeComplex<T>),
                cudaMemcpyHostToDevice, set_ptr->stream));
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->device_u_2dim_send_vec[i * _BF_ * _BF_ + 1],
                set_ptr->host_u_2dim_send_vec[i * _BF_ * _BF_ + 1],
                set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                    sizeof(LatticeComplex<T>),
                cudaMemcpyHostToDevice, set_ptr->stream));
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->device_u_2dim_recv_vec[i * _BF_ * _BF_ + 1],
                set_ptr->host_u_2dim_recv_vec[i * _BF_ * _BF_ + 1],
                set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                    sizeof(LatticeComplex<T>),
                cudaMemcpyHostToDevice, set_ptr->stream));
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->device_u_2dim_send_vec[i * _BF_ * _BF_ + 2],
                set_ptr->host_u_2dim_send_vec[i * _BF_ * _BF_ + 2],
                set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                    sizeof(LatticeComplex<T>),
                cudaMemcpyHostToDevice, set_ptr->stream));
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->device_u_2dim_recv_vec[i * _BF_ * _BF_ + 2],
                set_ptr->host_u_2dim_recv_vec[i * _BF_ * _BF_ + 2],
                set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                    sizeof(LatticeComplex<T>),
                cudaMemcpyHostToDevice, set_ptr->stream));
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->device_u_2dim_send_vec[i * _BF_ * _BF_ + 3],
                set_ptr->host_u_2dim_send_vec[i * _BF_ * _BF_ + 3],
                set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                    sizeof(LatticeComplex<T>),
                cudaMemcpyHostToDevice, set_ptr->stream));
            checkCudaErrors(cudaMemcpyAsync(
                set_ptr->device_u_2dim_recv_vec[i * _BF_ * _BF_ + 3],
                set_ptr->host_u_2dim_recv_vec[i * _BF_ * _BF_ + 3],
                set_ptr->lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                    sizeof(LatticeComplex<T>),
                cudaMemcpyHostToDevice, set_ptr->stream));
          }
        }
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
      // edge recv part
      make_clover_all<T>
          <<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
              gauge, clover, set_ptr->device_params,
              set_ptr->clover_coefficient(),
              set_ptr->device_u_1dim_recv_vec[_B_X_],
              set_ptr->device_u_1dim_recv_vec[_F_X_],
              set_ptr->device_u_1dim_recv_vec[_B_Y_],
              set_ptr->device_u_1dim_recv_vec[_F_Y_],
              set_ptr->device_u_1dim_recv_vec[_B_Z_],
              set_ptr->device_u_1dim_recv_vec[_F_Z_],
              set_ptr->device_u_1dim_recv_vec[_B_T_],
              set_ptr->device_u_1dim_recv_vec[_F_T_],
              set_ptr->device_u_2dim_recv_vec[_B_X_B_Y_],
              set_ptr->device_u_2dim_recv_vec[_F_X_B_Y_],
              set_ptr->device_u_2dim_recv_vec[_B_X_F_Y_],
              set_ptr->device_u_2dim_recv_vec[_F_X_F_Y_],
              set_ptr->device_u_2dim_recv_vec[_B_X_B_Z_],
              set_ptr->device_u_2dim_recv_vec[_F_X_B_Z_],
              set_ptr->device_u_2dim_recv_vec[_B_X_F_Z_],
              set_ptr->device_u_2dim_recv_vec[_F_X_F_Z_],
              set_ptr->device_u_2dim_recv_vec[_B_X_B_T_],
              set_ptr->device_u_2dim_recv_vec[_F_X_B_T_],
              set_ptr->device_u_2dim_recv_vec[_B_X_F_T_],
              set_ptr->device_u_2dim_recv_vec[_F_X_F_T_],
              set_ptr->device_u_2dim_recv_vec[_B_Y_B_Z_],
              set_ptr->device_u_2dim_recv_vec[_F_Y_B_Z_],
              set_ptr->device_u_2dim_recv_vec[_B_Y_F_Z_],
              set_ptr->device_u_2dim_recv_vec[_F_Y_F_Z_],
              set_ptr->device_u_2dim_recv_vec[_B_Y_B_T_],
              set_ptr->device_u_2dim_recv_vec[_F_Y_B_T_],
              set_ptr->device_u_2dim_recv_vec[_B_Y_F_T_],
              set_ptr->device_u_2dim_recv_vec[_F_Y_F_T_],
              set_ptr->device_u_2dim_recv_vec[_B_Z_B_T_],
              set_ptr->device_u_2dim_recv_vec[_F_Z_B_T_],
              set_ptr->device_u_2dim_recv_vec[_B_Z_F_T_],
              set_ptr->device_u_2dim_recv_vec[_F_Z_F_T_]);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
    }
  }
  void make(void *gauge) {
    // make clover
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    _make_mpi(gauge);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    err = cudaGetLastError();
    checkCudaErrors(err);
  }
  void make_test(void *gauge) {
    // make clover
    auto start = std::chrono::high_resolution_clock::now();
    make(gauge);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf("make clover total time: (without malloc free memcpy) :%.9lf sec\n ",
           double(duration) / 1e9);
  }
  void inverse() {
    // inverse clover
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    inverse_clover<T>
        <<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
            clover, set_ptr->device_params);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    err = cudaGetLastError();
    checkCudaErrors(err);
  }
  void inverse_test() {
    // inverse clover
    auto start = std::chrono::high_resolution_clock::now();
    inverse();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf("inverse clover total time: (without malloc free memcpy) :%.9lf "
           "sec\n ",
           double(duration) / 1e9);
  }
  void give(void *fermion_out) {
    // give clover
    // NOTE (2026-08-02): on a 1x1x1x1 process grid the main stream already
    // serializes give_clover against the producer/consumer kernels, so the two
    // cudaStreamSynchronize calls are pure overhead (~170 us each on this
    // WSL2/V100 box).  Multi-rank runs still sync (they may use other streams).
    const bool single_rank =
        set_ptr->host_params[_GRID_X_] == 1 &&
        set_ptr->host_params[_GRID_Y_] == 1 &&
        set_ptr->host_params[_GRID_Z_] == 1 &&
        set_ptr->host_params[_GRID_T_] == 1;
    if (!single_rank) checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    if (block_diagonal_) {
      give_clover_block<T>
          <<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
              clover, fermion_out, set_ptr->device_params);
    } else {
      give_clover<T>
          <<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
              clover, fermion_out, set_ptr->device_params);
    }
    if (!single_rank) checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    err = cudaGetLastError();
    checkCudaErrors(err);
  }
  void give(const void *fermion_in, void *fermion_out) {
    const bool single_rank =
        set_ptr->host_params[_GRID_X_] == 1 &&
        set_ptr->host_params[_GRID_Y_] == 1 &&
        set_ptr->host_params[_GRID_Z_] == 1 &&
        set_ptr->host_params[_GRID_T_] == 1;
    if (!single_rank) checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    if (block_diagonal_) {
      give_clover_to_block<T>
          <<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
              clover, fermion_in, fermion_out, set_ptr->device_params);
    } else {
      give_clover_to<T>
          <<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
              clover, fermion_in, fermion_out, set_ptr->device_params);
    }
    if (!single_rank) checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    err = cudaGetLastError();
    checkCudaErrors(err);
  }
  void give_with_subtract(const void *fermion_in, const void *subtract,
                          void *fermion_out, T subtract_scale) {
    const bool single_rank =
        set_ptr->host_params[_GRID_X_] == 1 &&
        set_ptr->host_params[_GRID_Y_] == 1 &&
        set_ptr->host_params[_GRID_Z_] == 1 &&
        set_ptr->host_params[_GRID_T_] == 1;
    if (!single_rank) checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    if (block_diagonal_) {
      give_clover_subtract_block<T>
          <<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
              clover, fermion_in, subtract, fermion_out, subtract_scale,
              set_ptr->device_params);
    } else {
      give_clover_subtract<T>
          <<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
              clover, fermion_in, subtract, fermion_out, subtract_scale,
              set_ptr->device_params);
    }
    if (!single_rank) checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    err = cudaGetLastError();
    checkCudaErrors(err);
  }
  void give_test(void *fermion_out) {
    // give clover
    auto start = std::chrono::high_resolution_clock::now();
    give(fermion_out);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf("give clover total time: (without malloc free memcpy) :%.9lf sec\n ",
           double(duration) / 1e9);
  }
  void end() {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    if (if_input == false) {
      checkCudaErrors(cudaFreeAsync(clover, set_ptr->stream));
    }
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
};

template <typename T>
bool clover_schur_matrices_block_diagonal(void *clover_ee_inv,
                                          void *clover_oo,
                                          LatticeSet<T> *set_ptr) {
  if (clover_ee_inv == nullptr || clover_oo == nullptr) return false;
  int *layout_flag = nullptr;
  checkCudaErrors(cudaMallocAsync(&layout_flag, sizeof(int), set_ptr->stream));
  checkCudaErrors(cudaMemsetAsync(layout_flag, 0, sizeof(int), set_ptr->stream));
  check_clover_block_diagonal<T>
      <<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
          clover_ee_inv, set_ptr->device_params, layout_flag);
  check_clover_block_diagonal<T>
      <<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
          clover_oo, set_ptr->device_params, layout_flag);
  checkCudaErrors(cudaGetLastError());
  int host_layout_flag = 1;
  checkCudaErrors(cudaMemcpyAsync(
      &host_layout_flag, layout_flag, sizeof(int), cudaMemcpyDeviceToHost,
      set_ptr->stream));
  checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  checkCudaErrors(cudaFreeAsync(layout_flag, set_ptr->stream));
  return host_layout_flag == 0;
}
} // namespace qcu
#endif
