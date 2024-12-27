#ifndef _LATTICE_WILSON_DSLASH_H
#define _LATTICE_WILSON_DSLASH_H
#include "./lattice_set.h"
#include "./lattice_mpi.h"
#include "./wilson_dslash.h"
namespace qcu
{
  template <typename T>
  struct LatticeWilsonDslash
  {
    LatticeSet<T> *set_ptr;
    cudaError_t err;
    void give(LatticeSet<T> *_set_ptr)
    {
      set_ptr = _set_ptr;
    }
    void run_mpi_non_block(void *fermion_out, void *fermion_in, void *gauge,
                           void *_device_params)
    {
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
      { // edge send part
        wilson_dslash_x_send<T><<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim, 0,
                                  set_ptr->stream_dims[_X_]>>>(
            gauge, fermion_in, _device_params, set_ptr->device_send_vec[_B_X_],
            set_ptr->device_send_vec[_F_X_]);
        if (set_ptr->host_params[_GRID_X_] != 1)
        { // x part d2h
          checkCudaErrors(cudaMemcpyAsync(
              set_ptr->host_send_vec[_B_X_], set_ptr->device_send_vec[_B_X_],
              sizeof(T) * set_ptr->lat_3dim_SC[_X_] / _EVEN_ODD_, cudaMemcpyDeviceToHost,
              set_ptr->stream_dims[_X_]));
          checkCudaErrors(cudaMemcpyAsync(
              set_ptr->host_send_vec[_F_X_], set_ptr->device_send_vec[_F_X_],
              sizeof(T) * set_ptr->lat_3dim_SC[_X_] / _EVEN_ODD_, cudaMemcpyDeviceToHost,
              set_ptr->stream_dims[_X_]));
        }
        wilson_dslash_y_send<T><<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim, 0,
                                  set_ptr->stream_dims[_Y_]>>>(
            gauge, fermion_in, _device_params, set_ptr->device_send_vec[_B_Y_],
            set_ptr->device_send_vec[_F_Y_]);
        if (set_ptr->host_params[_GRID_Y_] != 1)
        { // y part d2h
          checkCudaErrors(cudaMemcpyAsync(
              set_ptr->host_send_vec[_B_Y_], set_ptr->device_send_vec[_B_Y_],
              sizeof(T) * set_ptr->lat_3dim_SC[_Y_], cudaMemcpyDeviceToHost,
              set_ptr->stream_dims[_Y_]));
          checkCudaErrors(cudaMemcpyAsync(
              set_ptr->host_send_vec[_F_Y_], set_ptr->device_send_vec[_F_Y_],
              sizeof(T) * set_ptr->lat_3dim_SC[_Y_], cudaMemcpyDeviceToHost,
              set_ptr->stream_dims[_Y_]));
        }
        wilson_dslash_z_send<T><<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim, 0,
                                  set_ptr->stream_dims[_Z_]>>>(
            gauge, fermion_in, _device_params, set_ptr->device_send_vec[_B_Z_],
            set_ptr->device_send_vec[_F_Z_]);
        if (set_ptr->host_params[_GRID_Z_] != 1)
        { // z part d2h
          checkCudaErrors(cudaMemcpyAsync(
              set_ptr->host_send_vec[_B_Z_], set_ptr->device_send_vec[_B_Z_],
              sizeof(T) * set_ptr->lat_3dim_SC[_Z_], cudaMemcpyDeviceToHost,
              set_ptr->stream_dims[_Z_]));
          checkCudaErrors(cudaMemcpyAsync(
              set_ptr->host_send_vec[_F_Z_], set_ptr->device_send_vec[_F_Z_],
              sizeof(T) * set_ptr->lat_3dim_SC[_Z_], cudaMemcpyDeviceToHost,
              set_ptr->stream_dims[_Z_]));
        }
        wilson_dslash_t_send<T><<<set_ptr->gridDim_3dim[_T_], set_ptr->blockDim, 0,
                                  set_ptr->stream_dims[_T_]>>>(
            gauge, fermion_in, _device_params, set_ptr->device_send_vec[_B_T_],
            set_ptr->device_send_vec[_F_T_]);
        if (set_ptr->host_params[_GRID_T_] != 1)
        { // t part d2h
          checkCudaErrors(cudaMemcpyAsync(
              set_ptr->host_send_vec[_B_T_], set_ptr->device_send_vec[_B_T_],
              sizeof(T) * set_ptr->lat_3dim_SC[_T_], cudaMemcpyDeviceToHost,
              set_ptr->stream_dims[_T_]));
          checkCudaErrors(cudaMemcpyAsync(
              set_ptr->host_send_vec[_F_T_], set_ptr->device_send_vec[_F_T_],
              sizeof(T) * set_ptr->lat_3dim_SC[_T_], cudaMemcpyDeviceToHost,
              set_ptr->stream_dims[_T_]));
        }
      }
      {                                                          // inside compute part ans wait
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
        wilson_dslash_inside<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                                  set_ptr->stream>>>(gauge, fermion_in, fermion_out,
                                                     _device_params);
      }
      {
        // x edge part
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
        if (set_ptr->host_params[_GRID_X_] == 1)
        {
          // no comm
          // edge recv part
          wilson_dslash_x_recv<T><<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim, 0,
                                    set_ptr->stream>>>(
              gauge, fermion_out, _device_params, set_ptr->device_send_vec[_F_X_],
              set_ptr->device_send_vec[_B_X_]);
        }
        else
        {
          // comm
          _MPI_Isend<T>(set_ptr->host_send_vec[_B_X_], set_ptr->lat_3dim_SC[_X_] / _EVEN_ODD_,
                        set_ptr->move_wards[_B_X_], _B_X_, MPI_COMM_WORLD,
                        &set_ptr->send_request[_B_X_]);
          _MPI_Irecv<T>(set_ptr->host_recv_vec[_F_X_], set_ptr->lat_3dim_SC[_X_] / _EVEN_ODD_,
                        set_ptr->move_wards[_F_X_], _B_X_, MPI_COMM_WORLD,
                        &set_ptr->recv_request[_B_X_]);
          _MPI_Isend<T>(set_ptr->host_send_vec[_F_X_], set_ptr->lat_3dim_SC[_X_] / _EVEN_ODD_,
                        set_ptr->move_wards[_F_X_], _F_X_, MPI_COMM_WORLD,
                        &set_ptr->send_request[_F_X_]);
          _MPI_Irecv<T>(set_ptr->host_recv_vec[_B_X_], set_ptr->lat_3dim_SC[_X_] / _EVEN_ODD_,
                        set_ptr->move_wards[_B_X_], _F_X_, MPI_COMM_WORLD,
                        &set_ptr->recv_request[_F_X_]);
        }
      }
      {
        // y edge part
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
        if (set_ptr->host_params[_GRID_Y_] == 1)
        {
          // no comm
          // edge recv part
          wilson_dslash_y_recv<T><<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim, 0,
                                    set_ptr->stream>>>(
              gauge, fermion_out, _device_params, set_ptr->device_send_vec[_F_Y_],
              set_ptr->device_send_vec[_B_Y_]);
        }
        else
        {
          // comm
          _MPI_Isend<T>(set_ptr->host_send_vec[_B_Y_], set_ptr->lat_3dim_SC[_Y_],
                        set_ptr->move_wards[_B_Y_], _B_Y_, MPI_COMM_WORLD,
                        &set_ptr->send_request[_B_Y_]);
          _MPI_Irecv<T>(set_ptr->host_recv_vec[_F_Y_], set_ptr->lat_3dim_SC[_Y_],
                        set_ptr->move_wards[_F_Y_], _B_Y_, MPI_COMM_WORLD,
                        &set_ptr->recv_request[_B_Y_]);
          _MPI_Isend<T>(set_ptr->host_send_vec[_F_Y_], set_ptr->lat_3dim_SC[_Y_],
                        set_ptr->move_wards[_F_Y_], _F_Y_, MPI_COMM_WORLD,
                        &set_ptr->send_request[_F_Y_]);
          _MPI_Irecv<T>(set_ptr->host_recv_vec[_B_Y_], set_ptr->lat_3dim_SC[_Y_],
                        set_ptr->move_wards[_B_Y_], _F_Y_, MPI_COMM_WORLD,
                        &set_ptr->recv_request[_F_Y_]);
        }
      }
      {
        // z edge part
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
        if (set_ptr->host_params[_GRID_Z_] == 1)
        {
          // no comm
          // edge recv part
          wilson_dslash_z_recv<T><<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim, 0,
                                    set_ptr->stream>>>(
              gauge, fermion_out, _device_params, set_ptr->device_send_vec[_F_Z_],
              set_ptr->device_send_vec[_B_Z_]);
        }
        else
        {
          // comm
          _MPI_Isend<T>(set_ptr->host_send_vec[_B_Z_], set_ptr->lat_3dim_SC[_Z_],
                        set_ptr->move_wards[_B_Z_], _B_Z_, MPI_COMM_WORLD,
                        &set_ptr->send_request[_B_Z_]);
          _MPI_Irecv<T>(set_ptr->host_recv_vec[_F_Z_], set_ptr->lat_3dim_SC[_Z_],
                        set_ptr->move_wards[_F_Z_], _B_Z_, MPI_COMM_WORLD,
                        &set_ptr->recv_request[_B_Z_]);
          _MPI_Isend<T>(set_ptr->host_send_vec[_F_Z_], set_ptr->lat_3dim_SC[_Z_],
                        set_ptr->move_wards[_F_Z_], _F_Z_, MPI_COMM_WORLD,
                        &set_ptr->send_request[_F_Z_]);
          _MPI_Irecv<T>(set_ptr->host_recv_vec[_B_Z_], set_ptr->lat_3dim_SC[_Z_],
                        set_ptr->move_wards[_B_Z_], _F_Z_, MPI_COMM_WORLD,
                        &set_ptr->recv_request[_F_Z_]);
        }
      }
      {
        // t edge part
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
        if (set_ptr->host_params[_GRID_T_] == 1)
        {
          // no comm
          // edge recv part
          wilson_dslash_t_recv<T><<<set_ptr->gridDim_3dim[_T_], set_ptr->blockDim, 0,
                                    set_ptr->stream>>>(
              gauge, fermion_out, _device_params, set_ptr->device_send_vec[_F_T_],
              set_ptr->device_send_vec[_B_T_]);
        }
        else
        {
          // comm
          _MPI_Isend<T>(set_ptr->host_send_vec[_B_T_], set_ptr->lat_3dim_SC[_T_],
                        set_ptr->move_wards[_B_T_], _B_T_, MPI_COMM_WORLD,
                        &set_ptr->send_request[_B_T_]);
          _MPI_Irecv<T>(set_ptr->host_recv_vec[_F_T_], set_ptr->lat_3dim_SC[_T_],
                        set_ptr->move_wards[_F_T_], _B_T_, MPI_COMM_WORLD,
                        &set_ptr->recv_request[_B_T_]);
          _MPI_Isend<T>(set_ptr->host_send_vec[_F_T_], set_ptr->lat_3dim_SC[_T_],
                        set_ptr->move_wards[_F_T_], _F_T_, MPI_COMM_WORLD,
                        &set_ptr->send_request[_F_T_]);
          _MPI_Irecv<T>(set_ptr->host_recv_vec[_B_T_], set_ptr->lat_3dim_SC[_T_],
                        set_ptr->move_wards[_B_T_], _F_T_, MPI_COMM_WORLD,
                        &set_ptr->recv_request[_F_T_]);
        }
      }
      if (set_ptr->host_params[_GRID_X_] != 1)
      { // x part h2d
        MPI_Wait(&set_ptr->recv_request[_B_X_], MPI_STATUS_IGNORE);
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_F_X_], set_ptr->host_recv_vec[_F_X_],
            sizeof(T) * set_ptr->lat_3dim_SC[_X_] / _EVEN_ODD_, cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_X_]));
        MPI_Wait(&set_ptr->recv_request[_F_X_], MPI_STATUS_IGNORE);
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_B_X_], set_ptr->host_recv_vec[_B_X_],
            sizeof(T) * set_ptr->lat_3dim_SC[_X_] / _EVEN_ODD_, cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_X_]));
      }
      if (set_ptr->host_params[_GRID_Y_] != 1)
      { // y part h2d
        MPI_Wait(&set_ptr->recv_request[_B_Y_], MPI_STATUS_IGNORE);
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_F_Y_], set_ptr->host_recv_vec[_F_Y_],
            sizeof(T) * set_ptr->lat_3dim_SC[_Y_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_Y_]));
        MPI_Wait(&set_ptr->recv_request[_F_Y_], MPI_STATUS_IGNORE);
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_B_Y_], set_ptr->host_recv_vec[_B_Y_],
            sizeof(T) * set_ptr->lat_3dim_SC[_Y_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_Y_]));
      }
      if (set_ptr->host_params[_GRID_Z_] != 1)
      { // z part h2d
        MPI_Wait(&set_ptr->recv_request[_B_Z_], MPI_STATUS_IGNORE);
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_F_Z_], set_ptr->host_recv_vec[_F_Z_],
            sizeof(T) * set_ptr->lat_3dim_SC[_Z_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_Z_]));
        MPI_Wait(&set_ptr->recv_request[_F_Z_], MPI_STATUS_IGNORE);
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_B_Z_], set_ptr->host_recv_vec[_B_Z_],
            sizeof(T) * set_ptr->lat_3dim_SC[_Z_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_Z_]));
      }
      if (set_ptr->host_params[_GRID_T_] != 1)
      { // t part h2d
        MPI_Wait(&set_ptr->recv_request[_B_T_], MPI_STATUS_IGNORE);
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_F_T_], set_ptr->host_recv_vec[_F_T_],
            sizeof(T) * set_ptr->lat_3dim_SC[_T_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_T_]));
        MPI_Wait(&set_ptr->recv_request[_F_T_], MPI_STATUS_IGNORE);
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_B_T_], set_ptr->host_recv_vec[_B_T_],
            sizeof(T) * set_ptr->lat_3dim_SC[_T_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_T_]));
      }
      {
        // edge recv part
        if (set_ptr->host_params[_GRID_X_] != 1)
        { // x part recv
          checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
          wilson_dslash_x_recv<T><<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim, 0,
                                    set_ptr->stream>>>(
              gauge, fermion_out, _device_params, set_ptr->device_recv_vec[_B_X_],
              set_ptr->device_recv_vec[_F_X_]);
        }
        if (set_ptr->host_params[_GRID_Y_] != 1)
        { // y part recv
          checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
          wilson_dslash_y_recv<T><<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim, 0,
                                    set_ptr->stream>>>(
              gauge, fermion_out, _device_params, set_ptr->device_recv_vec[_B_Y_],
              set_ptr->device_recv_vec[_F_Y_]);
        }
        if (set_ptr->host_params[_GRID_Z_] != 1)
        { // z part recv
          checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
          wilson_dslash_z_recv<T><<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim, 0,
                                    set_ptr->stream>>>(
              gauge, fermion_out, _device_params, set_ptr->device_recv_vec[_B_Z_],
              set_ptr->device_recv_vec[_F_Z_]);
        }
        if (set_ptr->host_params[_GRID_T_] != 1)
        { // t part recv
          checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
          wilson_dslash_t_recv<T><<<set_ptr->gridDim_3dim[_T_], set_ptr->blockDim, 0,
                                    set_ptr->stream>>>(
              gauge, fermion_out, _device_params, set_ptr->device_recv_vec[_B_T_],
              set_ptr->device_recv_vec[_F_T_]);
        }
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
    }
    void run_mpi(void *fermion_out, void *fermion_in, void *gauge,
                 void *_device_params)
    {
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
      { // edge send part
        wilson_dslash_x_send<T><<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim, 0,
                                  set_ptr->stream_dims[_X_]>>>(
            gauge, fermion_in, _device_params, set_ptr->device_send_vec[_B_X_],
            set_ptr->device_send_vec[_F_X_]);
        if (set_ptr->host_params[_GRID_X_] != 1)
        { // x part d2h
          checkCudaErrors(cudaMemcpyAsync(
              set_ptr->host_send_vec[_B_X_], set_ptr->device_send_vec[_B_X_],
              sizeof(T) * set_ptr->lat_3dim_SC[_X_] / _EVEN_ODD_, cudaMemcpyDeviceToHost,
              set_ptr->stream_dims[_X_]));
          checkCudaErrors(cudaMemcpyAsync(
              set_ptr->host_send_vec[_F_X_], set_ptr->device_send_vec[_F_X_],
              sizeof(T) * set_ptr->lat_3dim_SC[_X_] / _EVEN_ODD_, cudaMemcpyDeviceToHost,
              set_ptr->stream_dims[_X_]));
        }
        wilson_dslash_y_send<T><<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim, 0,
                                  set_ptr->stream_dims[_Y_]>>>(
            gauge, fermion_in, _device_params, set_ptr->device_send_vec[_B_Y_],
            set_ptr->device_send_vec[_F_Y_]);
        if (set_ptr->host_params[_GRID_Y_] != 1)
        { // y part d2h
          checkCudaErrors(cudaMemcpyAsync(
              set_ptr->host_send_vec[_B_Y_], set_ptr->device_send_vec[_B_Y_],
              sizeof(T) * set_ptr->lat_3dim_SC[_Y_], cudaMemcpyDeviceToHost,
              set_ptr->stream_dims[_Y_]));
          checkCudaErrors(cudaMemcpyAsync(
              set_ptr->host_send_vec[_F_Y_], set_ptr->device_send_vec[_F_Y_],
              sizeof(T) * set_ptr->lat_3dim_SC[_Y_], cudaMemcpyDeviceToHost,
              set_ptr->stream_dims[_Y_]));
        }
        wilson_dslash_z_send<T><<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim, 0,
                                  set_ptr->stream_dims[_Z_]>>>(
            gauge, fermion_in, _device_params, set_ptr->device_send_vec[_B_Z_],
            set_ptr->device_send_vec[_F_Z_]);
        if (set_ptr->host_params[_GRID_Z_] != 1)
        { // z part d2h
          checkCudaErrors(cudaMemcpyAsync(
              set_ptr->host_send_vec[_B_Z_], set_ptr->device_send_vec[_B_Z_],
              sizeof(T) * set_ptr->lat_3dim_SC[_Z_], cudaMemcpyDeviceToHost,
              set_ptr->stream_dims[_Z_]));
          checkCudaErrors(cudaMemcpyAsync(
              set_ptr->host_send_vec[_F_Z_], set_ptr->device_send_vec[_F_Z_],
              sizeof(T) * set_ptr->lat_3dim_SC[_Z_], cudaMemcpyDeviceToHost,
              set_ptr->stream_dims[_Z_]));
        }
        wilson_dslash_t_send<T><<<set_ptr->gridDim_3dim[_T_], set_ptr->blockDim, 0,
                                  set_ptr->stream_dims[_T_]>>>(
            gauge, fermion_in, _device_params, set_ptr->device_send_vec[_B_T_],
            set_ptr->device_send_vec[_F_T_]);
        if (set_ptr->host_params[_GRID_T_] != 1)
        { // t part d2h
          checkCudaErrors(cudaMemcpyAsync(
              set_ptr->host_send_vec[_B_T_], set_ptr->device_send_vec[_B_T_],
              sizeof(T) * set_ptr->lat_3dim_SC[_T_], cudaMemcpyDeviceToHost,
              set_ptr->stream_dims[_T_]));
          checkCudaErrors(cudaMemcpyAsync(
              set_ptr->host_send_vec[_F_T_], set_ptr->device_send_vec[_F_T_],
              sizeof(T) * set_ptr->lat_3dim_SC[_T_], cudaMemcpyDeviceToHost,
              set_ptr->stream_dims[_T_]));
        }
      }
      {                                                          // inside compute part ans wait
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
        wilson_dslash_inside<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                                  set_ptr->stream>>>(gauge, fermion_in, fermion_out,
                                                     _device_params);
      }
      {
        // x edge part
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
        if (set_ptr->host_params[_GRID_X_] == 1)
        {
          // no comm
          // edge recv part
          wilson_dslash_x_recv<T><<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim, 0,
                                    set_ptr->stream>>>(
              gauge, fermion_out, _device_params, set_ptr->device_send_vec[_F_X_],
              set_ptr->device_send_vec[_B_X_]);
        }
        else
        {
          // comm
          _MPI_Sendrecv<T>(set_ptr->host_send_vec[_B_X_], set_ptr->lat_3dim_SC[_X_] / _EVEN_ODD_,
                           set_ptr->move_wards[_B_X_], _B_X_, set_ptr->host_recv_vec[_F_X_], set_ptr->lat_3dim_SC[_X_] / _EVEN_ODD_,
                           set_ptr->move_wards[_F_X_], _B_X_, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE);
          _MPI_Sendrecv<T>(set_ptr->host_send_vec[_F_X_], set_ptr->lat_3dim_SC[_X_] / _EVEN_ODD_,
                           set_ptr->move_wards[_F_X_], _F_X_, set_ptr->host_recv_vec[_B_X_], set_ptr->lat_3dim_SC[_X_] / _EVEN_ODD_,
                           set_ptr->move_wards[_B_X_], _F_X_, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE);
        }
      }
      {
        // y edge part
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
        if (set_ptr->host_params[_GRID_Y_] == 1)
        {
          // no comm
          // edge recv part
          wilson_dslash_y_recv<T><<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim, 0,
                                    set_ptr->stream>>>(
              gauge, fermion_out, _device_params, set_ptr->device_send_vec[_F_Y_],
              set_ptr->device_send_vec[_B_Y_]);
        }
        else
        {
          // comm
          _MPI_Sendrecv<T>(set_ptr->host_send_vec[_B_Y_], set_ptr->lat_3dim_SC[_Y_],
                           set_ptr->move_wards[_B_Y_], _B_Y_, set_ptr->host_recv_vec[_F_Y_], set_ptr->lat_3dim_SC[_Y_],
                           set_ptr->move_wards[_F_Y_], _B_Y_, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE);
          _MPI_Sendrecv<T>(set_ptr->host_send_vec[_F_Y_], set_ptr->lat_3dim_SC[_Y_],
                           set_ptr->move_wards[_F_Y_], _F_Y_, set_ptr->host_recv_vec[_B_Y_], set_ptr->lat_3dim_SC[_Y_],
                           set_ptr->move_wards[_B_Y_], _F_Y_, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE);
        }
      }
      {
        // z edge part
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
        if (set_ptr->host_params[_GRID_Z_] == 1)
        {
          // no comm
          // edge recv part
          wilson_dslash_z_recv<T><<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim, 0,
                                    set_ptr->stream>>>(
              gauge, fermion_out, _device_params, set_ptr->device_send_vec[_F_Z_],
              set_ptr->device_send_vec[_B_Z_]);
        }
        else
        {
          // comm
          _MPI_Sendrecv<T>(set_ptr->host_send_vec[_B_Z_], set_ptr->lat_3dim_SC[_Z_],
                           set_ptr->move_wards[_B_Z_], _B_Z_, set_ptr->host_recv_vec[_F_Z_], set_ptr->lat_3dim_SC[_Z_],
                           set_ptr->move_wards[_F_Z_], _B_Z_, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE);
          _MPI_Sendrecv<T>(set_ptr->host_send_vec[_F_Z_], set_ptr->lat_3dim_SC[_Z_],
                           set_ptr->move_wards[_F_Z_], _F_Z_, set_ptr->host_recv_vec[_B_Z_], set_ptr->lat_3dim_SC[_Z_],
                           set_ptr->move_wards[_B_Z_], _F_Z_, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE);
        }
      }
      {
        // t edge part
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
        if (set_ptr->host_params[_GRID_T_] == 1)
        {
          // no comm
          // edge recv part
          wilson_dslash_t_recv<T><<<set_ptr->gridDim_3dim[_T_], set_ptr->blockDim, 0,
                                    set_ptr->stream>>>(
              gauge, fermion_out, _device_params, set_ptr->device_send_vec[_F_T_],
              set_ptr->device_send_vec[_B_T_]);
        }
        else
        {
          // comm
          _MPI_Sendrecv<T>(set_ptr->host_send_vec[_B_T_], set_ptr->lat_3dim_SC[_T_],
                           set_ptr->move_wards[_B_T_], _B_T_, set_ptr->host_recv_vec[_F_T_], set_ptr->lat_3dim_SC[_T_],
                           set_ptr->move_wards[_F_T_], _B_T_, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE);
          _MPI_Sendrecv<T>(set_ptr->host_send_vec[_F_T_], set_ptr->lat_3dim_SC[_T_],
                           set_ptr->move_wards[_F_T_], _F_T_, set_ptr->host_recv_vec[_B_T_], set_ptr->lat_3dim_SC[_T_],
                           set_ptr->move_wards[_B_T_], _F_T_, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE);
        }
      }
      if (set_ptr->host_params[_GRID_X_] != 1)
      { // x part h2d
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_F_X_], set_ptr->host_recv_vec[_F_X_],
            sizeof(T) * set_ptr->lat_3dim_SC[_X_] / _EVEN_ODD_, cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_X_]));
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_B_X_], set_ptr->host_recv_vec[_B_X_],
            sizeof(T) * set_ptr->lat_3dim_SC[_X_] / _EVEN_ODD_, cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_X_]));
      }
      if (set_ptr->host_params[_GRID_Y_] != 1)
      { // y part h2d
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_F_Y_], set_ptr->host_recv_vec[_F_Y_],
            sizeof(T) * set_ptr->lat_3dim_SC[_Y_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_Y_]));
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_B_Y_], set_ptr->host_recv_vec[_B_Y_],
            sizeof(T) * set_ptr->lat_3dim_SC[_Y_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_Y_]));
      }
      if (set_ptr->host_params[_GRID_Z_] != 1)
      { // z part h2d
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_F_Z_], set_ptr->host_recv_vec[_F_Z_],
            sizeof(T) * set_ptr->lat_3dim_SC[_Z_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_Z_]));
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_B_Z_], set_ptr->host_recv_vec[_B_Z_],
            sizeof(T) * set_ptr->lat_3dim_SC[_Z_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_Z_]));
      }
      if (set_ptr->host_params[_GRID_T_] != 1)
      { // t part h2d
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_F_T_], set_ptr->host_recv_vec[_F_T_],
            sizeof(T) * set_ptr->lat_3dim_SC[_T_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_T_]));
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_B_T_], set_ptr->host_recv_vec[_B_T_],
            sizeof(T) * set_ptr->lat_3dim_SC[_T_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_T_]));
      }
      {
        // edge recv part
        if (set_ptr->host_params[_GRID_X_] != 1)
        { // x part recv
          checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
          wilson_dslash_x_recv<T><<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim, 0,
                                    set_ptr->stream>>>(
              gauge, fermion_out, _device_params, set_ptr->device_recv_vec[_B_X_],
              set_ptr->device_recv_vec[_F_X_]);
        }
        if (set_ptr->host_params[_GRID_Y_] != 1)
        { // y part recv
          checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
          wilson_dslash_y_recv<T><<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim, 0,
                                    set_ptr->stream>>>(
              gauge, fermion_out, _device_params, set_ptr->device_recv_vec[_B_Y_],
              set_ptr->device_recv_vec[_F_Y_]);
        }
        if (set_ptr->host_params[_GRID_Z_] != 1)
        { // z part recv
          checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
          wilson_dslash_z_recv<T><<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim, 0,
                                    set_ptr->stream>>>(
              gauge, fermion_out, _device_params, set_ptr->device_recv_vec[_B_Z_],
              set_ptr->device_recv_vec[_F_Z_]);
        }
        if (set_ptr->host_params[_GRID_T_] != 1)
        { // t part recv
          checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
          wilson_dslash_t_recv<T><<<set_ptr->gridDim_3dim[_T_], set_ptr->blockDim, 0,
                                    set_ptr->stream>>>(
              gauge, fermion_out, _device_params, set_ptr->device_recv_vec[_B_T_],
              set_ptr->device_recv_vec[_F_T_]);
        }
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
    }
    void run(void *fermion_out, void *fermion_in, void *gauge, void *_device_params)
    {
      run_mpi(fermion_out, fermion_in, gauge, _device_params);
      // run_mpi_non_block(fermion_out, fermion_in, gauge, _device_params);
    }
    void run_eo(void *fermion_out, void *fermion_in, void *gauge)
    {
      run(fermion_out, fermion_in, gauge, set_ptr->device_params_even_no_dag);
    }
    void run_oe(void *fermion_out, void *fermion_in, void *gauge)
    {
      run(fermion_out, fermion_in, gauge, set_ptr->device_params_odd_no_dag);
    }
    void run_eo_dag(void *fermion_out, void *fermion_in, void *gauge)
    {
      run(fermion_out, fermion_in, gauge, set_ptr->device_params_even_dag);
    }
    void run_oe_dag(void *fermion_out, void *fermion_in, void *gauge)
    {
      run(fermion_out, fermion_in, gauge, set_ptr->device_params_odd_dag);
    }
    void run_test(void *fermion_out, void *fermion_in, void *gauge)
    {
#ifdef PRINT_MULTI_GPU_WILSON_DSLASH
      set_ptr->_print();
#endif
      auto start = std::chrono::high_resolution_clock::now();
      run(fermion_out, fermion_in, gauge, set_ptr->device_params);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();
      err = cudaGetLastError();
      checkCudaErrors(err);
      printf("multi-gpu wilson dslash total time: (without malloc free memcpy) :%.9lf "
             "sec\n",
             double(duration) / 1e9);
    }
  };
}
#endif
