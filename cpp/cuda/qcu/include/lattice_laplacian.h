#ifndef _LATTICE_LAPLACIAN_H
#define _LATTICE_LAPLACIAN_H
#include "./lattice_set.h"
#include "./lattice_mpi.h"
#include "./laplacian.h"
namespace qcu
{
  template <typename T>
  struct LatticeLaplacian
  {
    LatticeSet<T> *set_ptr;
    cudaError_t err;
    void give(LatticeSet<T> *_set_ptr)
    {
      set_ptr = _set_ptr;
    }
    void run_mpi(void *laplacian_out, void *laplacian_in, void *gauge,
                 void *_device_params)
    {
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
      { // edge send part
        laplacian_x_send<T><<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim, 0,
                              set_ptr->stream_dims[_X_]>>>(
            gauge, laplacian_in, _device_params, set_ptr->device_send_vec[_B_X_],
            set_ptr->device_send_vec[_F_X_]);
        if (set_ptr->host_params[_GRID_X_] != 1)
        { // x part d2h
          checkCudaErrors(cudaMemcpyAsync(
              set_ptr->host_send_vec[_B_X_], set_ptr->device_send_vec[_B_X_],
              sizeof(T) * _REAL_IMAG_ * set_ptr->lat_3dim_C[_X_], cudaMemcpyDeviceToHost,
              set_ptr->stream_dims[_X_]));
          checkCudaErrors(cudaMemcpyAsync(
              set_ptr->host_send_vec[_F_X_], set_ptr->device_send_vec[_F_X_],
              sizeof(T) * _REAL_IMAG_ * set_ptr->lat_3dim_C[_X_], cudaMemcpyDeviceToHost,
              set_ptr->stream_dims[_X_]));
        }
        laplacian_y_send<T><<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim, 0,
                              set_ptr->stream_dims[_Y_]>>>(
            gauge, laplacian_in, _device_params, set_ptr->device_send_vec[_B_Y_],
            set_ptr->device_send_vec[_F_Y_]);
        if (set_ptr->host_params[_GRID_Y_] != 1)
        { // y part d2h
          checkCudaErrors(cudaMemcpyAsync(
              set_ptr->host_send_vec[_B_Y_], set_ptr->device_send_vec[_B_Y_],
              sizeof(T) * _REAL_IMAG_ * set_ptr->lat_3dim_C[_Y_], cudaMemcpyDeviceToHost,
              set_ptr->stream_dims[_Y_]));
          checkCudaErrors(cudaMemcpyAsync(
              set_ptr->host_send_vec[_F_Y_], set_ptr->device_send_vec[_F_Y_],
              sizeof(T) * _REAL_IMAG_ * set_ptr->lat_3dim_C[_Y_], cudaMemcpyDeviceToHost,
              set_ptr->stream_dims[_Y_]));
        }
        laplacian_z_send<T><<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim, 0,
                              set_ptr->stream_dims[_Z_]>>>(
            gauge, laplacian_in, _device_params, set_ptr->device_send_vec[_B_Z_],
            set_ptr->device_send_vec[_F_Z_]);
        if (set_ptr->host_params[_GRID_Z_] != 1)
        { // z part d2h
          checkCudaErrors(cudaMemcpyAsync(
              set_ptr->host_send_vec[_B_Z_], set_ptr->device_send_vec[_B_Z_],
              sizeof(T) * _REAL_IMAG_ * set_ptr->lat_3dim_C[_Z_], cudaMemcpyDeviceToHost,
              set_ptr->stream_dims[_Z_]));
          checkCudaErrors(cudaMemcpyAsync(
              set_ptr->host_send_vec[_F_Z_], set_ptr->device_send_vec[_F_Z_],
              sizeof(T) * _REAL_IMAG_ * set_ptr->lat_3dim_C[_Z_], cudaMemcpyDeviceToHost,
              set_ptr->stream_dims[_Z_]));
        }
      }
      {                                                          // inside compute part ans wait
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
        laplacian_inside<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                              set_ptr->stream>>>(gauge, laplacian_in, laplacian_out,
                                                 _device_params);
      }
      {
        // x edge part
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
        if (set_ptr->host_params[_GRID_X_] == 1)
        {
          // no comm
          // edge recv part
          laplacian_x_recv<T><<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim, 0,
                                set_ptr->stream>>>(
              gauge, laplacian_out, _device_params, set_ptr->device_send_vec[_F_X_],
              set_ptr->device_send_vec[_B_X_]);
        }
        else
        {
          // comm
          _MPI_Sendrecv<T>(set_ptr->host_send_vec[_B_X_], _REAL_IMAG_ * set_ptr->lat_3dim_C[_X_],
                           set_ptr->move_wards[_B_X_], _B_X_, set_ptr->host_recv_vec[_F_X_], _REAL_IMAG_ * set_ptr->lat_3dim_C[_X_],
                           set_ptr->move_wards[_F_X_], _B_X_, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE);
          _MPI_Sendrecv<T>(set_ptr->host_send_vec[_F_X_], _REAL_IMAG_ * set_ptr->lat_3dim_C[_X_],
                           set_ptr->move_wards[_F_X_], _F_X_, set_ptr->host_recv_vec[_B_X_], _REAL_IMAG_ * set_ptr->lat_3dim_C[_X_],
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
          laplacian_y_recv<T><<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim, 0,
                                set_ptr->stream>>>(
              gauge, laplacian_out, _device_params, set_ptr->device_send_vec[_F_Y_],
              set_ptr->device_send_vec[_B_Y_]);
        }
        else
        {
          // comm
          _MPI_Sendrecv<T>(set_ptr->host_send_vec[_B_Y_], _REAL_IMAG_ * set_ptr->lat_3dim_C[_Y_],
                           set_ptr->move_wards[_B_Y_], _B_Y_, set_ptr->host_recv_vec[_F_Y_], _REAL_IMAG_ * set_ptr->lat_3dim_C[_Y_],
                           set_ptr->move_wards[_F_Y_], _B_Y_, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE);
          _MPI_Sendrecv<T>(set_ptr->host_send_vec[_F_Y_], _REAL_IMAG_ * set_ptr->lat_3dim_C[_Y_],
                           set_ptr->move_wards[_F_Y_], _F_Y_, set_ptr->host_recv_vec[_B_Y_], _REAL_IMAG_ * set_ptr->lat_3dim_C[_Y_],
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
          laplacian_z_recv<T><<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim, 0,
                                set_ptr->stream>>>(
              gauge, laplacian_out, _device_params, set_ptr->device_send_vec[_F_Z_],
              set_ptr->device_send_vec[_B_Z_]);
        }
        else
        {
          // comm
          _MPI_Sendrecv<T>(set_ptr->host_send_vec[_B_Z_], _REAL_IMAG_ * set_ptr->lat_3dim_C[_Z_],
                           set_ptr->move_wards[_B_Z_], _B_Z_, set_ptr->host_recv_vec[_F_Z_], _REAL_IMAG_ * set_ptr->lat_3dim_C[_Z_],
                           set_ptr->move_wards[_F_Z_], _B_Z_, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE);
          _MPI_Sendrecv<T>(set_ptr->host_send_vec[_F_Z_], _REAL_IMAG_ * set_ptr->lat_3dim_C[_Z_],
                           set_ptr->move_wards[_F_Z_], _F_Z_, set_ptr->host_recv_vec[_B_Z_], _REAL_IMAG_ * set_ptr->lat_3dim_C[_Z_],
                           set_ptr->move_wards[_B_Z_], _F_Z_, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE);
        }
      }
      if (set_ptr->host_params[_GRID_X_] != 1)
      { // x part h2d
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_F_X_], set_ptr->host_recv_vec[_F_X_],
            sizeof(T) * _REAL_IMAG_ * set_ptr->lat_3dim_C[_X_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_X_]));
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_B_X_], set_ptr->host_recv_vec[_B_X_],
            sizeof(T) * _REAL_IMAG_ * set_ptr->lat_3dim_C[_X_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_X_]));
      }
      if (set_ptr->host_params[_GRID_Y_] != 1)
      { // y part h2d
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_F_Y_], set_ptr->host_recv_vec[_F_Y_],
            sizeof(T) * _REAL_IMAG_ * set_ptr->lat_3dim_C[_Y_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_Y_]));
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_B_Y_], set_ptr->host_recv_vec[_B_Y_],
            sizeof(T) * _REAL_IMAG_ * set_ptr->lat_3dim_C[_Y_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_Y_]));
      }
      if (set_ptr->host_params[_GRID_Z_] != 1)
      { // z part h2d
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_F_Z_], set_ptr->host_recv_vec[_F_Z_],
            sizeof(T) * _REAL_IMAG_ * set_ptr->lat_3dim_C[_Z_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_Z_]));
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_B_Z_], set_ptr->host_recv_vec[_B_Z_],
            sizeof(T) * _REAL_IMAG_ * set_ptr->lat_3dim_C[_Z_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_Z_]));
      }
      {
        // edge recv part
        if (set_ptr->host_params[_GRID_X_] != 1)
        { // x part recv
          checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
          laplacian_x_recv<T><<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim, 0,
                                set_ptr->stream>>>(
              gauge, laplacian_out, _device_params, set_ptr->device_recv_vec[_B_X_],
              set_ptr->device_recv_vec[_F_X_]);
        }
        if (set_ptr->host_params[_GRID_Y_] != 1)
        { // y part recv
          checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
          laplacian_y_recv<T><<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim, 0,
                                set_ptr->stream>>>(
              gauge, laplacian_out, _device_params, set_ptr->device_recv_vec[_B_Y_],
              set_ptr->device_recv_vec[_F_Y_]);
        }
        if (set_ptr->host_params[_GRID_Z_] != 1)
        { // z part recv
          checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
          laplacian_z_recv<T><<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim, 0,
                                set_ptr->stream>>>(
              gauge, laplacian_out, _device_params, set_ptr->device_recv_vec[_B_Z_],
              set_ptr->device_recv_vec[_F_Z_]);
        }
      }
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
    }
    void _run(void *laplacian_out, void *laplacian_in, void *gauge, void *_device_params)
    {
      run_mpi(laplacian_out, laplacian_in, gauge, _device_params);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
      laplacian_give_complete<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                                   set_ptr->stream>>>(
          laplacian_out, laplacian_in, _device_params);
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
      err = cudaGetLastError();
      checkCudaErrors(err);
    }
    void run(void *laplacian_out, void *laplacian_in, void *gauge)
    {
      _run(laplacian_out, laplacian_in, gauge, set_ptr->device_params);
    }
    void run_test(void *laplacian_out, void *laplacian_in, void *gauge)
    {
      auto start = std::chrono::high_resolution_clock::now();
      run(laplacian_out, laplacian_in, gauge);
      auto end = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();
      printf("multi-gpu laplacian total time: (without malloc free memcpy) :%.9lf "
             "sec\n",
             double(duration) / 1e9);
    }
  };
}
#endif
