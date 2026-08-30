#ifndef _LATTICE_WILSON_DSLASH_H
#define _LATTICE_WILSON_DSLASH_H
#include "./lattice_mpi.h"
#include "./lattice_set.h"
#include "./wilson_dslash.h"
namespace qcu {
template <typename T> struct LatticeWilsonDslash {
  LatticeSet<T> *set_ptr;
  cudaError_t err;
  // When true, the single-rank run_mpi path skips its final
  // cudaStreamSynchronize.  ONLY safe when every consumer of the dslash
  // output runs on the same main stream (in-stream ordering then guarantees
  // visibility).  Used by the MG fast fine iteration (2026-08-02) to save
  // ~170 us per run_mpi.
  bool skip_final_sync_ = false;
  void give(LatticeSet<T> *_set_ptr) { set_ptr = _set_ptr; }
  void run_mpi_non_block(void *fermion_out, void *fermion_in, void *gauge,
                         void *_device_params) {
    const bool force_mpi = _WILSON_AND_LAPLACIAN_TEST_SINGLE_IN_MULTI_ != 0;
    const bool device_mpi = qcu_mpi_can_use_buffer<T>(
        set_ptr->device_send_vec[_B_X_]);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
    { // edge send part
      wilson_dslash_x_send<T><<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim,
                                0, set_ptr->stream_dims[_X_]>>>(
          gauge, fermion_in, _device_params, set_ptr->device_send_vec[_B_X_],
          set_ptr->device_send_vec[_F_X_]);
      if (!device_mpi && (set_ptr->host_params[_GRID_X_] != 1 ||
                          force_mpi)) { // x part d2h
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_B_X_], set_ptr->device_send_vec[_B_X_],
            sizeof(T) * set_ptr->lat_3dim_SC[_X_], cudaMemcpyDeviceToHost,
            set_ptr->stream_dims[_X_]));
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_F_X_], set_ptr->device_send_vec[_F_X_],
            sizeof(T) * set_ptr->lat_3dim_SC[_X_], cudaMemcpyDeviceToHost,
            set_ptr->stream_dims[_X_]));
      }
      wilson_dslash_y_send<T><<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim,
                                0, set_ptr->stream_dims[_Y_]>>>(
          gauge, fermion_in, _device_params, set_ptr->device_send_vec[_B_Y_],
          set_ptr->device_send_vec[_F_Y_]);
      if (!device_mpi && (set_ptr->host_params[_GRID_Y_] != 1 ||
                          force_mpi)) { // y part d2h
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_B_Y_], set_ptr->device_send_vec[_B_Y_],
            sizeof(T) * set_ptr->lat_3dim_SC[_Y_], cudaMemcpyDeviceToHost,
            set_ptr->stream_dims[_Y_]));
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_F_Y_], set_ptr->device_send_vec[_F_Y_],
            sizeof(T) * set_ptr->lat_3dim_SC[_Y_], cudaMemcpyDeviceToHost,
            set_ptr->stream_dims[_Y_]));
      }
      wilson_dslash_z_send<T><<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim,
                                0, set_ptr->stream_dims[_Z_]>>>(
          gauge, fermion_in, _device_params, set_ptr->device_send_vec[_B_Z_],
          set_ptr->device_send_vec[_F_Z_]);
      if (!device_mpi && (set_ptr->host_params[_GRID_Z_] != 1 ||
                          force_mpi)) { // z part d2h
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_B_Z_], set_ptr->device_send_vec[_B_Z_],
            sizeof(T) * set_ptr->lat_3dim_SC[_Z_], cudaMemcpyDeviceToHost,
            set_ptr->stream_dims[_Z_]));
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_F_Z_], set_ptr->device_send_vec[_F_Z_],
            sizeof(T) * set_ptr->lat_3dim_SC[_Z_], cudaMemcpyDeviceToHost,
            set_ptr->stream_dims[_Z_]));
      }
      wilson_dslash_t_send<T><<<set_ptr->gridDim_3dim[_T_], set_ptr->blockDim,
                                0, set_ptr->stream_dims[_T_]>>>(
          gauge, fermion_in, _device_params, set_ptr->device_send_vec[_B_T_],
          set_ptr->device_send_vec[_F_T_]);
      if (!device_mpi && (set_ptr->host_params[_GRID_T_] != 1 ||
                          force_mpi)) { // t part d2h
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_B_T_], set_ptr->device_send_vec[_B_T_],
            sizeof(T) * set_ptr->lat_3dim_SC[_T_] / _EVEN_ODD_,
            cudaMemcpyDeviceToHost, set_ptr->stream_dims[_T_]));
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_F_T_], set_ptr->device_send_vec[_F_T_],
            sizeof(T) * set_ptr->lat_3dim_SC[_T_] / _EVEN_ODD_,
            cudaMemcpyDeviceToHost, set_ptr->stream_dims[_T_]));
      }
    }
    { // inside compute part ans wait
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
      wilson_dslash_inside<T>
          <<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
              gauge, fermion_in, fermion_out, _device_params);
    }
    {
      // x edge part
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
      if (set_ptr->host_params[_GRID_X_] == 1 && !force_mpi) {
        // no comm
        // edge recv part
        wilson_dslash_x_recv<T><<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim,
                                  0, set_ptr->stream>>>(
            gauge, fermion_out, _device_params, set_ptr->device_send_vec[_F_X_],
            set_ptr->device_send_vec[_B_X_]);
      } else {
        // comm
        const void *send_b = device_mpi ? set_ptr->device_send_vec[_B_X_]
                                        : set_ptr->host_send_vec[_B_X_];
        const void *send_f = device_mpi ? set_ptr->device_send_vec[_F_X_]
                                        : set_ptr->host_send_vec[_F_X_];
        void *recv_f = device_mpi ? set_ptr->device_recv_vec[_F_X_]
                                  : set_ptr->host_recv_vec[_F_X_];
        void *recv_b = device_mpi ? set_ptr->device_recv_vec[_B_X_]
                                  : set_ptr->host_recv_vec[_B_X_];
        _MPI_Isend<T>(send_b, set_ptr->lat_3dim_SC[_X_],
                      set_ptr->move_wards[_B_X_], _B_X_, MPI_COMM_WORLD,
                      &set_ptr->send_request[_B_X_]);
        _MPI_Irecv<T>(recv_f, set_ptr->lat_3dim_SC[_X_],
                      set_ptr->move_wards[_F_X_], _B_X_, MPI_COMM_WORLD,
                      &set_ptr->recv_request[_B_X_]);
        _MPI_Isend<T>(send_f, set_ptr->lat_3dim_SC[_X_],
                      set_ptr->move_wards[_F_X_], _F_X_, MPI_COMM_WORLD,
                      &set_ptr->send_request[_F_X_]);
        _MPI_Irecv<T>(recv_b, set_ptr->lat_3dim_SC[_X_],
                      set_ptr->move_wards[_B_X_], _F_X_, MPI_COMM_WORLD,
                      &set_ptr->recv_request[_F_X_]);
      }
    }
    {
      // y edge part
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
      if (set_ptr->host_params[_GRID_Y_] == 1 && !force_mpi) {
        // no comm
        // edge recv part
        wilson_dslash_y_recv<T><<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim,
                                  0, set_ptr->stream>>>(
            gauge, fermion_out, _device_params, set_ptr->device_send_vec[_F_Y_],
            set_ptr->device_send_vec[_B_Y_]);
      } else {
        // comm
        const void *send_b = device_mpi ? set_ptr->device_send_vec[_B_Y_]
                                        : set_ptr->host_send_vec[_B_Y_];
        const void *send_f = device_mpi ? set_ptr->device_send_vec[_F_Y_]
                                        : set_ptr->host_send_vec[_F_Y_];
        void *recv_f = device_mpi ? set_ptr->device_recv_vec[_F_Y_]
                                  : set_ptr->host_recv_vec[_F_Y_];
        void *recv_b = device_mpi ? set_ptr->device_recv_vec[_B_Y_]
                                  : set_ptr->host_recv_vec[_B_Y_];
        _MPI_Isend<T>(send_b, set_ptr->lat_3dim_SC[_Y_],
                      set_ptr->move_wards[_B_Y_], _B_Y_, MPI_COMM_WORLD,
                      &set_ptr->send_request[_B_Y_]);
        _MPI_Irecv<T>(recv_f, set_ptr->lat_3dim_SC[_Y_],
                      set_ptr->move_wards[_F_Y_], _B_Y_, MPI_COMM_WORLD,
                      &set_ptr->recv_request[_B_Y_]);
        _MPI_Isend<T>(send_f, set_ptr->lat_3dim_SC[_Y_],
                      set_ptr->move_wards[_F_Y_], _F_Y_, MPI_COMM_WORLD,
                      &set_ptr->send_request[_F_Y_]);
        _MPI_Irecv<T>(recv_b, set_ptr->lat_3dim_SC[_Y_],
                      set_ptr->move_wards[_B_Y_], _F_Y_, MPI_COMM_WORLD,
                      &set_ptr->recv_request[_F_Y_]);
      }
    }
    {
      // z edge part
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
      if (set_ptr->host_params[_GRID_Z_] == 1 && !force_mpi) {
        // no comm
        // edge recv part
        wilson_dslash_z_recv<T><<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim,
                                  0, set_ptr->stream>>>(
            gauge, fermion_out, _device_params, set_ptr->device_send_vec[_F_Z_],
            set_ptr->device_send_vec[_B_Z_]);
      } else {
        // comm
        const void *send_b = device_mpi ? set_ptr->device_send_vec[_B_Z_]
                                        : set_ptr->host_send_vec[_B_Z_];
        const void *send_f = device_mpi ? set_ptr->device_send_vec[_F_Z_]
                                        : set_ptr->host_send_vec[_F_Z_];
        void *recv_f = device_mpi ? set_ptr->device_recv_vec[_F_Z_]
                                  : set_ptr->host_recv_vec[_F_Z_];
        void *recv_b = device_mpi ? set_ptr->device_recv_vec[_B_Z_]
                                  : set_ptr->host_recv_vec[_B_Z_];
        _MPI_Isend<T>(send_b, set_ptr->lat_3dim_SC[_Z_],
                      set_ptr->move_wards[_B_Z_], _B_Z_, MPI_COMM_WORLD,
                      &set_ptr->send_request[_B_Z_]);
        _MPI_Irecv<T>(recv_f, set_ptr->lat_3dim_SC[_Z_],
                      set_ptr->move_wards[_F_Z_], _B_Z_, MPI_COMM_WORLD,
                      &set_ptr->recv_request[_B_Z_]);
        _MPI_Isend<T>(send_f, set_ptr->lat_3dim_SC[_Z_],
                      set_ptr->move_wards[_F_Z_], _F_Z_, MPI_COMM_WORLD,
                      &set_ptr->send_request[_F_Z_]);
        _MPI_Irecv<T>(recv_b, set_ptr->lat_3dim_SC[_Z_],
                      set_ptr->move_wards[_B_Z_], _F_Z_, MPI_COMM_WORLD,
                      &set_ptr->recv_request[_F_Z_]);
      }
    }
    {
      // t edge part
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
      if (set_ptr->host_params[_GRID_T_] == 1 && !force_mpi) {
        // no comm
        // edge recv part
        wilson_dslash_t_recv<T><<<set_ptr->gridDim_3dim[_T_], set_ptr->blockDim,
                                  0, set_ptr->stream>>>(
            gauge, fermion_out, _device_params, set_ptr->device_send_vec[_F_T_],
            set_ptr->device_send_vec[_B_T_]);
      } else {
        // comm
        const void *send_b = device_mpi ? set_ptr->device_send_vec[_B_T_]
                                        : set_ptr->host_send_vec[_B_T_];
        const void *send_f = device_mpi ? set_ptr->device_send_vec[_F_T_]
                                        : set_ptr->host_send_vec[_F_T_];
        void *recv_f = device_mpi ? set_ptr->device_recv_vec[_F_T_]
                                  : set_ptr->host_recv_vec[_F_T_];
        void *recv_b = device_mpi ? set_ptr->device_recv_vec[_B_T_]
                                  : set_ptr->host_recv_vec[_B_T_];
        _MPI_Isend<T>(send_b,
                      set_ptr->lat_3dim_SC[_T_] / _EVEN_ODD_,
                      set_ptr->move_wards[_B_T_], _B_T_, MPI_COMM_WORLD,
                      &set_ptr->send_request[_B_T_]);
        _MPI_Irecv<T>(recv_f,
                      set_ptr->lat_3dim_SC[_T_] / _EVEN_ODD_,
                      set_ptr->move_wards[_F_T_], _B_T_, MPI_COMM_WORLD,
                      &set_ptr->recv_request[_B_T_]);
        _MPI_Isend<T>(send_f,
                      set_ptr->lat_3dim_SC[_T_] / _EVEN_ODD_,
                      set_ptr->move_wards[_F_T_], _F_T_, MPI_COMM_WORLD,
                      &set_ptr->send_request[_F_T_]);
        _MPI_Irecv<T>(recv_b,
                      set_ptr->lat_3dim_SC[_T_] / _EVEN_ODD_,
                      set_ptr->move_wards[_B_T_], _F_T_, MPI_COMM_WORLD,
                      &set_ptr->recv_request[_F_T_]);
      }
    }
    if (set_ptr->host_params[_GRID_X_] != 1 || force_mpi) { // x part recv wait/h2d
      MPI_Wait(&set_ptr->recv_request[_B_X_], MPI_STATUS_IGNORE);
      if (!device_mpi)
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_F_X_], set_ptr->host_recv_vec[_F_X_],
            sizeof(T) * set_ptr->lat_3dim_SC[_X_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_X_]));
      MPI_Wait(&set_ptr->recv_request[_F_X_], MPI_STATUS_IGNORE);
      if (!device_mpi)
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_B_X_], set_ptr->host_recv_vec[_B_X_],
            sizeof(T) * set_ptr->lat_3dim_SC[_X_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_X_]));
    }
    if (set_ptr->host_params[_GRID_Y_] != 1 || force_mpi) { // y part recv wait/h2d
      MPI_Wait(&set_ptr->recv_request[_B_Y_], MPI_STATUS_IGNORE);
      if (!device_mpi)
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_F_Y_], set_ptr->host_recv_vec[_F_Y_],
            sizeof(T) * set_ptr->lat_3dim_SC[_Y_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_Y_]));
      MPI_Wait(&set_ptr->recv_request[_F_Y_], MPI_STATUS_IGNORE);
      if (!device_mpi)
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_B_Y_], set_ptr->host_recv_vec[_B_Y_],
            sizeof(T) * set_ptr->lat_3dim_SC[_Y_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_Y_]));
    }
    if (set_ptr->host_params[_GRID_Z_] != 1 || force_mpi) { // z part recv wait/h2d
      MPI_Wait(&set_ptr->recv_request[_B_Z_], MPI_STATUS_IGNORE);
      if (!device_mpi)
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_F_Z_], set_ptr->host_recv_vec[_F_Z_],
            sizeof(T) * set_ptr->lat_3dim_SC[_Z_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_Z_]));
      MPI_Wait(&set_ptr->recv_request[_F_Z_], MPI_STATUS_IGNORE);
      if (!device_mpi)
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_B_Z_], set_ptr->host_recv_vec[_B_Z_],
            sizeof(T) * set_ptr->lat_3dim_SC[_Z_], cudaMemcpyHostToDevice,
            set_ptr->stream_dims[_Z_]));
    }
    if (set_ptr->host_params[_GRID_T_] != 1 || force_mpi) { // t part recv wait/h2d
      MPI_Wait(&set_ptr->recv_request[_B_T_], MPI_STATUS_IGNORE);
      if (!device_mpi)
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_F_T_], set_ptr->host_recv_vec[_F_T_],
            sizeof(T) * set_ptr->lat_3dim_SC[_T_] / _EVEN_ODD_,
            cudaMemcpyHostToDevice, set_ptr->stream_dims[_T_]));
      MPI_Wait(&set_ptr->recv_request[_F_T_], MPI_STATUS_IGNORE);
      if (!device_mpi)
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->device_recv_vec[_B_T_], set_ptr->host_recv_vec[_B_T_],
            sizeof(T) * set_ptr->lat_3dim_SC[_T_] / _EVEN_ODD_,
            cudaMemcpyHostToDevice, set_ptr->stream_dims[_T_]));
    }
    {
      // edge recv part
      if (set_ptr->host_params[_GRID_X_] != 1 || force_mpi) { // x part recv
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
        wilson_dslash_x_recv<T><<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim,
                                  0, set_ptr->stream>>>(
            gauge, fermion_out, _device_params, set_ptr->device_recv_vec[_B_X_],
            set_ptr->device_recv_vec[_F_X_]);
      }
      if (set_ptr->host_params[_GRID_Y_] != 1 || force_mpi) { // y part recv
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
        wilson_dslash_y_recv<T><<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim,
                                  0, set_ptr->stream>>>(
            gauge, fermion_out, _device_params, set_ptr->device_recv_vec[_B_Y_],
            set_ptr->device_recv_vec[_F_Y_]);
      }
      if (set_ptr->host_params[_GRID_Z_] != 1 || force_mpi) { // z part recv
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
        wilson_dslash_z_recv<T><<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim,
                                  0, set_ptr->stream>>>(
            gauge, fermion_out, _device_params, set_ptr->device_recv_vec[_B_Z_],
            set_ptr->device_recv_vec[_F_Z_]);
      }
      if (set_ptr->host_params[_GRID_T_] != 1 || force_mpi) { // t part recv
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
        wilson_dslash_t_recv<T><<<set_ptr->gridDim_3dim[_T_], set_ptr->blockDim,
                                  0, set_ptr->stream>>>(
            gauge, fermion_out, _device_params, set_ptr->device_recv_vec[_B_T_],
            set_ptr->device_recv_vec[_F_T_]);
      }
    }
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream)); // needed
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_X_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Y_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_Z_]));
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream_dims[_T_]));
    // NOTE: run_mpi_non_block uses MPI_Isend/Irecv, wait for sends to complete.
    MPI_Wait(&set_ptr->send_request[_B_X_], MPI_STATUS_IGNORE);
    MPI_Wait(&set_ptr->send_request[_F_X_], MPI_STATUS_IGNORE);
    MPI_Wait(&set_ptr->send_request[_B_Y_], MPI_STATUS_IGNORE);
    MPI_Wait(&set_ptr->send_request[_F_Y_], MPI_STATUS_IGNORE);
    MPI_Wait(&set_ptr->send_request[_B_Z_], MPI_STATUS_IGNORE);
    MPI_Wait(&set_ptr->send_request[_F_Z_], MPI_STATUS_IGNORE);
    MPI_Wait(&set_ptr->send_request[_B_T_], MPI_STATUS_IGNORE);
    MPI_Wait(&set_ptr->send_request[_F_T_], MPI_STATUS_IGNORE);
  }
  void run_mpi(void *fermion_out, void *fermion_in, void *gauge,
               void *_device_params) {
    const bool force_mpi = _WILSON_AND_LAPLACIAN_TEST_SINGLE_IN_MULTI_ != 0;
    const bool device_mpi = qcu_mpi_can_use_buffer<T>(
        set_ptr->device_send_vec[_B_X_]);
    // ====================================================================
    // SINGLE-RANK FAST PATH (2026-08-02)
    // --------------------------------------------------------------------
    // On a 1x1x1x1 process grid there is NO inter-rank halo exchange: the
    // periodic boundary wraps within the rank.  The legacy path below still
    // launches the full send/inside/recv kernel decomposition across 5
    // streams and inserts ~9 cudaStreamSynchronize calls.  Each sync costs
    // ~170 us on this WSL2/V100 setup, so a single dslash was ~4.5 ms.
    //
    // For grid == [1,1,1,1] we launch every kernel (send x/y/z/t, inside,
    // recv x/y/z/t) on the MAIN stream in dependency order with NO
    // intermediate syncs and one sync at the end.  Same-stream launches
    // serialize by construction, so correctness is identical to the
    // multi-stream path (recv reads device_send_vec written by send; recv
    // adds into fermion_out written by inside).  This turns ~1.6 ms of
    // syncs into ~10 us of launches.
    // ====================================================================
    const bool single_rank =
        !force_mpi &&
        set_ptr->host_params[_GRID_X_] == 1 &&
        set_ptr->host_params[_GRID_Y_] == 1 &&
        set_ptr->host_params[_GRID_Z_] == 1 &&
        set_ptr->host_params[_GRID_T_] == 1;

    // The complete Wilson kernel already implements the periodic compact
    // parity layout used by this class (LatticeSet stores T/2 and the kernel
    // maps the t-neighbour with the spatial checkerboard parity).  On a
    // single rank there is no halo to overlap, so the send/inside/recv split
    // only adds launch and stream bookkeeping overhead.  Keep the split path
    // below for multi-rank and forced-MPI tests, where its halo buffers are
    // required.  In the normal fast path all consumers are on the main
    // stream; skip_final_sync_ therefore has the same meaning as below.
    if (single_rank) {
      wilson_dslash<T><<<set_ptr->gridDim, set_ptr->blockDim, 0,
                         set_ptr->stream>>>(
          gauge, fermion_in, fermion_out, _device_params);
      if (!skip_final_sync_)
        checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
      return;
    }

    // Lambda: sync a stream, but ONLY in the multi-rank path (the
    // single-rank path serializes everything on the main stream).
    auto sync_if_multi = [&](cudaStream_t s) {
      if (!single_rank) checkCudaErrors(cudaStreamSynchronize(s));
    };
    // Resolve per-dimension stream (single_rank -> main stream).
    auto dim_stream = [&](int d) {
      return single_rank ? set_ptr->stream : set_ptr->stream_dims[d];
    };
    sync_if_multi(set_ptr->stream);
    sync_if_multi(set_ptr->stream_dims[_X_]);
    sync_if_multi(set_ptr->stream_dims[_Y_]);
    sync_if_multi(set_ptr->stream_dims[_Z_]);
    sync_if_multi(set_ptr->stream_dims[_T_]);
    { // edge send part
      wilson_dslash_x_send<T><<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim,
                                0, dim_stream(_X_)>>>(
          gauge, fermion_in, _device_params, set_ptr->device_send_vec[_B_X_],
          set_ptr->device_send_vec[_F_X_]);
      // NOTE: for single_rank, dim_stream() returns the main stream, so all
      // send kernels serialize with the inside/recv kernels (no halo syncs).
      // (y/z/t sends follow the same pattern: stream = dim_stream(_Y_/Z_/T_).)
      if (!device_mpi && (set_ptr->host_params[_GRID_X_] != 1 ||
                          force_mpi)) { // x part d2h
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_B_X_], set_ptr->device_send_vec[_B_X_],
            sizeof(T) * set_ptr->lat_3dim_SC[_X_], cudaMemcpyDeviceToHost,
            set_ptr->stream_dims[_X_]));
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_F_X_], set_ptr->device_send_vec[_F_X_],
            sizeof(T) * set_ptr->lat_3dim_SC[_X_], cudaMemcpyDeviceToHost,
            set_ptr->stream_dims[_X_]));
      }
      wilson_dslash_y_send<T><<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim,
                                0, dim_stream(_Y_)>>>(
          gauge, fermion_in, _device_params, set_ptr->device_send_vec[_B_Y_],
          set_ptr->device_send_vec[_F_Y_]);
      if (!device_mpi && (set_ptr->host_params[_GRID_Y_] != 1 ||
                          force_mpi)) { // y part d2h
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_B_Y_], set_ptr->device_send_vec[_B_Y_],
            sizeof(T) * set_ptr->lat_3dim_SC[_Y_], cudaMemcpyDeviceToHost,
            set_ptr->stream_dims[_Y_]));
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_F_Y_], set_ptr->device_send_vec[_F_Y_],
            sizeof(T) * set_ptr->lat_3dim_SC[_Y_], cudaMemcpyDeviceToHost,
            set_ptr->stream_dims[_Y_]));
      }
      wilson_dslash_z_send<T><<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim,
                                0, dim_stream(_Z_)>>>(
          gauge, fermion_in, _device_params, set_ptr->device_send_vec[_B_Z_],
          set_ptr->device_send_vec[_F_Z_]);
      if (!device_mpi && (set_ptr->host_params[_GRID_Z_] != 1 ||
                          force_mpi)) { // z part d2h
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_B_Z_], set_ptr->device_send_vec[_B_Z_],
            sizeof(T) * set_ptr->lat_3dim_SC[_Z_], cudaMemcpyDeviceToHost,
            set_ptr->stream_dims[_Z_]));
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_F_Z_], set_ptr->device_send_vec[_F_Z_],
            sizeof(T) * set_ptr->lat_3dim_SC[_Z_], cudaMemcpyDeviceToHost,
            set_ptr->stream_dims[_Z_]));
      }
      wilson_dslash_t_send<T><<<set_ptr->gridDim_3dim[_T_], set_ptr->blockDim,
                                0, dim_stream(_T_)>>>(
          gauge, fermion_in, _device_params, set_ptr->device_send_vec[_B_T_],
          set_ptr->device_send_vec[_F_T_]);
      if (!device_mpi && (set_ptr->host_params[_GRID_T_] != 1 ||
                          force_mpi)) { // t part d2h
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_B_T_], set_ptr->device_send_vec[_B_T_],
            sizeof(T) * set_ptr->lat_3dim_SC[_T_] / _EVEN_ODD_,
            cudaMemcpyDeviceToHost, set_ptr->stream_dims[_T_]));
        checkCudaErrors(cudaMemcpyAsync(
            set_ptr->host_send_vec[_F_T_], set_ptr->device_send_vec[_F_T_],
            sizeof(T) * set_ptr->lat_3dim_SC[_T_] / _EVEN_ODD_,
            cudaMemcpyDeviceToHost, set_ptr->stream_dims[_T_]));
      }
    }
    { // inside compute part ans wait
      sync_if_multi(set_ptr->stream); // needed
      wilson_dslash_inside<T>
          <<<set_ptr->gridDim, set_ptr->blockDim, 0, set_ptr->stream>>>(
              gauge, fermion_in, fermion_out, _device_params);
    }
    {
      // x edge part
      sync_if_multi(set_ptr->stream_dims[_X_]);
      if (set_ptr->host_params[_GRID_X_] == 1 && !force_mpi) {
        // no comm
        // edge recv part
        wilson_dslash_x_recv<T><<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim,
                                  0, set_ptr->stream>>>(
            gauge, fermion_out, _device_params, set_ptr->device_send_vec[_F_X_],
            set_ptr->device_send_vec[_B_X_]);
      } else {
        // comm
        const void *send_b = device_mpi ? set_ptr->device_send_vec[_B_X_]
                                        : set_ptr->host_send_vec[_B_X_];
        const void *send_f = device_mpi ? set_ptr->device_send_vec[_F_X_]
                                        : set_ptr->host_send_vec[_F_X_];
        void *recv_f = device_mpi ? set_ptr->device_recv_vec[_F_X_]
                                  : set_ptr->host_recv_vec[_F_X_];
        void *recv_b = device_mpi ? set_ptr->device_recv_vec[_B_X_]
                                  : set_ptr->host_recv_vec[_B_X_];
        _MPI_Sendrecv<T>(send_b,
                         set_ptr->lat_3dim_SC[_X_], set_ptr->move_wards[_B_X_],
                         _B_X_, recv_f,
                         set_ptr->lat_3dim_SC[_X_], set_ptr->move_wards[_F_X_],
                         _B_X_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        _MPI_Sendrecv<T>(send_f,
                         set_ptr->lat_3dim_SC[_X_], set_ptr->move_wards[_F_X_],
                         _F_X_, recv_b,
                         set_ptr->lat_3dim_SC[_X_], set_ptr->move_wards[_B_X_],
                         _F_X_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
    {
      // y edge part
      sync_if_multi(set_ptr->stream_dims[_Y_]);
      if (set_ptr->host_params[_GRID_Y_] == 1 && !force_mpi) {
        // no comm
        // edge recv part
        wilson_dslash_y_recv<T><<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim,
                                  0, set_ptr->stream>>>(
            gauge, fermion_out, _device_params, set_ptr->device_send_vec[_F_Y_],
            set_ptr->device_send_vec[_B_Y_]);
      } else {
        // comm
        const void *send_b = device_mpi ? set_ptr->device_send_vec[_B_Y_]
                                        : set_ptr->host_send_vec[_B_Y_];
        const void *send_f = device_mpi ? set_ptr->device_send_vec[_F_Y_]
                                        : set_ptr->host_send_vec[_F_Y_];
        void *recv_f = device_mpi ? set_ptr->device_recv_vec[_F_Y_]
                                  : set_ptr->host_recv_vec[_F_Y_];
        void *recv_b = device_mpi ? set_ptr->device_recv_vec[_B_Y_]
                                  : set_ptr->host_recv_vec[_B_Y_];
        _MPI_Sendrecv<T>(send_b,
                         set_ptr->lat_3dim_SC[_Y_], set_ptr->move_wards[_B_Y_],
                         _B_Y_, recv_f,
                         set_ptr->lat_3dim_SC[_Y_], set_ptr->move_wards[_F_Y_],
                         _B_Y_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        _MPI_Sendrecv<T>(send_f,
                         set_ptr->lat_3dim_SC[_Y_], set_ptr->move_wards[_F_Y_],
                         _F_Y_, recv_b,
                         set_ptr->lat_3dim_SC[_Y_], set_ptr->move_wards[_B_Y_],
                         _F_Y_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
    {
      // z edge part
      sync_if_multi(set_ptr->stream_dims[_Z_]);
      if (set_ptr->host_params[_GRID_Z_] == 1 && !force_mpi) {
        // no comm
        // edge recv part
        wilson_dslash_z_recv<T><<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim,
                                  0, set_ptr->stream>>>(
            gauge, fermion_out, _device_params, set_ptr->device_send_vec[_F_Z_],
            set_ptr->device_send_vec[_B_Z_]);
      } else {
        // comm
        const void *send_b = device_mpi ? set_ptr->device_send_vec[_B_Z_]
                                        : set_ptr->host_send_vec[_B_Z_];
        const void *send_f = device_mpi ? set_ptr->device_send_vec[_F_Z_]
                                        : set_ptr->host_send_vec[_F_Z_];
        void *recv_f = device_mpi ? set_ptr->device_recv_vec[_F_Z_]
                                  : set_ptr->host_recv_vec[_F_Z_];
        void *recv_b = device_mpi ? set_ptr->device_recv_vec[_B_Z_]
                                  : set_ptr->host_recv_vec[_B_Z_];
        _MPI_Sendrecv<T>(send_b,
                         set_ptr->lat_3dim_SC[_Z_], set_ptr->move_wards[_B_Z_],
                         _B_Z_, recv_f,
                         set_ptr->lat_3dim_SC[_Z_], set_ptr->move_wards[_F_Z_],
                         _B_Z_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        _MPI_Sendrecv<T>(send_f,
                         set_ptr->lat_3dim_SC[_Z_], set_ptr->move_wards[_F_Z_],
                         _F_Z_, recv_b,
                         set_ptr->lat_3dim_SC[_Z_], set_ptr->move_wards[_B_Z_],
                         _F_Z_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
    {
      // t edge part
      sync_if_multi(set_ptr->stream_dims[_T_]);
      if (set_ptr->host_params[_GRID_T_] == 1 && !force_mpi) {
        // no comm
        // edge recv part
        wilson_dslash_t_recv<T><<<set_ptr->gridDim_3dim[_T_], set_ptr->blockDim,
                                  0, set_ptr->stream>>>(
            gauge, fermion_out, _device_params, set_ptr->device_send_vec[_F_T_],
            set_ptr->device_send_vec[_B_T_]);
      } else {
        // comm
        const void *send_b = device_mpi ? set_ptr->device_send_vec[_B_T_]
                                        : set_ptr->host_send_vec[_B_T_];
        const void *send_f = device_mpi ? set_ptr->device_send_vec[_F_T_]
                                        : set_ptr->host_send_vec[_F_T_];
        void *recv_f = device_mpi ? set_ptr->device_recv_vec[_F_T_]
                                  : set_ptr->host_recv_vec[_F_T_];
        void *recv_b = device_mpi ? set_ptr->device_recv_vec[_B_T_]
                                  : set_ptr->host_recv_vec[_B_T_];
        _MPI_Sendrecv<T>(
            send_b,
            set_ptr->lat_3dim_SC[_T_] / _EVEN_ODD_, set_ptr->move_wards[_B_T_],
            _B_T_, recv_f,
            set_ptr->lat_3dim_SC[_T_] / _EVEN_ODD_, set_ptr->move_wards[_F_T_],
            _B_T_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        _MPI_Sendrecv<T>(
            send_f,
            set_ptr->lat_3dim_SC[_T_] / _EVEN_ODD_, set_ptr->move_wards[_F_T_],
            _F_T_, recv_b,
            set_ptr->lat_3dim_SC[_T_] / _EVEN_ODD_, set_ptr->move_wards[_B_T_],
            _F_T_, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
    if (!device_mpi && (set_ptr->host_params[_GRID_X_] != 1 ||
                        force_mpi)) { // x part h2d
      checkCudaErrors(cudaMemcpyAsync(
          set_ptr->device_recv_vec[_F_X_], set_ptr->host_recv_vec[_F_X_],
          sizeof(T) * set_ptr->lat_3dim_SC[_X_], cudaMemcpyHostToDevice,
          set_ptr->stream_dims[_X_]));
      checkCudaErrors(cudaMemcpyAsync(
          set_ptr->device_recv_vec[_B_X_], set_ptr->host_recv_vec[_B_X_],
          sizeof(T) * set_ptr->lat_3dim_SC[_X_], cudaMemcpyHostToDevice,
          set_ptr->stream_dims[_X_]));
    }
    if (!device_mpi && (set_ptr->host_params[_GRID_Y_] != 1 ||
                        force_mpi)) { // y part h2d
      checkCudaErrors(cudaMemcpyAsync(
          set_ptr->device_recv_vec[_F_Y_], set_ptr->host_recv_vec[_F_Y_],
          sizeof(T) * set_ptr->lat_3dim_SC[_Y_], cudaMemcpyHostToDevice,
          set_ptr->stream_dims[_Y_]));
      checkCudaErrors(cudaMemcpyAsync(
          set_ptr->device_recv_vec[_B_Y_], set_ptr->host_recv_vec[_B_Y_],
          sizeof(T) * set_ptr->lat_3dim_SC[_Y_], cudaMemcpyHostToDevice,
          set_ptr->stream_dims[_Y_]));
    }
    if (!device_mpi && (set_ptr->host_params[_GRID_Z_] != 1 ||
                        force_mpi)) { // z part h2d
      checkCudaErrors(cudaMemcpyAsync(
          set_ptr->device_recv_vec[_F_Z_], set_ptr->host_recv_vec[_F_Z_],
          sizeof(T) * set_ptr->lat_3dim_SC[_Z_], cudaMemcpyHostToDevice,
          set_ptr->stream_dims[_Z_]));
      checkCudaErrors(cudaMemcpyAsync(
          set_ptr->device_recv_vec[_B_Z_], set_ptr->host_recv_vec[_B_Z_],
          sizeof(T) * set_ptr->lat_3dim_SC[_Z_], cudaMemcpyHostToDevice,
          set_ptr->stream_dims[_Z_]));
    }
    if (!device_mpi && (set_ptr->host_params[_GRID_T_] != 1 ||
                        force_mpi)) { // t part h2d
      checkCudaErrors(cudaMemcpyAsync(
          set_ptr->device_recv_vec[_F_T_], set_ptr->host_recv_vec[_F_T_],
          sizeof(T) * set_ptr->lat_3dim_SC[_T_] / _EVEN_ODD_,
          cudaMemcpyHostToDevice, set_ptr->stream_dims[_T_]));
      checkCudaErrors(cudaMemcpyAsync(
          set_ptr->device_recv_vec[_B_T_], set_ptr->host_recv_vec[_B_T_],
          sizeof(T) * set_ptr->lat_3dim_SC[_T_] / _EVEN_ODD_,
          cudaMemcpyHostToDevice, set_ptr->stream_dims[_T_]));
    }
    {
      // edge recv part
      if (set_ptr->host_params[_GRID_X_] != 1 || force_mpi) { // x part recv
        sync_if_multi(set_ptr->stream_dims[_X_]);
        wilson_dslash_x_recv<T><<<set_ptr->gridDim_3dim[_X_], set_ptr->blockDim,
                                  0, set_ptr->stream>>>(
            gauge, fermion_out, _device_params, set_ptr->device_recv_vec[_B_X_],
            set_ptr->device_recv_vec[_F_X_]);
      }
      if (set_ptr->host_params[_GRID_Y_] != 1 || force_mpi) { // y part recv
        sync_if_multi(set_ptr->stream_dims[_Y_]);
        wilson_dslash_y_recv<T><<<set_ptr->gridDim_3dim[_Y_], set_ptr->blockDim,
                                  0, set_ptr->stream>>>(
            gauge, fermion_out, _device_params, set_ptr->device_recv_vec[_B_Y_],
            set_ptr->device_recv_vec[_F_Y_]);
      }
      if (set_ptr->host_params[_GRID_Z_] != 1 || force_mpi) { // z part recv
        sync_if_multi(set_ptr->stream_dims[_Z_]);
        wilson_dslash_z_recv<T><<<set_ptr->gridDim_3dim[_Z_], set_ptr->blockDim,
                                  0, set_ptr->stream>>>(
            gauge, fermion_out, _device_params, set_ptr->device_recv_vec[_B_Z_],
            set_ptr->device_recv_vec[_F_Z_]);
      }
      if (set_ptr->host_params[_GRID_T_] != 1 || force_mpi) { // t part recv
        sync_if_multi(set_ptr->stream_dims[_T_]);
        wilson_dslash_t_recv<T><<<set_ptr->gridDim_3dim[_T_], set_ptr->blockDim,
                                  0, set_ptr->stream>>>(
            gauge, fermion_out, _device_params, set_ptr->device_recv_vec[_B_T_],
            set_ptr->device_recv_vec[_F_T_]);
      }
    }
    sync_if_multi(set_ptr->stream); // needed
    sync_if_multi(set_ptr->stream_dims[_X_]);
    sync_if_multi(set_ptr->stream_dims[_Y_]);
    sync_if_multi(set_ptr->stream_dims[_Z_]);
    sync_if_multi(set_ptr->stream_dims[_T_]);
    // NOTE: run_mpi uses blocking MPI_Sendrecv, no MPI_Wait needed here.
    // Single-rank path: all send/inside/recv kernels were launched on the
    // main stream with no intermediate syncs; one sync here guarantees the
    // dslash output is complete before the caller reuses any buffer.
    // The MG fast iteration sets skip_final_sync_ to rely on main-stream
    // ordering instead (saves ~170 us per run_mpi).
    if (single_rank && !skip_final_sync_)
      checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
  void _run(void *fermion_out, void *fermion_in, void *gauge,
            void *_device_params) {
    run_mpi(fermion_out, fermion_in, gauge, _device_params);
    // run_mpi_non_block(fermion_out, fermion_in, gauge, _device_params);
    err = cudaGetLastError();
    checkCudaErrors(err);
  }
  void run(void *fermion_out, void *fermion_in, void *gauge) {
    _run(fermion_out, fermion_in, gauge, set_ptr->device_params);
  }
  void run_eo(void *fermion_out, void *fermion_in, void *gauge) {
    _run(fermion_out, fermion_in, gauge, set_ptr->device_params_even_no_dag);
  }
  void run_oe(void *fermion_out, void *fermion_in, void *gauge) {
    _run(fermion_out, fermion_in, gauge, set_ptr->device_params_odd_no_dag);
  }
  void run_eo_dag(void *fermion_out, void *fermion_in, void *gauge) {
    _run(fermion_out, fermion_in, gauge, set_ptr->device_params_even_dag);
  }
  void run_oe_dag(void *fermion_out, void *fermion_in, void *gauge) {
    _run(fermion_out, fermion_in, gauge, set_ptr->device_params_odd_dag);
  }
  void run_test(void *fermion_out, void *fermion_in, void *gauge) {
    auto start = std::chrono::high_resolution_clock::now();
    run(fermion_out, fermion_in, gauge);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    printf("multi-gpu wilson dslash total time: (without malloc free memcpy) "
           ":%.9lf "
           "sec\n",
           double(duration) / 1e9);
  }
};
} // namespace qcu
#endif
