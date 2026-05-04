#ifndef _LATTICE_SET_H
#define _LATTICE_SET_H
#pragma once
#include "./define.h"
#include "./lattice_cuda.h"
namespace qcu {
template <typename T>
__global__ void give_param(void *device_param, int vals_index, int val);
template <typename T> struct LatticeSet {
  int lat_2dim[_2DIM_];
  int lat_3dim[_3DIM_];
  int lat_4dim;
  int lat_3dim_C[_3DIM_];
  int lat_4dim_C;
  int lat_3dim_Half_SC[_3DIM_];
  int lat_3dim_SC[_3DIM_];
  int lat_4dim_SC;
  int lat_4dim_SCSC;
  int lat_4dim_DCC;
  dim3 gridDim_3dim[_3DIM_];
  dim3 gridDim_2dim[_2DIM_];
  dim3 gridDim;
  dim3 blockDim;
  cublasHandle_t cublasH;
  cudaStream_t stream;
  cublasHandle_t cublasHs[_DIM_];
  cudaStream_t streams[_DIM_];
  cudaStream_t stream_dims[_DIM_];
  cudaStream_t stream_memcpy[_WARDS_];
  float time;
  cudaEvent_t start, stop;
  cudaError_t err;
  int move[_BF_];
  int move_wards[_WARDS_ + _WARDS_2DIM_];
  int grid_1dim[_1DIM_];
  int grid_2dim[_2DIM_];
  int grid_3dim[_3DIM_];
  int grid_index_1dim[_1DIM_];
  MPI_Request send_request[_WARDS_];
  MPI_Request recv_request[_WARDS_];
  int host_params[_PARAMS_SIZE_];
  T host_argv[_ARGV_SIZE_];
  void *device_send_vec[_WARDS_];
  void *device_recv_vec[_WARDS_];
  void *host_send_vec[_WARDS_];
  void *host_recv_vec[_WARDS_];
  void *device_u_1dim_send_vec[_WARDS_];
  void *device_u_1dim_recv_vec[_WARDS_];
  void *device_u_2dim_send_vec[_2DIM_ * _BF_ * _BF_];
  void *device_u_2dim_recv_vec[_2DIM_ * _BF_ * _BF_];
  void *host_u_1dim_send_vec[_WARDS_];
  void *host_u_1dim_recv_vec[_WARDS_];
  void *host_u_2dim_send_vec[_2DIM_ * _BF_ * _BF_];
  void *host_u_2dim_recv_vec[_2DIM_ * _BF_ * _BF_];
  void *device_params;
  void *device_params_even_no_dag;
  void *device_params_odd_no_dag;
  void *device_params_even_dag;
  void *device_params_odd_dag;
  void *device_vec0;
  void *device_vec1;
  void *device_vec2;
  void *device_vals;
  void give(void *_params, void *_argv) {
    for (int i = 0; i < _PARAMS_SIZE_; i++) {
      host_params[i] = static_cast<int *>(_params)[i];
    }
    if (host_params[_SET_PLAN_] ==
        _SET_PLAN_N_2_) // just for laplacian // no even-odd
    {
      printf("just for laplacian, lat_t = 1, lat_d = 3, no even-odd\n");
      host_params[_LAT_T_] = 1;
    } else {
      host_params[_LAT_T_] /= _EVEN_ODD_; // even-odd
    }
    host_params[_LAT_XYZT_] = host_params[_LAT_X_] * host_params[_LAT_Y_] *
                              host_params[_LAT_Z_] * host_params[_LAT_T_];
    if (static_cast<int *>(_params)[_PARITY_] == _EVEN_) {
      host_params[_PARITY_] = _EVEN_;
    } else if (static_cast<int *>(_params)[_PARITY_] == _ODD_) {
      host_params[_PARITY_] = _ODD_;
    } else {
      printf("error in parity\n");
      host_params[_PARITY_] = _EVEN_;
    }
    if (static_cast<int *>(_params)[_DAGGER_] == _NO_USE_) {
      host_params[_DAGGER_] = _NO_USE_;
    } else if (static_cast<int *>(_params)[_DAGGER_] == _USE_) {
      host_params[_DAGGER_] = _USE_;
    } else {
      printf("error in dagger\n");
      host_params[_DAGGER_] = _NO_USE_;
    }
    for (int i = 0; i < _ARGV_SIZE_; i++) {
      host_argv[i] = static_cast<T *>(_argv)[i];
    }
  }
  void init() {
    {   // basic set
      { // give params
        blockDim = _BLOCK_SIZE_;
        checkMpiErrors(
            MPI_Comm_rank(MPI_COMM_WORLD, host_params + _NODE_RANK_));
        checkMpiErrors(
            MPI_Comm_size(MPI_COMM_WORLD, host_params + _NODE_SIZE_));
        grid_1dim[_X_] = host_params[_GRID_X_];
        grid_1dim[_Y_] = host_params[_GRID_Y_];
        grid_1dim[_Z_] = host_params[_GRID_Z_];
        grid_1dim[_T_] = host_params[_GRID_T_];
        grid_2dim[_XY_] = host_params[_GRID_X_] * host_params[_GRID_Y_];
        grid_2dim[_XZ_] = host_params[_GRID_X_] * host_params[_GRID_Z_];
        grid_2dim[_XT_] = host_params[_GRID_X_] * host_params[_GRID_T_];
        grid_2dim[_YZ_] = host_params[_GRID_Y_] * host_params[_GRID_Z_];
        grid_2dim[_YT_] = host_params[_GRID_Y_] * host_params[_GRID_T_];
        grid_2dim[_ZT_] = host_params[_GRID_Z_] * host_params[_GRID_T_];
        grid_3dim[_YZT_] = host_params[_GRID_Y_] * host_params[_GRID_Z_] *
                           host_params[_GRID_T_];
        grid_3dim[_XZT_] = host_params[_GRID_X_] * host_params[_GRID_Z_] *
                           host_params[_GRID_T_];
        grid_3dim[_XYT_] = host_params[_GRID_X_] * host_params[_GRID_Y_] *
                           host_params[_GRID_T_];
        grid_3dim[_XYZ_] = host_params[_GRID_X_] * host_params[_GRID_Y_] *
                           host_params[_GRID_Z_];
        { // splite by [x,y,z,t]
          int tmp;
          tmp = host_params[_NODE_RANK_];
          grid_index_1dim[_X_] = tmp / grid_3dim[_YZT_];
          tmp -= grid_index_1dim[_X_] * grid_3dim[_YZT_];
          grid_index_1dim[_Y_] = tmp / grid_2dim[_ZT_];
          tmp -= grid_index_1dim[_Y_] * grid_2dim[_ZT_];
          grid_index_1dim[_Z_] = tmp / grid_1dim[_T_];
          grid_index_1dim[_T_] = tmp - grid_index_1dim[_Z_] * grid_1dim[_T_];
        }
        lat_2dim[_XY_] = host_params[_LAT_X_] * host_params[_LAT_Y_];
        lat_2dim[_XZ_] = host_params[_LAT_X_] * host_params[_LAT_Z_];
        lat_2dim[_XT_] = host_params[_LAT_X_] * host_params[_LAT_T_];
        lat_2dim[_YZ_] = host_params[_LAT_Y_] * host_params[_LAT_Z_];
        lat_2dim[_YT_] = host_params[_LAT_Y_] * host_params[_LAT_T_];
        lat_2dim[_ZT_] = host_params[_LAT_Z_] * host_params[_LAT_T_];
        gridDim_2dim[_XY_] = lat_2dim[_XY_] / _BLOCK_SIZE_;
        gridDim_2dim[_XZ_] = lat_2dim[_XZ_] / _BLOCK_SIZE_;
        gridDim_2dim[_XT_] = lat_2dim[_XT_] / _BLOCK_SIZE_;
        gridDim_2dim[_YZ_] = lat_2dim[_YZ_] / _BLOCK_SIZE_;
        gridDim_2dim[_YT_] = lat_2dim[_YT_] / _BLOCK_SIZE_;
        gridDim_2dim[_ZT_] = lat_2dim[_ZT_] / _BLOCK_SIZE_;
        lat_3dim[_YZT_] =
            host_params[_LAT_Y_] * host_params[_LAT_Z_] * host_params[_LAT_T_];
        lat_3dim[_XZT_] =
            host_params[_LAT_X_] * host_params[_LAT_Z_] * host_params[_LAT_T_];
        lat_3dim[_XYT_] =
            host_params[_LAT_X_] * host_params[_LAT_Y_] * host_params[_LAT_T_];
        lat_3dim[_XYZ_] =
            host_params[_LAT_X_] * host_params[_LAT_Y_] * host_params[_LAT_Z_];
        gridDim_3dim[_YZT_] = lat_3dim[_YZT_] / _BLOCK_SIZE_;
        gridDim_3dim[_XZT_] = lat_3dim[_XZT_] / _BLOCK_SIZE_;
        gridDim_3dim[_XYT_] = lat_3dim[_XYT_] / _BLOCK_SIZE_;
        gridDim_3dim[_XYZ_] = lat_3dim[_XYZ_] / _BLOCK_SIZE_;
        lat_4dim = host_params[_LAT_XYZT_];
        lat_4dim_C = lat_4dim * _LAT_C_;
        lat_4dim_SC = lat_4dim * _LAT_SC_;
        lat_4dim_SCSC = lat_4dim * _LAT_SCSC_;
        lat_4dim_DCC = lat_4dim * _LAT_CCD_;
        gridDim = lat_4dim / _BLOCK_SIZE_;
        for (int i = 0; i < _DIM_; i++) {
          lat_3dim_C[i] = lat_3dim[i] * _LAT_C_;
          lat_3dim_Half_SC[i] = lat_3dim[i] * _LAT_HALF_SC_;
          lat_3dim_SC[i] = lat_3dim[i] * _LAT_SC_;
        }
      }
      { // give move wards
        move_backward(move_wards[_B_X_], grid_index_1dim[_X_], grid_1dim[_X_]);
        move_backward(move_wards[_B_Y_], grid_index_1dim[_Y_], grid_1dim[_Y_]);
        move_backward(move_wards[_B_Z_], grid_index_1dim[_Z_], grid_1dim[_Z_]);
        move_backward(move_wards[_B_T_], grid_index_1dim[_T_], grid_1dim[_T_]);
        move_forward(move_wards[_F_X_], grid_index_1dim[_X_], grid_1dim[_X_]);
        move_forward(move_wards[_F_Y_], grid_index_1dim[_Y_], grid_1dim[_Y_]);
        move_forward(move_wards[_F_Z_], grid_index_1dim[_Z_], grid_1dim[_Z_]);
        move_forward(move_wards[_F_T_], grid_index_1dim[_T_], grid_1dim[_T_]);
      }
      { // splite by [x,y,z,t]
        move_wards[_B_T_] = host_params[_NODE_RANK_] + move_wards[_B_T_];
        move_wards[_B_Z_] =
            host_params[_NODE_RANK_] + move_wards[_B_Z_] * grid_1dim[_T_];
        move_wards[_B_Y_] =
            host_params[_NODE_RANK_] + move_wards[_B_Y_] * grid_2dim[_ZT_];
        move_wards[_B_X_] =
            host_params[_NODE_RANK_] + move_wards[_B_X_] * grid_3dim[_YZT_];
        move_wards[_F_T_] = host_params[_NODE_RANK_] + move_wards[_F_T_];
        move_wards[_F_Z_] =
            host_params[_NODE_RANK_] + move_wards[_F_Z_] * grid_1dim[_T_];
        move_wards[_F_Y_] =
            host_params[_NODE_RANK_] + move_wards[_F_Y_] * grid_2dim[_ZT_];
        move_wards[_F_X_] =
            host_params[_NODE_RANK_] + move_wards[_F_X_] * grid_3dim[_YZT_];
      }
      {   // give move wards
        { // splite by [x,y,z,t]
          int tmp;
          { // BB
            move_backward(tmp, grid_index_1dim[_Y_], grid_1dim[_Y_]);
            move_wards[_BX_BY_] = move_wards[_B_X_] + tmp * grid_2dim[_ZT_];
            move_backward(tmp, grid_index_1dim[_Z_], grid_1dim[_Z_]);
            move_wards[_BX_BZ_] = move_wards[_B_X_] + tmp * grid_1dim[_T_];
            move_backward(tmp, grid_index_1dim[_T_], grid_1dim[_T_]);
            move_wards[_BX_BT_] = move_wards[_B_X_] + tmp;
            move_backward(tmp, grid_index_1dim[_Z_], grid_1dim[_Z_]);
            move_wards[_BY_BZ_] = move_wards[_B_Y_] + tmp * grid_1dim[_T_];
            move_backward(tmp, grid_index_1dim[_T_], grid_1dim[_T_]);
            move_wards[_BY_BT_] = move_wards[_B_Y_] + tmp;
            move_backward(tmp, grid_index_1dim[_T_], grid_1dim[_T_]);
            move_wards[_BZ_BT_] = move_wards[_B_Z_] + tmp;
          }
          { // FB
            move_backward(tmp, grid_index_1dim[_Y_], grid_1dim[_Y_]);
            move_wards[_FX_BY_] = move_wards[_F_X_] + tmp * grid_2dim[_ZT_];
            move_backward(tmp, grid_index_1dim[_Z_], grid_1dim[_Z_]);
            move_wards[_FX_BZ_] = move_wards[_F_X_] + tmp * grid_1dim[_T_];
            move_backward(tmp, grid_index_1dim[_T_], grid_1dim[_T_]);
            move_wards[_FX_BT_] = move_wards[_F_X_] + tmp;
            move_backward(tmp, grid_index_1dim[_Z_], grid_1dim[_Z_]);
            move_wards[_FY_BZ_] = move_wards[_F_Y_] + tmp * grid_1dim[_T_];
            move_backward(tmp, grid_index_1dim[_T_], grid_1dim[_T_]);
            move_wards[_FY_BT_] = move_wards[_F_Y_] + tmp;
            move_backward(tmp, grid_index_1dim[_T_], grid_1dim[_T_]);
            move_wards[_FZ_BT_] = move_wards[_F_Z_] + tmp;
          }
          { // FB
            move_forward(tmp, grid_index_1dim[_Y_], grid_1dim[_Y_]);
            move_wards[_BX_FY_] = move_wards[_B_X_] + tmp * grid_2dim[_ZT_];
            move_forward(tmp, grid_index_1dim[_Z_], grid_1dim[_Z_]);
            move_wards[_BX_FZ_] = move_wards[_B_X_] + tmp * grid_1dim[_T_];
            move_forward(tmp, grid_index_1dim[_T_], grid_1dim[_T_]);
            move_wards[_BX_FT_] = move_wards[_B_X_] + tmp;
            move_forward(tmp, grid_index_1dim[_Z_], grid_1dim[_Z_]);
            move_wards[_BY_FZ_] = move_wards[_B_Y_] + tmp * grid_1dim[_T_];
            move_forward(tmp, grid_index_1dim[_T_], grid_1dim[_T_]);
            move_wards[_BY_FT_] = move_wards[_B_Y_] + tmp;
            move_forward(tmp, grid_index_1dim[_T_], grid_1dim[_T_]);
            move_wards[_BZ_FT_] = move_wards[_B_Z_] + tmp;
          }
          { // FF
            move_forward(tmp, grid_index_1dim[_Y_], grid_1dim[_Y_]);
            move_wards[_FX_FY_] = move_wards[_F_X_] + tmp * grid_2dim[_ZT_];
            move_forward(tmp, grid_index_1dim[_Z_], grid_1dim[_Z_]);
            move_wards[_FX_FZ_] = move_wards[_F_X_] + tmp * grid_1dim[_T_];
            move_forward(tmp, grid_index_1dim[_T_], grid_1dim[_T_]);
            move_wards[_FX_FT_] = move_wards[_F_X_] + tmp;
            move_forward(tmp, grid_index_1dim[_Z_], grid_1dim[_Z_]);
            move_wards[_FY_FZ_] = move_wards[_F_Y_] + tmp * grid_1dim[_T_];
            move_forward(tmp, grid_index_1dim[_T_], grid_1dim[_T_]);
            move_wards[_FY_FT_] = move_wards[_F_Y_] + tmp;
            move_forward(tmp, grid_index_1dim[_T_], grid_1dim[_T_]);
            move_wards[_FZ_FT_] = move_wards[_F_Z_] + tmp;
          }
        }
      }
      if (host_params[_TEST_IN_CPU_] != 1) {
        {
          cudaEventCreate(&start);
          cudaEventCreate(&stop);
          cudaEventRecord(start, 0);
          cudaEventSynchronize(start);
          // checkCudaErrors(cudaSetDevice(host_params[_NODE_RANK_])); // !!!!!!
          checkCudaErrors(cudaSetDevice(getLocalRank())); // !!!!!!
        }
        { // give basic cuda setup
          CUBLAS_CHECK(cublasCreate(&cublasH));
          checkCudaErrors(
              cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
          CUBLAS_CHECK(cublasSetStream(cublasH, stream));
        }
        { // give device params
          checkCudaErrors(cudaMallocAsync(&device_params,
                                          _PARAMS_SIZE_ * sizeof(int), stream));
          checkCudaErrors(cudaMallocAsync(&device_params_even_no_dag,
                                          _PARAMS_SIZE_ * sizeof(int), stream));
          checkCudaErrors(cudaMallocAsync(&device_params_odd_no_dag,
                                          _PARAMS_SIZE_ * sizeof(int), stream));
          checkCudaErrors(cudaMallocAsync(&device_params_even_dag,
                                          _PARAMS_SIZE_ * sizeof(int), stream));
          checkCudaErrors(cudaMallocAsync(&device_params_odd_dag,
                                          _PARAMS_SIZE_ * sizeof(int), stream));
          checkCudaErrors(cudaMemcpyAsync(device_params, host_params,
                                          _PARAMS_SIZE_ * sizeof(int),
                                          cudaMemcpyHostToDevice, stream));
          checkCudaErrors(cudaMemcpyAsync(
              device_params_even_no_dag, host_params,
              _PARAMS_SIZE_ * sizeof(int), cudaMemcpyHostToDevice, stream));
          give_param<T><<<1, 1, 0, stream>>>(device_params_even_no_dag,
                                             _PARITY_, _EVEN_);
          give_param<T><<<1, 1, 0, stream>>>(device_params_even_no_dag,
                                             _DAGGER_, _NO_USE_);
          checkCudaErrors(cudaMemcpyAsync(device_params_odd_no_dag, host_params,
                                          _PARAMS_SIZE_ * sizeof(int),
                                          cudaMemcpyHostToDevice, stream));
          give_param<T>
              <<<1, 1, 0, stream>>>(device_params_odd_no_dag, _PARITY_, _ODD_);
          give_param<T><<<1, 1, 0, stream>>>(device_params_odd_no_dag, _DAGGER_,
                                             _NO_USE_);
          checkCudaErrors(cudaMemcpyAsync(device_params_even_dag, host_params,
                                          _PARAMS_SIZE_ * sizeof(int),
                                          cudaMemcpyHostToDevice, stream));
          give_param<T>
              <<<1, 1, 0, stream>>>(device_params_even_dag, _PARITY_, _EVEN_);
          give_param<T>
              <<<1, 1, 0, stream>>>(device_params_even_dag, _DAGGER_, _USE_);
          checkCudaErrors(cudaMemcpyAsync(device_params_odd_dag, host_params,
                                          _PARAMS_SIZE_ * sizeof(int),
                                          cudaMemcpyHostToDevice, stream));
          give_param<T>
              <<<1, 1, 0, stream>>>(device_params_odd_dag, _PARITY_, _ODD_);
          give_param<T>
              <<<1, 1, 0, stream>>>(device_params_odd_dag, _DAGGER_, _USE_);
        }
      }
      if (host_params[_SET_PLAN_] == _SET_PLAN_N_2_) // just for laplacian
      {
        for (int i = 0; i < _DIM_; i++) { // give cuda setup
          checkCudaErrors(cudaStreamCreateWithFlags(&stream_dims[i],
                                                    cudaStreamNonBlocking));
        }
        for (int i = 0; i < _DIM_; i++) { // give memory malloc
          checkCudaErrors(cudaMallocAsync(
              &device_send_vec[i * _BF_],
              lat_3dim_C[i] * sizeof(LatticeComplex<T>), stream));
          checkCudaErrors(cudaMallocAsync(
              &device_send_vec[i * _BF_ + 1],
              lat_3dim_C[i] * sizeof(LatticeComplex<T>), stream));
          checkCudaErrors(cudaMallocAsync(
              &device_recv_vec[i * _BF_],
              lat_3dim_C[i] * sizeof(LatticeComplex<T>), stream));
          checkCudaErrors(cudaMallocAsync(
              &device_recv_vec[i * _BF_ + 1],
              lat_3dim_C[i] * sizeof(LatticeComplex<T>), stream));
          checkCudaErrors(
              cudaMallocHost(&host_send_vec[i * _BF_],
                             lat_3dim_C[i] * sizeof(LatticeComplex<T>)));
          checkCudaErrors(
              cudaMallocHost(&host_send_vec[i * _BF_ + 1],
                             lat_3dim_C[i] * sizeof(LatticeComplex<T>)));
          checkCudaErrors(
              cudaMallocHost(&host_recv_vec[i * _BF_],
                             lat_3dim_C[i] * sizeof(LatticeComplex<T>)));
          checkCudaErrors(
              cudaMallocHost(&host_recv_vec[i * _BF_ + 1],
                             lat_3dim_C[i] * sizeof(LatticeComplex<T>)));
        }
      }
      if (host_params[_SET_PLAN_] >= _SET_PLAN0_) // for wilson dslash
      {
        for (int i = 0; i < _DIM_; i++) { // give cuda setup
          checkCudaErrors(cudaStreamCreateWithFlags(&stream_dims[i],
                                                    cudaStreamNonBlocking));
        }
        for (int i = 0; i < _DIM_; i++) { // give memory malloc
          checkCudaErrors(cudaMallocAsync(
              &device_send_vec[i * _BF_],
              lat_3dim_Half_SC[i] * sizeof(LatticeComplex<T>), stream));
          checkCudaErrors(cudaMallocAsync(
              &device_send_vec[i * _BF_ + 1],
              lat_3dim_Half_SC[i] * sizeof(LatticeComplex<T>), stream));
          checkCudaErrors(cudaMallocAsync(
              &device_recv_vec[i * _BF_],
              lat_3dim_Half_SC[i] * sizeof(LatticeComplex<T>), stream));
          checkCudaErrors(cudaMallocAsync(
              &device_recv_vec[i * _BF_ + 1],
              lat_3dim_Half_SC[i] * sizeof(LatticeComplex<T>), stream));
          checkCudaErrors(
              cudaMallocHost(&host_send_vec[i * _BF_],
                             lat_3dim_Half_SC[i] * sizeof(LatticeComplex<T>)));
          checkCudaErrors(
              cudaMallocHost(&host_send_vec[i * _BF_ + 1],
                             lat_3dim_Half_SC[i] * sizeof(LatticeComplex<T>)));
          checkCudaErrors(
              cudaMallocHost(&host_recv_vec[i * _BF_],
                             lat_3dim_Half_SC[i] * sizeof(LatticeComplex<T>)));
          checkCudaErrors(
              cudaMallocHost(&host_recv_vec[i * _BF_ + 1],
                             lat_3dim_Half_SC[i] * sizeof(LatticeComplex<T>)));
        }
      }
      if (host_params[_SET_PLAN_] >=
          _SET_PLAN1_) // just for bistabcg and cg and the whole dslash for them
      {
        for (int i = 0; i < _DIM_; i++) { // give cuda setup
          CUBLAS_CHECK(cublasCreate(&cublasHs[i]));
          checkCudaErrors(
              cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
          CUBLAS_CHECK(cublasSetStream(cublasHs[i], streams[i]));
        }
        {
          checkCudaErrors(cudaMallocAsync(
              &device_vec0, lat_4dim_SC * sizeof(LatticeComplex<T>), stream));
          checkCudaErrors(cudaMallocAsync(
              &device_vec1, lat_4dim_SC * sizeof(LatticeComplex<T>), stream));
          checkCudaErrors(cudaMallocAsync(
              &device_vec2, lat_4dim_SC * sizeof(LatticeComplex<T>), stream));
          checkCudaErrors(cudaMallocAsync(
              &device_vals, _vals_size_ * sizeof(LatticeComplex<T>), stream));
          give_1custom<T>
              <<<1, 1, 0, stream>>>(device_vals, _lat_4dim_, T(lat_4dim), 0.0);
          checkCudaErrors(cudaStreamSynchronize(stream));
        }
      }
      if (host_params[_SET_PLAN_] >= _SET_PLAN2_) // for clover dslash
      {
        // give memory malloc
        for (int i = 0; i < _DIM_; i++) { // u in 1dim move
          checkCudaErrors(cudaMallocAsync(
              &device_u_1dim_send_vec[i * _BF_],
              lat_3dim[i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>), stream));
          checkCudaErrors(cudaMallocAsync(
              &device_u_1dim_send_vec[i * _BF_ + 1],
              lat_3dim[i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>), stream));
          checkCudaErrors(cudaMallocAsync(
              &device_u_1dim_recv_vec[i * _BF_],
              lat_3dim[i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>), stream));
          checkCudaErrors(cudaMallocAsync(
              &device_u_1dim_recv_vec[i * _BF_ + 1],
              lat_3dim[i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>), stream));
          host_u_1dim_send_vec[i * _BF_] = (void *)malloc(
              lat_3dim[i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>));
          host_u_1dim_send_vec[i * _BF_ + 1] = (void *)malloc(
              lat_3dim[i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>));
          host_u_1dim_recv_vec[i * _BF_] = (void *)malloc(
              lat_3dim[i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>));
          host_u_1dim_recv_vec[i * _BF_ + 1] = (void *)malloc(
              lat_3dim[i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>));
        }
        for (int i = 0; i < _2DIM_; i++) { // u in 2dim move
          checkCudaErrors(cudaMallocAsync(
              &device_u_2dim_send_vec[i * _BF_ * _BF_ + 0],
              lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>),
              stream));
          checkCudaErrors(cudaMallocAsync(
              &device_u_2dim_recv_vec[i * _BF_ * _BF_ + 0],
              lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>),
              stream));
          checkCudaErrors(cudaMallocAsync(
              &device_u_2dim_send_vec[i * _BF_ * _BF_ + 1],
              lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>),
              stream));
          checkCudaErrors(cudaMallocAsync(
              &device_u_2dim_recv_vec[i * _BF_ * _BF_ + 1],
              lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>),
              stream));
          checkCudaErrors(cudaMallocAsync(
              &device_u_2dim_send_vec[i * _BF_ * _BF_ + 2],
              lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>),
              stream));
          checkCudaErrors(cudaMallocAsync(
              &device_u_2dim_recv_vec[i * _BF_ * _BF_ + 2],
              lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>),
              stream));
          checkCudaErrors(cudaMallocAsync(
              &device_u_2dim_send_vec[i * _BF_ * _BF_ + 3],
              lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>),
              stream));
          checkCudaErrors(cudaMallocAsync(
              &device_u_2dim_recv_vec[i * _BF_ * _BF_ + 3],
              lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ * sizeof(LatticeComplex<T>),
              stream));
          host_u_2dim_send_vec[i * _BF_ * _BF_ + 0] =
              (void *)malloc(lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                             sizeof(LatticeComplex<T>));
          host_u_2dim_recv_vec[i * _BF_ * _BF_ + 0] =
              (void *)malloc(lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                             sizeof(LatticeComplex<T>));
          host_u_2dim_send_vec[i * _BF_ * _BF_ + 1] =
              (void *)malloc(lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                             sizeof(LatticeComplex<T>));
          host_u_2dim_recv_vec[i * _BF_ * _BF_ + 1] =
              (void *)malloc(lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                             sizeof(LatticeComplex<T>));
          host_u_2dim_send_vec[i * _BF_ * _BF_ + 2] =
              (void *)malloc(lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                             sizeof(LatticeComplex<T>));
          host_u_2dim_recv_vec[i * _BF_ * _BF_ + 2] =
              (void *)malloc(lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                             sizeof(LatticeComplex<T>));
          host_u_2dim_send_vec[i * _BF_ * _BF_ + 3] =
              (void *)malloc(lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                             sizeof(LatticeComplex<T>));
          host_u_2dim_recv_vec[i * _BF_ * _BF_ + 3] =
              (void *)malloc(lat_2dim[_2DIM_ - 1 - i] * _LAT_PCCD_ *
                             sizeof(LatticeComplex<T>));
        }
      }
      { // init end
        checkCudaErrors(cudaStreamSynchronize(stream));
      }
    }
  }
  int max_iter() { return host_params[_MAX_ITER_]; }
  T kappa() { return 1 / (2 * host_argv[_MASS_] + 8); }
  T atol() { return host_argv[_ATOL_]; }
  T atol2() { return host_argv[_ATOL_] * host_argv[_ATOL_]; }
  T sigma() { return host_argv[_SIGMA_]; }
  float get_time() {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    return time; // ms
  }
  void end() {
    if (host_params[_TEST_IN_CPU_] != 1) {
      checkCudaErrors(cudaStreamSynchronize(stream));
      checkCudaErrors(cudaFreeAsync(device_params, stream));
      checkCudaErrors(cudaFreeAsync(device_params_even_no_dag, stream));
      checkCudaErrors(cudaFreeAsync(device_params_odd_no_dag, stream));
      checkCudaErrors(cudaFreeAsync(device_params_even_dag, stream));
      checkCudaErrors(cudaFreeAsync(device_params_odd_dag, stream));
      if (host_params[_SET_PLAN_] == _SET_PLAN_N_2_) // just for laplacian
      {
        for (int i = 0; i < _DIM_; i++) {
          checkCudaErrors(cudaStreamSynchronize(stream_dims[i]));
          checkCudaErrors(cudaStreamDestroy(stream_dims[i]));
          checkCudaErrors(cudaFreeAsync(device_send_vec[i * _BF_], stream));
          checkCudaErrors(cudaFreeAsync(device_send_vec[i * _BF_ + 1], stream));
          checkCudaErrors(cudaFreeAsync(device_recv_vec[i * _BF_], stream));
          checkCudaErrors(cudaFreeAsync(device_recv_vec[i * _BF_ + 1], stream));
          checkCudaErrors(cudaFreeHost(host_send_vec[i * _BF_]));
          checkCudaErrors(cudaFreeHost(host_send_vec[i * _BF_ + 1]));
          checkCudaErrors(cudaFreeHost(host_recv_vec[i * _BF_]));
          checkCudaErrors(cudaFreeHost(host_recv_vec[i * _BF_ + 1]));
        }
      }
      if (host_params[_SET_PLAN_] >= _SET_PLAN0_) // for wilson dslash
      {
        for (int i = 0; i < _DIM_; i++) {
          checkCudaErrors(cudaStreamSynchronize(stream_dims[i]));
          checkCudaErrors(cudaStreamDestroy(stream_dims[i]));
          checkCudaErrors(cudaFreeAsync(device_send_vec[i * _BF_], stream));
          checkCudaErrors(cudaFreeAsync(device_send_vec[i * _BF_ + 1], stream));
          checkCudaErrors(cudaFreeAsync(device_recv_vec[i * _BF_], stream));
          checkCudaErrors(cudaFreeAsync(device_recv_vec[i * _BF_ + 1], stream));
          checkCudaErrors(cudaFreeHost(host_send_vec[i * _BF_]));
          checkCudaErrors(cudaFreeHost(host_send_vec[i * _BF_ + 1]));
          checkCudaErrors(cudaFreeHost(host_recv_vec[i * _BF_]));
          checkCudaErrors(cudaFreeHost(host_recv_vec[i * _BF_ + 1]));
        }
      }
      if (host_params[_SET_PLAN_] >=
          _SET_PLAN1_) // just for wilson bistabcg and cg
      {
        for (int i = 0; i < _DIM_; i++) {
          checkCudaErrors(cudaStreamSynchronize(streams[i]));
          CUBLAS_CHECK(cublasDestroy(cublasHs[i]));
          checkCudaErrors(cudaStreamDestroy(streams[i]));
        }
      }
      if (host_params[_SET_PLAN_] >= _SET_PLAN2_) // for clover dslash
      {
        for (int i = 0; i < _DIM_; i++) {
          checkCudaErrors(
              cudaFreeAsync(device_u_1dim_send_vec[i * _BF_], stream));
          checkCudaErrors(
              cudaFreeAsync(device_u_1dim_send_vec[i * _BF_ + 1], stream));
          checkCudaErrors(
              cudaFreeAsync(device_u_1dim_recv_vec[i * _BF_], stream));
          checkCudaErrors(
              cudaFreeAsync(device_u_1dim_recv_vec[i * _BF_ + 1], stream));
          free(host_u_1dim_send_vec[i * _BF_]);
          free(host_u_1dim_send_vec[i * _BF_ + 1]);
          free(host_u_1dim_recv_vec[i * _BF_]);
          free(host_u_1dim_recv_vec[i * _BF_ + 1]);
        }
        for (int i = 0; i < _2DIM_; i++) {
          checkCudaErrors(cudaFreeAsync(
              device_u_2dim_send_vec[i * _BF_ * _BF_ + 0], stream));
          checkCudaErrors(cudaFreeAsync(
              device_u_2dim_recv_vec[i * _BF_ * _BF_ + 0], stream));
          checkCudaErrors(cudaFreeAsync(
              device_u_2dim_send_vec[i * _BF_ * _BF_ + 1], stream));
          checkCudaErrors(cudaFreeAsync(
              device_u_2dim_recv_vec[i * _BF_ * _BF_ + 1], stream));
          checkCudaErrors(cudaFreeAsync(
              device_u_2dim_send_vec[i * _BF_ * _BF_ + 2], stream));
          checkCudaErrors(cudaFreeAsync(
              device_u_2dim_recv_vec[i * _BF_ * _BF_ + 2], stream));
          checkCudaErrors(cudaFreeAsync(
              device_u_2dim_send_vec[i * _BF_ * _BF_ + 3], stream));
          checkCudaErrors(cudaFreeAsync(
              device_u_2dim_recv_vec[i * _BF_ * _BF_ + 3], stream));
          free(host_u_2dim_send_vec[i * _BF_ * _BF_ + 0]);
          free(host_u_2dim_recv_vec[i * _BF_ * _BF_ + 0]);
          free(host_u_2dim_send_vec[i * _BF_ * _BF_ + 1]);
          free(host_u_2dim_recv_vec[i * _BF_ * _BF_ + 1]);
          free(host_u_2dim_send_vec[i * _BF_ * _BF_ + 2]);
          free(host_u_2dim_recv_vec[i * _BF_ * _BF_ + 2]);
          free(host_u_2dim_send_vec[i * _BF_ * _BF_ + 3]);
          free(host_u_2dim_recv_vec[i * _BF_ * _BF_ + 3]);
        }
      }
      // end end
      CUBLAS_CHECK(cublasDestroy(cublasH));
      checkCudaErrors(cudaStreamSynchronize(stream));
      checkCudaErrors(cudaStreamDestroy(stream));
      printf("lattice set whole time:%.9lf "
             "sec\n",
             get_time() / 1e3);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
    }
  }
  void _print() {
    // clang-format off
    printf("gridDim.x                                    :%d\n", gridDim.x);
    printf("blockDim.x                                   :%d\n", blockDim.x);
    printf("host_params[_LAT_X_]                         :%d\n", host_params[_LAT_X_]);
    printf("host_params[_LAT_Y_]                         :%d\n", host_params[_LAT_Y_]);
    printf("host_params[_LAT_Z_]                         :%d\n", host_params[_LAT_Z_]);
    printf("host_params[_LAT_T_]                         :%d\n", host_params[_LAT_T_]);
    printf("host_params[_LAT_XYZT_]                      :%d\n", host_params[_LAT_XYZT_]);
    printf("host_params[_GRID_X_]                        :%d\n", host_params[_GRID_X_]);
    printf("host_params[_GRID_Y_]                        :%d\n", host_params[_GRID_Y_]);
    printf("host_params[_GRID_Z_]                        :%d\n", host_params[_GRID_Z_]);
    printf("host_params[_GRID_T_]                        :%d\n", host_params[_GRID_T_]);
    printf("host_params[_PARITY_]                        :%d\n", host_params[_PARITY_]);
    printf("host_params[_NODE_RANK_]                     :%d\n", host_params[_NODE_RANK_]);
    printf("host_params[_NODE_SIZE_]                     :%d\n", host_params[_NODE_SIZE_]);
    printf("host_params[_DAGGER_]                        :%d\n", host_params[_DAGGER_]);
    printf("host_params[_MAX_ITER_]                      :%d\n", host_params[_MAX_ITER_]);
    printf("host_params[_DATA_TYPE_]                     :%d\n", host_params[_DATA_TYPE_]);
    printf("host_params[_SET_INDEX_]                     :%d\n", host_params[_SET_INDEX_]);
    printf("host_params[_SET_PLAN_]                      :%d\n", host_params[_SET_PLAN_]);
    printf("host_params[_MG_NUM_LEVEL_]                  :%d\n", host_params[_MG_NUM_LEVEL_]);
    printf("host_params[_MG_LEVEL_INDEX_]                :%d\n", host_params[_MG_LEVEL_INDEX_]);
    printf("host_params[_MG_LEVEL1_E_]                   :%d\n", host_params[_MG_LEVEL1_E_]);
    printf("host_params[_MG_LEVEL1_X_]                   :%d\n", host_params[_MG_LEVEL1_X_]);
    printf("host_params[_MG_LEVEL1_Y_]                   :%d\n", host_params[_MG_LEVEL1_Y_]);
    printf("host_params[_MG_LEVEL1_Z_]                   :%d\n", host_params[_MG_LEVEL1_Z_]);
    printf("host_params[_MG_LEVEL1_T_]                   :%d\n", host_params[_MG_LEVEL1_T_]);
    printf("host_params[_MG_LEVEL1_MAX_ITER_]            :%d\n", host_params[_MG_LEVEL1_MAX_ITER_]);
    printf("host_params[_MG_LEVEL1_DATA_TYPE_]           :%d\n", host_params[_MG_LEVEL1_DATA_TYPE_]);
    printf("host_params[_MG_LEVEL1_NUM_RESTART_]         :%d\n", host_params[_MG_LEVEL1_NUM_RESTART_]);
    printf("host_params[_MG_LEVEL2_E_]                   :%d\n", host_params[_MG_LEVEL2_E_]);
    printf("host_params[_MG_LEVEL2_X_]                   :%d\n", host_params[_MG_LEVEL2_X_]);
    printf("host_params[_MG_LEVEL2_Y_]                   :%d\n", host_params[_MG_LEVEL2_Y_]);
    printf("host_params[_MG_LEVEL2_Z_]                   :%d\n", host_params[_MG_LEVEL2_Z_]);
    printf("host_params[_MG_LEVEL2_T_]                   :%d\n", host_params[_MG_LEVEL2_T_]);
    printf("host_params[_MG_LEVEL2_MAX_ITER_]            :%d\n", host_params[_MG_LEVEL2_MAX_ITER_]);
    printf("host_params[_MG_LEVEL2_DATA_TYPE_]           :%d\n", host_params[_MG_LEVEL2_DATA_TYPE_]);
    printf("host_params[_MG_LEVEL2_NUM_RESTART_]         :%d\n", host_params[_MG_LEVEL2_NUM_RESTART_]);
    printf("host_params[_MG_LEVEL3_E_]                   :%d\n", host_params[_MG_LEVEL3_E_]);
    printf("host_params[_MG_LEVEL3_X_]                   :%d\n", host_params[_MG_LEVEL3_X_]);
    printf("host_params[_MG_LEVEL3_Y_]                   :%d\n", host_params[_MG_LEVEL3_Y_]);
    printf("host_params[_MG_LEVEL3_Z_]                   :%d\n", host_params[_MG_LEVEL3_Z_]);
    printf("host_params[_MG_LEVEL3_T_]                   :%d\n", host_params[_MG_LEVEL3_T_]);
    printf("host_params[_MG_LEVEL3_MAX_ITER_]            :%d\n", host_params[_MG_LEVEL3_MAX_ITER_]);
    printf("host_params[_MG_LEVEL3_DATA_TYPE_]           :%d\n", host_params[_MG_LEVEL3_DATA_TYPE_]);
    printf("host_params[_MG_LEVEL3_NUM_RESTART_]         :%d\n", host_params[_MG_LEVEL3_NUM_RESTART_]);
    printf("host_params[_MG_LEVEL4_E_]                   :%d\n", host_params[_MG_LEVEL4_E_]);
    printf("host_params[_MG_LEVEL4_X_]                   :%d\n", host_params[_MG_LEVEL4_X_]);
    printf("host_params[_MG_LEVEL4_Y_]                   :%d\n", host_params[_MG_LEVEL4_Y_]);
    printf("host_params[_MG_LEVEL4_Z_]                   :%d\n", host_params[_MG_LEVEL4_Z_]);
    printf("host_params[_MG_LEVEL4_T_]                   :%d\n", host_params[_MG_LEVEL4_T_]);
    printf("host_params[_MG_LEVEL4_MAX_ITER_]            :%d\n", host_params[_MG_LEVEL4_MAX_ITER_]);
    printf("host_params[_MG_LEVEL4_DATA_TYPE_]           :%d\n", host_params[_MG_LEVEL4_DATA_TYPE_]);
    printf("host_params[_MG_LEVEL4_NUM_RESTART_]         :%d\n", host_params[_MG_LEVEL4_NUM_RESTART_]);
    printf("host_params[_MG_PARAMS_SIZE_]                :%d\n", host_params[_MG_PARAMS_SIZE_]);
    printf("host_params[_VERBOSE_]                       :%d\n", host_params[_VERBOSE_]);
    printf("host_params[_SEED_]                          :%d\n", host_params[_SEED_]);
    printf("host_params[_TEST_IN_CPU_]                   :%d\n", host_params[_TEST_IN_CPU_]);
    printf("host_argv[_MASS_]                            :%e\n", host_argv[_MASS_]);
    printf("host_argv[_ATOL_]                            :%e\n", host_argv[_ATOL_]);
    printf("host_argv[_SIGMA_]                           :%e\n", host_argv[_SIGMA_]);
    printf("host_argv[__MG_LEVEL1_ATOL__]                :%e\n", host_argv[_MG_LEVEL1_ATOL_]);
    printf("host_argv[__MG_LEVEL2_ATOL__]                :%e\n", host_argv[_MG_LEVEL2_ATOL_]);
    printf("host_argv[__MG_LEVEL3_ATOL__]                :%e\n", host_argv[_MG_LEVEL3_ATOL_]);
    printf("host_argv[__MG_LEVEL4_ATOL__]                :%e\n", host_argv[_MG_LEVEL4_ATOL_]);
    printf("lat_2dim[_XY_]                               :%d\n", lat_2dim[_XY_]);
    printf("lat_2dim[_XZ_]                               :%d\n", lat_2dim[_XZ_]);
    printf("lat_2dim[_XT_]                               :%d\n", lat_2dim[_XT_]);
    printf("lat_2dim[_YZ_]                               :%d\n", lat_2dim[_YZ_]);
    printf("lat_2dim[_YT_]                               :%d\n", lat_2dim[_YT_]);
    printf("lat_2dim[_ZT_]                               :%d\n", lat_2dim[_ZT_]);
    printf("lat_3dim[_YZT_]                              :%d\n", lat_3dim[_YZT_]);
    printf("lat_3dim[_XZT_]                              :%d\n", lat_3dim[_XZT_]);
    printf("lat_3dim[_XYT_]                              :%d\n", lat_3dim[_XYT_]);
    printf("lat_3dim[_XYZ_]                              :%d\n", lat_3dim[_XYZ_]);
    printf("lat_4dim                                     :%d\n", lat_4dim);
    printf("grid_1dim[_X_]                               :%d\n", grid_1dim[_X_]);
    printf("grid_1dim[_Y_]                               :%d\n", grid_1dim[_Y_]);
    printf("grid_1dim[_Z_]                               :%d\n", grid_1dim[_Z_]);
    printf("grid_1dim[_T_]                               :%d\n", grid_1dim[_T_]);
    printf("grid_2dim[_XY_]                              :%d\n", grid_2dim[_XY_]);
    printf("grid_2dim[_XZ_]                              :%d\n", grid_2dim[_XZ_]);
    printf("grid_2dim[_XT_]                              :%d\n", grid_2dim[_XT_]);
    printf("grid_2dim[_YZ_]                              :%d\n", grid_2dim[_YZ_]);
    printf("grid_2dim[_YT_]                              :%d\n", grid_2dim[_YT_]);
    printf("grid_2dim[_ZT_]                              :%d\n", grid_2dim[_ZT_]);
    printf("grid_3dim[_YZT_]                             :%d\n", grid_3dim[_YZT_]);
    printf("grid_3dim[_XZT_]                             :%d\n", grid_3dim[_XZT_]);
    printf("grid_3dim[_XYT_]                             :%d\n", grid_3dim[_XYT_]);
    printf("grid_3dim[_XYZ_]                             :%d\n", grid_3dim[_XYZ_]);
    printf("grid_index_1dim[_X_]                         :%d\n", grid_index_1dim[_X_]);
    printf("grid_index_1dim[_Y_]                         :%d\n", grid_index_1dim[_Y_]);
    printf("grid_index_1dim[_Z_]                         :%d\n", grid_index_1dim[_Z_]);
    printf("grid_index_1dim[_T_]                         :%d\n", grid_index_1dim[_T_]);
    printf("move_wards[_B_X_]                            :%d\n", move_wards[_B_X_]);
    printf("move_wards[_B_Y_]                            :%d\n", move_wards[_B_Y_]);
    printf("move_wards[_B_Z_]                            :%d\n", move_wards[_B_Z_]);
    printf("move_wards[_B_T_]                            :%d\n", move_wards[_B_T_]);
    printf("move_wards[_F_X_]                            :%d\n", move_wards[_F_X_]);
    printf("move_wards[_F_Y_]                            :%d\n", move_wards[_F_Y_]);
    printf("move_wards[_F_Z_]                            :%d\n", move_wards[_F_Z_]);
    printf("move_wards[_F_T_]                            :%d\n", move_wards[_F_T_]);
    printf("move_wards[_BX_BY_]                          :%d\n", move_wards[_BX_BY_]);
    printf("move_wards[_BX_BZ_]                          :%d\n", move_wards[_BX_BZ_]);
    printf("move_wards[_BX_BT_]                          :%d\n", move_wards[_BX_BT_]);
    printf("move_wards[_BY_BZ_]                          :%d\n", move_wards[_BY_BZ_]);
    printf("move_wards[_BY_BT_]                          :%d\n", move_wards[_BY_BT_]);
    printf("move_wards[_BZ_BT_]                          :%d\n", move_wards[_BZ_BT_]);
    printf("move_wards[_FX_BY_]                          :%d\n", move_wards[_FX_BY_]);
    printf("move_wards[_FX_BZ_]                          :%d\n", move_wards[_FX_BZ_]);
    printf("move_wards[_FX_BT_]                          :%d\n", move_wards[_FX_BT_]);
    printf("move_wards[_FY_BZ_]                          :%d\n", move_wards[_FY_BZ_]);
    printf("move_wards[_FY_BT_]                          :%d\n", move_wards[_FY_BT_]);
    printf("move_wards[_FZ_BT_]                          :%d\n", move_wards[_FZ_BT_]);
    printf("move_wards[_BX_FY_]                          :%d\n", move_wards[_BX_FY_]);
    printf("move_wards[_BX_FZ_]                          :%d\n", move_wards[_BX_FZ_]);
    printf("move_wards[_BX_FT_]                          :%d\n", move_wards[_BX_FT_]);
    printf("move_wards[_BY_FZ_]                          :%d\n", move_wards[_BY_FZ_]);
    printf("move_wards[_BY_FT_]                          :%d\n", move_wards[_BY_FT_]);
    printf("move_wards[_BZ_FT_]                          :%d\n", move_wards[_BZ_FT_]);
    printf("move_wards[_FX_FY_]                          :%d\n", move_wards[_FX_FY_]);
    printf("move_wards[_FX_FZ_]                          :%d\n", move_wards[_FX_FZ_]);
    printf("move_wards[_FX_FT_]                          :%d\n", move_wards[_FX_FT_]);
    printf("move_wards[_FY_FZ_]                          :%d\n", move_wards[_FY_FZ_]);
    printf("move_wards[_FY_FT_]                          :%d\n", move_wards[_FY_FT_]);
    printf("move_wards[_FZ_FT_]                          :%d\n", move_wards[_FZ_FT_]);
  }
};
} // namespace qcu
#endif