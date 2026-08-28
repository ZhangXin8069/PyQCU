#ifndef _LATTICE_MULTIGRID_H
#define _LATTICE_MULTIGRID_H
#include "./define.h"
#include "./lattice_set.h"
#include "./lattice_mpi.h"
#include "./multigrid.h"
#include <limits>
#include <vector>
namespace qcu {

// The 33-point coarse stencil needs the 32 signed axial/two-axis neighbour
// directions.  The wire order is shared by the CUDA pack/unpack kernels and
// both MPI transports.  Direction 2*k is the canonical shift and 2*k+1 is
// its opposite; consequently ``direction ^ 1`` is always the opposite.
inline int multigrid_coarse_shift_component_host(int direction,
                                                 int component) {
  static const int shifts[32][4] = {
      {1, 0, 0, 0},  {-1, 0, 0, 0}, {0, 1, 0, 0},  {0, -1, 0, 0},
      {0, 0, 1, 0},  {0, 0, -1, 0}, {0, 0, 0, 1},  {0, 0, 0, -1},
      {1, 1, 0, 0},  {-1, -1, 0, 0}, {1, -1, 0, 0}, {-1, 1, 0, 0},
      {1, 0, 1, 0},  {-1, 0, -1, 0}, {1, 0, -1, 0}, {-1, 0, 1, 0},
      {1, 0, 0, 1},  {-1, 0, 0, -1}, {1, 0, 0, -1}, {-1, 0, 0, 1},
      {0, 1, 1, 0},  {0, -1, -1, 0}, {0, 1, -1, 0}, {0, -1, 1, 0},
      {0, 1, 0, 1},  {0, -1, 0, -1}, {0, 1, 0, -1}, {0, -1, 0, 1},
      {0, 0, 1, 1},  {0, 0, -1, -1}, {0, 0, 1, -1}, {0, 0, -1, 1}};
  if (direction < 0 || direction >= 32 || component < 0 || component >= 4)
    return 0;
  return shifts[direction][component];
}

inline size_t multigrid_coarse_face_sites_host(int direction, int X, int Y,
                                                int Z, int Lt) {
  const int dims[4] = {X, Y, Z, Lt};
  size_t result = 1;
  for (int d = 0; d < 4; ++d)
    if (multigrid_coarse_shift_component_host(direction, d) == 0)
      result *= static_cast<size_t>(dims[d]);
  return result;
}

template <typename SetT>
inline int multigrid_coarse_rank_shift(const LatticeSet<SetT> *set,
                                       const int shift[4]) {
  const int dims[4] = {
      set->host_params[_GRID_X_], set->host_params[_GRID_Y_],
      set->host_params[_GRID_Z_], set->host_params[_GRID_T_]};
  int coord[4] = {set->grid_index_1dim[_X_], set->grid_index_1dim[_Y_],
                  set->grid_index_1dim[_Z_], set->grid_index_1dim[_T_]};
  for (int d = 0; d < 4; ++d) {
    if (dims[d] <= 0)
      throw std::invalid_argument("coarse MPI process-grid dimension is not positive");
    // A non-distributed axis is periodic inside this rank.  Its sign still
    // selects the local boundary plane, but it must not contribute to the
    // MPI process displacement.
    coord[d] += dims[d] > 1 ? shift[d] : 0;
    if (coord[d] < 0) coord[d] += dims[d];
    if (coord[d] >= dims[d]) coord[d] -= dims[d];
  }
  return ((coord[0] * dims[1] + coord[1]) * dims[2] + coord[2]) * dims[3] +
         coord[3];
}

// The exchange object is type-erased by CoarseHaloExchangeAny below so mixed
// precision trees can use the same lifecycle.  The virtual interface keeps
// all CUDA/MPI state local to one level and makes it impossible for a float
// level to reinterpret a double level's staging buffer.
struct CoarseHaloExchangeBase {
  virtual ~CoarseHaloExchangeBase() {}
  virtual void begin(void *device_in, cudaStream_t compute_stream) = 0;
  virtual void finish(cudaStream_t compute_stream) = 0;
  virtual void release() = 0;
  virtual void *device_halo() const = 0;
  virtual bool pending() const = 0;
  virtual const char *transport_name() const = 0;
};

template <typename U, typename SetT> struct CoarseHaloExchange
    : public CoarseHaloExchangeBase {
  static_assert(sizeof(LatticeComplex<U>) == 2 * sizeof(U),
                "MPI wire layout requires contiguous real/imag scalars");

  LatticeSet<SetT> *set_ptr = nullptr;
  int E = 0, X = 0, Y = 0, Z = 0, Lt = 0;
  int rank = 0;
  int grid[4] = {1, 1, 1, 1};
  int coord[4] = {0, 0, 0, 0};
  int peer[32] = {};
  int real_count[32] = {};
  size_t max_face = 0;
  size_t direction_stride = 0;
  size_t packed_elements = 0;
  size_t packed_bytes = 0;
  size_t halo_bytes = 0;
  void *device_send = nullptr;
  void *device_recv = nullptr;
  void *halo = nullptr;
  void *host_send = nullptr;
  void *host_recv = nullptr;
  cudaStream_t exchange_stream = nullptr;
  cudaEvent_t input_ready = nullptr;
  cudaEvent_t halo_ready = nullptr;
  MPI_Request send_request[32];
  MPI_Request recv_request[32];
  bool request_active = false;
  bool use_device_mpi = false;
  bool use_nvshmem = false;

  CoarseHaloExchange(LatticeSet<SetT> *set, int dof, int x, int y, int z,
                     int t)
      : set_ptr(set), E(dof), X(x), Y(y), Z(z), Lt(t) {
    if (set_ptr == nullptr || E <= 0 || X <= 0 || Y <= 0 || Z <= 0 || Lt <= 0)
      throw std::invalid_argument("invalid coarse halo geometry");
    rank = set_ptr->host_params[_NODE_RANK_];
    grid[0] = set_ptr->host_params[_GRID_X_];
    grid[1] = set_ptr->host_params[_GRID_Y_];
    grid[2] = set_ptr->host_params[_GRID_Z_];
    grid[3] = set_ptr->host_params[_GRID_T_];
    coord[0] = set_ptr->grid_index_1dim[_X_];
    coord[1] = set_ptr->grid_index_1dim[_Y_];
    coord[2] = set_ptr->grid_index_1dim[_Z_];
    coord[3] = set_ptr->grid_index_1dim[_T_];
    for (int d = 0; d < 4; ++d)
      if (grid[d] <= 0)
        throw std::invalid_argument("coarse MPI process-grid dimension is not positive");

    // The shift table is intentionally copied into a local array: passing a
    // pointer into the helper above would make the direction-to-rank mapping
    // depend on a temporary object's lifetime.
    for (int d = 0; d < 32; ++d) {
      int shift[4];
      for (int k = 0; k < 4; ++k)
        shift[k] = multigrid_coarse_shift_component_host(d, k);
      peer[d] = multigrid_coarse_rank_shift(set_ptr, shift);
      const size_t face = multigrid_coarse_face_sites_host(d, X, Y, Z, Lt);
      if (face > max_face) max_face = face;
      const size_t count = static_cast<size_t>(2) * static_cast<size_t>(E) * face;
      if (count > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("coarse MPI halo message is too large");
      real_count[d] = static_cast<int>(count);
    }
    direction_stride = static_cast<size_t>(E) * max_face;
    packed_elements = 32 * direction_stride;
    packed_bytes = packed_elements * sizeof(LatticeComplex<U>);
    const size_t hvol = static_cast<size_t>(X + 2) * static_cast<size_t>(Y + 2) *
                        static_cast<size_t>(Z + 2) * static_cast<size_t>(Lt + 2);
    halo_bytes = static_cast<size_t>(E) * hvol * sizeof(LatticeComplex<U>);
    for (int d = 0; d < 32; ++d) {
      send_request[d] = MPI_REQUEST_NULL;
      recv_request[d] = MPI_REQUEST_NULL;
    }

    checkCudaErrors(cudaStreamCreateWithFlags(&exchange_stream,
                                              cudaStreamNonBlocking));
    checkCudaErrors(cudaEventCreateWithFlags(&input_ready, cudaEventDisableTiming));
    checkCudaErrors(cudaEventCreateWithFlags(&halo_ready, cudaEventDisableTiming));
    checkCudaErrors(cudaMalloc(&device_send, packed_bytes));
    checkCudaErrors(cudaMalloc(&halo, halo_bytes));
    checkCudaErrors(cudaMallocHost(&host_send, packed_bytes));
    checkCudaErrors(cudaMallocHost(&host_recv, packed_bytes));
#if defined(QCU_HAVE_NVSHMEM)
    // NVSHMEM is a process-wide collective.  It is selected only when the
    // caller explicitly requests it and the MPI rank/PE mapping is identical.
    use_nvshmem = qcu_nvshmem_init_for_mpi(
        set_ptr->host_params[_NODE_SIZE_]);
    if (use_nvshmem) device_recv = nvshmem_malloc(packed_bytes);
#endif
    if (device_recv == nullptr)
      checkCudaErrors(cudaMalloc(&device_recv, packed_bytes));
  }

  ~CoarseHaloExchange() override { release(); }

  void *slot(void *base, int direction) const {
    return static_cast<void *>(static_cast<char *>(base) +
                               static_cast<size_t>(direction) *
                                   direction_stride * sizeof(LatticeComplex<U>));
  }

  const void *slot_const(const void *base, int direction) const {
    return static_cast<const void *>(static_cast<const char *>(base) +
                                     static_cast<size_t>(direction) *
                                         direction_stride *
                                             sizeof(LatticeComplex<U>));
  }

  void *real_slot(void *base, int direction) const {
    return static_cast<void *>(static_cast<U *>(slot(base, direction)));
  }

  const void *real_slot_const(const void *base, int direction) const {
    return static_cast<const void *>(
        static_cast<const U *>(slot_const(base, direction)));
  }

  void wait_requests() {
    if (!request_active) return;
    checkMpiErrors(MPI_Waitall(32, recv_request, MPI_STATUSES_IGNORE));
    checkMpiErrors(MPI_Waitall(32, send_request, MPI_STATUSES_IGNORE));
    request_active = false;
  }

  void begin(void *device_in, cudaStream_t compute_stream) override {
    if (device_in == nullptr || compute_stream == nullptr)
      throw std::invalid_argument("invalid coarse halo input or stream");
    if (request_active)
      throw std::logic_error("coarse halo exchange is already in flight");

    use_device_mpi = !use_nvshmem && qcu_mpi_can_use_buffer<U>(device_send);
    checkCudaErrors(cudaEventRecord(input_ready, compute_stream));
    checkCudaErrors(cudaStreamWaitEvent(exchange_stream, input_ready, 0));
    const long long total = static_cast<long long>(packed_elements);
    const int blocks = static_cast<int>((total + _BLOCK_SIZE_ - 1) /
                                        static_cast<long long>(_BLOCK_SIZE_));
    multigrid_coarse_pack_halo<U><<<blocks, _BLOCK_SIZE_, 0, exchange_stream>>>(
        device_send, device_in, E, X, Y, Z, Lt, static_cast<int>(max_face));
    checkCudaErrors(cudaGetLastError());

    // A self-neighbour is a periodic wrap inside this rank.  Populate its
    // receive slot with a device copy; it must not be overwritten by the
    // host-receive staging path below.
    for (int h = 0; h < 32; ++h) {
      const int source_direction = h ^ 1;
      if (peer[h] == rank) {
        const size_t bytes = static_cast<size_t>(real_count[h]) * sizeof(U);
        checkCudaErrors(cudaMemcpyAsync(
            slot(device_recv, h), slot_const(device_send, source_direction),
            bytes, cudaMemcpyDeviceToDevice, exchange_stream));
      }
    }

    if (use_nvshmem) {
#if defined(QCU_HAVE_NVSHMEM)
      multigrid_nvshmem_put_halo<U><<<1, 32, 0, exchange_stream>>>(
          device_recv, device_send, E, X, Y, Z, Lt,
          static_cast<int>(max_face), grid[0], grid[1], grid[2], grid[3],
          coord[0], coord[1], coord[2], coord[3], rank);
      checkCudaErrors(cudaGetLastError());
#endif
    } else if (!use_device_mpi) {
      // The pinned host buffer is used only after the pack kernel has written
      // the device buffer.  Synchronizing this private stream makes it safe
      // for MPI to read host_send while the main stream computes the interior.
      checkCudaErrors(cudaMemcpyAsync(host_send, device_send, packed_bytes,
                                      cudaMemcpyDeviceToHost, exchange_stream));
    }
    checkCudaErrors(cudaStreamSynchronize(exchange_stream));

    if (!use_nvshmem) {
      // Receive slot h is the data sent in direction h^1 by rank peer[h].
      // Post receives first, then sends, so small and large MPI messages have
      // the same deadlock-free ordering on every implementation.
      for (int h = 0; h < 32; ++h) {
        const int source_direction = h ^ 1;
        if (peer[h] == rank) continue;
        void *buffer = use_device_mpi ? real_slot(device_recv, h)
                                      : real_slot(host_recv, h);
        checkMpiErrors(_MPI_Irecv<U>(
            buffer, real_count[h], peer[h], 700 + source_direction,
            MPI_COMM_WORLD, &recv_request[h]));
      }
      for (int d = 0; d < 32; ++d) {
        if (peer[d] == rank) continue;
        const void *buffer = use_device_mpi ? real_slot_const(device_send, d)
                                            : real_slot_const(host_send, d);
        checkMpiErrors(_MPI_Isend<U>(buffer, real_count[d], peer[d], 700 + d,
                                     MPI_COMM_WORLD, &send_request[d]));
      }
    }
    request_active = true;
  }

  void finish(cudaStream_t compute_stream) override {
    if (!request_active) return;
    if (use_nvshmem) {
#if defined(QCU_HAVE_NVSHMEM)
      // The put kernel has already called nvshmem_quiet().  The barrier makes
      // the remote writes visible before unpack reads the symmetric buffer.
      nvshmem_barrier_all();
#endif
    } else {
      wait_requests();
      if (!use_device_mpi) {
        for (int h = 0; h < 32; ++h) {
          const int source_direction = h ^ 1;
          if (peer[h] == rank) continue;
          const size_t bytes = static_cast<size_t>(real_count[h]) * sizeof(U);
          checkCudaErrors(cudaMemcpyAsync(
              slot(device_recv, h), slot_const(host_recv, h), bytes,
              cudaMemcpyHostToDevice, exchange_stream));
        }
      }
    }

    const long long total = static_cast<long long>(packed_elements);
    const int blocks = static_cast<int>((total + _BLOCK_SIZE_ - 1) /
                                        static_cast<long long>(_BLOCK_SIZE_));
    multigrid_coarse_unpack_halo<U><<<blocks, _BLOCK_SIZE_, 0, exchange_stream>>>(
        halo, device_recv, E, X, Y, Z, Lt, static_cast<int>(max_face));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(halo_ready, exchange_stream));
    checkCudaErrors(cudaStreamWaitEvent(compute_stream, halo_ready, 0));
    request_active = false;
  }

  void release() override {
    if (exchange_stream == nullptr && device_send == nullptr &&
        device_recv == nullptr && halo == nullptr && host_send == nullptr &&
        host_recv == nullptr)
      return;
    if (request_active) {
      if (use_nvshmem) {
#if defined(QCU_HAVE_NVSHMEM)
        nvshmem_barrier_all();
#endif
      } else {
        wait_requests();
      }
    }
    if (exchange_stream != nullptr)
      checkCudaErrors(cudaStreamSynchronize(exchange_stream));
#if defined(QCU_HAVE_NVSHMEM)
    if (use_nvshmem && device_recv != nullptr) {
      nvshmem_free(device_recv);
      device_recv = nullptr;
    }
#endif
    if (device_send != nullptr) {
      checkCudaErrors(cudaFree(device_send));
      device_send = nullptr;
    }
    if (device_recv != nullptr) {
      checkCudaErrors(cudaFree(device_recv));
      device_recv = nullptr;
    }
    if (halo != nullptr) {
      checkCudaErrors(cudaFree(halo));
      halo = nullptr;
    }
    if (host_send != nullptr) {
      checkCudaErrors(cudaFreeHost(host_send));
      host_send = nullptr;
    }
    if (host_recv != nullptr) {
      checkCudaErrors(cudaFreeHost(host_recv));
      host_recv = nullptr;
    }
    if (input_ready != nullptr) {
      checkCudaErrors(cudaEventDestroy(input_ready));
      input_ready = nullptr;
    }
    if (halo_ready != nullptr) {
      checkCudaErrors(cudaEventDestroy(halo_ready));
      halo_ready = nullptr;
    }
    if (exchange_stream != nullptr) {
      checkCudaErrors(cudaStreamDestroy(exchange_stream));
      exchange_stream = nullptr;
    }
    request_active = false;
  }

  void *device_halo() const override { return halo; }
  bool pending() const override { return request_active; }
  const char *transport_name() const override {
    if (use_nvshmem) return "nvshmem";
    return use_device_mpi ? "cuda-aware" : "pinned-host";
  }
};

template <typename SetT> struct CoarseHaloExchangeAny {
  CoarseHaloExchangeBase *impl = nullptr;

  ~CoarseHaloExchangeAny() { release(); }

  void init(LatticeSet<SetT> *set, int data_type, int E, int X, int Y, int Z,
            int Lt) {
    release();
    if (data_type == _LAT_C64_)
      impl = new CoarseHaloExchange<float, SetT>(set, E, X, Y, Z, Lt);
    else if (data_type == _LAT_C128_)
      impl = new CoarseHaloExchange<double, SetT>(set, E, X, Y, Z, Lt);
    else
      throw std::invalid_argument("unsupported coarse halo precision");
  }

  void begin(void *device_in, cudaStream_t compute_stream) {
    if (impl == nullptr) throw std::logic_error("coarse halo is not initialized");
    impl->begin(device_in, compute_stream);
  }
  void finish(cudaStream_t compute_stream) {
    if (impl == nullptr) throw std::logic_error("coarse halo is not initialized");
    impl->finish(compute_stream);
  }
  void release() {
    if (impl != nullptr) {
      impl->release();
      delete impl;
      impl = nullptr;
    }
  }
  bool valid() const { return impl != nullptr; }
  void *device_halo() const { return impl == nullptr ? nullptr : impl->device_halo(); }
  const char *transport_name() const {
    return impl == nullptr ? "disabled" : impl->transport_name();
  }
};

template <typename T> struct LatticeMultigridRestrict {
  LatticeSet<T> *set_ptr;
  cudaError_t err;
  void give(LatticeSet<T> *_set_ptr) { set_ptr = _set_ptr; }
  void run(void *coarse_out, void *fine_in, void *null_vecs, int E, int e,
           int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc) {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    int total_output = E * Xc * Yc * Zc * Tc;
    dim3 gridDim((total_output + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
    dim3 blockDim(_BLOCK_SIZE_);
    multigrid_restrict<T><<<gridDim, blockDim, 0, set_ptr->stream>>>(
        coarse_out, fine_in, null_vecs, E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
};
template <typename T> struct LatticeMultigridProLong {
  LatticeSet<T> *set_ptr;
  cudaError_t err;
  void give(LatticeSet<T> *_set_ptr) { set_ptr = _set_ptr; }
  void run(void *fine_out, void *coarse_in, void *null_vecs, int E, int e,
           int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc) {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    int total_output = e * Xf * Yf * Zf * Tf;
    dim3 gridDim((total_output + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
    dim3 blockDim(_BLOCK_SIZE_);
    multigrid_prolong<T><<<gridDim, blockDim, 0, set_ptr->stream>>>(
        fine_out, coarse_in, null_vecs, E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
};
template <typename T> struct LatticeMultigridCoarseDslash {
  LatticeSet<T> *set_ptr;
  cudaError_t err;
  void give(LatticeSet<T> *_set_ptr) { set_ptr = _set_ptr; }
  void run(void *fermion_out, void *fermion_in, void *hopping, void *sitting,
           int E, int X, int Y, int Z, int Lt) {
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    int total_output = E * X * Y * Z * Lt;
    dim3 gridDim((total_output + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
    dim3 blockDim(_BLOCK_SIZE_);
    multigrid_coarse_dslash<T><<<gridDim, blockDim, 0, set_ptr->stream>>>(
        fermion_out, fermion_in, hopping, sitting, E, X, Y, Z, Lt);
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
};
template <typename T> struct LatticeMultigridCoarseDslashWide {
  LatticeSet<T> *set_ptr;
  cudaError_t err;
  void *halo_device = nullptr;
  void *input_host = nullptr;
  void *halo_host = nullptr;

  ~LatticeMultigridCoarseDslashWide() { release_halo(); }

  void release_halo() {
    if (halo_device != nullptr) {
      cudaFree(halo_device);
      halo_device = nullptr;
    }
    if (input_host != nullptr) {
      cudaFreeHost(input_host);
      input_host = nullptr;
    }
    if (halo_host != nullptr) {
      cudaFreeHost(halo_host);
      halo_host = nullptr;
    }
  }

  void ensure_halo(int E, int X, int Y, int Z, int Lt) {
    const size_t hvol = static_cast<size_t>(X + 2) * static_cast<size_t>(Y + 2) *
                        static_cast<size_t>(Z + 2) * static_cast<size_t>(Lt + 2);
    const size_t local_bytes = static_cast<size_t>(E) * static_cast<size_t>(X) *
                               static_cast<size_t>(Y) * static_cast<size_t>(Z) *
                               static_cast<size_t>(Lt) * sizeof(LatticeComplex<T>);
    const size_t halo_bytes = static_cast<size_t>(E) * hvol *
                              sizeof(LatticeComplex<T>);
    if (halo_device != nullptr) return;
    checkCudaErrors(cudaMalloc(&halo_device, halo_bytes));
    checkCudaErrors(cudaMallocHost(&input_host, local_bytes));
    checkCudaErrors(cudaMallocHost(&halo_host, halo_bytes));
  }

  void give(LatticeSet<T> *_set_ptr) { set_ptr = _set_ptr; }
  void run(void *fermion_out, void *fermion_in, void *sitting, void *hop_nn,
           void *hop_diag, int E, int X, int Y, int Z, int Lt) {
    if (set_ptr == nullptr)
      throw std::logic_error("coarse dslash has no lattice set");
    int total_output = E * X * Y * Z * Lt;
    dim3 gridDim((total_output + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
    dim3 blockDim(_BLOCK_SIZE_);
    const bool distributed = set_ptr->host_params[_GRID_X_] > 1 ||
                             set_ptr->host_params[_GRID_Y_] > 1 ||
                             set_ptr->host_params[_GRID_Z_] > 1 ||
                             set_ptr->host_params[_GRID_T_] > 1;
    if (distributed) {
      // Keep this public one-shot entry point on the same transport and
      // overlap implementation as the recursive Clover-MG path.  The
      // exchange object is intentionally local: this API has no persistent
      // MG hierarchy, and its destructor releases all staging state after
      // the final stream synchronization below.
      CoarseHaloExchange<T, T> exchange(set_ptr, E, X, Y, Z, Lt);
      exchange.begin(fermion_in, set_ptr->stream);
      if (qcu_mpi_overlap_enabled()) {
        multigrid_coarse_dslash_wide_halo_region<T>
            <<<gridDim, blockDim, 0, set_ptr->stream>>>(
                fermion_out, fermion_in, exchange.device_halo(), sitting,
                hop_nn, hop_diag, E, X, Y, Z, Lt, 0);
        exchange.finish(set_ptr->stream);
        multigrid_coarse_dslash_wide_halo_region<T>
            <<<gridDim, blockDim, 0, set_ptr->stream>>>(
                fermion_out, fermion_in, exchange.device_halo(), sitting,
                hop_nn, hop_diag, E, X, Y, Z, Lt, 1);
      } else {
        exchange.finish(set_ptr->stream);
        multigrid_coarse_dslash_wide_halo<T>
            <<<gridDim, blockDim, 0, set_ptr->stream>>>(
                fermion_out, fermion_in, exchange.device_halo(), sitting,
                hop_nn, hop_diag, E, X, Y, Z, Lt);
      }
    } else {
      multigrid_coarse_dslash_wide<T><<<gridDim, blockDim, 0,
                                         set_ptr->stream>>>(
          fermion_out, fermion_in, sitting, hop_nn, hop_diag, E, X, Y, Z,
          Lt);
    }
    err = cudaGetLastError();
    checkCudaErrors(err);
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
  }
};
} // namespace qcu
#endif
