#ifndef _LATTICE_MPI_H
#define _LATTICE_MPI_H
#include "./include.h"

// OpenMPI and MPICH expose the CUDA-aware build/runtime query in their
// vendor extension header.  It is deliberately optional: the ordinary MPI
// ABI remains sufficient for the pinned-host fallback path.
#if defined(__has_include)
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#endif

#if defined(QCU_HAVE_NVSHMEM)
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

namespace qcu {
template <typename T>
int _MPI_Isend(const void *buf, int count, int dest, int tag, MPI_Comm comm,
               MPI_Request *request);
template <typename T>
int _MPI_Irecv(void *buf, int count, int source, int tag, MPI_Comm comm,
               MPI_Request *request);
template <typename T>
int _MPI_Sendrecv(const void *sendbuf, int sendcount, int dest, int sendtag,
                  void *recvbuf, int recvcount, int source, int recvtag,
                  MPI_Comm comm, MPI_Status *status);
template <typename T>
int _MPI_Allreduce(const void *sendbuf, void *recvbuf, int count, MPI_Op op,
                   MPI_Comm comm);

// ---------------------------------------------------------------------------
// Runtime communication policy
// ---------------------------------------------------------------------------
//
// PYQCU_MPI_DEVICE_AWARE controls whether CUDA buffers may be handed to MPI:
//   auto (default): use the vendor compile-time CUDA-aware capability query;
//   0/off         : force pinned-host staging;
//   1/on          : request CUDA-aware MPI, but still refuse it when the MPI
//                   library advertises no CUDA support.
//
// PYQCU_MPI_OVERLAP controls the coarse-grid interior/boundary split.  It is
// enabled by default for distributed runs and can be disabled independently
// of the transport selected above.
inline bool qcu_parse_bool_env(const char *name, bool default_value) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return default_value;
  if (std::strcmp(value, "0") == 0 || std::strcmp(value, "off") == 0 ||
      std::strcmp(value, "false") == 0 || std::strcmp(value, "no") == 0)
    return false;
  if (std::strcmp(value, "1") == 0 || std::strcmp(value, "on") == 0 ||
      std::strcmp(value, "true") == 0 || std::strcmp(value, "yes") == 0)
    return true;
  return default_value;
}

inline bool qcu_mpi_compile_cuda_aware() {
#if defined(MPIX_CUDA_AWARE_SUPPORT)
  return MPIX_CUDA_AWARE_SUPPORT != 0;
#elif defined(MPICH_GPU_SUPPORT_ENABLED)
  return true;
#elif defined(OMPI_HAVE_MPI_EXT_CUDA)
  return MPIX_Query_cuda_support() != 0;
#else
  // OMPI_HAVE_MPI_EXT_CUDA only says that the extension header was built; in
  // particular, OpenMPI defines it even when MPIX_CUDA_AWARE_SUPPORT is 0.
  // Treating that macro as sufficient would send device pointers into a
  // non-CUDA-aware MPI and usually ends in an invalid host access.
  return false;
#endif
}

inline bool qcu_mpi_device_aware() {
  const char *forced = std::getenv("PYQCU_MPI_DEVICE_AWARE");
  if (forced != nullptr &&
      (std::strcmp(forced, "0") == 0 || std::strcmp(forced, "off") == 0 ||
       std::strcmp(forced, "false") == 0))
    return false;

  // A forced 'on' is still guarded by the MPI vendor capability bit.  This
  // makes a deployment typo fail safe instead of corrupting a host buffer.
  // The capability is process-global, so caching it also avoids repeated
  // vendor-header logic in every coarse exchange.
  static std::atomic<int> cached(-1);
  int value = cached.load(std::memory_order_acquire);
  if (value < 0) {
    value = qcu_mpi_compile_cuda_aware() ? 1 : 0;
    cached.store(value, std::memory_order_release);
  }
  return value != 0;
}

inline bool qcu_mpi_overlap_enabled() {
  return qcu_parse_bool_env("PYQCU_MPI_OVERLAP", true);
}

inline bool qcu_cuda_pointer_is_device(const void *ptr) {
  if (ptr == nullptr) return false;
  cudaPointerAttributes attributes;
  cudaError_t status = cudaPointerGetAttributes(&attributes, ptr);
  if (status != cudaSuccess) {
    // cudaPointerGetAttributes leaves an error in the runtime for ordinary
    // host pointers on some CUDA versions.  Clear it before the caller uses
    // another CUDA API; host staging is the expected result here.
    (void)cudaGetLastError();
    return false;
  }
#if CUDART_VERSION >= 10000
  return attributes.type == cudaMemoryTypeDevice ||
         attributes.type == cudaMemoryTypeManaged;
#else
  return attributes.memoryType == cudaMemoryTypeDevice;
#endif
}

template <typename T>
inline bool qcu_mpi_can_use_buffer(const void *ptr) {
  (void)sizeof(T);
  return qcu_mpi_device_aware() && qcu_cuda_pointer_is_device(ptr);
}

inline const char *qcu_mpi_transport_name() {
  return qcu_mpi_device_aware() ? "cuda-aware" : "pinned-host";
}

// ---------------------------------------------------------------------------
// Optional NVSHMEM runtime
// ---------------------------------------------------------------------------
// NVSHMEM is intentionally opt-in at both build and runtime.  The MPI rank
// order is used as the PE order; if the launcher exposes a different number
// of PEs, the caller falls back to MPI before allocating a symmetric buffer.
#if defined(QCU_HAVE_NVSHMEM)
inline bool qcu_nvshmem_requested() {
  return qcu_parse_bool_env("PYQCU_NVSHMEM", false);
}

inline bool qcu_nvshmem_init_for_mpi(int expected_pes) {
  if (!qcu_nvshmem_requested()) return false;
  static std::mutex mutex;
  static bool initialized = false;
  std::lock_guard<std::mutex> guard(mutex);
  if (!initialized) {
    // Bind NVSHMEM to the already initialized MPI communicator.  Plain
    // nvshmem_init() may select a PMI/bootstrap world that is not identical
    // to MPI_COMM_WORLD (notably under OpenMPI), which would make PE numbers
    // disagree with the coarse rank map.
    MPI_Comm comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr;
    attr.mpi_comm = &comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    initialized = true;
  }
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return nvshmem_n_pes() == expected_pes && nvshmem_my_pe() == rank;
}
#endif
} // namespace qcu
#endif
