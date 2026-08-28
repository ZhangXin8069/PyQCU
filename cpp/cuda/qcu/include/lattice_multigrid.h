#ifndef _LATTICE_MULTIGRID_H
#define _LATTICE_MULTIGRID_H
#include "./define.h"
#include "./lattice_set.h"
#include "./lattice_mpi.h"
#include "./multigrid.h"
#include <limits>
#include <vector>
namespace qcu {

// Exchange the one-layer halo required by the 33-point coarse stencil.
//
// The coarse vector is stored as [E, X, Y, Z, T] in C order.  ``halo_host``
// uses [E, X+2, Y+2, Z+2, T+2], with the 32 entries
//   8 axial neighbours + 24 two-axis diagonal neighbours.
// Only the boundary planes/edges/corners that can be reached by one stencil
// hop are exchanged.  The helper deliberately uses host staging: it keeps
// the MPI ABI identical to the rest of this backend (real MPI_FLOAT/DOUBLE)
// and is valid for arbitrary process-grid dimensions, including periodic
// self-neighbours.
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

template <typename U, typename SetT>
inline void multigrid_exchange_coarse_halo(
    LatticeSet<SetT> *set, const void *device_in, int E, int X, int Y, int Z,
    int Lt, void *input_host_ptr, void *halo_host_ptr, void *halo_device) {
  if (set == nullptr || device_in == nullptr || input_host_ptr == nullptr ||
      halo_host_ptr == nullptr || halo_device == nullptr || E <= 0 || X <= 0 ||
      Y <= 0 || Z <= 0 || Lt <= 0) {
    throw std::invalid_argument("invalid coarse MPI halo arguments");
  }

  const int dims[4] = {X, Y, Z, Lt};
  const int vol = X * Y * Z * Lt;
  const size_t hvol = static_cast<size_t>(X + 2) * static_cast<size_t>(Y + 2) *
                      static_cast<size_t>(Z + 2) * static_cast<size_t>(Lt + 2);
  const size_t elem_bytes = sizeof(LatticeComplex<U>);
  LatticeComplex<U> *local = static_cast<LatticeComplex<U> *>(input_host_ptr);
  LatticeComplex<U> *halo = static_cast<LatticeComplex<U> *>(halo_host_ptr);

  checkCudaErrors(cudaMemcpyAsync(
      local, device_in, static_cast<size_t>(vol) * static_cast<size_t>(E) *
                            elem_bytes,
      cudaMemcpyDeviceToHost, set->stream));
  checkCudaErrors(cudaStreamSynchronize(set->stream));
  std::memset(halo, 0, hvol * static_cast<size_t>(E) * elem_bytes);

  static const int canonical[16][4] = {
      {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1},
      {1, 1, 0, 0}, {1, -1, 0, 0}, {1, 0, 1, 0}, {1, 0, -1, 0},
      {1, 0, 0, 1}, {1, 0, 0, -1}, {0, 1, 1, 0}, {0, 1, -1, 0},
      {0, 1, 0, 1}, {0, 1, 0, -1}, {0, 0, 1, 1}, {0, 0, 1, -1}};

  auto flat_site = [Y, Z, Lt](int x, int y, int z, int t) {
    return ((x * Y + y) * Z + z) * Lt + t;
  };
  auto halo_site = [X, Y, Z, Lt](int x, int y, int z, int t) {
    return (((x * (Y + 2) + y) * (Z + 2) + z) * (Lt + 2) + t);
  };
  auto free_volume = [&](const int shift[4]) {
    size_t result = 1;
    for (int d = 0; d < 4; ++d)
      if (shift[d] == 0) result *= static_cast<size_t>(dims[d]);
    return result;
  };
  size_t max_face = 0;
  for (const auto &shift : canonical)
    max_face = max_face > free_volume(shift) ? max_face : free_volume(shift);
  const size_t max_elements = static_cast<size_t>(E) * max_face;
  if (max_elements > static_cast<size_t>(std::numeric_limits<int>::max() / 2))
    throw std::invalid_argument("coarse MPI halo message is too large");
  std::vector<LatticeComplex<U>> send(max_elements);
  std::vector<LatticeComplex<U>> recv(max_elements);

  auto boundary_copy = [&](const int shift[4], bool send_side,
                           LatticeComplex<U> *buffer) {
    int free_dims[4], nfree = 0;
    for (int d = 0; d < 4; ++d)
      if (shift[d] == 0) free_dims[nfree++] = d;
    const size_t face_sites = free_volume(shift);
    for (int e = 0; e < E; ++e) {
      for (size_t k = 0; k < face_sites; ++k) {
        size_t q = k;
        int c[4] = {0, 0, 0, 0};
        for (int i = nfree - 1; i >= 0; --i) {
          const int d = free_dims[i];
          c[d] = static_cast<int>(q % static_cast<size_t>(dims[d]));
          q /= static_cast<size_t>(dims[d]);
        }
        for (int d = 0; d < 4; ++d) {
          if (shift[d] != 0) {
            const bool positive = shift[d] > 0;
            c[d] = send_side ? (positive ? dims[d] - 1 : 0)
                             : (positive ? 0 : dims[d] - 1);
          }
        }
        buffer[static_cast<size_t>(e) * face_sites + k] =
            local[static_cast<size_t>(e) * static_cast<size_t>(vol) +
                  static_cast<size_t>(flat_site(c[0], c[1], c[2], c[3]))];
      }
    }
  };
  auto halo_store = [&](const int shift[4], const LatticeComplex<U> *buffer) {
    int free_dims[4], nfree = 0;
    for (int d = 0; d < 4; ++d)
      if (shift[d] == 0) free_dims[nfree++] = d;
    const size_t face_sites = free_volume(shift);
    for (int e = 0; e < E; ++e) {
      for (size_t k = 0; k < face_sites; ++k) {
        size_t q = k;
        int c[4] = {1, 1, 1, 1};
        for (int i = nfree - 1; i >= 0; --i) {
          const int d = free_dims[i];
          c[d] = static_cast<int>(q % static_cast<size_t>(dims[d])) + 1;
          q /= static_cast<size_t>(dims[d]);
        }
        for (int d = 0; d < 4; ++d)
          if (shift[d] != 0) c[d] = shift[d] > 0 ? dims[d] + 1 : 0;
        halo[static_cast<size_t>(e) * hvol +
             static_cast<size_t>(halo_site(c[0], c[1], c[2], c[3]))] =
            buffer[static_cast<size_t>(e) * face_sites + k];
      }
    }
  };

  const int rank = set->host_params[_NODE_RANK_];
  for (int pair = 0; pair < 16; ++pair) {
    const int *q = canonical[pair];
    const int nq[4] = {-q[0], -q[1], -q[2], -q[3]};
    const size_t face_sites = free_volume(q);
    const size_t face_elements = static_cast<size_t>(E) * face_sites;
    const int qrank = multigrid_coarse_rank_shift(set, q);
    const int nrank = multigrid_coarse_rank_shift(set, nq);
    const int tag = 200 + pair;

    boundary_copy(q, true, send.data());
    if (qrank == rank && nrank == rank) {
      // Both neighbours are local periodic images.  The source for the
      // receiver's -q halo is the local q-side boundary itself; using
      // boundary_copy(q, false) here reverses the wrap and was the source of
      // the large error on partially distributed grids.
      halo_store(nq, send.data());
    } else {
      checkMpiErrors(_MPI_Sendrecv<U>(
          send.data(), static_cast<int>(2 * face_elements), qrank, tag,
          recv.data(), static_cast<int>(2 * face_elements), nrank, tag,
          MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      halo_store(nq, recv.data());
    }

    boundary_copy(nq, true, send.data());
    if (qrank == rank && nrank == rank) {
      halo_store(q, send.data());
    } else {
      checkMpiErrors(_MPI_Sendrecv<U>(
          send.data(), static_cast<int>(2 * face_elements), nrank, tag,
          recv.data(), static_cast<int>(2 * face_elements), qrank, tag,
          MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      halo_store(q, recv.data());
    }
  }
  checkCudaErrors(cudaMemcpyAsync(
      halo_device, halo, hvol * static_cast<size_t>(E) * elem_bytes,
      cudaMemcpyHostToDevice, set->stream));
}

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
    checkCudaErrors(cudaStreamSynchronize(set_ptr->stream));
    int total_output = E * X * Y * Z * Lt;
    dim3 gridDim((total_output + _BLOCK_SIZE_ - 1) / _BLOCK_SIZE_);
    dim3 blockDim(_BLOCK_SIZE_);
    const bool distributed = set_ptr->host_params[_GRID_X_] > 1 ||
                             set_ptr->host_params[_GRID_Y_] > 1 ||
                             set_ptr->host_params[_GRID_Z_] > 1 ||
                             set_ptr->host_params[_GRID_T_] > 1;
    if (distributed) {
      ensure_halo(E, X, Y, Z, Lt);
      multigrid_exchange_coarse_halo<T, T>(
          set_ptr, fermion_in, E, X, Y, Z, Lt, input_host, halo_host,
          halo_device);
      multigrid_coarse_dslash_wide_halo<T><<<gridDim, blockDim, 0,
                                              set_ptr->stream>>>(
          fermion_out, fermion_in, halo_device, sitting, hop_nn, hop_diag,
          E, X, Y, Z, Lt);
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
