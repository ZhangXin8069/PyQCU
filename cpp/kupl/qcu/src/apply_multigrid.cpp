#include "kupl/qcu.h"

#include "api_common.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <vector>

namespace {

template <typename Real>
int local_orthogonalize_cgs2_impl(
    void *flat_in, void *blocked_out, int E, int e, int Xf, int Yf, int Zf,
    int Tf, int Xc, int Yc, int Zc, int Tc) {
  if (flat_in == nullptr || blocked_out == nullptr || E <= 0 || e <= 0 ||
      Xf <= 0 || Yf <= 0 || Zf <= 0 || Tf <= 0 || Xc <= 0 || Yc <= 0 ||
      Zc <= 0 || Tc <= 0 || Xf % Xc || Yf % Yc || Zf % Zc || Tf % Tc) {
    return 1;
  }
  const int bx = Xf / Xc;
  const int by = Yf / Yc;
  const int bz = Zf / Zc;
  const int bt = Tf / Tc;
  const int local_dim = e * bx * by * bz * bt;
  if (E > local_dim) return 3;
  const int n_blocks = Xc * Yc * Zc * Tc;
  auto *input = static_cast<const kupl::qcu::Complex<Real> *>(flat_in);
  auto *output = static_cast<kupl::qcu::Complex<Real> *>(blocked_out);
  std::atomic<int> failed{0};

  auto block_coord = [&](int block) {
    std::array<int, 4> coordinate{};
    coordinate[3] = block % Tc;
    block /= Tc;
    coordinate[2] = block % Zc;
    block /= Zc;
    coordinate[1] = block % Yc;
    coordinate[0] = block / Yc;
    return coordinate;
  };
  auto local_coord = [&](int local) {
    std::array<int, 5> coordinate{};
    coordinate[4] = local % bt;
    local /= bt;
    coordinate[3] = local % bz;
    local /= bz;
    coordinate[2] = local % by;
    local /= by;
    coordinate[1] = local % bx;
    coordinate[0] = local / bx;
    return coordinate;
  };
  auto flat_index = [&](int vector, int block, int local) {
    const auto block_xyz = block_coord(block);
    const auto local_exyzt = local_coord(local);
    std::size_t index = static_cast<std::size_t>(vector);
    index = index * e + local_exyzt[0];
    index = index * Xf + (block_xyz[0] * bx + local_exyzt[1]);
    index = index * Yf + (block_xyz[1] * by + local_exyzt[2]);
    index = index * Zf + (block_xyz[2] * bz + local_exyzt[3]);
    return index * Tf + (block_xyz[3] * bt + local_exyzt[4]);
  };
  auto blocked_index = [&](int vector, int block, int local) {
    const auto block_xyz = block_coord(block);
    const auto local_exyzt = local_coord(local);
    std::size_t index = static_cast<std::size_t>(vector);
    index = index * e + local_exyzt[0];
    index = index * Xc + block_xyz[0];
    index = index * bx + local_exyzt[1];
    index = index * Yc + block_xyz[1];
    index = index * by + local_exyzt[2];
    index = index * Zc + block_xyz[2];
    index = index * bz + local_exyzt[3];
    index = index * Tc + block_xyz[3];
    return index * bt + local_exyzt[4];
  };

  #pragma omp parallel
  {
    std::vector<kupl::qcu::Complex<Real>> work(local_dim);
    std::vector<kupl::qcu::Complex<Real>> orthonormal(
        static_cast<std::size_t>(E) * local_dim);
    #pragma omp for schedule(static)
    for (int block = 0; block < n_blocks; ++block) {
      for (int column = 0; column < E; ++column) {
        for (int local = 0; local < local_dim; ++local) {
          work[local] = input[flat_index(column, block, local)];
        }
        for (int pass = 0; pass < 2; ++pass) {
          for (int previous = 0; previous < column; ++previous) {
            kupl::qcu::Complex<Real> coefficient(0, 0);
            for (int local = 0; local < local_dim; ++local) {
              coefficient += std::conj(
                  orthonormal[static_cast<std::size_t>(previous) * local_dim +
                              local]) * work[local];
            }
            for (int local = 0; local < local_dim; ++local) {
              work[local] -= coefficient *
                  orthonormal[static_cast<std::size_t>(previous) * local_dim +
                              local];
            }
          }
        }
        Real norm2 = 0;
        for (const auto &value : work) norm2 += std::norm(value);
        const Real threshold = std::numeric_limits<Real>::epsilon() *
                               static_cast<Real>(local_dim > E ? local_dim : E);
        if (!std::isfinite(norm2) || norm2 <= threshold * threshold) {
          failed.store(1);
          for (int local = 0; local < local_dim; ++local) {
            orthonormal[static_cast<std::size_t>(column) * local_dim + local] =
                kupl::qcu::Complex<Real>(0, 0);
          }
          continue;
        }
        const Real norm = std::sqrt(norm2);
        for (int local = 0; local < local_dim; ++local) {
          orthonormal[static_cast<std::size_t>(column) * local_dim + local] =
              work[local] / norm;
        }
      }
      for (int column = 0; column < E; ++column) {
        for (int local = 0; local < local_dim; ++local) {
          output[blocked_index(column, block, local)] =
              orthonormal[static_cast<std::size_t>(column) * local_dim + local];
        }
      }
    }
  }
  return failed.load() ? 3 : 0;
}

} // namespace

extern "C" {

void applyMultigridRestrictQcu(long long _coarse_out, long long _fine_in,
                               long long _null_vectors, long long _set_ptrs,
                               long long _params) {
  kupl::qcu::api::guarded("MULTIGRID_RESTRICT", [&] {
    const auto *params = kupl::qcu::api::params_pointer(_params);
    kupl::qcu::api::lattice_set(_set_ptrs, _params)
        ->multigrid_restrict(
            kupl::qcu::api::mutable_pointer(_coarse_out),
            kupl::qcu::api::pointer(_fine_in),
            kupl::qcu::api::pointer(_null_vectors),
            params[kupl::qcu::kMgLevel1E], kupl::qcu::kVectorSize,
            params[kupl::qcu::kLatX], params[kupl::qcu::kLatY],
            params[kupl::qcu::kLatZ], params[kupl::qcu::kLatT] / 2,
            params[kupl::qcu::kMgLevel1X], params[kupl::qcu::kMgLevel1Y],
            params[kupl::qcu::kMgLevel1Z], params[kupl::qcu::kMgLevel1T]);
  });
}

void applyMultigridProLongQcu(long long _fine_out, long long _coarse_in,
                              long long _null_vectors, long long _set_ptrs,
                              long long _params) {
  kupl::qcu::api::guarded("MULTIGRID_PROLONG", [&] {
    const auto *params = kupl::qcu::api::params_pointer(_params);
    kupl::qcu::api::lattice_set(_set_ptrs, _params)
        ->multigrid_prolong(
            kupl::qcu::api::mutable_pointer(_fine_out),
            kupl::qcu::api::pointer(_coarse_in),
            kupl::qcu::api::pointer(_null_vectors),
            params[kupl::qcu::kMgLevel1E], kupl::qcu::kVectorSize,
            params[kupl::qcu::kLatX], params[kupl::qcu::kLatY],
            params[kupl::qcu::kLatZ], params[kupl::qcu::kLatT] / 2,
            params[kupl::qcu::kMgLevel1X], params[kupl::qcu::kMgLevel1Y],
            params[kupl::qcu::kMgLevel1Z], params[kupl::qcu::kMgLevel1T]);
  });
}

void applyMultigridCoarseDslashQcu(long long _out, long long _in,
                                   long long _hopping, long long _sitting,
                                   long long _set_ptrs, long long _params) {
  kupl::qcu::api::guarded("MULTIGRID_COARSE_DSLASH", [&] {
    const auto *params = kupl::qcu::api::params_pointer(_params);
    kupl::qcu::api::lattice_set(_set_ptrs, _params)
        ->multigrid_coarse_dslash(
            kupl::qcu::api::mutable_pointer(_out),
            kupl::qcu::api::pointer(_in),
            kupl::qcu::api::pointer(_hopping),
            kupl::qcu::api::pointer(_sitting),
            params[kupl::qcu::kMgLevel1E], params[kupl::qcu::kMgLevel1X],
            params[kupl::qcu::kMgLevel1Y], params[kupl::qcu::kMgLevel1Z],
            params[kupl::qcu::kMgLevel1T]);
  });
}

void applyMultigridCoarseDslashWideQcu(
    long long _out, long long _in, long long _sitting, long long _hop_nn,
    long long _hop_diag, long long _set_ptrs, long long _params) {
  kupl::qcu::api::guarded("MULTIGRID_COARSE_DSLASH_WIDE", [&] {
    const auto *params = kupl::qcu::api::params_pointer(_params);
    kupl::qcu::api::lattice_set(_set_ptrs, _params)
        ->multigrid_coarse_dslash_wide(
            kupl::qcu::api::mutable_pointer(_out),
            kupl::qcu::api::pointer(_in),
            kupl::qcu::api::pointer(_sitting),
            kupl::qcu::api::pointer(_hop_nn),
            kupl::qcu::api::pointer(_hop_diag),
            params[kupl::qcu::kMgLevel1E], params[kupl::qcu::kMgLevel1X],
            params[kupl::qcu::kMgLevel1Y], params[kupl::qcu::kMgLevel1Z],
            params[kupl::qcu::kMgLevel1T]);
  });
}

int applyMultigridLocalOrthogonalizeQcu(
    long long _flat_in, long long _blocked_out, int E, int e, int Xf, int Yf,
    int Zf, int Tf, int Xc, int Yc, int Zc, int Tc, int data_type) {
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_LOCAL_ORTHOGONALIZE", [&] {
        void *input = kupl::qcu::api::mutable_pointer(_flat_in);
        void *output = kupl::qcu::api::mutable_pointer(_blocked_out);
        if (input == nullptr || output == nullptr) {
          throw std::runtime_error("null-vector pointers are null");
        }
        if (data_type == kupl::qcu::kLatC64) {
          return local_orthogonalize_cgs2_impl<float>(
              input, output, E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
        }
        if (data_type == kupl::qcu::kLatC128) {
          return local_orthogonalize_cgs2_impl<double>(
              input, output, E, e, Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
        }
        throw std::runtime_error("unsupported data type");
      });
}

int applyMultigridBuildNullVectorsQcu(
    long long _blocked_out, long long _gauge, long long _clover_even,
    long long _clover_odd, long long _clover_even_inv,
    long long _clover_odd_inv, long long _set_ptrs, long long _params, int E,
    int e, int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc,
    int nv_iters, unsigned long long seed, double nv_tol) {
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_BUILD_NULL_VECTORS", [&] {
        auto set = kupl::qcu::api::lattice_set(_set_ptrs, _params);
        const auto *params = kupl::qcu::api::params_pointer(_params);
        const int data_type = params[kupl::qcu::kDataType];
        const int64_t count = static_cast<int64_t>(E) * e * Xf * Yf * Zf * Tf;
        if (count <= 0) {
          throw std::invalid_argument("invalid null-vector dimensions");
        }
        int status = 0;
        if (data_type == kupl::qcu::kLatC64) {
          std::vector<kupl::qcu::Complex<float>> flat(
              static_cast<std::size_t>(count));
          set->multigrid_build_null_vectors(
              flat.data(), kupl::qcu::api::pointer(_gauge),
              kupl::qcu::api::pointer(_clover_even),
              kupl::qcu::api::pointer(_clover_odd),
              kupl::qcu::api::pointer(_clover_even_inv),
              kupl::qcu::api::pointer(_clover_odd_inv), E, e, Xf, Yf, Zf, Tf,
              nv_iters, seed, nv_tol);
          status = local_orthogonalize_cgs2_impl<float>(
              flat.data(), kupl::qcu::api::mutable_pointer(_blocked_out), E, e,
              Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
        } else if (data_type == kupl::qcu::kLatC128) {
          std::vector<kupl::qcu::Complex<double>> flat(
              static_cast<std::size_t>(count));
          set->multigrid_build_null_vectors(
              flat.data(), kupl::qcu::api::pointer(_gauge),
              kupl::qcu::api::pointer(_clover_even),
              kupl::qcu::api::pointer(_clover_odd),
              kupl::qcu::api::pointer(_clover_even_inv),
              kupl::qcu::api::pointer(_clover_odd_inv), E, e, Xf, Yf, Zf, Tf,
              nv_iters, seed, nv_tol);
          status = local_orthogonalize_cgs2_impl<double>(
              flat.data(), kupl::qcu::api::mutable_pointer(_blocked_out), E, e,
              Xf, Yf, Zf, Tf, Xc, Yc, Zc, Tc);
        } else {
          throw std::runtime_error("unsupported data type");
        }
        return status;
      });
}

int applyMultigridBuildStencilLocalQcu(
    long long _sitting, long long _hop_nn, long long _hop_diag,
    long long _null_vectors, long long _gauge, long long _clover_even,
    long long _clover_odd, long long _clover_even_inv,
    long long _clover_odd_inv, long long _set_ptrs, long long _params, int E,
    int e, int Xf, int Yf, int Zf, int Tf, int Xc, int Yc, int Zc, int Tc,
    int window, int batch_size) {
  return kupl::qcu::api::status_guarded(
      "MULTIGRID_BUILD_STENCIL_LOCAL", [&] {
        auto set = kupl::qcu::api::lattice_set(_set_ptrs, _params);
        if (_sitting == 0 || _hop_nn == 0 || _hop_diag == 0 ||
            _null_vectors == 0) {
          throw std::runtime_error("local stencil output pointers are null");
        }
        if (batch_size <= 0) {
          throw std::invalid_argument("stencil batch size must be positive");
        }
        (void)batch_size; // KUPL's implementation already owns OpenMP batching.
        set->multigrid_build_stencil_local(
            kupl::qcu::api::mutable_pointer(_sitting),
            kupl::qcu::api::mutable_pointer(_hop_nn),
            kupl::qcu::api::mutable_pointer(_hop_diag),
            kupl::qcu::api::pointer(_null_vectors),
            kupl::qcu::api::pointer(_gauge),
            kupl::qcu::api::pointer(_clover_even),
            kupl::qcu::api::pointer(_clover_odd),
            kupl::qcu::api::pointer(_clover_even_inv),
            kupl::qcu::api::pointer(_clover_odd_inv), E, e, Xf, Yf, Zf, Tf,
            Xc, Yc, Zc, Tc, window);
        return 0;
      });
}

} // extern "C"
