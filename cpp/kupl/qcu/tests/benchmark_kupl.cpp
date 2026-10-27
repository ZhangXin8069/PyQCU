#include "kupl/define.h"
#include "kupl/qcu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>

#include <mpi.h>
#include <omp.h>

#if defined(__linux__)
#include <sched.h>
#endif

namespace {

constexpr int kDim = 4;
constexpr int kSpin = 4;
constexpr int kColor = 3;
constexpr int kVector = 12;
constexpr int kGauge = 9;

template <typename Real> using Complex = std::complex<Real>;

std::array<int, kDim> parseVector(int argc, char **argv, const char *flag,
                                  const std::array<int, kDim> &fallback) {
  std::array<int, kDim> result = fallback;
  for (int index = 1; index + 4 < argc; ++index) {
    if (std::strcmp(argv[index], flag) == 0) {
      for (int dim = 0; dim < kDim; ++dim) {
        result[dim] = std::atoi(argv[index + 1 + dim]);
      }
      return result;
    }
  }
  return result;
}

int parseInteger(int argc, char **argv, const char *flag, int fallback) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (std::strcmp(argv[index], flag) == 0) {
      return std::atoi(argv[index + 1]);
    }
  }
  return fallback;
}

std::array<int, kDim> deriveGrid(int size) {
  std::array<int, kDim> grid{1, 1, 1, 1};
  int remaining = size;
  grid[0] = remaining;
  remaining = 1;
  (void)remaining;
  return grid;
}

template <typename Real>
void runBenchmark(const std::array<int, kDim> &local,
                  const std::array<int, kDim> &grid, int iterations,
                  const char *precision) {
  using C = Complex<Real>;
  if (local[0] <= 0 || local[1] <= 0 || local[2] <= 0 || local[3] <= 0 ||
      local[3] % 2 != 0) {
    throw std::runtime_error("lattice extents must be positive and T even");
  }
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int grid_product = 1;
  for (int value : grid) {
    grid_product *= value;
  }
  if (grid_product != size) {
    throw std::runtime_error("grid product must equal MPI rank count");
  }

  std::array<int32_t, kupl::qcu::kParamsSize> params{};
  std::array<Real, kupl::qcu::kArgvSize> argv{};
  std::array<int64_t, kupl::qcu::kSetPtrsSize> set_ptrs{};
  for (int dim = 0; dim < kDim; ++dim) {
    params[kupl::qcu::kLatX + dim] = local[dim];
    params[kupl::qcu::kGridX + dim] = grid[dim];
  }
  params[kupl::qcu::kLatXyzt] =
      local[0] * local[1] * local[2] * local[3];
  params[kupl::qcu::kParity] = 0;
  params[kupl::qcu::kNodeRank] = rank;
  params[kupl::qcu::kNodeSize] = size;
  params[kupl::qcu::kDataType] =
      sizeof(Real) == sizeof(float) ? kupl::qcu::kLatC64
                                    : kupl::qcu::kLatC128;
  params[kupl::qcu::kSetPlan] = kupl::qcu::kSetPlanWilson;
  params[kupl::qcu::kVerbose] = 0;
  argv[kupl::qcu::kMass] = Real(0.1);
  argv[kupl::qcu::kAtol] = Real(1e-8);
  argv[kupl::qcu::kSigma] = Real(0.1);
  argv[kupl::qcu::kCsw] = Real(1);

  // Bind and place the runtime before any benchmark tensor is first-touched.
  applyInitQcu(reinterpret_cast<long long>(set_ptrs.data()),
               reinterpret_cast<long long>(params.data()),
               reinterpret_cast<long long>(argv.data()));

  const int64_t half_sites =
      static_cast<int64_t>(local[0]) * local[1] * local[2] * (local[3] / 2);
  const int64_t gauge_components = 2 * kGauge * kDim;
  std::vector<C> gauge(
      static_cast<std::size_t>(gauge_components * half_sites));
  std::vector<C> input(
      static_cast<std::size_t>(2 * kVector * half_sites));
  std::vector<C> output(input.size());
  const std::size_t clover_size =
      static_cast<std::size_t>(kVector * kVector * half_sites);
  std::vector<C> clover_even(clover_size);
  std::vector<C> clover_odd(clover_size);
  std::vector<C> clover_even_inv(clover_size);
  std::vector<C> clover_odd_inv(clover_size);

  for (int64_t site = 0; site < half_sites; ++site) {
    for (int parity = 0; parity < 2; ++parity) {
      for (int direction = 0; direction < kDim; ++direction) {
        const Real angle =
            Real(0.013) *
            static_cast<Real>(site + 17 * direction + 31 * parity + 1);
        const C phase(std::cos(angle), std::sin(angle));
        const Real cosine = std::cos(Real(0.3) * angle);
        const Real sine = std::sin(Real(0.3) * angle);
        const C matrix[9] = {
            cosine, -sine * phase, C(0),
            sine * std::conj(phase), cosine, C(0),
            C(0), C(0), phase,
        };
        for (int row = 0; row < 3; ++row) {
          for (int column = 0; column < 3; ++column) {
            const int64_t index =
                ((((static_cast<int64_t>(parity) * 3 + row) * 3 + column) *
                          kDim +
                      direction) *
                     half_sites +
                 site);
            gauge[static_cast<std::size_t>(index)] =
                matrix[row * 3 + column];
          }
        }
      }
      for (int component = 0; component < kVector; ++component) {
        const Real phase =
            Real(0.019) * static_cast<Real>(site + 7 * component + parity);
        const int64_t index =
            (static_cast<int64_t>(parity) * kVector + component) * half_sites +
            site;
        input[static_cast<std::size_t>(index)] =
            C(std::cos(phase), std::sin(Real(0.7) * phase));
      }
    }
  }

  const int threads = omp_get_max_threads();
  if (std::getenv("KUPL_BENCH_SHOW_AFFINITY") != nullptr) {
#if defined(__linux__)
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0) {
      int first = -1;
      int last = -1;
      for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &affinity)) {
          first = first < 0 ? cpu : first;
          last = cpu;
        }
      }
      std::fprintf(stderr,
                   "rank_binding rank=%d size=%d cpus=%d first=%d last=%d "
                   "threads=%d\n",
                   rank, size, CPU_COUNT(&affinity), first, last, threads);
    }
#endif
  }
  int64_t local_volume = 1;
  for (int value : local) {
    local_volume *= value;
  }
  const int64_t global_volume = local_volume * size;
  auto measure = [&](const char *operator_name,
                     const std::function<void()> &operation,
                     double flops_per_site) {
    for (int warmup = 0; warmup < 2; ++warmup) {
      operation();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = omp_get_wtime();
    for (int iteration = 0; iteration < iterations; ++iteration) {
      operation();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double seconds = omp_get_wtime() - start;
    const double estimated_flops =
        flops_per_site * static_cast<double>(global_volume) * iterations;
    if (rank == 0) {
      std::printf(
          "precision=%s operator=%s ranks=%d grid=%d,%d,%d,%d "
          "local=%d,%d,%d,%d threads_per_rank=%d iterations=%d "
          "seconds=%.9f global_sites=%lld estimated_gflops=%.6f\n",
          precision, operator_name, size, grid[0], grid[1], grid[2], grid[3],
          local[0], local[1], local[2], local[3], threads, iterations,
          seconds, static_cast<long long>(global_volume),
          estimated_flops / seconds / 1.0e9);
    }
  };
  auto measure_setup = [&](const char *operator_name,
                           const std::function<void()> &operation) {
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = omp_get_wtime();
    operation();
    MPI_Barrier(MPI_COMM_WORLD);
    const double seconds = omp_get_wtime() - start;
    if (rank == 0) {
      std::printf(
          "precision=%s operator=%s ranks=%d grid=%d,%d,%d,%d "
          "local=%d,%d,%d,%d threads_per_rank=%d seconds=%.9f\n",
          precision, operator_name, size, grid[0], grid[1], grid[2], grid[3],
          local[0], local[1], local[2], local[3], threads, seconds);
    }
  };
  params[kupl::qcu::kSetPlan] = kupl::qcu::kSetPlanClover;
  params[kupl::qcu::kParity] = 0;
  measure_setup("clover_setup_even", [&] {
    applyCloversQcu(
        reinterpret_cast<long long>(clover_even.data()),
        reinterpret_cast<long long>(clover_even_inv.data()),
        reinterpret_cast<long long>(gauge.data()),
        reinterpret_cast<long long>(set_ptrs.data()),
        reinterpret_cast<long long>(params.data()));
  });
  params[kupl::qcu::kParity] = 1;
  measure_setup("clover_setup_odd", [&] {
    applyCloversQcu(
        reinterpret_cast<long long>(clover_odd.data()),
        reinterpret_cast<long long>(clover_odd_inv.data()),
        reinterpret_cast<long long>(gauge.data()),
        reinterpret_cast<long long>(set_ptrs.data()),
        reinterpret_cast<long long>(params.data()));
  });

  // The dslash entries write one parity half.  Report the equivalent work per
  // full lattice site: 8 directed SU(3) hops, plus an onsite product when the
  // entry includes Clover, with a complex multiply-add counted as 8 FLOPs.
  const double wilson_half_site_flops = 8.0 * kSpin * kColor * kColor * 8.0;
  const double onsite_half_site_flops = kVector * kVector * 8.0;
  const double wilson_flops_per_site = wilson_half_site_flops * 0.5;
  const double clover_flops_per_site =
      (wilson_half_site_flops + onsite_half_site_flops) * 0.5;
  const double schur_flops_per_site =
      2.0 * (wilson_half_site_flops + onsite_half_site_flops) * 0.5;
  params[kupl::qcu::kSetPlan] = kupl::qcu::kSetPlanWilson;
  measure(
      "wilson",
      [&] {
        applyWilsonDslashQcu(
            reinterpret_cast<long long>(output.data()),
            reinterpret_cast<long long>(input.data()),
            reinterpret_cast<long long>(gauge.data()),
            reinterpret_cast<long long>(set_ptrs.data()),
            reinterpret_cast<long long>(params.data()));
      },
      wilson_flops_per_site);
  measure(
      "clover",
      [&] {
        applyCloverDslashQcu(
            reinterpret_cast<long long>(output.data()),
            reinterpret_cast<long long>(input.data()),
            reinterpret_cast<long long>(gauge.data()),
            reinterpret_cast<long long>(set_ptrs.data()),
            reinterpret_cast<long long>(params.data()));
      },
      clover_flops_per_site);
  params[kupl::qcu::kSetPlan] = kupl::qcu::kSetPlanSolver;
  measure(
      "clover_schur",
      [&] {
        applyCloverBistabCgDslashQcu(
            reinterpret_cast<long long>(output.data()),
            reinterpret_cast<long long>(input.data()),
            reinterpret_cast<long long>(gauge.data()),
            reinterpret_cast<long long>(clover_even.data()),
            reinterpret_cast<long long>(clover_odd.data()),
            reinterpret_cast<long long>(clover_even_inv.data()),
            reinterpret_cast<long long>(clover_odd_inv.data()),
            reinterpret_cast<long long>(set_ptrs.data()),
            reinterpret_cast<long long>(params.data()));
      },
      schur_flops_per_site);
  applyEndQcu(reinterpret_cast<long long>(set_ptrs.data()),
              reinterpret_cast<long long>(params.data()));
}

} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int status = 0;
  try {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const std::array<int, kDim> local =
        parseVector(argc, argv, "--lattice", {16, 16, 16, 16});
    const std::array<int, kDim> grid =
        parseVector(argc, argv, "--grid", deriveGrid(
            [&] {
              int size = 1;
              MPI_Comm_size(MPI_COMM_WORLD, &size);
              return size;
            }()));
    const int iterations =
        std::max(1, parseInteger(argc, argv, "--iterations", 20));
    runBenchmark<float>(local, grid, iterations, "c64");
    runBenchmark<double>(local, grid, iterations, "c128");
  } catch (const std::exception &error) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      std::fprintf(stderr, "benchmark error: %s\n", error.what());
    }
    status = 1;
  }
  MPI_Finalize();
  return status;
}
