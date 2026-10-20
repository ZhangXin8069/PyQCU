#include "kupl/lattice_set.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#if defined(__linux__)
#include <sched.h>
#endif

namespace kupl::qcu {
namespace {

std::mutex &registry_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<const void *, std::shared_ptr<LatticeSetBase>> &registry() {
  static std::unordered_map<const void *, std::shared_ptr<LatticeSetBase>>
      value;
  return value;
}

#if defined(__linux__)
const std::vector<int> &initial_affinity_cpus() {
  static const std::vector<int> cpus = [] {
    std::vector<int> result;
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
      for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &set)) {
          result.push_back(cpu);
        }
      }
    }
    return result;
  }();
  return cpus;
}
#endif

template <typename Real>
std::shared_ptr<LatticeSetBase> create_lattice_set(const int32_t *params,
                                                   const void *argv) {
  return std::make_shared<LatticeSet<Real>>(
      params, static_cast<const Real *>(argv));
}

} // namespace

int configured_threads() {
#if defined(__linux__)
  const auto &affinity = initial_affinity_cpus();
#endif
  int requested = omp_get_max_threads();
  const char *text = std::getenv("KUPL_OMP_THREADS");
  const bool explicit_request = text != nullptr;
  if (explicit_request) {
    requested = std::max(1, std::atoi(text));
  }
#if defined(__linux__)
  if (!explicit_request && !affinity.empty()) {
    requested = std::min(requested, static_cast<int>(affinity.size()));
  }
#endif
  omp_set_dynamic(0);
  omp_set_num_threads(std::max(1, requested));
  return std::max(1, requested);
}

void bind_logical_numa() {
  const char *enabled = std::getenv("KUPL_NUMA_BIND");
  if (enabled == nullptr || std::atoi(enabled) == 0) {
    return;
  }
#if defined(__linux__)
  const char *omp_bind = std::getenv("OMP_PROC_BIND");
  if (omp_bind != nullptr && omp_bind[0] != '\0') {
    throw std::runtime_error(
        "KUPL_NUMA_BIND=1 owns CPU placement; unset OMP_PROC_BIND "
        "(use mpirun --bind-to none)");
  }
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    return;
  }
  MPI_Comm local_comm = MPI_COMM_NULL;
  if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                          MPI_INFO_NULL, &local_comm) != MPI_SUCCESS) {
    return;
  }
  int local_rank = 0;
  int local_size = 1;
  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_size(local_comm, &local_size);
  MPI_Comm_free(&local_comm);

  const std::vector<int> &cpus = initial_affinity_cpus();
  if (cpus.empty()) {
    return;
  }
  int per_rank = std::max(1, static_cast<int>(cpus.size()) / local_size);
  per_rank = std::min(per_rank, configured_threads());
  const int begin =
      std::min(static_cast<int>(cpus.size()), local_rank * per_rank);
  const int end = std::min(static_cast<int>(cpus.size()), begin + per_rank);
  cpu_set_t bound;
  CPU_ZERO(&bound);
  for (int index = begin; index < end; ++index) {
    CPU_SET(cpus[static_cast<std::size_t>(index)], &bound);
  }
  if (begin < end && sched_setaffinity(0, sizeof(bound), &bound) != 0) {
    throw std::runtime_error("KUPL_NUMA_BIND failed to set CPU affinity");
  }
#endif
}

void initialize_mpi() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    int provided = 0;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
      throw std::runtime_error("MPI_THREAD_FUNNELED is unavailable");
    }
  }
  bind_logical_numa();
  configured_threads();
}

std::shared_ptr<LatticeSetBase> make_lattice_set(const int32_t *params,
                                                 const void *argv) {
  if (params == nullptr || argv == nullptr) {
    throw std::runtime_error("null params or argv");
  }
  switch (params[kDataType]) {
  case kLatC64:
    return create_lattice_set<float>(params, argv);
  case kLatC128:
    return create_lattice_set<double>(params, argv);
  default:
    throw std::runtime_error("unsupported data type");
  }
}

std::shared_ptr<LatticeSetBase> register_lattice_set(const void *set_ptrs,
                                                     const int32_t *params,
                                                     const void *argv) {
  auto value = make_lattice_set(params, argv);
  std::lock_guard<std::mutex> lock(registry_mutex());
  registry()[set_ptrs] = value;
  return value;
}

std::shared_ptr<LatticeSetBase> find_lattice_set(const void *set_ptrs) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  const auto iterator = registry().find(set_ptrs);
  if (iterator == registry().end()) {
    throw std::runtime_error("applyInitQcu has not initialized this set_ptrs");
  }
  return iterator->second;
}

void erase_lattice_set(const void *set_ptrs) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  registry().erase(set_ptrs);
}

[[noreturn]] void abort_with_message(const char *submodule,
                                     const char *message) {
  std::fprintf(stderr, "PYQCU::KUPL::%s:\n %s\n", submodule, message);
  std::fflush(stderr);
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (initialized) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
  }
  std::abort();
}

} // namespace kupl::qcu
