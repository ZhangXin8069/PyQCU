#include "kupl/lattice_set.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <mpi.h>

#if defined(__linux__)
#include <linux/mempolicy.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/utsname.h>
#include <unistd.h>
#endif

namespace kupl::qcu {
namespace {

struct CpuNode {
  int id = -1;
  std::vector<int> cpus;
};

struct RuntimeState {
  int local_rank = -1;
  int local_size = -1;
  int topology_nodes = 0;
  int cpu_node = -1;
  int memory_node = -1;
  int threads = 1;
  int bound_threads = 0;
  bool numa_bind_requested = false;
  bool topology_required = false;
  bool memory_policy_applied = false;
  bool fused_bicg_dots = true;
  bool numa_warning_emitted = false;
  std::vector<int> cpus;
  std::string topology_source = "unavailable";
  std::string memory_policy = "default";
};

RuntimeState &runtime_state() {
  static RuntimeState state;
  return state;
}

std::mutex &registry_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<const void *, std::shared_ptr<LatticeSetBase>> &registry() {
  static std::unordered_map<const void *, std::shared_ptr<LatticeSetBase>>
      value;
  return value;
}

std::string trim(std::string text) {
  const auto first = std::find_if_not(text.begin(), text.end(), [](char value) {
    return std::isspace(static_cast<unsigned char>(value)) != 0;
  });
  const auto last = std::find_if_not(text.rbegin(), text.rend(), [](char value) {
                      return std::isspace(static_cast<unsigned char>(value)) !=
                             0;
                    }).base();
  if (first >= last) {
    return {};
  }
  return std::string(first, last);
}

bool env_enabled(const char *name) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
         std::strcmp(value, "no") != 0 && std::strcmp(value, "off") != 0;
}

int parse_integer(const std::string &text, const char *name) {
  const std::string clean = trim(text);
  if (clean.empty()) {
    throw std::runtime_error(std::string(name) + " must not be empty");
  }
  std::size_t consumed = 0;
  long value = 0;
  try {
    value = std::stol(clean, &consumed, 10);
  } catch (const std::exception &) {
    throw std::runtime_error(std::string(name) + " must be an integer");
  }
  if (consumed != clean.size() || value < 0) {
    throw std::runtime_error(std::string(name) + " must be non-negative");
  }
  return static_cast<int>(value);
}

int parse_positive_integer(const std::string &text, const char *name) {
  const int value = parse_integer(text, name);
  if (value <= 0) {
    throw std::runtime_error(std::string(name) + " must be positive");
  }
  return value;
}

std::string read_text_file(const std::string &path) {
  std::ifstream stream(path);
  if (!stream) {
    return {};
  }
  std::string value;
  std::getline(stream, value);
  return trim(value);
}

bool host_is_kunpeng() {
#if defined(__linux__)
  static const bool detected = [] {
    struct utsname system_name {};
    if (uname(&system_name) != 0) {
      return false;
    }
    std::string machine(system_name.machine);
    std::transform(machine.begin(), machine.end(), machine.begin(),
                   [](unsigned char value) {
                     return static_cast<char>(std::tolower(value));
                   });
    if (machine != "aarch64" && machine != "arm64") {
      return false;
    }
    std::ifstream stream("/proc/cpuinfo");
    if (!stream) {
      return false;
    }
    std::string line;
    while (std::getline(stream, line)) {
      std::transform(line.begin(), line.end(), line.begin(),
                     [](unsigned char value) {
                       return static_cast<char>(std::tolower(value));
                     });
      if (line.find("kunpeng") != std::string::npos ||
          line.find("hisilicon") != std::string::npos ||
          (line.find("cpu implementer") != std::string::npos &&
           line.find("0x48") != std::string::npos) ||
          (line.find("cpu part") != std::string::npos &&
           (line.find("0xd01") != std::string::npos ||
            line.find("0xd02") != std::string::npos))) {
        return true;
      }
    }
    return false;
  }();
  return detected;
#else
  return false;
#endif
}

int requested_profile_code() {
  const char *profile = std::getenv("PYQCU_KUPL_PROFILE");
  if (profile == nullptr || profile[0] == '\0' ||
      std::strcmp(profile, "generic") == 0) {
    return 0;
  }
  if (std::strcmp(profile, "kunpeng") == 0) {
    return 1;
  }
  if (std::strcmp(profile, "kunpeng-2numa") == 0) {
    return 2;
  }
  return 3;
}

std::vector<int> parse_cpu_list(const std::string &text) {
  std::vector<int> result;
  std::size_t begin = 0;
  while (begin < text.size()) {
    const std::size_t comma = text.find(',', begin);
    const std::string token = trim(text.substr(
        begin, comma == std::string::npos ? std::string::npos : comma - begin));
    if (!token.empty()) {
      const std::size_t dash = token.find('-');
      try {
        if (dash == std::string::npos) {
          result.push_back(std::stoi(token));
        } else {
          const int first = std::stoi(trim(token.substr(0, dash)));
          const int last = std::stoi(trim(token.substr(dash + 1)));
          if (first < 0 || last < first) {
            throw std::runtime_error("invalid CPU range");
          }
          for (int cpu = first; cpu <= last; ++cpu) {
            result.push_back(cpu);
          }
        }
      } catch (const std::exception &) {
        throw std::runtime_error("invalid NUMA cpulist: " + token);
      }
    }
    if (comma == std::string::npos) {
      break;
    }
    begin = comma + 1;
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

std::string format_cpu_set(const std::vector<int> &cpus) {
  if (cpus.empty()) {
    return "none";
  }
  std::string value;
  std::size_t begin = 0;
  while (begin < cpus.size()) {
    std::size_t end = begin + 1;
    while (end < cpus.size() && cpus[end] == cpus[end - 1] + 1) {
      ++end;
    }
    if (!value.empty()) {
      value.push_back(',');
    }
    value += std::to_string(cpus[begin]);
    if (end - begin > 1) {
      value.push_back('-');
      value += std::to_string(cpus[end - 1]);
    }
    begin = end;
  }
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

std::vector<CpuNode> read_numa_nodes() {
  const char *override_root = std::getenv("KUPL_SYSFS_NODE_ROOT");
  const std::string root = override_root == nullptr
                               ? "/sys/devices/system/node"
                               : std::string(override_root);
  std::vector<int> node_ids;
  const std::string online = read_text_file(root + "/online");
  if (!online.empty()) {
    node_ids = parse_cpu_list(online);
  } else {
    for (int node = 0; node < 1024; ++node) {
      const std::string cpulist =
          read_text_file(root + "/node" + std::to_string(node) + "/cpulist");
      if (!cpulist.empty()) {
        node_ids.push_back(node);
      }
    }
  }
  std::vector<CpuNode> result;
  for (int node : node_ids) {
    const std::string cpulist = read_text_file(
        root + "/node" + std::to_string(node) + "/cpulist");
    if (!cpulist.empty()) {
      result.push_back({node, parse_cpu_list(cpulist)});
    }
  }
  return result;
}
#endif

std::pair<int, int> local_rank_and_size() {
  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    return {-1, -1};
  }
  MPI_Comm local_comm = MPI_COMM_NULL;
  if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                          MPI_INFO_NULL, &local_comm) != MPI_SUCCESS) {
    return {-1, -1};
  }
  int local_rank = 0;
  int local_size = 1;
  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_size(local_comm, &local_size);
  MPI_Comm_free(&local_comm);
  return {local_rank, local_size};
}

std::vector<int> gather_local_integers(int value) {
  MPI_Comm local_comm = MPI_COMM_NULL;
  if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                          MPI_INFO_NULL, &local_comm) != MPI_SUCCESS) {
    return {};
  }
  int local_size = 1;
  MPI_Comm_size(local_comm, &local_size);
  std::vector<int> values(static_cast<std::size_t>(local_size));
  const int status =
      MPI_Allgather(&value, 1, MPI_INT, values.data(), 1, MPI_INT, local_comm);
  MPI_Comm_free(&local_comm);
  if (status != MPI_SUCCESS) {
    throw std::runtime_error("local MPI NUMA assignment gather failed");
  }
  return values;
}

int parse_memory_node(const std::string &spec, int cpu_node) {
  const std::string value = trim(spec);
  if (value == "local") {
    if (cpu_node < 0) {
      throw std::runtime_error(
          "KUPL_MEMORY_NODE=local requires a detected CPU NUMA node");
    }
    return cpu_node;
  }
  if (value.rfind("local+", 0) == 0) {
    if (cpu_node < 0) {
      throw std::runtime_error(
          "KUPL_MEMORY_NODE=local+N requires a detected CPU NUMA node");
    }
    return cpu_node + parse_positive_integer(value.substr(6), "memory offset");
  }
  return parse_integer(value, "KUPL_MEMORY_NODE");
}

bool apply_memory_policy(int node, const std::string &policy,
                         std::string &error) {
#if defined(__linux__) && defined(SYS_set_mempolicy)
  if (node < 0) {
    error = "memory NUMA node is unavailable";
    return false;
  }
  constexpr unsigned long bits = sizeof(unsigned long) * 8;
  const std::size_t words =
      (static_cast<std::size_t>(node) + bits) / bits;
  std::vector<unsigned long> mask(words, 0);
  mask[static_cast<std::size_t>(node) / bits] |=
      1UL << (static_cast<unsigned long>(node) % bits);
  int mode = MPOL_PREFERRED;
  if (policy == "bind") {
    mode = MPOL_BIND;
  } else if (policy != "preferred") {
    error = "KUPL_MEMORY_POLICY must be preferred or bind";
    return false;
  }
  if (syscall(SYS_set_mempolicy, mode, mask.data(), node + 1) != 0) {
    error = std::string("set_mempolicy failed: ") + std::strerror(errno);
    return false;
  }
  return true;
#else
  (void)node;
  (void)policy;
  error = "NUMA memory policy is unavailable on this platform";
  return false;
#endif
}

bool clear_memory_policy(std::string &error) {
#if defined(__linux__) && defined(SYS_set_mempolicy)
  if (syscall(SYS_set_mempolicy, MPOL_DEFAULT, nullptr, 0) != 0) {
    error = std::string("reset set_mempolicy failed: ") + std::strerror(errno);
    return false;
  }
  return true;
#else
  error = "NUMA memory policy is unavailable on this platform";
  return false;
#endif
}

template <typename Real>
std::shared_ptr<LatticeSetBase> create_lattice_set(const int32_t *params,
                                                   const void *argv) {
  return std::make_shared<LatticeSet<Real>>(
      params, static_cast<const Real *>(argv));
}

int uniform_world_value(int value, const char *name) {
  int minimum = 0;
  int maximum = 0;
  MPI_Allreduce(&value, &minimum, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&value, &maximum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (minimum != maximum) {
    throw std::runtime_error(std::string(name) +
                             " must be identical on every MPI rank");
  }
  return minimum;
}

} // namespace

int configured_threads() {
#if defined(__linux__)
  const auto &affinity = initial_affinity_cpus();
#endif
  int requested = omp_get_max_threads();
  const char *text = std::getenv("KUPL_OMP_THREADS");
  const bool explicit_request = text != nullptr && text[0] != '\0';
  if (explicit_request) {
    requested = parse_positive_integer(text, "KUPL_OMP_THREADS");
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

bool fused_bicg_dots_enabled() {
  return runtime_state().fused_bicg_dots;
}

void bind_logical_numa() {
  RuntimeState &state = runtime_state();
  const int profile_code =
      uniform_world_value(requested_profile_code(), "PYQCU_KUPL_PROFILE");
  const bool topology_report_requested =
      uniform_world_value(env_enabled("KUPL_TOPOLOGY_REPORT") ? 1 : 0,
                          "KUPL_TOPOLOGY_REPORT") != 0;
  const bool topology_required =
      uniform_world_value(env_enabled("KUPL_REQUIRE_TOPOLOGY") ? 1 : 0,
                          "KUPL_REQUIRE_TOPOLOGY") != 0;
  const bool allow_oversubscribe =
      uniform_world_value(env_enabled("KUPL_ALLOW_OVERSUBSCRIBE") ? 1 : 0,
                          "KUPL_ALLOW_OVERSUBSCRIBE") != 0;
  if (profile_code > 2) {
    throw std::runtime_error(
        "PYQCU_KUPL_PROFILE must be generic, kunpeng, or kunpeng-2numa");
  }
  if (profile_code > 0 && !host_is_kunpeng()) {
    throw std::runtime_error(
        "PYQCU_KUPL_PROFILE requests a Kunpeng profile on a non-Kunpeng host");
  }
  state.topology_required = topology_required || profile_code > 0;
  state.threads = configured_threads();
  bool bind_requested = env_enabled("KUPL_NUMA_BIND");
  bind_requested = uniform_world_value(bind_requested ? 1 : 0,
                                        "KUPL_NUMA_BIND") != 0;
  state.numa_bind_requested = bind_requested;
  const char *memory_spec = std::getenv("KUPL_MEMORY_NODE");
  bool memory_requested = memory_spec != nullptr && memory_spec[0] != '\0';
  memory_requested = uniform_world_value(memory_requested ? 1 : 0,
                                          "KUPL_MEMORY_NODE") != 0;
  const char *fused_text = std::getenv("KUPL_FUSED_BICG_DOTS");
  const int fused_requested =
      fused_text == nullptr || fused_text[0] == '\0' ||
              env_enabled("KUPL_FUSED_BICG_DOTS")
          ? 1
          : 0;
  state.fused_bicg_dots =
      uniform_world_value(fused_requested, "KUPL_FUSED_BICG_DOTS") != 0;
  if (!bind_requested && !memory_requested && !topology_report_requested &&
      !state.topology_required) {
#if defined(__linux__)
    const auto &affinity = initial_affinity_cpus();
    if (state.threads > static_cast<int>(affinity.size()) &&
        !allow_oversubscribe) {
      throw std::runtime_error(
          "KUPL_OMP_THREADS exceeds the process affinity CPU set; reduce "
          "threads or set KUPL_ALLOW_OVERSUBSCRIBE=1");
    }
#endif
    return;
  }

  const auto [local_rank, local_size] = local_rank_and_size();
  state.local_rank = local_rank;
  state.local_size = local_size;

#if defined(__linux__)
  const std::vector<CpuNode> all_nodes = read_numa_nodes();
  state.topology_nodes = static_cast<int>(all_nodes.size());
  if (!all_nodes.empty()) {
    state.topology_source = "sysfs";
  }
  if (state.topology_required && profile_code == 2 && all_nodes.size() != 2) {
    throw std::runtime_error(
        "PYQCU_KUPL_PROFILE=kunpeng-2numa requires exactly two NUMA nodes");
  }

  const std::vector<int> &affinity = initial_affinity_cpus();
  std::vector<CpuNode> local_nodes;
  for (const CpuNode &node : all_nodes) {
    std::vector<int> intersection;
    std::set_intersection(affinity.begin(), affinity.end(), node.cpus.begin(),
                          node.cpus.end(), std::back_inserter(intersection));
    if (!intersection.empty()) {
      local_nodes.push_back({node.id, std::move(intersection)});
    }
  }
  if (!bind_requested && state.threads > static_cast<int>(affinity.size()) &&
      !allow_oversubscribe) {
    throw std::runtime_error(
        "KUPL_OMP_THREADS exceeds the process affinity CPU set; reduce "
        "threads or set KUPL_ALLOW_OVERSUBSCRIBE=1");
  }

  const char *cpu_node_text = std::getenv("KUPL_CPU_NODE");
  const bool explicit_cpu_node =
      cpu_node_text != nullptr && cpu_node_text[0] != '\0';
  int explicit_node = -1;
  if (explicit_cpu_node) {
    explicit_node = parse_integer(cpu_node_text, "KUPL_CPU_NODE");
  }

  const std::vector<int> explicit_requests =
      gather_local_integers(explicit_cpu_node ? explicit_node : -1);
  if (bind_requested && explicit_requests.empty()) {
    throw std::runtime_error("unable to gather local NUMA rank assignment");
  }
  if (bind_requested) {
    const bool has_explicit = std::any_of(
        explicit_requests.begin(), explicit_requests.end(),
        [](int value) { return value >= 0; });
    const bool has_auto = std::any_of(
        explicit_requests.begin(), explicit_requests.end(),
        [](int value) { return value < 0; });
    if (has_explicit && has_auto) {
      throw std::runtime_error(
          "KUPL_CPU_NODE must be set or unset on every local MPI rank");
    }
  }

  if (bind_requested && local_rank >= 0 && !local_nodes.empty()) {
    const CpuNode *selected = nullptr;
    int rank_on_node = 0;
    int ranks_on_node = 1;
    if (explicit_cpu_node) {
      const auto node = std::find_if(
          local_nodes.begin(), local_nodes.end(),
          [explicit_node](const CpuNode &candidate) {
            return candidate.id == explicit_node;
          });
      if (node == local_nodes.end()) {
        throw std::runtime_error(
            "KUPL_CPU_NODE is not present in the process affinity set");
      }
      selected = &*node;
      ranks_on_node = 0;
      for (int index = 0; index < local_size; ++index) {
        if (explicit_requests[static_cast<std::size_t>(index)] == explicit_node) {
          if (index < local_rank) {
            ++rank_on_node;
          }
          ++ranks_on_node;
        }
      }
      ranks_on_node = std::max(1, ranks_on_node);
    } else {
      const int node_slot = local_rank % static_cast<int>(local_nodes.size());
      ranks_on_node =
          (local_size - node_slot + static_cast<int>(local_nodes.size()) - 1) /
          static_cast<int>(local_nodes.size());
      rank_on_node = local_rank / static_cast<int>(local_nodes.size());
      selected = &local_nodes[static_cast<std::size_t>(node_slot)];
    }
    if (selected != nullptr && !selected->cpus.empty()) {
      const std::size_t begin = selected->cpus.size() *
                                static_cast<std::size_t>(rank_on_node) /
                                static_cast<std::size_t>(ranks_on_node);
      const std::size_t end = selected->cpus.size() *
                              static_cast<std::size_t>(rank_on_node + 1) /
                              static_cast<std::size_t>(ranks_on_node);
      state.cpu_node = selected->id;
      state.cpus.assign(selected->cpus.begin() + static_cast<std::ptrdiff_t>(begin),
                        selected->cpus.begin() + static_cast<std::ptrdiff_t>(end));
    }
  }

  if (!bind_requested && local_nodes.size() == 1) {
    state.cpu_node = local_nodes.front().id;
    state.cpus = local_nodes.front().cpus;
  }

  if (!bind_requested && state.topology_required) {
    if (local_nodes.size() != 1) {
      throw std::runtime_error(
          "KUPL profile requires each rank to be bound to exactly one NUMA "
          "node before applyInitQcu");
    }
  }

  if (bind_requested) {
    for (const char *name : {"OMP_PROC_BIND", "OMP_PLACES",
                             "GOMP_CPU_AFFINITY", "KMP_AFFINITY"}) {
      const char *value = std::getenv(name);
      if (value != nullptr && value[0] != '\0') {
        throw std::runtime_error(
            std::string("KUPL_NUMA_BIND=1 owns CPU placement; unset ") +
            name + " (use mpirun --bind-to none)");
      }
    }
    if (state.cpus.empty()) {
      state.topology_source = "affinity-fallback";
      if (state.topology_required) {
        throw std::runtime_error(
            "KUPL_REQUIRE_TOPOLOGY requested but no NUMA cpulist is available");
      }
      state.cpus = affinity;
      if (state.cpus.empty()) {
        return;
      }
      const int per_rank = std::max(
          1, static_cast<int>(state.cpus.size()) /
                 std::max(1, state.local_size == -1 ? 1 : state.local_size));
      const int begin = std::min(
          static_cast<int>(state.cpus.size()),
          std::max(0, state.local_rank == -1 ? 0 : state.local_rank) * per_rank);
      const int end = std::min(static_cast<int>(state.cpus.size()), begin + per_rank);
      state.cpus = std::vector<int>(
          state.cpus.begin() + begin, state.cpus.begin() + end);
    }
    if (state.threads > static_cast<int>(state.cpus.size()) &&
        !allow_oversubscribe) {
      throw std::runtime_error(
          "KUPL_OMP_THREADS exceeds the NUMA-local CPU set; reduce threads, "
          "increase ranks, or set KUPL_ALLOW_OVERSUBSCRIBE=1");
    }
    if (state.threads < static_cast<int>(state.cpus.size())) {
      state.cpus.resize(static_cast<std::size_t>(state.threads));
    }
    cpu_set_t bound;
    CPU_ZERO(&bound);
    for (int cpu : state.cpus) {
      if (cpu >= 0 && cpu < CPU_SETSIZE) {
        CPU_SET(cpu, &bound);
      }
    }
    if (sched_setaffinity(0, sizeof(bound), &bound) != 0) {
      throw std::runtime_error(
          std::string("KUPL_NUMA_BIND failed to set CPU affinity: ") +
          std::strerror(errno));
    }
    std::atomic<int> thread_bind_failure{0};
    std::atomic<int> bound_thread_count{0};
#pragma omp parallel num_threads(state.threads) shared(bound, thread_bind_failure, bound_thread_count)
    {
      if (sched_setaffinity(0, sizeof(bound), &bound) == 0) {
        bound_thread_count.fetch_add(1, std::memory_order_relaxed);
      } else {
        int expected = 0;
        thread_bind_failure.compare_exchange_strong(
            expected, errno, std::memory_order_relaxed);
      }
    }
    state.bound_threads = bound_thread_count.load(std::memory_order_relaxed);
    if (thread_bind_failure.load(std::memory_order_relaxed) != 0) {
      throw std::runtime_error(
          std::string("KUPL_NUMA_BIND failed to bind an OpenMP thread: ") +
          std::strerror(thread_bind_failure.load(std::memory_order_relaxed)));
    }
    if (state.bound_threads != state.threads) {
      throw std::runtime_error(
          "KUPL_NUMA_BIND expected " + std::to_string(state.threads) +
          " OpenMP threads but bound " + std::to_string(state.bound_threads));
    }
  }
#else
  if (state.topology_required || bind_requested || memory_requested) {
    throw std::runtime_error("NUMA runtime controls require Linux");
  }
#endif

  if (bind_requested || memory_requested) {
    state.memory_node = bind_requested ? state.cpu_node : -1;
    if (memory_requested) {
      state.memory_node = parse_memory_node(memory_spec, state.cpu_node);
    }
    const char *policy_text = std::getenv("KUPL_MEMORY_POLICY");
    state.memory_policy = policy_text == nullptr || policy_text[0] == '\0'
                              ? "preferred"
                              : trim(policy_text);
    if (state.memory_policy != "default") {
      std::string policy_error;
      state.memory_policy_applied =
          apply_memory_policy(state.memory_node, state.memory_policy,
                              policy_error);
      if (!state.memory_policy_applied) {
        if (env_enabled("KUPL_REQUIRE_MEMORY_BIND")) {
          throw std::runtime_error("KUPL NUMA memory policy: " + policy_error);
        }
        if (!state.numa_warning_emitted) {
          std::fprintf(stderr, "PYQCU::KUPL::NUMA:\n %s\n",
                       policy_error.c_str());
          state.numa_warning_emitted = true;
        }
      } else {
        std::vector<std::string> thread_errors(
            static_cast<std::size_t>(state.threads));
#pragma omp parallel for schedule(static) num_threads(state.threads)
        for (int thread = 0; thread < state.threads; ++thread) {
          (void)apply_memory_policy(
              state.memory_node, state.memory_policy,
              thread_errors[static_cast<std::size_t>(thread)]);
        }
        const auto failure = std::find_if(
            thread_errors.begin(), thread_errors.end(),
            [](const std::string &thread_error) {
              return !thread_error.empty();
            });
        if (failure != thread_errors.end()) {
          const std::string failure_error = *failure;
          if (env_enabled("KUPL_REQUIRE_MEMORY_BIND")) {
            throw std::runtime_error(
                "KUPL NUMA memory policy failed for an OpenMP thread: " +
                failure_error);
          }
          if (!state.numa_warning_emitted) {
            std::fprintf(stderr,
                         "PYQCU::KUPL::NUMA:\n OpenMP thread policy failed: %s\n",
                         failure_error.c_str());
            state.numa_warning_emitted = true;
          }
          std::vector<std::string> reset_errors(
              static_cast<std::size_t>(state.threads));
#pragma omp parallel for schedule(static) num_threads(state.threads)
          for (int thread = 0; thread < state.threads; ++thread) {
            (void)clear_memory_policy(
                reset_errors[static_cast<std::size_t>(thread)]);
          }
          state.memory_policy_applied = false;
        }
      }
    }
  }

  if (topology_report_requested) {
    std::fprintf(stderr, "PYQCU::KUPL::TOPOLOGY:\n %s\n",
                 runtime_topology_report().c_str());
  }
}

std::string runtime_topology_report() {
  const RuntimeState &state = runtime_state();
  char buffer[512];
  std::snprintf(
      buffer, sizeof(buffer),
      "local_rank=%d local_size=%d numa_nodes=%d cpu_node=%d cpuset=%s "
      "memory_node=%d memory_policy=%s memory_bound=%s threads=%d "
      "bound_threads=%d source=%s",
      state.local_rank, state.local_size, state.topology_nodes, state.cpu_node,
      format_cpu_set(state.cpus).c_str(), state.memory_node,
      state.memory_policy.c_str(), state.memory_policy_applied ? "true" : "false",
      state.threads, state.bound_threads, state.topology_source.c_str());
  return buffer;
}

void advise_large_pages(const void *address, std::size_t bytes) {
  if (address == nullptr || bytes == 0 || !env_enabled("KUPL_HUGEPAGE")) {
    return;
  }
#if defined(__linux__) && defined(MADV_HUGEPAGE)
  const long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0) {
    return;
  }
  const std::uintptr_t page = static_cast<std::uintptr_t>(page_size);
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(address) &
                               ~(page - 1);
  const std::uintptr_t end =
      (reinterpret_cast<std::uintptr_t>(address) + bytes + page - 1) &
      ~(page - 1);
  (void)madvise(reinterpret_cast<void *>(begin), end - begin, MADV_HUGEPAGE);
#else
  (void)address;
  (void)bytes;
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
  configured_threads();
  bind_logical_numa();
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
