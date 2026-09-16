#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <stdexcept>
#include <string>

namespace quda
{
  namespace mg_profile
  {

    inline bool enabled()
    {
      static const bool value = []() {
        const char *path = std::getenv("QUDA_MG_TRACE_FILE");
        return path != nullptr && path[0] != '\0';
      }();
      return value;
    }

    inline std::ofstream &stream()
    {
      static std::ofstream output = []() {
        const char *path = std::getenv("QUDA_MG_TRACE_FILE");
        std::ofstream value(path, std::ios::out | std::ios::app);
        if (!value) throw std::runtime_error("cannot open QUDA_MG_TRACE_FILE");
        value << std::setprecision(17);
        value << "trace_version\t1\n";
        value.flush();
        return value;
      }();
      return output;
    }

    inline std::mutex &mutex()
    {
      static std::mutex value;
      return value;
    }

    inline double timestamp()
    {
      static const auto start = std::chrono::steady_clock::now();
      return std::chrono::duration<double>(
                 std::chrono::steady_clock::now() - start)
          .count();
    }

    inline std::atomic<long long> &cycle_counter()
    {
      static std::atomic<long long> value {0};
      return value;
    }

    inline long long &current_cycle()
    {
      static thread_local long long value = 0;
      return value;
    }

    inline long long &current_outer_iteration()
    {
      static thread_local long long value = 0;
      return value;
    }

    class OuterScope
    {
     public:
      OuterScope() : previous_(current_outer_iteration())
      {
        if (enabled()) current_outer_iteration() = 0;
      }

      ~OuterScope()
      {
        if (enabled()) current_outer_iteration() = previous_;
      }

      void set(long long iteration)
      {
        if (enabled()) current_outer_iteration() = iteration;
      }

     private:
      long long previous_;
    };

    inline long long begin_cycle(int level, int levels, const char *location)
    {
      if (!enabled()) return 0;
      long long cycle = 0;
      if (level == 0) {
        cycle = cycle_counter().fetch_add(1) + 1;
        current_cycle() = cycle;
      } else {
        cycle = current_cycle();
      }
      std::lock_guard<std::mutex> lock(mutex());
      stream() << "cycle_begin\t" << cycle << '\t' << level << '\t'
               << levels << '\t' << current_outer_iteration() << '\t'
               << (location == nullptr ? "" : location) << '\t'
               << timestamp() << '\n';
      stream().flush();
      return cycle;
    }

    inline void end_cycle(long long cycle, int level, double r2, double b2)
    {
      if (!enabled()) return;
      std::lock_guard<std::mutex> lock(mutex());
      stream() << "cycle_end\t" << cycle << '\t' << level << '\t' << r2
               << '\t' << (r2 > 0.0 ? std::sqrt(r2) : 0.0) << '\t' << b2
               << '\t' << (b2 > 0.0 ? std::sqrt(r2 / b2) : 0.0) << '\t'
               << timestamp() << '\n';
      stream().flush();
    }

    inline void stage(long long cycle, int level, const char *phase,
                      double seconds, int pre_iterations = -1,
                      int post_iterations = -1,
                      int coarse_iterations = -1)
    {
      if (!enabled()) return;
      std::lock_guard<std::mutex> lock(mutex());
      stream() << "stage\t" << cycle << '\t' << level << '\t'
               << (phase == nullptr ? "" : phase) << '\t' << seconds << '\t'
               << pre_iterations << '\t' << post_iterations << '\t'
               << coarse_iterations << '\t' << current_outer_iteration()
               << '\t' << timestamp() << '\n';
      stream().flush();
    }

    inline void residual(long long cycle, int level, const char *phase,
                         double r2, double b2)
    {
      if (!enabled()) return;
      std::lock_guard<std::mutex> lock(mutex());
      stream() << "residual\t" << cycle << '\t' << level << '\t'
               << (phase == nullptr ? "" : phase) << '\t' << r2 << '\t'
               << (r2 > 0.0 ? std::sqrt(r2) : 0.0) << '\t' << b2 << '\t'
               << (b2 > 0.0 ? std::sqrt(r2 / b2) : 0.0) << '\t'
               << current_outer_iteration() << '\t' << timestamp() << '\n';
      stream().flush();
    }

    inline void outer_begin(long long iteration, double r2, double b2)
    {
      if (!enabled()) return;
      std::lock_guard<std::mutex> lock(mutex());
      stream() << "outer_begin\t" << iteration << '\t' << r2 << '\t'
               << (r2 > 0.0 ? std::sqrt(r2) : 0.0) << '\t' << b2 << '\t'
               << (b2 > 0.0 ? std::sqrt(r2 / b2) : 0.0) << '\t'
               << timestamp() << '\n';
      stream().flush();
    }

    inline void outer_iteration(long long iteration, int cycle_iteration,
                                double iterated_r2, double b2,
                                double true_r2 = -1.0)
    {
      if (!enabled()) return;
      std::lock_guard<std::mutex> lock(mutex());
      stream() << "outer_iteration\t" << iteration << '\t'
               << cycle_iteration << '\t' << iterated_r2 << '\t'
               << (iterated_r2 > 0.0 ? std::sqrt(iterated_r2) : 0.0)
               << '\t' << b2 << '\t'
               << (b2 > 0.0 ? std::sqrt(iterated_r2 / b2) : 0.0) << '\t'
               << true_r2 << '\t'
               << (true_r2 > 0.0 ? std::sqrt(true_r2) : 0.0) << '\t'
               << (b2 > 0.0 && true_r2 >= 0.0 ? std::sqrt(true_r2 / b2) : 0.0)
               << '\t' << timestamp() << '\n';
      stream().flush();
    }

  } // namespace mg_profile
} // namespace quda
