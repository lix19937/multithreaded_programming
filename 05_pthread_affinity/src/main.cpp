
#include <pthread.h>
#include <sched.h>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

int main(int argc, const char** argv) {
  auto total_cores = std::thread::hardware_concurrency();
  printf("Starting ... %d cores\n", total_cores);

  constexpr unsigned num_threads = 4;
  // A mutex ensures orderly access to std::cout from multiple threads.
  std::mutex iomutex;
  std::vector<std::thread> threads(num_threads);

  for (unsigned i = 0; i < num_threads; ++i) {
    threads[i] = std::thread([&iomutex, i, &threads] {
      // Create a cpu_set_t object representing a set of CPUs. Clear it and mark
      // only CPU i as set.
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(i, &cpuset);
      int rc = pthread_setaffinity_np(threads[i].native_handle(), sizeof(cpu_set_t), &cpuset);
      if (rc != 0) {
        std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));

      while (1) {
        {
          // Use a lexical scope and lock_guard to safely lock the mutex only
          // for the duration of std::cout usage.
          std::lock_guard<std::mutex> iolock(iomutex);
          std::cout << "Thread #" << i << ": on CPU " << sched_getcpu() << "\n";
        }

        // Simulate important work done by the tread by sleeping for a bit...
        std::this_thread::sleep_for(std::chrono::milliseconds(900));
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
  return 0;
}

// https://stackoverflow.com/questions/24645880/set-cpu-affinity-when-create-a-thread
