
#include <chrono>
#include <iostream>
#include <vector>

#include "logger.hpp"
#include "threadpool.h"

int main() {
  ThreadPool pool(4);
  std::vector<std::future<int>> results;

  for (int i = 0; i < 8; ++i) {
    results.emplace_back(pool.enqueue([i] {
      LOGI("hello %d", i);
      std::this_thread::sleep_for(std::chrono::seconds(1));
      LOGI("world %d", i);
      return i * i;
    }));
  }

  for (auto&& result : results)
    LOGI("%d ", result.get());

  LOGI("");

  return 0;
}
