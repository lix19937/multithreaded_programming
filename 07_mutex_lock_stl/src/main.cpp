
#include <mutex>
#include <thread>
#include "logger.hpp"

int global_var = 0;

std::mutex mtx;

void task() {
  for (int i = 0; i < 100; ++i) {
    // mutex's lock will lock follow two-ops
    mtx.lock();

    //------------------------
    global_var++;
    global_var--;
    //------------------------

    mtx.unlock();
  }
}

int main() {
  std::thread t1(task);
  std::thread t2(task);

  t1.join();
  t2.join();

  LOGI("%d", global_var);

  return 0;
}
