
#include <chrono>
#include <string>
#include <thread>
#include "logger.hpp"

void run1(const std::string& name) {
  while (1) {
    LOGI("Runing thread %ld:%s", pthread_self(), name.c_str());
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
}

void run2(const std::string& name) {
  while (1) {
    LOGI("Runing thread %ld:%s", pthread_self(), name.c_str());
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
}

void run3(const std::string& name) {
  LOGI("Runing thread %ld:%s", pthread_self(), name.c_str());
  std::this_thread::sleep_for(std::chrono::milliseconds(2000));
}

int main() {
  std::thread t1(run1, PURPLE "thread_500ms" CLEAR);
  std::thread t2(run2, RED "thread_1000ms" CLEAR);
  std::thread t3(run3, BLUE "thread_2000ms" CLEAR);

  // wait & sync & release resource
  t1.join();
  t2.join();
  t3.join();

  return 0;
}
