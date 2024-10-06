
#include <stdio.h>
#include <mutex>
#include <thread>

// https://en.cppreference.com/w/cpp/thread/call_once

std::once_flag flag;

void do_once() {
  std::call_once(flag, []() { printf("Called once  tid:%ld\n", pthread_self()); });
}

int main() {
  std::thread t1(do_once);
  std::thread t2(do_once);
  std::thread t3(do_once);
  std::thread t4(do_once);

  t1.join();
  t2.join();
  t3.join();
  t4.join();

  return 0;
}
