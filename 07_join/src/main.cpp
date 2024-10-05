
#include <unistd.h>
#include <string>
#include <thread>

void task1(std::string msg) {
  printf("task1 says:%s\n", msg.c_str());
}

int test() {
  std::thread t1(task1, "hello1");
  t1.join();
  return 0;
}

// will error, raise `terminate called without an active exception`
int test2() {
  std::thread t1(task1, "hello2");
  return 0;
}

int test3() {
  {
    std::thread t1(task1, "hello3");
    t1.detach();
  }

  usleep(1000000);
  return 0;
}

int main() {
  test();
  test3();
  test2();

  return 0;
}

/*
terminate called without an active exception
Aborted

When a thread object goes out of scope and it is in joinable state (i.e. - assuming the thread object is named t then
t.joinable() returns true), the program is terminated. The Standard Committee had two other options for the destructor
of a joinable thread. It could quietly join -- but join might never return if the thread is stuck. Or it could detach
the thread (a detached thread is not joinable). However, detached threads are very tricky, since they might survive till
the end of the program and mess up the release of resources. So if you don't want to terminate your program, make sure
you join (or detach) every thread.

*/
