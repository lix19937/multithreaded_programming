
#include <pthread.h>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include "logger.hpp"

struct Data {
  int id;
  int count;
  int sleep_time;
  std::string msg;
};

auto msg_1 = std::make_shared<std::string>(DGREEN "sub_thread_100ms" CLEAR);
auto msg_2 = std::make_shared<std::string>(PURPLE "sub_thread_200ms" CLEAR);
auto d1 = std::make_shared<Data>(Data{0, 10, 100, *msg_1});
auto d2 = std::make_shared<Data>(Data{1, 10, 200, *msg_2});

void* run(void* args) {
  Data* d = static_cast<Data*>(args);

  for (int i = 0; i < d->count; i++) {
    LOGI("Runing thread %ld: %s, count: %d", d->id, d->msg.c_str(), i);
    std::this_thread::sleep_for(std::chrono::milliseconds(d->sleep_time));
  }

  d->msg = YELLOW "sub_thread finished " CLEAR;

  LOGW("Runing thread %ld done", d->id);
  return nullptr;
}

int main() {
  pthread_t t1;
  pthread_t t2;

  // use different para
  pthread_create(&t1, NULL, run, d1.get());
  pthread_create(&t2, NULL, run, d2.get());

  /*
  Sometimes we don't want to use `pthread_join` in the main thread because it blocks the main thread
  in this case, use `pthread_detach`

  To allow other threads to continue execution, the main thread should terminate by calling `pthread_exit` rather
  than exit(3).

  The pthread_detach() function marks the thread identified by
  thread as detached.  When a detached thread terminates, its
  resources are automatically released back to the system without
  the need for another thread to join with the terminated thread.
  */
  pthread_detach(t1);
  pthread_detach(t2);

  LOGW("main thread will pthread_exit");

  pthread_exit(NULL);

  LOGE("main thread not reach here !!!");

  return 0;
}
