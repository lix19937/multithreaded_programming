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

void* run(void* args) {
  Data* d = static_cast<Data*>(args);

  for (int i = 0; i < d->count; i++) {
    LOGI("Runing thread %ld: %s, count: %d", d->id, d->msg.c_str(), i);
    std::this_thread::sleep_for(std::chrono::milliseconds(d->sleep_time));
  }

  d->msg = YELLOW "sub_thread finished " CLEAR;
  // or use pthread_exit, current use return
  return NULL;
}

void* run_with_pexit(void* args) {
  Data* d = static_cast<Data*>(args);

  for (int i = 0; i < d->count; i++) {
    LOGI("Runing thread %ld: %s, count: %d", d->id, d->msg.c_str(), i);
    std::this_thread::sleep_for(std::chrono::milliseconds(d->sleep_time));
  }

  d->msg = YELLOW "sub_thread finished @run_with_pexit " CLEAR;
  // terminate calling thread
  pthread_exit((void*)d->msg.c_str());
}

int main() {
  const char* msg_1 = DGREEN "sub_thread_100ms" CLEAR;
  const char* msg_2 = RED "sub_thread_200ms" CLEAR;

  pthread_t t1;
  pthread_t t2;

  auto d1 = std::make_shared<Data>(Data{0, 10, 100, msg_1});
  auto d2 = std::make_shared<Data>(Data{1, 10, 200, msg_2});

  if (pthread_create(&t1, NULL, run, d1.get()) != 0) {
    perror("pthread_create() error");
    exit(1);
  }

  if (pthread_create(&t2, NULL, run_with_pexit, d2.get()) != 0) {
    perror("pthread_create() error");
    exit(2);
  }

  /*
    int pthread_join(pthread_t thread, void **status);

    pthread_join, wait for a thread to end
    If successful, pthread_join() returns 0.
    If unsuccessful, pthread_join() returns -1 and sets errno

    When pthread_join() returns successfully, the target thread has been detached.
    Multiple threads cannot use pthread_join() to wait for the same target thread to end.
    If a thread issues pthread_join() for a target thread after another thread has successfully issued pthread_join()
   for the same target thread, the second pthread_join() will be unsuccessful.
    If the thread calling pthread_join() is canceled, the target thread is not detached.
   *
   */
  if (pthread_join(t1, NULL) != 0) {
    perror("pthread_create() error");
    exit(3);
  }

  void* ret;
  if (pthread_join(t2, &ret) != 0) {
    perror("pthread_create() error");
    exit(4);
  }

  LOGI("thread %d: %s", d1->id, d1->msg.c_str());
  LOGI("thread %d: %s", d2->id, d2->msg.c_str());

  LOGI("thread exited with '%s'", ret);

  return 0;
}
