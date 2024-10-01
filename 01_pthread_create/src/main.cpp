#include <pthread.h>
#include <chrono>
#include <string>
#include <thread>
#include "logger.hpp"

void* run1(void* args) {
  while (1) {
    LOGI("Runing thread %ld:%s", pthread_self(), args);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
}

void* run2(void* args) {
  while (1) {
    LOGI("Runing thread %ld:%s", pthread_self(), args);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
}

int main() {
  const char* msg_1 = GREEN "sub_thread_500ms  " CLEAR;
  const char* msg_2 = RED "main_thread_1000ms" CLEAR;

  /*
    int pthread_create(
                    pthread_t *restrict thread,
                    const pthread_attr_t *restrict attr,
                    void *(*start_routine)(void *),
                    void *restrict arg);

   */
  pthread_t t1;
  pthread_create(&t1, NULL, run1, (void*)msg_1);

  run2((void*)msg_2);

  // wait t1 pthread
  pthread_join(t1, NULL);

  return 0;
}

// ref https://man7.org/linux/man-pages/man3/pthread_create.3.html
