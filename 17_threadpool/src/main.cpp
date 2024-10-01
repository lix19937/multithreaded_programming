#include "logger.hpp"
#include "thread_pool.hpp"

void producer_job() {
  LOGI("tid:%ld @cpu_%d", long(pthread_self()), sched_getcpu());
}

int main(int argc, char* argv[]) {
  int number_of_threads_ = 2;
  ThreadPool thread_pool_(number_of_threads_, true);

  LOGI("init");

  thread_pool_.enqueue([] { producer_job(); });

  thread_pool_.enqueue([] { producer_job(); });

  thread_pool_.enqueue([] { producer_job(); });

  thread_pool_.enqueue([] { producer_job(); });

  LOGI("done");

  return 0;
}
