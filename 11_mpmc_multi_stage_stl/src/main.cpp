#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include "logger.hpp"

struct Package {
  long unsigned int id;
  std::string status;
};

long unsigned int global_id = 0;
long unsigned int min_limit = 1;
long unsigned int max_limit = 4;

// product queue
std::queue<Package> q_;
std::mutex mtx_;
std::condition_variable cv_max_limit;
std::condition_variable cv_min_limit;
volatile bool over_limit = false;

/*
std::condition_variable is a synchronization primitive used with a std::mutex to block one or more threads until another
thread both modifies a shared variable (the condition) and notifies the std::condition_variable.

The thread that intends to modify the shared variable must:

Acquire a std::mutex (typically via std::lock_guard).
Modify the shared variable while the lock is owned.
Call notify_one or notify_all on the std::condition_variable (can be done after releasing the lock).
*/

void* produce(void* args) {
  while (1) {
    {
      std::unique_lock<std::mutex> lock(mtx_);
      /*
       * when len(queue) == max_limit, block producer,
       * then consumer begin to pop, until len(queue) == min_limit, block consumer,
       * then producer begin to push
       *
       * if len(queue) < max_limit, unblock, push to queue
       * if len(queue) >= max_limit, block, stop push to queue
       */
      cv_max_limit.wait(lock, []() { return q_.size() < max_limit; });

      Package p;
      p.id = global_id;
      p.status = "newly produced";
      q_.push(p);
      LOGI(
          GREEN "%-10s:%-7s package_%d, now queue size is %ld|%ld, tid:%ld",
          "Producer",
          "pushed",
          p.id,
          q_.size(),
          max_limit,
          long(pthread_self()));
      global_id++;
    }

    LOGI(
        "%-10s notify_all, pred():%-6s, tid:%ld",
        "Producer",
        q_.size() > min_limit ? "true" : "false",
        long(pthread_self()));

    // make sure mtx_ is release, so put notify_all outside
    cv_min_limit.notify_all();

    // control speed of produce
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
}

void* consume(void* args) {
  while (1) {
    {
      std::unique_lock<std::mutex> lock(mtx_);

      /*
       * get lock,
       * if consumer get cv_min_limit`s notify, then check `pred` function, {if len(queue) > min_limit} unblock
       * if consumer not get cv_min_limit`s notify, block
       *
       */
      cv_min_limit.wait(lock, []() { return q_.size() > min_limit; });

      Package p = q_.front();
      q_.pop();
      LOGI(
          RED "%-10s:%-7s package_%d, now queue size is %ld|%ld, tid:%ld",
          "Consumer",
          "poped",
          p.id,
          q_.size(),
          min_limit,
          long(pthread_self()));
    }

    LOGI(
        "%-10s notify_all, pred():%-6s, tid:%ld\n",
        "Consumer",
        q_.size() < max_limit ? "true" : "false",
        long(pthread_self()));

    /// Here if use notify_one will happen diff effect
    cv_max_limit.notify_all();

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
}

/*
 * set a case：
 * 1. within one lock cycle, the num of produce generate product is max_limit
 * 2. the speed of producer/consumer can be adjust by user
 * 3. when len(queue) > max_limit，block producer, notify consumer to pop product
 * 4. when len(queue) < min_limit，block consumer
 *
 * when producer speed > consumer speed，3 stages
 * 1. block consumer, producer unblock, push queue until len(queue) >= min_limit, then consumer pred true
 * 2. consumer, producer run at the same time，until len(queue) == max_limit
 * 3. block producer, two speed keep the same
 *
 * when producer speed <= consumer speed，3 stages
 * 1. block consumer, producer unblock, push queue until len(queue) >= min_limit, then consumer pred true
 * 2. block consumer, two speed keep the same
 */

int main() {
  // we can adjust the count of producer/consumer to impl different speed
  int producer_count = 2;
  int consumer_count = 1;

  std::vector<std::thread> producers;
  std::vector<std::thread> consumers;

  for (int i = 0; i < producer_count; i++) {
    producers.emplace_back(produce, nullptr);
  }
  for (int i = 0; i < consumer_count; i++) {
    consumers.emplace_back(consume, nullptr);
  }

  for (auto& producer : producers) {
    producer.join();
  }
  for (auto& consumer : consumers) {
    consumer.join();
  }

  return 0;
}
