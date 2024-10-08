
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

// use two cv
std::condition_variable cv_max_limit;
std::condition_variable cv_min_limit;

// sleep time of producer & consumer
int t_producer = 100;
int t_consumer = 500;

/*
std::condition_variable is a synchronization primitive used with a std::mutex to block one or more threads until another
thread both modifies a shared variable (the condition) and notifies the std::condition_variable.

The thread that intends to modify the shared variable must:

Acquire a std::mutex (typically via std::lock_guard).
Modify the shared variable while the lock is owned.
Call notify_one or notify_all on the std::condition_variable (can be done after releasing the lock).
*/

void* produce(int id) {
  while (1) {
    {
      LOGI("producer_%d started a new iter", id);

      std::unique_lock<std::mutex> lock(mtx_);
      LOGI("producer_%d acquired lock, other producers blocked", id);
      /*
       * when len(queue) == max_limit, block producer,
       * then consumer begin to pop, until len(queue) == min_limit, block consumer,
       * then producer begin to push
       *
       * if len(queue) < max_limit, unblock, push to queue
       * if len(queue) >= max_limit, block, stop push to queue
       */
      while (!(q_.size() < max_limit)) {
        cv_max_limit.wait(lock);
      }

      Package p;
      p.id = global_id;
      p.status = "newly produced";
      q_.push(p);
      ++global_id;

      LOGI(
          GREEN "producer_%d pushed package_%d, now queue size is %ld|%ld, will release lock",
          id,
          p.id,
          q_.size(),
          max_limit);
    }

    LOGI(
        "producer_%d notify_all, pred():%s, if false, will not real wakeup",
        id,
        q_.size() > min_limit ? "true" : "false");

    // make sure mtx_ is release, so put notify_all outside
    cv_min_limit.notify_all();

    // control speed of produce
    std::this_thread::sleep_for(std::chrono::milliseconds(t_producer));
  }
  return NULL;
}

void* consume(int id) {
  // for debug to show which way to lead to pop
  bool is_cv_notice_state;

  while (1) {
    {
      is_cv_notice_state = false;

      LOGI("consumer_%d started a new iter", id);

      std::unique_lock<std::mutex> lock(mtx_);
      LOGI("consumer_%d acquired lock, other consumers blocked", id);

      /*
       * acquired lock,
       * if consumer get cv_min_limit`s notify, then check `pred` function, {if len(queue) > min_limit} unblock
       * if consumer not get cv_min_limit`s notify, block
       *
       */
      while (!(q_.size() > min_limit)) {
        cv_min_limit.wait(lock);
        is_cv_notice_state = true;
      }

      if (is_cv_notice_state) {
        LOGI("consumer_%d unblocked by cv notice, allow to pop", id);
      } else {
        LOGI("consumer_%d len(queue) >= min_limit, allow to pop", id);
      }

      Package p = q_.front();
      q_.pop();
      LOGI(RED "consumer_%d poped package_%d, now queue size is %ld|%ld", id, p.id, q_.size(), min_limit);
    }

    LOGI(
        "consumer_%d notify_all, pred():%s, if false, will not real wakeup\n",
        id,
        q_.size() < max_limit ? "true" : "false");

    // Here if use notify_one will happen diff effect
    cv_max_limit.notify_all();

    std::this_thread::sleep_for(std::chrono::milliseconds(t_consumer));
  }
  return NULL;
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
  int num_producer = 2;
  int num_consumer = 2;

  std::vector<std::thread> producers;
  std::vector<std::thread> consumers;

  for (int i = 0; i < num_producer; ++i) {
    producers.emplace_back(produce, i);
  }
  for (int i = 0; i < num_consumer; ++i) {
    consumers.emplace_back(consume, i);
  }

  for (auto& producer : producers) {
    producer.join();
  }
  for (auto& consumer : consumers) {
    consumer.join();
  }

  return 0;
}
