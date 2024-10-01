#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include "logger.hpp"

struct Package {
  int id;
  std::string status;
};

int global_id = 0;
int min_count = 10;

std::queue<Package> q_;
std::mutex mtx_;
std::condition_variable cv_;

void* produce(void* args) {
  while (true) {
    { /*
       *
       * 通过在函数中设置block {}, 让unique_lock的生命周期只在这个block内
       * 当block结束的时候，自动释放 unique_lock，实现上锁解锁
       *
       */
      std::unique_lock<std::mutex> lock(mtx_);
      Package p;
      p.id = global_id;
      p.status = "newly produced";
      q_.push(p);
      LOGI(PURPLE "[Producer]:Produce pushed package %d, queue size is %d", p.id, q_.size());
      global_id++;
      if (q_.size() >= min_count) {
        LOGI(RED "[Producer]:Activate all blocked threads");
        cv_.notify_all();
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void* consume(void* args) {
  while (true) {
    {
      std::unique_lock<std::mutex> lock(mtx_);
      while (q_.empty()) {
        LOGI(RED "[Consumer]:The thread is blocked, waiting ...");
        cv_.wait(lock);
      }

      Package tmp = q_.front();
      q_.pop();
      LOGI(DGREEN "[Consumer]:Nonblocking pop package %d, queue size is %d", tmp.id, q_.size());
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

/*
 * 使用c++ std库中的condition_variable, mutex, unique_lock, thread来实现CPM
 * unique_lock是mutex lock的智能指针版，内部用类似与unique_ptr的方式封装的
 *
 * 解锁和上锁直接是通过unique_lock的生命周期来管理，生命周期结束自动释放，所以一般可以配合函数block来上锁/解锁
 * std::condition_variable的 wait 和 notify_all
 * 分别对应c语言版本的 pthread_cond_wait 和 pthread_cond_broadcast
 */
int main() {
  int producer_count = 1;
  int consumer_count = 2;

  std::vector<std::thread> producers;
  std::vector<std::thread> consumers;

  for (int i = 0; i < producer_count; i++) {
    producers.emplace_back(produce, nullptr);
  }
  for (int i = 0; i < consumer_count; i++) {
    consumers.emplace_back(consume, nullptr);
  }

  // wait and sync
  for (auto& producer : producers) {
    producer.join();
  }
  for (auto& consumer : consumers) {
    consumer.join();
  }

  return 0;
}
