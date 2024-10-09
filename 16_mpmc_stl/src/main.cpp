
#include <chrono>
#include <condition_variable>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include "logger.hpp"

//  conditional variable together with unique_lock always

struct Package {
  int id;
  std::string status;
};

int global_id = 0;
int min_count = 4;

std::queue<Package> q_;
std::mutex mtx_;
std::condition_variable cv_;

std::mutex hist_mtx_;
std::map<pthread_t, int> hist_table_;

void* produce(void* args) {
  while (1) {
    { /*
       *
       * use block {}, let unique_lock life time within block
       * when block over，auto release unique_lock
       *
       */
      std::unique_lock<std::mutex> lock(mtx_);
      Package p;
      p.id = global_id;
      p.status = "newly produced";
      q_.push(p);
      ++global_id;
      LOGI(PURPLE "[Producer]:pushed package_%d, queue size is %d", p.id, q_.size());

      if (q_.size() >= min_count) {
        LOGI(RED "[Producer]:Activate all blocked threads");
        cv_.notify_all(); // eq pthread_cond_broadcast
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  return NULL;
}

void* consume(void* args) {
  auto tid = pthread_self();

  // for debug to show which consumer to pop
  {
    std::unique_lock<std::mutex> lock(hist_mtx_);
    int len = hist_table_.size();
    if (hist_table_.find(tid) == hist_table_.end()) {
      hist_table_.insert(std::make_pair(tid, len));
    }
  }

  // for debug to show which way to lead to pop
  bool is_cv_notice_state;

  while (1) {
    {
      is_cv_notice_state = false;

      LOGI("tid:%ld ->         consumer_%d started a new iter", tid, hist_table_[tid]);

      std::unique_lock<std::mutex> lock(mtx_);
      LOGI("tid:%ld -->>>      consumer_%d acquired lock, other consumers blocked", tid, hist_table_[tid]);

      // when queue is empty, cv.wait
      while (q_.empty()) {
        LOGI(RED "tid:%ld ---------- consumer_%d thread is blocked, waiting...", tid, hist_table_[tid]);
        cv_.wait(lock); // eq pthread_cond_wait
        is_cv_notice_state = true;
      }

      if (is_cv_notice_state) {
        LOGI("tid:%ld ----->>>>  consumer_%d unblocked by cv notice, allow to pop", tid, hist_table_[tid]);
      } else {
        LOGI("tid:%ld ----->>>>  consumer_%d queue is not empty, allow to pop", tid, hist_table_[tid]);
      }

      Package tmp = q_.front();
      q_.pop();

      LOGI(
          DGREEN "tid:%ld ---------- consumer_%d poped package_%d, current length %d, will release lock",
          tid,
          hist_table_[tid],
          tmp.id,
          q_.size());
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return NULL;
}

/*
 * use c++ std condition_variable, mutex, unique_lock, thread
 * unique_lock is mutex lock shared_ptr warpper
 *
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
