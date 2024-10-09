
#include <pthread.h>
#include <chrono>

#include <map>
#include <string>
#include <thread>
#include "logger.hpp"

/*
 * fixed pattern
 * cv(pthread_cond_t) + mutex(pthread_mutex_t)
 *  - cv for block
 *  - mutex for sync
 *
 * for queue, producer push data, consumer pop data
 * if queue is empty, consumer stop pop data
 * if queue is full, producer stop push data
 *
 * pthread_cond_t used to block thread, and store the thread id of the wakeup thread
 *
 *
 *  pthread_cond_init
 *  pthread_cond_destroy
 *  pthread_cond_wait              (block thread)
 *  pthread_cond_timedwait         (block thread with limited time)
 *  pthread_cond_signal            (wake up a   thread on the cv)
 *  pthread_cond_broadcast         (wake up all thread on the cv)
 *
 */

// use a link as a queue
struct Package {
  int id;
  std::string status;
  std::shared_ptr<Package> next;
};

pthread_cond_t cond;
pthread_mutex_t mtx;

pthread_mutex_t hist_mtx;
std::map<pthread_t, int> hist_table_;

const int min_count = 4;
int global_id = 0;
int count = 0;
auto head = std::make_shared<Package>();

void* produce(void* args) {
  while (1) {
    pthread_mutex_lock(&mtx);
    ///-------------------------------------
    {
      auto pack = std::make_shared<Package>();
      pack->id = global_id;
      pack->status = "newly produced";
      pack->next = head;
      head = pack; ///
      ++global_id;
      ++count;
      LOGI(PURPLE "[Producer]:push package_%d, current length %d, tid:%ld", pack->id, count, pthread_self());

      /// or put follow outside ?
      if (count >= min_count) {
        LOGI(RED "[Producer]:Activate all blocked threads");
        pthread_cond_broadcast(&cond);
      }
    }
    ///---------------------------------------

    pthread_mutex_unlock(&mtx);

    std::this_thread::sleep_for(std::chrono::milliseconds(500)); // ms
  }
}

void* consume(void* args) {
  /*
   * per t, pop data
   *
   * when link is empty, block consumer; unblock action is raised by producer thread
   */
  auto tid = pthread_self();

  // for debug to show which consumer to pop
  pthread_mutex_lock(&hist_mtx);
  int len = hist_table_.size();
  if (hist_table_.find(tid) == hist_table_.end()) {
    hist_table_.insert(std::make_pair(tid, len));
  }
  pthread_mutex_unlock(&hist_mtx);

  // for debug to show which way to lead to pop
  bool is_cv_notice_state;

  while (1) {
    is_cv_notice_state = false;

    LOGI("tid:%ld ->         consumer_%d started a new iter", tid, hist_table_[tid]);

    pthread_mutex_lock(&mtx);

    /*
     * if a thread was blocked, pthread_cond_wait will do 3 things:
     *   release mutex lock
     *   commit cur thread in cv-queue and wait pthread_cond_signal/pthread_cond_broadcast call
     *   lock mutex
     */
    LOGI("tid:%ld -->>>      consumer_%d acquired lock, other consumers blocked", tid, hist_table_[tid]);

    // head->next == nullptr, means link has only one node
    while (head->next == nullptr) {
      LOGI(RED "tid:%ld ---------- consumer_%d thread is blocked, waiting...", tid, hist_table_[tid]);
      pthread_cond_wait(&cond, &mtx);
      is_cv_notice_state = true;
    }

    if (is_cv_notice_state) {
      LOGI("tid:%ld ----->>>>  consumer_%d unblocked by cv notice, allow to pop", tid, hist_table_[tid]);
    } else {
      LOGI("tid:%ld ----->>>>  consumer_%d queue is not empty, allow to pop", tid, hist_table_[tid]);
    }

    auto pack = head;
    count--;
    head = head->next;
    LOGI(
        DGREEN "tid:%ld ---------- consumer_%d pop package_%d, current length %d, will release lock",
        tid,
        hist_table_[tid],
        pack->id,
        count);

    pthread_mutex_unlock(&mtx);

    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // ms
  }
}

/*
 * assume a case:
 *
 * 1. use the same link/queue
 * 2. when link is empty, consume stop pop and block
 * 3. when len(link) > t, producer block and noticed consumer
 * 4. the speed of consumer > producer
 * 5. consumer producer not work at the same time !!!
 *
 * producer release the consumer block by pthread_cond_broadcast
 * consumer release the producer block by pthread_cond_broadcast, like when len(link) < t, producer will be wakeup
 */

int main() {
  pthread_cond_init(&cond, nullptr);
  pthread_mutex_init(&mtx, nullptr);

  int producer_count = 1;
  int consumer_count = 2;

  auto producers = (pthread_t*)malloc(producer_count * sizeof(pthread_t));
  auto consumers = (pthread_t*)malloc(consumer_count * sizeof(pthread_t));

  for (int i = 0; i < producer_count; i++) {
    pthread_create(&producers[i], nullptr, produce, nullptr);
  }
  for (int i = 0; i < consumer_count; i++) {
    pthread_create(&consumers[i], nullptr, consume, nullptr);
  }

  // wait and sync
  for (int i = 0; i < producer_count; i++) {
    pthread_join(producers[i], nullptr);
  }
  for (int i = 0; i < consumer_count; i++) {
    pthread_join(consumers[i], nullptr);
  }

  // release
  pthread_mutex_destroy(&mtx);
  pthread_cond_destroy(&cond);

  free(producers);
  free(consumers);
  return 0;
}
