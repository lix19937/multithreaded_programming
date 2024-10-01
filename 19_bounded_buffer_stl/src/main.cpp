

/*
https://en.wikipedia.org/wiki/Producer%E2%80%93consumer_problem
The lock_guard() statement instead of a lock() and unlock() pair is C++ RAII.
The lock_guard destructor ensures lock release in case of an exception.
This solution can handle multiple consumer threads and/or multiple producer threads.
*/

#include <stdio.h>
#include <condition_variable>
#include <mutex>
#include <thread>
#include "logger.hpp"

using Portion = int;

#define myassert(assertion)  \
  {                          \
    if (!(assertion)) {      \
      LOGE(#assertion "\n"); \
      abort();               \
    }                        \
  }

// base fifo rules
class Bounded_buffer {
  // clang-format off
  static const int N{32};
  Portion buffer[N];   // 0..N-1
  unsigned head, tail; // 0..N-1
  unsigned count;      // 0..N
  // A condition variable is an object able to block the calling thread until notified to resume.
  std::condition_variable nonempty, nonfull;
  std::mutex mtx;
  // clang-format on

 public:
  void append(Portion x) {
    {
      std::unique_lock<std::mutex> lck(mtx);
      /*
      template <class Predicate>  void wait (unique_lock<mutex>& lck, Predicate pred);
      if pred return true, thread unblock

      when no notified by other thread, current thread block;
      when notified by other thread, and check the pred function, only return true, current thread unblock
      */
      nonfull.wait(lck, [&] { return !(N == count); });
      myassert(0 <= count && count < N);
      buffer[tail++] = x;
      tail %= N;
      ++count;
    }
    LOGI(GREEN "append [%d], %u %u %u", x, head, tail, count);

    nonempty.notify_one();
  }

  Portion remove() {
    Portion x;
    std::unique_lock<std::mutex> lck(mtx);
    {
      nonempty.wait(lck, [&] { return !(0 == count); });
      myassert(0 < count && count <= N);
      x = buffer[head++];
      head %= N;
      --count;
    }
    LOGI(RED "remove [%d], %u %u %u", x, head, tail, count);

    nonfull.notify_one();
    return x;
  }

  Bounded_buffer() { head = 0, tail = 0, count = 0; }
};

void producer(Bounded_buffer* q) {
  q->append(12);
  q->append(25);
  q->append(34);
  q->append(60);
  q->append(80);
}

void consumer(Bounded_buffer* q) {
  q->remove();
  q->remove();
  q->remove();
  q->remove();
}

int main() {
  Bounded_buffer q;
  std::thread t1(producer, &q);
  std::thread t2(consumer, &q);

  t1.join();
  t2.join();

  LOGI("done");
  return 0;
}
