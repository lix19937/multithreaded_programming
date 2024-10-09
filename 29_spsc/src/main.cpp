

#include <atomic>
#include <chrono>
#include <thread>
#include "logger.hpp"

using Message = int;

// Without semaphores or monitors

enum { N = 4 };
Message buffer[N];
std::atomic<unsigned> count{0};

int sendMsg = 0;

int produceMessage() {
  std::this_thread::sleep_for(std::chrono::milliseconds(400));
  sendMsg++;
  LOGI("sendMsg = %d", sendMsg);
  return sendMsg;
}

void consumeMessage(int recvMsg) {
  LOGI("\t\trecvMsg = %d\n", recvMsg);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

void producer() {
  unsigned tail{0};
  for (;;) {
    Message message = produceMessage();
    while (N == count)
      ; // busy waiting
    buffer[tail++] = message;
    tail %= N;
    count.fetch_add(1, std::memory_order_relaxed);
  }
}

void consumer() {
  unsigned head{0};
  for (;;) {
    while (0 == count)
      ; // busy waiting
    Message message = buffer[head++];
    head %= N;
    count.fetch_sub(1, std::memory_order_relaxed);
    consumeMessage(message);
  }
}

int main() {
  std::thread t1(producer);
  std::thread t2(consumer);
  t1.join();
  t2.join();
  return 0;
}
