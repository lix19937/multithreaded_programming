
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include "logger.hpp"

std::mutex m;
std::condition_variable cv;
std::string data;
bool ready = false;
bool processed = false;

void worker_thread() {
  // wait until main() sends data
  std::unique_lock<std::mutex> lk(m);

  // if ready=true, unblock and get the lock
  cv.wait(lk, [] { return ready; });

  // after the wait, we own the lock
  LOGI("Worker-thread is processing data");
  data += "{ from tid1 }";

  // send data back to main-thread
  processed = true;

  LOGI("Worker-thread done\n");

  // manual unlocking is done before notifying, to avoid waking up
  // the waiting thread only to block again (see notify_one for details)
  lk.unlock();

  // calling notify_one unblocks one of the waiting threads
  cv.notify_one();
}

void worker_thread2() {
  // wait until main() sends data
  std::unique_lock<std::mutex> lk(m);

  // if ready=true, unblock and get the lock
  cv.wait(lk, [] { return ready; });

  // after the wait, we own the lock
  LOGI("Worker-thread2 is processing data");
  data += "{ from tid2 }";

  // send data back to main-thread
  processed = true;

  LOGI("Worker-thread2 done\n");

  // manual unlocking is done before notifying, to avoid waking up
  // the waiting thread only to block again (see notify_one for details)
  lk.unlock();

  // calling notify_one unblocks one of the waiting threads
  cv.notify_one();
}

int main() {
  std::thread worker(worker_thread);
  std::thread worker2(worker_thread2);

  data = "Example data";
  // send data to the worker thread
  {
    std::lock_guard<std::mutex> lk(m);
    ready = true;
  }

  LOGI("main-thread notice worker to ready for processing ...");
  cv.notify_all();

  /* !!!
    worker and worker2 will competitive each other
    and when notify_one of worker/worker2
    the main-thread will competitive with worker2/worker
  */

  // wait for the worker
  {
    std::unique_lock<std::mutex> lk(m);

    // cv.wait(lk, [] { return processed; });
    /// or use follow eq impl
    while (!processed) {
      cv.wait(lk);
    }
  }

  LOGI("Back in main(), data:%s", data.c_str());

  worker.join();
  worker2.join();

  return 0;
}

// https://cplusplus.com/reference/condition_variable/condition_variable/
