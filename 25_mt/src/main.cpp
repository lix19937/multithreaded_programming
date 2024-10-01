
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <thread>
#include <vector>

#include "Task.h"
#include "customTask.h"
#include "delayKernel.h"

inline void time_diff(const timespec& start, const timespec& end, timespec* result) {
  if ((end.tv_nsec - start.tv_nsec) < 0) {
    result->tv_sec = end.tv_sec - start.tv_sec - 1;
    result->tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
  } else {
    result->tv_sec = end.tv_sec - start.tv_sec;
    result->tv_nsec = end.tv_nsec - start.tv_nsec;
  }
}

inline timespec time_diff(const timespec& start, const timespec& end) {
  timespec result;
  time_diff(start, end, &result);
  return result;
}

//! the thread entry of each task
void* process(void* param) {
  if (!param) {
    printf("NULL thread parameter, exiting thread\n");
    return NULL;
  }
  auto task = (Task*)param;
  printf("%-16s task thread started, tid:%ld\n", task->getTaskType(), pthread_self());
  //
  while (1) {
    if (!task->Run())
      printf("task failed to Run\n");
  }
  printf("exiting thread\n");
}

int main() {
  char* CUDA_DEVICE_MAX_CONNECTIONS = getenv("CUDA_DEVICE_MAX_CONNECTIONS");
  printf("CUDA_DEVICE_MAX_CONNECTIONS=%s\n", CUDA_DEVICE_MAX_CONNECTIONS);

  cudaFree(0);

  std::vector<std::thread> threads;
  std::vector<Task*> tasks;

  std::vector<std::string> customTasks{"D2DCopy", "D2HCopy", "H2DCopy"};
  if (!customTasks.empty()) {
    for (size_t i = 0; i < customTasks.size(); i++) {
      printf("add custom task: %s\n", customTasks[i].c_str());
      CustomTask* custom_task = CustomTask::CreateObject(customTasks[i].c_str(), 2);
      //
      threads.push_back(std::thread(process, (void*)(custom_task)));
      tasks.push_back(custom_task);
    }
  }

  std::vector<std::string> delayKernelTasks{"5", "15", "20", "10"};
  if (!delayKernelTasks.empty()) {
    for (size_t i = 0; i < delayKernelTasks.size(); i++) {
      printf("add a kernel that delay %d millisecond\n", std::stoi(delayKernelTasks[i]));
      DelayKernelTask* task = new DelayKernelTask(std::stoi(delayKernelTasks[i]));
      threads.push_back(std::thread(process, (void*)(task)));
      tasks.push_back(task);
    }
  }

  timespec timeBegin;
  clock_gettime(CLOCK_REALTIME, &timeBegin);
  std::vector<int> last_processed_img(tasks.size());
  std::vector<int> processed_img_this_time(tasks.size());
  memset(processed_img_this_time.data(), 0, sizeof(int) * tasks.size());
  memset(last_processed_img.data(), 0, sizeof(int) * tasks.size());

  /*
   run benchmark for the duration time and print the benchmark log every 1s
  */
  int duration = 15;
  bool is_running_forever{false};
  if (duration == -1) {
    duration = 1;
    is_running_forever = true;
  }
  int t = 0;
  while (t < duration) {
    if (!is_running_forever) {
      t = t + 1;
    }
    sleep(1);

    uint64_t totalImages = 0;
    timespec timeNow;
    clock_gettime(CLOCK_REALTIME, &timeNow);
    const timespec timeElapsed = time_diff(timeBegin, timeNow);

    const double seconds = timeElapsed.tv_sec + double(timeElapsed.tv_nsec) * double(1e-9);

    for (size_t n = 0; n < tasks.size(); n++) {
      processed_img_this_time[n] = tasks[n]->imgProcessed() - last_processed_img[n];
      last_processed_img[n] = tasks[n]->imgProcessed();
    }

    for (size_t n = 0; n < tasks.size(); n++)
      totalImages += processed_img_this_time[n];

    const double imagesPerSec = double(totalImages) / seconds;

    printf("\ntotal: %f img/sec  (", imagesPerSec);
    for (size_t n = 0; n < tasks.size(); n++) {
      printf("%s %f img/sec", tasks[n]->getTaskType(), double(processed_img_this_time[n]) / seconds);
      if (n < tasks.size() - 1)
        printf(", ");
    }
    printf(")\n");
    clock_gettime(CLOCK_REALTIME, &timeBegin);
  }

  for (auto& it : threads) {
    it.join();
  }
  return 0;
}
