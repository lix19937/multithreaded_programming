// Copyright lix19937

#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include "logger.hpp"

class ThreadPool {
 public:
  using job_type = std::function<void()>;
  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(jobs_queue_mutex_);
      stop_ = true;
    }
    jobs_queue_cv_.notify_all();

    for (auto& t : threads_) {
      if (t.joinable()) {
        t.join();
      }
    }
    threads_.clear();
  }

  ThreadPool(int size, bool bindcpu = false) {
    auto totalcores = std::thread::hardware_concurrency();

    if (!threads_.empty()) {
      throw std::runtime_error("thread pool already started");
    }

    /// create thread
    /// lamada express as thread function, return tid async, thread_task run at another thread
    for (int i = 0; i < size; ++i) {
      threads_.emplace_back([this] { thread_task(); });
    }

    /// setaffinity
    /// at this timepoint, we have get all threads tid
    bindcpu_ = bindcpu;
    if (bindcpu_) {
      for (int i = 0; i < size; ++i) {
        // Create a cpu_set_t object representing a set of CPUs. Clear it and mark only CPU i as set.
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET((size_t)i, &cpuset);
        int rc = pthread_setaffinity_np(threads_[i].native_handle(), sizeof(cpu_set_t), &cpuset);
        if (rc != 0) {
          printf("Error calling pthread_setaffinity_np:%d\n", rc);
        }
      }
    }
  }

  void enqueue(job_type job) {
    if (job == nullptr) {
      throw std::invalid_argument("job is nullptr");
    }

    {
      std::unique_lock<std::mutex> l(jobs_queue_mutex_);
      jobs_queue_.push(job);
    }
    jobs_queue_cv_.notify_one();
  }

 private:
  void thread_task() {
    while (1) {
      job_type job;
      {
        std::unique_lock<std::mutex> lock(jobs_queue_mutex_);
        jobs_queue_cv_.wait(lock, [this] { return !jobs_queue_.empty() || stop_; });

        if (stop_) {
          break;
        }

        job = jobs_queue_.front();
        jobs_queue_.pop();
      }
      job();
    }
  }

  bool bindcpu_ = false;
  bool stop_ = false;

  // synchronization
  std::mutex jobs_queue_mutex_;
  std::condition_variable jobs_queue_cv_;

  // the task queue
  std::queue<job_type> jobs_queue_;

  // need to keep track of threads so we can join them
  std::vector<std::thread> threads_;
};
