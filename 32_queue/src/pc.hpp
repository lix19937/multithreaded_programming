#pragma once

#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>

namespace pc {

struct Job {
  std::shared_ptr<std::promise<std::string>> pro;
  std::string name;
};

void Producer(std::queue<Job> q);
void Consumer(std::queue<Job> q);
void VideoCapture();
void InferWorker();

static std::queue<Job> q_;
static std::queue<Job> images_;
static std::string s_base_ = "image_";
static int iter_ = 0;
static long unsigned int limit_ = 10;

static std::mutex lock_;
static std::condition_variable cv_;

}; // namespace pc
