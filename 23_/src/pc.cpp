#include "pc.hpp"
#include <chrono>
#include <thread>
#include "logger.hpp"

namespace pc {

void Producer(std::queue<Job> q) {
  while (1) {
    auto str = s_base_ + std::to_string(iter_++);
    Job job;
    {
      std::unique_lock<std::mutex> lck(lock_);
      cv_.wait(lck, [&]() { return q_.size() < limit_; });

      job.pro.reset(new std::promise<std::string>());
      job.name = str;
      q_.push(job);
      LOGI("Producer has produced %s, now queue size is %d", str.c_str(), q_.size());
    }

    auto res = job.pro->get_future().get();
    LOGI("Producer get result: %s\n", res.c_str());

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void Consumer(std::queue<Job> q) {
  while (1) {
    if (!q_.empty()) {
      {
        std::lock_guard<std::mutex> lck(lock_);
        // get
        auto pjob = q_.front();
        q_.pop();

        auto result = pjob.name + "{ ---resp--- }";
        pjob.pro->set_value(result);
        LOGI("\tConsumer has processed %s, now queue size is %d", pjob.name.c_str(), q_.size());
      }
      cv_.notify_one();

      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    std::this_thread::yield();
  }
}

void VideoCapture() {
  while (1) {
    auto str = s_base_ + std::to_string(iter_++);
    Job img;
    {
      std::unique_lock<std::mutex> lck(lock_);
      cv_.wait(lck, [&]() { return images_.size() < limit_; });

      img.pro.reset(new std::promise<std::string>());
      img.name = str;
      images_.push(img);
      LOGI("Generated image %s", img.name.c_str());
    }

    auto item = img.pro->get_future().get();
    LOGI("Infered image %s", item.c_str());
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
}

void InferWorker() {
  while (1) {
    if (!images_.empty()) {
      {
        std::lock_guard<std::mutex> lck(lock_);
        auto pjob = images_.front();
        images_.pop();

        LOGI("finished pop %s", pjob.name.c_str());
        auto result = pjob.name + "{---resp---}";
        pjob.pro->set_value(result);
      }
      cv_.notify_all();

      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    std::this_thread::yield();
  }
}

} // namespace pc
