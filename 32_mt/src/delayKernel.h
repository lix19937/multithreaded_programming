
#pragma once

#include "Task.h"

class DelayKernelTask : public Task {
 public:
  DelayKernelTask(int32_t milliseconds) {
    mDuration = milliseconds;
    cudaStreamCreate(&mNewStream);
  }

  bool Run() override;
  const char* getTaskType() override { return "delayKernel"; }

 private:
  int32_t mDuration;
  cudaStream_t mNewStream;
};
