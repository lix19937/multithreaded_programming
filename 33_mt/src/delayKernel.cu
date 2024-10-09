
#include <cuda_runtime.h>
#include "delayKernel.h"

__global__ void delayKernel(int32_t ms) {
  for (int32_t i = 0; i < ms; i++) {
    // Sleep 1 ms
    __nanosleep(1000000U);
  }
}

bool DelayKernelTask::Run() {
  delayKernel<<<1, 1, 0, mNewStream>>>(mDuration);
  cudaStreamSynchronize(mNewStream);
  mInference_count++;
  return true;
}