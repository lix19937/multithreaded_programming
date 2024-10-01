
#include "tools.h"

void checkCudaErrors(cudaError_t err) {
  if (err != cudaSuccess)
    throw std::runtime_error(cudaGetErrorName(err));
}

std::unique_ptr<CUstream_st, StreamDeleter> makeCudaStream(int flags, int priority) {
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreateWithPriority(&stream, flags, priority));
  return std::unique_ptr<CUstream_st, StreamDeleter>{stream};
}

std::unique_ptr<CUstream_st, StreamDeleter> makeCudaStreamNew() {
  cudaStream_t stream;
  checkCudaErrors(cudaStreamCreate(&stream));
  return std::unique_ptr<CUstream_st, StreamDeleter>{stream};
}

std::unique_ptr<CUevent_st, EventDeleter> makeCudaEvent(int flags) {
  cudaEvent_t event;
  checkCudaErrors(cudaEventCreateWithFlags(&event, flags));
  return std::unique_ptr<CUevent_st, EventDeleter>{event};
}

std::unique_ptr<CUevent_st, EventDeleter> makeCudaEventNew(void) {
  cudaEvent_t event;
  checkCudaErrors(cudaEventCreate(&event));
  return std::unique_ptr<CUevent_st, EventDeleter>{event};
}
