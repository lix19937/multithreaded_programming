#include <Task.h>
#include <iostream>

#include <cassert>

void Task::ReportSyncType() {
  std::cout << "Sync Type:";
  switch (mSyncType) {
    case SyncType::StreamDefault:
      std::cout << "cudaStreamDefault" << std::endl;
      break;
    case SyncType::StreamNonBlock:
      std::cout << "cudaStreamNonBlocking" << std::endl;
      break;
    case SyncType::EventDefault:
      std::cout << "cudaEventDefault " << std::endl;
      break;
    case SyncType::EventBlock:
      std::cout << "cudaEventBlockingSync " << std::endl;
      break;
    default:
      assert(false && "INVALID SYNC TYPE");
  }
}

void Task::TaskInit(SyncType syncType) {
  this->mInference_count = 0;
  this->mTotal_inference_time = 0;
  this->mSyncType = syncType;

  switch (syncType) {
    case SyncType::StreamDefault:
      this->mStream = makeCudaStream(cudaStreamDefault, /*priority*/ 0);
      this->mEvent = makeCudaEvent(cudaEventDefault);
      break;
    case SyncType::StreamNonBlock:
      this->mStream = makeCudaStream(cudaStreamNonBlocking, /*priority*/ 0);
      this->mEvent = makeCudaEvent(cudaEventDefault);
      break;
    case SyncType::EventDefault:
      this->mEvent = makeCudaEvent(cudaEventDefault);
      this->mStream = makeCudaStream(cudaStreamDefault, /*priority*/ 0);
      break;
    case SyncType::EventBlock:
      this->mEvent = makeCudaEvent(cudaEventBlockingSync);
      this->mStream = makeCudaStream(cudaStreamDefault, /*priority*/ 0);
      break;
    default:
      assert(false && "INVALID SYNC TYPE");
  }
}

void Task::Sync() {
  if (mSyncType == SyncType::StreamDefault || mSyncType == SyncType::StreamNonBlock) {
    if (cudaSuccess != cudaStreamSynchronize(mStream.get())) {
      std::cout << "cudaStreamSynchronize failed" << std::endl;
    }
  } else if (mSyncType == SyncType::EventDefault || mSyncType == SyncType::EventBlock) {
    cudaEventRecord(mEvent.get(), mStream.get());
    if (cudaSuccess != cudaEventSynchronize(mEvent.get())) {
      std::cout << "cudaEventSynchronize failed" << std::endl;
    }
  } else {
    assert(false && "INVALID SYNC TYPE");
  }
  return;
}
