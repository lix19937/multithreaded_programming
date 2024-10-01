
#pragma once

#include <algorithm>
#include <string>
#include "tools.h"

enum SyncType {
  EventDefault = 0,
  EventBlock = 1,
  StreamDefault = 2,
  StreamNonBlock = 3,
};

class Task {
 public:
  void TaskInit(SyncType syncType);
  virtual ~Task() = default;
  virtual bool Run() = 0;
  virtual const char* getTaskType() = 0;
  void ReportSyncType();
  void Sync();
  int imgProcessed() { return mInference_count; };

 protected:
  std::unique_ptr<CUstream_st, StreamDeleter> mStream;
  std::unique_ptr<CUevent_st, EventDeleter> mEvent;
  unsigned long long mLast_inference_time;
  unsigned long long mTotal_inference_time;
  int mInference_count;
  std::string mTaskName;
  SyncType mSyncType;
};
