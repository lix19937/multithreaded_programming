#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>

void checkCudaErrors(cudaError_t err);

constexpr int divUp(int a, int b) {
  return (a + b - 1) / b;
}

// RAII helpers to automatically manage memory resource and TensorRT objects.
template <typename T>
struct TrtDeleter {
  void operator()(T* p) noexcept {
    if (p != nullptr)
      delete p;
  }
};

template <typename T>
struct CuMemDeleter {
  void operator()(T* p) noexcept { checkCudaErrors(cudaFree(p)); }
};

template <typename T, template <typename> typename DeleterType = TrtDeleter>
using UniqPtr = std::unique_ptr<T, DeleterType<T>>;

template <typename T>
UniqPtr<T, CuMemDeleter> mallocCudaMem(size_t nbElems) {
  T* ptr = nullptr;
  checkCudaErrors(cudaMalloc((void**)&ptr, sizeof(T) * nbElems));
  return UniqPtr<T, CuMemDeleter>{ptr};
}

struct StreamDeleter {
  void operator()(CUstream_st* stream) noexcept { checkCudaErrors(cudaStreamDestroy(stream)); }
};

std::unique_ptr<CUstream_st, StreamDeleter> makeCudaStream(int flags = cudaStreamDefault, int priority = 0);

std::unique_ptr<CUstream_st, StreamDeleter> makeCudaStreamNew();
struct EventDeleter {
  void operator()(CUevent_st* event) noexcept { checkCudaErrors(cudaEventDestroy(event)); }
};

std::unique_ptr<CUevent_st, EventDeleter> makeCudaEvent(int flags = cudaEventDefault);

std::unique_ptr<CUevent_st, EventDeleter> makeCudaEventNew(void);

#ifndef checkCudaErrorsDRV
#define checkCudaErrorsDRV(err) __checkCudaErrorsDRV(err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrorsDRV(CUresult err, const char* file, const int line) {
  if (CUDA_SUCCESS != err) {
    const char* errorStr = NULL;
    cuGetErrorString(err, &errorStr);
    fprintf(
        stderr,
        "checkCudaErrorsDRV() Driver API error = %04d \"%s\" from file <%s>, "
        "line %i.\n",
        err,
        errorStr,
        file,
        line);
    exit(EXIT_FAILURE);
  }
}
#endif
