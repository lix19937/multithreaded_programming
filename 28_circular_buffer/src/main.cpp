
// https://en.wikipedia.org/wiki/Circular_buffer

#include "logger.hpp"

// size of circular buffer
enum { N = 16 };

// note: only (N - 1) elements can be stored at a given time
int buffer[N];
int writeIndx = 0;
int readIndx = 0;

int put(int item) {
  if ((writeIndx + 1) % N == readIndx) {
    // buffer is full, avoid overflow
    return 0;
  }

  // assign
  buffer[writeIndx] = item;

  // update
  writeIndx = (writeIndx + 1) % N;
  return 1;
}

int get(int* value) {
  if (readIndx == writeIndx) {
    // buffer is empty
    return 0;
  }

  // assign
  *value = buffer[readIndx];

  // update
  readIndx = (readIndx + 1) % N;
  return 1;
}

int main() {
  int value = 1001;
  while (put(value++))
    ;

  while (get(&value))
    LOGI("read %d", value);

  return 0;
}
