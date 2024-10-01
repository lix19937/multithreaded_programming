
#include <thread>
#include "pc.hpp"

void pc_test() {
  std::thread t1(pc::Producer, pc::q_);
  std::thread t2(pc::Consumer, pc::q_);
  t1.join();
  t2.join();
}

void video_infer_test() {
  std::thread t1(pc::VideoCapture);
  std::thread t2(pc::InferWorker);
  t1.join();
  t2.join();
}

int main() {
  pc_test();

  //   video_infer_test();
  return 0;
}
