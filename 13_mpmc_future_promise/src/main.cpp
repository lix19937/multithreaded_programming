
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include "logger.hpp"

struct Data {
  int id;
  std::shared_ptr<std::promise<std::string>> status;
};

std::queue<Data> q1_;
std::queue<Data> q2_;
std::queue<Data> q3_;
std::queue<Data> q4_;

std::mutex mtx_;
std::condition_variable cv_;

int global_id = 0;

void produce(int duration) {
  while (1) {
    LOGI(PURPLE "Producer produces task %d", global_id);

    // preprocess module
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    LOGI(DGREEN "\tPreprocess finished");

    // N infer module which exec by consume thread
    Data data1, data2, data3, data4;

    {
      std::unique_lock<std::mutex> lock(mtx_);
      cv_.wait(lock, []() { return q1_.empty() && q2_.empty() && q3_.empty() && q4_.empty(); });

      data1.status.reset(new std::promise<std::string>());
      data2.status.reset(new std::promise<std::string>());
      data3.status.reset(new std::promise<std::string>());
      data4.status.reset(new std::promise<std::string>());

      data1.id = global_id;
      data2.id = global_id;
      data3.id = global_id;
      data4.id = global_id;

      q1_.push(data1);
      q2_.push(data2);
      q3_.push(data3);
      q4_.push(data4);
    }
    cv_.notify_all();

    auto st1 = data1.status->get_future().get();
    auto st2 = data2.status->get_future().get();
    auto st3 = data3.status->get_future().get();
    auto st4 = data4.status->get_future().get();
    LOGI(DGREEN "\tInfer finished");

    // postprocess module
    std::this_thread::sleep_for(std::chrono::milliseconds(duration));
    LOGI(DGREEN "\tPostprocess finished");

    LOGI(
        PURPLE "\tAll Finished task %d, %s, %s, %s, %s\n",
        global_id,
        st1.c_str(),
        st2.c_str(),
        st3.c_str(),
        st4.c_str());

    global_id++;
  }
}

void consume(std::queue<Data>& q, const std::string& label, int duration) {
  while (1) {
    {
      std::unique_lock<std::mutex> lock(mtx_);
      cv_.wait(lock, [&]() { return !q.empty(); });

      Data cData = q.front();
      int id = cData.id;
      q.pop();

      // infer module
      // we use sleep to simulation infer process
      std::this_thread::sleep_for(std::chrono::milliseconds(duration));

      auto resp = label + "[" + std::to_string(cData.id) + "]";
      cData.status->set_value(resp);
      LOGI(RED "\t%-16s Infer finished task[%d]", label.c_str(), id);
    }
    cv_.notify_all();
  }
}

/*
 * 考虑一个场景，当producer生产完一个数据之后，consumer就开始处理这个数据
 * 生产数据和处理这个数据的过程，需要通过mutex lock来锁住，这个是我们到目前为止所作的事情
 *
 * 但是，如果producer希望后续的处理得到consumer的这个数据的话，
 * 就说明需要某种机制，让consumer处理完数据以后去把结果告诉给producer，因此需要实现线程间的通信
 * future, promise就是负责这个的
 *
 * 场景是这样的：
 * - 让producer分配n个数据，之后等待这n个数据被各个consumer处理, 等待期间需要同步
 * - n个consumer分别处理n个数据, 并将处理完的数据写回
 * - producer每检测到其中一个数据更新了，取消这个数据的等待
 * - 等所有的数据被consumer处理完了，producer就可以处理producer的后续计算
 * 以上就是一次任务的执行(1 producer - N consumer)。通过future-promise实现并行并大大提高一个任务的吞吐量
 *
 * 这个案例的实现是对每一个task分配一个对应的queue和线程，之后进行异步处理
 * 是否可以有更好的办法去处理呢？比如说只用一个queue来处理多个task的异步执行
 */

int main() {
  std::thread producer(produce, 100);

  std::thread consumer1(consume, std::ref(q1_), "object_detection", 10);
  std::thread consumer2(consume, std::ref(q2_), "segmentation", 15);
  std::thread consumer3(consume, std::ref(q3_), "depth_estimation", 20);
  std::thread consumer4(consume, std::ref(q4_), "pose_estimation", 15);

  producer.join();
  consumer1.join();
  consumer2.join();
  consumer3.join();
  consumer4.join();

  return 0;
}
