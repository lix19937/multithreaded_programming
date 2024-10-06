
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

// mutex + condition 实际上是绑定在一起的

// Declaration of thread condition variable
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

// declaring mutex
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

void* release_fun(void*) {
  pthread_mutex_lock(&lock);

  // Let's signal condition variable cond
  printf("Signaling condition variable cond\n");
  pthread_cond_signal(&cond);

  pthread_mutex_unlock(&lock);

  return NULL;
}

void* blocked_thread(void*) {
  // acquire a lock
  pthread_mutex_lock(&lock); /// +++++++++++ #1

  printf("Waiting on condition variable cond\n");
  pthread_cond_wait(&cond, &lock);

  /*
    Locking and unlocking of the thread will happen only when blocked_thread's pthread_cond_wait() function is signaled
    to be unblocked.
    https://sources.debian.org/src/glibc/2.40-3/nptl/pthread_cond_wait.c
    https://linux.die.net/man/3/pthread_cond_wait
    https://docs.oracle.com/cd/E19120-01/open.solaris/816-5137/6mba5vq3s/index.html

    释放mutex, 把当前线程放到condition的等待队列里     mutex_unlock;
    等待被唤醒（当其它线程调用pthread_cond_signal或者pthread_cond_broadcast时）;
    当前线程被唤醒之后，对mutex加锁，再返回             mutex_lock

  */

  // release lock
  pthread_mutex_unlock(&lock); /// ---------- #2

  printf("Returning thread\n");

  return NULL;
}

int main() {
  pthread_t tid;

  pthread_create(&tid, NULL, blocked_thread, NULL);

  // sleep for 1 sec so that thread 1 would get a chance to run first
  sleep(1);

  pthread_t tid2;
  pthread_create(&tid2, NULL, release_fun, NULL);

  pthread_join(tid2, NULL);
  pthread_join(tid, NULL);

  return 0;
}
