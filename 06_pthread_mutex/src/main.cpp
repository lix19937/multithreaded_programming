
#include <pthread.h>
#include <stdio.h>

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

volatile int mutex_cnt = 0;

void* mutex_conflict_test(void* para) {
  bool do_once = true;
  int local_cnt = 0;

  while (1) {
    pthread_mutex_lock(&lock);
    if (do_once) {
      printf(">> [%s], tid:%ld\n", (char*)para, pthread_self());
      do_once = false;
    }

    __sync_fetch_and_add(&mutex_cnt, 1);
    ++local_cnt;
    if (mutex_cnt > 120000000) {
      break;
    }
    pthread_mutex_unlock(&lock);
  }

  pthread_mutex_unlock(&lock);

  printf("[%s] local_cnt:%-6d tid:%ld\n", (char*)para, local_cnt, pthread_self());
  return NULL;
}

int main() {
  pthread_t tid1, tid2, tid3, tid4, tid5, tid6;

  pthread_mutex_lock(&lock);
  printf("main tid:%ld\n", pthread_self());

  pthread_create(&tid1, NULL, mutex_conflict_test, (char*)"1");
  pthread_create(&tid2, NULL, mutex_conflict_test, (char*)"2");
  pthread_create(&tid3, NULL, mutex_conflict_test, (char*)"3");
  pthread_create(&tid4, NULL, mutex_conflict_test, (char*)"4");
  pthread_create(&tid5, NULL, mutex_conflict_test, (char*)"5");
  pthread_create(&tid6, NULL, mutex_conflict_test, (char*)"6");

  printf("main done tid:%ld\n", pthread_self());

  pthread_mutex_unlock(&lock);

  pthread_join(tid1, NULL);
  pthread_join(tid2, NULL);
  pthread_join(tid3, NULL);
  pthread_join(tid4, NULL);
  pthread_join(tid5, NULL);
  pthread_join(tid6, NULL);

  return 0;
}
