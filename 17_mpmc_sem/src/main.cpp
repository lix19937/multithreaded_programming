
// https://github.com/codophobia/producer-consumer-problem-solution-in-c/tree/master
// https://github.com/codophobia/process-scheduling-algorithms/blob/master/roundrobin_scheduling.cpp

#include <pthread.h>
#include <semaphore.h>
#include <stdint.h>
#include "logger.hpp"

/*
  This program provides a possible solution for producer-consumer problem using mutex and semaphore.
  5 producers and 5 consumers to demonstrate the solution.
*/

#define MaxItems (5) // Maximum items a producer can produce or a consumer can consume
#define BufferSize (5) // Size of the buffer

sem_t empty;
sem_t full;
int in = 0;
int out = 0;
uint64_t buffer[BufferSize]{0, 0, 0, 0, 0};
pthread_mutex_t mutex;
uint64_t item = 1; // Produce content, here just use a uint64_t

void* producer(void* pno) {
  for (;;) {
    sem_wait(&empty);

    pthread_mutex_lock(&mutex);
    buffer[in] = item;
    LOGI(GREEN "Producer_%d: pushed %ld at %d", *((int*)pno), buffer[in], in);
    in = (in + 1) % BufferSize;
    ++item;
    pthread_mutex_unlock(&mutex);

    sem_post(&full);
  }
  return NULL;
}

void* consumer(void* cno) {
  for (;;) {
    sem_wait(&full);

    pthread_mutex_lock(&mutex);
    auto item = buffer[out];
    LOGI(RED "Consumer_%d: poped %ld from %d", *((int*)cno), item, out);
    out = (out + 1) % BufferSize;
    pthread_mutex_unlock(&mutex);

    sem_post(&empty);
  }
  return NULL;
}

int main() {
  pthread_t pro[5], con[5];
  pthread_mutex_init(&mutex, NULL);
  sem_init(&empty, 0, BufferSize);
  sem_init(&full, 0, 0);

  int a[5] = {1, 2, 3, 4, 5}; // Just used for numbering the producer and consumer

  int num_producer = 2;
  int num_consumer = 2;

  for (int i = 0; i < num_producer; i++) {
    pthread_create(&pro[i], NULL, producer, (void*)&a[i]);
  }
  for (int i = 0; i < num_consumer; i++) {
    pthread_create(&con[i], NULL, consumer, (void*)&a[i]);
  }

  for (int i = 0; i < num_producer; i++) {
    pthread_join(pro[i], NULL);
  }
  for (int i = 0; i < num_consumer; i++) {
    pthread_join(con[i], NULL);
  }

  pthread_mutex_destroy(&mutex);
  sem_destroy(&empty);
  sem_destroy(&full);

  return 0;
}
