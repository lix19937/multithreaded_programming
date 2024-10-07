
## base pattern  

 thread 1 is waiting for a condition, which may be fulfilled by thread 2.

```cpp

thread1:
    pthread_mutex_lock(&mutex);
    while (!condition)
        pthread_cond_wait(&cond, &mutex);

    // do something that requires holding the mutex and condition is true
    // ...

    pthread_mutex_unlock(&mutex);

thread2:
    pthread_mutex_lock(&mutex);

    // do something that might make condition true
    // ... 

    pthread_cond_signal(&cond);
    pthread_mutex_unlock(&mutex);
```

The two threads have some shared data structure that the mutex is protecting access to.     
The first thread wants to wait until some condition is true then immediately do some operation      
(with no race condition opportunity for some other thread to come in between the condition check and action and make the condition false.)   

The second thread is doing something that might make the condition true, so it needs to wake up anyone that might be waiting for it.    
  

## output   

```
2024-10-07 12:25:45 I main.cpp:62][Producer]:push package_0, current length 1, tid:140125676873472
2024-10-07 12:25:45 I main.cpp:100]tid:140125668480768 ->         consumer_0 started a new iter
2024-10-07 12:25:45 I main.cpp:100]tid:140125660088064 ->         consumer_1 started a new iter
2024-10-07 12:25:45 I main.cpp:110]tid:140125660088064 -->>>      consumer_1 acquired lock, other consumer blocked
2024-10-07 12:25:45 I main.cpp:122]tid:140125660088064 ----->>>>  consumer_1 queue is not empty, allow to pop
2024-10-07 12:25:45 I main.cpp:128]tid:140125660088064 ---------- consumer_1 pop package_0, current length 0, will release lock
2024-10-07 12:25:45 I main.cpp:110]tid:140125668480768 -->>>      consumer_0 acquired lock, other consumer blocked
2024-10-07 12:25:45 I main.cpp:114]tid:140125668480768 ---------- consumer_0 thread is blocked, waiting...
2024-10-07 12:25:45 I main.cpp:100]tid:140125660088064 ->         consumer_1 started a new iter
2024-10-07 12:25:45 I main.cpp:110]tid:140125660088064 -->>>      consumer_1 acquired lock, other consumer blocked
2024-10-07 12:25:45 I main.cpp:114]tid:140125660088064 ---------- consumer_1 thread is blocked, waiting...
2024-10-07 12:25:46 I main.cpp:62][Producer]:push package_1, current length 1, tid:140125676873472
2024-10-07 12:25:46 I main.cpp:62][Producer]:push package_2, current length 2, tid:140125676873472
2024-10-07 12:25:47 I main.cpp:62][Producer]:push package_3, current length 3, tid:140125676873472
2024-10-07 12:25:47 I main.cpp:62][Producer]:push package_4, current length 4, tid:140125676873472
2024-10-07 12:25:47 I main.cpp:66][Producer]:Activate all blocked threads
2024-10-07 12:25:47 I main.cpp:120]tid:140125668480768 ----->>>>  consumer_0 unblocked by cv notice, allow to pop
2024-10-07 12:25:47 I main.cpp:128]tid:140125668480768 ---------- consumer_0 pop package_4, current length 3, will release lock
2024-10-07 12:25:47 I main.cpp:120]tid:140125660088064 ----->>>>  consumer_1 unblocked by cv notice, allow to pop
2024-10-07 12:25:47 I main.cpp:128]tid:140125660088064 ---------- consumer_1 pop package_3, current length 2, will release lock
2024-10-07 12:25:47 I main.cpp:100]tid:140125668480768 ->         consumer_0 started a new iter
2024-10-07 12:25:47 I main.cpp:110]tid:140125668480768 -->>>      consumer_0 acquired lock, other consumer blocked
2024-10-07 12:25:47 I main.cpp:100]tid:140125660088064 ->         consumer_1 started a new iter
2024-10-07 12:25:47 I main.cpp:122]tid:140125668480768 ----->>>>  consumer_0 queue is not empty, allow to pop
2024-10-07 12:25:47 I main.cpp:128]tid:140125668480768 ---------- consumer_0 pop package_2, current length 1, will release lock
2024-10-07 12:25:47 I main.cpp:110]tid:140125660088064 -->>>      consumer_1 acquired lock, other consumer blocked
2024-10-07 12:25:47 I main.cpp:122]tid:140125660088064 ----->>>>  consumer_1 queue is not empty, allow to pop
2024-10-07 12:25:47 I main.cpp:128]tid:140125660088064 ---------- consumer_1 pop package_1, current length 0, will release lock
2024-10-07 12:25:47 I main.cpp:100]tid:140125668480768 ->         consumer_0 started a new iter
2024-10-07 12:25:47 I main.cpp:110]tid:140125668480768 -->>>      consumer_0 acquired lock, other consumer blocked
2024-10-07 12:25:47 I main.cpp:114]tid:140125668480768 ---------- consumer_0 thread is blocked, waiting...
2024-10-07 12:25:47 I main.cpp:100]tid:140125660088064 ->         consumer_1 started a new iter
2024-10-07 12:25:47 I main.cpp:110]tid:140125660088064 -->>>      consumer_1 acquired lock, other consumer blocked
2024-10-07 12:25:47 I main.cpp:114]tid:140125660088064 ---------- consumer_1 thread is blocked, waiting...
2024-10-07 12:25:48 I main.cpp:62][Producer]:push package_5, current length 1, tid:140125676873472
2024-10-07 12:25:48 I main.cpp:62][Producer]:push package_6, current length 2, tid:140125676873472
2024-10-07 12:25:49 I main.cpp:62][Producer]:push package_7, current length 3, tid:140125676873472
2024-10-07 12:25:49 I main.cpp:62][Producer]:push package_8, current length 4, tid:140125676873472
2024-10-07 12:25:49 I main.cpp:66][Producer]:Activate all blocked threads
2024-10-07 12:25:49 I main.cpp:120]tid:140125660088064 ----->>>>  consumer_1 unblocked by cv notice, allow to pop
2024-10-07 12:25:49 I main.cpp:128]tid:140125660088064 ---------- consumer_1 pop package_8, current length 3, will release lock
2024-10-07 12:25:49 I main.cpp:120]tid:140125668480768 ----->>>>  consumer_0 unblocked by cv notice, allow to pop
2024-10-07 12:25:49 I main.cpp:128]tid:140125668480768 ---------- consumer_0 pop package_7, current length 2, will release lock
2024-10-07 12:25:49 I main.cpp:100]tid:140125668480768 ->         consumer_0 started a new iter
2024-10-07 12:25:49 I main.cpp:110]tid:140125668480768 -->>>      consumer_0 acquired lock, other consumer blocked
2024-10-07 12:25:49 I main.cpp:100]tid:140125660088064 ->         consumer_1 started a new iter
2024-10-07 12:25:49 I main.cpp:122]tid:140125668480768 ----->>>>  consumer_0 queue is not empty, allow to pop
2024-10-07 12:25:49 I main.cpp:128]tid:140125668480768 ---------- consumer_0 pop package_6, current length 1, will release lock
2024-10-07 12:25:49 I main.cpp:110]tid:140125660088064 -->>>      consumer_1 acquired lock, other consumer blocked
2024-10-07 12:25:49 I main.cpp:122]tid:140125660088064 ----->>>>  consumer_1 queue is not empty, allow to pop
2024-10-07 12:25:49 I main.cpp:128]tid:140125660088064 ---------- consumer_1 pop package_5, current length 0, will release lock
2024-10-07 12:25:49 I main.cpp:100]tid:140125660088064 ->         consumer_1 started a new iter
2024-10-07 12:25:49 I main.cpp:110]tid:140125660088064 -->>>      consumer_1 acquired lock, other consumer blocked
2024-10-07 12:25:49 I main.cpp:114]tid:140125660088064 ---------- consumer_1 thread is blocked, waiting...
2024-10-07 12:25:49 I main.cpp:100]tid:140125668480768 ->         consumer_0 started a new iter
2024-10-07 12:25:49 I main.cpp:110]tid:140125668480768 -->>>      consumer_0 acquired lock, other consumer blocked
2024-10-07 12:25:49 I main.cpp:114]tid:140125668480768 ---------- consumer_0 thread is blocked, waiting...
2024-10-07 12:25:50 I main.cpp:62][Producer]:push package_9, current length 1, tid:140125676873472
2024-10-07 12:25:50 I main.cpp:62][Producer]:push package_10, current length 2, tid:140125676873472
```
