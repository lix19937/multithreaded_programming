
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
  
