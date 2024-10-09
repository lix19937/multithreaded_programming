
#include <stdio.h>
#include <mutex>
#include <thread>

/*

There are at least two things 'spurious wakeup' could mean:

A thread blocked in pthread_cond_wait can return from the call even though no call to pthread_cond_signal or
pthread_cond_broadcast on the condition occurred.

A thread blocked in pthread_cond_wait returns because of a call to
pthread_cond_signal or pthread_cond_broadcast, however after reacquiring the mutex the underlying predicate is found to
no longer be true.

https://stackoverflow.com/questions/8594591/why-does-pthread-cond-wait-have-spurious-wakeups

*/

int main() {
  return 0;
}
