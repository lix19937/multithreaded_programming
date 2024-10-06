# LLVM thread 的实现  

https://github.com/llvm/llvm-project/blob/release/3.8.x/libcxx/src/thread.cpp

本文分析的是llvm libc++的实现：https://github.com/llvm/llvm-project   

## class thread
thread类直接包装了一个pthread_t，在linux下实际是unsigned long int。

https://github.com/llvm/llvm-project/blob/release/3.8.x/libcxx/include/thread   

```cpp 
class  thread
{
    pthread_t __t_;
    id get_id() const _NOEXCEPT {return __t_;}
}

```

用了一个std::unique_ptr来包装用户定义的线程函数  



```cpp

namespace std
{

class thread{
public:
    class id;
    typedef pthread_t native_handle_type;

    thread() noexcept;
    template <class F, class ...Args> explicit thread(F&& f, Args&&... args);
    ~thread();

    thread(const thread&) = delete;
    thread(thread&& t) noexcept;

    thread& operator=(const thread&) = delete;
    thread& operator=(thread&& t) noexcept;

    void swap(thread& t) noexcept;

    bool joinable() const noexcept;
    void join();
    void detach();
    id get_id() const noexcept;
    native_handle_type native_handle();

    static unsigned hardware_concurrency() noexcept;
};

template <class _Tp>
class __thread_specific_ptr
{
    pthread_key_t __key_;
    static_assert((is_same<_Tp, __thread_struct>::value), "");
    __thread_specific_ptr();
    friend _LIBCPP_FUNC_VIS __thread_specific_ptr<__thread_struct>& __thread_local_data();

    __thread_specific_ptr(const __thread_specific_ptr&);
    __thread_specific_ptr& operator=(const __thread_specific_ptr&);

    static void __at_thread_exit(void*);
public:
    typedef _Tp* pointer;

    ~__thread_specific_ptr();

    pointer get() const {return static_cast<_Tp*>(pthread_getspecific(__key_));}
    pointer operator*() const {return *get();}
    pointer operator->() const {return get();}
    pointer release();
    void reset(pointer __p = nullptr);
};

__thread_specific_ptr<_Tp>::__thread_specific_ptr()
{
    int __ec = pthread_key_create(&__key_, &__thread_specific_ptr::__at_thread_exit);
}

__thread_specific_ptr<_Tp>::release()
{
    pointer __p = get();
    pthread_setspecific(__key_, 0);
    return __p;
}

void __thread_specific_ptr<_Tp>::reset(pointer __p)
{
    pointer __p_old = get();
    pthread_setspecific(__key_, __p);
    delete __p_old;
}

__thread_id get_id() _NOEXCEPT
{
    return pthread_self();
}

void* __thread_proxy(void* __vp)
{
    std::unique_ptr<_Fp> __p(static_cast<_Fp*>(__vp));
    (*__p)();
    return nullptr;
}

thread::thread(_Fp __f)
{
    int __ec = pthread_create(&__t_, 0, &__thread_proxy<_Fp>, __p.get());
}

void sleep_for(const chrono::duration<_Rep, _Period>& __d)
{
    using namespace chrono;
    if (__d > duration<_Rep, _Period>::zero())
    {
        _LIBCPP_CONSTEXPR duration<long double> _Max = nanoseconds::max();
        nanoseconds __ns;
        if (__d < _Max)
        {
            __ns = duration_cast<nanoseconds>(__d);
            if (__ns < __d)
                ++__ns;
        }
        else
            __ns = nanoseconds::max();
        sleep_for(__ns);
    }
}

void sleep_until(const chrono::time_point<_Clock, _Duration>& __t)
{
    using namespace chrono;
    mutex __mut;
    condition_variable __cv;
    unique_lock<mutex> __lk(__mut);
    while (_Clock::now() < __t)
        __cv.wait_until(__lk, __t);
}

void sleep_until(const chrono::time_point<chrono::steady_clock, _Duration>& __t)
{
    using namespace chrono;
    sleep_for(__t - steady_clock::now());
}

//===------------------------- thread.cpp----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

void thread::join()
{
  int ec = pthread_join(__t_, 0);

}

void thread::detach()
{
  ec = pthread_detach(__t_);

}

unsigned thread::hardware_concurrency() _NOEXCEPT
{
    long result = sysconf(_SC_NPROCESSORS_ONLN);
}

void sleep_for(const chrono::nanoseconds& ns)
{
  while (nanosleep(&ts, &ts) == -1 && errno == EINTR)
      ;
     
}
 
class _LIBCPP_HIDDEN __thread_struct_imp
{
    typedef vector<__assoc_sub_state*,
                          __hidden_allocator<__assoc_sub_state*> > _AsyncStates;

    typedef vector<pair<condition_variable*, mutex*>,
               __hidden_allocator<pair<condition_variable*, mutex*> > > _Notify;

    _AsyncStates async_states_;
    _Notify notify_;

    __thread_struct_imp(const __thread_struct_imp&);
    __thread_struct_imp& operator=(const __thread_struct_imp&);
public:
    __thread_struct_imp() {}
    ~__thread_struct_imp();

    void notify_all_at_thread_exit(condition_variable* cv, mutex* m);
    void __make_ready_at_thread_exit(__assoc_sub_state* __s);
};

__thread_struct_imp::~__thread_struct_imp()
{
    for (   _Notify::iterator i = notify_.begin(), 
            e = notify_.end();
            i != e; ++i)
    {
        i->second->unlock();
        i->first->notify_all();
    }
    for (_AsyncStates::iterator i = async_states_.begin(), e = async_states_.end();
            i != e; 
            ++i)
    {
        (*i)->__make_ready();
        (*i)->__release_shared();
    }
}

void __thread_struct_imp::notify_all_at_thread_exit(condition_variable* cv, mutex* m)
{
    notify_.push_back(pair<condition_variable*, mutex*>(cv, m));
}

void __thread_struct_imp::__make_ready_at_thread_exit(__assoc_sub_state* __s)
{
    async_states_.push_back(__s);
    __s->__add_shared();
}

void __thread_struct::notify_all_at_thread_exit(condition_variable* cv, mutex* m)
{
    __p_->notify_all_at_thread_exit(cv, m);
}

void __thread_struct::__make_ready_at_thread_exit(__assoc_sub_state* __s)
{
    __p_->__make_ready_at_thread_exit(__s);
}


```
