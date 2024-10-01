

## case 1    

Producer speed > Consumer speed      

```
2017-01-09 18:11:42 I main.cpp:54]Producer  :pushed  package_0, now queue size is 1|4, tid:139770361669376
2017-01-09 18:11:42 I main.cpp:65]Producer   notify_all, pred():false , tid:139770361669376
2017-01-09 18:11:42 I main.cpp:54]Producer  :pushed  package_1, now queue size is 2|4, tid:139770353276672
2017-01-09 18:11:42 I main.cpp:65]Producer   notify_all, pred():true  , tid:139770353276672  // begin to unblock Consumer
2017-01-09 18:11:42 I main.cpp:94]Consumer  :poped   package_0, now queue size is 1|1, tid:139770344883968
2017-01-09 18:11:42 I main.cpp:104]Consumer   notify_all, pred():true  , tid:139770344883968

2017-01-09 18:11:42 I main.cpp:54]Producer  :pushed  package_2, now queue size is 2|4, tid:139770353276672
2017-01-09 18:11:42 I main.cpp:94]Consumer  :poped   package_1, now queue size is 1|1, tid:139770344883968
2017-01-09 18:11:42 I main.cpp:65]Producer   notify_all, pred():false , tid:139770353276672
2017-01-09 18:11:42 I main.cpp:54]Producer  :pushed  package_3, now queue size is 2|4, tid:139770361669376
2017-01-09 18:11:42 I main.cpp:65]Producer   notify_all, pred():true  , tid:139770361669376
2017-01-09 18:11:42 I main.cpp:104]Consumer   notify_all, pred():true  , tid:139770344883968

2017-01-09 18:11:43 I main.cpp:54]Producer  :pushed  package_4, now queue size is 3|4, tid:139770353276672
2017-01-09 18:11:43 I main.cpp:65]Producer   notify_all, pred():true  , tid:139770353276672
2017-01-09 18:11:43 I main.cpp:54]Producer  :pushed  package_5, now queue size is 4|4, tid:139770361669376
2017-01-09 18:11:43 I main.cpp:65]Producer   notify_all, pred():true  , tid:139770361669376
2017-01-09 18:11:43 I main.cpp:94]Consumer  :poped   package_2, now queue size is 3|1, tid:139770344883968
2017-01-09 18:11:43 I main.cpp:104]Consumer   notify_all, pred():true  , tid:139770344883968

2017-01-09 18:11:43 I main.cpp:54]Producer  :pushed  package_6, now queue size is 4|4, tid:139770353276672
2017-01-09 18:11:43 I main.cpp:65]Producer   notify_all, pred():true  , tid:139770353276672
2017-01-09 18:11:43 I main.cpp:94]Consumer  :poped   package_3, now queue size is 3|1, tid:139770344883968
2017-01-09 18:11:43 I main.cpp:104]Consumer   notify_all, pred():true  , tid:139770344883968

2017-01-09 18:11:43 I main.cpp:54]Producer  :pushed  package_7, now queue size is 4|4, tid:139770361669376
2017-01-09 18:11:43 I main.cpp:65]Producer   notify_all, pred():true  , tid:139770361669376
2017-01-09 18:11:44 I main.cpp:94]Consumer  :poped   package_4, now queue size is 3|1, tid:139770344883968
2017-01-09 18:11:44 I main.cpp:104]Consumer   notify_all, pred():true  , tid:139770344883968

2017-01-09 18:11:44 I main.cpp:54]Producer  :pushed  package_8, now queue size is 4|4, tid:139770353276672
2017-01-09 18:11:44 I main.cpp:65]Producer   notify_all, pred():true  , tid:139770353276672
2017-01-09 18:11:44 I main.cpp:94]Consumer  :poped   package_5, now queue size is 3|1, tid:139770344883968
2017-01-09 18:11:44 I main.cpp:104]Consumer   notify_all, pred():true  , tid:139770344883968

2017-01-09 18:11:44 I main.cpp:54]Producer  :pushed  package_9, now queue size is 4|4, tid:139770353276672
2017-01-09 18:11:44 I main.cpp:65]Producer   notify_all, pred():true  , tid:139770353276672
2017-01-09 18:11:45 I main.cpp:94]Consumer  :poped   package_6, now queue size is 3|1, tid:139770344883968
2017-01-09 18:11:45 I main.cpp:104]Consumer   notify_all, pred():true  , tid:139770344883968

2017-01-09 18:11:45 I main.cpp:54]Producer  :pushed  package_10, now queue size is 4|4, tid:139770353276672
2017-01-09 18:11:45 I main.cpp:65]Producer   notify_all, pred():true  , tid:139770353276672
2017-01-09 18:11:45 I main.cpp:94]Consumer  :poped   package_7, now queue size is 3|1, tid:139770344883968
2017-01-09 18:11:45 I main.cpp:104]Consumer   notify_all, pred():true  , tid:139770344883968
```

## case 2   

Producer speed <= Consumer speed      

```   
2017-01-09 18:23:11 I main.cpp:54]Producer  :pushed  package_0, now queue size is 1|4, tid:140121222178560
2017-01-09 18:23:11 I main.cpp:65]Producer   notify_all, pred():false , tid:140121222178560
2017-01-09 18:23:11 I main.cpp:54]Producer  :pushed  package_1, now queue size is 2|4, tid:140121222178560
2017-01-09 18:23:11 I main.cpp:65]Producer   notify_all, pred():true  , tid:140121222178560 // begin to unblock Consumer
2017-01-09 18:23:11 I main.cpp:94]Consumer  :poped   package_0, now queue size is 1|1, tid:140121205393152
2017-01-09 18:23:11 I main.cpp:104]Consumer   notify_all, pred():true  , tid:140121205393152

2017-01-09 18:23:12 I main.cpp:54]Producer  :pushed  package_2, now queue size is 2|4, tid:140121222178560
2017-01-09 18:23:12 I main.cpp:65]Producer   notify_all, pred():true  , tid:140121222178560
2017-01-09 18:23:12 I main.cpp:94]Consumer  :poped   package_1, now queue size is 1|1, tid:140121205393152
2017-01-09 18:23:12 I main.cpp:104]Consumer   notify_all, pred():true  , tid:140121205393152

2017-01-09 18:23:12 I main.cpp:54]Producer  :pushed  package_3, now queue size is 2|4, tid:140121222178560
2017-01-09 18:23:12 I main.cpp:65]Producer   notify_all, pred():true  , tid:140121222178560
2017-01-09 18:23:12 I main.cpp:94]Consumer  :poped   package_2, now queue size is 1|1, tid:140121213785856
2017-01-09 18:23:12 I main.cpp:104]Consumer   notify_all, pred():true  , tid:140121213785856
```