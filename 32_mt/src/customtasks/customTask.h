
#pragma once

#include <map>
#include "Task.h"

// TODO: set it to real flag
#define DECLARE_CLASS()          \
 protected:                      \
  static ClassInfo ms_classinfo; \
                                 \
 public:                         \
  static CustomTask* CreateObject(const std::string& name, SyncType syncType);

#define IMPLEMENT_CLASS(interface_name, class_name)                                                  \
  ClassInfo class_name::ms_classinfo(interface_name, (ObjectConstructorFn)class_name::CreateObject); \
  CustomTask* class_name::CreateObject(const std::string& name, SyncType syncType) {                 \
    return new class_name(name, syncType);                                                           \
  };

class ClassInfo;
class CustomTask;
typedef CustomTask* (*ObjectConstructorFn)(const std::string&, SyncType);

class ClassInfo {
 public:
  ClassInfo(const std::string& className, ObjectConstructorFn ctor);
  ClassInfo();
  CustomTask* CreateObject(const std::string& name, SyncType syncType) const;

 public:
  std::string m_className;
  ObjectConstructorFn m_objectConstructor;
};

// -----------------------------------------------------
class CustomTask : public Task {
 public:
  virtual ~CustomTask() {}
  static void Register(ClassInfo* ci);
  static CustomTask* CreateObject(const std::string& name, SyncType syncType);
  static std::map<std::string, ClassInfo*>* classInfoMap_;
};
