
#include "customTask.h"
#include <map>

std::map<std::string, ClassInfo*>* CustomTask::classInfoMap_;

void CustomTask::Register(ClassInfo* ci) {
  if (classInfoMap_ == nullptr) {
    CustomTask::classInfoMap_ = new std::map<std::string, ClassInfo*>();
  }
  if (NULL != ci && classInfoMap_->find(ci->m_className) == classInfoMap_->end()) {
    classInfoMap_->insert(std::map<std::string, ClassInfo*>::value_type(ci->m_className, ci));
  }
}

CustomTask* CustomTask::CreateObject(const std::string& name, SyncType syncType) {
  std::map<std::string, ClassInfo*>::const_iterator iter = classInfoMap_->find(name);
  if (iter != classInfoMap_->end()) {
    return iter->second->CreateObject(name, syncType);
  }
  return NULL;
}

ClassInfo::ClassInfo(const std::string& className, ObjectConstructorFn ctor)
    : m_className(className), m_objectConstructor(ctor) {
  CustomTask::Register(this);
}

ClassInfo::ClassInfo() {}

CustomTask* ClassInfo::CreateObject(const std::string& name, SyncType syncType) const {
  return m_objectConstructor ? (*m_objectConstructor)(name, syncType) : 0;
}
