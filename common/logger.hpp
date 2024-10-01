
#pragma once

#define LOGD(...) logger::__make_log(__FILE__, __LINE__, logger::LogLevel::Debug, __VA_ARGS__)
#define LOGV(...) logger::__make_log(__FILE__, __LINE__, logger::LogLevel::Verbose, __VA_ARGS__)
#define LOGI(...) logger::__make_log(__FILE__, __LINE__, logger::LogLevel::Info, __VA_ARGS__)
#define LOGW(...) logger::__make_log(__FILE__, __LINE__, logger::LogLevel::Warning, __VA_ARGS__)
#define LOGE(...) logger::__make_log(__FILE__, __LINE__, logger::LogLevel::Error, __VA_ARGS__)
#define LOGF(...) logger::__make_log(__FILE__, __LINE__, logger::LogLevel::Fatal, __VA_ARGS__)

#define DGREEN "\033[36m"
#define BLUE "\033[34m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define RED "\033[31m"
#define PURPLE "\033[35m"
#define CLEAR "\033[0m"

namespace logger {
enum class LogLevel : int {
  Debug = 5,
  Verbose = 4,
  Info = 3,
  Warning = 2,
  Error = 1,
  Fatal = 0,
};
void set_log_level(LogLevel level);
void __make_log(const char* file, const int line, LogLevel level, const char* format, ...);
}; // namespace logger
