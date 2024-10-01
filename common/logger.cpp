
#include "logger.hpp"
#include <libgen.h>
#include <stdarg.h>
#include <time.h>
#include <iostream>
#include <string>

namespace logger {
LogLevel g_level = LogLevel::Info;

std::string time_now() {
  time_t timep;
  char time_string[36]{0};

  time(&timep);
  const tm& t = *localtime(&timep);
  snprintf(
      time_string,
      sizeof(time_string),
      "%04d-%02d-%02d %02d:%02d:%02d",
      t.tm_year + 1900,
      t.tm_mon + 1,
      t.tm_mday,
      t.tm_hour,
      t.tm_min,
      t.tm_sec);
  return time_string;
}

std::string level_string(LogLevel level) {
  switch (level) {
    case LogLevel::Debug:
      return "D";
    case LogLevel::Verbose:
      return "V";
    case LogLevel::Info:
      return "I";
    case LogLevel::Warning:
      return "W";
    case LogLevel::Error:
      return "E";
    case LogLevel::Fatal:
      return "F";
    default:
      return "unknown";
  }
}

void set_log_level(LogLevel level) {
  g_level = level;
}

void __make_log(const char* file, int line, LogLevel level, const char* format, ...) {
  va_list vl;
  va_start(vl, format);

  char buff[2048];
  int n = 0;
  auto now = time_now();

  // print time
  n += snprintf(buff + n, sizeof(buff) - n, CLEAR "%s" CLEAR, now.c_str());

  // print log level
  if (level == LogLevel::Debug) {
    n += snprintf(buff + n, sizeof(buff) - n, DGREEN " %s " CLEAR, level_string(level).c_str());
  } else if (level == LogLevel::Verbose) {
    n += snprintf(buff + n, sizeof(buff) - n, PURPLE " %s " CLEAR, level_string(level).c_str());
  } else if (level == LogLevel::Info) {
    n += snprintf(buff + n, sizeof(buff) - n, GREEN " %s " CLEAR, level_string(level).c_str());
  } else if (level == LogLevel::Warning) {
    n += snprintf(buff + n, sizeof(buff) - n, YELLOW " %s " CLEAR, level_string(level).c_str());
  } else if (level == LogLevel::Error || level == LogLevel::Fatal) {
    n += snprintf(buff + n, sizeof(buff) - n, RED " %s " CLEAR, level_string(level).c_str());
  }

  // print file and line
  n += snprintf(buff + n, sizeof(buff) - n, "%s:%d]", basename((char*)file), line);

  // print va_args
  n += vsnprintf(buff + n, sizeof(buff) - n, format, vl);
  fprintf(stdout, "%s\n", buff);

  // free va_list
  va_end(vl);

  // when Fatal is abort
  if (level == LogLevel::Fatal) {
    fflush(stdout);
    abort();
  }
}
} // namespace logger
