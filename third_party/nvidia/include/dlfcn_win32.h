#pragma once
// Windows compatibility shim for dlfcn.h (POSIX dynamic loading API)
#ifdef _WIN32
#ifndef TRITON_DLFCN_WIN32_H
#define TRITON_DLFCN_WIN32_H
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#define RTLD_NOLOAD 0
#define RTLD_LOCAL 0
#define RTLD_LAZY 0
inline void *dlopen(const char *name, int) { return (void *)LoadLibraryA(name); }
inline void *dlsym(void *handle, const char *name) { return (void *)GetProcAddress((HMODULE)handle, name); }
inline int dlclose(void *handle) { return FreeLibrary((HMODULE)handle) ? 0 : -1; }
inline const char *dlerror() {
  static thread_local char buf[256];
  FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, NULL, GetLastError(), 0, buf, sizeof(buf), NULL);
  return buf;
}
#endif // TRITON_DLFCN_WIN32_H
#else
#include <dlfcn.h>
#endif
