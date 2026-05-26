/*
 * Test 1: Driver API (muXXX) Profiling
 * =====================================
 * Directly calls mu* Driver API functions via dlsym.
 * Demonstrates: muInit, muDeviceGetCount, muDeviceGetName,
 *               muMemAlloc, muMemFree, muCtxCreate.
 *
 * Expected output: profiler report with "driver:mu*" entries.
 */

#include <stdio.h>
#include <dlfcn.h>
#include <stdint.h>

typedef int MUresult;
#define MUSA_SUCCESS 0

int main() {
    void* driver = dlopen("libmusa.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!driver) { fprintf(stderr, "dlopen libmusa.so.1 failed\n"); return 1; }

    typedef MUresult (*muInit_fn)(unsigned int);
    typedef MUresult (*muDeviceGetCount_fn)(int*);
    typedef MUresult (*muDeviceGetName_fn)(char*, int, int);
    typedef MUresult (*muMemAlloc_fn)(uintptr_t*, size_t);
    typedef MUresult (*muMemFree_fn)(uintptr_t);
    typedef MUresult (*muCtxCreate_fn)(void**, unsigned int, int);

    muInit_fn          pInit      = (muInit_fn)dlsym(driver, "muInit");
    muDeviceGetCount_fn pCount   = (muDeviceGetCount_fn)dlsym(driver, "muDeviceGetCount");
    muDeviceGetName_fn  pName    = (muDeviceGetName_fn)dlsym(driver, "muDeviceGetName");
    muMemAlloc_fn       pAlloc   = (muMemAlloc_fn)dlsym(driver, "muMemAlloc_v2");
    muMemFree_fn        pFree    = (muMemFree_fn)dlsym(driver, "muMemFree_v2");
    muCtxCreate_fn      pCtx     = (muCtxCreate_fn)dlsym(driver, "muCtxCreate");

    /* ── Exercise Driver APIs ── */
    pInit(0);

    int count = 0;
    pCount(&count);
    printf("[TEST1] Device count: %d\n", count);

    char name[256];
    pName(name, sizeof(name), 0);
    printf("[TEST1] Device 0: %s\n", name);

    void* ctx;
    pCtx(&ctx, 0, 0);

    uintptr_t ptr;
    pAlloc(&ptr, 1024 * 1024);   /* 1 MB */
    printf("[TEST1] Allocated 1MB at 0x%lx\n", (unsigned long)ptr);

    pFree(ptr);
    printf("[TEST1] Freed\n");

    dlclose(driver);
    printf("[TEST1] Done. Check profiler report above.\n");
    return 0;
}
