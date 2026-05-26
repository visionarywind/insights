/*
 * Test 2: Runtime API (musaXXX) Profiling
 * ========================================
 * Calls musa* Runtime API functions via dlsym on libmusart.so.
 * Demonstrates: musaGetDeviceCount, musaGetDevice, musaFree,
 *               musaGetDeviceProperties, musaDeviceSynchronize.
 * First initializes via Driver API (muInit).
 *
 * Expected output: profiler report with "runtime:musa*" entries
 *                  alongside "driver:mu*" entries (cross-layer).
 */

#include <stdio.h>
#include <dlfcn.h>
#include <stdint.h>

typedef int MUresult;

int main() {
    /* Step 1: Initialize via Driver API */
    void* driver = dlopen("libmusa.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!driver) { fprintf(stderr, "dlopen driver failed\n"); return 1; }

    typedef MUresult (*muInit_fn)(unsigned int);
    muInit_fn muInit = (muInit_fn)dlsym(driver, "muInit");
    muInit(0);
    printf("[TEST2] Driver initialized\n");

    /* Step 2: Call Runtime APIs */
    void* runtime = dlopen("libmusart.so.5", RTLD_LAZY | RTLD_LOCAL);
    if (!runtime) { fprintf(stderr, "dlopen runtime failed\n"); return 1; }

    typedef MUresult (*musaGetDeviceCount_fn)(int*);
    typedef MUresult (*musaGetDevice_fn)(int*);
    typedef MUresult (*musaGetDeviceProperties_fn)(void*, int);
    typedef MUresult (*musaFree_fn)(void*);
    typedef MUresult (*musaDeviceSynchronize_fn)(void);

    musaGetDeviceCount_fn fCount  = (musaGetDeviceCount_fn)dlsym(runtime, "musaGetDeviceCount");
    musaGetDevice_fn      fDev    = (musaGetDevice_fn)dlsym(runtime, "musaGetDevice");
    musaGetDeviceProperties_fn fProp = (musaGetDeviceProperties_fn)dlsym(runtime, "musaGetDeviceProperties");
    musaFree_fn           fFree   = (musaFree_fn)dlsym(runtime, "musaFree");
    musaDeviceSynchronize_fn fSync = (musaDeviceSynchronize_fn)dlsym(runtime, "musaDeviceSynchronize");

    int count = 0;
    fCount(&count);
    printf("[TEST2] musaGetDeviceCount → %d devices\n", count);

    int dev = 0;
    fDev(&dev);
    printf("[TEST2] musaGetDevice → %d\n", dev);

    char props[1024] = {};
    fProp(props, dev);
    printf("[TEST2] musaGetDeviceProperties done\n");

    fFree(NULL);   /* no-op but exercises the API path */
    fSync();       /* synchronize device */
    printf("[TEST2] musaFree + musaDeviceSynchronize done\n");

    dlclose(runtime);
    dlclose(driver);
    printf("[TEST2] Done. Check profiler report above.\n");
    return 0;
}
