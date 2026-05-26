/*
 * Test 3: Overlay Statistics (Cross-Layer)
 * ========================================
 * Calls musaMalloc → internally triggers muMemAlloc + muDevicePrimaryCtxRetain.
 * Calls musaMemcpyAsync → internally triggers muMemcpyHtoDAsync.
 * Calls musaDeviceSynchronize → internally triggers muCtxSynchronize.
 *
 * Demonstrated overlay: "runtime:musaMalloc" (wrapper) sits ABOVE
 * "driver:muMemAlloc" + "driver:muDevicePrimaryCtxRetain" (children).
 * The profiler report shows BOTH layers in the same table, differentiated
 * by "runtime:" / "driver:" prefix.
 *
 * Expected: "runtime:musaMalloc" has large Total time but its children
 *           (driver:mu*) consume most of it → detected as wrapper.
 */

#include <stdio.h>
#include <dlfcn.h>
#include <stdint.h>

typedef int MUresult;

int main() {
    void* driver = dlopen("libmusa.so.1", RTLD_LAZY | RTLD_LOCAL);
    void* runtime = dlopen("libmusart.so.5", RTLD_LAZY | RTLD_LOCAL);
    if (!driver || !runtime) { fprintf(stderr, "dlopen failed\n"); return 1; }

    typedef MUresult (*muInit_fn)(unsigned int);
    muInit_fn muInit = (muInit_fn)dlsym(driver, "muInit");
    muInit(0);

    /* ── Overlay example 1: musaMalloc → muMemAlloc ── */
    typedef MUresult (*musaMalloc_fn)(void**, size_t);
    musaMalloc_fn musaMalloc = (musaMalloc_fn)dlsym(runtime, "musaMalloc");
    void* ptr = NULL;
    musaMalloc(&ptr, 1024 * 1024);
    printf("[TEST3] musaMalloc → %p\n", ptr);

    /* ── Overlay example 2: musaMemcpyAsync → muMemcpyHtoDAsync ── */
    typedef MUresult (*musaMemcpyAsync_fn)(void*, const void*, size_t, int);
    musaMemcpyAsync_fn musaMemcpyAsync = (musaMemcpyAsync_fn)dlsym(runtime, "musaMemcpyAsync");
    int src = 42;
    musaMemcpyAsync(ptr, &src, sizeof(int), 0);
    printf("[TEST3] musaMemcpyAsync done\n");

    /* ── Overlay example 3: musaDeviceSynchronize → muCtxSynchronize ── */
    typedef MUresult (*musaDeviceSynchronize_fn)(void);
    musaDeviceSynchronize_fn musaSync = (musaDeviceSynchronize_fn)dlsym(runtime, "musaDeviceSynchronize");
    musaSync();
    printf("[TEST3] musaDeviceSynchronize done\n");

    typedef MUresult (*musaFree_fn)(void*);
    musaFree_fn musaFree = (musaFree_fn)dlsym(runtime, "musaFree");
    musaFree(ptr);
    printf("[TEST3] musaFree done\n");

    dlclose(runtime); dlclose(driver);
    printf("[TEST3] Done. Look for 'runtime:' and 'driver:' entries above.\n");
    printf("[TEST3] 'runtime:musaMalloc' should show as a wrapper with most\n");
    printf("[TEST3] time spent in 'driver:muDevicePrimaryCtxRetain' child calls.\n");
    return 0;
}
