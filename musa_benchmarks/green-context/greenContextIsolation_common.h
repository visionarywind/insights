#pragma once

#include "musa.h"
#include "musa_runtime.h"
#include "helper_musa.h"
#include "helper_musa_drvapi.h"

#ifdef TEST_ON_NVIDIA
#if !defined(CUDA_VERSION) || CUDA_VERSION < 12040
#define GREEN_CONTEXT_UNSUPPORTED_BUILD 1
#endif
#endif

#include <algorithm>
#include <cstdlib>
#include <iostream>

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#ifdef TEST_ON_NVIDIA
#define GREEN_CONTEXT_NOINLINE __noinline__
#else
#define GREEN_CONTEXT_NOINLINE __attribute__((noinline))
#endif

#if !defined(GREEN_CONTEXT_UNSUPPORTED_BUILD)

constexpr unsigned long long kDelaySpinCycles = 25000000ULL;
constexpr unsigned long long kCriticalSpinCycles = 6000000ULL;

constexpr int kSharedMemoryAlignment = 256;
constexpr int kSharedMemorySearchStep = 4 * 1024;

__global__ void nop_kernel() {}

__device__ GREEN_CONTEXT_NOINLINE unsigned char touch_dynamic_smem(
    volatile unsigned char* smem, int sharedBytes) {
    unsigned char acc = 0;
    for (int i = threadIdx.x; i < sharedBytes; i += blockDim.x) {
        smem[i] = static_cast<unsigned char>(i);
        acc |= smem[i];
    }
    return acc;
}

__global__ void sm_occupancy_spin_kernel(unsigned long long spinCycles,
                                         int* startedFlags, int marker,
                                         int sharedBytes) {
    extern __shared__ unsigned char smem[];
    touch_dynamic_smem(reinterpret_cast<volatile unsigned char*>(smem), sharedBytes);
    __syncthreads();

    if (threadIdx.x == 0 && startedFlags != nullptr) {
        // Callers use this path only for pre-timed delay kernels; critical
        // kernels pass nullptr so flag visibility is not part of their latency.
        __threadfence_system();
        startedFlags[blockIdx.x] = marker;
        __threadfence_system();
    }

    const unsigned long long start = clock64();
    while (clock64() - start < spinCycles) {
    }
}

static unsigned int align_up(unsigned int value, unsigned int alignment) {
    if (alignment == 0) {
        return value;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

static unsigned int sm_partition_granularity(const MUdevResource& res) {
    return res.sm.smCount >= 80 ? 8 : 2;
}

struct SmResourceSplit {
    MUdevResource all{};
    MUdevResource critical{};
    MUdevResource bulk{};
    unsigned int totalSms = 0;
    unsigned int granularity = 0;
    unsigned int criticalTarget = 0;
};

static SmResourceSplit split_sm_resources(MUdevice dev, unsigned int requestedCriticalSms,
                                          bool printTooSmallError) {
    SmResourceSplit split{};
    checkMuErrors(muDeviceGetDevResource(dev, &split.all, MU_DEV_RESOURCE_TYPE_SM));
    split.totalSms = split.all.sm.smCount;
    split.granularity = sm_partition_granularity(split.all);
    if (split.totalSms < split.granularity * 2) {
        if (printTooSmallError) {
            std::cerr << "Green Context benchmark requires at least "
                      << (split.granularity * 2) << " SMs, got "
                      << split.totalSms << std::endl;
        }
        std::exit(EXIT_WAIVED);
    }

    split.criticalTarget = align_up(requestedCriticalSms, split.granularity);
    if (split.criticalTarget + split.granularity > split.totalSms) {
        split.criticalTarget = split.granularity;
    }

    unsigned int nbGroups = 1;
    checkMuErrors(muDevSmResourceSplitByCount(
        &split.critical, &nbGroups, &split.all, &split.bulk, 0, split.criticalTarget));
    return split;
}

static musaStream_t runtime_stream(MUstream stream) {
    return reinterpret_cast<musaStream_t>(stream);
}

static MUstream driver_stream(musaStream_t stream) {
    return reinterpret_cast<MUstream>(stream);
}

static void check_kernel_launch(const char* label) {
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        std::cerr << "kernel launch failed at " << label << ": "
                  << musaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

struct LaunchConfig {
    int threadsPerBlock = 0;
    int sharedBytes = 0;
    int activeBlocksPerSm = 0;
};

static int align_down(int value, int alignment) {
    if (alignment == 0) {
        return value;
    }
    return (value / alignment) * alignment;
}

static int max_dynamic_shared_bytes() {
    musaDeviceProp prop{};
    checkMusaErrors(musaGetDeviceProperties(&prop, 0));
    size_t limit = prop.sharedMemPerBlock;
    if (prop.sharedMemPerBlockOptin > 0) {
        limit = std::max(limit, prop.sharedMemPerBlockOptin);
    }
    if (prop.sharedMemPerMultiprocessor > 0) {
        limit = std::min(limit, prop.sharedMemPerMultiprocessor);
    }
    return align_down(static_cast<int>(limit), kSharedMemoryAlignment);
}

static void set_dynamic_shared_limit(int sharedBytes) {
    checkMusaErrors(musaFuncSetAttribute(
        sm_occupancy_spin_kernel, musaFuncAttributeMaxDynamicSharedMemorySize,
        sharedBytes));
}

static LaunchConfig choose_launch_config(MUdevice dev) {
    int maxThreadsPerBlock = 0;
    checkMuErrors(muDeviceGetAttribute(
        &maxThreadsPerBlock, MU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev));

    const int threadCandidates[] = {1024, 768, 512, 384, 256, 128, 64};
    const int maxSharedBytes = max_dynamic_shared_bytes();
    LaunchConfig best{};
    int bestActiveBlocks = 1 << 30;

    for (int sharedBytes = maxSharedBytes;
         sharedBytes >= kSharedMemoryAlignment;
         sharedBytes -= kSharedMemorySearchStep) {
        const int alignedSharedBytes = align_down(sharedBytes, kSharedMemoryAlignment);
        if (alignedSharedBytes <= 0) {
            continue;
        }

        for (int threads : threadCandidates) {
            if (threads > maxThreadsPerBlock) {
                continue;
            }

            int activeBlocks = 0;
            checkMusaErrors(musaOccupancyMaxActiveBlocksPerMultiprocessor(
                &activeBlocks, sm_occupancy_spin_kernel, threads, alignedSharedBytes));

            if (activeBlocks == 1) {
                return {threads, alignedSharedBytes, activeBlocks};
            }

            if (activeBlocks > 0 &&
                (activeBlocks < bestActiveBlocks ||
                 (activeBlocks == bestActiveBlocks &&
                  (alignedSharedBytes > best.sharedBytes ||
                   (alignedSharedBytes == best.sharedBytes && threads > best.threadsPerBlock))))) {
                best = {threads, alignedSharedBytes, activeBlocks};
                bestActiveBlocks = activeBlocks;
            }
        }
    }

    if (best.threadsPerBlock == 0) {
        std::cerr << "cannot find a valid launch config for occupancy query" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return best;
}

struct GreenContextBundle {
    MUgreenCtx greenCtx = nullptr;
    MUcontext ctx = nullptr;
    MUstream stream = nullptr;
    MUdevResource smResource{};
    unsigned int smCount = 0;

    void create(MUdevice dev, const MUdevResource& smRes) {
        MUdevResourceDesc desc{};
        MUdevResource mutableRes = smRes;
        checkMuErrors(muDevResourceGenerateDesc(&desc, &mutableRes, 1));
        checkMuErrors(muGreenCtxCreate(&greenCtx, desc, dev, MU_GREEN_CTX_DEFAULT_STREAM));
        checkMuErrors(muCtxFromGreenCtx(&ctx, greenCtx));
        checkMuErrors(muCtxSetCurrent(ctx));
        checkMuErrors(muGreenCtxStreamCreate(&stream, greenCtx, MU_STREAM_NON_BLOCKING, 0));
        checkMuErrors(muGreenCtxGetDevResource(greenCtx, &smResource, MU_DEV_RESOURCE_TYPE_SM));
        smCount = smResource.sm.smCount;

        nop_kernel<<<1, 1, 0, runtime_stream(stream)>>>();
        check_kernel_launch("green context warmup");
        checkMusaErrors(musaStreamSynchronize(runtime_stream(stream)));
    }

    void destroy() {
        if (stream != nullptr) {
            checkMuErrors(muStreamDestroy(stream));
        }
        if (greenCtx != nullptr) {
            checkMuErrors(muGreenCtxDestroy(greenCtx));
        }
        stream = nullptr;
        greenCtx = nullptr;
        ctx = nullptr;
        smCount = 0;
        smResource = {};
    }
};

#endif
