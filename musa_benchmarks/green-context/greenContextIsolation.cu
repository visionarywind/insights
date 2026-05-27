#include "Celero.h"
#include "UserDefinedMeasurements.h"

#ifdef TEST_ON_NVIDIA
#include <cuda.h>
#include <cuda_runtime.h>
#if !defined(CUDA_VERSION) || CUDA_VERSION < 12040
#define GREEN_CONTEXT_UNSUPPORTED_BUILD 1
#endif
#else
#include "musa.h"
#include "musa_runtime.h"
#endif

#include "timer_his.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#ifdef TEST_ON_NVIDIA
#define GPU_NOINLINE __noinline__
#else
#define GPU_NOINLINE __attribute__((noinline))
#endif

static const int SamplesCount = 20;
static const int IterationsCount = 1;

#if defined(GREEN_CONTEXT_UNSUPPORTED_BUILD)

class UnsupportedGreenContextFixture : public TestFixture {
public:
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        return {0};
    }

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {m_Supported};
    }

    void recordUnsupported() {
        m_Supported->addValue(0);
    }

private:
    std::shared_ptr<UDMCount> m_Supported{new UDMCount("*Supported")};
};

int main(int argc, char** argv) {
    std::cout << "CUDA Green Context APIs are not available in this CUDA header set."
              << " Build with CUDA 12.4 or newer to enable the full benchmark."
              << std::endl;
    Printer::get().TableSetPbName("criticalSM");
    Run(argc, argv);
    return 0;
}

BASELINE_F(greenContextLatency, unsupportedCudaHeader,
           UnsupportedGreenContextFixture, 1, 1) {
    recordUnsupported();
}

BASELINE_F(greenContextLifecycle, unsupportedCudaHeader,
           UnsupportedGreenContextFixture, 1, 1) {
    recordUnsupported();
}

#else

#ifdef TEST_ON_NVIDIA
using GpuError = cudaError_t;
using GpuStream = cudaStream_t;
using GpuEvent = cudaEvent_t;
using GpuDeviceProp = cudaDeviceProp;
using DrvResult = CUresult;
using DrvDevice = CUdevice;
using DrvContext = CUcontext;
using DrvStream = CUstream;
using DrvGreenContext = CUgreenCtx;
using DrvDevResource = CUdevResource;
using DrvDevResourceDesc = CUdevResourceDesc;

#define GPU_BACKEND_NAME "CUDA"
#define GPU_SUCCESS cudaSuccess
#define GPU_STREAM_NON_BLOCKING cudaStreamNonBlocking
#define GPU_HOST_ALLOC_MAPPED cudaHostAllocMapped
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuGetDeviceProperties cudaGetDeviceProperties
#define gpuGetLastError cudaGetLastError
#define gpuGetErrorString cudaGetErrorString
#define gpuStreamCreateWithFlags cudaStreamCreateWithFlags
#define gpuStreamDestroy cudaStreamDestroy
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuHostAlloc cudaHostAlloc
#define gpuHostGetDevicePointer cudaHostGetDevicePointer
#define gpuFreeHost cudaFreeHost
#define gpuEventCreate cudaEventCreate
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventElapsedTime cudaEventElapsedTime
#define gpuEventDestroy cudaEventDestroy
#define gpuOccupancyMaxActiveBlocksPerMultiprocessor cudaOccupancyMaxActiveBlocksPerMultiprocessor

#define DRV_SUCCESS CUDA_SUCCESS
#define DRV_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
#define DRV_DEV_RESOURCE_TYPE_SM CU_DEV_RESOURCE_TYPE_SM
#define DRV_GREEN_CTX_DEFAULT_STREAM CU_GREEN_CTX_DEFAULT_STREAM
#define DRV_STREAM_NON_BLOCKING CU_STREAM_NON_BLOCKING
#define drvGetErrorString cuGetErrorString
#define drvInit cuInit
#define drvDeviceGet cuDeviceGet
#define drvDeviceGetAttribute cuDeviceGetAttribute
#define drvDevicePrimaryCtxRetain cuDevicePrimaryCtxRetain
#define drvDevicePrimaryCtxRelease cuDevicePrimaryCtxRelease
#define drvCtxSetCurrent cuCtxSetCurrent
#define drvDeviceGetDevResource cuDeviceGetDevResource
#define drvDevSmResourceSplitByCount cuDevSmResourceSplitByCount
#define drvDevResourceGenerateDesc cuDevResourceGenerateDesc
#define drvGreenCtxCreate cuGreenCtxCreate
#define drvGreenCtxDestroy cuGreenCtxDestroy
#define drvCtxFromGreenCtx cuCtxFromGreenCtx
#define drvGreenCtxGetDevResource cuGreenCtxGetDevResource
#define drvGreenCtxStreamCreate cuGreenCtxStreamCreate
#define drvStreamDestroy cuStreamDestroy
#else
using GpuError = musaError_t;
using GpuStream = musaStream_t;
using GpuEvent = musaEvent_t;
using GpuDeviceProp = musaDeviceProp;
using DrvResult = MUresult;
using DrvDevice = MUdevice;
using DrvContext = MUcontext;
using DrvStream = MUstream;
using DrvGreenContext = MUgreenCtx;
using DrvDevResource = MUdevResource;
using DrvDevResourceDesc = MUdevResourceDesc;

#define GPU_BACKEND_NAME "MUSA"
#define GPU_SUCCESS musaSuccess
#define GPU_STREAM_NON_BLOCKING musaStreamNonBlocking
#define GPU_HOST_ALLOC_MAPPED musaHostAllocMapped
#define gpuGetDeviceCount musaGetDeviceCount
#define gpuGetDeviceProperties musaGetDeviceProperties
#define gpuGetLastError musaGetLastError
#define gpuGetErrorString musaGetErrorString
#define gpuStreamCreateWithFlags musaStreamCreateWithFlags
#define gpuStreamDestroy musaStreamDestroy
#define gpuStreamSynchronize musaStreamSynchronize
#define gpuDeviceSynchronize musaDeviceSynchronize
#define gpuHostAlloc musaHostAlloc
#define gpuHostGetDevicePointer musaHostGetDevicePointer
#define gpuFreeHost musaFreeHost
#define gpuEventCreate musaEventCreate
#define gpuEventRecord musaEventRecord
#define gpuEventSynchronize musaEventSynchronize
#define gpuEventElapsedTime musaEventElapsedTime
#define gpuEventDestroy musaEventDestroy
#define gpuOccupancyMaxActiveBlocksPerMultiprocessor musaOccupancyMaxActiveBlocksPerMultiprocessor

#define DRV_SUCCESS MUSA_SUCCESS
#define DRV_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK MU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
#define DRV_DEV_RESOURCE_TYPE_SM MU_DEV_RESOURCE_TYPE_SM
#define DRV_GREEN_CTX_DEFAULT_STREAM MU_GREEN_CTX_DEFAULT_STREAM
#define DRV_STREAM_NON_BLOCKING MU_STREAM_NON_BLOCKING
#define drvGetErrorString muGetErrorString
#define drvInit muInit
#define drvDeviceGet muDeviceGet
#define drvDeviceGetAttribute muDeviceGetAttribute
#define drvDevicePrimaryCtxRetain muDevicePrimaryCtxRetain
#define drvDevicePrimaryCtxRelease muDevicePrimaryCtxRelease
#define drvCtxSetCurrent muCtxSetCurrent
#define drvDeviceGetDevResource muDeviceGetDevResource
#define drvDevSmResourceSplitByCount muDevSmResourceSplitByCount
#define drvDevResourceGenerateDesc muDevResourceGenerateDesc
#define drvGreenCtxCreate muGreenCtxCreate
#define drvGreenCtxDestroy muGreenCtxDestroy
#define drvCtxFromGreenCtx muCtxFromGreenCtx
#define drvGreenCtxGetDevResource muGreenCtxGetDevResource
#define drvGreenCtxStreamCreate muGreenCtxStreamCreate
#define drvStreamDestroy muStreamDestroy
#endif

static void check_gpu_errors(GpuError err, const char* file, int line) {
    if (err != GPU_SUCCESS) {
        std::cerr << "Runtime API error at " << file << ":" << line
                  << " code=" << static_cast<int>(err)
                  << " message=" << gpuGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static void check_drv_errors(DrvResult err, const char* file, int line) {
    if (err != DRV_SUCCESS) {
        const char* errorStr = nullptr;
        drvGetErrorString(err, &errorStr);
        std::cerr << "Driver API error at " << file << ":" << line
                  << " code=" << static_cast<int>(err)
                  << " message=" << (errorStr ? errorStr : "<unknown>")
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_GPU(err) check_gpu_errors((err), __FILE__, __LINE__)
#define CHECK_DRV(err) check_drv_errors((err), __FILE__, __LINE__)

constexpr unsigned long long kDelaySpinCycles = 25000000ULL;
constexpr unsigned long long kCriticalSpinCycles = 6000000ULL;

constexpr int kBlockSharedBytes = 48 * 1024 - 16;

__global__ void nop_kernel() {}

// The benchmark needs a background kernel that really holds SM resources.
// Large static shared memory makes each block consume almost one whole SM, so
// block count is close to "occupied SM count". This makes the experiment easy
// to interpret: N launched blocks roughly means N SMs are occupied.
__device__ GPU_NOINLINE unsigned char touch_static_smem(volatile unsigned char* smem) {
    unsigned char acc = 0;
    for (int i = threadIdx.x; i < kBlockSharedBytes; i += blockDim.x) {
        smem[i] = static_cast<unsigned char>(i);
        acc |= smem[i];
    }
    return acc;
}

__global__ void sm_occupancy_spin_kernel(unsigned long long spinCycles,
                                         int* startedFlags, int marker) {
    __shared__ unsigned char smem[kBlockSharedBytes];
    const unsigned char smemTag =
        touch_static_smem(reinterpret_cast<volatile unsigned char*>(smem));
    __syncthreads();

    if (threadIdx.x == 0 && startedFlags != nullptr) {
        startedFlags[blockIdx.x] = smemTag ? marker : marker;
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

// MUSA 5.1 exposes only smCount in MUdevSmResource. CUDA headers expose Green
// Context APIs with the same split call. Use a conservative architecture-level
// granularity so the same source works on both backends.
static unsigned int sm_partition_granularity(const DrvDevResource& res) {
    return res.sm.smCount >= 80 ? 8 : 2;
}

static GpuStream runtime_stream(DrvStream stream) {
    return reinterpret_cast<GpuStream>(stream);
}

static DrvStream driver_stream(GpuStream stream) {
    return reinterpret_cast<DrvStream>(stream);
}

static void check_kernel_launch(const char* label) {
    GpuError err = gpuGetLastError();
    if (err != GPU_SUCCESS) {
        std::cerr << "kernel launch failed at " << label << ": "
                  << gpuGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

struct BlockConfig {
    int threadsPerBlock = 0;
    int activeBlocksPerSm = 0;
};

// Pick a launch block size from the current device instead of hard-coding it.
// The benchmark prefers one active block per SM because that creates clearer
// SM occupancy pressure. If no candidate reaches one block per SM, use the
// candidate with the smallest active block count.
static BlockConfig choose_block_config(DrvDevice dev) {
    int maxThreadsPerBlock = 0;
    CHECK_DRV(drvDeviceGetAttribute(
        &maxThreadsPerBlock, DRV_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev));

    const int candidates[] = {1024, 768, 512, 384, 256, 128, 64};
    BlockConfig best{};
    int bestActiveBlocks = 1 << 30;

    for (int candidate : candidates) {
        if (candidate > maxThreadsPerBlock) {
            continue;
        }

        int activeBlocks = 0;
        CHECK_GPU(gpuOccupancyMaxActiveBlocksPerMultiprocessor(
            &activeBlocks, sm_occupancy_spin_kernel, candidate, 0));

        if (activeBlocks == 1) {
            return {candidate, activeBlocks};
        }

        if (activeBlocks > 0 &&
            (activeBlocks < bestActiveBlocks ||
             (activeBlocks == bestActiveBlocks && candidate > best.threadsPerBlock))) {
            best = {candidate, activeBlocks};
            bestActiveBlocks = activeBlocks;
        }
    }

    if (best.threadsPerBlock == 0) {
        std::cerr << "cannot find a valid block size for occupancy query" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return best;
}

struct GreenContextBundle {
    DrvGreenContext greenCtx = nullptr;
    DrvContext ctx = nullptr;
    DrvStream stream = nullptr;
    DrvDevResource smResource{};
    unsigned int smCount = 0;

    void create(DrvDevice dev, const DrvDevResource& smRes) {
        // Driver API flow:
        // 1. convert SM resource to descriptor;
        // 2. create Green Context;
        // 3. derive executable context;
        // 4. create stream bound to this Green Context.
        DrvDevResourceDesc desc{};
        DrvDevResource mutableRes = smRes;
        CHECK_DRV(drvDevResourceGenerateDesc(&desc, &mutableRes, 1));
        CHECK_DRV(drvGreenCtxCreate(&greenCtx, desc, dev, DRV_GREEN_CTX_DEFAULT_STREAM));
        CHECK_DRV(drvCtxFromGreenCtx(&ctx, greenCtx));
        CHECK_DRV(drvCtxSetCurrent(ctx));
        CHECK_DRV(drvGreenCtxStreamCreate(&stream, greenCtx, DRV_STREAM_NON_BLOCKING, 0));
        CHECK_DRV(drvGreenCtxGetDevResource(greenCtx, &smResource, DRV_DEV_RESOURCE_TYPE_SM));
        smCount = smResource.sm.smCount;

        nop_kernel<<<1, 1, 0, runtime_stream(stream)>>>();
        check_kernel_launch("green context warmup");
        CHECK_GPU(gpuStreamSynchronize(runtime_stream(stream)));
    }

    void destroy() {
        if (stream != nullptr) {
            CHECK_DRV(drvStreamDestroy(stream));
        }
        if (greenCtx != nullptr) {
            CHECK_DRV(drvGreenCtxDestroy(greenCtx));
        }
        stream = nullptr;
        greenCtx = nullptr;
        ctx = nullptr;
        smCount = 0;
        smResource = {};
    }
};

class GreenContextLatencyFixture : public TestFixture {
public:
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        return {8, 16};
    }

    void setUp(const TestFixture::ExperimentValue& experimentValue) override {
        CHECK_DRV(drvInit(0));
        CHECK_DRV(drvDeviceGet(&m_Device, 0));
        CHECK_DRV(drvDevicePrimaryCtxRetain(&m_PrimaryCtx, m_Device));
        CHECK_DRV(drvCtxSetCurrent(m_PrimaryCtx));

        CHECK_DRV(drvDeviceGetDevResource(m_Device, &m_AllSm, DRV_DEV_RESOURCE_TYPE_SM));
        m_TotalSms = m_AllSm.sm.smCount;
        m_Granularity = sm_partition_granularity(m_AllSm);
        if (m_TotalSms < m_Granularity * 2) {
            std::cerr << "Green Context benchmark requires at least "
                      << (m_Granularity * 2) << " SMs, got " << m_TotalSms << std::endl;
            std::exit(EXIT_WAIVED);
        }

        m_CriticalTarget = align_up(static_cast<unsigned int>(experimentValue.Value), m_Granularity);
        if (m_CriticalTarget + m_Granularity > m_TotalSms) {
            m_CriticalTarget = m_Granularity;
        }

        unsigned int nbGroups = 1;
        // Split all SMs into two resources:
        // - critical resource: latency-sensitive stream;
        // - bulk resource: background load stream.
        CHECK_DRV(drvDevSmResourceSplitByCount(
            &m_CriticalSm, &nbGroups, &m_AllSm, &m_BulkSm, 0, m_CriticalTarget));
        m_CriticalBlocks = static_cast<int>(m_CriticalSm.sm.smCount);
        m_BulkBlocks = static_cast<int>(m_BulkSm.sm.smCount);
        m_FullBlocks = static_cast<int>(m_TotalSms);
        const BlockConfig blockConfig = choose_block_config(m_Device);
        m_ThreadsPerBlock = blockConfig.threadsPerBlock;
        m_ActiveBlocksPerSm = blockConfig.activeBlocksPerSm;

        if (m_CriticalBlocks <= 0 || m_BulkBlocks <= 0) {
            std::cerr << "invalid Green Context SM split: critical="
                      << m_CriticalBlocks << ", bulk=" << m_BulkBlocks << std::endl;
            std::exit(EXIT_FAILURE);
        }

        m_GreenCritical.create(m_Device, m_CriticalSm);
        m_GreenBulk.create(m_Device, m_BulkSm);
        if (m_GreenCritical.smCount != m_CriticalSm.sm.smCount ||
            m_GreenBulk.smCount != m_BulkSm.sm.smCount) {
            std::cerr << "Green Context provisioned SM count does not match split result" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        CHECK_DRV(drvCtxSetCurrent(m_PrimaryCtx));
        CHECK_GPU(gpuStreamCreateWithFlags(&m_PrimaryDelayStream, GPU_STREAM_NON_BLOCKING));
        CHECK_GPU(gpuStreamCreateWithFlags(&m_PrimaryCriticalStream, GPU_STREAM_NON_BLOCKING));

        int* rawHostFlags = nullptr;
        // Mapped host flags are used only to ensure the delay kernel has
        // started on all expected blocks before the critical kernel is timed.
        // They are not part of the measured critical latency interval.
        CHECK_GPU(gpuHostAlloc(&rawHostFlags, sizeof(int) * m_TotalSms, GPU_HOST_ALLOC_MAPPED));
        m_HostStartedFlags = rawHostFlags;
        CHECK_GPU(gpuHostGetDevicePointer(
            reinterpret_cast<void**>(&m_DeviceStartedFlags), rawHostFlags, 0));

        CHECK_GPU(gpuDeviceSynchronize());
    }

    void tearDown() override {
        CHECK_GPU(gpuDeviceSynchronize());
        if (m_PrimaryDelayStream != nullptr) {
            CHECK_GPU(gpuStreamDestroy(m_PrimaryDelayStream));
            m_PrimaryDelayStream = nullptr;
        }
        if (m_PrimaryCriticalStream != nullptr) {
            CHECK_GPU(gpuStreamDestroy(m_PrimaryCriticalStream));
            m_PrimaryCriticalStream = nullptr;
        }
        if (m_HostStartedFlags != nullptr) {
            CHECK_GPU(gpuFreeHost(const_cast<int*>(m_HostStartedFlags)));
            m_HostStartedFlags = nullptr;
            m_DeviceStartedFlags = nullptr;
        }
        m_GreenCritical.destroy();
        m_GreenBulk.destroy();
        if (m_PrimaryCtx != nullptr) {
            CHECK_DRV(drvDevicePrimaryCtxRelease(m_Device));
            m_PrimaryCtx = nullptr;
        }
    }

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {m_SoloMs,
                m_CriticalMs,
                m_IsolationScore,
                m_CriticalSmCount,
                m_BulkSmCount,
                m_BlockSize,
                m_ActiveBlocksPerSmCount};
    }

    float measureSoloCritical(DrvStream stream, DrvContext ctx) {
        GpuStream gpuStream = runtime_stream(stream);
        CHECK_DRV(drvCtxSetCurrent(ctx));

        GpuEvent start{}, stop{};
        CHECK_GPU(gpuEventCreate(&start));
        CHECK_GPU(gpuEventCreate(&stop));
        clearStartedFlags(std::max(m_CriticalBlocks, 1));

        CHECK_GPU(gpuEventRecord(start, gpuStream));
        sm_occupancy_spin_kernel<<<m_CriticalBlocks, m_ThreadsPerBlock, 0, gpuStream>>>(
            kCriticalSpinCycles, m_DeviceStartedFlags, 2);
        check_kernel_launch("solo critical");
        CHECK_GPU(gpuEventRecord(stop, gpuStream));
        CHECK_GPU(gpuEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_GPU(gpuEventElapsedTime(&ms, start, stop));
        CHECK_GPU(gpuEventDestroy(start));
        CHECK_GPU(gpuEventDestroy(stop));
        return ms;
    }

    // Measure the latency of the critical kernel while another stream/context
    // is already running the SM-occupying delay kernel. Only the critical
    // stream event interval is recorded, so the metric represents critical
    // workload latency under contention, not total test duration.
    float measureContendedCritical(DrvStream delayStream, DrvContext delayCtx,
                                   int delayBlocks, DrvStream criticalStream,
                                   DrvContext criticalCtx, const char* label) {
        GpuStream delay = runtime_stream(delayStream);
        GpuStream critical = runtime_stream(criticalStream);
        clearStartedFlags(delayBlocks);

        CHECK_DRV(drvCtxSetCurrent(criticalCtx));
        GpuEvent criticalStart{}, criticalStop{};
        CHECK_GPU(gpuEventCreate(&criticalStart));
        CHECK_GPU(gpuEventCreate(&criticalStop));

        CHECK_DRV(drvCtxSetCurrent(delayCtx));
        sm_occupancy_spin_kernel<<<delayBlocks, m_ThreadsPerBlock, 0, delay>>>(
            kDelaySpinCycles, m_DeviceStartedFlags, 1);
        check_kernel_launch(label);
        if (!waitStartedFlags(delayBlocks, label)) {
            std::exit(EXIT_FAILURE);
        }

        CHECK_DRV(drvCtxSetCurrent(criticalCtx));
        CHECK_GPU(gpuEventRecord(criticalStart, critical));
        sm_occupancy_spin_kernel<<<m_CriticalBlocks, m_ThreadsPerBlock, 0, critical>>>(
            kCriticalSpinCycles, m_DeviceStartedFlags, 2);
        check_kernel_launch(label);
        CHECK_GPU(gpuEventRecord(criticalStop, critical));
        CHECK_GPU(gpuEventSynchronize(criticalStop));

        float ms = 0.0f;
        CHECK_GPU(gpuEventElapsedTime(&ms, criticalStart, criticalStop));
        CHECK_GPU(gpuEventDestroy(criticalStart));
        CHECK_GPU(gpuEventDestroy(criticalStop));

        CHECK_DRV(drvCtxSetCurrent(delayCtx));
        CHECK_GPU(gpuStreamSynchronize(delay));
        return ms;
    }

    void recordLatency(float soloMs, float criticalMs) {
        m_SoloMs->addValue(soloMs);
        m_CriticalMs->addValue(criticalMs);
        // Isolation score = solo latency / contended latency.
        // A value close to 1 means the critical workload keeps its solo
        // latency under background load. Lower values indicate interference.
        m_IsolationScore->addValue(criticalMs > 0.0f ? soloMs / criticalMs : 0.0f);
        m_CriticalSmCount->addValue(m_CriticalBlocks);
        m_BulkSmCount->addValue(m_BulkBlocks);
        m_BlockSize->addValue(m_ThreadsPerBlock);
        m_ActiveBlocksPerSmCount->addValue(m_ActiveBlocksPerSm);
    }

    DrvDevice m_Device{};
    DrvContext m_PrimaryCtx = nullptr;
    DrvDevResource m_AllSm{};
    DrvDevResource m_CriticalSm{};
    DrvDevResource m_BulkSm{};
    unsigned int m_TotalSms = 0;
    unsigned int m_Granularity = 0;
    unsigned int m_CriticalTarget = 0;
    int m_CriticalBlocks = 0;
    int m_BulkBlocks = 0;
    int m_FullBlocks = 0;
    int m_ThreadsPerBlock = 0;
    int m_ActiveBlocksPerSm = 0;
    GpuStream m_PrimaryDelayStream = nullptr;
    GpuStream m_PrimaryCriticalStream = nullptr;
    volatile int* m_HostStartedFlags = nullptr;
    int* m_DeviceStartedFlags = nullptr;
    GreenContextBundle m_GreenCritical{};
    GreenContextBundle m_GreenBulk{};

    std::shared_ptr<UDMGPUTime> m_SoloMs{
        new UDMGPUTime("solo(ms)", StatsView::MEAN | StatsView::MIN | StatsView::MAX)};
    std::shared_ptr<UDMGPUTime> m_CriticalMs{
        new UDMGPUTime("crit(ms)", StatsView::MEAN | StatsView::MIN | StatsView::MAX)};
    std::shared_ptr<UDMRatio> m_IsolationScore{
        new UDMRatio("*Iso", StatsView::MEAN | StatsView::MIN | StatsView::MAX)};
    std::shared_ptr<UDMCount> m_CriticalSmCount{new UDMCount("critSM")};
    std::shared_ptr<UDMCount> m_BulkSmCount{new UDMCount("bulkSM")};
    std::shared_ptr<UDMCount> m_BlockSize{new UDMCount("block")};
    std::shared_ptr<UDMCount> m_ActiveBlocksPerSmCount{new UDMCount("blk/SM")};

private:
    void clearStartedFlags(int count) {
        for (int i = 0; i < count; ++i) {
            m_HostStartedFlags[i] = 0;
        }
    }

    bool waitStartedFlags(int count, const char* label) {
        for (int retry = 0; retry < 100000; ++retry) {
            int seen = 0;
            for (int i = 0; i < count; ++i) {
                seen += (m_HostStartedFlags[i] != 0) ? 1 : 0;
            }
            if (seen == count) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
        std::cerr << "delay kernel did not occupy all expected SM blocks in "
                  << label << ", expected blocks=" << count << std::endl;
        return false;
    }
};

class GreenContextLifecycleFixture : public TestFixture {
public:
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        return {8, 16};
    }

    void setUp(const TestFixture::ExperimentValue& experimentValue) override {
        CHECK_DRV(drvInit(0));
        CHECK_DRV(drvDeviceGet(&m_Device, 0));
        CHECK_DRV(drvDeviceGetDevResource(m_Device, &m_AllSm, DRV_DEV_RESOURCE_TYPE_SM));
        m_TotalSms = m_AllSm.sm.smCount;
        m_Granularity = sm_partition_granularity(m_AllSm);
        if (m_TotalSms < m_Granularity * 2) {
            std::exit(EXIT_WAIVED);
        }
        m_CriticalTarget = align_up(static_cast<unsigned int>(experimentValue.Value), m_Granularity);
        if (m_CriticalTarget + m_Granularity > m_TotalSms) {
            m_CriticalTarget = m_Granularity;
        }
        unsigned int nbGroups = 1;
        CHECK_DRV(drvDevSmResourceSplitByCount(
            &m_CriticalSm, &nbGroups, &m_AllSm, &m_BulkSm, 0, m_CriticalTarget));
    }

    void tearDown() override {}

    std::vector<std::shared_ptr<UserDefinedMeasurement>> getUserDefinedMeasurements() const override {
        return {m_CreateDestroyUs, m_CreateThroughput, m_CriticalSmCount, m_BulkSmCount};
    }

    void measureCreateDestroy() {
        constexpr int repeat = 10;
        CPerfCounter timer;
        timer.Restart();
        for (int i = 0; i < repeat; ++i) {
            GreenContextBundle critical{};
            GreenContextBundle bulk{};
            critical.create(m_Device, m_CriticalSm);
            bulk.create(m_Device, m_BulkSm);
            CHECK_GPU(gpuDeviceSynchronize());
            critical.destroy();
            bulk.destroy();
        }
        timer.Stop();
        const double totalUs = timer.GetElapsedSeconds() * 1000.0 * 1000.0;
        const double perPairUs = totalUs / repeat;
        m_CreateDestroyUs->addValue(perPairUs);
        m_CreateThroughput->addValue(perPairUs > 0.0 ? 1000000.0 / perPairUs : 0.0);
        m_CriticalSmCount->addValue(static_cast<int>(m_CriticalSm.sm.smCount));
        m_BulkSmCount->addValue(static_cast<int>(m_BulkSm.sm.smCount));
    }

private:
    DrvDevice m_Device{};
    DrvDevResource m_AllSm{};
    DrvDevResource m_CriticalSm{};
    DrvDevResource m_BulkSm{};
    unsigned int m_TotalSms = 0;
    unsigned int m_Granularity = 0;
    unsigned int m_CriticalTarget = 0;

    std::shared_ptr<UDMCPUTime> m_CreateDestroyUs{
        new UDMCPUTime("create(us)", StatsView::MEAN | StatsView::MIN | StatsView::MAX)};
    std::shared_ptr<UDMThroughPut> m_CreateThroughput{
        new UDMThroughPut("*CreateTP(s^-1)", StatsView::MEAN | StatsView::MIN | StatsView::MAX)};
    std::shared_ptr<UDMCount> m_CriticalSmCount{new UDMCount("critSM")};
    std::shared_ptr<UDMCount> m_BulkSmCount{new UDMCount("bulkSM")};
};

int main(int argc, char** argv) {
    int deviceCount = 0;
    CHECK_GPU(gpuGetDeviceCount(&deviceCount));
    GpuDeviceProp prop{};
    CHECK_GPU(gpuGetDeviceProperties(&prop, 0));
    console::SetConsoleColor(console::ConsoleColor::Yellow);
    std::cout << "## " << argv[0] << " on:" << prop.name
              << " backend:" << GPU_BACKEND_NAME << std::endl;
    Printer::get().TableSetPbName("criticalSM");
    Run(argc, argv);
    return 0;
}

BASELINE_F(greenContextLatency, primaryFullContention,
           GreenContextLatencyFixture, SamplesCount, IterationsCount) {
    // Baseline 1: both delay and critical workloads run in the primary
    // context. Delay uses all SMs, representing unpartitioned contention.
    const float soloMs = measureSoloCritical(
        driver_stream(m_PrimaryCriticalStream), m_PrimaryCtx);
    const float criticalMs = measureContendedCritical(
        driver_stream(m_PrimaryDelayStream), m_PrimaryCtx, m_FullBlocks,
        driver_stream(m_PrimaryCriticalStream), m_PrimaryCtx,
        "primary full contention");
    recordLatency(soloMs, criticalMs);
}

BENCHMARK_F(greenContextLatency, primaryBulkOnly,
            GreenContextLatencyFixture, SamplesCount, IterationsCount) {
    // Baseline 2: still primary context only, but delay uses the same number
    // of blocks as the Green Context bulk partition. This separates the effect
    // of "less background work" from the effect of Green Context isolation.
    const float soloMs = measureSoloCritical(
        driver_stream(m_PrimaryCriticalStream), m_PrimaryCtx);
    const float criticalMs = measureContendedCritical(
        driver_stream(m_PrimaryDelayStream), m_PrimaryCtx, m_BulkBlocks,
        driver_stream(m_PrimaryCriticalStream), m_PrimaryCtx,
        "primary bulk-only contention");
    recordLatency(soloMs, criticalMs);
}

BENCHMARK_F(greenContextLatency, greenPartitioned,
            GreenContextLatencyFixture, SamplesCount, IterationsCount) {
    // Target case: delay runs on the bulk Green Context and critical work runs
    // on the critical Green Context. The expected advantage is stable critical
    // latency while background work is active.
    const float soloMs = measureSoloCritical(m_GreenCritical.stream, m_GreenCritical.ctx);
    const float criticalMs = measureContendedCritical(
        m_GreenBulk.stream, m_GreenBulk.ctx, m_BulkBlocks,
        m_GreenCritical.stream, m_GreenCritical.ctx, "green partitioned");
    recordLatency(soloMs, criticalMs);
}

BASELINE_F(greenContextLifecycle, createDestroyPair,
           GreenContextLifecycleFixture, SamplesCount, IterationsCount) {
    // Lifecycle guard: Green Context creation is not free. This case tracks
    // create/destroy cost so latency gains are evaluated together with setup
    // overhead.
    measureCreateDestroy();
}

#endif
