// ============================================================================
// greenContextIsolation_common.h — Green Context 基准测试的共享定义
// ============================================================================
//
// Green Context 是 CUDA 12.4+ 的特性（MUSA 同样支持），允许将 GPU 的 SM
// 划分为隔离的执行域。每个域拥有独立的 CUcontext 和 CUstream，运行在不同
// 域中的 kernel 在 SM 层面保证互不干扰。
//
// 本头文件为两个 Green Context 基准测试提供共享基础设施：
//   - greenContextIsolation.cu  — 测量关键负载在后台干扰下是否保持隔离
//   - greenContextLifecycle.cu  — 测量 Green Context 分区的创建/销毁开销
//
// 内容（按顺序）：
//   1. 构建配置（GREEN_CONTEXT_UNSUPPORTED_BUILD, noinline）
//   2. GPU kernel 定义（nop_kernel, sm_occupancy_spin_kernel）
//   3. SM 资源分区（split_sm_resources）
//   4. 工具函数（stream 类型转换, 错误检查, 对齐）
//   5. launch 配置选择（choose_launch_config）
//   6. GreenContextBundle — Green Context 生命周期的 RAII 封装
//   7. GreenContextSmPartitionFixture — 基准测试 fixture 基类
//   8. 共享测量辅助函数（flag 轮询, kernel 延迟测量）
//   9. DECLARE_GREEN_CONTEXT_UNSUPPORTED_MAIN — 旧 CUDA SDK 的回退方案
// ============================================================================

#pragma once

#include "musa.h"
#include "musa_runtime.h"
#include "helper_musa.h"
#include "helper_musa_drvapi.h"
#include "timer_his.h"

// ============================================================================
// 构建配置
// ============================================================================

// Green Context API（muGreenCtxCreate, muDevSmResourceSplitByCount 等）
// 在 CUDA 12.4 中引入。当构建目标是更旧的 SDK 时，生成一个输出
// "unsupported" 的桩 benchmark，而非编译失败。
#ifdef TEST_ON_NVIDIA
#if !defined(CUDA_VERSION) || CUDA_VERSION < 12040
#define GREEN_CONTEXT_UNSUPPORTED_BUILD 1
#endif
#endif

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

// EXIT_WAIVED (2) 表示"测试不适用于当前硬件"，而非"测试失败"。
// 测试运行器将退出码 2 视为跳过。
#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

// CUDA 的 __noinline__ vs 标准 __attribute__((noinline))。
// 用在 touch_dynamic_smem 上，防止编译器将 shared memory 访问模式
// 优化掉——其核心目的就是通过 shared memory 占用制造 occupancy 压力。
#ifdef TEST_ON_NVIDIA
#define GREEN_CONTEXT_NOINLINE __noinline__
#else
#define GREEN_CONTEXT_NOINLINE __attribute__((noinline))
#endif

// ============================================================================
// 活跃代码路径（仅在非 unsupported 构建时编译）
// ============================================================================
#if !defined(GREEN_CONTEXT_UNSUPPORTED_BUILD)

// --------------------------------------------------------------------------
// 自旋周期常量
// --------------------------------------------------------------------------
// kDelaySpinCycles：后台（delay）kernel 的自旋时长。
//   ~25M 周期 ≈ 在 2.5 GHz 下约 10ms——足够覆盖整个 critical kernel
//   的测量窗口。
// kCriticalSpinCycles：被测（critical）kernel 的自旋时长。
//   ~6M 周期 ≈ 2.4ms——足够短以精确测量延迟，足够长以产生有意义的数值。
constexpr unsigned long long kDelaySpinCycles = 25000000ULL;
constexpr unsigned long long kCriticalSpinCycles = 6000000ULL;

// occupancy 调优的 shared memory 配置。
// kSharedMemoryAlignment：shared memory 请求必须以 256 字节对齐。
// kSharedMemorySearchStep：搜索最小化 occupancy 的配置时，以 4 KiB 步长递减。
constexpr int kSharedMemoryAlignment = 256;
constexpr int kSharedMemorySearchStep = 4 * 1024;

// ============================================================================
// GPU Kernel 定义
// ============================================================================

// --------------------------------------------------------------------------
// nop_kernel
// --------------------------------------------------------------------------
// 用于 Green Context 创建后做 warmup launch 的最小 kernel。
// 确保 Green Context 的 stream 和 context 在计时工作开始前已完全初始化。
// 没有这次 warmup，新 Green Context 上的首次 launch 可能包含延迟初始化开销。
__global__ void nop_kernel() {}

// --------------------------------------------------------------------------
// touch_dynamic_smem
// --------------------------------------------------------------------------
// 用已知模式填充已分配的动态 shared memory 的每个字节，并读回。
// 累加器防止编译器将其优化为空操作。
//
// 为什么重要：仅声明 extern __shared__ 并不会实际预留 SM 的 shared memory——
// 硬件只在 kernel 真正触碰时才分配。我们需要 occupancy 压力是真实的，这样
// occupancy 查询（musaOccupancyMaxActiveBlocksPerMultiprocessor）才能为所选
// shared memory 大小返回正确的 active block 数量。
__device__ GREEN_CONTEXT_NOINLINE unsigned char touch_dynamic_smem(
    volatile unsigned char* smem, int sharedBytes) {
    unsigned char acc = 0;
    for (int i = threadIdx.x; i < sharedBytes; i += blockDim.x) {
        smem[i] = static_cast<unsigned char>(i);
        acc |= smem[i];
    }
    return acc;
}

// --------------------------------------------------------------------------
// sm_occupancy_spin_kernel
// --------------------------------------------------------------------------
// delay（后台）和 critical（被测）两种负载共用的核心 kernel。
//
// 参数：
//   spinCycles  — 忙等的 clock64() 滴答数
//   startedFlags — 若非空，每个 block 在 __syncthreads() 后向
//                  startedFlags[blockIdx.x] 写入标记值。
//                  delay kernel 使用此路径，host 可以验证所有 block
//                  已在计时开始前启动。
//   marker      — 写入 startedFlags 的值（始终为 1）
//   sharedBytes — 要分配并触碰的动态 shared memory 大小
//
// kernel 首先触碰所有已分配动态 shared memory 并同步（确保 SM 实际预留），
// 然后进入自旋。
__global__ void sm_occupancy_spin_kernel(unsigned long long spinCycles,
                                         int* startedFlags, int marker,
                                         int sharedBytes) {
    extern __shared__ unsigned char smem[];
    touch_dynamic_smem(reinterpret_cast<volatile unsigned char*>(smem), sharedBytes);
    __syncthreads();

    // 通知 host 本 block 已启动。仅 delay kernel 使用此路径；
    // critical kernel 传入 startedFlags=nullptr，使 flag 可见性
    // 不计入其测量延迟中。
    if (threadIdx.x == 0 && startedFlags != nullptr) {
        __threadfence_system();
        startedFlags[blockIdx.x] = marker;
        __threadfence_system();
    }

    const unsigned long long start = clock64();
    while (clock64() - start < spinCycles) {
    }
}

// ============================================================================
// SM 资源分区
// ============================================================================

// --------------------------------------------------------------------------
// align_up / align_down
// --------------------------------------------------------------------------
// 整数对齐工具。用于将 SM 数量向上对齐到分区粒度，将 shared memory
// 大小向下对齐到硬件对齐边界。
static unsigned int align_up(unsigned int value, unsigned int alignment) {
    if (alignment == 0) {
        return value;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

static int align_down(int value, int alignment) {
    if (alignment == 0) {
        return value;
    }
    return (value / alignment) * alignment;
}

// --------------------------------------------------------------------------
// sm_partition_granularity
// --------------------------------------------------------------------------
// 返回可分配到一个分区的最小 SM 数量。
// - ≥80 SM（大 GPU）：8 SM 粒度
// - <80 SM（小 GPU）：2 SM 粒度
// 这些值与 CUDA 的 Green Context SM 分区规则一致。
static unsigned int sm_partition_granularity(const MUdevResource& res) {
    return res.sm.smCount >= 80 ? 8 : 2;
}

// --------------------------------------------------------------------------
// SmResourceSplit
// --------------------------------------------------------------------------
// split_sm_resources() 的返回结果。包含完整的设备 SM 资源、
// critical 和 bulk 分区，以及派生的元数据。
struct SmResourceSplit {
    MUdevResource all{};          // 设备所有 SM
    MUdevResource critical{};     // 分配给 critical 域的 SM
    MUdevResource bulk{};         // 分配给 bulk（后台）域的 SM
    unsigned int totalSms = 0;    // SM 总数
    unsigned int granularity = 0; // 分区粒度
    unsigned int criticalTarget = 0; // 请求的 critical SM 数量（已对齐）
};

// --------------------------------------------------------------------------
// split_sm_resources
// --------------------------------------------------------------------------
// 将设备的 SM 划分为 critical 和 bulk 两个分区。
//
// 参数：
//   dev                   — MUSA 设备句柄
//   requestedCriticalSms  — 期望的 critical 域 SM 数量
//   printTooSmallError    — 若为 true，在设备 SM 太少时打印错误信息后再退出
//
// 行为：
//   1. 通过 muDeviceGetDevResource 查询 SM 总数。
//   2. 若总数 < granularity * 2，以 EXIT_WAIVED 退出（测试不适用——
//      至少需要 2 个分区）。
//   3. 将 requestedCriticalSms 向上对齐到粒度。
//   4. 若对齐值 + 粒度 > 总数，回退到粒度值。
//   5. 调用 muDevSmResourceSplitByCount 执行实际分区。
//
// 返回包含两个分区的 SmResourceSplit。
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

// ============================================================================
// Stream / Context 工具函数
// ============================================================================

// --------------------------------------------------------------------------
// runtime_stream / driver_stream
// --------------------------------------------------------------------------
// MUSA 的 Runtime API 和 Driver API 使用不同的 stream 句柄类型
//（musaStream_t vs MUstream），但它们是二进制兼容的指针。
// 这些转换辅助函数使代码可以在两个 API 层之间传递同一个 stream，
// 而无需在代码各处散布不安全的 reinterpret_cast。
static musaStream_t runtime_stream(MUstream stream) {
    return reinterpret_cast<musaStream_t>(stream);
}

static MUstream driver_stream(musaStream_t stream) {
    return reinterpret_cast<MUstream>(stream);
}

// --------------------------------------------------------------------------
// check_kernel_launch
// --------------------------------------------------------------------------
// 通过调用 musaGetLastError() 检查 <<<...>>> kernel launch 的结果。
// 这是捕获异步 launch 错误（非法 grid/block 维度、资源不足等）的标准模式。
static void check_kernel_launch(const char* label) {
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        std::cerr << "kernel launch failed at " << label << ": "
                  << musaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// ============================================================================
// Launch 配置选择
// ============================================================================

// --------------------------------------------------------------------------
// LaunchConfig
// --------------------------------------------------------------------------
// choose_launch_config() 的返回结果。描述一个在最大化 shared memory 使用
// 的同时，使每个 SM 上活跃 block 数最小（理想为 1）的 kernel launch 配置。
struct LaunchConfig {
    int threadsPerBlock = 0;
    int sharedBytes = 0;
    int activeBlocksPerSm = 0;
};

// --------------------------------------------------------------------------
// max_dynamic_shared_bytes
// --------------------------------------------------------------------------
// 返回每个 block 的最大动态 shared memory。同时考虑默认限制
//（sharedMemPerBlock）和可选限制（sharedMemPerBlockOptin）。
// 也被 sharedMemPerMultiprocessor（每个 SM 的总 shared memory，
// 当目标为 1 block/SM 时限制了每 block 的用量）所限制。
// 结果向下对齐到 kSharedMemoryAlignment。
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

// --------------------------------------------------------------------------
// set_dynamic_shared_limit
// --------------------------------------------------------------------------
// 配置 sm_occupancy_spin_kernel 请求指定数量的动态 shared memory。
// 必须在 kernel launch 之前调用，使 occupancy 查询反映实际的
// shared memory 使用量。
static void set_dynamic_shared_limit(int sharedBytes) {
    checkMusaErrors(musaFuncSetAttribute(
        sm_occupancy_spin_kernel, musaFuncAttributeMaxDynamicSharedMemorySize,
        sharedBytes));
}

// --------------------------------------------------------------------------
// choose_launch_config
// --------------------------------------------------------------------------
// 搜索一个 launch 配置（每 block 线程数, shared memory 大小），使
// sm_occupancy_spin_kernel 在每个 SM 上的活跃 block 数最小。
//
// 为什么要最小化 occupancy：基准测试需要每个 SM 只运行一个 block（或
// 尽可能少），以制造最大的争用压力。当 delay kernel 的所有 block 占满
// 全部 SM 时，critical kernel 除了自己的 Green Context 分区外无处可运行。
//
// 算法：
//   遍历每种 shared memory 大小（最大 → 最小，步长 kSharedMemorySearchStep）：
//     遍历每种线程数候选值（1024 → 64）：
//       查询 musaOccupancyMaxActiveBlocksPerMultiprocessor。
//       若 activeBlocks == 1 → 立即返回（最优解）。
//       否则跟踪最佳配置（最少活跃 block 数，平局时优先更大的 shared memory，
//       再优先更多的线程）。
//
// 返回找到的最佳 LaunchConfig。失败时退出。
static LaunchConfig choose_launch_config(MUdevice dev) {
    int maxThreadsPerBlock = 0;
    checkMuErrors(muDeviceGetAttribute(
        &maxThreadsPerBlock, MU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev));

    const int threadCandidates[] = {1024, 768, 512, 384, 256, 128, 64};
    const int maxSharedBytes = max_dynamic_shared_bytes();
    LaunchConfig best{};
    int bestActiveBlocks = 1 << 30; // 哨兵值：大于任何实际值

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

            // 1 block/SM 是理想情况：最大的 SM 占用压力。
            if (activeBlocks == 1) {
                return {threads, alignedSharedBytes, activeBlocks};
            }

            // 跟踪当前找到的最佳配置。优先级：
            //   1. 每个 SM 活跃 block 更少（压力更大）
            //   2. 更多 shared memory（资源占用更多）
            //   3. 更多线程（warp 占用更多）
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

// ============================================================================
// GreenContextBundle — Green Context 生命周期的 RAII 封装
// ============================================================================
//
// 封装一个 Green Context 分区：Green Context 本身、派生出的 CUcontext、
// 一个非阻塞 stream，以及分配给它的 SM 资源。
//
// 用法：
//   GreenContextBundle bundle;
//   bundle.initialize(device, smResource);  // 初始化 Green Context + stream + warmup
//   // ... 在 bundle.ctx / bundle.stream 上发起工作 ...
//   bundle.deinitialize();                  // 先销毁 stream，再销毁 Green Context
//
// warmup（单次 nop_kernel launch）确保延迟初始化在计时区域外完成。
struct GreenContextBundle {
    MUgreenCtx greenCtx = nullptr;    // Green Context 句柄
    MUcontext ctx = nullptr;          // 从 greenCtx 派生的 CUcontext
    MUstream stream = nullptr;        // 该 context 中的非阻塞 stream
    MUdevResource smResource{};       // 分配给该分区的 SM 资源
    unsigned int smCount = 0;         // 实际 SM 数量（创建后从驱动获取）

    void initialize(MUdevice dev, const MUdevResource& smRes) {
        // 步骤 1：从 SM 资源生成资源描述符
        MUdevResourceDesc desc{};
        MUdevResource mutableRes = smRes;
        checkMuErrors(muDevResourceGenerateDesc(&desc, &mutableRes, 1));

        // 步骤 2：使用 SM 分区创建 Green Context
        checkMuErrors(muGreenCtxCreate(&greenCtx, desc, dev, MU_GREEN_CTX_DEFAULT_STREAM));

        // 步骤 3：从 Green Context 派生常规 CUcontext
        checkMuErrors(muCtxFromGreenCtx(&ctx, greenCtx));
        checkMuErrors(muCtxSetCurrent(ctx));

        // 步骤 4：在该 context 中创建非阻塞 stream
        checkMuErrors(muGreenCtxStreamCreate(&stream, greenCtx, MU_STREAM_NON_BLOCKING, 0));

        // 步骤 5：查询实际分配的 SM 资源（应与 smRes 一致）
        checkMuErrors(muGreenCtxGetDevResource(greenCtx, &smResource, MU_DEV_RESOURCE_TYPE_SM));
        smCount = smResource.sm.smCount;

        // 步骤 6：warmup——强制完成延迟初始化
        nop_kernel<<<1, 1, 0, runtime_stream(stream)>>>();
        check_kernel_launch("green context warmup");
        checkMusaErrors(musaStreamSynchronize(runtime_stream(stream)));
    }

    void deinitialize() {
        if (stream != nullptr) {
            checkMuErrors(muStreamDestroy(stream));
        }
        if (ctx != nullptr) {
            MUcontext current = nullptr;
            checkMuErrors(muCtxGetCurrent(&current));
            if (current == ctx) {
                checkMuErrors(muCtxSetCurrent(nullptr));
            }
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

// ============================================================================
// 基础 Fixture：SM 分区设置
// ============================================================================
//
// GreenContextSmPartitionFixture 处理两个 Green Context 基准测试共享的
// 通用设置：
//   - muInit + 设备查询
//   - Primary context 保留（用于非 Green Context 的基线测试）
//   - SM 资源划分为 critical + bulk 分区
//   - Launch 配置选择（线程数、shared memory、活跃 block 数）
//
// 子类在 setUp() 开头调用 setUpSmPartition()，在 tearDown() 末尾调用
// tearDownSmPartition()。然后可以添加各自特定的设置（Green Context 创建、
// stream 创建、flag 分配）。
class GreenContextSmPartitionFixture : public TestFixture {
protected:
    // -----------------------------------------------------------------------
    // setUpSmPartition
    // -----------------------------------------------------------------------
    // 初始化驱动、查询设备 0、保留 primary context、划分 SM 资源，
    // 并选择最优 launch 配置。
    //
    // 参数：
    //   experimentValue    — 来自 Celero 的 getExperimentValues()；
    //                        .Value 为请求的 critical SM 数量（8 或 16）
    //   printTooSmallError — 转发给 split_sm_resources
    void setUpSmPartition(const TestFixture::ExperimentValue& experimentValue,
                          bool printTooSmallError) {
        checkMuErrors(muInit(0));
        checkMuErrors(muDeviceGet(&m_Device, 0));
        checkMuErrors(muDevicePrimaryCtxRetain(&m_PrimaryCtx, m_Device));
        checkMuErrors(muCtxSetCurrent(m_PrimaryCtx));
        const auto smSplit = split_sm_resources(
            m_Device, static_cast<unsigned int>(experimentValue.Value),
            printTooSmallError);
        m_TotalSms = smSplit.totalSms;
        m_CriticalSm = smSplit.critical;
        m_BulkSm = smSplit.bulk;
        m_CriticalBlocks = static_cast<int>(m_CriticalSm.sm.smCount);
        m_BulkBlocks = static_cast<int>(m_BulkSm.sm.smCount);
        m_FullBlocks = static_cast<int>(m_TotalSms);
        const LaunchConfig launchConfig = choose_launch_config(m_Device);
        m_ThreadsPerBlock = launchConfig.threadsPerBlock;
        m_SharedBytes = launchConfig.sharedBytes;
        m_ActiveBlocksPerSm = launchConfig.activeBlocksPerSm;
    }

    // -----------------------------------------------------------------------
    // tearDownSmPartition
    // -----------------------------------------------------------------------
    // 若 primary context 已被保留，则释放之。子类应在 tearDown() 末尾、
    // 销毁所有特定资源后调用此方法。
    void tearDownSmPartition() {
        if (m_PrimaryCtx != nullptr) {
            checkMuErrors(muDevicePrimaryCtxRelease(m_Device));
            m_PrimaryCtx = nullptr;
        }
    }

    // --- 由 setUpSmPartition 填充的成员 ---
    MUdevice m_Device{};              // 设备 0 的句柄
    MUcontext m_PrimaryCtx = nullptr; // primary context（已保留）
    MUdevResource m_CriticalSm{};     // critical 分区的 SM 资源
    MUdevResource m_BulkSm{};         // bulk 分区的 SM 资源
    unsigned int m_TotalSms = 0;      // 设备 SM 总数
    int m_CriticalBlocks = 0;         // critical 分区 block 数（= SM 数）
    int m_BulkBlocks = 0;             // bulk 分区 block 数
    int m_FullBlocks = 0;             // 全设备（未分区）情况下的 block 数
    int m_ThreadsPerBlock = 0;        // 来自 choose_launch_config
    int m_SharedBytes = 0;            // 每 block 动态 shared memory
    int m_ActiveBlocksPerSm = 0;      // 每 SM 活跃 block 数（理想为 1）
};

// ============================================================================
// 共享测量辅助函数
// ============================================================================

// --------------------------------------------------------------------------
// clear_started_flags
// --------------------------------------------------------------------------
// 在 launch delay kernel 之前将 host 端的 started-flags 数组清零。
// flag 位于 mapped host memory（musaHostAllocMapped）中，GPU kernel
// 可以直接写入。
static void clear_started_flags(volatile int* hostFlags, int count) {
    for (int i = 0; i < count; ++i) {
        hostFlags[i] = 0;
    }
}

// --------------------------------------------------------------------------
// wait_started_flags
// --------------------------------------------------------------------------
// 轮询 host 端 started-flags 数组，直到所有 `count` 个条目均为非零值，
// 表示 delay kernel 的所有 block 都已进入自旋循环。
// 使用 10 µs 轮询间隔，最多重试 100k 次（总计约 1 秒超时）。
//
// 这确保了 delay kernel 已在所有分配的 SM 上就位后，才 launch critical
// kernel 并开始计时。没有这个屏障，critical kernel 可能在 delay kernel
// 完全部署前就开始运行，导致虚假的乐观隔离分数。
static bool wait_started_flags(volatile int* hostFlags, int count, const char* label) {
    for (int retry = 0; retry < 100000; ++retry) {
        int seen = 0;
        for (int i = 0; i < count; ++i) {
            seen += (hostFlags[i] != 0) ? 1 : 0;
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

// --------------------------------------------------------------------------
// measure_critical_kernel_latency
// --------------------------------------------------------------------------
// 在给定 stream/context 上 launch critical（自旋）kernel，使用
// musaEventElapsedTime 测量其 GPU 执行时间。
//
// kernel 以 startedFlags=nullptr 启动——跳过了 sm_occupancy_spin_kernel
// 内部的 flag 写入路径，因此测量到的延迟仅反映自旋循环本身加上 launch 开销。
//
// 参数：
//   stream          — launch 目标 stream
//   ctx             — launch 前要切换到的 context
//   criticalBlocks  — grid 大小（每个 critical SM 一个 block）
//   threadsPerBlock — 来自 choose_launch_config
//   sharedBytes     — 每 block 动态 shared memory
//   label           — 错误信息中的诊断标签
//
// 返回 GPU 经过时间，单位为毫秒。
static float measure_critical_kernel_latency(MUstream stream, MUcontext ctx,
                                              int criticalBlocks, int threadsPerBlock,
                                              int sharedBytes, const char* label) {
    musaStream_t runtimeStream = runtime_stream(stream);
    checkMuErrors(muCtxSetCurrent(ctx));
    musaEvent_t start{}, stop{};
    checkMusaErrors(musaEventCreate(&start));
    checkMusaErrors(musaEventCreate(&stop));
    checkMusaErrors(musaEventRecord(start, runtimeStream));
    sm_occupancy_spin_kernel<<<criticalBlocks, threadsPerBlock, sharedBytes, runtimeStream>>>(
        kCriticalSpinCycles, nullptr, 0, sharedBytes);
    check_kernel_launch(label);
    checkMusaErrors(musaEventRecord(stop, runtimeStream));
    checkMusaErrors(musaEventSynchronize(stop));
    float ms = 0.0f;
    checkMusaErrors(musaEventElapsedTime(&ms, start, stop));
    checkMusaErrors(musaEventDestroy(start));
    checkMusaErrors(musaEventDestroy(stop));
    return ms;
}

#endif  // !GREEN_CONTEXT_UNSUPPORTED_BUILD

// ============================================================================
// 回退方案：不支持的 CUDA Header
// ============================================================================
//
// 当构建目标是 CUDA 12.4 之前的 SDK（或没有 Green Context 支持的 MUSA SDK）
// 时，Green Context API 不可用。与其编译失败，不如让每个 benchmark 生成一个
// 桩可执行文件，在输出中报告 Supported=0。这样测试运行器可以优雅地跳过
// 基准测试，而不是崩溃。
//
// 用法：在每个 benchmark 的 .cu 文件顶部，GREEN_CONTEXT_UNSUPPORTED_BUILD
// 检查之后：
//
//   DECLARE_GREEN_CONTEXT_UNSUPPORTED_MAIN(greenContextLatency)
//
// 这会生成一个最小的 main() + Fixture + BASELINE_F，所有指标记录为
// Supported=0。
#ifdef GREEN_CONTEXT_UNSUPPORTED_BUILD

#define DECLARE_GREEN_CONTEXT_UNSUPPORTED_MAIN(suiteName)                              \
class UnsupportedGreenContextFixture : public TestFixture {                            \
public:                                                                                \
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {   \
        return {0};                                                                    \
    }                                                                                  \
    std::vector<std::shared_ptr<UserDefinedMeasurement>>                               \
        getUserDefinedMeasurements() const override {                                  \
        return {m_Supported};                                                          \
    }                                                                                  \
    void recordUnsupported() { m_Supported->addValue(0); }                             \
private:                                                                               \
    std::shared_ptr<UDMCount> m_Supported{new UDMCount("*Supported")};                 \
};                                                                                     \
int main(int argc, char** argv) {                                                      \
    std::cout << "CUDA Green Context APIs are not available in this CUDA header set." \
              << " Build with CUDA 12.4 or newer to enable the full benchmark."       \
              << std::endl;                                                            \
    Printer::get().TableSetPbName("criticalSM");                                       \
    Run(argc, argv);                                                                   \
    return 0;                                                                          \
}                                                                                      \
BASELINE_F(suiteName, unsupportedCudaHeader,                                           \
           UnsupportedGreenContextFixture, 1, 1) {                                     \
    recordUnsupported();                                                               \
}

#endif  // GREEN_CONTEXT_UNSUPPORTED_BUILD
