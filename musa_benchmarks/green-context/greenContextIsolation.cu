// ============================================================================
// greenContextIsolation.cu — Green Context SM 隔离性基准测试
// ============================================================================
//
// 测量 Green Context 分区是否真的在 SM 层面隔离了关键负载，使其不受
// 后台干扰的影响。
//
// 本 benchmark 比较三种场景：
//
//   1. primaryFullContention（BASELINE）：
//      delay 和 critical kernel 均在 primary context 中运行。
//      delay 使用全部 SM。这是最坏情况——critical kernel 无处可跑，
//      只能等待 delay block 完成。
//
//   2. primaryBulkOnly：
//      两个 kernel 都在 primary context 中，但 delay 仅使用与 bulk 分区
//      相同数量的 block。这用于将"后台工作量减少"的效果与 Green Context
//      分区本身的效果区分开。
//
//   3. greenPartitioned（TARGET）：
//      delay 在 bulk Green Context 中运行，critical 在 critical Green
//      Context 中运行。如果 Green Context 工作正常，critical kernel 的
//      延迟应接近其独立运行的延迟，与 bulk 分区在做什么无关。
//
// 指标：隔离分数 = solo_latency / contended_latency
//   - ~1.0 → 完美隔离（关键负载不受后台干扰）
//   - << 1.0 → 存在干扰（后台工作拖慢了关键 kernel）
//
// 实验参数：critical SM 数量 = 8 或 16。
// ============================================================================

#include "Celero.h"
#include "UserDefinedMeasurements.h"
#include "greenContextIsolation_common.h"

#include <memory>
#include <vector>

// 每个样本运行一次完整的 measureContendedCritical 流程（launch delay、
// 等待所有 block 就位、测量 critical）。20 个样本可给出统计上有意义的
// 均值/最小值/最大值。每次样本 1 次迭代，避免对不同后台负载状态取平均。
static const int SamplesCount = 20;
static const int IterationsCount = 1;

// ============================================================================
// 不支持的构建：回退桩
// ============================================================================
#ifdef GREEN_CONTEXT_UNSUPPORTED_BUILD

// 生成输出 Supported=0 的桩程序。参见 greenContextIsolation_common.h
// 中的宏定义。
DECLARE_GREEN_CONTEXT_UNSUPPORTED_MAIN(greenContextLatency)

#else

// ============================================================================
// GreenContextLatencyFixture
// ============================================================================
//
// 用于在各种争用场景下测量 critical kernel 延迟的测试 fixture。
// 继承自基类的 SM 分区设置。
//
// 每个实验参数（8 或 16 个 critical SM）的生命周期：
//   setUp()         — 划分 SM、创建 Green Context、分配 flag
//   onExperimentStart() → （继承，空操作）
//   [每个样本循环]：
//     setUp() 已完成
//     BASELINE_F / BENCHMARK_F 体：
//       measureSoloCritical()          → 独立延迟
//       measureContendedCritical()     → 争用下的延迟
//       recordLatency(solo, contended) → 写入指标
//   [样本结束]
//   onExperimentEnd() → （继承，空操作）
//   tearDown()        — 销毁 stream、释放 flag、销毁 Green Context
class GreenContextLatencyFixture : public GreenContextSmPartitionFixture {
public:
    // -------------------------------------------------------------------
    // 实验参数
    // -------------------------------------------------------------------
    // 测试 8 和 16 个 critical SM。bulk 分区获得剩余 SM
    //（总数 - critical - 粒度开销）。
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        return {8, 16};
    }

    // -------------------------------------------------------------------
    // setUp — 每个实验参数的一次性初始化
    // -------------------------------------------------------------------
    void setUp(const TestFixture::ExperimentValue& experimentValue) override {
        // 步骤 1：SM 分区（来自基类）。printTooSmallError=true，
        // 因为隔离性是主 benchmark——我们希望知道被跳过原因。
        setUpSmPartition(experimentValue, /*printTooSmallError=*/true);

        // 步骤 2：验证划分产生了可用分区。
        // 如果 setUpSmPartition 成功，这里应该不会失败，但防御性
        // 检查可以避免令人困惑的下游错误。
        if (m_CriticalBlocks <= 0 || m_BulkBlocks <= 0) {
            std::cerr << "invalid Green Context SM split: critical="
                      << m_CriticalBlocks << ", bulk=" << m_BulkBlocks << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // 步骤 3：将自旋 kernel 配置为使用我们选择的 shared memory 大小。
        // 必须在任何 kernel launch 之前完成，使 occupancy 反映实际资源使用量。
        set_dynamic_shared_limit(m_SharedBytes);

        // 步骤 4：确保在创建 Green Context 之前处于 primary context
        //（muGreenCtxCreate 可能改变当前 context）。
        checkMuErrors(muCtxSetCurrent(m_PrimaryCtx));

        // 步骤 5：为 critical 和 bulk 域创建 Green Context 分区。
        // 每次创建包含一次 nop_kernel warmup，以强制完成延迟初始化。
        m_GreenCritical.create(m_Device, m_CriticalSm);
        m_GreenBulk.create(m_Device, m_BulkSm);
        if (m_GreenCritical.smCount != m_CriticalSm.sm.smCount ||
            m_GreenBulk.smCount != m_BulkSm.sm.smCount) {
            std::cerr << "Green Context provisioned SM count does not match split result"
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // 步骤 6：为基线场景（场景 1 和 2）创建 primary-context stream。
        // 非阻塞 stream 允许 delay 和 critical kernel 在不同 stream 上
        // 并发运行。
        checkMuErrors(muCtxSetCurrent(m_PrimaryCtx));
        checkMusaErrors(musaStreamCreateWithFlags(
            &m_PrimaryDelayStream, musaStreamNonBlocking));
        checkMusaErrors(musaStreamCreateWithFlags(
            &m_PrimaryCriticalStream, musaStreamNonBlocking));

        // 步骤 7：分配 mapped host memory 用于 started-flags 同步机制。
        // musaHostAllocMapped 让 CPU 和 GPU 都能访问同一物理页面——
        // GPU 写入 flag，CPU 轮询。
        //
        // 每个 SM 一个 flag（m_TotalSms），使每个 delay block 可以
        // 独立发出信号。
        int* rawHostFlags = nullptr;
        checkMusaErrors(musaHostAlloc(
            &rawHostFlags, sizeof(int) * m_TotalSms, musaHostAllocMapped));
        m_HostStartedFlags = rawHostFlags;
        checkMusaErrors(musaHostGetDevicePointer(
            reinterpret_cast<void**>(&m_DeviceStartedFlags), rawHostFlags, 0));

        // 步骤 8：同步，确保所有设置工作已完成。
        checkMusaErrors(musaDeviceSynchronize());
    }

    // -------------------------------------------------------------------
    // tearDown — 按 setUp 的逆序清理
    // -------------------------------------------------------------------
    void tearDown() override {
        // 在销毁资源前排空所有待处理工作。
        checkMusaErrors(musaDeviceSynchronize());

        // 销毁 primary-context stream（基线场景）。
        if (m_PrimaryDelayStream != nullptr) {
            checkMusaErrors(musaStreamDestroy(m_PrimaryDelayStream));
            m_PrimaryDelayStream = nullptr;
        }
        if (m_PrimaryCriticalStream != nullptr) {
            checkMusaErrors(musaStreamDestroy(m_PrimaryCriticalStream));
            m_PrimaryCriticalStream = nullptr;
        }

        // 释放 mapped host memory（同时使 m_DeviceStartedFlags 失效）。
        if (m_HostStartedFlags != nullptr) {
            checkMusaErrors(musaFreeHost(const_cast<int*>(m_HostStartedFlags)));
            m_HostStartedFlags = nullptr;
            m_DeviceStartedFlags = nullptr;
        }

        // 销毁 Green Context 分区（同时销毁内部 stream）。
        m_GreenCritical.destroy();
        m_GreenBulk.destroy();

        // 释放 primary context（来自基类）。
        tearDownSmPartition();
    }

    // -------------------------------------------------------------------
    // 指标
    // -------------------------------------------------------------------
    // 输出 CSV 中的列：
    //   solo(ms)   — 无后台负载时 critical kernel 的延迟
    //   crit(ms)   — 争用下 critical kernel 的延迟
    //   *Iso       — 隔离分数 = solo / crit（1.0 = 完美隔离）
    //   critSM     — critical 分区中的 SM 数量
    //   bulkSM     — bulk 分区中的 SM 数量
    //   block      — 每 block 线程数（来自 choose_launch_config）
    //   smem(KiB)  — 每 block 动态 shared memory（KiB）
    //   blk/SM     — 每 SM 活跃 block 数（理想为 1）
    std::vector<std::shared_ptr<UserDefinedMeasurement>>
        getUserDefinedMeasurements() const override {
        return {m_SoloMs, m_CriticalMs, m_IsolationScore,
                m_CriticalSmCount, m_BulkSmCount,
                m_BlockSize, m_SharedKiB, m_ActiveBlocksPerSmCount};
    }

    // -------------------------------------------------------------------
    // measureSoloCritical
    // -------------------------------------------------------------------
    // 测量 critical kernel 在无任何后台负载时的执行时间。
    // 这是隔离分数进行比较的基线延迟。
    //
    // 委托给共享的 measure_critical_kernel_latency 辅助函数。
    float measureSoloCritical(MUstream stream, MUcontext ctx) {
        return measure_critical_kernel_latency(
            stream, ctx, m_CriticalBlocks, m_ThreadsPerBlock,
            m_SharedBytes, "solo critical");
    }

    // -------------------------------------------------------------------
    // measureContendedCritical
    // -------------------------------------------------------------------
    // 在 delay（后台）kernel 已占用 SM 的情况下测量 critical kernel 的
    // 执行时间。
    //
    // 协议：
    //   1. 清零 started-flags 数组（mapped host memory）。
    //   2. 在 delayStream/delayCtx 上 launch delay kernel，使用
    //      delayBlocks 个 block。每个 block 在 syncthreads 后写入 1
    //      到其 flag，然后进入自旋循环。
    //   3. 轮询 started-flags，直到所有 delay block 已启动。
    //      这确保 delay kernel 完全占满了其 SM。
    //   4. 在 criticalStream/criticalCtx 上 launch 并计时 critical kernel。
    //   5. 同步 delay stream（清理）。
    //
    // critical kernel 在 launch 时不等待 delay kernel 完成——
    // 这就是我们在测量的争用场景。
    //
    // 参数：
    //   delayStream / delayCtx      — 后台负载的运行位置
    //   delayBlocks                 — 后台负载的 block 数量
    //   criticalStream / criticalCtx — 关键工作的运行位置
    //   label                       — 错误信息的诊断标签
    //
    // 返回 critical kernel 的 GPU 经过时间（毫秒）。
    float measureContendedCritical(MUstream delayStream, MUcontext delayCtx,
                                   int delayBlocks, MUstream criticalStream,
                                   MUcontext criticalCtx, const char* label) {
        musaStream_t delay = runtime_stream(delayStream);

        // launch 前重置 flag，以便检测所有 block 何时启动。
        clear_started_flags(m_HostStartedFlags, delayBlocks);

        // Launch delay kernel。每个 block：触碰 smem → syncthreads →
        // 写入 flag → 自旋 kDelaySpinCycles 周期。
        checkMuErrors(muCtxSetCurrent(delayCtx));
        sm_occupancy_spin_kernel<<<delayBlocks, m_ThreadsPerBlock,
                                   m_SharedBytes, delay>>>(
            kDelaySpinCycles, m_DeviceStartedFlags, 1, m_SharedBytes);
        check_kernel_launch(label);

        // 忙等，直到所有 delay block 已进入自旋循环。
        // 关键：如果不等待，critical kernel 可能在 delay kernel
        // 占满所有 SM 之前运行，导致虚假的乐观隔离分数。
        if (!wait_started_flags(m_HostStartedFlags, delayBlocks, label)) {
            std::exit(EXIT_FAILURE);
        }

        // 现在测量争用下的 critical kernel。
        const float ms = measure_critical_kernel_latency(
            criticalStream, criticalCtx, m_CriticalBlocks,
            m_ThreadsPerBlock, m_SharedBytes, label);

        // 清理：在返回前等待 delay kernel 完成。
        checkMuErrors(muCtxSetCurrent(delayCtx));
        checkMusaErrors(musaStreamSynchronize(delay));
        return ms;
    }

    // -------------------------------------------------------------------
    // recordLatency
    // -------------------------------------------------------------------
    // 将一个样本的独立和争用延迟写入指标收集器。
    // 同时记录实验配置（SM 数量、block 大小等），以便在输出 CSV 中
    // 追踪。
    void recordLatency(float soloMs, float criticalMs) {
        m_SoloMs->addValue(soloMs);
        m_CriticalMs->addValue(criticalMs);
        // 隔离分数 = solo / contended。
        //   = 1.0：后台负载零影响（完美隔离）
        //   < 1.0：后台负载拖慢了 critical kernel
        // 防止除零（实际中不应发生）。
        m_IsolationScore->addValue(
            criticalMs > 0.0f ? soloMs / criticalMs : 0.0f);
        m_CriticalSmCount->addValue(m_CriticalBlocks);
        m_BulkSmCount->addValue(m_BulkBlocks);
        m_BlockSize->addValue(m_ThreadsPerBlock);
        m_SharedKiB->addValue(m_SharedBytes / 1024);
        m_ActiveBlocksPerSmCount->addValue(m_ActiveBlocksPerSm);
    }

protected:
    // --- Primary-context 资源（基线场景 1 和 2 使用） ---
    musaStream_t m_PrimaryDelayStream = nullptr;
    musaStream_t m_PrimaryCriticalStream = nullptr;

    // --- 同步 flag（mapped host memory） ---
    // m_HostStartedFlags：CPU 端指针，用于轮询。
    // m_DeviceStartedFlags：GPU 端指针，传递给 delay kernel。
    volatile int* m_HostStartedFlags = nullptr;
    int* m_DeviceStartedFlags = nullptr;

    // --- Green Context 分区 ---
    GreenContextBundle m_GreenCritical{};
    GreenContextBundle m_GreenBulk{};

    // --- 指标收集器 ---
    // m_SoloMs：无后台负载时 critical kernel 的延迟。
    //   MEAN | MIN | MAX — 需要同时观察平均值和最坏情况。
    std::shared_ptr<UDMGPUTime> m_SoloMs{
        new UDMGPUTime("solo(ms)", StatsView::MEAN | StatsView::MIN | StatsView::MAX)};
    // m_CriticalMs：争用下 critical kernel 的延迟。
    std::shared_ptr<UDMGPUTime> m_CriticalMs{
        new UDMGPUTime("crit(ms)", StatsView::MEAN | StatsView::MIN | StatsView::MAX)};
    // m_IsolationScore：solo / crit。* 前缀 = 以比率形式输出。
    std::shared_ptr<UDMRatio> m_IsolationScore{
        new UDMRatio("*Iso", StatsView::MEAN | StatsView::MIN | StatsView::MAX)};
    // 实验配置（每个实验参数恒定，记录以在输出 CSV 中追踪）。
    std::shared_ptr<UDMCount> m_CriticalSmCount{new UDMCount("critSM")};
    std::shared_ptr<UDMCount> m_BulkSmCount{new UDMCount("bulkSM")};
    std::shared_ptr<UDMCount> m_BlockSize{new UDMCount("block")};
    std::shared_ptr<UDMCount> m_SharedKiB{new UDMCount("smem(KiB)")};
    std::shared_ptr<UDMCount> m_ActiveBlocksPerSmCount{new UDMCount("blk/SM")};
};

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    musaDeviceProp prop{};
    checkMusaErrors(musaGetDeviceProperties(&prop, 0));
    console::SetConsoleColor(console::ConsoleColor::Yellow);
    std::cout << "## " << argv[0] << " on:" << prop.name << std::endl;
    Printer::get().TableSetPbName("criticalSM");
    Run(argc, argv);
    return 0;
}

// ============================================================================
// 场景 1：BASELINE — Primary Context，完全争用
// ============================================================================
// delay 和 critical kernel 都在 primary context 中，在独立的非阻塞
// stream 上运行。delay 使用全部 SM（m_FullBlocks）。
//
// 预期：最差隔离分数。delay kernel 占满所有 SM，因此 critical kernel
// 只有在某些 delay block 完成后才能开始。这确定了隔离的下限。
BASELINE_F(greenContextLatency, primaryFullContention,
           GreenContextLatencyFixture, SamplesCount, IterationsCount) {
    const float soloMs = measureSoloCritical(
        driver_stream(m_PrimaryCriticalStream), m_PrimaryCtx);
    const float criticalMs = measureContendedCritical(
        driver_stream(m_PrimaryDelayStream), m_PrimaryCtx, m_FullBlocks,
        driver_stream(m_PrimaryCriticalStream), m_PrimaryCtx,
        "primary full contention");
    recordLatency(soloMs, criticalMs);
}

// ============================================================================
// 场景 2：Primary Context，仅 Bulk 争用
// ============================================================================
// 两个 kernel 都在 primary context 中，但 delay 仅使用 m_BulkBlocks
//（与 bulk Green Context 会获得的 SM 数量相同）。
//
// 目的：将"后台 SM 更少"的效果与"Green Context SM 隔离"的效果
// 区分开。如果这里的隔离分数已经很好，那么 Green Context 并不是
// 区分因素——仅靠减少争用 SM 数量就足够了。
BENCHMARK_F(greenContextLatency, primaryBulkOnly,
            GreenContextLatencyFixture, SamplesCount, IterationsCount) {
    const float soloMs = measureSoloCritical(
        driver_stream(m_PrimaryCriticalStream), m_PrimaryCtx);
    const float criticalMs = measureContendedCritical(
        driver_stream(m_PrimaryDelayStream), m_PrimaryCtx, m_BulkBlocks,
        driver_stream(m_PrimaryCriticalStream), m_PrimaryCtx,
        "primary bulk-only contention");
    recordLatency(soloMs, criticalMs);
}

// ============================================================================
// 场景 3：Green Context 分区（目标场景）
// ============================================================================
// delay 在 bulk Green Context 中运行，critical 在 critical Green
// Context 中运行。每个 Green Context 有自己的 CUcontext 和 SM 分区。
//
// 预期：如果 Green Context 工作正常，critical kernel 的延迟应接近其
// 独立运行延迟，与 delay kernel 的活动无关。隔离分数应接近 1.0。
//
// 这是 Green Context 特性的关键指标——它验证 SM 级别隔离是否真的
// 阻止了干扰。
BENCHMARK_F(greenContextLatency, greenPartitioned,
            GreenContextLatencyFixture, SamplesCount, IterationsCount) {
    const float soloMs = measureSoloCritical(
        m_GreenCritical.stream, m_GreenCritical.ctx);
    const float criticalMs = measureContendedCritical(
        m_GreenBulk.stream, m_GreenBulk.ctx, m_BulkBlocks,
        m_GreenCritical.stream, m_GreenCritical.ctx, "green partitioned");
    recordLatency(soloMs, criticalMs);
}

#endif
