// ============================================================================
// greenContextLifecycle.cu — Green Context 创建/销毁开销基准测试
// ============================================================================
//
// 测量创建和销毁 Green Context 分区的开销。
// Green Context 创建涉及：
//   - muDevResourceGenerateDesc    （资源描述符生成）
//   - muGreenCtxCreate             （Green Context 创建）
//   - muCtxFromGreenCtx            （CUcontext 派生）
//   - muGreenCtxStreamCreate       （在 Green Context 中创建 stream）
//   - muGreenCtxGetDevResource     （SM 资源查询）
//   - nop_kernel warmup launch     （强制完成延迟初始化）
//
// 销毁涉及：
//   - muStreamDestroy
//   - muGreenCtxDestroy
//
// 为什么重要：如果 Green Context 创建很昂贵，那么 SM 隔离带来的
// 延迟改善必须超过其设置成本。本 benchmark 提供数据以做出该权衡决策。
//
// 实验参数：critical SM 数量 = 8 或 16。
// ============================================================================

#include "Celero.h"
#include "UserDefinedMeasurements.h"
#include "greenContextIsolation_common.h"

#include <memory>
#include <vector>

// 20 个样本可给出创建/销毁开销的稳定均值/最小值/最大值。
// 每次样本 1 次迭代——每次迭代内部已运行 10 对创建/销毁
//（参见 measureCreateDestroy）。
static const int SamplesCount = 20;
static const int IterationsCount = 1;

// ============================================================================
// 不支持的构建：回退桩
// ============================================================================
#ifdef GREEN_CONTEXT_UNSUPPORTED_BUILD

// 生成输出 Supported=0 的桩程序。
DECLARE_GREEN_CONTEXT_UNSUPPORTED_MAIN(greenContextLifecycle)

#else

// ============================================================================
// GreenContextLifecycleFixture
// ============================================================================
//
// 用于测量 Green Context 创建/销毁开销的测试 fixture。
// 继承自 GreenContextSmPartitionFixture 的 SM 分区设置。
//
// 与 greenContextIsolation.cu 不同，本 benchmark 不在 setUp() 中
// 创建持久 Green Context。而是在 measureCreateDestroy() 的计时区域
// 内创建和销毁它们。
//
// 每个实验参数的生命周期：
//   setUp()         — 仅划分 SM（不创建 Green Context）
//   [每个样本循环]：
//     BASELINE_F 体：
//       measureCreateDestroy() → 10 对创建/销毁，CPU 计时
//   tearDown()       — （自动生成，空操作，因为我们不持有任何资源）
class GreenContextLifecycleFixture : public GreenContextSmPartitionFixture {
public:
    // -------------------------------------------------------------------
    // 实验参数
    // -------------------------------------------------------------------
    // 测试 critical 分区中 8 和 16 个 SM，以观察创建/销毁开销是否
    // 随分区大小扩展。
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        return {8, 16};
    }

    // -------------------------------------------------------------------
    // setUp — 仅 SM 分区，不创建 Green Context
    // -------------------------------------------------------------------
    // printTooSmallError=false：低于 SM 阈值的设备以 EXIT_WAIVED 静默退出。
    // 与隔离性 benchmark 不同，我们不需要知道原因——生命周期开销不是
    // 主要特性测试。
    void setUp(const TestFixture::ExperimentValue& experimentValue) override {
        setUpSmPartition(experimentValue, /*printTooSmallError=*/false);
    }

    // -------------------------------------------------------------------
    // 指标
    // -------------------------------------------------------------------
    // 输出 CSV 中的列：
    //   create(us)     — 每对创建/销毁的平均时间（µs，CPU 时间）
    //   *CreateTP(s^-1) — 吞吐量：每秒可创建/销毁的对数
    //   critSM         — critical 分区中的 SM 数量
    //   bulkSM         — bulk 分区中的 SM 数量
    //
    // 注意：create(us) 使用 UDMCPUTime（CPU 计时器），而非 UDMGPUTime。
    // Green Context 创建是同步驱动操作——我们测量墙钟时间而非 GPU 时间。
    std::vector<std::shared_ptr<UserDefinedMeasurement>>
        getUserDefinedMeasurements() const override {
        return {m_CreateDestroyUs, m_CreateThroughput,
                m_CriticalSmCount, m_BulkSmCount};
    }

    // -------------------------------------------------------------------
    // measureCreateDestroy
    // -------------------------------------------------------------------
    // 创建并销毁一对 Green Context（critical + bulk），重复 `repeat` 次，
    // 使用 CPU 高精度计时器测量总墙钟时间。
    //
    // 每次迭代的协议：
    //   1. 创建 critical GreenContextBundle（在 m_CriticalSm 上调用 create()）
    //   2. 创建 bulk GreenContextBundle（在 m_BulkSm 上调用 create()）
    //   3. musaDeviceSynchronize() — 排空所有待处理工作
    //   4. 销毁 bulk bundle
    //   5. 销毁 critical bundle
    //
    // 步骤 3 的同步确保每次迭代是自包含的：create() 内部的 nop_kernel
    // warmup 在销毁前完成。
    //
    // 每个 GreenContextBundle 是栈上局部对象，因此其析构函数不会调用
    // destroy()——我们在循环内显式调用 destroy() 以将其包含在计时区域内。
    void measureCreateDestroy() {
        constexpr int repeat = 10;
        CPerfCounter timer;
        timer.Restart();
        for (int i = 0; i < repeat; ++i) {
            GreenContextBundle critical{};
            GreenContextBundle bulk{};
            critical.create(m_Device, m_CriticalSm);
            bulk.create(m_Device, m_BulkSm);
            checkMusaErrors(musaDeviceSynchronize());
            critical.destroy();
            bulk.destroy();
        }
        timer.Stop();

        // 转换为微秒并计算每对平均值。
        const double totalUs = timer.GetElapsedSeconds() * 1000.0 * 1000.0;
        const double perPairUs = totalUs / repeat;
        m_CreateDestroyUs->addValue(perPairUs);

        // 吞吐量 = 1 / 每对秒数 = 1e6 / 每对微秒。
        m_CreateThroughput->addValue(
            perPairUs > 0.0 ? 1000000.0 / perPairUs : 0.0);

        // 记录分区大小以供追踪。
        m_CriticalSmCount->addValue(
            static_cast<int>(m_CriticalSm.sm.smCount));
        m_BulkSmCount->addValue(static_cast<int>(m_BulkSm.sm.smCount));
    }

private:
    // --- 指标收集器 ---
    // m_CreateDestroyUs：每对创建/销毁的平均 CPU 时间（µs）。
    //   MEAN | MIN | MAX — 需要同时观察平均值和最坏情况。
    std::shared_ptr<UDMCPUTime> m_CreateDestroyUs{
        new UDMCPUTime("create(us)",
                       StatsView::MEAN | StatsView::MIN | StatsView::MAX)};
    // m_CreateThroughput：每秒可创建/销毁的对数。* 前缀 = 以吞吐量形式输出。
    std::shared_ptr<UDMThroughPut> m_CreateThroughput{
        new UDMThroughPut("*CreateTP(s^-1)",
                          StatsView::MEAN | StatsView::MIN | StatsView::MAX)};
    // 实验配置（每个实验参数恒定）。
    std::shared_ptr<UDMCount> m_CriticalSmCount{new UDMCount("critSM")};
    std::shared_ptr<UDMCount> m_BulkSmCount{new UDMCount("bulkSM")};
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
// BASELINE：创建/销毁对
// ============================================================================
// 唯一的测试用例：测量创建和销毁一对 critical + bulk Green Context 的
// 开销。
//
// 没有对比用例，因为 Green Context 创建是一个独立操作——没有"不使用
// Green Context"的等效操作（不创建 Green Context 就无法划分 SM）。
// 本 benchmark 仅确定绝对开销。
BASELINE_F(greenContextLifecycle, createDestroyPair,
           GreenContextLifecycleFixture, SamplesCount, IterationsCount) {
    measureCreateDestroy();
}

#endif
