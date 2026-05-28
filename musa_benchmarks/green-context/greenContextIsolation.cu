#include "Celero.h"
#include "UserDefinedMeasurements.h"
#include "greenContextIsolation_common.h"

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

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

#else


class GreenContextLatencyFixture : public TestFixture {
public:
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        return {8, 16};
    }

    void setUp(const TestFixture::ExperimentValue& experimentValue) override {
        checkMuErrors(muInit(0));
        checkMuErrors(muDeviceGet(&m_Device, 0));
        checkMuErrors(muDevicePrimaryCtxRetain(&m_PrimaryCtx, m_Device));
        checkMuErrors(muCtxSetCurrent(m_PrimaryCtx));

        const auto smSplit = split_sm_resources(
            m_Device, static_cast<unsigned int>(experimentValue.Value), true);
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
        set_dynamic_shared_limit(m_SharedBytes);

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

        checkMuErrors(muCtxSetCurrent(m_PrimaryCtx));
        checkMusaErrors(musaStreamCreateWithFlags(&m_PrimaryDelayStream, musaStreamNonBlocking));
        checkMusaErrors(musaStreamCreateWithFlags(&m_PrimaryCriticalStream, musaStreamNonBlocking));

        int* rawHostFlags = nullptr;
        // Mapped host flags are used only to ensure the delay kernel has
        // started on all expected blocks before the critical kernel is timed.
        // They are not part of the measured critical latency interval.
        checkMusaErrors(musaHostAlloc(&rawHostFlags, sizeof(int) * m_TotalSms, musaHostAllocMapped));
        m_HostStartedFlags = rawHostFlags;
        checkMusaErrors(musaHostGetDevicePointer(
            reinterpret_cast<void**>(&m_DeviceStartedFlags), rawHostFlags, 0));

        checkMusaErrors(musaDeviceSynchronize());
    }

    void tearDown() override {
        checkMusaErrors(musaDeviceSynchronize());
        if (m_PrimaryDelayStream != nullptr) {
            checkMusaErrors(musaStreamDestroy(m_PrimaryDelayStream));
            m_PrimaryDelayStream = nullptr;
        }
        if (m_PrimaryCriticalStream != nullptr) {
            checkMusaErrors(musaStreamDestroy(m_PrimaryCriticalStream));
            m_PrimaryCriticalStream = nullptr;
        }
        if (m_HostStartedFlags != nullptr) {
            checkMusaErrors(musaFreeHost(const_cast<int*>(m_HostStartedFlags)));
            m_HostStartedFlags = nullptr;
            m_DeviceStartedFlags = nullptr;
        }
        m_GreenCritical.destroy();
        m_GreenBulk.destroy();
        if (m_PrimaryCtx != nullptr) {
            checkMuErrors(muDevicePrimaryCtxRelease(m_Device));
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
                m_SharedKiB,
                m_ActiveBlocksPerSmCount};
    }

    float measureSoloCritical(MUstream stream, MUcontext ctx) {
        return measureCriticalKernel(stream, ctx, "solo critical");
    }

    // Measure the latency of the critical kernel while another stream/context
    // is already running the SM-occupying delay kernel. Only the critical
    // stream event interval is recorded, so the metric represents critical
    // workload latency under contention, not total test duration.
    float measureContendedCritical(MUstream delayStream, MUcontext delayCtx,
                                   int delayBlocks, MUstream criticalStream,
                                   MUcontext criticalCtx, const char* label) {
        musaStream_t delay = runtime_stream(delayStream);
        clearStartedFlags(delayBlocks);

        checkMuErrors(muCtxSetCurrent(delayCtx));
        sm_occupancy_spin_kernel<<<delayBlocks, m_ThreadsPerBlock, m_SharedBytes, delay>>>(
            kDelaySpinCycles, m_DeviceStartedFlags, 1, m_SharedBytes);
        check_kernel_launch(label);
        if (!waitStartedFlags(delayBlocks, label)) {
            std::exit(EXIT_FAILURE);
        }

        const float ms = measureCriticalKernel(criticalStream, criticalCtx, label);

        checkMuErrors(muCtxSetCurrent(delayCtx));
        checkMusaErrors(musaStreamSynchronize(delay));
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
        m_SharedKiB->addValue(m_SharedBytes / 1024);
        m_ActiveBlocksPerSmCount->addValue(m_ActiveBlocksPerSm);
    }

    MUdevice m_Device{};
    MUcontext m_PrimaryCtx = nullptr;
    MUdevResource m_CriticalSm{};
    MUdevResource m_BulkSm{};
    unsigned int m_TotalSms = 0;
    int m_CriticalBlocks = 0;
    int m_BulkBlocks = 0;
    int m_FullBlocks = 0;
    int m_ThreadsPerBlock = 0;
    int m_SharedBytes = 0;
    int m_ActiveBlocksPerSm = 0;
    musaStream_t m_PrimaryDelayStream = nullptr;
    musaStream_t m_PrimaryCriticalStream = nullptr;
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
    std::shared_ptr<UDMCount> m_SharedKiB{new UDMCount("smem(KiB)")};
    std::shared_ptr<UDMCount> m_ActiveBlocksPerSmCount{new UDMCount("blk/SM")};

private:
    float measureCriticalKernel(MUstream stream, MUcontext ctx, const char* label) {
        musaStream_t runtimeStream = runtime_stream(stream);
        checkMuErrors(muCtxSetCurrent(ctx));

        musaEvent_t start{}, stop{};
        checkMusaErrors(musaEventCreate(&start));
        checkMusaErrors(musaEventCreate(&stop));

        checkMusaErrors(musaEventRecord(start, runtimeStream));
        sm_occupancy_spin_kernel<<<m_CriticalBlocks, m_ThreadsPerBlock, m_SharedBytes, runtimeStream>>>(
            kCriticalSpinCycles, nullptr, 0, m_SharedBytes);
        check_kernel_launch(label);
        checkMusaErrors(musaEventRecord(stop, runtimeStream));
        checkMusaErrors(musaEventSynchronize(stop));

        float ms = 0.0f;
        checkMusaErrors(musaEventElapsedTime(&ms, start, stop));
        checkMusaErrors(musaEventDestroy(start));
        checkMusaErrors(musaEventDestroy(stop));
        return ms;
    }

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


int main(int argc, char** argv) {
    musaDeviceProp prop{};
    checkMusaErrors(musaGetDeviceProperties(&prop, 0));
    console::SetConsoleColor(console::ConsoleColor::Yellow);
    std::cout << "## " << argv[0] << " on:" << prop.name << std::endl;
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

#endif
