#include "Celero.h"
#include "UserDefinedMeasurements.h"
#include "greenContextIsolation_common.h"
#include "timer_his.h"

#include <memory>
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
    std::shared_ptr<UDMCount> m_Supported{new UDMCount(*Supported)};
};

int main(int argc, char** argv) {
    std::cout << CUDA Green Context APIs are not available in this CUDA header set.
              <<  Build with CUDA 12.4 or newer to enable the full benchmark.
              << std::endl;
    Printer::get().TableSetPbName(criticalSM);
    Run(argc, argv);
    return 0;
}

BASELINE_F(greenContextLifecycle, unsupportedCudaHeader,
           UnsupportedGreenContextFixture, 1, 1) {
    recordUnsupported();
}

#else


class GreenContextLifecycleFixture : public TestFixture {
public:
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        return {8, 16};
    }

    void setUp(const TestFixture::ExperimentValue& experimentValue) override {
        checkMuErrors(muInit(0));
        checkMuErrors(muDeviceGet(&m_Device, 0));
        checkMuErrors(muDeviceGetDevResource(m_Device, &m_AllSm, MU_DEV_RESOURCE_TYPE_SM));
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
        checkMuErrors(muDevSmResourceSplitByCount(
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
            checkMusaErrors(musaDeviceSynchronize());
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
    MUdevice m_Device{};
    MUdevResource m_AllSm{};
    MUdevResource m_CriticalSm{};
    MUdevResource m_BulkSm{};
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
    musaDeviceProp prop{};
    checkMusaErrors(musaGetDeviceProperties(&prop, 0));
    console::SetConsoleColor(console::ConsoleColor::Yellow);
    std::cout << "## " << argv[0] << " on:" << prop.name << std::endl;
    Printer::get().TableSetPbName("criticalSM");
    Run(argc, argv);
    return 0;
}

BASELINE_F(greenContextLifecycle, createDestroyPair,
           GreenContextLifecycleFixture, SamplesCount, IterationsCount) {
    // Lifecycle guard: Green Context creation is not free. This case tracks
    // create/destroy cost so latency gains are evaluated together with setup
    // overhead.
    measureCreateDestroy();
}

#endif
