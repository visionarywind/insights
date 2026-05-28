// ============================================================================
// greenContextLifecycle.cu — Green Context Create/Destroy Overhead Benchmark
// ============================================================================
//
// Measures the cost of creating and destroying Green Context partitions.
// Green Context creation involves:
//   - muDevResourceGenerateDesc    (resource descriptor generation)
//   - muGreenCtxCreate             (Green Context creation)
//   - muCtxFromGreenCtx            (CUcontext derivation)
//   - muGreenCtxStreamCreate       (stream creation in Green Context)
//   - muGreenCtxGetDevResource     (SM resource query)
//   - nop_kernel warmup launch     (force lazy initialization)
//
// And destruction:
//   - muStreamDestroy
//   - muGreenCtxDestroy
//
// Why this matters: if Green Context creation is expensive, the latency
// improvement from SM isolation must outweigh the setup cost. This
// benchmark provides the data to make that tradeoff decision.
//
// Experiment values: critical SM count = 8 or 16.
// ============================================================================

#include "Celero.h"
#include "UserDefinedMeasurements.h"
#include "greenContextIsolation_common.h"

#include <memory>
#include <vector>

// 20 samples give a stable mean/min/max for the create/destroy cost.
// 1 iteration per sample — each iteration already runs 10 create/destroy
// pairs internally (see measureCreateDestroy).
static const int SamplesCount = 20;
static const int IterationsCount = 1;

// ============================================================================
// Unsupported Build Fallback
// ============================================================================
#ifdef GREEN_CONTEXT_UNSUPPORTED_BUILD

// Generates a stub that reports Supported=0.
DECLARE_GREEN_CONTEXT_UNSUPPORTED_MAIN(greenContextLifecycle)

#else

// ============================================================================
// GreenContextLifecycleFixture
// ============================================================================
//
// Test fixture for measuring Green Context create/destroy overhead.
// Inherits SM partition setup from GreenContextSmPartitionFixture.
//
// Unlike greenContextIsolation.cu, this benchmark does NOT create
// persistent Green Contexts in setUp(). Instead, it creates and
// destroys them inside the timed region of measureCreateDestroy().
//
// Lifecycle per experiment value:
//   setUp()         — partition SMs only (no Green Context creation)
//   [for each sample]:
//     BASELINE_F body:
//       measureCreateDestroy() → 10 create/destroy pairs, CPU timed
//   tearDown()       — (auto-generated, no-op since we hold no resources)
class GreenContextLifecycleFixture : public GreenContextSmPartitionFixture {
public:
    // -------------------------------------------------------------------
    // Experiment Values
    // -------------------------------------------------------------------
    // Tests with 8 and 16 SMs in the critical partition to see if
    // create/destroy cost scales with partition size.
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        return {8, 16};
    }

    // -------------------------------------------------------------------
    // setUp — SM partition only, no Green Context creation
    // -------------------------------------------------------------------
    // printTooSmallError=false: devices below the SM threshold exit
    // silently with EXIT_WAIVED. Unlike the isolation benchmark, we
    // don't need to know why — lifecycle overhead is not the primary
    // feature test.
    void setUp(const TestFixture::ExperimentValue& experimentValue) override {
        setUpSmPartition(experimentValue, /*printTooSmallError=*/false);
    }

    // -------------------------------------------------------------------
    // Metrics
    // -------------------------------------------------------------------
    // Reported columns in output CSV:
    //   create(us)     — average time per create/destroy pair in µs (CPU time)
    //   *CreateTP(s^-1) — throughput: pairs per second
    //   critSM         — number of SMs in critical partition
    //   bulkSM         — number of SMs in bulk partition
    //
    // Note: create(us) uses UDMCPUTime (CPU timer), NOT UDMGPUTime.
    // Green Context creation is a synchronous driver operation — we
    // measure wall-clock time, not GPU time.
    std::vector<std::shared_ptr<UserDefinedMeasurement>>
        getUserDefinedMeasurements() const override {
        return {m_CreateDestroyUs, m_CreateThroughput,
                m_CriticalSmCount, m_BulkSmCount};
    }

    // -------------------------------------------------------------------
    // measureCreateDestroy
    // -------------------------------------------------------------------
    // Creates and destroys a pair of Green Contexts (critical + bulk)
    // `repeat` times, measuring the total wall-clock time with a CPU
    // high-resolution timer.
    //
    // Protocol per iteration:
    //   1. Create critical GreenContextBundle (calls create() on m_CriticalSm)
    //   2. Create bulk GreenContextBundle (calls create() on m_BulkSm)
    //   3. musaDeviceSynchronize() — drain any pending work
    //   4. Destroy bulk bundle
    //   5. Destroy critical bundle
    //
    // The synchronize in step 3 ensures each iteration is self-contained:
    // the nop_kernel warmup inside create() completes before we destroy.
    //
    // Each GreenContextBundle is a stack-local object, so its destructor
    // does NOT run destroy() — we call destroy() explicitly inside the
    // loop to include it in the timed region.
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

        // Convert to microseconds and compute per-pair average.
        const double totalUs = timer.GetElapsedSeconds() * 1000.0 * 1000.0;
        const double perPairUs = totalUs / repeat;
        m_CreateDestroyUs->addValue(perPairUs);

        // Throughput = 1 / per-pair-seconds = 1e6 / per-pair-µs.
        m_CreateThroughput->addValue(
            perPairUs > 0.0 ? 1000000.0 / perPairUs : 0.0);

        // Record the partition sizes for traceability.
        m_CriticalSmCount->addValue(
            static_cast<int>(m_CriticalSm.sm.smCount));
        m_BulkSmCount->addValue(static_cast<int>(m_BulkSm.sm.smCount));
    }

private:
    // --- Metrics collectors ---
    // m_CreateDestroyUs: average CPU time per create/destroy pair in µs.
    //   MEAN | MIN | MAX — want to see both average and worst-case.
    std::shared_ptr<UDMCPUTime> m_CreateDestroyUs{
        new UDMCPUTime("create(us)",
                       StatsView::MEAN | StatsView::MIN | StatsView::MAX)};
    // m_CreateThroughput: pairs per second. * prefix = output as throughput.
    std::shared_ptr<UDMThroughPut> m_CreateThroughput{
        new UDMThroughPut("*CreateTP(s^-1)",
                          StatsView::MEAN | StatsView::MIN | StatsView::MAX)};
    // Experiment configuration (constant per experiment value).
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
// BASELINE: Create/Destroy Pair
// ============================================================================
// The only test case: measure the cost of creating and destroying a
// critical + bulk Green Context pair.
//
// There is no comparison case because Green Context creation is a
// standalone operation — there is no "without Green Context" equivalent
// (you can't partition SMs without creating Green Contexts). The
// benchmark simply establishes the absolute cost.
BASELINE_F(greenContextLifecycle, createDestroyPair,
           GreenContextLifecycleFixture, SamplesCount, IterationsCount) {
    measureCreateDestroy();
}

#endif
