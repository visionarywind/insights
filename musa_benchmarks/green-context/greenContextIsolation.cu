// ============================================================================
// greenContextIsolation.cu — Green Context SM Isolation Benchmark
// ============================================================================
//
// Measures whether Green Context partitions actually isolate critical
// workloads from background interference at the SM level.
//
// The benchmark compares three scenarios:
//
//   1. primaryFullContention (BASELINE):
//      Both delay and critical kernels run in the primary context.
//      Delay uses ALL SMs. This represents worst-case interference —
//      the critical kernel has nowhere to run until delay blocks finish.
//
//   2. primaryBulkOnly:
//      Both kernels in primary context, but delay uses only the same
//      number of blocks as the bulk partition would. This isolates the
//      effect of "less background work" from the effect of Green Context
//      partitioning itself.
//
//   3. greenPartitioned (TARGET):
//      Delay runs in the bulk Green Context, critical runs in the
//      critical Green Context. If Green Context works correctly, the
//      critical kernel's latency should be close to its solo latency
//      regardless of what the bulk partition is doing.
//
// Metric: Isolation Score = solo_latency / contended_latency
//   - ~1.0 → perfect isolation (critical workload unaffected by background)
//   - << 1.0 → interference (background work delays critical kernel)
//
// Experiment values: critical SM count = 8 or 16.
// ============================================================================

#include "Celero.h"
#include "UserDefinedMeasurements.h"
#include "greenContextIsolation_common.h"

#include <memory>
#include <vector>

// Each sample runs the full measureContendedCritical flow (launch delay,
// wait for all blocks, measure critical). 20 samples give statistically
// meaningful mean/min/max. 1 iteration per sample avoids averaging
// across different background load states.
static const int SamplesCount = 20;
static const int IterationsCount = 1;

// ============================================================================
// Unsupported Build Fallback
// ============================================================================
#ifdef GREEN_CONTEXT_UNSUPPORTED_BUILD

// Generates a stub that reports Supported=0. See the macro definition
// in greenContextIsolation_common.h for details.
DECLARE_GREEN_CONTEXT_UNSUPPORTED_MAIN(greenContextLatency)

#else

// ============================================================================
// GreenContextLatencyFixture
// ============================================================================
//
// Test fixture for measuring critical kernel latency under various
// contention scenarios. Inherits SM partition setup from the base class.
//
// Lifecycle per experiment value (8 or 16 critical SMs):
//   setUp()         — partition SMs, create Green Contexts, alloc flags
//   onExperimentStart() → (inherited, no-op)
//   [for each sample]:
//     setUp() already done
//     BASELINE_F / BENCHMARK_F body:
//       measureSoloCritical()          → solo latency
//       measureContendedCritical()     → latency under contention
//       recordLatency(solo, contended) → push to metrics
//   [end samples]
//   onExperimentEnd() → (inherited, no-op)
//   tearDown()        — destroy streams, free flags, destroy Green Contexts
class GreenContextLatencyFixture : public GreenContextSmPartitionFixture {
public:
    // -------------------------------------------------------------------
    // Experiment Values
    // -------------------------------------------------------------------
    // Tests with 8 and 16 critical SMs. The bulk partition gets the
    // remaining SMs (total - critical - granularity overhead).
    std::vector<TestFixture::ExperimentValue> getExperimentValues() const override {
        return {8, 16};
    }

    // -------------------------------------------------------------------
    // setUp — One-time initialization per experiment value
    // -------------------------------------------------------------------
    void setUp(const TestFixture::ExperimentValue& experimentValue) override {
        // Step 1: SM partition (from base class). printTooSmallError=true
        // because isolation is the main benchmark — we want to know why
        // it was skipped.
        setUpSmPartition(experimentValue, /*printTooSmallError=*/true);

        // Step 2: Validate the split produced usable partitions.
        // This should never fail if setUpSmPartition succeeded, but
        // defensive checks prevent confusing downstream errors.
        if (m_CriticalBlocks <= 0 || m_BulkBlocks <= 0) {
            std::cerr << "invalid Green Context SM split: critical="
                      << m_CriticalBlocks << ", bulk=" << m_BulkBlocks << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // Step 3: Configure the spin kernel to use our chosen shared
        // memory size. Must be done before any kernel launch so that
        // occupancy reflects the actual resource usage.
        set_dynamic_shared_limit(m_SharedBytes);

        // Step 4: Ensure we are in the primary context before creating
        // Green Contexts (muGreenCtxCreate may change current context).
        checkMuErrors(muCtxSetCurrent(m_PrimaryCtx));

        // Step 5: Create Green Context partitions for the critical and
        // bulk domains. Each creation includes a nop_kernel warmup to
        // force lazy initialization.
        m_GreenCritical.create(m_Device, m_CriticalSm);
        m_GreenBulk.create(m_Device, m_BulkSm);
        if (m_GreenCritical.smCount != m_CriticalSm.sm.smCount ||
            m_GreenBulk.smCount != m_BulkSm.sm.smCount) {
            std::cerr << "Green Context provisioned SM count does not match split result"
                      << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // Step 6: Create primary-context streams for the baseline cases
        // (scenarios 1 and 2). Non-blocking streams allow the delay and
        // critical kernels to run concurrently on different streams.
        checkMuErrors(muCtxSetCurrent(m_PrimaryCtx));
        checkMusaErrors(musaStreamCreateWithFlags(
            &m_PrimaryDelayStream, musaStreamNonBlocking));
        checkMusaErrors(musaStreamCreateWithFlags(
            &m_PrimaryCriticalStream, musaStreamNonBlocking));

        // Step 7: Allocate mapped host memory for the started-flags
        // synchronization mechanism. musaHostAllocMapped gives both CPU
        // and GPU access to the same physical pages — the GPU writes
        // flags, the CPU polls them.
        //
        // One flag per SM (m_TotalSms) so every delay block can signal
        // independently.
        int* rawHostFlags = nullptr;
        checkMusaErrors(musaHostAlloc(
            &rawHostFlags, sizeof(int) * m_TotalSms, musaHostAllocMapped));
        m_HostStartedFlags = rawHostFlags;
        checkMusaErrors(musaHostGetDevicePointer(
            reinterpret_cast<void**>(&m_DeviceStartedFlags), rawHostFlags, 0));

        // Step 8: Synchronize to ensure all setup work is complete.
        checkMusaErrors(musaDeviceSynchronize());
    }

    // -------------------------------------------------------------------
    // tearDown — Cleanup in reverse order of setUp
    // -------------------------------------------------------------------
    void tearDown() override {
        // Drain all pending work before destroying resources.
        checkMusaErrors(musaDeviceSynchronize());

        // Destroy primary-context streams (baseline cases).
        if (m_PrimaryDelayStream != nullptr) {
            checkMusaErrors(musaStreamDestroy(m_PrimaryDelayStream));
            m_PrimaryDelayStream = nullptr;
        }
        if (m_PrimaryCriticalStream != nullptr) {
            checkMusaErrors(musaStreamDestroy(m_PrimaryCriticalStream));
            m_PrimaryCriticalStream = nullptr;
        }

        // Free mapped host memory (also invalidates m_DeviceStartedFlags).
        if (m_HostStartedFlags != nullptr) {
            checkMusaErrors(musaFreeHost(const_cast<int*>(m_HostStartedFlags)));
            m_HostStartedFlags = nullptr;
            m_DeviceStartedFlags = nullptr;
        }

        // Destroy Green Context partitions (destroys internal streams too).
        m_GreenCritical.destroy();
        m_GreenBulk.destroy();

        // Release primary context (from base class).
        tearDownSmPartition();
    }

    // -------------------------------------------------------------------
    // Metrics
    // -------------------------------------------------------------------
    // Reported columns in output CSV:
    //   solo(ms)   — critical kernel latency with no background load
    //   crit(ms)   — critical kernel latency under contention
    //   *Iso       — isolation score = solo / crit (1.0 = perfect)
    //   critSM     — number of SMs in critical partition
    //   bulkSM     — number of SMs in bulk partition
    //   block      — threads per block (from choose_launch_config)
    //   smem(KiB)  — dynamic shared memory per block in KiB
    //   blk/SM     — active blocks per SM (ideally 1)
    std::vector<std::shared_ptr<UserDefinedMeasurement>>
        getUserDefinedMeasurements() const override {
        return {m_SoloMs, m_CriticalMs, m_IsolationScore,
                m_CriticalSmCount, m_BulkSmCount,
                m_BlockSize, m_SharedKiB, m_ActiveBlocksPerSmCount};
    }

    // -------------------------------------------------------------------
    // measureSoloCritical
    // -------------------------------------------------------------------
    // Measures the critical kernel's execution time with NO background
    // load. This is the baseline latency that the isolation score
    // compares against.
    //
    // Delegates to the shared measure_critical_kernel_latency helper.
    float measureSoloCritical(MUstream stream, MUcontext ctx) {
        return measure_critical_kernel_latency(
            stream, ctx, m_CriticalBlocks, m_ThreadsPerBlock,
            m_SharedBytes, "solo critical");
    }

    // -------------------------------------------------------------------
    // measureContendedCritical
    // -------------------------------------------------------------------
    // Measures the critical kernel's execution time while a delay
    // (background) kernel is already occupying SMs.
    //
    // Protocol:
    //   1. Clear the started-flags array (mapped host memory).
    //   2. Launch the delay kernel on delayStream/delayCtx with
    //      delayBlocks blocks. Each block writes 1 to its flag after
    //      syncthreads, then enters the spin loop.
    //   3. Poll started-flags until ALL delay blocks have started.
    //      This ensures the delay kernel fully occupies its SMs.
    //   4. Launch and time the critical kernel on criticalStream/criticalCtx.
    //   5. Synchronize the delay stream (cleanup).
    //
    // The critical kernel is launched WITHOUT waiting for the delay
    // kernel to finish — this is the contention scenario we're measuring.
    //
    // Parameters:
    //   delayStream / delayCtx      — where the background load runs
    //   delayBlocks                 — how many blocks of background load
    //   criticalStream / criticalCtx — where the critical work runs
    //   label                       — diagnostic label for error messages
    //
    // Returns the critical kernel's GPU elapsed time in milliseconds.
    float measureContendedCritical(MUstream delayStream, MUcontext delayCtx,
                                   int delayBlocks, MUstream criticalStream,
                                   MUcontext criticalCtx, const char* label) {
        musaStream_t delay = runtime_stream(delayStream);

        // Reset flags before launch so we can detect when all blocks start.
        clear_started_flags(m_HostStartedFlags, delayBlocks);

        // Launch the delay kernel. Each block: touch smem → syncthreads →
        // write flag → spin for kDelaySpinCycles.
        checkMuErrors(muCtxSetCurrent(delayCtx));
        sm_occupancy_spin_kernel<<<delayBlocks, m_ThreadsPerBlock,
                                   m_SharedBytes, delay>>>(
            kDelaySpinCycles, m_DeviceStartedFlags, 1, m_SharedBytes);
        check_kernel_launch(label);

        // Busy-wait until all delay blocks have started their spin loop.
        // Critical: if we don't wait, the critical kernel might run before
        // the delay kernel occupies all its SMs, giving a falsely
        // optimistic isolation score.
        if (!wait_started_flags(m_HostStartedFlags, delayBlocks, label)) {
            std::exit(EXIT_FAILURE);
        }

        // Now measure the critical kernel under contention.
        const float ms = measure_critical_kernel_latency(
            criticalStream, criticalCtx, m_CriticalBlocks,
            m_ThreadsPerBlock, m_SharedBytes, label);

        // Cleanup: wait for delay kernel to finish before returning.
        checkMuErrors(muCtxSetCurrent(delayCtx));
        checkMusaErrors(musaStreamSynchronize(delay));
        return ms;
    }

    // -------------------------------------------------------------------
    // recordLatency
    // -------------------------------------------------------------------
    // Pushes one sample's solo and contended latency into the metrics
    // collectors. Also records the experiment configuration (SM counts,
    // block size, etc.) for traceability in the output CSV.
    void recordLatency(float soloMs, float criticalMs) {
        m_SoloMs->addValue(soloMs);
        m_CriticalMs->addValue(criticalMs);
        // Isolation score = solo / contended.
        //   = 1.0: background load had zero impact (perfect isolation)
        //   < 1.0: background load slowed down the critical kernel
        // Guard against division by zero (should never happen in practice).
        m_IsolationScore->addValue(
            criticalMs > 0.0f ? soloMs / criticalMs : 0.0f);
        m_CriticalSmCount->addValue(m_CriticalBlocks);
        m_BulkSmCount->addValue(m_BulkBlocks);
        m_BlockSize->addValue(m_ThreadsPerBlock);
        m_SharedKiB->addValue(m_SharedBytes / 1024);
        m_ActiveBlocksPerSmCount->addValue(m_ActiveBlocksPerSm);
    }

protected:
    // --- Primary-context resources (used by baseline cases 1 & 2) ---
    musaStream_t m_PrimaryDelayStream = nullptr;
    musaStream_t m_PrimaryCriticalStream = nullptr;

    // --- Synchronization flags (mapped host memory) ---
    // m_HostStartedFlags: CPU-side pointer for polling.
    // m_DeviceStartedFlags: GPU-side pointer passed to the delay kernel.
    volatile int* m_HostStartedFlags = nullptr;
    int* m_DeviceStartedFlags = nullptr;

    // --- Green Context partitions ---
    GreenContextBundle m_GreenCritical{};
    GreenContextBundle m_GreenBulk{};

    // --- Metrics collectors ---
    // m_SoloMs: critical kernel latency without background load.
    //   MEAN | MIN | MAX — want to see both average and worst case.
    std::shared_ptr<UDMGPUTime> m_SoloMs{
        new UDMGPUTime("solo(ms)", StatsView::MEAN | StatsView::MIN | StatsView::MAX)};
    // m_CriticalMs: critical kernel latency under contention.
    std::shared_ptr<UDMGPUTime> m_CriticalMs{
        new UDMGPUTime("crit(ms)", StatsView::MEAN | StatsView::MIN | StatsView::MAX)};
    // m_IsolationScore: solo / crit. * prefix = output as ratio.
    std::shared_ptr<UDMRatio> m_IsolationScore{
        new UDMRatio("*Iso", StatsView::MEAN | StatsView::MIN | StatsView::MAX)};
    // Experiment configuration (constant per experiment value, recorded
    // for traceability in the output CSV).
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
// Scenario 1: BASELINE — Primary Context, Full Contention
// ============================================================================
// Both delay and critical kernels run in the primary context on separate
// non-blocking streams. Delay uses ALL SMs (m_FullBlocks).
//
// Expected: worst isolation score. The delay kernel occupies every SM,
// so the critical kernel cannot begin until some delay blocks finish.
// This establishes the lower bound for isolation.
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
// Scenario 2: Primary Context, Bulk-Only Contention
// ============================================================================
// Both kernels in primary context, but delay uses only m_BulkBlocks
// (the same number of SMs the bulk Green Context would get).
//
// Purpose: separates the effect of "fewer background SMs" from the
// effect of "Green Context SM isolation". If the isolation score here
// is already good, then Green Context is not the differentiator —
// simply having fewer contending SMs is enough.
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
// Scenario 3: Green Context Partitioned (Target Case)
// ============================================================================
// Delay runs in the bulk Green Context, critical runs in the critical
// Green Context. Each Green Context has its own CUcontext and SM
// partition.
//
// Expected: if Green Context works correctly, the critical kernel's
// latency should be close to its solo latency regardless of the delay
// kernel's activity. The isolation score should approach 1.0.
//
// This is the key metric for the Green Context feature — it validates
// that SM-level isolation actually prevents interference.
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
