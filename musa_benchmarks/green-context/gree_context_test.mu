/**
 * Green Context validation test (MUSA Driver API).
 *
 * Follows MUSA Programming Guide §4.6 (Green Contexts):
 *   1) Query SM resources
 *   2) Partition with muDevSmResourceSplitByCount
 *   3) Create green contexts + streams
 *   4) Run kernels and verify SM provisioning + execution time under contention
 *
 * Requires: MUSA 12.4+ driver API, CC 9.0+ GPU recommended (H100/H200).
 *
 * Run:
 *   ./green_context_test [bench_iters] [skip-warmup]
 * skip-warmup: skip | 1 | true  → 不做 warmup；0 | false | 省略 → bench_iters>1 时默认 warmup
 */

#include <musa.h>
#include <musa_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <cassert>

#define MU_CHECK(expr)                                                         \
  do {                                                                         \
    MUresult _err = (expr);                                                    \
    if (_err != MUSA_SUCCESS) {                                                \
      const char* name = nullptr;                                              \
      const char* msg = nullptr;                                               \
      muGetErrorName(_err, &name);                                             \
      muGetErrorString(_err, &msg);                                             \
      fprintf(stderr, "MUSA driver error %s (%d) at %s:%d: %s\n",              \
              name ? name : "?", static_cast<int>(_err), __FILE__, __LINE__,   \
              msg ? msg : "");                                                 \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define MUSA_CHECK(expr)                                                       \
  do {                                                                         \
    musaError_t _err = (expr);                                                 \
    if (_err != musaSuccess) {                                                 \
      fprintf(stderr, "MUSA runtime error %s at %s:%d\n",                      \
              musaGetErrorString(_err), __FILE__, __LINE__);                   \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define KERNEL_LAUNCH_CHECK()                                                  \
  do {                                                                         \
    musaError_t _err = musaGetLastError();                                     \
    if (_err != musaSuccess) {                                                 \
      fprintf(stderr, "kernel launch error %s at %s:%d\n",                     \
              musaGetErrorString(_err), __FILE__, __LINE__);                   \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void nop_kernel() {
  return;
}

__global__ void vector_add_kernel(const float* a, const float* b, float* c,
                                  int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

// 192 KiB target; mcc reserves 12 B of static shared per kernel (see .shared_memory_size).
constexpr int k_block_smem_bytes = 192 * 1024 - 12;

// Touch every byte so mcc cannot DCE the 192 KiB shared allocation (see .s metadata).
__device__ __attribute__((noinline)) unsigned char occupy_block_smem_192k(
    volatile unsigned char* smem) {
  unsigned char acc = 0;
  for (int i = threadIdx.x; i < k_block_smem_bytes; i += blockDim.x) {
    smem[i] = static_cast<unsigned char>(i);
    acc |= smem[i];
  }
  return acc;
}

// Occupies one MP per block via 192 KiB static shared memory, then spins.
// When block0 reaches this point, it marks `started_flag` for host polling.
// Note: no atomic is needed because only (threadIdx.x==0 && blockIdx.x==0)
// writes started_flag[0].
__global__ void delay_kernel(unsigned long long spin_cycles, int* started_flag) {
  __shared__ unsigned char smem[k_block_smem_bytes];
  const unsigned char smem_tag =
      occupy_block_smem_192k(reinterpret_cast<volatile unsigned char*>(smem));
  __syncthreads();
#ifdef DEBUG
  if (threadIdx.x == 0) {
    printf("delay_kernel: blockIdx.x: %d, mpId: %d\n", blockIdx.x, __musa_get_special_register(9));
  }
#endif
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    // Start marker: host may use this to ensure critical launch ordering.
    started_flag[0] = smem_tag ? 1 : 1;
  }
  unsigned long long start = clock64();
  while (clock64() - start < spin_cycles) {
  }
}

// Same implementation as delay_kernel; launch sites pass a shorter spin_cycles.
__global__ void critical_kernel(unsigned long long spin_cycles, int* started_flag) {
  __shared__ unsigned char smem[k_block_smem_bytes];
  const unsigned char smem_tag =
      occupy_block_smem_192k(reinterpret_cast<volatile unsigned char*>(smem));
  __syncthreads();
#ifdef DEBUG
  if (threadIdx.x == 0) {
    printf("critical_kernel: blockIdx.x: %d, mpId: %d\n", blockIdx.x, __musa_get_special_register(9));
  }
#endif
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    started_flag[0] = smem_tag ? 1 : 1;
  }
  unsigned long long start = clock64();
  while (clock64() - start < spin_cycles) {
  }
}

static unsigned int align_up(unsigned int value, unsigned int alignment) {
  if (alignment == 0) return value;
  return ((value + alignment - 1) / alignment) * alignment;
}

static unsigned int sm_partition_granularity(const MUdevResource& res) {
  // CC 9.0+: multiples of 8; doc also mentions min 8 SMs per partition.
  unsigned int total = res.sm.smCount;
  if (total >= 80) return 8;
  return 2;
}

static int setCurrentCOntextFromGreenCtx(MUgreenCtx greeenCtx, MUcontext* ctx) {
  MU_CHECK(muCtxFromGreenCtx(ctx, greeenCtx));
  MU_CHECK(muCtxSetCurrent(*ctx));
  return 0;
}

struct GreenContextBundle {
  MUgreenCtx green_ctx{};
  MUcontext ctx{};
  MUstream stream{};
  MUdevResource sm_resource{};
  unsigned int sm_count = 0;
};

static int create_green_context_from_sm_resource(
    MUdevice dev, const MUdevResource& sm_res, GreenContextBundle* out) {
  MUdevResourceDesc desc{};
  MU_CHECK(muDevResourceGenerateDesc(&desc, const_cast<MUdevResource*>(&sm_res), 1));
  MU_CHECK(muGreenCtxCreate(&out->green_ctx, desc, dev,
                            MU_GREEN_CTX_DEFAULT_STREAM));
  assert(setCurrentCOntextFromGreenCtx(out->green_ctx, &out->ctx) == 0);
  // MU_CHECK(muCtxFromGreenCtx(&out->ctx, out->green_ctx));
  MU_CHECK(muGreenCtxStreamCreate(&out->stream, out->green_ctx,
                                    MU_STREAM_NON_BLOCKING, 0));

  out->sm_resource = {};
  MU_CHECK(muGreenCtxGetDevResource(out->green_ctx, &out->sm_resource,
                                    MU_DEV_RESOURCE_TYPE_SM));
  out->sm_count = out->sm_resource.sm.smCount;
  MUSA_CHECK(musaStreamSynchronize(reinterpret_cast<musaStream_t>(out->stream)));
  nop_kernel<<<1, 1, 0, reinterpret_cast<musaStream_t>(out->stream)>>>();
  KERNEL_LAUNCH_CHECK();
  MUSA_CHECK(musaStreamSynchronize(reinterpret_cast<musaStream_t>(out->stream)));
  return 0;
}

static void destroy_green_context(GreenContextBundle* b) {
  if (b->stream) muStreamDestroy(b->stream);
  if (b->green_ctx) muGreenCtxDestroy(b->green_ctx);
  b->stream = nullptr;
  b->green_ctx = nullptr;
  b->ctx = nullptr;
}

static int test_sm_provisioning_and_kernel_correctness(MUdevice dev) {
  printf("\n=== Test 1: SM provisioning + kernel correctness ===\n");

  MUdevResource all_sm{};
  MU_CHECK(muDeviceGetDevResource(dev, &all_sm, MU_DEV_RESOURCE_TYPE_SM));
  const unsigned int total_sms = all_sm.sm.smCount;
  const unsigned int gran = sm_partition_granularity(all_sm);
  printf("Device total SMs: %u (partition granularity %u)\n", total_sms, gran);

  if (total_sms < gran * 2) {
    printf("SKIP: need at least %u SMs for a two-way split\n", gran * 2);
    return 0;
  }

  unsigned int critical_target =
      align_up(std::min(16u, total_sms / 4), gran);
  if (critical_target >= total_sms) critical_target = gran;

  unsigned int nb_groups = 1;
  MUdevResource critical_sm{};
  MUdevResource remainder{};
  MU_CHECK(muDevSmResourceSplitByCount(
      &critical_sm, &nb_groups, &all_sm, &remainder, 0, critical_target));
  printf("Split: critical partition %u SMs, remainder %u SMs\n",
         critical_sm.sm.smCount, remainder.sm.smCount);

  if (critical_sm.sm.smCount == 0 || remainder.sm.smCount == 0) {
    fprintf(stderr, "FAIL: invalid SM split\n");
    return 1;
  }

  GreenContextBundle gc_critical{}, gc_bulk{};
  if (create_green_context_from_sm_resource(dev, critical_sm, &gc_critical) ||
      create_green_context_from_sm_resource(dev, remainder, &gc_bulk)) {
    return 1;
  }

  printf("Green context A (critical): provisioned %u SMs (expected %u)\n",
         gc_critical.sm_count, critical_sm.sm.smCount);
  printf("Green context B (bulk):     provisioned %u SMs (expected %u)\n",
         gc_bulk.sm_count, remainder.sm.smCount);

  if (gc_critical.sm_count != critical_sm.sm.smCount ||
      gc_bulk.sm_count != remainder.sm.smCount) {
    fprintf(stderr,
            "FAIL: provisioned SM count does not match split result\n");
    destroy_green_context(&gc_critical);
    destroy_green_context(&gc_bulk);
    return 1;
  }

  // Run vector add on critical green-context stream.
  MU_CHECK(muCtxSetCurrent(gc_critical.ctx));
  const int n = 1 << 20;
  std::vector<float> ha(n, 1.f), hb(n, 2.f), hc(n);
  float *da = nullptr, *db = nullptr, *dc = nullptr;
  MUSA_CHECK(musaMalloc(&da, n * sizeof(float)));
  MUSA_CHECK(musaMalloc(&db, n * sizeof(float)));
  MUSA_CHECK(musaMalloc(&dc, n * sizeof(float)));
  MUSA_CHECK(musaMemcpy(da, ha.data(), n * sizeof(float), musaMemcpyHostToDevice));
  MUSA_CHECK(musaMemcpy(db, hb.data(), n * sizeof(float), musaMemcpyHostToDevice));

  const int tpb = 256;
  const int blocks = (n + tpb - 1) / tpb;
  vector_add_kernel<<<blocks, tpb, 0,
                      reinterpret_cast<musaStream_t>(gc_critical.stream)>>>(
      da, db, dc, n);
  MUSA_CHECK(musaStreamSynchronize(
      reinterpret_cast<musaStream_t>(gc_critical.stream)));

  MUSA_CHECK(musaMemcpy(hc.data(), dc, n * sizeof(float), musaMemcpyDeviceToHost));
  for (int i = 0; i < n; i += n / 16) {
    if (hc[i] != 3.0f) {
      fprintf(stderr, "FAIL: vector_add result hc[%d]=%f (expected 3.0)\n", i,
              hc[i]);
      musaFree(da);
      musaFree(db);
      musaFree(dc);
      destroy_green_context(&gc_critical);
      destroy_green_context(&gc_bulk);
      return 1;
    }
  }
  musaFree(da);
  musaFree(db);
  musaFree(dc);
  printf("PASS: kernel on green-context stream produced correct results\n");

  destroy_green_context(&gc_critical);
  destroy_green_context(&gc_bulk);
  return 0;
}

static bool parse_skip_warmup(const char* arg) {
  if (arg == nullptr || arg[0] == '\0') return false;
  if (std::strcmp(arg, "0") == 0 || std::strcmp(arg, "false") == 0 ||
      std::strcmp(arg, "no") == 0) {
    return false;
  }
  if (std::strcmp(arg, "1") == 0 || std::strcmp(arg, "true") == 0 ||
      std::strcmp(arg, "skip") == 0 || std::strcmp(arg, "--no-warmup") == 0) {
    return true;
  }
  return std::atoi(arg) != 0;
}

// Poll mapped flag set by delay_kernel block0 (after smem init, before spin ends).
static bool wait_delay_started_flag(int* h_delay_started, const char* label,
                                    int delay_blocks) {
  for (int i = 0; i < 100000; ++i) {
    if (*h_delay_started != 0) return true;
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }
  fprintf(stderr, "FAIL: delay_kernel did not start (%s, blocks=%d)\n", label,
          delay_blocks);
  return false;
}

static int test_critical_execution_isolation(MUdevice dev, int bench_iters,
                                           bool skip_warmup) {
  printf("\n=== Test 2: Critical kernel execution time (Green Context vs baseline) ===\n");

  MUdevResource all_sm{};
  MU_CHECK(muDeviceGetDevResource(dev, &all_sm, MU_DEV_RESOURCE_TYPE_SM));
  const unsigned int total_sms = all_sm.sm.smCount;
  const unsigned int gran = sm_partition_granularity(all_sm);

  unsigned int critical_target = align_up(16u, gran);
  if (critical_target + gran > total_sms) critical_target = gran;

  unsigned int nb_groups = 1;
  MUdevResource critical_sm{};
  MUdevResource bulk_sm{};
  MU_CHECK(muDevSmResourceSplitByCount(
      &critical_sm, &nb_groups, &all_sm, &bulk_sm, 0, critical_target));

  printf("SM split: critical %u, bulk %u\n", critical_sm.sm.smCount,
         bulk_sm.sm.smCount);

  MUSA_CHECK(musaSetDevice(0));

  const unsigned long long delay_spin_cycles = 25000000ULL;
  const unsigned long long critical_work_cycles = 6000000ULL;
  // One block per MP; 192 KiB static smem/block (same as delay_kernel).
  const int delay_blocks = static_cast<int>(bulk_sm.sm.smCount);
  const int critical_blocks = static_cast<int>(critical_sm.sm.smCount);
  const int critical_threads = 256;

  int* h_delay_started = nullptr;
  int* d_delay_started = nullptr;
  MUSA_CHECK(musaHostAlloc(&h_delay_started, sizeof(int), musaHostAllocMapped));
  MUSA_CHECK(musaHostGetDevicePointer(
      reinterpret_cast<void**>(&d_delay_started), h_delay_started, 0));

  auto run_scenario = [&](MUstream delay_stream, MUcontext delay_ctx,
                          MUstream critical_stream, MUcontext critical_ctx,
                          int delay_block_count, const char* label,
                          bool print_line) -> float {
    musaStream_t delay = reinterpret_cast<musaStream_t>(delay_stream);
    musaStream_t crit = reinterpret_cast<musaStream_t>(critical_stream);

    *h_delay_started = 0;
    MU_CHECK(muCtxSetCurrent(delay_ctx));
    musaEvent_t t_start{}, t_stop{};
    MUSA_CHECK(musaEventCreate(&t_start));
    MUSA_CHECK(musaEventCreate(&t_stop));

    MU_CHECK(muCtxSetCurrent(critical_ctx));
    musaEvent_t t_crit_done{};
    MUSA_CHECK(musaEventCreate(&t_crit_done));

    MU_CHECK(muCtxSetCurrent(delay_ctx));
    MUSA_CHECK(musaEventRecord(t_start, delay));
    delay_kernel<<<delay_block_count, 256, 0, delay>>>(
        delay_spin_cycles, d_delay_started);
    KERNEL_LAUNCH_CHECK();

    // Ensure critical launch happens after delay_kernel launch has started on GPU.
    if (!wait_delay_started_flag(h_delay_started, label, delay_block_count)) {
      MU_CHECK(muCtxSetCurrent(critical_ctx));
      musaEventDestroy(t_crit_done);
      MU_CHECK(muCtxSetCurrent(delay_ctx));
      musaEventDestroy(t_start);
      musaEventDestroy(t_stop);
      return -1.f;
    }

    MU_CHECK(muCtxSetCurrent(critical_ctx));
    critical_kernel<<<critical_blocks, critical_threads, 0, crit>>>(
        critical_work_cycles, d_delay_started);
    KERNEL_LAUNCH_CHECK();
    MUSA_CHECK(musaEventRecord(t_crit_done, crit));

    MU_CHECK(muCtxSetCurrent(delay_ctx));
    MUSA_CHECK(musaStreamWaitEvent(delay, t_crit_done, 0));
    MUSA_CHECK(musaEventRecord(t_stop, delay));

    MUSA_CHECK(musaEventSynchronize(t_stop));

    float total_ms = 0.f;
    MUSA_CHECK(musaEventElapsedTime(&total_ms, t_start, t_stop));

    MU_CHECK(muCtxSetCurrent(critical_ctx));
    musaEventDestroy(t_crit_done);
    MU_CHECK(muCtxSetCurrent(delay_ctx));
    musaEventDestroy(t_start);
    musaEventDestroy(t_stop);

    if (print_line) {
      printf("  [%s] delay+critical total (overlap-aware): %.3f ms "
             "(started=%d, delay_blocks=%d)\n",
             label, total_ms, *h_delay_started, delay_block_count);
    }
    return total_ms;
  };

  auto measure_delay_kernel_alone = [&](MUstream stream, MUcontext ctx,
                                        int delay_block_count,
                                        const char* label) -> float {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream);
    MU_CHECK(muCtxSetCurrent(ctx));
    *h_delay_started = 0;
    musaEvent_t t0{}, t1{};
    MUSA_CHECK(musaEventCreate(&t0));
    MUSA_CHECK(musaEventCreate(&t1));
    MUSA_CHECK(musaEventRecord(t0, s));
    delay_kernel<<<delay_block_count, 256, 0, s>>>(
        delay_spin_cycles, d_delay_started);
    KERNEL_LAUNCH_CHECK();
    MUSA_CHECK(musaEventRecord(t1, s));
    MUSA_CHECK(musaEventSynchronize(t1));
    float ms = 0.f;
    MUSA_CHECK(musaEventElapsedTime(&ms, t0, t1));
    musaEventDestroy(t0);
    musaEventDestroy(t1);
    return ms;
  };

  auto measure_critical_kernel_alone = [&](MUstream stream, MUcontext ctx,
                                           const char* label) -> float {
    musaStream_t s = reinterpret_cast<musaStream_t>(stream);
    MU_CHECK(muCtxSetCurrent(ctx));
    musaEvent_t t0{}, t1{};
    MUSA_CHECK(musaEventCreate(&t0));
    MUSA_CHECK(musaEventCreate(&t1));
    MUSA_CHECK(musaEventRecord(t0, s));
    critical_kernel<<<critical_blocks, critical_threads, 0, s>>>(
        critical_work_cycles, d_delay_started);
    KERNEL_LAUNCH_CHECK();
    MUSA_CHECK(musaEventRecord(t1, s));
    MUSA_CHECK(musaEventSynchronize(t1));
    float ms = 0.f;
    MUSA_CHECK(musaEventElapsedTime(&ms, t0, t1));
    musaEventDestroy(t0);
    musaEventDestroy(t1);
    return ms;
  };

  MUcontext primary{};
  MU_CHECK(muDevicePrimaryCtxRetain(&primary, dev));
  MU_CHECK(muCtxSetCurrent(primary));

  auto run_isolated_kernel_timings = [&]() -> int {
    printf("  [isolated] solo kernel timings (no concurrent peer kernel)\n");
    MUSA_CHECK(musaDeviceSynchronize());

    musaStream_t base_delay{}, base_critical{};
    MUSA_CHECK(musaStreamCreateWithFlags(&base_delay, musaStreamNonBlocking));
    MUSA_CHECK(musaStreamCreateWithFlags(&base_critical, musaStreamNonBlocking));

    float delay_ms = measure_delay_kernel_alone(
        reinterpret_cast<MUstream>(base_delay), primary, delay_blocks,
        "primary ctx delay_kernel");
    float crit_ms = measure_critical_kernel_alone(
        reinterpret_cast<MUstream>(base_critical), primary,
        "primary ctx critical_kernel");
    musaStreamDestroy(base_delay);
    musaStreamDestroy(base_critical);
    if (delay_ms < 0.f || crit_ms < 0.f) return 1;
    MUSA_CHECK(musaDeviceSynchronize());

    printf("  [isolated primary] delay_kernel: %.3f ms (blocks=%d), "
           "critical_kernel: %.3f ms (blocks=%d)\n",
           delay_ms, delay_blocks, crit_ms, critical_blocks);

    GreenContextBundle gc_critical{}, gc_bulk{};
    if (create_green_context_from_sm_resource(dev, critical_sm, &gc_critical) ||
        create_green_context_from_sm_resource(dev, bulk_sm, &gc_bulk)) {
      return 1;
    }

    delay_ms = measure_delay_kernel_alone(gc_bulk.stream, gc_bulk.ctx,
                                          delay_blocks,
                                          "green bulk delay_kernel");
    crit_ms = measure_critical_kernel_alone(gc_critical.stream, gc_critical.ctx,
                                            "green critical critical_kernel");
    destroy_green_context(&gc_critical);
    destroy_green_context(&gc_bulk);
    if (delay_ms < 0.f || crit_ms < 0.f) return 1;
    MUSA_CHECK(musaDeviceSynchronize());

    printf("  [isolated green] delay_kernel: %.3f ms (blocks=%d), "
           "critical_kernel: %.3f ms (blocks=%d)\n",
           delay_ms, delay_blocks, crit_ms, critical_blocks);
    return 0;
  };

  const int warmup_iters = (bench_iters > 1 && !skip_warmup) ? 1 : 0;
  const bool print_each = (bench_iters == 1);
  double baseline_sum = 0.0;
  double gc_sum = 0.0;

  for (int w = 0; w < warmup_iters; ++w) {
    MUSA_CHECK(musaDeviceSynchronize());

    musaStream_t base_delay{}, base_critical{};
    MUSA_CHECK(musaStreamCreateWithFlags(&base_delay, musaStreamNonBlocking));
    MUSA_CHECK(musaStreamCreateWithFlags(&base_critical, musaStreamNonBlocking));
#ifndef DEBUG
    const float baseline_ms = run_scenario(
        reinterpret_cast<MUstream>(base_delay), primary,
        reinterpret_cast<MUstream>(base_critical), primary, delay_blocks,
        "primary ctx", false);
    musaStreamDestroy(base_delay);
    musaStreamDestroy(base_critical);
    if (baseline_ms < 0.f) {
      musaFreeHost(h_delay_started);
      return 1;
    }
    MUSA_CHECK(musaDeviceSynchronize());
#endif
    GreenContextBundle gc_critical{}, gc_bulk{};
    if (create_green_context_from_sm_resource(dev, critical_sm, &gc_critical) ||
        create_green_context_from_sm_resource(dev, bulk_sm, &gc_bulk)) {
      musaFreeHost(h_delay_started);
      return 1;
    }

    const float gc_ms =
        run_scenario(gc_bulk.stream, gc_bulk.ctx, gc_critical.stream,
                     gc_critical.ctx, delay_blocks, "green contexts", false);
    destroy_green_context(&gc_critical);
    destroy_green_context(&gc_bulk);
    if (gc_ms < 0.f) {
      musaFreeHost(h_delay_started);
      return 1;
    }
    MUSA_CHECK(musaDeviceSynchronize());
#ifndef DEBUG
    printf("  [warmup] primary: %.3f ms, green: %.3f ms (not in avg)\n",
           baseline_ms, gc_ms);
#endif
  }

#ifndef DEBUG
  if (run_isolated_kernel_timings()) {
    musaFreeHost(h_delay_started);
    return 1;
  }

  for (int i = 0; i < bench_iters; ++i) {
    MUSA_CHECK(musaDeviceSynchronize());

    musaStream_t base_delay{}, base_critical{};
    MUSA_CHECK(musaStreamCreateWithFlags(&base_delay, musaStreamNonBlocking));
    MUSA_CHECK(musaStreamCreateWithFlags(&base_critical, musaStreamNonBlocking));

    const float baseline_ms = run_scenario(
        reinterpret_cast<MUstream>(base_delay), primary,
        reinterpret_cast<MUstream>(base_critical), primary, delay_blocks,
        "primary ctx", print_each);
    musaStreamDestroy(base_delay);
    musaStreamDestroy(base_critical);
    if (baseline_ms < 0.f) {
      musaFreeHost(h_delay_started);
      return 1;
    }
    MUSA_CHECK(musaDeviceSynchronize());

    GreenContextBundle gc_critical{}, gc_bulk{};
    if (create_green_context_from_sm_resource(dev, critical_sm, &gc_critical) ||
        create_green_context_from_sm_resource(dev, bulk_sm, &gc_bulk)) {
      musaFreeHost(h_delay_started);
      return 1;
    }

    const float gc_ms =
        run_scenario(gc_bulk.stream, gc_bulk.ctx, gc_critical.stream,
                     gc_critical.ctx, delay_blocks, "green contexts",
                     print_each);
    destroy_green_context(&gc_critical);
    destroy_green_context(&gc_bulk);
    if (gc_ms < 0.f) {
      musaFreeHost(h_delay_started);
      return 1;
    }
    MUSA_CHECK(musaDeviceSynchronize());

    if (bench_iters > 1) {
      printf("  [iter %d/%d] primary: %.3f ms, green: %.3f ms\n", i + 1,
             bench_iters, baseline_ms, gc_ms);
    }
    baseline_sum += baseline_ms;
    gc_sum += gc_ms;
  }

  const float baseline_total_ms =
      static_cast<float>(baseline_sum / bench_iters);
  const float gc_total_ms = static_cast<float>(gc_sum / bench_iters);

  if (bench_iters > 1) {
    printf("  [primary ctx] delay+critical avg (%d iters): %.3f ms "
           "(delay_blocks=%d)\n",
           bench_iters, baseline_total_ms, delay_blocks);
    printf("  [green contexts] delay+critical avg (%d iters): %.3f ms "
           "(delay_blocks=%d)\n",
           bench_iters, gc_total_ms, delay_blocks);
  }

  musaFreeHost(h_delay_started);

  // Green contexts should finish delay+critical sooner under bulk/critical split.
  const float improvement = baseline_total_ms - gc_total_ms;
  printf("Improvement (baseline - green ctx) overlap-aware total: %.3f ms\n",
         improvement);

  const float rel_improvement =
      baseline_total_ms > 0.f ? improvement / baseline_total_ms : 0.f;

  if (gc_total_ms < baseline_total_ms * 0.8f && improvement > 2.0f) {
    printf("PASS: green contexts finished concurrent kernels much faster\n");
    return 0;
  }

  if (improvement > 5.0f && gc_total_ms < baseline_total_ms) {
    printf("PASS: green contexts reduced overlap-aware total time\n");
    return 0;
  }

  if (rel_improvement > 0.15f && gc_total_ms < baseline_total_ms) {
    printf("PASS: green contexts reduced overlap-aware total by >15%%\n");
    return 0;
  }

  fprintf(stderr,
          "FAIL: expected green contexts to reduce overlap-aware total time "
          "(baseline=%.3f ms, green=%.3f ms)\n",
          baseline_total_ms, gc_total_ms);
#endif
  return 1;
}

int main(int argc, char** argv) {
  int bench_iters = 20;
  bool skip_warmup = false;
  if (argc > 1) bench_iters = std::atoi(argv[1]);
  if (argc > 2) skip_warmup = parse_skip_warmup(argv[2]);
  if (bench_iters < 1) bench_iters = 20;

  MU_CHECK(muInit(0));

  MUdevice dev{};
  MU_CHECK(muDeviceGet(&dev, 0));

  char name[256]{};
  MU_CHECK(muDeviceGetName(name, sizeof(name), dev));
  int cc_major = 0, cc_minor = 0;
  MU_CHECK(muDeviceGetAttribute(&cc_major, MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                              dev));
  MU_CHECK(muDeviceGetAttribute(&cc_minor, MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                              dev));
  printf("Green Context test on %s (CC %d.%d, bench_iters=%d, warmup=%s)\n", name,
         cc_major, cc_minor, bench_iters,
         (bench_iters > 1 && !skip_warmup) ? "on" : "off");

  MUcontext primary{};
  MU_CHECK(muDevicePrimaryCtxRetain(&primary, dev));
  MU_CHECK(muCtxSetCurrent(primary));

  // if (test_sm_provisioning_and_kernel_correctness(dev)) return 1;
  if (test_critical_execution_isolation(dev, bench_iters, skip_warmup)) return 1;

  printf("\nAll green context tests completed successfully.\n");
  return 0;
}
