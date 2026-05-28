#include "kernel_profiler.h"
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cxxabi.h>
#include <dlfcn.h>
#include <cstring>

namespace KernelProfiler {

MUpti::DriverExportTable* g_DriverExport = nullptr;
Driver::ExportTable*      g_DriverApiTable = nullptr;
std::atomic<double>       g_GpuTimestampPeriodNs{0.0};

std::unordered_map<uint64_t, KernelRecord*> g_Records;
std::mutex g_Mutex;

static void LazyQueryGpuFrequency() {
    if (g_GpuTimestampPeriodNs.load() > 0.0) return;

    void* lib = dlopen("libmusa.so", RTLD_NOLOAD);
    if (!lib) lib = dlopen("libmusa.so.1", RTLD_NOLOAD);
    if (!lib) return;

    using muDeviceGet_fn = MUresult(*)(MUdevice*, int);
    using muDeviceGetAttribute_fn = MUresult(*)(int*, MUdevice_attribute, MUdevice);

    auto muDevGet = reinterpret_cast<muDeviceGet_fn>(
        dlsym(lib, "muDeviceGet"));
    auto muDevGetAttr = reinterpret_cast<muDeviceGetAttribute_fn>(
        dlsym(lib, "muDeviceGetAttribute"));
    if (!muDevGet || !muDevGetAttr) return;

    MUdevice dev;
    if (muDevGet(&dev, 0) != MUSA_SUCCESS) return;

    int clockKHz = 0;
    if (muDevGetAttr(&clockKHz, MU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev) != MUSA_SUCCESS
        || clockKHz <= 0) return;

    g_GpuTimestampPeriodNs.store(1e6 / static_cast<double>(clockKHz));
}

static char* Demangle(const char* mangled) {
    if (!mangled) return nullptr;
    if (mangled[0] != '_' || mangled[1] != 'Z') return strdup(mangled);
    int status = 0;
    char* d = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
    return (status == 0 && d) ? d : strdup(mangled);
}

std::string TimestampStr() {
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    std::stringstream ss;
    ss << std::put_time(std::localtime(&tt), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

void PrintKernelReport(const KernelRecord& rec) {
    double cpuSchedUs = (rec.queued_time > 0 && rec.submit_time > rec.queued_time)
        ? (rec.submit_time - rec.queued_time) / 1000.0 : -1.0;

    printf("[PROFILER] %s | %-50s | Grid(%u,%u,%u) Block(%u,%u,%u) Smem:%lu\n",
           TimestampStr().c_str(), rec.kernelName.c_str(),
           rec.gridSize.x, rec.gridSize.y, rec.gridSize.z,
           rec.blockSize.x, rec.blockSize.y, rec.blockSize.z,
           rec.dynamicSmem);

    double periodNs = g_GpuTimestampPeriodNs.load();
    if (rec.gpu_begin > 0 && rec.gpu_end > rec.gpu_begin) {
        double gpuExecTicks = static_cast<double>(rec.gpu_end - rec.gpu_begin);
        if (periodNs > 0.0) {
            double gpuExecUs = gpuExecTicks * periodNs / 1000.0;
            printf("[PROFILER]   CPU调度: %8.2f us | GPU执行: %8.2f us (%10.0f ticks)\n",
                   cpuSchedUs, gpuExecUs, gpuExecTicks);
        } else {
            printf("[PROFILER]   CPU调度: %8.2f us | GPU执行(raw): %10.0f ticks (需校准频率)\n",
                   cpuSchedUs, gpuExecTicks);
        }
    } else {
        printf("[PROFILER]   CPU调度: %8.2f us | GPU时间戳不可用\n", cpuSchedUs);
    }
    fflush(stdout);
}

void OnAssignKernelToKick(uint64_t kernelId, uint64_t submissionId) {
    (void)kernelId; (void)submissionId;
}

MUpti::Context* OnRegisterKernelV2(Musa::DispatchCommand* command) {
    if (!g_DriverExport || !command) return nullptr;

    auto* cmdBase    = g_DriverExport->CommandBase;
    auto* dispAccess = g_DriverExport->DispatchCommand;
    auto* funcAccess = g_DriverExport->Function;
    auto* cmdPtr     = reinterpret_cast<Musa::Command*>(command);

    MUfunction func = dispAccess->GetFunction(command);
    uint32_t corrId = cmdBase->GetCorrelationId(cmdPtr);

    auto* rec = new KernelRecord();
    rec->correlationId  = corrId;
    {
        char* d = Demangle(func ? funcAccess->GetName(func) : nullptr);
        rec->kernelName = d ? d : "(unknown)";
        if (d) free(d);
    }
    rec->gridSize       = dispAccess->GetGridSize(command);
    rec->blockSize      = dispAccess->GetBlockSize(command);
    rec->dynamicSmem    = dispAccess->GetDynamicSharedMemoryUsage(command);
    rec->queued_time    = cmdBase->GetQueuedTimestamp(cmdPtr);
    rec->has_submit_time = false;

    { std::lock_guard<std::mutex> lk(g_Mutex); g_Records[corrId] = rec; }
    return reinterpret_cast<MUpti::Context*>(rec);
}

MUpti::Context* OnRegisterGraphKernel(Musa::GraphCommand*, Musa::GraphNode* node) {
    if (!g_DriverExport || !node) return nullptr;

    auto* ga = g_DriverExport->Graph;
    uint32_t nid = ga->NodeGetId(node);

    auto* rec = new KernelRecord();
    rec->correlationId  = nid;
    rec->kernelName     = "(graph kernel)";
    rec->has_submit_time = false;

    { std::lock_guard<std::mutex> lk(g_Mutex); g_Records[nid] = rec; }
    return reinterpret_cast<MUpti::Context*>(rec);
}

void OnMarkKernelQueued(uint64_t correlationId) {
    std::lock_guard<std::mutex> lk(g_Mutex);
    auto* rec = g_Records.count(correlationId) ? g_Records[correlationId] : nullptr;
    if (rec) rec->queued_time = Now();
}

void OnMarkKernelSubmitted(uint64_t correlationId) {
    std::lock_guard<std::mutex> lk(g_Mutex);
    auto* rec = g_Records.count(correlationId) ? g_Records[correlationId] : nullptr;
    if (rec) { rec->submit_time = Now(); rec->has_submit_time = true; }
}

void OnMarkCommandBeginEnd(MUpti::Context* context, Musa::Command* command) {
    auto* rec = reinterpret_cast<KernelRecord*>(context);
    if (!rec || !g_DriverExport || !command) return;

    LazyQueryGpuFrequency();  // 第一次 kernel 完成时查询 GPU 频率

    auto* cb = g_DriverExport->CommandBase;
    if (rec->queued_time == 0) rec->queued_time = cb->GetQueuedTimestamp(command);
    if (!rec->has_submit_time) rec->submit_time = cb->GetSubmittedTimestamp(command);
    cb->GetBeginEndTimestamp(command, &rec->gpu_begin, &rec->gpu_end);

    PrintKernelReport(*rec);
    { std::lock_guard<std::mutex> lk(g_Mutex); g_Records.erase(rec->correlationId); }
    delete rec;
}

void OnMarkGraphNodeBeginEndV2(MUpti::Context* context, Musa::GraphNode* node, Musa::GraphExec* exec) {
    auto* rec = reinterpret_cast<KernelRecord*>(context);
    if (!rec || !g_DriverExport || !node) return;

    LazyQueryGpuFrequency();

    auto* ga = g_DriverExport->Graph;
    if (!rec->has_submit_time) rec->submit_time = ga->NodeGetSubmittedTimestamp(node);

    uint64_t b = 0, e = 0;
    ga->NodeGetBeginEndTimestampV2(node, &b, &e, exec);
    rec->gpu_begin = b; rec->gpu_end = e;

    PrintKernelReport(*rec);
    { std::lock_guard<std::mutex> lk(g_Mutex); g_Records.erase(rec->correlationId); }
    delete rec;
}

} // namespace KernelProfiler

extern "C" {

int InitializeInjection() {
    using namespace KernelProfiler;
    using muGetExportTable_fn = MUresult(const void**, const MUuuid*);

    void* musaLib = dlopen("libmusa.so", RTLD_NOLOAD | RTLD_GLOBAL);
    if (!musaLib) {
        musaLib = dlopen("libmusa.so.1", RTLD_NOLOAD | RTLD_GLOBAL);
    }
    if (!musaLib) {
        fprintf(stderr, "[PROFILER] ERROR: libmusa.so not loaded: %s\n", dlerror());
        return -1;
    }

    auto* getTable = reinterpret_cast<muGetExportTable_fn*>(
        dlsym(musaLib, "muGetExportTable"));
    if (!getTable) {
        fprintf(stderr, "[PROFILER] ERROR: muGetExportTable not found: %s\n", dlerror());
        return -2;
    }

    // 1. Get MUpti::DriverExportTable (for hooks and command introspection)
    MUuuid muptiUuid;
    *reinterpret_cast<Client*>(&muptiUuid) = Client::MUpti;
    const void* tPtr = nullptr;
    if (getTable(&tPtr, &muptiUuid) != MUSA_SUCCESS || !tPtr) {
        fprintf(stderr, "[PROFILER] ERROR: get MUpti export table failed\n");
        return -2;
    }
    g_DriverExport = const_cast<MUpti::DriverExportTable*>(
        static_cast<const MUpti::DriverExportTable*>(tPtr));

    // 2. Get Driver::ExportTable (skipped: causes crash, needs further debug)
    // auto* apiTable = new Driver::ExportTable{};
    // ...
    g_DriverApiTable = nullptr;

    // 3. Register hooks — fill ALL hooks to avoid calling AccessorHint stub
    g_DriverExport->MUpti->Enable([](MUpti::MUptiDriverHooks* hooks) {
        // Kernel profiling hooks (active)
        hooks->RegisterKernelV2        = KernelProfiler::OnRegisterKernelV2;
        hooks->MarkCommandBeginEnd     = KernelProfiler::OnMarkCommandBeginEnd;
        hooks->RegisterGraphKernel     = KernelProfiler::OnRegisterGraphKernel;
        hooks->MarkGraphNodeBeginEndV2 = KernelProfiler::OnMarkGraphNodeBeginEndV2;
        hooks->AssignKernelToKick      = KernelProfiler::OnAssignKernelToKick;
        hooks->MarkKernelQueued        = KernelProfiler::OnMarkKernelQueued;
        hooks->MarkKernelSubmitted     = KernelProfiler::OnMarkKernelSubmitted;

        // No-op hooks — required to prevent crashes (no AccessorHint guards)
        auto noopCtx = [](Musa::MemcpyCommand*) -> MUpti::Context* { return nullptr; };
        auto noopCtxMemset = [](Musa::MemsetCommand*) -> MUpti::Context* { return nullptr; };
        auto noopCtxAtomic = [](Musa::MemoryAtomicCommand*) -> MUpti::Context* { return nullptr; };
        auto noopCtxAtomicV2 = [](Musa::MemoryAtomicCommand*, uint64_t) -> MUpti::Context* { return nullptr; };
        auto noopCtxAtomicVal = [](Musa::MemoryAtomicValueCommand*) -> MUpti::Context* { return nullptr; };
        auto noopCtxTransfer = [](Musa::MemoryTransferCommand*) -> MUpti::Context* { return nullptr; };
        auto noopCtxTransferV2 = [](Musa::MemoryTransferCommand*, uint64_t) -> MUpti::Context* { return nullptr; };

        hooks->EnterMemcpy              = noopCtx;
        hooks->ExitMemcpy               = [](MUpti::Context*) {};
        hooks->EnterMemset              = noopCtxMemset;
        hooks->ExitMemset               = [](MUpti::Context*) {};
        hooks->EnterMemoryAtomic        = noopCtxAtomic;
        hooks->EnterMemoryAtomicV2      = noopCtxAtomicV2;
        hooks->EnterMemoryAtomicValue   = noopCtxAtomicVal;
        hooks->EnterMemoryTransfer      = noopCtxTransfer;
        hooks->EnterMemoryTransferV2    = noopCtxTransferV2;
        hooks->RegisterKernel           = [](Musa::DispatchCommand*) {};
        hooks->RegisterKernelV2         = KernelProfiler::OnRegisterKernelV2; // overwrite with real impl
        hooks->MarkCommandBeginEnd      = KernelProfiler::OnMarkCommandBeginEnd;
        hooks->MarkCommandBeginEndV2    = [](MUpti::Context*, Musa::Command*, uint64_t) {};
        hooks->RegisterGraphKernel      = KernelProfiler::OnRegisterGraphKernel;
        hooks->RegisterGraphMemcpy      = [](Musa::GraphCommand*, Musa::GraphNode*) -> MUpti::Context* { return nullptr; };
        hooks->RegisterGraphMemset      = [](Musa::GraphCommand*, Musa::GraphNode*) -> MUpti::Context* { return nullptr; };
        hooks->RegisterGraphMemoryAtomic      = [](Musa::GraphCommand*, Musa::GraphNode*) -> MUpti::Context* { return nullptr; };
        hooks->RegisterGraphMemoryAtomicValue = [](Musa::GraphCommand*, Musa::GraphNode*) -> MUpti::Context* { return nullptr; };
        hooks->RegisterGraphMemoryTransfer    = [](Musa::GraphCommand*, Musa::GraphNode*) -> MUpti::Context* { return nullptr; };
        hooks->MarkGraphNodeBeginEndV2 = KernelProfiler::OnMarkGraphNodeBeginEndV2;

        // These hooks have built-in AccessorHint guards in tracepoints.h — safe to skip
        hooks->CreateContext            = hooks->CreateContext; // keep default (safe via guard)
        hooks->DestroyContext           = hooks->DestroyContext;
        hooks->CreateStream             = hooks->CreateStream;
        hooks->DestroyStream            = hooks->DestroyStream;
        hooks->StartHostMemcpy          = hooks->StartHostMemcpy;
        hooks->StopHostMemcpy           = hooks->StopHostMemcpy;
        hooks->StartHostMemset          = hooks->StartHostMemset;
        hooks->StopHostMemset           = hooks->StopHostMemset;
        hooks->RegisterStreamWaitEvent  = hooks->RegisterStreamWaitEvent;
        hooks->StartStreamWaitEvent     = hooks->StartStreamWaitEvent;
        hooks->StopStreamWaitEvent      = hooks->StopStreamWaitEvent;
        hooks->RegisterEventSynchronize = hooks->RegisterEventSynchronize;
        hooks->StartEventSynchronize    = hooks->StartEventSynchronize;
        hooks->StopEventSynchronize     = hooks->StopEventSynchronize;
        hooks->RegisterStreamSynchronize = hooks->RegisterStreamSynchronize;
        hooks->StartStreamSynchronize   = hooks->StartStreamSynchronize;
        hooks->StopStreamSynchronize    = hooks->StopStreamSynchronize;
        hooks->RegisterContextSynchronize = hooks->RegisterContextSynchronize;
        hooks->StartContextSynchronize  = hooks->StartContextSynchronize;
        hooks->StopContextSynchronize   = hooks->StopContextSynchronize;
        hooks->AssignSubmissionToCorrelation = hooks->AssignSubmissionToCorrelation;
        hooks->CheckGraphTraceEnabled   = hooks->CheckGraphTraceEnabled;
        hooks->RegisterGraphTrace       = hooks->RegisterGraphTrace;
        hooks->MarkGraphTraceBegin      = hooks->MarkGraphTraceBegin;
        hooks->MarkGraphTraceEnd        = hooks->MarkGraphTraceEnd;
    });

    fprintf(stdout, "[PROFILER] Kernel profiler initialized successfully.\n");
    fflush(stdout);
    return 0;
}

} // extern "C"
