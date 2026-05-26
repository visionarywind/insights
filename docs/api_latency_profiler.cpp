#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <mutex>
#include <string>
#include <dlfcn.h>

#include "export_table.h"
#include "mupti/mupti_driver_cbid.h"

typedef struct muModuleGetFunction_params_st {
    MUfunction *hfunc;
    MUmodule hmod;
    const char *name;
} muModuleGetFunction_params;

typedef struct muLaunchKernel_params_st {
    MUfunction f;
    unsigned int gridDimX, gridDimY, gridDimZ;
    unsigned int blockDimX, blockDimY, blockDimZ;
    unsigned int sharedMemBytes;
    MUstream hStream;
    void **kernelParams;
    void **extra;
} muLaunchKernel_params;

struct CallRecord {
    CallRecord() = default;
    CallRecord(const char* n, uint64_t t) : name(n), start_ns(t) {}
    std::string name;
    uint64_t     start_ns;
};

struct ApiStats {
    uint64_t total_ns = 0;
    uint64_t count    = 0;
    uint64_t min_ns   = UINT64_MAX;
    uint64_t max_ns   = 0;
};

class ApiLatencyProfiler {
public:
    void OnEnter(const char* name, uint32_t) {
        std::lock_guard<std::mutex> lock(m_mutex);
        uint64_t now = std::chrono::steady_clock::now().time_since_epoch().count();
        m_callstack.emplace_back(name, now);
    }

    void OnExit(uint32_t, uint32_t) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_callstack.empty()) return;
        auto record = m_callstack.back();
        m_callstack.pop_back();
        uint64_t now = std::chrono::steady_clock::now().time_since_epoch().count();
        uint64_t elapsed = now - record.start_ns;
        auto& st = m_stats[record.name];
        st.total_ns += elapsed;
        st.count++;
        if (elapsed < st.min_ns) st.min_ns = elapsed;
        if (elapsed > st.max_ns) st.max_ns = elapsed;
    }

    void PrintReport() {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<std::pair<std::string, ApiStats>> sorted(m_stats.begin(), m_stats.end());
        std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) { return a.second.total_ns > b.second.total_ns; });
        fprintf(stderr, "\n");
        fprintf(stderr, "==============================================================================\n");
        fprintf(stderr, "                      MUSA API Latency Report\n");
        fprintf(stderr, "==============================================================================\n");
        fprintf(stderr, "  %-40s %8s %12s %10s %10s %10s\n",
                "Name", "Count", "Total(us)", "Avg(us)", "Min(us)", "Max(us)");
        fprintf(stderr, "------------------------------------------------------------------------------\n");
        for (const auto& [name, st] : sorted) {
            fprintf(stderr, "  %-40s %8lu %12lu %10lu %10lu %10lu\n",
                    name.c_str(), st.count,
                    st.total_ns / 1000,
                    st.total_ns / 1000 / std::max(st.count, 1UL),
                    st.min_ns / 1000,
                    st.max_ns / 1000);
        }
        fprintf(stderr, "==============================================================================\n\n");
    }

    ~ApiLatencyProfiler() { PrintReport(); }

private:
    std::mutex m_mutex;
    std::vector<CallRecord> m_callstack;
    std::unordered_map<std::string, ApiStats> m_stats;
};

static ApiLatencyProfiler g_profiler;
static std::unordered_map<MUfunction, std::string> g_kernel_names;
static std::mutex g_kernel_names_mutex;

static void ProfilerCallback(void*, MUtools_cb_domain domain, uint32_t cbid, const void* pParams) {
    if (domain == MU_TOOLS_CB_DOMAIN_DRIVER_API) {
        auto* api = static_cast<const MUtoolsTraceApiMusa*>(pParams);
        if (cbid == MUPTI_DRIVER_TRACE_CBID_muModuleGetFunction && api->apiEnterOrExit == MU_TOOLS_API_EXIT) {
            auto* fp = static_cast<const muModuleGetFunction_params*>(api->params);
            if (fp->hfunc && *fp->hfunc && fp->name) {
                std::lock_guard<std::mutex> lock(g_kernel_names_mutex);
                g_kernel_names[*fp->hfunc] = fp->name;
            }
            return;
        }
        if (cbid == MUPTI_DRIVER_TRACE_CBID_muLaunchKernel) {
            auto* kp = static_cast<const muLaunchKernel_params*>(api->params);
            const char* op_name = api->functionName;
            if (kp->f) {
                std::lock_guard<std::mutex> lock(g_kernel_names_mutex);
                auto it = g_kernel_names.find(kp->f);
                if (it != g_kernel_names.end()) op_name = it->second.c_str();
            }
            if (api->apiEnterOrExit == MU_TOOLS_API_ENTER) g_profiler.OnEnter(op_name, *api->pCorrelationId);
            else g_profiler.OnExit(*api->pCorrelationId, *api->pStatus);
            return;
        }
        if (api->apiEnterOrExit == MU_TOOLS_API_ENTER) g_profiler.OnEnter(api->functionName, *api->pCorrelationId);
        else g_profiler.OnExit(*api->pCorrelationId, *api->pStatus);
        return;
    }
    if (domain == MU_TOOLS_CB_DOMAIN_RUNTIME_API) {
        auto* api = static_cast<const MUtoolsTraceRuntimeApiMusa*>(pParams);
        if (api->apiEnterOrExit == MU_TOOLS_API_ENTER) g_profiler.OnEnter(api->functionName, *api->pCorrelationId);
        else g_profiler.OnExit(*api->pCorrelationId, *api->pStatus);
        return;
    }
}

extern "C" void InitializeInjection() {
    // Find muGetExportTable via dlsym (no link-time dependency on libmusa)
    typedef MUresult (*GetExportTable_fn)(void const**, MUuuid const*);
    GetExportTable_fn muGetExportTable = nullptr;

    void* self = dlopen(nullptr, RTLD_LAZY | RTLD_NOLOAD);
    if (self) muGetExportTable = (GetExportTable_fn)dlsym(self, "muGetExportTable");
    if (!muGetExportTable) muGetExportTable = (GetExportTable_fn)dlsym(RTLD_DEFAULT, "muGetExportTable");
    if (!muGetExportTable) {
        void* musa = dlopen("libmusa.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (musa) muGetExportTable = (GetExportTable_fn)dlsym(musa, "muGetExportTable");
    }
    if (!muGetExportTable) { fprintf(stderr, "[PROFILER] Failed to find muGetExportTable\n"); return; }

    Tools::CallbackControllers cb = {};
    Tools::ContextAccessors ca = {};
    Tools::StreamAccessors sa = {};
    Tools::GreenContextAccessors ga = {};
    Tools::InternalApiAccessors ia = {};
    Tools::ExportTable toolsTable = {};
    toolsTable.callback = &cb;
    toolsTable.context = &ca;
    toolsTable.stream = &sa;
    toolsTable.greenCtx = &ga;
    toolsTable.internal = &ia;

    const void* pExportTable = &toolsTable;
    MUuuid toolsUuid = {};
    *reinterpret_cast<uint32_t*>(&toolsUuid) = 0x76543215;

    MUresult res = muGetExportTable(&pExportTable, &toolsUuid);
    if (res != MUSA_SUCCESS) { fprintf(stderr, "[PROFILER] Failed to get Tools export table: %d\n", res); return; }

    MUtoolsCbSubscriberHandle handle = MU_TOOLS_SUBSCRIBER_HANDLE_INVALID;
    res = cb.Subscribe(&handle, ProfilerCallback, nullptr);
    if (res != MUSA_SUCCESS) { fprintf(stderr, "[PROFILER] Failed to subscribe: %d\n", res); return; }

    MUtools_cb_domain domains[] = { MU_TOOLS_CB_DOMAIN_DRIVER_API, MU_TOOLS_CB_DOMAIN_RUNTIME_API };
    for (auto domain : domains) {
        res = cb.EnableAllCallbacksInDomain(1, handle, domain);
        if (res != MUSA_SUCCESS) { fprintf(stderr, "[PROFILER] Failed to enable domain %d: %d\n", domain, res); return; }
    }
    fprintf(stderr, "[PROFILER] Initialized, tracing Driver + Runtime APIs w/ kernel names\n");
}
