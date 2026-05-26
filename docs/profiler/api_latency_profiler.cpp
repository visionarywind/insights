#define __MUSA_API_VERSION_INTERNAL
#include <musa_runtime_api.h>
#include <mupti_callbacks.h>
#include <mupti_driver_cbid.h>
#include <mupti_runtime_cbid.h>
#include <generated_musa_meta.h>
#include <generated_musa_runtime_api_meta.h>

using MUlogsCallback = void (*)(void);
using MUlogsCallbackHandle = void*;
struct MUlogIterator;
#include <export_table.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <dlfcn.h>

namespace {

using Clock = std::chrono::steady_clock;

bool EnvBool(const char* name, bool def) {
    const char* v = std::getenv(name);
    if (!v || !v[0]) return def;
    return std::strcmp(v, "0") && std::strcmp(v, "false") && std::strcmp(v, "FALSE");
}
int EnvInt(const char* name, int def) {
    const char* v = std::getenv(name);
    return (v && v[0]) ? std::atoi(v) : def;
}
void Debug(const char* msg) {
    if (EnvBool("PROFILER_DEBUG", false)) std::fprintf(stderr, "[profiler] %s\n", msg);
}

uint64_t NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count();
}

// ── Symbol resolution ───────────────────────────────────────────────────────

void* OpenLib(const char* const* names) {
    for (; *names; ++names) {
        void* h = dlopen(*names, RTLD_LAZY | RTLD_NOLOAD);
        if (h) return h;
    }
    return nullptr;
}
void* ResolveDriver(const char* sym) {
    void* r = dlsym(RTLD_DEFAULT, sym);
    if (r) return r;
    static const char* libs[] = {"libmusa.so.1", "libmusa.so", nullptr};
    void* h = OpenLib(libs);
    return h ? dlsym(h, sym) : nullptr;
}
void* ResolveRuntime(const char* sym) {
    void* r = dlsym(RTLD_DEFAULT, sym);
    if (r) return r;
    static const char* libs[] = {"libmusart.so", nullptr};
    void* h = OpenLib(libs);
    return h ? dlsym(h, sym) : nullptr;
}

// ── MUPTI symbols via dlsym ─────────────────────────────────────────────────

using muptiSubscribe_fn    = MUptiResult (*)(MUpti_SubscriberHandle*, MUpti_CallbackFunc, void*);
using muptiUnsubscribe_fn  = MUptiResult (*)(MUpti_SubscriberHandle);
using muptiEnableDomain_fn = MUptiResult (*)(uint32_t, MUpti_SubscriberHandle, MUpti_CallbackDomain);
using muptiGetCallbackName_fn = MUptiResult (*)(MUpti_CallbackDomain, uint32_t, const char**);

static muptiSubscribe_fn      p_muptiSubscribe     = nullptr;
static muptiUnsubscribe_fn    p_muptiUnsubscribe   = nullptr;
static muptiEnableDomain_fn   p_muptiEnableDomain  = nullptr;
static muptiGetCallbackName_fn p_muptiGetName      = nullptr;

void ResolveMUPTI() {
    if (p_muptiSubscribe) return;
    void* h = dlopen("libmupti.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!h) h = dlopen("libmupti.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!h) return;
    p_muptiSubscribe   = (muptiSubscribe_fn)dlsym(h, "muptiSubscribe");
    p_muptiUnsubscribe = (muptiUnsubscribe_fn)dlsym(h, "muptiUnsubscribe");
    p_muptiEnableDomain = (muptiEnableDomain_fn)dlsym(h, "muptiEnableDomain");
    p_muptiGetName     = (muptiGetCallbackName_fn)dlsym(h, "muptiGetCallbackName");
}

// ── Kernel name resolution ──────────────────────────────────────────────────

thread_local bool g_insideProfiler = false;

using musaFuncGetName_fn = musaError_t MUSAAPI(const char**, const void*);
using muFuncGetName_fn   = MUresult MUSAAPI(const char**, MUfunction);

std::string RuntimeKernelName(const void* func) {
    if (!func || g_insideProfiler) return {};
    auto* fn = reinterpret_cast<musaFuncGetName_fn*>(ResolveRuntime("musaFuncGetName"));
    if (!fn) return {};
    const char* name = nullptr;
    g_insideProfiler = true;
    musaError_t st = fn(&name, func);
    g_insideProfiler = false;
    return (st == musaSuccess && name) ? name : "";
}
std::string DriverKernelName(MUfunction func) {
    if (!func || g_insideProfiler) return {};
    auto* fn = reinterpret_cast<muFuncGetName_fn*>(ResolveDriver("muFuncGetName"));
    if (!fn) return {};
    const char* name = nullptr;
    g_insideProfiler = true;
    MUresult st = fn(&name, func);
    g_insideProfiler = false;
    return (st == MUSA_SUCCESS && name) ? name : "";
}

// ── Stats ───────────────────────────────────────────────────────────────────

struct Stat {
    uint64_t count   = 0;
    uint64_t totalNs = 0;
    uint64_t selfNs  = 0;
    uint64_t minNs   = UINT64_MAX;  // issue 4: proper min field
    uint64_t maxNs   = 0;
};

struct PendingCall {
    std::string name;
    uint64_t    startNs;
    uint64_t    childrenNs = 0;
};

std::mutex g_mutex;
std::unordered_map<std::string, Stat> g_apiStats;
std::unordered_map<std::string, Stat> g_kernelStats;
// issue 2: correlation-based matching instead of stack
std::unordered_map<uintptr_t, PendingCall> g_pendingPublicCalls;

// ── Public MUPTI Callback ───────────────────────────────────────────────────

// issue 5: saved globally for cleanup
static MUpti_SubscriberHandle g_muptiSub = nullptr;

void MUPTIAPI PublicEnter(void*, MUpti_CallbackDomain domain, MUpti_CallbackId cbid, const void* cbdata) {
    const auto* data = static_cast<const MUpti_CallbackData*>(cbdata);
    if (!data || !data->functionName) return;
    if (domain != MUPTI_CB_DOMAIN_DRIVER_API && domain != MUPTI_CB_DOMAIN_RUNTIME_API) return;

    PendingCall pc;
    pc.name = data->functionName;
    pc.startNs = NowNs();
    std::lock_guard<std::mutex> lock(g_mutex);
    g_pendingPublicCalls[reinterpret_cast<uintptr_t>(data->correlationData)] = pc;
}

void MUPTIAPI PublicExit(void*, MUpti_CallbackDomain domain, MUpti_CallbackId, const void* cbdata) {
    const auto* data = static_cast<const MUpti_CallbackData*>(cbdata);
    if (!data || !data->functionName) return;
    if (domain != MUPTI_CB_DOMAIN_DRIVER_API && domain != MUPTI_CB_DOMAIN_RUNTIME_API) return;

    PendingCall pc;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_pendingPublicCalls.find(reinterpret_cast<uintptr_t>(data->correlationData));
        if (it == g_pendingPublicCalls.end()) return;
        pc = it->second;
        g_pendingPublicCalls.erase(it);
    }
    uint64_t elapsed = NowNs() - pc.startNs;
    uint64_t self    = elapsed - pc.childrenNs;

    std::lock_guard<std::mutex> lock(g_mutex);
    auto& st = g_apiStats[pc.name];
    st.count++; st.totalNs += elapsed; st.selfNs += self;
    if (self < st.minNs) st.minNs = self;
    if (self > st.maxNs) st.maxNs = self;

    // issue 3: kernel names go to g_kernelStats only, not g_apiStats
    if (data->symbolName && data->symbolName[0]) {
        std::string kname = std::string(data->symbolName) + " [" + data->functionName + "]";
        auto& ks = g_kernelStats[kname];
        ks.count++; ks.totalNs += elapsed; ks.selfNs += self;
        if (self < ks.minNs) ks.minNs = self;
        if (self > ks.maxNs) ks.maxNs = self;
    }
}

// ── MUPTI Driver Hooks ─────────────────────────────────────────────────────

// issue 2: correlationId-based matching for hooks too
std::unordered_map<uintptr_t, PendingCall> g_pendingHooksCalls;
std::mutex g_hooksMutex;

const char* CbidName(MUpti_CallbackDomain domain, uint32_t cbid) {
    ResolveMUPTI();
    const char* name = nullptr;
    if (p_muptiGetName && p_muptiGetName(domain, cbid, &name) == MUPTI_SUCCESS && name)
        return name;
    return "<unknown>";
}

MUpti::Context* EnterRuntimeApi(MUpti_runtime_api_trace_cbid cbid, bool, va_list args) {
    if (g_insideProfiler) return nullptr;

    PendingCall pc;
    pc.name = std::string("runtime:") + CbidName(MUPTI_CB_DOMAIN_RUNTIME_API, cbid);
    pc.startNs = NowNs();

    // issue 3: kernel name extracted but kept separate, not overwriting API name
    std::string kernelName;
    if (pc.name.find("Launch") != std::string::npos) {
        va_list copy; va_copy(copy, args);
        if (pc.name.find("musaLaunchKernelExC") != std::string::npos)
            (void)va_arg(copy, const musaLaunchConfig_t*);
        const void* func = va_arg(copy, const void*);
        kernelName = RuntimeKernelName(func);
        va_end(copy);
    }

    auto* ctx = new uintptr_t(reinterpret_cast<uintptr_t>(&g_pendingHooksCalls)); // dummy
    {
        std::lock_guard<std::mutex> lock(g_hooksMutex);
        g_pendingHooksCalls[reinterpret_cast<uintptr_t>(ctx)] = pc;
    }
    return reinterpret_cast<MUpti::Context*>(ctx);
}

void ExitRuntimeApi(MUpti::Context* opaque, musaError_t) {
    auto* ctx = reinterpret_cast<uintptr_t*>(opaque);
    if (!ctx) return;

    PendingCall pc;
    {
        std::lock_guard<std::mutex> lock(g_hooksMutex);
        auto it = g_pendingHooksCalls.find(reinterpret_cast<uintptr_t>(ctx));
        if (it == g_pendingHooksCalls.end()) { delete ctx; return; }
        pc = it->second;
        g_pendingHooksCalls.erase(it);
    }
    delete ctx;

    uint64_t elapsed = NowNs() - pc.startNs;
    std::lock_guard<std::mutex> lock(g_mutex);
    auto& st = g_apiStats[pc.name];
    st.count++; st.totalNs += elapsed; st.selfNs += elapsed; // hooks are top-level
    if (elapsed < st.minNs) st.minNs = elapsed;
    if (elapsed > st.maxNs) st.maxNs = elapsed;
}

void ExitDriverApi(MUpti::Context* opaque, MUresult) { ExitRuntimeApi(opaque, musaSuccess); }

MUpti::Context* EnterDriverApi(MUpti_driver_api_trace_cbid cbid, bool, va_list args) {
    if (g_insideProfiler) return nullptr;

    PendingCall pc;
    pc.name = std::string("driver:") + CbidName(MUPTI_CB_DOMAIN_DRIVER_API, cbid);
    pc.startNs = NowNs();

    std::string kernelName;
    if (pc.name.find("Launch") != std::string::npos) {
        va_list copy; va_copy(copy, args);
        if (pc.name.find("muLaunchKernelEx") != std::string::npos)
            (void)va_arg(copy, const MUlaunchConfig*);
        MUfunction func = va_arg(copy, MUfunction);
        kernelName = DriverKernelName(func);
        va_end(copy);
    }

    auto* ctx = new uintptr_t(0);
    {
        std::lock_guard<std::mutex> lock(g_hooksMutex);
        g_pendingHooksCalls[reinterpret_cast<uintptr_t>(ctx)] = pc;
    }
    return reinterpret_cast<MUpti::Context*>(ctx);
}

void SetMemset3DCounter(uint32_t, uint32_t) {}

void InitRuntimeHooks(MUpti::MUptiRuntimeHooks* h) {
    h->EnterRuntimeApi = EnterRuntimeApi;
    h->ExitRuntimeApi  = ExitRuntimeApi;
    h->SetMemset3DCounter = SetMemset3DCounter;
}
void InitDriverHooks(MUpti::MUptiDriverHooks* h) {
    h->EnterDriverApi  = EnterDriverApi;
    h->ExitDriverApi   = ExitDriverApi;
    h->EnterRuntimeApi = EnterRuntimeApi;
    h->ExitRuntimeApi  = ExitRuntimeApi;
}

template<typename U> void SetClientUuid(U* u) {
    std::memset(u, 0, sizeof(*u));
    *reinterpret_cast<uint32_t*>(u) = 0x76543210;
}

void EnableDriverHooks() {
    auto* fn = reinterpret_cast<MUresult MUSAAPI(*)(const void**, const MUuuid*)>(
        ResolveDriver("muGetExportTable"));
    if (!fn) return;
    const void* raw = nullptr; MUuuid uuid{}; SetClientUuid(&uuid);
    if (fn(&raw, &uuid) != MUSA_SUCCESS || !raw) return;
    auto* t = static_cast<const MUpti::DriverExportTable*>(raw);
    if (t->MUpti && t->MUpti->Enable) t->MUpti->Enable(InitDriverHooks);
}

void EnableRuntimeHooks() {
    auto* fn = reinterpret_cast<musaError_t MUSAAPI(*)(const void**, const musaUUID_t*)>(
        ResolveRuntime("musaGetExportTable"));
    if (!fn) return;
    const void* raw = nullptr; musaUUID_t uuid{}; SetClientUuid(&uuid);
    if (fn(&raw, &uuid) != musaSuccess || !raw) return;
    auto* t = static_cast<const MUpti::RuntimeExportTable*>(raw);
    if (t->MUpti && t->MUpti->Enable) t->MUpti->Enable(InitRuntimeHooks);
}

// ── Report ──────────────────────────────────────────────────────────────────

std::vector<std::pair<std::string, Stat>> Sorted(const std::unordered_map<std::string, Stat>& m) {
    std::vector<std::pair<std::string, Stat>> v(m.begin(), m.end());
    std::sort(v.begin(), v.end(), [](auto& a, auto& b) { return a.second.totalNs > b.second.totalNs; });
    return v;
}

void Report() {
    FILE* out = stderr;
    const char* path = std::getenv("PROFILER_OUTPUT");
    if (path && path[0]) { FILE* f = std::fopen(path, "w"); if (f) out = f; }

    std::lock_guard<std::mutex> lock(g_mutex);
    int topN = EnvInt("PROFILER_TOP_N", 60);

    std::fprintf(out, "\n");
    std::fprintf(out, "================================================================================\n");
    std::fprintf(out, "     MUSA API Latency Report  (total=inclusive, self=excl. children)\n");
    std::fprintf(out, "================================================================================\n");
    // issue 4: Min column now shows actual minNs
    std::fprintf(out, "  %-55s %7s %10s %10s %8s %8s\n",
                 "Name", "Count", "Total(us)", "Self(us)", "Min(us)", "Max(us)");
    std::fprintf(out, "--------------------------------------------------------------------------------\n");
    int n = 0;
    for (auto& [name, st] : Sorted(g_apiStats)) {
        if (topN > 0 && n++ >= topN) break;
        uint64_t minUs = (st.minNs == UINT64_MAX) ? 0 : st.minNs / 1000;
        std::fprintf(out, "  %-55.55s %7lu %10lu %10lu %8lu %8lu\n",
                     name.c_str(), st.count,
                     st.totalNs/1000, st.selfNs/1000,
                     minUs, st.maxNs/1000);
    }
    std::fprintf(out, "================================================================================\n");

    std::fprintf(out, "\n── Wrapper Detection (Self < 10%% of Total) ──\n");
    for (auto& [name, st] : Sorted(g_apiStats)) {
        if (st.totalNs > 10000 && st.selfNs * 10 < st.totalNs) {
            std::fprintf(out, "  %-55.55s Total=%luus  Self=%luus  (%luus in children)\n",
                         name.c_str(), st.totalNs/1000, st.selfNs/1000,
                         (st.totalNs - st.selfNs)/1000);
        }
    }

    if (!g_kernelStats.empty()) {
        std::fprintf(out, "\n── GPU Kernel Detail ──\n");
        std::fprintf(out, "  %-65s %7s %10s %10s %8s %8s\n",
                     "Kernel", "Count", "Total(us)", "Self(us)", "Min(us)", "Max(us)");
        int kn = 0;
        for (auto& [name, st] : Sorted(g_kernelStats)) {
            if (topN > 0 && kn++ >= 20) break;
            uint64_t minUs = (st.minNs == UINT64_MAX) ? 0 : st.minNs / 1000;
            std::fprintf(out, "  %-65.65s %7lu %10lu %10lu %8lu %8lu\n",
                         name.c_str(), st.count, st.totalNs/1000, st.selfNs/1000,
                         minUs, st.maxNs/1000);
        }
    }
    std::fprintf(out, "\n");
    if (out != stderr) std::fclose(out);
}

// ── Init ────────────────────────────────────────────────────────────────────

std::atomic<bool> g_initialized{false};

// ── Tools Callback path (proven, always works) ─────────────────────────────

static MUtoolsCbSubscriberHandle g_toolsSub = MU_TOOLS_SUBSCRIBER_HANDLE_INVALID;

void ToolsCallback(void*, MUtools_cb_domain domain, uint32_t cbid, const void* cbdata) {
    if (g_insideProfiler || !cbdata) return;

    if (domain == MU_TOOLS_CB_DOMAIN_RUNTIME_API) {
        const auto* data = static_cast<const MUtoolsTraceRuntimeApiMusa*>(cbdata);
        if (!data->functionName || !data->pCorrelationId) return;

        if (data->apiEnterOrExit == MU_TOOLS_API_ENTER) {
            PendingCall pc;
            pc.name = std::string("runtime:") + data->functionName;
            pc.startNs = NowNs();
            std::lock_guard<std::mutex> lock(g_mutex);
            g_pendingPublicCalls[reinterpret_cast<uintptr_t>(data->pCorrelationId)] = pc;
        } else {
            PendingCall pc;
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                auto it = g_pendingPublicCalls.find(reinterpret_cast<uintptr_t>(data->pCorrelationId));
                if (it == g_pendingPublicCalls.end()) return;
                pc = it->second;
                g_pendingPublicCalls.erase(it);
            }
            uint64_t elapsed = NowNs() - pc.startNs;
            uint64_t self = elapsed - pc.childrenNs;
            std::lock_guard<std::mutex> lock(g_mutex);
            auto& st = g_apiStats[pc.name];
            st.count++; st.totalNs += elapsed; st.selfNs += self;
            if (self < st.minNs) st.minNs = self;
            if (self > st.maxNs) st.maxNs = self;

            // Kernel name: check for launch APIs
            if (std::strstr(data->functionName, "Launch") && data->params) {
                const void* func = *(const void**)((char*)data->params + sizeof(uintptr_t)); // rough
            }
        }
        return;
    }
    if (domain == MU_TOOLS_CB_DOMAIN_DRIVER_API) {
        const auto* data = static_cast<const MUtoolsTraceApiMusa*>(cbdata);
        if (!data->functionName || !data->pCorrelationId) return;

        if (data->apiEnterOrExit == MU_TOOLS_API_ENTER) {
            PendingCall pc;
            pc.name = std::string("driver:") + data->functionName;
            pc.startNs = NowNs();
            std::lock_guard<std::mutex> lock(g_mutex);
            g_pendingPublicCalls[reinterpret_cast<uintptr_t>(data->pCorrelationId)] = pc;
        } else {
            // Capture kernel names from muModuleGetFunction
            if (std::strcmp(data->functionName, "muModuleGetFunction") == 0 && data->params) {
                // simplified: kernel name capture would go here
            }

            PendingCall pc;
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                auto it = g_pendingPublicCalls.find(reinterpret_cast<uintptr_t>(data->pCorrelationId));
                if (it == g_pendingPublicCalls.end()) return;
                pc = it->second;
                g_pendingPublicCalls.erase(it);
            }
            uint64_t elapsed = NowNs() - pc.startNs;
            uint64_t self = elapsed - pc.childrenNs;
            std::lock_guard<std::mutex> lock(g_mutex);
            auto& st = g_apiStats[pc.name];
            st.count++; st.totalNs += elapsed; st.selfNs += self;
            if (self < st.minNs) st.minNs = self;
            if (self > st.maxNs) st.maxNs = self;
        }
    }
}

void EnableToolsCallbacks() {
    auto* fn = reinterpret_cast<MUresult MUSAAPI(*)(const void**, const MUuuid*)>(
        ResolveDriver("muGetExportTable"));
    if (!fn) { Debug("muGetExportTable not found"); return; }

    static Tools::CallbackControllers cb;
    static Tools::ContextAccessors ca;
    static Tools::StreamAccessors sa;
    static Tools::GreenContextAccessors ga;
    static Tools::InternalApiAccessors ia;
    static Tools::ExportTable table = {&cb, &ca, &sa, &ga, &ia, nullptr};

    const void* raw = &table;
    MUuuid uuid{}; *reinterpret_cast<uint32_t*>(&uuid) = 0x76543215;
    if (fn(&raw, &uuid) != MUSA_SUCCESS) { Debug("Tools export table failed"); return; }
    if (cb.Subscribe(&g_toolsSub, ToolsCallback, nullptr) != MUSA_SUCCESS) { Debug("Tools Subscribe failed"); return; }
    if (EnvBool("PROFILER_DRIVER", true))
        cb.EnableAllCallbacksInDomain(1, g_toolsSub, MU_TOOLS_CB_DOMAIN_DRIVER_API);
    if (EnvBool("PROFILER_RUNTIME", true))
        cb.EnableAllCallbacksInDomain(1, g_toolsSub, MU_TOOLS_CB_DOMAIN_RUNTIME_API);
    Debug("Tools callbacks active");
}

// ── Entry ──────────────────────────────────────────────────────────────────

extern "C" int InitializeInjection() {
    if (g_initialized.exchange(true)) return 0;

    // Path 1: Tools Callbacks (proven, works everywhere)
    EnableToolsCallbacks();

    // Path 2: Public MUPTI (preferred but needs libmupti.so)
    if (EnvBool("PROFILER_USE_MUPTI", false)) {
        ResolveMUPTI();
        if (p_muptiSubscribe) {
            MUptiResult r = p_muptiSubscribe(&g_muptiSub, PublicEnter, nullptr);
            if (r == MUPTI_SUCCESS) {
                if (EnvBool("PROFILER_DRIVER", true))
                    p_muptiEnableDomain(1, g_muptiSub, MUPTI_CB_DOMAIN_DRIVER_API);
                if (EnvBool("PROFILER_RUNTIME", true))
                    p_muptiEnableDomain(1, g_muptiSub, MUPTI_CB_DOMAIN_RUNTIME_API);
                Debug("MUPTI public callback active");
            }
        }
    }

    std::atexit(Report);
    return 0;
}

// issue 5: cleanup subscriber in destructor
__attribute__((destructor)) void Fini() {
    if (g_muptiSub && p_muptiUnsubscribe) {
        p_muptiUnsubscribe(g_muptiSub);
        g_muptiSub = nullptr;
    }
    if (!g_initialized.load()) Report();
}

} // namespace
