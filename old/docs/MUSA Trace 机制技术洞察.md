# MUSA Trace 机制技术洞察

> 分析日期: 2026-05-19 | 覆盖: MUSA Driver (linux-ddk) + MUSA-Runtime

---

## 一、架构总览

MUSA Trace 系统是一个**四层可插拔探针架构**，在 600 个公开 API 的每一个入口/出口自动插入追踪代码，并暴露结构化接口供外部工具消费。

```
┌──────────────────────────────────────────────────────────────────┐
│                         外部工具层                                 │
│  nsight-compute / nsight-systems / 自定义 profiler               │
│                                                                  │
│  通过 muGetExportTable() 获取 MUpti::ExportTable                   │
│  → toolsSubscribe(callback, userdata)                            │
│  → toolsEnableCallback(enable, domain, cbid)                     │
│  → 接收 ENTER/EXIT 回调 → 关联 ID → 时间线构建                     │
└──────────────────────────────┬───────────────────────────────────┘
                               │ 运行时动态链接
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ [层 1] Generated Wrapper — mu_wrappers_generated.cpp (22k+ 行)    │
│                                                                  │
│  每个 API 函数体中：                                               │
│  1. ApiTrace trace(status, "muXxx", CBID, args...)  ← RAII 守卫  │
│  2. if (toolsCallbackEnabled(domain, cbid)) {                    │
│  3.     构造 MUtoolsTraceApiMusa{ params, correlationId }        │
│  4.     toolsIssueCallback(ENTER, ...);                          │
│  5.     if (!skipDriverImpl) status = muapiXxx(params);          │
│  6.     toolsIssueCallback(EXIT, ...);                           │
│     } else {                                                     │
│  7.     status = muapiXxx(args...);  ← 无追踪直接执行              │
│     }                                                            │
└──────────────────────────────┬───────────────────────────────────┘
                               │ toolsIssueCallback
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ [层 2] Callback Subsystem — callback.cpp (244 行)                 │
│                                                                  │
│  全局状态:                                                        │
│    g_subscriber        — 单一订阅者 (seq-lock 保护)                │
│    g_callbackEnabled[] — 4 个 CB domain × 每 CBID 1 bit 控制位图  │
│    g_hasSubscriber     — 原子开关 (订阅者存在则 1)                  │
│                                                                  │
│  关键函数:                                                        │
│    toolsSubscribe()       — 注册回调 (CAS 原子操作)                │
│    toolsEnableCallback()  — 按 CBID 开启/关闭追踪                  │
│    toolsCallbackEnabled() — O(1) 查询某 CBID 是否启用              │
│    toolsIssueCallback()   — seq-lock 安全地调用回调                │
└──────────────────────────────┬───────────────────────────────────┘
                               │ callback(userdata, domain, cbid, params)
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│ [层 3] MUPTI Activity Trace — 运行时内部探针                       │
│                                                                  │
│  export_table.h MUpti::ExportTable 暴露 50+ 函数指针:              │
│    CommandBaseAccessors    — GetStream, GetType, GetTimestamp     │
│    DispatchCommandAccessors — GetFunction, GetGridSize, ...       │
│    MemcpyCommandAccessors  — GetCopyKind, GetSrcMemKind, GetSize │
│    MemoryTransferAccessors — GetBase, GetSize                    │
│    ContextAccessors        — GetId, GetDevice                    │
│    StreamAccessors         — GetId, GetContext                   │
│    FunctionAccessors       — GetName, GetSharedMemSize            │
│                                                                  │
│  Activity Hooks (per command type):                               │
│    EnterMemset(cmd) → StartHostMemset → ... → ExitMemset         │
│    EnterMemcpy(cmd) → StartHostMemcpy → ... → ExitMemcpy         │
│    EnterDispatch(cmd) → MarkKernelQueued → ... → ExitDispatch    │
│    RegisterKernel → AssignKernelToKick → MarkKernelBeginEnd       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 二、Layer 1: Generated Wrapper — 自动探针插入

### 2.1 代码生成模式

`mu_wrappers_generated.cpp` 是一个 **22,000+ 行的自动生成文件**。每个公开 API 函数遵循以下模板：

```cpp
// mu_wrappers_generated.cpp:14-51 (muCtxGetId 示例)
MUresult MUSAAPI muCtxGetId(MUcontext ctx, unsigned long long* ctxId) {
    MUresult status = MUSA_SUCCESS;

    // ★ Step 1: RAII 守卫 — 记录 API 名 + CBID，析构时自动 set last error + print backtrace
    ApiTrace trace(status, "muCtxGetId", MUPTI_CBID_GET(muCtxGetId), ctx, ctxId);

    // ★ Step 2: 检查该 API 的追踪是否被启用
    if (toolsCallbackEnabled(MU_TOOLS_SUBSCRIBER_HANDLE,
                             MU_TOOLS_CB_DOMAIN_DRIVER_API,
                             MUPTI_DRIVER_TRACE_CBID_muCtxGetId)) {

        // ★ Step 3: 构造参数结构体
        MUtoolsTraceApiMusa inParams{};
        uint32_t correlationId = Util::ThreadInfo::Get().ApiSeqNum();
        uint32_t skipDriverImpl = 0;

        muCtxGetId_params params{};
        params.ctx = ctx;
        params.ctxId = ctxId;

        inParams.struct_size    = sizeof(inParams);
        inParams.pCorrelationId = &correlationId;       // 用于工具关联 ENTER↔EXIT
        inParams.pStatus        = &status;
        inParams.functionName   = "muCtxGetId";
        inParams.functionId     = MUPTI_DRIVER_TRACE_CBID_muCtxGetId;
        inParams.params         = &params;
        inParams.apiEnterOrExit = MU_TOOLS_API_ENTER;
        inParams.pSkipDriverImpl = &skipDriverImpl;     // ★ 工具可拦截执行

        // ★ Step 4: 通知工具 — API ENTER
        toolsIssueCallback(..., MU_TOOLS_API_ENTER, &inParams);

        // ★ Step 5: 执行实际 API (除非被工具拦截)
        if (!skipDriverImpl) {
            status = muapiCtxGetId(params.ctx, params.ctxId);
        }

        // ★ Step 6: 通知工具 — API EXIT
        inParams.apiEnterOrExit = MU_TOOLS_API_EXIT;
        toolsIssueCallback(..., MU_TOOLS_API_EXIT, &inParams);

    } else {
        // ★ 无追踪快速路径 — 直接执行，零开销
        status = muapiCtxGetId(ctx, ctxId);
    }

    return status;
}
```

### 2.2 关键设计点

| 设计点 | 实现 | 目的 |
|--------|------|------|
| **if/else 双路径** | 有订阅者时走追踪路径，无订阅者时直接执行 | 零开销无追踪模式 |
| **skipDriverImpl** | 工具在 ENTER 回调中设为 1 | 允许工具 mock/stub API 调用 |
| **correlationId** | 线程局部递增序列号 | ENTER/EXIT 匹配 |
| **params 结构体** | 每 API 在 `generated_musa_meta.h` 中有独立 struct | 类型安全 + ABI 稳定 |
| **CBID 枚举** | 812 个唯一 ID，从 1 开始 | O(1) 位图查表 |

### 2.3 ApiTrace — RAII 守卫

```cpp
// driver/internal.h:212-229
class ApiTrace {
public:
    template<typename ... Ts>
    ApiTrace(MUresult& status, const char* apiName,
             MUpti_driver_api_trace_cbid cbid, const Ts&... args)
        : m_Guard(apiName, static_cast<int>(cbid), args...)
        , m_Status(status) {}

    ~ApiTrace() noexcept {
        m_Guard.SetReturnValue(m_Status);
        if (m_Status != MUSA_SUCCESS && m_Status != MUSA_ERROR_NOT_READY) {
            GetTlsData().SetLastError(m_Status);   // 设置线程局部 last error
        }
    }
private:
    ApiInvocationGuard m_Guard;  // 调用状态追踪 + backtrace
    MUresult&          m_Status;
};
```

**ApiTrace 的双重职责**：
1. **错误时自动 backtrace**：`ApiInvocationGuard` 析构时如果 `m_Status` 非成功且非 `NOT_READY`，自动调用 `backtrace()` + `backtrace_symbols()` → stderr
2. **线程局部 last error**：`GetTlsData().SetLastError(m_Status)` 使得 `muGetLastError()` 可直接查询

---

## 三、Layer 2: Callback Subsystem — 订阅/分发机制

### 3.1 全局状态

```cpp
// callback.cpp:4-26
// 4 个追踪域的 CBID → enabled 位图
static constexpr uint32_t g_domainSize[MU_TOOLS_CB_DOMAIN_SIZE] = {
    0,                                  // MU_TOOLS_CB_DOMAIN_INVALID
    MUPTI_DRIVER_TRACE_CBID_SIZE,       // 812 — 驱动 API
    MUPTI_RUNTIME_TRACE_CBID_SIZE,      // — 运行时 API
    MU_TOOLS_CBID_RESOURCE_SIZE,        // 21 — 资源事件
    MU_TOOLS_CBID_SYNCHRONIZE_SIZE,     // 3  — 同步事件
};

uint32_t driverApiCallbackEnabled[812]   = {}; // 812 bits = 26 uint32
uint32_t runtimeApiCallbackEnabled[...]  = {};
uint32_t resourceCallbackEnabled[21]     = {};
uint32_t syncCallbackEnabled[3]          = {};

// 单一订阅者 (设计限制: 一次只允许一个外部工具)
std::atomic<uint32_t>       g_hasSubscriber{0};
static volatile MUtoolsCbSubscriber g_subscriber{};
```

### 3.2 追踪域 (Callback Domains)

| Domain | CBID 数 | 覆盖 |
|--------|---------|------|
| `MU_TOOLS_CB_DOMAIN_DRIVER_API` | **812** | 所有 `mu*` 驱动 API |
| `MU_TOOLS_CB_DOMAIN_RUNTIME_API` | ~200 | 所有 `musa*` 运行时 API |
| `MU_TOOLS_CB_DOMAIN_RESOURCE` | 21 | 资源创建/销毁事件 |
| `MU_TOOLS_CB_DOMAIN_SYNCHRONIZE` | 3 | 同步操作事件 |

### 3.3 toolsSubscribe — 注册订阅者 (CAS 保护)

```cpp
// callback.cpp:109-137
MUresult toolsSubscribe(MUtoolsCbSubscriberHandle* pHandle,
                        MUtoolsCbFunc_fn* callback,
                        void* userdata) {
    // 1. CAS 抢占订阅者槽位 (仅允许一个)
    oldValue = g_hasSubscriber.exchange(1);
    if (oldValue != 0) return MUSA_ERROR_UNKNOWN;

    // 2. seq-lock 写入:  seq++ → 写 userdata → seq++ → 写 callback
    pSubscriber->seq.fetch_add(1);
    pSubscriber->userdata = userdata;
    pSubscriber->seq.fetch_add(1);
    pSubscriber->callback.store(callback);

    *pHandle = MU_TOOLS_SUBSCRIBER_HANDLE;
}
```

### 3.4 toolsIssueCallback — seq-lock 安全回调

```cpp
// callback.cpp:40-55
void IssueCallback(MUtools_cb_domain domain, uint32_t cbid, const void* pParams) {
    // 1. 读 seq (偶数 = 无写入进行中)
    seqFirst  = pSubscriber->seq.load(seq_cst);
    userdata  = pSubscriber->userdata;
    callback  = pSubscriber->callback.load(seq_cst);
    seqSecond = pSubscriber->seq.load(seq_cst);

    // 2. seq 未变 + callback 非空 → 安全调用
    if (callback != nullptr && seqFirst == seqSecond) {
        callback(userdata, domain, cbid, pParams);
    }
}
```

### 3.5 toolsEnableCallback — 细粒度控制

```cpp
// 按单个 CBID 启用
toolsEnableCallback(1, handle, DRIVER_API, MUPTI_DRIVER_TRACE_CBID_muMemAlloc);

// 按域全部启用
toolsEnableAllCallbacksInDomain(1, handle, DRIVER_API);

// 启用所有域的所有 CBID
toolsEnableAllCallbacks(1, handle);
```

### 3.6 无订阅者时的零开销路径

```cpp
// callback.cpp:82-85
MUresult toolsIssueCallback(...) {
    if (g_hasSubscriber.load(seq_cst) == 0)
        return MUSA_ERROR_UNKNOWN;  // ← 立即返回，无锁竞争
    // ...
}

// mu_wrappers_generated.cpp 中
if (toolsCallbackEnabled(...)) {  // ← 检查 CBID 位图
    // 追踪路径 (只有启用时才执行)
} else {
    status = muapiXxx(args...);   // ← 无追踪直接执行
}
```

**关键**: `toolsCallbackEnabled` 是 O(1) 位图查询 —— `g_callbackEnabled[domain][cbid]`。即使没有订阅者，wrapper 也会跳过整个追踪代码块。

---

## 四、Layer 3: MUPTI Activity Trace — 运行时内部探针

### 4.1 导出表暴露的接口

```cpp
// export_table.h:727-830 (MUpti::ExportTable 部分)
namespace MUpti {

struct ThreadLocalInfoAccessors {
    GetThreadLocalInfo_fn* Get;
    GetProcessId_fn*       GetProcessId;
    GetThreadId_fn*        GetThreadId;
    GetCorrelationId_fn*   GetCorrelationId;
};

struct CommandBaseAccessors {
    CommandGetStream_fn*             GetStream;
    CommandGetCorrelationId_fn*      GetCorrelationId;
    CommandGetQueuedTimestamp_fn*    GetQueuedTimestamp;
    CommandGetType_fn*               GetType;
    CommandGetBeginEndTimestamp_fn*  GetBeginEndTimestamp;
    CommandGetSubmittedTimestamp_fn* GetSubmittedTimestamp;
    CommandCheckFromGraph_fn*        CheckFromGraph;
    CommandGetGraphId_fn*            GetGraphId;
    CommandGetGraphNodeId_fn*        GetGraphNodeId;
};

struct DispatchCommandAccessors {
    GetBase_fn*  GetBase;
    GetFunction_fn* GetFunction;
    GetGridSize_fn* GetGridSize;
    GetBlockSize_fn* GetBlockSize;
    GetDynamicSharedMemoryUsage_fn* GetDynamicSharedMemoryUsage;
};

struct MemcpyCommandAccessors {
    GetBase_fn*      GetBase;
    GetCopyKind_fn*  GetCopyKind;
    GetSrcMemKind_fn* GetSrcMemKind;
    GetDstMemKind_fn* GetDstMemKind;
    GetSize_fn*      GetSize;
};

struct MemsetCommandAccessors { ... };
struct MemoryTransferCommandAccessors { ... };
struct DeviceAccessors { GetId_fn* GetId; };
struct ContextAccessors { GetId_fn* GetId; GetDevice_fn* GetDevice; };
struct StreamAccessors { GetId_fn* GetId; GetContext_fn* GetContext; };
struct FunctionAccessors {
    GetName_fn* GetName;
    GetStaticSharedMemSize_fn* GetStaticSharedMemSize;
    GetDynamicSharedMemSize_fn* GetDynamicSharedMemSize;
    GetLocalMemSize_fn* GetLocalMemSize;
};
```

### 4.2 命令级 Activity Hook

每种命令类型在创建时触发对应的 MUPTI entry hook：

```cpp
// memsetCommand.cpp 构造函数中:
if (RecordMUptiActivity()) {
    m_ptiCtx = MUpti::EnterMemset(this);  // → 通知工具 "a memset command is created"
}

// memcpyCommand.cpp / dispatchCommand.cpp 同理:
if (RecordMUptiActivity()) {
    m_ptiCtx = MUpti::EnterMemcpy(this);    // memcpy
    m_ptiCtx = MUpti::EnterDispatch(this);  // kernel launch
}
```

工具通过 `export_table.h` 中的函数指针注册这些 hook：

```cpp
using EnterMemset_fn  = MUpti::Context*(Musa::MemsetCommand*);
using ExitMemset_fn   = void(MUpti::Context*);
using StartHostMemset_fn = void(uint64_t);
using StopHostMemset_fn  = void(uint64_t);

using EnterMemcpy_fn  = MUpti::Context*(Musa::MemcpyCommand*);
using EnterDispatch_fn = MUpti::Context*(Musa::DispatchCommand*);
using MarkKernelBeginEnd_fn = void(MUpti::Context*, uint64_t, uint64_t);
```

### 4.3 上下文生命周期 Hook

```cpp
using CreateContext_fn = void(MUcontext);   // Context 创建时
using DestroyContext_fn = void(MUcontext);  // Context 销毁时
using CreateStream_fn  = void(MUstream);    // Stream 创建时
using DestroyStream_fn = void(MUstream);    // Stream 销毁时
```

---

## 五、CBID 枚举 — 每个 API 的唯一 ID

`mupti_driver_cbid.h` 是一个**自动生成**的枚举，包含 **812 个条目**：

```c
typedef enum MUpti_driver_api_trace_cbid_enum {
    MUPTI_DRIVER_TRACE_CBID_INVALID                    = 0,
    MUPTI_DRIVER_TRACE_CBID_muInit                     = 1,
    MUPTI_DRIVER_TRACE_CBID_muDriverGetVersion         = 2,
    MUPTI_DRIVER_TRACE_CBID_muDeviceGet                = 3,
    // ... 812 entries total ...
    MUPTI_DRIVER_TRACE_CBID_SIZE                       = 812,
};
```

**命名规则**: `MUPTI_DRIVER_TRACE_CBID_` + API 函数名。每个 API 变体（`_v2`, `_ptsz`, `_async`）获得独立 CBID。

**CBID → 函数名映射**: 通过 `#define MUPTI_CBID_GET(name) MUPTI_DRIVER_TRACE_CBID_##name` 宏在 wrapper 中展开。

---

## 六、使用方式 — 外部工具侧

### 6.1 完整订阅流程

```cpp
// ===== Step 1: 获取 Export Table =====
Driver::ExportTable* pDriverTable = nullptr;
muGetExportTable((const void**)&pDriverTable, &kDriverTableUuid);

MUpti::ExportTable* pMuptiTable = nullptr;
muGetExportTable((const void**)&pMuptiTable, &kMuptiTableUuid);

// ===== Step 2: 获取工具操作函数 =====
Tools::ExportTable* pToolsTable = nullptr;
muGetExportTable((const void**)&pToolsTable, &kToolsTableUuid);

// ===== Step 3: 定义回调函数 =====
void MyTraceCallback(void* userdata, MUtools_cb_domain domain,
                     uint32_t cbid, const void* pParams) {
    auto* params = static_cast<const MUtoolsTraceApiMusa*>(pParams);

    if (params->apiEnterOrExit == MU_TOOLS_API_ENTER) {
        printf("[ENTER] %s (cbid=%u, corr=%u)\n",
               params->functionName, cbid, *params->pCorrelationId);

        // ★ 可选: 拦截执行
        // *params->pSkipDriverImpl = 1;
        // 然后直接设置 *params->pStatus = MUSA_SUCCESS;
    } else {
        printf("[EXIT]  %s → %d\n",
               params->functionName, *params->pStatus);
    }
}

// ===== Step 4: 订阅 =====
MUtoolsCbSubscriberHandle handle;
pToolsTable->subscribe_fn(&handle, MyTraceCallback, nullptr);

// ===== Step 5: 启用追踪 =====
pToolsTable->enable_fn(1, handle, MU_TOOLS_CB_DOMAIN_DRIVER_API, 1);  // muInit
pToolsTable->enable_fn(1, handle, MU_TOOLS_CB_DOMAIN_DRIVER_API, 29); // muMemAlloc
// 或全量启用:
pToolsTable->enableAllInDomain_fn(1, handle, MU_TOOLS_CB_DOMAIN_DRIVER_API);

// ===== Step 6: 执行被追踪的代码 =====
muInit(0);
muMemAlloc(&dptr, 4096);
// → MyTraceCallback 收到 ENTER muInit → EXIT muInit
// → MyTraceCallback 收到 ENTER muMemAlloc → EXIT muMemAlloc

// ===== Step 7: 取消订阅 =====
pToolsTable->unsubscribe_fn(handle);
```

### 6.2 真实使用案例 — muCommandAccessors 测试

```cpp
// musa/unittest/driver/Entry_Point_Access/muCommandAccessors.cpp

// 定义 hook 函数 — 拦截 MemcpyCommand 创建
static MUpti::Context* EnterMemcpyCallbackForSize(MemcpyCommand* command) {
    g_capturedMemcpyCommand.store(command);
    if (command->GetCopyParams().copySize == 1024) {
        g_accessorTestPassed.store(true);
    }
    return nullptr;
}

TEST(TestMemcpyCommandGetSize, Test) {
    // 1. 注册 hook
    MUpti::ExportTable* muptiTable = ...;
    muptiTable->MemcpyCommand->EnterMemcpy = EnterMemcpyCallbackForSize;

    // 2. 创建 context + stream
    muCtxCreate(&ctx, 0, 0);
    muStreamCreate(&stream, 0);

    // 3. 分配 + 执行 memcpy (触发 hook)
    muMemAlloc(&src, 1024);
    muMemAlloc(&dst, 1024);
    muMemcpyAsync(dst, src, 1024, stream);  // ← EnterMemcpyCallbackForSize 被调用
    muStreamSynchronize(stream);

    // 4. 验证 hook 捕获的数据
    EXPECT_NE(g_capturedMemcpyCommand.load(), nullptr);
    EXPECT_TRUE(g_accessorTestPassed.load());
}
```

### 6.3 进程注入方式

```cpp
// 通过环境变量 MUSA_INJECTION64_PATH 注入 profiler 库
// internal.cpp: InjectionManager
InjectionManager::Init() {
    const char* injectionPath = getenv("MUSA_INJECTION64_PATH");
    if (injectionPath) {
        void* lib = dlopen(injectionPath, RTLD_NOW);
        auto initFn = (InitializeInjection_fn*)dlsym(lib, "InitializeInjection");
        initFn(&g_ExportTable);  // 传递 export table 给注入库
    }
}
```

---

## 七、性能设计

### 7.1 无订阅者零开销

```
无订阅者:  toolsCallbackEnabled → g_callbackEnabled[domain][cbid] → 0
           → 直接走 else 分支 → muapiXxx(args...) → 无任何额外开销

有订阅者:  toolsCallbackEnabled → g_callbackEnabled[domain][cbid] → 1
           → 构造 params 结构体 → ENTER callback → muapiXxx → EXIT callback
           → 额外开销: params 构造 + 2 次函数指针调用
```

### 7.2 热点路径分析

| 操作 | 无订阅者 | 有订阅者 |
|------|---------|---------|
| O(1) 位图查询 | ✅ `g_callbackEnabled[domain][cbid]` | — |
| 无锁读取 | ✅ 无原子操作 | — |
| 函数调用 | 直接 muapiXxx | +2 次 callback |
| 内存分配 | 无 | params struct (栈) |
| 原子操作 | 无 | correlationId 自增 |

### 7.3 单订阅者限制

```
g_hasSubscriber.compare_exchange → 保证只有一个外部工具注册
原因: 多工具同时订阅会导致回调顺序不确定 + 性能不可控
替代: 工具间通过 injection 层自己的多路分发实现
```

---

## 八、与 CUDA CUPTI 的对应关系

| CUDA | MUSA | 说明 |
|------|------|------|
| `CUpti_ActivityKind` | `MUpti::ExportTable` 中的 activity hook | 命令级事件 |
| `CUpti_CallbackId` | `MUpti_driver_api_trace_cbid` 枚举 | API 级追踪 |
| `CUpti_Subscribe` / `CUpti_EnableCallback` | `toolsSubscribe` / `toolsEnableCallback` | 订阅/启用 |
| `CUpti_CallbackFunc` | `MUtoolsCbFunc_fn` | 回调签名 |
| `CUpti_ActivityAPI` | `MUtoolsTraceApiMusa` | API 追踪参数 |
| `cuptiGetTimestamp` | `util/TimeStamp.h` (MUSA 内部) | 时间戳 |

---

## 九、关键文件速查

| 文件 | 行数 | 作用 |
|------|------|------|
| `driver/mu_wrappers_generated.cpp` | 22,628 | 自动生成的 wrapper (每个 API 一个函数，含 trace 探针) |
| `driver/callback.cpp` | 244 | 订阅/分发/启用/禁用核心逻辑 |
| `driver/callback.h` | 12 | tools* 函数声明 |
| `driver/internal.h:73-229` | — | ApiInvocationGuard + ApiTrace RAII 类 |
| `driver/mu_entry.cpp` | 1942 | 导出表初始化，tools 函数注册 |
| `musa_shared_include/export_table.h` | 1686 | 导出表类型定义 (MUpti, Tools, Driver 等) |
| `musa_shared_include/mupti/mupti_driver_cbid.h` | 829 | 812 个 CBID 枚举 |
| `musa_shared_include/generated_musa_meta.h` | 3945 | 每 API 的 params 结构体 |
| `musa/core/command/memsetCommand.cpp` | 339 | MemsetCommand: MUpti::EnterMemset 调用点 |
| `musa/core/command/memcpyCommand.cpp` | — | MemcpyCommand: MUpti::EnterMemcpy 调用点 |
| `musa/core/command/dispatchCommand.cpp` | — | DispatchCommand: MUpti::EnterDispatch 调用点 |
| `MUSA-Runtime/src/musaapi.h` | 350 | 运行时内部 API 声明 |
| `MUSA-Runtime/src/internal.cpp` | 2388 | 运行时 InjectionManager |
