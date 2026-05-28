# MUPTI (MUSA Profiling Tools Interface) — 完整设计分析

> **类比**: MUPTI 就是 MUSA 的 "CUDA Profiling Tools Interface" (CUPTI)。它允许外部性能分析工具（如 Nsight Systems、自定义 profiler）在不修改用户代码的情况下，拦截和监控所有 MUSA API 调用和 GPU 命令执行。

---

## 一、MUPTI 是什么？

MUPTI 是 MUSA 驱动层暴露给外部 profiling/debugging 工具的 **回调拦截框架**。

### 核心能力

| 能力 | 说明 |
|------|------|
| **API 拦截** | 每个 `muXxx()` 调用前后都会触发 ENTER/EXIT 回调 |
| **命令级追踪** | Memcpy、Memset、Dispatch（kernel launch）等 GPU 命令的执行可以被追踪 |
| **资源生命周期** | Context/Stream 创建销毁、Module 加载卸载等事件通知 |
| **Graph 追踪** | Graph 执行、Graph 节点级别的 begin/end 标记 |
| **同步事件** | StreamSynchronize、EventSynchronize、ContextSynchronize 的注册/开始/结束 |
| **多订阅者** | 最多 4 个工具同时订阅，广播分发 |
| **跳过驱动** | profiler 可以设置 `skipDriverImpl=1` 来阻止实际驱动执行（用于 mock 测试） |

### 设计哲学

```
用户代码调用 muMemAlloc()
        │
        ▼
┌─────────────────────────────────────────┐
│  mu_wrappers_generated.cpp (自动生成的包装器) │
│  1. 检查 toolsCallbackEnabled()          │
│  2. 如果启用 → toolsIssueCallback(ENTER)  │
│  3. 执行实际驱动实现 muapiMemAlloc()       │
│  4. toolsIssueCallback(EXIT)             │
└─────────────────────────────────────────┘
        │
        ▼
   实际驱动逻辑
```

**关键设计**: 当没有 profiler 订阅时，`toolsCallbackEnabled()` 返回 0，代码直接走 `else` 分支调用驱动实现，**零开销**。

---

## 二、整体架构：三层拦截体系（Driver 层）

> ⚠️ 以下描述的"三层拦截"**仅针对 Driver 层** (`libmusa.so`)。Runtime 层 (`libmusa_runtime.so`) 有自己的 MUPTI 框架，但只包含 API 级回调（层 1），没有命令级 intercept 和 Accessor 导出表。

MUPTI Driver 层由三个正交的拦截层组成：

### 层 1: API 级回调 (Callback Framework)

```
文件: src/driver/callback.cpp, callback.h
      src/driver/mu_wrappers_generated.cpp (自动生成)
```

这是最上层的拦截，针对每个 `mu*` API 函数。

#### 核心数据结构

```cpp
// export_table.h — 订阅者结构
typedef struct MUtoolsCbSubscriber_st {
    std::atomic<MUtoolsCbFunc_fn*> callback{nullptr};  // 回调函数指针
    std::atomic<uint32_t> seq{0};                       // 序列号（无锁一致性协议）
    void* userdata;                                     // 用户自定义数据
} MUtoolsCbSubscriber;

// 回调函数签名
using MUtoolsCbFunc_fn = void(void* userdata, MUtools_cb_domain domain, uint32_t cbid, const void* pParams);

// 回调传递的参数结构
typedef struct MUtoolsTraceApiMusa_st {
    uint32_t struct_size;
    uint32_t* pCorrelationId;      // 关联 ID（用于匹配 ENTER/EXIT）
    MUresult* pStatus;             // 返回值指针
    const char* functionName;      // 函数名，如 "muMemAlloc"
    uint32_t functionId;           // CBID 枚举值
    uint64_t contextId;            // 当前 Context 的序列 ID
    void* params;                  // 函数参数结构体指针
    MUcontext context;             // 当前 TLS Context
    MUtools_api_enter_exit apiEnterOrExit;  // ENTER 或 EXIT
    uint32_t* pSkipDriverImpl;     // 如果设为 1，跳过驱动实现
} MUtoolsTraceApiMusa;
```

#### 回调域 (Domain)

```cpp
typedef enum MUtools_cb_domain_enum {
    MU_TOOLS_CB_DOMAIN_INVALID      = 0,
    MU_TOOLS_CB_DOMAIN_DRIVER_API   = 1,   // Driver API (mu* 函数)
    MU_TOOLS_CB_DOMAIN_RUNTIME_API  = 2,   // Runtime API (musa* 函数)
    MU_TOOLS_CB_DOMAIN_RESOURCE     = 3,   // 资源生命周期事件
    MU_TOOLS_CB_DOMAIN_SYNCHRONIZE  = 4,   // 同步事件
    MU_TOOLS_CB_DOMAIN_SIZE         = 5,
} MUtools_cb_domain;
```

#### 启用/禁用机制

```cpp
// 每个域 × 每个 CBID 都有一个 uint32_t 标志位
uint32_t driverApiCallbackEnabled[MUPTI_DRIVER_TRACE_CBID_SIZE] = {};
uint32_t runtimeApiCallbackEnabled[MUPTI_RUNTIME_TRACE_CBID_SIZE] = {};
uint32_t resourceCallbackEnabled[MU_TOOLS_CBID_RESOURCE_SIZE] = {};
uint32_t syncCallbackEnabled[MU_TOOLS_CBID_SYNCHRONIZE_SIZE] = {};
```

**快速路径**: `toolsCallbackEnabled(handle, domain, cbid)` 直接读取数组中的值，如果为 0 则跳过所有回调逻辑。

#### 多订阅者广播

```cpp
#define MAX_TOOLS_SUBSCRIBERS 4

static volatile MUtoolsCbSubscriber g_subscribers[MAX_TOOLS_SUBSCRIBERS] = {};
static std::atomic<uint32_t> g_subscriberCount{0};

void IssueCallback(MUtools_cb_domain domain, uint32_t cbid, const void* pParams) {
    // 广播给所有活跃的订阅者
    for (int i = 0; i < MAX_TOOLS_SUBSCRIBERS; i++) {
        volatile MUtoolsCbSubscriber* pSubscriber = &g_subscribers[i];
        MUtoolsCbFunc_fn* callback = nullptr;
        void* userdata = nullptr;
        uint32_t seqFirst, seqSecond;

        // 无锁一致性读取协议：
        // 写入方: seq++ → 写 userdata → seq++ → 写 callback
        // 读取方: 读 seqFirst → 读 userdata → 读 callback → 读 seqSecond
        //         如果 seqFirst == seqSecond，说明读取是一致的
        seqFirst  = pSubscriber->seq.load(std::memory_order_seq_cst);
        userdata  = pSubscriber->userdata;
        callback  = pSubscriber->callback.load(std::memory_order_seq_cst);
        seqSecond = pSubscriber->seq.load(std::memory_order_seq_cst);

        if (callback != nullptr && seqFirst == seqSecond) {
            callback(userdata, domain, cbid, pParams);
        }
    }
}
```

### 层 2: MUpti Driver Hooks (命令级拦截)

```
文件: src/driver/mupti/hooks.h, hooks.cpp
      src/driver/mupti/tracepoints.h
```

这是更底层的拦截，针对 GPU 命令（Memcpy、Memset、Dispatch 等）的执行。

#### 核心结构

```cpp
// hooks.h — 全局 hook 存储
struct MUptiDriverHookStorage {
    std::atomic<bool> ready;           // MUPTI 是否已启用
    MUpti::MUptiDriverHooks hooks;     // 所有 hook 函数指针
    uintptr_t reserved[8];
};

extern MUptiDriverHookStorage G_MUPTI_DRIVER_HOOKS;
```

#### MUptiDriverHooks — 60 个 hook 函数

```cpp
struct MUptiDriverHooks {
    // === 遗留接口 (legacy) ===
    ADDMEMBER(EnterDriverApi_fn*, EnterDriverApi);
    ADDMEMBER(ExitDriverApi_fn*, ExitDriverApi);
    ADDMEMBER(EnterRuntimeApi_fn*, EnterRuntimeApi);
    ADDMEMBER(ExitRuntimeApi_fn*, ExitRuntimeApi);
    ADDMEMBER(EnterMemcpy_fn*, EnterMemcpy);
    ADDMEMBER(ExitMemcpy_fn*, ExitMemcpy);
    ADDMEMBER(RegisterKernel_fn*, RegisterKernel);
    ADDMEMBER(AssignKernelToKick_fn*, AssignKernelToKick);
    ADDMEMBER(StartHostMemcpy_fn*, StartHostMemcpy);
    ADDMEMBER(StopHostMemcpy_fn*, StopHostMemcpy);
    ADDMEMBER(CreateContext_fn*, CreateContext);
    ADDMEMBER(DestroyContext_fn*, DestroyContext);
    ADDMEMBER(CreateStream_fn*, CreateStream);
    ADDMEMBER(DestroyStream_fn*, DestroyStream);
    ADDMEMBER(EnterMemset_fn*, EnterMemset);
    ADDMEMBER(ExitMemset_fn*, ExitMemset);
    ADDMEMBER(StartHostMemset_fn*, StartHostMemset);
    ADDMEMBER(StopHostMemset_fn*, StopHostMemset);
    ADDMEMBER(RegisterStreamWaitEvent_fn*, RegisterStreamWaitEvent);
    ADDMEMBER(StartStreamWaitEvent_fn*, StartStreamWaitEvent);
    ADDMEMBER(StopStreamWaitEvent_fn*, StopStreamWaitEvent);
    ADDMEMBER(MarkKernelQueued_fn*, MarkKernelQueued);
    ADDMEMBER(MarkKernelSubmitted_fn*, MarkKernelSubmitted);

    // === 新增接口 ===
    ADDMEMBER(AssignSubmissionToCorrelation_fn*, AssignSubmissionToCorrelation);
    ADDMEMBER(RegisterEventSynchronize_fn*, RegisterEventSynchronize);
    ADDMEMBER(StartEventSynchronize_fn*, StartEventSynchronize);
    ADDMEMBER(StopEventSynchronize_fn*, StopEventSynchronize);
    ADDMEMBER(RegisterStreamSynchronize_fn*, RegisterStreamSynchronize);
    ADDMEMBER(StartStreamSynchronize_fn*, StartStreamSynchronize);
    ADDMEMBER(StopStreamSynchronize_fn*, StopStreamSynchronize);
    ADDMEMBER(RegisterContextSynchronize_fn*, RegisterContextSynchronize);
    ADDMEMBER(StartContextSynchronize_fn*, StartContextSynchronize);
    ADDMEMBER(StopContextSynchronize_fn*, StopContextSynchronize);
    ADDMEMBER(RegisterKernelV2_fn*, RegisterKernelV2);
    ADDMEMBER(MarkKernelBeginEnd_fn*, MarkKernelBeginEnd);
    ADDMEMBER(SetMemset3DCounter_fn*, SetMemset3DCounter);
    ADDMEMBER(MarkMemcpyBeginEnd_fn*, MarkMemcpyBeginEnd);
    ADDMEMBER(MarkMemsetBeginEnd_fn*, MarkMemsetBeginEnd);
    ADDMEMBER(RegisterGraphTrace_fn*, RegisterGraphTrace);
    ADDMEMBER(MarkGraphTraceBegin_fn*, MarkGraphTraceBegin);
    ADDMEMBER(MarkGraphTraceEnd_fn*, MarkGraphTraceEnd);
    ADDMEMBER(RegisterGraphKernel_fn*, RegisterGraphKernel);
    ADDMEMBER(RegisterGraphMemcpy_fn*, RegisterGraphMemcpy);
    ADDMEMBER(RegisterGraphMemset_fn*, RegisterGraphMemset);
    ADDMEMBER(RegisterGraphMemoryAtomic_fn*, RegisterGraphMemoryAtomic);
    ADDMEMBER(RegisterGraphMemoryAtomicValue_fn*, RegisterGraphMemoryAtomicValue);
    ADDMEMBER(EnterMemoryAtomic_fn*, EnterMemoryAtomic);
    ADDMEMBER(MarkMemoryAtomicBeginEnd_fn*, MarkMemoryAtomicBeginEnd);
    ADDMEMBER(EnterMemoryAtomicValue_fn*, EnterMemoryAtomicValue);
    ADDMEMBER(MarkMemoryAtomicValueBeginEnd_fn*, MarkMemoryAtomicValueBeginEnd);
    ADDMEMBER(MarkCommandBeginEnd_fn*, MarkCommandBeginEnd);
    ADDMEMBER(MarkGraphNodeBeginEnd_fn*, MarkGraphNodeBeginEnd);
    ADDMEMBER(CheckGraphTraceEnabled_fn*, CheckGraphTraceEnabled);
    ADDMEMBER(MarkGraphNodeBeginEndV2_fn*, MarkGraphNodeBeginEndV2);
    ADDMEMBER(EnterMemoryTransfer_fn*, EnterMemoryTransfer);
    ADDMEMBER(RegisterGraphMemoryTransfer_fn*, RegisterGraphMemoryTransfer);
    ADDMEMBER(EnterMemoryAtomicV2_fn*, EnterMemoryAtomicV2);
    ADDMEMBER(EnterMemoryTransferV2_fn*, EnterMemoryTransferV2);
    ADDMEMBER(MarkCommandBeginEndV2_fn*, MarkCommandBeginEndV2);
};
```

#### Tracepoints — 内联检查函数

```cpp
// tracepoints.h — 每个 tracepoint 都是内联函数，检查 G_MUPTI_DRIVER_HOOKS.ready
namespace MUpti {

inline bool CheckMUptiEnabled() {
    return G_MUPTI_DRIVER_HOOKS.ready.load(std::memory_order_acquire);
}

inline MUpti::Context* EnterMemcpy(Musa::MemcpyCommand* command) {
    if (CheckMUptiEnabled()) {
        return G_MUPTI_DRIVER_HOOKS.hooks.EnterMemcpy(command);
    }
    return nullptr;
}

inline void ExitMemcpy(MUpti::Context* context) {
    if (context != nullptr) {
        G_MUPTI_DRIVER_HOOKS.hooks.ExitMemcpy(context);
    }
}

// ... 其他 58 个 tracepoint 函数
}
```

**性能优化**: `CheckMUptiEnabled()` 是 `atomic<bool>` 的 load 操作，当 MUPTI 未启用时，所有 tracepoint 函数直接返回，开销极小。

### 层 3: Accessor 导出表 (工具 introspection)

```
文件: src/musa_shared_include/export_table.h
```

MUPTI 通过 `muGetExportTable()` 向外部工具暴露 **Accessor 函数指针表**，让工具可以读取内部对象（Command、Context、Stream 等）的信息，而无需知道内部实现细节。

```cpp
struct DriverExportTable {
    MUptiDriverControllers* MUpti;        // Enable/Disable MUPTI
    ThreadLocalInfoAccessors* TidInfo;    // 线程本地信息
    CommandBaseAccessors* CommandBase;    // Command 基本信息
    DispatchCommandAccessors* DispatchCommand; // Kernel dispatch 信息
    MemcpyCommandAccessors* MemcpyCommand;     // Memcpy 信息
    MemsetCommandAccessors* MemsetCommand;     // Memset 信息
    DeviceAccessors* Device;
    ContextAccessors* Context;
    StreamAccessors* Stream;
    FunctionAccessors* Function;
    ProfilerControllers* Profiler;
    EventAccessors* Event;
    GraphAccessors* Graph;
    MemoryAtomicCommandAccessors* MemoryAtomicCommand;
    MemoryAtomicValueCommandAccessors* MemoryAtomicValueCommand;
    MemoryTransferCommandAccessors* MemoryTransferCommand;
    PCSamplingControllers* PCSampling;
    // ...
};
```

---

## 三、MUPTI 的完整生命周期

### 阶段 1: 初始化

```
用户调用 muInit(Flags)
    │
    ▼
mu_wrappers_generated.cpp: muInit()
    │
    ├── toolsIssueCallback(MU_TOOLS_API_ENTER, muInit)  ← 如果 profiler 已订阅
    │
    ├── g_InjectionManager.Init()                        ← 加载注入库
    │       │
    │       └── 如果注入库存在 → 调用 MUpti::MUptiDriverControllers::Enable(initer)
    │               │
    │               └── initer(&G_MUPTI_DRIVER_HOOKS.hooks)  ← 填充 hook 函数指针
    │               └── G_MUPTI_DRIVER_HOOKS.ready = true       ← 启用 MUPTI
    │
    ├── muapiInit(Flags)                                 ← 实际驱动初始化
    │
    └── toolsIssueCallback(MU_TOOLS_API_EXIT, muInit)   ← EXIT 回调
```

### 阶段 2: 运行时拦截

#### API 级拦截流程（以 muMemAlloc 为例）

```cpp
// mu_wrappers_generated.cpp 中的自动生成代码
MUresult MUSAAPI muMemAlloc(MUdeviceptr_v1 *dptr, unsigned int bytesize) {
    MUresult status = MUSA_SUCCESS;
    ApiTrace trace(status, "muMemAlloc", MUPTI_CBID_GET(muMemAlloc), dptr, bytesize);

    // 步骤 1: 快速检查 — 这个 CBID 是否被订阅？
    if (toolsCallbackEnabled(MU_TOOLS_SUBSCRIBER_HANDLE, MU_TOOLS_CB_DOMAIN_DRIVER_API,
                             MUPTI_DRIVER_TRACE_CBID_muMemAlloc)) {

        MUtoolsTraceApiMusa inParams{};
        uint32_t correlationId = Util::ThreadInfo::Get().ApiSeqNum();  // 线程唯一的关联 ID
        uint32_t skipDriverImpl = 0;                                    // 默认不跳过

        muMemAlloc_params params{};
        params.dptr = dptr;
        params.bytesize = bytesize;

        inParams.struct_size = sizeof(inParams);
        inParams.pCorrelationId = &correlationId;
        inParams.pStatus = &status;
        inParams.functionName = "muMemAlloc";
        inParams.functionId = MUPTI_DRIVER_TRACE_CBID_muMemAlloc;
        inParams.context = TlsCtxTop();  // 当前线程的 Context
        inParams.contextId = inParams.context != nullptr ?
            (Musa::IntrusiveCast<Musa::Context>(inParams.context))->GetSeqID() : 0;
        inParams.params = &params;
        inParams.apiEnterOrExit = MU_TOOLS_API_ENTER;
        inParams.pSkipDriverImpl = &skipDriverImpl;

        // 步骤 2: 发送 ENTER 回调
        toolsIssueCallback(MU_TOOLS_SUBSCRIBER_HANDLE, MU_TOOLS_CB_DOMAIN_DRIVER_API,
                          MUPTI_DRIVER_TRACE_CBID_muMemAlloc, &inParams);

        // 步骤 3: 如果 profiler 要求跳过，就不执行驱动实现
        if (!skipDriverImpl) {
            status = muapiMemAlloc(params.dptr, params.bytesize);
        }

        // 步骤 4: 发送 EXIT 回调
        inParams.apiEnterOrExit = MU_TOOLS_API_EXIT;
        toolsIssueCallback(MU_TOOLS_SUBSCRIBER_HANDLE, MU_TOOLS_CB_DOMAIN_DRIVER_API,
                          MUPTI_DRIVER_TRACE_CBID_muMemAlloc, &inParams);
    } else {
        // 快速路径: 没有订阅者，直接调用驱动
        status = muapiMemAlloc(dptr, bytesize);
    }

    return status;
}
```

#### 命令级拦截流程（以 Memcpy 为例）

```
Stream::CmdMemcpy() 创建 MemcpyCommand
    │
    ▼
MemcpyCommand::Submit()
    │
    ├── MUpti::Context* ctx = MUpti::EnterMemcpy(this);  ← 如果 MUPTI 启用
    │       │
    │       └── G_MUPTI_DRIVER_HOOKS.hooks.EnterMemcpy(command)
    │               → profiler 记录开始时间、命令信息
    │
    ├── 实际执行 memcpy (DMA/CE/CPU)
    │
    ├── MUpti::StartHostMemcpy(correlationId);  ← 如果是 Host memcpy
    │       └── G_MUPTI_DRIVER_HOOKS.hooks.StartHostMemcpy(correlationId)
    │
    ├── ... memcpy 完成 ...
    │
    ├── MUpti::StopHostMemcpy(correlationId);
    │
    └── MUpti::ExitMemcpy(ctx);  ← 如果 ctx != nullptr
            └── G_MUPTI_DRIVER_HOOKS.hooks.ExitMemcpy(ctx)
                    → profiler 记录结束时间
```

### 阶段 3: 资源事件通知

当资源（Context、Stream、Module 等）创建或销毁时，通过 `MU_TOOLS_CB_DOMAIN_RESOURCE` 域发送事件：

```cpp
// 资源事件 CBID (export_table.h)
typedef enum MUtools_cbid_resource_enum {
    MU_TOOLS_CBID_RESOURCE_INVALID                  = 0,
    MU_TOOLS_CBID_RESOURCE_CONTEXT_CREATED           = 1,
    MU_TOOLS_CBID_RESOURCE_CONTEXT_DESTROY_STARTING  = 2,
    MU_TOOLS_CBID_RESOURCE_STREAM_CREATED            = 3,
    MU_TOOLS_CBID_RESOURCE_STREAM_DESTROY_STARTING   = 4,
    MU_TOOLS_CBID_RESOURCE_MU_INIT_FINISHED          = 5,
    MU_TOOLS_CBID_RESOURCE_MODULE_LOADED             = 6,
    MU_TOOLS_CBID_RESOURCE_MODULE_UNLOAD_STARTING    = 7,
    MU_TOOLS_CBID_RESOURCE_MODULE_PROFILED           = 8,
    MU_TOOLS_CBID_RESOURCE_GRAPH_CREATED             = 9,
    MU_TOOLS_CBID_RESOURCE_GRAPH_DESTROY_STARTING    = 10,
    MU_TOOLS_CBID_RESOURCE_GRAPH_CLONED              = 11,
    MU_TOOLS_CBID_RESOURCE_GRAPHNODE_CREATE_STARTING = 12,
    MU_TOOLS_CBID_RESOURCE_GRAPHNODE_CREATED         = 13,
    MU_TOOLS_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING = 14,
    MU_TOOLS_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_CREATED = 15,
    MU_TOOLS_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_DESTROY_STARTING = 16,
    MU_TOOLS_CBID_RESOURCE_GRAPHEXEC_CREATE_STARTING = 17,
    MU_TOOLS_CBID_RESOURCE_GRAPHEXEC_CREATED         = 18,
    MU_TOOLS_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING = 19,
    MU_TOOLS_CBID_RESOURCE_GRAPHNODE_CLONED          = 20,
} MUtools_cbid_resource;
```

### 阶段 4: 同步事件追踪

```cpp
// 同步事件有 3 阶段: Register → Start → Stop
// 以 StreamSynchronize 为例:

MUpti::Context* ctx = MUpti::RegisterStreamSynchronize(correlationId, stream);
    → profiler 记录 "stream sync 已注册"

MUpti::StartStreamSynchronize(ctx);
    → profiler 记录 "stream sync 开始等待"

// ... 实际等待 stream 完成 ...

MUpti::StopStreamSynchronize(ctx);
    → profiler 记录 "stream sync 完成"
```

---

## 四、CBID 枚举体系

MUPTI 使用 **CBID (Callback ID)** 来唯一标识每个可拦截的 API 函数。

### Driver API CBID

```cpp
// mupti_driver_cbid.h — 自动生成的枚举
typedef enum MUpti_driver_api_trace_cbid_enum {
    MUPTI_DRIVER_TRACE_CBID_INVALID           = 0,
    MUPTI_DRIVER_TRACE_CBID_muInit            = 1,
    MUPTI_DRIVER_TRACE_CBID_muDriverGetVersion = 2,
    MUPTI_DRIVER_TRACE_CBID_muDeviceGet       = 3,
    MUPTI_DRIVER_TRACE_CBID_muDeviceGetCount  = 4,
    // ... 共 700+ 个 CBID，覆盖所有 mu* 函数
    MUPTI_DRIVER_TRACE_CBID_muLaunchKernel    = 307,
    MUPTI_DRIVER_TRACE_CBID_muMemAlloc_v2     = 243,
    MUPTI_DRIVER_TRACE_CBID_muMemFree_v2      = 245,
    MUPTI_DRIVER_TRACE_CBID_muMemcpyAsync     = 306,
    // ...
    MUPTI_DRIVER_TRACE_CBID_SIZE              // 总数
} MUpti_driver_api_trace_cbid;
```

### Runtime API CBID

```cpp
// mupti_runtime_cbid.h
typedef enum MUpti_runtime_api_trace_cbid_enum {
    MUPTI_RUNTIME_TRACE_CBID_INVALID          = 0,
    MUPTI_RUNTIME_TRACE_CBID_musaInit         = 1,
    MUPTI_RUNTIME_TRACE_CBID_musaMalloc       = 2,
    MUPTI_RUNTIME_TRACE_CBID_musaFree         = 3,
    MUPTI_RUNTIME_TRACE_CBID_musaMemcpy       = 4,
    MUPTI_RUNTIME_TRACE_CBID_musaMemcpyAsync  = 5,
    MUPTI_RUNTIME_TRACE_CBID_musaLaunchKernel = 6,
    // ... 覆盖所有 musa* 函数
} MUpti_runtime_api_trace_cbid;
```

---

## 五、实际使用示例

### 示例 1: 外部 Profiler 订阅 API 回调

```cpp
// 这是外部 profiler 工具（如 musaProfile）的代码
#include "export_table.h"
#include "driver/callback.h"

// 步骤 1: 定义回调函数
void MyProfilerCallback(void* userdata, MUtools_cb_domain domain,
                        uint32_t cbid, const void* pParams) {
    const MUtoolsTraceApiMusa* params = static_cast<const MUtoolsTraceApiMusa*>(pParams);

    if (params->apiEnterOrExit == MU_TOOLS_API_ENTER) {
        printf("[ENTER] %s (correlationId=%u, contextId=%lu)\n",
               params->functionName, *params->pCorrelationId, params->contextId);
    } else {
        printf("[EXIT]  %s (status=%d)\n", params->functionName, *params->pStatus);
    }
}

// 步骤 2: 订阅
MUtoolsCbSubscriberHandle handle;
MUresult result = toolsSubscribe(&handle, MyProfilerCallback, nullptr);
// result == MUSA_SUCCESS, handle == 1

// 步骤 3: 启用特定 API 的回调
toolsEnableCallback(1, handle, MU_TOOLS_CB_DOMAIN_DRIVER_API,
                    MUPTI_DRIVER_TRACE_CBID_muMemAlloc_v2);
toolsEnableCallback(1, handle, MU_TOOLS_CB_DOMAIN_DRIVER_API,
                    MUPTI_DRIVER_TRACE_CBID_muMemFree_v2);
toolsEnableCallback(1, handle, MU_TOOLS_CB_DOMAIN_DRIVER_API,
                    MUPTI_DRIVER_TRACE_CBID_muLaunchKernel);

// 或者启用整个域
toolsEnableAllCallbacksInDomain(1, handle, MU_TOOLS_CB_DOMAIN_DRIVER_API);

// 步骤 4: 运行用户程序...
// 此时每个 muMemAlloc/muMemFree/muLaunchKernel 调用都会触发回调

// 步骤 5: 清理
toolsUnsubscribe(handle);
```

### 示例 2: 通过 Export Table 获取 Command 信息

```cpp
// Profiler 获取 DispatchCommand 的详细信息
const DriverExportTable* exportTable = nullptr;
muGetExportTable((const void**)&exportTable, &kDriverExportTableUuid);

// 获取 kernel 名称
const char* kernelName = exportTable->DispatchCommand->GetFunction(dispatchCmd);

// 获取 grid/block 大小
dim3 gridSize = exportTable->DispatchCommand->GetGridSize(dispatchCmd);
dim3 blockSize = exportTable->DispatchCommand->GetBlockSize(dispatchCmd);

// 获取动态共享内存使用量
uint64_t dynamicSmem = exportTable->DispatchCommand->GetDynamicSharedMemoryUsage(dispatchCmd);

// 获取命令的时间戳
uint64_t beginTime, endTime;
exportTable->CommandBase->GetBeginEndTimestampV2(dispatchCmd, index, &beginTime, &endTime);
```

### 示例 3: MUpti Driver Hooks 的启用

```cpp
// 这是 MUSA-runtime 中 InjectionManager 的代码流程
void InjectionManager::Init() {
    // 1. 尝试加载注入库 (如 libmusaProfile.so)
    void* lib = dlopen("libmusaProfile.so", RTLD_NOW);
    if (!lib) return;

    // 2. 获取 MUpti 控制器
    auto getExportTable = (MUresult(*)(const void**, const MUuuid*))
        dlsym(lib, "muGetExportTable");

    MUpti::DriverExportTable* driverExport = nullptr;
    getExportTable((const void**)&driverExport, &kDriverExportTableUuid);

    // 3. 通过 Enable 函数注册 hooks
    driverExport->MUpti->Enable([](MUpti::MUptiDriverHooks* hooks) {
        hooks->EnterMemcpy = MyEnterMemcpy;
        hooks->ExitMemcpy = MyExitMemcpy;
        hooks->RegisterKernel = MyRegisterKernel;
        hooks->CreateContext = MyCreateContext;
        hooks->DestroyContext = MyDestroyContext;
        // ... 填充所有需要的 hook
    });
    // 此时 G_MUPTI_DRIVER_HOOKS.ready = true
}
```

### 示例 4: 完整的 Kernel Launch 追踪时序

```
用户线程                          Driver (mu_wrappers)                    GPU (Stream)
   │                                    │                                     │
   │ muLaunchKernel(func, grid, block)  │                                     │
   │───────────────────────────────────>│                                     │
   │                                    │ toolsIssueCallback(ENTER, muLaunchKernel)
   │                                    │────────────────────────────────────>│
   │                                    │   (profiler 记录: "launch kernel X")
   │                                    │                                     │
   │                                    │ muapiLaunchKernel()                 │
   │                                    │   → Context::CmdDispatch()          │
   │                                    │   → 创建 DispatchCommand            │
   │                                    │                                     │
   │                                    │ MUpti::RegisterKernelV2(cmd)        │
   │                                    │────────────────────────────────────>│
   │                                    │   (profiler 记录 kernel 参数)        │
   │                                    │                                     │
   │                                    │ MUpti::MarkKernelQueued(corrId)     │
   │                                    │────────────────────────────────────>│
   │                                    │   (profiler 记录: "kernel queued")   │
   │                                    │                                     │
   │                                    │ DispatchCommand::Submit()           │
   │                                    │   → Build() → Submit() to GPU queue │
   │                                    │────────────────────────────────────>│
   │                                    │                                     │ GPU 执行 kernel
   │                                    │                                     │
   │                                    │ MUpti::MarkKernelSubmitted(corrId)  │
   │                                    │────────────────────────────────────>│
   │                                    │   (profiler 记录: "kernel submitted")│
   │                                    │                                     │
   │                                    │ toolsIssueCallback(EXIT, muLaunchKernel)
   │                                    │────────────────────────────────────>│
   │                                    │   (profiler 记录: "API returned")    │
   │                                    │                                     │
   │<───────────────────────────────────│                                     │
   │  return MUSA_SUCCESS              │                                     │
   │                                    │                                     │
   │                                    │          (后续 GPU 完成时)            │
   │                                    │ MUpti::MarkKernelBeginEnd(ctx, cmd) │
   │                                    │────────────────────────────────────>│
   │                                    │   (profiler 记录: begin/end 时间戳)  │
```

---

## 六、关键设计洞察

### 1. 零开销快速路径

```cpp
if (toolsCallbackEnabled(handle, domain, cbid)) {
    // 慢路径: 构建参数、发送回调
} else {
    // 快速路径: 直接调用驱动
    status = muapiXxx(...);
}
```

**当没有 profiler 订阅时**，`toolsCallbackEnabled()` 返回 0，代码直接走 `else` 分支，**没有任何额外开销**。

### 2. 无锁订阅者协议

```cpp
// 写入方 (subscribe/unsubscribe):
pSubscriber->seq.fetch_add(1);     // seq = 1 (标记写入开始)
pSubscriber->userdata = userdata;   // 写数据
pSubscriber->seq.fetch_add(1);     // seq = 2 (标记数据写完)
pSubscriber->callback.store(cb);    // 写回调指针

// 读取方 (IssueCallback):
seqFirst = seq.load();             // 读 seq
userdata = pSubscriber->userdata;   // 读数据
callback = callback.load();         // 读回调
seqSecond = seq.load();            // 再读 seq
if (seqFirst == seqSecond) {       // 一致 → 数据有效
    callback(userdata, ...);
}
```

这保证了在多线程环境下，读取到的 `userdata` 和 `callback` 是一致的配对。

### 3. 三层拦截的分工

| 层 | 拦截点 | 用途 | 性能影响 |
|----|--------|------|----------|
| **API 回调** | 每个 `mu*` 函数入口/出口 | API 调用追踪、参数记录、性能分析 | 仅当订阅时 |
| **Driver Hooks** | GPU 命令执行时 | 命令级追踪、kernel 参数、时间戳 | 仅当 ready=true |
| **Accessor 导出表** | 工具主动查询 | 获取内部对象信息 | 按需调用 |

### 4. correlationId 的作用

`correlationId` 是一个线程唯一的递增计数器（通过 `Util::ThreadInfo::Get().ApiSeqNum()` 获取），用于：

- **匹配 ENTER/EXIT**: 同一个 API 调用的 ENTER 和 EXIT 回调共享同一个 correlationId
- **关联 API 和 Command**: API 调用产生的 correlationId 会被传递给后续创建的 Command
- **性能分析**: profiler 可以用 correlationId 计算 API 的执行时间

### 5. skipDriverImpl 的用途

```cpp
uint32_t skipDriverImpl = 0;
inParams.pSkipDriverImpl = &skipDriverImpl;

toolsIssueCallback(..., &inParams);  // profiler 可以修改 skipDriverImpl

if (!skipDriverImpl) {
    status = muapiXxx(...);  // 如果 profiler 设为 1，这里跳过
}
```

这允许 profiler 或调试工具 **拦截 API 调用但不实际执行**，用于：
- Mock 测试
- API 调用计数
- 参数验证

### 6. AccessorHint 哨兵值

```cpp
constexpr uintptr_t AccessorHint = 0xDEADBEEF;
#define ADDMEMBER(type, name) type name = reinterpret_cast<type>(AccessorHint)
```

所有 Accessor 函数指针默认初始化为 `0xDEADBEEF`。在 tracepoints 中：

```cpp
if (fn != reinterpret_cast<void*>(AccessorHint)) {
    fn(correlationId);  // 只有被 profiler 填充过才调用
}
```

这避免了调用未初始化的函数指针。

---

## 七、MUPTI 与相关组件的关系

> ⚠️ **重要说明**: MUSA Driver (`libmusa.so`) 和 MUSA Runtime (`libmusa_runtime.so`) 是**两个独立的库**。
> - Driver 层在 `musa/src/driver/` 中实现（本仓库）
> - Runtime 层 (`musa* API`) 在**另一个独立仓库**中实现，不在本仓库中
> - Runtime MUPTI 的 CBID 枚举已定义（`mupti_runtime_cbid.h`），但生成本仓库中**没有** `musa*` 函数的 MUPTI 包装器代码
> - 两个库共用同一个 `callback.cpp` 基础设施（位于 `libmusa.so` 中）

```
┌──────────────────────────────────────────────────────────┐
│                    用户应用程序                            │
│        musaMalloc() / musaLaunchKernel()                  │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│     MUSA-Runtime (libmusa_runtime.so) — 独立仓库           │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  musa* API 实现                                    │ │
│  │  (每个 musa* 函数内部调用对应的 mu* 函数)             │ │
│  │                                                     │ │
│  │  ⚠️ Runtime MUPTI 包装器                            │ │
│  │  (CBID 已定义但在此仓库中未实现，可能在 Runtime 仓库)   │ │
│  │  → toolsIssueCallback(DOMAIN_RUNTIME_API, ...)      │ │
│  └─────────────────────────────────────────────────────┘ │
└────────────────────────┬─────────────────────────────────┘
                         │ 调用 mu*() 函数 (通过 libmusa.so 导出)
                         ▼
┌──────────────────────────────────────────────────────────┐
│               MUSA-Driver (libmusa.so)                    │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  mu_wrappers_generated.cpp                          │ │
│  │  ├── toolsCallbackEnabled() → 快速路径检查           │ │
│  │  ├── toolsIssueCallback(DRIVER_API, ENTER) ← 域=1   │ │
│  │  ├── muapiXxx() → 实际驱动逻辑                      │ │
│  │  └── toolsIssueCallback(DRIVER_API, EXIT)           │ │
│  └─────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  mupti/tracepoints.h (命令级追踪，检查 ready 标志)   │ │
│  │  ├── MUpti::EnterMemcpy() / ExitMemcpy()            │ │
│  │  ├── MUpti::RegisterKernelV2()                      │ │
│  │  ├── MUpti::MarkKernelQueued() / Submitted()        │ │
│  │  └── 40+ tracepoint 函数                             │ │
│  └─────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  callback.cpp ← 两个库共用此基础设施                  │ │
│  │  ├── g_subscribers[4] — 共享订阅者数组               │ │
│  │  ├── g_callbackEnabled[DOMAIN][CBID] — 共享标志位    │ │
│  │  ├── toolsSubscribe() → handle 1~4                  │ │
│  │  ├── toolsIssueCallback() → 广播给所有订阅者          │ │
│  │  └── IssueCallback() → 无锁 seq 协议                │ │
│  └─────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  export_table.h                                     │ │
│  │  ├── DriverExportTable (Accessor 函数指针表)         │ │
│  │  ├── MUptiDriverHooks (60 个 hook 函数)             │ │
│  │  ├── MUptiRuntimeHooks (仅 3 个 hook 函数)          │ │
│  │  └── 各种 Command/Context/Stream Accessors           │ │
│  └─────────────────────────────────────────────────────┘ │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│                  M3D / HAL 层                             │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Command::Build() / Submit()                        │ │
│  │  → GPU 队列提交                                      │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

---

## 八、同时启用 Driver 和 Runtime MUPTI 的场景分析

> 这是对上述设计的**关键复盘**。当 Driver 层 (`libmusa.so`) 和 Runtime 层 (`libmusa_runtime.so`) 同时启用 MUPTI 时，会出现一系列需要理解的重要行为。

### 8.1 两个独立的库，两套 MUPTI 定义

| 层面 | 库文件 | API 前缀 | MUPTI 域 | 实现状态 |
|------|--------|----------|----------|----------|
| **Driver MUPTI** | `libmusa.so` | `mu*` | `MU_TOOLS_CB_DOMAIN_DRIVER_API` | ✅ **已实现** (此仓库) |
| **Runtime MUPTI** | `libmusa_runtime.so` | `musa*` | `MU_TOOLS_CB_DOMAIN_RUNTIME_API` | ⚠️ **CBID 已定义，包装器未生成** (可能在 Runtime 仓库) |

**关键事实**:
- Runtime 层 (`src/runtime/`) **不存在于本仓库**中，它位于一个独立的仓库
- 本项目的 `src/musa_shared_include/mupti/mupti_runtime_cbid.h` 定义了 480+ 个 `musa*` 函数的 CBID 枚举
- 但本仓库中 **没有任何 .cpp 文件** 实际使用 `MUtoolsTraceRuntimeApiMusa` 结构体或调用 `toolsIssueCallback(DOMAIN_RUNTIME_API, ...)`
- Runtime 的 MUPTI 包装器（类似 `mu_wrappers_generated.cpp` 但针对 `musa*` 函数）预期在 Runtime 仓库中生成

### 8.2 共享的 Callback 基础设施

**最重要的一点**: `callback.cpp` 被编译进 **`libmusa.so`**（Driver 库）。Runtime 库在运行时链接 `libmusa.so`，所以两者共用**同一个全局状态**：

```cpp
// callback.cpp — 全局共享状态
static volatile MUtoolsCbSubscriber g_subscribers[MAX_TOOLS_SUBSCRIBERS] = {};
static std::atomic<uint32_t> g_subscriberCount{0};

// Driver 域和 Runtime 域的启用标志位在同一个进程地址空间中
uint32_t driverApiCallbackEnabled[MUPTI_DRIVER_TRACE_CBID_SIZE] = {};
uint32_t runtimeApiCallbackEnabled[MUPTI_RUNTIME_TRACE_CBID_SIZE] = {};
```

**这意味着**:
- 一次 `toolsSubscribe()` 调用注册的回调函数会收到**两个域**的事件
- Driver 发出的 `toolsIssueCallback(DRIVER_API, ...)` 和 Runtime 发出的 `toolsIssueCallback(RUNTIME_API, ...)` 都通过**同一个** `IssueCallback()` 广播给所有订阅者
- 两个库共享 4 个订阅者插槽的总容量

### 8.3 双重触发问题 (Double-triggering)

这是最关键的逻辑问题。当用户调用一个 `musa*` 函数时，会触发**两层** MUPTI 回调：

```
用户调用 musaMalloc()
    │
    ├── [Runtime MUPTI] toolsIssueCallback(RUNTIME_API, ENTER, musaMalloc)
    │       correlationId = R1 (Runtime 的序列号)
    │
    ├── musaMalloc() 内部调用 muMemAlloc()  ← 进入 Driver 库
    │       │
    │       ├── [Driver MUPTI] toolsIssueCallback(DRIVER_API, ENTER, muMemAlloc)
    │       │       correlationId = D1 (Driver 的序列号)
    │       │
    │       ├── muapiMemAlloc()  ← 实际分配内存
    │       │
    │       └── [Driver MUPTI] toolsIssueCallback(DRIVER_API, EXIT, muMemAlloc)
    │
    └── [Runtime MUPTI] toolsIssueCallback(RUNTIME_API, EXIT, musaMalloc)
```

**结果**: 1 次用户级 `musaMalloc()` 调用产生 **4 次 MUPTI 回调**（2 ENTER + 2 EXIT）。

#### 对 Profiler 的影响

| 场景 | 回调次数 | 含义 |
|------|----------|------|
| 只启用 Driver MUPTI | 2 次 (ENTER+EXIT) | 只追踪到 mu* 级 |
| 只启用 Runtime MUPTI | 2 次 (ENTER+EXIT) | 只追踪到 musa* 级 |
| 同时启用两层 | 4 次 | 两层都追踪 |

**如果 profiler 简单地统计"API 调用次数"**，会看到 `muMemAlloc` 和 `musaMalloc` 各一次。但如果 profiler 只订阅了 `DRIVER_API` 域仍然只收到 2 次。

### 8.4 correlationId 的独立性问题

**Runtime 和 Driver 使用各自独立的 correlationId 序列**：

```cpp
// Driver 层: 使用 Util::ThreadInfo::Get().ApiSeqNum()
// (定义在 src/driver/internal.h 中，通过 ThreadLocalInfo 获取)
uint32_t correlationId = Util::ThreadInfo::Get().ApiSeqNum();

// Runtime 层: 使用独立的序列号 (假设定义在 Runtime 仓库中)
uint32_t runtimeCorrId = Runtime::GetApiSeqNum();
```

**问题**: 你在 Runtime 看到 `musaMalloc(correlationId=R1, ENTER)` 和 `musaMalloc(correlationId=R1, EXIT)`，然后看到 Driver 层的 `muMemAlloc(correlationId=D1, ENTER)`。**R1 和 D1 不相等**，所以你不能直接用 correlationId 来关联 Runtime API 调用和它触发的 Driver API 调用。

**解决方案**: Profiler 可以通过以下方式关联：
- **线程时序**: Runtime `musaMalloc` EXIT 和 Driver `muMemAlloc` ENTER 发生在同一个线程中，且时间上连续
- **contextId**: 两个回调中的 `contextId` 字段都是同一个 Context 的序列 ID，可以用于关联
- **参数关联**: `muMemAlloc` 返回的 `dptr` 就是 `musaMalloc` 请求的指针

### 8.5 skipDriverImpl 的跨层交互

当两层都启用时，`skipDriverImpl` 的行为变得复杂：

```cpp
// Runtime 包装器 (假设实现)
uint32_t skipDriverImpl = 0;
inParams.pSkipDriverImpl = &skipDriverImpl;
toolsIssueCallback(RUNTIME_API, ..., &inParams);

// profiler 在 Runtime 回调中设置 skipDriverImpl=1
// 但这只会阻止 Runtime 层调用 muMemAlloc()，不会影响 Driver 层的回调
if (!skipDriverImpl) {
    status = muMemAlloc(...);  // 如果 Runtime 跳过了，Driver 层的 muMemAlloc 不会被调用
}
```

如果 Runtime 调用了 `muMemAlloc()`，而 Driver 的包装器也检查 `skipDriverImpl`:

```cpp
// Driver 包装器 (mu_wrappers_generated.cpp)
uint32_t skipDriverImpl = 0;  // 每个包装器独立的局部变量
inParams.pSkipDriverImpl = &skipDriverImpl;
toolsIssueCallback(DRIVER_API, ..., &inParams);

// 注意: 这里的 skipDriverImpl 是 Driver 包装器自己的局部变量
// profiler 在 Runtime 回调中设置的 skipDriverImpl 影响不到这里
if (!skipDriverImpl) {
    status = muapiMemAlloc(...);  // Driver 层依然会执行
}
```

⚠️ **每个包装器都有自己的 `skipDriverImpl` 局部变量**，互不影响。Runtime 的 `skipDriverImpl` 和 Driver 的 `skipDriverImpl` 是**完全独立**的。

### 8.6 Runtime 没有命令级追踪能力

| 能力 | Driver MUPTI | Runtime MUPTI |
|------|-------------|---------------|
| API 级 ENTER/EXIT | ✅ (700+ CBID) | ✅ (480+ CBID) |
| 命令级 tracepoints (40+ 个) | ✅ | ❌ |
| Graph 节点追踪 | ✅ | ❌ |
| 同步事件追踪 | ✅ | ❌ |
| 资源生命周期追踪 | ✅ | ❌ |
| Accessor 导出表 | ✅ (20+ Accessor) | ❌ |

Runtime MUPTI 的 hook 结构总共只有 **3 个函数**：

```cpp
struct MUptiRuntimeHooks {
    ADDMEMBER(EnterRuntimeApi_fn*, EnterRuntimeApi);  // API 进入
    ADDMEMBER(ExitRuntimeApi_fn*, ExitRuntimeApi);    // API 退出
    ADDMEMBER(SetMemset3DCounter_fn*, SetMemset3DCounter); // 3D memset 计数器
};
```

相比之下，Driver 的 `MUptiDriverHooks` 有 **60 个 hook 函数**，涵盖命令级别的详细追踪。

### 8.7 初始化路径不同

```
Driver MUPTI 初始化:                    Runtime MUPTI 初始化:
    muInit()                                musaInit()
        │                                       │
        ├── toolsIssueCallback(ENTER)            ├── toolsIssueCallback(ENTER)
        ├── g_InjectionManager.Init()            ├── 加载 Runtime MUPTI 注入库
        │   └── dlopen("libmusaProfile.so")      │   └── 调用 MUptiRuntimeControllers::Enable()
        │   └── EnableMUptiDriver(initer)        │       └── 填充 G_MUPTI_RUNTIME_HOOKS
        │       └── 填充 G_MUPTI_DRIVER_HOOKS    │
        │       └── G_MUPTI_DRIVER_HOOKS.ready   │
        ├── muapiInit()                          ├── 初始化 Runtime 内部状态
        └── toolsIssueCallback(EXIT)             └── toolsIssueCallback(EXIT)
```

**关键区别**:
- Driver 的 `G_MUPTI_DRIVER_HOOKS` 通过 `EnableMUptiDriver()` 填充函数指针
- Runtime 的 `G_MUPTI_RUNTIME_HOOKS` 通过 `EnableMUptiRuntime()` 填充
- 两者是**独立的全局变量**，互不影响
- `callback.cpp` 中的 `g_callbackEnabled[][]` 是**共享的**——Driver 和 Runtime 操作同一个数组

### 8.8 时序图: 完整的两层 MUPTI 调用链

```
用户线程                    Runtime库                   Driver库                    GPU
  │                           │                           │                       │
  │ musaMalloc(size)          │                           │                       │
  │──────────────────────────>│                           │                       │
  │                           │ toolsCallbackEnabled(RUNTIME_API, musaMalloc)?    │
  │                           │ │是                           │                       │
  │                           │ toolsIssueCallback(ENTER)    │                       │
  │                           │──>[profiler: musaMalloc enter]                       │
  │                           │                           │                       │
  │                           │ muMemAlloc(&dptr, size)    │                       │
  │                           │──────────────────────────>│                       │
  │                           │                           │ toolsCallbackEnabled(DRIVER_API, muMemAlloc)?
  │                           │                           │ │是                    │
  │                           │                           │ toolsIssueCallback(ENTER)
  │                           │                           │──>[profiler: muMemAlloc enter]
  │                           │                           │                       │
  │                           │                           │ muapiMemAlloc()       │
  │                           │                           │   → Memory::GeneralAlloc()
  │                           │                           │   → 分配 GPU 内存      │
  │                           │                           │                       │
  │                           │                           │ toolsIssueCallback(EXIT)
  │                           │                           │──>[profiler: muMemAlloc exit]
  │                           │                           │                       │
  │                           │<──return MUSA_SUCCESS ────│                       │
  │                           │                           │                       │
  │                           │ toolsIssueCallback(EXIT)   │                       │
  │                           │──>[profiler: musaMalloc exit]                      │
  │<──return musaSuccess ─────│                           │                       │
```

### 8.9 问题总结与设计建议

| # | 问题 | 影响 | 解决方案 |
|---|------|------|----------|
| 1 | **双重触发**: 1 次 `musa*` 调用产生 4 次回调 | Profiler 看到 2 倍事件 | Profiler 应区分 DOMAIN（DRIVER vs RUNTIME），分别统计 |
| 2 | **correlationId 不连续**: Runtime 和 Driver 使用独立序列号 | 无法直接关联跨层调用 | 使用 contextId + 线程时序做关联，或在 profiler 中维护映射 |
| 3 | **skipDriverImpl 独立**: 每层各有自己的标志位 | Runtime 跳过不影响 Driver | 如需整体跳过，需在两层都设置 |
| 4 | **共享订阅者插槽**: 最多 4 个订阅者 | 两层共享 capacity | 规划订阅者数量时考虑两层共用 |
| 5 | **Runtime 无命令级追踪**: 只有 3 个 hook | Runtime 看不到 GPU 命令细节 | 需要命令级信息时订阅 Driver MUPTI |
| 6 | **Runtime MUPTI 未实现**: CBID 存在但包装器未生成 | Runtime MUPTI 目前不可用 | 需在 Runtime 仓库中生成包装器 |

**结论**: 同时启用两层 MUPTI 时，profiler 会看到**更完整**的调用链（从 `musa*` 到 `mu*` 到 GPU 命令），但需要理解两层之间的 correlationId 独立性、skipDriverImpl 隔离性等问题，才能正确解析事件流。

---

## 九、文件清单

| 文件 | 作用 |
|------|------|
| `src/driver/mu_wrappers_generated.cpp` | 自动生成的 API 包装器，每个 mu* 函数的 ENTER/EXIT 拦截 |
| `src/driver/callback.cpp` | 订阅/取消订阅、启用/禁用回调、广播分发 |
| `src/driver/callback.h` | 回调 API 声明 |
| `src/driver/mupti/hooks.h` | MUptiDriverHookStorage 全局结构 |
| `src/driver/mupti/hooks.cpp` | EnableMUptiDriver / DisableMUptiDriver 实现 |
| `src/driver/mupti/tracepoints.h` | 40+ 个内联 tracepoint 函数 |
| `src/musa_shared_include/mupti/mupti_driver_cbid.h` | Driver API CBID 枚举 (700+) |
| `src/musa_shared_include/mupti/mupti_runtime_cbid.h` | Runtime API CBID 枚举 |
| `src/musa_shared_include/export_table.h` | 所有 Accessor 结构、Hook 结构、ExportTable |
| `src/driver/muapi.h` | muapi* 函数声明（实际驱动实现） |
| `src/driver/mu_profiler.cpp` | muapiProfilerStart/Stop 实现 (stub) |

---

## 十、总结

MUPTI 是一个 **多域、多订阅者、零开销** 的 profiling 拦截框架，分为 Driver 和 Runtime 两个独立层面：

### Driver MUPTI（三层拦截，已实现）

| 层 | 拦截点 | 用途 |
|----|--------|------|
| **API 回调层** | 每个 `mu*` 函数入口/出口 | API 调用追踪、参数记录 |
| **命令 Hook 层** | GPU 命令执行时 (60 个 hook) | 命令级追踪、kernel 参数、时间戳 |
| **Accessor 导出表** | 工具主动查询 | 获取内部对象信息 |

### Runtime MUPTI（仅 API 层，CBID 已定义但本仓库未实现）

| 能力 | 说明 |
|------|------|
| API 级 ENTER/EXIT | 480+ CBID 枚举已定义，在 Runtime 仓库生成包装器 |
| 命令级追踪 | ❌ 不支持 |
| Hook 函数 | 仅 3 个: EnterRuntimeApi / ExitRuntimeApi / SetMemset3DCounter |

### 关键设计亮点
- 无订阅者时零开销（快速路径直接调用驱动）
- 最多 4 个工具同时订阅，Driver 和 Runtime 共享订阅者插槽
- 无锁 seq 协议保证多线程安全
- correlationId 关联 API 调用和 GPU 命令（Driver/Runtime 独立序列）
- skipDriverImpl 允许拦截但不执行（每层独立）
- AccessorHint 哨兵值防止调用未初始化函数

### ⚠️ 同时启用两层时的注意事项
1. 1 次 `musa*` 调用产生 **4 次回调**（2 层 × ENTER/EXIT）
2. Driver 和 Runtime 的 correlationId **互相独立**，不能直接关联
3. `skipDriverImpl` **每层独立**，跨层跳过需要分别设置
4. 两层共用 `callback.cpp` 基础设施（4 个订阅者插槽共享）
5. Runtime 无命令级追踪能力，需命令信息时需订阅 Driver MUPTI
