# musa Driver MUPTI 技术洞察

## 1. 源码范围

| 文件 | 作用 |
|---|---|
| `linux-ddk/musa/src/driver/callback.cpp` | Tools callback 订阅、启用、分发实现 |
| `linux-ddk/musa/src/driver/callback.h` | callback 接口声明 |
| `linux-ddk/musa/src/driver/mu_wrappers_generated.cpp` | 自动生成的 Driver API enter/exit callback |
| `linux-ddk/musa/src/driver/internal.h` | Driver `ApiInvocationGuard`、`ApiTrace`、CBID 宏 |
| `linux-ddk/musa/src/driver/mupti/hooks.h` | Driver MUPTI hook 全局存储 |
| `linux-ddk/musa/src/driver/mupti/hooks.cpp` | Driver MUPTI hook enable/disable |
| `linux-ddk/musa/src/driver/mupti/tracepoints.h` | command、kernel、graph、sync、resource tracepoint |
| `linux-ddk/musa/src/driver/mu_entry.cpp` | Driver / Tools / MUPTI export table 构造 |
| `linux-ddk/musa/src/musa_shared_include/export_table.h` | Tools callback、MUPTI hook、accessor 数据结构 |
| `linux-ddk/musa/src/musa_shared_include/mupti/mupti_driver_cbid.h` | Driver API CBID 枚举 |
| `linux-ddk/musa/src/musa/core/command/*.cpp` | kernel、memcpy、memset、graph command tracepoint 调用点 |
| `linux-ddk/musa/src/musa/core/stream.cpp` | queue、submit、synchronize tracepoint 调用点 |
| `linux-ddk/musa/src/musa/core/context.cpp` | context、stream 生命周期 tracepoint 调用点 |

Driver 侧 MUPTI 是 MUSA tracing 的主实现层。它同时承担 public API callback、内部 activity tracepoint、对象 accessor、profiler controller、injection 支撑等职责。

## 2. 核心结论

Driver 侧 MUPTI 分为两套机制：

1. **Tools callback 机制**：面向 Runtime API 和 Driver API 的 enter/exit callback。它提供订阅、按 domain/CBID 启用、事件分发能力。
2. **MUPTI hook / tracepoint 机制**：面向 Driver 内部对象和执行活动。它覆盖 kernel、command、memcpy、memset、memory transfer、graph、sync、context、stream 等活动。

二者的关系：

```text
public API callback
  关注 API 层：mu* / musa* 进入、退出、参数、返回码

internal tracepoint
  关注执行层：command、kernel、submission、graph node、sync、resource
```

性能分析和白盒软件建模必须同时使用这两类事件。只看 API callback 只能得到外层耗时，无法解释 command merge、submit、kernel relation、graph node、stream wait 等内部行为。

## 3. Tools callback 机制

### 3.1 Domain 和 CBID

`export_table.h` 定义 callback domain：

```cpp
typedef enum MUtools_cb_domain_enum {
    MU_TOOLS_CB_DOMAIN_INVALID      = 0,
    MU_TOOLS_CB_DOMAIN_DRIVER_API   = 1,
    MU_TOOLS_CB_DOMAIN_RUNTIME_API  = 2,
    MU_TOOLS_CB_DOMAIN_RESOURCE     = 3,
    MU_TOOLS_CB_DOMAIN_SYNCHRONIZE  = 4,
    MU_TOOLS_CB_DOMAIN_SIZE         = 5,
} MUtools_cb_domain;
```

`callback.cpp` 为每个 domain 分配独立启用数组：

```cpp
static constexpr uint32_t g_domainSize[MU_TOOLS_CB_DOMAIN_SIZE] = {
    0,
    MUPTI_DRIVER_TRACE_CBID_SIZE,
    MUPTI_RUNTIME_TRACE_CBID_SIZE,
    MU_TOOLS_CBID_RESOURCE_SIZE,
    MU_TOOLS_CBID_SYNCHRONIZE_SIZE,
};

uint32_t driverApiCallbackEnabled[MUPTI_DRIVER_TRACE_CBID_SIZE] = {};
uint32_t runtimeApiCallbackEnabled[MUPTI_RUNTIME_TRACE_CBID_SIZE] = {};
uint32_t resourceCallbackEnabled[MU_TOOLS_CBID_RESOURCE_SIZE] = {};
uint32_t syncCallbackEnabled[MU_TOOLS_CBID_SYNCHRONIZE_SIZE] = {};
```

当前源码中的关键规模：

```text
MUPTI_DRIVER_TRACE_CBID_SIZE   = 822
MUPTI_RUNTIME_TRACE_CBID_SIZE  = 531
MU_TOOLS_CBID_RESOURCE_SIZE    = 21
MU_TOOLS_CBID_SYNCHRONIZE_SIZE = 3
```

Driver 常见 CBID 示例：

```text
muMemAlloc              = 29
muStreamSynchronize     = 126
muLaunchKernel          = 307
muMemAllocAsync         = 598
muLaunchKernelEx        = 652
```

### 3.2 Subscriber 模型

Driver callback 当前只支持一个 subscriber：

```cpp
std::atomic<uint32_t> g_hasSubscriber{0};
static volatile MUtoolsCbSubscriber g_subscriber{};
```

subscriber 结构：

```cpp
typedef struct MUtoolsCbSubscriber_st {
    std::atomic<MUtoolsCbFunc_fn*> callback{nullptr};
    std::atomic<uint32_t> seq{0};
    void* userdata;
} MUtoolsCbSubscriber;
```

`toolsSubscribe` 使用 `g_hasSubscriber.exchange(1)` 抢占全局槽位。如果已经有 subscriber，则返回错误。该实现成本低，但会导致多个 profiler 不能同时订阅同一进程。

### 3.3 启用和分发

启用单个 callback：

```cpp
MUresult toolsEnableCallback(uint32_t enable,
                             MUtoolsCbSubscriberHandle subscriberHandle,
                             MUtools_cb_domain domain,
                             uint32_t cbid) {
    ...
    EnableCallbackIndex(domain, cbid, enable);
}
```

启用整个 domain：

```cpp
for (uint32_t cbid = 0; cbid < g_domainSize[domain]; cbid++) {
    EnableCallbackIndex(domain, cbid, enable);
}
```

热路径判断：

```cpp
uint32_t toolsCallbackEnabled(MUtoolsCbSubscriberHandle subscriberHandle,
                              MUtools_cb_domain domain,
                              uint32_t cbid) {
    ...
    res = g_callbackEnabled[domain][cbid];
}
```

分发逻辑：

```cpp
void IssueCallback(MUtools_cb_domain domain, uint32_t cbid, const void* pParams) {
    seqFirst  = pSubscriber->seq.load(std::memory_order_seq_cst);
    userdata  = pSubscriber->userdata;
    callback  = pSubscriber->callback.load(std::memory_order_seq_cst);
    seqSecond = pSubscriber->seq.load(std::memory_order_seq_cst);

    if (callback != nullptr && seqFirst == seqSecond) {
        callback(userdata, domain, cbid, pParams);
    }
}
```

这里使用 `seq` 做轻量一致性检查，避免 subscriber 正在更新时读到不一致的 `callback/userdata`。

### 3.4 低开销设计

未启用 callback 时，每个 Driver API wrapper 只执行：

```text
toolsCallbackEnabled(domain, cbid)
  -> 参数检查
  -> g_callbackEnabled[domain][cbid]
  -> false 分支直接调用 muapi*
```

没有字符串匹配、没有动态分配、没有参数结构体构造。

只有某个 API CBID 被启用后，wrapper 才构造 `MUtoolsTraceApiMusa` 和参数结构体。

## 4. Driver API wrapper 实现

Driver public API 由 `mu_wrappers_generated.cpp` 自动生成。每个 `mu*` API 结构一致：

```text
mu* API
  -> 创建 ApiTrace
  -> toolsCallbackEnabled(DRIVER_API, cbid)
  -> 如果未启用:
       直接调用 muapi*
     如果启用:
       构造 params
       构造 MUtoolsTraceApiMusa
       IssueCallback(ENTER)
       如果 profiler 未设置 skipDriverImpl:
           调用 muapi*
       IssueCallback(EXIT)
  -> return status
```

以 `muCtxGetId` 为例：

```cpp
MUresult MUSAAPI muCtxGetId(MUcontext ctx, unsigned long long* ctxId) {
    MUresult status = MUSA_SUCCESS;
    ApiTrace trace(status, "muCtxGetId", MUPTI_CBID_GET(muCtxGetId), ctx, ctxId);

    if (toolsCallbackEnabled(MU_TOOLS_SUBSCRIBER_HANDLE,
                             MU_TOOLS_CB_DOMAIN_DRIVER_API,
                             MUPTI_DRIVER_TRACE_CBID_muCtxGetId)) {
        MUtoolsTraceApiMusa inParams{};
        uint32_t correlationId = Util::ThreadInfo::Get().ApiSeqNum();
        uint32_t skipDriverImpl = 0;

        muCtxGetId_params params{};
        params.ctx = ctx;
        params.ctxId = ctxId;

        inParams.pCorrelationId = &correlationId;
        inParams.pStatus = &status;
        inParams.functionName = "muCtxGetId";
        inParams.functionId = MUPTI_DRIVER_TRACE_CBID_muCtxGetId;
        inParams.params = &params;
        inParams.apiEnterOrExit = MU_TOOLS_API_ENTER;
        inParams.pSkipDriverImpl = &skipDriverImpl;

        toolsIssueCallback(..., &inParams);

        if (!skipDriverImpl) {
            status = muapiCtxGetId(params.ctx, params.ctxId);
        }

        inParams.apiEnterOrExit = MU_TOOLS_API_EXIT;
        toolsIssueCallback(..., &inParams);
    } else {
        status = muapiCtxGetId(ctx, ctxId);
    }

    return status;
}
```

Driver API callback payload：

```cpp
typedef struct MUtoolsTraceApiMusa_st {
    uint32_t struct_size;
    uint32_t* pCorrelationId;
    MUresult* pStatus;
    const char* functionName;
    uint32_t functionId;
    uint64_t contextId;
    void* params;
    MUcontext context;
    MUtools_api_enter_exit apiEnterOrExit;
    uint32_t* pSkipDriverImpl;
} MUtoolsTraceApiMusa;
```

`pSkipDriverImpl` 是 Driver API callback 特有能力。subscriber 可以在 ENTER 阶段设置该字段，阻止真实 `muapi*` 执行。这适合调试、拦截或模拟，但性能建模采集时不应使用。

## 5. Driver ApiTrace 与 correlation id

`driver/internal.h` 中的 `ApiInvocationGuard` 负责：

| 职责 | 实现 |
|---|---|
| 外层 API 识别 | `t_DuringApiInvocation` |
| correlation id 分配 | `g_CorrelationId.fetch_add(1)` |
| API 日志 | `LOG_API` |
| 错误回栈 | `LOG_ERR` |
| last error | `ApiTrace` 析构时更新 TLS |

外层 Driver API 进入时：

```cpp
if (!m_IsInvocation) {
    t_DuringApiInvocation = true;
    tinfo.SetApiSeqNum(g_CorrelationId.fetch_add(1, std::memory_order_relaxed));
}
```

该 `ApiSeqNum` 被 wrapper 放入 `MUtoolsTraceApiMusa::pCorrelationId`。后续 command、submission、kernel tracepoint 也围绕这个 id 建立关系。

## 6. MUPTI hook 表

Driver 内部活动不走 Tools API callback，而是走 `MUptiDriverHooks`。

`hooks.h` 定义：

```cpp
struct MUptiDriverHookStorage {
    std::atomic<bool> ready;
    MUpti::MUptiDriverHooks hooks;
    uintptr_t reserved[8];
};

extern MUptiDriverHookStorage G_MUPTI_DRIVER_HOOKS;
```

启用流程：

```cpp
void EnableMUptiDriver(MUpti::InitializeMUptiDriverHooks_fn initer) {
    initer(&G_MUPTI_DRIVER_HOOKS.hooks);
    G_MUPTI_DRIVER_HOOKS.ready.store(true, std::memory_order_release);
}
```

关闭流程：

```cpp
void DisableMUptiDriver() {
    G_MUPTI_DRIVER_HOOKS.ready.store(false, std::memory_order_release);
}
```

`MUptiDriverHooks` 在 `export_table.h` 中定义，包含：

```text
API legacy:
  EnterDriverApi / ExitDriverApi
  EnterRuntimeApi / ExitRuntimeApi

memory:
  EnterMemcpy / ExitMemcpy
  StartHostMemcpy / StopHostMemcpy
  EnterMemset / ExitMemset
  StartHostMemset / StopHostMemset
  EnterMemoryAtomicV2
  EnterMemoryAtomicValue
  EnterMemoryTransferV2

kernel / command:
  RegisterKernel
  RegisterKernelV2
  MarkKernelQueued
  MarkKernelSubmitted
  MarkKernelBeginEnd
  MarkCommandBeginEnd
  MarkCommandBeginEndV2

submission relation:
  AssignKernelToKick
  AssignSubmissionToCorrelation

resource:
  CreateContext / DestroyContext
  CreateStream / DestroyStream

sync:
  RegisterEventSynchronize
  StartEventSynchronize / StopEventSynchronize
  RegisterStreamWaitEvent
  StartStreamWaitEvent / StopStreamWaitEvent
  RegisterStreamSynchronize
  StartStreamSynchronize / StopStreamSynchronize
  RegisterContextSynchronize
  StartContextSynchronize / StopContextSynchronize

graph:
  RegisterGraphTrace
  RegisterGraphTraceV2
  MarkGraphTraceBegin / MarkGraphTraceEnd
  RegisterGraphKernel / RegisterGraphMemcpy / RegisterGraphMemset
  RegisterGraphMemoryAtomic / RegisterGraphMemoryAtomicValue / RegisterGraphMemoryTransfer
  MarkGraphNodeBeginEnd / MarkGraphNodeBeginEndV2
  CheckGraphTraceEnabled
```

## 7. tracepoint 封装

`driver/mupti/tracepoints.h` 把 hook 表封装成 inline 函数。典型模式：

```cpp
inline bool CheckMUptiEnabled() {
    return G_MUPTI_DRIVER_HOOKS.ready.load(std::memory_order_acquire);
}

inline MUpti::Context* RegisterKernelV2(Musa::DispatchCommand* command) {
    if (CheckMUptiEnabled()) {
        auto fn = G_MUPTI_DRIVER_HOOKS.hooks.RegisterKernelV2;
        if (fn != reinterpret_cast<void*>(AccessorHint)) {
            return fn(command);
        }
    }
    return nullptr;
}
```

设计要点：

| 设计 | 作用 |
|---|---|
| `CheckMUptiEnabled()` | 未启用时快速返回 |
| `AccessorHint` 判断 | 兼容旧 MUPTI，允许 hook 可选 |
| `MUpti::Context*` | MUPTI 侧为一次活动建立上下文，结束时回传 |
| inline 封装 | 调用点集中，编译器可优化未开启路径 |

## 8. ExportTable 设计

`muapiGetExportTable` 在 `driver/mu_entry.cpp` 中构造多个 export table：

```text
Client::MUpti
  -> MUpti::DriverExportTable

Client::Tools
  -> Tools::ExportTable

Client::Runtime
  -> Runtime::ExportTable

Client::Driver
  -> Driver::ExportTable

Client::Injection
  -> Injection::ExportTable
```

MUPTI Driver export table 提供大量 accessor：

| accessor | 用途 |
|---|---|
| `ThreadLocalInfoAccessors` | 获取 pid、tid、correlation id |
| `CommandBaseAccessors` | 获取 command stream、type、correlation id、timestamp、submission id |
| `DispatchCommandAccessors` | 获取 kernel function、grid、block、dynamic shared memory |
| `MemcpyCommandAccessors` | 获取 memcpy kind、src/dst memory kind、size |
| `MemsetCommandAccessors` | 获取 memset kind、value、size |
| `MemoryTransferCommandAccessors` | 获取 memory transfer size |
| `DeviceAccessors` | 获取 device id |
| `ContextAccessors` | 获取 context id、device、engine sync capability |
| `StreamAccessors` | 获取 stream id、context |
| `FunctionAccessors` | 获取 kernel name、shared memory、local memory |
| `GraphAccessors` | 获取 graph command、graph node、node type、node timestamp、graph id |
| `ProfilerControllers` | profiling session、config、enable/disable |
| `PCSamplingControllers` | PC sampling enable/start/stop |
| `MsysAccessors` | 获取兼容 MUPTI 版本 |

Tools export table 主要提供 callback 和少量 object accessor：

```cpp
struct Tools::ExportTable {
    CallbackControllers* callback;
    ContextAccessors* context;
    StreamAccessors* stream;
    GreenContextAccessors* greenCtx;
    InternalApiAccessors* internal;
    uintptr_t* reserved = nullptr;
};
```

Runtime 通过 Tools export table 发送 Runtime API callback。外部 profiler 也通过 Tools table 订阅 Driver/Runtime API callback。

## 9. Kernel / command 事件链路

### 9.1 kernel 注册

`DispatchCommand` 构造时注册 kernel：

```cpp
if (RecordMUptiActivity()) {
    if (deviceProperties.ipProperties.supportEngineSync) {
        m_ptiCtx = MUpti::RegisterKernelV2(this);
    } else {
        MUpti::RegisterKernel(this);
    }
}
```

含义：

| 硬件能力 | 使用接口 |
|---|---|
| 支持 engine sync | `RegisterKernelV2`，后续可通过 command timestamp 获取 begin/end |
| 不支持 engine sync | `RegisterKernel` + `MarkKernelQueued/Submitted` legacy 路径 |

### 9.2 queued 时间

`Stream::QueueCommand` 中：

```cpp
if (command->GetType() == Command::Type::Dispatch) {
    command->SetQueuedTimestamp(GetCurrentTime());
}

if (command->GetType() == Command::Type::Dispatch &&
    !supportEngineSync) {
    MUpti::MarkKernelQueued(command->GetCorId());
}
```

新硬件可通过 command accessor 读取 queued timestamp；旧硬件通过显式 hook 标记。

### 9.3 submitted / submission relation

`Stream::AsyncSubmit` 路径中：

```cpp
uint64_t uniqueId = merged->CheckFromGraph()
    ? (static_cast<uint64_t>(merged->GetCorId()) << 32) + merged->GetGraphNode()->GetCorId()
    : merged->GetCorId();

switch (merged->GetType()) {
case Command::Type::Dispatch:
    merged->SetSubmittedTimestamp(GetCurrentTime());
    if (!supportEngineSync) {
        MUpti::MarkKernelSubmitted(merged->GetCorId());
    }
    MUpti::AssignKernelToKick(uniqueId, submissionId);
    break;
case Command::Type::AsyncMemcpy:
case Command::Type::Memset:
case Command::Type::MemoryAtomic:
case Command::Type::MemoryAtomicValue:
case Command::Type::MemoryTransfer:
    MUpti::AssignSubmissionToCorrelation(uniqueId, submissionId);
    break;
}
```

这里建立关系：

```text
API correlation id
  -> command correlation id
  -> submission id
  -> kernel/kick
```

### 9.4 begin/end 时间

`DispatchCommand::ReleaseResources` 中：

```cpp
if (m_TimestampMem) {
    if (m_ptiCtx) {
        MUpti::MarkCommandBeginEnd(m_ptiCtx, this);
    } else {
        uint64_t begin = GetTimestamp(Musa::beginTime);
        uint64_t end = GetTimestamp(Musa::endTime);
        GetGraphNode()->UpdateGraphTraceTimestamp(m_CorId, begin, end, true);
    }
}
```

这把 command 执行完成后的时间戳交给 MUPTI，用于生成 kernel/activity begin/end。

## 10. Memcpy / memset / memory transfer 事件链路

`MemcpyCommand` 构造：

```cpp
if (RecordMUptiActivity()) {
    m_ptiCtx = MUpti::EnterMemcpy(this);
}
```

资源释放：

```cpp
if (m_TimestampMem) {
    MUpti::MarkCommandBeginEnd(m_ptiCtx, this);
} else {
    MUpti::ExitMemcpy(m_ptiCtx);
}
```

`MemsetCommand` 同样使用：

```cpp
m_ptiCtx = MUpti::EnterMemset(this);
...
MUpti::MarkCommandBeginEnd(m_ptiCtx, this);
...
MUpti::ExitMemset(m_ptiCtx);
```

host 路径还会调用：

```cpp
MUpti::StartHostMemset(uniqueId);
MUpti::StopHostMemset(uniqueId);
```

Memory atomic 和 memory transfer 使用 V2 接口，支持一个 command 内多个 op index：

```text
EnterMemoryAtomicV2(command, opIndex)
EnterMemoryTransferV2(command, index)
MarkCommandBeginEndV2(context, command, index)
```

## 11. Graph 事件链路

`GraphCommand::Submit`：

```cpp
MUpti::RegisterGraphTrace(this);
MUresult status = Execute();
```

`GraphCommand::Execute`：

```cpp
MUpti::MarkGraphTraceBegin(m_CorId);
return ExecuteImpl(0);
```

`GraphCommand::Postprocess`：

```cpp
MUpti::MarkGraphTraceEnd(m_CorId);
...
MUpti::MarkGraphNodeBeginEndV2(m_PtiCtxs[kick.node], kick.node, m_GraphExec);
```

Graph node context 创建：

```cpp
switch(pNode->GetType()) {
case MU_GRAPH_NODE_TYPE_KERNEL:
    m_PtiCtxs.insert({pNode, MUpti::RegisterGraphKernel(this, pNode)});
    break;
case MU_GRAPH_NODE_TYPE_MEMCPY:
    m_PtiCtxs.insert({pNode, MUpti::RegisterGraphMemcpy(this, pNode)});
    break;
case MU_GRAPH_NODE_TYPE_MEMSET:
    m_PtiCtxs.insert({pNode, MUpti::RegisterGraphMemset(this, pNode)});
    break;
...
}
```

Graph submission relation 在 `universalManager.cpp` 中建立：

```cpp
uint64_t nodeUniqueId =
    (static_cast<uint64_t>(cmd->GetCorId()) << 32) + node->GetCorId();

MUpti::AssignKernelToKick(nodeUniqueId, cmd->GetSubId());
MUpti::AssignSubmissionToCorrelation(nodeUniqueId, cmd->GetSubId());
```

Graph 模式下使用 `(graph command correlation id << 32) + graph node correlation id` 作为唯一 id，避免一个 graph launch 中多个 node 共享同一个 API correlation id。

## 12. Sync 和 resource 事件链路

### 12.1 Stream synchronize

`Stream::Synchronize`：

```cpp
auto muptiContext = MUpti::RegisterStreamSynchronize(
    Util::ThreadInfo::Get().ApiSeqNum(), this);

MUpti::StartStreamSynchronize(muptiContext);
...
MUpti::StopStreamSynchronize(muptiContext);
```

这条链路能区分 API 总耗时中的 host wait 部分。

### 12.2 Stream wait event

`BarrierCommand` 在 host wait 路径中：

```cpp
MUpti::StartStreamWaitEvent(m_ptiCtx);
status = HostWaitSemaphores();
...
MUpti::StopStreamWaitEvent(m_ptiCtx);
```

### 12.3 Context / stream 生命周期

`Context::Init` 中：

```cpp
if (MUpti::CheckMUptiEnabled()) {
    Driver::TlsCtxPush(this);
    MUpti::CreateContext(this);
    Driver::TlsCtxPop();
}
```

`Context::Dispose` 中：

```cpp
MUpti::DestroyContext(this);
```

`Context::CreateStream`：

```cpp
MUpti::CreateStream(stream);
```

`Context::DestroyStream`：

```cpp
MUpti::DestroyStream(stream);
```

这些事件为 resource domain 提供 context/stream 生命周期。

## 13. 与 Runtime MUPTI 的关系

Driver 是 MUPTI 的中心节点：

```text
MUSA-Runtime
  -> 加载 libmusa
  -> 获取 Driver table
  -> 获取 Tools table
  -> Runtime wrapper 通过 Tools table 发送 RUNTIME_API callback

Driver
  -> 提供 Tools callback 订阅和分发
  -> 提供 DRIVER_API callback
  -> 提供内部 activity hook
  -> 提供 command / graph / stream / context accessor
```

Runtime 侧没有独立 callback manager。Runtime API callback 走 Driver 的 Tools callback 子系统。

## 14. 当前实现的优势

| 优势 | 说明 |
|---|---|
| API callback 和内部 tracepoint 分层 | API 事件不污染 command/kernel 热路径 |
| 按 domain/CBID 精确启用 | 可降低采集噪声和开销 |
| 自动生成 wrapper | Driver API coverage 高，模式一致 |
| accessor 丰富 | MUPTI 可以读取 command、kernel、graph、stream、context 细节 |
| relation hook 明确 | `AssignKernelToKick`、`AssignSubmissionToCorrelation` 能串起 submit 和 kernel |
| 支持 graph node 级别追踪 | graph launch 不会只表现为一个粗粒度 API |
| 兼容旧硬件路径 | `supportEngineSync` 决定使用 timestamp accessor 或 legacy mark hook |

## 15. 当前实现的限制

| 限制 | 影响 |
|---|---|
| callback 只支持单 subscriber | `torch.profiler`、自研 collector 等不能直接并行订阅 |
| API callback 只覆盖 wrapper 层 | 无法解释 pool miss、command build、queue wait 等内部成本 |
| internal tracepoint 覆盖偏 activity | memory pool、ioctl、dependency resolve、command merge 决策没有统一事件 |
| hook 表是函数指针 ABI | 扩展字段需要考虑版本兼容 |
| `pSkipDriverImpl` 能改变执行 | profiler 误用会影响被测程序行为 |
| resource/sync domain 较窄 | resource 主要是 context/stream/graph，sync 只有少量同步事件 |

## 16. 对性能建模的意义

Driver MUPTI 已经能支撑以下建模能力：

| 建模能力 | 数据来源 |
|---|---|
| Driver API 耗时 | `MU_TOOLS_CB_DOMAIN_DRIVER_API` enter/exit |
| Runtime API 耗时 | `MU_TOOLS_CB_DOMAIN_RUNTIME_API` enter/exit |
| kernel launch 关系 | `RegisterKernelV2`、`AssignKernelToKick` |
| submission 关系 | `AssignSubmissionToCorrelation` |
| kernel queued/submitted | `MarkKernelQueued`、`MarkKernelSubmitted` 或 command accessor |
| command begin/end | `MarkCommandBeginEnd`、timestamp accessor |
| memcpy/memset activity | `EnterMemcpy`、`EnterMemset`、`MarkCommandBeginEnd` |
| graph node activity | `RegisterGraph*`、`MarkGraphNodeBeginEndV2` |
| stream sync wait | `Register/Start/StopStreamSynchronize` |
| resource 生命周期 | `CreateContext`、`CreateStream` 等 |

仍需补齐的白盒软件建模事件：

```text
MemoryPoolLookup
MemoryPoolHit/Miss
ChunkAllocateBegin/End
BoAllocIoctlBegin/End
MapIoctlBegin/End
ResolveDependencyBegin/End
QueueCommandBegin/End
CommandBuildBegin/End
MergeCheck
MergeFlush
SubmitBegin/End
InflightWaitBegin/End
GraphInstantiateBegin/End
GraphUpdateBegin/End
GraphRebuildDecision
WaitReason
```

这些事件建议作为新的 MUPTI internal event domain 或 private hook 表接入。不要写临时日志，因为日志难以按 correlation id 与 API、command、kernel 对齐，也难以控制热路径开销。

## 17. 推荐的 collector 使用方式

最小采集组合：

```text
EnableAllCallbacksInDomain(DRIVER_API)
EnableAllCallbacksInDomain(RUNTIME_API)
Enable selected internal hooks:
  kernel
  command
  memcpy/memset
  graph
  sync
  resource
```

输出时至少保留：

```text
domain
cbid / event type
functionName
correlationId
contextId
streamId
commandId
submissionId
kernel/function name
enter/exit timestamp
status
```

对性能建模最关键的是关系表：

```text
Runtime API correlation id
  -> Driver API correlation id
  -> command correlation id
  -> submission id
  -> kernel / graph node
```

没有这张关系表，API 热点和 kernel 热点只能分别统计，不能解释“哪个 API 触发了哪个 kernel、哪个 submit、哪个 wait”。
