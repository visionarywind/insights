# MUSA-Runtime MUPTI 技术洞察

## 1. 源码范围

| 文件 | 作用 |
|---|---|
| `MUSA-Runtime/src/mupti/hooks.h` | Runtime 侧 MUPTI hook 存储结构和 enable/disable 声明 |
| `MUSA-Runtime/src/mupti/hooks.cpp` | Runtime hook 初始化、关闭逻辑 |
| `MUSA-Runtime/src/internal.h` | `ApiInvocationGuard`、`ApiTrace`、`EnterRuntimeApi`、`ExitRuntimeApi` |
| `MUSA-Runtime/src/musa_wrappers_generated.cpp` | 自动生成的 `musa*` API wrapper 和 Runtime API callback 发送逻辑 |
| `MUSA-Runtime/src/musa_entry.cpp` | `musaapiGetExportTable` 导出 Runtime MUPTI controller |
| `MUSA-Runtime/src/musa_memory.cpp` | `musaMemset3D` / `musaMemset3DAsync` 的 MUPTI 计数补充 |
| `linux-ddk/musa/src/musa_shared_include/export_table.h` | `MUptiRuntimeHooks`、`RuntimeExportTable`、Tools callback 数据结构 |
| `linux-ddk/musa/src/musa_shared_include/mupti/mupti_runtime_cbid.h` | Runtime API CBID 枚举 |

Runtime 侧 MUPTI 实现主要覆盖 public Runtime API 的进入、退出、参数、返回码和 correlation id。它不直接采集 Driver 内部 command、kernel、graph、sync 活动，这些由 Driver 侧 MUPTI 负责。

## 2. 核心结论

MUSA-Runtime 的 MUPTI 设计有两条采集路径：

1. **Tools callback 路径**：当 Driver 支持 Tools callback table 时，Runtime wrapper 直接通过 Driver 导出的 `Tools::ExportTable` 发送 Runtime API enter/exit 事件。
2. **Runtime hook 路径**：当 Tools callback 不可用时，Runtime 使用 `G_MUPTI_RUNTIME_HOOKS` 中的 `EnterRuntimeApi` / `ExitRuntimeApi` hook。

源码中的判断条件如下：

```cpp
if (G_MUPTI_RUNTIME_HOOKS.ready.load(std::memory_order_acquire) &&
    !Runtime::g_ExportTable.SupportToolsCallback()) {
    context = G_MUPTI_RUNTIME_HOOKS.hooks.EnterRuntimeApi(id, isInvocation, args);
}
```

因此，在新版本 Driver 支持 Tools callback 的情况下，Runtime API 事件会走 Driver callback 子系统；Runtime hook 是兼容路径。

## 3. Runtime MUPTI 导出入口

`MUSA-Runtime/src/musa_entry.cpp` 中的 `musaapiGetExportTable` 导出 Runtime 侧 MUPTI controller：

```cpp
static MUpti::MUptiRuntimeControllers muptiRuntimeControllers = {
    EnableMUptiRuntime,
    DisableMUptiRuntime,
    {}
};

static MUpti::RuntimeExportTable muptiRuntimeTable = {
    &muptiRuntimeControllers,
    {}
};
```

当外部 MUPTI 组件通过 `Client::MUpti` 请求 Runtime export table 时，Runtime 返回 `muptiRuntimeTable`。这个表只包含 controller，不包含 callback 分发器。

Runtime 侧 hook 类型定义在 `export_table.h`：

```cpp
struct MUptiRuntimeHooks {
    EnterRuntimeApi_fn* EnterRuntimeApi;
    ExitRuntimeApi_fn* ExitRuntimeApi;
    SetMemset3DCounter_fn* SetMemset3DCounter;
    uintptr_t reserved = 0;
};

struct MUptiRuntimeControllers {
    EnableMUptiRuntime_fn* Enable;
    DisableMUptiRuntime_fn* Disable;
    uintptr_t* reserved = nullptr;
};
```

这说明 Runtime 侧 hook 范围较窄：API enter、API exit、`memset3D` 计数补充。

## 4. Runtime hook 生命周期

`MUSA-Runtime/src/mupti/hooks.h` 定义全局 hook storage：

```cpp
struct MUptiRuntimeHookStorage {
    std::atomic<bool> ready;
    MUpti::MUptiRuntimeHooks hooks;
    uintptr_t reserved[8];
};

extern MUptiRuntimeHookStorage G_MUPTI_RUNTIME_HOOKS;
```

`hooks.cpp` 初始化为未启用：

```cpp
MUptiRuntimeHookStorage G_MUPTI_RUNTIME_HOOKS = {
    {false},
    {},
    {},
};
```

启用流程：

```cpp
void EnableMUptiRuntime(MUpti::InitializeMUptiRuntimeHooks_fn initer) {
    initer(&G_MUPTI_RUNTIME_HOOKS.hooks);
    G_MUPTI_RUNTIME_HOOKS.ready.store(true, std::memory_order_release);
}
```

设计要点：

| 设计点 | 说明 |
|---|---|
| `ready` 原子变量 | 未启用时热路径只读一次原子变量 |
| `initer` 回调 | MUPTI 组件负责填充 hook 函数表 |
| release/acquire | 保证 hook 表写入完成后再对 Runtime API 热路径可见 |
| `DisableMUptiRuntime` | 只把 `ready` 置为 false，不清空 hook 表 |

## 5. Runtime wrapper 的 API 追踪流程

Runtime public API 在 `musa_wrappers_generated.cpp` 中自动生成。每个 `musa*` API 都遵循同一模式：

```text
musa* API
  -> 创建 ApiTrace
  -> ApiTrace 初始化 ExportTable
  -> 创建 ApiInvocationGuard
  -> 尝试 Runtime hook EnterRuntimeApi
  -> 如果 Driver Tools callback 可用且当前 CBID 已启用:
       构造 MUtoolsTraceRuntimeApiMusa
       IssueCallback(ENTER)
       执行 musaapi*
       IssueCallback(EXIT)
     否则:
       直接执行 musaapi*
  -> ApiTrace 析构
  -> Runtime hook ExitRuntimeApi
  -> 设置 last error
```

以 `musaDeviceSetLimit` 为例，wrapper 会构造：

```cpp
MUtoolsTraceRuntimeApiMusa inParams{};
uint32_t correlationId = Utils::ThreadInfo::Get().ApiSeqNum();

musaDeviceSetLimit_v3020_params params{};
params.limit = limit;
params.value = value;

inParams.struct_size = sizeof(inParams);
inParams.pCorrelationId = &correlationId;
inParams.pStatus = &status;
inParams.functionName = "musaDeviceSetLimit_v3020";
inParams.functionId = MUPTI_RUNTIME_TRACE_CBID_musaDeviceSetLimit_v3020;
inParams.params = &params;
inParams.apiEnterOrExit = MU_TOOLS_API_ENTER;
```

然后调用 Driver Tools callback：

```cpp
g_ExportTable.GetToolsTable().callback->IssueCallback(
    MU_TOOLS_SUBSCRIBER_HANDLE,
    MU_TOOLS_CB_DOMAIN_RUNTIME_API,
    MUPTI_RUNTIME_TRACE_CBID_musaDeviceSetLimit_v3020,
    &inParams);
```

Runtime API 事件的数据结构为：

```cpp
typedef struct MUtoolsTraceRuntimeApiMusa_st {
    uint32_t struct_size;
    uint32_t* pCorrelationId;
    musaError_t* pStatus;
    const char* functionName;
    uint32_t functionId;
    const void* params;
    MUtools_api_enter_exit apiEnterOrExit;
} MUtoolsTraceRuntimeApiMusa;
```

该结构只包含 Runtime API 层信息，不包含 Driver context、stream、command 或 kernel 信息。

## 6. ApiInvocationGuard 的职责

`ApiInvocationGuard` 位于 `MUSA-Runtime/src/internal.h`。它承担四类职责：

| 职责 | 实现 |
|---|---|
| API 嵌套识别 | `GetDuringApiInvocation()` / `SetDuringApiInvocation(true)` |
| correlation id 分配 | 外层 API 调用时 `tinfo.SetApiSeqNum(IncrCorrelationId())` |
| 日志 | `LOG_API` 开启时打印进入和退出 |
| 错误回栈 | `LOG_ERR` 开启且返回错误时打印 backtrace |

Runtime API 经常在内部再次调用 Driver API 或 Runtime helper。`m_IsInvocation` 用来区分“用户直接调用的外层 API”和“内部递归调用”。这能避免 correlation id 被内部调用覆盖。

## 7. ApiTrace 的职责

`ApiTrace` 是 Runtime wrapper 的 RAII 入口：

```cpp
ApiTrace(musaError_t& status, std::string_view apiName,
         MUpti_runtime_api_trace_cbid cbid, const Ts&... args)
    : m_status(status)
    , m_guard()
    , m_ptiContext(nullptr)
    , m_setLastError(cbid != MUPTI_RUNTIME_TRACE_CBID_musaGetLastError_v3020) {
    m_status = g_ExportTable.Init();
    if (m_status == musaSuccess) {
        m_guard = std::make_unique<ApiInvocationGuard>(apiName, cbid, args...);
        m_ptiContext = MUpti::EnterRuntimeApi(cbid, m_guard->GetIsInvocation(), args...);
    }
}
```

析构时：

```cpp
~ApiTrace() noexcept {
    if (m_guard) {
        m_guard->SetReturnValue(m_status);
    }
    if (m_setLastError && m_status != musaSuccess && m_status != musaErrorNotReady) {
        GetTlsData().SetLastError(m_status);
    }
    MUpti::ExitRuntimeApi(m_ptiContext, m_status);
}
```

关键点：

1. `g_ExportTable.Init()` 在每个 Runtime API wrapper 开始执行，初始化 Driver/Tools/Injection 表。
2. `EnterRuntimeApi` 返回 `MUpti::Context*`，析构时交给 `ExitRuntimeApi`。
3. 如果走 Tools callback 路径，`m_ptiContext` 通常为 `nullptr`，事件由 wrapper 中的 `IssueCallback` 完成。
4. `ApiTrace` 同时负责 last error 更新，这与 profiling 逻辑共用一个 RAII 生命周期。

## 8. ExportTableManager 与 Tools callback 探测

Runtime 通过 `ExportTableManager::Init()` 加载 Driver：

```text
Reload libmusa.so.1 / libmusa.so
  -> dlsym(muGetExportTable)
  -> 获取 Runtime table
  -> 获取 Driver table
  -> 获取 Tools table
  -> 获取 Injection table
```

Tools callback 是可选能力。Runtime 判断方式：

```cpp
const enum Client uuid = Client::Tools;
Tools::ExportTable const* pToolsTable = &m_ToolsTable;
musaError_t status = ToMusaError(muGetExportTable(...));

if (status == musaSuccess &&
    m_ToolsTable.callback->Subscribe != reinterpret_cast<toolsSubscribe_fn*>(AccessorHint)) {
    m_SupportToolsCallback = true;
}
```

`AccessorHint` 表示该函数未被 Driver 填充。Runtime 必须检查 `Subscribe` 是否真实可用，不能只看 `muGetExportTable` 返回值。

这个设计兼容旧 Driver：

| Driver 能力 | Runtime 行为 |
|---|---|
| 支持 Tools callback | Runtime wrapper 使用 `Tools::ExportTable` 发送 Runtime API callback |
| 不支持 Tools callback | Runtime 使用 `G_MUPTI_RUNTIME_HOOKS` 走 legacy hook |
| 不支持 Injection table | Runtime 不加载注入库 |

## 9. CBID 设计

Runtime API 编号由 `mupti_runtime_cbid.h` 自动生成。当前源码中：

```text
MUPTI_RUNTIME_TRACE_CBID_musaMemcpyAsync_v3020        = 41
MUPTI_RUNTIME_TRACE_CBID_musaStreamSynchronize_v3020  = 131
MUPTI_RUNTIME_TRACE_CBID_musaMemset3D_v3020           = 142
MUPTI_RUNTIME_TRACE_CBID_musaLaunchKernel_v7000       = 211
MUPTI_RUNTIME_TRACE_CBID_musaLaunchKernelExC_v11060   = 430
MUPTI_RUNTIME_TRACE_CBID_SIZE                         = 531
```

wrapper 通过宏映射 API 名称到 CBID：

```cpp
#define MUPTI_CBID_GET(apiName) MUPTI_RUNTIME_TRACE_CBID_##apiName
```

CBID 的价值：

1. 热路径用整数判断是否启用，不做字符串匹配。
2. profiler 可以按 domain + cbid 精确开关。
3. callback payload 中同时保留 `functionName` 和 `functionId`，便于离线解析。

## 10. `musaMemset3D` 的特殊处理

`musaapiMemset3D` 和 `musaapiMemset3DAsync` 在某些路径下会拆成多次 `muMemsetD2D8` 或 `muMemsetD2D8Async`：

```cpp
MUpti::SetMemset3DCounter(
    Utils::ThreadInfo::Get().ApiSeqNum(),
    static_cast<uint32_t>(extent.depth));
```

这个 hook 用于告诉 MUPTI：一个 Runtime 层的 3D memset 会对应多次 Driver 层 2D memset。否则 profiler 只能看到多次 Driver memset，无法准确回溯到一个 Runtime API。

该设计本质上是关系补充：

```text
musaMemset3D(correlation_id = X, depth = D)
  -> muMemsetD2D8 #0
  -> muMemsetD2D8 #1
  -> ...
  -> muMemsetD2D8 #(D-1)
```

## 11. 与 Driver MUPTI 的边界

Runtime MUPTI 只提供 API 层信息：

```text
Runtime API name
Runtime API CBID
Runtime API params
Runtime status
correlation id
enter / exit
```

Driver MUPTI 提供更靠近执行层的信息：

```text
Driver API callback
Command object accessor
Dispatch / memcpy / memset / memory transfer accessor
Kernel queued / submitted
Command begin / end
Graph node begin / end
Stream / context / event synchronization
Context / stream resource lifecycle
```

因此性能建模时不能只依赖 Runtime 侧事件。Runtime 事件适合解释框架调用、Runtime wrapper 开销、首次初始化和 Runtime API 与 Driver API 的映射关系。Driver 内部成本必须用 Driver 侧 MUPTI 事件补齐。

## 12. 低开销设计

Runtime 热路径控制成本主要依赖四个设计：

| 设计 | 效果 |
|---|---|
| CBID 数组开关 | 判断某 API 是否被订阅只需要整数索引 |
| `SupportToolsCallback()` 分支 | Driver 支持新 callback 时不走 legacy Runtime hook |
| `ready` 原子变量 | Runtime hook 未开启时快速返回 |
| RAII 统一封装 | API enter、exit、last error、日志共享同一生命周期 |

未启用 MUPTI 或 callback 时，wrapper 直接执行：

```cpp
status = musaapiDeviceSetLimit(limit, value);
```

启用 callback 时才构造 `params` 和 `MUtoolsTraceRuntimeApiMusa`。

## 13. 当前实现的限制

| 限制 | 影响 |
|---|---|
| Runtime hook 能力少 | 只能记录 API enter/exit 和 `memset3D` 计数 |
| Runtime 内部状态没有通用事件 | ExportTable 初始化、module cache、function cache 不能直接从 MUPTI 事件分解 |
| Tools callback 由 Driver 提供 | Runtime API 事件依赖 Driver Tools table 可用性 |
| 单 subscriber 由 Driver callback 子系统决定 | 与其它 profiler 同时订阅会冲突 |
| Runtime API 事件不包含 command/kernel 关系 | 需要 Driver MUPTI 事件串联 |

## 14. 对性能建模的意义

Runtime MUPTI 可以支撑以下建模项：

| 建模项 | 数据来源 |
|---|---|
| Runtime API self time | Runtime API enter/exit |
| Runtime -> Driver 调用映射 | Runtime correlation id + Driver callback |
| Runtime 参数分布 | `MUtoolsTraceRuntimeApiMusa::params` |
| Runtime 错误路径 | `pStatus` |
| `musaMemset3D` 拆分关系 | `SetMemset3DCounter` |

仍需新增的 Runtime 内部埋点：

```text
ExportTableInitBegin/End
DriverTableLoadBegin/End
ToolsTableDetectBegin/End
InjectionLoadBegin/End
ProgramStateRegisterFatBinary
ModuleCacheHit/Miss
FunctionCacheHit/Miss
RuntimeToDriverDispatch
```

这些事件建议接入统一 MUPTI internal event path，而不是写日志或临时文件。
