# OKR MUPTI 埋点性能建模技术方案

依据：

| 文档 | 用途 |
|---|---|
| `insights/okr/okr-2026.md` | OKR 原始要求 |
| `insights/okr/okr-2026-landing-plan.md` | 白盒软件性能建模落地方案 |
| `insights/mupti/MUSA-Runtime-MUPTI-技术洞察.md` | Runtime 侧 MUPTI 实现分析 |
| `insights/mupti/musa-Driver-MUPTI-技术洞察.md` | Driver 侧 MUPTI 实现分析 |

## 1. 结论

OKR 中的 Driver/API 性能建模应采用以下方案：

```text
现有 MUPTI API callback
  -> 采集 Runtime API / Driver API enter-exit

现有 Driver MUPTI tracepoint
  -> 采集 kernel、command、submission、graph、sync、resource 关系

新增 MUPTI internal event hook
  -> 采集 memory pool、dependency、command build、merge、submit、ioctl、wait reason 等白盒成本点

collector
  -> 统一采集 API 事件、内部事件、关系事件、kernel activity

white-box model
  -> 根据源码状态机重建执行过程，使用事件数据校准成本项
```

该方案不把 MUPTI trace 当成模型。模型规则来自 `musa` 和 `MUSA-Runtime` 源码；MUPTI 事件用于采集证据、校准成本和验证模型。

## 2. 为什么必须使用 MUPTI

MUPTI 已经是 Runtime 和 Driver 中现成的 profiling 接入层，具备三项关键能力：

| 能力 | 当前实现 | 对 OKR 的价值 |
|---|---|---|
| API callback | `driver/callback.cpp`、`mu_wrappers_generated.cpp`、`musa_wrappers_generated.cpp` | 覆盖 Top90 Driver API 和 Runtime API 热点 |
| 执行活动 tracepoint | `driver/mupti/tracepoints.h`、`command/*.cpp`、`stream.cpp`、`graphCommand.cpp` | 建立 API、command、submission、kernel、graph node 关系 |
| hook export table | `export_table.h`、`mu_entry.cpp`、`musa_entry.cpp` | 支撑新增内部事件，不需要临时日志通道 |

不建议用日志、printf 或独立文件写埋点数据：

1. 难以用 `correlation_id` 串起 API、command、submission、kernel。
2. 热路径开销不可控。
3. 与现有 profiler、activity、accessor 无法统一。
4. 跨 SDK 版本难以维护 schema。

## 3. 现有能力复用

### 3.1 Runtime API 事件

Runtime API 已经由 `musa_wrappers_generated.cpp` 自动生成 callback。

当前可直接采集：

```text
musa* API name
Runtime API CBID
params
status
correlation id
enter / exit
```

关键源码：

| 文件 | 能力 |
|---|---|
| `MUSA-Runtime/src/internal.h` | `ApiTrace`、`ApiInvocationGuard`、`EnterRuntimeApi`、`ExitRuntimeApi` |
| `MUSA-Runtime/src/musa_wrappers_generated.cpp` | `MUtoolsTraceRuntimeApiMusa` enter/exit callback |
| `MUSA-Runtime/src/mupti/hooks.h` | `G_MUPTI_RUNTIME_HOOKS` |
| `MUSA-Runtime/src/musa_entry.cpp` | Runtime MUPTI controller |

Runtime 侧当前不提供 command、kernel、graph、sync 细节。Runtime 事件用于解释 framework 调用和 Runtime wrapper 成本，不承担 Driver 内部建模。

### 3.2 Driver API 事件

Driver API callback 已经由 `mu_wrappers_generated.cpp` 自动生成。

当前可直接采集：

```text
mu* API name
Driver API CBID
params
status
context / contextId
correlation id
enter / exit
```

关键源码：

| 文件 | 能力 |
|---|---|
| `linux-ddk/musa/src/driver/callback.cpp` | `Subscribe`、`EnableCallback`、`EnableAllCallbacksInDomain`、`IssueCallback` |
| `linux-ddk/musa/src/driver/mu_wrappers_generated.cpp` | `MUtoolsTraceApiMusa` enter/exit callback |
| `linux-ddk/musa/src/driver/internal.h` | Driver `ApiTrace` 和 correlation id |
| `linux-ddk/musa/src/musa_shared_include/mupti/mupti_driver_cbid.h` | Driver API CBID |

Driver callback 当前支持：

```text
MU_TOOLS_CB_DOMAIN_DRIVER_API
MU_TOOLS_CB_DOMAIN_RUNTIME_API
MU_TOOLS_CB_DOMAIN_RESOURCE
MU_TOOLS_CB_DOMAIN_SYNCHRONIZE
```

限制：

1. 当前只支持一个 subscriber。
2. API callback 只覆盖 wrapper 层，不能解释 pool miss、command build、merge、submit 等内部成本。
3. `pSkipDriverImpl` 能改变真实执行，collector 必须禁止使用该能力。

### 3.3 Driver 内部 activity 事件

Driver MUPTI tracepoint 已覆盖多类执行活动：

| 类型 | 当前接口 | 用途 |
|---|---|---|
| kernel 注册 | `RegisterKernel`、`RegisterKernelV2` | 建立 kernel activity |
| kernel queued/submitted | `MarkKernelQueued`、`MarkKernelSubmitted` | 记录排队和提交时间 |
| command begin/end | `MarkCommandBeginEnd`、`MarkCommandBeginEndV2` | 记录 command 执行时间 |
| submission 关系 | `AssignKernelToKick`、`AssignSubmissionToCorrelation` | 串起 command 和 submission |
| memcpy/memset | `EnterMemcpy`、`EnterMemset`、`ExitMemcpy`、`ExitMemset` | 记录数据搬运 activity |
| graph | `RegisterGraphTrace`、`RegisterGraphKernel`、`MarkGraphNodeBeginEndV2` | 记录 graph node activity |
| sync | `RegisterStreamSynchronize`、`Start/StopStreamSynchronize` | 记录 host wait |
| resource | `CreateContext`、`CreateStream` | 记录 context/stream 生命周期 |

这些能力可以直接用于 KR2 的 Top50 kernel 排序对齐和 API -> command -> submission -> kernel 关系重建。

## 4. 当前缺口

现有 MUPTI 能解释“发生了哪些 API 和 kernel”，但不能完整解释“Driver 源码为什么这样执行”。OKR 要做白盒软件性能建模，必须补齐以下内部事件。

| 模块 | 当前能力 | 缺口 |
|---|---|---|
| Runtime init | Runtime API callback | ExportTable init、Tools table detect、Injection load 无分项事件 |
| Runtime module | Runtime API callback | fatbin register、module cache、function cache 无 hit/miss 事件 |
| Driver API | Driver API callback | 参数检查、context lookup、object lookup 无分项事件 |
| Memory | memcpy/memset activity | pool lookup、pool hit/miss、chunk allocate、BO alloc/map ioctl 无事件 |
| Stream | stream sync activity | dependency resolve、queue depth、async capacity wait 无事件 |
| Command | command begin/end | command build、merge check、merge flush、submit begin/end 无事件 |
| Graph | graph node activity | instantiate、update、rebuild decision 无事件 |
| Sync | sync begin/end | wait target state、wait reason 无事件 |
| IOCTL | 无统一事件 | KMD 边界耗时不可拆分 |

因此，OKR 的 MUPTI 埋点方案必须新增 internal event，不应只依赖现有 API callback。

## 5. 技术路线

### 5.1 推荐路线

采用“两层复用 + 一层新增”的路线：

```text
第一层：复用 Tools callback
  Runtime API / Driver API enter-exit

第二层：复用 Driver MUPTI tracepoint
  kernel / command / submission / graph / sync / resource

第三层：新增 ModelEvent hook
  memory / stream / command / graph / sync / ioctl 的白盒内部事件
```

推荐先实现 private hook，不直接扩展 public callback domain。

理由：

1. 当前 public callback domain 固定为 5 类，直接新增 domain 涉及 ABI 和工具兼容。
2. Driver 已有 `MUptiDriverHooks`，Runtime 已有 `MUptiRuntimeHooks`，新增 hook 与现有架构一致。
3. private hook 可快速验证 OKR 建模价值，稳定后再升级为正式 internal event domain。

### 5.2 接口选择

| 方案 | 做法 | 评价 |
|---|---|---|
| A | 新增 public callback domain `MODEL_INTERNAL` | 对外语义清晰，但改动 callback domain、CBID、工具 ABI |
| B | 在 `MUptiDriverHooks` / `MUptiRuntimeHooks` 中新增 `EmitModelEvent` | 改动集中，适合第一阶段落地 |
| C | 用现有 tracepoint 增加大量专用函数 | 类型安全，但 hook 表膨胀快 |
| D | 写日志或 JSON 文件 | 不推荐，热路径开销和关联关系不可控 |

建议采用 B：

```text
Phase 1:
  private EmitModelEvent hook

Phase 2:
  schema 稳定后再抽象为 public internal event domain
```

## 6. ModelEvent 设计

### 6.1 事件头

新增内部事件使用统一 header：

```cpp
enum MusaModelEventPhase : uint16_t {
    MODEL_EVENT_POINT = 0,
    MODEL_EVENT_BEGIN = 1,
    MODEL_EVENT_END = 2,
};

enum MusaModelEventLayer : uint16_t {
    MODEL_LAYER_RUNTIME = 1,
    MODEL_LAYER_DRIVER = 2,
    MODEL_LAYER_CORE = 3,
    MODEL_LAYER_HAL = 4,
};

struct MusaModelEventHeader {
    uint16_t version;
    uint16_t layer;
    uint16_t domain;
    uint16_t phase;
    uint32_t event_id;
    uint32_t payload_size;
    uint64_t timestamp_ns;
    uint64_t correlation_id;
    uint64_t thread_id;
    uint64_t context_id;
    uint64_t stream_id;
};
```

字段要求：

| 字段 | 要求 |
|---|---|
| `version` | schema 版本，跨 SDK 解析必须依赖该字段 |
| `domain/event_id` | 热路径只使用整数，不使用字符串 |
| `phase` | 区分 begin/end/point |
| `correlation_id` | 关联 API、command、submission、kernel |
| `context_id/stream_id` | 关联资源状态 |
| `payload_size` | 支持不同 payload 类型 |

### 6.2 Hook 类型

建议在 `export_table.h` 增加：

```cpp
using EmitModelEvent_fn =
    void(const MusaModelEventHeader* header,
         const void* payload,
         uint16_t payload_size);
```

Driver hook：

```cpp
struct MUptiDriverHooks {
    ...
    ADDMEMBER(EmitModelEvent_fn*, EmitModelEvent);
    uintptr_t reserved = 0;
};
```

Runtime hook：

```cpp
struct MUptiRuntimeHooks {
    ADDMEMBER(EnterRuntimeApi_fn*, EnterRuntimeApi);
    ADDMEMBER(ExitRuntimeApi_fn*, ExitRuntimeApi);
    ADDMEMBER(SetMemset3DCounter_fn*, SetMemset3DCounter);
    ADDMEMBER(EmitModelEvent_fn*, EmitModelEvent);
    uintptr_t reserved = 0;
};
```

调用前必须检查：

```cpp
if (G_MUPTI_DRIVER_HOOKS.ready.load(std::memory_order_acquire)) {
    auto fn = G_MUPTI_DRIVER_HOOKS.hooks.EmitModelEvent;
    if (fn != reinterpret_cast<void*>(AccessorHint)) {
        fn(&header, &payload, sizeof(payload));
    }
}
```

Runtime 侧同理使用 `G_MUPTI_RUNTIME_HOOKS`。

### 6.3 事件 domain

```text
MODEL_RUNTIME_INTERNAL
MODEL_DRIVER_API_INTERNAL
MODEL_MEMORY
MODEL_STREAM
MODEL_COMMAND
MODEL_GRAPH
MODEL_SYNC
MODEL_IOCTL
MODEL_RELATION
```

### 6.4 事件命名

事件名不进入热路径。源码中只使用枚举：

```cpp
enum MusaModelEventId : uint32_t {
    kRuntimeExportTableInit = 1001,
    kRuntimeModuleCacheDecision = 1101,

    kDriverContextLookup = 2001,
    kDriverObjectLookup = 2002,

    kMemoryPoolLookup = 3001,
    kMemoryPoolAllocDecision = 3002,
    kMemoryChunkAllocate = 3003,
    kMemoryBoAllocIoctl = 3004,
    kMemoryMapIoctl = 3005,

    kStreamResolveDependency = 4001,
    kStreamQueueCommand = 4002,
    kStreamAsyncCapacityWait = 4003,

    kCommandBuild = 5001,
    kCommandMergeCheck = 5002,
    kCommandMergeFlush = 5003,
    kCommandSubmit = 5004,

    kGraphInstantiate = 6001,
    kGraphUpdate = 6002,
    kGraphRebuildDecision = 6003,

    kSyncWaitReason = 7001,
    kRelationApiCommand = 8001,
};
```

离线工具负责把 `event_id` 映射为名称。

## 7. 埋点切点

### 7.1 不需要新增埋点的部分

以下能力直接复用，不需要改源码：

| 需求 | 现有机制 |
|---|---|
| Runtime API enter/exit | `musa_wrappers_generated.cpp` + Tools callback |
| Driver API enter/exit | `mu_wrappers_generated.cpp` + Tools callback |
| kernel register | `RegisterKernel` / `RegisterKernelV2` |
| kernel queued/submitted | `MarkKernelQueued` / `MarkKernelSubmitted` 或 command accessor |
| command begin/end | `MarkCommandBeginEnd` |
| graph node begin/end | `MarkGraphNodeBeginEndV2` |
| stream sync begin/end | `RegisterStreamSynchronize` + `Start/StopStreamSynchronize` |
| context/stream lifecycle | `CreateContext`、`CreateStream` |

### 7.2 Runtime 新增切点

| 文件 | 切点 | 事件 |
|---|---|---|
| `MUSA-Runtime/src/internal.h` | `ExportTableManager::Init` | `RuntimeExportTableInitBegin/End` |
| `MUSA-Runtime/src/internal.h` | `m_MusaLib.Reload` | `RuntimeLoadDriverBegin/End` |
| `MUSA-Runtime/src/internal.h` | Tools table 探测 | `RuntimeToolsTableDetect` |
| `MUSA-Runtime/src/internal.h` | Injection table 加载 | `RuntimeInjectionLoadBegin/End` |
| `MUSA-Runtime/src/internal.h` / module 相关路径 | fatbin/module/function cache | `RuntimeModuleCacheDecision`、`RuntimeFunctionCacheDecision` |

Runtime 新增埋点重点是首次调用和 module/function cache。它们会影响模型推理启动阶段和首次 kernel launch。

### 7.3 Driver API 新增切点

| 文件 | 切点 | 事件 |
|---|---|---|
| `linux-ddk/musa/src/driver/mu_memory.cpp` | 参数检查后、core call 前 | `DriverApiValidation`、`DriverObjectLookup` |
| `linux-ddk/musa/src/driver/mu_stream.cpp` | stream 获取、sync 入口 | `DriverStreamLookup` |
| `linux-ddk/musa/src/driver/mu_graph.cpp` | graph / graphExec 获取 | `DriverGraphLookup` |
| `linux-ddk/musa/src/driver/mu_module.cpp` | module/function 查询 | `DriverModuleLookup` |
| `linux-ddk/musa/src/driver/mu_context.cpp` | context 获取、primary context | `DriverContextLookup` |

这些事件用于把 Driver API self time 拆成 validation、lookup、domain dispatch、core call。

### 7.4 Memory 新增切点

| 文件 | 切点 | 事件 |
|---|---|---|
| `linux-ddk/musa/src/musa/core/memory.cpp` | `Memory::Init` | `MemoryInitBegin/End` |
| `linux-ddk/musa/src/musa/core/context.cpp` | `CreateMemory`、`MapToPeers` | `MemoryCreateBegin/End`、`MemoryMapToPeersBegin/End` |
| `linux-ddk/musa/src/hal/m3d/memMgr.cpp` | pool lookup / route | `MemoryPoolLookup` |
| `linux-ddk/musa/src/hal/m3d/memoryPool.cpp` | `SubAllocate` | `MemoryPoolHit/Miss` |
| `linux-ddk/musa/src/hal/m3d/memoryPool.cpp` | `ChunkAllocate` | `MemoryChunkAllocateBegin/End` |
| HAL memory / KMD 边界 | BO alloc、map、unmap、free | `MemoryBoAllocIoctlBegin/End`、`MemoryMapIoctlBegin/End` |

Memory 是 OKR 中最需要新增白盒事件的模块。现有 MUPTI 只能看到 memcpy/memset activity，不能解释 allocator 成本。

### 7.5 Stream / Command 新增切点

| 文件 | 切点 | 事件 |
|---|---|---|
| `linux-ddk/musa/src/musa/core/context.cpp` | `ResolveDependencyAndQueueCommand` | `ResolveDependencyBegin/End`、`DependencyEdgeCreated` |
| `linux-ddk/musa/src/musa/core/stream.cpp` | `QueueCommand` | `QueueCommandBegin/End`、`QueueDepth` |
| `linux-ddk/musa/src/musa/core/stream.cpp` | async capacity wait | `AsyncCapacityWaitBegin/End` |
| `linux-ddk/musa/src/musa/core/command/command.cpp` | `Build` | `CommandBuildBegin/End` |
| `linux-ddk/musa/src/musa/core/stream.cpp` | merge check / flush | `CommandMergeCheck`、`CommandMergeFlush` |
| `linux-ddk/musa/src/musa/core/command/command.cpp` / HAL queue | submit | `SubmitBegin/End`、`SubmitIoctlBegin/End` |
| submit / inflight 控制 | inflight wait | `InflightWaitBegin/End` |

这些事件用于解释小 kernel 多、stream 依赖复杂、submit 过多、merge 失败等软件开销。

### 7.6 Graph 新增切点

| 文件 | 切点 | 事件 |
|---|---|---|
| `linux-ddk/musa/src/driver/mu_graph.cpp` | API 到 core 前 | `DriverGraphLookup` |
| `linux-ddk/musa/src/musa/core/graph/graph1/graphExec.cpp` | instantiate | `GraphInstantiateBegin/End` |
| `linux-ddk/musa/src/musa/core/graph/graph1/graphExec.cpp` | resource / submission prepare | `GraphPrepareSubmissionBegin/End` |
| `linux-ddk/musa/src/musa/core/graph/graph2/graphExec.cpp` | graph2 launch path | `Graph2LaunchPath` |
| graph update 路径 | update / rebuild | `GraphUpdateBegin/End`、`GraphRebuildDecision` |

现有 graph tracepoint 能得到 graph node activity，但不能解释 instantiate/update/rebuild 的 host 侧成本。

### 7.7 Sync 新增切点

| 文件 | 切点 | 事件 |
|---|---|---|
| `linux-ddk/musa/src/musa/core/stream.cpp` | `Synchronize` / `WaitFinish` | `StreamWaitReason`、`WaitTargetState` |
| `linux-ddk/musa/src/musa/core/context.cpp` | `Synchronize` / `LockedWait` | `ContextWaitReason` |
| `linux-ddk/musa/src/musa/core/command/barrierCommand.cpp` | event wait | `EventWaitReason` |

现有 `Start/StopStreamSynchronize` 可以测等待时长，但不能解释等待对象和等待原因。新增事件必须输出 wait target、timeline、last submitted、last completed 等字段。

## 8. Collector 设计

collector 以注入库运行：

```bash
MUSA_INJECTION64_PATH=./libmusa_model_collector.so \
MUSA_MODEL_TRACE_OUTPUT=./model_events \
MUSA_MODEL_TRACE_DOMAINS=api,kernel,graph,sync,memory,stream,command,ioctl \
./workload
```

collector 初始化流程：

```text
InitializeInjection
  -> 获取 Tools::ExportTable
  -> Subscribe callback
  -> EnableAllCallbacksInDomain(DRIVER_API)
  -> EnableAllCallbacksInDomain(RUNTIME_API)
  -> 按需启用 RESOURCE / SYNCHRONIZE
  -> 获取 MUpti::DriverExportTable
  -> 调用 Driver MUpti controller Enable，填充 Driver hooks
  -> 获取 MUpti::RuntimeExportTable
  -> 调用 Runtime MUpti controller Enable，填充 Runtime hooks
  -> 初始化 TLS ring buffer 和后台 flush 线程
```

collector 输出：

```text
run_metadata.yaml
api_events.parquet
model_events.parquet
activity_events.parquet
relations.parquet
model_features.parquet
collector_dropped_events.txt
```

### 8.1 单 subscriber 处理

当前 Driver callback 只支持一个 subscriber。OKR 采集阶段采用 collector 独占 MUPTI。

处理策略：

| 场景 | 做法 |
|---|---|
| 与 `torch.profiler` 冲突 | 使用 collector 采集 API/internal 事件，kernel 数据由 collector 内部 activity 采集 |
| 必须使用外部 profiler | 分两次运行，并用 workload phase、shape、kernel name、sequence 对齐 |
| 长期方案 | collector 提供转发接口，或推动 MUPTI 多 subscriber |

## 9. 低开销控制

### 9.1 未开启 collector

未开启时要求：

```text
不分配内存
不格式化字符串
不加锁
不写文件
不构造 payload
只允许原子 ready 检查和分支
```

实现方式：

```cpp
if (!G_MUPTI_DRIVER_HOOKS.ready.load(std::memory_order_acquire)) {
    return;
}
```

内部事件必须在 `ready` 判断之后再填充 payload。

### 9.2 开启 collector

开启后仍要控制成本：

| 设计 | 要求 |
|---|---|
| domain 开关 | 只采集本次分析需要的 domain |
| event_id 开关 | 支持单个事件启用 |
| fixed payload | 结构体 payload，不写字符串 |
| TLS ring buffer | 热路径写 thread-local buffer |
| 后台 flush | 不在 API 或 command 热路径写磁盘 |
| dropped counter | ring buffer 满时计数丢弃，不阻塞业务 |
| sampling | 高频 point event 支持采样 |

验收标准：

```text
collector 关闭：
  API microbench p50/p90 变化在噪声范围内。

collector 开启 API-only：
  能稳定采集 Runtime/Driver API enter-exit，不改变 API 返回值。

collector 开启 internal targeted domain：
  只对被启用 domain 产生额外开销，报告中必须给出 collector overhead。
```

## 10. 数据建模流程

### 10.1 事件归一化

```text
raw callback / hook event
  -> timestamp 标准化
  -> domain/event_id 名称映射
  -> enter/exit 配对
  -> relation 展开
  -> feature 表生成
```

### 10.2 关系重建

必须重建以下关系：

```text
Runtime API correlation id
  -> Driver API correlation id
  -> command correlation id
  -> submission id
  -> kernel id / graph node id
```

Graph 模式下使用：

```text
graph_node_unique_id =
  (graph_command_correlation_id << 32) + graph_node_correlation_id
```

### 10.3 成本项

模型输出以下分项：

```text
T_api =
  T_runtime_wrapper
  + T_driver_validation
  + T_context_lookup
  + T_object_lookup
  + T_core_call

T_memory =
  T_memory_init
  + T_pool_lookup
  + T_suballocate
  + miss_count * T_chunk_allocate
  + ioctl_count * T_ioctl
  + T_peer_map

T_stream_command =
  T_dependency_resolve
  + T_queue_command
  + T_async_wait
  + T_command_build
  + T_merge_check
  + T_submit
  + T_inflight_wait

T_graph =
  T_instantiate
  + T_prepare_submission
  + T_update
  + T_rebuild
  + T_graph_launch

T_sync =
  T_wait_target_ready_check
  + T_host_wait
  + T_finish_signal
```

## 11. OKR 对齐

### KR1：覆盖 Top90% Driver API

实现方式：

1. 用现有 Driver API callback 采集 3 个模型的 `mu*` API 耗时。
2. 生成 Top90 Driver API 列表。
3. 对 Top90 API 建源码规则。
4. 在规则涉及的内部路径补 ModelEvent。
5. 输出 API 分项成本。

交付物：

```text
top90_driver_api.csv
api_source_rule.yaml
instrumentation_coverage.md
api_cost_breakdown.parquet
```

验收口径：

```text
每个 Top90 API 必须具备：
  API callback
  源码规则
  内部事件覆盖
  成本分项
  误差归因
```

### KR2：Top50 kernel Top-K overlap 90%

实现方式：

1. 复用 `RegisterKernelV2`、`MarkCommandBeginEnd`、`AssignKernelToKick`、`AssignSubmissionToCorrelation`。
2. 建立 API -> command -> submission -> kernel 关系表。
3. 用 kernel activity 生成 measured Top50。
4. 用模型预测 kernel launch sequence 和 host submit 延迟。
5. 比较 Top-K overlap。

新增内部事件的作用：

| 问题 | 需要的内部事件 |
|---|---|
| kernel 顺序变化 | dependency、queue、merge、submit |
| submit 延迟变大 | command build、merge flush、submit ioctl、inflight wait |
| graph kernel 映射错误 | graph node relation、graph launch、graph rebuild |

### KR3：模型报告

报告必须能回答：

```text
哪个 API 是热点
热点 API 的成本来自哪个源码阶段
哪个 kernel 是热点
kernel 是由哪个 API / command / graph node 触发
submit 或 wait 是否导致性能下降
优化应修改哪个源码模块
```

## 12. 分阶段计划

### 阶段 0：无源码改动验证

使用现有 MUPTI：

```text
Runtime API callback
Driver API callback
kernel / command / graph / sync tracepoint
resource lifecycle
```

输出：

```text
api_events
activity_events
relations
current_gap_report
```

验收：

```text
能采集 3 个模型的 API 热点。
能建立主要 kernel 的 API -> command -> kernel 关系。
明确缺失的白盒成本点。
```

### 阶段 1：ModelEvent hook 基础设施

改动：

```text
export_table.h
driver/mupti/hooks.h
driver/mupti/hooks.cpp
driver/mupti/tracepoints.h
MUSA-Runtime/src/mupti/hooks.h
MUSA-Runtime/src/mupti/hooks.cpp
```

新增：

```text
MusaModelEventHeader
EmitModelEvent_fn
Driver EmitModelEvent
Runtime EmitModelEvent
event_id registry
collector model event parser
```

验收：

```text
collector 能收到 Runtime 和 Driver 内部 ModelEvent。
关闭 collector 时无 payload 构造。
```

### 阶段 2：最小闭环 API

先覆盖三个 API：

```text
muLaunchKernel
muMemAlloc
muStreamSynchronize
```

对应事件：

```text
muLaunchKernel:
  function lookup
  dependency resolve
  queue command
  command build
  merge check
  submit
  kernel relation

muMemAlloc:
  memory init
  pool lookup
  pool hit/miss
  chunk allocate
  BO alloc/map ioctl

muStreamSynchronize:
  stream lookup
  wait begin/end
  wait target
  wait reason
```

验收：

```text
能输出三个 API 的完整成本分解。
能解释一次 kernel launch、一次分配、一次同步等待。
```

### 阶段 3：扩展到 Top90 API

做法：

```text
根据阶段 0 的 Top90 API 列表逐个补源码规则。
按模块补 ModelEvent，而不是按 API 重复埋点。
优先 memory、stream/command、graph、sync。
```

验收：

```text
Top90 API 都能落到已有源码规则和内部事件。
Top90 API 成本分项覆盖率达到验收要求。
```

### 阶段 4：报告和 CTS

输出：

```text
3 个模型性能建模报告
3 个模型行为基线
CTS 行为用例
版本差异分析
```

CTS 事件签名示例：

```text
DriverApiEnter(muMemAlloc)
MemoryPoolLookup
MemoryPoolMiss
MemoryChunkAllocateBegin
MemoryBoAllocIoctlBegin
MemoryBoAllocIoctlEnd
MemoryChunkAllocateEnd
DriverApiExit(muMemAlloc)
```

## 13. 风险与处理

| 风险 | 影响 | 处理 |
|---|---|---|
| MUPTI 单 subscriber | 不能与外部 profiler 同时订阅 | collector 独占，或 collector 内部集成 kernel activity |
| hook ABI 变更 | SDK 兼容风险 | 新字段加在 hook 表尾部，使用 `AccessorHint` 判断 |
| 埋点影响性能 | 结果失真 | ready 检查、domain/event 开关、TLS ring buffer、采样 |
| correlation 断链 | 无法解释 API 到 kernel | API、command、submission、kernel 统一 relation event |
| graph id 复杂 | graph node 映射错误 | 使用 `(graph_command_correlation_id << 32) + node_correlation_id` |
| ioctl 位置分散 | KMD 边界不完整 | 先覆盖 BO alloc/map/submit/wait，再扩展其它 ioctl |
| Runtime 内部事件不足 | 首次调用成本无法拆分 | 补 ExportTable、module cache、function cache 事件 |

## 14. 实施优先级

优先级如下：

| 优先级 | 内容 | 原因 |
|---|---|---|
| P0 | collector 复用现有 API/kernel/graph/sync 事件 | 不改源码即可验证数据链路 |
| P1 | `EmitModelEvent` 基础设施 | 后续白盒事件的统一出口 |
| P2 | `muLaunchKernel`、`muMemAlloc`、`muStreamSynchronize` 最小闭环 | 覆盖 launch、memory、sync 三类核心成本 |
| P3 | memory pool 和 stream/command 埋点 | 对模型性能影响最大 |
| P4 | graph instantiate/update/rebuild | 对推理图模式和动态图有直接价值 |
| P5 | Runtime init/module/function cache | 支撑首次调用和冷启动分析 |

最终交付必须包含：

```text
collector
event schema
event_id registry
instrumented SDK build
model feature generator
white-box model
Top90 API coverage report
Top50 kernel overlap report
3 个模型性能建模报告
CTS behavior cases
```
