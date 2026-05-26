# MUSA Driver/API 白盒软件性能建模落地方案

依据：`insights/okr/okr-2026.md`。  
源码范围：`musa`、`MUSA-Runtime`。  
方案主线：源码埋点、MUPTI 采集、白盒建模、profiling 对齐。

## 1. 方案定位

本方案不把 Driver 当黑盒。模型不依赖“API 输入到 kernel 输出”的经验拟合，而是直接建模 Runtime 和 Driver 源码中的状态、分支、队列、缓存、同步和系统调用边界。

执行路径如下：

```text
Runtime / Driver 源码
  -> 在关键状态转移和成本点埋点
  -> 通过 MUPTI callback / hook 输出事件
  -> collector 采集 API 事件、内部事件、kernel 事件
  -> 重建软件执行过程
  -> 拟合源码成本项
  -> 输出性能模型、瓶颈归因、优化建议
```

MUPTI 在本方案中是统一采集通道。埋点事件是模型证据，不是模型本身。模型规则来自源码，事件数据用于校准成本和验证规则。

## 2. 建模对象

| 层级 | 建模内容 | 不建模内容 |
|---|---|---|
| `MUSA-Runtime` | `musa*` wrapper、`ApiInvocationGuard`、TLS、ExportTable、lazy init、module/function cache | GPU 侧执行 |
| Driver API | `mu*` wrapper、参数检查、context 获取、对象查找、domain 分发 | 硬件执行细节 |
| Core | context、stream、event、memory、module、graph、command 状态机 | SM/cache/warp 级行为 |
| HAL/M3D | memory pool、queue、command buffer、submit、KMD 边界 | KMD 内部调度细节 |
| MUPTI | API callback、command/kernel/graph tracepoint、内部模型事件采集 | 不作为性能模型规则来源 |
| profiler kernel 数据 | kernel 名称、启动顺序、执行耗时、Top50 排序校准 | 不推导 Driver 内部成本 |

## 3. 现有基础

当前源码中已经具备可复用的 MUPTI 机制：

| 路径 | 已有能力 |
|---|---|
| `MUSA-Runtime/src/internal.h` | `ApiInvocationGuard`、`EnterRuntimeApi`、`ExitRuntimeApi` |
| `MUSA-Runtime/src/mupti/hooks.h` | Runtime MUPTI hook storage 和 enable/disable 开关 |
| `MUSA-Runtime/src/musa_entry.cpp` | Runtime MUPTI export table |
| `linux-ddk/musa/src/driver/mu_wrappers_generated.cpp` | Driver API enter/exit callback |
| `linux-ddk/musa/src/driver/callback.cpp` | `Subscribe`、`EnableCallback`、`toolsIssueCallback` |
| `linux-ddk/musa/src/driver/mupti/hooks.h` | Driver MUPTI hook storage 和 enable/disable 开关 |
| `linux-ddk/musa/src/driver/mupti/tracepoints.h` | kernel、memcpy、memset、sync、graph tracepoint |
| `linux-ddk/musa/src/musa_shared_include/export_table.h` | MUPTI hook 类型、accessor、export table 定义 |

已有机制能够覆盖 Runtime API、Driver API、kernel 注册、kernel submit、memcpy、memset、stream/context sync 和 graph 节点。新增工作重点是把 memory pool、stream queue、command build、submit、ioctl 边界等内部软件成本点纳入统一事件体系。

## 4. 总体架构

```text
                 workload
                    |
                    v
        +------------------------+
        | MUSA-Runtime / Driver  |
        | source instrumentation |
        +------------------------+
          | API events
          | internal events
          | state transition events
          | relation events
          v
        +------------------------+
        | MUPTI event path       |
        | callback / hook        |
        +------------------------+
                    |
                    v
        +------------------------+
        | model collector        |
        | binary/jsonl output    |
        +------------------------+
                    |
                    v
        +------------------------+
        | event normalizer       |
        | ordering/correlation   |
        +------------------------+
                    |
                    v
        +------------------------+
        | white-box model        |
        | state + transition     |
        | cost calibration       |
        +------------------------+
                    |
                    v
        +------------------------+
        | report                 |
        | API hotspot            |
        | kernel hotspot         |
        | source-level advice    |
        +------------------------+
```

采集链路必须满足三点：

1. 未开启时对正常执行路径影响极低。
2. 开启后能用 `correlation_id` 串起 API、内部状态、command、submission、kernel。
3. 事件字段稳定，能跨 SDK 版本对比。

## 5. 埋点事件体系

### 5.1 事件分类

| 类型 | 用途 | 示例 |
|---|---|---|
| API event | 记录 public API enter/exit | `musaLaunchKernel`、`muMemAlloc`、`muStreamSynchronize` |
| internal span | 记录源码内部阶段耗时 | `MemoryPool::SubAllocate` begin/end |
| state transition | 记录状态变化 | pool miss、command queued、graph dirty |
| counter event | 记录数量和状态 | queue depth、free bytes、merge batch size |
| relation event | 建立对象关系 | API -> command -> submission -> kernel |
| sync event | 记录阻塞原因 | stream wait、context wait、event wait |
| ioctl event | 记录 KMD 边界 | BO alloc、map、submit、wait |

这里的 state transition 指源码状态变化，例如 pool 从 hit 变成 miss、command 从 queued 变成 submitted、graph 从 clean 变成 dirty。

### 5.2 事件字段

所有内部事件使用固定字段头，payload 只放必要信息：

```c
struct MusaModelEventHeader {
    uint16_t version;        // 事件格式版本，用于跨 SDK 解析
    uint16_t domain;         // event domain，例如 memory、stream、graph
    uint32_t event_id;       // 事件编号，不能用字符串做热路径判断
    uint64_t timestamp_ns;   // 单调时钟时间戳
    uint64_t correlation_id; // 串联 API、command、submission、kernel
    uint64_t thread_id;      // host 线程
    uint64_t context_id;     // MUSA context
    uint64_t stream_id;      // MUSA stream，没有则为 0
};
```

建议 payload 分层：

| payload | 字段 |
|---|---|
| memory | `bytes`、`pool_id`、`hit`、`chunk_id`、`free_bytes`、`fragmentation` |
| stream | `queue_depth`、`dependency_count`、`async_count`、`blocking_reason` |
| command | `command_type`、`command_id`、`merge_batch_size`、`build_bytes` |
| graph | `graph_id`、`node_count`、`edge_count`、`dirty`、`rebuild` |
| ioctl | `ioctl_cmd`、`bytes`、`fd`、`status` |
| relation | `src_type`、`src_id`、`dst_type`、`dst_id` |

### 5.3 event domain 规划

| domain | 覆盖范围 |
|---|---|
| `RUNTIME_API` | 已有 Runtime API enter/exit |
| `DRIVER_API` | 已有 Driver API enter/exit |
| `MODEL_RUNTIME_INTERNAL` | Runtime init、ExportTable、module/function cache |
| `MODEL_MEMORY` | memory tracker、pool、chunk、peer map、free merge |
| `MODEL_STREAM` | dependency、queue、async wait、submit thread |
| `MODEL_COMMAND` | command build、merge、submission relation |
| `MODEL_GRAPH` | instantiate、update、rebuild、launch、node mapping |
| `MODEL_SYNC` | stream/event/context wait |
| `MODEL_IOCTL` | KMD syscall 边界 |
| `MODEL_RELATION` | API、command、submission、kernel 关联 |

有疑问：当前 public MUPTI callback domain 是否允许扩展内部 domain。可选方案如下：

| 选择 | 做法 | 影响 |
|---|---|---|
| A | 新增 MUPTI internal event domain | 语义最清晰，需要维护 ABI |
| B | 复用现有 Driver hooks 增加 internal hook 表 | 改动较小，collector 需要识别内部事件 |
| C | 先用 private activity buffer，后续接入 public MUPTI | 上线风险低，但对外接口需要二次统一 |

建议先采用 B，保留 A 的 ABI 设计。这样可以快速验证模型价值，同时不阻塞后续标准化。

## 6. 低开销埋点设计

### 6.1 热路径原则

1. 未开启时只允许一次原子读和一次分支。
2. 未开启时不能分配内存，不能格式化字符串，不能加锁。
3. 开启后事件写入 thread-local ring buffer。
4. payload 使用固定结构体，不在热路径写 JSON。
5. 字符串只允许使用枚举 ID，离线阶段再映射名称。
6. 支持按 domain、event_id、采样率启用。

### 6.2 推荐宏

```cpp
// 未开启时只做 enabled 判断，不构造 payload。
#define MUSA_MODEL_EVENT_ENABLED(event_id) \
    (MusaModelTrace::Enabled(event_id))

// span 用于记录 begin/end 两个事件，适合函数内部阶段耗时。
#define MUSA_MODEL_SCOPE(event_id, payload) \
    MusaModelTrace::ScopeGuard model_scope_guard_##__LINE__(event_id, payload)

// point event 用于记录一次状态变化，例如 pool miss 或 command queued。
#define MUSA_MODEL_POINT(event_id, payload) \
    do { \
        if (MUSA_MODEL_EVENT_ENABLED(event_id)) { \
            MusaModelTrace::Emit(event_id, payload); \
        } \
    } while (0)
```

示例：

```cpp
MUresult MemoryPool::SubAllocate(size_t bytes, Allocation* out) {
    // 未开启 MODEL_MEMORY 时，这里只会执行一次 enabled 分支。
    if (MUSA_MODEL_EVENT_ENABLED(kMemoryPoolSubAllocate)) {
        MemoryPayload payload{};
        payload.bytes = bytes;
        payload.pool_id = GetPoolId();
        MUSA_MODEL_SCOPE(kMemoryPoolSubAllocate, payload);
    }

    auto result = TrySubAllocate(bytes, out);

    // 记录分配结果。模型用 hit/miss 区分普通路径和 chunk allocate 路径。
    if (MUSA_MODEL_EVENT_ENABLED(kMemoryPoolAllocDecision)) {
        MemoryPayload payload{};
        payload.bytes = bytes;
        payload.pool_id = GetPoolId();
        payload.hit = (result == AllocResult::Hit);
        payload.free_bytes = GetFreeBytes();
        MUSA_MODEL_POINT(kMemoryPoolAllocDecision, payload);
    }

    return Translate(result);
}
```

### 6.3 与 MUPTI 对接

新增内部事件不直接写文件。Driver/Runtime 只负责把事件交给 MUPTI 路径：

```cpp
inline void EmitModelEvent(const MusaModelEventHeader& header,
                           const void* payload,
                           uint16_t payload_size) {
    // ready 为 false 时直接返回，保证未开启时不进入慢路径。
    if (!G_MUPTI_DRIVER_HOOKS.ready.load(std::memory_order_acquire)) {
        return;
    }

    auto fn = G_MUPTI_DRIVER_HOOKS.hooks.EmitModelEvent;
    if (fn != reinterpret_cast<void*>(AccessorHint)) {
        fn(header, payload, payload_size);
    }
}
```

有疑问：当前 `MUptiDriverHooks` 中没有 `EmitModelEvent`。需要新增 hook 类型、export table 字段和 MUPTI 侧实现。若 ABI 变更窗口不足，先为 memory、stream、command、graph 各增加少量专用 hook。

## 7. 源码埋点切点

### 7.1 Runtime

| 模块 | 文件 | 埋点 |
|---|---|---|
| Runtime API | `MUSA-Runtime/src/internal.h`、`musa_wrappers_generated.cpp` | API enter/exit、fast path/slow path |
| ExportTable | `MUSA-Runtime/src/internal.cpp` | `dlopen`、`dlsym`、table init、PostFillAccessors |
| lazy init | `MUSA-Runtime/src/internal.h`、`internal.cpp` | platform init、device init、first API |
| module | `MUSA-Runtime/src/musa_module.cpp` | fatbin register、module load、function cache hit/miss |
| memory/stream/event/graph wrapper | `musa_memory.cpp`、`musa_stream.cpp`、`musa_event.cpp`、`musa_graph.cpp` | Runtime wrapper self time、driver dispatch relation |

输出指标：

```text
runtime_api_self_time
runtime_init_time
export_table_init_time
module_cache_hit_rate
function_cache_hit_rate
runtime_to_driver_dispatch_count
```

### 7.2 Driver API

| 模块 | 文件 | 埋点 |
|---|---|---|
| Driver API wrapper | `linux-ddk/musa/src/driver/mu_wrappers_generated.cpp` | 已有 API enter/exit callback |
| callback 分发 | `linux-ddk/musa/src/driver/callback.cpp` | callback enabled、issue callback 自身开销 |
| context API | `linux-ddk/musa/src/driver/mu_context.cpp` | context lookup、primary context、create/destroy |
| memory API | `linux-ddk/musa/src/driver/mu_memory.cpp` | 参数检查、memory object lookup、core call |
| stream API | `linux-ddk/musa/src/driver/mu_stream.cpp` | stream lookup、sync/register wait |
| graph API | `linux-ddk/musa/src/driver/mu_graph.cpp` | graph lookup、instantiate/update/launch |
| module API | `linux-ddk/musa/src/driver/mu_module.cpp` | module load、function lookup、kernel handle |

输出指标：

```text
driver_api_self_time
parameter_validation_time
context_lookup_time
object_lookup_time
domain_dispatch_time
error_path_count
```

### 7.3 Memory

| 模块 | 文件 | 埋点 |
|---|---|---|
| core memory | `linux-ddk/musa/src/musa/core/memory.cpp` | `Memory::Init`、allocation type、free type |
| context memory | `linux-ddk/musa/src/musa/core/context.cpp` | `CreateMemory`、`MapToPeers` |
| memory manager | `linux-ddk/musa/src/hal/m3d/memMgr.cpp` | pool lookup、pool create、allocation route |
| memory pool | `linux-ddk/musa/src/hal/m3d/memoryPool.cpp` | `FullAllocate`、`SubAllocate`、`ChunkAllocate`、`Free`、merge |
| KMD boundary | HAL memory/ioctl call site | BO alloc、map、unmap、free |

必须采集的事件：

```text
MemoryAllocBegin/End
MemoryPoolLookup
MemoryPoolHit
MemoryPoolMiss
ChunkAllocateBegin/End
BoAllocIoctlBegin/End
MapIoctlBegin/End
MapToPeersBegin/End
MemoryFreeBegin/End
FreeMergeDecision
PoolTrimDecision
```

输出指标：

```text
pool_hit_count
pool_miss_count
chunk_allocate_count
bo_alloc_ioctl_count
map_ioctl_count
peer_map_count
free_merge_count
fragmentation_ratio
memory_self_time
```

### 7.4 Stream / Command / Submit

| 模块 | 文件 | 埋点 |
|---|---|---|
| dependency | `linux-ddk/musa/src/musa/core/context.cpp` | `ResolveDependencyAndQueueCommand` |
| stream queue | `linux-ddk/musa/src/musa/core/stream.cpp` | `QueueCommand`、`AsyncSubmit`、`WaitFinish` |
| command | `linux-ddk/musa/src/musa/core/command/command.cpp`、各 command 子类 | `Build`、`SubmitToQueue`、begin/end |
| dispatch | `linux-ddk/musa/src/musa/core/command/dispatchCommand.cpp` | kernel command register、queued、submitted |
| HAL queue | `linux-ddk/musa/src/hal/m3d/queue.cpp` | submit、wait、queue depth |

必须采集的事件：

```text
ResolveDependencyBegin/End
DependencyEdgeCreated
QueueCommandBegin/End
AsyncCapacityWaitBegin/End
CommandBuildBegin/End
MergeCheck
MergeFlush
SubmitBegin/End
SubmitIoctlBegin/End
InflightWaitBegin/End
ApiCommandRelation
CommandSubmissionRelation
SubmissionKernelRelation
```

输出指标：

```text
dependency_edge_count
serialized_command_count
queue_depth_p50/p90
merge_batch_size_distribution
submit_count
submit_ioctl_count
inflight_wait_time
command_build_time
```

### 7.5 Graph

| 模块 | 文件 | 埋点 |
|---|---|---|
| graph API | `linux-ddk/musa/src/driver/mu_graph.cpp` | instantiate、update、launch API |
| graph1 exec | `linux-ddk/musa/src/musa/core/graph/graph1/graphExec.cpp` | init、resource create、submission prepare |
| graph1 manager | `linux-ddk/musa/src/musa/core/graph/graph1/universalManager.cpp` | graph submit、node mapping |
| graph2 exec | `linux-ddk/musa/src/musa/core/graph/graph2/graphExec.cpp` | graph2 路径、stream create、launch |
| graph command | `linux-ddk/musa/src/musa/core/command/graphCommand.cpp` | graph trace begin/end、node begin/end |

必须采集的事件：

```text
GraphInstantiateBegin/End
GraphValidateBegin/End
GraphTopologySortBegin/End
GraphPrepareSubmissionBegin/End
GraphUpdateBegin/End
GraphRebuildDecision
GraphLaunchBegin/End
GraphNodeCommandRelation
GraphNodeKernelRelation
```

输出指标：

```text
graph_instantiate_time
graph_update_time
graph_rebuild_count
graph_submission_count
graph_launch_self_time
graph_node_kernel_mapping
fallback_launch_count
```

### 7.6 Sync / Wait

| 模块 | 文件 | 埋点 |
|---|---|---|
| stream sync | `linux-ddk/musa/src/musa/core/stream.cpp` | `Synchronize`、`WaitFinish` |
| context sync | `linux-ddk/musa/src/musa/core/context.cpp` | `Synchronize`、`LockedWait` |
| event sync/wait | `mu_event.cpp`、barrier command | event wait、stream wait event |
| existing MUPTI tracepoint | `linux-ddk/musa/src/driver/mupti/tracepoints.h` | `RegisterStreamSynchronize`、`Start/StopStreamSynchronize` 等 |

必须采集的事件：

```text
StreamSynchronizeBegin/End
ContextSynchronizeBegin/End
EventSynchronizeBegin/End
StreamWaitEventBegin/End
WaitTargetState
WaitReason
```

输出指标：

```text
sync_wait_time
wait_target_type
wait_reason_distribution
host_blocking_api_count
implicit_sync_count
```

## 8. Collector 设计

collector 以 `.so` 注入方式运行，订阅 MUPTI 事件并写出二进制或 JSONL 数据。

### 8.1 启动方式

```bash
MUSA_INJECTION64_PATH=./libmusa_model_collector.so \
MUSA_MODEL_TRACE_DOMAINS=runtime,driver,memory,stream,command,graph,sync,ioctl \
MUSA_MODEL_TRACE_OUTPUT=./model_events.jsonl \
./run_model_workload
```

### 8.2 采集内容

| 数据 | 来源 | 用途 |
|---|---|---|
| Runtime API enter/exit | Runtime MUPTI hook | Runtime wrapper 成本 |
| Driver API enter/exit | Driver callback | Driver API 成本 |
| internal model event | 新增 MUPTI event path | 状态重建、成本校准 |
| command/kernel relation | 已有 tracepoint + 新增 relation event | Top50 kernel 对齐 |
| graph node relation | graph tracepoint | graph launch 和 node 映射 |
| sync begin/end | 现有 sync tracepoint + 新增 reason | 阻塞归因 |
| ioctl begin/end | 新增边界事件 | KMD 边界成本 |

### 8.3 输出文件

```text
run_metadata.yaml
  workload、SDK 版本、driver commit、runtime commit、collector 版本、host、device

events.jsonl
  原始事件流，按采集顺序写入

normalized_events.parquet
  标准化事件，完成 name 映射和时间单位统一

relations.parquet
  api_id、command_id、submission_id、kernel_id、graph_node_id 关系表

model_features.parquet
  用于拟合和验证的特征表
```

### 8.4 单订阅者限制

现有 MUPTI callback 通常只允许一个 subscriber。和 `torch.profiler` 同时使用时可能冲突。

处理方式：

1. 短期：collector 独占 MUPTI，kernel 时间从同一次运行的 MUPTI activity 或第二次对齐运行获取。
2. 中期：collector 内部合并 API callback、internal event、kernel activity，作为统一 profiler。
3. 长期：MUPTI 支持多 consumer 或 collector 提供转发接口。

## 9. 建模方法

### 9.1 软件状态

模型维护源码中的关键状态：

```text
RuntimeState:
  initialized
  export_table_ready
  current_device_per_thread
  module_cache
  function_cache

DriverState:
  current_context
  context_map
  stream_map
  event_map
  graph_map

MemoryState:
  pool_map
  free_buckets
  chunks
  allocated_objects
  fragmentation

StreamState:
  queue_depth
  dependency_edges
  inflight_count
  merge_list

GraphState:
  graph_nodes
  graph_edges
  dirty_nodes
  exec_resources
```

### 9.2 状态转移规则

状态转移规则从源码提取，不从 trace 猜测：

```text
muMemAlloc:
  Driver API validation
  context lookup
  Memory::Init
  MemMgr::Allocate
  MemoryPool::SubAllocate
  if pool miss:
    ChunkAllocate
    BO_ALLOC ioctl
    MAP ioctl
  if peer enabled:
    MapToPeers

muLaunchKernel:
  Runtime wrapper
  Driver API validation
  function lookup
  DispatchCommand create
  ResolveDependencyAndQueueCommand
  QueueCommand
  CommandBuild
  Submit
  Kernel queued/submitted

muStreamSynchronize:
  Driver API validation
  stream lookup
  WaitFinish
  wait until target timeline reached
```

### 9.3 成本项

模型成本来自内部事件持续时间：

```text
T_api =
  T_runtime_wrapper
  + T_driver_api_validation
  + T_context_lookup
  + T_object_lookup
  + T_core

T_memory =
  T_memory_tracker
  + T_pool_lookup
  + T_suballocate
  + miss_count * (T_chunk_allocate + T_bo_alloc_ioctl + T_map_ioctl)
  + peer_map_count * T_peer_map
  + T_free_merge

T_stream_submit =
  T_dependency_resolve
  + T_queue_command
  + T_command_build
  + T_merge_check
  + submit_count * T_submit_ioctl
  + T_inflight_wait

T_graph =
  T_validate
  + node_count * T_node_process
  + edge_count * T_edge_process
  + submission_count * T_prepare_submission
  + rebuild_count * T_rebuild
  + T_graph_launch
```

输出必须包含分项耗时，不能只输出总耗时。

### 9.4 校准

校准分三类：

| 类型 | 数据 | 用途 |
|---|---|---|
| 直接测量 | internal span begin/end | 固定成本和阶段成本 |
| 计数校准 | state/counter event | pool miss、submit 次数、graph rebuild 次数 |
| 排序校准 | kernel activity / profiler | Top50 kernel 排序和 API -> kernel 关系 |

校准结果写入 profile：

```yaml
profile:
  sdk_version: "<sdk>"
  driver_commit: "<commit>"
  runtime_commit: "<commit>"
  device: "<sku>"
  costs_ns:
    runtime_wrapper_base:
    driver_validation_base:
    context_lookup:
    memory_pool_lookup:
    memory_chunk_allocate:
    ioctl_bo_alloc:
    command_build:
    submit_ioctl:
    graph_node_process:
  thresholds:
    memory_pool_chunk_size:
    command_merge_limit:
    stream_async_capacity:
```

### 9.5 验证

验证不只比较总时间：

| 验证项 | 标准 |
|---|---|
| API 覆盖 | trace 中累计耗时 Top90% Driver API 有源码规则和埋点 |
| 状态路径 | pool hit/miss、submit、graph rebuild、sync wait 与事件一致 |
| 成本分项 | API self time、memory、stream、submit、sync 分项误差可解释 |
| kernel 关系 | API -> command -> submission -> kernel 关系可重建 |
| Top50 kernel | 按累计耗时排序 Top-K overlap 达到 90% 以上 |
| 版本对比 | 同一 workload 在两个 SDK 版本的差异能定位到源码模块 |

## 10. OKR 落地

### KR1：覆盖 Top90% Driver API

做法：

1. 用 collector 采集 3 个模型推理/训练的 API event。
2. 统计累计耗时 Top90% Driver API。
3. 为这些 API 建立源码状态转移规则。
4. 在规则涉及的内部成本点补齐 MUPTI internal event。
5. 输出每个 API 的分项成本和源码切点。

交付：

```text
top90_driver_api_list.csv
api_source_rule.yaml
instrumentation_coverage_report.md
api_cost_breakdown.parquet
```

### KR2：Top50 kernel 排序对齐

做法：

1. 采集 kernel register、queued、submitted、begin/end 事件。
2. 建立 API -> command -> submission -> kernel 关系。
3. 用 profiler kernel 时间校准 kernel duration。
4. 用模型预测 kernel 启动顺序和 host submit 延迟。
5. 比较 Top50 kernel 累计耗时排序的 Top-K overlap。

交付：

```text
kernel_relation_table.parquet
top50_kernel_overlap_report.md
kernel_launch_sequence_diff.md
```

### KR3：3 个模型性能建模报告

每个模型报告必须包含：

```text
API 热点
  Runtime self time
  Driver API self time
  memory / stream / graph / sync / ioctl 分项

kernel 热点
  Top50 kernel 排序
  API -> kernel 关系
  submit 延迟

源码瓶颈
  具体文件
  具体函数
  具体状态变量
  具体分支或等待原因

优化建议
  预期影响
  需要修改的模块
  风险
  验证方法
```

## 11. 与 CTS 的关系

CTS 不沉淀“某个模型的一段 trace”。CTS 沉淀可复现的软件行为。

建议沉淀以下行为用例：

| 行为 | CTS 用例 |
|---|---|
| memory pool hit/miss | 固定大小反复分配、跨 bucket 分配、碎片触发 |
| stream dependency | default stream、blocking stream、non-blocking stream、event wait |
| graph update/rebuild | 参数兼容更新、参数不兼容重建、fallback launch |
| module/function cache | 首次 kernel、重复 kernel、不同 module |
| sync wait | stream sync、event sync、context sync、隐式等待 |
| command merge/submit | 小 kernel 连续 launch、merge flush、inflight wait |

CTS 验收不只看 API 返回值，还要看事件签名：

```text
expected_event_signature:
  - DriverApiEnter(muMemAlloc)
  - MemoryPoolLookup
  - MemoryPoolMiss
  - ChunkAllocateBegin
  - BoAllocIoctlBegin
  - BoAllocIoctlEnd
  - ChunkAllocateEnd
  - DriverApiExit(muMemAlloc)
```

## 12. 实施计划

### 阶段 1：MUPTI collector 和最小事件闭环

范围：

1. 复用现有 Runtime/Driver API callback。
2. 复用已有 kernel、memcpy、memset、sync、graph tracepoint。
3. 新增 `EmitModelEvent` 或等价 internal hook。
4. collector 写出 `events.jsonl` 和 `run_metadata.yaml`。
5. 先覆盖 `muLaunchKernel`、`muMemAlloc`、`muStreamSynchronize`。

验收：

```text
能够在一个小 workload 中重建：
  API enter/exit
  memory pool hit/miss
  command queued/submitted
  stream sync wait
  API -> command -> kernel relation
```

### 阶段 2：Top90 API 源码规则和埋点覆盖

范围：

1. 对 3 个模型采集 API 热点。
2. 选出累计耗时 Top90% Driver API。
3. 为这些 API 建立 source rule。
4. 补齐 memory、stream、command、graph、sync、ioctl 内部事件。
5. 生成 coverage report。

验收：

```text
Top90% Driver API 均有：
  源码规则
  内部事件
  成本分项
  验证样例
```

### 阶段 3：白盒模型 v1

范围：

1. 实现 state reconstruction。
2. 实现 memory、stream、command、graph、sync 子模型。
3. 用 internal span 拟合成本项。
4. 输出 API 成本分解和 kernel 关系。

验收：

```text
模型能够解释：
  某 API 为什么慢
  多出来的 submit 来自哪里
  pool miss 来自哪里
  graph rebuild 来自哪里
  stream wait 等待哪个对象
```

### 阶段 4：模型报告和 CTS 沉淀

范围：

1. 对 3 个模型生成性能建模报告。
2. 对一个 SDK 版本建立行为基线。
3. 把典型状态路径沉淀为 CTS。
4. 跨 SDK 版本比较事件签名和成本变化。

验收：

```text
报告能够给出源码级优化建议。
CTS 能复现关键软件行为。
版本差异能定位到具体模块和事件。
```

## 13. 风险与处理

| 风险 | 影响 | 处理 |
|---|---|---|
| MUPTI 单 subscriber 限制 | 与 torch.profiler 冲突 | collector 合并采集，或分两次运行对齐 |
| 内部 event domain ABI 未确定 | 影响对外接口 | 先用 private hook，稳定后升级为正式 domain |
| 埋点过多影响性能 | 热路径扰动 | domain/event_id 开关、固定 payload、TLS ring buffer、采样 |
| correlation_id 不完整 | 无法串起 API 和 kernel | 在 API、command、submission、kernel 处统一生成和传递 |
| 时间戳时钟不一致 | 分项时间不可信 | host 事件用统一 monotonic clock，device 时间独立校准 |
| 事件字段频繁变化 | 跨版本无法对比 | event version、schema registry、兼容解析 |
| 源码规则维护成本高 | SDK 版本升级后模型失效 | source rule 绑定 commit，CI 检查埋点覆盖 |

## 14. Definition of Done

完成后应具备以下能力：

1. 对 3 个模型采集 Runtime API、Driver API、内部状态、command、graph、sync、kernel 关系事件。
2. 覆盖累计耗时 Top90% Driver API 的源码规则和内部埋点。
3. 输出 API 分项成本，不只输出 API 总耗时。
4. 重建 API -> command -> submission -> kernel 关系。
5. Top50 kernel 按累计耗时排序的 Top-K overlap 达到 90% 以上。
6. 报告能定位到具体源码文件、函数、状态变量和等待原因。
7. CTS 能复现 memory、stream、graph、sync、command merge 等关键行为。
8. 未开启 collector 时，埋点仅保留 low overhead 分支，不改变正常执行行为。

## 15. 当前优先级

优先做最小闭环：

```text
muLaunchKernel:
  API enter/exit
  function lookup
  DispatchCommand create
  ResolveDependencyAndQueueCommand
  QueueCommand
  CommandBuild
  Submit
  kernel queued/submitted

muMemAlloc:
  API enter/exit
  Memory::Init
  MemMgr::Allocate
  MemoryPool lookup
  pool hit/miss
  ChunkAllocate
  BO alloc/map ioctl

muStreamSynchronize:
  API enter/exit
  stream lookup
  wait begin/end
  wait target
  wait reason
```

这三个 API 覆盖 kernel launch、内存分配、同步等待三类核心软件成本。最小闭环跑通后，再扩展到 Top90% Driver API。
