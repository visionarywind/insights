# MUSA Driver/API 白盒软件性能建模落地方案

依据：`insights/okr/okr-2026.md`。  
仓库范围：`MUSA-Runtime`、`linux-ddk/musa`、`MUPTI`、`musa_benchmarks`。  
方案主线：源码埋点、MUPTI 采集、白盒建模、profiling 对齐。

## 1. 方案定位

本方案不把 Driver 当黑盒。模型不依赖“API 输入到 kernel 输出”的经验拟合，而是直接建模 Runtime 和 Driver 源码中的状态、分支、队列、缓存、同步、提交和系统调用边界。

执行路径如下：

```text
Runtime / Driver / HAL 源码
  -> 在关键状态转移和成本点埋点
  -> 通过 MUPTI callback / hook 输出事件
  -> collector 采集 API 事件、内部事件、kernel 事件
  -> musa_benchmarks 校准成本项并验证插桩开销
  -> 重建软件执行过程
  -> 拟合源码成本项
  -> 输出性能模型、瓶颈归因、优化建议
```

MUPTI 在本方案中是统一采集通道。埋点事件是模型证据，不是模型本身。模型规则来自源码，事件数据用于校准成本和验证规则。`musa_benchmarks` 用于成本校准、插桩开销验收和行为用例沉淀。

## 2. 建模对象

| 层级 | 建模内容 | 不建模内容 |
|---|---|---|
| `MUSA-Runtime` | `musa*` wrapper、`ApiInvocationGuard`、TLS、ExportTable、lazy init、module/function cache | GPU 侧执行 |
| Driver API | `mu*` wrapper、参数检查、context 获取、对象查找、domain 分发 | 硬件执行细节 |
| Core | context、stream、event、memory、module、graph、command 状态机 | SM/cache/warp 级行为 |
| HAL/M3D | memory pool、queue、command buffer、submit、KMD 边界 | KMD 内部调度细节 |
| MUPTI | API callback、command/kernel/graph tracepoint、内部模型事件采集 | 不作为性能模型规则来源 |
| `musa_benchmarks` | launch、memory、stream/event、graph、overhead、event signature 校准和验收 | 不替代真实模型 profiling |
| profiler kernel 数据 | kernel 名称、启动顺序、执行耗时、Top50 排序校准 | 不推导 Driver 内部成本 |

## 3. 现有基础

当前四个仓库已经具备以下基础：

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
| `MUPTI/src/injection/injection.cpp` | 注入 Driver/Runtime hook，导入 accessor |
| `MUPTI/src/api/activity.cpp`、`src/core/buffer.cpp` | activity enable、buffer request/complete、flush |
| `MUPTI/src/core/process_callback.cpp` | Runtime/Driver Tools callback 转 activity |
| `musa_benchmarks/memory` | malloc/free、memcpy、memset、host register 等校准用例 |
| `musa_benchmarks/schedule` | kernel launch、sync、graph、stream concurrency 等校准用例 |
| `musa_benchmarks/musaOnly` | MUSA 专属 launch、MCCL/CE/graph、atomic 等用例 |

已有机制能够覆盖 Runtime API、Driver API、kernel 注册、kernel submit、memcpy、memset、stream/context sync 和 graph 节点。新增工作重点是把 memory pool、stream queue、command build、submit、ioctl 边界等内部软件成本点纳入统一事件体系，并用 `musa_benchmarks` 建立可重复的校准和验收数据。

四个仓库的交付分工如下：

| 仓库 | 交付职责 |
|---|---|
| `MUSA-Runtime` | Runtime API callback 覆盖、Runtime 初始化和 cache 事件、Runtime 到 Driver 的 relation |
| `linux-ddk/musa` | Top90 Driver API source rule、Driver/Core/HAL/M3D 内部事件、源码级瓶颈归因 |
| `MUPTI` | ModelEvent hook、collector、schema、activity 支持矩阵、buffer 和 flush 验证 |
| `musa_benchmarks` | calibration suite、trace overhead suite、event signature CTS、跨 SDK baseline |

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
                    ^
                    |
        +------------------------+
        | musa_benchmarks        |
        | calibration / overhead |
        | event signature CTS    |
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
| relation 完整性 | API -> command -> submission -> kernel 关系重建召回率不低于 95% |
| 事件质量 | internal event drop rate、flush error、buffer overflow 必须可见 |
| 插桩开销 | trace off 影响在 benchmark 噪声内；trace on level 1 开销可量化 |
| 版本对比 | 同一 workload 在两个 SDK 版本的差异能定位到源码模块 |

### 9.6 `musa_benchmarks` 校准和验收

`musa_benchmarks` 不替代真实模型 profiling。它提供可重复、可隔离的成本校准和插桩验收。

| 方向 | 用例 | 校准或验收内容 |
|---|---|---|
| kernel launch | `kernelLaunchLatencyMode`、`kernelLaunchThroughputMode`、`kernelLaunchThroughputModeWithoutMerge`、`kernelLaunchApiLatency` | launch wrapper、command build、merge、submit |
| memory | `MallocRateFrom1Bto4GB`、`Copy1DAlignedRate`、`CopyPinnedRate`、`HostRegisterAndUnRegister` | pool hit/miss、alloc/free、H2D/D2H/D2D bandwidth |
| stream/event/sync | `efficiencyOfSync`、`streamConcurrencyCompute`、`streamConcurrencyMemcpy`、`parallelismOfDifferentCommands` | wait reason、event dependency、stream overlap |
| graph | `efficiencyOfGraph`、`efficiencyOfGraphLaunch`、`graphLaunchThroughputMode` | graph instantiate、graph launch、node relation |
| multi-card | `CopyP2PRate`、`kernelLaunchMulCards` | peer copy、multi-device context、multi-stream |
| trace overhead | 新增 `traceOffOnOverhead` | baseline、trace off、trace on level 1/2 的开销 |
| event signature | 新增 `modelEventSignature` | 断言关键事件序列完整 |

### 9.7 Roofline 与 Occupancy 辅助分析（补充缺口）

**动机**：当前方案聚焦 Runtime→Driver→KMD 软件栈，对 kernel 层只做 profiler 时间校准（KR2 §10）。缺少两个关键分析维度：(1) kernel 的计算/访存天花板分析，解释"为什么这个 kernel 是这个耗时"；(2) occupancy 分析，解释寄存器/shared memory 对 warp 并发度的限制。

**目标**：为每个 kernel 输出 `roofline_category`（compute-bound / memory-bound / balanced）和 `occupancy_breakdown`（active blocks/SM、limiting factor），作为 KR2 Top50 kernel 排序的辅助解释维度，不要求 cycle-accurate。

#### 9.7.1 峰值天花板计算

利用已有基础设施，无需新增 profiling：

| 天花板 | 数据来源 | 公式 |
|--------|---------|------|
| 峰值计算吞吐 | `musaDeviceProp.clockRate` (kHz) × `MUPTI_METRIC_PROPERTY_FLOP_SP_PER_CYCLE` | `peak_TFLOPS = clockRate × SMs × FLOPs_per_cycle / 1e6` |
| 峰值带宽 | `musaDeviceProp.memoryClockRate` (kHz) × `musaDeviceProp.memoryBusWidth` (bits) | `peak_GBs = memoryClockRate × 1000 × (memoryBusWidth / 8) × 2 / 1e9` |

`FLOP_SP_PER_CYCLE` 可从 `mupti_metrics.h` 的 `MUPTI_MetricProperty` 枚举查询（`MUPTI_METRIC_PROPERTY_FLOP_SP_PER_CYCLE`），也可用设备型号查表替代。

#### 9.7.2 算术强度计算

每 kernel 的算术强度 = `total_FLOPs / total_bytes_transferred`，数据来源：

| 数据 | 来源 | 说明 |
|------|------|------|
| `total_FLOPs` | msight-compute 指标规则 YAML（`mt_rules/`）或 MUPTI Profiling API | 通过 `mupti_profiler_target.h` 采集 kernel 级指令计数 |
| `total_bytes` | 同上，DRAM read/write throughput 指标 | 注意：仅 DRAM，不含 L1/L2 hit |

若 msight-compute 指标规则不可用，降级方案：从 ELF 解析指令序列，按 opcode 分类计数（PSC `pscopcodes.h` 的 `PSCOP_MAD`/`PSCOP_FMA` 为 FLOP，`PSCOP_LD`/`PSCOP_ST`/`PSCOP_DMA` 为访存），乘以网格规模估算。

#### 9.7.3 Occupancy 分解

利用 `musa_occupancy.h` 的 standalone 计算器（不需要 GPU）：

```text
输入：musaOccDeviceProp（从 musaDeviceProp 构造）
      musaOccFuncAttributes（从 musaFuncGetAttributes 或 MusaMeta ELF 解析构造）
      blockSize
      dynamicSmemSize

输出：musaOccResult
  - activeBlocksPerMultiprocessor
  - limitingFactors（WARPS / REGISTERS / SHARED_MEMORY / BLOCKS）
  - allocatedRegistersPerBlock, allocatedSharedMemPerBlock
```

`MusaMeta.h` 提供离线 ELF 解析：`KernMeta.attr_reg_count`、`temp_reg_count`、`slot_reg_count`、`shared_memory_size`、`max_block_size`，可直接构造 `musaOccFuncAttributes`，无需加载 kernel。

#### 9.7.4 模型输出增强

每个 kernel 的输出增加以下字段：

```yaml
kernel:
  name: "fused_attention_kernel"
  duration_us: 142.3
  roofline_category: "compute_bound"     # 新增
  arithmetic_intensity: 12.4             # 新增：FLOP/byte
  peak_compute_pct: 78.5                 # 新增：实测/峰值 TFLOPS
  peak_bandwidth_pct: 32.1               # 新增：实测/峰值 GB/s
  occupancy:
    active_blocks_per_sm: 4              # 新增
    limiting_factor: "REGISTERS"         # 新增
    theoretical_occupancy_pct: 50.0      # 新增
    regs_per_thread: 64                  # 新增
    smem_per_block_kb: 32                # 新增
```

#### 9.7.5 可复用基础设施清单

| 能力 | 位置 | 状态 |
|------|------|------|
| 离线 occupancy 计算 | `MUSA-Runtime/include/musa_occupancy.h` | ✅ standalone header |
| Kernel 资源元数据 | `linux-ddk/.../musameta/MusaMeta.h` | ✅ ELF 解析 |
| 设备属性查询 | `musa_runtime_api.h` `musaDeviceProp` | ✅ Runtime API |
| 峰值 FLOP 属性 | `MUPTI/include/mupti_metrics.h` | ✅ MetricProperty |
| PSC 指令 opcode 定义 | `linux-ddk/.../psc/inc/pscopcodes.h` | ✅ 30+ opcodes |
| PSC 反汇编器 | `linux-ddk/.../psc/src/decode/sudi/disasm.cpp` | ✅ |
| msight-compute 指标规则 | 远程 `msight-compute/module/mt_rules/` | ⚠️ 需拉取 |
| DRAM 读写指标 | 同上 | ⚠️ 需确认指标名称 |
| 指令周期延迟表 | **不存在** | ❌ 需微基准测量或内部文档 |

#### 9.7.6 实施优先级

| 阶段 | 内容 | 依赖 |
|------|------|------|
| 阶段二（随 OKR 主线） | 峰值天花板计算 + occupancy 分解 | `musa_occupancy.h` + `MusaMeta.h`（已就绪） |
| 阶段二（随 OKR 主线） | 算术强度计算（msight-compute 指标） | msight-compute 指标规则拉取 |
| 阶段三（工程化） | PSC 指令级成本模型（替代 profiler 依赖） | 微基准或内部文档提供 cycle latency |

建议新增输出：



```text
benchmark_results.csv
model_events.jsonl
calibration_features.parquet
overhead_report.md
event_signature_report.md
```

## 10. OKR 落地

### KR1：覆盖 Top90% Driver API

做法：

1. 用 collector 采集 3 个模型推理/训练的 Runtime API、Driver API、internal event 和 activity。
2. 同时统计 `host_self_time`、`inclusive_time`、`sync_wait_time`，避免把同步等待和 API 自身开销混在一起。
3. 选出累计耗时 Top90% Driver API。
4. 为这些 API 建立 source rule，包含源码路径、状态转移、事件签名、payload、成本项和验证用例。
5. 在规则涉及的内部成本点补齐 MUPTI ModelEvent。
6. 用 `musa_benchmarks` 校准 launch、memory、stream/event、graph、sync 的基础成本。
7. 输出每个 API 的分项成本和源码切点。

交付：

```text
top90_driver_api_list.csv
api_source_rule.yaml
instrumentation_coverage_report.md
api_cost_breakdown.parquet
benchmark_calibration_report.md
```

### KR2：Top50 kernel 排序对齐

做法：

1. 采集 kernel register、queued、submitted、begin/end 事件。
2. 建立 API -> command -> submission -> kernel 关系。
3. 统计 relation 重建召回率，记录无法关联的 API、command、submission 和 kernel。
4. 用 profiler kernel 时间校准 kernel duration。
5. 用模型预测 kernel 启动顺序、host submit 延迟和 stream gap。
6. 比较 Top50 kernel 累计耗时排序的 Top-K overlap。
7. 输出误差归因：relation 缺失、kernel name 归并错误、stream 顺序错误、submit 延迟错误、device duration 误差。

#### 补充：设备侧 kernel 排序模型（补充缺口）

**动机**：当前 KR2 的 kernel 排序完全依赖 profiler 实测 `kernel duration`（步骤 4："用 profiler kernel 时间校准 kernel duration"）。这提供了高精度但存在两个局限：(1) 必须运行 profiler，无法在离线/无 GPU 场景做预测；(2) 无法解释两个 kernel 耗时差异的根源（是寄存器压力、共享内存限制、还是计算密度差异）。

**目标**：增加一层轻量 kernel 成本预测模型，不要求 cycle-accurate，只要求 kernel 间相对排序与 profiling 对齐（Top-K overlap 90%+）。profiler 数据仍为主校准源，预测模型用于辅助排序验证和差异归因。

**方法**：指令分类 × 资源缩放模型

```text
kernel_cost_predicted =
    (Σ opcode_weight[i] × instruction_count[i])  # 指令级基础成本
    × occupancy_scale(regs, smem, block_size)     # 资源并发度缩放
    / num_sms                                       # SM 级并行
```

**指令分类**（基于 PSC opcodes，来源 `pscopcodes.h`）：

| 类别 | 包含 opcode | 权重来源 |
|------|------------|---------|
| 计算密集型 | `PSCOP_MAD`, `PSCOP_FMA`, `PSCOP_ADD`, `PSCOP_MUL`, `PSCOP_DP` 等 | 微基准校准或固定权重（如 MAD=4, ADD=2） |
| 访存密集型 | `PSCOP_LD`, `PSCOP_ST`, `PSCOP_DMA` | 微基准校准（含 DRAM/L1/L2 分层权重） |
| 控制流 | `PSCOP_BRANCH`, `PSCOP_BARRIER` | 固定开销权重 |
| 特殊功能 | `PSCOP_SFU`, `PSCOP_TEX` | 微基准校准 |

**资源缩放因子**（利用 `musa_occupancy.h` standalone 计算器）：

```text
occupancy_scale = 1.0 / occupancy_pct
其中 occupancy_pct = active_blocks_per_sm / max_blocks_per_sm
```

解释：occupancy 低 → 更多 blocks 串行执行 → 单 block 成本被放大。

**指令计数获取路径**：

| 路径 | 精度 | 适用场景 |
|------|------|---------|
| **路径 A**：msight-compute 采集指令计数（`inst_executed` 等 MUPTI 指标） | 高 | 在线 profiling |
| **路径 B**：PSC 反汇编器解析 ELF → 按 opcode 分类计数 | 中 | 离线预测，无 GPU |
| **路径 C**：从 kernel 元数据估算（regs × smem × block_size 查表） | 低 | 快速初筛 |

**模型校准**：

```text
calibration_target:
  对 3 个模型的 Top50 kernel：
    - 路径 A：采集实际指令计数，拟合 opcode_weight[] 使预测误差最小
    - 路径 B：反汇编指令计数，同上拟合
    - 验证：Top-K overlap 90%+ 且 MAE < 15%
```

**可复用基础设施**：

| 能力 | 位置 | 状态 |
|------|------|------|
| PSC opcode 定义 | `linux-ddk/.../pscopcodes.h` | ✅ 30+ opcodes |
| PSC 反汇编器 | `linux-ddk/.../disasm.cpp` | ✅ |
| ELF 解析 (regs/smem) | `linux-ddk/.../MusaMeta.h` | ✅ |
| occupancy 计算 | `MUSA-Runtime/include/musa_occupancy.h` | ✅ standalone |
| MUPTI 指令计数指标 | msight-compute `mt_rules/` | ⚠️ 需拉取确认指标名 |
| opcode 延迟表 | **不存在** | ❌ 需微基准测量 |

**与 profiler 校准的关系**：

```text
            profiler kernel duration（主校准源，高精度）
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
     kernel 排序验证   差异归因      离线预测
    (Top-K overlap)  (为什么A比B慢)  (无GPU场景)
                          │
            ┌─────────────┘
            ▼
     指令分类 × 资源缩放模型（辅助，中等精度）
```

profiler 实测数据始终是主校准源。预测模型提供补充解释和离线能力，不替代 profiler。

交付：

```text
kernel_relation_table.parquet
top50_kernel_overlap_report.md
kernel_launch_sequence_diff.md
relation_recall_report.md
event_loss_report.md
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

校准和验收
  musa_benchmarks 对应 case
  trace off / trace on 开销
  event signature 完整性
  跨 SDK baseline 差异
```

## 11. 与 CTS 的关系

CTS 不沉淀“某个模型的一段 trace”。CTS 沉淀可复现的软件行为。

第一阶段建议先把行为用例放入 `musa_benchmarks` 或配套 CTS 目录。用例稳定后，再迁移到正式 MUSA CTS。

建议沉淀以下行为用例：

| 行为 | CTS 用例 |
|---|---|
| memory pool hit/miss | 固定大小反复分配、跨 bucket 分配、碎片触发 |
| stream dependency | default stream、blocking stream、non-blocking stream、event wait |
| graph update/rebuild | 参数兼容更新、参数不兼容重建、fallback launch |
| module/function cache | 首次 kernel、重复 kernel、不同 module |
| sync wait | stream sync、event sync、context sync、隐式等待 |
| command merge/submit | 小 kernel 连续 launch、merge flush、inflight wait |
| attention / MLP | dense decoder 中连续 GEMM、softmax、elementwise、transpose/view |
| KV cache | cache allocate、cache copy、stream overlap、host/device transfer |
| expert routing | MoE topk、index/gather/scatter、expert combine |

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

### 阶段 0：模型选择和现有数据盘点

范围：

1. 选定 3 个模型，覆盖 dense decoder、MoE、训练或长上下文场景。
2. 使用现有 profiler/MUPTI 能力采集 API、kernel、memcpy、graph、sync 数据。
3. 输出 Top API、Top kernel、sync wait、graph、memory 热点。
4. 列出当前 trace 无法解释的缺口。

验收：

```text
top_api.csv
top_kernel.csv
trace_gap_report.md
model_workload_config.yaml
```

### 阶段 1：MUPTI collector 和最小事件闭环

范围：

1. 复用现有 Runtime/Driver API callback。
2. 复用已有 kernel、memcpy、memset、sync、graph tracepoint。
3. 新增 `EmitModelEvent` 或等价 private hook。
4. collector 写出 `events.jsonl`、`relations.parquet` 和 `run_metadata.yaml`。
5. 先覆盖 `muLaunchKernel`、`muMemAlloc`、`muStreamSynchronize`。

验收：

```text
能够在一个小 workload 中重建：
  API enter/exit
  memory pool hit/miss
  command queued/submitted
  stream sync wait
  API -> command -> submission -> kernel relation
```

### 阶段 2：Top90 API 源码规则和埋点覆盖

范围：

1. 对 3 个模型采集 API 热点。
2. 按 `host_self_time`、`inclusive_time`、`sync_wait_time` 三个口径选出 Top90 Driver API。
3. 为这些 API 建立 source rule。
4. 补齐 memory、stream、command、graph、sync、ioctl 内部事件。
5. 生成 coverage report 和 event signature report。

验收：

```text
Top90% Driver API 均有：
  源码规则
  内部事件
  成本分项
  验证样例
  事件签名
```

### 阶段 3：`musa_benchmarks` 校准和插桩开销验收

范围：

1. 新增或整理 launch、memory、stream/event、graph、overhead、event signature 用例。
2. 对比 baseline、trace off、trace on level 1、trace on level 2。
3. 生成成本校准数据和插桩开销报告。
4. 对关键 source rule 做最小可复现验证。

验收：

```text
benchmark_results.csv
calibration_features.parquet
overhead_report.md
event_signature_report.md

trace off:
  性能变化 <= 0.1%，或在 benchmark 噪声内

trace on level 1:
  端到端开销可量化，建议 <= 1%-3%
```

### 阶段 4：白盒模型 v1

范围：

1. 实现 state reconstruction。
2. 实现 memory、stream、command、graph、sync、ioctl 子模型。
3. 用 internal span 和 `musa_benchmarks` 结果拟合成本项。
4. 输出 API 成本分解、kernel 关系和源码级瓶颈归因。

验收：

```text
模型能够解释：
  某 API 为什么慢
  多出来的 submit 来自哪里
  pool miss 来自哪里
  graph rebuild 来自哪里
  stream wait 等待哪个对象
  ioctl 边界消耗多少时间
```

### 阶段 5：模型报告和 CTS 沉淀

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

## 13. 回归门禁与自动化验收（补充缺口）

**动机**：当前方案对跨版本稳定性、插桩开销上限、kernel 排序精度等关键指标缺乏量化定义。`musa_samples/benchmarks/pytest/` 有基于 `benchmark_cfg.csv` 的阈值门禁（`result < baseline × factor → FAIL`），`musa_benchmarks/` 有基于 `calculateScoreOfSuit.py` 的相对评分体系，但两者均未与 OKR 性能建模的验收标准对齐。

**目标**：定义明确的量化回归门禁指标，覆盖模型精度、插桩开销、事件质量三个维度，并与现有 CI 基础设施集成。

### 13.1 回归门禁指标体系

| 门禁项 | 阈值 | 测量方法 | 阻断级别 |
|--------|------|---------|---------|
| **Top90 API 建模误差 (MAE)** | < 5% | 对 Top90 Driver API 的 `predicted_self_time` vs `profiled_self_time` 计算 MAE | **阻断** |
| **Top50 kernel 排序 overlap** | > 90% | 按累计耗时排序，Top-K overlap（K=10, 20, 50 三个点） | **阻断** |
| **API→kernel 关系召回率** | ≥ 95% | `matched_relations / total_api_calls` | **阻断** |
| **插桩开销 (trace on level 1)** | < 1% 端到端时间 | `(trace_on_wall_time - trace_off_wall_time) / trace_off_wall_time` | **阻断** |
| **插桩开销 (trace on level 2)** | < 3% 端到端时间 | 同上 | **警告** |
| **事件丢失率** | < 0.1% | `drop_count / emit_count` | **阻断** |
| **Buffer overflow 次数** | = 0 | collector buffer flush 错误计数 | **阻断** |
| **事件签名完整性** | 100% 关键事件序列出现 | 每个 API 的 `expected_event_signature` vs 实际采集 | **阻断** |
| **Roofline category 一致性** | 分类一致率 > 95% | 模型预测 `roofline_category` vs 基于 profiler 实测 FLOP/带宽计算的类别 | **警告** |
| **Occupancy 预测误差** | < 10% | 模型预测 `active_blocks_per_sm` vs 实际 occupancy（可通过 profiler 反推） | **信息** |

### 13.2 跨版本回归门禁

| 门禁项 | 阈值 | 说明 |
|--------|------|------|
| SDK minor 版本（1.x→1.y） | 上述所有阻断门禁仍满足 | 不重训模型，直接验证 |
| SDK major 版本（1.x→2.0） | 上述所有阻断门禁仍满足 | 完整重建模型后验证 |
| 源码规则变更 | 变更的 API 需重新验收事件签名 | source rule 绑定 commit hash |

### 13.3 CI 集成方案

利用现有 CI 基础设施：

| 阶段 | 现有基础设施 | 新增内容 |
|------|------------|---------|
| 构建 | `MUPTI/.ciConfig.yaml` (x86_64/aarch64 Release/Debug) | 新增 `build.perf_model` job：编译 collector + 模型代码 |
| 单元测试 | `MUSA-Runtime/unittest/` CTest | 新增 `perf_model` 测试目录：事件 schema 验证、cost 项单元测试 |
| 微基准 | `musa_benchmarks/scripts/autorun.py` | 新增 `calibration` suite：launch/memory/sync 成本微基准自动运行 |
| 回归门禁 | `musa_samples/benchmarks/pytest/` 阈值比较 | 新增 `perf_model_gate.py`：读取模型输出 parquet，逐项检查 §13.1 门禁 |
| 评分 | `musa_benchmarks/scripts/calculateScoreOfSuit.py` | 新增 `model_score` 维度：基于 MAE 的对数评分（MAE=1%→score=2.0, MAE=5%→score≈1.3, MAE=10%→score=1.0） |
| 报告 | Allure + InfluxDB + CSV | 新增 `gate_report.md`：逐项 PASS/WARN/FAIL + 趋势图（InfluxDB 时间序列） |

**CI pipeline 流程**：

```text
commit push
  → 构建（MUSA-Runtime + MUPTI + musa_benchmarks）
  → 部署到测试节点（SSH remote_test.sh 模式）
  → 运行 3 个模型 workload（trace on level 1）
  → collector 采集 + 模型重建
  → perf_model_gate.py 检查 §13.1 门禁
  → musa_benchmarks calibration suite 验证成本项
  → 生成 gate_report.md
  → 任一阻断门禁 FAIL → pipeline 阻断
```

### 13.4 评分映射（兼容现有 musa_benchmarks 对数评分）

```text
model_score = baseScore × Σ(caseWeight × score_ratio) / totalWeight

其中 score_ratio = 1 + log10(threshold / actual_error)
  - MAE = 1% → score_ratio ≈ 1.70
  - MAE = 5%（阈值）→ score_ratio = 1.00（基线）
  - MAE = 10% → score_ratio ≈ 0.70
  - MAE = 50% → score_ratio = 0.00

model_score 可与现有 musa_benchmarks 的 memoryOp/mulStreams 等 suite 评分
合并为总评，用于跨 SDK 版本对比。
```

### 13.5 门禁配置文件

新增 `musa_benchmarks/TestSuitConfig.json` 扩展：

```json
{
  "suites": {
    "perfModel": {
      "baseline": "profiler_ground_truth",
      "baseScore": 1000,
      "gates": {
        "api_mae_pct": { "max": 5.0, "level": "block" },
        "kernel_top50_overlap_pct": { "min": 90.0, "level": "block" },
        "relation_recall_pct": { "min": 95.0, "level": "block" },
        "trace_overhead_level1_pct": { "max": 1.0, "level": "block" },
        "trace_overhead_level2_pct": { "max": 3.0, "level": "warn" },
        "event_drop_rate_pct": { "max": 0.1, "level": "block" },
        "buffer_overflow_count": { "max": 0, "level": "block" },
        "event_signature_completeness_pct": { "min": 100.0, "level": "block" },
        "roofline_category_consistency_pct": { "min": 95.0, "level": "warn" },
        "occupancy_error_pct": { "max": 10.0, "level": "info" }
      }
    }
  }
}
```

### 13.6 与现有回归基础设施的关系

| 现有系统 | 定位 | 与 OKR 门禁的关系 |
|---------|------|------------------|
| `musa_samples/benchmarks/pytest/` | 内存带宽 + kernel 延迟微基准回归 | **不替代**：专注于底层 Driver 性能，OKR 门禁专注于模型精度 |
| `musa_benchmarks/` Celero 评分 | 跨平台（CUDA/MUSA/ROCm）Driver 性能对比 | **补充**：新增 `perfModel` suite 评分并入总评 |
| `MUSA-Runtime/unittest/` CTest | Runtime API 正确性 | **不重叠**：OKR 门禁新增独立的模型精度测试 |
| `MUPTI/.ciConfig.yaml` | MUPTI 构建 + CTS 冒烟 | **扩展**：新增 `perf_model` 构建 job |

## 14. 风险与处理

| 风险 | 影响 | 处理 |
|---|---|---|
| MUPTI 单 subscriber 限制 | 与 torch.profiler 冲突 | collector 合并采集，或分两次运行对齐 |
| 内部 event domain ABI 未确定 | 影响对外接口 | 先用 private hook，稳定后升级为正式 domain |
| 埋点过多影响性能 | 热路径扰动 | domain/event_id 开关、固定 payload、TLS ring buffer、采样 |
| correlation_id 不完整 | 无法串起 API 和 kernel | 在 API、command、submission、kernel 处统一生成和传递 |
| 时间戳时钟不一致 | 分项时间不可信 | host 事件用统一 monotonic clock，device 时间独立校准 |
| 事件字段频繁变化 | 跨版本无法对比 | event version、schema registry、兼容解析 |
| 源码规则维护成本高 | SDK 版本升级后模型失效 | source rule 绑定 commit，CI 检查埋点覆盖 |
| `musa_benchmarks` 覆盖不足 | 成本校准和插桩开销无法闭环 | 新增 calibration、overhead、event signature 用例 |
| Top50 kernel overlap 被误用 | 只证明 kernel 列表对齐，不证明软件模型正确 | 增加 relation recall、事件质量、成本分项误差 |
| PSC 指令周期延迟表缺失 | 设备侧 kernel 成本模型精度不足 | 先用 msight-compute 指令计数 + 微基准校准；延迟表缺失时用固定权重降级方案 |
| msight-compute 指标规则不可访问 | 算术强度和指令计数无法采集 | 降级方案：PSC 反汇编器解析 ELF 做 opcode 分类计数；或仅用 profiler kernel duration 做纯查表模型 |
| 回归门禁首次集成失败率高 | CI pipeline 频繁阻断，影响开发效率 | 门禁分阶段上线：先 warning-only → 再 block；预留门禁跳过机制（需审批） |

## 15. Definition of Done

完成后应具备以下能力：

1. 对 3 个模型采集 Runtime API、Driver API、内部状态、command、graph、sync、kernel 关系事件。
2. 覆盖累计耗时 Top90% Driver API 的源码规则和内部埋点。
3. 输出 API 分项成本，不只输出 API 总耗时。
4. 重建 API -> command -> submission -> kernel 关系。
5. API -> command -> submission -> kernel 关系重建召回率不低于 95%。
6. Top50 kernel 按累计耗时排序的 Top-K overlap 达到 90% 以上。
7. internal event drop rate、flush error、buffer overflow 可统计；超过阈值时报告标记为无效。
8. 报告能定位到具体源码文件、函数、状态变量和等待原因。
9. `musa_benchmarks` 能完成成本校准、trace overhead 验收和 event signature 验收。
10. CTS 能复现 memory、stream、graph、sync、command merge、KV cache、attention/MLP、expert routing 等关键行为。
11. 未开启 collector 时，埋点仅保留 low overhead 分支，不改变正常执行行为。
12. 每个 kernel 输出 `roofline_category`（compute-bound / memory-bound / balanced）和 `occupancy_breakdown`（active blocks/SM、limiting factor），用于解释 kernel 性能差异的根源。
13. 设备侧 kernel 排序模型（指令分类 × 资源缩放）的 Top-K overlap 与 profiling 对齐 90%+，MAE < 15%。
14. 回归门禁自动化：阻断门禁（MAE、overlap、召回率、插桩开销、事件丢失率）全部通过；跨 SDK 版本回归报告自动生成。

## 16. 当前优先级

优先做最小闭环：

```text
MUPTI / collector:
  EmitModelEvent private hook
  events.jsonl
  relations.parquet
  drop_count / flush_error_count

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

musa_benchmarks:
  kernelLaunchLatencyMode
  MallocRateFrom1Bto4GB
  efficiencyOfSync
  traceOffOnOverhead
  modelEventSignature
```

这三个 API 覆盖 kernel launch、内存分配、同步等待三类核心软件成本。配套 benchmark 用于校准基础成本和验证插桩开销。最小闭环跑通后，再扩展到 Top90% Driver API。
