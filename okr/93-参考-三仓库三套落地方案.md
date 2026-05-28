# MUPTI / musa / MUSA-Runtime 白盒性能建模 OKR 三套落地方案

## 1. 结论

当前 OKR 不应把 Driver 当成黑盒。可落地的方向是：

1. 从 `MUSA-Runtime` 和 `linux-ddk/musa` 源码提取 API、状态机、队列、内存池、同步和提交路径。
2. 在关键状态转移处埋点，事件通过 MUPTI 统一采集。
3. 用事件重建一次模型执行中的 Runtime、Driver、Core、HAL/M3D 成本分布。
4. 用 profiling 的 kernel、memcpy、sync 结果校准模型，不用 profiling 直接替代模型。

推荐路线：

| 阶段 | 方案 | 用途 |
| --- | --- | --- |
| M0-M1 | 方案一：最小闭环方案 | 快速证明 MUPTI 埋点链路、事件 schema、API 到 kernel 关系可用 |
| M2-M4 | 方案二：OKR 主线方案 | 覆盖 Top90 Driver API，完成 3 个模型的白盒性能建模和行为分析 |
| Q4/SDK 发布 | 方案三：工程化方案 | 固化为 SDK 版本级工具链、CTS、回归和跨版本报告 |

方案二应作为 OKR 主方案。方案一是方案二的第一阶段，方案三是方案二稳定后的工程化升级。

## 2. 当前远程源码基础

远程主机：`shanfeng@10.18.32.25`  
远程根目录：`/home/shanfeng/workspace`

本次分析聚焦三个仓库：

| 仓库 | 角色 | 当前可用基础 | 当前缺口 |
| --- | --- | --- | --- |
| `MUPTI` | 统一采集层 | 已有 injection、Tools callback、activity buffer、Runtime/Driver activity、kernel/memcpy/sync tracepoint 对接 | 缺少 memory pool、stream dependency、command build/merge/submit、HAL/M3D ioctl 等内部软件成本事件 |
| `MUSA-Runtime` | Runtime API 层 | 已有 `ApiTrace`、`ApiInvocationGuard`、Runtime MUPTI hook、Runtime export table、Runtime wrapper callback | 缺少 Runtime 初始化、module/function cache、Runtime 到 Driver 映射的细分事件 |
| `linux-ddk/musa` | Driver/Core/HAL 实现层 | 已有 Driver Tools callback、Driver MUPTI hook、stream/command/memory/core 源码路径、已有部分 MUPTI tracepoint | 缺少可用于白盒建模的统一 `ModelEvent`；缺少内部等待原因、队列长度、merge 决策、pool hit/miss、ioctl 边界成本 |

当前仓库已有调试日志改动。后续实施应保留这些改动，不应回退用户已有插桩。

## 3. 三仓库现有调用链

### 3.1 MUPTI 当前链路

关键源码：

| 文件 | 作用 |
| --- | --- |
| `MUPTI/src/injection/injection.cpp` | 加载 `libmusa.so` 和 `libmusart.so`，获取 Driver/Runtime export table，注入 MUPTI hook，订阅 Tools callback |
| `MUPTI/src/core/process_callback.cpp` | 把 Runtime/Driver Tools callback 转换为 MUPTI callback 和 activity record |
| `MUPTI/src/api/activity.cpp` | 实现 `muptiActivityEnable`、`muptiActivityRegisterCallbacks`、`muptiActivityGetNextRecord`、`muptiActivityFlushAll` |
| `MUPTI/src/core/buffer.h`、`MUPTI/src/core/buffer.cpp` | 管理 activity buffer、record 分配、flush、dropped record 统计 |

当前 MUPTI 的实际流程：

```text
collector / profiler
  -> 调用 muptiActivityRegisterCallbacks
  -> 调用 muptiActivityEnable(Runtime / Driver / Kernel / Memcpy / Sync)
  -> MUPTI injection 加载 Driver / Runtime export table
  -> MUPTI 向 Driver Tools callback subscribe
  -> Runtime / Driver wrapper 在 API enter/exit 时 IssueCallback
  -> MUPTI ProcessDriverCallback 生成 callback 或 activity record
  -> ActivityBufferManager 写入用户 buffer
  -> collector 调用 muptiActivityFlushAll / GetNextRecord 读取数据
```

当前 MUPTI 能回答的问题：

| 问题 | 是否可回答 | 依据 |
| --- | --- | --- |
| 调用了哪些 Runtime API | 可以 | Runtime wrapper + Tools callback |
| 调用了哪些 Driver API | 可以 | Driver wrapper + Tools callback |
| API enter/exit 时间 | 可以 | `ProcessDriverCallback` 记录 start/end |
| kernel、memcpy、sync 与 correlation id 的关系 | 部分可以 | Driver tracepoint 已覆盖一部分 |
| API 内部为何耗时 | 不完整 | 缺少内部状态机事件 |
| memory pool 是否命中 | 不完整 | 当前 activity 看不到 pool 分支 |
| command 是否 merge，为什么 merge 或停止 merge | 不完整 | 当前 tracepoint 未覆盖 build/merge 决策 |
| submit 到 HAL/M3D/ioctl 的成本 | 不完整 | 需要在 Core/HAL/M3D 补边界事件 |

### 3.2 MUSA-Runtime 当前链路

关键源码：

| 文件 | 作用 |
| --- | --- |
| `MUSA-Runtime/src/internal.h` | `ApiInvocationGuard`、`ApiTrace`、`EnterRuntimeApi`、`ExitRuntimeApi` |
| `MUSA-Runtime/src/mupti/hooks.h` | Runtime MUPTI hook storage 和 ready 开关 |
| `MUSA-Runtime/src/musa_entry.cpp` | `musaGetExportTable` 返回 Runtime MUPTI controller |
| `MUSA-Runtime/src/musa_wrappers_generated.cpp` | Runtime API wrapper 生成 Runtime Tools callback |
| `MUSA-Runtime/src/musa_memory.cpp`、`src/musa_stream.cpp`、`src/musa_module.cpp` | Runtime API 转 Driver API 的主要入口 |

Runtime 侧的典型 API 调用流程：

```text
用户调用 musaLaunchKernel / musaMemcpyAsync / musaStreamSynchronize
  -> Runtime wrapper 创建 ApiTrace
  -> ApiTrace 初始化 Driver export table
  -> ApiInvocationGuard 记录 API 调用上下文和返回值
  -> 如果 Tools callback 开启，wrapper 发送 Runtime API enter callback
  -> 调用 musaapiXXX 实现
  -> musaapiXXX 通过 g_ExportTable.GetDriverTable() 调 Driver API
  -> 如果 Tools callback 开启，wrapper 发送 Runtime API exit callback
  -> ApiTrace 析构，执行 Runtime MUPTI hook exit
```

Runtime 层对模型的价值：

| 成本项 | 说明 |
| --- | --- |
| Runtime wrapper 成本 | API 包装、参数转换、错误状态处理 |
| Runtime 初始化成本 | 首次 API、export table 初始化、设备初始化 |
| Runtime 到 Driver 映射 | 一个 Runtime API 会对应一个或多个 Driver API |
| module/function cache 成本 | kernel launch 首次加载、符号查找、cache 命中/未命中 |

Runtime 层不适合作为 Driver 内部建模来源。Runtime 只能解释 Runtime 自身和 Runtime 到 Driver 的边界，Driver 内部队列、内存池、提交路径必须在 `linux-ddk/musa` 分析。

### 3.3 linux-ddk/musa 当前链路

关键源码：

| 模块 | 文件 | 作用 |
| --- | --- | --- |
| Driver callback | `src/driver/callback.cpp` | `toolsSubscribe`、`toolsEnableCallback`、`IssueCallback`，目前单 subscriber |
| Driver wrapper | `src/driver/mu_wrappers_generated.cpp` | Driver API enter/exit callback |
| Driver API 实现 | `src/driver/mu_memory.cpp`、`mu_stream.cpp`、`mu_module.cpp`、`mu_context.cpp`、`mu_device.cpp` | API 参数检查、对象获取、调用 Core |
| MUPTI hook | `src/driver/mupti/hooks.h`、`hooks.cpp` | Driver MUPTI hook ready 开关和 hook 存储 |
| MUPTI tracepoint | `src/driver/mupti/tracepoints.h` | kernel、memcpy、memset、sync、context、stream、submission relation 的现有 tracepoint |
| Stream | `src/musa/core/stream.cpp` | `QueueCommand`、`AsyncSubmit`、`AsyncWait`、同步等待 |
| Command | `src/musa/core/command/*.cpp` | `Build`、`Submit`、`SubmitToQueue` |
| Memory pool | `src/musa/core/memoryPool.cpp` | pool 初始化、权限修改、paging、trim |
| HAL/M3D | `src/hal/m3d/queue.cpp`、`src/hal/m3d/m3d/src/core/queue.cpp`、`src/hal/m3d/m3d/src/core/os/drm/mthreads/mtgpuQueue.cpp`、`mtgpuDevice.cpp` | HAL queue submit、M3D submit、DRM/KMD 提交 |

Driver/Core/HAL 的典型 kernel launch 路径：

```text
Runtime musaLaunchKernel
  -> Driver muLaunchKernel wrapper
  -> Driver muapiLaunchKernel
  -> Core Stream::CmdLaunchKernel
  -> Context::ResolveDependencyAndQueueCommand
  -> Stream::QueueCommand
  -> Stream::AsyncSubmit
  -> Command::Build
  -> DispatchCommand::Submit
  -> Command::SubmitToQueue
  -> HAL Queue::Submit
  -> M3D Queue::SubmitInternal
  -> M3D Queue::SubmitCommandBuffer
  -> DRM Queue::OsSubmit
  -> mtgpu Queue::LaunchCommandStreams
  -> mtgpu Device::SubmitCommands / SubmitCommandsWithDoorbell
  -> KMD / firmware
```

Driver/Core/HAL 的典型 async memcpy 路径：

```text
Runtime musaMemcpyAsync
  -> Driver muMemcpyAsync / muMemcpyHtoDAsync / muMemcpyDtoHAsync
  -> Core Stream::CmdCopyMemory
  -> Context::ResolveDependencyAndQueueCommand
  -> Stream::QueueCommand
  -> Stream::AsyncSubmit
  -> AsyncMemcpyCommand::Build
  -> AsyncMemcpyCommand::Submit
  -> Command::SubmitToQueue
  -> HAL/M3D submit
```

Driver/Core 的典型同步路径：

```text
Runtime musaStreamSynchronize
  -> Driver muStreamSynchronize wrapper
  -> Driver muapiStreamSynchronize
  -> Core Stream::WaitFinish
  -> Command::Wait 或 Stream::AsyncWait
  -> timeline / hardware semaphore wait
  -> engine last error query
  -> 更新 user pool 状态
```

这些源码路径决定白盒模型的分项结构。

## 4. 白盒性能模型的基本形式

每个 API 的耗时应拆成可解释分项：

```text
T_api =
  T_runtime_wrapper
+ T_driver_wrapper
+ T_validation
+ T_object_lookup
+ T_state_transition
+ T_memory_pool
+ T_dependency
+ T_command_build
+ T_command_merge
+ T_submit
+ T_hal_m3d
+ T_sync_wait
+ T_error_check
```

不同 API 只启用其中一部分成本项。

示例：`musaLaunchKernel / muLaunchKernel`

```text
T_launch =
  T_runtime_wrapper
+ T_driver_wrapper
+ T_function_lookup
+ T_stream_lookup
+ T_dispatch_command_create
+ T_dependency_resolve
+ T_queue_command
+ T_command_build
+ T_merge_decision
+ T_submit_to_queue
+ T_hal_m3d_submit
```

示例：`musaMalloc / muMemAlloc`

```text
T_alloc =
  T_runtime_wrapper
+ T_driver_wrapper
+ T_context_lookup
+ T_pool_lookup
+ T_pool_lock
+ T_pool_hit_or_grow
+ T_hal_allocate_or_map
+ T_access_modify
+ T_error_check
```

示例：`musaStreamSynchronize / muStreamSynchronize`

```text
T_stream_sync =
  T_runtime_wrapper
+ T_driver_wrapper
+ T_stream_lookup
+ T_last_command_lookup
+ T_wait
+ T_engine_error_query
+ T_user_pool_update
```

模型输入来自源码规则和埋点事件：

| 输入 | 来源 |
| --- | --- |
| API 名称、cbid、start/end、return value | Runtime/Driver Tools callback |
| correlation id | Runtime/Driver callback、Driver tracepoint |
| command id、stream id、context id、submission id | Driver tracepoint + 新增 ModelEvent |
| pool hit/miss、pool grow、trim、paging count | memory pool ModelEvent |
| dependency 数量、等待对象、等待原因 | context/stream/command ModelEvent |
| command type、build time、merge 决策、submit time | command/stream ModelEvent |
| HAL/M3D submit、doorbell、ioctl 边界 | HAL/M3D ModelEvent |
| kernel duration、memcpy duration、sync activity | MUPTI activity 或 profiler |

## 5. 方案一：最小闭环方案

### 5.1 定位

方案一用于验证链路，不追求完整覆盖 Top90 API。该方案只覆盖最关键的三类路径：

1. kernel launch：`musaLaunchKernel`、`muLaunchKernel`
2. memory allocation：`musaMalloc`、`muMemAlloc`、`muMemFree`
3. stream sync：`musaStreamSynchronize`、`muStreamSynchronize`

### 5.2 改动范围

| 仓库 | 改动 |
| --- | --- |
| `MUPTI` | 增加最小 collector；复用 activity buffer；新增最小 `ModelEvent` record 或 private hook |
| `MUSA-Runtime` | 在 `ApiTrace`、Runtime init、module/function cache 边界增加少量事件 |
| `linux-ddk/musa` | 在 launch、alloc/free、stream sync 路径增加最小事件 |

### 5.3 必须补齐的事件

| 事件 | 建议切点 | payload |
| --- | --- | --- |
| `RUNTIME_API_ENTER/EXIT` | `MUSA-Runtime/src/internal.h` `ApiTrace` | api id、is invocation、status、thread id |
| `DRIVER_API_ENTER/EXIT` | Driver wrapper 或 Tools callback | api id、status、correlation id |
| `STREAM_QUEUE_COMMAND` | `Stream::QueueCommand` | stream、command、type、queue size、async count |
| `COMMAND_BUILD_BEGIN/END` | `Command::Build`、`DispatchCommand::Build` | command、type、dependency count、status |
| `COMMAND_SUBMIT_BEGIN/END` | `Command::SubmitToQueue` | command、engine、wait/signal count、submission id |
| `STREAM_SYNC_WAIT_BEGIN/END` | `Stream::WaitFinish` | stream、last command、wait status、wait reason |
| `MEMORY_POOL_ALLOC_BEGIN/END` | `MemoryPool` 或 `mu_memory.cpp` | size、pool、hit/miss、有无 grow |

### 5.4 数据输出

最小闭环只要求输出以下文件：

| 文件 | 内容 |
| --- | --- |
| `api_events.jsonl` | Runtime/Driver API enter/exit |
| `model_events.jsonl` | 内部 ModelEvent |
| `activity_events.jsonl` | kernel、memcpy、sync activity |
| `relations.csv` | runtime api、driver api、command、submission、kernel 的关系 |
| `api_cost_breakdown.csv` | 每个 API 的分项耗时 |
| `trace_quality_report.md` | dropped records、relation recall、event loss |

### 5.5 验证方式

使用一个最小程序即可验证：

```text
musaSetDevice
musaMalloc
musaLaunchKernel
musaStreamSynchronize
musaFree
```

验收标准：

| 指标 | 标准 |
| --- | --- |
| 事件链路 | 能重建 Runtime API -> Driver API -> Stream -> Command -> Submit -> Kernel |
| relation recall | launch 路径不低于 95% |
| dropped record | 默认 buffer 配置下为 0；压力测试必须报告 dropped 数量 |
| 关闭开销 | 未开启 collector 时仅保留 ready 分支，开销应接近噪声 |
| 打开开销 | 最小事件开启时 API 热路径开销建议控制在 1%-3% |

### 5.6 优点和风险

| 项目 | 结论 |
| --- | --- |
| 优点 | 工期短，能快速验证 private hook、schema、collector、relation 重建 |
| 风险 | 不能直接完成 Top90 API 覆盖，不能支撑完整 OKR 报告 |
| 适用阶段 | M0-M1 |

## 6. 方案二：OKR 主线方案

### 6.1 定位

方案二是推荐主方案。它面向 OKR 的两个方向：

1. MUSA driver/API 白盒性能建模。
2. MUSA 行为分析和 CTS 沉淀。

该方案要覆盖 3 个主流模型推理/训练 trace 中累计耗时 Top90% 的 Driver API，并解释 Top50 kernel 的来源关系。

### 6.2 总体架构

```text
真实模型运行
  -> MUPTI collector 采集 Runtime API / Driver API / activity / ModelEvent
  -> trace normalizer 生成统一事件表
  -> relation builder 重建 API -> command -> submission -> kernel
  -> source rule engine 根据源码规则拆分 API 成本项
  -> cost fitter 校准每个成本项参数
  -> report generator 输出 API 热点、kernel 热点、瓶颈归因、优化建议
  -> behavior extractor 抽象行为用例并沉淀 CTS
```

### 6.3 仓库分工

| 仓库 | 交付 |
| --- | --- |
| `MUPTI` | `ModelEvent` private hook、collector、schema、activity 支持矩阵、buffer 压力测试、flush/dropped record 验证 |
| `MUSA-Runtime` | Runtime API source rule、Runtime init/cache 事件、Runtime 到 Driver relation、Runtime hook 开销验证 |
| `linux-ddk/musa` | Top90 Driver API source rule、memory/stream/event/graph/command/HAL/M3D 事件、源码级瓶颈归因 |

`musa_benchmarks` 可作为验证载体，但不是本方案的核心源码范围。建议用于 calibration、overhead 和 CTS 用例承载。

### 6.4 ModelEvent 设计

建议先使用 private hook，不立即做 public MUPTI ABI。

```cpp
struct MusaModelEventHeader {
    uint16_t version;
    uint16_t layer;      // runtime / driver / core / hal / m3d
    uint16_t domain;     // memory / stream / command / sync / graph / module
    uint16_t phase;      // begin / end / instant / relation
    uint32_t event_id;
    uint32_t size;
    uint64_t timestamp_ns;
    uint64_t process_id;
    uint64_t thread_id;
    uint64_t correlation_id;
};
```

事件设计原则：

| 原则 | 要求 |
| --- | --- |
| 默认关闭 | 未开启 collector 时只检查 atomic ready flag |
| 固定 schema | 事件 header 固定，payload 按 event id 解码 |
| 成对事件 | 耗时事件使用 begin/end；状态事件使用 instant |
| 关系显式 | API、command、submission、kernel 关系必须单独输出 relation event |
| 不打印日志 | 正式埋点不使用 printf，不直接写文件 |
| 可采样 | 高频事件支持 domain/event 级开关和采样 |

### 6.5 Top90 Driver API source rule

每个 Top90 API 必须有 source rule。格式固定：

| 字段 | 内容 |
| --- | --- |
| API | `muLaunchKernel` |
| 入口 | wrapper 文件和 `muapiXXX` 实现 |
| 核心路径 | 从 Driver API 到 Core/HAL/M3D 的函数链 |
| 状态变量 | stream、context、memory pool、command、submission、semaphore |
| ModelEvent | 需要哪些内部事件 |
| 成本项 | API 拆成哪些时间项 |
| 验证用例 | 用哪个 microbench 或模型片段触发 |
| 异常分支 | invalid handle、pool miss、sync wait、submit error 等 |

示例：`muStreamSynchronize`

| 字段 | 内容 |
| --- | --- |
| 入口 | `linux-ddk/musa/src/driver/mu_stream.cpp` |
| 核心路径 | `muapiStreamSynchronize -> Stream::WaitFinish -> Command::Wait / semaphore wait -> GetEngineLastError -> UpdateUserPools` |
| 事件 | `STREAM_SYNC_BEGIN/END`、`COMMAND_WAIT_BEGIN/END`、`SEMAPHORE_WAIT_BEGIN/END`、`ENGINE_ERROR_QUERY`、`POOL_UPDATE` |
| 成本项 | stream lookup、wait、engine error query、pool update |
| 主要风险 | 把真正的 device 等待误算为 Driver CPU 成本 |

示例：`muMemAlloc`

| 字段 | 内容 |
| --- | --- |
| 入口 | `linux-ddk/musa/src/driver/mu_memory.cpp` |
| 核心路径 | `muapiMemAlloc -> context/device -> memory pool -> HAL memory manager` |
| 事件 | `MEM_POOL_LOOKUP`、`MEM_POOL_ALLOC_BEGIN/END`、`MEM_POOL_GROW`、`HAL_ALLOC_BEGIN/END` |
| 成本项 | context lookup、pool lock、pool hit/miss、HAL allocation、mapping |
| 主要风险 | 不区分 pool hit 和 grow 会导致分配耗时模型失真 |

示例：`muLaunchKernel`

| 字段 | 内容 |
| --- | --- |
| 入口 | `linux-ddk/musa/src/driver/mu_module.cpp`、launch 相关 wrapper |
| 核心路径 | `muapiLaunchKernel -> Stream::CmdLaunchKernel -> Context::ResolveDependencyAndQueueCommand -> Stream::QueueCommand -> AsyncSubmit -> DispatchCommand::Build/Submit -> Command::SubmitToQueue -> HAL/M3D` |
| 事件 | `FUNCTION_LOOKUP`、`COMMAND_CREATE`、`DEPENDENCY_RESOLVE`、`QUEUE_COMMAND`、`COMMAND_BUILD`、`MERGE_DECISION`、`SUBMIT_TO_QUEUE`、`M3D_SUBMIT` |
| 成本项 | function lookup、command create、dependency、queue、build、merge、submit |
| 主要风险 | 只看 kernel duration 会漏掉 launch 侧 CPU 提交成本 |

### 6.6 关键埋点切点

| 层级 | 文件 | 事件 |
| --- | --- | --- |
| Runtime | `MUSA-Runtime/src/internal.h` | Runtime API enter/exit、Runtime hook 开销 |
| Runtime | `MUSA-Runtime/src/musa_entry.cpp` | Runtime export table、MUPTI Runtime enable/disable |
| Runtime | `MUSA-Runtime/src/musa_module.cpp` | module/function cache、kernel symbol lookup |
| Runtime | `MUSA-Runtime/src/musa_memory.cpp` | Runtime memory API 到 Driver API 映射 |
| Runtime | `MUSA-Runtime/src/musa_stream.cpp` | Runtime stream/sync API 到 Driver API 映射 |
| MUPTI | `MUPTI/src/injection/injection.cpp` | hook 注入、Tools callback subscribe、Runtime/Driver export table |
| MUPTI | `MUPTI/src/core/process_callback.cpp` | callback 到 activity、correlation id、depth 处理 |
| MUPTI | `MUPTI/src/core/buffer.cpp` | record 分配、flush、dropped record |
| Driver | `linux-ddk/musa/src/driver/callback.cpp` | subscriber、callback enable、IssueCallback |
| Driver | `linux-ddk/musa/src/driver/mupti/tracepoints.h` | kernel、memcpy、memset、sync、submission relation |
| Driver API | `linux-ddk/musa/src/driver/mu_memory.cpp` | alloc/free/memcpy API 分支 |
| Driver API | `linux-ddk/musa/src/driver/mu_stream.cpp` | stream create/destroy/sync/wait |
| Driver API | `linux-ddk/musa/src/driver/mu_module.cpp` | module load、function lookup、kernel launch |
| Core | `linux-ddk/musa/src/musa/core/stream.cpp` | queue、dependency、merge、async submit、async wait |
| Core | `linux-ddk/musa/src/musa/core/command/*.cpp` | build、submit、wait、submit to queue |
| Core | `linux-ddk/musa/src/musa/core/memoryPool.cpp` | pool hit/miss、grow、trim、paging |
| HAL | `linux-ddk/musa/src/hal/m3d/queue.cpp` | HAL queue submit |
| M3D | `linux-ddk/musa/src/hal/m3d/m3d/src/core/queue.cpp` | M3D submit internal、validate、pre/post process |
| M3D DRM | `linux-ddk/musa/src/hal/m3d/m3d/src/core/os/drm/mthreads/mtgpuQueue.cpp` | LaunchCommandStreams |
| M3D DRM | `linux-ddk/musa/src/hal/m3d/m3d/src/core/os/drm/mthreads/mtgpuDevice.cpp` | submit with doorbell、KMD 边界 |

### 6.7 数据模型

建议归一化成五张表：

| 表 | 主键 | 内容 |
| --- | --- | --- |
| `api_event` | `api_event_id` | Runtime/Driver API enter/exit、cbid、status、thread、correlation |
| `model_event` | `event_id` | 内部状态事件和成本事件 |
| `activity_event` | `activity_id` | kernel、memcpy、memset、sync、graph 等 activity |
| `relation` | `relation_id` | api、command、submission、kernel、stream、context 关系 |
| `feature` | `feature_id` | 模型特征，如 queue depth、pool hit、dependency count、merge size |

### 6.8 建模方法

方案二不使用单一拟合曲线。每个 API 建立 source rule，再用事件校准分项参数。

```text
source rule
  -> 决定 API 由哪些成本项组成

ModelEvent
  -> 量化每个成本项的出现次数、状态和耗时

activity / profiler
  -> 校准 kernel、memcpy、sync 的真实设备侧耗时

cost fitter
  -> 拟合无法直接测量的常量项和分支惩罚
```

输出示例：

| API | total_us | runtime_us | driver_us | pool_us | queue_us | build_us | submit_us | sync_wait_us | bottleneck |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `muLaunchKernel` | 18.2 | 1.1 | 2.4 | 0.0 | 3.0 | 5.2 | 6.1 | 0.0 | submit |
| `muMemAlloc` | 74.6 | 0.8 | 2.2 | 41.5 | 0.0 | 0.0 | 27.4 | 0.0 | pool grow |
| `muStreamSynchronize` | 310.0 | 0.7 | 1.9 | 0.0 | 0.0 | 0.0 | 0.0 | 304.8 | device wait |

### 6.9 OKR 验收

| OKR | 验收标准 |
| --- | --- |
| KR1 Top90 Driver API | 3 个模型 trace 中累计耗时 Top90% Driver API 均有 source rule、事件签名、成本分项 |
| KR2 Top50 kernel overlap | Top50 kernel 按累计耗时排序的 Top-K overlap 不低于 90% |
| KR3 报告 | 每个模型输出 API 热点、kernel 热点、源码级瓶颈、优化建议 |
| 行为 KR1 | dense 和 MoE 至少 5 个行为特征用例，覆盖 attention、MLP、KV cache、expert routing |
| 行为 KR2 | memory、stream/event、graph 等关键 API 覆盖 trace 出现 API 的 90% 以上 |
| 行为 KR3 | 一个 SDK 版本建立 3 个模型行为基线，并能说明版本差异 |

补充质量指标：

| 指标 | 标准 |
| --- | --- |
| relation recall | API -> command -> submission -> kernel 关系不低于 95% |
| event loss | dropped record 必须统计；正常模型 trace 不应出现不可解释 dropped |
| 成本分项覆盖 | Top90 API 的可解释成本占 API inclusive time 的 85% 以上 |
| 关闭开销 | 未开启 collector 时开销接近噪声，需要 microbench 证明 |
| 打开开销 | API-only tracing 建议不超过 1%；internal targeted tracing 建议不超过 3% |
| 版本可复现 | schema、source rule、collector、report 均记录 commit 和 SDK version |

### 6.10 风险

| 风险 | 影响 | 处理 |
| --- | --- | --- |
| MUPTI 当前单 subscriber | 与 torch profiler、msight-system 存在订阅冲突 | 短期 collector 独占；中期 collector 集成 kernel activity；长期支持多 consumer |
| 事件过多影响性能 | 热路径扰动 | domain/event 开关、TLS buffer、固定 payload、采样 |
| source rule 维护成本高 | SDK 升级后失效 | rule 绑定 commit；CI 检查 source rule 覆盖 |
| sync wait 被误算为 CPU 成本 | 瓶颈归因错误 | sync_wait 独立成本项，区分 host wait 和 device execution |
| HAL/M3D 边界难覆盖 | submit 成本不完整 | 先覆盖 HAL queue 和 M3D submit，再确认 ioctl/KMD 边界 |

## 7. 方案三：SDK 工程化方案

### 7.1 定位

方案三面向 SDK 发布和长期回归。它把方案二中的 private 能力固化为稳定工具链。

适用条件：

1. 方案二已完成 3 个模型闭环。
2. Top90 API source rule 已稳定。
3. ModelEvent schema 已经过至少一个 SDK 版本验证。

### 7.2 工程化内容

| 方向 | 内容 |
| --- | --- |
| MUPTI ABI | 将 private `ModelEvent` hook 固化为 stable internal ABI，或升级为正式 MUPTI activity kind/internal domain |
| collector | 支持配置文件、domain 开关、采样、buffer 压测、异常退出 flush |
| schema | 版本化 schema，支持新旧 SDK 兼容 |
| report | 自动生成 3 个模型的 API、kernel、行为、版本差异报告 |
| CTS | 沉淀行为用例和事件签名验收 |
| CI | 每个 SDK 版本跑 overhead、event loss、relation recall、Top-K overlap |
| dashboard | 保存跨版本趋势、热点变化、回归列表 |

### 7.3 工程化交付物

| 交付物 | 内容 |
| --- | --- |
| `mupti_model_event.h` | 稳定事件头和 event id 定义 |
| `mupti-model-collector` | 统一采集工具 |
| `source_rules/` | Top90 API source rule 文件 |
| `trace_normalizer` | 原始事件转规范表 |
| `relation_builder` | API、command、submission、kernel 关系重建 |
| `cost_model` | 白盒成本模型 |
| `report_generator` | 模型报告和版本对比报告 |
| `behavior_cts/` | 行为用例和事件签名断言 |
| `overhead_suite/` | 插桩开销看护 |

### 7.4 SDK 发布验收

| 指标 | 标准 |
| --- | --- |
| schema 稳定性 | SDK minor version 内向后兼容 |
| collector 稳定性 | 长 trace 不丢事件或明确报告 dropped |
| 事件覆盖 | Top90 Driver API 事件覆盖率保持达标 |
| Top50 kernel 对齐 | Top-K overlap 不低于 90% |
| 行为 CTS | attention、MLP、KV cache、expert routing、memory/stream/graph/sync 均有事件签名 |
| 回归报告 | 每个 SDK 版本能输出性能差异和源码归因 |

### 7.5 优点和风险

| 项目 | 结论 |
| --- | --- |
| 优点 | 可长期维护，适合 SDK 发布和跨版本性能看护 |
| 风险 | 工程成本最高，需要 ABI、工具链、CI、报告流程共同维护 |
| 适用阶段 | 方案二稳定后进入 Q4/SDK 发布阶段 |

## 8. 三套方案对比

| 维度 | 方案一：最小闭环 | 方案二：OKR 主线 | 方案三：SDK 工程化 |
| --- | --- | --- | --- |
| 主要目标 | 验证链路 | 完成 OKR | 长期发布和回归 |
| API 覆盖 | 3-6 个关键 API | Top90 Driver API | Top90 API 持续维护 |
| 事件范围 | launch、alloc、sync | memory、stream、event、graph、command、HAL/M3D | 全量稳定 schema |
| MUPTI 改动 | private hook MVP | 完整 ModelEvent collector | 稳定 ABI/activity domain |
| Runtime 改动 | ApiTrace、init/cache 少量事件 | Runtime source rule 和 relation | 版本化 Runtime 事件 |
| Driver 改动 | 最小路径埋点 | Top90 API 相关模块埋点 | 事件签名和 CI 看护 |
| 工期 | 2-4 周 | 8-12 周 | 一个 SDK 发布周期 |
| 风险 | 覆盖不足 | 跨仓库联调复杂 | 工程维护成本高 |
| 验收 | 链路可重建 | OKR 指标达标 | SDK 可重复发布 |

## 9. 推荐实施节奏

### M0：基线采集

交付：

1. 选定 3 个模型和场景。
2. 使用现有 MUPTI/profiler 采集 API、kernel、memcpy、sync。
3. 生成 `top_api.csv`、`top_kernel.csv`、`trace_gap_report.md`。
4. 明确 Top90 Driver API 列表。

### M1：最小 ModelEvent 闭环

交付：

1. `EmitModelEvent` private hook。
2. Runtime/Driver/Core 最小事件。
3. collector 输出 `api_event`、`model_event`、`activity_event`、`relation`。
4. 对 `musaMalloc`、`musaLaunchKernel`、`musaStreamSynchronize` 输出成本分项。

### M2：扩展到 Top90 API

交付：

1. Top90 Driver API source rule。
2. memory、stream、event、graph、command、HAL/M3D 事件覆盖。
3. relation recall 和 event loss 报告。
4. 插桩开销报告。

### M3：白盒模型 v1

交付：

1. source rule engine。
2. cost fitter。
3. API cost breakdown。
4. Top50 kernel overlap 报告。
5. 3 个模型的初版分析报告。

### M4：行为 CTS 和版本报告

交付：

1. attention、MLP、KV cache、expert routing 行为用例。
2. memory、stream/event、graph、sync 行为用例。
3. 事件签名断言。
4. SDK 版本行为基线。
5. 版本差异报告。

## 10. 有疑问与可选方案

| 问题 | 影响 | 可选方案 |
| --- | --- | --- |
| 3 个主流模型名单未固定 | 决定 Top90 API 列表 | A. Llama dense 推理；B. DeepSeek/MoE 推理；C. 训练场景如 transformer block 或 DDP |
| Top90 API 耗时口径未固定 | 影响 KR1 统计 | A. inclusive time；B. host self time；C. sync_wait 单独拆分。建议主口径用 inclusive time，报告同时输出 self 和 wait |
| ModelEvent 进入 public ABI 还是 private hook | 影响开发周期和兼容性 | A. private hook 先落地；B. internal domain；C. public activity kind。建议先 private hook |
| MUPTI 单 subscriber 与 profiler 冲突 | 影响同进程采集 | A. collector 独占；B. collector 内集成 kernel activity；C. 支持多 consumer。建议先 collector 独占 |
| HAL/M3D 到 KMD 边界可观测性不足 | submit 成本不完整 | A. 先记录 HAL/M3D 边界；B. 增加 mtgpu submit 前后事件；C. 与 KMD trace 对齐 |
| CTS 最终目录未定 | 影响落库流程 | A. 先放 `musa_benchmarks`；B. 放 `musa_samples`；C. 独立 CTS 仓库。建议先放 `musa_benchmarks` |
| 插桩开销阈值未定 | 影响上线判断 | A. API-only <=1%；B. targeted internal <=3%；C. full trace 只用于诊断不进默认发布 |

## 11. 最终交付清单

| 类别 | 交付物 |
| --- | --- |
| 源码规则 | Top90 Driver API source rule |
| 事件通路 | Runtime/Driver `ModelEvent` private hook |
| 采集工具 | MUPTI collector |
| 数据格式 | `api_event`、`model_event`、`activity_event`、`relation`、`feature` schema |
| 建模工具 | source rule engine、relation builder、cost fitter |
| 报告 | 3 个模型 API 热点、kernel 热点、瓶颈归因、优化建议 |
| 验收 | Top90 coverage、Top50 overlap、relation recall、event loss、overhead report |
| 行为沉淀 | attention、MLP、KV cache、expert routing、memory、stream/event、graph、sync CTS |

## 12. Definition of Done

1. Top90 Driver API 均有源码路径、source rule、事件签名和成本分项。
2. MUPTI collector 能采集 Runtime API、Driver API、activity 和内部 ModelEvent。
3. 对关键路径能重建 `Runtime API -> Driver API -> Core Command -> HAL/M3D submit -> kernel/activity`。
4. Top50 kernel 按累计耗时排序的 Top-K overlap 不低于 90%。
5. 报告能说明 API 热点、kernel 热点、等待来源、内存池行为、command merge/submit 行为。
6. 关闭 collector 时不改变正常执行路径；开启 collector 时开销有量化报告。
7. 3 个模型形成 SDK 版本行为基线。
8. 行为 CTS 能复现关键软件行为，并校验事件签名。
