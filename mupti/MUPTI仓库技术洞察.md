# MUPTI 仓库技术洞察

## 1. 源码范围

| 目录或文件 | 作用 |
|---|---|
| `MUPTI/include/` | 对外头文件，包含 Callback、Activity、Profiler、PC Sampling、Event、Metric 等接口声明 |
| `MUPTI/src/api/` | 对外 API 实现，主要包含 callback、activity、profiler、PC sampling、version、result、internal relation API |
| `MUPTI/src/core/` | 内部状态、activity buffer、事件关联、device monitor、Tools callback 转换逻辑 |
| `MUPTI/src/injection/` | 与 `libmusa.so`、`libmusart.so` 建立连接，注册 MUPTI hooks，导入 Driver accessor |
| `MUPTI/src/utils/` | 动态库加载、锁封装、日志、时间、backoff 工具 |
| `MUPTI/daemon/` | 注入式采集 daemon 示例，自动开启 Runtime/Driver API activity |
| `MUPTI/unittests/` | callback、activity、profiler 使用示例 |
| `linux-ddk/musa/src/musa_shared_include/export_table.h` | MUPTI 与 Driver/Runtime 共享的 export table、hook、accessor、Tools callback 数据结构 |

MUPTI 仓库是 profiling 库本体。它不实现 Driver API 或 Runtime API 的业务逻辑，而是通过 Driver/Runtime 暴露的 export table、hook 和 accessor 读取执行状态，并把状态转换成 MUPTI callback 或 activity record。

## 2. 核心结论

MUPTI 的实现由四层组成：

```text
Public API
  -> muptiSubscribe / muptiActivityEnable / muptiProfiler* / muptiPCSampling*

Injection
  -> dlopen libmusa.so / libmusart.so
  -> 获取 DriverExportTable / RuntimeExportTable / Tools::ExportTable
  -> 注册 MUPTI hooks
  -> 导入 Driver accessor

Event Collection
  -> Tools callback 采集 Runtime/Driver API enter-exit
  -> Driver MUPTI hooks 采集 kernel、memcpy、memset、graph、sync、resource
  -> MT-Perf 采集 HW/KM 事件，用于补齐 kernel/memop 时间戳

Record Output
  -> ActivityBufferManager 分配 activity record
  -> 用户注册 buffer request / complete callback
  -> muptiActivityFlushAll 输出完整 record
```

当前 MUPTI 的强项是把 API、command、kernel、memcpy、memset、graph、sync 事件串到同一套 activity 体系中。它适合做 profiling 数据采集和关联关系重建。

当前 MUPTI 的主要缺口是内部软件成本事件不足。它能记录“API 调用了什么、kernel 何时执行、memcpy 何时完成”，但不能直接解释 memory pool hit/miss、command build、dependency resolve、submit、ioctl wait 等 Driver 内部成本。

## 3. 构建和导出符号

`MUPTI/CMakeLists.txt` 构建 `mupti` 动态库，默认也构建 `mupti_static` 静态库。`MUPTI/src/CMakeLists.txt` 有几个关键设计：

| 设计 | 说明 |
|---|---|
| `-fvisibility=hidden` | 默认隐藏符号，减少 ABI 暴露面 |
| `exported_symbols.ver` | 只导出明确列出的 MUPTI API |
| `mtperf_static` | 链接 MT-Perf，用于硬件和内核事件采集 |
| `extract_enum.py` | 从 result、driver cbid、runtime cbid 头文件生成 `.inc` 文件 |
| `ENABLE_COMPILE_STATIC_LIB` | 同时生成静态库，Release 下会做符号处理 |

当前 `exported_symbols.ver` 导出的接口集中在以下几类：

| 类别 | 代表 API |
|---|---|
| Activity | `muptiActivityEnable`、`muptiActivityRegisterCallbacks`、`muptiActivityFlushAll` |
| Callback | `muptiSubscribe`、`muptiEnableCallback`、`muptiEnableDomain` |
| Profiler | `muptiProfilerBeginSession`、`muptiProfilerSetConfig`、`muptiProfilerEnableProfiling` |
| PC Sampling | `muptiPCSamplingEnable`、`muptiPCSamplingStart`、`muptiPCSamplingStop` |
| 关联查询 | `muptiGetKickFromCorrelationId`、`muptiGetSubmissionIdsFromCorrelationId` |
| 基础能力 | `muptiGetTimestamp`、`muptiGetVersion`、`muptiGetResultString`、`muptiFinalize` |

需要注意：`include/` 中声明了 Event、Metric、部分 Activity attribute 和更多 PC Sampling API，但当前导出符号和 `src/api/` 实现并未覆盖全部声明。这说明头文件保留了兼容接口面，当前仓库实现重点不是完整 Event/Metric API。

## 4. 初始化流程

几乎所有对外 API 都先调用 `MUpti::init()`。初始化只执行一次，由 `g_hook_initialized` 原子变量保护。

```cpp
MUptiResult init() {
    bool expected = false;

    // 只有第一个线程真正执行注入，其余线程直接返回成功。
    if (g_hook_initialized.compare_exchange_strong(expected, true)) {
        bool success = MUinteraction::inject_musa();
        return success ? MUPTI_SUCCESS : MUPTI_ERROR_NOT_INITIALIZED;
    }

    return MUPTI_SUCCESS;
}
```

`inject_musa()` 的执行顺序：

```text
inject_musa
  -> get_export_table_from_musa_driver
       -> dlopen("libmusa.so.1")，失败后尝试 dlopen("libmusa.so")
       -> dlsym("muGetExportTable")
       -> 获取 DriverExportTable(Client::MUpti)
       -> 尝试获取 Tools::ExportTable(Client::Tools)
       -> 如果 Tools callback 可用，订阅 ProcessDriverCallback
  -> DriverExportTable.MUpti->Enable(init_mupti_driver_hooks)
  -> import_musa_accessors
  -> get_export_table_from_musa_runtime
       -> dlopen("libmusart.so")
       -> dlsym("musaGetExportTable")
       -> 获取 RuntimeExportTable(Client::MUpti)
  -> RuntimeExportTable.MUpti->Enable(init_mupti_runtime_hooks)
```

这里有两个关键点：

1. Driver 是必需依赖。没有 `libmusa.so` 或 Driver export table，MUPTI 初始化失败。
2. Runtime 是可选依赖。`libmusart.so` 不存在时，MUPTI 仍可采集 Driver 侧事件。

## 5. Export Table、Hook 和 Accessor

MUPTI 与 Driver/Runtime 的连接由 `export_table.h` 定义。

### 5.1 Hook

Hook 是 Driver/Runtime 调用 MUPTI 的入口。MUPTI 在初始化时把自己的函数指针填入 Driver/Runtime 的 hook 表。

Driver hook 覆盖范围较大：

| 类型 | Hook 示例 | 用途 |
|---|---|---|
| API | `EnterDriverApi`、`ExitDriverApi`、`EnterRuntimeApi`、`ExitRuntimeApi` | 兼容旧路径的 API callback/activity |
| Kernel | `RegisterKernelV2`、`MarkKernelBeginEnd`、`AssignKernelToKick` | 建立 kernel record 和 kick 关系 |
| Memory | `EnterMemcpy`、`EnterMemset`、`MarkMemcpyBeginEnd`、`MarkMemsetBeginEnd` | 建立 memcpy/memset record |
| Graph | `RegisterGraphTrace`、`RegisterGraphKernel`、`MarkGraphNodeBeginEndV2` | 建立 graph trace 和 graph node record |
| Sync | `RegisterStreamSynchronize`、`StartStreamSynchronize`、`StopStreamSynchronize` | 记录 host sync 等待 |
| Resource | `CreateContext`、`CreateStream` | 记录 context/stream 生命周期 callback |
| Relation | `AssignSubmissionToCorrelation` | 建立 submission 与 correlation 的关系 |

Runtime hook 只包含：

```text
EnterRuntimeApi
ExitRuntimeApi
SetMemset3DCounter
```

这说明 Runtime 侧只负责 API 外壳事件和 `memset3D` 计数补充，执行细节主要由 Driver 侧提供。

### 5.2 Accessor

Accessor 是 MUPTI 读取 Driver 内部对象字段的入口。例如：

| Accessor | 读取内容 |
|---|---|
| `CommandBase.GetCorrelationId` | command 对应的 correlation id |
| `CommandBase.GetBeginEndTimestamp` | command 设备侧时间戳 |
| `DispatchCommand.GetGridSize` | kernel grid 信息 |
| `MemcpyCommand.GetSize` | memcpy 字节数 |
| `Context.GetId` | context id |
| `Stream.GetId` | stream id |
| `Function.GetName` | kernel 名称 |
| `Graph.NodeGetType` | graph node 类型 |

MUPTI 通过 accessor 读取 Driver 内部状态，而不是复制 Driver 对象定义。这降低了编译耦合，但 accessor 表就是 ABI 合约，新增字段需要保持兼容。

### 5.3 兼容策略

`AccessorHint = 0xDEADBEEF` 表示函数指针未填充。MUPTI 初始化时使用 `fill_hooks` 和 `fill_accessors`：

```cpp
void fill_hooks(uintptr_t* pDst, uintptr_t* pSrc) {
    // 只填充仍为 AccessorHint 或 0 的槽位。
    // 这样可以兼容旧版本 Driver/Runtime 的较短 hook 表。
    while (pDst != nullptr &&
           (*pDst == AccessorHint || *pDst == 0) &&
           pSrc != nullptr && *pSrc != 0) {
        *pDst = *pSrc;
        pDst++;
        pSrc++;
    }
}
```

该设计允许 MUPTI 与不同版本的 Driver/Runtime 共存，但要求 hook/accessor 只能追加，不能随意重排。

## 6. Callback 机制

MUPTI 对外 callback API 位于 `MUPTI/src/api/callback.cpp`。

用户使用方式：

```text
muptiSubscribe
  -> 创建 MUpti_Subscriber_st
  -> 保存用户 callback 和 userdata

muptiEnableCallback / muptiEnableDomain / muptiEnableAllDomains
  -> 修改 subscriber 内部 enable 数组
  -> 如果 Driver Tools callback 可用，同步调用 ToolsCallback.Enable*

Driver/Runtime API 执行
  -> Tools callback 或 legacy hook 产生 enter/exit 事件
  -> MUPTI 转换成 MUpti_CallbackData
  -> 调用用户 callback
```

`MUpti_Subscriber_st` 为每个 domain 保存独立 enable 数组：

```text
driver_api_callback_enabled
runtime_api_callback_enabled
resource_callback_enabled
sync_callback_enabled
mttx_callback_enabled
```

当前 MUPTI 只支持一个 subscriber。`muptiSubscribe` 检查 `g_subscriber`，已有 subscriber 时返回 `MUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED`。

### 6.1 Tools callback 路径

新版本 Driver 支持 `Tools::ExportTable`。MUPTI 初始化时订阅 Driver Tools callback：

```text
ToolsCallback.Subscribe(&g_driver_subscribe_handle, ProcessDriverCallback, nullptr)
```

Driver 或 Runtime wrapper 产生的 `MUtoolsTraceApiMusa` / `MUtoolsTraceRuntimeApiMusa` 会进入 `ProcessDriverCallback`。

`ProcessDriverCallback` 做三件事：

1. 如果用户启用了对应 CBID，构造 `MUpti_CallbackData` 并回调用户函数。
2. 如果启用了 API activity，记录 API start/end。
3. 用 thread-local depth 过滤嵌套 API activity，避免内部递归调用污染外层 API 耗时。

运行时 API activity 记录逻辑：

```text
Runtime API ENTER
  -> 如果 runtime activity 已启用，且当前线程 runtime depth 为 0
       保存 start timestamp
  -> runtime depth++

Runtime API EXIT
  -> runtime depth--
  -> 如果回到 depth 0
       创建 MUPTI_ACTIVITY_KIND_RUNTIME record
       写入 start、end、cbid、returnValue、processId、threadId、correlationId
```

Driver API activity 额外要求当前线程没有 Runtime API 包裹，避免把 Runtime 内部调用的 Driver API 当成独立外层 API。

### 6.2 Legacy hook 路径

如果 Driver 不支持 Tools callback，MUPTI 通过 `EnterDriverApi` / `ExitDriverApi` 和 `EnterRuntimeApi` / `ExitRuntimeApi` hook 采集 API。

该路径保留兼容价值，但参数解析覆盖有限。源码中只对部分 API 构造参数结构，例如：

| Domain | 已显式解析的 API 示例 |
|---|---|
| Driver | `muMemcpyHtoD_v2`、`muMemcpyDtoH_v2`、`muLaunchKernel`、`muLaunchKernelEx`、`muProfilerStart`、`muProfilerStop` |
| Runtime | `musaMalloc_v3020`、`musaFree_v3020`、`musaProfilerStart_v4000`、`musaProfilerStop_v4000` |

因此，新方案应优先依赖 Tools callback。legacy hook 路径更适合作为旧版本兼容。

## 7. Activity 机制

Activity API 位于 `MUPTI/src/api/activity.cpp`，内部缓冲区由 `ActivityBufferManager` 管理。

### 7.1 用户侧调用流程

典型使用流程：

```text
muptiActivityEnable(kind)
muptiActivityRegisterCallbacks(bufferRequested, bufferCompleted)
运行 MUSA 程序
muptiActivityFlushAll(MUPTI_ACTIVITY_FLAG_FLUSH_FORCED)
```

用户必须注册两个 callback：

| callback | 作用 |
|---|---|
| `bufferRequested` | MUPTI 需要新 buffer 时，由用户分配内存 |
| `bufferCompleted` | MUPTI flush buffer 时，把有效 record 交给用户解析 |

`muptiActivityGetNextRecord` 按 record kind 推进指针。record 大小由 `activity_record_kinds.inc` 映射得到。

### 7.2 Buffer 管理

`ActivityBuffer` 使用一个原子 `state_` 同时保存标志位和正在写入的 record 数量：

```text
bit 63: FLUSHED
bit 62: REFUSING
bit 61: CHECKING
bit 60..0: 正在写入的 record 数量
```

分配 record 的关键路径：

```text
acquire_record
  -> 检查 buffer 未 REFUSING / FLUSHED
  -> state_ 写入计数 +1
  -> byte_used_ CAS 分配空间
  -> 返回 ActivityRecordPtr

ActivityRecordPtr 析构
  -> release_record
  -> state_ 写入计数 -1
```

这个设计避免每条 activity record 都走全局锁。只有当前 buffer 不存在或已满时，才调用用户的 `bufferRequested` 获取新 buffer。

### 7.3 Flush 完整性

MUPTI 不会随意 flush 未完成 record。`try_mark_completed()` 会逐条检查 record 是否还在内部 map 中：

| Record 类型 | 完整性判断 |
|---|---|
| Kernel | `g_kernel_map` 中不存在对应 correlation/graph node |
| Memcpy/Memset/MemoryTransfer | `g_memop_map` 中不存在对应 correlation/graph node |
| GraphTrace | `g_graph_map` 中不存在对应 correlation |
| Synchronization | `g_sync_map` 中不存在对应 correlation |

这保证用户拿到的 activity record 尽量包含完整 start/end 时间戳。

强制 flush 时，MUPTI 最多等待约 1 秒：

```text
wait_interval = 10000 us
wait_iteration = 100
```

如果还有 kernel/memop 未完成，MUPTI 会尝试 flush MT-Perf stream，并在超时后输出错误日志。超时后仍可能返回缺失时间戳的 record。

## 8. Kernel、Memory、Graph、Sync 记录流程

### 8.1 Kernel

普通 kernel activity 流程：

```text
Driver 创建 DispatchCommand
  -> RegisterKernelV2(command)
       -> 分配 MUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL record
       -> 读取 function name、grid、block、shared memory、local memory
       -> 写入 deviceId、contextId、streamId、correlationId
       -> graph 场景下记录 graphId、graphNodeId
  -> command queue / submit
  -> command 完成或 HW event 返回
       -> MarkCommandBeginEnd / MarkKernelBeginEnd
       -> 写入 start、end、completed
       -> 从 g_kernel_map 移除，允许 buffer flush
```

如果 kernel 来自 graph，MUPTI 使用复合 key：

```text
uniqueId = (correlationId << 32) + graphNodeId
```

这样可以区分同一次 graph launch 中不同 graph node 的 activity。

### 8.2 Memcpy、Memset、MemoryTransfer

memory activity 流程：

```text
EnterMemcpy / EnterMemset / EnterMemoryTransfer
  -> 分配对应 activity record
  -> 读取 copy kind、memory kind、bytes、device/context/stream
  -> 放入 g_memop_map

MarkMemcpyBeginEnd / MarkMemsetBeginEnd / MarkCommandBeginEnd
  -> 从 command 或 HW event 获取 begin/end
  -> 转换为 OS timestamp
  -> 写入 activity record
  -> 从 g_memop_map 移除
```

`musaMemset3D` 可能拆成多个 memset command。Runtime 侧通过 `SetMemset3DCounter` 告诉 MUPTI 当前 correlation 下有几个 memset 子命令。MUPTI 只有在计数归零后才移除 memop record。

### 8.3 Graph

Graph 有两类记录：

| 类型 | 说明 |
|---|---|
| `MUPTI_ACTIVITY_KIND_GRAPH_TRACE` | 表示一次 graph 执行整体区间 |
| graph node activity | graph 中 kernel、memcpy、memset、memory atomic、memory transfer 节点的 activity |

如果启用了 `GRAPH_TRACE`，MUPTI 记录 graph 整体时间。如果没有启用 `GRAPH_TRACE`，则可以记录 graph 内部 node activity。

Graph trace V2 同时接收 device begin/end 和 host begin/end：

```text
graphTrace.start = min(hostBegin, deviceBeginToOs)
graphTrace.end   = max(hostEnd, deviceEndToOs)
```

这可以覆盖 graph launch 的 host 侧开销和 device 侧执行时间。

### 8.4 Synchronization

同步 activity 包含：

| 类型 | Hook |
|---|---|
| event synchronize | `RegisterEventSynchronize`、`StartEventSynchronize`、`StopEventSynchronize` |
| stream wait event | `RegisterStreamWaitEvent`、`StartStreamWaitEvent`、`StopStreamWaitEvent` |
| stream synchronize | `RegisterStreamSynchronize`、`StartStreamSynchronize`、`StopStreamSynchronize` |
| context synchronize | `RegisterContextSynchronize`、`StartContextSynchronize`、`StopContextSynchronize` |

这类 activity 记录 host 等待区间。它能回答“是否发生同步等待”和“等待多久”，但不能直接给出 wait reason。要做白盒性能建模，还需要补充等待目标状态、queue depth、inflight command、event 状态等内部事件。

## 9. MT-Perf Device Monitor

`DeviceMonitorManager` 使用 MT-Perf 获取硬件和内核事件。

初始化流程：

```text
DeviceMonitorManager::init
  -> perf::initLibrary
  -> perf::GpuDevice::enumDevices
  -> 如果是 DDK2.0，检查 Driver 版本和 engine sync 能力
  -> 枚举可见设备
  -> 为每个设备创建 DeviceMonitor
```

`DeviceMonitor` 内部有两个 stream：

| Stream | 来源 | 用途 |
|---|---|---|
| `perf::KmStream` | KM event | DMA enqueue/start/end 等内核侧事件 |
| `perf::HwStream` | HW event | CDM/TDM/CE kick/finish 等硬件侧事件 |

事件处理分为 legacy 和 DDK2.0 两套路径：

| 路径 | 事件示例 | 行为 |
|---|---|---|
| Legacy | `HW_EVENT_TYPE_CDMKICK`、`HW_EVENT_TYPE_CDMFINISHED` | 通过 `external_job_ref` 标记 kernel/memop begin/end |
| DDK2.0 | `PERF_EVENT_HOST_DMA_START_USER`、`HW_EVENT_CDM_KICK` | 通过 submission id 或 cb id 标记 begin/end |

`g_support_engine_sync` 会影响时间戳来源：

1. 如果 Driver command 自身提供 begin/end timestamp，MUPTI 可以直接用 `MarkCommandBeginEnd` 写 activity。
2. 如果不能直接从 command 获取完整时间，MUPTI 通过 MT-Perf 事件和 relation map 补齐时间。

## 10. Profiler 和 PC Sampling

Profiler API 是 Driver profiler controller 的薄封装。

| API | 行为 |
|---|---|
| `muptiProfilerBeginSession` | 只支持 `MUPTI_UserReplay` |
| `muptiProfilerSetConfig` | 优先调用新 `Profiler.SetConfig(ctx, config)`，否则回退旧 PFM 接口 |
| `muptiProfilerBeginPass` / `EndPass` | 控制 pass 生命周期，旧接口下维护 `passIndex` |
| `muptiProfilerInitialize` / `DeInitialize` | 设置 `g_profiler_initilized`，调用 Driver profiler 初始化/反初始化 |

PC Sampling API 也是 Driver controller 的薄封装：

```text
muptiPCSamplingEnable
muptiPCSamplingDisable
muptiPCSamplingSetConfigurationAttribute
muptiPCSamplingStart
muptiPCSamplingStop
```

当前实现没有覆盖头文件中声明的全部 PC Sampling 查询接口，例如 `muptiPCSamplingGetData`、stall reason 查询、SASS/source correlation 等。这部分能力需要结合 Driver controller 和导出符号继续补齐。

## 11. Daemon 和单元测试

`MUPTI/daemon/daemon.cpp` 提供注入式采集示例：

```text
动态库加载
  -> 全局对象构造
  -> initTrace()
       -> enable DRIVER activity
       -> enable RUNTIME activity
       -> register activity buffer callbacks
  -> 进程退出时析构
       -> flush
       -> disable activity
```

单元测试覆盖三类典型用法：

| 测试 | 说明 |
|---|---|
| `callbackTimestampTest.cpp` | 订阅 Driver API callback，记录 memcpy、launch、stream sync 的 enter/exit 时间 |
| `activityRuntimeTest.cpp` | 开启 Runtime/Driver/memcpy/memset/kernel activity，注册 buffer callback 并解析 record |
| `activityDriverTest.cpp` | 使用 Driver API 执行 vector add，验证 activity 输出 |
| `userReplayTest.cpp` | 使用 profiler user replay 流程 |

这些测试更像 API 使用样例。它们没有覆盖内部 race、buffer 压力、graph、sync、engine sync、MT-Perf 丢事件等复杂场景。

## 12. 当前实现的关键设计

| 设计 | 价值 |
|---|---|
| Hook + accessor 分离 | hook 用于事件进入，accessor 用于读取 Driver 内部对象字段 |
| Tools callback 优先 | 新版本 API 参数和返回码由 wrapper 统一发送，覆盖面更完整 |
| ActivityBuffer 无全局锁写入 | 常规 record 分配主要依赖原子计数和 CAS |
| 完整性检查后 flush | 减少用户拿到半成品 activity record 的概率 |
| correlation/kick/submission 多级关系 | 支持 API、command、submission、kernel/memop 关联 |
| graph node 复合 key | 支持一次 graph launch 中多个 node 的 record 区分 |
| MT-Perf fallback | command 时间戳不足时可用硬件事件补齐 begin/end |
| export table 追加兼容 | 支持不同 Driver/Runtime 版本共存 |

## 13. 当前风险和缺口

| 问题 | 影响 |
|---|---|
| 头文件声明大于实际实现 | 用户可能误以为 Event/Metric/部分 PC Sampling API 已可用 |
| 只支持单 subscriber | 多个 profiling 工具不能同时订阅同一进程 |
| legacy hook 参数解析覆盖有限 | 旧 Driver 环境下 callback 覆盖不完整 |
| `muptiSupportedDomains` 只返回 Driver/Runtime/Resource | Synchronize domain 虽在枚举中存在，但支持声明不完整 |
| Activity kind enable 不校验实际支持范围 | 某些 kind 可设置为 enabled，但没有对应 record 生产路径 |
| Activity kind 边界检查允许 `kind == MUPTI_ACTIVITY_KIND_COUNT` | `g_activity_kind_enabled` 数组大小为 `MUPTI_ACTIVITY_KIND_COUNT`，该值作为索引存在越界风险 |
| Flush 依赖内部 map 完整性 | relation 丢失或事件未回收时可能延迟 flush 或产生不完整时间戳 |
| MT-Perf 依赖外部库和设备事件 | 无设备、权限不足或事件丢失时 kernel/memop 时间戳质量下降 |
| 内部软件成本事件不足 | 不能直接解释 Driver 白盒成本，如 pool lookup、command build、submit、ioctl wait |
| 全局状态较多 | `muptiFinalize`、多线程使用、重复初始化需要严格测试 |

## 14. 对白盒性能建模的价值

MUPTI 当前可以直接提供三类数据：

| 数据 | 来源 | 可建模内容 |
|---|---|---|
| API enter/exit | Tools callback / legacy hook | Runtime/Driver API 外层耗时、Top API、参数分布 |
| Activity record | Driver hooks + ActivityBuffer | kernel、memcpy、memset、graph、sync 的执行区间 |
| Relation map | kick/submission/correlation API | API 到 command、submission、kernel 的关联 |

这些数据可以支撑：

```text
API 层
  -> 哪些 Runtime/Driver API 是热点

执行层
  -> kernel/memcpy/memset/graph/sync 的时间线

关联层
  -> 一个 API 触发了哪些 command、submission、kernel 或 memory operation
```

但要形成真正的软件性能模型，还需要新增内部事件：

| 模块 | 建议补充事件 |
|---|---|
| Runtime 初始化 | export table init、Tools table detect、injection load |
| Module/Function | module cache hit/miss、function lookup、fatbin register |
| Memory | memory pool lookup、pool hit/miss、chunk allocate、BO alloc/map ioctl |
| Stream | dependency resolve、queue depth、async capacity wait |
| Command | command build、merge check、merge flush、submit begin/end |
| Graph | instantiate、update、node rebuild decision |
| Sync | wait target、wait reason、inflight count |
| IOCTL | ioctl enter/exit、opcode、status、blocking reason |

推荐基于现有 hook 表新增统一 `EmitModelEvent`，先作为 private hook 验证，不直接扩大 public callback ABI。

## 15. 建议后续工作

1. 补一份 API 支持矩阵：列出 `include/` 中声明的 API、`src/api/` 中已实现的 API、`exported_symbols.ver` 中已导出的 API。
2. 增加 callback domain 一致性检查：明确 `SYNCHRONIZE` 是否作为 public callback domain 支持。
3. 增加 activity kind 支持矩阵：区分“可 enable”和“实际会产生 record”。
4. 增加 buffer 压力测试：覆盖 buffer 满、record 未完成、forced flush、dropped record。
5. 增加 graph 和 sync 测试：验证 graph trace、graph node activity、stream/event/context sync record。
6. 为白盒建模新增 internal model event：优先覆盖 memory、command、stream、sync、ioctl 五个高价值模块。
7. 建立 overhead 评估：分别测量未启用、启用 API callback、启用 activity、启用新增 internal event 的热路径开销。
