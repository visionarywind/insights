# MUSA Driver/API 性能建模落地方案

说明：本文是早期 trace replay 落地方案。当前 OKR 主路径以 `insights/okr/okr-2026-landing-plan.md` 为准，采用源码埋点、MUPTI ModelEvent 采集、白盒建模和 `musa_benchmarks` 校准。本文后续只作为 trace schema、linker、validator 和 replay 校验工具的参考，不作为主实施方案。

## 1. 目标定义

OKR 中的 MUSA driver/API 性能建模，第一阶段不做完整 GPU 内部仿真，也不做复杂数学拟合。第一阶段目标是建立基于真实 trace 的 replay 性能模型。

模型定义：

```text
输入：真实模型运行产生的 Runtime API trace、driver command trace、profiler kernel trace
输出：可重放的执行时间线、API/kernel/sync 热点、瓶颈归因、predicted vs profiling 对齐结果
```

核心链路：

```text
模型运行
  -> MUSA Runtime API
  -> Driver command
  -> Stream / Event / Graph / Memory
  -> Kernel launch / memcpy / sync
  -> Profiling timeline
```

第一阶段需要先回答：

- 哪些 Runtime API 占主要 host 时间或 sync wait 时间。
- 每个 `musaLaunchKernel` 对应哪个 driver command 和哪个 profiler kernel。
- 每个 command 在 stream 中经历了哪些状态。
- `musaStreamSynchronize`、`musaEventSynchronize`、`musaStreamWaitEvent` 在等什么。
- kernel 热点是否与 profiling 数据对齐。
- 性能瓶颈来自 kernel、memcpy、sync wait、queue delay、driver submit 还是 idle gap。

## 2. 仓库分工

两个仓库分别负责不同层级的 trace 采集。

| 仓库 | 责任 | 采集内容 |
|---|---|---|
| `MUSA-Runtime` | Runtime API 入口层 | API start/end、参数摘要、stream、kernel symbol、memcpy size、返回值 |
| `linux-ddk/musa` | Driver / UMD 行为层 | command 创建、入队、build、submit、wait、complete、semaphore、stream dependency |
| `insights/simulation/okr` 或外部工具目录 | 离线建模工具 | trace 解析、link、replay、validator、report |

建模链路：

```text
musaLaunchKernel
  -> driver muLaunchKernel
  -> Context::ResolveDependencyAndQueueCommand
  -> Stream::QueueCommand
  -> Stream::AsyncSubmit
  -> Command::Build
  -> Command::SubmitToQueue
  -> profiler kernel activity
  -> Stream::AsyncWait / command completed
```

## 3. 第一版覆盖对象

第一版只覆盖 trace 中累计耗时 Top90% 的 driver/runtime API，不要求覆盖全部 API。

优先覆盖：

1. Kernel launch
2. Memcpy / Memset
3. Stream synchronize / query
4. Event record / wait / synchronize
5. Malloc / Free / MallocAsync / FreeAsync
6. Graph launch / capture / instantiate

对应源码切点：

| 类别 | 关键源码位置 |
|---|---|
| kernel launch | `MUSA-Runtime/src/musa_module.cpp::musaapiLaunchKernel` |
| memory API | `MUSA-Runtime/src/musa_memory.cpp::musaapiMalloc`、`musaapiFree`、`musaapiMemcpyAsync` |
| stream API | `MUSA-Runtime/src/musa_stream.cpp::musaapiStreamSynchronize`、`musaapiStreamWaitEvent` |
| event API | `MUSA-Runtime/src/musa_event.cpp::musaapiEventRecord`、`musaapiEventSynchronize` |
| dependency 解析 | `linux-ddk/musa/src/musa/core/context.cpp::Context::ResolveDependencyAndQueueCommand` |
| command 入队 | `linux-ddk/musa/src/musa/core/stream.cpp::Stream::QueueCommand` |
| command build/submit | `linux-ddk/musa/src/musa/core/stream.cpp::Stream::AsyncSubmit` |
| queue submit | `linux-ddk/musa/src/musa/core/command/command.cpp::Command::SubmitToQueue` |
| dispatch submit | `linux-ddk/musa/src/musa/core/command/dispatchCommand.cpp::DispatchCommand::Submit` |
| command complete/error | `linux-ddk/musa/src/musa/core/stream.cpp::Stream::AsyncWait` |

### 3.1 源码插桩切点清单

这一节是第一版必须落地的插桩点。插桩只记录事件，不改变调度、同步和错误处理逻辑。

#### 3.1.1 Runtime API 层插桩

Runtime API 层用于记录用户看到的 MUSA API 行为，包括 API host 耗时、参数摘要和返回状态。

| 文件 | 函数 | 插桩位置 | 输出事件 | 必须记录字段 |
|---|---|---|---|---|
| `MUSA-Runtime/src/musa_module.cpp:28` | `musaapiLaunchKernel` | 函数入口、调用 `muLaunchKernel` 前后、函数返回前 | `runtime_api_begin`、`runtime_driver_call`、`runtime_api_end` | `api`、`stream`、`func`、`gridDim`、`blockDim`、`sharedMem`、`args_hash`、`status`、`ts_start/end` |
| `MUSA-Runtime/src/musa_module.cpp:74` | `musaapiLaunchKernelExC` | 函数入口、调用 `muLaunchKernelEx` 前后、函数返回前 | `runtime_api_begin/end` | `api`、`stream`、`func`、`gridDim`、`blockDim`、`attrs`、`status` |
| `MUSA-Runtime/src/musa_memory.cpp:97` | `musaapiMalloc` | 函数入口、`muMemAlloc_v2` 返回后、函数返回前 | `runtime_alloc` | `size`、`devPtr`、`status`、`ctx`、`ts_start/end` |
| `MUSA-Runtime/src/musa_memory.cpp:195` | `musaapiFree` | 函数入口、`muMemFree_v2` 返回后、函数返回前 | `runtime_free` | `devPtr`、`status`、`ctx`、`ts_start/end` |
| `MUSA-Runtime/src/musa_memory.cpp:374` | `musaapiMemcpyAsync` | 函数入口、具体 `muMemcpy*Async` 调用前后、函数返回前 | `runtime_memcpy_async` | `dst`、`src`、`count`、`kind`、`stream`、`status`、`ts_start/end` |
| `MUSA-Runtime/src/musa_stream.cpp:201` | `musaapiStreamSynchronize` | 函数入口、`muStreamSynchronize` 返回后、函数返回前 | `runtime_stream_sync` | `stream`、`status`、`wait_time_us`、`ts_start/end` |
| `MUSA-Runtime/src/musa_stream.cpp:210` | `musaapiStreamWaitEvent` | 函数入口、`muStreamWaitEvent` 返回后、函数返回前 | `runtime_stream_wait_event` | `stream`、`event`、`flags`、`status`、`ts_start/end` |
| `MUSA-Runtime/src/musa_event.cpp:53` | `musaapiEventRecord` | 函数入口、`muEventRecord` 返回后、函数返回前 | `runtime_event_record` | `event`、`stream`、`status`、`ts_start/end` |
| `MUSA-Runtime/src/musa_event.cpp:77` | `musaapiEventSynchronize` | 函数入口、`muEventSynchronize` 返回后、函数返回前 | `runtime_event_sync` | `event`、`status`、`wait_time_us`、`ts_start/end` |

`musaapiLaunchKernel` 的插桩重点：

```text
函数入口：
  记录 API begin、线程、stream、grid/block、func。

调用 muLaunchKernel 前：
  记录已经通过 runtime 参数检查，准备进入 driver。

muLaunchKernel 返回后：
  记录 driver 返回状态。

函数返回前：
  记录 API end 和 host duration。
```

第一版不需要展开 `void **args` 的全部内容，建议先记录：

```text
args_count
args_hash
前 N 个指针参数的地址摘要
```

后续需要定位参数错误时，再扩展参数解析。

#### 3.1.2 Driver command 层插桩

Driver command 层用于记录 Runtime API 进入 driver 后如何被转换为 command、如何入队、如何 build、如何 submit、如何完成。

| 文件 | 函数 | 插桩位置 | 输出事件 | 必须记录字段 |
|---|---|---|---|---|
| `linux-ddk/musa/src/musa/core/context.cpp:1845` | `Context::ResolveDependencyAndQueueCommand` | 依赖解析完成后、调用 `pStream->QueueCommand` 前后 | `driver_dependency_resolved`、`driver_queue_result` | `ctx`、`stream`、`command_id`、`command_type`、`blocking`、`dependency_count`、`dependency_ids`、`status` |
| `linux-ddk/musa/src/musa/core/stream.cpp:1015` | `Stream::QueueCommand` | `command->SetStatus(queued)` 后、`m_CommandList.push_back` 后 | `command_queued` | `stream`、`command_id`、`command_type`、`prev_command_id`、`async_count`、`command_list_size`、`queued_ts` |
| `linux-ddk/musa/src/musa/core/stream.cpp:1118` | `Stream::AsyncSubmit` | `buildCommand` 中 `command->Build` 前后、`SetStatus(built)` 后 | `command_build_begin`、`command_built` | `stream`、`command_id`、`command_type`、`engine`、`merge_level`、`signal_value`、`status` |
| `linux-ddk/musa/src/musa/core/stream.cpp:1118` | `Stream::AsyncSubmit` | `submitMergingList` 中调用 `Submit()` 前后 | `command_submit_begin`、`command_submitted` | `stream`、`primary_command_id`、`merged_command_ids`、`engine`、`submission_id`、`status` |
| `linux-ddk/musa/src/musa/core/command/command.cpp:648` | `Command::SubmitToQueue` | 组装 `QueueSubmitInfo` 后、`pQueue->Submit` 前后 | `queue_submit` | `command_id`、`command_type`、`engine`、`submission_id`、`cmdBufferCount`、`waitSemaphoreCount`、`signalSemaphoreCount`、`wait_values`、`signal_values`、`status` |
| `linux-ddk/musa/src/musa/core/stream.cpp:1290` | `Stream::AsyncWait` | 等待 semaphore 前后、`SetStatus(completed/error)` 后 | `command_wait_begin`、`command_completed`、`command_error` | `stream`、`command_id`、`command_type`、`engine`、`signal_value`、`wait_status`、`last_error`、`completed_ts` |
| `linux-ddk/musa/src/musa/core/command/command.cpp:285` | `Command::WaitFinish` | 函数入口、等待返回后 | `command_wait_finish` | `command_id`、`command_type`、`status`、`wait_time_us` |

#### 3.1.3 具体 command 类型插桩

具体 command 类型用于补充 kernel、memcpy、memset、event record 等业务语义。

| 文件 | 函数 | 插桩位置 | 输出事件 | 必须记录字段 |
|---|---|---|---|---|
| `linux-ddk/musa/src/musa/core/command/dispatchCommand.cpp:235` | `DispatchCommand::Submit` | `SubmitToQueue` 前 | `dispatch_submit_detail` | `command_id`、`kernel_name`、`engine=cdm`、`cmdBufferCount`、`grid/block`、`submission_id` |
| `linux-ddk/musa/src/musa/core/command/AsyncMemcpyCommand.cpp:102` | `AsyncMemcpyCommand::Submit` | `SubmitToQueue` 前 | `memcpy_submit_detail` | `command_id`、`copy_direction`、`bytes`、`engine`、`src_type`、`dst_type`、`submission_id` |
| `linux-ddk/musa/src/musa/core/command/memsetCommand.cpp:201` | `MemsetCommand::Submit` | `SubmitToQueue` 前 | `memset_submit_detail` | `command_id`、`bytes`、`engine`、`region_count`、`submission_id` |
| `linux-ddk/musa/src/musa/core/command/recordCommand.cpp:106` | `RecordCommand::Submit` | `SubmitToQueue` 前 | `event_record_submit_detail` | `command_id`、`event`、`stream`、`engine`、`submission_id` |

#### 3.1.4 Memory 生命周期插桩

Memory 建模需要把 Runtime 层的 `musaMalloc/musaFree` 与 driver 内部 memory tracker 对齐。

第一版至少记录：

```text
alloc:
  ptr
  size
  context
  api seq
  status

free:
  ptr
  context
  api seq
  status

kernel/memcpy 使用:
  ptr
  offset
  size
  command_id
  stream
  submission_id
```

第一版 checker 要能发现：

```text
free 后继续被 kernel/memcpy 使用
buffer 仍在 submitted/inflight command 中就 free
memcpy 访问范围超过已知 allocation size
重复 free
```

#### 3.1.5 插桩开关和输出要求

插桩必须通过环境变量关闭，默认不影响性能。

建议开关：

```text
MUSA_PERF_TRACE=0/1
MUSA_PERF_TRACE_FILE=/path/to/trace.bin
MUSA_PERF_TRACE_SCOPE=runtime|driver|all
MUSA_PERF_TRACE_LEVEL=1
```

输出要求：

```text
热路径只写二进制 POD 事件
离线 converter 将 trace.bin 转成 JSONL / CSV / report
所有事件必须带 seq、timestamp、pid、tid、layer、op/status
同一个 command 必须使用稳定 command_id
同一个 stream/event/memory 尽量使用稳定逻辑 id，而不只依赖裸指针
```

第一版最小事件集：

```text
runtime_api_begin
runtime_api_end
runtime_memcpy_async
runtime_stream_sync
runtime_event_record
runtime_stream_wait_event
driver_dependency_resolved
command_queued
command_built
queue_submit
command_completed
command_error
```

## 4. 插桩低开销设计

插桩不能按普通日志系统设计。普通日志通常会格式化字符串、加锁、写文件，这些行为会影响 Runtime API 和 driver command 热路径。

本方案中的插桩必须按 tracing 系统设计：

```text
热路径只写二进制事件到内存
后台线程异步落盘
离线工具再解析成 JSONL / CSV / report
```

### 4.1 设计原则

核心原则：

```text
默认关闭时接近零成本；
开启后只记录必要字段；
热路径不做慢操作；
异常时丢 trace，不阻塞业务。
```

默认配置：

```text
MUSA_PERF_TRACE=0
```

关闭状态下，插桩只允许存在一个轻量判断：

```cpp
if (MUSA_UNLIKELY(g_trace_enabled)) {
    TraceWrite(...);
}
```

关闭状态下不允许：

- 分配内存
- 格式化字符串
- 写文件
- 拿锁
- 调用复杂函数
- 读取环境变量
- 解析 kernel name
- 扫描 kernel args

目标：

```text
trace off 性能影响 <= 0.1%，或者落在测试噪声内。
```

### 4.2 热路径禁止事项

以下操作不得出现在 Runtime API 和 driver command 热路径中：

```text
printf / tprintf
fprintf
std::ofstream
JSON 序列化
std::string 拼接
malloc / new
mutex lock
getenv
符号解析 / demangle
遍历完整参数列表
拷贝大块 kernel args
同步 flush 文件
```

特别是以下函数中不允许引入阻塞行为：

- `Stream::AsyncSubmit`
- `Command::SubmitToQueue`
- `Stream::AsyncWait`
- `musaapiLaunchKernel`
- `musaapiMemcpyAsync`
- `musaapiStreamSynchronize`

### 4.3 全局配置

进程初始化时读取环境变量，并写入全局配置。后续热路径只读 atomic，不再调用 `getenv`。

建议配置结构：

```cpp
struct TraceConfig {
    std::atomic<bool> enabled;
    uint32_t scope;
    uint32_t level;
    uint32_t api_mask;
    uint32_t sample_rate;
};
```

建议环境变量：

```text
MUSA_PERF_TRACE=0/1
MUSA_PERF_TRACE_FILE=/path/to/trace.bin
MUSA_PERF_TRACE_SCOPE=runtime|driver|all
MUSA_PERF_TRACE_LEVEL=1
MUSA_PERF_TRACE_SAMPLE_RATE=1
```

### 4.4 宏封装

插桩入口统一用宏封装，保证关闭时成本可控。

示例：

```cpp
#define MUSA_TRACE_EVENT(code, payload)                         \
    do {                                                        \
        if (MUSA_UNLIKELY(g_trace_config.enabled.load(          \
                std::memory_order_relaxed))) {                  \
            MusaTraceWrite(code, payload);                      \
        }                                                       \
    } while (0)
```

关闭时只剩一个可预测分支。

### 4.5 事件结构

热路径写入 POD 结构体，不写字符串。

示例：

```cpp
struct TraceEvent {
    uint64_t seq;
    uint64_t ts_ns;
    uint32_t pid;
    uint32_t tid;
    uint16_t layer;
    uint16_t op;
    uint64_t id0;
    uint64_t id1;
    uint64_t value0;
    uint64_t value1;
    uint64_t value2;
    uint64_t value3;
};
```

Runtime API 示例：

```text
op       = runtime_launch_kernel
id0      = stream
id1      = func pointer
value0   = gridDim packed
value1   = blockDim packed
value2   = sharedMem
value3   = status
```

kernel 名称、函数名、demangle 信息放到离线解析阶段处理。

### 4.6 Buffer 设计

第一版建议使用 thread-local ring buffer：

```text
每个线程一个固定大小 ring buffer
业务线程只写自己的 buffer
后台线程定期 drain
buffer 满了直接丢事件
```

buffer 满时不能阻塞业务：

```cpp
if (buffer_full) {
    drop_count++;
    return;
}
```

报告必须输出：

```text
trace_drop_count
trace_buffer_size
trace_flush_error_count
```

这样可以评估 trace 完整性，但不会影响模型执行。

### 4.7 Trace Level

不要一开始全量记录。

建议分层：

| Level | 内容 | 用途 |
|---|---|---|
| 0 | 关闭 | 默认模式 |
| 1 | Runtime API summary | API 热点、host duration、基础 replay |
| 2 | Driver command lifecycle | command queue/build/submit/complete |
| 3 | 详细参数、semaphore、memory range | 深度诊断 |

默认建议：

```text
MUSA_PERF_TRACE_LEVEL=1
```

只有定位复杂问题时才开启 level 2 或 level 3。

### 4.8 Runtime 层低开销策略

Runtime 层只记录 API 边界，不做深度解析。

以 `musaapiLaunchKernel` 为例：

```text
函数入口：
  runtime_api_begin

调用 muLaunchKernel 前：
  runtime_driver_call_begin

muLaunchKernel 返回后：
  runtime_driver_call_end

函数返回前：
  runtime_api_end
```

字段只记录摘要：

```text
api id
stream
func pointer
gridDim
blockDim
sharedMem
status
timestamp
```

`void **args` 第一版不展开，只记录：

```text
args pointer
args hash
```

### 4.9 Driver 层低开销策略

Driver 层只记录 command 状态变化。

状态：

```text
created
queued
built
submitted
completed
error
```

关键插桩点：

```text
Context::ResolveDependencyAndQueueCommand
  记录 dependency_count

Stream::QueueCommand
  记录 command_queued

Stream::AsyncSubmit
  记录 command_build_begin / command_built / command_submitted

Command::SubmitToQueue
  记录 submission_id、wait/signal semaphore count

Stream::AsyncWait
  记录 command_completed / command_error
```

第一版不记录完整 command buffer 内容，只记录：

```text
command_id
command_type
stream
engine
submission_id
wait_count
signal_count
status
timestamp
```

command buffer dump 或 packet 解析只在 debug 模式中开启。

### 4.10 性能保护机制

必须具备以下保护：

| 机制 | 作用 |
|---|---|
| 环境变量关闭 | 默认不影响业务 |
| 编译宏开关 | release 可彻底裁剪 |
| thread-local buffer | 避免全局锁 |
| binary event | 避免 JSON/string 开销 |
| 异步 flush | 避免业务线程写文件 |
| buffer full drop | 避免阻塞业务 |
| sample rate | 控制高频事件数量 |
| trace level | 控制字段详细程度 |
| drop count | 评估 trace 完整性 |

### 4.11 插桩性能验收

插桩本身必须独立验收。

至少跑三组：

```text
baseline：
  无插桩代码，或编译关闭。

trace off：
  有插桩代码，但 MUSA_PERF_TRACE=0。

trace on：
  开启 level=1 / level=2。
```

验收建议：

```text
trace off:
  API benchmark 性能变化 <= 0.1%，或在测试噪声内。

trace on level=1:
  普通模型端到端开销 <= 1%-3%。

trace on level=2:
  可接受诊断开销，但不能死锁、不能阻塞、不能改变执行结果。

trace buffer:
  drop_count 可见；
  flush 失败不影响业务。
```

### 4.12 插桩落地方式

插桩代码分为三层，热路径只调用第一层。

| 层级 | 位置 | 职责 | 性能要求 |
|---|---|---|---|
| trace macro | Runtime API / driver command 热路径 | 判断开关，写入固定字段 | 关闭时只有一次分支判断 |
| trace runtime | Runtime 或 driver 公共目录 | 管理 thread-local buffer、seq、timestamp、drop count | 不加锁，不分配内存 |
| trace tools | `insights/simulation/okr/tools/` 或外部工具 | `trace.bin` 转 JSONL/CSV，生成报告 | 离线执行，不影响业务 |

建议代码形态：

```cpp
MUSA_TRACE_RUNTIME_API_BEGIN(api_id, stream, id0, id1);
MUSA_TRACE_RUNTIME_API_END(api_id, stream, status);
MUSA_TRACE_COMMAND_STATE(command_id, command_type, stream, state);
```

宏内部只接收整数、指针地址、枚举和时间戳。kernel 名称、API 名称、command 类型字符串在离线阶段用字典解析。

### 4.13 负面影响控制清单

插桩合入前必须检查以下项目。

| 项目 | 要求 |
|---|---|
| 默认行为 | `MUSA_PERF_TRACE=0`，默认不采集 |
| 编译保护 | 支持编译宏关闭整套 trace |
| 热路径操作 | 不允许日志格式化、文件写入、锁、动态分配 |
| buffer 策略 | 固定大小 thread-local ring buffer，满了丢弃事件 |
| 错误处理 | trace 失败不能改变 Runtime/Driver 返回值 |
| 采样控制 | 支持 `MUSA_PERF_TRACE_SAMPLE_RATE` 降低高频事件数量 |
| 输出完整性 | 报告 `drop_count`、`flush_error_count`、`trace_level` |
| 性能验收 | 必须比较 baseline、trace off、trace on 三组结果 |

插桩不能改变以下行为：

```text
API 返回码
command 入队顺序
stream/event 同步语义
driver submit 时序
错误处理路径
```

## 5. 性能模型形式

第一版采用 replay 模型，不采用黑盒拟合。

总体分解：

```text
T_total =
  T_api_host
+ T_queue_delay
+ T_command_build
+ T_submit
+ T_device_kernel_or_memcpy
+ T_sync_wait
+ T_idle_gap
```

各项来源：

| 模型项 | 来源 |
|---|---|
| `T_api_host` | Runtime API trace 的 start/end |
| `T_queue_delay` | command queued 到 build/submit 的间隔 |
| `T_command_build` | command build 阶段耗时 |
| `T_submit` | `Command::SubmitToQueue` 耗时 |
| `T_device_kernel_or_memcpy` | profiler kernel/memcpy activity |
| `T_sync_wait` | stream/event/context sync API 的等待时间 |
| `T_idle_gap` | stream 上一个 device op 结束到下一个 device op 开始的空洞 |

第一版的 kernel cost 可以直接来自 profiler：

```text
kernel_cost = median(duration by kernel name / grid / block / shape bucket)
```

后续版本再将 kernel cost 参数化。

## 6. Trace Schema

统一建模输入使用 JSONL，每行一个事件。Runtime 和 driver 热路径可以先输出二进制 `trace.bin`，再由离线 converter 转成 JSONL。

### 6.1 Runtime API 事件

```json
{
  "seq": 1,
  "ts_start_us": 100.0,
  "ts_end_us": 112.0,
  "layer": "runtime",
  "api": "musaLaunchKernel",
  "stream": "s0",
  "kernel_symbol": "rms_norm",
  "grid": [128, 1, 1],
  "block": [256, 1, 1],
  "corr_id": 1001,
  "status": "success"
}
```

### 6.2 Driver command 事件

```json
{
  "seq": 20,
  "ts_us": 118.0,
  "layer": "driver",
  "op": "command_status",
  "command_id": "cmd42",
  "type": "Dispatch",
  "stream": "s0",
  "engine": "cdm",
  "from": "built",
  "to": "submitted",
  "submission_id": 77
}
```

### 6.3 Profiler kernel 事件

```json
{
  "seq": 40,
  "layer": "profiler",
  "type": "kernel",
  "kernel": "rms_norm",
  "corr_id": 1001,
  "stream": "s0",
  "ts_start_us": 130.0,
  "ts_end_us": 138.0
}
```

### 6.4 Memcpy 事件

```json
{
  "seq": 41,
  "layer": "profiler",
  "type": "memcpy",
  "kind": "H2D",
  "stream": "s1",
  "bytes": 1048576,
  "ts_start_us": 150.0,
  "ts_end_us": 180.0
}
```

## 7. Trace 对齐

性能模型能否成立，关键在于把三类 trace 对齐：

```text
Runtime API -> Driver Command -> Profiler Kernel
```

映射字段优先级：

1. correlation id
2. submission id
3. stream id
4. timestamp window
5. kernel symbol
6. command type

对齐后的链路示例：

```text
musaLaunchKernel corr_id=1001
  host time: 12 us
  -> cmd42 queued
  -> cmd42 built
  -> cmd42 submitted
  -> rms_norm kernel
     device time: 8 us
  -> cmd42 completed
```

没有这一步，性能建模只能停留在 profiler 表格分析。

## 8. Replay 逻辑

离线 replay 工具按 stream 重放执行时间线。

需要维护的状态：

```text
stream_tail[stream]：该 stream 上最后一个 device op 的结束时间
command_table：command 生命周期
kernel_table：profiler kernel activity
event_table：event record/wait 状态
memory_table：buffer 生命周期
```

核心伪代码：

```python
for event in runtime_trace:
    if event.api == "musaLaunchKernel":
        kernel = find_kernel_by_corr_id(event.corr_id)
        start = max(stream_tail[event.stream], event.ts_end_us)
        end = start + kernel_cost[kernel.signature]
        stream_tail[event.stream] = end
        record_kernel_prediction(event, kernel, start, end)

    if event.api == "musaMemcpyAsync":
        memcpy = find_memcpy_activity(event)
        start = max(stream_tail[event.stream], event.ts_end_us)
        end = start + memcpy_cost(event.bytes, event.kind)
        stream_tail[event.stream] = end
        record_memcpy_prediction(event, start, end)

    if event.api == "musaStreamSynchronize":
        wait = max(0, stream_tail[event.stream] - event.ts_start_us)
        predicted_api_time = wait + sync_host_overhead
        record_sync_prediction(event, wait, predicted_api_time)

    if event.api == "musaEventRecord":
        event_table[event.event].record_stream = event.stream
        event_table[event.event].ready_time = stream_tail[event.stream]

    if event.api == "musaStreamWaitEvent":
        ready_time = event_table[event.event].ready_time
        stream_tail[event.stream] = max(stream_tail[event.stream], ready_time)
```

## 9. 工具链规划

建议在 `insights/simulation/okr/tools/` 或独立工程中实现离线工具。

模块拆分：

```text
trace_loader.py
  读取 runtime / driver / profiler trace

trace_converter.py
  将 runtime / driver 二进制 trace 转成 JSONL / CSV

trace_linker.py
  关联 Runtime API -> Driver Command -> Kernel

cost_model.py
  统计 API cost、kernel cost、memcpy bandwidth、sync overhead

replay_model.py
  按 stream/event/command 状态重放 timeline

validator.py
  比较 predicted vs profiling

report_generator.py
  生成 API 热点、kernel 热点、sync 热点、误差归因和优化建议
```

第一版可以只支持 CSV/JSONL 文件输入，不需要接入复杂数据库。

## 10. 落地里程碑

### M1：离线 trace 解析

目标：

- 能读取 profiler 导出的 API/kernel CSV。
- 能统计 Top API、Top kernel。
- 能用 correlation id 对齐 `musaLaunchKernel -> kernel`。

输出：

```text
top_api.csv
top_kernel.csv
launch_kernel_mapping.csv
```

### M2：Runtime API trace hook

目标：

- 在 `MUSA-Runtime` 加轻量 trace hook。
- 使用环境变量控制开关，例如 `MUSA_PERF_TRACE=1`。
- 输出 API start/end、stream、kernel、memcpy size、ptr、status。

输出：

```text
runtime_trace.bin
runtime_trace.jsonl
```

### M3：Driver command trace hook

目标：

- 在 `linux-ddk/musa` 记录 command 生命周期。
- 覆盖 `queued`、`built`、`submitted`、`completed`、`error`。
- 记录 command type、engine、stream、submission id、semaphore。

输出：

```text
driver_command_trace.bin
driver_command_trace.jsonl
```

### M4：Replay model

目标：

- 按 stream 重放 kernel、memcpy、sync、event wait。
- 输出 predicted timeline。
- 计算 API host time、kernel time、sync wait、queue delay、idle gap。

输出：

```text
predicted_timeline.csv
breakdown.csv
sync_wait_attribution.csv
```

### M5：对齐验证

目标：

- 对 3 个模型分别验证：
  - Top50 kernel set overlap >= 90%
  - Top50 cumulative time error
  - API Top90% 覆盖率
  - sync wait 归因是否能解释主要等待

输出：

```text
validation_report.md
kernel_overlap.csv
api_coverage.csv
```

### M6：分析报告

每个模型输出一份报告：

```text
API 热点
kernel 热点
memory 热点
stream/event sync 热点
graph 行为
predicted vs profiling 对齐
误差归因
优化建议
```

## 11. OKR 验收方式

### KR1：覆盖 Top90% driver API

执行方法：

1. 统计 3 个模型 trace 中 Runtime/Driver API 的 host time 和 sync wait time。
2. 按累计耗时排序。
3. 选出累计 Top90% API。
4. 对这些 API 建 replay 规则。

验收输出：

```text
api_coverage.csv
api_model_rule_mapping.csv
```

### KR2：Top50 kernel 对齐

执行方法：

1. 从 profiling 数据统计真实 Top50 kernel。
2. 从 replay 模型输出预测 Top50 kernel。
3. 计算集合 overlap：

```text
overlap = |pred_top50 ∩ real_top50| / 50
```

建议同时输出：

```text
Top50 cumulative time error
Spearman rank correlation
```

验收标准：

```text
Top50 kernel set overlap >= 90%
```

### KR3：发布 3 个模型性能建模报告

每份报告至少包含：

- 模型和运行配置
- trace 覆盖率
- API 热点
- kernel 热点
- memory 热点
- stream/event sync 热点
- graph 行为
- predicted vs profiling 对齐结果
- 误差归因
- 优化建议

## 12. 第一版不做的事情

第一版不做：

- 不模拟 kernel 内部指令执行。
- 不模拟完整 GPU 调度器。
- 不接入复杂 DCSim/Aic。
- 不覆盖全部 Runtime API。
- 不做黑盒拟合。
- 不做复杂 GUI。

第一版只做：

```text
真实 trace -> 执行链路重建 -> stream replay -> 热点归因 -> profiling 对齐
```

## 13. 第一版交付清单

建议交付：

```text
trace_schema.json
runtime_api_trace_collector
driver_command_trace_collector
trace_converter
profiler_trace_importer
trace_linker
replay_model
cost_model
validator
report_generator
3 个模型性能建模报告
```

## 14. Definition of Done

第一版完成标准：

1. 能采集 3 个模型的 Runtime API trace。
2. 能采集 driver command 生命周期 trace。
3. 能导入 profiler kernel trace。
4. 能关联 `musaLaunchKernel -> command -> kernel`。
5. 能 replay 出 stream timeline。
6. 能解释 `musaStreamSynchronize`、`musaEventSynchronize`、`musaStreamWaitEvent` 在等哪个 kernel 或 command。
7. Top50 kernel overlap 达到 90%。
8. 输出 3 个模型的性能建模报告。
9. trace off 对 API benchmark 的影响在测试噪声内。
10. trace on level=1 的普通模型端到端开销不超过 1%-3%。

## 15. 总结

MUSA driver/API 性能建模第一阶段应按 trace replay 落地。

关键不是先写复杂公式，而是把真实 perf 里的执行链路还原清楚：

```text
谁调用
谁入队
谁 build
谁 submit
谁执行
谁等待
等待多久
为什么等待
```

还原清楚后，模型才能用于热点解释、profiling 对齐、版本对比和优化建议。
