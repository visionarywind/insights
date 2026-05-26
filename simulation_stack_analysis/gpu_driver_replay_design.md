# GPU Driver 仿真如何复现问题

## 复现范围

GPU Driver 仿真的复现对象是 Runtime 到设备执行之间的驱动层行为。它不直接复现 kernel 内部指令执行，而是复现驱动层看到的资源、同步、提交和 launch 上下文。

可以复现的问题：

| 问题类型 | 是否适合 Driver Replay 复现 | 原因 |
|---|---:|---|
| buffer 提前释放后又被 kernel 使用 | 是 | trace 中可以记录 malloc、free、kernel 参数 |
| event 没有 record 就 wait | 是 | trace 中可以记录 event record / wait 顺序 |
| fence 死等 | 是 | trace 中可以记录 submit、signal、wait 关系 |
| command buffer packet 错误 | 是 | trace 中可以保存 command buffer 或 packet |
| kernel launch 参数错误 | 是 | trace 中可以保存 grid、block、args、buffer 信息 |
| 小 kernel 过多导致 driver 开销高 | 是 | trace 中可以保存 submit timeline |
| kernel 内部计算错误 | 部分支持 | Driver Replay 只能导出 launch 上下文，内部执行交给 Aic |
| 设备访存拥塞 | 部分支持 | Driver Replay 只能导出任务流，设备内部行为交给 DCSim |

复现目标：

```text
复现 driver 层看到的执行过程，
并复现资源、同步、提交、command、launch 上下文中的问题。
```

## 总体流程

```text
真实运行
  -> trace_collector 采集事件
  -> trace.json
  -> driver_replay 离线重放
  -> checker 检查非法状态
  -> report 输出复现结果
```

核心思想：

1. 真实运行时记录关键事件。
2. 离线按事件顺序重建 driver 状态。
3. checker 在重放过程中检查非法状态。
4. 如果同样的非法状态再次出现，就完成 driver 层问题复现。

## Trace 采集

trace_collector 需要在 Runtime、UMD、ioctl 边界记录事件。

第一版建议至少记录：

- context_create
- context_destroy
- stream_create
- stream_destroy
- malloc
- free
- memcpy
- kernel_launch
- event_create
- event_record
- event_wait
- stream_sync
- ioctl_submit
- fence_create
- fence_wait
- fence_signal

每条事件建议包含统一字段：

```json
{
  "seq": 1024,
  "ts_ns": 39128400122,
  "pid": 1201,
  "tid": 1208,
  "context": "ctx0",
  "stream": "stream3",
  "op": "kernel_launch",
  "kernel": "attention_kernel",
  "grid": [128, 1, 1],
  "block": [256, 1, 1],
  "args": [
    {"name": "q", "buffer": "buf17", "offset": 0, "size": 8388608},
    {"name": "k", "buffer": "buf18", "offset": 0, "size": 8388608}
  ],
  "command_buffer": "cmd_1024.bin",
  "fence": "fence77"
}
```

字段要求：

- `seq`：事件顺序编号，用于稳定重放。
- `ts_ns`：真实时间戳，用于性能时间线分析。
- `pid` / `tid`：进程和线程信息，用于多线程问题定位。
- `context`：上下文 ID，用于区分不同 GPU context。
- `stream`：stream ID，用于重建 stream queue。
- `op`：事件类型。
- `kernel`：kernel 名称。
- `grid` / `block`：kernel launch 配置。
- `args`：kernel 参数，建议使用逻辑 buffer ID，不只记录裸地址。
- `command_buffer`：command buffer dump 或 hash。
- `fence`：本次提交对应的 fence。

## Driver 状态重建

driver_replay 不直接执行 GPU kernel，而是重建 driver 状态机。

需要维护的核心状态：

```text
contexts:
  ctx0:
    state: alive

streams:
  stream3:
    state: alive
    queue: []

buffers:
  buf17:
    state: allocated
    size: 8388608
    mapped: true
    in_use_by: [fence77]

events:
  event5:
    state: created
    recorded: false

fences:
  fence77:
    submitted: true
    signaled: false
```

重放逻辑：

```python
for event in trace:
    if event.op == "context_create":
        create_context(event.context)

    if event.op == "stream_create":
        create_stream(event.context, event.stream)

    if event.op == "malloc":
        create_buffer(event.buffer, event.size)

    if event.op == "free":
        check_buffer_can_be_freed(event.buffer)
        free_buffer(event.buffer)

    if event.op == "kernel_launch":
        check_stream_alive(event.stream)
        check_kernel_args(event.args)
        append_to_stream_queue(event.stream, event)
        mark_buffers_in_use(event.args, event.fence)

    if event.op == "event_wait":
        check_event_recorded(event.event)

    if event.op == "fence_wait":
        check_fence_exists(event.fence)
```

## Checker 设计

checker 负责在重放过程中检查非法状态。第一版建议实现以下 checker。

### Buffer 生命周期检查

检查内容：

- 未分配 buffer 被使用。
- 已释放 buffer 被再次使用。
- buffer 仍被未完成 fence 引用时释放。
- kernel 参数中的 offset / size 超过 buffer 范围。
- buffer 重复释放。

示例错误：

```text
[ERROR] use-after-free
  seq: 30
  kernel: attention_kernel
  buffer: buf1
  allocated at seq: 10
  freed at seq: 20
  reused at seq: 30
```

### Event / Fence 检查

检查内容：

- event 未创建就使用。
- event 未 record 就 wait。
- event 被错误重复使用。
- fence 未创建就 wait。
- fence wait 之前没有 submit。
- fence 长时间不 signal。

示例错误：

```text
[ERROR] wait event before record
  seq: 55
  stream: stream2
  event: event9
  event state: created, not recorded
```

### Stream / Queue 检查

检查内容：

- stream 未创建就提交。
- stream 已销毁后继续提交。
- stream sync 等待不存在的任务。
- queue 中依赖关系不完整。
- 多 stream event 依赖顺序不正确。

示例错误：

```text
[ERROR] submit to destroyed stream
  seq: 88
  stream: stream4
  destroyed at seq: 73
```

### Command Buffer 检查

检查内容：

- command buffer 缺少必要 packet。
- packet 顺序错误。
- kernel launch packet 参数数量不匹配。
- DMA packet 地址无效。
- sync packet 引用不存在的 fence/event。

示例错误：

```text
[ERROR] invalid command packet order
  seq: 120
  command_buffer: cmd_120.bin
  detail: sync packet appears before kernel launch packet dependency is defined
```

## 复现示例：buffer 仍在使用时被释放

真实程序：

```text
musaMalloc(buf)
musaLaunchKernel(stream0, kernel, buf)
musaFree(buf)
musaStreamSynchronize(stream0)
```

如果 `musaFree` 没有正确等待 `stream0` 上的 kernel 完成，就可能出现 buffer 仍在被设备使用时已经释放的问题。

采集到的 trace：

```json
[
  {"seq": 1, "op": "malloc", "buffer": "buf0", "size": 4096},
  {
    "seq": 2,
    "op": "kernel_launch",
    "stream": "s0",
    "kernel": "k0",
    "args": [{"buffer": "buf0", "offset": 0, "size": 4096}],
    "fence": "f0"
  },
  {"seq": 3, "op": "free", "buffer": "buf0"},
  {"seq": 4, "op": "fence_signal", "fence": "f0"}
]
```

重放过程：

```text
seq=1:
  buf0 = allocated

seq=2:
  k0 使用 buf0
  buf0 标记为 in_use_by=f0
  f0 状态为 submitted, not signaled

seq=3:
  free buf0
  checker 发现 buf0 仍被 f0 引用，且 f0 尚未 signal
```

输出：

```text
[ERROR] free buffer while still in use
  buffer: buf0
  free seq: 3
  pending fence: f0
  fence signal seq: 4
```

这个问题不需要真实 GPU，也可以离线复现。

## 性能问题复现

性能问题通过 timeline 复现。

示例事件：

```text
seq=100 kernel_launch rms_norm
seq=101 ioctl_submit
seq=102 fence_wait
seq=103 fence_signal
```

如果大量小 kernel 都有类似模式：

```text
kernel 执行时间: 3 us
driver submit 开销: 8 us
fence wait: 5 us
```

报告可以输出：

```text
small kernel count: 12000
average kernel time: 3 us
average submit overhead: 8 us
driver overhead > kernel time
```

结论：

```text
端到端性能损失主要来自小 kernel 数量过多和 driver submit / sync 开销，
不是单个 kernel 内部计算过慢。
```

## Kernel 崩溃问题复现

如果问题发生在 kernel 内部，Driver Replay 不直接模拟指令，但可以复现 launch 上下文。

需要导出：

- kernel name
- kernel binary
- grid / block
- args
- buffer address
- shape
- layout
- workspace

连接方式：

```text
trace.json
  -> aic_launch_context_exporter
  -> Aic 仿真
```

分析目标：

```text
判断问题来自 driver 传参错误，还是 kernel 内部实现错误。
```

## 最小可交付版本

第一版建议交付：

```text
trace_schema.json
trace_collector
driver_replay
buffer_lifetime_checker
event_fence_checker
stream_queue_timeline
report.md
```

第一版覆盖的问题：

- buffer 生命周期错误
- event / fence 依赖错误
- stream 同步错误
- kernel launch 参数记录
- driver 提交时间线分析

验收标准：

```text
给定一份真实运行 trace，
driver_replay 能按 seq 顺序重放事件，
重建 context、stream、buffer、event、fence 状态，
并在同一位置报告资源错误、同步错误或提交异常。
```

## 结论

复现问题依赖 trace。真实运行时记录 driver 关键事件，离线按顺序重放，并重建资源状态。重放过程中如果再次出现相同非法状态，就完成 driver 层问题复现。

Driver Replay 复现的是驱动层行为；kernel 内部执行交给 Aic，设备侧调度和访存交给 DCSim，端到端性能趋势交给 SPM-E2E。
