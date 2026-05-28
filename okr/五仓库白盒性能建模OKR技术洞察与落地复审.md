# 五仓库白盒性能建模 OKR 技术洞察与落地复审

## 1. 结论

本次以远程环境 `/home/shanfeng/workspace` 为准，复审以下仓库和 OKR 文档：

| 目标 | 实际远程路径 | 角色 |
| --- | --- | --- |
| musa | `/home/shanfeng/workspace/linux-ddk/musa` | Driver/Core/HAL/M3D 主实现，白盒事件的主要来源 |
| MUSA-Runtime | `/home/shanfeng/workspace/MUSA-Runtime` | Runtime API wrapper、Runtime->Driver 映射、Runtime 侧 source rule |
| MUPTI | `/home/shanfeng/workspace/MUPTI` | 统一 collector、activity/callback、relation 查询与数据出口 |
| msight-compute | `/home/shanfeng/workspace/msight-compute` | kernel/submission/report 侧消费与验证工具 |
| msight-system | `/home/shanfeng/workspace/msight-system` | 系统级 trace、API trace、timeline 和跨层时间校准工具 |

远程没有 `/home/shanfeng/workspace/musa` 独立目录，当前应把 `linux-ddk/musa` 作为 OKR 中的 musa/DDK 仓库。

现有 OKR 方向是正确的：不能只依赖 profiling 黑盒结果，必须用 `MUSA-Runtime`、`linux-ddk/musa` 和 `MUPTI` 建立 Runtime API -> Driver API -> Core Command -> Submit -> Kernel/Activity 的白盒链路，再用 msight-compute/msight-system 做真实 workload 校准、报告和回归展示。

当前主要缺口不是“有没有 trace”，而是“缺一个跨仓库统一的 ModelEvent schema、relation schema、OKR result schema 和验收产物目录”。因此后续落地应从 M1 最小闭环收敛到五仓库共同交付，而不是继续堆分散日志。

## 2. OKR 目标重新落地为五仓库交付

### 2.1 Objective 1：Driver/API 白盒性能建模

目标应落到以下可验证产物：

| KR | 当前文档目标 | 五仓库落地产物 |
| --- | --- | --- |
| KR1 | 3 个 workload 基线、Top90 Driver API | msight-system/msight-compute 采集基线；MUPTI 输出 API/activity；insights 固化 Top90 清单 |
| KR2 | Top90 source rule、ModelEvent、成本分项 | `MUSA-Runtime` + `linux-ddk/musa` source rule；DDK 内部 ModelEvent；MUPTI schema |
| KR3 | collector、relation builder、cost model v1 | MUPTI collector + relation builder；msight 报告 join 校验器；cost breakdown 输出 |
| KR4 | profiling 校准和 3 个模型报告 | msight-compute kernel/submission 对齐；msight-system 时间线/系统事件对齐；模型报告 |

### 2.2 Objective 2：行为分析与 CTS 沉淀

行为 CTS 不应只校验 API 返回值，应校验事件签名：

| 行为 | 事件签名核心 | 承载建议 |
| --- | --- | --- |
| launch pattern | Runtime API、Driver API、command create/build/submit、kernel activity | `musa_benchmarks` + MUPTI collector |
| memory pool | alloc/free、pool hit/miss、grow/trim、allocation relation | `linux-ddk/musa` memoryPool + benchmarks |
| stream/event sync | stream wait、waited command/activity、sync reason | `linux-ddk/musa` stream/command + msight-system timeline |
| graph launch | graph instantiate/update/launch/node relation | DDK graph tracepoint + msight-compute report |
| Green Context/resource | context/resource split、stream binding、kernel isolation metric | `musa_benchmarks` GreenContext case |

## 3. 仓库技术洞察

### 3.1 `linux-ddk/musa`

当前能力：

- Driver API wrapper、callback 和 correlation id 基础已经存在。
- `src/driver/internal.h` 已有 API wrapper correlation id 相关逻辑。
- `src/driver/callback.cpp` 已有 subscribe/enable/issue callback 机制。
- `src/driver/mupti/tracepoints.h` 已有 kernel、memcpy、memset、sync、graph hooks。
- `src/musa_shared_include/export_table.h` 已有 submission/correlation 关联函数，例如 `AssignSubmissionToCorrelation`。
- M1 三条主路径已经能从源码追踪：
  - launch：`src/driver/mu_module.cpp` -> `src/musa/core/stream.cpp` -> `src/musa/core/context.cpp` -> `src/musa/core/command/dispatchCommand.cpp` -> `src/musa/core/command/command.cpp` -> HAL/M3D queue。
  - alloc/free：`src/driver/mu_memory.cpp`、`src/musa/core/memoryPool.cpp`。
  - stream sync：`src/driver/mu_stream.cpp`、`src/musa/core/stream.cpp`、command wait/semaphore wait。

关键缺口：

- 现有 hook 更偏 activity/callflow，缺统一 `ModelEvent`。
- `ResolveDependency`、`QueueCommand`、`DispatchCommand::Build`、`Command::SubmitToQueue`、HAL/M3D submit、DRM ioctl 边界没有稳定 span/relation 事件。
- memory pool 的 hit/miss/grow/trim、free merge、pool lock 等仍未形成可建模字段。
- sync wait 缺少 wait reason、waited command/activity、device wait 与 host overhead 的分离。

落地任务：

| 优先级 | 任务 | 切点 |
| --- | --- | --- |
| P0 | 定义 DDK 内部 ModelEvent 发射接口 | `src/driver/mupti/tracepoints.h` 或新 private hook |
| P0 | launch M1 事件 | `Stream::CmdLaunchKernel`、`Context::ResolveDependencyAndQueueCommand`、`Stream::QueueCommand`、`DispatchCommand::Build`、`Command::SubmitToQueue` |
| P0 | relation 事件 | API correlation -> command -> submission -> kernel activity |
| P1 | memory pool 事件 | `src/musa/core/memoryPool.cpp`、`src/driver/mu_memory.cpp` |
| P1 | sync wait 事件 | `src/driver/mu_stream.cpp`、`src/musa/core/stream.cpp`、command wait/semaphore wait |
| P2 | HAL/M3D/DRM 边界 | `src/hal/m3d/.../queue.cpp`、DRM queue submit |

### 3.2 `MUSA-Runtime`

当前能力：

- `src/internal.cpp`/`src/internal.h` 已围绕 `ExportTableManager` 管理 Runtime/Driver export table。
- Runtime 侧可通过 `IncrCorrelationId`、`GetDuringApiInvocation`、`SetDuringApiInvocation` 维护 API invocation 状态。
- `src/mupti/hooks.cpp`、`src/mupti/hooks.h` 已提供 Runtime MUPTI hook 基础。
- `src/musa_wrappers_generated.cpp` 覆盖大量 runtime wrapper，是 Runtime API enter/exit 和 source rule 的入口。
- 当前远程已有未提交改动，涉及 `src/internal.h`、`src/musa_device.cpp`、`src/musa_memory.cpp`、`src/musa_module.cpp`、`src/musa_stream.cpp`，说明 Runtime 侧 callflow/可观测性正在演进，不能回退。

关键缺口：

- Runtime API 到 Driver API 的关系没有被统一输出为 relation event。
- Runtime 初始化、module/function cache、stream/memory API 映射仅能靠源码推理，缺少可采集阶段事件。
- Runtime wrapper 的开销没有单独进入 `runtime_us` 成本项。
- `musaLaunchKernel`、`musaMalloc/musaFree`、`musaStreamSynchronize` 与对应 Driver API 的 source rule 还缺 machine-readable 版本。

落地任务：

| 优先级 | 任务 | 切点 |
| --- | --- | --- |
| P0 | Runtime API span | generated wrapper 或 `ApiTrace`/`ApiInvocationGuard` 等统一入口 |
| P0 | Runtime->Driver relation | Runtime wrapper 调 Driver export table 前后 |
| P0 | M1 source rule | `musaLaunchKernel`、`musaMalloc`、`musaFree`、`musaStreamSynchronize` |
| P1 | module/function cache event | `src/musa_module.cpp` |
| P1 | memory/stream API mapping event | `src/musa_memory.cpp`、`src/musa_stream.cpp` |
| P2 | Runtime overhead report | trace off/API-only/internal targeted 三模式 |

### 3.3 `MUPTI`

当前能力：

- `src/api/activity.cpp` 已有 `muptiActivityEnable`、`muptiActivityRegisterCallbacks`、`muptiActivityFlushAll`、external correlation API。
- `src/api/callback.cpp` 已有 subscribe、domain、callback enable 控制。
- `src/api/internal.cpp` 已导出 correlation/submission/graph 查询能力，如 kick/submission/correlation 反查，适合 relation builder。
- `src/core/process_callback.cpp` 已在 Runtime/Driver callback 中处理 `correlationId` 和 external correlation。
- `src/core/core.cpp`、`src/core/static.cpp` 维护 kernel/kick、submission/correlation、graph/sync map。
- `src/core/buffer.cpp` 有 activity buffer、flush 和 dropped record 统计。
- `src/injection/injection.cpp` 已负责加载 Runtime/Driver export table、注入 hook、订阅 Tools callback。

关键缺口：

- `activity_record_kinds.inc` 当前主要覆盖 DRIVER/RUNTIME/MEMCPY/MEMSET/KERNEL/CONCURRENT_KERNEL/SYNCHRONIZATION/EXTERNAL_CORRELATION/GRAPH_TRACE/部分 memory op；overhead、marker、module 等 record 仍不完整。
- 缺 `ModelEvent` record/private hook 的正式 schema。
- collector 输出还没固定为 `api_events`、`model_events`、`activity_events`、`relations`、`api_cost_breakdown`。
- buffer 质量指标已有基础，但尚未作为 OKR 必交付物输出。

落地任务：

| 优先级 | 任务 | 切点 |
| --- | --- | --- |
| P0 | ModelEvent schema/private hook | `include/mupti_activity.h`、`src/api/activity.cpp`、`src/core/types.h` |
| P0 | 最小 collector | activity callback + flush + JSONL/Parquet 导出 |
| P0 | relation builder | `src/api/internal.cpp` correlation/submission 查询 |
| P0 | event quality report | dropped record、buffer overflow、flush error |
| P1 | activity 支持矩阵 | `src/core/activity_record_kinds.inc` |
| P1 | overhead 压测 | buffer size、flush 策略、debug 开关 |

### 3.4 `msight-compute`

当前能力：

- `protos/mc_common.proto` 已有 MUPTI 配置、kernel/device/distribution 数据模型。
- `module/capture/mc_app_replay_profiler.cpp` 能下发 profile/MUPTI 配置给注入层。
- `module/capture/mc_mupti_capture.cpp` 能回收 `KernelInfo`、`ExtraGpuConfig`、`DistributionInfo`、`DeviceAttrs`。
- `mcu_cli/mc_capture_handler.cpp`、`module/report/write/mc_report_writer.cpp`、`module/report/read/mc_report_reader.cpp` 已有报告写入/读取链路。
- GUI/CLI 已能消费 kernel、submission、details、timeline 等报告数据。

关键缺口：

- 没有直接表达 Driver ModelEvent、stage latency、source rule/cost breakdown 的 proto/report page。
- 现有 report 更偏 kernel/metrics，不足以承载 OKR 的 API 成本分项和源码归因。
- compute 与 system 之间缺统一 correlation schema。

落地任务：

| 优先级 | 任务 | 切点 |
| --- | --- | --- |
| P1 | OKR result proto/JSON | baseline、target、actual、pass/fail、regression reason |
| P1 | cost breakdown report | API -> runtime/driver/memory/queue/build/submit/sync/unknown |
| P1 | kernel overlap report | Top50 kernel Top-K overlap >= 90% |
| P2 | GUI/CLI 展示 | report reader/exporter、timeline/details 页面 |

### 3.5 `msight-system`

当前能力：

- `protos/event_type.proto` 覆盖系统指标事件类型。
- `protos/target_profile.proto` 覆盖 trace/session/report 状态。
- `sub_module/apitrace_adapter/src/resolve_facade.cpp` 已有 API trace 时间校准和解析基础。
- mtml basic 模块能采集温度、功耗、PCIe、MTLink 等系统指标。
- 现有 insight 文档显示 msight-system 已有 MUPTI 注入、activity buffer 回调、报告写入、statistics、expert rules 等链路。

关键缺口：

- 偏系统级观测，不是白盒 ModelEvent 主报告面。
- 与 msight-compute/MUPTI 的统一时间戳、pid/tid、stream、submission、correlation schema 未固化。
- 缺最小 join 校验器，无法自动证明系统 trace、API trace、kernel report 和 ModelEvent 的时间线一致。

落地任务：

| 优先级 | 任务 | 切点 |
| --- | --- | --- |
| P1 | 统一关联键 | pid/tid/device/context/stream/submission/correlation/timestamp |
| P1 | compute/system join validator | `mcu-rep` + `msys-rep` 输出对齐 |
| P2 | system context enrichment | 温度/功耗/PCIe/MTLink 对 API/kernel 区间聚合 |
| P2 | regression explanation | 区分软件提交成本、设备执行成本、系统资源变化 |

## 4. 当前技术方案复审

### 4.1 正确部分

1. 把 Driver/API 建模定义为白盒模型是正确的。仅靠 profiling 无法解释 Runtime wrapper、Driver queue、command build、submit、sync wait、memory pool 等软件成本。
2. M1 从 `muLaunchKernel`、`muMemAlloc/muMemFree`、`muStreamSynchronize` 切入是正确的。这三类 API 分别覆盖 command、memory、sync 三条主路径。
3. MUPTI 作为统一 collector 是正确的。Runtime/Driver callback、activity buffer、correlation/submission 查询都已经有基础。
4. `musa_benchmarks` 作为 calibration/overhead/CTS 载体是正确的，但它不是 ModelEvent 的主实现仓库。
5. msight-compute/msight-system 应作为校准、展示和报告工具，而不是替代 Runtime/DDK 内部事件。

### 4.2 需要修正的部分

| 问题 | 影响 | 修正 |
| --- | --- | --- |
| 方案文档偏多，交付边界分散 | 难以执行 | 固化五仓库交付矩阵和 M1 DoD |
| ModelEvent schema 未落到代码接口 | 无法采集内部成本 | 先做 private hook，不急于 public ABI |
| relation schema 未统一 | API->kernel 召回率无法验收 | 统一 correlation、command、submission、activity ID |
| msight 侧缺 OKR result schema | 报告难以自动验收 | 增加 JSON/proto 输出和 join validator |
| source rule 仍偏文档 | 不能自动覆盖 Top90 | YAML 化 source rule，绑定事件和成本项 |
| overhead 验收未工程化 | 埋点可能污染性能 | 固定 trace off/API-only/internal targeted 三模式 |

## 5. 推荐实施路线

### M0：冻结目标和基线

交付物：

```text
workload_matrix.yaml
run_metadata.yaml
top_runtime_api.csv
top_driver_api.csv
top_kernel.csv
trace_gap_report.md
```

任务：

1. 固定 3 个 workload：dense decoder、MoE、训练或长上下文。
2. 用 msight-system/msight-compute/MUPTI 采集基线。
3. 固定 Top90 Driver API 清单。
4. 标记当前 trace 缺口：缺 command、submission、memory pool、sync reason、HAL/M3D submit。

### M1：最小白盒闭环

M1 只做三类 API：

```text
muLaunchKernel / musaLaunchKernel
muMemAlloc / muMemFree / musaMalloc / musaFree
muStreamSynchronize / musaStreamSynchronize
```

五仓库交付：

| 仓库 | M1 交付 |
| --- | --- |
| `MUSA-Runtime` | Runtime API span、Runtime->Driver relation、三类 Runtime source rule |
| `linux-ddk/musa` | command/memory/sync ModelEvent、command/submission relation |
| `MUPTI` | collector、ModelEvent record、relation builder、event quality |
| `msight-compute` | kernel/submission 对齐报告、Top-K overlap 验证 |
| `msight-system` | API/system timeline 对齐、join validator |

M1 DoD：

1. 三类 API 均能输出 Runtime API、Driver API、ModelEvent、activity 四类事件。
2. 能重建 Runtime API -> Driver API -> command -> submission -> kernel。
3. `muStreamSynchronize` 能说明等待对象和等待原因。
4. alloc/free 能说明 pool hit/miss/grow/free 行为。
5. `api_cost_breakdown` 输出 runtime、driver、memory、queue、build、submit、sync、unknown。
6. launch relation recall >= 95%。
7. unknown cost <= 15%。
8. trace off/API-only/internal targeted 三模式 overhead 有报告。

### M2：扩展 Top90 API

任务：

1. 把 M1 schema 扩展到 Top90 Driver API。
2. source rule YAML 化，字段固定：入口、核心路径、状态变量、事件、成本项、验证用例、异常分支。
3. 增加 graph、event、memcpy/memset、module/function、green context、memory pool async 等路径。
4. 用 `musa_benchmarks` 补齐 microbench/CTS 触发用例。

### M3：模型 v1 和报告

任务：

1. relation builder 输出 `relations.parquet` 或等价表。
2. cost model 输出 `api_cost_breakdown.parquet`。
3. msight-compute 输出 kernel overlap 报告。
4. msight-system 输出 timeline join 质量报告。
5. 三个 workload 输出白盒报告：API 热点、kernel 热点、成本分项、源码归因、事件质量、overhead、优化建议。

### M4：行为 CTS 和版本回归

任务：

1. 把 launch/memory/sync/graph/green-context 行为固化成 CTS 或 benchmark。
2. 每个行为用例校验事件签名和关系，不只看 API 返回值。
3. 输出 SDK diff report：区分预期变化、性能回退、事件缺失、模型不可解释。

## 6. 立即可执行的工程清单

### 6.1 第一周：冻结 schema 和 M1 source rule

| 事项 | 输出 |
| --- | --- |
| ModelEvent header/schema | `model_event_schema.md` |
| Relation schema | `relation_schema.md` |
| OKR result schema | `okr_result_schema.md` |
| M1 source rules | `source_rules/muLaunchKernel.yaml`、`muMemAlloc.yaml`、`muMemFree.yaml`、`muStreamSynchronize.yaml` |
| M1 benchmark/demo | `musaSetDevice -> musaMalloc -> launch -> stream sync -> free` |

### 6.2 第二周：打通 private hook 和 collector

| 仓库 | 任务 |
| --- | --- |
| MUPTI | 新增 ModelEvent private record、collector JSONL 输出、event quality |
| MUSA-Runtime | Runtime API span、Runtime->Driver relation |
| linux-ddk/musa | launch command/build/submit span、memory pool span、sync wait span |

### 6.3 第三周：relation builder 和 cost breakdown

| 事项 | 验收 |
| --- | --- |
| API->command->submission->kernel relation | recall >= 95% |
| alloc/free relation | ptr/size/pool 行为可关联 |
| sync relation | waited command/activity 可解释 |
| cost breakdown | unknown <= 15% |

### 6.4 第四周：msight 校准和报告

| 工具 | 输出 |
| --- | --- |
| msight-compute | kernel overlap、submission 对齐、OKR result JSON |
| msight-system | timeline join、系统资源上下文 |
| insights/okr | M1 验收报告、M2 Top90 扩展计划 |

## 7. 建议新增/统一的目录结构

建议在 `insights/okr` 下固定：

```text
insights/okr/
  model_event_schema.md
  relation_schema.md
  okr_result_schema.md
  source_rules/
    muLaunchKernel.yaml
    muMemAlloc.yaml
    muMemFree.yaml
    muStreamSynchronize.yaml
  m1_validation/
    event_quality_report.md
    relation_recall_report.md
    overhead_report.md
    api_cost_breakdown.csv
  workload_baseline/
    workload_matrix.yaml
    run_metadata.yaml
    top_runtime_api.csv
    top_driver_api.csv
    top_kernel.csv
```

这些文件应作为 OKR 工程事实来源，避免继续散落在长篇方案文档中。

## 9. ModelEvent 分阶段 API 清单与执行路径

本节把 M1-M4 的 API 范围、执行路径、需要补齐的 ModelEvent 和阶段验收固定下来。原则是：

```text
M1：只做最小闭环，证明链路可用。
M2：扩展到 Top90 Driver API 的 source rule 和事件覆盖。
M3：把事件和 relation 转成 cost model、msight 对齐和 workload 报告。
M4：把 API 行为沉淀成 CTS / benchmark / SDK 回归签名。
```

### 9.1 当前 MUPTI 能力复用边界

ModelEvent 不替代现有 MUPTI callback/activity，而是复用现有能力并补齐 Driver/Core 内部阶段。

| 能力 | 当前已有 | ModelEvent 复用方式 |
| --- | --- | --- |
| Runtime callback/activity | 已能记录 Runtime API enter/exit | 继续作为 `RUNTIME_API_SPAN` 的外层 API 时间来源 |
| Driver callback/activity | 已能记录 Driver API enter/exit | 继续作为 `DRIVER_API_SPAN` 的外层 API 时间来源 |
| correlation id | Runtime/Driver callback 已有基础 | ModelEvent 携带同一个 `correlation_id` |
| kernel/memcpy/memset/sync activity | 已有 activity record 基础 | 用于 `submission -> activity` 对齐和 kernel overlap 校准 |
| activity buffer | 已有 buffer request/flush/drop 统计 | 新增 private ModelEvent record，复用 buffer 和质量指标 |
| injection/export table | 已能注入 Runtime/Driver hook | 新增 private `MuptiEmitModelEvent` / `MuptiModelEventEnabled` hook |
| submission/correlation map | 已有 submission/correlation 查询基础 | ModelEvent 补充 `command_id`、`submission_id` 后复用 relation builder |

当前 launch 覆盖边界：

| 链路节点 | 当前 MUPTI 覆盖 | ModelEvent 需要补齐 |
| --- | --- | --- |
| `musaLaunchKernel` | Runtime API enter/exit | Runtime->Driver relation |
| `muLaunchKernel` | Driver API enter/exit | Driver API -> command relation |
| `Stream::CmdLaunchKernel` | 无稳定内部事件 | `COMMAND_CREATE` |
| `Context::ResolveDependencyAndQueueCommand` | 无稳定内部事件 | `DEPENDENCY_RESOLVE_SPAN` |
| `Stream::QueueCommand` | 无稳定内部事件 | `STREAM_QUEUE_COMMAND_SPAN` |
| `Stream::AsyncSubmit` | 无稳定内部事件 | `COMMAND_MERGE_DECISION` |
| `DispatchCommand::Build` | 无稳定内部事件 | `COMMAND_BUILD_SPAN` |
| `Command::SubmitToQueue` | 不完整 | `COMMAND_SUBMIT_SPAN`、`COMMAND_SUBMISSION_RELATION` |
| HAL/M3D/DRM submit | 基本未覆盖 | `HAL_M3D_SUBMIT_SPAN`、`DRM_IOCTL_SUBMIT_SPAN` |
| kernel activity | 已有基础 | 与 submission 关联、校准 Top-K overlap |

### 9.2 M1 API 清单与执行路径

M1 只覆盖三类 API，目标是打通 `Runtime API -> Driver API -> command/memory/sync ModelEvent -> activity -> cost breakdown`。

#### 9.2.1 `musaLaunchKernel` / `muLaunchKernel`

执行路径：

```text
用户调用 musaLaunchKernel
  -> MUSA-Runtime wrapper / ApiTrace / ApiInvocationGuard
  -> Runtime export table 调 Driver muLaunchKernel
  -> Driver muLaunchKernel wrapper
  -> muapiLaunchKernel
  -> function / module / stream 参数解析
  -> Stream::CmdLaunchKernel
  -> DispatchCommand 创建
  -> Context::ResolveDependencyAndQueueCommand
  -> Stream::QueueCommand
  -> Stream::AsyncSubmit
  -> DispatchCommand::Build
  -> DispatchCommand::Submit
  -> Command::SubmitToQueue
  -> HAL Queue::Submit
  -> M3D Queue::SubmitInternal
  -> DRM Queue::OsSubmit
  -> KMD / firmware
  -> kernel activity
```

M1 必须事件：

| 事件 | 类型 | 位置 | 成本项 |
| --- | --- | --- | --- |
| `RUNTIME_API_SPAN` | span | Runtime wrapper | `runtime_wrapper` |
| `RUNTIME_DRIVER_RELATION` | relation | Runtime 调 Driver 前后 | relation |
| `DRIVER_API_SPAN` | span | Driver wrapper | `driver_wrapper` |
| `COMMAND_CREATE` | instant | `Stream::CmdLaunchKernel` | `command_create` |
| `DEPENDENCY_RESOLVE_SPAN` | span | `Context::ResolveDependencyAndQueueCommand` | `dependency` |
| `STREAM_QUEUE_COMMAND_SPAN` | span | `Stream::QueueCommand` | `queue` |
| `COMMAND_MERGE_DECISION` | instant/counter | `Stream::AsyncSubmit` | `merge_check` |
| `COMMAND_BUILD_SPAN` | span | `DispatchCommand::Build` | `command_build` |
| `COMMAND_SUBMIT_SPAN` | span | `Command::SubmitToQueue` | `submit` |
| `COMMAND_SUBMISSION_RELATION` | relation | submit 成功后 | relation |
| `SUBMISSION_ACTIVITY_RELATION` | relation | MUPTI relation builder | relation |

#### 9.2.2 `musaMalloc` / `muMemAlloc`

执行路径：

```text
用户调用 musaMalloc
  -> MUSA-Runtime wrapper / ApiTrace / ApiInvocationGuard
  -> Runtime export table 调 Driver muMemAlloc
  -> Driver muMemAlloc wrapper
  -> muapiMemAlloc
  -> current context / device lookup
  -> memory object / allocation descriptor 创建
  -> memory pool lookup
  -> pool allocate
  -> pool hit 或 pool miss
  -> pool grow / HAL allocate / map
  -> allocation object 注册
  -> 返回 device pointer
```

M1 必须事件：

| 事件 | 类型 | 位置 | 成本项 |
| --- | --- | --- | --- |
| `RUNTIME_API_SPAN` | span | Runtime wrapper | `runtime_wrapper` |
| `RUNTIME_DRIVER_RELATION` | relation | Runtime 调 Driver 前后 | relation |
| `DRIVER_API_SPAN` | span | `mu_memory.cpp` | `driver_wrapper` |
| `CONTEXT_LOOKUP_SPAN` | span | Driver/Core context lookup | `context_lookup` |
| `MEM_POOL_LOOKUP` | instant | `memoryPool.cpp` | `pool_lookup` |
| `MEM_POOL_ALLOC_SPAN` | span | `memoryPool.cpp` | `pool_alloc` |
| `MEM_POOL_HIT` | instant/counter | `memoryPool.cpp` | `pool_hit` |
| `MEM_POOL_GROW_SPAN` | span | `memoryPool.cpp` / HAL alloc | `pool_grow` |
| `HAL_ALLOC_SPAN` | span | HAL allocate/map | `hal_alloc` |
| `API_MEMORY_RELATION` | relation | allocation object 建立后 | relation |

#### 9.2.3 `musaFree` / `muMemFree`

执行路径：

```text
用户调用 musaFree
  -> MUSA-Runtime wrapper / ApiTrace / ApiInvocationGuard
  -> Runtime export table 调 Driver muMemFree
  -> Driver muMemFree wrapper
  -> muapiMemFree
  -> memory object lookup
  -> allocation ownership / context 校验
  -> memory pool free
  -> free merge / pool update
  -> 可选 trim / release
  -> allocation object 注销
```

M1 必须事件：

| 事件 | 类型 | 位置 | 成本项 |
| --- | --- | --- | --- |
| `RUNTIME_API_SPAN` | span | Runtime wrapper | `runtime_wrapper` |
| `RUNTIME_DRIVER_RELATION` | relation | Runtime 调 Driver 前后 | relation |
| `DRIVER_API_SPAN` | span | `mu_memory.cpp` | `driver_wrapper` |
| `MEM_OBJECT_LOOKUP_SPAN` | span | memory object lookup | `memory_lookup` |
| `MEM_POOL_FREE_SPAN` | span | `memoryPool.cpp` | `pool_free` |
| `MEM_FREE_MERGE` | instant/counter | `memoryPool.cpp` | `free_merge` |
| `MEM_POOL_UPDATE_SPAN` | span | `memoryPool.cpp` | `pool_update` |
| `API_MEMORY_RELATION` | relation | free 对象确认后 | relation |

#### 9.2.4 `musaStreamSynchronize` / `muStreamSynchronize`

执行路径：

```text
用户调用 musaStreamSynchronize
  -> MUSA-Runtime wrapper / ApiTrace / ApiInvocationGuard
  -> Runtime export table 调 Driver muStreamSynchronize
  -> Driver muStreamSynchronize wrapper
  -> muapiStreamSynchronize
  -> stream lookup
  -> Stream::WaitFinish
  -> last command lookup
  -> Command::Wait 或 Stream::AsyncWait
  -> semaphore / timeline wait
  -> engine last error query
  -> user pool update
  -> 返回同步状态
```

M1 必须事件：

| 事件 | 类型 | 位置 | 成本项 |
| --- | --- | --- | --- |
| `RUNTIME_API_SPAN` | span | Runtime wrapper | `runtime_wrapper` |
| `RUNTIME_DRIVER_RELATION` | relation | Runtime 调 Driver 前后 | relation |
| `DRIVER_API_SPAN` | span | `mu_stream.cpp` | `driver_wrapper` |
| `STREAM_LOOKUP_SPAN` | span | Driver stream lookup | `stream_lookup` |
| `STREAM_WAIT_FINISH_SPAN` | span | `Stream::WaitFinish` | `sync_wait` |
| `LAST_COMMAND_LOOKUP` | instant | `Stream::WaitFinish` | `last_command_lookup` |
| `COMMAND_WAIT_SPAN` | span | command wait | `command_wait` |
| `SEMAPHORE_WAIT_SPAN` | span | semaphore/timeline wait | `device_wait` |
| `ENGINE_ERROR_QUERY_SPAN` | span | engine error query | `error_query` |
| `SYNC_WAIT_REASON` | instant | wait 分支判断处 | `sync_reason` |
| `API_SYNC_RELATION` | relation | waited command/activity 确认后 | relation |

M1 DoD：

```text
launch relation recall >= 95%
unknown cost <= 15%
trace off overhead <= 0.1% 或 benchmark 噪声内
API-only overhead <= 1%
internal targeted overhead <= 3%
dropped record / overflow / flush error 必须输出
```

### 9.3 M2 API 清单与执行路径

M2 扩展到 Top90 Driver API。实际 Top90 以 M0 workload 基线为准；在没有最终 Top90 CSV 前，先按以下 API 家族作为 M2 source rule 全量候选。M2 的目标是每个 API 都有 source rule、事件签名、成本项和验证用例。

#### 9.3.1 Launch / module / function API

| API | 执行路径 | 必要 ModelEvent |
| --- | --- | --- |
| `musaLaunchKernel` / `muLaunchKernel` | Runtime wrapper -> Driver wrapper -> `Stream::CmdLaunchKernel` -> dependency -> queue -> build -> submit -> HAL/M3D -> activity | M1 launch 全套事件 |
| `musaLaunchCooperativeKernel` / `muLaunchCooperativeKernel` | Runtime wrapper -> Driver cooperative launch -> cooperative 参数校验 -> stream command -> dependency -> queue -> build -> submit | `COOPERATIVE_LAUNCH_VALIDATE_SPAN` + launch 全套事件 |
| `musaFuncGetAttributes` / `muFuncGetAttribute` | Runtime wrapper -> Driver module/function lookup -> attribute query/cache | `FUNCTION_LOOKUP_SPAN`、`MODULE_METADATA_QUERY_SPAN` |
| `musaFuncSetAttribute` / `muFuncSetAttribute` | Runtime wrapper -> Driver function lookup -> attribute validate -> metadata update | `FUNCTION_LOOKUP_SPAN`、`FUNCTION_ATTRIBUTE_UPDATE` |
| `musaModuleLoad` / `muModuleLoad` | Runtime/Driver wrapper -> module file load -> ELF/fatbin parse -> code object register | `MODULE_LOAD_SPAN`、`MODULE_PARSE_SPAN`、`MODULE_REGISTER_SPAN` |
| `musaModuleLoadData` / `muModuleLoadData` | Runtime/Driver wrapper -> memory image parse -> code object register | `MODULE_LOAD_DATA_SPAN`、`MODULE_PARSE_SPAN` |
| `musaModuleGetFunction` / `muModuleGetFunction` | Runtime/Driver wrapper -> module lookup -> function symbol lookup -> cache hit/miss | `MODULE_LOOKUP_SPAN`、`FUNCTION_LOOKUP_SPAN`、`FUNCTION_CACHE_HIT` |
| `musaGetKernel` / driver function lookup 等价路径 | Runtime wrapper -> module/function registry lookup -> kernel handle 返回 | `FUNCTION_LOOKUP_SPAN`、`FUNCTION_CACHE_HIT` |

#### 9.3.2 Memory allocation / free / pool API

| API | 执行路径 | 必要 ModelEvent |
| --- | --- | --- |
| `musaMalloc` / `muMemAlloc` | Runtime wrapper -> Driver wrapper -> context lookup -> pool lookup -> pool alloc -> hit/grow -> HAL alloc/map | M1 alloc 全套事件 |
| `musaFree` / `muMemFree` | Runtime wrapper -> Driver wrapper -> memory lookup -> pool free -> merge/update/trim | M1 free 全套事件 |
| `musaMallocAsync` / `muMemAllocAsync` | Runtime wrapper -> Driver async alloc -> stream lookup -> pool alloc command -> queue -> submit | `ASYNC_ALLOC_COMMAND_CREATE`、`MEM_POOL_ALLOC_SPAN`、launch queue/submit 事件 |
| `musaFreeAsync` / `muMemFreeAsync` | Runtime wrapper -> Driver async free -> stream lookup -> free command -> dependency -> queue -> submit | `ASYNC_FREE_COMMAND_CREATE`、`MEM_POOL_FREE_SPAN`、queue/submit 事件 |
| `musaMallocHost` / `muMemHostAlloc` | Runtime wrapper -> Driver host alloc -> pinned memory allocate -> map/register | `HOST_ALLOC_SPAN`、`HOST_PIN_SPAN`、`HOST_MAP_SPAN` |
| `musaHostRegister` / `muMemHostRegister` | Runtime wrapper -> Driver host register -> page pin -> device map | `HOST_REGISTER_SPAN`、`HOST_PIN_SPAN`、`HOST_MAP_SPAN` |
| `musaHostUnregister` / `muMemHostUnregister` | Runtime wrapper -> Driver unregister -> unmap -> unpin | `HOST_UNMAP_SPAN`、`HOST_UNPIN_SPAN` |
| `musaMemGetInfo` / `muMemGetInfo` | Runtime wrapper -> Driver memory info query -> device/pool stats | `MEM_INFO_QUERY_SPAN` |
| `musaPointerGetAttributes` / `muPointerGetAttribute(s)` | Runtime wrapper -> Driver pointer lookup -> allocation metadata query | `POINTER_LOOKUP_SPAN`、`MEM_OBJECT_LOOKUP_SPAN` |
| memory pool trim / attribute APIs | Runtime/Driver wrapper -> pool lookup -> trim/update attributes | `MEM_POOL_LOOKUP`、`MEM_POOL_TRIM_SPAN`、`MEM_POOL_ATTRIBUTE_UPDATE` |

#### 9.3.3 Memcpy / memset API

| API | 执行路径 | 必要 ModelEvent |
| --- | --- | --- |
| `musaMemcpy` / `muMemcpy` | Runtime wrapper -> Driver copy classify -> sync/async copy command -> dependency -> queue -> build -> submit -> memcpy activity | `COPY_CLASSIFY_SPAN`、`COPY_COMMAND_CREATE`、queue/build/submit、`SUBMISSION_ACTIVITY_RELATION` |
| `musaMemcpyAsync` / `muMemcpyAsync` | Runtime wrapper -> Driver async copy -> stream lookup -> copy command -> queue -> submit -> memcpy activity | `COPY_COMMAND_CREATE`、`STREAM_QUEUE_COMMAND_SPAN`、`COPY_BUILD_SPAN`、`COMMAND_SUBMIT_SPAN` |
| `musaMemcpyHtoD` / `muMemcpyHtoD` | Driver copy H2D path -> host pointer validate -> dst lookup -> copy command -> submit | `COPY_H2D_VALIDATE_SPAN`、`COPY_COMMAND_CREATE` |
| `musaMemcpyDtoH` / `muMemcpyDtoH` | Driver copy D2H path -> src lookup -> host pointer validate -> copy command -> submit | `COPY_D2H_VALIDATE_SPAN`、`COPY_COMMAND_CREATE` |
| `musaMemcpyDtoD` / `muMemcpyDtoD` | Driver copy D2D path -> src/dst lookup -> peer/local classify -> copy command -> submit | `COPY_D2D_CLASSIFY_SPAN`、`COPY_COMMAND_CREATE` |
| `musaMemcpy2D/3D` / `muMemcpy2D/3D` | Runtime/Driver wrapper -> copy descriptor normalize -> pitch/extent validate -> copy command -> submit | `COPY_DESCRIPTOR_NORMALIZE_SPAN`、`COPY_COMMAND_CREATE` |
| `musaMemset` / `muMemset*` | Runtime/Driver wrapper -> memory lookup -> memset command -> queue -> submit -> memset activity | `MEMSET_COMMAND_CREATE`、`MEMSET_BUILD_SPAN`、`COMMAND_SUBMIT_SPAN` |
| `musaMemPrefetchAsync` / `muMemPrefetchAsync` | Runtime/Driver wrapper -> pointer lookup -> migration command -> stream queue -> submit | `PREFETCH_COMMAND_CREATE`、`MIGRATION_PLAN_SPAN`、queue/submit |

#### 9.3.4 Stream / event / synchronization API

| API | 执行路径 | 必要 ModelEvent |
| --- | --- | --- |
| `musaStreamCreate` / `muStreamCreate` | Runtime wrapper -> Driver stream create -> context lookup -> stream object allocate/register | `STREAM_CREATE_SPAN`、`STREAM_REGISTER_SPAN` |
| `musaStreamDestroy` / `muStreamDestroy` | Runtime wrapper -> Driver stream destroy -> wait/cleanup -> unregister/free | `STREAM_DESTROY_SPAN`、`STREAM_CLEANUP_SPAN` |
| `musaStreamSynchronize` / `muStreamSynchronize` | Runtime wrapper -> Driver sync -> stream lookup -> wait finish -> command wait -> error query | M1 sync 全套事件 |
| `musaStreamWaitEvent` / `muStreamWaitEvent` | Runtime/Driver wrapper -> stream/event lookup -> wait command create -> queue -> submit | `EVENT_WAIT_COMMAND_CREATE`、`STREAM_QUEUE_COMMAND_SPAN`、`COMMAND_SUBMIT_SPAN` |
| `musaStreamQuery` / `muStreamQuery` | Runtime/Driver wrapper -> stream lookup -> last command status query | `STREAM_STATUS_QUERY_SPAN`、`LAST_COMMAND_LOOKUP` |
| `musaDeviceSynchronize` / `muCtxSynchronize` | Runtime/Driver wrapper -> context/device stream set lookup -> wait all -> error query | `CONTEXT_WAIT_ALL_SPAN`、`COMMAND_WAIT_SPAN`、`SYNC_WAIT_REASON` |
| `musaEventCreate` / `muEventCreate` | Runtime/Driver wrapper -> event object allocate/register | `EVENT_CREATE_SPAN` |
| `musaEventRecord` / `muEventRecord` | Runtime/Driver wrapper -> stream/event lookup -> event record command -> queue -> submit | `EVENT_RECORD_COMMAND_CREATE`、queue/build/submit |
| `musaEventSynchronize` / `muEventSynchronize` | Runtime/Driver wrapper -> event lookup -> waited command lookup -> wait | `EVENT_WAIT_SPAN`、`COMMAND_WAIT_SPAN`、`SYNC_WAIT_REASON` |
| `musaEventElapsedTime` / driver event elapsed query | Runtime wrapper -> event timestamps query -> convert | `EVENT_TIMESTAMP_QUERY_SPAN` |

#### 9.3.5 Graph API

| API | 执行路径 | 必要 ModelEvent |
| --- | --- | --- |
| `musaGraphCreate` / `muGraphCreate` | Runtime/Driver wrapper -> graph object allocate/register | `GRAPH_CREATE_SPAN` |
| `musaGraphAddKernelNode` / driver graph add node | Runtime/Driver wrapper -> graph lookup -> kernel node validate -> node create | `GRAPH_NODE_CREATE_SPAN`、`GRAPH_KERNEL_NODE_VALIDATE_SPAN` |
| `musaGraphAddMemcpyNode` / driver graph memcpy node | graph lookup -> copy descriptor normalize -> node create | `GRAPH_NODE_CREATE_SPAN`、`COPY_DESCRIPTOR_NORMALIZE_SPAN` |
| `musaGraphInstantiate` / `muGraphInstantiate` | Runtime/Driver wrapper -> graph validate -> executable graph build -> dependency plan | `GRAPH_VALIDATE_SPAN`、`GRAPH_INSTANTIATE_SPAN`、`GRAPH_DEPENDENCY_PLAN_SPAN` |
| `musaGraphLaunch` / `muGraphLaunch` | Runtime/Driver wrapper -> exec graph lookup -> graph command create -> queue -> submit -> graph/kernel activities | `GRAPH_LAUNCH_COMMAND_CREATE`、queue/build/submit、`GRAPH_ACTIVITY_RELATION` |
| `musaGraphExecUpdate` / driver update | exec graph lookup -> diff/update validate -> patch executable graph | `GRAPH_EXEC_UPDATE_SPAN`、`GRAPH_DIFF_SPAN` |
| `musaGraphDestroy` / graph destroy | graph lookup -> cleanup nodes/deps -> unregister | `GRAPH_DESTROY_SPAN` |

#### 9.3.6 Context / device / resource / Green Context API

| API | 执行路径 | 必要 ModelEvent |
| --- | --- | --- |
| `musaSetDevice` / `muCtxSetCurrent` | Runtime wrapper -> device/context lookup -> TLS/current context update | `DEVICE_LOOKUP_SPAN`、`CONTEXT_SWITCH_SPAN` |
| `musaGetDeviceProperties` / `muDeviceGetAttribute` | Runtime/Driver wrapper -> device lookup -> property/attribute query | `DEVICE_ATTRIBUTE_QUERY_SPAN` |
| `muDeviceGetDevResource` | Driver wrapper -> device resource query -> SM/resource descriptor build | `DEVICE_RESOURCE_QUERY_SPAN` |
| `muDevSmResourceSplitByCount` | Driver wrapper -> resource validate -> SM resource split | `SM_RESOURCE_SPLIT_SPAN` |
| `muDevResourceGenerateDesc` | Driver wrapper -> resource descriptor generate | `RESOURCE_DESC_GENERATE_SPAN` |
| `muGreenCtxCreate` | Driver wrapper -> resource desc validate -> context/resource create -> KMD bind | `GREEN_CTX_CREATE_SPAN`、`RESOURCE_BIND_SPAN` |
| `muCtxFromGreenCtx` | Driver wrapper -> green context lookup -> context handle derive | `GREEN_CTX_TO_CONTEXT_SPAN` |
| `muGreenCtxStreamCreate` | Driver wrapper -> green context lookup -> stream create/bind | `GREEN_STREAM_CREATE_SPAN`、`RESOURCE_BIND_SPAN` |
| `muGreenCtxDestroy` | Driver wrapper -> stream/context cleanup -> resource unbind/destroy | `GREEN_CTX_DESTROY_SPAN`、`RESOURCE_UNBIND_SPAN` |

#### 9.3.7 HAL/M3D/KMD 边界 API/内部提交点

这些不是用户 API，但属于 Top90 API 成本分项的必要边界。

| 内部边界 | 执行路径 | 必要 ModelEvent |
| --- | --- | --- |
| HAL Queue submit | `Command::SubmitToQueue` -> HAL queue submit | `HAL_QUEUE_SUBMIT_SPAN` |
| M3D submit internal | HAL -> M3D queue -> command buffer submit | `M3D_SUBMIT_INTERNAL_SPAN` |
| DRM ioctl submit | M3D queue -> DRM queue -> ioctl | `DRM_IOCTL_SUBMIT_SPAN` |
| doorbell / firmware notify | KMD submit -> doorbell/firmware notify | `FIRMWARE_NOTIFY_SPAN` 或 counter |

M2 DoD：

```text
Top90 Driver API source rule coverage = 100%
每个 source rule 都有执行路径、事件签名、成本项、验证用例
Top90 event coverage report 输出 missing event 明细
api_cost_terms_matrix.csv 覆盖 runtime/driver/memory/queue/build/submit/sync/ioctl/unknown
```

### 9.4 M3 API/工具链路径与模型输出

M3 不新增大量底层 API，重点是把 M1/M2 事件转成模型和报告。涉及的是采集、relation、join、报告 API/工具链。

| 组件/API | 执行路径 | 输出 |
| --- | --- | --- |
| `muptiActivityEnable` | collector -> MUPTI activity enable -> kind/filter state | activity/model event enable 状态 |
| `muptiActivityRegisterCallbacks` | collector -> buffer callbacks 注册 -> MUPTI buffer manager | buffer request/complete |
| `muptiActivityFlushAll` | collector -> flush activity/model buffers -> dropped/overflow 统计 | `event_quality_report.md` |
| ModelEvent private hook | Runtime/DDK -> `MuptiModelEventEnabled` -> `MuptiEmitModelEvent` -> activity buffer | `model_events.jsonl/parquet` |
| relation builder internal query | MUPTI internal maps -> correlation/submission/activity 查询 -> relation join | `relations.parquet` |
| cost model builder | api events + model events + source rules -> cost term aggregation | `api_cost_breakdown.parquet` |
| msight-compute capture | MUPTI kernel/submission info -> report writer/reader -> kernel ranking | `kernel_overlap_report.md` |
| msight-system timeline | API trace + system events -> timebase align -> timeline join | `timeline_join_report.md` |
| OKR result writer | validation metrics -> schema writer | `okr_result.json` |

M3 需要覆盖的建模 API 来源仍来自 M2 Top90；每个 API 输出统一成本项：

```text
T_api =
  T_runtime_wrapper
+ T_driver_wrapper
+ T_validation
+ T_object_lookup
+ T_memory_pool
+ T_dependency
+ T_queue
+ T_command_build
+ T_command_merge
+ T_submit
+ T_hal_m3d
+ T_ioctl
+ T_sync_wait
+ T_error_check
+ T_unknown
```

M3 DoD：

```text
Top50 kernel Top-K overlap >= 90%
Top20 Driver API host inclusive time p50 误差建议 <= 20%，超出项必须归因
Top90 API 主要耗时可解释覆盖率 >= 85%
unknown cost <= 15%
输出 model_validation_report、kernel_overlap_report、timeline_join_report、overhead_report
```

### 9.5 M4 行为 CTS / SDK 回归 API 清单与执行路径

M4 把 API 行为固化成 CTS、benchmark 和 SDK diff。每个行为用例必须校验 API 序列、ModelEvent 序列、relation、counter 和 pass/fail 规则。

#### 9.5.1 launch pattern

覆盖 API：

```text
musaLaunchKernel / muLaunchKernel
musaLaunchCooperativeKernel / muLaunchCooperativeKernel
musaFuncGetAttributes / muFuncGetAttribute
musaModuleGetFunction / muModuleGetFunction
musaStreamSynchronize / muStreamSynchronize
```

执行路径签名：

```text
Runtime launch
  -> Driver launch
  -> command create
  -> dependency resolve
  -> stream queue
  -> build
  -> merge decision
  -> submit
  -> activity
  -> sync wait 或 activity completion
```

必须校验：

```text
RUNTIME_API_SPAN -> DRIVER_API_SPAN -> COMMAND_CREATE -> COMMAND_BUILD_SPAN -> COMMAND_SUBMIT_SPAN -> KERNEL_ACTIVITY
relation recall = 100% for CTS case
dropped_records = 0
```

#### 9.5.2 memory pool behavior

覆盖 API：

```text
musaMalloc / muMemAlloc
musaFree / muMemFree
musaMallocAsync / muMemAllocAsync
musaFreeAsync / muMemFreeAsync
musaMemGetInfo / muMemGetInfo
musaPointerGetAttributes / muPointerGetAttribute(s)
```

执行路径签名：

```text
Runtime memory API
  -> Driver memory API
  -> context/device lookup
  -> memory object lookup
  -> pool lookup
  -> pool hit/miss/grow/free/merge/trim
  -> optional HAL alloc/map
```

必须校验：

```text
MEM_POOL_ALLOC_SPAN / MEM_POOL_GROW_SPAN / MEM_POOL_FREE_SPAN
API_MEMORY_RELATION
pool hit/miss/grow counter
```

#### 9.5.3 stream/event sync behavior

覆盖 API：

```text
musaStreamCreate / muStreamCreate
musaStreamDestroy / muStreamDestroy
musaStreamWaitEvent / muStreamWaitEvent
musaStreamSynchronize / muStreamSynchronize
musaStreamQuery / muStreamQuery
musaEventCreate / muEventCreate
musaEventRecord / muEventRecord
musaEventSynchronize / muEventSynchronize
musaEventElapsedTime
musaDeviceSynchronize / muCtxSynchronize
```

执行路径签名：

```text
Runtime sync/event API
  -> Driver sync/event API
  -> stream/event lookup
  -> event record/wait command create
  -> queue/submit
  -> wait finish
  -> command/semaphore wait
  -> sync reason
```

必须校验：

```text
STREAM_WAIT_FINISH_SPAN
COMMAND_WAIT_SPAN
SEMAPHORE_WAIT_SPAN
SYNC_WAIT_REASON
API_SYNC_RELATION
```

#### 9.5.4 graph behavior

覆盖 API：

```text
musaGraphCreate / muGraphCreate
musaGraphAddKernelNode
musaGraphAddMemcpyNode
musaGraphInstantiate / muGraphInstantiate
musaGraphLaunch / muGraphLaunch
musaGraphExecUpdate
musaGraphDestroy
```

执行路径签名：

```text
Graph create
  -> node add
  -> instantiate validate/build dependency plan
  -> graph launch command create
  -> queue/build/submit
  -> graph activity / kernel activity relation
```

必须校验：

```text
GRAPH_CREATE_SPAN
GRAPH_NODE_CREATE_SPAN
GRAPH_INSTANTIATE_SPAN
GRAPH_LAUNCH_COMMAND_CREATE
GRAPH_ACTIVITY_RELATION
```

#### 9.5.5 Green Context / resource behavior

覆盖 API：

```text
muDeviceGetDevResource
muDevSmResourceSplitByCount
muDevResourceGenerateDesc
muGreenCtxCreate
muCtxFromGreenCtx
muGreenCtxStreamCreate
muGreenCtxGetDevResource
muGreenCtxDestroy
musaLaunchKernel / muLaunchKernel
musaStreamSynchronize / muStreamSynchronize
```

执行路径签名：

```text
resource query
  -> SM resource split
  -> resource desc generate
  -> green context create
  -> context derive
  -> stream bind
  -> launch on green stream
  -> resource-bound activity
  -> stream sync
  -> green context destroy
```

必须校验：

```text
DEVICE_RESOURCE_QUERY_SPAN
SM_RESOURCE_SPLIT_SPAN
GREEN_CTX_CREATE_SPAN
GREEN_STREAM_CREATE_SPAN
RESOURCE_BIND_SPAN
GREEN_CTX_DESTROY_SPAN
launch relation + green context/resource fields
```

#### 9.5.6 model workload behavior

覆盖行为/API 组合：

| 行为 | 典型 API 序列 | 执行路径签名 |
| --- | --- | --- |
| dense decoder launch burst | module/function lookup -> repeated launch -> stream sync | launch pattern + command merge/queue depth |
| MoE expert routing | small kernel burst -> memcpy/memset -> sync/event wait | launch + copy + sync relation |
| KV cache allocation | alloc/free/async alloc/free -> memcpy/prefetch -> sync | memory pool + copy + sync |
| training step | alloc -> memcpy -> launch groups -> event record/wait -> sync | memory + launch + event sync + activity timeline |
| long context | repeated launches + memory pressure + sync | queue depth + pool grow + sync wait reason |

M4 DoD：

```text
behavior_cts 至少覆盖 launch、memory、stream/event sync、graph、Green Context/resource
每个 CTS 校验事件签名，不只校验 API 返回值
sdk_diff_report 能区分：预期变化、性能回退、事件缺失、模型不可解释
行为基线包含 API 序列、ModelEvent 序列、relation、counter、成本项和阈值
```

### 9.6 分阶段交付矩阵

| 阶段 | API 范围 | 主要产物 | 退出标准 |
| --- | --- | --- | --- |
| M1 | launch、alloc/free、stream sync | private ModelEvent hook、collector、M1 source rule、relations、cost breakdown | relation recall >= 95%，unknown <= 15%，overhead 达标 |
| M2 | Top90 Driver API 候选家族 | Top90 source rules、event coverage、missing event、cost terms matrix | Top90 source rule coverage = 100% |
| M3 | M2 API + msight 工具链 | cost model v1、kernel overlap、timeline join、model validation | Top50 kernel overlap >= 90%，可解释覆盖 >= 85% |
| M4 | 行为 API 组合 | behavior CTS、event signature rules、SDK diff | 行为用例可复现、可比较、可解释回归 |

## 10. 最终建议

1. 保留现有 OKR 主线，但把执行入口切到 M1 最小闭环。
2. 不再新增纯日志式插桩，统一走 ModelEvent schema + MUPTI collector。
3. `linux-ddk/musa` 优先补 command/build/submit、memory pool、sync wait 三类内部事件。
4. `MUSA-Runtime` 优先补 Runtime span 和 Runtime->Driver relation。
5. `MUPTI` 优先补 ModelEvent private record、collector、relation builder、event quality。
6. `msight-compute` 和 `msight-system` 优先补 OKR result schema、join validator 和报告输出。
7. 每个阶段必须有 DoD：relation recall、unknown cost、overhead、event quality，不满足则不能进入下一阶段。
