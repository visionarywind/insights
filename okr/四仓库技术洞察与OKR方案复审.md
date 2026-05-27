# 四仓库技术洞察与 OKR 方案复审

## 结论

当前 OKR 方向应保留“白盒软件性能建模”：从 `MUSA-Runtime` 和 `linux-ddk/musa` 源码提取状态机、队列、内存池、同步、提交和 ioctl 边界，通过 MUPTI 统一采集事件，再用模型还原执行过程和成本来源。

当前方案的主要缺口不是方向错误，而是四个仓库的工程职责没有完全闭合：

| 仓库 | OKR 中应承担的职责 | 当前缺口 |
|---|---|---|
| `MUSA-Runtime` | Runtime API 边界、Runtime 初始化、导出表、module/function cache、Runtime 到 Driver 的分发关系 | 方案已有切点，但应避免逐个手写 wrapper 插桩；应复用自动生成 wrapper、`ApiTrace`、`ApiInvocationGuard` 和 Tools callback |
| `linux-ddk/musa` | Driver API、Context、Memory、Stream、Command、Graph、HAL/M3D、ioctl 边界的白盒事件 | 源码洞察充分，但 OKR 方案需要把 Top90 API 映射到明确 source rule 和事件签名 |
| `MUPTI` | 统一采集通道、callback/activity、hook/accessor、collector、schema、flush 与 overhead 验证 | 当前 MUPTI 缺少内部软件成本事件；需要新增 ModelEvent 通路或 private hook |
| `musa_benchmarks` | 成本校准、插桩开销验收、行为 CTS、跨版本回归基线 | 当前 OKR 主方案没有把它纳入交付链路 |

`insights/okr/okr-2026-landing-plan.md` 已经比早期 replay 方案更接近正确方向。`insights/simulation/okr/musa_driver_api_perf_model_landing_plan.md` 仍以 trace replay 为第一阶段，应降级为离线验证工具，不应作为 OKR 主方案。

## 四个仓库技术洞察复审

## `linux-ddk/musa`

已有技术洞察覆盖了 Driver API、Core、HAL、M3D、memory API、command stream、graph、kernel launch、copy manager 等核心路径。对 OKR 最有价值的源码位置包括：

| 模块 | 关键位置 | 建模价值 |
|---|---|---|
| Driver API wrapper | `src/driver/mu_wrappers_generated.cpp` | Driver API enter/exit、CBID、correlation id |
| callback 分发 | `src/driver/callback.cpp` | Tools callback 开销和启用状态 |
| Context | `src/musa/core/context.cpp::ResolveDependencyAndQueueCommand` | dependency、blocking、command 入队前状态 |
| Stream | `src/musa/core/stream.cpp::QueueCommand`、`AsyncSubmit`、`Synchronize` | queue depth、build、merge、submit、wait |
| Command | `src/musa/core/command/*.cpp` | Dispatch、Memcpy、Memset、Graph command 的生命周期 |
| Memory | `src/musa/core/memory.cpp`、`src/hal/m3d/memMgr.cpp`、`src/hal/m3d/memoryPool.cpp` | pool hit/miss、chunk allocate、free merge |
| HAL/M3D | `src/hal/m3d/queue.cpp`、`cmdBuffer.cpp`、`memory.cpp` | command buffer、queue submit、KMD 边界 |
| MUPTI tracepoint | `src/driver/mupti/tracepoints.h` | kernel、memcpy、graph、sync activity 的现有入口 |

需要补齐的内容：

1. 为 Top90 Driver API 建立 source rule，不只写模块说明。每个 API 至少要包含入口函数、核心路径、内部事件、输出字段、错误分支。
2. 对 memory、stream、command、graph、sync、ioctl 定义稳定事件 ID 和 payload，不使用临时日志文本作为建模输入。
3. 在 HAL/M3D 到 KMD 的边界补齐事件。没有 ioctl 边界事件时，模型无法区分 Driver 内部成本和 KMD 等待成本。
4. 对 command merge、async capacity wait、inflight wait、dependency edge 给出事件签名。否则只能看到 submit 结果，不能解释为什么 submit 延迟。

## `MUSA-Runtime`

已有技术洞察覆盖了公共 API、自动生成 wrapper、`ExportTableManager`、`ApiInvocationGuard`、`ApiTrace`、MUPTI Runtime hooks。它适合作为 Runtime 层白盒建模入口，但不应承担 Driver 内部建模。

Runtime 层应输出以下信息：

| 类别 | 事件 |
|---|---|
| API 边界 | Runtime API enter/exit、status、thread、correlation id |
| Driver 分发 | Runtime API 调用哪个 Driver API、分发耗时 |
| 初始化 | `dlopen`、`dlsym`、export table 初始化、`muInit` |
| module/function | fatbin register、module load、function cache hit/miss |
| graph/runtime wrapper | graph launch、stream capture、memcpy wrapper 参数摘要 |

需要注意：

1. 不要逐个手写修改所有 Runtime wrapper。应优先复用自动生成 wrapper、Tools callback 和 `ApiTrace`。
2. Runtime 成本和 Driver 成本要拆开。`musaLaunchKernel` 的 host time 不能直接等于 Driver launch 成本。
3. Runtime 初始化成本要单独建模。首次 API 的 `dlopen`、export table、device init 会污染 steady-state 性能。
4. 对模型报告而言，Runtime 层主要回答“用户 API 调用边界”和“进入 Driver 前发生了什么”。

## `musa_benchmarks`

已有技术洞察准确说明了 benchmark 框架、Celero 三层循环、UDM 指标、memory/schedule/mulStreams/graph/resource 等 suite。它不只是性能对比工具，应纳入 OKR 的校准和验收链路。

建议把现有用例映射为模型校准项：

| 用例方向 | 现有或应补充 case | 校准内容 |
|---|---|---|
| kernel launch | `kernelLaunchLatencyMode`、`kernelLaunchThroughputMode`、`kernelLaunchThroughputModeWithoutMerge`、`kernelLaunchApiLatency` | Runtime/Driver launch、command build、merge、submit |
| memory | `MallocRateFrom1Bto4GB`、`Copy1DAlignedRate`、`CopyPinnedRate`、`HostRegisterAndUnRegister` | pool hit/miss、alloc/free、H2D/D2H/D2D bandwidth |
| stream/event/sync | `efficiencyOfSync`、`streamConcurrencyCompute`、`streamConcurrencyMemcpy`、`parallelismOfDifferentCommands` | wait reason、stream overlap、event dependency |
| graph | `efficiencyOfGraph`、`efficiencyOfGraphLaunch`、`graphLaunchThroughputMode` | graph instantiate、launch、node relation |
| multi-card | `CopyP2PRate`、`kernelLaunchMulCards` | peer copy、multi-device context 和 stream |
| trace overhead | 新增 `traceOffOnOverhead` | 插桩关闭、开启 level 1、开启 level 2 的性能影响 |
| event signature | 新增 `modelEventSignature` | 断言关键事件序列是否完整 |

当前 OKR 方案缺少 `musa_benchmarks` 的明确交付物。建议新增：

```text
model_calibration_benchmarks/
  launch/
  memory/
  stream_event/
  graph/
  overhead/

输出：
  benchmark_results.csv
  model_events.jsonl
  calibration_features.parquet
  overhead_report.md
  event_signature_report.md
```

## `MUPTI`

已有技术洞察已经确认 MUPTI 的核心结构：Public API、Injection、Tools callback、Driver/Runtime hook、ActivityBuffer、MT-Perf、relation map。它适合作为 OKR 的统一采集层。

MUPTI 当前能直接提供：

| 能力 | 数据 |
|---|---|
| Runtime API | Runtime callback/activity |
| Driver API | Driver callback/activity |
| kernel/memcpy/memset | activity record、begin/end、stream、correlation |
| graph | graph trace、graph node activity |
| sync | stream/event/context synchronize activity |
| relation | correlation、submission、kick、kernel |

MUPTI 当前不能直接提供：

```text
MemoryPoolLookup
MemoryPoolHit / MemoryPoolMiss
ChunkAllocateBegin / End
BoAllocIoctlBegin / End
ResolveDependencyBegin / End
QueueCommandBegin / End
CommandBuildBegin / End
MergeCheck / MergeFlush
SubmitBegin / End
InflightWaitBegin / End
GraphRebuildDecision
WaitReason
```

建议新增统一 `ModelEvent` 通路。优先方案是 private hook，稳定后再升级为正式 activity kind 或 internal domain。

MUPTI 侧必须新增的工程项：

1. `EmitModelEvent` hook 或等价 internal event hook。
2. `MODEL_MEMORY`、`MODEL_STREAM`、`MODEL_COMMAND`、`MODEL_GRAPH`、`MODEL_SYNC`、`MODEL_IOCTL` domain。
3. collector，输出 `events.jsonl`、`relations.parquet`、`model_features.parquet`。
4. activity kind 支持矩阵，明确哪些 kind 能 enable，哪些会真实产出 record。
5. buffer 压力测试，覆盖 buffer 满、forced flush、未完成 record、drop count。
6. 单 subscriber 处理策略。短期由 OKR collector 独占 MUPTI；中期由 collector 合并 profiler 能力。

## 当前 OKR 方案复审

## 正确部分

1. 方案已经明确不把 Driver 当黑盒。
2. 方案主线是源码埋点、MUPTI 采集、白盒建模、profiling 对齐，方向正确。
3. 事件体系覆盖 API、internal span、state transition、counter、relation、sync、ioctl，基本完整。
4. 低开销原则正确：默认关闭、固定 payload、TLS buffer、离线解析、按 domain 启用。
5. 最小闭环选择 `muLaunchKernel`、`muMemAlloc`、`muStreamSynchronize` 是合理的。

## 主要问题

| 问题 | 影响 | 修正 |
|---|---|---|
| OKR 主方案只把源码范围写成 `musa`、`MUSA-Runtime` | `MUPTI` 和 `musa_benchmarks` 没有成为交付仓库 | 把四个仓库全部纳入交付分工 |
| `musa_benchmarks` 未进入校准和验收链路 | 模型缺少可重复 lower-bound、overhead 和 CTS 验证 | 增加 calibration benchmark、overhead benchmark、event signature CTS |
| KR2 只看 Top50 kernel overlap | 只能证明 kernel 列表基本对齐，不能证明 Driver/API 软件模型正确 | 增加 API-source rule coverage、relation recall、event loss、成本分项误差 |
| Top90 Driver API 的统计口径不清楚 | host self time、sync wait time、Runtime 包裹时间可能混在一起 | 拆成 `api_self_time`、`sync_wait_time`、`driver_internal_time` |
| 早期 replay 文档仍存在 | 容易把 OKR 拉回黑盒 trace replay | 明确 replay 只作为离线校验工具，不作为主模型 |
| 内部事件 ABI 未定 | 影响 MUPTI、Driver、Runtime 联调节奏 | 先走 private hook，后续再转正式 domain |
| 单 subscriber 风险未形成执行策略 | 与 torch profiler 或其他 profiler 冲突 | OKR collector 先独占，统一采集 API、activity、internal event |

## 修订后的 OKR 实施主线

```text
3 个模型 profiling
  -> 统计 Top Driver API、Top kernel、sync、graph、memory 热点
  -> 为 Top90 Driver API 建 source rule
  -> 在 MUSA-Runtime / Driver / HAL/M3D 补齐 ModelEvent
  -> MUPTI collector 采集 API、activity、internal event、relation
  -> musa_benchmarks 校准成本项和验证插桩开销
  -> 白盒模型重建状态和成本
  -> 输出模型报告和 CTS 行为用例
```

模型输出不能只给总耗时，必须给出分项：

```text
T_api =
  T_runtime_wrapper
+ T_driver_wrapper
+ T_validation
+ T_context_lookup
+ T_memory
+ T_stream_queue
+ T_command_build
+ T_submit
+ T_sync_wait
+ T_ioctl
```

## 四仓库交付清单

| 仓库 | 必须交付 |
|---|---|
| `MUSA-Runtime` | Runtime API callback 覆盖报告、Runtime init/cache ModelEvent、Runtime-to-Driver relation、Runtime overhead 验证 |
| `linux-ddk/musa` | Top90 API source rule、memory/stream/command/graph/sync/ioctl ModelEvent、HAL/M3D submit/ioctl 事件、source-level 瓶颈归因 |
| `MUPTI` | ModelEvent hook、collector、schema、activity 支持矩阵、buffer 压力测试、单 subscriber 策略 |
| `musa_benchmarks` | calibration suite、trace overhead suite、event signature CTS、跨 SDK baseline、benchmark 与 trace 对齐报告 |

## 推荐里程碑

| 阶段 | 范围 | 验收 |
|---|---|---|
| M0 | 选定 3 个模型，采集现有 profiler/MUPTI 数据，统计 Top API/Top kernel | 产出 `top_api.csv`、`top_kernel.csv`、`trace_gap_report.md` |
| M1 | 打通 `EmitModelEvent` private hook，覆盖 `muLaunchKernel`、`muMemAlloc`、`muStreamSynchronize` | 能重建 API -> command -> submission -> kernel；能区分 pool hit/miss 和 stream wait |
| M2 | 扩展到 Top90 Driver API，补齐 memory、stream、command、graph、sync、ioctl 事件 | Top90 API 都有 source rule、事件签名和分项成本 |
| M3 | 接入 `musa_benchmarks` 做校准和 overhead 验收 | trace off 影响在噪声内；trace on level 1 端到端开销可量化 |
| M4 | 生成 3 个模型报告，沉淀 CTS 行为用例 | 报告能定位源码文件、函数、状态变量和等待原因 |

## 建议验收指标

| 类别 | 指标 |
|---|---|
| API 覆盖 | 累计耗时 Top90% Driver API 均有 source rule、事件签名、成本分项 |
| relation 完整性 | API -> command -> submission -> kernel 关系重建召回率不低于 95% |
| kernel 对齐 | Top50 kernel 按累计耗时排序的 Top-K overlap 不低于 90% |
| 事件质量 | internal event drop rate 必须输出；超过阈值时报告无效 |
| 插桩开销 | trace off 影响 <= 0.1% 或在 benchmark 噪声内；trace on level 1 端到端开销建议 <= 1%-3% |
| 成本分项 | API self、memory、command build、submit、sync wait、ioctl 分项必须可解释 |
| CTS | memory pool、stream/event、graph、sync、command merge、KV cache、attention/MLP、expert routing 至少形成可复现行为用例 |

## 有疑问的点

1. 3 个主流模型尚未明确。建议至少包含一个 dense decoder 推理、一个 MoE 推理、一个训练或长上下文场景。
2. Top90 Driver API 的耗时口径需要确认。建议同时输出 `host_self_time`、`inclusive_time`、`sync_wait_time`，KR 主口径使用可解释的 `inclusive_time`，报告中拆分等待时间。
3. `ModelEvent` 是进入正式 MUPTI public ABI，还是先做 private hook。建议先 private hook。
4. CTS 最终沉淀到 `musa_benchmarks`、`musa_samples` 还是独立 MUSA CTS 仓库需要确认。建议第一阶段先放在 `musa_benchmarks`，稳定后迁移。
5. 单 subscriber 与现有 profiler 的冲突需要确认。建议 OKR collector 统一采集，不和 torch profiler 同进程并行订阅。

## 对当前文档的处理建议

1. 保留 `insights/okr/okr-2026-landing-plan.md`，但补充四仓库分工、`musa_benchmarks` 校准链路、MUPTI repo 交付项。
2. 保留 `insights/okr/MUPTI埋点性能建模技术方案.md`，继续作为 ModelEvent 和 collector 设计依据。
3. 将 `insights/simulation/okr/musa_driver_api_perf_model_landing_plan.md` 标注为早期 replay 方案。后续只复用 trace schema、linker、validator，不再作为 OKR 主路径。
4. 每个 Top90 API 增加独立 source rule 文件或表格，格式固定为：API、源码路径、状态转移、事件签名、payload、成本项、验证用例。
