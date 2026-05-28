# 软件性能建模 OKR 评估与改进方案

## 1. 评估结论

当前 OKR 的方向正确：围绕 3 个模型建立 MUSA driver/API 性能建模能力，并把模型行为沉淀到 CTS。主要问题不在目标方向，而在 KR 的工程定义不够完整。

当前 OKR 更偏结果描述，缺少以下硬性验收：

1. 什么是“软件性能模型”。
2. 哪些源码规则必须覆盖。
3. 哪些埋点事件必须采集。
4. 如何证明 API、command、submission、kernel 的关系正确。
5. 如何证明模型不是黑盒 trace replay。
6. 如何控制插桩开销。
7. 如何把模型能力沉淀为可回归工具链。

建议把当前 OKR 从“profiling 对齐型目标”改为“白盒软件建模型目标”。

```text
当前表达：
  建模结果与 profiling 数据对齐

建议表达：
  基于源码规则和 MUPTI ModelEvent，建立可解释、可校准、可回归的 MUSA Driver/API 白盒软件性能模型。
```

## 2. 当前 OKR 逐项评估

### 2.1 性能建模 OKR

原始目标：

```text
对 3 个主流模型推理/训练进行 MUSA driver/API 性能建模，
建立可与 profiling 数据对齐的性能预测与瓶颈分析能力。
```

评估：

| 维度 | 结论 | 问题 |
| --- | --- | --- |
| 方向 | 正确 | 覆盖真实模型、Driver/API、profiling 对齐，方向成立 |
| 建模定义 | 不完整 | 没有定义模型输入、状态、成本项、输出格式 |
| 白盒程度 | 不足 | 原始 OKR 未明确 source rule、ModelEvent、内部状态机 |
| 验收标准 | 偏弱 | Top50 kernel overlap 不能证明 Driver/API 软件模型正确 |
| 工程交付 | 不完整 | 缺少 collector、schema、relation builder、cost model、overhead suite |
| 风险控制 | 不完整 | 缺少 dropped record、buffer overflow、trace overhead、single subscriber 策略 |

### 2.2 KR1 评估

原始 KR：

```text
基于 3 个模型推理/训练建立性能模型，
覆盖 trace 中累积耗时 Top90% 的 driver API。
```

问题：

1. “建立性能模型”没有可验收定义。
2. “覆盖 Top90% driver API”没有说明覆盖什么。
3. 没有区分 `inclusive_time`、`host_self_time`、`sync_wait_time`。
4. 没有要求每个 API 具备源码路径、状态转移、事件签名和成本分项。
5. 没有要求模型输出分项解释。

建议改为：

```text
KR1：完成 3 个模型的 Driver/API 白盒建模覆盖。
验收：累计耗时 Top90% Driver API 均具备 source rule、ModelEvent 事件签名、关系重建和成本分项输出。
```

### 2.3 KR2 评估

原始 KR：

```text
建模结果与 profiling 数据对齐：
Top50 kernel 按累计耗时排序的 Top-K overlap 达到 90% 以上。
```

问题：

1. Top50 kernel overlap 只能证明 kernel 列表基本一致。
2. 它不能证明 API 到 kernel 的关系正确。
3. 它不能证明 Driver 内部成本建模正确。
4. 它不能证明 memory、stream、graph、sync 等软件路径建模正确。

建议保留该指标，但降级为校准指标之一，并新增以下指标：

| 指标 | 建议验收 |
| --- | --- |
| relation recall | API -> command -> submission -> kernel 关系召回率不低于 95% |
| event quality | 标准 workload 下 dropped record 为 0；压力场景必须输出 dropped count |
| cost explainability | Top90 API 的主要耗时能归入明确成本项，覆盖率不低于 85% |
| API cost error | Top20 Driver API 的 host inclusive time p50 误差建议不高于 20%，p90 误差必须可解释 |
| Top50 kernel overlap | Top-K overlap 不低于 90%，作为设备侧排序校准 |

### 2.4 KR3 评估

原始 KR：

```text
针对一个 SDK 版本发布 3 个模型的性能建模分析报告，
包含 API 热点、kernel 热点和优化建议。
```

问题：

1. 报告内容偏少。
2. 没有要求源码归因。
3. 没有要求事件质量和插桩开销。
4. 没有要求模型 profile、source rule、数据集可复现。

建议改为：

```text
KR3：发布一个 SDK 版本的 3 个模型白盒性能建模报告。
报告必须包含 API 热点、kernel 热点、关系质量、成本分项、源码归因、事件质量、插桩开销和优化建议。
```

### 2.5 行为分析与 CTS OKR 评估

原始目标正确，但和性能建模没有完全打通。

问题：

| 问题 | 影响 |
| --- | --- |
| 行为用例与 ModelEvent 事件签名未绑定 | CTS 只能验证 API 返回值，不能验证软件行为路径 |
| trace 中关键 API 覆盖率没有明确 source rule 关系 | 行为分析和性能建模会重复建设 |
| 版本差异报告缺少归因字段 | 无法定位差异来自 Runtime、Driver、memory、stream、graph 还是 submit |

建议：

1. CTS 不沉淀原始模型 trace，沉淀可复现软件行为。
2. 每个 CTS 用例必须定义事件签名。
3. 行为基线必须复用性能建模的 source rule、ModelEvent 和 relation 表。

## 3. 改进后的 OKR 建议

### Objective 1：建立 MUSA Driver/API 白盒软件性能建模能力

基于 `MUSA-Runtime`、`linux-ddk/musa`、`MUPTI` 和 `musa_benchmarks`，建立可解释、可校准、可回归的 MUSA Driver/API 白盒软件性能模型。

### KR1：完成 3 个模型的建模基线和 Top90 API 定义

验收标准：

| 项目 | 标准 |
| --- | --- |
| 模型选择 | 固定 3 个 workload，至少包含 dense decoder 推理、MoE 推理、训练或长上下文场景 |
| 基线数据 | 每个 workload 输出 Runtime API、Driver API、kernel、memcpy、sync、graph 基线 |
| Top API | 同时输出 `inclusive_time`、`host_self_time`、`sync_wait_time` 三个口径 |
| Top90 列表 | 固定累计耗时 Top90% Driver API 列表 |
| 元数据 | 记录 SDK version、driver commit、runtime commit、MUPTI commit、device、运行参数 |

交付物：

```text
workload_matrix.yaml
run_metadata.yaml
top_runtime_api.csv
top_driver_api.csv
top_kernel.csv
trace_gap_report.md
```

### KR2：完成 Top90 Driver API 的白盒建模覆盖

验收标准：

| 项目 | 标准 |
| --- | --- |
| source rule | Top90 Driver API 均有 source rule |
| 源码路径 | 每个 source rule 包含入口、核心路径、状态变量、异常分支 |
| 事件签名 | 每个 source rule 绑定 ModelEvent domain、event id、payload |
| 成本分项 | 每个 source rule 输出 Runtime、Driver、Memory、Stream、Command、Submit、Sync、IOCTL 中的适用项 |
| 最小验证 | 每类核心路径至少有一个 benchmark 或最小用例验证 |

source rule 格式：

```text
api:
entry:
source_path:
state_variables:
state_transitions:
required_events:
relations:
cost_terms:
error_paths:
validation_case:
```

交付物：

```text
source_rules/
model_event_schema.md
top90_api_coverage_report.md
api_cost_terms_matrix.csv
```

### KR3：打通 MUPTI ModelEvent collector 和关系重建

验收标准：

| 项目 | 标准 |
| --- | --- |
| 采集通路 | Runtime API、Driver API、activity、ModelEvent 均能统一采集 |
| 事件类型 | ModelEvent 支持 `span`、`instant`、`counter`、`relation` |
| ID 体系 | 支持 `correlation_id`、`span_id`、`parent_id`、`command_id`、`submission_id`、`activity_id` |
| relation recall | API -> command -> submission -> kernel 关系召回率不低于 95% |
| event quality | 输出 dropped record、buffer overflow、flush error |
| timeline | 输出 Perfetto 兼容 trace 或等价 timeline |

交付物：

```text
mupti_model_collector
api_events.parquet
model_events.parquet
activity_events.parquet
relations.parquet
trace.perfetto
relation_recall_report.md
event_quality_report.md
```

### KR4：完成白盒成本模型 v1 和 profiling 校准

验收标准：

| 项目 | 标准 |
| --- | --- |
| 成本模型 | 输出每个 API 的分项耗时，不只输出总耗时 |
| 可解释覆盖 | Top90 API 主要耗时归因覆盖率不低于 85% |
| kernel 对齐 | Top50 kernel 按累计耗时排序的 Top-K overlap 不低于 90% |
| API 误差 | Top20 Driver API host inclusive time p50 误差建议不高于 20%，超出项必须归因 |
| profile | 输出按 SDK 和设备绑定的模型 profile |

交付物：

```text
cost_model_v1/
model_profile.yaml
api_cost_breakdown.parquet
kernel_overlap_report.md
model_validation_report.md
```

### KR5：发布 3 个模型的白盒性能建模报告

验收标准：

每个模型报告必须包含：

| 内容 | 要求 |
| --- | --- |
| API 热点 | Runtime API、Driver API，含 self、inclusive、sync wait |
| kernel 热点 | Top50 kernel、累计耗时、Top-K overlap |
| 关系质量 | relation recall、未关联 API/kernel 清单 |
| 成本分项 | Runtime、Driver、Memory、Stream、Command、Submit、Sync、IOCTL |
| 源码归因 | 文件、函数、状态变量、source rule |
| 事件质量 | dropped record、flush error、buffer overflow |
| 插桩开销 | trace off、API-only、internal targeted |
| 优化建议 | 明确到模块和路径 |

### Objective 2：建立 MUSA 行为分析与 CTS 回归能力

基于真实模型抽象可复现的软件行为，把行为路径沉淀为 CTS 或 benchmark 用例，支撑 SDK 版本回归。

### KR1：抽象不少于 5 类模型行为特征

建议行为特征：

| 行为 | 说明 |
| --- | --- |
| attention launch pattern | QKV、attention、projection 的 kernel/API 模式 |
| MLP launch pattern | gate/up/down projection 的 kernel/API 模式 |
| KV cache allocation pattern | cache 分配、复用、增长、释放 |
| expert routing pattern | routing、topk、dispatch、combine、小 kernel 密集调用 |
| stream/event sync pattern | stream wait、event record、sync wait |
| graph launch pattern | graph instantiate、update、launch、node relation |
| memory pool behavior | pool hit/miss、grow、trim、free merge |

每类行为必须定义：

```text
trigger
API sequence
required ModelEvent
required relation
expected counters
pass/fail rule
```

### KR2：建立行为 CTS 和事件签名验收

验收标准：

| 项目 | 标准 |
| --- | --- |
| 用例覆盖 | memory、stream/event、graph、sync、command merge 至少各有一个用例 |
| trace 覆盖 | 行为用例覆盖 trace 中关键 memory、stream/event、graph API 的 90% 以上 |
| 事件签名 | 每个用例校验事件序列和关键字段 |
| 回归输出 | 输出 pass/fail、事件缺失、耗时变化、状态变化 |

交付物：

```text
behavior_cts/
event_signature_rules/
behavior_coverage_report.md
```

### KR3：发布 SDK 版本行为基线和版本差异报告

验收标准：

| 项目 | 标准 |
| --- | --- |
| 行为基线 | 3 个模型均有行为基线 |
| 版本差异 | 能说明 API、kernel、memory、stream、graph、sync、submit 的变化 |
| 源码归因 | 差异能关联 source rule、事件签名和源码模块 |
| 回归判断 | 能区分预期变化、性能回退、事件缺失、模型不可解释 |

## 4. 当前 OKR 到改进 OKR 的映射

| 原始内容 | 保留方式 | 改进点 |
| --- | --- | --- |
| 3 个模型推理/训练 | 保留 | 固定 workload matrix 和运行元数据 |
| 覆盖 Top90 Driver API | 保留 | 改为 source rule、事件签名、成本分项三重覆盖 |
| Top50 kernel overlap 90% | 保留 | 降级为校准指标，新增 relation recall 和成本分项指标 |
| 3 个模型分析报告 | 保留 | 增加源码归因、事件质量、插桩开销、关系质量 |
| dense/MoE 行为用例 | 保留 | 绑定 ModelEvent 事件签名 |
| memory/stream/event/graph API 覆盖 | 保留 | 统一纳入 source rule 和 CTS coverage |
| SDK 版本行为基线 | 保留 | 增加版本差异源码归因 |

## 5. 工程落地路径

### M0：定义 workload 和基线

交付：

1. 3 个 workload。
2. profile 采集脚本。
3. `top_api.csv`、`top_kernel.csv`。
4. Top90 Driver API 列表。
5. trace gap report。

验收：

```text
能够明确本季度要建模哪些 API、哪些 kernel、哪些行为。
```

### M1：最小白盒闭环

覆盖：

```text
muLaunchKernel
muMemAlloc / muMemFree
muStreamSynchronize
```

交付：

1. 3 个 API 的 source rule。
2. ModelEvent private hook MVP。
3. collector MVP。
4. relation builder MVP。
5. `api_cost_breakdown` MVP。

验收：

```text
能重建 API -> command -> submission -> kernel。
能输出 launch、alloc、sync 的分项耗时。
```

### M2：Top90 API 覆盖

交付：

1. Top90 API source rule。
2. memory、stream、command、graph、sync、HAL/M3D ModelEvent。
3. event schema v1。
4. coverage report。

验收：

```text
Top90 API 均有源码规则、事件签名和成本分项。
```

### M3：模型 v1 和校准

交付：

1. cost model v1。
2. relation recall report。
3. kernel overlap report。
4. overhead report。
5. model validation report。

验收：

```text
Top50 kernel overlap >= 90%。
API -> command -> submission -> kernel relation recall >= 95%。
Top90 API 主要耗时可解释覆盖率 >= 85%。
```

### M4：报告和 CTS

交付：

1. 3 个模型性能建模报告。
2. 行为 CTS 用例。
3. SDK 行为基线。
4. 版本差异报告。

验收：

```text
报告能定位源码模块、函数、状态变量和等待原因。
CTS 能复现关键软件行为并校验事件签名。
```

## 6. 必须补齐的标准件

| 标准件 | 用途 |
| --- | --- |
| `workload_matrix.yaml` | 固定 3 个模型场景和运行参数 |
| `run_metadata.yaml` | 保证结果可复现 |
| `model_event_schema.md` | 固定事件 domain、event id、payload |
| `source_rules/` | 固定每个 API 的源码规则 |
| `mupti_model_collector` | 统一采集 API、activity、ModelEvent |
| `relation_builder` | 重建 API、command、submission、kernel 关系 |
| `cost_model_v1` | 输出分项耗时 |
| `model_profile.yaml` | 保存 SDK/设备成本参数 |
| `overhead_suite` | 验证插桩开销 |
| `event_signature_cts` | 验证软件行为路径 |
| `report_generator` | 自动生成模型报告 |

## 7. 硬性验收指标

| 类别 | 指标 |
| --- | --- |
| API 覆盖 | 累计耗时 Top90% Driver API 均有 source rule、事件签名、成本分项 |
| 关系完整性 | API -> command -> submission -> kernel relation recall 不低于 95% |
| kernel 对齐 | Top50 kernel 按累计耗时排序 Top-K overlap 不低于 90% |
| 成本解释 | Top90 API 主要耗时可解释覆盖率不低于 85% |
| API 误差 | Top20 Driver API host inclusive time p50 误差建议不高于 20%，超出项必须归因 |
| 事件质量 | dropped record、buffer overflow、flush error 必须输出 |
| 插桩关闭开销 | trace off 开销不高于 0.1% 或落在 benchmark 噪声内 |
| 插桩开启开销 | API-only 建议不高于 1%；internal targeted 建议不高于 3% |
| 报告质量 | 必须包含源码归因、关系质量、事件质量、插桩开销 |
| CTS | 行为用例必须校验事件签名，不只校验 API 返回值 |

## 8. 当前需要立即修改的点

1. 将 OKR 标题从“性能建模（模拟）”改为“白盒软件性能建模”。
2. 将 Top50 kernel overlap 从核心验收改为校准指标之一。
3. 在 KR1 中明确 Top90 API 的覆盖对象是 source rule、ModelEvent、relation、cost breakdown。
4. 在 KR2 中新增 relation recall、event quality、cost explainability、overhead。
5. 在 KR3 中明确报告必须包含源码归因、事件质量、插桩开销。
6. 将 MUPTI、musa_benchmarks 明确纳入 OKR 交付仓库。
7. 将 replay 方案标注为验证工具，不作为主模型。
8. 为每个 Top90 API 建 source rule 文件。
9. 将 ModelEvent schema 固化为 `span`、`instant`、`counter`、`relation` 四类。
10. 将 Perfetto timeline 输出纳入工程化交付。

## 9. 改进后的 OKR 文案建议

```text
Objective 1：MUSA Driver/API 白盒软件性能建模

基于 3 个主流模型推理/训练场景，建立 MUSA Driver/API 白盒软件性能模型。
模型必须基于 Runtime/Driver 源码规则、MUPTI ModelEvent、activity 和 relation 数据，
输出 API 成本分项、源码瓶颈归因和配置/优化建议。

KR1：完成 3 个 workload 的基线采集和 Top90 Driver API 定义。
交付 top_api、top_kernel、run_metadata、trace_gap_report。

KR2：完成 Top90 Driver API 的 source rule、ModelEvent 事件签名和成本分项覆盖。
Top90 API 均能输出 Runtime、Driver、Memory、Stream、Command、Submit、Sync、IOCTL 中适用的成本项。

KR3：完成 MUPTI ModelEvent collector、relation builder 和 cost model v1。
API -> command -> submission -> kernel relation recall 不低于 95%，Top50 kernel overlap 不低于 90%。

KR4：发布 3 个模型的白盒性能建模报告。
报告包含 API 热点、kernel 热点、成本分项、源码归因、事件质量、插桩开销和优化建议。

Objective 2：MUSA 行为分析与 CTS 沉淀

基于 3 个模型抽象可复现的软件行为，把 memory、stream/event、graph、sync、
attention、MLP、KV cache、expert routing 等行为沉淀为带事件签名的 CTS。

KR1：抽象不少于 5 类模型行为特征，并定义 API 序列、ModelEvent 事件签名和 pass/fail 规则。

KR2：建立行为 CTS，用例覆盖 trace 中关键 memory、stream/event、graph API 的 90% 以上。

KR3：发布一个 SDK 版本的 3 个模型行为基线和版本差异报告，
差异必须能归因到 source rule、事件签名和源码模块。
```

## 10. 最终建议

当前 OKR 可以保留目标方向，但需要重写验收口径。

真正的软件性能建模不是把 profiling 数据拟合成一条曲线，也不是只对齐 kernel 排名。它应把 Runtime/Driver 源码路径转化为可执行规则，并用 MUPTI 事件验证这些规则。

建议以以下顺序推进：

1. 先固定 3 个 workload 和 Top90 Driver API。
2. 先完成 `muLaunchKernel`、`muMemAlloc`、`muStreamSynchronize` 最小闭环。
3. 再扩展到 Top90 API。
4. 最后发布 3 个模型报告和 CTS 行为基线。

若不先完成 source rule、ModelEvent、collector、relation builder，当前 OKR 容易退化为普通 profiling 报告，无法体现软件性能建模价值。
