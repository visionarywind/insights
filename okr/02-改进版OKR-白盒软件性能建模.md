# MUSA Driver/API 白盒软件性能建模 OKR

## Objective 1：MUSA Driver/API 白盒软件性能建模

基于 3 个主流模型推理/训练场景，建立 MUSA Driver/API 白盒软件性能模型。

模型必须基于 `MUSA-Runtime`、`linux-ddk/musa`、`MUPTI` 的源码规则、ModelEvent、activity 和 relation 数据，输出 API 成本分项、源码瓶颈归因和优化建议。

### KR1：完成 3 个 workload 的基线采集和 Top90 Driver API 定义

验收标准：

| 项目 | 标准 |
| --- | --- |
| workload | 固定 3 个模型场景，至少包含 dense decoder 推理、MoE 推理、训练或长上下文场景 |
| 元数据 | 记录 SDK version、driver commit、runtime commit、MUPTI commit、device、运行参数 |
| API 基线 | 输出 Runtime API、Driver API 的 `inclusive_time`、`host_self_time`、`sync_wait_time` |
| kernel 基线 | 输出 Top kernel、累计耗时、stream、correlation 信息 |
| Top90 API | 固定累计耗时 Top90% Driver API 清单 |

交付物：

```text
workload_matrix.yaml
run_metadata.yaml
top_runtime_api.csv
top_driver_api.csv
top_kernel.csv
trace_gap_report.md
```

### KR2：完成 Top90 Driver API 的 source rule、ModelEvent 和成本分项覆盖

验收标准：

| 项目 | 标准 |
| --- | --- |
| source rule | Top90 Driver API 均有独立 source rule |
| 源码路径 | 每个 source rule 包含入口、核心路径、状态变量、异常分支 |
| 事件签名 | 每个 source rule 绑定 ModelEvent domain、event id、payload |
| 成本分项 | 每个 source rule 输出适用的 Runtime、Driver、Memory、Stream、Command、Submit、Sync、IOCTL 成本项 |
| 验证用例 | 每类核心路径至少有一个 benchmark 或最小用例验证 |

交付物：

```text
source_rules/
20-ModelEvent-Schema.md
top90_api_coverage_report.md
api_cost_terms_matrix.csv
```

### KR3：完成 MUPTI ModelEvent collector、relation builder 和 cost model v1

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
cost_model_v1/
model_profile.yaml
api_cost_breakdown.parquet
```

### KR4：完成 profiling 校准和 3 个模型白盒性能建模报告

验收标准：

| 项目 | 标准 |
| --- | --- |
| 成本模型 | 每个 API 输出分项耗时，不只输出总耗时 |
| 可解释覆盖 | Top90 API 主要耗时归因覆盖率不低于 85% |
| kernel 对齐 | Top50 kernel 按累计耗时排序的 Top-K overlap 不低于 90% |
| API 误差 | Top20 Driver API host inclusive time p50 误差建议不高于 20%，超出项必须归因 |
| 报告内容 | 包含 API 热点、kernel 热点、成本分项、源码归因、事件质量、插桩开销和优化建议 |

交付物：

```text
kernel_overlap_report.md
model_validation_report.md
model_reports/
overhead_report.md
```

## Objective 2：MUSA 行为分析与 CTS 沉淀

基于 3 个模型抽象可复现的软件行为，把 memory、stream/event、graph、sync、attention、MLP、KV cache、expert routing 等行为沉淀为带事件签名的 CTS 或 benchmark 用例。

### KR1：抽象不少于 5 类模型行为特征

建议行为特征：

| 行为 | 覆盖内容 |
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

交付物：

```text
behavior_baseline/
sdk_diff_report.md
```

## 关键验收指标

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

## 阶段计划

| 阶段 | 范围 | 验收 |
| --- | --- | --- |
| M0 | 固定 workload，采集 API/kernel 基线，生成 Top90 Driver API | 明确本季度要建模的 API、kernel、行为 |
| M1 | 覆盖 `muLaunchKernel`、`muMemAlloc/muMemFree`、`muStreamSynchronize` 最小闭环 | 能重建 API -> command -> submission -> kernel，并输出分项耗时 |
| M2 | 扩展 Top90 API source rule 和 ModelEvent | Top90 API 均有源码规则、事件签名和成本分项 |
| M3 | 完成 cost model v1、relation recall、kernel overlap、overhead 验证 | relation recall >=95%，Top50 overlap >=90%，可解释覆盖 >=85% |
| M4 | 发布 3 个模型报告和行为 CTS | 报告能定位源码模块、函数、状态变量和等待原因 |
