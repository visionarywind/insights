# OKR 方案阅读索引

这个目录用于沉淀 MUSA Driver/API 白盒软件性能建模与行为 CTS 的 OKR、技术方案、分阶段落地文档和工程标准件。

本索引中的编号含义：

```text
00             阅读入口
01-08          当前主线文档
20-22          工程 schema / 输出契约
90-94          历史参考文档，不作为当前实施入口
source_rules/  API 级 source rule
```

当前执行主线是：

```text
原始 OKR
  -> 改进版 OKR
  -> 五仓库技术洞察与落地复审
  -> MUPTI / ModelEvent 技术方案
  -> M1 最小闭环
  -> musa 核心对象与 API 执行流程全景
  -> M2 Top90 API 扩展
  -> M3 白盒模型 v1
  -> M4 行为 CTS / SDK 回归
```

## 1. 推荐阅读顺序

### 1.1 先看目标

1. [01-原始OKR.md](01-原始OKR.md)
   - 原始 OKR 输入。
   - 用于理解最初目标：Driver/API 性能建模与行为分析 / CTS 沉淀。

2. [02-改进版OKR-白盒软件性能建模.md](02-改进版OKR-白盒软件性能建模.md)
   - 改进后的正式 OKR 文案。
   - 把目标拆成 workload 基线、Top90 API、ModelEvent、relation builder、cost model、profiling 校准和报告交付。

### 1.2 再看主线落地方案

3. [03-五仓库技术洞察与落地复审.md](03-五仓库技术洞察与落地复审.md)
   - 当前最重要的总体技术路线文档。
   - 覆盖 `linux-ddk/musa`、`MUSA-Runtime`、`MUPTI`、`msight-compute`、`msight-system` 五个仓库的分工。
   - 给出 M0/M1/M2/M3/M4 分阶段交付路径。

4. [04-MUPTI-ModelEvent技术方案.md](04-MUPTI-ModelEvent技术方案.md)
   - ModelEvent collector、MUPTI 复用边界、低开销控制、关系重建、成本分解的核心技术方案。
   - 用于指导 MUPTI private ModelEvent collector 的实现。

### 1.3 最后看分阶段工程细化

5. [05-M1-API执行流程与ModelEvent插点.md](05-M1-API执行流程与ModelEvent插点.md)
   - M1 四个 API 的端到端执行流程和 ModelEvent 插点分析。
   - 覆盖 `muLaunchKernel`、`muMemAlloc_v2`、`muMemFree_v2`、`muStreamSynchronize`。

6. [06-M1-最小闭环实施清单.md](06-M1-最小闭环实施清单.md)
   - M1 工程实施 checklist。
   - 用于确认最小闭环是否具备 source rule、collector 输出、relation builder、cost breakdown 和 overhead 验收。

7. [07-M2-Top90API执行流程与ModelEvent插点.md](07-M2-Top90API执行流程与ModelEvent插点.md)
   - M2 Top90 API 扩展的端到端代码路径和插点分析。
   - 覆盖 launch/module/function、memory/pool、copy/memset、stream/event/sync、graph、context/device/resource/Green Context、Core/HAL/M3D/OS submit boundary。

8. [08-musa核心对象与API执行流程全景.md](08-musa核心对象与API执行流程全景.md)
   - 基于 `linux-ddk/musa` 源码梳理 Driver/Core/HAL/M3D 层次、核心对象职责和 API family 执行流程。
   - 用 ASCII flowchart 和 sequence diagram 展示 API 如何通过核心对象完成 launch、memory、copy、stream/event、graph、GreenContext 和 submit。

## 2. 当前主线文档

以下文档是当前方案入口。实施时以这一组为准，`90-94` 参考文档只用于追溯设计来源。

| 类型 | 文档 | 作用 |
| --- | --- | --- |
| 原始目标 | [01-原始OKR.md](01-原始OKR.md) | 原始 OKR 输入 |
| 正式 OKR | [02-改进版OKR-白盒软件性能建模.md](02-改进版OKR-白盒软件性能建模.md) | 改进后的 Objective / KR |
| 总体方案 | [03-五仓库技术洞察与落地复审.md](03-五仓库技术洞察与落地复审.md) | 五仓库分工与 M0-M4 路线 |
| Collector 方案 | [04-MUPTI-ModelEvent技术方案.md](04-MUPTI-ModelEvent技术方案.md) | MUPTI / ModelEvent / collector 技术设计 |
| M1 细化 | [05-M1-API执行流程与ModelEvent插点.md](05-M1-API执行流程与ModelEvent插点.md) | M1 API 执行流程和插点 |
| M1 checklist | [06-M1-最小闭环实施清单.md](06-M1-最小闭环实施清单.md) | M1 工程闭环验收 |
| M2 细化 | [07-M2-Top90API执行流程与ModelEvent插点.md](07-M2-Top90API执行流程与ModelEvent插点.md) | M2 Top90 API 扩展和端到端插点 |
| API 全景 | [08-musa核心对象与API执行流程全景.md](08-musa核心对象与API执行流程全景.md) | musa 核心对象层次、API family 流程图和时序图 |

## 3. 工程标准件

| 文档 / 目录 | 作用 |
| --- | --- |
| [20-ModelEvent-Schema.md](20-ModelEvent-Schema.md) | ModelEvent record 字段、domain、event id 和质量要求 |
| [21-Relation-Schema.md](21-Relation-Schema.md) | API / ModelEvent / Activity / Submission 之间的关系边定义 |
| [22-OKR-Result-Schema.md](22-OKR-Result-Schema.md) | OKR 输出 JSON 字段和 M1 必须指标 |
| [source_rules/](source_rules/) | 每个 API 的源码路径、事件、关系和成本规则 |

当前已有 M1 source rule：

```text
source_rules/muLaunchKernel.yaml
source_rules/muMemAlloc.yaml
source_rules/muMemFree.yaml
source_rules/muStreamSynchronize.yaml
```

## 4. 历史 / 参考文档

这些文档保留为背景材料和早期推导，不作为当前最新实施方案入口。若与 `01-07` 主线文档冲突，以 `01-07` 为准：

| 文档 | 参考价值 |
| --- | --- |
| [90-参考-软件性能建模技术洞察.md](90-参考-软件性能建模技术洞察.md) | 业界方案对标、source rule、状态机、关系重建和成本分解方法论 |
| [91-参考-OKR评估与改进方案.md](91-参考-OKR评估与改进方案.md) | 早期 OKR 评估和改写建议 |
| [92-参考-四仓库技术洞察与OKR方案复审.md](92-参考-四仓库技术洞察与OKR方案复审.md) | 五仓库复审前的四仓库版本 |
| [93-参考-三仓库三套落地方案.md](93-参考-三仓库三套落地方案.md) | 最小闭环、OKR 主线、SDK 工程化三套方案对比 |
| [94-参考-早期完整落地方案.md](94-参考-早期完整落地方案.md) | 早期完整落地方案，包含事件体系、collector、建模方法和 CI 门禁 |

## 5. M0-M4 实施入口

### M0：评审并冻结目标和基线

入口文档：

```text
02-改进版OKR-白盒软件性能建模.md
03-五仓库技术洞察与落地复审.md
```

关键输出：

```text
3 个 workload 基线
Top90 Driver API 口径和列表
profiling / trace 基线
source rule 目录结构
```

### M1：最小白盒闭环

入口文档：

```text
05-M1-API执行流程与ModelEvent插点.md
06-M1-最小闭环实施清单.md
20-ModelEvent-Schema.md
21-Relation-Schema.md
22-OKR-Result-Schema.md
source_rules/*.yaml
```

关键输出：

```text
api_events
model_events
activity_events
relations
api_cost_breakdown
event_quality_report
overhead_report
```

### M2：Top90 API 扩展

入口文档：

```text
07-M2-Top90API执行流程与ModelEvent插点.md
08-musa核心对象与API执行流程全景.md
04-MUPTI-ModelEvent技术方案.md
```

关键输出：

```text
Top90 API source rule
API family ModelEvent 插点
Core/HAL/M3D/OS submit boundary 插点
relation recall report
unknown cost 占比下降
```

### M3：白盒模型 v1 和报告

入口文档：

```text
03-五仓库技术洞察与落地复审.md
02-改进版OKR-白盒软件性能建模.md
20-ModelEvent-Schema.md
21-Relation-Schema.md
22-OKR-Result-Schema.md
```

关键输出：

```text
3 个模型白盒性能建模报告
API cost breakdown
kernel Top-K overlap
profiling / msight 校准结果
优化建议
```

### M4：行为 CTS / SDK 回归

入口文档：

```text
03-五仓库技术洞察与落地复审.md
02-改进版OKR-白盒软件性能建模.md
22-OKR-Result-Schema.md
```

关键输出：

```text
launch pattern CTS
memory pool behavior CTS
stream/event sync CTS
graph behavior CTS
Green Context / resource CTS
SDK 版本行为差异报告
```

## 6. 仓库分工速览

| 仓库 | 主要职责 |
| --- | --- |
| `MUSA-Runtime` | Runtime API wrapper、`ApiTrace`、Runtime callback、correlation id 入口 |
| `linux-ddk/musa` | Driver API、Core command、Stream queue、Memory pool、Graph、HAL/M3D/OS submit、ModelEvent 主要插点 |
| `MUPTI` | callback / activity 基础能力、private ModelEvent collector、buffer / flush / dropped record、relation 输入 |
| `msight-system` | timeline 校准、API / kernel / memcpy 顺序对齐 |
| `msight-compute` | kernel 指标校准、occupancy / roofline / stall 信息辅助解释 |
| `insights` | OKR 文档、source rule、schema、报告和验收标准 |

## 7. 当前执行建议

短期不要继续新增平行方案文档，优先按下面顺序推进：

1. 评审并冻结 `20-ModelEvent-Schema.md`、`21-Relation-Schema.md`、`22-OKR-Result-Schema.md`，避免实现阶段字段漂移。
2. 基于 M1 文档实现 private ModelEvent collector 和四个 API 的最小闭环。
3. 用 M1 checklist 验收 collector 输出、关系重建、cost breakdown 和 overhead。
4. 基于 `08-musa核心对象与API执行流程全景.md` 和 M2 文档扩展 Top90 API family 和 submit boundary 插点。
5. 再进入 M3 报告和 M4 CTS 沉淀。

如果后续发现 `90-94` 参考文档中的内容与主线文档冲突，不要回滚主线方案，应把仍然有效的论据合并进 `01-07` 后继续执行。
