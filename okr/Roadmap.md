# Driver Perf Roadmap

## 1. 目标

性能建模的核心目标，是在系统正式部署到生产环境之前，通过低成本的模型化手段，对系统的真实运行表现进行精准预测，将性能风险前置化解。

针对 Driver 性能建模，核心目标是在版本发布前对 Driver 的性能进行量化评估，并精准定位性能瓶颈。

| 目标 | 说明 |
| --- | --- |
| 量化评估性能表现 | 在版本发布前，通过 Perf / profiling 手段发现系统在不同模型下的关键性能指标，对 Driver 版本进行性能看护。 |
| 精准定位性能瓶颈 | 通过对 Driver 内部事件的采集，精准识别出影响整体性能的核心约束环节。 |

## 2. 整体思路

整体思路：通过 MUPTI 埋点，将 Driver 的执行过程串成可解释的白盒性能模型。

把一次 Runtime / Driver API 调用拆开，分析 Driver 内部经过了哪些链路，每个链路分别消耗了多少时间，最终能形成调用链：

```text
API total time
  = runtime wrapper
  + driver wrapper
  + object / context lookup
  + memory / stream / command processing
  + dependency / queue / build
  + submit / HAL / M3D / OS boundary
```

RoadMap：先跑通链路，然后选择单个模型实现原型，最后覆盖更多 API 和模型。

| 阶段 | 计划 |
| --- | --- |
| M0 | 针对内存 API 跑通链路 |
| M1 | 确定模型，并覆盖该模型的 Top90 API |
| M2 | 抽象主流模型行为模式，并验证行为级性能建模 |
| M3 | 使用 msight / profiling 对齐性能建模结果 |
| M4 | 沉淀到 CTS，并形成性能看护 |

## 3. 阶段规划

### M0：针对内存 API 跑通链路

针对内存 API 跑通链路，实现 demo，达到解释“API 后经过了哪些链路、每个链路耗时是什么样”的目标。

这一阶段重点不是覆盖所有 API，而是先跑通原型。

重点覆盖：

```text
musaMalloc / muMemAlloc -> musaFree / muMemFree
```

在内存 API 链路跑通后，可以在同一套机制上继续扩展到 launch / stream sync，但这不改变 M0 的核心目标。

输出目标：

- 能采集 Runtime API / Driver API / Driver 内部 ModelEvent。
- 能说明一次 API 调用经过了哪些 Driver 内部阶段。
- 能输出每个阶段的耗时分解。

### M1：确定模型并覆盖模型 Top90 API

选择一个模型，达到针对模型本身的 Top90 API 覆盖。

输出目标：

- 输出 Runtime API、Driver API、kernel 的基线统计。
- 生成 Top90 Driver API 清单。

### M2：抽象主流模型行为模式

抽象主流模型的行为模式，并对抽象出的行为验证性能建模。

这一阶段关注的不是单个 API，而是模型执行中稳定出现的行为模式。

行为模式：

- attention launch pattern
- MLP launch pattern
- KV cache allocation pattern
- expert routing pattern
- graph launch pattern

输出目标：

- 抽象主流模型中的关键行为模式。
- 为每类行为定义 API sequence。
- 用 micro benchmark 或 CTS 风格用例验证这些行为是否可复现、可解释。

### M3：完成 API 覆盖以及与 msight / profiling 对齐

完成 Top90 API 覆盖，使用 msight / profiling 对性能建模进行对齐，达到“模型耗时在 Driver 侧的统计”可解释、可校准。

这一阶段要把白盒事件和 profiling 结果对齐，证明模型不是只在 Driver 内部自洽，而是能和真实 workload 的 kernel、timeline、activity 对上。

输出目标：

- 输出 API cost breakdown。
- 能说明模型耗时中 Driver 侧成本、kernel 侧成本分别占多少。

### M4：沉淀 CTS 和性能看护

沉淀到 CTS 中，并对性能建模进行看护。

这一阶段把稳定的行为模式和事件签名固化成回归用例，用于 SDK / Driver 版本发布前的性能看护。

输出目标：

- 将 launch、memory、stream / event、graph、GreenContext 等行为沉淀为 CTS 或 benchmark。
- 支持 SDK 版本差异分析，形成初步的报告。

## 4. 最终目标

形成一套可解释、可校准、可回归的 Driver/API 白盒性能建模体系：

```text
workload baseline
  -> Top90 API
  -> ModelEvent
  -> MUPTI collector
  -> cost breakdown
  -> behavior CTS / performance guard
```

回答：

- 哪些 API 是当前模型 workload 的性能热点？
- 每个热点 API 的耗时花在 Driver 内部哪些阶段？
- 版本变化后，是 API 行为变了、Driver 内部成本变了，还是 kernel / device 执行变了？
