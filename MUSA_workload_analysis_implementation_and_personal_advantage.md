# MUSA 模型行为分析落地实现与个人优势结合

## 核心落地路线

当前最适合的落地方式，不是直接做一个很大的性能平台，而是先做一个 MUSA 模型行为分析闭环。

闭环路径：

```text
3 个模型
  ↓
采集 trace
  ↓
建立行为分类
  ↓
对齐 API / kernel / timeline
  ↓
输出瓶颈解释
  ↓
抽象 CTS
  ↓
进入 SDK 版本回归
```

这个闭环跑通以后，再扩展成工具、CTS 和 SDK 回归体系。

## 一、优先建设 Workload Signature

最应该优先做的是：

```text
MUSA Workload Signature
```

也就是把一个模型 workload 的执行行为压缩成一份可比较的行为指纹。

行为指纹包含：

```text
API TopN
Kernel TopN
Memory pattern
Stream/Event dependency
Graph usage
同步点
Critical path
Overlap ratio
Allocator pressure
KV cache 行为
MoE routing 行为
```

这个方向适合当前背景，因为它同时连接：

```text
模型语义
框架执行
MUSA API
driver 行为
kernel timeline
SDK 版本差异
CTS 抽象
```

这正好发挥 MindSpore 出身和 MUSA driver 工作之间的跨层优势。

## 二、第一阶段：Trace 解释器

第一阶段不要直接做复杂预测，先实现一个离线分析工具：

```text
输入 profiler trace
输出结构化报告
```

最小功能：

```text
1. 解析 API trace
2. 解析 kernel trace
3. 按 stream 建 timeline
4. 统计 Top API
5. 统计 Top kernel
6. 建立 API/kernel correlation
7. 输出 memory、stream/event、graph 分类统计
```

第一版目标不是预测性能，而是能稳定回答：

```text
这个模型主要时间花在哪里？
哪些 API 是热点？
哪些 kernel 是热点？
memory API 占比多少？
stream/event 同步点在哪里？
graph 有没有生效？
```

这一步可以直接支撑 KR1 和 KR3。

## 三、第二阶段：行为标注

第二阶段要把 trace 翻译成模型语义。

普通 driver 视角看到的是：

```text
musaLaunchKernel
musaMemcpyAsync
musaEventRecord
```

而行为分析需要翻译成：

```text
attention prefill
decode KV append
MLP projection
MoE dispatch
expert GEMM
optimizer step
```

落地方法不要一开始追求全自动，先用半自动规则：

```text
kernel name pattern
operator name
shape signature
执行顺序
stream id
layer repeat pattern
framework annotation
```

示例规则：

```text
连续 Q/K/V GEMM + attention kernel + output GEMM -> attention block
连续 gate/up/down GEMM -> MLP block
append/read + small copy/gather -> KV cache behavior
topk + scatter/gather + grouped GEMM -> MoE routing/dispatch
```

第一版可以先支持 3 个模型，不追求泛化所有模型。

## 四、第三阶段：API Cost Model

API 建模建议从简单、可解释的模型做起。

示例：

```text
musaMemcpyAsync:
  影响因子 = size + direction + stream + pinned/pageable + overlap 状态

musaMalloc/musaFree:
  影响因子 = size + pool hit/miss + 是否触发同步 + fragmentation

musaLaunchKernel:
  影响因子 = kernel 参数数量 + graph/non-graph + launch density

musaEventSynchronize / musaStreamSynchronize:
  影响因子 = 等待的下游 kernel/copy 是否完成
```

重点区分两类耗时：

```text
API self cost:
  driver/runtime 自身处理开销

API wait cost:
  因为依赖、同步、资源竞争导致的等待
```

这一阶段的目标：

```text
Top90 API 都能解释为什么慢。
```

输出不应只是表格，而要能归因：

```text
musaMemcpyAsync 慢：size 大还是 copy engine 被占？
musaStreamSynchronize 慢：等哪个 stream / kernel？
musaMalloc 慢：allocator 本身慢还是触发隐式同步？
```

## 五、第四阶段：Kernel Ranking Model

KR2 的目标是 Top50 kernel 排序 overlap，不是绝对时间误差。所以应该做 ranking model。

第一版可以使用：

```text
kernel_signature = kernel_name + op_type + dtype + shape + grid/block
```

kernel 分类：

```text
GEMM-like
Memory-bound
Reduction/softmax
Scatter/gather
Small kernel
Communication
```

预测方式：

```text
GEMM:
  flops / effective_tflops

Memory-bound:
  bytes / effective_bandwidth

Small kernel:
  launch_bound_latency + tiny_compute

Scatter/gather:
  token_count + index_bytes + irregular_penalty
```

再用 profiling 数据做 calibration：

```text
同类 kernel + 相似 shape -> median / p90 time
```

这样可以优先把 Top50 排序做准。

## 六、第五阶段：Critical Path

Critical Path 是最能体现技术深度的部分。

从 timeline 构建 DAG：

```text
节点:
  API
  kernel
  memcpy
  graph launch

边:
  same stream order
  event record/wait
  stream sync
  device sync
  memory dependency
  graph dependency
```

计算：

```text
critical path
blocking API
blocking kernel
copy/compute overlap
stream idle gap
pipeline break
```

这个能力比普通 profiling 高一层，因为它能回答：

```text
累计耗时最高的是不是瓶颈？
真正阻塞模型 step 的是哪几个点？
```

这会直接帮助 driver/runtime 优化决策。

## 七、第六阶段：CTS 沉淀

CTS 不要从 API 文档出发，要从真实模型行为出发。

建议先做 5 个：

```text
1. KV cache decode CTS
2. Attention graph CTS
3. Dense MLP burst CTS
4. MoE routing/dispatch CTS
5. Dynamic-shape decode CTS
```

每个 CTS 都输出：

```text
功能正确性
API sequence signature
kernel sequence signature
Top API
Top kernel
latency / p90 / p99
memory footprint
stream/event dependency
```

这样 CTS 就不只是“能不能跑”，而是能发现：

```text
SDK 新版本 API 行为变了
kernel 排序变了
graph fallback 了
同步点变多了
memory allocation 次数异常了
```

## 八、如何结合自身优势

最大优势不是单纯会 MindSpore，也不是单纯做 driver，而是跨层理解能力。

### 1. 懂框架执行图

相比纯 driver 工程师，更容易识别：

```text
哪个 kernel 属于 attention
哪个属于 MLP
哪个属于 optimizer
哪个属于 KV cache
哪个属于 MoE routing
```

因此适合负责：

```text
Workload Annotator
模型行为分类
kernel -> operator -> model phase 映射
```

### 2. 懂模型执行流程

能理解 prefill、decode、forward、backward 的执行差异。

因此能判断：

```text
decode 慢是不是 launch overhead
prefill 慢是不是 attention/GEMM
训练慢是不是 backward/optimizer/memory
MoE 慢是不是 dispatch/expert imbalance
```

这能让报告从普通性能表升级成模型级瓶颈解释。

### 3. 当前做 MUSA Driver

能够把上层问题落到 driver/API 改进点：

```text
memory pool
stream/event
graph launch
kernel launch overhead
implicit sync
copy engine overlap
API instrumentation
profiling 字段补齐
```

这让分析能闭环到 SDK 优化，而不是停在框架侧建议。

## 九、实际执行顺序

建议按以下顺序推进：

```text
第 1 步：
  选 1 个 dense inference 模型，跑通 trace parser 和 Top API/Top kernel 报告。

第 2 步：
  加入 stream timeline 和 memory 分类，输出 workload signature v0。

第 3 步：
  扩展到 MoE inference，补充 routing/dispatch/expert 行为标注。

第 4 步：
  扩展到 training workload，补充 backward/optimizer/memory 行为。

第 5 步：
  建 API Top90 cost model。

第 6 步：
  建 kernel Top50 ranking model。

第 7 步：
  建 critical path analyzer。

第 8 步：
  从 3 个模型中抽象 5 个 CTS。

第 9 步：
  做 SDK vX vs vY 行为和性能 diff。

第 10 步：
  固化成自动报告和 SDK 回归入口。
```

## 十、个人技术标签

不要把自己定位成“做 profiling 的”或“写 CTS 的”，更适合定位成：

```text
MUSA AI workload 行为分析与性能建模
```

或者：

```text
AI workload-aware GPU driver/runtime performance engineer
```

可以形成的标志性成果：

```text
MUSA Model Behavior Signature & Performance Modeling Framework
```

它的长期价值：

```text
模型来了，可以快速识别行为模式；
SDK 变了，可以快速判断性能变化原因；
driver 优化了，可以验证对真实模型是否有效；
CTS 沉淀了，可以防止关键模型行为回退。
```

最务实的突破口：

```text
先把一个 dense LLM decode 的 trace 解释清楚。
```

只要能把 API、kernel、KV cache、stream/event、graph、critical path 串起来，后面扩展到 MoE 和 training 就有清晰路径。
