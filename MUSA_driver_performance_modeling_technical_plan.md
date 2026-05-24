# MUSA Driver/API 性能建模与行为分析技术方案

## 1. 目标与边界

### 1.1 建设目标

围绕 3 个主流模型的推理/训练 workload，建立一套面向 MUSA driver/API 的性能建模、行为分析和 CTS 沉淀体系。

核心目标：

```text
模型 trace -> 行为抽象 -> API/kernel 建模 -> profiling 对齐 -> SDK 版本回归 -> 优化建议
```

最终能力包括：

- 覆盖 trace 中累计耗时 Top90% 的 driver API。
- 对 Top50 kernel 的累计耗时排序建模，Top-K overlap 达到 90% 以上。
- 输出 3 个模型在一个 SDK 版本上的性能建模分析报告。
- 抽象 dense 和 MoE 模型中的关键行为，沉淀到 MUSA CTS。
- 建立 SDK 版本间的行为和性能差异分析能力。

### 1.2 技术边界

本方案不直接替代 profiler、benchmark 或 CTS，而是在三者之间建立统一分析层：

```text
Profiler:
  提供 API、kernel、timeline、memory、stream/event、graph 等原始数据。

Performance Model:
  对 API/kernel 耗时和瓶颈进行解释与预测。

Behavior CTS:
  把模型中的真实执行行为抽象成可复现、可回归的小型用例。
```

### 1.3 主要用户

- MUSA driver/runtime 开发者：定位 API、memory、stream/event、graph 行为瓶颈。
- Kernel library 开发者：定位 Top kernel 和 shape/dtype/layout 相关瓶颈。
- SDK 发布负责人：判断新版本是否引入行为或性能回退。
- 框架适配负责人：判断模型执行图和 runtime 调用模式是否合理。

## 2. 总体架构

### 2.1 系统模块

建议建设 6 个模块：

```text
1. Trace Collector
  采集 MUSA API trace、kernel trace、memory trace、stream/event trace、graph trace。

2. Trace Normalizer
  统一不同 profiler 输出格式，生成标准 trace schema。

3. Workload Annotator
  把原始 API/kernel 序列映射到模型行为片段，例如 attention、MLP、KV cache、MoE routing。

4. Performance Modeling Engine
  建立 API cost model、kernel ranking model、critical path model。

5. Behavior Signature & Diff
  生成 workload 行为签名，并支持 SDK 版本间 diff。

6. CTS Generator / CTS Library
  把高频模型行为抽象成可复现的 MUSA CTS 用例。
```

### 2.2 数据流

```text
模型执行
  ↓
Profiler / Trace 工具
  ↓
Raw trace
  ↓
Trace schema 标准化
  ↓
行为标注与 API/kernel 分类
  ↓
性能建模与 profiling 对齐
  ↓
行为签名、瓶颈分析、CTS 抽象
  ↓
SDK 版本报告与优化建议
```

### 2.3 核心设计原则

- 先排序，后绝对值：早期优先保证 Top API、Top kernel 和关键路径判断准确。
- 先规则，后学习：先用可解释的参数化模型和 bucket 校准，后续再引入 ML。
- 先行为，后 API：CTS 不只覆盖 API，而是覆盖模型中的关键行为路径。
- 先单机，后分布式：先把单卡/单机的 memory、stream、graph、kernel 行为建扎实，再扩展到通信。
- 先离线分析，后自动化闭环：先保证报告质量，再接入 CI 或 SDK 发布流程。

## 3. Trace Schema 设计

### 3.1 API Event

每个 MUSA API 调用记录为一个事件：

```text
api_event:
  api_name
  api_category
  thread_id
  process_id
  start_ts
  end_ts
  duration
  stream_id
  context_id
  device_id
  args
  return_code
  correlation_id
  sync_type
  dependency_hint
```

API 分类建议：

```text
Memory:
  malloc/free/memcpy/memset/memory_pool

Kernel:
  launch/config/argument_setup

Stream/Event:
  create/destroy/record/wait/query/sync

Graph:
  capture/instantiate/update/launch/destroy

Module:
  load/unload/function_lookup

Device/Context:
  set_device/context_create/context_destroy/device_sync
```

### 3.2 Kernel Event

```text
kernel_event:
  kernel_name
  demangled_name
  op_type
  start_ts
  end_ts
  duration
  stream_id
  device_id
  grid_dim
  block_dim
  shared_memory
  registers
  dtype
  shape_signature
  input_bytes
  output_bytes
  estimated_flops
  estimated_memory_bytes
  correlation_id
```

### 3.3 Memory Event

```text
memory_event:
  operation
  pointer
  size
  direction
  memory_type
  alignment
  stream_id
  start_ts
  end_ts
  duration
  is_async
  is_pinned
  pool_id
```

### 3.4 Stream/Event/Graph Event

```text
stream_event:
  stream_id
  operation
  priority
  flags
  wait_target
  dependency_source
  timestamp

graph_event:
  graph_id
  operation
  node_count
  kernel_node_count
  memcpy_node_count
  instantiate_time
  launch_time
  update_time
```

### 3.5 Workload Segment

把一段 API/kernel 序列归属到模型行为：

```text
workload_segment:
  model_name
  phase
  layer_id
  behavior_type
  start_ts
  end_ts
  api_events
  kernel_events
  shape_context
```

`phase` 示例：

```text
prefill
decode
forward
backward
optimizer
communication
```

`behavior_type` 示例：

```text
attention
mlp
kv_cache
moe_routing
moe_dispatch
expert_compute
moe_combine
memory_management
graph_replay
```

## 4. 性能建模方案

### 4.1 API Cost Model

API cost model 分三层。

第一层：基础统计模型。

```text
api_name + api_category -> count / total_time / avg / p50 / p90 / p99
```

第二层：参数化模型。

```text
memcpy_time = base_latency + size / effective_bandwidth + dependency_wait
malloc_time = base_latency + pool_miss_penalty + fragmentation_penalty
kernel_launch_time = base_latency + arg_count_factor + stream_state_factor
event_wait_time = dependency_gap + runtime_overhead
graph_launch_time = replay_base + node_count_factor + update_penalty
```

第三层：上下文模型。

```text
api_cost = f(api_name, args, stream_state, dependency_state, memory_state, graph_state)
```

重点区分：

```text
API 自身开销:
  runtime/driver 处理调用所需时间。

依赖等待开销:
  因 stream/event/device sync 或资源冲突导致的等待。

隐式同步开销:
  由 pageable memory、memory allocation、graph update 等触发的同步。

资源竞争开销:
  copy engine、compute engine、allocator、command queue 等资源拥塞。
```

### 4.2 Kernel Ranking Model

目标不是一开始预测每个 kernel 的绝对耗时，而是让 Top50 kernel 排序和 profiling 对齐。

模型流程：

```text
kernel_event
  ↓
kernel fingerprint
  ↓
op type 分类
  ↓
shape bucket
  ↓
cost predictor
  ↓
Top-K ranking
```

kernel fingerprint：

```text
kernel_name
op_type
dtype
shape_signature
grid/block
estimated_flops
estimated_memory_bytes
stream_id
phase
behavior_type
```

分类模型：

```text
GEMM-like:
  time ~= flops / effective_tflops(shape, dtype, layout)

Memory-bound:
  time ~= bytes / effective_bandwidth(access_pattern)

Reduction/softmax:
  time ~= f(seq_len, head_num, block_dim, sync_cost)

Scatter/gather:
  time ~= base + token_count * irregularity_factor + index_bytes / bandwidth

Small kernel:
  time ~= launch_bound_latency + minimal_compute_time

Communication:
  time ~= latency + bytes / bandwidth + topology_penalty
```

对齐指标：

```text
Top-K overlap = |predicted_top_k ∩ profiled_top_k| / K
```

建议统计：

```text
Top10 overlap
Top20 overlap
Top50 overlap
Spearman rank correlation
累计耗时覆盖率
误差最大的 kernel 列表
```

### 4.3 Critical Path Model

单纯累计 API/kernel 耗时容易误判，因为 MUSA 执行是异步和多 stream 的。

需要构建依赖 DAG：

```text
node:
  API event
  kernel event
  memcpy event
  graph event

edge:
  same stream order
  event record/wait
  device sync
  stream sync
  memory dependency
  graph dependency
```

输出：

```text
critical_path_duration
critical_path_nodes
blocking_api
blocking_kernel
overlap_ratio
idle_gap
```

关键价值：

```text
区分累计耗时高但被 overlap 掉的事件
识别出现次数少但阻塞全局的同步 API
发现 stream pipeline 断点
发现 copy/compute overlap 下降
```

### 4.4 Behavior Signature

每个模型 workload 生成行为签名：

```text
workload_signature:
  model_name
  sdk_version
  hardware
  workload_config
  api_top_list
  kernel_top_list
  memory_pattern
  stream_graph
  graph_usage
  sync_points
  overlap_ratio
  launch_density
  allocator_pressure
  behavior_segment_stats
```

用于 SDK diff：

```text
api_hotspot_diff
kernel_ranking_diff
critical_path_diff
memory_pattern_diff
stream_dependency_diff
graph_behavior_diff
cts_signature_diff
```

## 5. 模型选择与测试矩阵

### 5.1 推荐模型组合

建议选择 3 类互补 workload：

```text
1. Dense LLM inference
  例如 Llama/Qwen dense。
  覆盖 attention、MLP、KV cache、prefill/decode。

2. MoE LLM inference
  例如 DeepSeek/Mixtral/Qwen-MoE 类。
  覆盖 routing、dispatch、expert compute、combine。

3. Dense training
  例如 Transformer training 或 ViT/Diffusion training。
  覆盖 forward、backward、optimizer、activation memory、同步行为。
```

### 5.2 配置矩阵

每个模型至少选择 2 到 3 个配置，避免只对单点 shape 建模：

```text
Dense inference:
  batch = 1 / 4 / 8
  seq_len = 128 / 2048 / long context
  phase = prefill / decode

MoE inference:
  batch = 1 / 4
  seq_len = 128 / 2048
  top_k = model default
  expert_parallel = off / on if available

Training:
  batch = small / medium
  seq_len or image_size = representative values
  precision = fp16/bf16 if supported
```

### 5.3 必须固化的环境信息

每份 trace 和报告必须记录：

```text
SDK version
driver version
firmware version
hardware SKU
card count
framework version
model commit/config
precision
batch size
sequence length
warmup steps
measurement steps
profiling tool version
environment variables
```

否则版本差异分析不可信。

## 6. CTS 设计方案

### 6.1 CTS 用例原则

CTS 不应只是 API smoke test，而要抽象模型中的真实行为。

每个 CTS 用例包含：

```text
functional_check:
  结果正确性或行为完整性。

performance_check:
  latency、bandwidth、throughput、p50/p90/p99。

behavior_check:
  API sequence、kernel sequence、memory pattern、stream/event dependency。
```

### 6.2 建议沉淀的 5 类 CTS

#### 6.2.1 Long-context KV Cache CTS

目标：

```text
覆盖 KV cache allocate/append/read/reuse。
验证 decode 阶段 memory 行为和 latency 稳定性。
```

核心指标：

```text
KV append latency
KV read latency
device memory footprint
allocation count
memcpy count
decode step latency
```

#### 6.2.2 Attention Graph CTS

目标：

```text
覆盖 QKV、attention、output projection 的 stream/event/graph 行为。
```

核心指标：

```text
graph capture time
graph instantiate time
graph launch time
kernel launch count
Top kernel ranking
```

#### 6.2.3 Dense MLP Burst CTS

目标：

```text
覆盖连续 GEMM、activation、memory reuse。
检查 launch overhead 和 allocator 行为。
```

核心指标：

```text
GEMM kernel time
activation kernel time
temporary memory allocation
stream overlap
```

#### 6.2.4 MoE Routing CTS

目标：

```text
覆盖 router、top-k、dispatch、expert compute、combine。
```

核心指标：

```text
tokens per expert distribution
dispatch latency
combine latency
expert GEMM utilization
small kernel count
```

#### 6.2.5 Dynamic-shape Decode CTS

目标：

```text
覆盖 batch、seq_len、token 数变化下的 graph update、kernel launch、memory pool 行为。
```

核心指标：

```text
graph update count
graph update latency
fallback launch count
memory pool hit rate
decode latency variance
```

### 6.3 CTS 验收标准

每个 CTS 至少输出：

```text
pass/fail
API TopN
Kernel TopN
behavior signature
performance metrics
baseline diff
```

进入 SDK 回归后，至少检查：

```text
行为签名是否变化
Top API 是否变化
Top kernel 是否变化
关键路径是否变化
核心指标是否超阈值退化
```

## 7. 报告模板

每个模型输出一份分析报告，建议结构如下：

```text
1. 测试环境与 workload 配置
2. Trace 覆盖情况
3. API 热点分析
4. Kernel 热点分析
5. Memory 行为分析
6. Stream/Event/Graph 行为分析
7. Critical path 与 overlap 分析
8. 性能模型对齐结果
9. SDK 版本差异分析
10. 优化建议与优先级
```

优化建议要按收益和落地难度分级：

```text
P0:
  明确阻塞关键路径，且 driver/runtime 可直接优化。

P1:
  明确热点，但需要 kernel/framework/runtime 协同。

P2:
  现阶段影响较小，作为趋势观察。
```

## 8. 落地实施步骤

### 阶段 0：准备与对齐

周期：1 到 2 周。

目标：

```text
确定 3 个模型、测试配置、硬件环境、SDK 版本、profiling 工具链。
```

交付：

```text
模型清单
测试配置表
trace 字段清单
报告模板
CTS 候选列表
```

关键动作：

```text
1. 固定 SDK 和硬件版本。
2. 固定模型和输入配置。
3. 明确 profiler 能采集哪些字段，哪些字段需要补采。
4. 明确 Top90 API 和 Top50 kernel 的计算口径。
5. 明确 Top-K overlap 的 K 值，例如 Top10/Top20/Top50。
```

风险：

```text
profiling 字段不完整，导致 API 和 kernel 无法通过 correlation id 对齐。
```

应对：

```text
先做最小闭环，只要求时间戳、stream id、api name、kernel name、duration 能对齐。
```

### 阶段 1：Trace 标准化与 Baseline 建立

周期：2 到 3 周。

目标：

```text
拿到 3 个模型的 baseline trace，并标准化成统一 schema。
```

交付：

```text
trace normalizer
标准 trace 数据
3 个模型 baseline
API TopN / Kernel TopN 初版报告
```

关键动作：

```text
1. 解析 profiler 输出，生成 api_event、kernel_event、memory_event。
2. 对齐 API 和 kernel 的 correlation id。
3. 标准化 stream、event、graph 信息。
4. 生成基础统计：Top API、Top kernel、memory summary、stream summary。
5. 输出第一版 workload signature。
```

验收：

```text
每个模型至少能输出：
- Top API by cumulative time
- Top kernel by cumulative time
- API category coverage
- kernel category coverage
- timeline summary
```

### 阶段 2：API Cost Model 建设

周期：3 到 4 周。

目标：

```text
覆盖 trace 中累计耗时 Top90% 的 driver API。
```

交付：

```text
API 分类器
API 参数化 cost model
API 对齐报告
API 瓶颈归因
```

关键动作：

```text
1. 按 memory、kernel、stream/event、graph、device/context 分类。
2. 对 Top API 建立参数化模型。
3. 区分 API 自身开销、依赖等待、隐式同步、资源竞争。
4. 标注疑似阻塞 API。
5. 建立 API coverage 统计。
```

验收：

```text
Top90% 累计 API 耗时均有模型解释。
每个 Top API 都能给出主要影响因子。
能输出 API 瓶颈优先级。
```

### 阶段 3：Kernel Ranking Model 建设

周期：3 到 4 周。

目标：

```text
Top50 kernel 按累计耗时排序的 Top-K overlap 达到 90% 以上。
```

交付：

```text
kernel fingerprint
kernel op type 分类器
shape bucket
kernel ranking predictor
Top-K overlap 报告
```

关键动作：

```text
1. 从 kernel name、shape、dtype、grid/block 中提取 fingerprint。
2. 按 GEMM、memory-bound、reduction、scatter/gather、small kernel 分类。
3. 建立 shape bucket 和 calibration table。
4. 输出预测 Top50 kernel。
5. 与 profiling Top50 对齐，分析 miss case。
```

验收：

```text
Top10/Top20/Top50 overlap 达到目标。
Top50 中未命中的 kernel 有明确原因分析。
Top kernel 变化能映射到模型行为片段。
```

### 阶段 4：Critical Path 与 Behavior Signature

周期：3 到 5 周。

目标：

```text
建立异步执行下的关键路径分析和 SDK diff 能力。
```

交付：

```text
dependency DAG
critical path analyzer
overlap analyzer
workload signature diff
SDK diff 报告
```

关键动作：

```text
1. 根据 same-stream order、event wait、sync、graph dependency 建 DAG。
2. 计算 critical path。
3. 计算 compute/copy overlap、stream overlap、idle gap。
4. 识别阻塞 API 和 pipeline break。
5. 生成 workload signature，并与上一 SDK 版本 diff。
```

验收：

```text
能区分累计热点和关键路径热点。
能解释 SDK 版本间性能变化的主要来源。
能输出关键同步点和 overlap 变化。
```

### 阶段 5：CTS 抽象与沉淀

周期：4 到 6 周。

目标：

```text
针对 dense 和 MoE 模型抽象不少于 5 个行为特征用例。
覆盖 memory、stream/event、graph 等关键 API 达到 90% 以上。
```

交付：

```text
5 类行为 CTS
CTS baseline
CTS 行为签名
CTS 性能阈值
CTS SDK 回归报告
```

关键动作：

```text
1. 从真实 trace 中截取和抽象关键行为。
2. 将 attention、MLP、KV cache、MoE routing 等行为最小化为 CTS。
3. 为每个 CTS 建立功能正确性、性能指标、行为签名。
4. 建立 baseline 和阈值。
5. 接入 SDK 回归流程。
```

验收：

```text
不少于 5 个行为 CTS。
CTS 覆盖 trace 中 memory、stream/event、graph 关键 API 90% 以上。
每个 CTS 都能输出 baseline diff。
```

### 阶段 6：报告自动化与优化闭环

周期：持续迭代。

目标：

```text
把分析能力固化到 SDK 发布流程。
```

交付：

```text
自动报告生成器
SDK 版本对比 dashboard
优化建议列表
问题归因记录
```

关键动作：

```text
1. 固化报告模板。
2. 自动生成 3 个模型的性能建模报告。
3. 自动生成 SDK diff。
4. 将 P0/P1 优化项分派给 driver/runtime/kernel/framework 责任模块。
5. 下一版本复测闭环。
```

验收：

```text
一个 SDK 版本能稳定发布 3 个模型分析报告。
报告包含 API 热点、kernel 热点、critical path、优化建议。
关键优化建议有复测数据闭环。
```

## 9. 风险与应对

### 9.1 Trace 不完整

风险：

```text
API 和 kernel 缺少 correlation id，memory 或 graph 字段缺失。
```

应对：

```text
先用 timestamp + stream id + launch order 做弱关联。
逐步推动 profiler 补齐 correlation id 和 graph node id。
```

### 9.2 模型行为难以自动标注

风险：

```text
无法直接从 kernel name 判断 attention、MLP、MoE 行为。
```

应对：

```text
先用规则和时间窗口标注。
结合框架侧 operator name 或 debug tag。
必要时在框架侧插入轻量 annotation。
```

### 9.3 绝对时间预测误差较大

风险：

```text
硬件状态、温度、频率、系统噪声导致耗时波动。
```

应对：

```text
早期以排序准确性为主。
用 median/p90 和多轮采样降低噪声。
绝对误差只作为辅助指标。
```

### 9.4 CTS 过大或过拟合

风险：

```text
直接搬真实模型片段，导致 CTS 太重或依赖复杂。
```

应对：

```text
抽象行为，不复制完整模型。
保留 API/kernel 行为特征，降低框架依赖。
```

### 9.5 SDK diff 难以归因

风险：

```text
多个模块同时变化，性能变化难以归因。
```

应对：

```text
使用行为签名先判断执行结构是否变化。
再用 critical path 和 Top kernel diff 定位主要变化点。
必要时做二分版本验证。
```

## 10. 优先级建议

### P0：必须先做

```text
trace schema
3 个模型 baseline
Top API / Top kernel 统计
API Top90 coverage
Kernel Top50 overlap
报告模板
```

### P1：形成壁垒

```text
critical path analyzer
workload signature diff
KV cache behavior model
MoE routing behavior model
5 类行为 CTS
```

### P2：持续增强

```text
自动 dashboard
ML-based cost predictor
分布式通信建模
跨模型泛化
CI 自动回归
```

## 11. 推荐里程碑

```text
M1:
  完成 3 个模型 baseline trace 和基础 TopN 报告。

M2:
  完成 Top90 API cost model 和 API 瓶颈归因。

M3:
  完成 Top50 kernel ranking model，Top-K overlap >= 90%。

M4:
  完成 workload signature、critical path、SDK diff 初版。

M5:
  完成 5 个行为 CTS，并建立 baseline。

M6:
  针对一个 SDK 版本发布 3 个模型性能建模与行为分析报告。
```

## 12. 最终交付形态

建议最终形成一个内部工程项目：

```text
MUSA Model Behavior Benchmark & Performance Modeling Framework
```

包含：

```text
trace parser
trace schema
API model
kernel model
critical path analyzer
workload signature diff
CTS case library
report generator
SDK baseline database
```

它应该能稳定回答：

```text
1. 当前 SDK 下，3 个模型的 API 热点是什么？
2. 当前 SDK 下，3 个模型的 kernel 热点是什么？
3. Top50 kernel 排序预测和 profiling 是否一致？
4. 哪些 API/kernel 在关键路径上？
5. 新旧 SDK 的行为是否变化？
6. 性能变化来自 memory、stream/event、graph、kernel 还是模型行为？
7. 哪些行为应该沉淀到 CTS，防止后续回退？
```

