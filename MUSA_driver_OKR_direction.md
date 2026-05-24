# MUSA Driver/API 性能建模与行为分析发力方向

## 核心定位

当前 OKR 可以归纳为一句话：

> 把模型 workload 变成 MUSA driver/API 可解释、可预测、可回归、可优化的系统对象。

重点不应停留在 profiling 报告或普通 CTS 用例，而应从模型 trace 中抽象出 driver 行为模式，建立性能模型和行为基线，让 SDK 每个版本都能回答：

- 为什么快或慢
- 慢在哪里
- 哪些 API 或 kernel 是关键瓶颈
- 下一步应该优化哪里

结合 MindSpore 背景和当前 MUSA driver 开发岗位，最有价值的位置在三层之间：

```text
模型/框架语义
    ↓
Runtime / Driver API 行为
    ↓
Kernel / Memory / Stream / Graph 执行
```

很多 driver 工程师只看 API 和 kernel，不知道上层模型为什么这样调用；很多框架工程师只看 operator，不知道底层为什么慢。你最适合做的是中间层：

```text
模型行为语义化 + MUSA API 性能建模 + SDK 回归分析
```

## 一、Trace Taxonomy：建立模型行为分类体系

不要一开始就建复杂模型，应该先把 3 个模型的 trace 分成稳定类别。

建议至少抽象以下行为：

```text
Attention:
  QKV projection
  RoPE / position embedding
  attention score
  softmax
  value aggregation
  output projection

MLP:
  gate/up projection
  activation
  down projection

KV Cache:
  allocate
  append/write
  read/gather
  page/block mapping
  cache reuse
  eviction or reset

MoE:
  router / gating
  top-k select
  token dispatch
  expert GEMM
  token combine
  all-to-all / all-gather if distributed

Runtime Behavior:
  memory alloc/free
  H2D/D2H/D2D copy
  stream create/sync/wait
  event record/wait/query
  graph capture/instantiate/launch
  kernel launch
```

CTS 用例也应该来自这个 taxonomy，而不是凭经验造 API case。

最终要形成映射：

```text
模型行为片段 -> MUSA API 序列 -> kernel 序列 -> 性能指标
```

这是后续所有 KR 的基础。

## 二、API Cost Model：覆盖 Top90% Driver API

KR1 要求覆盖 trace 中累计耗时 Top90% 的 driver API。这里不要只做平均耗时表，而要建立参数化模型。

示例：

```text
memcpy_time = launch_overhead + size / bandwidth + sync_penalty
malloc_time = base + pool_miss_penalty + fragmentation_penalty
kernel_launch_time = base + arg_setup + graph_mode_delta
event_wait_time = dependency_gap + scheduling_overhead
graph_launch_time = instantiate_amortized + replay_overhead
```

重点建模 API 类别：

```text
Memory:
  musaMalloc / musaFree
  musaMemcpy / musaMemcpyAsync
  musaMemset / musaMemsetAsync
  memory pool APIs if any

Stream/Event:
  stream create/destroy
  stream sync
  event record
  event wait
  event query

Kernel:
  launch overhead
  argument setup
  occupancy-sensitive kernel time
  small kernel latency

Graph:
  capture
  instantiate
  update
  launch
  replay
```

关键点是：API 耗时不能只按 API 名建模，要按上下文建模。

同样是 `musaMemcpyAsync`，性能取决于：

```text
copy direction
size
alignment
pinned or pageable host memory
same stream or cross-stream dependency
是否触发隐式同步
是否和 kernel overlap
copy engine 是否被占满
```

性能模型需要能回答：

```text
这个 API 慢，是 API 本身慢，还是被前面的依赖阻塞？
```

这是 profiling 数据对齐的关键。

## 三、Kernel Ranking Model：先追 Top50 排序，不追绝对误差

KR2 要求 Top50 kernel 按累计耗时排序的 Top-K overlap 达到 90%。这个目标很合理，因为早期模型最重要的是找对瓶颈，而不是把每个 kernel 的绝对误差压到极低。

建议采用两层模型：

```text
第一层：kernel fingerprint
  kernel name
  op type
  shape
  dtype
  layout
  batch/seq length
  hidden size
  expert count
  token count
  stream
  launch order

第二层：cost predictor
  static features + profiling calibration
```

kernel 分类：

```text
GEMM-like:
  time ~= FLOPs / effective_TFLOPS(shape, dtype)

Memory-bound:
  time ~= bytes / effective_bandwidth(pattern)

Reduction/softmax:
  time ~= f(seq_len, head_num, block size, bandwidth, sync cost)

Dispatch/scatter/gather:
  time ~= tokens + index bytes + irregularity penalty

Communication:
  time ~= latency + size / bandwidth + topology penalty
```

早期不一定要做机器学习模型。更建议：

```text
规则模型 + shape bucket + profiling calibration
```

例如维护：

```text
kernel_signature -> median_time / p90_time / variance / input_shape
```

然后逐步泛化到：

```text
same op type + similar shape -> predicted time
```

## 四、CTS：沉淀模型行为测试，而不是普通 API 测试

CTS 最有挑战的地方不是覆盖 API，而是把模型中的真实行为抽象成小而稳定的 case。

建议沉淀 5 类高价值 CTS：

```text
1. Long-context KV cache case
  连续 decode，多次 KV append/read，检查 latency 和显存行为

2. Attention graph case
  QKV + attention + output projection，覆盖 stream/event/graph

3. Dense MLP burst case
  连续 GEMM + activation + memory reuse，检查 launch 和 allocator 行为

4. MoE routing case
  top-k routing + token dispatch + expert compute + combine

5. Dynamic-shape decode case
  batch/seq/token 数变化，检查 graph update、kernel launch、memory pool
```

每个 CTS 都应该包含三类输出：

```text
功能正确性
性能指标
行为签名
```

行为签名可以包括：

```text
API sequence hash
kernel sequence hash
memory allocation pattern
stream/event dependency graph
top kernel ranking
API cumulative time ranking
```

这样 SDK 新版本一跑，就能判断：

```text
功能没错，但行为变了
行为没变，但性能慢了
性能快了，是哪个 API / kernel / dependency 变了
```

这就是 CTS 从测试升级为行为基线系统。

## 五、真正有挑战的方向

### 方向 1：MUSA Workload Signature

做一个模型行为签名系统。

输入：

```text
MUSA trace + kernel trace + profiling metrics
```

输出：

```text
这个 workload 的行为指纹
```

行为指纹包括：

```text
API 热点
kernel 热点
memory pattern
stream dependency graph
graph usage pattern
同步点
overlap 率
launch 密度
allocator 压力
```

价值在于每个 SDK 版本都可以自动 diff。

示例报告：

```text
SDK vX 到 vY:
- kernel A 耗时下降 18%
- musaEventSynchronize 累计耗时上升 42%
- stream overlap 从 63% 降到 49%
- memory allocation 次数增加 3.2x
- Top50 kernel overlap 仍为 92%，说明主要计算结构没变
```

### 方向 2：Driver API Critical Path 分析

很多 API 耗时不能看 wall time 累加，因为异步执行会 overlap。

应该建立：

```text
API trace / kernel trace -> dependency DAG -> critical path
```

这样可以区分：

```text
累计耗时高但不在关键路径
累计耗时不高但阻塞全局
```

示例：

```text
musaMemcpyAsync 累计时间很高，但和 GEMM overlap，不是瓶颈
musaStreamSynchronize 只出现 3 次，但切断 pipeline，是真瓶颈
```

这类分析比普通 profiling 高一个层次。

### 方向 3：MoE 行为建模

MoE 是最值得挑战的方向之一。

可以专门建立这些指标：

```text
expert load balance
tokens per expert distribution
dispatch/combine overhead
expert GEMM utilization
all-to-all volume
routing kernel latency
small expert GEMM fragmentation
```

MoE 的 driver/API 优化点包括：

```text
减少 token dispatch 的小 kernel
优化 scatter/gather memory pattern
expert GEMM batching
stream 并行 expert execution
通信和计算 overlap
hot expert cache
```

如果能把 MoE 的行为抽象成 CTS，会很有价值，因为未来模型会越来越多走 MoE。

### 方向 4：KV Cache Memory Model

KV cache 是推理场景最确定的长期方向。

可以建立：

```text
KV cache bytes per token
append latency
read latency
cache hit/reuse
page/block fragmentation
memory pool pressure
decode step latency
```

然后从 driver/API 角度分析：

```text
是否频繁 malloc/free
是否有隐式同步
是否 D2D copy 过多
是否 cache layout 导致 gather 低效
是否 graph capture 被动态地址破坏
```

这个方向很适合当前 OKR，因为它同时覆盖：

```text
memory API
attention 行为
decode 推理
CTS 沉淀
性能建模
```

## 六、建议选择的 3 个模型

建议覆盖三类，不要都选同一种 Transformer。

```text
1. Dense LLM
  例如 Llama/Qwen dense
  目标：attention + MLP + KV cache baseline

2. MoE LLM
  例如 DeepSeek/Qwen-MoE/Mixtral 类
  目标：expert routing + dispatch + expert GEMM

3. 训练模型
  可以选 dense transformer training 或 diffusion/ViT
  目标：backward、optimizer、activation memory、通信/同步
```

如果资源有限，优先：

```text
dense decode inference
MoE decode inference
dense training
```

这三者对 driver/API 的压力形态差异最大。

## 七、阶段性产出节奏

### 第 1 阶段

```text
完成 trace schema
完成 API 分类
完成 kernel 分类
拿到 3 个模型 baseline trace
输出 Top API / Top kernel 初版报告
```

### 第 2 阶段

```text
完成 Top90 API cost model
完成 Top50 kernel ranking model
完成 workload signature diff 工具
```

### 第 3 阶段

```text
沉淀 5 个行为 CTS
建立 SDK version baseline
输出 dense / MoE / training 三份分析报告
```

### 第 4 阶段

```text
把报告自动化
每个 SDK 版本自动生成性能 diff
推动 driver/runtime 优化闭环
```

## 八、个人能力发力点

不需要变成纯 driver 内核工程师，也不要只停留在框架层。差异化能力应该是：

```text
模型语义 x MUSA runtime/driver x 性能建模 x 自动化分析
```

重点补强：

```text
1. GPU execution model
  stream、event、copy engine、kernel launch、graph、memory pool

2. Profiling methodology
  trace 对齐、时间线分析、critical path、overlap 分析

3. LLM inference internals
  prefill/decode、KV cache、attention variants、MoE routing

4. Performance modeling
  latency/bandwidth/FLOPs 模型、shape bucket、误差分析

5. 自动化工具能力
  trace parser、report generator、SDK regression dashboard
```

## 九、建议形成的标志性成果

可以把目标沉淀成：

```text
MUSA Model Behavior Benchmark & Performance Modeling Framework
```

它不是简单 benchmark，而是能告诉团队：

```text
SDK 新版本为什么变快/变慢；
driver API 哪些是关键瓶颈；
kernel 排序是否符合预期；
模型行为是否发生异常；
下一步优化哪个 API / kernel / runtime 策略最划算。
```

这条线做深，会比单点 API 优化更有挑战，也更容易形成个人技术影响力。
