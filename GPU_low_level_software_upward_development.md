# GPU 底层软件如何向上发展

## 核心判断

GPU 底层软件“向上发展”的核心，不是离开 driver，而是把 driver 的能力从硬件抽象层升级成 AI workload 的系统执行层。

过去 GPU 底层软件主要回答：

```text
API 是否正确
kernel 是否能 launch
memory 是否能分配
stream/event 是否符合语义
```

下一阶段要回答：

```text
这个模型为什么慢
这个 workload 的关键路径在哪里
driver/runtime 应该给框架暴露什么能力
SDK 新版本为什么让某类模型变快/变慢
```

## 一、底层软件的发展层次

GPU 软件栈大致可以分成：

```text
硬件 / firmware
    ↓
kernel driver
    ↓
user-mode driver / runtime API
    ↓
kernel library / compiler / graph runtime
    ↓
framework backend
    ↓
LLM serving / training system
    ↓
model architecture
```

以前底层工程师主要在前 3 层工作。现在更有价值的方向是往上理解到：

```text
runtime API + kernel library + framework backend + 模型执行模式
```

也就是既懂底层资源，又懂上层 workload。

## 二、为什么必须向上走

AI workload 的瓶颈越来越不是单点 API，而是跨层问题。

例如 KV cache 慢，可能来自：

```text
allocator 问题
cache layout 问题
graph capture 被 dynamic shape 破坏
D2D copy 太多
attention kernel 访存不连续
```

MoE 慢，可能来自：

```text
routing kernel 慢
token dispatch 小 kernel 太多
expert GEMM 太碎
all-to-all 没 overlap
stream 调度没有压住 bubble
```

这些问题单看 driver API 解决不了，单看框架也解决不了。需要有人把模型行为翻译成底层资源行为。

这正是 GPU 底层软件向上发展的关键机会。

## 三、底层软件向上的 6 个发力方向

### 1. 从 API 语义走向 Workload 语义

不要只看：

```text
musaMalloc
musaMemcpyAsync
musaLaunchKernel
musaEventRecord
```

而要识别它们属于：

```text
attention prefill
decode KV append
MLP projection
MoE dispatch
expert compute
optimizer step
activation recompute
```

未来的 profiler、runtime、CTS 都应该从 API 级别升级到 workload 级别。

技术目标可以是：

```text
把 API trace 解释成模型行为 trace。
```

这是底层软件向上的第一步。

### 2. 从单 API 性能走向 Critical Path 分析

传统 driver 优化喜欢看 API 平均耗时。但 AI workload 是异步、多 stream、多 kernel、多 copy engine 的。

真正要看：

```text
哪些 API 在关键路径上
哪些 kernel 被 overlap 掉了
哪里发生 pipeline break
哪里有隐式同步
哪里 stream dependency 设计不合理
```

例如：

```text
musaMemcpyAsync 累计时间很高，但完全和 compute overlap，未必是瓶颈。

musaStreamSynchronize 只出现几次，但每次切断 decode pipeline，可能是真瓶颈。
```

向上的关键能力是：

```text
从 timeline 统计升级到执行图分析。
```

### 3. 从通用 Memory 管理走向 KV Cache / Activation / Expert Cache 管理

大模型里 memory 不再只是 `malloc/free`。

它有明确语义：

```text
KV cache
activation
workspace
temporary buffer
expert weights
optimizer states
communication buffer
graph static buffer
```

这些内存生命周期、访问模式和性能目标完全不同。

底层软件向上可以做：

```text
KV cache allocator
page/block-based cache manager
memory pool behavior modeling
long-context memory pressure analysis
prefix cache sharing
expert hot cache
activation rematerialization memory model
```

尤其是 KV cache，非常值得深挖。它是推理系统里的长期核心资源。

### 4. 从 Kernel Launch 走向 Graph / Runtime 调度

未来性能优化不是简单减少某个 kernel 时间，而是减少整个 decode step 或 training step 的调度开销。

需要关注：

```text
kernel launch overhead
graph capture
graph instantiate
graph update
graph replay
dynamic shape fallback
small kernel fusion
stream scheduling
copy/compute overlap
```

底层向上的方向是：

```text
为框架提供更稳定、更低开销、更可预测的 graph/runtime 执行能力。
```

基于 MUSA driver/API 性能建模，可以自然延伸到：

```text
graph 行为模型
dynamic-shape graph 回退分析
decode step launch density 分析
```

### 5. 从 Kernel 热点走向模型结构热点

普通 profiler 说：

```text
kernel_A 占 20%
kernel_B 占 15%
```

但上层真正想知道：

```text
attention 占多少
MLP 占多少
KV cache 占多少
MoE routing 占多少
dispatch/combine 占多少
通信占多少
```

所以底层软件要向上提供：

```text
kernel -> operator -> model block -> phase
```

的映射。

这能直接支持优化决策：

```text
如果 MLP 占主导，优化 GEMM/activation/fusion。
如果 KV cache 占主导，优化 memory layout/allocator。
如果 MoE dispatch 占主导，优化 scatter/gather/token packing。
如果 graph launch 占主导，优化 runtime 调度。
```

### 6. 从正确性 CTS 走向行为 CTS

传统 CTS 验证：

```text
API 调用是否返回正确
边界条件是否符合规范
```

AI 时代还需要验证：

```text
模型关键行为是否稳定
API 序列是否异常变化
kernel 排序是否异常变化
memory pattern 是否异常变化
stream/event 依赖是否异常变化
graph replay 是否退化
```

未来高质量 CTS 应该是小型 workload test，而不是普通 API test。

示例：

```text
Long-context KV cache CTS
MoE routing CTS
Attention graph CTS
Dynamic-shape decode CTS
Dense MLP burst CTS
Training optimizer/memory CTS
```

## 四、个人技术纵深建议

结合 MindSpore 背景和 MUSA driver 工作，建议沿这条线发展：

```text
模型语义
  ↓
框架执行图
  ↓
MUSA Runtime/API
  ↓
kernel timeline
  ↓
driver 资源调度
  ↓
性能建模与回归分析
```

这条线的稀缺性很高，因为它横跨框架和底层。

可以把个人定位成：

```text
AI workload-aware GPU runtime/driver performance engineer
```

或者更具体：

```text
MUSA 模型行为分析与性能建模负责人
```

## 五、最有挑战的 4 个方向

按价值排序，建议重点发力：

```text
1. Workload Signature
   把模型 trace 变成可 diff 的行为签名。

2. Critical Path Analyzer
   从异步 timeline 中找真正瓶颈。

3. KV Cache Memory Model
   面向长上下文推理的 memory 子系统分析。

4. MoE Runtime Behavior Model
   routing、dispatch、expert compute、combine、通信 overlap 建模。
```

这 4 个方向做出来，影响力会明显高于普通 API 优化报告。

## 六、底层软件向上的最终形态

未来 GPU driver/runtime 不应该只是暴露：

```text
malloc
memcpy
launch
stream
event
graph
```

而应该逐步具备这些能力：

```text
知道 workload 的资源使用模式
能解释 API/kernel 行为
能识别关键路径
能给框架提供调度 hint
能支持模型级 CTS 回归
能面向 SDK 版本生成性能 diff
能支撑 KV cache、MoE、graph、低精度等主流 AI 模式
```

一句话总结：

> GPU 底层软件向上发展的方向，是从 API 实现者变成 AI workload 执行系统的设计者。

当前 MUSA driver/API 性能建模与行为分析 OKR 正好可以作为切入口：先用 3 个模型把 trace、建模、行为分析、CTS、SDK diff 做成闭环。这个闭环一旦跑通，就不只是完成 OKR，而是在建立一套面向 AI workload 的 MUSA 软件栈分析方法。
