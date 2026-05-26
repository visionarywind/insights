# DCSim、SPM-E2E、Aic 与 GPU Driver 仿真差异

相关文档：

- [`gpu_driver_replay_design.md`](gpu_driver_replay_design.md)：说明 GPU Driver 仿真如何通过 trace replay 复现资源、同步、提交和 launch 上下文问题。

## 总体分层

四类仿真覆盖的层级不同。可以按从上到下的执行链路理解：

```text
模型 / 框架 / Runtime
        |
        v
GPU Driver 仿真
        |
        v
DCSim 设备级仿真
        |
        v
Aic kernel / AI Core 级仿真

SPM-E2E 仿真横向覆盖端到端链路，用于评估整体性能趋势。
```

核心区别如下：

| 仿真类型 | 主要对象 | 覆盖范围 | 精度 | 速度 | 典型用途 |
|---|---|---:|---:|---:|---|
| SPM-E2E 仿真 | 模型端到端执行 | 最大 | 中等 | 快 | 评估整体吞吐、延迟、瓶颈分布 |
| GPU Driver 仿真 | Runtime 到设备之间的驱动行为 | 较大 | 中高 | 中等 | 分析提交、内存、同步、queue、fence、ioctl 开销 |
| DCSim 仿真 | 设备侧任务执行 | 中等 | 较高 | 较慢 | 分析设备调度、访存、资源冲突、任务执行时序 |
| Aic 仿真 | 单 kernel / AI Core 内部执行 | 最小 | 最高 | 最慢 | 验证 kernel 正确性、指令流水、片上存储访问 |

## SPM-E2E 仿真

SPM-E2E 仿真关注完整业务链路的性能。它通常不精确模拟每条指令，而是用性能模型描述模型推理或训练过程中的主要耗时。

典型输入：

- 模型结构
- batch size
- sequence length
- 算子耗时模型
- 通信耗时模型
- 并行策略
- 调度策略

典型输出：

- 端到端延迟
- token 吞吐
- prefill / decode 时间占比
- attention / MLP / MoE / 通信耗时占比
- 设备利用率趋势
- 性能瓶颈位置

适合回答的问题：

- 一个模型在目标硬件上大概能达到多少吞吐。
- batch size 或 sequence length 变化后，整体性能如何变化。
- 端到端耗时主要消耗在计算、通信还是调度。
- 模型结构调整后，整体性能趋势是否改善。

局限：

- 通常不精确描述 driver submit、fence wait、ioctl 等细节。
- 通常不精确描述单个 kernel 内部指令执行。
- 对小 kernel 过多、同步过多、driver 开销明显的场景，需要结合 GPU Driver 仿真补充。

## DCSim 仿真

DCSim 仿真关注设备侧任务如何执行。它比 SPM-E2E 更接近硬件执行过程，但通常不进入单 kernel 的每条指令细节。

典型输入：

- command stream
- kernel task 描述
- DMA / memory copy task
- stream / queue 依赖关系
- 设备资源模型
- 访存模型

典型输出：

- 设备侧任务时间线
- queue 执行顺序
- kernel / DMA overlap 情况
- device idle 时间
- 访存拥塞情况
- 资源冲突情况
- 多 stream 调度行为

适合回答的问题：

- 任务在设备上是否并行执行。
- queue 是否存在阻塞。
- DMA 和 kernel 是否能 overlap。
- 访存是否成为瓶颈。
- 多 stream 或多 context 调度是否合理。

局限：

- 需要上游提供准确的任务序列或 command stream。
- 对 Runtime API 到 command buffer 的生成过程覆盖不足。
- 对单 kernel 内部计算细节覆盖不足，需要结合 Aic 仿真。

## Aic 仿真

Aic 仿真关注单个 kernel 或 AI Core 内部执行。它是精度最高、速度最慢的一层。

典型输入：

- kernel binary 或 kernel IR
- launch grid / block 配置
- kernel 参数
- 输入输出 buffer
- tensor shape / stride / layout
- shared memory / local memory 配置

典型输出：

- kernel 计算结果
- 指令级执行时序
- cycle 估计
- 片上存储访问情况
- load / store 行为
- barrier / sync 行为
- kernel 内部错误信息

适合回答的问题：

- 某个 kernel 计算结果是否正确。
- kernel 内部访存是否合理。
- 指令流水是否存在明显 stall。
- local memory、shared memory、寄存器使用是否合理。
- 单 kernel 优化是否有效。

局限：

- 只看单 kernel，不适合直接评估完整模型端到端性能。
- 不负责 Runtime、driver、queue、fence 等系统级行为。
- 需要准确的 kernel 上下文，由 GPU Driver 仿真或真实 trace 提供更合适。

## GPU Driver 仿真

GPU Driver 仿真位于 Runtime 和设备执行之间。它不直接替代 SPM-E2E、DCSim 或 Aic，而是补齐驱动层行为。

重点仿真的内容：

- context 创建和销毁
- device capability 查询
- 显存分配、释放、映射
- VA / PA 地址管理
- command buffer 构造
- kernel launch 参数封装
- stream / queue 提交
- ioctl 调用顺序
- fence / event / sync 等待
- 多 stream 依赖关系
- 多进程、多 context、优先级调度
- 中断、超时、reset、page fault 等异常路径

典型输入：

- Runtime API trace
- kernel launch trace
- memory allocation trace
- memcpy / DMA trace
- stream / event / fence trace
- ioctl trace
- command buffer dump

典型输出：

- API timeline
- ioctl timeline
- stream / queue timeline
- command buffer 列表
- buffer 生命周期
- fence / event 依赖图
- kernel launch 参数
- memcpy / DMA 记录
- host submit gap
- device idle 原因
- 同步等待原因统计
- 异常路径复现结果

适合回答的问题：

- Runtime 到 driver 的提交开销有多大。
- 小 kernel 是否因为 launch overhead 过多影响端到端性能。
- stream synchronize、event wait、fence wait 是否导致 device idle。
- command buffer 是否构造正确。
- memory object 生命周期是否正确。
- 多 stream 依赖是否完整。
- ioctl 调用是否过多。
- page fault、timeout、reset 等异常路径是否可复现。

## GPU Driver 仿真如何补充现有工具

### 补充 SPM-E2E

SPM-E2E 主要关注整体性能模型，但真实执行中还有 driver 层开销：

```text
Runtime API
  -> 参数检查
  -> UMD 构造 command buffer
  -> ioctl submit
  -> KMD 入队
  -> fence / event 等待
  -> device 执行
```

GPU Driver 仿真可以把这些开销加入端到端分析：

- kernel launch overhead
- memcpy submit overhead
- stream synchronize overhead
- command buffer 构造成本
- ioctl 次数和耗时
- event / fence 等待时间
- 小算子过多导致的提交开销

### 补充 DCSim

DCSim 需要设备侧任务序列。GPU Driver 仿真可以从 Runtime API 或 command buffer 中生成更接近真实执行的输入：

```text
Runtime API trace
  -> GPU Driver 仿真
  -> command stream
  -> DCSim
```

这样 DCSim 可以分析更完整的问题：

- 任务是如何被 driver 提交的。
- stream 依赖是否正确。
- command buffer 是否合理。
- memory object 生命周期是否正确。
- 是否存在不必要的同步。
- submit 粒度是否过细。

### 补充 Aic

Aic 需要准确的 kernel 上下文。GPU Driver 仿真可以从 launch packet 中提取：

- kernel binary
- grid / block
- kernel args
- buffer address
- tensor shape
- tensor layout
- workspace
- stream / context

连接方式：

```text
kernel launch packet
  -> 提取 kernel 上下文
  -> Aic 仿真
  -> 返回 kernel 结果 / cycle / error
```

这样 Aic 可以放回真实 launch 场景中验证，而不是孤立运行单 kernel。

## 建议建设路径

### 阶段一：Trace Replay

先记录真实运行中的关键事件，再离线 replay。

建议记录：

- malloc / free
- memcpy / DMA
- kernel launch
- event record / wait
- stream synchronize
- ioctl submit
- fence wait

示例 trace：

```json
{
  "op": "kernel_launch",
  "stream": 3,
  "kernel": "attention_kernel",
  "grid": [128, 1, 1],
  "block": [256, 1, 1],
  "args": ["0x1000", "0x2000"],
  "deps": [41, 42]
}
```

优势：

- 实现成本低。
- 便于快速分析真实 workload。
- 不需要一开始就完整模拟 UMD / KMD。

### 阶段二：Mock UMD / ioctl

在 Runtime 和 driver 之间放置 mock 层：

```text
Runtime
  -> Mock UMD
  -> Fake ioctl
  -> Driver Simulator
```

需要覆盖：

- create context
- allocate buffer
- map / unmap buffer
- submit command buffer
- wait fence
- query device info
- destroy resource

这一阶段可以让部分 runtime 测试在没有真实 GPU 的情况下执行。

### 阶段三：Command Buffer 级仿真

解析 command buffer packet：

```text
command buffer
  -> packet parser
  -> kernel launch packet
  -> DMA packet
  -> sync packet
  -> queue model
```

可分析的问题：

- command buffer 是否合法。
- packet 顺序是否正确。
- dependency 是否完整。
- memory 地址是否有效。
- fence 是否会死等。
- queue 是否存在 head-of-line blocking。

### 阶段四：接入 DCSim / Aic / SPM-E2E

GPU Driver 仿真负责分发和汇总：

```text
kernel launch packet
  -> Aic 仿真单 kernel

device task stream
  -> DCSim 仿真设备调度和访存

完整 workload timeline
  -> SPM-E2E 汇总端到端性能
```

## 最终工具关系

```text
SPM-E2E
  使用整体 workload timeline，评估端到端吞吐和延迟。

GPU Driver 仿真
  还原 Runtime 到设备之间的提交、内存、同步、queue 和 command buffer 行为。

DCSim
  使用 driver 生成的 device task stream，分析设备侧调度和访存。

Aic
  使用 driver 提取的 kernel launch 上下文，分析单 kernel 内部执行。
```

## 结论

SPM-E2E 用于全局性能评估；DCSim 用于设备级执行分析；Aic 用于 kernel 级精细验证；GPU Driver 仿真用于补齐 Runtime 到设备之间的提交、内存、同步和调度行为。

四类工具组合后，可以覆盖从模型端到端性能、driver 提交开销、设备任务调度到单 kernel 内部执行的完整链路。
