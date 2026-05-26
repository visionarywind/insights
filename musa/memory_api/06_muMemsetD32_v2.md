# muMemsetD32 / muMemsetD8 / muMemsetD16 — GPU 内存设置（源码逐行分析）

> 源码文件：`musa/src/driver/mu_memory.cpp` (set 函数), `musa/src/musa/core/context.cpp:733-766`

## 1. 功能概述

将设备内存设置为指定的值。提供三种粒度的变体：

| API | 单位 | 典型用途 |
|-----|------|----------|
| `muMemsetD8(dst, value, count)` | 1 字节 | 清零字节缓冲区 |
| `muMemsetD16(dst, value, count)` | 2 字节 | 半精度浮点填充 |
| `muMemsetD32(dst, value, count)` | 4 字节 | 整型/单精度浮点填充 |

所有变体最终归一化为 `MUSA_MEMSET_NODE_PARAMS`，统一由 `Context::GeneralMemset` 处理。

## 2. 调用链总览

```
muapiMemsetD32(dstDevice, value, count)
  │
  ├─ InitPlatform()                                      [internal.h:306]
  │
  ├─ 构造 MUSA_MEMSET_NODE_PARAMS:                       [Driver 层]
  │     .dst = dstDevice
  │     .value     = value (填充值, 按 elementSize 截取)
  │     .count     = count (元素个数)
  │     .pitch     = 0
  │     .height    = 1
  │     .elementSize = 4 (D32) / 2 (D16) / 1 (D8)
  │
  └─ Context::GeneralMemset(ctx, nullptr, params, true)   [context.cpp]
        │
        ├─ Platform::ValidateContext(ctx)
        ├─ Context::InfoStream(ctx, hStream)
        ├─ Context::CreateMemsetNode(params, &node, wait)
        │   └─ GraphMemsetNode::Init(wait)
        │       └─ GraphMemsetNode::MemsetLegalize(...)
        └─ Stream::CmdMemset(node, wait)
            └─ MemsetCommand::Submit()
```

## 3. Context::GeneralMemset 源码逐行分析

```cpp
// context.cpp:733
MUresult Context::GeneralMemset(Context* ctx, MUstream hStream,
                                MUSA_MEMSET_NODE_PARAMS& memsetParam,
                                bool wait)
{
    MUresult status = MUSA_SUCCESS;
    do {
        // ── Step 1: 平台/上下文校验 ──
        if (MUSA_SUCCESS != Platform::Get().ValidateContext(ctx)) {
            status = MUSA_ERROR_CONTEXT_IS_DESTROYED;
            break;
        }

        // ── Step 2: 解析 Stream ──
        Musa::Stream* stream = InfoStream(ctx, hStream);
        if (!stream) {
            status = MUSA_ERROR_INVALID_HANDLE;
            break;
        }

        // ── Step 3: 零尺寸保护 ──                       [context.cpp:750]
        // ⚠ width * height * elementSize 必须不为 0
        //   否则跳过 (静默成功), 不创建节点
        if (0 != memsetParam.width *
                  memsetParam.height *
                  memsetParam.elementSize) {

        // ── Step 4: 创建图节点 ──
            Musa::IGraphNode* pGraphNode;
            status = ctx->CreateMemsetNode(
                memsetParam, &pGraphNode, wait);
            // 调用链:
            //   Context::CreateMemsetNode
            //   → new GraphMemsetNode
            //   → GraphMemsetNode::Init(wait)
            //   → GraphMemsetNode::MemsetLegalize(...)
            //
            // 注意: wait 是引用参数。MemsetLegalize 会根据目标内存类型、
            // syncMemOps 和 Context flag 调整 wait。
            if (MUSA_SUCCESS != status) {
                break;
            }

            // ── Step 5: 提交或捕获 ──
            if (stream->GetCaptureStatus() ==
                MU_STREAM_CAPTURE_STATUS_ACTIVE) {
                // Graph Capture 模式
                status = stream->CaptureNode(
                    static_cast<GraphNode*>(pGraphNode));
            } else if (stream->GetCaptureStatus() ==
                       MU_STREAM_CAPTURE_STATUS_INVALIDATED) {
                status = MUSA_ERROR_STREAM_CAPTURE_INVALIDATED;
            } else {
                // 正常执行路径
                status = stream->CmdMemset(pGraphNode, wait);
                // [stream.cpp:721]
            }
        }
    } while(0);

    return status;
}
```

## 4. Stream::CmdMemset 源码

```cpp
// stream.cpp:721
MUresult Stream::CmdMemset(IGraphNode* pGraphNode, bool blocking)
{
    // ── 创建 MemsetCommand ──
    std::shared_ptr<Command> command =
        make_shared<MemsetCommand>(
            this,                        // Stream*
            ICast<GraphNode>(pGraphNode), // 图节点
            m_ParentCtx->PfmCheckEnable(),
            m_ParentCtx->PfmGetConfig());

    // ── 提交到 Stream ──
    return m_ParentCtx->ResolveDependencyAndQueueCommand(
        move(command), this, blocking);
}
```

## 5. MemsetCommand 底层执行

```
MemsetCommand::Execute/Submit()
  │
  ├─ Step 1: 获取参数
  │   dstAddr = graphNode->GetParams().dst + byteOffset
  │   value   = graphNode->GetParams().value
  │   count   = graphNode->GetParams().count
  │   elementSize = graphNode->GetParams().elementSize
  │
  ├─ Step 2: 大小判定与策略选择
  │   │
  │   ├─ [小尺寸 ≤ 阈值, 通常 64 bytes]:
  │   │     CPU 直写 (store 指令)
  │   │     优势: 零 DMA 引擎启动开销
  │   │
  │   └─ [大尺寸]:
  │         GPU DMA Memset 指令编码
  │         ├─ 源寄存器 = 填充值 (广播到每个 element)
  │         ├─ 目标地址 = dstDevice + offset
  │         ├─ 传输长度 = count × elementSize
  │         └─ Issue 到 compute queue
  │
  ├─ Step 3: 依赖管理
  │   前驱命令完成 → 自动触发
  │
└─ Step 4: 完成通知
        command->SetStatus(completed)
        通知等待的同步点
```

## 6. 日志验证结果

最小用例 `memory_api_callflow_demo.cpp` 打开 `MUSA_DRIVER_CALLFLOW_DEBUG=1` 后确认了 `muMemsetD32` 的逐层路径：

```text
muapiMemsetD32_v2
  -> Context::GeneralMemset
  -> Context::InfoStream
  -> Context::CreateMemsetNode
  -> Stream::CmdMemset
  -> Context::ResolveDependencyAndQueueCommand
  -> MemsetCommand::Submit
```

本次用例中，`MemsetCommand::Submit` 选择了队列提交路径：

```text
MemsetCommand::Submit
  -> Command::SubmitToQueue
```

如果命令被判定为 CPU 或 DMA 路径，则会分别进入 `CpuExecute()` 或 `DmaExecute()`。

## 7. 2D Memset (Pitch 版本)

```cpp
// Driver 入口
muapiMemsetD32Pitch(dstDevice, pitch, value, width, height)
  │
  └─ MUSA_MEMSET_NODE_PARAMS:
        .dst = dstDevice
        .value     = value
        .count     = width         ← 每行元素数
        .pitch     = pitch         ← 每行字节数 (含 padding)
        .height    = height        ← 行数
        .elementSize = 4
```

执行时按行处理:
```
for (row = 0; row < height; row++) {
    dstAddr = dstDevice + row * pitch;
    DMA_Memset(dstAddr, value, count * elementSize);
}
```

## 8. 变体参数对照

| 变体 | elementSize | value 类型 | count 含义 |
|------|-------------|-----------|-----------|
| `muMemsetD8` | 1 | uint8_t | 字节数 |
| `muMemsetD16` | 2 | uint16_t | 半字数 |
| `muMemsetD32` | 4 | uint32_t | 字数 |
| `muMemsetD32Pitch` | 4 | uint32_t | 每行字数 |

## 9. 与 memcpy 的归一化对比

```
memcpy 归一化: MUSA_MEMCPY3D_PEER
  {src/dst MemoryType, src/dst 地址, Width/Height/Depth, Pitch}
  → Context::GeneralMemcpy() → GraphMemcpyNode

memset 归一化: MUSA_MEMSET_NODE_PARAMS
  {dst 地址, value, count, pitch, height, elementSize}
  → Context::GeneralMemset() → GraphMemsetNode
```

## 10. 常见问题

### Q: 为什么 memset 也有 Graph 节点?
A: 为了与 stream 中的其他命令保持正确的依赖关系,
   也支持 capture 模式下的图录制。

### Q: 小尺寸 memset 为什么 CPU 直写?
A: GPU DMA 引擎启动有固定开销 (~几微秒)。
   对于极小的填充 (如清标志位), CPU store 指令比提交 DMA 更高效。

### Q: MemsetD8/16/32 共享同一个内核实现?
A: 是的。elementSize 作为参数传入,
   GPU 指令编码时按 elementSize 广播 value。

## 11. 相关源码位置

| 文件 | 行数 | 说明 |
|------|------|------|
| `musa/src/driver/mu_memory.cpp` | - | MemsetD8/D16/D32/D32Pitch Driver 入口 |
| `musa/src/musa/core/context.cpp` | 733-766 | `Context::GeneralMemset` |
| `musa/src/musa/core/stream.cpp` | 721-730 | `Stream::CmdMemset` |
| `musa/src/musa/core/command/memsetCommand.h` | - | MemsetCommand 类定义 |
| `musa/src/musa/core/node/graphMemsetNode.h` | - | GraphMemsetNode 节点 |
| `musa/src/driver/mu_memory.cpp` | 13-41 | 辅助函数 `GetMemcpy3DFrom1D` (memset 不使用, 仅作对比参考) |
