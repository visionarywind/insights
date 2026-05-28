# muMemsetD32_v2 — GPU Memset

## 功能

在 GPU 设备内存上设置 32-bit 值。同步版本，等待 GPU 完成后返回。对应 CUDA `cudaMemsetD32`。

## 完整调用链

```
用户代码: muMemsetD32_v2(dstDevice, 0, 1024)   // 1024 个 uint32_t = 4096 bytes
  │
  ├─ 1. mu_wrappers_generated.cpp — MUPTI 插桩
  │
  ├─ 2. mu_memory.cpp:1589 — muapiMemsetD32_v2
  │    ├─ InitPlatform()
  │    ├─ MUSA_MEMSET_NODE_PARAMS{
  │    │    dstDevice,                     // 目标地址
  │    │    dstPitch = N * sizeof(uint32_t) = 4096,  // 1D 无 pitch, 等于总字节数
  │    │    value = 0,                     // 填充值
  │    │    elementSize = sizeof(uint32_t) = 4,
  │    │    width = N = 1024,              // 元素个数
  │    │    height = 1}                    // 1D
  │    └─ Context::GeneralMemset(ctx, nullptr, params, wait=true)
  │
  ├─ 3. context.cpp:733 — Context::GeneralMemset
  │    ├─ ValidateContext(ctx) → context 销毁检查
  │    ├─ InfoStream(ctx, hStream=nullptr) → 默认 stream
  │    ├─ CreateMemsetNode(params, &pGraphNode, wait)
  │    │    解析 param → MemsetParameter + size/value/pitch/elementSize
  │    └─ [正常模式] stream->CmdMemset(pGraphNode, wait=true)
  │
  ├─ 4. stream.cpp:721 — Stream::CmdMemset
  │    ├─ new MemsetCommand(this, node, ...)
  │    └─ ResolveDependencyAndQueueCommand(command, this, blocking=true)
  │
  ├─ 5. [Submit Thread] MemsetCommand::Build
  │    ├─ GetHalCmdBuffer(true)
  │    ├─ pHalCmdBuffer->Begin(beginInfo)
  │    ├─ Command::Build(mergingList) → 依赖解析
  │    ├─ ResolveSubmitWait → 编码 wait semaphore
  │    │
  │    ├─ 分支: CpuExecute vs DmaExecute
  │    │   ├─ [小量数据] CpuExecute():
  │    │   │    直接往 m_pHalMemory→m_MappedHost 写值
  │    │   │    不走 GPU, 零调度开销
  │    │   │
  │    │   └─ [大量数据] DmaExecute():
  │    │        └─ BuildFillMemory():
  │    │             Hal::FillMemoryParameter{
  │    │               pDstMemory = halMem,
  │    │               dstOffset = offset,
  │    │               fillSize = width * elementSize,
  │    │               data = value
  │    │             }
  │    │             pHalCmdBuffer->CmdFillMemory(fillMemory)
  │    │
  │    └─ [Timestamp] CmdWriteTimestamp (可选)
  │
  ├─ 6. MemsetCommand::Submit
  │    ├─ ResolveSubmitSignal → 编码 signal semaphore
  │    ├─ pHalCmdBuffer->End()
  │    └─ SubmitToQueue(queue, submitInfo)
  │
  ├─ 7. [HAL] IQueue::Submit → M3dQueue::Submit → DRM ioctl
  │
  └─ 8. [Sync] Command::WaitFinish()
        └─ 等待 signal semaphore → GPU 完成
```

## 时序图

```
应用层          Wrapper        Driver(mu_mem)     Context         Stream         MemsetCommand      HAL Queue        KMD/GPU
  │              │              │                  │              │              │                  │                │
  │ muMemsetD32  │              │                  │              │              │                  │                │
  │─────────────>│              │                  │              │              │                  │                │
  │              │              │                  │              │              │                  │                │
  │              │ muapiMemset  │                  │              │              │                  │                │
  │              │ D32_v2       │                  │              │              │                  │                │
  │              │─────────────>│                  │              │              │                  │                │
  │              │              │                  │              │              │                  │                │
  │              │              │ InitPlatform()   │              │              │                  │                │
  │              │              │                  │              │              │                  │                │
  │              │              │ MemsetParams:    │              │              │                  │                │
  │              │              │ dst=dst,         │              │              │                  │                │
  │              │              │ pitch=4096,      │              │              │                  │                │
  │              │              │ value=0,         │              │              │                  │                │
  │              │              │ elemSize=4,      │              │              │                  │                │
  │              │              │ width=1024,      │              │              │                  │                │
  │              │              │ height=1         │              │              │                  │                │
  │              │              │                  │              │              │                  │                │
  │              │              │ GeneralMemset    │              │              │                  │                │
  │              │              │ (null,params,    │              │              │                  │                │
  │              │              │  wait=true)      │              │              │                  │                │
  │              │              │─────────────────>│              │              │                  │                │
  │              │              │                  │              │              │                  │                │
  │              │              │                  │ ValidateCtx  │              │                  │                │
  │              │              │                  │ InfoStream   │              │                  │                │
  │              │              │                  │─────────────>│              │                  │                │
  │              │              │                  │<── Stream* ──│              │                  │                │
  │              │              │                  │              │              │                  │                │
  │              │              │                  │ CreateMemsetNode()        │                  │                │
  │              │              │                  │─ 解析 params  │              │                  │                │
  │              │              │                  │  确定 size/   │              │                  │                │
  │              │              │                  │  value/engine │              │                  │                │
  │              │              │                  │              │              │                  │                │
  │              │              │                  │ CmdMemset    │              │                  │                │
  │              │              │                  │ (node, true) │              │                  │                │
  │              │              │                  │─────────────>│              │                  │                │
  │              │              │                  │              │              │                  │                │
  │              │              │                  │              │ new Memset   │                  │                │
  │              │              │                  │              │ Command      │                  │                │
  │              │              │                  │              │─────────────>│                  │                │
  │              │              │                  │              │              │                  │                │
  │              │              │                  │              │ ResolveDeps+ │                  │                │
  │              │              │                  │              │ QueueCommand │                  │                │
  │              │              │                  │              │─────────────>│--m_CommandList   │                │
  │              │              │                  │              │              │  push_back        │                │
  │              │              │                  │              │              │                  │                │
  │ ─────────────┤ Submit Thread│                  │              │              │                  │                │
  │              │              │                  │              │              │                  │                │
  │              │              │                  │              │              │ Build()          │                │
  │              │              │                  │              │              │─────────────────>│                │
  │              │              │                  │              │              │                  │                │
  │              │              │                  │              │              │ GetHalCmdBuffer  │                │
  │              │              │                  │              │              │ Begin            │                │
  │              │              │                  │              │              │ ResolveWait      │                │
  │              │              │                  │              │              │                  │                │
  │              │              │                  │              │              │ ╔══════════════╗ │                │
  │              │              │                  │              │              │ ║ CpuExecute?  ║ │                │
  │              │              │                  │              │              │ ║ 小量数据     ║ │                │
  │              │              │                  │              │              │ ║ (阈值内?)    ║ │                │
  │              │              │                  │              │              │ ║   │          ║ │                │
  │              │              │                  │              │              │ ║   ├─YES: CPU  ║ │                │
  │              │              │                  │              │              │ ║   │  直接写    ║ │                │
  │              │              │                  │              │              │ ║   │  memset(   ║ │                │
  │              │              │                  │              │              │ ║   │  mappedPtr,║ │                │
  │              │              │                  │              │              │ ║   │  value,    ║ │                │
  │              │              │                  │              │              │ ║   │  size)     ║ │                │
  │              │              │                  │              │              │ ║   │           ║ │                │
  │              │              │                  │              │              │ ║   └─NO: DMA  ║ │                │
  │              │              │                  │              │              │ ║      Build-  ║ │                │
  │              │              │                  │              │              │ ║      FillMem ║ │                │
  │              │              │                  │              │              │ ╚══════════════╝ │                │
  │              │              │                  │              │              │                  │                │
  │              │              │                  │              │              │ [DMA路径]         │                │
  │              │              │                  │              │              │ CmdFillMemory    │                │
  │              │              │                  │              │              │─────────────────>│                │
  │              │              │                  │              │              │ pDstMemory=halMem│--CmdFillMem    │
  │              │              │                  │              │              │ dstOffset=offset │  (DMA engine)  │
  │              │              │                  │              │              │ fillSize=4096    │                │
  │              │              │                  │              │              │ data=0           │                │
  │              │              │                  │              │              │                  │                │
  │              │              │                  │              │              │ Submit()          │                │
  │              │              │                  │              │              │─────────────────>│                │
  │              │              │                  │              │              │                  │                │
  │              │              │                  │              │              │ ResolveSignal    │                │
  │              │              │                  │              │              │ End              │                │
  │              │              │                  │              │              │ SubmitToQueue    │                │
  │              │              │                  │              │              │─────────────────>│--IQueue::Submit│
  │              │              │                  │              │              │                  │---ioctl-------->│
  │              │              │                  │              │              │                  │                  │--DMA引擎填充
  │              │              │                  │              │              │                  │                  │  4096 bytes 0
  │              │              │                  │              │              │                  │                  │
  │              │              │                  │              │              │ WaitFinish()     │                  │
  │              │              │                  │              │              │── spin 等待      │                  │
  │              │              │                  │              │              │  semaphore       │                  │
  │              │              │                  │              │              │                  │                  │
  │              │              │                  │              │              │                  │<── semaphore ───│
  │              │              │                  │              │              │<── completed ────│ signal           │
  │              │              │                  │              │              │                  │                  │
  │              │<── OK ───────│<── OK ───────────│<── OK ───────│<── OK ───────│<── OK ───────────│<── OK ──────────│
```

## 关键代码路径

### Driver 入口

```cpp
// mu_memory.cpp:1589
MUresult muapiMemsetD32_v2(MUdeviceptr dstDevice, unsigned int ui, size_t N) {
    // 所有 memset 统一转为 MUSA_MEMSET_NODE_PARAMS
    //
    // memset D8:  elemSize=1, value=uc
    // memset D16: elemSize=2, value=us
    // memset D32: elemSize=4, value=ui       ← 本例
    //
    // 1D: width=N, height=1, pitch=N*elemSize
    // 2D: width, height, dstPitch

    MUSA_MEMSET_NODE_PARAMS params = {
        dstDevice,
        N * sizeof(unsigned int),   // dstPitch = 总字节数
        ui,                          // value = 0
        sizeof(unsigned int),        // elementSize = 4
        N,                           // width = 1024 个元素
        1                            // height = 1
    };
    return Context::GeneralMemset(TlsCtxTop(), nullptr, params, true);
}
```

### GeneralMemset — 和 GeneralMemcpy 相同的三路分支

```cpp
// context.cpp:733
MUresult Context::GeneralMemset(Context* ctx, MUstream hStream,
                                 MUSA_MEMSET_NODE_PARAMS& memsetParam, bool wait) {
    // 1. ValidateContext(ctx) — context 是否已销毁
    // 2. InfoStream(ctx, hStream) — 解析 stream
    // 3. CreateMemsetNode(params, &pGraphNode, wait)
    //    创建 GraphMemsetNode, 解析:
    //      memset 的 dst 内存对象 (通过 memoryTracker),
    //      计算填充偏移, 设置 engine 类型
    // 4. 三路:
    //    [Capture active] → CaptureNode
    //    [Capture invalid] → 错误
    //    [正常] → stream->CmdMemset(pGraphNode, wait)
}
```

### CpuExecute vs DmaExecute

```cpp
// MemsetCommand (伪代码)
MUresult MemsetCommand::BuildFillMemory(Hal::ICmdBuffer& cmdBuffer) {
    const size_t totalSize = m_NodeParams.width *
                             m_NodeParams.height *
                             m_NodeParams.elementSize;

    // CPU 路径: 直接写 host 可访问的内存 (GPU 映射的 host memory)
    // 适用: 小量数据, 避免 GPU 调度开销
    if (totalSize <= CPU_EXEC_THRESHOLD && cpuExecSupported) {
        CpuExecute();  // 直接 memset(mappedHost + offset, value, size)
        return;
    }

    // DMA 路径: 通过 GPU DMA engine 执行
    // 适用: 大量数据, 利用 GPU 高带宽
    // 或者: 设备不支持 CPU 直接写
    DmaExecute(cmdBuffer);
}

MUresult MemsetCommand::DmaExecute(Hal::ICmdBuffer& cmdBuffer) {
    // memoryTypePinnedHost / General: 从 memoryTracker 找 dstMemory
    auto dstMemory = Platform::Get().GetMemoryTracker()
                     .FindRange(params.dstDevice, &offset);

    // 构造 FillMemoryParameter
    Hal::FillMemoryParameter fillMemory{};
    fillMemory.pDstMemory = dstMemory->Hal();        // HAL memory
    fillMemory.dstOffset  = offset;                   // sub-allocation offset
    fillMemory.fillSize   = width * elementSize;       // 填充字节数
    fillMemory.data       = value;                     // 填充值

    // ★ 写入 HAL cmd buffer → GPU DMA engine 执行
    cmdBuffer.CmdFillMemory(fillMemory);
}
```

### MemsetCommand::Submit

```cpp
MUresult MemsetCommand::Submit() {
    // 如果走的是 CPU 路径, 不需要真正 submit
    if (m_WasCpuExecuted) return MUSA_SUCCESS;

    // GPU 路径: 正常提交
    pHalCmdBuffer = GetHalCmdBuffer(false);
    ResolveSubmitSignal(pHalCmdBuffer, SignalBetweenCmd, device);
    pHalCmdBuffer->End();

    Hal::QueueSubmitInfo submitInfo{};
    submitInfo.ppCmdBuffers = &pHalCmdBuffer;
    submitInfo.cmdBufferCount = 1;
    return SubmitToQueue(queue, submitInfo);
    // 同步 version: submit 后执行 WaitFinish()
}
```

## 与 muMemcpy 的异同

| 特性 | muMemsetD32 | muMemcpyHtoD |
|------|-------------|--------------|
| GPU 指令 | `CmdFillMemory` | `CmdCopyMemoryAdvanced` |
| CPU 优化路径 | 小量数据直接 CPU memset | 无 CPU 路径 (总需硬件 DMA) |
| 参数结构 | `MUSA_MEMSET_NODE_PARAMS` | `MUSA_MEMCPY3D_PEER` |
| 归一化 | 统一转为 memset_params | 统一转为 copy3D |
| engine | cdm/ce (取决于路径) | dma/ce/tdm (取决于方向) |
| 命令类型 | `MemsetCommand` | `AsyncMemcpyCommand/SyncMemcpyCommand` |
| 2D 支持 | `muMemsetD2D*` → width/height/pitch | `muMemcpy2D` → `MUSA_MEMCPY2D` |

## 关键设计要点

1. **CPU vs GPU 双路径**: 小量 memset 直接在 CPU 端写入 host 可映射内存，减少 GPU 调度延迟；大量数据走 DMA engine 利用 GPU 带宽
2. **统一参数模型**: 所有 memset 变体（D8/D16/D32/D2D）统一转为 `MUSA_MEMSET_NODE_PARAMS`
3. **1D/2D 同一指令**: 1D 和 2D 都最终调用 `CmdFillMemory`，2D 的 pitch 参数通过 `s_PerTDMFillMaxSize` 约束确保不超过硬件 DMA 限制
4. **无 Command 分支**: 与 memcpy 不同，memset 只有一个 `MemsetCommand`，没有 async/sync 之分（区别仅在于 submit 后是否 wait）
