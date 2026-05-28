# muMemcpyHtoD_v2 — 主机到设备内存拷贝

## 功能

将数据从主机内存拷贝到 GPU 设备内存。同步版本，等待 GPU 完成才返回。对应 CUDA `cudaMemcpyHostToDevice`。

## 完整调用链

```
用户代码: muMemcpyHtoD_v2(dstDevice, srcHost, ByteCount)
  │
  ├─ 1. mu_wrappers_generated.cpp — MUPTI 插桩
  │
  ├─ 2. mu_memory.cpp:807 — muapiMemcpyHtoD_v2
  │    ├─ InitPlatform()
  │    ├─ GetMemcpy3DFrom1D(copy3D, dst, src, size, memcpy_host_to_device)
  │    │   将 1D 参数填充到 MUSA_MEMCPY3D_PEER 结构体
  │    └─ Context::GeneralMemcpy(ctx, nullptr, copy3D, wait=true)
  │
  ├─ 3. context.cpp:699 — Context::GeneralMemcpy
  │    ├─ InfoStream(ctx, hStream=nullptr) → 默认 stream
  │    ├─ CreateMemcpyNode(copy3D, &pGraphNode, wait) → GraphMemcpyNode
  │    │    解析 copy3D 参数 → CopyParameter + CopyDirection
  │    └─ [Capture? Invalidated? 正常]
  │         → [正常] pStream->CmdCopyMemory(pGraphNode, wait=true)
  │
  ├─ 4. stream.cpp:663 — Stream::CmdCopyMemory
  │    ├─ [IsAsyncMemcpyCmd?]
  │    │   YES → new AsyncMemcpyCommand(this, node, ...)
  │    │   NO  → new SyncMemcpyCommand(this, node, ...)
  │    ├─ [Peer serial] 跨 device: AddCurrentDependencies + RecordCommand
  │    └─ ResolveDependencyAndQueueCommand(command, this, blocking=true)
  │
  ├─ 5. stream.cpp — ResolveDependencyAndQueueCommand
  │    ├─ 解析前序命令依赖关系 (m_CurrentDependencies)
  │    └─ Stream::QueueCommand(command)
  │         └─ m_CommandList.push_back(command) + notify submit thread
  │
  ├─ 6. [Submit Thread] AsyncMemcpyCommand::Build() - AsyncMemcpyCommand.cpp:27
  │    ├─ GetHalCmdBuffer(true)
  │    ├─ pHalCmdBuffer->Begin(beginInfo)
  │    ├─ Command::Build(mergingList) → 依赖解析
  │    ├─ ResolveSubmitWait → 编码 wait semaphore
  │    ├─ [Timestamp] CmdWriteTimestamp
  │    └─ switch(copyDirection):
  │         DeviceToDevice → BuildCopyMemory(*pHalCmdBuffer)
  │         DeviceToArray  → BuildCopyMemoryToImage
  │         ArrayToDevice  → BuildCopyImageToMemory
  │         ArrayToArray   → BuildCopyImage
  │
  ├─ 7. AsyncMemcpyCommand::BuildCopyMemory — AsyncMemcpyCommand.cpp:162
  │    ├─ srcMem.GetHalMemory(pCurDevice)   → HAL src memory
  │    ├─ dstMem.GetHalMemory(pCurDevice)   → HAL dst memory
  │    ├─ 计算 srcBiasBase / dstBiasBase (3D 偏移 + sub-allocation offset)
  │    └─ pHalCmdBuffer->CmdCopyMemoryAdvanced(copyMemoryAdvanced)
  │         fill: pSrcMem, pDstMem, copyRegion{...}
  │
  ├─ 8. AsyncMemcpyCommand::Submit() — AsyncMemcpyCommand.cpp:102
  │    ├─ ResolveSubmitSignal → 编码 signal semaphore
  │    ├─ pHalCmdBuffer->End()
  │    └─ SubmitToQueue(pQueue, submitInfo)
  │
  ├─ 9. [HAL] Hal::M3d::Queue::Submit
  │    ├─ Hal semaphore → M3D semaphore 转换
  │    ├─ Hal cmd buffer → M3D cmd buffer 转换
  │    ├─ 填充 IM3d::MultiSubmitInfo
  │    └─ m_M3dQueue->Submit(m3dSubmitInfo)
  │
  ├─ 10. [KMD] DRM_IOCTL_MTGPU_SUBMIT
  │     └─ GPU DMA engine 执行拷贝
  │
  └─ 11. [Sync] Command::WaitFinish()
        └─ 等待 signal semaphore → GPU 完成通知
```

## 时序图

```
应用层         Wrapper          Driver(mu_mem)     Context          Stream         Command(Async)     HAL Queue        KMD/GPU
  │             │                │                  │                │              │                  │                │
  │ muMemcpy    │                │                  │                │              │                  │                │
  │ HtoD_v2     │                │                  │                │              │                  │                │
  │────────────>│                │                  │                │              │                  │                │
  │             │                │                  │                │              │                  │                │
  │             │ muapiMemcpy    │                  │                │              │                  │                │
  │             │ HtoD_v2        │                  │                │              │                  │                │
  │             │───────────────>│                  │                │              │                  │                │
  │             │                │                  │                │              │                  │                │
  │             │                │ InitPlatform()   │                │              │                  │                │
  │             │                │ (通常已初始化)   │                │              │                  │                │
  │             │                │                  │                │              │                  │                │
  │             │                │ GetMemcpy3DFrom1D()              │              │                  │                │
  │             │                │ ───── 填充 ───── │                │              │                  │                │
  │             │                │ MUSA_MEMCPY3D    │                │              │                  │                │
  │             │                │ PEER结构体:      │                │              │                  │                │
  │             │                │ srcHost=srcHost  │                │              │                  │                │
  │             │                │ dstDevice=devPtr │                │              │                  │                │
  │             │                │ WidthInBytes=size│                │              │                  │                │
  │             │                │ Height=1,Depth=1 │                │              │                  │                │
  │             │                │ srcPitch=size    │                │              │                  │                │
  │             │                │ dstPitch=size    │                │              │                  │                │
  │             │                │ srcMemType=HOST  │                │              │                  │                │
  │             │                │ dstMemType=DEVICE│                │              │                  │                │
  │             │                │ direction=HtoD   │                │              │                  │                │
  │             │                │                  │                │              │                  │                │
  │             │                │ GeneralMemcpy    │                │              │                  │                │
  │             │                │ (ctx,null,copy3D,│                │              │                  │                │
  │             │                │  wait=true)      │                │              │                  │                │
  │             │                │─────────────────>│                │              │                  │                │
  │             │                │                  │                │              │                  │                │
  │             │                │                  │ InfoStream     │              │                  │                │
  │             │                │                  │ (ctx,nullptr)  │              │                  │                │
  │             │                │                  │─ 默认 stream ──│              │                  │                │
  │             │                │                  │<── Stream* ────│              │                  │                │
  │             │                │                  │                │              │                  │                │
  │             │                │                  │ CreateMemcpyNode()
  │             │                │                  │─ 解析 copy3D  ──> GraphMemcpyNode
  │             │                │                  │   提取拷贝参数   │ CopyParameter{copyDir,  │                │
  │             │                │                  │   创建GraphNode │ srcMem, dstMem,          │                │
  │             │                │                  │                │ srcX/Y/Z, Width,         │                │
  │             │                │                  │                │ Height, Depth, Pitch     │                │
  │             │                │                  │                │ pSrcCtx, pDstCtx}        │                │
  │             │                │                  │                │              │                  │                │
  │             │                │                  │ CmdCopyMemory  │              │                  │                │
  │             │                │                  │ (node, true)   │              │                  │                │
  │             │                │                  │───────────────>│              │                  │                │
  │             │                │                  │                │              │                  │                │
  │             │                │                  │                │ [Async?]     │                  │                │
  │             │                │                  │                │ YES: new     │                  │                │
  │             │                │                  │                │  AsyncMemcpy  │                  │                │
  │             │                │                  │                │  Command     │                  │                │
  │             │                │                  │                │  (this,node)  │                  │                │
  │             │                │                  │                │──── create ──>│                  │                │
  │             │                │                  │                │              │                  │                │
  │             │                │                  │                │ [Peer?]      │                  │                │
  │             │                │                  │                │(srcCtx!=null)│                  │                │
  │             │                │                  │                │ NO (普通拷贝) │                  │                │
  │             │                │                  │                │              │                  │                │
  │             │                │                  │                │ ResolveDeps+ │                  │                │
  │             │                │                  │                │ QueueCmd()   │                  │                │
  │             │                │                  │                │─────────────>│--SetDependencies │                │
  │             │                │                  │                │              │  (前序命令)       │                │
  │             │                │                  │                │              │--QueueCommand    │                │
  │             │                │                  │                │              │  m_CommandList   │                │
  │             │                │                  │                │              │  push_back       │                │
  │             │                │                  │                │              │  notify submit   │                │
  │             │                │                  │                │              │                  │                │
  │   ┌─────────┤ Submit Thread  │                  │                │              │                  │                │
  │   │         │                │                  │                │              │                  │                │
  │   │         │                │                  │                │              │ Build()          │                │
  │   │         │                │                  │                │              │─────────────────>│                │
  │   │         │                │                  │                │              │                  │                │
  │   │         │                │                  │                │              │ GetHalCmdBuffer  │                │
  │   │         │                │                  │                │              │──从 stream pool  │                │
  │   │         │                │                  │                │              │  取空闲 cmd buf  │                │
  │   │         │                │                  │                │              │                  │                │
  │   │         │                │                  │                │              │ Begin(beginInfo) │                │
  │   │         │                │                  │                │              │─────────────────>│--Begin cmd buf │
  │   │         │                │                  │                │              │                  │                │
  │   │         │                │                  │                │              │ ResolveSubmit    │                │
  │   │         │                │                  │                │              │ Wait(cmdBuf,     │                │
  │   │         │                │                  │                │              │  WaitBetweenCmd) │                │
  │   │         │                │                  │                │              │─────────────────>│--CmdBarrier    │
  │   │         │                │                  │                │              │  编码 wait       │  (if needed)   │
  │   │         │                │                  │                │              │  semaphore|HW    │                │
  │   │         │                │                  │                │              │                  │                │
  │   │         │                │                  │                │              │ BuildCopyMemory  │                │
  │   │         │                │                  │                │              │─────────────────>│                │
  │   │         │                │                  │                │              │                  │                │
  │   │         │                │                  │                │              │ srcMem.GetHal    │                │
  │   │         │                │                  │                │              │ Memory(device)   │                │
  │   │         │                │                  │                │              │ dstMem.GetHal    │                │
  │   │         │                │                  │                │              │ Memory(device)   │                │
  │   │         │                │                  │                │              │  (处理 peer 映射) │                │
  │   │         │                │                  │                │              │                  │                │
  │   │         │                │                  │                │              │ 计算偏移:        │                │
  │   │         │                │                  │                │              │ srcBias =        │                │
  │   │         │                │                  │                │              │  srcZ*pitch*h    │                │
  │   │         │                │                  │                │              │  +srcY*pitch     │                │
  │   │         │                │                  │                │              │  +srcX +srcOffset│                │
  │   │         │                │                  │                │              │ dstBias = ...    │                │
  │   │         │                │                  │                │              │                  │                │
  │   │         │                │                  │                │              │ CmdCopyMemAdv    │                │
  │   │         │                │                  │                │              │─────────────────>│--CmdCopyMem    │
  │   │         │                │                  │                │              │ pSrcMem,pDstMem  │  (DMA engine)  │
  │   │         │                │                  │                │              │ copyRegion{      │                │
  │   │         │                │                  │                │              │  srcOff,pitch,   │                │
  │   │         │                │                  │                │              │  slice,          │                │
  │   │         │                │                  │                │              │  dstOff,pitch,   │                │
  │   │         │                │                  │                │              │  slice,          │                │
  │   │         │                │                  │                │              │  extent{size,1,1}}               │
  │   │         │                │                  │                │              │                  │                │
  │   │         │                │                  │                │              │ Submit()         │                │
  │   │         │                │                  │                │              │─────────────────>│                │
  │   │         │                │                  │                │              │                  │                │
  │   │         │                │                  │                │              │ ResolveSubmit    │                │
  │   │         │                │                  │                │              │ Signal           │                │
  │   │         │                │                  │                │              │─────────────────>│--CmdSignal     │
  │   │         │                │                  │                │              │  编码 signal     │  semaphore     │
  │   │         │                │                  │                │              │  semaphore       │                │
  │   │         │                │                  │                │              │                  │                │
  │   │         │                │                  │                │              │ End()            │                │
  │   │         │                │                  │                │              │─────────────────>│--End cmd buf   │
  │   │         │                │                  │                │              │  → executable   │                │
  │   │         │                │                  │                │              │                  │                │
  │   │         │                │                  │                │              │ SubmitToQueue    │                │
  │   │         │                │                  │                │              │ (queue,submitInfo)               │
  │   │         │                │                  │                │              │─────────────────>│--MultiSubmit   │
  │   │         │                │                  │                │              │ ppCmdBufs        │  Info          │
  │   │         │                │                  │                │              │ waitSemaphores   │  →M3dQueue::   │
  │   │         │                │                  │                │              │ signalSemaphores │  Submit         │
  │   │         │                │                  │                │              │                  │───ioctl────────>│
  │   │         │                │                  │                │              │                  │                  │
  │   │         │                │                  │                │              │                  │                  │--GPU DMA engine
  │   │         │                │                  │                │              │                  │                  │  执行拷贝
  │   │         │                │                  │                │              │                  │                  │    src=host mem
  │   │         │                │                  │                │              │                  │                  │    dst=GPU mem
  │   │         │                │                  │                │              │                  │                  │
  │   │         │                │                  │                │              │ WaitFinish()    │                  │
  │   │         │                │                  │                │              │── 等待 signal    │                  │
  │   │         │                │                  │                │              │  semaphore      │                  │
  │   │         │                │                  │                │              │  (spin/sched_yield)              │
  │   │         │                │                  │                │              │                  │                  │
  │   │         │                │                  │                │              │                  │<── semaphore ────│
  │   │         │                │                  │                │              │<── completed ────│ signal           │
  │   │         │                │                  │                │              │                  │                  │
  │   │         │                │                  │<── OK ────────│<── OK ────────│<── OK ──────────│<── OK ───────────│
  │   │         │<── OK ─────────│<── OK ───────────│              │               │                  │                  │
  │   │<── OK ──│                │                  │              │               │                  │                  │
```

## 关键代码路径

### Driver 入口 — 统一 3D 模型

```cpp
// mu_memory.cpp:807
MUresult muapiMemcpyHtoD_v2(MUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    MUSA_MEMCPY3D_PEER copy3D{};
    // 所有 memcpy 变体都归一化为 MUSA_MEMCPY3D_PEER:
    //   srcXInBytes=0, srcY=0, srcZ=0
    //   srcMemoryType=HOST, srcHost=srcHost
    //   dstXInBytes=0, dstY=0, dstZ=0
    //   dstMemoryType=DEVICE, dstDevice=dstDevice
    //   WidthInBytes=ByteCount, Height=1, Depth=1
    //   srcPitch=ByteCount, srcHeight=1
    //   dstPitch=ByteCount, dstHeight=1
    GetMemcpy3DFrom1D(copy3D, dstDevice, srcHost, ByteCount, memcpy_host_to_device);

    // wait=true → 同步 (submit 后 WaitFinish)
    // hStream=nullptr → 默认 stream
    return Context::GeneralMemcpy(TlsCtxTop(), nullptr, copy3D, true);
}
```

### Context::GeneralMemcpy — 三路分支

```cpp
// context.cpp:699
MUresult Context::GeneralMemcpy(Context* ctx, MUstream hStream,
                                 MUSA_MEMCPY3D_PEER& memcpyParam, bool wait) {
    // 1. 解析 stream (hStream=nullptr → 默认 stream)
    pStream = InfoStream(ctx, hStream);

    // 2. 创建 GraphMemcpyNode — 包含所有拷贝参数
    ctx->CreateMemcpyNode(memcpyParam, &pGraphNode, wait);
    //    这个函数会:
    //    a) 根据 srcMemType/dstMemType 确定 CopyDirection
    //    b) 创建 GraphMemcpyNode 存参数
    //    c) 如果 peer copy, 存 pSrcCtx/pDstCtx

    // 3. 三路分支
    if (capture active)      → CaptureNode(record to graph)
    else if (capture invalid) → return error
    else                     → pStream->CmdCopyMemory(pGraphNode, wait)
}
```

### Stream::CmdCopyMemory — 创建 Command

```cpp
// stream.cpp:663
MUresult Stream::CmdCopyMemory(IGraphNode* pGraphNode, bool blocking) {
    GraphMemcpyNode* node = static_cast<GraphMemcpyNode*>(pGraphNode);

    // 根据 node 属性选择 async 还是 sync command
    std::shared_ptr<Command> command;
    if (node->IsAsyncMemcpyCmd(node)) {
        command = make_shared<AsyncMemcpyCommand>(this, node, ...);
        // Async: 走 Build → Submit 完整生命周期
        // 可合并 (mergeLevel=secondary)
    } else {
        command = make_shared<SyncMemcpyCommand>(this, node, ...);
        // Sync: 实际走相同的 Build 但 Submit 不同
    }

    // 跨 device (peer) 拷贝需要额外序列化
    if (isPeerSerial) {
        // 让两个 context 都依赖这个 command
        pSrcCtx->AddCurrentDependencies(command);
        pDstCtx->AddCurrentDependencies(command);
        // 并在拷贝后插入 event 同步
    }

    // ★ 核心: 插入 stream 命令队列
    return m_ParentCtx->ResolveDependencyAndQueueCommand(command, this, blocking);
}
```

### BuildCopyMemory — GPU 指令编码

```cpp
// AsyncMemcpyCommand.cpp:162
void AsyncMemcpyCommand::BuildCopyMemory(Hal::ICmdBuffer& cmdBuffer) {
    const auto& copyParam = GetCopyParams();

    // 获取 HAL memory (处理 peer mapping)
    auto pSrcHalMem = srcMemory.GetHalMemory(pCurDevice);
    auto pDstHalMem = dstMemory.GetHalMemory(pCurDevice);
    // GetHalMemory 会:
    //   - 同一 device: 直接返回 m_pHalMemory
    //   - peer device: 返回 m_pHalMemory->GetPeerMemory(&peerDev->Hal())

    // 计算 3D region 偏移
    size_t srcBias = srcZ * srcPitch * srcHeight
                   + srcY * srcPitch
                   + srcXInBytes
                   + srcMemory.GetOffset();  // sub-allocation offset
    size_t dstBias = dstZ * dstPitch * dstHeight
                   + dstY * dstPitch
                   + dstXInBytes
                   + dstMemory.GetOffset();

    // 填充 HAL 拷贝参数
    Hal::CopyMemoryAdvancedParameter copy{};
    copy.pSrcMemory = pSrcHalMem;
    copy.pDstMemory = pDstHalMem;
    copy.copyRegion = MemoryCopy3DRegion{
        srcBias, srcPitch, srcPitch*srcHeight,   // src offset, row, slice
        dstBias, dstPitch, dstPitch*dstHeight,   // dst offset, row, slice
        Extent3D{WidthInBytes, Height, Depth}    // 3D extent
    };

    // ★ 写入 HAL cmd buffer → GPU DMA engine 执行
    cmdBuffer.CmdCopyMemoryAdvanced(copy);
}
```

### Submit 阶段

```cpp
// AsyncMemcpyCommand.cpp:102
MUresult AsyncMemcpyCommand::Submit() {
    // 编码 signal semaphore (告诉 KMD 完成任务时要更新信号)
    ResolveSubmitSignal(pHalCmdBuffer, SignalBetweenCmd, device);

    // 结束命令缓冲区编码
    pHalCmdBuffer->End();

    // 组装 QueueSubmitInfo
    Hal::QueueSubmitInfo submitInfo{};
    submitInfo.ppCmdBuffers = &pHalCmdBuffer;
    submitInfo.cmdBufferCount = 1;
    // wait semaphores 已在 Build 阶段编码进 cmd buffer
    // signal semaphores 在 ResolveSubmitSignal 编码

    // 提交到 HAL Queue → M3D → KMD
    return SubmitToQueue(queue, submitInfo);
    // SubmitToQueue 内部:
    //   Hal::IQueue::Submit(submitInfo)
    //   → M3dQueue::Submit
    //     → m_M3dQueue->Submit(MultiSubmitInfo)
    //       → DRM_IOCTL_MTGPU_SUBMIT
}
```

## 关键设计要点

1. **1D→3D 统一模型**: 所有 memcpy 变体归一化为 `MUSA_MEMCPY3D_PEER`，1D 转 Height=1 Depth=1，2D 转 Depth=1
2. **Command 生命周期**: 拷贝不立即执行，创建 `AsyncMemcpyCommand` 插入 stream 队列，由 submit thread 统一处理
3. **同步 vs 异步**: 唯一区别是 submit 后是否调用 `WaitFinish()` 阻塞等待 signal semaphore
4. **Peer mapping 处理**: `GetHalMemory(device)` 跨 device 时返回 peer mapping 的 HAL memory
5. **Sub-allocation 偏移**: 计算拷贝地址时加上 `memory.GetOffset()`，确保 sub-allocation 内的正确偏移
6. **CPU→GPU 传输**: 在 M3D 层，`memoryAllocTypeHost` 的内存通过 PCIe/GART 由 DMA engine 搬运到 GPU 本地显存
