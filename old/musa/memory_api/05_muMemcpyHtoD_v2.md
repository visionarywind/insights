# muMemcpyHtoD / muMemcpy* 全系列 — 源码级调用链分析

> 源码文件：`musa/src/driver/mu_memory.cpp:13-41, 779-1156`，`musa/src/musa/core/context.cpp:699-731`
> 
> 详细内存分配链路见 `11_GeneralAlloc_deep_dive.md`。

## 1. 功能概述

所有 memcpy/memset API 最终归一化为 `MUSA_MEMCPY3D_PEER` 结构（拷贝）或 `MUSA_MEMSET_NODE_PARAMS` 结构（填充），统一由 `Context::GeneralMemcpy` / `Context::GeneralMemset` 处理。

本文件覆盖全部拷贝变体：

| API | 方向 | 类型 |
|-----|------|------|
| `muMemcpyHtoD` / `HtoD_v2` | 主机→设备 | 1D |
| `muMemcpyDtoH` / `DtoH_v2` | 设备→主机 | 1D |
| `muMemcpyDtoD` / `DtoD_v2` | 设备→设备(同设备) | 1D |
| `muMemcpyPeer` / `PeerAsync` | 设备→设备(跨设备) | 1D |
| `muMemcpy` / `Async` | 通用(UVA推断) | 1D |
| `muMemcpy2D` / `2D_v2` | 通用 | 2D |
| `muMemcpyAtoD` / `DtoA` / `AtoA` | Array变体 | 1D/Array |
| `muMemcpyBatchAsync` | 批量 | 1D |
| `muMemoryTransferBatchAsync` | 批量(图节点) | 1D |

## 2. 核心归一化函数 GetMemcpy3DFrom1D

```cpp
// mu_memory.cpp:13
static void GetMemcpy3DFrom1D(MUSA_MEMCPY3D_PEER& copy3D,
                              void* dst,
                              const void* src,
                              size_t ByteCount,
                              memcpy_kind_t copyKind)    // ← 决定方向推导
{
    // ── 源端设置 ──
    copy3D.srcXInBytes = 0;
    copy3D.srcY = 0;
    copy3D.srcZ = 0;
    copy3D.srcLOD = 0;

    switch (copyKind) {
    case memcpy_host_to_device:
        copy3D.srcMemoryType = MU_MEMORYTYPE_HOST;       // ← src 在主机
        copy3D.srcHost = src;
        copy3D.srcDevice = 0;
        copy3D.srcArray = nullptr;
        break;
    case memcpy_device_to_host:
        copy3D.srcMemoryType = MU_MEMORYTYPE_DEVICE;     // ← src 在设备
        copy3D.srcHost = nullptr;
        copy3D.srcDevice = reinterpret_cast<MUdeviceptr>(src);
        copy3D.srcArray = nullptr;
        break;
    case memcpy_device_to_device:
    case memcpy_peer_to_peer:
        copy3D.srcMemoryType = MU_MEMORYTYPE_DEVICE;
        copy3D.srcHost = nullptr;
        copy3D.srcDevice = reinterpret_cast<MUdeviceptr>(src);
        copy3D.srcArray = nullptr;
        break;
    case memcpy_default:
        // UVA 模式: 由运行时推断 (当前实现硬编码为 device)
        copy3D.srcMemoryType = MU_MEMORYTYPE_DEVICE;
        copy3D.srcHost = nullptr;
        copy3D.srcDevice = reinterpret_cast<MUdeviceptr>(src);
        copy3D.srcArray = nullptr;
        break;
    }

    // ── 目标端设置 ──
    copy3D.dstXInBytes = 0;
    copy3D.dstY = 0;
    copy3D.dstZ = 0;
    copy3D.dstLOD = 0;

    switch (copyKind) {
    case memcpy_host_to_device:
        copy3D.dstMemoryType = MU_MEMORYTYPE_DEVICE;
        copy3D.dstHost = nullptr;
        copy3D.dstDevice = reinterpret_cast<MUdeviceptr>(dst);
        copy3D.dstArray = nullptr;
        break;
    case memcpy_device_to_host:
        copy3D.dstMemoryType = MU_MEMORYTYPE_HOST;
        copy3D.dstHost = dst;
        copy3D.dstDevice = 0;
        copy3D.dstArray = nullptr;
        break;
    case memcpy_device_to_device:
    case memcpy_peer_to_peer:
        copy3D.dstMemoryType = MU_MEMORYTYPE_DEVICE;
        copy3D.dstHost = nullptr;
        copy3D.dstDevice = reinterpret_cast<MUdeviceptr>(dst);
        copy3D.dstArray = nullptr;
        break;
    case memcpy_default:
        copy3D.dstMemoryType = MU_MEMORYTYPE_DEVICE;
        copy3D.dstHost = nullptr;
        copy3D.dstDevice = reinterpret_cast<MUdeviceptr>(dst);
        copy3D.dstArray = nullptr;
        break;
    }

    // ── 维度 ──
    copy3D.srcPitch = 0;
    copy3D.srcHeight = 1;      // ⚠ 1D 拷贝 height=1
    copy3D.dstPitch = 0;
    copy3D.dstHeight = 1;
    copy3D.WidthInBytes = ByteCount;
    copy3D.Height = 1;
    copy3D.Depth = 1;

    // ── Context (用于 peer 拷贝) ──
    copy3D.srcContext = nullptr;      // 由 Peer API 填充
    copy3D.dstContext = nullptr;
}
```

## 3. 各 API 源码逐行分析

### 3.1 muMemcpyHtoD_v2

```cpp
// mu_memory.cpp:807
MUresult muapiMemcpyHtoD_v2(MUdeviceptr dstDevice,
                            const void *srcHost, size_t ByteCount)
{
    MUresult status = InitPlatform();

    if (status == MUSA_SUCCESS) {
        MUSA_MEMCPY3D_PEER copy3D{};
        // 1. 归一化为 3D 结构
        //    srcMemoryType = HOST, srcHost = srcHost
        //    dstMemoryType = DEVICE, dstDevice = dstDevice
        GetMemcpy3DFrom1D(copy3D,
            reinterpret_cast<void*>(dstDevice), srcHost,
            ByteCount, memcpy_host_to_device);
        //    ↑ copyKind = memcpy_host_to_device

        // 2. 同步调用 (wait=true)
        status = Musa::Context::GeneralMemcpy(
            TlsCtxTop(),        // ← 当前上下文
            nullptr,            // ← 默认流 (blocking)
            copy3D, true);      // ← wait=true
    }
    return status;
}
```

### 3.2 muMemcpyHtoDAsync_v2

```cpp
// mu_memory.cpp:823
MUresult muapiMemcpyHtoDAsync_v2(MUdeviceptr dstDevice,
                                 const void *srcHost,
                                 size_t ByteCount, MUstream hStream)
{
    MUresult status = InitPlatform();

    if (status == MUSA_SUCCESS) {
        MUSA_MEMCPY3D_PEER copy3D{};
        GetMemcpy3DFrom1D(copy3D,
            reinterpret_cast<void*>(dstDevice), srcHost,
            ByteCount, memcpy_host_to_device);

        // 异步调用 (wait=false)
        status = Musa::Context::GeneralMemcpy(
            TlsCtxTop(), hStream, copy3D, false);
        //                        ↑ 指定 stream, 非阻塞
    }
    return status;
}
```

### 3.3 muMemcpyDtoH_v2

```cpp
// mu_memory.cpp:839
MUresult muapiMemcpyDtoH_v2(void *dstHost, MUdeviceptr srcDevice,
                            size_t ByteCount)
{
    // 与 HtoD 对称, 方向相反
    MUSA_MEMCPY3D_PEER copy3D{};
    GetMemcpy3DFrom1D(copy3D,
        dstHost, reinterpret_cast<const void*>(srcDevice),
        ByteCount, memcpy_device_to_host);
    //  ↑ copyKind = memcpy_device_to_host

    status = Musa::Context::GeneralMemcpy(
        TlsCtxTop(), nullptr, copy3D, true);   // 同步
}
```

### 3.4 muMemcpyDtoHAsync_v2

```cpp
// mu_memory.cpp:855
// 与 HtoDAsync 对称, 仅方向和 copyKind 不同
GetMemcpy3DFrom1D(copy3D,
    dstHost, reinterpret_cast<const void*>(srcDevice),
    ByteCount, memcpy_device_to_host);

status = Musa::Context::GeneralMemcpy(
    TlsCtxTop(), hStream, copy3D, false);      // 异步
```

### 3.5 muMemcpyDtoD_v2

```cpp
// mu_memory.cpp:871
MUresult muapiMemcpyDtoD_v2(MUdeviceptr dstDevice,
                            MUdeviceptr srcDevice, size_t ByteCount)
{
    MUSA_MEMCPY3D_PEER copy3D{};
    GetMemcpy3DFrom1D(copy3D,
        reinterpret_cast<void*>(dstDevice),
        reinterpret_cast<const void*>(srcDevice),
        ByteCount, memcpy_device_to_device);

    status = Musa::Context::GeneralMemcpy(
        TlsCtxTop(), nullptr, copy3D, true);   // 同步
}
```

### 3.6 muMemcpyDtoDAsync_v2

```cpp
// mu_memory.cpp:976
// 与 DtoD 对称, 仅 async
GetMemcpy3DFrom1D(copy3D, ..., memcpy_device_to_device);
status = Context::GeneralMemcpy(ctx, hStream, copy3D, false);
```

### 3.7 muMemcpyPeer

```cpp
// mu_memory.cpp:779
MUresult muapiMemcpyPeer(MUdeviceptr dstDevice, MUcontext dstContext,
                         MUdeviceptr srcDevice, MUcontext srcContext,
                         size_t ByteCount)
{
    MUSA_MEMCPY3D_PEER copy3D{};
    GetMemcpy3DFrom1D(copy3D,
        reinterpret_cast<void*>(dstDevice),
        reinterpret_cast<const void*>(srcDevice),
        ByteCount, memcpy_peer_to_peer);    // ← peer 类型

    // ⚠ 关键差异: 设置 src/dst Context
    copy3D.srcContext = srcContext;
    copy3D.dstContext = dstContext;

    status = Musa::Context::GeneralMemcpy(
        TlsCtxTop(), nullptr, copy3D, true);
}
```

### 3.8 muMemcpyPeerAsync

```cpp
// mu_memory.cpp:793
// 与 Peer 同步版本相同, 仅 async
copy3D.srcContext = srcContext;
copy3D.dstContext = dstContext;
status = Context::GeneralMemcpy(ctx, hStream, copy3D, false);
```

### 3.9 muMemcpy (UVA 默认)

```cpp
// mu_memory.cpp:997
MUresult muapiMemcpy(MUdeviceptr dst, MUdeviceptr src,
                     size_t ByteCount)
{
    MUSA_MEMCPY3D_PEER copy3D{};
    GetMemcpy3DFrom1D(copy3D,
        reinterpret_cast<void*>(dst),
        reinterpret_cast<const void*>(src),
        ByteCount, memcpy_default);   // ← 默认模式
    // ⚠ 注释说明: 仅在支持 UVA 的上下文中有效

    status = Context::GeneralMemcpy(
        TlsCtxTop(), nullptr, copy3D, true);   // 同步
}
```

### 3.10 muMemcpy2D_v2

```cpp
// mu_memory.cpp:1234
MUresult muapiMemcpy2D_v2(const MUSA_MEMCPY2D *pCopy)
{
    MUSA_MEMCPY3D_PEER copy3D{};
    // 使用专用转换函数 (非 GetMemcpy3DFrom1D)
    GetMemcpy3DFrom2D(copy3D, *pCopy);   // [mu_memory.cpp:13]

    // 转换逻辑:
    // dstXInBytes  = pCopy->dstXInBytes
    // dstY          = pCopy->dstY
    // dstZ          = 0
    // srcXInBytes   = pCopy->srcXInBytes
    // srcY          = pCopy->srcY
    // srcZ          = 0
    // WidthInBytes  = pCopy->WidthInBytes
    // Height        = pCopy->Height
    // Depth         = 1                ← 2D 固定 depth=1
    // srcPitch/dstPitch 按原样传递

    status = Context::GeneralMemcpy(ctx, nullptr, copy3D, true);
}
```

### 3.11 Array 拷贝变体 (AtoD/DtoA/AtoA)

```cpp
// mu_memory.cpp:887 muapiMemcpyDtoA_v2
// mu_memory.cpp:916 muapiMemcpyAtoD_v2
// mu_memory.cpp:945 muapiMemcpyAtoA_v2

// 共同模式:
// 1. 获取 Array 对象 (IntrusiveCast<Array>)
// 2. 校验 offset + bytesize 不跨行
//     ⚠ offset % rowPitch + byteCount <= rowPitch (必须不跨行)
//     否则 → MUSA_ERROR_INVALID_VALUE
// 3. 用 GetMemcpy3DFrom1D 构造 copy3D
//     (Array 端地址 = array->GetDevicePtr() + offset)
// 4. Context::GeneralMemcpy(ctx, nullptr, copy3D, true)
```

### 3.12 muMemcpyBatchAsync

```cpp
// mu_memory.cpp:1021
MUresult muapiMemcpyBatchAsync(
    MUdeviceptr *dsts, MUdeviceptr *srcs, size_t *sizes,
    size_t count, MUmemcpyAttributes *attrs,
    size_t *attrsIdxs, size_t numAttrs,
    size_t *failIdx, MUstream hStream)
{
    // ── 参数校验 ──
    // failIdx 必须非空
    // count > 0, numAttrs > 0, numAttrs <= count
    // dsts/srcs/sizes/attrs/attrsIdxs 必须非空
    // attrsIdxs[0] 必须为 0

    // ── 按属性分组处理 ──
    for (size_t attrId = 0; attrId < numAttrs && success; ++attrId) {
        size_t copyEnd = (attrId+1) < numAttrs ?
            attrsIdxs[attrId+1] : count;
        // 每组共享相同的 srcAccessOrder

        for (size_t copyId = attrsIdxs[attrId];
             copyId < copyEnd; ++copyId) {
            // 逐条构造 copy3D 并提交
            GetMemcpy3DFrom1D(copy3D, ..., memcpy_default);

            switch (attrs[attrId].srcAccessOrder) {
            case MU_MEMCPY_SRC_ACCESS_ORDER_STREAM:
            case MU_MEMCPY_SRC_ACCESS_ORDER_ANY:
                Context::GeneralMemcpy(ctx, hStream,
                                       copy3D, false);    // async
                break;
            case MU_MEMCPY_SRC_ACCESS_ORDER_DURING_API_CALL:
                Context::GeneralMemcpy(ctx, hStream,
                                       copy3D, true);     // sync
                break;
            }
        }
    }
}
```

### 3.13 muMemoryTransferBatchAsync (图节点批量版)

```cpp
// mu_memory.cpp:1082
MUresult muapiMemoryTransferBatchAsync(...)
{
    // 与 memcpyBatchAsync 不同之处:
    // 1. 收集一批 {dst, src, size} 到 paramArray
    // 2. 调用 CreateMemoryTransferNode (批量创建图节点)
    //    → 相比逐条 CreateMemcpyNode 更高效
    // 3. 通过 CmdMemoryTransfer 提交 (MemoryTransferCommand)
    //    → 相比逐条 CmdCopyMemory 更高效

    // ⚠ 仅支持 MU_MEMCPY_SRC_ACCESS_ORDER_STREAM/ANY
    //   不支持 DURING_API_CALL
}
```

## 4. Context::GeneralMemcpy 调用链 (统一入口)

```cpp
// context.cpp:699
MUresult Context::GeneralMemcpy(Context* ctx, MUstream hStream,
                                MUSA_MEMCPY3D_PEER& memcpyParam,
                                bool wait)
{
    // ── Step 1: 解析 Stream ──
    Stream* pStream = InfoStream(ctx, hStream);   // [context.cpp:560]
    // NULL→默认流, MU_STREAM_LEGACY→默认流,
    // MU_STREAM_PER_THREAD→线程默认流

    // ── Step 2: 校验 Stream ──
    // pStream 必须在当前 ctx 或其 peer ctx 中注册

    // ── Step 3: 创建图节点 ──                          [context.cpp:717]
    if (memcpyParam.WidthInBytes != 0) {
        IGraphNode* pGraphNode;
        status = pStream->GetParentCtx()
            ->CreateMemcpyNode(memcpyParam, &pGraphNode, wait);

        // ── Step 4: 提交或捕获 ──
        if (status == MUSA_SUCCESS) {
            if (capture == ACTIVE) {
                pStream->CaptureNode(
                    static_cast<GraphNode*>(pGraphNode));
            } else if (capture == INVALIDATED) {
                status = MUSA_ERROR_STREAM_CAPTURE_INVALIDATED;
            } else {
                // 正常执行路径
                status = pStream->CmdCopyMemory(
                    pGraphNode, wait);                  // [stream.cpp:663]
            }
        }
    }
}
```

## 5. Stream::CmdCopyMemory 调用链

```cpp
// stream.cpp:663
MUresult Stream::CmdCopyMemory(IGraphNode* pGraphNode, bool blocking)
{
    GraphMemcpyNode* node = static_cast<GraphMemcpyNode*>(pGraphNode);

    // ── Step 1: 创建命令对象 ──
    std::shared_ptr<Command> command;
    if (node->IsAsyncMemcpyCmd(node)) {
        command = make_shared<AsyncMemcpyCommand>(
            this, ICast<GraphNode>(node), pfmCheck, pfmConfig);
    } else {
        command = make_shared<SyncMemcpyCommand>(
            this, ICast<GraphNode>(node), pfmCheck, pfmConfig);
    }

    // ── Step 2: Peer Serialization (跨设备拷贝) ──     [stream.cpp:687-716]
    IContext* pSrcCtx = node->GetCopyParams().pSrcCtx;
    IContext* pDstCtx = node->GetCopyParams().pDstCtx;
    bool isPeerSerial = (pSrcCtx != nullptr && pDstCtx != nullptr);

    if (isPeerSerial) {
        // ⚠ 跨设备拷贝需要确保源端完成后目标端再开始
        static_cast<Context*>(pSrcCtx)
            ->AddCurrentDependencies(command);
        static_cast<Context*>(pDstCtx)
            ->AddCurrentDependencies(command);
    }

    // ── Step 3: 提交到 Stream ──
    status = m_ParentCtx->ResolveDependencyAndQueueCommand(
        move(command), this, blocking);

    // ── Step 4: Peer Serialization 同步 ──              [stream.cpp:696-716]
    if (status == MUSA_SUCCESS && isPeerSerial) {
        // 创建 Event 用于同步
        status = m_ParentCtx->CreateEvent(&pEevent, 0);
        // 插入 RecordCommand (记录 event)
        auto recordCommand = make_shared<RecordCommand>(this, event);
        status = m_ParentCtx->ResolveDependencyAndQueueCommand(
            recordCommand, this, blocking);
        // 源端等待 event
        status = pSrcCtx->WaitEvent(pEevent);
        // 目标端等待 event
        status = pDstCtx->WaitEvent(pEevent);
        // 销毁 event
        status = m_ParentCtx->DestroyEvent(pEevent);
    }
}
```

## 6. Peer Serialization 机制

```
跨设备拷贝时序:

  Src Device                    Dst Device
  ──────────                    ──────────
  ┌────────┐                   ┌────────┐
  │ Memcpy │─── 提交命令 ────→  │        │
  │ Command│                   │        │
  └───┬────┘                   └───┬────┘
      │                            │
      ▼ 插入 RecordCommand         │
  ┌────────┐                      │
  │Record  │─── event 信号 ──────→│ Wait   │
  │Command │                      │ Event  │
  └───┬────┘                      └───┬────┘
      │                               │
  Src 等待完成                       Dst 开始执行
      │                               │
      ▼                               ▼
  销毁 event                      拷贝完成
```

## 7. AsyncMemcpyCommand vs SyncMemcpyCommand

```
AsyncMemcpyCommand:
  ├─ Encode() 编码 DMA 指令
  │   ├─ 编码源地址 (pinned host 或 device 指针)
  │   ├─ 编码目标地址
  │   ├─ 编码传输长度
  │   ├─ 编码方向标志 (H2D/D2H/D2D)
  │   └─ 提交到硬件队列
  │
  └─ 完成时触发 callback (通知 Stream)

SyncMemcpyCommand:
  ├─ Execute() 直接执行拷贝
  │   (同步等待拷贝完成)
  │
  └─ 适用于 blocking=true 的场景
```

## 8. 归一化汇总

```
                        GetMemcpy3DFrom1D(copy3D, dst, src, size, kind)
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           │                          │                          │
     memcpy_host_to_device     memcpy_device_to_host    memcpy_device_to_device
           │                          │                          │
     src=Host,dst=Device       src=Device,dst=Host       src=Device,dst=Device
           │                          │                          │
           ▼                          ▼                          ▼
    Context::GeneralMemcpy()   Context::GeneralMemcpy()   Context::GeneralMemcpy()
           │                          │                          │
           └──────────┬───────────────┘──────────────────────────┘
                      │
              创建 GraphMemcpyNode
                      │
              Stream::CmdCopyMemory
                      │
              AsyncMemcpy/SyncMemcpy Command
                      │
              硬件编码 + 提交
```

## 9. 相关源码位置

| 文件 | 行数 | 说明 |
|------|------|------|
| `musa/src/driver/mu_memory.cpp` | 13-41 | `GetMemcpy3DFrom1D` |
| `musa/src/driver/mu_memory.cpp` | 779-837 | Peer + HtoD + DtoH API |
| `musa/src/driver/mu_memory.cpp` | 871-974 | DtoA/AtoD/AtoA API |
| `musa/src/driver/mu_memory.cpp` | 997-1019 | `muMemcpy` UVA |
| `musa/src/driver/mu_memory.cpp` | 1021-1080 | `muMemcpyBatchAsync` |
| `musa/src/driver/mu_memory.cpp` | 1082-1156 | `muMemoryTransferBatchAsync` |
| `musa/src/musa/core/context.cpp` | 699-731 | `GeneralMemcpy` |
| `musa/src/musa/core/stream.cpp` | 663-719 | `CmdCopyMemory` + Peer Serial |
| `musa/src/musa/core/command/memcpyCommand.h` | - | AsyncMemcpyCommand/SyncMemcpyCommand |
| `musa/src/musa/core/node/graphMemcpyNode.h` | - | GraphMemcpyNode |