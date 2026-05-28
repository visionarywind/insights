# MUSA Memory 设计与实现分析

> **📖 相关文档**:
> - **入门**: [memory_complete_guide.md](memory_complete_guide.md) — 一条主线串起全部机制, 从零开始
> - 逐 API 源码级分析: [memory_api_deep_analysis.md](memory_api_deep_analysis.md)
> - 同步 vs 异步对比: [muMemAlloc_vs_muMemAllocAsync.md](muMemAlloc_vs_muMemAllocAsync.md)
> - MemoryPool 设计原理: [mempool_design.md](mempool_design.md)
> - 池化技术详解: [pooling_analysis.md](pooling_analysis.md)
> - Stream/Command 架构: [stream_command_analysis.md](stream_command_analysis.md)
> - 流程决策逻辑: [decision_logic.md](decision_logic.md)
> - 26 篇分文件深度分析: [memory_api/](memory_api/README.md)

## 一、架构总览

```
用户应用 (muMemAlloc / muMemcpyHtoD / muMemsetD32 / muMemFree)
    │
    ▼
Generated Wrapper (mu_wrappers_generated.cpp — MUPTI 插桩)
    │
    ▼
Driver API 层 (mu_memory.cpp / mu_vmm.cpp / mu_mempool.cpp)
  ├── 参数校验 (nullptr? size==0? context valid?)
  ├── TLS 上下文查找
  └── 派发到 Runtime Core
    │
    ▼
Runtime Core (musa/core/)
  ├── context.cpp : Context::CreateMemory / DestroyMemory / GeneralMemcpy / GeneralMemset
  ├── memory.h/cpp : Memory 对象 (包装 Hal::IMemory)
  ├── memoryPool.h/cpp : MemoryPool 对象 (池化管理)
  ├── memoryTracker.cpp : 全局指针→内存对象的映射
  └── command/ : MemcpyCommand / MemsetCommand / PagingCommand ...
    │
    ▼
HAL 接口层 (hal/halMemory.h / halCmdBuffer.h / halMemMgr.h)
  ├── Hal::IMemory: Create, Destroy, Map, Unmap, Export, Peer
  ├── Hal::ICmdBuffer: CmdCopyMemory, CmdFillMemory, CmdWriteBuffer
  └── Hal::IMemMgr: Allocate, Free (sub-allocation)
    │
    ▼
M3D 实现层 (hal/m3d/memory.cpp / cmdBuffer.cpp)
    │
    ▼
KMD / 内核驱动 (DRM ioctl → GPU 硬件)
```

## 二、核心文件索引

### Public API Headers
```
musa/src/musa_shared_include/musa.h              (27k+ lines, 所有公开 C API)
musa/src/musa_shared_include/driver_types.h      (MUdeviceptr, MUSA_MEMCPY3D_PEER 等)
```

### Driver Layer
```
musa/src/driver/mu_memory.cpp          (2949 lines, 主内存操作: alloc/free/copy/memset/ipc)
musa/src/driver/mu_mempool.cpp         (389 lines, memory pool 操作)
musa/src/driver/mu_vmm.cpp             (471 lines, Virtual Memory Management)
musa/src/driver/mu_peer.cpp            (100 lines, peer access)
musa/src/driver/mu_wrappers_generated.cpp (22k+ lines, 自动生成的 MUPTI 插桩包装)
```

### Runtime Core
```
musa/musaMemory.h                      (IMemory 抽象接口 + MemoryType/Shape 等)
musa/core/memory.h                     (Musa::Memory 类声明)
musa/core/memory.cpp                   (Musa::Memory 实现, 8 种分配路径)
musa/core/memoryPool.h/cpp            (Musa::MemoryPool 池化管理)
musa/core/memoryTracker.cpp           (全局指针区间查找)
musa/core/externalMemory.h/cpp        (外部内存导入)
musa/core/context.cpp                 (CreateMemory/DestroyMemory/GeneralMemcpy/GeneralMemset)
musa/core/stream.cpp                  (CmdCopyMemory/CmdMemset/CmdMemAlloc/CmdMemFree/CmdPaging)
musa/core/command/memcpyCommand.h/cpp       (MemcpyCommand)
musa/core/command/AsyncMemcpyCommand.cpp    (AsyncMemcpyCommand 实现)
musa/core/command/SyncMemcpyCommand.cpp     (SyncMemcpyCommand 实现)
musa/core/command/memsetCommand.h/cpp       (MemsetCommand)
musa/core/command/memoryAtomicCommand.h/cpp (MemoryAtomicCommand)
musa/core/command/pagingCommand.h/cpp       (PagingCommand)
musa/core/copyManager2/                    (H2H/H2D/D2H/D2D 等 CPU copy 分发)
musa/core/internal_types.h                 (memcpy_kind_t 10 种方向)
```

### HAL Interface
```
hal/halMemory.h                      (Hal::IMemory 抽象接口)
hal/halMemMgr.h                      (Hal::IMemMgr sub-allocation 管理)
hal/halMemoryPool.h                  (Hal::IMemoryPool 池接口)
hal/halCmdBuffer.h                   (Hal::ICmdBuffer: CmdCopyMemory, CmdFillMemory 等)
```

### HAL M3d Implementation
```
hal/m3d/memory.h/cpp                 (M3d::Memory 实现, 8 种 Init 路径)
hal/m3d/memoryPool.h/cpp             (M3d::MemoryPool, ResSegment 空闲链表管理)
hal/m3d/memMgr.h/cpp                 (M3d::MemMgr)
hal/m3d/virtualMemory.h/cpp          (虚拟内存管理)
hal/m3d/cmdBuffer.h/cpp              (M3d::CmdBuffer, 各 engine 硬件 cmd 实现)
hal/m3d/m3d/src/core/hw/dmaCmdBuffer.h/cpp     (DMA engine)
hal/m3d/m3d/src/core/hw/xferCmdBuffer.h/cpp    (Transfer engine)
hal/m3d/m3d/src/core/hw/computeCmdBuffer.h/cpp (Compute engine)
hal/m3d/m3d/src/core/hw/mthreads/dmaip/dma1/copyCmdBuffer.h/cpp
hal/m3d/m3d/src/core/hw/mthreads/dmaip/dma2/copyCmdBuffer.h/cpp
hal/m3d/m3d/src/core/hw/mthreads/xferip/xfer1/transferCmdBuffer.h/cpp
```

### Test Files
```
musa/unittest/driver/Entry_Point_Access/muCommandAccessors.cpp  (memcpy/memset cmd 测试)
gr-umd/unittests/services/external/musa_memory_test/musa_memory_test.c (性能/带宽测试)
hal/m3d/m3d/test/createAllocation/allocation.h/cpp              (HAL 层分配测试)
hal/m3d/m3d/test/transfer/transfer.h/cpp                        (HAL 层传输测试)
```

## 三、API 分类与设计

### 1. 分配与释放 (9 种内存类型)

| API | 内存类型 | Driver 入口 | Runtime 派发 |
|-----|---------|------------|-------------|
| `muMemAlloc` | Device General | `muapiMemAlloc_v2` | `Memory::GeneralAlloc` |
| `muMemAllocPitch` | Pitched | `muapiMemAllocPitch_v2` | `Memory::PitchedGeneralAlloc` |
| `muMemAllocHost` | Pinned Host | `muapiMemHostAlloc` | `Memory::PinnedHostAlloc` |
| `muMemAllocManaged` | Managed | `muapiMemAllocManaged` | `Memory::ManagedAlloc` |
| `muMemAllocAsync` | Async Pool | `muapiMemAllocAsync` | `Stream::CmdMemAlloc` |
| `muMemAllocFromPoolAsync` | Pool Async | `muapiMemAllocFromPoolAsync` | `Stream::CmdMemAlloc` |
| `muMemFree` | Free | `muapiMemFree_v2` | `Synchronize` + `DestroyMemory` |
| `muMemFreeAsync` | Free Async | `muapiMemFreeAsync` | `Stream::CmdMemFree` |
| `muMemFreeHost` | Free Host | `muapiMemFreeHost` | `Synchronize` + `DestroyMemory` |
| `muMemHostRegister` | Register user ptr | `muapiMemHostRegister_v2` | `Memory::PinnedHostRegister` |
| `muMemHostUnregister` | Unregister | `muapiMemHostUnregister` | `DestroyMemory` |

### 2. Memcpy (15+ 种变体)

所有 memcpy **统一转为 `MUSA_MEMCPY3D_PEER` 结构体**，然后调用同一入口 `Context::GeneralMemcpy`。

| API | 方向 | 同步/异步 |
|-----|------|-----------|
| `muMemcpy` | D→D | 同步 |
| `muMemcpyHtoD` | H→D | 同步 |
| `muMemcpyDtoH` | D→H | 同步 |
| `muMemcpyDtoD` | D→D | 同步 |
| `muMemcpyPeer` | Peer D→D | 同步 |
| `muMemcpy2D` | 2D任意方向 | 同步 |
| `muMemcpy3D` | 3D任意方向 | 同步 |
| `muMemcpyAsync` | D→D | 异步 |
| `muMemcpyHtoDAsync` | H→D | 异步 |
| `muMemcpyDtoHAsync` | D→H | 异步 |
| `muMemcpyDtoDAsync` | D→D | 异步 |
| `muMemcpyBatchAsync` | 批量 | 异步 |
| `muMemoryTransferBatchAsync` | 批量传输 | 异步 |
| `muMemoryAtomicBatchAsync` | 批量原子 | 异步 |

### 3. Memset

统一转为 `MUSA_MEMSET_NODE_PARAMS` 后走 `Context::GeneralMemset`。

| API | 数据宽度 | 维度 |
|-----|---------|------|
| `muMemsetD8 / D16 / D32` | 8/16/32-bit | 1D |
| `muMemsetD2D8 / D2D16 / D2D32` | 8/16/32-bit | 2D pitched |
| `muMemsetD8Async / D16Async / D32Async` | 8/16/32-bit | 1D Async |
| `muMemsetD2D8Async / D2D16Async / D2D32Async` | 8/16/32-bit | 2D Async |

### 4. Virtual Memory Management (VMM)

| API (muapiMem*) | 作用 |
|-----------------|------|
| `AddressReserve` | 预留虚拟地址范围 |
| `AddressFree` | 释放虚拟地址 |
| `Create` | 创建物理分配 |
| `Release` | 释放分配 handle |
| `Map` | 映射物理内存到虚拟地址 |
| `Unmap` | 解映射 |
| `SetAccess` | 设置访问权限 |
| `GetAccess` | 获取访问权限 |
| `ExportToShareableHandle` | 导出为 dmabuf/fd |
| `ImportFromShareableHandle` | 导入外部内存 |
| `GetAllocationGranularity` | 查询最小粒度 |
| `GetAllocationPropertiesFromHandle` | 查询属性 |
| `RetainAllocationHandle` | 保留 handle |

### 5. Memory Pool

| API | 作用 |
|-----|------|
| `muMemPoolCreate/Destroy` | 创建/销毁池 |
| `muMemPoolSetAttribute/GetAttribute` | 池属性读写 |
| `muMemPoolSetAccess/GetAccess` | 池访问控制 |
| `muMemPoolExportPointer/ImportPointer` | 跨进程指针共享 |
| `muMemPoolExportToShareableHandle` | 池导出 |
| `muMemPoolTrimTo` | 修剪空闲内存 |

### 6. IPC / Peer / 查询

| API | 作用 |
|-----|------|
| `muIpcGetMemHandle` | 导出 IPC handle |
| `muIpcOpenMemHandle` | 导入 IPC handle |
| `muIpcCloseMemHandle` | 关闭 IPC handle |
| `muDeviceCanAccessPeer` | 查询是否支持 P2P |
| `muCtxEnablePeerAccess` | 开启 Peer 访问 |
| `muCtxDisablePeerAccess` | 关闭 Peer 访问 |
| `muMemGetInfo` | 查询空闲/总显存 |
| `muMemGetAddressRange` | 查询指针范围 |
| `muMemHostGetDevicePointer` | Host→Device 指针 |
| `muPointerGetAttribute` | 指针属性查询 |
| `muMemRangeGetAttribute` | Unified Memory 属性 |

## 四、核心实现模式

### 4.1 Memory 对象模型

```cpp
class Memory : public IMemory {
    Hal::IMemory* m_pHalMemory;    // HAL 层真实分配
    size_t        m_Offset;        // sub-allocation 偏移
    MemoryType    m_Type;          // 9 种类型之一
    MemoryShape   m_Shape;         // {widthInBytes, height, depth, pitch}
    unsigned int  m_Flags;
    void*         m_pMapped;       // CPU 映射指针 (lazy)
    MemoryTracker m_PhysTracker;   // Virtual Memory 的物理块跟踪
    MemoryPool*   m_pPool;         // 所属 pool
    Context*      m_Context;       // 所属 context
};
```

MemoryType 枚举 (musaMemory.h:24-43):
```
memoryTypeGeneral               → 通用设备内存
memoryTypePitchedGeneral        → 带 pitch 对齐的设备内存
memoryTypePinnedHost            → 页锁定主机内存
memoryTypeRegisteredPinnedHost  → 用户指针注册
memoryTypeManaged               → 托管内存
memoryTypeIpcImport             → IPC 导入
memoryTypeExternal              → 外部句柄导入
memoryTypePrealloc              → 预分配 VA
memoryTypeVirtual               → 虚拟地址
```

### 4.2 Init 分发 — Memory::Init (memory.cpp:378)

```cpp
MUresult Memory::Init(const MemoryCreateInfo& createInfo) {
    switch (m_Type) {
        case memoryTypeGeneral:            return GeneralAlloc(...);
        case memoryTypePitchedGeneral:     return PitchedGeneralAlloc(...);
        case memoryTypePinnedHost:         return PinnedHostAlloc(...);
        case memoryTypeRegisteredPinnedHost: return PinnedHostRegister(...);
        case memoryTypeManaged:            return ManagedAlloc(...);
        case memoryTypeIpcImport:          return IpcImportAlloc(...);
        case memoryTypeExternal:           return ExternalAlloc(...);
        case memoryTypePrealloc:           return PreallocAlloc(...);
        case memoryTypeVirtual:            return VirtualAlloc(...);
        default:                           return MUSA_ERROR_INVALID_VALUE;
    }
}
```

### 4.3 三层内存分配器

这是 MUSA memory 最核心的设计 — **从粗到细的三层分配**：

```
Layer 1: Hal::IMemory (裸分配)
  └── Hal::IDevice::CreateMemory(Hal::MemoryCreateInfo)  → 向 KMD 申请整块
  └── 适用于: Managed, External, IpcImport, 独立大块

Layer 2: Hal::IMemMgr (sub-allocation via pool)
  └── Hal::IMemMgr::Allocate(Hal::MemoryAllocInfo) → 从 pool 中切 sub-allocation (带 offset)
  └── 适用于: General, PitchedGeneral, PinnedHost (suballocatable 属性)

Layer 3: Musa::MemoryPool (用户可见池)
  └── 用户 muMemPoolCreate 创建, muMemAllocFromPoolAsync 分配
  └── Hal::IMemoryPool::FullAllocate 分配
```

`GeneralAlloc` (memory.cpp:462-497) 的代码展示了 sub-allocation vs 裸分配的选择：

```cpp
if (createInfo.alloc.property & Hal::memoryPropertySubAllocatable) {
    // 走 MemMgr sub-allocation (从 pool 已有大块中切出一块)
    m_Context->GetParentDevice()->Hal().GetMemMgr()->Allocate(allocInfo, &offset, &pHalMemory);
} else {
    // 直接创建独立 HAL memory (向 KMD 申请新分配)
    m_Context->GetParentDevice()->Hal().CreateMemory(createInfo, &pHalMemory);
}
```

### 4.4 GeneralAlloc 完整流程

```cpp
MUresult Memory::GeneralAlloc(size_t size, size_t alignment, unsigned int flags) {
    m_Shape = { size, 1, 1, size };
    m_Flags = flags;
    Hal::MemoryCreateInfo createInfo{};
    createInfo.type = Hal::memoryTypeAlloc;
    createInfo.alloc.type = Hal::memoryAllocTypeDeviceLocal;
    createInfo.alloc.heap = Hal::MemoryHeap::largePage;

    // ★ 属性构建: flags 为基础，叠加 Physical + SharedVA
    createInfo.alloc.property = flags;                          // Virtual | DeviceMapped | SubAllocatable
    createInfo.alloc.property |= Hal::memoryPropertyPhysical |
                                 Hal::memoryPropertySharedVirtualAddress;
    createInfo.alloc.property |= flags & Hal::memoryPropertyVirtual ?
                                 Hal::memoryPropertyDeviceVisible |
                                 Hal::memoryPropertyHostVisible |
                                 Hal::memoryPropertyHostCoherent |
                                 Hal::memoryPropertyDeviceWriteable |
                                 Hal::memoryPropertyDeviceCached : 0;

    // ★ ViewCapability: DeviceMapped → PeerAccessible + IpcExportable
    createInfo.alloc.viewCapability = Hal::memoryViewCapabilityExportable;
    createInfo.alloc.viewCapability |= (flags & Hal::memoryPropertyDeviceMapped) ?
        (Hal::memoryViewCapabilityPeerAccessible | Hal::memoryViewCapabilityIpcExportable) : 0;

    createInfo.alloc.size = size;
    createInfo.alloc.alignment = std::max(alignment, device->memAllocAlignment);

    if (createInfo.alloc.property & Hal::memoryPropertySubAllocatable)
        MemMgr->Allocate(allocInfo, &m_Offset, &m_pHalMemory);   // 子分配
    else
        Device->CreateMemory(createInfo, &m_pHalMemory);          // 裸分配
    return status;
}
```

### 4.5 PinnedHostAlloc 流程

```cpp
MUresult Memory::PinnedHostAlloc(size_t size, unsigned int flags, int numaId) {
    Hal::MemoryCreateInfo createInfo{};
    createInfo.type = Hal::memoryTypeAlloc;
    createInfo.alloc.type = Hal::memoryAllocTypeHost;  // 从主机内存分配
    createInfo.alloc.property = Hal::memoryPropertyPhysical |
                                Hal::memoryPropertySharedVirtualAddress |
                                Hal::memoryPropertyHostMapped;
    if (DEVICEMAP) {
        createInfo.alloc.property |= Hal::memoryPropertyVirtual |
                                     Hal::memoryPropertyHostVisible |
                                     Hal::memoryPropertyDeviceVisible |
                                     Hal::memoryPropertyDeviceWriteable |
                                     Hal::memoryPropertyDeviceCached;
    }
    // LargePage 失败时自动降级到 General heap
    if (heap == LargePage && status == MAP_FAILED) {
        heap = Hal::MemoryHeap::general;
        retry;
    }
}
```

### 4.6 Memcpy 统一执行路径

所有 memcpy API 的路径如下：

```
muMemcpyHtoD(dst, src, size)
  → muapiMemcpyHtoD_v2(dst, src, size)
    → GetMemcpy3DFrom1D(copy3D, dst, src, size, memcpy_host_to_device)
    → Context::GeneralMemcpy(ctx, nullptr, copy3D, true)
      → Stream::InfoStream(ctx, hStream)    // 解析 stream
      → Context::CreateMemcpyNode(...)       // 创建 GraphMemcpyNode
      → Stream::CmdCopyMemory(pGraphNode, wait)
        → new AsyncMemcpyCommand / SyncMemcpyCommand
        → ResolveDependencyAndQueueCommand
        → Command::Build
          ├── Hal::CmdBufferBegin
          ├── ResolveSubmitWait (semaphore)
          ├── CmdWriteTimestamp (可选)
          ├── CmdCopyMemoryAdvanced / CmdCopyMemory / CmdCopyImage
          └── CmdWriteTimestamp (可选)
        → Command::Submit
          ├── pHalCmdBuffer->End()
          └── SubmitToQueue(pQueue, submitInfo)
            → Hal::IQueue::Submit (进入 KMD/GPU)
```

同步 vs 异步的唯一区别：`wait=true` 时 submit 后调用 `Command::WaitFinish()` 阻塞等待 GPU 完成。

### 4.7 AsyncMemcpyCommand::BuildCopyMemory

`AsyncMemcpyCommand::BuildCopyMemory` (AsyncMemcpyCommand.cpp:162) — GPU 拷贝指令的具体构造：

```cpp
void AsyncMemcpyCommand::BuildCopyMemory(Hal::ICmdBuffer& cmdBuffer) {
    auto pSrcHalMem = srcMemory.GetHalMemory(pCurDevice);
    auto pDstHalMem = dstMemory.GetHalMemory(pCurDevice);

    Hal::CopyMemoryAdvancedParameter copyMemoryAdvanced{};
    copyMemoryAdvanced.pSrcMemory = pSrcHalMem;
    copyMemoryAdvanced.pDstMemory = pDstHalMem;
    copyMemoryAdvanced.copyRegion = Hal::MemoryCopy3DRegion{
        srcBiasBase, srcPitch, srcSlicePitch,
        dstBiasBase, dstPitch, dstSlicePitch,
        Hal::Extent3D{WidthInBytes, Height, Depth}};
    cmdBuffer.CmdCopyMemoryAdvanced(copyMemoryAdvanced);
}
```

### 4.8 MemsetCommand 流程

```
muMemsetD32(dptr, 0, 1024)
  → muapiMemsetD32_v2
    → MemsetParams = {dptr, dptr, 4096, 0, 4, 1024, 1}
    → Context::GeneralMemset(ctx, nullptr, params, true)
      → Stream::CmdMemset(node, true)
        → new MemsetCommand(this, node)
        → ResolveDependencyAndQueueCommand
        → Command::Build
          ├── CmdBufferBegin
          ├── CmdFillMemory(dst=dptr, data=0, size=4096)
          └── CmdBufferEnd
        → Command::Submit → Hal::IQueue::Submit
```

### 4.9 Memory Tracker (全局指针查找)

memoryTracker.cpp 维护一个 **区间树 (range-based map)**，将 `MUdeviceptr` 映射到 `shared_ptr<IMemory>`：

```cpp
// 注册: 分配时调用
Platform::Get().GetMemoryTracker().TrackMemory(memory_sp);

// 查找: 任何 muMemFree/muMemcpy 等 API 的第一步
pMemory = Platform::Get().GetMemoryTracker().FindRange(ptr, &offset)->get();
```

所有 API 只要传入 `MUdeviceptr`，都通过 tracker 找到对应的 `IMemory` 对象（mu_memory.cpp:115）。

### 4.10 IPC 实现

`IpcHandleInternal` (memory.h:24-44) 是 IPC 句柄的内部表示：

```cpp
struct IpcHandleInternal {
    Hal::MemoryExternalHandle halHandle;  // KMD handle
    Hal::MemoryHeap heap;                 // 内存 heap
    char pciBusId[16];                    // 设备 PCI 总线 ID
    bool fromMempool;
    size_t allocOffset;                   // sub-allocation 偏移
    size_t allocSize;                     // 分配大小
    Util::Os::Tid pid;                    // 进程 ID (防止同进程自我导入)
    uint64_t serial;                      // 唯一序列号
};
```

**Export**: 调用 `Hal::IMemory::ExportExternalHandle(info, &handle)` 获取 KMD handle，填充 PCI Bus ID + offset + size + pid + serial。

**Import**: 用 `Hal::memoryViewTypeExternal` 创建 HAL memory view，传入 `kernelManagedGlobal` 类型的 handle。重建后的 memory 指向与原进程相同的物理内存。

### 4.11 Peer Access

```cpp
// 开启 peer: muCtxEnablePeerAccess
PeerOpenInfo openInfo{};
openInfo.openType = Hal::MemoryExternalHandleType::dmaBuf;
openInfo.deviceMapped = false;
mem->Hal()->OpenPeerMemory(&peerDevice->Hal(), openInfo);

// peer memcpy: muMemcpyPeer
// 通过 MemcpyCommand 走 GPU DMA，源和目标在不同 device
// 多了一个 peer serialization: 在两个 context 之间插入 RecordCommand
if (isPeerSerial) {
    AddCurrentDependencies(command);   // 源 ctx
    peerCtx->AddCurrentDependencies(command);  // 目标 ctx
    CreateEvent + RecordCommand + WaitEvent  // 序列化
}
```

## 五、完整 API 到 GPU 的执行流程示例

以 muMemAlloc + muMemcpyHtoD + muMemsetD32 + muMemFree 的完整生命周期为例：

```
=== muMemAlloc(&dptr, 4096) ===
muapiMemAlloc_v2(dptr, 4096)
  → CreateInfo{type=General, size=4096, flags=subAllocatable|deviceMapped}
  → Context::CreateMemory (context.cpp:915)
    → new Memory(this)
    → Memory::Init → GeneralAlloc
      → Hal::GetMemMgr()->Allocate(allocInfo, &offset, &pHalMemory)
      → m_pHalMemory = pHalMemory, m_Offset = offset
    → AddMemory to context
    → MapToPeers (if peerAccessible)
    → MemoryTracker::TrackMemory
  → return dptr = pMemory->GetDevicePointer()

=== muMemcpyHtoD(dptr, host_src, 4096) ===
muapiMemcpyHtoD_v2(dptr, host_src, 4096)
  → GetMemcpy3DFrom1D(copy3D, dptr, host_src, 4096, memcpy_host_to_device)
  → Context::GeneralMemcpy(ctx, nullptr, copy3D, true)
    → CreateMemcpyNode → GraphMemcpyNode(host_to_device, ...)
    → Stream::CmdCopyMemory(node, true)
      → new AsyncMemcpyCommand(this, node)
      → ResolveDependencyAndQueueCommand
        → 插入 stream 的 m_CommandList
      → [Submit Thread] Command::Build
        → Hal::CmdBufferBegin
        → ResolveSubmitWait
        → CmdWriteTimestamp (begin)
        → BuildCopyMemory
          → Hal::CmdCopyMemoryAdvanced(src=host, dst=device, size=4096)
        → CmdWriteTimestamp (end)
      → Command::Submit
        → ResolveSubmitSignal
        → Hal::CmdBufferEnd
        → SubmitToQueue
          → Hal::IQueue::Submit(submitInfo)
            → [KMD] DRM_IOCTL_MTGPU_SUBMIT
            → [GPU] DMA engine 执行拷贝
      → Command::WaitFinish → 等待 GPU 完成

=== muMemsetD32(dptr, 0, 1024) ===
muapiMemsetD32_v2(dptr, 0, 1024)
  → MemsetParams = {dptr, 4096, 0, 4, 1024, 1}
  → Context::GeneralMemset(ctx, nullptr, params, true)
    → CreateMemsetNode
    → Stream::CmdMemset(node, true)
      → new MemsetCommand(this, node)
      → ResolveDependencyAndQueueCommand
      → Command::Build
        → CmdFillMemory(dst=dptr, data=0, size=4096)
      → Command::Submit
        → Hal::IQueue::Submit

=== muMemFree(dptr) ===
muapiMemFree_v2(dptr)
  → MemoryTracker::FindRange(dptr) → Memory*
  → Synchronize (等所有 GPU 访问完成)
  → Context::DestroyMemory
    → RemoveMemory from context
    → Memory::~Memory
      → Hal::GetMemMgr()->Free(pHalMemory, dptr, size)  // 归还 pool
      → MemoryTracker::UntrackMemory
```

## 六、HAL ICmdBuffer 内存相关接口

Hal::ICmdBuffer (halCmdBuffer.h) 定义了所有 GPU 可执行的命令：

```
CmdCopyMemory(CopyMemoryParameter)           — 1D 内存拷贝
CmdCopyMemoryAdvanced(CopyMemoryAdvancedParameter) — 3D 区域拷贝
CmdCopyImage(CopyImageParameter)             — 图像拷贝
CmdCopyMemoryToImage / CmdCopyImageToMemory  — memory↔image
CmdFillMemory(FillMemoryParameter)           — GPU memset
CmdWriteBuffer(WriteBufferParameter)         — GPU 写值 (signal)
CmdWriteTimestamp(WriteTimestampParameter)    — 写 GPU 时间戳
CmdBarrier(BarrierParameter)                 — 内存屏障
CmdWaitMemoryValue(MemoryWaitValueParam)     — 等待内存达到某值
CmdMemoryAtomic / CmdMemoryAtomicValue       — GPU 原子操作
CmdBindSpillMemoryRange                       — 绑定 spill 内存
```

## 七、CTS（单元测试）用例分析

### 7.1 muCommandAccessors.cpp 测试模式

路径: `musa/unittest/driver/Entry_Point_Access/muCommandAccessors.cpp`

典型测试 (MemcpyCommandGetSize 测试):

```cpp
TEST(TestMemcpyCommandGetSize, Test) {
    // 1. 初始化 MUPTI hook
    exportTable->MUpti->Enable(InitializeHooksForMemcpy);

    // 2. 创建 context 和 stream
    muCtxCreate(&ctx, 0, 0);
    muStreamCreate(&stream, 0);

    // 3. 分配设备内存
    muMemAlloc(&src, 1024);
    muMemAlloc(&dst, 1024);

    // 4. 执行异步 memcpy (触发 MUPTI hook 捕获 Command 指针)
    muMemcpyAsync(dst, src, 1024, stream);

    // 5. 等待完成
    muStreamSynchronize(stream);

    // 6. 验证 Command 对象被捕获且参数正确
    void* captured = g_capturedMemcpyCommand.load();
    EXPECT_NE(captured, nullptr);
    EXPECT_TRUE(g_accessorTestPassed.load());

    // 7. 清理
    muMemFree(src);
    muMemFree(dst);
    muStreamDestroy(stream);
    muCtxDestroy(ctx);
}
```

MUPTI hook 回调中验证 command 的内部参数：

```cpp
static MUpti::Context* EnterMemcpyCallbackForSize(MemcpyCommand* command) {
    g_capturedMemcpyCommand.store(command);
    // 验证 copy size
    if (command->GetCopyParams().copySize == 1024) {
        g_accessorTestPassed.store(true);
    }
    return nullptr;
}
```

### 7.2 musa_memory_test.c 性能测试

路径: `gr-umd/unittests/services/external/musa_memory_test/musa_memory_test.c`

这是一个底层的带宽测试，通过 KMD 接口直接测试:

```c
// 分配 cached / uncached / write-combined 设备内存
PVRSRVAllocDeviceMemMIW(hGeneralHeap, blockSize, alignment,
    flags | PVRSRV_MEMALLOCFLAG_CACHE_INCOHERENT, ...);
PVRSRVAllocDeviceMemMIW(hGeneralHeap, blockSize, alignment,
    flags | PVRSRV_MEMALLOCFLAG_CPU_UNCACHED, ...);
PVRSRVAllocDeviceMemMIW(hGeneralHeap, blockSize, alignment,
    flags | PVRSRV_MEMALLOCFLAG_CPU_UNCACHED_WC, ...);

// 获取 CPU 映射
MTSRVAcquireCPUMapping(devMem->hMemDesc, &cpuVirtAddr);

// 测试各种方向带宽:
// CPU(ZeroPg) → CPU(ZeroPg)       / CPU(Cached) → CPU(Cached)
// CPU(Cached) → DEV(Cached)       / DEV(Cached) → CPU(Cached)
// CPU(Cached) → DEV(Uncached)     / DEV(Uncached) → CPU(Cached)
// CPU(Cached) → DEV(Write-Combined) / DEV(Write-Combined) → CPU(Cached)
// (可选) 每块拷贝后 flush cache

// 验证数据正确性
for (int i = 0; i < blockSize; i++) {
    if (dest[i] != expected) FAIL_MEMORY(i, dest[i], expected);
}
```

## 八、设计要点总结

1. **三级内存分配**: Hal::IMemory(裸分配) → Hal::IMemMgr(sub-allocation) → Musa::MemoryPool(用户池)

2. **9 种内存类型统一框架**: 每种类型在 Memory::Init 中走不同的分配路径, 封装在同一个 Memory 类中

3. **统一 memcpy 模型**: 所有 15+ 种变体 (1D/2D/3D/HtoD/DtoH/Async/Peer) 归一化为 MUSA_MEMCPY3D_PEER 结构体, 走统一入口 GeneralMemcpy

4. **Command 机制**: 内存操作不直接执行, 而是创建 AsyncMemcpyCommand / MemsetCommand 等 Command 对象, 通过 Stream submit thread 统一提交给 HAL

5. **全局 MemoryTracker**: range-based map 实现 MUdeviceptr → IMemory* 的快速查找

6. **6 层接口**: User → Wrapper(MUPTI) → Driver → Runtime → HAL → KMD → GPU

7. **Sub-allocation**: muMemAlloc 默认使用 sub-allocation (从 pool 中切), 减少 KMD 分配次数, 提高内存碎片管理
