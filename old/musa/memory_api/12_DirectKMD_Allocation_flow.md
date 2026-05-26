# 12_DirectKMD_Allocation_flow — 裸 KMD 分配完整调用链

> 源码文件：`musa/src/hal/m3d/memory.cpp:366-426` (Linux), `musa/src/hal/m3d/memory.cpp:428-528` (Host)
> Linux DRM 实现: `musa/src/hal/m3d/m3d/src/core/os/drm/mthreads/mtgpuMemory.cpp`
> Windows WDDM2: `musa/src/hal/m3d/m3d/src/core/os/wddm/wddmGpuMemory.cpp`

## 1. 功能概述

"裸 KMD 分配"指不经过 MemoryPool 子分配、直接调用 KMD (Kernel Mode Driver) 创建 GPU 内存对象 (`GpuMemory`) 的路径。

触发条件:
- `GeneralAlloc` 中 flags **不含** `SubAllocatable`
- `PitchedGeneralAlloc` (总是走 pool, 不存在裸路径)
- `PinnedHostAlloc` LargePage→General 降级后的 fallback
- `ManagedAlloc` (固定走裸路径)

## 2. 调用链概览

```
Memory::GeneralAlloc() / ManagedAlloc() / PinnedHostAlloc()
  │
  +-- Hal::CreateMemory(createInfo, &m_pHalMemory)        [memory.cpp:494/706/600]
        │
        +-- Memory::Init(createInfo)                        [memory.cpp:115]
              │
              +-- ValidateCreateInfo()
              │
              +-- switch (alloc.type):
              │     ├─ memoryAllocTypeDeviceLocal → InitGeneralDeviceMemory()
              │     └─ memoryAllocTypeHost      → InitGeneralHostMemory()
              │
              +-- AddGpuMemoryReferences()  [Windows only]     [memory.cpp:168-176]
```

## 3. InitGeneralDeviceMemory (GPU 显存分配)

```
Memory::InitGeneralDeviceMemory(createInfo)                  [memory.cpp:366]
  │
  ├─ 1. 计算对齐:                                             [memory.cpp:373-375]
  │   alignment = max(device.capability.allocationAlignSize, user_align)
  │   alignment = AlignUp(alignment, heap.largestPageSize)
  │
  ├─ 2. 构建 GpuMemoryCreateInfo:                             [memory.cpp:378-410]
  │   size = allocInfo.size
  │   alignment = 上述计算值
  │
  │   heapAccess = GpuHeapAccessExplicit
  │   heaps[0] = GpuHeapLocal  (总是 GPU 显存)
  │   heapCount = 1
  │
│   +-- texture 堆:                          ⚠️【死代码】
   │   │     heaps[1] = GpuHeapGartUswc   (若 heap == texture)
   │   │     usage = TextureState
   │   │     heapCount = 2
   │   │
   │   +-- USC 堆:                              ⚠️【死代码】
   │   │     heaps[1] = GpuHeapGartUswc   (若 heap == usc)
   │   │     usage = ShaderProgram
   │   │     heapCount = 2
   │   │
   │   ⚠ 注意: InitGeneralDeviceMemory 只接收 memoryAllocTypeDeviceLocal 类型,
   │   │   heap 固定为 largePage, texture/USC 分支仅为 HAL 未来扩展预留.
   │   │
   │   +-- peerWritable = canMapPeerMemory
  │   │
  │   +-- flags.physicalAlloc = (!virtual) ? 1 : 0          [仅 Linux]
  │   +-- flags.discontinuousAlloc = 1                      [仅 Linux]
  │   │
  │   +-- vaRange:
  │   │     [Windows]: Default
  │   │     [Linux]:  largePage ? Svm : Default
  │   │
  │   +-- flags.svmAlloc = (property & SharedVirtualAddress) ? 1 : 0  [仅 Linux]
  │   +-- flags.globalGpuVa = unifiedAddressing
  │   +-- flags.gl2Uncached = 0
  │
  ├─ 3. GetGpuMemorySize()  (预查询所需空间)                  [memory.cpp:411]
  │     gpuMemObjSize = m3dDevice->GetGpuMemorySize(createInfo)
  │     → ioctl(mtgpuGetMemInfo)  或同等级 ioctl
  │
  ├─ 4. Reserve + CreateGpuMemory                             [memory.cpp:416-419]
  │     m_M3dGpuMemory.Reserve(gpuMemObjSize)
  │     m3dDevice->CreateGpuMemory(createInfo, ...)
  │     → ioctl(mtgpuBoAlloc)  [Linux DRM]
  │     → (Windows: D3DKMTAllocateMemory)
  │
  └─ 5. 若 hostMapped: Map(0, size, &m_pMappedHost)          [memory.cpp:420-422]
        → ioctl(mtgpuBoCpuMap)  [Linux DRM]
```

## 4. InitGeneralHostMemory (主机可访问内存)

此路径处理 `memoryAllocTypeHost` 类型, 分两个子分支:

### 4a. SharedVirtualAddress (SVM) 路径

```
if (property & SharedVirtualAddress):                       [memory.cpp:438]
  │
  +-- 分配大小含 CE 填充:
  │   allocSize = size + ceExtraPadding                     [memory.cpp:440]
  │
  +-- SvmGpuMemoryCreateInfo:                                [memory.cpp:446-472]
  │   size = AlignUp(allocSize, alignment)
  │   heap = GartUswc (WriteCombined)
  │       或 GartCacheable (Cached)
  │       或默认: snoopingHostCache ? Cacheable : Uswc
  │   flags.discontinuousAlloc = 1   [Linux]
  │   flags.numaAffinitive = supportNonUniformMemAccess
  │   flags.peerWritable = canMapHostMemory
  │   flags.globalGpuVa = unifiedAddressing
  │   flags.gl2Uncached = 1
  │
  +-- CreateSvmGpuMemory()                                   [memory.cpp:479]
        → ioctl (mtgpuCreateSvmMemory)  [Linux]
        → (Windows: 对应 WDDM2 SVM 调用)
```

**SVM 内存特点**: CPU 和 GPU 共享同一虚拟地址, 无需显式拷贝。

### 4b. 非 SVM (普通主机内存) 路径

```
else:                                                       [memory.cpp:482]
  │
  +-- GpuMemoryCreateInfo:                                   [memory.cpp:483-511]
  │   size, alignment 同上
  │   heapAccess = GpuHeapAccessExplicit
  │   heaps[0] = GartUswc / GartCacheable / 默认
  │   flags.discontinuousAlloc = 1   [Linux]
  │   flags.numaAffinitive = ...
  │   flags.peerWritable = canMapHostMemory
  │   vaRange = largePage ? Svm : Default
  │   flags.svmAlloc = 0
  │
  +-- CreateGpuMemory()                                      [memory.cpp:518]
        → ioctl(mtgpuBoAlloc)  [Linux DRM]
        → (Windows: D3DKMTAllocateMemory)
```

## 4a. InitPreallocMemory — 死代码（当前无 API 触发）

> ⚠️【死代码】`memoryTypePrealloc` 仅由 `PreallocAlloc()` 返回 `MUSA_ERROR_NOT_SUPPORTED`，
> 没有任何代码路径能通过 `Init()` 的 `memoryTypePrealloc` case 成功进入此函数。
>
> `InitPrealloc(Hal::IMemory*, size_t, size_t)`（memory.cpp:455）是独立的直接设置函数，
> 不走 `Init()` → `InitPreallocMemory()` 路径。
>
> 若未来 `InitPreallocMemory(createInfo)` 被调用:
> ```
> PreallocGpuMemoryCreateInfo preallocGpuMemoryCreateInfo = {};
> if (device 类型):
>   preallocGpuMemoryCreateInfo.gpuVirtAddr = devVA
>   preallocGpuMemoryCreateInfo.pCpuVirtAddr = 映射的 CPU VA  (仅 MT trace capture)
> else (主机类型):
>   preallocGpuMemoryCreateInfo.vaRange = Cpu
>   preallocGpuMemoryCreateInfo.pCpuVirtAddr = hostPtr
> preallocGpuMemoryCreateInfo.size = size
> → GetPreallocGpuMemorySize() + CreatePreallocGpuMemory()  (ioctl)
> ```

## 4b. InitSharedMemory — 死代码（当前无 API 触发）

> ⚠️【死代码】`memoryViewTypeShared` 没有任何公共或内部 API 设置，`Init()` 中的对应 case 永远不会被执行。
> 
> 若未来有代码调用 `CreateMemory({type: memoryViewType, view: {type: memoryViewTypeShared, ...}})`，则进入此分支。
> 
> ```
> Memory::InitSharedMemory(createInfo)                    [memory.cpp:686]
>   └─ OpenSharedGpuMemory()  ← ioctl
> ```

## 4c. InitMigratedSharedVirtualMemory — 死代码（当前无 API 触发）

> ⚠️【死代码】`memoryViewTypeMigratedSvm` 没有任何公共或内部 API 设置。
> 
> ```
> Memory::InitMigratedSharedVirtualMemory(createInfo)    [memory.cpp:646]
>   └─ CreateSvmGpuMemory()
> ```

## 4d. InitPeerMemory — 可达（通过 OpenPeerMemory）

```
Memory::InitPeerMemory(createInfo)                      [memory.cpp:710]
  │                                                      仅当 view.type == memoryViewTypePeer
  └─ 仅从 OpenPeerMemory() → CreateMemory() 调用          （MapToPeers 触发）
```

## 4e. InitExternalMemory (MMIO) — 可达（通过 HostRegister + IOMEMORY）

```
Memory::InitExternalMemory(createInfo)                  [memory.cpp:744]
  │                                                      仅当 view.external.type == mmio
  └─ 仅从 PinnedHostRegister(IOMEMORY) → ExternalAlloc() 触发  [memory.cpp:651]
```

## 5. InitLockedMemory — 可达

```
Memory::InitLockedMemory(createInfo)                    [memory.cpp:590]
  │                                                      仅当 view.type == memoryViewTypeLocked
  └─ 仅从 PinnedHostRegister(非IOMEMORY) 触发              [memory.cpp:660-665]
```

## 6. InitLockedMemory (注册用户指针)

```
Memory::InitLockedMemory(createInfo)                        [memory.cpp:590]
  │
  +-- 1. 页对齐处理:                                         [memory.cpp:598-604]
  │     alignment = VirtualPageSize()
  │     hostPtr   = ptr & ~(alignment-1)     (向下对齐)
  │     hostSize  = AlignUp(size + offset, alignment)
  │     m_Offset  = ptr & (alignment-1)     (页内偏移)
  │
  +-- 2. PinnedGpuMemoryCreateInfo:                          [memory.cpp:606-618]
  │     pSysMem = hostPtr   (用户指针所在页起始)
  │     size    = hostSize
  │     heap    = GartCacheable (支持 snoop) 或 GartUswc
  │     flags.gpuReadOnly  = READ_ONLY
  │     flags.peerWritable = canMapHostMemory
  │
  +-- 3. CreatePinnedGpuMemory()
        → ioctl(mtgpuPinMemory)   [pin 用户物理页]
```

## 7. InitExternalMemory (外部内存导入)

```
Memory::InitExternalMemory(createInfo)                      [memory.cpp:744]
  │
  +-- switch (handleType):                                   [memory.cpp:760-793]
  │     ├─ DmaBufFd:
  │     │     resourceInfo.hExternalResource = fd
  │     │     handleType = HandleType::DmaBufFd
  │     │
  │     ├─ kernelManagedGlobal:
  │     │     hExternalResource = globalHandle
  │     │     handleType = HandleType::Global
  │     │
  │     ├─ Fabric:
  │     │     hExternalResource = fabricHandle
  │     │     handleType = HandleType::Fabric
  │     │
  │     └─ Mmio (仅 Linux):
  │           base = AlignDown(mmioPtr, pageSize)
  │           size = AlignUp(mmioPtr + size, pageSize) - base
  │           m_Offset = mmioPtr - base
  │           handleType = HandleType::Mmio
  │
  +-- OpenExternalSharedGpuMemory()                         [memory.cpp:814]
        → 跨平台 ioctl
```

## 8. InitSharedMemory / InitPeerMemory

```
InitSharedMemory()  [memory.cpp:686]:
  └─ OpenSharedGpuMemory()  ← 打开同一进程内其他设备共享的内存
     ioctl(mtgpuOpenSharedMemory)

InitPeerMemory()  [memory.cpp:710]:
  └─ OpenPeerGpuMemory()    ← 打开 peer device 的内存
     ioctl(mtgpuOpenPeerMemory)
     flags.mapToGpuVa = deviceMapped
     flags.mtlinkPath  = 非 PCIE 拓扑时
```

## 9. Windows WDDM2 对比

| 操作 | Linux DRM | Windows WDDM2 |
|------|-----------|---------------|
| 分配显存 | `mtgpuBoAlloc` | `D3DKMTAllocateMemory` |
| CPU 映射 | `mtgpuBoCpuMap` | `D3DKMTLock` |
| Pin 内存 | `mtgpuPinMemory` | `D3DKMTLock` (用户指针) |
| 创建 SVM | `mtgpuCreateSvmMemory` | `D3DKMTSetVidPnSourceOwner2` |
| 导入 DMA-BUF | `mtgpuImportDmaBuf` | `D3DKMTOpenAllocationFromNtHandle` |
| Peer 访问 | `mtgpuOpenPeerMemory` | `D3DKMTOpenKeyedMutex` + `D3DKMTShareObject` |

## 10. 相关源码位置

| 文件 | 行数 | 说明 |
|------|------|------|
| `musa/src/hal/m3d/memory.cpp` | 366-426 | `InitGeneralDeviceMemory` |
| `musa/src/hal/m3d/memory.cpp` | 428-528 | `InitGeneralHostMemory` |
| `musa/src/hal/m3d/memory.cpp` | 530-588 | `InitPreallocMemory` |
| `musa/src/hal/m3d/memory.cpp` | 590-644 | `InitLockedMemory` |
| `musa/src/hal/m3d/memory.cpp` | 646-684 | `InitMigratedSharedVirtualMemory` |
| `musa/src/hal/m3d/memory.cpp` | 686-708 | `InitSharedMemory` |
| `musa/src/hal/m3d/memory.cpp` | 710-742 | `InitPeerMemory` |
| `musa/src/hal/m3d/memory.cpp` | 744-836 | `InitExternalMemory` |