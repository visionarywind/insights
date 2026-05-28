# 直接 KMD 分配显存 — 完整执行流程深度分析

> 当 `memoryPropertySubAllocatable` **未设置**时，`GeneralAlloc` 走裸分配路径，每次 `muMemAlloc` 都直接向 KMD 发起 ioctl。本文逐层拆解完整调用链。

---

## 一、功能概述

| 维度 | 直接 KMD 裸分配 |
|------|----------------|
| **入口 API** | `muMemAlloc`（flags 不含 `SubAllocatable`） |
| **核心路径** | `Hal::Device::CreateMemory` → `GpuMemory::Init` → `AllocateOrPinMemory` |
| **KMD 调用次数** | 每次分配至少 **2 次 ioctl**（AllocBuffer + MapVirtualAddress） |
| **分配粒度** | 精确按请求大小（对齐后） |
| **适用场景** | 大分配（≥64KB）、特殊 flags、不走 pool 的场景 |
| **与 SubAlloc 区别** | SubAlloc 从预分配的 64KB chunk 中切分，小分配零 KMD 调用 |

---

## 二、完整调用链

```
用户态 (User API)
─────────────────────────────────────────────────────────────
muapiMemAlloc_v2                          driver/mu_memory.cpp:265
  │  构造 CreateInfo(flags = Virtual|DeviceMapped，不含 SubAllocatable)
  └─ Context::CreateMemory                context.cpp:915
       │  Step1: Stream Capture 检查
       │  Step2: Memory::Init             memory.cpp:115
       │       └─ GeneralAlloc            memory.cpp:462
       │            └─ [!SubAllocatable]  memory.cpp:493-494
       │                 └─ Hal::Device::CreateMemory   device.cpp:169
       │                      └─ new Memory(device)
       │                      └─ Memory::Init(createInfo)
       │                           └─ InitGeneralDeviceMemory  memory.cpp:366
       │                                └─ 构造 GpuMemoryCreateInfo
       │                                └─ m3dDevice->CreateGpuMemory  device.cpp:1168
       │                                     └─ new GpuMemory
       │                                     └─ GpuMemory::Init(gpuMemory.cpp:539)
       │                                          └─ Heap 选择 / 对齐 / flags 设置
       │                                          └─ AllocateOrPinMemory()
       │                                               │
       └─ Step3: MapToPeers                context.cpp:162
       └─ Step4: TrackMemory + 注册        context.cpp:628-629

HAL 层 (Hal::M3d)
─────────────────────────────────────────────────────────────
Device::CreateMemory                      device.cpp:169
  └─ Memory::Init                          memory.cpp:115
       └─ InitGeneralDeviceMemory          memory.cpp:366
            └─ Device::CreateGpuMemory     device.cpp:1168
                 └─ GpuMemory::Init        gpuMemory.cpp:539
                      └─ AllocateOrPinMemory()

OS 层 (Platform Specific)
─────────────────────────────────────────────────────────────
Linux DRM (mtgpuMemory.cpp):
  AllocateOrPinMemory                      mtgpuMemory.cpp:387
    ├─ ① AssignVirtualAddress()            mtgpuMemory.cpp:456
    │    └─ pDevice->AssignVirtualAddress()
    ├─ ② AllocBuffer()                     mtgpuMemory.cpp:687
    │    └─ pDevice->AllocBuffer()          mtgpuDevice.cpp:962
    │         └─ mtgpuBoAlloc() ioctl ←──── KMD 入口
    └─ ③ MapVirtualAddress()               mtgpuMemory.cpp:705
         └─ pDevice->MapVirtualAddress()    mtgpuDevice.cpp:1401
              └─ mtgpuBoVmMapV2() ioctl ←── VA→PA 映射

Windows WDDM2 (wddmGpuMemory.cpp):
  AllocateOrPinMemory                      wddmGpuMemory.cpp:549
    ├─ ① ReserveGpuVirtualAddress()         wddmGpuMemory.cpp:619
    ├─ ② AllocateOrPinMemoryInternal()      wddmGpuMemory.cpp:628
    │    └─ Thunk::CreateAllocation()       → KMD ioctl
    └─ ③ AcquireGpuVirtualAddress()         wddmGpuMemory.cpp:646
```

---

## 三、ASCII 时序图

```
muapiMemAlloc_v2                           Driver                      Context                     Memory                         HAL/M3D                       KMD
   │                                        │                           │                            │                              │                          │
   │ muapiMemAlloc_v2                        │                           │                            │                              │                          │
   │────────────────────────────────────────>│                           │                            │                              │                          │
   │                                        │ CreateMemory(createInfo)  │                            │                              │                          │
   │                                        │──────────────────────────>│                            │                              │                          │
   │                                        │                           │ Init(createInfo)            │                              │                          │
   │                                        │                           │────────────────────────────>│                              │                          │
   │                                        │                           │                             │ GeneralAlloc()               │                          │
   │                                        │                           │                             │─────────────────────────────>│                          │
   │                                        │                           │                             │ [!SubAllocatable]             │                          │
   │                                        │                           │                             │ CreateMemory(Hal)            │                          │
   │                                        │                           │                             │─────────────────────────────>│                          │
   │                                        │                           │                             │ InitGeneralDeviceMemory()    │                          │
   │                                        │                           │                             │─────────────────────────────>│                          │
   │                                        │                           │                             │                              │ CreateGpuMemory()          │
   │                                        │                           │                             │                              │──────────────────────────>│
   │                                        │                           │                             │                              │ Init(CREATE)              │
   │                                        │                           │                             │                              │──────────────────────────>│
   │                                        │                           │                             │                              │ AllocateOrPinMemory()     │
   │                                        │                           │                             │                              │    │                      │
   │                                        │                           │                             │                              │    ├─ AssignVirtualAddr() │
   │                                        │                           │                             │                              │    │                      │
   │                                        │                           │                             │                              │    ├─ AllocBuffer()       │
   │                                        │                           │                             │                              │    │  [mtgpuBoAlloc ioctl] │
   │                                        │                           │                             │                              │    │─── ioctl ────────────>│
   │                                        │                           │                             │                              │    │◄─── KMD resp ─────────│
   │                                        │                           │                             │                              │    │                      │
   │                                        │                           │                             │                              │    └─ MapVirtualAddress() │
   │                                        │                           │                             │                              │         [mtgpuBoVmMapV2] │
   │                                        │                           │                             │                              │         ─── ioctl ─────>│
   │                                        │                           │                             │                              │         ◄─── KMD resp ──│
   │                                        │                           │                             │◄─────────────────────────────│                          │
   │                                        │                           │ ◄─ Init() ok ───────────────│                              │                          │
   │                                        │                           │                             │                              │                          │
   │                                        │                           │ MapToPeers(pMemory)          │                              │                          │
   │                                        │                           │ (multi-device peer map)      │                              │                          │
   │                                        │                           │ AddMemory(pMemory)           │                              │                          │
   │                                        │                           │ TrackMemory(memory_sp)       │                              │                          │
   │◄─── OK + dptr ─────────────────────────│                           │                              │                              │                          │
```

---

## 四、逐层关键代码路径

### Layer 1: Driver 入口 — `muapiMemAlloc_v2`

**文件**: `musa/src/driver/mu_memory.cpp:265`

```cpp
MUresult muapiMemAlloc_v2(MUdeviceptr *dptr, size_t bytesize) {
    Musa::MemoryCreateInfo createInfo{};
    createInfo.type = Musa::memoryTypeGeneral;
    createInfo.general.size = bytesize;
    createInfo.general.alignment = 0;
    createInfo.general.flags = Hal::memoryPropertyVirtual |
                               Hal::memoryPropertyDeviceMapped;
    // ← 注意: 没有 Hal::memoryPropertySubAllocatable
    Context::CreateMemory(&pMemory, createInfo);
    *dptr = pMemory->GetDevicePointer();
}
```

**与 SubAlloc 路径唯一区别**: flags 中 **不包含** `Hal::memoryPropertySubAllocatable`。

---

### Layer 2: 分流判断 — `Memory::GeneralAlloc`

**文件**: `musa/src/musa/core/memory.cpp:491-496`

```cpp
if (createInfo.alloc.property & Hal::memoryPropertySubAllocatable) {
    // Path A: MemMgr::Allocate → MemoryPool (sub-allocation)
    status = HalToMuResult(m_Context->GetParentDevice()->Hal()
        .GetMemMgr()->Allocate(createInfo.alloc, &m_Offset, &m_pHalMemory));
} else {
    // Path B: Hal::Device::CreateMemory (裸 KMD 分配) ← 我们分析的路径
    status = HalToMuResult(m_Context->GetParentDevice()->Hal()
        .CreateMemory(createInfo, &m_pHalMemory));
}
```

**flags 推导链** (在进入 GeneralAlloc 之前):
```
Virtual                                → bit 0
  → DeviceMapped                       → bit 1
    → (SubAllocatable = false)         → bit 2 不设置
```

GeneralAlloc 内部会将 flags 展开为:
```
property |= Physical | SharedVirtualAddress                // 自动添加
property |= (Virtual ? DeviceVisible|HostVisible|... : 0)  // 由 Virtual 推导
viewCapability |= (DeviceMapped ? PeerAccessible|IpcExportable : None)
```

---

### Layer 3: M3D Device — `Device::CreateMemory`

**文件**: `musa/src/hal/m3d/device.cpp:169`

```cpp
Result Device::CreateMemory(const MemoryCreateInfo& createInfo, IMemory** ppMemory) {
    Memory* pMemory = new Memory(*this);
    res = pMemory->Init(createInfo);  // 进入 Hal::Memory::Init
    *ppMemory = pMemory;
}
```

简单的工厂模式: 创建 `Hal::M3d::Memory` 对象并调用其 `Init`。

---

### Layer 4: `Memory::Init` 类型分发

**文件**: `musa/src/hal/m3d/memory.cpp:115-179`

```cpp
Result Memory::Init(const MemoryCreateInfo& createInfo) {
    if (createInfo.type == memoryTypeAlloc) {
        m_Size = createInfo.alloc.size;
        m_Heap = createInfo.alloc.heap;          // = largePage
        m_Props = createInfo.alloc.property;
        m_ViewCapability = createInfo.alloc.viewCapability;
        m_NumaId = createInfo.alloc.numaId;

        switch (createInfo.alloc.type) {
        case memoryAllocTypeDeviceLocal:
            result = InitGeneralDeviceMemory(createInfo);  // ← GPU 显存
            break;
        case memoryAllocTypeHost:
            result = InitGeneralHostMemory(createInfo);    // ← Host 内存
            break;
        }
    }
    // ... 其他 view type 分支
}
```

因为 `type = memoryTypeAlloc` + `alloc.type = memoryAllocTypeDeviceLocal`，进入 `InitGeneralDeviceMemory`。

---

### Layer 5: `InitGeneralDeviceMemory` — 构造 KMD 参数

**文件**: `musa/src/hal/m3d/memory.cpp:366-426`

```cpp
Result Memory::InitGeneralDeviceMemory(const MemoryCreateInfo& createInfo) {
    // 1. 计算对齐
    DevSize alignment = std::max(
        m_Device.GetCapabilities().memoryAllocationAlignSize,
        createInfo.alloc.alignment);
    alignment = ::Util::AlignUp(alignment,
        m_Device.GetHeapInfos()[heap].largestPageSize);

    // 2. 构造 GpuMemoryCreateInfo
    IM3d::GpuMemoryCreateInfo gpuMemoryCreateInfo = {};
    gpuMemoryCreateInfo.size = createInfo.alloc.size;
    gpuMemoryCreateInfo.alignment = alignment;
    gpuMemoryCreateInfo.heapAccess = IM3d::GpuHeapAccessExplicit;
    gpuMemoryCreateInfo.heaps[0] = IM3d::GpuHeap::GpuHeapLocal;
    gpuMemoryCreateInfo.flags.peerWritable = canMapPeerMemory;

    // 3. 物理/virtual 属性
    gpuMemoryCreateInfo.flags.physicalAlloc =
        (createInfo.alloc.property & memoryPropertyVirtual) == 0;  // true
    gpuMemoryCreateInfo.flags.discontinuousAlloc = 1;

    // 4. VA range
    gpuMemoryCreateInfo.vaRange = IM3d::VaRange::Svm;  // largePage → SVM
    gpuMemoryCreateInfo.flags.svmAlloc = (property & memoryPropertySharedVirtualAddress) != 0;
    gpuMemoryCreateInfo.flags.globalGpuVa = unifiedAddressing;

    // 5. 获取所需 object 大小 + 真正分配
    gpuMemObjSize = m3dDevice->GetGpuMemorySize(gpuMemoryCreateInfo, &m3dRes);
    m_M3dGpuMemory.Reserve(gpuMemObjSize);
    result = M3dToHalResult(m3dDevice->CreateGpuMemory(
        gpuMemoryCreateInfo,
        m_M3dGpuMemory.GetAlloc(),
        &m_M3dGpuMemory()));
}
```

**关键参数**:
- `physicalAlloc = true`: 分配物理连续显存（因为 `Virtual` flag 实际对应 HAL 层的 virtual 属性，M3D 层用 `physicalAlloc=true` 表示非 SVM）
- `heaps[0] = GpuHeapLocal`: 优先本地 VRAM
- `vaRange = Svm` + `svmAlloc = true`: 共享虚拟地址（64KB 大页对齐）

---

### Layer 6: `GpuMemory::Init` — M3D 核心初始化

**文件**: `musa/src/hal/m3d/m3d/src/core/gpuMemory.cpp:539-1070`

这是 M3D 框架中最大的初始化函数 (~530 行)。核心逻辑:

#### (a) 属性传递
```cpp
m_desc.vaRange     = createInfo.vaRange;    // Svm
m_desc.usage       = createInfo.usage;      // Generic
m_desc.flags       = ...;                    // isVirtual, isSvmAlloc 等
m_flags.isClient   = 1;
m_flags.alwaysResident = ...;
```

#### (b) Heap 选择策略 (`heapAccess = GpuHeapAccessExplicit`)
```cpp
case GpuHeapAccess::GpuHeapAccessExplicit:
    m_heapCount = createInfo.heapCount;  // = 1
    m_heaps[0] = createInfo.heaps[0];    // = GpuHeapLocal
    break;
```

#### (c) ZFB (Zero Frame Buffer) 优化 — 过滤 invisible/local heap
```cpp
// 如果 invisible heap 为空，强制过滤掉
if (m_pDevice->HeapProperties(GpuHeapInvisible).heapSize == 0)
    移除 GpuHeapInvisible;
// 如果 local heap 为空，强制过滤掉
if (m_pDevice->HeapProperties(GpuHeapLocal).heapSize == 0)
    移除 GpuHeapLocal;
```

#### (d) LargePage / BigPage 对齐优化
```cpp
if (IsVirtual() == false && baseVirtAddr == 0) {
    idealAlignment = max(largePageSize, bigPageSize);
    m_desc.alignment = Pow2Align(m_desc.alignment, idealAlignment);
    m_desc.size      = Pow2Align(m_desc.size, idealAlignment);
}
```

#### (e) 调用 `AllocateOrPinMemory()`
```cpp
result = AllocateOrPinMemory(baseVirtAddr, pPagingFence, ...);
```

---

### Layer 7: OS 平台 `AllocateOrPinMemory`

#### Linux DRM (`mtgpuMemory.cpp:387-743`)

```
AllocateOrPinMemory
  │
  ├─ ① VA 分配
  │    if (IsSvmAlloc())
  │      AllocateSvmVirtualAddressWithVaMgr(baseVirtAddr, size, alignment)
  │    else
  │      pDevice->AssignVirtualAddress(this, vaBaseAlignment, &baseVirtAddr)
  │
  ├─ ② 物理 Buffer 分配
  │    if (IsPinned())
  │      pDevice->PinMemory(&pinnedRequest, &m_offset, &bufferHandle)
  │    else
  │      pDevice->AllocBuffer(&allocRequest, &bufferHandle)
  │         └─ mtgpuBoAlloc() KMD ioctl ★
  │
  └─ ③ VA → Buffer 映射
       pDevice->MapVirtualAddress(
           bufferHandle, m_offset, size, gpuVirtAddr,
           alignment, mapFlags, mapPfm)
          └─ mtgpuBoVmMapV2() KMD ioctl ★
```

**`AllocBuffer` 调用参数构造** (`l.613-687`):
- `preferred_heap`: 根据 heap 类型设置 `MTGPU_BO_DOMAIN_VRAM` 或 `MTGPU_BO_DOMAIN_GTT`
- `flags`: CPU 可见性 (`CPU_READ_WRITE`)、缓存类型 (`GPU_CACHED`/`GPU_UNCACHED_WC`/`GPU_CACHE_COHERENT`)、`GPU_READ_WRITE`
- `phys_alignment`: 物理对齐
- `alloc_size`: 分配大小
- 特殊 flags: `NON_CONTIGUOUS`(非连续)、`NUMA_ENABLE`、`ZERO_ON_ALLOC` 等

**`MapVirtualAddress` 调用** (`l.705-712`):
```cpp
pDevice->MapVirtualAddress(
    bufferHandle,
    m_offset,
    m_desc.size + (needCePadding ? vaBaseAlignment : 0),
    m_desc.gpuVirtAddr,
    align,
    mapFlags & MTGPU_BO_FLAGS_MAPPING_MASK,
    m_flags.pfmAlloc);
```

#### Windows WDDM2 (`wddmGpuMemory.cpp:549-680`)

```
AllocateOrPinMemory
  │
  ├─ ① SVM 路径 (if IsSvmAlloc)
  │    ReserveGpuVirtualAddress()
  │    VirtualReserve(size, &gpuVirtAddr, alignment)
  │    VirtualCommit(gpuVirtAddr, size, IsExecutable())
  │
  ├─ ② VA 预留 (if Virtual || GlobalGpuVa || Chunked)
  │    ReserveGpuVirtualAddress(baseVirtAddr, virtualAccessMode)
  │
  ├─ ③ 物理分配 (if !IsVirtual)
  │    AllocateOrPinMemoryInternal()
  │      └─ Thunk::CreateAllocation() → KMD ioctl ★
  │
  └─ ④ VA 映射
       AcquireGpuVirtualAddress(baseVirtAddr, pagingFence) ★
         或 AcquireGpuVirtualAddressWithSpaceOptimize()
```

---

## 五、WDDM2 特有: `AllocateOrPinMemoryInternal`

**文件**: `wddmGpuMemory.cpp:1001` (默认实现为空，由子类实现)

在 Mthreads WDDM2 驱动 (`wddm2GpuMemory.cpp`) 中实现:
- 调用 `Thunk::CreateAllocation()` 触发 KMD 创建底层 `D3DKMT_ALLOCATIONINFO`
- 支持 chunked allocation (将大分配拆分为多个小 chunk)
- 处理 eviction / residency 管理

---

## 六、关键设计要点

### 1. 两次 ioctl 的必然性
无论分配多大，Linux DRM 路径至少需要两次 KMD 调用:
- `mtgpuBoAlloc`: 在 GPU 物理内存中分配 buffer object
- `mtgpuBoVmMapV2`: 在 GPU 虚拟地址空间中建立 VA → BO 映射

这是因为 MUSA/GPU 的寻址模型要求: VA (虚拟地址) 和 PA (物理地址) 是分离的，分配物理内存和建立虚拟映射是两个独立的 KMD 操作。

### 2. SVM (Shared Virtual Addressing) 的影响
当 `memoryPropertySharedVirtualAddress` 设置时:
- `vaRange = Svm` → 驱动使用统一的 CPU/GPU 虚拟地址空间
- `svmAlloc = true` → 标记为 SVM 分配
- Linux 下走 `AllocateSvmVirtualAddressWithVaMgr`，由 VA Manager 统一管理地址分配

### 3. LargePage 对齐
```cpp
alignment = max(memAllocAlignment, largestPageSize)
size = Pow2Align(size, largePageSize)  // 至少 64KB 对齐
```
当 `useReservedGpuVa` 为 true 时，甚至可以对齐到 BigPage (如 2MB) 以提升 TLB 效率。

### 4. Heap 降级逻辑
如果首选 heap 类型不可用:
- `GpuHeapInvisible` 缺失 → 强制移除，fallback 到 visible local
- `GpuHeapLocal` 缺失 → 强制移除，fallback 到 GART (系统内存)
- 最终保底: `GpuHeapGartUswc` (WC 缓存的 GART)

### 5. CE Extra Padding
```cpp
if (enableCeExtraPadding && IsPinned() && !IsSvmAlloc())
    pinned_size += vaBaseAlignment;  // 额外的 32B CE 填充
```
解决 CE (Copy Engine) 在 buffer 末尾的 dummy write 问题。

### 6. 与 Sub-Allocation 的性能对比

```
场景: 连续 16 次 muMemAlloc(4KB)

裸 KMD 路径:
  16 × (AllocBuffer ioctl + MapVirtualAddress ioctl) = 32 次 ioctl
  总耗时: ~3200μs (假设每次 ioctl ~100μs)

Sub-Alloc 路径:
  第 1 次: ChunkAllocate(64KB ioctl) + MapVirtualAddress(ioctl) + SubAllocate(切分)
         = 2 次 ioctl
  第 2-16 次: SubAllocate(空闲链表查找 + 切分)
         = 0 次 ioctl
  总耗时: ~200μs
  
性能提升: ~16x
```

---

## 七、M3D GpuMemory 对象生命周期

```
创建: GpuMemory::Init(CREATE) → AllocateOrPinMemory() → 物理分配 + VA 映射
引用: AddGpuMemoryReferences()         ← 增加引用计数 (Windows)
共享: OpenSharedMemory() / OpenPeerMemory()  ← 跨进程/跨设备共享
销毁: GpuMemory::Destroy() → DestroyAllocation() → Thunk::DestroyAllocation() → KMD 释放
```

---

## 八、文件索引

| 文件 | 行数 | 角色 |
|------|------|------|
| `musa/src/driver/mu_memory.cpp` | 265 | `muapiMemAlloc_v2` 入口 |
| `musa/src/musa/core/memory.cpp` | 462-496 | `GeneralAlloc` 分流判断 |
| `musa/src/musa/core/context.cpp` | 915-965 | `CreateMemory` 四步流程 |
| `musa/src/hal/m3d/device.cpp` | 169-179 | `Device::CreateMemory` |
| `musa/src/hal/m3d/memory.cpp` | 115-179 | `Memory::Init` 类型分发 |
| `musa/src/hal/m3d/memory.cpp` | 366-426 | `InitGeneralDeviceMemory` |
| `musa/src/hal/m3d/m3d/src/core/gpuMemory.cpp` | 539-1070 | `GpuMemory::Init(CREATE)` |
| `musa/src/hal/m3d/m3d/src/core/os/drm/mthreads/mtgpuMemory.cpp` | 387-743 | Linux DRM `AllocateOrPinMemory` |
| `musa/src/hal/m3d/m3d/src/core/os/drm/mthreads/mtgpuDevice.cpp` | 962-968 | `AllocBuffer` → ioctl |
| `musa/src/hal/m3d/m3d/src/core/os/drm/mthreads/mtgpuDevice.cpp` | 1401-1440 | `MapVirtualAddress` → ioctl |
| `musa/src/hal/m3d/m3d/src/core/os/wddm/wddmGpuMemory.cpp` | 549-680 | Windows WDDM2 `AllocateOrPinMemory` |
| `musa/src/hal/m3d/m3d/src/core/os/wddm/wddm2GpuMemory.cpp` | 269+ | WDDM2 `AllocateOrPinMemoryInternal` |