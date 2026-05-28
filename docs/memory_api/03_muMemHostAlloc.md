# muMemHostAlloc — 主机页锁定内存分配

## 功能

分配页锁定（pinned）的主机内存，GPU 可通过 DMA 直接访问（无需 staged 拷贝）。对应 CUDA `cudaHostAlloc`。

## 完整调用链

```
用户代码: muMemHostAlloc(&pp, 4096, MU_MEMHOSTALLOC_PORTABLE)
  │
  ├─ 1. mu_wrappers_generated.cpp — MUPTI 插桩
  │
  ├─ 2. mu_memory.cpp:497 — muapiMemHostAlloc
  │    └─ mu_memory.cpp:59 — imuapiMemHostAlloc
  │         ├─ InitPlatform()
  │         ├─ Flags 校验: 只允许 PORTABLE|DEVICEMAP|WRITECOMBINED
  │         ├─ TlsCtxTop() → Context
  │         ├─ MemoryCreateInfo{type=memoryTypePinnedHost, size, flags+NUMA+SUBALLOC, numaId}
  │         └─ pContext->CreateMemory(&pMemory, createInfo)
  │
  ├─ 3. context.cpp:915 — Context::CreateMemory
  │    ├─ new Memory(this)
  │    ├─ pMemory->Init(createInfo) → memoryTypePinnedHost → PinnedHostAlloc
  │    ├─ MapToPeers(pMemory)
  │    ├─ AddMemory + TrackMemory
  │    └─ *pp = pMemory->GetHostPointer()  ← 返回 CPU 地址
  │
  ├─ 4. memory.cpp:532 — Memory::PinnedHostAlloc
  │    ├─ 检查设备能力: canMapHostMemory? unifiedAddressing?
  │    ├─ 默认添加 DEVICEMAP (如果设备支持)
  │    ├─ Hal::MemoryCreateInfo
  │    │    type=Alloc, allocType=Host (非 DeviceLocal!)
  │    │    heap=LargePage(初始)
  │    │    property=Physical|SharedVA|HostMapped|...Devicemap属性
  │    │
  │    ├─ [subAllocatable] → MemMgr::Allocate (从 HostMapped pool)
  │    └─ [else]          → IDevice::CreateMemory
  │
  │    ⚠ 如果 LargePage 分配失败(MAP_FAILED):
  │      → static 变量 pinnedHostAllocHeap 永久降级到 General
  │      → 使用 General heap 重试分配
  │
  └─ 5. [HAL] 如果 memoryAllocTypeHost → InitGeneralHostMemory
       └─ m3dDevice->CreateGpuMemory(heap=GpuHeapGartUswc)
            → PCIe/GART 可见的主机内存
```

## 时序图

```
应用层              Wrapper            Driver(mu_memory)    Context            Memory            MemMgr/HAL         KMD
  │                  │                  │                   │                 │                 │                  │
  │ muMemHostAlloc   │                  │                   │                 │                 │                  │
  │─────────────────>│                  │                   │                 │                 │                  │
  │                  │                  │                   │                 │                 │                  │
  │                  │ muapiMemHostAlloc│                   │                 │                 │                  │
  │                  │─────────────────>│                   │                 │                 │                  │
  │                  │                  │                   │                 │                 │                  │
  │                  │                  │ Flags 校验:       │                 │                 │                  │
  │                  │                  │ 只允许 PORTABLE|  │                 │                 │                  │
  │                  │                  │ DEVICEMAP|WC      │                 │                 │                  │
  │                  │                  │                   │                 │                 │                  │
  │                  │                  │ TlsCtxTop()       │                 │                 │                  │
  │                  │                  │──────────────────>│                 │                 │                  │
  │                  │                  │<── ctx* ──────────│                 │                 │                  │
  │                  │                  │                   │                 │                 │                  │
  │                  │                  │ CreateInfo:       │                 │                 │                  │
  │                  │                  │ type=PinnedHost   │                 │                 │                  │
  │                  │                  │ size=4096         │                 │                 │                  │
  │                  │                  │ flags=PORTABLE|   │                 │                 │                  │
  │                  │                  │  NUMA_AFFINITIVE| │                 │                 │                  │
  │                  │                  │  SUBALLOCATABLE   │                 │                 │                  │
  │                  │                  │ numaId=NUMA_NO_NODE                │                 │                  │
  │                  │                  │                   │                 │                 │                  │
  │                  │                  │ CreateMemory      │                 │                 │                  │
  │                  │                  │──────────────────>│                 │                 │                  │
  │                  │                  │                   │                 │                 │                  │
  │                  │                  │                   │ new Memory(this)│                 │                  │
  │                  │                  │                   │ Init(PinnedHost)│                 │                  │
  │                  │                  │                   │────────────────>│                 │                  │
  │                  │                  │                   │                 │                 │                  │
  │                  │                  │                   │                 │ PinnedHostAlloc  │                  │
  │                  │                  │                   │                 │ size, flags,     │                  │
  │                  │                  │                   │                 │ numaId           │                  │
  │                  │                  │                   │                 │                  │                  │
  │                  │                  │                   │                 │ 创建设备能力       │                  │
  │                  │                  │                   │                 │ canMapHostMemory? │                  │
  │                  │                  │                   │                 │ unifiedAddressing?│                  │
  │                  │                  │                   │                 │                  │                  │
  │                  │                  │                   │                 │ 默认 +DEVICEMAP   │                  │
  │                  │                  │                   │                 │ 如果设备支持       │                  │
  │                  │                  │                   │                 │                  │                  │
  │                  │                  │                   │                 │ createInfo:       │                  │
  │                  │                  │                   │                 │ type=Alloc        │                  │
  │                  │                  │                   │                 │ allocType=Host    │                  │
  │                  │                  │                   │                 │ heap=LargePage    │                  │
  │                  │                  │                   │                 │ property=Physical │                  │
  │                  │                  │                   │                 │  |SharedVA|HostMap│                  │
  │                  │                  │                   │                 │  [+Virtual|HostVis│                  │
  │                  │                  │                   │                 │   |DevVisible|...]│                  │
  │                  │                  │                   │                 │                  │                  │
  │                  │                  │                   │                 │ [SubAllocatable?] │                  │
  │                  │                  │                   │                 │   │               │                  │
  │                  │                  │                   │                 │ YES│              │                  │
  │                  │                  │                   │                 │    └─MemMgr::     │                  │
  │                  │                  │                   │                 │      Allocate────>│                  │
  │                  │                  │                   │                 │                   │ GetInternalPool  │
  │                  │                  │                   │                 │                   │ (HostMapped)     │
  │                  │                  │                   │                 │                   │ FullAllocate      │
  │                  │                  │                   │                 │<── offset+mem ────│                  │
  │                  │                  │                   │                 │                  │                  │
  │                  │                  │                   │                 │ NO│               │                  │
  │                  │                  │                   │                 │   └─IDevice::     │                  │
  │                  │                  │                   │                 │     CreateMemory  │                  │
  │                  │                  │                   │                 │──────────────────>│                  │
  │                  │                  │                   │                 │                   │ Init(Host)       │
  │                  │                  │                   │                 │                   │ InitGeneralHost  │
  │                  │                  │                   │                 │                   │ Memory()         │
  │                  │                  │                   │                 │                   │ CreateGpuMemory  │
  │                  │                  │                   │                 │                   │ (GpuHeapGart)    │
  │                  │                  │                   │                 │                   │─────────────────>│──分配GART内存  │
  │                  │                  │                   │                 │                   │<── OK ───────────│                  │
  │                  │                  │                   │                 │<── OK ────────────│                  │
  │                  │                  │                   │                 │                  │                  │
  │                  │                  │                   │                 │ ⚠ MAP_FAILED?    │                  │
  │                  │                  │                   │                 │ 静态变量永久降级    │                  │
  │                  │                  │                   │                 │ heap=General      │                  │
  │                  │                  │                   │                 │ retry: ...        │                  │
  │                  │                  │                   │                 │                  │                  │
  │                  │                  │                   │ MapToPeers()    │                  │                  │
  │                  │                  │                   │ AddMemory()     │                  │                  │
  │                  │                  │                   │ TrackMemory()   │                  │                  │
  │                  │                  │                   │                  │                  │                  │
  │                  │                  │                   │ GetHostPointer()│                  │                  │
  │                  │                  │                   │────────────────>│                  │                  │
  │                  │                  │                   │                 │ lazy Map(0,size, │                  │
  │                  │                  │                   │                 │        &m_pMapped)│                  │
  │                  │                  │                   │                 │  + m_Offset      │                  │
  │                  │                  │                   │<── hostPtr* ────│                  │                  │
  │                  │                  │                   │                  │                  │                  │
  │                  │                  │<── OK + pp ───────│                  │                  │                  │
  │                  │<── OK ───────────│                   │                  │                  │                  │
  │<── pp (CPU地址)──│                  │                   │                  │                  │                  │
```

## 关键代码路径

### Driver 入口

```cpp
// mu_memory.cpp:59
static MUresult imuapiMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags) {
    // 1. Flags 校验: 只接受 PORTABLE | DEVICEMAP | WRITECOMBINED
    // 2. TlsCtxTop() + null 校验
    // 3. CreateInfo:
    //    type = memoryTypePinnedHost
    //    pinnedHost = {size, Flags | NUMA_AFFINITIVE | SUBALLOCATABLE, NUMA_NO_NODE}
    // 4. pContext->CreateMemory(&pMemory, createInfo)
    // 5. *pp = pMemory->GetHostPointer()  // 返回 CPU 可访问指针
}
```

### PinnedHostAlloc 实现

```cpp
// memory.cpp:532
MUresult Memory::PinnedHostAlloc(size_t size, unsigned int flags, int numaId) {
    // 1. 检查 canMapHostMemory → 不支持则报错
    // 2. 如果设备支持, 自动添加 DEVICEMAP flag
    // 3. 构造 HAL MemoryCreateInfo:
    //    type = Alloc, allocType = Host        ← 不是 DeviceLocal
    //    heap = LargePage (static 变量, 可降级)
    //    property:
    //      Physical | SharedVirtualAddress | HostMapped
    //      + [DEVICEMAP]: Virtual | HostVisible | DeviceVisible | DeviceWriteable | DeviceCached
    //      + SubAllocatable
    //    viewCapability: [DEVICEMAP] → PeerAccessible | IpcExportable | Exportable
    // 4. 分配:
    //    [SubAllocatable] → MemMgr::Allocate
    //    [else] → Device::CreateMemory
    // 5. 如果 LargePage 分配失败 (MAP_FAILED):
    //    static pinnedHostAllocHeap = General  // ← 永久降级!
    //    用 General heap 重试
}
```

### 内存类型区别

```
muMemAlloc:
  allocType = DeviceLocal → GPU 显存 (GpuHeapLocal)
  heap = LargePage
  GPU 访问: 直接 (本地显存带宽)

muMemHostAlloc:
  allocType = Host → 系统内存 (GpuHeapGartUswc)
  heap = LargePage → General (降级)
  GPU 访问: 通过 PCIe/GART DMA
  CPU 访问: 直接 (页锁定, 不换页)
```

### LargePage→General 降级机制

```cpp
// 关键代码: static 变量 + MAP_FAILED 判断
static Hal::MemoryHeap pinnedHostAllocHeap = Hal::MemoryHeap::largePage;

// 首次分配:
createInfo.alloc.heap = pinnedHostAllocHeap;  // largePage
status = MemMgr->Allocate(...);  // 或 Device->CreateMemory(...)

// 如果失败了:
if (pinnedHostAllocHeap == Hal::MemoryHeap::largePage && status == MUSA_ERROR_MAP_FAILED) {
    tprintf(LOG_MEM, "modify the heap of pinned host allocation from LargePage to General permanently\n");
    pinnedHostAllocHeap = Hal::MemoryHeap::general;  // 永久降级!
    createInfo.alloc.heap = pinnedHostAllocHeap;
    status = MemMgr->Allocate(...);  // 重试
}
```

## 关键设计要点

1. **mmap + 页锁定**: 分配的 host 内存被 mmap 并锁定（mlock），保证不会被换出到磁盘，GPU 才能安全地通过 DMA 直接访问
2. **DEVICEMAP 自动添加**: 如果设备支持 `canMapHostMemory`，自动加上 DEVICEMAP 标志，使 GPU 能直接访问
3. **CPU 共享虚拟地址 (SVA)**: `memoryPropertySharedVirtualAddress` 使 GPU 和 CPU 使用相同的虚拟地址（如果硬件支持）
4. **LargePage 降级**: 当系统大页内存耗尽时自动降级到 General heap，是生产环境的关键容错逻辑
5. **Lazy CPU mapping**: CPU 映射在第一次调用 `GetHostPointer()` 时才实际进行，而不是在分配时
