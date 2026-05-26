# 14_Memory_API_source_deep_dive — Core 层 9 种分配路径逐函数源码分析

> 源码文件：`musa/src/musa/core/memory.cpp` (827 行)

## 1. 概览: 8 种内存类型 × 9 种初始化函数

`Memory::Init()` (memory.cpp:378) 根据 `createInfo.type` 分发到以下初始化函数：

```
memoryTypeGeneral         → GeneralAlloc()             [memory.cpp:462]
memoryTypePitchedGeneral  → PitchedGeneralAlloc()      [memory.cpp:499]
memoryTypePinnedHost      → PinnedHostAlloc()          [memory.cpp:532]
memoryTypeRegisteredPinnedHost → PinnedHostRegister()  [memory.cpp:611]
memoryTypeManaged         → ManagedAlloc()             [memory.cpp:673]
memoryTypeIpcImport       → IpcImportAlloc()           [memory.cpp:715]
memoryTypeExternal        → ExternalAlloc()            [memory.cpp:760]
memoryTypePrealloc        → PreallocAlloc()            [memory.cpp:801]
memoryTypeVirtual         → VirtualAlloc()             [memory.cpp:805]
```

## 2. 函数 1: GeneralAlloc (设备内存分配)

**行号**: 462-497 | **调用者**: `muMemAlloc` 默认路径

```cpp
MUresult Memory::GeneralAlloc(size_t size, size_t alignment, unsigned int flags)
{
    // Step 1: 构建 Shape
    m_Shape = { size, 1, 1, size };
    m_Flags = flags;

    // Step 2: 构建 HAL 层创建信息
    Hal::MemoryCreateInfo createInfo{};
    createInfo.type = Hal::memoryTypeAlloc;
    createInfo.alloc.type = Hal::memoryAllocTypeDeviceLocal;   // GPU 显存
    createInfo.alloc.heap = Hal::MemoryHeap::largePage;        // 优先大页
    createInfo.alloc.property = flags;                          // 用户 flags (0x07)

    // Core 自动追加属性:
    createInfo.alloc.property |= Hal::memoryPropertyPhysical |
                                 Hal::memoryPropertySharedVirtualAddress;

    // 由 Virtual flag 推导 (0x01):
    if (flags & Hal::memoryPropertyVirtual) {
        createInfo.alloc.property |= Hal::memoryPropertyDeviceVisible |
                                     Hal::memoryPropertyHostVisible |
                                     Hal::memoryPropertyHostCoherent |
                                     Hal::memoryPropertyDeviceWriteable |
                                     Hal::memoryPropertyDeviceCached;
    }

    // 由 DeviceMapped flag 推导 (0x02):
    createInfo.alloc.viewCapability = Hal::memoryViewCapabilityExportable;
    if (flags & Hal::memoryPropertyDeviceMapped) {
        createInfo.alloc.viewCapability |= Hal::memoryViewCapabilityPeerAccessible |
                                           Hal::memoryViewCapabilityIpcExportable;
    }

    // 对齐 + NUMA
    createInfo.alloc.alignment = std::max(alignment, device.minAllocAlign);
    createInfo.alloc.numaId = NUMA_NO_NODE;

    // Step 3: 分叉选择分配路径
    if (createInfo.alloc.property & Hal::memoryPropertySubAllocatable) {
        // 路径 A: Pool 子分配 (默认 flags=0x07 时走这里)
        status = HalToMuResult(
            m_Context->GetParentDevice()->Hal().GetMemMgr()->Allocate(
                createInfo.alloc, &m_Offset, &m_pHalMemory));
    } else {
        // 路径 B: 裸 KMD 直接分配
        status = HalToMuResult(
            m_Context->GetParentDevice()->Hal().CreateMemory(createInfo, &m_pHalMemory));
    }
    return status;
}
```

**关键设计**:
- flags=0x07 时, `SubAllocatable` 置位 → 走 `MemMgr::Allocate` → Pool 子分配
- 去掉 `SubAllocatable` 后 → 走 `Hal::CreateMemory` → 直接 ioctl

## 3. 函数 2: PitchedGeneralAlloc (对齐设备内存分配)

**行号**: 499-530 | **调用者**: `muMemAllocPitch`

```cpp
MUresult Memory::PitchedGeneralAlloc(size_t widthInBytes, size_t height, size_t depth, size_t pitch)
{
    m_Shape = { widthInBytes, height, depth, pitch };

    Hal::MemoryCreateInfo createInfo{};
    createInfo.type = Hal::memoryTypeAlloc;
    createInfo.alloc.type = Hal::memoryAllocTypeDeviceLocal;
    createInfo.alloc.heap = Hal::MemoryHeap::largePage;

    // 属性: 全部能力全开 (HostVisible + DeviceVisible + Writeable + Cached + ...)
    createInfo.alloc.property = Hal::memoryPropertyPhysical
                              | Hal::memoryPropertyVirtual
                              | Hal::memoryPropertyHostVisible
                              | Hal::memoryPropertyHostCoherent
                              | Hal::memoryPropertyDeviceVisible
                              | Hal::memoryPropertyDeviceMapped
                              | Hal::memoryPropertyDeviceWriteable
                              | Hal::memoryPropertyDeviceCached
                              | Hal::memoryPropertySharedVirtualAddress
                              | Hal::memoryPropertySubAllocatable;    // ← 也走 Pool!

    createInfo.alloc.viewCapability = Hal::memoryViewCapabilityPeerAccessible
                                     | Hal::memoryViewCapabilityIpcExportable
                                     | Hal::memoryViewCapabilityExportable;

    createInfo.alloc.size = pitch * height * depth;        // 总大小 = pitch × height × depth
    // 溢出检查:
    if (createInfo.alloc.size < pitch || createInfo.alloc.size < height || createInfo.alloc.size < depth)
        return MUSA_ERROR_OUT_OF_MEMORY;

    createInfo.alloc.alignment = device.minAllocAlign;
    createInfo.alloc.numaId = NUMA_NO_NODE;

    // 固定走 Pool 子分配路径:
    status = HalToMuResult(
        m_Context->GetParentDevice()->Hal().GetMemMgr()->Allocate(
            createInfo.alloc, &m_Offset, &m_pHalMemory));
    return status;
}
```

**与 GeneralAlloc 的区别**:
- 属性全开 (Host+Device 都可见可写)
- 固定使用 Sub-Allocation (不存在裸 KMD 路径的分支判断)
- 总大小 = pitch × height × depth (支持 3D 内存)

## 4. 函数 3: PinnedHostAlloc (主机页锁定内存)

**行号**: 532-609 | **调用者**: `muMemHostAlloc` / `muMemAllocHost`

```cpp
MUresult Memory::PinnedHostAlloc(size_t size, unsigned int flags, int numaId)
{
    // 1. 能力校验
    if (PORTABLE && !unifiedAddressing) → WARN
    if (DEVICEMAP && !canMapHostMemory) → ERR_NOT_SUPPORTED
    if (canMapHostMemory) → 自动追加 DEVICEMAP flag

    m_Shape = { size, 1, 1, size };
    m_Flags = flags;

    // 2. heap 选择 (static 变量, 一旦降级就永久生效)
    static Hal::MemoryHeap pinnedHostAllocHeap = Hal::MemoryHeap::largePage;

    // 3. 构建 createInfo
    createInfo.type = Hal::memoryTypeAlloc;
    createInfo.alloc.type = Hal::memoryAllocTypeHost;       // ← HOST, 不是 DeviceLocal!
    createInfo.alloc.heap = pinnedHostAllocHeap;             // largePage → 可能降级

    createInfo.alloc.property = Hal::memoryPropertyPhysical |
                                Hal::memoryPropertySharedVirtualAddress |
                                Hal::memoryPropertyHostMapped;   // HostMapped (可被 CPU 映射)

    if (DEVICEMAP) {
        createInfo.alloc.property |= Hal::memoryPropertyVirtual |
                                     Hal::memoryPropertyHostVisible |
                                     Hal::memoryPropertyDeviceVisible |
                                     Hal::memoryPropertyDeviceWriteable |
                                     Hal::memoryPropertyDeviceCached;
    }

    // WRITECOMBINED vs COHERENT
    if (flags & WRITECOMBINED) → Hal::memoryPropertyHostWriteCombined;
    else                       → Hal::memoryPropertyHostCoherent;

    // SUBALLOCATABLE (从 flags 继承)
    // NUMA_CURRENT (从 flags 继承)

    createInfo.alloc.viewCapability = DEVICEMAP ?
        (PeerAccessible | IpcExportable | Exportable) : None;

    createInfo.alloc.size = size;
    createInfo.alloc.alignment = device.minAllocAlign;
    createInfo.alloc.numaId = numaId;

    // 4. 分配 (同 GeneralAlloc 的子分配分叉)
    if (createInfo.alloc.property & SubAllocatable) {
        status = MemMgr::Allocate(...);
    } else {
        status = Hal::CreateMemory(...);
    }

    // 5. LargePage → General 降级 ★★★
    if (pinnedHostAllocHeap == largePage && status == MAP_FAILED) {
        tprintf("modify heap from LargePage to General permanently");
        pinnedHostAllocHeap = Hal::MemoryHeap::general;  // static 变量修改!
        // 重新分配...
    }

    return status;
}
```

**关键设计**:
- `static Hal::MemoryHeap pinnedHostAllocHeap` — 进程级静态变量, 降级后永久生效
- `alloc.type = memoryAllocTypeHost` — 与 GeneralAlloc 的 `memoryAllocTypeDeviceLocal` 不同
- 属性中包含 `HostMapped` — 标志 CPU 需要映射访问

## 5. 函数 4: PinnedHostRegister (注册用户指针)

**行号**: 611-671 | **调用者**: `muMemHostRegister`

```cpp
MUresult Memory::PinnedHostRegister(void* ptr, size_t size, unsigned int flags)
{
    // 1. 能力校验
    if (!hostRegisterSupported) → ERR_NOT_SUPPORTED
    if (PORTABLE && !unifiedAddressing) → WARN
    if (DEVICEMAP && !canMapHostMemory) → ERR_NOT_SUPPORTED
    if (READ_ONLY && !supportReadonlyHostRegister) → ERR_NOT_SUPPORTED
    if (canMapHostMemory) → 自动追加 DEVICEMAP

    // 2. 重叠检测 (OVERLAP_CHECK flag)
    if (OVERLAP_CHECK) {
        MemoryTracker::IterateMemories(ptr, size, overlapCheck);
        // 已注册 → HOST_MEMORY_ALREADY_REGISTERED
        // 已管理 → INVALID_VALUE
    }

    m_Shape = { size, 1, 1, size };
    m_Flags = flags;

    // 3. 分支: MMIO vs Locked
    Hal::MemoryCreateInfo createInfo{};

    if (flags & MU_MEMHOSTREGISTER_IOMEMORY) {
        // 分支 A: MMIO (IOMEMORY)
        createInfo.type = Hal::memoryTypeView;
        createInfo.view.type = Hal::memoryViewTypeExternal;
        createInfo.view.external.isDeviceMapped = true;
        createInfo.view.external.isPeerAccessible = true;
        createInfo.view.external.heap = Hal::MemoryHeap::general;
        createInfo.view.external.type = Hal::MemoryExternalHandleType::mmio;
        createInfo.view.external.handle.data.u64Data = (uint64_t)ptr;  // ← 用户指针直接作为 MMIO 地址
        createInfo.view.external.size = size;
    } else {
        // 分支 B: Locked (页锁定)
        createInfo.type = Hal::memoryTypeView;
        createInfo.view.type = Hal::memoryViewTypeLocked;
        createInfo.view.locked.pHost = ptr;                          // ← 用户指针
        createInfo.view.locked.size = size;
        createInfo.view.locked.isDeviceMapped = (DEVICEMAP);
        createInfo.view.locked.isDeviceReadOnly = (READ_ONLY);
        createInfo.view.locked.isPeerAccessible = (PORTABLE) || unifiedAddressing;
    }

    // 4. 调用 HAL
    status = Hal::CreateMemory(createInfo, &m_pHalMemory);

    // HAL 内部 → InitLockedMemory() 或 InitExternalMemory(MMIO)
    return status;
}
```

**两个分支的 HAL 路径**:
- **Locked**: `InitLockedMemory()` → pin 用户物理页 → `CreatePinnedGpuMemory()`
- **MMIO**: `InitExternalMemory()` → 映射 MMIO 地址空间 → `OpenExternalSharedGpuMemory(Mmio)`

## 6. 函数 5: ManagedAlloc (托管内存)

**行号**: 673-713 | **调用者**: `muMemAllocManaged`

```cpp
MUresult Memory::ManagedAlloc(size_t size, unsigned int flags)
{
    // flags 校验: 必须是 ATTACH_GLOBAL 或 ATTACH_HOST
    if (!(ATTACH_GLOBAL || ATTACH_HOST)) → ERR_INVALID_VALUE

    // ATTACH_HOST 需要 concurrent managed access 能力
    if (ATTACH_HOST && !concurrentManagedAccess) → ERR_NOT_SUPPORTED

    // ATTACH_GLOBAL 需要 unified addressing
    if (ATTACH_GLOBAL && !unifiedAddressing) → 去掉 GLOBAL flag (降级)

    m_Shape = { size, 1, 1, size };
    m_Flags = flags;

    Hal::MemoryCreateInfo createInfo{};
    createInfo.type = Hal::memoryTypeAlloc;

    // 分配类型: 由 managedForceDeviceAlloc 设置决定
    createInfo.alloc.type = managedForceDeviceAlloc ?
        Hal::memoryAllocTypeDeviceLocal : Hal::memoryAllocTypeHost;

    createInfo.alloc.heap = Hal::MemoryHeap::largePage;

    // 属性: 全能力集合
    createInfo.alloc.property = Hal::memoryPropertyPhysical |
                                Hal::memoryPropertyVirtual |
                                Hal::memoryPropertyHostVisible |
                                Hal::memoryPropertyHostMapped |
                                Hal::memoryPropertyDeviceVisible |
                                Hal::memoryPropertyDeviceWriteable |
                                Hal::memoryPropertyDeviceMapped |
                                Hal::memoryPropertyHostCoherent |
                                Hal::memoryPropertyDeviceCached |
                                Hal::memoryPropertySharedVirtualAddress;

    createInfo.alloc.viewCapability = Hal::memoryViewCapabilityPeerAccessible;
    createInfo.alloc.size = size;

    // ★ 直接调用 Hal::CreateMemory() — 不走 Pool!
    status = Hal::CreateMemory(createInfo, &m_pHalMemory);

    if (success) {
        GetHostPointer();   // 强制映射主机端访问
    }

    return status;
}
```

**关键设计**:
- **不使用 Sub-Allocation**: managed 内存需要 page fault 按需迁移, pool 简单切分模型不适用
- `managedForceDeviceAlloc`: 为 true 时在 GPU 显存分配, 为 false 时在系统内存分配 (GPU 可通过 PCIE 访问)
- 分配后强制调用 `GetHostPointer()` 建立 CPU 映射

## 7. 函数 6: IpcImportAlloc (IPC 导入)

**行号**: 715-758 | **调用者**: `muMemImportShareableHandle`

```cpp
MUresult Memory::IpcImportAlloc(IpcHandleInternal handle, unsigned int flags)
{
    // flags 校验: 必须是 IPC lazy flags 之一
    switch(flags) {
    case MU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS:
    case MU_IPC_MEM_LAZY_UNIDIR_INTERLEAVE_PEER_ACCESS:
    case MU_IPC_MEM_LAZY_BIDIR_INTERLEAVE_PEER_ACCESS:
    case ...:
        break;
    default:
        → ERR_INVALID_VALUE
    }

    // 不能导入自己进程的 IPC (fromMempool=false 且 pid==self)
    if (handle.pid == self && !handle.fromMempool) → ERR_INVALID_CONTEXT

    Hal::MemoryCreateInfo createInfo{};
    createInfo.type = Hal::memoryTypeView;
    createInfo.view.type = Hal::memoryViewTypeExternal;
    createInfo.view.external.type = Hal::MemoryExternalHandleType::kernelManagedGlobal;
    createInfo.view.external.handle = handle.halHandle;    // ← HAL 层的全局句柄
    createInfo.view.external.isDeviceMapped = true;
    createInfo.view.external.isPeerAccessible = true;
    createInfo.view.external.isUva = true;
    createInfo.view.external.heap = handle.heap;

    status = Hal::CreateMemory(createInfo, &m_pHalMemory);

    if (success) {
        m_Offset = handle.allocOffset;         // ← 子分配偏移 (来自导出端)
        m_Shape = { handle.allocSize, 1, 1, handle.allocSize };
        m_Flags = flags;
    }

    return status;
}
```

**关键设计**:
- 通过 `handle.halHandle` 在 KMD 侧打开已导出的全局内存
- `m_Offset` 从 handle 中恢复, 因为可能只导出了子分配块
- `isUva = true` — 设置统一虚拟地址属性

## 8. 函数 7: ExternalAlloc (外部内存导入)

**行号**: 760-799 | **调用者**: `muMemImportShareableHandle` (非 IPC) 和其他外部导入路径

```cpp
MUresult Memory::ExternalAlloc(const ExternalMemoryInfo& externalInfo)
{
    m_HandleType = externalInfo.handleInfo.type;

    Hal::MemoryCreateInfo createInfo{};
    createInfo.type = Hal::memoryTypeView;
    createInfo.view.type = Hal::memoryViewTypeExternal;
    createInfo.view.external.isDeviceMapped = externalInfo.deviceMapped;
    createInfo.view.external.isPeerAccessible = true;
    createInfo.view.external.isUva = true;

    // heap 选择: graphics → general, 否则 → largePage
    createInfo.view.external.heap = externalInfo.fromGraphics ?
        Hal::MemoryHeap::general : Hal::MemoryHeap::largePage;

    // 根据 handle 类型设置具体参数
    switch (externalInfo.handleInfo.type) {
    case MemoryHandleType::Fabric:
        createInfo.view.external.type = Hal::MemoryExternalHandleType::fabric;
        createInfo.view.external.handle.data.u64Data = &fabricHandle;
        break;
    case MemoryHandleType::OpaqueFd:
        createInfo.view.external.type = Hal::MemoryExternalHandleType::dmaBuf;
        createInfo.view.external.handle.data.intData = fd;
        createInfo.view.external.size = size;
        break;
    default:
        → ERR_NOT_SUPPORTED
    }

    status = Hal::CreateMemory(createInfo, &m_pHalMemory);

    if (success) {
        m_Offset = externalInfo.viewOffset;
        size_t viewSize = externalInfo.viewSize ?: m_pHalMemory->GetSize();
        m_Shape = { viewSize, 1, 1, viewSize };
        m_Flags = externalInfo.handleInfo.flags;
    }

    return status;
}
```

**支持的外部句柄类型**:
- `Fabric` — 跨节点 fabric 内存
- `OpaqueFd` — DMA-BUF 文件描述符 (Linux)

## 9. 函数 8: VirtualAlloc (虚拟地址空间分配)

**行号**: 805-825 | **调用者**: `muMemAddressReserve` 等

```cpp
MUresult Memory::VirtualAlloc(size_t size, size_t alignment, MUdeviceptr uniAddr, unsigned long long flags)
{
    m_Shape = { size, 1, 1, size };
    m_Flags = flags;

    // flags 必须为 0 (当前不支持任何 flag)
    if (flags != 0) → ERR_INVALID_VALUE

    Hal::MemoryCreateInfo createInfo{};
    createInfo.type = Hal::memoryTypeVirtual;                // ← 纯虚拟, 不分配物理内存
    createInfo.virt.heap = Hal::MemoryHeap::largePage;
    createInfo.virt.size = size;
    createInfo.virt.addr = uniAddr;                           // 指定的起始 VA (0=任意)
    createInfo.virt.alignment = alignment;

    // 直接调用 HAL, 不走 Pool (仅创建 VA 空间)
    status = Platform::Get().Hal().CreateMemory(createInfo, &m_pHalMemory);

    return status;
}
```

**关键设计**:
- `type = memoryTypeVirtual` — 仅分配虚拟地址空间, 不分配物理内存
- 后续通过 `Bind()` 将物理内存映射到虚拟地址
- 用于 GPU sub-allocation 场景: 先预留 VA 空间, 然后按需绑定物理页

## 10. 各函数的 Hal 调用汇总表

| 函数 | Hal CreateMemory type | 内存类型 | Heap | 是否走 Pool |
|------|----------------------|----------|------|------------|
| GeneralAlloc | memoryTypeAlloc | DeviceLocal | largePage | YES (SubAllocatable) |
| PitchedGeneralAlloc | memoryTypeAlloc | DeviceLocal | largePage | YES (固定) |
| PinnedHostAlloc | memoryTypeAlloc | Host | largePage/general | YES (若 SubAllocatable) |
| PinnedHostRegister(Locked) | memoryTypeView | View/Locked | — | NO |
| PinnedHostRegister(MMIO) | memoryTypeView | View/External(MMIO) | general | NO |
| ManagedAlloc | memoryTypeAlloc | DeviceLocal 或 Host | largePage | NO |
| IpcImportAlloc | memoryTypeView | View/External(Global) | — | NO |
| ExternalAlloc | memoryTypeView | View/External(Fabric/DmaBuf) | general/largePage | NO |
| VirtualAlloc | memoryTypeVirtual | Virtual(仅VA) | largePage | NO |

## 11. 关键源码位置

| 文件 | 行数 | 说明 |
|------|------|------|
| `musa/src/musa/core/memory.cpp` | 1-827 | 所有 Core 层初始化函数 |
| `musa/src/musa/core/memory.h` | - | Memory 类声明 |
| `musa/src/musa/core/context.cpp` | 915-975 | CreateMemory 入口 + MapToPeers |
| `musa/src/hal/m3d/memory.h` | - | Hal::Memory 类声明 |
| `musa/src/hal/m3d/memory.cpp` | 115-837 | Hal::Memory::Init (全部分支) |
| `musa/src/hal/m3d/memMgr.h` | - | IMemMgr 接口 |
| `musa/src/hal/m3d/memoryPool.h` | - | MemoryPool 类声明 |
| `musa/src/musa/core/stream.cpp` | 519-580 | AsyncMemAlloc (stream-ordered 分配) |