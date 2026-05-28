# MUSA Memory API 源码深度分析（重写版）

> 基于 `musa/src/musa/core/memory.cpp`（827 行）逐函数逐行的完整代码拆解。

---

## 一、整体架构：9 种分配路径

`Memory::Init`（`memory.cpp:378`）是统一入口，根据 `createInfo.type` 分发到 9 个函数：

```
Memory::Init(createInfo)
  │
  ├─ memoryTypeGeneral         → GeneralAlloc()           ← 最常用，可选 sub-allocation
  ├─ memoryTypePitchedGeneral  → PitchedGeneralAlloc()    ← 3D 内存，带 pitch
  ├─ memoryTypePinnedHost      → PinnedHostAlloc()        ← 页锁定主机内存
  ├─ memoryTypeRegisteredPinnedHost → PinnedHostRegister() ← 注册已有主机指针
  ├─ memoryTypeManaged         → ManagedAlloc()           ← 统一内存（Managed）
  ├─ memoryTypeIpcImport       → IpcImportAlloc()         ← IPC 导入
  ├─ memoryTypeExternal        → ExternalAlloc()          ← 外部内存（DMA-Buf/Fabric）
  ├─ memoryTypePrealloc        → PreallocAlloc()          ← 预分配（未实现）
  └─ memoryTypeVirtual         → VirtualAlloc()           ← 仅 VA，无物理 backing
```

---

## 二、GeneralAlloc — 核心设备内存分配（逐行分析）

**文件**: `memory.cpp:462-497`

```cpp
MUresult Memory::GeneralAlloc(size_t size, size_t alignment, unsigned int flags) {
    m_Shape = { size, 1, 1, size };
    m_Flags = flags;
```

### Step 1: 构造 Hal::MemoryCreateInfo

```cpp
    Hal::MemoryCreateInfo createInfo{};
    createInfo.type = Hal::memoryTypeAlloc;                    // ① 类型：分配新内存
    createInfo.alloc.type = Hal::memoryAllocTypeDeviceLocal;   // ② 堆类型：设备本地
    createInfo.alloc.heap = Hal::MemoryHeap::largePage;        // ③ 堆：LargePage
```

**关键设计点**：
- `memoryAllocTypeDeviceLocal`：显式指定为设备本地内存，而非主机内存。这决定了 M3D 层走 `InitGeneralDeviceMemory` 而非 `InitGeneralHostMemory`。
- `MemoryHeap::largePage`：优先使用大页（LargePage），与 `PinnedHostAlloc` 一致。注释提到这是 PH1 之前的临时设定。

### Step 2: Property 推导链（最关键的部分）

```cpp
    // ③ 基础属性：Physical + SharedVA（总是添加）
    createInfo.alloc.property |= Hal::memoryPropertyPhysical |
                                 Hal::memoryPropertySharedVirtualAddress;

    // ④ 如果用户传了 Virtual → 展开一整组属性
    createInfo.alloc.property |= flags & Hal::memoryPropertyVirtual ?
                                  Hal::memoryPropertyDeviceVisible |
                                  Hal::memoryPropertyHostVisible |
                                  Hal::memoryPropertyHostCoherent |
                                  Hal::memoryPropertyDeviceWriteable |
                                  Hal::memoryPropertyDeviceCached :
                                  Hal::memoryPropertyNone;
```

**Property 推导逻辑树**：

```
输入 flags（来自 muapiMemAlloc_v2）
  │
  ├─ bit0: Hal::memoryPropertyVirtual
  │    └─ YES → 自动推导:
  │         DeviceVisible  → GPU 可以通过 VA 访问
  │         HostVisible    → CPU 可以通过 VA 访问
  │         HostCoherent   → CPU 写入对 GPU 可见（一致性）
  │         DeviceWriteable → GPU 可以写入
  │         DeviceCached   → GPU 端走缓存（不是 write-combining）
  │
  ├─ bit1: Hal::memoryPropertyDeviceMapped（来自 muapiMemAlloc_v2）
  │    └─ YES → 允许在其他 GPU 上建立 peer 映射
  │
  ├─ bit2: Hal::memoryPropertySubAllocatable（来自 muapiMemAlloc_v2）
  │    └─ YES → 走 MemMgr::Allocate（pool 子分配）
  │    └─ NO  → 走 Hal::CreateMemory（裸 KMD 分配）
  │
  └─ 总是添加:
       memoryPropertyPhysical         → 物理连续内存
       memoryPropertySharedVirtualAddress → 共享虚拟地址（SVM）
```

**为什么需要 memoryPropertySharedVirtualAddress**：
MUSA 驱动的 GPU 使用 SVM（Shared Virtual Addressing）模型，CPU 和 GPU 共享同一个虚拟地址空间。设置此 flag 后，M3D 层会将 `vaRange = VaRange::Svm` + `svmAlloc = true`，确保分配的 VA 在 CPU 和 GPU 端都可见。这是 MUSA 与传统 CUDA（分离 VA 空间）的核心区别之一。

### Step 3: View Capability 推导

```cpp
    createInfo.alloc.viewCapability = Hal::memoryViewCapabilityExportable;  // 总是可导出

    createInfo.alloc.viewCapability |= (flags & Hal::memoryPropertyDeviceMapped) ?
                                        (Hal::memoryViewCapabilityPeerAccessible |
                                         Hal::memoryViewCapabilityIpcExportable) :
                                        Hal::memoryViewCapabilityNone;
```

**逻辑**：
- 基础：`Exportable`（允许导出 external handle）
- 如果 `DeviceMapped`：额外添加 `PeerAccessible`（peer GPU 可访问）+ `IpcExportable`（IPC 可导出）

### Step 4: Size + Alignment

```cpp
    createInfo.alloc.size = size;
    createInfo.alloc.alignment = std::max(alignment,
        static_cast<size_t>(m_Context->GetParentDevice()->Hal()
            .GetProperties().memoryProperties.memAllocAlignment));
```

**memAllocAlignment**：来自 KMD 导出的设备属性，通常为 4KB 或 64KB。用户传的 alignment 如果小于这个值，会被提升到 KMD 要求的最小值。

### Step 5: 分流 — SubAllocation vs Direct KMD

```cpp
    if (createInfo.alloc.property & Hal::memoryPropertySubAllocatable) {
        // Path A：子分配（默认路径）
        status = HalToMuResult(
            m_Context->GetParentDevice()->Hal().GetMemMgr()->Allocate(
                createInfo.alloc, &m_Offset, &m_pHalMemory));
    } else {
        // Path B：直接 KMD 分配
        status = HalToMuResult(
            m_Context->GetParentDevice()->Hal().CreateMemory(
                createInfo, &m_pHalMemory));
    }
```

**两条路径的本质区别**：

| 维度 | Path A: SubAllocation | Path B: Direct KMD |
|------|-----------------------|--------------------|
| **调用** | `MemMgr::Allocate()` | `Device::CreateMemory()` |
| **KMD 次数** | 首次 pool 空时 1 次，后续 0 次 | 每次 2 次（Alloc + Map） |
| **分配来源** | 从预分配的 2MB chunk 中切分 | 直接向 KMD 申请 |
| **适用场景** | 默认，小分配 | 大分配，特殊 flags |
| **offset 处理** | 返回 chunk 内偏移 | 偏移始终为 0 |
| **底层对象** | 共享 Hal::IMemory（chunk） | 独占 Hal::IMemory |

---

## 三、PitchedGeneralAlloc — 3D 内存分配

**文件**: `memory.cpp:499-530`

```cpp
MUresult Memory::PitchedGeneralAlloc(size_t widthInBytes, size_t height, size_t depth, size_t pitch) {
    m_Shape = { widthInBytes, height, depth, pitch };

    Hal::MemoryCreateInfo createInfo{};
    createInfo.type = Hal::memoryTypeAlloc;
    createInfo.alloc.type = Hal::memoryAllocTypeDeviceLocal;
    createInfo.alloc.heap = Hal::MemoryHeap::largePage;
    createInfo.alloc.property = Hal::memoryPropertyPhysical
                              | Hal::memoryPropertyVirtual
                              | Hal::memoryPropertyHostVisible
                              | Hal::memoryPropertyHostCoherent
                              | Hal::memoryPropertyDeviceVisible
                              | Hal::memoryPropertyDeviceMapped
                              | Hal::memoryPropertyDeviceWriteable
                              | Hal::memoryPropertyDeviceCached
                              | Hal::memoryPropertySharedVirtualAddress;
    createInfo.alloc.property |= Hal::memoryPropertySubAllocatable;   // ← 总是启用 sub-alloc
```

**与 GeneralAlloc 的关键区别**：

1. **Property 是硬编码的**：不像 GeneralAlloc 那样从用户 flags 推导，而是直接写死了一组属性（全部打开）。这意味着 `PitchedGeneralAlloc` 总是创建 GPU 可见 + CPU 可见 + 一致性 + 可缓存的内存。

2. **总是启用 SubAllocatable**：`createInfo.alloc.property |= Hal::memoryPropertySubAllocatable;` 硬编码启用 pool。

3. **Size = pitch × height × depth**：线性化为一维 size。

4. **Overflow 检查**（`memory.cpp:521`）：
   ```cpp
   if (createInfo.alloc.size < pitch || createInfo.alloc.size < height || createInfo.alloc.size < depth)
       status = MUSA_ERROR_OUT_OF_MEMORY;
   ```
   这是一个**粗略的溢出检查**：如果 `pitch * height * depth` 在 64 位乘法中回绕溢出，结果会变小，小于任何一个乘数时就能检测出来。

---

## 四、PinnedHostAlloc — 页锁定主机内存

**文件**: `memory.cpp:532-609`

### 整段代码：

```cpp
MUresult Memory::PinnedHostAlloc(size_t size, unsigned int flags, int numaId) {
    MUresult status = MUSA_SUCCESS;

    // ① 参数校验
    const Hal::DeviceProperties& deviceProperties = m_Context->GetParentDevice()->Hal().GetProperties();
    if ((flags & MU_MEMORY_HOSTALLOC_PORTABLE) && !deviceProperties.memoryProperties.unifiedAddressing) {
        tprintf(LOG_WARN, "...");   // 只是 warning，不失败
    }
    if ((flags & MU_MEMORY_HOSTALLOC_DEVICEMAP) && !deviceProperties.memoryProperties.canMapHostMemory) {
        tprintf(LOG_ERR, "device cannot access the host memory\n");
        status = MUSA_ERROR_NOT_SUPPORTED;
    }
    if (deviceProperties.memoryProperties.canMapHostMemory) {
        flags |= MU_MEMORY_HOSTALLOC_DEVICEMAP;  // 设备支持映射 → 自动启用
    }

    // ② 构造 createInfo
    if (status == MUSA_SUCCESS) {
        static Hal::MemoryHeap pinnedHostAllocHeap = Hal::MemoryHeap::largePage;  // ③ 注意：static!

        Hal::MemoryCreateInfo createInfo{};
        createInfo.type = Hal::memoryTypeAlloc;
        createInfo.alloc.type = Hal::memoryAllocTypeHost;         // ← Host 类型!
        createInfo.alloc.heap = pinnedHostAllocHeap;              // ← static 变量

        // 基础属性
        createInfo.alloc.property = Hal::memoryPropertyPhysical |
                                     Hal::memoryPropertySharedVirtualAddress |
                                     Hal::memoryPropertyHostMapped;

        // 根据 DEVICEMAP 展开
        createInfo.alloc.property |= (flags & MU_MEMORY_HOSTALLOC_DEVICEMAP) ?
                                     Hal::memoryPropertyVirtual |
                                     Hal::memoryPropertyHostVisible |
                                     Hal::memoryPropertyDeviceVisible |
                                     Hal::memoryPropertyDeviceWriteable |
                                     Hal::memoryPropertyDeviceCached :
                                     Hal::memoryPropertyIsPhysAlloc;   // ← 纯物理分配

        // WriteCombined / Coherent
        createInfo.alloc.property |= (flags & MU_MEMORY_HOSTALLOC_WRITECOMBINED) ?
                                     Hal::memoryPropertyHostWriteCombined : Hal::memoryPropertyHostCoherent;

        // SubAllocatable
        createInfo.alloc.property |= (flags & MU_MEMORY_HOSTALLOC_SUBALLOCATABLE) ?
                                     Hal::memoryPropertySubAllocatable : Hal::memoryPropertyNone;

        // NUMA
        createInfo.alloc.property |= (flags & MU_MEMORY_HOSTALLOC_NUMA_CURRENT) ?
                                     Hal::memoryPropertyThreadNumaAffinitive : Hal::memoryPropertyNone;

        // view capability
        createInfo.alloc.viewCapability = (flags & MU_MEMORY_HOSTALLOC_DEVICEMAP) ?
                                           Hal::memoryViewCapabilityPeerAccessible |
                                           Hal::memoryViewCapabilityIpcExportable |
                                           Hal::memoryViewCapabilityExportable :
                                           Hal::memoryViewCapabilityNone;

        createInfo.alloc.size = size;
        createInfo.alloc.alignment = m_Context->GetParentDevice()->Hal()
            .GetProperties().memoryProperties.memAllocAlignment;
        createInfo.alloc.numaId = numaId;

        // ④ 分流
        if (createInfo.alloc.property & Hal::memoryPropertySubAllocatable) {
            status = HalToMuResult(m_Context->GetParentDevice()->Hal()
                .GetMemMgr()->Allocate(createInfo.alloc, &m_Offset, &m_pHalMemory));
        } else {
            status = HalToMuResult(m_Context->GetParentDevice()->Hal()
                .CreateMemory(createInfo, &m_pHalMemory));
        }

        // ⑤ LargePage → General 降级（PH1 临时方案）
        if (pinnedHostAllocHeap == Hal::MemoryHeap::largePage &&
            status == MUSA_ERROR_MAP_FAILED) {
            tprintf(LOG_MEM, "modify the heap from LargePage to General permanently\n");
            pinnedHostAllocHeap = Hal::MemoryHeap::general;  // 永久切换
            createInfo.alloc.heap = pinnedHostAllocHeap;
            // ... 重新分配 ...
        }
    }
    return status;
}
```

### 关键设计点解析：

**1. `static Hal::MemoryHeap pinnedHostAllocHeap`**

```cpp
static Hal::MemoryHeap pinnedHostAllocHeap = Hal::MemoryHeap::largePage;
```

这是一个**函数级静态变量**，在整个进程生命周期内只初始化一次。它的作用是：
- 首次调用尝试 `LargePage` 堆（性能最优）
- 如果 `LargePage` 失败（`MUSA_ERROR_MAP_FAILED`），永久降级为 `General` 堆
- 后续所有调用都使用降级后的堆

**注意**：代码注释 `// TODO(Hongwei.Liu): Enable this logic only before PH1` 说明这是 PH1 之前的临时方案。

**2. `alloc.type = Hal::memoryAllocTypeHost`**

与 `GeneralAlloc` 使用 `memoryAllocTypeDeviceLocal` 不同，`PinnedHostAlloc` 指定为 `memoryAllocTypeHost`。这导致 M3D 层走 `InitGeneralHostMemory` 路径：
```
Memory::Init → type=memoryTypeAlloc → alloc.type=memoryAllocTypeHost
  → InitGeneralHostMemory() → 创建 SVM 内存
```

**3. 属性推导逻辑的对比**：

```
GeneralAlloc (设备内存):
  property = Physical | SharedVA
            | Virtual→(DeviceVis|HostVis|HostCoh|DeviceWr|DeviceCache)
            | (DeviceMapped → PeerAcc|IpcExp)

PinnedHostAlloc (主机内存):
  property = Physical | SharedVA | HostMapped
            | Devicemap→(Virtual|HostVis|DeviceVis|DeviceWr|DeviceCache)
            | (!Devicemap → IsPhysAlloc)    ← 纯物理标识
            | (WriteCombined → HostWrComb) | (→ HostCoh)
            | (SubAllocatable) | (NumaCurrent)
```

**4. LargePage → General 降级逻辑**：

```
尝试 LargePage 堆:
  ├─ 成功 → 返回
  └─ 失败 (MAP_FAILED) → 永久切换为 General 堆 → 重试
```

降级原因：某些系统在 `LargePage` 堆耗尽时会返回 `MAP_FAILED`（VA 不足），而非 `OUT_OF_MEMORY`。此时切换到 `General` 堆可以绕过 VA 限制。

---

## 五、PinnedHostRegister — 注册已有主机内存

**文件**: `memory.cpp:611-671`

### 整段代码的关键结构：

```cpp
MUresult Memory::PinnedHostRegister(void* ptr, size_t size, unsigned int flags) {
    // ① 校验设备能力
    if (!deviceProperties.memoryProperties.hostRegisterSupported)
        return MUSA_ERROR_NOT_SUPPORTED;
    if ((flags & MU_MEMORY_REGISTER_READ_ONLY) && !supportReadonlyHostRegister)
        return MUSA_ERROR_NOT_SUPPORTED;

    // ② 重叠检查（可选）
    if (flags & MU_MEMORY_REGISTER_OVERLAP_CHECK) {
        status = Platform::Get().GetMemoryTracker()
            .IterateMemories(ptr, size, false, overlapCheck);
    }

    // ③ 双路径分流
    if (flags & MU_MEMHOSTREGISTER_IOMEMORY) {
        // 路径 A: I/O Memory (MMIO)
        createInfo.view.type = Hal::memoryViewTypeExternal;
        createInfo.view.external.type = Hal::MemoryExternalHandleType::mmio;
        createInfo.view.external.handle.data.u64Data = (MUdeviceptr)(ptr);  // ← 指针作为 handle
    } else {
        // 路径 B: Locked Memory
        createInfo.view.type = Hal::memoryViewTypeLocked;
        createInfo.view.locked.pHost = ptr;
        createInfo.view.locked.size = size;
        createInfo.view.locked.isDeviceMapped = (flags & MU_MEMORY_REGISTER_DEVICEMAP);
        createInfo.view.locked.isDeviceReadOnly = (flags & MU_MEMORY_REGISTER_READ_ONLY);
        createInfo.view.locked.isPeerAccessible = (flags & MU_MEMORY_REGISTER_PORTABLE) || unifiedAddressing;
    }

    // ④ 统一调用 CreateMemory
    status = HalToMuResult(m_Context->GetParentDevice()->Hal().CreateMemory(createInfo, &m_pHalMemory));
}
```

### 两条路径的本质：

| 维度 | Locked 路径 | I/O Memory 路径 |
|------|-------------|-----------------|
| **适用场景** | 普通的 malloc/memalign 分配的内存 | 映射到 PCI BAR 的设备寄存器 |
| **底层机制** | OS 将 host VA 范围 pin 住，建立 GPU VA → host PA 映射 | 将 PCI BAR 的物理地址映射到 GPU VA |
| **host 指针** | `createInfo.view.locked.pHost = ptr` | `handle.data.u64Data = (MUdeviceptr)ptr` |
| **size** | 需要传入 | size=0（BAR 大小由 OS 决定） |
| **peer access** | 可选（通过 `MU_MEMORY_REGISTER_PORTABLE`） | 自动允许（`isPeerAccessible=true`） |
| **device map** | 需要 `MU_MEMORY_REGISTER_DEVICEMAP` | 总是允许（`isDeviceMapped=true`） |

### 关键校验逻辑：

```
flags 检查顺序:
  1. hostRegisterSupported? → 不支持直接 NOT_SUPPORTED
  2. READ_ONLY + !supportReadonly? → NOT_SUPPORTED
  3. canMapHostMemory? → 自动追加 DEVICEMAP
  4. overlapCheck? → 遍历 MemoryTracker 查重
```

---

## 六、ManagedAlloc — 统一内存

**文件**: `memory.cpp:673-713`

```cpp
MUresult Memory::ManagedAlloc(size_t size, unsigned int flags) {
    // ① 特殊 flag 校验
    if (flags == MU_MEM_ATTACH_HOST && !concurrentManagedAccess)
        return MUSA_ERROR_NOT_SUPPORTED;
    if (flags == MU_MEM_ATTACH_GLOBAL && !unifiedAddressing) {
        flags &= ~MU_MEM_ATTACH_GLOBAL;  // 静默移除，不报错
    }

    // ② alloc type 选择
    createInfo.alloc.type = Platform::Get().GetSettings().managedForceDeviceAlloc
                            ? Hal::memoryAllocTypeDeviceLocal
                            : Hal::memoryAllocTypeHost;

    // ③ 全属性集合
    createInfo.alloc.property = Physical | Virtual | HostVisible | HostMapped |
                                DeviceVisible | DeviceWriteable | DeviceMapped |
                                HostCoherent | DeviceCached | SharedVirtualAddress;

    // ④ 总是走 Hal::CreateMemory（无 sub-allocation）
    status = HalToMuResult(m_Context->GetParentDevice()->Hal()
        .CreateMemory(createInfo, &m_pHalMemory));

    // ⑤ 成功后立即 map 到 host
    if (status == MUSA_SUCCESS) {
        GetHostPointer();  // 立即映射 CPU 端地址
    }
}
```

**关键设计点**：

1. **`managedForceDeviceAlloc` 标志**：这是一个全局设置，决定了 managed 内存是分配在设备本地还是主机端。
   - `true` → `memoryAllocTypeDeviceLocal`：GPU 端性能更好，CPU 端通过 SVM 访问
   - `false` → `memoryAllocTypeHost`：分配在主机端，GPU 通过 bus 访问

2. **ManagedAlloc 不支持 sub-allocation**：代码中没有 `SubAllocatable` flag 的判断，始终走 `Hal::CreateMemory`。这是因为 managed 内存需要特殊的 page fault 机制来支持按需迁移，pool 的简单切分模型不适用。

3. **`GetHostPointer()` 调用**：分配成功后立即映射 CPU 端虚拟地址，确保后续 CPU 访问不会触发 page fault。

4. **`MU_MEM_ATTACH_GLOBAL` 静默降级**：如果设备不支持 unified addressing，不会报错，而是静默移除该 flag。这是用户友好设计。

5. **`MU_MEM_ATTACH_HOST` 严格校验**：如果设备不支持 concurrent managed access（`concurrentManagedAccess=false`），直接返回 `NOT_SUPPORTED`。这是硬性限制。

---

## 七、IpcImportAlloc — IPC 导入内存

**文件**: `memory.cpp:715-758`

```cpp
MUresult Memory::IpcImportAlloc(IpcHandleInternal handle, unsigned int flags) {
    // ① flag 白名单校验
    switch(flags) {
        case MU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS:
        case MU_IPC_MEM_LAZY_UNIDIR_INTERLEAVE_PEER_ACCESS:
        ...
        case MU_IPC_MEM_LAZY_PCIE_PEER_ACCESS:
            break;
        default:
            status = MUSA_ERROR_INVALID_VALUE;
    }

    // ② 同进程检查
    if (handle.pid == Util::Os::GetProcessId() && !handle.fromMempool)
        return MUSA_ERROR_INVALID_CONTEXT;

    // ③ 构造 createInfo
    createInfo.view.type = Hal::memoryViewTypeExternal;
    createInfo.view.external.type = Hal::MemoryExternalHandleType::kernelManagedGlobal;
    createInfo.view.external.handle = handle.halHandle;   // ← 来自导出方的 global handle
    createInfo.view.external.isDeviceMapped = true;
    createInfo.view.external.isPeerAccessible = true;
    createInfo.view.external.isUva = true;
    createInfo.view.external.heap = handle.heap;          // ← 使用导出方指定的 heap
```

**关于 `handle.fromMempool` 的重要逻辑**：

```cpp
if (handle.pid == Util::Os::GetProcessId() && !handle.fromMempool)
    return MUSA_ERROR_INVALID_CONTEXT;
```

- `handle.pid == GetProcessId()`：导入方和导出方在同一个进程
- `handle.fromMempool == false`：这块内存不是从 pool 导出的

**只有同一进程且非 pool 导出时才拒绝**。原因是：
- 同一进程内的 pool 子分配不需要 IPC 导入，可以直接用指针（已在同一虚拟地址空间）
- 跨进程或 pool 导出的内存才需要走 IPC 导入流程

---

## 八、ExternalAlloc — 外部内存

**文件**: `memory.cpp:760-799`

```cpp
MUresult Memory::ExternalAlloc(const ExternalMemoryInfo& externalInfo) {
    m_HandleType = externalInfo.handleInfo.type;

    switch (externalInfo.handleInfo.type) {
        case MemoryHandleType::Fabric:
            createInfo.view.external.type = Hal::MemoryExternalHandleType::fabric;
            createInfo.view.external.handle.data.u64Data = (uint64_t)(&fabric);
            break;
        case MemoryHandleType::OpaqueFd:
            createInfo.view.external.type = Hal::MemoryExternalHandleType::dmaBuf;
            createInfo.view.external.handle.data.intData = fd;
            createInfo.view.external.size = size;
            break;
    }

    createInfo.view.external.isPeerAccessible = true;   // ← 总是允许 peer 访问
    createInfo.view.external.heap = fromGraphics
                                    ? Hal::MemoryHeap::general
                                    : Hal::MemoryHeap::largePage;
```

**两个 handle 类型的区别**：

| 维度 | OpaqueFd (DMA-Buf) | Fabric |
|------|---------------------|--------|
| **来源** | 通过 `MU_MEM_ALLOCATION_TYPE_DMABUF` 导出 | 通过 `MU_MEM_ALLOCATION_TYPE_FABRIC` 导出 |
| **handle 字段** | `intData` (文件描述符) | `u64Data` (fabric 结构体指针) |
| **heap** | 默认 `largePage` | `general` |
| **典型用途** | 跨进程共享 GPU buffer | 跨 fabric 拓扑的内存共享 |

---

## 九、VirtualAlloc — 虚拟地址分配

**文件**: `memory.cpp:805-825`

```cpp
MUresult Memory::VirtualAlloc(size_t size, size_t alignment, MUdeviceptr uniAddr, unsigned long long flags) {
    if (flags != 0) {
        status = MUSA_ERROR_INVALID_VALUE;  // ← 当前 flags 必须为 0
    } else {
        Hal::MemoryCreateInfo createInfo{};
        createInfo.type = Hal::memoryTypeVirtual;          // ← 虚拟类型
        createInfo.virt.heap = Hal::MemoryHeap::largePage;
        createInfo.virt.size = size;
        createInfo.virt.addr = uniAddr;                    // ← 指定的统一地址
        createInfo.virt.alignment = alignment;

        status = HalToMuResult(Platform::Get().Hal()
            .CreateMemory(createInfo, &m_pHalMemory));    // ← 不走 MemMgr
    }
}
```

**关键限制**：
1. `flags` 当前必须为 0，否则返回 `INVALID_VALUE`。这说明 virtual allocation 还没有扩展属性的支持。
2. `createInfo.type = Hal::memoryTypeVirtual` — M3D 层只预留虚拟地址空间，不分配物理内存。
3. 物理内存在后续通过 `Memory::Bind()` 绑定（`memory.cpp:152`）。

---

## 十、Destructor — 析构函数中的两条释放路径

**文件**: `memory.cpp:358-376`

```cpp
Memory::~Memory() {
    if (m_pMapped) {
        m_pHalMemory->Unmap();
    }
    if (m_Type != memoryTypePrealloc && m_pHalMemory) {
        if (m_pHalMemory->GetProps() & Hal::memoryPropertySubAllocatable) {
            // 路径 A：Sub-Allocation 释放
            if (m_pPool == nullptr) {
                // 通过 MemMgr 释放（GeneralAlloc 默认路径）
                m_Context->GetParentDevice()->Hal().GetMemMgr()->Free(
                    m_pHalMemory, GetDevicePointer(), GetSize());
            } else {
                // 通过用户指定的 pool 释放
                m_pPool->Hal()->Free(
                    m_pHalMemory, GetDevicePointer(), GetSize());
            }
        } else {
            // 路径 B：裸 KMD 释放
            m_PhysTracker.Cleanup();           // 清理物理内存追踪
            m_pHalMemory->Destroy();           // 调用 KMD Destroy
        }
    }
}
```

**两条释放路径的核心区别**：

| 维度 | SubAllocatable 释放 | 非 SubAllocatable 释放 |
|------|----------------------|------------------------|
| **判断依据** | `m_pHalMemory->GetProps() & memoryPropertySubAllocatable` | 同上，取反 |
| **释放调用** | `MemMgr::Free()` 或 `Pool::Free()` | `m_pHalMemory->Destroy()` |
| **offset 传参** | 需要传 `GetDevicePointer()` 和 `GetSize()` | 不需要 |
| **PhysTracker** | 不需要清理 | `m_PhysTracker.Cleanup()` |
| **底层行为** | 将段标记为空闲，可能延迟销毁 chunk | 直接销毁整个 KMD 分配对象 |
| **m_pPool 来源** | `InitFromPool()` 设置 | `GeneralAlloc()` 时为 nullptr |

**为什么 `InitFromPool` 强制设置 SubAllocatable**：

```cpp
// memory.cpp:449
m_pHalMemory->SetProps(Hal::memoryPropertySubAllocatable);
```

`InitFromPool` 的内存一定来自 pool（由 `muMemPoolCreate` + `muMemAllocFromPool` 创建），所以析构时必须走 `Pool::Free` 而非 `Destroy`。通过在 hal memory 上标记 `SubAllocatable` 属性，析构函数能自动判断释放路径。

---

## 十一、InitFromPool — pool 分配路径

**文件**: `memory.cpp:427-453`

```cpp
MUresult Memory::InitFromPool(MemoryPool* pPool, size_t size) {
    Hal::IMemoryPool* pHalPool = pPool->Hal();
    Hal::MemoryAllocInfo allocInfo{};
    allocInfo.type           = pHalPool->GetInfo().type;       // 继承 pool 的类型
    allocInfo.heap           = pHalPool->GetInfo().heap;       // 继承 pool 的 heap
    allocInfo.property       = pHalPool->GetInfo().property;   // 继承 pool 的 property
    allocInfo.viewCapability = pHalPool->GetInfo().viewCapability;
    allocInfo.size           = size;
    allocInfo.alignment      = 0;

    Hal::IMemory* pHalMemory = nullptr;
    size_t        offset     = 0;
    status = HalToMuResult(pHalPool->FullAllocate(allocInfo, &offset, &pHalMemory));

    if (status == MUSA_SUCCESS) {
        m_Type = memoryTypeVirtual;             // ← 类型设为 virtual!
        m_Shape = {size, 1, 1, size};
        m_Offset = offset;                      // ← 保存 chunk 内偏移
        m_pHalMemory = pHalMemory;              // ← 共享 chunk 的 IMemory
        m_pPool = pPool;                        // ← 保存 pool 指针
        m_pHalMemory->SetProps(Hal::memoryPropertySubAllocatable);
    }
}
```

**与 GeneralAlloc 的关键区别**：

| 维度 | InitFromPool | GeneralAlloc (SubAlloc) |
|------|-------------|------------------------|
| **入口** | `muMemAllocFromPool(pool, &dptr, size)` | `muMemAlloc(&dptr, size)` |
| **m_Type** | `memoryTypeVirtual` | `memoryTypeGeneral` |
| **pool 信息** | 从 `pPool` 继承 (type, heap, property) | 从用户 flags 推导 |
| **alignment** | 传 0 | 用户指定 + memAllocAlignment |
| **m_pPool** | 设置 | nullptr |
| **Offset** | 保存（chunk 内偏移） | 保存（SubAllocate 返回） |

**为什么 `m_Type = memoryTypeVirtual`**：
pool 子分配返回的内存本质上是 chunk 中的一段区间（offset + size），不是一块完整的、物理上独立的分配。因此在 MUSA 框架中将其视为"虚拟"内存。真正的物理内存在析构时才通过 `Pool::Free` 释放回 chunk。

---

## 十二、所有 API 的 Flags 对比表

### GeneralAlloc（设备内存）

| Flag | 值 | 来源 |
|------|-----|------|
| `Physical` | ✅ 总是添加 | `memory.cpp:472` |
| `SharedVirtualAddress` | ✅ 总是添加 | `memory.cpp:473` |
| `Virtual` | 用户传 | `muapiMemAlloc_v2` |
| `DeviceVisible` | Virtual?是:否 | `memory.cpp:474-480` |
| `HostVisible` | Virtual?是:否 | 同上 |
| `HostCoherent` | Virtual?是:否 | 同上 |
| `DeviceWriteable` | Virtual?是:否 | 同上 |
| `DeviceCached` | Virtual?是:否 | 同上 |
| `DeviceMapped` | 用户传 | `muapiMemAlloc_v2` |
| `PeerAccessible` | DeviceMapped?是:否 | `memory.cpp:483-485` |
| `IpcExportable` | DeviceMapped?是:否 | 同上 |
| `SubAllocatable` | 用户传 | `muapiMemAlloc_v2` |

### PinnedHostAlloc（主机内存）

| Flag | 值 | 来源 |
|------|-----|------|
| `Physical` | ✅ 总是添加 | `memory.cpp:563` |
| `SharedVirtualAddress` | ✅ 总是添加 | 同上 |
| `HostMapped` | ✅ 总是添加 | `memory.cpp:565` |
| `Virtual` | DeviceMap?是:否 | `memory.cpp:566-572` |
| `DeviceVisible` | DeviceMap?是:否 | 同上 |
| `DeviceWriteable` | DeviceMap?是:否 | 同上 |
| `DeviceCached` | DeviceMap?是:否 | 同上 |
| `HostWriteCombined` | 用户传（替代 HostCoherent） | `memory.cpp:573` |
| `SubAllocatable` | 用户传 `MU_MEMORY_HOSTALLOC_SUBALLOCATABLE` | `memory.cpp:574` |
| `IsPhysAlloc` | !DeviceMap 时添加 | `memory.cpp:572` |

### ManagedAlloc

| Flag | 值 | 来源 |
|------|-----|------|
| `Physical` | ✅ | 硬编码 |
| `Virtual` | ✅ | 硬编码 |
| `HostVisible` | ✅ | 硬编码 |
| `HostMapped` | ✅ | 硬编码 |
| `HostCoherent` | ✅ | 硬编码 |
| `DeviceVisible` | ✅ | 硬编码 |
| `DeviceWriteable` | ✅ | 硬编码 |
| `DeviceCached` | ✅ | 硬编码 |
| `DeviceMapped` | ✅ | 硬编码 |
| `SharedVirtualAddress` | ✅ | 硬编码 |
| `SubAllocatable` | ❌ 不设置 | 不支持 pool |

### PitchedGeneralAlloc

| Flag | 值 | 来源 |
|------|-----|------|
| 所有属性与 GeneralAlloc 一致 | ✅ | 硬编码（不依赖用户 flags） |
| `SubAllocatable` | ✅ 总是添加 | `memory.cpp:516` |

---

## 十三、`m_Offset` 的含义

`m_Offset` 的值根据分配路径不同而不同：

| 路径 | `m_Offset` 值 | 含义 |
|------|---------------|------|
| `GeneralAlloc` (SubAlloc) | `subAllocAddr - chunkBase` | 子分配段在 chunk 内的偏移 |
| `GeneralAlloc` (Direct KMD) | 0 | KMD 分配的基地址就是起始地址 |
| `InitFromPool` | `offset` (FullAllocate 返回) | 子分配段在 chunk 内的偏移 |
| `PinnedHostRegister` (Locked) | 0（通常） | |
| `IpcImportAlloc` | `handle.allocOffset` | 导出时记录的偏移 |
| `ExternalAlloc` | `externalInfo.viewOffset` | 视图偏移量 |

最终设备指针的计算：
```cpp
// IMemory::GetDevicePointer()
return GetGpuMemory()->Desc().gpuVirtAddr + m_Offset;
```

**Sub-Allocation 的核心机制**：多个子分配共享同一个底层 chunk（`m_pHalMemory`），通过不同的 `m_Offset` 区分各自的地址范围。

---

## 十四、`m_pPool` 的作用

`m_pPool` 只有在 `InitFromPool` 时才被设置，用于区分两种 Sub-Allocation 的释放路径：

```cpp
// 析构函数中:
if (m_pHalMemory->GetProps() & Hal::memoryPropertySubAllocatable) {
    if (m_pPool == nullptr) {
        // GeneralAlloc (SubAlloc) 路径 → 通过 MemMgr 释放
        m_Context->GetParentDevice()->Hal().GetMemMgr()->Free(
            m_pHalMemory, GetDevicePointer(), GetSize());
    } else {
        // InitFromPool 路径 → 通过用户 pool 释放
        m_pPool->Hal()->Free(
            m_pHalMemory, GetDevicePointer(), GetSize());
    }
}
```

- **`m_pPool == nullptr` + SubAllocatable**：说明内存来自 `GeneralAlloc` 的默认路径。释放时通过 `MemMgr::Free` 找到对应的内部 pool。
- **`m_pPool != nullptr` + SubAllocatable**：说明内存来自 `muMemAllocFromPool`。释放时归还到用户指定的 pool。

两者最终都调用 `MemoryPool::Free()`，区别在于 pool 的生命周期和管理方式不同。

---

## 十五、`PinnedHostAlloc` 的 LargePage→General 降级详解

```cpp
static Hal::MemoryHeap pinnedHostAllocHeap = Hal::MemoryHeap::largePage;
// ↑ static 变量，进程级只初始化一次

// 首次分配尝试 LargePage
createInfo.alloc.heap = pinnedHostAllocHeap;
status = Hal::CreateMemory(createInfo, &m_pHalMemory);

// 如果失败且是 MAP_FAILED（不是 OUT_OF_MEMORY）
if (pinnedHostAllocHeap == Hal::MemoryHeap::largePage &&
    status == MUSA_ERROR_MAP_FAILED) {
    // 永久降级
    pinnedHostAllocHeap = Hal::MemoryHeap::general;
    createInfo.alloc.heap = pinnedHostAllocHeap;
    // 重新分配
}
```

**为什么是 `MAP_FAILED` 而不是 `OUT_OF_MEMORY`**：
- `MAP_FAILED` 意味着物理内存足够，但无法将分配映射到 CPU 虚拟地址空间（VA 耗尽）
- `OUT_OF_MEMORY` 意味着物理内存确实不够
- 从 LargePage 切换到 General 堆可以扩大可用的 VA 范围（General 堆通常使用较小的页面）

**static 变量的副作用**：
- 多线程并发调用时存在数据竞争（多个线程可能同时触发降级逻辑）
- 一旦降级，整个进程永久使用 General 堆，即使后续释放了部分内存

---

## 十六、文件索引

| 文件 | 说明 |
|------|------|
| `musa/src/musa/core/memory.cpp`（827 行） | MUSA Core 层所有内存 API 实现 |
| `musa/src/driver/mu_memory.cpp`（2949 行） | Driver 层 API 入口 |
| `musa/src/hal/m3d/memoryPool.cpp`（533 行） | Sub-Allocation 内存池实现 |
| `musa/src/hal/m3d/memMgr.cpp`（237 行） | MemMgr：pool 管理 + 分配路由 |
| `musa/src/hal/m3d/memory.cpp`（838 行） | Hal 层 Memory 类：InitGeneralDeviceMemory 等 |
| `musa/src/hal/m3d/m3d/src/core/gpuMemory.cpp`（1410 行） | M3D 层 GpuMemory 分配 + OS 相关实现 |
| `musa/doc/memory_api/12_DirectKMD_Allocation_flow.md` | 裸 KMD 分配的完整调用链 |
| `musa/doc/memory_api/13_MemoryPool_deep_dive.md` | MemoryPool 子分配算法详解 |
| 本文档 | MUSA Core 层 Memory API 逐函数深度分析 |