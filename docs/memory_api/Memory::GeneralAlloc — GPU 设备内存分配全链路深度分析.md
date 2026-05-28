# Memory::GeneralAlloc — GPU 设备内存分配全链路深度分析

> 以 `muMemAlloc(&dptr, 4096)` 为例，逐行解析内存是如何从 0 变成一块真实的 GPU 显存。

---

## 一、完整调用链

```
muMemAlloc(&dptr, 4096)
  │
  ├─ muapiMemAlloc_v2  (driver/mu_memory.cpp:265)
  │    │  cpu: 构造 CreateInfo, 设置 flags
  │    └─ Context::CreateMemory(ppMemory, createInfo)
  │         │  cpu: 四步流程 (capture检查 + Init + MapToPeers + 注册)
  │         └─ Memory::Init(createInfo)
  │              │  cpu: 根据 type 分发
  │              └─ Memory::GeneralAlloc(size, alignment, flags)
  │                   │  cpu: 构造 Hal::MemoryCreateInfo, 合并 flags
  │                   │
  │                   ├─ [SubAllocatable] → MemMgr::Allocate
  │                   │    └─ MemoryPool::FullAllocate
  │                   │         ├─ SubAllocate (从空闲链表找)
  │                   │         │   └─ (第一次) → 池为空 → errorNotFound
  │                   │         └─ ChunkAllocate (池为空, 向 KMD 申请一个 chunk)
  │                   │              └─ M3d::Device::CreateMemory
  │                   │                   └─ InitGeneralDeviceMemory
  │                   │                        └─ m3dDevice->CreateGpuMemory  → KMD ioctl → GPU 显存
  │                   │              → SubAllocate (重试, 从 chunk 中切出所需大小)
  │                   │              → return offset + IMemory*
  │                   │
  │                   └─ [非 SubAlloc] → Hal::IDevice::CreateMemory
  │                        └─ InitGeneralDeviceMemory → KMD → GPU 显存
  │
  └─ return dptr = pMemory->GetDevicePointer()
```

---

## 二、逐层代码分析

### Layer 1: Driver 入口 — muapiMemAlloc_v2

```cpp
// mu_memory.cpp:265
MUresult muapiMemAlloc_v2(MUdeviceptr *dptr, size_t bytesize) {
    // ...
    Musa::MemoryCreateInfo createInfo{};
    createInfo.type = Musa::memoryTypeGeneral;
    createInfo.general.size = 4096;
    createInfo.general.alignment = 0;
    createInfo.general.flags = Hal::memoryPropertyVirtual |           // bit 0: GPU 可访问
                               Hal::memoryPropertyDeviceMapped |      // bit 1: 允许 device 映射
                               Hal::memoryPropertySubAllocatable;     // bit 2: 从 pool 切分

    Musa::IMemory* pMemory;
    status = pContext->CreateMemory(&pMemory, createInfo);
    *dptr = pMemory->GetDevicePointer();  // = HalVA + m_Offset
}
```

**flags 组合含义**:
- `Virtual` → 内存可被 GPU 访问 (非纯物理)
- `DeviceMapped` → 允许建立 device 映射
- `SubAllocatable` → **从 pool 中 sub-allocate**, 而非每次都向 KMD 裸分配

### Layer 2: Memory::Init — 类型分发

```cpp
// memory.cpp:378
MUresult Memory::Init(const MemoryCreateInfo& createInfo) {
    m_Type = createInfo.type;  // = memoryTypeGeneral
    switch (m_Type) {
        case memoryTypeGeneral:
            status = GeneralAlloc(createInfo.general.size,      // 4096
                                   createInfo.general.alignment, // 0
                                   createInfo.general.flags);    // Virtual|DeviceMapped|SubAllocatable
            break;
        // ... 其他 8 种类型
    }
}
```

### Layer 3: GeneralAlloc — 构造 HAL 分配描述符

```cpp
// memory.cpp:462
MUresult Memory::GeneralAlloc(size_t size, size_t alignment, unsigned int flags) {
    MUresult status = MUSA_SUCCESS;
    // ① 设置通用 shape (1D = width=size, height=1, depth=1, pitch=size)
    m_Shape = { size, 1, 1, size };
    m_Flags = flags;

    // ② 构造 HAL 层的 MemoryCreateInfo
    Hal::MemoryCreateInfo createInfo{};
    createInfo.type            = Hal::memoryTypeAlloc;          // 真分配, 非 View
    createInfo.alloc.type      = Hal::memoryAllocTypeDeviceLocal; // 设备本地内存 (显存)
    createInfo.alloc.heap      = Hal::MemoryHeap::largePage;      // 大页堆
    createInfo.alloc.size      = 4096;
    createInfo.alloc.alignment = max(0, deviceAlignment);
    createInfo.alloc.numaId    = NUMA_NO_NODE;

    // ③ 合并属性 — 从 flags 推导
    createInfo.alloc.property = flags;
    //    flags = Virtual | DeviceMapped | SubAllocatable
    createInfo.alloc.property |= Hal::memoryPropertyPhysical;              // + 物理
    createInfo.alloc.property |= Hal::memoryPropertySharedVirtualAddress;  // + 共享 VA

    // ★ 关键: 如果 flags 包含 Virtual → 添加 5 个扩展属性
    if (flags & Hal::memoryPropertyVirtual) {   // ← true!
        createInfo.alloc.property |= Hal::memoryPropertyDeviceVisible |
                                     Hal::memoryPropertyHostVisible |
                                     Hal::memoryPropertyHostCoherent |
                                     Hal::memoryPropertyDeviceWriteable |
                                     Hal::memoryPropertyDeviceCached;
    }
    // 否则添加 None (纯物理分配)

    // ④ 设置 view 能力
    createInfo.alloc.viewCapability = Hal::memoryViewCapabilityExportable;
    if (flags & Hal::memoryPropertyDeviceMapped) {   // ← true!
        createInfo.alloc.viewCapability |= Hal::memoryViewCapabilityPeerAccessible |
                                           Hal::memoryViewCapabilityIpcExportable;
    }

    // ⑤ ★ 核心分叉: SubAllocatable → MemMgr, 否则 → CreateMemory
    if (createInfo.alloc.property & Hal::memoryPropertySubAllocatable) {  // ← true!
        status = HalToMuResult(
            m_Context->GetParentDevice()->Hal()
                .GetMemMgr()->Allocate(createInfo.alloc,   // ← 走 sub-allocation 路径
                                        &m_Offset,          // 输出: sub-allocation 偏移
                                        &m_pHalMemory));    // 输出: HAL 内存对象
    } else {
        status = HalToMuResult(
            m_Context->GetParentDevice()->Hal()
                .CreateMemory(createInfo, &m_pHalMemory));  // 裸分配, 直接 KMD
    }
    return status;
}
```

### GeneralAlloc 最终属性总结

| 属性 | 值来源 | 含义 |
|------|--------|------|
| `Physical` | 强制添加 | 分配物理内存 (非仅 VA) |
| `SharedVirtualAddress` | 强制添加 | CPU/GPU 共享 VA (SVM) |
| `Virtual` | flags from driver | GPU 可以访问 |
| `DeviceMapped` | flags from driver | 允许建立 device mapping |
| `SubAllocatable` | flags from driver | 从 pool 中子分配 |
| `DeviceVisible` | 由 Virtual 推导 | GPU 可读 |
| `HostVisible` | 由 Virtual 推导 | CPU 可映射 |
| `HostCoherent` | 由 Virtual 推导 | 保证 CPU/GPU 缓存一致性 |
| `DeviceWriteable` | 由 Virtual 推导 | GPU 可写 |
| `DeviceCached` | 由 Virtual 推导 | GPU L2 可缓存 |
| `viewCapability=Exportable\|PeerAccessible\|IpcExportable` | 由 DeviceMapped 推导 | 支持导出/Peer/IPC |

---

## 三、SubAllocatable 路径 — MemMgr → MemoryPool → ChunkAllocate

### 3.1 MemMgr::Allocate (memMgr.cpp:81)

```cpp
Result MemMgr::Allocate(const MemoryAllocInfo& allocInfo,
                         DevSize* pOffset, IMemory** ppMemory,
                         IMemoryPool** ppMemoryPool) {
    // ① 校验: size==0? alignment overflow? > totalGlobalMem?
    if (allocInfo.size == 0)      → errorInvalidValue
    if (overflow)                  → errorOutOfMemory
    if (size > totalGlobalMem)     → errorOutOfMemory

    // ② 查找或创建 pool (按 allocInfo 的 type/heap/property/viewCapability/numaId 组合)
    // 如果调用方已经指定了 pool → 直接使用
    // 否则 → 根据 allocInfo 在 m_PoolRefs SplayTree 中查找
    // 如果没找到匹配的 pool → CreatePoolNoLock 创建新 pool

    MemoryPoolInfo poolInfo{};
    poolInfo.type           = allocInfo.type;            // memoryAllocTypeDeviceLocal
    poolInfo.heap           = allocInfo.heap;            // largePage
    poolInfo.property       = allocInfo.property;        // Physical|SVA|Virtual|...
    poolInfo.viewCapability = allocInfo.viewCapability;  // Exportable|Peer|Ipc
    poolInfo.numaId         = allocInfo.numaId;          // NUMA_NO_NODE
    pPool = m_PoolRefs.Get(MakeKey(poolInfo))->m_Value;
    // 如果 pool 不存在或属性不匹配 → CreatePool:
    //   new MemoryPool(m_pDevice)
    //   pPool->Init(poolCreateInfo)
    //   m_PoolRefs.Insert(MakeKey(...), pPool)

    // ③ ★ 从 pool 中分配
    res = pPool->FullAllocate(allocInfo, pOffset, ppMemory);
}
```

### 3.2 MemoryPool::FullAllocate (memoryPool.cpp:82)

```cpp
Result MemoryPool::FullAllocate(const MemoryAllocInfo& allocInfo,
                                 DevSize* pOffset, IMemory** ppMemory) {
    std::lock_guard<std::recursive_mutex> lg(m_Lock);

    // 第一步: 尝试从现有空闲块中 sub-allocate
    Result res = SubAllocate(allocInfo, pOffset, ppMemory);
    // 如果成功 → 返回 (从已有 chunk 中切出所需大小)
    // 如果失败 (池为空) → 进入第二步

    if (res == Result::errorNotFound) {
        // 第二步: 分配一个新 chunk (向 KMD 申请一大块)
        res = ChunkAllocate(allocInfo);

        if (res == Result::success) {
            // 第三步: 重试 sub-allocate (从新 chunk 中切分)
            res = SubAllocate(allocInfo, pOffset, ppMemory);
        }
    }
    return res;
}
```

**关键设计**: pool 是"先切后扩"——先尝试从已有空闲块中切，切不出来再向 KMD 申请一大块 chunk，然后从 chunk 中切。

### 3.3 MemoryPool::ChunkAllocate (memoryPool.cpp:153)

```cpp
Result MemoryPool::ChunkAllocate(const MemoryAllocInfo& allocInfo) {
    DevSize chunkSize = allocInfo.size;        // = 4096
    DevSize chunkAlignment = m_ChunkAlignment;  // = 64KB (默认)

    // 计算实际 chunk 大小 (向上对齐)
    if (allocInfo.alignment > chunkAlignment) {
        chunkSize += allocInfo.alignment - chunkAlignment;
    }
    chunkSize = ::Util::AlignUp(chunkSize, chunkAlignment);  // 4096 → 64KB
    chunkSize = ::Util::AlignUp(chunkSize, m_ChunkAllocSize); // 64KB → ...

    // ★ 向 KMD 申请 chunk (和裸分配走同样的 M3D 路径)
    IMemory* pChunkMem = nullptr;
    if (!(m_Info.property & memoryPropertyPhysical)) {
        // Virtual 类型的 chunk → Platform::CreateMemory(memoryTypeVirtual)
        Platform::Get().CreateMemory(chunkCreateInfo, &pChunkMem);
    } else {
        // Physical 类型的 chunk → Device::CreateMemory → InitGeneralDeviceMemory
        m_pDevice->CreateMemory(chunkCreateInfo, &pChunkMem);
        //       ↓
        //    InitGeneralDeviceMemory → m3dDevice->CreateGpuMemory → KMD → GPU 显存
    }

    // 将新 chunk 的整个区域作为空闲段加入 pool
    ResSegment* pMemRes = new ResSegment(base, size);
    pMemRes->pChunkMem = pChunkMem;   // 记录该段的 backing memory
    pMemRes->chunkBase  = base;
    pMemRes->isLeftMost = true;
    pMemRes->isRightMost = true;
    ResourceAdd(pMemRes);  // 加入段链表 + 空闲桶

    m_TotalSize += size;
    m_FreeSize  += size;   // 整块 chunk 都是空闲的
}
```

**ChunkAllocate 关键值**:
```
用户请求: 4096 bytes
chunk 对齐: 64KB (s_DefaultChunkAllocSize)
实际 chunk Size: max(4096, 64KB) = 64KB
向 KMD 分配: 64KB GPU 显存
加入 pool: 64KB 的空闲段
```

### 3.4 MemoryPool::SubAllocate — 从 chunk 中切分 (memoryPool.cpp:97)

```cpp
Result MemoryPool::SubAllocate(const MemoryAllocInfo& allocInfo,
                                DevSize* pOffset, IMemory** ppMemory) {
    // ① 定位: 根据 size/alignment 找到合适的空闲桶
    uint32_t indexLow  = ::Util::Log2(allocInfo.size);     // 4096 = 2^12 → indexLow = 12
    uint32_t indexHigh = (alignment > 1) ? Log2(size + alignment - 1) : indexLow;

    // ② 查找: 在空闲桶中搜索合适的块
    if (policy == assuredFit) {    // 默认策略
        uint32_t index = 0;
        // 用位图查找: m_EltMappingHash 的哪位有可用块
        DevSize optionalMappingHash = ~((1ULL << (indexHigh + 1)) - 1) & m_EltMappingHash;
        if (BitMaskScanForward(&index, optionalMappingHash)) {
            if (index != s_FreeTableLimit) {
                pMemRes = FindBucket(m_FreeBuckets[index], size, alignment,
                                     1);  // tryLimit=1: 只看该桶第一个
            } else {
                // 向下遍历更大桶
                for (index = indexHigh; index >= indexLow; index--) {
                    pMemRes = FindBucket(m_FreeBuckets[index], size, alignment,
                                         UINT64_MAX);
                    if (pMemRes) break;
                }
            }
        }
    }

    if (!pMemRes) return errorNotFound;

    // ③ 切分: 从空闲段中切出所需大小
    DevSize subAllocAddr = 0;
    res = ResourceSplit(pMemRes, allocInfo.size, allocInfo.alignment,
                         &subAllocAddr);
    // ResourceSplit 做的事:
    //   1. 计算对齐后的起始地址: alignedBase = AlignUp(base, alignment)
    //   2. 前面多出的部分 → 分裂为一个新空闲段 (left)
    //   3. 后面剩余的部分 → 分裂为另一个新空闲段 (right)
    //   4. 中间部分 → 标记为 busy, 移除空闲链表
    //
    //   例如 chunk [0, 64KB]:
    //     分配 4096 bytes, alignment=0:
    //       左段 [0, 4096)    → busy (返回)
    //       右段 [4096, 64KB) → free (放回空闲链表)

    // ④ 返回结果
    *ppMemory   = pMemRes->pChunkMem;   // chunk 的 HAL IMemory (指向 64KB KMD 分配)
    *pOffset    = subAllocAddr - pMemRes->chunkBase;  // = 0 (如果是第一个分配)
    m_FreeSize  -= allocInfo.size;       // 更新空闲统计
}
```

**SubAllocate 切分示意**:
```
首次分配 4096 bytes:

Chunk (64KB):  [============================================================]
                          ^ chunkBase=GpuVA

ResourceSplit 后:
  [   ← 4096 → ][ <------------- 剩余 60KB+ --------------> ]
      ↑ busy               ↑ free (放回空闲链表)
    offset=0

返回:
  ppMemory = &chunkHalMemory  (64KB 的 HAL::IMemory)
  pOffset  = 0                (在 chunk 中的偏移)
```

### 3.5 FindBucket — 空闲块搜索算法 (memoryPool.cpp:14)

```cpp
MemoryPool::ResSegment* MemoryPool::FindBucket(ResSegment* first,
                                                DevSize size, DevSize alignment,
                                                DevSize tryLimit) {
    for (walker = first; walker && tryLimit != 0; walker = walker->pNextFree) {
        const DevSize alignedBase =
            (alignment > 1) ? AlignUp(walker->base, alignment) : walker->base;

        if (walker->base + walker->size >= alignedBase + size) {
            return walker;  // 找到! 此块足够大
        }
        if (tryLimit != UINT64_MAX) tryLimit--;
    }
    return nullptr;
}
```

**内存池数据结构**:
```
MemoryPool
 ├── m_FreeBuckets[0..63]    // 按 size log2 分桶的空闲段链表
 │    自由桶: 链表头指针数组, 按 2^log2(size) 分桶
 │    例如: bucket[12] → 4KB-8KB 的空闲段链表
 │          bucket[16] → 64KB-128KB 的空闲段链表
 │
 ├── m_EltMappingHash        // 位图: bit i=1 表示 bucket[i] 非空
 │    加快查找: 一步确定哪个桶有可用块
 │
 ├── m_pHeadSegment          // 按地址排序的双向链表 (所有段)
 │    [free,4KB] ⇄ [busy,4KB] ⇄ [free,60KB]  ← 按 VA 排序
 │
 ├── m_SegmentTracker       // std::map<range, ResSegment*> 地址查找
 │
 ├── m_FreeSize              // 当前空闲字节总数
 ├── m_TotalSize             // pool 总容量
 └── m_ChunkAlignment        // = 64KB (s_DefaultChunkAllocSize)
```

---

## 四、KMD 分配路径 (ChunkAllocate → InitGeneralDeviceMemory)

当 pool 需要新 chunk 时, 走和裸分配相同的路径:

```cpp
// hal/m3d/memory.cpp:366
Result Memory::InitGeneralDeviceMemory(const MemoryCreateInfo& createInfo) {
    // ① 计算对齐 (device min align + page size)
    alignment = max(deviceAllocAlign, allocAlignment);
    alignment = AlignUp(alignment, heap->largestPageSize);

    // ② 填充 M3D GpuMemoryCreateInfo
    IM3d::GpuMemoryCreateInfo info = {};
    info.size      = chunkSize;                    // 64KB
    info.alignment = alignment;                    // 对齐
    info.heapAccess = GpuHeapAccessExplicit;
    info.heaps[0]  = GpuHeapLocal;                 // GPU 本地显存
    info.heapCount = 1;
    info.flags.peerWritable    = canMapPeerMemory;
    info.flags.physicalAlloc   = (property & Virtual) == 0;  // = true: 物理分配
    info.flags.discontinuousAlloc = 1;                         // 允许非连续
    info.flags.svmAlloc       = (property & SharedVA) != 0;  // = true
    info.vaRange              = largePage ? Svm : Default;     // = Svm

    // ③ 获取内存对象大小
    gpuMemObjSize = m3dDevice->GetGpuMemorySize(info, &m3dRes);

    // ④ 分配 M3D GPU 内存对象
    m_M3dGpuMemory.Reserve(gpuMemObjSize);
    result = m3dDevice->CreateGpuMemory(info,
                                         m_M3dGpuMemory.GetAlloc(),
                                         &m_M3dGpuMemory());
    // ★ 这是真正的 KMD 调用 → DRM ioctl → 分配物理页面 + 建立 GPU 页表

    // ⑤ 如果需要 host mapped → 做 CPU mmap
    if (property & memoryPropertyHostMapped) {
        result = Map(0, m_Size, &m_pMappedHost);
    }
}
```

---

## 五、时序图

```
Driver(mu_mem)     Core(Context)      Core(Memory)        HAL(MemMgr)       HAL(MemoryPool)    HAL/M3D(Memory)      KMD/GPU
  │                  │                  │                   │                 │                   │                     │
  │ muapiMemAlloc    │                  │                   │                 │                   │                     │
  │─────────────────>│                  │                   │                 │                   │                     │
  │ flags=Virtual|   │                  │                   │                 │                   │                     │
  │ DeviceMapped|    │                  │                   │                 │                   │                     │
  │ SubAllocatable   │                  │                   │                 │                   │                     │
  │                  │                  │                   │                 │                   │                     │
  │                  │ CreateMemory     │                   │                 │                   │                     │
  │                  │─────────────────>│                   │                 │                   │                     │
  │                  │                  │                   │                 │                   │                     │
  │                  │                  │ Init(General)     │                 │                   │                     │
  │                  │                  │──────────────────>│                 │                   │                     │
  │                  │                  │                   │                 │                   │                     │
  │                  │                  │                   │ GeneralAlloc    │                   │                     │
  │                  │                  │                   │ size=4096       │                   │                     │
  │                  │                  │                   │ alignment=0     │                   │                     │
  │                  │                  │                   │ flags=0xB       │                   │                     │
  │                  │                  │                   │                 │                   │                     │
  │                  │                  │                   │ 构造 Hal::CreateInfo:             │                     │
  │                  │                  │                   │  type=Alloc                     │                     │
  │                  │                  │                   │  allocType=DeviceLocal          │                     │
  │                  │                  │                   │  heap=LargePage                 │                     │
  │                  │                  │                   │  property:                      │                     │
  │                  │                  │                   │   Physical|SharedVA             │                     │
  │                  │                  │                   │   |Virtual|DeviceMapped         │                     │
  │                  │                  │                   │   |SubAllocatable               │                     │
  │                  │                  │                   │   |DeviceVisible|HostVisible     │                     │
  │                  │                  │                   │   |HostCoherent|DeviceWriteable  │                     │
  │                  │                  │                   │   |DeviceCached                 │                     │
  │                  │                  │                   │  viewCapability:                │                     │
  │                  │                  │                   │   Exportable|PeerAccessible     │                     │
  │                  │                  │                   │   |IpcExportable                │                     │
  │                  │                  │                   │  size=4096                     │                     │
  │                  │                  │                   │  alignment=max(0,devAlign)     │                     │
  │                  │                  │                   │                 │                   │                     │
  │                  │                  │                   │ [SubAllocatable?]                 │                     │
  │                  │                  │                   │   YES                           │                     │
  │                  │                  │                   │                 │                   │                     │
  │                  │                  │                   │ MemMgr::Allocate                  │                     │
  │                  │                  │                   │─────────────────>│                 │                     │
  │                  │                  │                   │ (allocInfo)     │                 │                     │
  │                  │                  │                   │                 │                 │                     │
  │                  │                  │                   │                 │ 校验: size?     │                     │
  │                  │                  │                   │                 │ overflow?       │                     │
  │                  │                  │                   │                 │ >totalMem?      │                     │
  │                  │                  │                   │                 │                 │                     │
  │                  │                  │                   │                 │ 按 key 查找 pool │                     │
  │                  │                  │                   │                 │ m_PoolRefs.Get  │                     │
  │                  │                  │                   │                 │ (SplayTree)     │                     │
  │                  │                  │                   │                 │                 │                     │
  │                  │                  │                   │                 │ [pool 已存在?]  │                     │
  │                  │                  │                   │                 │   YES→使用已有   │                     │
  │                  │                  │                   │                 │   NO →CreatePoolNoLock           │
  │                  │                  │                   │                 │      new MemoryPool              │
  │                  │                  │                   │                 │      pool->Init(createInfo)      │
  │                  │                  │                   │                 │      m_PoolRefs.Insert(key,pool) │
  │                  │                  │                   │                 │                 │                     │
  │                  │                  │                   │                 │ FullAllocate    │                     │
  │                  │                  │                   │                 │────────────────>│                     │
  │                  │                  │                   │                 │ (allocInfo,     │                     │
  │                  │                  │                   │                 │  &offset,&mem)  │                     │
  │                  │                  │                   │                 │                 │                     │
  │                  │                  │                   │                 │                 │ SubAllocate         │
  │                  │                  │                   │                 │                 │─────────────────     │
  │                  │                  │                   │                 │                 │ 按 Log2(size)     │
  │                  │                  │                   │                 │                 │ 定位空闲桶         │
  │                  │                  │                   │                 │                 │ FindBucket(桶,    │
  │                  │                  │                   │                 │                 │  size,align,      │
  │                  │                  │                   │                 │                 │  tryLimit)         │
  │                  │                  │                   │                 │                 │                     │
  │                  │                  │                   │                 │                 │ [池为空, 首次?]    │
  │                  │                  │                   │                 │                 │  errorNotFound      │
  │                  │                  │                   │                 │                 │                     │
  │                  │                  │                   │                 │                 │ ChunkAllocate       │
  │                  │                  │                   │                 │                 │─────────────────     │
  │                  │                  │                   │                 │                 │                     │
  │                  │                  │                   │                 │                 │ chunkSize =         │
  │                  │                  │                   │                 │                 │  max(4096, 64KB)    │
  │                  │                  │                   │                 │                 │  = 64KB             │
  │                  │                  │                   │                 │                 │                     │
  │                  │                  │                   │                 │                 │ m_pDevice->         │
  │                  │                  │                   │                 │                 │ CreateMemory()      │
  │                  │                  │                   │                 │                 │────────────────────>│
  │                  │                  │                   │                 │                 │  (64KB, alignment)  │
  │                  │                  │                   │                 │                 │                     │
  │                  │                  │                   │                 │                 │                     │ InitGeneralDeviceMem
  │                  │                  │                   │                 │                 │                     │────────────────  │
  │                  │                  │                   │                 │                 │                     │ alignment         │
  │                  │                  │                   │                 │                 │                     │ =max(devAlign,    │
  │                  │                  │                   │                 │                 │                     │  pageSize)        │
  │                  │                  │                   │                 │                 │                     │                  │
  │                  │                  │                   │                 │                 │                     │ m3dDevice->       │
  │                  │                  │                   │                 │                 │                     │ GetGpuMemorySize()│
  │                  │                  │                   │                 │                 │                     │ gpuMemObjSize    │
  │                  │                  │                   │                 │                 │                     │ Reserve(objSize) │
  │                  │                  │                   │                 │                 │                     │                  │
  │                  │                  │                   │                 │                 │                     │ CreateGpuMemory() │
  │                  │                  │                   │                 │                 │                     │(info,alloc,      │
  │                  │                  │                   │                 │                 │                     │ &M3dGpuMem())     │
  │                  │                  │                   │                 │                 │                     │─────────────────>│
  │                  │                  │                   │                 │                 │                     │                  │
  │                  │                  │                   │                 │                 │                     │                  │ DRM_IOCTL
  │                  │                  │                   │                 │                 │                     │                  │ ────
  │                  │                  │                   │                 │                 │                     │                  │ 1. 分配物理页面
  │                  │                  │                   │                 │                 │                     │                  │  (64KB 显存)
  │                  │                  │                   │                 │                 │                     │                  │
  │                  │                  │                   │                 │                 │                     │                  │ 2. 建立 GPU 页表
  │                  │                  │                   │                 │                 │                     │                  │  映射:VA→物理
  │                  │                  │                   │                 │                 │                     │                  │
  │                  │                  │                   │                 │                 │                     │                  │ 3. 返回 GPU VA
  │                  │                  │                   │                 │                 │                     │<── OK ───────────│
  │                  │                  │                   │                 │                 │                     │                  │
  │                  │                  │                   │                 │                 │<── OK ─────────────│                  │
  │                  │                  │                   │                 │                 │                     │                  │
  │                  │                  │                   │                 │                 │ [创建选段]:        │                  │
  │                  │                  │                   │                 │                 │ base = GPU VA     │                  │
  │                  │                  │                   │                 │                 │ size = 64KB       │                  │
  │                  │                  │                   │                 │                 │ pChunkMem = KMD   │                  │
  │                  │                  │                   │                 │                 │   GpuMemory       │                  │
  │                  │                  │                   │                 │                 │                     │                  │
  │                  │                  │                   │                 │                 │ ResourceAdd(段)    │                  │
  │                  │                  │                   │                 │                 │─────────────────     │                  │
  │                  │                  │                   │                 │                 │ 加入段链表         │                  │
  │                  │                  │                   │                 │                 │ 加入空闲桶         │                  │
  │                  │                  │                   │                 │                 │ 更新 SegmentMap    │                  │
  │                  │                  │                   │                 │                 │ m_FreeSize += 64KB │                  │
  │                  │                  │                   │                 │                 │ m_TotalSize+=64KB │                  │
  │                  │                  │                   │                 │                 │                     │                  │
  │                  │                  │                   │                 │                 │ SubAllocate again  │                  │
  │                  │                  │                   │                 │                 │─────────────────     │                  │
  │                  │                  │                   │                 │                 │ (这次有 chunk 了) │                  │
  │                  │                  │                   │                 │                 │                     │                  │
  │                  │                  │                   │                 │                 │ FindBucket → 找到  │                  │
  │                  │                  │                   │                 │                 │ ResourceSplit      │                  │
  │                  │                  │                   │                 │                 │─────────────────     │                  │
  │                  │                  │                   │                 │                 │ Chunk: [0,64KB]   │                  │
  │                  │                  │                   │                 │                 │ 切分:              │                  │
  │                  │                  │                   │                 │                 │  [0,4096) → busy   │                  │
  │                  │                  │                   │                 │                 │  [4096,64KB)→free  │                  │
  │                  │                  │                   │                 │                 │                     │                  │
  │                  │                  │                   │                 │                 │ m_FreeSize -=4096  │                  │
  │                  │                  │                   │                 │                 │                     │                  │
  │                  │                  │                   │                 │<── offset+mem ────│                     │                  │
  │                  │                  │                   │<── offset+mem ───│                     │                  │
  │                  │                  │                   │                  │                     │                  │
  │                  │                  │                   │ Store:            │                     │                  │
  │                  │                  │                   │ m_pHalMemory=mem  │                     │                  │
  │                  │                  │                   │ m_Offset=offset   │                     │                  │
  │                  │                  │                   │                  │                     │                  │
  │                  │<── OK ───────────│                   │                  │                     │                  │
  │<── dptr(=VA+offset)│                  │                   │                  │                     │                  │
```

---

## 六、最终返回给用户的值

```cpp
// GeneralAlloc 设置:
m_Shape        = {4096, 1, 1, 4096}
m_Flags        = Virtual|DeviceMapped|SubAllocatable
m_pHalMemory   = chunkHalMemory    ← 指向 KMD 分配的 64KB GpuMemory
m_Offset       = 0                 ← 在 chunk 中的偏移

// GetDevicePointer 返回:
return m_pHalMemory->GetDeviceVirtualAddress() + m_Offset;
     = GPU_VA_of_64KB_Chunk + 0
     = 某个 GPU 虚拟地址 (如 0x70000000000)
```

```
用户看到:
  dptr = 0x70000000000

实际在 GPU 显存中:
  ┌──────────────────────────┬──────────────────────────────┐
  │ busy: 4096 bytes         │ free: 60KB+                  │
  │ (本次分配)                │ (下次 muMemAlloc 复用)        │
  └──────────────────────────┴──────────────────────────────┘
  ↑ dptr = chunkVA + 0       ↑ 下次分配从这里切
```

---

## 七、与非 SubAllocatable 路径对比

```
GeneralAlloc 属性 SubAllocatable=YES (默认):
  → MemMgr::Allocate
    → Pool::FullAllocate
      → SubAllocate (从已有 chunk 中切)
         [首次] → ChunkAllocate → KMD 申请 64KB chunk
                  → SubAllocate → 从 chunk 中切出 4096
      → return pHalMemory 指向 64KB chunk

GeneralAlloc 属性 SubAllocatable=NO:
  → Hal::IDevice::CreateMemory
    → M3d::Memory::Init(memoryTypeAlloc)
      → InitGeneralDeviceMemory
        → m3dDevice->CreateGpuMemory → KMD 申请 4096 bytes
    → return pHalMemory 指向独立的 4096 GPU 内存
```

| 对比维度 | SubAllocatable (默认) | 非 SubAllocatable |
|---------|----------------------|-------------------|
| KMD 分配粒度 | 64KB chunk | 恰好需要的 size |
| 后续分配 | 从 chunk 剩余部分切, 无 KMD 调用 | 每次都调 KMD |
| KMD 调用频率 | 低 (仅 pool 耗尽时) | 高 (每次) |
| 内存利用率 | 高 (chunk 内复用) | 低 (容易碎片) |
| 适用场景 | 频繁小分配 | 大分配或特殊用途 |
