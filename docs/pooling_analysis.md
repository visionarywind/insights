# MUSA 池化技术深度分析

> **📖 相关文档**: [memory_analysis.md](memory_analysis.md) (架构总览) | [memory_api_deep_analysis.md](memory_api_deep_analysis.md) (API流程) | [muMemAlloc_vs_muMemAllocAsync.md](muMemAlloc_vs_muMemAllocAsync.md) (对比) | [stream_command_analysis.md](stream_command_analysis.md) (Stream) | [decision_logic.md](decision_logic.md) (决策分支)

**生成时间**: 2026-05-19
**项目**: MUSA User-Mode GPU Driver
**分析范围**: 内存池 (MemoryPool)、命令池 (CmdPool)、命令分配器 (CmdAllocator)、分配器类型

---

## 目录

1. [总览：MUSA 中的池化技术全景](#1-总览musa-中的池化技术全景)
2. [内存池 (MemoryPool) 三层架构](#2-内存池-memorypool-三层架构)
3. [M3D MemoryPool：Bucket 自由链表分配器](#3-m3d-memorypoolbucket-自由链表分配器)
4. [命令池 (CmdPool)](#4-命令池-cmdpool)
5. [命令分配器 (CmdAllocator)](#5-命令分配器-cmdallocator)
6. [MemMgr：池管理器](#6-memmgr池管理器)
7. [M3D 层其他分配器](#7-m3d-层其他分配器)
8. [总结与设计洞察](#8-总结与设计洞察)

---

## 1. 总览：MUSA 中的池化技术全景

MUSA 项目实现了多层级的池化/分配器体系：

```
┌────────────────────────────────────────────────────────────────────┐
│                        应用层 API                                   │
│     muMemPoolCreate / muMemPoolDestroy / muMemPool...              │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────────┐
│                      DRIVER 层 (验证/转发)                          │
│     src/driver/mu_mempool.cpp                                      │
│     └─ 参数校验 → Hal::MemoryPoolCreateInfo 构建 → 委托 Core        │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────────┐
│                      CORE 层 (对象管理)                              │
│     src/musa/core/memoryPool.cpp/h                                 │
│     └─ MemoryPool 对象生命周期、属性管理、IPC 共享内存               │
│     └─ 委托 Hal::IMemoryPool (Hal 层) 进行实际内存操作               │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────────┐
│                      HAL 层 (抽象接口)                               │
│     src/hal/halMemoryPool.h → IMemoryPool                           │
│     └─ SubAllocate / ChunkAllocate / FullAllocate / Free            │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────────┐
│                 M3D HAL 层 (Bucket 自由链表实现)                     │
│     src/hal/m3d/memoryPool.cpp/h                                   │
│     └─ 64 个 bucket (0..63) 的自由链表                              │
│     └─ ResSegment 双向链表 + SegmentTracker (区间树)                 │
│     └─ lazy-free 策略 + TrimPool                                    │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────────┐
│                      M3D 硬件层                                     │
│     └─ GPU 内存分配 (DRM IOCTL / WDDM)                              │
└────────────────────────────────────────────────────────────────────┘
```

### 池化技术清单

| 池化组件 | 层次 | 文件 | 核心算法 |
|---------|------|------|---------|
| **MemoryPool** | Driver/Core/HAL | `mu_mempool.cpp` → `memoryPool.cpp` → `hal/m3d/memoryPool.cpp` | Bucket 自由链表 + 区间树 |
| **CmdPool** | HAL | `hal/m3d/cmdPool.cpp` | M3D CmdAllocator + ExternalAllocator 重用池 |
| **CmdAllocator** | M3D | `m3d/inc/core/m3dCmdAllocator.h` | 多类型子分配 (CommandData/EmbeddedData/GpuScratchMem 等) |
| **MemMgr** | HAL | `hal/m3d/memMgr.cpp` | SplayTree + Internal Pool Lists |
| **InternalMemMgr** | M3D | `m3d/src/core/internalMemMgr.h` | BuddyAllocator + 区间树 + 延迟释放哈希表 |
| **DMA Upload Ring** | M3D | `m3d/src/core/dmaUploadRing.h` | 环形 Entry 槽位 + GPU Fence 同步 + 动态扩容 |
| **HW IP ctxMemFreeList** | M3D HW | 9个引擎各自维护 | IntrusiveList PushBack/PopFront (零分配开销) |
| **ScratchMemMgr** | M3D HW | `m3d/src/core/hw/scratchMemMgr.h` | 5-stage 虚拟+物理暂存，动态缩放 |
| **PcSamplerMemMgr** | M3D HW | `m3d/src/core/hw/pcSamplerMemMgr.h` | 预分配 4GB VA，64 个计数器槽位 |
| **VirtualLinearAllocator** | M3D Util | `m3d/inc/util/m3dLinearAllocator.h` | 线性分配 + 回退 (Rewind) |
| **BuddyAllocator** | M3D Util | `m3d/inc/util/m3dBuddyAllocator.h` | 伙伴算法 |
| **BestFitAllocator** | M3D Util | `m3d/inc/util/m3dBestFitAllocator.h` | 最佳适配 |
| **BitMapAllocator** | M3D Util | `m3d/inc/util/m3dBitMapAllocator.h` | 位图分配 |
| **GpuMemoryAllocator** | M3D Util | `m3d/inc/util/m3dGpuMemoryAllocator.h` | GPU 内存分配器基类 |
| **RingBuffer** | M3D Util | `m3d/inc/util/m3dRingBuffer.h` | 信号量保护的环形槽位 |

---

## 2. 内存池 (MemoryPool) 三层架构

### 2.1 Driver 层：API 入口 (`mu_mempool.cpp`)

```
muapiMemPoolCreate(pool, poolProps)
  │
  ├─ InitPlatform()                       // ① 守卫：检查平台是否已初始化
  │
  ├─ 参数校验                              // ② pool != nullptr, poolProps != nullptr
  │
  ├─ 获取 Context 和 Device               // ③ TlsCtxTop() → GetParentDevice()
  │
  ├─ 构建 Hal::MemoryPoolCreateInfo        // ④ 根据 poolProps->location.type 构建
  │    ├─ MU_MEM_LOCATION_TYPE_DEVICE     → memoryAllocTypeDeviceLocal
  │    └─ MU_MEM_LOCATION_TYPE_HOST       → memoryAllocTypeHost
  │
  ├─ 创建 Musa::MemoryPool 对象            // ⑤ new Musa::MemoryPool(pDevice)
  │
  ├─ pMemoryPool->Init(createInfo)         // ⑥ 初始化 → 委托 HAL 创建底层池
  │    └─ m_pDevice->Hal().GetMemMgr()->CreateUserPool(...)
  │
  └─ *pool = reinterpret_cast<MUmemoryPool>(pMemoryPool)
```

**关键参数**:
```cpp
// 每个 MemoryPool 的默认配置
createInfo.minEnlargeChunkSize = 32 MB   // 每次扩容的 Chunk 大小
createInfo.reuseCountLimit = UINT64_MAX  // 复用次数无限制
createInfo.reuseSizeLimit = UINT64_MAX   // 复用大小无限制
createInfo.size = 0                       // 懒分配：初始不分配物理内存
createInfo.alignment = 0                  // 对齐由 HAL 层决定
createInfo.info.property = memoryPropertyVirtual          // 虚拟内存
                         | memoryPropertyHostVisible       // Host 可见
                         | memoryPropertyDeviceVisible     // Device 可见
                         | memoryPropertyDeviceWriteable   // Device 可写
                         | memoryPropertyDeviceCached      // Device 缓存
                         | memoryPropertyHostCoherent      // Host 一致性
                         | memoryPropertySharedVirtualAddress // 共享 VA
                         | memoryPropertySubAllocatable;   // 可子分配
```

### 2.2 Core 层：对象管理 (`memoryPool.cpp/h`)

```cpp
class MemoryPool {
    Hal::IMemoryPool* m_pHalPool;          // 底层 HAL 池指针
    Device*            m_pDevice;           // 所属设备
    Stream*            m_pStream;           // 操作流

    // 复用属性 (CUDA-compatible)
    bool     m_ReuseAllowance;
    uint64_t m_MinBytesToKeep;
    bool     m_DisableReuseViaEventDependencies;
    bool     m_DisableReuseOpportunistic;
    bool     m_DisableReuseViaFalseDependencies;

    // 位置访问控制
    std::map<int, MUmemAccess_flags> m_LocationAccessMap;
    std::unordered_set<Memory*>       m_MemoryAllocations;

    // IPC 相关
    IpcMemPoolData_t m_IpcMemPoolData;     // 共享内存用于跨进程池共享
};
```

**三种初始化类型**:

```
MemoryPoolInitType::GeneralAlloc    → 调用 Hal::GetMemMgr()->CreateUserPool()
MemoryPoolInitType::ExternalAlloc   → 从外部句柄导入 (IPC)
MemoryPoolInitType::InternalAlloc   → 从 MemMgr 获取内部池 (用于内部内存)
```

**内存分配流程** (`CreateMemory`):

```
MemoryPool::CreateMemory(ppMemory, ptr, size)
  │
  ├─ 创建 Memory 对象                    // std::make_shared<Memory>(nullptr)
  │
  ├─ pMemory->InitFromPool(this, size)  // Memory 从 Pool 初始化
  │    └─ 调用 Hal::IMemoryPool::FullAllocate()
  │
  ├─ 注册到 MemoryTracker               // Platform::Get().GetMemoryTracker().TrackMemory()
  │
  ├─ *ptr = pMemory->GetDevicePointer() // 返回 GPU 设备指针
  │
  └─ 记录到 m_MemoryAllocations         // 用于后续 SetAccess 批量操作
```

**IPC 跨进程池共享**:

```
进程A: muMemPoolExportToShareableHandle()
  ├─ CreateIpcMemPoolShmemIfNeed()
  │    ├─ mkstemp("/tmp/mempoolXXXXXX")  → 生成临时文件名
  │    ├─ 重命名: "MUSA_" + random
  │    ├─ shm_open()                     → POSIX 共享内存
  │    ├─ ftruncate(sizeof(IpcMemPoolShmem_t))
  │    ├─ mmap(MAP_SHARED)
  │    └─ owners = 1                     → 初始化引用计数
  └─ dup(m_IpcFd) → 返回文件描述符

进程B: muMemPoolImportFromShareableHandle()
  ├─ dup(fd)                             → 复制文件描述符
  ├─ mmap(MAP_SHARED)                    → 映射共享内存
  ├─ owners += 1                         → 增加引用计数
  └─ m_IsImported = true
```

---

## 3. M3D MemoryPool：Bucket 自由链表分配器

这是 MUSA 项目中**最核心的池化算法实现**，位于 `src/hal/m3d/memoryPool.cpp`。

### 3.1 数据结构

```
┌───────────────────────────────────────────────────────────────┐
│                      MemoryPool                                │
├───────────────────────────────────────────────────────────────┤
│  m_FreeBuckets[0..63]   ← 64 个 bucket 的自由链表头           │
│  m_EltMappingHash       ← 64 位 bitmap，标记非空 bucket       │
│  m_pHeadSegment         ← ResSegment 双向链表头                │
│  m_SegmentTracker       ← std::map<MemoryRange, ResSegment*>   │
│                           (区间树，用于 Free 时快速查找)        │
│  m_FreeSize / m_TotalSize                                     │
│  m_Policy               ← {insertion, selection, noSplit}      │
└───────────────────────────────────────────────────────────────┘

ResSegment (内存资源段):
┌────────────────────────────────────────────────┐
│  pChunkMem      → IMemory* (物理 Chunk)        │
│  base            → DevSize (段起始地址)          │
│  size            → DevSize (段大小)              │
│  chunkBase       → DevSize (Chunk 基地址)        │
│  busy            → bool (是否已分配)             │
│  isLeftMost      → bool (Chunk 最左段)           │
│  isRightMost     → bool (Chunk 最右段)           │
│  lazyFreeCount   → DevSize (lazy-free 计数)      │
│                                                  │
│  pNextSegment    → ResSegment* (segment 链表)    │
│  pPrevSegment    → ResSegment*                   │
│                                                  │
│  pNextFree       → ResSegment* (free 链表)       │
│  pPrevFree       → ResSegment*                   │
└────────────────────────────────────────────────┘
```

**ResSegment 维护两个维度的链表**:
- **Segment 链表** (pNextSegment/pPrevSegment): 所有段按地址序排列的双向链表
- **Free 链表** (pNextFree/pPrevFree): 仅空闲段，挂在对应 bucket 下

### 3.2 Bucket 组织原理

每个 bucket 的索引 = `log2(segment->size)`:

```
Bucket[0]  → 空闲段 1B..2B
Bucket[1]  → 空闲段 2B..4B
Bucket[2]  → 空闲段 4B..8B
...
Bucket[20] → 空闲段 1MB..2MB
...
Bucket[63] → 空闲段 2^63..2^64
```

`m_EltMappingHash` 是一个 64 位 bitmap，bit[i]=1 表示 Bucket[i] 非空：

```cpp
// 插入时设置 bit
m_EltMappingHash |= (1ULL << index);

// 移除时清除 bit
if (!m_FreeBuckets[index]) {
    m_EltMappingHash &= ~(1ULL << index);
}
```

### 3.3 SubAllocate：子分配流程（核心算法）

```
SubAllocate(allocInfo, pOffset, ppMemory)
  │
  ├─ ① 计算 bucket 索引范围
  │    indexLow  = Log2(allocInfo.size)           // 最小满足 bucket
  │    indexHigh = Log2(size + alignment - 1)      // 考虑对齐的最大 bucket
  │
  ├─ ② 查找合适的空闲段
  │    │
  │    ├─ assuredFit 策略 (默认):
  │    │   optionalMappingHash = ~((1 << (indexHigh+1)) - 1) & m_EltMappingHash
  │    │   // 找到 >= indexHigh+1 的第一个非空 bucket
  │    │   if (BitMaskScanForward(&index, optionalMappingHash)) {
  │    │       if (index != s_FreeTableLimit) {
  │    │           // 从该 bucket 找一个满足 size+alignment 的段 (tryLimit=1)
  │    │           pMemRes = FindBucket(m_FreeBuckets[index], size, alignment, 1);
  │    │       } else {
  │    │           // 回退：从 indexHigh 向下查找 (tryLimit=UINT64_MAX)
  │    │           for (index = indexHigh; index >= indexLow; index--) {
  │    │               pMemRes = FindBucket(...);
  │    │           }
  │    │       }
  │    │   }
  │    │
  │    └─ bestFit 策略:
  │        // 从小到大遍历所有满足的 bucket，找最小满足的段
  │        ...
  │
  ├─ ③ ResourceSplit(pMemRes, size, alignment, &subAllocAddr)
  │    │
  │    ├─ 对齐检查: alignedBase = AlignUp(pMemRes->base, alignment)
  │    │
  │    ├─ 前部分裂 (如果 alignedBase > base):
  │    │   创建新段: [base, alignedBase) → 插入 FreeList
  │    │   当前段缩小为: [alignedBase, base+size)
  │    │
  │    ├─ 后部分裂 (如果 pMemRes->size > size):
  │    │   创建新段: [base+size, base+oldSize) → 插入 FreeList
  │    │   当前段缩小为: [alignedBase, alignedBase+size)
  │    │
  │    └─ 将分配段加入 SegmentTracker (区间树)
  │
  ├─ ④ 更新统计
  │    *ppMemory = pMemRes->pChunkMem
  │    *pOffset  = subAllocAddr - pMemRes->chunkBase
  │    m_FreeSize -= allocInfo.size
  │
  └─ ⑤ 返回 MUSA_SUCCESS
```

### 3.4 分裂图解

假设一个 16MB Chunk，请求分配 3MB (alignment=1MB):

```
分配前:
┌──────────────────────────────────────────────────────┐
│              16MB Chunk (base=0x10000000)              │
│           [全部空闲, 挂在 Bucket[24] 下]                 │
└──────────────────────────────────────────────────────┘

1. 对齐: alignedBase = AlignUp(0x10000000, 1MB) = 0x10000000

2. 前部分裂: alignedBase == base → 不需要前部

3. 后部分裂: base+size = 0x10000000+3MB = 0x10300000
   剩余: 16MB - 3MB = 13MB

分裂后:
┌──────────────┬──────────────────────────────────────┐
│   3MB (busy) │          13MB (free)                  │
│  0x10000000  │  0x10300000                          │
│  已分配✓     │  挂入 Bucket[24] (8MB~16MB)          │
└──────────────┴──────────────────────────────────────┘
```

### 3.5 Free：释放与合并

```
Free(pMemory, base, size)
  │
  ├─ ① 从 SegmentTracker (区间树) 查找段
  │    auto it = m_SegmentTracker.find(MemoryRange{base, size})
  │    pMemRes = it->second
  │
  ├─ ② 尝试与左邻合并
  │    if (!pMemRes->isLeftMost && !pPrevSegment->busy) {
  │        FreeListRemove(pPrevSegment)
  │        SegmentListRemove(pPrevSegment)
  │        pMemRes->base  = pPrevSegment->base
  │        pMemRes->size += pPrevSegment->size
  │        isLeftMost = pPrevSegment->isLeftMost
  │        delete pPrevSegment
  │    }
  │
  ├─ ③ 尝试与右邻合并
  │    if (!pMemRes->isRightMost && !pNextSegment->busy) {
  │        FreeListRemove(pNextSegment)
  │        SegmentListRemove(pNextSegment)
  │        pMemRes->size += pNextSegment->size
  │        isRightMost = pNextSegment->isRightMost
  │        delete pNextSegment
  │    }
  │
  ├─ ④ ResourceRemove: Lazy-free 检查
  │    if (isLeftMost && isRightMost) {  // 整个 Chunk 都空闲了
  │        lazyFreeCount++
  │        if (lazyFreeCount > m_ReuseCountLimit ||
  │            chunkSize > m_ReuseSizeLimit) {
  │            // 销毁 Chunk，归还 GPU 内存
  │            pChunkMem->Destroy()
  │            delete pMemRes
  │            return
  │        }
  │    }
  │
  └─ ⑤ 将段重新插入 FreeList
       FreeListInsert(pMemRes)
```

**合并图解**:

```
释放前:
┌────────┬────────┬────────┬──────────┐
│ 2MB    │ 3MB    │ 1MB    │ 10MB     │
│ free   │ busy   │ free   │ busy     │
└────────┴────────┴────────┴──────────┘

释放 3MB 段后 (左右合并):
┌─────────────────────┬──────────┐
│       6MB (free)    │ 10MB     │
│   2MB + 3MB + 1MB   │ busy     │
└─────────────────────┴──────────┘
→ 从 Bucket[21] (1MB-2MB) 和 Bucket[22] (2MB-4MB) 移除
→ 插入 Bucket[23] (4MB-8MB)
```

### 3.6 ChunkAllocate：创建新的物理 Chunk

```
ChunkAllocate(allocInfo)
  │
  ├─ ① 计算 Chunk 大小
  │    chunkSize = allocInfo.size
  │    if (alignment > chunkAlignment) chunkSize += alignment - chunkAlignment
  │    chunkSize = AlignUp(chunkSize, chunkAlignment)    // 对齐到页面大小
  │    chunkSize = AlignUp(chunkSize, m_ChunkAllocSize)  // 对齐到 Chunk 大小 (默认 2MB)
  │
  ├─ ② 分配物理 GPU 内存
  │    虚拟内存: m_pDevice->GetPlatform().CreateMemory(..., &pChunkMem)
  │    物理内存: m_pDevice->CreateMemory(..., &pChunkMem)
  │
  ├─ ③ 创建 ResSegment
  │    base = virtual ? pChunkMem->GetDeviceVirtualAddress() : chunkAlignment
  │    size = pChunkMem->GetSize()
  │    new ResSegment(base, size) { pChunkMem, chunkBase=base, ... }
  │
  └─ ④ ResourceAdd → 插入 Segment 链表和 FreeList
       m_TotalSize += size
       m_FreeSize  += size
```

### 3.7 TrimPool：内存回收

```
TrimPool(value)
  │
  └─ while (m_TotalSize > value):
       for each Bucket[index]:
           for each segment in Bucket[index]:
               if (segment->isLeftMost && segment->isRightMost):
                   // 整个 Chunk 完全空闲 → 销毁
                   FreeListRemove(segment)
                   SegmentListRemove(segment)
                   m_TotalSize -= segment->size
                   m_FreeSize  -= segment->size
                   pChunkMem->Destroy()
                   delete segment
                   released = true
                   break
           if (!released) break  // 没有完整空闲的 Chunk 了
```

---

## 4. 命令池 (CmdPool)

`src/hal/m3d/cmdPool.cpp` — GPU 命令缓冲区的池化管理。

### 4.1 架构

```
┌──────────────────────────────────────────────────────────────┐
│                         CmdPool                               │
├──────────────────────────────────────────────────────────────┤
│  m_M3dCmdAllocator      → M3D::ICmdAllocator*                │
│                             (M3D 层命令内存子分配器)           │
│                                                              │
│  m_ExternalAllocators   → deque<unique_ptr<VirtualLinearAllocator>> │
│                             (外部线性分配器重用池, poolSize=4) │
│                                                              │
│  m_QueueFamilyIndex     → 队列族索引                          │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 初始化流程

```
CmdPool::Init(createInfo)
  │
  ├─ ① 根据 QueueFamily 确定 EngineType:
  │    "CDM"       → EngineTypeCompute
  │    "TDM"       → EngineTypeTransfer
  │    "CE"        → EngineTypeDma
  │    "ACE"       → EngineTypeAce
  │    "DMA"       → EngineTypeHdma
  │    "UNIVERSAL" → EngineTypeUniversal
  │    "MMU"       → EngineTypeCompute
  │
  ├─ ② 从 DeviceProperties 获取 preferredCmdAllocInfo
  │    每个 EngineType 预设了每种 CmdAllocType 的 allocHeap/allocSize/suballocSize
  │
  ├─ ③ 构建 CmdAllocatorCreateInfo
  │    for each CmdAllocType:
  │        allocInfo[i].allocHeap    = preferredCmdAllocInfo[i].allocHeap
  │        allocInfo[i].suballocSize = preferredCmdAllocInfo[i].suballocSize
  │        allocInfo[i].allocSize    = max(createInfo.allocSize, preferred.allocSize)
  │
  ├─ ④ 创建 M3D::ICmdAllocator
  │    cmdAllocatorObjSize = m3dDevice->GetCmdAllocatorSize(...)
  │    m_M3dCmdAllocator.Reserve(cmdAllocatorObjSize)
  │    m3dDevice->CreateCmdAllocator(..., &m_M3dCmdAllocator())
  │
  └─ ⑤ 预分配 ExternalAllocator 池 (4个)
       for (i = 0; i < 4 && res == success; i++)
           m_ExternalAllocators.push_back(new VirtualLinearAllocator(4096))
```

### 4.3 ExternalAllocator 重用池

```
GetExternalAllocator():
  │
  ├─ lock(m_ExternalAllocatorMtx)
  ├─ if (!m_ExternalAllocators.empty()):
  │     allocator = std::move(m_ExternalAllocators.front())
  │     m_ExternalAllocators.pop_front()
  │     return allocator
  │
  └─ else:
        // 池空了，创建新的
        allocator = new VirtualLinearAllocator(4096)
        allocator->Init()
        return allocator

RecycleExternalAllocator(allocator):
  │
  ├─ lock(m_ExternalAllocatorMtx)
  ├─ if (m_ExternalAllocators.size() < 4):
  │     allocator->Rewind(allocator->Start(), false)  // 重置指针，不释放内存
  │     m_ExternalAllocators.push_back(std::move(allocator))
  │
  └─ else:
        // 池满了，allocator 超出作用域即销毁
```

**设计要点**:
- 池大小固定为 4，避免无限膨胀
- `Rewind(Start(), false)` 只重置写指针，**不释放底层 GPU 内存**
- 这是一个 "warm pool" 模式 — 回收的分配器保留已分配的内存，下次使用直接重用

---

## 5. 命令分配器 (CmdAllocator)

M3D 层的 `ICmdAllocator` 用于为 GPU 命令缓冲区分配和子分配内存。

### 5.1 支持的分配类型

```
enum CmdAllocType {
    CommandDataAlloc    = 0,  // 可执行命令
    EmbeddedDataAlloc   = 1,  // 嵌入数据
    GpuScratchMemAlloc  = 2,  // GPU-only 暂存内存
    CpuGpuSharedMemAlloc= 3,  // CPU/GPU 共享内存
    ShaderPgmAlloc      = 5,  // USC 着色器程序
    ShaderPgmConstAlloc = 6,  // 着色器常量
    ShaderImgConstAlloc = 7,  // 着色器图像常量
    ComponentControlAlloc=8,  // 组件控制流
};
```

### 5.2 内存模型

```
CmdAllocatorCreateInfo {
    flags: {
        threadSafe              // 线程安全
        autoMemoryReuse         // 自动追踪 GPU 完成并重用
        disableBusyChunkTracking // 禁用 Busy Chunk 追踪
        cmdStreamReadOnly       // 命令流只读
    }
    allocInfo[CmdAllocatorTypeCount]: {
        allocHeap               // GPU 堆类型
        allocSize               // 每次分配的 GPU 内存大小
        suballocSize            // 给 CmdBuffer 的子分配大小
    }
}
```

**关键特性**:
- **Sub-allocation**: 分配大块 GPU 内存，然后子分配给各个 CommandBuffer
- **autoMemoryReuse**: 启用后，追踪 GPU 执行进度，自动回收 GPU 已完成的内存
- **Reset**: 将命令分配器重置，所有内部 GPU 内存标记为未使用（客户端需保证所有关联 CmdBuffer 已完成执行）

---

## 6. MemMgr：池管理器

`src/hal/m3d/memMgr.cpp/h` — 所有内存池的中央管理器。

### 6.1 池管理结构

```cpp
class MemMgr {
    Device* m_pDevice;

    // 内部内存池 (由系统管理)
    std::array<MemPoolList, InternalPoolType::Count> m_InternalPoolRefs;

    // 外部内存池 (由 SplayTree 索引)
    MemPoolTree m_PoolRefs;        // SplayTree<Key, MemoryPool*>

    // 用户内存池
    MemPoolList m_UserPools;

    std::mutex m_Lock;
};
```

### 6.2 Pool Key 设计

每个池通过一个 **64 位 Key** 唯一标识，编码全部属性:

```
Key 编码 (64 bits):
┌──────────┬──────────┬─────┬─────┬──────┐
│  property│viewCap   │type │heap │numa  │
│  (bits)  │(6 bits)  │(1b) │(3b) │(4b)  │
└──────────┴──────────┴─────┴─────┴──────┘

static inline Key MakeKey(const MemoryPoolInfo& poolInfo) {
    return (property & s_PropertyMask) |
           ((viewCapability & s_CapabilityMask) << s_PropertyShift) |
           ((type & s_TypeMask) << (s_PropertyShift + s_CapabilityShift)) |
           ((heap & s_HeapMask) << (...)) |
           ((numaId & s_NumaMask) << (...));
}
```

### 6.3 内部池类型

```cpp
static constexpr std::array<MemoryPoolInfo, 5> s_InternalInfos = {
    // [0] Default: 设备本地虚拟内存
    InfoTemplate<>,

    // [1] HostMapped: 设备本地 + HostMapped
    InfoTemplate<false, memoryPropertyHostMapped>,

    // [2] Host: Host 端分配
    InfoTemplate<true>,

    // [3] General Heap: general heap
    InfoTemplate<false, memoryPropertyHostMapped, MemoryHeap::general>,

    // [4] USC Heap: USC shader 内存
    InfoTemplate<false, 0, MemoryHeap::usc>,
};
```

### 6.4 分配决策流程

```
MemMgr::Allocate(allocInfo, pOffset, ppMemory, ppMemoryPool)
  │
  ├─ ① 校验: size=0? overflow? out of global mem?
  │
  ├─ ② 如果 ppMemoryPool 已指定:
  │      pPool = *ppMemoryPool
  │      pPool->FullAllocate(allocInfo, pOffset, ppMemory)
  │      ↓ (如果失败则继续)
  │
  ├─ ③ 在 Pool Tree 中查找匹配的池
  │      key = MakeKey(allocInfo)
  │      pPool = m_PoolRefs.Find(key)
  │      if (pPool) pPool->FullAllocate(...)
  │      ↓ (如果没有匹配的池则继续)
  │
  ├─ ④ 创建新池
  │      CreatePoolNoLock(poolCreateInfo, &pPool)
  │      m_PoolRefs.Insert(key, pPool)  // 注册到 SplayTree
  │      pPool->FullAllocate(...)
  │      ↓ (如果创建失败)
  │
  └─ ⑤ 尝试内部池 (fallback)
         for each internal pool type:
             pPool = GetInternalPool(...)
             pPool->FullAllocate(...)
```

### 6.5 SplayTree 查找优化

```
为什么用 SplayTree？
  ├─ 频繁访问的 Key 会被旋转到根附近 → O(log n) 摊还
  ├─ 内存池会被反复访问 (每次分配都查)
  └─ 比 std::map 有更好的局部性
```

---

## 7. M3D 层其他分配器

### 7.1 VirtualLinearAllocator — 线性分配器

```
┌───────────────────────────────────────────────────────┐
│              VirtualLinearAllocator                     │
├───────────────────────────────────────────────────────┤
│  特点: 只前进，不释放单个分配                            │
│                                                       │
│  Alloc(size):                                         │
│    ptr = m_CurrentPtr                                  │
│    m_CurrentPtr += AlignUp(size, alignment)            │
│    return ptr                                          │
│                                                       │
│  Rewind(ptr, releaseMem):                              │
│    m_CurrentPtr = ptr                                  │
│    if (releaseMem) 释放底层内存                          │
│                                                       │
│  适用场景:                                              │
│    - CmdPool 的外部分配器 (一次录制，批量重置)             │
│    - 临时数据 (生命周期明确)                              │
└───────────────────────────────────────────────────────┘
```

### 7.2 BuddyAllocator — 伙伴分配器

```
基于 2 的幂次分裂的二分伙伴算法:

Alloc(size):
  └─ 找到 >=size 的最小 2^k
  └─ 如果该大小的空闲块存在 → 分配
  └─ 否则: 递归分裂更大的块

Free(block):
  └─ 检查伙伴是否也空闲 → 合并
  └─ 递归向上合并

优点: 天然对齐、低碎片化
缺点: 内部碎片 (向上取整到 2^k)
```

### 7.3 BestFitAllocator — 最佳适配

```
从空闲链表中找满足请求的最小块:

while (walker) {
    if (walker->size >= requested_size) {
        if (walker->size < best_so_far) {
            best_so_far = walker;
        }
    }
    walker = walker->next;
}
return best_so_far;

优点: 最小化浪费
缺点: 搜索开销大、长期运行后碎片化
```

### 7.4 BitMapAllocator — 位图分配器

```
将内存按最小分配单元 (例如 4KB) 划分为固定大小的块:

bitmap: [1 1 0 0 1 0 1 1 ...]  (1=已分配, 0=空闲)

Alloc(n_blocks):
  └─ 在位图中找到连续 n 个 0
  └─ 将它们置为 1

Free(start_block, n_blocks):
  └─ 将对应位清 0

优点: 分配/释放 O(1)、无外部碎片
缺点: 只支持固定大小块
```

### 7.5 InternalMemMgr — M3D 内部内存管理器

`src/hal/m3d/m3d/src/core/internalMemMgr.h` — M3D 层内部小型 GPU 分配的管理。

```
┌───────────────────────────────────────────────────────────────┐
│                    InternalMemMgr                              │
├───────────────────────────────────────────────────────────────┤
│  m_poolList         → list<GpuMemoryPool*>                     │
│                       每个 GpuMemoryPool 包含:                  │
│                         - GpuMemory* (GPU 内存对象)             │
│                         - BuddyAllocator (伙伴分配器)           │
│                         - heap preferences + VA range          │
│                                                               │
│  m_references       → IntervalTree (区间树)                    │
│                       追踪所有已分配的范围                       │
│                                                               │
│  m_deferFreeMap     → HashMap<GpuMemProperties, GpuMemoryList> │
│                       32 个桶的延迟释放哈希表                    │
│                       释放的 GPU 内存不立即归还 OS，             │
│                       而是放入 defer-free map 等待复用           │
│                                                               │
│  线程安全:                                                      │
│    m_allocatorLock   → Util::Mutex  (分配/释放)                 │
│    m_referenceLock   → Util::RWLock (区间树读写锁)              │
│    m_deferFreeLock   → Util::RWLock (延迟释放读写锁)            │
└───────────────────────────────────────────────────────────────┘
```

**延迟释放流程**:

```
Free(offset, size)
  │
  ├─ 1. 从 m_references 区间树中查找并删除记录
  │
  ├─ 2. 调用 GpuMemoryPool->buddyAllocator.Free(offset)
  │      (归还给伙伴分配器)
  │
  └─ 3. 如果整个 GpuMemoryPool 完全空闲:
         GpuMemoryPool 移入 m_deferFreeMap["deferred"]
         (不立即销毁，等待未来分配时复用)
```

---

## 8. DMA Upload Ring — CPU→GPU 环形命令池

`src/hal/m3d/m3d/src/core/dmaUploadRing.h` — 用于 CPU 向 GPU 上传数据的环形缓冲区。

### 8.1 数据结构

```
┌───────────────────────────────────────────────────────────────┐
│                     DmaUploadRing                              │
├───────────────────────────────────────────────────────────────┤
│  m_entries[]         → Entry[N] (环形数组，初始 512)            │
│                        每个 Entry = CmdBuffer* + Fence*        │
│                                                               │
│  m_curEntry          → 当前写入位置 (生产者指针)                  │
│  m_entriesSubmitted  → 已提交但未完成的 entry 计数              │
│  m_gpuFence          → GPU 完成的 fence                       │
│                                                               │
│  容量动态扩展:                                                   │
│    当 ring 满时 → AllocNewRing() → double 容量                 │
│    → 分配更大的 GpuMemory → 重建 Entry 数组                     │
└───────────────────────────────────────────────────────────────┘
```

### 8.2 生命周期

```
AcquireRingSlot(dataSize)       // ① 获取一个 slot
  ├─ 如果 ring 满 → 扩展容量 (double)
  ├─ 将数据写入 Entry 的 embedded data 区域 (cmd buffer)
  └─ 返回 Entry*

Submit(entry)                   // ② 提交到 GPU
  ├─ 将 cmd buffer 提交到 DMA engine queue
  ├─ 关联 fence 用于追踪完成
  └─ m_entriesSubmitted++

FreeFinishedSlots()             // ③ 回收已完成的 slot
  ├─ 检查每个已提交 entry 的 fence 是否 signaled
  ├─ 如果完成 → 重置 entry (可被下一个 AcquireRingSlot 复用)
  └─ m_entriesSubmitted--
```

### 8.3 设计要点

- **环形结构**：Entry 数组循环使用，不需要每次分配/释放
- **动态扩容**：当所有 slot 都在使用中时，自动 double 容量
- **GPU Fence 同步**：通过 fence 确认 GPU 已完成 DMA，确保 slot 可安全重用
- **嵌入数据**：Entry 中的 CmdBuffer 预分配好 embedded data 空间，减少内存分配

---

## 9. HW IP Context Memory Free Lists — 硬件引擎上下文内存池

每个 GPU 硬件引擎 (Compute, GFX, DMA, Xfer 等) 都维护一个**侵入式自由链表**来回收提交上下文内存。

### 9.1 引擎清单 (9 个引擎)

| 引擎 | 类型 | 自由链表变量 | 文件 |
|------|------|-------------|------|
| Compute | 计算 | `m_ctxMemFreeList` | `computeip/computeDevice.h:431` |
| GFX | 图形 | `m_ctxMemFreeList` | `gfxip/gfxDevice.cpp:338` |
| Transfer | 传输 | `m_ctxMemFreeList` | `xferip/xferDevice.cpp:209` |
| DMA | DMA | `m_ctxMemFreeList` | `dmaip/dmaDevice.cpp:219` |
| ACE | ACE | `m_ctxMemFreeList` | `aceip/aceDevice.cpp:201` |
| TCE | TCE | `m_ctxMemFreeList` | `tceip/tceDevice.cpp:200` |
| HDMA | HDMA | `m_ctxMemFreeList` | `hdmaip/hdmaDevice.cpp:196` |
| Codec | 编解码 | `m_ctxMemFreeList` | `codecip/codecDevice.cpp:195` |
| Universal | 通用 | `m_ctxMemFreeList` | `universalip/universalDevice.cpp:265` |

### 9.2 统一模式

```
提交上下文生命周期:

1. 分配:
   ├─ if (!m_ctxMemFreeList.empty()):
   │     ctxMem = m_ctxMemFreeList.PopFront()  // 复用
   │     ctxMem->Reset()
   └─ else:
         ctxMem = new SubmissionContextMem()    // 新建

2. 使用: GPU 提交期间保持引用

3. 释放:
   └─ m_ctxMemFreeList.PushBack(ctxMem)       // 归还
       (不 delete，等待下次 PopFront 复用)
```

**设计优势**:
- 所有 9 个引擎使用**完全相同的模式** (IntrusiveList PushBack/PopFront)
- 零动态内存分配开销 (热路径上)
- 引擎间完全隔离 (无跨引擎竞争)

---

## 10. ScratchMemMgr — GPU 暂存内存管理

`src/hal/m3d/m3d/src/core/hw/scratchMemMgr.h` — 管理 GPU shader 溢出到显存的暂存空间。

```
ScratchMemMgr
├─ 5 个管线阶段 (per-stage):
│   VDM (顶点), DDM (域), PDM (图元), GfxCompute, CDM (计算)
│
├─ 每阶段:
│   virtualMemory[stage]  → 虚拟内存 (单个大块)
│   physicalMemory[stage][16] → 物理内存 (最多 16 个槽位)
│
├─ 动态缩放:
│   RequireAndRemapScratchMemory()
│   ├─ 检查当前 scratch 是否足够
│   └─ 如果不足 → 分配更大的 physical memory → 重新映射
│
└─ 配额:
    最大总分配: 1.5 GB
    最大单任务溢出: 8 MB
```

---

## 11. PcSamplerMemMgr — 性能采样内存管理

`src/hal/m3d/m3d/src/core/hw/pcSamplerMemMgr.h` — 为性能计数器采样预分配 GPU VA 空间。

```
PcSamplerMemMgr
├─ 预分配 4GB 虚拟地址空间
├─ m_counterGpuMemory[64]  → 最多 64 个计数器缓冲区槽位
└─ 每个槽位: 4MB 对齐的 GpuMemory 块
```

---

## 12. 总结与设计洞察

### 12.1 架构模式

| 模式 | 实现 | 目的 |
|-----|------|------|
| **分层委托** | Driver → Core → HAL → M3D | 职责分离：API 验证 → 对象管理 → 算法实现 |
| **两阶段分配** | Chunk + SubAllocate | 减少系统调用，提高分配效率 |
| **Lazy Allocation** | size=0 时，首次 SubAllocate 才 ChunkAllocate | 避免预分配浪费 |
| **Lazy Free** | lazyFreeCount/ReuseSizeLimit | 延迟归还 Chunk，提高复用率 |
| **区间树追踪** | SegmentTracker (std::map) | Free 时 O(log n) 查找已分配段 |
| **Bucket 分级** | 64 个 log2 分级的自由链表 | 快速定位合适大小的空闲段 |
| **Bitmap 加速** | m_EltMappingHash | O(1) 判断 bucket 是否非空 |
| **Warm Pool** | ExternalAllocator 池 (size=4) | 避免频繁创建/销毁临时分配器 |
| **Deferred Free** | m_deferFreeMap (InternalMemMgr) | GPU 内存不立即归还 OS，等待复用 |
| **Intrusive Free List** | HW IP 引擎的 ctxMemFreeList | 零分配开销的提交上下文回收 |
| **Ring Buffer** | DMA Upload Ring | 固定大小环形槽位 + 动态扩容 |

### 12.2 关键性能决策

```
1. assuredFit vs bestFit
   ├─ assuredFit (默认): 从足够大的 bucket 直接取，大概率成功
   │   速度快 (tryLimit=1)，但可能有内部碎片
   └─ bestFit: 从小到大找最佳匹配
       减少碎片，但搜索开销更高

2. fast vs optimal insertion
   ├─ fast (默认): 插入到 bucket 头部 O(1)
   └─ optimal: 按大小排序插入 O(n)
       但优化了后续查找

3. 默认 Chunk 大小
   ├─ 用户池: s_DefaultChunkAllocSize = 32 MB  (Core 层)
   └─ HAL 池:  s_DefaultChunkAllocSize = 2 MB   (HAL 层)
       小 Chunk 减少内部碎片，大 Chunk 减少系统调用

4. 线程安全
   ├─ std::recursive_mutex (M3D MemoryPool)
   │   递归锁支持 FullAllocate → SubAllocate → ChunkAllocate → SubAllocate 嵌套
   ├─ Util::RWLock (InternalMemMgr references + deferFree)
   │   读多写少场景的读写锁优化
   └─ std::mutex (MemMgr, CmdPool 等)

5. 引擎隔离
   └─ 9 个 HW IP 引擎各自维护独立的 ctxMemFreeList
      避免跨引擎竞争，利用引擎级别的并行性
```

### 12.3 已知问题与 TODO

| 问题 | 位置 | 说明 |
|------|------|------|
| Lock 可能不必要 | `memoryPool.cpp:103,202,215` | 多处标注 `//TODO: We may be enable not to use the lock` |
| Pool createInfo 重建 | `mu_mempool.cpp:37,98` | Driver 层使用 HAL createInfo 而非 MUSA 层自己的 |
| autoMemoryReuse 未启用 | `cmdPool.cpp:62` | `// wait for m3d to add support for busy chunk tracking` |
| NUMA 亲和性未实现 | `memory.cpp:464` | `// TODO: Support thread numa affinitive` |
| SVM Windows 不支持 | `memory.cpp:401` | `// TODO: svm is not supported on windows yet` |

### 12.4 时序图：一次完整的内存池分配

```
应用程序          Driver层           Core层             HAL/M3D层           GPU
  │                │                  │                    │                  │
  │ muMemPoolCreate│                  │                    │                  │
  │───────────────>│                  │                    │                  │
  │                │ InitPlatform()   │                    │                  │
  │                │────────┐         │                    │                  │
  │                │<───────┘         │                    │                  │
  │                │                  │                    │                  │
  │                │ new MemoryPool() │                    │                  │
  │                │────────────────>│                    │                  │
  │                │                  │ Init(createInfo)   │                  │
  │                │                  │───────────────────>│                  │
  │                │                  │                    │ MemMgr::         │
  │                │                  │                    │ CreateUserPool() │
  │                │                  │                    │────────┐         │
  │                │                  │                    │ new    │         │
  │                │                  │                    │ MemoryPool       │
  │                │                  │                    │ Init() │         │
  │                │                  │                    │<───────┘         │
  │                │                  │<───────────────────│                  │
  │                │<─────────────────│                    │                  │
  │    pool handle  │                  │                    │                  │
  │<───────────────│                  │                    │                  │
  │                │                  │                    │                  │
  │ muMemAllocFromPool               │                    │                  │
  │───────────────>│                  │                    │                  │
  │                │ CreateMemory()   │                    │                  │
  │                │─────────────────>│                    │                  │
  │                │                  │ InitFromPool()     │                  │
  │                │                  │───────────────────>│                  │
  │                │                  │                    │ FullAllocate()   │
  │                │                  │                    │────────┐         │
  │                │                  │                    │  ①     │         │
  │                │                  │                    │ SubAllocate()    │
  │                │                  │                    │ ─ 找Bucket       │
  │                │                  │                    │ ─ FindBucket     │
  │                │                  │                    │ ─ ResourceSplit  │
  │                │                  │                    │   (分裂段)       │
  │                │                  │                    │  ②              │
  │                │                  │                    │ if NotFound:     │
  │                │                  │                    │   ChunkAllocate()│
  │                │                  │                    │   ───────────────>│CreateMemory
  │                │                  │                    │   ─ ResourceAdd  │──>│
  │                │                  │                    │   ─ 再SubAllocate│   │
  │                │                  │                    │<────────┘         │   │
  │                │                  │<───────────────────│                  │   │
  │                │<─────────────────│                    │                  │   │
  │    device ptr   │                  │                    │                  │   │
  │<───────────────│                  │                    │                  │   │
  │                │                  │                    │                  │   │
  │ muLaunchKernel  │                  │                    │                  │   │
  │───────────────>│ 使用 GPU 内存执行计算 ──────────────────────────────────────>│
  │                │                  │                    │                  │   │
  │ muMemFree       │                  │                    │                  │   │
  │───────────────>│                  │                    │                  │   │
  │                │ DestroyMemory()  │                    │                  │   │
  │                │─────────────────>│                    │                  │   │
  │                │                  │ Free(pMemory)      │                  │   │
  │                │                  │───────────────────>│                  │   │
  │                │                  │                    │ ─ 合并相邻段      │   │
  │                │                  │                    │ ─ Lazy-free检查   │   │
  │                │                  │                    │ ─ 重新插入FreeList│   │
  │                │                  │<───────────────────│                  │   │
  │                │<─────────────────│                    │                  │   │
  │                │                  │                    │                  │   │
```

---

*本文档基于 musa 项目源码分析生成，commit: 9ba99a5d, branch: bugfix/sw-79049*
