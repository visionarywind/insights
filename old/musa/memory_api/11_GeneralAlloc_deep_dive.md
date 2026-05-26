# 11_GeneralAlloc_deep_dive — 通用设备内存分配深度分析

> 源码文件：`musa/src/musa/core/memory.cpp:462-497`，`musa/src/hal/m3d/memMgr.cpp:81-147`，`musa/src/hal/m3d/memoryPool.cpp:82-212`，`musa/src/hal/m3d/memory.cpp:366-426`

## 1. 功能概述

`GeneralAlloc` 是设备内存分配的核心实现，对应 `muMemAlloc` 的默认路径。本文档详细拆解从 `GeneralAlloc` 开始到最终 KMD ioctl 的完整调用链。

## 2. GeneralAlloc 源码逐行分析

```cpp
// memory.cpp:462
MUresult Memory::GeneralAlloc(size_t size, size_t alignment, unsigned int flags)
```

### Step 1: 构建 Shape 和保存 flags

```cpp
m_Shape = { size, 1, 1, size };    // width/height/depth/pitch
m_Flags = flags;                    // = 0x07 (Virtual|DeviceMapped|SubAllocatable)
```

### Step 2: 构建 Hal::MemoryCreateInfo

```cpp
Hal::MemoryCreateInfo createInfo{};
createInfo.type                = Hal::memoryTypeAlloc;
createInfo.alloc.type          = Hal::memoryAllocTypeDeviceLocal;   // GPU 显存
createInfo.alloc.heap          = Hal::MemoryHeap::largePage;        // 优先大页
createInfo.alloc.property      = flags;                              // 用户传入的 0x07

// Core 自动追加的属性:
createInfo.alloc.property     |= Hal::memoryPropertyPhysical |
                                 Hal::memoryPropertySharedVirtualAddress;

// 由 Virtual flag 推导的属性组:
if (flags & Hal::memoryPropertyVirtual) {
    createInfo.alloc.property |= Hal::memoryPropertyDeviceVisible |
                                 Hal::memoryPropertyHostVisible |
                                 Hal::memoryPropertyHostCoherent |
                                 Hal::memoryPropertyDeviceWriteable |
                                 Hal::memoryPropertyDeviceCached;
}

// 由 DeviceMapped flag 推导的 view capability:
createInfo.alloc.viewCapability = Hal::memoryViewCapabilityExportable;
if (flags & Hal::memoryPropertyDeviceMapped) {
    createInfo.alloc.viewCapability |= Hal::memoryViewCapabilityPeerAccessible |
                                       Hal::memoryViewCapabilityIpcExportable;
}

// 对齐 (取用户对齐与设备最小分配对齐的较大值):
createInfo.alloc.alignment = std::max(alignment, device.minAllocAlign);
createInfo.alloc.numaId    = NUMA_NO_NODE;
createInfo.alloc.size      = size;
```

### Step 3: 分叉 — Sub-Allocation vs 裸分配

```cpp
if (createInfo.alloc.property & Hal::memoryPropertySubAllocatable) {
    // 路径 A: Pool 子分配 (默认)
    status = HalToMuResult(device.Hal().GetMemMgr()->Allocate(
        createInfo.alloc, &m_Offset, &m_pHalMemory));
} else {
    // 路径 B: 直接创建 HAL 内存 (裸 KMD)
    status = HalToMuResult(device.Hal().CreateMemory(createInfo, &m_pHalMemory));
}
```

## 3. 路径 A: MemMgr::Allocate 调用链

```
MemMgr::Allocate(allocInfo, &offset, &ppMemory, ppPool)    [memMgr.cpp:81]
  │
  ├─ 1. 参数校验:
  │     size != 0, size+alignment 不溢出
  │     deviceLocal 且 size > totalGlobalMem → OUT_OF_MEMORY
  │
  ├─ 2. 确定 pool:                                          [memMgr.cpp:98-134]
  │     if (ppMemoryPool && *ppMemoryPool):
  │       pPool = *ppMemoryPool          ← 用户指定了 pool
  │       校验 pool 属性与 allocInfo 一致
  │     else:
  │       poolInfo = {type, heap, property, viewCapability, numaId}
  │       pPool = m_PoolRefs.Get(MakeKey(poolInfo))  ← 查找已有 pool
  │
  │       // 未命中则创建:
  │       if (!pPool || 属性不匹配):
  │         poolCreateInfo = {
  │           info: poolInfo,
  │           usageFlags: {userManaged=false, internal=false},
  │           internalType: Default,
  │           minEnlargeChunkSize: 2MB,          ← 关键常量
  │           reuseCountLimit: DefaultLazyFreeThreshold,
  │           reuseSizeLimit: 2MB,
  │           size: allocInfo.size,
  │           alignment: allocInfo.alignment
  │         }
  │         CreatePoolNoLock(poolCreateInfo, &pPool)
  │
  ├─ 3. pPool->FullAllocate(allocInfo, &offset, &ppMemory) [memMgr.cpp:144]
  │     (详见 memoryPool 文档)
  │     offset → m_Offset (Sub-Allocation 块在 chunk 中的偏移)
  │     ppMemory → m_pHalMemory (底层物理内存对象)
  │
  └─ 返回: ppMemory = 底层 HAL IMemory*, offset = 子分配偏移
```

## 4. MemoryPool::FullAllocate 调用链

```
MemoryPool::FullAllocate(allocInfo, &offset, &ppMemory)    [memoryPool.cpp:82]
  │
  │  std::lock_guard<std::recursive_mutex> lg(m_Lock);
  │
  ├─ 1. SubAllocate(allocInfo, &offset, &ppMemory)          [memoryPool.cpp:85]
  │     │
  │     ├─ 1a. 计算 size 对应的 bucket index:
  │     │     indexLow = Log2(size)
  │     │     indexHigh = Log2(size + alignment - 1) (若 alignment > 1)
  │     │
  │     ├─ 1b. 选择查找策略:
  │     │     [bestFit]:  遍历 indexLow→indexHigh, 找最小的满足桶
  │     │     [assuredFit]: 优先 indexLow+1 桶, 失败再从高往低扫描
  │     │
  │     ├─ 1c. FindBucket(bucket, size, alignment, tryLimit)  [memoryPool.cpp:14]
  │     │     遍历桶内空闲链表, 找 base+size >= alignedBase+size 的块
  │     │     tryLimit=1 (assuredFit) 或 UINT64_MAX (bestFit)
  │     │
  │     ├─ 1d. ResourceSplit(pMemRes, size, alignment, &subAllocAddr) [memoryPool.cpp:358]
  │     │     │
  │     │     ├─ 计算 alignedBase = AlignUp(pMemRes->base, alignment)
  │     │     │
  │     │     ├─ 前间隙处理 (alignedBase > base):
  │     │     │     创建新 ResSegment, 大小 = alignedBase - base
  │     │     │     插入链表, 加入空闲列表
  │     │     │     pMemRes 指向新段
  │     │     │
  │     │     ├─ 后多余处理 (size < pMemRes->size):
  │     │     │     创建新 ResSegment, 大小 = 原大小 - size
  │     │     │     插入链表, 加入空闲列表
  │     │     │
  │     │     ├─ 更新 pMemRes->size = size, busy=true
  │     │     │
  │     │     └─ m_SegmentTracker[{alignedBase, size}] = pMemRes  (仅 Virtual)
  │     │       (建立 VA 范围 → segment 的映射, 用于 Free 和 FindRange)
  │     │
  │     └─ *offset = alignedBase - pMemRes->chunkBase + chunkBaseVA
  │       *ppMemory = pMemRes->pChunkMem
  │       m_FreeSize -= size
  │
  │
  ├─ 2. (SubAllocate 失败 → errorNotFound)
  │
  └─ 3. ChunkAllocate(allocInfo)                             [memoryPool.cpp:153]
        │
        ├─ 计算 chunkSize:
        │   chunkSize = size
        │   + (alignment > chunkAlignment ? alignment - chunkAlignment : 0)
        │   = AlignUp(chunkSize, chunkAlignment)
        │   = AlignUp(chunkSize, 2MB)    ← 最小 chunk 大小
        │
        ├─ 根据 heap 和 property 选择创建路径:                  [memoryPool.cpp:174-191]
        │   if (!(property & Physical)):
        │     // 虚拟内存路径
        │     createInfo = { type=Virtual, size=chunkSize,
        │                     alignment=chunkAlignment, heap=allocInfo.heap }
        │     Platform::CreateMemory(createInfo, &pChunkMem)
        │   else:
        │     // 物理内存路径
        │     createInfo = { type=Alloc, alloc=allocInfo }
        │     createInfo.alloc.size = chunkSize
        │     createInfo.alloc.alignment = chunkAlignment
        │     device->CreateMemory(createInfo, &pChunkMem)
        │
        ├─ 计算 base:                                         [memoryPool.cpp:194]
        │   base = (Virtual property ? pChunkMem->GetDeviceVirtualAddress()
        │                                : chunkAlignment)
        │   size = pChunkMem->GetSize()
        │
        ├─ 创建 ResSegment:                                   [memoryPool.cpp:197-200]
        │   pMemRes = new ResSegment(base, size)
        │   pMemRes->tag = MakeKey(poolInfo)
        │   pMemRes->pChunkMem = pChunkMem
        │   pMemRes->chunkBase = base
        │
        ├─ ResourceAdd(pMemRes)  [memoryPool.cpp:305]        加入链表+空闲表
        │   ├─ pMemRes->pNextSegment = m_pHeadSegment        头插法
        │   ├─ FreeListInsert(pMemRes)                       加入空闲桶
        │   │     index = Log2(size)
        │   │     m_EltMappingHash |= (1ULL << index)       更新位图
        │   └─ m_TotalSize += size, m_FreeSize += size
        │
        └─ 4. 再次 SubAllocate(allocInfo, &offset, &ppMemory) [memoryPool.cpp:90]
              (此时一定能从新 chunk 中切分出所需内存)
```

## 5. FreeListInsert 算法细节

```
MemoryPool::FreeListInsert(pMemRes)                        [memoryPool.cpp:416]
  │
  +─── index = Log2(pMemRes->size)                         按大小对数分桶
  │
  +─── if (InsertPolicy::optimal):                          [memoryPool.cpp:420]
  │     按 size 从小到大排序插入链表
  │     保持: prev->size < pMemRes->size <= next->size
  │     (提高后续 bestFit 查找效率)
  │
  └─── else: (InsertPolicy::fast)                          [memoryPool.cpp:447]
        头插法: O(1) 插入
        pMemRes->pNextFree = m_FreeBuckets[index]
        m_FreeBuckets[index] = pMemRes
```

## 6. 内存释放与合并

```
MemoryPool::Free(pMemory, base, size)                      [memoryPool.cpp:214]
  │
  +── 通过 m_SegmentTracker[{base, size}] 定位 ResSegment  [memoryPool.cpp:220-224]
  │
  +── 标记空闲: pMemRes->busy = false                      [memoryPool.cpp:226]
  │   m_FreeSize += size
  │
  +── 左合并:                                              [memoryPool.cpp:229-237]
  │   if (!isLeftMost && !pPrev->busy):
  │     从空闲表移除 pPrev
  │     pMemRes->base = pPrev->base
  │     pMemRes->size += pPrev->size
  │     pMemRes->isLeftMost = pPrev->isLeftMost
  │     delete pPrev
  │
  +── 右合并:                                              [memoryPool.cpp:241-249]
  │   if (!isRightMost && !pNext->busy):
  │     从空闲表移除 pNext
  │     pMemRes->size += pNext->size
  │     pMemRes->isRightMost = pNext->isRightMost
  │     delete pNext
  │
  └── 尝试完全释放 ResSegment:                             [memoryPool.cpp:251]
      ResourceRemove(pMemRes) → true (仅剩一个 chunk)
      → pChunkMem->Destroy() + delete pMemRes
      → m_TotalSize, m_FreeSize 减少
```

## 7. TrimPool: 惰性释放机制

```
当 User Pool 空闲时, MemMgr::UpdateUserPools() 会被调用:     [memMgr.cpp:229-235]
  │
  +-- for each userPool:
  │     pool->TrimPool(releaseThreshold)
  │
TrimPool(value):                                            [memoryPool.cpp:480]
  │
  +── while (m_TotalSize > value):
  │     for (每个空闲桶, 从小到大):
  │       查找 isLeftMost && isRightMost 的完整 chunk
  │       找到 → 销毁 chunk, 减少 TotalSize/FreeSize
  │       找不到 → break
  │
  +── 目的: 释放长时间未使用的 chunk, 减少 GPU 显存占用
```

## 8. property 推导完整汇总

```
输入 flags = Virtual(0x01) | DeviceMapped(0x02) | SubAllocatable(0x04) = 0x07
             │
             ├── Core 自动追加: Physical(0x08) | SharedVA(0x10)
             │
             ├── 由 Virtual(0x01):
             │     DeviceVisible(0x20) | HostVisible(0x40)
             │     HostCoherent(0x80) | DeviceWriteable(0x100)
             │     DeviceCached(0x200)
             │
             └── 由 DeviceMapped(0x02):
                   (property 部分无追加)
                   viewCapability 追加: PeerAccessible | IpcExportable

最终 HAL property = 0x3FF (所有位全开除 HostWriteCombined/ThreadNumaAffinitive)
最终 viewCapability = Exportable(0x01) | PeerAccessible(0x02) | IpcExportable(0x04) = 0x07
```

## 9. 相关源码位置

| 文件 | 行数 | 说明 |
|------|------|------|
| `musa/src/musa/core/memory.cpp` | 462-497 | `GeneralAlloc` |
| `musa/src/hal/m3d/memMgr.cpp` | 81-147 | `MemMgr::Allocate` |
| `musa/src/hal/m3d/memoryPool.cpp` | 82-151 | `FullAllocate/SubAllocate/ChunkAllocate` |
| `musa/src/hal/m3d/memoryPool.cpp` | 214-259 | `Pool::Free` (合并) |
| `musa/src/hal/m3d/memoryPool.cpp` | 358-413 | `ResourceSplit` |
| `musa/src/hal/m3d/memoryPool.cpp` | 416-478 | `FreeListInsert/Remove` |
| `musa/src/hal/m3d/memoryPool.cpp` | 480-510 | `TrimPool` |
| `musa/src/hal/m3d/memory.cpp` | 366-426 | `InitGeneralDeviceMemory` |