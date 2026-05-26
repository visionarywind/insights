# 13_MemoryPool_deep_dive — MemoryPool 子分配算法详解

> 源码文件：`musa/src/hal/m3d/memoryPool.cpp` (533 行), `musa/src/hal/m3d/memMgr.cpp:46-69` (内部池)

## 1. 数据结构

### ResSegment (资源段)

```cpp
struct ResSegment {
    DevSize base;            // 段基地址 (VA 或物理地址)
    DevSize size;            // 段大小
    bool busy;               // 是否正在使用
    bool isLeftMost;         // 是否为 chunk 最左段
    bool isRightMost;        // 是否为 chunk 最右段
    uint64_t tag;            // pool key 标签, 用于校验归属
    IMemory* pChunkMem;      // 所属 chunk 的 HAL IMemory 指针
    DevSize chunkBase;       // chunk 基地址
    int lazyFreeCount;       // 惰性释放计数器
    ResSegment* pPrevSegment; // 双向链表前驱
    ResSegment* pNextSegment; // 双向链表后继
    ResSegment* pPrevFree;   // 空闲链表前驱 (同大小桶内)
    ResSegment* pNextFree;   // 空闲链表后继
};
```

### MemoryPool 核心成员

```cpp
class MemoryPool {
    Device* m_pDevice;                        // 所属设备
    Hal::MemoryPoolInfo m_Info;               // 池属性 (type/heap/property/viewCapability)
    Hal::MemoryInternalType m_InternalType;    // 内部类型 (Default/UserManaged/Internal)
    DevSize m_ChunkAlignment;                 // chunk 对齐 (heap largestPageSize)
    DevSize m_ChunkAllocSize;                 // 最小 chunk 大小 (= 2MB)
    DevSize m_ReleaseThreshold;               // 惰性释放阈值
    DevSize m_FreeSize;                       // 当前空闲总量
    DevSize m_TotalSize;                      // 当前总分配量
    DevSize m_MemUsedSizeHigh;                // 历史最高使用量
    DevSize m_TotalSizeHigh;                  // 历史最高总量
    ResSegment* m_pHeadSegment;               // 段链表头
    uint64_t m_EltMappingHash;                // 位图: 标记哪些大小桶非空
    ResSegment* m_FreeBuckets[s_FreeTableLimit];  // 空闲桶数组 (48 或 52 个)
    std::recursive_mutex m_Lock;
    // ...
};
```

### 常量

```cpp
static const uint32_t s_FreeTableLimit = s_MtgpuMaxGpuMmuBits;  // 48 (Linux) 或 52 (Windows)
static const DevSize s_DefaultChunkAllocSize = 2 * 1024 * 1024; // 2MB
static const DevSize s_DefaultLazyFreeThreshold = ...;          // 默认惰性释放阈值
```

## 2. 分配算法: FullAllocate

```
FullAllocate(allocInfo, &offset, &ppMemory)
  │
  ├─ SubAllocate()  ──→  尝试 O(1) 找到空闲块
  │     │
  │     └─ 失败 (errorNotFound)
  │         │
  │         └─ ChunkAllocate()  ──→  向 KMD 申请 2MB chunk
  │               │
  │               └─ 再次 SubAllocate()  ──→  从新 chunk 切分 (必定成功)
  │
  └─ 返回 offset 和 ppMemory
```

## 3. 查找算法: SubAllocate

```
SubAllocate(allocInfo, &offset, &ppMemory)
  │
  ├─ 1. 计算 size 对应的 bucket 范围:
  │     indexLow  = Log2(size)
  │     indexHigh = Log2(size + alignment - 1)  (若 alignment > 1)
  │
  ├─ 2. 根据策略查找:
  │
  │   [bestFit 策略]:
  │     optionalMappingHash = ~((1 << indexLow) - 1) & m_EltMappingHash
  │     扫描 indexLow → max, 找第一个非空桶
  │     在每个桶内遍历链表, 找第一个满足 size+alignment 的块
  │
  │   [assuredFit 策略 (默认)]:
  │     optionalMappingHash = ~((1 << (indexHigh+1)) - 1) & m_EltMappingHash
  │     优先 indexHigh+1 桶 (刚好匹配), O(1) 查找
  │     失败则从 indexHigh 递减到 indexLow, 找第一个非空桶
  │     在桶内只查 1 个元素 (tryLimit=1, 最坏情况查遍 indexLow 到 indexHigh)
  │
  ├─ 3. 若找到 pMemRes:
  │     ResourceSplit(pMemRes, size, alignment, &subAllocAddr)
  │     ──→ 分割前后多余部分, 返回对齐后的 subAllocAddr
  │     *offset = subAllocAddr - pMemRes->chunkBase
  │     *ppMemory = pMemRes->pChunkMem
  │
  └─ 4. 未找到 → Result::errorNotFound
```

## 4. 分割算法: ResourceSplit

```
ResourceSplit(pMemRes, size, alignment, &pBase)
  │
  ├─ 1. 计算 alignedBase = AlignUp(pMemRes->base, alignment)
  │
  ├─ 2. 从空闲列表/链表中移除 pMemRes
  │     (FreeListRemove, 设置 busy=true)
  │
  ├─ 3. 若 noSplit 策略, 直接返回 alignedBase
  │
  ├─ 4. 前间隙处理 (alignedBase > pMemRes->base):
  │     splitSize = alignedBase - pMemRes->base
  │     创建 pNeighbour = new ResSegment(alignedBase, size-splitSize)
  │     插入 pMemRes 之后
  │     pMemRes->size = splitSize
  │     pMemRes 加入空闲列表 (前间隙)
  │     pMemRes = pNeighbour  (指向有效段)
  │
  ├─ 5. 后多余处理 (pMemRes->size > size):
  │     创建 pNeighbour = new ResSegment(base+size, size-leftover)
  │     插入 pMemRes 之后
  │     pMemRes->size = size
  │     pNeighbour 加入空闲列表 (后多余)
  │
  └─ 6. 将 pMemRes 加入 m_SegmentTracker[{alignedBase, size}] (仅 Virtual)
      *pBase = alignedBase
```

## 5. 释放算法: Free

```
Free(pMemory, base, size)
  │
  ├─ 1. 通过 m_SegmentTracker[{base, size}] 找到 ResSegment
  │   (m_SegmentTracker 是一个 std::map, key 为 MemoryRange{base, size})
  │
  ├─ 2. 校验: pMemRes->base == base && pMemRes->size == size
  │
  ├─ 3. 从 SegmentTracker 移除, m_FreeSize += size
  │
  ├─ 4. 左合并:
  │   if (!isLeftMost && !pPrev->busy):
  │     从空闲列表/链表中移除 pPrev
  │     pMemRes->base = pPrev->base
  │     pMemRes->size += pPrev->size
  │     pMemRes->isLeftMost = pPrev->isLeftMost
  │     delete pPrev
  │
  ├─ 5. 右合并:
  │   if (!isRightMost && !pNext->busy):
  │     从空闲列表/链表中移除 pNext
  │     pMemRes->size += pNext->size
  │     pMemRes->isRightMost = pNext->isRightMost
  │     delete pNext
  │
  └─ 6. 若合并后只剩一个段 (isLeftMost && isRightMost):
      尝试 ResourceRemove() → 销毁整个 chunk
      否则: FreeListInsert() → 加入空闲列表
```

## 6. 惰性释放: ResourceRemove & TrimPool

### ResourceRemove (尝试销毁 chunk)

```
ResourceRemove(pMemRes)
  │
  +-- if (isLeftMost && isRightMost):  ← 整个 chunk 只有一个段
  │     lazyFreeCount++
  │     if (lazyFreeCount > reuseCountLimit ||
  │         chunkSize > reuseSizeLimit):
  │       SegmentListRemove(pMemRes)
  │       pChunkMem->Destroy()     ← 真正销毁 KMD 内存
  │       delete pMemRes
  │       return true (已释放)
  └-- return false (仍保留)
```

### TrimPool (主动释放)

```
TrimPool(targetSize)
  │
  +-- while (m_TotalSize > targetSize):
  │     for (每个空闲桶, 从小到大):
  │       查找 isLeftMost && isRightMost 的完整 chunk
  │       找到 → 销毁, TotalSize/FreeSize 减少, break
  │     若本轮未释放任何 chunk → break
```

## 7. 执行示例

### 示例 1: 首次 8KB 分配

```
初始状态: pool 为空
├─ ChunkAllocate(8192):
│   chunkSize = AlignUp(8192, 2MB) = 2MB
│   向 KMD 申请 2MB chunk (ioctl)
│   创建 ResSegment{base=0, size=2MB}
│   FreeListInsert → bucket[21] (Log2(2MB)=21)
│   m_EltMappingHash |= (1<<21)
│
├─ SubAllocate(8192):
│   indexLow = Log2(8192) = 13
│   扫描 bucket[21] → 找到 2MB 空闲块
│   ResourceSplit:
│     alignedBase = base (已对齐)
│     size = 8192, 原 size = 2MB
│     后多余 = 2MB - 8192 = 2088960
│     创建右邻居 ResSegment{8192, 2088960}
│     左邻居无 (无前间隙)
│   FreeListInsert(右邻居) → bucket[21]
│   有效段 busy=true, 加入 SegmentTracker
│
结果:
  chunkVA=0x1000, 返回 offset=0
  空闲: [8192, 2MB) (2MB-8192 大小)
```

### 示例 2: 二次 4KB 分配

```
当前状态:
  bucket[21]: ResSegment{8192, 2088960}
  m_EltMappingHash = ... | (1<<21)

SubAllocate(4096):
  indexLow = Log2(4096) = 12
  扫描: bucket[12] 空 → bucket[13] 空 → ... → bucket[21] 有
  FindBucket: 8192+2088960 >= 0+4096 ✓
  ResourceSplit(ResSegment{8192, 2088960}):
    alignedBase = AlignUp(8192, 4096) = 8192
    前间隙 = 0 (已对齐)
    后多余 = 2088960 - 4096 = 2084864
    创建右邻居 {12288, 2084864}
    原段变为 {8192, 4096}, busy=true
  FreeListInsert({12288, 2084864}) → bucket[21]

结果:
  返回 offset=8192 (chunk 内偏移)
  空闲: [12288, 2MB) 和 任何其他历史释放块
```

### 示例 3: 释放 8KB (回到示例 1 状态)

```
Free(virtMem, base=0, size=8192):
  SegmentTracker 查找 → ResSegment{0, 8192}
  移除 SegmentTracker 记录
  m_FreeSize += 8192

  左邻居: isLeftMost=true → 无
  右邻居: {8192, 2088960}, busy=false → 合并!
    移除右邻居空闲列表和链表
    pMemRes->size = 8192 + 2088960 = 2097152 = 2MB
    pMemRes->isRightMost = true

  ResourceRemove(pMemRes):
    isLeftMost && isRightMost → true
    lazyFreeCount > limit → true (假设)
    销毁 chunk (ioctl)
    delete pMemRes
    TotalSize -= 2MB, FreeSize -= 2MB

结果: 完全释放, 回归空池状态
```

### 示例 4: 对齐分配导致的前后分割

```
当前状态: bucket[21] 有 ResSegment{0, 2MB}

SubAllocate(1000, alignment=4096):
  indexLow = Log2(1000) = 9 (向上取 10)
  indexHigh = Log2(1000+4096-1) = Log2(5095) = 12

  扫描 bucket[12] (assuredFit) → bucket[21] 有
  FindBucket → ResSegment{0, 2MB}

  ResourceSplit:
    alignedBase = AlignUp(0, 4096) = 0 (已对齐, 此例无前间隙)
    后多余: 2MB - 1000 = 2096576
    创建右邻居 {1000, 2096576}
    1000 不对齐到页, 但 size 只有 1000
    实际上需要对齐 base 到 4096 → 前间隙 0 → base=0
    但 size=1000, 下一页对齐 = 4096
    
    修正: 实际分配 4096 (覆盖整个页)
    但代码中 size 保持 1000, 分割后:
    pMemRes{0, 1000}, 邻居{1000, 2096576}
    
    (注: 实际对齐逻辑中, 若 alignment 大于 size,
     会分配 alignment 大小的块)
```

## 8. InsertPolicy 的影响

```
InsertPolicy::fast (默认, CPU/memoryPool.h):
  +-- 头插法, O(1)
  +-- 桶内顺序不确定
  +-- 适合频繁分配/释放
  +-- FindBucket 可能需要遍历更多元素

InsertPolicy::optimal:
  +-- 按 size 从小到大排序插入, O(n)
  +-- 桶内有序, FindBucket 第一个命中即最优
  +-- 适合 bestFit 策略, 减少内存碎片
```

## 9. Hash 冲突与 m_EltMappingHash

`m_EltMappingHash` 是一个 64 位位图:

```
bit[i] = 1  表示 bucket[i] 中至少有一个空闲段
bit[i] = 0  表示 bucket[i] 为空

用途:
  - 快速判断是否有满足 size 要求的空闲块
  - ~((1 << indexLow) - 1) & hash  → 只看 ≥indexLow 的桶
  - BitMaskScanForward 找最低位 1 → 最小满足桶
  - BitMaskScanReverse 找最高位 1 → 最大满足桶
```

`m_EltMappingHash` 的最大位数 = `s_FreeTableLimit` = GPU MMU 虚拟地址位数 (48 或 52 位)。

## 10. 相关源码位置

| 文件 | 行数 | 说明 |
|------|------|------|
| `musa/src/hal/m3d/memoryPool.cpp` | 全文件 533 行 | 完整 pool 实现 |
| `musa/src/hal/m3d/memMgr.cpp` | 46-69 | 内部 pool 创建 (GetInternalPool) |
| `musa/src/hal/m3d/memory.h` | - | MemoryPool 类声明 |
| `musa/src/hal/m3d/memoryPool.h` | - | ResSegment / 常量定义 |