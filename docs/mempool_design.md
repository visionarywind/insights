# MUSA MemoryPool 设计原理深度剖析

> **📖 相关文档**: [memory_analysis.md](memory_analysis.md) | [decision_logic.md](decision_logic.md) | [pooling_analysis.md](pooling_analysis.md)

本文从**数据结构** → **算法实现** → **具体数值示例** → **完整时序图** 四个维度，逐层剖析 MUSA 中 MemoryPool 的设计原理。

---

## 目录

1. [MemoryPool 在 MUSA 中的位置](#1-memorypool-在-musa-中的位置)
2. [核心数据结构：ResSegment + 双链表 + Bucket](#2-核心数据结构ressegment--双链表--bucket)
3. [SubAllocate：子分配算法（带数值示例）](#3-suballocate子分配算法带数值示例)
4. [ResourceSplit：分裂算法（带对齐示例）](#4-resourcesplit分裂算法带对齐示例)
5. [Free：释放与合并（带合并示例）](#5-free释放与合并带合并示例)
6. [ChunkAllocate：创建新 Chunk](#6-chunkallocate创建新-chunk)
7. [Lazy Free：延迟回收策略](#7-lazy-free延迟回收策略)
8. [TrimPool：内存回收](#8-trimpool内存回收)
9. [三层 API 完整时序图](#9-三层-api-完整时序图)

---

## 1. MemoryPool 在 MUSA 中的位置

### 1.1 三层架构

```
用户代码
  │ muMemPoolCreate(pool, props)
  ▼
┌──────────────────────────────────────────────────────┐
│ Driver 层 (mu_mempool.cpp)                           │
│  - 参数校验 (pool != nullptr, props 合法)             │
│  - 构建 Hal::MemoryPoolCreateInfo                     │
│  - new Musa::MemoryPool(pDevice) → Init(createInfo)   │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│ Core 层 (memoryPool.cpp/h)                           │
│  - 封装 Hal::IMemoryPool                              │
│  - 管理分配追踪 (m_MemoryAllocations)                  │
│  - 处理 IPC 共享内存 (shm_open/mmap)                   │
│  - 代理 SetAccess/GetAccess/Trim                      │
│  - 默认 Chunk 大小: 32 MB                             │
└──────────────────────┬───────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────┐
│ HAL/M3D 层 (hal/m3d/memoryPool.cpp/h) — 核心算法      │
│  - 64-Bucket 自由链表                                 │
│  - ResSegment 双向链表 + 区间树                        │
│  - SubAllocate / ChunkAllocate / Free                 │
│  - 默认 Chunk 大小: 2 MB                              │
└──────────────────────────────────────────────────────┘
```

### 1.2 为什么需要 MemoryPool？

GPU 内存分配需要经过 KMD（内核驱动）的 ioctl 调用，开销较大。MemoryPool 的思路是：

```
不用 MemoryPool:                       用 MemoryPool:
                                       
用户: muMemAlloc(4KB)                  初始化: ChunkAllocate(2MB) → 一次 ioctl
  → ioctl → KMD 分配 4KB               │
用户: muMemAlloc(4KB)                  用户1: SubAllocate(4KB)  → 从 Chunk 切
  → ioctl → KMD 分配 4KB              用户2: SubAllocate(1MB)  → 从 Chunk 切
用户: muMemAlloc(1MB)                  用户3: SubAllocate(4KB)  → 从 Chunk 切
  → ioctl → KMD 分配 1MB              ...
                                       
100次分配 = 100次 ioctl                100次分配 = 1次 ioctl (初始 Chunk)
```

---

## 2. 核心数据结构：ResSegment + 双链表 + Bucket

### 2.1 ResSegment — 内存资源段

```cpp
struct ResSegment {
    // ===== GPU 内存 Chunk =====
    IMemory* pChunkMem;      // 指向底层 GPU 内存块 (Hal::IMemory)
    DevSize  chunkBase;      // 该 Chunk 的起始地址 (VA)
    DevSize  base;           // 此段在 Chunk 内的起始偏移
    DevSize  size;           // 此段的大小
    bool     busy;           // true=已分配, false=空闲
    
    // ===== Chunk 边界标记 =====
    bool isLeftMost;         // 此段是不是 Chunk 的最左段
    bool isRightMost;        // 此段是不是 Chunk 的最右段
    
    // ===== Lazy Free =====
    DevSize lazyFreeCount;   // 整个 Chunk 变为空闲的次数
    
    // ===== 地址序双向链表 (Segment List) =====
    ResSegment* pNextSegment; // 下一个段 (地址递增)
    ResSegment* pPrevSegment; // 上一个段 (地址递减)
    
    // ===== 空闲链表 (Free List) =====
    ResSegment* pNextFree;   // 同一 bucket 中的下一个空闲段
    ResSegment* pPrevFree;   // 同一 bucket 中的上一个空闲段
    
    DevSize tag;             // Pool 属性 Key (用于 MemMgr 查找)
};
```

**关键设计：一个 ResSegment 同时存在于两个链表中**

```
Chunk: ┌────────┬────────┬─────────┐
       │ Seg A  │ Seg B  │  Seg C  │
       │ (busy) │ (free) │ (busy)  │
       └────────┴────────┴─────────┘

Segment List (pNextSegment/pPrevSegment):
  Seg A → Seg B → Seg C    (按地址序排列，所有段都在这)

Free List (pNextFree/pPrevFree):
  Bucket[20] → Seg B       (只有空闲段挂在对应 Bucket 下)
```

### 2.2 64-Bucket 自由链表

```
m_FreeBuckets[0..63]:  每个 bucket 是一个空闲段的单向链表头

Bucket[0]   → 段大小 1B ~ 2B
Bucket[1]   → 段大小 2B ~ 4B
Bucket[2]   → 段大小 4B ~ 8B
...
Bucket[10]  → 段大小 1KB ~ 2KB
...
Bucket[20]  → 段大小 1MB ~ 2MB
...
Bucket[63]  → 段大小 2^63 ~ 2^64

bucket index = Log2(segment->size)
```

**Bitmap 加速 (m_EltMappingHash)**:

```cpp
// 64 位 bitmap, bit[i]=1 表示 Bucket[i] 非空
DevSize m_EltMappingHash;

// 插入时设置 bit
m_EltMappingHash |= (1ULL << index);

// 移除时清除 bit (仅当该 bucket 变空)
if (!m_FreeBuckets[index]) {
    m_EltMappingHash &= ~(1ULL << index);
}

// 查找非空 bucket: O(1) 的 BitMaskScanForward
BitMaskScanForward(&index, m_EltMappingHash & mask);
```

### 2.3 区间树 (Segment Tracker)

```cpp
// std::map<MemoryRange, ResSegment*>
// 用于 Free 时快速查找已分配的段

struct MemoryRange {
    DevSize m_BasePointer;
    DevSize m_EndPointer;  // base + size - 1
    
    bool operator<(const MemoryRange& other) const {
        return this->m_EndPointer < other.m_BasePointer;
        // ★ 区间树排序: 按 end 比较, 不相交区间自动排序
    }
};

// Free 时:
auto it = m_SegmentTracker.find(MemoryRange{base, size});
// O(log n) 找到对应的 ResSegment*
```

---

## 3. SubAllocate：子分配算法（带数值示例）

### 3.1 算法伪代码

```
SubAllocate(allocInfo):
  ① 确定搜索范围: indexLow = Log2(size), indexHigh = Log2(size+alignment-1)
  
  ② 查找空闲段 (assuredFit 策略):
     - 从 bitmap 找到 >= indexHigh+1 的第一个非空 bucket
     - 如果找到: 在该 bucket 中搜索 (tryLimit=1)
     - 如果没找到: 从 indexHigh 向下遍历 (直到 indexLow)
     
  ③ 分裂: ResourceSplit(segment, size, alignment)
     → 将找到的段分裂成 [分配] + [剩余-前] + [剩余-后]
     
  ④ 返回: *ppMemory = segment->pChunkMem
            *pOffset  = 分配的起始偏移
```

### 3.2 数值示例：分配 3MB（对齐 64KB）

**初始状态**：Pool 中有一个 16MB 的 Chunk

```
Chunk: ┌──────────────────────────────────────────────┐
       │              16MB (free)                     │
       │   base=0x10000000, size=16MB                 │
       │   挂在 Bucket[24] (8MB~16MB)                  │
       └──────────────────────────────────────────────┘

m_FreeBuckets:
  [0]..[23] = nullptr
  [24] → Seg(0x10000000, 16MB, busy=false)    ← 只有这个
  [25]..[63] = nullptr

m_EltMappingHash = 1 << 24   (bit 24 = 1)
```

**Step 1: 确定搜索范围**

```
allocInfo.size      = 3MB = 0x300000
allocInfo.alignment = 64KB = 0x10000

indexLow  = Log2(3MB)     = 22   (2^22 = 4MB, 取 ceil → 实际需扫描到 22)
indexHigh = Log2(3MB + 64KB - 1) = 22   (对齐后仍 < 4MB)
```

**Step 2: assuredFit 查找**

```
optionalMappingHash = ~((1 << (22+1)) - 1) & m_EltMappingHash
                    = ~((1 << 23) - 1) & (1 << 24)
                    = ~0x7FFFFF & 0x1000000
                    = 0x1000000   (只有 bit 24)

BitMaskScanForward(&index, 0x1000000) → index = 24

FindBucket(m_FreeBuckets[24], 3MB, 64KB, tryLimit=1):
  walker = Seg(0x10000000, 16MB)
  alignedBase = AlignUp(0x10000000, 64KB) = 0x10000000
  if (0x10000000 + 16MB >= 0x10000000 + 3MB) ✓  → 找到!
  return Seg(0x10000000, 16MB)
```

**Step 3: ResourceSplit 分裂**

```
输入: Seg(0x10000000, 16MB), size=3MB, alignment=64KB

alignedBase = AlignUp(0x10000000, 64KB) = 0x10000000

1. 前部分裂: alignedBase == base → 不需要
2. 后部分裂: 3MB < 16MB → 需要

   分配段: Seg(0x10000000, 3MB)    busy=true
   剩余段: Seg(0x10300000, 13MB)   busy=false
           └─ 插入 FreeList: Bucket[24] (Log2(13MB)=24)

结果:
Chunk:
┌───────────────┬────────────────────────────────────┐
│  3MB (busy)   │        13MB (free)                  │
│  0x10000000   │  0x10300000                         │
└───────────────┴────────────────────────────────────┘
                ↑ 插入 m_SegmentTracker

m_FreeSize = 13MB
m_EltMappingHash = 1 << 24   (bit 24 仍为 1，因为有 13MB 段)
```

**再分配 5MB 后的状态**:

```
第二次分配: 5MB, alignment=64KB

Chunk:
┌───────┬───────┬────────────────────────────────────┐
│ 3MB   │ 5MB   │        8MB (free)                   │
│ busy  │ busy  │                                     │
└───────┴───────┴────────────────────────────────────┘

m_EltMappingHash = 1 << 23    (8MB → Bucket[23])

Segment List (pNextSegment/pPrevSegment):
  Seg(0x10000000,3MB,busy) → Seg(0x10300000,5MB,busy) → Seg(0x10800000,8MB,free)

Free List:
  Bucket[23] → Seg(0x10800000,8MB)
```

---

## 4. ResourceSplit：分裂算法（带对齐示例）

### 4.1 完整流程

```
ResourceSplit(pMemRes, size, alignment, pBase):

输入: Seg(0x10004000, 10MB), 请求 3MB, 对齐 1MB

① 计算对齐地址:
   alignedBase = AlignUp(0x10004000, 1MB) = 0x10100000

② 前部分裂 (alignedBase > base):
   原段: [0x10004000, 0x10A04000)  size=10MB
   
   创建前部: Seg(0x10004000, 0xC000)  ← 前面不满足对齐的碎片
   原段缩小为: Seg(0x10100000, 0x9A4000)
   
   前部插入 FreeList, 原段变成对齐后的段

③ 后部分裂 (pMemRes->size > size):
   当前段: Seg(0x10100000, 0x9A4000) = ~9.64MB
   
   分配部分: Seg(0x10100000, 3MB)    busy=true, 插入 SegmentTracker
   剩余部分: Seg(0x10400000, ~6.64MB) busy=false, 插入 FreeList

④ 返回: *pBase = 0x10100000

最终视图:
┌─────────┬───────────────┬──────────────────────────┐
│ 0xC000  │     3MB       │        ~6.64MB            │
│ (free)  │   (busy)      │        (free)             │
│10004000 │  10100000     │  10400000                 │
└─────────┴───────────────┴──────────────────────────┘
  Bucket[16]               Bucket[23]
```

### 4.2 align 导致的前部分裂示意

```
没有对齐需求时:              有对齐需求时 (align=1MB):
┌──────────────────┐        ┌────┬───────────────┐
│  3MB (busy)      │        │碎片│  3MB (busy)    │
│  base=0x10004000 │        │free│  aligned to    │
│                  │        │    │  0x10100000    │
└──────────────────┘        └────┴───────────────┘
一次分裂                     两次分裂 (前部+后部)
```

---

## 5. Free：释放与合并（带合并示例）

### 5.1 完整流程

```
Free(pMemory, base, size):

① 从 m_SegmentTracker 查找:
   MemoryRange range{base, size};
   auto it = m_SegmentTracker.find(range);
   → 找到对应的 busy ResSegment*

② 尝试与左邻合并:
   pLeft = pMemRes->pPrevSegment
   if (exists && !busy):
       FreeListRemove(pLeft)     // 从 bucket 中移除
       SegmentListRemove(pLeft)  // 从 segment 链表中移除
       pMemRes->base = pLeft->base
       pMemRes->size += pLeft->size
       pMemRes->isLeftMost = pLeft->isLeftMost
       delete pLeft

③ 尝试与右邻合并:
   pRight = pMemRes->pNextSegment
   if (exists && !busy):
       FreeListRemove(pRight)
       SegmentListRemove(pRight)
       pMemRes->size += pRight->size
       pMemRes->isRightMost = pRight->isRightMost
       delete pRight

④ Lazy Free 检查:
   if (isLeftMost && isRightMost):  // 整个 Chunk 都空闲了
       lazyFreeCount++
       if (lazyFreeCount > ReuseCountLimit || size > ReuseSizeLimit):
           → 销毁 Chunk, 归还 GPU 内存, delete pMemRes
           return
           
⑤ 重新插入 FreeList:
   FreeListInsert(pMemRes)  // 根据新 size 挂到对应 bucket
```

### 5.2 合并数值示例

```
初始状态 (释放前):
Segment List:
  Seg A(0x10000000,2MB,free) → Seg B(0x10200000,3MB,busy) → Seg C(0x10500000,1MB,free) → Seg D(0x10600000,10MB,busy)

Free List:
  Bucket[21] → Seg A     (2MB)
  Bucket[20] → Seg C     (1MB)

释放 Seg B (3MB):

① 从 SegmentTracker 找到 Seg B

② 左合并: Seg A(2MB,free) 存在 → 合并
   新 Seg B: base=0x10000000, size=5MB, isLeftMost=true
   删除 Seg A

③ 右合并: Seg C(1MB,free) 存在 → 合并
   新 Seg B: base=0x10000000, size=6MB, isRightMost=false (Seg D 在右边)
   删除 Seg C

④ 不是整个 Chunk (isRightMost=false) → 不触发 Lazy Free

⑤ 重新插入 FreeList: Bucket[23] (Log2(6MB)=23)

最终状态:
Segment List:
  Seg B(0x10000000,6MB,free) → Seg D(0x10600000,10MB,busy)

Free List:
  Bucket[23] → Seg B (6MB)
```

---

## 6. ChunkAllocate：创建新 Chunk

```
ChunkAllocate(allocInfo):

① 计算 Chunk 大小:
   chunkSize = allocInfo.size
   if (alignment > chunkAlignment):
       chunkSize += alignment - chunkAlignment   // 对齐填充
   chunkSize = AlignUp(chunkSize, chunkAlignment)  // 对齐到页
   chunkSize = AlignUp(chunkSize, m_ChunkAllocSize) // 对齐到 Chunk 大小

   例如: 请求 3MB, chunkAlignment=64KB, m_ChunkAllocSize=2MB
   → chunkSize = AlignUp(3MB + 63KB, 64KB) = 3.06MB
   → chunkSize = AlignUp(3.06MB, 2MB) = 4MB

② 向 KMD 申请 GPU 内存:
   虚拟内存池: m_pDevice->GetPlatform().CreateMemory(virtCreateInfo, &pChunkMem)
   物理内存池: m_pDevice->CreateMemory(allocCreateInfo, &pChunkMem)
   → 最终调用 m3dDevice->CreateGpuMemory() → DRM ioctl

③ 创建 ResSegment:
   base = pChunkMem->GetDeviceVirtualAddress()
   size = pChunkMem->GetSize()
   new ResSegment(base, size) {
       pChunkMem = pChunkMem,
       chunkBase = base,
       isLeftMost = true,
       isRightMost = true,
       busy = false
   }

④ ResourceAdd:
   - SegmentListInsert: 插入到双向链表头
   - FreeListInsert: 根据 Log2(size) 插入对应 bucket

⑤ 更新统计:
   m_TotalSize += size
   m_FreeSize += size
```

---

## 7. Lazy Free：延迟回收策略

### 7.1 设计原理

当整个 Chunk 的所有段都变为空闲时，不立即归还 GPU 内存。原因是：后续分配很可能还需要同样大小的 Chunk → 此时直接复用，避免重复的 KMD 分配。

### 7.2 触发条件

```cpp
// ResourceRemove:
if (pMemRes->isLeftMost && pMemRes->isRightMost) {
    // ★ 整个 Chunk 完全空闲
    pMemRes->lazyFreeCount++;
    
    if (lazyFreeCount > m_ReuseCountLimit ||      // 复用次数超限
        pMemRes->pChunkMem->GetSize() > m_ReuseSizeLimit) {  // Chunk 太大
        // 销毁 Chunk
        SegmentListRemove(pMemRes);
        pMemRes->pChunkMem->Destroy();  // → KMD 释放
        delete pMemRes;
        return true;
    }
    // 否则保留 Chunk，等待下次分配时直接切片使用
}
return false;
```

### 7.3 配置

```
Core 层 (用户池):
  m_ReuseCountLimit = UINT64_MAX   → 永不因次数超限而释放
  m_ReuseSizeLimit  = UINT64_MAX   → 永不因大小超限而释放

HAL 层:
  s_DefaultLazyFreeThreshold = UINT64_MAX   → 同上的无限策略
```

---

## 8. TrimPool：内存回收

当用户调用 `muMemPoolTrimTo(pool, minBytes)` 时：

```
TrimPool(minBytes):

while (m_TotalSize > minBytes):
    for each Bucket[index] (0..63):
        for each segment in Bucket[index]:
            if (segment->isLeftMost && segment->isRightMost):
                // ★ 整个 Chunk 完全空闲
                FreeListRemove(segment)
                SegmentListRemove(segment)
                
                m_TotalSize -= segment->size
                m_FreeSize  -= segment->size
                
                segment->pChunkMem->Destroy()  // 归还 GPU 内存
                delete segment
                released = true
                break
        if (!released) break  // 无完整空闲 Chunk → 停止
```

---

## 9. 三层 API 完整时序图

```
应用程序      Driver层          Core层                HAL/M3D层           KMD/GPU
  │            │                 │                      │                  │
  │ muMemPool  │                 │                      │                  │
  │ Create()   │                 │                      │                  │
  │───────────>│                 │                      │                  │
  │            │ InitPlatform()  │                      │                  │
  │            │ TlsCtxTop()     │                      │                  │
  │            │                 │                      │                  │
  │            │ new MemoryPool──>│                      │                  │
  │            │ (device)        │                      │                  │
  │            │                 │                      │                  │
  │            │ Init(createInfo)│                      │                  │
  │            │────────────────>│                      │                  │
  │            │                 │ MemMgr::              │                  │
  │            │                 │ CreateUserPool()─────>│                  │
  │            │                 │                      │                  │
  │            │                 │    new Hal::M3d::     │                  │
  │            │                 │    MemoryPool(device) │                  │
  │            │                 │    │                  │                  │
  │            │                 │    │ Init(createInfo):│                  │
  │            │                 │    │  设置 ChunkAlign  │                  │
  │            │                 │    │  设置 ChunkAlloc  │                  │
  │            │                 │    │  设置 ReuseLimit  │                  │
  │            │                 │    │  size==0 → 懒分配│                  │
  │            │                 │    │<─返回             │                  │
  │            │                 │    │                  │                  │
  │            │                 │<───返回 pool 指针──────│                  │
  │            │<──返回 pool ────│                      │                  │
  │<─pool──────│                 │                      │                  │
  │            │                 │                      │                  │
  │═══════════════════════════════════════════════════════════════════════│
  │            │                 │                      │                  │
  │ muMemAlloc │                 │                      │                  │
  │ FromPool   │                 │                      │                  │
  │───────────>│                 │                      │                  │
  │            │ CreateMemory()  │                      │                  │
  │            │────────────────>│                      │                  │
  │            │                 │ InitFromPool(this,   │                  │
  │            │                 │   size)              │                  │
  │            │                 │  └─>Hal::FullAllocate│                  │
  │            │                 │─────────────────────>│                  │
  │            │                 │                      │ ① SubAllocate()  │
  │            │                 │                      │   ──────────┐    │
  │            │                 │                      │   indexLow  │    │
  │            │                 │                      │    =Log2(sz)│    │
  │            │                 │                      │   indexHigh │    │
  │            │                 │                      │   BitMask   │    │
  │            │                 │                      │   ScanFwd   │    │
  │            │                 │                      │   FindBucket│    │
  │            │                 │                      │   ───┐ 在    │    │
  │            │                 │                      │   找到│Bucket │    │
  │            │                 │                      │   空闲│中搜索  │    │
  │            │                 │                      │   段  │      │    │
  │            │                 │                      │   ───┘      │    │
  │            │                 │                      │   ResourceSplit   │
  │            │                 │                      │   ├─ 前部分裂│    │
  │            │                 │                      │   ├─ 后部分裂│    │
  │            │                 │                      │   └─ 插入   │    │
  │            │                 │                      │   SegmentTracker│  │
  │            │                 │                      │   <─────────┘    │
  │            │                 │                      │                  │
  │            │                 │                      │  if (not found): │
  │            │                 │                      │   ② ChunkAllocate│
  │            │                 │                      │    ──────────────│──>│
  │            │                 │                      │    计算 Chunk 大小│  │
  │            │                 │                      │    CreateMemory  │  │
  │            │                 │                      │    (GPU 内存)    │──>│
  │            │                 │                      │    ── DRM ioctl  │  │
  │            │                 │                      │    <──返回───────│<─│
  │            │                 │                      │    创建 ResSegment│ │
  │            │                 │                      │    ResourceAdd   │  │
  │            │                 │                      │    (插入 SegmentList│ │
  │            │                 │                      │     + FreeList)   │  │
  │            │                 │                      │   ③ 再 SubAllocate│ │
  │            │                 │                      │    ──从新Chunk切  │ │
  │            │                 │                      │                  │
  │            │                 │<── *ppMemory+offset──│                  │
  │            │                 │                      │                  │
  │            │ *ptr = VA       │                      │                  │
  │            │ TrackMemory     │                      │                  │
  │            │<────────────────│                      │                  │
  │<─*dptr─────│                 │                      │                  │
  │            │                 │                      │                  │
  │═══════════════════════════════════════════════════════════════════════│
  │            │                 │                      │                  │
  │ muMemFree  │                 │                      │                  │
  │───────────>│                 │                      │                  │
  │            │ DestroyMemory() │                      │                  │
  │            │────────────────>│                      │                  │
  │            │                 │ ~MemoryPool 触发     │                  │
  │            │                 │ ~Memory()            │                  │
  │            │                 │  └─>Hal::Free()─────>│                  │
  │            │                 │                      │ ① SegmentTracker │
  │            │                 │                      │    .find(range)  │
  │            │                 │                      │                  │
  │            │                 │                      │ ② 左邻合并       │
  │            │                 │                      │   if (left free)│
  │            │                 │                      │   → merge left  │
  │            │                 │                      │                  │
  │            │                 │                      │ ③ 右邻合并       │
  │            │                 │                      │   if (right free)│
  │            │                 │                      │   → merge right │
  │            │                 │                      │                  │
  │            │                 │                      │ ④ LazyFree Check │
  │            │                 │                      │   if (entire     │
  │            │                 │                      │    chunk free):  │
  │            │                 │                      │     lazyFreeCount│
  │            │                 │                      │     ++          │
  │            │                 │                      │     if (超限)    │
  │            │                 │                      │       → Destroy │
  │            │                 │                      │     else:       │
  │            │                 │                      │       保留 Chunk │
  │            │                 │                      │                  │
  │            │                 │                      │ ⑤ FreeListInsert │
  │            │                 │                      │    插入到对应    │
  │            │                 │                      │    Bucket       │
  │            │                 │<─────────────────────│                  │
  │            │                 │                      │                  │
  │            │                 │                      │                  │
  │═══════════════════════════════════════════════════════════════════════│
  │            │                 │                      │                  │
  │ muMemPool  │                 │                      │                  │
  │ Destroy()  │                 │                      │                  │
  │───────────>│                 │                      │                  │
  │            │ delete pool─────>│                      │                  │
  │            │                 │ ~MemoryPool()        │                  │
  │            │                 │  ├─ 释放所有          │                  │
  │            │                 │  │  未释放的分配       │                  │
  │            │                 │  └─ MemMgr::         │                  │
  │            │                 │    DestroyUserPool──>│                  │
  │            │                 │                      │ ~MemoryPool()    │
  │            │                 │                      │  遍历所有ResSeg  │
  │            │                 │                      │  释放所有Chunk   │
  │            │                 │                      │  归还 GPU 内存──>│ DestroyGpuMemory
  │<─OK────────│<────────────────│                      │                  │<─OK

```

---

## 关键设计洞察

| 设计 | 实现 | 为什么 |
|------|------|--------|
| **64-Bucket 分级** | Log2(size) 索引 | O(1) 定位到大小合适的 bucket |
| **Bitmap 加速** | m_EltMappingHash (64bit) | O(1) 判断哪些 bucket 非空，避免遍历 64 个 bucket |
| **双链表并存** | SegmentList + FreeList | 地址序链表用于合并(Free)，空闲链表用于分配(SubAllocate) |
| **区间树追踪** | std::map<MemoryRange, ResSegment*> | Free 时 O(log n) 查找已分配段 |
| **Lazy Free** | 延迟销毁空闲 Chunk | 避免频繁 KMD 分配/释放，提高复用率 |
| **递归锁** | std::recursive_mutex | 支持 FullAllocate→SubAllocate→ChunkAllocate→SubAllocate 嵌套 |
| **两阶段分配** | SubAllocate + ChunkAllocate | 先尝试从池中找，找不到才创建新 Chunk |
| **对齐分裂** | 前部+后部分裂 | 严格满足对齐要求，碎片自动回收 |

---

*本文档基于 musa 项目源码分析生成，commit: 9ba99a5d, branch: bugfix/sw-79049*
