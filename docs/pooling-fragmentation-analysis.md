# 池化技术调研：显存池碎片率提升分析

> 项目：MUSA GPU 驱动（linux-ddk）
> 日期：2026-05-18

---

## 一、项目中的池化体系架构

该项目采用**四层池化体系**，层层递进：

| 层级 | 组件 | 核心文件 | 核心职责 |
|------|------|----------|----------|
| **Runtime** | `MemoryPool` | `musa/src/musa/core/memoryPool.cpp` | CUDA-like API 封装、IPC 共享、跨设备 P2P |
| **HAL** | `MemoryPool` (M3D) | `musa/src/hal/m3d/memoryPool.cpp` | **核心池**：分段式空闲链表分配器 |
| **KM-RA** | `RA` (Resource Arena) | `gr-kmd/services/shared/` | 内核态资源竞技场，基于量子的子分配 |
| **KM-RM** | `Range Allocator` | `gr-kmd/mt-rm/common/lib/range_allocator.c` | 底层2的幂次桶空闲范围分配器 |

```
User API (muapiMemPoolCreate, etc.)
    |
    v
MUSA Runtime MemoryPool (musa/src/musa/core/memoryPool.cpp)
    |   - 属性管理 (释放阈值、水位线、复用控制)
    |   - IPC 共享 (POSIX shm)
    |   - 跨设备 P2P 访问管理
    v
HAL IMemMgr (musa/src/hal/halMemMgr.h)
    |
    v
HAL MemMgr (musa/src/hal/m3d/memMgr.cpp)
    |   - Splay Tree 按键查找池 (类型+堆+属性+NUMA)
    |   - 内部池: Default, HostMapped, Host, Profile, Shader
    |   - 用户池、Graph 池
    v
HAL MemoryPool (musa/src/hal/m3d/memoryPool.cpp)  ← 核心分析对象
    |   - 2的幂次分段式空闲链表 (Buddy-like)
    |   - Best-fit / Assured-fit 分配策略
    |   - 分配时分裂 + 释放时合并
    |   - 惰性释放 chunk 回收
    |   - Trim 到阈值
    v
Device Memory (GPU 物理/虚拟内存)

Kernel 侧:
    Resource Allocator (gr-kmd/mt-rm/common/lib/resource_allocator.c)
        |   - 基于 flag 的子分配器分离 (Splay Tree)
        v
    Range Allocator (gr-kmd/mt-rm/common/lib/range_allocator.c)
        |   - 2的幂次桶空闲链表 (RANGE_MAX_ORDER=40)
        |   - Best-fit / Fast-fit 策略
        |   - 分配时分裂 + 释放时合并
```

---

## 二、显存池核心数据结构（HAL 层）

**位置**: `musa/src/hal/m3d/memoryPool.h:144-150`

```cpp
ResSegment* m_FreeBuckets[s_FreeTableLimit];  // 2的幂次桶空闲链表数组 (Buddy-like)
DevSize     m_EltMappingHash;                 // 64位bitmap，标记各桶是否为空 (O(1)查找)
ResSegment* m_pHeadSegment;                   // 全段双向链表 (有序), O(1)邻居合并
MemSegTree  m_SegmentTracker;                 // std::map 区间树, 按 VA 范围索引 busy 段
```

**ResSegment 结构** (`memoryPool.h:49-85`):

```cpp
struct ResSegment {
    IMemory*     pChunkMem;      // 所属 chunk 内存对象
    DevSize      lazyFreeCount;  // 惰性释放计数
    ResSegment*  pNextSegment;   // 全段链表后继
    ResSegment*  pPrevSegment;   // 全段链表前驱
    ResSegment*  pNextFree;      // 空闲链表后继
    ResSegment*  pPrevFree;      // 空闲链表前驱
    bool         isLeftMost;     // 是否为 chunk 最左段
    bool         isRightMost;    // 是否为 chunk 最右段
    bool         busy;           // 是否被占用
    DevSize      base;           // 起始地址
    DevSize      size;           // 段大小
    DevSize      chunkBase;      // chunk 基地址
};
```

**关键常量**:
- `s_FreeTableLimit` = 最大 GPU MMU 位数 (~40，支持 1TB 地址范围)
- `s_DefaultChunkAllocSize` = 2MB (HAL 层默认 chunk 增长粒度)
- Runtime 层默认 chunk = 32MB

---

## 三、提升碎片率的六大机制

### 机制 1：即时合并 (Merge-on-Free) + 相邻段不变量

**位置**: `musa/src/hal/m3d/memoryPool.cpp:228-249`

**最核心的反碎片机制**——释放时立即合并左右相邻空闲段：

```cpp
void MemoryPool::Free(IMemory* pMemory, DevSize base, DevSize size) {
    // 从区间树查找 busy 段
    auto it = m_SegmentTracker.find(vaRange);
    auto pMemRes = it->second;

    // 从 tracker 移除
    m_SegmentTracker.erase(it);
    m_FreeSize += pMemRes->size;

    // 尝试与左邻接合并
    auto pNeighbour = pMemRes->pPrevSegment;
    if (!pMemRes->isLeftMost && !pNeighbour->busy) {
        FreeListRemove(pNeighbour);
        SegmentListRemove(pNeighbour);
        pMemRes->base  = pNeighbour->base;   // 向左扩展基址
        pMemRes->size += pNeighbour->size;   // 吸收邻居大小
        pMemRes->isLeftMost = pNeighbour->isLeftMost;
        delete pNeighbour;
    }

    // 尝试与右邻接合并 (同理)
    pNeighbour = pMemRes->pNextSegment;
    if (!pMemRes->isRightMost && !pNeighbour->busy) {
        // ... 合并逻辑
    }

    // 如果合并后的段是整个chunk，考虑惰性释放
    if (!ResourceRemove(pMemRes)) {
        FreeListInsert(pMemRes);  // 插回空闲链表
    }
}
```

**为什么有效**:

- 全段链表保持地址有序性，释放时邻居查找为 O(1)
- 内核态 `range_allocator.c:71` 运行时断言强制不变量：**相邻两个段不可能同时为 free**
- 这确保空闲空间始终被表示为最大的连续段，彻底消除因碎片产生的小空闲洞

对比无池化场景：每次分配直接向 GPU 驱动申请，释放后空闲空间可能散布在不同物理页中，驱动层面不一定立即合并，导致外部碎片累积。

**内核态对应实现** (`gr-kmd/mt-rm/common/lib/range_allocator.c:260-303`):

```c
static struct range *free_range(struct range_allocator *allocator, struct range *r) {
    // 尝试合并左邻接
    if (!r->left_most && neighbour->is_free) {
        list_del(&neighbour->free_node);
        list_del(&neighbour->node);
        r->start = neighbour->start;
        r->size += neighbour->size;
        r->left_most = neighbour->left_most;
        rm_os_free(neighbour);
    }
    // 尝试合并右邻接 (同理)
}
```

运行时验证 (`range_allocator.c:63-92`):

```c
static int range_allocator_validate(struct range_allocator *allocator) {
    // ...
    // 相邻 range 永远不能同时为 free
    rm_assert(!prev->is_free || !r->is_free);
    // 相邻 range 总是可合并的 (地址连续)
    rm_assert(prev->start + prev->size == r->start);
}
```

---

### 机制 2：最佳适配分配 (Best-Fit Selection)

**位置**: `musa/src/hal/m3d/memoryPool.cpp:101-116`

每次分配时从小桶到大桶扫描，选择**最小能容纳请求的段**：

```cpp
if (m_Policy.selection == SelectPolicy::bestFit) {
    // 从小桶向大桶扫描，找到第一个能容纳的最小段
    uint32_t index = 0, endIndex = 0;
    DevSize optionalMappingHash = ~((1ULL << indexLow) - 1) & m_EltMappingHash;
    if (BitMaskScanForward(&index, optionalMappingHash) &&
        BitMaskScanReverse(&endIndex, optionalMappingHash)) {
        while (index <= endIndex && pMemRes == nullptr) {
            pMemRes = FindBucket(m_FreeBuckets[index], size, alignment, UINT64_MAX);
            index++;
        }
    }
}
```

对比 assuredFit 模式 (`memoryPool.cpp:117-130`):

```cpp
else {
    // assuredFit: 优先从保证能容纳的大桶选 (快速)
    if (index != s_FreeTableLimit) {
        pMemRes = FindBucket(m_FreeBuckets[index], size, alignment, 1);  // 仅取第一个
    } else {
        // 回退：从高桶向低桶扫描
        for (index = indexHigh; index >= indexLow; index--) {
            pMemRes = FindBucket(m_FreeBuckets[index], size, alignment, UINT64_MAX);
        }
    }
}
```

**为什么有效**:

外部碎片率的经典定义：

> **外部碎片率 = 1 - (最大连续空闲块 / 总空闲空间)**

Best-fit 分配通过优先消耗小段来**最大化"最大连续空闲块"的保留概率**，直接降低外部碎片率。相比之下，assuredFit 可能从大段中分配小请求，产生不必要的分裂，长期导致大段被"蚕食"为小段。

---

### 机制 3：2的幂次分桶 + Bitmap 加速

**位置**: `musa/src/hal/m3d/memoryPool.h:144-146` + `memoryPool.cpp:460-458`

空闲段按 `log2(size)` 分入对应桶，用单一 bitmask 加速查找：

```cpp
// 插入空闲链表
void MemoryPool::FreeListInsert(ResSegment* pMemRes) {
    uint32_t index = ::Util::Log2(pMemRes->size);  // 根据 log2(size) 决定桶号
    // 插入对应桶的链表头部
    pMemRes->pNextFree = m_FreeBuckets[index];
    m_FreeBuckets[index] = pMemRes;
    pMemRes->busy = false;
    m_EltMappingHash |= (1ULL << index);  // 标记桶非空
}

// 从空闲链表移除
void MemoryPool::FreeListRemove(ResSegment* pMemRes) {
    uint32_t index = ::Util::Log2(pMemRes->size);
    // 从桶链表移除
    if (!m_FreeBuckets[index]) {
        m_EltMappingHash &= ~(1ULL << index);  // 桶变空，清除 bit
    }
}
```

**搜索加速** (`memoryPool.cpp:101-130`):

```cpp
// 用 BitScanForward 单指令定位第一个非空桶
BitMaskScanForward(&index, optionalMappingHash);
// 直接跳到该桶搜索，跳过所有空桶
```

**为什么有效**:

- 每个桶内的段大小在 `[2^i, 2^(i+1))` 范围内，查找时最多遍历 ~40 个桶 (而非遍历所有空闲段)
- `m_EltMappingHash` 用单条 `BitScanForward` 指令即可定位第一个非空桶，近似 O(1)
- 对比经典 Buddy 系统：不强制 2 的幂次对齐分割，允许任意大小段存在，更灵活

---

### 机制 4：分配时分裂 (Split-on-Allocate)

**位置**: `musa/src/hal/m3d/memoryPool.cpp:358-413`

当空闲段大于请求大小时，精确分裂为三部分：

```cpp
Result MemoryPool::ResourceSplit(ResSegment* pMemRes, DevSize size,
                                  DevSize alignment, DevSize* pBase) {
    FreeListRemove(pMemRes);  // 先从空闲链移除

    if (!m_Policy.noSplit) {
        // 1. 对齐前导填充 → 形成独立 free 段
        if (alignedBase > pMemRes->base) {
            ResSegment* pNeighbour = new ResSegment(alignedBase,
                                                     pMemRes->size - splitSize);
            SegmentListInsertAfter(pMemRes, pNeighbour);
            pMemRes->size = splitSize;
            FreeListInsert(pMemRes);  // 前导填充插回空闲链 (可被后续小请求复用)
            pMemRes = pNeighbour;
        }

        // 2. 尾部剩余 → 形成独立 free 段
        if (pMemRes->size > size) {
            ResSegment* pNeighbour = new ResSegment(pMemRes->base + size,
                                                     pMemRes->size - size);
            SegmentListInsertAfter(pMemRes, pNeighbour);
            pMemRes->size = size;
            FreeListInsert(pNeighbour);  // 尾部剩余插回空闲链
        }
    }

    // 3. 将分配段插入 busy tracker
    MemoryRange vaRange{alignedBase, size};
    m_SegmentTracker.insert({vaRange, pMemRes});
    *pBase = alignedBase;
}
```

**为什么有效**:

- **消除内部碎片**：精确分配，不多占
- **对齐填充可复用**：对齐要求产生的前导填充作为独立 free 段，可被后续小请求使用
- 无池化场景下，对齐浪费通常不可回收

---

### 机制 5：最优插入排序 (Optimal Insertion)

**位置**: `musa/src/hal/m3d/memoryPool.cpp:420-446`

当 `insertion=optimal` 时，free 段按大小升序插入桶内：

```cpp
void MemoryPool::FreeListInsert(ResSegment* pMemRes) {
    uint32_t index = ::Util::Log2(pMemRes->size);

    if (m_Policy.insertion == InsertPolicy::optimal) {
        // 在桶内找到 "刚好比 pMemRes 大" 的位置插入 (保持升序)
        ResSegment* pMemResPrev = nullptr;
        while (pMemResTmp && pMemResTmp->size < pMemRes->size) {
            pMemResPrev = pMemResTmp;
            pMemResTmp = pMemResTmp->pNextFree;
        }
        // 插入到 pMemResPrev 和 pMemResTmp 之间
        pMemRes->pNextFree = pMemResTmp;
        pMemRes->pPrevFree = pMemResPrev;
        // ...
    } else {
        // fast 模式：直接插入桶链表头部
        pMemRes->pNextFree = m_FreeBuckets[index];
        m_FreeBuckets[index] = pMemRes;
    }
}
```

**为什么有效**:

配合 `bestFit` 选择策略，桶内链表有序时 `FindBucket` 遍历遇到第一个满足的即是最小的，可以**提前终止搜索**。这进一步保证每次分配选的是最小适配段，最大化大段保留概率。

---

### 机制 6：惰性释放 + 主动裁剪 (Lazy-Free + Trim)

**位置**: `musa/src/hal/m3d/memoryPool.cpp:318-332`

Chunk 完全空闲时不立即销毁，而是记数并判断是否达到回收阈值：

```cpp
bool MemoryPool::ResourceRemove(ResSegment* pMemRes) {
    if (pMemRes->isLeftMost && pMemRes->isRightMost) {
        pMemRes->lazyFreeCount++;
        // 超过复用次数阈值 或 超过大小阈值 → 真正销毁
        if (pMemRes->lazyFreeCount > m_ReuseCountLimit ||
            pMemRes->pChunkMem->GetSize() > m_ReuseSizeLimit) {
            SegmentListRemove(pMemRes);
            pMemRes->pChunkMem->Destroy();
            delete pMemRes;
            return true;  // 已销毁
        }
    }
    return false;  // 保留暂不销毁
}
```

**主动裁剪** (`memoryPool.cpp:480-510`):

```cpp
void MemoryPool::TrimPool(DevSize value) {
    while (m_TotalSize > value) {
        bool released = false;
        for (uint32_t index = 0; index < s_FreeTableLimit && !released; ++index) {
            ResSegment* pCurrent = m_FreeBuckets[index];
            while (pCurrent) {
                if (pCurrent->isLeftMost && pCurrent->isRightMost) {
                    // 完全空闲的 chunk → 释放回 GPU
                    FreeListRemove(pCurrent);
                    SegmentListRemove(pCurrent);
                    m_TotalSize -= pCurrent->size;
                    m_FreeSize  -= pCurrent->size;
                    pCurrent->pChunkMem->Destroy();
                    delete pCurrent;
                    released = true;
                    break;
                }
                pCurrent = pCurrent->pNextFree;
            }
        }
        if (!released) break;  // 没有可释放的 chunk 了
    }
}
```

**为什么有效**:

- **防止 chunk 抖动**：避免 "分配-释放-再分配" 循环中反复创建/销毁 chunk，chunk 被保留供快速复用
- **Trim API 提供主动控制**：上层可在内存压力下调 `TrimTo` 释放空闲显存
- **ReleaseThreshold**：配合水位线追踪，自动或按需收缩

---

## 四、内核态 RM Range Allocator

**位置**: `gr-kmd/mt-rm/common/lib/range_allocator.c`

与 HAL 层完全一致的合并、分裂、分桶策略，外加：

### 4.1 导入乘数 (import_multiplier)

`resource_allocator.c:220` — 按需从外部导入时一次导入 `size * multiplier`:

```c
import_size = roundup(size, align) * import_multiplier;
segment = allocator->import_alloc(allocator->import_handle, import_size, flags);
```

摊销导入开销，同时预留在同一 chunk 内分配后续请求的趋势。

### 4.2 活跃范围查询 (Live Range Query)

`range_allocator.c:487-531` — 快速定位所有 busy 段的边界范围：

```c
int range_allocator_query_live_range(struct range_allocator *allocator,
                                     uint64_t *p_start, uint64_t *p_size) {
    // 找到最左和最右的 busy 段，计算覆盖范围
}
```

可用于备份/迁移场景，也便于评估碎片分布。

### 4.3 设计验证断言

`range_allocator.c:63-92` — 运行时不变式检查：

```c
// 相邻段不能同时为 free （强制合并保证）
rm_assert(!prev->is_free || !r->is_free);
// 相邻段地址必须连续
rm_assert(prev->start + prev->size == r->start);
// 相邻段必须来自同一来源
rm_assert(prev->priv == r->priv);
```

---

## 五、RA (Resource Arena)

**位置**: `gr-kmd/services/shared/include/ra.h`

RA 是内核态的资源竞技场，策略标志定义 (`ra.h:107-159`):

| 策略 | 值 | 含义 |
|------|-----|------|
| `RA_POLICY_ALLOC_FAST` | 0 | 选取第一个满足的节点 |
| `RA_POLICY_ALLOC_OPTIMAL` | 1 | 选取最小满足节点 → **减少碎片** |
| `RA_POLICY_BUCKET_ASSURED_FIT` | 0 | 保证能容纳的桶 |
| `RA_POLICY_BUCKET_BEST_FIT` | 4 | 最佳适配桶 → 减少碎片，响应时间可变 |
| `RA_POLICY_NO_SPLIT` | 8 | 不分裂，整个 import 直接分配 |

---

## 六、策略组合与适用场景

**位置**: `musa/src/hal/halMemoryPool.h:36-58`

| InsertPolicy | SelectPolicy | 碎片抵抗力 | 分配速度 | 适用场景 |
|-------------|-------------|-----------|---------|---------|
| `fast` | `assuredFit` | 中等 | **最快** | 默认通用场景 |
| `optimal` | `bestFit` | **最强** | 较慢 | 对碎片敏感、大块分配频繁的关键路径 |

---

## 七、碎片率量化评估

### 外部碎片率定义

> **外部碎片率 = 1 - (最大连续空闲块大小 / 总空闲空间大小)**

池化技术从以下维度降低该比率：

| 维度 | 无池化 (每次独立分配) | 有池化 (MemoryPool) | 改善机制 |
|------|----------------------|---------------------|----------|
| **外部碎片** | 高 — 分配/释放交替导致空闲块被人为拆分 | 低 — 即时合并确保空闲段始终最大化 | 机制1, 2, 5 |
| **内部碎片** | 高 — 对齐浪费不可回收 | 低 — 对齐填充作为独立 free 段可复用 | 机制4 |
| **分配抖动** | 每次分配/释放都走驱动层 | Chunk 级惰性复用，减少驱动调用 | 机制6 |
| **大块分配成功率** | 随运行时间下降 (碎片累积) | 稳定 (best-fit 保护大段) | 机制2, 3 |
| **空闲显存可回收性** | 被动等待 GC/无控制 | 主动 TrimTo / ReleaseThreshold | 机制6 |

### 池化开销

| 开销项 | 说明 |
|--------|------|
| 内存开销 | 每个 ResSegment 约 ~128 bytes；每个 chunk 保留在池中不释放 |
| CPU 开销 | bestFit + optimal 策略下，free 插入为 O(n_bucket)，alloc 遍历为 O(n_buckets)；fast 策略接近 O(1) |
| 锁竞争 | `std::recursive_mutex` 保护整个池，高并发下可能成为瓶颈 |

---

## 八、总结

该项目显存池通过**五管齐下**的策略将碎片化空间持续合并为最大连续段：

```
 分配时分裂 (Split-on-Alloc)   →  消除内部碎片，对齐填充可复用
          +
 释放时合并 (Merge-on-Free)    →  消除外部碎片，空闲空间最大化
          +
 2的幂次分桶 (Power-of-2)      →  快速定位合适大小段 (O(log_range))
          +
 最佳适配选择 (Best-Fit)        →  保护大段，提高大块分配成功率
          +
 惰性 chunk 复用 (Lazy-Free)   →  防止分配抖动，Trim 按需回收
```

这五层机制从 HAL 用户态到内核 RM 层统一实现，贯穿整个 GPU 驱动栈，确保在长时间运行、频繁分配释放的场景下，显存碎片率保持在低位，大块显存分配维持高成功率。
