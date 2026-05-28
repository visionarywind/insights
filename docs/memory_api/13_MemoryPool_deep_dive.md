# MemoryPool 深度分析 — 子分配算法与执行示例

> 基于 `musa/src/hal/m3d/memoryPool.h` 和 `memoryPool.cpp` 的完整代码分析。

---

## 一、核心数据结构

### 1.1 ResSegment（内存资源段）

```
每个 ResSegment 描述一段连续的内存区间 [base, base+size)

chunkBase ──────────────────┐
                            ▼
[======整个 chunk (比如 2MB)=============================================]
  │                        │
  ├── base=0x0, size=4KB   ├── base=0x1000, size=8KB
  │   busy=true            │   busy=false (空闲)
  │   isLeftMost=true      │   isLeftMost=false
  │   isRightMost=false    │   isRightMost=true
  │                        │
  pChunkMem → 底层 Hal::IMemory (同一个 chunk)
```

关键字段：
- **busy**：true=已分配 / false=空闲
- **isLeftMost / isRightMost**：标识该段是否是 chunk 的最左/最右边界（用于判断是否可以释放整个 chunk）
- **pChunkMem**：所有子段共享同一个底层 KMD 分配对象
- **chunkBase**：chunk 的起始地址，用于计算返回给上层的 offset
- **pNextFree / pPrevFree**：空闲链表（按 size 排序）
- **pNextSegment / pPrevSegment**：按地址顺序串联所有段的双向链表

### 1.2 FreeBuckets（空闲哈希桶）

```
m_FreeBuckets[0]  → size=1 字节的空闲段链表 (log2(1)=0)
m_FreeBuckets[1]  → size=2 字节的空闲段链表
...
m_FreeBuckets[12] → size=4KB 的空闲段链表
...
m_FreeBuckets[21] → size=2MB 的空闲段链表
...
m_FreeBuckets[47] → size=最大 的空闲段链表
        ↑
        s_FreeTableLimit (通常 48 或 52)
```

**索引规则**：`index = floor(log2(segment_size))`，下标即桶号。

**m_EltMappingHash**：位图，bit i=1 表示 `m_FreeBuckets[i]` 非空。
```
例如 m_EltMappingHash = 0b...101100
                       bit:  543210
                       → 桶2(4B)、桶3(8B)、桶5(32B) 有空闲段
```

### 1.3 SegmentTracker

```cpp
std::map<MemoryRange, ResSegment*> m_SegmentTracker;
// key = {basePointer, endPointer}
// 仅虚拟内存池(physical=false)使用，用于精确查找要释放的段
```

### 1.4 Policy（分配策略）

```cpp
struct Policy {
    InsertPolicy insertion;   // fast（头插LIFO） 或 optimal（按size有序插入）
    SelectPolicy selection;   // assuredFit（最快找到即停） 或 bestFit（遍历找最紧的）
    bool noSplit;             // true=不分割空闲段（直接整段分配）
};
// 默认: {fast, assuredFit, false}
```

---

## 二、核心算法详解

### 2.1 `FindBucket` — 在桶内查找满足条件的空闲段

```cpp
// memoryPool.cpp:14-40
static ResSegment* FindBucket(ResSegment* first,
                              DevSize size,
                              DevSize alignment,
                              DevSize tryLimit) {
    ResSegment *walker, *answer = nullptr;
    for (walker = first; walker && tryLimit != 0; walker = walker->pNextFree) {
        DevSize alignedBase = (alignment > 1)
            ? Util::AlignUp(walker->base, alignment)
            : walker->base;
        if (walker->base + walker->size >= alignedBase + size) {
            answer = walker;   // 找到了！
            break;
        }
        if (tryLimit != UINT64_MAX) tryLimit--;
    }
    return answer;
}
```

**tryLimit 的作用**：
- `assuredFit` 模式：`tryLimit = 1` → 每个桶只看链表头的第一个段
- `bestFit` 模式：`tryLimit = UINT64_MAX` → 遍历整个链表找最合适的

**对齐处理**：
```
空闲段: [base=0x1A00, size=16KB]
请求: size=4KB, alignment=4KB(0x1000)
alignedBase = AlignUp(0x1A00, 0x1000) = 0x2000
检查: 0x1A00 + 0x4000 >= 0x2000 + 0x1000 → 0x5A00 >= 0x3000 ✅ 满足
     返回 offset=0x600 (对齐后剩余的前间隙)
```

---

### 2.2 `SubAllocate` — 子分配主逻辑

```
                    ┌─────────────────────────────────┐
                    │     SubAllocate(4KB 请求)        │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │ Step 1: 计算桶范围               │
                    │ indexLow = log2(4096) = 12       │
                    │ indexHigh = log2(4096+对齐间隙)  │
                    └──────────────┬──────────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │  assuredFit (默认)                       │ bestFit
              │                                         │
              │ ① 检查 m_EltMappingHash                 │ ① 计算有效位图
              │    bit 12..max 是否有1?                  │    ~((1<<12)-1) & hash
              │                                         │
              │ ② 从 indexHigh 桶开始向下扫描           │ ② 从最低非空桶到indexHigh
              │    FindBucket(bucket[i], tryLimit=1)     │    FindBucket(bucket[i], UINT64_MAX)
              │    不满足 → i-- 继续                     │    找第一个满足的即返回
              │                                         │
              │ ③ 找到即返回，否则 errorNotFound         │
              └─────────────────────────────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │ Step 2: ResourceSplit(找到的段)  │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │ Step 3: 更新统计                  │
                    │ m_FreeSize -= requested_size     │
                    │ 返回 (pChunkMem, chunk内offset)  │
                    └─────────────────────────────────┘
```

---

### 2.3 `ResourceSplit` — 段分割算法

```cpp
// memoryPool.cpp:358-413
```

**输入**：一个空闲段 + 请求大小 + 对齐要求
**输出**：分割后的 busy 段（分配出去）+ 剩余的 free 段

**三种分割场景**：

#### 场景 A：仅后面分裂
```
请求: size=4KB, alignment=0
空闲段: [base=0, size=64KB]

处理:
  alignedBase = 0 (无需对齐调整)
  pMemRes->base = 0, 已对齐
  pMemRes->size(64KB) > requested(4KB) → 需要分裂

结果:
  [0, 4KB)     → busy (返回给客户端)
  [4KB, 64KB)  → free (右邻居)
```

#### 场景 B：前面因对齐产生间隙 + 后面也有多余
```
请求: size=4KB, alignment=64KB(0x10000)
空闲段: [base=0x1000, size=128KB]

处理:
  alignedBase = AlignUp(0x1000, 64K) = 0x10000
  alignedBase(0x10000) > base(0x1000) → 需要前分裂
    → 创建左邻居 [0x1000, 0xF000) = 60KB (free)
    → pMemRes 移动为 [0x10000, 128KB-60KB=68KB)
  pMemRes->size(68KB) > requested(4KB) → 需要后分裂
    → 创建右邻居 [0x11000, 64KB) (free)
    → pMemRes = [0x10000, 4KB) → busy (返回)

结果:
  [0x1000, 0x10000)  → free (前间隙, 已插入空闲表)
  [0x10000, 0x11000) → busy (返回给客户端)
  [0x11000, 0x20000) → free (后多余, 已插入空闲表)
```

#### 场景 C：完全匹配（不分裂）
```
请求: size=4KB, alignment=0
空闲段: [base=0, size=4KB]

处理:
  alignedBase = 0
  size 完全匹配 → 不分裂

结果:
  [0, 4KB) → busy (返回给客户端)
  (无剩余空闲段)
```

---

### 2.4 `ChunkAllocate` — 向 KMD 申请新 chunk

```cpp
// memoryPool.cpp:153-212
```

**Size 对齐规则**（从用户请求到实际分配）：
```
用户请求: 4096 字节

Step 1: 补偿对齐间隙
  if (userAlignment > chunkAlignment)
    chunkSize += userAlignment - chunkAlignment
  例: chunkAlignment=2MB, userAlignment=0 → chunkSize=4096

Step 2: 对齐到 chunkAlignment
  chunkSize = AlignUp(4096, 2MB) = 2MB

Step 3: 对齐到 m_ChunkAllocSize (默认也是 2MB)
  chunkSize = AlignUp(2MB, 2MB) = 2MB

最终: 分配 2MB (512 个 4KB 页)
```

**分配目标**：
- 虚拟内存池 → `Platform::CreateMemory(memoryTypeVirtual, ...)` → OS `mmap`/`VirtualAlloc`
- 物理内存池 → `Device::CreateMemory(memoryTypeAlloc, ...)` → **完整的 6 层 KMD 调用链**

---

### 2.5 `Free` — 释放 + 合并

```
状态: [A:free 4KB][B:busy 8KB][C:free 4KB][D:busy 4KB][E:free 16KB]

释放 B (8KB):
  1. 从 SegmentTracker 找到 B 的段
  2. 左邻居 A 空闲? ✅ → 合并 AC = [A+C: 8KB free]
  3. 右邻居 C 空闲? ✅ → 合并 AC = [A+B+C: 12KB free]
  4. D 是 busy, 停止右合并
  5. 新段 [A+B+C: 12KB free] 插入空闲表

释放 D (4KB):
  1. 左邻居 AC 空闲? ✅ → 合并 ACD = [16KB free]
  2. 右邻居 E 空闲? ✅ → 合并 ACD+E = [32KB free]
  3. ACD+E 覆盖整个 chunk → isLeftMost && isRightMost
  4. ResourceRemove → 整个 chunk 被销毁
```

---

## 三、完整执行示例

### 示例 1：连续分配 3 次 4KB，然后释放

**初始状态**：空 pool，等待首次分配。

**第 1 次 `muMemAlloc(&d0, 4096)`**：
```
FullAllocate(4096)
  → SubAllocate: m_EltMappingHash=0, 无空闲段 → errorNotFound
  → ChunkAllocate(4096):
     chunkSize = AlignUp(AlignUp(4096, 2MB), 2MB) = 2MB
     Device::CreateMemory(2MB) → KMD ioctl ★ (第1次)
     创建 ResSegment{base=0, size=2MB}
     ResourceAdd → 加入空闲表
  → Retry SubAllocate:
     FindBucket(2MB桶, 4096, 0, 1) → 找到
     ResourceSplit: [0,4KB)busy | [4KB,2MB)free
  → 返回: d0 = HalVA + 0
```

```
内存状态:
[========== 2MB chunk ==========]
[4KB busy | 2093056B free]
 ^d0
```

**第 2 次 `muMemAlloc(&d1, 8192)`**：
```
FullAllocate(8192)
  → SubAllocate:
     indexLow = log2(8192) = 13
     indexHigh = log2(8192) = 13
     optionalMappingHash = ~((1<<14)-1) & m_EltMappingHash
       m_EltMappingHash = (1<<21) (只有桶21有段)
       ~((1<<14)-1) 保留bit14以上的位
       → optionalMappingHash = (1<<21)
     BitMaskScanForward → index=21 (最低非空桶)
     FindBucket(m_FreeBuckets[21], 8192, 0, 1)
     → m_FreeBuckets[21] 链表中第一个段 [4KB, 2MB) 大小=2093056
     → 2093056 >= 8192 ✅ tryLimit=1 找到一个即停

  → ResourceSplit([4KB,2MB), 8192):
    alignedBase = AlignUp(4096, 0) = 4096 (alignment=0)
    pMemRes->base(4096) == alignedBase(4096) → 无需前分裂
    pMemRes->size(2093056) > 8192 → 后分裂
      创建右邻居 [4096+8192, 2MB) = [12288, 2MB) free
      pMemRes = [4096, 12288) → busy

  → 返回: d1 = HalVA + 4096
```

```
内存状态:
[========== 2MB chunk ==========]
[4KB busy | 8KB busy | 2080768B free]
 ^d0       ^d1

**第 3 次 `muMemAlloc(&d2, 65536)` (64KB, 带 alignment=64KB)**：
```
FullAllocate(65536, alignment=65536)
  → SubAllocate:
     indexLow = log2(65536) = 16
     indexHigh = log2(65536 + 65536 - 1) = 17
     optionalMappingHash = ~((1<<18)-1) & (1<<21) = 0x200000
     BitMaskScanForward → index=21
     FindBucket(m_FreeBuckets[21], 65536, 65536, 1)
     → 空闲段 [12288, 2MB):
        alignedBase = AlignUp(12288, 65536) = 65536
        检查: 12288 + 2080768 >= 65536 + 65536 → 2093056 >= 131072 ✅
     → 找到!
     
  → ResourceSplit([12288, 2080768), 65536, 65536):
    alignedBase = 65536
    前分裂: 65536 - 12288 = 53248 字节间隙 → 创建 [12288, 65536) free
    pMemRes 移动为 [65536, 剩余)
    后分裂: size = 2080768 - 53248 = 2027520, > 65536
            → 创建 [131072, 2MB) free
            pMemRes = [65536, 131072) → busy
            
  → 返回: d2 = HalVA + 65536
```

```
内存状态:
[============== 2MB chunk ==============]
[4KB][ 8KB ][48KB free][64KB busy][~1955KB free]
^d0  ^d1              ^d2
```

**`muMemFree(d0, 4096)`**：
```
Pool::Free(base=0, size=4096)
  → SegmentTracker 找到 [0, 4096)
  → 标记为空闲
  → 左邻居: 无 (isLeftMost=true)
  → 右邻居 [4096, 12288) busy → 不合并
  → FreeListInsert([0, 4096))
```

```
内存状态:
[4KBFREE][8KB busy][48KB free][64KB busy][~1955KB free]
```

**`muMemFree(d1, 8192)`**：
```
Pool::Free(base=4096, size=8192)
  → SegmentTracker 找到 [4096, 12288)
  → 标记为空闲
  → 左邻居 [0, 4096) 空闲! → 合并为 [0, 12288)
  → 右邻居 [12288, 65536) 空闲! → 再合并为 [0, 65536)
  → 前后空闲段全部加入空闲表
```

```
内存状态:
[~80KB free              ][64KB busy][~1955KB free]
  [0, 65536) 合并后                   
```

---

### 示例 2：多 chunk 场景

**场景**：pool 大小 2MB，先消耗完第一个 chunk，触发 ChunkAllocate。

```
初始: 1 个 chunk [0, 2MB) 全部空闲

分配序列（每次 1MB，alignment=0）:

Allocate(1MB):
  SubAllocate → 找到 [0, 2MB) → 分裂
    [0, 1MB) busy | [1MB, 2MB) free
  返回 d0

Allocate(1MB):
  SubAllocate → 找到 [1MB, 2MB) → 分裂
    [1MB, 2MB) busy | 空闲 = 0
  返回 d1

Allocate(1MB):
  SubAllocate → 无空闲段 → errorNotFound
  ChunkAllocate(1MB):
    chunkSize = AlignUp(AlignUp(1MB, 2MB), 2MB) = 2MB
    Device::CreateMemory(2MB) → KMD ioctl ★
    新 chunk [2MB, 4MB)
  Retry SubAllocate → 找到 [2MB, 4MB)
    [2MB, 3MB) busy | [3MB, 4MB) free
  返回 d2
```

```
最终状态:
[==== chunk 0: 2MB ====][==== chunk 1: 2MB ====]
[1MB][1MB][empty]       [1MB][1MB free]
 d0   d1                d2
```

---

### 示例 3：释放触发 chunk 销毁（lazy free）

```
假设: reuseCountLimit=3, reuseSizeLimit=4MB

释放 d0 (1MB):
  [0, 1MB) 空闲
  不是左且右 (chunk 还有 d1)
  → 仅加入空闲表, lazyFreeCount=0

释放 d1 (1MB):
  [1MB, 2MB) 空闲
  合并 → [0, 2MB) 整个 chunk 空闲
  isLeftMost && isRightMost → ResourceRemove
  lazyFreeCount=1 < 3 → 不销毁

释放 d2 (1MB):
  第二个 chunk [2MB, 3MB) 空闲
  不是整个 chunk → 加入空闲表
  lazyFreeCount 保持（只对完整 chunk 计数）
```

---

### 示例 4：bestFit vs assuredFit 对比

```
空闲表状态:
  桶[12](4KB): [segA: 4KB][segB: 4KB]
  桶[13](8KB): [segC: 8KB]
  桶[21](2MB): [segD: 2MB]

请求: size=4096, alignment=0

assuredFit:
  indexHigh = log2(4096) = 12
  检查桶12: FindBucket(桶12头, 4096, 0, 1) → segA ✅ 满足
  返回 segA，总耗时: 1 次桶查找 + 1 次链表遍历

bestFit:
  optionalMappingHash = ~((1<<12)-1) & hash = hash (所有>=4KB的桶)
  扫描从最低位(bit12)开始:
    桶12: FindBucket(segA, 4096, 0, UINT64_MAX) → segA 满足
  返回 segA (恰好也是 best fit)

请求: size=5000, alignment=0

assuredFit:
  indexHigh = log2(5000) = 13 (向上取整)
  检查桶13: FindBucket(segC, 5000, 0, 1) → 8KB >= 5000 ✅
  返回 segC (8KB 满足但不是最优)

bestFit:
  扫描桶12: FindBucket(桶12头, 5000, 0, UINT64_MAX)
    segA: 4KB < 5000 ❌
    segB: 4KB < 5000 ❌
  扫描桶13: FindBucket(segC, 5000, 0, UINT64_MAX)
    segC: 8KB >= 5000 ✅ → 返回 segC (best fit)
```

---

## 四、关键设计总结

### 4.1 为什么需要 MemoryPool？

```
裸 KMD 分配:
  每次 muMemAlloc → CreateGpuMemory → AllocBuffer(ioctl) + MapVirtualAddress(ioctl)
  1000 次 4KB 分配 = 2000 次 ioctl ≈ 200ms (每次 ~100μs)

Pool 子分配:
  首次: ChunkAllocate(2MB ioctl) + MapVirtualAddress(ioctl) = 2 次 ioctl
  后续 511 次: SubAllocate(纯链表操作) = 0 次 ioctl
  1000 次 4KB 分配 ≈ 4 次 ioctl ≈ 0.4ms
  
性能提升: ~500x
```

### 4.2 关键参数一览

| 参数 | 默认值 | 来源 | 含义 |
|------|--------|------|------|
| `s_DefaultChunkAllocSize` | 2MB | memoryPool.h:101 | `ChunkAllocate` 的最小分配单位 |
| `s_DefaultLazyFreeThreshold` | `UINT64_MAX` | memoryPool.h:102 | chunk 释放前的最大复用次数 |
| `s_FreeTableLimit` | 48/52 | memoryPool.h:99 | 空闲桶数量（= MMU 位数） |
| 默认 SelectPolicy | `assuredFit` | memoryPool.cpp:54 | 最快找到即停 |
| 默认 InsertPolicy | `fast` | memoryPool.cpp:54 | 空闲表头插 |
| `m_ChunkAllocSize` | 2MB | 初始化时传入 | pool 扩容时的 chunk 大小 |

### 4.3 分配路径总览

```
muMemAlloc(4KB)
  │
  ├─ [SubAllocatable=true] → MemMgr::Allocate
  │                          └─ 查找/创建匹配的 Pool
  │                          └─ Pool::FullAllocate
  │                             ├─ SubAllocate (O(1) ~ O(log N))
  │                             │   └─ FindBucket → ResourceSplit
  │                             ├─ [失败] ChunkAllocate
  │                             │   └─ KMD ioctl (2 次: Alloc + Map)
  │                             └─ Retry SubAllocate
  │
  └─ [SubAllocatable=false] → Hal::Device::CreateMemory
                              └─ 每次完整 KMD 调用链 (2+ 次 ioctl)
```

---

## 五、文件索引

| 文件 | 行数 | 说明 |
|------|------|------|
| `musa/src/hal/m3d/memoryPool.h` | 157 | 数据结构与接口声明 |
| `musa/src/hal/m3d/memoryPool.cpp` | 533 | 核心实现（SubAllocate/FullAllocate/Free/ResourceSplit/ChunkAllocate/TrimPool） |
| `musa/src/hal/m3d/memMgr.cpp:81-147` | 67 | MemMgr::Allocate（pool 选择与入口） |
| `musa/src/hal/m3d/memory.cpp:366-426` | 61 | InitGeneralDeviceMemory（pool 内部调用的 KMD 入口） |
| `musa/doc/memory_api/11_GeneralAlloc_deep_dive.md` | — | 上层 GeneralAlloc 链路 |
| `musa/doc/memory_api/12_DirectKMD_Allocation_flow.md` | — | 裸 KMD 分配路径 |
| `musa/doc/memory_api/13_MemoryPool_deep_dive.md` | — | 本文档 |