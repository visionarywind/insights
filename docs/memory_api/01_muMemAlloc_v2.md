# muMemAlloc_v2 — 设备内存分配完整流程分析

> `muMemAlloc_v2` 对应 CUDA `cudaMalloc`，是 MUSA Runtime 中调用频率最高的内存分配 API。本文档从用户调用到 KMD ioctl，对每一层进行完整的逐行源码拆解。

---

## 1. 功能概述

在 GPU 设备上分配一段线性设备内存，返回 `MUdeviceptr`（设备虚拟地址）。内部分 **Pool Sub-Allocation**（默认）和 **裸 KMD 分配**两条路径，前者约 500x 性能提升。

### 全部变体

| API | 文件:行号 | 说明 |
|-----|----------|------|
| `muMemAlloc_v2` | `mu_wrappers_generated.cpp:10177` → `mu_memory.cpp:265` | v2 入口（64 位 dptr） |
| `muMemAlloc` | `mu_memory.cpp:299` | v1 兼容接口，直接委托 v2 |
| `muMemAllocAsync` | `mu_memory.cpp:303` | 异步分配（走 `memoryTypeVirtual` + bind） |
| `muMemAllocPitch_v2` | `mu_memory.cpp:330` | 带 pitch 的 2D/3D 分配（委托 `PitchedGeneralAlloc`） |

---

## 2. 关键数据结构

### MemoryCreateInfo

```cpp
// 定义位置: musa/core/internal_types.h (或 musa.h 中)
struct MemoryCreateInfo {
    memoryType type;  // memoryTypeGeneral / PitchedGeneral / PinnedHost / Managed / IpcImport / Virtual ...
    union {
        struct { size_t size; size_t alignment; unsigned int flags; } general;
        struct { MUipcMemHandle handle; unsigned int flags; } ipcImport;
        struct { void* ptr; size_t size; unsigned int flags; } registeredPinnedHost;
        // ... 其他类型
    };
};
```

### HAL 层 MemoryAllocInfo

```cpp
// 定义位置: hal/halMemory.h
struct MemoryAllocInfo {
    MemoryAllocType type;           // DeviceLocal / Host
    MemoryHeap      heap;           // largePage (2MB) / general
    uint32_t        property;       // 位或组合 (见下文 property 表)
    uint32_t        viewCapability; // Exportable / PeerAccessible / IpcExportable
    DevSize         size;
    DevSize         alignment;
    int             numaId;
};
```

### HAL Property Flags 参考表

| Flag (十六进制) | 含义 | 来源 |
|:---:|---|---|
| `0x001` | `Virtual` — 创建虚拟地址映射 | Driver 层硬编码 |
| `0x002` | `DeviceMapped` — 在 Device 侧可见 | Driver 层硬编码 |
| `0x004` | `SubAllocatable` — 允许从 Pool 子分配 | Driver 层硬编码 |
| `0x008` | `Physical` — 分配物理内存页 | Core: `GeneralAlloc` 自动添加 |
| `0x010` | `HostVisible` — CPU 可访问 | Core: 由 Virtual 推导 |
| `0x020` | `DeviceVisible` — GPU 可访问 | Core: 由 Virtual 推导 |
| `0x040` | `HostCoherent` — CPU 端缓存一致性 | Core: 由 Virtual 推导 |
| `0x080` | `DeviceWriteable` — GPU 可写 | Core: 由 Virtual 推导 |
| `0x100` | `DeviceCached` — GPU 端使用缓存 | Core: 由 Virtual 推导 |
| `0x200` | `SharedVirtualAddress` — GPU/CPU 共享 VA | Core: 始终添加 |
| `0x3FF` | **最终总和** | Virtual × DeviceMapped × SubAlloc 时 |

| ViewCapFlag | 含义 | 来源 |
|:---:|---|---|
| `0x01` | `Exportable` — 可导出为外部句柄 | Core: 始终添加 |
| `0x02` | `PeerAccessible` — 可建立 Peer 映射 | Core: 由 DeviceMapped 推导 |
| `0x04` | `IpcExportable` — 可 IPC 导出 | Core: 由 DeviceMapped 推导 |
| `0x07` | **最终总和** | DeviceMapped 时 |

---

## 3. 完整调用链（六层逐行）

```
═══════════════════════════════════════════════════════════════════════════════
  第 0 层：用户层
═══════════════════════════════════════════════════════════════════════════════
  muMemAlloc_v2(&dptr, bytesize)
    │
    ▼
═══════════════════════════════════════════════════════════════════════════════
  第 1 层：Wrapper (mu_wrappers_generated.cpp:10177-10215)
═══════════════════════════════════════════════════════════════════════════════
  MUresult MUSAAPI muMemAlloc_v2(MUdeviceptr* dptr, size_t bytesize)
    │
    ├─ ApiTrace trace(...)  // MUPTI 记录 API 调用
    │
    ├─ if (toolsCallbackEnabled)  // 第三方工具回调路径
    │    ├─ 获取 correlationId = ThreadInfo::Get().ApiSeqNum()
    │    ├─ TlsCtxTop() → context → contextId
    │    ├─ 构造 MUtoolsTraceApiMusa inParams
    │    ├─ toolsIssueCallback(API_ENTER, ...)  // 通知进入
    │    ├─ if (!skipDriverImpl)
    │    │     status = muapiMemAlloc_v2(params.dptr, params.bytesize)
    │    └─ toolsIssueCallback(API_EXIT, ...)   // 通知退出
    │
    └─ else  // 普通路径（无工具）
         status = muapiMemAlloc_v2(dptr, bytesize)
    │
    ▼
═══════════════════════════════════════════════════════════════════════════════
  第 2 层：Driver (mu_memory.cpp:265-297)
═══════════════════════════════════════════════════════════════════════════════
  MUresult muapiMemAlloc_v2(MUdeviceptr *dptr, size_t bytesize)
    │
    ├─ (2.1) InitPlatform()
    │    │    ★ 懒初始化，首次调用时完成平台初始化
    │    │    包括: 加载驱动 → 枚举设备 → 创建默认 Context
    │    │    后续调用直接返回 MUSA_SUCCESS (快速路径)
    │
    ├─ (2.2) 参数校验
    │    ├─ dptr == nullptr → MUSA_ERROR_INVALID_VALUE
    │    │  bytesize == 0
    │    │    ├─ *dptr = 0
    │    │    └─ MUSA_ERROR_INVALID_VALUE  // CUDA 对齐: size=0 返回错误
    │    │
    │    └─ pContext = TlsCtxTop()
    │       └─ nullptr → MUSA_ERROR_INVALID_CONTEXT
    │
    ├─ (2.3) 构造 MemoryCreateInfo
    │    ├─ createInfo.type  = memoryTypeGeneral
    │    ├─ createInfo.general.size = bytesize
    │    ├─ createInfo.general.alignment = 0   // 不指定对齐
    │    └─ createInfo.general.flags  = 0x0007
    │         = memoryPropertyVirtual (0x0001)
    │         | memoryPropertyDeviceMapped (0x0002)
    │         | memoryPropertySubAllocatable (0x0004)
    │         ★ 三个 flag 均由 Driver 层硬编码，用户不可控制
    │
    └─ (2.4) pContext->CreateMemory(&pMemory, createInfo)
    │    └─ 详见第 3 层
    │
    └─ (2.5) *dptr = pMemory->GetDevicePointer()
         = m_pHalMemory->GetDeviceVirtualAddress() + m_Offset
         ★ 地址合成: 基地址(HAL VA) + 子分配偏移
    │
    ▼
═══════════════════════════════════════════════════════════════════════════════
  第 3 层：Core — Context (context.cpp:915-965)
═══════════════════════════════════════════════════════════════════════════════
  Context::CreateMemory(IMemory** ppMemory, const MemoryCreateInfo& createInfo)
    │
    ├─ (3.1) Stream Capture 状态检查
    │    ReadLockedAccessor -> 遍历所有 Streams
    │    for each stream in ctxCrit->Streams():
    │      if (captureStatus == ACTIVE && NeedStrictlyCheck()
    │          || captureStatus == INVALIDATED)
    │        → MUSA_ERROR_STREAM_CAPTURE_UNSUPPORTED
    │        → InvalidateCaptureStatus()  // 标记所有 capture 为 invalid
    │        ★ CUDA 对齐: 有 stream 处于 capture 时不允许分配
    │
    ├─ (3.2) 创建 Memory 对象
    │    memory_sp = std::make_shared<Memory>(this)
    │    │  构造函数: m_pHalMemory=nullptr, m_Offset=0, m_pPool=nullptr
    │    │  (详见第 4 层)
    │    └─ pMemory->Init(createInfo)
    │       │  设置 m_Type = createInfo.type
    │       └─ switch(m_Type):
    │            case memoryTypeGeneral:
    │              → GeneralAlloc(size, alignment, flags)
    │              ★ 详见第 4 层
    │
    ├─ (3.3) Peer 映射
    │    WriteLockedAccessor -> MapToPeers(pMemory)
    │    │  条件: pMemory->Hal()->GetCapability() & PeerAccessible
    │    │  ★ DeviceMapped=YES 时 viewCap 包含 PeerAccessible，进入此路径
    │    │
    │    │  对每个 peer device:
    │    │    Hal::PeerOpenInfo{ kernelManagedGlobal, deviceMapped=true,
    │    │                        deviceMapType=m_Peers[peer] }
    │    │    → pMemory->Hal()->OpenPeerMemory(peerHal, openInfo)
    │    │    ★ 打开 Peer 设备上的内存映射，使其他 GPU 可访问
    │    │
    │    └─ if (status == MUSA_SUCCESS) AddMemory(pMemory)
    │         → ctxCrit->m_Memories.insert(pMemory)
    │         ★ 将 Memory 加入 Context 内部集合（用于析构时遍历清理）
    │
    ├─ (3.4) memset 清零 (仅 Trace Capture 构建)
    │    #if M3D_BUILD_MT_TRACE_CAPTURE:
    │    if (type != RegisteredPinnedHost && type != IpcImport)
    │      std::memset(pMemory->GetHostPointer(), 0, pMemory->GetSize())
    │    ★ 排除类型: 已注册的主机指针、IPC 导入的内存不重置
    │
    ├─ (3.5) 注册到全局 MemoryTracker
    │    Platform::Get().GetMemoryTracker().TrackMemory(memory_sp)
    │    │  (详见第 5 层)
    │    │
    │    └─ if (!pMemory->IsPhysical())
    │         Platform::Get().SetMemorySeqID(pMemory)
    │         ★ 给非物理内存分配序列号（Virtual / Managed 等）
    │
    └─ (3.6) *ppMemory = pMemory
    │
    ▼
═══════════════════════════════════════════════════════════════════════════════
  第 4 层：Core — Memory (memory.cpp:343-497)
═══════════════════════════════════════════════════════════════════════════════
  Memory::GeneralAlloc(size, alignment, flags)
    │
    ├─ (4.1) 设置 shape + flags
    │    m_Shape = { size, 1, 1, size }
    │    m_Flags = flags (=0x0007 from Driver)
    │
    ├─ (4.2) 构造 HAL MemoryCreateInfo
    │    createInfo.alloc.type     = deviceLocal       // GPU 显存
    │    createInfo.alloc.heap     = largePage         // 2MB 大页
    │    createInfo.alloc.property = flags (0x0007)    // 继承 Driver 的 flags
    │
    ├─ (4.3) ★ property 推导链（核心逻辑）
    │    │
    │    │ 步骤 A: 始终添加的基础属性
    │    │   property |= Physical (0x0008)
    │    │   property |= SharedVirtualAddress (0x0200)
    │    │   → property = 0x0007 | 0x0008 | 0x0200 = 0x020F
    │    │
    │    │ 步骤 B: 如果 Virtual flag 置位 (0x0001 & 0x0007 = YES)
    │    │   property |= DeviceVisible   (0x0020)
    │    │   property |= HostVisible     (0x0010)
    │    │   property |= HostCoherent    (0x0040)
    │    │   property |= DeviceWriteable (0x0080)
    │    │   property |= DeviceCached    (0x0100)
    │    │   → property = 0x020F | 0x01F0 = 0x03FF
    │    │
    │    │ 步骤 C: viewCapability
    │    │   viewCap = memoryViewCapabilityExportable (0x01)
    │    │   如果 DeviceMapped flag 置位 (0x0002 & 0x0007 = YES)
    │    │     viewCap |= PeerAccessible  (0x02)
    │    │     viewCap |= IpcExportable   (0x04)
    │    │   → viewCap = 0x07
    │    │
    │    │ ★ 最终: property=0x03FF, viewCapability=0x07
    │    │   0x03FF = Virtual|DeviceMapped|SubAlloc|Physical|HostVisible|
    │    │           DeviceVisible|HostCoherent|DeviceWriteable|DeviceCached|
    │    │           SharedVA
    │    │   0x07   = Exportable|PeerAccessible|IpcExportable
    │    │
    │    └─ allocInfo.alignment = max(alignment, device.memAllocAlignment)
    │
    ├─ (4.4) ★ 分配路径分支
    │    │
    │    │  if (property & SubAllocatable)  // 0x0004 & 0x03FF = YES
    │    │    → GetMemMgr()->Allocate(allocInfo, &m_Offset, &m_pHalMemory)
    │    │      ★ Sub-Allocation 路径（默认，详见第 6 层）
    │    │
    │    │  else  // SubAllocatable 未置位（仅有 !Virtual 或用户显式禁用时）
    │    │    → Device::CreateMemory(createInfo, &m_pHalMemory)
    │    │      ★ 裸 KMD 分配路径
    │    │
    │    └─ 返回 status
    │
    ▼
═══════════════════════════════════════════════════════════════════════════════
  第 5 层：MemoryTracker (memoryTracker.cpp)
═══════════════════════════════════════════════════════════════════════════════
  MemoryTracker::TrackMemory(shared_ptr<IMemory> memory)
    │
    ├─ 获取 deviceVA = pMemory->GetDevicePointer() (= HalVA + m_Offset)
    ├─ 获取 size = pMemory->GetSize()
    │
    ├─ 构造 VA Range: Interval(deviceVA, size)
    │
    └─ 插入到全局 map（有序）
         m_TrackedMemory[VA_start] = {VA_end, shared_ptr<IMemory>}
         ★ 后续所有指针查询 API 均通过此 map 进行 O(log n) 查找
    │
    ▼
═══════════════════════════════════════════════════════════════════════════════
  第 6 层：HAL — MemMgr & MemoryPool (memMgr.cpp + memoryPool.cpp)
═══════════════════════════════════════════════════════════════════════════════
  MemMgr::Allocate(allocInfo, &offset, &ppMemory)   // memMgr.cpp:81-146
    │
    ├─ (6.1) 合法性校验
    │    ├─ size==0 → errorInvalidValue
    │    ├─ size+alignment 溢出 → errorOutOfMemory
    │    └─ deviceLocal && size > totalGlobalMem → errorOutOfMemory
    │
    ├─ (6.2) 获取或创建 Pool
    │    │
    │    │ 构造 MemoryPoolInfo key:
    │    │   poolInfo.type     = allocInfo.type      (deviceLocal)
    │    │   poolInfo.heap     = allocInfo.heap      (largePage)
    │    │   poolInfo.property = allocInfo.property  (0x03FF)
    │    │   poolInfo.viewCap  = allocInfo.viewCap   (0x07)
    │    │   poolInfo.numaId   = allocInfo.numaId    (NUMA_NO_NODE)
    │    │
    │    │ pPool = m_PoolRefs.Get(MakeKey(poolInfo))->m_Value
    │    │   ★ 在 SplayTree 中按 hash key 查找已有 pool
    │    │
    │    │ if (pPool 不存在 || 属性不匹配)
    │    │   → CreatePoolNoLock(poolCreateInfo, &pPool)
    │    │     ★ 创建新 MemoryPool（首次分配时）
    │    │     ★ poolCreateInfo.minEnlargeChunkSize = s_DefaultChunkAllocSize (2MB)
    │    │     ★ poolCreateInfo.reuseCountLimit = s_DefaultLazyFreeThreshold
    │    │
    │    └─ 返回 pPool
    │
    ├─ (6.3) pPool->FullAllocate(allocInfo, &offset, &ppMemory)  // ★ 核心
    │    │                                   // memoryPool.cpp:82-95
    │    ├─ Round 1: SubAllocate(allocInfo, offset, ppMemory)
    │    │    │
    │    │    ├─ 计算桶索引:
    │    │    │   indexLow  = Log2(size)
    │    │    │   indexHigh = Log2(size + alignment - 1)
    │    │    │
    │    │    ├─ 位图查找:
    │    │    │   optionalMappingHash = m_EltMappingHash >> indexHigh
    │    │    │   ★ 用位图快速定位哪个自由桶中有合适大小的碎片
    │    │    │
    │    │    ├─ FindBucket(m_FreeBuckets[index], size, alignment, tryLimit)
    │    │    │   ★ 遍历桶中的 ResSegment 链表
    │    │    │   ★ 首次找到满足 (base+size >= alignedBase+requestSize) 的段即返回
    │    │    │   ★ O(1) 近似查找
    │    │    │
    │    │    ├─ if (找到 free segment)
    │    │    │   → ResourceSplit(pMemRes, size, alignment, &subAllocAddr)
    │    │    │     ★ 切分: 对齐前间隙插入 free list + 分配区标记 busy + 多余后段插入 free
    │    │    │     ★ 更新 m_FreeSize -= allocInfo.size
    │    │    │     ★ *pOffset = subAllocAddr - chunkBase
    │    │    │     ★ *ppMemory = pMemRes->pChunkMem
    │    │    │
    │    │    └─ if (没找到 free segment)
    │    │         → 返回 errorNotFound
    │    │
    │    ├─ Round 2: if (Round 1 == errorNotFound)
    │    │    ChunkAllocate(allocInfo)   // 向 KMD 申请新 chunk
    │    │    │
    │    │    ├─ 计算 chunk size:
    │    │    │   chunkSize = max(requestSize, s_DefaultChunkAllocSize (2MB))
    │    │    │   chunkSize = AlignUp(chunkSize, chunkAlignment)
    │    │    │
    │    │    ├─ 创建底层内存:
    │    │    │   if (物理内存) → m_pDevice->CreateMemory(chunkCreateInfo, &pChunkMem)
    │    │    │     ★ 调用 KMD: mtgpuBoAlloc + mtgpuBoVmMapV2 (2 次 ioctl)
    │    │    │   if (虚拟内存) → platform.CreateMemory(chunkCreateInfo, &pChunkMem)
    │    │    │
    │    │    ├─ 创建 ResSegment:
    │    │    │   pMemRes = new ResSegment(base, chunkSize)
    │    │    │   pMemRes->chunkBase = base
    │    │    │   pMemRes->pChunkMem = pChunkMem
    │    │    │
    │    │    ├─ ResourceAdd(pMemRes)
    │    │    │   → SegmentListInsert (双向链表头插)
    │    │    │   → FreeListInsert (加入自由桶)
    │    │    │
    │    │    └─ m_TotalSize += chunkSize
    │    │       m_FreeSize += chunkSize
    │    │
    │    └─ if (Round 2 == success) → goto Round 1 (重试 SubAllocate)
    │
    └─ 返回 *offset (chunk 内偏移), *ppMemory (chunk 的 HAL Memory 对象)
    │
    ▼
═══════════════════════════════════════════════════════════════════════════════
  KMD 层 (Linux DRM / Windows WDDM)
═══════════════════════════════════════════════════════════════════════════════
  ChunkAllocate → m_pDevice->CreateMemory()
    │
    ├─ mtgpuBoAlloc (ioctl #1)
    │   → 在 GPU 显存中分配 buffer 对象
    │   → 返回 GEM handle
    │
    └─ mtgpuBoVmMapV2 (ioctl #2)
        → 将 buffer 映射到进程的 GPU 虚拟地址空间
        → 返回 GPU VA
```

---

## 4. 完整 ASCII 时序图

### 4.1 首次分配（Pool 尚未创建）

```
应用层       Wrapper(MPUTI)  Driver(mu_memory)   Context           Memory             MemMgr           Pool            Device(HAL)        KMD
  │               │               │                  │                  │                  │                │                  │                │
  │ muMemAlloc_v2 │               │                  │                  │                  │                │                  │                │
  │──────────────>│               │                  │                  │                  │                │                  │                │
  │               │ ApiTrace      │                  │                  │                  │                │                  │                │
  │               │ toolsCallback?│                  │                  │                  │                │                  │                │
  │               │ muapiMemAlloc │                  │                  │                  │                │                  │                │
  │               │──────────────>│                  │                  │                  │                │                  │                │
  │               │               │ InitPlatform()    │                  │                  │                │                  │                │
  │               │               │ (首次: 加载驱动+枚举设备)                │                  │                │                  │                │
  │               │               │                  │                  │                  │                │                  │                │
  │               │               │ TlsCtxTop()      │                  │                  │                │                  │                │
  │               │               │ → 线程局部 Context │                  │                  │                │                  │                │
  │               │               │                  │                  │                  │                │                  │                │
  │               │               │ createInfo:      │                  │                  │                │                  │                │
  │               │               │  type=General    │                  │                  │                │                  │                │
  │               │               │  size=bytesize   │                  │                  │                │                  │                │
  │               │               │  flags=0x7       │                  │                  │                │                  │                │
  │               │               │                  │                  │                  │                │                  │                │
  │               │               │─────────────────>│                  │                  │                │                  │                │
  │               │               │                  │ CreateMemory()   │                  │                │                  │                │
  │               │               │                  │ ① 检查所有Stream │                  │                │                  │                │
  │               │               │                  │   capture 状态   │                  │                │                  │                │
  │               │               │                  │ ② new Memory()  │                  │                │                  │                │
  │               │               │                  │ ③ Init(createInfo)│                 │                │                  │                │
  │               │               │                  │    │             │                  │                │                  │                │
  │               │               │                  │    │ switch(type):│                  │                │                  │                │
  │               │               │                  │    │  General    │                  │                │                  │                │
  │               │               │                  │    ▼             │                  │                │                  │                │
  │               │               │                  │  GeneralAlloc()  │                  │                │                  │                │
  │               │               │                  │    │             │                  │                │                  │                │
  │               │               │                  │    ├─ property推导  │                │                │                  │                │
  │               │               │                  │    │  0x7→0x3FF  │                  │                │                  │                │
  │               │               │                  │    │  viewCap=0x07│                  │                │                  │                │
  │               │               │                  │    │             │                  │                │                  │                │
  │               │               │                  │    ├─ heap=LP    │                  │                │                  │                │
  │               │               │                  │    │  type=DL    │                  │                │                  │                │
  │               │               │                  │    │             │                  │                │                  │                │
  │               │               │                  │    └─ SubAlloc? │                  │                │                  │                │
  │               │               │                  │     YES → MemMgr::  │               │                │                  │                │
  │               │               │                  │      Allocate() │─────────────────>│                │                  │                │
  │               │               │                  │                │  ① 校验 size    │                │                  │                │
  │               │               │                  │                │  ② SplayTree    │                │                  │                │
  │               │               │                  │                │   查找 pool     │                │                  │                │
  │               │               │                  │                │  ③ pool为空!   │                │                  │                │
  │               │               │                  │                │  ④ CreatePool  │                │                  │                │
  │               │               │                  │                │    ───────────────────────────────────────────────>│                  │                │
  │               │               │                  │                │                 │                │   ⑤ Pool::Init  │                  │                │
  │               │               │                  │                │                 │                │   ChunkAlloc   │                  │                │
  │               │               │                  │                │                 │                │    CreateMemory│                  │                │
  │               │               │                  │                │                 │                │    (2MB chunk) │                  │                │
  │               │               │                  │                │                 │                │    ────────────┼──► mtgpuBoAlloc │                │
  │               │               │                  │                │                 │                │    ←───────────┼── ioctl ★ (1)  │                │
  │               │               │                  │                │                 │                │    ────────────┼──► mtgpuVmMapV2 │                │
  │               │               │                  │                │                 │                │    ←───────────┼── ioctl ★ (2)  │                │
  │               │               │                  │                │                 │                │    ResourceAdd │                  │                │
  │               │               │                  │                │                 │                │    (加入free链) │                  │                │
  │               │               │                  │                │                 │                │    m_TotalSize │                  │                │
  │               │               │                  │                │                 │                │     += 2MB     │                │                │
  │               │               │                  │                │                 │                │    Pool: 2MB   │                │                │
  │               │               │                  │                │                 │                │    free pool   │                │                │
  │               │               │                  │                │                 │  ←─────────────│                  │                │
  │               │               │                  │                │  ⑥ FullAllocate │                │                  │                │
  │               │               │                  │                │    ────────────────────────────────────────>│                  │                │
  │               │               │                  │                │                 │                │  SubAllocate   │                  │                │
  │               │               │                  │                │                 │                │  ① 查找桶     │                  │                │
  │               │               │                  │                │                 │                │  ② FindBucket │                  │                │
  │               │               │                  │                │                 │                │    找到2MB段  │                  │                │
  │               │               │                  │                │                 │                │  ③ ResourceSplit│                │                │
  │               │               │                  │                │                 │                │    切出请求大小│                  │                │
  │               │               │                  │                │                 │                │    间隙→free   │                  │                │
  │               │               │                  │                │                 │                │    多余→free   │                  │                │
  │               │               │                  │                │                 │                │  m_FreeSize-=  │                  │                │
  │               │               │                  │                │  ← offset, pChunkMem──────────────────────│                  │                │
  │               │               │                  │  ← m_Offset=m_pHalMemory                                 │                  │                │
  │               │               │                  │ ④ MapToPeers()  │                  │                │                  │                │
  │               │               │                  │   遍历所有peer  │                  │                │                  │                │
  │               │               │                  │   → OpenPeerMem │                  │                │                  │                │
  │               │               │                  │ ⑤ AddMemory()  │                  │                │                  │                │
  │               │               │                  │   加入 m_Memories│                  │                │                  │                │
  │               │               │                  │ ⑥ #if Trace    │                  │                │                  │                │
  │               │               │                  │   memset(ptr,0,size)             │                │                  │                │
  │               │               │                  │ ⑦ TrackMemory()│                  │                │                  │                │
  │               │               │                  │   MemoryTracker │                  │                │                  │                │
  │               │               │                  │   .insert(VA→Mem)                 │                │                  │                │
  │               │               │                  │                 │                  │                │                  │                │
  │               │               │◄─────────────────│                 │                  │                │                  │                │
  │               │               │  *dptr = HalVA + m_Offset                            │                │                  │                │
  │               │               │  = GetDevicePointer()                                │                │                  │                │
  │               │               │                  │                  │                  │                │                  │                │
  │               │◄──────────────│                  │                  │                  │                │                  │                │
  │◄──────────────│               │                  │                  │                  │                │                  │                │
  │  返回 dptr     │               │                  │                  │                  │                │                  │                │
```

### 4.2 后续分配（Pool 命中，零 ioctl）

```
应用层       Driver           Context          Memory           MemMgr           Pool
  │               │               │                 │                 │                │
  │ muMemAlloc_v2 │               │                 │                 │                │
  │──────────────>│               │                 │                 │                │
  │               │ ──→ CreateMemory ──────────────→│                 │                │
  │               │               │                 │ GeneralAlloc    │                │
  │               │               │                 │  → MemMgr::     │                │
  │               │               │                 │    Allocate()   │                │
  │               │               │                 │    ────────────→│                │
  │               │               │                 │                 │ SplayTree      │
  │               │               │                 │                 │ 查找 命中!     │
  │               │               │                 │                 │ 已有 pool      │
  │               │               │                 │                 │ FullAllocate() │
  │               │               │                 │                 │────────────────>│
  │               │               │                 │                 │                │ SubAllocate()
  │               │               │                 │                 │                │ ① Log2(size)
  │               │               │                 │                 │                │ ② 位图查找桶
  │               │               │                 │                 │                │ ③ FindBucket
  │               │               │                 │                 │                │    O(1) 命中!
  │               │               │                 │                 │                │ ④ ResourceSplit
  │               │               │                 │                 │                │    (零 ioctl!)
  │               │               │                 │                 │ ← offset       │
  │               │               │                 │ ← m_Offset      │                │
  │               │               │                 │ = offset        │                │
  │               │ ← *dptr       │                 │                 │                │
  │◄──────────────│               │                 │                 │                │
  │  返回 dptr     │               │                 │                 │                │
```

### 4.3 裸 KMD 分配路径（SubAllocatable flag 未置位时）

```
Driver           Context          Memory           Device(HAL)        KMD
  │               │                 │                 │                  │
  │ ──→ CreateMemory               │                 │                  │
  │               │                 │                 │                  │
  │               │  GeneralAlloc   │                 │                  │
  │               │  property 不含   │                 │                  │
  │               │  SubAllocatable  │                 │                  │
  │               │  → else 分支     │                 │                  │
  │               │    Device::      │                 │                  │
  │               │    CreateMemory  │─────────────────>│                  │
  │               │                 │                 │ mtgpuBoAlloc     │
  │               │                 │                 │ ────────────────>│
  │               │                 │                 │ ← ioctl (1)      │
  │               │                 │                 │ mtgpuVmMapV2     │
  │               │                 │                 │ ────────────────>│
  │               │                 │                 │ ← ioctl (2)      │
  │               │ ← m_pHalMemory  │                 │                  │
  │               │ (m_Offset = 0)  │                 │                  │
  │ ← *dptr = HalVA+0               │                 │                  │
  │  返回 dptr                      │                 │                  │
```

---

## 5. 关键源码逐行分析

### 5.1 Driver 层入口 (`mu_memory.cpp:265-297`)

```cpp
MUresult muapiMemAlloc_v2(MUdeviceptr *dptr, size_t bytesize) {
    MUresult status = InitPlatform();                     // ① 懒初始化
                                                         //   首次调用: CreatePlatform → Platform::Init
                                                         //      → 加载驱动 → 枚举设备 → 创建默认 Context
                                                         //   后续调用: 直接返回 MUSA_SUCCESS

    if (status == MUSA_SUCCESS) {
        if (nullptr == dptr) {
            status = MUSA_ERROR_INVALID_VALUE;            // ② dptr 空指针校验
        } else if (0 == bytesize) {
            *dptr = 0;                                   // ③ size=0 校验 (CUDA 对齐)
            status = MUSA_ERROR_INVALID_VALUE;            //    *dptr 将被清零，但返回错误
        } else {
            Musa::IContext* pContext = TlsCtxTop();       // ④ 获取线程局部 Context
            if (nullptr == pContext)  {                   //   线程未绑定 Context → 错误
                status = MUSA_ERROR_INVALID_CONTEXT;
            } else {
                Musa::MemoryCreateInfo createInfo{};      // ⑤ 构造创建信息
                createInfo.type = Musa::memoryTypeGeneral;//   类型: 通用显存
                createInfo.general.size = bytesize;       //   大小: 用户请求
                createInfo.general.alignment = 0;         //   对齐: 由底层决定
                createInfo.general.flags =                //   ★ flags=0x7:
                    Hal::memoryPropertyVirtual |          //     0x0001: 创建虚拟地址映射
                    Hal::memoryPropertyDeviceMapped |     //     0x0002: 设备端可访问
                    Hal::memoryPropertySubAllocatable;    //     0x0004: 允许 Sub-Allocation
                                                         //   ★ 三个 flag 均由 Driver 硬编码
                                                         //   ★ 用户无法通过 API 控制

                Musa::IMemory* pMemory;
                status = pContext->CreateMemory(          // ⑥ 进入 Core 层
                    &pMemory, createInfo);

                if (status == MUSA_SUCCESS) {
                    *dptr = pMemory->GetDevicePointer();   // ⑦ 合成设备指针
                                                           //    = m_pHalMemory->GetDeviceVA() + m_Offset
                                                           //    Sub-Alloc 时: offset = chunk内偏移
                                                           //    裸 KMD 时: offset = 0
                }
            }
        }
    }

    return status;
}
```

### 5.2 Context::CreateMemory (`context.cpp:915-965`)

```cpp
MUresult Context::CreateMemory(IMemory** ppMemory, const MemoryCreateInfo& createInfo) {
    MUresult status = MUSA_SUCCESS;

    // ★ 流捕获保护: 遍历所有 Stream，检查 capture 状态
    {
        ReadLockedAccessor<CriticalBase> ctxCrit(m_CriticalData);
        for (auto streamI = ctxCrit->Streams().begin();
             streamI != ctxCrit->Streams().end(); streamI++) {
            if (((*streamI)->GetCaptureStatus() == MU_STREAM_CAPTURE_STATUS_ACTIVE &&
                 (*streamI)->NeedStrictlyCheck()) ||
                (*streamI)->GetCaptureStatus() == MU_STREAM_CAPTURE_STATUS_INVALIDATED) {
                status = MUSA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
                (*streamI)->InvalidateCaptureStatus();      // 标记所有 active → invalidated
                break;
            }
        }
    }
    // ★ CUDA 对齐: 有 stream 处于 capture 时不允许分配内存
    //   因为捕获期间分配的内存无法被正确回放

    std::shared_ptr<IMemory> memory_sp;
    Memory* pMemory = nullptr;
    if (status == MUSA_SUCCESS) {
        memory_sp = std::make_shared<Memory>(this);        // 创建 Memory 对象
        pMemory = static_cast<Memory*>(memory_sp.get());
        status = pMemory->Init(createInfo);                // ★ 初始化 (核心)
    }

    if (status == MUSA_SUCCESS) {
        WriteLockedAccessor<CriticalBase> ctxCrit(m_CriticalData);
        if (pMemory->Hal()->GetCapability() &
            Hal::memoryViewCapabilityPeerAccessible) {
            status = ctxCrit->MapToPeers(pMemory);          // ★ 自动 Peer 映射
            // 对每个已启用 PeerAccess 的 device:
            //   Hal::PeerOpenInfo{ kernelManagedGlobal, deviceMapped=true, mapType }
            //   → pMemory->Hal()->OpenPeerMemory(peerDeviceHal, openInfo)
            //   → 在每个 peer device 上创建对该 memory 的访问映射
        }
        if (status == MUSA_SUCCESS) {
            ctxCrit->AddMemory(pMemory);                   // 加入 Context 集合 (析构用)
        }
    }

    if (status == MUSA_SUCCESS) {
        // Trace Capture 构建: 清零设备内存 (排除 IpcImport/RegisteredPinnedHost)
#if M3D_BUILD_MT_TRACE_CAPTURE
        if (createInfo.type != Musa::memoryTypeRegisteredPinnedHost &&
            createInfo.type != Musa::memoryTypeIpcImport) {
            std::memset(pMemory->GetHostPointer(), 0, pMemory->GetSize());
        }
#endif
        Platform::Get().GetMemoryTracker().TrackMemory(memory_sp);
        // ★ 将 (deviceVA, size) → shared_ptr<IMemory> 注册到全局 MemoryTracker
        //   后续所有 FindRange / FindByPointer 查询依赖此映射

        if (!pMemory->IsPhysical()) {
            Platform::Get().SetMemorySeqID(pMemory);       // 给非物理内存分配序列号
        }
        *ppMemory = pMemory;
    }

    return status;
}
```

### 5.3 Memory::GeneralAlloc (`memory.cpp:462-497`)

```cpp
MUresult Memory::GeneralAlloc(size_t size, size_t alignment, unsigned int flags) {
    MUresult status = MUSA_SUCCESS;
    m_Shape = { size, 1, 1, size };                        // shape: width×height×depth, pitch
    m_Flags = flags;                                       // 保存 flags

    Hal::MemoryCreateInfo createInfo{};
    createInfo.type = Hal::memoryTypeAlloc;                // ★ 类型: 分配型内存
    createInfo.alloc.type = Hal::memoryAllocTypeDeviceLocal;// ★ 设备本地显存
    createInfo.alloc.heap = Hal::MemoryHeap::largePage;     // ★ 堆: 2MB 大页
    createInfo.alloc.property = flags;                     // property 初始 = 0x0007

    // ★★★ property 推导链 ★★★
    // 步骤 1: 强制添加的基础属性
    createInfo.alloc.property |= Hal::memoryPropertyPhysical |
                                 Hal::memoryPropertySharedVirtualAddress;
    // property = 0x0007 | 0x0008 | 0x0200 = 0x020F

    // 步骤 2: 如果 Virtual flag 置位 (Driver 始终置位)
    createInfo.alloc.property |= flags & Hal::memoryPropertyVirtual ?
        Hal::memoryPropertyDeviceVisible |
        Hal::memoryPropertyHostVisible |
        Hal::memoryPropertyHostCoherent |
        Hal::memoryPropertyDeviceWriteable |
        Hal::memoryPropertyDeviceCached :
        Hal::memoryPropertyNone;
    // property = 0x020F | 0x01F0 = 0x03FF

    // 步骤 3: viewCapability 推导
    createInfo.alloc.viewCapability = Hal::memoryViewCapabilityExportable;
    createInfo.alloc.viewCapability |= (flags & Hal::memoryPropertyDeviceMapped) ?
        (Hal::memoryViewCapabilityPeerAccessible |
         Hal::memoryViewCapabilityIpcExportable) :
        Hal::memoryViewCapabilityNone;
    // viewCap = 0x01 | 0x06 = 0x07

    // 对齐: 至少满足设备的 memAllocAlignment
    createInfo.alloc.size = size;
    createInfo.alloc.alignment = std::max(alignment,
        static_cast<size_t>(m_Context->GetParentDevice()->Hal()
            .GetProperties().memoryProperties.memAllocAlignment));

    createInfo.alloc.numaId = NUMA_NO_NODE;

    // ★ 路径分支 ★
    if (createInfo.alloc.property & Hal::memoryPropertySubAllocatable) {
        // Sub-Allocation 路径 (默认)
        status = HalToMuResult(
            m_Context->GetParentDevice()->Hal().GetMemMgr()->Allocate(
                createInfo.alloc, &m_Offset, &m_pHalMemory));
    } else {
        // 裸 KMD 分配路径
        status = HalToMuResult(
            m_Context->GetParentDevice()->Hal().CreateMemory(
                createInfo, &m_pHalMemory));
    }
    return status;
}
```

### 5.4 MemMgr::Allocate (`memMgr.cpp:81-146`)

```cpp
Result MemMgr::Allocate(const MemoryAllocInfo& allocInfo,
                        DevSize* pOffset, IMemory** ppMemory,
                        IMemoryPool** ppMemoryPool) {
    Result res = Result::success;
    IMemoryPool* pPool;

    // ★ 合法性检查
    if (allocInfo.size == 0) return errorInvalidValue;
    if (allocInfo.alignment > 0 &&
        allocInfo.size + allocInfo.alignment - 1 < allocInfo.size)
        return errorOutOfMemory;  // 整数溢出

    if (allocInfo.type == memoryAllocTypeDeviceLocal &&
        allocInfo.size > m_pDevice->GetProperties().memoryProperties.totalGlobalMem)
        return errorOutOfMemory;  // 超过显存总量

    if (res == Result::success) {
        // ★ 用户指定 pool? (m_pPool != nullptr 时走 InitFromPool 路径, 否则 nullptr)
        // muMemAlloc_v2 走 else 分支
        MemoryPoolInfo poolInfo{};
        poolInfo.type           = allocInfo.type;
        poolInfo.heap           = allocInfo.heap;
        poolInfo.property       = allocInfo.property;
        poolInfo.viewCapability = allocInfo.viewCapability;
        poolInfo.numaId         = allocInfo.numaId;

        std::lock_guard<std::mutex> lg(m_Lock);
        pPool = m_PoolRefs.Get(MakeKey(poolInfo))->m_Value;
        // ★ 在 SplayTree 中按 property 哈希查找已有 pool
        //   首次分配时 pool 为 nullptr

        if (!pPool || 属性不匹配) {
            // ★ 创建新 MemoryPool
            MemoryPoolCreateInfo poolCreateInfo{};
            poolCreateInfo.info               = poolInfo;
            poolCreateInfo.usageFlags.userManaged = false;
            poolCreateInfo.usageFlags.internal    = false;
            poolCreateInfo.minEnlargeChunkSize = MemoryPool::s_DefaultChunkAllocSize; // 2MB
            poolCreateInfo.reuseCountLimit      = MemoryPool::s_DefaultLazyFreeThreshold;
            poolCreateInfo.reuseSizeLimit       = MemoryPool::s_DefaultChunkAllocSize; // 2MB
            poolCreateInfo.size                 = allocInfo.size;
            poolCreateInfo.alignment            = allocInfo.alignment;

            res = CreatePoolNoLock(poolCreateInfo, &pPool);
        }
    }

    if (res == Result::success) {
        // ★ 调用 pool 的 FullAllocate
        res = pPool->FullAllocate(allocInfo, pOffset, ppMemory);
    }
    return res;
}
```

### 5.5 MemoryPool::FullAllocate → SubAllocate → ChunkAllocate

```cpp
// ★ 双回合机制
Result MemoryPool::FullAllocate(const MemoryAllocInfo& allocInfo,
                                DevSize* pOffset, IMemory** ppMemory) {
    std::lock_guard<std::recursive_mutex> lg(m_Lock);
    Result res = SubAllocate(allocInfo, pOffset, ppMemory);  // Round 1

    if (res == Result::errorNotFound) {
        res = ChunkAllocate(allocInfo);                       // Round 2: 向 KMD 申请
        if (res == Result::success) {
            res = SubAllocate(allocInfo, pOffset, ppMemory);  // Round 3: 重试子分配
        }
    }
    return res;
}

// SubAllocate: O(1) 哈希桶查找
Result MemoryPool::SubAllocate(const MemoryAllocInfo& allocInfo,
                               DevSize* pOffset, IMemory** ppMemory) {
    uint32_t indexLow  = Log2(allocInfo.size);
    uint32_t indexHigh = Log2(allocInfo.size + allocInfo.alignment - 1);
    DevSize optionalMappingHash = ~((1ULL << (indexHigh + 1)) - 1) & m_EltMappingHash;
    // ★ 位图: m_EltMappingHash 的每一位表示对应大小桶有无空闲块
    //   用位运算快速跳过空的桶

    if (BitMaskScanForward(&index, optionalMappingHash)) {
        if (index != s_FreeTableLimit) {
            pMemRes = FindBucket(m_FreeBuckets[index], size, alignment, 1);
            // ★ 精确匹配: 只 try 1 次
        } else {
            // ★ 回退: 从 indexHigh 向下逐桶查找
            for (index = indexHigh; index != indexLow - 1; index--) {
                pMemRes = FindBucket(m_FreeBuckets[index], size, alignment, UINT64_MAX);
                if (pMemRes) break;
            }
        }
    }

    if (pMemRes) {
        ResourceSplit(pMemRes, size, alignment, &subAllocAddr);
        // ★ 切分: 左边间隙→free, 中间分配, 右边余量→free
        *ppMemory = pMemRes->pChunkMem;       // 返回 chunk 的 HAL Memory
        *pOffset  = subAllocAddr - chunkBase;  // chunk 内偏移
        m_FreeSize -= allocInfo.size;
    }
}

// ChunkAllocate: 向 KMD 申请 chunk
Result MemoryPool::ChunkAllocate(const MemoryAllocInfo& allocInfo) {
    DevSize chunkSize = AlignUp(max(allocInfo.size, m_ChunkAllocSize), // 至少 2MB
                                m_ChunkAlignment);
    // ...
    IMemory* pChunkMem = nullptr;
    if (物理内存) {
        res = m_pDevice->CreateMemory(chunkCreateInfo, &pChunkMem);
        // ★ KMD ioctl: mtgpuBoAlloc + mtgpuBoVmMapV2
    }
    // ...

    if (res == success) {
        ResSegment* pMemRes = new ResSegment(base, size);
        pMemRes->pChunkMem = pChunkMem;
        pMemRes->chunkBase = base;
        ResourceAdd(pMemRes);   // 加入链表 + 自由桶
        m_TotalSize += size;
        m_FreeSize  += size;
    }
    return res;
}
```

---

## 6. 释放路径回顾

```
Memory::~Memory() (memory.cpp:358-376)
    │
    ├─ if (m_pMapped) → m_pHalMemory->Unmap()
    │
    └─ if (m_Type != Prealloc && m_pHalMemory)
         │
         ├─ if (m_pHalMemory->GetProps() & SubAllocatable)
         │    │
         │    ├─ if (m_pPool == nullptr)  // 默认 pool (GeneralAlloc 路径)
         │    │     → GetMemMgr()->Free(m_pHalMemory, GetDevicePointer(), GetSize())
         │    │       → 按 property hash 查找 pool → pool->Free(deviceVA, size)
         │    │       → 合并相邻空闲段 → FreeListInsert
         │    │       → m_FreeSize += size
         │    │       ★ chunk 本身不会被释放 (惰性回收)
         │    │
         │    └─ if (m_pPool != nullptr)  // 用户 pool (InitFromPool 路径)
         │          → m_pPool->Hal()->Free(m_pHalMemory, GetDevicePointer(), GetSize())
         │
         └─ else (非 SubAllocatable)
              → m_PhysTracker.Cleanup()  // 清理 VA-PA 绑定
              → m_pHalMemory->Destroy()  // 直接释放 KMD 资源
         │
         ★ m_pPool 用于区分释放路径:
           nullptr   → MemMgr::Free (默认 pool，惰性回收)
           非 nullptr → Pool::Free (用户指定 pool)
```

---

## 7. 关键设计要点

### 7.1 three-layer allocator

```
Musa::MemoryPool (用户层, 可显式创建)
        ↓
Hal::IMemMgr (中间层, SplayTree 管理 pool)
        ↓
Hal::IMemory (底层, 裸 KMD 分配)
```

### 7.2 Property 推导链

```
Driver 传入: Virtual(1) | DeviceMapped(2) | SubAllocatable(4) = 0x0007
    ↓ Core 自动添加:
  Physical(8) | SharedVA(0x200) → 0x020F
    ↓ Virtual → 推导:
  DeviceVisible(0x20) | HostVisible(0x10) | HostCoherent(0x40)
  | DeviceWriteable(0x80) | DeviceCached(0x100) → 0x03FF
    ↓ DeviceMapped → 推导 ViewCap:
  PeerAccessible(2) | IpcExportable(4) | Exportable(1) → 0x07
```

### 7.3 Sub-Allocation 双回合机制

```
FullAllocate:
  Round 1: SubAllocate = O(1) 位图查找 + 桶遍历
    命中 → 返回 (零 ioctl)
    未命中 → Round 2
  Round 2: ChunkAllocate = KMD ioctl × 2 (申请 2MB chunk)
    ResourceAdd (加入 pool)
    成功 → goto Round 1 重试
```

### 7.4 地址合成

```
*dptr = pMemory->GetDevicePointer()
      = m_pHalMemory->GetDeviceVirtualAddress() + m_Offset

Sub-Alloc 路径: HalVA = chunk 基地址, m_Offset = chunk 内偏移
裸 KMD 路径:    HalVA = 分配地址,    m_Offset = 0
```

### 7.5 内存清零

```
Trace Capture 构建 (#if M3D_BUILD_MT_TRACE_CAPTURE):
  → memset(ptr, 0, size)  // 清零设备内存

排除类型:
  - memoryTypeRegisteredPinnedHost  // 用户已注册的指针，不应被修改
  - memoryTypeIpcImport            // 其他进程导入的内存，不应被修改
```

### 7.6 Chunk 惰性回收

```
Free 路径:
  pool->Free() → 仅归还到 free list (m_FreeSize += size)
  chunk 本身不释放

UpdateUserPools:
  按 releaseThreshold 判断空闲 chunk 是否应归还 KMD
  ★ 避免频繁 ioctl 的开销
```

### 7.7 流捕获保护

```
CreateMemory 前检查所有 stream:
  任何 stream 处于 ACTIVE capture → 返回 CAPTURE_UNSUPPORTED
  → 同时 InvalidateCaptureStatus() 标记所有 active → invalidated
  ★ CUDA 对齐: 捕获期间不允许分配新内存
```

---

## 8. 死代码 / 不可达路径标注

1. **裸 KMD 路径不可达（正常场景）**
   `GeneralAlloc` 中 `SubAllocatable` flag 由 Driver 层硬编码为 YES（flags=0x7），因此 `else` 分支中的 `Device::CreateMemory` 在正常 `muMemAlloc_v2` 路径下不会被触发。只有用户通过其他 API（如 `muMemAllocFromPool`）显式禁用或通过 `memoryTypePitchedGeneral` 才可能进入。

2. **`muapiMemAlloc` (v1) 委托 (`mu_memory.cpp:299`)**
   `muapiMemAlloc` 直接委托 `muapiMemAlloc_v2`，仅做类型转换（`MUdeviceptr_v1 → MUdeviceptr`），无额外逻辑。

3. **PitchedGeneralAlloc 的 flag 差异 (`memory.cpp:499-530`)**
   `PitchedGeneralAlloc` **不使用 Driver 传入的 flags**，而是直接硬编码全套 property（含 `SubAllocatable`），且 **不检查 alignment 溢出**（仅在 size 溢出时返回 OUT_OF_MEMORY）。其 property 值与 `GeneralAlloc` 推导结果完全相同（0x03FF）。

---

## 9. 性能关键路径总结

```
首次 muMemAlloc_v2(4KB):
  1. InitPlatform()                ← CPU: 懒初始化 (仅首次)
  2. TlsCtxTop()                   ← CPU: 线程局部变量 O(1)
  3. CreateMemory()                ← CPU: capture 检查 + CreateMemory
  4. GeneralAlloc()                ← CPU: property 推导
  5. GetMemMgr()->Allocate()       ← CPU: SplayTree 查找
  6. CreatePoolNoLock()            ← CPU: 创建 Pool
  7. ChunkAllocate(2MB)            ← CPU→KMD: mtgpuBoAlloc ioctl (1)
                                   ← CPU→KMD: mtgpuBoVmMapV2 ioctl (2)
  8. SubAllocate(4KB)              ← CPU: O(1) 位图+桶查找
  9. MapToPeers()                  ← CPU: 遍历 peers
  10. TrackMemory()                ← CPU: map insert O(log n)
  → 总耗时: ~0.4ms (2 次 ioctl)

后续 muMemAlloc_v2(4KB):
  1. TlsCtxTop()                   ← CPU
  2-4. CreateMemory → GeneralAlloc ← CPU
  5. GetMemMgr()->Allocate()       ← CPU: SplayTree 命中
  6. FullAllocate → SubAllocate    ← CPU: O(1) 查找
  → 总耗时: ~0.001ms (0 次 ioctl)
  → 提升: ~500x

裸 KMD 路径 (非 SubAllocatable):
  1-4. 同上
  5. Device::CreateMemory          ← CPU→KMD: mtgpuBoAlloc ioctl (1)
                                   ← CPU→KMD: mtgpuBoVmMapV2 ioctl (2)
  → 总耗时: ~0.2ms (2 次 ioctl)
  → 每次分配都是 2 次 ioctl
```

---

## 10. 交叉引用

- `02_muMemFree_v2.md` — 释放路径（含析构双路径 + 死代码标注）
- `10_CreateMemory_and_MapToPeers.md` — CreateMemory 统一入口 + 9 种类型分发
- `11_GeneralAlloc_deep_dive.md` — Sub-Allocation 池化分配全链路
- `12_DirectKMD_Allocation_flow.md` — 裸 KMD 分配路径（非 SubAlloc）
- `13_MemoryPool_deep_dive.md` — Pool 算法的 4 个执行示例
- `19_VA_PA_binding.md` — VA↔PA 绑定机制
- `24_muLaunchKernel_Flow.md` — kernel 启动完整流程