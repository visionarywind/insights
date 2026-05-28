# MUSA 流程决策逻辑全景

> **📖 相关文档**: [memory_analysis.md](memory_analysis.md) (架构总览) | [memory_api_deep_analysis.md](memory_api_deep_analysis.md) (API流程) | [muMemAlloc_vs_muMemAllocAsync.md](muMemAlloc_vs_muMemAllocAsync.md) (对比) | [pooling_analysis.md](pooling_analysis.md) (池化) | [stream_command_analysis.md](stream_command_analysis.md) (Stream)

本文档集中描述 MUSA 驱动中所有关键的**条件分支**和**流程决策点**。每个决策点都展示：判断条件 → 每个分支做什么 → 为什么这样设计。

---

## 目录

1. [内存分配类型分发](#1-内存分配类型分发)
2. [GeneralAlloc：SubAllocatable 分支 + 属性推导链](#2-generalallocsuballocatable-分支--属性推导链)
3. [PinnedHostAlloc：LargePage 降级](#3-pinnedhostalloclargepage-降级)
4. [HostRegister：IOMMU vs Locked](#4-hostregisteriommu-vs-locked)
5. [Memcpy：CopyManager 选择优先级链](#5-memcpycopymanager-选择优先级链)
6. [Memcpy：Async vs Sync 命令选择](#6-memcpyasync-vs-sync-命令选择)
7. [Memset：引擎选择](#7-memset引擎选择)
8. [Memset Submit：CPU / DMA / GPU Queue 三路](#8-memset-submitcpu--dma--gpu-queue-三路)
9. [Stream 三路分支：Capture / Invalidated / Normal](#9-stream-三路分支capture--invalidated--normal)
10. [隐式依赖：四种 Stream 类型的依赖规则](#10-隐式依赖四种-stream-类型的依赖规则)
11. [Command 合并：CanMergeTo 条件](#11-command-合并canmergeto-条件)
12. [muMemFree 类型分发](#12-mumemfree-类型分发)
13. [AsyncMemAlloc 三步分配](#13-asyncmemalloc-三步分配)
14. [CopyManager 引擎能力判定](#14-copymanager-引擎能力判定)

---

## 1. 内存分配类型分发

**入口**: `Memory::Init(const MemoryCreateInfo& createInfo)` — `memory.cpp:378`

```
用户调用 muMemAlloc / muMemHostAlloc / muMemAllocManaged / ...

    │ createInfo.type 决定走哪个分支
    ▼
┌──────────────────────────────────────────────────────┐
│  switch (m_Type)                                     │
├──────────────────────────────────────────────────────┤
│  memoryTypeGeneral               → GeneralAlloc      │
│  memoryTypePitchedGeneral        → PitchedGeneralAlloc│
│  memoryTypePinnedHost            → PinnedHostAlloc   │
│  memoryTypeRegisteredPinnedHost  → PinnedHostRegister│
│  memoryTypeManaged               → ManagedAlloc      │
│  memoryTypeIpcImport             → IpcImportAlloc    │
│  memoryTypeExternal              → ExternalAlloc     │
│  memoryTypePrealloc              → PreallocAlloc     │
│  memoryTypeVirtual               → VirtualAlloc      │
│  default                         → INVALID_VALUE     │
└──────────────────────────────────────────────────────┘

调用方设置 createInfo.type 的方式:
  muMemAlloc     → memoryTypeGeneral
  muMemHostAlloc → memoryTypePinnedHost
  muMemAllocAsync→ memoryTypeVirtual (内部) + memoryTypeGeneral (物理)
  ...
```

---

## 2. GeneralAlloc：SubAllocatable 分支 + 属性推导链

**入口**: `Memory::GeneralAlloc()` — `memory.cpp:462`

### 2.1 属性推导链

```
输入: flags = Virtual | DeviceMapped | SubAllocatable

① property = flags    (初始值 = 用户的 flags)
② property |= Physical | SharedVirtualAddress     (强制项)
③ if (flags & Virtual):
     property |= DeviceVisible | HostVisible | HostCoherent | DeviceWriteable | DeviceCached
   else:
     property |= None
④ viewCapability = Exportable
⑤ if (flags & DeviceMapped):
     viewCapability |= PeerAccessible | IpcExportable

最终 property = Physical | SharedVA | Virtual | DeviceVisible | HostVisible 
               | HostCoherent | DeviceWriteable | DeviceCached | SubAllocatable
               
最终 viewCapability = Exportable | PeerAccessible | IpcExportable
```

### 2.2 SubAllocatable 分支

```
    if (property & SubAllocatable)?
       │
  YES  │              NO
       ▼               ▼
┌──────────────┐  ┌─────────────────┐
│ MemMgr::     │  │ Device::        │
│ Allocate()   │  │ CreateMemory()  │
│              │  │                 │
│ 从已有 pool   │  │ 向 KMD 申请     │
│ 中切出一块    │  │ 全新独立内存     │
│ (带 offset)   │  │                 │
│              │  │ 适用于: Managed, │
│ 适用于:       │  │   External,     │
│ muMemAlloc,  │  │   IPC Import,   │
│ muMemHostAlloc│  │   大块独立分配   │
└──────────────┘  └─────────────────┘
```

**为什么默认用 SubAllocatable？** 大多数 `muMemAlloc` 都是小分配。每次调用 KMD (ioctl) 开销很大 → 预分配大块，从池中切分，减少系统调用。

---

## 3. PinnedHostAlloc：LargePage 降级

**入口**: `Memory::PinnedHostAlloc()` — `memory.cpp:532`

```
① heap = LargePage (默认)

② 尝试分配:
     if (SubAllocatable)
         MemMgr::Allocate(heap=LargePage)
     else
         Device::CreateMemory(heap=LargePage)
         
③ if (status == MAP_FAILED && heap == LargePage):
     │  ★ 降级触发: LargePage 映射失败
     │
     ├─ heap = MemoryHeap::general   ← static 变量!
     │   从此该进程所有后续 PinnedHostAlloc 都用 General heap
     │
     └─ 重试分配:
          if (SubAllocatable) MemMgr::Allocate(heap=General)
          else Device::CreateMemory(heap=General)
          
④ if (分配成功) return

⚠ pinnedHostAllocHeap 是 static 变量 → 降级是永久性的
```

**为什么？** 某些系统不支持 LargePage 映射 (如内存碎片、内核限制)。一次失败意味着系统不支持 → 没必要每次重试。

---

## 4. HostRegister：IOMMU vs Locked

**入口**: `Memory::PinnedHostRegister()` — `memory.cpp:611`

```
用户传入指针 + flags

    │
    ├─ if (OVERLAP_CHECK):
    │     MemoryTracker::IterateMemories(ptr..ptr+size)
    │     检查该区间是否已有 RegisteredPinnedHost 内存
    │     if (已有) → HOST_MEMORY_ALREADY_REGISTERED
    │
    ├─ if (IOMEMORY flag):
    │     │ 用户传入的是 MMIO 地址 (如 PCIe BAR 空间)
    │     │
    │     └─ createInfo.type = memoryTypeView
    │        view.type = memoryViewTypeExternal
    │        view.external.type = MMIO
    │        view.external.handle = ptr (物理地址)
    │
    └─ else (普通 Host 内存):
          │ 用户传入普通 malloc 内存
          │
          └─ createInfo.type = memoryTypeView
             view.type = memoryViewTypeLocked
             view.locked.pHost = ptr          ← 用户指针
             view.locked.size = size
             view.locked.isDeviceMapped  = (flags & DEVICEMAP)
             view.locked.isDeviceReadOnly = (flags & READ_ONLY)
             view.locked.isPeerAccessible = (flags & PORTABLE)
             
             → HAL: LockGpuMemory(ptr,size) → 锁定页表
             → HAL: MapGpuPageTable → 建立 GPU MMU 映射
```

---

## 5. Memcpy：CopyManager 选择优先级链

**入口**: `GraphMemcpyNode::SelectPerfCopyManager()` — `graphMemcpyNode.cpp:484`

`copyManagerSelector` 是一个**优先级选择器**：按顺序尝试每个候选，选第一个可用的。

### 5.1 H2D (Host → Device)

```
优先级 (从高到低):
  ① COPY_MANAGER_CPU   ← if (copySize < s_H2dCEorTDMThreshold && dstCanMapToCpu)
                         ★ 小拷贝直接用 CPU memcpy，比 GPU 更快
  ② COPY_MANAGER_CE    ← if (EnableUserQueue(ce) && dstCanUseCe)
                         ★ CE 引擎可用且支持用户队列 → CE
  ③ COPY_MANAGER_TDM   ← 始终可用 (兜底)
                         ★ Transfer Data Mover, GPU 传输引擎
  ④ COPY_MANAGER_CE    ← if (dstCanUseCe)
                         ★ fallback: CE 可用但未启用用户队列
  ⑤ COPY_MANAGER_CPU   ← if (dstCanMapToCpu)
                         ★ 最终兜底: CPU memcpy
```

### 5.2 D2H (Device → Host)

```
  ① COPY_MANAGER_CPU   ← if (copySize < s_D2hTDMThreshold && srcCanMapToCpu)
  ② COPY_MANAGER_CE    ← if (EnableUserQueue(ce) && srcCanUseCe)
  ③ COPY_MANAGER_TDM   ← 始终可用
  ④ COPY_MANAGER_CE    ← if (srcCanUseCe)
  ⑤ COPY_MANAGER_CPU   ← if (srcCanMapToCpu)
```

### 5.3 D2D (Device → Device)

```
  ① COPY_MANAGER_CE     ← if (hasHost || isDiff) && canUseCe
                          ★ 跨设备或含 host 内存 → CE 优先
  ② COPY_MANAGER_SHADER ← if (IP >= 3 || Height>1 || Depth>1)
                          ★ PH1+ 或 2D/3D → Shader (一次完成)
  ③ COPY_MANAGER_TDM    ← 始终可用
  ④ COPY_MANAGER_CE     ← fallback CE
  ⑤ COPY_MANAGER_CPU    ← if (canMapToCpu)
```

### 5.4 A2D / D2A (Array ↔ Device)

```
  ① COPY_MANAGER_SHADER ← if (Format != HALF)  ★ shader 不支持半精度
  ② COPY_MANAGER_TDM    ← 始终可用
  ③ COPY_MANAGER_CE     ← CE 兼容检查通过
```

### 5.5 H2H (Host → Host)

```
  ① COPY_MANAGER_CPU    ← 始终可用 (只有 CPU 能做 H2H)
```

---

## 6. Memcpy：Async vs Sync 命令选择

**入口**: `Stream::CmdCopyMemory()` — `stream.cpp:662`  
**判断**: `GraphMemcpyNode::IsAsyncMemcpyCmd()` — `graphMemcpyNode.cpp:643`

```
┌────────────────────────────────────────────────────────────────┐
│  IsAsyncMemcpyCmd(node) ?                                      │
│                                                                │
│  = (Direction ∈ {D2D, D2A, A2D, A2A})                         │
│    AND                                                         │
│    (Engine ∈ {CDM, TDM, CE})                                   │
│                                                                │
│  YES → new AsyncMemcpyCommand  │  NO → new SyncMemcpyCommand   │
│  ┌─────────────────────────┐   │  ┌──────────────────────────┐ │
│  │ 走标准 GPU 路径:         │   │  │ 走 CopyManager CPU 路径: │ │
│  │ Build → CmdCopyMemory   │   │  │ Submit → CpyMgr->        │ │
│  │ Submit → IQueue::Submit │   │  │   MemcpyH2D/H2H/D2H()   │ │
│  │ → GPU DMA engine 执行   │   │  │ → 主机端 CPU memcpy      │ │
│  └─────────────────────────┘   │  └──────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

**为什么 H2D/D2H 不用 Async？** H2D 和 D2H 涉及 CPU 可以访问的内存。SyncMemcpyCommand 通过 CopyManager 直接在 CPU 上 `memcpy`，比 GPU DMA 提交开销更低（尤其是小拷贝）。

---

## 7. Memset：引擎选择

**入口**: `MemsetCommand 构造函数` — `memsetCommand.cpp:25-62`

```
Platform Settings: memsetPath (可配置覆盖默认行为)

switch (memsetPath):
┌──────────────────────────────────────────────────────────────┐
│ SET_EXECUTOR_DEFAULT (默认):                                  │
│   ┌─ if (HalQueue(CDM) != nullptr):                          │
│   │     engine = CDM, merge = true                           │
│   │     ★ CDM 可用 → 走 GPU 计算引擎                          │
│   └─ else:                                                   │
│         engine = count  (无效引擎标记)                        │
│         ★ CDM 不可用 → 标记为 CPU fallback                    │
│                                                              │
│ SET_EXECUTOR_CDM:   engine=CDM,   merge=true                 │
│ SET_EXECUTOR_CE:    engine=CE,    merge=true                 │
│ SET_EXECUTOR_TDM:   engine=TDM                               │
│ SET_EXECUTOR_DMA:   engine=DMA                               │
│ SET_EXECUTOR_CPU:   engine=count  (无效引擎标记)              │
│ SET_EXECUTOR_SHADER:engine=CDM,   merge=true                 │
└──────────────────────────────────────────────────────────────┘
```

**⚠ 已知限制**: DEFAULT 路径只尝试 CDM，失败就直接 CPU。应该也尝试 CE、TDM、DMA 作为 fallback。

---

## 8. Memset Submit：CPU / DMA / GPU Queue 三路

**入口**: `MemsetCommand::Submit()` — `memsetCommand.cpp:190`

```
┌────────────────────────────────────────────────────────────┐
│  if (engine == count)                    ← CPU 路径        │
│    └─ CpuExecute()                                        │
│       └─ 直接写 GPU 可映射内存 (CPU 端完成)                  │
│                                                           │
│  else if (engine == DMA)                 ← DMA 路径        │
│    └─ DmaExecute()                                        │
│       └─ 使用 DMA Engine 完成 GPU 填充                     │
│                                                           │
│  else                                    ← GPU Queue 路径  │
│    └─ 标准 GPU 提交:                                       │
│       CmdBuffer::End() → QueueSubmitInfo → Queue::Submit() │
│       └─ 通过 CDM/CE/TDM 引擎的硬件队列提交                │
└────────────────────────────────────────────────────────────┘
```

---

## 9. Stream 三路分支：Capture / Invalidated / Normal

**入口**: 所有 `Stream::CmdXXX()` 方法

```
每个 CmdXXX 方法都有相同的三路分支:

Stream::CmdMemset(node, blocking)        // 示例
  │
  ├─ if (m_CaptureStatus == ACTIVE):
  │     return CaptureNode(pGraphNode)
  │     │
  │     └─ m_CaptureGraph->AddGraphNode(node, deps)
  │        将操作记录到 CUDA Graph 中，不实际执行
  │        ★ 只记录图结构，GPU 操作被延迟到图重放时
  │
  ├─ else if (m_CaptureStatus == INVALIDATED):
  │     return ERROR_STREAM_CAPTURE_INVALIDATED
  │     ★ 图捕获已失效，拒绝所有操作
  │
  └─ else (NONE):
        return AsyncXXX(param, blocking)
        │
        └─ 正常执行路径:
           创建 Command → QueueCommand → AsyncSubmit 线程处理
```

**为什么需要 INVALIDATED 状态？** 如果捕获过程中出现不支持的操作（如 `muMemAlloc`），整个捕获失效，必须拒绝后续所有操作，不能静默丢失。

---

## 10. 隐式依赖：四种 Stream 类型的依赖规则

**入口**: `Context::ResolveDependencyAndQueueCommand()` — `context.cpp:1859`

```
每个命令入队时，自动建立跨 Stream 的依赖关系:

┌───────────────────────────────────────────────────────────────┐
│  命令目标 stream:                                              │
│                                                               │
│  if (stream == m_DefaultStream):                              │
│    ┌──────────────────────────────────────────┐              │
│    │ for each other_stream in context:         │              │
│    │   if (not default && not barrier &&       │              │
│    │       not nonBlocking):                   │              │
│    │     command->RecordDependency(            │              │
│    │       other_stream->LastCommand())        │              │
│    │                                           │              │
│    │ ★ Default Stream 依赖所有 Blocking Stream │              │
│    │ ★ 保证 Default Stream 上的操作在所有       │              │
│    │   Blocking Stream 的已有操作完成后才执行    │              │
│    └──────────────────────────────────────────┘              │
│                                                               │
│  else if (stream == m_BarrierStream):                         │
│    ┌──────────────────────────────────────────┐              │
│    │ for each other_stream in context:         │              │
│    │   if (not barrier):                       │              │
│    │     command->RecordDependency(            │              │
│    │       other_stream->LastCommand())        │              │
│    │                                           │              │
│    │ ★ Barrier Stream 依赖所有其他 Stream      │              │
│    │ ★ 全局同步点：等所有 stream 都完成         │              │
│    └──────────────────────────────────────────┘              │
│                                                               │
│  else if (!stream->IsNonBlocking()):    // Blocking Stream     │
│    ┌──────────────────────────────────────────┐              │
│    │ command->RecordDependency(               │              │
│    │   m_DefaultStream->LastCommand())        │              │
│    │                                           │              │
│    │ ★ Blocking Stream 依赖 Default Stream     │              │
│    │ ★ 保证与 Default Stream 的顺序一致性       │              │
│    └──────────────────────────────────────────┘              │
│                                                               │
│  else (NonBlocking Stream):                                   │
│    └─ 不添加任何隐式依赖                                      │
│       ★ 最大并行度，与其他 Stream 可自由重排                  │
│                                                               │
│  ★★★ 所有 Stream 都额外依赖:                                   │
│  barrierCommand = m_BarrierStream->LastCommand()              │
│  if (barrierCommand && barrierCommand->Status > completed):    │
│    command->RecordDependency(barrierCommand)                  │
│    ★ 保证 Barrier 操作在所有 Stream 上都是全局顺序点           │
└───────────────────────────────────────────────────────────────┘
```

**依赖如何生效？** `command->Build()` 阶段，将 `m_ExecutionDependencies` 中的每个依赖命令的 `(信号量, 信号值)` 编码为 `CmdWaitSemaphore` → GPU 在执行命令前必须等待所有依赖完成。

---

## 11. Command 合并：CanMergeTo 条件

**入口**: `Command::CanMergeTo()` — `command.h:109`

```
CanMergeTo(mergingList) 返回 true 当且仅当:

① SupportMerge() == true
   Memset, Dispatch, AsyncMemcpy → 支持合并
   SyncMemcpy, Graph, Barrier   → 不支持合并

② m_ExecutionDependencies.empty()
   没有跨 Stream 依赖 → 如果有依赖，不能合并
   (合并后依赖会变得更复杂)

③ !mergingList.empty()
   MergingList 必须已有至少一个命令

④ mergingList.back()->SupportMerge()
   前一个命令也支持合并

⑤ mergingList.back()->GetEngine() == GetEngine()
   与前一个命令使用相同的 GPU 引擎

全部满足 → true → 合并到当前 MergingList
任一不满足 → false → 先提交当前 MergingList，再单独 Build
```

**停止合并条件 (stopMerging)**:

```
stopMerging() 返回 true 当:

① MergingList.size() >= 32     ← 硬限制: 最多合并 32 个
   或

② EngineSubmissionReady()
   Inflight 命令数 < 限制 (3 for userQ, 2 for non-userQ)
   ★ 引擎有空闲了 → 赶紧提交
```

---

## 12. muMemFree 类型分发

**入口**: `muapiMemFree_v2()` — `mu_memory.cpp:709`

```
① MemoryTracker::FindRange(dptr) → 找到 Memory 对象

② 类型检查 (只允许释放这些类型):
   if (type != General && type != PitchedGeneral &&
       type != Managed && type != External &&
       type != Virtual) → INVALID_VALUE
   
   if (type == Virtual && pool == nullptr) → INVALID_VALUE
   ★ Virtual 内存必须属于一个 pool 才能释放

③ offset != 0 → INVALID_VALUE
   ★ sub-allocation 不能单独释放中间片段，必须释放整个 Memory

④ Synchronize() → 等待所有 GPU 操作完成

⑤ 按类型分发释放:
   ┌──────────────────────────────────────────────┐
   │ if (type == Virtual):                        │
   │   └─ Stream::CmdMemFree(dptr, true)          │
   │      └─ DisableAccess + CallbackCommand       │
   │         ★ 异步释放: 等 GPU 完成后再清理       │
   │                                              │
   │ else (General/Pitched/Managed):              │
   │   └─ Context::DestroyMemory(pMemory)          │
   │      └─ RemoveMemory + UntrackMemory          │
   │         → ~Memory()                           │
   │           ├─ [SubAlloc] MemMgr::Free → 归还池 │
   │           └─ [裸分配] m_pHalMemory->Destroy() │
   │              → KMD 释放 GPU 物理内存          │
   └──────────────────────────────────────────────┘
```

---

## 13. AsyncMemAlloc 三步分配

**入口**: `Stream::AsyncMemAlloc()` — `stream.cpp:518`

```
muMemAllocAsync(dptr, size, hStream)
  │
  ├─ ① 从内存池分配虚拟地址
  │   pPool = param.pool ? param.pool : device->GetMemoryPool()
  │   pPool->CreateMemory(&virt, &virtAddr, allocSize)
  │   │
  │   └─ Pool::FullAllocate → [SubAllocate 或 ChunkAllocate+SubAllocate]
  │      ★ 这是 POOL 分配，只有 VA，没有物理内存
  │      ★ virt->GetType() == memoryTypeVirtual
  │
  ├─ ② 独立分配物理内存
  │   new Memory(ctx)
  │   physical->Init(createInfo)  → GeneralAlloc(size, 0, flags=0)
  │   │
  │   └─ property = Physical | SharedVA
  │      ★ flags=0 → 不含 Virtual/DeviceVisible 等
  │      ★ 纯物理分配: 有 VRAM 存储，无独立 VA
  │      ★ physical->GetType() == memoryTypeGeneral
  │
  └─ ③ 绑定 + Paging (异步页表设置)
      │
      ├─ virt->Bind(physical, allocSize, 0, 0)
      │   └─ virt.PhysTracker 登记 VA→物理 映射
      │
      └─ pPool->ModifyAccess(virt, physical, allocSize, blocking, stream)
          │
          ├─ 同步: OpenPeerMemory (每个 peer 设备)
          │   → 创建跨设备引用，但不填充页表
          │
          └─ 异步: Stream::CmdPaging(pagingParams, blocking)
              └─ PagingCommand 入队到 stream
                 → AsyncSubmit 线程在依赖就绪后:
                   HostWaitSemaphores() → 等前序 GPU 命令完成
                   然后 MMU->Paging(info) → 填充页表
                   然后 HostSignalFinish() → 通知后续命令
```

**为什么 split Virtual + Physical？** Async 分配的结果绑定在 stream 上。如果一步完成（像 muMemAlloc），内存立即可用 → 不保证与 stream 的顺序。分开后：VA 立即可用，物理绑定通过 PagingCommand 在 stream 队列按序执行 → 与 stream 上的其他操作有明确的顺序关系。

---

## 14. CopyManager 引擎能力判定

各引擎的条件判断 (GraphMemcpyNode.cpp):

| 引擎 | 条件 |
|------|------|
| **CPU** | `!src->IsVirtual() && !dst->IsVirtual()` — 源和目标都可 CPU 映射 |
| **DMA** | `!src->IsVirtual() && !dst->IsVirtual()` 或 `isSrcPageable && isDstPageable` |
| **TDM** | 始终可用 (除了 Pageable→Pageable 场景) |
| **CE** | `IsCeCompatible(memory, device)` — 检查内存是否在 CE 可见的堆上 |
| **CDM** | 始终可用 (CDM 是 Compute Data Mover) |
| **Shader** | `!IsHalfFormat` — 不支持半精度; D2D 需要 IP≥3 或 2D/3D |
| **ACE** | 仅在 CE 路径 + D2D + 跨设备 + 1D 时替代 CE |

**DEFAULT 与显式路径**: 当 `memcpyPath == DEFAULT` 时走 `SelectPerfCopyManager`（自动选择最优引擎）。当 `memcpyPath` 显式指定时走 `CopyManagerSelectPass`（验证指定引擎是否可用）。

---

## 15. MapToPeers：Peer 类型分发

**入口**: `Context::CriticalBase::MapToPeers()` — `context.cpp:487-543`

两个 `switch(memoryType)` 决定 peer 数量和 peer 映射方式:

### 15.1 Peer 数量

```
switch (memory->GetType()):
  External/IPC/RegisteredPinned/PinnedHost → ALL devices (全部)
  General/PitchedGeneral → 仅 context 已配对的 peers
  Managed → 取决于 managedForceDeviceAlloc 设置
  default → 0 (无 peer)
  
for each peer:
  memory->Hal()->OpenPeerMemory(&remoteDevice->Hal(), openInfo)
  → 向 KMD 导出 global handle → 在 peer 设备导入 → 映射 peer VA 空间
```

### 15.2 Peer 映射标志

```
switch (memory->GetType()):
  General/PitchedGeneral → 使用 context 存储的 peer flags
  PinnedHost/RegisteredPinnedHost → PCIEONLY (仅 PCIe 路径)
  Managed → 条件: concurrentManagedAccess + managedForceDeviceAlloc
  IPCImport → flags - 1 (M3D-TODO: workaround)
  External → INVALID (不支持 peer)
```

---

## 16. ManagedAlloc：标志校验

**入口**: `Memory::ManagedAlloc()` — `memory.cpp:677-683`

```
if (flags == MU_MEM_ATTACH_HOST && !concurrentManagedAccess):
    → NOT_SUPPORTED    ★ ATTACH_HOST 需要并发管理访问支持
    
if (flags == MU_MEM_ATTACH_GLOBAL && !unifiedAddressing):
    → 移除 ATTACH_GLOBAL + 警告  ★ 不支持统一地址空间，静默降级
    
allocType = managedForceDeviceAlloc ? DeviceLocal : Host
```

---

## 17. ExternalAlloc：外部句柄类型分发

**入口**: `Memory::ExternalAlloc()` — `memory.cpp:772-785`

```
switch (handleInfo.type):
  Fabric    → 使用 fabric 句柄类型 (内核间互连)
  OpaqueFd  → 使用 dmaBuf 句柄类型 (POSIX fd)
  default   → NOT_SUPPORTED
  
handleInfo.viewSize = (viewSize == 0) ? fullSize : viewSize
```

---

## 18. PinnedHostAlloc 的 SubAllocatable 分支 (与 GeneralAlloc 同模式)

**入口**: `Memory::PinnedHostAlloc()` — `memory.cpp:584-588`

```
与 GeneralAlloc 完全相同的模式:

if (SubAllocatable):
    MemMgr->Allocate(allocInfo, &offset, &pHalMemory)   // 从池中切
else:
    Device->CreateMemory(createInfo, &pHalMemory)         // 向 KMD 申请独立块
```

---

## 19. Memset TDM 引擎的分块限制

**入口**: `MemsetCommand::BuildFillMemory()` — `memsetCommand.cpp:268`

```
if (engine == TDM):
    ★ TDM 引擎有单次操作大小限制
    将 fill 分成多个 chunk，每个 chunk ≤ s_PerTDMFillMaxSize
    循环多次 CmdFillMemory(), 每次向前推进 offset

else:
    单次 CmdFillMemory() 完成整个操作
```

---

## 20. Memset CpuExecute：元素大小分发

**入口**: `MemsetCommand::CpuExecute()` — `memsetCommand.cpp:295-322`

```
switch (elementSize):
  1 → memset(hostPtr, value, totalBytes)      // byte 级
  2 → for loop: *(uint16_t*)ptr = (uint16_t)value  // short 级
  4 → for loop: *(uint32_t*)ptr = (uint32_t)value  // int 级
  default → INVALID_VALUE

⚠ Virtual Memory 路径:
  if (IsVirtual()):
    memory = GetPhysMemory()  → 获取物理内存的 host 指针
    否则直接使用原 memory 的 host 指针
```

---

## 关键设计模式总结

| 模式 | 示例 | 原理 |
|------|------|------|
| **优先级链选择** | CopyManager H2D: CPU→CE→TDM→CPU | 按性能递减尝试，先选快的 |
| **三路分支** | Stream CmdXXX: Capture/Invalidated/Normal | CUDA Graph 兼容 |
| **类型分发** | Memory::Init: switch on 9 types | 统一接口，内部多态 |
| **属性推导** | GeneralAlloc: flags → property 推导 | 用户简洁 flags → 系统完整 property |
| **静默降级** | PinnedHostAlloc: LargePage→General | 系统不支持的优化自动回退 |
| **延迟绑定** | AsyncMemAlloc: VA first, PA later | VA 立即可用，物理绑定走 stream 顺序 |
| **依赖注入** | Default/Barrier/Blocking 自动依赖 | CUDA 隐式同步语义 |

---

*本文档基于 musa 项目源码分析生成，commit: 9ba99a5d, branch: bugfix/sw-79049*
