# MUSA Memory 分配机制完全指南

> **本文目标**: 用一条主线串起 MUSA 内存分配的全部机制。读完本文，你将能够：
> - 画出一张 `muMemAlloc` 到 GPU 硬件的完整数据流图
> - 解释每一步"做了什么"和"为什么这样做"
> - 理解 9 种内存类型的本质差异
> - 理解同步分配 vs 异步分配的根本区别
> - 理解 MemoryPool 如何优化分配性能

---

## 第一章：你必须先理解的三个概念

在深入代码之前，先建立三个核心概念。这些概念是理解所有后续内容的基础。

### 概念 1：GPU 有两套地址

```
CPU 看到的:                    GPU 看到的:
┌──────────────┐              ┌──────────────┐
│ Host 内存     │              │ Device 内存   │
│ (DDR RAM)    │              │ (VRAM/显存)   │
│              │              │              │
│ 地址:        │              │ 地址:        │
│ 0x7f... (VA) │              │ 0x100... (VA)│
│ ↓ MMU翻译    │              │ ↓ GPU MMU翻译 │
│ 物理页       │              │ 物理页        │
└──────────────┘              └──────────────┘

关键点:
- CPU 和 GPU 的地址空间是独立的
- GPU 不能直接访问 CPU 的虚拟地址 (反之亦然)
- 要让 GPU 访问某块内存，必须"映射"——在 GPU MMU 页表中建立 VA→PA 的翻译
```

### 概念 2：一块"可用"的 GPU 内存需要三样东西

```
一块完整的 GPU 内存 =
    Virtual Address (VA)    ← 程序通过它访问 (类似 C 指针)
  + Physical Pages (PA)     ← 实际存储数据的地方 (VRAM 中的物理页)
  + MMU Page Table Entry    ← 将 VA 翻译为 PA 的"字典项"

三者缺一不可:
  - 有 VA 无 PA → 访问会触发 page fault (类似访问未分配的内存)
  - 有 PA 无 VA → GPU 程序找不到这块内存
  - 有 VA+PA 无页表 → GPU MMU 不知道如何翻译, 访问失败

对应 MUSA 的标志位:
  memoryPropertyVirtual   = 有 GPU VA
  memoryPropertyPhysical  = 有物理存储
  memoryPropertyDeviceMapped = 页表已填充 (GPU 可访问)
```

### 概念 3：分配 GPU 内存很"贵"

```
向 GPU 分配内存需要:
  1. 调用内核驱动 (ioctl) — 用户态→内核态切换
  2. 内核分配 VRAM 物理页
  3. 内核分配 GPU 虚拟地址空间
  4. 内核编程 GPU MMU 页表

一次 ioctl ≈ 几微秒 ~ 几十微秒的开销

因此 MUSA 设计了 "MemoryPool":
  不是每次 muMemAlloc(4KB) 都调 ioctl
  而是一次调 ioctl 分配 2MB 大块
  然后自己把 2MB 切成很多 4KB 发出去
  这样 100 次小分配只需要 1 次 ioctl
```

---

## 第二章：跟着一次 muMemAlloc 走完全程

现在让我们跟踪一次最简单的内存分配：`muMemAlloc(&dptr, 4096)` (分配 4KB GPU 内存)。

### Step 1: 用户调用

```c
MUdeviceptr dptr;  // 出参: GPU 虚拟地址
muMemAlloc(&dptr, 4096);
// 成功后, dptr = 0x10000000 (一个 GPU VA)
```

### Step 2: Driver 层 — 参数校验 + 构造请求 (`mu_memory.cpp:265`)

```
muapiMemAlloc_v2(dptr, 4096):

① InitPlatform()          — 确保 MUSA 已初始化 (首次调用时创建 Platform 单例)
② TlsCtxTop()             — 获取当前线程绑定的 GPU Context
③ 构造 MemoryCreateInfo:
     type = memoryTypeGeneral     ← "通用设备内存"
     size = 4096
     flags = Virtual | DeviceMapped | SubAllocatable
           ↑          ↑               ↑
           │          │               └─ 允许从 Pool 中切分 (不是每次调 ioctl)
           │          └─ 页表立即可填充
           └─ 分配 GPU VA
④ pContext->CreateMemory(&pMemory, createInfo)  → 进入 Core 层
```

### Step 3: Core 层 — 创建 Memory 对象 (`context.cpp:915`)

```
Context::CreateMemory(createInfo):

① 检查是否有 Stream 在 Capture 模式 → 有则报错
② new Memory(this) + Memory::Init(createInfo)  → 真正分配
③ MapToPeers: 如果启用了 Peer Access, 在 peer 设备上也建立映射
④ AddMemory: 将 Memory 对象加入 Context 的追踪表
⑤ TrackMemory: 将 (dptr → Memory*) 注册到全局 MemoryTracker
⑥ 返回 *ppMemory = pMemory
```

### Step 4: Memory::GeneralAlloc — 决定分配方式 (`memory.cpp:462`)

```
GeneralAlloc(size=4096, alignment=0, flags=Virtual|DeviceMapped|SubAllocatable):

① 属性推导:
   property = flags                              = Virtual | DeviceMapped | SubAllocatable
   property |= Physical | SharedVirtualAddress   ← 强制项
   property |= (flags & Virtual) ? DeviceVisible | HostVisible | ... : 0

   最终 property = Physical | SharedVA | Virtual | DeviceVisible | 
                  HostVisible | HostCoherent | DeviceWriteable | 
                  DeviceCached | SubAllocatable

② SubAllocatable 分支:
   if (property & SubAllocatable):   ← YES!
       → MemMgr::Allocate(info, &offset, &pHalMemory)
         // 从 Pool 中切一块, 不调 ioctl (除非 Pool 需要扩容)
   else:
       → Device::CreateMemory(info, &pHalMemory)
         // 直接调 ioctl 分配新内存
```

### Step 5: MemMgr — 找到或创建合适的 Pool

```
MemMgr::Allocate(allocInfo):

① 计算 Pool Key: 将 allocInfo 的属性编码为一个 64 位 key
   key = (property << 0) | (viewCapability << 32) | (type << 38) | (heap << 39) | (numa << 42)

② 在 SplayTree 中查找匹配的 Pool:
   pPool = m_PoolRefs.Find(key)
   if (pPool):  →  Step 6 (直接使用已有 Pool)

③ 如果没有匹配的 Pool:  创建新的
   new MemoryPool(device) + Init(createInfo)
   注册到 m_PoolRefs

④ pPool->FullAllocate(allocInfo, &offset, &pHalMemory)  →  Step 6
```

### Step 6: MemoryPool — 从 Pool 中切一块

```
MemoryPool::FullAllocate(allocInfo):

① SubAllocate(allocInfo):   ← 先尝试从已有 Chunk 中找空闲段
   计算 indexLow = Log2(4096) = 12, indexHigh = 12
   从 m_EltMappingHash 找 >= 13 的非空 bucket
   假设找到 Bucket[21] 有一个 2MB 的空闲段
   
   FindBucket(Bucket[21], 4096, 0, 1):
     2MB >= 4096 → 找到!
   
   ResourceSplit:
     前部分裂: 不需要 (对齐为0)
     后部分裂: 4096 < 2MB → 切出 4KB
     
     结果:
     ┌──────┬─────────────────────────┐
     │ 4KB  │     2MB - 4KB (free)    │
     │ busy │                         │
     └──────┴─────────────────────────┘
     
   *ppMemory = segment->pChunkMem   ← 这个 Chunk 的 IMemory
   *pOffset  = 4KB 的偏移            ← 在 Chunk 内的位置
   
② 如果 SubAllocate 失败 (没找到足够大的空闲段):
   ChunkAllocate(allocInfo):
     chunkSize = AlignUp(4096, 2MB) = 2MB
     Device::CreateMemory(2MB)  → ioctl → KMD 分配 2MB GPU 内存
     创建 ResSegment(base=VA, size=2MB)
     再 SubAllocate (从新 Chunk 中切)
```

### Step 7: 返回给用户

```
Core 层:
  *dptr = pMemory->GetDevicePointer()
        = m_pHalMemory->GetDeviceVirtualAddress() + m_Offset
        = Chunk 的起始 VA + 4KB 偏移
        = 0x10000000 + 0x1000
        = 0x10001000

MemoryTracker 注册:
  (0x10001000, 4096) → shared_ptr<Memory>  ← 后续 muMemFree 时通过这个反查

返回给用户: *dptr = 0x10001000
```

### 完整流程图

```
muMemAlloc(&dptr, 4096)
  │
  ▼
Driver: muapiMemAlloc_v2
  ├─ InitPlatform (首次: Hal::CreatePlatform → 枚举GPU设备)
  ├─ TlsCtxTop (获取线程的 GPU Context)
  └─ CreateMemory(type=General, flags=Virtual|DeviceMapped|SubAllocatable)
      │
      ▼
Core: Context::CreateMemory
  ├─ new Memory(this)
  ├─ Memory::Init → GeneralAlloc
  │   │
  │   ├─ 属性推导: flags → property (Physical|SharedVA|Virtual|...)
  │   │
  │   └─ [SubAllocatable?] → MemMgr::Allocate
  │       │
  │       ├─ Key 编码 → SplayTree 查找 Pool
  │       │
  │       └─ Pool::FullAllocate
  │           ├─ SubAllocate: 从 Chunk 切 (O(1) bucket查找)
  │           │   └─ ResourceSplit: 分裂段 → 插入 SegmentTracker
  │           │
  │           └─ [NotFound?] ChunkAllocate: ioctl → KMD → 2MB
  │               └─ 再 SubAllocate
  │
  ├─ MapToPeers (如果启用了 Peer Access)
  ├─ AddMemory (加入 Context 追踪表)
  ├─ TrackMemory (全局 MemoryTracker 注册)
  └─ 返回 *dptr = GetDevicePointer() = VA + offset
```

---

## 第三章：跟着一次 muMemFree 走完全程

分配了内存，必然要释放。`muMemFree(dptr)` 比分配更复杂——因为它需要先通过指针反查 Memory 对象，再等待 GPU 完成所有操作，最后决定归还给 Pool 还是销毁。

### Step 1: 用户调用

```c
muMemFree(dptr);  // dptr = 0x10001000 (之前 muMemAlloc 返回的)
// 成功返回 0
```

### Step 2: Driver 层 — 反查 + 类型校验 (`mu_memory.cpp:709`)

```
muapiMemFree_v2(dptr):

① InitPlatform()  — 确保 MUSA 已初始化

② 反查 Memory 对象:
   pMemory = Platform::Get().GetMemoryTracker().FindRange(dptr, &offset)->get()
   // MemoryTracker 是一个区间树 (range-based map)
   // 输入: dptr=0x10001000
   // 输出: offset=0 (相对于 Chunk 开始), pMemory=该内存的 Memory 对象

③ 类型校验 (只允许释放这些类型):
   if (type != General && type != PitchedGeneral &&
       type != Managed && type != External &&
       type != Virtual):
       → INVALID_VALUE
   
   if (type == Virtual && pool == nullptr):
       → INVALID_VALUE  ★ Virtual 必须属于一个 Pool

④ offset 校验:
   if (offset != 0):
       → INVALID_VALUE
       ★ Sub-allocation 的中间片段不能单独释放
       ★ 必须释放从 offset=0 开始的完整 Memory 对象

⑤ 同步等待:
   pMemory->Synchronize()
   // 等待该内存上所有未完成的 GPU 操作
   // 包括: 本设备的所有 stream + 所有 peer 设备上映射了此内存的 context

⑥ 按类型分发:
   if (type == Virtual):
       // 走 Stream 异步释放 (因为 Virtual 内存有 PagingCommand 绑定)
       pStream->CmdMemFree(dptr, true)
       → DisableAccess (禁用 peer)
       → CallbackCommand (GPU 完成后的回调中真正清理)
   else:
       // General/Pitched/Managed/External: 直接销毁
       pMemory->GetContext()->DestroyMemory(pMemory)
```

### Step 3: Memory::Synchronize — 等待 GPU (`memory.cpp:115`)

```
Memory::Synchronize():

① 判断内存类型:
   if (m_Type == Virtual):
       // Virtual 内存绑定了多个物理块 → 逐个同步
       for each physical memory in PhysTracker:
           physical->GetContext()->LockedWait()
           // LockedWait: 等待该 context 下所有 stream 的所有 inflight 命令
   else:
       // 1. 等待自己的 context
       m_Context->LockedWait()
       
       // 2. 等待所有 peer device 上映射了此内存的 context
       for (devId = 0; devId < deviceCount; devId++):
           if (peerDev && m_pHalMemory->GetPeerMemory(&peerDev->Hal())):
               for each peerCtx in peerDev:
                   peerCtx->Synchronize()
                   // ★ 确保没有 peer 设备还在访问这块内存
```

**为什么要同步所有 peer？**
```
场景: Device 0 分配了内存, Device 1 通过 Peer Access 也在使用它

Device 0:                Device 1:
  muMemAlloc → dptr       muCtxEnablePeerAccess(dev0)
  kernel_A(dptr)           kernel_B(dptr)  ← 同一块物理内存!
  muMemFree(dptr)          ...

如果在 Device 0 释放时只等待 Device 0 的 stream:
  → kernel_B 可能还在 Device 1 上执行
  → 物理内存被释放 → kernel_B 访问已释放内存 → GPU page fault!

所以必须等待所有 peer device 上都完成。
```

### Step 4: Context::DestroyMemory — 注销追踪 (`context.cpp:967`)

```
Context::DestroyMemory(pMemory):

① RemoveMemory: 从 context 的 m_Memories set 中删除
② UntrackMemory: 从全局 MemoryTracker 删除 (dptr→Memory 映射)
③ 返回 (不 delete! Memory 是 shared_ptr 管理的)
```

**关键设计**: `DestroyMemory` 不直接 delete Memory 对象。Memory 用 `shared_ptr` 管理，只有当所有引用（Context、MemoryTracker、Stream 命令等）都释放后，引用计数归零，析构函数才真正执行。

### Step 5: Memory 析构 — 归还或销毁 (`memory.cpp:358`)

```
~Memory():

① 解除 CPU 映射:
   if (m_pMapped):                  // 如果调用过 GetHostPointer()
       m_pHalMemory->Unmap()        // 解除 mmap

② 如果不是 Prealloc 类型:
   if (m_Type != memoryTypePrealloc && m_pHalMemory):

       ③ ★ 核心分支: SubAllocatable vs 独立分配
       if (m_pHalMemory->GetProps() & Hal::memoryPropertySubAllocatable):
           // Sub-allocated: 归还给 Pool
           if (m_pPool == nullptr):
               // 从 MemMgr 的 Pool 中分配的
               m_Context->GetParentDevice()->Hal().GetMemMgr()
                   ->Free(m_pHalMemory, GetDevicePointer(), GetSize())
               // → Pool::Free → 合并相邻段 → 插入 Bucket → lazyFree 检查
           else:
               // 从用户 Pool 中分配的
               m_pPool->Hal()->Free(m_pHalMemory, GetDevicePointer(), GetSize())
               // → 同上
       else:
           // 独立分配: 直接销毁
           m_PhysTracker.Cleanup()
           m_pHalMemory->Destroy()
           // → M3D::Memory::Destroy → m3dDevice->DestroyGpuMemory()
           // → ioctl → KMD 释放 VRAM 物理页 + GPU VA
```

**两种释放路径对比**:

```
SubAllocatable 路径 (默认, muMemAlloc):
  ~Memory()
    └─ MemMgr::Free(pHalMem, dptr, size)
        └─ Pool::Free
            ├─ SegmentTracker.find(range)  ← 找到要释放的段
            ├─ 左邻合并 (如果左边是空闲段)
            ├─ 右邻合并 (如果右边是空闲段)
            ├─ LazyFree 检查 (整个 Chunk 空闲?)
            │   ├─ YES (超限) → Destroy Chunk → ioctl 归还 GPU
            │   └─ NO  → 保持在 Bucket 中, 等待下次 SubAllocate 复用
            └─ FreeListInsert  (重新插入空闲链表)

直接分配路径 (特殊场景, 如 Managed/External):
  ~Memory()
    └─ m_pHalMemory->Destroy()
        └─ M3D::Destroy() → m3dDevice->DestroyGpuMemory()
            └─ ioctl → KMD 立即回收 VRAM 物理页
```

### Step 6: muMemFreeAsync — 异步释放 (`mu_memory.cpp:386`)

```
muMemFreeAsync(dptr, hStream):

① 反查 + 类型校验 (同 muMemFree)

② 按类型分发:
   ┌────────────────────────────────────────────────────┐
   │ IPC/External:                                      │
   │   → DestroyMemory() 直接销毁 (不涉及同步)            │
   │                                                     │
   │ General/Pitched/Managed:                            │
   │   → Synchronize() → DestroyMemory()                 │
   │   ★ 同步等待 + 同步销毁                             │
   │                                                     │
   │ Virtual (from pool):                                │
   │   → pStream->CmdMemFree(dptr, false)                │
   │   ★ 异步入队!                                       │
   │   ┌─────────────────────────────────────────────┐  │
   │   │ AsyncMemFree(virtAddr, blocking=false):     │  │
   │   │  ① DisableAccess — 禁用所有 peer 设备访问    │  │
   │   │  ② new CallbackCommand(lambda:              │  │
   │   │       virt->DestroyPhysMemories()           │  │
   │   │       pool->DestroyMemory(virt)             │  │
   │   │     )                                       │  │
   │   │  ③ QueueCommand → Stream 队列                │  │
   │   │                                             │  │
   │   │  ★ Callback 在 stream 上前序命令完成后执行   │  │
   │   │  ★ 保证不会在有 GPU 操作访问时释放内存       │  │
   │   └─────────────────────────────────────────────┘  │
   └────────────────────────────────────────────────────┘
```

### muMemFree 完整流程图

```
muMemFree(dptr=0x10001000)
  │
  ▼
Driver: muapiMemFree_v2
  ├─ MemoryTracker::FindRange(0x10001000) → Memory*, offset=0
  ├─ 类型校验: type ∈ {General, Pitched, Managed, External, Virtual}
  ├─ offset=0 校验: sub-allocation 不能从中间释放
  ├─ Synchronize():
  │   ├─ 等待自己的 context (所有 stream 完成)
  │   └─ 等待所有 peer device 的 context
  │
  └─ 按类型分发:
      │
      ├─ [Virtual] → Stream::CmdMemFree(dptr, true)  // 同步异步释放
      │
      └─ [General/Pitched/Managed/External]:
          │
          ▼
          Context::DestroyMemory
            ├─ RemoveMemory (从 context 删除)
            └─ UntrackMemory (从 MemoryTracker 删除)
                │
                ▼  (shared_ptr 引用计数归零时)
          ~Memory()
            ├─ [SubAlloc?] → MemMgr::Free → Pool::Free
            │   ├─ 左合并 → 右合并
            │   ├─ LazyFree? → 保留 Chunk 或 Destroy
            │   └─ FreeListInsert
            │
            └─ [Direct] → m_pHalMemory->Destroy()
                └─ M3D::DestroyGpuMemory → ioctl → KMD 回收
```

### muMemFree vs muMemFreeAsync 对比

```
                muMemFree              muMemFreeAsync
──────────────────────────────────────────────────────
参数            dptr                    dptr + hStream
等待 GPU        同步 (调用线程阻塞)      异步 (通过 Stream)
Virtual 处理    CmdMemFree(true)        CmdMemFree(false)
                同步执行 Callback       异步入队 Callback
General 处理    Synchronize +           同左 (同步)
                DestroyMemory          
适用场景        独立分配                 从 Pool 的 Async 分配
```

### 关键设计要点

| 要点 | 为什么 |
|------|--------|
| **先反查再操作** | MemoryTracker 是全局的 (dptr→Memory), 任何 API 只要传 dptr 都先通过它找到 Memory 对象 |
| **Synchronize 等所有 peer** | 防止在 peer 设备还在使用时释放物理内存 |
| **offset≠0 不能释放** | Sub-allocation 的最小单元是整个 Memory 对象, 不能只释放中间一段 |
| **shared_ptr 延迟析构** | Stream 命令可能还持有对 Memory 的引用, 必须等所有引用释放后才真正回收 |
| **LazyFree** | 归还给 Pool 的 Chunk 不一定立即销毁, 留给后续分配复用 |
| **Virtual 必须异步释放** | Virtual 内存通过 PagingCommand 绑定物理页, 释放也必须在 Stream 上按序执行 |

---

## 第四章：9 种内存类型 — 它们解决什么问题？

MUSA 支持 9 种内存类型，但它们不是 9 种完全不同的东西 —— 它们只是"属性组合 + 分配策略"的不同。

```
类型              用途              虚拟地址    物理存储    来源
────────────────────────────────────────────────────────────────
General          普通 GPU 内存      从 Pool 切   从 Pool     KMD
PitchedGeneral   2D/3D 对齐内存     从 Pool 切   从 Pool     KMD
PinnedHost       CPU 锁定内存       GPU VA       Host RAM   KMD+mmap
Registered       用户已分配的        GPU VA       Host RAM   锁定已有
PinnedHost       Host 内存                                    内存
Managed          CPU+GPU 统一       GPU VA       Host RAM   KMD
                 内存 (自动迁移)                 或 VRAM
IpcImport        其他进程导出的      已有         已有        IPC fd
                 共享内存
External         外部句柄导入       已有         已有        DMA-BUF
Prealloc         预分配 VA          已有         已有        保留
Virtual          纯虚拟地址         Pool VA      无           Pool
                 (无物理存储)

关键理解:
- 只有 General/PitchedGeneral 是"标准 GPU 显存分配"
- Virtual 是"纯地址空间分配"(无物理存储), 用于 Async 分配
- PinnedHost 是"CPU 内存 + GPU 可见性"
- IPC/External 是"借用别人的内存"
```

### 类型之间的关系

```
muMemAlloc        → General        (最常用)
muMemAllocPitch   → PitchedGeneral (带 pitch 对齐)
muMemHostAlloc    → PinnedHost     (CPU 内存, GPU 可访问)
muMemHostRegister → Registered     (用户已有的内存)
muMemAllocManaged → Managed        (自动迁移)
muMemAllocAsync   → Virtual        (地址) + General (物理) = 两步
muIpcOpenMemHandle→ IpcImport      (其他进程的)
外部导入           → External      (DMA-BUF 等)
VMM Reserve       → Virtual        (纯地址)
```

---

## 第五章：两种分配模式 — 同步 vs 异步

### 4.1 同步分配 (muMemAlloc)

```
muMemAlloc(&dptr, 4096):
  ┌────────────────────────────────────┐
  │ 一次调用完成:                       │
  │ ① Pool 中切出 VA                   │
  │ ② 获得物理存储 (Pool已有或新Chunk)  │
  │ ③ 本地 MMU 页表填充                │
  │ ④ Peer 设备页表填充                │
  │                                    │
  │ 返回时 *dptr 立即可用               │
  │ 不涉及 Stream                      │
  └────────────────────────────────────┘

内存结构:
┌─────────────────────┐
│ VA + PA (合一)       │  ← 一个 Memory 对象
│ m_Type = General     │
│ m_pPool = nullptr    │  ← 不属于用户 Pool
└─────────────────────┘
```

### 4.2 异步分配 (muMemAllocAsync)

```
muMemAllocAsync(&dptr, 4096, hStream):
  ┌────────────────────────────────────────────────────┐
  │ 分三步:                                            │
  │                                                    │
  │ ① Pool 中切出 VA (同步)                             │
  │    创建 Virtual Memory 对象                         │
  │    m_Type = Virtual, m_pPool = pool                 │
  │                                                    │
  │ ② 独立分配 PA (同步)                                │
  │    创建 General Memory 对象                         │
  │    m_Type = General                                 │
  │                                                    │
  │ ③ Bind + PagingCommand 入队 (异步)                  │
  │    virt->Bind(physical)  — 建立 VA→PA 映射          │
  │    CmdPaging → 页表设置命令入队到 Stream             │
  │    ★ 用户 API 立即返回, 不等待页表设置完成           │
  │                                                    │
  │ ④ 页表在 Stream 上异步完成:                          │
  │    AsyncSubmit 线程 → PagingCommand::Submit()       │
  │    → HostWaitSemaphores (等前序命令完成)             │
  │    → MMU->Paging (填充页表)                         │
  └────────────────────────────────────────────────────┘

内存结构:
┌──────────────┐     Bind     ┌──────────────┐
│ Virtual (VA) │ ← ─ ─ ─ ─ → │ General (PA) │
│ m_Type=Virt  │              │ m_Type=Gen   │
│ m_pPool=pool │              │              │
└──────────────┘              └──────────────┘
         │
         │ PhysTracker: [VA→PA mapping]
         │
         └─ CmdPaging (异步) → 填充 MMU 页表
```

### 4.3 为什么需要 Async 模式？

```
场景: 在 Stream 上有序操作

Stream: [kernel_A] → [kernel_B] → [kernel_C 需要新内存]

如果用 muMemAlloc:
  kernel_C 需要的 dptr 在分配时就完全就绪
  但 kernel_A, kernel_B 可能还没完成
  → 语义上没问题 (dptr 立即可用)
  → 但物理内存可能被其他 Stream 也访问到 (没有顺序保证)

如果用 muMemAllocAsync:
  ① dptr (VA) 立即返回 (立即可用)
  ② 物理绑定作为 PagingCommand 入队到 Stream
  ③ PagingCommand 依赖于前面的 kernel_B 完成
  ④ 只有 kernel_B 完成后, 才真正设置页表
  → ★ 页表设置与 Stream 上的其他操作有明确的顺序关系
  → ★ 物理内存在页表设置完成前不会被其他 Stream 访问
```

---

## 第六章：MemoryPool 如何工作

### 5.1 数据结构全景

```
┌────────────────────────────────────────────────────────────┐
│                      MemoryPool                             │
│                                                            │
│  m_FreeBuckets[0..63]  ← 64 个 bucket 的头指针              │
│    Bucket[i] 存的是 size ∈ [2^i, 2^(i+1)) 的空闲段          │
│                                                            │
│  m_EltMappingHash (64-bit)                                │
│    bit[i]=1 表示 Bucket[i] 非空 → O(1) 判断哪些 bucket 可用 │
│                                                            │
│  m_pHeadSegment  →  ResSegment 双向链表头                   │
│    所有段 (busy+free) 按地址序排列                           │
│                                                            │
│  m_SegmentTracker  →  std::map<Range, ResSegment*>         │
│    区间树: Free 时 O(log n) 查找已分配的段                   │
│                                                            │
│  每个 Chunk:                                                │
│  ┌──────────────────────────────────────┐                  │
│  │ ResSegment A │ ResSegment B │ Res C │  ...              │
│  │ (busy/free)  │ (busy/free)  │       │                   │
│  └──────────────────────────────────────┘                  │
│   ↑ 地址序双向链表 (pNextSegment/pPrevSegment)               │
│   ↑ 空闲段还挂在对应 bucket 下 (pNextFree/pPrevFree)        │
└────────────────────────────────────────────────────────────┘
```

### 5.2 一个 Chunk 从创建到销毁的生命周期

```
① ChunkAllocate: ioctl → KMD → 2MB GPU 内存 → 创建 ResSegment
                     ┌──────────────────────┐
                     │   2MB (free, bucket 21)│
                     └──────────────────────┘

② SubAllocate(256KB): 从 2MB 切出 256KB
   ┌──────────┬─────────────────────────┐
   │ 256KB    │    1.75MB (free,B21)     │
   │ busy     │                          │
   └──────────┴─────────────────────────┘

③ SubAllocate(512KB): 从 1.75MB 切出 512KB
   ┌──────┬──────┬──────────────────────┐
   │256KB │512KB │   1.25MB (free,B21)   │
   │busy  │busy  │                       │
   └──────┴──────┴──────────────────────┘

④ Free(256KB段): 左合并(无) → 右合并(512KB邻段 busy, 不合并)
   ┌──────┬──────┬──────────────────────┐
   │256KB │512KB │   1.25MB (free,B21)   │
   │free  │busy  │                       │
   └──────┴──────┴──────────────────────┘
   移入 Bucket[18] (Log2(256KB)=18)

⑤ Free(512KB段): 左合并(256KB free!) + 右合并(1.25MB free!)
   ┌────────────────────────────────────┐
   │            2MB (free, B21)          │
   │            isLeftMost=true          │
   │            isRightMost=true         │  ← 整个 Chunk 都空闲了!
   └────────────────────────────────────┘

⑥ Lazy Free 检查:
   lazyFreeCount = 1  (第一次整个 Chunk 空闲)
   if (1 > UINT64_MAX):  ← NO (配置为无限)
   → 保留 Chunk, 等待下次分配时复用

⑦ 后续 SubAllocate: 直接从 Bucket[21] 找到这个 2MB 段, 切分使用
   ★ 不需要再调 ioctl!
```

---

## 第七章：从内存分配到内存释放的完整生命周期

```
时间线:
──────

① muInit()
   → Platform::Init() → Hal::CreatePlatform() → 枚举 GPU 设备
   → 创建 Default Context + Default Stream
   → 创建 MemMgr (内存管理器)
   → 创建初始 MemoryPool

② muMemAlloc(&dptr_A, 4096)
   → Pool::FullAllocate → [Pool 空] → ChunkAllocate(2MB) → ioctl
   → SubAllocate(4KB) → dptr_A = Chunk_VA + 0

③ muMemAlloc(&dptr_B, 1MB)
   → Pool::FullAllocate → SubAllocate(1MB) → dptr_B = Chunk_VA + 4KB

④ muLaunchKernel(kernel, dptr_A, dptr_B, stream)
   → DispatchCommand → stream 队列 → GPU 执行

⑤ muMemcpy(dptr_C, host_data, 8MB, stream)
   → CopyManager 选择引擎 → AsyncMemcpyCommand → GPU DMA

⑥ muMemFree(dptr_A)
   → MemoryTracker::FindRange(dptr_A) → Memory*
   → Synchronize() (等待 GPU 完成)
   → Context::DestroyMemory → RemoveMemory + UntrackMemory
   → ~Memory() → [SubAlloc] MemMgr::Free(dptr_A) → Pool::Free
     → 合并相邻段 → Bucket 重新插入

⑦ muMemFree(dptr_B)
   → 同上 → Pool::Free → 合并 → 整个 Chunk 空闲
   → LazyFree: 保留 Chunk

⑧ muMemPoolTrimTo(pool, 0)
   → TrimPool(0): 遍历所有 bucket, 释放所有完整空闲 Chunk
   → Chunk->Destroy() → ioctl → KMD 回收 2MB

⑨ muCtxDestroy()
   → 清理所有 Memory, Stream, Event
   → ~MemMgr → 销毁所有 Pool → 回收所有 Chunk
```

---

## 第八章：完整的数据流图

```
                         ┌──────────────┐
                         │  用户程序     │
                         │ muMemAlloc() │
                         └──────┬───────┘
                                │
                    ┌───────────▼───────────┐
                    │  Driver Layer          │
                    │  mu_memory.cpp:265     │
                    │  - 参数校验             │
                    │  - type=General        │
                    │  - flags=Virtual|DM|SA │
                    └───────────┬───────────┘
                                │ CreateMemory
                    ┌───────────▼───────────┐
                    │  Core Layer            │
                    │  context.cpp:915       │
                    │  - new Memory          │
                    │  - Init→GeneralAlloc   │
                    │  - MapToPeers          │
                    │  - TrackMemory         │
                    └───────────┬───────────┘
                                │ SubAllocatable?
                    ┌───────────▼───────────┐
                    │  MemMgr                │
                    │  memMgr.cpp:81         │
                    │  - Key 编码            │
                    │  - SplayTree 查找 Pool │
                    │  - FullAllocate        │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  MemoryPool (HAL)      │
                    │  memoryPool.cpp:82     │
                    │  - Bucket 查找         │
                    │  - FindBucket          │
                    │  - ResourceSplit       │
                    │  - 或 ChunkAllocate    │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  KMD (内核驱动)        │
                    │  DRM ioctl             │
                    │  - 分配 VRAM 物理页    │
                    │  - 分配 GPU VA 空间    │
                    │  - 填充 GPU MMU 页表   │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  GPU 硬件              │
                    │  - VRAM 物理存储       │
                    │  - MMU 页表            │
                    │  - DMA 引擎            │
                    │  - 计算单元 (CDM)      │
                    └───────────────────────┘
```

---

## 第九章：文档导航 — 如何深入每个子系统

现在你对整体机制有了清晰的理解。如果想深入某个具体子系统，按以下地图导航：

| 我想了解... | 看这个文档 |
|------------|----------|
| 整体架构和文件索引 | [memory_analysis.md](memory_analysis.md) |
| 每个 API 的逐行代码流程 | [memory_api_deep_analysis.md](memory_api_deep_analysis.md) |
| muMemAlloc vs muMemAllocAsync 的本质差异 | [muMemAlloc_vs_muMemAllocAsync.md](muMemAlloc_vs_muMemAllocAsync.md) |
| MemoryPool 数据结构和算法细节 | [mempool_design.md](mempool_design.md) |
| 所有池化机制 (15种) | [pooling_analysis.md](pooling_analysis.md) |
| 每个 if/switch 的判断逻辑 | [decision_logic.md](decision_logic.md) |
| Stream + Command 子系统 | [stream_command_analysis.md](stream_command_analysis.md) |
| MemoryPool 的 Bucket 分配算法 | [mempool_design.md §3](mempool_design.md) |
| SubAllocate 的具体数值例子 | [mempool_design.md §3.2](mempool_design.md) |
| Free + 合并的具体例子 | [mempool_design.md §5](mempool_design.md) |

---

*本文档基于 musa 项目源码分析生成，commit: 9ba99a5d, branch: bugfix/sw-79049*
