# muMemAlloc vs muMemAllocAsync 深度剖析

> **📖 相关文档**: [memory_analysis.md](memory_analysis.md) (架构总览) | [memory_api_deep_analysis.md](memory_api_deep_analysis.md) (逐API流程) | [pooling_analysis.md](pooling_analysis.md) (池化) | [stream_command_analysis.md](stream_command_analysis.md) (Stream) | [decision_logic.md](decision_logic.md) (决策分支)

> 为理解以下分析，需先掌握两个基础概念。若不熟悉，请先阅读 **[附录A：GPU内存模型](#附录agpu内存模型基础)** 和 **[附录B：Stream命令执行模型](#附录bstream命令执行模型基础)**。

---

## 一、muMemAlloc：同步分配全路径

`muMemAlloc` 调用时，**不涉及任何 stream**，不需要任何 stream 句柄作为参数。所有操作在调用线程上同步执行完毕才返回。

### 1.1 完整调用链

```
用户调用 muMemAlloc(dptr, bytesize)
  │
  └─ muapiMemAlloc_v2 (mu_memory.cpp:265)
      │
      │  // 构建 MemoryCreateInfo，flags 包含三项关键标志:
      │  //   Virtual | DeviceMapped | SubAllocatable
      │  //   含义：这会是"虚拟+物理"合一的内存，设备立即可访问
      │
      └─ pContext->CreateMemory(&pMemory, createInfo)
          └─ Context::CreateMemory (context.cpp:915)
              │
              │  ★ 步骤1: 分配虚存+物理内存 (一次HAL调用完成)
              │
              ├─ Memory::Init(createInfo)
              │   └─ GeneralAlloc(size, alignment, flags)
              │       │
              │       │  // 构建 HAL 层 createInfo，合并 flags：
              │       │  //   输入 flags: Virtual | DeviceMapped | SubAllocatable
              │       │  //   最终 property:
              │       │  //     Physical                 (必有，表示含物理存储)
              │       │  //     SharedVirtualAddress     (必有)
              │       │  //     DeviceVisible            (Virtual flag 触发)
              │       │  //     HostVisible | HostCoherent (Virtual flag 触发)
              │       │  //     DeviceWriteable | DeviceCached (Virtual flag 触发)
              │       │  //   最终 viewCapability:
              │       │  //     Exportable | PeerAccessible | IpcExportable
              │       │  //
              │       │  //   ★ 关键：createInfo.alloc.property 同时
              │       │  //     包含 Physical 和 Virtual 相关属性，
              │       │  //     这意味着 HAL 层会分配一块同时具备
              │       │  //     "物理存储 + GPU虚拟地址映射" 的内存
              │       │
              │       └─ [SubAllocatable flag 存在]
              │           → MemMgr->Allocate(info, &offset, &pHalMemory)
              │             │
              │             │  // MemMgr 是内存管理器，通过内存池 (MemoryPool) 管理
              │             │  // 如果该属性组合的池不存在则创建，然后
              │             │  // FullAllocate → [SubAllocate 或 ChunkAllocate then SubAllocate]
              │             │  //
              │             │  // ChunkAllocate 最终调用:
              │             │  //   Device→CreateMemory → Hal::Memory::Init
              │             │  //     → InitGeneralDeviceMemory
              │             │  //       → m3dDevice->CreateGpuMemory()
              │             │  //         ★ 内核 ioctl 调用：
              │             │  //           1. 分配 GPU VRAM 物理页
              │             │  //           2. 分配 GPU 虚拟地址范围
              │             │  //           3. 填充 GPU MMU 页表 (VA→物理页映射)
              │             │  //         ★ 这是一次同步阻塞的内核调用
              │             │
              │             └─ 返回 pHalMemory + offset
              │                         (pHalMemory->GetDeviceVirtualAddress() 就是 GPU 虚址)
              │
              │  ★ 步骤2: 跨设备 peer 映射 (同步HAL调用)
              │
              ├─ MapToPeers(pMemory)    ← 仅当内存支持 PeerAccessible 时执行
              │   └─ CriticalBase::MapToPeers (context.cpp:483)
              │       │
              │       │  // 对于 memoryTypeGeneral:
              │       │  //   peerCount = m_Peers.size()
              │       │  //   (context enable peer access 时记录的其他设备列表)
              │       │
              │       └─ for each peer device:
              │           └─ memory->Hal()->OpenPeerMemory(&remoteDevice->Hal(), openInfo)
              │               │
              │               │  ★ HAL层同步调用，内部流程：
              │               │   1. 从原始内存导出全局句柄 (ExportExternalHandle)
              │               │      → 内核生成唯一 global handle
              │               │   2. 在 peer 设备上导入该句柄 (ImportMemory)
              │               │      → ioctl: 获取 peer 设备的 buffer handle
              │               │   3. 映射到 peer 设备的 GPU 虚址空间
              │               │      → 分配 peer 设备 SVM 虚拟地址
              │               │      → pDevice->MapVirtualAddress()
              │               │        ★ 编程 peer 设备的 IOMMU/MMU 页表，
              │               │          指向原始内存的物理页
              │               │      ★ 整个调用同步阻塞直到页表编程完成
              │               │   4. 缓存 peer memory 对象供后续使用
              │               │
              │               └─ break on error (任何 peer 映射失败则停止)
              │
              ├─ ctxCrit->AddMemory(pMemory)   // 记录到 context 内存管理表
              └─ MemoryTracker.TrackMemory()   // 全局内存追踪器登记
              └─ *ppMemory = pMemory
      │
      └─ *dptr = pMemory->GetDevicePointer()  // m_pHalMemory->GetDeviceVirtualAddress() + m_Offset
      └─ return MUSA_SUCCESS
          ★ 此时所有操作均已同步完成，*dptr 可立即用于任何 GPU 操作
```

### 1.2 同步特性明细

| 阶段 | 做了什么事 | 怎么做的 |
|------|-----------|---------|
| **虚存+物理内存分配** | 一次 HAL 调用创建同时含 Virtual+Physical 属性的内存块 | `MemMgr->Allocate` → `Device->CreateMemory` → ioctl 分配 VRAM 物理页 + 虚址 + MMU 页表 |
| **跨设备映射** | 对每个已启用 peer access 的设备建立页表映射 | `MapToPeers` 中循环 `OpenPeerMemory` → ioctl 设置 peer 设备 MMU 页表 |
| **调用返回时的状态** | 虚存+物理+本地页表+所有 peer 页表全部就绪 | 所有 ioctl 调用均同步完成 |

**核心结论：`muMemAlloc` 从不经过任何 stream，从不创建任何 command 对象，从不调用 `ResolveDependencyAndQueueCommand`。所有操作都是直接的 HAL 层同步调用（最终是内核 ioctl），在调用线程上阻塞直到完成。**

---

## 二、muMemAllocAsync：异步分配全路径

`muMemAllocAsync` 需要一个 `MUstream hStream` 参数。内存分配本身同步完成，但跨设备的页表设置通过 stream 命令队列异步入队。

### 2.1 完整调用链

```
用户调用 muMemAllocAsync(dptr, bytesize, hStream)
  │
  └─ muapiMemAllocAsync (mu_memory.cpp:303)
      │
      │  // 1. 校验参数，获取 stream 对象
      │  pStream = Context::InfoStream(pContext, hStream)
      │
      │  // 2. 调 stream 的分配方法，blocking=false (非阻塞)
      └─ pStream->CmdMemAlloc(memAllocParam, false)
          └─ Stream::CmdMemAlloc (stream.cpp:570)
              │
              ├─ [stream capture active]   → 图捕获模式，建图节点，此处不展开
              ├─ [stream capture invalid]  → 返回错误
              │
              └─ [正常模式] → AsyncMemAlloc(param, blocking=false)
                  └─ Stream::AsyncMemAlloc (stream.cpp:519)
                      │
                      │  ═══════════ 以下均为同步操作 ═══════════
                      │
                      │  ★ 步骤A: 从内存池分配虚拟内存
                      │
                      ├─ pPool = param.pool ? param.pool : device->GetMemoryPool()
                      ├─ pPool->CreateMemory(&virt, &virtAddr, allocSize)
                      │   └─ MemoryPool::CreateMemory (memoryPool.cpp:374)
                      │       ├─ Memory::InitFromPool(pPool, size)
                      │       │   └─ pHalPool->FullAllocate(allocInfo, &offset, &pHalMemory)
                      │       │       │
                      │       │       │  // 从池的 free-list 找可用段
                      │       │       │  // 找不到则 ChunkAllocate(新分配一个大块如2MB)
                      │       │       │  // 再从中 SubAllocate 一块
                      │       │       │  // 返回的 pHalMemory 是 virtual-only (无 physical 属性)
                      │       │       │  ★ 这是 HAL 层同步调用
                      │       │       │
                      │       │       └─ m_pPool = pPool;  // 标记此内存归属该池
                      │       └─ *ptr = virt->GetDevicePointer()  // 虚址立即可用
                      │
                      │  ★ 步骤B: 分配独立的物理内存
                      │
                      ├─ physical = new Memory(ctx)
                      ├─ physical->Init(createInfo)
                      │   └─ GeneralAlloc(allocSize, 0, 0)
                      │       │
                      │       │  // flags=0 → 最终的 property 为:
                      │       │  //   Physical | SharedVirtualAddress
                      │       │  //   (不含 Virtual flag, 所以不含
                      │       │  //    DeviceVisible/HostVisible 等)
                      │       │  //   这是一块"纯物理"内存
                      │       │  ★ HAL 层同步调用
                      │       │
                      │       └─ MemMgr->Allocate / Hal->CreateMemory
                      │
                      │  ★ 步骤C: 绑定虚拟内存到物理内存
                      │
                      └─ virt->Bind(physical, allocSize, 0, 0)
                          └─ Memory::Bind (memory.cpp:152)
                              │
                              │  // 仅 memoryTypeVirtual 才能作为 bind 目标
                              │  // 检查 VA 范围内无重叠绑定
                              │
                              ├─ virt.m_pHalMemory->SetProps(memoryPropertyPhysical)
                              │   // 标记：此虚拟内存现在有物理存储
                              │
                              ├─ physical.m_pHalMemory->SetProps(memoryPropertyVirtual)
                              │   // 标记：此物理内存已被虚拟映射
                              │
                              └─ virt.m_PhysTracker.TrackMemory(ptr, size, physical)
                                  // 在虚拟内存的 PhysTracker 中记录:
                                  //   VA范围 → 对应的物理内存对象
                                  ★ 纯数据结构操作，同步完成
                      │
                      │  ═══════════ 异步部分从此开始 ═══════════
                      │
                      └─ pPool->ModifyAccess(virt, physical, allocSize, blocking=false, stream)
                          └─ MemoryPool::ModifyAccess (memoryPool.cpp:295)
                              │
                              │  ★ 遍历池的 m_LocationAccessMap
                              │     (池的跨设备访问权限表，记录了哪些设备对
                              │      此池中的内存具有什么访问权限)
                              │
                              ├─ for each (deviceId, flags) in m_LocationAccessMap:
                              │   ├─ if deviceId == 本地设备: skip
                              │   ├─ if flags == PROT_NONE: skip
                              │   │
                              │   └─ physical->Hal()->OpenPeerMemory(&remoteDev->Hal(), openInfo)
                              │       │
                              │       │  ★ 注意：OpenPeerMemory 本身是同步 HAL 调用
                              │       │  它完成的只是"打开 peer" (建立跨设备引用),
                              │       │  并不是"设置页表让 GPU 可以访问"
                              │       │
                              │       └─ 构建 MemoryPaging 条目:
                              │            { device, virtMem, physMem, offset, size, flags }
                              │
                              └─ if (!pagingParams.memoryPagings.empty()):
                                  └─ stream->CmdPaging(pagingParams, blocking=false)
                                      │          ★★★ 关键分叉点 ★★★
                                      └─ Stream::CmdPaging (stream.cpp:496)
                                          │
                                          ├─ 创建 PagingCommand 对象
                                          │   (engine=mmu, supportMerge=false)
                                          │
                                          └─ m_ParentCtx->ResolveDependencyAndQueueCommand(
                                                  command, stream, blocking=false)
                                              │
                                              └─ Context::ResolveDependencyAndQueueCommand
                                                  (context.cpp:1859)
                                                  │
                                                  ├─ 记录流间依赖
                                                  │   (如果是 default stream → 依赖所有阻塞流)
                                                  │   (如果是阻塞流 → 依赖 default stream)
                                                  │   (barrier stream → 依赖所有流)
                                                  │
                                                  ├─ 记录流上前序命令依赖
                                                  │   command->RecordDependency(前一个cmd)
                                                  │
                                                  ├─ pStream->QueueCommand(command)
                                                  │   → 加入 m_CommandList
                                                  │   → m_SubmitCv.notify_one()
                                                  │     ★ 唤醒 AsyncSubmit 工作线程
                                                  │
                                                  └─ if (blocking || pfmEnabled)
                                                        command->Wait()
                                                      // blocking=false
                                                      // → ★ 跳过 Wait()，立即返回!
                                                      //   调用线程不等待 PagingCommand 完成
```

### 2.2 PagingCommand 何时真正执行？

`PagingCommand` 在 stream 的 **AsyncSubmit 线程** 中执行，流程如下：

```
[AsyncSubmit 线程] (stream.cpp:1108)
  ├─ 从 m_CommandList 取出 PagingCommand
  ├─ buildCommand():
  │   ├─ FilterDependency()          // 移除已完成的依赖
  │   ├─ CanMergeTo() → false        // mmu引擎不支持merge
  │   ├─ 分配 SignalSemaphoreValue
  │   ├─ Command::Build()            // 将依赖转为 timeline semaphore wait
  │   │   └─ 为每个未完成的依赖Cmd添加:
  │   │       (依赖Cmd的signal_semaphore, 依赖Cmd的signal_value)
  │   │       到 m_WaitSemaphoreInfos 列表
  │   │   ★ 这意味着 PagingCommand 必须等待所有依赖命令的 GPU 执行完成
  │   └─ 加入 m_MergingList
  │
  └─ submitMergingList():
      └─ Command::Submit()
          └─ PagingCommand::Submit() (pagingCommand.cpp:14)
              │
              ├─ HostWaitSemaphores()
              │   ★ 在 CPU 端等待所有依赖 semaphore 信号
              │   ★ 即等待前序 GPU 命令全部完成
              │
              ├─ 将 Musa::MemoryPaging 转为 Hal::MemoryPaging
              │   │
              │   │  关键转换：
              │   │    如果是 self-mapping → 从 PhysTracker 解析物理内存
              │   │    否则 → physMem->GetHalMemory(device) 获取 peer 设备的内存视图
              │   │
              │   │  flags 转换：
              │   │    PROT_NONE → memoryPropertyNone (取消映射)
              │   │    PROT_READ → DeviceVisible
              │   │    PROT_READWRITE → DeviceVisible | DeviceWriteable
              │   │
              │   └─ paging.pPhysicalMemory = physMem->GetHalMemory(device)
              │       // 获取的是 OpenPeerMemory 阶段创建的 peer 设备视图
              │
              ├─ m_ParentStream->GetHalQueue(Device::Engine::mmu)->Paging(info)
              │   ★ 向 MMU 引擎提交页表修改请求
              │   ★ 编程目标设备的 IOMMU/MMU 页表，
              │     使目标设备可以通过虚拟地址访问此物理内存
              │   ★ 这是一次主机端同步 HAL 调用
              │
              ├─ HostSignalFinish()
              │   ★ 向 timeline semaphore 写入 signal_value
              │   ★ 通知等待此命令完成的后续命令：可以继续了
              │
              └─ 更新物理内存 property flags
                  (设置/清除 DeviceMapped, DeviceVisible, DeviceWriteable)
```

### 2.3 异步特性明细

| 阶段 | 做了什么事 | 同步/异步 |
|------|-----------|----------|
| **虚拟内存分配** | `pPool->CreateMemory` → 从池的 free-list 分配 | **同步** |
| **物理内存分配** | `physical->Init` → `GeneralAlloc` → HAL 分配 | **同步** |
| **虚存-物理绑定** | `virt->Bind(physical)` → PhysTracker 登记 | **同步** |
| **跨设备 peer 打开** | `OpenPeerMemory` → ioctl 创建 peer 设备引用 | **同步** |
| **跨设备页表设置** | `CmdPaging` → `PagingCommand` 入队 | **异步** |
| **调用返回时状态** | 虚存+物理已分配绑定，OpenPeerMemory 已完成，PagingCommand 已入队但**可能尚未执行** | `*dptr` 立即可用；页表待 stream 调度执行 |

---

## 三、两种分配路径的本质差异对比

### 3.1 内存结构差异

```
muMemAlloc 产生的内存:
┌─────────────────────────────────────────┐
│  一块 MemMgr 管理的内存块                │
│  ├── 虚拟地址 (GPU VA)     ← 一次分配      │
│  ├── 物理页 (VRAM)         ← 同时获得      │
│  ├── 本地 MMU 页表         ← 已填充        │
│  └── Peer 设备 MMU 页表    ← MapToPeers 同步填充 │
│                                             │
│  m_pPool = nullptr  (不属于任何池)           │
│  m_Type = memoryTypeGeneral                 │
└─────────────────────────────────────────┘

muMemAllocAsync 产生的内存:
┌──────────────────────┐    ┌──────────────────────┐
│ Virtual Memory (池中) │    │ Physical Memory      │
│ ├── GPU VA            │    │ ├── 物理页 (VRAM)     │
│ ├── m_pPool = pPool   │    │ ├── m_pPool = nullptr │
│ └── m_Type = Virtual  │    │ └── m_Type = General  │
│                       │    │                       │
│ m_PhysTracker:        │    └──────────────────────┘
│   [VA范围] → physical │             ↑
│                       │         Bind 绑定
└───────────────────────┘
         │
         │ ModifyAccess + CmdPaging (异步)
         ▼
┌─────────────────────────────────────────┐
│  Peer 设备 MMU 页表设置                   │
│  通过 PagingCommand 在 stream 上异步入队  │
│  保证与 stream 上前序/后继 GPU 操作有序   │
└─────────────────────────────────────────┘
```

### 3.2 关键差异汇总

| 维度 | muMemAlloc | muMemAllocAsync |
|------|-----------|----------------|
| **stream 参数** | 无 | 必需 hStream |
| **虚存来源** | 一次 HAL 分配 (虚+物合一) | 从设备内存池分配 (虚存独立) |
| **物理内存** | 同一 HAL 调用获得 | 独立分配，手动 Bind |
| **本地 MMU 填充** | HAL 分配时自动完成 | 虚存分配时完成 (virtual-only) |
| **Peer 打开** | `MapToPeers` 同步 OpenPeerMemory | `ModifyAccess` 中同步 OpenPeerMemory |
| **Peer 页表填充** | 随 OpenPeerMemory 同步完成(导入即映射) | **PagingCommand 异步入队** |
| **命令入队** | 无 | PagingCommand 入队到 stream |
| **流依赖管理** | 无 | 记录 default/barrier/前序命令依赖 |
| **调用线程阻塞** | 阻塞到 HAL ioctl 完成 | 不等待 PagingCommand → 立即返回 |
| **内存释放方式** | `muMemFree` 同步释放 | `muMemFreeAsync` 异步释放 (CallbackCommand 入队) |
| **内存池管理** | 不支持 | 支持池的 trim/reuse/access 策略 |
| **图捕获** | 禁止 (检测到 capture 返回错误) | 支持 (转为图节点) |

### 3.3 阻塞 vs 非阻塞的核心机制

```
                        muMemAlloc                    muMemAllocAsync
                            │                              │
                            ▼                              ▼
                  Context::CreateMemory         pStream->CmdMemAlloc(false)
                            │                              │
                  ┌─────────┴─────────┐         ┌─────────┴─────────┐
                  │ Memory::Init      │         │ AsyncMemAlloc     │
                  │ (HAL ioctl 阻塞)  │         │ (同步部分)         │
                  ├───────────────────┤         ├───────────────────┤
                  │ MapToPeers        │         │ ModifyAccess      │
                  │ (OpenPeerMemory   │         │ ↓                 │
                  │  每个peer同步阻塞) │         │ CmdPaging(false)  │
                  └─────────┬─────────┘         │ ↓                 │
                            │                   │ QueueCommand      │
                            ▼                   │ ↓                 │
                      return                   │ ★ 不调用 Wait()   │
                     (全部完成)                  └─────────┬─────────┘
                                                          ▼
                                                    return
                                                 (PagingCommand 未完成)

                                              ... 稍后在 AsyncSubmit 线程中:

                                              PagingCommand::Submit()
                                              ├─ HostWaitSemaphores() (等前序cmd)
                                              ├─ MMU->Paging(info) (页表修改)
                                              └─ HostSignalFinish() (通知后续cmd)
```

---

## 附录A：GPU内存模型基础

### 虚拟地址 vs 物理内存

GPU 内存管理类似 CPU 虚拟内存：

```
程序看到的 GPU 指针 (MUdeviceptr)
        │
        ▼
    GPU 虚拟地址 (VA)
        │
        │  GPU MMU/IOMMU 页表 翻译
        │
        ▼
    GPU 物理页 (VRAM中实际存储)
```

- **虚拟地址 (Virtual)**：GPU 程序（kernel）通过虚拟地址访问内存。这是连续的、程序可用的地址空间。
- **物理内存 (Physical)**：实际存储数据的 GPU VRAM 物理页。可能不连续，由 MMU 页表连接。
- **绑定 (Bind)**：建立虚拟地址范围到物理内存的映射关系。类似 CPU 的 `mmap`。
- **页表 (Page Table)**：GPU MMU 使用的翻译表，将 VA 翻译为物理页地址。**填充页表 = 让 GPU 能够通过某虚拟地址访问某物理内存**。

### 内存属性标志

关键标志及其互斥关系：

```
memoryPropertyVirtual   (0x1)   ← 有GPU虚拟地址
memoryPropertyPhysical  (0x2)   ← 有物理存储(VRAM)
memoryPropertyDeviceMapped (0x8) ← 设备MMU页表已填充,GPU可访问

一块完整可用的设备内存通常需要三者的组合:
  Virtual | Physical | DeviceMapped

muMemAlloc:         一次分配同时获得 Virtual+Physical+DeviceMapped
muMemAllocAsync:    分别分配 Virtual 和 Physical，绑定后通过
                    PagingCommand 异步设置 DeviceMapped 页表
```

### 内存池 (MemoryPool)

内存池是一个管理设备内存预分配和细粒度分配的机制：

```
MemoryPool (例如 2MB 大块)
├── Chunk 0 (2MB 物理内存)
│   ├── Sub-allocation A (用户A 的 256KB)
│   ├── [free]
│   └── Sub-allocation B (用户B 的 512KB)
├── Chunk 1 (2MB 物理内存)
│   ├── Sub-allocation C ...
│   └── [free]
└── ...

优势:
  - 避免频繁的 HAL 层分配/释放（类似 malloc 的 arena）
  - 支持 trim 收缩 (释放空闲 Chunk)
  - 支持访问策略 (m_LocationAccessMap 控制哪些设备可访问)
```

---

## 附录B：Stream 命令执行模型基础

### Stream 是什么

Stream 是 GPU 操作的命令队列。所有 GPU 操作（kernel启动、memcpy、内存操作等）都通过对 stream 发送命令来执行。

### 两种工作线程

```cpp
// stream Init 时创建 (stream.cpp:978-979)
m_SubmitThread = std::thread(&Stream::AsyncSubmit, this);  // 命令构建+提交线程
m_WaitThread   = std::thread(&Stream::AsyncWait,   this);  // 命令完成等待线程
```

### 命令生命周期

```
用户API
  │
  ├─ 创建 Command 对象
  ├─ ResolveDependencyAndQueueCommand()
  │   ├─ 记录依赖（前序命令、其他流）
  │   ├─ QueueCommand() → m_CommandList.push_back()
  │   │                   → m_SubmitCv.notify_one()
  │   └─ if blocking: command->Wait()  (spin/yield直到完成)
  │
  ▼
AsyncSubmit 线程:
  ├─ 从 m_CommandList 取出命令
  ├─ Build: FilterDependency, 分配signal value, 构建semaphore wait
  ├─ 加入 m_MergingList (可能合并多个命令一起提交)
  └─ Submit: 提交到GPU引擎 (Command::Submit())
  ├─ 移入 m_InflightList
  └─ m_WaitCv.notify_one()
  │
  ▼
AsyncWait 线程:
  ├─ 从 m_InflightList 取出命令
  ├─ 等待 GPU semaphore 信号 (主机端阻塞等待)
  ├─ Postprocess: 处理错误,释放cmd buffer
  ├─ 设置状态: completed / error
  └─ m_AsyncCount-- (解除入队限流)
```

### 依赖管理 (`RecordDependency`)

```cpp
// 三种流间依赖规则 (context.cpp:1869-1893):
default stream    → 依赖所有 blocking stream
blocking stream   → 依赖 default stream
non-blocking      → 不依赖其他流
barrier stream    → 依赖所有流
```

命令 A 依赖命令 B → 在 A 的 `Build()` 中，B 的 timeline semaphore 信号值被加入 A 的等待列表 → A 的 `Submit()` 会先 `HostWaitSemaphores()` 等待 B 的信号 → 保证 B 的 GPU 工作完成后 A 才开始 GPU 工作。

### `Command::Wait()` 的阻塞机制

```cpp
// command.cpp:226
MUresult Command::Wait() {
    Schedule([this] {
        return m_Status == Status::completed || m_Status == Status::error;
    });
    return m_LastError;
}
// Schedule 内部: 循环检查 m_Status，用 spin/yield/sleep 等待
// m_Status 由 AsyncWait 线程在 GPU 完成后设置
```

---

## 源码索引

| 文件 (相对 musa/src/) | 行号 | 内容 |
|----------------------|------|------|
| `driver/mu_memory.cpp` | 265 | `muapiMemAlloc_v2` 同步分配入口 |
| `driver/mu_memory.cpp` | 303 | `muapiMemAllocAsync` 异步分配入口 |
| `musa/core/context.cpp` | 915 | `Context::CreateMemory` 同步核心 |
| `musa/core/context.cpp` | 483 | `MapToPeers` 同步 peer 映射 |
| `musa/core/context.cpp` | 1859 | `ResolveDependencyAndQueueCommand` 阻塞控制核心 |
| `musa/core/memory.cpp` | 378 | `Memory::Init` 类型分发 |
| `musa/core/memory.cpp` | 462 | `GeneralAlloc` 虚存+物理合并分配 |
| `musa/core/memory.cpp` | 427 | `InitFromPool` 从池分配虚拟内存 |
| `musa/core/memory.cpp` | 152 | `Bind` 虚拟绑定物理 |
| `musa/core/memory.cpp` | 216 | `Unbind` 解绑 |
| `musa/core/memory.h` | 47-140 | Memory 类定义 |
| `musa/core/memoryPool.cpp` | 374 | `MemoryPool::CreateMemory` 池分配 |
| `musa/core/memoryPool.cpp` | 295 | `ModifyAccess` 异步跨设备访问设置 |
| `musa/core/memoryPool.cpp` | 374 | `CreateMemory` (池版本) |
| `musa/core/stream.cpp` | 519 | `AsyncMemAlloc` 三步分配+ModifyAccess |
| `musa/core/stream.cpp` | 570 | `CmdMemAlloc` 分配命令分发 |
| `musa/core/stream.cpp` | 496 | `CmdPaging` PagingCommand 创建入队 |
| `musa/core/stream.cpp` | 1005 | `QueueCommand` 命令入队实现 |
| `musa/core/stream.cpp` | 1108 | `AsyncSubmit` 命令提交线程 |
| `musa/core/stream.cpp` | 1280 | `AsyncWait` 命令等待线程 |
| `musa/core/command/pagingCommand.cpp` | 14 | `PagingCommand::Submit` 页表修改执行 |
| `musa/core/command/command.cpp` | 122 | `RecordDependency` 命令依赖记录 |
| `musa/core/command/command.cpp` | 165 | `Command::Build` 依赖→semaphore转换 |
| `musa/core/command/command.cpp` | 226 | `Command::Wait` 阻塞等待实现 |
| `musa/musaStream.h` | 105 | `MemoryAllocParameter` 结构定义 |
| `musa/musaStream.h` | 95 | `MemoryPaging` 结构定义 |
| `hal/halTypes.h` | 41-106 | 内存属性 flag 定义 |
| `hal/halMemory.h` | 271 | `IMemory` HAL 接口定义 |
| `hal/halMemory.h` | 209 | `MemoryCreateInfo` HAL 结构 |
| `hal/halMemMgr.h` | 15 | `IMemMgr` HAL 接口定义 |
| `hal/halMemoryPool.h` | 61 | `IMemoryPool` HAL 接口定义 |
| `hal/m3d/memory.cpp` | 264 | `OpenPeerMemory` HAL 层实现 |
| `hal/m3d/memMgr.cpp` | 81 | `MemMgr::Allocate` 内存管理器分配 |
| `hal/m3d/memoryPool.cpp` | 82 | `MemoryPool::FullAllocate` 池完整分配 |
| `hal/m3d/m3d/src/core/os/drm/mthreads/mtgpuMemory.cpp` | 828 | `GpuMemory::OpenPeerMemory` 内核级实现 |
| `hal/m3d/m3d/src/core/os/drm/mthreads/mtgpuMemory.cpp` | 112 | `GpuMemory::ImportMemory` 内核 ioctl 导入 |
