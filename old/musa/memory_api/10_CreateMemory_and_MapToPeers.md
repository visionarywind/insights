# 10_CreateMemory_and_MapToPeers — 分配总入口与 Peer 映射

> 源码文件：`musa/src/musa/core/context.cpp:915-965, 483-558`

## 1. 功能概述

`Context::CreateMemory` 是 MUSA 内存分配的**统一入口**。所有用户 API (`muMemAlloc`, `muMemHostAlloc`, `muMemHostRegister`, `muMemAllocManaged`, `muMemAllocPitch` 等) 最终都调用此函数。

创建成功后，自动执行 **MapToPeers** 将内存映射到所有已启用 peer access 的设备。

## 2. CreateMemory 完整流程

```
Context::CreateMemory(ppMemory, createInfo)                [context.cpp:915]
  │
  ├─ Step 0: Capture 检查                                   [context.cpp:922-931]
  │   遍历所有 stream, 若有处于 ACTIVE capture 状态 → 返回错误
  │   (确保分配操作不在 graph capture 期间执行)
  │
  ├─ Step 1: 创建 Memory 对象                               [context.cpp:937]
  │   memory_sp = make_shared<Memory>(this)
  │   pMemory = static_cast<Memory*>(memory_sp.get())
  │
  ├─ Step 2: 初始化内存                                     [context.cpp:939]
  │   pMemory->Init(createInfo)  →  跳转到对应类型的初始化
  │   (详见下方 §3 各类型初始化路径)
  │
  ├─ Step 3: Peer 映射 (条件触发)                           [context.cpp:943-946]
  │   if (pMemory->Hal()->GetCapability() & PeerAccessible):
  │     ctxCrit->MapToPeers(pMemory)
  │
  ├─ Step 4: 加入上下文管理                                 [context.cpp:948]
  │   ctxCrit->AddMemory(pMemory)
  │   m_Memories.insert(pMemory)
  │
  ├─ Step 5: 注册到全局跟踪器                                [context.cpp:957]
  │   Platform::Get().GetMemoryTracker().TrackMemory(memory_sp)
  │   (建立 MUdeviceptr → shared_ptr<IMemory> 的全局映射)
  │
  ├─ Step 6: 分配 SeqID                                     [context.cpp:959]
  │   if (!pMemory->IsPhysical()):
  │     Platform::Get().SetMemorySeqID(pMemory)
  │     (为后续 PointerGetAttributes 的 BUFFER_ID 提供编号)
  │
  └─ *ppMemory = pMemory                                    [context.cpp:961]
```

## 3. 各类型初始化路径汇总

`Memory::Init(createInfo)` 根据 `createInfo.type` 分发到不同函数:

```
Memory::Init()  [memory.cpp:378-425]
  │
  ├─ memoryTypeGeneral → GeneralAlloc()                    [memory.cpp:462]
  │     flags = Virtual | DeviceMapped | SubAllocatable
  │     Hal::MemoryAllocInfo.type = memoryAllocTypeDeviceLocal
  │     Hal::MemoryAllocInfo.heap = MemoryHeap::largePage
  │     property 自动追加: Physical|SharedVA|DeviceVisible|HostVisible
  │                        |HostCoherent|DeviceWriteable|DeviceCached
  │     viewCapability 自动追加: PeerAccessible|IpcExportable
  │     → MemMgr::Allocate() (Pool 子分配)  或  Hal::CreateMemory()
  │
  ├─ memoryTypePitchedGeneral → PitchedGeneralAlloc()      [memory.cpp:499]
  │     固定 property = 全部打开 (Full Access)
  │     viewCapability = PeerAccessible|IpcExportable|Exportable
  │     → MemMgr::Allocate() (必走 Pool, 无裸路径)
  │
  ├─ memoryTypePinnedHost → PinnedHostAlloc()              [memory.cpp:532]
  │     alloc.type = memoryAllocTypeHost
  │     heap = largePage (static, 不支持降级为 texture/usc)
  │     property = Physical|SharedVA|HostMapped
  │              + (DEVICEMAP ? Virtual|HostVisible|DeviceVisible|...)
  │     heap 降级: largePage → general (一旦触发, 永久生效)
  │
  ├─ memoryTypeRegisteredPinnedHost → PinnedHostRegister() [memory.cpp:611]
  │     type = memoryTypeView
  │     view.type = memoryViewTypeLocked
  │     或 view.type = memoryViewTypeExternal (MMIO)
  │
  ├─ memoryTypeManaged → ManagedAlloc()                    [memory.cpp:673]
  │     type = memoryTypeAlloc
  │     alloc.type = memoryAllocTypeHost (或 DeviceLocal)
  │     property = full access 全开
  │     → Hal::CreateMemory() (不走 pool)
  │
  ├─ memoryTypeIpcImport → IpcImportAlloc()                [memory.cpp:715]
  │     type = memoryTypeView
  │     view.type = memoryViewTypeExternal
  │     handle = kernelManagedGlobal
  │
  ├─ memoryTypeExternal → ExternalAlloc()                  [memory.cpp:760]
  │     type = memoryTypeView
  │     view.type = memoryViewTypeExternal
  │     handle = dmaBuf | kernelManagedGlobal | fabric
  │
  ├─ memoryTypeVirtual → VirtualAlloc()                    [memory.cpp:805]
  │     type = memoryTypeVirtual
  │     heap = largePage
  │     → Hal::CreateMemory(Virtual)  (仅创建 VA 空间)
  │
  └─ memoryTypePrealloc → PreallocAlloc()                  [memory.cpp:801]
        return MUSA_ERROR_NOT_SUPPORTED
```

## 4. MapToPeers 详解

```
Context::MapToPeers(memory)                                [context.cpp:483]
  │
  ├─ 确定 peer 数量:                                       [context.cpp:486-503]
  │   switch (memory->GetType()):
  │     External/IpcImport/PinnedHost/RegisteredPinnedHost:
  │       peerCount = Platform::Get().GetDeviceCount()  (全部设备)
  │     General/PitchedGeneral:
  │       peerCount = m_Peers.size()  (仅已启用 peer 的设备)
  │     Managed:
  │       managedForceDeviceAlloc ? m_Peers.size()
  │                              : Platform::Get().GetDeviceCount()
  │
  ├─ 遍历每个 peer device (跳过自身):                       [context.cpp:506-556]
  │   │
  │   ├─ 确定 mapFlags:
  │   │   General/PitchedGeneral → m_Peers[peer]
  │   │   PinnedHost/RegisteredPinnedHost → PCIEONLY
  │   │   Managed(ATTACH_HOST) → 仅 concurrent access 设备
  │   │   Managed(ATTACH_GLOBAL) → m_Peers[peer] (force alloc 时)
  │   │   IpcImport → flags - 1
  │   │   External → INVALID (跳过)
  │   │
  │   ├─ if (mapFlags != INVALID):
  │   │   openInfo.type = kernelManagedGlobal
  │   │   openInfo.deviceMapped = true
  │   │   openInfo.deviceMapType = mapFlags
  │   │   memory->Hal()->OpenPeerMemory(&peerHal, openInfo)
  │   │
  │   └─ 失败即中止 (break, 返回 error)
  │
  └─ 返回 status
```

## 5. Peer OpenInfo 的 HAL 处理

```
Hal::Memory::OpenPeerMemory(peerDevice, openInfo)        [hal/m3d/memory.cpp:264]
  │
  +-- if (!m_PeerMemories[peerId]):   (尚未打开)
  │     │
  │     +-- switch(openInfo.openType):
  │     │     │
  │     │     ├─ kernelManagedGlobal (IPC Import / Alloc):
  │     │     │     createInfo.type = memoryTypeView
  │     │     │     createInfo.view.type = memoryViewTypePeer
  │     │     │     createInfo.view.peer.pMemory = this
  │     │     │     algoTopoType = PCIE_TOPO / UNIDIR_TOPO / BIDIR_TOPO 等
  │     │     │                    (根据 P2P 能力和 mapFlags 确定)
  │     │     │     └─ 若不支持 mtlink → 降级为 PCIE_TOPO
  │     │     │
  │     │     └─ dmaBuf (External Alloc with DMA-BUF):
  │     │           先 ExportDmaBufHandle() 获取 fd
  │     │           createInfo.type = memoryTypeView
  │     │           createInfo.view.type = memoryViewTypeExternal
  │     │           handle = dmaBuf fd, size, isDeviceMapped 等
  │     │
  │     +-- peerDevice->CreateMemory(createInfo, &peerMem)
  │     │
  │     +-- m_PeerMemories[peerId] = peerMem
  │
  └─ 返回 Result::success 或已存在
```

## 6. DestroyPeerMemory

```
Hal::Memory::DestroyPeerMemory(peerDevice)                [hal/m3d/memory.cpp:354]
  │
  +-- peerMemory = m_PeerMemories[peerId]
  +-- peerMemory->Destroy()    (销毁 HAL 内存对象)
  +-- m_PeerMemories[peerId] = nullptr
```

## 7. MemoryTracker 全局映射

```
Platform::Get().GetMemoryTracker().TrackMemory(memory_sp)
  │
  └── 将 memory 加入 MemoryTracker 的区间树
      key:   [devicePointer, devicePointer + size)
      value: shared_ptr<IMemory> (即 memory_sp)

后续所有通过 device pointer 查找内存的操作:
  - muapiMemFree_v2
  - muapiPointerGetAttribute
  - muapiPointerGetAttributes
  - Context::GetMemoryByDevicePointer
  - Stream::AsyncMemFree
  
均通过 MemoryTracker::FindRange() 完成反向查找
```

## 8. DestroyMemory 与内存泄漏检测

```
Context::DestroyMemory(pMemory)                         [context.cpp:967]
  │
  +-- ctxCrit->RemoveMemory(pMem)
  │     m_Memories.erase(pMemory)
  │
  +-- MemoryTracker::UntrackMemory(pMemory)

Context 析构时的泄漏检查:                                [context.cpp:383-391]
  for each pMemory in m_Memories:
    if (pMemory->GetDevicePointer()):
      tprintf(LOG_WARN, "WARNING: cleanup unfreed memory: %#llx", ...)
    else:
      tprintf(LOG_WARN, "WARNING: cleanup unfreed phys memory: %p", ...)
    MemoryTracker::UntrackMemory(pMemory)
```

## 9. 相关源码位置

| 文件 | 行数 | 说明 |
|------|------|------|
| `musa/src/musa/core/context.cpp` | 483-558 | `MapToPeers` |
| `musa/src/musa/core/context.cpp` | 915-965 | `CreateMemory` |
| `musa/src/musa/core/context.cpp` | 967-975 | `DestroyMemory` |
| `musa/src/musa/core/context.cpp` | 304-404 | `ReleaseClientResources` (泄漏检测) |
| `musa/src/musa/core/memory.cpp` | 378-425 | `Memory::Init` (类型分发) |
| `musa/src/hal/m3d/memory.cpp` | 264-352 | `OpenPeerMemory` + `DestroyPeerMemory` |
| `musa/src/driver/internal.h` | - | `MemoryTracker` 接口 |