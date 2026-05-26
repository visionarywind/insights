# 09_usage_patterns — 内存 API 内部调用者分析

> 本文档追踪 MUSA 运行时内部各处对内存 API 的调用，揭示框架层的内存使用模式。

## 1. 调用矩阵

| 内部调用者 | 调用的 API | 调用位置 | 内存类型 | 用途 |
|-----------|-----------|----------|----------|------|
| Stream::AsyncMemAlloc | `pPool->CreateMemory()` | stream.cpp:537 | Virtual(pooled) | 流式异步分配 |
| Stream::AsyncMemAlloc | `physical->Init(General)` | stream.cpp:552 | General | 物理后备内存 |
| Context::CreateMemory | `pMemory->Init(createInfo)` | context.cpp:939 | 由 createInfo 决定 | 通用入口 |
| Context::CreateMemory | `ctxCrit->MapToPeers()` | context.cpp:945 | 同左 | Peer 映射 |
| Memory::Bind | `phys->ExportDmaBufHandle()` | memory.cpp:169 | Physical | DMA-BUF 导出 |
| Memory::Bind | `sPhys->ExternalAlloc()` | memory.cpp:184 | External | 外部内存包装 |
| Memory::Unbind | `stream->CmdPaging()` | memory.cpp:245 | — | 解除页面映射 |
| Memory::EnableAccess | `mem->Hal()->OpenPeerMemory()` | memory.cpp:307 | Peer view | Peer 内存打开 |
| Memory::EnableAccess | `stream->CmdPaging()` | memory.cpp:326 | — | 设置页面权限 |
| PinnedHostRegister | `Hal::CreateMemory(Locked)` | memory.cpp:667 | Locked | 注册用户指针 |
| PinnedHostRegister | `Hal::CreateMemory(External/MMIO)` | memory.cpp:657 | External | MMIO 映射 |
| IpcImportAlloc | `Hal::CreateMemory(External)` | memory.cpp:748 | External | IPC 导入 |
| ExternalAlloc | `Hal::CreateMemory(External)` | memory.cpp:789 | External | 外部句柄导入 |
| ManagedAlloc | `Hal::CreateMemory(Alloc)` | memory.cpp:706 | General | Managed 分配 (不经过 pool) |
| PitchedGeneralAlloc | `MemMgr::Allocate()` | memory.cpp:526 | General(pooled) | Pitched 分配 |

## 2. Graph 场景的内存分配

### 图捕获阶段的分配

```
Stream::CaptureMemAlloc(param)                           [stream.cpp:502]
  │
  +-- ctx->CreateMemAllocNode(memAllocParam, &pGraphNode)
  │     MUSA_MEM_ALLOC_NODE_PARAMS = {bytesize}
  │
  +-- captureGraph->AddGraphNode(pGraphNode, deps)
  │
  +-- param.virtAddress = memAllocParam.dptr
        (此时已分配虚拟地址, 物理内存在图执行时绑定)
```

### 图执行时的实际分配

```
(在图展开执行时)
GraphMemAllocNode::Execute()
  │
  +-- 调用 AsyncMemAlloc() 逻辑 (同 stream.cpp:519)
  │     step 1: pool->CreateMemory() → 虚拟内存
  │     step 2: physical->Init(General) → 物理内存
  │     step 3: virt->Bind(physical) → 建立映射
  │     step 4: pool->ModifyAccess() → GPU 可见
```

### 图释放

```
GraphMemFreeNode::Execute()
  → Stream::AsyncMemFree(virtAddress, blocking=false)
  → DestroyPhysMemories() + Pool::DestroyMemory(virt)
```

## 3. Peer 映射的自动触发

**每次成功的 CreateMemory 都会检查并自动建立 Peer 映射:**

```
Context::CreateMemory()  [context.cpp:943-946]
  │
  +-- if (pMemory->Hal()->GetCapability() & PeerAccessible):
  │     ctxCrit->MapToPeers(pMemory)
  │
MapToPeers()  [context.cpp:483-558]
  │
  +-- 遍历所有 device (跳过自身):
  │     │
  │     +-- General/PitchedGeneral:
  │     │     mapFlags = m_Peers[peer]  (PEERMAP_FLAG_PCIEONLY 等)
  │     │
  │     +-- PinnedHost/RegisteredPinnedHost:
  │     │     mapFlags = PEERMAP_FLAG_PCIEONLY
  │     │
  │     +-- Managed:
  │     │     MU_MEM_ATTACH_HOST → 仅 concurrent access 设备
  │     │     其他 → PEERMAP_FLAG_PCIEONLY
  │     │
  │     +-- IpcImport:
  │     │     mapFlags = flags - 1  (IPC flag 直接转为 map flag)
  │     │
  │     +-- External:
  │           mapFlags = PEERMAP_FLAG_INVALID  (不映射)
  │
  │     +-- if (mapFlags != INVALID):
  │           Hal::OpenPeerMemory(peerDevice, openInfo)
```

## 4. IPC 导出/导入的内存流

### 导出端

```
Memory::ExportIpcHandle()  [memory.cpp:53]
  │
  +-- Hal::ExportExternalHandle(kernelManagedGlobal)
  │     → m_M3dGpuMemory->ExportExternalHandle(ExportHandleType::Global)
  │
  +-- 填充 IpcHandleInternal:
  │     halHandle   = HAL 层的全局句柄
  │     pciBusId    = 设备 PCI 总线 ID
  │     heap        = 内存堆类型
  │     allocOffset = m_Offset (子分配偏移)
  │     allocSize   = 实际大小
  │     pid         = 进程 ID
  │     serial      = m_SeqID
```

### 导入端

```
IpcImportAlloc(handle, flags)  [memory.cpp:715]
  │
  +-- Hal::CreateMemory(View/External)
  │     createInfo.view.external.type = kernelManagedGlobal
  │     createInfo.view.external.handle = handle.halHandle
  │
  +-- m_Offset = handle.allocOffset
  +-- m_Shape  = {handle.allocSize, 1, 1, handle.allocSize}
```

## 5. 销毁链

Context 销毁时自动清理:

```
Context::CriticalBase::ReleaseClientResources()  [context.cpp:304]
  │
  +-- 1. 同步所有 stream:
  │     for each stream: stream->WaitFinish()
  │
  +-- 2. 销毁 textures, surfaces, graphics resources
  │
  +-- 3. 销毁 streams (先非默认, 后默认):
  │     delete pBarrierStream
  │     delete pDefaultStream
  │
  +-- 4. 释放所有内存 (WARN 若有泄漏):
  │     for each pMemory in m_Memories:
  │       MemoryTracker::UntrackMemory(pMemory)
  │       (Memory 析构函数自动释放 HAL 资源)
  │
  +-- 5. 销毁 events, arrays, modules, graph execs
  │
  +-- 6. 销毁 external semaphores, external memories
```

## 6. 关键设计模式

### RAII 模式

```
Memory 对象生命周期:
  1. new Memory(ctx)        → 构造 (m_pHalMemory=nullptr)
  2. memory->Init(info)     → 分配 HAL 资源
  3. ctx->AddMemory(mem)    → 加入跟踪
  4. ... 使用 ...
  5. ctx->DestroyMemory(mem)→ 从跟踪移除
  6. delete memory           → 析构自动释放 HAL 资源
```

### 双重释放防护

```
muapiMemFree_v2 检查:
  ├─ pMemory->GetType() == Virtual && pMemory->GetPool() == nullptr
  │   → MUSA_ERROR_INVALID_VALUE (非法: 裸 virtual 地址, pool 为空)
  │
  ├─ 仅允许释放通过 Alloc 得到的基指针
  │   offset != 0 → MUSA_ERROR_INVALID_VALUE
  │
  └─ 类型白名单: General, PitchedGeneral, Managed, External, Virtual(pool)
     其他类型 → MUSA_ERROR_INVALID_VALUE
```

## 7. 相关源码位置

| 文件 | 行数 | 说明 |
|------|------|------|
| `musa/src/driver/mu_memory.cpp` | 全文件 | 所有 Driver 层 API 入口 |
| `musa/src/musa/core/memory.cpp` | 全文件 | Core 层内存管理 |
| `musa/src/musa/core/stream.cpp` | 502-580, 582-638 | AsyncMemAlloc/AsyncMemFree |
| `musa/src/musa/core/context.cpp` | 304-404, 483-558, 915-965 | 创建/销毁/Peer映射 |
| `musa/src/musa/core/memory.cpp` | 53-75, 152-265 | 导出/绑定/解绑 |
| `musa/src/hal/m3d/memory.cpp` | 264-352 | OpenPeerMemory/DestroyPeerMemory |