# muMemFree / muMemFree_v2 — 设备内存释放（源码逐行分析）

> 源码文件：`musa/src/driver/mu_memory.cpp:709-752, 386-443`，`musa/src/musa/core/memory.cpp:115-144, 358-376`，`musa/src/musa/core/stream.cpp:601-638`，`musa/src/musa/core/context.cpp:967-975`

## 1. 功能概述

释放由 `muMemAlloc` 等 API 分配的内存。根据内存类型不同，走不同的释放路径。

- **v2** (`muapiMemFree_v2`): 新接口，`dptr` 为 `MUdeviceptr`
- **v1** (`muapiMemFree`): 旧接口，`dptr` 为 `MUdeviceptr_v1`

## 2. muapiMemFree_v2 源码逐行分析

```cpp
// mu_memory.cpp:709
MUresult muapiMemFree_v2(MUdeviceptr dptr) {
    MUresult status = InitPlatform();                      // [internal.h:306]

    if (status == MUSA_SUCCESS) {
        if (0 != dptr) {                                    // 校验 dptr 非零
            size_t offset;
            Musa::IMemory* pMemory = Musa::Platform::Get()
                .GetMemoryByDevicePointer(dptr, &offset)    // [context.cpp:977]
                .get();                                     // 全局 MemoryTracker 查找

            if (!pMemory) {
                status = MUSA_ERROR_INVALID_VALUE;           // 未找到 → 失败
            } else if (pMemory->GetType() != Musa::memoryTypeGeneral &&
                       pMemory->GetType() != Musa::memoryTypePitchedGeneral &&
                       pMemory->GetType() != Musa::memoryTypeManaged &&
                       pMemory->GetType() != Musa::memoryTypeExternal &&
                       pMemory->GetType() != Musa::memoryTypeVirtual ||
                       (pMemory->GetType() == Musa::memoryTypeVirtual &&
                        pMemory->GetPool() == nullptr)) {    // 类型白名单校验
                // ⚠ 注意: memoryTypeVirtual 必须 pool != nullptr
                //   否则为非法偏移（子分配不能直接用基地址释放）
                status = MUSA_ERROR_INVALID_VALUE;
            } else if (offset != 0) {                         // ⚠ 必须从 allocation 基地址释放
                // 子分配偏移非零 → 拒绝
                // 用户不能释放子分配的偏移地址，必须释放基地址
                status = MUSA_ERROR_INVALID_VALUE;
            } else {
                status = pMemory->Synchronize();             // [memory.cpp:115] 等待完成
```

## 3. Synchronize 调用链

```cpp
// memory.cpp:115
MUresult Memory::Synchronize() const {
    MUresult status = MUSA_SUCCESS;

    if (m_Type == Musa::memoryTypeVirtual) {
        // 虚拟内存: 需要同步所有关联的物理子内存
        auto syncMemory = [this](MUdeviceptr ptr,
            std::shared_ptr<IMemory> memory) -> MUresult {
            MUresult status = memory->GetContext()->LockedWait();
            return status;
        };
        status = const_cast<MemoryTracker*>(&m_PhysTracker)
            ->IterateMemories(GetDevicePointer(), GetSize(), false,
                              std::move(syncMemory));
    } else {
        status = m_Context->LockedWait();                     // 等待默认流
        for (int devId = 0; MUSA_SUCCESS == status &&
             devId < Platform::Get().GetDeviceCount(); devId++) {
            auto peerDev = static_cast<Device*>(
                Platform::Get().GetIDeviceView(devId));
            if (peerDev && m_pHalMemory->GetPeerMemory(&peerDev->Hal())) {
                // ⚠ 同步所有 peer 设备的上下文
                ReadLockedAccessor<Device::CriticalBase> peerDevCrit(
                    peerDev->CriticalData());
                for (auto peerCtx : peerDevCrit->ctxs()) {
                    status = peerCtx->Synchronize();
                    if (MUSA_SUCCESS != status) break;
                }
            }
        }
    }
    return status;
}
```

### Synchronize 源码分析要点

```
Synchronize()
  │
  ├─ [Virtual 类型]:
  │     遍历 m_PhysTracker 中所有物理子内存
  │     对每个子内存: memory->GetContext()->LockedWait()
  │     ⚠ 等待的是子内存所在上下文, 不是当前上下文
  │
  └─ [非 Virtual 类型]:
        m_Context->LockedWait()     ← 等待默认流
        +-- 遍历所有 peer 设备
              if (peerDev 有 peer memory):
                遍历 peer 设备所有上下文 → Synchronize()
```

## 4. 同步释放分支（续）

```cpp
// 接 muapiMemFree_v2
                if (status == MUSA_SUCCESS) {
                    if (pMemory->GetType() == Musa::memoryTypeVirtual) {
                        // 路径 A: Virtual (sub-allocated) 内存
                        Musa::Context* pContext = TlsCtxTop();
                        if (pContext != nullptr) {
                            Musa::IStream* pStream =
                                Musa::Context::InfoStream(pContext, nullptr);
                            status = pStream->CmdMemFree(dptr, true);  // [stream.cpp:628]
                            //          ↑ blocking=true (同步)
                        } else {
                            status = MUSA_ERROR_INVALID_CONTEXT;
                        }
                    } else {
                        // 路径 B: General / PitchedGeneral /
                        //         Managed / External
                        status = pMemory->GetContext()
                            ->DestroyMemory(pMemory);   // [context.cpp:967]
                    }
                }
            }
        }
    }
    return status;
}
```

## 5. 路径 A: Virtual 内存释放调用链

```
Stream::CmdMemFree(dptr, true)                         [stream.cpp:628]
  │
  +-- if (capture 模式):
  │     Stream::CaptureMemFree(virtAddress)             [stream.cpp:588]
  │       +-- ctx->CreateMemFreeNode(&pGraphNode, virtAddress)
  │       +-- captureGraph->AddGraphNode(pGraphNode, deps)
  │       +-- SetLastCapturedNodes(...)
  │
  +-- else if (capture invalidated):
  │     MUSA_ERROR_STREAM_CAPTURE_INVALIDATED
  │
  +-- else (正常路径):
        Stream::AsyncMemFree(virtAddress, true)         [stream.cpp:601]

Stream::AsyncMemFree(virtAddress, blocking)             [stream.cpp:601]
  │
  ├─ Step 1: 查找内存对象                               [stream.cpp:605-606]
  │   virt   = Platform::Get()
  │           .GetMemoryByDevicePointer(virtAddress)
  │           .get()
  │   physical = virt->GetPhysMemory(virtAddress).get()
  │   pool     = virt->GetPool()
  │
  ├─ Step 2: 禁用 GPU 访问                             [stream.cpp:608]
  │   pool->DisableAccess(virt, physical,
  │                      blocking, this)
  │   │
  │   ⚠ 此步骤发送 PagingCommand 到 stream
  │   ⚠ 将虚拟页的 GPU 访问权限设为 NONE
  │
  │   if (失败):
  │     回滚 (不创建回调)
  │
  └─ Step 3: 创建流完成回调                             [stream.cpp:611-626]
        command = CallbackCommand(stream, callback)
        callback = [virt, command, pPool]() {
            virt->DestroyPhysMemories()    ← [memory.cpp:146]
            if (!virt->IsGraphAlloc()) {
                pPool->DestroyMemory(virt)  ← 归还 pool
                //      [memoryPool.cpp:214]
            }
        }
        m_ParentCtx->ResolveDependencyAndQueueCommand(
            command, stream, blocking)
```

## 6. 路径 B: General/PitchedGeneral/External/内存释放

```
Context::DestroyMemory(pMemory)                        [context.cpp:967]
  │
  +-- ctxCrit->RemoveMemory(pMem)                      [context.cpp:970]
  │     m_Memories.erase(pMemory)
  │
  └── Platform::Get().GetMemoryTracker()
        .UntrackMemory(pMemory)                        从全局映射移除

  ⚠ 注意: Memory 对象本身未被 delete
  ⚠ 由调用者 (或智能指针) 管理生命周期
  ⚠ Memory 析构函数会自动释放 HAL 资源 [memory.cpp:358]
```

## 7. Memory 析构函数双路径

```cpp
// memory.cpp:358
Memory::~Memory() {
    if (m_pMapped) {
        MUSA_ASSERT(m_pHalMemory);
        m_pHalMemory->Unmap();                          // 取消 CPU 映射
    }
    // ⚠ 仅非预分配内存需要手动释放
    if (m_Type != memoryTypePrealloc && m_pHalMemory) {
        if (m_pHalMemory->GetProps()
            & Hal::memoryPropertySubAllocatable) {
            // ┌─────────────────────────────────────┐
            // │ 路径 A: Sub-Allocation 释放           │
            // │ m_pPool == nullptr → 系统内部 pool    │
            // │ m_pPool != nullptr → 用户指定 pool   │
            // └─────────────────────────────────────┘
            if (m_pPool == nullptr) {                  // [memory.cpp:366]
                m_Context->GetParentDevice()
                    ->Hal().GetMemMgr()->Free(
                        m_pHalMemory,
                        GetDevicePointer(),
                        GetSize());                    // [memMgr.cpp:149]
                // → MemMgr::Free → m_PoolRefs.Get()
                //   → pool->Free(halMem, offset, size)
                //   → Pool::Free() [memoryPool.cpp:214]
            } else {                                   // [memory.cpp:369]
                m_pPool->Hal()->Free(
                    m_pHalMemory,
                    GetDevicePointer(),
                    GetSize());
                // → MemoryPool::Free()
                //   → SegmentTracker 查找
                //   → 标记空闲 + 左/右合并
                //   → ResourceRemove() (尝试销毁 chunk)
            }
        } else {
            // ┌─────────────────────────────────────┐
            // │ 路径 B: 裸分配释放                     │
            // │ 直接销毁 HAL 内存对象                   │
            // └─────────────────────────────────────┘
            m_PhysTracker.Cleanup();                   // [memory.cpp:146-150]
            m_pHalMemory->Destroy();                    // 调用 KMD 释放 ioctl
        }
    }
}
```

### 析构函数死代码分析

```
Memory::DestroyPhysMemories()                         [memory.cpp:146]
  │
  +-- GetHalMemory(nullptr)->SetProps(Physical, true)
  │     ⚠ 注意: 此处传入 nullptr → 使用 m_Context->GetParentDevice()
  │     设置 props 标志, 不执行任何释放操作
  │
  +-- m_PhysTracker.Cleanup()
        遍历并销毁所有物理子内存跟踪记录
        但此函数本身不释放 KMD 资源
        (KMD 资源释放由 Hal 引用计数管理)
```

## 8. Async 释放路径

```
muapiMemFreeAsync(dptr, hStream)                      [mu_memory.cpp:386]
  │
  +-- InitPlatform()
  │
  +-- do { } while(0) 宏包裹
  │
  +-- dptr == 0 → 直接 break (无操作)                 // ⚠ 0 指针静默成功
  │
  +-- TlsCtxTop() → pContext
  │
  +-- Context::InfoStream(ctx, hStream) → pStream     // [context.cpp:560]
  │
  +-- GetMemoryByDevicePointer(dptr) → virtMem
  │
  +-- memory = IntrusiveCast<Memory>(virtMem->get())
  │
  +-- switch (memory->GetType()):                      // [mu_memory.cpp:415-438]
        │
        ├─ memoryTypeIpcImport:                         // ┌【不会进入】
        │   └─ pContext->DestroyMemory(memory)          │  IPC Import 不应
        │       └─ DestroyMemory 仅注销跟踪              │  走 async 路径
        │                                              │  (sync path 已处理)
        │   ⚠ 注意: 此分支与 sync muapiMemFree_v2 不同  │
        │     sync: 拒绝 IPC Import (类型白名单)        │
        │     async: 接受并调用 DestroyMemory            │
        │     但 DestroyMemory 不会释放 KMD 资源        │
        │     ⚠ 实际上 IPC Import 不应在此释放           │
        │                                              │
        ├─ memoryTypeExternal:                          // ┌【不会进入】
        │   └─ pContext->DestroyMemory(memory)          │  同 IPC Import
        │                                              │
        ├─ memoryTypeGeneral /                          //
        │   memoryTypePitchedGeneral /                  //
        │   memoryTypeManaged:                          // ┌【主要路径】
        │     memory->Synchronize()                     │  等待完成 + DestroyMemory
        │     +-- ctx->DestroyMemory(memory)            │
        │                                              │
        │   ⚠ Synchronize() 等待的是默认流                │
        │     不是 hStream!                             │
        │     这意味着 async 释放实际上不是              │
        │     在 hStream 上异步执行                      │
        │                                              │
        └─ memoryTypeVirtual:                           // ┌【主要路径】
              if (memory->GetPool() != nullptr) {       //
                pStream->CmdMemFree(dptr, false)        // 真正的异步释放
                //      ↑ blocking=false
              } else {
                status = MUSA_ERROR_INVALID_VALUE       // ⚠ pool 为空 → 失败
              }
              break;
        └─ default:
              status = MUSA_ERROR_INVALID_VALUE
              break;
```

### Async 释放死代码/问题标注

```
⚠ ISSUE: General/PitchedGeneral/Managed 类型在 async 路径中
         实际上是同步等待的 (memory->Synchronize())
         不是真正的异步释放
         
⚠ ISSUE: IPC Import / External 类型在此进入 DestroyMemory
         但 DestroyMemory 仅从 ctx 内存列表中移除
         不释放底层 KMD 资源 (资源属于导出进程)
         这可能导致资源泄漏
```

## 9. 完整调用链汇总

```
┌─────────────────────────────────────────────────────────────────┐
│                    muapiMemFree_v2(dptr)                         │
│                          │                                      │
│              InitPlatform() ──── TlsCtxTop()                     │
│                          │                                      │
│              GetMemoryByDevicePointer(dptr, &offset)             │
│              └── MemoryTracker::FindRange()                     │
│                          │                                      │
│              ┌─── offset != 0? ──→ ERROR                         │
│              │                     (不可释放子分配偏移)           │
│              │                                                  │
│              ├─── Virtual(pool!=nullptr)?                        │
│              │     │                                             │
│              │     YES ──→ pStream->CmdMemFree(dptr, true)      │
│              │     │       ├─ Sync: AsyncMemFree + WaitFinish   │
│              │     │       └─ Async: AsyncMemFree + callback    │
│              │     │           ├─ DisableAccess (paging cmd)    │
│              │     │           ├─ CallbackCommand:              │
│              │     │           │   ├─ DestroyPhysMemories()     │
│              │     │           │   └─ Pool::DestroyMemory()     │
│              │     │           │       └─ Pool::Free()          │
│              │     │           │           ├─ Segment merge      │
│              │     │           │           └─ Chunk destroy?     │
│              │     │           │                                 │
│              │     │           └─ ResolveDependencyAndQueue      │
│              │     │                                             │
│              │     NO (pool==nullptr) ──→ ERROR                  │
│              │            (非法: 裸 virtual 指针)                │
│              │                                                  │
│              ├─── General/PitchedGeneral?                        │
│              │     YES ──→ DestroyMemory(pMemory)               │
│              │     │       ├─ RemoveMemory from ctx              │
│              │     │       └─ UntrackMemory from tracker         │
│              │     │       └─ ~Memory() [后续析构]:              │
│              │     │           ├─ SubAllocatable?                │
│              │     │           │   YES ──→ MemMgr::Free()        │
│              │     │           │   │     └─ Pool::Free()         │
│              │     │           │   NO  ──→ Destroy() (KMD)       │
│              │     │           │                                 │
│              │     NO → ERROR (类型拒绝)                         │
│              │                                                  │
│              ├─── Managed?                                      │
│              │     YES ──→ 同 General                            │
│              │     NO → ERROR                                    │
│              │                                                  │
│              └─── External?                                     │
│                    YES ──→ 同 General                            │
│                    NO → ERROR                                    │
└─────────────────────────────────────────────────────────────────┘
```

## 10. 源码位置对照表

| 步骤 | 函数 | 文件:行数 | 说明 |
|------|------|----------|------|
| 参数校验 | `InitPlatform()` | `internal.h:306` | 平台初始化检查 |
| 类型查找 | `GetMemoryByDevicePointer()` | `context.cpp:977` | 全局 `MemoryTracker::FindRange()` |
| 同步等待 | `Memory::Synchronize()` | `memory.cpp:115-144` | Virtual/非 Virtual 双路径 |
| 路径 A (Virtual) | `Stream::CmdMemFree()` | `stream.cpp:628-638` | 入队释放命令 |
| 路径 A → 禁用访问 | `Stream::AsyncMemFree()` | `stream.cpp:601-626` | DisableAccess + Callback |
| 路径 A → 释放资源 | `Memory::DestroyPhysMemories()` | `memory.cpp:146-150` | 清除物理跟踪 |
| 路径 A → 归还 Pool | `MemoryPool::DestroyMemory()` | `memory.cpp:358-376` | 析构归还 |
| 路径 A → Pool::Free | `MemoryPool::Free()` | `memoryPool.cpp:214-259` | 子分配释放 + 合并 |
| 路径 B (Direct) | `Context::DestroyMemory()` | `context.cpp:967-975` | 注销跟踪 |
| 路径 B → 析构 | `Memory::~Memory()` | `memory.cpp:358-376` | SubAlloc→Pool::Free / else→Destroy |
| Async 入口 | `muapiMemFreeAsync()` | `mu_memory.cpp:386-443` | 类型分发 |
| Pool::Free → Remove | `ResourceRemove()` | `memoryPool.cpp:318-332` | 惰性释放 chunk |
| Pool::Free → Merge | `FreeListInsert()` | `memoryPool.cpp:416-458` | 合并后插入空闲列表 |