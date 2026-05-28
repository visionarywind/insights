# muMemFreeAsync — 异步内存释放

## 功能

在指定 stream 上按 stream 顺序异步释放内存。`muMemFreeAsync` 保证 stream 中所有先于它的操作完成后才真正释放内存，防止 use-after-free。对应 CUDA `cudaFreeAsync`。

## 完整调用链

```
用户代码: muMemFreeAsync(dptr, hStream)
  │
  ├─ 1. mu_wrappers_generated.cpp — MUPTI 插桩
  │
  ├─ 2. mu_memory.cpp:386 — muapiMemFreeAsync
  │    ├─ InitPlatform()
  │    ├─ dptr == 0 → 空操作 (no-op)
  │    ├─ TlsCtxTop() → Context
  │    ├─ InfoStream(ctx, hStream) → Stream*
  │    ├─ GetMemoryByDevicePointer(dptr) → Memory*
  │    │    (通过 MemoryTracker 反查)
  │    └─ switch (memory->GetType()):
  │         ├─ memoryTypeIpcImport:
  │         │   → pContext->DestroyMemory(memory)       // 立即销毁
  │         ├─ memoryTypeExternal:
  │         │   → pContext->DestroyMemory(memory)       // 立即销毁
  │         ├─ memoryTypeGeneral/Pitched/Managed:
  │         │   → memory->Synchronize() + DestroyMemory()  // 同步等待
  │         └─ memoryTypeVirtual (with pool):
  │              → pStream->CmdMemFree(dptr, false)     // stream 异步
  │
  ├─ 3. stream.cpp:628 — Stream::CmdMemFree
  │    ├─ [Capture active]  → CaptureMemFree (记录到 graph)
  │    ├─ [Capture invalid] → 错误
  │    └─ [正常]            → AsyncMemFree(virtAddress, false)  // 主要路径
  │
  ├─ 4. stream.cpp:601 — Stream::AsyncMemFree
  │    ├─ 获取 virtual memory + physical memory (from MemoryTracker)
  │    ├─ pPool->DisableAccess(virt, physical, false, this)  // 禁用 peer access
  │    │    发送 PagingCommand 到 stream, 更新 GPU 页表
  │    │    移除 peer device 对该内存的访问权限
  │    │
  │    ├─ new CallbackCommand(this, callback)
  │    │   └── 回调函数本体:
  │    │       [=]() {
  │    │           virt->DestroyPhysMemories();           // 解绑物理内存
  │    │           if (!virt->IsGraphAlloc())
  │    │               pPool->DestroyMemory(virt);        // 归还给 pool
  │    │       }
  │    │
  │    └─ ResolveDependencyAndQueueCommand(callbackCmd, this, false)
  │         callback 命令在 stream 中所有前序命令完成后执行
  │
  └─ 5. [Stream Executor Thread] CallbackCommand 执行
       └─ 执行回调:
            ├─ DestroyPhysMemories(): 清除虚拟内存的物理绑定
            └─ Pool::DestroyMemory(): 归还内存到 pool, 可被后续分配复用
```

## 时序图

```
应用层           Wrapper          Driver(mu_mem)      Context         Stream          Command(Callback)    MemoryPool     KMD/GPU
  │               │                │                   │              │               │                    │              │
  │ muMemFreeAsync│                │                   │              │               │                    │              │
  │──────────────>│                │                   │              │               │                    │              │
  │               │                │                   │              │               │                    │              │
  │               │ muapiMemFree   │                   │              │               │                    │              │
  │               │ Async          │                   │              │               │                    │              │
  │               │───────────────>│                   │              │               │                    │              │
  │               │                │                   │              │               │                    │              │
  │               │                │ InitPlatform()    │              │               │                    │              │
  │               │                │                   │              │               │                    │              │
  │               │                │ dptr==0? 空操作   │              │               │                    │              │
  │               │                │                   │              │               │                    │              │
  │               │                │ TlsCtxTop()       │              │               │                    │              │
  │               │                │──────────────────>│              │               │                    │              │
  │               │                │<── ctx* ──────────│              │               │                    │              │
  │               │                │                   │              │               │                    │              │
  │               │                │ InfoStream(ctx,   │              │               │                    │              │
  │               │                │   hStream)        │              │               │                    │              │
  │               │                │──────────────────>│─────────────>│               │                    │              │
  │               │                │<── Stream* ───────│<── Stream* ──│               │                    │              │
  │               │                │                   │              │               │                    │              │
  │               │                │ GetMemoryByDevice │              │               │                    │              │
  │               │                │ Pointer(dptr)     │              │               │                    │              │
  │               │                │──────────────────>│              │               │                    │              │
  │               │                │<── Memory* ───────│              │               │                    │              │
  │               │                │                   │              │               │                    │              │
  │               │                │ switch(type):     │              │               │                    │              │
  │               │                │                   │              │               │                    │              │
  │               │                │ [General/Pitched/Managed]        │               │                    │              │
  │               │                │   Sync+Destroy    │              │               │                    │              │
  │               │                │  (同步, 非异步)   │              │               │                    │              │
  │               │                │                   │              │               │                    │              │
  │               │                │ [IPC/External]    │              │               │                    │              │
  │               │                │   DestroyMemory   │              │               │                    │              │
  │               │                │  (立即销毁)       │              │               │                    │              │
  │               │                │                   │              │               │                    │              │
  │               │                │ [Virtual]         │              │               │                    │              │
  │               │                │   CmdMemFree      │              │               │                    │              │
  │               │                │──────────────────>│─────────────>│               │                    │              │
  │               │                │                   │              │               │                    │              │
  │               │                │                   │              │ AsyncMemFree   │                    │              │
  │               │                │                   │              │───────────────>│                    │              │
  │               │                │                   │              │               │                    │              │
  │               │                │                   │              │ 获取 virt +    │                    │              │
  │               │                │                   │              │ physical mem   │                    │              │
  │               │                │                   │              │ (GetPhysMemory) │                   │              │
  │               │                │                   │              │               │                    │              │
  │               │                │                   │              │ DisableAccess()│                    │              │
  │               │                │                   │              │───────────────>│--PagingCommand    │              │
  │               │                │                   │              │               │  (CmdPaging)       │              │
  │               │                │                   │              │               │  移除 peer device  │              │
  │               │                │                   │              │               │  的 GPU 页表映射   │              │
  │               │                │                   │              │               │───────────────────>│--GPU更新页表  │
  │               │                │                   │              │               │<── OK ─────────────│<── OK ───────│
  │               │                │                   │              │               │                    │              │
  │               │                │                   │              │ new CallbackCmd│                    │              │
  │               │                │                   │              │───────────────>│                    │              │
  │               │                │                   │              │               │ 回调:              │              │
  │               │                │                   │              │               │  - DestroyPhysMem  │              │
  │               │                │                   │              │               │  - Pool::DestroyMem│              │
  │               │                │                   │              │               │                    │              │
  │               │                │                   │              │ ResolveDeps +  │                    │              │
  │               │                │                   │              │ QueueCommand   │                    │              │
  │               │                │                   │              │───────────────>│--SetDeps +         │              │
  │               │                │                   │              │               │  m_CommandList     │              │
  │               │                │                   │              │               │  push_back          │              │
  │               │                │                   │              │               │                    │              │
  │               │<── OK ─────────│<── OK ────────────│<── OK ───────│<── OK ────────│                    │              │
  │<── OK ────────│                │                   │              │               │                    │              │
  │               │                │                   │              │               │                    │              │
  │               │                │                   │              │  [Stream 执行所有前序命令]               │              │
  │               │                │                   │              │               │                    │              │
  │               │                │                   │              │  → 前序命令完成                         │              │
  │               │                │                   │              │               │                    │              │
  │               │                │                   │              │               │ ★Callback触发      │              │
  │               │                │                   │              │               │─────────────────────                │              │
  │               │                │                   │              │               │                    │              │
  │               │                │                   │              │               │ DestroyPhysMem     │              │
  │               │                │                   │              │               │── 清除物理绑定      │              │
  │               │                │                   │              │               │                    │              │
  │               │                │                   │              │               │ Pool::DestroyMemory │              │
  │               │                │                   │              │               │───────────────────>│              │
  │               │                │                   │              │               │                    │ 回收内存     │
  │               │                │                   │              │               │                    │ 下次分配可复用│
  │               │                │                   │              │               │<── OK ─────────────│              │
  │               │                │                   │              │               │                    │              │
```

## 关键代码路径

### Driver 入口 — 类型分叉

```cpp
// mu_memory.cpp:386
MUresult muapiMemFreeAsync(MUdeviceptr dptr, MUstream hStream) {
    // 1. InitPlatform()
    // 2. dptr==0 → 空操作
    // 3. TlsCtxTop() + InfoStream()
    // 4. MemoryTracker::FindRange → Memory*
    // 5. 根据内存类型分叉:
    switch (memory->GetType()) {
        case IpcImport:
        case External:
            // IPC/外部内存: 立即销毁 (不影响其他进程)
            pContext->DestroyMemory(memory);
            break;

        case General:
        case PitchedGeneral:
        case Managed:
            // 常规内存: 同步等待后再销毁
            // 注意: 这里并非真的"异步", 而是同步等待
            memory->Synchronize();
            memory->GetContext()->DestroyMemory(memory);
            break;

        case Virtual:
            // Virtual memory from pool: ★ 真正的 stream-ordered 释放
            if (memory->GetPool() != nullptr)
                pStream->CmdMemFree(dptr, false);  // callback 异步
            else
                error;
            break;
    }
}
```

### Stream::AsyncMemFree — CallbackCommand 机制

```cpp
// stream.cpp:601
MUresult Stream::AsyncMemFree(uint64_t virtAddress, bool blocking) {
    // 1. 获取 VirtualMemory 和它绑定的 PhysicalMemory
    Memory* virt = GetMemoryByDevicePointer(virtAddress)->get();
    Memory* physical = virt->GetPhysMemory(virtAddress)->get();
    MemoryPool* pPool = virt->GetPool();

    // 2. 先禁用 peer access (通过 PagingCommand 更新 GPU 页表)
    status = pPool->DisableAccess(virt, physical, blocking, this);
    // DisableAccess 发送 CmdPaging:
    //   paging.flags = MU_MEM_ACCESS_FLAGS_PROT_NONE
    //   设置所有 peer device 的页表权限为 "禁止访问"

    // 3. 创建 CallbackCommand, 在 GPU 完成前序命令后执行
    if (status == MUSA_SUCCESS) {
        auto command = make_shared<CallbackCommand>(this, function<void()>());

        // 回调函数: 在 stream 中所有前序命令完成后才执行
        function<void()> callback = [virt, command, pPool]() {
            // Step A: 销毁虚拟内存的物理绑定
            virt->DestroyPhysMemories();
            //   → 遍历 m_PhysTracker, 逐一解除物理绑定
            //   → 对每个物理块: DestroyPeerMemory on all peers

            // Step B: 内存归还到 pool (可被后续分配复用)
            if (!virt->IsGraphAlloc()) {
                pPool->DestroyMemory(virt);
                //   → Hal::IMemoryPool::Free
                //   → ResSegment 回收
            }
        };

        command->SetCallback(move(callback));

        // 插入 stream 队列 (依赖前序命令)
        status = m_ParentCtx->ResolveDependencyAndQueueCommand(
                     move(command), this, blocking);
    }
    return status;
}
```

### muMemFreeAsync 释放策略总览

| 内存类型 | 释放策略 | 是否真正"异步" | 说明 |
|---------|---------|---------------|------|
| General/Pitched/Managed | `Synchronize() + DestroyMemory()` | ❌ 同步 | 等待所有 GPU 访问完成后立即归还 pool |
| IPC Import | `DestroyMemory()` | ✅ 立即 | 只是减少引用计数, 不影响原进程 |
| External | `DestroyMemory()` | ✅ 立即 | 只是减少引用计数 |
| Virtual (from pool) | `CallbackCommand` 异步释放 | ✅ 真正异步 | GPU 完成后才通过回调真正归还内存 |

### CallbackCommand 的执行时机

```
Stream 命令队列: [前序命令A] → [前序命令B] → [CallbackCommand(释放)] → ...
                                                    ↑
                                             只有 A 和 B 都完成后
                                             才执行释放回调
```

这保证了:
1. 前序 GPU kernel 还在读该内存时 → 不会释放
2. 前序 memcpy 还在写该内存时 → 不会释放
3. 所有操作都完成后 → callback 触发, 执行 `DestroyPhysMemories` + `Pool::DestroyMemory`

## 关键设计要点

1. **CallbackCommand 模式**: 通过 stream 中的回调命令实现异步释放，保证释放操作在依赖命令完成后才执行
2. **DisableAccess 先行**: 释放前先通过 `PagingCommand` 移除所有 peer device 的页表映射，防止其他 device 在释放后访问无效内存
3. **类型分叉**: 不同类型的 AsyncFree 行为不同——General 实际上同步等待，Virtual(with pool) 才是真正的 stream-ordered 异步
4. **double-check 安全**: `DestroyPhysMemories` 清除物理绑定，`Pool::DestroyMemory` 归还内存到池中，两步完成完整释放
5. **Graph 保护**: `IsGraphAlloc()` 检查确保 graph 专用的 allocation 不会被 pool 回收（graph 的生命周期由 graph exec 管理）
