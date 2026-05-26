# muMemFreeAsync — 异步流式内存释放（源码逐行分析）

> 源码文件：`musa/src/driver/mu_memory.cpp:386-443`，`musa/src/musa/core/stream.cpp:582-638`，`musa/src/musa/core/memory.cpp:146-150, 216-265, 358-376`，`musa/src/musa/core/context.cpp:967-975`
> 
> 配套阅读：`07_muMemAllocAsync.md` (对应分配), `02_muMemFree_v2.md` (同步释放)

## 1. 功能概述

在指定的 stream 上**异步**释放内存。释放操作被编码为 stream 中的 **CallbackCommand**，在 stream 执行到该命令时完成释放。

## 2. Driver 入口源码逐行分析

```cpp
// mu_memory.cpp:386
MUresult muapiMemFreeAsync(MUdeviceptr dptr, MUstream hStream)
{
    MUresult status = InitPlatform();

    if (status == MUSA_SUCCESS) {
        do {
            if (0 == dptr) {
                break;                    // ⚠ dptr=0 → 静默成功 (无操作)
            }

            Musa::IContext* pContext = TlsCtxTop();
            if (pContext == nullptr) {
                status = MUSA_ERROR_INVALID_CONTEXT;
                break;
            }

            // 解析 Stream
            Musa::IStream* pStream = nullptr;
            pStream = Musa::Context::InfoStream(
                Musa::ICast<Musa::Context>(pContext),
                Musa::ICast<Musa::Stream>(hStream));
            if (!pStream) {
                status = MUSA_ERROR_INVALID_HANDLE;
                break;
            }

            // 通过全局 MemoryTracker 查找内存对象
            auto virtMem = Musa::Platform::Get()
                .GetMemoryByDevicePointer(dptr, nullptr);
            if (virtMem->get() == nullptr) {
                status = MUSA_ERROR_INVALID_VALUE;  // 未找到 → 失败
                break;
            }

            // 转换为 Memory* (带类型信息)
            auto memory = Musa::IntrusiveCast<Musa::Memory>(
                virtMem->get());

            // ── 按内存类型分类处理 ──
            switch (memory->GetType()) {

            case Musa::memoryTypeIpcImport:
            case Musa::memoryTypeExternal:
                // IPC Import / External → 同步销毁
                // ⚠ 注意: 这两个类型在 sync API 中是被拒绝的
                //   但在 async 路径中却允许
                // ⚠ DestroyMemory 仅从 ctx 列表移除, 不释放 KMD 资源
                //   (资源归属导出进程)
                status = pContext->DestroyMemory(memory);
                break;

            case Musa::memoryTypeGeneral:
            case Musa::memoryTypePitchedGeneral:
            case Musa::memoryTypeManaged:
                // General/PitchedGeneral/Managed → 同步等待 + 销毁
                // ⚠ ISSUE: 名义上是 "async", 但实际上
                //   Synchronize() 等待的是默认流 (不是 hStream!)
                status = memory->Synchronize();     // [memory.cpp:115]
                if (status == MUSA_SUCCESS) {
                    status = memory->GetContext()
                        ->DestroyMemory(memory);     // [context.cpp:967]
                }
                break;

            case Musa::memoryTypeVirtual:
                // Virtual (sub-allocated) → 真正的异步释放
                if (memory->GetPool() != nullptr) {
                    // ✅ 正常路径: pool 子分配的内存
                    status = pStream->CmdMemFree(
                        dptr, false);               // [stream.cpp:628]
                } else {
                    // ⚠ pool==nullptr → 非法
                    //   裸 virtual 内存 (未通过 pool 分配)
                    //   不允许通过此 API 释放
                    status = MUSA_ERROR_INVALID_VALUE;
                }
                break;

            default:
                status = MUSA_ERROR_INVALID_VALUE;
                break;
            }
        } while (0);
    }

    return status;
}
```

### 死代码/问题标注

```
⚠ ISSUE 1: General/PitchedGeneral/Managed 类型的 async 释放
   实际上是同步的 (memory->Synchronize() 等待默认流)
   不是真正的异步释放, 与函数名不符

⚠ ISSUE 2: IPC Import / External 类型在此进入 DestroyMemory
   但 DestroyMemory 仅从 ctx 列表中移除对象
   不释放底层 KMD 资源 (资源属于导出进程)
   这可能导致导出端释放后, 导入端仍持有引用
   (虽然 ExportIpcHandle 会缓存 global handle, 避免重复导出,
    但 Import 端从未通知 Export 端取消导出)
```

## 3. 路径 A: Virtual 内存释放 (核心路径)

```
Stream::CmdMemFree(dptr, false)                      [stream.cpp:628]
  │
  +── if (capture 模式):
  │     Stream::CaptureMemFree(virtAddress)            [stream.cpp:588]
  │       +── ctx->CreateMemFreeNode(&pGraphNode, virtAddress)
  │       +── captureGraph->AddGraphNode(pGraphNode, deps)
  │       +── SetLastCapturedNodes(&pGraphNode, 1)
  │
  +── else if (capture invalidated):
  │     MUSA_ERROR_STREAM_CAPTURE_INVALIDATED
  │
  +── else (正常路径):                                   [stream.cpp:635]
        Stream::AsyncMemFree(virtAddress, false)        [stream.cpp:601]
```

## 4. Stream::AsyncMemFree 源码

```cpp
// stream.cpp:601
MUresult Stream::AsyncMemFree(uint64_t virtAddress, bool blocking)
{
    MUresult status = MUSA_SUCCESS;

    // ── Step 1: 查找内存对象 ──
    Memory* virt = IntrusiveCast<Memory>(
        Platform::Get().GetMemoryByDevicePointer(
            virtAddress, nullptr)->get());
    Memory* physical = IntrusiveCast<Memory>(
        virt->GetPhysMemory(virtAddress)->get());
    //   ↑ 通过 m_PhysTracker 查找 virt 对应的 physical 内存

    MemoryPool* pPool = virt->GetPool();                 // [memory.h]
    // ↑ 获取 pool 指针 (用于后续归还)

    // ── Step 2: 禁用 GPU 对虚拟页的访问 ──
    status = pPool->DisableAccess(virt, physical,
                                  blocking, this);       // [memoryPool.h]
    //   ↑ 发送 PagingCommand (GPU page table update)
    //   ↑ 将对应虚拟页的访问权限设为 NONE
    //   ↑ 此步骤需要等待对应的 stream 命令执行完成

    // ── Step 3: 创建流完成回调 ──                       [stream.cpp:611-626]
    if (status == MUSA_SUCCESS) {
        std::shared_ptr<Command> command =
            make_shared<CallbackCommand>(this, function<void()>());

        std::function<void()> callback =
            [virt, command, pPool] () {
                // 回调在 stream 执行完成后异步执行

                virt->DestroyPhysMemories();             // [memory.cpp:146]
                //  ├─ SetProps(Physical) on virtual memory
                //  └─ m_PhysTracker.Cleanup()
                //      遍历所有物理子内存
                //      → DestroyPeerMemory() (销毁 peer view)
                //      → 清除跟踪记录

                if (!virt->IsGraphAlloc()) {             // [memory.h]
                    // 非 graph 分配的才归还 pool
                    // (graph 分配由 graph 执行框架管理)
                    pPool->DestroyMemory(virt);           // [memory.cpp:376]
                    //  → ~Memory() 析构
                    //  → 属性含 SubAllocatable?
                    //    ├─ YES → Pool::Free()
                    //    │        ├─ SegmentTracker 查找
                    //    │        ├─ 标记空闲 + 合并邻居
                    //    │        └─ ResourceRemove (尝试销毁 chunk)
                    //    └─ NO  → m_pHalMemory->Destroy()
                    //               (KMD ioctl 释放)
                }
            };

        static_cast<CallbackCommand*>(command.get())
            ->SetCallback(move(callback));

        status = m_ParentCtx->ResolveDependencyAndQueueCommand(
            move(command), this, blocking);
        //   ↑ 将 CallbackCommand 插入 stream 命令列表
        //   ↑ 依赖关系自动处理
        //   ↑ stream 执行到此命令时, callback 被调用
    }

    return status;
}
```

## 5. 释放流程时序图

```
Stream::AsyncMemFree(virtAddr, stream)
  │
  ├─ DisableAccess(virt, physical, stream)
  │     └─ PagingCommand 入队 (GPU 页表: 权限→NONE)
  │
  └─ CallbackCommand 入队 (stream 完成时回调)
        │
        ├─ [stream 执行中...]
        │     前面的命令依次执行
        │     PagingCommand 执行 → GPU 无法访问该虚拟页
        │
        └─ [CallbackCommand 执行]
              ├─ virt->DestroyPhysMemories()
              │     ├─ 遍历 PhysTracker 中所有物理子内存
              │     └─ DestroyPeerMemory (每个 peer device)
              │
              └─ pPool->DestroyMemory(virt)
                    ├─ ~Memory() 析构
                    ├─ 属性 & SubAllocatable?
                    │     ├─ YES → Pool::Free() → 合并/延迟销毁 chunk
                    │     └─ NO  → Hal::Destroy() → KMD ioctl
                    └─ delete this
```

## 6. GraphAlloc 标记检查

```cpp
// stream.cpp:615
if (!virt->IsGraphAlloc()) {
    pPool->DestroyMemory(virt);
}
```

```
IsGraphAlloc() 返回 true 的情况:
  - 通过 Stream::CaptureMemAlloc() 分配的虚拟内存
    (创建了 GraphMemAllocNode, 在图执行时实际分配)
  - 这类内存在 graph 释放时由 GraphMemFreeNode 统一处理
  - 此处不应重复归还 pool

IsGraphAlloc() 返回 false 的情况:
  - 通过常规 muMemAllocAsync/muMemAlloc 分配的内存
  - 需要在此处归还 pool
```

## 7. 同步 vs 异步释放对比

```
┌─────────────────────┬────────────────────────┬────────────────────────┐
│      特性           │  muMemFree (同步)       │  muMemFreeAsync        │
├─────────────────────┼────────────────────────┼────────────────────────┤
│ 释放时机            │ 立即等待完成            │ Stream 执行到该命令时   │
│ Synchronize()       │ 显式调用                │ 仅 General 类型调用     │
│ DisableAccess       │ 不需要 (直接销毁)       │ 需要 (PagingCommand)    │
│ 回调                │ 不需要                 │ CallbackCommand         │
│ 适用内存类型        │ 所有类型                │ Virtual 为主            │
│                     │                        │ (General 为伪异步)       │
│ 返回值              │ MUSA_SUCCESS/错误       │ MUSA_SUCCESS/错误       │
└─────────────────────┴────────────────────────┴────────────────────────┘
```

## 8. 相关源码位置

| 文件 | 行数 | 说明 |
|------|------|------|
| `musa/src/driver/mu_memory.cpp` | 386-443 | `muapiMemFreeAsync` 入口 (类型分发) |
| `musa/src/musa/core/stream.cpp` | 582-599 | `CaptureMemFree` (Graph Capture) |
| `musa/src/musa/core/stream.cpp` | 601-626 | `AsyncMemFree` (核心实现) |
| `musa/src/musa/core/stream.cpp` | 628-638 | `CmdMemFree` (分发) |
| `musa/src/musa/core/memory.cpp` | 146-150 | `DestroyPhysMemories` |
| `musa/src/musa/core/memory.cpp` | 216-265 | `Unbind` (解除映射) |
| `musa/src/musa/core/memory.cpp` | 358-376 | `~Memory` 析构 (双路径释放) |
| `musa/src/musa/core/context.cpp` | 967-975 | `DestroyMemory` (注销跟踪) |
| `musa/src/hal/m3d/memoryPool.cpp` | 214-259 | `Pool::Free` (子分配释放) |
| `musa/src/hal/m3d/memoryPool.cpp` | 480-510 | `TrimPool` (惰性释放) | |