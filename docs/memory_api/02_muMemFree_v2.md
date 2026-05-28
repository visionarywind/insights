# muMemFree_v2 — 设备内存释放

## 功能

释放 `muMemAlloc` 分配的 GPU 设备内存，等待所有 GPU 访问完成后归还内存。对应 CUDA `cudaFree`。

## 完整调用链

```
用户代码: muMemFree_v2(dptr)
  │
  ├─ 1. mu_wrappers_generated.cpp — MUPTI 插桩
  │
  ├─ 2. mu_memory.cpp:709 — muapiMemFree_v2
  │    ├─ InitPlatform()
  │    ├─ GetMemoryByDevicePointer(dptr, &offset)  → Memory*  (通过 MemoryTracker 反查)
  │    ├─ 类型检查: 必须是 General/PitchedGeneral/Managed/External/Virtual
  │    ├─ offset != 0 → 拒绝释放 (sub-allocation 不允许单独释放)
  │    ├─ pMemory->Synchronize()                     → 等待所有 GPU 操作完成
  │    │    └─ m_Context->LockedWait()               → 等待所有 stream in-flight 命令
  │    │    └─ for each peerDevice: peerCtx->Synchronize()
  │    └─ pContext->DestroyMemory(pMemory)           → 归还/销毁
  │
  ├─ 3. context.cpp:967 — Context::DestroyMemory
  │    ├─ RemoveMemory(pMem) — 从 context 的 m_Memories set 中删除
  │    └─ TrackMemory.UntrackMemory(pMemory) — 从全局区间 map 删除
  │
  └─ 4. memory.cpp:358 — ~Memory() (shared_ptr 引用归零时自动调用)
       ├─ m_pHalMemory->Unmap()                    (如果 map 过 host)
       ├─ [subAllocatable] → MemMgr::Free          (归还给 pool)
       │    └─ IMemoryPool::Free(pHalMem, devAddr, size)
       │         └─ ResSegment 回收 → 空闲链表
       └─ [else] → m_pHalMemory->Destroy()         (直接销毁)
            └─ m3dDevice->DestroyGpuMemory()       → KMD 释放
```

## 时序图

```
应用层            Wrapper            Driver(mu_memory)     Context            Memory          MemMgr/M3D          KMD
  │                │                  │                    │                  │               │                   │
  │ muMemFree_v2   │                  │                    │                  │               │                   │
  │───────────────>│                  │                    │                  │               │                   │
  │                │                  │                    │                  │               │                   │
  │                │ muapiMemFree_v2  │                    │                  │               │                   │
  │                │─────────────────>│                    │                  │               │                   │
  │                │                  │                    │                  │               │                   │
  │                │                  │ InitPlatform()     │                  │               │                   │
  │                │                  │ (lazy init, 通常已初始化)            │               │                   │
  │                │                  │                    │                  │               │                   │
  │                │                  │ GetMemoryByDevicePointer(dptr)       │               │                   │
  │                │                  │───────────────────>│                  │               │                   │
  │                │                  │                    │ MemoryTracker::  │               │                   │
  │                │                  │                    │ FindRange(dptr)  │               │                   │
  │                │                  │                    │── 区间树查找 ───>│               │                   │
  │                │                  │                    │<── IMemory* ─────│               │                   │
  │                │                  │<── Memory* ────────│                  │               │                   │
  │                │                  │                    │                  │               │                   │
  │                │                  │ 类型检查:          │                  │               │                   │
  │                │                  │ 必须是 General/     │                  │               │                   │
  │                │                  │ Pitched/Managed/   │                  │               │                   │
  │                │                  │ External/Virtual   │                  │               │                   │
  │                │                  │                    │                  │               │                   │
  │                │                  │ offset==0?         │                  │               │                   │
  │                │                  │  sub-alloc 偏移    │                  │               │                   │
  │                │                  │  不能单独 free     │                  │               │                   │
  │                │                  │                    │                  │               │                   │
  │                │                  │ Synchronize()      │                  │               │                   │
  │                │                  │───────────────────>│─────────────────>│               │                   │
  │                │                  │                    │                  │               │                   │
  │                │                  │                    │                  │ LockedWait()   │                   │
  │                │                  │                    │                  │── 等待本 context │                  │
  │                │                  │                    │                  │  所有 stream    │                  │
  │                │                  │                    │                  │  inflight 完成   │                  │
  │                │                  │                    │                  │               │                   │
  │                │                  │                    │                  │ for each peer:  │                  │
  │                │                  │                    │                  │  peerCtx->Sync  │                  │
  │                │                  │                    │                  │── 等 peer device│                  │
  │                │                  │                    │                  │  上的 context   │                  │
  │                │                  │                    │                  │               │                   │
  │                │                  │                    │<── OK ───────────│               │                   │
  │                │                  │<── OK ─────────────│                  │               │                   │
  │                │                  │                    │                  │               │                   │
  │                │                  │ DestroyMemory()    │                  │               │                   │
  │                │                  │───────────────────>│                  │               │                   │
  │                │                  │                    │                  │               │                   │
  │                │                  │                    │ RemoveMemory()   │               │                   │
  │                │                  │                    │── m_Memories     │               │                   │
  │                │                  │                    │  .erase(pMemory) │               │                   │
  │                │                  │                    │                  │               │                   │
  │                │                  │                    │ UntrackMemory()  │               │                   │
  │                │                  │                    │── MemoryTracker  │               │                   │
  │                │                  │                    │  删除区间映射     │               │                   │
  │                │                  │                    │                  │               │                   │
  │                │                  │                    │ [Memory 析构]    │               │                   │
  │                │                  │                    │ (shared_ptr      │               │                   │
  │                │                  │                    │  引用归零)       │               │                   │
  │                │                  │                    │ ~Memory()        │               │                   │
  │                │                  │                    │─────────────────>│               │                   │
  │                │                  │                    │                  │               │                   │
  │                │                  │                    │                  │ Unmap()        │                   │
  │                │                  │                    │                  │ (if m_pMapped) │                   │
  │                │                  │                    │                  │               │                   │
  │                │                  │                    │                  │ [SubAlloc?]    │                   │
  │                │                  │                    │                  │   │            │                   │
  │                │                  │                    │                  │   │ YES         │                   │
  │                │                  │                    │                  │   │ MemMgr::    │                   │
  │                │                  │                    │                  │   │ Free()──────>│                   │
  │                │                  │                    │                  │   │              │ IMemoryPool::    │
  │                │                  │                    │                  │   │              │ Free(pHalMem,    │
  │                │                  │                    │                  │   │              │   devAddr, size)  │
  │                │                  │                    │                  │   │              │ 回收 → ResSegment│
  │                │                  │                    │                  │   │              │ 空闲链表          │
  │                │                  │                    │                  │   │<── OK ───────│                   │
  │                │                  │                    │                  │   │              │                   │
  │                │                  │                    │                  │   │ NO            │                   │
  │                │                  │                    │                  │   │ pMemory->     │                   │
  │                │                  │                    │                  │   │ Destroy()────>│                   │
  │                │                  │                    │                  │   │               │ DestroyGpuMemory() │
  │                │                  │                    │                  │   │               │──────────────────>│──DRM释放
  │                │                  │                    │                  │   │               │<── OK ────────────│
  │                │                  │                    │                  │   │<── OK ────────│                   │
  │                │                  │                    │<── OK ──────────│               │                   │
  │                │                  │<── OK ─────────────│                  │               │                   │
  │                │<── OK ───────────│                    │                  │               │                   │
  │<── OK ─────────│                  │                    │                  │               │                   │
```

## 关键代码路径

### Driver 入口

```cpp
// mu_memory.cpp:709
MUresult muapiMemFree_v2(MUdeviceptr dptr) {
    // 1. MemoryTracker::FindRange(dptr, &offset) → IMemory*
    //    通过全局区间树反查 dptr 对应的 Memory 对象 + sub-allocation 偏移
    //    如果没找到 → MUSA_ERROR_INVALID_VALUE
    // 2. 类型校验
    //    只接受: General/PitchedGeneral/Managed/External/Virtual(with Pool)
    //    offset != 0 → 拒绝（sub-allocation 不可单独释放）
    // 3. pMemory->Synchronize()
    //    等待所有 GPU 操作完成（包括 peer device）
    // 4. 分叉:
    //    Virtual → pStream->CmdMemFree(dptr, blocking)
    //    其他    → pMemory->GetContext()->DestroyMemory(pMemory)
}
```

### Context::DestroyMemory

```cpp
// context.cpp:967
MUresult Context::DestroyMemory(IMemory* pMemory) {
    Memory* pMem = static_cast<Memory*>(pMemory);
    if (pMem->GetParentCtx()) {
        WriteLockedAccessor<CriticalBase> ctxCrit(pMem->GetParentCtx()->CriticalData());
        ctxCrit->RemoveMemory(pMem);  // 从 context 的 m_Memories unordered_set 中删除
    }
    Platform::Get().GetMemoryTracker().UntrackMemory(pMemory);  // 从全局区间树删除
    return MUSA_SUCCESS;  // 不 delete! shared_ptr 引用归零时自动调用 ~Memory
}
```

### Memory 析构 — 真正的资源回收

```cpp
// memory.cpp:358
Memory::~Memory() {
    if (m_pMapped) {
        m_pHalMemory->Unmap();  // 解除 CPU 映射
    }
    if (m_Type != memoryTypePrealloc && m_pHalMemory) {
        if (m_pHalMemory->GetProps() & Hal::memoryPropertySubAllocatable) {
            // ① sub-allocated: 归还给 pool
            if (m_pPool == nullptr)
                m_Context->GetParentDevice()->Hal().GetMemMgr()
                    ->Free(m_pHalMemory, GetDevicePointer(), GetSize());
            else
                m_pPool->Hal()->Free(m_pHalMemory, GetDevicePointer(), GetSize());
        } else {
            // ② 独立分配: 直接在 HAL 层销毁
            m_PhysTracker.Cleanup();  // 清理虚拟内存的物理绑定
            m_pHalMemory->Destroy();  // → m3dDevice->DestroyGpuMemory()
        }
    }
}
```

### Memory::Synchronize

```cpp
// memory.cpp:115
MUresult Memory::Synchronize() const {
    if (m_Type == memoryTypeVirtual) {
        // 遍历所有物理绑定块, 逐个等待
        m_PhysTracker.IterateMemories(..., [](ptr, memory) {
            return memory->GetContext()->LockedWait();
        });
    } else {
        // 1. 等待本 context 所有 stream 完成
        m_Context->LockedWait();
        // 2. 遍历所有建立了 peer mapping 的 device
        for each peerDevice:
            for each peerCtx on peerDevice:
                peerCtx->Synchronize();
    }
}
```

## 关键设计要点

1. **MemoryTracker 反查**: 通过 `dptr` 在全局区间树中找到对应的 `Memory*`，这是所有释放/拷贝 API 的第一步
2. **Synchronize 确保安全**: 释放前等待所有 GPU 操作完成（包括 peer device 上的 context），防止 use-after-free
3. **Sub-allocation 防护**: `offset != 0` 即不是完整的分配起始地址时拒绝释放，防止破坏 pool 内部状态
4. **双路径释放**:
   - `MemMgr::Free` — sub-allocated 内存归还给 pool，下次可复用，不涉及 KMD
   - `m_pHalMemory->Destroy()` — 独立分配直接通知 KMD 释放 GPU 物理内存
5. **shared_ptr 生命周期**: DestroyMemory 只做 bookkeeping，真正析构在 `shared_ptr` 引用归零时，这样 MemoryTracker 和其他持有 shared_ptr 的地方不会悬空
