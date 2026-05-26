# muMemAllocAsync — 异步流式内存分配（源码逐行分析）

> 源码文件：`musa/src/driver/mu_memory.cpp:303-340, 342-384`，`musa/src/musa/core/stream.cpp:502-580`
> 
> 配套阅读：`01_muMemAlloc_v2.md` (同步分配), `08_muMemFreeAsync.md` (异步释放)

## 1. 功能概述

在指定的 stream 上**异步**分配设备内存。分配操作被编码为 stream 命令流的一部分，流有序 (stream-ordered)。

变体:
| API | 说明 |
|-----|------|
| `muMemAllocAsync(dptr, bytesize, stream)` | 使用设备默认 pool |
| `muMemAllocFromPoolAsync(dptr, bytesize, pool, stream)` | 使用用户指定 pool |

## 2. Driver 入口源码逐行分析

```cpp
// mu_memory.cpp:303
MUresult muapiMemAllocAsync(MUdeviceptr *dptr,
                            size_t bytesize, MUstream hStream)
{
    MUresult status = InitPlatform();

    if (status == MUSA_SUCCESS) {
        if (nullptr == dptr) {
            status = MUSA_ERROR_INVALID_VALUE;   // dptr 为空
        } else if (0 == bytesize) {
            *dptr = 0;                           // size=0 返回 0
            // ⚠ 注意: size=0 不返回错误, 与 sync 版一致
        } else {
            Musa::IContext* pContext = TlsCtxTop();  // [internal.h:231]
            if (pContext == nullptr) {
                status = MUSA_ERROR_INVALID_CONTEXT;
            }

            // 解析 Stream
            Musa::IStream* pStream = nullptr;
            if (status == MUSA_SUCCESS) {
                pStream = Musa::Context::InfoStream(
                    Musa::ICast<Musa::Context>(pContext),
                    Musa::ICast<Musa::Stream>(hStream));
                if (!pStream) {
                    status = MUSA_ERROR_INVALID_HANDLE;
                }
            }

            if (status == MUSA_SUCCESS) {
                // 构造分配参数
                Musa::MemoryAllocParameter memAllocParam{};
                memAllocParam.size = bytesize;
                // pool = nullptr → 使用设备默认 pool

                // 提交到 Stream (非阻塞)
                status = pStream->CmdMemAlloc(memAllocParam, false);
                //                                  ↑ blocking=false
                if (status != MUSA_SUCCESS) {
                    *dptr = 0;
                } else {
                    *dptr = memAllocParam.virtAddress;
                    // 返回虚拟地址 (分配完成后有效)
                }
            }
        }
    }
    return status;
}
```

## 3. muMemAllocFromPoolAsync

```cpp
// mu_memory.cpp:342
MUresult muapiMemAllocFromPoolAsync(MUdeviceptr* dptr, size_t bytesize,
                                    MUmemoryPool pool, MUstream hStream)
{
    // 流程与 AllocAsync 相同, 差异:
    // 1. 参数校验增加: pool 不能是 imported pool
    //    pool->IpcMemPoolData().m_IsImported → INVALID_VALUE
    // 2. memAllocParam.pool = 指定的用户 pool

    Musa::MemoryPool* pMemoryPool =
        reinterpret_cast<Musa::MemoryPool*>(pool);

    // ... (其余流程相同)
    memAllocParam.pool = pMemoryPool;  // ← 指定 pool
    status = pStream->CmdMemAlloc(memAllocParam, false);
}
```

## 4. Stream::CmdMemAlloc 调用链

```cpp
// stream.cpp:570
MUresult Stream::CmdMemAlloc(MemoryAllocParameter& param,
                             bool blocking)
{
    MUresult status = MUSA_SUCCESS;
    if (m_CaptureStatus == MU_STREAM_CAPTURE_STATUS_ACTIVE) {
        // Graph Capture 模式
        status = CaptureMemAlloc(param);                        // [502]
    } else if (m_CaptureStatus == MU_STREAM_CAPTURE_STATUS_INVALIDATED) {
        status = MUSA_ERROR_STREAM_CAPTURE_INVALIDATED;
    } else {
        // 正常执行路径
        status = AsyncMemAlloc(param, blocking);               // [519]
    }
    return status;
}
```

## 5. Stream::CaptureMemAlloc (Graph Capture 模式)

```cpp
// stream.cpp:502
MUresult Stream::CaptureMemAlloc(MemoryAllocParameter& param)
{
    MUresult status = MUSA_SUCCESS;
    IGraphNode* pGraphNode;

    // 构造图节点参数
    MUSA_MEM_ALLOC_NODE_PARAMS memAllocParam = {};
    memAllocParam.bytesize = param.size;

    // 创建图节点 (不实际分配, 仅记录意图)
    status = m_ParentCtx->CreateMemAllocNode(
        &pGraphNode, memAllocParam);

    if (status == MUSA_SUCCESS) {
        // 实际地址在图执行时确定
        param.virtAddress = memAllocParam.dptr;

        // 添加到捕获图
        status = m_CaptureGraph->AddGraphNode(
            pGraphNode,
            m_LastCapturedNodes.data(),
            m_LastCapturedNodes.size());
    }

    if (status == MUSA_SUCCESS) {
        SetLastCapturedNodes(
            reinterpret_cast<IGraphNode**>(&pGraphNode), 1);
    }
    return status;
}
```

## 6. Stream::AsyncMemAlloc 核心实现

```cpp
// stream.cpp:519
MUresult Stream::AsyncMemAlloc(MemoryAllocParameter& param,
                               bool blocking)
{
    MUresult status = MUSA_SUCCESS;

    // ── Step 1: 确定物理页对齐大小 ──
    const uint64_t physPageSize =
        m_ParentCtx->GetParentDevice()->GetAllocationGranularity(
            MU_MEM_LOCATION_TYPE_DEVICE,
            MU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
    uint64_t allocSize = Util::AlignUp(param.size, physPageSize);
    // ⚠ 分配大小向上对齐到推荐粒度 (通常 4KB 或更大)

    // ── Step 2: 确定内存池 ──
    MemoryPool* pPool =
        param.pool != nullptr ?
            param.pool :                // 用户指定 pool
            GetParentCtx()->GetParentDevice()->GetMemoryPool();
        //                               // 设备默认 pool

    Memory* virt = nullptr;
    MUdeviceptr virtAddr = 0;

    if (pPool == nullptr) {
        status = MUSA_ERROR_OUT_OF_MEMORY;  // 无可用 pool
    } else {
        // 绑定 stream 到 pool
        pPool->SetStream(this);              // ← 关键: 关联 stream

        // ── Step 3: 从 pool 分配虚拟内存 ──
        status = pPool->CreateMemory(&virt, &virtAddr, allocSize);
        // 核心路径: Pool → FullAllocate → SubAllocate/ChunkAllocate
        // 返回: Memory* (虚拟内存对象) + MUdeviceptr (虚拟地址)
    }

    param.virtAddress = virtAddr;            // 返回虚拟地址

    // ── Step 4: 分配物理内存 (GPU 显存) ──
    std::shared_ptr<IMemory> spPhysical;
    spPhysical = make_shared<Memory>(m_ParentCtx);
    Memory* physical = IntrusiveCast<Memory>(spPhysical.get());

    Musa::MemoryCreateInfo createInfo{};
    createInfo.type = Musa::memoryTypeGeneral;
    createInfo.general.size = allocSize;
    createInfo.general.alignment = 0;
    // ⚠ flags = 0! 不含 SubAllocatable → 裸 KMD 分配
    createInfo.general.flags = 0;

    if (status == MUSA_SUCCESS) {
        status = physical->Init(createInfo);
        // 调用链: Init → GeneralAlloc(flags=0) → !SubAllocatable
        //                → Hal::CreateMemory(createInfo) → KMD ioctl
    }

    // ── Step 5: 绑定虚拟 ↔ 物理内存 ──
    if (status == MUSA_SUCCESS) {
        // 建立虚拟页到物理页的映射
        status = virt->Bind(spPhysical, allocSize, 0, 0);
        // [memory.cpp:152] Bind 实现:
        //   1. 导出物理内存 DMA-BUF
        //   2. 创建外部内存对象
        //   3. 设置虚拟内存为 Physical 属性
        //   4. Physical 设置为 Virtual 属性
        //   5. 在 m_PhysTracker 中跟踪映射关系

        if (status == MUSA_SUCCESS) {
            // 通知 GPU 新的映射关系
            status = pPool->ModifyAccess(virt, physical,
                                         allocSize, blocking, this);
            // 核心: 修改 GPU 页表, 设置访问权限
            // 失败时回滚
        }

        if (status != MUSA_SUCCESS) {
            // 回滚 Bind
            virt->Unbind(allocSize, virt->GetOffset(),
                        virt->GetParentCtx(), nullptr);
            pPool->DestroyMemory(virt);
        }
    } else {
        // Step 4 失败, 释放虚拟内存
        pPool->DestroyMemory(virt);
    }

    return status;
}
```

## 7. AsyncMemAlloc 执行流程图

```
Stream::AsyncMemAlloc(param, blocking)
  │
  ├─ 1. physPageSize = GetAllocationGranularity(RECOMMENDED)
  │     allocSize = AlignUp(param.size, physPageSize)
  │
  ├─ 2. pPool = param.pool ?: device.GetMemoryPool()
  │     pPool->SetStream(this)
  │
  ├─ 3. pPool->CreateMemory(&virt, &virtAddr, allocSize)     [A]
  │     │
  │     └─ MemoryPool::FullAllocate → SubAllocate/ChunkAllocate
  │       → 返回虚拟内存 (memoryTypeVirtual, pool 子分配)
  │
  ├─ 4. physical = new Memory(ctx)                           [B]
  │     physical->Init({type=General, size=allocSize, flags=0})
  │     │
  │     └─ GeneralAlloc(flags=0) → !SubAllocatable
  │       → Hal::CreateMemory() → KMD ioctl (裸分配)
  │       → GPU 显存中的物理内存
  │
  ├─ 5. virt->Bind(physical, allocSize, 0, 0)                [C]
  │     │
  │     ├─ 导出 physical DMA-BUF
  │     ├─ 用 DMA-BUF 创建 External Memory
  │     ├─ virt 添加 Physical 属性
  │     └─ physical 添加 Virtual 属性
  │
  ├─ 6. pPool->ModifyAccess(virt, physical, allocSize)       [D]
  │     │
  │     └─ 更新 GPU 页表 → 虚拟地址 → 物理地址映射
  │     └─ (失败 → Unbind + DestroyMemory 回滚)
  │
  └─ param.virtAddress = virtAddr (返回给调用者)

  [A]: Pool 子分配 (O(1), 极快)
  [B]: 裸 KMD 分配 (ioctl, 较慢)
  [C]: 建立虚拟↔物理绑定
  [D]: GPU 页表更新, 使 GPU 可见
```

## 8. 与同步分配的对比

```
┌─────────────────────┬────────────────────────┬────────────────────────┐
│      特性           │  muMemAlloc (同步)      │  muMemAllocAsync       │
├─────────────────────┼────────────────────────┼────────────────────────┤
│ 分配时机            │ 立即执行               │ Stream 执行到该命令时   │
│ 虚拟内存            │ 可选 (SubAlloc 路径)    │ 总是                   │
│ 物理内存            │ Pool 子分配             │ 裸 KMD 分配 (flags=0)  │
│ Bind/Unbind         │ 不需要                 │ 需要                   │
│ GPU 可见性          │ 分配后立即可见          │ Stream 执行后可见       │
│ 适用场景            │ 简单分配               │ 图执行/Stream Ordered   │
│ 返回值含义          │ GPU VA (SubAlloc)       │ GPU VA (虚拟地址)       │
│                     │ (立即有效)              │ (Stream 执行后有效)     │
└─────────────────────┴────────────────────────┴────────────────────────┘
```

## 9. 注意事项

### 9.1 返回值语义

```
*dptr = memAllocParam.virtAddress
```

- 这是**虚拟地址**, 不是物理地址
- 在 stream 执行到分配命令之前, 地址不可用
- 后续命令 (memset/memcpy) 可以立即使用此地址 (流有序保证)

### 9.2 隐式释放

```
AsyncMemAlloc 分配的内存, 在流的生命周期结束时需要显式释放:
  muMemFreeAsync(virtAddr, stream)  或  muMemFree(virtAddr)

如果没有释放, Context 析构时会 WARN:
  "WARNING: cleanup unfreed memory: %#llx"
```

### 9.3 Graph 场景

```
在 Graph Capture 模式中:
  - CaptureMemAlloc() 创建 GraphMemAllocNode (仅记录)
  - 图执行时展开为实际的 AsyncMemAlloc 调用
  - 图释放时对应 GraphMemFreeNode 触发 AsyncMemFree
```

## 10. 相关源码位置

| 文件 | 行数 | 说明 |
|------|------|------|
| `musa/src/driver/mu_memory.cpp` | 303-340 | `muapiMemAllocAsync` 入口 |
| `musa/src/driver/mu_memory.cpp` | 342-384 | `muapiMemAllocFromPoolAsync` 入口 |
| `musa/src/musa/core/stream.cpp` | 502-516 | `CaptureMemAlloc` |
| `musa/src/musa/core/stream.cpp` | 519-580 | `AsyncMemAlloc` 核心 |
| `musa/src/musa/core/stream.cpp` | 570-580 | `CmdMemAlloc` 分发 |
| `musa/src/musa/core/memory.cpp` | 152-214 | `Memory::Bind` (虚拟↔物理) |
| `musa/src/musa/core/memory.cpp` | 216-265 | `Memory::Unbind` (解除绑定) |
| `musa/src/musa/core/memory.cpp` | 427-453 | `InitFromPool` (Pool 分配) |
| `musa/src/musa/core/memory.cpp` | 462-497 | `GeneralAlloc` (物理分配) |