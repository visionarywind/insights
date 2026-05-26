# muMemAlloc / muMemAlloc_v2 — 设备内存分配（源码逐行分析）

> 源码文件：`musa/src/driver/mu_memory.cpp:265-301`，`musa/src/musa/core/memory.cpp:378-425, 462-497`，`musa/src/hal/m3d/memMgr.cpp:81-147`，`musa/src/hal/m3d/memoryPool.cpp:82-151`，`musa/src/hal/m3d/memory.cpp:366-426`
> 
> 详细内存分配链路分析见同目录 **11_GeneralAlloc_deep_dive.md**。本文件聚焦 API 入口到 GeneralAlloc 的完整调用链。

## 1. 功能概述

`muMemAlloc` 在当前设备的主上下文中分配一块设备全局内存，返回 `MUdeviceptr`。

| API | 签名差异 |
|-----|---------|
| `muapiMemAlloc` (v1) | `bytesize` 为 `unsigned int` |
| `muapiMemAlloc_v2` (v2) | `bytesize` 为 `size_t` |

两者最终调用 `muapiMemAlloc_v2` 实现。

默认行为：
- 内存类型 = `memoryTypeGeneral`（设备本地）
- flags = `Virtual | DeviceMapped | SubAllocatable` = 0x07
- 走 **Sub-Allocation 路径**（通过 `MemMgr::Allocate` → `MemoryPool::FullAllocate`）

## 2. Driver 入口源码逐行分析

```cpp
// mu_memory.cpp:265
MUresult muapiMemAlloc_v2(MUdeviceptr *dptr, size_t bytesize) {
    MUresult status = InitPlatform();                        // [internal.h:306]
    // 作用: 检查 GetTlsData().CheckEntry() + Musa::GetPlatform()

    if (status == MUSA_SUCCESS) {
        if (nullptr == dptr) {
            status = MUSA_ERROR_INVALID_VALUE;               // dptr 为空 → 失败
        } else if (0 == bytesize) {
            *dptr = 0;
            status = MUSA_ERROR_INVALID_VALUE;               // size=0 → 失败
        } else {
            Musa::IContext* pContext = TlsCtxTop();           // [internal.h:231]
            // 从 TLS 获取当前线程的 Context*
            // ⚠ TLS 机制允许同一线程在不同 Context 间切换
            if (nullptr == pContext)  {
                status = MUSA_ERROR_INVALID_CONTEXT;         // 无活跃上下文
            } else {
                Musa::MemoryCreateInfo createInfo{};
                createInfo.type = Musa::memoryTypeGeneral;    // 设备本地内存
                createInfo.general.size = bytesize;
                createInfo.general.alignment = 0;
                createInfo.general.flags = Hal::memoryPropertyVirtual |
                                           Hal::memoryPropertyDeviceMapped |
                                           Hal::memoryPropertySubAllocatable;
                // flags = 0x07 → 默认启用三层属性:
                //   0x01 Virtual        → Core 推导 DeviceVisible 等
                //   0x02 DeviceMapped   → Core 推导 PeerAccessible | IpcExportable
                //   0x04 SubAllocatable → 走 Pool 子分配路径

                Musa::IMemory* pMemory;
                status = pContext->CreateMemory(&pMemory, createInfo);
                // [context.cpp:915] 统一内存创建入口
                if (status == MUSA_SUCCESS) {
                    *dptr = pMemory->GetDevicePointer();      // [memory.cpp]
                    // Sub-Allocation: base + m_Offset (chunk 内偏移)
                    // 裸 KMD: GPU VA (m_Offset 恒为 0)
                }
            }
        }
    }
    return status;
}
```

## 3. v1 封装

```cpp
// mu_memory.cpp:299
MUresult muapiMemAlloc(MUdeviceptr_v1 *dptr, unsigned int bytesize) {
    return muapiMemAlloc_v2(reinterpret_cast<MUdeviceptr*>(dptr),
                            static_cast<size_t>(bytesize));
    // 仅做类型转换，委托 v2
}
```

## 4. CreateMemory 调用链

```
pContext->CreateMemory(&pMemory, createInfo)                 [context.cpp:915]
  │
  ├─ Step 0: Capture 状态检查                                 [context.cpp:922-931]
  │     遍历 ctx 的所有 stream
  │     若有 stream 处于 ACTIVE capture → MUSA_ERROR
  │     若有 stream 处于 INVALIDATED → MUSA_ERROR
  │
  ├─ Step 1: 创建 Memory 对象                                [context.cpp:937]
  │     memory_sp = make_shared<Memory>(this)
  │     pMemory = static_cast<Memory*>(memory_sp.get())
  │
  ├─ Step 2: 初始化                                          [context.cpp:939]
  │     pMemory->Init(createInfo)  →  Memory::Init()         [memory.cpp:378]
  │       │
  │       m_Type = memoryTypeGeneral                          [memory.cpp:383]
  │       │
  │       └─ GeneralAlloc(size=bytesize, alignment=0,        [memory.cpp:462]
  │                        flags=0x07)
  │           │
  │           +-- 构建 Hal::MemoryCreateInfo                  [memory.cpp:467-489]
  │           │     type      = memoryTypeAlloc
  │           │     alloc.type = memoryAllocTypeDeviceLocal
  │           │     alloc.heap = MemoryHeap::largePage
  │           │     alloc.property = 0x07 (用户传入)
  │           │       │
  │           │       ├── Core 追加: 0x08 Physical
  │           │       ├── Core 追加: 0x10 SharedVirtualAddress
  │           │       │
  │           │       ├── 由 Virtual(0x01) 推导:
  │           │       │     + 0x20 DeviceVisible
  │           │       │     + 0x40 HostVisible
  │           │       │     + 0x80 HostCoherent
  │           │       │     + 0x100 DeviceWriteable
  │           │       │     + 0x200 DeviceCached
  │           │       │
  │           │       └── 由 DeviceMapped(0x02) 推导:
  │           │             viewCapability:
  │           │             + 0x02 PeerAccessible
  │           │             + 0x04 IpcExportable
  │           │
  │           │     最终 property = 0x3FF
  │           │     最终 viewCapability = 0x07
  │           │
  │           +-- alignment = max(0, minAllocAlign)            [memory.cpp:488]
  │           │
  │           +-- 分叉:                                        [memory.cpp:491]
  │           │     property & SubAllocatable (0x04) → YES
  │           │     ├─ YES → MemMgr::Allocate()               [memory.cpp:492]
  │           │     │         走 Pool 子分配路径
  │           │     │         (详见 11_GeneralAlloc_deep_dive.md)
  │           │     │
  │           │     └─ NO  → Hal::CreateMemory()              [memory.cpp:494]
  │           │               走裸 KMD 分配
  │           │               (详见 12_DirectKMD_Allocation_flow.md)
  │
  ├─ Step 3: Peer 映射 (仅在 capability 包含 PeerAccessible 时)  [context.cpp:944-945]
  │     ctxCrit->MapToPeers(pMemory)
  │     遍历所有 peer device → Hal::OpenPeerMemory()
  │     (详见 10_CreateMemory_and_MapToPeers.md)
  │
  ├─ Step 4: 加入上下文管理                                   [context.cpp:948]
  │     ctxCrit->AddMemory(pMemory)
  │     m_Memories.insert(pMemory)
  │
  ├─ Step 5: 加入全局跟踪器                                   [context.cpp:957]
  │     MemoryTracker::TrackMemory(memory_sp)
  │     建立 MUdeviceptr → shared_ptr<IMemory> 映射
  │
  └─ Step 6: 分配 SeqID                                       [context.cpp:959]
        if (!pMemory->IsPhysical()):
          Platform::Get().SetMemorySeqID(pMemory)
```

## 5. 返回值

```cpp
*dptr = pMemory->GetDevicePointer()                          // [mu_memory.cpp:290]
```

| 分配路径 | GetDevicePointer() 返回值 |
|---------|--------------------------|
| Sub-Allocation | `m_pHalMemory->GetDeviceVirtualAddress() + m_Offset` |
| 裸 KMD | `m_pHalMemory->GetDeviceVirtualAddress()` (m_Offset=0) |

## 6. 内存类型矩阵

| 内存类型 | createInfo.type | 是否走 Pool | 物理位置 | 典型 API |
|---------|----------------|------------|---------|---------|
| General | memoryTypeAlloc | YES (SubAllocatable) | GPU 显存 | muMemAlloc |
| PitchedGeneral | memoryTypeAlloc | YES (SubAllocatable) | GPU 显存 | muMemAllocPitch |
| PinnedHost | memoryTypeAlloc | YES (若 SubAllocatable) | 系统内存 | muMemHostAlloc |
| RegisteredPinned | memoryTypeView | NO | 系统内存 | muMemHostRegister |
| Managed | memoryTypeAlloc | NO | GPU/系统 | muMemAllocManaged |
| IPC Import | memoryTypeView | NO | GPU 显存 | muMemImportShareableHandle |
| External | memoryTypeView | NO | 外部 | muMemImportShareableHandle |
| Virtual | memoryTypeVirtual | NO | 仅 VA 空间 | muMemAddressReserve |

## 7. 三层内存分配器架构

```
┌──────────────────────────────────────────────────────────────┐
│  L3: Musa::MemoryPool (用户池)                                │  ← muMemCreatePool
│     用户可创建自定义池, 通过 muMemAllocFromPoolAsync 使用     │
├──────────────────────────────────────────────────────────────┤
│  L2: Hal::IMemMgr + MemoryPool (系统池)                       │  ← 默认路径
│     按 {type,heap,property,viewCapability,numaId} 查找/创建   │
│     首次创建 2MB chunk, 后续 O(1) 子分配                      │
├──────────────────────────────────────────────────────────────┤
│  L1: Hal::IMemory (裸 KMD 分配)                              │  ← 去掉 SubAllocatable flag
│     每次分配至少 2 次 ioctl                                  │
│     (mtgpuBoAlloc + mtgpuBoVmMapV2)                          │
└──────────────────────────────────────────────────────────────┘
```

## 8. Sub-Allocation 性能对比

```
场景: 1000 次 4KB 分配 + 释放

裸 KMD (去掉 SubAllocatable):
  每次: AllocBuffer(ioctl) + MapVirtualAddress(ioctl) = 2 ioctl
  1000 次: 2000 ioctl ≈ 200ms

Sub-Allocation (默认):
  首次: 创建 Pool → 2 ioctl (AllocBuffer + MapVirtualAddress for 2MB chunk)
  后续: 内存操作 (哈希查找 + 链表操作) ≈ O(1)
  1000 次: 2 ioctl + ~0ms ≈ 0.4ms

性能提升: ~500x
```

## 9. 相关源码位置

| 文件 | 行数 | 说明 |
|------|------|------|
| `musa/src/driver/mu_memory.cpp` | 265-301 | Driver 入口 (muapiMemAlloc_v2) |
| `musa/src/driver/mu_memory.cpp` | 299-301 | v1 封装 |
| `musa/src/driver/internal.h` | 306-316 | InitPlatform |
| `musa/src/driver/internal.h` | 231 | TlsCtxTop |
| `musa/src/musa/core/context.cpp` | 915-965 | CreateMemory 统一入口 |
| `musa/src/musa/core/memory.cpp` | 378-425 | Memory::Init (类型分发) |
| `musa/src/musa/core/memory.cpp` | 462-497 | GeneralAlloc (Core 分配实现) |
| `musa/src/hal/m3d/memMgr.cpp` | 81-147 | MemMgr::Allocate |
| `musa/src/hal/m3d/memoryPool.cpp` | 82-212 | Pool::FullAllocate/SubAllocate/ChunkAllocate |
| `musa/src/hal/m3d/memory.cpp` | 366-426 | InitGeneralDeviceMemory (HAL 层) |

> **详细分叉分析**: Sub-Allocation 池化算法 → `11_GeneralAlloc_deep_dive.md`
> **裸 KMD 分配全链路**: → `12_DirectKMD_Allocation_flow.md`
> **MemoryPool 子分配算法**: → `13_MemoryPool_deep_dive.md`