# muMemHostAlloc — 主机页锁定内存分配（源码逐行分析）

> 源码文件：`musa/src/driver/mu_memory.cpp:59-101, 489-499`，`musa/src/musa/core/memory.cpp:532-609`

## 1. 功能概述

分配主机端**页锁定 (pinned)** 内存，使其可以被 GPU 直接通过 DMA 访问。

三个入口函数最终都汇聚到 `imuapiMemHostAlloc`：

| API | 最终 flags |
|-----|-----------|
| `muMemAllocHost_v2` | `MU_MEMHOSTALLOC_PORTABLE \| MU_MEMHOSTALLOC_DEVICEMAP` |
| `muMemAllocHost` | 同上 (v1 封装) |
| `muMemHostAlloc` | 用户自定义 flags |

## 2. imuapiMemHostAlloc 源码逐行分析

```cpp
// mu_memory.cpp:59
static MUresult imuapiMemHostAlloc(void **pp, size_t bytesize,
                                   unsigned int Flags) {
    MUresult status = InitPlatform();                          // [internal.h:306]

    if (status == MUSA_SUCCESS) {
        if (0 != bytesize) {
            // 1. flags 掩码校验
            int userFlagsMask = MU_MEMHOSTALLOC_PORTABLE |
                                MU_MEMHOSTALLOC_DEVICEMAP |
                                MU_MEMHOSTALLOC_WRITECOMBINED;
            if (Flags & ~userFlagsMask) {
                // ⚠ 只允许 PORTABLE / DEVICEMAP / WRITECOMBINED 三个 flag
                //   其他 bit 一律视为非法
                status = MUSA_ERROR_INVALID_VALUE;
            } else if (nullptr == pp) {
                status = MUSA_ERROR_INVALID_VALUE;             // 输出指针为空
            } else {
                Musa::IContext* pContext = TlsCtxTop();        // [internal.h:231]
                if (nullptr == pContext)  {
                    status = MUSA_ERROR_INVALID_CONTEXT;       // 无活跃上下文
                } else {
                    // 2. 构造 MemoryCreateInfo
                    Musa::MemoryCreateInfo createInfo{};
                    createInfo.type = Musa::memoryTypePinnedHost;
                    createInfo.pinnedHost.size = bytesize;
                    // NOTE: 无论用户是否指定 PORTABLE, 都追加
                    //   NUMA_AFFINITIVE: 在 NUMA 节点上本地分配
                    //   SUBALLOCATABLE: 允许子分配池化
                    createInfo.pinnedHost.flags = Flags |
                        (Musa::MU_MEMORY_HOSTALLOC_NUMA_AFFINITIVE |
                         Musa::MU_MEMORY_HOSTALLOC_SUBALLOCATABLE);
                    createInfo.pinnedHost.numaId = NUMA_NO_NODE;
                    //        ↑ 默认不指定 NUMA, 由运行时选择

                    // 3. 调用统一创建入口
                    Musa::IMemory* pMemory;
                    status = pContext->CreateMemory(&pMemory, createInfo);
                    // [context.cpp:915]
                    if (status == MUSA_SUCCESS) {
                        *pp = pMemory->GetHostPointer();        // 获取主机虚拟地址
                    }
                }
            }
        } else if (nullptr != pp) {                             // bytesize == 0
            *pp = nullptr;                                      // size=0 → 返回 nullptr
        }
    }
    return status;
}
```

## 3. v1/v2 封装

```cpp
// mu_memory.cpp:489
MUresult muapiMemAllocHost_v2(void **pp, size_t bytesize) {
    return imuapiMemHostAlloc(pp, bytesize,
        MU_MEMHOSTALLOC_PORTABLE | MU_MEMHOSTALLOC_DEVICEMAP);
    // 默认 flags: PORTABLE + DEVICEMAP
}

// mu_memory.cpp:493
MUresult muapiMemAllocHost(void **pp, unsigned int bytesize) {
    return muapiMemAllocHost_v2(pp, static_cast<size_t>(bytesize));
    // v1 → v2, 仅类型转换
}

// mu_memory.cpp:497
MUresult muapiMemHostAlloc(void **pp, size_t bytesize,
                           unsigned int Flags) {
    return imuapiMemHostAlloc(pp, bytesize, Flags);
    // 直接透传用户 flags
}
```

## 4. MemoryCreateInfo 到 HAL 的完整链路

```
Context::CreateMemory(&pMemory, createInfo)                  [context.cpp:915]
  │
  ├─ (通用流程已在 01 中详述，此处聚焦 PinnedHost 分叉)
  │
  └─ pMemory->Init(createInfo)                               [memory.cpp:378]
        │
        ├─ m_Type = memoryTypePinnedHost                      [memory.cpp:392]
        │
        └─ PinnedHostAlloc(size, flags, numaId)              [memory.cpp:532]
```

## 5. PinnedHostAlloc 源码逐行分析

```cpp
// memory.cpp:532
MUresult Memory::PinnedHostAlloc(size_t size,
                                 unsigned int flags, int numaId)
{
    // ──────────────────────────────────────────
    // Step 1: 能力校验
    // ──────────────────────────────────────────
    const Hal::DeviceProperties& deviceProperties =
        m_Context->GetParentDevice()->Hal().GetProperties();

    if ((flags & MU_MEMORY_HOSTALLOC_PORTABLE) &&
        !deviceProperties.memoryProperties.unifiedAddressing) {
        // ⚠ 仅 WARN, 不报错: portable 在非 UVA 设备上仅本地有效
        tprintf(LOG_WARN, "the host memory would not be portable "
                "on all MUSA contexts\n");
    }
    if ((flags & MU_MEMORY_HOSTALLOC_DEVICEMAP) &&
        !deviceProperties.memoryProperties.canMapHostMemory) {
        // 设备不支持映射主机内存 → 硬错误
        tprintf(LOG_ERR, "device cannot access the host memory\n");
        status = MUSA_ERROR_NOT_SUPPORTED;
    }
    if (deviceProperties.memoryProperties.canMapHostMemory) {
        // 设备支持映射 → 自动追加 DEVICEMAP
        flags |= MU_MEMORY_HOSTALLOC_DEVICEMAP;
    }

    if (status == MUSA_SUCCESS) {
        // ──────────────────────────────────────────
        // Step 2: 构建 MemoryCreateInfo
        // ──────────────────────────────────────────
        m_Shape = { size, 1, 1, size };
        m_Flags = flags;

        // heap 选择 (static 变量, 进程生命周期内生效)
        static Hal::MemoryHeap pinnedHostAllocHeap =
            Hal::MemoryHeap::largePage;
        // ⚠ 一旦因 MAP_FAILED 降级为 general, 后续所有分配都用 general

        Hal::MemoryCreateInfo createInfo{};
        createInfo.type = Hal::memoryTypeAlloc;
        createInfo.alloc.type = Hal::memoryAllocTypeHost;    // ← HOST, 不是 DeviceLocal
        createInfo.alloc.heap = pinnedHostAllocHeap;          // largePage 或降级后 general

        // 属性组合:
        createInfo.alloc.property =
            Hal::memoryPropertyPhysical |                     // 物理内存
            Hal::memoryPropertySharedVirtualAddress |         // 共享虚拟地址
            Hal::memoryPropertyHostMapped;                    // CPU 需要映射

        if (flags & MU_MEMORY_HOSTALLOC_DEVICEMAP) {
            // 设备映射 → 追加 GPU 可见/可写属性
            createInfo.alloc.property |=
                Hal::memoryPropertyVirtual |
                Hal::memoryPropertyHostVisible |
                Hal::memoryPropertyDeviceVisible |
                Hal::memoryPropertyDeviceWriteable |
                Hal::memoryPropertyDeviceCached;
        } else {
            // 无设备映射 → 标记为纯物理分配
            createInfo.alloc.property |=
                Hal::memoryPropertyIsPhysAlloc;
        }

        // 写策略
        if (flags & MU_MEMORY_HOSTALLOC_WRITECOMBINED) {
            createInfo.alloc.property |=
                Hal::memoryPropertyHostWriteCombined;
        } else {
            createInfo.alloc.property |=
                Hal::memoryPropertyHostCoherent;
        }

        // 子分配能力 (从 flags 继承)
        if (flags & MU_MEMORY_HOSTALLOC_SUBALLOCATABLE) {
            createInfo.alloc.property |=
                Hal::memoryPropertySubAllocatable;
        }

        // NUMA 亲和性
        if (flags & MU_MEMORY_HOSTALLOC_NUMA_CURRENT) {
            createInfo.alloc.property |=
                Hal::memoryPropertyThreadNumaAffinitive;
        }

        // view capability
        createInfo.alloc.viewCapability =
            (flags & MU_MEMORY_HOSTALLOC_DEVICEMAP) ?
            (Hal::memoryViewCapabilityPeerAccessible |
             Hal::memoryViewCapabilityIpcExportable |
             Hal::memoryViewCapabilityExportable) :
            Hal::memoryViewCapabilityNone;

        createInfo.alloc.size = size;
        createInfo.alloc.alignment =
            m_Context->GetParentDevice()->Hal()
                .GetProperties().memoryProperties.memAllocAlignment;
        createInfo.alloc.numaId = numaId;

        // ──────────────────────────────────────────
        // Step 3: 分配 (同 GeneralAlloc 的分叉逻辑)
        // ──────────────────────────────────────────
        if (createInfo.alloc.property &
            Hal::memoryPropertySubAllocatable) {
            // 路径 A: Pool 子分配
            status = HalToMuResult(
                m_Context->GetParentDevice()->Hal().GetMemMgr()
                    ->Allocate(createInfo.alloc,
                               &m_Offset, &m_pHalMemory));
        } else {
            // 路径 B: 裸 KMD 直接分配
            status = HalToMuResult(
                m_Context->GetParentDevice()->Hal().CreateMemory(
                    createInfo, &m_pHalMemory));
        }

        // ──────────────────────────────────────────
        // Step 4: LargePage → General 降级 (static!)
        // ──────────────────────────────────────────
        // ⚠ 关键: pinnedHostAllocHeap 是 static 变量
        //   一旦降级, 进程生命周期内永远使用 general heap
        if (pinnedHostAllocHeap == Hal::MemoryHeap::largePage &&
            status == MUSA_ERROR_MAP_FAILED) {
            tprintf(LOG_MEM, "modify the heap of pinned host "
                    "allocation from LargePage to General "
                    "permanently\n");
            pinnedHostAllocHeap = Hal::MemoryHeap::general;
            createInfo.alloc.heap = pinnedHostAllocHeap;

            // 用降级后的 heap 重新分配
            if (createInfo.alloc.property &
                Hal::memoryPropertySubAllocatable) {
                status = HalToMuResult(
                    m_Context->GetParentDevice()->Hal().GetMemMgr()
                        ->Allocate(createInfo.alloc,
                                   &m_Offset, &m_pHalMemory));
            } else {
                status = HalToMuResult(
                    m_Context->GetParentDevice()->Hal()
                        .CreateMemory(createInfo, &m_pHalMemory));
            }
        }
    }
    return status;
}
```

## 6. LargePage → General 降级机制分析

```
┌─────────────────────────────────────────────────────────────┐
│  static Hal::MemoryHeap pinnedHostAllocHeap = largePage;      │
│  ↑ 进程级静态变量, 仅在首次分配时初始化为 largePage            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  首次分配:                                                    │
│    heap = largePage                                           │
│    CreateMemory → KMD 在 largePage 堆中分配                   │
│    if MAP_FAILED:                                             │
│      pinnedHostAllocHeap = general  ← 永久修改!               │
│      用 general heap 重新分配                                 │
│                                                               │
│  后续所有分配:                                                │
│    heap = general (static 变量已修改)                          │
│    不再尝试 largePage                                         │
└─────────────────────────────────────────────────────────────┘
```

### 降级原因

LargePage 堆资源有限, 当系统无法提供大页时, 直接降级到 General 堆。
由于 `pinnedHostAllocHeap` 是 `static` 变量, 降级是全局的、不可逆的,
避免了每次分配都尝试大页失败的 overhead。

## 7. 与 GeneralAlloc 的属性对比

| 属性 | GeneralAlloc (Device) | PinnedHostAlloc (Host) |
|------|----------------------|----------------------|
| alloc.type | DeviceLocal | **Host** |
| heap | largePage | largePage→general |
| Physical | ✅ | ✅ |
| SharedVA | ✅ | ✅ |
| HostMapped | ❌ | ✅ |
| Virtual | (DeviceMapped时) | (DeviceMapped时) |
| HostVisible | (Virtual时) | (DeviceMapped时) |
| DeviceVisible | (Virtual时) | (DeviceMapped时) |
| DeviceWriteable | (Virtual时) | (DeviceMapped时) |
| DeviceCached | (Virtual时) | (DeviceMapped时) |
| SubAllocatable | ✅ (默认) | (从flags继承) |

## 8. 与 muMemHostRegister 的区别

| 特性 | muMemHostAlloc | muMemHostRegister |
|------|---------------|------------------|
| 内存分配方 | MUSA 运行时分配 | 用户自行 malloc/mmap |
| 内存释放方 | `muMemFreeHost` 或析构 | 用户自行 free/munmap |
| 内存类型 | memoryTypePinnedHost | memoryTypeRegisteredPinnedHost |
| HAL 类型 | memoryTypeAlloc (Alloc) | memoryTypeView (View/Locked) |
| GPU 映射 | 可选 DEVICEMAP | 可选 DEVICEMAP |
| MMIO 支持 | ❌ | ✅ (IOMEMORY flag) |
| 页对齐 | 由 KMD 处理 | HAL 层 AlignUp 到页边界 |

## 9. 相关源码位置

| 文件 | 行数 | 说明 |
|------|------|------|
| `musa/src/driver/mu_memory.cpp` | 59-101 | `imuapiMemHostAlloc` 实现 |
| `musa/src/driver/mu_memory.cpp` | 489-499 | `muapiMemAllocHost_v2` / `muapiMemHostAlloc` |
| `musa/src/musa/core/memory.cpp` | 532-609 | `PinnedHostAlloc` 核心 + 降级逻辑 |
| `musa/src/musa/core/context.cpp` | 915-965 | `CreateMemory` 入口 + MapToPeers |
| `musa/src/hal/m3d/memory.cpp` | 530-605 | `InitGeneralHostMemory` (HAL 层) |