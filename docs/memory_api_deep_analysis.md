# MUSA Memory API 深度分析 — 逐 API 代码流程拆解

> **📖 相关文档**: [memory_analysis.md](memory_analysis.md) (架构总览) | [muMemAlloc_vs_muMemAllocAsync.md](muMemAlloc_vs_muMemAllocAsync.md) (同步异步对比) | [pooling_analysis.md](pooling_analysis.md) (池化) | [stream_command_analysis.md](stream_command_analysis.md) (Stream) | [decision_logic.md](decision_logic.md) (决策分支) | [memory_api/](memory_api/README.md) (26篇分文件)

> 本文对每个核心内存 API 进行逐步代码级分析，从用户调用到底层 GPU 执行，包含时序图、关键代码片段和完整调用栈。

---

## 🎓 预备知识 (新手必读)

如果你想读懂本文，建议先了解以下概念：

### 电脑内存的基本概念

| 概念 | 通俗解释 | 在 GPU 编程中的体现 |
|------|---------|-------------------|
| **Host 内存** | 你电脑主板上的 DDR 内存条（CPU 直接使用） | `muMemHostAlloc` 分配的就是这种 |
| **Device 内存** | 显卡上的显存 (VRAM)，GPU 直接使用 | `muMemAlloc` 分配的就是这种 |
| **Pinned / Page-locked 内存** | 被"钉住"的 Host 内存，操作系统不会把它换出到磁盘 | 这样 GPU 才能通过 DMA 直接访问 |
| **虚拟地址 (Virtual Address)** | 程序看到的地址编号，不一定是真实的物理位置 | GPU 和 CPU 都用虚拟地址，通过 MMU/页表翻译到物理内存 |
| **DMA (Direct Memory Access)** | 硬件引擎不经过 CPU 直接搬数据 | memcpy 走的是 DMA engine，而不是 CPU 逐字节拷贝 |
| **Peer Access** | 两块 GPU 卡之间直接互访内存 | 通过 PCIe 或者 NVLink 实现 |
| **CUDA Graph** | 把一串 GPU 操作预先录制下来，之后反复重放 | 减少 CPU 提交开销 |

### MUSA 项目的分层架构

```
┌─────────────┐
│  用户代码     │ 调用 muMemAlloc, muMemcpy ...
├─────────────┤
│  Driver 层   │ 参数校验、API 路由、TLS 管理
├─────────────┤
│  Core 层     │ 对象生命周期管理 (Memory, Stream, Context)
├─────────────┤
│  HAL 层      │ 硬件抽象 (IMemory, IQueue, ICmdBuffer)
├─────────────┤
│  M3D 层      │ MT-GPU 硬件具体实现
├─────────────┤
│  KMD (内核)  │ DRM ioctl → 物理 GPU 操作
└─────────────┘
```

### 核心对象简介

| 对象 | 作用 | 类比 |
|------|------|------|
| `Context` | GPU 上下文，管理所有资源 (内存、流、事件) | Workspace |
| `Stream` | GPU 命令队列，操作按顺序执行 | 生产线 |
| `Memory` | GPU/CPU 内存的抽象封装 | 内存块 |
| `Command` | GPU 执行的单个操作 (memset, memcpy, kernel) | 任务工单 |
| `MemoryPool` | 内存池，大块切小块提高效率 | 仓库 |
| `MemoryTracker` | 全局指针→Memory 对象的查找表 | 目录索引 |

---

## 约定说明

- 代码路径相对于 `musa/src/`
- 箭头 `→` 表示函数调用
- 带 `[x]` 前缀的为时序图中的角色
- 缩进表示调用层级

---

## 目录

1. [muMemAlloc_v2 — 设备内存分配](#1-mumemalloc_v2--设备内存分配)
2. [muMemFree_v2 — 设备内存释放](#2-mumemfree_v2--设备内存释放)
3. [muMemHostAlloc — 主机内存分配](#3-mumemhostalloc--主机内存分配)
4. [muMemHostRegister_v2 — 用户指针注册](#4-mumemhostregister_v2--用户指针注册)
5. [muMemAllocAsync — 异步内存分配](#5-mumemallocasync--异步内存分配)
6. [muMemcpyHtoD_v2 — 主机到设备拷贝](#6-mumemcpyhtod_v2--主机到设备拷贝)
7. [muMemsetD32_v2 — GPU Memset](#7-mumemsetd32_v2--gpu-memset)
8. [muMemFreeAsync — 异步内存释放](#8-mumemfreeasync--异步内存释放)
9. [总结 — 设计模式归纳](#9-总结--设计模式归纳)

---

## 1. muMemAlloc_v2 — 设备内存分配

### 1.1 功能概述

在 GPU 设备上分配一段线性内存，返回 `MUdeviceptr`（设备虚拟地址）。对应 CUDA 的 `cudaMalloc`。

### 1.2 完整调用栈

```
muMemAlloc_v2(dptr, bytesize)                     // User entry
  → [Wrapper]  mu_wrappers_generated.cpp          // MUPTI 插桩
    → [Driver]  muapiMemAlloc_v2                  // mu_memory.cpp:265
      → [Core]   Context::CreateMemory             // context.cpp:915
        → [Core]   Memory::Init(memoryTypeGeneral) // memory.cpp:378
          → [Core]   Memory::GeneralAlloc          // memory.cpp:462
            → [HAL]    IDevice::GetMemMgr()->Allocate  // sub-alloc pool
            → [HAL]    IDevice::CreateMemory           // 或裸分配
              → [M3D]    M3d::Memory::Init(memoryTypeAlloc)
                → [M3D]    InitGeneralDeviceMemory
                  → [M3D]    m3dDevice->CreateGpuMemory  // KMD 调用
        → [Core]   Context::MapToPeers              // peer 映射
        → [Core]   AddMemory to context
        → [Core]   MemoryTracker::TrackMemory      // 全局注册
      → return muDeviceptr = pMemory->GetDevicePointer()
```

### 1.3 逐步代码分析

#### Step 1: Generated Wrapper (`mu_wrappers_generated.cpp:10177-10215`)

```cpp
MUresult MUSAAPI muMemAlloc_v2(MUdeviceptr* dptr, size_t bytesize) {
    MUresult status = MUSA_SUCCESS;
    ApiTrace trace(status, "muMemAlloc_v2", ...);

    if (toolsCallbackEnabled(...)) {
        // 1. 构造参数结构体并填充
        muMemAlloc_v2_params params{};
        params.dptr = dptr;
        params.bytesize = bytesize;

        // 2. 填充 MUtoolsTraceApiMusa (profiler 上下文)
        inParams.functionName = "muMemAlloc_v2";
        inParams.context = TlsCtxTop();    // 当前线程上下文

        // 3. 触发 ENTER 回调 (profiler 开始追踪)
        toolsIssueCallback(ENTER, &inParams);

        // 4. 派发到真正的实现
        if (!skipDriverImpl) {
            status = muapiMemAlloc_v2(params.dptr, params.bytesize);
        }

        // 5. 触发 EXIT 回调
        inParams.apiEnterOrExit = MU_TOOLS_API_EXIT;
        toolsIssueCallback(EXIT, &inParams);
    } else {
        // 无 profiler: 直接调用
        status = muapiMemAlloc_v2(dptr, bytesize);
    }
    return status;
}
```

**关键设计**: 每个公开 API 都有一个自动生成的 wrapper，统一做两件事：
1. 记录了 `ApiTrace` (调试日志)
2. 如果 profiler 注册了回调，插入 ENTER/EXIT 回调并传递参数快照

#### Step 2: Driver Entry (`mu_memory.cpp:265-297`)

```cpp
MUresult muapiMemAlloc_v2(MUdeviceptr *dptr, size_t bytesize) {
    MUresult status = InitPlatform();   // 确保 Platform 单例已初始化

    if (status == MUSA_SUCCESS) {
        if (nullptr == dptr) {
            status = MUSA_ERROR_INVALID_VALUE;
        } else if (0 == bytesize) {
            *dptr = 0;
            status = MUSA_ERROR_INVALID_VALUE;
        } else {
            Musa::IContext* pContext = TlsCtxTop();  // 获取线程绑定的 context
            if (nullptr == pContext)  {
                status = MUSA_ERROR_INVALID_CONTEXT;
            } else {
                // 构造 MemoryCreateInfo: 指定 type=General
                Musa::MemoryCreateInfo createInfo{};
                createInfo.type = Musa::memoryTypeGeneral;
                createInfo.general.size = bytesize;
                createInfo.general.alignment = 0;
                // 关键标志: subAllocatable 表示从 pool 中切分
                createInfo.general.flags = Hal::memoryPropertyVirtual |
                                           Hal::memoryPropertyDeviceMapped |
                                           Hal::memoryPropertySubAllocatable;

                Musa::IMemory* pMemory;
                status = pContext->CreateMemory(&pMemory, createInfo);
                if (status == MUSA_SUCCESS) {
                    *dptr = pMemory->GetDevicePointer();  // 返回 GPU 虚拟地址
                }
            }
        }
    }
    return status;
}
```

**验证检查**:
- `InitPlatform()` — 首次调用时执行：`Hal::CreatePlatform()` → 枚举设备 → 初始化 HAL
- `TlsCtxTop()` — 从线程局部存储取当前 context（由 `muCtxPushCurrent` 设置）
- `bytesize == 0` → 返回 INVALID_VALUE（CUDA 兼容）

**标志含义**:
- `memoryPropertyVirtual` — 分配的内存有 GPU 虚拟地址 (VA)，可通过 VA 访问
- `memoryPropertyDeviceMapped` — 允许 device 页表映射此内存
- `memoryPropertySubAllocatable` — 允许从 pool 中切分 (细粒度子分配)

#### Step 3: Context::CreateMemory (`context.cpp:915-964`)

```cpp
MUresult Context::CreateMemory(IMemory** ppMemory, const MemoryCreateInfo& createInfo) {
    MUresult status = MUSA_SUCCESS;

    // Step 3.1: 检查是否有 stream 处于 capture 状态
    {
        ReadLockedAccessor<CriticalBase> ctxCrit(m_CriticalData);
        for (auto streamI = ctxCrit->Streams().begin(); streamI != ctxCrit->Streams().end(); streamI++) {
            if (((*streamI)->GetCaptureStatus() == MU_STREAM_CAPTURE_STATUS_ACTIVE &&
                 (*streamI)->NeedStrictlyCheck()) ||
                (*streamI)->GetCaptureStatus() == MU_STREAM_CAPTURE_STATUS_INVALIDATED) {
                status = MUSA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
                (*streamI)->InvalidateCaptureStatus();
                break;
            }
        }
    }

    // Step 3.2: 创建 Memory 对象并初始化
    std::shared_ptr<IMemory> memory_sp;
    Memory* pMemory = nullptr;
    if (status == MUSA_SUCCESS) {
        memory_sp = std::make_shared<Memory>(this);   // shared_ptr 管理生命周期
        pMemory = static_cast<Memory*>(memory_sp.get());
        status = pMemory->Init(createInfo);            // 核心: 分配 GPU 内存
    }

    // Step 3.3: Peer 映射 (如果可 peer accessible)
    if (status == MUSA_SUCCESS) {
        WriteLockedAccessor<CriticalBase> ctxCrit(m_CriticalData);
        if (pMemory->Hal()->GetCapability() & Hal::memoryViewCapabilityPeerAccessible) {
            status = ctxCrit->MapToPeers(pMemory);     // 映射到所有 peer device
        }
        if (status == MUSA_SUCCESS) {
            ctxCrit->AddMemory(pMemory);               // 加入 context 的内存集合
        }
    }

    // Step 3.4: 全局注册
    if (status == MUSA_SUCCESS) {
        Platform::Get().GetMemoryTracker().TrackMemory(memory_sp);  // 全局指针→IMemory 映射
        if (!pMemory->IsPhysical()) {
            Platform::Get().SetMemorySeqID(pMemory);    // 分配唯一序列号
        }
        *ppMemory = pMemory;   // 返回给调用方
    }
    return status;
}
```

**关键设计点**:
- `std::make_shared<Memory>(this)` — Memory 对象用 shared_ptr 管理，因为有 MemoryTracker 等多处引用
- 初始化后立即做 `MapToPeers` — 不同 device 之间如果已启用 peer access，自动建立跨设备映射
- `TrackMemory` 是全局的区间映射，供后续 `muMemFree`、`muMemcpy` 等 API 通过 `MUdeviceptr` 反查 `IMemory*`

#### Step 4: Memory::Init → GeneralAlloc (`memory.cpp:378-497`)

```cpp
MUresult Memory::Init(const MemoryCreateInfo& createInfo) {
    m_Type = createInfo.type;
    switch (m_Type) {
        case memoryTypeGeneral:
            status = GeneralAlloc(createInfo.general.size,
                                  createInfo.general.alignment,
                                  createInfo.general.flags);
            break;
        // ... 其他类型
    }
    return status;
}

MUresult Memory::GeneralAlloc(size_t size, size_t alignment, unsigned int flags) {
    m_Shape = { size, 1, 1, size };
    m_Flags = flags;

    Hal::MemoryCreateInfo createInfo{};
    createInfo.type = Hal::memoryTypeAlloc;                    // 分配类型
    createInfo.alloc.type = Hal::memoryAllocTypeDeviceLocal;    // 设备本地内存
    createInfo.alloc.heap = Hal::MemoryHeap::largePage;         // 大页堆

    // ★ 属性构建: flags 为基础，叠加 Physical + SharedVA
    createInfo.alloc.property = flags;                         // 传入的 Virtual | DeviceMapped | SubAllocatable
    createInfo.alloc.property |= Hal::memoryPropertyPhysical |  // 标记为物理内存 (有 backing pages)
                                 Hal::memoryPropertySharedVirtualAddress; // 多设备共享 VA 空间

    // 如果标记了 Virtual，自动追加 Device/Host 可见性属性
    createInfo.alloc.property |= flags & Hal::memoryPropertyVirtual ?
                                 Hal::memoryPropertyDeviceVisible |
                                 Hal::memoryPropertyHostVisible |
                                 Hal::memoryPropertyHostCoherent |
                                 Hal::memoryPropertyDeviceWriteable |
                                 Hal::memoryPropertyDeviceCached :
                                 Hal::memoryPropertyNone;

    // ★ ViewCapability: 如果 DeviceMapped，支持 Peer Access + IPC Export
    createInfo.alloc.viewCapability = Hal::memoryViewCapabilityExportable;
    createInfo.alloc.viewCapability |= (flags & Hal::memoryPropertyDeviceMapped) ?
                                       (Hal::memoryViewCapabilityPeerAccessible |
                                        Hal::memoryViewCapabilityIpcExportable) :
                                        Hal::memoryViewCapabilityNone;

    createInfo.alloc.size = size;
    createInfo.alloc.alignment = std::max(alignment, device->memAllocAlignment);
    createInfo.alloc.numaId = NUMA_NO_NODE;

    // *** 核心分叉: SubAllocatable → MemMgr, 否则 → CreateMemory ***
    if (createInfo.alloc.property & Hal::memoryPropertySubAllocatable) {
        status = HalToMuResult(m_Context->GetParentDevice()->Hal()
                               .GetMemMgr()->Allocate(createInfo.alloc, &m_Offset, &m_pHalMemory));
    } else {
        status = HalToMuResult(m_Context->GetParentDevice()->Hal()
                               .CreateMemory(createInfo, &m_pHalMemory));
    }
    return status;
}
```

**SubAllocatable 分支** — `MemMgr::Allocate` 从已有的 pool 大块中切出一个小块，返回 `offset`：
```
内存池 (IMemoryPool): [       大块分配 (比如 64MB)       ]
                         ↑ sub-alloc 1 (offset=0, size=4KB)
                         ↑ sub-alloc 2 (offset=4KB, size=1MB)
                         ↑ ...
```

> **🔑 为什么需要 SubAllocation？**
> 想象你要向操作系统要 100 个 4KB 的内存。每次都调 `malloc(4KB)` 就要 100 次系统调用。更好的办法是先调一次 `malloc(4MB)`，然后自己把 4MB 切成 100 个 4KB 发出去。SubAllocation 就是这个思路——向 KMD (内核驱动) 申请大块 GPU 内存，切成小块给每个 API 请求，减少 kernel 调用次数，提高分配速度。

- 好处：减少 KMD 分配次数，提高内存利用率
- 释放时归还给 pool 而不是 KMD

**非 SubAllocatable 分支** — 直接调用 `IDevice::CreateMemory`，向 KMD 申请独立内存。

#### Step 5: HAL M3D InitGeneralDeviceMemory (`hal/m3d/memory.cpp:366-426`)

```cpp
Result Memory::InitGeneralDeviceMemory(const MemoryCreateInfo& createInfo) {
    // 计算对齐 (device align + page size)
    DevSize alignment = std::max(static_cast<DevSize>(m_Device.GetCapabilities()
                                .memoryAllocationAlignSize), createInfo.alloc.alignment);
    alignment = Util::AlignUp(alignment, m_Device.GetHeapInfos()[
                              static_cast<uint32_t>(m_Heap)].largestPageSize);

    // 填充 M3D GpuMemoryCreateInfo
    IM3d::GpuMemoryCreateInfo gpuMemoryCreateInfo = {};
    gpuMemoryCreateInfo.size = createInfo.alloc.size;
    gpuMemoryCreateInfo.alignment = alignment;

    // 物理属性
    gpuMemoryCreateInfo.heapAccess = IM3d::GpuHeapAccessExplicit;
    gpuMemoryCreateInfo.heaps[0] = IM3d::GpuHeapLocal;  // GPU 本地显存
    gpuMemoryCreateInfo.heapCount = 1;
    gpuMemoryCreateInfo.flags.peerWritable = canMapPeerMemory;

    // 虚拟地址属性
    gpuMemoryCreateInfo.vaRange = (heap == largePage) ? IM3d::VaRange::Svm
                                                      : IM3d::VaRange::Default;
    gpuMemoryCreateInfo.flags.svmAlloc = (property & memoryPropertySharedVirtualAddress) != 0;

    // 获取内存对象大小并分配
    gpuMemObjSize = m3dDevice->GetGpuMemorySize(gpuMemoryCreateInfo, &m3dRes);
    m_M3dGpuMemory.Reserve(gpuMemObjSize);
    result = m3dDevice->CreateGpuMemory(gpuMemoryCreateInfo,
                                        m_M3dGpuMemory.GetAlloc(),
                                        &m_M3dGpuMemory());
    return result;
}
```

**M3D 层含义**: `CreateGpuMemory` 通过 `IM3d::IDevice` 接口进入 KMD，最终调用 DRM `ioctl` 分配 GPU 物理内存和虚拟地址空间。

### 1.4 时序图

```
用户代码            Wrapper              Driver               Context              Memory              HAL/M3D               MemMgr           KMD
  |                  |                    |                    |                    |                   |                    |               |
  |--muMemAlloc_v2-->|                    |                    |                    |                   |                    |               |
  |                  |--toolsIssueCallback(ENTER)             |                    |                   |                    |               |
  |                  |--muapiMemAlloc_v2-->|                    |                    |                   |                    |               |
  |                  |                    |--InitPlatform()-->|                    |                   |                    |               |
  |                  |                    |                    |--[if first]--------|------------------->|                    |               |
  |                  |                    |                    |                    |                   |--CreatePlatform()  |               |
  |                  |                    |                    |                    |                   |--GetDeviceCount()  |               |
  |                  |                    |                    |<-------------------|-------------------|                    |               |
  |                  |                    |<--OK---------------|                    |                   |                    |               |
  |                  |                    |                    |                    |                   |                    |               |
  |                  |                    |--CreateMemory----->|                    |                   |                    |               |
  |                  |                    |                    |--new Memory(this)  |                   |                    |               |
  |                  |                    |                    |--Init(General)----->|                   |                    |               |
  |                  |                    |                    |                    |--GeneralAlloc----->|                    |               |
  |                  |                    |                    |                    |                   |                    |               |
  |                  |                    |                    |                    |  [SubAllocatable?] |                    |               |
  |                  |                    |                    |                    |      |             |                    |               |
  |                  |                    |                    |                    |  YES |--MemMgr::   |                    |               |
  |                  |                    |                    |                    |      |  Allocate-->|--Pool::GetPool()   |               |
  |                  |                    |                    |                    |      |             |--Pool::FullAllocate|               |
  |                  |                    |                    |                    |      |             |--返回 offset+pHALMem|               |
  |                  |                    |                    |                    |      |<------------|                    |               |
  |                  |                    |                    |                    |      |             |                    |               |
  |                  |                    |                    |                    |  NO  |--CreateMem--|-> InitGeneral      |               |
  |                  |                    |                    |                    |      |  ory        |   DeviceMemory     |               |
  |                  |                    |                    |                    |      |            |   ->CreateGpuMem--> |---ioctl----->|
  |                  |                    |                    |                    |      |            |<----OK-------------|<--alloc ok---|
  |                  |                    |                    |                    |<-----|            |                    |               |
  |                  |                    |                    |  (return m_pHalMemory, m_Offset) |   |                    |               |
  |                  |                    |                    |                    |                   |                    |               |
  |                  |                    |                    |--MapToPeers()      |                   |                    |               |
  |                  |                    |                    |  for each peer:    |                   |                    |               |
  |                  |                    |                    |  OpenPeerMemory--->|--Hal::OpenPeerMem->|--m3dDevice->Peer  |               |
  |                  |                    |                    |                    |                   |<---OK--------------|               |
  |                  |                    |                    |--AddMemory(ctx)    |                   |                    |               |
  |                  |                    |                    |--TrackMemory(global)|                  |                    |               |
  |                  |                    |<--OK---------------|                    |                   |                    |               |
  |                  |<--OK---------------|                    |                    |                   |                    |               |
  |                  |--toolsIssueCallback(EXIT)              |                    |                   |                    |               |
  |<--return dptr----|                    |                    |                    |                   |                    |               |
```

---

## 2. muMemFree_v2 — 设备内存释放

### 2.1 功能概述

释放 `muMemAlloc` 分配的设备内存。对应 CUDA 的 `cudaFree`。

### 2.2 完整调用栈

```
muMemFree_v2(dptr)
  → [Wrapper]  mu_wrappers_generated.cpp
    → [Driver]  muapiMemFree_v2              // mu_memory.cpp:709
      → [Core]   MemoryTracker::FindRange     // 通过 dptr 找 Memory 对象
      → [Core]   Memory::Synchronize()        // 等待所有 GPU 访问完成
      → [Core]   Context::DestroyMemory       // context.cpp:967
        → [Core]   RemoveMemory from context
        → [Core]   MemoryTracker::UntrackMemory
        → [Core]   ~Memory()                  // 析构: 归还资源
          → [HAL]    MemMgr::Free / pMemory->Destroy
```

### 2.3 逐步代码分析

#### Step 1: Driver Entry (`mu_memory.cpp:709-748`)

```cpp
MUresult muapiMemFree_v2(MUdeviceptr dptr) {
    MUresult status = InitPlatform();

    if (status == MUSA_SUCCESS) {
        if (0 != dptr) {
            size_t offset;
            // Step A: 通过 MemoryTracker 查找指针所在的 Memory 对象
            Musa::IMemory* pMemory = Musa::Platform::Get()
                .GetMemoryByDevicePointer(dptr, &offset)->get();

            if (!pMemory) {
                status = MUSA_ERROR_INVALID_VALUE;     // 未找到
            }
            // Step B: 类型检查 — 只允许释放特定类型
            else if (pMemory->GetType() != Musa::memoryTypeGeneral &&
                     pMemory->GetType() != Musa::memoryTypePitchedGeneral &&
                     pMemory->GetType() != Musa::memoryTypeManaged &&
                     pMemory->GetType() != Musa::memoryTypeExternal &&
                     pMemory->GetType() != Musa::memoryTypeVirtual ||
                     (pMemory->GetType() == Musa::memoryTypeVirtual &&
                      pMemory->GetPool() == nullptr)) {
                status = MUSA_ERROR_INVALID_VALUE;
            }
            // Step C: offset != 0 表示是 sub-allocation 的一部分, 不能单独 free
            else if (offset != 0) {
                status = MUSA_ERROR_INVALID_VALUE;
            }
            // Step D: 同步 + 销毁
            else {
                status = pMemory->Synchronize();        // 等待 GPU 访问完成

                if (status == MUSA_SUCCESS) {
                    if (pMemory->GetType() == Musa::memoryTypeVirtual) {
                        // Virtual Memory: 走 stream 命令
                        Musa::Context* pContext = TlsCtxTop();
                        if (pContext != nullptr) {
                            Musa::IStream* pStream = Musa::Context::InfoStream(pContext, nullptr);
                            status = pStream->CmdMemFree(dptr, true);
                        }
                    } else {
                        // General/Pitched/Managed: 直接销毁
                        status = pMemory->GetContext()->DestroyMemory(pMemory);
                    }
                }
            }
        }
    }
    return status;
}
```

**关键验证点**:
- `GetMemoryByDevicePointer(dptr, &offset)` — 通过 MemoryTracker 的区间树反查
- `offset != 0` — 禁止释放 sub-allocation 的中间偏移（整个 Memory 对象必须从偏移 0 开始释放）
- `Synchronize()` — 释放前必须等待所有 peer device 上的 GPU 操作完成

#### Step 2: Memory::Synchronize (`memory.cpp:115-144`)

```cpp
MUresult Memory::Synchronize() const {
    MUresult status = MUSA_SUCCESS;

    if (m_Type == Musa::memoryTypeVirtual) {
        // Virtual memory: 同步所有绑定的物理内存
        auto syncMemory = [this](MUdeviceptr ptr, std::shared_ptr<IMemory> memory) -> MUresult {
            return memory->GetContext()->LockedWait();
        };
        status = const_cast<MemoryTracker*>(&m_PhysTracker)
                 ->IterateMemories(GetDevicePointer(), GetSize(), false, std::move(syncMemory));
    } else {
        // 先等待自己的 context
        status = m_Context->LockedWait();
        // 再等待所有建立了 peer mapping 的 device 上的 context
        for (int devId = 0; MUSA_SUCCESS == status && devId < Platform::Get().GetDeviceCount(); devId++) {
            auto peerDev = static_cast<Device*>(Platform::Get().GetIDeviceView(devId));
            if (peerDev && m_pHalMemory->GetPeerMemory(&peerDev->Hal())) {
                ReadLockedAccessor<Device::CriticalBase> peerDevCrit(peerDev->CriticalData());
                for (auto peerCtx : peerDevCrit->ctxs()) {
                    status = peerCtx->Synchronize();
                }
            }
        }
    }
    return status;
}
```

`LockedWait()` 最终等待当前 context 下所有 stream 的所有 in-flight command 完成。

#### Step 3: Context::DestroyMemory (`context.cpp:967-975`)

```cpp
MUresult Context::DestroyMemory(IMemory* pMemory) {
    Memory* pMem = static_cast<Memory*>(pMemory);
    if (pMem->GetParentCtx()) {
        WriteLockedAccessor<CriticalBase> ctxCrit(pMem->GetParentCtx()->CriticalData());
        ctxCrit->RemoveMemory(pMem);       // 从 context 的 m_Memories set 中删除
    }
    Platform::Get().GetMemoryTracker().UntrackMemory(pMemory);  // 从全局 tracker 删除
    return MUSA_SUCCESS;
}
```

注意 `DestroyMemory` 不直接 delete Memory 对象。实际的回收发生在 `Memory` 析构时（`shared_ptr` 引用计数归零）。

#### Step 4: Memory 析构 (`memory.cpp:358-376`)

```cpp
Memory::~Memory() {
    if (m_pMapped) {                      // 如果 map 过 host 端
        MUSA_ASSERT(m_pHalMemory);
        m_pHalMemory->Unmap();             // 解除映射
    }
    if (m_Type != memoryTypePrealloc && m_pHalMemory) {
        if (m_pHalMemory->GetProps() & Hal::memoryPropertySubAllocatable) {
            // Sub-allocated: 归还给 pool (MemMgr 或 user pool)
            if (m_pPool == nullptr) {
                m_Context->GetParentDevice()->Hal().GetMemMgr()
                    ->Free(m_pHalMemory, GetDevicePointer(), GetSize());
            } else {
                m_pPool->Hal()->Free(m_pHalMemory, GetDevicePointer(), GetSize());
            }
        } else {
            // 独立分配: 销毁物理内存 + 销毁 HAL memory
            m_PhysTracker.Cleanup();
            m_pHalMemory->Destroy();        // → m3dDevice->DestroyGpuMemory
        }
    }
}
```

**SubAllocatable 路径**: `MemMgr::Free` 将内存块标记为空闲，归还到 pool 的空闲链表。pool 会在下次分配时复用。

**裸分配路径**: `m_pHalMemory->Destroy()` → M3D 层 → `m3dDevice->DestroyGpuMemory()` → KMD 释放 GPU 物理内存。

### 2.4 时序图

```
用户代码          Wrapper            Driver               Context           Memory              MemMgr              KMD
  |                |                  |                    |                |                   |                  |
  |--muMemFree---->|                  |                    |                |                   |                  |
  |                |--muapiMemFree-->|                    |                |                   |                  |
  |                |                  |--Tracker::Find    |                |                   |                  |
  |                |                  |  Range(dptr)----->|                |                   |                  |
  |                |                  |<-Memory* ---------|                |                   |                  |
  |                |                  |                    |                |                   |                  |
  |                |                  |--Synchronize------|--------------->|                   |                  |
  |                |                  |                    |                |--LockedWait()      |                  |
  |                |                  |                    |                |  Wait所有完成       |                  |
  |                |                  |<---OK-------------|----------------|                   |                  |
  |                |                  |                    |                |                   |                  |
  |                |                  |--DestroyMemory--->|                |                   |                  |
  |                |                  |                    |--RemoveMemory  |                   |                  |
  |                |                  |                    |--UntrackMemory  |                   |                  |
  |                |                  |                    |                |                   |                  |
  |                |                  |                    | [~Memory析构]   |                   |                  |
  |                |                  |                    |----------------|--[SubAlloc?]       |                  |
  |                |                  |                    |                |  YES: MemMgr::Free  |                  |
  |                |                  |                    |                |       return->pool  |                  |
  |                |                  |                    |                |  NO:  pMemory->     |                  |
  |                |                  |                    |                |       Destroy()---->|---KMD free---->|
  |                |                  |                    |                |<------OK-----------|<--OK-----------|
  |                |                  |<---OK-------------|----------------|                   |                  |
  |                |<--OK------------|                    |                |                   |                  |
  |<--0------------|                  |                    |                |                   |                  |
```

---

## 3. muMemHostAlloc — 主机内存分配

### 3.1 功能概述

分配页锁定（pinned）的主机内存，GPU 可以通过 DMA 直接访问该内存（无需额外的 staged 拷贝）。对应 CUDA 的 `cudaHostAlloc`。

### 3.2 完整调用栈

```
muMemHostAlloc(pp, bytesize, Flags)
  → [Wrapper]  mu_wrappers_generated.cpp
    → [Driver]  muapiMemHostAlloc          // mu_memory.cpp:497
      → [Driver]  imuapiMemHostAlloc       // mu_memory.cpp:59
        → [Core]   Context::CreateMemory
          → [Core]   Memory::Init(memoryTypePinnedHost)
            → [Core]   Memory::PinnedHostAlloc   // memory.cpp:532
              → [HAL]    IDevice::GetMemMgr()->Allocate  // sub-alloc (LargePage)
              → [HAL]    IDevice::CreateMemory           // 或裸分配
                → [M3D]    InitGeneralHostMemory         // 分配 host 内存
        → [Core]   AddMemory + TrackMemory
        → return *pp = pMemory->GetHostPointer()
```

### 3.3 逐步代码分析

#### Step 1: Driver Entry (`mu_memory.cpp:59-101`)

```cpp
static MUresult imuapiMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags) {
    MUresult status = InitPlatform();
    if (status == MUSA_SUCCESS) {
        if (0 != bytesize) {
            // 校验 Flags: 只允许 PORTABLE | DEVICEMAP | WRITECOMBINED
            int userFlagsMask = MU_MEMHOSTALLOC_PORTABLE |
                                MU_MEMHOSTALLOC_DEVICEMAP |
                                MU_MEMHOSTALLOC_WRITECOMBINED;
            if (Flags & ~userFlagsMask) {
                status = MUSA_ERROR_INVALID_VALUE;
            } else {
                if (nullptr == pp) {
                    status = MUSA_ERROR_INVALID_VALUE;
                } else {
                    Musa::IContext* pContext = TlsCtxTop();
                    if (nullptr == pContext)  {
                        status = MUSA_ERROR_INVALID_CONTEXT;
                    } else {
                        Musa::MemoryCreateInfo createInfo{};
                        createInfo.type = Musa::memoryTypePinnedHost;
                        createInfo.pinnedHost.size = bytesize;
                        // 自动添加 NUMA_AFFINITIVE + SUBALLOCATABLE
                        createInfo.pinnedHost.flags = Flags |
                            (Musa::MU_MEMORY_HOSTALLOC_NUMA_AFFINITIVE |
                             Musa::MU_MEMORY_HOSTALLOC_SUBALLOCATABLE);
                        createInfo.pinnedHost.numaId = NUMA_NO_NODE;

                        Musa::IMemory* pMemory;
                        status = pContext->CreateMemory(&pMemory, createInfo);
                        if (status == MUSA_SUCCESS) {
                            *pp = pMemory->GetHostPointer();  // 返回 CPU 可访问指针
                        }
                    }
                }
            }
        } else if (nullptr != pp) {
            *pp = nullptr;   // 0 字节返回 null
        }
    }
    return status;
}
```

与 `muMemAlloc` 的关键区别:
- `type = memoryTypePinnedHost`, 不是 `General`
- `pinnedHost.numaId` — 支持 NUMA 节点亲和性
- 返回 `GetHostPointer()` 而不是 `GetDevicePointer()` — 因为 pinned host 内存 CPU 和 GPU 都能访问

#### Step 2: Memory::PinnedHostAlloc (`memory.cpp:532-609`)

```cpp
MUresult Memory::PinnedHostAlloc(size_t size, unsigned int flags, int numaId) {
    MUresult status = MUSA_SUCCESS;

    // 检查设备能力
    const Hal::DeviceProperties& deviceProperties = m_Context->GetParentDevice()->Hal().GetProperties();
    if ((flags & MU_MEMORY_HOSTALLOC_PORTABLE) && !deviceProperties.memoryProperties.unifiedAddressing) {
        tprintf(LOG_WARN, "the host memory would not be portable on all MUSA contexts\n");
    }
    if ((flags & MU_MEMORY_HOSTALLOC_DEVICEMAP) && !deviceProperties.memoryProperties.canMapHostMemory) {
        status = MUSA_ERROR_NOT_SUPPORTED;
    }

    // 如果设备支持 Host mapping, 默认加上 DEVICEMAP
    if (deviceProperties.memoryProperties.canMapHostMemory) {
        flags |= MU_MEMORY_HOSTALLOC_DEVICEMAP;
    }

    if (status == MUSA_SUCCESS) {
        m_Shape = { size, 1, 1, size };
        m_Flags = flags;

        Hal::MemoryCreateInfo createInfo{};
        createInfo.type = Hal::memoryTypeAlloc;
        createInfo.alloc.type = Hal::memoryAllocTypeHost;    // ← 主机内存类型
        createInfo.alloc.heap = pinnedHostAllocHeap;          // LargePage 初始

        // 核心属性: 物理 + 共享虚拟地址 + 主机可映射
        createInfo.alloc.property = Hal::memoryPropertyPhysical |
                                    Hal::memoryPropertySharedVirtualAddress |
                                    Hal::memoryPropertyHostMapped;
        if (DEVICEMAP) {  // GPU 也能访问此 host 内存
            createInfo.alloc.property |= Hal::memoryPropertyVirtual |
                                         Hal::memoryPropertyHostVisible |
                                         Hal::memoryPropertyDeviceVisible |
                                         Hal::memoryPropertyDeviceWriteable |
                                         Hal::memoryPropertyDeviceCached;
        }
        createInfo.alloc.size = size;
        createInfo.alloc.alignment = device->memAllocAlignment;
        createInfo.alloc.numaId = numaId;

        // 尝试 LargePage → 失败则降级到 General
        if (subAllocatable) {
            status = HalToMuResult(MemMgr->Allocate(allocInfo, &m_Offset, &m_pHalMemory));
        } else {
            status = HalToMuResult(Device->CreateMemory(createInfo, &m_pHalMemory));
        }

        // ★ LargePage 分配失败降级到 General heap:
        if (pinnedHostAllocHeap == LargePage && status == MUSA_ERROR_MAP_FAILED) {
            pinnedHostAllocHeap = Hal::MemoryHeap::general;   // 永久降级！
            createInfo.alloc.heap = general;
            if (subAllocatable)
                status = MemMgr->Allocate(...);
            else
                status = Device->CreateMemory(...);
        }
    }
    return status;
}
```

**LargePage → General 降级机制**: 变量 `pinnedHostAllocHeap` 是 `static` 的, 一旦因 large page 映射失败降级到 general heap, **后续所有 host 分配都用 general heap**（即降级是永久性的，直到进程退出）。这是一个设计权衡：如果 LargePage 失败一次，大概率以后也会失败，直接降级避免反复重试。

#### Step 3: HAL M3D InitGeneralHostMemory

`memoryAllocTypeHost` 走 `InitGeneralHostMemory` 路径，在 M3D 层分配主机端内存：
- `memoryAllocTypeHost` → `GpuHeapGartUswc`（GART/PCIe 可见内存）
- GPU 可以通过 PCIe DMA 直接访问
- 分配物理内存 + 建立 GPU 页表映射

#### Step 4: GetHostPointer — Lazy CPU Mapping (`memory.cpp:18-37`)

```cpp
void* Memory::GetHostPointer() {
    std::lock_guard<std::mutex> lk(m_MapMtx);
    if (!m_pMapped) {     // 延迟映射: 首次调用才做 mmap
        if (IsVirtual()) {
            pMemory = GetPhysMemory(...)->get()->m_pHalMemory;
        } else {
            pMemory = m_pHalMemory;
        }
        status = pMemory->Map(0, pMemory->GetSize(), &m_pMapped);
        m_pMapped = reinterpret_cast<uint8_t*>(m_pMapped) + m_Offset;
    }
    return m_pMapped;
}
```

**Lazy mapping**: CPU 映射不是分配时做的，而是第一次调用 `GetHostPointer()` 时。`m_Offset` 处理 sub-allocation 偏移。

### 3.4 时序图

```
用户代码          Wrapper             Driver               Context             Memory(PinnedHost)     HAL/M3D
  |                |                   |                    |                   |                     |
  |--muMemHostAlloc|                   |                    |                   |                     |
  | (pp,size,flags)|                   |                    |                   |                     |
  |                |--muapiMemHostAlloc|                    |                   |                     |
  |                |                   |--Validate Flags    |                   |                     |
  |                |                   |--TlsCtxTop()       |                   |                     |
  |                |                   |--CreateMemory------>|                   |                     |
  |                |                   | (PinnedHost)       |--new Memory       |                     |
  |                |                   |                    |--Init(PinnedHost)->|                     |
  |                |                   |                    |                   |--PinnedHostAlloc     |
  |                |                   |                    |                   |  size, numaId, flags |
  |                |                   |                    |                   |                     |
  |                |                   |                    |                   |--createInfo.type=    |
  |                |                   |                    |                   |  Alloc(memoryAlloc   |
  |                |                   |                    |                   |  TypeHost)           |
  |                |                   |                    |                   |                     |
  |                |                   |                    |                   |  [SubAllocatable?]   |
  |                |                   |                    |                   |    |                 |
  |                |                   |                    |                   |  YES: MemMgr::       |
  |                |                   |                    |                   |    Allocate          |
  |                |                   |                    |                   |    (from LargePage)  |
  |                |                   |                    |                   |    |                 |
  |                |                   |                    |                   |  NO:  Device::       |
  |                |                   |                    |                   |    CreateMemory----->|
  |                |                   |                    |                   |                     |--Init(Host)
  |                |                   |                    |                   |                     |  InitGeneralHostMem
  |                |                   |                    |                   |                     |  ->m3dDevice->
  |                |                   |                    |                   |                     |    CreateGpuMemory
  |                |                   |                    |                   |                     |    (GpuHeapGart)
  |                |                   |                    |                   |<-------OK------------|--Map host ptr
  |                |                   |                    |                   |                     |
  |                |                   |                    |                   |  [降级? MAP_FAILED]  |                     |
  |                |                   |                    |                   |  → General heap retry|
  |                |                   |                    |                   |                     |
  |                |                   |                    |--AddMemory        |                     |
  |                |                   |                    |--TrackMemory      |                     |
  |                |                   |                    |--GetHostPointer-->|--lazy Map(mmap)     |
  |                |                   |<---OK + pp---------|-------------------|                     |
  |                |<--OK--------------|                    |                   |                     |
  |<--pp-----------|                   |                    |                   |                     |
```

---

## 4. muMemHostRegister_v2 — 用户指针注册

### 4.1 功能概述

将用户已分配的 host 内存注册为页锁定内存，使 GPU 可以直接 DMA 访问该内存区域（无需额外 copy）。对应 CUDA 的 `cudaHostRegister`。

### 4.2 完整调用栈

```
muMemHostRegister_v2(ptr, bytesize, Flags)
  → [Wrapper]  mu_wrappers_generated.cpp
    → [Driver]  muapiMemHostRegister_v2     // mu_memory.cpp:1693
      → [Core]   Context::CreateMemory
        → [Core]   Memory::Init(memoryTypeRegisteredPinnedHost)
          → [Core]   Memory::PinnedHostRegister    // memory.cpp:611
            → [HAL]    IDevice::CreateMemory
              → [M3D]    Memory::Init(memoryViewTypeLocked/External)
```

### 4.3 逐步代码分析

#### Step 1: Driver Entry (`mu_memory.cpp:1693-1724`)

```cpp
MUresult muapiMemHostRegister_v2(void *p, size_t bytesize, unsigned int Flags) {
    MUresult status = InitPlatform();

    if (status == MUSA_SUCCESS) {
        do {
            if (nullptr == p || bytesize == 0) {
                status = MUSA_ERROR_INVALID_VALUE;
                break;
            }
            Musa::IContext* pContext = TlsCtxTop();
            if (nullptr == pContext)  {
                status = MUSA_ERROR_INVALID_CONTEXT;
                break;
            }
            // 校验 flags 合法性
            if (Flags & ~Musa::MU_MEMORY_REGISTER_API_MASK) {
                status = MUSA_ERROR_INVALID_VALUE;
                break;
            }
            Musa::IMemory* pMemory;
            Musa::MemoryCreateInfo createInfo{};
            createInfo.type = Musa::memoryTypeRegisteredPinnedHost;
            createInfo.registeredPinnedHost.ptr = p;          // 用户指针
            createInfo.registeredPinnedHost.size = bytesize;
            // 自动添加 OVERLAP_CHECK: 检查是否已经注册过
            createInfo.registeredPinnedHost.flags = Flags | Musa::MU_MEMORY_REGISTER_OVERLAP_CHECK;
            status = pContext->CreateMemory(&pMemory, createInfo);
        } while(0);
    }
    return status;
}
```

关键区别：这里传入的是用户已有的 `void* ptr`，不是让驱动分配新内存。

#### Step 2: Memory::PinnedHostRegister (`memory.cpp:611-671`)

```cpp
MUresult Memory::PinnedHostRegister(void* ptr, size_t size, unsigned int flags) {
    MUresult status = MUSA_SUCCESS;

    // 重叠检查: 这个地址范围内是否已有注册的 pinned host 内存?
    auto overlapCheck = [](MUdeviceptr base, std::shared_ptr<IMemory> memory) -> MUresult {
        return memory->GetType() == memoryTypeRegisteredPinnedHost ?
               MUSA_ERROR_HOST_MEMORY_ALREADY_REGISTERED : MUSA_ERROR_INVALID_VALUE;
    };

    // 设备能力检查
    if (!deviceProperties.memoryProperties.hostRegisterSupported) {
        status = MUSA_ERROR_NOT_SUPPORTED;
    }
    if ((flags & READ_ONLY) && !deviceProperties.memoryProperties.supportReadonlyHostRegister) {
        status = MUSA_ERROR_NOT_SUPPORTED;
    }
    if (canMapHostMemory) {
        flags |= MU_MEMORY_REGISTER_DEVICEMAP;  // 设备可访问
    }

    if (MUSA_SUCCESS == status) {
        // 如果需要检查重叠
        if (flags & MU_MEMORY_REGISTER_OVERLAP_CHECK) {
            status = Platform::Get().GetMemoryTracker()
                     .IterateMemories(reinterpret_cast<MUdeviceptr>(ptr), size,
                                      false, std::move(overlapCheck));
        }

        if (status == MUSA_ERROR_HOST_MEMORY_ALREADY_REGISTERED) {
            // 已经注册, 报错
        } else {
            status = MUSA_SUCCESS;
            m_Shape = { size, 1, 1, size };
            m_Flags = flags;

            Hal::MemoryCreateInfo createInfo{};
            // ★★ 核心分叉: IOMEMORY vs Locked ★★
            if (flags & MU_MEMHOSTREGISTER_IOMEMORY) {
                // I/O 内存 (MMIO 区域)
                createInfo.type = Hal::memoryTypeView;
                createInfo.view.type = Hal::memoryViewTypeExternal;
                createInfo.view.external.type = Hal::MemoryExternalHandleType::mmio;
                createInfo.view.external.handle.data.u64Data =
                    reinterpret_cast<MUdeviceptr>(ptr);  // 物理地址
            } else {
                // 普通 host 内存: 使用 locked memory view
                createInfo.type = Hal::memoryTypeView;
                createInfo.view.type = Hal::memoryViewTypeLocked;
                createInfo.view.locked.pHost = ptr;         // 用户指针
                createInfo.view.locked.size = size;
                createInfo.view.locked.isDeviceMapped = (flags & DEVICEMAP);
                createInfo.view.locked.isDeviceReadOnly = (flags & READ_ONLY);
                createInfo.view.locked.isPeerAccessible = (flags & PORTABLE) || unifiedAddressing;
            }
            status = HalToMuResult(m_Context->GetParentDevice()->Hal()
                                    .CreateMemory(createInfo, &m_pHalMemory));
        }
    }
    return status;
}
```

**关键设计: `memoryViewTypeLocked` vs `memoryViewTypeExternal`**

- **Locked**: 用户已分配的普通 host 内存，通过 HAL 锁定页表（使其变为 non-pageable），并建立 GPU 页表映射（`isDeviceMapped`）
- **External (MMIO)**: 用户传递的是 MMIO 物理地址（I/O 内存区域），HAL 创建外部内存 view

#### Step 3: HAL M3D InitLockedMemory

在 M3D 层，`InitLockedMemory` 会：
1. 调用 `m3dDevice->LockGpuMemory(ptr, size)` — 锁定用户内存页表
2. 设置 GPU 页表映射 — 使 GPU 可以通过 DMA 访问该区域
3. 如果 `isDeviceReadOnly`, 设置 GPU 只读权限

### 4.4 时序图

```
用户代码          Wrapper             Driver               Context             Memory(Register)    HAL/M3D
  |                |                   |                    |                   |                   |
  |--muHostReg---->|                   |                    |                   |                   |
  | (ptr,size,flg) |--muapiHostReg---->|                    |                   |                   |
  |                |                   |--Validate: p==0?   |                   |                   |
  |                |                   |           size==0? |                   |                   |
  |                |                   |           Flags?   |                   |                   |
  |                |                   |--TlsCtxTop()       |                   |                   |
  |                |                   |                     |                   |                   |
  |                |                   |--CreateMemory------>|                   |                   |
  |                |                   | (RegisteredPinned)  |--new Memory       |                   |
  |                |                   |                     |--Init->PinnedHost |                   |
  |                |                   |                     |  Register-------->|                   |
  |                |                   |                     |                   |                   |
  |                |                   |                     |                   |--[IOMEMORY?]       |
  |                |                   |                     |                   |   |               |
  |                |                   |                     |                   | YES: External     |
  |                |                   |                     |                   |      (mmio type)  |
  |                |                   |                     |                   |   |               |
  |                |                   |                     |                   | NO: Locked        |
  |                |                   |                     |                   |   (viewTypeLocked)|
  |                |                   |                     |                   |                   |
  |                |                   |                     |                   |--CreateMemory---->|
  |                |                   |                     |                   |  (memoryTypeView) |
  |                |                   |                     |                   |                   |--InitLockedMem
  |                |                   |                     |                   |                   |  LockGpuMemory
  |                |                   |                     |                   |                   |  MapGpuPageTable
  |                |                   |                     |                   |                   |  (mmap to GPU)
  |                |                   |                     |                   |<---OK-------------|
  |                |                   |                     |                   |                   |
  |                |                   |                     |--AddMemory        |                   |
  |                |                   |                     |--TrackMemory      |                   |
  |                |                   |<---OK---------------|-------------------|                   |
  |                |<--OK--------------|                     |                   |                   |
  |<--OK-----------|                   |                     |                   |                   |
```

---

## 5. muMemAllocAsync — 异步内存分配

### 5.1 功能概述

在指定 stream 上"异步"分配设备内存。**注意**：并非真正在 GPU 上异步分配，而是：
1. 立即从 pool 分配 virtual memory（同步）
2. 立即分配 physical memory（同步）  
3. 通过 stream 发送 paging 命令绑定物理到虚拟地址（可配置是否阻塞等待）

对应 CUDA 的 `cudaMallocAsync`。

**核心区别**：与同步 `muMemAlloc` 相比，Async 版本的分配结果**绑定在指定 stream 上**，即在 paging 命令在 stream 中执行完成之前，虚拟地址已返回但物理内存还未绑定完成。

### 5.2 完整调用栈

```
muMemAllocAsync(dptr, bytesize, hStream)
  → [Wrapper]  mu_wrappers_generated.cpp
    → [Driver]  muapiMemAllocAsync               // mu_memory.cpp:303
      → [Core]   Stream::CmdMemAlloc              // stream.cpp:569
        → [Core]   Stream::AsyncMemAlloc          // stream.cpp:518
          → [Step 1] Pool::CreateMemory           // 从 pool 分配虚拟内存 (同步)
            → [Core]   Memory::InitFromPool
              → [HAL]    Pool->FullAllocate(subAlloc)
          → [Step 2] Memory::Init(General)        // 分配物理内存 (同步)
            → [Core]   Memory::GeneralAlloc
              → [HAL]    CreateMemory / MemMgr::Allocate
          → [Step 3] virt->Bind(physical)         // 绑定虚拟→物理
          → [Step 3] Pool->ModifyAccess(virt, physical, blocking, stream)
            → [Core]   Stream::CmdPaging          // 发送 paging 命令到 stream
```

### 5.3 逐步代码分析

#### Step 1: Driver Entry (`mu_memory.cpp:303-340`)

```cpp
MUresult muapiMemAllocAsync(MUdeviceptr *dptr, size_t bytesize, MUstream hStream) {
    MUresult status = InitPlatform();

    if (status == MUSA_SUCCESS) {
        if (nullptr == dptr) {
            status = MUSA_ERROR_INVALID_VALUE;
        } else if (0 == bytesize) {
            *dptr = 0;    // 0 字节返回空指针
        } else {
            Musa::IContext* pContext = TlsCtxTop();
            if (pContext == nullptr) {
                status = MUSA_ERROR_INVALID_CONTEXT;
            }

            Musa::IStream* pStream = nullptr;
            if (status == MUSA_SUCCESS) {
                // 解析 stream (支持 per-thread default stream)
                pStream = Musa::Context::InfoStream(
                    Musa::ICast<Musa::Context>(pContext),
                    Musa::ICast<Musa::Stream>(hStream));
                if (!pStream) {
                    status = MUSA_ERROR_INVALID_HANDLE;
                }
            }

            if (status == MUSA_SUCCESS) {
                Musa::MemoryAllocParameter memAllocParam{};
                memAllocParam.size = bytesize;      // 需要的字节数
                memAllocParam.pool = nullptr;       // 使用默认 pool (由 Device 提供)

                // ★ 关键: 通过 stream 来执行分配流程
                status = pStream->CmdMemAlloc(memAllocParam, false);

                if (status != MUSA_SUCCESS) {
                    *dptr = 0;
                } else {
                    *dptr = memAllocParam.virtAddress;  // 提前返回 VA，物理绑定尚未完成
                }
            }
        }
    }
    return status;
}
```

#### Step 2: Stream::CmdMemAlloc → AsyncMemAlloc (`stream.cpp:569-577`)

```cpp
MUresult Stream::CmdMemAlloc(MemoryAllocParameter& param, bool blocking) {
    MUresult status = MUSA_SUCCESS;
    // 三路分支 (与 memcpy/memset 相同模式)
    if (m_CaptureStatus == MU_STREAM_CAPTURE_STATUS_ACTIVE) {
        // CUDA Graph 录制模式: 将分配记录到 graph 节点
        status = CaptureMemAlloc(param);
    } else if (m_CaptureStatus == MU_STREAM_CAPTURE_STATUS_INVALIDATED) {
        // Graph 失效: 返回错误
        status = MUSA_ERROR_STREAM_CAPTURE_INVALIDATED;
    } else {
        // 正常模式: 立即执行
        status = AsyncMemAlloc(param, blocking);
    }
    return status;
}
```

#### Step 3: AsyncMemAlloc — 三步分配 (`stream.cpp:518-567`)

```cpp
MUresult Stream::AsyncMemAlloc(MemoryAllocParameter& param, bool blocking) {
    MUresult status = MUSA_SUCCESS;

    // 获取物理页面大小 (用于向上对齐)
    const uint64_t physPageSize = m_ParentCtx->GetParentDevice()->GetAllocationGranularity(
                                  MU_MEM_LOCATION_TYPE_DEVICE,
                                  MU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
    uint64_t allocSize = Util::AlignUp(param.size, physPageSize);

    // ===========================================
    // Step 1: 分配虚拟内存 (Virtual Memory  — 同步)
    // ===========================================
    MemoryPool* pPool = param.pool != nullptr
        ? param.pool                                    // 用户指定 pool
        : GetParentCtx()->GetParentDevice()->GetMemoryPool(); // 默认 pool

    Memory* virt = nullptr;
    MUdeviceptr virtAddr = 0;
    if (pPool == nullptr) {
        status = MUSA_ERROR_OUT_OF_MEMORY;
    } else {
        pPool->SetStream(this);                         // pool 绑定到此 stream
        status = pPool->CreateMemory(&virt, &virtAddr, allocSize);
        //  ↑ 从 pool 中 sub-allocate 一段虚拟地址空间
        //  ↓ 只是虚拟地址，还没有物理内存
    }
    param.virtAddress = virtAddr;

    // ===========================================
    // Step 2: 分配物理内存 (Physical Memory — 同步)
    // ===========================================
    std::shared_ptr<IMemory> spPhysical;
    spPhysical = std::make_shared<Memory>(m_ParentCtx);
    Memory* physical = IntrusiveCast<Memory>(spPhysical.get());

    // 标准 General 内存分配 (走 SubAllocatable/裸分配 路径)
    Musa::MemoryCreateInfo createInfo{};
    createInfo.type = Musa::memoryTypeGeneral;
    createInfo.general.size = allocSize;
    createInfo.general.alignment = 0;
    createInfo.general.flags = 0;  // 不设置 SubAllocatable，独立分配
    if (status == MUSA_SUCCESS) {
        status = physical->Init(createInfo);
    }

    // ===========================================
    // Step 3: 绑定虚拟→物理 + 配置 peer 访问权限
    // ===========================================
    if (status == MUSA_SUCCESS) {
        // 3a: 建立虚拟地址到物理内存的映射
        virt->Bind(spPhysical, allocSize, 0, 0);

        // 3b: 通过 stream 发送 paging 命令配置访问权限
        //     (这里才真正有"异步"语义 — 可以不等命令执行完就返回)
        status = pPool->ModifyAccess(virt, physical, allocSize, blocking, this);

        if (status != MUSA_SUCCESS) {
            // 回滚: 如果 paging 失败，解绑并释放
            virt->Unbind(allocSize, virt->GetOffset(), virt->GetParentCtx(), nullptr);
            pPool->DestroyMemory(virt);
        }
    } else {
        // 物理分配失败: 释放虚拟内存
        pPool->DestroyMemory(virt);
    }
    return status;
}
```

**三步分配图解**:

```
                     Virtual Memory                Physical Memory
                      (虚拟地址空间)                (实际 GPU 显存)
                      
Step 1: alloc       ┌─────────────────┐
   virtual memory   │  VA: 0x7F000000 │  → (地址已分配，但无物理内存)
  (from pool)       │  Size: 4MB      │
                    └─────────────────┘

Step 2: alloc       ┌─────────────────┐     ┌──────────────────┐
   physical memory  │  VA: 0x7F000000 │     │  Phys: 0xA000000 │
  (standard/fresh)  │  Size: 4MB      │     │  Size: 4MB       │
                    └─────────────────┘     └──────────────────┘

Step 3: bind +      ┌─────────────────┐
   paging command   │  VA: 0x7F000000 │────→┌──────────────────┐
                    │  Size: 4MB      │     │  Phys: 0xA000000 │
                    └─────────────────┘     │  Size: 4MB       │
                                           │  Access: READWRITE│
  ModifyAccess →                           │  (peer configured) │
  Stream::CmdPaging                        └──────────────────┘
  (可以异步等待)
```

### 5.4 与同步分配的区别

| 特性 | muMemAlloc | muMemAllocAsync |
|------|-----------|----------------|
| **分配时机** | 全部同步完成才返回 | VA 提前返回，物理绑定可异步 |
| **virtual memory** | 隐含（直接分配物理+ VA） | 显式：先从 pool 取 VA，再分配物理 |
| **physical memory** | 一步完成 (SubAllocatable或独立) | 分两步：独立物理分配 + paging 绑定 |
| **stream 关联** | 不关联 stream | 绑定到指定 stream |
| **返回地址** | 物理已就绪 | VA 立即可用，物理在 stream fence 后可用 |
| **适用场景** | 通用分配 | stream-ordered allocation |

### 5.5 时序图

```
用户代码         Driver            Stream              Pool               Memory(phys)        Memory(virt)
  |               |                 |                   |                     |                  |
  |--muAllocAsync>|                 |                   |                     |                  |
  |  (dptr,size,  |                 |                   |                     |                  |
  |   hStream)    |--InitPlatform   |                   |                     |                  |
  |               |--CmdMemAlloc--->|                   |                     |                  |
  |               |                 |--AsyncMemAlloc    |                     |                  |
  |               |                 |  (param, false)   |                     |                  |
  |               |                 |                   |                     |                  |
  |               |                 |  [Step 1: 分配虚拟内存]                     |                  |
  |               |                 |--CreateMemory---->|                     |                  |
  |               |                 |  (virt, addr,size)|--FullAllocate       |                  |
  |               |                 |                   |  (sub-alloc VA)     |                  |
  |               |                 |                   |--InitFromPool----->|                  |
  |               |                 |                   |<---OK--------------|                  |
  |               |                 |<--OK + virtAddr---|                     |                  |
  |               |                 |                   |                     |                  |
  |               |                 |  [Step 2: 分配物理内存]                    |                  |
  |               |                 |                   |                     |                  |
  |               |                 |-------------------------------------->|                  |
  |               |                 |                   |  new Memory(ctx)   |                  |
  |               |                 |                   |  Init(GeneralAlloc)|                  |
  |               |                 |                   |  → MemMgr::Allocate|                  |
  |               |                 |                   |  或 CreateMemory   |                  |
  |               |                 |                   |<---OK (m_pHalMemory)                 |
  |               |                 |                   |                     |                  |
  |               |                 |  [Step 3: 绑定 + Paging]                |                  |
  |               |                 |---------------Bind(spPhysical,size)----|                  |
  |               |                 |  (映射 VA→物理)    |                     |                  |
  |               |                 |                   |                     |                  |
  |               |                 |--ModifyAccess---->|                     |                  |
  |               |                 |  (blocking=false) |--CmdPaging-------->|                  |
  |               |                 |                   |  pagingParams      |                  |
  |               |                 |                   |  → 发到 stream 队列   |                  |
  |               |                 |<--OK--------------|                     |                  |
  |               |                 |                   |                     |                  |
  |<--OK (dptr)----|                 |                   |                     |                  |
  | dptr 立即可用  |                 |                   |                     |                  |
  | [但物理绑定要等  |                 |                   |                     |                  |
  |  stream fence]  |                 |                   |                     |                  |
```

---

## 6. muMemcpyHtoD_v2 — 主机到设备拷贝

### 6.1 功能概述

将数据从主机内存拷贝到设备内存。同步版本，等待 GPU 完成才返回。对应 CUDA 的 `cudaMemcpyHostToDevice`。

> **🔑 关键概念 — GPU 上的"拷贝引擎"**：
> GPU 不是只有计算单元。它还有专门的 DMA 引擎来搬数据。MUSA 支持多种引擎：
> - **DMA** (Direct Memory Access) — 专用拷贝引擎，最快
> - **CE** (Copy Engine) — 另一个拷贝引擎
> - **CDM** (Compute Data Mover) — S5000 的专用拷贝引擎
> - **CPU Fallback** — 如果以上引擎都不支持，CPU 直接写（最慢）
> 
> 引擎选择策略：CDM → CE → TDM → DMA → CPU（从快到慢自动降级）

### 6.2 完整调用栈

```
muMemcpyHtoD_v2(dstDevice, srcHost, ByteCount)
  → [Wrapper]  mu_wrappers_generated.cpp
    → [Driver]  muapiMemcpyHtoD_v2                      // mu_memory.cpp:807
      → [Driver]  GetMemcpy3DFrom1D(copy3D, ...)          // 1D→3D 结构体转换
      → [Core]   Context::GeneralMemcpy(ctx, nullptr, copy3D, true)  // wait=true
        → [Core]   Stream::InfoStream(ctx, hStream)
        → [Core]   Context::CreateMemcpyNode              // 创建 GraphMemcpyNode
          → [Core]   Stream::CmdCopyMemory(pGraphNode, wait=true)
            → [Core]   new AsyncMemcpyCommand / SyncMemcpyCommand
            → [Core]   ResolveDependencyAndQueueCommand
              → [Core]   QueueCommand(command)
                → [Core]   Command::Build()               // 构建 GPU 命令
                  → [HAL]    CmdBufferBegin
                  → [HAL]    ResolveSubmitWait (semaphore)
                  → [HAL]    CmdCopyMemoryAdvanced(copyRegion)
                → [Core]   Command::Submit()              // 提交到 GPU
                  → [HAL]    CmdBufferEnd
                  → [HAL]    IQueue::Submit(submitInfo)
                    → [M3D]    M3dQueue::Submit
                      → [KMD]    DRM IOCTL
                → [Core]   Command::WaitFinish()           // 同步等待完成
```

### 6.3 逐步代码分析

#### Step 1: Driver Entry (`mu_memory.cpp:807-817`)

```cpp
MUresult muapiMemcpyHtoD_v2(MUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    MUresult status = InitPlatform();

    if (status == MUSA_SUCCESS) {
        // 将 1D copy 参数转为统一的 MUSA_MEMCPY3D_PEER 结构
        MUSA_MEMCPY3D_PEER copy3D{};
        GetMemcpy3DFrom1D(copy3D,
                          reinterpret_cast<void*>(dstDevice),   // dst
                          srcHost,                              // src
                          ByteCount,
                          memcpy_host_to_device);               // 方向

        // ★ 所有 memcpy 统一入口
        status = Musa::Context::GeneralMemcpy(TlsCtxTop(), nullptr, copy3D, true);
    }
    return status;
}
```

**`GetMemcpy3DFrom1D`** 将 1D 参数填充到 `MUSA_MEMCPY3D_PEER`:
```cpp
// copy3D 结构体:
{
    srcXInBytes = 0, srcY = 0, srcZ = 0,
    srcMemoryType = memorytypeHost,
    srcHost = srcHost,       // 主机指针
    srcDevice = 0,
    dstXInBytes = 0, dstY = 0, dstZ = 0,
    dstMemoryType = memorytypeDevice,
    dstHost = nullptr,
    dstDevice = dstDevice,   // 设备地址
    WidthInBytes = ByteCount,
    Height = 1, Depth = 1,
    srcPitch = ByteCount, srcHeight = 1,
    dstPitch = ByteCount, dstHeight = 1,
    srcContext = nullptr,
    dstContext = nullptr,
}
```

**统一模型**: 所有 memcpy 变体都归一化为 `MUSA_MEMCPY3D_PEER`：
- 1D copy → Width=ByteCount, Height=1, Depth=1
- 2D copy → Width, Height=rows, Depth=1
- 3D copy → Width, Height, Depth
- Peer copy → 额外设置 srcContext/dstContext

#### Step 2: Context::GeneralMemcpy (`context.cpp:699-731`)

```cpp
MUresult Context::GeneralMemcpy(Musa::Context* ctx,
                                MUstream hStream,
                                MUSA_MEMCPY3D_PEER& memcpyParam,
                                bool wait) {
    MUresult status = MUSA_SUCCESS;
    Stream* pStream = nullptr;

    if (!ctx) {
        status = MUSA_ERROR_INVALID_CONTEXT;
    } else {
        pStream = InfoStream(ctx, hStream);   // hStream=nullptr → 默认 stream
        if (!pStream) {
            status = MUSA_ERROR_INVALID_HANDLE;
        }
    }

    if (status == MUSA_SUCCESS && memcpyParam.WidthInBytes != 0) {
        IGraphNode* pGraphNode;

        // Step A: 创建 GraphMemcpyNode (包含拷贝参数 + copy direction)
        status = pStream->GetParentCtx()->CreateMemcpyNode(memcpyParam, &pGraphNode, wait);

        if (MUSA_SUCCESS == status) {
            // Step B: 三路分支
            if (pStream->GetCaptureStatus() == MU_STREAM_CAPTURE_STATUS_ACTIVE) {
                status = pStream->CaptureNode(static_cast<GraphNode*>(pGraphNode));
                // 捕获模式: 添加到 CUDA Graph 而非执行
            } else if (pStream->GetCaptureStatus() == MU_STREAM_CAPTURE_STATUS_INVALIDATED) {
                status = MUSA_ERROR_STREAM_CAPTURE_INVALIDATED;
            } else {
                // ★ 正常模式: 创建 Command 并提交到 GPU
                status = pStream->CmdCopyMemory(pGraphNode, wait);
            }
        }
    }
    return status;
}
```

**`wait` 参数含义**:
- `wait=true` (同步): `CmdCopyMemory` 中 submit 后调用 `Command::WaitFinish()` 阻塞等待 GPU 完成
- `wait=false` (异步): 只 queue command 到 stream，不等待

#### Step 3: Stream::CmdCopyMemory (`stream.cpp:663-719`)

```cpp
MUresult Stream::CmdCopyMemory(IGraphNode* pGraphNode, bool blocking) {
    GraphMemcpyNode* node = static_cast<GraphMemcpyNode*>(pGraphNode);
    std::shared_ptr<Command> command;

    // 选择 Async 还是 Sync 命令
    if (node->IsAsyncMemcpyCmd(node)) {
        command = std::make_shared<AsyncMemcpyCommand>(
            this, ICast<GraphNode>(node), ...);
    } else {
        command = std::make_shared<SyncMemcpyCommand>(
            this, ICast<GraphNode>(node), ...);
    }

    // Peer copy 需要序列化: 在两个 context 间插入 event
    IContext* pSrcCtx = node->GetCopyParams().pSrcCtx;
    IContext* pDstCtx = node->GetCopyParams().pDstCtx;
    bool isPeerSerial = pSrcCtx != nullptr && pDstCtx != nullptr;

    if (isPeerSerial) {
        // 让两个 context 都依赖这个 command
        static_cast<Context*>(pSrcCtx)->AddCurrentDependencies(command);
        static_cast<Context*>(pDstCtx)->AddCurrentDependencies(command);
    }

    // ★ 核心: 将 command 插入 stream 并决定提交方式
    status = m_ParentCtx->ResolveDependencyAndQueueCommand(std::move(command), this, blocking);

    // Peer serial: 额外插入 RecordCommand + WaitEvent 保证顺序
    if (status == MUSA_SUCCESS && isPeerSerial) {
        CreateEvent + RecordCommand + WaitEvent  // 序列化两个 device 上的 stream
    }
    return status;
}
```

#### Step 4: Command::Build — AsyncMemcpyCommand (`AsyncMemcpyCommand.cpp:27-100`)

```cpp
MUresult AsyncMemcpyCommand::Build(const std::list<std::shared_ptr<Command>>& mergingList) {
    // 获取 HAL cmd buffer
    pHalCmdBuffer = GetHalCmdBuffer(true);
    pHalCmdBuffer->Begin(beginInfo);

    // 解析依赖 (wait semaphores)
    Command::Build(mergingList);
    ResolveSubmitWait(pHalCmdBuffer, WaitBetweenCmd, device);

    // 开始 timestamp (profiling)
    if (GetTimestampMem()) {
        pHalCmdBuffer->CmdWriteTimestamp(gpuVa);  // TopOfPipe
    }

    // ★ 根据拷贝方向调用不同的 Build 函数
    switch (GetCopyParams().copyDirection) {
        case DeviceToDevice:
            BuildCopyMemory(*pHalCmdBuffer);     // ← HtoD 走这个分支
            break;
        case DeviceToArray:
            BuildCopyMemoryToImage(*pHalCmdBuffer);
            break;
        case ArrayToDevice:
            BuildCopyImageToMemory(*pHalCmdBuffer);
            break;
        case ArrayToArray:
            BuildCopyImage(*pHalCmdBuffer);
            break;
    }
    // 结束 timestamp
    if (GetTimestampMem())
        pHalCmdBuffer->CmdWriteTimestamp(gpuVa); // BottomOfPipe
    return status;
}
```

#### Step 5: BuildCopyMemory (`AsyncMemcpyCommand.cpp:162-194`)

```cpp
void AsyncMemcpyCommand::BuildCopyMemory(Hal::ICmdBuffer& cmdBuffer) {
    const auto& copyParam = GetCopyParams();

    // 获取 source/destination 的 HAL memory (处理 peer 映射)
    auto pSrcHalMem = srcMemory.GetHalMemory(pCurDevice);
    auto pDstHalMem = dstMemory.GetHalMemory(pCurDevice);

    // 计算 3D region 的 base 偏移 (含 sub-allocation offset)
    size_t srcBiasBase = copyParam.srcZ * copyParam.srcPitch * copyParam.srcHeight
                       + copyParam.srcY * copyParam.srcPitch
                       + copyParam.srcXInBytes
                       + srcMemory.GetOffset();
    size_t dstBiasBase = ... + dstMemory.GetOffset();

    // 填充 HAL CopyMemoryAdvancedParameter
    Hal::CopyMemoryAdvancedParameter copyMemoryAdvanced{};
    copyMemoryAdvanced.pSrcMemory = pSrcHalMem;
    copyMemoryAdvanced.pDstMemory = pDstHalMem;
    copyMemoryAdvanced.copyRegion = Hal::MemoryCopy3DRegion{
        srcBiasBase, copyParam.srcPitch, copyParam.srcPitch * copyParam.srcHeight,
        dstBiasBase, copyParam.dstPitch, copyParam.dstPitch * copyParam.dstHeight,
        Hal::Extent3D{copyParam.WidthInBytes, copyParam.Height, copyParam.Depth}
    };

    // ★ 写入 HAL cmd buffer (最终通过 DMA engine 执行)
    cmdBuffer.CmdCopyMemoryAdvanced(copyMemoryAdvanced);
}
```

#### Step 6: Command::Submit (`AsyncMemcpyCommand.cpp:102-160`)

```cpp
MUresult AsyncMemcpyCommand::Submit() {
    pHalCmdBuffer = GetHalCmdBuffer(false);

    // 编码 signal semaphores (提交后的完成信号)
    ResolveSubmitSignal(pHalCmdBuffer, SignalBetweenCmd, device);

    pHalCmdBuffer->End();   // 结束命令缓冲区的记录

    // 组装 QueueSubmitInfo
    Hal::QueueSubmitInfo submitInfo{};
    submitInfo.ppCmdBuffers = &pHalCmdBuffer;
    submitInfo.cmdBufferCount = 1;
    // wait/signal semaphores 已在 Build 阶段设置好

    // ★ 提交到 HAL Queue → M3D Queue → KMD
    status = SubmitToQueue(GetParentStream()->GetHalQueue(GetEngine()), submitInfo);
    return status;
}
```

### 6.4 时序图

```
用户代码              Wrapper             Driver              Context            Stream           Command            HAL Queue          KMD/GPU
  |                    |                   |                   |                 |                |                  |                 |
  |--muMemcpyHtoD----->|                   |                   |                 |                |                  |                 |
  | (dst,src,size)     |--toolsCallback    |                   |                 |                |                  |                 |
  |                    |--muapiMemcpyHtoD->|                   |                 |                |                  |                 |
  |                    |                   |--GetMemcpy3DFrom1D|                 |                |                  |                 |
  |                    |                   |--GeneralMemcpy--->|                 |                |                  |                 |
  |                    |                   |  (ctx,null,3D,   |                 |                |                  |                 |
  |                    |                   |   wait=true)      |                 |                |                  |                 |
  |                    |                   |                   |--CreateMemcpyNode|               |                  |                 |
  |                    |                   |                   |  (GraphMemcpyNode)|               |                  |                 |
  |                    |                   |                   |--CmdCopyMemory-->|                |                  |                 |
  |                    |                   |                   |                 |--new AsyncMemcpy|                 |                 |
  |                    |                   |                   |                 |  Command        |                 |                 |
  |                    |                   |                   |                 |                  |                 |                 |
  |                    |                   |                   |                 |--ResolveDeps+   |                 |                 |
  |                    |                   |                   |                 |  QueueCommand----|                 |                 |
  |                    |                   |                   |                 |                  |--QueueCommand   |                 |
  |                    |                   |                   |                 |                  |  → m_CommandList|                 |
  |                    |                   |                   |                 |                  |                 |                 |
  |  [Submit Thread]   |                   |                   |                 |                  |                 |                 |
  |                    |                   |                   |                 |                  |--Build()        |                 |
  |                    |                   |                   |                 |                  |  GetHalCmdBuffer|                 |
  |                    |                   |                   |                 |                  |  Begin----------|-->Begin cmd buf |
  |                    |                   |                   |                 |                  |  ResolveWait    |--CmdBarrier     |
  |                    |                   |                   |                 |                  |  CmdCopyMemAdv--|-->CmdCopyMem    |
  |                    |                   |                   |                 |                  |                 |  (DMA engine)   |
  |                    |                   |                   |                 |                  |                 |                 |
  |                    |                   |                   |                 |                  |--Submit()       |                 |
  |                    |                   |                   |                 |                  |  ResolveSignal  |--CmdSignal      |
  |                    |                   |                   |                 |                  |  End------------|-->End           |
  |                    |                   |                   |                 |                  |  SubmitToQueue--|-->IQueue::Submit|
  |                    |                   |                   |                 |                  |                 |--MultiSubmitInfo|
  |                    |                   |                   |                 |                  |                 |---ioctl----------->|
  |                    |                   |                   |                 |                  |                 |                    |--GPU DMA copy
  |                    |                   |                   |                 |                  |                 |<---done------------|
  |                    |                   |                   |                 |                  |                 |                   |
  |                    |                   |                   |                 |--WaitFinish()     |                 |                   |
  |                    |                   |                   |                 |  (spin waiting)    |                 |                   |
  |                    |                   |                   |                 |  for semaphore     |                 |                   |
  |                    |                   |                   |                 |  signal            |                 |                   |
  |                    |                   |<---OK------------|-----------------|-------------------|-----------------|                   |
  |                    |<--OK--------------|                   |                 |                  |                 |                   |
  |<--OK---------------|                   |                   |                 |                  |                 |                   |
```

### 6.5 同步 vs 异步 memcpy 的异同

| 特性 | SyncMemcpyCommand | AsyncMemcpyCommand |
|------|------------------|-------------------|
| Build 阶段 | 相同 (走 MemcpyCommand 基类) | 相同 |
| Submit 阶段 | 执行 `WaitFinish()` 阻塞等待 | 不等待, 立即返回 |
| Wait semaphore | 无 (同步不需要) | 需要编码等待前序命令 |
| 合并策略 | 不可合并 | 可作为 secondary 合并到其他命令 |

---

## 7. muMemsetD32_v2 — GPU Memset

### 7.1 功能概述

在 GPU 设备内存上设置 32-bit 值。同步版本, 等待 GPU 完成后返回。对应 CUDA 的 `cudaMemsetD32`。

### 7.2 完整调用栈

```
muMemsetD32_v2(dstDevice, ui, N)
  → [Wrapper]  mu_wrappers_generated.cpp
    → [Driver]  muapiMemsetD32_v2                      // mu_memory.cpp:1589
      → [Driver]  MUSA_MEMSET_NODE_PARAMS {dst, pitch, value, elemSize, width, height}
      → [Core]   Context::GeneralMemset(ctx, nullptr, params, true)
        → [Core]   Stream::InfoStream(ctx, hStream)
        → [Core]   Context::CreateMemsetNode
        → [Core]   Stream::CmdMemset(pGraphNode, true)
          → [Core]   new MemsetCommand
          → [Core]   ResolveDependencyAndQueueCommand
            → [Core]   Command::Build()
              → [HAL]    CmdBufferBegin
              → [HAL]    CmdFillMemory(dst, data, size)
            → [Core]   Command::Submit()
              → [HAL]    CmdBufferEnd
              → [HAL]    IQueue::Submit
            → [Core]   Command::WaitFinish()
```

### 7.3 逐步代码分析

#### Step 1: Driver Entry (`mu_memory.cpp:1589-1598`)

```cpp
MUresult muapiMemsetD32_v2(MUdeviceptr dstDevice, unsigned int ui, size_t N) {
    MUresult status = InitPlatform();

    if (status == MUSA_SUCCESS) {
        // 构造统一的 MUSA_MEMSET_NODE_PARAMS:
        // {dst, pitch, value, elemSize, width, height}
        MUSA_MEMSET_NODE_PARAMS memsetParams = {
            dstDevice,                          // dstDevice
            N * sizeof(unsigned int),           // dstPitch (= 总字节数, 1D 无 pitch)
            ui,                                  // value
            sizeof(unsigned int),               // elementSize
            N,                                   // width (元素个数)
            1                                    // height
        };
        status = Musa::Context::GeneralMemset(TlsCtxTop(), nullptr, memsetParams, true);
    }
    return status;
}
```

所有 memset API（D8/D16/D32/D2D8/D2D16/D2D32）都统一转为 `MUSA_MEMSET_NODE_PARAMS`:
```
muMemsetD8:  elemSize=1, value=uc
muMemsetD16: elemSize=2, value=us
muMemsetD32: elemSize=4, value=ui
muMemsetD2D: 多了 pitch 和 height 参数
```

#### Step 2: Context::GeneralMemset (`context.cpp:733-766`)

```cpp
MUresult Context::GeneralMemset(Context* ctx, MUstream hStream,
                                 MUSA_MEMSET_NODE_PARAMS& memsetParam, bool wait) {
    MUresult status = MUSA_SUCCESS;
    do {
        if (MUSA_SUCCESS != Platform::Get().ValidateContext(ctx)) {
            status = MUSA_ERROR_CONTEXT_IS_DESTROYED;
            break;
        }

        Musa::Stream* stream = InfoStream(ctx, hStream);
        if (!stream) {
            status = MUSA_ERROR_INVALID_HANDLE;
            break;
        }

        if (0 != memsetParam.width * memsetParam.height * memsetParam.elementSize) {
            Musa::IGraphNode* pGraphNode;

            // 创建 MemsetNode (参数封装)
            status = ctx->CreateMemsetNode(memsetParam, &pGraphNode, wait);
            if (MUSA_SUCCESS != status) break;

            // 三路分支: Capture / Invalidated / 正常
            if (stream->GetCaptureStatus() == MU_STREAM_CAPTURE_STATUS_ACTIVE) {
                status = stream->CaptureNode(...);
            } else if (stream->GetCaptureStatus() == MU_STREAM_CAPTURE_STATUS_INVALIDATED) {
                status = MUSA_ERROR_STREAM_CAPTURE_INVALIDATED;
            } else {
                status = stream->CmdMemset(pGraphNode, wait);
            }
        }
    } while(0);
    return status;
}
```

与 `GeneralMemcpy` 相同的三路分支模式。

#### Step 3: Stream::CmdMemset (`stream.cpp:721-730`)

```cpp
MUresult Stream::CmdMemset(IGraphNode* pGraphNode, bool blocking) {
    std::shared_ptr<Command> command = std::make_shared<MemsetCommand>(
        this,
        ICast<GraphNode>(pGraphNode),
        m_ParentCtx->PfmCheckEnable(),
        m_ParentCtx->PfmGetConfig()
    );
    return m_ParentCtx->ResolveDependencyAndQueueCommand(std::move(command), this, blocking);
}
```

比 memcpy 更简单: 没有 Async/Sync 之分, 直接创建 `MemsetCommand`。

#### Step 4: MemsetCommand::Build → BuildFillMemory (`memsetCommand.h:23-54`)

```cpp
MUresult MemsetCommand::BuildFillMemory(Hal::ICmdBuffer& cmdBuffer) {
    const auto& memsetParams = m_MemsetParams;

    // 根据大小和设备能力选择执行路径:
    // 小操作: CPU 执行 (直接写内存)
    // 大操作: GPU DMA Fill (通过 DMA engine)

    if (memsetParams.size <= CPU_EXEC_THRESHOLD && deviceSupportsCpuFill) {
        return CpuExecute();
    } else {
        return DmaExecute();
    }

    // DmaExecute 最终:
    Hal::FillMemoryParameter fillMemory{};
    fillMemory.pDstMemory = dstMem->Hal();         // 目标 HAL 内存
    fillMemory.dstOffset  = dstMem->GetOffset() + offset;
    fillMemory.fillSize   = memsetParams.width * memsetParams.elementSize;
    fillMemory.data       = memsetParams.value;
    cmdBuffer.CmdFillMemory(fillMemory);            // ← GPU 执行
}
```

**CPU 还是 GPU 执行的决策**:
- 小量数据（如 memset few bytes）：CPU 直接写 GPU 可映射内存 → 无需 GPU 调度开销
- 大量数据（如 memset 1GB）：走 `CmdFillMemory` → DMA engine → GPU 执行

#### Step 5: Command::Submit

```cpp
MUresult MemsetCommand::Submit() {
    pHalCmdBuffer = GetHalCmdBuffer(false);
    ResolveSubmitSignal(pHalCmdBuffer, SignalBetweenCmd, device);
    pHalCmdBuffer->End();

    Hal::QueueSubmitInfo submitInfo{};
    submitInfo.ppCmdBuffers = &pHalCmdBuffer;
    submitInfo.cmdBufferCount = 1;
    status = SubmitToQueue(GetParentStream()->GetHalQueue(GetEngine()), submitInfo);
    // 同步: WaitFinish()
    return status;
}
```

### 7.4 时序图

```
用户代码              Wrapper             Driver            Context           Stream           MemsetCommand       HAL Queue      KMD/GPU
  |                    |                   |                 |                |                |                  |              |
  |--muMemsetD32------>|                   |                 |                |                |                  |              |
  | (dst, val, N)      |--toolsCallback    |                 |                |                |                  |              |
  |                    |--muapiMemsetD32-->|                 |                |                |                  |              |
  |                    |                   |--GeneralMemset->|                |                |                  |              |
  |                    |                   | (params, true)  |--CreateMemset  |                |                  |              |
  |                    |                   |                 |  Node          |                |                  |              |
  |                    |                   |                 |--CmdMemset---->|                |                  |              |
  |                    |                   |                 |                |--new MemsetCmd |                  |              |
  |                    |                   |                 |                |--ResolveDeps+  |                  |              |
  |                    |                   |                 |                |  QueueCommand-->|--Build()         |              |
  |                    |                   |                 |                |                |  GetHalCmdBuffer  |              |
  |                    |                   |                 |                |                |  ResolveWait+Build|             |
  |                    |                   |                 |                |                |  [size small?]    |              |
  |                    |                   |                 |                |                |   YES: CpuExecute()|             |
  |                    |                   |                 |                |                |        直接写 host 映射         |
  |                    |                   |                 |                |                |   NO: CmdFillMem-->|--CmdFillMem  |
  |                    |                   |                 |                |                |                  |  (DMA engine)  |
  |                    |                   |                 |                |                |--Submit()        |              |
  |                    |                   |                 |                |                |  End             |              |
  |                    |                   |                 |                |                |  SubmitToQueue-->|--IQueue::Submit|
  |                    |                   |                 |                |                |                  |---ioctl------->|
  |                    |                   |                 |                |                |                  |               |--GPU fills
  |                    |                   |                 |                |--WaitFinish()   |                  |               |
  |                    |                   |                 |                |  (spin/sleep)    |                  |               |
  |                    |                   |                 |                |                  |                  |<---done-------|
  |                    |                   |<---OK-----------|----------------|------------------|------------------|               |
  |<--OK---------------|                   |                 |                |                  |                  |               |
```

---

## 8. muMemFreeAsync — 异步内存释放

### 8.1 功能概述

在指定 stream 上按 stream 顺序异步释放内存。当 stream 中所有先于 free 操作完成时, 才真正释放。对应 CUDA 的 `cudaFreeAsync`。

### 8.2 完整调用栈

```
muMemFreeAsync(dptr, hStream)
  → [Wrapper]  mu_wrappers_generated.cpp
    → [Driver]  muapiMemFreeAsync           // mu_memory.cpp:386
      → [Core]   MemoryTracker::FindRange
      → [Core]   类型检查 (General/Pitched/Virtual/IPC/External)
      → [Core]   Synchronize (Virtual 类型在命令中释放, 不需要)
      → [Core]   pStream->CmdMemFree(dptr, false)    // stream 异步
        → [Core]   AsyncMemFree(virtAddress, false)   // stream.cpp:601
          → [Core]   GetPhysMemory (virtual memory)
          → [Core]   pPool->DisableAccess            // 禁用 peer 访问
          → [Core]   new CallbackCommand(callback)    // 释放回调
              └ callback: DestroyPhysMemories + pPool->DestroyMemory
          → [Core]   ResolveDependencyAndQueueCommand
```

### 8.3 逐步代码分析

#### Step 1: Driver Entry (`mu_memory.cpp:386-443`)

```cpp
MUresult muapiMemFreeAsync(MUdeviceptr dptr, MUstream hStream) {
    MUresult status = InitPlatform();

    if (status == MUSA_SUCCESS) {
        do {
            if (0 == dptr) break;  // 空指针: no-op

            Musa::IContext* pContext = TlsCtxTop();
            if (pContext == nullptr) { status = MUSA_ERROR_INVALID_CONTEXT; break; }

            Musa::IStream* pStream = InfoStream(ctx, hStream);
            if (!pStream) { status = MUSA_ERROR_INVALID_HANDLE; break; }

            // 通过 MemoryTracker 找到 Memory 对象
            auto virtMem = Platform::Get().GetMemoryByDevicePointer(dptr, nullptr);
            if (virtMem->get() == nullptr) { status = MUSA_ERROR_INVALID_VALUE; break; }

            auto memory = IntrusiveCast<Memory>(virtMem->get());
            switch (memory->GetType()) {
                case memoryTypeIpcImport:
                case memoryTypeExternal:
                    // IPC/External: 立即销毁
                    status = pContext->DestroyMemory(memory);
                    break;
                case memoryTypeGeneral:
                case memoryTypePitchedGeneral:
                case memoryTypeManaged:
                    // 常规内存: Synchronize + DestroyMemory
                    status = memory->Synchronize();
                    if (status == MUSA_SUCCESS)
                        status = memory->GetContext()->DestroyMemory(memory);
                    break;
                case memoryTypeVirtual:
                    // Virtual Memory: 通过 stream 异步释放
                    if (memory->GetPool() != nullptr)
                        status = pStream->CmdMemFree(dptr, false);
                    else
                        status = MUSA_ERROR_INVALID_VALUE;
                    break;
                default:
                    status = MUSA_ERROR_INVALID_VALUE;
            }
        } while(0);
    }
    return status;
}
```

**释放策略因类型而异**:
- `General/Pitched/Managed`: 同步等待完成后销毁
- `IPC/External`: 直接销毁（同步不影响其他进程）
- `Virtual (from pool)`: 通过 stream 异步释放（`CallbackCommand`）

#### Step 2: Stream::AsyncMemFree (`stream.cpp:601-626`)

```cpp
MUresult Stream::AsyncMemFree(uint64_t virtAddress, bool blocking) {
    MUresult status = MUSA_SUCCESS;

    // 获取 virtual memory 和 physical memory
    Memory* virt = IntrusiveCast<Memory>(
        Platform::Get().GetMemoryByDevicePointer(virtAddress, nullptr)->get());
    Memory* physical = IntrusiveCast<Memory>(virt->GetPhysMemory(virtAddress)->get());
    MemoryPool* pPool = virt->GetPool();

    // 禁用 peer access
    status = pPool->DisableAccess(virt, physical, blocking, this);

    // 创建 CallbackCommand, 在 GPU 完成依赖后才执行回调
    if (status == MUSA_SUCCESS) {
        std::shared_ptr<Command> command = std::make_shared<CallbackCommand>(this,
            std::function<void()>());
        std::function<void()> callback = [virt, command, pPool] () {
            virt->DestroyPhysMemories();        // 解除物理绑定
            if (!virt->IsGraphAlloc()) {
                pPool->DestroyMemory(virt);     // 归还给 pool
            }
        };
        static_cast<CallbackCommand*>(command.get())->SetCallback(std::move(callback));
        status = m_ParentCtx->ResolveDependencyAndQueueCommand(std::move(command), this, blocking);
    }
    return status;
}
```

**CallbackCommand**: 这是 stream 中的"后门"命令 — 当 stream 中所有前序命令执行完后, 在用户态执行回调函数, 真正完成内存释放。这确保了内存不会在仍有 GPU 操作访问它时被释放。

---

## 9. 总结 — 设计模式归纳

### 9.1 统一 API 模式

每个公开 API 都遵循这 5 步模式:

```
1. InitPlatform()              — 确保平台初始化
2. TlsCtxTop()                  — 获取线程绑定的 context
3. 参数校验 (nullptr, size==0, flags) — CUDA 兼容性检查
4. 调用 Core 层 (CreateMemory / GeneralMemcpy / CmdMemset ...)
5. 返回结果
```

### 9.2 三路分支模式 (Stream command)

所有 stream 操作都走 `CmdXXX → Capture | Invalidated | Async` 三路:

```
Stream::CmdMemcpy / CmdMemset / CmdMemAlloc / CmdMemFree
  ├── CaptureStatus == ACTIVE    → CaptureNode (记录到 CUDA Graph)
  ├── CaptureStatus == INVALIDATED → 返回错误
  └── CaptureStatus == NONE      → AsyncXXX (正常执行)
```

### 9.3 Command 生命周期

```
创建 (new AsyncMemcpyCommand)
  → Queued (ResolveDependencyAndQueueCommand)
    → Built (Build: CmdBufferBegin + GPU 指令编码 + dependency encoding)
      → Submitted (Submit: CmdBufferEnd + IQueue::Submit)
        → Completed (semaphore signal → WaitFinish 返回)
```

### 9.4 内存分配层次

```
Hal::IMemory (裸 KMD 分配)
    ↑ sub-allocation ↓
Hal::IMemMgr (pool 管理)
    ↑ user pool ↓
Musa::MemoryPool (用户池管理)
```

### 9.5 MemoryTracker 全局映射

所有 `MUdeviceptr` → `IMemory*` 的查找都通过 `MemoryTracker`:

```
TrackMemory:   分配时注册 (MUdeviceptr → shared_ptr<IMemory>)
FindRange:     释放/拷贝时通过 ptr 反查 Memory 对象
UntrackMemory: 释放时删除注册
```

### 9.6 1D→3D 统一模型

所有 memcpy 都转为 `MUSA_MEMCPY3D_PEER`, 所有 memset 都转为 `MUSA_MEMSET_NODE_PARAMS`:

```
memcpy 1D → WidthInBytes=size, Height=1, Depth=1
memcpy 2D → WidthInBytes, Height, Depth=1
memcpy 3D → WidthInBytes, Height, Depth

memset 1D → width=N, height=1, pitch=totalBytes
memset 2D → width, height, pitch
```
