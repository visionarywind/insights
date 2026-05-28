# 内存查询与信息获取 API 深度源码分析

> **文档编号**: 15  
> **相关文件**: `musa/doc/memory_api/README.md`  
> **源码路径**:  
> - `musa/src/driver/mu_memory.cpp` (Driver 层入口)  
> - `musa/src/musa/core/memory.cpp` (Core 层 IMemory 接口)  
> - `musa/src/musa/core/context.cpp` (Context 层实现)  
> - `musa/src/hal/m3d/memory.h` (HAL 层接口)  

---

## 1. 功能概述

本篇覆盖四个内存查询/信息获取 API：

| API | 功能 | 层级 |
|-----|------|------|
| `muMemGetInfo` / `muMemGetInfo_v2` | 获取设备内存空闲/总量 | Driver |
| `muMemGetAddressRange` / `muMemGetAddressRange_v2` | 根据设备指针查询所属内存块基址和大小 | Driver + Core |
| `muMemHostGetDevicePointer` / `muMemHostGetDevicePointer_v2` | 根据主机指针获取对应的设备指针 | Driver + Core |
| `muMemHostGetFlags` | 查询主机内存的注册 flags | Driver + Core |

> **注意**: `muMemPtrGetInfo` 并非 MUSA 公开 API。MUSA 提供的是 `muPointerGetAttributes`（见文档 16），其功能更丰富且为通用接口。

---

## 2. muMemGetInfo / muMemGetInfo_v2

### 2.1 函数签名

```c
// v2 版本（推荐）
MUresult muapiMemGetInfo_v2(size_t *free, size_t *total);

// v1 版本（已弃用，uint 宽度）
MUresult muapiMemGetInfo(unsigned int *free, unsigned int *total);
```

### 2.2 完整调用链

```
User Code
  │
  ├─ muMemGetInfo_v2(&free, &total)
  │     │
  │     ├─ InitPlatform()                          // 确保平台已初始化
  │     ├─ TlsCtxTop()                             // 获取当前线程 TLS Context
  │     ├─ ctx->GetParentDevice()                  // 获取关联 Device
  │     ├─ device->GetProperties().totalGlobalMem   // 总内存（来自 HAL 属性）
  │     └─ device->QueryGlobalMemFreeSize(&free)    // 空闲内存（实时查询 KMD）
  │
  └─ muMemGetInfo(pFree, pTotal)
        │
        └─ muapiMemGetInfo_v2()                    // 直接委托给 v2 版本
```

### 2.3 源码逐行分析

**`mu_memory.cpp:535-558`**:

```cpp
MUresult muapiMemGetInfo_v2(size_t *free, size_t *total) {
    MUresult status = InitPlatform();                // Step 1: 平台初始化

    if (status == MUSA_SUCCESS) {
        Musa::Context* ctx = TlsCtxTop();            // Step 2: 获取 TLS Context
        if (nullptr == ctx) {
            status = MUSA_ERROR_INVALID_CONTEXT;     // 无上下文 → 报错
        } else {
            Musa::Device* device = ctx->GetParentDevice(); // Step 3: 获取 Device
            if (nullptr != total) {
                *total = device->GetProperties().totalGlobalMem;  // Step 4a
            }
            if (nullptr != free) {
                status = device->QueryGlobalMemFreeSize(free);   // Step 4b
            }
        }
    }
    return status;
}
```

**逐行分析**：

| 行 | 代码 | 分析 |
|----|------|------|
| 535 | `MUresult muapiMemGetInfo_v2(size_t *free, size_t *total)` | v2 版本，size_t 宽度，支持 >4GB 内存 |
| 536 | `InitPlatform()` | 确保 MUSA 运行时已初始化，线程安全幂等 |
| 539 | `TlsCtxTop()` | 获取当前线程栈顶的 Context，每个线程独立 |
| 543 | `ctx->GetParentDevice()` | Context 与 Device 绑定，返回父设备 |
| 545 | `*total = device->GetProperties().totalGlobalMem` | 总量来自设备属性，**编译期/初始化时确定**，不随运行时分配变化 |
| 548 | `device->QueryGlobalMemFreeSize(free)` | **实时查询**空闲量，需 HAL → KMD 通信 |

**关键设计**：

- **`total` 字段**：直接读取 `Device::GetProperties().totalGlobalMem`，这是一个静态值，在设备枚举阶段从 HAL 层 `Hal::IDevice::GetProperties()` 获取，**不会随分配/释放变化**。
- **`free` 字段**：调用 `Device::QueryGlobalMemFreeSize()`，该调用通过 HAL 抽象接口最终触发 KMD ioctl 查询 GPU 显存实时空闲量，有一定开销。
- **v1 → v2 关系**：`muapiMemGetInfo` 直接 `reinterpret_cast<size_t*>(free/total)` 后调用 v2，仅做宽度转换，无额外逻辑。

### 2.4 Device::QueryGlobalMemFreeSize 流程

```
Device::QueryGlobalMemFreeSize(size_t* free)
  │
  └─ m_pHalDevice->QueryGlobalMemFreeSize(free)
        │
        └─ [HAL 抽象接口]
              │
              ├─ Linux DRM:  ioctl(MTGPU_IOCTL_QUERY_MEMINFO)
              └─ Windows WDDM: D3DKMTQueryVideoMemoryInfo
```

### 2.5 设计要点

| 要点 | 说明 |
|------|------|
| 线程安全 | 通过 `TlsCtxTop()` 获取线程局部上下文，天然线程安全 |
| total 静态性 | total 来自设备属性表，不反映当前分配状态 |
| 性能 | free 查询涉及 HAL → KMD 上下文切换，建议低频调用 |
| 错误处理 | 仅两种失败路径：初始化失败、上下文为空 |

---

## 3. muMemGetAddressRange / muMemGetAddressRange_v2

### 3.1 函数签名

```c
// v2 版本
MUresult muapiMemGetAddressRange_v2(MUdeviceptr *pbase, size_t *psize, MUdeviceptr dptr);

// v1 版本
MUresult muapiMemGetAddressRange(MUdeviceptr_v1 *pbase, unsigned int *psize, MUdeviceptr_v1 dptr);
```

### 3.2 完整调用链

```
User Code
  │
  └─ muMemGetAddressRange_v2(&base, &size, dptr)
        │
        ├─ InitPlatform()
        ├─ TlsCtxTop() → 获取 IContext
        ├─ ctx->GetMemoryByDevicePointer(dptr, &offset)
        │     │
        │     └─ MemoryTracker::FindRange(dptr, &offset)
        │         └─ m_RangeMap.upper_bound(ptr)
        │              ├─ 找到包含 ptr 的区间 → 返回 shared_ptr<IMemory> + offset
        │              └─ 未找到 → 返回 nullptr
        │
        ├─ pMemory == nullptr ?
        │     ├─ YES → MUSA_ERROR_NOT_FOUND
        │     └─ NO  ↓
        │
        ├─ pMemory->GetType() == memoryTypeVirtual ?
        │     ├─ YES → pMemory->GetPhysMemory(dptr, &tmpOff)
        │     │         遍历 m_VirtToPhysMap 找到物理映射
        │     │         ├─ 找到 → 物理 IMemory + 偏移
        │     │         └─ 未找到 → MUSA_ERROR_NOT_FOUND
        │     └─ NO  → 直接使用 pMemory
        │
        ├─ pbase = pMemory->GetDevicePointer()    // 基址（首字节）
        └─ psize = pMemory->GetSize()              // 内存块总大小
```

### 3.3 源码逐行分析

**`mu_memory.cpp:560-591`**:

```cpp
MUresult muapiMemGetAddressRange_v2(MUdeviceptr *pbase, size_t *psize, MUdeviceptr dptr) {
    MUresult status = InitPlatform();

    if (status == MUSA_SUCCESS) {
        do {
            Musa::IContext* pContext = TlsCtxTop();      // Step 1: 获取上下文
            if (nullptr == pContext) {
                status = MUSA_ERROR_INVALID_CONTEXT;
                break;                                    // break 跳出 do-while
            }
            if (nullptr != pbase || nullptr != psize) {  // Step 2: 至少一个输出
                Musa::IMemory* pMemory = pContext->GetMemoryByDevicePointer(dptr, &offset)->get();
                size_t tmpOff = 0;
                if (!pMemory) {                          // Step 3a: 未找到
                    status = MUSA_ERROR_NOT_FOUND;
                } else if (pMemory->GetType() == Musa::memoryTypeVirtual &&
                          !(pMemory = pMemory->GetPhysMemory(dptr, &tmpOff)->get())) {
                    // Step 3b: 虚拟内存需要查物理映射
                    status = MUSA_ERROR_NOT_FOUND;
                } else {
                    // Step 3c: 找到有效内存
                    if (nullptr != pbase) { *pbase = pMemory->GetDevicePointer(); }
                    if (nullptr != psize) { *psize = pMemory->GetSize(); }
                }
            }
        } while(0);
    }
    return status;
}
```

**逐行分析**：

| 行 | 代码 | 分析 |
|----|------|------|
| 564 | `do {` | do-while(0) 惯用法，便于 break 统一退出 |
| 565 | `TlsCtxTop()` | 获取当前线程上下文 |
| 566 | `nullptr == pContext` | 上下文检查，无上下文则 INVALID_CONTEXT |
| 570 | `nullptr != pbase \|\| nullptr != psize` | 至少一个输出参数非空时才执行查询 |
| 571 | `GetMemoryByDevicePointer(dptr, &offset)` | **核心查找**：通过 MemoryTracker 全局反查 |
| 573-574 | `!pMemory` → NOT_FOUND | 指针未在任何已分配内存范围内 |
| 575-576 | `memoryTypeVirtual` → `GetPhysMemory()` | 虚拟内存需额外查找物理绑定（VA→PA 映射） |
| 577 | 物理映射失败 → NOT_FOUND | 虚拟地址未绑定物理内存 |
| 579-580 | 返回基址和大小 | `pbase` 为内存块起始设备指针，非用户传入的 `dptr` |

**⚠️ 潜在 Bug**：变量 `offset` 在函数内被引用但未声明（`pContext->GetMemoryByDevicePointer(dptr, &offset)`），这是一个源码 bug。

**关键设计**：

- **`pbase` 返回的是基址**：不是用户传入的 `dptr`，而是该内存分配的起始设备指针
- **`psize` 是分配总大小**：不是 `dptr` 到末尾的剩余大小
- **Virtual→Physical 路径**：对于虚拟内存（Managed Memory），需要两层查找：先找到虚拟内存对象，再查物理绑定

### 3.4 MemoryTracker::FindRange 工作原理

```
MemoryTracker::FindRange(MUdeviceptr ptr, size_t* offset)
  │
  └─ std::map<MUdeviceptr, shared_ptr<IMemory>>::upper_bound(ptr)
       │
       ├─ it == begin() → 未找到，返回 nullptr
       ├─ it != begin() → prev(it) 为候选
       │     ├─ ptr < prev->first + prev->size → 命中，返回 prev
       │     └─ 否则 → 未找到
       └─ 返回 IMemory + offset (ptr - base)
```

### 3.5 IMemory::GetPhysMemory 工作原理

```
Memory::GetPhysMemory(MUdeviceptr virtPtr, size_t* offset)
  │
  └─ 遍历 m_PhysMemories (std::vector<PhysMemBinding>)
       ├─ 检查 virtPtr 是否在 [virtBase, virtBase+size) 范围内
       ├─ 命中 → 返回 physMemory + (virtPtr - virtBase)
       └─ 未命中 → 返回 nullptr
```

---

## 4. muMemHostGetDevicePointer / muMemHostGetDevicePointer_v2

### 4.1 函数签名

```c
MUresult muapiMemHostGetDevicePointer_v2(MUdeviceptr *pdptr, void *p, unsigned int Flags);
MUresult muapiMemHostGetDevicePointer(MUdeviceptr_v1 *pdptr, void *p, unsigned int Flags);
```

### 4.2 完整调用链

```
User Code
  │
  └─ muMemHostGetDevicePointer_v2(&dptr, hostPtr, 0)
        │
        ├─ InitPlatform()
        ├─ TlsCtxTop() → 获取 IContext
        ├─ Flags == 0 ? → 必须为 0
        │
        ├─ ctx->GetMemoryByDevicePointer(reinterpret_cast<uint64_t>(p), &offset)
        │     └─ MemoryTracker::FindRange()
        │
        ├─ pMemory->GetType() 检查:
        │     ├─ memoryTypePinnedHost        ✓
        │     ├─ memoryTypeRegisteredPinned  ✓
        │     ├─ memoryTypeManaged           ✓
        │     └─ 其他                        → MUSA_ERROR_INVALID_VALUE
        │
        └─ *pdptr = pMemory->GetDevicePointer() + offset
```

### 4.3 源码逐行分析

**`mu_memory.cpp:593-624`**:

```cpp
MUresult muapiMemHostGetDevicePointer_v2(MUdeviceptr *pdptr, void *p, unsigned int Flags) {
    MUresult status = InitPlatform();

    if (status == MUSA_SUCCESS) {
        do {
            if (0 != Flags ||                       // Flags 必须为 0
                nullptr == pdptr ||
                nullptr == p) {
                status = MUSA_ERROR_INVALID_VALUE;
                break;
            }
            Musa::IContext* pContext = TlsCtxTop();
            if (nullptr == pContext) {
                status = MUSA_ERROR_INVALID_CONTEXT;
                break;
            }
            size_t offset = 0;
            Musa::IMemory* pMemory = pContext->GetMemoryByDevicePointer(
                reinterpret_cast<uint64_t>(p), &offset)->get();
            if (!pMemory || (pMemory->GetType() != Musa::memoryTypePinnedHost &&
                             pMemory->GetType() != Musa::memoryTypeRegisteredPinnedHost &&
                             pMemory->GetType() != Musa::memoryTypeManaged)) {
                status = MUSA_ERROR_INVALID_VALUE;
                break;
            }
            *pdptr = pMemory->GetDevicePointer() + offset;
            tprintf(LOG_MEM, "host_ptr=%p returned device_ptr=%p\n",
                    p, reinterpret_cast<void*>(*pdptr));
        } while(0);
    }
    return status;
}
```

**关键分析**：

- **Flags 硬编码为 0**：与 CUDA `cudaHostGetDevicePointer` 不同，MUSA 不接受任何 flags。注释标明 "From spec: must be 0"。
- **指针查找语义**：通过 `reinterpret_cast<uint64_t>(p)` 将主机指针转为数值，在 MemoryTracker 中查找。这里存在**架构隐患**：
  - Fixed Host Memory（`memoryTypePinnedHost`）的 key 是 `devicePointer`
  - Registered Pinned Host（`memoryTypeRegisteredPinnedHost`）的 key 是 `hostPointer`
  - 传入主机指针 `p`，对 Fixed Host 查找会失败（key 类型不匹配）
  - **这可能是源码 bug 或设计特殊性** ⚠️
- **支持类型**：仅 `PinnedHost`、`RegisteredPinnedHost`、`Managed` 三种。

---

## 5. muMemHostGetFlags

### 5.1 函数签名

```c
MUresult muapiMemHostGetFlags(unsigned int* pFlags, void* p);
```

### 5.2 完整调用链

```
User Code
  │
  └─ muMemHostGetFlags(&flags, hostPtr)
        │
        ├─ InitPlatform()
        ├─ TlsCtxTop() → 获取 IContext
        ├─ ctx->GetMemoryByDevicePointer(reinterpret_cast<uint64_t>(p))
        │     └─ MemoryTracker::FindRange()
        ├─ pMemory 类型检查:
        │     ├─ PinnedHost / RegisteredPinnedHost / Managed → 有效
        │     └─ 其他 → MUSA_ERROR_INVALID_VALUE
        │
        ├─ *pFlags = pMemory->GetFlags()
        │
        └─ 按类型做 flags 掩码过滤:
             ├─ PinnedHost:       mask = PORTABLE | DEVICEMAP | WRITECOMBINED
             ├─ RegisteredPinned: mask = PORTABLE | DEVICEMAP | READ_ONLY | IOMEMORY
             └─ Managed:          不做掩码（直接返回原始 flags）
```

### 5.3 源码逐行分析

**`mu_memory.cpp:630-677`**:

```cpp
MUresult muapiMemHostGetFlags(unsigned int* pFlags, void* p) {
    MUresult status = InitPlatform();

    if (status == MUSA_SUCCESS) {
        do {
            if (nullptr == pFlags || nullptr == p) {
                status = MUSA_ERROR_INVALID_VALUE;
                break;
            }
            Musa::IContext* pContext = TlsCtxTop();
            if (nullptr == pContext) {
                status = MUSA_ERROR_INVALID_CONTEXT;
                break;
            }
            Musa::IMemory* pMemory = pContext->GetMemoryByDevicePointer(
                reinterpret_cast<uint64_t>(p))->get();
            if (!pMemory || (pMemory->GetType() != Musa::memoryTypePinnedHost &&
                             pMemory->GetType() != Musa::memoryTypeRegisteredPinnedHost &&
                             pMemory->GetType() != Musa::memoryTypeManaged)) {
                tprintf(LOG_ERR, "ptr:%p: cannot find pointer info or wrong alloc type\n", p);
                status = MUSA_ERROR_INVALID_VALUE;
                break;
            }
            *pFlags = pMemory->GetFlags();
            switch (pMemory->GetType()) {
                case Musa::memoryTypePinnedHost: {
                    int userFlagsMask = MU_MEMHOSTALLOC_PORTABLE |
                                        MU_MEMHOSTALLOC_DEVICEMAP |
                                        MU_MEMHOSTALLOC_WRITECOMBINED;
                    *pFlags = *pFlags & userFlagsMask;
                    break;
                }
                case Musa::memoryTypeRegisteredPinnedHost: {
                    int userFlagsMask = MU_MEMHOSTREGISTER_PORTABLE |
                                        MU_MEMHOSTREGISTER_DEVICEMAP |
                                        MU_MEMHOSTREGISTER_READ_ONLY |
                                        MU_MEMHOSTREGISTER_IOMEMORY;
                    *pFlags = *pFlags & userFlagsMask;
                    break;
                }
                case Musa::memoryTypeManaged:
                default:
                    break;
            }
        } while(0);
    }

    return status;
}
```

**关键分析**：

- **掩码设计**：内部 flags 包含驱动内部标记（如 `MU_MEMORY_ALLOC_OVERLAP_CHECK`），输出给用户前需掩码过滤。
- **PinnedHost 掩码**: `PORTABLE | DEVICEMAP | WRITECOMBINED` — 用户分配时可指定的 flags。
- **RegisteredPinned 掩码**: `PORTABLE | DEVICEMAP | READ_ONLY | IOMEMORY` — 额外支持只读和 MMIO。
- **Managed 类型**: 不做掩码，但 Managed 内存通常 flags=0，由运行时自动管理访问属性。
- **日志输出**: 对 PinnedHost/RegisteredPinned 类型，使用掩码后 flags 为 0 时日志仍可能输出（日志在掩码前）。

### 5.4 与 muMemHostRegister 的互操作

| 操作 | API | 关系 |
|------|-----|------|
| 注册内存 | `muMemHostRegister(p, size, flags)` | flags 含内部标记 |
| 查询 flags | `muMemHostGetFlags(&flags, p)` | flags 仅含用户可见标记 |
| 取消注册 | `muMemHostUnregister(p)` | 销毁对应 IMemory |

> `MU_MEMHOSTREGISTER_OVERLAP_CHECK` 在注册时自动添加（文档 04 分析），但不出现在 `GetFlags` 返回值中。

---

## 6. 潜在 Bug 与设计隐患汇总

| 问题 | 位置 | 严重性 | 说明 |
|------|------|--------|------|
| 未声明变量 `offset` | `muapiMemGetAddressRange_v2` L571 | **高** | 引用了未声明的局部变量，编译可能通过（隐式声明）或行为未定义 |
| 指针查找语义歧义 | `muapiMemHostGetDevicePointer` | **中** | 函数名传主机指针，但 `GetMemoryByDevicePointer` 按设备指针查找；对 Fixed Host 内存 key 不匹配 |
| 边界条件: 无输出参数 | `muapiMemGetAddressRange_v2` | **低** | `pbase==nullptr && psize==nullptr` 时返回 `SUCCESS` 但不做任何有意义的操作 |

---

## 7. 附录: 相关内存类型与 key 映射关系

| 内存类型 | MemoryTracker Key | Host Pointer | Device Pointer |
|----------|-------------------|--------------|----------------|
| `memoryTypePinnedHost` | `devicePointer` | 有 | 有（DMA 映射） |
| `memoryTypeRegisteredPinnedHost` | `hostPointer` | 有 | 无（或有 MMIO） |
| `memoryTypeManaged` | `devicePointer` | 有 | 有（统一地址） |
| `memoryTypeGeneral` | `devicePointer` | 无 | 有 |
| `memoryTypeVirtual` | `devicePointer` (VIRT) | 无 | 有（虚拟） |

> 详见文档 03（muMemHostAlloc）和文档 14（Memory API Source Deep Dive）。
