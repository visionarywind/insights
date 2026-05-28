# 通用指针属性查询 API 深度源码分析

> **文档编号**: 16  
> **相关文件**: `musa/doc/memory_api/README.md`  
> **源码路径**:  
> - `musa/src/driver/mu_memory.cpp` (Driver 层入口)  
> - `musa/src/musa/core/memory.h` (IMemory 接口)  
> - `musa/src/hal/m3d/memory.h` (HAL 层)  
> - `musa/src/musa/shared_include/musa.h` (MUSA API 定义)  

---

## 1. 功能概述

`muPointerGetAttributes` 是 MUSA 内存子系统的**通用指针内省接口**，通过单个 API 可查询任意设备指针的多种属性，替代了 CUDA 中分散的多个查询函数。

### 1.1 函数签名

```c
// 单属性查询（原始接口）
MUresult muapiPointerGetAttribute(void *data, MUpointer_attribute attribute, MUdeviceptr ptr);

// 多属性批量查询（推荐接口）
MUresult muapiPointerGetAttributes(unsigned int numAttributes,
                                    MUpointer_attribute* attributes,
                                    void **data,
                                    MUdeviceptr ptr);
```

### 1.2 支持的 attribute 枚举

```c
typedef enum {
    MU_POINTER_ATTRIBUTE_CONTEXT              = 1,    // 所属上下文
    MU_POINTER_ATTRIBUTE_MEMORY_TYPE           = 2,    // 内存类型
    MU_POINTER_ATTRIBUTE_DEVICE_POINTER        = 3,    // 设备指针值
    MU_POINTER_ATTRIBUTE_HOST_POINTER          = 4,    // 主机指针值
    MU_POINTER_ATTRIBUTE_IS_MANAGED            = 5,    // 是否为 Managed 内存
    MU_POINTER_ATTRIBUTE_DEVICE_ORDINAL        = 6,    // 设备序号
    MU_POINTER_ATTRIBUTE_RANGE_START_ADDR      = 7,    // 所属内存块起始地址
    MU_POINTER_ATTRIBUTE_RANGE_SIZE            = 8,    // 所属内存块大小
    MU_POINTER_ATTRIBUTE_MAPPED                = 9,    // 是否映射到设备
    MU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 10, // 是否支持 GPU Direct RDMA
    MU_POINTER_ATTRIBUTE_IS_LEGACY_MUSA_IPC_CAPABLE  = 11, // IPC 导出能力
    MU_POINTER_ATTRIBUTE_SYNC_MEMOPS           = 12,   // 是否同步 MemOps
    MU_POINTER_ATTRIBUTE_ACCESS_FLAGS          = 13,   // 访问标志
    MU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES  = 14,   // 允许的导出句柄类型
    MU_POINTER_ATTRIBUTE_BUFFER_ID             = 15,   // Buffer ID
    MU_POINTER_ATTRIBUTE_P2P_TOKENS            = 16,   // P2P Tokens（不支持）
    MU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE        = 17,   // MemPool Handle（不支持）
} MUpointer_attribute;
```

### 1.3 对应 CUDA

| MUSA | CUDA |
|------|------|
| `muPointerGetAttribute` | `cuPointerGetAttribute` |
| `muPointerGetAttributes` | `cuPointerGetAttributes` |

---

## 2. 完整调用链

```
User Code
  │
  ├─ muPointerGetAttributes(numAttrs, attrs, data, ptr)
  │     │
  │     ├─ 参数校验 (numAttributes, data, attributes)
  │     ├─ ptr == 0 → memset(data, 0, ...) 快速返回
  │     │
  │     └─ for (i = 0; i < numAttributes; i++)
  │           │
  │           └─ imuapiPointerGetAttribute(data[i], attrs[i], ptr)
  │                 │
  │                 ├─ InitPlatform()
  │                 ├─ 参数校验 (data, ptr)
  │                 │
  │                 ├─ MemoryTracker::FindRange(ptr, &offset)
  │                 │     └─ m_RangeMap 二分查找
  │                 │
  │                 ├─ memoryTypeVirtual ?
  │                 │     ├─ YES → GetPhysMemory(ptr, &offset)
  │                 │     │         查物理映射，失败 → NOT_FOUND
  │                 │     └─ NO  → 继续
  │                 │
  │                 ├─ pContext = pMemory->GetContext()
  │                 │
  │                 └─ switch (attribute):
  │                       ├─ CONTEXT          → *data = pContext
  │                       ├─ MEMORY_TYPE      → 查表转换
  │                       ├─ DEVICE_POINTER   → devicePtr + offset
  │                       ├─ HOST_POINTER     → hostPtr + offset (仅主机可见)
  │                       ├─ IS_MANAGED       → type == Managed
  │                       ├─ DEVICE_ORDINAL   → ctx->deviceId
  │                       ├─ RANGE_START_ADDR → ptr - offset
  │                       ├─ RANGE_SIZE       → pMemory->GetSize()
  │                       ├─ MAPPED           → props & DeviceMapped
  │                       ├─ GPU_DIRECT_RDMA  → type == General/PitchedGeneral
  │                       ├─ IPC_CAPABLE      → viewCap & IpcExportable
  │                       ├─ SYNC_MEMOPS      → attrs.syncMemOps
  │                       ├─ ACCESS_FLAGS     → props 转换
  │                       ├─ ALLOWED_HANDLES  → type 检查
  │                       ├─ BUFFER_ID       → pMemory->GetSeqID()
  │                       └─ P2P_TOKENS /    → NOT_SUPPORTED
  │                          MEMPOOL_HANDLE
```

---

## 3. 源码逐行分析

### 3.1 `imuapiPointerGetAttribute` — 内部实现

**`mu_memory.cpp:103-259`**:

```cpp
static MUresult imuapiPointerGetAttribute(void *data, MUpointer_attribute attribute, MUdeviceptr ptr) {
    MUresult status = InitPlatform();                           // L104: 平台初始化
    if (status == MUSA_SUCCESS) {
        Musa::Context* pContext = nullptr;
        Musa::IMemory* pMemory  = nullptr;
        size_t offset;

        if (nullptr == data || 0 == ptr) {                      // L110: 参数校验
            status = MUSA_ERROR_INVALID_VALUE;
        }

        if (status == MUSA_SUCCESS) {
            // L115: 通过 MemoryTracker 全局反查
            pMemory = Musa::Platform::Get().GetMemoryTracker()
                .FindRange(ptr, &offset, Hal::memoryPropertyPhysical)->get();
            if (!pMemory) {                                     // L117: 未找到
                tprintf(LOG_ERR, "ptr:%#llx: cannot find pointer info\n", ptr);
                status = MUSA_ERROR_INVALID_VALUE;
            }
        }

        // L122: 虚拟内存 → 查物理映射
        if ((status == MUSA_SUCCESS) &&
            (pMemory->GetType() == Musa::memoryTypeVirtual)) {
            pMemory = pMemory->GetPhysMemory(ptr, &offset)->get();
            if (!pMemory) {                                     // L125: 物理映射失败
                tprintf(LOG_ERR, "ptr:%#llx doesn't bind to any physical memory\n", ptr);
                status = MUSA_ERROR_INVALID_VALUE;
            }
        }

        if (status == MUSA_SUCCESS) {
            pContext = static_cast<Musa::Context*>(pMemory->GetContext()); // L131
            switch (attribute) {
            // ... [各 attribute 处理，见下文分析]
            default:
                status = MUSA_ERROR_INVALID_VALUE;              // L252: 未知属性
                break;
            }
        }
    }
    return status;                                               // L258
}
```

**关键设计**：

- **统一查找入口**：所有查询共享同一个 `FindRange` 前缀查找逻辑，避免重复代码
- **三层查找**：`Tracker::FindRange` → 虚拟内存二次 `GetPhysMemory` → 属性/方法读取
- **函数内静态**：`imuapiPointerGetAttribute` 为 `static` 函数，仅在本文件可见

### 3.2 各 attribute 逐行分析

#### MU_POINTER_ATTRIBUTE_CONTEXT (L133-135)

```cpp
case MU_POINTER_ATTRIBUTE_CONTEXT:
    *static_cast<MUcontext*>(data) = pContext;
    break;
```

返回内存对象所属的 `Context`。注意：返回的是 `Context` 的原始指针，调用者需确保 Context 生命周期。

#### MU_POINTER_ATTRIBUTE_MEMORY_TYPE (L137-158)

```cpp
case MU_POINTER_ATTRIBUTE_MEMORY_TYPE: {
    unsigned int* pType = static_cast<unsigned int*>(data);
    switch (pMemory->GetType()) {
        case Musa::memoryTypeGeneral:
        case Musa::memoryTypePitchedGeneral:
        case Musa::memoryTypeManaged:
        case Musa::memoryTypeIpcImport:
        case Musa::memoryTypeExternal:
            *pType = MU_MEMORYTYPE_DEVICE;     // 设备可见内存 → DEVICE
            break;
        case Musa::memoryTypePinnedHost:
        case Musa::memoryTypeRegisteredPinnedHost:
            *pType = MU_MEMORYTYPE_HOST;       // 主机锁定内存 → HOST
            break;
        case Musa::memoryTypePrealloc:
        case Musa::memoryTypeVirtual:
        default:
            ShouldNotReachHere();              // ← 死代码/不可达路径 ⚠️
            break;
    }
    break;
}
```

**类型映射表**:

| IMemory Type | MU_MEMORY_TYPE | 说明 |
|---|---|---|
| `memoryTypeGeneral` | `MU_MEMORYTYPE_DEVICE` | 通用设备内存 |
| `memoryTypePitchedGeneral` | `MU_MEMORYTYPE_DEVICE` | Pitched 2D 设备内存 |
| `memoryTypeManaged` | `MU_MEMORYTYPE_DEVICE` | 统一内存（Managed） |
| `memoryTypeIpcImport` | `MU_MEMORYTYPE_DEVICE` | IPC 导入内存 |
| `memoryTypeExternal` | `MU_MEMORYTYPE_DEVICE` | 外部导入内存 |
| `memoryTypePinnedHost` | `MU_MEMORYTYPE_HOST` | 页锁定主机内存 |
| `memoryTypeRegisteredPinnedHost` | `MU_MEMORYTYPE_HOST` | 注册的页锁定内存 |
| `memoryTypePrealloc` | **不可达** ⚠️ | 预分配内存，无公开 API 触发 |
| `memoryTypeVirtual` | **不可达** ⚠️ | 虚拟内存，不应直接查询 |

> **死代码标注**: `memoryTypePrealloc` 和 `memoryTypeVirtual` 出现在 `default` 之前，意味着如果指针指向这两种内存类型，`ShouldNotReachHere()` 将触发断言。这两种类型的内存**不应直接通过用户可见指针查询**。

#### MU_POINTER_ATTRIBUTE_DEVICE_POINTER (L161-164)

```cpp
case MU_POINTER_ATTRIBUTE_DEVICE_POINTER: {
    *static_cast<MUdeviceptr*>(data) = pMemory->GetDevicePointer() + offset;
    break;
}
```

返回内存块的设备指针 + sub-allocation 偏移。即使是 host-visible 内存，也返回其设备映射地址。

#### MU_POINTER_ATTRIBUTE_HOST_POINTER (L166-175)

```cpp
case MU_POINTER_ATTRIBUTE_HOST_POINTER: {
    if (pMemory->GetType() == Musa::memoryTypePinnedHost ||
        pMemory->GetType() == Musa::memoryTypeRegisteredPinnedHost ||
        pMemory->GetType() == Musa::memoryTypeManaged) {
        *static_cast<void**>(data) =
            static_cast<char*>(pMemory->GetHostPointer()) + offset;
    } else {
        status = MUSA_ERROR_INVALID_VALUE;          // 设备内存无主机指针
    }
    break;
}
```

仅主机可见内存（PinnedHost、RegisteredPinnedHost、Managed）支持查询主机指针。

**⚠️ Bug**: `memoryTypeManaged` 的 `GetHostPointer()` 行为取决于具体实现 — Managed 内存通常使用统一虚拟地址，其 "host pointer" 语义可能不同于传统 pinned memory。

#### MU_POINTER_ATTRIBUTE_IS_MANAGED (L177-179)

```cpp
case MU_POINTER_ATTRIBUTE_IS_MANAGED:
    *static_cast<bool*>(data) =
        pMemory->GetType() == Musa::memoryTypeManaged;
    break;
```

#### MU_POINTER_ATTRIBUTE_DEVICE_ORDINAL (L181-183)

```cpp
case MU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
    *static_cast<int*>(data) = pContext->GetDevice()->GetId();
    break;
```

#### MU_POINTER_ATTRIBUTE_RANGE_START_ADDR (L185-187)

```cpp
case MU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
    *static_cast<void**>(data) = reinterpret_cast<void*>(ptr - offset);
    break;
```

**关键**: 返回 `ptr - offset`，即用户传入指针所属内存块的**起始地址**。这是 sub-allocation 场景下的逆运算。

#### MU_POINTER_ATTRIBUTE_RANGE_SIZE (L189-191)

```cpp
case MU_POINTER_ATTRIBUTE_RANGE_SIZE:
    *static_cast<size_t*>(data) = pMemory->GetSize();
    break;
```

返回内存块**总大小**，而非从指针到末尾的剩余大小。

#### MU_POINTER_ATTRIBUTE_MAPPED (L193-195)

```cpp
case MU_POINTER_ATTRIBUTE_MAPPED:
    *static_cast<bool*>(data) =
        pMemory->GetProps(pContext->GetDevice()) & Hal::memoryPropertyDeviceMapped;
    break;
```

查询 `memoryPropertyDeviceMapped` 属性位。

#### MU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE (L197-200)

```cpp
case MU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE:
    *static_cast<bool*>(data) =
        pMemory->GetType() == Musa::memoryTypeGeneral ||
        pMemory->GetType() == Musa::memoryTypePitchedGeneral;
    break;
```

仅 General/PitchedGeneral 内存类型支持 GPU Direct RDMA。

#### MU_POINTER_ATTRIBUTE_IS_LEGACY_MUSA_IPC_CAPABLE (L202-204)

```cpp
case MU_POINTER_ATTRIBUTE_IS_LEGACY_MUSA_IPC_CAPABLE:
    *static_cast<bool*>(data) =
        pMemory->GetViewCapability(pContext->GetDevice()) &
        Hal::memoryViewCapabilityIpcExportable;
    break;
```

查询 `viewCapability` 中的 IPC 导出位。

#### MU_POINTER_ATTRIBUTE_SYNC_MEMOPS (L206-209)

```cpp
case MU_POINTER_ATTRIBUTE_SYNC_MEMOPS:
    *static_cast<bool*>(data) = pMemory->GetAttributes().syncMemOps;
    break;
```

#### MU_POINTER_ATTRIBUTE_ACCESS_FLAGS (L213-226)

```cpp
case MU_POINTER_ATTRIBUTE_ACCESS_FLAGS: {
    auto props = pMemory->GetProps(pContext->GetDevice());
    if (props & Hal::memoryPropertyDeviceMapped) {
        if (props & Hal::memoryPropertyDeviceWriteable &&
            props & Hal::memoryPropertyDeviceVisible) {
            *static_cast<MUmemAccess_flags*>(data) =
                MU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        } else {
            *static_cast<MUmemAccess_flags*>(data) =
                MU_MEM_ACCESS_FLAGS_PROT_READ;
        }
    } else {
        *static_cast<MUmemAccess_flags*>(data) =
            MU_MEM_ACCESS_FLAGS_PROT_NONE;
    }
    break;
}
```

**访问标志转换逻辑**:

```
Device Mapped ?
  ├─ YES → Device Writeable && Device Visible ?
  │         ├─ YES → READWRITE
  │         └─ NO  → READ
  └─ NO  → NONE
```

#### MU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES (L230-237)

```cpp
case MU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES: {
    *static_cast<int*>(data) = 0;
    if (pMemory->GetType() != Musa::memoryTypeIpcImport &&
        pMemory->GetType() != Musa::memoryTypeExternal) {
        *static_cast<int*>(data) |= MU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    }
    break;
}
```

仅非 IPC Import、非 External 内存支持 POSIX 文件描述符导出。

#### MU_POINTER_ATTRIBUTE_BUFFER_ID (L241-243)

```cpp
case MU_POINTER_ATTRIBUTE_BUFFER_ID:
    *static_cast<uint64_t*>(data) = pMemory->GetSeqID();
    break;
```

#### MU_POINTER_ATTRIBUTE_P2P_TOKENS / MU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE (L246-249)

```cpp
case MU_POINTER_ATTRIBUTE_P2P_TOKENS:
case MU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE:
    status = MUSA_ERROR_NOT_SUPPORTED;
    break;
```

> **死代码/不支持**: 这两个 attribute 直接返回 `NOT_SUPPORTED`。

### 3.3 `muPointerGetAttributes` — 多属性批量查询

**`mu_memory.cpp:683-707`**:

```cpp
MUresult muapiPointerGetAttributes(unsigned int numAttributes,
                                    MUpointer_attribute* attributes,
                                    void **data,
                                    MUdeviceptr ptr) {
    MUresult status = MUSA_SUCCESS;

    do {
        if (0 == numAttributes || nullptr == data || nullptr == attributes) {
            status = MUSA_ERROR_INVALID_VALUE;
            break;
        }
        if (0 == ptr) {                              // 指针为 0 → 输出全 0
            memset(data, 0, numAttributes * sizeof(void*));
            break;
        }
        for (unsigned int i = 0; i < numAttributes; i++) {
            status = imuapiPointerGetAttribute(data[i], attributes[i], ptr);
            if (status != MUSA_SUCCESS) {
                break;                                // 任一失败立即终止
            }
        }
    } while(0);

    return status;
}
```

**关键设计**：

- **串型查询**：多个 attribute 串行逐个查询，一个失败立即终止
- **ptr == 0 优化**：指针为 0 时直接 memset 输出为 0，快速返回
- **原子性**：非全有或全无，任一失败前已成功的查询结果保留在输出中

---

## 4. 与相关 API 对比

| 查询需求 | 推荐 API | 备注 |
|----------|----------|------|
| 内存类型 | `muPointerGetAttribute(ATTR_MEMORY_TYPE)` | 比 `muMemGetAddressRange` 更通用 |
| 所属上下文 | `muPointerGetAttribute(ATTR_CONTEXT)` | 唯一途径 |
| 设备指针 | `muPointerGetAttribute(ATTR_DEVICE_POINTER)` | 等价于 `muMemHostGetDevicePointer` |
| 主机指针 | `muPointerGetAttribute(ATTR_HOST_POINTER)` | 仅主机可见内存 |
| 内存块范围 | `muPointerGetAttribute(ATTR_RANGE_START/SIZE)` | 比 `muMemGetAddressRange` 支持更多内存类型 |
| IPC 能力 | `muPointerGetAttribute(ATTR_IS_LEGACY_MUSA_IPC_CAPABLE)` | 查询 viewCapability |
| 管理状态 | `muPointerGetAttribute(ATTR_IS_MANAGED)` | Managed 内存标识 |

---

## 5. 设计要点与陷阱

| 要点 | 说明 |
|------|------|
| 统一入口 | 所有指针属性查询走同一路径，减少代码重复 |
| 虚拟内存透明 | 对 `memoryTypeVirtual` 自动查物理映射，用户无感知 |
| `ShouldNotReachHere` | `memoryTypePrealloc` 和 `memoryTypeVirtual` 在 TYPE 分支中被标记为不可达 |
| 偏移语义 | `DEVICE_POINTER` 和 `HOST_POINTER` 返回的都是基址+偏移，而非用户传入的原始指针 |
| 不支持项 | `P2P_TOKENS` 和 `MEMPOOL_HANDLE` 直接返回 `NOT_SUPPORTED` |
| ⚠️ Managed Host Pointer | Managed 内存的 `GetHostPointer()` 语义可能与 PinnedHost 不同，需关注具体 HAL 实现 |

---

## 6. ASCII 时序图

```
User                           Driver (mu_memory.cpp)              Core/Memory
  │                                  │                               │
  │ muPointerGetAttributes()         │                               │
  │─────────────────────────────────>│                               │
  │                                  │ imuapiPointerGetAttribute()   │
  │                                  │──────────────────────────────>│
  │                                  │  InitPlatform()               │
  │                                  │  FindRange(ptr)               │
  │                                  │<── IMemory + offset ──────────│
  │                                  │                               │
  │                                  │  switch(attribute)            │
  │                                  │    case CONTEXT:              │
  │                                  │      pMemory->GetContext()    │
  │                                  │<── pContext ──────────────────│
  │                                  │    case MEMORY_TYPE:          │
  │                                  │      pMemory->GetType()       │
  │                                  │      → MU_MEMORYTYPE_*        │
  │                                  │    case DEVICE_POINTER:       │
  │                                  │      pMem->GetDevicePtr()+off │
  │                                  │<── devicePtr ─────────────────│
  │<──────────── result ─────────────│                               │
  │                                  │                               │
  │ [Virtual Memory Path]            │                               │
  │                                  │  GetPhysMemory(ptr)           │
  │                                  │──────────────────────────────>│
  │                                  │  遍历 m_VirtToPhysMap         │
  │                                  │<── phys IMemory + off ────────│
  │                                  │                               │
```
