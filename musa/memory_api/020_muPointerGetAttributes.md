# muPointerGetAttributes — 指针属性查询

> 源码文件：`musa/src/driver/mu_memory.cpp:103-259` (含 `imuapiPointerGetAttribute`)

## 1. 功能概述

查询一个 `MUdeviceptr` 指针的各类属性，包括内存类型、设备指针、主机指针、所属上下文、IPC 能力等。

## 2. 完整调用链

```
muapiPointerGetAttribute(data, attribute, ptr)            [mu_memory.cpp:679]
  │  (直接转发)
  │
  +-- imuapiPointerGetAttribute(data, attribute, ptr)     [mu_memory.cpp:103]
        │
        +-- InitPlatform()                                [internal.h:306]
        │
        +-- data != nullptr && ptr != 0 检查              [mu_memory.cpp:110]
        │
        +-- MemoryTracker::FindRange(ptr, &offset,        [mu_memory.cpp:115]
        │     Hal::memoryPropertyPhysical)
        │     通过全局 MemoryTracker 查找 ptr 所属的 IMemory*
        │     返回: pMemory + offset (在 allocation 中的偏移)
        │
        +-- if 找不到 → MUSA_ERROR_INVALID_VALUE
        │
        +-- if (pMemory->GetType() == memoryTypeVirtual):  [mu_memory.cpp:122-128]
        │     pMemory = pMemory->GetPhysMemory(ptr, &offset)
        │     获取虚拟内存对应的实际物理内存
        │     (遍历 MemoryTracker 的物理子内存)
        │
        +-- pContext = pMemory->GetContext()               [mu_memory.cpp:131]
        │
        +-- switch (attribute):                            [mu_memory.cpp:132-255]
              │
              ├─ MU_POINTER_ATTRIBUTE_CONTEXT:
              │     *data = pContext (转为 MUcontext)
              │
              ├─ MU_POINTER_ATTRIBUTE_MEMORY_TYPE:
              │     switch (pMemory->GetType()):
              │       General/PitchedGeneral/Managed/
              │       IpcImport/External → MU_MEMORYTYPE_DEVICE
              │       PinnedHost/RegisteredPinnedHost → MU_MEMORYTYPE_HOST
              │
              ├─ MU_POINTER_ATTRIBUTE_DEVICE_POINTER:
              │     *data = pMemory->GetDevicePointer() + offset
              │
              ├─ MU_POINTER_ATTRIBUTE_HOST_POINTER:
              │     if (PinnedHost/RegisteredPinnedHost/Managed):
              │       *data = pMemory->GetHostPointer() + offset
              │     else:
              │       MUSA_ERROR_INVALID_VALUE (不支持)
              │
              ├─ MU_POINTER_ATTRIBUTE_IS_MANAGED:
              │     *data = (type == memoryTypeManaged)
              │
              ├─ MU_POINTER_ATTRIBUTE_DEVICE_ORDINAL:
              │     *data = pContext->GetDevice()->GetId()
              │
              ├─ MU_POINTER_ATTRIBUTE_RANGE_START_ADDR:
              │     *data = ptr - offset (allocation 的起始地址)
              │
              ├─ MU_POINTER_ATTRIBUTE_RANGE_SIZE:
              │     *data = pMemory->GetSize()
              │
              ├─ MU_POINTER_ATTRIBUTE_MAPPED:
              │     *data = pMemory->GetProps(device) & DeviceMapped
              │     (通过 Hal::IMemory::GetProps 查询)
              │
              ├─ MU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE:
              │     *data = (type == General || PitchedGeneral)
              │     (仅设备本地内存支持 RDMA)
              │
              ├─ MU_POINTER_ATTRIBUTE_IS_LEGACY_MUSA_IPC_CAPABLE:
              │     *data = capability & IpcExportable
              │     (检查 IPC 导出能力)
              │
              ├─ MU_POINTER_ATTRIBUTE_SYNC_MEMOPS:
              │     *data = attributes.syncMemOps   [memory.cpp:208]
              │     (通过 SetAttribute 设置)
              │
              ├─ MU_POINTER_ATTRIBUTE_ACCESS_FLAGS:
              │     if DeviceMapped:
              │       if DeviceWriteable && DeviceVisible:
              │         → MU_MEM_ACCESS_FLAGS_PROT_READWRITE
              │       else:
              │         → MU_MEM_ACCESS_FLAGS_PROT_READ
              │     else:
              │       → MU_MEM_ACCESS_FLAGS_PROT_NONE
              │
              ├─ MU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES:
              │     if (type != IpcImport && type != External):
              │       → MU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
              │     (IPC/External 类型不支持文件描述符导出,
              │      因为它们已通过其他句柄传递)
              │
              ├─ MU_POINTER_ATTRIBUTE_BUFFER_ID:
              │     *data = pMemory->GetSeqID()
              │     (唯一 allocation 编号)
              │
              ├─ MU_POINTER_ATTRIBUTE_P2P_TOKENS /
              │   MU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE:
              │     MUSA_ERROR_NOT_SUPPORTED (暂不支持)
```

## 3. 虚拟内存的特殊处理

```
如果 pMemory->GetType() == memoryTypeVirtual:
  │
  ├─ 说明 ptr 是一个子分配的虚拟地址
  │  (来自 MemoryPool 的 SubAllocate)
  │
  └─ pMemory->GetPhysMemory(ptr, &offset)
      │
      └─ m_PhysTracker.FindRange(ptr, offset, None)
          返回实际物理内存 (IMemory*) 及子偏移
```

## 4. muPointerGetAttributes (多属性批量查询)

```
muapiPointerGetAttributes(numAttributes, attributes, data, ptr)  [mu_memory.cpp:683]
  │
  +-- 参数校验:
  │     numAttributes > 0
  │     attributes != nullptr
  │     data != nullptr
  │
  +-- for i = 0..numAttributes-1:
  │     imuapiPointerGetAttribute(data[i], attributes[i], ptr)
  │     (逐条调用单属性查询, 遇到错误即中止)
```

## 5. Mapped 属性的含义

`MU_POINTER_ATTRIBUTE_MAPPED` 返回 `true` 的条件:

```
pMemory->GetProps(pContext->GetDevice()) & Hal::memoryPropertyDeviceMapped
```

这意味着:
- 设备可以通过 DMA 直接访问该内存
- 对于 PinnedHost: 仅当用户传入 `MU_MEMHOSTALLOC_DEVICEMAP` 时为 true
- 对于 RegisteredPinnedHost: 仅当用户传入 `MU_MEMHOSTREGISTER_DEVICEMAP` 时为 true
- 对于 General 设备内存: 恒为 true

## 6. 日志验证结果

最小用例 `memory_api_callflow_demo.cpp` 打开 `MUSA_DRIVER_CALLFLOW_DEBUG=1` 后确认批量查询不会一次性解析所有属性，而是逐项调用 `imuapiPointerGetAttribute`：

```text
muapiPointerGetAttributes
  -> imuapiPointerGetAttribute(attr=MEMORY_TYPE)
  -> imuapiPointerGetAttribute(attr=DEVICE_POINTER)
  -> imuapiPointerGetAttribute(attr=RANGE_START_ADDR)
  -> imuapiPointerGetAttribute(attr=RANGE_SIZE)
  -> imuapiPointerGetAttribute(attr=DEVICE_ORDINAL)
```

每一次 `imuapiPointerGetAttribute` 都会重新执行 `MemoryTracker::FindRange(ptr, &offset, Hal::memoryPropertyPhysical)`，再根据 attribute 分支写入输出数据。

## 7. 相关源码位置

| 文件 | 行数 | 说明 |
|------|------|------|
| `musa/src/driver/mu_memory.cpp` | 103-259 | `imuapiPointerGetAttribute` 实现 |
| `musa/src/driver/mu_memory.cpp` | 679-681 | `muapiPointerGetAttribute` 转发 |
| `musa/src/driver/mu_memory.cpp` | 683-707 | `muapiPointerGetAttributes` 批量转发 |
| `musa/src/musa/core/memory.cpp` | 18-37 | `GetHostPointer` (映射主机访问) |
| `musa/src/musa/core/memory.cpp` | 39-51 | `SetAttribute` (syncMemOps) |
| `musa/src/musa/core/platform.h` | - | `MemoryTracker::FindRange` 接口 |
