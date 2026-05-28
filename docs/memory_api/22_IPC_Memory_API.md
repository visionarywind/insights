# IPC Memory API — muIpcGetMemHandle / muIpcOpenMemHandle / muIpcCloseMemHandle

> MUSA IPC Memory API 对应 CUDA IPC API（`cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle` / `cudaIpcCloseMemHandle`），用于跨进程共享设备内存。

---

## 功能

| API | 功能 |
|-----|------|
| `muapiIpcGetMemHandle` | 导出已分配设备内存的 IPC 句柄，供其他进程使用 |
| `muapiIpcOpenMemHandle` | 打开 IPC 句柄，在当前进程映射共享内存（v2 版本） |
| `muapiIpcOpenMemHandle` (v1) | v1 版本，直接委托给 v2 |
| `muapiIpcCloseMemHandle` | 关闭 IPC 导入的内存映射，释放资源 |

---

## 完整调用链

### GetMemHandle (导出)

```
User Code
  │
  └─ muapiIpcGetMemHandle(&handle, dptr)
        │
        ├─ InitPlatform()
        ├─ pHandle==nullptr → INVALID_VALUE
        ├─ dptr==0 → INVALID_VALUE
        ├─ TlsCtxTop() → pContext
        ├─ pContext->GetMemoryByDevicePointer(dptr) → pMemory
        │     └─ MemoryTracker::FindRange(dptr) 全局查找
        ├─ pMemory==nullptr → INVALID_VALUE
        ├─ pMemory->GetType() != General && != PitchedGeneral → INVALID_VALUE
        │     └─ ★ 只有 General/PitchedGeneral 类型允许导出
        └─ pMemory->ExportIpcHandle(pHandle)
              │
              └─ HAL 层导出底层 buffer 的 IPC handle
```

### OpenMemHandle (导入)

```
User Code
  │
  └─ muapiIpcOpenMemHandle_v2(&dptr, handle, flags)
        │
        ├─ InitPlatform()
        ├─ pdptr==nullptr → INVALID_VALUE
        ├─ flags 白名单校验:
        │     LAZY_ENABLE_PEER_ACCESS
        │     LAZY_UNIDIR_INTERLEAVE_PEER_ACCESS
        │     LAZY_BIDIR_INTERLEAVE_PEER_ACCESS
        │     LAZY_HETERO_INTERLEAVE_PEER_ACCESS
        │     LAZY_HETERO_BIDIR_INTERLEAVE_PEER_ACCESS
        │     LAZY_HALF_BIDIR_INTERLEAVE_PEER_ACCESS
        │     LAZY_PCIE_PEER_ACCESS
        │   其他值 → INVALID_VALUE
        ├─ TlsCtxTop() → pContext
        ├─ 构造 MemoryCreateInfo:
        │     type = memoryTypeIpcImport
        │     ipcImport.handle = handle
        │     ipcImport.flags = flags
        └─ pContext->CreateMemory(&pMemory, createInfo)
              │
              ├─ Context::CreateMemory()
              │     ├─ Stream Capture 检查
              │     ├─ new Memory(this)->Init(createInfo)
              │     │     └─ Memory::Init() → case memoryTypeIpcImport:
              │     │          └─ IpcImportAlloc(handle, flags)
              │     │               └─ HAL 层导入 IPC handle，创建设备侧映射
              │     ├─ MapToPeers()  ← ★ 自动 Peer 映射
              │     │     └─ IPC Import 类型: mapFlags = flags - 1
              │     ├─ AddMemory(pMemory)
              │     └─ TrackMemory(memory)  ← 注册到 MemoryTracker
              └─ *pdptr = pMemory->GetDevicePointer()
```

### CloseMemHandle (关闭)

```
User Code
  │
  └─ muapiIpcCloseMemHandle(dptr)
        │
        ├─ InitPlatform()
        ├─ dptr==0 → INVALID_VALUE
        ├─ GetMemoryByDevicePointer(dptr) → pMemory
        │     └─ MemoryTracker 查找
        ├─ pMemory==nullptr → INVALID_VALUE
        ├─ pMemory->GetType() != memoryTypeIpcImport → INVALID_VALUE
        ├─ pMemory->Synchronize()    ← ★ 必须先同步
        └─ pMemory->GetContext()->DestroyMemory(pMemory)
              │
              └─ Context::DestroyMemory()
                    ├─ CriticalBase::RemoveMemory(pMem)
                    └─ MemoryTracker::UntrackMemory(pMemory)
```

---

## 时序图

### 导出 → 导入 流程

```
进程 A (Exporter)                       进程 B (Importer)
     │                                        │
     │ muMemAlloc(&dptr, size)               │
     │──────────────────────────────────────>│
     │ 分配内存，注册到 MemoryTracker          │
     │                                        │
     │ muIpcGetMemHandle(&handle, dptr)       │
     │──────────────────────────────────────>│
     │ 导出 IPC handle                        │
     │◄──────────────────────────────────────│
     │  handle 跨进程传递 (OS 机制)             │
     │                                        │
     │                              muIpcOpenMemHandle(&dptr, handle, flags)
     │                                        │─────────────────────────>│
     │                                        │ IpcImportAlloc()         │
     │                                        │ MapToPeers()             │
     │                                        │ 注册到 MemoryTracker      │
     │◄──────────────────────────────────────│                          │
     │                              返回 dptr                           │
     │                                        │                          │
     │          (两端可并发访问同一物理内存)     │                          │
     │                                        │                          │
     │                              muIpcCloseMemHandle(dptr)           │
     │                                        │─ Synchronize()           │
     │                                        │─ DestroyMemory()         │
     │                                        │  (资源释放)               │
     │◄───────────────────────────────────────│                          │
```

---

## 关键代码路径

### 导出端校验 (`mu_memory.cpp:2062-2067`)

```cpp
Musa::IMemory* pMemory = pContext->GetMemoryByDevicePointer(dptr)->get();
if (!pMemory) {
    status = MUSA_ERROR_INVALID_VALUE;
} else if (pMemory->GetType() != Musa::memoryTypeGeneral &&
           pMemory->GetType() != Musa::memoryTypePitchedGeneral) {
    tprintf(LOG_MEM, "only support to export the memory allocated by musaMalloc\n");
    status = MUSA_ERROR_INVALID_VALUE;
}
```

> 只有 `memoryTypeGeneral` 和 `memoryTypePitchedGeneral` 允许导出。其他类型（Managed、PinnedHost、Virtual、External、IpcImport、Prealloc）均拒绝。

### 导入端 flags 校验 (`mu_memory.cpp:2086-2099`)

```cpp
switch(flags) {
    case MU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS:
    case MU_IPC_MEM_LAZY_UNIDIR_INTERLEAVE_PEER_ACCESS:
    case MU_IPC_MEM_LAZY_BIDIR_INTERLEAVE_PEER_ACCESS:
    case MU_IPC_MEM_LAZY_HETERO_INTERLEAVE_PEER_ACCESS:
    case MU_IPC_MEM_LAZY_HETERO_BIDIR_INTERLEAVE_PEER_ACCESS:
    case MU_IPC_MEM_LAZY_HALF_BIDIR_INTERLEAVE_PEER_ACCESS:
    case MU_IPC_MEM_LAZY_PCIE_PEER_ACCESS:
        break;
    default:
        status = MUSA_ERROR_INVALID_VALUE;
        break;
}
```

> 所有合法 flags 均包含 `LAZY` 前缀，表示延迟映射语义。

---

## Peer 映射规则 (导入端)

`MapToPeers` 中 IPC Import 类型的映射规则:

```
mapFlags = memory->GetFlags() - 1  // 即 flags 枚举值 - 1
```

| 导入 flags | mapFlags 值 | 含义 |
|-----------|------------|------|
| `LAZY_ENABLE_PEER_ACCESS`(0x1) | 0x0000 | 启用所有 Peer，支持 RC/Atomics |
| `LAZY_UNIDIR_INTERLEAVE_PEER_ACCESS`(0x3) | 0x0002 | 单向交错 |
| `LAZY_BIDIR_INTERLEAVE_PEER_ACCESS`(0x5) | 0x0004 | 双向交错 |
| `LAZY_HETERO_INTERLEAVE_PEER_ACCESS`(0x7) | 0x0006 | 异构交错 |
| `LAZY_HETERO_BIDIR_INTERLEAVE_PEER_ACCESS`(0x9) | 0x0008 | 异构双向交错 |
| `LAZY_HALF_BIDIR_INTERLEAVE_PEER_ACCESS`(0xB) | 0x000A | 半双向交错 |
| `LAZY_PCIE_PEER_ACCESS`(0xD) | 0x000C | 仅 PCIe 映射 |

---

## 潜在资源泄漏问题

### 问题描述

当 `muapiIpcCloseMemHandle` 被调用时，如果 `Synchronize()` 失败（返回非成功状态），代码会跳过 `DestroyMemory()` 调用，**但 `pMemory` 对象已经在 MemoryTracker 中注册**，导致：

1. **MemoryTracker 泄漏**: `TrackMemory` 已注册但 `UntrackMemory` 未执行
2. **HAL 层资源泄漏**: `IpcImportAlloc` 创建的底层 HAL 内存对象未被释放
3. **设备 VA 映射泄漏**: `MapToPeers` 已建立但 `UnmapFromPeers` 未调用

```cpp
// mu_memory.cpp:2140-2143
status = pMemory->Synchronize();
if (MUSA_SUCCESS == status) {
    status = pMemory->GetContext()->DestroyMemory(pMemory);  // ★ 仅 Synchronize 成功时执行
}
// ★ Synchronize 失败时: pMemory 泄漏
```

### 建议修复

应确保即使 `Synchronize()` 失败，`DestroyMemory` 仍被调用（或至少释放 HAL 层资源），避免资源泄漏累积。

---

## 设计要点

1. **单向导出**: 只有原始分配者（进程 A）能导出 handle，导入者（进程 B）不能反向导出同一内存
2. **Lazy 语义**: flags 均含 `LAZY` 前缀，表示映射延迟建立（首次访问时触发），非立即映射
3. **类型限制**: 仅 `General` / `PitchedGeneral` 类型可导出，Managed 内存、Pinned Host 内存等均不支持
4. **自动 Peer 映射**: 导入后自动在所有可达设备上建立 Peer 映射，与 `CreateMemory` 行为一致
5. **同步要求**: Close 时强制 `Synchronize()`，确保所有未完成操作完成后才释放资源
6. **IPC Event 联动**: `IpcGetEventHandle` / `IpcOpenEventHandle` 提供跨进程事件同步机制，与 IPC Memory 配合使用