# IPC Import 资源泄漏深入分析

> 专题文档：分析 `muIpcOpenMemHandle` / `muIpcCloseMemHandle` 路径中的资源泄漏风险。

---

## 概述

在 MUSA IPC Memory 的导入端（`muapiIpcOpenMemHandle_v2`），内存通过 `memoryTypeIpcImport` 类型创建。在关闭端（`muapiIpcCloseMemHandle`），存在以下潜在资源泄漏路径。

---

## 泄漏路径一：Synchronize 失败导致的内存泄漏

### 代码位置

`mu_memory.cpp:2140-2143`:

```cpp
status = pMemory->Synchronize();
if (MUSA_SUCCESS == status) {
    status = pMemory->GetContext()->DestroyMemory(pMemory);
}
// ★ Synchronize 失败 → DestroyMemory 被跳过
```

### 影响链

```
Synchronize() 失败
  │
  ├─ DestroyMemory() 未调用
  │     ├─ Context::DestroyMemory() 未执行
  │     │     ├─ CriticalBase::RemoveMemory() 未执行  ← Context 内存集合泄漏
  │     │     └─ MemoryTracker::UntrackMemory() 未执行 ← 全局 MemoryTracker 泄漏
  │     │
  │     └─ Memory 析构函数未执行
  │           ├─ m_pHalMemory->Destroy() 未调用        ← HAL 层 GPU 内存泄漏
  │           └─ HAL 驱动层 m3dBo 引用计数未释放       ← KMD 端 buffer 泄漏
  │
  └─ 重复调用 Close 同一 dptr → 重复查找同一已注册内存 → 重复 Synchronize 失败
     → 泄漏持续累积，无法自愈
```

### 根本原因

`Synchronize()` 失败时，错误处理将 `status` 保留为错误码并直接返回，跳过了资源释放逻辑。这是一个经典的 **"early return 遗漏清理"** 模式问题。

### 触发场景

| 场景 | 是否可能触发 |
|------|-------------|
| GPU 处于错误状态（如 ECC Error） | ✅ 可能 |
| GPU hang / GPU reset 进行中 | ✅ 可能 |
| 驱动内部状态不一致 | ✅ 可能 |
| 正常操作流程 | ❌ 不会 |

---

## 泄漏路径二：MapToPeers 未撤销

### 代码位置

`context.cpp:483-558` — `MapToPeers()` 在 IPC Import 创建时自动调用。

### 问题

```
CreateMemory (IPC Import)
  ├─ Init() → IpcImportAlloc()       // HAL 层导入，创建 GPU 映射
  ├─ MapToPeers()                     // 在所有可达设备上建立 Peer 映射
  │     └─ Hal::OpenPeerMemory()      // 每个 Peer 设备上调用
  └─ ...

// CloseMemHandle 时:
DestroyMemory
  ├─ RemoveMemory() ← 从 Context 集合移除
  └─ UntrackMemory() ← 从 MemoryTracker 移除
  // ★ 未调用 UnmapFromPeers() 或类似逆向操作
```

### 影响

1. **Peer 设备上的 VA 映射泄漏**: 每个 Peer 设备上通过 `OpenPeerMemory` 建立的映射未被撤销
2. **Peer 设备内存引用计数**: HAL 层的 Peer 内存引用计数未递减，可能导致 Peer 设备无法被正确卸载/重置
3. **长期运行的进程**: 如果反复导入/关闭（即使每次 Close 都成功），中间步骤的 Peer 映射可能残留

> **注意**: 此路径在 `Synchronize` 成功 + `DestroyMemory` 被调用的场景下也可能存在，因为 `DestroyMemory` → `RemoveMemory` 仅移除集合条目，不保证撤销 Peer 映射。

---

## 泄漏路径三：Event IPC Handle 泄漏

### 代码位置

`mu_event.cpp:211-236`:

```cpp
MUresult muapiIpcGetEventHandle(MUipcEventHandle *pHandle, MUevent event) {
    // ...
    status = pEvent->ExportIpcHandle(pHandle);
    // ...
}

MUresult muapiIpcOpenEventHandle(MUevent *phEvent, MUipcEventHandle handle) {
    // ...
    status = pContext->CreateEvent(&pEvent, MU_EVENT_INTERPROCESS | MU_EVENT_DISABLE_TIMING);
    if (status == MUSA_SUCCESS) {
        status = pEvent->ImportIpcHandle(handle);
        if (status != MUSA_SUCCESS) {
            pContext->DestroyEvent(pEvent);  // ✅ 此处处理正确
        }
    }
}
```

### 对比

Event 的 IPC 导入在 `ImportIpcHandle` 失败时正确销毁了已创建的对象。但 Memory 的 `IpcImportAlloc` 失败路径未被类似保护。

---

## 泄漏路径四：Sub-Allocated IPC 内存的 Pool 归还失败

### 代码路径

```
IpcImportAlloc()
  ├─ 可能走 GeneralAlloc 路径
  │     └─ SubAllocatable=YES → MemMgr::Allocate() → Pool->FullAllocate()
  │           └─ 从 Pool 中切分 chunk
  │
  // CloseMemHandle 时:
  DestroyMemory()
  ├─ 检查: GetProps() & memoryPropertySubAllocatable
  │     ├─ YES → MemMgr::Free() → Pool::Free() ← 归还 chunk 到 Pool
  │     └─ NO  → 直接 Destroy()
  //
  // 问题: 如果内存是从 IPC 导入的 Pool 子分配，但 SubAllocatable
  // 属性在导入时未被正确设置，可能导致释放路径错误
```

### 触发条件

如果 `IpcImportAlloc` 创建的内存被标记为 `SubAllocatable`，但底层实际并非从 Pool 分配（例如 HAL 层直接创建了完整 buffer），则释放时尝试归还 Pool 将导致 **双重释放** 或 **非法内存访问**。

---

## 泄漏汇总表

| 泄漏路径 | 泄漏类型 | 触发条件 | 严重程度 |
|---------|---------|---------|---------|
| Synchronize 失败跳过 Destroy | Memory + HAL Buffer | GPU 错误状态 | **高** — 不可自愈 |
| MapToPeers 未撤销 | VA 映射 + Peer 引用 | 所有 IPC Import | **中** — 随导入次数累积 |
| Event IPC 失败处理正确 | N/A | N/A | ✅ 已处理 |
| SubAllocation Pool 归还错误 | 潜在双重释放 | IPC 内存被错误标记 | **高** — 导致 crash |

---

## 建议修复方案

### 修复一：CloseMemHandle 确保资源释放

```cpp
MUresult muapiIpcCloseMemHandle(MUdeviceptr dptr) {
    // ... 参数校验 ...
    Musa::IMemory* pMemory = Platform::Get().GetMemoryByDevicePointer(dptr)->get();
    if (!pMemory || pMemory->GetType() != memoryTypeIpcImport) {
        return MUSA_ERROR_INVALID_VALUE;
    }

    MUresult status = pMemory->Synchronize();
    // ★ 无论 Synchronize 是否成功，都执行清理
    status = pMemory->GetContext()->DestroyMemory(pMemory);
    // 保留 Synchronize 的错误码，或按需覆盖
    return status;
}
```

### 修复二：DestroyMemory 中撤销 Peer 映射

```cpp
// Memory::Destroy() 或 Context::DestroyMemory() 中:
if (m_pHalMemory) {
    // 撤销所有 Peer 映射
    for (auto& peer : m_PeerMemories) {
        peer->ClosePeerMemory();
    }
    m_PeerMemories.clear();
    Destroy();  // 释放 HAL 层资源
}
```

### 修复三：IPC Import 明确设置 SubAllocatable 属性

```cpp
case memoryTypeIpcImport: {
    // IPC 导入的内存不应使用 SubAllocation
    createInfo.flags &= ~Hal::memoryPropertySubAllocatable;
    status = IpcImportAlloc(*handle, createInfo.ipcImport.flags);
}
```

---

## 测试验证建议

1. **GPU Error 注入**: 使用 fault injection 工具模拟 GPU hang，验证 CloseMemHandle 后资源是否完全释放
2. **Valgrind/ASan**: 运行 IPC 导入/关闭循环，监控内存增长
3. **Peer 映射计数**: 统计 Peer 设备上 `OpenPeerMemory`/`ClosePeerMemory` 调用次数是否平衡
4. **Pool 状态检查**: CloseMemHandle 后检查对应 Pool 的 chunk 计数是否恢复

---

## 交叉引用

- `22_IPC_Memory_API.md` — IPC Memory API 完整流程（含上述问题的简要标注）
- `09_usage_patterns.md` — IPC 在内部使用场景中的调用模式
- `18_Stream_subsystem.md` — Synchronize 的 Stream 等待机制
- `19_VA_PA_binding.md` — MapToPeers 的 VA→PA 绑定机制