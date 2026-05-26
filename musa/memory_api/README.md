# MUSA Memory API 源码级分析 — 文档索引

> 基于 MUSA Runtime 源码 (`linux-ddk/musa/`) 的深度调用链分析

## 文档列表

| # | 文件 | API/主题 | 核心源码 |
|---|------|---------|---------|
| 01 | `01_muMemAlloc_v2.md` | `muMemAlloc` / `muMemAlloc_v2` | `mu_memory.cpp:265`, `core/memory.cpp:462`, `memMgr.cpp:81`, `memoryPool.cpp:82`, `hal/memory.cpp:366` |
| 02 | `02_muMemFree_v2.md` | `muMemFree` / `muMemFree_v2` | `mu_memory.cpp:709`, `core/memory.cpp:358`, `core/stream.cpp:601`, `core/context.cpp:967` |
| 03 | `03_muMemHostAlloc.md` | `muMemHostAlloc` / `muMemAllocHost` | `mu_memory.cpp:59`, `core/memory.cpp:532` |
| 04 | `04_muMemHostRegister_v2.md` | `muMemHostRegister` | `mu_memory.cpp:611`, `core/memory.cpp:611`, `hal/memory.cpp:590` |
| 05 | `05_muMemcpyHtoD_v2.md` | `muMemcpy*` 全系列 (HtoD/DtoH/DtoD/Peer/2D/Array/Batch) | `mu_memory.cpp:13-1156`, `core/context.cpp:699`, `core/stream.cpp:663` |
| 06 | `06_muMemsetD32_v2.md` | `muMemsetD8/D16/D32` | `core/context.cpp:733`, `core/stream.cpp:721` |
| 07 | `07_muMemAllocAsync.md` | `muMemAllocAsync` / `muMemAllocFromPoolAsync` | `mu_memory.cpp:303`, `core/stream.cpp:519`, `core/memoryPool.cpp:ModifyAccess`, `command/pagingCommand.cpp` |
| 08 | `08_muMemFreeAsync.md` | `muMemFreeAsync` | `mu_memory.cpp:386`, `core/stream.cpp:601`, `core/memoryPool.cpp:DisableAccess`, `command/pagingCommand.cpp`, `command/callbackCommand.cpp` |
| 09 | `09_usage_patterns.md` | 内部调用者分析 (Graph/Peer/IPC/Export) | 多文件交叉引用 |
| 10 | `10_CreateMemory_and_MapToPeers.md` | `CreateMemory` + `MapToPeers` 统一入口 | `core/context.cpp:915`, `core/context.cpp:483` |
| 11 | `11_GeneralAlloc_deep_dive.md` | 通用设备内存分配深度拆解 | `core/memory.cpp:462`, `memMgr.cpp:81`, `memoryPool.cpp:82-413` |
| 12 | `12_DirectKMD_Allocation_flow.md` | 裸 KMD 分配 (Linux DRM + Windows WDDM2) | `hal/memory.cpp:366-836` |
| 13 | `13_MemoryPool_deep_dive.md` | MemoryPool 子分配算法 (4 个执行示例) | `hal/memoryPool.cpp` 全文件 |
| 14 | `14_Memory_API_source_deep_dive.md` | Core 层 9 种分配路径逐函数分析 | `core/memory.cpp` 全文件 |
| 15 | `memory_api_callflow_validation.md` | Memory API 调用链埋点验证 | `MUSA_DRIVER_CALLFLOW_DEBUG` 日志 |

## 三层内存分配器架构

```
┌─────────────────────────────────────────────┐
│  L3: Musa::MemoryPool (用户池)               │  ← muMemCreatePool / muMemAllocFromPoolAsync
│     内存池化管理, 用户可自定义                │
├─────────────────────────────────────────────┤
│  L2: Hal::IMemMgr + MemoryPool (系统池)      │  ← MemMgr::Allocate (memMgr.cpp:81)
│     按 {type, heap, property, viewCap} 查找  │  ← 首次按需创建, 后续复用
│     Sub-Allocation (O(1) 哈希桶查找)          │
├─────────────────────────────────────────────┤
│  L1: Hal::IMemory (裸 KMD 分配)              │  ← Hal::CreateMemory / IDevice::CreateGpuMemory
│     直接调用 KMD ioctl (mtgpuBoAlloc 等)      │  ← 每次至少 2 次 ioctl
└─────────────────────────────────────────────┘
```

## Key Concepts

### Sub-Allocation 性能对比
| 指标 | 裸 KMD (2 ioctl) | Sub-Allocation (Pool) |
|------|-------------------|----------------------|
| 单次 4KB 分配延迟 | ~0.2ms | ~0.0004ms |
| 1000 次 4KB 总耗时 | ~200ms | ~0.4ms |
| 加速比 | 1x | **~500x** |

### flags = 0x07 推导链
```
输入: Virtual(0x01) | DeviceMapped(0x02) | SubAllocatable(0x04)
         │
         ├── Core 自动追加: Physical(0x08) | SharedVA(0x10)
         │
         ├── 由 Virtual 推导:
         │     DeviceVisible(0x20) | HostVisible(0x40)
         │     HostCoherent(0x80) | DeviceWriteable(0x100)
         │     DeviceCached(0x200)
         │
         └── 由 DeviceMapped 推导:
               viewCapability: PeerAccessible | IpcExportable

最终 HAL property = 0x3FF (10 个属性位全开)
最终 viewCapability = 0x07 (Exportable | PeerAccessible | IpcExportable)
```

### 内存类型矩阵

| 类型 | 分配函数 | HAL type | Pool? | 物理位置 | CPU 可见 | GPU 可见 |
|------|---------|----------|-------|---------|---------|---------|
| General | muMemAlloc | memoryTypeAlloc (DeviceLocal) | YES | GPU 显存 | 否 (需映射) | 是 |
| PitchedGeneral | muMemAllocPitch | memoryTypeAlloc (DeviceLocal) | YES | GPU 显存 | 否 (需映射) | 是 |
| PinnedHost | muMemHostAlloc | memoryTypeAlloc (Host) | YES* | 系统内存 | 是 | 是 (映射后) |
| RegisteredPinned | muMemHostRegister | memoryTypeView (Locked) | NO | 系统内存 | 是 | 是 (映射后) |
| Managed | muMemAllocManaged | memoryTypeAlloc | NO | GPU/系统 | 是 | 是 |
| IPC Import | muMemImportShareableHandle | memoryTypeView (External) | NO | GPU 显存 | 否 | 是 |
| External | muMemImportShareableHandle | memoryTypeView (External) | NO | 外部 | 视类型 | 视类型 |
| Virtual | muMemAddressReserve | memoryTypeVirtual | NO | 仅 VA | — | — |

*PinnedHostAlloc 在 LargePage 不可用时会降级为 General heap

## 关键设计模式

1. **RAII**: Memory 析构自动释放 HAL 资源 (包括 Sub-Allocation 归还 Pool)
2. **双重释放防护**: 类型白名单 + offset==0 校验
3. **析构双路径**: `SubAllocatable` 属性决定走 `Pool::Free` 还是 `Destroy()`
4. **流式分配**: `muMemAllocAsync` 正常路径在 API 调用期间创建虚拟/物理内存并完成绑定，stream 中排序执行的是页表访问权限更新 `PagingCommand`；Graph Capture 路径记录图节点。
5. **自动 Peer 映射**: CreateMemory 后自动在所有已启用 Peer 的设备上建立映射
6. **惰性释放**: TrimPool 按阈值释放空闲 chunk, 减少 GPU 显存碎片

## 源码路径

- `musa/src/driver/mu_memory.cpp` — 所有 Driver 层 API 入口 (2949 行)
- `musa/src/musa/core/memory.cpp` — Core 层初始化函数 (827 行)
- `musa/src/musa/core/context.cpp` — CreateMemory/MapToPeers/DestroyMemory (1431+ 行)
- `musa/src/musa/core/stream.cpp` — AsyncMemAlloc/AsyncMemFree/Command 提交 (1266+ 行)
- `musa/src/musa/core/memoryPool.cpp` — `ModifyAccess` / `DisableAccess` 触发 `Stream::CmdPaging`
- `musa/src/musa/core/command/pagingCommand.cpp` — 页表访问权限更新命令
- `musa/src/musa/core/command/callbackCommand.cpp` — 异步释放回调命令
- `musa/src/hal/m3d/memory.cpp` — HAL 层内存初始化全路径 (838 行)
- `musa/src/hal/m3d/memoryPool.cpp` — Sub-Allocation 算法 (533 行)
- `musa/src/hal/m3d/memMgr.cpp` — Pool 管理/创建/查找 (237 行)
- `musa/src/hal/m3d/m3d/src/core/os/drm/mthreads/mtgpuMemory.cpp` — Linux DRM ioctl
- `musa/src/hal/m3d/m3d/src/core/os/wddm/wddmGpuMemory.cpp` — Windows WDDM2
