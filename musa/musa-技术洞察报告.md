# MUSA GPU Driver — 技术洞察报告

**分析日期：** 2026-05-26
**仓库路径：** `/home/shanfeng/workspace/linux-ddk/musa/`
**代码规模：** ~315K 行总计，~112K 行核心 C++ 源码
**定位：** MUSA GPU 用户态驱动程序（libmusa.so），CUDA Driver API 兼容层

---

## 一、项目定位

`linux-ddk/musa/` 是 MThreads MUSA GPU 栈的**核心用户态驱动**，在整个软件栈中处于承上启下的关键位置：

```
用户代码 / AI 框架
  │
  ├── MUSA-Runtime (libmusart.so)      ← musa* API (CUDA Runtime 兼容)
  │     └── ExportTable ──→ dynamic load
  │
  ▼
  MUSA Driver (libmusa.so)             ← 本仓库 (560 muapi* 函数)
  │  ┌─ API Layer    (src/driver/)      ← 公共入口，参数校验
  │  ├─ Core Layer   (src/musa/core/)  ← 设备/上下文/内存/流/图/模块 管理
  │  ├─ HAL Layer    (src/hal/)         ← 硬件抽象接口
  │  └─ M3D Layer    (src/hal/m3d/)    ← mtgpu GPU 具体实现
  │       └── ioctl ──→
  ▼
  gr-kmd (Kernel Mode Driver)           ← GPU 硬件交互
```

**类比**：MUSA Driver ≈ CUDA Driver API（`cu*` 函数），负责所有实际 GPU 操作。

---

## 二、整体架构

```
╔═══════════════════════════════════════════════════════════════════════╗
║  MUSA Driver (linux-ddk/musa/) — ~315K total, ~112K src C++         ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  ┌─────────────────────────────────────────────────────────────┐     ║
║  │ [1] API Layer  —  src/driver/  (27 .cpp, ~40K lines)        │     ║
║  │                                                              │     ║
║  │  560 muapi* 公共函数入口，按领域分文件:                          │     ║
║  │                                                              │     ║
║  │  muapi.h              (类型安全的 API 声明)                    │     ║
║  │  mu_wrappers_generated.cpp (22,628 行，自动生成包装器)         │     ║
║  │  mu_memory.cpp  2,949  │ mu_graph.cpp   3,329  ← 最多行数    │     ║
║  │  mu_stream.cpp    951  │ mu_context.cpp   656                │     ║
║  │  mu_module.cpp    563  │ mu_mempool.cpp   563                │     ║
║  │  mu_tensor.cpp    533  │ mu_vmm.cpp       476                │     ║
║  │  mu_greencontext  468  │ mu_device.cpp    374                │     ║
║  │  mu_library.cpp   366  │ mu_event.cpp     282                │     ║
║  │  mu_gfxinterop    270  │ mu_coredump      253                │     ║
║  │  mu_texture.cpp   232  │ mu_external      234                │     ║
║  │  mu_raytracing    222  │ mu_occupancy     216                │     ║
║  │  mu_notificatio   142  │ mu_peer.cpp      100                │     ║
║  │  mu_oglinterop    102  │ mu_error.cpp     724                │     ║
║  │  mu_entry.cpp   1,958  │ mu_deprecated    703                │     ║
║  │  mu_profiler.cpp   19  │ mu_log.cpp        24                │     ║
║  └─────────────────────────────────────────────────────────────┘     ║
║                              │                                        ║
║                              ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────┐     ║
║  │ [2] Core Layer  —  src/musa/core/  (~25 .cpp, ~12K lines)   │     ║
║  │                                                              │     ║
║  │  设备/上下文/内存/流/纹理/模块/图 — 核心业务逻辑               │     ║
║  │                                                              │     ║
║  │  context.cpp  2,631  │ stream.cpp    1,802                   │     ║
║  │  device.cpp   1,270  │ texture.cpp     949                   │     ║
║  │  memory.cpp     827  │ platform.cpp    622                   │     ║
║  │  memoryPool.cpp 600  │ library.cpp     516                   │     ║
║  │  graph.cpp      486  │ symbol.cpp      458                   │     ║
║  │  module.cpp     416  │ event.cpp       311                   │     ║
║  │  platformSett   559  │ logManager      306                   │     ║
║  │  printManager   339  │ surface.cpp     251                   │     ║
║  │  array.cpp      722  │ memoryTrack     187                   │     ║
║  │  lib.cpp         31  │ greenContext     43                   │     ║
║  │  graphicsRes    200  │ externalMem      99                   │     ║
║  │  externalSema    59  │ userObject       22                   │     ║
║  └─────────────────────────────────────────────────────────────┘     ║
║                              │                                        ║
║                              ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────┐     ║
║  │ [3] HAL Layer  —  src/hal/  (~30 .h 接口定义, ~3.4K lines)  │     ║
║  │                                                              │     ║
║  │  硬件抽象接口 (核心类型):                                       │     ║
║  │  halDevice.h    366  │ halMemory.h    403                    │     ║
║  │  halCmdBuffer.h 762  │ halQueue.h     335                    │     ║
║  │  halFormat.h    298  │ halImage.h     180                    │     ║
║  │  halKernel.h    240  │ halLibDeserialize 220                 │     ║
║  │  halMemoryPool  103  │ halMemMgr.h     44                    │     ║
║  │  halSemaphore   159  │ halPlatform    151                    │     ║
║  │  halSampler      38  │ halEvent        25                    │     ║
║  │  halTypes       368  │ halBase          12                    │     ║
║  └─────────────────────────────────────────────────────────────┘     ║
║                              │                                        ║
║                              ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────┐     ║
║  │ [4] M3D Layer  —  src/hal/m3d/  (mtgpu GPU 实现, ~8K lines)  │     ║
║  │                                                              │     ║
║  │  具体 GPU 硬件实现:                                            │     ║
║  │  device.cpp   1,329  │ cmdBuffer.cpp 1,139                   │     ║
║  │  memory.cpp     838  │ image.cpp       631                   │     ║
║  │  memoryPool.cpp 533  │ queue.cpp       446                   │     ║
║  │  library.cpp    463  │ memMgr.cpp(core)237                   │     ║
║  │  kernel.cpp     166  │ memMgr.cpp(hal) 128                   │     ║
║  │  fence.cpp       51  │ cmdPool.cpp     106                   │     ║
║  │                                                              │     ║
║  │  操作系统后端:                                                 │     ║
║  │  src/hal/m3d/m3d/src/core/os/drm/mthreads/  ← Linux DRM      │     ║
║  │  src/hal/m3d/m3d/src/core/os/wddm/          ← Windows WDDM2  │     ║
║  └─────────────────────────────────────────────────────────────┘     ║
║                                                                        ║
╚═══════════════════════════════════════════════════════════════════════╝
```

---

## 三、API 面全景：560 muapi* 函数按领域分布

| 领域 | 文件 | 行数 | 典型 API |
|------|------|------|---------|
| **Memory** | mu_memory.cpp | 2,949 | `muMemAlloc_v2`, `muMemFree_v2`, `muMemcpy`, `muMemAllocAsync` |
| **Graph** | mu_graph.cpp | 3,329 | `muGraphCreate`, `muGraphInstantiate`, `muGraphLaunch`, 80+ |
| **Entry/Init** | mu_entry.cpp | 1,958 | `muInit`, `muGetExportTable`, driver/runtime 入口 |
| **Stream** | mu_stream.cpp | 951 | `muStreamCreate`, `muStreamSynchronize`, `muStreamBeginCapture` |
| **Context** | mu_context.cpp | 656 | `muCtxCreate`, `muCtxSetCurrent`, `muCtxSynchronize` |
| **Module** | mu_module.cpp | 563 | `muModuleLoad`, `muModuleGetFunction`, `muModuleLoadFatBinary` |
| **MemPool** | mu_mempool.cpp | 563 | `muMemPoolCreate`, `muMemPoolTrimTo`, `muMemAllocFromPoolAsync` |
| **Error/Debug** | mu_error.cpp | 724 | `muGetErrorString`, `muGetErrorName`, coredump |
| **Deprecated** | mu_deprecated.cpp | 703 | 旧 API 重定向 |
| **Tensor** | mu_tensor.cpp | 533 | `muTensorMapEncode`, Tensor Core 指令 |
| **VMM (Virtual Memory)** | mu_vmm.cpp | 476 | `muMemAddressReserve`, `muMemMap`, `muMemUnmap` |
| **Green Context** | mu_greencontext.cpp | 468 | 低开销 context 切换 |
| **Device** | mu_device.cpp | 374 | `muDeviceGetCount`, `muDeviceGetAttribute`, `muDevicePrimaryCtxRetain` |
| **Library** | mu_library.cpp | 366 | `muLibraryLoadData`, `muLibraryGetKernel` |
| **Event** | mu_event.cpp | 282 | `muEventCreate`, `muEventRecord`, `muEventSynchronize` |
| **Graphics Interop** | mu_gfxinterop.cpp | 270 | `muGraphicsMapResources`, `muGraphicsUnmapResources` |
| **Raytracing** | mu_raytracing.cpp | 222 | OptiX 兼容 API |
| **Occupancy** | mu_occupancy.cpp | 216 | `muOccupancyMaxActiveBlocksPerMultiprocessor` |
| **External** | mu_external.cpp | 234 | `muImportExternalMemory`, `muImportExternalSemaphore` |
| **Texture** | mu_texture.cpp | 232 | `muTexRefCreate`, `muTexRefSetAddress`, `muArrayCreate` |
| **Notification** | mu_notification.cpp | 142 | 异步通知注册 |
| **Peer Access** | mu_peer.cpp | 100 | `muDeviceEnablePeerAccess`, `muDeviceCanAccessPeer` |
| **OpenGL Interop** | mu_oglinterop.cpp | 102 | `muGraphicsGLRegisterBuffer` |
| **Profiler** | mu_profiler.cpp | 19 | `muProfilerStart`, `muProfilerStop`（桩） |
| **Log** | mu_log.cpp | 24 | 日志控制 |
| **Coredump** | mu_coredump.cpp | 253 | GPU 异常状态 dump |

---

## 四、内存分配器三层架构（核心子系统）

MUSA 内存管理是整个驱动中**代码量最大、设计最复杂**的子系统。

```
┌──────────────────────────────────────────────────────────┐
│  L3: 用户 MemoryPool                                      │
│    API: muMemCreatePool → muMemAllocFromPoolAsync        │
│    用户可自定义池属性，池隔离，TrimTo 惰性释放               │
├──────────────────────────────────────────────────────────┤
│  L2: 系统 MemMgr + MemoryPool (Sub-Allocation 引擎)      │
│    memMgr.cpp: 按 {type,heap,property,viewCap} O(1) 查池  │
│    memoryPool.cpp: 哈希桶分桶 + First-Fit + merge          │
│    分配 ~0.0004ms (vs 裸 KMD ~0.2ms, ~500x 加速)          │
├──────────────────────────────────────────────────────────┤
│  L1: 裸 KMD 分配 (Hal::CreateMemory)                      │
│    Linux:  ioctl(mtgpuBoAlloc) → GEM obj → GPU MMU map    │
│    Windows: D3DKMTAllocateMemory → WDDM2                  │
└──────────────────────────────────────────────────────────┘
```

### 9 种内存分配路径

| # | 函数 | API | Hal 类型 | 走 Pool? |
|---|------|-----|----------|---------|
| 1 | GeneralAlloc | muMemAlloc | DeviceLocal | ✅ (SubAllocatable) |
| 2 | PitchedGeneralAlloc | muMemAllocPitch | DeviceLocal | ✅ (固定) |
| 3 | PinnedHostAlloc | muMemHostAlloc | Host | ✅ (可降级) |
| 4 | PinnedHostRegister | muMemHostRegister | View/Locked | ❌ |
| 5 | PinnedHostRegister(MMIO) | muMemHostRegister(IOMEMORY) | External(MMIO) | ❌ |
| 6 | ManagedAlloc | muMemAllocManaged | DeviceLocal/Host | ❌ |
| 7 | IpcImportAlloc | muMemImportShareableHandle | External(Global) | ❌ |
| 8 | ExternalAlloc | 外部导入 | External(Fabric/DmaBuf) | ❌ |
| 9 | VirtualAlloc | muMemAddressReserve | Virtual(仅VA) | ❌ |

---

## 五、关键设计模式

| 模式 | 说明 | 应用场景 |
|------|------|---------|
| **RAII 生命周期** | Memory 析构自动归还 Pool 或 Free KMD | 所有内存类型 |
| **析构双路径** | SubAllocatable 位决定 Pool::Free vs Destroy | 内存释放 |
| **双重释放防护** | 类型白名单 + offset==0 校验 | 防 Double Free |
| **Auto Peer 映射** | CreateMemory 后自动建立跨设备映射 | Peer Access |
| **Stream 命令化** | Async 分配编码为 Stream Command | Graph 捕获安全 |
| **惰性释放** | TrimTo 按空闲率阈值归还 chunk 给 KMD | 减少碎片 |
| **原子初始化** | InitPlatform 用 atomic state machine 保证线程安全 | 首次调用 |
| **导出表模式** | muapiGetExportTable 填充函数指针表 | Runtime↔Driver 解耦 |

---

## 六、源码目录速查

| 目录 | 行数 | 说明 |
|------|------|------|
| `src/driver/` | ~40K | API 入口层，27 个 .cpp，mu_wrappers_generated.cpp(22.6K行自动生成) |
| `src/musa/core/` | ~12K | Core 业务逻辑层，20+ .cpp/.h |
| `src/hal/` | ~3.4K | HAL 接口定义，30+ .h |
| `src/hal/m3d/` | ~8K | mtgpu GPU 实现（device/cmdBuffer/memory/image/kernel/queue） |
| `src/musa/` | 30 .h | Core 层 C++ 内部类型（Memory/Stream/Graph/Device/Context...） |
| `src/hal/m3d/m3d/src/core/os/` | — | OS 后端：drm/ (Linux) + wddm/ (Windows) |

---

## 七、构建与平台支持

| 项目 | 值 |
|------|-----|
| **语言** | C++17 |
| **构建** | CMake (通过 ddk-build2.0 Docker) |
| **产物** | libmusa.so |
| **平台** | Linux x86_64 (DRM), Windows x86_64 (WDDM2), Linux aarch64 |
| **GPU 架构** | qy1/qy2/ph1/ph1s/hg/ls/hs (mp_21~mp_43) |

---

## 八、与 MUSA-Runtime 的关系

| 维度 | MUSA-Runtime (libmusart.so) | MUSA Driver (libmusa.so) |
|------|---------------------------|-------------------------|
| **定位** | CUDA Runtime 兼容 | CUDA Driver 兼容 |
| **API** | `musa*` (337 导出) | `mu*` (560 入口) |
| **职责** | 封装 + 转发 | 实际 GPU 操作 |
| **加载** | dlopen libmusa.so → ExportTable | 直接加载 |
| **代码量** | ~27K src C++ | ~40K src C++ (driver/) + ~12K (core/) + ~8K (hal/m3d/) |
| **总代码量** | ~219K (含 UT/头文件) | ~315K (含 build/m3d/OS 后端) |

---

## 九、核心指标总结

| 指标 | 数值 |
|------|------|
| 总代码量 | ~315,000 行 |
| 核心 C++ 源码 | ~112,000 行 |
| API 入口数 | 560 muapi* 函数 (168 声明) |
| Driver 层文件数 | 27 .cpp |
| Core 层文件数 | ~25 .cpp |
| HAL 接口数 | ~30 .h |
| 最大文件 | mu_wrappers_generated.cpp (22,628 行) |
| 最大领域文件 | mu_graph.cpp (3,329 行), mu_memory.cpp (2,949 行) |
| 构建产物 | libmusa.so |
| 平台 | Linux DRM + Windows WDDM2 |
