# MUSA 文档索引

## 📂 顶层文档

| 文件 | 内容 | 适用读者 |
|------|------|---------|
| [memory_analysis.md](../memory_analysis.md) | 内存子系统架构总览 + 文件索引 + API 分类 | 快速了解整体架构 |
| [memory_api_deep_analysis.md](../memory_api_deep_analysis.md) | 8 个核心内存 API 逐行流程分析 + 时序图 | 需要理解具体 API 执行流程 |
| [muMemAlloc_vs_muMemAllocAsync.md](../muMemAlloc_vs_muMemAllocAsync.md) | 同步 vs 异步分配的深度对比 + 附录 GPU 基础 | 理解 alloc/allocAsync 的本质差异 |
| [pooling_analysis.md](../pooling_analysis.md) | 15 种池化/分配器完整架构分析 | 需要了解内存池、命令池等 |
| [stream_command_analysis.md](../stream_command_analysis.md) | Stream + Command 子系统架构 + 双线程模型 | 需要理解异步执行引擎 |
| [**decision_logic.md**](../decision_logic.md) | **14 个关键决策分支全景** — 每个 if/switch 的判断条件 + 分支行为 | **理解流程走向** |

## 📂 memory_api/ 分文件索引

> 每个文件包含: 功能概述 → 完整调用链 → 时序图 → 关键代码路径 → 设计要点  
> **新手建议先阅读顶层的 [memory_analysis.md](../memory_analysis.md) 和 [muMemAlloc_vs_muMemAllocAsync.md](../muMemAlloc_vs_muMemAllocAsync.md) 了解基础概念**

| # | 文件 | API | 功能 | 对应 CUDA |
|---|------|-----|------|-----------|
| 1 | [01_muMemAlloc_v2.md](01_muMemAlloc_v2.md) | `muMemAlloc_v2` | 设备通用内存分配（六层逐行拆解 + 3 种时序图） | `cudaMalloc` |
| 2 | [02_muMemFree_v2.md](02_muMemFree_v2.md) | `muMemFree_v2` | 设备内存释放 | `cudaFree` |
| 3 | [03_muMemHostAlloc.md](03_muMemHostAlloc.md) | `muMemHostAlloc` | 主机页锁定内存分配 | `cudaHostAlloc` |
| 4 | [04_muMemHostRegister_v2.md](04_muMemHostRegister_v2.md) | `muMemHostRegister_v2` | 用户指针注册为页锁定 | `cudaHostRegister` |
| 5 | [05_muMemcpyHtoD_v2.md](05_muMemcpyHtoD_v2.md) | `muMemcpyHtoD_v2` | 主机→设备内存拷贝 | `cudaMemcpyHostToDevice` |
| 6 | [06_muMemsetD32_v2.md](06_muMemsetD32_v2.md) | `muMemsetD32_v2` | GPU 内存填充 | `cudaMemsetD32` |
| 7 | [07_muMemAllocAsync.md](07_muMemAllocAsync.md) | `muMemAllocAsync` | 异步内存分配 | `cudaMallocAsync` |
| 8 | [08_muMemFreeAsync.md](08_muMemFreeAsync.md) | `muMemFreeAsync` | 异步内存释放 | `cudaFreeAsync` |
| 9 | [09_usage_patterns.md](09_usage_patterns.md) | 内部调用模式分析 | Graph/IPC/Peer/Export/RAII | — |
| 10 | [10_CreateMemory_and_MapToPeers.md](10_CreateMemory_and_MapToPeers.md) | `CreateMemory` / `MapToPeers` | 统一入口 + Peer 自动映射 | — |
| — | *GeneralAlloc 全链路* | **⏳ 待创建** | Sub-Allocation 池化分配（参见 pooling_analysis.md） | — |
| 12 | [12_DirectKMD_Allocation_flow.md](12_DirectKMD_Allocation_flow.md) | 裸 KMD 分配 | 9 种 Init 函数 + 死代码标注 | — |
| 13 | [13_MemoryPool_deep_dive.md](13_MemoryPool_deep_dive.md) | MemoryPool 算法 | SubAllocate/ChunkAllocate/TrimPool | — |
| 14 | [14_Memory_API_source_deep_dive.md](14_Memory_API_source_deep_dive.md) | 9 种 Core 初始化函数 | 逐函数源码分析 | — |
| 15 | [15_muMemGetInfo_and_QueryAPI.md](15_muMemGetInfo_and_QueryAPI.md) | `muMemGetInfo` / `muMemGetAddressRange` 等 | 内存查询与信息获取 | `cudaMemGetInfo` |
| 16 | [16_muPointerGetAttributes.md](16_muPointerGetAttributes.md) | `muPointerGetAttributes` | 通用指针属性批量查询 | `cuPointerGetAttributes` |
| 17 | [17_PeerAccess_and_CanAccessPeer.md](17_PeerAccess_and_CanAccessPeer.md) | `muDeviceCanAccessPeer` / `muCtxEnablePeerAccess` 等 | Peer Access 查询与启用 | `cudaDeviceCanAccessPeer` 等 |
| 18 | [18_Stream_subsystem.md](18_Stream_subsystem.md) | `muStreamCreate` / `muStreamSynchronize` / `muStreamBeginCapture` 等 | Stream 子系统完整实现机制 | `cudaStreamCreate` 等 |
| — | *VA↔PA 绑定机制* | **⏳ 待创建** | MemoryTracker + m_PhysTracker + HAL Peer 映射（参见 muMemAlloc_vs_muMemAllocAsync.md 附录） | — |
| 20 | [20_muMemAdvise_and_MemRangeAttribute.md](20_muMemAdvise_and_MemRangeAttribute.md) | `muMemAdvise` / `muMemRangeGetAttribute` 等 | 内存建议与范围属性查询 | `cudaMemAdvise` 等 |
| 21 | [21_Event_API.md](21_Event_API.md) | `muEventCreate` / `Record` / `Synchronize` / `ElapsedTime` / `Destroy` | Event 创建、记录、同步、计时 | `cudaEventCreate` 等 |
| 22 | [22_IPC_Memory_API.md](22_IPC_Memory_API.md) | `muIpcGetMemHandle` / `muIpcOpenMemHandle` / `muIpcCloseMemHandle` | IPC 内存句柄导出/导入/关闭 | `cudaIpcGetMemHandle` 等 |
| 23 | [23_IPC_Import_Leak_Analysis.md](23_IPC_Import_Leak_Analysis.md) | IPC Import 资源泄漏分析 | Synchronize 失败/Peer 未撤销/Pool 归还等泄漏路径 | — |
| 24 | [24_muLaunchKernel_Flow.md](24_muLaunchKernel_Flow.md) | `muLaunchKernel` / `muLaunchKernelEx` / `muLaunchCooperativeKernel` | kernel 启动完整六层调用链 + 双线程模型 + 合并策略 | `cudaLaunchKernel` |
| 25 | [25_Threading_Model.md](25_Threading_Model.md) | 线程模型深度分析 | 全线程清单/锁体系/原子操作/TLS/5 场景/8 优化方案 | — |
| 26 | [26_Architecture_Summary.md](26_Architecture_Summary.md) | 完整架构总结 | 六层架构图/命令流程/线程模型/内存管理/信号量/状态机/环境变量/设计模式 | — |

## 各文件内容结构

每个文件包含:

1. **功能概述** — API 的作用和对应的 CUDA API
2. **完整调用链** — 从用户代码到 KMD/GPU 的逐层调用栈 (箭头格式)
3. **时序图** — ASCII 时序图, 展示每层交互的详细时间顺序
4. **关键代码路径** — 每个核心函数的完整代码 + 逐行注释 (非伪代码, 基于源码)
5. **关键设计要点** — 提炼该 API 的核心设计模式和架构决策

## 核心代码路径速查

| 层 | 文件 | 关键函数 |
|----|------|---------|
| Wrapper | `driver/mu_wrappers_generated.cpp` | `muMemAlloc_v2`, `muMemFree_v2`, ... (自动生成, MUPTI 插桩) |
| Driver | `driver/mu_memory.cpp` | `muapiMemAlloc_v2:265`, `muapiMemFree_v2:709`, `muapiMemcpyHtoD_v2:807`, `muapiMemsetD32_v2:1589` |
| Driver Module | `driver/mu_module.cpp` | `muapiLaunchKernel:232`, `muapiLaunchKernelEx:274` |
| Driver VMM | `driver/mu_vmm.cpp` | `muapiMemAddressReserve:14`, `muapiMemCreate:70`, `muapiMemMap:169` |
| Context | `core/context.cpp` | `CreateMemory:915`, `GeneralLaunchKernel:633`, `CreateKernelNode:2031`, `ResolveDependencyAndQueueCommand:1859` |
| Memory | `core/memory.cpp` | `GeneralAlloc:462`, `PinnedHostAlloc:532`, `ManagedAlloc:673` |
| Stream | `core/stream.cpp` | `CmdLaunchKernel:1415`, `QueueCommand:1005`, `AsyncSubmit:1108`, `AsyncWait:1280` |
| Command | `core/command/dispatchCommand.cpp` | `Build:67`, `Submit:224` |
| HAL | `hal/halCmdBuffer.h` | `ICmdBuffer` 接口 |
| M3D | `hal/m3d/memMgr.cpp` | `MemMgr::Allocate`, `Free` |
| M3D | `hal/m3d/memoryPool.cpp` | `FullAllocate`, `SubAllocate`, `ChunkAllocate` |

## 设计模式汇总

1. **Wrapper → Driver → Core → HAL → M3D → KMD**: 6 层接口模式
2. **三路分支**: `Stream::CmdXXX → Capture | Invalidated | Async`
3. **MemoryTracker**: 全局 `MUdeviceptr → IMemory*` 区间映射
4. **三层内存分配**: `Hal::IMemory(裸)` → `Hal::IMemMgr(sub)` → `Musa::MemoryPool(用户)`
5. **1D→3D 统一**: memcpy 都转为 `MUSA_MEMCPY3D_PEER`, memset 转为 `MUSA_MEMSET_NODE_PARAMS`
6. **Command 生命周期**: Created → Queued → Built → Submitted → Completed
7. **双线程异步**: AsyncSubmit + AsyncWait per Stream
8. **属性推导链**: Driver flags → Core 推导 → HAL property
9. **双回合分配**: SubAllocate → ChunkAllocate → SubAllocate

## 参考

- 源码目录: `musa/src/driver/`, `musa/src/musa/core/`, `musa/src/hal/`
- 完整架构总结: [26_Architecture_Summary.md](26_Architecture_Summary.md)