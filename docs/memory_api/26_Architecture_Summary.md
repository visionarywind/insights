# MUSA Runtime 完整架构总结

> 汇总 MUSA Runtime 六层架构、核心设计模式、完整命令执行流程和线程模型。

---

## 1. 六层架构全景图

```
┌─────────────────────────────────────────────────────────────────────────┐
│  应用层 (User Code)                                                     │
│  musaMalloc / muLaunchKernel / muStreamSynchronize / ...               │
├─────────────────────────────────────────────────────────────────────────┤
│  第 1 层: Wrapper (mu_wrappers_generated.cpp)                          │
│  ├─ ApiTrace 记录 (MUPTI 插桩)                                         │
│  ├─ toolsCallbackEnabled → toolsIssueCallback (第三方 profiler)         │
│  └─ 参数打包/解包 → 调用 Driver 层 API                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  第 2 层: Driver (mu_memory.cpp / mu_module.cpp / mu_stream.cpp)       │
│  ├─ InitPlatform() — 懒初始化                                           │
│  ├─ TlsCtxTop() → 线程局部 Context                                     │
│  ├─ 参数校验 + MemoryCreateInfo 构造                                    │
│  └─ 委托到 Core 层 (Context::XXX)                                       │
├─────────────────────────────────────────────────────────────────────────┤
│  第 3 层: Core (context.cpp / memory.cpp / stream.cpp / device.cpp)    │
│  ├─ Context:   对象生命周期管理 / 依赖解析 / 命令入队                     │
│  ├─ Memory:    内存类型分发 / Property 推导 / VA↔PA 绑定                │
│  ├─ Stream:    双线程模型 / 命令合并 / 图捕获                            │
│  └─ Device:    Peer Access / 引擎管理 / Pool 管理                       │
├─────────────────────────────────────────────────────────────────────────┤
│  第 4 层: Command (command/ — Dispatch / Memset / Memcpy / Barrier等)   │
│  ├─ Command 基类: 状态机 / 信号量依赖 / 提交抽象                         │
│  ├─ DispatchCommand: 内核参数设置 / 隐式资源分配 / CmdDispatch           │
│  ├─ MemsetCommand:   引擎选择 (CDM→CE→TDM→DMA→CPU) + 执行              │
│  └─ GraphCommand:    图捕获回放                                          │
├─────────────────────────────────────────────────────────────────────────┤
│  第 5 层: HAL (hal/ — IMemory / IQueue / ICmdBuffer / ISemaphore)      │
│  ├─ IMemory:    内存分配/释放/映射接口                                   │
│  ├─ IMemMgr:    Sub-Allocation 管理器 (SplayTree → Pool)               │
│  ├─ ICmdBuffer: GPU 命令录制接口 (Begin → CmdXXX → End)                │
│  ├─ IQueue:     HAL 队列提交                                            │
│  └─ ISemaphore: Timeline / Hardware 信号量                              │
├─────────────────────────────────────────────────────────────────────────┤
│  第 6 层: M3D (hal/m3d/ — GPU 硬件接口)                                │
│  ├─ Device:    GpuMemory 分配 / Queue 创建 / 引擎管理                     │
│  ├─ Memory:    DRM mtgpuBoAlloc / mtgpuBoVmMapV2                       │
│  ├─ MemoryPool: 物理内存 Sub-Allocation 算法                            │
│  └─ Semaphore: 硬件二值信号量 / Timeline 信号量                          │
├─────────────────────────────────────────────────────────────────────────┤
│  第 7 层: KMD (gr-kmd/ — 内核驱动, ioctl 接口)                          │
│  ├─ mtgpuBoAlloc:    分配 GPU buffer                                    │
│  ├─ mtgpuBoVmMapV2:  映射到 GPU VA 空间                                 │
│  ├─ mtgpuQueueSubmit: 提交 Command Buffer 到硬件队列                     │
│  └─ mtgpuWaitFence:  等待 GPU 完成                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心 API 执行流程全景

```
                    muLaunchKernel              muMemsetD8           muMemcpyHtoD
                         │                          │                     │
    ┌────────────────────┼──────────────────────────┼─────────────────────┼─────────┐
    │ Driver             │                          │                     │         │
    │  mu_module.cpp:232 │                          │                     │         │
    │  ├─ InitPlatform   │                          │                     │         │
    │  ├─ TlsCtxTop      │                          │                     │         │
    │  ├─ nodeParams={}  │                          │                     │         │
    │  └─ GeneralLaunch  │                          │                     │         │
    └─────────┬──────────┘     ┌────────────────────┤  mu_memory.cpp:807  │         │
              │                │ mu_memory.cpp:1531 │  ├─ GetMemcpy3D     │         │
              ▼                │ ├─ memsetParams={} │  └─ GeneralMemcpy   │         │
    ┌─────────────────────────┐│ └─ GeneralMemset   │                     │         │
    │ Core: Context            ││        │           │                     │         │
    │  context.cpp:633         ││        ▼           │                     │         │
    │  ├─ InfoStream(hStream)  ││ ┌──────────────┐   │                     │         │
    │  ├─ CreateKernelNode()   ││ │ Core: Context │   │                     │         │
    │  │  └─ UpdateParams()    ││ │ context:733   │   │                     │         │
    │  │    ├─ grid/block 校验 ││ │ ├─ InfoStream  │   │                     │         │
    │  │    ├─ cluster 校验    ││ │ └─ CreateMemset│   │                     │         │
    │  │    └─ CreateState()   ││ │    Node()      │   │                     │         │
    │  └─ CmdLaunchKernel()    ││ └──────┬─────────┘   │                     │         │
    └──────────┬───────────────┘│        │              │                     │         │
               │                │        ▼              ▼                     │         │
               ▼                │ ┌──────────────────────────────────────────┐│
    ┌─────────────────────┐     │ │ Command 层                                ││
    │ Stream::CmdXXX()     │     │ │ ├─ DispatchCommand / MemsetCommand /     ││
    │ ├─ new *Command      │     │ │ │  MemcpyCommand                        ││
    │ └─ ResolveDependency │     │ │ ├─ Build(): 依赖构建 + 信号量等待        ││
    │    AndQueueCommand()  │     │ │ │  ├─ CmdBufferBegin                   ││
    └──────────┬───────────┘     │ │ │  ├─ 隐式资源分配 (Spill/Print/Cluster) ││
               │                 │ │ │  ├─ 引擎选择 (CDM→CE→TDM→DMA→CPU)     ││
               ▼                 │ │ │  └─ CmdDispatch/CmdFillMemory/CmdCopy  ││
    ┌────────────────────────┐   │ │ ├─ Submit(): CmdBufferEnd + SubmitToQueue││
    │ 依赖解析 + 命令合并     │   │ │ └─ Postprocess(): 释放 CmdBuffer         ││
    │ context.cpp:1859        │   │ └──────────────────────────────────────────┘│
    │ ├─ DefaultStream 依赖    │   │                    │                       │
    │ ├─ Barrier 依赖          │   │                    ▼                       │
    │ ├─ 同流依赖链             │   │ ┌──────────────────────────────┐          │
    │ └─ QueueCommand()        │   │ │ HAL 层                       │          │
    │    └─ notify AsyncSubmit  │   │ │ SubmitToQueue(IQueue, info)  │          │
    └────────────────────────┘   │ │  └─ KMD mtgpuQueueSubmit       │          │
                                 │ └──────────────────────────────┘          │
              ┌──────────────────┘                                           │
              ▼                                                              │
    ┌─────────────────────────────────────────────────────────────────────┐ │
    │ 双线程异步执行                                                       │ │
    │ ┌──────────────┐     ┌──────────────┐                               │ │
    │ │ AsyncSubmit  │ ──→ │  AsyncWait   │                               │ │
    │ │ Build+Merge  │     │  Wait+Clean  │                               │ │
    │ │ +Submit      │     │  +ErrorCheck │                               │ │
    │ └──────────────┘     └──────────────┘                               │ │
    └─────────────────────────────────────────────────────────────────────┘ │
```

---

## 3. 线程模型全景

```
┌─────────────────────────────────────────────────────────────┐
│  Thread Pool                                                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Platform::m_Executor (1× Worker)                    │    │
│  │  ├─ ReleaseResources() — 延迟释放时间戳内存          │    │
│  │  └─ ~Stream() 同步 — 确保资源释放完成                │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│  Per-Stream Threads (2× per Stream)                         │
│  ┌─────────────────────┐   ┌─────────────────────┐         │
│  │ AsyncSubmit         │   │ AsyncWait           │         │
│  │ stream.cpp:1108     │   │ stream.cpp:1280     │         │
│  │                     │   │                     │         │
│  │ ① Wait CmdList≠∅    │   │ ① Wait Inflight≠∅  │         │
│  │ ② 依赖过滤+合并判定 │   │ ② 等待 Semaphore   │         │
│  │ ③ Build(依赖链)     │   │ ③ 引擎错误检查     │         │
│  │ ④ 合并到 MergingList│   │ ④ Postprocess      │         │
│  │ ⑤ 主命令 Submit     │   │ ⑤ 状态 → completed │         │
│  │ ⑥ → InflightList    │   │ ⑥ m_AsyncCount--   │         │
│  │                     │   │ ⑦ ReleaseResources │         │
│  │ 数据:               │   │    → Executor       │         │
│  │  m_CommandList      │   │ 数据:              │         │
│  │  m_MergingList      │   │  m_InflightList    │         │
│  │  m_SubmitMtx        │   │  m_WaitingList     │         │
│  │  m_SubmitCv         │   │  m_WaitMtx         │         │
│  └─────────────────────┘   │  m_WaitCv          │         │
│                             └─────────────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  User Threads (N×)                                          │
│  ├─ TlsCtxTop() → 当前线程 Context                          │
│  ├─ QueueCommand() → m_CommandList (轻量入队)               │
│  └─ m_AsyncCount 背压控制                                   │
├─────────────────────────────────────────────────────────────┤
│  辅助线程 (按需)                                             │
│  ├─ PrintManager::m_flushRoutine — GPU printf 输出          │
│  ├─ NotificationManager::m_Thread — 异步回调监控            │
│  └─ MUgdb Monitor — 调试器 hook                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 内存管理全景

```
                         muMemAlloc_v2
                              │
                              ▼
    ┌─────────────────────────────────────────────────┐
    │  Driver: flags = Virtual|DeviceMapped|SubAlloc  │
    │         = 0x0007                                │
    └────────────────────┬────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────────┐
    │  Core: GeneralAlloc                             │
    │  ├─ property 推导: 0x0007 → 0x03FF              │
    │  │   +Physical +SharedVA +DeviceVisible         │
    │  │   +HostVisible +HostCoherent +DeviceWriteable│
    │  │   +DeviceCached                              │
    │  └─ viewCap 推导: 0x00 → 0x07                   │
    │      +Exportable +PeerAccessible +IpcExportable  │
    └────────┬───────────────────┬────────────────────┘
             │                   │
    SubAlloc=YES (默认)    SubAlloc=NO (特殊)
             │                   │
             ▼                   ▼
    ┌─────────────────┐  ┌──────────────────┐
    │ MemMgr::Allocate│  │CreateMemory(裸)  │
    │  ├─ SplayTree   │  │ → 每次 2× ioctl  │
    │  │  查找/创建Pool│  │ mtgpuBoAlloc     │
    │  └─ FullAllocate │  │ mtgpuBoVmMapV2   │
    │     │            │  │                  │
    │     ▼            │  │ 性能: 每次 ~0.2ms│
    │ ┌──────────┐     │  └──────────────────┘
    │ │SubAllocate│    │
    │ │ O(1) 位图 │    │
    │ │ 桶查找   │     │
    │ │ 命中率 99+%│   │
    │ │ → 零 ioctl │   │
    │ └─────┬─────┘    │
    │       │未命中    │
    │       ▼          │
    │ ┌──────────┐     │
    │ │ChunkAlloc│     │
    │ │2MB chunk │     │
    │ │2× ioctl  │     │
    │ │仅首次    │     │
    │ └──────────┘     │
    └─────────────────┘

    释放路径:
    ~Memory()
      ├─ SubAlloc → MemMgr::Free / Pool::Free
      │   └─ 归还 free list → 惰性回收 chunk
      └─ 非 SubAlloc → Destroy() → KMD 立即释放
```

---

## 5. 信号量与同步体系

```
    ┌──────────────────────────────────────────────────────┐
    │  跨 Stream 同步 (Stream 级信号量)                    │
    │                                                      │
    │  ┌─────────────────┐    ┌─────────────────┐          │
    │  │Timeline Semaphore│    │Hardware Semaphore│         │
    │  │ 连续递增 uint64  │    │ 二值 (0/1)      │          │
    │  │ Wait(value)      │    │ Wait()          │          │
    │  │ Signal(value)    │    │ Signal()        │          │
    │  │ GPU→GPU 依赖     │    │ GPU→CPU 通知    │          │
    │  └─────────────────┘    └─────────────────┘          │
    │                                                      │
    │  选择逻辑 (GetPreferredSemaphoreType):               │
    │    EnableUserQueue → Hardware (低延迟)               │
    │    !EnableUserQueue → Timeline (通用)                │
    ├──────────────────────────────────────────────────────┤
    │  跨进程同步 (IPC 级)                                  │
    │                                                      │
    │  ┌──────────────────────────────────────────────┐    │
    │  │ IPC Event Ring Buffer (共享内存, 无锁)       │    │
    │  │                                              │    │
    │  │  write_index (atomic)    ← 记录端递增         │    │
    │  │  read_index  (CAS)       ← 等待端 CAS 更新    │    │
    │  │  signal[N]   (atomic)    ← 每槽自旋锁        │    │
    │  └──────────────────────────────────────────────┘    │
    └──────────────────────────────────────────────────────┘
```

---

## 6. 命令生命周期状态机

```
                  ┌────────────────────────────────────┐
                  │ 用户线程                            │
                  │  Stream::QueueCommand()            │
                  │  → SetStatus(queued)               │
                  │  → push m_CommandList              │
                  │  → notify AsyncSubmit              │
                  └──────────────┬─────────────────────┘
                                 │
                    Created ──→ Queued
                                 │
                  ┌──────────────▼─────────────────────┐
                  │ AsyncSubmit 线程                    │
                  │  Command::Build(mergingList)       │
                  │  → 信号量依赖构建                   │
                  │  → 隐式资源分配                     │
                  │  → CmdBufferBegin/End              │
                  │  → SetStatus(built)               │
                  └──────────────┬─────────────────────┘
                                 │
                    Queued ──→ Built
                                 │
                  ┌──────────────▼─────────────────────┐
                  │ AsyncSubmit 线程                    │
                  │  Command::Submit()                 │
                  │  → CmdBufferEnd()                  │
                  │  → SubmitToQueue() → ioctl         │
                  │  → SetStatus(submitted)            │
                  │  → move to InflightList            │
                  └──────────────┬─────────────────────┘
                                 │
                    Built ──→ Submitted
                                 │
                  ┌──────────────▼─────────────────────┐
                  │ GPU 硬件执行                         │
                  │  CmdDispatch / CmdFillMemory 等     │
                  │  → 完成后信号 Semaphore             │
                  └──────────────┬─────────────────────┘
                                 │
                  ┌──────────────▼─────────────────────┐
                  │ AsyncWait 线程                      │
                  │  Semaphore::Wait(signalValue)      │
                  │  → 引擎错误检查                     │
                  │  → Postprocess (释放 CmdBuffer)    │
                  │  → ReleaseResources → Executor     │
                  │  → SetStatus(completed / error)    │
                  │  → m_AsyncCount--                   │
                  └─────────────────────────────────────┘

    compare_exchange_strong 确保状态转移的原子性和唯一性
```

---

## 7. 环境变量速查表

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MUSA_LOG` | `0` | 日志位掩码，`0xffff`=全部 |
| `MUSA_VISIBLE_DEVICES` | `0,1,...7` | 可见设备列表 |
| `MUSA_LAUNCH_BLOCKING` | `false` | 内核启动阻塞等待 |
| `MUSA_STREAM_ASYNC_CAPACITY` | `1024` | 流异步命令容量 |
| `MUSA_MEMCPY_PATH` | `0` (Default) | memcpy 引擎选择: 1=DMA 2=TDM 3=CE 4=CPU 5=CDM |
| `MUSA_MEMSET_PATH` | `0` (Default) | memset 引擎选择: 同上 |
| `MUSA_INFLIGHT_SUBMISSION_LIMIT` | `0` | 飞行提交数限制 (0=自动) |
| `MUSA_FORCE_SINGLE_CORE` | `false` | 强制单核执行 |
| `MUSA_CDM_PREFETCH` | `false` | CDM prefetch 开关 |
| `MUSA_SEMAPHORE_OPEN_MODE` | `1` | 信号量模式: 0=MTLink优先 1=仅PCIe |
| `MUSA_DEVICE_ORDER` | `FASTEST_FIRST` | 设备枚举顺序 |
| `MUSA_DEVICE_PAGE_SIZE` | `0x40000` (256KB) | 内存映射对齐 |
| `MUSA_EXECUTION_TIMEOUT` | `200000` (ms) | 内核/内存操作超时 |
| `MUSA_ENABLE_COREDUMP_ON_EXCEPTION` | `true` | GPU 异常时生成 core dump |
| `MUSA_COREDUMP_FILE` | `""` | core dump 文件名模板 |

---

## 8. 设计模式总结

| 模式 | 描述 | 文件示例 |
|------|------|---------|
| **六层接口** | Wrapper→Driver→Core→Command→HAL→M3D→KMD | 贯穿全部 API |
| **三路分发** | Capture / Invalidated / Async | `stream.cpp` 所有 CmdXXX |
| **双线程异步** | Submit + Wait 线程 per Stream | `stream.cpp:1108,1280` |
| **共享锁读写** | `ReadWriteLockedBase<shared_mutex>` | `context.h:160` |
| **位掩码日志** | `MUSA_LOG` 按 bit 控制输出 | `utilLog.h:92-101` |
| **属性推导链** | Driver flags → Core 推导 → HAL property | `memory.cpp:462-497` |
| **双回合分配** | SubAllocate → ChunkAllocate → SubAllocate | `memoryPool.cpp:82-95` |
| **命令合并** | 同引擎相邻命令合并到同一 CmdBuffer | `stream.cpp:1192-1228` |
| **惰性回收** | Pool chunk 归还后不立即释放 | `memoryPool.cpp:214-259` |
| **粘性错误** | 首个错误阻塞整个 Stream | `stream.cpp:1360-1366` |
| **TLS Context 栈** | `thread_local TlsData` 每线程独立 | `internal.cpp:14` |

---

## 9. 文档索引

| # | 文档 | 内容 |
|---|------|------|
| 01 | `01_muMemAlloc_v2.md` | 设备内存分配完整六层逐行分析 |
| 02 | `02_muMemFree_v2.md` | 内存释放双路径 + 死代码 |
| 05 | `05_muMemcpyHtoD_v2.md` | 全部拷贝变体 + 1D→3D 归一化 |
| 06 | `06_muMemsetD32_v2.md` | GPU Memset 引擎选择 |
| 07-08 | `07/08` Async 系列 | 异步分配/释放 + Bind/Unbind |
| 10 | `10_CreateMemory_and_MapToPeers.md` | 统一入口 + 9 种类型分发 |
| 11-14 | `11-14` Pool 系列 | SubAllocation 全链路 |
| 18 | `18_Stream_subsystem.md` | Stream 六层架构 + 双线程 |
| 19 | `19_VA_PA_binding.md` | VA↔PA 三层绑定体系 |
| 21 | `21_Event_API.md` | Event 完整生命周期 |
| 22 | `22_IPC_Memory_API.md` | IPC 内存共享三 API |
| 23 | `23_IPC_Import_Leak_Analysis.md` | IPC 资源泄漏 4 条路径 |
| 24 | `24_muLaunchKernel_Flow.md` | 内核启动完整六层调用链 |
| 25 | `25_Threading_Model.md` | 线程模型 5 场景 + 8 优化方向 |
| 26 | `26_Architecture_Summary.md` | 🆕 本文档 — 架构总结 |
