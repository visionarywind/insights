# MUSA Stream 子系统深度源码分析

> **文档编号**: 18  
> **相关文件**: `musa/doc/memory_api/README.md`  
> **预估总行数**: ~1000+ 行  
> **核心源码文件**:  
> - `musa/src/musa/core/stream.h` (242 行) — Stream 类声明  
> - `musa/src/musa/core/stream.cpp` (1793 行) — Stream 类实现  
> - `musa/src/musa/core/command/command.h` (336 行) — Command 基类  
> - `musa/src/musa/core/command/command.cpp` (798 行) — Command 基类实现  
> - `musa/src/driver/mu_stream.cpp` (951 行) — Driver 层 Stream API 入口  
> - `musa/src/musa/musaStream.h` (269 行) — IStream 抽象接口  
> - `musa/src/musa/core/context.cpp` — ResolveDependencyAndQueueCommand  

---

## 1. 功能概述

Stream（流）是 MUSA Runtime 的**核心调度单元**，承担三个关键职责：

1. **命令队列**：接收并缓存 KernelLaunch / Memcpy / Memset / Barrier 等各类命令
2. **异步执行**：通过独立线程将命令提交到 GPU 硬件队列，不阻塞调用线程
3. **同步协调**：提供 Synchronize / Event / Semaphore 等机制控制执行顺序

### 1.1 对应 CUDA

| MUSA | CUDA |
|------|------|
| `muStreamCreate` | `cudaStreamCreate` |
| `muStreamCreateWithPriority` | `cudaStreamCreateWithPriority` |
| `muStreamSynchronize` | `cudaStreamSynchronize` |
| `muStreamQuery` | `cudaStreamQuery` |
| `muStreamWaitEvent` | `cudaStreamWaitEvent` |
| `muStreamAddCallback` | `cudaStreamAddCallback` |
| `muStreamBeginCapture` | `cudaStreamBeginCapture` |
| `muStreamEndCapture` | `cudaStreamEndCapture` |
| `muStreamWriteValue32` | `cudaStreamWriteValue32` |
| `muStreamBatchMemOp_v2` | `cudaStreamBatchMemOp` |

---

## 2. 整体架构与六层调用链

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              User Code                                     │
│  musaStreamCreate()  musaStreamSynchronize()  musaMemcpyAsync()           │
└──────────────────────┬─────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────────────────────────────┐
│  Layer 1: Runtime API (musaStream*)                                        │
│  文件: musa/src/musa/runtime_api/ (自动生成或手写)                          │
│  功能: 参数校验 → 调用 Driver API                                         │
└──────────────────────┬─────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────────────────────────────┐
│  Layer 2: Driver Wrapper (muapiStream*)                                    │
│  文件: musa/src/driver/mu_stream.cpp (951行)                               │
│  功能: InitPlatform → TlsCtxTop → 参数校验 → 委托 Core                    │
│  关键: muapiStreamCreate → pContext->CreateStream()                        │
│        muapiStreamSynchronize → pStream->Synchronize()                    │
└──────────────────────┬─────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────────────────────────────┐
│  Layer 3: Core Stream 类                                                  │
│  文件: musa/src/musa/core/stream.cpp (1793行)                             │
│  功能: 命令生命周期管理 / 异步提交 / 同步等待 / 图捕获                      │
│  关键: QueueCommand() → 写入 m_CommandList                                │
│        AsyncSubmit线程 → 从CommandList取命令 → Build → Submit              │
│        AsyncWait线程  → 从InflightList取命令 → WaitFinish                  │
└──────────────────────┬─────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────────────────────────────┐
│  Layer 4: Command 类体系                                                  │
│  文件: musa/src/musa/core/command/command.cpp (798行)                      │
│  继承体系:                                                                 │
│    Command (基类: 生命周期/依赖/信号/提交)                                  │
│    ├─ DispatchCommand       (内核启动)                                     │
│    ├─ SyncMemcpyCommand     (同步拷贝)                                     │
│    ├─ AsyncMemcpyCommand    (异步拷贝)                                     │
│    ├─ MemsetCommand         (内存填充)                                     │
│    ├─ CallbackCommand       (回调函数)                                     │
│    ├─ GraphCommand          (图执行)                                       │
│    ├─ BarrierCommand        (屏障同步)                                     │
│    ├─ RecordCommand         (事件记录)                                     │
│    ├─ MemoryAtomicCommand   (原子操作)                                     │
│    ├─ PagingCommand         (内存分页)                                     │
│    ├─ SignalExternalSemaphoreCommand / WaitExternalSemaphoreCommand        │
│    └─ AccelStructBuild/Copy/EmitCommand                                    │
└──────────────────────┬─────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────────────────────────────┐
│  Layer 5: HAL 接口 (Hal::IQueue / Hal::ICmdBuffer / Hal::ISemaphore)      │
│  文件: hal/halQueue.h / hal/halCmdBuffer.h / hal/halSemaphore.h            │
│  功能: 硬件抽象，屏蔽 Linux DRM / Windows WDDM 差异                        │
└──────────────────────┬─────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────────────────────────────┐
│  Layer 6: KMD (M3D) / GPU 硬件                                            │
│  文件: hal/m3d/*.cpp / Linux DRM ioctl / Windows WDDM DDI                  │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. IStream 抽象接口

**`musaStream.h:224-267`** 定义了 `IStream` 纯虚接口，是 Driver 层和 Core 层的协议边界：

```cpp
class IStream : public MUstream_st {
public:
    virtual MUresult BeginCapture(IGraph*, MUstreamCaptureMode, ...) = 0;
    virtual MUresult EndCapture(IGraph**) = 0;
    virtual MUresult Query() = 0;
    virtual MUresult Synchronize() = 0;

    // 命令提交接口
    virtual MUresult CmdLaunchRays(...) = 0;
    virtual MUresult CmdBuildAccelStruct(...) = 0;
    virtual MUresult CmdCopyMemory(...) = 0;        // ← graph node 级别
    virtual MUresult CmdMemset(...) = 0;
    virtual MUresult CmdLaunchGraph(IGraphExec*) = 0;
    virtual MUresult CmdMemAlloc(...) = 0;
    virtual MUresult CmdMemFree(...) = 0;
    virtual MUresult CmdMemoryTransfer(...) = 0;

    // 外部信号量
    virtual MUresult CmdSignalExternalSemaphore(...) = 0;
    virtual MUresult CmdWaitExternalSemaphore(...) = 0;

    // 属性接口
    virtual unsigned int GetFlags() const = 0;
    virtual int GetPriority() const = 0;
    virtual uint64_t GetSeqID() const = 0;
    virtual IContext* GetContext() const = 0;
};
```

> **设计分析**: `IStream` 将"命令提交"和"流控制"合二为一，而非像某些 GPU API 那样分离为独立的 Queue 和 Stream 对象。所有命令通过 `Cmd*` 系列虚函数提交，由 Stream 统一管理生命周期。

---

## 4. Stream 类深度分析

### 4.1 数据成员分类

**`stream.h:170-240`** 中定义了以下核心数据：

**4.1.1 命令管线（Pipeline）**

```
m_CommandList     — 待构建的命令列表 (从 QueueCommand 写入)
m_MergingList     — 正在合并的命令组 (构建阶段使用)
m_InflightList    — 已提交但未完成的命令 (等待线程消费)
m_WaitingList     — 当前等待线程正在等待的命令
m_LastCommand     — 上一条命令 (用于链式依赖)
```

**命令流动模型**:
```
QueueCommand()
    │
    ▼
m_CommandList ──(AsyncSubmit线程)──> Build() ──> m_MergingList
    │                                              │
    │                                    Submit() │
    │                                              ▼
    │                                    m_InflightList ──(AsyncWait线程)──> 完成
    │                                                    │
    │                                          WaitFinish()
    │                                                    ▼
    └──────────────────── m_LastCommand ←──────────────┘
```

**4.1.2 异步线程**

```cpp
m_SubmitThread   — 提交线程 (AsyncSubmit)
m_SubmitMtx/Cv   — 提交线程的互斥锁和条件变量
m_SubmitStopToken — 停止标志

m_WaitThread     — 等待线程 (AsyncWait)
m_WaitMtx/Cv     — 等待线程的互斥锁和条件变量
m_WaitStopToken  — 停止标志
```

**4.1.3 信号量体系**

```cpp
// 流级别信号量（命令间同步）
m_TimelineSemaphore      — 时间线信号量 (跨命令)
m_HardwareSemaphore      — 硬件信号量 (跨命令)
m_TimelineValue          — 下一个要 signal 的时间线值

// 命令内信号量（命令内等待/信号）
m_InternalTimelineSemaphore  — 内部时间线信号量
m_InternalHardwareSemaphore  — 内部硬件信号量
m_InternalTimelineValue      — 内部时间线值
```

**4.1.4 引擎资源**

```cpp
m_EngineResources[Engine::count] — 每个引擎类型的资源包
    ├─ queue         — Hal::IQueue (GPU 硬件队列)
    ├─ cmdPool       — Hal::ICmdPool (命令缓冲池)
    ├─ cmdBufferLists — 命令缓冲列表 (按 CmdBufferListType 分类)
    └─ mtx           — 互斥锁

m_InflightSubmissionCounts[Engine::count] — 每个引擎的 inflight 提交计数
```

**4.1.5 捕获相关**

```cpp
m_CaptureGraph          — 正在捕获的图
m_CaptureStatus         — 捕获状态 (NONE/ACTIVE/INVALIDATED)
m_IsOriginStream        — 是否为捕获起始流
m_LastCapturedNodes     — 最后捕获的节点列表
m_PrimaryCaptureStream  — 主捕获流
m_CapturedEvents        — 捕获期间的事件集合
```

### 4.2 Init() 初始化流程

**`stream.cpp:885-983`** — 初始化函数，核心步骤：

```
Step 1: 为每种引擎类型创建队列 (halDevice.CreateQueue)
Step 2: 为每种引擎类型创建命令池 (halDevice.CreateCmdPool)
Step 3: 为 Universal 引擎预分配命令缓冲 (s_CmdBufferListSize = 0 → 初始无缓冲)
Step 4: 创建时间线信号量 (Hal::SemaphoreType::Timeline)
Step 5: 如果设备支持引擎同步，创建硬件信号量
Step 6: 预创建 Surface 对象 (s_SurfaceObjLimit = 8)
Step 7: 启动异步提交线程 AsyncSubmit
Step 8: 启动异步等待线程 AsyncWait
```

**关键常量**:

| 常量 | 值 | 含义 |
|------|-----|------|
| `s_CmdBufferListSize` | 0 | 初始命令缓冲数量（按需分配） |
| `s_SurfaceObjLimit` | 8 | 预分配 Surface 对象数 |
| `s_InflightSubmissionLimit` | 3 | 飞行中提交数上限（启用用户队列时） |
| `s_NonUserQueueInflightSubmissionLimit` | 2 | 非用户队列飞行提交数上限 |
| `reserveSpillSize` | 1 << 26 (64MB) | 溢出内存预留大小 |

### 4.3 QueueCommand() — 命令入队

**`stream.cpp:1005-1065`** — 所有命令进入同一入口：

1. **背压控制**: 自旋等待直到 `m_AsyncCount < asyncCapacity`
2. **设置前驱**: `command->SetPrevCommand(m_LastCommand)` — 建立链式依赖
3. **设置状态**: `command->SetStatus(Command::Status::queued)`
4. **记录时间戳**: Graph/Dispatch 类型记录排队时间
5. **入队**: `m_CommandList.push_back(command)`
6. **通知**: `m_SubmitCv.notify_one()` 唤醒 AsyncSubmit 线程

### 4.4 AsyncSubmit() — 异步提交线程

**`stream.cpp:1108-1278`** — 核心提交循环：

```
循环:
  1. 等待条件: !m_CommandList.empty() 或 stopMerging() 或 m_SubmitStopToken
  2. 检查是否需要停止合并 (stopMerging):
     - 追踪捕获模式下: 达到 traceMergeThreshold 或命令列表为空
     - 否则: 引擎就绪 或 mergingList >= 32
  3. 如果需要停止合并 → submitMergingList()
  4. 从 m_CommandList 取出队首命令
  5. 调用 buildCommand():
     a. command->FilterDependency() — 过滤已完成依赖
     b. 检查能否与 MergingList 合并 command->CanMergeTo()
     c. 设置信号量值 (合并时复用 primary 的值)
     d. command->Build(m_MergingList) — 构建命令缓冲
     e. command->SetStatus(Built)
     f. 加入 m_MergingList
```

**命令合并 (Command Merge)**:
- 相同引擎的相邻命令如果都支持合并 (`SupportMerge()`)，会被合并到同一条 `halCmdBuffer` 中
- `SyncMemcpyCommand` 和 `GraphCommand` **不支持合并**，会强制刷新 MergingList
- 合并后只有 primary command 真正执行 submit，secondary 共享其状态

### 4.5 AsyncWait() — 异步等待线程

**`stream.cpp:1280-1398`** — 核心等待循环：

```
循环:
  1. 等待条件: !m_InflightList.empty() 或 m_WaitStopToken
  2. 将 primary command 从 InflightList 移到 WaitingList
  3. 检查错误: 如果 WaitingList 中有 error，取第一个错误
  4. 如果 primary command 是主合并层级:
     a. 如果是 Hardware semaphore:
        - 等待 semaphore value
        - 同时 signal timeline semaphore
     b. 如果是 Timeline semaphore:
        - 等待 semaphore value
  5. 检查引擎错误 (GetEngineLastError)
  6. 遍历 WaitingList:
     a. 输出 PrintBuffer (检测 kernel 断言)
     b. 设置 command 状态 (completed/error)
     c. 递减 m_AsyncCount
     d. 调用 Postprocess() (回收命令缓冲)
     e. 调用 SetStatus()
  7. 释放等待列表中的资源
```

### 4.6 Synchronize() — 同步接口

**`stream.cpp:221-237`**：

```
Synchronize():
  如果是捕获状态 → 报错 STREAM_CAPTURE_UNSUPPORTED
  否则:
    如果是默认流 → LockedSyncDefaultStream() [等待所有非阻塞流]
    否则 → WaitFinish() [等待最后一条命令完成]
```

### 4.7 流间依赖模型

**`context.cpp:1859-1921`** — `ResolveDependencyAndQueueCommand`:

```
Default Stream:  依赖所有非阻塞流 (blocking streams) 的最后一条命令
Barrier Stream:  依赖所有其他流的最后一条命令
Blocking Stream: 依赖默认流的最后一条命令
```

图示：
```
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  DefaultStream│────>│ BlockStream1 │────>│ BlockStream2 │
  │  (auto-sync)  │     │ (sync->def)  │     │ (sync->def)  │
  └──────────────┘     └──────────────┘     └──────────────┘
       ▲                                               │
       │               ┌──────────────┐               │
       └───────────────│ BarrierStream│◄──────────────┘
                       │ (sync->ALL)  │
                       └──────────────┘
```

---

## 5. Command 基类深度分析

### 5.1 生命周期状态机

**`command.h:35-42`**:

```
created ──(queued)──> queued ──(built)──> built ──(submitted)──> submitted ──(completed/error)──> 完成
          CAS                                                                    CAS
```

每个状态转换通过 `compare_exchange_strong` 原子操作完成，确保线程安全。

### 5.2 信号量同步机制

**信号类型选择** (`command.cpp:277-279`):
```cpp
GetPreferredSemaphoreType() {
    return EnableUserQueue(engine) ? Hardware : Timeline;
}
```

**信号策略 (`command.cpp:340-366`)**:
```
SignalBetweenCmd:  本命令完成后 signal → 等待者通过 GetPeerSemaphore 获取 peer 信号量
SignalWithinCmd:   命令内部 signal (使用内部时间线信号量)
```

**等待策略 (`command.cpp:291-337`)**:
```
WaitBetweenCmd:  等待上一条命令的 semaphore
    ├─ Timeline → GetPeerSemaphore → 等待
    └─ Hardware → CmdWaitMemoryValue (内存等待)

WaitWithinCmd:   等待 Stream 内部时间线
    ├─ Timeline → GetPeerSemaphore(内部) → 等待
    └─ Hardware → CmdWaitMemoryValue (内部内存等待)
```

### 5.3 Build() 依赖构建

**`command.cpp:165-224`**:

1. 如果有前驱命令 (`m_PrevCommand`)，且需要同步:
   - 等待前驱状态达到 `built`
   - 根据信号量类型和是否同设备选择同步方式
2. 对所有执行依赖 (`m_ExecutionDependencies`) 重复上述过程
3. 等待完成后清空依赖列表

---

## 6. 图捕获 (Graph Capture) 机制

### 6.1 捕获生命周期

```
BeginCapture(graph, mode)    → m_CaptureStatus = ACTIVE, 创建空图
  │
  ├─ CmdLaunchKernel(node)   → Stream::CmdLaunchKernel
  │     └─ 正常路径: 创建 DispatchCommand → QueueCommand
  │     └─ 捕获路径: 创建 GraphNode → 添加到 m_CaptureGraph
  │
  ├─ CmdMemcpy(node)         → 类似
  │
  └─ ... 其他命令 ...
  │
EndCapture(&graph)           → 验证叶节点 → 返回图对象
```

### 6.2 捕获模式

| 模式 | 行为 |
|------|------|
| `MU_STREAM_CAPTURE_MODE_GLOBAL` | 全局捕获: 所有流上的操作都会被捕获 (检查 s_BeginCaptureNum) |
| `MU_STREAM_CAPTURE_MODE_THREAD_LOCAL` | 线程局部: 仅当前线程的流 (检查 t_BeginCaptureNum) |
| `MU_STREAM_CAPTURE_MODE_RELAXED` | 宽松: 不做严格线程检查 |

### 6.3 图执行

图执行时 (`Stream::CmdLaunchGraph`)，按 BFS 拓扑顺序遍历所有节点:
```
for node in bfsNodes:
    switch node.type:
        KERNEL      → CmdLaunchKernel
        MEMCPY      → CmdCopyMemory
        MEMSET      → CmdMemset
        HOST        → CmdLaunchHostFunc
        EMPTY       → (无操作)
        WAIT_EVENT  → AsyncWaitEvent
        EVENT_RECORD→ AsyncSetEvent
        MEM_ATOMIC  → CmdMemoryAtomic
        ...
```

---

## 7. 实际使用示例

### 7.1 基础用法：创建流并执行内存拷贝

参考仓库中测试代码: `/home/shanfeng/linux-ddk/musa/unittest/driver/`

```c
// 1. 创建流
MUstream stream;
muStreamCreate(&stream, 0);

// 2. 分配内存
MUdeviceptr d_A, d_B;
muMemAlloc(&d_A, size);
muMemAlloc(&d_B, size);

// 3. 异步内存拷贝 (通过流)
muMemcpyHtoDAsync(d_A, h_A, size, stream);
muMemcpyHtoDAsync(d_B, h_B, size, stream);

// 4. 同步等待
muStreamSynchronize(stream);

// 5. 销毁
muMemFree(d_A);
muMemFree(d_B);
muStreamDestroy(stream);
```

**调用链追踪**:
```
muStreamCreate
  → muapiStreamCreate → muapiStreamCreateWithPriority
    → InitPlatform → TlsCtxTop → pContext->CreateStream(flags, priority)
      → new Stream(ctx, flags, priority) → Stream::Init()
        → 创建队列 + 命令池 + 信号量 + 启动 AsyncSubmit/AsyncWait 线程

muMemcpyHtoDAsync
  → muapiMemcpyHtoDAsync_v2
    → pStream->CmdCopyMemory(node, false) [通过 GeneralMemcpy 路径]
      → ResolveDependencyAndQueueCommand(command, stream, blocking=false)
        → RecordDependency(defaultStream->LastCommand)
        → stream->QueueCommand(command)
          → m_CommandList.push_back(command)

muStreamSynchronize
  → muapiStreamSynchronize
    → pStream->Synchronize()
      → (非默认流) → WaitFinish()
        → LastCommand()->Wait()
          → Schedule + isFinished 检查
          → 阻塞直到 AsyncWait 线程标记为 completed
```

### 7.2 图捕获与重放

```c
// 捕获阶段
MUgraph graph;
muStreamBeginCapture(stream, MU_STREAM_CAPTURE_MODE_GLOBAL);
  // ... 在流上执行一系列操作 ...
  muMemcpyHtoDAsync(d_dst, h_src, size, stream);
  muLaunchKernel(stream, func, grid, block, ...);
muStreamEndCapture(stream, &graph);

// 实例化
MUgraphExec graphExec;
muGraphInstantiate(&graphExec, graph, ...);

// 重放
muStreamLaunchGraph(stream, graphExec);

// 清理
muGraphExecDestroy(graphExec);
muGraphDestroy(graph);
```

### 7.3 事件同步

```c
MUevent event;
muEventCreate(&event, 0);

// 流 A 上记录事件
muStreamWaitEvent(streamB, event, 0);  // 流 B 等待事件
muStreamAddCallback(streamA, callback, userData, 0);
muEventRecord(event, streamA);         // 流 A 记录事件

muStreamSynchronize(streamB);
muEventDestroy(event);
```

### 7.4 统一内存 (Managed Memory) 操作

```c
MUdeviceptr d_ptr;
muMemAllocManaged(&d_ptr, size, MU_MEM_ATTACH_GLOBAL);

// CPU 端可直接读写 (通过 page fault 迁移)
memcpy((void*)d_ptr, data, size);

// GPU 端通过流操作
muMemcpyDtoDAsync(d_dst, d_ptr, size, stream);
muStreamSynchronize(stream);
```

### 7.5 仓库中的真实使用位置

| 位置 | 用途 |
|------|------|
| `musa/unittest/driver/Entry_Point_Access/muCommandAccessors.cpp` | 测试所有流 API 的基本调用 |
| `musa/unittest/driver/Memory/` | 测试内存分配+流式拷贝 |
| `musa/unittest/driver/Graph/` | 测试图捕获与重放 |
| `musa/unittest/driver/` | 测试 StreamCreate、StreamSynchronize、StreamQuery 等 |

---

## 8. 关键设计决策

### 8.1 双线程模型 (Submit + Wait)

```
┌────────────┐    ┌────────────┐
│ AsyncSubmit │    │  AsyncWait  │
│  线程       │    │  线程       │
│             │    │             │
│ 从          │    │ 从          │
│ m_CommandList│───>│ m_InflightList│
│ 取出命令     │    │ 取出命令      │
│ Build+Submit│    │ WaitFinish  │
│ →Inflight   │    │ →Release    │
└────────────┘    └────────────┘
```

- **Submit 线程**负责: 从命令列表取出 → Build(构建 HAL 命令缓冲) → SubmitToQueue(提交到 GPU)
- **Wait 线程**负责: 等待已提交命令完成 → 执行 Postprocess(回收资源) → 更新状态
- **优势**: 调用线程 (User Thread) 只做命令入队和轻量同步，不阻塞在 GPU 执行上

### 8.2 命令合并 (Merge) vs 不合并

| 命令类型 | 支持合并 | 原因 |
|----------|----------|------|
| DispatchCommand | ✅ | 相同引擎的相邻 Kernel 可批处理 |
| AsyncMemcpyCommand | ✅ | 相同引擎的 DMA 可批处理 |
| MemsetCommand | ✅ | 填充操作可合并 |
| SyncMemcpyCommand | ❌ | 需要立即同步，强制刷新管线 |
| GraphCommand | ❌ | 图执行语义要求独立提交 |
| BarrierCommand | ❌ | 屏障语义要求独立提交 |

### 8.3 信号量架构选择

```
               Timeline Semaphore              Hardware Semaphore
               ┌─────────────────┐            ┌─────────────────┐
 优势          │ 精确值等待        │           │ 低延迟, 硬件原生  │
               │ 跨队列/跨设备     │            │ GPU-CPU 共享     │
               │ 值单调递增        │            │                  │
               ├─────────────────┤            ├─────────────────┤
 劣势          │ 依赖驱动支持      │           │ 不支持跨设备      │
               │ 某些场景延迟较高   │            │ 值可能回绕        │
               └─────────────────┘            └─────────────────┘

选择逻辑 (GetPreferredSemaphoreType):
  EnableUserQueue(engine) ? Hardware : Timeline
  → 仅在支持 engine sync 的设备上使用硬件信号量
```

---

## 9. 可观测性 (MUPTI 追踪)

Stream 的每个关键操作都有 MUPTI 回调:
```
muapiStreamCreate          → MUPTI_DRIVER_TRACE_CBID_muStreamCreate
muapiStreamSynchronize     → MUPTI_DRIVER_TRACE_CBID_muStreamSynchronize
muapiStreamQuery           → MUPTI_DRIVER_TRACE_CBID_muStreamQuery
```

命令级别的追踪通过 `Command::RecordMUptiActivity()` 实现:
- `MarkKernelQueued` / `MarkKernelSubmitted` / `MarkKernelBegin` / `MarkKernelEnd`
- 仅在非 `supportEngineSync` 的旧架构上启用

---

## 10. 潜在问题与注意事项

### 10.1 线程安全

- `m_CommandList` 通过 `m_SubmitMtx` 互斥锁保护
- `m_InflightList` 通过 `m_WaitMtx` 互斥锁保护
- `m_LastCommand` 设置时持有锁，但 `GetLastCommand()` 也需锁保护
- `m_AsyncCount` 使用 `atomic`，但 load + add 非原子组合 (注释已知)

### 10.2 资源释放顺序

`Stream::~Stream()`:
```
1. 同步默认流 (如果是非阻塞流)
2. 同步屏障流
3. 停止 Submit 线程 / Wait 线程
4. 等待线程池中任务完成
5. 清空所有命令缓冲列表
6. 释放溢出内存
```

> **注意**: 析构函数中调用了 `Platform::Get().GetExecutor().QueueTask([]{})` 并等待，这可能在高负载时阻塞。

### 10.3 错误粘性

一旦 `m_LastError` 被设置为非成功值，后续入队的命令都会被拒绝:
```cpp
// QueueCommand()
status = GetLastError();  // 如果有 sticky error，直接返回
if (status == MUSA_SUCCESS) { ... 入队 ... }
```

这意味着一旦流遇到错误，需要销毁并重新创建流才能继续使用。

---

## 11. 附录：Stream API 全表

| API | Driver 函数 | Core 路径 | 是否阻塞 |
|-----|------------|-----------|----------|
| `muStreamCreate` | `muapiStreamCreate` | `Context::CreateStream` | 否 |
| `muStreamCreateWithPriority` | `muapiStreamCreateWithPriority` | `Context::CreateStream` | 否 |
| `muStreamDestroy` | `muapiStreamDestroy` | `Context::DestroyStream` | 是 (LockedWait) |
| `muStreamSynchronize` | `muapiStreamSynchronize` | `Stream::Synchronize` | 是 |
| `muStreamQuery` | `muapiStreamQuery` | `Stream::Query` | 否 |
| `muStreamWaitEvent` | `muapiStreamWaitEvent` | `Stream::CmdWaitEvent` | 取决于 blocking |
| `muStreamAddCallback` | `muapiStreamAddCallback` | `Stream::CmdExecCallback` | 否 |
| `muStreamBeginCapture` | `muapiStreamBeginCapture` | `Stream::BeginCapture` | 否 |
| `muStreamEndCapture` | `muapiStreamEndCapture` | `Stream::EndCapture` | 否 |
| `muStreamWriteValue32` | `muapiStreamWriteValue32` | `Stream::CmdMemoryWaitWrite` | 否 |
| `muStreamBatchMemOp` | `muapiStreamBatchMemOp_v2` | `Stream::CmdMemoryWaitWrite` | 否 |
| `muStreamAttachMemAsync` | `muapiStreamAttachMemAsync` | `Stream::CmdAttachMemory` | 否 |
DOCEND
echo "Document 18 written: $(wc -l < /home/shanfeng/linux-ddk/musa/doc/memory_api/18_Stream_subsystem.md) lines"