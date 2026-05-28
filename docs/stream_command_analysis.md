# MUSA Stream & Command 子系统深度分析

> **📖 相关文档**: [memory_analysis.md](memory_analysis.md) (架构总览) | [memory_api_deep_analysis.md](memory_api_deep_analysis.md) (API流程) | [muMemAlloc_vs_muMemAllocAsync.md](muMemAlloc_vs_muMemAllocAsync.md) (对比) | [pooling_analysis.md](pooling_analysis.md) (池化) | [decision_logic.md](decision_logic.md) (决策分支)

**生成时间**: 2026-05-19
**项目**: MUSA User-Mode GPU Driver
**分析范围**: Stream 生命周期、命令队列、双线程模型、Command 状态机、依赖管理、同步机制

---

## 目录

1. [Stream 架构总览](#1-stream-架构总览)
2. [Stream 初始化：引擎资源与双线程](#2-stream-初始化引擎资源与双线程)
3. [命令生命周期：从 API 调用到 GPU 执行](#3-命令生命周期从-api-调用到-gpu-执行)
4. [ResolveDependencyAndQueueCommand：依赖解析核心](#4-resolvedependencyandqueuecommand依赖解析核心)
5. [AsyncSubmit：提交线程详解](#5-asyncsubmit提交线程详解)
6. [Command Merge：命令合并优化](#6-command-merge命令合并优化)
7. [AsyncWait：等待线程详解](#7-asyncwait等待线程详解)
8. [Stream 销毁与同步](#8-stream-销毁与同步)
9. [完整时序图：LaunchKernel → GPU 执行](#9-完整时序图launchkernel--gpu-执行)
10. [设计洞察与总结](#10-设计洞察与总结)

---

## 1. Stream 架构总览

### 1.1 Stream 是什么？

```
Stream = GPU 命令队列 + 双线程执行引擎 + 引擎资源池 + 依赖管理 + 同步原语
```

Stream 是 MUSA 中最核心的执行抽象。它不仅仅是 CUDA Stream 语义的实现（操作按序执行），更是一个**完整的异步执行引擎**：每个 stream 拥有自己的提交线程和等待线程，独立于调用者线程运行。

### 1.2 核心数据结构

```
┌──────────────────────────────────────────────────────────────────────┐
│                              Stream                                   │
├──────────────────────────────────────────────────────────────────────┤
│  m_SubmitThread     → 提交线程：Build + Merge + Submit                │
│  m_WaitThread       → 等待线程：Wait semaphore + Complete             │
│                                                                      │
│  m_CommandList      → 待处理命令队列 (API 线程 push)                   │
│  m_MergingList      → 正在合并的命令组                                  │
│  m_InflightList     → 已提交、GPU 执行中的命令                          │
│  m_WaitingList      → 等待线程正在等待完成的命令                         │
│                                                                      │
│  m_LastCommand      → stream 中最后一个命令 (构建链式依赖)              │
│  m_CurrentDependencies → 当前跨 stream 依赖 (如 peer memcpy)            │
│                                                                      │
│  m_EngineResources[] → 每引擎资源:                                     │
│    queue            → Hal::IQueue (GPU 硬件队列)                       │
│    cmdPool          → Hal::ICmdPool (命令缓冲区池)                     │
│    cmdBufferLists[] → 预分配的 Hal::ICmdBuffer 池                      │
│                                                                      │
│  m_TimelineSemaphore  → 命令间同步 (Timeline 类型)                      │
│  m_InternalTimelineSemaphore → 命令内同步                               │
│  m_HardwareSemaphore  → 硬件加速同步 (PH 平台)                          │
│                                                                      │
│  m_CaptureGraph       → CUDA Graph 捕获模式                            │
│  m_CaptureStatus      → ACTIVE / INVALIDATED / NONE                   │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.3 Stream 类型

| 类型 | Flags | 行为 |
|------|-------|------|
| **Default Stream** | `MU_STREAM_DEFAULT` | 隐式同步所有 blocking stream |
| **Blocking Stream** | (默认) | 依赖 default stream |
| **Non-Blocking Stream** | `MU_STREAM_NON_BLOCKING` | 不参与隐式同步 |
| **Barrier Stream** | (内部) | 依赖所有其他 stream 的最后一个命令 |

### 1.4 隐式同步规则 (ResolveDependencyAndQueueCommand)

```
命令插入时，自动建立跨 stream 依赖：

if (目标 == Default Stream):
    → 依赖所有 blocking stream 的 LastCommand
elif (目标 == Barrier Stream):
    → 依赖所有其他 stream 的 LastCommand  
elif (目标 == Blocking Stream && 目标 != Default):
    → 依赖 Default Stream 的 LastCommand
elif (目标 == Non-Blocking Stream):
    → 不添加隐式依赖

所有 stream:
    → 依赖 Barrier Stream 的 LastCommand (如果存在)
```

---

## 2. Stream 初始化：引擎资源与双线程

### 2.1 Stream::Init() 流程

```
Stream::Init()
  │
  ├─ ① 遍历所有 GPU 引擎 (Compute, DMA, CE, TDM, etc.)
  │    for each engine:
  │    │
  │    ├─ GetQueueFamilyIndex(engine)
  │    │   └─ 查询该引擎的队列族索引 (如果引擎不存在则跳过)
  │    │
  │    ├─ Hal::CreateQueue()           // 创建 GPU 硬件队列
  │    │   参数: familyIndex, priority(high/medium/low),
  │    │         userQueue (用户态提交, 门铃通知)
  │    │   返回: Hal::IQueue*
  │    │
  │    ├─ Hal::CreateCmdPool()         // 创建命令缓冲区池
  │    │   参数: queueFamilyIndex, allocSize
  │    │   返回: Hal::ICmdPool* (内含 M3D CmdAllocator + ExternalAllocator 池)
  │    │
  │    └─ 预分配 Hal::ICmdBuffer 池
  │        为每个 cmdBufferList 创建 s_CmdBufferListSize 个 CmdBuffer
  │        (Universal engine 有更多列表，其他引擎 1 个)
  │
  ├─ ② 创建 Timeline Semaphores (命令间同步)
  │    ├─ m_TimelineSemaphore (Hal::SemaphoreType::Timeline)
  │    └─ m_InternalTimelineSemaphore
  │
  ├─ ③ 创建 Hardware Semaphores (如果硬件支持引擎同步)
  │    只在 PH 平台 (ipProperties.supportEngineSync == true)
  │    硬件信号量比软件 timeline 延迟更低
  │
  ├─ ④ 预分配 Surface 对象池 (8 个)
  │    for (i = 0; i < 8; i++):
  │        new Surface + Init → m_pSurfaceList
  │
  ├─ ⑤ 启动 Submit 线程
  │    m_SubmitThread = std::thread(&Stream::AsyncSubmit, this)
  │
  └─ ⑥ 启动 Wait 线程
       m_WaitThread = std::thread(&Stream::AsyncWait, this)
```

**为什么每个 Stream 有自己的线程？**

- CUDA 语义要求 stream 内操作按序执行
- GPU 提交和完成等待都是阻塞操作 (ioctl 系统调用)
- 用独立线程处理，用户的 API 调用线程不会被阻塞
- `blocking=true` 的 API 在命令入队后调用 `command->Wait()` 实现同步

---

## 3. 命令生命周期：从 API 调用到 GPU 执行

### 3.1 整体流程

```
API 调用 (用户线程)         提交线程              GPU              等待线程
    │                        │                    │                  │
 ┌──┴──────────────────┐     │                    │                  │
 │ API Wrapper          │     │                    │                  │
 │ → Driver Entry      │     │                    │                  │
 │ → CmdXXX 方法       │     │                    │                  │
 │ → Create GraphNode  │     │                    │                  │
 │ → new Command        │     │                    │                  │
 │                      │     │                    │                  │
 │ ResolveDependency    │     │                    │                  │
 │ AndQueueCommand()    │     │                    │                  │
 │  ├─ 建立隐式依赖      │     │                    │                  │
 │  ├─ RecordDependency │     │                    │                  │
 │  └─ QueueCommand ────┼──>  │                    │                  │
 │     push到m_CommandList   │                    │                  │
 │     notify m_SubmitCv     │                    │                  │
 │                      │     │                    │                  │
 │ (如果 blocking=true) │  ┌──┴──────────────────┐ │                  │
 │   command->Wait()    │  │ AsyncSubmit 线程     │ │                  │
 │   阻塞等待信号量      │  │                     │ │                  │
 │                      │  │ 从 CommandList 取命令  │                  │
 │                      │  │                     │ │                  │
 │                      │  │ FilterDependency()  │ │                  │
 │                      │  │ Build()             │ │                  │
 │                      │  │  └─ GetHalCmdBuffer │ │                  │
 │                      │  │  └─ CmdBufferBegin  │ │                  │
 │                      │  │  └─ Wait/Barrier 编码│ │                  │
 │                      │  │  └─ GPU 命令编码     │ │                  │
 │                      │  │  └─ Signal 编码      │ │                  │
 │                      │  │                     │ │                  │
 │                      │  │ Merge (如果需要)     │ │                  │
 │                      │  │  多命令合并到一次提交  │ │                  │
 │                      │  │                     │ │                  │
 │                      │  │ Submit()            │ │                  │
 │                      │  │  └─ CmdBufferEnd    │ │                  │
 │                      │  │  └─ SubmitToQueue ──┼──> IQueue::Submit  │
 │                      │  │                     │    → KMD IOCTL    │
 │                      │  │                     │    → GPU 开始执行  │
 │                      │  │                     │                   │
 │                      │  │ 移入 m_InflightList  │                   │
 │                      │  │ notify m_WaitCv     │                   │
 │                      │  │                     │          ┌────────┴──────────┐
 │                      │  │                     │          │ AsyncWait 线程     │
 │                      │  │                     │          │                   │
 │                      │  │                     │          │ 从 InflightList   │
 │                      │  │                     │          │ 移入 WaitingList   │
 │                      │  │                     │          │                   │
 │                      │  │                     │          │ Wait Semaphore    │
 │                      │  │                     │          │  └─ Timeline::Wait│
 │                      │  │                     │          │  └─ HW Sem::Wait  │
 │                      │  │                     │          │                   │
 │                      │  │                     │  GPU 完成 │ ← semaphore signaled
 │                      │  │                     │          │                   │
 │                      │  │                     │          │ Postprocess()     │
 │                      │  │                     │          │ SetStatus(completed)
 │                      │  │                     │          │ AsyncCount--      │
 │                      │  │                     │          │                   │
 │  (blocking API 返回)  │  │                     │          │ ReleaseResources()│
 │  ← Wait() 返回        │  │                     │          │ (通过 Executor    │
 │                       │  │                     │          │  异步释放)         │
```

### 3.2 Command 状态机

```
        new Command()
             │
             ▼
        ┌─────────┐
        │ created │  (状态=4)
        └────┬────┘
             │ QueueCommand()
             ▼
        ┌─────────┐
        │ queued  │  (状态=3) — 在 CommandList 中等待提交线程处理
        └────┬────┘
             │ AsyncSubmit::buildCommand()
             │  → Build()
             ▼
        ┌─────────┐
        │ built   │  (状态=2) — GPU 命令已编码到 CmdBuffer
        └────┬────┘
             │ AsyncSubmit::submitMergingList()
             │  → Submit()
             ▼
        ┌──────────┐
        │ submitted│  (状态=1) — 已提交到 GPU 队列
        └────┬─────┘
             │ AsyncWait::Wait semaphore
             │  → GPU 执行完成
             ▼
        ┌───────────┐
        │ completed │  (状态=0) — 成功完成
        └───────────┘
             │ 如果出错:
             ▼
        ┌───────────┐
        │  error    │  (状态=-1) — 执行出错
        └───────────┘
```

---

## 4. ResolveDependencyAndQueueCommand：依赖解析核心

这是 Stream 和 Command 之间最重要的桥接函数：

```cpp
// context.cpp:1859
MUresult Context::ResolveDependencyAndQueueCommand(
    std::shared_ptr<Command>&& command,
    Stream* pStream,
    bool blocking)
{
    // ===== 第1步: PFM 序列化锁 =====
    // 如果性能监控(PFM)已启用，需要串行化命令提交
    if (pfmEnabled) {
        while (GetParentDevice()->GetPfmSerialLock().test_and_set(...)) {
            sleep(1ms);
        }
    }

    // ===== 第2步: 隐式跨 Stream 依赖 =====
    if (pStream == m_DefaultStream) {
        // Default Stream: 依赖所有 blocking stream 的最后一个命令
        ctxCrit->for_each(stream):
            if (stream != default && stream != barrier && !nonBlocking):
                command->RecordDependency(stream->LastCommand());
    }
    else if (pStream == m_BarrierStream) {
        // Barrier Stream: 依赖所有其他 stream
        ctxCrit->for_each(stream):
            if (stream != barrier):
                command->RecordDependency(stream->LastCommand());
    }
    else if (!pStream->IsNonBlocking()) {
        // 普通 Blocking Stream: 依赖 Default Stream
        command->RecordDependency(m_DefaultStream->LastCommand());
    }

    // 所有 stream: 依赖 Barrier Stream
    if (barrierCommand && barrierCommand->GetStatus() > completed) {
        command->RecordDependency(barrierCommand);
    }

    // ===== 第3步: 显式依赖 (Peer memcpy) =====
    for (auto& dep : pStream->GetCurrentDependencies()) {
        command->RecordDependency(dep);
    }
    pStream->SetCurrentDependencies({});  // 清空

    // ===== 第4步: 入队到 Stream 的 CommandList =====
    MUresult status = pStream->QueueCommand(std::move(command));

    // ===== 第5步: 阻塞等待 (如果 API 是同步的) =====
    if (status == MUSA_SUCCESS && (blocking || pfmEnabled)) {
        status = commandRef->Wait();
    }

    // ===== 第6步: 释放 PFM 锁 =====
    if (pfmEnabled) {
        GetParentDevice()->GetPfmSerialLock().clear(...);
    }
    return status;
}
```

**依赖如何工作？**

```
RecordDependency(producerCommand):
    command->m_ExecutionDependencies.push_back(producerCommand)

在 Build() 阶段:
    for (auto& dep : m_ExecutionDependencies):
        cmdBuffer->CmdWaitSemaphore(dep->GetTimelineSemaphore(),
                                     dep->GetSignalSemaphoreValue())
    // → 命令在 GPU 上会被阻塞，直到所有依赖的 semaphore 到达指定值
```

---

## 5. AsyncSubmit：提交线程详解

### 5.1 核心循环

```cpp
void Stream::AsyncSubmit() {
    lock(m_SubmitMtx);
    while (true) {
        // ① 等待唤醒条件:
        //    - CommandList 有命令 或
        //    - MergingList 需要提交 或
        //    - StopToken (析构)
        m_SubmitCv.wait(submitLock, [] {
            return !m_CommandList.empty() || stopMerging() || m_SubmitStopToken;
        });

        if (m_SubmitStopToken) break;

        // ② 先尝试提交 MergingList (如果有可提交的)
        if (stopMerging()) {
            unlock → submitMergingList() → lock
        }

        // ③ 从 CommandList 取一个命令 Build
        if (!m_CommandList.empty()) {
            command = pop_front(m_CommandList)
            unlock → buildCommand(command) → lock
        }
    }
}
```

### 5.2 Build 一个命令

```
buildCommand(command):
  │
  ├─ FilterDependency()            // 清理已完成的依赖
  │
  ├─ 决定 Timeline Value:
  │    if (可合并到当前 MergingList):
  │        command->SetSignalSemaphoreValue(primary->GetSignalSemaphoreValue())
  │        // 合并的命令共享同一个信号量值
  │    else:
  │        command->SetSignalSemaphoreValue(++m_TimelineValue)
  │        // 自增 timeline 值
  │
  ├─ 如果不可合并 → 先提交当前 MergingList
  │    submitMergingList()
  │
  ├─ command->Build(m_MergingList)
  │    └─ 具体子类实现 (DispatchCommand, MemsetCommand, MemcpyCommand...)
  │       └─ 编码 GPU 命令到 Hal::ICmdBuffer
  │
  ├─ command->SetStatus(Command::Status::built)
  │
  └─ m_MergingList.push_back(command)
```

### 5.3 提交 MergingList

```
submitMergingList():
  │
  ├─ EngineSubmissionSchedule(engine)
  │   // 检查是否超过 in-flight 限制 (默认 3 / 非用户队列 2)
  │   // 如果满了 → 阻塞等待
  │
  ├─ 设置 submission ID
  │    if (pfm enabled 或 无 user queue):
  │        submissionId = UpdateGlobalSubId()  // 全局递增
  │    else:
  │        submissionId = userQueueSubmissionId
  │
  ├─ 通知 MUpti (profiler):
  │    for each merged command:
  │        if Dispatch: MarkKernelSubmitted, AssignKernelToKick
  │        if Memcpy/Memset: AssignSubmissionToCorrelation
  │
  ├─ m_MergingList.front()->Submit()
  │    └─ 具体子类实现
  │       └─ CmdBuffer::End() → IQueue::Submit() → KMD IOCTL
  │
  ├─ 标记所有合并命令为 submitted
  │    for each: SetLastError(status), SetStatus(submitted)
  │
  └─ 移入 m_InflightList
       lock(m_WaitMtx)
       m_InflightList.splice(end, m_MergingList)
       unlock → notify m_WaitCv
```

---

## 6. Command Merge：命令合并优化

### 6.1 为什么要合并？

多个小命令 (如连续的 memset) 可以编码到**同一个** CmdBuffer 中，然后一次提交。这减少了：
- KMD IOCTL 调用次数
- GPU 调度开销
- 信号量数量

### 6.2 合并条件

```cpp
// command.h:109
virtual bool CanMergeTo(const list<Command>& mergingList) const {
    return SupportMerge() &&                    // 该类型支持合并
           m_ExecutionDependencies.empty() &&    // 没有跨 stream 依赖
           !mergingList.empty() &&              // MergingList 非空
           mergingList.back()->SupportMerge() && // 前一个也支持合并
           mergingList.back()->GetEngine() == GetEngine();  // 同一引擎
}
```

支持合并的命令类型: Dispatch, Memset, AsyncMemcpy, MemoryAtomic, MemoryAtomicValue, MemoryTransfer

**不支持合并**: SyncMemcpy, Graph, Barrier (有特殊依赖需求)

### 6.3 合并时机

```
停止合并的条件 (stopMerging):

1. 追踪捕获模式 (MT Trace Capture):
   if (traceMergeThreshold > 0):
       MergingList.size() >= threshold || CommandList 为空

2. 正常模式:
   - 引擎就绪 (inflight 未满) 或
   - MergingList >= 32 (最大合并数限制)
```

### 6.4 合并图解

```
不合并:                          合并后:
                                 
Memset A: [CmdBuffer #1]        ┌─────────────────────┐
  → Submit → IOCTL              │ CmdBuffer #1         │
                                │  ├─ Wait Semaphore   │
Memset B: [CmdBuffer #2]        │  ├─ FillMemory A     │
  → Submit → IOCTL              │  ├─ FillMemory B     │
                                │  ├─ FillMemory C     │
Memset C: [CmdBuffer #3]        │  └─ Signal Semaphore │
  → Submit → IOCTL              └─────────────────────┘
                                   → 一次 Submit → 一次 IOCTL

3 次 IOCTL  →  1 次 IOCTL
3 个信号量    →  1 个信号量
```

---

## 7. AsyncWait：等待线程详解

### 7.1 核心循环

```cpp
void Stream::AsyncWait() {
    lock(m_WaitMtx);
    while (true) {
        // ① 等待 InflightList 非空 或 StopToken
        m_WaitCv.wait(waitLock, [] {
            return !m_InflightList.empty() || m_WaitStopToken;
        });

        if (m_WaitStopToken) break;

        // ② 从 InflightList 取 primary 命令到 WaitingList
        //    跳过 secondary (它们共享 primary 的状态)
        m_WaitingList.splice(end, m_InflightList,
            find_if(..., [](cmd) { return cmd->GetMergeLevel() == primary; }));

        unlock → 执行等待 → lock
    }
}
```

### 7.2 等待一个 Primary 命令

```
for each primary command in WaitingList:

  ① 检查错误状态
     if (m_WaitingList 中任何命令已经有错误):
         commandStatus = 该错误

  ② 等待信号量 (仅 primary)
     if (preferred == Hardware):
         m_HardwareSemaphore->Wait(value)   // 硬件信号量 (PH 平台)
         m_TimelineSemaphore->Signal(value)  // 同时更新软件 timeline
     else:
         m_TimelineSemaphore->Wait(value)    // 软件 Timeline 信号量

  ③ 引擎错误检查
     EngineSubmissionFeedback(engine)
     通过 QueryRobustInfo 检查 GPU 引擎异常

  ④ 输出检查 (kernel printf)
     if (Dispatch): static_cast<DispatchCommand&>(*cmd).OutputPrintBuffer(...)
     检查 kernel 中的 printf 输出和断言

  ⑤ 处理结果
     for each command in WaitingList:
         if (commandStatus != SUCCESS):
             if (第一个错误):
                 m_LastError = commandStatus
                 ErrorHandler()  // 生成错误 dump
             打印错误日志 (kernel 名、位置)
         else:
             打印完成日志
         Postprocess()            // 具体子类清理
         SetStatus(completed/error)
         m_AsyncCount--           // 释放异步容量

  ⑥ 异步释放资源 (通过 Executor 线程池)
     Platform::GetExecutor().QueueTask([postprocessList] {
         for each: command->ReleaseResources()
         // 归还 CmdBuffer 到池中
         // 释放 Timestamp 内存
     });
```

### 7.3 Timeline Semaphore 机制

```
Timeline Semaphore = 单调递增的信号量

命令1: Signal(value=1)        → GPU 完成后触发
命令2: Wait(value=1)          → 等待命令1完成，然后执行
       Signal(value=2)        → 完成后触发
命令3: Wait(value=2)          → 等待命令2完成
       ...

这确保:
- 命令2 在 GPU 上不会先于命令1 执行 (即使同引擎)
- 跨引擎的命令也能正确排序
- 合并命令共享同一个 signal value
```

---

## 8. Stream 销毁与同步

```cpp
Stream::~Stream() {
    // ① 如果是 blocking stream:
    //    等待 default stream 完成 (因为它可能依赖本 stream)
    if (!IsNonBlocking() && this != defaultStream):
        defaultStream->WaitFinish()

    // ② 等待 barrier stream 完成
    if (this != barrierStream && this != defaultStream):
        barrierStream->WaitFinish()

    // ③ 停止提交线程
    if (m_SubmitThread.joinable()):
        lock → m_SubmitStopToken = true
        unlock → notify m_SubmitCv
        m_SubmitThread.join()

    // ④ 停止等待线程
    if (m_WaitThread.joinable()):
        lock → m_WaitStopToken = true
        unlock → notify m_WaitCv
        m_WaitThread.join()

    // ⑤ 等待 Executor 清空
    Platform::GetExecutor().QueueTask([]{}).wait()

    // ⑥ 释放所有引擎资源
    for each engineResource:
        cmdBufferLists.clear()
        cmdPool.Reset()
        queue.Reset()

    // ⑦ 释放 Surface 对象
    // ⑧ 释放 Spill Memory
}
```

---

## 9. 完整时序图：LaunchKernel → GPU 执行

```
用户线程            Driver层            Stream             提交线程           Wait线程          GPU/硬件队列
  │                  │                   │                   │                │                 │
  │ muLaunchKernel   │                   │                   │                │                 │
  │─────────────────>│                   │                   │                │                 │
  │                  │ InitPlatform      │                   │                │                 │
  │                  │ TlsCtxTop         │                   │                │                 │
  │                  │ InfoStream        │                   │                │                 │
  │                  │                   │                   │                │                 │
  │                  │ CreateKernelNode  │                   │                │                 │
  │                  │  (GraphKernelNode)│                   │                │                 │
  │                  │                   │                   │                │                 │
  │                  │ CmdLaunchKernel──>│                   │                │                 │
  │                  │                   │                   │                │                 │
  │                  │                   │ [三路分支]        │                │                 │
  │                  │                   │ Capture? No      │                │                 │
  │                  │                   │                   │                │                 │
  │                  │                   │ new DispatchCmd──>│ (构造函数)     │                │
  │                  │                   │                   │                │                 │
  │                  │ ResolveDependency │                   │                │                 │
  │                  │ AndQueueCommand() │                   │                │                 │
  │                  │  ├─ DefaultStream?│                   │                │                 │
  │                  │  │  → RecordDeps  │                   │                │                 │
  │                  │  │  (blocking     │                   │                │                 │
  │                  │  │   streams)     │                   │                │                 │
  │                  │  ├─ Barrier?      │                   │                │                 │
  │                  │  └─ CurDeps       │                   │                │                 │
  │                  │                   │                   │                │                 │
  │                  │ QueueCommand()    │                   │                │                 │
  │                  │──────────────────>│                   │                │                 │
  │                  │                   │ push CommandList  │                │                 │
  │                  │                   │ notify SubmitCv──┼───────────────>│                 │
  │                  │                   │                   │  唤醒           │                │
  │                  │                   │                   │                │                 │
  │                  │ (blocking=false) │                   │                │                 │
  │                  │<──OK─────────────│                   │                │                 │
  │<─OK──────────────│                   │                   │                │                 │
  │                  │                   │                   │                │                 │
  │                  │                   │                   │ pop CommandList │                 │
  │                  │                   │                   │                │                 │
  │                  │                   │                   │ FilterDependency│                 │
  │                  │                   │                   │ (清理已完成dep)  │                 │
  │                  │                   │                   │                │                 │
  │                  │                   │                   │ SetSignalValue  │                 │
  │                  │                   │                   │ (++Timeline)    │                 │
  │                  │                   │                   │                │                 │
  │                  │                   │                   │ Build()         │                 │
  │                  │                   │                   │  ├─ GetHalCmdBuf│                 │
  │                  │                   │                   │  ├─ Begin       │                 │
  │                  │                   │                   │  ├─ WaitSemaphore│                 │
  │                  │                   │                   │  │  (依赖前序)   │                 │
  │                  │                   │                   │  ├─ Timestamp   │                 │
  │                  │                   │                   │  ├─ 编码 Launch │                 │
  │                  │                   │                   │  │  kernel params│                │
  │                  │                   │                   │  │  grid/block   │                │
  │                  │                   │                   │  ├─ Timestamp   │                 │
  │                  │                   │                   │  └─ (IMPLICIT   │                 │
  │                  │                   │                   │      END)      │                 │
  │                  │                   │                   │ → SetStatus(built)               │
  │                  │                   │                   │ → push MergingList               │
  │                  │                   │                   │                │                 │
  │                  │                   │                   │ [stopMerging?] │                 │
  │                  │                   │                   │  YES           │                 │
  │                  │                   │                   │                │                 │
  │                  │                   │                   │ submitMerging()│                 │
  │                  │                   │                   │  ├─ Schedule   │                 │
  │                  │                   │                   │  │  (inflight限)│                 │
  │                  │                   │                   │  ├─ PFM/MUpti  │                 │
  │                  │                   │                   │  ├─ Submit()   │                 │
  │                  │                   │                   │  │  ├─ CmdBuf  │                 │
  │                  │                   │                   │  │  │  .End()  │                 │
  │                  │                   │                   │  │  ├─ Signal  │                 │
  │                  │                   │                   │  │  └─ Submit──┼────────────────>│
  │                  │                   │                   │  │     toQueue  │   IOCTL → GPU  │
  │                  │                   │                   │  └─ SetStatus  │                 │
  │                  │                   │                   │     (submitted)│                 │
  │                  │                   │                   │                │                 │
  │                  │                   │                   │ splice to      │                 │
  │                  │                   │                   │ InflightList   │                 │
  │                  │                   │                   │ notify WaitCv──┼───────────────>│
  │                  │                   │                   │                │  唤醒           │
  │                  │                   │                   │                │                 │
  │                  │                   │                   │                │ 从 InflightList │
  │                  │                   │                   │                │ 取 primary cmd  │
  │                  │                   │                   │                │                 │
  │                  │                   │                   │                │ Wait Semaphore  │
  │                  │                   │                   │                │  (Timeline Wait)│
  │                  │                   │                   │                │  ← 阻塞等待     │
  │                  │                   │                   │                │                 │
  │                  │                   │                   │                │      ...        │
  │                  │                   │                   │                │  GPU 完成      │
  │                  │                   │                   │                │  Semaphore     │
  │                  │                   │                   │                │  Signaled      │
  │                  │                   │                   │                │<────────────────│
  │                  │                   │                   │                │                 │
  │                  │                   │                   │                │ ErrorCheck      │
  │                  │                   │                   │                │ PrintfCheck     │
  │                  │                   │                   │                │                 │
  │                  │                   │                   │                │ Postprocess     │
  │                  │                   │                   │                │ SetStatus       │
  │                  │                   │                   │                │  (completed)    │
  │                  │                   │                   │                │ AsyncCount--    │
  │                  │                   │                   │                │                 │
  │                  │                   │                   │                │ ReleaseResources│
  │                  │                   │                   │                │ (async via      │
  │                  │                   │                   │                │  Executor)      │
```

---

## 10. 设计洞察与总结

### 10.1 架构模式

| 模式 | 实现 | 目的 |
|-----|------|------|
| **双线程模型** | Submit Thread + Wait Thread | 异步化提交和等待，用户线程不阻塞 |
| **状态机驱动** | created→queued→built→submitted→completed | 清晰的命令生命周期 |
| **命令合并** | MergingList (最多 32 个) | 减少 KMD IOCTL 次数 |
| **Timeline Semaphore** | 单调递增信号量 | 流内+跨流依赖的 GPU 端实现 |
| **二级合并** | primary/secondary MergeLevel | primary 负责 Submit/Wait，secondary 共享状态 |
| **Inflight 限制** | 3 (user queue) / 2 (non-user) | 控制 GPU 队列深度 |
| **隐式同步** | default/blocking/barrier stream 自动建依赖 | CUDA 兼容的同步语义 |
| **资源池化** | Hal::ICmdPool + cmdBufferLists | 避免 CmdBuffer 频繁创建/销毁 |

### 10.2 关键设计决策

1. **为什么每个 Stream 有两个线程而不是一个？**
   - Submit 和 Wait 的瓶颈不同：Submit 卡在 IOCTL (KMD 提交)，Wait 卡在 semaphore (GPU 完成)
   - 分离后可以在等 GPU 完成的同时继续提交新命令 (pipeline 化)

2. **为什么要合并命令？**
   - 假设 100 个连续 memset：合并后只需 1 次 IOCTL 而非 100 次
   - KMD IOCTL 是上下文切换 → 高开销
   - 合并也有上限 (32) → 防止单个提交过大

3. **为什么有 Default/Barrier/Blocking/NonBlocking 四种类型？**
   - CUDA 语义要求：default stream 隐式同步所有，blocking stream 与 default 互相同步
   - Barrier stream 是内部实现：context 内的全局同步点
   - NonBlocking stream 不参与隐式同步 → 最高性能

### 10.3 已知 TODO

| 位置 | 内容 |
|------|------|
| `stream.cpp:517` | `// TODO(hongwei.liu): Refine this logic` — AsyncMemAlloc 逻辑待优化 |
| `stream.cpp:353,356` | 注释掉的 `CmdMemAlloc/CmdMemFree` 图节点展开 |
| `stream.h:238` | `thread_local t_CaptureMode` — 捕获模式使用 thread_local 存储 |
| `stream.cpp:1247` | `mergingListMaxSize = 32` — 硬编码的合并上限 |

### 10.4 线程安全

| 数据结构 | 保护机制 |
|---------|---------|
| `m_CommandList` | `m_SubmitMtx` (提交线程取出时加锁) |
| `m_MergingList` | 仅提交线程访问 (无竞争) |
| `m_InflightList` | `m_WaitMtx` (提交线程移入、等待线程取出) |
| `m_LastCommand` | `m_SubmitMtx` |
| `m_CurrentDependencies` | API 线程设置 → 提交线程清空 (单向) |
| `m_AsyncCount` | `std::atomic` |
| `engineResource.cmdBufferLists` | `engineResource.mtx` |

---

*本文档基于 musa 项目源码分析生成，commit: 9ba99a5d, branch: bugfix/sw-79049*
