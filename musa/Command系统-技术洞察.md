# MUSA Command 系统技术洞察

> 全面分析 `musa/src/musa/core/command/` 目录下的命令系统架构：基类设计、状态机、依赖解析、合并优化、生命周期管理、16 种命令类型、以及完整的"创建→排队→构建→提交→完成"流水线。

---

## 1. 总览

Command 系统是 MUSA 驱动中 GPU 操作的**统一抽象层**。所有需要提交到 GPU 的操作——内核启动、内存拷贝、清零、原子操作、同步等等——都被建模为一个 `Command` 子类实例。

```
用户 API 调用
  │
  ▼
Stream::CmdXxx()  ─→  new XxxCommand(stream, params)
  │                     │
  ▼                     ▼
Context::ResolveDependencyAndQueueCommand()
  │
  ├─ 依赖解析（Default Stream / Blocking Stream / Barrier Stream）
  │
  ▼
Stream::QueueCommand()
  │
  ├─ 推送 m_CommandList
  ├─ 通知 AsyncSubmit 线程
  │
  ▼
Stream::AsyncSubmit()  （提交线程）
  │
  ├─ MergeCheck → CanMergeTo()  ← 合并优化
  ├─ Build()                    ← 编码 GPU 命令到 cmd buffer
  ├─ Submit()                   ← 提交到 HAL Queue
  │   └─ SubmitToQueue()        ← semaphore wait/signal + kick
  │
  ▼
GPU 执行 → Postprocess() → ReleaseResources()
```

---

## 2. Command 基类

**文件**：`command.h` (336 行)、`command.cpp` (806 行)

### 2.1 状态机

```cpp
enum class Status : int32_t {
    error     = -1,    // 执行失败
    completed = 0,     // 执行完成
    submitted = 1,     // 已提交到 HAL Queue
    built     = 2,     // 已编码到 cmd buffer
    queued    = 3,     // 已入队（在 m_CommandList 中）
    created   = 4,     // 初始状态
};
```

**状态转换是单向的 CAS**（`SetStatus`）：

```
created ──→ queued ──→ built ──→ submitted ──→ completed
                                       │
                                       └──→ error （任意时刻可跳转）
```

### 2.2 核心成员变量

```cpp
class Command {
protected:
    std::atomic<Status>       m_Status;          // 原子状态
    Type                      m_Type;            // 命令类型（Dispatch/Memcpy/...）
    Device::Engine            m_Engine;          // 目标引擎（CDM/CE/TDM/UQ/DMA）
    bool                      m_SupportMerge;    // 是否支持合并
    MergeLevel                m_MergeLevel;      // primary 或 secondary
    Stream*                   m_ParentStream;    // 所属 stream
    uint32_t                  m_CorId;           // API correlation ID
    uint64_t                  m_SubId;           // 提交 ID（全局递增）

    // 依赖关系
    std::list<shared_ptr<Command>> m_ExecutionDependencies;  // 执行前必须完成的命令
    std::list<shared_ptr<Command>> m_SubmitDependencies;     // 提交时的依赖
    shared_ptr<Command>       m_PrevCommand;     // 同一 stream 上的前一个命令

    // 信号量
    Hal::ISemaphore*          m_pSignalTimelineSemaphore;   // Timeline semaphore
    Hal::ISemaphore*          m_pSignalHardwareSemaphore;   // Hardware semaphore
    uint64_t                  m_SignalSemaphoreValue;       // 信号值
    list<SemaphoreInfo>       m_WaitSemaphoreInfos;         // CPU 端等待列表
    list<SemaphoreInfo>       m_SubWaitSemaphoreInfos;      // GPU 端等待列表
    list<SemaphoreInfo>       m_SubSignalSemaphoreInfo;     // GPU 端信号列表

    // 命令缓冲区
    vector<HalPtr<ICmdBuffer>> m_pHalCmdBuffers;

    // 时间戳
    uint64_t m_QueuedTimestamp;
    uint64_t m_SubmittedTimestamp;
    Memory*  m_TimestampMem;                             // GPU 时间戳内存
    array<DevSize, 2>   m_TimestampGpuAddrs;             // begin/end GPU 地址
    array<uint64_t*, 2> m_TimestampCpuAddrs;             // begin/end CPU 地址

    // Profiling
    bool                  m_PfmEnabled;
    Hal::IPerfExperiment* m_PerfExperiment;

    // MUPTI / Graph
    MUpti::Context* m_ptiCtx;
    GraphNode*      m_pGraphNode;
    bool            m_IsFromGraph;
};
```

### 2.3 16 种命令类型

```cpp
enum class Type {
    Dispatch,               // 内核启动
    Barrier,                // GPU 屏障
    Callback,               // CPU 回调
    Memcpy,                 // 同步 memcpy
    AsyncMemcpy,            // 异步 memcpy
    Memset,                 // 内存清零
    Graph,                  // CUDA Graph 启动
    MemoryAtomic,           // 原子操作
    MemoryAtomicValue,      // 原子值返回
    MemoryWaitWrite,        // 写屏障
    Paging,                 // 页表更新
    Record,                 // event 记录
    SignalExternalSemaphore,// 外部信号量通知
    WaitExternalSemaphore,  // 外部信号量等待
    MemoryTransfer,         // 跨设备传输
    AccelBuild,             // RT 加速结构构建
    DispatchRay,            // 光线追踪调度
};
```

### 2.4 命令类型与引擎映射

| 命令类型 | 默认引擎 | 支持合并 |
|----------|---------|---------|
| Dispatch | CDM | ✅ |
| Memcpy | 多引擎（CE/DMA/TDM/CDM） | ❌ |
| AsyncMemcpy | 多引擎 | ✅ |
| Memset | 多引擎 | ✅ |
| Graph | Universal (UQ) | ❌ |
| MemoryAtomic | CDM | ✅ |
| MemoryAtomicValue | CDM | ✅ |
| MemoryTransfer | 多引擎 | ✅ |
| Barrier | 多引擎 | ❌ |
| Paging | 多引擎 | ❌ |

---

## 3. 依赖解析

### 3.1 依赖来源

**文件**：`context.cpp:1884` → `ResolveDependencyAndQueueCommand`

一个命令入队时，可能依赖以下源：

```
1. Default Stream 上的命令
   → 所有 blocking stream 的最后一个命令

2. Blocking Stream 上的命令
   → Default Stream 的最后一个命令

3. Barrier Stream 上的命令
   → 所有其他 stream 的最后一个命令

4. 同一 Stream 上的前一个命令
   → m_LastCommand (prevCommand)

5. Stream 的当前依赖
   → 用户通过 muStreamWaitEvent 等显式设置的依赖
```

### 3.2 依赖记录与过滤

```cpp
// RecordDependency: 记录执行依赖
void Command::RecordDependency(shared_ptr<Command>&& producer) {
    m_SubmitDependencies.push_back(producer);
    if (需要) {
        m_ExecutionDependencies.push_back(producer);  // 等待它完成才能 Build
    }
}

// FilterDependency: 移除已完成的依赖（条件编译优化）
void Command::FilterDependency() {
    for (auto iter = m_ExecutionDependencies.begin(); ...) {
        if (iter->GetStatus() == Status::completed) {
            iter = m_ExecutionDependencies.erase(iter);  // 已完成的不用等了
        }
    }
}
```

### 3.3 Build 阶段的依赖转换

**文件**：`command.cpp:166`

`Build()` 将 `m_ExecutionDependencies` 转换为**信号量等待**：

```cpp
MUresult Command::Build(const list<shared_ptr<Command>>& mergingList) {
    // ① 对前一个命令建立 semaphore 依赖
    if (m_PrevCommand && needExplicitSemaphore) {
        buildSemaphoreDependency(m_PrevCommand);
        // → 根据 semaphore 类型决定等待方式：
        //   Timeline → m_WaitSemaphoreInfos（CPU 端等）
        //   Hardware → m_WaitSemaphoreInfos（等提交后 GPU 端 wait）
    }

    // ② 对执行依赖建立 semaphore 依赖
    for_each(m_ExecutionDependencies, buildSemaphoreDependency);
    m_ExecutionDependencies.clear();
}
```

### 3.4 Semaphore 类型选择

```cpp
Hal::SemaphoreType Command::GetPreferredSemaphoreType() const {
    return m_ParentStream->EnableUserQueue(m_Engine) ?
        Hal::SemaphoreType::Hardware : Hal::SemaphoreType::Timeline;
    //  UserQueue 启用 → Hardware semaphore（更低延迟，GPU 端 wait）
    //  UserQueue 未启用 → Timeline semaphore（CPU 端 wait）
}
```

**两种信号量的区别**：

| | Timeline Semaphore | Hardware Semaphore |
|---|---|---|
| **等待位置** | CPU 端（`Wait()`） | GPU 端（`CmdWaitMemoryValue`） |
| **开销** | syscall（可能） | GPU 命令（零 CPU 开销） |
| **适用** | 跨设备 / 未启用 UserQueue | 同设备 / UserQueue 启用 |

---

## 4. Merge 合并优化

### 4.1 合并条件

```cpp
virtual bool CanMergeTo(const list<shared_ptr<Command>>& mergingList) const {
    return SupportMerge() &&                        // 命令类型支持合并
           m_ExecutionDependencies.empty() &&       // 没有未完成的依赖
           !mergingList.empty() &&
           mergingList.back()->SupportMerge() &&
           mergingList.back()->GetEngine() == GetEngine();  // 同一引擎
}
```

### 4.2 合并等级

```cpp
enum class MergeLevel { primary, secondary };
```

- **primary**：合并列表中的主命令，负责实际 Build + Submit
- **secondary**：被合并的次命令，共享主命令的 cmd buffer 和 semaphore

多个 kernel launch 可以合并到一个 cmd buffer 中，用 `CmdBarrier` 代替 semaphore 同步，减少 kick 次数。

### 4.3 合并对 MUPTI 的影响

```cpp
// AsyncSubmit 中：
for_each(m_MergingList, [submissionId](auto& merged) {
    switch(merged->GetType()) {
        case Dispatch:
            MUpti::AssignKernelToKick(uniqueId, submissionId);  // 多个内核 → 同一个 kick
            break;
        case AsyncMemcpy:
            MUpti::AssignSubmissionToCorrelation(uniqueId, submissionId);
            break;
    }
});
```

---

## 5. 生命周期详解

### 5.1 完整时序

```
时间线 |
       |
       | ① 构造 (new XxxCommand)
       |    Status = created
       |    RecordMUptiActivity() → EnterMemcpy/RegisterKernel...
       |
       | ② ResolveDependencyAndQueueCommand()
       |    RecordDependency(prev commands)
       |    → QueueCommand()
       |       Status = queued
       |       推入 m_CommandList
       |       唤醒 AsyncSubmit 线程
       |
       | ③ AsyncSubmit 线程
       |    MergeCheck → CanMergeTo()
       |    Build()
       |       Status = built
       |       编码 PM4 命令到 cmd buffer
       |       解析 semaphore 依赖
       |    Submit()
       |       SubmitToQueue()
       |         ResolveSubmitWait() → GPU 端 wait semaphores
       |         ResolveSubmitSignal() → GPU 端 signal semaphores
       |         pQueue->Submit() → kick()
       |       Status = submitted
       |
       ▼ GPU 执行中...
       |
       | ④ GPU 完成回调
       |    Postprocess() → 释放 cmd buffer
       |    ReleaseResources() → 释放 timestamp memory
       |    Status = completed
       ▼
```

### 5.2 Stream::QueueCommand — 入队

**文件**：`stream.cpp:1027`

```cpp
MUresult Stream::QueueCommand(shared_ptr<Command>&& command) {
    // ① 背压控制：asyncCapacity 限制积压
    while (m_AsyncCount.load() >= asyncCapacity) {
        yield();  // 等待提交线程消化
    }
    m_AsyncCount.fetch_add(1);

    // ② 记录与前一个命令的关系
    command->SetPrevCommand(m_LastCommand);
    m_LastCommand = command;

    // ③ 推入命令队列
    m_CommandList.push_back(command);
    m_SubmitCv.notify_one();  // 唤醒提交线程
}
```

### 5.3 Stream::AsyncSubmit — 出队 + 提交

**文件**：`stream.cpp:1141`

```cpp
void Stream::AsyncSubmit() {
    while (true) {
        // ① 从 m_CommandList 取命令，尝试合并
        while (m_MergingList.back()->CanMergeTo(m_MergingList)) {
            m_MergingList.push_back(next);  // 合并
        }

        // ② 编码 + 提交
        m_MergingList.front()->Build(m_MergingList);
        m_MergingList.front()->Submit();  // → SubmitToQueue → kick

        // ③ 后处理
        m_MergingList.front()->Postprocess();
        m_MergingList.front()->ReleaseResources();
        m_AsyncCount.fetch_sub(merged_count);
    }
}
```

### 5.4 SubmitToQueue — 最终提交

**文件**：`command.cpp:646`

```cpp
MUresult Command::SubmitToQueue(Hal::IQueue* pQueue, Hal::QueueSubmitInfo& submitInfo) {
    // ① 构建 GPU 端 wait semaphore 列表
    for (wait infos) {
        if (Timeline) → GetPeerSemaphore (跨设备)
        if (Hardware in user queue) → CmdWaitMemoryValue (GPU 端等)
        if (Hardware + Timeline mix) → CPU 端等 timeline
    }

    // ② 构建 GPU 端 signal semaphore 列表
    for (signal infos) {
        signalHalSemaphores[i] = peer semaphore;
        signalHalSemaphoresValues[i] = value;
    }

    // ③ 设置 MUPTI correlation（仅 Memcpy/AsyncMemcpy）
    if (Type == Memcpy) {
        MUpti::AssignSubmissionToCorrelation(uniqueId, GetSubId());
    }

    // ④ 最终提交
    pQueue->Submit(submitInfo);  // → HalQueue → M3D → KMD → kick()
}
```

---

## 6. Semaphore 子系统

### 6.1 信号量在 Command 中的角色

每个 Command 在**构造时**从所属 Stream 获取信号量：

```cpp
m_pSignalTimelineSemaphore = stream->GetTimelineSemaphore();
m_pSignalHardwareSemaphore = stream->GetHardwareSemaphore();
```

### 6.2 WaitMode / SignalMode

用于 Graph 执行中的 inter-submission 依赖：

```cpp
enum class WaitMode {
    Nowait,            // GPU 端不需要等（command 之间用 barrier）
    WaitBetweenCmd,    // 等前一个 graph command 完成
    WaitWithinCmd,     // 等同一 graph 内的前一个 submission
};

enum class SignalMode {
    Nosignal,          // 不需要 signal
    SignalBetweenCmd,  // signal 给下一个 graph command
    SignalWithinCmd,   // signal 给同一 graph 内的下一个 submission
};
```

### 6.3 ResolveSubmitWait — GPU 端依赖编码

```cpp
MUresult Command::ResolveSubmitWait(cmdBuffer, waitMode, pDevice) {
    if (waitMode == WaitBetweenCmd) {
        for (semaphoreInfo : m_WaitSemaphoreInfos) {
            if (Timeline) → GetPeerSemaphore → 加入 m_SubWaitSemaphoreInfos
            if (Hardware) → CmdWaitMemoryValue  // 编码 GPU 端等待命令
        }
    } else if (waitMode == WaitWithinCmd) {
        // 等 stream 内部 timeline → CmdWaitMemoryValue
    }
}
```

### 6.4 信号量语义总结

```
Producer Command                    Consumer Command
─────────────────                   ─────────────────
SignalSemaphore(value=N)
  │
  ├─ Timeline: semaphore->Signal(N) ──→  consumer.Wait() [CPU spin/yield/sleep]
  │
  └─ Hardware:                        →  consumer.Build()
       m_SubSignalSemaphoreInfo              CmdWaitMemoryValue(addr, GE, N)
       → submitInfo.signalSemaphores               ↓
                                            GPU 端硬件等待
```

---

## 7. 16 个命令子类

| 文件 | 类名 | 基类 | 关键特性 |
|------|------|------|----------|
| `dispatchCommand.cpp/h` | `DispatchCommand` | `Command` | 内核启动，ELF 加载，spill memory，抢占支持 |
| `memcpyCommand.cpp/h` | `MemcpyCommand` | `Command` | 异步 memcpy 基类，多引擎路径选择 |
| `SyncMemcpyCommand.cpp` | `SyncMemcpyCommand` | `MemcpyCommand` | 同步 memcpy，多次 Submit |
| `AsyncMemcpyCommand.cpp` | `AsyncMemcpyCommand` | `Command` | 异步 memcpy，stream 排队 |
| `memsetCommand.cpp/h` | `MemsetCommand` | `Command` | memset，引擎选择（CDM→CE→TDM→DMA→CPU） |
| `graphCommand.cpp/h` | `GraphCommand` | `Command` | CUDA Graph，递归执行子图，条件分支 |
| `barrierCommand.cpp/h` | `BarrierCommand` | `Command` | GPU 内部 barrier |
| `callbackCommand.cpp/h` | `CallbackCommand` | `Command` | CPU callback |
| `pagingCommand.cpp/h` | `PagingCommand` | `Command` | 内存 paging/页表更新 |
| `recordCommand.cpp/h` | `RecordCommand` | `Command` | 记录 event |
| `memoryAtomicCommand.cpp/h` | `MemoryAtomicCommand` | `Command` | 原子操作 |
| `memoryAtomicValueCommand.cpp/h` | `MemoryAtomicValueCommand` | `Command` | 原子操作+返回值 |
| `memoryWaitWriteCommand.cpp/h` | `MemoryWaitWriteCommand` | `Command` | 写内存屏障 |
| `memoryTransferCommand.cpp/h` | `MemoryTransferCommand` | `Command` | Peer 传输 |
| `signalExternalSemaphoreCommand.cpp/h` | `SignalExternalSemaphoreCommand` | `Command` | Vulkan/GL 信号 |
| `waitExternalSemaphoreCommand.cpp/h` | `WaitExternalSemaphoreCommand` | `Command` | Vulkan/GL 等待 |
| `accelStructBuildCommand.cpp/h` | — | `Command` | RT 加速结构构建 |
| `accelStructCopyCommand.cpp/h` | — | `Command` | RT 加速结构拷贝 |
| `accelStructEmitCommand.cpp/h` | — | `Command` | RT 加速结构发射 |
| `dispatchRayCommand.cpp/h` | — | `Command` | 光追调度 |

---

## 8. 错误处理 (ErrorHandler)

**文件**：`command.cpp:369`

当命令执行失败时，`ErrorHandler` 触发 core dump：

```cpp
MUresult Command::ErrorHandler(DumpType dumpType) {
    // ① 生成错误消息
    auto msg = GenerateErrorMessage();
    // "Application: xxx encountered ERROR in stream 1 of context 2 of device 0
    //  when executing a command of type Dispatch"

    // ② 收集硬件错误信息
    if (dumpHardWare) {
        pHalQueue->QueryRobustInfo(printableDataVerbose);  // 队列状态
        pHalQueue->QueryRobustInfo(mpExceptionInfo);       // MP 异常信息
        pHalQueue->QueryRobustInfo(mpWaveInfo);            // Wave 状态
    }

    // ③ Shader 反汇编
    ctxCrit->FindSymbolAndDisassembleInst(pc, base, oss);
    // 输出：异常 PC → shader function → 反汇编指令

    // ④ 写入文件或 pipe
    file << oss.str();

    // ⑤ 可选 abort
    if (!skipAbort) std::abort();
}
```

---

## 9. Pfm / PC Sampling / Timestamp

### 9.1 Pfm (Performance Monitor)

```cpp
// 每个 command 可以绑定一个 PerfExperiment
MUresult InitPfm() {
    device->Hal().CreatePerfExperiment(info, &m_PerfExperiment);
    m_PerfExperiment->AddPfmTrace(*m_PfmDumpConfig);
    m_PerfExperiment->Finalize();
}

void BeginPfm(cmdBuffer) { cmdBuffer->CmdBeginPerfExperiment(m_PerfExperiment); }
void EndPfm(cmdBuffer)   { cmdBuffer->CmdEndPerfExperiment(m_PerfExperiment); }
```

### 9.2 PC Sampling

```cpp
void BeginPcSampling(cmdBuffer) { cmdBuffer->CmdBeginPcSampling(); }
void EndPcSampling(cmdBuffer)   { cmdBuffer->CmdEndPcSampling(); }
void QueryPcSamplingShaderInfo(cmdBuffer) {  // 写入 shader 映射文件
    cmdBuffer->QueryPcSamplingShaderInfo(fileName);
}
```

### 9.3 Timestamp

```cpp
MUresult AllocateTimestampMem() {
    // 分配 GPU 可写的 timestamp 内存（Host-visible）
    pDevice->AllocateInternalMem(size, alignment, &m_TimestampMem, Profile);

    // begin 时间戳地址 = base
    // end   时间戳地址 = base + oneElementSize
    m_TimestampGpuAddrs[beginTime] = gpuBase;
    m_TimestampGpuAddrs[endTime]   = gpuBase + elementsize;
}
```

---

## 10. 相关源文件索引

| 文件 | 行数 | 核心内容 |
|------|------|----------|
| `command.h` | 336 | 基类定义、状态机、16 种类型、成员变量 |
| `command.cpp` | 806 | Build/SubmitToQueue/Wait/ErrorHandler/Pfm/Timestamp |
| `stream.cpp` | 1844 | QueueCommand (入队)、AsyncSubmit (出队+合并+提交) |
| `context.cpp:1884` | ~60 | ResolveDependencyAndQueueCommand (依赖解析) |
| `dispatchCommand.cpp` | 1154 | 内核启动：ELF 加载、spill memory、抢占 |
| `memcpyCommand.cpp` | ~ | 异步 memcpy 基类、多引擎路径 |
| `graphCommand.cpp` | 650 | 图启动：submission type 分发、递归、条件分支 |
| `memsetCommand.cpp` | ~ | memset 引擎选择（CDM→CE→TDM→DMA→CPU） |
