# MUSA Command 系统：从 musaLaunchKernel 到 GPU 执行

> 以一次 `musaLaunchKernel` 调用为线索，追踪 Command 在 MUSA 驱动中的完整生命周期。
> 借此理解状态机、依赖解析、合并优化、信号量同步、错误处理等核心机制。

---

## 例子：用户发起一次内核启动

```c
// 用户代码
musaLaunchKernel(myKernel, grid, block, args, 0, stream);
```

这背后会经历 **7 个阶段**：

```
① 创建 DispatchCommand     ② 依赖解析        ③ 入队
    new                        RecordDependency    QueueCommand
    │                          │                   │
    ▼                          ▼                   ▼
[created] ──────────────→ [queued] ──────────→ [built] ──────→ [submitted] ──────→ [completed]
                              ▲                   ▲               ▲                   ▲
                          ④ 唤醒提交线程       ⑤ Build         ⑥ Submit           ⑦ GPU 完成回调
                             AsyncSubmit          编码到cmd buffer  SubmitToQueue       AsyncWait
                             + Merge 决策         + semaphore     + kick()             + Postprocess
```

---

## 阶段 ①：创建 DispatchCommand

**文件**：`stream.cpp:1482` → `CmdLaunchKernel`

```cpp
MUresult Stream::CmdLaunchKernel(IGraphNode* pGraphNode, bool blocking) {
    // ①-1: 构造命令对象
    std::shared_ptr<Command> command = std::make_shared<DispatchCommand>(
        this,                              // 所属 stream
        ICast<GraphNode>(pGraphNode),      // 包含 kernel 函数、grid/block 参数
        m_ParentCtx->PfmCheckEnable(),     // 是否启用 performance monitor
        m_ParentCtx->PfmGetConfig()        // Pfm 配置
    );
```

**`DispatchCommand` 构造函数中的关键操作**（`dispatchCommand.cpp:10`）：

```cpp
DispatchCommand::DispatchCommand(Stream* stream, GraphNode* pGraphNode, ...)
    : Command(Type::Dispatch,          // 类型 = 内核启动
              stream,                  // 绑定到调用者的 stream
              Device::Engine::cdm,     // 引擎 = CDM (Compute Dispatch Manager)
              true,                    // 支持 merge = true
              muptiContext, ...)       // MUPTI profiling context
{
    // ①-2: 如果有 MUPTI 启用，注册 kernel 追踪
    if (RecordMUptiActivity()) {
        if (deviceProperties.ipProperties.supportEngineSync) {
            m_ptiCtx = MUpti::RegisterKernelV2(this);  // 新平台：注册 kernel
        } else {
            MUpti::RegisterKernel(this);                // 旧平台
        }
    }

    // ①-3: 处理 spill memory（如果 kernel 需要）
    if (hwManagedSpillMemory && pKernel->spilledPrivateMemorySize > 0) {
        // 计算需要的 spill memory 大小
        // 尝试从 stream 预留的 spill buffer 分配
        m_UsePerStreamSpillMemory = stream->RequestSpillBase(spillSize);
    }
}
```

构造完成时，命令处于 `Status::created` 状态。

---

## 阶段 ②：依赖解析

**文件**：`context.cpp:1884` → `ResolveDependencyAndQueueCommand`

构造好命令后，不能立即提交——必须解析它**依赖哪些之前的命令**。这一步决定"这个 kernel 要等谁完成才能开始"。

```cpp
MUresult Context::ResolveDependencyAndQueueCommand(
    std::shared_ptr<Command>&& command,  // 刚创建的 DispatchCommand
    Stream* pStream,                     // 目标 stream
    bool blocking) {                     // 是否阻塞等待

    // ②-1: Default Stream 的语义
    if (pStream == m_DefaultStream) {
        // default stream 依赖当前 context 中所有 blocking stream 的最后一条命令
        for (auto pOtherStream : ctxCrit->Streams()) {
            if (pOtherStream != m_DefaultStream && !pOtherStream->IsNonBlocking()) {
                command->RecordDependency(pOtherStream->LastCommand());
                //  ↑ 把其他 stream 的最后一条命令记录为依赖
            }
        }
    }

    // ②-2: Blocking Stream 的语义
    else if (!pStream->IsNonBlocking()) {
        // blocking stream 依赖 default stream 的最后一条命令
        command->RecordDependency(m_DefaultStream->LastCommand());
    }

    // ②-3: 同一 stream 上的前一个命令
    command->SetPrevCommand(pStream->LastCommand());
    //        ↑ m_PrevCommand 记录同 stream 上的前一条，用于 Build 时的 semaphore sync

    // ②-4: Barrier stream（用于跨 stream 批量同步）
    auto barrierCmd = m_BarrierStream->LastCommand();
    if (barrierCmd != nullptr && barrierCmd->GetStatus() > completed) {
        command->RecordDependency(std::move(barrierCmd));
    }

    // ②-5: 用户显式设置的依赖（如 muStreamWaitEvent）
    for (auto& dep : pStream->GetCurrentDependencies()) {
        command->RecordDependency(std::move(dep));
    }

    // ②-6: 入队
    return pStream->QueueCommand(std::move(command));
}
```

**场景示例 — 依赖长什么样**：

```
Stream 0:  [kernel_A] ──→ [kernel_B] ──→ [kernel_C]    ← CmdLaunchKernel 刚创建的
                                          ↑
                                     m_PrevCommand = kernel_B
                                     (kernel_C 必须等 kernel_B 完成)

Stream 1:  [memcpy_X] ──→ [memcpy_Y] ──→ [kernel_D]    ← 另一个 blocking stream
              ↑
         m_DefaultStream->LastCommand() = kernel_C
         (kernel_D 必须等 default stream 的 kernel_C 完成)
```

**`RecordDependency` 做了什么**（`command.cpp:123`）：

```cpp
void Command::RecordDependency(std::shared_ptr<Command>&& producer) {
    m_SubmitDependencies.push_back(producer);     // 提交时检查
    m_ExecutionDependencies.push_back(producer);  // 执行时必须等待
    //  ↑ 后续 Build() 阶段，这些依赖被转换为 semaphore wait
}
```

---

## 阶段 ③：入队 — QueueCommand

**文件**：`stream.cpp:1027`

```cpp
MUresult Stream::QueueCommand(std::shared_ptr<Command>&& command) {
    // ③-1: 背压控制 — 防止命令积压过多
    const uint32_t asyncCapacity = Platform::Get().GetSettings().streamAsyncCapacity;
    while (m_AsyncCount.load() >= asyncCapacity) {
        std::this_thread::yield();   // 等提交线程消化
    }

    m_AsyncCount.fetch_add(1);     // 原子增加 inflight 计数

    // ③-2: 记录 stream 内的前后关系
    std::unique_lock<std::mutex> lk(m_SubmitMtx);
    command->SetPrevCommand(m_LastCommand);
    m_LastCommand = command;

    command->SetStatus(Command::Status::queued);   // ★ 状态: created → queued

    // ③-3: 记录时间戳
    if (command->GetType() == Command::Type::Dispatch) {
        command->SetQueuedTimestamp(GetCurrentTime());   // 供 MUPTI 使用
        MUpti::MarkKernelQueued(command->GetCorId());    // 通知 MUPTI
    }

    // ③-4: 推入命令队列
    m_CommandList.push_back(std::move(command));
    //  ↑ 这是一个 std::list，生产者和消费者用 mutex + condition_variable 同步

    lk.unlock();
    m_SubmitCv.notify_one();   // ★ 唤醒 AsyncSubmit 线程！
}
```

---

## 阶段 ④：AsyncSubmit 线程 — Merge + Build + Submit

**文件**：`stream.cpp:1141`

每个 Stream 有一个后台线程 `AsyncSubmit`，不断从 `m_CommandList` 取命令，执行合并→构建→提交。

### ④-a: 主循环

```cpp
void Stream::AsyncSubmit() {
    std::unique_lock<std::mutex> submitLock(m_SubmitMtx);

    while (true) {
        // 等待：有命令可出队 OR 合并列表需要 flush
        m_SubmitCv.wait(submitLock, [this, &stopMerging] {
            return !m_CommandList.empty() || stopMerging() || m_SubmitStopToken;
        });

        // 如果合并列表需要 flush，先提交当前合并批
        if (stopMerging()) {
            submitMergingList();  // → 阶段 ④-d
        }

        // 从 m_CommandList 头部取出一个命令
        if (!m_CommandList.empty()) {
            auto front = std::move(m_CommandList.front());
            m_CommandList.pop_front();
            buildCommand(std::move(front));  // → 阶段 ④-b
        }
    }
}
```

### ④-b: buildCommand — 合并决策 + Build

```cpp
auto buildCommand = [this, &submitMergingList](shared_ptr<Command>&& command) {
    // ④-b-1: 过滤已完成的依赖
    command->FilterDependency();

    // ④-b-2: 决定是否合并到当前批次
    bool keepMerging = command->CanMergeTo(m_MergingList);
    //  ↑ CanMergeTo 条件（command.h:109）：
    //    1. 命令支持 merge（m_SupportMerge == true）
    //    2. 没有未完成的执行依赖
    //    3. 合并列表不为空
    //    4. 列表最后一个命令也支持 merge
    //    5. 同一个引擎（如都是 CDM）

    // ④-b-3: 分配 semaphore signal 值
    if (keepMerging && pPrimaryCommand) {
        command->SetSignalSemaphoreValue(pPrimaryCommand->GetSignalSemaphoreValue());
        //  ↑ 合并的命令共享同一个 signal 值 → GPU 一次性完成所有
    } else {
        command->SetSignalSemaphoreValue(++m_TimelineValue);
        //  ↑ Timeline key 递增，用于 Track 依赖
    }

    // ④-b-4: 如果不能合并，先提交当前批次
    if (!keepMerging) {
        submitMergingList();  // 把之前合并的一批先提交了
    }

    // ④-b-5: Build — 编码 GPU 命令 + 建立 semaphore 依赖
    command->Build(m_MergingList);  // → 阶段 ⑤

    command->SetStatus(Command::Status::built);   // ★ 状态: queued → built

    // ④-b-6: 加入合并列表
    m_MergingList.push_back(command);
};
```

### ④-c: 合并实例 — 两个连续 kernel

```
用户代码：
  musaLaunchKernel(kernel1, ..., stream);   // → DispatchCommand_1
  musaLaunchKernel(kernel2, ..., stream);   // → DispatchCommand_2

AsyncSubmit 的处理：

  DispatchCommand_1 出队：
    → CanMergeTo(空列表) = false（列表空，不合并）
    → Build() → 编码到 cmd_buffer_1
    → m_MergingList = [DispatchCommand_1]   ← 作为 primary

  DispatchCommand_2 出队：
    → CanMergeTo([DispatchCommand_1]) = true（同引擎 CDM，都支持 merge）
    → keepMerging = true
    → 共享 signal semaphore value
    → Build() → 编码到同一个 cmd_buffer_1 中！
    → m_MergingList = [DispatchCommand_1, DispatchCommand_2]

  后续来了一个 Memcpy 命令（引擎 = DMA）：
    → CanMergeTo(...) = false（不同引擎）
    → 先 submitMergingList() 提交当前批次
    → 再单独处理 Memcpy
```

### ④-d: submitMergingList — 提交合并批次

```cpp
auto submitMergingList = [this]() {
    // ④-d-1: MUPTI correlation
    for (auto& merged : m_MergingList) {
        if (merged->GetType() == Dispatch) {
            MUpti::AssignKernelToKick(uniqueId, submissionId);
            // ↑ 告诉 MUPTI：这些 kernel 将共享一个 kick
        }
    }

    // ④-d-2: 提交 primary 命令（所有 merged 命令共享状态）
    m_MergingList.front()->Submit();  // → 阶段 ⑥

    // ④-d-3: 设置所有命令状态
    for (auto& merged : m_MergingList) {
        merged->SetStatus(Command::Status::submitted);  // ★ built → submitted
    }

    // ④-d-4: 移入 inflight 列表（等待 GPU 完成）
    m_InflightList.splice(end, m_MergingList);
    m_WaitCv.notify_one();  // ★ 唤醒 AsyncWait 线程
};
```

---

## 阶段 ⑤：Build — 编码 + Semaphore

**文件**：`command.cpp:166`

```cpp
MUresult Command::Build(const list<shared_ptr<Command>>& mergingList) {
    // ⑤-1: 对前一个命令建立 semaphore 依赖
    if (m_PrevCommand && needExplicitSemaphore) {
        buildSemaphoreDependency(m_PrevCommand);
    }

    // ⑤-2: 对所有执行依赖建立 semaphore 依赖
    for (auto& dep : m_ExecutionDependencies) {
        buildSemaphoreDependency(dep);
    }
    m_ExecutionDependencies.clear();  // 转换为 semaphore 后清理

    // buildSemaphoreDependency 内部的逻辑：
    //   if (dep->status > submitted) → spin wait（还没提交）
    //   if (dep->lastError != SUCCESS) → 传播错误
    //   → 根据 semaphore 类型决定等待方式：
    //       Timeline  → m_WaitSemaphoreInfos（CPU 端 HostWaitSemaphores）
    //       Hardware  → m_WaitSemaphoreInfos（GPU 端 CmdWaitMemoryValue）
}
```

**Semaphore 类型选择**：

```cpp
Hal::SemaphoreType GetPreferredSemaphoreType() const {
    return stream->EnableUserQueue(engine) ?
        Hardware : Timeline;
}
```

| 场景 | Semaphore 类型 | 等待方式 |
|------|---------------|---------|
| UserQueue 启用 + 同设备 | Hardware | GPU 端 `CmdWaitMemoryValue`（零 CPU 开销） |
| UserQueue 未启用 或 跨设备 | Timeline | CPU 端 `semaphore->Wait()` |

**具体来说，Build 把一个依赖关系从"语义上的"变成"可执行的"**：

```
依赖关系（阶段 ② 记录的）:
  DispatchCommand_C 依赖 DispatchCommand_B

Build 后变成：
  DispatchCommand_C.m_WaitSemaphoreInfos = [(semaphore_B, value=5)]
  ↑ 含义：在 CPU 上等待 semaphore_B 达到值 5，才能继续提交
  或者：在 GPU cmd buffer 里插入 CmdWaitMemoryValue(semaphore_B_addr, GE, 5)
```

---

## 阶段 ⑥：Submit → SubmitToQueue → Kick

**文件**：`command.cpp:646` → `SubmitToQueue`

每个 `Command` 子类必须实现 `Submit()`。以 `DispatchCommand` 为例，`Submit()` 最终调用基类的 `SubmitToQueue`：

```cpp
MUresult Command::SubmitToQueue(Hal::IQueue* pQueue, Hal::QueueSubmitInfo& submitInfo) {
    // ⑥-1: 构建 GPU wait semaphore 列表
    for (semaphoreInfo : m_SubWaitSemaphoreInfos) {
        if (Timeline && Hardware) {
            // 跨设备：在 CPU 端等 timeline，GPU 端不用等
            semaphoreInfo.first->Wait(value);
        } else if (Hardware) {
            // 同设备 UserQueue：编码到 cmd buffer 里的 CmdWaitMemoryValue
            // （已在 ResolveSubmitWait 中处理）
        } else {
            // Timeline：加入 wait 列表
            waitHalSemaphores.push_back(peerSemaphore);
        }
    }
    submitInfo.ppWaitSemaphores   = waitHalSemaphores.data();
    submitInfo.waitSemaphoreCount = waitHalSemaphores.size();

    // ⑥-2: 构建 GPU signal semaphore 列表
    submitInfo.ppSignalSemaphores     = signalHalSemaphores.data();
    submitInfo.signalSemaphoreCount   = signalHalSemaphores.size();

    // ⑥-3: 设置 Pfm + PC Sampling
    submitInfo.pPerfExperiment = m_PerfExperiment;

    // ⑥-4: MUPTI correlation（仅 Memcpy）
    if (GetType() == Command::Type::Memcpy) {
        MUpti::AssignSubmissionToCorrelation(uniqueId, GetSubId());
    }

    // ⑥-5: ★ 最终提交 — 触发 kick
    pQueue->Submit(submitInfo);
    //  → HalQueue → M3D::Queue::Submit → CmdBuffer::End → OsSubmit → ioctl → CCB → fw_kick
}
```

---

## 阶段 ⑦：AsyncWait 线程 — 等待完成 + 后处理

**文件**：`stream.cpp:1347`

每个 Stream 还有一个 `AsyncWait` 线程，负责等待 inflight 命令完成：

```cpp
void Stream::AsyncWait() {
    while (true) {
        // ⑦-1: 等 inflight 列表有内容
        m_WaitCv.wait(waitLock, [this] { return !m_InflightList.empty(); });

        // ⑦-2: 取出需要等待的命令（primary 级别的）
        m_WaitingList.splice(end, m_InflightList.begin(), primary_commands);

        Command& frontCommand = *m_WaitingList.front();

        // ⑦-3: 等待 semaphore
        if (frontCommand.GetMergeLevel() == primary) {
            if (preferred == Hardware) {
                frontCommand.GetHardwareSemaphore()->Wait(value);  // CPU 端等硬件 semaphore
                m_TimelineSemaphore->Signal(value);                 // 同步 CPU timeline
            } else {
                frontCommand.GetTimelineSemaphore()->Wait(value);   // CPU 端等 timeline
            }
        }

        // ⑦-4: 检查 GPU 错误
        GetEngineLastError(frontCommand.GetEngine(), &commandStatus);

        // ⑦-5: 检查 kernel 内的 printf/assert
        if (command->GetType() == Dispatch) {
            static_cast<DispatchCommand&>(*command).OutputPrintBuffer(hasError);
        }

        // ⑦-6: 如果有错误，触发 core dump
        if (commandStatus != MUSA_SUCCESS && streamStillSuccess) {
            command->ErrorHandler(DumpType::All);  // → 收集硬件异常 + wave 状态 + shader 反汇编
        }

        // ⑦-7: Postprocess — 归还 cmd buffer 到 pool
        command->Postprocess();
        // → ReleaseCmdBuffer(engine, cmdBuffer);

        // ⑦-8: ReleaseResources — 释放 timestamp memory, Pfm experiment 等
        command->ReleaseResources();

        // ⑦-9: 标记完成
        command->SetStatus(Command::Status::completed);  // ★ submitted → completed
        m_AsyncCount.fetch_sub(1);  // 减少 inflight 计数
    }
}
```

---

## 完整时序图

```
用户线程                    AsyncSubmit 线程              GPU                    AsyncWait 线程
───────                    ──────────────                ───                    ──────────────
musaLaunchKernel()
│
├─ new DispatchCommand     [主线程阻塞]
│   Status=created
│
├─ ResolveDependency
│   RecordDependency()
│
├─ QueueCommand()
│   Status=queued
│   m_CommandList.push()
│   notify_one() ──────────────────▶ 唤醒
│                                      │
│                                      ├─ buildCommand()
│                                      │   CanMergeTo() → false
│                                      │   Build():
│                                      │     semaphore 依赖转换
│                                      │   Status=built
│                                      │   m_MergingList.push()
│                                      │
│                                      ├─ submitMergingList()
│                                      │   Submit():
│                                      │     SubmitToQueue()
│                                      │       pQueue->Submit()
│                                      │         ───────────▶ kick()
│                                      │   Status=submitted              执行 kernel
│                                      │   m_InflightList.push()            │
│                                      │   notify_one() ──────────────────────────▶ 唤醒
│                                      │                                      ├─ Wait semaphore
│                                      │                                      ├─ 检查错误
│                                      │                                      ├─ Postprocess()
│                                      │                                      ├─ ReleaseResources()
│                                      │                                      └─ Status=completed
│                                      │
└─ return                            │
  (如果非 blocking)                    │
```

---

## 状态机速查

```
created ──→ queued ──→ built ──→ submitted ──→ completed
   │                                                 │
   └──────────────────── error ──────────────────────┘
   (任意时刻可跳转到 error)

状态转换位置：
  created  → queued     QueueCommand (stream.cpp:1057)
  queued   → built      buildCommand (stream.cpp:1290)
  built    → submitted  submitMergingList (stream.cpp:1244)
  submitted→ completed  AsyncWait (stream.cpp:1452)
  *        → error      SetLastError 后 SetStatus(error)
```

---

## 关键设计原则

1. **三层线程模型**：用户线程（生产）、AsyncSubmit（合并+提交）、AsyncWait（等待+后处理），通过 `m_CommandList` / `m_MergingList` / `m_InflightList` 三个队列连接。

2. **合并减少 kick**：同一引擎的连续 kernel 合并到同一个 cmd buffer，一次 kick 提交。DDK 约束要求合并的命令共享 signal semaphore value。

3. **依赖 → semaphore**：语义依赖在 `RecordDependency` 阶段记录为 `shared_ptr<Command>` 引用，在 `Build` 阶段转换为具体的 semaphore wait。

4. **双信号量策略**：UserQueue 启用时优先 Hardware semaphore（GPU 端零 CPU 开销），否则用 Timeline semaphore（通用但需 CPU 端 wait）。

5. **MUPTI 贯穿全生命周期**：构造时 `RegisterKernel`、入队时 `MarkKernelQueued`、提交时 `AssignKernelToKick`、完成时 `MarkKernelBeginEnd`。
