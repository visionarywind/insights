# Command 依赖系统：m_SubmitDependencies 的角色与去重逻辑

> 确认代码逻辑后，详细分析 Command 的三个依赖列表（`m_PrevCommand`、`m_SubmitDependencies`、`m_ExecutionDependencies`）各自的职责、时机和关系。

---

## 1. 三个依赖列表对比

**文件**：`command.h:227-229`

```cpp
std::list<std::shared_ptr<Command>>  m_ExecutionDependencies;  // 执行依赖：Build 时转 semaphore wait
std::list<std::shared_ptr<Command>>  m_SubmitDependencies;     // 提交依赖：仅供 RecordDependency 内部去重
std::shared_ptr<Command>             m_PrevCommand;            // 同 stream 前一个命令：Build 时转 semaphore wait
```

| 列表 | 什么时候写入 | 什么时候读取 | 作用 |
|------|------------|------------|------|
| `m_PrevCommand` | `QueueCommand`（stream.cpp:1082） | `Build`（command.cpp:213） | 同 stream 内前后命令的 semaphore 同步 |
| `m_SubmitDependencies` | `RecordDependency`（command.cpp:129） | **仅在 `RecordDependency` 自身**（command.cpp:132） | 去重：判断下一个命令是否需要重复依赖 |
| `m_ExecutionDependencies` | `RecordDependency` 去重后（command.cpp:141） | `Build`（command.cpp:221）+ `FilterDependency`（command.cpp:147） | 跨 stream 依赖→semaphore wait |

**关键结论**：`m_SubmitDependencies` 不参与 Build、不参与 Submit、不影响 GPU 执行。它的唯一作用是 `RecordDependency` 内部的去重判断。

---

## 2. 依赖列表的入口：谁在调用 RecordDependency

**文件**：`context.cpp:1893` → `ResolveDependencyAndQueueCommand`

一个命令入队时，会调用多次 `RecordDependency`，每次传入不同的 producer：

```cpp
MUresult Context::ResolveDependencyAndQueueCommand(command, pStream, blocking) {
    // ① Default Stream 语义
    if (pStream == m_DefaultStream) {
        // default stream 依赖所有 blocking stream 的最后命令
        for (auto pOtherStream : blocking streams) {
            command->RecordDependency(pOtherStream->LastCommand());  // ← producer = 其他 stream 最后命令
        }
    }
    // ② Blocking Stream 语义
    else if (!pStream->IsNonBlocking()) {
        command->RecordDependency(m_DefaultStream->LastCommand());   // ← producer = default stream 最后命令
    }

    // ③ Barrier Stream 语义
    auto barrierCmd = m_BarrierStream->LastCommand();
    if (barrierCmd && barrierCmd->GetStatus() > completed) {
        command->RecordDependency(std::move(barrierCmd));           // ← producer = barrier stream 最后命令
    }

    // ④ 用户显式依赖（muStreamWaitEvent 等）
    for (auto& dep : pStream->GetCurrentDependencies()) {
        command->RecordDependency(std::move(dep));                  // ← producer = 用户指定的依赖
    }

    // ★ 注意：同 stream 的前后依赖不在这里处理！
    //         它在后续的 QueueCommand 里通过 SetPrevCommand 单独设置
    pStream->QueueCommand(std::move(command));
}
```

④ 之后调 `QueueCommand`，它在 `RecordDependency` 调用之后执行：

```cpp
// stream.cpp:1082 → QueueCommand
command->SetPrevCommand(m_LastCommand);  // 同 stream 前一个命令
//        ↑ 记到 m_PrevCommand，不是 m_SubmitDependencies
```

**同 stream 前后依赖和跨 stream 依赖是分开记录的**：前一个通过 `m_PrevCommand`，跨 stream 的通过 `RecordDependency` → `m_ExecutionDependencies`。

---

## 3. RecordDependency 内部逻辑（逐行分析）

**文件**：`command.cpp:123`

```cpp
void Command::RecordDependency(std::shared_ptr<Command>&& producer) {
    // 第 124-126 行：入口过滤
    //   如果 producer 为空 → 跳过
    //   如果 producer 已经 completed → 跳过（不需要等了）
    if (producer
#ifndef M3D_BUILD_MT_TRACE_CAPTURE
        && producer->GetStatus() != Status::completed
#endif
        ) {

        // 第 129 行：★ 无条件加入 m_SubmitDependencies
        //   不管这个 producer 是否已经在前一个命令的依赖列表里出现过，
        //   都先加进去。这个列表不 override，只 append。
        m_SubmitDependencies.push_back(producer);

        // 第 130 行：标记是否需要加入执行依赖
        bool needRecord = true;

        // 第 131 行：取同 stream 上紧挨着的前一个命令
        //   ★ 注意：此时新命令还没有 push 到 stream，所以 LastCommand()
        //     返回的是前一个命令，不是它自己
        auto prevCommand = m_ParentStream->LastCommand();

        // 第 132 行：检查前一个命令的 m_SubmitDependencies
        //   只有当前一个命令存在且它的 m_SubmitDependencies 不为空时才检查
        if (prevCommand && !prevCommand->GetSubmitDependencies().empty()) {

            // 第 133 行：取前一个命令的提交依赖列表
            const auto& prevCommandDependencies = prevCommand->GetSubmitDependencies();

            // 第 134 行：在前一个命令的依赖列表中查找 producer
            auto it = std::find(prevCommandDependencies.begin(),
                                prevCommandDependencies.end(),
                                producer);

            // 第 135-137 行：如果找到了 → 去重
            if (prevCommandDependencies.end() != it) {
                needRecord = false;
            }
        }

        // 第 140-143 行：只有去重失败的才加入执行依赖
        if (needRecord) {
            m_ExecutionDependencies.push_back(std::move(producer));
        }
    }
}
```

### 3.1 第 129 行的含义

`m_SubmitDependencies.push_back(producer)` 在这里的作用**不是记录"我要等谁"**——它的唯一目的是**给下一个命令看去重用**。

即使这个 producer 在前一个命令的 `m_SubmitDependencies` 里已经存在（意味着前一个命令已经替我等了），当前命令仍然把它加入自己的 `m_SubmitDependencies`。为什么？因为**再下一个命令可能也需要查这个列表来判断去重**。

```
举例：
  三个连续命令都依赖 producer_X：

  cmd_A.RecordDependency(producer_X):
    m_SubmitDependencies = [X]            ← 加入
    prevCommand = null → needRecord=true
    m_ExecutionDependencies = [X]

  cmd_B.RecordDependency(producer_X):
    m_SubmitDependencies = [X]            ← 加入（即使会被去重）
    prevCommand=cmd_A
    cmd_A.m_SubmitDependencies=[X] → 找到 → needRecord=false
    m_ExecutionDependencies = []           ← 跳过

  cmd_C.RecordDependency(producer_X):
    m_SubmitDependencies = [X]            ← 加入
    prevCommand=cmd_B
    cmd_B.m_SubmitDependencies=[X] → 找到 → needRecord=false
    m_ExecutionDependencies = []           ← 跳过
```

如果 `cmd_B` 不加 `procuder_X` 到自己的 `m_SubmitDependencies`，`cmd_C` 去重时就找不到它了。

### 3.2 去重为什么安全

```
cmd_A 的 m_ExecutionDependencies = [producer_X]
cmd_B 的 m_PrevCommand = cmd_A
cmd_C 的 m_PrevCommand = cmd_B

Build 时：
  cmd_A → semaphore wait producer_X
  cmd_B → semaphore wait cmd_A  ← 等 cmd_A 的时候，间接等了 producer_X
  cmd_C → semaphore wait cmd_B  ← 等 cmd_B 的时候，间接等了 producer_X

所以 cmd_B 和 cmd_C 不需要直接等 producer_X，
通过 m_PrevCommand 链条间接保证了。
```

### 3.3 去重失败的情况

去重只检查**紧挨着的前一个命令**（第 131 行 `LastCommand()`），不遍历全部队列。如果中间隔了一个不依赖 `producer_X` 的命令，去重就会失败：

```
cmd_A → RecordDependency(producer_X)    → m_ExecutionDependencies = [X]
cmd_B → 不依赖 producer_X（没有调 RecordDependency 或依赖不同的）
cmd_C → RecordDependency(producer_X)

cmd_C 检查 cmd_B.m_SubmitDependencies，找不到 X → needRecord=true
→ cmd_C.m_ExecutionDependencies = [X]   ← 重复记录了
```

**这不影响正确性**——只是多了一条冗余的 semaphore wait。Build 阶段会正确处理。

---

## 4. 三个列表在 Build 中的使用

**文件**：`command.cpp:166`

```cpp
MUresult Command::Build(const list<shared_ptr<Command>>& mergingList) {
    // ① 处理 m_PrevCommand（同 stream 前一个命令）
    if (m_PrevCommand && needExplicitSemaphore) {
        buildSemaphoreDependency(m_PrevCommand, Status::built);
        // → 根据 semaphore 类型，在 GPU 命令里插入 CmdWaitMemoryValue 或
        //   在 CPU 端等 timeline semaphore
    }
    m_PrevCommand.reset();  // 用完就清掉

    // ② 处理 m_ExecutionDependencies（跨 stream 依赖）
    for (auto& dep : m_ExecutionDependencies) {
        buildSemaphoreDependency(dep, Status::submitted);
    }
    m_ExecutionDependencies.clear();  // 用完就清掉

    // ★ m_SubmitDependencies 不在这里处理 —— 它只参与去重
}
```

---

## 5. 完整时序

```
时间线 ─────────────────────────────────────────────────────────────────→

Thread A (用户线程):

  musaLaunchKernel
    │
    ├─ new DispatchCommand
    │   m_PrevCommand = null
    │   m_SubmitDependencies = []
    │   m_ExecutionDependencies = []
    │
    ├─ ResolveDependencyAndQueueCommand()
    │   ├─ RecordDependency(defaultStream->LastCommand())   ← 跨 stream
    │   │   m_SubmitDependencies = [X]
    │   │   m_ExecutionDependencies = [X]  (去重通过)
    │   │
    │   ├─ RecordDependency(barrierStream->LastCommand())   ← barrier
    │   │   m_SubmitDependencies = [X, Y]
    │   │   m_ExecutionDependencies = [X, Y]  (去重通过)
    │   │
    │   └─ RecordDependency(userExplicitDependency)          ← 用户显式
    │       m_SubmitDependencies = [X, Y, Z]
    │       m_ExecutionDependencies = [X, Y, Z]  (去重通过)
    │
    └─ QueueCommand()
        ├─ SetPrevCommand(m_LastCommand)           ← 同 stream 前一个
        │   m_PrevCommand = cmd_prev
        │
        ├─ SetStatus(queued)
        └─ m_CommandList.push_back(this)           ← 入队
          

AsyncSubmit 线程:

  buildCommand(cmd)
    ├─ FilterDependency()                          ← 去掉已完成的
    │   m_ExecutionDependencies.erase(completed 的)
    │
    ├─ Build(mergingList)
    │   ├─ buildSemaphoreDependency(m_PrevCommand)      ← 同 stream
    │   │   → m_WaitSemaphoreInfos.push(semaphore_prev)
    │   │
    │   ├─ buildSemaphoreDependency(X)                   ← 跨 stream
    │   │   → etc.
    │   ├─ buildSemaphoreDependency(Y)
    │   └─ buildSemaphoreDependency(Z)
    │
    │   m_PrevCommand.reset()
    │   m_ExecutionDependencies.clear()
    │
    └─ Submit()
        └─ SubmitToQueue()
            → pQueue->Submit(wait_semaphores, signal_semaphores)
            → kick()
```

---

## 6. 总结

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  m_PrevCommand:                                                  │
│    写入时机: QueueCommand (stream.cpp:1082)                      │
│    消费时机: Build (command.cpp:213)                              │
│    作用:     同 stream 前后命令的 semaphore 同步                   │
│                                                                  │
│  m_SubmitDependencies:                                           │
│    写入时机: RecordDependency (command.cpp:129)                  │
│    消费时机: ★ 仅 RecordDependency 内部 (command.cpp:132)         │
│    作用:     去重 —— 让同一个 stream 上连续依赖                  │
│              同一个 producer 的命令不产生重复的执行依赖             │
│    清理时机: 析构函数 (command.cpp:78)                             │
│                                                                  │
│  m_ExecutionDependencies:                                        │
│    写入时机: RecordDependency 去重后 (command.cpp:141)            │
│    消费时机: Build (command.cpp:221) + FilterDependency           │
│    作用:     跨 stream 依赖 → semaphore wait                      │
│                                                                  │
│  关键设计:                                                       │
│    同 stream 前后依赖 (m_PrevCommand) 和                             │
│    跨 stream 依赖 (m_ExecutionDependencies) 是两条独立路径           │
│                                                                  │
│    m_SubmitDependencies 是纯辅助列表，                              │
│    不参与任何实际的同步操作                                        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```
