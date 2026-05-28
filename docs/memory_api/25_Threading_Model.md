# MUSA Runtime 线程模型深度分析

> 本文档对 MUSA Runtime 的完整线程模型进行系统性拆解，涵盖所有线程实例、锁体系、原子操作、TLS 设计及跨线程状态机，并针对典型使用场景提出优化方向。

---

## 1. 架构总览

MUSA Runtime 采用 **多线程异步提交 + 全局轻量线程池** 的混合模型：

```
                              ┌──────────────────────────────────────┐
                              │          Platform::m_Executor        │
                              │         (全局单线程 Worker)          │
                              │    QueueTask → 资源延迟释放/流销毁   │
                              └──────────────────────────────────────┘
                                        ▲                ▲
                            ReleaseResources()    ~Stream 同步
                                        │                │
 ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
 │ 用户线程 A   │    │ 用户线程 B   │    │ 用户线程 C   │
 │ TlsCtxTop() │    │ TlsCtxTop() │    │ TlsCtxTop() │
 │             │    │             │    │             │
 │ muLaunch*() │    │ muLaunch*() │    │ muMemcpy*() │
 │ QueueCmd()  │    │ QueueCmd()  │    │ QueueCmd()  │
 └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
        │                  │                  │
        ▼                  ▼                  ▼
 ┌──────────────────────────────────────────────────────┐
 │              Stream (per-stream 双线程)              │
 │  ┌──────────────┐          ┌──────────────┐         │
 │  │ AsyncSubmit  │  ──→     │  AsyncWait   │         │
 │  │ Build+Submit │  Inflight│ Wait+Cleanup │         │
 │  └──────────────┘          └──────────────┘         │
 │        ▲                                              │
 │   m_CommandList → m_MergingList → m_InflightList    │
 └──────────────────────────────────────────────────────┘
        │
        ▼
 ┌──────────────────────────────────────────────────────┐
 │                  GPU 硬件队列 (CDM / CE)              │
 └──────────────────────────────────────────────────────┘
```

**核心设计原则**:
- **调用线程不做重活**: 用户线程仅将命令入队（轻量级操作），Submit/Wait 由独立的后台线程完成
- **每 Stream 双线程**: Submit 线程负责依赖构建和提交，Wait 线程负责完成等待和清理
- **Command 状态机跨线程**: 在同一 Stream 的多个线程之间流转，通过 `atomic<Status>` 同步
- **全局 Executor 兜底**: 将资源释放等非关键路径延迟到全局 Worker 线程

---

## 2. 线程清单

### 2.1 持久线程

| 线程 | 生命周期 | 所属对象 | 核心职责 |
|------|---------|---------|---------|
| **AsyncSubmit** | Stream 创建→销毁 | `Stream::m_SubmitThread` | 消费 `m_CommandList`，构建信号量依赖链，合并命令，提交到 GPU HAL 队列 |
| **AsyncWait** | Stream 创建→销毁 | `Stream::m_WaitThread` | 等待 GPU 完成（信号量等待），检查引擎错误，输出 printf 缓冲区，标记命令完成，委托资源释放 |
| **Executor Worker** | Platform 创建→销毁 | `Platform::m_Executor` | 延迟执行 `ReleaseResources()` 和 Stream 销毁同步 |
| **PrintManager Flush** | 首个 printf kernel→最后一个释放 | `PrintManager::m_flushRoutine` | 每 1ms 轮询 GPU printf FIFO 缓冲区，格式化输出到 stdout |
| **NotificationManager** | 首次注册回调→Manager 销毁 | `NotificationManager::m_Thread` | 每 100μs 轮询设备空闲内存，低于 20% 阈值时触发异步回调 |

> 每个 Stream 创建时自动启动 2 个后台线程。用户每创建一个非阻塞 Stream，系统就增加 2 个线程。

### 2.2 按需线程

| 线程 | 触发条件 | 生命周期 |
|------|---------|---------|
| **MUgdb Monitor** | 检测到含 trap 指令的 kernel 启动且调试器未就绪 | 调试器连接后自动销毁 |

---

## 3. 锁体系

### 3.1 锁层级（由外到内）

```
Platform::CriticalBase::m_sharedMutex    ← 全局平台状态
    │
    ├─ Device::CriticalBase::m_sharedMutex  ← 设备级状态
    │     │
    │     └─ Context::CriticalBase::m_sharedMutex  ← Context 级状态 (最频繁)
    │           │
    │           ├─ Stream::m_SubmitMtx          ← 命令入队/提交协调
    │           ├─ Stream::m_WaitMtx            ← 飞行命令/等待协调
    │           ├─ Stream::EngineResource.mtx   ← 命令缓冲区池
    │           │
    │           ├─ Module::m_SymbolLock         ← 模块符号表
    │           ├─ Library::m_KernelsLock       ← 内核注册表
    │           ├─ PrintManager::m_resourceMtx  ← printf 缓冲区管理
    │           │
    │           └─ Command::m_Status (atomic)   ← ★ 最低层: 跨线程状态机
    │
    └─ MemoryPool::m_Lock (recursive_mutex)     ← HAL 内存池 (分配/释放)
```

### 3.2 读写锁（shared_mutex）使用模式

所有关键对象（Platform、Device、Context）使用 `ReadWriteLockedBase<std::shared_mutex>` + `ReadLockedAccessor`/`WriteLockedAccessor` RAII 封装。

**Context::CriticalBase 是最热锁**（约 85 处读写锁定点）:

```
读操作 (并发安全):                写操作 (互斥):
  - FindStream()                  - AddStream()
  - FindModule()                  - AddModule()
  - GetMemoryByDevicePointer()    - AddMemory()
  - ValidateEvent()               - AddEvent()
  - 大部分 API 的参数校验+查找    - MapToPeers()
                                  - EnablePeer()
```

### 3.3 递归锁场景

唯一使用 `std::recursive_mutex` 的位置: `MemoryPool::m_Lock`。

**原因**: `FullAllocate()` 持有锁 → 调用 `SubAllocate()` → 可能触发 `ChunkAllocate()` → `ResourceAdd()` 需要在同一锁下操作链表。递归锁避免自死锁。

### 3.4 流级锁协作图

```
                    m_SubmitMtx                          m_WaitMtx
用户线程                 │                                    │
  │                     │                                    │
  ├─ QueueCommand() ──→ │ lock                              │
  │  push CommandList    │ unlock                            │
  │  notify SubmitCv     │                                    │
  │                     │                                    │
  ▼                     ▼                                    │
AsyncSubmit 线程 ──→   │ lock                              │
  pop CommandList        │ buildCommand()                     │
  push MergingList       │ submitMergingList()                │
                         │ unlock                            │
                         │              ──→                   │ lock
                         │               move InflightList    │ unlock
                         │               notify WaitCv        │
                         │                                    ▼
                         │                           AsyncWait 线程
                         │                              lock
                         │                              等待信号量
                         │                              Postprocess()
                         │                              m_AsyncCount--
                         │                              unlock
```

**关键点**: `m_AsyncCount` 是跨线程的 `atomic<uint32_t>`，用户在入队前 `spin yield` 直到低于容量阈值，Wait 线程完成后递减。

---

## 4. 原子操作体系

### 4.1 Command 状态机 (核心)

```cpp
// command.h:200
std::atomic<Status> m_Status;

// 状态转移: created → queued → built → submitted → completed/error
// 每次转移都是 compare_exchange_strong 确保串行化

bool Command::SetStatus(Status status) {
    Status expected = Status::created;
    if (status == Status::queued)   expected = Status::created;
    if (status == Status::built)    expected = Status::queued;
    if (status == Status::submitted) expected = Status::built;
    if (status == Status::completed) expected = Status::submitted;
    return m_Status.compare_exchange_strong(expected, status);
}
```

**使用**: 用户线程设置 `queued`，AsyncSubmit 线程设置 `built` → `submitted`，AsyncWait 线程设置 `completed`/`error`。任何线程都可以通过 `Command::Wait()` 轮询 `m_Status` 等待完成。

### 4.2 背压控制 (m_AsyncCount)

```cpp
// stream.cpp:1011-1013
while (m_AsyncCount.load() >= asyncCapacity) {
    std::this_thread::yield();  // 背压: 等待槽位
}
m_AsyncCount.fetch_add(1);      // 获取槽位
```

### 4.3 PFM 串行化锁

```cpp
// device.h:212
std::atomic_flag m_PfmSerialLock = ATOMIC_FLAG_INIT;

// context.cpp:1863-1866 — 获取
while (GetParentDevice()->GetPfmSerialLock().test_and_set(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}
// context.cpp:1912 — 释放
GetParentDevice()->GetPfmSerialLock().clear(std::memory_order_release);
```

当 PFM 启用时，**所有命令提交被串行化**，确保性能计数器转储与 kernel 启动一一对应。

### 4.4 全局 Submission ID

```cpp
// command.cpp:25
std::atomic<uint64_t> g_SubmissionId(1);  // 每提交递增
```

### 4.5 IPC Event 无锁环

```
             write_index (atomic<int>)     ← 记录进程写入后递增
             read_index  (atomic<int>)     ← 等待进程 CAS 更新
             signal[N]   (atomic<int>)     ← 每槽自旋锁 (store 1/0)
```

跨进程 IPC 事件使用纯原子操作的无锁环形缓冲区，无需系统级同步原语。

---

## 5. 线程局部存储 (TLS)

### 5.1 TlsCtxTop — Context 栈

```cpp
// driver/internal.cpp:14
static thread_local TlsData t_TlsData;

class TlsData {
    std::stack<Musa::Context*> m_CtxStack;  // ★ 每线程独立的 context 栈
    MUresult m_LastError;
    int m_MemAffinityNode;
};

// 所有入口 API 均调用 TlsCtxTop() 获取当前线程绑定的 Context
Context* CtxTop() const { return m_CtxStack.empty() ? nullptr : m_CtxStack.top(); }
```

**CUDA 语义**: `cuCtxPushCurrent` → `CtxPush`, `cuCtxPopCurrent` → `CtxPop`, `cuCtxGetCurrent` → `CtxTop`。

### 5.2 Per-Thread Default Stream

```cpp
// context.cpp:1123-1127
IStream* Context::GetPerThreadDefaultStream() const {
    static thread_local std::unordered_map<uint64_t, Stream*> perThreadStreams;
    uint64_t key = (deviceId << 32) | contextId;
    // 每线程每(设备, Context)对维护独立的默认 Stream
}
```

每个线程有自己独立的默认 Stream 实例，线程间互不干扰。

### 5.3 其他 TLS 变量

| 变量 | 用途 |
|------|------|
| `Stream::t_BeginCaptureNum` | 每线程 graph capture 嵌套深度 |
| `Stream::t_CaptureMode` | 每线程 capture 模式 |
| `ThreadInfo::info` | 每线程唯一标识 (PID/TID/API seq number) |
| `InjectionManager::t_recursive` | 防止注入库递归初始化 |

---

## 6. 场景分析

### 场景 1：单线程 + 默认 Stream

```
用户线程 (唯一)
  │
  ├─ muMemAlloc()      → Context::CriticalBase(Read) → CreateMemory()
  ├─ muLaunchKernel()  → Context::CriticalBase(Read) → QueueCommand()
  │                       ├─ m_AsyncCount++           (atomic)
  │                       └─ notify SubmitCv
  │
  │  (AsyncSubmit 被唤醒)
  ├─ [后台] Build      → Context::CriticalBase(Read) → 依赖构建
  ├─ [后台] Submit     → HAL Queue Submit → ioctl → GPU
  │
  │  (AsyncWait 被唤醒)
  ├─ [后台] Wait       → Semaphore::Wait → WaitFinish
  ├─ [后台] Postprocess → ReleaseCmdBuffer
  ├─ [后台] ReleaseRes  → Executor::QueueTask
  │
  ├─ muStreamSynchronize() → Command::Wait()
  │                            └─ 轮询 m_Status (Yield/Spin) 直到 completed
```

**锁竞争**: 极低。用户线程 + 2 个后台线程在独立数据上工作，仅在 `m_SubmitMtx` 和 `m_WaitMtx` 上有短暂的临界区。

### 场景 2：多线程并发 Launch

```
用户线程 A              用户线程 B              用户线程 C
  │                       │                       │
  ├─ LaunchKernel(1)     ├─ LaunchKernel(2)     ├─ LaunchKernel(3)
  │  QueueCommand(S1)    │  QueueCommand(S1)    │  QueueCommand(S2)
  │   lock m_SubmitMtx   │   spin/yield 直到 lock│   lock S2.m_SubmitMtx
  │   push CommandList   │   lock m_SubmitMtx    │   push CommandList
  │   unlock             │   push CommandList    │   unlock
  ▼                       │   unlock             ▼
                       │   ▼
                       │
  Stream S1:  AsyncSubmit                          Stream S2:  AsyncSubmit
  │  pop CommandList                               │  pop CommandList
  │  构建 1 和 2 的依赖链                          │  构建 3 的依赖链
  │  合并 (CanMergeTo)                              │  提交到 GPU CE
  │  提交到 GPU CDM                                │
  ▼                                                 ▼
  GPU CDM 队列                                    GPU CE 队列
```

**关键点**:
- **同一 Stream 的多线程入队**: `m_SubmitMtx` 保护 `m_CommandList` 和 `m_LastCommand`。后入队的命令自动依赖先入队的命令（通过 `SetPrevCommand`）。
- **不同 Stream**: 完全独立，无锁竞争（除了共享 `Platform::CriticalBase`/`Context::CriticalBase` 的读锁）。
- **命令合并**: AsyncSubmit 线程在构建时检查 `CanMergeTo()`，相同引擎的兼容命令可以被合并到同一个 `CmdBuffer`。

### 场景 3：Graph Capture

```
用户线程
  │
  ├─ muStreamBeginCapture(S, mode)
  │    ├─ 设置 m_CaptureGraph
  │    ├─ 记录 m_BeginThreadId (非 relaxed 模式)
  │    └─ 设置 m_CaptureStatus = ACTIVE
  │
  ├─ muLaunchKernel(S, ...)
  │    └─ GeneralLaunchKernel → CreateKernelNode → CaptureNode → 加入图
  │
  ├─ muMemcpy(S, ...)
  │    └─ GeneralMemcpy → CreateMemcpyNode → CaptureNode → 加入图
  │
  ├─ muEventRecord(S, event)
  │    └─ CaptureSetEvent → 记录 LastCapturedNodes 到 Event
  │
  └─ muStreamEndCapture(S, &graph)
       ├─ 校验: m_BeginThreadId == 当前线程 (非 relaxed)
       └─ 构建 Graph → UniversalManager 管理
            ★ 图执行时: GraphCommand → UniversalManager → 独立提交路径
```

**线程约束**:
- 默认模式下，`EndCapture` 必须与 `BeginCapture` 在同一线程调用（通过 `m_BeginThreadId` 校验）
- Relaxed 模式下允许多线程操作同一 capture stream
- 捕获期间禁止内存分配（`CreateMemory` 前检查所有 stream 的 capture 状态）

### 场景 4：多 Context + 多 Stream

```
Context A                    Context B
  │                             │
  ├─ DefaultStream             ├─ DefaultStream
  ├─ Stream A1 (non-blocking)  ├─ Stream B1 (non-blocking)
  └─ Stream A2 (blocking)      └─ Stream B2 (blocking)

Context A::CriticalBase(shared_mutex)    Context B::CriticalBase(shared_mutex)
  独立锁域                                 独立锁域
        │                                      │
        └────────── 共享 (读) ──────────────────┘
              Device::CriticalBase(shared_mutex)
                    │
                    └────────── 共享 (读) ──────┘
                  Platform::CriticalBase(shared_mutex)
```

**隔离性**:
- **Stream 级**: 完全隔离（独立线程、独立锁）
- **Context 级**: 通过 shared_mutex 读共享写互斥
- **Device 级**: 多 Context 共享设备，但资源分配走独立 MemMgr/Pool
- **Platform 级**: 全局唯一，所有跨设备操作需要获取

### 场景 5：IPC 跨进程

```
进程 A                               进程 B
  │                                    │
  ├─ muIpcGetMemHandle(mem, &handle)  │
  │   → ExportIpcHandle()             │
  │                                    │
  │  ② OS 传递 handle                  │
  │───────────────────────────────────→│
  │                                    │
  │                              muIpcOpenMemHandle(&dptr, handle)
  │                                    ├─ memoryTypeIpcImport → IpcImportAlloc
  │                                    ├─ MapToPeers (自动 Peer 映射)
  │                                    └─ TrackMemory (注册到 MemoryTracker)
  │                                    │
  │  ③ 两端并发访问同一物理内存          │
  │                                    │
  │                                    │ muIpcCloseMemHandle(dptr)
  │                                    └─ Synchronize → DestroyMemory
  │
  ├─ muIpcGetEventHandle(event, &h)   │
  │   → 导出 ring buffer 共享内存       │
  │                                    │
  │  ② OS 传递 handle                  │
  │───────────────────────────────────→│
  │                                    │
  │                              muIpcOpenEventHandle(&event, handle)
  │                                    ├─ CreateEvent(INTERPROCESS|DISABLE_TIMING)
  │                                    └─ ImportIpcHandle → 映射 ring buffer
  │                                    │
  │  ③ muEventRecord(eventA, stream)   │
  │     write_index++                  │
  │     signal[slot]=1                 │
  │                                    │
  │                              muStreamWaitEvent(stream, eventA)
  │                                    └─ CmdWaitEvent → 等待 signal[slot]=0
  │
  │  ④ 跨进程同步完成                   │
```

**同步机制**: IPC Event 使用共享内存中的 lock-free ring buffer:
- `write_index` (atomic): 记录端写入后原子递增
- `read_index` (atomic): 等待端通过 CAS 更新
- `signal[N]` (atomic): 每槽自旋锁 (store 1/0)

---

## 7. 等待策略

### 7.1 ScheduleMode 三态

```cpp
// util/utilSched.h
enum class ScheduleMode {
    Spin,   // asm("rep; nop") / asm("yield") — CPU pause
    Yield,  // std::this_thread::yield() — 交出时间片
    Sleep   // std::this_thread::sleep_for(100us) — OS 调度
};
```

**选择逻辑**:
- `Command::Schedule(mode, predicate)` — 轮询 `m_Status` 时使用
- 模式继承自 `Context::m_WaitMode`，可由 `cuCtxSetFlags` 配置
- `musaDeviceScheduleSpin` (1): 最低延迟，最高 CPU 占用
- `musaDeviceScheduleYield` (2): 平衡
- `musaDeviceScheduleBlockingSync` (4): 映射到 Yield

### 7.2 关键旋转点

| 位置 | 模式 | 目的 |
|------|------|------|
| `Command::Build()` | Yield | 等待依赖命令达到 built/submitted |
| `QueueCommand()` | Yield | 背压: 飞行命令数超限 |
| `Semaphore::Wait()` | Yield + 超时检测 | 等待 GPU 信号量 |
| `Device::FenceWait()` | Yield + 错误轮询 | 等待 GPU fence |
| `IPC Event Signal/Wait` | Yield | 跨进程事件同步 |
| `PFM Serial Lock` | Sleep(1ms) | 性能监控串行化 |

---

## 8. 当前设计的潜在问题

### 8.1 线程爆炸

每个 Stream 创建 2 个持久线程（AsyncSubmit + AsyncWait）。如果有 100 个 Stream，就是 200 个线程。高并发场景下可能触发 OS 线程数限制。

**现状**: `m_AsyncCount` 背压机制（默认 `streamAsyncCapacity`）限制了飞行命令数，间接限制了有效并发度。

### 8.2 PFM 串行化瓶颈

当 PFM 启用时，**所有 Command 提交被 `PfmSerialLock` 串行化**。在 profiling 场景下，一个 kernel launch 的延迟可能直接影响下一个 Stream 的提交。

**现状**: 这是 profiling 的必要代价。可通过批量提交（合并）减少 `test_and_set` 调用次数。

### 8.3 Context::CriticalBase 写锁争用

任何修改操作（AddStream、AddModule、AddMemory、EnablePeer）都需要获取写锁，阻塞所有并发读者。在高频分配/释放场景下可能成为瓶颈。

**现状**: 多数 API 调用仅需读锁（FindXxx），写锁仅在创建/销毁时短暂持有。

### 8.4 Hal MemoryPool 递归锁

`MemoryPool::m_Lock` 是 `recursive_mutex`，每次 SubAllocate 和 Free 都需获取。多 Stream 共享同一 Pool 时，SubAllocate 和 Free 在锁内竞争。

**现状**: 每个属性组合（property + heap + type）有独立的 Pool，不同属性的分配走不同 Pool。但 `muMemAlloc_v2` 所有分配共享同一 Pool。

### 8.5 全局 Executor 单线程

`Platform::m_Executor` 只有一个 Worker 线程处理所有 Stream 的资源释放。大量 Stream 并发完成时，`ReleaseResources()` 会排队。

**现状**: `ReleaseResources()` 主要是 `FreeInternalMem`（内存释放），通常轻量。但极端场景下可能成为延迟来源。

### 8.6 命令合并的线程间依赖传递

`Command::Build()` 中 `while (dependant->GetStatus() > waitStage) yield()` 是**跨流依赖**的实现方式。当流 A 的命令依赖流 B 的命令时，流 A 的 AsyncSubmit 会在 Build 阶段自旋等待流 B 的命令完成构建。

**风险**: 多个流形成依赖链时可能导致级联等待。

---

## 9. 优化方向

### 9.1 线程池替代 per-Stream 线程对

**现状**: 每个 Stream 有独立的 `std::thread`。
**优化**: 将 AsyncSubmit 和 AsyncWait 的任务提交到共享线程池。

```cpp
// 设想:
class Stream {
    // 不再有 m_SubmitThread 和 m_WaitThread
    // 改为:
    void ScheduleSubmit();  // → GlobalSubmitPool::Enqueue(this);
    void ScheduleWait();    // → GlobalWaitPool::Enqueue(this);
};
```

**收益**: 
- N 个 Stream 从 2N 个线程降为固定数量（如 CPU 核心数）
- 减少上下文切换和 OS 线程管理开销
- 可以实现负载均衡（空闲线程可以处理多个 Stream）

**代价**: 需要工作窃取或优先级调度避免头阻塞。

### 9.2 Lock-free 命令列表

**现状**: `m_CommandList`、`m_MergingList` 等受 `m_SubmitMtx` 保护。
**优化**: 使用 lock-free SPSC (Single Producer Single Consumer) 队列。

```cpp
// 设想: moodycamel::ReaderWriterQueue 或 folly::MPMCQueue
// producer: 用户线程 QueueCommand
// consumer: AsyncSubmit 线程
```

**收益**: 消除 `m_SubmitMtx` 在入队路径上的争用。
**约束**: 需要确保多个用户线程对同一 Stream 的入队互不冲突（或使用 MPSC 队列）。

### 9.3 Per-Core 或 Per-NUMA Pool 分片

**现状**: 所有 `muMemAlloc_v2` 的 SubAllocation 共享同一个 `MemoryPool`（按 property hash）。
**优化**: 按 CPU 核心或 NUMA 节点分片 Pool。

```cpp
// 设想:
class ShardedMemoryPool {
    MemoryPool m_Pools[SHARD_COUNT];  // 16 或 32 分片
    MemoryPool& GetPool() { return m_Pools[current_core % SHARD_COUNT]; }
};
```

**收益**: SubAllocate 路径上的锁竞争降低 N 倍。
**代价**: 每个分片的利用率可能不均衡（需要负载感知的分配策略）。

### 9.4 命令构建的预计算

**现状**: `DispatchCommand::Build()` 中大量 CPU 工作（隐式资源分配、const buffer 填充、依赖等待）全部在 AsyncSubmit 线程串行完成。
**优化**: 将可预先计算的部分提前到用户线程的 `QueueCommand` 阶段。

```cpp
// 设想:
// QueueCommand 阶段:
command->PreBuild();  // 乐观地分配隐式资源、写入 const buffer
                      // 如果依赖尚未满足 → 回退到 Build 阶段完成

// Build 阶段:
command->Build();     // 仅完成依赖等待 + 信号量配置
```

**收益**: 减少 AsyncSubmit 线程的 CPU 占用，降低提交延迟。

### 9.5 Executor 多线程 + 优先级

**现状**: 全局 Executor 单线程。
**优化**: 多 Worker + 优先级队列。

```cpp
// 设想:
class PriorityExecutor {
    std::vector<std::thread> m_Workers;  // N 个 Worker
    PriorityQueue<Task> m_Tasks;         // 高优先级: 内存在线释放
};
```

**收益**: 大量 Stream 并发完成时资源释放不会排队阻塞。

### 9.6 SubAllocate 热路径去锁

**现状**: `MemoryPool::SubAllocate()` 持有 `recursive_mutex`。
**优化**: 在 high-frequency 场景下使用 lock-free free list。

```cpp
// 设想: 用 CAS 实现 lock-free free list
struct FreeNode {
    DevSize base, size;
    std::atomic<FreeNode*> next;
};
// SubAllocate: CAS 从 free list 取出节点
// Free: CAS 将节点归还到 free list
```

**收益**: SubAllocate 和 Free 路径完全无锁，多 Stream 并发分配/释放时零竞争。

### 9.7 信号量等待的硬件辅助

**现状**: AsyncWait 线程通过 `Semaphore::Wait()` 轮询（Yield 模式），CPU 一直活跃。
**优化**: 利用 OS 级等待原语。

```cpp
// 设想:
if (m_EnableUserQueue(engine)) {
    // 使用 eventfd / KMD 的 IRQ 通知
    poll(deviceFd, POLLIN, timeoutMs);  // OS 阻塞等待
} else {
    // 回退到 spin+yield
    Semaphore::Wait(...);
}
```

**收益**: AsyncWait 线程在 GPU 空闲时进入 OS 等待状态，降低 CPU 功耗。

### 9.8 Context 锁粒度细化

**现状**: `Context::CriticalBase::m_sharedMutex` 保护所有 Context 级数据。
**优化**: 拆分锁域。

```cpp
// 设想:
class Context {
    std::shared_mutex m_StreamLock;    // 仅保护 Stream 集合
    std::shared_mutex m_MemoryLock;    // 仅保护 Memory 集合
    std::shared_mutex m_ModuleLock;    // 仅保护 Module 集合
    // ...
};
```

**收益**: 分配（需要 Memory 锁）和 Launch（需要 Stream 锁）不再互相阻塞。
**代价**: 增加内存开销和获取多个锁的复杂性，需定义明确的锁序避免死锁。

---

## 10. 总结

| 维度 | 当前设计 | 适用场景 | 优化方向 |
|------|---------|---------|---------|
| **线程模型** | per-Stream 双线程 | 低并发 (≤32 Streams) | 线程池化 |
| **命令入队** | mutex + 条件变量 | 通用 | lock-free 队列 |
| **内存分配** | recursive_mutex + 单 Pool | 低频率分配 | Sharded Pool / lock-free |
| **GPU 等待** | Yield 轮询 | 低延迟要求 | OS 级等待辅助 |
| **资源释放** | 单线程 Executor | 轻量释放 | 多 Worker + 优先级 |
| **PFM** | 全局串行锁 | profiling 专用 | (无需优化，intentional) |
| **Context 锁** | 单一 shared_mutex | 多数读操作 | 拆分锁域 |
| **IPC 事件** | lock-free ring buffer | ✅ 已是最优 | — |

**优化优先级建议**:
1. 🔴 **线程池化** (线程数暴涨 → 固定)
2. 🟡 **MemoryPool 分片化** (高频分配锁竞争)
3. 🟡 **Context 锁粒度细化** (分配与 Launch 互不阻塞)
4. 🟢 **Executor 多线程** (资源释放排队)
5. 🟢 **Lock-free 命令列表** (入队路径极致优化)