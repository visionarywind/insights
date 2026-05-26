# MUPTI Hook 点与实现机制分析

> **MUPTI = MUSA Profiling Tools Interface**，MUSA 平台的 CUPTI 等价物。
> 它在 musa 驱动中插入 hook 点，供外部分析工具（profiler、debugger、sanitizer）收集性能数据和行为信息。
>
> 分析基于源码：`linux-ddk/musa/src/driver/mupti/`、`musa/core/`、`MUSA-Runtime/src/mupti/`

---

## 1. 架构总览：双轨制 Hook 体系

MUPTI 的 hook 体系分为两层：

```
                          ┌─────────────────────────────┐
                          │    MUPTI 分析工具            │
                          │   (libmupti.so)             │
                          └──────────┬──────────────────┘
                                     │ 注册回调函数指针
                                     │ (EnableMUptiDriver/EnableMUptiRuntime)
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │ Driver API Trace │  │ Runtime API Trace│  │ 语义 Hook 层     │
   │ (822 CBIDs)      │  │ (531 CBIDs)      │  │ (tracepoints.h)  │
   │ mu_entry.cpp     │  │ wrappers_generated│  │ musa/core/*.cpp  │
   └──────────────────┘  └──────────────────┘  └────────┬─────────┘
                                                        │
              ┌──────────────────────────────────────────┘
              ▼
   ┌──────────────────────────────────────────────────┐
   │          G_MUPTI_DRIVER_HOOKS                     │
   │  - ready (atomic<bool>)                          │
   │  - hooks (MUptiDriverHooks: ~40 个函数指针)       │
   │  - 快速路径: CheckMUptiEnabled() → atomic load    │
   └──────────────────────────────────────────────────┘
```

**两层 Hook**：

| 层 | 触发位置 | 粒度 | CBID 数量 |
|----|----------|------|-----------|
| **API 级** | `mu_entry.cpp`（Driver）/ `musa_wrappers_generated.cpp`（Runtime） | 每个 API 调用 | 822 + 531 = 1353 |
| **语义级** | `musa/core/` 的命令、流、上下文代码 | 内核启动、memcpy、同步等关键事件 | ~40 个语义钩子 |

---

## 2. 核心基础设施

### 2.1 全局 Hook 存储

**文件**：`musa/src/driver/mupti/hooks.h`

```cpp
struct MUptiDriverHookStorage {
    std::atomic<bool> ready;          // 全局开关：false → 零开销快速路径
    MUpti::MUptiDriverHooks hooks;    // ~40 个函数指针
    uintptr_t reserved[8];
};

extern MUptiDriverHookStorage G_MUPTI_DRIVER_HOOKS;  // 全局单例
```

**启用/禁用**（`hooks.cpp`）：
```cpp
void EnableMUptiDriver(MUpti::InitializeMUptiDriverHooks_fn initer) {
    initer(&G_MUPTI_DRIVER_HOOKS.hooks);  // MUPTI 库填充所有函数指针
    G_MUPTI_DRIVER_HOOKS.ready.store(true, std::memory_order_release);
}

void DisableMUptiDriver() {
    G_MUPTI_DRIVER_HOOKS.ready.store(false, std::memory_order_release);
}
```

**关键设计**：
- `ready` 是 `atomic<bool>`，未启用时 hook 路径只需一次原子读 → **零开销**
- 函数指针表由 MUPTI 库在 `EnableMUptiDriver()` 时一次性注册
- `DisableMUptiDriver()` 不清理函数指针，只设置 `ready=false`

### 2.2 Hook 函数指针表

**文件**：`musa/src/musa_shared_include/export_table.h:927`

`MUptiDriverHooks` 结构体包含 **~40 个函数指针**，按功能分为以下几类：

```
Enter/Exit 对（资源追踪）：
  EnterMemcpy, ExitMemcpy            — memcpy 操作
  EnterMemset, ExitMemset            — memset 操作
  EnterMemoryAtomic, EnterMemoryAtomicV2 — 原子操作
  EnterMemoryAtomicValue             — 原子值返回操作
  EnterMemoryTransfer, EnterMemoryTransferV2 — 跨设备传输

Register 模式（内核/事件注册）：
  RegisterKernel, RegisterKernelV2   — 内核启动注册
  RegisterEventSynchronize           — 事件同步注册
  RegisterStreamWaitEvent            — 流等待事件注册
  RegisterStreamSynchronize          — 流同步注册
  RegisterContextSynchronize         — 上下文同步注册

Mark/Assign 模式（时间戳 + 关联）：
  MarkKernelQueued                   — 标记"已排队"时间戳
  MarkKernelSubmitted                — 标记"已提交"时间戳
  AssignKernelToKick                 — 内核 → kick 关联
  AssignSubmissionToCorrelation      — 提交 → correlation ID 关联
  MarkCommandBeginEnd, MarkCommandBeginEndV2  — 命令起止

Start/Stop 模式（同步等待追踪）：
  StartEventSynchronize, StopEventSynchronize
  StartStreamWaitEvent, StopStreamWaitEvent
  StartStreamSynchronize, StopStreamSynchronize
  StartContextSynchronize, StopContextSynchronize
  StartHostMemcpy, StopHostMemcpy
  StartHostMemset, StopHostMemset

Graph 模式（CUDA Graph 追踪）：
  RegisterGraphTrace                 — 注册 graph trace
  MarkGraphTraceBegin, MarkGraphTraceEnd  — graph trace 边界
  RegisterGraphKernel                 — graph 内 kernel 节点
  RegisterGraphMemcpy, RegisterGraphMemset
  RegisterGraphMemoryAtomic, RegisterGraphMemoryAtomicValue
  RegisterGraphMemoryTransfer
  MarkGraphNodeBeginEnd, MarkGraphNodeBeginEndV2  — graph node 起止
  CheckGraphTraceEnabled             — 检查是否启用 graph trace

生命周期：
  CreateContext, DestroyContext       — 上下文创建/销毁
  CreateStream, DestroyStream         — 流创建/销毁
```

### 2.3 快速路径：`CheckMUptiEnabled()`

```cpp
// tracepoints.h:9
inline bool CheckMUptiEnabled() {
    return G_MUPTI_DRIVER_HOOKS.ready.load(std::memory_order_acquire);
}
```

每个 hook 函数的第一行都是 `if (CheckMUptiEnabled())`。未启用时，这是一次原子加载 + 分支 → **数个 CPU 周期**。

### 2.4 Accessor 导出表

**文件**：`musa/src/driver/mu_entry.cpp:1051-1265`

MUPTI 库不在 musa 驱动内部，而是通过导出表访问驱动的内部类型。`mu_entry.cpp` 导出 **20 个 accessor 结构体**：

| Accessor 表 | 提供的能力 |
|-------------|-----------|
| `CommandBaseAccessors` | GetStream, GetCorrelationId, GetQueuedTimestamp, GetType, GetSubmittedTimestamp, GetGraphId, GetGraphNodeId |
| `DispatchCommandAccessors` | GetFunction, GetGridSize, GetBlockSize, GetDynamicSharedMemUsage |
| `MemcpyCommandAccessors` | GetCopyKind, GetSrcMemKind, GetDstMemKind, GetSize |
| `MemoryTransferCommandAccessors` | GetSize, GetSizeV2 |
| `MemsetCommandAccessors` | (memset 参数) |
| `MemoryAtomicCommandAccessors` | (atomic 参数) |
| `MemoryAtomicValueCommandAccessors` | (atomic value 参数) |
| `ContextAccessors` | GetId, GetDevice, GetContext, GetDeviceId, CheckEngineSyncCap |
| `StreamAccessors` | GetId, GetContext |
| `DeviceAccessors` | GetId |
| `FunctionAccessors` | GetName, GetStaticSharedMemSize, GetLocalMemSize |
| `EventAccessors` | (event 属性) |
| `GraphAccessors` | (graph 属性) |
| `ThreadLocalInfoAccessors` | ProcessId, ThreadId, CorrelationId |
| `ProfilerControllers` | 控制 profiler 状态 |
| `ProfilerPfmControllers` | PFM 计数器控制 |
| `PCSamplingControllers` | PC 采样控制 |
| `InternalAccessors` | 内部辅助 |
| `MsysAccessors` | 系统级追踪 |
| `MUptiDriverControllers` | EnableMUptiDriver, DisableMUptiDriver |

这种 **accessor 模式** 是设计的关键：
- MUPTI 库不需要了解 `Musa::DispatchCommand` 的内部结构
- 通过函数指针间接访问所有需要的字段
- musa 驱动内部类型变更时，只需更新 accessor 函数实现

---

## 3. 三类 Hook 模式详解

### 3.1 Enter/Exit 模式（memcpy、memset、atomic）

**用途**：追踪异步操作的完整生命周期。

**示例 — memcpy**：

```
用户调用 muMemcpyAsync
  → MemcpyCommand 构造
    → m_ptiCtx = MUpti::EnterMemcpy(this)    // Hook: 注册 memcpy 开始
      → MUPTI 创建 tracking context，记录 correlation ID
  → ... (命令排队、提交、执行) ...
  → MemcpyCommand::Execute / 回调
    → MUpti::MarkCommandBeginEnd(m_ptiCtx)   // Hook: 记录 GPU 时间戳
    → MUpti::ExitMemcpy(m_ptiCtx)            // Hook: 标记操作完成
```

**关键代码**（`memcpyCommand.cpp`）：
```cpp
// 构造时
m_ptiCtx = MUpti::EnterMemcpy(this);

// 执行完成时
MUpti::MarkCommandBeginEnd(m_ptiCtx, this);
MUpti::ExitMemcpy(m_ptiCtx);
```

**`EnterMemcpy` 的职责**：MUPTI 库解析 `command` 对象（通过 accessor 表），提取源/目标类型、大小、流 ID，创建内部 tracking 记录，返回 context 指针。

**`ExitMemcpy` 的职责**：标记操作完成，计算耗时，生成 activity 记录（供 `muptiActivityGetNextRecord` 消费）。

### 3.2 Register/Mark/Assign 模式（kernel launch）

**用途**：追踪内核启动的完整流水线：排队 → 提交 → GPU 执行。

```
用户调用 muLaunchKernel
  → DispatchCommand 构造
    → m_ptiCtx = MUpti::RegisterKernelV2(this)   // Hook: 注册内核，生成 uniqueId
    → MUpti::RegisterKernel(this)                 // Hook: 注册内核名/参数
  → Stream::CmdDispatch()
    → MUpti::MarkKernelQueued(correlationId)      // Hook: 记录"排队"时间戳
  → Stream::AsyncSubmit()
    → MUpti::MarkKernelSubmitted(correlationId)   // Hook: 记录"提交到 GPU"时间戳
    → MUpti::AssignKernelToKick(uniqueId, submissionId)  // 内核 → kick 关联
    → MUpti::AssignSubmissionToCorrelation(correlationId, submissionId)
```

**关键时间戳**（`stream.cpp`）：

```cpp
// 平台差异：PH 平台从命令对象读时间戳，QY2 从 MUPTI 读
// On PH, queued timestamp can be read by mupti in MUpti::MarkCommandBeginEnd
// Only call MUpti::MarkKernelQueued for QY2 and before
MUpti::MarkKernelQueued(command->GetCorId());

// On PH, submitted timestamp can be read by mupti in MUpti::MarkCommandBeginEnd
// Only call MUpti::MarkKernelSubmitted for QY2 and before
MUpti::MarkKernelSubmitted(merged->GetCorId());

// 内核与 kick 的关联
MUpti::AssignKernelToKick(uniqueId, submissionId);
MUpti::AssignSubmissionToCorrelation(uniqueId, submissionId);
```

**`AssignKernelToKick` / `AssignSubmissionToCorrelation` 的意义**：将"内核 → GPU 提交 → correlation ID → 硬件 kick"四者关联起来。这使得 MUPTI 能把 GPU 硬件计数器采集到的数据与用户 API 调用对应上。

### 3.3 Register/Start/Stop 模式（同步操作）

**用途**：追踪同步等待操作（流同步、事件同步、上下文同步）。

```
用户调用 muStreamSynchronize
  → Stream::Synchronize()
    → ctx = MUpti::RegisterStreamSynchronize(correlationId, stream)  // Hook: 注册
    → MUpti::StartStreamSynchronize(ctx)   // Hook: 开始等待
    → ... (实际等待 GPU) ...
    → MUpti::StopStreamSynchronize(ctx)    // Hook: 等待完成
```

**关键代码**（`stream.cpp`）：
```cpp
auto muptiContext = MUpti::RegisterStreamSynchronize(
    Util::ThreadInfo::Get().ApiSeqNum(), this);
MUpti::StartStreamSynchronize(muptiContext);
// ... GPU 同步等待 ...
MUpti::StopStreamSynchronize(muptiContext);
```

**三个函数的职责**：
- `Register*`：创建 tracking context，记录 correlation ID
- `Start*`：记录等待开始时间戳
- `Stop*`：记录等待结束时间戳，计算阻塞时长

### 3.4 生命周期 Hook

**用途**：追踪 GPU 资源（context、stream）的创建和销毁。

```cpp
// context.cpp
MUpti::CreateContext(this);   // context 创建后
MUpti::DestroyContext(this);  // context 销毁前
MUpti::CreateStream(stream);  // stream 创建后
MUpti::DestroyStream(stream); // stream 销毁前
```

这些 hook 让 MUPTI 知道哪些 context/stream 是活跃的，用于 activity 记录的 context/stream ID 解析。

---

## 4. 所有 Hook 位点（21 个文件）

### 4.1 命令执行层（command/）

| 文件 | Hook 调用 | 用途 |
|------|-----------|------|
| `command/dispatchCommand.cpp` | `RegisterKernelV2`, `RegisterKernel`, `MarkCommandBeginEnd` | 内核启动 |
| `command/memcpyCommand.cpp` | `EnterMemcpy`, `MarkCommandBeginEnd`, `ExitMemcpy` | memcpy |
| `command/memsetCommand.cpp` | `EnterMemset`, `MarkCommandBeginEnd`, `ExitMemset` | memset |
| `command/memoryAtomicCommand.cpp` | `EnterMemoryAtomicV2`, `MarkCommandBeginEnd` | 原子操作 |
| `command/memoryAtomicValueCommand.cpp` | `EnterMemoryAtomicValue`, `MarkCommandBeginEnd` | 原子值返回 |
| `command/memoryTransferCommand.cpp` | `EnterMemoryTransferV2`, `MarkCommandBeginEnd` | 跨设备传输 |
| `command/graphCommand.cpp` | `RegisterGraphTrace`, `RegisterGraphKernel/Memcpy/Memset/Atomic/Transfer`, `MarkGraphTraceBegin/End`, `MarkGraphNodeBeginEndV2` | 图执行追踪 |
| `command/barrierCommand.cpp` | `MarkCommandBeginEnd` | GPU 屏障 |
| `command/command.cpp` | `MarkCommandBeginEnd` | 通用命令 |
| `command/dispatchRayCommand.cpp` | `MarkCommandBeginEnd` | 光线追踪调度 |
| `command/accelStructBuildCommand.cpp` | `MarkCommandBeginEnd` | 加速结构构建 |
| `command/accelStructCopyCommand.cpp` | `MarkCommandBeginEnd` | 加速结构拷贝 |
| `command/accelStructEmitCommand.cpp` | `MarkCommandBeginEnd` | 加速结构发射 |

### 4.2 流与上下文层

| 文件 | Hook 调用 | 用途 |
|------|-----------|------|
| `core/stream.cpp` | `RegisterStreamSynchronize`, `Start/StopStreamSynchronize`, `RegisterStreamWaitEvent`, `Start/StopStreamWaitEvent`, `MarkKernelQueued`, `MarkKernelSubmitted`, `AssignKernelToKick`, `AssignSubmissionToCorrelation` | 流同步、内核提交关联 |
| `core/context.cpp` | `CreateContext`, `DestroyContext`, `CreateStream`, `DestroyStream`, `RegisterContextSynchronize`, `Start/StopContextSynchronize`, `RegisterStreamWaitEvent` | 资源生命周期、上下文同步 |
| `core/event.cpp` | `RegisterEventSynchronize`, `Start/StopEventSynchronize` | 事件同步 |

### 4.3 拷贝管理器

| 文件 | Hook 调用 | 用途 |
|------|-----------|------|
| `copyManager2/copyManager2.cpp` | `MarkCommandBeginEnd` | 拷贝管理 |
| `copyManager2/cpuCopyManager/cpuCopyManager.cpp` | `StartHostMemcpy`, `StopHostMemcpy` | CPU 端拷贝计时 |
| `copyManager2/dmaCopyManager/dmaCopyManager.cpp` | `StartHostMemcpy`, `StopHostMemcpy` | DMA 端拷贝计时 |

### 4.4 图执行

| 文件 | Hook 调用 | 用途 |
|------|-----------|------|
| `graph/graph1/graphExec.cpp` | `CheckGraphTraceEnabled`, `MarkGraphNodeBeginEndV2` | 图节点执行 |
| `graph/graph1/universalManager.cpp` | 图管理相关 | 通用图管理 |

---

## 5. API 级 Tracing（CBID 体系）

除了内核级的语义 hook，MUPTI 还提供每个 API 调用的进入/退出追踪。

### 5.1 Driver API CBID（822 个）

**文件**：`musa/src/musa_shared_include/mupti/mupti_driver_cbid.h`

每个 musa Driver API 有一个唯一的 CBID：

```c
typedef enum MUpti_driver_api_trace_cbid_enum {
    MUPTI_DRIVER_TRACE_CBID_muInit                      = 1,
    MUPTI_DRIVER_TRACE_CBID_muMemAlloc                  = 29,
    MUPTI_DRIVER_TRACE_CBID_muMemcpyHtoD                = 43,
    MUPTI_DRIVER_TRACE_CBID_muLaunchKernel              = 92,
    MUPTI_DRIVER_TRACE_CBID_muStreamSynchronize         = 132,
    // ... 共 822 个
    MUPTI_DRIVER_TRACE_CBID_SIZE                        = 822,
};
```

### 5.2 Runtime API CBID（531 个）

**文件**：`musa/src/musa_shared_include/mupti/mupti_runtime_cbid.h`

```c
typedef enum MUpti_runtime_api_trace_cbid_enum {
    MUPTI_RUNTIME_TRACE_CBID_musaMalloc_v3020           = 20,
    MUPTI_RUNTIME_TRACE_CBID_musaMemcpy_v3020           = 31,
    MUPTI_RUNTIME_TRACE_CBID_musaLaunchKernel_v3020     = 92,
    // ... 共 531 个
    MUPTI_RUNTIME_TRACE_CBID_SIZE                       = 531,
};
```

**总计 1353 个 API 级 trace 点**。

### 5.3 API Trace 实现机制

API 级的 Enter/Exit 追踪不在 musa 驱动代码中实现，而是：
- **Driver API**：`mu_entry.cpp` 的 API 包装中调用 `MUpti::EnterDriverApi` / `ExitDriverApi`
- **Runtime API**：`musa_wrappers_generated.cpp` 的 `ApiInvocationGuard` 中调用 `EnterRuntimeApi` / `ExitRuntimeApi`

这些 enter/exit 函数由 MUPTI 库注册到 `MUptiDriverHooks` / `MUptiRuntimeHooks` 函数指针表中。

---

## 6. 三套 Tracepoint 体系

musa 驱动中有三套独立的 tracepoint 系统，共用相同的 `hooks.h → G_xxx_DRIVER_HOOKS → ready + 函数指针表` 架构：

| 系统 | 目录 | 用途 | 函数数量 |
|------|------|------|----------|
| **MUPTI** | `musa/src/driver/mupti/` | 性能剖析（profiling/tracing） | ~40 个 hook |
| **MUASAN** | `musa/src/driver/muasan/` | GPU 内存安全检测（AddressSanitizer） | ~10 个 hook |
| **MUGDB** | `musa/src/driver/mugdb/` | GPU 调试器（GDB for GPU） | ~10 个 hook |

三套系统使用**完全相同的基础设施模式**：
```cpp
// 通用模式：
G_XXX_DRIVER_HOOKS.ready.load(...)   // 快速路径检查
G_XXX_DRIVER_HOOKS.hooks.XXX(...)   // 调用已注册的回调
```

---

## 7. 完整数据流：从一个 muLaunchKernel 到 MUPTI 记录

```
1. 用户调用 muLaunchKernel
   ├─ mu_entry.cpp: API 包装
   │   └─ EnterDriverApi(CBID=92) → MUPTI 记录 API 进入
   │
2. DispatchCommand 构造
   ├─ MUpti::RegisterKernelV2(this)      → 生成 uniqueId, 解析 kernel 名/参数
   └─ MUpti::RegisterKernel(this)        → 注册 kernel 元数据
   │
3. Stream::CmdDispatch()
   └─ MUpti::MarkKernelQueued(corrId)    → 记录 queued 时间戳
   │
4. Stream::AsyncSubmit()
   ├─ MUpti::MarkKernelSubmitted(corrId) → 记录 submitted 时间戳
   ├─ MUpti::AssignKernelToKick(uid, sid)→ 内核-kick 关联
   └─ MUpti::AssignSubmissionToCorrelation(corrId, sid)
                                         → 提交-correlation 关联
   │
5. 命令执行完成回调
   └─ MUpti::MarkCommandBeginEnd(ctx, cmd)→ 记录 GPU 起止时间戳
   │
6. mu_entry.cpp: API 返回
   └─ ExitDriverApi(CBID=92) → MUPTI 记录 API 退出

MUPTI 输出：
  ┌─ Activity API record: muLaunchKernel, start=100ns, end=500μs
  ├─ Kernel activity: "myKernel", queued=200ns, submitted=500ns
  ├─ Kernel activity: GPU start=2μs, GPU end=150μs
  └─ Correlation: kernel↔submission↔kick (用于 GPU 硬件计数器关联)
```

---

## 8. 相关源文件索引

### 核心基础设施
| 文件 | 内容 |
|------|------|
| `musa/src/driver/mupti/tracepoints.h` | **主 hook 入口**：405 行内联包装函数 |
| `musa/src/driver/mupti/hooks.h` | 全局存储 `G_MUPTI_DRIVER_HOOKS` 声明 |
| `musa/src/driver/mupti/hooks.cpp` | `EnableMUptiDriver` / `DisableMUptiDriver` 实现 |
| `musa/src/musa_shared_include/export_table.h:927-1003` | `MUptiDriverHooks` 结构体（~40 个函数指针） |
| `musa/src/musa_shared_include/mupti/mupti_driver_cbid.h` | 822 个 Driver API CBID |
| `musa/src/musa_shared_include/mupti/mupti_runtime_cbid.h` | 531 个 Runtime API CBID |
| `musa/src/driver/mu_entry.cpp:1051-1265` | **Accessor 导出表**：20 个 accessor 结构体 |

### Hook 插入点（21 个文件）
| 文件 | Hook 数量 | 覆盖内容 |
|------|----------|----------|
| `musa/core/command/dispatchCommand.cpp` | 3 | 内核启动 |
| `musa/core/command/memcpyCommand.cpp` | 3 | memcpy |
| `musa/core/command/memsetCommand.cpp` | 3+ | memset |
| `musa/core/command/graphCommand.cpp` | 13 | Graph 执行 |
| `musa/core/command/memoryAtomicCommand.cpp` | 2 | 原子操作 |
| `musa/core/command/memoryAtomicValueCommand.cpp` | 2 | 原子值返回 |
| `musa/core/command/memoryTransferCommand.cpp` | 2 | 跨设备传输 |
| `musa/core/command/barrierCommand.cpp` | 1 | 屏障 |
| `musa/core/command/command.cpp` | 1 | 通用命令 |
| `musa/core/command/dispatchRayCommand.cpp` | 1 | 光追 |
| `musa/core/command/accelStructBuildCommand.cpp` | 1 | AS 构建 |
| `musa/core/command/accelStructCopyCommand.cpp` | 1 | AS 拷贝 |
| `musa/core/command/accelStructEmitCommand.cpp` | 1 | AS 发射 |
| `musa/core/stream.cpp` | 14 | 流同步、内核提交关联 |
| `musa/core/context.cpp` | 9 | 资源生命周期、上下文同步 |
| `musa/core/event.cpp` | 3 | 事件同步 |
| `musa/core/copyManager2/copyManager2.cpp` | 1 | 拷贝管理 |
| `musa/core/copyManager2/cpuCopyManager/cpuCopyManager.cpp` | 2 | CPU 拷贝计时 |
| `musa/core/copyManager2/dmaCopyManager/dmaCopyManager.cpp` | 2 | DMA 拷贝计时 |
| `musa/core/graph/graph1/graphExec.cpp` | 1+ | 图执行 |
| `musa/core/graph/graph1/universalManager.cpp` | 1+ | 图管理 |

### 其他 Trace 体系
| 文件 | 内容 |
|------|------|
| `musa/src/driver/muasan/tracepoints.h` | MUASAN hook 点（内存安全检测） |
| `musa/src/driver/mugdb/tracepoints.h` | MUGDB hook 点（GPU 调试器） |
| `MUSA-Runtime/src/mupti/hooks.h` | Runtime 端 hook 存储 |
| `MUSA-Runtime/src/mupti/hooks.cpp` | Runtime 端 `EnableMUptiRuntime` |

---

## 9. 设计特点总结

1. **零开销默认状态**：`atomic<bool>::load(acquire)` + 分支，未启用时 hook 路径只有数个 CPU 周期
2. **解耦架构**：MUPTI 库通过函数指针表注册，musa 驱动不直接依赖 MUPTI
3. **Accessor 模式**：通过导出表暴露内部对象的访问接口，MUPTI 库不需要了解命令对象的内部结构
4. **三种语义模式**：Enter/Exit（生命周期）、Register/Mark/Assign（内核/关联）、Start/Stop（同步等待），覆盖所有 GPU 操作类型
5. **平台差异处理**：PH 平台从命令对象直接读时间戳（更精确），QY2 通过 MUPTI hook 记录
6. **多体系共用**：MUPTI、MUASAN、MUGDB 三套系统使用相同的 `hooks.h → G_xxx_DRIVER_HOOKS → ready + 函数指针表` 架构
