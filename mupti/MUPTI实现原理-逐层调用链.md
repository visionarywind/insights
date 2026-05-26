# MUPTI 实现原理：逐层调用链分析

> 以 `musaMemcpyAsync` 为例，追踪从用户启用 MUPTI 到最终拿到 activity 记录的**每一层函数调用**。
>
> 涉及三个 .so 文件：
> - `libmusart.so` — MUSA Runtime
> - `libmusa.so` — MUSA Driver  
> - `libmupti.so` — MUPTI Profiling Library

---

## 总览：完整调用链

```
用户代码
  │ ① muptiActivityEnable(MUPTI_ACTIVITY_KIND_MEMCPY)    ← 用户启用追踪
  │
  ▼
libmupti.so
  │ ② MUpti::init()                                      ← 首次调用时触发注入
  │ ③ inject_musa()                                      ← 核心：注入 hook 函数指针
  │     ├─ dlopen("libmusa.so.1")                        ← 加载 musa 驱动
  │     ├─ dlsym("muGetExportTable")                     ← 找到导出表入口
  │     ├─ muGetExportTable(&driver_table, Client::MUpti)← 获取驱动导出表
  │     ├─ driver_table->MUpti->Enable(hook_init_func)   ← 写入 hook 函数指针
  │     └─ import_musa_accessors(driver_table)           ← 导入 accessor 表
  │
  │ 用户代码继续执行...
  │ ④ musaMemcpyAsync(dst, src, size, HtoD, stream)      ← 用户调用 GPU API
  │
  ▼
libmusart.so → libmusa.so
  │ ⑤ MemcpyCommand 构造                                   ← 驱动内部创建命令对象
  │ ⑥ MUpti::EnterMemcpy(this)                            ← 驱动代码中的 hook 点
  │     → G_MUPTI_DRIVER_HOOKS.hooks.EnterMemcpy(command) ← 通过函数指针跳到 libmupti.so
  │
  ▼
libmupti.so
  │ ⑦ MUinteraction::EnterMemcpy(command)                ← MUPTI 的 hook 实现
  │     ├─ g_activity_buffer_manager.acquire_record()     ← 从 buffer 分配 activity 记录
  │     ├─ MemcpyCommand.GetCopyKind(command)             ← 通过 accessor 读取命令字段
  │     ├─ Device.GetId(device)                           ← 通过 accessor 读取设备 ID
  │     ├─ g_memop_map.emplace(corrId, record)            ← 暂存到 map
  │     └─ return new MUpti::Context{ record, corrId }    ← 返回 context
  │
  │ ... GPU 执行 memcpy ...
  │
  │ ⑧ MUpti::MarkCommandBeginEnd(ctx, command)            ← 驱动回调：GPU 时间戳
  │     → core.cpp:597 MarkCommandBeginEnd()
  │     ├─ CommandBase.GetBeginEndTimestamp(command)       ← 读取 GPU 起止时间戳
  │     ├─ HwStream::socTimestampToOsTimestamp(begin/end)  ← SoC→OS 时间转换
  │     └─ memcpyAct->start/end = ...                     ← 填入时间戳
  │
  │ ⑨ MUpti::ExitMemcpy(ctx)                              ← 驱动回调：操作完成
  │     → core.cpp:1384 ExitMemcpy()
  │     └─ delete ctx                                     ← 标记完成，从 g_memop_map 移除
  │
  │ ⑩ muptiActivityFlushAll(0)                            ← 用户请求刷新
  │     → g_activity_buffer_manager.flush_buffers()
  │       → collect_flushing_buffers()
  │         → try_mark_completed() → check_record_completed()
  │       → on_complete_(buffer, validSize)               ← 调用用户注册的回调
  │         → bufferCompleted() → muptiActivityGetNextRecord() → 用户拿到记录
```

---

## 阶段 1：初始化 — 注入 Hook 函数指针

### 步骤 1.1：用户调用 `muptiActivityEnable`

**文件**：`MUPTI/src/api/activity.cpp:90`

```cpp
MUptiResult muptiActivityEnable(MUpti_ActivityKind kind) {
    // kind = MUPTI_ACTIVITY_KIND_MEMCPY
    MUPTI_TRY(MUpti::enable(kind));         // → 步骤 1.2
    MUpti::g_activity_kind_enabled[kind] = true;  // 标记此类型已启用
    return MUPTI_SUCCESS;
}
```

### 步骤 1.2：`enable()` → 首次调用触发 `init()`

**文件**：`MUPTI/src/core/init.cpp:38`

```cpp
MUptiResult enable(MUpti_ActivityKind kind) {
    MUPTI_TRY(init());          // → 步骤 1.3，首次调用执行注入
    // 如果是 kernel/memcpy/memset，还需要打开 device monitor:
    g_device_monitor_manager.init();
    g_device_monitor_manager.open();
    return MUPTI_SUCCESS;
}
```

### 步骤 1.3：`init()` — 使用 CAS 保证只注入一次

**文件**：`MUPTI/src/core/init.cpp:16`

```cpp
MUptiResult init() {
    bool expected = false;
    // compare_exchange_strong: 只有第一次调用时 expected==false 才会成功
    if (g_hook_initialized.compare_exchange_strong(expected, true)) {
        // 首次调用：执行注入
        bool success = MUinteraction::inject_musa();  // → 步骤 1.4
        return success ? MUPTI_SUCCESS : MUPTI_ERROR_NOT_INITIALIZED;
    }
    // 后续调用：直接返回成功
    return MUPTI_SUCCESS;
}
```

### 步骤 1.4：`inject_musa()` — 核心注入流程

**文件**：`MUPTI/src/injection/injection.cpp:271`

```cpp
bool inject_musa() {
    // ===== 子步骤 A: 获取 Driver 导出表 =====
    get_export_table_from_musa_driver();    // → 步骤 1.5
    
    // ===== 子步骤 B: 注入 Driver hook 函数指针 =====
    if (g_musa_driver_table != nullptr) {
        // 调驱动的 Enable，传入我们的 init_mupti_driver_hooks 函数
        g_musa_driver_table->MUpti->Enable(init_mupti_driver_hooks);  // → 步骤 1.7
        // 导入驱动的 accessor 表（用于读取命令内部字段）
        import_musa_accessors(g_musa_driver_table);                   // → 步骤 1.9
    }
    
    // ===== 子步骤 C: 获取 Runtime 导出表（可选） =====
    get_export_table_from_musa_runtime();     // → 步骤 1.10
    if (g_musa_runtime_table != nullptr) {
        g_musa_runtime_table->MUpti->Enable(init_mupti_runtime_hooks);
    }
    
    return true;
}
```

### 步骤 1.5：`get_export_table_from_musa_driver()` — dlopen + dlsym

**文件**：`MUPTI/src/injection/injection.cpp:121`

```cpp
void get_export_table_from_musa_driver() {
    // ① dlopen libmusa.so
    g_musa_driver_lib_handle = DynamicLibrary::create("libmusa.so.1");
    //    └→ 等价于 dlopen("libmusa.so.1", RTLD_NOW)
    
    // ② dlsym 找到入口函数
    auto symbol = g_musa_driver_lib_handle->get_symbol<muGetExportTable_fn>("muGetExportTable");
    //    └→ 等价于 dlsym(handle, "muGetExportTable")
    muGetExportTable_fn& muGetExportTable = **symbol;
    
    // ③ 调 muGetExportTable，传入 Client::MUpti 标识符
    const enum Client uuid = Client::MUpti;
    muGetExportTable(&g_musa_driver_table, &uuid);
    //   ↑ 这调到了 musa 驱动内部（mu_entry.cpp），驱动返回 DriverExportTable 结构体
    
    // ④ 如果驱动支持 Tools callback，订阅它
    if (IsPfnValid(ToolsCallback.Subscribe)) {
        ToolsCallback.Subscribe(&g_driver_subscribe_handle, ProcessDriverCallback, nullptr);
        g_driver_support_callback = true;
    }
}
```

**返回的 `DriverExportTable` 结构体长什么样**（`mu_entry.cpp:1243`）：

```cpp
static MUpti::DriverExportTable muptiDriverTable = {
    &muptiDriverControllers,    // → Enable/Disable 函数指针
    &muptiThreadLocalInfoAccessors,
    &commandBaseAccessors,      // → CommandBase 的 GetStream/GetCorrelationId/...
    &dispatchCommandAccessors,  // → DispatchCommand 的 GetFunction/GetGridSize/...
    &memcpyCommandAccessors,    // → MemcpyCommand 的 GetCopyKind/GetSize/...
    &deviceAccessors,           // → Device 的 GetId
    &contextAccessors,          // → Context 的 GetId/GetDevice/...
    &streamAccessors,           // → Stream 的 GetId/GetContext
    &functionAccessors,
    // ... 共 19 个子表
};
```

### 步骤 1.6：驱动侧的 `muGetExportTable` 实现

**文件**：`musa/src/driver/mu_entry.cpp`

驱动根据 `Client` 枚举值返回不同的导出表。`Client::MUpti` 对应上面那个 19 个子表的结构体。

```cpp
// 简化逻辑
MUresult muGetExportTable(void const** table, MUuuid const* uuid) {
    if (*uuid == Client::MUpti) {
        *table = &muptiDriverTable;     // 返回上面那 19 个 accessor 子表
    } else if (*uuid == Client::Tools) {
        *table = &toolsTable;
    }
    return MUSA_SUCCESS;
}
```

### 步骤 1.7：`init_mupti_driver_hooks()` — 函数指针复制

**文件**：`MUPTI/src/injection/injection.cpp:237`

```cpp
void init_mupti_driver_hooks(MUpti::MUptiDriverHooks* entry) {
    // entry = &G_MUPTI_DRIVER_HOOKS.hooks  (驱动侧的全局 hook 表)
    uintptr_t* pDst = reinterpret_cast<uintptr_t*>(entry);
    uintptr_t* pSrc = reinterpret_cast<uintptr_t*>(&driver_hooks);
    //        ↑ driver_hooks 是 injection.cpp 里预定义的局部变量（第 28-89 行）
    fill_hooks(pDst, pSrc);    // → 步骤 1.8
}
```

**MUPTI 库侧的 `driver_hooks` 预定义**（`injection.cpp:28`）：

```cpp
// 这是在 libmupti.so 的 .data 段里的一个结构体，编译时就写死了
MUpti::MUptiDriverHooks driver_hooks = {
    &EnterMemcpy,   // 指向 core.cpp 第 941 行的 EnterMemcpy
    &ExitMemcpy,    // 指向 core.cpp 第 1384 行的 ExitMemcpy
    &EnterMemset,   // 指向 core.cpp 第 1115 行的 EnterMemset
    &ExitMemset,    // 指向 core.cpp 第 1388 行的 ExitMemset
    &RegisterKernel,
    // ... 共 40+ 个
};
```

### 步骤 1.8：`fill_hooks()` — 逐字段复制

**文件**：`MUPTI/src/injection/injection.cpp:216`

```cpp
void fill_hooks(uintptr_t* pDst, uintptr_t* pSrc) {
    // pDst 指向驱动侧 G_MUPTI_DRIVER_HOOKS.hooks
    // pSrc 指向 MUPTI 侧 driver_hooks
    
    // 遍历 MUptiDriverHooks 结构体的每一个字段（按内存布局，逐 8 字节）
    while (pDst != nullptr && (*pDst == AccessorHint || *pDst == 0) && pSrc != nullptr && *pSrc != 0) {
        *pDst = *pSrc;   // 写入：驱动侧函数指针 = MUPTI 侧函数指针
        pDst++;           // 移动到下一个字段
        pSrc++;
    }
}
```

**执行前后的状态变化**：

```
执行前（驱动侧）:
  G_MUPTI_DRIVER_HOOKS.hooks.EnterMemcpy = AccessorHint (0xDEAD...)
  G_MUPTI_DRIVER_HOOKS.hooks.ExitMemcpy  = AccessorHint
  G_MUPTI_DRIVER_HOOKS.hooks.RegisterKernel = AccessorHint
  ...

执行后（驱动侧）:
  G_MUPTI_DRIVER_HOOKS.hooks.EnterMemcpy = &MUinteraction::EnterMemcpy (core.cpp:941)
  G_MUPTI_DRIVER_HOOKS.hooks.ExitMemcpy  = &MUinteraction::ExitMemcpy  (core.cpp:1384)
  G_MUPTI_DRIVER_HOOKS.hooks.RegisterKernel = &MUinteraction::RegisterKernel
  ...
```

### 步骤 1.9：`import_musa_accessors()` — 导入 Accessor 表

**文件**：`MUPTI/src/injection/injection.cpp:249`

```cpp
void import_musa_accessors(MUpti::DriverExportTable const* musa) {
    // musa 是驱动返回的导出表（步骤 1.5-1.6）
    
    bool stop = false;
    // 把驱动侧的 accessor 函数指针复制到 MUPTI 侧的局部变量
    fill_accessors(&TidInfo,          musa->TidInfo,          stop);
    fill_accessors(&CommandBase,      musa->CommandBase,      stop);
    fill_accessors(&DispatchCommand,  musa->DispatchCommand,  stop);
    fill_accessors(&MemcpyCommand,    musa->MemcpyCommand,    stop);
    fill_accessors(&Device,           musa->Device,           stop);
    fill_accessors(&Context,          musa->Context,          stop);
    fill_accessors(&Stream,           musa->Stream,           stop);
    fill_accessors(&Function,         musa->Function,         stop);
    // ... 共 18 组
}
```

**执行后 MUPTI 侧的 accessor 被填充**：

```
执行前:
  MemcpyCommand.GetCopyKind = AccessorHint (占位值)

执行后:
  MemcpyCommand.GetCopyKind = 驱动 mu_entry.cpp 里的 MemcpyCommandGetCopyKind 函数指针
  MemcpyCommand.GetSize     = MemcpyCommandGetSize
  Device.GetId              = DeviceGetId
  Context.GetDevice         = ContextGetDevice
  ...
```

**`fill_accessors` 函数**（同文件 225 行）：

```cpp
void fill_accessors(uintptr_t* pDst, uintptr_t* pSrc, bool& stop) {
    // pDst 指向 MUPTI 侧的 accessor 局部变量
    // pSrc 指向驱动返回的 accessor 表（mu_entry.cpp 里定义的）
    
    while (!stop && *pDst == AccessorHint && *pSrc != 0) {
        *pDst = *pSrc;   // 写入：MUPTI 侧 accessor = 驱动侧 accessor 函数指针
        pDst++;
        pSrc++;
    }
}
```

**为什么需要 accessor 表？**

MUPTI 库编译时**不链接** libmusa.so，所以它**不知道 `Musa::MemcpyCommand` 的内部结构**。它通过 accessor 函数间接访问：

```cpp
// MUPTI 库不知道 MemcpyCommand 类长什么样，但可以通过函数指针读取其字段：
int copyKind = MemcpyCommand.GetCopyKind(command);  
//            ↑ 这实际调的是驱动的 MemcpyCommandGetCopyKind(command)
//              驱动的实现: return command->m_CopyKind;
```

### 步骤 1.10：Runtime 端的注入（同理）

**文件**：`MUPTI/src/injection/injection.cpp:185`

```cpp
void get_export_table_from_musa_runtime() {
    g_musa_runtime_lib_handle = DynamicLibrary::create("libmusart.so");
    auto symbol = g_musa_runtime_lib_handle->get_symbol("musaGetExportTable");
    musaGetExportTable_fn& musaGetExportTable = **symbol;
    musaGetExportTable(&g_musa_runtime_table, &uuid);
}
```

Runtime 端注入更简单，只有 3 个 hook：`EnterRuntimeApi`、`ExitRuntimeApi`、`SetMemset3DCounter`。

### 步骤 1.11：驱动的 `EnableMUptiDriver()` 被调用

**文件**：`musa/src/driver/mupti/hooks.cpp:9`（这是 musa 驱动里的代码）

```cpp
void EnableMUptiDriver(MUpti::InitializeMUptiDriverHooks_fn initer) {
    // initer = init_mupti_driver_hooks  (MUPTI 库的函数指针)
    
    // 步骤 1: 调 MUPTI 库的 init 函数，将函数指针写入 G_MUPTI_DRIVER_HOOKS.hooks
    initer(&G_MUPTI_DRIVER_HOOKS.hooks);
    //      ↑ 这个 initer 就是步骤 1.7 的 init_mupti_driver_hooks
    //        它做了 fill_hooks(&G_MUPTI_DRIVER_HOOKS.hooks, &driver_hooks)
    
    // 步骤 2: 打开全局开关
    G_MUPTI_DRIVER_HOOKS.ready.store(true, std::memory_order_release);
    //      ↑ 之后所有 CheckMUptiEnabled() 调用都会返回 true
}
```

---

## 阶段 2：Hook 被触发 — 以 `musaMemcpyAsync` 为例

### 步骤 2.1：用户调用 musaMemcpyAsync

```c
// 用户代码
musaMemcpyAsync(dst, src, 1024, musaMemcpyHostToDevice, stream);
```

### 步骤 2.2：Runtime 层 → Driver 层

**文件**：`MUSA-Runtime/src/musa_wrappers_generated.cpp`

```cpp
musaError_t musaMemcpyAsync(void* dst, const void* src, size_t count,
                             musaMemcpyKind kind, musaStream_t stream) {
    // ApiInvocationGuard 构造 → EnterRuntimeApi(CBID=41) → MUPTI 记录 Runtime API 进入
    
    // 通过导出表调 Driver API:
    musaapiMemcpyAsync_v3020(dst, src, count, kind, stream);
    
    // ApiInvocationGuard 析构 → ExitRuntimeApi(CBID=41) → MUPTI 记录 API 退出
}
```

### 步骤 2.3：Driver 层 → 创建 MemcpyCommand

**文件**：`musa/src/driver/mu_memory.cpp` 和 `musa/src/musa/core/command/memcpyCommand.cpp`

```cpp
// muMemcpyAsync 实现
MUresult muMemcpyAsync(...) {
    // 创建命令对象
    auto cmd = new Musa::MemcpyCommand(context, stream, dst, src, size, kind);
    //   ↑ 构造函数里触发了 hook（步骤 2.4）
    
    // 排队到 stream
    stream->CmdMemcpy(cmd);
}
```

### 步骤 2.4：MemcpyCommand 构造函数 → 触发 `EnterMemcpy` hook

**文件**：`musa/src/musa/core/command/memcpyCommand.cpp:34`

```cpp
MemcpyCommand::MemcpyCommand(...) {
    // ... 初始化 ...
    
    m_ptiCtx = MUpti::EnterMemcpy(this);   // ← 这是 hook 点
    //          ↑ 调用 tracepoints.h 里的内联包装函数（步骤 2.5）
}
```

### 步骤 2.5：`MUpti::EnterMemcpy()` 内联包装函数

**文件**：`musa/src/driver/mupti/tracepoints.h:13`（这是 musa 驱动里的代码）

```cpp
inline MUpti::Context* EnterMemcpy(Musa::MemcpyCommand* command) {
    // ① 快速路径检查：读 atomic<bool>
    if (CheckMUptiEnabled()) {
        //  ↑ 展开为: G_MUPTI_DRIVER_HOOKS.ready.load(std::memory_order_acquire)
        //  如果 MUPTI 未启用，这里返回 false，整个函数直接 return nullptr
        //  开销：一次原子读 + 一个分支（~3 CPU 周期）
        
        // ② 如果启用，通过函数指针调用 MUPTI 库的实现
        return G_MUPTI_DRIVER_HOOKS.hooks.EnterMemcpy(command);
        //     ↑ 此时这个函数指针已被注入为 &MUinteraction::EnterMemcpy
        //       （步骤 1.8 写入的）
    }
    return nullptr;
}
```

**`CheckMUptiEnabled()` 展开**（同文件第 9 行）：

```cpp
inline bool CheckMUptiEnabled() {
    return G_MUPTI_DRIVER_HOOKS.ready.load(std::memory_order_acquire);
    //     ↑ G_MUPTI_DRIVER_HOOKS 是 hooks.cpp 里的全局变量
    //       .ready 在步骤 1.11 被设置为 true
}
```

### 步骤 2.6：跳到 MUPTI 库的 `EnterMemcpy` 实现

通过函数指针调用，执行流从 `libmusa.so` 跳转到 `libmupti.so`：

**文件**：`MUPTI/src/core/core.cpp:941`

```cpp
MUpti::Context* EnterMemcpy(Musa::MemcpyCommand* command) {
    // ===== 检查 1：用户是否启用了 MEMCPY activity？ =====
    if (!MUpti::g_activity_kind_enabled[MUPTI_ACTIVITY_KIND_MEMCPY]) {
        return nullptr;  // 未启用 → 返回 null → 驱动侧的 m_ptiCtx = nullptr
    }
    
    // ===== 检查 2：从 buffer 分配一条 activity 记录 =====
    auto memcpyAct = MUpti::g_activity_buffer_manager.acquire_record<
        MUPTI_ACTIVITY_KIND_MEMCPY>();    // → 步骤 2.7
    if (!memcpyAct) {
        return nullptr;  // buffer 满了 → 返回 null
    }
    
    // ===== 步骤 A：通过 accessor 表读取 command 的内部字段 =====
    memcpyAct->copyKind = MemcpyCommand.GetCopyKind(command);
    //  ↑ MemcpyCommand 是 MUPTI 侧的 accessor 局部变量（步骤 1.9 时填充的）
    //    实际调用的是驱动 mu_entry.cpp 里的 MemcpyCommandGetCopyKind(command)
    //    驱动的实现：return command->m_CopyKind;
    
    memcpyAct->srcKind  = MemcpyCommand.GetSrcMemKind(command);
    memcpyAct->dstKind  = MemcpyCommand.GetDstMemKind(command);
    memcpyAct->bytes    = MemcpyCommand.GetSize(command);
    memcpyAct->start    = 0;  // GPU 时间戳稍后由 MarkCommandBeginEnd 填入
    memcpyAct->end      = 0;
    
    // ===== 步骤 B：通过 accessor 链获取 context/device/stream ID =====
    auto base = MemcpyCommand.GetBase(command);     // 获取命令的基类指针
    MUstream stream = CommandBase.GetStream(base);  // 获取 stream
    MUcontext ctx    = Stream.GetContext(stream);    // stream → context
    MUdevice device  = Context.GetDevice(ctx);       // context → device
    
    memcpyAct->deviceId  = Device.GetId(device);     // 0
    memcpyAct->contextId = Context.GetId(ctx);       // 1
    memcpyAct->streamId  = Stream.GetId(stream);     // 2
    
    // ===== 步骤 C：获取 correlation ID =====
    uint32_t correlationId = CommandBase.GetCorrelationId(base);
    memcpyAct->correlationId = correlationId;        // 42
    memcpyAct->runtimeCorrelationId = correlationId;
    
    // ===== 步骤 D：检查是否从 CUDA Graph 中来 =====
    bool fromGraph = CommandBase.CheckFromGraph(base);
    if (fromGraph) {
        memcpyAct->graphNodeId = CommandBase.GetGraphNodeId(base);
        memcpyAct->graphId     = CommandBase.GetGraphId(base);
    } else {
        memcpyAct->graphNodeId = 0;
        memcpyAct->graphId     = 0;
    }
    
    // ===== 步骤 E：创建 MUPTI context，关联 activity 记录 =====
    MUpti::Context* ptiCtx = new MUpti::Context;
    ptiCtx->activity = std::move(memcpyAct);   // 移动 activity 记录指针到 context
    ptiCtx->correlationId = correlationId;
    
    // ===== 步骤 F：将记录存入 g_memop_map（后续用于 MarkCommandBeginEnd 匹配） =====
    {
        auto guard = MUpti::g_memop_map.lock();
        guard->emplace(correlationId, memcpyAct);
        //  ↑ 这个 map 的作用：
        //    - MarkCommandBeginEnd 被调用时，通过 correlationId 找到记录，填入 GPU 时间戳
        //    - ExitMemcpy 被调用时，通过 correlationId 删除记录
        //    - flush_buffers 时，检查 map 里是否还有该记录（有 = 未完成）
    }
    
    return ptiCtx;    // 返回给驱动，驱动存到 m_ptiCtx 成员变量
}
```

### 步骤 2.7：`acquire_record()` — 从 buffer 分配内存

**文件**：`MUPTI/src/core/buffer.h:203`

```cpp
template <MUpti_ActivityKind kind>
ActivityRecordPtr<ActivityRecordType_t<kind>> acquire_record() {
    size_t size = activity_record_size(kind);
    //  ↑ 展开为 sizeof(MUpti_ActivityMemcpy4)
    
    auto [ptr, buffer] = acquire_record_raw(size);  // → 步骤 2.8
    
    if (ptr != nullptr) {
        reinterpret_cast<MUpti_Activity*>(ptr)->kind = kind;
        //  ↑ 在分配的内存开头写入 kind 字段
    }
    
    // 返回 RAII 智能指针，析构时自动 release_record()
    return ActivityRecordPtr<ActivityRecordType_t<kind>>{ ptr, buffer };
}
```

### 步骤 2.8：`acquire_record_raw()` — buffer 管理

**文件**：`MUPTI/src/core/buffer.cpp:649`

```cpp
std::tuple<uint8_t*, ActivityBuffer const*> acquire_record_raw(size_t size) {
    while (true) {
        size_t old_seq;
        {
            auto readable_buffer = current_buffer_.rlock();
            old_seq = readable_buffer->seq;
            
            if (readable_buffer->buffer != nullptr) {
                // 尝试在当前 buffer 中分配
                uint8_t* ptr = readable_buffer->buffer->acquire_record(size);
                //  ↑ 使用 CAS 原子操作分配空间（buffer.cpp:330）
                if (ptr != nullptr) {
                    return { ptr, readable_buffer->buffer };
                }
            }
        }
        
        // 当前 buffer 不可用或满了，向用户请求新 buffer
        if (!request_buffer(old_seq, size)) {
            // 用户没有提供 buffer → 丢弃记录
            num_dropped_records_.fetch_add(1);
            return { nullptr, nullptr };
        }
        // request_buffer 成功后，重试循环
    }
}
```

---

## 阶段 3：GPU 时间戳填充 — `MarkCommandBeginEnd`

### 步骤 3.1：驱动回调 → 触发时间戳 hook

**文件**：`musa/src/musa/core/command/memcpyCommand.cpp:93`

```cpp
// memcpy 的 GPU 执行完成回调
void MemcpyCommand::OnComplete() {
    MUpti::MarkCommandBeginEnd(m_ptiCtx, this);
    //  ↑ m_ptiCtx 是步骤 2.6 的返回值（EnterMemcpy 返回的 context）
    
    MUpti::ExitMemcpy(m_ptiCtx);
}
```

### 步骤 3.2：`MarkCommandBeginEnd` 包装函数

**文件**：`musa/src/driver/mupti/tracepoints.h:281`

```cpp
inline void MarkCommandBeginEnd(MUpti::Context* context, Musa::Command* command) {
    if (CheckMUptiEnabled()) {
        auto fn = G_MUPTI_DRIVER_HOOKS.hooks.MarkCommandBeginEnd;
        if (fn != reinterpret_cast<void*>(AccessorHint)) {
            fn(context, command);    // 跳到 MUPTI 库 core.cpp:597
        }
    }
}
```

### 步骤 3.3：MUPTI 库的 `MarkCommandBeginEnd` 实现

**文件**：`MUPTI/src/core/core.cpp:597`

```cpp
void MarkCommandBeginEnd(MUpti::Context* ptiCtx, Musa::Command* command) {
    if (ptiCtx != nullptr) {
        // ===== 步骤 A：通过 accessor 读取命令类型和 GPU 时间戳 =====
        CommandType type = CommandBase.GetType(command);
        uint64_t begin = 0, end = 0;
        CommandBase.GetBeginEndTimestamp(command, &begin, &end);
        //  ↑ 驱动内部已从 GPU 硬件计数器采集到时间戳，通过 accessor 暴露
        
        // ===== 步骤 B：根据命令类型分发 =====
        if (type == CommandType::Memcpy || type == CommandType::AsyncMemcpy) {
            // 从 g_memop_map 中找到对应的 activity 记录
            MUpti::MemcpyActivity* memcpy = nullptr;
            get_memop_ownership(&memcpy, ptiCtx->correlationId);
            //  ↑ 在 g_memop_map 中查找 correlationId → 取出记录 → 删除 map 条目
            
            if (memcpy != nullptr) {
                // ===== 步骤 C：SoC 时间戳 → OS 时间戳转换 =====
                perf::HwStream* cur_stream = g_activity_buffer_manager.hw_streams_[memcpy->deviceId];
                uint64_t startTime = cur_stream->socTimestampToOsTimestamp(begin);
                uint64_t endTime   = cur_stream->socTimestampToOsTimestamp(end);
                
                // ===== 步骤 D：填入 activity 记录 =====
                if (memcpy->start == 0 || memcpy->start > startTime) {
                    memcpy->start = startTime;
                }
                if (memcpy->end == 0 || memcpy->end < endTime) {
                    memcpy->end = endTime;
                }
                // 至此，activity 记录的 start/end 字段从 0 → 实际的 GPU 执行时间
            }
        }
        // 其他命令类型（Dispatch/Memset/Barrier/MemoryAtomic）同理
        
        delete ptiCtx;   // 释放 context
    }
}
```

---

## 阶段 4：操作完成 — `ExitMemcpy`

### 步骤 4.1：`ExitMemcpy` 包装函数

**文件**：`musa/src/driver/mupti/tracepoints.h:38`

```cpp
inline void ExitMemcpy(MUpti::Context* context) {
    if (context != nullptr) {
        G_MUPTI_DRIVER_HOOKS.hooks.ExitMemcpy(context);
    }
}
```

### 步骤 4.2：MUPTI 库的 `ExitMemcpy` 实现

**文件**：`MUPTI/src/core/core.cpp:1384`

```cpp
void ExitMemcpy(MUpti::Context* ptiCtx) {
    delete ptiCtx;
    //  ↑ 析构时：
    //    1. activity 记录指针（ActivityRecordPtr）析构
    //       → release_record() → buffer 引用计数 -1
    //    2. 如果之前在 MarkCommandBeginEnd 中已经 get_memop_ownership 取出了记录
    //       g_memop_map 中已经不再有该 correlationId 的条目
    //    3. flush_buffers 时检查 g_memop_map 不再有该条目 → 认为记录已完成 → 可以 flush
}
```

---

## 阶段 5：用户获取结果 — Flush

### 步骤 5.1：用户调用 `muptiActivityFlushAll`

**文件**：`MUPTI/src/api/activity.cpp:187`

```cpp
MUptiResult muptiActivityFlushAll(uint32_t flag) {
    bool force = (flag & MUPTI_ACTIVITY_FLAG_FLUSH_FORCED) != 0;
    
    if (!g_activity_buffer_manager.callback_registered()) {
        return MUPTI_ERROR_INVALID_OPERATION;  // 用户没注册 callback
    }
    
    g_activity_buffer_manager.flush_buffers(force);  // → 步骤 5.2
    return MUPTI_SUCCESS;
}
```

### 步骤 5.2：`flush_buffers()` — 收集并输出完成的 buffer

**文件**：`MUPTI/src/core/buffer.cpp:545`

```cpp
bool ActivityBufferManager::flush_buffers(bool force) {
    // ===== 步骤 A: force 模式下等待所有 kernel/memop 完成 =====
    if (force) {
        // 等待 g_kernel_map + g_memop_map 全部清空（最多 1000ms）
        for (int i = 0; i < 100; i++) {
            if (g_kernel_map.size() + g_memop_map.size() == 0) break;
            usleep(10000);  // 10ms
        }
    }
    
    // ===== 步骤 B: 收集可 flush 的 buffer =====
    BufferList flushing_buffers = collect_flushing_buffers(force);
    //  ↑ 对每个 buffer，调用 try_mark_completed()
    //    try_mark_completed() 遍历 buffer 里的每条记录，调用 check_record_completed()
    //    check_record_completed() 检查：
    //      MEMCPY: g_memop_map 里是否还有该 correlationId → 没有 = 已完成
    //      KERNEL: g_kernel_map 里是否还有该 correlationId → 没有 = 已完成
    //      DRIVER/RUNTIME: 直接已完成（API 记录不需要等 GPU）
    
    // ===== 步骤 C: 逐个 buffer 调用用户回调 =====
    for (auto& buffer : flushing_buffers) {
        size_t valid_size = buffer.byte_used_.load() & ~REFUSING;
        on_complete_(nullptr, 0, buffer.inner_, buffer.byte_cap_, valid_size);
        //  ↑ 这就是用户注册的 bufferCompleted 函数！
    }
    
    return true;
}
```

### 步骤 5.3：用户的 `bufferCompleted` 回调

```cpp
// 用户代码（demo.cu）
void bufferCompleted(MUcontext ctx, uint32_t streamId, 
                     uint8_t *buffer, size_t size, size_t validSize) {
    MUpti_Activity* record = nullptr;
    do {
        muptiActivityGetNextRecord(buffer, validSize, &record);
        //  ↑ 遍历 buffer 里的每条 activity 记录
        
        if (record->kind == MUPTI_ACTIVITY_KIND_MEMCPY) {
            auto* memcpy = (MUpti_ActivityMemcpy4*)record;
            printf("MEMCPY HtoD [ %llu - %llu ] size %llu, corr %u\n",
                   memcpy->start, memcpy->end, memcpy->bytes, memcpy->correlationId);
            // 输出: MEMCPY HtoD [ 12345678 - 87654321 ] size 1024, corr 42
        }
    } while (status == MUPTI_SUCCESS);
}
```

---

## 核心原理总结

```
┌─────────────────────────────────────────────────────────────────┐
│                    关键的三个技术点                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ① 函数指针注入                                                  │
│     libmupti.so 通过 dlopen + dlsym 找到 libmusa.so 里的         │
│     G_MUPTI_DRIVER_HOOKS 全局变量，把自己的 40+ 个函数指针        │
│     写入其中。之后驱动调 hook → 执行的是 MUPTI 的代码。            │
│                                                                 │
│  ② Accessor 解耦                                                │
│     MUPTI 库不链接 libmusa.so，不知道 Command 类的内部结构。       │
│     它通过驱动的导出表获取一系列 accessor 函数指针，              │
│     间接读取命令的字段。驱动内部结构变化不影响 MUPTI 库。           │
│                                                                 │
│  ③ 零开销快速路径                                                │
│     未启用时，每个 hook 点只需：                                  │
│       atomic<bool>::load() + if(false) → 3 个 CPU 周期            │
│     全部 hook 包装函数都是 inline 的。                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**四个关键内存位置**：

| 位置 | 所属 .so | 内容 | 读写者 |
|------|----------|------|--------|
| `G_MUPTI_DRIVER_HOOKS` | `libmusa.so` | `atomic<bool> ready` + 40+ 函数指针 | MUPTI 库**写** / 驱动**读** |
| `driver_hooks` | `libmupti.so` | 编译时写死的 40+ 函数指针 | MUPTI 库**读**（注入时） |
| `g_musa_driver_table` | `libmupti.so` | 19 个 accessor 子表指针 | 驱动**写**（muGetExportTable）/ MUPTI 库**读** |
| accessor 局部变量 | `libmupti.so` | 从驱动复制来的 accessor 函数指针 | MUPTI 库**读**（在 EnterMemcpy 等函数里调用） |
