# MUPTI Tools Callback 机制详解

> Tools Callback 是 MUPTI 的**API 级 tracing 机制**——在每个 Driver API / Runtime API 的 Enter 和 Exit 处触发回调，产出 `DRIVER API ...` 和 `RUNTIME API ...` 这样的 activity 记录。
>
> 它和前面分析的语义 hook（EnterMemcpy/RegisterKernel 等）是**两套独立的机制**。

---

## 1. 两种 Tracing 机制对比

| 特性 | 语义 Hook（tracepoints.h） | Tools Callback |
|------|---------------------------|----------------|
| **触发位置** | 命令构造、执行、同步等关键点 | 每个 API 函数的 Enter/Exit |
| **覆盖范围** | ~40 个语义事件 | 822 Driver API + 531 Runtime API = **1353 个 API** |
| **注入方式** | `EnableMUptiDriver` → 写 `G_MUPTI_DRIVER_HOOKS` 函数指针表 | `ToolsCallback.Subscribe` → 写驱动的 `g_subscriber` 单槽 |
| **订阅者数量** | 无限制（函数指针表） | **只有 1 个**（全局单槽 `g_subscriber`） |
| **输出记录类型** | MEMCPY, KERNEL, MEMSET, SYNC 等 | DRIVER, RUNTIME |
| **典型输出** | `MEMCPY HtoD [12345-67890] size 1024` | `DRIVER API muMemcpyAsync [100-500] cbid=60` |

---

## 2. 订阅机制

### 2.1 驱动侧：单槽订阅者

**文件**：`musa/src/driver/callback.cpp`

驱动的 Tools Callback 是**单槽**设计——同时只有一个工具可以订阅：

```cpp
// 驱动的全局状态
static volatile MUtoolsCbSubscriber g_subscriber{};   // 只有一个订阅者槽位
std::atomic<uint32_t> g_hasSubscriber{0};              // 0=空闲, 1=已被占用

// 每个 domain 的 CBID 启用位图
uint32_t driverApiCallbackEnabled[822] = {};    // 822 个 Driver API
uint32_t runtimeApiCallbackEnabled[531] = {};   // 531 个 Runtime API
```

### 2.2 订阅流程

```cpp
// 驱动侧 callback.cpp:109
MUresult toolsSubscribe(handle, callback, userdata) {
    // ① CAS 检查槽位是否空闲
    oldValue = g_hasSubscriber.exchange(1);  // 0→1
    if (oldValue != 0) return ERROR;          // 已被占用

    // ② 用 seq 计数器保护写入
    pSubscriber->seq.fetch_add(1);            // 奇数 → 正在写
    pSubscriber->userdata = userdata;
    pSubscriber->seq.fetch_add(1);            // 偶数 → 写完成
    pSubscriber->callback.store(callback);

    *handle = MU_TOOLS_SUBSCRIBER_HANDLE;     // 固定 handle 值
}
```

### 2.3 MUPTI 库的订阅时机

**文件**：`MUPTI/src/injection/injection.cpp:160-182`

```cpp
// 在 inject_musa() 中，获取驱动导出表后
if (IsPfnValid(ToolsCallback.Subscribe)) {
    // 注册 ProcessDriverCallback 为回调函数
    ToolsCallback.Subscribe(&g_driver_subscribe_handle, ProcessDriverCallback, nullptr);
    g_driver_support_callback = true;
}
```

---

## 3. 驱动如何触发回调

### 3.1 生成的 Wrapper 代码

**文件**：`musa/src/driver/mu_wrappers_generated.cpp`

每个 API 的包装函数在 Enter 和 Exit 处各触发一次回调。以 `muCtxSynchronize` 为例：

```cpp
// mu_wrappers_generated.cpp（自动生成，每个 API 都有类似模式）

MUresult muCtxSynchronize(MUcontext ctx) {
    MUtoolsTraceApiMusa inParams = {};      // 参数结构体
    MUresult status;

    // ===== ENTER: 检查是否启用 → 触发回调 =====
    if (toolsCallbackEnabled(handle, MU_TOOLS_CB_DOMAIN_DRIVER_API,
                             MUPTI_DRIVER_TRACE_CBID_muCtxSynchronize)) {
        inParams.functionName   = "muCtxSynchronize";
        inParams.functionId     = MUPTI_DRIVER_TRACE_CBID_muCtxSynchronize;
        inParams.apiEnterOrExit = MU_TOOLS_API_ENTER;    // ← 标记为 ENTER
        inParams.pCorrelationId = &correlationId;
        inParams.params         = &ctx;                  // API 参数
        inParams.pStatus        = nullptr;               // 此时还不知道返回值

        toolsIssueCallback(handle, MU_TOOLS_CB_DOMAIN_DRIVER_API,
                          MUPTI_DRIVER_TRACE_CBID_muCtxSynchronize, &inParams);
    }

    // ===== 实际执行 API =====
    status = real_muCtxSynchronize(ctx);

    // ===== EXIT: 触发回调（带返回值） =====
    if (toolsCallbackEnabled(handle, MU_TOOLS_CB_DOMAIN_DRIVER_API,
                             MUPTI_DRIVER_TRACE_CBID_muCtxSynchronize)) {
        inParams.apiEnterOrExit = MU_TOOLS_API_EXIT;     // ← 标记为 EXIT
        inParams.pStatus        = &status;               // 带上返回值

        toolsIssueCallback(handle, MU_TOOLS_CB_DOMAIN_DRIVER_API,
                          MUPTI_DRIVER_TRACE_CBID_muCtxSynchronize, &inParams);
    }

    return status;
}
```

### 3.2 `toolsCallbackEnabled` — 快速路径检查

```cpp
// callback.cpp:57
uint32_t toolsCallbackEnabled(handle, domain, cbid) {
    return g_callbackEnabled[domain][cbid];
    //  ↑ 直接数组访问，O(1)
    //  返回 0 → 整个 if 块被跳过，零开销
}
```

### 3.3 `toolsIssueCallback` → `IssueCallback` → `ProcessDriverCallback`

```
toolsIssueCallback()
  → IssueCallback(domain, cbid, &inParams)      // callback.cpp:40
    → 用 seq 计数器确保 callback 指针读取安全
    → callback(userdata, domain, cbid, &inParams) // 实际调用
      → MUPTI::ProcessDriverCallback(...)         // process_callback.cpp:30
```

---

## 4. ProcessDriverCallback 的处理逻辑

**文件**：`MUPTI/src/core/process_callback.cpp:30`

`ProcessDriverCallback` 同时做两件事：**产出 activity 记录** 和 **分派用户 callback**。

### 4.1 Enter 时的处理

```cpp
void ProcessDriverCallback(userdata, domain, cbid, pParams) {
    if (domain == MU_TOOLS_CB_DOMAIN_RUNTIME_API) {
        auto* data = (MUtoolsTraceRuntimeApiMusa*)pParams;
        // data->functionName = "musaMemcpyAsync"
        // data->functionId   = 41 (CBID)
        // data->apiEnterOrExit = MU_TOOLS_API_ENTER

        // ===== 子功能 A: 分派用户注册的 callback =====
        if (g_subscriber != nullptr && g_subscriber->domain_callback_enabled[RUNTIME][cbid]) {
            MUpti_CallbackData cbData;
            cbData.callbackSite     = MUPTI_API_ENTER;
            cbData.functionName     = data->functionName;   // "musaMemcpyAsync"
            cbData.functionParams   = data->params;         // 传给 API 的参数
            cbData.correlationId    = *(data->pCorrelationId);

            // 调用用户的 callback 函数
            g_subscriber->callback(userdata, MUPTI_CB_DOMAIN_RUNTIME_API, cbid, &cbData);
        }

        // ===== 子功能 B: 产出 activity 记录 =====
        bool recordActivity = (g_activity_kind_enabled[RUNTIME] && t_runtime_api_depth == 0);
        t_runtime_api_depth++;

        if (recordActivity) {
            // 记录 API 开始时间戳（进栈）
            t_api_start_timestamp_stack.push(TimeNanos());
        }
    }
}
```

### 4.2 Exit 时的处理

```cpp
    } else if (data->apiEnterOrExit == MU_TOOLS_API_EXIT) {
        t_runtime_api_depth--;

        // ===== 子功能 A: 产出 activity 记录 =====
        bool recordActivity = (g_activity_kind_enabled[RUNTIME] && t_runtime_api_depth == 0);
        if (recordActivity && !t_api_start_timestamp_stack.empty()) {
            // 从 buffer 分配一条 RUNTIME activity 记录
            auto apiAct = g_activity_buffer_manager.acquire_record<RUNTIME>();

            apiAct->end   = TimeNanos();                    // API 结束时间
            apiAct->start = t_api_start_timestamp_stack.top(); // API 开始时间（出栈）
            t_api_start_timestamp_stack.pop();

            apiAct->cbid          = data->functionId;       // CBID=41
            apiAct->correlationId = *(data->pCorrelationId);
            apiAct->returnValue   = *(data->pStatus);       // musaSuccess
            apiAct->processId     = TidInfo.GetProcessId(tls_info);
            apiAct->threadId      = TidInfo.GetThreadId(tls_info);
        }

        // ===== 子功能 B: 分派用户注册的 callback =====
        if (g_subscriber != nullptr && g_subscriber->domain_callback_enabled[RUNTIME][cbid]) {
            // 同上，但 callbackSite = MUPTI_API_EXIT
            g_subscriber->callback(userdata, MUPTI_CB_DOMAIN_RUNTIME_API, cbid, &cbData);
        }
    }
```

### 4.3 嵌套 API 的深度处理

`t_runtime_api_depth` 和 `t_driver_api_depth` 是 thread_local 计数器：

```
用户调用 musaMemcpyAsync:      depth 0→1  (recordActivity=true, 记录 start)
  → Runtime 包装调 Driver API:  driver depth 0→1
    → Driver 包装调 muMemcpyAsync:  (另一个 driver depth)
    ...
  ← Driver 包装返回:              driver depth 1→0 (recordActivity=true, 写 activity 记录)
← musaMemcpyAsync 返回:         depth 1→0  (recordActivity=true, 写 activity 记录)
```

**关键逻辑**：只有 `depth == 0` 时才产生 activity 记录。这意味着嵌套的中间层 API 调用不会被记录——只记录最外层用户调用的起止时间。

---

## 5. 启用/禁用流程

### 5.1 用户启用 DRIVER activity

```cpp
// activity.cpp:90
muptiActivityEnable(MUPTI_ACTIVITY_KIND_DRIVER) {
    MUpti::enable(kind);        // 首次调用触发 init → inject_musa → Subscribe
    g_activity_kind_enabled[DRIVER] = true;

    // 如果驱动支持 Tools callback：
    if (g_driver_support_callback) {
        // 启用全部 822 个 Driver API 的 callback
        ToolsCallback.EnableAllCallbacksInDomain(1, g_driver_subscribe_handle,
                                                  MU_TOOLS_CB_DOMAIN_DRIVER_API);
    }
}
```

**`EnableAllCallbacksInDomain` 在驱动侧的执行**：

```cpp
// callback.cpp:204
MUresult toolsEnableAllCallbacksInDomain(enable, handle, domain) {
    for (uint32_t cbid = 0; cbid < g_domainSize[domain]; cbid++) {
        EnableCallbackIndex(domain, cbid, enable);
        // ↑ g_callbackEnabled[domain][cbid] = 1;
        //   之后 toolsCallbackEnabled() 返回 1 → 触发回调
    }
}
```

### 5.2 用户禁用

```cpp
muptiActivityDisable(MUPTI_ACTIVITY_KIND_DRIVER) {
    g_activity_kind_enabled[DRIVER] = false;
    MUpti::disable(kind);
    // 注意：不调用 EnableAllCallbacksInDomain(0) 来关闭！
    // 而是靠 g_activity_kind_enabled 在 ProcessDriverCallback 里判断
}
```

实际上，驱动侧的 `g_callbackEnabled` 位图在 enable 时全部设为 1 后就保持 1。关闭 tracing 依靠 MUPTI 库侧 `g_activity_kind_enabled` 的判断——callback 仍然被调用，但 `ProcessDriverCallback` 内部跳过 activity 记录创建。只有 `muptiFinalize` → `Unsubscribe` 时才清零。

---

## 6. 完整数据流

```
用户调用 musaMemcpyAsync(dst, src, size, HtoD, stream)
  │
  ├─ Runtime 包装 (musa_wrappers_generated.cpp)
  │   ├─ toolsCallbackEnabled(RUNTIME, CBID=41) → 1
  │   ├─ toolsIssueCallback(ENTER, "musaMemcpyAsync", CBID=41, params)
  │   │   → IssueCallback()
  │   │     → ProcessDriverCallback(...)          ← libmupti.so
  │   │       ├─ t_runtime_api_depth: 0→1
  │   │       ├─ 记录 start 时间戳入栈
  │   │       └─ 分派用户 callback (如果注册了)
  │   │
  │   ├─ 调 Driver API: muMemcpyAsync(...)
  │   │   ├─ Driver 包装 (mu_wrappers_generated.cpp)
  │   │   │   ├─ toolsCallbackEnabled(DRIVER, CBID=60) → 1
  │   │   │   ├─ toolsIssueCallback(ENTER, "muMemcpyAsync", CBID=60, params)
  │   │   │   │   → ProcessDriverCallback(...)
  │   │   │   │     ├─ t_driver_api_depth: 0→1  (-depth>0, 不产出 activity)
  │   │   │   │     └─ 分派用户 callback
  │   │   │   │
  │   │   │   ├─ 实际执行 memcpy: MemcpyCommand 构造 + 提交
  │   │   │   │   └─ MUpti::EnterMemcpy(this)    ← 语义 hook (另一套机制!)
  │   │   │   │
  │   │   │   ├─ toolsIssueCallback(EXIT, "muMemcpyAsync", status=SUCCESS)
  │   │   │   │   → ProcessDriverCallback(...)
  │   │   │   │     ├─ t_driver_api_depth: 1→0
  │   │   │   │     ├─ 产出 DRIVER activity 记录:
  │   │   │   │     │   "DRIVER API muMemcpyAsync [100-450] cbid=60 return=0"
  │   │   │   │     └─ 分派用户 callback
  │   │   │   └─ return
  │   │
  │   ├─ toolsIssueCallback(EXIT, "musaMemcpyAsync", status=SUCCESS)
  │   │   → ProcessDriverCallback(...)
  │   │     ├─ t_runtime_api_depth: 1→0
  │   │     ├─ 产出 RUNTIME activity 记录:
  │   │     │   "RUNTIME API musaMemcpyAsync [50-500] cbid=41 return=0"
  │   │     └─ 分派用户 callback
  │   └─ return musaSuccess
  │
  └─ 用户代码继续...
```

输出（demo.cu 的 printActivity 会打出来）：

```
RUNTIME API musaMemcpyAsync [ 50 - 500 ] cbid=41, correlation 42
DRIVER API muMemcpyAsync [ 100 - 450 ] cbid=60, correlation 42
MEMCPY HtoD [ 12345 - 67890 ] size 1024, correlation 42
```

---

## 7. 与语义 Hook 的关系

```
┌────────────────────────────────────────────────────┐
│                   MUPTI Tracing                      │
│                                                      │
│  ┌─────────────────────┐  ┌──────────────────────┐  │
│  │  Tools Callback     │  │  语义 Hook            │  │
│  │  (API 级)            │  │  (命令级)              │  │
│  ├─────────────────────┤  ├──────────────────────┤  │
│  │ 触发: 每个 API Enter │  │ 触发: 命令构造/执行    │  │
│  │       /Exit          │  │       /同步/完成       │  │
│  │ 输出: DRIVER/RUNTIME │  │ 输出: MEMCPY/KERNEL/  │  │
│  │       activity       │  │       MEMSET/SYNC     │  │
│  │ 注册: Subscribe      │  │ 注册: EnableMUpti     │  │
│  │       (单槽)          │  │       Driver(函数指针) │  │
│  │ 启用: EnableAllCb    │  │ 启用: ready=true      │  │
│  │        InDomain      │  │       (全局开关)       │  │
│  └─────────────────────┘  └──────────────────────┘  │
│                                                      │
│                    共用:                              │
│            g_activity_buffer_manager                 │
│            (同一个 buffer 管理器)                      │
└────────────────────────────────────────────────────┘
```

两者**并行运行**，互不干扰。语义 hook 追踪 GPU 命令的实际执行，Tools callback 追踪 API 调用本身。它们的记录最终进入**同一个 buffer**，用户用同一个 `muptiActivityGetNextRecord` 读出。

---

## 8. 局限

1. **单槽设计**：同时只能有一个工具订阅 Tools callback。如果 nsys/ncu 已经在用，MUPTI 不能用。
2. **不支持每-CBID 启用**：`EnableAllCallbacksInDomain` 全开或全关，不能只追踪某个特定 API。
3. **新驱动才支持**：`g_driver_support_callback` 可能为 false（老驱动），此时 DRIVER/RUNTIME activity 不可用。
4. **Generated 代码依赖**：callback 触发点在 `mu_wrappers_generated.cpp` 里，如果新 API 的 wrapper 没加 toolsCallbackEnabled/toolsIssueCallback，这个 API 就不会被追踪。
