# MUSA Trace 机制：从入门到设计原理

> 从一个真实需求出发——"我想统计每个 API 的调用耗时"——逐步揭示整个 trace 系统的设计

---

## 第一章：需求 — 统计 API 调用耗时

假设你是一个性能工程师，在 S5000 上跑 AI 推理，发现某段代码比预期慢。你想知道：

> "到底是 `muMemAlloc` 慢，还是 `muMemcpyHtoD` 慢？每个 API 被调用了多少次？平均耗时多少？"

你需要一个**不会修改被测代码**的方案——不能在每个 `muMemAlloc` 调用前后手动加 `clock()`。

MUSA 驱动的 trace 系统就是为此设计的。你只需要写一个小的 `.so` 库，注入到进程，就能拦截所有 600 个 API 的进入和退出。

---

## 第二章：最小可用方案 — 拦截一个 API

### 2.1 先写一个最简单的 profiler

```cpp
// my_profiler.cpp — 编译为 libmyprofiler.so

#include <cstdio>
#include <chrono>
#include <unordered_map>
#include "export_table.h"        // MUSA 公开的类型定义
#include "mupti_driver_cbid.h"   // API → CBID 映射

// 存储每个 API 调用开始时间的哈希表
std::unordered_map<uint32_t, uint64_t> g_apiStartTimes;

// ★ 这是核心回调函数 — 每个 API 调用都会触发
void MyApiCallback(void* userdata, MUtools_cb_domain domain,
                   uint32_t cbid, const void* pParams)
{
    auto* params = static_cast<const MUtoolsTraceApiMusa*>(pParams);

    if (params->apiEnterOrExit == MU_TOOLS_API_ENTER) {
        // API 进入 — 记录开始时间
        g_apiStartTimes[*params->pCorrelationId] =
            std::chrono::steady_clock::now().time_since_epoch().count();

    } else {  // MU_TOOLS_API_EXIT
        // API 退出 — 计算耗时
        auto it = g_apiStartTimes.find(*params->pCorrelationId);
        if (it != g_apiStartTimes.end()) {
            auto now = std::chrono::steady_clock::now().time_since_epoch().count();
            auto elapsed = now - it->second;

            printf("[PROFILER] %-30s → %d, 耗时: %lu ns\n",
                   params->functionName, *params->pStatus, elapsed);

            g_apiStartTimes.erase(it);
        }
    }
}
```

**关键点**:
- `pParams->apiEnterOrExit` 告诉你这是 ENTER 还是 EXIT
- `*pParams->pCorrelationId` 是本次调用的唯一 ID，ENTER 和 EXIT 共享
- `params->functionName` 是 API 名称字符串（如 `"muMemAlloc"`）
- `*pParams->pStatus` 是返回码（EXIT 时才有效）

### 2.2 注册 profiler — 三个步骤

```cpp
// 在 my_profiler.cpp 中添加

// ★ 这是注入入口 — 驱动初始化时会调用此函数
extern "C" void InitializeInjection() {
    // Step 1: 获取工具操作表
    const void* pExportTable = nullptr;
    MUuuid toolsUuid = {};
    *(uint32_t*)&toolsUuid = 0x76543211;  // Client::Tools 的 magic number

    muGetExportTable(&pExportTable, &toolsUuid);
    auto* tools = static_cast<const Tools::ExportTable*>(pExportTable);

    // Step 2: 订阅 (Subscribe) — 告诉驱动 "我要接收回调"
    MUtoolsCbSubscriberHandle handle;
    tools->callback->Subscribe(&handle, MyApiCallback, nullptr);

    // Step 3: 选择要追踪的 API — 这里追踪 muMemAlloc
    // CBID_29 = muMemAlloc (在 mupti_driver_cbid.h 中定义)
    tools->callback->EnableCallback(1, handle,
        MU_TOOLS_CB_DOMAIN_DRIVER_API, 29);
    // CBID_43 = muMemcpyHtoD
    tools->callback->EnableCallback(1, handle,
        MU_TOOLS_CB_DOMAIN_DRIVER_API, 43);
    // CBID_60 = muMemcpyHtoDAsync
    tools->callback->EnableCallback(1, handle,
        MU_TOOLS_CB_DOMAIN_DRIVER_API, 60);
}
```

### 2.3 部署与运行

```bash
# 编译 profiler
g++ -fPIC -shared my_profiler.cpp -o libmyprofiler.so -I/usr/local/musa/include

# 注入到任何使用 MUSA 的应用
MUSA_INJECTION64_PATH=./libmyprofiler.so ./my_app
```

输出示例:
```
[PROFILER] muMemAlloc                     → 0, 耗时: 45231 ns
[PROFILER] muMemcpyHtoD                   → 0, 耗时: 8123 ns
[PROFILER] muMemAlloc                     → 0, 耗时: 38129 ns
[PROFILER] muMemcpyHtoDAsync             → 0, 耗时: 512 ns
```

到这里，你已经有了一个**可工作的 API 耗时 profiler**，总共约 50 行代码。下面解释每一层的工作原理。

---

## 第三章：设计原理 — 逐层拆解

上面的 50 行代码之所以能工作，是因为 MUSA 驱动在编译时就为**每一个 API 函数**插入了追踪探针。我们从上到下逐层拆解。

### 3.1 第一层：自动生成的探针 (mu_wrappers_generated.cpp)

每个公开 API（如 `muMemAlloc`）的函数体是自动生成的。以 `muCtxGetId` 为例：

```cpp
// mu_wrappers_generated.cpp (自动生成, 22,000+ 行)
MUresult MUSAAPI muCtxGetId(MUcontext ctx, unsigned long long* ctxId) {
    MUresult status = MUSA_SUCCESS;

    // ★ 1. 创建 RAII 守卫 — 记录 API 名、分配关联 ID
    ApiTrace trace(status, "muCtxGetId",
                   MUPTI_CBID_GET(muCtxGetId), ctx, ctxId);

    // ★ 2. 检查 "我这个 API 被 profiler 订阅了吗?"
    if (toolsCallbackEnabled(MU_TOOLS_SUBSCRIBER_HANDLE,
                             MU_TOOLS_CB_DOMAIN_DRIVER_API,
                             MUPTI_DRIVER_TRACE_CBID_muCtxGetId))
    {
        // ★ 3. 有 profiler 关注 — 构造参数包
        MUtoolsTraceApiMusa inParams{};
        inParams.functionName   = "muCtxGetId";
        inParams.functionId     = MUPTI_DRIVER_TRACE_CBID_muCtxGetId;
        inParams.pCorrelationId = &correlationId;    // 这次调用的唯一 ID
        inParams.pStatus        = &status;
        inParams.apiEnterOrExit = MU_TOOLS_API_ENTER;

        // ★ 4. 通知 profiler: "muCtxGetId 要开始执行了"
        toolsIssueCallback(HANDLE, DRIVER_API, CBID, &inParams);

        // ★ 5. 执行真正的驱动代码
        status = muapiCtxGetId(ctx, ctxId);

        // ★ 6. 通知 profiler: "muCtxGetId 执行完了, 返回码是 status"
        inParams.apiEnterOrExit = MU_TOOLS_API_EXIT;
        toolsIssueCallback(HANDLE, DRIVER_API, CBID, &inParams);

    } else {
        // ★ 无 profiler — 直接执行, 零开销
        status = muapiCtxGetId(ctx, ctxId);
    }

    return status;
}
```

**设计要点**:

| 问题 | 答案 |
|------|------|
| 这个代码是谁写的？ | **自动生成**的。每个 API 有独立函数，模式完全一致 |
| 如果没有 profiler 注入，有性能开销吗？ | **零开销**。`toolsCallbackEnabled` 只是一个整数数组的 O(1) 查表，如果对应 CBID 没被启用，直接走 else 分支 |
| 为什么需要 `correlationId`？ | 因为 API 可能是多线程并发调用的，ENTER 和 EXIT 之间可能插入其他 API 的 ENTER。`correlationId` 保证你能正确配对 |
| `ApiTrace` 是干什么的？ | RAII 守卫。构造时记录日志；析构时如果返回码是错误，自动打印 backtrace + 设置线程局部 last error |

### 3.2 第二层：CBID — 给每个 API 一个编号

`mupti_driver_cbid.h` 是一个自动生成的枚举，给每个 API 分配唯一 ID：

```c
typedef enum MUpti_driver_api_trace_cbid_enum {
    MUPTI_DRIVER_TRACE_CBID_INVALID           = 0,
    MUPTI_DRIVER_TRACE_CBID_muInit            = 1,
    // ...
    MUPTI_DRIVER_TRACE_CBID_muMemAlloc        = 29,
    // ...
    MUPTI_DRIVER_TRACE_CBID_muMemcpyHtoD      = 43,
    MUPTI_DRIVER_TRACE_CBID_muMemcpyHtoDAsync = 60,
    // ... 共 812 个条目 ...
    MUPTI_DRIVER_TRACE_CBID_SIZE              = 812,
};
```

**为什么需要编号而不是用字符串匹配？** 性能。`toolsCallbackEnabled` 内部是 `g_callbackEnabled[domain][cbid]` — 一次数组索引，不需要字符串比较。812 个条目的数组只占 ~3KB。

### 3.3 第三层：callback.cpp — 订阅与分发

你写的 `MyApiCallback` 是怎么被调到的？看 `callback.cpp` 的三个关键函数：

```cpp
// ============ 订阅 (Subscribe) ============
MUresult toolsSubscribe(handle, callback, userdata) {
    // 1. CAS 原子操作抢占全局槽位 (只允许一个 profiler)
    if (g_hasSubscriber.exchange(1) != 0)
        return ERROR;  // 已经被别的 profiler 占用了

    // 2. 用 seq-lock 写入 callback 指针和 userdata
    g_subscriber.seq++;        // 奇数 = "正在写入, 不要读"
    g_subscriber.userdata = userdata;
    g_subscriber.seq++;        // 偶数 = "写入完成, 可以安全读"
    g_subscriber.callback = callback;

    return SUCCESS;
}

// ============ 启用/禁用 (Enable) ============
MUresult toolsEnableCallback(enable, handle, domain, cbid) {
    g_callbackEnabled[domain][cbid] = enable;  // 就一行!
}

// ============ 分发 (Issue) ============
void IssueCallback(domain, cbid, params) {
    // seq-lock 读取: 如果 seq 前后一致, 说明没有并发写入
    uint32_t s1 = g_subscriber.seq;
    auto cb     = g_subscriber.callback;
    void* data  = g_subscriber.userdata;
    uint32_t s2 = g_subscriber.seq;

    if (s1 == s2 && cb != nullptr) {
        cb(data, domain, cbid, params);  // → 你的 MyApiCallback
    }
}
```

**设计要点**:

| 问题 | 答案 |
|------|------|
| 为什么只允许一个 profiler？ | 简化并发模型。多个 profiler 会导致回调顺序不确定、性能不可控。多个工具的需求通过注入层自己多路分发解决 |
| `seq-lock` 是什么？ | 一种无锁并发技术。写入者递增序列号，读取者在读前后检查序列号是否一致。避免了互斥锁的开销 |
| 为什么 `EnableCallback` 不在 `Subscribe` 时一次性启用全部？ | 细粒度控制。你可能只想追踪 `muMemAlloc`，不关心 `muEventCreate`。按 API 开关减少 99% 的噪音 |

### 3.4 第四层：注入机制 — 如何不用改一行代码

```cpp
// driver/internal.h:318-387
class InjectionManager {
    MUresult Init() {
        // 读取环境变量
        const char* path = getenv("MUSA_INJECTION64_PATH");
        if (!path) return SUCCESS;  // 没有设置 → 跳过

        // dlopen 你的 profiler 库
        void* lib = dlopen(path, RTLD_NOW);

        // 查找 InitializeInjection 函数并调用
        auto init = (void(*)())dlsym(lib, "InitializeInjection");
        init();  // → 你的 InitializeInjection() 执行
    }
};

// InjectionManager 在 muInit() 内部被调用
// 所以你的 profiler 在所有 GPU 操作之前就已经注册好了
```

**为什么选择环境变量注入而不是链接？**
- 不需要修改被测应用的编译脚本
- 不需要修改被测应用的代码
- 可以在 CI 流水线中按需开关：`MUSA_INJECTION64_PATH=./profiler.so pytest ...`

### 3.5 第五层：MUSA-Runtime 侧

MUSA-Runtime（`libmusart.so`）有**完全相同的机制**，追踪的是 `musa*` 运行时 API（如 `musaMalloc`, `musaMemcpy`）。

区别仅在于：
- Domain 是 `MU_TOOLS_CB_DOMAIN_RUNTIME_API`（值为 2，驱动是 1）
- CBID 来自 `mupti_runtime_cbid.h`（522 个条目，驱动是 812）
- 参数结构体是 `MUtoolsTraceRuntimeApiMusa`

如果你同时注入驱动层和运行时层的追踪，你会看到：
```
[ENTER] musaMalloc(dptr, 4096)       ← runtime 层
  [ENTER] muMemAlloc(dptr, 4096)     ← driver 层
  [EXIT]  muMemAlloc → SUCCESS
[EXIT]  musaMalloc → SUCCESS
```

---

## 第四章：进阶 — 不止是 API 耗时

### 4.1 拦截执行 (Mock/Stub)

你的回调可以在 ENTER 阶段**阻止真实 API 执行**：

```cpp
void MyCallback(void* data, MUtools_cb_domain domain,
                uint32_t cbid, const void* pParams) {
    auto* params = (MUtoolsTraceApiMusa*)pParams;

    if (params->apiEnterOrExit == MU_TOOLS_API_ENTER) {
        // 如果这是 muMemAlloc，直接返回模拟的成功
        if (params->functionId == 29) {  // CBID_muMemAlloc
            *params->pSkipDriverImpl = 1;    // ← 跳过真实执行!
            *params->pStatus = MUSA_SUCCESS;  // ← 手动设置返回码
        }
    }
}
```

这在你想要模拟 GPU 资源不足或故障注入时非常有用。

### 4.2 参数修改

回调函数也可以**修改**传入参数：

```cpp
void MyCallback(void* data, ...) {
    auto* params = (MUtoolsTraceApiMusa*)pParams;

    if (params->functionId == 29) {  // muMemAlloc
        // params->params 指向 muMemAlloc_params 结构体
        auto* allocParams = (muMemAlloc_params*)params->params;

        // 把每次分配的大小限制在 1GB
        if (allocParams->bytesize > 1024*1024*1024) {
            allocParams->bytesize = 1024*1024*1024;
        }
    }
}
```

### 4.3 命令行工具: 808 个 API 全量追踪

```cpp
// 一键启用所有 API 追踪
tools->callback->EnableAllCallbacksInDomain(1, handle, DRIVER_API);

// 现在你的 MyApiCallback 会收到全部 812 个 API 的 ENTER/EXIT
// 可以构建完整的调用时间线
```

---

## 第五章：技术决策回顾

| 决策 | 为什么这样设计 |
|------|---------------|
| **自动生成探针代码** | 600 个 API，手工写探针不可维护。生成器保证一致性 |
| **CBID 枚举 + 位图** | O(1) 查询某个 API 是否被追踪，无字符串比较开销 |
| **单一订阅者 + seq-lock** | 避免互斥锁在热路径上的开销。多 profiler 场景极少 |
| **环境变量注入** | 零侵入。不改被测代码，不改编译选项 |
| **ENTER/EXIT 双回调** | 工具可以精确计算耗时、构建调用树 |
| **pSkipDriverImpl** | 支持 mock、fault injection、参数修改等高级场景 |
| **correlationId** | 多线程环境下正确配对 ENTER/EXIT |
| **双路径 (if/else)** | 无 profiler 时零开销 — `toolsCallbackEnabled` 返回 0 就直接走 else |

---

## 第六章：完整代码 — 生产级 API 耗时统计

```cpp
// api_latency_profiler.cpp
// 编译: g++ -fPIC -shared -O2 api_latency_profiler.cpp -o liblatency.so
// 使用: MUSA_INJECTION64_PATH=./liblatency.so ./your_app

#include "export_table.h"
#include "mupti_driver_cbid.h"
#include <cstdio>
#include <chrono>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <mutex>

struct CallRecord {
    std::string name;
    uint64_t     start_ns;
};

struct ApiStats {
    uint64_t total_ns = 0;
    uint64_t count    = 0;
};

class ApiLatencyProfiler {
    std::mutex m_mutex;
    std::unordered_map<uint32_t, CallRecord> m_active;  // corrId → record
    std::unordered_map<std::string, ApiStats> m_stats;   // name → stats

public:
    void OnCallback(MUtoolsTraceApiMusa* params) {
        if (params->apiEnterOrExit == MU_TOOLS_API_ENTER) {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_active[*params->pCorrelationId] = {
                params->functionName,
                std::chrono::steady_clock::now().time_since_epoch().count()
            };
        } else {
            std::lock_guard<std::mutex> lock(m_mutex);
            auto it = m_active.find(*params->pCorrelationId);
            if (it == m_active.end()) return;

            auto now = std::chrono::steady_clock::now().time_since_epoch().count();
            auto& st = m_stats[it->second.name];
            st.total_ns += (now - it->second.start_ns);
            st.count++;
            m_active.erase(it);
        }
    }

    void PrintReport() {
        std::lock_guard<std::mutex> lock(m_mutex);
        std::vector<std::pair<std::string, ApiStats>> sorted(
            m_stats.begin(), m_stats.end());

        std::sort(sorted.begin(), sorted.end(),
            [](auto& a, auto& b) { return a.second.total_ns > b.second.total_ns; });

        printf("\n========== API Latency Report ==========\n");
        printf("%-35s %10s %12s %12s\n", "API", "Count", "Total(us)", "Avg(us)");
        printf("--------------------------------------------------------------\n");
        for (auto& [name, st] : sorted) {
            printf("%-35s %10lu %12lu %12lu\n",
                   name.c_str(), st.count,
                   st.total_ns / 1000, st.total_ns / 1000 / std::max(st.count, 1UL));
        }
    }

    ~ApiLatencyProfiler() { PrintReport(); }
};

// 全局实例 (析构时自动输出报告)
static ApiLatencyProfiler g_profiler;

// 回调适配器
static void ProfilerCallback(void*, MUtools_cb_domain,
                             uint32_t, const void* pParams) {
    g_profiler.OnCallback((MUtoolsTraceApiMusa*)pParams);
}

// 注入入口
extern "C" void InitializeInjection() {
    const void* pTable = nullptr;
    MUuuid uuid = {};
    *(uint32_t*)&uuid = 0x76543211;  // Tools
    muGetExportTable(&pTable, &uuid);
    auto* tools = (Tools::ExportTable*)pTable;

    MUtoolsCbSubscriberHandle h;
    tools->callback->Subscribe(&h, ProfilerCallback, nullptr);
    tools->callback->EnableAllCallbacksInDomain(1, h, MU_TOOLS_CB_DOMAIN_DRIVER_API);
}
```

**输出示例**:
```
========== API Latency Report ==========
API                                  Count    Total(us)      Avg(us)
--------------------------------------------------------------
muMemAlloc                           1024        46231          45
muMemcpyHtoD                          512         8123          15
muLaunchKernel                        256       123456         482
muStreamSynchronize                   256        89123         348
muMemFree                            1024         3456           3
```

---

## 附录：文件速查

| 文件 | 作用 |
|------|------|
| `driver/mu_wrappers_generated.cpp` | 自动生成的探针代码 (22k 行) |
| `driver/callback.cpp` | 订阅/启用/分发核心逻辑 (244 行) |
| `driver/callback.h` | tools* 函数声明 |
| `driver/internal.h:73-229` | ApiInvocationGuard + ApiTrace RAII 类 |
| `mupti_driver_cbid.h` | 812 个 CBID 枚举 |
| `export_table.h:32-40` | Domain 枚举 (DRIVER=1, RUNTIME=2, RESOURCE=3, SYNC=4) |
| `export_table.h:82-93` | MUtoolsTraceApiMusa 参数结构体 |
| `export_table.h:288-386` | MUpti 驱动 hook 类型定义 (50+ 函数指针) |
| `export_table.h:727-830` | MUpti::ExportTable 访问器结构体 |
| `MUSA-Runtime/src/mupti/` | 运行时侧并行 MUPTI 实现 |
