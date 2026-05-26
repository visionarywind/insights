# MUSA API Profiler — 工作全记录

## 一、目标

为 MUSA GPU 驱动栈开发一个零代码注入的 API Profiler，实现：
- 拦截 **Driver API (mu\*)** 调用并统计耗时
- 拦截 **Runtime API (musa\*)** 调用并统计耗时
- 双层叠加统计（Runtime → Driver 调用链可见）
- 在 **sglang** 等上层框架中零修改注入

## 二、Profiler 版本演进

| 版本 | 文件 | 核心改进 |
|------|------|---------|
| 原始 | `docs/api_latency_profiler.cpp` | 扁平统计，Tools Callbacks，mu* 命名 |
| v2 | `/tmp/api_latency_profiler_v2.cpp` | 手工 mu*→musa* 映射表（废弃） |
| v3 | `/tmp/api_latency_profiler_v3.cpp` | **MUPTI CBID 编译时名称表**（权威命名） |
| v4 | `/tmp/api_latency_profiler_v4.cpp` | **嵌套 self-time 追踪**（children_ns 扣除） |
| v5 | `/tmp/api_latency_profiler_v5.cpp` | 三路径架构（Public MUPTI + Driver Hooks + constructor） |
| **v6** | `docs/profiler/api_latency_profiler.cpp` | **5 个问题修复** + Tools Callbacks 主路径 |

### v6 修复的 5 个问题

| # | 问题 | 修复 |
|---|------|------|
| 1 | Constructor 提前初始化（LD_PRELOAD 时序错） | 删除 `AutoInit()`，仅 `InitializeInjection()` |
| 2 | thread_local stack 无 correlationId 匹配 | 改为 `g_pendingPublicCalls` map 按 ID 匹配 |
| 3 | Launch API 名被 kernel 名覆盖 | API 名保持不变，kernel 名进 g_kernelStats |
| 4 | Min 列打印 Self 值 | 新增 `minNs` 字段，正确跟踪最小值 |
| 5 | Subscriber handle 未保存/未清理 | 全局保存 + 析构中 unsubscribe |

## 三、容器环境演变

| 阶段 | 容器 | 状态 |
|------|------|------|
| 初始 | mochi-sglang (linux-ddk 镜像) | torch 正常，MUSA_INJECTION64_PATH 可用 |
| 中间 | MUSA toolkit 被删除 | torch 崩溃（libmusart/mudnn 丢失） |
| 修复 | 从 sglang_profile 复制 MUSA 4.3.5 库 | torch 恢复 |
| 更新 1 | 容器重置 | libmusart/mudnn ABI 不匹配（const vs 非 const） |
| 更新 2 | 容器重置 | 路径变为 `/usr/local/musa`，torch 正常 |
| 最终 | sglang_profile (备用) | torch 始终正常，但无 MUSA_INJECTION64_PATH |

## 四、MUPTI 技术分析

### 三条 Profiling 路径

```
路径 1: Public MUPTI API (muptiSubscribe + MUpti_CallbackData)
  UUID: N/A (公共 API)
  状态: 需要 libmupti.so，受 MUPTI_ERROR_MULTIPLE_SUBSCRIBERS 限制

路径 2: MUPTI Driver Hooks (muGetExportTable → Client::MUpti 0x76543210)
  提供: EnterRuntimeApi/ExitRuntimeApi, EnterDriverApi/ExitDriverApi
  状态: 与 Tools Callbacks 互斥，当前环境不可用

路径 3: Tools Callbacks (muGetExportTable → Client::Tools 0x76543215) ★ 当前使用
  提供: Tools::CallbackControllers.Subscribe → MUtoolsTraceApiMusa
  状态: 始终可用，v6 主路径
```

### 命名机制

- **编译时**：从 `mupti_driver_cbid.h` / `mupti_runtime_cbid.h` 提取 CBID 枚举
- **生成**：`cbid_names_driver.inc` (811 条目) + `cbid_names_runtime.inc` (519 条目)
- **运行时**：`CbidToName(cbid, isRuntime)` 查表 → `driver:mu*` 或 `runtime:musa*`

## 五、测试用例

### Test 1: Driver API (mu\*)

```
MUSA_INJECTION64_PATH=./liblatency_profiler.so tests/test_driver_api
```
结果：`driver:muCtxCreate`, `driver:muInit`, `driver:muMemAlloc_v2`, `driver:muMemFree_v2` 等全部捕获。

### Test 2: Runtime API (musa\*)

```
MUSA_INJECTION64_PATH=./liblatency_profiler.so tests/test_runtime_api
```
结果：`runtime:musaFree_v3020`, `runtime:musaGetDeviceCount_v3020` 等 + 底层 `driver:mu*` 同框出现。

### Test 3: Overlay 叠加统计

```
MUSA_INJECTION64_PATH=./liblatency_profiler.so tests/test_overlay
```
结果：`runtime:musaMalloc` (64ms) 叠加在 `driver:muDevicePrimaryCtxRetain` (64ms) + `driver:muMemAlloc` (78µs) 之上，调用链清晰可见。

### Test 4: sglang 集成

```
LD_PRELOAD=./musa_profiler.so MUPTI_API_PROFILE_USE_PUBLIC_CALLBACK=true \
  python -m sglang.bench_one_batch --model-path ... --batch 1 ...
```

**无 Graph 结果 (GPU 1):**
| 指标 | 值 |
|------|-----|
| Bench Prefill | 57ms (4,477 tok/s) |
| Decode Median | 27.4ms (36.5 tok/s) |
| Total | 1.82s (176 tok/s) |

**Top Profiler APIs:**
| API | Calls | Total |
|-----|-------|-------|
| driver:muModuleLoadData | 23 | 8,907ms |
| driver:muMemcpyHtoDAsync_v2 | 429 | 1,077ms |
| driver:muLaunchKernel | 53,387 | 223ms |
| driver:muCtxGetDevice | 550,877 | 91ms |
| driver:muStreamCreateWithPriority | 96 | 129ms |

**有 Graph 结果 (GPU 7):**
| 指标 | 值 | vs 无 Graph |
|------|-----|------------|
| Bench Prefill | 55ms (4,665 tok/s) | — |
| Decode Median | **15.5ms** (65 tok/s) | **1.8x 快** |
| Total | **1.01s** (316 tok/s) | **1.8x 快** |

## 六、Profiler 核心机制

### 注入方式

```bash
# 方式 1: MUSA driver 自动注入 (旧容器可用)
export MUSA_INJECTION64_PATH=/path/to/liblatency_profiler.so

# 方式 2: LD_PRELOAD + constructor (通用)
export LD_PRELOAD=/path/to/musa_profiler.so
export MUPTI_API_PROFILE_USE_PUBLIC_CALLBACK=true

# 方式 3: Python ctypes 手动调用 (精准控制)
lib = ctypes.CDLL("musa_profiler.so")
lib.InitializeInjection()
```

### 统计结构

```cpp
g_apiStats["driver:muModuleLoadData"] = {count:23, total:8907ms, self:8907ms, min:55us, max:2647ms}
g_apiStats["runtime:musaMalloc_v3020"] = {count:1, total:64ms, self:64ms, ...}
g_kernelStats["batch_gemv... [muLaunchKernel]"] = {count:10248, total:47.5ms, ...}
```

### 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `PROFILER_OUTPUT` | stderr | 报告输出文件 |
| `PROFILER_TOP_N` | 60 | 限制报告行数 |
| `PROFILER_DEBUG` | 0 | 调试日志 |
| `PROFILER_DRIVER` | 1 | 启用 Driver API 捕获 |
| `PROFILER_RUNTIME` | 1 | 启用 Runtime API 捕获 |
| `PROFILER_USE_MUPTI` | 0 | Public MUPTI 路径（需 libmupti） |

## 七、产出文件

```
docs/profiler/
├── api_latency_profiler.cpp      v6 源码 (524 行)
├── cbid_names_driver.inc         811 Driver CBID 名称表
├── cbid_names_runtime.inc        519 Runtime CBID 名称表
├── gen_cbid_names.py             名称表生成脚本
├── Makefile                      构建 + make test1/2/3/4
├── README.md                     使用文档
├── SUMMARY.md                    本文档
├── tests/
│   ├── commands.sh               全部运行命令
│   ├── test_driver_api.c         Test 1 源码
│   ├── test_runtime_api.c        Test 2 源码
│   ├── test_overlay.c            Test 3 源码
│   └── test_sglang.sh            Test 4 脚本
└── results/
    ├── test1_profiler.txt        Test 1 profiler 报告
    ├── test2_profiler.txt        Test 2 profiler 报告
    ├── test3_profiler.txt        Test 3 profiler 报告
    ├── test4_profiler.txt        Test 4 profiler 报告 (无 graph)
    ├── test4_graph_enabled.txt   Test 4 性能对比 (有 graph)
    └── test4_graph_result.txt    Graph 兼容性记录
```

## 八、已知限制

1. **MUPTI Driver Hooks 与 Tools Callbacks 互斥** — 无法同时使用两层
2. **Public MUPTI 受单一订阅者限制** — 容器中其他 profiler 占用时不可用
3. **GPU kernel 执行时间不可拆分** — API Profiler 只能拿到 launch 开销 + sync 等待时间，无法拆分单个 kernel 的 GPU 执行时间
4. **MUSA Graph 需要驱动兼容** — 部分驱动版本不支持 stream capture
5. **容器环境依赖** — profiler .so 需要在目标容器内编译或使用匹配的库版本
