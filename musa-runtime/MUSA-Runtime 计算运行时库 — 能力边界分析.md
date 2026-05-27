# MUSA-Runtime 计算运行时库 — 能力边界分析

**分析日期：** 2026-05-25
**仓库路径：** `/home/shanfeng/workspace/MUSA-Runtime/`
**版本：** 5.1.0（develop），commit a6aa4733
**分析对象：** `include/`（57 个头文件）+ `src/`（23 个 .cpp，29 个源文件）

---

## 零、核心结论

```
MUSA-Runtime 的能力边界 = CUDA Runtime API (cudart) 兼容层
  ├── 定位: 用户态计算最核心一层
  ├── 上层: mtcc 编译器生成代码
  ├── 下层: 动态加载 libmusa.so (MUSA Driver) → 驱动 ioctl → KMD
  ├── API 兼容度: CUDA Runtime 12.x 水平 (__MUSART_API_VERSION = 12080)
  └── 不对称性: 运行时层不做任何 GPU 硬件操作 — 全部委托给驱动
```

---

## 一、仓库元信息

| 项目 | 值 |
|------|-----|
| 语言 | C++17 (Linux) / C++20 (Windows) |
| 构建 | CMake ≥ 3.10 |
| 产物 | `libmusart.so` (SOVERSION=5), `libmusart_static.a`, `musaInfo` |
| 公共 API 符号 | **337 个** (`runtime_symbols.ver`) |
| musa* 函数声明 | **430 个** (`musa_runtime_api.h`) |
| 内部 musaapi* 函数 | **341 个** (`musaapi.h`) |
| 实现文件 | 23 个 `.cpp` (29 个源文件，含 `.h`) |
| 最文件 | `musa_wrappers_generated.cpp` (12,642 行 — 自动生成) |
| 公共头文件 | 48 个 `.h` (15,936 行主 API 头) |
| 设备端头文件 | device_atomic, device_functions, math, mp_ext_* 等 |
| 单元测试 | 25 个 Google Test 模块目录 |
| 平台支持 | Linux x86_64 (主力) + aarch64 交叉编译 + Windows x86_64 |
| git 子模块 | `musa_shared_include/` (与驱动共享类型定义) |

---

## 二、能力边界：架构定位

### 2.1 在 MUSA 软件栈中的位置

```
用户代码 / AI 框架 (PyTorch, TensorFlow, SGLang ...)
  │
  ├──> musaRuntime API (musaMalloc, musaLaunchKernel, ...)
  │    ┌─────────────────────────────────────────┐
  │    │  MUSA-Runtime (本仓库)                    │
  │    │  337 个导出 musa* 函数                   │
  │    │  负责: API 兼容, 初始化, 上下文管理,      │
  │    │         错误传播, fat binary 注册         │
  │    └──────────────┬──────────────────────────┘
  │                   │ ExportTable (动态加载)
  │                   ▼
  │    ┌─────────────────────────────────────────┐
  │    │  libmusa.so (MUSA Driver)               │
  │    │  560 个 muapi* 函数                     │
  │    │  负责: 实际 GPU 操作                    │
  │    └──────────────┬──────────────────────────┘
  │                   │ ioctl
  │                   ▼
  │    ┌─────────────────────────────────────────┐
  │    │  gr-kmd (GPU 内核驱动)                   │
  │    │  负责: 硬件交互                          │
  │    └─────────────────────────────────────────┘
```

### 2.2 运行时 vs 驱动的职责边界

| 职责 | MUSA-Runtime | MUSA Driver (libmusa.so) |
|------|-------------|-------------------------|
| API 接口 | `musa*` (CUDA Runtime 兼容) | `mu*` (CUDA Driver API 兼容) |
| 设备管理 | 设备枚举、属性缓存、初始化 | 实际设备打开、Context 创建 |
| 内存分配 | `musaMalloc` → 参数校验 + 转发 | `muMemAlloc_v2` → ioctl → KMD |
| Kernel 启动 | `musaLaunchKernel` → 参数打包 | `muLaunchKernel` → GPU 调度 |
| Graph 管理 | `musaGraphCreate` → 转发 | `muGraphCreate` → 内核态图管理 |
| Stream 管理 | `musaStreamCreate` → 转发 | `muStreamCreate` → GPU 硬件队列 |
| Error 处理 | `musaGetLastError`, 错误码映射 | `MUresult` 返回 |
| 日志追踪 | `ApiInvocationGuard`, `MUSA_LOG` | — |
| 初始化 | `dlopen("libmusa.so")`, fat binary 注册 | `muInit(0)` |
| 线程安全 | TlsData, 原子初始化状态机 | — |
| 设备端头文件 | `device_functions.h`, `math_functions.h` | — (仅运行时提供) |

### 2.3 关键设计决策

**运行时与驱动不共享地址空间**（运行时 `musa*` → 驱动 `mu*` 是函数调用，不是链接）：

```
运行时          驱动
musa*()  ──→  ExportTable.memory->muMemAlloc_v2()
                  │
                  └── 一个函数指针跳转 (无 IPC/序列化)
```

**驱动动态加载**：运行时不链接 `libmusa.so`，启动时 `dlopen` 查找。失败则所有 API 返回 `musaErrorInitializationError`。

**自动生成包装器**：`musa_wrappers_generated.cpp` (12,642 行) 为每个公共 `musa*` API 插入统一的 `ApiInvocationGuard` 追踪、错误处理、MUPTI 回调。

---

## 三、能力边界：API 覆盖

### 3.1 公共 musa* API 全景

337 个导出符号按领域分布：

| API 域 | 导出数 | 说明 | CUDA 对应 |
|--------|-------|------|-----------|
| **Device** | ~30 | 设备枚举/属性/同步/重置 | `cudaGetDevice`, `cudaDeviceSynchronize` |
| **Memory** | ~45 | malloc/free/memcpy/memset/memAdvise/IPC | `cudaMalloc`, `cudaMemcpy`, `cudaFree` |
| **Stream** | ~35 | 创建/同步/捕获/回调/优先级 | `cudaStreamCreate`, `cudaStreamSynchronize` |
| **Event** | ~10 | 创建/记录/同步/耗时 | `cudaEventCreate`, `cudaEventElapsedTime` |
| **Graph** | ~80 | 创建/节点/实例化/启动/条件句柄 | `cudaGraphCreate`, `cudaGraphLaunch` |
| **Module/Kernel** | ~25 | 函数属性/启动/动态并行 | `cudaLaunchKernel`, `cudaFuncGetAttributes` |
| **Texture/Surface** | ~20 | 纹理对象/表面对象/纹理引用 | `cudaCreateTextureObject` |
| **Occupancy** | ~5 | Block 大小计算 | `cudaOccupancyMaxActiveBlocksPerMultiprocessor` |
| **Peer Access** | ~5 | 跨设备直接访问 | `cudaDeviceCanAccessPeer` |
| **IPC** | ~8 | 跨进程内存/事件共享 | `cudaIpcGetMemHandle` |
| **Interop (GL/EGL/VDPAU)** | ~10 | 图形 API 互操作 | `cudaGraphicsGLRegisterBuffer` |
| **Profiler** | ~5 | 性能分析 start/stop | `cudaProfilerStart` |
| **Error** | ~5 | getLastError/getErrorString | `cudaGetLastError` |
| **Version** | ~5 | 运行时/驱动版本查询 | `cudaRuntimeGetVersion` |
| **External** | ~10 | 外部内存/信号量导入 | `cudaImportExternalMemory` |
| **MemPool** | ~20 | Pool 创建/属性/async alloc | `cudaMemPoolCreate`, `cudaMallocAsync` |
| **Library** | ~10 | 动态库加载 | `cuLibraryLoadData` |
| **运行时内部** | ~10 | fat binary 注册/函数注册 | `__cudaRegisterFatBinary` |

### 3.2 已覆盖的 CUDA 特性

对照 CUDA Runtime API，MUSA-Runtime 覆盖了 **几乎全部主要功能域**：

| 功能 | 状态 | 备注 |
|------|------|------|
| 设备管理 (get/set/sync/reset) | ✅ | 完整覆盖 |
| 内存管理 (malloc/free/memcpy/memset) | ✅ | 完整覆盖 |
| Stream (同步/异步/回调/优先级) | ✅ | 完整覆盖, 含 per-thread default stream |
| Event (记录/同步/耗时) | ✅ | 完整覆盖 |
| **Graph (全套)** | ✅ | 含 Conditional Handle/EdgeData/UserObject/InstantiateWithParams |
| **MemPool / Async Alloc** | ✅ | `musaMallocAsync`, `musaMemPoolCreate`, `musaMemPoolExportPointer` |
| **External Semaphore/Memory** | ✅ | `musaImportExternalSemaphore`, `musaSignalExternalSemaphoresAsync` |
| **Library API** | ✅ | `musaLibraryLoadData`, `musaLibraryGetKernel` (对标 CUDA 12.x) |
| Texture Object | ✅ | 创建/销毁/查询 |
| Surface Object | ✅ | 创建/销毁/查询 |
| Occupancy | ✅ | 完整覆盖 |
| Peer Access | ✅ | enable/disable/canAccess |
| IPC (内存+事件) | ✅ | GetMemHandle/OpenMemHandle |
| OpenGL/EGL/VDPAU 互操作 | ✅ | 注册/映射/取消映射 |
| 纹理引用 (TexRef) | ✅ | 完整覆盖 (create/destroy/set/get) |
| Tensor Core 指令 | ✅ | `muTensorDescriptorEncode`, `muTensorDirectConvEncode` |
| Thread Block Cluster | ✅ | `musaGetDeviceAttribute` 含 cluster 相关属性 |
| Cooperative Launch | ✅ | `musaLaunchCooperativeKernel` |
| 动态并行 | ✅ | 设备侧 `cudaGetParameterBuffer` 等 |

### 3.3 与 CUDA Runtime API 的关键差异

| 差异点 | CUDA Runtime | MUSA-Runtime |
|--------|-------------|-------------|
| 版本号 | `CUDART_VERSION` | `MUSART_VERSION = 50100` |
| API 版本 | `cuCtxGetApiVersion` | `__MUSART_API_VERSION = 12080` |
| 设备属性 | `cudaDeviceProp` 标准字段 | 同, 含 MThreads 扩展字段 |
| 错误码 | `cudaError_t` | `musaError_t` (含 `musaErrorTmeIllegalParameters` 等) |
| PEERMAP 标志 | CUDA 无 | `MUSA_PEER_MAP_PCIEONLY` 等 MThreads 专用 |
| 扩展指令 | — | `mp_ext_20/30/32/35/60` 按架构分级 |
| 部分函数名 | `cudaMalloc3D` | `muMemAllocPitch` 类似但名不同 |

---

## 四、能力边界：纵轴深度

### 4.1 MUSA-Runtime 自身不做什么

```
                      MUSA-Runtime 的边界
                          │
                          │  (自己做)
                          ▼
   ┌───────────────────────────────────────┐
   │  musa* API 包装           - 自动生成  │
   │  错误码映射 (MUresult→musaError_t)    │
   │  初始化状态机 (原子安全)               │
   │  TLS 上下文管理                       │
   │  ApiInvocationGuard 日志/追踪          │
   │  Fat Binary 注册管理 (ProgramState)    │
   │  ExportTable 管理和降级填充            │
   │  MUPTI 回调钩子                       │
   └───────────────────────────────────────┘
                          │
                          │  (委托给驱动)
                          ▼
   ┌───────────────────────────────────────┐
   │  所有实际 GPU 操作:                    │
   │  • 设备显存分配/释放                  │
   │  • Kernel 在 GPU 上的调度执行         │
   │  • Stream/Event 硬件同步              │
   │  • Graph 创建/实例化/启动             │
   │  • 纹理/表面采样                      │
   │  • IPC 句柄创建/打开                  │
   └───────────────────────────────────────┘
```

**MUSA-Runtime 不做的事（与 Driver 的边界）：**

| 操作 | 谁做 | 原因 |
|------|------|------|
| 实际 GPU 内存分配 | Driver (`muMemAlloc_v2 → ioctl`) | 需要 KMD 操作 |
| Kernel 发射到 GPU | Driver (`muLaunchKernel → ioctl`) | 需要 GPU 调度器 |
| 物理设备枚举 | Driver (`muDeviceGetCount → sysfs`) | 需要系统调用 |
| Hardware-specific 初始化 | Driver (`muInit(0)`) | 需要硬件访问 |
| GPU 内存映射 (MMU) | Driver (`muMemMap/Unmap`) | 需要页表操作 |
| GRAPH 实例化 | Driver (`muGraphInstantiate`) | 需要 GPU 编译 |

### 4.2 MUSA-Runtime 做不到的事（功能缺失）

**已知实现的桩函数：**

| 函数 | 状态 | 返回 |
|------|------|------|
| `muProfilerStart` | ![stub] 桩 | `MUSA_ERROR_STUB_LIBRARY` |
| `muProfilerStop` | ![stub] 桩 | `MUSA_ERROR_STUB_LIBRARY` |
| `musaProfilerStart` | ![stub] 桩 | `musaErrorNotSupported` |
| `musaProfilerStop` | ![stub] 桩 | `musaErrorNotSupported` |
| `musaClangSet*` / `musaClangLaunchKernel` | ![stub] 桩 | `musaErrorNotSupported` |

**标记 TODO 的代码点：**

| 位置 | TODO 内容 |
|------|----------|
| `musa_clang.cpp:44` | "have no idea the purpose of this CUDA API, let it be empty for the moment" |
| `musa_profiler.cpp:10` | "Add actual profiler start implementation, this is for temporarily passing cases" |
| `musa_profiler.cpp:15` | "Add actual profiler start implementation" |
| `internal.h:643` | TODO |
| `musa_module.cpp:125` | "Add more checks" |

**不支持的场景（运行时显式返回错误）：**

| 场景 | 错误码 |
|------|--------|
| 32-bit non-float 归一化读取 | `musaErrorNotSupported` |
| 文件作用域纹理/表面 (TextureObject 替代) | `musaErrorNotSupported` |
| 设备间 Peer Access 不存在 | `musaErrorNotSupported` |
| PTX 由不支持的工具链编译 | `musaErrorNotSupported` |
| 操作系统调用失败 | `musaErrorOSCallFailed` |
| 架构不支持该 limit | `musaErrorNotSupported` |
| 动态并行 + MPS | `musaErrorNotSupported` |

### 4.3 比对 CUDA Runtime 12.x 的潜在缺失

基于 CUDA 12.x 的已知功能，MUSA-Runtime 可能缺失或未实现的领域：

| 功能 | 状态估计 |
|------|---------|
| `cudaMemPoolTrimTo` | 已导出，取决于驱动实现 |
| `cudaGraphInstantiateWithParams` | 已导出 (Graph 80+) |
| `cudaGraphConditionalHandleCreate` | 已导出 |
| `cudaGraphExecUpdate_v2` | 已导出 |
| `cudaMemAdvise` (统一内存) | 已导出 |
| `cudaArrayGetSparseProperties` | 已导出 |
| 9 系列及之后新 GPU 架构的微调属性 | 取决于驱动支持 |
| `cudaGetDriverEntryPoint` | 可能为桩 |

---

## 五、能力边界：平台与架构

### 5.1 平台支持矩阵

| 平台 | CPU 架构 | 状态 | 备注 |
|------|---------|------|------|
| Linux | x86_64 | **✅ 主力平台** | CI 全量测试 |
| Linux | aarch64 | ✅ 交叉编译 | 构建可交叉编译 |
| Windows | x86_64 | ✅ 支持 | 代码中有 17 处 `_WIN32` 条件编译 |
| macOS | x86_64 / arm64 | ❌ 不支持 | 无任何 Apple 平台代码 |
| Android | aarch64 | ❌ 不支持 | 无相关代码 |

**Windows 支持的证据：**
- `src/musart.def` — Windows DLL 符号导出定义
- `src/internal.cpp` 中 17 处 `#if defined(_WIN32)` 条件编译
- `-fstack-protector-all` 等 GCC 安全标志在 Windows 用 MSVC 时不会启用
- C++20 标准 (vs Linux C++17)

### 5.2 GPU 架构支持

| 架构代号 | 头文件 | CI 测试 | 备注 |
|---------|--------|---------|------|
| mp_20 (qy1) | `mp_ext_20_intrinsics.h` + atomics | — | 最基础架构 |
| mp_30 (ph1) | `mp_ext_30_intrinsics.h` | — | — |
| mp_31 (ph1) | — | **✅ 全量** | 主目标架构 |
| mp_32 (ph1s) | `mp_ext_32_atomic_functions.h` | — | ph1 优化版 |
| mp_35 | `mp_ext_35_intrinsics.h` + atomics | — | — |
| mp_60 | `mp_ext_60_atomic_functions.h` | — | 最新架构 |
| mp_41 (hg) | — | — | 无专用头文件 |
| mp_42 (ls) | — | — | 无专用头文件 |
| mp_43 (hs) | — | — | 无专用头文件 |

**主目标**：mp_31 (ph1) — CI 压力和性能测试、CTS Part1~4、ASAN、mttrace 全量运行在此架构上。

### 5.3 测试覆盖矩阵

| 测试类型 | 硬件 | 覆盖内容 |
|---------|------|---------|
| Unit Test (25 模块) | 无 (纯逻辑) | 每个 API 域的独立测试 |
| CTS Part1~4 | qy2 + ph1 | MUSA API 兼容性全套 |
| mttrace smoke | ph1 | 性能分析工具兼容性 |
| ASAN smoke | ph1 | 内存错误检测 |
| Benchmark | qy2 + ph1 | 性能基线 |
| Part2~4 (含 cuda-samples) | ph1 | 第三方代码兼容性 |

**有 UT 的模块（25 个）：**

```
Device_Management, Memory_Management, Stream_Management,
Event_Management, Graph_Management, Peer_Device_Memory_Access,
External_Graph_Management, External_Management,
Occupancy_Management, Texture_Object_Management,
Texture_Reference_Management, Surface_Object_Management,
Graphics_Interoperability, Profiler_Management,
Version_Management, Error_Management, Notification_Management,
Peer_Management, Thread_Block_Cluster, Thread_Management,
Entry_Management, Internal_Management, Excution_Control,
Surface_Object, (and more)
```

**无独立 UT 的模块：** Library API, Tensor 相关

---

## 六、能力边界：代码覆盖热力图

```
src/
├── musa_wrappers_generated.cpp   ████████████████████████████  (12,642 行 — 自动生成)
├── internal.cpp                  ████████████████████████████  (2,388 行 — 核心基础设施)
├── musa_graph.cpp                ██████████████████████████░    (1,732 行 — Graph 80+ API)
├── musa_memory.cpp               ██████████████████████████     (1,561 行 — 内存管理)
├── musa_deprecated.cpp           ████████████████████░░░░░     (959 行 — 遗留 API)
├── musa_device.cpp               ████████████████████░░░░░     (841 行 — 设备管理)
├── musa_mempool.cpp              ████████████████░░░░░░░░░     (535 行 — MemPool)
├── musa_module.cpp               █████████████░░░░░░░░░░░░     (454 行 — Module)
├── musa_texobj.cpp               ████████████░░░░░░░░░░░░░     (435 行 — 纹理/表面)
├── musa_stream.cpp               ██████████░░░░░░░░░░░░░░░░    (319 行 — Stream)
├── musa_library.cpp              ██████░░░░░░░░░░░░░░░░░░░░    (190 行 — Library API)
├── musa_external.cpp             █████░░░░░░░░░░░░░░░░░░░░░    (137 行 — External)
├── musa_event.cpp                ████░░░░░░░░░░░░░░░░░░░░░░    (124 行 — Event)
├── musa_occupancy.cpp            ████░░░░░░░░░░░░░░░░░░░░░░    (111 行 — Occupancy)
├── musa_clang.cpp                ████░░░░░░░░░░░░░░░░░░░░░░    (111 行 — Clang 桥)
├── musa_oglinterop.cpp           ███░░░░░░░░░░░░░░░░░░░░░░░    (83 行 — GL 互操作)
├── musa_gfxinterop.cpp           ██░░░░░░░░░░░░░░░░░░░░░░░░    (75 行 — 图形互操作)
├── musa_entry.cpp                ██░░░░░░░░░░░░░░░░░░░░░░░░    (66 行 — 驱动入口)
├── musa_peer.cpp                 ██░░░░░░░░░░░░░░░░░░░░░░░░    (61 行 — Peer Access)
├── musa_error.cpp                █░░░░░░░░░░░░░░░░░░░░░░░░░    (38 行 — 错误码)
├── musa_notification.cpp         █░░░░░░░░░░░░░░░░░░░░░░░░░    (28 行 — 通知)
├── musa_version.cpp              █░░░░░░░░░░░░░░░░░░░░░░░░░    (26 行 — 版本)
└── musa_profiler.cpp             ░░░░░░░░░░░░░░░░░░░░░░░░░░    (17 行 — 桩)

include/
├── musa_runtime_api.h            ████████████████████████████  (15,936 行 — 主 API)
├── musa_runtime.h                █████████████████████████░    (3,094 行 — C++ 封装)
├── musa_occupancy.h              █████████████████████░░░░░    (1,802 行 — Occupancy)
├── mp_ext_20_intrinsics.h        ████████████████████░░░░░░    (1,497 行 — mp_20 内建)
├── musaTypedefs.h                ████████████████████░░░░░░    (926 行 — 类型)
├── ... 其他头文件                 ██~████░░░░░░░░░░░░░░░░░░░░   (其余 43 个)
```

图例：██ = 行数对应比例 (满=最大文件)

---

## 七、能力边界总结表

| 维度 | 在边界内 (Can do) | 在边界外 (Cannot do) |
|------|-------------------|---------------------|
| **API 风格** | CUDA Runtime (`musa*`) | CUDA Driver (`mu*`) — 委托给 libmusa.so |
| **GPU 操作** | 仅参数校验 + 转发 | 不做实际 GPU 操作 (Driver 做) |
| **初始化** | dlopen + ExportTable 初始化 | 硬件初始化 (`muInit(0)` 委托) |
| **平台** | Linux x86_64, aarch64, Windows x86_64 | macOS, Android, 其他 UNIX |
| **GPU 架构** | mp_20/30/31/32/35/60 (含扩展指令) | 新架构支持取决于驱动 |
| **CUDA 兼容** | CUDA Runtime 12.x 全套功能 | Profiler 为桩, Clang 桥不完整 |
| **测试** | 25 UT 模块 + CTS Part1~4 | 无 Library/Tensor UT |
| **设备端代码** | 提供头文件 (math/atomic/intrinsic) | 不包含任何 GPU 可执行代码 |
| **错误处理** | 120+ 错误码映射 + ApiInvocationGuard | — |
| **调试追踪** | MUSA_LOG (位掩码), MUPTI 回调 | 无内置 debugger (通过 mugdb) |
| **性能分析** | ApiInvocationGuard 计时 | Profiler start/stop 是桩 |
| **安全** | ASAN/TSAN, -fstack-protector-all, 符号加密 | — |
| **升级** | 运行时/驱动独立升级 (ABI 解耦) | ABI 兼容性缺少编译期检查 |

### 关键不对称性

1. **运行时→驱动不对称**:
   - 运行时 `musa*` 函数数 (337) 远小于驱动 `mu*` 函数数 (560)
   - 原因：一个 `musa*` 函数可能使用多个 `mu*` 调用
   - 运行时不做 1:1 包装 — 提供高层次抽象

2. **功能完成度不对称**:
   - Graph 管理 80+ API ✅ 完整
   - MemPool 20+ API ✅ 完整
   - Profiler 5 API ❌ 全是桩
   - 说明：工程优先级倾向于核心计算功能

3. **平台支持不对称**:
   - Linux x86_64 全量 CI 测试
   - Windows 支持但无 CI 验证
   - aarch64 可交叉编译但 CI 覆盖率未知

---

## 八、与 Driver API (mu*) 的映射关系

### 8.1 运行时包装 1:1 映射

部分 `musa*` 函数直接映射到单个 `mu*` 函数调用：

```
musaMalloc        → ExportTable.memory->muMemAlloc_v2
musaFree          → ExportTable.memory->muMemFree_v2
musaMemcpy        → ExportTable.memory->muMemcpy (UVA 推断)
musaStreamCreate  → ExportTable.stream->muStreamCreate
musaEventCreate   → ExportTable.event->muEventCreate
musaGraphCreate   → ExportTable.graph->muGraphCreate
```

### 8.2 运行时包装 N:1 映射

部分 `musa*` 函数组合多个 `mu*` 调用：

```
musaDeviceSynchronize → muCtxSynchronize (包含 muStreamSynchronize)
musaSetDevice         → muDevicePrimaryCtxRetain + muCtxSetCurrent
musaMallocManaged     → muMemAllocManaged (含 flags 推导)
musaLaunchKernel      → muLaunchKernel (含 fat binary → Module 解析)
```

### 8.3 运行时独有的功能（驱动不做）

| 功能 | 原因 |
|------|------|
| Fat Binary 注册/管理 | 驱动不知 fat binary 格式 |
| `ApiInvocationGuard` 日志/追踪 | 纯用户态功能 |
| `musaGetLastError` | 纯用户态 TLS |
| `musaGetErrorString` | 纯用户态字符串表 |
| Per-thread default stream | 纯用户态语义 |
| Device-side headers | 编译器用，驱动不关心 |

---

## 九、使用者指南

### 什么时候应该用 MUSA-Runtime

- 开发直接使用 MUSA GPU 的应用程序
- 将现有 CUDA 代码迁移到 MThreads GPU（API 签名兼容）
- 通过 `musaLaunchKernel` 和 `musaMalloc` 等 API 直接控制 GPU
- 需要 Graph、Stream、Event、IPC 等高级功能
- 兼容 PyTorch/TensorFlow 等框架（框架对接运行时层）

### 什么时候不应该用 MUSA-Runtime

- 需要直接控制 KMD 驱动行为 → 应使用 Driver API (`mu*`)
- 需要 GPU 固件/微架构级操作 → 应使用 `linux-ddk/`
- 需要在 AI 框架之上开发 → 应通过框架的 MUSA 后端（如 `sglang/`）
- 需要调试 GPU 硬件 → 应使用 KMD/Driver 层工具

### 与 Driver API 的选择建议

| 场景 | 推荐 API |
|------|---------|
| 日常 GPU 计算 | `musa*` (Runtime) |
| 细粒度资源控制 | `mu*` (Driver) |
| 框架集成 (PyTorch 等) | `mu*` (Driver) |
| 性能 profiling | `musa*` + MUPTI |
| 内核调试 | `mu*` + mugdb |

---

*本分析基于 `/home/shanfeng/workspace/MUSA-Runtime/` 源码分析*
