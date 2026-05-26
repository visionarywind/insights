# MUSA-Runtime 技术洞察报告

**分析日期：** 2026-05-25
**仓库路径：** `/home/shanfeng/workspace/MUSA-Runtime/`
**版本：** 5.1.0（develop）
**提交：** a6aa4733（master 分支）

---

## 一、项目概述

MUSA-Runtime 是**摩尔线程 MUSA GPU 的计算运行时库**，对标 NVIDIA CUDA Runtime API（`cudart`）。它提供设备管理、内存分配/拷贝、Stream/Event、Graph、纹理/表面对象、Profiler、进程间通信（IPC）、外部信号量/内存导入、OpenGL/EGL/VDPAU 互操作等完整功能。它是 C++17 项目，使用 CMake 构建系统。

**一句话定位：** MThreads GPU 生态中用户态计算最核心的一层。上层对接 mtcc（MUSA 编译器）生成的代码，下层通过动态加载 `libmusa.so` 与 MUSA Driver（类似 CUDA Driver API）通信。

---

## 二、目录结构全景

```
MUSA-Runtime/
├── include/                      # 公共 API 头文件（57 个文件）
│   ├── musa_runtime_api.h        # C API 主体（~340 个导出符号）
│   ├── musa_runtime.h            # C++ 便捷封装层
│   ├── musa_device_runtime_api.h # 设备侧入口点
│   ├── musaTypedefs.h            # 类型别名
│   ├── musart_platform.h         # 平台检测宏
│   ├── device_atomic_functions.{h,hpp}   # GPU 原子操作
│   ├── device_functions.h        # GPU 内建函数
│   ├── mp_ext_{20,30,32,35,60}_{...}.{h,hpp}  # 按架构分级的扩展指令
│   ├── math_functions.h          # GPU 数学函数
│   ├── musa_occupancy.h          # Occupancy 计算器
│   ├── musa_pipeline.h           # 异步 Pipeline
│   ├── musa_awbarrier.h          # 异步 Barrier
│   ├── musa_gl_interop.h         # OpenGL 互操作
│   ├── musa_egl_interop.h        # EGL 互操作
│   ├── musa_vdpau_interop.h      # VDPAU 互操作
│   ├── musa_profiler_api.h       # Profiler 接口
│   ├── CL/                       # OpenCL 兼容层（8 个头文件）
│   └── crt/                      # 设备侧 CRT（22 个头文件，仅设备编译）
│
├── src/                          # 实现代码（约 30 个 .cpp，扁平结构）
│   ├── musa_{device,memory,stream,event,graph,...}.cpp  # 按领域划分的实现
│   ├── musa_entry.cpp            # 驱动入口点解析
│   ├── musa_wrappers_generated.cpp  # 自动生成的 API 包装器
│   ├── internal.{h,cpp}          # 运行时核心：TlsData, ExportTableManager, ProgramState
│   ├── musaapi.h                 # 内部 API 声明（350+ musaapi* 函数）
│   ├── utils.h                   # 日志、动态库加载、线程信息工具
│   ├── runtime_symbols.ver       # Release 构建符号可见性控制
│   ├── musart.def                # Windows 符号导出定义
│   └── mupti/                    # MUPTI 回调钩子
│
├── musa_shared_include/          # GIT 子模块：运行时与驱动共享的内部头文件
│   ├── driver_types.h            # 运行时 ↔ 驱动的类型契约
│   ├── musa.h                    # 驱动 API 内部声明
│   ├── vector_types.h            # dim3, float3, int2 等向量类型
│   ├── export_table.h            # 动态符号导出表（驱动侧完整 API 签名）
│   ├── generated_musa_meta.h     # 自动生成的元数据
│   ├── mupti/                    # Profiler 回调 ID 定义
│   ├── mugdb/                    # 调试器支持
│   └── raytracing/               # 光线追踪支持
│
├── unittest/                     # Google Test（25 个模块目录）
│   ├── Device_Management/
│   ├── Memory_Management/
│   ├── Stream_Management/
│   ├── Event_Management/
│   ├── Graph_Management/
│   └── ...（共 25 个）
│
├── tools/                        # musaInfo 设备信息工具
├── module_version/               # 版本查询基础设施（子模块）
├── scripts/                      # clang_tidy.py, stub_scan.py
├── CMakeLists.txt                # 根构建文件
├── build.sh                      # CI 构建入口
├── install.sh                    # 安装到 /usr/local/musa/
└── .ciConfig.yaml                # CI 流水线配置
```

---

## 三、体系结构深度分析

### 3.1 四层架构

```
用户代码 (mtcc 编译的 GPU 代码)
   ↓ musaLaunchKernel, musaMalloc, ...
┌─────────────────────────────────────┐
│ Layer 1: API 包装器层               │
│ musa_wrappers_generated.cpp         │  ← 自动生成
│ (调用 musaapi* 内部函数)            │
├─────────────────────────────────────┤
│ Layer 2: 运行时实现层               │
│ musa_{domain}.cpp                   │
│ (musaapi* 函数，调用驱动表)          │
├─────────────────────────────────────┤
│ Layer 3: 导出表管理层               │
│ ExportTableManager (internal.h)     │
│ (封装驱动函数指针表)                 │
├─────────────────────────────────────┤
│ Layer 4: 驱动加载层                 │
│ Utils::Library (utils.h)            │
│ dlopen("libmusa.so.1") → muGetExportTable    │
│                                      │
│   ┌─────────────────────────────┐   │
│   │ libmusa.so (MUSA Driver)    │   │  ← 独立动态库
│   └─────────────┬───────────────┘   │
│                 ↓ (ioctl → 内核)     │
│   ┌─────────────────────────────┐   │
│   │ gr-kmd (GPU 内核驱动)       │   │
│   └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

**关键设计选择：**

- 运行时与驱动通过**动态加载 + 导出表**解耦。运行时初始化时 `dlopen("libmusa.so.1")`，查询 `muGetExportTable` 函数指针，再通过 UUID 获取 14 个子功能表。
- 运行时和驱动可以独立升级，只需保持导出表 ABI 兼容即可。
- **没有直接函数调用**——所有驱动 `mu*` 函数都通过 `g_ExportTable.GetDriverTable().domain->function` 访问器转发。

### 3.2 导出表（Export Table）设计

`export_table.h`（~714 行）定义了运行时与驱动之间的完整 API 合约。`ExportTableManager` 管理 14 个驱动子功能表：

| 子表 | 访问器路径 | 主要功能 |
|------|-----------|---------|
| General | `Driver::GeneralApiAccessors` | muInit, muGetExportTable |
| Device | `Driver::DeviceApiAccessors` | 设备枚举、属性查询、PCI Bus ID |
| Deprecated | `Driver::DeprecatedApiAccessors` | 遗留 API 兼容 |
| GfxInterop | `Driver::GfxInteropApiAccessors` | OpenGL/EGL/VDPAU 互操作 |
| Graph | `Driver::GraphApiAccessors` | CUDA Graph 全套操作（~80 个函数）|
| Context | `Driver::ContextApiAccessors` | Context 创建/销毁/切换 |
| Event | `Driver::EventApiAccessors` | Event 创建/记录/同步 |
| Memory | `Driver::MemoryApiAccessors` | malloc/free/memcpy/memset + MemPool |
| Stream | `Driver::StreamApiAccessors` | Stream 创建/同步/捕获 |
| Module | `Driver::ModuleApiAccessors` | Module 加载/函数查找 |
| Occupancy | `Driver::OccupancyApiAccessors` | Occupancy 计算 |
| Texture | `Driver::TextureApiAccessors` | 纹理/表面对象 |
| External | `Driver::ExternalApiAccessors` | 外部信号量/内存 |
| Library | `Driver::LibraryApiAccessors` | CUDA Library API |

此外还有 Runtime、Tools（回调）、Injection（动态注入库）三个表。

**初始化时使用 `PostFillAccessors`**——对未解析的函数指针（值为 `AccessorHint = 0xDEADBEEF`）填充为 `DriverApiNotFound` 哨兵，实现优雅降级。

### 3.3 核心基础设施

#### TlsData（线程局部存储）

每个线程持有一个 `TlsData` 实例，包含：
- `m_Initialized` — 初始化状态
- `m_LastError` — 支持 `musaGetLastError` 机制
- `m_CurrentCtx` — 当前 `MUcontext`
- `m_ExecStack` — 执行配置栈（支持 `<<<grid, block>>>` 语法）

#### ApiInvocationGuard

每个公共 API 入口自动构造 `ApiInvocationGuard`，提供：
- **API 调用追踪**（通过 `MUSA_LOG` 环境变量控制，位掩码日志级别）
- **错误回栈打印**（非成功返回时自动打印 backtrace）
- **MUPTI 回调触发**（`m_CallbackId` 用于性能分析工具注入）
- **Invocation 嵌套检测**（防止递归/重入调用）

#### ProgramState

管理运行时加载的所有 fat binary：
- 每个 fat binary 注册为 `Module` 对象
- 每个 Module 管理函数/变量/纹理/表面的 host→device 名称映射
- `GetFunction` 采用延迟初始化：按需 `muModuleLoadData` + 设备端模块加载
- 函数/变量查找使用**双重缓存**（`m_FunctionMapCache`）

### 3.4 初始化流程

```
1. 用户调用任意 musa* API
   ↓
2. musa_wrappers_generated.cpp 中的包装器
   ↓
3. musaapi* 实现函数
   ↓
4. ExportTableManager::Init()
   [原子状态机: UNINITIALIZED → INITIALIZING → INITIALIZED]
   ↓
5.   dlopen("libmusa.so.1") → 回退 "libmusa.so" (4.3.x 兼容)
   ↓
6.   dlsym("muGetExportTable")
   ↓
7.   通过 UUID 获取 4 张表: Runtime, Driver, Tools(可选), Injection(可选)
   ↓
8.   PostFillAccessors → 未填充函数填 DriverApiNotFound 哨兵
   ↓
9. InitPlatform() → muInit(0)
   ↓
10. InitDevice(deviceId) → muDevicePrimaryCtxRetain + muCtxSetCurrent
```

**注：** `ExportTableManager::Init()` 使用 `thread_local bool t_recursive` 和 `compare_exchange_strong` 原子状态机保证**线程安全 + 可重入**。

---

## 四、API 表面积

### 4.1 公共 API 符号（`runtime_symbols.ver`）

导出 **337 个公共符号**，涵盖以下领域：

| API 家族 | 函数数 | CUDA 对应 |
|----------|--------|-----------|
| Device Management | ~30 | cudaGetDevice, cudaSetDevice, cudaDeviceSynchronize... |
| Memory Management | ~45 | cudaMalloc, cudaFree, cudaMemcpy, cudaMemset... |
| Stream Management | ~35 | cudaStreamCreate, cudaStreamSynchronize, stream capture... |
| Event Management | ~10 | cudaEventCreate, cudaEventRecord, cudaEventElapsedTime... |
| Graph Management | ~80 | cudaGraphCreate, cudaGraphInstantiate, cudaGraphLaunch... |
| Module/Kernel | ~25 | cudaLaunchKernel, cudaFuncGetAttributes... |
| Texture/Surface | ~20 | cudaCreateTextureObject, cudaBindTexture... |
| Occupancy | ~5 | cudaOccupancyMaxActiveBlocksPerMultiprocessor... |
| Peer Access | ~5 | cudaDeviceCanAccessPeer, cudaDeviceEnablePeerAccess... |
| IPC | ~8 | cudaIpcGetMemHandle, cudaIpcOpenEventHandle... |
| Interop (GL/EGL/VDPAU) | ~10 | cudaGraphicsGLRegisterBuffer... |
| Profiler | ~5 | cudaProfilerStart, cudaProfilerStop... |
| Error/Handling | ~5 | cudaGetLastError, cudaGetErrorString... |
| Version | ~5 | cudaRuntimeGetVersion, cudaDriverGetVersion... |
| External | ~10 | cudaImportExternalMemory, cudaSignalExternalSemaphoresAsync... |
| MemPool | ~20 | cudaMemPoolCreate, cudaMallocAsync, cudaFreeAsync... |
| Library | ~10 | cuLibraryLoadData, cuLibraryGetKernel... |
| 运行时内部 | ~10 | __musaRegisterFatBinary, __musaRegisterFunction... |

**CUDA 兼容深度评估：**
- `MUSART_VERSION = 50100`（5.1.0）
- `__MUSART_API_VERSION = 12080`（内部 API 版本 ≈ CUDA 12.8 兼容水平）
- 支持 per-thread default stream（`MUSA_API_PER_THREAD_DEFAULT_STREAM`）
- 支持 Graph 全套：节点操作、条件句柄、用户对象、EdgeData、InstantiateWithParams
- 支持 MemPool / async allocator（`musaMallocAsync`, `musaMemPoolCreate`）
- 支持 CUDA 12.x 的外部信号量/内存导入机制
- 支持 Library API（对标 `cuLibrary*`）

### 4.2 内部 API（`musaapi.h`）

定义了 **~350 个 `musaapi*` 函数**作为公共 API 与驱动之间的桥梁。每个公共 `musa*` 函数由自动生成的包装器调用对应的 `musaapi*` 实现。

### 4.3 API 调用路径示例

```
musaMalloc(devPtr, size)
  → musa_wrappers_generated.cpp: musaMalloc(devPtr, size)
    → musaapiMalloc(devPtr, size)                [musa_memory.cpp]
      → InitPlatform(), InitDevice()
        → g_ExportTable.GetDriverTable().memory->muMemAlloc_v2(devPtr, size)
          → libmusa.so → ioctl → KMD
```

---

## 五、构建系统

### 5.1 CMake 构建

| 项目 | 值 |
|------|-----|
| 最小 CMake 版本 | 3.10 |
| C++ 标准 | C++17（Linux）/ C++20（Windows） |
| 共享库 | `libmusart.so`（SOVERSION=5） |
| 静态库 | `libmusart_static.a`（Release 含符号加密） |
| 工具 | `musaInfo`, `musa_runtime_version` |

**关键构建选项：**
- `MUSA_BUILD_DEBUG` — Debug 模式
- `MUSA_BUILD_ASAN` — AddressSanitizer
- `MUSA_BUILD_TSAN` — ThreadSanitizer（与 ASAN 互斥）
- `MUSA_BUILD_UT` — 启用单元测试
- `CSV_UNSUPPORTED` — Python 导出不支持 API 报表

### 5.2 Release 构建安全措施

```
编译 → 链接版本脚本(runtime_symbols.ver) → objcopy/strip 剥离未使用符号
      → 静态库额外: python 符号加密(encrypt_symbol.py)
```

安全编译选项：`-fstack-protector-all`, `-Wl,-z,now`, `-Wl,-z,relro`, `-Wl,-z,noexecstack`

### 5.3 CI 流水线

基于 Docker（`sh-harbor.mthreads.com/qa/linux-ddk:v20`）：

| 构建类型 | 产物 |
|---------|------|
| Release | `musaRuntime.tar.gz` |
| Debug+ASAN | `musaRuntime_debug_asan.tar.gz` |
| clang-tidy | — |
| stub scan | — |

**测试覆盖硬件：**
- **qy2**（mp_22）— 基准测试 + CTS smoke
- **ph1**（mp_31）— CTS Part1~4 + mtrace + ASAN + 性能测试

---

## 六、GPU 架构支持

### 6.1 架构分级扩展指令

| 头文件 | 架构 | 代号 |
|--------|------|------|
| `mp_ext_20_*` | mp_20 | qy1 |
| `mp_ext_30_*` | mp_30 | ph1 |
| `mp_ext_32_*` | mp_32 | ph1s |
| `mp_ext_35_*` | mp_35 | — |
| `mp_ext_60_*` | mp_60 | — |

**主目标架构：mp_31 (ph1)**，CI 全量测试在此架构上运行。

### 6.2 设备属性查询

通过 `muDeviceGetAttribute` 驱动调用获取硬件属性，架构感知主要在设备端头文件（条件编译）实现。

---

## 七、测试体系

### 7.1 单元测试

25 个 Google Test 模块目录，按功能域划分：

```
Device_Management, Memory_Management, Stream_Management,
Event_Management, Graph_Management, Peer_Device_Memory_Access,
Occupancy_Management, Texture_Object_Management,
Texture_Reference_Management, Surface_Object_Management,
Graphics_Interoperability, Profiler_Management,
External_Management, Version_Management, Error_Management,
Notification_Management, Thread_Block_Cluster, ...
```

使用 Google Test 1.17.0（从内部仓库获取）。

### 7.2 CTS 测试

通过 `musa_cts/` 仓库覆盖 smoke 全量测试（Part1~4），在真实 GPU 硬件上运行。

---

## 八、关键设计模式

### 8.1 自动生成的 API 包装器

- **生成的文件：** `musa_wrappers_generated.cpp`, `generated_musa_meta.h`, 等
- **规则：** 不手动编辑——由构建工具链生成
- **作用：** 为每个公共 `musa*` API 生成包装器，自动插入 `ApiInvocationGuard`

### 8.2 错误传播模式

```c
MUresult driverResult = g_ExportTable.GetDriverTable().memory->muMemAlloc_v2(...);
musaError_t rtError = ToMusaError(driverResult);  // MUresult → musaError_t
guard.SetReturnValue(rtError);                     // ApiInvocationGuard 记录
return rtError;                                    // 传递给用户
```

运行时定义了 **120+ 种错误码**，覆盖 CUDA 标准错误以及 MThreads 自定义错误（如 `musaErrorTmeIllegalParameters`、`musaErrorTcePairNotSupport` 等）。

### 8.3 日志系统

通过 `MUSA_LOG` 环境变量控制（位掩码），支持 11 个日志域：

| 标志 | 位 | 含义 |
|------|----|------|
| `LOG_ERR` | 0x01 | 错误信息 |
| `LOG_API` | 0x02 | API 追踪（参数+返回值+耗时） |
| `LOG_SYNC` | 0x04 | 同步操作 |
| `LOG_MEM` | 0x08 | 内存分配/释放 |
| `LOG_COPY` | 0x10 | 内存拷贝 |
| `LOG_INIT` | 0x80 | 初始化流程 |
| 其他 | ... | Queue, Command, Cache, Invocation |

`ApiInvocationGuard` 的构造与析构自动打印 `<<rt-api ...(args)` 和 `>> ret=X(...) +N us` 格式日志。

---

## 九、评估与总结

### 优势

1. **CUDA 兼容度极高：** 336+ 个公共 API 导出，覆盖 CUDA Runtime 12.x 几乎所有功能域（Graph、MemPool、Async Alloc、External Semaphore、Library API、Conditional Handle 等高级功能）
2. **架构设计清晰：** 导出表模式优雅解耦运行时与驱动边界，支持独立升级部署
3. **API 追踪基础设施完善：** `ApiInvocationGuard` 提供统一的日志、计时、错误回栈、MUPTI 回调注入
4. **初始化线程安全：** 原子状态机 + `thread_local` 重入保护，支持递归/嵌套调用
5. **测试覆盖广泛：** 25 个 UT 模块 + CTS Part1~4 在 qy2/ph1 硬件上持续验证
6. **跨平台：** Linux 主力 + Windows 支持，x86_64 + aarch64 交叉编译
7. **安全措施到位：** 符号可见性控制、静态库符号加密、ASAN/TSAN、编译安全标志

### 潜在风险 / 技术债务

1. **代码组织扁平：** `src/` 下 30+ 个 .cpp 同级堆放，随着功能增长将难以导航
2. **代码生成工具链不透明：** `musa_wrappers_generated.cpp` 等自动生成文件缺乏显式的生成脚本引用，增加调试难度
3. **导出表 ABI 稳定性：** 数百个函数指针顺序必须与驱动侧严格同步，缺少编译期 ABI 兼容性检查
4. **部分 API 为桩：** 存在 `stubs/` 和 `stub_scan.py` 专门扫描未实现 API
5. **`internal.h` 过重：** ~1300 行，集中承载所有核心基础设施（ExportTableManager、ProgramState、ApiInvocationGuard、错误码映射），维护难度较高

### 总体评估

MUSA-Runtime 是一个**工程成熟度较高**的 CUDA Runtime 兼容实现。其导出表驱动的运行时-驱动分离架构、`ApiInvocationGuard` 统一追踪框架、原子安全初始化等设计均达到生产级质量标准。整体而言，代码质量在国产 GPU 软件栈中属于第一梯队，体现出清晰的工程纪律和良好的模块化思维。
