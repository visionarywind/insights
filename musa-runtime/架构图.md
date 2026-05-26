# MUSA-Runtime 软件设计架构图

## 一、整体分层架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          用户代码 / AI 框架                                  │
│                    (PyTorch, SGLang, TensorFlow, 直接 MUSA API)             │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │ musaMalloc(), musaLaunchKernel(), ...
                                ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                           MUSA-Runtime (libmusart.so)                         ║
║                                    v5.1.0                                     ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │                   [1] Public API Layer (include/)                     │    ║
║  │  ┌───────────────────────┐  ┌────────────────────────────────────┐   │    ║
║  │  │  musa_runtime_api.h   │  │  musa_runtime.h (C++ wrapper)      │   │    ║
║  │  │  ~430 function decls  │  │  thin convenience layer           │   │    ║
║  │  └───────────────────────┘  └────────────────────────────────────┘   │    ║
║  │                                                                       │    ║
║  │  Device-side headers:                                                │    ║
║  │    device_functions.h  math_functions.h  device_atomic_functions.h   │    ║
║  │    mp_ext_{20,30,32,35,60}_{intrinsics,atomic}.{h,hpp}              │    ║
║  │                                                                       │    ║
║  │  Interop:  musa_gl_interop.h  musa_egl_interop.h  musa_vdpau_interop│    ║
║  │  Advanced: musa_occupancy.h   musa_pipeline.h    musa_awbarrier.h   │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                    │                                        ║
║                                    │ MUSARTAPI 入口                         ║
║                                    ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │              [2] Generated Wrapper Layer                              │    ║
║  │         musa_wrappers_generated.cpp  (12,642 lines, auto-generated)   │    ║
║  │                                                                       │    ║
║  │  每个公共 musa* API 经过统一的 wrapper:                                 │    ║
║  │    ┌──────────────────────┐                                          │    ║
║  │    │  ApiInvocationGuard   │  ← 日志 / 追踪 / MUPTI 回调              │    ║
║  │    ├──────────────────────┤                                          │    ║
║  │    │  musaapi* 内部调用    │  ← 转发到领域实现层                       │    ║
║  │    ├──────────────────────┤                                          │    ║
║  │    │  SetReturnValue()     │  ← 错误码映射 + 异常记录                  │    ║
║  │    └──────────────────────┘                                          │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                    │                                        ║
║                                    │ musaapiGraphLaunch(), ...               ║
║                                    ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │              [3] Internal API Layer (musaapi.h, 341 functions)        │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                    │                                        ║
║                                    ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │              [4] Domain Implementation Layer (src/musa_*.cpp)         │    ║
║  │                                                                       │    ║
║  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │    ║
║  │  │  device   │ │  memory  │ │  stream  │ │  event   │ │   graph   │  │    ║
║  │  │  841 行   │ │  1561行  │ │  319 行  │ │  124 行  │ │  1732 行  │  │    ║
║  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └───────────┘  │    ║
║  │                                                                       │    ║
║  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │    ║
║  │  │  module  │ │  texobj  │ │occupancy │ │   peer   │ │  library  │  │    ║
║  │  │  454 行  │ │  435 行  │ │  111 行  │ │   61 行  │ │   190 行  │  │    ║
║  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └───────────┘  │    ║
║  │                                                                       │    ║
║  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │    ║
║  │  │ external │ │  interop │ │ mempool  │ │  error   │ │  version  │  │    ║
║  │  │  137 行  │ │  158 行  │ │  535 行  │ │   38 行  │ │   26 行   │  │    ║
║  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └───────────┘  │    ║
║  │                                                                       │    ║
║  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                │    ║
║  │  │ profiler │ │  clang   │ │ notific  │ │deprecated│                │    ║
║  │  │   17 行  │ │  111 行  │ │   28 行  │ │   959行  │                │    ║
║  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘                │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                    │                                        ║
║                                    │ g_ExportTable.GetDriverTable().XXX     ║
║                                    ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │              [5] Driver Bridge Layer (ExportTableManager)             │    ║
║  │                                                                       │    ║
║  │  ┌──────────────────────────────────────────────────────────────┐   │    ║
║  │  │  ExportTableManager (internal.h)                              │   │    ║
║  │  │  ┌─────────────────────────────────────────────────────┐     │   │    ║
║  │  │  │  Init():  dlopen("libmusa.so")                       │     │   │    ║
║  │  │  │           解析 muGetExportTable                       │     │   │    ║
║  │  │  │           填充 14 个 driver 子表                      │     │   │    ║
║  │  │  └─────────────────────────────────────────────────────┘     │   │    ║
║  │  │                                                                │   │    ║
║  │  │  Driver Sub-tables (14):                                       │   │    ║
║  │  │  ┌────────┬────────┬────────┬────────┬────────┬────────┐     │   │    ║
║  │  │  │ context│ device │ memory │ stream │ event  │ exec   │     │   │    ║
║  │  │  ├────────┼────────┼────────┼────────┼────────┼────────┤     │   │    ║
║  │  │  │ graph  │ texRef │ surfRef│ module │occupancy│ utils │     │   │    ║
║  │  │  ├────────┼────────┼────────┼────────┼────────┼────────┤     │   │    ║
║  │  │  │vdpau   │  gfx   │  ogl   │  egl   │  prof  │  blah │     │   │    ║
║  │  │  └────────┴────────┴────────┴────────┴────────┴────────┘     │   │    ║
║  │  └──────────────────────────────────────────────────────────────┘   │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │              [6] Internal Infrastructure                               │    ║
║  │                                                                       │    ║
║  │  ┌───────────────┐  ┌──────────────────┐  ┌────────────────────┐     │    ║
║  │  │   TlsData      │  │  ExportTableMgr  │  │   ProgramState     │     │    ║
║  │  │  per-thread    │  │  atomic init     │  │   fat binary管理    │     │    ║
║  │  │  context/last  │  │  reentrant-safe  │  │   func registry    │     │    ║
║  │  │  error/stack   │  │  state machine   │  │                    │     │    ║
║  │  └───────────────┘  └──────────────────┘  └────────────────────┘     │    ║
║  │                                                                       │    ║
║  │  ┌───────────────┐  ┌──────────────────┐  ┌────────────────────┐     │    ║
║  │  │MusaEvent/MStream│  │  ApiInvocation   │  │    mupti/hooks    │     │    ║
║  │  │  事件/流封装    │  │  Guard (日志/追踪)│  │  profiling 回调   │     │    ║
║  │  └───────────────┘  └──────────────────┘  └────────────────────┘     │    ║
║  │                                                                       │    ║
║  │  ┌───────────────┐  ┌──────────────────┐                            │    ║
║  │  │musa_shared_inc │  │   musaapi.h      │                            │    ║
║  │  │runtime↔driver  │  │   341 internal   │                            │    ║
║  │  │  共享类型       │  │   API declarations│                           │    ║
║  │  └───────────────┘  └──────────────────┘                            │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────┐    ║
║  │              [7] Sub-modules                                         │    ║
║  │  stubs/mu_stubs.cpp  (link-only stubs for unimplemented mu* APIs)    │    ║
║  │  tools/musaInfo      (device info query tool)                        │    ║
║  │  unittest/           (25 Google Test modules)                        │    ║
║  │  encrypt_symbol.py   (static lib symbol obfuscation)                 │    ║
║  └─────────────────────────────────────────────────────────────────────┘    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                │
                                │ dlopen / dlsym (运行时动态加载)
                                ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        MUSA Driver (libmusa.so)                               ║
║  560 muapi* functions                                                         ║
║                                                                               ║
║  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌────────┐  ┌──────────────┐    ║
║  │  API    │→ │  CORE    │→ │   HAL     │→ │  M3D   │→ │   KMD ioctl  │    ║
║  │ (入口)  │  │ (逻辑)   │  │ (硬件抽象)│  │ (MUSA) │  │ (gr-kmd)     │    ║
║  └─────────┘  └──────────┘  └───────────┘  └────────┘  └──────────────┘    ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## 二、核心调用链

```
用户调用 musaMalloc(..., &ptr, size)
  │
  ├─[1] musa_wrappers_generated.cpp
  │     ApiInvocationGuard guard("musaMalloc");
  │     status = musaapiMalloc(...);         ← 转发到内部
  │     guard.SetReturnValue(status);
  │
  ├─[2] musa_memory.cpp
  │     musaapiMalloc(...) {
  │       InitPlatformAndDevice(0);           ← 懒初始化
  │       if (status != musaSuccess) return;
  │       status = ToMusaError(
  │         g_ExportTable.GetDriverTable()
  │           .memory->muMemAlloc_v2(...)     ← 驱动调用
  │       );
  │     }
  │
  └─[3] Driver (libmusa.so)
        muMemAlloc_v2(...) → 参数校验 → HAL → M3D → ioctl → KMD
```

---

## 三、ExportTable 架构（14 个驱动子表）

```
ExportTableManager::Init()
│
├── dlopen("libmusa.so" / "libmusa.so.1")
│
├── 入口解析:
│    musaapiGetExportTable → 填充 ExportTable 结构
│
└── 14 个驱动子表 (函数指针表):

    1. context        ── muCtxCreate, muCtxDestroy, muCtxSetCurrent, ...
    2. device         ── muDeviceGetCount, muDeviceGetAttribute, ...
    3. memory         ── muMemAlloc_v2, muMemFree_v2, muMemcpy, ...
    4. stream         ── muStreamCreate, muStreamSynchronize, ...
    5. event          ── muEventCreate, muEventRecord, muEventSynchronize, ...
    6. exec           ── muLaunchKernel, muLaunchCooperativeKernel, ...
    7. graph          ── muGraphCreate, muGraphLaunch, muGraphInstantiate, ...
    8. texRef         ── muTexRefCreate, muTexRefSetAddress, ...
    9. surfRef        ── muSurfRefCreate, muSurfRefSetArray, ...
    10. module        ── muModuleLoadData, muModuleLoadFatBinary, ...
    11. occupancy     ── muOccupancyMaxActiveBlocks, ...
    12. vdpau         ── muGraphicsVDPAURegisterVideoSurface, ...
    13. gfxInterop    ── muGraphicsMapResources, muGraphicsUnmapResources, ...
    14. oglInterop    ── muGraphicsGLRegisterBuffer, ...

此外还有:
    runtime          ── muDriverGetVersion, muGetExportTable, ...
    tools            ── muProfilerStart, muProfilerStop, ...
    injection        ── callback 注入, ...
```

---

## 四、初始化状态机

```
进程启动
  │
  ├── 首次调用 musa* API 时触发 InitPlatformAndDevice(0)
  │
  │   ┌──────────────────────────────────────────────┐
  │   │  InitPlatform()                               │
  │   │  ├─ atomic state: UNINIT → INITIALIZING       │
  │   │  ├─ dlopen("libmusa.so")                      │
  │   │  ├─ dlsym("muGetExportTable")                 │
  │   │  └─ 填充 14 个驱动子表                         │
  │   └──────────────────────────────────────────────┘
  │       │
  │   ┌──────────────────────────────────────────────┐
  │   │  InitDevice(i)                                 │
  │   │  ├─ muInit(0)         ← 驱动硬件初始化          │
  │   │  ├─ muDevicePrimaryCtxRetain → context ready  │
  │   │  └─ 设置当前 device                            │
  │   └──────────────────────────────────────────────┘
  │
  ├── 线程安全: 原子状态机 + per-thread TLS guard
  │
  └── 后续每次 musa* API 调用:
      InitPlatformAndDevice(0) → 检查已初始化 → 直接返回
```

---

## 五、TLS (Thread-Local Storage) 设计

```
每个线程独立的 TlsData:

TlsData {
    ┌──────────────────────┐
    │ ctx_         ← 当前 MUSA Context        │
    │ err_         ← 上次错误码               │
    │ execStack_   ← 执行上下文栈             │
    │ device_      ← 当前 device index       │
    │ defaultStream_ ← per-device 默认流     │
    │ stream_capture_ ← stream capture 状态  │
    └──────────────────────┘
}
```

---

## 六、符号导出控制

```
Release 构建:
  runtime_symbols.ver  (GNU ld 版本脚本)
    ├── 337 个 musa* 公共 API 符号
    ├── runtime 内部符号 (__musaRegisterFatBinary, ...)
    └── 其他符号全部隐藏
    │
    后处理: objcopy + strip 移除调试符号

Debug/UT 构建:
  所有符号可见
  Windows: EXPORT_ALL_SYMBOLS

Static 构建:
  encrypt_symbol.py 加密内部符号 (发布前)
```

---

## 七、测试覆盖

```
unittest/  (25 个 Google Test 模块)
├── Device_Management/
├── Memory_Management/
├── Stream_Management/
├── Event_Management/
├── Graph_Management/
├── Peer_Device_Memory_Access/
├── External_Management/ + External_Graph_Management/
├── Occupancy_Management/
├── Texture_Object_Management/ + Texture_Reference_Management/
├── Surface_Object_Management/
├── Graphics_Interoperability/
├── Profiler_Management/
├── Version_Management/
├── Error_Management/
├── Notification_Management/
├── Peer_Management/
├── Thread_Block_Cluster/
├── Thread_Management/
├── Entry_Management/
├── Internal_Management/
├── Excution_Control/
├── Surface_Object/
├── ...
```

---

## 八、构建产物

```
libmusart.so         (动态库, SOVERSION=5)
libmusart_static.a   (静态库, 符号加密后发布)
musaInfo             (命令行诊断工具, 查询 GPU 信息/驱动版本/API 版本)
```
