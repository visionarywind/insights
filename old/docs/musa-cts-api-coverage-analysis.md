# MUSA CTS 测试用例分类与 API 覆盖分析

> 分析日期: 2026-05-19 | 仓库: /home/shanfeng/musa_cts | 驱动: /home/shanfeng/linux-ddk

---

## 一、测试架构总览

musa_cts 是独立于 linux-ddk 的兼容性测试套件，使用 Google Test + pytest 双层架构。

### 1.1 测试层级

```
musa_cts/
│
├─ API_TEST/                       # API 单元测试 (按 API 命名)
│   ├─ dt_api_test/                # Driver API 测试 (mu* 函数) — 70+ 文件
│   └─ rt_api_test/                # Runtime API 测试 (musa* 函数) — 18+ 模块
│
├─ functional_test/                # 功能/集成测试 (40 子模块)
│   ├─ Memory/                     # 内存拷贝、IPC、Peer 访问
│   ├─ Virtual_Memory_Management/  # VMM: AddressReserve, Create, Map
│   ├─ UserMemoryPool/             # 内存池管理
│   ├─ Stream/                     # Stream/Event 组合测试
│   ├─ Graph/                      # CUDA Graph 测试
│   ├─ Texture/ / Surface/         # 纹理/表面测试
│   └─ ...
│
├─ STRESS_TEST/                    # 压力/稳定性测试
│   ├─ dt_st/                      # Driver API 压力 (单 dt_st 二进制，-n 选择用例)
│   └─ rt_st/                      # Runtime API 压力
│
└─ pytest/                         # Python 测试编排
    └─ test_musa_cts/config/
        ├─ musa_dt_api_test_cfg.csv   # Driver API 测试配置
        ├─ musa_rt_api_test_cfg.csv   # Runtime API 测试配置
        ├─ musa_functional_test_cfg.csv # 功能测试配置
        ├─ musa_dt_st_cfg.csv         # Driver 压力配置
        └─ musa_rt_st_cfg.csv         # Runtime 压力配置
```

### 1.2 测试触发模式

| 类别 | 触发方式 | 测试框架 |
|------|---------|---------|
| BASIC | CSV 驱动 → pytest → 编译后的测试二进制 | GTest (TEST_F) |
| FUNCTIONAL | CSV 驱动 → pytest → 编译后的测试二进制 | GTest + 自定义 |
| MULTI_DEV | CSV 驱动 → pytest（多设备专用） | GTest |
| XORG | 仅 X.org 环境下运行 | GTest |
| STRESS | 独立二进制 dt_st/rt_st → `-n` 参数选择用例 | 自定义 runner |

---

## 二、API 类别 → 测试用例对照

### 2.1 Device Management (设备管理)

| API | dt_api_test 用例 | 功能测试 | 备注 |
|-----|-----------------|---------|------|
| `muInit` | `muInit.cpp` | — | 驱动初始化 |
| `muDeviceGet` | `muDeviceGet.cpp` | — | 获取当前设备 |
| `muDeviceGetCount` | `muDeviceGetCount.cpp` | — | 设备数量 |
| `muDeviceGetName` | `muDeviceGetName.cpp` | — | 设备名称 |
| `muDeviceTotalMem` | `muDeviceTotalMem.cpp`, `_v2.cpp` | — | 总显存 |
| `muDeviceGetAttribute` | `muDeviceGetAttribute.cpp` | — | 设备属性 (100+ 属性) |
| `muDeviceGetProperties` | `muDeviceGetProperties.cpp` | — | 设备属性结构体 |
| `muDeviceComputeCapability` | `muDeviceComputeCapability.cpp` | — | 计算能力 |
| `muDeviceGetUuid` | `muDeviceGetUuid_v2.cpp` | — | 设备 UUID |
| `muDeviceGetP2PAttribute` | `muDeviceGetP2PAttribute.cpp` | `P2P_DirectAccess/` | P2P 属性 |
| `muDevicePrimaryCtxRetain` | `muDevicePrimaryCtxRetain.cpp` | — | 主上下文保留 |
| `muDevicePrimaryCtxRelease` | `muDevicePrimaryCtxRelease.cpp`, `_v2` | — | 主上下文释放 |
| `muDevicePrimaryCtxReset` | `muDevicePrimaryCtxReset.cpp`, `_v2` | — | 主上下文重置 |
| `muDevicePrimaryCtxSetFlags` | `muDevicePrimaryCtxSetFlags.cpp`, `_v2` | — | 主上下文标志 |
| `muDriverGetVersion` | `muDriverGetVersion.cpp` | — | 驱动版本 |

### 2.2 Context Management (上下文管理)

| API | dt_api_test 用例 | 功能测试 |
|-----|-----------------|---------|
| `muCtxCreate` / `_v2` | `Context_Management/` (目录) | — |
| `muCtxDestroy` / `_v2` | `Context_Management/` | — |
| `muCtxGetCurrent` | `Context_Management/` | — |
| `muCtxSetCurrent` | `Context_Management/` | — |
| `muCtxPushCurrent` / `_v2` | `Context_Management/` | — |
| `muCtxPopCurrent` / `_v2` | `Context_Management/` | — |
| `muCtxGetDevice` | `Context_Management/` | — |
| `muCtxGetApiVersion` | `Context_Management/` | — |
| `muCtxGetFlags` | `Context_Management/` | — |
| `muCtxGetLimit` / `muCtxSetLimit` | `Context_Management/` | — |
| `muCtxGetCacheConfig` / `Set` | `Context_Management/` | — |
| `muCtxGetSharedMemConfig` / `Set` | `Context_Management/` | — |
| `muCtxSynchronize` | `Context_Management/` | — |
| `muCtxGetStreamPriorityRange` | `Context_Management/` | — |

### 2.3 Stream Management (流管理)

| API | dt_api_test | 功能测试 |
|-----|------------|---------|
| `muStreamCreate` / `WithPriority` | `Stream_Management/` | `Stream/`, `stream_event/` |
| `muStreamDestroy` / `_v2` | `Stream_Management/` | — |
| `muStreamQuery` | `Stream_Management/` | — |
| `muStreamSynchronize` | `Stream_Management/` | — |
| `muStreamWaitEvent` | `Stream_Management/` | `stream_event/` |
| `muStreamGetFlags` | `Stream_Management/` | — |
| `muStreamGetPriority` | `Stream_Management/` | — |
| `muStreamAddCallback` | `Stream_Management/` | — |

### 2.4 Event Management (事件管理)

| API | dt_api_test |
|-----|------------|
| `muEventCreate` | `Event_Management/` |
| `muEventDestroy` / `_v2` | `Event_Management/` |
| `muEventRecord` | `Event_Management/` |
| `muEventQuery` | `Event_Management/` |
| `muEventSynchronize` | `Event_Management/` |
| `muEventElapsedTime` | `Event_Management/` |

### 2.5 Memory Management — 核心重点

#### 2.5.1 基本分配/释放

| API | dt_api_test | 功能测试 |
|-----|------------|---------|
| `muMemAlloc` / `_v2` | `Memory_Management/` | `Memory_management/` |
| `muMemFree` / `_v2` | `Memory_Management/` | — |
| `muMemAllocHost` / `_v2` | `Memory_Management/` | — |
| `muMemFreeHost` | `Memory_Management/` | — |
| `muMemAllocManaged` | `Memory_Management/` | — |
| `muMemGetInfo` / `_v2` | `Memory_Management/` | — |
| `muMemGetAddressRange` / `_v2` | `Memory_Management/` | — |

#### 2.5.2 Memcpy (15+ 方向)

| API | 功能测试用例 |
|-----|------------|
| `muMemcpyHtoD` / `Async` | `Memory/rt_memcpy_HtoD*.cu`, `Memory/rt_memcpyHtoD*.cu` |
| `muMemcpyDtoH` / `Async` | `Memory/rt_memcpy_DtoH*.cu` |
| `muMemcpyDtoD` / `Async` | `Memory/rt_memcpyDtoD*.cu` |
| `muMemcpyPeer` / `Async` | `Memory/rt_memcpy_peer*.cu` |
| `muMemcpy3D` / `Async` | `Memory/rt_memcpy3D*.cu` |
| `muMemcpyBatchAsync` | `Memory/rt_memcpy_batch*.cu` |
| `muMemcpy2D` / `Async` | `Memory/rt_memcpy2D*.cu` |

#### 2.5.3 Memset

| API | 功能测试用例 |
|-----|------------|
| `muMemsetD8/D16/D32` | `Memory/rt_memset*.cu` |
| `muMemsetD8Async/D16Async/D32Async` | `Memory/rt_memset_async*.cu` |
| `muMemsetD2D8/D2D16/D2D32` | `Memory/rt_memset2D*.cu` |
| `muMemsetD2D8Async` 等 | `Memory/rt_memset2D_async*.cu` |

#### 2.5.4 Virtual Memory Management (VMM) — 🔴 重点关注

| API | dt_api_test | 功能测试 |
|-----|------------|---------|
| `muMemAddressReserve` | `Virtual_Memory_Management/` | `vmm/tests/` (GTest), `Virtual_Memory_Management/` |
| `muMemAddressFree` | `Virtual_Memory_Management/` | `vmm/tests/` |
| `muMemCreate` | `Virtual_Memory_Management/` | `vmm/tests/` |
| `muMemRelease` | `Virtual_Memory_Management/` | — |
| `muMemMap` | `Virtual_Memory_Management/` | `vmm/tests/` |
| `muMemUnmap` | `Virtual_Memory_Management/` | — |
| `muMemSetAccess` | `Virtual_Memory_Management/` | — |
| `muMemGetAccess` | `Virtual_Memory_Management/` | — |
| `muMemGetAllocationGranularity` | `Virtual_Memory_Management/` | `vmm/tests/` |
| `muMemExportToShareableHandle` | `Virtual_Memory_Management/` | — |
| `muMemImportFromShareableHandle` | `Virtual_Memory_Management/` | — |
| `muMemGetAllocationPropertiesFromHandle` | `Virtual_Memory_Management/` | — |
| `muMemRetainAllocationHandle` | `Virtual_Memory_Management/` | — |

**vmm/tests/ 结构** (GTest 框架):
```
functional_test/vmm/tests/
├── vmm_test_base.cpp/h       # 测试基类: SetUp, CheckVMMSupport, 泄漏检测
├── vmm_address_reserve.cpp   # muMemAddressReserve 测试
├── vmm_create.cpp            # muMemCreate 测试
├── vmm_map.cpp               # muMemMap 测试
├── vmm_granularity.cpp       # muMemGetAllocationGranularity 测试
└── ...
```

#### 2.5.5 IPC / Peer Access

| API | dt_api_test | 功能测试 |
|-----|------------|---------|
| `muIpcGetMemHandle` | — | `Memory/rt_memcpy_d2d_multiDev_ipc*.cu` |
| `muIpcOpenMemHandle` | — | `IPC_SynchronizationTest/` |
| `muIpcCloseMemHandle` | — | `Memory/rt_ipc*.cu` |
| `muDeviceCanAccessPeer` | `Peer_Context_Memory_Access/` | `P2P_DirectAccess/` |
| `muCtxEnablePeerAccess` | `Peer_Context_Memory_Access/` | `P2P_DirectAccess/` |
| `muCtxDisablePeerAccess` | `Peer_Context_Memory_Access/` | — |
| `muDeviceGetP2PAttribute` | `Peer_Context_Memory_Access/` | — |

#### 2.5.6 Memory Pool

| API | dt_api_test | 功能测试 |
|-----|------------|---------|
| `muMemPoolCreate` / `Destroy` | `UserMemoryPool/` | `UserMemoryPool/` |
| `muMemPoolTrimTo` | `UserMemoryPool/` | — |
| `muMemPoolSetAttribute` / `GetAttribute` | `UserMemoryPool/` | — |
| `muMemPoolSetAccess` / `GetAccess` | `UserMemoryPool/` | — |
| `muMemPoolExportPointer` / `ImportPointer` | `UserMemoryPool/` | — |
| `muMemAllocFromPoolAsync` | `UserMemoryPool/` | — |
| `muMemAllocAsync` | `Memory_Management/` | — |
| `muMemFreeAsync` | `Memory_Management/` | — |

#### 2.5.7 统一内存 (Unified Memory)

| API | dt_api_test |
|-----|------------|
| `muMemAdvise` / `_v2` | `Unified_Addressing/` |
| `muMemPrefetchAsync` / `_v2` | `Unified_Addressing/` |
| `muMemRangeGetAttribute` / `GetAttributes` | `Unified_Addressing/` |

#### 2.5.8 Memory Transfer / Atomic

| API | dt_api_test |
|-----|------------|
| `muMemoryTransfer` / `Async` | `MemoryTransfer/` |
| `muMemoryAtomicAsync` | `CE_Atomic/` |
| `muMemoryAtomicValueAsync` | `CE_Atomic/` |
| `muMemoryAtomicBatchAsync` (已废弃) | — |

### 2.6 Module Management (模块管理)

| API | dt_api_test |
|-----|------------|
| `muModuleLoad` / `LoadData` | `Module_Management/` |
| `muModuleUnload` | `Module_Management/` |
| `muModuleGetFunction` | `Module_Management/` |
| `muModuleGetGlobal` | `Module_Management/` |

### 2.7 Execution Control (执行控制)

| API | dt_api_test | 功能测试 |
|-----|------------|---------|
| `muLaunchKernel` | `Execution_Control/` | `Execution_Control/` |
| `muFuncGetAttribute` | `Execution_Control/` | — |
| `muFuncSetAttribute` | `Execution_Control/` | — |
| `muFuncSetCacheConfig` | `Execution_Control/` | — |
| `muFuncSetSharedMemConfig` | `Execution_Control/` | — |

### 2.8 Graph Management

| 功能测试 |
|---------|
| `Graph/` — 图创建/执行/更新 |
| `Graphs_Management/` — 图生命周期 |
| `GraphApp/` — 图应用场景 |
| `GraphUserq/` — User Queue 图 |
| `Stream_Capture_Graph/` — Stream Capture 图 |

### 2.9 其他 API 类别

| 类别 | dt_api_test | 功能测试 |
|------|------------|---------|
| Error | `muGetErrorName.cpp`, `muGetErrorString.cpp` | `Error_Management/` |
| Version | `muDriverGetVersion.cpp` | `Version_Management/` |
| Texture | `Texture_Object_Management/`, `Texture_Reference_Management/` | `Texture/`, `Texture_Special_Test/` |
| Surface | `Surface_Object_Management/`, `Surface_Reference_Management/` | `Surface/`, `Surface_Special_Test/`, `SurRef/` |
| Symbol | `Symbol/` | `Memory/rt_symbol*` |
| Green Context | `GreenCtx/` | `GreenCtx/` |
| Graphics Interop | — | `Graphics_Interoperability/`, `OpenGL_Interoperability/` |
| Multi-Device | — | `IPC_SynchronizationTest/`, `P2P_DirectAccess/`, `Memory/*multiDev*` |

### 2.10 不支持 API 清单

| 文件 | 内容 |
|------|------|
| `musa_dt_api_test_not_support_api.cpp` | 列出 **不支持的驱动 API** |
| `musa_rt_api_test_not_support_api.cpp` | 列出 **不支持的运行时 API** |

---

## 三、覆盖率统计

### 3.1 驱动 API 覆盖率 (mu*)

| 类别 | musa.h API 数 | 有单元测试 | 有功能测试 | 缺口 |
|------|:---:|:---:|:---:|------|
| Device | ~30 | ✅ 15+ | P2P 测试 | 部分 v2 变体 |
| Context | ~25 | ✅ 12+ | 少 | GreenCtx, 部分 limit |
| Stream | ~12 | ✅ 8 | ✅ stream_event | — |
| Event | ~8 | ✅ 6 | — | — |
| Memory Alloc/Free | ~15 | ✅ 10+ | ✅ Memory_management | — |
| Memcpy | ~25 | — | ✅ Memory/ (15+ 方向) | BatchAsync |
| Memset | ~18 | — | ✅ Memory/ (12+ 方向) | — |
| **VMM** | ~13 | ✅ 13 | ✅ vmm/tests/ (GTest) | 无缺口 |
| IPC | ~5 | — | ✅ Memory/*multiDev_ipc* | — |
| Peer | ~5 | ✅ | ✅ P2P_DirectAccess | — |
| Memory Pool | ~14 | ✅ | ✅ UserMemoryPool | — |
| Unified Memory | ~6 | ✅ | — | 功能测试少 |
| Module | ~15 | ✅ | — | — |
| Graph | ~30 | — | ✅ 5 子模块 | — |
| Texture/Surface | ~20 | ✅ | ✅ | — |
| GreenCtx | ~5 | ✅ 部分 | ✅ | — |
| **Raytracing (Photon)** | ~20 | — | ❌ 无测试 | 🔴 完全缺失 |
| Profiler | ~10 | — | ❌ 无测试 | 🔴 完全缺失 |
| **VDPAU** | ~5 | — | ❌ 无测试 | 🔴 完全缺失 |
| **OpenGL interop** | ~10 | — | ✅ XORG 测试 | — |

### 3.2 运行时 API 覆盖率 (musa*)

| 类别 | 测试目录 | 覆盖率 |
|------|---------|:---:|
| Device | `rt_api_test/Device_Management/` | 🟡 中等 |
| Stream/Event | `rt_api_test/Stream_Management/`, `Event_Management/` | 🟢 高 |
| Memory | `rt_api_test/Memory_Management/` | 🟢 高 |
| Error/Version | `rt_api_test/Error_Management/`, `Version_Management/` | 🟢 高 |
| Texture/Surface | `rt_api_test/Texture_*/`, `Surface_*/` | 🟢 高 |

---

## 四、关键 API 调用流 (Driver → HAL → KMD)

### 4.1 muMemAlloc 完整调用流

```
CTS 测试: muMemAlloc(&dptr, 4096)
  │
  └─ muapiMemAlloc_v2()                          driver/mu_memory.cpp
       ├─ InitPlatform()
       ├─ MemoryTracker::FindRange 检查重复
       └─ Platform::MemAlloc(ptr, size)
            └─ Context::CreateMemory(createInfo)   musa/core/context.cpp:915
                 └─ new Memory(this)
                      └─ Memory::Init → GeneralAlloc(size, align, flags)
                           └─ Hal::CreateMemory(info, &m_pHalMemory)
                                └─ m3dDevice->CreateGpuMemory()
                                     └─ GpuMemory::Init()
                                          ├─ CE padding (QY2 only: +32B)
                                          └─ AllocateOrPinMemory()
                                               └─ mtgpu_bo_alloc()
                                                    └─ ioctl(BO_ALLOC) → KMD
```

### 4.2 muMemAddressReserve 完整调用流 (8 层)

参见已写入文档: `docs/muMemAddressReserve-call-flow-analysis.md`

### 4.3 muMemsetD8Async 完整调用流

```
CTS 测试: muMemsetD8Async(dptr, 0, 4096, stream)
  │
  └─ muapiMemsetD8Async()                        driver/mu_memory.cpp
       └─ Context::GeneralMemset(ctx, stream, params)  context.cpp:733
            ├─ CreateMemsetNode() → GraphMemsetNode
            └─ stream->CmdMemset(node, wait)           stream.cpp:720
                 └─ new MemsetCommand(stream, node)    memsetCommand.cpp:11
                      ├─ 引擎选择 (memsetPath):
                      │   DEFAULT: CDM → count(CPU)  [仅两类!]
                      │   DMA: 强制 DMA
                      │   CE:  强制 CE
                      └─ Build() → Submit()
                           ├─ CpuExecute() [CPU 路径]  memsetCommand.cpp:282
                           │    └─ pMemory->GetHostPointer()
                           │         └─ m_pHalMemory->Map()
                           │              └─ mmap BO → CPU addr
                           └─ DmaExecute() [DMA 路径]
                                └─ cmdBuffer.CmdFillMemory()
                                     └─ ioctl(JOB_SUBMIT) → KMD → GPU
```

---

## 五、测试缺口与风险

| 优先级 | 缺口 | 影响 |
|:---:|------|------|
| 🔴 | **Raytracing (Photon) 零测试** | 20+ API 无覆盖，回归风险高 |
| 🔴 | **Profiler API 零测试** | mupti 接口无验证 |
| 🟡 | **Unified Memory 功能测试少** | 仅 dt_api_test，无集成场景 |
| 🟡 | **S4000/S5000 平台特异性** | CE padding、引擎可用性差异未全量覆盖 |
| 🟡 | **VMM size=0 边界测试** | 修复后需确认 CTS 已包含对应用例 |
| 🟢 | **Memcpy/Memset 覆盖充分** | 15+ 方向，12+ 宽度均有测试 |

---

## 六、CTS 与驱动代码对应速查

| CTS 测试 | 测试的 API | 驱动实现文件 | 核心函数 |
|---------|-----------|------------|---------|
| `vmm_address_reserve.cpp` | `muMemAddressReserve` | `driver/mu_vmm.cpp:14` | `muapiMemAddressReserve` |
| `Memory_Management/` | `muMemAlloc`, `muMemFree` | `driver/mu_memory.cpp` | `muapiMemAlloc_v2` |
| `Memory/rt_memset*.cu` | `muMemsetD8/16/32` | `driver/mu_memory.cpp` | `muapiMemsetD8_v2` |
| `Memory/rt_memcpy_HtoD*.cu` | `muMemcpyHtoD` | `driver/mu_memory.cpp` | `muapiMemcpyHtoD_v2` |
| `Virtual_Memory_Management/` | `muMemCreate`, `muMemMap` | `driver/mu_vmm.cpp` | `muapiMemCreate` |
| `Peer_Context_Memory_Access/` | `muCtxEnablePeerAccess` | `driver/mu_peer.cpp` | `muapiCtxEnablePeerAccess` |
| `UserMemoryPool/` | `muMemPoolCreate` | `driver/mu_mempool.cpp` | `muapiMemPoolCreate` |
| `Unified_Addressing/` | `muMemAdvise` | `driver/mu_memory.cpp` | `muapiMemAdvise` |
| `Module_Management/` | `muModuleLoad` | `driver/mu_module.cpp` | `muapiModuleLoad` |
| `Graph/`, `Graphs_Management/` | Graph APIs | `driver/mu_graph.cpp` | `muapiGraphCreate` |
