# MUSA 深化架构分析 — 源码级追踪

> 审计日期: 2026-05-19 | 已验证代码版本 | 修正文档不准确项 6 处

---

## 一、文档审计 — 发现的不准确项

| # | 文档 | 声称 | 实际代码 | 严重程度 |
|---|------|------|---------|---------|
| 1 | AGENTS.md:25 | 入口文件 `mu_api.cpp` | **文件不存在**。实际入口为 `mu_wrappers_generated.cpp`, `mu_entry.cpp`, `internal.cpp` | 🔴 高 |
| 2 | AGENTS.md:72 | "Engine selection: CDM → CE → TDM → DMA → CPU" | 实际默认只执行 CDM→CPU。其余路径需显式 `memsetPath` 配置 | 🔴 高 |
| 3 | musa/AGENTS.md:31 | `memsetCommand.cpp:292` → CpuExecute | CpuExecute 实际在 **line 282**，偏移 10 行 | 🟡 低 |
| 4 | AGENTS.md:108 | "muFuncGetParamCount undeclared" | API 已从源码**完全移除**，不再存在 | 🟡 中 |
| 5 | AGENTS.md:57 | "make musaCore" | 实际为 cmake 生成 Makefile，非直接 make | 🟢 风格 |
| 6 | ARCHITECTURE.md §4.3 | muFuncGetParamCount 实现层存在但声明层缺失 | 已完全移除，该分析章节已过时 | 🔴 高 |

---

## 二、Platform 单例与初始化链

### 2.1 单例模式 — Meyer's Singleton

```cpp
// musa/core/platform.cpp:12
Platform& Platform::Get() {
    static Platform platform;    // C++11 线程安全懒初始化
    return platform;
}
```

**初始化触发**: 任何调用 `Platform::Get()` 的 API（`muMemAlloc`, `muMemAddressReserve` 等）都会触发首次构造。构造函数（line 17-25）仅初始化成员变量为默认值，**不执行 HAL 初始化**。

### 2.2 InitPlatform() — 延迟 HAL 初始化

```cpp
// musa/core/platform.cpp (Platform::Init 内部)
// 首次调用时:
//   1. Hal::CreatePlatform() → 创建 HAL 平台对象
//   2. m_pHalPlatform->Init() → 枚举 GPU 设备、初始化 M3D 平台
//   3. 初始化设备列表 (m_Devices)
//   4. 设置 MemoryProperties (supportUnifiedAddressing, 等)
```

所有 `muapi*` 入口函数的第一行都是 `InitPlatform()`（例如 `mu_vmm.cpp:15`），确保 HAL 在首次 API 调用时完成初始化。

### 2.3 Platform 初始化顺序

```
muapiMemAlloc() / muapiMemAddressReserve() / ...
  └─ InitPlatform()
       └─ Platform::Get().Init()
            ├─ 加载 HAL 动态库 (libhalM3d, 静态链接)
            ├─ Hal::CreatePlatform() → hal/m3d/platform.cpp:113
            │    └─ IM3d::CreatePlatform(m3dPlatformCreateInfo)
            │         ├─ enableSvmMode = 1 (硬编码)
            │         └─ enableUvaMode = 1 (硬编码)
            ├─ m_pHalPlatform->Init() → 枚举设备
            │    └─ m_M3dPlatform->EnumerateDevices(count, devices)
            │         └─ M3d::Drm::Mthreads::EnumerateDevices()
            │              ├─ 打开 /dev/dri/cardX
            │              ├─ 查询 PCI 信息
            │              └─ 创建 Mthreads::Device 对象
            └─ 构建设备列表 (Hal::M3d::Device[])
                 └─ Device::Init()
                      ├─ CommitSettingsAndInit() → 读取 KMD 设备属性
                      ├─ 设置 Capabilities (引擎数量、对齐粒度...)
                      └─ 计算 unifiedAddressing, canMapPeerMemory 等
```

---

## 三、MemoryTracker — 全局指针查找

### 3.1 数据结构

```cpp
// musa/core/memoryTracker.cpp
class MemoryTracker {
    std::map<MemoryRange, std::shared_ptr<IMemory>> m_TrackerClient;       // device ptr → memory
    std::map<void*, std::list<std::shared_ptr<IMemory>>> m_TrackerHandleClient; // handle → memory
    std::mutex m_Lock;                                                     // 线程安全
    std::shared_ptr<IMemory> m_Null;                                       // 哨兵空指针
};
```

### 3.2 FindRange — 所有内存 API 的第一步

```cpp
// memoryTracker.cpp:19
std::shared_ptr<IMemory>* MemoryTracker::FindRange(MUdeviceptr ptr, size_t* offset, uint32_t property) {
    std::lock_guard<std::mutex> lg(m_Lock);
    auto iter = m_TrackerClient.find(MemoryRange(ptr, 1));  // 区间树查找
    if (iter != m_TrackerClient.end()) {
        memory = &(iter->second);
        if (offset) *offset = ptr - iter->first.m_BasePointer;
    }
}
```

**调用场景**: 任何需要 `IMemory*` 的 API（`muMemFree`, `muMemcpy`, `muMemset` 等）第一步都调 `Platform::Get().GetMemoryTracker().FindRange(ptr)`。

### 3.3 TrackMemory — 分配时注册

```cpp
// memoryTracker.cpp:48
void MemoryTracker::TrackRange(const std::shared_ptr<IMemory>& memory) {
    m_TrackerClient.insert({MemoryRange(ptr, size), memory});  // O(log n) 插入
    if (memory->GetType() == memoryTypeRegisteredPinnedHost) {
        m_TrackerClient.insert({MemoryRange(hostVa, size), memory});  // 双重注册
    }
}
```

---

## 四、引擎选择机制 — 完整源码分析

### 4.1 Device::Engine 枚举

```cpp
// hal/m3d/device.h (推测位置)
enum Engine { cdm = 0, ce = 1, tdm = 2, dma = 3, ..., count = 7 };
// count = 7 → CPU fallback (不是真实硬件引擎)
```

### 4.2 MemsetCommand 构造函数 — 引擎选择

```cpp
// memsetCommand.cpp:25-62
MemsetCommand::MemsetCommand(...) {
    int memsetPath = Platform::Get().GetSettings().memsetPath;
    switch (memsetPath) {
        case SET_EXECUTOR_DEFAULT:         // ← 默认路径
            if (m_ParentStream->GetHalQueue(Device::Engine::cdm)) {
                m_Engine = Device::Engine::cdm;     // CDM 可用 → 选 CDM
                m_SupportMerge = true;
            } else {
                m_Engine = Device::Engine::count;   // CDM 不可用 → CPU (无中间回退!)
            }
            break;
        case SET_EXECUTOR_DMA:  m_Engine = Device::Engine::dma;  break;
        case SET_EXECUTOR_TDM:  m_Engine = Device::Engine::tdm;  break;
        case SET_EXECUTOR_CE:   m_Engine = Device::Engine::ce;   break;
        case SET_EXECUTOR_CPU:  m_Engine = Device::Engine::count; break;
        case SET_EXECUTOR_CDM:  m_Engine = Device::Engine::cdm;  break;
        case SET_EXECUTOR_SHADER: m_Engine = Device::Engine::cdm; break;
    }
}
```

**关键发现**: `SET_EXECUTOR_DEFAULT` 路径只有 CDM→CPU 两个选项。**没有** CE/TDM/DMA 中间回退。S4000 (QY2 iGPU) 无 CDM，直接落 CPU → `CpuExecute()` → 需 `GetHostPointer()` → 在虚拟内存场景下返回 null → SIGSEGV。

### 4.3 平台引擎可用性

```cpp
// mtgpuDevice.cpp:162,166 — QY2 iGPU 特殊处理
pIpLevels->dma  = ASICDEV_IS_QY2_IGPU(revisionId) ? DmaIpLevel::None : DmaIpLevel::DmaIp1_0;
pIpLevels->hdma = ASICDEV_IS_QY2_IGPU(revisionId) ? HdmaIpLevel::None : HdmaIpLevel::HdmaIp1;
```

| 引擎 | QY2 iGPU (S4000) | QY2 dGPU | PH1 (S5000) |
|------|:---:|:---:|:---:|
| CDM | ✗ | ✗ | ✓ |
| CE | ✗¹ | ✓ | ✓ |
| TDM | ✓² | ✓ | ✓ |
| DMA | ✗ (IpLevel::None) | ✓ | ✓ |
| CPU | ✓ | ✓ | ✓ |

> ¹ `supportCopyEngine = dmaLevel != None && !integrated` → iGPU 上强制 false  
> ² TDM 在 iGPU 上可用但 memset path 默认不选它

---

## 五、Context 生命周期

### 5.1 线程局部存储 (TLS)

```cpp
// internal.cpp
thread_local Context* tls_pCurrentCtx = nullptr;  // 每线程当前上下文

inline Context* TlsCtxTop() { return tls_pCurrentCtx; }
```

所有 API 调用通过 `TlsCtxTop()` 获取当前线程的上下文，无上下文则报错。

### 5.2 命令执行流

```
API 调用 (mu_memory.cpp)
  └─ Context::GeneralMemset(ctx, stream, params)         context.cpp:733
       └─ Context::CreateMemsetNode()
            └─ GraphMemsetNode::Init()
       └─ stream->CmdMemset(node, wait)                   stream.cpp:720
            └─ new MemsetCommand(stream, node)            memsetCommand.cpp:11
                 └─ 引擎选择 (见 §4.2)
            └─ ResolveDependencyAndQueueCommand(command)   context.cpp:1859
                 └─ 将 command 插入 stream 的 m_CommandList
                      └─ [Submit 线程] Command::Build()
                           ├─ CmdBufferBegin
                           ├─ 编码命令 (CpuExecute / DmaExecute / ...)
                           └─ CmdBufferEnd
                      └─ Command::Submit()
                           └─ Hal::IQueue::Submit(submitInfo)
                                └─ [DRM ioctl] → KMD → GPU
```

### 5.3 同步 vs 异步

```cpp
// mu_memory.cpp 典型模式
MUresult muapiMemsetD32_v2(MUdeviceptr dptr, unsigned int val, size_t num) {
    // ...
    status = Context::GeneralMemset(ctx, nullptr, params, true);
    //                                                      ^^^^  wait=true → 同步
    // 异步版本传 wait=false
}
```

---

## 六、导出表 (Export Table) 注册机制

### 6.1 addEntryPoints — API 名称→函数指针映射

```cpp
// driver/internal.cpp:400
addEntryPoints("muMemAddressReserve",
    {{40300, reinterpret_cast<void*>(&muMemAddressReserve), nullptr}});
//    ^^^^^  API 兼容性版本号 (4.3.0 起可用)
```

### 6.2 mu_entry.cpp — 静态访问器数组

```cpp
// driver/mu_entry.cpp
static Driver::MemoryApiAccessors MemoryApiAccessors = {
    .muMemAlloc_v2               = muMemAlloc,
    .muMemFree_v2                = muMemFree,
    .muMemAddressReserve          = muMemAddressReserve,   // ← 必须与 export_table.h 顺序一致
    // ...
};
```

### 6.3 muGetExportTable — 单一入口点

```cpp
// 外部工具/运行时通过此函数获取驱动函数表
MUresult muGetExportTable(const void** ppExportTable, const MUuuid* pTableId) {
    // 根据 UUID 返回对应的 ExportTable (Driver, MUpti, MUgdb, MUasan, Tools...)
}
```

---

## 七、关键源码行号速查 (已验证)

| 功能 | 文件 | 行号 | 备注 |
|------|------|------|------|
| Platform singleton | `musa/core/platform.cpp` | 12 | `static Platform platform` |
| Platform 构造 | `musa/core/platform.cpp` | 17 | 空初始化，延迟 HAL 加载 |
| MemoryTracker::FindRange | `musa/core/memoryTracker.cpp` | 19 | `std::map + mutex` |
| MemoryTracker::TrackRange | `musa/core/memoryTracker.cpp` | 48 | O(log n) 插入 |
| MemsetCommand 构造 | `musa/core/command/memsetCommand.cpp` | 11 | 引擎选择入口 |
| SET_EXECUTOR_DEFAULT | `memsetCommand.cpp` | 29 | CDM → CPU (无中间回退) |
| CpuExecute | `memsetCommand.cpp` | 282 | CPU memset 实现 |
| Context::GeneralMemset | `musa/core/context.cpp` | 733 | memset 统一入口 |
| VirtualAlloc | `musa/core/memory.cpp` | 804 | 虚拟内存分配 |
| Memory::Init 分发 | `musa/core/memory.cpp` | 378 | 9 种内存类型 switch |
| CE padding | `m3d/src/core/gpuMemory.cpp` | 643 | QY2 +32B 逻辑 |
| enableCeExtraPadding 设置 | `m3d/src/core/device.cpp` | 4167 | familyId==QY2 |
| unifiedAddressing 计算 | `hal/m3d/device.cpp` | 863 | 4 条件 AND |
| virtualMemory::InitInternal | `hal/m3d/virtualMemory.cpp` | 21 | VA reservation 核心 |
| AddrReserve 入口 | `driver/mu_vmm.cpp` | 14 | muapiMemAddressReserve |
| export_table.h ADDMEMBER 数 | `musa_shared_include/export_table.h` | — | 570 个 API 指针 |
| musa.h 公开 API 数 | `musa_shared_include/musa.h` | — | 639 个 MUresult 函数 |

---

## 八、CTS VMM 测试基础设施 (musa_cts)

CTS (`/home/shanfeng/musa_cts`) 是独立仓库，通过 GTest 测试 MUSA VMM API 兼容性。

### 8.1 VMMTestBase 架构

```cpp
// functional_test/vmm/tests/vmm_test_base.cpp
class VMMTestBase : public ::testing::Test {
    // 单例静态成员 (懒初始化 + 线程安全):
    static int& DeviceId();               // 设备 ID
    static MUdevice& CUDevice();          // CUDA 设备句柄
    static MUcontext& Context();          // CUDA 上下文
    static DeviceInfo& DeviceInfo();      // {name, total_memory, supports_vmm}
    static bool& VMMSupported();          // VMM 能力标志
    static std::vector<TrackedAlloc>& TrackedAllocations(); // 泄漏检测
    static std::mutex& AllocationMutex(); // 分配器锁
};
```

### 8.2 VMM 能力检测

```cpp
// vmm_test_base.cpp:148
bool VMMTestBase::CheckVMMSupport() {
    int attr_val = 0;
    muDeviceGetAttribute(&attr_val,
        MU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, CUDevice());
    return attr_val == 1;
}
```

**关键**: 使用 `MU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` (值 102) 查询。这与驱动层 `muMemAddressReserve` 中的 `MU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING` 检查是**不同的属性**：
- `VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` (102) — 查询设备是否支持 VMM API
- `UNIFIED_ADDRESSING` — 查询是否支持统一虚拟地址空间

CTS 使用前者做测试跳过，驱动使用后者做 API 入口检查。

### 8.3 测试生命周期

```
SetUpTestSuite()           # 整个测试套件启动
  ├─ InitializeCUDA()      # muInit → muDeviceGet → muCtxCreate
  ├─ CheckVMMSupport()     # 查询 VMM 能力 → 设置跳过标志
  └─ 配置 TestConfig       # enable_strict_validation, max_allocation_size
       │
       ├─ SetUp()           # 每个测试前
       │    └─ 清空 TrackedAllocations
       │
       ├─ [测试体]          # TEST_F(VMMAddressReserveTest, ...)
       │
       ├─ TearDown()        # 每个测试后
       │    └─ CheckMemoryLeaks() → 报告泄漏
       │
TearDownTestSuite()         # 整个套件结束
  ├─ muCtxDestroy()
  └─ 清空 TrackedAllocations
```

---

## 九、构建系统实测

```
# 实际可用的构建命令 (非文档中的 "make musaCore")
cd musa/build
cmake .. -DDDK_2_0=ON -DLIBDRM_PATH=../libdrm-mt -DSHARED_INCLUDE_PATH=../shared_include
make -j32 musaCore          # 静态库
make -j32 musa_dynamic      # libmusa.so
make -j32 m3d               # M3D SDK
```

构建产物链:
```
libutil.a → libm3d.a + libscpc.a → libhalM3d.a → libmusaCore.a → libmusa.so
```
