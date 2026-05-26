# muMemAddressReserve 完整调用流与 S4000/S5000 分叉分析

> 分析日期: 2026-05-18 | 涉及文件: 12 个核心源文件 | 跨 8 层调用栈

---

## 一、调用流全览

`muMemAddressReserve` 从用户 API 调用到最终 KMD ioctl，穿越 **8 个软件层**。下图展示完整路径及 S4000(QY2 iGPU) 与 S5000(非 QY2 dGPU) 在 `size=0` 场景下的分叉点。

```
用户代码
  │  muMemAddressReserve(&ptr, size=0, alignment=0, addr=0, flags=0)
  │
  ▼
┌──────────────────────────────────────────────────────────────────┐
│ [层 0] Generated Wrapper — mu_wrappers_generated.cpp               │
│   muMemAddressReserve() → muapiMemAddressReserve()                 │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ [层 1] Driver API — mu_vmm.cpp:14                                 │
│   ✅ InitPlatform()                                                │
│   🛑 size==0 → MUSA_ERROR_INVALID_VALUE  (已修复，最早拦截点)       │
│   ✅ flags/ptr/alignment 检查                                      │
│   ✅ UNIFIED_ADDRESSING 检查                                       │
│   ✅ IsMultipleOf(0, gran) == true  (0 是任意数的倍数)              │
│   └─ Platform::CreateMemory(ptr, createInfo{type=Virtual,size=0})  │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ [层 2] MUSA Core Platform — musa/core/platform.cpp:355            │
│   type == memoryTypeVirtual ✓                                     │
│   new Memory(nullptr)  ← context 为 null!                         │
│   └─ pMemory->Init(createInfo)                                    │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ [层 3] MUSA Core Memory — musa/core/memory.cpp                    │
│   Memory::Init() → switch → case memoryTypeVirtual (line 412)     │
│   └─ VirtualAlloc(size=0, alignment=0, addr=0, flags=0) (line 804)│
│       构造 Hal::MemoryCreateInfo{ type=memoryTypeVirtual, size=0 } │
│       └─ Platform::Get().Hal().CreateMemory(info, &m_pHalMemory)   │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ [层 4] HAL Platform — hal/m3d/platform.cpp:180                    │
│   type == memoryTypeVirtual → new VirtualMemory(*this)            │
│   └─ pMemory->Init(createInfo) → InitInternal()                   │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ [层 5] VirtualMemory — hal/m3d/virtualMemory.cpp:21               │
│   ✅ supportUnifiedAddressing 检查                                 │
│   ✅ hostPageSize 对齐 (0 对齐通过)                                │
│   ✅ heap.vaSize 检查 (0 < heapSize)                              │
│   构造 GpuMemoryCreateInfo{ size=0, virtualAlloc=1, svmAlloc=1 }  │
│   └─ m3dDevice->CreateGpuMemory(info, alloc, &mem)  (line 73)     │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ [层 5½] M3D 装饰器链 (decorator chain)                             │
│   m3dDevice->CreateGpuMemory() 是 IM3d::IDevice 虚函数              │
│   traceCapture::Device → interfaceLogger::Device → DeviceDecorator │
│   └─ 最终落脚: Device::CreateGpuMemory()  device.cpp:1168          │
│        ConstructGpuMemoryObject() → new GpuMemory                  │
│        └─ pGpuMemory->Init(createInfo, internalInfo)               │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ [层 6] GpuMemory::Init — gpuMemory.cpp:539  ⚡ 唯一分叉点          │
│                                                                    │
│   gpusize localSize = createInfo.size;  // = 0                     │
│                                                                    │
│   ┌────────────────────┬──────────────────────────┐               │
│   │     S4000 (QY2)    │     S5000 (非 QY2)       │               │
│   ├────────────────────┼──────────────────────────┤               │
│   │ enableCeExtraPadding│ enableCeExtraPadding     │               │
│   │   = true           │   = false                │               │
│   │   (familyId==QY2)  │   (familyId!=QY2)        │               │
│   ├────────────────────┼──────────────────────────┤               │
│   │ localSize += 32    │ (跳过)                   │               │
│   │   0 → 32           │ localSize 仍为 0          │               │
│   ├────────────────────┼──────────────────────────┤               │
│   │ Pow2Align(32,      │ Pow2Align(0,             │               │
│   │   262144)          │   262144)                │               │
│   │ = 262144           │ = 0                      │               │
│   └────────────────────┴──────────────────────────┘               │
│                                                                    │
│   m_desc.size = 262144 (S4000)  /  0 (S5000)                      │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ [层 7] AllocateOrPinMemory — mtgpuMemory.cpp:387                  │
│   IsVirtual() == true → 跳过物理 BO 分配                           │
│   IsSvmAlloc() == true → SVM VA 分配                              │
│                                                                    │
│   S4000: AllocateSvmVirtualAddress(size=262144) → ✅ 成功          │
│   S5000: AllocateSvmVirtualAddress(size=0)      → ❌ -EINVAL      │
└──────────────────────────┬───────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│ [层 8] KMD — gr-kmd (内核态)                                       │
│                                                                    │
│   S4000: ioctl(VM_RESERVE, size=262144) → RM 分配 VA → ✅ SUCCESS │
│   S5000: ioctl(VM_RESERVE, size=0)      → size=0 非法 → ❌ EINVAL │
└──────────────────────────────────────────────────────────────────┘
```

---

## 二、核心分叉点详解：CE Padding 机制

### 2.1 CE Padding 是什么

CE (Copy Engine) 在执行拷贝操作时，会在最后一个写入元素的后面 32 字节触发一次 dummy write。为防止越界写入损坏相邻内存分配，QY2 系列 GPU 需要在每次分配时额外增加 32 字节的安全填充。

**JIRA 引用**: SW-23343 — "CE may trigger a dummy write on the following 32B of last written element"

### 2.2 CE Padding 触发条件

`gpuMemory.cpp:643-656`：

```cpp
if (m_pDevice->GetPublicSettings()->enableCeExtraPadding &&   // ← 芯片族决定
    (m_pImage == nullptr) &&                                   // 非图像
    (createInfo.usage == GpuMemoryUsage::Generic) &&           // 通用用途
    (IsExternal() == false) &&                                 // 非外部导入
    (IsPinned() == false) &&                                   // 非固定内存
    (IsPhysical() == false) &&                                 // 非物理分配
    (m_flags.dxvaAlloc == 0))                                  // 非 DXVA
{
    localSize += 32;  // 增加 32 字节安全填充
}
```

**关键发现**: CE padding 条件中**没有** `IsVirtual()` 检查。虚拟内存分配 (`virtualAlloc=1`) 同样会触发 CE padding！这是导致 `size=0` 被意外放大的直接原因。

### 2.3 enableCeExtraPadding 由芯片族决定

`m3d/src/core/device.cpp:4167`：

```cpp
m_publicSettings.enableCeExtraPadding =
    (m_chipProperties.familyId == FAMILY_QY2) ? true : false;
```

| 平台 | familyId | enableCeExtraPadding | CE padding 生效? |
|------|----------|---------------------|-----------------|
| **S4000** | FAMILY_QY2 (0x03) | **true** | ✅ 是 |
| **S5000** | 非 QY2 | **false** | ❌ 否 |

### 2.4 size=0 在两种平台上的数学推演

```
S4000 (QY2):                         S5000 (非 QY2):
─────────────────────────────────    ─────────────────────────────
createInfo.size = 0                  createInfo.size = 0
enableCeExtraPadding = true          enableCeExtraPadding = false
localSize = 0                        localSize = 0
localSize += 32 → 32                 (CE padding 跳过)
allocGranularity = 262144            allocGranularity = 262144
Pow2Align(32, 262144) = 262144      Pow2Align(0, 262144) = 0
m_desc.size = 262144                 m_desc.size = 0
                                     │
KMD 收到 262144 ──► ✅ 成功           KMD 收到 0 ──► ❌ EINVAL
```

---

## 三、M3D 装饰器链：virtualMemory 如何调用到 device.cpp

### 3.1 `m3dDevice` 的真实类型

```
virtualMemory.cpp:67:
  m3dDevice = device->GetM3dDevice()    ← Hal::M3d::Device::GetM3dDevice()
  │                                       返回 m_M3dDevice (device.h:69)
  │
  └── 该指针由 M3D DRM Platform 创建
      m3d/src/core/os/drm/platform.cpp:243
        M3d::Drm::Mthreads::EnumerateDevices(this, ...)
          → 创建 Mthreads::Device 对象
          → 存储于 Platform::m_pDevice[]
          → 被 EnumerateDevices() 返回给 HAL 层
```

### 3.2 装饰器链调用路径

M3D 使用装饰器模式 (Decorator Pattern)，每一层都继承自 `DeviceDecorator`，在 `CreateGpuMemory` 中调用 `m_pNextLayer->CreateGpuMemory(...)`：

```
m3dDevice->CreateGpuMemory(info, alloc, &mem)     ← virtualMemory.cpp:73
  │  IM3d::IDevice 虚函数
  │
  ├─ traceCapture::Device::CreateGpuMemory          traceCaptureDevice.cpp:97
  │     └─ CreateGpuMemoryInternal → m_pNextLayer->CreateGpuMemory(...)
  │
  ├─ interfaceLogger::Device::CreateGpuMemory        interfaceLoggerDevice.cpp:718
  │     └─ Base::CreateGpuMemory → m_pNextLayer->CreateGpuMemory(...)
  │
  ├─ gpuProfiler::Device::CreateGpuMemory (可选)     gpuProfilerDevice.cpp:302
  │     └─ m_pNextLayer->CreateGpuMemory(...)
  │
  ├─ DeviceDecorator::CreateGpuMemory                decorators.cpp:661
  │     │  通用装饰器基类 — 仅修正子对象指针后透传
  │     └─ m_pNextLayer->CreateGpuMemory(nextCreateInfo, ...)  (line 696)
  │
  └─▶ Device::CreateGpuMemory                        device.cpp:1168
         │  最终核心实现 (非装饰器)
         │
         ├─ ConstructGpuMemoryObject(pPlacementAddr, threadSafe)  (line 1205)
         │    → placement new GpuMemory 对象 (复用预分配内存)
         │
         └─ pGpuMemory->Init(createInfo, internalInfo)           (line 1207)
               │
               └─▶ GpuMemory::Init()                            gpuMemory.cpp:539
```

### 3.3 装饰器透明性

所有装饰器层都是**透明代理**，不修改 `CreateGpuMemory` 的语义。它们只做：
- 日志记录 (interfaceLogger)
- 性能跟踪 (gpuProfiler)
- API trace (traceCapture)
- 子对象指针转换 (DeviceDecorator: Image → NextImage, Device → NextDevice)

最终 `GpuMemoryCreateInfo` 原样到达 `Device::CreateGpuMemory`。

---

## 四、AllocateOrPinMemory：虚拟内存路径

`GpuMemory::Init` 完成后进入 `AllocateOrPinMemory` (`mtgpuMemory.cpp:387`)，虚拟内存在此只分配 VA，不分配物理 BO：

```cpp
// mtgpuMemory.cpp:405-464 (简化)

// 1. 对齐粒度重新对齐 size
m_desc.size = Pow2Align(m_desc.size,
    IsVirtual() ? virtAllocGranularity : physAllocGranularity);
// S4000: Pow2Align(262144, 262144) = 262144
// S5000: Pow2Align(0, 262144)      = 0

if (IsPhysical() == false) {          // ✅ 虚拟分配
    if (IsSvmAlloc()) {               // ✅ svmAlloc=1
        // SVM VA 管理器分配虚拟地址
        result = IsSvmVaMgr() ?
            AllocateSvmVirtualAddressWithVaMgr(baseVirtAddr, m_desc.size, alignment) :
            AllocateSvmVirtualAddressWithKernel(baseVirtAddr, m_desc.size, alignment);
        // S4000: 请求 size=262144 → ✅ 成功, 返回 VA
        // S5000: 请求 size=0      → ❌ 失败, 返回 ErrorNotMappable
    }
}

if (IsVirtual() == false) {           // ❌ 虚拟分配跳过此块
    // ... 物理 BO 分配 (Pinned, External 等)
}
// 虚拟分配直接进入 ReservePrtVaRange (预留 VA 但不提交物理页)
```

**关键**: 虚拟分配只预留 VA 范围，不分配物理内存页。`ReservePrtVaRange` 最终通过 `mtgpu_vm_reserve` → `ioctl(DRM_IOCTL_MTGPU_VM_RESERVE)` 进入 KMD。

---

## 五、根因分析

### 5.1 五层 Why 追溯

```
Layer 0 (直接原因):
  size=0 未被拦截，流入 GpuMemory::Init

  ↓ 为什么 size=0 通过了前置检查？

Layer 1:
  IsMultipleOf(0, granularity) 恒为 true (0 整除任意数)
  算法层面: 没有检查 0 的特异性

  ↓ 为什么 0 最终被 KMD 接受？

Layer 2:
  QY2 的 enableCeExtraPadding 将 0 → 32 → Pow2Align(32, 262144) = 262144
  KMD 收到合法的非零值，正常分配

  ↓ 为什么 CE padding 会影响虚拟内存？

Layer 3:
  CE padding 检查条件 (gpuMemory.cpp:643-651) 未排除 IsVirtual()
  虚拟内存分配同样触发 CE padding

  ↓ 为什么 IsVirtual() 没被排除？

Layer 4 (根因):
  CE padding 的设计初衷是保护物理内存分配免受 CE dummy write 影响
  但虚拟内存分配只是 VA reservation，不存在被 CE 覆盖的物理页
  IsVirtual() 排除条件的遗漏是一个设计疏忽
  在非 QY2 平台上不触发问题（enableCeExtraPadding=false），仅 QY2 暴露
```

### 5.2 根因

| 维度 | 内容 |
|------|------|
| **类别** | 设计疏忽 — CE padding 对虚拟内存过度保护 |
| **本质** | CE padding 为保护物理内存而设计，但错误地对虚拟地址预留也生效。在 QY2 平台上，`enableCeExtraPadding=true` 将 size=0 膨胀为非零值，绕过了应有的 size 合法性检查 |
| **暴露条件** | 仅 QY2 系列 (familyId == FAMILY_QY2) + 虚拟内存分配路径 |

### 5.3 同类风险

| 位置 | 风险描述 |
|------|---------|
| `mtgpuMemory.cpp:474` — pinned memory CE padding | 固定内存也使用 `enableCeExtraPadding`，但已正确限定在 `IsPinned()` 内 |
| `mtgpuMemory.cpp:697` — 同样检查 | 已正确包含 `IsPinned() && (IsSvmAlloc()==false)` 限定 |
| `mtgpuDevice.cpp:1562` — VA assign | 同上，已有合理限定条件 |

---

## 六、修复方案

### 6.1 方案选择

| 方案 | 修复点 | 优缺点 |
|------|--------|--------|
| **A (已采用)** | `mu_vmm.cpp:19-23` — 入口处拦截 `size==0` | ✅ 最早拦截 ✅ 语义正确 ✅ 对所有平台生效 ✅ 对已有路径无影响 |
| B | `gpuMemory.cpp:643` — CE padding 增加 `!IsVirtual()` 条件 | 只修虚拟路径，size=0 的其他路径未保护 |
| C | KMD 侧拦截 | 太晚，且在非 QY2 上本来就会被拦 |

### 6.2 已采用修复 (方案 A)

```cpp
// mu_vmm.cpp:14-23 (修复后)
MUresult muapiMemAddressReserve(MUdeviceptr *ptr, size_t size, ...) {
    MUresult status = InitPlatform();
    if (status == MUSA_SUCCESS) {
        do {
            if (size == 0) {                              // ← 新增
                tprintf(LOG_ERR,                          //
                    "reservation size can not be zero\n"); //
                status = MUSA_ERROR_INVALID_VALUE;         //
                break;                                     //
            }                                              //
            if (flags != 0 || ptr == nullptr || ...) {
                ...
```

**为什么选这个位置**：
- 最早拦截点，在所有后续检查（UVA、granularity、CreateMemory）之前
- 语义正确：size=0 的虚拟地址预留没有意义
- 避免依赖底层 CE padding / KMD / VaMgr 的硬件差异
- 对非 QY2 平台行为无影响（它们已在更下层被 KMD 拦截）

---

## 七、涉及文件索引

| 文件 | 函数/行号 | 作用 |
|------|----------|------|
| `musa/src/driver/mu_vmm.cpp:14` | `muapiMemAddressReserve` | 驱动入口，参数校验 |
| `musa/src/musa/core/platform.cpp:355` | `Platform::CreateMemory` | 创建 Virtual Memory 对象 |
| `musa/src/musa/core/memory.cpp:412,804` | `Memory::Init`, `VirtualAlloc` | 内存类型分发，构造 HAL 信息 |
| `musa/src/hal/m3d/platform.cpp:180` | `Platform::CreateMemory` | HAL 层 VirtualMemory 构造 |
| `musa/src/hal/m3d/virtualMemory.cpp:21` | `VirtualMemory::InitInternal` | VA 预留核心逻辑，构造 M3D 信息 |
| `m3d/src/core/device.cpp:1168` | `Device::CreateGpuMemory` | 创建 GpuMemory 并调用 Init |
| `m3d/src/core/device.cpp:4167` | `enableCeExtraPadding` 设置 | 芯片族决定 CE padding 开关 |
| `m3d/src/core/gpuMemory.cpp:539,643` | `GpuMemory::Init`, CE padding | **分叉点**：size 放大逻辑 |
| `m3d/src/core/os/drm/mthreads/mtgpuMemory.cpp:387` | `AllocateOrPinMemory` | SVM VA 分配，KMD ioctl |
| `m3d/src/core/layers/decorators.cpp:661` | `DeviceDecorator::CreateGpuMemory` | 装饰器链节点 |
| `m3d/src/core/layers/traceCapture/traceCaptureDevice.cpp:97` | trace layer | 装饰器链节点 |
| `m3d/src/core/layers/interfaceLogger/interfaceLoggerDevice.cpp:718` | log layer | 装饰器链节点 |

---

## 八、平台差异速查

| 特性 | S4000 (QY2 iGPU) | S5000 (非 QY2 dGPU) |
|------|-----------------|---------------------|
| familyId | FAMILY_QY2 (0x03) | 非 QY2 |
| enableCeExtraPadding | **true** | **false** |
| size=0 + CE padding → size | 0→32→**262144** | 0→0→**0** |
| size=0 KMD 结果 | ✅ 意外成功 | ❌ EINVAL (-9) |
| DMA 引擎 | None (无) | 有 |
| CE 引擎 | 禁用 (supportCopyEngine=false) | 有 |
| CDM 引擎 | 无 | 有 |
| UVA 支持 | ✅ 支持 | ✅ 支持 |
| SVM 支持 | ✅ 支持 | ✅ 支持 |
