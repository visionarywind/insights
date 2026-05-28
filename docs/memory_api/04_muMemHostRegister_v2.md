# muMemHostRegister_v2 — 用户指针注册为页锁定内存

## 功能

将用户已分配的 host 内存注册为页锁定内存，使 GPU 可以通过 DMA 直接访问。**不分配新内存**，而是锁定用户已有的内存页。对应 CUDA `cudaHostRegister`。

## 完整调用链

```
用户代码: muMemHostRegister_v2(host_ptr, 4096, MU_MEMHOSTREGISTER_DEVICEMAP)
  │
  ├─ 1. mu_wrappers_generated.cpp — MUPTI 插桩
  │
  ├─ 2. mu_memory.cpp:1693 — muapiMemHostRegister_v2
  │    ├─ InitPlatform()
  │    ├─ 校验: p==nullptr? bytesize==0? Flags 超出 API_MASK?
  │    ├─ TlsCtxTop() → Context
  │    ├─ CreateInfo:
  │    │   type = memoryTypeRegisteredPinnedHost
  │    │   registeredPinnedHost = {ptr=host_ptr, size=4096, flags=DEVICEMAP|OVERLAP_CHECK}
  │    └─ pContext->CreateMemory(&pMemory, createInfo)
  │
  ├─ 3. context.cpp:915 — Context::CreateMemory
  │    ├─ new Memory(this)
  │    ├─ pMemory->Init → memoryTypeRegisteredPinnedHost → PinnedHostRegister
  │    └─ Context::CreateMemory 中是:
  │        memoryTypeRegisteredPinnedHost 的 peerAccessible map 不经过此路径
  │        (peer access 在 PinnedHostRegister 内部处理)
  │
  ├─ 4. memory.cpp:611 — Memory::PinnedHostRegister
  │    ├─ 设备能力检查: hostRegisterSupported? canMapHostMemory? supportReadonlyHostRegister?
  │    ├─ 默认 +DEVICEMAP (如果设备支持)
  │    ├─ 重叠检查: IterateMemories 遍历该地址范围, 看是否已有其他 registered memory
  │    │   └─ 如果已有 → MUSA_ERROR_HOST_MEMORY_ALREADY_REGISTERED
  │    ├─ Hal::MemoryCreateInfo
  │    │   │
  │    │   ├─ [IOMEMORY flag]: → 创建 HAL extern memory view
  │    │   │   type = View, viewType = External
  │    │   │   external.type = mmio (MMIO 物理地址映射)
  │    │   │   external.handle = reinterpret_cast<MUdeviceptr>(ptr)
  │    │   │
  │    │   └─ [普通]: → 创建 HAL locked memory view
  │    │       type = View, viewType = Locked
  │    │       locked.pHost = ptr              // 用户指针
  │    │       locked.size = size
  │    │       locked.isDeviceMapped = (flags & DEVICEMAP)
  │    │       locked.isDeviceReadOnly = (flags & READ_ONLY)
  │    │       locked.isPeerAccessible = (flags & PORTABLE) || unifiedAddressing
  │    │
  │    └─ Device->CreateMemory(createInfo, &m_pHalMemory)
  │
  └─ 5. [HAL] M3d::Memory::Init(memoryViewTypeLocked/External)
       │
       ├─ [Locked] InitLockedMemory:
       │    ├─ m3dDevice->LockGpuMemory(ptr, size)  // 锁定页表, 不让换页
       │    ├─ m3dDevice->MapGpuMemory(ptr, size)    // 建立 GPU 页表映射
       │    └─ [ReadOnly] → 设置 GPU 只读权限
       │
       └─ [External MMIO] InitExternalMemory:
            ├─ m3dDevice->MapExternalMemory(mmuAddr, size) // 映射 MMIO 区域到 GPU VA
            └─ (不锁定页表, 因为是 MMIO 固定地址)
```

## 时序图

```
应用层              Wrapper            Driver              Context             Memory           HAL/M3D             KMD
  │                  │                  │                   │                  │                │                   │
  │ muHostRegister   │                  │                   │                  │                │                   │
  │─────────────────>│                  │                   │                  │                │                   │
  │                  │                  │                   │                  │                │                   │
  │                  │ muapiHostReg     │                   │                  │                │                   │
  │                  │─────────────────>│                   │                  │                │                   │
  │                  │                  │                   │                  │                │                   │
  │                  │                  │ InitPlatform()    │                  │                │                   │
  │                  │                  │ (已初始化, 快速返回)                  │                │                   │
  │                  │                  │                   │                  │                │                   │
  │                  │                  │ 校验: p? size? flags?                │                │                   │
  │                  │                  │                   │                  │                │                   │
  │                  │                  │ TlsCtxTop()       │                  │                │                   │
  │                  │                  │──────────────────>│                  │                │                   │
  │                  │                  │<── ctx* ──────────│                  │                │                   │
  │                  │                  │                   │                  │                │                   │
  │                  │                  │ CreateMemory      │                  │                │                   │
  │                  │                  │ type=Registered   │                  │                │                   │
  │                  │                  │  PinnedHost       │                  │                │                   │
  │                  │                  │──────────────────>│                  │                │                   │
  │                  │                  │                   │                  │                │                   │
  │                  │                  │                   │ new Memory(this) │                │                   │
  │                  │                  │                   │ Init->PinnedHost │                │                   │
  │                  │                  │                   │ Register        │                │                   │
  │                  │                  │                   │─────────────────>│                │                   │
  │                  │                  │                   │                  │                │                   │
  │                  │                  │                   │                  │ 设备能力检查      │                   │
  │                  │                  │                   │                  │ hostRegister     │                   │
  │                  │                  │                   │                  │  Supported?      │                   │
  │                  │                  │                   │                  │ canMapHostMem?   │                   │
  │                  │                  │                   │                  │ readonlyRegister? │                  │
  │                  │                  │                   │                  │                  │                   │
  │                  │                  │                   │                  │ 默认+DEVICEMAP   │                   │
  │                  │                  │                   │                  │                  │                   │
  │                  │                  │                   │                  │ 重叠检查:         │                   │
  │                  │                  │                   │                  │ IterateMemories  │                   │
  │                  │                  │                   │                  │ (ptr,size)       │                   │
  │                  │                  │                   │                  │ 是否已有注册?     │                   │
  │                  │                  │                   │                  │                  │                   │
  │                  │                  │                   │                  │ ╔════════════════╗│                   │
  │                  │                  │                   │                  │ ║ [IOMEMORY?]    ║│                   │
  │                  │                  │                   │                  │ ║   │            ║│                   │
  │                  │                  │                   │                  │ ║   ├─YES: External│                  │
  │                  │                  │                   │                  │ ║   │  type=mmio   │                   │
  │                  │                  │                   │                  │ ║   │  addr=ptr    │                   │
  │                  │                  │                   │                  │ ║   │             │                   │
  │                  │                  │                   │                  │ ║   └─NO: Locked  │                   │
  │                  │                  │                   │                  │ ║      pHost=ptr  │                   │
  │                  │                  │                   │                  │ ║      size=4096  │                   │
  │                  │                  │                   │                  │ ║      devMap=flg │                   │
  │                  │                  │                   │                  │ ║      readOnly=  │                   │
  │                  │                  │                   │                  │ ║      peerAcc=   │                   │
  │                  │                  │                   │                  │ ╚════════════════╝│                   │
  │                  │                  │                   │                  │                  │                   │
  │                  │                  │                   │                  │ Device::         │                   │
  │                  │                  │                   │                  │ CreateMemory     │                   │
  │                  │                  │                   │                  │ (memoryTypeView)  │                   │
  │                  │                  │                   │                  │─────────────────>│                   │
  │                  │                  │                   │                  │                  │                   │
  │                  │                  │                   │                  │                  │ Init()            │
  │                  │                  │                   │                  │                  │ viewType=Locked   │
  │                  │                  │                   │                  │                  │                   │
  │                  │                  │                   │                  │                  │ m3dDevice->       │
  │                  │                  │                   │                  │                  │ LockGpuMemory()──>│──锁定页表
  │                  │                  │                   │                  │                  │                   │──mlock(ptr,size)
  │                  │                  │                   │                  │                  │<── OK ────────────│
  │                  │                  │                   │                  │                  │                   │
  │                  │                  │                   │                  │                  │ m3dDevice->       │
  │                  │                  │                   │                  │                  │ MapGpuMemory()───>│──建立GPU页表
  │                  │                  │                   │                  │                  │  (GPU VA → host   │──映射到GPU VA
  │                  │                  │                   │                  │                  │   物理地址)       │
  │                  │                  │                   │                  │                  │<── OK ────────────│
  │                  │                  │                   │                  │                  │                   │
  │                  │                  │                   │                  │<── OK ───────────│                   │
  │                  │                  │                   │                  │                  │                   │
  │                  │                  │                   │ AddMemory()     │                  │                   │
  │                  │                  │                   │ TrackMemory()   │                  │                   │
  │                  │                  │                   │                  │                  │                   │
  │                  │                  │<── OK ────────────│                  │                  │                   │
  │                  │<── OK ───────────│                   │                  │                  │                   │
  │<── OK ───────────│                  │                   │                  │                  │                   │
```

## 关键代码路径

### Driver 入口

```cpp
// mu_memory.cpp:1693
MUresult muapiMemHostRegister_v2(void *p, size_t bytesize, unsigned int Flags) {
    // 1. InitPlatform()
    // 2. 校验: p==nullptr? bytesize==0? Flags 只允许 API_MASK
    // 3. TlsCtxTop()
    // 4. CreateInfo:
    //    type = memoryTypeRegisteredPinnedHost
    //    registeredPinnedHost = {ptr=p, size=bytesize, flags=Flags|OVERLAP_CHECK}
    //    注: 自动加上 OVERLAP_CHECK, 检查是否重复注册
    // 5. pContext->CreateMemory(&pMemory, createInfo)
}
```

### PinnedHostRegister 实现

```cpp
// memory.cpp:611
MUresult Memory::PinnedHostRegister(void* ptr, size_t size, unsigned int flags) {
    // 1. 设备能力检查:
    //    hostRegisterSupported? → 必须支持
    //    canMapHostMemory? → DEVICEMAP 要求
    //    supportReadonlyHostRegister? → READ_ONLY 要求
    //
    // 2. 如果设备支持 canMapHostMemory, 自动 +DEVICEMAP
    //
    // 3. 重叠检查:
    //    if (flags & OVERLAP_CHECK):
    //        MemoryTracker::IterateMemories(ptr, size, callback)
    //        callback: 如果发现已有 registeredPinnedHost 在同一区域
    //                  → MUSA_ERROR_HOST_MEMORY_ALREADY_REGISTERED
    //
    // 4. 构造 HAL CreateInfo:
    //    [IOMEMORY]:  (直接 MMIO 地址映射)
    //       type = memoryTypeView
    //       view.type = memoryViewTypeExternal
    //       external.type = mmio
    //       external.handle = ptr  (MMIO 物理地址)
    //
    //    [普通]: (用户已有 host 内存)
    //       type = memoryTypeView
    //       view.type = memoryViewTypeLocked
    //       locked.pHost = ptr
    //       locked.size = size
    //       locked.isDeviceMapped = DEVICEMAP
    //       locked.isDeviceReadOnly = READ_ONLY
    //       locked.isPeerAccessible = PORTABLE || unifiedAddressing
    //
    // 5. Device->CreateMemory(createInfo, &m_pHalMemory)  → HAL/M3D
}
```

### HAL InitLockedMemory (概念)

```cpp
// hal/m3d/memory.cpp :: InitLockedMemory (伪代码)
Result Memory::InitLockedMemory(const MemoryCreateInfo& createInfo) {
    // 1. 验证用户指针是否有效
    m_mappedHost = createInfo.view.locked.pHost;  // 指向用户已有内存
    m_size = createInfo.view.locked.size;

    // 2. 锁定内存页 (使页不可换出)
    m3dDevice->LockGpuMemory(createInfo.view.locked.pHost,
                             createInfo.view.locked.size);

    // 3. 如果需要 device mapping
    if (createInfo.view.locked.isDeviceMapped) {
        // 在 GPU 页表中建立映射: GPU VA → 用户内存物理页
        m3dDevice->MapGpuMemory(createInfo.view.locked.pHost,
                                createInfo.view.locked.size,
                                createInfo.view.locked.isDeviceReadOnly);
    }
    return Result::success;
}
```

## muMemHostRegister vs muMemHostAlloc 对比

| 特性 | muMemHostAlloc | muMemHostRegister |
|------|---------------|-------------------|
| 分配新内存? | 是, 驱动分配 | 否, 使用用户已有内存 |
| Hal 分配类型 | `memoryTypeAlloc(allocType=Host)` | `memoryTypeView(viewType=Locked/External)` |
| KMD 操作 | 分配 + 映射 | 锁定 + 映射 |
| 返回指针 | 驱动分配的页锁定内存 | 用户原始指针 |
| 创建方式 | `PinnedHostAlloc` | `PinnedHostRegister` |
| NUMA 指定 | 支持 `numaId` | 不支持 (用户已分配) |
| 释放 | muMemFreeHost | muMemHostUnregister |
| 重叠检查 | 不需要 (新内存) | 需要 (同一地址不能注册两次) |

## 关键设计要点

1. **不分配新内存**: `muMemHostRegister` 使用用户已有的 `void*`，只做页锁定 + GPU 映射
2. **两套 HAL 路径**: 普通 host 内存走 `memoryViewTypeLocked`，MMIO 物理地址走 `memoryViewTypeExternal(mmiotype)`
3. **页锁定**: 通过 KMD 将用户虚拟地址范围的物理页锁定（mlock），防止换页导致 GPU DMA 到错误的物理地址
4. **GPU 只读支持**: 如果指定 `MU_MEMHOSTREGISTER_READ_ONLY`，GPU 页表设为只读权限，GPU 侧的意外写入会触发 page fault
5. **重叠保护**: 自动检查地址范围是否已被注册，防止同一内存区域被多次映射导致竞争
