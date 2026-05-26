# muMemHostRegister / muMemHostRegister_v2 — 主机内存注册（源码逐行分析）

> 源码文件：`musa/src/driver/mu_memory.cpp:611-671`，`musa/src/musa/core/memory.cpp:611-671`

## 1. 功能概述

将用户已分配的**主机虚拟地址**注册到 MUSA 运行时，使 GPU 可以通过 DMA 访问这块内存。

与 `muMemHostAlloc` 的本质区别：**用户自己管理内存的分配和释放**，MUSA 只负责"注册"（pin/映射）。

## 2. 两个入口

```cpp
// mu_memory.cpp:611
MUresult muapiMemHostRegister_v2(void *ptr, size_t size,
                                 unsigned int Flags) {
    // flags 完整集: MU_MEMHOSTREGISTER_PORTABLE |
    //              MU_MEMHOSTREGISTER_DEVICEMAP |
    //              MU_MEMHOSTREGISTER_READ_ONLY |
    //              MU_MEMHOSTREGISTER_IOMEMORY |
    //              MU_MEMHOSTREGISTER_OVERLAP_CHECK

// mu_memory.cpp:626
MUresult muapiMemHostRegister(void *ptr, unsigned int size,
                              unsigned int Flags) {
    return muapiMemHostRegister_v2(ptr, static_cast<size_t>(size),
                                   Flags);
    // v1 → v2, 仅类型转换
}
```

## 3. Driver 入口到 Core 层的逐层路径

```
muapiMemHostRegister(p, size, Flags)
  │
  └─ muapiMemHostRegister_v2(p, static_cast<size_t>(size), Flags)
      │
      ├─ InitPlatform()
      ├─ 参数校验
      │   ├─ ptr == nullptr -> MUSA_ERROR_INVALID_VALUE
      │   └─ bytesize == 0  -> MUSA_ERROR_INVALID_VALUE
      │
      ├─ TlsCtxTop()
      │   └─ 当前线程没有 Context -> MUSA_ERROR_INVALID_CONTEXT
      │
      ├─ 构造 MemoryCreateInfo
      │   ├─ type = memoryTypeRegisteredPinnedHost
      │   ├─ registeredPinnedHost.ptr = p
      │   ├─ registeredPinnedHost.size = bytesize
      │   └─ registeredPinnedHost.flags =
      │      Flags | MU_MEMORY_REGISTER_OVERLAP_CHECK
      │
      └─ IContext::CreateMemory(&pMemory, createInfo)
          │
          └─ Context::CreateMemory(&pMemory, createInfo)
              │
              └─ Memory::Init(createInfo)
                  │
                  └─ Memory::PinnedHostRegister(ptr, size, flags)
```

这一层路径中，MUSA 不分配用户主机内存，只把用户提供的地址范围注册为 `memoryTypeRegisteredPinnedHost`。实际 pin 和 GPU 映射在 `PinnedHostRegister` 内部通过 HAL view 创建完成。

## 4. 核心实现源码逐行分析

```cpp
// memory.cpp:611
MUresult Memory::PinnedHostRegister(void* ptr, size_t size,
                                    unsigned int flags)
{
    // ──────────────────────────────────────────
    // Step 1: 能力校验
    // ──────────────────────────────────────────
    auto overlapCheck = [](MUdeviceptr base,
        std::shared_ptr<IMemory> memory) -> MUresult {
        return memory->GetType() == memoryTypeRegisteredPinnedHost ?
            MUSA_ERROR_HOST_MEMORY_ALREADY_REGISTERED :
            MUSA_ERROR_INVALID_VALUE;
    };

    const Hal::DeviceProperties& deviceProperties =
        m_Context->GetParentDevice()->Hal().GetProperties();

    if (!deviceProperties.memoryProperties.hostRegisterSupported) {
        // 设备不支持 host register → 硬错误
        tprintf(LOG_ERR, "device does not support host "
                "memory registration\n");
        status = MUSA_ERROR_NOT_SUPPORTED;
    }

    if ((flags & MU_MEMORY_REGISTER_PORTABLE) &&
        !deviceProperties.memoryProperties.unifiedAddressing) {
        // PORTABLE 需要 UVA 支持
        tprintf(LOG_WARN, "the host memory would not be portable "
                "on all MUSA contexts\n");
    }

    if ((flags & MU_MEMORY_REGISTER_DEVICEMAP) &&
        !deviceProperties.memoryProperties.canMapHostMemory) {
        // 设备不支持映射主机内存 → 硬错误
        tprintf(LOG_ERR, "device cannot access the host memory\n");
        status = MUSA_ERROR_NOT_SUPPORTED;
    } else if ((flags & MU_MEMORY_REGISTER_READ_ONLY) &&
               !deviceProperties.memoryProperties
                   .supportReadonlyHostRegister) {
        // 设备不支持只读注册 → 硬错误
        status = MUSA_ERROR_NOT_SUPPORTED;
    }

    if (deviceProperties.memoryProperties.canMapHostMemory) {
        // 设备支持映射 → 自动追加 DEVICEMAP
        flags |= MU_MEMORY_REGISTER_DEVICEMAP;
    }

    if (MUSA_SUCCESS == status) {
        // ──────────────────────────────────────────
        // Step 2: 重叠检测 (OVERLAP_CHECK)
        // ──────────────────────────────────────────
        if (flags & MU_MEMORY_REGISTER_OVERLAP_CHECK) {
            status = Platform::Get().GetMemoryTracker()
                .IterateMemories(
                    reinterpret_cast<MUdeviceptr>(ptr), size, false,
                    std::move(overlapCheck));
            // 查找全局的 MemoryTracker, 检查 [ptr, ptr+size)
            // 是否与已注册/已管理的内存重叠

            if (status == MUSA_ERROR_HOST_MEMORY_ALREADY_REGISTERED) {
                tprintf(LOG_ERR, "the host va range [%#llx, %#llx) "
                        "is already registered\n",
                        reinterpret_cast<MUdeviceptr>(ptr),
                        reinterpret_cast<MUdeviceptr>(ptr) + size);
            } else if (status == MUSA_ERROR_INVALID_VALUE) {
                tprintf(LOG_ERR, "the host va range [%#llx, %#llx) "
                        "is already managed in MUSA\n",
                        reinterpret_cast<MUdeviceptr>(ptr),
                        reinterpret_cast<MUdeviceptr>(ptr) + size);
            } else {
                status = MUSA_SUCCESS;
            }
        }

        if (status == MUSA_SUCCESS) {
            // ──────────────────────────────────────────
            // Step 3: 构造 MemoryCreateInfo (双分支)
            // ──────────────────────────────────────────
            m_Shape = { size, 1, 1, size };
            m_Flags = flags;

            Hal::MemoryCreateInfo createInfo{};
            createInfo.type = Hal::memoryTypeView;
            //    ↑ 注册内存始终是 View 类型, 不是 Alloc

            if (flags & MU_MEMHOSTREGISTER_IOMEMORY) {
                // ┌─ 分支 A: MMIO 映射 ──────────────────┐
                createInfo.view.type = Hal::memoryViewTypeExternal;
                createInfo.view.external.isDeviceMapped = true;
                createInfo.view.external.isPeerAccessible = true;
                createInfo.view.external.heap =
                    Hal::MemoryHeap::general;
                createInfo.view.external.type =
                    Hal::MemoryExternalHandleType::mmio;
                //  用户指针直接作为 MMIO 物理地址传入
                createInfo.view.external.handle.data.u64Data =
                    reinterpret_cast<MUdeviceptr>(ptr);
                createInfo.view.external.size = size;
                // └─────────────────────────────────────┘
            } else {
                // ┌─ 分支 B: 页锁定 (Locked) ─────────────┐
                createInfo.view.type = Hal::memoryViewTypeLocked;
                createInfo.view.locked.pHost = ptr;
                //   ↑ 用户虚拟地址指针
                createInfo.view.locked.size = size;
                createInfo.view.locked.isDeviceMapped =
                    (flags & MU_MEMORY_REGISTER_DEVICEMAP);
                createInfo.view.locked.isDeviceReadOnly =
                    (flags & MU_MEMORY_REGISTER_READ_ONLY);
                createInfo.view.locked.isPeerAccessible =
                    (flags & MU_MEMORY_REGISTER_PORTABLE) ||
                    deviceProperties.memoryProperties
                        .unifiedAddressing;
                // └─────────────────────────────────────┘
            }

            // ──────────────────────────────────────────
            // Step 4: 调用 HAL 层创建
            // ──────────────────────────────────────────
            status = HalToMuResult(
                m_Context->GetParentDevice()->Hal().CreateMemory(
                    createInfo, &m_pHalMemory));
            // → Hal::Memory::Init()
            //   → InitLockedMemory()   (Locked 分支)
            //   → InitExternalMemory() (MMIO 分支)
        }
    }
    return status;
}
```

## 5. HAL 层: InitLockedMemory 源码

```cpp
// hal/m3d/memory.cpp:590
Result Memory::InitLockedMemory(const MemoryCreateInfo& createInfo)
{
    Result result = Result::success;
    IM3d::Result m3dRes;
    size_t gpuMemObjSize;
    void* pMemObjectMem;
    IM3d::IDevice* m3dDevice = m_Device.GetM3dDevice();

    do {
        // ── 页对齐处理 ──
        DevSize alignment = ::Util::VirtualPageSize();
        DevSize hostPtr = reinterpret_cast<DevSize>(
            createInfo.view.locked.pHost);
        DevSize hostSize = createInfo.view.locked.size;
        DevSize ptrOffset = hostPtr & (alignment - 1);
        //  ↑ 计算页内偏移 (低 12 位)

        hostPtr -= ptrOffset;    // 向下对齐到页边界
        hostSize = ::Util::AlignUp(hostSize + ptrOffset, alignment);
        //  ↑ 大小扩展到整页覆盖
        m_Offset = ptrOffset;
        //  ↑ m_Offset 记录原始指针在页内的偏移

        // ── 创建 PinnedGpuMemory ──
        IM3d::PinnedGpuMemoryCreateInfo pinnedGpuMemoryCreateInfo{};
        pinnedGpuMemoryCreateInfo.pSysMem =
            reinterpret_cast<void*>(hostPtr);
        //   ↑ 传入对齐后的主机物理页起始地址
        pinnedGpuMemoryCreateInfo.size = hostSize;

        // 堆选择: snooping → Cacheable, 否则 → Uswc
        pinnedGpuMemoryCreateInfo.heap =
            m_Device.GetProperties().memoryProperties
                .supportSnoopingHostCache ?
            IM3d::GpuHeap::GpuHeapGartCacheable :
            IM3d::GpuHeap::GpuHeapGartUswc;

        pinnedGpuMemoryCreateInfo.flags.gpuReadOnly =
            createInfo.view.locked.isDeviceReadOnly;
        pinnedGpuMemoryCreateInfo.flags.peerWritable =
            m_Device.GetProperties().memoryProperties
                .canMapHostMemory;
        pinnedGpuMemoryCreateInfo.flags.globalGpuVa =
            m_Device.GetProperties().memoryProperties
                .unifiedAddressing;
        pinnedGpuMemoryCreateInfo.flags.gl2Uncached = 0;

        gpuMemObjSize = m3dDevice->GetPinnedGpuMemorySize(
            pinnedGpuMemoryCreateInfo, &m3dRes);
        //  ↑ 查询 GPU MMU 页表所需大小

        if (m3dRes != IM3d::Result::Success) {
            result = Result::errorInvalidDevice;
            break;
        }

        m_M3dGpuMemory.Reserve(gpuMemObjSize);
        result = M3dToHalResult(
            m3dDevice->CreatePinnedGpuMemory(
                pinnedGpuMemoryCreateInfo,
                m_M3dGpuMemory.GetAlloc(),
                &m_M3dGpuMemory()));
        //  ↑ ioctl: mtgpuPinMemory (pin 用户物理页, 创建 GPU 映射)

        if (result == Result::success) {
            m_Size = createInfo.view.locked.size;    // 原始 size (含偏移)
            m_Props = memoryPropertyPhysical |
                      memoryPropertyVirtual |
                      memoryPropertyHostVisible |
                      memoryPropertyHostMapped |
                      memoryPropertyDeviceVisible;
            m_Props |= createInfo.view.locked.isDeviceReadOnly ?
                memoryPropertyNone :
                memoryPropertyDeviceWriteable;
            m_Props |= createInfo.view.locked.isDeviceMapped ?
                memoryPropertyDeviceMapped :
                memoryPropertyNone;

            m_ViewCapability =
                createInfo.view.locked.isPeerAccessible ?
                memoryViewCapabilityPeerAccessible :
                memoryViewCapabilityNone;
            m_ViewCapability |=
                createInfo.view.locked.isDeviceMapped ?
                memoryViewCapabilityIpcExportable :
                memoryViewCapabilityNone;
            m_ViewCapability |=
                memoryViewCapabilityShareable |
                memoryViewCapabilityExportable;
        }
    } while(0);

    return result;
}
```

## 6. HAL 层: InitExternalMemory (MMIO) 源码

```cpp
// hal/m3d/memory.cpp:744
Result Memory::InitExternalMemory(const MemoryCreateInfo& createInfo)
{
    // ...
    do {
        // ── 根据 handle type 处理 ──
        switch (createInfo.view.external.type) {
        case MemoryExternalHandleType::mmio: {
            // ⚠ 仅 Linux 有效
            DevSize mmioPtr = m3dHandle;      // 用户传入的指针作为 MMIO 地址
            DevSize alignment = ::Util::VirtualPageSize();
            DevSize mmioBase =
                ::Util::AlignDown(mmioPtr, alignment);   // 向下对齐
            DevSize mmioEnd =
                ::Util::AlignUp(mmioPtr +
                    createInfo.view.external.size, alignment);
            DevSize mmioSize = mmioEnd - mmioBase;
            m_Offset = mmioPtr - mmioBase;    // 记录页内偏移

            externalGpuMemoryOpenInfo
                .resourceInfo.hExternalResource = mmioBase;
            externalGpuMemoryOpenInfo
                .resourceInfo.gpuMemSize = mmioSize;
            externalGpuMemoryOpenInfo
                .resourceInfo.handleType =
                    ::M3d::HandleType::Mmio;
            break;
        }
        // ... 其他 handle 类型
        }

        // ── 打开外部共享 GPU 内存 ──
        result = M3dToHalResult(
            m3dDevice->OpenExternalSharedGpuMemory(
                externalGpuMemoryOpenInfo,
                m_M3dGpuMemory.GetAlloc(),
                &gpuMemoryCreateInfo,
                &m_M3dGpuMemory(),
                nullptr, nullptr));

        if (result == Result::success) {
            m_Size = gpuMemoryCreateInfo.size;
            m_Props = memoryPropertyPhysical;
            m_Props |= createInfo.view.external.isDeviceMapped ?
                (memoryPropertyVirtual |
                 memoryPropertyDeviceMapped |
                 memoryPropertyDeviceWriteable) :
                memoryPropertyIsPhysAlloc;
            // ...
        }
    } while(0);
    return result;
}
```

## 7. 与 muMemHostAlloc 的关键区别

```
┌─────────────────────┬──────────────────────┬──────────────────────┐
│      特性           │  muMemHostAlloc      │  muMemHostRegister   │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ 内存分配方          │ MUSA 运行时          │ 用户自行 malloc/mmap │
│ 内存释放方          │ muMemFreeHost        │ 用户自行 free/munmap │
│ 内存类型            │ PinnedHost           │ RegisteredPinnedHost │
│ HAL 类型            │ memoryTypeAlloc      │ memoryTypeView       │
│ GPU 映射            │ 可选 DEVICEMAP       │ 可选 DEVICEMAP       │
│ MMIO 支持          │ ❌                   │ ✅ (IOMEMORY flag)   │
│ 页对齐              │ 由 KMD 处理          │ HAL 层 AlignUp       │
│ Peer Access         │ ❌ 不支持            │ ✅ (PORTABLE flag)   │
│ 内部分配路径        │ LargePage→General    │ 无 (直接 pin 用户页) │
│ 降级机制            │ static heap 降级     │ 无                   │
└─────────────────────┴──────────────────────┴──────────────────────┘
```

## 8. Peer Access (PORTABLE flag)

```
PinnedHostAlloc:
  PORTABLE flag → 仅设置 m_Props, 但 init 中不设置 viewCapability
  ⚠ 实际不支持 peer access

PinnedHostRegister (Locked):
  PORTABLE flag → view.locked.isPeerAccessible = true
  → viewCapability 包含 memoryViewCapabilityPeerAccessible
  → CreateMemory 后 MapToPeers 会建立 peer 映射
```

## 9. 日志验证结果

最小用例 `memory_api_callflow_demo.cpp` 打开 `MUSA_DRIVER_CALLFLOW_DEBUG=1` 后确认入口层级：

```text
muapiMemHostRegister_v2
  -> TlsCtxTop
  -> IContext::CreateMemory
  -> Context::CreateMemory
  -> Memory::Init
  -> Memory::PinnedHostRegister
```

本次用例中传入 flags 为 `0x0`，driver 追加 `MU_MEMORY_REGISTER_OVERLAP_CHECK` 后进入 `CreateMemory` 的 flags 为 `0x100000`。

## 10. 相关源码位置

| 文件 | 行数 | 说明 |
|------|------|------|
| `musa/src/driver/mu_memory.cpp` | 611-671 | Driver 入口 |
| `musa/src/musa/core/memory.cpp` | 611-671 | `PinnedHostRegister` 核心逻辑 |
| `musa/src/hal/m3d/memory.cpp` | 590-644 | `InitLockedMemory` (pin 用户页) |
| `musa/src/hal/m3d/memory.cpp` | 744-836 | `InitExternalMemory` (MMIO 映射) |
| `musa/src/musa/core/context.cpp` | 915-965 | `CreateMemory` 入口 + MapToPeers |
