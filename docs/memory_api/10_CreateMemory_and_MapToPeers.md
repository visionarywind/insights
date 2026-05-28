# Context::CreateMemory 与 MapToPeers 深度分析

## 功能概述

`Context::CreateMemory` 是所有内存分配的**最终统一入口**（无论 `muMemAlloc`/`muMemHostAlloc`/`muMemHostRegister` 等最后都调用它）。它完成：
1. 分配 GPU 物理内存（通过 `Memory::Init`）
2. 自动在其他 GPU 上建立 peer 映射（通过 `MapToPeers`）
3. 全局注册 + Debug 初始化

---

## 完整代码分析

### CreateMemory 函数体

```cpp
// context.cpp:915-965
MUresult Context::CreateMemory(IMemory** ppMemory, const MemoryCreateInfo& createInfo) {
    MUresult status = MUSA_SUCCESS;

    // ═══ Step 1: Stream capture 检查 ═══
    {
        ReadLockedAccessor<CriticalBase> ctxCrit(m_CriticalData);
        for (auto streamI = ctxCrit->Streams().begin();
             streamI != ctxCrit->Streams().end(); streamI++) {
            if (((*streamI)->GetCaptureStatus() == MU_STREAM_CAPTURE_STATUS_ACTIVE &&
                 (*streamI)->NeedStrictlyCheck()) ||
                (*streamI)->GetCaptureStatus() == MU_STREAM_CAPTURE_STATUS_INVALIDATED) {
                status = MUSA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
                (*streamI)->InvalidateCaptureStatus();
                break;
            }
        }
    }

    // ═══ Step 2: 创建 Memory 对象并 Init ═══
    std::shared_ptr<IMemory> memory_sp;
    Memory* pMemory = nullptr;
    if (status == MUSA_SUCCESS) {
        memory_sp = std::make_shared<Memory>(this);
        pMemory = static_cast<Memory*>(memory_sp.get());
        status = pMemory->Init(createInfo);
        // Init() 根据 createInfo.type 分发到:
        //   GeneralAlloc / PitchedGeneralAlloc / PinnedHostAlloc
        //   PinnedHostRegister / ManagedAlloc / IpcImportAlloc
        //   ExternalAlloc / PreallocAlloc / VirtualAlloc
    }

    // ═══ Step 3: MapToPeers + AddMemory ═══
    if (status == MUSA_SUCCESS) {
        WriteLockedAccessor<CriticalBase> ctxCrit(m_CriticalData);
        if (pMemory->Hal()->GetCapability() &
            Hal::memoryViewCapabilityPeerAccessible) {
            status = ctxCrit->MapToPeers(pMemory); // ← 在多 GPU 上建立映射
        }
        if (status == MUSA_SUCCESS) {
            ctxCrit->AddMemory(pMemory);            // ← 注册到 context
        }
    }

    // ═══ Step 4: Debug 清零 + 全局注册 ═══
    if (status == MUSA_SUCCESS) {
#if M3D_BUILD_MT_TRACE_CAPTURE
        if (createInfo.type != memoryTypeRegisteredPinnedHost &&
            createInfo.type != memoryTypeIpcImport) {
            std::memset(pMemory->GetHostPointer(), 0, pMemory->GetSize());
        }
#endif
        Platform::Get().GetMemoryTracker().TrackMemory(memory_sp);
        // 全局 MUdeviceptr → IMemory* 区间映射
        if (!pMemory->IsPhysical()) {
            Platform::Get().SetMemorySeqID(pMemory);
            // 分配唯一序列号
        }
        *ppMemory = pMemory;
    }

    return status;
}
```

---

## 四步详细拆解

### Step 1 — Stream Capture 检查

**作用**: 与 CUDA 对齐 — stream capture 期间不允许创建内存。

```cpp
for each stream in context:
    if stream->GetCaptureStatus() == ACTIVE:
        return STREAM_CAPTURE_UNSUPPORTED
        // 因为 graph capture 需要确定性的内存分配，
        // 而 CreateMemory 的底层分配对 capture 不可见
```

当 stream 正在被 capture（记录为 CUDA Graph），且 `NeedStrictlyCheck()` 返回 true，或者 capture 已经失效，则拒绝分配并让该 stream 失效。

### Step 2 — Memory::Init 分发

`Memory::Init` (`memory.cpp:378-425`) 根据 `createInfo.type` 分发到 9 种分配路径：

```
                                   ┌───────────── memoryTypeGeneral
                                   │               GeneralAlloc
                                   │               → MemMgr::Allocate
                                   │               → Hal::CreateMemory
                                   │
                                   ├───────────── memoryTypePitchedGeneral
                                   │               PitchedGeneralAlloc
                                   │               → MemMgr::Allocate (带 pitch)
                                   │
                                   ├───────────── memoryTypePinnedHost
                                   │               PinnedHostAlloc
                                   │               → allocType=Host
                                   │               → InitGeneralHostMemory
                                   │
                                   ├───────────── memoryTypeRegisteredPinnedHost
                                   │               PinnedHostRegister
                                   │               → viewType=Locked/External
                                   │               → LockGpuMemory+MapGpuMemory
                                   │
                                   ├───────────── memoryTypeManaged
                                   │               ManagedAlloc
                                   │               → Host/Device local + SVA
                                   │
        createInfo.type ───────────┤
                                   ├───────────── memoryTypeIpcImport
                                   │               IpcImportAlloc
                                   │               → viewType=External
                                   │               → kernelManagedGlobal handle
                                   │
                                   ├───────────── memoryTypeExternal
                                   │               ExternalAlloc
                                   │               → viewType=External
                                   │               → dmaBuf/fabric handle
                                   │
                                   ├───────────── memoryTypePrealloc
                                   │               PreallocAlloc
                                   │               → 直接绑定已有 Hal::IMemory
                                   │               (不分配新内存)
                                   │
                                   ├───────────── memoryTypeVirtual
                                   │               VirtualAlloc
                                   │               → memoryTypeVirtual
                                   │               → 只创建 VA 不绑定物理
                                   │
                                   └───────────── default → error
```

每个 `Init` 路径最终调用 `Hal::CreateMemory` 或 `MemMgr::Allocate`，进入 M3D 层。

### Step 3 — MapToPeers

**作用**: 将刚分配的内存自动在其他 GPU device 上建立 peer 映射，使得其他 device 的 kernel 也能直接访问。

#### MapToPeers 函数体

```cpp
// context.cpp:483-558
MUresult Context::CriticalBase::MapToPeers(Memory* memory) {
    MUresult status = MUSA_SUCCESS;

    // ── 3a. 确定 peer 数量 ──
    const int peerCount = [&]() {
        switch(memory->GetType()) {
            case External:
            case IpcImport:
            case RegisteredPinnedHost:
            case PinnedHost:
                return Platform::Get().GetDeviceCount();
                // ↑ Host/IPC 内存: 对所有设备映射
                //   因为这些内存通过 PCIe，任何 device 都能 DMA 访问
            case General:
            case PitchedGeneral:
                return static_cast<int>(m_Peers.size());
                // ↑ Device 内存: 只对已启用 peer access 的设备映射
            case Managed:
                return managedForceDeviceAlloc ? m_Peers.size() : GetDeviceCount();
            default:
                return 0;
        }
    }();

    Device* localDevice = static_cast<Device*>(memory->GetContext()->GetDevice());

    // ── 3b. 遍历每个 peer device ──
    for (int peer = 0; peer < peerCount; peer++) {
        if (peer == localDevice->GetId()) {
            continue;  // 跳过自己
        }

        int mapFlags = PEERMAP_FLAG_INVALID;

        // ── 3c. 确定映射方式 (mapFlags) ──
        switch(memory->GetType()) {
            case General:
            case PitchedGeneral:
                mapFlags = m_Peers[peer];
                // ↑ 用户在 muCtxEnablePeerAccess 时设置的映射方式
                break;

            case PinnedHost:
            case RegisteredPinnedHost:
                mapFlags = PEERMAP_FLAG_PCIEONLY;
                // ↑ Host 内存只能走 PCIe 路径
                break;

            case Managed:
                if (memory->GetFlags() == MU_MEM_ATTACH_HOST &&
                    !peerDevice->GetProperties().concurrentManagedAccess) {
                    mapFlags = PEERMAP_FLAG_INVALID;  // 不支持就不要映射
                } else {
                    mapFlags = m_Peers[peer];  // 或 PCIEONLY
                }
                break;

            case IpcImport:
                mapFlags = memory->GetFlags() - 1;  // flags 值编码了映射方式
                break;

            case External:
                mapFlags = PEERMAP_FLAG_INVALID;  // 外部内存不创建 peer 映射
                break;
        }

        // ── 3d. 调用 HAL 建立 peer 映射 ──
        if (mapFlags != PEERMAP_FLAG_INVALID) {
            Hal::PeerOpenInfo openInfo{};
            openInfo.openType      = Hal::MemoryExternalHandleType::kernelManagedGlobal;
            openInfo.deviceMapped  = true;
            openInfo.deviceMapType = static_cast<Hal::PeerMapType>(mapFlags);

            // ★★★ 核心: 在 peer 上打开内存 ★★★
            status = HalToMuResult(memory->Hal()->OpenPeerMemory(
                &peerDevice->Hal(), openInfo));
            // OpenPeerMemory 流程:
            //   1. ExportExternalHandle → 获取 KMD 全局句柄
            //   2. 在 peer 上 ImportExternalHandle → 创建 peer memory view
            //   3. 在 peer 的 GPU 页表建立映射 (peer VA → 原物理内存)
            //   4. 结果存入 m_pHalMemory 的 peer 映射表

            if (status != MUSA_SUCCESS) break;
        }
    }
    return status;
}
```

#### mapFlags 定义 (`internal_types.h:88-95`)

| 常量 | 值 | 含义 |
|------|-----|------|
| `PEERMAP_FLAG_INVALID` | -1 | 未初始化 / 不建立映射 |
| `PEERMAP_FLAG_DEFAULT` | 0 | MTlink 或 PCIe 路径（自动选择） |
| `PEERMAP_FLAG_UNIDIR_IL` | 1 | MTlink+PCIe 交织（单向拓扑） |
| `PEERMAP_FLAG_BIDIR_IL` | 2 | 双向交织 |
| `PEERMAP_FLAG_HETERO_IL` | 3 | 异构交织 |
| `PEERMAP_FLAG_HETERO_BIDIR_IL` | 4 | 异构双向交织 |
| `PEERMAP_FLAG_HALF_BIDIR_IL` | 5 | 半双向交织 |
| `PEERMAP_FLAG_PCIEONLY` | 6 | 强制走 PCIe（Host 内存专用） |

这些值由 `muCtxEnablePeerAccess` 在调用时根据设备的拓扑关系自动确定，存入 `m_Peers[peer]`。

#### m_Peers 的初始化

```cpp
// context.cpp:73-84 — CriticalBase::Init
MUresult Context::CriticalBase::Init() {
    // ...
    m_Peers.resize(Platform::Get().GetDeviceCount(), PEERMAP_FLAG_INVALID);
    // 初始所有 peer 都是 INVALID
}

// muCtxEnablePeerAccess 时:
// ctxCrit->AddPeer(peerDeviceId, computedMapFlags);
// → m_Peers[peerDeviceId] = computedMapFlags  (如 DEFAULT 或 UNIDIR_IL)
```

### Step 4 — 全局注册 + 清理

```cpp
// Step 4a: 调试用 — 分配后立即清零 (只有 debug build)
std::memset(pMemory->GetHostPointer(), 0, pMemory->GetSize());

// Step 4b: 全局区间映射
Platform::Get().GetMemoryTracker().TrackMemory(memory_sp);
// → 将 MUdeviceptr 范围 (ptr ~ ptr+size) 映射到 shared_ptr<IMemory>
//   供后续 muMemFree/muMemcpy 等 API 通过 dptr 反查 Memory 对象

// Step 4c: 唯一序列号 (用于调试/追踪)
if (!pMemory->IsPhysical()) {
    Platform::Get().SetMemorySeqID(pMemory);
}

// Step 4d: 返回给调用方
*ppMemory = pMemory;
```

---

## 时序图

```
muMemAlloc_v2                           Driver                           Context                           Memory                         HAL/M3D                       Peer Device
  │                                       │                                │                                │                              │                              │
  │ muapiMemAlloc_v2                     │                                │                                │                              │                              │
  │─────────────────────────────────────>│                                │                                │                              │                              │
  │                                       │                                │                                │                              │                              │
  │                                       │ CreateMemory(createInfo)      │                                │                              │                              │
  │                                       │──────────────────────────────>│                                │                              │                              │
  │                                       │                                │                                │                              │                              │
  │                                       │                                │ ┌─ Step 1 ─────────────────────                              │                              │
  │                                       │                                │ │ 遍历所有 stream              │                              │                              │
  │                                       │                                │ │ 检查 CaptureStatus           │                              │                              │
  │                                       │                                │ └──────────────────────────────                              │                              │
  │                                       │                                │                                │                              │                              │
  │                                       │                                │ ┌─ Step 2 ─────────────────────                              │                              │
  │                                       │                                │ │ new Memory(this)              │                              │                              │
  │                                       │                                │────────────── create ──────────>│                              │                              │
  │                                       │                                │                                │                              │                              │
  │                                       │                                │ Init(createInfo)                │                              │                              │
  │                                       │                                │────────────────────────────────>│                              │                              │
  │                                       │                                │                                │                              │                              │
  │                                       │                                │                                │ 根据 type 分发:              │                              │
  │                                       │                                │                                │   General → GeneralAlloc     │                              │
  │                                       │                                │                                │   PinnedHost → PinnedHostAlloc│                              │
  │                                       │                                │                                │   ...                        │                              │
  │                                       │                                │                                │                              │                              │
  │                                       │                                │                                │ [subAllocatable?]            │                              │
  │                                       │                                │                                │   YES: MemMgr::Allocate      │                              │
  │                                       │                                │                                │   NO:  Device::CreateMemory  │                              │
  │                                       │                                │                                │──────────────────────────────>│                              │
  │                                       │                                │                                │                              │ InitGeneralDeviceMemory()     │
  │                                       │                                │                                │                              │ m3dDevice->CreateGpuMemory()  │
  │                                       │                                │                                │                              │──────────────────────────────>│
  │                                       │                                │                                │                              │ ←── KMD alloc ok ─────────────│
  │                                       │                                │                                │<── m_pHalMemory ─────────────│                              │
  │                                       │                                │<── OK ─────────────────────────│                              │                              │
  │                                       │                                │                                │                              │                              │
  │                                       │                                │ ┌─ Step 3 ─────────────────────                              │                              │
  │                                       │                                │ │ MapToPeers(pMemory)          │                              │                              │
  │                                       │                                │────────────────────────────────>│                              │                              │
  │                                       │                                │                                │                              │                              │
  │                                       │                                │ [如果 memoryViewCapability    │                              │                              │
  │                                       │                                │  PeerAccessible]               │                              │                              │
  │                                       │                                │                                │                              │                              │
  │                                       │                                │ 确定 peerCount:                │                              │                              │
  │                                       │                                │   General → m_Peers.size()     │                              │                              │
  │                                       │                                │   Host    → GetDeviceCount()   │                              │                              │
  │                                       │                                │                                │                              │                              │
  │                                       │                                │ for each peer != myself:      │                              │                              │
  │                                       │                                │  ┌─────────── peer 0 ──────────                              │                              │
  │                                       │                                │  │ 确定 mapFlags:               │                              │                              │
  │                                       │                                │  │  General → m_Peers[0]       │                              │                              │
  │                                       │                                │  │  Host    → PCIEONLY          │                              │                              │
  │                                       │                                │  │                             │                              │                              │
  │                                       │                                │  │ OpenPeerMemory(             │                              │                              │
  │                                       │                                │  │   &peerDev->Hal(),          │                              │                              │
  │                                       │                                │  │   {kernelManagedGlobal,     │                              │                              │
  │                                       │                                │  │    deviceMapped=true,       │                              │                              │
  │                                       │                                │  │    deviceMapType=flags})    │                              │                              │
  │                                       │                                │  │─────────────────────────────>│                              │                              │
  │                                       │                                │  │                             │ ExportExternalHandle          │                              │
  │                                       │                                │  │                             │──────────────────────────────>│                              │
  │                                       │                                │  │                             │ ←── KMD handle ──────────────│                              │
  │                                       │                                │  │                             │                              │                              │
  │                                       │                                │  │                             │ OpenPeerMemory               │                              │
  │                                       │                                │  │                             │ (在 peer 上建立映射)          │                              │
  │                                       │                                │  │                             │──────────────────────────────>│──────────────────────────────>│
  │                                       │                                │  │                             │                              │ ImportExternalHandle          │
  │                                       │                                │  │                             │                              │ CreatePeerMemoryView          │
  │                                       │                                │  │                             │                              │ MapGpuPageTable               │
  │                                       │                                │  │                             │                              │ ←── OK ──────────────────────│
  │                                       │                                │  │<── OK ──────────────────────│<── OK ───────────────────────│                              │
  │                                       │                                │  └──────────────────────────────                              │                              │
  │                                       │                                │                                │                              │                              │
  │                                       │                                │  for each peer != myself:      │                              │                              │
  │                                       │                                │  ┌─────────── peer 1 ──────────                              │                              │
  │                                       │                                │  │ ... 同上 ...                 │                              │                              │
  │                                       │                                │  └──────────────────────────────                              │                              │
  │                                       │                                │                                │                              │                              │
  │                                       │                                │ AddMemory(pMemory)              │                              │                              │
  │                                       │                                │ (m_Memories.insert)            │                              │                              │
  │                                       │                                │                                │                              │                              │
  │                                       │                                │ ┌─ Step 4 ─────────────────────                              │                              │
  │                                       │                                │ │ TrackMemory (全局注册)        │                              │                              │
  │                                       │                                │ │ SetMemorySeqID                │                              │                              │
  │                                       │                                │ └──────────────────────────────                              │                              │
  │                                       │                                │                                │                              │                              │
  │                                       │<── pMemory* + dptr ───────────│                                │                              │                              │
  │<── OK ────────────────────────────────│                                │                                │                              │                              │
```

---

## 完整示例：双 GPU 场景

```
系统配置: GPU 0 和 GPU 1 (已启用双向 peer access)

muCtxEnablePeerAccess(peerCtx, MU_BIDIRECTIONAL_PEER);
  → ctxCrit->AddPeer(1, PEERMAP_FLAG_DEFAULT);

muMemAlloc(&dptr, 4096);  // 在 GPU 0 上分配
  → Context::CreateMemory(type=General)
    → Memory::Init → GeneralAlloc → KMD 在 GPU 0 上分配 4KB 显存
    → MapToPeers:
        peerCount = 1 (m_Peers.size())
        peer[0] = GPU 1, mapFlags = PEERMAP_FLAG_DEFAULT
        OpenPeerMemory(&dev1->Hal(), {default, deviceMapped=true})
          → GPU 0: ExportExternalHandle → 获取 KMD 全局句柄
          → GPU 1: ImportExternalHandle → 建立 GPU 1 VA → GPU 0 物理内存映射
    → TrackMemory(dptr → Memory*)
    → return dptr

此时 GPU 1 的 kernel 可以直接读写 dptr（走 MTlink 或 PCIe）
```

## 关键设计要点

1. **自动 peer 映射**: 分配后自动完成 peer 映射，用户无需每次分配后手动操作
2. **按类型区分映射策略**:
   - Host 内存 → 强制 PCIEONLY（host 内存只有 PCIe 路径）
   - Device 内存 → 使用 `muCtxEnablePeerAccess` 时确定的映射方式
   - IPC 导入 → 使用 flags 编码的映射方式
   - External → 不建立 peer 映射
3. **m_Peers 数组**: 以 device ID 为索引，值为映射方式标志，初始全部为 `INVALID`，`muCtxEnablePeerAccess` 时设置
4. **shared_ptr 生命周期**: `CreateMemory` 用 `shared_ptr` 管理 `Memory` 对象，因为 `MemoryTracker`、`m_Memories` 等多处持有引用，确保引用归零时才析构
5. **HAL OpenPeerMemory 三步骤**: Export handle → Import on peer → Map page table
