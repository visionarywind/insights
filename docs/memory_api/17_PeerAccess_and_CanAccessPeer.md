# Peer Access 与 DeviceCanAccessPeer API 深度源码分析

> **文档编号**: 17  
> **相关文件**: `musa/doc/memory_api/README.md`  
> **源码路径**:  
> - `musa/src/driver/mu_peer.cpp` (Driver 层入口)  
> - `musa/src/driver/mu_memory.cpp` (内存侧入口)  
> - `musa/src/musa/core/device.cpp` (Device::CanAccessPeer)  
> - `musa/src/musa/core/context.cpp` (Context::EnablePeer/DisablePeer)  
> - `musa/src/musa/core/device.h` (CanAccessPeer 声明)  

---

## 1. 功能概述

Peer Access 是 MUSA 多 GPU 编程的核心机制，允许一个 GPU 直接访问对等 GPU 的内存。本篇覆盖相关 API：

| API | 功能 | 层级 |
|-----|------|------|
| `muDeviceCanAccessPeer` | 查询两个设备间是否支持 Peer Access | Driver |
| `muCtxEnablePeerAccess` | 启用当前上下文对 Peer Context 的访问 | Driver |
| `muCtxDisablePeerAccess` | 禁用 Peer Access | Driver |
| `Device::CanAccessPeer` | Core 层 Peer 能力查询 | Core |
| `Context::EnablePeer` | Core 层启用 Peer 映射 | Core |
| `Context::DisablePeer` | Core 层禁用 Peer 映射 | Core |

### 1.1 对应 CUDA

| MUSA | CUDA |
|------|------|
| `muDeviceCanAccessPeer` | `cudaDeviceCanAccessPeer` |
| `muCtxEnablePeerAccess` | `cudaDeviceEnablePeerAccess` |
| `muCtxDisablePeerAccess` | `cudaDeviceDisablePeerAccess` |

---

## 2. muDeviceCanAccessPeer

### 2.1 函数签名

```c
MUresult muapiDeviceCanAccessPeer(int* canAccessPeer, MUdevice dev, MUdevice peerDev);
```

### 2.2 完整调用链

```
User Code
  │
  └─ muDeviceCanAccessPeer(&canAccess, dev, peerDev)
        │
        ├─ InitPlatform()
        │
        ├─ GetDeviceCount() > 0 ?
        │     ├─ NO  → MUSA_ERROR_NO_DEVICE
        │     └─ YES ↓
        │
        ├─ canAccessPeer != nullptr ?
        │     ├─ NO  → MUSA_ERROR_INVALID_VALUE
        │     └─ YES ↓
        │
        ├─ GetDevice(dev) → IDevice* pDevice
        ├─ GetDevice(peerDev) → IDevice* pPeerDevice
        │     ├─ 任一为 nullptr → MUSA_ERROR_INVALID_DEVICE
        │     └─ 均有效 ↓
        │
        └─ pDevice->CanAccessPeer(pPeerDevice)
              │
              └─ [Core 层实现, 见下文 2.3]
```

### 2.3 源码逐行分析

**`mu_peer.cpp:13-37`**:

```cpp
MUresult muapiDeviceCanAccessPeer(int* canAccessPeer, MUdevice dev, MUdevice peerDev) {
    MUresult status = InitPlatform();

    if (status == MUSA_SUCCESS) {
        do {
            if (0 == Musa::Platform::Get().GetDeviceCount()) {
                // Step 1: 没有设备
                status = MUSA_ERROR_NO_DEVICE;
                break;
            }
            if (nullptr == canAccessPeer) {
                // Step 2: 输出参数为空
                status = MUSA_ERROR_INVALID_VALUE;
                break;
            }
            Musa::IDevice* pDevice = Musa::Platform::Get().GetDevice(dev);
            Musa::IDevice* pPeerDevice = Musa::Platform::Get().GetDevice(peerDev);
            if (!pDevice || !pPeerDevice) {
                // Step 3: 设备 ID 无效
                status = MUSA_ERROR_INVALID_DEVICE;
                break;
            }
            // Step 4: 委托给 Core 层
            *canAccessPeer = pDevice->CanAccessPeer(pPeerDevice);
        } while(0);
    }

    return status;
}
```

**关键分析**：

| 行 | 分析 |
|----|------|
| `GetDeviceCount() == 0` → NO_DEVICE | 即使查询的设备 ID 合法，无设备加载也返回此错误 |
| `canAccessPeer == nullptr` → INVALID_VALUE | 输出参数必填，不接受 NULL |
| `GetDevice()` 返回 nullptr | 设备越界时返回 `INVALID_DEVICE` |
| `CanAccessPeer()` 返回 int | 0 = false, 非0 = true（布尔值通过 int 传出） |

### 2.4 Device::CanAccessPeer 实现

**`device.cpp:573-580`**:

```cpp
bool Device::CanAccessPeer(IDevice* pPeerDev) const {
    Device* peer = static_cast<Device*>(pPeerDev);
    if (m_SeqID == peer->m_SeqID) {
        // 同一设备不支持 Peer Access
        return false;
    } else {
        // 通过 HAL 查询 P2P 能力
        return Platform::Get().Hal().GetDeviceP2pCapability(
            m_pHalDevice, peer->m_pHalDevice).field.read;
    }
}
```

**关键设计**：

- **自反性排除**: 同一设备 (`m_SeqID == peer->m_SeqID`) 返回 `false`（CUDA 行为一致）
- **HAL 查询**: `Hal().GetDeviceP2pCapability()` 返回包含 `read`/`write` 字段的结构体
- **仅查 read 能力**: MUSA 当前只检查 `.field.read`，即单向读能力
- **平台分发**: 调用到 `Hal::IDevice::GetDeviceP2pCapability()`，Linux 和 Windows 有不同实现

### 2.5 GetDeviceP2pCapability 流程

```
Platform::Get().Hal().GetDeviceP2pCapability(pDev, pPeerDev)
  │
  ├─ Linux (DRM):
  │     ioctl(MTGPU_IOCTL_P2P_CAPABILITY, &params)
  │     → KMD 查询 PCIe P2P 路由表
  │
  └─ Windows (WDDM):
        D3DKMTQueryP2PInfo(&p2pInfo)
        → KMD 通过 DXGK 查询 P2P 支持
```

---

## 3. muCtxEnablePeerAccess

### 3.1 函数签名

```c
MUresult muapiCtxEnablePeerAccess(MUcontext peerContext, unsigned int Flags);
```

### 3.2 完整调用链

```
User Code (on Device A)
  │
  └─ muCtxEnablePeerAccess(peerCtx_B, flags)
        │                                    // peerCtx_B = Device B 的 Context
        ├─ flags <= MU_PEERACCESS_POLICY_MAX ?
        │     ├─ NO  → INVALID_VALUE
        │     └─ YES ↓
        │
        ├─ peerCtx != nullptr ?
        │     ├─ NO  → INVALID_CONTEXT
        │     └─ YES ↓
        │
        ├─ TlsCtxTop() → pLocalCtx (Device A)
        │     ├─ nullptr → INVALID_CONTEXT
        │     └─ 有效 ↓
        │
        ├─ pLocalCtx->GetDevice() == pPeerCtx->GetDevice() ?
        │     ├─ YES → PEER_ACCESS_UNSUPPORTED (同一设备)
        │     └─ NO  ↓
        │
        └─ pPeerCtx->EnablePeer(pLocalDevice, flags)
              │
              └─ [Core 层 Context::EnablePeer, 见下文 3.3]
```

### 3.3 源码逐行分析

**`mu_peer.cpp:39-64`**:

```cpp
MUresult muapiCtxEnablePeerAccess(MUcontext peerCtx, unsigned int flags) {
    MUresult status = MUSA_SUCCESS;          // 注意: 非 InitPlatform() 入口

    if (flags > MU_PEERACCESS_POLICY_MAX) {  // Step 1: flags 上限校验
        status = MUSA_ERROR_INVALID_VALUE;
    } else if (nullptr == peerCtx) {         // Step 2: peerCtx 空检查
        status = MUSA_ERROR_INVALID_CONTEXT;
    } else {
        Musa::IContext* pContext = TlsCtxTop();  // Step 3: 获取当前上下文
        if (!pContext) {
            status = MUSA_ERROR_INVALID_CONTEXT;
        } else {
            auto pPeerContext = Musa::ICast<Musa::IContext>(peerCtx);
            // Step 4: 同一设备检查
            if (pContext->GetDevice() == pPeerContext->GetDevice()) {
                status = MUSA_ERROR_PEER_ACCESS_UNSUPPORTED;
            }
            if (MUSA_SUCCESS == status) {
                // Step 5: 委托 Core 层，注意方向:
                // pPeerCtx->EnablePeer(localDevice) → 在 peer context 上建立映射
                status = pPeerContext->EnablePeer(pContext->GetDevice(), flags);
            }
        }
    }

    return status;
}
```

**关键设计**：

- **方向性**: `pPeerCtx->EnablePeer(localDevice)` — 在**对方的 Context 上**建立映射，使得对方的 memory 可以被本设备访问
- **不自调用**: 与 `muapiMemHostRegister` 不同，此函数开头**未调用** `InitPlatform()`，因为它假设平台已初始化（否则 `TlsCtxTop()` 会失败）
- **同一设备快速失败**: `pContext->GetDevice() == pPeerContext->GetDevice()` 时直接返回 `PEER_ACCESS_UNSUPPORTED`
- **⚠️ status 初始化**: 使用 `MUSA_SUCCESS` 而非 `InitPlatform()` 返回值，与其他 Driver 函数不同

### 3.4 Context::EnablePeer 实现

需要深入 `context.cpp` 查看。该函数负责：

```
Context::EnablePeer(peerDevice, flags)
  │
  ├─ 检查 m_Peers[peerDeviceId] 是否已启用
  ├─ 设置 m_Peers[peerDeviceId] = PEERMAP_FLAG_ENABLED
  ├─ 遍历 m_Memories，为已有内存建立 peer 映射
  │     └─ Memory::EnablePeerAccess(peerDevice)
  └─ 在 HAL 层建立实际的 GPU 映射
```

### 3.5 Context::DisablePeer 实现

```
Context::DisablePeer(peerDevice)
  │
  ├─ 遍历 m_Memories，移除 peer 映射
  │     └─ Memory::DisablePeerAccess(peerDevice)
  ├─ 清除 m_Peers[peerDeviceId]
  └─ 在 HAL 层移除 GPU 映射
```

---

## 4. muCtxDisablePeerAccess

### 4.1 函数签名

```c
MUresult muapiCtxDisablePeerAccess(MUcontext peerContext);
```

### 4.2 源码分析

**`mu_peer.cpp:66-82`**:

```cpp
MUresult muapiCtxDisablePeerAccess(MUcontext peerCtx) {
    MUresult status = MUSA_SUCCESS;

    if (nullptr == peerCtx) {
        status = MUSA_ERROR_INVALID_CONTEXT;
    } else {
        Musa::IContext* pContext = TlsCtxTop();
        if (!pContext) {
            status = MUSA_ERROR_INVALID_CONTEXT;
        } else {
            auto pPeerContext = Musa::ICast<Musa::IContext>(peerCtx);
            // 直接禁用，错误码来自 DisablePeer
            status = pPeerContext->DisablePeer(pContext->GetDevice());
        }
    }

    return status;
}
```

**与 EnablePeer 的区别**：

- 参数更少：无需 `flags`
- 无同一设备检查：因为 `DisablePeer` 内部会处理（不存在则无操作）
- 错误路径更简单：仅需检查 `peerCtx` 和 `TlsCtxTop()`

---

## 5. 与 Memory 操作的交互

### 5.1 CreateMemory 中的自动 Peer 映射

在 `10_CreateMemory_and_MapToPeers.md` 中已详细分析，`Context::CreateMemory` 会在所有已启用 Peer Access 的设备上自动建立映射。该流程依赖于 `m_Peers` 数组：

```cpp
// context.cpp 伪代码
Context::CreateMemory(IMemory** ppMemory, const MemoryCreateInfo& createInfo) {
    // 1. 创建 IMemory
    // 2. 检查 m_Peers[]，为每个启用了 peer access 的设备建立映射
    for (int i = 0; i < deviceCount; i++) {
        if (m_Peers[i] == PEERMAP_FLAG_ENABLED) {
            pMemory->AddPeerMapping(device[i]);
        }
    }
    // 3. 注册到 MemoryTracker
}
```

### 5.2 Peer Copy 依赖关系

```
muDeviceCanAccessPeer(devA, devB) → true
    │
    ├─ muCtxEnablePeerAccess(ctxB, flags)
    │     └─ Device B 的 memory 在 Device A 上建立映射
    │
    └─ muMemcpy(peerSrc, peerDst, size, stream)
          └─ 需要 EnablePeer 已完成映射
```

> **注意**: `muCtxEnablePeerAccess` 中的 `peerContext` 参数是**被访问方**的 Context。调用方在 Device A 上执行，传入 Device B 的 Context，表示"允许 Device A 访问 Device B 的内存"。函数内部在 Device B 的 Context 上调用 `EnablePeer(DeviceA, flags)`，建立双向（或单向）映射。

---

## 6. Peer Access 标志位

| 标志 | 值 | 含义 |
|------|----|------|
| `MU_PEER_ACCESS_LAZY` | 0x01 | 延迟映射，仅在访问时建立 PTE |
| `MU_PEER_ACCESS_SYSTEM` | 0x02 | 允许系统内存 Peer 访问 |
| `MU_PEER_ACCESS_RETAIN_ALLOCATED` | 0x08 | 保留已分配映射 |
| `MU_PEERACCESS_POLICY_MAX` | 0x0F | 策略掩码上限 |

> 具体标志值定义在 `musa.h` 中，此处不展开。

---

## 7. 设计要点

| 要点 | 说明 |
|------|------|
| 方向性 | EnablePeer 在**参数 Peer 的 Context**上建立映射，方向易混淆 |
| 同一设备 | CanAccessPeer 同一设备返回 false，EnablePeer 返回 PEER_ACCESS_UNSUPPORTED |
| 错误码选择 | 同一设备用 `PEER_ACCESS_UNSUPPORTED` 而非 `INVALID_VALUE`，语义更精确 |
| 平台分发 | P2P 能力查询通过 HAL 抽象，Linux(DRM) 和 Windows(WDDM) 各自实现 |
| 初始化假设 | EnablePeer/DisablePeer 未调用 `InitPlatform()`，假设平台已就绪 |
| 自动映射 | CreateMemory 后检查 `m_Peers[]`，自动为已启用 peer 的设备建立映射 |

---

## 8. ASCII 时序图

```
Device A (Caller)                  Device B (Peer)
  │                                    │
  │ muDeviceCanAccessPeer(&ok, A, B)  │
  │───────────────────────────────────>│
  │                                    │ CanAccessPeer()
  │                                    │── GetDeviceP2pCapability()
  │<────────────── ok=1 ──────────────│──→ ioctl P2P_CAPABILITY
  │                                    │
  │ muCtxEnablePeerAccess(ctxB, 0)    │
  │───────────────────────────────────>│
  │                                    │ EnablePeer(DeviceA, 0)
  │                                    │── 设置 m_Peers[A]=ENABLED
  │                                    │── 遍历 B 的 memories
  │                                    │     AddPeerMapping(DeviceA)
  │                                    │── HAL 层建立 PTE 映射
  │<────────── SUCCESS ───────────────│
  │                                    │
  │ muMemcpy(peerPtr_B, peerPtr_A)    │
  │───────────────────────────────────>│
  │                                    │ 通过已建立的映射访问
  │<────────────── DONE ──────────────│
  
  (禁用时方向相反)
  │ muCtxDisablePeerAccess(ctxB)      │
  │───────────────────────────────────>│
  │                                    │ DisablePeer(DeviceA)
  │                                    │── 移除所有 peer 映射
  │<────────── SUCCESS ───────────────│
```
