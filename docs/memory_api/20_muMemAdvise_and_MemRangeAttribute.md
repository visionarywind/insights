# muMemAdvise & MemRange Attribute — 内存建议与范围属性查询

> `muMemAdvise` / `muMemAdvise_v2` / `muMemRangeGetAttribute` / `muMemRangeGetAttributes` / `muMemRangeSetAttribute` 对应 CUDA `cudaMemAdvise` / `cuMemRangeGetAttribute` 等。

---

## 功能

| API | 功能 |
|-----|------|
| `muMemAdvise` | 对设备内存区域给出访问建议（读多、写多、随机等），供驱动优化页面迁移/缓存策略 |
| `muMemAdvise_v2` | v2 版本，用 `MUmemLocation` 替代 `MUdevice`，支持 NUMA 节点粒度 |
| `muMemRangeGetAttribute` | 查询设备内存范围的属性值（只读单属性） |
| `muMemRangeGetAttributes` | 批量查询设备内存范围的多个属性值 |
| `muMemRangeSetAttribute` | 设置设备内存范围的属性值 |

---

## 完整调用链

```
User Code
  │
  ├─ muMemAdvise(devPtr, count, advice, device)
  │     │
  │     ├─ InitPlatform()
  │     ├─ devPtr==0 → INVALID_VALUE
  │     ├─ device >= deviceCount → INVALID_VALUE (但 device==CPU 时跳过检查)
  │     ├─ TlsCtxTop() → nullptr → INVALID_CONTEXT
  │     └─ return NOT_SUPPORTED          ← ★ 始终返回不支持
  │
  ├─ muMemAdvise_v2(devPtr, count, advice, location)
  │     │
  │     ├─ InitPlatform()
  │     ├─ devPtr==0 或 count==0 → INVALID_VALUE
  │     ├─ location.type 无效/越界 → INVALID_VALUE
  │     ├─ advice 枚举范围校验
  │     ├─ TlsCtxTop() → nullptr → INVALID_CONTEXT
  │     └─ return NOT_SUPPORTED          ← ★ 始终返回不支持
  │
  └─ muMemRangeGetAttribute / muMemRangeGetAttributes
        │
        └─ return NOT_SUPPORTED          ← ★ 始终返回不支持
```

---

## 关键代码路径

**`mu_memory.cpp:2288-2335`**:

```cpp
// muMemAdvise — 第一版，仅检查参数后返回不支持
MUresult muapiMemAdvise(MUdeviceptr devPtr, size_t count, MUmem_advise advice, MUdevice device) {
    MUresult status = InitPlatform();
    if (status == MUSA_SUCCESS) {
        if (0 == devPtr || (device != MU_DEVICE_CPU && device >= Musa::Platform::Get().GetDeviceCount())) {
            status = MUSA_ERROR_INVALID_VALUE;
        } else {
            Musa::IContext* pContext = TlsCtxTop();
            if (pContext == nullptr) {
                status = MUSA_ERROR_INVALID_CONTEXT;
            } else {
                status = MUSA_ERROR_NOT_SUPPORTED;  // ★ 死代码式返回
            }
        }
    }
    return status;
}

// muMemAdvise_v2 — 第二版，参数校验更完善，但仍返回不支持
MUresult muapiMemAdvise_v2(MUdeviceptr devPtr, size_t count, MUmem_advise advice, MUmemLocation location) {
    // 校验: devPtr/count/location.type/location.id/advice 枚举范围
    // 获取 Context
    // return NOT_SUPPORTED
}

// muMemRangeGetAttribute — 直接返回不支持
MUresult muapiMemRangeGetAttribute(...) { return MUSA_ERROR_NOT_SUPPORTED; }
MUresult muapiMemRangeGetAttributes(...) { return MUSA_ERROR_NOT_SUPPORTED; }
```

---

## 死代码/不可达路径标注

1. **`muapiMemAdvise` 第 14-17 行**: `pContext` 非空时的逻辑体只有 `return NOT_SUPPORTED`，Context 校验形同虚设
2. **`muapiMemAdvise_v2` 的完整参数校验**: 包含 `location.type` 枚举全范围校验、`advice` 枚举范围校验（`MU_MEM_ADVISE_SET_READ_MOSTLY` ~ `MU_MEM_ADVISE_UNSET_ACCESSED_BY`），但无论参数如何都返回 `NOT_SUPPORTED`
3. **`muapiMemRangeGetAttribute` / `muapiMemRangeGetAttributes`**: 函数体内无任何逻辑，直接返回

这些 API 在 MUSA 1.x 中均为 **Stub 实现**，尚未对接底层驱动。

---

## CUDA 对比

| CUDA API | MUSA API | CUDA 状态 | MUSA 状态 |
|----------|----------|-----------|-----------|
| `cudaMemAdvise` | `muMemAdvise` | 完整支持 | NOT_SUPPORTED |
| `cudaMemAdvise_v2` | `muMemAdvise_v2` | 完整支持 | NOT_SUPPORTED |
| `cuMemRangeGetAttribute` | `muMemRangeGetAttribute` | 完整支持 | NOT_SUPPORTED |
| `cuMemRangeGetAttributes` | `muMemRangeGetAttributes` | 完整支持 | NOT_SUPPORTED |
| `cuMemRangeSetAttribute` | `muMemRangeSetAttribute` | 完整支持 | **无对应 API** |

---

## 设计要点

1. **接口占位**: 函数签名、参数校验逻辑均已完备，仅缺底层实现
2. **v2 演进**: `muMemAdvise_v2` 用 `MUmemLocation` 替代 `MUdevice`，支持 `HOST_NUMA_CURRENT` 等细粒度位置描述
3. **`muMemRangeSetAttribute` 缺失**: 公开头文件中无此函数声明，CUDA 对应 `cuMemRangeSetAttribute` 亦无 MUSA 移植
4. **潜在实现路径**: 需 HAL 层提供 `IMemory::Advise()` + `IMemory::GetRangeAttribute()` 接口，当前 HAL 尚无此类方法