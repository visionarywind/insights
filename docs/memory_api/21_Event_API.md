# Event API — muEventCreate / Record / Synchronize / ElapsedTime / Destroy

> MUSA Event API 对应 CUDA Event API（`cudaEventCreate` / `cudaEventRecord` / `cudaEventSynchronize` / `cudaEventElapsedTime` / `cudaEventDestroy`）。

---

## 功能

| API | 功能 |
|-----|------|
| `muEventCreate` | 创建 Event 对象，可指定 flags（BlockingSync / DisableTiming / Interprocess） |
| `muapiEventRecord` | 将 Event 记录到指定 Stream（默认 flags） |
| `muapiEventRecordWithFlags` | 将 Event 记录到指定 Stream（自定义 flags） |
| `muapiEventSynchronize` | 阻塞 CPU 等待 Event 完成 |
| `muapiEventElapsedTime` | 计算两个 Event 之间的时间差（毫秒） |
| `muapiEventDestroy` | 销毁 Event 对象 |

---

## 完整调用链

```
User Code
  │
  ├─ muapiEventCreate(&hEvent, Flags)
  │     │
  │     ├─ InitPlatform()
  │     ├─ phEvent==nullptr → INVALID_VALUE
  │     ├─ Flags 不在合法集合中 → INVALID_VALUE
  │     │   合法值: DEFAULT(0), BLOCKING_SYNC(1), DISABLE_TIMING(2),
  │     │   BLOCKING_SYNC|DISABLE_TIMING(3), DISABLE_TIMING|INTERPROCESS(6),
  │     │   BLOCKING_SYNC|DISABLE_TIMING|INTERPROCESS(7)
  │     ├─ TlsCtxTop() → nullptr → INVALID_CONTEXT
  │     ├─ pContext->CreateEvent(&pEvent, Flags)
  │     │     └─ 创建底层 HAL Event 资源
  │     └─ *phEvent = pEvent
  │
  ├─ muapiEventRecord(hEvent, hStream)
  │     └─ imuapiEventRecordWithFlags(hEvent, hStream, MU_EVENT_RECORD_DEFAULT)
  │           │
  │           ├─ InitPlatform()
  │           ├─ TlsCtxTop() → pContext
  │           ├─ Context::InfoStream() → pStream (支持 nullptr → 默认流)
  │           ├─ pContext->ValidateEvent(pEvent)
  │           ├─ pStream->GetContext() == pEvent->GetContext()  // 必须同 Context
  │           ├─ SetEventParameter{ pEvent, flags }
  │           └─ pStream->CmdSetEvent(param, false)
  │                     │
  │                     └─ 向 Stream 命令列表追加 CmdSetEvent 命令
  │
  ├─ muapiEventRecordWithFlags(hEvent, hStream, flags)
  │     └─ 同上，flags 直接传入
  │
  ├─ muapiEventSynchronize(hEvent)
  │     │
  │     ├─ InitPlatform()
  │     ├─ hEvent==nullptr → INVALID_HANDLE
  │     ├─ TlsCtxTop() → pContext
  │     ├─ pContext->ValidateEvent(pEvent)
  │     └─ pEvent->Synchronize()       ← 阻塞 CPU
  │
  ├─ muapiEventElapsedTime(&ms, hStart, hEnd)
  │     │
  │     ├─ InitPlatform()
  │     ├─ pMilliseconds==nullptr → INVALID_VALUE
  │     ├─ hStart/hEnd==nullptr → INVALID_HANDLE
  │     ├─ TlsCtxTop() → pContext
  │     ├─ pContext->ValidateEvent(pStart/pEnd)
  │     └─ pContext->GetEventElapsedTime(&ms, pStart, pEnd)
  │
  └─ muapiEventDestroy(hEvent)
        └─ muapiEventDestroy_v2(hEvent)
              │
              ├─ InitPlatform()
              ├─ hEvent==nullptr → INVALID_HANDLE
              ├─ TlsCtxTop() → pContext
              ├─ pContext->ValidateEvent(pEvent)
              └─ pEvent->GetContext()->DestroyEvent(pEvent)
```

---

## 时序图

```
应用层         Wrapper          Driver(mu_event)    Context           Stream             HAL
  │               │                  │                  │                  │                │
  │ EventCreate   │                  │                  │                  │                │
  │──────────────>│                  │                  │                  │                │
  │               │ muapiEventCreate │                  │                  │                │
  │               │─────────────────>│                  │                  │                │
  │               │                  │ CreateEvent()    │                  │                │
  │               │                  │─────────────────────────────────────────────────────────>│
  │               │                  │                  │ 分配底层 event 资源 │                │
  │               │                  │◄─────────────────────────────────────────────────────────│
  │◄──────────────┼──────────────────┘                  │                  │                │
  │               │                                     │                  │                │
  │ EventRecord   │                  │                  │                  │                │
  │──────────────>│                  │                  │                  │                │
  │               │ muapiEventRecord │                  │                  │                │
  │               │─────────────────>│                  │                  │                │
  │               │                  │ InfoStream()     │                  │                │
  │               │                  │ ValidateEvent()  │                  │                │
  │               │                  │                  │ CmdSetEvent()    │                │
  │               │                  │                  │─────────────────────────────────────>│
  │               │                  │                  │ 追加 CmdSetEvent  │                │
  │               │                  │                  │ 到命令列表        │                │
  │◄──────────────┼──────────────────┘                  │                  │◄──────────────────│
  │               │                                     │                  │   事件触发          │
  │ EventSynchronize│                 │                  │                  │                │
  │──────────────>│                  │                  │                  │                │
  │               │                  │                  │                  │                │
  │               │   (CPU 阻塞等待 event 完成)           │                  │                │
  │◄──────────────┼─────────────────────────────────────────────────────────────────────────────│
```

---

## 关键代码路径

**Event Record (`mu_event.cpp:10-51`)**:

```cpp
static MUresult imuapiEventRecordWithFlags(MUevent hEvent, MUstream hStream, unsigned int flags) {
    MUresult status = InitPlatform();
    if (status == MUSA_SUCCESS) {
        do {
            Musa::IContext* pContext = TlsCtxTop();
            if (!pContext) { status = MUSA_ERROR_INVALID_CONTEXT; break; }
            if (!hEvent) { status = MUSA_ERROR_INVALID_HANDLE; break; }

            Musa::IStream* pStream = Musa::Context::InfoStream(..., hStream);
            if (!pStream) { status = MUSA_ERROR_INVALID_CONTEXT; break; }

            // ★ hEvent 和 hStream 必须属于同一个 Context
            if (pStream->GetContext() != pEvent->GetContext()) {
                status = MUSA_ERROR_INVALID_HANDLE; break;
            }

            Musa::SetEventParameter param{};
            param.pEvent = pEvent;
            param.flags = flags;
            status = pStream->CmdSetEvent(param, false);  // false = 非内部调用
        } while (0);
    }
    return status;
}
```

**Event Create flags 校验 (`mu_event.cpp:72-76`)**:

```cpp
if (Flags != MU_EVENT_DEFAULT &&
    Flags != MU_EVENT_BLOCKING_SYNC && Flags != MU_EVENT_DISABLE_TIMING &&
    Flags != (MU_EVENT_BLOCKING_SYNC | MU_EVENT_DISABLE_TIMING) &&
    Flags != (MU_EVENT_DISABLE_TIMING | MU_EVENT_INTERPROCESS) &&
    Flags != (MU_EVENT_BLOCKING_SYNC | MU_EVENT_DISABLE_TIMING | MU_EVENT_INTERPROCESS)) {
    status = MUSA_ERROR_INVALID_VALUE;
}
```

> 注意：不允许单独指定 `MU_EVENT_INTERPROCESS`（必须与 `DISABLE_TIMING` 组合）。

---

## IPC Event 相关

### `muapiIpcGetEventHandle`

- **约束**: 依赖 `#if ATOMIC_INT_LOCK_FREE == 2`
- 若不满足（编译器不支持 lock-free atomic），直接返回 `NOT_SUPPORTED`
- 调用 `pEvent->ExportIpcHandle(pHandle)` 导出底层 event 的 IPC 句柄

### `muapiIpcOpenEventHandle`

- **约束**: 同样依赖 `ATOMIC_INT_LOCK_FREE == 2`
- 创建 Event 时强制设置 `MU_EVENT_INTERPROCESS | MU_EVENT_DISABLE_TIMING`
- 调用 `pEvent->ImportIpcHandle(handle)` 导入 IPC 句柄
- 导入失败时会销毁已创建的 event（资源清理）

---

## Stream WaitEvent

```
User Code
  │
  └─ muapiStreamWaitEvent(hStream, hEvent, Flags)
        │
        ├─ InitPlatform()
        ├─ hEvent==nullptr → INVALID_HANDLE
        ├─ TlsCtxTop() → pContext
        ├─ InfoStream() → pStream
        ├─ ValidateEvent(pEvent)
        └─ pStream->CmdWaitEvent(param, false)
```

`StreamWaitEvent` 向 Stream 插入等待屏障，使 Stream 等待指定 Event 完成后才继续执行后续命令。

---

## 设计要点

1. **Event-Stream 同 Context 约束**: Record 和 WaitEvent 要求 Event 与 Stream 属于同一个 Context，跨 Context 会返回 `INVALID_HANDLE`
2. **flags 白名单校验**: Event Create 严格限定 flags 组合，不允许任意位或
3. **Interprocess Event**: 仅在 `ATOMIC_INT_LOCK_FREE == 2` 平台支持，IPC 导出/导入时自动设置 `DISABLE_TIMING`（因为跨进程事件不支持计时）
4. **EventRecord 为异步**: 仅将命令加入 Stream，实际触发由 Stream 执行驱动
5. **EventSynchronize 为同步**: 阻塞 CPU 直到 GPU 完成该 Event 对应的时间点
6. **`muapiEventRecord_ptsz` / `muapiEventRecordWithFlags_ptsz`**: 仅将 `musaStreamDefault` 映射为 `musaStreamPerThread`，然后调用非 ptsz 版本