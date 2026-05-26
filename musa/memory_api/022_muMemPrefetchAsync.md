# muMemPrefetchAsync — 当前实现为参数校验路径

> 源码文件：`musa/src/driver/mu_memory.cpp` 中的 `muapiMemPrefetchAsync`、`muapiMemPrefetchAsync_v2`

## 1. 功能结论

当前源码中，`muMemPrefetchAsync` 不是实际的数据预取实现。函数注释明确说明这是 fake implementation，因为当前 MUSA managed memory 的同步访问不需要通过该 API 做数据搬迁。

因此当前实现只做以下事情：

- 初始化平台。
- 校验指针、大小、目标设备或 location。
- 获取当前线程 TLS 中的 Context。
- 解析传入的 stream 句柄。
- 返回校验结果。

当前实现不做以下事情：

- 不查询 `MemoryTracker`。
- 不检查目标内存对象的属性。
- 不创建 `PrefetchCommand`。
- 不调用 `Stream::CmdPrefetchAsync`。
- 不向 stream 入队。
- 不触发页迁移或数据搬迁。

## 2. muMemPrefetchAsync 调用链

```
muapiMemPrefetchAsync(devPtr, count, dstDevice, hStream)
  │
  ├─ InitPlatform()
  │
  ├─ 参数校验
  │   ├─ devPtr == 0        -> MUSA_ERROR_INVALID_VALUE
  │   ├─ count == 0         -> MUSA_ERROR_INVALID_VALUE
  │   └─ dstDevice 非 CPU 且超出设备数量 -> MUSA_ERROR_INVALID_VALUE
  │
  ├─ TlsCtxTop()
  │   └─ 当前线程没有 Context -> MUSA_ERROR_INVALID_CONTEXT
  │
  ├─ Context::InfoStream(ctx, hStream)
  │   └─ stream 无效 -> MUSA_ERROR_INVALID_HANDLE
  │
  └─ return status
```

## 3. muMemPrefetchAsync_v2 调用链

```
muapiMemPrefetchAsync_v2(devPtr, count, location, flags, hStream)
  │
  ├─ InitPlatform()
  │
  ├─ 参数校验
  │   ├─ devPtr == 0 -> MUSA_ERROR_INVALID_VALUE
  │   ├─ count == 0  -> MUSA_ERROR_INVALID_VALUE
  │   ├─ location.type == MU_MEM_LOCATION_TYPE_INVALID
  │   │   -> MUSA_ERROR_INVALID_VALUE
  │   ├─ location.type > MU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT
  │   │   -> MUSA_ERROR_INVALID_VALUE
  │   └─ location.type == DEVICE 且 location.id 超出设备数量
  │       -> MUSA_ERROR_INVALID_VALUE
  │
  ├─ TlsCtxTop()
  │   └─ 当前线程没有 Context -> MUSA_ERROR_INVALID_CONTEXT
  │
  ├─ Context::InfoStream(ctx, hStream)
  │   └─ stream 无效 -> MUSA_ERROR_INVALID_HANDLE
  │
  └─ return status
```

## 4. 验证日志

使用 `MUSA_DRIVER_CALLFLOW_DEBUG=1` 执行最小用例后，日志显示该 API 只解析 Context 和 Stream：

```text
muapiMemPrefetchAsync enter ptr=0x1000c400000 count=1024 dstDevice=0 stream=0x...
muapiMemPrefetchAsync fake path resolved context=0x... stream=0x... no command is queued
muapiMemPrefetchAsync exit status=0
```

日志中没有 `Stream::CmdPaging`、`PrefetchCommand`、`MemoryPool::ModifyAccess` 或其他 stream command 提交记录。

## 5. 阅读结论

分析当前版本时，应把 `muMemPrefetchAsync` 视为兼容性 API。它能够检查参数和 stream 句柄，但不改变内存位置、访问权限或 stream 命令队列。
