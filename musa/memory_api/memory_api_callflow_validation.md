# Memory API 调用链埋点验证

> 验证仓库：`/home/shanfeng/workspace/linux-ddk/musa`  
> 验证用例：`/home/shanfeng/workspace/insights/musa/memory_api/memory_api_callflow_demo.cpp`  
> 日志开关：`MUSA_DRIVER_CALLFLOW_DEBUG=1`

## 1. 验证目的

本次验证用于校正 `insights/musa/memory_api` 中的执行路径描述。重点确认文档是否跳过关键层级，尤其是 stream command、memory pool、paging command、callback command 之间的真实关系。

## 2. 埋点范围

埋点全部复用 `src/driver/internal.h` 中的 `MUSA_DRIVER_CALLFLOW_LOG(...)`，默认关闭，只有设置 `MUSA_DRIVER_CALLFLOW_DEBUG` 且取值非空、非 `0` 时才输出日志。

已补充的关键埋点：

| 文件 | 关键函数 |
|------|----------|
| `src/driver/mu_memory.cpp` | `imuapiMemHostAlloc`、`muapiMemHostRegister_v2`、`muapiPointerGetAttributes`、`imuapiPointerGetAttribute`、`muapiMemsetD32_v2`、`muapiMemAllocAsync`、`muapiMemFreeAsync`、`muapiMemPrefetchAsync` |
| `src/musa/core/context.cpp` | `Context::GeneralMemset` |
| `src/musa/core/stream.cpp` | `Stream::CmdMemAlloc`、`Stream::AsyncMemAlloc`、`Stream::CmdMemFree`、`Stream::AsyncMemFree`、`Stream::CmdMemset` |
| `src/musa/core/memoryPool.cpp` | `MemoryPool::CreateMemory`、`MemoryPool::ModifyAccess`、`MemoryPool::DisableAccess`、`MemoryPool::DestroyMemory` |
| `src/musa/core/command/pagingCommand.cpp` | `PagingCommand::Submit` |
| `src/musa/core/command/callbackCommand.cpp` | `CallbackCommand::Submit` |
| `src/musa/core/command/memsetCommand.cpp` | `MemsetCommand::Submit` |

## 3. 构建与运行

Driver 构建命令：

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 shanfeng@10.18.32.25 \
  "docker exec mochi-build bash -lc 'cd /root/linux-ddk/musa && cmake --build build --target musa_dynamic -j 8'"
```

用例编译命令：

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 shanfeng@10.18.32.25 \
  "docker exec -w /workspace/workspace/insights/musa/memory_api mochi-sglang bash -lc \
  'g++ -std=c++17 -o /tmp/memory_api_callflow_demo memory_api_callflow_demo.cpp \
  -I/workspace/workspace/linux-ddk/musa/src/musa_shared_include \
  -L/workspace/workspace/linux-ddk/musa/build/lib \
  -Wl,-rpath,/workspace/workspace/linux-ddk/musa/build/lib \
  -lmusa -pthread -ldl'"
```

用例运行命令：

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 shanfeng@10.18.32.25 \
  "docker exec -w /workspace/workspace/insights/musa/memory_api mochi-sglang bash -lc \
  'MUSA_DRIVER_CALLFLOW_DEBUG=1 \
  LD_LIBRARY_PATH=/workspace/workspace/linux-ddk/musa/build/lib:/usr/local/musa/lib \
  /tmp/memory_api_callflow_demo > /tmp/memory_api_callflow_demo.log 2>&1'"
```

## 4. 验证结果

用例中所有 API 返回 `MUSA_SUCCESS`：

```text
muMemAlloc                   status=0
muMemsetD32                  status=0
muMemcpyHtoD                 status=0
muPointerGetAttributes       status=0
muMemPrefetchAsync           status=0
muMemHostAlloc               status=0
muMemHostRegister            status=0
muMemAllocAsync              status=0
muMemFreeAsync               status=0
muMemFree                    status=0
```

## 5. 已确认调用链

### 5.1 muMemsetD32

```text
muapiMemsetD32_v2
  -> Context::GeneralMemset
  -> Context::InfoStream
  -> Context::CreateMemsetNode
  -> Stream::CmdMemset
  -> Context::ResolveDependencyAndQueueCommand
  -> MemsetCommand::Submit
```

本次用例中 `MemsetCommand::Submit` 进入 `Command::SubmitToQueue`。源码中还存在 `CpuExecute()` 和 `DmaExecute()` 分支，具体选择由 `m_Engine` 决定。

### 5.2 muMemcpyHtoD

```text
muapiMemcpyHtoD_v2
  -> Context::GeneralMemcpy
  -> Context::CreateMemcpyNode
  -> Stream::CmdCopyMemory
  -> SyncMemcpyCommand::Submit
  -> copyManager=MemcpyH2D
```

### 5.3 muPointerGetAttributes

```text
muapiPointerGetAttributes
  -> imuapiPointerGetAttribute(attr=MEMORY_TYPE)
  -> imuapiPointerGetAttribute(attr=DEVICE_POINTER)
  -> imuapiPointerGetAttribute(attr=RANGE_START_ADDR)
  -> imuapiPointerGetAttribute(attr=RANGE_SIZE)
  -> imuapiPointerGetAttribute(attr=DEVICE_ORDINAL)
```

每个 attribute 都会重新进入 `imuapiPointerGetAttribute`，并执行 `MemoryTracker::FindRange(ptr, &offset, Hal::memoryPropertyPhysical)`。

### 5.4 muMemPrefetchAsync

```text
muapiMemPrefetchAsync
  -> InitPlatform
  -> TlsCtxTop
  -> Context::InfoStream
  -> return
```

日志明确显示：

```text
muapiMemPrefetchAsync fake path resolved context=... stream=... no command is queued
```

当前实现不创建 prefetch command，不向 stream 入队。

### 5.5 muMemHostAlloc

```text
imuapiMemHostAlloc
  -> TlsCtxTop
  -> IContext::CreateMemory
  -> Context::CreateMemory
  -> Memory::Init
  -> Memory::PinnedHostAlloc
```

本次用例传入 flags 为 `0x0`，进入 `CreateMemory` 前 runtime 追加后的 pinned host flags 为 `0x180000`。

### 5.6 muMemHostRegister

```text
muapiMemHostRegister_v2
  -> TlsCtxTop
  -> IContext::CreateMemory
  -> Context::CreateMemory
  -> Memory::Init
  -> Memory::PinnedHostRegister
```

本次用例传入 flags 为 `0x0`，driver 追加 `MU_MEMORY_REGISTER_OVERLAP_CHECK` 后进入 `CreateMemory` 的 flags 为 `0x100000`。

### 5.7 muMemAllocAsync

```text
muapiMemAllocAsync
  -> Context::InfoStream
  -> Stream::CmdMemAlloc
  -> Stream::AsyncMemAlloc
  -> MemoryPool::CreateMemory
  -> physical->Init(General, flags=0)
  -> MemoryPool::ModifyAccess
  -> Stream::CmdPaging
  -> PagingCommand::Submit
```

当前正常路径没有单独的 `MemAllocCommand`。API 调用期间完成虚拟内存对象、物理内存对象和绑定关系创建，stream 中排序执行的是 `PagingCommand`。

### 5.8 muMemFreeAsync

```text
muapiMemFreeAsync
  -> Context::InfoStream
  -> Stream::CmdMemFree
  -> Stream::AsyncMemFree
  -> MemoryPool::DisableAccess
  -> Stream::CmdPaging
  -> PagingCommand::Submit
  -> CallbackCommand::Submit
```

`CallbackCommand::Submit` 执行 callback。callback 内部释放物理映射并归还 pool：

```text
virt->DestroyPhysMemories()
pPool->DestroyMemory(virt)
```

## 6. 文档修正结论

已据此修正以下内容：

- `022_muMemPrefetchAsync.md`：改为参数校验路径，不再描述不存在的 `PrefetchCommand`。
- `07_muMemAllocAsync.md`：补齐 `MemoryPool::ModifyAccess -> Stream::CmdPaging -> PagingCommand::Submit`。
- `08_muMemFreeAsync.md`：补齐 `MemoryPool::DisableAccess -> Stream::CmdPaging -> PagingCommand::Submit -> CallbackCommand::Submit`。
- `06_muMemsetD32_v2.md`：补齐 `CreateMemsetNode`、`GraphMemsetNode::Init`、`MemsetCommand::Submit`。
- `03_muMemHostAlloc.md`、`04_muMemHostRegister_v2.md`：补齐 driver 入口到 `Memory::Init` 的层级。
- `020_muPointerGetAttributes.md`：明确批量查询逐项调用内部单属性函数。
