# M2 API 执行流程与 ModelEvent 插点分析

> 目标：把 M2 阶段从“API 清单”推进到“基于当前源码可直接落地的白盒插点方案”。本文按真实代码路径展开 Runtime API -> Driver API -> Core 对象/Command -> Stream Queue -> Command Submit -> HAL/M3D/OS Submit -> MUPTI 的端到端链路，明确当前已有 MUPTI 插点、缺口，以及新增 ModelEvent 的插入位置。

## 1. M2 范围与结论

M2 覆盖 Top90 Driver/Runtime API 中除 M1 外的高频 API 家族：

1. Launch / Module / Function
2. Memory allocation / free / pool
3. Memcpy / memset / memory transfer / memory atomic
4. Stream / Event / Synchronization
5. Graph
6. Context / Device / Resource / Green Context
7. Command submit boundary: Core -> HAL -> M3D -> OS/KMD

M2 的核心判断：

- 当前 MUPTI 已覆盖“外层 API 进入/退出”和一部分“GPU activity/同步 activity/graph activity/关联关系”。
- 当前 MUPTI 不足以解释 Driver 内部软件耗时来源，例如参数归一化、上下文查找、stream 选择、pool 分配、dependency resolve、queue 等待、command build、HAL/M3D submit 翻译、OS submit 等。
- ModelEvent 不应替代 MUPTI callback/activity，而应复用 callback/activity 的 `correlation_id`、activity buffer、flush/drop 机制和 submission/kick 关联表，在 Driver/Core 源码内部补充 span/point 级事件。

## 2. 当前已有 MUPTI 插点基线

### 2.1 Runtime / Driver API callback 与 activity

源码位置：`MUPTI/src/core/process_callback.cpp::ProcessDriverCallback`

当前能力：

- Runtime API enter/exit callback。
- Driver API enter/exit callback。
- Runtime API activity：`MUPTI_ACTIVITY_KIND_RUNTIME` enabled 且 callback depth 为 0 时在 API exit 记录。
- Driver API activity：`MUPTI_ACTIVITY_KIND_DRIVER` enabled 且 callback depth 为 0 时在 API exit 记录。
- 使用 `correlationId` 串联 Runtime/Driver/API activity。
- 调用 `processExternalCorrelations(correlationId)` 处理外部相关性栈。

Runtime 侧入口：

- `MUSA-Runtime/src/internal.h::ApiTrace`
- `MUSA-Runtime/src/internal.h::EnterRuntimeApi`
- `MUSA-Runtime/src/internal.h::ExitRuntimeApi`
- `MUSA-Runtime/src/musa_wrappers_generated.cpp` 中所有 generated wrapper 均创建 `ApiTrace`，再调用 `musaapiXXX`。

因此 M2 不需要重新实现 Runtime/Driver API 外层计时，只需要把 ModelEvent 通过相同 correlation 接入。

### 2.2 已启用 activity kind

源码位置：`MUPTI/src/core/activity_record_kinds.inc`

已启用：

- `MUPTI_ACTIVITY_KIND_MEMCPY`
- `MUPTI_ACTIVITY_KIND_MEMSET`
- `MUPTI_ACTIVITY_KIND_KERNEL`
- `MUPTI_ACTIVITY_KIND_DRIVER`
- `MUPTI_ACTIVITY_KIND_RUNTIME`
- `MUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL`
- `MUPTI_ACTIVITY_KIND_SYNCHRONIZATION`
- `MUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION`
- `MUPTI_ACTIVITY_KIND_GRAPH_TRACE`
- `MUPTI_ACTIVITY_KIND_MEMORY_ATOMIC`
- `MUPTI_ACTIVITY_KIND_MEMORY_ATOMIC_VALUE`
- `MUPTI_ACTIVITY_KIND_MEMORY_TRANSFER`

未启用/注释状态：

- `MUPTI_ACTIVITY_KIND_MODULE`
- `MUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE`
- `MUPTI_ACTIVITY_KIND_STREAM`
- `MUPTI_ACTIVITY_KIND_MEMORY_POOL`

M2 建议：不要急于把这些未启用 activity kind 公开化；先用 private ModelEvent 记录内部 span，稳定后再决定是否提升为 public activity。

### 2.3 Driver/Core 已有 MUPTI tracepoint

源码位置：`linux-ddk/musa/src/driver/mupti/tracepoints.h`

当前已有 hook：

| 能力 | tracepoint | 说明 |
|---|---|---|
| memcpy activity | `EnterMemcpy` / `ExitMemcpy` | 以 `MemcpyCommand*` 为输入生成 memcpy activity |
| host memcpy | `StartHostMemcpy` / `StopHostMemcpy` | host copy path 的 CPU span |
| memset activity | `EnterMemset` | 以 `MemsetCommand*` 为输入生成 memset activity |
| memory atomic | `EnterMemoryAtomicV2` | memory atomic command activity |
| memory atomic value | `EnterMemoryAtomicValue` | memory atomic value activity |
| memory transfer | `EnterMemoryTransferV2` | memory transfer command activity |
| kernel | `RegisterKernel` / `RegisterKernelV2` | 记录 DispatchCommand 的 kernel activity metadata |
| kernel->kick | `AssignKernelToKick` | 建立 kernel correlation 到 kick/submission 的映射 |
| memcpy submission | `AssignSubmissionToCorrelation` | 建立 memop correlation 到 submission 的映射 |
| context 生命周期 | `CreateContext` / `DestroyContext` | 记录 context create/destroy |
| stream 生命周期 | `CreateStream` / `DestroyStream` | 记录 stream create/destroy |
| event sync | `RegisterEventSynchronize` / `StartEventSynchronize` / `StopEventSynchronize` | event sync activity |
| stream wait event | `RegisterStreamWaitEvent` / `StartStreamWaitEvent` / `StopStreamWaitEvent` | stream wait event sync activity |
| stream sync | `RegisterStreamSynchronize` / `StartStreamSynchronize` / `StopStreamSynchronize` | stream sync activity |
| context sync | `RegisterContextSynchronize` / `StartContextSynchronize` / `StopContextSynchronize` | context sync activity |
| graph trace | `RegisterGraphTrace` / `RegisterGraphTraceV2` / `MarkGraphTraceBegin` / `MarkGraphTraceEnd` | graph command/node trace |
| graph node activity | `RegisterGraphKernel` / `RegisterGraphMemcpy` / `RegisterGraphMemset` / `RegisterGraphMemoryAtomic` / `RegisterGraphMemoryAtomicValue` / `RegisterGraphMemoryTransfer` | graph 内节点 activity |

M2 缺口：这些 hook 多数在 command/activity 已经成形之后触发，缺少 command 形成前后的软件路径解释。

## 3. M2 ModelEvent 通用落地方式

### 3.1 私有接口

建议在 Driver/Core 内部新增轻量接口：

```cpp
bool MuptiModelEventEnabled(uint32_t domain, uint32_t event_id);
void MuptiEmitModelEvent(const ModelEventHeader* header,
                         const void* payload,
                         uint32_t payload_size);
```

trace off 快路径要求：

- 只读 `ready/enabled` 原子状态。
- 不分配内存。
- 不格式化字符串。
- 不加锁。
- span guard 只能在 enabled 后构造。

### 3.2 推荐基础事件字段

所有 M2 ModelEvent 至少携带：

- `event_type`: point/span begin/span end
- `domain`: runtime/driver/memory/stream/command/submit/graph/module/resource/hal/m3d
- `event_id`
- `timestamp_ns`
- `pid/tid`
- `correlation_id`
- `span_id/parent_id`
- `context_id`
- `stream_id`
- `command_id`
- `submission_id`
- `graph_id/graph_node_id`
- `status`
- payload

### 3.3 通用插点优先级

P0：直接解释 API 软件耗时大头，且插点稳定。

- API 参数校验与对象查找 span
- stream 解析 span
- command 创建 span
- dependency resolve span
- stream queue span
- command submit span
- HAL/M3D/OS submit span
- pool alloc/free/grow span
- graph instantiate/launch span

P1：解释配置/资源管理耗时。

- module load/function lookup span
- context/device/GreenContext resource span
- event/stream create/destroy span
- graph node add/update span

P2：更细粒度 payload 或质量报告。

- 参数归一化细分字段
- merge decision reason
- fallback reason
- wait reason histogram

## 4. Launch / Module / Function 家族

### 4.1 API 范围

Runtime 入口：`MUSA-Runtime/src/musa_module.cpp`

- `musaapiLaunchKernel`
- `musaapiLaunchKernelExC`
- `musaapiLaunchCooperativeKernel`
- `musaapiLaunchCooperativeKernelMultiDevice`
- `musaapiFuncGetAttributes`
- `musaapiFuncGetName`
- `musaapiFuncGetParamInfo`
- `musaapiFuncSetAttribute`
- `musaapiFuncSetCacheConfig`
- `musaapiFuncSetSharedMemConfig`

Driver 入口：`linux-ddk/musa/src/driver/mu_module.cpp`

- `muapiModuleLoad`
- `muapiModuleLoadData`
- `muapiModuleLoadFatBinary`
- `muapiModuleGetFunction`
- `muapiLaunchKernel`
- `muapiLaunchKernelEx`
- `muapiLaunchCooperativeKernel`
- `muapiModuleUnload`
- `muapiFuncGetAttribute`
- `muapiFuncSetAttribute`
- `muapiFuncGetName`
- `muapiFuncGetParamCount`
- `muapiFuncGetParamInfo`
- `muapiFuncLoad`

### 4.2 端到端执行路径

#### 4.2.1 Kernel launch

端到端路径：

1. `MUSA-Runtime/src/musa_wrappers_generated.cpp::musaLaunchKernel`
   - 创建 `ApiTrace`。
   - `EnterRuntimeApi` 触发 Runtime callback。
   - Tools callback enter。
   - 调用 `musaapiLaunchKernel`。
2. `MUSA-Runtime/src/musa_module.cpp::musaapiLaunchKernel`
   - Runtime 参数转换。
   - 进入 Driver export table 或直接调用 Driver launch。
3. `linux-ddk/musa/src/driver/mu_module.cpp::muapiLaunchKernel`
   - `InitPlatform()`。
   - `TlsCtxTop()` 获取当前 context。
   - 校验 `hfunc`、grid/block、params、stream。
   - `Context::InfoStream(...)` 获取 stream。
   - 组织 dispatch/kernel 参数。
   - 进入 stream command 创建路径。
4. Core 层创建 `DispatchCommand`。
5. dependency resolve。
6. `linux-ddk/musa/src/musa/core/stream.cpp::Stream::QueueCommand`
   - 等待 `m_AsyncCount < streamAsyncCapacity`。
   - `GetLastError()`。
   - `m_AsyncCount.fetch_add(1)`。
   - 持有 `m_SubmitMtx`。
   - `command->ChoosePerfEngine(m_LastCommand)`。
   - `command->SetPrevCommand(m_LastCommand)`。
   - `command->SetStatus(Command::Status::queued)`。
   - Dispatch command 设置 queued timestamp。
   - 在不支持 engine sync 时调用 `MUpti::MarkKernelQueued(command->GetCorId())`。
   - `m_CommandList.push_back(...)`。
   - `m_SubmitCv.notify_one()`。
7. stream submit thread 取出 command。
8. Dispatch command build cmd buffer。
9. `Command::SubmitToQueue` 或 dispatch-specific submit。
10. `Hal::IQueue::Submit`。
11. `hal/m3d/queue.cpp::Queue::Submit`。
12. `hal/m3d/m3d/src/core/queue.cpp::Queue::Submit`。
13. `ValidateSubmit` / `PreProcessSubmit` / `OsSubmit` / `PostProcessSubmit`。
14. KMD/OS submit。
15. MUPTI kernel activity 通过 `RegisterKernel/RegisterKernelV2`、`AssignKernelToKick` 与 kick/submission 关联。

#### 4.2.2 Module load / function lookup

`muapiModuleLoad` 真实路径：

1. `muapiModuleLoad(module, fname)`。
2. `InitPlatform()`。
3. 校验 `fname != nullptr`。
4. `std::ifstream file(fname, std::ios::binary | std::ios::ate)`。
5. `tellg()` 得到 image size。
6. 分配 `std::make_unique<char[]>(imageSize + 1)`。
7. `file.read(image.get(), imageSize)`。
8. 调用 `muapiModuleLoadData(module, image.get())`。
9. `muapiModuleLoadData` 中：
   - `TlsCtxTop()`。
   - 校验 `module/image/context`。
   - `pContext->CreateModule(&pModule)`。
   - `pModule->LoadFatBinary(image)`。
   - 成功后 `*module = pModule`；失败则 `pContext->DestroyModule(pModule)`。

`muapiModuleGetFunction` 真实路径：

1. `InitPlatform()`。
2. 校验 `TlsCtxTop()`、`hmod`、`hfunc`、`name`。
3. `Musa::ICast<Musa::IModule>(hmod)`。
4. `pModule->GetFunction(&pFunction, name)`。
5. 成功后返回 `MUfunction`。

Function attribute/query 路径：

- `muapiFuncGetAttribute` -> `TlsCtxTop()` -> `ICast<IFunction>` -> `pFunction->GetAttribute(pi, attrib)`。
- `muapiFuncSetAttribute` -> `ICast<IFunction>` -> `pFunction->SetAttribute(attrib, value)`。
- `muapiFuncGetName` -> `pFunction->GetFuncName(name)`。
- `muapiFuncGetParamCount` -> `pFunction->GetParamCount(paramCount)`。
- `muapiFuncGetParamInfo` -> `pFunction->GetParamInfo(paramIndex, paramOffset, paramSize)`。
- `muapiFuncLoad` -> `ICast<Function>` -> function lazy load path。

### 4.3 当前已有 MUPTI 插点

- API 外层：Runtime/Driver callback/activity 已覆盖。
- Kernel metadata：`RegisterKernel/RegisterKernelV2`。
- Kernel queued：`Stream::QueueCommand` 中 `MUpti::MarkKernelQueued(command->GetCorId())`。
- Kernel -> kick：`AssignKernelToKick`。
- Graph kernel：`RegisterGraphKernel`。

缺口：

- `muapiLaunchKernel` 内部校验、stream lookup、dispatch command 创建无事件。
- `Stream::QueueCommand` 的 async capacity wait、engine choice、prev command link、mutex queue 时间无事件。
- module file IO、module create、fatbin load、function lookup/lazy load 无内部事件。
- HAL/M3D submit 内部阶段无事件。

### 4.4 新增 ModelEvent 插点

| 事件 | 文件/函数 | begin/end 位置 | payload | relation/cost |
|---|---|---|---|---|
| `LAUNCH_VALIDATE_SPAN` | `mu_module.cpp::muapiLaunchKernel/muapiLaunchKernelEx` | 参数校验前后 | grid/block/sharedMem/flags/status | `api -> validate`，解释参数处理成本 |
| `STREAM_RESOLVE_SPAN` | `mu_module.cpp::muapiLaunchKernel` | 调 `Context::InfoStream` 前后 | `hStream/context/stream/status` | `api -> stream` |
| `DISPATCH_COMMAND_CREATE_SPAN` | dispatch command 创建处 | `make_shared/new DispatchCommand` 前后 | function/grid/block/sharedMem/command_id | `api -> command` |
| `KERNEL_PARAM_PACK_SPAN` | launch 参数打包处 | params normalize 前后 | param_count/param_size | launch CPU cost |
| `STREAM_QUEUE_COMMAND_SPAN` | `stream.cpp::Stream::QueueCommand` | 函数入口到 push_back 后 | stream/command/type/async_count | `command -> stream` |
| `ASYNC_CAPACITY_WAIT_SPAN` | `Stream::QueueCommand` | while 前后，仅发生等待时 | async_count/capacity/wait_ns | queue backpressure |
| `COMMAND_ENGINE_CHOOSE_SPAN` | `Stream::QueueCommand` | `ChoosePerfEngine` 前后 | prev_command/engine | engine selection cost |
| `COMMAND_PREV_LINK_RELATION` | `Stream::QueueCommand` | `SetPrevCommand` 后 | command/prev_command | dependency chain |
| `MODULE_FILE_READ_SPAN` | `muapiModuleLoad` | open/read image 前后 | fname/image_size/status | module IO cost |
| `MODULE_LOAD_FATBIN_SPAN` | `muapiModuleLoadData` | `LoadFatBinary` 前后 | image/module/status | module parse/load cost |
| `FUNCTION_LOOKUP_SPAN` | `muapiModuleGetFunction` | `GetFunction` 前后 | module/name/status | symbol lookup cost |
| `FUNCTION_ATTRIBUTE_SPAN` | `muapiFunc*` | `pFunction->Get/Set...` 前后 | attrib/value/status | function metadata cost |

## 5. Memory allocation / free / pool 家族

### 5.1 API 范围

Runtime：`MUSA-Runtime/src/musa_memory.cpp`

- `musaapiMalloc`
- `musaapiMallocPitch`
- `musaapiMalloc3D`
- `musaapiMallocHost`
- `musaapiMallocManaged`
- `musaapiFree`
- `musaapiFreeHost`
- `musaapiHostAlloc`
- `musaapiPointerGetAttributes`
- `musaapiMemPrefetchAsync`

Driver：`linux-ddk/musa/src/driver/mu_memory.cpp`

- `muapiMemAlloc_v2`
- `muapiMemAllocAsync`
- `muapiMemAllocFromPoolAsync`
- `muapiMemFreeAsync`
- `muapiMemAllocPitch_v2`
- `muapiMemHostAlloc`
- `muapiMemAllocManaged`
- `muapiMemGetInfo_v2`
- `muapiPointerGetAttribute(s)`
- `muapiMemFree_v2`
- `muapiMemFreeHost`

### 5.2 端到端执行路径

#### 5.2.1 `muapiMemAllocAsync`

真实路径：

1. `muapiMemAllocAsync(dptr, bytesize, hStream)`。
2. `InitPlatform()`。
3. 校验：
   - `dptr == nullptr` -> `MUSA_ERROR_INVALID_VALUE`
   - `bytesize == 0` -> `*dptr = 0` 后返回
4. `TlsCtxTop()` 获取 context。
5. `Musa::Context::InfoStream(context, hStream)` 获取 stream。
6. 构造 `Musa::MemoryAllocParameter memAllocParam{}`。
7. `memAllocParam.size = bytesize`。
8. `pStream->CmdMemAlloc(memAllocParam, false)`。
9. 成功：`*dptr = memAllocParam.virtAddress`；失败：`*dptr = 0`。

Core path：

1. `stream.cpp::Stream::CmdMemAlloc(param, blocking)`。
2. 如果 stream capture active：`CaptureMemAlloc(param)`。
3. 如果 capture invalidated：`MUSA_ERROR_STREAM_CAPTURE_INVALIDATED`。
4. 否则：`AsyncMemAlloc(param, blocking)`。
5. `Stream::AsyncMemAlloc`：
   - `GetAllocationGranularity(..., MU_MEM_ALLOC_GRANULARITY_RECOMMENDED)`。
   - `allocSize = Util::AlignUp(param.size, physPageSize)`。
   - 选择 `param.pool` 或 device default memory pool。
   - `pPool->SetStream(this)`。
   - `pPool->CreateMemory(&virt, &virtAddr, allocSize)` 创建虚拟 memory。
   - `param.virtAddress = virtAddr`。
   - 创建 physical memory：`std::make_shared<Memory>(m_ParentCtx)`。
   - `physical->Init(createInfo)`。
   - `virt->Bind(spPhysical, allocSize, 0, 0)`。
   - `pPool->ModifyAccess(virt, physical, allocSize, blocking, this)`。
   - 失败回滚：`virt->Unbind(...)` / `pPool->DestroyMemory(virt)`。

#### 5.2.2 `muapiMemAllocFromPoolAsync`

与 `muapiMemAllocAsync` 类似，差异：

1. `Musa::MemoryPool* pMemoryPool = reinterpret_cast<Musa::MemoryPool*>(pool)`。
2. 校验 imported IPC pool：`pMemoryPool->IpcMemPoolData().m_IsImported`。
3. 校验 `pool == nullptr`。
4. `memAllocParam.pool = reinterpret_cast<Musa::MemoryPool*>(pool)`。
5. 后续进入同一个 `Stream::CmdMemAlloc -> AsyncMemAlloc`。

#### 5.2.3 `muapiMemFreeAsync`

真实路径：

1. `muapiMemFreeAsync(dptr, hStream)`。
2. `InitPlatform()`。
3. `dptr == 0` 直接成功。
4. `TlsCtxTop()`。
5. `Context::InfoStream(...)`。
6. `Platform::Get().GetMemoryByDevicePointer(dptr, nullptr)`。
7. `IntrusiveCast<Memory>`。
8. 根据 memory type 分支：
   - `memoryTypeIpcImport` / `memoryTypeExternal`: `pContext->DestroyMemory(memory)`。
   - `memoryTypeGeneral` / `memoryTypePitchedGeneral` / `memoryTypeManaged`: `memory->Synchronize()` 后 `memory->GetContext()->DestroyMemory(memory)`。
   - `memoryTypeVirtual` 且 `memory->GetPool() != nullptr`: `pStream->CmdMemFree(dptr, false)`。
   - 其他 invalid。

Core virtual pool free path：

1. `Stream::CmdMemFree(virtAddress, blocking)`。
2. capture active：`CaptureMemFree`。
3. capture invalid：返回 invalidated。
4. 否则：`AsyncMemFree(virtAddress, blocking)`。
5. `Stream::AsyncMemFree`：
   - `Platform::Get().GetMemoryByDevicePointer`。
   - `virt->GetPhysMemory(virtAddress)`。
   - `MemoryPool* pPool = virt->GetPool()`。
   - `pPool->DisableAccess(virt, physical, blocking, this)`。
   - 创建 `CallbackCommand`。
   - callback 中 `virt->DestroyPhysMemories()`，并在非 graph alloc 时 `pPool->DestroyMemory(virt)`。
   - `m_ParentCtx->ResolveDependencyAndQueueCommand(std::move(command), this, blocking)`。

### 5.3 当前已有 MUPTI 插点

- API 外层 Runtime/Driver callback/activity 已覆盖。
- Memory pool activity kind 当前未启用。
- async alloc/free 内部没有对应 activity hook。
- free 中可能产生 callback command，但当前没有表达 “free 由 DisableAccess + callback command 延迟销毁” 的关系。

缺口：

- pool 选择、alloc size 对齐、virtual memory 创建、physical memory 初始化、bind、modify access、disable access 全部不可见。
- free 的同步销毁 vs pool virtual async free 分支不可见。
- pool grow/reuse/miss 原因不可见。
- graph capture 分支不可见。

### 5.4 新增 ModelEvent 插点

| 事件 | 文件/函数 | begin/end 位置 | payload | relation/cost |
|---|---|---|---|---|
| `MEM_ALLOC_VALIDATE_SPAN` | `mu_memory.cpp::muapiMemAllocAsync/FromPoolAsync` | 参数校验前后 | bytesize/pool/imported/status | API validation cost |
| `MEM_ALLOC_STREAM_RESOLVE_SPAN` | `mu_memory.cpp` | `Context::InfoStream` 前后 | context/hStream/stream/status | api -> stream |
| `MEM_ALLOC_CAPTURE_DECISION` | `stream.cpp::Stream::CmdMemAlloc` | capture 分支判断后 | capture_status/blocking | capture relation |
| `MEM_POOL_SELECT_SPAN` | `Stream::AsyncMemAlloc` | pool 选择前后 | requested_size/aligned_size/pool/default_pool | pool choice |
| `MEM_POOL_CREATE_VIRT_SPAN` | `Stream::AsyncMemAlloc` | `pPool->CreateMemory` 前后 | pool/allocSize/virtAddr/status | pool virtual alloc cost |
| `MEM_PHYSICAL_INIT_SPAN` | `Stream::AsyncMemAlloc` | `physical->Init` 前后 | size/type/status | physical allocation cost |
| `MEM_BIND_SPAN` | `Stream::AsyncMemAlloc` | `virt->Bind` 前后 | virt/physical/size | bind cost |
| `MEM_POOL_MODIFY_ACCESS_SPAN` | `Stream::AsyncMemAlloc` | `pPool->ModifyAccess` 前后 | blocking/stream/status | access command cost |
| `MEM_ALLOC_RESULT_POINT` | `mu_memory.cpp` | 设置 `*dptr` 后 | dptr/virtAddr/status | api result |
| `MEM_FREE_LOOKUP_SPAN` | `muapiMemFreeAsync` | `GetMemoryByDevicePointer` 前后 | dptr/status/memory_type | lookup cost |
| `MEM_FREE_TYPE_DECISION` | `muapiMemFreeAsync` | switch 分支后 | memory_type/path | cost attribution |
| `MEM_POOL_DISABLE_ACCESS_SPAN` | `Stream::AsyncMemFree` | `pPool->DisableAccess` 前后 | virt/physical/blocking/status | free access cost |
| `MEM_FREE_CALLBACK_COMMAND_CREATE_SPAN` | `Stream::AsyncMemFree` | `CallbackCommand` 创建前后 | virt/pool/command_id | delayed free relation |
| `MEM_FREE_CALLBACK_RELATION` | `Stream::AsyncMemFree` | lambda 设置后 | virt/command/pool | memory -> callback command |

## 6. Memcpy / memset / memory transfer / memory atomic 家族

### 6.1 API 范围

Runtime：`MUSA-Runtime/src/musa_memory.cpp`

- `musaapiMemcpy`
- `musaapiMemcpyAsync`
- `musaapiMemcpy2D/2DAsync`
- `musaapiMemcpy3D/3DAsync/3DPeerAsync`
- `musaapiMemcpyPeer/PeerAsync`
- `musaapiMemset/MemsetAsync`
- `musaapiMemset2D/2DAsync`
- `musaapiMemset3D/3DAsync`
- `musaapiMemoryAtomicAsync`
- `musaapiMemoryAtomicValueAsync`
- `musaapiMemoryTransfer/Async`

Driver：`linux-ddk/musa/src/driver/mu_memory.cpp`

- `muapiMemcpyHtoD_v2/HtoDAsync_v2`
- `muapiMemcpyDtoH_v2/DtoHAsync_v2`
- `muapiMemcpyDtoD_v2/DtoDAsync_v2`
- `muapiMemcpy/MemcpyAsync`
- `muapiMemcpy2D_v2/2DAsync_v2`
- `muapiMemcpy3D_v2/3DAsync_v2`
- `muapiMemcpyPeer/PeerAsync`
- `muapiMemsetD8_v2/D8Async`
- `muapiMemsetD32_v2/D32Async`
- `muapiMemsetD2D8_v2/D2D8Async`

### 6.2 端到端执行路径

#### 6.2.1 2D/3D memcpy

`muapiMemcpy2D_v2`：

1. `InitPlatform()`。
2. `MUSA_MEMCPY3D_PEER copy3D{}`。
3. `GetMemcpy3DFrom2D(copy3D, *pCopy)`。
4. `Musa::Context::GeneralMemcpy(TlsCtxTop(), nullptr, copy3D, true)`。

`muapiMemcpy2DAsync_v2`：

1. `InitPlatform()`。
2. `GetMemcpy3DFrom2D(copy3D, *pCopy)`。
3. `Context::GeneralMemcpy(TlsCtxTop(), hStream, copy3D, false)`。

`muapiMemcpy3D_v2`：

1. `InitPlatform()`。
2. `MUSA_MEMCPY3D_PEER copy3dPeer{}`。
3. `GetMemcpy3DPeer(copy3dPeer, *pCopy)`。
4. `Context::GeneralMemcpy(TlsCtxTop(), nullptr, copy3dPeer, true)`。

Core path：

1. `Context::GeneralMemcpy`。
2. 参数/地址 kind 解析。
3. stream 选择。
4. 根据 copy kind 选择 host memcpy 或 device command。
5. device path 创建 `MemcpyCommand` / `AsyncMemcpyCommand` / `MemoryTransferCommand`。
6. `Stream::QueueCommand`。
7. command submit：
   - `AsyncMemcpyCommand.cpp::SubmitToQueue`
   - `memoryTransferCommand.cpp::SubmitToQueue`
   - 最终进入 `Command::SubmitToQueue` 或 command-specific HAL submit。
8. `Command::SubmitToQueue` 设置 wait/signal semaphore、submission id，并调用 `MUpti::AssignSubmissionToCorrelation`。
9. `Hal::IQueue::Submit` -> `M3D Queue::Submit` -> `OsSubmit`。

#### 6.2.2 memset

`muapiMemsetD8_v2`：

1. `InitPlatform()`。
2. 构造 `MUSA_MEMSET_NODE_PARAMS memsetParams = {dstDevice, N, uc, sizeof(unsigned char), N, 1}`。
3. `Context::GeneralMemset(TlsCtxTop(), nullptr, memsetParams, true)`。

`muapiMemsetD32Async`：

1. `InitPlatform()`。
2. 构造 `MUSA_MEMSET_NODE_PARAMS`。
3. `Context::GeneralMemset(TlsCtxTop(), hStream, memsetParams, false)`。

Core path：

1. `context.cpp::Context::GeneralMemset`。
2. `Stream::CmdMemset`。
3. 创建 `MemsetCommand`。
4. `Stream::QueueCommand`。
5. `memsetCommand.cpp::SubmitToQueue`。
6. `Command::SubmitToQueue`。
7. HAL/M3D/OS submit。

### 6.3 当前已有 MUPTI 插点

- memcpy：`tracepoints.h::EnterMemcpy/ExitMemcpy`。
- host memcpy：`StartHostMemcpy/StopHostMemcpy`。
- memset：`EnterMemset`。
- memory transfer：`EnterMemoryTransferV2`。
- memory atomic：`EnterMemoryAtomicV2`。
- memory atomic value：`EnterMemoryAtomicValue`。
- submission relation：`Command::SubmitToQueue` 中 `MUpti::AssignSubmissionToCorrelation(uniqueId, GetSubId())`。
- graph memop：`RegisterGraphMemcpy/RegisterGraphMemset/RegisterGraphMemoryTransfer/RegisterGraphMemoryAtomic/RegisterGraphMemoryAtomicValue`。

缺口：

- 2D -> 3D copy descriptor normalize 不可见。
- `GeneralMemcpy/GeneralMemset` 的地址属性解析、copy kind 决策、host/device path 决策不可见。
- command 创建、merge/拆分、queue wait、build cmd buffer 不可见。
- host path 与 device path 的分支原因不可见。

### 6.4 新增 ModelEvent 插点

| 事件 | 文件/函数 | begin/end 位置 | payload | relation/cost |
|---|---|---|---|---|
| `COPY_DESCRIPTOR_NORMALIZE_SPAN` | `mu_memory.cpp::muapiMemcpy2D*/3D*` | `GetMemcpy3DFrom2D/GetMemcpy3DPeer` 前后 | width/height/depth/src/dst kind/status | descriptor cost |
| `COPY_GENERAL_MEMCPY_SPAN` | `context.cpp::Context::GeneralMemcpy` | 函数入口/出口 | blocking/stream/copyKind/bytes/status | API -> copy engine |
| `COPY_KIND_DECISION` | `Context::GeneralMemcpy` | kind 决策后 | srcKind/dstKind/path/peer | branch reason |
| `COPY_COMMAND_CREATE_SPAN` | command 创建处 | make command 前后 | command_type/bytes/stream | command cost |
| `HOST_COPY_SPAN` | host copy path | host copy 前后 | bytes/kind | CPU copy cost，复用 Start/StopHostMemcpy |
| `MEMSET_PARAM_NORMALIZE_SPAN` | `mu_memory.cpp::muapiMemset*` | memsetParams 构造前后 | element_size/N/pitch/height | descriptor cost |
| `MEMSET_GENERAL_SPAN` | `context.cpp::Context::GeneralMemset` | 函数入口/出口 | bytes/blocking/stream/status | memset path cost |
| `MEMSET_COMMAND_CREATE_SPAN` | `Stream::CmdMemset` | command 创建前后 | bytes/value/command_id | command relation |
| `MEMOP_SUBMISSION_RELATION` | `Command::SubmitToQueue` | `AssignSubmissionToCorrelation` 后 | correlation/submission/from_graph | relation quality |

## 7. Stream / Event / Synchronization 家族

### 7.1 API 范围

Runtime：

- `MUSA-Runtime/src/musa_stream.cpp`
- `MUSA-Runtime/src/musa_event.cpp`
- `MUSA-Runtime/src/musa_device.cpp::musaapiDeviceSynchronize`

Driver：

- `linux-ddk/musa/src/driver/mu_stream.cpp`
- `linux-ddk/musa/src/driver/mu_event.cpp`
- `linux-ddk/musa/src/driver/mu_context.cpp::muapiCtxSynchronize`

关键 API：

- `muapiStreamCreate/CreateWithPriority/Destroy/Query/Synchronize/WaitEvent`
- `muapiEventCreate/Record/Synchronize/Destroy/Query/ElapsedTime`
- `muapiCtxSynchronize`

### 7.2 端到端执行路径

#### 7.2.1 Event create

`muapiEventCreate(phEvent, Flags)`：

1. `InitPlatform()`。
2. 校验 `phEvent`。
3. 校验 flags：只允许 default/blocking/disable timing/interprocess 组合。
4. `TlsCtxTop()`。
5. `pContext->CreateEvent(&pEvent, Flags)`。
6. 成功后 `*phEvent = pEvent`。

#### 7.2.2 Event synchronize

`muapiEventSynchronize(hEvent)`：

1. `InitPlatform()`。
2. 校验 `hEvent`。
3. `TlsCtxTop()`。
4. `ICast<IEvent>(hEvent)`。
5. `pContext->ValidateEvent(pEvent)`。
6. `pEvent->Synchronize()`。
7. 内部会进入 wait path，当前 MUPTI 可通过 `RegisterEventSynchronize/Start/Stop` 记录同步 activity。

#### 7.2.3 Stream wait event

`muapiStreamWaitEvent(hStream, hEvent, Flags)`：

1. `InitPlatform()`。
2. 校验 `hEvent`。
3. `TlsCtxTop()`。
4. `Context::InfoStream(context, hStream)`。
5. `ICast<IEvent>(hEvent)`。
6. `pContext->ValidateEvent(pEvent)`。
7. 构造 `WaitEventParameter`。
8. `pStream->CmdWaitEvent(param, false)`。
9. Core 创建 wait/record command。
10. `Stream::QueueCommand`。
11. `RecordCommand::Submit` -> `SubmitToQueue` -> HAL/M3D。

#### 7.2.4 Stream query / synchronize

`muapiStreamQuery(hStream)`：

1. `InitPlatform()`。
2. `TlsCtxTop()`。
3. `Context::InfoStream(...)`。
4. `pStream->Query()`。

`muapiStreamSynchronize` / `muapiCtxSynchronize`：

1. API 校验和 stream/context 查找。
2. 注册 MUPTI synchronization context。
3. Start sync。
4. 调用 stream/context wait finish。
5. Stop sync。

### 7.3 当前已有 MUPTI 插点

- `RegisterEventSynchronize/StartEventSynchronize/StopEventSynchronize`
- `RegisterStreamWaitEvent/StartStreamWaitEvent/StopStreamWaitEvent`
- `RegisterStreamSynchronize/StartStreamSynchronize/StopStreamSynchronize`
- `RegisterContextSynchronize/StartContextSynchronize/StopContextSynchronize`
- `CreateStream/DestroyStream`
- `CreateContext/DestroyContext`
- API callback/activity

缺口：

- validate event/context/stream 成本不可见。
- sync wait reason 不可见：等待 GPU、等待 async submit queue、等待 callback、错误状态、已完成 fast path 无法区分。
- `Stream::QueueCommand` 的 wait event command 创建/排队不可见。
- event create/destroy 的 context critical section 不可见。

### 7.4 新增 ModelEvent 插点

| 事件 | 文件/函数 | begin/end 位置 | payload | relation/cost |
|---|---|---|---|---|
| `EVENT_VALIDATE_SPAN` | `mu_event.cpp::muapiEvent*` | event/context validate 前后 | event/context/status | validation cost |
| `EVENT_CREATE_SPAN` | `muapiEventCreate` | `pContext->CreateEvent` 前后 | flags/event/status | event create cost |
| `EVENT_SYNC_WAIT_SPAN` | event sync core wait | wait 前后 | event/wait_mode/status | wait cost |
| `STREAM_RESOLVE_SPAN` | `mu_stream.cpp` | `Context::InfoStream` 前后 | hStream/stream/status | stream lookup |
| `STREAM_WAIT_EVENT_COMMAND_CREATE_SPAN` | `Stream::CmdWaitEvent` | command 创建前后 | stream/event/flags/command_id | event dependency command |
| `STREAM_QUERY_SPAN` | `muapiStreamQuery` / `Stream::Query` | Query 前后 | stream/status | query cost |
| `STREAM_WAIT_FINISH_SPAN` | `Stream::Synchronize` | wait finish 前后 | stream/pending_count/status | synchronization cost |
| `SYNC_WAIT_REASON` | sync wait path | wait 结束后 point | reason/gpu_done/queue_empty/last_error | model attribution |

## 8. Graph 家族

### 8.1 API 范围

Runtime：`MUSA-Runtime/src/musa_graph.cpp`

- `musaapiGraphAddKernelNode`
- `musaapiGraphAddMemcpyNode`
- `musaapiGraphAddMemsetNode`
- `musaapiGraphAddMemAllocNode`
- `musaapiGraphAddMemFreeNode`
- `musaapiGraphCreate/Destroy`
- `musaapiGraphInstantiate/InstantiateWithFlags/InstantiateWithParams`
- `musaapiGraphLaunch`
- `musaapiGraphExecUpdate`

Driver：`linux-ddk/musa/src/driver/mu_graph.cpp`

- `muapiGraphCreate/Destroy`
- `muapiGraphAddKernelNode`
- `muapiGraphAddMemcpyNode`
- `muapiGraphAddMemsetNode`
- `muapiGraphAddMemAllocNode`
- `muapiGraphAddMemFreeNode`
- `muapiGraphInstantiate_v2`
- `muapiGraphInstantiateWithFlags`
- `muapiGraphInstantiateWithParams`
- `muapiGraphLaunch`
- `muapiGraphExecUpdate`

### 8.2 端到端执行路径

#### 8.2.1 Graph create

`muapiGraphCreate(phGraph, flags)`：

1. `InitPlatform()`。
2. 校验 `phGraph != nullptr && flags == 0`。
3. `TlsCtxTop()`。
4. `pContext->CreateGraph(&pGraph)`。
5. `Context::CreateGraph`：
   - `auto graph = new Graph(this)`。
   - `WriteLockedAccessor<CriticalBase> ctxCrit(m_CriticalData)`。
   - `ctxCrit->AddGraph(graph)`。
   - `*ppGraph = graph`。

#### 8.2.2 Graph instantiate

`muapiGraphInstantiate_v2(phGraphExec, hGraph, ...)`：

1. `InitPlatform()`。
2. `TlsCtxTop()`。
3. `pContext->ValidateGraph(ICast<IGraph>(hGraph))`。
4. 校验 `phGraphExec`。
5. `pContext->CreateGraphExec(&pGraphExec, ICast<IGraph>(hGraph))`。
6. `Context::CreateGraphExec`：
   - 如果 `graphUserQ && supportEngineSync && GetPerfGraphExecVersion()==2`，创建 `GraphExec2`。
   - 否则创建 `GraphExec`。
   - `graphExec->Init(pGraph)`。
   - 成功后 `ctxCrit->AddGraphExec(graphExec)`。
   - `*ppGraphExec = graphExec`。

#### 8.2.3 Graph launch

`muapiGraphLaunch(hGraphExec, hStream)`：

1. `InitPlatform()`。
2. 校验 `hGraphExec`。
3. `TlsCtxTop()`。
4. `Context::InfoStream(context, hStream)`。
5. `pContext->ValidateGraphExec(ICast<IGraphExec>(hGraphExec))`。
6. `pStream->CmdLaunchGraph(ICast<IGraphExec>(hGraphExec))`。
7. `Stream::CmdLaunchGraph`：
   - version 1：
     - `GraphExec* pGraphExec = static_cast<GraphExec*>(graphExec)`。
     - 创建 `GraphCommand(this, pGraphExec)`。
     - `pGraphExec->SetCommand(graphCommand)`。
     - `GraphCommand::SetGraphID(pGraphExec->GetGraph()->GetCorId())`。
     - `m_ParentCtx->ResolveDependencyAndQueueCommand(std::move(graphCommand), this, false)`。
   - version 2：进入 `GraphExec2` user queue 路径。
8. `GraphCommand::ExecuteImpl`：
   - 遍历 graph submissions。
   - `MUSA_SUBMISSION_DEVICE`：必要时 `m_GraphExec->PrepareSubmission(submission)`，然后 `universalMgr->CmdSubmission(...)`。
   - `MUSA_SUBMISSION_HOST`：`universalMgr->CmdCallback(...)`。
   - `MUSA_SUBMISSION_HOST_DEVICE`：`universalMgr->CmdHostDevice(...)`。
   - `MUSA_SUBMISSION_GRAPH`：递归 `GraphCommand::ExecuteImpl` 后 `universalMgr->CmdGraph(...)`。
   - `MUSA_SUBMISSION_CONDITIONAL`：conditional path。
9. UniversalManager 进入 command/HAL/M3D submit。

### 8.3 当前已有 MUPTI 插点

- graph trace：`RegisterGraphTrace/RegisterGraphTraceV2/MarkGraphTraceBegin/MarkGraphTraceEnd`。
- graph node activity：
  - `RegisterGraphKernel`
  - `RegisterGraphMemcpy`
  - `RegisterGraphMemset`
  - `RegisterGraphMemoryAtomic`
  - `RegisterGraphMemoryAtomicValue`
  - `RegisterGraphMemoryTransfer`
- `MUPTI_ACTIVITY_KIND_GRAPH_TRACE` 已启用。
- `MUPTI/src/api/internal.cpp` 导出：
  - `muptiGraphGetKickFromCorrelationId`
  - `muptiGraphGetSubmissionIdsFromCorrelationId`

缺口：

- graph create/add node/instantiate/update 的 CPU 建图成本不可见。
- `GraphExec` vs `GraphExec2` 选择原因不可见。
- `graphExec->Init` 内部编译/拓扑排序/参数复制成本不可见。
- `GraphCommand::ExecuteImpl` 中不同 submission type 的执行成本不可见。
- child graph/conditional graph 的递归关系需要 ModelEvent relation 增强。

### 8.4 新增 ModelEvent 插点

| 事件 | 文件/函数 | begin/end 位置 | payload | relation/cost |
|---|---|---|---|---|
| `GRAPH_CREATE_SPAN` | `mu_graph.cpp::muapiGraphCreate` / `Context::CreateGraph` | `CreateGraph` 前后 | context/graph/status | graph create cost |
| `GRAPH_ADD_NODE_SPAN` | `muapiGraphAdd*Node` | node add 前后 | graph/node_type/deps/status | graph build cost |
| `GRAPH_NODE_PARAM_NORMALIZE_SPAN` | `muapiGraphAddMemcpy/Memset/KernelNode` | params normalize 前后 | node_type/bytes/grid/block | descriptor cost |
| `GRAPH_INSTANTIATE_SPAN` | `muapiGraphInstantiate*` / `Context::CreateGraphExec` | API enter 到 Init 后 | graph/exec/version/status | instantiate cost |
| `GRAPH_EXEC_VERSION_DECISION` | `Context::CreateGraphExec` | GraphExec/GraphExec2 分支后 | graphUserQ/supportEngineSync/perfVersion | branch reason |
| `GRAPH_EXEC_INIT_SPAN` | `Context::CreateGraphExec` | `graphExec->Init` 前后 | graph/node_count/status | compile/topology cost |
| `GRAPH_LAUNCH_SPAN` | `muapiGraphLaunch` | validate 到 `CmdLaunchGraph` 返回 | graphExec/stream/status | launch CPU cost |
| `GRAPH_COMMAND_CREATE_SPAN` | `Stream::CmdLaunchGraph` | `GraphCommand` 创建前后 | graph/graphExec/command_id | graph -> command |
| `GRAPH_SUBMISSION_EXECUTE_SPAN` | `GraphCommand::ExecuteImpl` | 每个 submission 分支前后 | submission_type/index/kick_count/status | graph execution cost |
| `GRAPH_CHILD_RELATION` | child graph 分支 | recursiveCall 创建后 | parent_graph/child_graph/submission | relation recall |

## 9. Context / Device / Resource / Green Context 家族

### 9.1 API 范围

Driver：

- `linux-ddk/musa/src/driver/mu_context.cpp`
  - `muapiCtxSynchronize`
  - `muapiCtxCreate_v2`
  - `muapiCtxDestroy_v2`
  - `muapiCtxPushCurrent_v2`
  - `muapiCtxGetCurrent`
  - `muapiCtxSetCurrent`
  - `muapiDevicePrimaryCtxRetain`
  - `muapiDevicePrimaryCtxRelease_v2`
- `linux-ddk/musa/src/driver/mu_device.cpp`
  - `muapiDeviceGet`
  - `muapiDeviceGetCount`
  - `muapiDeviceGetProperties`
  - `muapiDeviceGetAttribute`
  - `muapiDeviceGetDefaultMemPool`
  - `muapiDeviceSetMemPool`
  - `muapiDeviceGetMemPool`
- `linux-ddk/musa/src/driver/mu_greencontext.cpp`
  - `muapiDeviceGetDevResource`
  - `muapiDevSmResourceSplitByCount`
  - `muapiDevResourceGenerateDesc`
  - `muapiGreenCtxCreate`
  - `muapiGreenCtxDestroy`
  - `muapiCtxFromGreenCtx`
  - `muapiGreenCtxStreamCreate`
  - `muapiGreenCtxGetDevResource`

### 9.2 Green Context 端到端执行路径

#### 9.2.1 `muapiDeviceGetDevResource`

1. `InitPlatform()`。
2. 校验 `resource != nullptr`。
3. `Platform::Get().GetIDeviceView(device)`。
4. `pDevice->GetDeviceResources(resource, type)`。

#### 9.2.2 `muapiDevSmResourceSplitByCount`

1. `InitPlatform()`。
2. 校验 `result/nbGroups/input/remaining`。
3. 从 `input->_internal_padding` 取 `device/mpx_mask`。
4. `Platform::Get().GetIDeviceView(device)`。
5. 读取 HAL properties：
   - `spuNumPerCore`
   - `uscNumPerSpu`
   - `numGpuCores`
6. `minMPXCount = DivideRoundUp(minCount, mpNumPerMpx)`。
7. 遍历 `nbGroups`。
8. 使用 `hasMpxValid` 从 `mpx_mask` 中按 MPC/MPX 粒度切分。
9. 写入 `result[i].sm.smCount/type/internal padding`。
10. 写入 remaining resource。

#### 9.2.3 `muapiDevResourceGenerateDesc`

1. `InitPlatform()`。
2. 校验 `phDesc/resources`。
3. `new Musa::DevResourceDesc`。
4. `desc->type = resources[0].type`。
5. SM resource：遍历 resources，合并 `padding->mpx_mask`。
6. 成功后 `*phDesc = desc`。

#### 9.2.4 `muapiGreenCtxCreate`

1. `InitPlatform()`。
2. 校验 `phCtx`。
3. `Platform::Get().GetDevice(dev)`。
4. 校验 flags。
5. `pDevice->CreateGreenContext(&pContext, flags, desc)`。
6. 成功后：
   - `*phCtx = pContext`。
   - `IntrusiveCast<GreenContext>`。
   - `Platform::Get().LockedAddCtx(greenContext)`。

#### 9.2.5 `muapiGreenCtxStreamCreate`

1. 校验 green context。
2. `musaGreenCtx->CreateStream(&pStream, flags, priority)`。
3. 成功后 `*phStream = pStream`。
4. stream create 可复用现有 `MUpti::CreateStream`。

### 9.3 当前已有 MUPTI 插点

- Context create/destroy：`CreateContext/DestroyContext`。
- Stream create/destroy：`CreateStream/DestroyStream`。
- Device/resource/GreenContext API 只有外层 Driver callback/activity。
- Device attribute/stream/memory pool activity kind 当前没有作为 public activity 启用。

缺口：

- device resource 查询、SM split、resource desc 生成、GreenContext create/destroy 内部路径不可见。
- GreenContext 资源 mask、SM count、flags 与 context/stream relation 不可见。
- primary context retain/release 与 LockedAddCtx/ValidateContext critical section 成本不可见。

### 9.4 新增 ModelEvent 插点

| 事件 | 文件/函数 | begin/end 位置 | payload | relation/cost |
|---|---|---|---|---|
| `DEVICE_LOOKUP_SPAN` | `mu_device.cpp/mu_greencontext.cpp` | `GetDevice/GetIDeviceView` 前后 | device/status | device lookup cost |
| `DEVICE_ATTRIBUTE_QUERY_SPAN` | `muapiDeviceGetAttribute/GetProperties` | HAL/device property 查询前后 | attrib/value/status | property cost |
| `DEV_RESOURCE_QUERY_SPAN` | `muapiDeviceGetDevResource` | `GetDeviceResources` 前后 | device/type/resource_mask/status | resource query |
| `DEV_SM_RESOURCE_SPLIT_SPAN` | `muapiDevSmResourceSplitByCount` | split loop 前后 | minCount/nbGroups/mpxMask/resultMask | resource split cost |
| `DEV_RESOURCE_DESC_CREATE_SPAN` | `muapiDevResourceGenerateDesc` | desc new/merge 前后 | nbResources/type/mpxMask/status | desc cost |
| `GREEN_CTX_CREATE_SPAN` | `muapiGreenCtxCreate` | `CreateGreenContext` 前后 | device/flags/desc/ctx/status | green ctx create |
| `GREEN_CTX_REGISTER_SPAN` | `muapiGreenCtxCreate` | `LockedAddCtx` 前后 | ctx/device | platform registry cost |
| `GREEN_CTX_STREAM_CREATE_SPAN` | `muapiGreenCtxStreamCreate` | `CreateStream` 前后 | greenCtx/stream/flags/priority/status | green ctx -> stream relation |
| `CTX_VALIDATE_SPAN` | `muapiGreenCtxDestroy/CtxFromGreenCtx` | `ValidateContext` 前后 | ctx/status | validation cost |

## 10. Command submit boundary: Core -> HAL -> M3D -> OS

### 10.1 端到端路径

关键路径：

1. Core command submit：`linux-ddk/musa/src/musa/core/command/command.cpp::Command::SubmitToQueue`
2. HAL queue submit：`linux-ddk/musa/src/hal/m3d/queue.cpp::Queue::Submit`
3. M3D queue submit：`linux-ddk/musa/src/hal/m3d/m3d/src/core/queue.cpp::Queue::Submit`
4. OS submit：`linux-ddk/musa/src/hal/m3d/m3d/src/core/queue.cpp::OsSubmit`

`Command::SubmitToQueue` 真实步骤：

1. 准备 wait semaphore vector。
2. 遍历 `m_SubWaitSemaphoreInfos`：
   - 如果 preferred hardware semaphore 但 wait semaphore 是 timeline，先 `semaphoreInfo.first->Wait(waitInfo)`。
   - 否则 push 到 HAL wait list。
3. 写入 `submitInfo.ppWaitSemaphores/pWaitSemaphoreValues/waitSemaphoreCount`。
4. 准备 signal semaphore vector。
5. 对 memcpy command：
   - 从 parent stream 获取 user queue submission id。
   - 根据 user queue / PfmCheck 决定 `SetSubId(UpdateGlobalSubId())` 或复用 user queue id。
   - graph memcopy 使用 `(correlation << 32) + graphNodeCorId` 作为 unique id。
   - `MUpti::AssignSubmissionToCorrelation(uniqueId, GetSubId())`。
6. 写入 `submitInfo.submissionId/pPerfExperiment/pPfmDumpConfig`。
7. 调用 `pQueue->Submit(submitInfo)`。
8. `HalToMuResult` 转换 status。

`hal/m3d/queue.cpp::Queue::Submit` 真实步骤：

1. 检查 user queue 不支持 fence。
2. 构造 `IM3d::MultiSubmitInfo`。
3. 根据平台限制准备 wait/signal semaphore 数组。
4. 超过限制的 wait semaphore 单独 `WaitQueueSemaphore`。
5. 恢复 queue priority。
6. submit cmd buffer：转换 HAL submit info 为 M3D submit info。
7. 调 `m_M3dQueue->Submit(...)`。

`m3d/src/core/queue.cpp::Queue::Submit` 真实步骤：

1. 为每个 subqueue 调 `SubmitConfig`。
2. `ValidateSubmit(submitInfo)`。
3. 每个 subqueue `QueueContext::PreProcessSubmit(...)`。
4. `CmdBuffer::IncrementSubmitCount()`。
5. fence `AssociateWithContext`。
6. `CaptureSubmitInfoDump`。
7. `OsSubmit(submitInfo, &internalSubmitInfo[0])`。
8. 成功后 `QueueContext::PostProcessSubmit(...)`。

### 10.2 当前已有 MUPTI 插点

- `Command::SubmitToQueue` 对 memcpy command 调 `AssignSubmissionToCorrelation`。
- Launch path 通过 `AssignKernelToKick` 建立 kernel 到 kick 的映射。
- Graph path 有 graph kick/submission 查询接口。

缺口：

- semaphore prepare 成本不可见。
- timeline semaphore fallback wait 不可见。
- HAL submit info -> M3D submit info 翻译不可见。
- M3D Validate/PreProcess/OsSubmit/PostProcess 不可见。
- OS/KMD ioctl 成本不可见。
- submission id 的生成/复用原因不可见。

### 10.3 新增 ModelEvent 插点

| 事件 | 文件/函数 | begin/end 位置 | payload | relation/cost |
|---|---|---|---|---|
| `COMMAND_SUBMIT_SPAN` | `Command::SubmitToQueue` | 函数入口/出口 | command/type/engine/submission/status | command submit cost |
| `SUBMIT_SEMAPHORE_PREPARE_SPAN` | `Command::SubmitToQueue` | wait/signal vector 构造前后 | wait_count/signal_count/timeline_wait_count | dependency cost |
| `SUBMISSION_ID_DECISION` | `Command::SubmitToQueue` | `SetSubId` 后 | old/new/user_queue/pfm/from_graph | relation quality |
| `HAL_QUEUE_SUBMIT_SPAN` | `Command::SubmitToQueue` | `pQueue->Submit` 前后 | queue/submission/status | HAL cost |
| `HAL_M3D_SUBMIT_TRANSLATE_SPAN` | `hal/m3d/queue.cpp::Queue::Submit` | MultiSubmitInfo 构造前后 | cmdBufferCount/wait/signal | HAL translation cost |
| `HAL_EXCEEDED_WAIT_SEMAPHORE_SPAN` | `hal/m3d/queue.cpp::Queue::Submit` | exceeded wait loop | wait_idx/status | semaphore overflow cost |
| `M3D_QUEUE_SUBMIT_SPAN` | `m3d/src/core/queue.cpp::Queue::Submit` | 函数入口/出口 | perSubQueueInfoCount/fenceCount/status | M3D submit cost |
| `M3D_VALIDATE_SUBMIT_SPAN` | `Queue::Submit` | `ValidateSubmit` 前后 | queue/cmdBufferCount/status | validation cost |
| `M3D_PREPROCESS_SUBMIT_SPAN` | `Queue::Submit` | `PreProcessSubmit` 前后 | subqueue/cmdBufferCount/status | preprocess cost |
| `OS_SUBMIT_SPAN` | `Queue::Submit` | `OsSubmit` 前后 | queue/internalSubmitInfo/status | OS boundary |
| `M3D_POSTPROCESS_SUBMIT_SPAN` | `Queue::Submit` | `PostProcessSubmit` 前后 | subqueue/status | postprocess cost |

## 11. 统一落地矩阵

| API 家族 | 当前 MUPTI 覆盖 | M2 必加 ModelEvent | 优先级 |
|---|---|---|---|
| launch | Runtime/Driver callback, kernel activity, queued/kick relation | validate、stream resolve、dispatch create、queue、submit、HAL/M3D | P0 |
| module/function | API callback only | file read、fatbin load、function lookup、function attr | P1 |
| alloc/free/pool | API callback only | pool select、create virt、physical init、bind、modify/disable access、callback command | P0 |
| memcpy/memset | memcpy/memset/activity/submission relation | descriptor normalize、kind decision、command create、host/device branch | P0 |
| memory atomic/transfer | activity hook exists | command create、descriptor normalize、submit relation | P1 |
| stream/event/sync | sync activity、stream create/destroy | validate、wait reason、wait finish、wait-event command | P0 |
| graph | graph trace/node activity | graph build、instantiate、exec version decision、submission execute | P0 |
| context/device/resource | context create/destroy, API callback | device lookup、attribute query、resource split、GreenContext create | P1 |
| submit boundary | partial correlation mapping | command submit、semaphore prepare、HAL/M3D/OS submit spans | P0 |

## 12. 事件关系输出

M2 必须能输出以下 relation：

1. `runtime_api -> driver_api`：复用现有 correlation。
2. `driver_api -> model_span`：所有 ModelEvent 携带 `correlation_id`。
3. `driver_api -> stream`：`STREAM_RESOLVE_SPAN`。
4. `driver_api -> command`：`*_COMMAND_CREATE_SPAN`。
5. `command -> previous_command`：`COMMAND_PREV_LINK_RELATION`。
6. `command -> stream_queue`：`STREAM_QUEUE_COMMAND_SPAN`。
7. `command -> submission`：`SUBMISSION_ID_DECISION` / existing `AssignSubmissionToCorrelation`。
8. `kernel -> kick/submission`：复用 `AssignKernelToKick`。
9. `graph -> graph_exec -> graph_command -> graph_submission -> child_graph/node`。
10. `memory -> pool -> physical_memory -> access_command/callback_command`。

## 13. 验证方案

### 13.1 最小 workload

1. launch：单 kernel launch + stream sync。
2. module：module load + get function + launch。
3. alloc/free：`muMemAllocAsync` / `muMemAllocFromPoolAsync` / `muMemFreeAsync`。
4. memcpy/memset：2D/3D copy、async copy、memset async。
5. stream/event：event create/record/sync、stream wait event、stream query/sync。
6. graph：create graph、add kernel/memcpy/memset node、instantiate、launch、exec update。
7. GreenContext：resource query、SM split、desc generate、green ctx create、green stream create、destroy。

### 13.2 验收指标

- trace off overhead：仅 ready branch，无 allocation/format/log。
- trace on relation recall：
  - API -> command >= 95%
  - command -> submission >= 95%
  - graph node -> activity >= 95%
  - memop -> submission >= 95%
- 每类 API 至少能拆出：
  - validate/lookup
  - descriptor normalize 或 command create
  - queue/dependency
  - submit/HAL/M3D/OS
  - wait/sync reason
- event quality report 可发现：
  - missing correlation
  - missing command id
  - missing submission id
  - span begin/end mismatch
  - dropped records

## 14. 建议的代码落地顺序

### Step 1：MUPTI private ModelEvent collector

修改范围：

- `MUPTI/src/core`：新增 private model event record、buffer dump、drop stats。
- `MUPTI/src/injection/hooks.h` / `injection.cpp`：新增 ModelEvent function pointer。
- `linux-ddk/musa/src/driver/mupti/tracepoints.h`：新增 `EmitModelEvent` fast wrapper。

### Step 2：P0 command/submit 插点

先插稳定核心路径：

- `Stream::QueueCommand`
- `Command::SubmitToQueue`
- `hal/m3d/queue.cpp::Queue::Submit`
- `m3d/src/core/queue.cpp::Queue::Submit`

### Step 3：P0 API family 插点

- alloc/free/pool
- memcpy/memset
- stream/event/sync
- graph instantiate/launch
- launch command create

### Step 4：P1 resource/module/device 插点

- module/function
- context/device
- GreenContext resource

### Step 5：工具链消费

- `msight-system`：展示 API -> ModelEvent -> Activity 的时间线。
- `msight-compute`：把 launch/module/function/pool/submit 软件成本接入白盒模型。
- OKR 输出：`api_cost_breakdown`、`relation_recall_report`、`event_quality_report`、`overhead_report`。

## 15. 总结

M2 不是简单增加更多 API callback。现有 MUPTI 已能回答“哪个 API 慢、哪个 kernel/memcpy/activity 慢”，但不能回答“Driver 源码内部哪一步慢”。M2 应以 private ModelEvent 补齐 API 到 command、stream queue、dependency、pool、graph、HAL/M3D/OS submit 的内部 span，并复用现有 MUPTI 的 correlation、activity buffer 和 submission/kick 关系。这样才能形成可验证、可回归、可落地的白盒软件性能模型。
