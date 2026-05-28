# M1 API 执行流程与 ModelEvent 插点分析

## 1. 范围与结论

M1 覆盖四条 Runtime/Driver API 对：

```text
musaLaunchKernel        / muLaunchKernel
musaMalloc              / muMemAlloc_v2
musaFree                / muMemFree_v2
musaStreamSynchronize   / muStreamSynchronize
```

当前源码里已经具备三类可复用能力：

1. Runtime wrapper 已统一经过 `ApiTrace` / `ApiInvocationGuard`，能生成 Runtime API enter/exit 和 correlation id。
2. Driver wrapper / Tools callback / MUPTI activity 已能覆盖 Driver API 外层 enter/exit。
3. DDK 内已有部分 MUPTI tracepoint，例如 stream synchronize、context synchronize、submission/correlation 关联。

M1 新增 ModelEvent 的重点不是重建 MUPTI，而是在 Driver/Core 内部补齐以下白盒阶段：

```text
command create
function/module/device validation
context/stream/memory object lookup
dependency resolve
stream queue
command merge decision
command build
command submit
HAL queue submit
memory pool allocate/free/map/unmap
sync wait reason
waited command/activity relation
```

## 2. 公共 Runtime 入口

### 2.1 现有 Runtime wrapper 基础

所有 M1 Runtime API 都在 `MUSA-Runtime/src/musa_wrappers_generated.cpp` 里创建：

```text
ApiTrace trace(status, "api_name", cbid, ...)
```

`ApiTrace` 位于：

```text
MUSA-Runtime/src/internal.h
```

关键流程：

```text
ApiTrace::ApiTrace
  -> g_ExportTable.Init()
  -> ApiInvocationGuard(apiName, cbid, args...)
      -> GetDuringApiInvocation()
      -> SetDuringApiInvocation(true)
      -> IncrCorrelationId()
      -> ThreadInfo::SetApiSeqNum(correlation)
  -> MUpti::EnterRuntimeApi(cbid, isInvocation, args...)

ApiTrace::~ApiTrace
  -> ApiInvocationGuard::SetReturnValue(status)
  -> MUpti::ExitRuntimeApi(context, status)
```

因此 M1 不需要在每个 wrapper 重复实现 Runtime span。建议新增统一 ModelEvent 或 relation hook 的位置是：

| 插点 | 文件 | 事件 | 说明 |
| --- | --- | --- | --- |
| Runtime API begin | `internal.h::ApiTrace` 构造 | `RUNTIME_API_SPAN begin` | 复用 `cbid`、api name、correlation id |
| Runtime API end | `internal.h::ApiTrace` 析构 | `RUNTIME_API_SPAN end` | 记录 status、duration |
| Runtime->Driver relation | `musaapi*` 调 Driver export table 前后 | `RUNTIME_DRIVER_RELATION` | 只在 M1 四个 API 的 runtime impl 先加 |

### 2.2 Runtime 侧 M1 API wrapper

| Runtime API | wrapper 位置 | Runtime impl | Driver 调用 |
| --- | --- | --- | --- |
| `musaLaunchKernel` | `musa_wrappers_generated.cpp` | `musa_module.cpp::musaapiLaunchKernel` | `module->muLaunchKernel` |
| `musaMalloc` | `musa_wrappers_generated.cpp` | `musa_memory.cpp::musaapiMalloc` | `memory->muMemAlloc_v2` |
| `musaFree` | `musa_wrappers_generated.cpp` | `musa_memory.cpp::musaapiFree` | `memory->muMemFree_v2` |
| `musaStreamSynchronize` | `musa_wrappers_generated.cpp` | `musa_stream.cpp::musaapiStreamSynchronize` | `stream->muStreamSynchronize` |

## 3. `musaLaunchKernel` / `muLaunchKernel`

### 3.1 执行流程

源码路径：

```text
用户调用 musaLaunchKernel
  -> MUSA-Runtime/src/musa_wrappers_generated.cpp::musaLaunchKernel
  -> Runtime::ApiTrace / ApiInvocationGuard
  -> MUSA-Runtime/src/musa_module.cpp::musaapiLaunchKernel
      -> InitPlatformAndDevice(0)
      -> Driver stream->muStreamGetDevice(stream, &dev)
      -> ProgramState::GetFunction(host_func, &f, &mod, dev)
      -> Driver device->muDeviceGetProperties(&devProp, dev)
      -> grid/block/sharedMem 参数合法性检查
      -> ProgramState::UpdateTexRefAttr(mod, dev)
      -> Driver module->muLaunchKernel(f, grid, block, sharedMem, stream, args, nullptr)
  -> linux-ddk/musa/src/driver/mu_module.cpp::muapiLaunchKernel
      -> InitPlatform()
      -> ICast<IFunction>(f)
      -> KernelReplayHandler(pFunction, hStream, kernelParams, extra)
      -> 填充 MUSA_KERNEL_NODE_PARAMS
      -> Context::GeneralLaunchKernel(TlsCtxTop(), hStream, nodeParams, ..., launchBlocking)
  -> linux-ddk/musa/src/musa/core/context.cpp::Context::GeneralLaunchKernel
      -> Context::CreateKernelNode(kernelParam, &pGraphNode, ...)
      -> Stream::CmdLaunchKernel(pGraphNode, wait)
  -> linux-ddk/musa/src/musa/core/stream.cpp::Stream::CmdLaunchKernel
      -> std::make_shared<DispatchCommand>(...)
      -> Context::ResolveDependencyAndQueueCommand(command, stream, blocking)
  -> linux-ddk/musa/src/musa/core/context.cpp::ResolveDependencyAndQueueCommand
      -> default/barrier/blocking stream dependency collection
      -> stream current dependency collection
      -> Stream::QueueCommand(command)
      -> optional commandRef->Wait() when blocking or pfm enabled
  -> linux-ddk/musa/src/musa/core/stream.cpp::Stream::QueueCommand
      -> async capacity wait
      -> GetLastError()
      -> m_AsyncCount++
      -> lock m_SubmitMtx
      -> command->ChoosePerfEngine(m_LastCommand)
      -> command->SetPrevCommand(m_LastCommand)
      -> m_LastCommand = command
      -> command->SetStatus(queued)
      -> optional MUpti::MarkKernelQueued(corId)
      -> m_CommandList.push_back(command)
      -> notify submit thread
  -> linux-ddk/musa/src/musa/core/stream.cpp::Stream::AsyncSubmit
      -> command merge list scheduling
      -> submission id 分配
      -> AssignKernelToKick / AssignSubmissionToCorrelation
      -> command->Submit()
  -> linux-ddk/musa/src/musa/core/command/dispatchCommand.cpp::DispatchCommand::Build
      -> InitPfm()
      -> GetHalCmdBuffer()
      -> CmdBuffer Begin
      -> kernel command buffer build
      -> 内部 memory 分配，如 spill/print/cluster/const data
  -> linux-ddk/musa/src/musa/core/command/dispatchCommand.cpp::DispatchCommand::Submit
      -> Hal::QueueSubmitInfo
      -> Command::SubmitToQueue(cdm queue, submitInfo)
  -> linux-ddk/musa/src/musa/core/command/command.cpp::Command::SubmitToQueue
      -> submit wait semaphore 组装
      -> submit signal semaphore 组装
      -> submitInfo.submissionId = GetSubId()
      -> pQueue->Submit(submitInfo)
  -> HAL/M3D/DRM/KMD
  -> MUPTI kernel activity
```

### 3.2 需要添加的 ModelEvent

| 顺序 | 插点 | 事件 | 类型 | payload 建议 | 成本项 |
| --- | --- | --- | --- | --- | --- |
| L1 | `ApiTrace` 构造/析构 | `RUNTIME_API_SPAN` | span | api cbid、api name、correlation id、status | `runtime_wrapper` |
| L2 | `musaapiLaunchKernel` 调 `muStreamGetDevice` | `RUNTIME_STREAM_DEVICE_QUERY_SPAN` | span | stream、device、status | `runtime_stream_lookup` |
| L3 | `ProgramState::GetFunction` | `RUNTIME_FUNCTION_LOOKUP_SPAN` | span | host func、device、module、function、hit/miss | `runtime_function_lookup` |
| L4 | `muDeviceGetProperties` + 参数检查 | `RUNTIME_LAUNCH_VALIDATE_SPAN` | span | grid/block/sharedMem、status | `runtime_validation` |
| L5 | `musaapiLaunchKernel` 调 `muLaunchKernel` 前后 | `RUNTIME_DRIVER_RELATION` | relation | runtime correlation、driver cbid、function、stream | relation |
| L6 | `muapiLaunchKernel` 入口/出口 | `DRIVER_API_SPAN` | span | function、grid/block、sharedMem、stream、status | `driver_wrapper` |
| L7 | `ICast<IFunction>` | `DRIVER_FUNCTION_CAST_SPAN` | span | function handle、status | `driver_validation` |
| L8 | `KernelReplayHandler` | `KERNEL_REPLAY_HANDLER_SPAN` | span | function、stream、enabled/status | `kernel_replay` |
| L9 | `Context::GeneralLaunchKernel` | `KERNEL_NODE_CREATE_SPAN` | span | context、stream、graph node、grid/block | `kernel_node_create` |
| L10 | `Stream::CmdLaunchKernel` 创建 `DispatchCommand` 后 | `COMMAND_CREATE` | instant | command_id、command ptr、stream、context、correlation id | `command_create` |
| L11 | `ResolveDependencyAndQueueCommand` 入口/出口 | `DEPENDENCY_RESOLVE_SPAN` | span | command_id、stream kind、dependency count、pfm、blocking | `dependency` |
| L12 | `Stream::QueueCommand` 入口/出口 | `STREAM_QUEUE_COMMAND_SPAN` | span | command_id、stream、async_count、capacity、queue size、status | `queue` |
| L13 | `AsyncSubmit` merge list 处理 | `COMMAND_MERGE_DECISION` | instant/counter | command_id、merge list size、command type、primary/secondary、reason | `merge_check` |
| L14 | `DispatchCommand::Build` 入口/出口 | `COMMAND_BUILD_SPAN` | span | command_id、merge level、cmd buffer、status | `command_build` |
| L15 | `DispatchCommand::Build` 内部资源分配 | `COMMAND_INTERNAL_ALLOC_SPAN` | span | alloc type、size、status | `command_build_memory` |
| L16 | `DispatchCommand::Submit` | `COMMAND_SUBMIT_PREPARE_SPAN` | span | command_id、queue、freeSchedule | `submit_prepare` |
| L17 | `Command::SubmitToQueue` 入口/出口 | `COMMAND_SUBMIT_SPAN` | span | command_id、submission_id、wait count、signal count、engine、status | `submit` |
| L18 | `Command::SubmitToQueue` 调 `pQueue->Submit` | `HAL_QUEUE_SUBMIT_SPAN` | span | submission_id、queue、cmdBufferCount、status | `hal_submit` |
| L19 | `AssignKernelToKick` / `AssignSubmissionToCorrelation` | `COMMAND_SUBMISSION_RELATION` | relation | command_id/correlation_id、submission_id | relation |
| L20 | MUPTI relation builder | `SUBMISSION_ACTIVITY_RELATION` | relation | submission_id、activity_id、kernel record | relation |

### 3.3 M1 launch 最小必加点

如果只做第一版，最少加：

```text
COMMAND_CREATE
DEPENDENCY_RESOLVE_SPAN
STREAM_QUEUE_COMMAND_SPAN
COMMAND_MERGE_DECISION
COMMAND_BUILD_SPAN
COMMAND_SUBMIT_SPAN
COMMAND_SUBMISSION_RELATION
```

Runtime `RUNTIME_API_SPAN` 和 Driver `DRIVER_API_SPAN` 优先复用现有 callback/activity，只有 relation 或缺字段不足时再补 ModelEvent。

## 4. `musaMalloc` / `muMemAlloc_v2`

### 4.1 执行流程

```text
用户调用 musaMalloc
  -> MUSA-Runtime/src/musa_wrappers_generated.cpp::musaMalloc
  -> Runtime::ApiTrace / ApiInvocationGuard
  -> MUSA-Runtime/src/musa_memory.cpp::musaapiMalloc
      -> InitPlatformAndDevice(0)
      -> if size != 0: Driver memory->muMemAlloc_v2(devPtr, size)
      -> if size == 0: *devPtr = nullptr
  -> linux-ddk/musa/src/driver/mu_memory.cpp::muapiMemAlloc_v2
      -> InitPlatform()
      -> 参数检查：dptr != nullptr、bytesize != 0
      -> TlsCtxTop()
      -> 填充 MemoryCreateInfo
          type = memoryTypeGeneral
          size = bytesize
          flags = Virtual | DeviceMapped | SubAllocatable
      -> IContext::CreateMemory(&pMemory, createInfo)
      -> *dptr = pMemory->GetDevicePointer()
  -> linux-ddk/musa/src/musa/core/context.cpp::Context::CreateMemory
      -> capture stream 状态检查
      -> std::make_shared<Memory>(this)
      -> Memory::Init(createInfo)
      -> MapToPeers if peer accessible
      -> CriticalBase::AddMemory(pMemory)
      -> MemoryTracker::TrackMemory(memory_sp)
      -> SetMemorySeqID
  -> linux-ddk/musa/src/musa/core/memory.cpp::Memory::Init / GeneralAlloc
      -> createInfo.alloc.size/alignment
      -> if SubAllocatable: Hal().GetMemMgr()->Allocate(...)
      -> else: Hal().CreateMemory(...)
```

注意：同步 `muMemAlloc_v2` 当前不是 `MemoryPool::CreateMemory` 路径，而是 `Context::CreateMemory -> Memory::Init -> Hal mem manager allocate/CreateMemory`。`MemoryPool::CreateMemory/ModifyAccess` 主要覆盖 async alloc 或 explicit pool 路径，M1 若只看 `musaMalloc/muMemAlloc_v2`，不要误把所有成本归到 `MemoryPool::CreateMemory`。

### 4.2 需要添加的 ModelEvent

| 顺序 | 插点 | 事件 | 类型 | payload 建议 | 成本项 |
| --- | --- | --- | --- | --- | --- |
| A1 | `ApiTrace` | `RUNTIME_API_SPAN` | span | cbid、size、correlation、status | `runtime_wrapper` |
| A2 | `musaapiMalloc` 调 `muMemAlloc_v2` | `RUNTIME_DRIVER_RELATION` | relation | runtime correlation、size、driver API | relation |
| A3 | `muapiMemAlloc_v2` 入口/出口 | `DRIVER_API_SPAN` | span | bytesize、dptr、status | `driver_wrapper` |
| A4 | `muapiMemAlloc_v2` 参数检查 | `MEM_ALLOC_VALIDATE_SPAN` | span | bytesize、dptr slot、status | `validation` |
| A5 | `TlsCtxTop()` | `CONTEXT_LOOKUP_SPAN` | span | context ptr、status | `context_lookup` |
| A6 | `MemoryCreateInfo` 填充后 | `MEM_CREATE_INFO` | instant | type、size、flags、alignment | `memory_create_info` |
| A7 | `Context::CreateMemory` 入口/出口 | `CONTEXT_CREATE_MEMORY_SPAN` | span | context、memory type、size、status | `context_create_memory` |
| A8 | capture stream 状态检查 | `CAPTURE_STATE_CHECK_SPAN` | span | active/invalidated count、status | `capture_check` |
| A9 | `Memory::Init` | `MEMORY_INIT_SPAN` | span | memory ptr、type、size、flags、status | `memory_init` |
| A10 | `Hal().GetMemMgr()->Allocate` | `HAL_MEM_ALLOC_SPAN` | span | size、alignment、heap/property、offset、status | `hal_alloc` |
| A11 | `Hal().CreateMemory` fallback | `HAL_CREATE_MEMORY_SPAN` | span | size、property、status | `hal_alloc` |
| A12 | `MapToPeers` | `MEMORY_PEER_MAP_SPAN` | span | peer count、status | `peer_map` |
| A13 | `AddMemory` / `TrackMemory` | `MEMORY_TRACK_SPAN` | span | memory ptr、device ptr、seq id | `memory_track` |
| A14 | 成功返回 device pointer | `API_MEMORY_RELATION` | relation | correlation_id、memory_id、device pointer、size | relation |

### 4.3 M1 alloc 最小必加点

```text
CONTEXT_LOOKUP_SPAN
CONTEXT_CREATE_MEMORY_SPAN
MEMORY_INIT_SPAN
HAL_MEM_ALLOC_SPAN 或 HAL_CREATE_MEMORY_SPAN
MEMORY_TRACK_SPAN
API_MEMORY_RELATION
```

如果第一阶段时间紧，`CAPTURE_STATE_CHECK_SPAN` 和 `MEMORY_PEER_MAP_SPAN` 可先作为 payload 字段，不单独成 span。

## 5. `musaFree` / `muMemFree_v2`

### 5.1 执行流程

```text
用户调用 musaFree
  -> MUSA-Runtime/src/musa_wrappers_generated.cpp::musaFree
  -> Runtime::ApiTrace / ApiInvocationGuard
  -> MUSA-Runtime/src/musa_memory.cpp::musaapiFree
      -> InitPlatformAndDevice(0)
      -> Driver memory->muMemFree_v2(devPtr)
  -> linux-ddk/musa/src/driver/mu_memory.cpp::muapiMemFree_v2
      -> InitPlatform()
      -> if dptr != 0
      -> Platform::GetMemoryByDevicePointer(dptr, &offset)
      -> memory type 校验
      -> offset 校验
      -> pMemory->Synchronize()
      -> if memoryTypeVirtual:
          -> TlsCtxTop()
          -> Context::InfoStream(pContext, nullptr)
          -> IStream::CmdMemFree(dptr, true)
      -> else:
          -> IContext::DestroyMemory(pMemory)
  -> linux-ddk/musa/src/musa/core/memory.cpp::Memory::Synchronize
      -> virtual memory 时同步 phys memory context LockedWait
  -> linux-ddk/musa/src/musa/core/stream.cpp::Stream::CmdMemFree
      -> if capture: CaptureMemFree
      -> else: AsyncMemFree
  -> linux-ddk/musa/src/musa/core/stream.cpp::Stream::AsyncMemFree
      -> Platform::GetMemoryByDevicePointer
      -> MemoryPool::DisableAccess
      -> optional MemoryPool::DestroyMemory
      -> enqueue CallbackCommand via ResolveDependencyAndQueueCommand
  -> linux-ddk/musa/src/musa/core/context.cpp::Context::DestroyMemory
      -> RemoveMemory
      -> MemoryTracker::UntrackMemory
  -> linux-ddk/musa/src/musa/core/memory.cpp::Memory::~Memory
      -> if SubAllocatable and no pool: Hal().GetMemMgr()->Free(...)
      -> if pool: pool->Hal()->Free(...)
      -> else: phys tracker cleanup / Hal memory cleanup
```

### 5.2 需要添加的 ModelEvent

| 顺序 | 插点 | 事件 | 类型 | payload 建议 | 成本项 |
| --- | --- | --- | --- | --- | --- |
| F1 | `ApiTrace` | `RUNTIME_API_SPAN` | span | devPtr、correlation、status | `runtime_wrapper` |
| F2 | `musaapiFree` 调 `muMemFree_v2` | `RUNTIME_DRIVER_RELATION` | relation | runtime correlation、ptr、driver API | relation |
| F3 | `muapiMemFree_v2` 入口/出口 | `DRIVER_API_SPAN` | span | dptr、status | `driver_wrapper` |
| F4 | `GetMemoryByDevicePointer` | `MEM_OBJECT_LOOKUP_SPAN` | span | dptr、offset、memory ptr、type、status | `memory_lookup` |
| F5 | memory type/offset 校验 | `MEM_FREE_VALIDATE_SPAN` | span | memory type、offset、status | `validation` |
| F6 | `pMemory->Synchronize()` | `MEMORY_SYNCHRONIZE_SPAN` | span | memory ptr、type、status | `memory_sync` |
| F7 | virtual path `TlsCtxTop/InfoStream` | `FREE_STREAM_LOOKUP_SPAN` | span | context、stream、status | `stream_lookup` |
| F8 | `Stream::CmdMemFree` | `MEM_FREE_COMMAND_CREATE` | instant/span | dptr、stream、blocking、capture status | `free_command_create` |
| F9 | `MemoryPool::DisableAccess` | `MEM_POOL_DISABLE_ACCESS_SPAN` | span | pool、virt、physical、paging count、status | `pool_disable_access` |
| F10 | `Stream::AsyncMemFree` enqueue callback command | `MEM_FREE_CALLBACK_COMMAND_RELATION` | relation | free API correlation、callback command id | relation |
| F11 | `Context::DestroyMemory` | `CONTEXT_DESTROY_MEMORY_SPAN` | span | memory ptr、type、status | `context_destroy_memory` |
| F12 | `MemoryTracker::UntrackMemory` | `MEMORY_UNTRACK_SPAN` | span | memory ptr、device ptr | `memory_untrack` |
| F13 | `Memory::~Memory` Hal free | `HAL_MEM_FREE_SPAN` | span | ptr、size、pool/null、status | `hal_free` |
| F14 | free relation | `API_MEMORY_RELATION` | relation | correlation_id、memory_id、device pointer、size | relation |

### 5.3 M1 free 最小必加点

```text
MEM_OBJECT_LOOKUP_SPAN
MEMORY_SYNCHRONIZE_SPAN
CONTEXT_DESTROY_MEMORY_SPAN 或 MEM_FREE_COMMAND_CREATE
MEM_POOL_DISABLE_ACCESS_SPAN（virtual/pool path）
HAL_MEM_FREE_SPAN
API_MEMORY_RELATION
```

注意 free 有两条主路径：

```text
普通 memory：pMemory->Synchronize -> Context::DestroyMemory -> Memory destructor/HAL free
virtual/pool memory：pMemory->Synchronize -> CmdMemFree -> AsyncMemFree -> MemoryPool::DisableAccess/DestroyMemory -> command queue
```

ModelEvent payload 必须带 `memory_type`，否则 cost breakdown 会把同步 free 和 async/pool free 混在一起。

## 6. `musaStreamSynchronize` / `muStreamSynchronize`

### 6.1 执行流程

```text
用户调用 musaStreamSynchronize
  -> MUSA-Runtime/src/musa_wrappers_generated.cpp::musaStreamSynchronize
  -> Runtime::ApiTrace / ApiInvocationGuard
  -> MUSA-Runtime/src/musa_stream.cpp::musaapiStreamSynchronize
      -> InitPlatformAndDevice(0)
      -> Driver stream->muStreamSynchronize(stream)
  -> linux-ddk/musa/src/driver/mu_stream.cpp::muapiStreamSynchronize
      -> InitPlatform()
      -> TlsCtxTop()
      -> Context::InfoStream(context, hStream)
      -> pStream->Synchronize()
  -> linux-ddk/musa/src/musa/core/stream.cpp::Stream::Synchronize
      -> capture status 检查
      -> MUpti::RegisterStreamSynchronize(ApiSeqNum, this)
      -> MUpti::StartStreamSynchronize(context)
      -> if default stream:
          -> Context::LockedSyncDefaultStream()
      -> else:
          -> Stream::WaitFinish()
      -> MUpti::StopStreamSynchronize(context)
  -> linux-ddk/musa/src/musa/core/stream.cpp::Stream::WaitFinish
      -> LastCommand()
      -> lastCommand ? lastCommand->Wait() : m_LastError
      -> Hal().GetMemMgr()->UpdateUserPools()
      -> sticky error public log
  -> linux-ddk/musa/src/musa/core/command/command.cpp::Command::WaitFinish
      -> choose hardware or timeline semaphore
      -> semaphore->Wait(timeout)
```

### 6.2 当前已有 MUPTI 能力

`linux-ddk/musa/src/driver/mupti/tracepoints.h` 已有：

```text
RegisterStreamSynchronize(correlationId, stream)
StartStreamSynchronize(context)
StopStreamSynchronize(context)
RegisterContextSynchronize(correlationId, ctx)
StartContextSynchronize(context)
StopContextSynchronize(context)
AssignSubmissionToCorrelation(correlationId, submissionId)
```

因此 `musaStreamSynchronize/muStreamSynchronize` 外层 sync activity 可以复用现有机制。M1 新增 ModelEvent 重点是 wait 的内部原因和等待对象。

### 6.3 需要添加的 ModelEvent

| 顺序 | 插点 | 事件 | 类型 | payload 建议 | 成本项 |
| --- | --- | --- | --- | --- | --- |
| S1 | `ApiTrace` | `RUNTIME_API_SPAN` | span | stream、correlation、status | `runtime_wrapper` |
| S2 | `musaapiStreamSynchronize` 调 Driver | `RUNTIME_DRIVER_RELATION` | relation | runtime correlation、stream、driver API | relation |
| S3 | `muapiStreamSynchronize` 入口/出口 | `DRIVER_API_SPAN` | span | hStream、status | `driver_wrapper` |
| S4 | `TlsCtxTop + InfoStream` | `STREAM_LOOKUP_SPAN` | span | context、hStream、pStream、default/null、status | `stream_lookup` |
| S5 | `Stream::Synchronize` capture check | `STREAM_SYNC_VALIDATE_SPAN` | span | capture status、status | `validation` |
| S6 | `Register/Start/StopStreamSynchronize` | 复用 sync activity | activity/span | correlation、stream | `sync_outer` |
| S7 | default stream branch | `SYNC_WAIT_REASON` | instant | reason=`default_stream_wait_all_blocking_streams` | `sync_reason` |
| S8 | non-default branch | `SYNC_WAIT_REASON` | instant | reason=`wait_last_command` | `sync_reason` |
| S9 | `Stream::WaitFinish` 入口/出口 | `STREAM_WAIT_FINISH_SPAN` | span | stream、last_command_id、last_error、status | `sync_wait` |
| S10 | `LastCommand()` 后 | `LAST_COMMAND_LOOKUP` | instant | last command ptr/id/type/status/submission id | `last_command_lookup` |
| S11 | `lastCommand->Wait()` | `COMMAND_WAIT_SPAN` | span | command_id、submission_id、activity_id、status | `command_wait` |
| S12 | `Command::WaitFinish` semaphore wait | `SEMAPHORE_WAIT_SPAN` | span | semaphore type、value、timeout、status | `device_wait` |
| S13 | `UpdateUserPools()` | `MEM_POOL_UPDATE_SPAN` | span | device、pool update status | `pool_update` |
| S14 | sync relation | `API_SYNC_RELATION` | relation | sync correlation、waited command/submission/activity | relation |

### 6.4 M1 sync 最小必加点

```text
STREAM_LOOKUP_SPAN
SYNC_WAIT_REASON
STREAM_WAIT_FINISH_SPAN
LAST_COMMAND_LOOKUP
COMMAND_WAIT_SPAN
SEMAPHORE_WAIT_SPAN
API_SYNC_RELATION
```

## 7. MUPTI private ModelEvent collector 复用点

### 7.1 现有能力

MUPTI 已有：

```text
MUPTI/src/core/process_callback.cpp
  -> ProcessDriverCallback
  -> Runtime/Driver activity start/end

MUPTI/src/core/buffer.cpp
  -> ActivityBuffer::allocate_record/acquire_record
  -> ActivityBufferManager::request_buffer
  -> ActivityBufferManager::flush_buffers
  -> dropped/overflow/flush 质量基础

linux-ddk/musa/src/driver/mupti/tracepoints.h
  -> Stream/Event/Context synchronize hooks
  -> AssignSubmissionToCorrelation
```

### 7.2 M1 应新增的 private hook

建议新增 private ABI，不进入 public MUPTI ABI：

```cpp
bool MuptiModelEventEnabled(uint32_t domain, uint32_t event_id);
void MuptiEmitModelEvent(const ModelEventHeader* header,
                         const void* payload,
                         uint32_t payload_size);
```

挂载路径：

```text
MUPTI injection/export table
  -> Runtime G_MUPTI_RUNTIME_HOOKS
  -> Driver G_MUPTI_DRIVER_HOOKS
  -> tracepoints.h thin wrapper
```

### 7.3 M1 record 字段

ModelEvent header 至少包含：

```text
version
event_type: span_begin/span_end/instant/counter/relation
domain: runtime/driver/memory/stream/command/submit/sync/relation
event_id
timestamp_ns
pid/tid
correlation_id
span_id/parent_id
context_id
stream_id
command_id
submission_id
activity_id
status
payload_size
```

### 7.4 trace off 性能要求

所有内部插点必须是：

```cpp
if (!MuptiModelEventEnabled(domain, event_id)) {
    return;
}
```

或 RAII span 构造时只做一个 ready flag 判断。trace off 不允许：

```text
字符串格式化
堆分配
锁竞争
系统调用
文件写入
复杂 map 查询
```

## 8. M1 插点优先级

### P0：必须有，否则 M1 闭环不成立

| API | P0 ModelEvent |
| --- | --- |
| launch | `COMMAND_CREATE`、`DEPENDENCY_RESOLVE_SPAN`、`STREAM_QUEUE_COMMAND_SPAN`、`COMMAND_BUILD_SPAN`、`COMMAND_SUBMIT_SPAN`、`COMMAND_SUBMISSION_RELATION` |
| alloc | `CONTEXT_CREATE_MEMORY_SPAN`、`MEMORY_INIT_SPAN`、`HAL_MEM_ALLOC_SPAN`、`MEMORY_TRACK_SPAN`、`API_MEMORY_RELATION` |
| free | `MEM_OBJECT_LOOKUP_SPAN`、`MEMORY_SYNCHRONIZE_SPAN`、`CONTEXT_DESTROY_MEMORY_SPAN`/`MEM_FREE_COMMAND_CREATE`、`HAL_MEM_FREE_SPAN`、`API_MEMORY_RELATION` |
| sync | `STREAM_WAIT_FINISH_SPAN`、`LAST_COMMAND_LOOKUP`、`COMMAND_WAIT_SPAN`、`SEMAPHORE_WAIT_SPAN`、`SYNC_WAIT_REASON`、`API_SYNC_RELATION` |

### P1：用于降低 unknown cost

| API | P1 ModelEvent |
| --- | --- |
| launch | `RUNTIME_FUNCTION_LOOKUP_SPAN`、`RUNTIME_LAUNCH_VALIDATE_SPAN`、`KERNEL_REPLAY_HANDLER_SPAN`、`COMMAND_MERGE_DECISION`、`COMMAND_INTERNAL_ALLOC_SPAN`、`HAL_QUEUE_SUBMIT_SPAN` |
| alloc | `CAPTURE_STATE_CHECK_SPAN`、`MEMORY_PEER_MAP_SPAN`、`HAL_CREATE_MEMORY_SPAN` |
| free | `MEM_FREE_VALIDATE_SPAN`、`MEM_POOL_DISABLE_ACCESS_SPAN`、`MEMORY_UNTRACK_SPAN` |
| sync | `STREAM_LOOKUP_SPAN`、`STREAM_SYNC_VALIDATE_SPAN`、`MEM_POOL_UPDATE_SPAN` |

### P2：M2/M3 再补

```text
HAL/M3D/DRM ioctl submit 深层边界
Graph launch/update/instantiate 细分
async alloc/free 完整 pool hit/miss/grow/trim
module load/function cache 全链路
Green Context resource 细分
```

## 9. M1 成本分项输出

### 9.1 launch

```text
T_launch =
  T_runtime_wrapper
+ T_runtime_stream_lookup
+ T_runtime_function_lookup
+ T_runtime_validation
+ T_driver_wrapper
+ T_kernel_node_create
+ T_command_create
+ T_dependency_resolve
+ T_stream_queue
+ T_merge_check
+ T_command_build
+ T_submit
+ T_hal_submit
+ T_unknown
```

### 9.2 alloc

```text
T_alloc =
  T_runtime_wrapper
+ T_driver_wrapper
+ T_context_lookup
+ T_context_create_memory
+ T_memory_init
+ T_hal_alloc
+ T_peer_map
+ T_memory_track
+ T_unknown
```

### 9.3 free

```text
T_free =
  T_runtime_wrapper
+ T_driver_wrapper
+ T_memory_lookup
+ T_memory_sync
+ T_context_destroy_memory or T_free_command
+ T_pool_disable_access
+ T_hal_free
+ T_memory_untrack
+ T_unknown
```

### 9.4 stream sync

```text
T_stream_sync =
  T_runtime_wrapper
+ T_driver_wrapper
+ T_stream_lookup
+ T_sync_outer
+ T_last_command_lookup
+ T_command_wait
+ T_semaphore_wait
+ T_pool_update
+ T_unknown
```

## 10. 验收建议

M1 最小 demo：

```text
musaSetDevice(0)
musaMalloc(&ptr, size)
musaLaunchKernel(kernel, grid, block, args, sharedMem, stream)
musaStreamSynchronize(stream)
musaFree(ptr)
```

必须输出：

```text
api_events.jsonl
model_events.jsonl
activity_events.jsonl
relations.csv
api_cost_breakdown.csv
event_quality_report.md
relation_recall_report.md
overhead_report.md
```

M1 通过条件：

```text
四类 API 均有 Runtime API、Driver API、ModelEvent、relation
launch 能重建 Runtime API -> Driver API -> command -> submission -> kernel activity
sync 能说明 waited command/submission/activity 和 wait reason
alloc/free 能说明 memory object、HAL alloc/free 或 pool path
unknown cost <= 15%
launch relation recall >= 95%
trace off overhead <= 0.1% 或 benchmark 噪声内
API-only overhead <= 1%
internal targeted overhead <= 3%
dropped record / overflow / flush error 必须输出
```
