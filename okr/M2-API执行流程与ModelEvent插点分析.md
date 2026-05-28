# M2 API 执行流程与 ModelEvent 插点分析

## 1. 范围与定位

M2 的目标是把 M1 最小闭环扩展到 Top90 Driver API 候选家族。实际 Top90 以后续 workload 基线统计为准；本文件按当前 OKR 文档中固定的 M2 候选范围做源码路径级分析。

M2 覆盖七类 API：

```text
1. Launch / module / function
2. Memory allocation / free / pool
3. Memcpy / memset / migration / memory command
4. Stream / event / synchronization
5. Graph
6. Context / device / resource / Green Context
7. HAL / M3D / KMD submit boundary
```

M2 和 M1 的关系：

```text
M1：打通 musaLaunchKernel/muLaunchKernel、musaMalloc/muMemAlloc、musaFree/muMemFree、musaStreamSynchronize/muStreamSynchronize 的白盒闭环。
M2：把同一套 Runtime API -> Driver API -> Core object/command -> queue/build/submit -> activity/relation 方法扩展到 Top90 API 家族。
```

M2 不应重复实现已有 MUPTI callback/activity，而是在 M1 的 private ModelEvent collector 基础上补齐：

```text
object lookup
parameter normalize / validate
module/function metadata query
copy descriptor normalize
stream/event/graph object lifecycle
graph instantiate/update/launch
memory pool hit/grow/trim/update
host pin/map/unmap
M3D/DRM submit boundary
```

## 2. 公共入口与复用原则

### 2.1 Runtime 侧统一入口

M2 Runtime API 主要在以下文件：

```text
MUSA-Runtime/src/musa_module.cpp
MUSA-Runtime/src/musa_memory.cpp
MUSA-Runtime/src/musa_event.cpp
MUSA-Runtime/src/musa_stream.cpp
MUSA-Runtime/src/musa_graph.cpp
MUSA-Runtime/src/musa_device.cpp
```

Runtime wrapper 仍统一经过 `MUSA-Runtime/src/internal.h::ApiTrace`：

```text
ApiTrace constructor
  -> ApiInvocationGuard
  -> ThreadInfo::SetApiSeqNum(IncrCorrelationId())
  -> MUpti::EnterRuntimeApi(cbid, isInvocation, args...)

ApiTrace destructor
  -> MUpti::ExitRuntimeApi(context, status)
```

因此 M2 每个 Runtime API 的外层 span 和 correlation 复用现有能力：

```text
RUNTIME_API_SPAN：复用 Runtime callback/activity
RUNTIME_DRIVER_RELATION：在 musaapi* 调 Driver export table 的边界补 ModelEvent relation
```

### 2.2 Driver/Core 侧公共路径

M2 Driver API 主要在：

```text
linux-ddk/musa/src/driver/mu_module.cpp
linux-ddk/musa/src/driver/mu_memory.cpp
linux-ddk/musa/src/driver/mu_event.cpp
linux-ddk/musa/src/driver/mu_stream.cpp
linux-ddk/musa/src/driver/mu_graph.cpp
linux-ddk/musa/src/driver/mu_device.cpp
linux-ddk/musa/src/driver/mu_context.cpp
linux-ddk/musa/src/driver/mu_greencontext.cpp
```

Core 命令公共路径：

```text
Driver muapiXxx
  -> context/stream/memory/event/graph object lookup
  -> Stream::CmdXxx or Context::CreateXxx
  -> Command ctor
  -> Context::ResolveDependencyAndQueueCommand
  -> Stream::QueueCommand
  -> Stream::AsyncSubmit
  -> Command::Build
  -> Command::Submit
  -> Command::SubmitToQueue
  -> Hal::IQueue::Submit
  -> M3D Queue::Submit
  -> OS/KMD submit
```

M2 应把这条公共路径抽象成可复用事件组：

```text
OBJECT_LOOKUP_SPAN
PARAM_VALIDATE_SPAN
DESCRIPTOR_NORMALIZE_SPAN
COMMAND_CREATE
DEPENDENCY_RESOLVE_SPAN
STREAM_QUEUE_COMMAND_SPAN
COMMAND_BUILD_SPAN
COMMAND_SUBMIT_SPAN
HAL_QUEUE_SUBMIT_SPAN
M3D_QUEUE_SUBMIT_SPAN
OS_SUBMIT_SPAN
SUBMISSION_ACTIVITY_RELATION
```

## 3. Launch / module / function API

### 3.1 API 清单

源码入口：

```text
MUSA-Runtime/src/musa_module.cpp:12   musaapiLaunchHostFunc
MUSA-Runtime/src/musa_module.cpp:28   musaapiLaunchKernel
MUSA-Runtime/src/musa_module.cpp:84   musaapiLaunchKernelExC
MUSA-Runtime/src/musa_module.cpp:129  musaapiLaunchCooperativeKernel
MUSA-Runtime/src/musa_module.cpp:186  musaapiGetSymbolAddress
MUSA-Runtime/src/musa_module.cpp:209  musaapiGetSymbolSize
MUSA-Runtime/src/musa_module.cpp:234  musaapiFuncGetAttributes
MUSA-Runtime/src/musa_module.cpp:331  musaapiFuncSetAttribute
MUSA-Runtime/src/musa_module.cpp:440  musaapiGetFuncBySymbol

linux-ddk/musa/src/driver/mu_module.cpp:88   muapiModuleLoad
linux-ddk/musa/src/driver/mu_module.cpp:123  muapiModuleLoadData
linux-ddk/musa/src/driver/mu_module.cpp:154  muapiModuleLoadFatBinary
linux-ddk/musa/src/driver/mu_module.cpp:158  muapiModuleGetFunction
linux-ddk/musa/src/driver/mu_module.cpp:181  muapiModuleGetGlobal_v2
linux-ddk/musa/src/driver/mu_module.cpp:232  muapiLaunchKernel
linux-ddk/musa/src/driver/mu_module.cpp:282  muapiLaunchKernelEx
linux-ddk/musa/src/driver/mu_module.cpp:351  muapiLaunchCooperativeKernel
linux-ddk/musa/src/driver/mu_module.cpp:356  muapiModuleUnload
linux-ddk/musa/src/driver/mu_module.cpp:374  muapiFuncGetAttribute
linux-ddk/musa/src/driver/mu_module.cpp:394  muapiFuncSetAttribute
linux-ddk/musa/src/driver/mu_module.cpp:443  muapiFuncGetModule
linux-ddk/musa/src/driver/mu_module.cpp:462  muapiFuncGetName
linux-ddk/musa/src/driver/mu_module.cpp:481  muapiFuncGetParamCount
linux-ddk/musa/src/driver/mu_module.cpp:500  muapiFuncGetParamInfo
linux-ddk/musa/src/driver/mu_module.cpp:536  muapiFuncLoad
```

### 3.2 `musaLaunchCooperativeKernel` / `muLaunchCooperativeKernel`

执行路径：

```text
musaLaunchCooperativeKernel
  -> Runtime ApiTrace
  -> musaapiLaunchCooperativeKernel
      -> InitPlatformAndDevice
      -> Runtime function lookup / attribute query
      -> cooperative launch 参数校验
      -> Driver muLaunchCooperativeKernel
  -> muapiLaunchCooperativeKernel
      -> cooperative grid/block/sharedMem 校验
      -> MUSA_KERNEL_NODE_PARAMS 构造
      -> Context::GeneralLaunchKernel
      -> Stream::CmdLaunchKernel
      -> DispatchCommand
      -> ResolveDependencyAndQueueCommand
      -> QueueCommand
      -> AsyncSubmit
      -> DispatchCommand::Build
      -> DispatchCommand::Submit
      -> Command::SubmitToQueue
      -> Hal::IQueue::Submit
```

ModelEvent：

| 插点 | 事件 | payload | 成本项 |
| --- | --- | --- | --- |
| Runtime 调 Driver 边界 | `RUNTIME_DRIVER_RELATION` | cbid、correlation、function、stream | relation |
| cooperative 参数校验 | `COOPERATIVE_LAUNCH_VALIDATE_SPAN` | grid/block/sharedMem、device limits、status | validation |
| command 创建 | `COMMAND_CREATE` | command_id、type=Dispatch、cooperative=true | command_create |
| queue/build/submit | 复用 M1 launch 全套事件 | command_id、submission_id、engine | launch pipeline |

### 3.3 Module load / load data / load fat binary

执行路径：

```text
musa module API or driver module API
  -> Runtime ApiTrace if Runtime API
  -> muapiModuleLoad / muapiModuleLoadData / muapiModuleLoadFatBinary
      -> InitPlatform
      -> TlsCtxTop / device context lookup
      -> input file or memory image validate
      -> module image load
      -> ELF/fatbin/code object parse
      -> Module object create/register
      -> Function/global symbol table build
      -> module handle return
```

ModelEvent：

| 插点 | 事件 | payload | 成本项 |
| --- | --- | --- | --- |
| 文件读取或 image 输入检查 | `MODULE_INPUT_VALIDATE_SPAN` | fname/image ptr、size、status | validation |
| 模块加载总段 | `MODULE_LOAD_SPAN` / `MODULE_LOAD_DATA_SPAN` | module_id、source type、status | module_load |
| ELF/fatbin 解析 | `MODULE_PARSE_SPAN` | section count、code object count、status | module_parse |
| code object 注册 | `MODULE_REGISTER_SPAN` | module_id、function count、global count | module_register |
| symbol table 构建 | `MODULE_SYMBOL_TABLE_SPAN` | function/global symbol count | symbol_build |

### 3.4 Module/function/global query and attribute API

执行路径：

```text
musaFuncGetAttributes / muFuncGetAttribute
musaFuncSetAttribute  / muFuncSetAttribute
musaModuleGetFunction / muModuleGetFunction
musaModuleGetGlobal   / muModuleGetGlobal
  -> Runtime ApiTrace if Runtime API
  -> Driver muapiXxx
  -> module handle cast / lookup
  -> function/global symbol lookup
  -> metadata cache hit/miss
  -> attribute query/update or handle return
```

ModelEvent：

```text
MODULE_LOOKUP_SPAN
FUNCTION_LOOKUP_SPAN
GLOBAL_SYMBOL_LOOKUP_SPAN
FUNCTION_CACHE_HIT
MODULE_METADATA_QUERY_SPAN
FUNCTION_ATTRIBUTE_UPDATE
FUNCTION_LOAD_SPAN
```

Payload 建议：

```text
module handle
function handle or symbol name hash
attribute id
old/new value
cache hit/miss
status
```

## 4. Memory allocation / free / pool API

### 4.1 API 清单

源码入口：

```text
MUSA-Runtime/src/musa_memory.cpp:114  musaapiMalloc
MUSA-Runtime/src/musa_memory.cpp:131  musaapiMallocPitch
MUSA-Runtime/src/musa_memory.cpp:145  musaapiMalloc3D
MUSA-Runtime/src/musa_memory.cpp:179  musaapiMallocHost
MUSA-Runtime/src/musa_memory.cpp:190  musaapiMallocManaged
MUSA-Runtime/src/musa_memory.cpp:216  musaapiFree
MUSA-Runtime/src/musa_memory.cpp:244  musaapiFreeHost
MUSA-Runtime/src/musa_memory.cpp:253  musaapiHostAlloc
MUSA-Runtime/src/musa_memory.cpp:288  musaapiPointerGetAttributes
MUSA-Runtime/src/musa_memory.cpp:765  musaapiHostRegister
MUSA-Runtime/src/musa_memory.cpp:782  musaapiHostUnregister
MUSA-Runtime/src/musa_memory.cpp:791  musaapiMemGetInfo
MUSA-Runtime/src/musa_memory.cpp:1277 musaapiMemPrefetchAsync

linux-ddk/musa/src/driver/mu_memory.cpp:276  muapiMemAlloc_v2
linux-ddk/musa/src/driver/mu_memory.cpp:322  muapiMemAllocAsync
linux-ddk/musa/src/driver/mu_memory.cpp:368  muapiMemAllocFromPoolAsync
linux-ddk/musa/src/driver/mu_memory.cpp:419  muapiMemFreeAsync
linux-ddk/musa/src/driver/mu_memory.cpp:488  muapiMemAllocPitch_v2
linux-ddk/musa/src/driver/mu_memory.cpp:540  muapiMemHostAlloc
linux-ddk/musa/src/driver/mu_memory.cpp:544  muapiMemAllocManaged
linux-ddk/musa/src/driver/mu_memory.cpp:578  muapiMemGetInfo_v2
linux-ddk/musa/src/driver/mu_memory.cpp:645  muapiMemHostGetDevicePointer_v2
linux-ddk/musa/src/driver/mu_memory.cpp:731  muapiPointerGetAttribute
linux-ddk/musa/src/driver/mu_memory.cpp:735  muapiPointerGetAttributes
linux-ddk/musa/src/driver/mu_memory.cpp:764  muapiMemFree_v2
linux-ddk/musa/src/driver/mu_memory.cpp:817  muapiMemFreeHost
linux-ddk/musa/src/driver/mu_memory.cpp:1803 muapiMemHostRegister_v2
linux-ddk/musa/src/driver/mu_memory.cpp:1846 muapiMemHostUnregister
```

### 4.2 Async alloc/free

关键源码证据：

```text
linux-ddk/musa/src/driver/mu_memory.cpp:351  muapiMemAllocAsync -> IStream::CmdMemAlloc
linux-ddk/musa/src/driver/mu_memory.cpp:402  muapiMemAllocFromPoolAsync -> IStream::CmdMemAlloc
linux-ddk/musa/src/driver/mu_memory.cpp:470  muapiMemFreeAsync -> IStream::CmdMemFree
linux-ddk/musa/src/musa/core/stream.cpp:524 Stream::AsyncMemAlloc
linux-ddk/musa/src/musa/core/stream.cpp:586 Stream::CmdMemAlloc
linux-ddk/musa/src/musa/core/stream.cpp:623 Stream::AsyncMemFree
linux-ddk/musa/src/musa/core/stream.cpp:656 Stream::CmdMemFree
```

执行路径：

```text
musaMallocAsync / muMemAllocAsync
  -> Runtime ApiTrace
  -> Runtime Driver export table
  -> muapiMemAllocAsync
      -> InitPlatform
      -> validate dptr/bytesize
      -> TlsCtxTop
      -> Context::InfoStream
      -> MemoryAllocParameter 填充
      -> IStream::CmdMemAlloc(param, false)
  -> Stream::CmdMemAlloc
      -> capture path or AsyncMemAlloc
  -> Stream::AsyncMemAlloc
      -> pool 选择 / size align
      -> MemoryPool::CreateMemory for virtual memory
      -> physical Memory::Init
      -> MemoryPool::ModifyAccess
      -> return virtual address
```

```text
musaFreeAsync / muMemFreeAsync
  -> Runtime ApiTrace
  -> muapiMemFreeAsync
      -> InitPlatform
      -> stream lookup
      -> pointer/memory lookup
      -> IStream::CmdMemFree(dptr, false)
  -> Stream::CmdMemFree
      -> capture path or AsyncMemFree
  -> Stream::AsyncMemFree
      -> Platform::GetMemoryByDevicePointer
      -> virtual/physical/pool resolve
      -> MemoryPool::DisableAccess
      -> MemoryPool::DestroyMemory if needed
      -> enqueue CallbackCommand
      -> ResolveDependencyAndQueueCommand
      -> QueueCommand
      -> SubmitToQueue when callback command is submitted
```

ModelEvent：

```text
ASYNC_ALLOC_COMMAND_CREATE
ASYNC_FREE_COMMAND_CREATE
MEM_POOL_LOOKUP_SPAN
MEM_POOL_ALLOC_SPAN
MEM_POOL_CREATE_MEMORY_SPAN
MEMORY_INIT_SPAN
MEM_POOL_MODIFY_ACCESS_SPAN
MEM_POOL_DISABLE_ACCESS_SPAN
MEM_POOL_DESTROY_MEMORY_SPAN
MEM_FREE_CALLBACK_COMMAND_RELATION
STREAM_QUEUE_COMMAND_SPAN
COMMAND_SUBMIT_SPAN
```

Payload 建议：

```text
requested_size
aligned_size
pool_id
stream_id
virt_memory_id
physical_memory_id
virt_addr
blocking
capture_status
paging_count
status
```

### 4.3 Host pinned/register/unregister

执行路径：

```text
musaMallocHost / musaHostAlloc / muMemHostAlloc
  -> Runtime ApiTrace
  -> Driver muapiMemHostAlloc
  -> host allocation parameter validate
  -> pinned host memory allocate
  -> device visible mapping/register
  -> return host pointer
```

```text
musaHostRegister / muMemHostRegister
  -> Runtime ApiTrace
  -> Driver muapiMemHostRegister
  -> host pointer range validate
  -> page pin
  -> device map
  -> tracker/register metadata update
```

```text
musaHostUnregister / muMemHostUnregister
  -> Runtime ApiTrace
  -> Driver muapiMemHostUnregister
  -> host pointer lookup
  -> device unmap
  -> page unpin
  -> tracker unregister
```

ModelEvent：

```text
HOST_ALLOC_SPAN
HOST_REGISTER_SPAN
HOST_RANGE_VALIDATE_SPAN
HOST_PIN_SPAN
HOST_MAP_SPAN
HOST_UNMAP_SPAN
HOST_UNPIN_SPAN
HOST_MEMORY_TRACK_SPAN
```

### 4.4 Memory query / pointer attribute / pool attribute

执行路径：

```text
musaMemGetInfo / muMemGetInfo
  -> Runtime/Driver wrapper
  -> device/context lookup
  -> memory manager / pool stats query
  -> free/total return
```

```text
musaPointerGetAttributes / muPointerGetAttribute(s)
  -> Runtime/Driver wrapper
  -> Platform::GetMemoryByDevicePointer or tracker lookup
  -> allocation metadata query
  -> context/device/type/range attributes return
```

```text
memory pool trim / attribute APIs
  -> device/pool handle lookup
  -> pool stats/attribute query or update
  -> optional trim/update user pools
```

ModelEvent：

```text
MEM_INFO_QUERY_SPAN
POINTER_LOOKUP_SPAN
MEM_OBJECT_LOOKUP_SPAN
MEM_METADATA_QUERY_SPAN
MEM_POOL_LOOKUP_SPAN
MEM_POOL_TRIM_SPAN
MEM_POOL_ATTRIBUTE_QUERY
MEM_POOL_ATTRIBUTE_UPDATE
```

## 5. Memcpy / memset / migration / memory command API

### 5.1 API 清单

源码入口：

```text
MUSA-Runtime/src/musa_memory.cpp:350  musaapiMemcpy
MUSA-Runtime/src/musa_memory.cpp:416  musaapiMemcpyAsync
MUSA-Runtime/src/musa_memory.cpp:478  musaapiMemcpy2D
MUSA-Runtime/src/musa_memory.cpp:500  musaapiMemcpy2DAsync
MUSA-Runtime/src/musa_memory.cpp:660  musaapiMemcpy3D
MUSA-Runtime/src/musa_memory.cpp:717  musaapiMemcpy3DAsync
MUSA-Runtime/src/musa_memory.cpp:741  musaapiMemcpy3DPeerAsync
MUSA-Runtime/src/musa_memory.cpp:939  musaapiMemset
MUSA-Runtime/src/musa_memory.cpp:948  musaapiMemsetAsync
MUSA-Runtime/src/musa_memory.cpp:957  musaapiMemset2D
MUSA-Runtime/src/musa_memory.cpp:966  musaapiMemset2DAsync
MUSA-Runtime/src/musa_memory.cpp:975  musaapiMemset3D
MUSA-Runtime/src/musa_memory.cpp:1021 musaapiMemset3DAsync
MUSA-Runtime/src/musa_memory.cpp:1277 musaapiMemPrefetchAsync
MUSA-Runtime/src/musa_memory.cpp:1527 musaapiMemcpyPeer
MUSA-Runtime/src/musa_memory.cpp:1548 musaapiMemcpyPeerAsync
MUSA-Runtime/src/musa_memory.cpp:1569 musaapiMemoryAtomicAsync
MUSA-Runtime/src/musa_memory.cpp:1578 musaapiMemoryAtomicValueAsync
MUSA-Runtime/src/musa_memory.cpp:1587 musaapiMemoryTransfer
MUSA-Runtime/src/musa_memory.cpp:1596 musaapiMemoryTransferAsync

linux-ddk/musa/src/driver/mu_memory.cpp:870  muapiMemcpyHtoD_v2
linux-ddk/musa/src/driver/mu_memory.cpp:891  muapiMemcpyHtoDAsync_v2
linux-ddk/musa/src/driver/mu_memory.cpp:912  muapiMemcpyDtoH_v2
linux-ddk/musa/src/driver/mu_memory.cpp:933  muapiMemcpyDtoHAsync_v2
linux-ddk/musa/src/driver/mu_memory.cpp:954  muapiMemcpyDtoD_v2
linux-ddk/musa/src/driver/mu_memory.cpp:1065 muapiMemcpyDtoDAsync_v2
linux-ddk/musa/src/driver/mu_memory.cpp:1086 muapiMemcpy
linux-ddk/musa/src/driver/mu_memory.cpp:1103 muapiMemcpyAsync
linux-ddk/musa/src/driver/mu_memory.cpp:1334 muapiMemcpy2D_v2
linux-ddk/musa/src/driver/mu_memory.cpp:1368 muapiMemcpy2DAsync_v2
linux-ddk/musa/src/driver/mu_memory.cpp:1402 muapiMemcpy3D_v2
linux-ddk/musa/src/driver/mu_memory.cpp:1463 muapiMemcpy3DAsync_v2
linux-ddk/musa/src/driver/mu_memory.cpp:1631 muapiMemsetD8_v2
linux-ddk/musa/src/driver/mu_memory.cpp:1652 muapiMemsetD8Async
linux-ddk/musa/src/driver/mu_memory.cpp:1689 muapiMemsetD32_v2
linux-ddk/musa/src/driver/mu_memory.cpp:1709 muapiMemsetD32Async
linux-ddk/musa/src/driver/mu_memory.cpp:1725 muapiMemsetD2D8_v2
linux-ddk/musa/src/driver/mu_memory.cpp:1740 muapiMemsetD2D8Async
```

### 5.2 Memcpy path

执行路径：

```text
musaMemcpy / musaMemcpyAsync / muMemcpy* / muMemcpy*Async
  -> Runtime ApiTrace if Runtime API
  -> Runtime copy kind normalize
  -> Driver muapiMemcpyXxx
      -> InitPlatform
      -> pointer/context/stream lookup
      -> copy kind classify: H2D / D2H / D2D / Peer / Array / 2D / 3D
      -> descriptor normalize: address, pitch, width/height/depth, array offsets
      -> sync copy path or async stream path
      -> Stream::CmdCopyMemory / CmdMemoryTransfer
      -> copy command create
      -> ResolveDependencyAndQueueCommand
      -> QueueCommand
      -> AsyncSubmit
      -> AsyncMemcpyCommand / MemoryTransferCommand Build
      -> SubmitToQueue
      -> memcpy activity
```

关键源码证据：

```text
linux-ddk/musa/src/musa/core/command/AsyncMemcpyCommand.cpp:155 SubmitToQueue
linux-ddk/musa/src/musa/core/command/memoryTransferCommand.cpp:141 SubmitToQueue
linux-ddk/musa/src/musa/core/stream.cpp:731 CmdCopyMemory -> ResolveDependencyAndQueueCommand
linux-ddk/musa/src/driver/mu_memory.cpp:1238 CmdMemoryTransfer
linux-ddk/musa/src/driver/mu_memory.cpp:3001/3032/3063 CmdMemoryTransfer
```

ModelEvent：

```text
COPY_CLASSIFY_SPAN
COPY_H2D_VALIDATE_SPAN
COPY_D2H_VALIDATE_SPAN
COPY_D2D_CLASSIFY_SPAN
COPY_PEER_CLASSIFY_SPAN
COPY_DESCRIPTOR_NORMALIZE_SPAN
COPY_COMMAND_CREATE
COPY_BUILD_SPAN
STREAM_QUEUE_COMMAND_SPAN
COMMAND_SUBMIT_SPAN
SUBMISSION_ACTIVITY_RELATION
```

Payload：

```text
copy_kind
src/dst pointer class
src/dst memory_id
byte_count
pitch/extent
array flags
peer src/dst device/context
stream_id
command_id
submission_id
blocking
status
```

### 5.3 Memset path

执行路径：

```text
musaMemset / musaMemsetAsync / muMemsetD* / muMemsetD*Async
  -> Runtime ApiTrace
  -> Runtime memset width/height/depth normalize
  -> Driver muapiMemsetXxx
      -> pointer lookup
      -> memset descriptor build
      -> Context::GeneralMemset
      -> Stream::CmdMemset
      -> MemsetCommand create
      -> ResolveDependencyAndQueueCommand
      -> QueueCommand
      -> MemsetCommand::Build
      -> MemsetCommand::Submit
      -> Command::SubmitToQueue
      -> memset activity
```

关键源码证据：

```text
linux-ddk/musa/src/musa/core/context.cpp:777 GeneralMemset -> Stream::CmdMemset
linux-ddk/musa/src/musa/core/stream.cpp:761 Stream::CmdMemset
linux-ddk/musa/src/musa/core/command/memsetCommand.cpp:234 SubmitToQueue
```

ModelEvent：

```text
MEMSET_DESCRIPTOR_NORMALIZE_SPAN
MEMSET_COMMAND_CREATE
MEMSET_BUILD_SPAN
STREAM_QUEUE_COMMAND_SPAN
COMMAND_SUBMIT_SPAN
SUBMISSION_ACTIVITY_RELATION
```

### 5.4 Prefetch / memory atomic / memory transfer

执行路径：

```text
musaMemPrefetchAsync / muMemPrefetchAsync
  -> pointer lookup
  -> migration target device/location normalize
  -> migration/paging plan
  -> stream command
  -> queue/submit
```

```text
musaMemoryAtomicAsync / musaMemoryAtomicValueAsync
  -> driver memory atomic API
  -> graph/memory atomic node create
  -> Stream::CmdMemoryAtomic / CmdMemoryAtomicValue
  -> MemoryAtomicCommand / MemoryAtomicValueCommand
  -> SubmitToQueue
```

```text
musaMemoryTransfer / musaMemoryTransferAsync
  -> memory transfer node/command create
  -> Stream::CmdMemoryTransfer
  -> MemoryTransferCommand
  -> SubmitToQueue
```

ModelEvent：

```text
PREFETCH_COMMAND_CREATE
MIGRATION_PLAN_SPAN
PAGING_COMMAND_CREATE
MEMORY_ATOMIC_COMMAND_CREATE
MEMORY_TRANSFER_COMMAND_CREATE
COMMAND_BUILD_SPAN
COMMAND_SUBMIT_SPAN
```

## 6. Stream / event / synchronization API

### 6.1 API 清单

源码入口：

```text
MUSA-Runtime/src/musa_stream.cpp:21   musaapiStreamCreate
MUSA-Runtime/src/musa_stream.cpp:152  musaapiStreamCreateWithFlags
MUSA-Runtime/src/musa_stream.cpp:167  musaapiStreamCreateWithPriority
MUSA-Runtime/src/musa_stream.cpp:187  musaapiStreamDestroy
MUSA-Runtime/src/musa_stream.cpp:196  musaapiStreamQuery
MUSA-Runtime/src/musa_stream.cpp:205  musaapiStreamSynchronize
MUSA-Runtime/src/musa_stream.cpp:214  musaapiStreamWaitEvent
MUSA-Runtime/src/musa_stream.cpp:227  musaapiStreamGetFlags
MUSA-Runtime/src/musa_stream.cpp:236  musaapiStreamGetPriority
MUSA-Runtime/src/musa_stream.cpp:245  musaapiStreamGetId
MUSA-Runtime/src/musa_stream.cpp:254  musaapiStreamGetDevice
MUSA-Runtime/src/musa_stream.cpp:289  musaapiStreamSetAttribute
MUSA-Runtime/src/musa_stream.cpp:298  musaapiStreamGetAttribute

MUSA-Runtime/src/musa_event.cpp:12   musaapiEventCreate
MUSA-Runtime/src/musa_event.cpp:30   musaapiEventDestroy
MUSA-Runtime/src/musa_event.cpp:44   musaapiEventQuery
MUSA-Runtime/src/musa_event.cpp:53   musaapiEventRecord
MUSA-Runtime/src/musa_event.cpp:77   musaapiEventSynchronize
MUSA-Runtime/src/musa_event.cpp:86   musaapiEventElapsedTime

linux-ddk/musa/src/driver/mu_stream.cpp:16   muapiStreamCreate
linux-ddk/musa/src/driver/mu_stream.cpp:21   muapiStreamCreateWithPriority
linux-ddk/musa/src/driver/mu_stream.cpp:163  muapiStreamSynchronize
linux-ddk/musa/src/driver/mu_stream.cpp:179  muapiStreamDestroy_v2
linux-ddk/musa/src/driver/mu_stream.cpp:206  muapiStreamQuery
linux-ddk/musa/src/driver/mu_stream.cpp:357  muapiStreamWaitEvent
linux-ddk/musa/src/driver/mu_stream.cpp:392  muapiStreamBeginCapture_v2
linux-ddk/musa/src/driver/mu_stream.cpp:474  muapiStreamEndCapture
linux-ddk/musa/src/driver/mu_stream.cpp:642  muapiStreamWaitValue32_v2
linux-ddk/musa/src/driver/mu_stream.cpp:751  muapiStreamWriteValue32_v2
linux-ddk/musa/src/driver/mu_stream.cpp:860  muapiStreamBatchMemOp_v2

linux-ddk/musa/src/driver/mu_event.cpp:57   muapiEventCreate
linux-ddk/musa/src/driver/mu_event.cpp:96   muapiEventRecord
linux-ddk/musa/src/driver/mu_event.cpp:104  muapiEventSynchronize
linux-ddk/musa/src/driver/mu_event.cpp:128  muapiEventElapsedTime
linux-ddk/musa/src/driver/mu_event.cpp:160  muapiEventDestroy_v2
linux-ddk/musa/src/driver/mu_event.cpp:188  muapiEventQuery
```

### 6.2 Stream lifecycle/query

执行路径：

```text
musaStreamCreate / muStreamCreate
  -> Runtime ApiTrace
  -> Driver muapiStreamCreate
  -> InitPlatform
  -> context lookup
  -> stream object allocate
  -> stream register into context/device
  -> stream handle return
```

```text
musaStreamDestroy / muStreamDestroy
  -> stream lookup
  -> optional wait/cleanup
  -> unregister from context
  -> destroy stream object
```

```text
musaStreamQuery / muStreamQuery
  -> stream lookup
  -> last command lookup
  -> last command status / stream error query
```

ModelEvent：

```text
STREAM_CREATE_SPAN
STREAM_REGISTER_SPAN
STREAM_DESTROY_SPAN
STREAM_CLEANUP_SPAN
STREAM_STATUS_QUERY_SPAN
LAST_COMMAND_LOOKUP
STREAM_ATTRIBUTE_QUERY
STREAM_ATTRIBUTE_UPDATE
```

### 6.3 Event lifecycle/record/wait/elapsed

执行路径：

```text
musaEventCreate / muEventCreate
  -> event flags validate
  -> context lookup
  -> event object allocate/register
```

```text
musaEventRecord / muEventRecord
  -> event lookup
  -> stream lookup
  -> event record command create
  -> ResolveDependencyAndQueueCommand
  -> QueueCommand
  -> record command SubmitToQueue
```

```text
musaEventSynchronize / muEventSynchronize
  -> event lookup
  -> recorded command lookup
  -> command/semaphore wait
```

```text
musaEventElapsedTime / driver elapsed query
  -> start/end event lookup
  -> event timestamp query
  -> timestamp convert
```

关键源码证据：

```text
linux-ddk/musa/src/musa/core/command/recordCommand.cpp:121 SubmitToQueue
linux-ddk/musa/src/driver/mu_event.cpp:96 muapiEventRecord
linux-ddk/musa/src/driver/mu_event.cpp:104 muapiEventSynchronize
```

ModelEvent：

```text
EVENT_CREATE_SPAN
EVENT_REGISTER_SPAN
EVENT_RECORD_COMMAND_CREATE
EVENT_WAIT_SPAN
EVENT_TIMESTAMP_QUERY_SPAN
EVENT_DESTROY_SPAN
EVENT_COMMAND_RELATION
```

### 6.4 Stream wait/write value and batch mem op

执行路径：

```text
muStreamWaitValue32/64 / muStreamWriteValue32/64 / muStreamBatchMemOp
  -> stream lookup
  -> memory address validate
  -> memory wait/write graph node or command create
  -> Stream::CmdMemoryWaitWrite
  -> MemoryWaitWriteCommand
  -> ResolveDependencyAndQueueCommand
  -> SubmitToQueue
```

关键源码证据：

```text
linux-ddk/musa/src/driver/mu_stream.cpp:669/708/743/778/817/852/888 CmdMemoryWaitWrite
linux-ddk/musa/src/musa/core/stream.cpp:793 Stream::CmdMemoryWaitWrite
linux-ddk/musa/src/musa/core/command/memoryWaitWriteCommand.cpp:93 SubmitToQueue
```

ModelEvent：

```text
STREAM_MEM_OP_VALIDATE_SPAN
MEMORY_WAIT_WRITE_COMMAND_CREATE
STREAM_QUEUE_COMMAND_SPAN
COMMAND_SUBMIT_SPAN
```

## 7. Graph API

### 7.1 API 清单

Runtime graph APIs 集中在：

```text
MUSA-Runtime/src/musa_graph.cpp
```

重要入口：

```text
musaapiGraphCreate
musaapiGraphDestroy
musaapiGraphAddKernelNode
musaapiGraphAddMemcpyNode
musaapiGraphAddMemsetNode
musaapiGraphAddMemAllocNode
musaapiGraphAddMemFreeNode
musaapiGraphAddEventRecordNode
musaapiGraphAddEventWaitNode
musaapiGraphInstantiate / WithFlags / WithParams
musaapiGraphLaunch
musaapiGraphExecUpdate
musaapiGraphExecKernelNodeSetParams
musaapiGraphExecMemcpyNodeSetParams
musaapiGraphExecMemsetNodeSetParams
```

Driver graph APIs 集中在：

```text
linux-ddk/musa/src/driver/mu_graph.cpp
```

关键入口：

```text
linux-ddk/musa/src/driver/mu_graph.cpp:1148 muapiGraphCreate
linux-ddk/musa/src/driver/mu_graph.cpp:1170 muapiGraphDestroy
linux-ddk/musa/src/driver/mu_graph.cpp:702  muapiGraphAddKernelNode
linux-ddk/musa/src/driver/mu_graph.cpp:811  muapiGraphAddMemcpyNode
linux-ddk/musa/src/driver/mu_graph.cpp:863  muapiGraphAddMemsetNode
linux-ddk/musa/src/driver/mu_graph.cpp:327  muapiGraphAddMemAllocNode
linux-ddk/musa/src/driver/mu_graph.cpp:367  muapiGraphAddMemFreeNode
linux-ddk/musa/src/driver/mu_graph.cpp:2262 muapiGraphInstantiate_v2
linux-ddk/musa/src/driver/mu_graph.cpp:2288 muapiGraphInstantiateWithFlags
linux-ddk/musa/src/driver/mu_graph.cpp:2310 muapiGraphInstantiateWithParams
linux-ddk/musa/src/driver/mu_graph.cpp:2332 muapiGraphLaunch
linux-ddk/musa/src/driver/mu_graph.cpp:1598 muapiGraphExecUpdate
```

### 7.2 Graph create/add node/destroy

执行路径：

```text
musaGraphCreate / muGraphCreate
  -> Runtime ApiTrace
  -> Driver muapiGraphCreate
  -> TlsCtxTop
  -> Context::CreateGraph
  -> Graph object allocate/register
```

```text
musaGraphAddKernelNode / muGraphAddKernelNode
  -> graph lookup
  -> dependencies normalize
  -> kernel node params validate
  -> function handle validate
  -> graph kernel node create
  -> edge/dependency attach
```

```text
musaGraphAddMemcpyNode / muGraphAddMemcpyNode
  -> graph lookup
  -> copy descriptor normalize
  -> pointer/array/context validate
  -> graph memcpy node create
  -> dependency attach
```

```text
musaGraphAddMemsetNode / muGraphAddMemsetNode
  -> graph lookup
  -> memset descriptor normalize
  -> graph memset node create
```

ModelEvent：

```text
GRAPH_CREATE_SPAN
GRAPH_REGISTER_SPAN
GRAPH_LOOKUP_SPAN
GRAPH_DEPENDENCY_NORMALIZE_SPAN
GRAPH_NODE_CREATE_SPAN
GRAPH_KERNEL_NODE_VALIDATE_SPAN
GRAPH_MEMCPY_NODE_VALIDATE_SPAN
GRAPH_MEMSET_NODE_VALIDATE_SPAN
GRAPH_EDGE_UPDATE_SPAN
GRAPH_DESTROY_SPAN
```

### 7.3 Graph instantiate/update/launch

关键源码证据：

```text
linux-ddk/musa/src/driver/mu_graph.cpp:2274 CreateGraphExec
linux-ddk/musa/src/driver/mu_graph.cpp:2300 CreateGraphExec(flags)
linux-ddk/musa/src/driver/mu_graph.cpp:2322 CreateGraphExec(instantiateParams->flags)
linux-ddk/musa/src/driver/mu_graph.cpp:2354 pStream->CmdLaunchGraph
linux-ddk/musa/src/musa/core/context.cpp:1338 Context::CreateGraphExec
linux-ddk/musa/src/musa/core/stream.cpp:284 Stream::CmdLaunchGraph
linux-ddk/musa/src/musa/core/command/graphCommand.cpp:226/262/264/276/289/297/299 universalMgr->CmdGraph
linux-ddk/musa/src/musa/core/graph/graph1/universalManager.cpp:371 UniversalManager::CmdGraph
```

执行路径：

```text
musaGraphInstantiate / muGraphInstantiate
  -> Runtime ApiTrace
  -> Driver muapiGraphInstantiate*
      -> graph handle cast
      -> context lookup
      -> Context::CreateGraphExec
      -> graph validate
      -> executable graph build
      -> dependency plan
      -> node command templates / universal manager state build
```

```text
musaGraphExecUpdate / muGraphExecUpdate
  -> graph exec lookup
  -> input graph lookup
  -> diff
  -> update validate
  -> patch executable graph
```

```text
musaGraphLaunch / muGraphLaunch
  -> Runtime ApiTrace
  -> Driver muapiGraphLaunch
      -> stream lookup
      -> graph exec lookup
      -> Stream::CmdLaunchGraph
      -> GraphCommand create
      -> ResolveDependencyAndQueueCommand
      -> QueueCommand
      -> GraphCommand build/submit
      -> UniversalManager::CmdGraph recursively emits kernel/memcpy/memset/memory commands
      -> Command::SubmitToQueue
      -> graph/kernel/memcpy/memset activities
```

ModelEvent：

```text
GRAPH_VALIDATE_SPAN
GRAPH_INSTANTIATE_SPAN
GRAPH_DEPENDENCY_PLAN_SPAN
GRAPH_EXEC_BUILD_SPAN
GRAPH_EXEC_UPDATE_SPAN
GRAPH_DIFF_SPAN
GRAPH_LAUNCH_COMMAND_CREATE
GRAPH_COMMAND_BUILD_SPAN
GRAPH_UNIVERSAL_CMD_SPAN
GRAPH_ACTIVITY_RELATION
GRAPH_NODE_ACTIVITY_RELATION
```

Payload：

```text
graph_id
graph_exec_id
node_count
edge_count
node_type
updated_node_count
launch_stream_id
graph_command_id
child command id
submission_id
activity_id
status
```

## 8. Context / device / resource / Green Context API

### 8.1 API 清单

```text
MUSA-Runtime/src/musa_device.cpp:188  musaapiDeviceReset
MUSA-Runtime/src/musa_device.cpp:212  musaapiDeviceSynchronize
MUSA-Runtime/src/musa_device.cpp:224  musaapiGetDevice
MUSA-Runtime/src/musa_device.cpp:245  musaapiGetDeviceCount
MUSA-Runtime/src/musa_device.cpp:254  musaapiSetDevice
MUSA-Runtime/src/musa_device.cpp:293  musaapiGetDeviceProperties_v2
MUSA-Runtime/src/musa_device.cpp:500  musaapiDeviceGetAttribute

linux-ddk/musa/src/driver/mu_device.cpp:14   muapiDeviceGet
linux-ddk/musa/src/driver/mu_device.cpp:40   muapiDeviceGetCount
linux-ddk/musa/src/driver/mu_device.cpp:54   muapiDeviceGetProperties
linux-ddk/musa/src/driver/mu_device.cpp:75   muapiDeviceGetAttribute
linux-ddk/musa/src/driver/mu_device.cpp:310  muapiDeviceGetDefaultMemPool
linux-ddk/musa/src/driver/mu_device.cpp:332  muapiDeviceSetMemPool
linux-ddk/musa/src/driver/mu_device.cpp:358  muapiDeviceGetMemPool

linux-ddk/musa/src/driver/mu_context.cpp:135  muapiCtxSynchronize
linux-ddk/musa/src/driver/mu_context.cpp:154  muapiCtxCreate_v2
linux-ddk/musa/src/driver/mu_context.cpp:186  muapiCtxDestroy_v2
linux-ddk/musa/src/driver/mu_context.cpp:249  muapiCtxPushCurrent_v2
linux-ddk/musa/src/driver/mu_context.cpp:268  muapiCtxGetCurrent
linux-ddk/musa/src/driver/mu_context.cpp:282  muapiCtxSetCurrent
linux-ddk/musa/src/driver/mu_context.cpp:478  muapiDevicePrimaryCtxRetain
linux-ddk/musa/src/driver/mu_context.cpp:548  muapiDevicePrimaryCtxRelease_v2

linux-ddk/musa/src/driver/mu_greencontext.cpp:16   muapiCtxGetDevResource
linux-ddk/musa/src/driver/mu_greencontext.cpp:44   muapiGreenCtxGetDevResource
linux-ddk/musa/src/driver/mu_greencontext.cpp:72   muapiGreenCtxStreamCreate
linux-ddk/musa/src/driver/mu_greencontext.cpp:100  muapiGreenCtxCreate
linux-ddk/musa/src/driver/mu_greencontext.cpp:127  muapiGreenCtxDestroy
linux-ddk/musa/src/driver/mu_greencontext.cpp:162  muapiDevResourceGenerateDesc
linux-ddk/musa/src/driver/mu_greencontext.cpp:198  muapiCtxFromGreenCtx
linux-ddk/musa/src/driver/mu_greencontext.cpp:364  muapiDeviceGetDevResource
linux-ddk/musa/src/driver/mu_greencontext.cpp:411  muapiDevSmResourceSplitByCount
```

### 8.2 Device/context API

执行路径：

```text
musaSetDevice / muCtxSetCurrent
  -> Runtime ApiTrace
  -> device id validate
  -> device/context lookup
  -> TLS current context update
  -> optional primary context retain/init
```

```text
musaGetDeviceProperties / muDeviceGetAttribute
  -> device lookup
  -> HAL/device property query
  -> property translate/copy out
```

```text
musaDeviceSynchronize / muCtxSynchronize
  -> context lookup
  -> wait all streams / LockedWait
  -> command wait / semaphore wait
  -> error query
```

ModelEvent：

```text
DEVICE_LOOKUP_SPAN
DEVICE_ATTRIBUTE_QUERY_SPAN
DEVICE_PROPERTY_QUERY_SPAN
CONTEXT_LOOKUP_SPAN
CONTEXT_SWITCH_SPAN
PRIMARY_CONTEXT_RETAIN_SPAN
PRIMARY_CONTEXT_RELEASE_SPAN
CONTEXT_WAIT_ALL_SPAN
COMMAND_WAIT_SPAN
SYNC_WAIT_REASON
```

### 8.3 Green Context / resource API

执行路径：

```text
muDeviceGetDevResource
  -> device lookup
  -> device resource query
  -> SM/resource descriptor build
```

```text
muDevSmResourceSplitByCount
  -> resource validate
  -> SM count/granularity check
  -> resource split
  -> remaining resource update
```

```text
muDevResourceGenerateDesc
  -> resources array validate
  -> resource descriptor generate
```

```text
muGreenCtxCreate
  -> resource descriptor validate
  -> device/context lookup
  -> GreenContext object create
  -> resource bind / KMD bind
  -> context handle association
```

```text
muGreenCtxStreamCreate
  -> green context lookup
  -> stream create
  -> stream bind to green resource/context
```

```text
muGreenCtxDestroy
  -> green context lookup
  -> stream/context cleanup
  -> resource unbind
  -> object destroy
```

ModelEvent：

```text
DEVICE_RESOURCE_QUERY_SPAN
SM_RESOURCE_SPLIT_SPAN
RESOURCE_DESC_GENERATE_SPAN
GREEN_CTX_CREATE_SPAN
RESOURCE_BIND_SPAN
GREEN_CTX_TO_CONTEXT_SPAN
GREEN_STREAM_CREATE_SPAN
GREEN_CTX_DESTROY_SPAN
RESOURCE_UNBIND_SPAN
```

Payload：

```text
device_id
resource_type
sm_count
min_count
split_group_count
remaining_sm_count
green_ctx_id
context_id
stream_id
bind status
```

## 9. HAL / M3D / KMD submit boundary

M2 必须补齐 M1 没覆盖的底层 submit 边界，否则 Top90 API 成本只能停在 `Command::SubmitToQueue`。

关键源码证据：

```text
linux-ddk/musa/src/musa/core/command/command.cpp:646 Command::SubmitToQueue
linux-ddk/musa/src/musa/core/command/command.cpp:702 calls hal=IQueue::Submit
linux-ddk/musa/src/hal/m3d/queue.cpp:178 Hal M3D Queue::Submit
linux-ddk/musa/src/hal/m3d/queue.cpp:348 m_M3dQueue->Submit
linux-ddk/musa/src/hal/m3d/m3d/src/core/queue.cpp:150 M3D Queue::Submit
linux-ddk/musa/src/hal/m3d/m3d/src/core/queue.cpp:1276 OsSubmit
linux-ddk/musa/src/hal/m3d/m3d/src/core/queue.cpp:1483 OsSubmit
```

执行路径：

```text
Command::SubmitToQueue
  -> wait semaphore vector prepare
  -> signal semaphore vector prepare
  -> submitInfo.submissionId assign
  -> Hal::IQueue::Submit
  -> linux-ddk/musa/src/hal/m3d/queue.cpp::Queue::Submit
      -> Hal submit info translate to M3D submit info
      -> M3D queue submit info build
      -> m_M3dQueue->Submit
  -> linux-ddk/musa/src/hal/m3d/m3d/src/core/queue.cpp::Queue::Submit
      -> ValidateSubmit
      -> queue context PreProcessSubmit
      -> OsSubmit
      -> queue context PostProcessSubmit
  -> OS/KMD ioctl submit
  -> firmware/doorbell notify depending queue mode
```

ModelEvent：

```text
SUBMIT_SEMAPHORE_PREPARE_SPAN
HAL_QUEUE_SUBMIT_SPAN
HAL_M3D_SUBMIT_TRANSLATE_SPAN
M3D_QUEUE_SUBMIT_SPAN
M3D_VALIDATE_SUBMIT_SPAN
M3D_PREPROCESS_SUBMIT_SPAN
OS_SUBMIT_SPAN
DRM_IOCTL_SUBMIT_SPAN
M3D_POSTPROCESS_SUBMIT_SPAN
FIRMWARE_NOTIFY_SPAN
```

Payload：

```text
command_id
submission_id
engine
queue_id
cmdBufferCount
waitSemaphoreCount
signalSemaphoreCount
fence info
perSubQueue count
os submit status
ioctl errno/result
doorbell enabled
```

## 10. M2 事件优先级

### P0：Top90 source rule 必须闭环

| API 家族 | P0 ModelEvent |
| --- | --- |
| module/function | `MODULE_LOAD_SPAN`、`MODULE_PARSE_SPAN`、`MODULE_REGISTER_SPAN`、`FUNCTION_LOOKUP_SPAN`、`FUNCTION_ATTRIBUTE_UPDATE` |
| async memory/pool | `ASYNC_ALLOC_COMMAND_CREATE`、`ASYNC_FREE_COMMAND_CREATE`、`MEM_POOL_ALLOC_SPAN`、`MEM_POOL_DISABLE_ACCESS_SPAN`、`MEM_POOL_DESTROY_MEMORY_SPAN` |
| memcpy/memset | `COPY_CLASSIFY_SPAN`、`COPY_DESCRIPTOR_NORMALIZE_SPAN`、`COPY_COMMAND_CREATE`、`MEMSET_COMMAND_CREATE`、`COPY_BUILD_SPAN`、`MEMSET_BUILD_SPAN` |
| stream/event | `STREAM_CREATE_SPAN`、`STREAM_DESTROY_SPAN`、`EVENT_CREATE_SPAN`、`EVENT_RECORD_COMMAND_CREATE`、`EVENT_WAIT_SPAN` |
| graph | `GRAPH_NODE_CREATE_SPAN`、`GRAPH_INSTANTIATE_SPAN`、`GRAPH_LAUNCH_COMMAND_CREATE`、`GRAPH_ACTIVITY_RELATION` |
| device/context/resource | `DEVICE_LOOKUP_SPAN`、`CONTEXT_SWITCH_SPAN`、`CONTEXT_WAIT_ALL_SPAN`、`DEVICE_RESOURCE_QUERY_SPAN`、`GREEN_CTX_CREATE_SPAN` |
| submit boundary | `HAL_QUEUE_SUBMIT_SPAN`、`M3D_QUEUE_SUBMIT_SPAN`、`OS_SUBMIT_SPAN` |

### P1：降低 unknown cost

```text
MODULE_SYMBOL_TABLE_SPAN
FUNCTION_CACHE_HIT
HOST_PIN_SPAN / HOST_MAP_SPAN / HOST_UNPIN_SPAN
POINTER_LOOKUP_SPAN
MEM_INFO_QUERY_SPAN
COPY_PEER_CLASSIFY_SPAN
STREAM_ATTRIBUTE_QUERY / UPDATE
GRAPH_DIFF_SPAN
GRAPH_DEPENDENCY_PLAN_SPAN
M3D_VALIDATE_SUBMIT_SPAN
DRM_IOCTL_SUBMIT_SPAN
```

### P2：M3/M4 可后移

```text
firmware notify 深层细分
所有 graph node type 的逐字段 validate
所有 device property query 的逐字段展开
IPC memory/event 全链路
external semaphore/graphics interop 全链路
```

## 11. M2 成本分项模型

### 11.1 Module/function

```text
T_module =
  T_runtime_wrapper
+ T_driver_wrapper
+ T_module_input_validate
+ T_module_load
+ T_module_parse
+ T_symbol_table
+ T_module_register
+ T_unknown
```

```text
T_function_query =
  T_runtime_wrapper
+ T_driver_wrapper
+ T_module_lookup
+ T_function_lookup
+ T_metadata_query
+ T_cache_hit_miss
+ T_unknown
```

### 11.2 Async memory/pool

```text
T_async_alloc =
  T_runtime_wrapper
+ T_driver_wrapper
+ T_stream_lookup
+ T_pool_lookup
+ T_pool_alloc
+ T_memory_init
+ T_pool_modify_access
+ T_relation
+ T_unknown
```

```text
T_async_free =
  T_runtime_wrapper
+ T_driver_wrapper
+ T_stream_lookup
+ T_memory_lookup
+ T_pool_disable_access
+ T_callback_command
+ T_queue
+ T_submit
+ T_unknown
```

### 11.3 Copy/memset

```text
T_copy =
  T_runtime_wrapper
+ T_driver_wrapper
+ T_copy_classify
+ T_descriptor_normalize
+ T_object_lookup
+ T_copy_command_create
+ T_dependency_resolve
+ T_queue
+ T_copy_build
+ T_submit
+ T_activity_relation
+ T_unknown
```

```text
T_memset =
  T_runtime_wrapper
+ T_driver_wrapper
+ T_descriptor_normalize
+ T_memset_command_create
+ T_dependency_resolve
+ T_queue
+ T_memset_build
+ T_submit
+ T_activity_relation
+ T_unknown
```

### 11.4 Stream/event/sync

```text
T_event_record =
  T_runtime_wrapper
+ T_driver_wrapper
+ T_event_lookup
+ T_stream_lookup
+ T_record_command_create
+ T_queue
+ T_submit
+ T_unknown
```

```text
T_event_sync =
  T_runtime_wrapper
+ T_driver_wrapper
+ T_event_lookup
+ T_waited_command_lookup
+ T_command_wait
+ T_semaphore_wait
+ T_unknown
```

### 11.5 Graph

```text
T_graph_instantiate =
  T_runtime_wrapper
+ T_driver_wrapper
+ T_graph_lookup
+ T_graph_validate
+ T_dependency_plan
+ T_exec_build
+ T_unknown
```

```text
T_graph_launch =
  T_runtime_wrapper
+ T_driver_wrapper
+ T_graph_exec_lookup
+ T_graph_command_create
+ T_dependency_resolve
+ T_queue
+ T_graph_build
+ T_submit
+ T_child_activity_relation
+ T_unknown
```

### 11.6 Submit boundary

```text
T_submit_boundary =
  T_command_submit_to_queue
+ T_hal_submit_translate
+ T_m3d_queue_submit
+ T_m3d_validate
+ T_m3d_preprocess
+ T_os_submit
+ T_m3d_postprocess
+ T_unknown
```

## 12. M2 source rule 模板

每个 M2 API 都应生成一份 source rule，最少包含：

```yaml
api:
  runtime: musaXxx
  driver: muXxx
  cbid: MUPTI_CBID_...
entry:
  runtime_file: MUSA-Runtime/src/...
  driver_file: linux-ddk/musa/src/driver/...
path:
  - runtime_wrapper
  - runtime_impl
  - driver_wrapper
  - object_lookup
  - core_command_or_object
  - dependency_queue_submit_or_lifecycle
model_events:
  - event_id: ...
    type: span|instant|relation|counter
    insert_at: file:function
    payload: [...]
relations:
  - runtime_api -> driver_api
  - driver_api -> object|command
  - command -> submission
  - submission -> activity
cost_terms:
  - ...
validation:
  workload: ...
  expected_events: ...
```

## 13. M2 验收建议

M2 通过条件：

```text
1. Top90 候选 API 每个都有 source rule。
2. 每个 source rule 至少包含 Runtime/Driver 入口、Core 执行路径、ModelEvent 签名、成本项、验证 case。
3. Launch/copy/memset/graph launch 均能建立 command -> submission -> activity relation。
4. Sync/event API 能解释 wait reason 和 waited command/activity。
5. Memory/pool API 能区分 sync alloc/free、async alloc/free、host pin/map、pool hit/grow/trim。
6. Module/function API 能区分 load/parse/register/query/cache hit。
7. Submit boundary 能至少拆到 Command::SubmitToQueue、Hal::IQueue::Submit、M3D Queue::Submit、OsSubmit。
8. M2 API unknown cost 目标 <= 20%，核心 Top API unknown cost <= 15%。
9. trace off overhead 保持在 benchmark 噪声内；targeted internal tracing overhead <= 3%。
10. event_quality_report 必须报告 dropped/overflow/missing relation/missing span。
```

建议 M2 验证 workload：

```text
module/function：module load + get function + func attr query/set + launch
memory/pool：malloc/free + mallocAsync/freeAsync + host register/unregister + pointer attrs
copy/memset：H2D/D2H/D2D/peer/2D/3D memcpy + memset D8/D32/2D/3D
stream/event：stream create/destroy/query + event record/sync/elapsed + stream wait event/value
 graph：graph create + add kernel/memcpy/memset/mem alloc/free node + instantiate + launch + update
context/device/resource：set device + ctx sync + device attr/property + GreenContext create/stream/destroy
submit boundary：single kernel/copy/memset/graph launch with M3D submit tracing enabled
```
