# musa 核心对象与 API 执行流程全景

本文基于 `/home/shanfeng/workspace/linux-ddk/musa` 当前源码梳理，用于支撑 OKR 中 ModelEvent、source rule、relation builder、cost breakdown 和行为 CTS 的落地。

核心结论：`musa` 仓库的 Driver API 不是简单的函数封装，而是围绕 `Platform -> Device -> Context -> Stream -> Command -> HAL/M3D/OS submit` 形成的分层执行系统。API 性能建模要抓住三类边界：

```text
1. API 边界：Runtime/Driver wrapper、参数校验、TLS context/stream/object lookup
2. Core 边界：Context/Stream/Memory/Graph/Module 等核心对象完成语义转换
3. Submit 边界：Command build/merge/dependency/queue submit/wait -> HAL/M3D/OS
```

## 1. 总体对象层次

```text
Application / Framework
  |
  | Runtime API，例如 musaLaunchKernel / musaMalloc / musaMemcpyAsync
  v
MUSA-Runtime
  |  ApiTrace / Runtime callback / Runtime activity / correlation id
  v
linux-ddk/musa Driver API
  |  muapi* entry
  |  InitPlatform()
  |  TlsCtxTop()
  |  ApiTrace / InjectionManager / MUPTI tracepoints
  v
Core object layer
  |
  +-- Platform
  |     +-- global singleton
  |     +-- HAL platform owner
  |     +-- device/context/memory global lookup
  |
  +-- Device
  |     +-- physical GPU abstraction
  |     +-- HAL device owner
  |     +-- context / green context creation
  |     +-- queue family / internal memory pool / copy manager
  |
  +-- Context / GreenContext
  |     +-- resource owner
  |     +-- stream/event/module/memory/graph creation and validation
  |     +-- GeneralLaunchKernel / GeneralMemcpy / GeneralMemset
  |     +-- dependency resolve and QueueCommand
  |
  +-- Stream
  |     +-- command queue owner
  |     +-- default stream semantics
  |     +-- command list / merging list / inflight list / waiting list
  |     +-- submit thread / wait thread
  |     +-- HAL queue / semaphore / cmd buffer allocation
  |
  +-- Command subclasses
  |     +-- DispatchCommand
  |     +-- MemcpyCommand / SyncMemcpyCommand / AsyncMemcpyCommand
  |     +-- MemsetCommand
  |     +-- GraphCommand
  |     +-- RecordCommand / BarrierCommand / CallbackCommand
  |     +-- PagingCommand / memory atomic / memory transfer / external semaphore
  |
  +-- Memory / MemoryPool
  |     +-- synchronous general memory
  |     +-- virtual memory / pooled memory / async alloc/free
  |     +-- HAL memory / HAL memory pool
  |
  +-- Module / Function
  |     +-- library/image loading
  |     +-- symbol/function lookup
  |     +-- kernel state creation
  |
  +-- Event
  |     +-- query / synchronize / IPC / captured graph node
  |
  +-- Graph / GraphNode / GraphExec
        +-- graph topology
        +-- node semantic validation and legalization
        +-- graph flatten / resolve / submission preparation
        +-- GraphCommand launch
  v
HAL layer
  +-- Hal::IPlatform / IDevice / IQueue / ICmdBuffer / IMemory / IMemoryPool / ISemaphore
  v
M3D / OS layer
  +-- M3D queue / command buffer / memory / semaphore
  +-- DRM/WKMD/WDDM OS submit boundary
```

## 2. 核心对象功能解读

### 2.1 `Platform`

源码锚点：`src/musa/core/platform.h`

`Platform` 是全局入口对象，持有 `Hal::IPlatform`，提供全局 device/context/memory 管理能力。Driver API 中大量函数先调用 `InitPlatform()`，再通过 `Platform::Get()` 访问全局状态。

关键职责：

```text
- 初始化 HAL platform
- 维护 device 列表和 global settings
- CreateMemory / DestroyMemory 这类 platform-level memory 操作
- GetMemoryByDevicePointer / GetMemoryByHandle
- ValidateContext
- P2P attribute 查询
```

对 OKR 的意义：`Platform` 是全局 lookup、初始化、settings、memory pointer relation 的关键位置。ModelEvent 应覆盖 `PLATFORM_INIT_SPAN`、`MEMORY_POINTER_LOOKUP_SPAN`、`DEVICE_LOOKUP_SPAN`。

### 2.2 `Device`

源码锚点：`src/musa/core/device.h`

`Device` 封装物理 GPU 和 `Hal::IDevice`。它负责创建普通 `Context` 和 `GreenContext`，也管理 internal memory pool、copy managers、queue family index。

关键职责：

```text
- GetAttribute / device capability query
- CreateContext / DestroyContext
- CreateGreenContext
- GetDeviceResources
- InitCopyManagers
- QueryGlobalMemFreeSize
- AllocateInternalMem
- GetQueueFamilyIndex
```

对 OKR 的意义：device/resource/green context 类 API 的 cost 主要来自 device lookup、HAL device query、resource split/desc validation、context creation。应记录 `DEVICE_QUERY_SPAN`、`DEVICE_CREATE_CONTEXT_SPAN`、`DEVICE_CREATE_GREEN_CONTEXT_SPAN`、`DEV_RESOURCE_SPLIT_POINT`。

### 2.3 `Context` / `GreenContext`

源码锚点：`src/musa/core/context.h`、`src/musa/core/greenContext.h`

`Context` 是绝大多数 API 的核心资源所有者。Driver API 通常通过 `TlsCtxTop()` 找到当前 context，然后在 context 上创建或校验 object。`GreenContext` 继承 `Context` 并实现资源隔离语义。

关键职责：

```text
- CreateMemory / DestroyMemory
- CreateStream / DestroyStream
- CreateModule / DestroyModule
- CreateEvent / DestroyEvent
- CreateGraph / CreateGraphExec
- CreateKernelNode / CreateMemcpyNode / CreateMemsetNode / CreateMemAllocNode / CreateMemFreeNode
- GeneralLaunchKernel
- GeneralMemcpy
- GeneralMemset
- RecordEvent / WaitEvent
- Synchronize
- ResolveDependencyAndQueueCommand
- AddCurrentDependencies
- LockedSyncDefaultStream
```

对 OKR 的意义：`Context` 是 API 语义从 Driver wrapper 转入 Core 的主边界。Launch/memcpy/memset/graph/event/sync 都会在这里生成 node/command 或 resolve dependency。应重点记录 `CONTEXT_LOOKUP_SPAN`、`CONTEXT_VALIDATE_OBJECT_SPAN`、`CONTEXT_CREATE_OBJECT_SPAN`、`DEPENDENCY_RESOLVE_SPAN`。

### 2.4 `Stream`

源码锚点：`src/musa/core/stream.h`

`Stream` 是命令队列和异步提交核心。API 中的 stream handle 会通过 `Context::InfoStream` 解析为 `Stream` 对象。异步 API 最终通常落到 `Stream::Cmd*` 或 `Stream::QueueCommand`。

关键职责：

```text
- Query / Synchronize / WaitFinish
- CmdLaunchKernel / CmdCopyMemory / CmdMemset / CmdLaunchGraph
- CmdMemAlloc / CmdMemFree
- CmdSetEvent / CmdWaitEvent
- CmdMemoryAtomic / CmdMemoryTransfer
- QueueCommand
- LastCommand
- RequestCmdBuffer
- GetHalQueue
- AsyncSubmit / AsyncWait
```

内部状态：

```text
m_CommandList   : 等待提交的 command
m_MergingList   : 可合并 command
m_InflightList  : 已提交未完成 command
m_WaitingList   : 等待完成/回收 command
m_LastCommand   : stream 上一个 command，用于依赖
m_SubmitThread  : 异步 submit 线程
m_WaitThread    : 异步 wait 线程
```

对 OKR 的意义：`Stream` 是 queue latency、merge、dependency、wait 的关键位置。应记录 `STREAM_RESOLVE_SPAN`、`STREAM_QUEUE_COMMAND_SPAN`、`STREAM_WAIT_FINISH_SPAN`、`COMMAND_MERGE_DECISION`、`ASYNC_SUBMIT_WAKEUP_POINT`。

### 2.5 `Command` 及子类

源码锚点：`src/musa/core/command/command.h`、`src/musa/core/command/*.h`

`Command` 是提交到 HAL/M3D 的统一抽象。不同 API 最终转换为不同 command 子类：kernel launch 对应 `DispatchCommand`，memcpy 对应 `MemcpyCommand`，memset 对应 `MemsetCommand`，graph launch 对应 `GraphCommand`。

关键职责：

```text
- RecordDependency
- SetPrevCommand
- CanMergeTo
- Build
- Submit
- SubmitToQueue
- ResolveSubmitWait
- ResolveSubmitSignal
- GetHalCmdBuffer
- Wait / WaitFinish
```

主要子类：

```text
DispatchCommand              kernel launch
MemcpyCommand                memcpy base
SyncMemcpyCommand            synchronous copy
AsyncMemcpyCommand           async copy with command buffer build
MemsetCommand                memset
GraphCommand                 graph exec launch and execution
RecordCommand                event record
BarrierCommand               barrier / dependency command
CallbackCommand              deferred callback / deferred free
PagingCommand                memory paging
MemoryAtomicCommand          memory atomic
MemoryAtomicValueCommand     memory atomic value
MemoryTransferCommand        memory transfer
Wait/SignalExternalSemaphore external semaphore interop
AccelStruct*Command          ray tracing accel structure
```

对 OKR 的意义：`Command` 是 Core 到 HAL 的模型关键点。要把 API cost 拆到 `COMMAND_CREATE_SPAN`、`COMMAND_BUILD_SPAN`、`COMMAND_SUBMIT_SPAN`、`HAL_QUEUE_SUBMIT_SPAN`、`COMMAND_WAIT_SPAN`。

### 2.6 `Memory` / `MemoryPool`

源码锚点：`src/musa/core/memory.h`、`src/musa/core/memoryPool.h`

`Memory` 封装 device pointer、HAL memory、memory type、pool ownership 和 synchronize/bind/unbind 语义。`MemoryPool` 管理 pooled/virtual memory 的 access、modify、disable 和 pool allocation。

关键职责：

```text
Memory:
- Init / InitFromPool / InitPrealloc
- GeneralAlloc / VirtualAlloc
- GetDevicePointer
- Synchronize
- Bind / Unbind
- GetPhysMemory

MemoryPool:
- Init
- SetAttribute / GetAttribute
- SetAccess / GetAccess
- ModifyAccess / DisableAccess
- CreateMemory / DestroyMemory
```

重要区分：

```text
muMemAlloc_v2 / musaMalloc:
  synchronous general memory path
  Context::CreateMemory -> Memory::Init -> HAL memory allocate/map

muMemAllocAsync / muMemAllocFromPoolAsync:
  stream async path
  Stream::CmdMemAlloc -> AsyncMemAlloc -> MemoryPool path

muMemFree_v2 / musaFree:
  common general memory path
  Platform::GetMemoryByDevicePointer -> IMemory::Synchronize -> Context::DestroyMemory

virtual memory with pool free:
  Stream::CmdMemFree(blocking=true) -> AsyncMemFree -> MemoryPool::DisableAccess -> CallbackCommand
```

对 OKR 的意义：同步 allocation/free 不应错误建模为 pool hit/miss；pool hit/miss/grow 只属于 async/pool/VMM 路径。

### 2.7 `Module` / `Function`

源码锚点：`src/musa/core/module.h`、`src/musa/core/symbol.h`

`Module` 管理 fat binary/image/library 加载和符号查找；`Function` 管理 kernel symbol、HAL kernel、program address、kernel attribute 和 kernel state 创建。

关键职责：

```text
Module:
- LoadFatBinary
- LoadImage
- GetFunction
- GetGlobal
- FindSymbolByName
- FindFunc

Function:
- GetAttribute / SetAttribute
- GetFuncName
- CreateState
- GetProgramAddr
- Hal kernel access
```

对 OKR 的意义：launch 前的 module/function 成本常被归到 API unknown cost，应通过 `MODULE_LOAD_SPAN`、`FUNCTION_LOOKUP_SPAN`、`KERNEL_STATE_CREATE_SPAN` 拆出来。

### 2.8 `Event`

源码锚点：`src/musa/core/event.h`

`Event` 负责 query、synchronize、IPC 和 graph capture node 关联。event record/wait API 最终会走 context/stream command 路径。

关键职责：

```text
- Query
- Synchronize
- ExportIpcHandle / ImportIpcHandle
- AddCapturedNode
```

对 OKR 的意义：event API 的关键成本是 object validation、stream command 插入、wait/synchronize。应记录 `EVENT_VALIDATE_SPAN`、`EVENT_RECORD_COMMAND_SPAN`、`EVENT_WAIT_COMMAND_SPAN`、`EVENT_SYNCHRONIZE_SPAN`。

### 2.9 `Graph` / `GraphNode` / `GraphExec`

源码锚点：`src/musa/core/graph.h`、`src/musa/core/node/*.h`、`src/musa/core/graph/graph1/graphExec.*`、`src/musa/core/graph/graph2/graphExec.*`

`Graph` 持有 graph topology，`GraphNode` 表示 kernel/memcpy/memset/host/event/mem alloc/mem free 等语义节点，`GraphExec` 负责 instantiate 后的 flatten、resolve、submission prepare、cmd buffer 写入。

关键职责：

```text
Graph:
- AddGraphNode
- DeleteNode
- ValidateNode
- AddDependencies / RemoveDependencies
- GetRootNodes / GetNodes / GetEdges / GetInDegree

GraphNode families:
- GraphKernelNode
- GraphMemcpyNode
- GraphMemsetNode
- GraphHostNode
- GraphEventRecordNode / GraphEventWaitNode
- GraphMemoryAllocNode / GraphMemoryFreeNode
- GraphMemoryAtomicNode / GraphMemoryTransferNode / GraphMemoryWaitWriteNode
- GraphChildGraphNode / GraphConditionalNode / GraphExternalSemaphore*Node

GraphExec:
- Init
- ResolveGraph / ResolveGraphImpl
- PrepareAllSubmissions / PrepareSubmission
- WriteSubmission
- WriteDispatchCmd / WriteMemcpyCmd / WriteMemsetCmd
- WriteMemoryAtomicCmd / WriteMemoryTransferCmd
- RequestCmdBuffer / ReleaseCmdBuffer

GraphCommand:
- Build
- Submit
- Execute
- ExecuteImpl
```

对 OKR 的意义：graph API 不能只看 `muGraphLaunch`，必须分为 graph construction、node legalize、instantiate flatten/resolve、submission write、launch execute 五段。

## 3. Driver API 通用执行框架

大多数 `muapi*` API 遵循如下结构：

```text
muapiXxx(...)
  |
  +-- InitPlatform()
  |
  +-- parameter validation
  |
  +-- TlsCtxTop() / Platform::Get() / object cast / object validation
  |
  +-- Context / Stream / Device / Module / Graph core operation
  |
  +-- optional MUPTI tracepoint / activity registration
  |
  +-- return MUresult
```

如果经过 Runtime API，则整体链路是：

```text
musaXxx Runtime API
  |
  +-- Runtime ApiTrace / Runtime callback / Runtime activity
  |
  +-- Runtime wrapper calls muXxx Driver API
  |
  +-- Driver ApiTrace / Driver callback / Driver activity
  |
  +-- Driver muapiXxx implementation
  |
  +-- Core objects / HAL / M3D
```

ModelEvent 不替代现有 MUPTI callback/activity，而是补足 Driver/Core 内部阶段：

```text
现有 MUPTI：API enter/exit、kernel/memcpy/memset/sync/graph activity、correlation id
ModelEvent：Context/Stream/Command/Memory/Graph/HAL 内部阶段 cost 和 relation
```

## 4. API 家族分类与执行流程

### 4.1 Device / Context / Resource 家族

代表 API：

```text
muInit
muDeviceGet
muDeviceGetCount
muDeviceGetAttribute
muDeviceGetName
muDeviceTotalMem
muCtxCreate / muCtxDestroy
muCtxSetCurrent / muCtxGetCurrent
muCtxSynchronize
muDeviceGetDevResource
muDevSmResourceSplitByCount
muDevResourceGenerateDesc
```

#### ASCII flowchart

```text
[Driver API]
    |
    v
[InitPlatform]
    |
    +-- device API? ---------> [Platform::Get device] -> [Device::GetAttribute/GetResource]
    |
    +-- context create? -----> [Device::CreateContext] -> [Context::Init] -> [TLS context stack]
    |
    +-- context sync? -------> [TlsCtxTop] -> [Context::Synchronize] -> [all streams wait]
    |
    +-- resource split? -----> [Device resource] -> [resource desc/split validation]
    v
[return MUresult]
```

#### ASCII sequence

```text
App        Driver API        Platform        Device        Context/HAL
 |             |                |              |              |
 | muCtxCreate |                |              |              |
 |-----------> | InitPlatform   |              |              |
 |             |--------------> |              |              |
 |             | get device     |              |              |
 |             |------------------------------>|              |
 |             | CreateContext                 |              |
 |             |------------------------------>| Context ctor |
 |             |                               |------------->|
 |             | push TLS ctx                  |              |
 |<----------- | status         |              |              |
```

#### 使用核心对象完成工作的方式

```text
- device query 类 API 主要使用 Platform/Device/HAL device。
- context lifecycle 类 API 由 Device 创建 Context，再由 Driver TLS 管理 current context。
- context synchronize 使用 TlsCtxTop 找到 Context，Context 再遍历/同步其 stream 和 command。
- resource split/desc 走 Device resource 能力，GreenContext 后续复用这些 desc 创建隔离 context。
```

#### ModelEvent 建议

```text
PLATFORM_INIT_SPAN
DEVICE_LOOKUP_SPAN
DEVICE_ATTRIBUTE_QUERY_SPAN
CONTEXT_CREATE_SPAN
CONTEXT_DESTROY_SPAN
CONTEXT_TLS_PUSH_POP_POINT
CONTEXT_SYNCHRONIZE_SPAN
DEV_RESOURCE_QUERY_SPAN
DEV_RESOURCE_SPLIT_SPAN
```

### 4.2 Module / Function / Launch 家族

代表 API：

```text
muModuleLoad / muModuleLoadData / muModuleLoadFatBinary
muModuleUnload
muModuleGetFunction
muModuleGetGlobal
muLaunchKernel
muLaunchKernelEx
muLaunchCooperativeKernel
muLaunchHostFunc
```

#### ASCII flowchart

```text
[muModuleLoadData]
    |
    v
[InitPlatform]
    |
    v
[TlsCtxTop]
    |
    v
[Context::CreateModule]
    |
    v
[Module::LoadFatBinary / LoadImage]
    |
    v
[HAL library load]

[muModuleGetFunction]
    |
    v
[InitPlatform -> TlsCtxTop]
    |
    v
[Module::GetFunction / FindSymbolByName]
    |
    v
[Function handle]

[muLaunchKernel]
    |
    v
[InitPlatform]
    |
    v
[Function cast + launch parameter validation]
    |
    v
[Context::GeneralLaunchKernel]
    |
    v
[Context creates/uses GraphKernelNode-like launch params]
    |
    v
[Stream::CmdLaunchKernel]
    |
    v
[DispatchCommand create]
    |
    v
[Context::ResolveDependencyAndQueueCommand]
    |
    v
[Stream::QueueCommand]
    |
    v
[Command::Build -> Submit -> SubmitToQueue]
    |
    v
[HAL queue submit -> M3D/OS]
```

#### ASCII sequence

```text
App       Driver        Context        Stream        DispatchCommand        HAL/M3D
 |           |             |             |                 |                  |
 | launch    |             |             |                 |                  |
 |---------> | Init/TLS    |             |                 |                  |
 |           | GeneralLaunchKernel      |                 |                  |
 |           |-----------> | InfoStream  |                 |                  |
 |           |             |-----------> |                 |                  |
 |           |             | CmdLaunchKernel             |                  |
 |           |             |-----------> | create command  |                  |
 |           |             | ResolveDependencyAndQueueCommand             |
 |           |             |-----------> | QueueCommand    |                  |
 |           |             |             |---------------> | Build cmd buffer |
 |           |             |             |                 |---------------->|
 |           |             |             |---------------> | SubmitToQueue   |
 |           |             |             |                 |---------------->|
 |<--------- | status      |             |                 |                  |
```

#### 使用核心对象完成工作的方式

```text
- Module 把 fatbin/image 转成 HAL library 和 symbol。
- Function 保存 kernel symbol 与 HAL kernel，launch 时用于创建 kernel state。
- Context::GeneralLaunchKernel 是 Driver API 到 Core launch 语义的入口。
- Stream 承接具体队列语义和 default stream 语义。
- DispatchCommand 负责 command buffer build、dependency semaphore、queue submit。
```

#### 当前 MUPTI 覆盖

```text
- Runtime/Driver API callback/activity 可覆盖 musaLaunchKernel/muLaunchKernel enter/exit。
- tracepoints.h 中已有 RegisterKernel、AssignKernelToKick、AssignSubmissionToCorrelation。
- kernel activity 可覆盖最终 kernel execution。
```

#### ModelEvent 建议

```text
MODULE_LOAD_SPAN
FUNCTION_LOOKUP_SPAN
LAUNCH_VALIDATE_SPAN
KERNEL_STATE_CREATE_SPAN
STREAM_RESOLVE_SPAN
COMMAND_CREATE_SPAN(type=DispatchCommand)
DEPENDENCY_RESOLVE_SPAN
STREAM_QUEUE_COMMAND_SPAN
COMMAND_BUILD_SPAN
COMMAND_SUBMIT_SPAN
HAL_QUEUE_SUBMIT_SPAN
SUBMISSION_CORRELATION_POINT
```

### 4.3 Synchronous memory allocation/free 家族

代表 API：

```text
muMemAlloc_v2 / musaMalloc
muMemFree_v2 / musaFree
muMemGetInfo
muPointerGetAttribute / muPointerGetAttributes
```

#### ASCII flowchart

```text
[muMemAlloc_v2]
    |
    v
[InitPlatform]
    |
    v
[validate dptr/bytesize]
    |
    v
[TlsCtxTop]
    |
    v
[build MemoryCreateInfo(type=memoryTypeGeneral)]
    |
    v
[Context::CreateMemory]
    |
    v
[Memory::Init]
    |
    v
[HAL memory allocate/map]
    |
    v
[return device pointer]

[muMemFree_v2]
    |
    v
[InitPlatform]
    |
    v
[Platform::GetMemoryByDevicePointer]
    |
    v
[memory type / offset validation]
    |
    v
[IMemory::Synchronize]
    |
    +-- non-virtual/general -> [Context::DestroyMemory]
    |
    +-- virtual with pool --> [Stream::CmdMemFree(blocking=true)]
                           -> [MemoryPool::DisableAccess]
                           -> [CallbackCommand deferred destroy]
```

#### ASCII sequence

```text
App        Driver        Platform        Context        Memory        HAL Memory
 |           |              |              |              |              |
 | malloc    |              |              |              |              |
 |---------> | Init/TLS     |              |              |              |
 |           | CreateMemory                |              |              |
 |           |---------------------------->| new Memory   |              |
 |           |                             |------------->| Init         |
 |           |                             |              |------------->|
 |<--------- | dptr         |              |              |              |

App        Driver        Platform        Memory        Context/Stream        HAL
 |           |              |              |              |                 |
 | free      |              |              |              |                 |
 |---------> | lookup ptr   |              |              |                 |
 |           |------------> | memory       |              |                 |
 |           | synchronize                 |              |                 |
 |           |---------------------------->|              |                 |
 |           | DestroyMemory or CmdMemFree                |                 |
 |           |------------------------------------------->|                 |
 |<--------- | status       |              |              |                 |
```

#### 使用核心对象完成工作的方式

```text
- allocation 主路径由 Context 创建 Memory，Memory::Init 进入 HAL memory。
- free 主路径先由 Platform 做 pointer -> Memory relation lookup，再根据 memory type 分支。
- common musaMalloc/musaFree 是 general memory，不应默认归入 MemoryPool hit/miss。
- virtual/pool branch 才使用 Stream::CmdMemFree 和 MemoryPool::DisableAccess。
```

#### ModelEvent 建议

```text
MEM_ALLOC_VALIDATE_SPAN
CONTEXT_LOOKUP_SPAN
MEMORY_CREATE_INFO_POINT
CONTEXT_CREATE_MEMORY_SPAN
MEMORY_INIT_SPAN
HAL_ALLOC_SPAN
MEMORY_POINTER_LOOKUP_SPAN
MEM_FREE_TYPE_DECISION
MEMORY_SYNCHRONIZE_SPAN
CONTEXT_DESTROY_MEMORY_SPAN
MEM_POOL_DISABLE_ACCESS_SPAN(virtual/pool branch only)
CALLBACK_COMMAND_CREATE_SPAN(virtual/pool branch only)
API_MEMORY_RELATION
```

### 4.4 Async memory / MemoryPool / VMM 家族

代表 API：

```text
muMemAllocAsync
muMemAllocFromPoolAsync
muMemFreeAsync
muMemPoolCreate / muMemPoolDestroy
muMemPoolSetAttribute / muMemPoolGetAttribute
muMemPoolSetAccess / muMemPoolGetAccess
muMemAddressReserve / muMemAddressFree
muMemCreate / muMemRelease
muMemMap / muMemUnmap
muMemSetAccess / muMemGetAccess
```

#### ASCII flowchart

```text
[muMemAllocAsync / muMemAllocFromPoolAsync]
    |
    v
[InitPlatform -> TlsCtxTop]
    |
    v
[Context::InfoStream]
    |
    v
[build MemoryAllocParameter]
    |
    v
[Stream::CmdMemAlloc(blocking=false)]
    |
    v
[Stream::AsyncMemAlloc]
    |
    v
[MemoryPool::CreateMemory]
    |
    v
[MemoryPool::ModifyAccess]
    |
    v
[Paging/Callback/Command queue]

[muMemFreeAsync]
    |
    v
[lookup Memory]
    |
    v
[Context::InfoStream]
    |
    v
[Stream::CmdMemFree(blocking=false)]
    |
    v
[Stream::AsyncMemFree]
    |
    v
[MemoryPool::DisableAccess]
    |
    v
[CallbackCommand deferred destroy]
```

#### ASCII sequence

```text
Driver        Context        Stream        MemoryPool        Memory        Command
  |             |              |              |               |             |
  | InfoStream  |              |              |               |             |
  |-----------> |------------> |              |               |             |
  | CmdMemAlloc                |              |               |             |
  |--------------------------> | AsyncMemAlloc|               |             |
  |                            |------------> | CreateMemory  |             |
  |                            |              |-------------> |             |
  |                            |------------> | ModifyAccess  |             |
  |                            | create queue command                         |
  |                            |-------------------------------------------->|
```

#### 使用核心对象完成工作的方式

```text
- async alloc/free 是 stream ordered 语义，因此核心对象从 Context/Memory 转到 Stream/MemoryPool/Command。
- MemoryPool 管理真实 pool allocation 和 access modify/disable。
- Stream 确保 alloc/free 与其他 command 的顺序关系。
- CallbackCommand 用于延迟销毁，避免过早释放仍被 GPU 使用的 memory。
```

#### ModelEvent 建议

```text
STREAM_RESOLVE_SPAN
MEM_POOL_LOOKUP_SPAN
MEM_POOL_ALLOC_SPAN
MEM_POOL_HIT_POINT
MEM_POOL_GROW_SPAN
MEM_POOL_MODIFY_ACCESS_SPAN
MEM_POOL_DISABLE_ACCESS_SPAN
ASYNC_MEMORY_COMMAND_CREATE_SPAN
CALLBACK_COMMAND_CREATE_SPAN
API_MEMORY_RELATION
```

### 4.5 Memcpy / Memset / Memory transfer / Memory atomic 家族

代表 API：

```text
muMemcpy / muMemcpyAsync
muMemcpyHtoD / muMemcpyDtoH / muMemcpyDtoD
muMemcpy2D / muMemcpy2DAsync
muMemcpy3D / muMemcpy3DAsync
muMemsetD8/D16/D32
muMemsetD8Async/D16Async/D32Async
memory transfer node/API
memory atomic node/API
```

#### ASCII flowchart

```text
[muMemcpy* sync]
    |
    v
[InitPlatform]
    |
    v
[normalize copy params to MUSA_MEMCPY3D_PEER]
    |
    v
[Context::GeneralMemcpy(ctx, stream=null, wait=true)]
    |
    v
[Context/GraphMemcpyNode legalize]
    |
    v
[Stream::CmdCopyMemory]
    |
    v
[SyncMemcpyCommand or AsyncMemcpyCommand]
    |
    v
[Command build/submit]
    |
    v
[wait if sync]

[muMemcpy* async]
    |
    v
[Context::GeneralMemcpy(ctx, hStream, wait=false)]
    |
    v
[Stream::CmdCopyMemory]
    |
    v
[AsyncMemcpyCommand queued]

[muMemset*]
    |
    v
[build MemsetParameter]
    |
    v
[Context::GeneralMemset]
    |
    v
[Stream::CmdMemset]
    |
    v
[MemsetCommand]
```

#### ASCII sequence

```text
Driver        Context        GraphMemcpy/MemsetNode        Stream        Command        HAL
  |             |                    |                       |             |            |
  | GeneralMemcpy/GeneralMemset      |                       |             |            |
  |-----------> | legalize/check      |                       |             |            |
  |             |-------------------> |                       |             |            |
  |             | CmdCopy/CmdMemset                           |             |            |
  |             |--------------------------------------------> | create      |            |
  |             | ResolveDependencyAndQueueCommand            |------------>| Build      |
  |             |--------------------------------------------> |             |---------->|
  |             |                                            |------------>| Submit     |
  |             |                                            |             |---------->|
```

#### 使用核心对象完成工作的方式

```text
- Driver API 先把 1D/2D/3D/HtoD/DtoH/DtoD 等变体归一化为统一参数。
- Context::GeneralMemcpy/GeneralMemset 做语义入口和 wait/sync 语义选择。
- GraphMemcpyNode/GraphMemsetNode 复用参数 legalize、range check、optimization、copy manager select 等逻辑。
- Stream 生成 copy/memset command 并参与 dependency/queue。
- Command 负责写 HAL cmd buffer 和 submit。
```

#### 当前 MUPTI 覆盖

```text
- tracepoints.h 已有 EnterMemcpy、EnterMemset。
- activity_record_kinds.inc 已启用 MEMCPY、MEMSET、MEMORY_TRANSFER、MEMORY_ATOMIC、MEMORY_ATOMIC_VALUE。
```

#### ModelEvent 建议

```text
COPY_DESCRIPTOR_NORMALIZE_SPAN
COPY_LEGALIZE_SPAN
COPY_RANGE_CHECK_SPAN
COPY_MANAGER_SELECT_SPAN
MEMSET_PARAMETER_BUILD_SPAN
COMMAND_CREATE_SPAN(type=Memcpy/Memset/MemoryAtomic/MemoryTransfer)
COMMAND_BUILD_SPAN
COMMAND_SUBMIT_SPAN
SYNC_WAIT_SPAN(sync variants)
API_ACTIVITY_RELATION
```

### 4.6 Stream / Event / Synchronization 家族

代表 API：

```text
muStreamCreate / muStreamCreateWithPriority
muStreamDestroy
muStreamQuery
muStreamSynchronize
muStreamWaitEvent
muStreamGetFlags / muStreamGetPriority / muStreamGetCtx / muStreamGetId
muEventCreate / muEventDestroy
muEventRecord / muEventRecordWithFlags
muEventSynchronize
muEventQuery
muEventElapsedTime
muCtxSynchronize
```

#### ASCII flowchart

```text
[muStreamCreate]
    |
    v
[InitPlatform -> TlsCtxTop]
    |
    v
[Context::CreateStream]
    |
    v
[Stream::Init -> HAL queue/semaphore]

[muStreamSynchronize]
    |
    v
[InitPlatform -> TlsCtxTop]
    |
    v
[Context::InfoStream]
    |
    v
[Stream::Synchronize]
    |
    v
[Stream::WaitFinish]
    |
    v
[wait inflight/waiting command complete]

[muEventRecord]
    |
    v
[validate Event + resolve Stream]
    |
    v
[Context::RecordEvent / Stream::CmdSetEvent]
    |
    v
[RecordCommand queued]

[muStreamWaitEvent]
    |
    v
[validate Event + resolve Stream]
    |
    v
[Stream::CmdWaitEvent]
    |
    v
[Barrier/Wait command queued]
```

#### ASCII sequence

```text
Driver        Context        Stream        Command/Event        HAL Semaphore
  |             |              |              |                     |
  | StreamSync  |              |              |                     |
  |-----------> | InfoStream   |              |                     |
  |             |------------> | WaitFinish   |                     |
  |             |              |------------> | command Wait        |
  |             |              |              |-------------------> |
  |<----------- | status       |              |                     |

Driver        Context        Stream        Record/WaitCommand   HAL
  |             |              |              |                     |
  | EventRecord |              |              |                     |
  |-----------> | validate     |              |                     |
  |             | CmdSetEvent  |              |                     |
  |             |------------> | queue command|                     |
  |             |              |------------> | Build/Submit        |
```

#### 使用核心对象完成工作的方式

```text
- stream lifecycle 由 Context 创建/销毁 Stream。
- stream query/sync 通过 Context::InfoStream 找到 Stream，再调用 Query/Synchronize/WaitFinish。
- event record/wait 不是单纯修改 Event 对象，而是在 Stream 上插入 command，形成 GPU-side 顺序关系。
- context synchronize 是更大范围的 stream/command wait。
```

#### 当前 MUPTI 覆盖

```text
tracepoints.h 已有：
- CreateStream / DestroyStream
- RegisterEventSynchronize
- RegisterStreamWaitEvent
- RegisterStreamSynchronize
- RegisterContextSynchronize
```

#### ModelEvent 建议

```text
STREAM_CREATE_SPAN
STREAM_DESTROY_SPAN
STREAM_RESOLVE_SPAN
STREAM_QUERY_SPAN
STREAM_WAIT_FINISH_SPAN
SYNC_WAIT_REASON_POINT
EVENT_CREATE_SPAN
EVENT_RECORD_COMMAND_SPAN
EVENT_WAIT_COMMAND_SPAN
EVENT_SYNCHRONIZE_SPAN
COMMAND_WAIT_SPAN
```

### 4.7 Graph 家族

代表 API：

```text
muGraphCreate / muGraphDestroy
muGraphAddKernelNode
muGraphAddMemcpyNode
muGraphAddMemsetNode
muGraphAddHostNode
muGraphAddEventRecordNode / muGraphAddEventWaitNode
muGraphAddMemAllocNode / muGraphAddMemFreeNode
muGraphAddChildGraphNode
muGraphAddDependencies / muGraphRemoveDependencies
muGraphInstantiate / muGraphInstantiateWithFlags / muGraphInstantiateWithParams
muGraphExecUpdate
muGraphLaunch
muGraphExecDestroy
```

#### ASCII flowchart

```text
[muGraphCreate]
    |
    v
[InitPlatform -> TlsCtxTop]
    |
    v
[Context::CreateGraph]

[muGraphAdd*Node]
    |
    v
[InitPlatform -> validate Graph]
    |
    v
[Context::Create*Node]
    |
    v
[GraphNode::Init / legalize / resource prepare]
    |
    v
[Graph::AddGraphNode]
    |
    v
[update topology edges/indegree/root/leaf]

[muGraphInstantiate]
    |
    v
[validate Graph]
    |
    v
[Context::CreateGraphExec]
    |
    v
[GraphExec::Init]
    |
    v
[GraphExec::ResolveGraph / GraphFlatten]
    |
    v
[PrepareAllSubmissions]
    |
    v
[WriteSubmission / Write*Cmd]

[muGraphLaunch]
    |
    v
[validate GraphExec + resolve Stream]
    |
    v
[Stream::CmdLaunchGraph]
    |
    v
[GraphCommand]
    |
    v
[GraphCommand::Build -> Submit -> Execute]
```

#### ASCII sequence

```text
Driver        Context        Graph        GraphNode        GraphExec        Stream        GraphCommand/HAL
  |             |             |              |               |              |              |
  | AddNode     |             |              |               |              |              |
  |-----------> | CreateNode  |              | Init/legalize  |              |              |
  |             |-----------> | AddGraphNode |               |              |              |
  |             |------------>| update edges |               |              |              |
  | Instantiate |             |              |               |              |              |
  |-----------> | CreateGraphExec             |               |              |              |
  |             |-------------------------------------------> | ResolveGraph |              |
  |             |                                            | WriteSubmission             |
  | Launch      |             |              |               |              |              |
  |-----------> | validate    |              |               |              | CmdLaunchGraph              |
  |             |---------------------------------------------------------->| create/submit |
  |             |                                                           |------------->|
```

#### 使用核心对象完成工作的方式

```text
- Graph 保存拓扑，不直接提交。
- GraphNode 保存每个节点的 API 语义，并在 Init/UpdateParams 阶段做参数合法化和资源准备。
- GraphExec 是 instantiate 后的可执行图，负责 flatten、submission 分组、cmd buffer 写入。
- GraphCommand 是把 GraphExec 放到 Stream 上执行的 command。
```

#### 当前 MUPTI 覆盖

```text
- activity_record_kinds.inc 已启用 GRAPH_TRACE。
- tracepoints.h 已有 RegisterGraphTrace。
```

#### ModelEvent 建议

```text
GRAPH_CREATE_SPAN
GRAPH_NODE_CREATE_SPAN
GRAPH_NODE_LEGALIZE_SPAN
GRAPH_ADD_DEPENDENCY_SPAN
GRAPH_INSTANTIATE_SPAN
GRAPH_RESOLVE_SPAN
GRAPH_FLATTEN_SPAN
GRAPH_PREPARE_SUBMISSION_SPAN
GRAPH_WRITE_SUBMISSION_SPAN
GRAPH_LAUNCH_COMMAND_CREATE_SPAN
GRAPH_EXECUTE_SPAN
GRAPH_TRACE_RELATION
```

### 4.8 GreenContext / resource isolation 家族

代表 API：

```text
muDeviceGetDevResource
muDevSmResourceSplitByCount
muDevResourceGenerateDesc
muGreenCtxCreate
muGreenCtxDestroy
muCtxFromGreenCtx
muGreenCtxStreamCreate
muGreenCtxGetDevResource
```

#### ASCII flowchart

```text
[muDeviceGetDevResource]
    |
    v
[InitPlatform]
    |
    v
[Device::GetDeviceResources]

[muDevSmResourceSplitByCount]
    |
    v
[validate resource + split count]
    |
    v
[produce critical/bulk resource partitions]

[muDevResourceGenerateDesc]
    |
    v
[resource desc validation]
    |
    v
[MUdevResourceDesc]

[muGreenCtxCreate]
    |
    v
[InitPlatform]
    |
    v
[Device lookup]
    |
    v
[Device::CreateGreenContext(desc, flags)]
    |
    v
[GreenContext(Context)::Init with resource desc]

[muGreenCtxStreamCreate]
    |
    v
[cast GreenContext]
    |
    v
[GreenContext::CreateStream]
    |
    v
[Stream bound to GreenContext resource]
```

#### ASCII sequence

```text
Driver        Device        GreenContext        Stream        HAL
  |             |               |                |            |
  | GetResource |               |                |            |
  |-----------> | query HAL/resource             |            |
  | Split/Desc  | local validation/desc build    |            |
  | GreenCreate |               |                |            |
  |-----------> | CreateGreenContext             |            |
  |             |-------------> | Context init   |            |
  | StreamCreate|               |                |            |
  |---------------------------> | CreateStream   |----------> |
```

#### 使用核心对象完成工作的方式

```text
- resource API 使用 Device 获取硬件资源视图。
- split/desc API 把资源描述转换成 GreenContext 可消费的 desc。
- GreenContext 继承 Context，因此 stream/module/memory/event 等对象创建仍复用 Context 能力。
- GreenContext 的差异在于 Context 绑定了受限 device resource，后续 Stream/Command submit 受该资源约束。
```

#### ModelEvent 建议

```text
DEV_RESOURCE_QUERY_SPAN
DEV_RESOURCE_SPLIT_SPAN
DEV_RESOURCE_DESC_BUILD_SPAN
GREEN_CONTEXT_CREATE_SPAN
GREEN_CONTEXT_DESTROY_SPAN
GREEN_CONTEXT_STREAM_CREATE_SPAN
GREEN_CONTEXT_RESOURCE_RELATION
```

### 4.9 Peer / external / texture / raytracing / tensor / profiler 其他家族

这些 API 在 Top90 中可能不是 M1 主路径，但 M2/M4 应纳入 source rule 分类。

```text
Peer:
  muDeviceCanAccessPeer / muCtxEnablePeerAccess / muCtxDisablePeerAccess
  使用 Platform/Device/Context/HAL peer capability 和 mapping。

External interop:
  external memory / external semaphore / graphics interop
  使用 imported handle -> Memory/Semaphore/Command relation。

Texture/surface:
  texture object / array / resource descriptor
  使用 Memory/Module/Context/HAL image/resource object。

Raytracing:
  accel structure build/copy/emit、dispatch ray
  使用 AccelStruct*Command / DispatchRayCommand / HAL M3D converter。

Tensor:
  tensor descriptor / operation API
  需要按具体 API 识别是否进入 command submit 或 host-side descriptor path。

Profiler/log/notification:
  多为工具控制或状态 API，应作为 profiling control domain，不与 workload cost 混淆。
```


## 4.10 API 级核心对象交互时序图补充

前面 `4.1` 到 `4.9` 给出了 API family 级 flowchart 和 sequence。为了让 source rule 可以直接落到单 API，本节按“同一核心对象路径可归并”的原则补齐 API 级核心对象交互时序图。

归并规则：同一类 API 如果只是参数形态不同、同步/异步 wait flag 不同、或 v1/v2 wrapper 不同，则共用一张时序图，并在标题中列出覆盖 API。

### 4.10.1 `muInit` / platform 初始化

```text
App        Driver API        Platform        HAL Platform
 |             |                |                |
 | muInit      |                |                |
 |-----------> | InitPlatform   |                |
 |             |--------------> | create/init    |
 |             |                |--------------> |
 |             |                | init devices   |
 |             |<-------------- | status         |
 |<----------- | status         |                |
```

核心对象：`Platform`、`Hal::IPlatform`。

### 4.10.2 `muDeviceGetCount` / `muDeviceGet` / `muDeviceGetName` / `muDeviceGetAttribute` / `muDeviceTotalMem`

```text
App        Driver API        Platform        Device        HAL Device
 |             |                |              |             |
 | device api  |                |              |             |
 |-----------> | InitPlatform   |              |             |
 |             |--------------> |              |             |
 |             | get/validate device index      |             |
 |             |------------------------------> |             |
 |             | attribute/name/mem query       |             |
 |             |------------------------------> | query       |
 |             |                               |-----------> |
 |<----------- | result/status  |              |             |
```

核心对象：`Platform`、`Device`、`Hal::IDevice`。

### 4.10.3 `muCtxCreate` / `muCtxDestroy`

```text
App        Driver API        Platform        Device        Context        HAL
 |             |                |              |              |          |
 | ctx create  |                |              |              |          |
 |-----------> | InitPlatform   |              |              |          |
 |             | get Device     |              |              |          |
 |             |------------------------------> |              |          |
 |             | Device::CreateContext          |              |          |
 |             |------------------------------> | Context init |--------> |
 |             | push/register context          |              |          |
 |<----------- | MUcontext      |              |              |          |

App        Driver API        Context        Stream/Command        HAL
 |             |                |                |                |
 | ctx destroy |                |                |                |
 |-----------> | validate ctx   |                |                |
 |             |------------->  | synchronize/destroy resources      |
 |             |                |--------------> | wait/release   |
 |             | unregister ctx |                |                |
 |<----------- | status         |                |                |
```

核心对象：`Device`、`Context`、`Stream`、`Command`、`HAL`。

### 4.10.4 `muCtxSetCurrent` / `muCtxGetCurrent` / context TLS API

```text
App        Driver API        TLS data        Context
 |             |                |              |
 | set current |                |              |
 |-----------> | validate ctx   |              |
 |             |--------------> | push/swap ctx |
 |<----------- | status         |              |

App        Driver API        TLS data        Context
 |             |                |              |
 | get current |                |              |
 |-----------> | TlsCtxTop      |              |
 |             |--------------> | current ctx   |
 |<----------- | MUcontext      |              |
```

核心对象：`TlsData`、`Context`。

### 4.10.5 `muCtxSynchronize`

```text
App        Driver API        TLS data        Context        Stream        Command
 |             |                |              |              |           |
 | ctx sync    |                |              |              |           |
 |-----------> | InitPlatform   |              |              |           |
 |             | TlsCtxTop      |              |              |           |
 |             |--------------> | Context      |              |           |
 |             | Context::Synchronize           |              |           |
 |             |------------------------------> | WaitFinish   |           |
 |             |                               |------------> | Wait      |
 |<----------- | status         |              |              |           |
```

核心对象：`Context`、`Stream`、`Command`。

### 4.10.6 `muDeviceGetDevResource` / `muDevSmResourceSplitByCount` / `muDevResourceGenerateDesc`

```text
App        Driver API        Platform        Device        Resource Desc
 |             |                |              |              |
 | get resource|                |              |              |
 |-----------> | InitPlatform   |              |              |
 |             | get Device     |              |              |
 |             |----------------------------->  | query SM/dev resource
 |<----------- | MUdevResource  |              |              |

App        Driver API        Resource Desc Builder
 |             |                    |
 | split/desc  |                    |
 |-----------> | validate resource  |
 |             | split by SM count  |
 |             | generate desc      |
 |<----------- | resource/desc      |
```

核心对象：`Device`、`DevResourceDesc`。

### 4.10.7 `muGreenCtxCreate` / `muGreenCtxDestroy` / `muCtxFromGreenCtx` / `muGreenCtxStreamCreate` / `muGreenCtxGetDevResource`

```text
App        Driver API        Platform        Device        GreenContext        Stream
 |             |                |              |              |              |
 | GreenCreate |                |              |              |              |
 |-----------> | InitPlatform   |              |              |              |
 |             | get Device     |              |              |              |
 |             |----------------------------->  | CreateGreenContext(desc)
 |             |                               |------------> | Context init with resource
 |<----------- | MUgreenCtx     |              |              |              |

App        Driver API        GreenContext        Stream
 |             |                    |              |
 | StreamCreate|                    |              |
 |-----------> | cast/validate       |              |
 |             | GreenContext::CreateStream     |
 |             |-------------------> | Stream::Init |
 |<----------- | MUstream            |              |

App        Driver API        GreenContext
 |             |                    |
 | GetResource |                    |
 |-----------> | GetDevResource      |
 |<----------- | MUdevResource       |
```

核心对象：`Device`、`GreenContext`、`Stream`。

### 4.10.8 `muModuleLoad` / `muModuleLoadData` / `muModuleLoadFatBinary` / `muModuleUnload`

```text
App        Driver API        TLS data        Context        Module        HAL Library
 |             |                |              |              |             |
 | load module |                |              |              |             |
 |-----------> | InitPlatform   |              |              |             |
 |             | TlsCtxTop      |              |              |             |
 |             |--------------> | Context      |              |             |
 |             | Context::CreateModule          |              |             |
 |             |------------------------------> | new Module   |             |
 |             | Module::LoadFatBinary/LoadImage|------------> |
 |<----------- | MUmodule       |              |              |             |

App        Driver API        Context        Module        HAL Library
 |             |                |              |             |
 | unload      |                |              |             |
 |-----------> | TlsCtxTop      |              |             |
 |             | Context::DestroyModule         |             |
 |             |----------------------------->  | release     |
 |<----------- | status         |              |             |
```

核心对象：`Context`、`Module`、`Hal::ILibrary`。

### 4.10.9 `muModuleGetFunction` / `muModuleGetGlobal`

```text
App        Driver API        Context        Module        Function/Symbol        Memory
 |             |                |              |              |                 |
 | get func    |                |              |              |                 |
 |-----------> | Init/TLS       |              |              |                 |
 |             | validate module|              |              |                 |
 |             |----------------------------->  | FindSymbol   |                 |
 |             |                               |------------> | Function handle |
 |<----------- | MUfunction     |              |              |                 |

App        Driver API        Module        Symbol/Memory
 |             |                |              |
 | get global  |                |              |
 |-----------> | validate module|              |
 |             | Module::GetGlobal             |
 |             |--------------> | symbol dptr/size
 |<----------- | dptr/bytes     |              |
```

核心对象：`Context`、`Module`、`Function`、`SymbolBase_t`、`Memory`。

### 4.10.10 `muLaunchKernel` / `muLaunchKernelEx` / `muLaunchCooperativeKernel`

```text
App        Driver API        Context        Function        Stream        DispatchCommand        HAL/M3D
 |             |                |              |              |              |                 |
 | launch      |                |              |              |              |                 |
 |-----------> | Init/TLS       |              |              |              |                 |
 |             | validate func/params           |              |              |                 |
 |             | Context::GeneralLaunchKernel   |              |              |                 |
 |             |--------------> | Create launch node/params          |                 |
 |             |                | Function::CreateState              |                 |
 |             |                |------------> |              |              |                 |
 |             |                | Context::InfoStream                |                 |
 |             |                |----------------------------->      |                 |
 |             |                | Stream::CmdLaunchKernel            |                 |
 |             |                |-----------------------------> create DispatchCommand |
 |             |                | ResolveDependencyAndQueueCommand   |---------------->|
 |             |                |              |              | QueueCommand |                 |
 |             |                |              |              |------------> | Build/Submit    |
 |             |                |              |              |              |---------------> |
 |<----------- | status         |              |              |              |                 |
```

核心对象：`Context`、`Function`、`Stream`、`DispatchCommand`、`Hal::ICmdBuffer`、`Hal::IQueue`。

### 4.10.11 `muLaunchHostFunc`

```text
App        Driver API        Context        Stream        Host callback command
 |             |                |              |              |
 | host func   |                |              |              |
 |-----------> | Init/TLS       |              |              |
 |             | Context::InfoStream            |              |
 |             |--------------> |------------> |              |
 |             | GeneralLaunchHostFunc          |              |
 |             |--------------> | Stream queue callback command      |
 |             |                |------------> | execute after prior stream work    |
 |<----------- | status         |              |              |
```

核心对象：`Context`、`Stream`、callback command。

### 4.10.12 `muMemAlloc_v2` / `muMemAlloc`

```text
App        Driver API        TLS data        Context        Memory        HAL Memory
 |             |                |              |              |              |
 | mem alloc   |                |              |              |              |
 |-----------> | InitPlatform   |              |              |              |
 |             | validate size  |              |              |              |
 |             | TlsCtxTop      |              |              |              |
 |             |--------------> | Context      |              |              |
 |             | build MemoryCreateInfo         |              |              |
 |             | Context::CreateMemory          |              |              |
 |             |------------------------------> | new Memory   |              |
 |             |                               |------------> | allocate/map |
 |<----------- | dptr/status    |              |              |              |
```

核心对象：`Context`、`Memory`、`Hal::IMemory`。

### 4.10.13 `muMemFree_v2` / `muMemFree`

```text
App        Driver API        Platform        Memory        Context        Stream/CallbackCommand
 |             |                |              |              |              |
 | mem free    |                |              |              |              |
 |-----------> | InitPlatform   |              |              |              |
 |             | GetMemoryByDevicePointer       |              |              |
 |             |--------------> | memory+offset |              |              |
 |             | type/offset validation         |              |              |
 |             | IMemory::Synchronize           |              |              |
 |             |----------------------------->  |              |              |
 |             | non-virtual: Context::DestroyMemory           |              |
 |             |--------------------------------------------> |              |
 |             | virtual/pool: Stream::CmdMemFree(blocking)    |              |
 |             |--------------------------------------------> | CallbackCommand
 |<----------- | status         |              |              |              |
```

核心对象：`Platform`、`Memory`、`Context`、`Stream`、`MemoryPool`、`CallbackCommand`。

### 4.10.14 `muMemGetInfo` / pointer attribute APIs

```text
App        Driver API        Platform        Device/Memory        HAL
 |             |                |              |                  |
 | query memory|                |              |                  |
 |-----------> | InitPlatform   |              |                  |
 |             | get ctx/device or pointer lookup              |
 |             |--------------> |------------> |                  |
 |             | query free/total or memory metadata            |
 |             |-----------------------------> | HAL query       |
 |<----------- | result/status  |              |                  |
```

核心对象：`Platform`、`Device`、`Memory`、`Hal::IDevice`、`Hal::IMemory`。

### 4.10.15 `muMemAllocAsync` / `muMemAllocFromPoolAsync`

```text
App        Driver API        Context        Stream        MemoryPool        Memory        Command
 |             |                |              |              |               |           |
 | alloc async |                |              |              |               |           |
 |-----------> | Init/TLS       |              |              |               |           |
 |             | Context::InfoStream            |              |               |           |
 |             |--------------> |------------> |              |               |           |
 |             | build MemoryAllocParameter     |              |               |           |
 |             | Stream::CmdMemAlloc            |              |               |           |
 |             |-----------------------------> | AsyncMemAlloc |               |
 |             |                               |------------> | CreateMemory  |
 |             |                               |              |-------------> |
 |             |                               |------------> | ModifyAccess  |
 |             |                               | queue ordered command          |
 |<----------- | dptr/status    |              |              |               |           |
```

核心对象：`Context`、`Stream`、`MemoryPool`、`Memory`、ordered command。

### 4.10.16 `muMemFreeAsync`

```text
App        Driver API        Platform        Context        Stream        MemoryPool        CallbackCommand
 |             |                |              |              |              |              |
 | free async  |                |              |              |              |              |
 |-----------> | InitPlatform   |              |              |              |              |
 |             | pointer lookup |              |              |              |              |
 |             |--------------> | Memory       |              |              |              |
 |             | Context::InfoStream            |              |              |              |
 |             |-----------------------------> |              |              |              |
 |             | Stream::CmdMemFree             |              |              |              |
 |             |-----------------------------> | AsyncMemFree |              |              |
 |             |                               |------------> | DisableAccess |              |
 |             |                               | create deferred destroy command|------------>|
 |<----------- | status         |              |              |              |              |
```

核心对象：`Platform`、`Memory`、`Context`、`Stream`、`MemoryPool`、`CallbackCommand`。

### 4.10.17 `muMemPoolCreate` / `muMemPoolDestroy` / `muMemPoolSetAttribute` / `muMemPoolGetAttribute` / `muMemPoolSetAccess` / `muMemPoolGetAccess`

```text
App        Driver API        Platform/Device        MemoryPool        HAL MemoryPool
 |             |                    |                  |                |
 | pool create |                    |                  |                |
 |-----------> | InitPlatform       |                  |                |
 |             | validate props/device             |                |
 |             |------------------> | new MemoryPool   |                |
 |             |                    |----------------> | create/init    |
 |<----------- | pool handle        |                  |                |

App        Driver API        MemoryPool        HAL MemoryPool
 |             |                  |                |
 | attr/access |                  |                |
 |-----------> | validate pool      |                |
 |             | Set/GetAttribute or Set/GetAccess |
 |             |---------------->  | HAL operation   |
 |<----------- | status/value       |                |
```

核心对象：`Device`、`MemoryPool`、`Hal::IMemoryPool`。

### 4.10.18 VMM APIs：`muMemAddressReserve` / `muMemAddressFree` / `muMemCreate` / `muMemRelease` / `muMemMap` / `muMemUnmap` / `muMemSetAccess` / `muMemGetAccess`

```text
App        Driver API        Platform        Context/Device        Memory        HAL Memory
 |             |                |                 |              |              |
 | VMM api     |                |                 |              |              |
 |-----------> | InitPlatform   |                 |              |              |
 |             | validate address/size/handle     |              |              |
 |             | resolve context/device/memory     |              |              |
 |             |--------------> |---------------> |              |              |
 |             | reserve/create/map/access operation             |
 |             |--------------------------------> | HAL VMM op   |
 |<----------- | handle/status   |                 |              |              |
```

核心对象：`Platform`、`Context`、`Device`、`Memory`、`Hal::IMemory`。

### 4.10.19 `muMemcpy*` sync APIs

覆盖：`muMemcpy`、`muMemcpyHtoD`、`muMemcpyDtoH`、`muMemcpyDtoD`、`muMemcpy2D`、`muMemcpy3D` 等同步变体。

```text
App        Driver API        Context        GraphMemcpyNode        Stream        SyncMemcpyCommand        HAL/M3D
 |             |                |                 |                  |              |                  |
 | memcpy sync |                |                 |                  |              |                  |
 |-----------> | Init/TLS       |                 |                  |              |                  |
 |             | normalize copy descriptor        |                  |              |                  |
 |             | Context::GeneralMemcpy(wait=true)|                  |              |                  |
 |             |--------------> | legalize/range/check/select copy manager      |                  |
 |             |                |---------------> |                  |              |                  |
 |             |                | Stream::CmdCopyMemory                         |                  |
 |             |                |--------------------------------------------> | create/build/submit|
 |             |                |                                                |-----------------> |
 |             | wait complete                    |                  |              |                  |
 |<----------- | status         |                 |                  |              |                  |
```

核心对象：`Context`、`GraphMemcpyNode`、`Stream`、`SyncMemcpyCommand`、`Hal::ICmdBuffer`、`Hal::IQueue`。

### 4.10.20 `muMemcpy*Async` APIs

覆盖：`muMemcpyAsync`、`muMemcpyHtoDAsync`、`muMemcpyDtoHAsync`、`muMemcpyDtoDAsync`、`muMemcpy2DAsync`、`muMemcpy3DAsync` 等异步变体。

```text
App        Driver API        Context        GraphMemcpyNode        Stream        AsyncMemcpyCommand        HAL/M3D
 |             |                |                 |                  |              |                   |
 | memcpy async|                |                 |                  |              |                   |
 |-----------> | Init/TLS       |                 |                  |              |                   |
 |             | normalize descriptor             |                  |              |                   |
 |             | GeneralMemcpy(wait=false)        |                  |              |                   |
 |             |--------------> | legalize/select  |                  |              |                   |
 |             |                |----------------> |                  |              |                   |
 |             |                | Stream::CmdCopyMemory                         |                   |
 |             |                |--------------------------------------------> | queue command     |
 |             |                |                                                | Build/Submit async|
 |<----------- | status         |                 |                  |              |                   |
```

核心对象：`Context`、`GraphMemcpyNode`、`Stream`、`AsyncMemcpyCommand`、`HAL/M3D`。

### 4.10.21 `muMemsetD8/D16/D32` sync/async APIs

```text
App        Driver API        Context        GraphMemsetNode        Stream        MemsetCommand        HAL/M3D
 |             |                |                 |                  |              |              |
 | memset      |                |                 |                  |              |              |
 |-----------> | Init/TLS       |                 |                  |              |              |
 |             | build MemsetParameter            |                  |              |              |
 |             | Context::GeneralMemset(wait flag)|                  |              |              |
 |             |--------------> | legalize/update params            |              |              |
 |             |                |----------------> |                  |              |              |
 |             |                | Stream::CmdMemset                  |              |              |
 |             |                |---------------------------------> | create/build/submit|
 |             | wait if sync                    |                  |              |              |
 |<----------- | status         |                 |                  |              |              |
```

核心对象：`Context`、`GraphMemsetNode`、`Stream`、`MemsetCommand`、`HAL/M3D`。

### 4.10.22 memory transfer / memory atomic APIs and graph nodes

```text
App        Driver/Graph API        Context        GraphMemory*Node        Stream        Memory*Command        HAL/M3D
 |              |                    |                 |                  |              |              |
 | mem op       |                    |                 |                  |              |              |
 |------------> | Init/TLS/validate   |                 |                  |              |              |
 |              | create/update memory transfer/atomic node       |              |              |
 |              |-------------------> |---------------->|                  |              |              |
 |              | if immediate stream op: CmdMemoryTransfer/Atomic |              |              |
 |              |-----------------------------------------------> | create/build/submit|
 |              | if graph path: Graph::AddGraphNode then GraphExec writes cmd later     |
 |<------------ | status              |                 |                  |              |              |
```

核心对象：`Context`、`GraphMemoryTransferNode`、`GraphMemoryAtomicNode`、`Stream`、`MemoryTransferCommand`、`MemoryAtomicCommand`。

### 4.10.23 `muStreamCreate` / `muStreamCreateWithPriority` / `muStreamDestroy`

```text
App        Driver API        Context        Stream        HAL Queue/Semaphore
 |             |                |              |              |
 | create      |                |              |              |
 |-----------> | Init/TLS       |              |              |
 |             | Context::CreateStream           |              |
 |             |--------------> | new Stream   |              |
 |             |                |------------> | init queue/semaphore
 |<----------- | MUstream       |              |              |

App        Driver API        Context        Stream        Command/HAL
 |             |                |              |              |
 | destroy     |                |              |              |
 |-----------> | resolve stream |              |              |
 |             |--------------> |------------> | WaitFinish/release
 |             | Context::DestroyStream          |              |
 |<----------- | status         |              |              |
```

核心对象：`Context`、`Stream`、`Hal::IQueue`、`Hal::ISemaphore`。

### 4.10.24 `muStreamQuery` / `muStreamSynchronize`

```text
App        Driver API        Context        Stream        Command/HAL Semaphore
 |             |                |              |              |
 | query/sync  |                |              |              |
 |-----------> | Init/TLS       |              |              |
 |             | Context::InfoStream             |              |
 |             |--------------> |------------> |              |
 |             | Stream::Query or Synchronize    |              |
 |             |----------------------------->  | check/wait timeline
 |             |                               |-----------> |
 |<----------- | status         |              |              |
```

核心对象：`Context`、`Stream`、`Command`、`Hal::ISemaphore`。

### 4.10.25 `muStreamWaitEvent`

```text
App        Driver API        Context        Stream        Event        Barrier/WaitCommand        HAL
 |             |                |              |            |              |              |
 | wait event  |                |              |            |              |              |
 |-----------> | Init/TLS       |              |            |              |              |
 |             | validate event + resolve stream |            |              |              |
 |             |--------------> |------------> | Event      |              |              |
 |             | Stream::CmdWaitEvent            |            |              |              |
 |             |----------------------------->  |----------> | create/build |----->|
 |<----------- | status         |              |            |              |              |
```

核心对象：`Context`、`Stream`、`Event`、wait/barrier command、`HAL`。

### 4.10.26 `muEventCreate` / `muEventDestroy`

```text
App        Driver API        Context        Event
 |             |                |              |
 | event create|                |              |
 |-----------> | Init/TLS       |              |
 |             | Context::CreateEvent            |
 |             |--------------> | new Event    |
 |<----------- | MUevent        |              |

App        Driver API        Context        Event
 |             |                |              |
 | destroy     |                |              |
 |-----------> | validate event |              |
 |             | Context::DestroyEvent           |
 |             |--------------> | release      |
 |<----------- | status         |              |
```

核心对象：`Context`、`Event`。

### 4.10.27 `muEventRecord` / `muEventRecordWithFlags`

```text
App        Driver API        Context        Stream        Event        RecordCommand        HAL
 |             |                |              |            |              |          |
 | record      |                |              |            |              |          |
 |-----------> | Init/TLS       |              |            |              |          |
 |             | validate event + resolve stream |            |              |          |
 |             | Context::RecordEvent            |            |              |          |
 |             |--------------> | Stream::CmdSetEvent         |              |          |
 |             |                |------------> |----------> | create/build |-------->|
 |<----------- | status         |              |            |              |          |
```

核心对象：`Context`、`Stream`、`Event`、`RecordCommand`、`HAL`。

### 4.10.28 `muEventSynchronize` / `muEventQuery` / `muEventElapsedTime`

```text
App        Driver API        Context        Event        HAL Semaphore/Timestamp
 |             |                |              |              |
 | event query |                |              |              |
 |-----------> | Init/TLS       |              |              |
 |             | validate event |              |              |
 |             |--------------> | Event::Query/Synchronize/ElapsedTime
 |             |                |------------> | wait/query timestamp
 |<----------- | result/status  |              |              |
```

核心对象：`Context`、`Event`、`Hal::ISemaphore` / timestamp resource。

### 4.10.29 `muGraphCreate` / `muGraphDestroy`

```text
App        Driver API        Context        Graph
 |             |                |              |
 | graph create|                |              |
 |-----------> | Init/TLS       |              |
 |             | Context::CreateGraph            |
 |             |--------------> | new Graph    |
 |<----------- | MUgraph        |              |

App        Driver API        Context        Graph
 |             |                |              |
 | destroy     |                |              |
 |-----------> | validate graph |              |
 |             | Context destroys graph          |
 |             |--------------> | release nodes/resource
 |<----------- | status         |              |
```

核心对象：`Context`、`Graph`。

### 4.10.30 `muGraphAddKernelNode` / `muGraphAddMemcpyNode` / `muGraphAddMemsetNode` / `muGraphAddHostNode`

```text
App        Driver API        Context        GraphNode        Graph        Module/Function/Memory
 |             |                |              |              |              |
 | add node    |                |              |              |              |
 |-----------> | Init/TLS       |              |              |              |
 |             | validate graph/dependencies     |              |              |
 |             | Context::Create*Node            |              |              |
 |             |--------------> |------------> | Init/legalize/check resources  |
 |             |                |              |------------> | AddGraphNode |
 |             |                |              |              | update edges/indegree/root/leaf
 |<----------- | node/status    |              |              |              |
```

核心对象：`Context`、`GraphNode` 子类、`Graph`、`Function`、`Memory`。

### 4.10.31 `muGraphAddEventRecordNode` / `muGraphAddEventWaitNode` / `muGraphAddMemAllocNode` / `muGraphAddMemFreeNode` / child/conditional/external semaphore nodes

```text
App        Driver API        Context        Event/Memory        GraphNode        Graph
 |             |                |              |                 |              |
 | add special |                |              |                 |              |
 |-----------> | Init/TLS       |              |                 |              |
 |             | validate graph + object handles |                 |              |
 |             |--------------> | Event/Memory validation            |              |
 |             | Context::Create*Node                              |              |
 |             |-------------------------------> | Init/update params                 |
 |             |                                               |--> AddGraphNode
 |<----------- | node/status    |              |                 |              |
```

核心对象：`Context`、`Event`、`Memory`、`GraphNode` 子类、`Graph`。

### 4.10.32 `muGraphAddDependencies` / `muGraphRemoveDependencies`

```text
App        Driver API        Context        Graph        GraphNode
 |             |                |              |             |
 | dep update  |                |              |             |
 |-----------> | Init/TLS       |              |             |
 |             | validate graph/nodes            |             |
 |             |--------------> |------------> | validate node membership
 |             |                | Graph::AddDependencies/RemoveDependencies
 |             |                |------------> | update edges/indegree/root/leaf
 |<----------- | status         |              |             |
```

核心对象：`Context`、`Graph`、`GraphNode`。

### 4.10.33 `muGraphInstantiate` / `muGraphInstantiateWithFlags` / `muGraphInstantiateWithParams`

```text
App        Driver API        Context        Graph        GraphExec        GraphNode        HAL CmdBuffer
 |             |                |              |              |                |              |
 | instantiate |                |              |              |                |              |
 |-----------> | Init/TLS       |              |              |                |              |
 |             | validate graph |              |              |                |              |
 |             | Context::CreateGraphExec        |              |                |              |
 |             |--------------> |------------> | GraphExec::Init |              |              |
 |             |                               | ResolveGraph/Flatten over nodes    |              |
 |             |                               |---------------->|              |              |
 |             |                               | PrepareAllSubmissions              |              |
 |             |                               | WriteSubmission/Write*Cmd          |------------>|
 |<----------- | graphExec      |              |              |                |              |
```

核心对象：`Context`、`Graph`、`GraphExec`、`GraphNode`、`Hal::ICmdBuffer`。

### 4.10.34 `muGraphExecUpdate` / graph exec node set-param APIs

```text
App        Driver API        Context        GraphExec        Graph        GraphNode
 |             |                |              |              |              |
 | update exec |                |              |              |              |
 |-----------> | Init/TLS       |              |              |              |
 |             | validate graphExec/node          |              |              |
 |             |--------------> |------------> | Set*NodeParams/Update
 |             |                               | validate compatibility with GraphNode
 |             |                               |-----------------------------> |
 |             |                               | refresh flattened/submission state if needed
 |<----------- | status/result  |              |              |              |
```

核心对象：`Context`、`GraphExec`、`Graph`、`GraphNode`。

### 4.10.35 `muGraphLaunch` / `muGraphExecDestroy`

```text
App        Driver API        Context        Stream        GraphExec        GraphCommand        HAL/M3D
 |             |                |              |              |                |             |
 | graph launch|                |              |              |                |             |
 |-----------> | Init/TLS       |              |              |                |             |
 |             | validate graphExec + resolve stream             |             |
 |             |--------------> |------------> |              |                |             |
 |             | Stream::CmdLaunchGraph          |              |                |             |
 |             |----------------------------->  |------------> | create/build  |
 |             |                               |              |-------------> | Execute/Submit
 |<----------- | status         |              |              |                |             |

App        Driver API        Context        GraphExec
 |             |                |              |
 | exec destroy|                |              |
 |-----------> | validate exec  |              |
 |             | Context destroys GraphExec      |
 |<----------- | status         |              |
```

核心对象：`Context`、`Stream`、`GraphExec`、`GraphCommand`、`HAL/M3D`。

### 4.10.36 Peer APIs：`muDeviceCanAccessPeer` / `muCtxEnablePeerAccess` / `muCtxDisablePeerAccess`

```text
App        Driver API        Platform        Device(src/dst)        Context        HAL
 |             |                |                    |              |          |
 | peer api    |                |                    |              |          |
 |-----------> | InitPlatform   |                    |              |          |
 |             | resolve devices/context          |              |          |
 |             |--------------> |------------------> |              |          |
 |             | capability query or enable/disable peer mapping  |          |
 |             |-----------------------------------------------> | HAL peer op
 |<----------- | result/status  |                    |              |          |
```

核心对象：`Platform`、`Device`、`Context`、`HAL`。

### 4.10.37 External memory / external semaphore / graphics interop APIs

```text
App        Driver API        Platform        Context        Memory/Semaphore        Command/HAL
 |             |                |              |              |                    |
 | external api|                |              |              |                    |
 |-----------> | InitPlatform   |              |              |                    |
 |             | import/export external handle   |              |                    |
 |             |--------------> |------------> | create Memory/Semaphore wrapper   |
 |             | if wait/signal semaphore: stream command          |                    |
 |             |-----------------------------------------------> | build/submit      |
 |<----------- | handle/status  |              |              |                    |
```

核心对象：`Platform`、`Context`、`Memory`、`Hal::ISemaphore`、external semaphore command。

### 4.10.38 Texture / surface / array resource APIs

```text
App        Driver API        Context        Memory/Array        Texture/Surface resource        HAL
 |             |                |              |                    |              |
 | texture api |                |              |                    |              |
 |-----------> | Init/TLS       |              |                    |              |
 |             | validate descriptor + memory/array handles       |              |
 |             |--------------> |------------> | create/update resource descriptor |
 |             |                                            |----> HAL resource op
 |<----------- | handle/status  |              |                    |              |
```

核心对象：`Context`、`Memory`/array resource、texture/surface object、`HAL`。

### 4.10.39 Raytracing APIs：accel structure build/copy/emit and dispatch ray

```text
App        Driver API        Context        Stream        AccelStruct/DispatchRayCommand        HAL M3D
 |             |                |              |                    |                    |
 | ray api     |                |              |                    |                    |
 |-----------> | Init/TLS       |              |                    |                    |
 |             | validate build/copy/dispatch params             |                    |
 |             | Context::InfoStream             |                    |                    |
 |             |--------------> |------------> |                    |                    |
 |             | create AccelStruct*Command or DispatchRayCommand |                    |
 |             |----------------------------->  | Build converter/cmd buffer         |
 |             |                               |----------------------------------> |
 |             |                               | SubmitToQueue                      |
 |<----------- | status         |              |                    |                    |
```

核心对象：`Context`、`Stream`、`AccelStruct*Command`、`DispatchRayCommand`、`Hal::M3d` converter/queue。

### 4.10.40 Tensor descriptor / tensor operation APIs

```text
App        Driver API        Context/Device        Tensor descriptor/resource        Optional Command/HAL
 |             |                    |                         |                  |
 | tensor api  |                    |                         |                  |
 |-----------> | InitPlatform/TLS    |                         |                  |
 |             | validate descriptor/device capability            |                  |
 |             |------------------> | create/update descriptor/resource |
 |             | if GPU work required: stream command build/submit |---------------> |
 |<----------- | handle/status       |                         |                  |
```

核心对象：`Context`、`Device`、tensor descriptor/resource、可选 `Stream`/`Command`/`HAL`。

### 4.10.41 Profiler / log / notification / error API

```text
App        Driver API        Platform/Settings        Tooling state        Optional MUPTI
 |             |                    |                    |              |
 | control api |                    |                    |              |
 |-----------> | InitPlatform if needed           |              |
 |             | read/update settings or tooling state            |
 |             |------------------> |-----------------> | notify tool if needed
 |<----------- | result/status       |                    |              |
```

核心对象：`Platform settings`、tooling/profiler state、可选 MUPTI 控制面。

### 4.10.42 API 到核心对象覆盖检查表

| API / API group | 核心对象交互时序图 | 主要对象 |
| --- | --- | --- |
| `muInit` | 4.10.1 | Platform / HAL Platform |
| device query APIs | 4.10.2 | Platform / Device / HAL Device |
| context create/destroy | 4.10.3 | Device / Context / Stream / Command |
| context current TLS APIs | 4.10.4 | TlsData / Context |
| `muCtxSynchronize` | 4.10.5 | Context / Stream / Command |
| device resource APIs | 4.10.6 | Device / DevResourceDesc |
| GreenContext APIs | 4.10.7 | Device / GreenContext / Stream |
| module load/unload | 4.10.8 | Context / Module / HAL Library |
| module function/global lookup | 4.10.9 | Module / Function / Symbol / Memory |
| kernel launch APIs | 4.10.10 | Context / Function / Stream / DispatchCommand |
| host function launch | 4.10.11 | Context / Stream / callback command |
| sync memory allocation | 4.10.12 | Context / Memory / HAL Memory |
| sync memory free | 4.10.13 | Platform / Memory / Context / Stream |
| memory query / pointer attributes | 4.10.14 | Platform / Device / Memory |
| async memory allocation | 4.10.15 | Context / Stream / MemoryPool / Memory |
| async memory free | 4.10.16 | Platform / Stream / MemoryPool / CallbackCommand |
| memory pool APIs | 4.10.17 | Device / MemoryPool / HAL MemoryPool |
| VMM APIs | 4.10.18 | Platform / Context / Device / Memory |
| sync memcpy APIs | 4.10.19 | Context / GraphMemcpyNode / Stream / SyncMemcpyCommand |
| async memcpy APIs | 4.10.20 | Context / GraphMemcpyNode / Stream / AsyncMemcpyCommand |
| memset APIs | 4.10.21 | Context / GraphMemsetNode / Stream / MemsetCommand |
| memory transfer / atomic | 4.10.22 | GraphMemory*Node / Stream / Memory*Command |
| stream create/destroy | 4.10.23 | Context / Stream / HAL Queue |
| stream query/sync | 4.10.24 | Context / Stream / Command |
| stream wait event | 4.10.25 | Context / Stream / Event / wait command |
| event create/destroy | 4.10.26 | Context / Event |
| event record | 4.10.27 | Context / Stream / Event / RecordCommand |
| event query/sync/elapsed | 4.10.28 | Context / Event / HAL timestamp/semaphore |
| graph create/destroy | 4.10.29 | Context / Graph |
| graph regular node add | 4.10.30 | Context / GraphNode / Graph |
| graph special node add | 4.10.31 | Context / Event / Memory / GraphNode / Graph |
| graph dependencies | 4.10.32 | Context / Graph / GraphNode |
| graph instantiate | 4.10.33 | Context / Graph / GraphExec / HAL CmdBuffer |
| graph exec update | 4.10.34 | Context / GraphExec / GraphNode |
| graph launch/destroy | 4.10.35 | Context / Stream / GraphExec / GraphCommand |
| peer APIs | 4.10.36 | Platform / Device / Context / HAL |
| external interop APIs | 4.10.37 | Platform / Context / Memory / Semaphore / Command |
| texture/surface APIs | 4.10.38 | Context / Memory / texture/surface resource |
| raytracing APIs | 4.10.39 | Context / Stream / ray commands / HAL M3D |
| tensor APIs | 4.10.40 | Context / Device / tensor resource / optional Command |
| profiler/log/notification/error APIs | 4.10.41 | Platform settings / tooling state / MUPTI |
```


## 4.11 API family 到 source rule 的落地细化

本节把 `4.10` 的核心对象时序图进一步细化为 source rule 可落地的信息：源码锚点、关键分支、必须 ModelEvent、必须 relation 和成本项。后续可以直接按这里的 family 模板批量生成 `source_rules/*.yaml`。

### 4.11.1 Device / Context / Resource

#### 源码锚点

```text
src/driver/mu_device.cpp
src/driver/mu_context.cpp
src/driver/mu_greencontext.cpp
src/driver/internal.h: InitPlatform / TlsCtxTop / ApiTrace
src/musa/core/platform.h / platform.cpp
src/musa/core/device.h / device.cpp
src/musa/core/context.h / context.cpp
src/musa/core/greenContext.h / greenContext.cpp
src/hal/halDevice.h
```

#### 关键分支

```text
device query:
  InitPlatform -> Platform device table -> Device/HAL attribute query

context create:
  InitPlatform -> Device::CreateContext -> Context init -> TLS current context

context sync:
  TlsCtxTop -> Context::Synchronize -> Stream::WaitFinish -> Command::Wait

resource / GreenContext:
  Device::GetDeviceResources -> resource split/desc -> Device::CreateGreenContext -> GreenContext(Context)
```

#### 必须 ModelEvent

| 事件 | 类型 | 建议位置 | 说明 |
| --- | --- | --- | --- |
| `PLATFORM_INIT_SPAN` | span | `InitPlatform()` | 首次初始化和后续 fast path 分开记录 |
| `DEVICE_LOOKUP_SPAN` | span | device index/handle lookup | 解释 device API host 侧开销 |
| `DEVICE_ATTRIBUTE_QUERY_SPAN` | span | `Device::GetAttribute` / HAL query | 区分 cache query 和 HAL query |
| `CONTEXT_CREATE_SPAN` | span | `Device::CreateContext` | context lifecycle 成本 |
| `CONTEXT_DESTROY_SPAN` | span | `Device/Context::DestroyContext` | destroy 释放成本 |
| `CONTEXT_TLS_UPDATE_POINT` | instant | TLS push/pop/swap | current context relation |
| `CONTEXT_SYNCHRONIZE_SPAN` | span | `Context::Synchronize` | context 级等待 |
| `DEV_RESOURCE_QUERY_SPAN` | span | `Device::GetDeviceResources` | GreenContext resource 前置 |
| `DEV_RESOURCE_SPLIT_SPAN` | span | SM resource split | resource partition 成本 |
| `GREEN_CONTEXT_CREATE_SPAN` | span | `Device::CreateGreenContext` | GreenContext 创建成本 |

#### 必须 relation

```text
API -> Platform
API -> Device
API -> Context
Context -> Device
GreenContext -> DeviceResourceDesc
ContextSynchronize -> Stream
Stream -> Command(waited)
```

#### 成本拆分

```text
T_device_context_api =
  T_driver_wrapper
+ T_platform_init
+ T_device_or_context_lookup
+ T_core_object_operation
+ T_hal_query_or_init
+ T_sync_wait_if_any
```

### 4.11.2 Module / Function / Launch

#### 源码锚点

```text
src/driver/mu_module.cpp: muapiModuleLoadData / muapiModuleGetFunction / muapiLaunchKernel / muapiLaunchKernelEx
src/musa/core/context.h / context.cpp: GeneralLaunchKernel / CreateModule / DestroyModule
src/musa/core/module.h / module.cpp: LoadFatBinary / LoadImage / GetFunction / GetGlobal
src/musa/core/symbol.h / symbol.cpp: Function / CreateState
src/musa/core/stream.h / stream.cpp: CmdLaunchKernel / QueueCommand
src/musa/core/command/dispatchCommand.h / dispatchCommand.cpp
src/musa/core/command/command.h / command.cpp
src/driver/mupti/tracepoints.h: RegisterKernel / AssignKernelToKick / AssignSubmissionToCorrelation
```

#### 关键分支

```text
module load:
  TlsCtxTop -> Context::CreateModule -> Module::LoadFatBinary/LoadImage -> HAL library

function lookup:
  Module::GetFunction -> FindSymbolByName -> Function handle

launch:
  validate function/grid/block/sharedMem/params
  -> Context::GeneralLaunchKernel
  -> Context::InfoStream
  -> Function::CreateState
  -> Stream::CmdLaunchKernel
  -> DispatchCommand
  -> dependency resolve
  -> Stream::QueueCommand
  -> Command::Build / Submit
  -> HAL/M3D/OS submit
```

#### 必须 ModelEvent

| 事件 | 类型 | 建议位置 | 说明 |
| --- | --- | --- | --- |
| `MODULE_CREATE_SPAN` | span | `Context::CreateModule` | module object 创建 |
| `MODULE_LOAD_SPAN` | span | `Module::LoadFatBinary/LoadImage` | fatbin/image/HAL library load |
| `MODULE_DESTROY_SPAN` | span | `Context::DestroyModule` | module unload |
| `FUNCTION_LOOKUP_SPAN` | span | `Module::GetFunction` | symbol lookup 成本 |
| `FUNCTION_GLOBAL_LOOKUP_SPAN` | span | `Module::GetGlobal` | global symbol lookup |
| `LAUNCH_VALIDATE_SPAN` | span | `muapiLaunchKernel*` 参数校验 | host validation |
| `KERNEL_STATE_CREATE_SPAN` | span | `Function::CreateState` | launch 内部关键成本 |
| `STREAM_RESOLVE_SPAN` | span | `Context::InfoStream` | stream/default stream 分支 |
| `DISPATCH_COMMAND_CREATE_SPAN` | span | `Stream::CmdLaunchKernel` | command 创建 |
| `DEPENDENCY_RESOLVE_SPAN` | span | `Context::ResolveDependencyAndQueueCommand` | 依赖解析 |
| `COMMAND_BUILD_SPAN` | span | `DispatchCommand::Build` | cmd buffer build |
| `COMMAND_SUBMIT_SPAN` | span | `DispatchCommand::Submit` / `Command::SubmitToQueue` | submit 前后 |
| `SUBMISSION_CORRELATION_POINT` | instant | `AssignSubmissionToCorrelation` 附近 | API/kernel/submission 关系 |

#### 必须 relation

```text
Runtime API -> Driver API(correlation_id)
Driver API -> Context
Driver API -> Function
Function -> Module
Launch API -> Stream
Launch API -> DispatchCommand
DispatchCommand -> Submission
Submission -> Kernel activity
Submission -> Stream
```

#### 成本拆分

```text
T_launch_api =
  T_runtime_wrapper
+ T_driver_wrapper
+ T_launch_validate
+ T_context_lookup
+ T_function_kernel_state
+ T_stream_resolve
+ T_dependency_resolve
+ T_command_create
+ T_command_build
+ T_command_submit
+ T_hal_m3d_os_submit
```

### 4.11.3 Synchronous memory allocation/free

#### 源码锚点

```text
src/driver/mu_memory.cpp: muapiMemAlloc_v2 / muapiMemFree_v2 / pointer query APIs
src/musa/core/context.h / context.cpp: CreateMemory / DestroyMemory
src/musa/core/memory.h / memory.cpp: Memory::Init / Synchronize / GeneralAlloc / VirtualAlloc
src/musa/core/platform.h / platform.cpp: GetMemoryByDevicePointer
src/hal/halMemory.h
```

#### 关键分支

```text
muMemAlloc_v2:
  validate dptr/bytesize
  -> TlsCtxTop
  -> build MemoryCreateInfo(memoryTypeGeneral)
  -> Context::CreateMemory
  -> Memory::Init
  -> HAL allocate/map

muMemFree_v2:
  Platform::GetMemoryByDevicePointer
  -> memory type / offset validation
  -> IMemory::Synchronize
  -> non-virtual: Context::DestroyMemory
  -> virtual with pool: Stream::CmdMemFree(blocking=true) -> MemoryPool::DisableAccess -> CallbackCommand
```

#### 必须 ModelEvent

| 事件 | 类型 | 建议位置 | 说明 |
| --- | --- | --- | --- |
| `MEM_ALLOC_VALIDATE_SPAN` | span | `muapiMemAlloc_v2` | 参数检查 |
| `MEMORY_CREATE_INFO_POINT` | instant | createInfo 构造后 | flags/type/size 记录 |
| `CONTEXT_CREATE_MEMORY_SPAN` | span | `Context::CreateMemory` | context memory object 创建 |
| `MEMORY_INIT_SPAN` | span | `Memory::Init` | HAL allocate/map 前后 |
| `HAL_MEMORY_ALLOC_SPAN` | span | HAL memory allocate/map | 低层成本 |
| `MEMORY_POINTER_LOOKUP_SPAN` | span | `Platform::GetMemoryByDevicePointer` | free/query 成本 |
| `MEM_FREE_TYPE_DECISION` | instant | type/offset validation 后 | general/virtual/pool 分支 |
| `MEMORY_SYNCHRONIZE_SPAN` | span | `IMemory::Synchronize` | free 前等待 |
| `CONTEXT_DESTROY_MEMORY_SPAN` | span | `Context::DestroyMemory` | general memory release |

#### 必须 relation

```text
API -> Context
API -> Memory
Memory -> Context
Memory -> HAL Memory
Free API -> Memory(looked up by dptr)
Memory -> Stream/CallbackCommand(virtual-pool branch)
```

#### 成本拆分

```text
T_mem_alloc_sync =
  T_driver_wrapper
+ T_validation
+ T_context_lookup
+ T_memory_create_info
+ T_context_create_memory
+ T_memory_init
+ T_hal_alloc_or_map

T_mem_free_sync =
  T_driver_wrapper
+ T_pointer_lookup
+ T_memory_type_decision
+ T_memory_synchronize
+ T_destroy_or_pool_disable
```

### 4.11.4 Async memory / MemoryPool / VMM

#### 源码锚点

```text
src/driver/mu_memory.cpp: muapiMemAllocAsync / muapiMemAllocFromPoolAsync / muapiMemFreeAsync
src/driver/mu_mempool.cpp
src/driver/mu_vmm.cpp
src/musa/core/stream.h / stream.cpp: CmdMemAlloc / CmdMemFree / AsyncMemAlloc / AsyncMemFree
src/musa/core/memoryPool.h / memoryPool.cpp: CreateMemory / ModifyAccess / DisableAccess
src/musa/core/command/callbackCommand.h / callbackCommand.cpp
src/musa/core/command/pagingCommand.h / pagingCommand.cpp
```

#### 关键分支

```text
async alloc:
  TlsCtxTop -> Context::InfoStream -> build MemoryAllocParameter
  -> Stream::CmdMemAlloc(blocking=false)
  -> Stream::AsyncMemAlloc
  -> MemoryPool::CreateMemory
  -> MemoryPool::ModifyAccess
  -> ordered command/callback

async free:
  pointer lookup -> Context::InfoStream
  -> Stream::CmdMemFree(blocking=false)
  -> Stream::AsyncMemFree
  -> MemoryPool::DisableAccess
  -> CallbackCommand deferred destroy

VMM:
  reserve/create/map/unmap/setAccess/release
  -> Platform/Context/Device/Memory lookup
  -> HAL VMM operation
```

#### 必须 ModelEvent

| 事件 | 类型 | 建议位置 | 说明 |
| --- | --- | --- | --- |
| `MEM_POOL_LOOKUP_SPAN` | span | pool/default pool resolve | pool 选择 |
| `MEM_POOL_CREATE_SPAN` | span | `MemoryPool::Init` | pool lifecycle |
| `MEM_POOL_ALLOC_SPAN` | span | `MemoryPool::CreateMemory` | pool allocation |
| `MEM_POOL_HIT_MISS_POINT` | instant | pool allocate decision | hit/miss/grow |
| `MEM_POOL_MODIFY_ACCESS_SPAN` | span | `MemoryPool::ModifyAccess` | access map |
| `MEM_POOL_DISABLE_ACCESS_SPAN` | span | `MemoryPool::DisableAccess` | free/unmap |
| `ASYNC_MEM_COMMAND_CREATE_SPAN` | span | `Stream::CmdMemAlloc/Free` | stream ordered command |
| `CALLBACK_COMMAND_CREATE_SPAN` | span | deferred destroy | delayed release |
| `VMM_RESERVE_SPAN` | span | address reserve/free | virtual address cost |
| `VMM_MAP_UNMAP_SPAN` | span | map/unmap | map cost |
| `VMM_SET_ACCESS_SPAN` | span | access update | permission cost |

#### 必须 relation

```text
API -> Context
API -> Stream
API -> MemoryPool
MemoryPool -> Memory
Memory -> VirtualAddress
Async free API -> CallbackCommand
VMM map API -> physical Memory
VMM map API -> virtual Memory/address range
```

### 4.11.5 Memcpy / Memset / Memory transfer / Atomic

#### 源码锚点

```text
src/driver/mu_memory.cpp: memcpy/memset API family
src/musa/core/context.h / context.cpp: GeneralMemcpy / GeneralMemset
src/musa/core/node/graphMemcpyNode.cpp: MemcpyLegalize / ParameterCheckPass / RangeCheckPass / OptimizationPass / CopyManagerSelectPass
src/musa/core/node/graphMemsetNode.cpp: MemsetLegalize / UpdateParams
src/musa/core/stream.h / stream.cpp: CmdCopyMemory / CmdMemset / CmdMemoryTransfer / CmdMemoryAtomic
src/musa/core/command/memcpyCommand.h / memcpyCommand.cpp
src/musa/core/command/memsetCommand.h / memsetCommand.cpp
src/musa/core/command/memoryTransferCommand.h / memoryTransferCommand.cpp
src/musa/core/command/memoryAtomicCommand.h / memoryAtomicCommand.cpp
src/driver/mupti/tracepoints.h: EnterMemcpy / EnterMemset
```

#### 关键分支

```text
memcpy sync:
  normalize descriptor -> GeneralMemcpy(wait=true)
  -> GraphMemcpyNode legalize/range/copy-manager selection
  -> Stream::CmdCopyMemory
  -> SyncMemcpyCommand/AsyncMemcpyCommand
  -> wait complete

memcpy async:
  normalize descriptor -> GeneralMemcpy(wait=false)
  -> Stream::CmdCopyMemory
  -> AsyncMemcpyCommand queued

memset:
  build MemsetParameter -> GeneralMemset
  -> GraphMemsetNode legalize
  -> Stream::CmdMemset
  -> MemsetCommand

memory transfer / atomic:
  node parameter validate
  -> GraphMemory*Node or immediate Stream::CmdMemory*
  -> Memory*Command
```

#### 必须 ModelEvent

| 事件 | 类型 | 建议位置 | 说明 |
| --- | --- | --- | --- |
| `COPY_DESCRIPTOR_NORMALIZE_SPAN` | span | Driver API descriptor conversion | 1D/2D/3D/HtoD/DtoH 归一化 |
| `COPY_LEGALIZE_SPAN` | span | `GraphMemcpyNode::MemcpyLegalize` | 参数合法化 |
| `COPY_RANGE_CHECK_SPAN` | span | `RangeCheckPass` | memory range 检查 |
| `COPY_OPTIMIZATION_DECISION` | instant | `OptimizationPass` | path/engine 决策 |
| `COPY_MANAGER_SELECT_SPAN` | span | `CopyManagerSelectPass` | CE/TDM/copy manager |
| `MEMSET_LEGALIZE_SPAN` | span | `GraphMemsetNode` | memset 参数合法化 |
| `MEMCPY_COMMAND_CREATE_SPAN` | span | `Stream::CmdCopyMemory` | copy command 创建 |
| `MEMSET_COMMAND_CREATE_SPAN` | span | `Stream::CmdMemset` | memset command 创建 |
| `MEMORY_ATOMIC_COMMAND_CREATE_SPAN` | span | memory atomic command | atomic command 创建 |
| `SYNC_COPY_WAIT_SPAN` | span | sync copy wait | 同步 copy 等待 |

#### 必须 relation

```text
API -> Context
API -> Stream
API -> source Memory / host pointer
API -> destination Memory / host pointer
API -> CopyCommand/MemsetCommand
CopyCommand -> Submission
Submission -> MEMCPY/MEMSET activity
MemoryAtomicCommand -> MEMORY_ATOMIC activity
MemoryTransferCommand -> MEMORY_TRANSFER activity
```

### 4.11.6 Stream / Event / Synchronization

#### 源码锚点

```text
src/driver/mu_stream.cpp
src/driver/mu_event.cpp
src/musa/core/context.h / context.cpp: CreateStream / DestroyStream / CreateEvent / DestroyEvent / RecordEvent / WaitEvent
src/musa/core/stream.h / stream.cpp: Query / Synchronize / WaitFinish / CmdSetEvent / CmdWaitEvent
src/musa/core/event.h / event.cpp: Query / Synchronize
src/musa/core/command/recordCommand.h / recordCommand.cpp
src/musa/core/command/barrierCommand.h / barrierCommand.cpp
src/driver/mupti/tracepoints.h: CreateStream / DestroyStream / RegisterEventSynchronize / RegisterStreamWaitEvent / RegisterStreamSynchronize / RegisterContextSynchronize
```

#### 关键分支

```text
stream create/destroy:
  TlsCtxTop -> Context::CreateStream/DestroyStream -> Stream::Init/WaitFinish

stream sync/query:
  Context::InfoStream -> Stream::Query/Synchronize -> WaitFinish/timeline check

event record:
  validate Event -> Context::RecordEvent -> Stream::CmdSetEvent -> RecordCommand

event wait:
  validate Event -> Stream::CmdWaitEvent -> Barrier/Wait command

event sync/query:
  validate Event -> Event::Query/Synchronize -> HAL semaphore/timestamp wait
```

#### 必须 ModelEvent

| 事件 | 类型 | 建议位置 | 说明 |
| --- | --- | --- | --- |
| `STREAM_CREATE_SPAN` | span | `Context::CreateStream` | stream lifecycle |
| `STREAM_INIT_SPAN` | span | `Stream::Init` | HAL queue/semaphore init |
| `STREAM_DESTROY_SPAN` | span | `Context::DestroyStream` | release/wait |
| `STREAM_QUERY_SPAN` | span | `Stream::Query` | timeline query |
| `STREAM_WAIT_FINISH_SPAN` | span | `Stream::WaitFinish` | sync 主体 |
| `SYNC_WAIT_REASON_POINT` | instant | wait 前 | waiting for command/event/context |
| `EVENT_CREATE_SPAN` | span | `Context::CreateEvent` | event lifecycle |
| `EVENT_RECORD_COMMAND_CREATE_SPAN` | span | `CmdSetEvent` | record command |
| `EVENT_WAIT_COMMAND_CREATE_SPAN` | span | `CmdWaitEvent` | wait command |
| `EVENT_SYNCHRONIZE_SPAN` | span | `Event::Synchronize` | event wait |

#### 必须 relation

```text
API -> Context
API -> Stream
API -> Event
EventRecord API -> RecordCommand
StreamWaitEvent API -> Wait/BarrierCommand
StreamSynchronize API -> waited Command list
EventSynchronize API -> Event -> producer Command
```

### 4.11.7 Graph

#### 源码锚点

```text
src/driver/mu_graph.cpp
src/musa/core/graph.h / graph.cpp
src/musa/core/node/*.h / *.cpp
src/musa/core/graph/graph1/graphExec.h / graphExec.cpp
src/musa/core/graph/graph2/graphExec.h / graphExec.cpp
src/musa/core/command/graphCommand.h / graphCommand.cpp
src/driver/mupti/tracepoints.h: RegisterGraphTrace
```

#### 关键分支

```text
graph create/destroy:
  TlsCtxTop -> Context::CreateGraph / destroy graph

add node:
  validate graph/dependencies
  -> Context::Create*Node
  -> GraphNode::Init / UpdateParams / legalize
  -> Graph::AddGraphNode
  -> topology update

instantiate:
  validate graph
  -> Context::CreateGraphExec
  -> GraphExec::Init
  -> ResolveGraph / GraphFlatten
  -> PrepareAllSubmissions
  -> WriteSubmission / Write*Cmd

launch:
  validate GraphExec
  -> Context::InfoStream
  -> Stream::CmdLaunchGraph
  -> GraphCommand::Build / Submit / Execute
```

#### 必须 ModelEvent

| 事件 | 类型 | 建议位置 | 说明 |
| --- | --- | --- | --- |
| `GRAPH_CREATE_SPAN` | span | `Context::CreateGraph` | graph lifecycle |
| `GRAPH_NODE_CREATE_SPAN` | span | `Context::Create*Node` | node object |
| `GRAPH_NODE_LEGALIZE_SPAN` | span | node `Init/UpdateParams` | node 参数合法化 |
| `GRAPH_ADD_NODE_SPAN` | span | `Graph::AddGraphNode` | topology update |
| `GRAPH_DEPENDENCY_UPDATE_SPAN` | span | add/remove dependency | edge update |
| `GRAPH_INSTANTIATE_SPAN` | span | `Context::CreateGraphExec` | instantiate 总成本 |
| `GRAPH_RESOLVE_SPAN` | span | `GraphExec::ResolveGraph` | topology resolve |
| `GRAPH_FLATTEN_SPAN` | span | graph2 flatten or graph1 resolve | flatten 成本 |
| `GRAPH_PREPARE_SUBMISSION_SPAN` | span | `PrepareAllSubmissions` | submission grouping |
| `GRAPH_WRITE_SUBMISSION_SPAN` | span | `WriteSubmission` | cmd buffer 写入 |
| `GRAPH_LAUNCH_COMMAND_CREATE_SPAN` | span | `Stream::CmdLaunchGraph` | graph command |
| `GRAPH_EXECUTE_SPAN` | span | `GraphCommand::Execute` | launch execute |

#### 必须 relation

```text
API -> Graph
Graph -> GraphNode
GraphNode -> Function/Memory/Event/ChildGraph
GraphExec -> Graph
GraphExec -> GraphNode(flattened)
GraphExec -> Submission
GraphLaunch API -> Stream
GraphLaunch API -> GraphCommand
GraphCommand -> GraphExec
GraphCommand -> GRAPH_TRACE activity
```

### 4.11.8 GreenContext / resource isolation

#### 源码锚点

```text
src/driver/mu_greencontext.cpp
src/musa/core/greenContext.h / greenContext.cpp
src/musa/core/device.h / device.cpp: CreateGreenContext / GetDeviceResources
src/musa/core/context.h / context.cpp: inherited Context behavior
src/musa/core/stream.h / stream.cpp: CreateStream under GreenContext
```

#### 关键分支

```text
resource query/split/desc:
  Device resource -> split by SM count -> DevResourceDesc

green context create:
  Device::CreateGreenContext(desc, flags)
  -> GreenContext(Context) init with restricted resource

green stream create:
  GreenContext::CreateStream
  -> Stream::Init bound to GreenContext parent

normal work under GreenContext:
  same Context/Stream/Command path as normal context
  but context resource relation differs
```

#### 必须 ModelEvent

| 事件 | 类型 | 建议位置 | 说明 |
| --- | --- | --- | --- |
| `DEV_RESOURCE_QUERY_SPAN` | span | `Device::GetDeviceResources` | device resource |
| `DEV_RESOURCE_SPLIT_SPAN` | span | split API | SM partition |
| `DEV_RESOURCE_DESC_BUILD_SPAN` | span | desc generation | desc 构造 |
| `GREEN_CONTEXT_CREATE_SPAN` | span | `Device::CreateGreenContext` | context 创建 |
| `GREEN_CONTEXT_DESTROY_SPAN` | span | destroy API | context release |
| `GREEN_CONTEXT_STREAM_CREATE_SPAN` | span | `GreenContext::CreateStream` | stream 绑定 |
| `GREEN_CONTEXT_RESOURCE_RELATION_POINT` | instant | create 后 | context-resource relation |

#### 必须 relation

```text
GreenContext API -> Device
GreenContext -> DevResourceDesc
DevResourceDesc -> MUdevResource partitions
GreenContext -> Stream
Stream -> Command
Command -> Submission
Submission -> restricted resource context
```

### 4.11.9 Peer / external / texture / raytracing / tensor / profiler

#### 源码锚点

```text
src/driver/mu_peer.cpp
src/driver/mu_external.cpp
src/driver/mu_gfxinterop.cpp
src/driver/mu_oglinterop.cpp
src/driver/mu_texture.cpp
src/driver/mu_raytracing.cpp
src/driver/mu_tensor.cpp
src/driver/mu_profiler.cpp
src/driver/mu_notification.cpp
src/musa/core/command/accelStruct*Command.h / *.cpp
src/musa/core/command/dispatchRayCommand.h / dispatchRayCommand.cpp
src/musa/core/command/signalExternalSemaphoreCommand.h / waitExternalSemaphoreCommand.h
```

#### 关键分支

```text
peer:
  Platform/Device capability -> Context peer enable/disable -> HAL peer mapping

external memory/semaphore:
  import handle -> Memory/Semaphore wrapper
  wait/signal semaphore -> Stream command -> HAL semaphore

texture/surface:
  validate descriptor -> Memory/array/resource lookup -> HAL texture/surface resource

raytracing:
  validate accel/dispatch params -> Stream -> AccelStruct*Command/DispatchRayCommand -> HAL M3D

tensor:
  descriptor/resource setup; if GPU work exists, route through Stream/Command

profiler/log/notification:
  tool control plane; should be modeled separately from workload cost
```

#### 必须 ModelEvent

| 事件 | 类型 | 建议位置 | 说明 |
| --- | --- | --- | --- |
| `PEER_CAPABILITY_QUERY_SPAN` | span | peer query | device-to-device capability |
| `PEER_ACCESS_UPDATE_SPAN` | span | enable/disable peer | mapping update |
| `EXTERNAL_HANDLE_IMPORT_SPAN` | span | external import | fd/handle import |
| `EXTERNAL_SEMAPHORE_COMMAND_CREATE_SPAN` | span | wait/signal external sema | interop command |
| `TEXTURE_RESOURCE_CREATE_SPAN` | span | texture/surface create | descriptor/resource cost |
| `RAYTRACING_COMMAND_CREATE_SPAN` | span | accel/dispatch command | RT command create |
| `RAYTRACING_COMMAND_BUILD_SPAN` | span | accel/dispatch build | HAL M3D build |
| `TENSOR_DESCRIPTOR_SPAN` | span | tensor descriptor | tensor metadata path |
| `PROFILER_CONTROL_SPAN` | span | profiler/log control | 工具控制面 |

#### 必须 relation

```text
Peer API -> src Device / dst Device / Context
External memory API -> external handle -> Memory
External semaphore API -> external handle -> Semaphore -> Command
Texture API -> Memory/Array -> texture resource
Raytracing API -> Stream -> AccelStruct/DispatchRay Command -> Submission
Tensor API -> descriptor/resource -> optional Command
Profiler API -> tooling state / MUPTI state
```

## 4.12 source rule 优先级与落地顺序

为了避免 Top90 API 平铺导致实现分散，建议按共享主干优先：

```text
P0: submit boundary shared trunk
  Command::Build
  Command::Submit
  Command::SubmitToQueue
  Stream::QueueCommand
  HAL/M3D/OS submit

P1: M1 API
  muLaunchKernel
  muMemAlloc_v2
  muMemFree_v2
  muStreamSynchronize

P2: high-frequency extensions
  memcpy/memset sync/async
  stream/event create/record/wait/sync
  module/function lookup
  graph launch/instantiate

P3: memory pool / VMM / graph node families
  async alloc/free
  memory pool access
  VMM map/unmap/access
  graph add node / dependency / update

P4: specialized domains
  GreenContext/resource
  peer/external interop
  texture/surface
  raytracing
  tensor
  profiler/log/notification control plane
```

每个 source rule 文件最少包含以下字段：

```yaml
api: muXxx
family: <api-family>
runtime_api: musaXxx | null
entry:
  driver: <file>: <function>
  runtime: <file>: <function> | null
core_objects:
  - Context
  - Stream
  - Command
core_path:
  - step with source anchor
branches:
  - condition: sync | async | graph | pool | default stream | virtual memory
    path: ...
existing_mupti:
  callbacks: []
  activities: []
  tracepoints: []
model_events:
  - name: COMMAND_BUILD_SPAN
    type: span
    anchor: <file/function>
relations:
  - API -> Context
  - API -> Stream
  - API -> Command
  - Command -> Submission
cost_breakdown:
  - T_driver_wrapper
  - T_lookup
  - T_core_operation
  - T_command_build
  - T_submit
quality:
  required_ids:
    - correlation_id
    - context_id
    - stream_id
    - command_id
    - submission_id
  overhead_budget: trace-off ready flag only
```

## 5. Core -> HAL -> M3D -> OS submit boundary

所有异步 GPU 工作最终都应落到类似边界：

```text
Stream::QueueCommand
  |
  v
Command::CanMergeTo / Stream merging decision
  |
  v
Command::Build
  |
  +-- Command::GetHalCmdBuffer
  +-- Command::ResolveSubmitWait
  +-- Command::ResolveSubmitSignal
  +-- subclass writes dispatch/copy/memset/graph cmds
  |
  v
Command::Submit
  |
  v
Command::SubmitToQueue(Hal::IQueue, Hal::QueueSubmitInfo)
  |
  v
Hal::IQueue submit
  |
  v
M3D queue submit
  |
  v
OS submit / DRM / WKMD / WDDM
```

对应时序图：

```text
Stream SubmitThread      Command        Hal::ICmdBuffer        Hal::IQueue        M3D Queue        OS
       |                    |                  |                    |                |             |
       | take command       |                  |                    |                |             |
       |------------------->| Build            |                    |                |             |
       |                    |----------------->| write packets      |                |             |
       |                    | Resolve wait/signal semaphore       |                |             |
       |                    | Submit           |                    |                |             |
       |                    |------------------------------------>| submit info    |             |
       |                    |                                     |--------------->| queue submit |
       |                    |                                     |                |-----------> |
       | move inflight      |                  |                    |                |             |
```

ModelEvent 应在这条边界上稳定输出：

```text
COMMAND_BUILD_SPAN
COMMAND_MERGE_DECISION
COMMAND_SUBMIT_SPAN
HAL_QUEUE_SUBMIT_SPAN
M3D_QUEUE_SUBMIT_SPAN
OS_SUBMIT_SPAN
SUBMISSION_ID_ASSIGN_POINT
COMMAND_WAIT_SPAN
```

这些事件是把 API cost 从“API 慢了”拆到“build 慢、merge 阻塞、submit 慢、OS submit 慢、wait 慢”的关键。

## 6. ModelEvent/source rule 落地矩阵

| API 家族 | 关键 Core 对象 | 当前 MUPTI 基线 | ModelEvent 补充重点 |
| --- | --- | --- | --- |
| Device/Context | Platform / Device / Context | Driver API callback/activity、context tracepoints | init、device lookup、context create/destroy、context sync |
| Module/Function/Launch | Module / Function / Context / Stream / DispatchCommand | Runtime/Driver API、RegisterKernel、AssignKernelToKick、kernel activity | module load、function lookup、kernel state、command build/submit |
| Sync memory | Context / Memory / Platform | Runtime/Driver API | MemoryCreateInfo、Memory::Init、HAL alloc、pointer lookup、destroy |
| Async memory/pool | Stream / MemoryPool / Memory / CallbackCommand | Runtime/Driver API | pool hit/grow、ModifyAccess/DisableAccess、deferred destroy |
| Memcpy/Memset | Context / GraphMemcpyNode / GraphMemsetNode / Stream / Command | memcpy/memset activity、EnterMemcpy/EnterMemset | descriptor normalize、legalize、copy manager select、command submit |
| Stream/Event/Sync | Context / Stream / Event / RecordCommand / BarrierCommand | stream/event/sync tracepoints | stream wait reason、event command、command wait |
| Graph | Graph / GraphNode / GraphExec / GraphCommand | GRAPH_TRACE activity | node legalize、instantiate、resolve、submission write、launch execute |
| GreenContext | Device / GreenContext / Stream | Driver API callback/activity | resource split/desc、green context create、resource relation |
| Submit boundary | Stream / Command / HAL queue / M3D queue | AssignSubmissionToCorrelation | command build/merge/submit、HAL/M3D/OS submit |

## 7. 建议的 source rule 模板

每个 API source rule 不应只写 API 名称，而应至少覆盖：

```yaml
api: muXxx
runtime_api: musaXxx
family: launch | memory | stream | event | graph | context | green_context | ...
entry:
  runtime: MUSA-Runtime/src/...
  driver: linux-ddk/musa/src/driver/...
core_path:
  - driver validation and InitPlatform
  - TlsCtxTop / Platform / object lookup
  - Context/Stream/Memory/Graph core object operation
  - Command creation/build/submit if any
  - HAL/M3D/OS boundary if any
existing_mupti:
  - callback/activity/tracepoint names
model_events:
  - required ModelEvent names
relations:
  - API -> context/stream/memory/command/submission/activity
cost_breakdown:
  - T_runtime_wrapper
  - T_driver_wrapper
  - T_lookup
  - T_core_operation
  - T_command_build
  - T_submit
  - T_wait
notes:
  - branch conditions and unsupported paths
```

## 8. 对 OKR 实施的直接建议

1. M1 继续以 `muLaunchKernel`、`muMemAlloc_v2`、`muMemFree_v2`、`muStreamSynchronize` 做最小闭环，但事件命名要与本文对象边界一致。
2. M2 的 Top90 不要按单个 API 平铺实现，应先按本文九类 API family 写 source rule，再扩展具体 API。
3. submit boundary 是跨 API 共享的主干，一旦 `Command::Build/Submit/SubmitToQueue` 和 HAL/M3D submit 事件打通，launch/memcpy/memset/graph/atomic/transfer 都能复用。
4. memory 家族必须区分同步 general allocation 与 async/pool allocation，避免把 `musaMalloc` 误解释成 memory pool hit/miss。
5. graph 家族必须拆成 construction、node legalize、instantiate、submission write、launch execute，否则 graph cost 会被错误归入 `muGraphLaunch`。
6. GreenContext 复用 Context/Stream/Command 主体，差异点是 Device resource 和 GreenContext resource relation，应单独输出 relation。
7. 每个 ModelEvent 都要能回连到 correlation id、context id、stream id、command id、submission id；否则 cost breakdown 能看到阶段，但不能解释 API 与 activity 的因果关系。

