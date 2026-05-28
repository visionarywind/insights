# muLaunchKernel — 内核启动完整流程分析

> `muLaunchKernel` 对应 CUDA `cudaLaunchKernel`，是 MUSA Runtime 的核心 API。本文档从用户调用入口到 KMD ioctl，对每一层进行逐行源码拆解。

---

## 1. 功能概述

在 GPU 上启动一个已编译的 kernel 函数，指定网格维度（grid）、线程块维度（block）、共享内存大小和执行流。

### 变体

| API | 文件 | 说明 |
|-----|------|------|
| `muapiLaunchKernel` | `mu_module.cpp:232` | 标准启动接口 |
| `muapiLaunchKernelEx` | `mu_module.cpp:274` | 扩展启动接口，支持 cluster dim、PDL 等属性 |
| `muapiLaunchCooperativeKernel` | `mu_module.cpp:343` | 协作式内核（直接委托给 `LaunchKernel`） |
| `muapiLaunchKernel_ptsz` | `mu_module.cpp:545` | 多线程安全版本，`hStream` 为 `musaStreamDefault` 时替换为 `musaStreamPerThread` |

---

## 2. 数据结构

### MUSA_KERNEL_NODE_PARAMS

```
定义位置: musa.h (MUSA API 公开头)
```

由 Driver 层构造，传递给 Core 层 `CreateKernelNode`：

```cpp
// mu_module.cpp:255-265
MUSA_KERNEL_NODE_PARAMS nodeParams = {};
nodeParams.func             = f;           // MUfunction (已加载的模块中的函数)
nodeParams.gridDimX         = gridDimX;
nodeParams.gridDimY         = gridDimY;
nodeParams.gridDimZ         = gridDimZ;
nodeParams.blockDimX        = blockDimX;
nodeParams.blockDimY        = blockDimY;
nodeParams.blockDimZ        = blockDimZ;
nodeParams.sharedMemBytes   = sharedMemBytes;
nodeParams.kernelParams     = kernelParams; // 指向 kernel 参数数组的指针
nodeParams.extra            = extra;        // 额外参数（通常为 nullptr）
```

### MUlaunchConfig（仅 Ex 版本）

```cpp
// muapi.h 中定义
struct MUlaunchConfig {
    unsigned int gridDimX, gridDimY, gridDimZ;
    unsigned int blockDimX, blockDimY, blockDimZ;
    unsigned int sharedMemBytes;
    unsigned int numAttrs;
    MUlaunchAttribute *attrs;
    MUstream hStream;
};
```

支持的属性（`MUlaunchAttributeID`）：
- `MU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION` — 集群维度
- `MU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` — 启用 PDL
- 其余属性（AccessPolicy、Cooperative、SynchronizationPolicy 等）均返回 `NOT_SUPPORTED`

---

## 3. 完整调用链（六层）

```
═══════════════════════════════════════════════════════════════════════
  用户层 (User Code)
═══════════════════════════════════════════════════════════════════════
  muLaunchKernel(f, gridDim, blockDim, sharedMem, stream, params, extra)
       │
       ▼
═══════════════════════════════════════════════════════════════════════
  Wrapper 层 (mu_wrappers_generated.cpp)
═══════════════════════════════════════════════════════════════════════
  muapiLaunchKernel(f, gridDimX/Y/Z, blockDimX/Y/Z, sharedMemBytes,
                    hStream, kernelParams, extra)
       │  (MUPTI 插桩: ApiTrace trace + 回调通知)
       ▼
═══════════════════════════════════════════════════════════════════════
  Driver 层 (mu_module.cpp:232-272)
═══════════════════════════════════════════════════════════════════════
  muapiLaunchKernel()
       │
       ├─ InitPlatform()                      // 懒初始化平台
       ├─ ICast<IFunction>(f) → pFunction     // 校验 MUfunction 有效性
       │
       ├─ KernelReplayHandler(pFunction, hStream, kernelParams, extra)
       │     └─ TODO: 内核重放功能（当前为空占位）
       │
       ├─ 构造 MUSA_KERNEL_NODE_PARAMS nodeParams
       │     .func = f
       │     .gridDimX/Y/Z = ...
       │     .blockDimX/Y/Z = ...
       │     .sharedMemBytes = ...
       │     .kernelParams = ...
       │     .extra = ...
       │
       └─ Context::GeneralLaunchKernel(
               TlsCtxTop(),                    // 当前线程的 Context
               hStream,                        // MUstream
               nodeParams,                     // MUSA_KERNEL_NODE_PARAMS
               0, 0, 0,                        // clusterDimX/Y/Z (默认无集群)
               0,                              // enablePdl (默认禁用)
               Platform::Get().launchBlocking) // wait 标志
       │
       ▼
═══════════════════════════════════════════════════════════════════════
  Core 层: Context (context.cpp:633-674)
═══════════════════════════════════════════════════════════════════════
  Context::GeneralLaunchKernel(ctx, hStream, kernelParam,
                               clusterDimX/Y/Z, enablePdl, wait)
       │
       ├─ (1) 参数校验
       │     ├─ ctx==nullptr → INVALID_CONTEXT
       │     │
       │     └─ pStream = InfoStream(ctx, hStream)
       │         ├─ hStream==nullptr/musaStreamDefault → 默认流
       │         └─ pStream==nullptr → INVALID_HANDLE
       │
       ├─ (2) 校验 Stream ↔ Context 归属
       │     条件: pStream->GetParentCtx() != ctx
       │       且 pStream->GetCaptureStatus() != ACTIVE
       │     → INVALID_HANDLE
       │     ★ 被捕获的 Stream 可以属于不同的 Context
       │
       ├─ (3) 创建 GraphNode
       │     pContext->CreateKernelNode(kernelParam, &pGraphNode,
       │         clusterDimX, clusterDimY, clusterDimZ, enablePdl)
       │       │
       │       └─ new GraphKernelNode(this, nodeParams, clusterDim, enablePdl)
       │       │     └─ SetParams(&nodeParams)     // 复制参数
       │       │
       │       └─ graphNode->UpdateParams()        // ★ 参数校验 + 资源创建
       │             │
       │             ├─ (a) Function → maxBlockSize 校验
       │             ├─ (b) 校验 gridDim ≤ maxGridSize
       │             │        blockDim ≤ maxThreadsDim
       │             │        blockDim 乘积 ≤ maxThreadsPerBlock
       │             │        blockDim 乘积 ≤ maxBlockSize
       │             │   ★ 失败 → INVALID_VALUE / LAUNCH_OUT_OF_RESOURCES
       │             │
       │             ├─ (c) TCE SQMMA/CONV 约束:
       │             │   blockSize % (simdNumPerUsc × warpSize) == 0
       │             │   ★ 失败 → INVALID_VALUE
       │             │
       │             ├─ (d) Cluster Size 校验:
       │             │   gridDim % clusterDim == 0 (每维)
       │             │   clusterSize ≤ minActiveMpNumPerCore
       │             │   ★ 失败 → INVALID_CLUSTER_SIZE
       │             │
       │             ├─ (e) Function::CreateState(sharedMem, params, extra,
       │             │        blockSize, &pHalKernelState)
       │             │     在 HAL 层创建 KernelState，设置共享内存和参数
       │             │
       │             ├─ (f) 创建/重置 KernelResource
       │             │     m_KernelResource->m_pHalKernelState = pHalKernelState
       │             │
       │             └─ (g) RegisteredHostPointerCheckout()
       │                   检查 kernel 参数中的主机指针是否需要 checkout
       │
       ├─ (4) 分发模式选择
       │     if (captureStatus == ACTIVE)        // 图捕获模式
       │       → pStream->CaptureNode(pGraphNode)
       │       → 节点加入捕获图，不立即执行
       │     else if (captureStatus == INVALIDATED)
       │       → STREAM_CAPTURE_INVALIDATED 错误
       │     else                                // ★ 正常执行路径
       │       → pStream->CmdLaunchKernel(pGraphNode, wait)
       │
       └─ 返回 status
       │
       ▼
═══════════════════════════════════════════════════════════════════════
  Core 层: Stream (stream.cpp:1415-1429)
═══════════════════════════════════════════════════════════════════════
  Stream::CmdLaunchKernel(pGraphNode, blocking)
       │
       ├─ 创建 DispatchCommand
       │     std::make_shared<DispatchCommand>(
       │         this,                           // Stream*
       │         ICast<GraphNode>(pGraphNode),   // GraphNode*
       │         PfmCheckEnable(),               // 性能监控
       │         PfmGetConfig())                 // 性能配置
       │
       └─ ResolveDependencyAndQueueCommand(command, this, blocking)
       │
       ▼
═══════════════════════════════════════════════════════════════════════
  Core 层: Context (context.cpp:1859-1921)
═══════════════════════════════════════════════════════════════════════
  Context::ResolveDependencyAndQueueCommand(command, pStream, blocking)
       │
       ├─ (1) PFM 串行锁（性能监控互斥）
       │     if (pfmEnabled) Lock(pfmSerialLock)
       │
       ├─ (2) 依赖关系建立
       │     if (pStream == m_DefaultStream)
       │       → 依赖所有非默认、非 Barrier 的阻塞流
       │     else if (pStream == m_BarrierStream)
       │       → 依赖所有非 Barrier 流
       │     else if (!pStream->IsNonBlocking())
       │       → 依赖默认流（阻塞流语义）
       │     → command->RecordDependency(otherStream->LastCommand())
       │
       ├─ (3) Barrier 依赖
       │     if (barrierCommand 存在 && status > completed)
       │       → command->RecordDependency(barrierCommand)
       │
       ├─ (4) 当前流依赖
       │     for (auto& dep : pStream->GetCurrentDependencies())
       │       → command->RecordDependency(dep)
       │     pStream->SetCurrentDependencies({})  // 清空
       │
       ├─ (5) 入队
       │     pStream->QueueCommand(std::move(command))
       │       │
       │       ├─ while (m_AsyncCount >= asyncCapacity) yield()
       │       │   // 背压: 等待异步槽位
       │       │
       │       ├─ m_AsyncCount++
       │       │
       │       ├─ command->SetPrevCommand(m_LastCommand)
       │       │   // 同一流的相邻命令链
       │       ├─ m_LastCommand = command
       │       │
       │       ├─ command->SetStatus(Status::queued)
       │       │
       │       ├─ m_CommandList.push_back(command)
       │       │   // 加入命令列表（等待 AsyncSubmit 线程消费）
       │       │
       │       └─ m_SubmitCv.notify_one()
       │             // 唤醒 AsyncSubmit 线程
       │
       ├─ (6) 阻塞等待（可选）
       │     if (blocking || pfmEnabled)
       │       → command->Wait()
       │         → Schedule([]{ return status==completed || error; })
       │         → 阻塞调用线程直到命令完成
       │
       └─ 解锁 PFM 串行锁
       │
       ▼
═══════════════════════════════════════════════════════════════════════
  AsyncSubmit 线程 (stream.cpp:1108-1278)
═══════════════════════════════════════════════════════════════════════
  Stream::AsyncSubmit()  // 独立线程，循环运行
       │
       ├─ 等待条件: !m_CommandList.empty() || stopMerging() || stopToken
       │  (通过 m_SubmitCv 条件变量唤醒)
       │
       ├─ stepMergingList = [核心提交逻辑]
       │     │
       │     ├─ (A) 引擎调度
       │     │     EngineSubmissionSchedule(engine)
       │     │
       │     ├─ (B) 合并策略判断
       │     │     commandType = MergingList.front()->GetType()
       │     │     if Dispatch 或 Memset 或 AsyncMemcpy (非Graph/Memcpy):
       │     │       → 支持合并 (CanMergeTo)
       │     │     else:
       │     │       → 不合并
       │     │
       │     ├─ (C) 分配 submissionId
       │     │     if (userQueueSubmissionId==0 || PfmEnabled)
       │     │       submissionId = UpdateGlobalSubId()
       │     │     else
       │     │       submissionId = userQueueSubmissionId
       │     │
       │     ├─ (D) 遍历 MergingList 设置 correlation
       │     │     for each merged:
       │     │       merged->SetSubId(submissionId)
       │     │       if Dispatch:
       │     │         MUpti::MarkKernelSubmitted(corId)
       │     │         MUpti::AssignKernelToKick(uniqueId, subId)
       │     │
       │     ├─ (E) 提交主命令
       │     │     status = m_MergingList.front()->Submit()
       │     │       │
       │     │       └─ DispatchCommand::Submit()    // ★ 详见第 4 节
       │     │
       │     ├─ (F) 状态传播
       │     │     for each merged:
       │     │       merged->SetLastError(status)
       │     │       merged->SetStatus(Status::submitted)
       │     │
       │     └─ (G) 移入飞行列表
       │           m_InflightList.splice(end, MergingList)
       │           m_WaitCv.notify_one()  // 唤醒 AsyncWait
       │
       │
       ├─ buildCommand = [构建逻辑]          // ★ 详见第 5 节
       │     │
       │     ├─ (1) command->FilterDependency()
       │     │     移除已完成的执行依赖
       │     │
       │     ├─ (2) 判断是否可合并
       │     │     keepMerging = command->CanMergeTo(m_MergingList)
       │     │     DispatchCommand::CanMergeTo:
       │     │       spill==0 || usePerStreamSpill → 可合并
       │     │       否则 → 不可合并
       │     │
       │     ├─ (3) 信号量值分配
       │     │     if (keepMerging && primary存在)
       │     │       → 共享主命令的 signalSemaphoreValue
       │     │     else
       │     │       → ++m_TimelineValue  // 新时间戳
       │     │
       │     ├─ (4) 如不可合并 → submitMergingList() 刷新
       │     │
       │     ├─ (5) command->Build(m_MergingList)
       │     │     ★ 详见第 6 节
       │     │
       │     ├─ (6) command->SetStatus(Status::built)
       │     │     ★ built 状态意味着时间戳值已确定
       │     │
       │     └─ (7) m_MergingList.push_back(command)
       │
       └─ 主循环: 消费 m_CommandList → buildCommand → 可能 submitMergingList
       │
       ▼
═══════════════════════════════════════════════════════════════════════
  AsyncWait 线程 (stream.cpp:1280-1398)
═══════════════════════════════════════════════════════════════════════
  Stream::AsyncWait()  // 独立线程，循环运行
       │
       ├─ 等待条件: !m_InflightList.empty() || stopToken
       │  (通过 m_WaitCv 条件变量唤醒)
       │
       ├─ (1) 移动命令到等待列表
       │     m_WaitingList.splice(end, m_InflightList,
       │       begin, find_if(primary_merge_level))
       │
       ├─ (2) 等待主命令完成
       │     frontCommand = m_WaitingList.front()
       │     if (frontCommand->GetPreferredSemaphoreType() == Hardware)
       │       → m_HardwareSemaphore->Wait(signalValue)
       │         → 超时: executionTimeoutMs
       │         → 成功后 → m_TimelineSemaphore->Signal(signalValue)
       │     else (Timeline)
       │       → m_TimelineSemaphore->Wait(signalValue)
       │
       ├─ (3) 引擎提交反馈
       │     EngineSubmissionFeedback(engine)
       │
       ├─ (4) 引擎错误检查
       │     GetEngineLastError(engine, &commandStatus)
       │     → QueryRobustInfo(exceptionType)
       │     → GetLastError(exceptionType)
       │
       ├─ (5) 遍历等待列表中的所有命令
       │     for each command in m_WaitingList:
       │       │
       │       ├─ if (Dispatch/DispatchRay/Graph):
       │       │     updatePrintfStatus(command)
       │       │     // 检测内核断言 (assert)
       │       │
       │       ├─ if (commandStatus != SUCCESS):
       │       │     command->SetLastError(status)
       │       │     command->ErrorHandler(dumpType)
       │       │     // 生成 core dump (.mudmp 文件)
       │       │     // 查询异常PC → 反汇编 → wave信息
       │       │
       │       ├─ command->Postprocess()
       │       │     // 释放 HAL command buffer
       │       │     m_ParentStream->ReleaseCmdBuffer(engine, cmdBuffer)
       │       │
       │       ├─ command->SetStatus(completed / error)
       │       │
       │       └─ m_AsyncCount--
       │
       └─ (6) 异步释放资源
             Platform::Get().GetExecutor().QueueTask(
               [postprocessList]{
                 for each: command->ReleaseResources()
                 // 释放时间戳内存、perf experiment 等
               })
       │
       ▼
═══════════════════════════════════════════════════════════════════════
  HAL 层 (dispatchCommand.cpp)
═══════════════════════════════════════════════════════════════════════
  DispatchCommand::Build(mergingList)  // (stream.cpp 第6节)
       │
       ├─ (1) 确定合并级别
       │     if (mergingList.empty())
       │       → MergeLevel::primary    // 首个命令，拥有时间戳
       │     else
       │       → MergeLevel::secondary  // 被合并的命令，共享时间戳
       │
       ├─ (2) PrintManager 初始化
       │     PrintManager::CheckEnabled(kernel)
       │     → 启用 printf 则注册内核到 PrintManager
       │
       ├─ (3) InitPfm() — 性能监控初始化
       │
       ├─ (4) 获取 HAL Command Buffer
       │     if (primary)
       │       → GetHalCmdBuffer(true)   // 开始新 buffer
       │     else (secondary)
       │       → pPrimaryCommand->GetHalCmdBuffer(false) // 共享 buffer
       │
       ├─ (5) 如果是主命令 → CmdBufferBegin(beginInfo)
       │     beginInfo.prefetchShaders = enableCdmPrefetch
       │     beginInfo.prefetchShaderConsts = enableCdmPrefetch
       │     beginInfo.pcSamplingSettings = PCPSamplingConfig
       │
       ├─ (6) 如果是次命令且不满足 PDL 条件:
       │     → pHalCmdBuffer->CmdBarrier()
       │     // PDL (Programmatic Stream Serialization) 条件:
       │     //   EnablePdl==true && IP major>=4 && 依赖数==1
       │
       ├─ (7) BeginPfm(pHalCmdBuffer)
       ├─ (8) BeginPcSampling(pHalCmdBuffer)
       │
       ├─ (9) 分配时间戳内存 (如果启用 MUPTI)
       │     AllocateTimestampMem()
       │     CmdWriteTimestamp(TopOfPipe, timestampGpuVa[beginTime])
       │
       ├─ (10) UpdateImplicitResources()  // ★ 更新隐式资源
       │     │
       │     ├─ 为 kernel 参数中的 implicit 资源分配内存:
       │     │   - SPILL (溢出内存)
       │     │   - PRINT (printf 缓冲区)
       │     │   - CLUSTER_BARRIER (集群屏障)
       │     │   - CLUSTER_DATA (集群数据)
       │     │   - CLUSTER_FLAG (集群标志)
       │     │   - PROGRAM_TENSOR (程序张量)
       │     │   - GLOBAL_MEM_CONST_DATA (全局常量数据)
       │     │
       │     ├─ 每种类型的分配:
       │     │   AllocateInternalMemory(type, pHalCmdBuffer, size)
       │     │     → 优先使用 CmdAllocateEmbeddedData (嵌入式数据)
       │     │     → 回退到 Device::AllocateInternalMem (通用堆)
       │     │
       │     └─ 将资源绑定写入 const buffer
       │
       ├─ (11) Command::Build(mergingList)  // ★ 核心基类 Build
       │     │
       │     ├─ 等待前序命令:
       │     │   if (m_PrevCommand && (major<4 || !userQueue || engine不同))
       │     │     → buildSemaphoreDependency(m_PrevCommand, Status::built)
       │     │     // 等待前序命令完成:
       │     │     //   while (dependant->GetStatus() > waitStage) yield()
       │     │     //   根据信号量类型(Hardware/Timeline)记录等待
       │     │
       │     └─ 遍历所有执行依赖:
       │         for each m_ExecutionDependencies:
       │           → buildSemaphoreDependency(dep, Status::submitted)
       │           // 等待依赖的 submitted 状态
       │           // 记录 semaphore wait 信息
       │           // 根据类型选择:
       │           //   Timeline Timeline → TimelineSemaphore
       │           //   Hardware Timeline → HardwareSemaphore (trace capture)
       │           //   Timeline Hardware → 二者都等 (trace capture)
       │
       ├─ (12) 解析提交等待 (ResolveSubmitWait)
       │     if (PDL && IP major>=4 && deps==1)
       │       → 设置 pdlAddress/pdlValue (硬件等待)
       │     else
       │       → ResolveSubmitWait(CmdBuffer, WaitBetweenCmd, device)
       │         // 在 cmdbuffer 中插入等待指令:
       │         //   - Timeline sem → CmdWaitMemoryValue
       │         //   - Hardware sem → CmdWaitMemoryValue
       │         //   - internal sem → CmdWaitMemoryValue
       │
       ├─ (13) CmdBindKernel(bindKernel)   // 绑定内核对象
       ├─ (14) CmdBindKernelState(bindKernelState)  // 绑定内核状态
       ├─ (15) CmdSetLLCPersistcyWindow(accessPolicyWindow)  // LLC 策略
       ├─ (16) CmdSetMultiCoreMode(setMultiCoreMode)  // 多核模式
       │     if (green 模式)
       │       → AffinityMpxMask (亲和性掩码)
       │     else
       │       → ForceSingleCore or Default
       │
       ├─ (17) 如果使用 per-stream spill memory:
       │     CmdBindSpillMemoryRange(bindSpillMemoryRange)
       │
       └─ (18) CmdDispatch(dispatch)
             // 真正的 GPU 派发指令
             dispatch.workgroupSize = {blockDimX, blockDimY, blockDimZ}
             dispatch.workgroupCount = {gridDimX, gridDimY, gridDimZ}
             dispatch.blockClusterSize = {clusterX, clusterY, clusterZ}
             dispatch.constBufferGpuVa = ...
             dispatch.globalMemConstBufferCpuVa = ...
       │
       └─ CmdWriteTimestamp(BottomOfPipe, timestampGpuVa[endTime])
       │
       ▼
═══════════════════════════════════════════════════════════════════════
  DispatchCommand::Submit()  // (dispatchCommand.cpp:224-269)
═══════════════════════════════════════════════════════════════════════
       │
       ├─ (1) ResolveSubmitSignal(cmdBuffer, SignalBetweenCmd, device)
       │     // 在命令结束时发送信号:
       │     if (Timeline semaphore)
       │       → GetPeerSemaphore → 记录到 m_SubSignalSemaphoreInfos
       │     else (Hardware semaphore)
       │       → 记录 hardware semaphore
       │
       ├─ (2) EndPfm(pHalCmdBuffer)           // 性能监控结束
       ├─ (3) EndPcSampling(pHalCmdBuffer)     // PC 采样结束
       │
       ├─ (4) CmdBufferEnd()                   // 结束命令缓冲区录制
       │
       ├─ (5) MUgdb::KernelLoadingProcess(this)   // 调试器处理
       ├─ (6) MUasan::KernelLoadingProcess(this)  // ASAN 处理
       │
       ├─ (7) 创建 QueueSubmitInfo
       │     submitInfo.ppCmdBuffers = &pHalCmdBuffer
       │     submitInfo.cmdBufferCount = 1
       │     if (PDL && major>=4 && deps==1)
       │       submitInfo.freeSchedule = 1
       │
       ├─ (8) SubmitToQueue(halQueue(CDM), submitInfo)
       │     │
       │     └─ HAL 层 → m3d → KMD ioctl (mtgpuQueueSubmit)
       │         ★ 真正的 GPU 提交
       │
       ▼
═══════════════════════════════════════════════════════════════════════
  KMD 层 (ioctl)
═══════════════════════════════════════════════════════════════════════
  mtgpuQueueSubmit (Linux DRM / Windows WDDM)
       │
       ├─ 提交 cmdbuffer 到 GPU 硬件队列
       ├─ GPU 开始执行 CmdDispatch
       │     → 分发 gridDim × blockDim 个线程组
       │     → 每个线程组执行 kernel 函数
       ├─ 执行 CmdWriteTimestamp (BottomOfPipe)
       ├─ 信号 Timeline Semaphore / Hardware Semaphore
       │
       ▼
═══════════════════════════════════════════════════════════════════════
  完成通知 (AsyncWait 线程)
═══════════════════════════════════════════════════════════════════════
  AsyncWait 检测到 Semaphore 信号
       │
       ├─ 设置 command->SetStatus(Status::completed)
       ├─ command->Postprocess()  // 释放 cmdbuffer
       ├─ command->ReleaseResources()  // 释放时间戳内存等
       │     (异步排队到 executor 线程)
       └─ m_AsyncCount--
       │
       ▼
  如果 blocking=true:
       command->Wait() 返回 → 返回到用户层
```

---

## 4. DispatchCommand::Build 详细源码分析

**文件**: `command/dispatchCommand.cpp:67-222`

```
步骤 1-3: 合并级别判定 + PFM 初始化 + 获取 CmdBuffer (共 28 行)

    primary? → CmdBufferBegin(beginInfo)      // 开始录制
    secondary? → CmdBarrier (如不满足 PDL 条件) // 插入屏障
    → BeginPfm / BeginPcSampling              // 性能监控开始

步骤 4: 时间戳 (仅 MUPTI) (12 行)

    AllocateTimestampMem()
    CmdWriteTimestamp(TopOfPipe, beginTime)

步骤 5: UpdateImplicitResources (仅非 graph 来源) (8 行)

    → GraphKernelNode::UpdateImplicitResources()
      → 为 spill/print/cluster/programTensor/constData 分配内存
      → 将 kernel 参数写入 const buffer

步骤 6: Command::Build(mergingList) (15 行)

    → 等待 m_PrevCommand (如需要)
    → 等待所有 m_ExecutionDependencies
    → 记录 semaphore 等待信息

步骤 7: ResolveSubmitWait (PDL 路径或标准路径) (15 行)

    PDL 路径: 设置 pdlAddress/pdlValue
    标准路径: 在 cmdbuffer 中插入 CmdWaitMemoryValue

步骤 8: GPU 命令录制 (17 行)

    CmdBindKernel → 绑定 CUfunction 对象的 HAL 资源
    CmdBindKernelState → 绑定 KernelState (寄存器/共享内存配置)
    CmdSetLLCPersistcyWindow → LLC 缓存策略
    CmdSetMultiCoreMode → 多核/单核模式
    CmdBindSpillMemoryRange → 溢出内存绑定 (如需要)

步骤 9: CmdDispatch (13 行)

    真正的 GPU 派发: workgroupSize + workgroupCount + constBuffer

步骤 10: CmdWriteTimestamp (BottomOfPipe) + 返回 (7 行)
```

---

## 5. DispatchCommand::Submit 详细源码分析

**文件**: `command/dispatchCommand.cpp:224-269`

```
步骤 1: ResolveSubmitSignal(SignalBetweenCmd)
    → 获取 TimelineSemaphore 或 HardwareSemaphore
    → 记录到 m_SubSignalSemaphoreInfos

步骤 2: EndPfm + EndPcSampling
    → 性能监控数据结束采集

步骤 3: CmdBufferEnd()
    → 结束命令缓冲区录制

步骤 4: MUgdb::KernelLoadingProcess + MUasan::KernelLoadingProcess
    → 调试器/内存检测工具处理 (trace/debug 专用)

步骤 5: QueueSubmitInfo 构造
    → ppCmdBuffers / cmdBufferCount
    → freeSchedule (PDL 优化)

步骤 6: SubmitToQueue(halQueue(CDM), submitInfo)
    → HAL 层 → M3D → KMD ioctl (mtgpuQueueSubmit)
    → 将命令缓冲区提交到 GPU 计算引擎队列
```

---

## 6. 完整 ASCII 时序图

### 6.1 正常执行流程 (非阻塞)

```
线程A (调用线程)                  线程B (AsyncSubmit)              线程C (AsyncWait)                GPU
      │                               │                              │                               │
      │ muLaunchKernel()              │                              │                               │
      │  InitPlatform()               │                              │                               │
      │  ICast<IFunction>(f)          │                              │                               │
      │  KernelReplayHandler()        │                              │                               │
      │  nodeParams = {...}           │                              │                               │
      │  GeneralLaunchKernel()        │                              │                               │
      │    │                          │                              │                               │
      │    ├─ pStream = InfoStream()  │                              │                               │
      │    ├─ CreateKernelNode()      │                              │                               │
      │    │    ├─ GraphKernelNode()  │                              │                               │
      │    │    ├─ UpdateParams()     │                              │                               │
      │    │    │  ├─ 校验 grid/block │                               │                               │
      │    │    │  ├─ blockSize % warp == 0                         │                               │
      │    │    │  ├─ clusterDim 校验                                │                               │
      │    │    │  ├─ CreateState() [HAL层]                          │                               │
      │    │    │  └─ RegisteredHostPointerCheckout()                │                               │
      │    │    └─ 返回 pGraphNode     │                              │                               │
      │    │                          │                              │                               │
      │    ├─ (非捕获模式)             │                              │                               │
      │    │  CmdLaunchKernel()        │                              │                               │
      │    │    ├─ new DispatchCommand │                              │                               │
      │    │    │  ├─ 选择 perf engine │                              │                               │
      │    │    │  └─ 计算 spill 大小  │                              │                               │
      │    │    └─ ResolveDependency&  │                              │                               │
      │    │       QueueCommand()      │                              │                               │
      │    │         ├─ 依赖处理        │                              │                               │
      │    │         ├─ m_AsyncCount++  │                              │                               │
      │    │         ├─ m_LastCommand= │                              │                               │
      │    │         ├─ SetStatus(Q)   │                              │                               │
      │    │         ├─ push CommandList│                             │                               │
      │    │         └─ notify SubmitCv│                              │                               │
      │    │                          │                              │                               │
      │    │ (blocking? false → 返回) │                              │                               │
      │    ▼                          │                              │                               │
      │  返回 status                  │                              │                               │
      │                               │                              │                               │
      │                               │  AsyncSubmit() 循环:        │                               │
      │                               │    等待 m_SubmitCv           │                               │
      │                               │    ├─ 消费 m_CommandList     │                               │
      │                               │    ├─ buildCommand:          │                               │
      │                               │    │  ├─ FilterDependency()  │                               │
      │                               │    │  ├─ CanMergeTo()?        │                               │
      │                               │    │  ├─ ++m_TimelineValue   │                               │
      │                               │    │  ├─ Build():            │                               │
      │                               │    │  │  ├─ CmdBufferBegin   │                               │
      │                               │    │  │  ├─ BeginPfm         │                               │
      │                               │    │  │  ├─ AllocTimestamp   │                               │
      │                               │    │  │  ├─ WriteTimestamp   │◄──── 时间戳写入 (TopOfPipe)
      │                               │    │  │  ├─ UpdateImplicit   │                               │
      │                               │    │  │  ├─ Build(依赖)      │                               │
      │                               │    │  │  ├─ ResolveSubmitWait│                               │
      │                               │    │  │  ├─ CmdBindKernel    │                               │
      │                               │    │  │  ├─ CmdBindKernelSt  │                               │
      │                               │    │  │  ├─ CmdSetLLCWindow  │                               │
      │                               │    │  │  ├─ CmdSetMultiCore  │                               │
      │                               │    │  │  ├─ CmdBindSpillMem  │                               │
      │                               │    │  │  ├─ CmdDispatch  ────┼───► GPU 执行 kernel ◄──────────┤
      │                               │    │  │  ├─ WriteTimestamp   │◄──── 时间戳写入 (BottomOfPipe)    │
      │                               │    │  │  └─ 返回              │                               │
      │                               │    │  │                       │                               │
      │                               │    │  ├─ SetStatus(built)    │                               │
      │                               │    │  └─ push MergingList    │                               │
      │                               │    │                        │                               │
      │                               │    │  submitMergingList(): │                               │
      │                               │    │  ├─ EngineSchedul      │                               │
      │                               │    │  ├─ 设置 submissionId  │                               │
      │                               │    │  ├─ primary->Submit(): │                               │
      │                               │    │  │  ├─ ResolveSignal    │                               │
      │                               │    │  │  ├─ EndPfm           │                               │
      │                               │    │  │  ├─ EndPcSampling    │                               │
      │                               │    │  │  ├─ CmdBufferEnd()   │                               │
      │                               │    │  │  ├─ MUgdb/MUasan     │                               │
      │                               │    │  │  └─ SubmitToQueue ───┼───► HAL ioctl (mtgpuQueueSubmit)│
      │                               │    │  │                      │  ├─ 提交 cmdbuffer 到 GPU 队列  │
      │                               │    │  │                      │  └─ GPU 开始执行               │
      │                               │    │  │                      │                               │
      │                               │    │  ├─ 所有 merged:        │                               │
      │                               │    │  │  SetLastError(status)│                               │
      │                               │    │  │  SetStatus(submitted)│                               │
      │                               │    │  │                      │                               │
      │                               │    │  └─ InflightList ←─────│── (move from MergingList)       │
      │                               │    │                       │  └─ notify WaitCv               │
      │                               │    ▼                       ▼                               ▼
      │                               │                  AsyncWait 循环:              GPU 执行:
      │                               │                    等待 WaitCv                  CmdDispatch:
      │                               │                    移动 primary 到 WaitingList    分发线程组
      │                               │                    等待 Semaphore.wait ◄───────── 完成后信号
      │                               │                    │                               │
      │                               │                    ▼                               │
      │                               │              检查前序命令错误                    │
      │                               │              更新 printf 缓冲区                    │
      │                               │              Postprocess():                    │
      │                               │                ReleaseCmdBuffer()               │
      │                               │              SetStatus(completed)               │
      │                               │              m_AsyncCount--                     │
      │                               │              异步 ReleaseResources()            │
      │                               │                                              │
      │                               │  (GPU 完成所有工作)                           │
```

### 6.2 阻塞调用流程 (blocking=true)

```
线程A (调用线程)                         线程B (AsyncSubmit)                GPU
      │                                    │                                 │
      │ muLaunchKernel(..., blocking=true) │                                 │
      │  ... 同上直到 QueueCommand ...      │                                 │
      │  QueueCommand(command)             │                                 │
      │    ├─ m_LastCommand = command      │                                 │
      │    └─ notify SubmitCv             │                                 │
      │                                    │                                 │
      │  ResolveDependencyAndQueueCommand │                                 │
      │    └─ blocking → command->Wait()   │                                 │
      │         │                          │                                 │
      │         └─ Schedule([](){          │                                 │
      │           return status==completed │                                 │
      │                 || error; })      │                                 │
      │         │                          │                                 │
      │         └─ 更新 UserPools()        │                                 │
      │         │                          │                                 │
      │         └─ 阻塞等待 ──────────────────┼── AsyncSubmit 构建+提交        │
      │              ↑                     │  → SubmitToQueue ──→ GPU执行     │
      │              │                     │                                 │
      │              └── AsyncWait 完成 ◄───┼── GPU 完成信号                 │
      │                                    │                                 │
      │  Wait() 返回 → 返回到用户层        │                                 │
```

### 6.3 PDL (Programmatic Device Launch) 路径

```
当 enablePdl=true && IP major >= 4 && 依赖数 == 1:

  Build 阶段:
    跳过 ResolveSubmitWait 的 CmdWaitMemoryValue 写入
    → 改为在 submitInfo 中设置 pdlAddress/pdlValue
    → 硬件自动等待 semaphore (无需在 cmdbuffer 中插入等待指令)

  Submit 阶段:
    QueueSubmitInfo.freeSchedule = 1
    → 硬件在 semaphore 满足时自动调度执行

  优势: 减少 cmdbuffer 中的等待指令，降低 CPU 侧开销
```

---

## 7. 关键设计要点

### 7.1 双线程异步模型

| 线程 | 职责 | 关键数据结构 |
|------|------|-------------|
| **AsyncSubmit** | Build + Submit | `m_CommandList`, `m_MergingList`, `m_SubmitMtx`, `m_SubmitCv` |
| **AsyncWait** | WaitFinish + Postprocess | `m_InflightList`, `m_WaitingList`, `m_WaitMtx`, `m_WaitCv` |
| **调用线程** | 仅做轻量入队 | `m_AsyncCount` (背压控制) |

### 7.2 命令合并策略

```
MergingList 中的命令分两种角色:
  - Primary (MergeLevel::primary):   首个命令，拥有独立的时间戳
  - Secondary (MergeLevel::secondary): 被合并到 Primary，共享时间戳

合并条件 (DispatchCommand::CanMergeTo):
  - kernel 没有使用 spill memory (spilledPrivateMemorySize == 0)
  - 或使用 per-stream spill memory

不可合并的命令:
  - SyncMemcpyCommand (Command::Type::Memcpy)
  - GraphCommand (Command::Type::Graph)
  - 有 spill memory 且不使用 per-stream spill 的 Dispatch

合并中断条件:
  - MergingList 达到 32 个
  - 引擎就绪 (EngineSubmissionReady)
  - SyncMemcpy / Graph / Barrier 类型的命令强制刷新
```

### 7.3 信号量体系

```
两个层级的信号量:

1. Stream 级 (跨命令同步):
   - TimelineSemaphore: 连续递增值，用于 GPU-GPU 同步
   - HardwareSemaphore: 二值信号，用于 GPU-CPU 同步

2. Internal 级 (命令内同步):
   - InternalTimelineSemaphore: 命令间等待 (WaitWithinCmd)
   - InternalHardwareSemaphore: 命令间等待 (WaitWithinCmd)

选择逻辑 (GetPreferredSemaphoreType):
  if (EnableUserQueue(engine)) → Hardware   // 低延迟路径
  else → Timeline                           // 通用路径
```

### 7.4 性能监控 (PFM)

```
流程:
  1. ResolveDependencyAndQueueCommand 中获取 PFM 串行锁
  2. DispatchCommand 构造时决定是否启用 PFM
  3. Build 中调用 InitPfm() / BeginPfm()
  4. Submit 中调用 EndPfm()
  5. 错误时调用 ErrorHandler → 生成 core dump (.mudmp)
```

### 7.5 隐式资源管理

```
每个 kernel 自动管理的 8 类内部内存:
  SPILL (0)                — 寄存器溢出到内存
  PRINT (1)                — printf 输出缓冲区
  CLUSTER_BARRIER (2)      — 集群同步屏障
  CLUSTER_DATA (3)         — 集群共享数据
  CLUSTER_FLAG (4)         — 集群标志位
  PROGRAM_TENSOR (5)       — 程序张量 (AI/HP 模式)
  GLOBAL_MEM_CONST_DATA (6) — 全局常量数据
  CONST_BUFFER (7)         — 常量缓冲区 (GRAPH_BUFFER_COUNT = 8)
```

### 7.6 Graph Capture 模式下的特殊处理

```
  Capture 状态下的 LaunchKernel:
  → 调用 CaptureNode(pGraphNode) 而非 CmdLaunchKernel
  → 节点加入 m_CaptureGraph，不生成实际命令
  → 节点保存到 m_LastCapturedNodes，供后续 Record/WaitEvent 关联
  → CaptureEnd 时一次性构建整个图
  → GraphCommand 提交走独立路径 (UniversalManager::Submit)
```

---

## 8. 死代码 / 不可达路径标注

1. **`muapiLaunchCooperativeKernel`** (`mu_module.cpp:343-346`): 没有任何额外检查就委托给 `muapiLaunchKernel`，且传入 `extra=nullptr`。注释 `TODO: Add check for cooperative launch capability` 表明尚未实现协作式 kernel 的能力校验

2. **`muapiLaunchKernelEx` 中的属性处理** (`mu_module.cpp:293-318`): 以下属性 ID 均返回 `NOT_SUPPORTED`:
   - `MU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW`
   - `MU_LAUNCH_ATTRIBUTE_COOPERATIVE`
   - `MU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY`
   - `MU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE`
   - `MU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT`
   - `MU_LAUNCH_ATTRIBUTE_PRIORITY`
   - `MU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP`
   - `MU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN`

3. **`DispatchCommand::CanMergeTo`** 中的 `m_UsePerStreamSpillMemory` 路径: 当 `hwManagedSpillMemory == 1` 且 kernel 有 spill 时才会启用。早期架构 (不支持 `hwManagedSpillMemory`) 的 kernel 只要有 spill 就**永远不可合并**。

---

## 9. 错误粘性 (Sticky Error) 传播

```
stream.cpp:1360-1366:
  一旦 commandStatus != MUSA_SUCCESS:
    → m_LastError = 第一个错误 (保留，不覆盖)
    → 后续所有命令都继承该错误状态
    → 体现"粘性"语义: 一个错误污染整个流
```

---

## 10. 性能关键路径总结

```
非阻塞调用的延迟路径:
  muLaunchKernel()
  → CreateKernelNode()            // CPU: 参数校验 + CreateState
  → CmdLaunchKernel()             // CPU: 创建 DispatchCommand + 入队
  → AsyncSubmit 线程 Build()      // CPU: CmdBufferBegin → CmdDispatch → CmdBufferEnd
  → SubmitToQueue()               // CPU→KMD: ioctl (mtgpuQueueSubmit)
  → GPU 异步执行
  → AsyncWait 线程 WaitFinish()   // CPU: semaphore wait (或空闲)

关键性能数字:
  - CmdBufferBegin/End 之间的 CPU 操作: ~10-20 微秒
  - ioctl 提交延迟: ~1-5 微秒
  - GPU 执行: 取决于 kernel 复杂度
  - 合并 (merge) 可减少 CmdBufferBegin/End/ioctl 的次数
```