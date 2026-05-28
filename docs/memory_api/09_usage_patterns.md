# MUSA Memory API 使用模式分析

> 本文分析 MUSA-Runtime 项目中各模块如何使用内存 API，包括内部调用链、测试用例模式、以及关键使用场景。

---

## 1. 调用者总览

```
用户代码 / 测试用例
  │
  ├── muMemAlloc / muMemFree → Driver → Context::CreateMemory → GeneralAlloc
  ├── muMemcpyHtoD / muMemcpyAsync → Driver → GeneralMemcpy → Stream::CmdCopyMemory
  ├── muMemsetD32 / muMemsetD32Async → Driver → GeneralMemset → Stream::CmdMemset
  ├── muMemHostAlloc → Driver → Context::CreateMemory → PinnedHostAlloc
  └── muMemHostRegister → Driver → Context::CreateMemory → PinnedHostRegister
       │
       ▼
  Internal callers (Runtime内部)
  ├── Device::AllocateInternalMem → Memory::InitPrealloc  (内部信号/时间戳/Spill内存)
  ├── MemoryPool::CreateMemory   → Memory::InitFromPool   (池化虚拟内存分配)
  ├── Stream::AsyncMemAlloc      → Pool::CreateMemory + Memory::Init + Bind  (异步分配)
  ├── CopyManager (DMA/GPU/CPU)  → Hal::ICmdBuffer::CmdCopyMemory  (GPU命令构建)
  ├── GraphicsResource::Map      → Memory::InitPrealloc  (Vulkan/GL互操作)
  ├── Module::Load               → Memory::InitPrealloc  (全局变量/常量内存)
  └── unittest muCommandAccessors → muMemAlloc/muMemcpy/muMemset  (命令访问器测试)
```

## 2. 主要调用者及其调用模式

### 2.1 Driver 层 — mu_memory.cpp (主要外部入口)

`mu_memory.cpp` 是内存 API 的**最主要调用者**，所有用户态内存操作最终都在这里调用 `Context::CreateMemory` 或 `Stream::CmdXXX`。

| 调用者函数 | 调用的 Runtime API | 内存类型 | 用途 |
|-----------|-------------------|---------|------|
| `muapiMemAlloc_v2:288` | `pContext->CreateMemory` | `memoryTypeGeneral` | 通用设备内存分配 |
| `muapiMemAllocPitch_v2:472` | `pContext->CreateMemory` | `memoryTypePitchedGeneral` | 对齐分配 |
| `muapiMemHostAlloc:88` | `pContext->CreateMemory` | `memoryTypePinnedHost` | 主机页锁定内存 |
| `muapiMemHostRegister_v2:1719` | `pContext->CreateMemory` | `memoryTypeRegisteredPinnedHost` | 注册用户内存 |
| `muapiMemAllocManaged:524` | `pContext->CreateMemory` | `memoryTypeManaged` | 托管内存 |
| `muapiMemFree_v2:740` | `pContext->DestroyMemory` | — | 释放设备内存 |
| `muapiMemFreeHost:770` | `pContext->DestroyMemory` | — | 释放主机内存 |
| `muapiMemcpyHtoD_v2:813` | `Context::GeneralMemcpy` | — | H→D 拷贝 |
| `muapiMemsetD32_v2:1589` | `Context::GeneralMemset` | — | GPU Memset |
| `muapiMemAllocAsync:329` | `pStream->CmdMemAlloc` | — | 异步分配 |
| `muapiMemFreeAsync:425-430` | `pStream->CmdMemFree` / `DestroyMemory` | — | 异步释放 |

**典型模式** — 同步分配:
```cpp
// mu_memory.cpp:265
MUresult muapiMemAlloc_v2(MUdeviceptr *dptr, size_t bytesize) {
    // 1. InitPlatform + TlsCtxTop
    // 2. 构造 MemoryCreateInfo{type=General, size, flags}
    // 3. pContext->CreateMemory(&pMemory, createInfo)
    // 4. *dptr = pMemory->GetDevicePointer()
}
```

**典型模式** — 异步分配:
```cpp
// mu_memory.cpp:303
MUresult muapiMemAllocAsync(MUdeviceptr *dptr, size_t bytesize, MUstream hStream) {
    // 1. InitPlatform + TlsCtxTop + InfoStream
    // 2. 构造 MemoryAllocParameter{size}
    // 3. pStream->CmdMemAlloc(memAllocParam, false)
    // 4. *dptr = memAllocParam.virtAddress
}
```

### 2.2 VMM 层 — mu_vmm.cpp

`mu_vmm.cpp` 调用 `Context::CreateMemory` 创建虚拟和外部内存：

```cpp
// mu_vmm.cpp:137 — muapiMemCreate
pDevice->GetPrimaryContext()->CreateMemory(&pMemory, createInfo);
// 创建 physical allocation 后绑定到虚拟地址

// mu_vmm.cpp:381 — muapiMemImportFromShareableHandle
pContext->CreateMemory(&pMemory, createInfo);
// 导入外部 dma-buf 句柄
```

### 2.3 Memory Pool — mu_mempool.cpp / core/memoryPool.cpp

`MemoryPool::CreateMemory` 被 `Stream::AsyncMemAlloc` 调用：

```cpp
// memoryPool.cpp:374
MUresult MemoryPool::CreateMemory(IMemory** ppMemory, MUdeviceptr* pVirtAddr, size_t size) {
    // 1. 从 HAL pool sub-allocate
    // 2. pMemory->InitFromPool(this, size)  → memoryTypeVirtual
    // 3. TrackMemory 注册
}
```

**调用者**: `Stream::AsyncMemAlloc` (stream.cpp:537) 和 `GraphMemoryAllocNode::Init` (graphMemoryAllocNode.cpp:32)。

### 2.4 Stream 异步分配 — stream.cpp

`Stream::AsyncMemAlloc` 是异步内存分配的**核心实现**，组合多个子系统：

```cpp
// stream.cpp:519
MUresult Stream::AsyncMemAlloc(MemoryAllocParameter& param, bool blocking) {
    // Step 1: 从 MemoryPool 分配虚拟内存
    MemoryPool* pPool = ...;
    pPool->CreateMemory(&virt, &virtAddr, allocSize);

    // Step 2: 分配物理内存 (调用 Memory::Init → GeneralAlloc)
    auto spPhysical = std::make_shared<Memory>(m_ParentCtx);
    physical->Init(createInfo);   // ← 创建真正的 GPU 物理内存

    // Step 3: 绑定虚拟→物理 + 启用访问权限
    virt->Bind(spPhysical, allocSize, 0, 0);
    pPool->ModifyAccess(virt, physical, allocSize, blocking, this);
}
```

### 2.5 CopyManager 调度链

Memcpy 的**最终执行者**是 CopyManager，而不是直接走 Command::Build。这里是调度关系：

```
muMemcpyHtoD_v2
  → Context::GeneralMemcpy
    → GraphMemcpyNode::Init  ← 解析拷贝参数, 选择 CopyManager
    → Stream::CmdCopyMemory
      → AsyncMemcpyCommand / SyncMemcpyCommand
        → Command::Build
          (只做 Hal::CmdBuffer 的 begin/wait 等)
        → Command::Submit
          → ★ GetCpyMgr()->MemcpyH2D(this)  ★  ← 真正的拷贝执行

Copymanager 选择方式:
  GraphMemcpyNode 从 device 遍历支持的引擎
  Device::m_CpyMgrs[COPY_MANAGER_TDM / CE / DMA / CDM / CPU / SHADER]
```

CopyManager 的具体实现:

```cpp
// DmaCopyManager::MemcpyH2D (dmaCopyManager.cpp)
// 1. 注册 host 内存 (如果需要)
// 2. 填充 Hal::CopyMemoryParameter
// 3. cmdBuffer->CmdCopyMemory(copyMemory)  ← 写入 GPU 命令
```

```cpp
// GpuCopyManager2::MemcpyH2D (gpuCopyManager2.cpp)
// 1. 注册 host 内存
// 2. 用 shader kernel 执行拷贝 (compute-based copy)
// 3. cmdBuffer->CmdDispatch(...)  ← 发射计算 kernel
```

### 2.6 内部内存分配 — Device::AllocateInternalMem

**`AllocateInternalMem`** 是 Runtime 内部使用的专用分配路径，**不经过** `Context::CreateMemory`，而是直接通过 internal memory pool sub-allocate：

```cpp
// device.cpp:1105
MUresult Device::AllocateInternalMem(size_t size, size_t alignment,
                                      Memory** ppMemory,
                                      Hal::InternalMemoryPoolType poolType) {
    // 1. 获取内部 pool (lazy init)
    MemoryPool* pPool = GetInternalPool(poolType);

    // 2. 从 pool sub-allocate HAL memory
    pHalPool->FullAllocate(allocInfo, &offset, &pHalMemory);

    // 3. Memory 对象用 InitPrealloc (不经过 GeneralAlloc)
    Memory* pMemory = new Memory(m_PrimaryCtx.get());
    pMemory->InitPrealloc(pHalMemory, size, offset);  // ← 直接绑定 HAL memory
    pMemory->SetPool(pPool);
}
```

**调用场景** — 被 12 个不同模块调用:

| 调用位置 | 用途 |
|---------|------|
| `dispatchCommand.cpp:395` | Spill memory / 打印缓冲区 / 集群 barrier/data/flag |
| `dispatchRayCommand.cpp:363` | Ray tracing 内部缓冲区 |
| `graphKernelNode.cpp:127-593` | Constant buffer / spill / print / launch params |
| `graphExec.cpp:874` | Graph exec 时间戳内存 |
| `command.cpp:593,621` | Per-command profiling 时间戳 |
| `event.cpp:35,51` | Event 信号量 + 时间戳内存 |
| `stream.cpp:1723` | Stream spill base |
| `printManager.cpp:58-82` | Print FIFO buffer |
| `context.cpp:2332` | Conditional handle |

### 2.7 MemoryTracker 使用模式

`MemoryTracker` 实现「指针→对象」的区间映射，是所有内存操作的核心基础设施。

**注册 (TrackMemory)** — 分配后立即调用:
```
CreateMemory/DestroyMemory/MemoryPool::CreateMemory/GraphicsResource::Map/Module::Load
  └── 都调用 Platform::Get().GetMemoryTracker().TrackMemory(memory_sp)
```

**查找 (FindRange)** — 每 API 第一步:
```
muMemFree / muMemcpy / muMemset / muPointerGetAttribute / 任何需要 dptr→Memory* 的 API
  └── 都调用 Platform::Get().GetMemoryTracker().FindRange(ptr, &offset)
       → 返回 shared_ptr<IMemory>*
```

**清除 (UntrackMemory)** — 销毁时:
```
Context::DestroyMemory / MemoryPool::DestroyMemory / GraphicsResource::~ / Module:~
  └── 都调用 Platform::Get().GetMemoryTracker().UntrackMemory(pMemory)
```

### 2.8 模块加载中的内存使用

`Module::Load` 在加载 ELF kernel 时需要将全局变量和函数代码映射到 GPU 内存:

```cpp
// module.cpp:210
varMem->InitPrealloc(pHalMemory, size, offset);
// 预分配内存: 将 ELF 段中的数据直接绑定到已分配的 HAL memory
Platform::Get().GetMemoryTracker().TrackMemory(varMem_sp);
```

### 2.9 GraphicsResource 互操作

Vulkan/OpenGL 互操作中的内存使用:

```cpp
// graphicsResource.cpp:75
m_pMemory->InitPrealloc(m_HalMemory, registeredSize, 0);
// 使用已分配的 HAL memory (来自外部图形 API)
Platform::Get().GetMemoryTracker().TrackMemory(memory_sp);
```

## 3. 测试用例使用模式

### 3.1 muCommandAccessors.cpp — 命令访问器测试

这是**最主要的测试文件** (1711 行)，测试模式如下:

**基本模式 — memcpy 命令访问器:**
```cpp
// 1. 创建 context + stream
muCtxCreate(&ctx, 0, 0);
muStreamCreate(&stream, 0);

// 2. 分配设备内存
muMemAlloc(&src, 1024);
muMemAlloc(&dst, 1024);

// 3. 执行内存操作 (触发 MUPTi hook 捕获 Command 对象)
muMemcpyAsync(dst, src, 1024, stream);

// 4. 同步等待完成
muStreamSynchronize(stream);

// 5. 验证 Command 内部参数 (通过 MUPTi hook 捕获的指针)
void* captured = g_capturedMemcpyCommand.load();
EXPECT_NE(captured, nullptr);

// 6. 清理
muMemFree(src);
muMemFree(dst);
muStreamDestroy(stream);
muCtxDestroy(ctx);
```

**方向测试 — 验证拷贝方向枚举:**
```cpp
// HtoD
muMemcpyHtoDAsync(dst, hostData, 1024, stream);
// → g_capturedCopyKind == MUPTI_ACTIVITY_MEMCPY_KIND_HTOD

// DtoH
muMemcpyDtoHAsync(hostData, src, 1024, stream);
// → g_capturedCopyKind == MUPTI_ACTIVITY_MEMCPY_KIND_DTOH

// DtoD
muMemcpyAsync(dst, src, 1024, stream);
// → g_capturedCopyKind == MUPTI_ACTIVITY_MEMCPY_KIND_DTOD
```

**Memset 测试:**
```cpp
muMemAlloc(&dst, SIZE);
muMemsetD32Async(dst, 0xDEADBEEF, SIZE / sizeof(uint32_t), stream);
muStreamSynchronize(stream);
// 验证: g_capturedMemsetValue == 0xDEADBEEF
//       g_capturedMemsetSize == SIZE
//       g_capturedMemsetMemKind == MUPTI_ACTIVITY_MEMORY_KIND_DEVICE
muMemFree(dst);
```

**Memory atomic 测试:**
```cpp
muMemAlloc(&dst, sizeof(uint64_t));
muMemcpyHtoD(dst, &zero, sizeof(zero));
muMemAlloc(&data_srcs[0], elements * sizeof(int));
muMemsetD8_v2(data_srcs[0], 0, elements * sizeof(int));
muMemcpyHtoD(data_srcs[0], srcData.data(), elements * sizeof(int));

muMemoryAtomicBatchAsync(data_dsts.data(), data_srcs.data(),
    elements.data(), operation.data(), batchCount, stream);
```

**Graph 节点测试:**
```cpp
muMemAlloc(&src, 1024);
muMemAlloc(&dst, 1024);
muGraphAddMemcpyNode(&node, graph, nullptr, 0, &memcpyParams, ctx);
// 测试: GraphNodeGetId, GraphNodeGetType, GraphMemcpyNodeGetCopyKind, ...
muMemFree(src);
muMemFree(dst);

muMemAlloc(&dst, 1024);
muGraphAddMemsetNode(&node, graph, nullptr, 0, &memsetParams, ctx);
// 测试: GraphMemsetNodeGetValue, GetSize, GetMemKind
muMemFree(dst);
```

### 3.2 其他测试文件中的内存使用

```cpp
// muModuleTest.cpp — 模块加载 + kernel 执行的基本流程
muCHECK(muMemAlloc((MUdeviceptr*)&d_value, sizeof(int)));
muCHECK(muMemcpyHtoD_v2((MUdeviceptr)d_value, &h_init, sizeof(int)));
// ... launch kernel ...
muCHECK(muMemcpyDtoH_v2(&h_result, (MUdeviceptr)d_value, sizeof(int)));
muCHECK(muMemFree((MUdeviceptr)d_value));

// muInternalAccessors.cpp — 内部 API 验证
muCHECK(muMemAlloc(&src, 1024));
muCHECK(muMemAlloc(&dst, 1024));
// ... 执行并验证 CmdCopyMemory 的参数 ...
muCHECK(muMemFree(src));
muCHECK(muMemFree(dst));
```

## 4. 完整生命周期示例

一个典型的 MUSA 程序使用模式:

```cpp
// 1. 初始化
muInit(0);
muCtxCreate(&ctx, 0, 0);
muStreamCreate(&stream, 0);

// 2. 分配内存
muMemAlloc(&devPtr, 4096);                    // → Context::CreateMemory(General)
                                                  Memory::GeneralAlloc(subAllocatable)
                                                  MemMgr::Allocate → FullAllocate
                                                  TrackMemory, MapToPeers

void* hostMem;
muMemHostAlloc(&hostMem, 4096, 0);               // → Context::CreateMemory(PinnedHost)
                                                  Memory::PinnedHostAlloc(memoryAllocTypeHost)
                                                  GetHostPointer(lazy mmap)

// 3. 数据传输
muMemcpyHtoD(devPtr, hostMem, 4096);              // → GeneralMemcpy(memcpy_host_to_device)
                                                  CreateMemcpyNode
                                                  CmdCopyMemory → AsyncMemcpyCommand
                                                    Build→CmdCopyMemoryAdvanced
                                                    Submit→Hal::IQueue::Submit
                                                    WaitFinish

// 4. 计算
muLaunchKernel(...);                              // → DispatchCommand → CmdDispatch

// 5. 数据传回
muMemcpyDtoH(hostMem, devPtr, 4096);              // → GeneralMemcpy(memcpy_device_to_host)

// 6. 清理
muMemFree(devPtr);                                // → Syncronize + DestroyMemory
                                                  → MemMgr::Free(return to pool)
muMemFreeHost(hostMem);                           // → Syncronize + DestroyMemory
                                                  → Hal::DestroyMemory
muStreamDestroy(stream);
muCtxDestroy(ctx);
```

## 5. 关键架构总结

1. **Driver 层是唯一的外部接口**: 所有用户 API 最终都通过 Driver 层调用 Runtime Core
2. **Context::CreateMemory 是总入口**: 所有分配（General/PinnedHost/Registered/Managed/IPC/Virtual）都通过它
3. **Stream::CmdXXX 是异步路径入口**: CmdMemAlloc/CmdMemFree/CmdCopyMemory/CmdMemset 都先检查 capture 状态再派发
4. **MemoryTracker 全局区间映射**: 是「地址→对象」的反查基础，所有需要指针查找的 API 都依赖它
5. **Device::AllocateInternalMem 是内部专用路径**: 跳过 context 直接分配，用于信号量/时间戳/spill 等内部内存
6. **CopyManager 是拷贝的最终执行者**: AsyncMemcpyCommand 只负责命令生命周期管理，真正的拷贝指令由 CopyManager 填充到 HAL cmd buffer
