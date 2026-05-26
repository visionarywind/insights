# AsyncMemAlloc 执行流程：同步与异步分界点分析

> 追踪 `musaMemallocAsync` / `Stream::AsyncMemAlloc` 的完整执行流程，定位同步操作与异步操作之间的精确分界点。

---

## 1. 总览

`AsyncMemAlloc` 分为**四个步骤**，前三个同步，第四个异步：

```
AsyncMemAlloc(param, blocking)
  │
  ├─ Step 1: pool->CreateMemory()      ← [同步] VA 预留（从 pool 子分配）
  ├─ Step 2: physical->Init()          ← [同步] 创建物理内存对象
  ├─ Step 3: virt->Bind()              ← [同步] 建立 VA→PA 映射关系
  │
  └─ Step 4: pool->ModifyAccess()      ← 分界点所在
       │
       ├─ OpenPeerMemory()             ← [同步] 创建 dma-buf handle
       └─ stream->CmdPaging()           ← [异步] 提交 PagingCommand 到 stream
```

**用户立即拿到 VA 指针**（Step 1 完成后），物理页的 GPU 页表更新通过 stream 异步完成。

---

## 2. 逐层调用链

### 2.1 调用入口

**文件**：`musa/src/musa/core/stream.cpp:575`

```cpp
MUresult Stream::CmdMemAlloc(MemoryAllocParameter& param, bool blocking) {
    if (m_CaptureStatus == MU_STREAM_CAPTURE_STATUS_ACTIVE) {
        // 图捕获模式：只创建 node，等 instantiate 时才真正分配
        return CaptureMemAlloc(param);
    } else if (m_CaptureStatus == MU_STREAM_CAPTURE_STATUS_INVALIDATED) {
        return MUSA_ERROR_STREAM_CAPTURE_INVALIDATED;
    } else {
        // 非捕获模式：直接分配
        return AsyncMemAlloc(param, blocking);     // → 2.2
    }
}
```

### 2.2 Step 1 — 从 Pool 子分配虚拟地址

**文件**：`musa/src/musa/core/stream.cpp:524`

```cpp
MUresult Stream::AsyncMemAlloc(MemoryAllocParameter& param, bool blocking) {
    uint64_t allocSize = Util::AlignUp(param.size, physPageSize);

    // ===== Step 1: 虚拟地址分配（同步） =====
    MemoryPool* pPool = param.pool ? param.pool : GetParentCtx()->GetParentDevice()->GetMemoryPool();
    pPool->SetStream(this);
    status = pPool->CreateMemory(&virt, &virtAddr, allocSize);    // → 2.2.1

    param.virtAddress = virtAddr;  // ← 用户立即拿到指针！
```

**文件**：`musa/src/musa/core/memoryPool.cpp:374`

```cpp
MUresult MemoryPool::CreateMemory(Memory** ppMemory, MUdeviceptr* ptr, size_t size) {
    std::shared_ptr<IMemory> memory_sp = std::make_shared<Memory>(nullptr);
    auto pMemory = static_cast<Memory*>(memory_sp.get());

    status = pMemory->InitFromPool(this, size);  // 从 pool 子分配 VA 空间
    //        ↑ 内部调 HAL SubAllocate → 从预保留的 VA range 中切一块

    *ptr = pMemory->GetDevicePointer();  // 用户拿到的指针
    *ppMemory = pMemory;
    return status;
}
```

**本质**：这是虚拟地址空间预留（VA reservation），不是物理页分配。Pool 预保留了一大段 VA range，这里只是从中切一块。

### 2.3 Step 2 — 创建物理内存对象

**文件**：`musa/src/musa/core/stream.cpp:546`

```cpp
    // ===== Step 2: 创建物理内存对象（同步） =====
    std::shared_ptr<IMemory> spPhysical = std::make_shared<Memory>(m_ParentCtx);
    Memory* physical = IntrusiveCast<Memory>(spPhysical.get());

    Musa::MemoryCreateInfo createInfo{};
    createInfo.type   = Musa::memoryTypeGeneral;
    createInfo.general.size = allocSize;
    status = physical->Init(createInfo);     // 创建 Memory 对象（不一定立即分配物理页）
```

### 2.4 Step 3 — 绑定虚拟地址到物理内存

**文件**：`musa/src/musa/core/stream.cpp:560`

```cpp
    // ===== Step 3: VA→PA 绑定（同步） =====
    if (status == MUSA_SUCCESS) {
        virt->Bind(spPhysical, allocSize, 0, 0);  // 建立映射关系
```

### 2.5 Step 4 — 分界点：`ModifyAccess` → `CmdPaging`

**文件**：`musa/src/musa/core/stream.cpp:561`

```cpp
        // ===== Step 4: 修改访问权限 + GPU 页表更新 =====
        status = pPool->ModifyAccess(virt, physical, allocSize, blocking, this);
        //                                                   ↑
        //                        内部有两步：OpenPeerMemory (同步) + CmdPaging (异步)
```

**文件**：`musa/src/musa/core/memoryPool.cpp:295`

```cpp
MUresult MemoryPool::ModifyAccess(Memory* virt, Memory* physical, size_t size, 
                                   bool blocking, Stream* stream) {
    MemoryPagingParameter pagingParams{};

    // ===== 4a: OpenPeerMemory — 同步 =====
    for (const auto& accessPair : m_LocationAccessMap) {
        int deviceId = accessPair.first;
        MUmemAccess_flags flags = accessPair.second;

        if (deviceId != GetDevice()->GetId() && flags != MU_MEM_ACCESS_FLAGS_PROT_NONE) {
            // 为每个需要访问此内存的其他 GPU 创建 dma-buf handle
            Hal::PeerOpenInfo openInfo{};
            openInfo.openType = Hal::MemoryExternalHandleType::dmaBuf;
            status = HalToMuResult(
                physical->Hal()->OpenPeerMemory(&otherDevice->Hal(), openInfo));
            //        ↑ 同步：创建 dma-buf fd，让其他 GPU 可以访问此物理页
        }

        // 收集 paging 参数
        MemoryPaging paging{};
        paging.pDevice = accessDev;
        paging.pVirtMem = virt;
        paging.pPhysMem = physical;
        paging.size = size;
        paging.flags = flags;
        pagingParams.memoryPagings.emplace_back(paging);
    }

    // ===== 4b: CmdPaging — 异步 =====
    if (status == MUSA_SUCCESS && !pagingParams.memoryPagings.empty()) {
        status = stream->CmdPaging(pagingParams, blocking);  // → 2.5.1
    }

    return status;
}
```

### 2.6 `CmdPaging` — 将 paging 操作提交到 stream

**文件**：`musa/src/musa/core/stream.cpp:501`

```cpp
MUresult Stream::CmdPaging(const MemoryPagingParameter& param, bool blocking) {
    std::shared_ptr<Command> command = std::make_shared<PagingCommand>(this, param);
    //  ↑ PagingCommand 是 GPU 命令：更新 GPU 页表，设置访问权限

    return m_ParentCtx->ResolveDependencyAndQueueCommand(
        std::move(command), this, blocking);
    //  ↑ 如果 blocking=true：等待前面命令完成，然后执行 paging
    //    如果 blocking=false：只建立依赖关系，在 stream 上按序执行
}
```

**`PagingCommand` 最终会**：
1. 进入 `Command::Build()` → 编码 PM4 包到 GPU cmd buffer
2. 进入 `Stream::AsyncSubmit()` → `HalQueue::Submit()` → `kick()`
3. GPU 上执行：更新页表条目，设置内存访问权限

---

## 3. 分界点图示

```
        用户线程                                                    Stream 命令队列
        ────────                                                   ────────────────
musaMemAllocAsync()
  │
  ├─ CreateMemory() ─────────── [同步] VA 预留 ──→ param.virtAddress ←── 用户拿到指针
  ├─ physical->Init() ───────── [同步] 创建对象
  ├─ virt->Bind() ───────────── [同步] VA↔PA
  │
  ├─ ModifyAccess()
  │   ├─ OpenPeerMemory() ───── [同步] dma-buf
  │   │
  │   └─ stream->CmdPaging() ── ★★★ 异步分界点 ★★★
  │       │                             │
  │       │   ResolveDependencyAndQueue │
  │       │   Command()                 │
  │       │                             ├─ PrevCommand (wait)
  │       │                             ├─ PagingCommand   ← GPU 页表更新
  │       │                             ├─ KernelCommand   ← 用户的内核
  │       │                             ├─ MemcpyCommand   ← 用户的拷贝
  │       │                             └─ ...
  │       │
  │       └─ return ─────────────────── 立即返回，不等 paging 完成
  │
  └─ return MUSA_SUCCESS ──────── 调用者继续执行
```

---

## 4. 图模式下的对比

在 CUDA Graph 模式下，MemAlloc node 走的是**不同路径**：

**文件**：`musa/src/musa/core/graph/graph1/graphExec.cpp:1170`

```cpp
case MU_GRAPH_NODE_TYPE_MEM_ALLOC: {
    kick.nodeParams = static_cast<GraphMemoryAllocNode*>(curNode)->GetAllocParams();
    submissionType = MUSA_SUBMISSION_HOST_DEVICE;  // ← 标记为 Host 端执行
    break;
}
```

**图启动时**（`graphCommand.cpp:220`）：

```cpp
case MUSA_SUBMISSION_HOST_DEVICE:
    → universalMgr->CmdHostDevice(&submission, this, waitMode, signalMode);
      → ExecuteMemAlloc(kick, cmd, waitMode, signalMode);    // universalManager.cpp:199
        ├─ Wait semaphore
        ├─ physical->Init()                                   // 同步
        ├─ virt->Bind()                                       // 同步
        ├─ OpenPeerMemory (for each device)                   // 同步
        ├─ 更新页表                                            // 同步
        └─ Signal semaphore                                   // 通知下游可以开始了
```

图模式下 MemAlloc **全部同步**，因为在 GPU 端操作开始之前，内存必须完全就绪。图执行在 CPU 端是串行的：先做完所有 HOST_DEVICE 类型的 submission（包括 MemAlloc），再提交 DEVICE 类型的 submission 到 GPU。

---

## 5. 为什么叫 "Async"

| 特性 | 同步分配 (`muMemAlloc`) | 异步分配 (`musaMemAllocAsync`) |
|------|------------------------|-------------------------------|
| **VA 指针返回** | 函数内返回 | 函数内返回（相同） |
| **物理页映射** | 函数内同步完成 | 部分同步（dma-buf），页表更新异步 |
| **GPU 页表更新** | 阻塞等待 | **PagingCommand 提交到 stream 异步执行** |
| **与 stream 上操作的关系** | 隐式同步 | 自动依赖——paging 在 kernel/memcpy 之前执行 |
| **从 stream 的视角** | 没有 stream 概念 | **作为 stream 上一个命令节点**，与其他命令序列化 |
| **调用者阻塞** | 阻塞到完全可用 | VA 立即返回，物理就绪通过 stream 保证 |

"Async" 不是指后台线程分配——而是指 **物理页的 GPU 端映射通过 stream 的异步命令机制完成**，与 stream 上其他操作（kernel、memcpy）自动形成正确的依赖顺序。

---

## 6. 相关源文件索引

| 文件 | 函数 | 行号 | 作用 |
|------|------|------|------|
| `musa/core/stream.cpp` | `CmdMemAlloc` | 575 | 入口：图捕获 / 非捕获分发 |
| `musa/core/stream.cpp` | `AsyncMemAlloc` | 524 | **核心**：4 步分配流程 |
| `musa/core/stream.cpp` | `CmdPaging` | 501 | 异步提交 PagingCommand |
| `musa/core/memoryPool.cpp` | `CreateMemory` | 374 | 从 pool 子分配 VA |
| `musa/core/memoryPool.cpp` | `ModifyAccess` | 295 | **分界点**：OpenPeerMemory + CmdPaging |
| `musa/core/graph/graph1/graphExec.cpp` | (resolve) | 1170 | 图模式：MemAlloc → HOST_DEVICE |
| `musa/core/graph/graph1/universalManager.cpp` | `ExecuteMemAlloc` | 199 | 图模式：同步执行 MemAlloc |
| `musa/core/graph/graph1/universalManager.cpp` | `CmdHostDevice` | 349 | 图模式：HOST_DEVICE 分发 |
| `musa/core/command/graphCommand.cpp` | `ExecuteImpl` | 192 | 图启动：submission type 分发 |
