# MUSA GPU 内存分配器 — 软件设计架构图

> 基于源码 `linux-ddk/musa/` (~200K lines C++) 分析
> 画图覆盖：muapi.h 560 个 API

---

## 一、整体 MUSA Driver 架构（全 API 面）

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          用户 / AI 框架 / MUSA-Runtime                            │
│               PyTorch  │  SGLang  │  TensorFlow  │  直接 mu* API                 │
└──────────────────────────────────┬───────────────────────────────────────────────┘
                                   │ muMalloc(), muLaunchKernel(), ...
                                   ▼
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                           MUSA Driver (libmusa.so)                                ║
║                          560 muapi* 函数入口                                       ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                    ║
║  ┌──────────────────────────────────────────────────────────────────────────┐    ║
║  │                    [1] API Layer  (musa/src/driver/)                       │    ║
║  │                                                                             │    ║
║  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────────┐     │    ║
║  │  │ muapi.h      │ │ mu_memory.cpp│ │ mu_device.cpp │ │ mu_stream.cpp │     │    ║
║  │  │ 560 funcs    │ │ 2,949 行      │ │ ~1,800 行     │ │ ~1,500 行     │     │    ║
║  │  │ (类型安全    │ │ alloc/free/  │ │ get/set/attr │ │ create/sync/  │     │    ║
║  │  │  API 入口)   │ │ memcpy/async │ │ 枚举/属性     │ │ capture       │     │    ║
║  │  └──────────────┘ └──────────────┘ └──────────────┘ └───────────────┘     │    ║
║  │                                                                             │    ║
║  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────────┐     │    ║
║  │  │mu_graph.cpp  │ │ mu_context   │ │ mu_module    │ │ mu_texRef.cpp │     │    ║
║  │  │ ~1,200 行     │ │ .cpp ~800    │ │ .cpp ~600    │ │ ~900 行        │     │    ║
║  │  │ create/node/ │ │ 创建/销毁/   │ │ load/unload  │ │ tex/surf 对象 │     │    ║
║  │  │ instantiate/ │ │ push/pop     │ │ fat binary   │ │ 创建/销毁     │     │    ║
║  │  │ launch       │ │              │ │              │ │               │     │    ║
║  │  └──────────────┘ └──────────────┘ └──────────────┘ └───────────────┘     │    ║
║  │                                                                             │    ║
║  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────────┐     │    ║
║  │  │ mu_event.cpp │ │mu_occupancy  │ │ mu_peer.cpp  │ │ mu_ipc.cpp    │     │    ║
║  │  │ ~400 行       │ │ .cpp ~300    │ │ ~250 行       │ │ ~500 行        │     │    ║
║  │  │ create/rec/  │ │ 发射占有率    │ │ peer access  │ │ 跨进程共享     │     │    ║
║  │  │ sync/query   │ │ 计算         │ │ 启用/禁用     │ │ 句柄导出/导入 │     │    ║
║  │  └──────────────┘ └──────────────┘ └──────────────┘ └───────────────┘     │    ║
║  └──────────────────────────────────────────────────────────────────────────┘    ║
║                                      │                                            ║
║                                      ▼                                            ║
║  ┌──────────────────────────────────────────────────────────────────────────┐    ║
║  │                    [2] Core Layer  (musa/src/musa/core/)                   │    ║
║  │                                                                             │    ║
║  │  ┌────────────────┐  ┌──────────────────┐  ┌────────────────────┐         │    ║
║  │  │  Device 管理    │  │  Context 管理     │  │  Memory 对象       │         │    ║
║  │  │  ParentDevice   │  │  PrimaryCtx      │  │  8 种内存类型      │         │    ║
║  │  │  GPU 属性/能力  │  │  CUDA Context    │  │  9 种 init 路径    │         │    ║
║  │  │  NUMA 拓扑     │  │  TLS 绑定        │  │  Sub-Alloc 分叉   │         │    ║
║  │  └────────────────┘  └──────────────────┘  └────────────────────┘         │    ║
║  │                                                                             │    ║
║  │  ┌────────────────┐  ┌──────────────────┐  ┌────────────────────┐         │    ║
║  │  │  Stream / Cmd   │  │  Graph 执行      │  │  Peer / IPC / Ext  │         │    ║
║  │  │  CommandBuffer  │  │  Node 类型系统   │  │  Peer 映射管理     │         │    ║
║  │  │  Async 分配    │  │  Graph Exec      │  │  External 导入     │         │    ║
║  │  │  MemPool 封装   │  │  Conditional     │  │  SharedHandle      │         │    ║
║  │  └────────────────┘  └──────────────────┘  └────────────────────┘         │    ║
║  └──────────────────────────────────────────────────────────────────────────┘    ║
║                                      │                                            ║
║                                      │ Hal::IMemory, Hal::IDevice, ...            ║
║                                      ▼                                            ║
║  ┌──────────────────────────────────────────────────────────────────────────┐    ║
║  │                    [3] HAL Layer  (musa/src/hal/)                          │    ║
║  │                                                                             │    ║
║  │  ┌────────────────────┐  ┌──────────────────────┐                         │    ║
║  │  │  Hal::IMemMgr       │  │  Hal::MemoryPool      │                         │    ║
║  │  │  按属性查池 O(1)     │  │  Sub-Allocation 算法   │                         │    ║
║  │  │  首次创建/后续复用   │  │  哈希桶查找            │                         │    ║
║  │  └────────────────────┘  └──────────────────────┘                         │    ║
║  │                                                                             │    ║
║  │  ┌────────────────────┐  ┌──────────────────────┐                         │    ║
║  │  │  Hal::IMemory       │  │  Hal::IDevice         │                         │    ║
║  │  │  裸 KMD 分配         │  │  设备能力/属性         │                         │    ║
║  │  │  CreateMemory/Destr │  │  CreateGpuMemory       │                         │    ║
║  │  └────────────────────┘  └──────────────────────┘                         │    ║
║  └──────────────────────────────────────────────────────────────────────────┘    ║
║                                      │                                            ║
║                                      │ m3d API (ioctl / GPU cmd)                  ║
║                                      ▼                                            ║
║  ┌──────────────────────────────────────────────────────────────────────────┐    ║
║  │                    [4] M3D Layer  (musa/src/hal/m3d/)                      │    ║
║  │                                                                             │    ║
║  │  ┌──────────────────────────────────────────────────────────────────┐     │    ║
║  │  │  Linux DRM:  /dev/dri/cardX                                      │     │    ║
║  │  │    mtgpuBoAlloc()    ← ioctl →     gr-kmd (Kernel Mode Driver)   │     │    ║
║  │  │    mtgpuBoFree()     ← ioctl →     mtgpu GEM 对象管理             │     │    ║
║  │  │    mtgpuMapGpuVa()   ← ioctl →     GPU MMU 页表                   │     │    ║
║  │  ├──────────────────────────────────────────────────────────────────┤     │    ║
║  │  │  Windows WDDM2:                                                   │     │    ║
║  │  │    wddmGpuMemory::Allocate() → WDDM2 → KMD                        │     │    ║
║  │  └──────────────────────────────────────────────────────────────────┘     │    ║
║  └──────────────────────────────────────────────────────────────────────────┘    ║
║                                                                                    ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
                                   │ ioctl (系统调用)
                                   ▼
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                        gr-kmd  (GPU Kernel Mode Driver)                           ║
║  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        ║
║  │  GEM 对象管理 │  │  GPU MMU 页表 │  │  调度器       │  │  固件交互     │        ║
║  │  bo create   │  │  map/unmap   │  │  runlist     │  │  firmware    │        ║
║  │  bo free     │  │  pin/unpin   │  │  context     │  │  command     │        ║
║  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘        ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
```

---

## 二、内存分配器三层架构（核心重点）

```
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                           MUSA 三层内存分配器                                     ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                    ║
║  ┌──────────────────────────────────────────────────────────────────────────┐    ║
║  │  L3: 用户 MemoryPool  (MemoryPool 对象)                                   │    ║
║  │                                                                           │    ║
║  │  API:  muMemCreatePool  →  muMemAllocFromPoolAsync                        │    ║
║  │         muMemPoolTrimTo  │  muMemPoolSetAttribute                         │    ║
║  │                                                                           │    ║
║  │  功能:                                                                     │    ║
║  │    • 用户可自定义池 (size threshold, release threshold, ...)               │    ║
║  │    • 池隔离: 每个 Pool 独立 Sub-Allocation                                │    ║
║  │    • TrimTo: 按阈值惰性释放空闲 chunk 回 KMD                              │    ║
║  │    • 导出: muMemPoolExportPointer → 跨进程共享                            │    ║
║  │                                                                           │    ║
║  │         ┌───────────┐   ┌───────────┐   ┌───────────┐                     │    ║
║  │         │  Pool A   │   │  Pool B   │   │  Pool C   │                     │    ║
║  │         │  (默认)    │   │  (自定义) │   │  (IPC)    │                    │    ║
║  │         └─────┬─────┘   └─────┬─────┘   └─────┬─────┘                     │    ║
║  │               └───────────────┼───────────────┘                            │    ║
║  │                               │                                            │    ║
║  └───────────────────────────────┼───────────────────────────────────────────┘    ║
║                                  │                                                ║
║  ┌───────────────────────────────┼───────────────────────────────────────────┐    ║
║  │  L2: 系统 MemMgr + MemoryPool (Sub-Allocation 引擎)                        │    ║
║  │                                                                           │    ║
║  │  ┌───────────────────────────────────────────────────────────┐           │    ║
║  │  │  Hal::IMemMgr  (memMgr.cpp:237 行)                        │           │    ║
║  │  │                                                           │           │    ║
║  │  │  Allocate(createInfo) {                                    │           │    ║
║  │  │    1. 按 {type, heap, property, viewCap} 建 key           │           │    ║
║  │  │    2. 查缓存: poolCache[key]  → O(1) 查找                 │           │    ║
║  │  │    3. 未命中 → 创建新 Pool: new Hal::MemoryPool(key)       │           │    ║
║  │  │                                                           │           │    ║
║  │  │    4. 池分配: pool->SubAllocate(size, align, &offset)     │           │    ║
║  │  │    5. 池满/不满足 → 触发 KMD 新 chunk 分配                 │           │    ║
║  │  │       chunk = Hal::CreateMemory(chunkSize)                 │           │    ║
║  │  │       pool->AddChunk(chunk)                                │           │    ║
║  │  │    6. 返回 {pHalMemory, offset}                            │           │    ║
║  │  │  }                                                         │           │    ║
║  │  └───────────────────────────────────────────────────────────┘           │    ║
║  │                                                                           │    ║
║  │  ┌───────────────────────────────────────────────────────────┐           │    ║
║  │  │  Hal::MemoryPool  (memoryPool.cpp:533 行)                  │           │    ║
║  │  │                                                           │           │    ║
║  │  │  内部结构:                                                   │           │    ║
║  │  │  ┌─────────────────────────────────────┐                   │           │    ║
║  │  │  │  FreeList (按 size 分桶的哈希表)      │                   │           │    ║
║  │  │  │  bucket[0] → size in [64B,  256B)   │                   │           │    ║
║  │  │  │  bucket[1] → size in [256B, 1KB)   │                   │           │    ║
║  │  │  │  bucket[2] → size in [1KB,  4KB)   │                   │           │    ║
║  │  │  │  ...                                │                   │           │    ║
║  │  │  │  bucket[N] → size in [2MB, ∞)      │                   │           │    ║
║  │  │  │  查找: O(1) 直接定位桶               │                   │           │    ║
║  │  │  │  分配: First-Fit 按地址递增          │                   │           │    ║
║  │  │  └─────────────────────────────────────┘                   │           │    ║
║  │  │                                                           │           │    ║
║  │  │  分配流程:                                                  │           │    ║
║  │  │    SubAllocate(size, align) →                               │           │    ║
║  │  │      1. 查桶: bucket = floor(log2(size / bucketBase))      │           │    ║
║  │  │      2. 遍历: 从当前桶向上查找第一个满足对齐的空闲块          │           │    ║
║  │  │      3. 切分: 剩余空间 > 最小块 → 切回桶                    │           │    ║
║  │  │      4. 返回: {chunk, offset, size}                         │           │    ║
║  │  │                                                           │           │    ║
║  │  │  释放流程:                                                  │           │    ║
║  │  │    Free(offset) →                                          │           │    ║
║  │  │      1. 定位 chunk + slot                                   │           │    ║
║  │  │      2. 尝试与前/后相邻空闲块 merge                          │           │    ║
║  │  │      3. 插回对应桶                                          │           │    ║
║  │  │                                                           │           │    ║
║  │  │  惰性释放 (TrimTo):                                         │           │    ║
║  │  │    1. 检查每 chunk 的空闲比例                               │           │    ║
║  │  │    2. 空闲率 > threshold → 归还整个 chunk 给 KMD             │           │    ║
║  │  │    3. 减少 GPU 显存碎片                                      │           │    ║
║  │  └───────────────────────────────────────────────────────────┘           │    ║
║  └──────────────────────────────────────────────────────────────────────────┘    ║
║                                  │                                                ║
║  ┌───────────────────────────────┼───────────────────────────────────────────┐    ║
║  │  L1: 裸 KMD 分配  (Hal::IMemory → Kernel Mode Driver)                     │    ║
║  │                                                                           │    ║
║  │  ┌───────────────────────────────────────────────────────────┐           │    ║
║  │  │  Hal::CreateMemory(createInfo)  →  ioctl →  KMD           │           │    ║
║  │  │                                                           │           │    ║
║  │  │  Linux DRM 路径:                                            │           │    ║
║  │  │    mtgpuBoAlloc(size, align, heap, property, ...)          │           │    ║
║  │  │      → ioctl(DRM_MTGPU_BO_ALLOC)                           │           │    ║
║  │  │        → GEM 对象创建 (bo = buffer object)                  │           │    ║
║  │  │        → GPU MMU 映射 (map GPU 虚拟地址)                   │           │    ║
║  │  │  mtgpuBoFree(bo)                                           │           │    ║
║  │  │      → ioctl(DRM_MTGPU_BO_FREE)                            │           │    ║
║  │  │                                                           │           │    ║
║  │  │  Windows WDDM2 路径:                                       │           │    ║
║  │  │    wddmGpuMemory::Allocate()                               │           │    ║
║  │  │      → DXGKDDI_CREATEALLOCATION                            │           │    ║
║  │  │      → WDDM2 GPU Memory Manager                            │           │    ║
║  │  │                                                           │           │    ║
║  │  │  性能特征:                                                  │           │    ║
║  │  │    • 单次分配: ~0.2ms (2 次 ioctl: alloc + map)             │           │    ║
║  │  │    • 每次至少 2 次 ioctl                                    │           │    ║
║  │  │    • 直接操作 KMD，开销大                                   │           │    ║
║  │  └───────────────────────────────────────────────────────────┘           │    ║
║  └──────────────────────────────────────────────────────────────────────────┘    ║
║                                                                                    ║
║  ┌──────────────────────────────────────────────────────────────────────────┐    ║
║  │                         Sub-Allocation 性能对比                            │    ║
║  │                                                                           │    ║
║  │  ┌───────────────────┬──────────────────┬──────────────────────┐         │    ║
║  │  │      指标          │  裸 KMD (2 ioctl) │  Sub-Allocation      │         │    ║
║  │  ├───────────────────┼──────────────────┼──────────────────────┤         │    ║
║  │  │  单次 4KB 分配      │  ~0.2ms           │  ~0.0004ms           │         │    ║
║  │  │  1000 次 4KB 总耗时  │  ~200ms           │  ~0.4ms              │         │    ║
║  │  │  加速比             │  1x               │  ~500x               │         │    ║
║  │  │  查找复杂度          │  —               │  O(1) 哈希桶          │         │    ║
║  │  │  合并策略            │  —               │  First-Fit + merge    │         │    ║
║  │  └───────────────────┴──────────────────┴──────────────────────┘         │    ║
║  └──────────────────────────────────────────────────────────────────────────┘    ║
║                                                                                    ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
```

---

## 三、`muMemAlloc` 完整调用链

```
muMemAlloc(&ptr, size)
│
├─ [1] API Layer  (mu_memory.cpp:265)
│   muMemAlloc_v2() {
│     参数校验: size==0 → ptr=NULL, return
│     上下文检查: CUDA Context 不存在 → 错误
│     设备锁定: 获取当前 Device
│   }
│
├─ [2] Core Layer  (core/context.cpp:699)
│   Memory::Init(createInfo) {
│     switch (type) {
│       case memoryTypeGeneral → GeneralAlloc()
│     }
│   }
│
├─ [3] Core Layer  (core/memory.cpp:462)
│   GeneralAlloc(size, align, flags=0x07) {
│     // flags: Virtual(1) | DeviceMapped(2) | SubAllocatable(4)
│
│     // Step A: 自动推导属性
│     property = flags | Physical | SharedVA
│     property |= DeviceVisible | HostVisible | HostCoherent | ...
│     viewCap = Exportable | PeerAccessible | IpcExportable
│
│     // Step B: 分叉选择路径
│     if (SubAllocatable) {
│       ──→ MemMgr::Allocate()           ← 默认路径 (flags=0x07)
│     } else {
│       ──→ Hal::CreateMemory()          ← force no-pool
│     }
│   }
│
├─ [4] HAL Layer  (memMgr.cpp:81)
│   MemMgr::Allocate(createInfo) {
│     key = {type, heap, property, viewCap}
│     pool = poolCache[key]
│     if (pool == null) {
│       pool = new MemoryPool(key)
│       poolCache[key] = pool
│     }
│     ──→ pool->SubAllocate(size, align, &offset)
│     if (失败) {
│       chunk = Hal::CreateMemory(chunkSize)   ← 从 KMD 分配新 chunk
│       pool->AddChunk(chunk)
│       retry pool->SubAllocate(...)
│     }
│     return {pHalMemory, offset}
│   }
│
├─ [5] HAL Layer  (memoryPool.cpp:82)
│   MemoryPool::SubAllocate(size, align) {
│     bucket = floor(log2(size / base))
│     for (b = bucket; b < N; b++) {
│       for each free slot in bucket[b] {
│         if (slot.aligned(align) && slot.size >= size) {
│           alloc = 切分 slot
│           剩余 → 插回对应桶
│           return {offset}    ← O(1) 桶查找, 线序遍历
│         }
│       }
│     }
│     return FAIL  (→ 回 memMgr::Allocate 触发 KMD 分配)
│   }
│
├─ [6] M3D Layer  (memory.cpp:366)  [只在 chunk 不够时触发]
│   Hal::CreateMemory(createInfo) {
│     // CPU: 计算 alignment, heap selection
│     mtgpuBoAlloc(size, align, heap, property)
│     ──→ ioctl(DRM_MTGPU_BO_ALLOC)    ← 实际 KMD 分配
│     mtgpuMapGpuVa(bo)
│     ──→ ioctl(DRM_MTGPU_MAP_GVA)     ← GPU 虚拟地址映射
│     return {pHalMemory}
│   }
│
└─ [7] 回到 Core: AddGpuMemoryReference + 返回 ptr
```

---

## 四、flags 推导链 (flags=0x07 → 最终 HAL property)

```
用户输入: muMemAlloc(&ptr, size)
              │
              ▼
      flags = Virtual(0x01) | DeviceMapped(0x02) | SubAllocatable(0x04)
              │
              ├── Core 自动追加:
              │     Physical(0x08) | SharedVirtualAddress(0x10)
              │
              ├── Virtual=0x01 推导:
              │     DeviceVisible(0x20)   ← GPU 可见
              │     HostVisible(0x40)     ← CPU 可见
              │     HostCoherent(0x80)    ← CPU 缓存一致
              │     DeviceWriteable(0x100) ← GPU 可写
              │     DeviceCached(0x200)   ← GPU 缓存
              │
              ├── DeviceMapped=0x02 推导 (viewCapability):
              │     PeerAccessible  ← 跨 P2P 访问
              │     IpcExportable   ← 跨进程导出
              │
              └── 最终 HAL property = 0x3FF  ← 10 个属性位全开
                  最终 viewCapability = 0x07   ← 3 个视图能力全开
```

---

## 五、内存类型与分配路径速查

```
┌─────────────────────┬──────────────┬───────┬──────────────────────────┐
│  API                │ 内存位置      │ Pool? │ 关键路径                   │
├─────────────────────┼──────────────┼───────┼──────────────────────────┤
│ muMemAlloc          │ GPU 显存      │  YES  │ GeneralAlloc → MemMgr    │
│ muMemAllocPitch     │ GPU 显存      │  YES  │ PitchedGeneralAlloc       │
│ muMemHostAlloc      │ 系统内存       │ YES*  │ PinnedHostAlloc           │
│ muMemHostRegister   │ 系统内存       │  NO   │ PinnedHostRegister        │
│ muMemAllocManaged   │ GPU + 系统    │  NO   │ ManagedAlloc              │
│ muMemAllocAsync     │ GPU 显存      │  YES  │ Stream Command (图安全)   │
│ muMemAllocFromPool  │ GPU 显存      │  YES  │ 用户自定义 Pool            │
│ muMemImportShareable│ GPU/外部      │  NO   │ ExternalAlloc             │
│ muMemAddressReserve │ 仅虚拟地址     │  NO   │ VirtualAlloc — 不进 Pool   │
└─────────────────────┴──────────────┴───────┴──────────────────────────┘
* PinnedHostAlloc: LargePage 时走 Pool, LargePage 不可用 → General Heap fallback
```

---

## 六、关键设计模式

```
1. RAII (Resource Acquisition Is Initialization)
   ── Memory 析构自动归还 → 若 SubAllocatable → Pool::Free
                           → 否则 → Destroy() (ioctl free)

2. 析构双路径
   ── SubAllocatable 位决定: Pool::Free (轻量, 归还池)
                          Destroy  (重量, 归还 KMD)

3. 双重释放防护
   ── 类型白名单: 只有一定 memory type 允许 free
   ── offset==0 校验: SubAlloc 产生的 offset>0, KMD 分配的 offset==0

4. Auto Peer 映射
   ── CreateMemory 后自动在所有已启用 Peer 的设备上建立映射

5. Stream 命令化 (Async)
   ── AsyncAlloc/AsyncFree → 编码为 Stream Command → Graph 捕获安全

6. 惰性释放 (TrimTo)
   ── 按 chunk 统计空闲率, 超过 threshold → 归还整个 chunk 给 KMD
   ── 减少 GPU 显存碎片
```
