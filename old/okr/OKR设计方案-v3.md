# MUSA Driver 软件性能建模 — 方案 v3

> 2026-05-26 | 从"API trace 分析"转向"Driver 软件算法建模"

---

## 零、关键转折

v2 被驳回的原因：**方案的本质仍然是对 API trace 做统计回归，没有对 musa 软件本身进行建模。**

v3 的定位：

```
            v2 (被驳回)                          v3 (本方案)
            ──────────                          ──────────
 输入:      Profiling trace                      Driver 源码中的算法逻辑
 建模对象:  API 延迟 = f(输入参数)                  Driver 内部状态机 = f(API 调用序列)
 输出:      预测新 batch_size 的耗时               模拟 driver 内部行为（ioctl次数、碎片率、合并效率）
 验证:      Top50 kernel overlap                  strace ioctl 计数 + profiling trace 对齐
```

**一句话**：从 Driver 源码中提取每个性能关键子系统的算法，构建可执行的软件行为模拟器。Profiling trace 只用来**校准模型常数**和**验证模拟准确性**。

---

## 一、模型架构：6 个子系统模拟器

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MUSA Driver Software Performance Model                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  输入: API 调用序列                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐     │
│  │ muMemAlloc│  │muMemFree │  │muLaunch  │  │muMemcpy  │  │muStreamSync│    │
│  │ (size)    │  │ (ptr)    │  │Kernel(dim)│ │HtoDAsync │  │ (...)     │     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────┬─────┘     │
│       │             │             │             │               │            │
│       ▼             ▼             ▼             ▼               ▼            │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                    Simulator Core                                   │     │
│  │                                                                     │     │
│  │  ┌──────────────────┐  ┌──────────────────┐                        │     │
│  │  │ S1: 内存分配器     │  │ S2: 命令提交管线  │                        │     │
│  │  │                  │  │                  │                        │     │
│  │  │ PoolState {      │  │ SubmitState {    │                        │     │
│  │  │  freeBuckets[64] │  │  mergingLists[]  │                        │     │
│  │  │  chunks[]        │  │  inflightCount[] │                        │     │
│  │  │  totalFree       │  │  timelineValues  │                        │     │
│  │  │ }                │  │ }                │                        │     │
│  │  └──────────────────┘  └──────────────────┘                        │     │
│  │                                                                     │     │
│  │  ┌──────────────────┐  ┌──────────────────┐                        │     │
│  │  │ S3: 流依赖解析    │  │ S4: 图执行      │                        │     │
│  │  │                  │  │                  │                        │     │
│  │  │ DepState {       │  │ GraphState {     │                        │     │
│  │  │  streamLastCmd[] │  │  isDirty         │                        │     │
│  │  │  barrierStream   │  │  nodeCount       │                        │     │
│  │  │ }                │  │  submissions[]   │                        │     │
│  │  └──────────────────┘  └──────────────────┘                        │     │
│  │                                                                     │     │
│  │  ┌──────────────────┐  ┌──────────────────┐                        │     │
│  │  │ S5: Peer映射     │  │ S6: Ioctl计数    │                        │     │
│  │  │                  │  │                  │                        │     │
│  │  │ PeerState {      │  │ ioctlCounts {    │                        │     │
│  │  │  peerEnabled     │  │  BO_ALLOC: N     │                        │     │
│  │  │  allocations[]   │  │  VM_MAP: M       │                        │     │
│  │  │ }                │  │  SUBMIT_V3: K    │                        │     │
│  │  └──────────────────┘  │  GEM_CLOSE: L    │                        │     │
│  │                         │ }               │                        │     │
│  │                         └──────────────────┘                        │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                    │                                         │
│                                    ▼                                         │
│  输出: 驱动内部行为预测                                                        │
│  ┌─────────────────────┬─────────────────────┬─────────────────────────┐    │
│  │ Ioctl 次数（验证点）  │ Pool 碎片率           │ 命令合并效率               │    │
│  │ BO_ALLOC: 12         │ 空闲空间 / 总空间     │ merge_batch_size 分布    │    │
│  │ SUBMIT_V3: 53400     │ 最大连续空块大小       │ inflight 阻塞次数        │    │
│  └─────────────────────┴─────────────────────┴─────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、S1: 内存分配器模型

### 2.1 源码依据

直接从 `memoryPool.cpp` + `memMgr.cpp` 逐行翻译。

### 2.2 状态结构

```python
@dataclass
class Chunk:
    """从 memoryPool.cpp 提取"""
    base: int           # GPU 虚拟地址
    size: int           # = alignUp(request, m_ChunkAllocSize)
    memory: object      # Hal::IMemory handle (KMD 分配的)
    lazy_free_count: int = 0  # 惰性释放计数器

@dataclass
class Segment:
    """从 memoryPool.h ResSegment 翻译"""
    base: int
    size: int
    busy: bool
    is_leftmost: bool
    is_rightmost: bool
    chunk: Chunk
    chunk_base: int

@dataclass
class PoolState:
    """从 memoryPool.h 翻译"""
    free_buckets: list[list[Segment]]  # 64 buckets, 索引=Log2(size)
    bucket_bitmask: int                # m_EltMappingHash: 哪些桶非空
    chunks: list[Chunk]                # 所有 KMD chunk
    total_free: int                    # m_FreeSize
    total_size: int                    # m_TotalSize
    chunk_alloc_size: int = 2*1024*1024  # s_DefaultChunkAllocSize
    reuse_count_limit: int = 2**64-1   # s_DefaultLazyFreeThreshold
    policy: str = "assuredFit"         # "assuredFit" | "bestFit"
```

### 2.3 核心算法（逐行翻译）

```python
def sub_allocate(state: PoolState, size: int, alignment: int) -> tuple[Segment | None, int | None]:
    """
    从 memoryPool.cpp:97-151 翻译
    """
    index_low  = log2(size)
    index_high = log2(size + alignment - 1) if alignment > 1 else index_low

    if state.policy == "bestFit":
        # 扫描从 index_low 向上所有非空桶，找最合适的
        mask = ~((1 << index_low) - 1) & state.bucket_bitmask
        idx = bit_scan_forward(mask)
        end_idx = bit_scan_reverse(mask)
        while idx <= end_idx and segment is None:
            segment = find_bucket(state.free_buckets[idx], size, alignment, UNLIMITED)
            idx += 1
    else:  # assuredFit (默认)
        # 优先从 index_high+1 开始，只试 1 个
        mask = ~((1 << (index_high + 1)) - 1) & state.bucket_bitmask
        idx = bit_scan_forward(mask)
        if idx != FREE_TABLE_LIMIT:
            segment = find_bucket(state.free_buckets[idx], size, alignment, TRY_LIMIT=1)
        else:
            # fallback: 从 index_high 向下扫描
            for idx in range(index_high, index_low - 1, -1):
                segment = find_bucket(state.free_buckets[idx], size, alignment, UNLIMITED)
                if segment: break

    if not segment:
        return None, None  # → 触发 ChunkAllocate

    # ResourceSplit (memoryPool.cpp:358-413)
    aligned_base = align_up(segment.base, alignment)
    offset = aligned_base - segment.chunk_base

    # 从 free list 中移除原 segment
    free_list_remove(state, segment)

    # 前面间隙（alignment 造成的）
    if aligned_base > segment.base:
        front = Segment(segment.base, aligned_base - segment.base, busy=False, ...)
        free_list_insert(state, front)

    # 后面剩余
    remainder = segment.size - (offset + size - (segment.base - segment.chunk_base))
    if remainder > 0:
        back = Segment(aligned_base + size, remainder, busy=False, ...)
        free_list_insert(state, back)

    state.total_free -= size
    return segment.chunk.memory, offset


def full_allocate(state: PoolState, size: int, alignment: int) -> tuple:
    """
    从 memoryPool.cpp:82-95 翻译
    """
    memory, offset = sub_allocate(state, size, alignment)
    if memory is None:
        # ← 关键决策点: 触发 KMD ioctl
        chunk_allocate(state, size, alignment)
        memory, offset = sub_allocate(state, size, alignment)
    return memory, offset


def chunk_allocate(state: PoolState, size: int, alignment: int):
    """
    从 memoryPool.cpp:153-212 翻译
    """
    chunk_size = size
    if alignment > state.chunk_alloc_size:
        chunk_size += alignment - state.chunk_alloc_size
    chunk_size = align_up(chunk_size, state.chunk_alloc_size)  # → 2MB 粒度!

    # 调用 KMD (模拟为: 创建 Chunk 对象)
    chunk = Chunk(base=next_gpu_va(), size=chunk_size)

    segment = Segment(chunk.base, chunk.size, busy=False,
                      is_leftmost=True, is_rightmost=True,
                      chunk=chunk, chunk_base=chunk.base)
    free_list_insert(state, segment)
    state.chunks.append(chunk)
    state.total_free += chunk.size
    state.total_size += chunk.size
    # 实际 driver 在此调用: m_pDevice->CreateMemory() → ioctl(MTGPU_BO_CMD_ALLOC)
```

### 2.4 验证方式

**模拟器输入**: Qwen-7B 推理的 muMemAlloc 调用序列（来自 profiler trace）  
**模拟器输出**: ChunkAllocate 触发次数 = BO_ALLOC ioctl 次数  
**真实值**: `strace -e ioctl` 计数 `MTGPU_BO_CMD_ALLOC`  
**验证标准**: 预测值 == 真实值（确定性算法，应 100% 匹配）

---

## 三、S2: 命令提交管线模型

### 3.1 源码依据

直接从 `stream.cpp:1008-1281` 逐行翻译。

### 3.2 状态结构

```python
@dataclass
class SubmitState:
    """从 stream.cpp + stream.h 提取"""
    # 每个 stream 的状态
    command_queue: list[Command]    # m_CommandList
    merging_list: list[Command]     # m_MergingList (当前合并批次)
    inflight_list: list[Command]    # m_InflightList
    inflight_count: dict[int,int]   # 每个 engine 的 inflight 提交数
    timeline_value: int             # m_TimelineValue (递增的信号值)

    # 常量
    MERGE_MAX_SIZE = 32             # mergingListMaxSize
    INFLIGHT_USER_LIMIT = 3         # s_InflightSubmissionLimit
    INFLIGHT_NON_USER_LIMIT = 2     # s_NonUserQueueInflightSubmissionLimit
    ASYNC_CAPACITY = 1024           # streamAsyncCapacity (spin-lock 阈值)
```

### 3.3 核心算法

```python
def queue_command(state: SubmitState, cmd: Command):
    """
    从 stream.cpp:1008-1058 翻译
    """
    # 节流: spin-wait until queue has room
    while state.async_count >= state.ASYNC_CAPACITY:
        yield_cpu()

    state.async_count += 1
    cmd.prev = state.last_command
    state.last_command = cmd
    state.command_queue.append(cmd)
    notify_submit_thread()


def submit_thread_loop(state: SubmitState):
    """
    从 stream.cpp:1111 + 1195 + 1233 综合翻译
    """
    while True:
        wait_for_commands(state)

        while state.command_queue:
            cmd = state.command_queue.pop(0)

            # Step 1: build
            cmd.filter_dependency()
            can_merge = cmd.can_merge_to(state.merging_list)

            if not can_merge:
                # 当前批次无法继续合并 → flush
                flush_merging_list(state)

            cmd.build(state.merging_list)

            # Step 2: push to merging list
            state.merging_list.append(cmd)

            # Step 3: check if should flush now
            if should_stop_merging(state):
                flush_merging_list(state)


def should_stop_merging(state: SubmitState) -> bool:
    """
    从 stream.cpp:1233-1249 翻译
    """
    if not state.merging_list:
        return False
    return (state.merging_list.size >= state.MERGE_MAX_SIZE or      # ← 32 上限
            engine_submission_ready(state.merging_list[0].engine))  # ← 有 slot 就提交


def engine_submission_ready(state: SubmitState, engine: int) -> bool:
    """检查引擎的 inflight 提交数是否小于 limit"""
    limit = (state.INFLIGHT_USER_LIMIT if engine.is_user_queue
             else state.INFLIGHT_NON_USER_LIMIT)
    return state.inflight_count[engine] < limit


def flush_merging_list(state: SubmitState):
    """
    从 stream.cpp:1112-1193 翻译
    """
    if not state.merging_list:
        return

    engine = state.merging_list[0].engine

    # 等待 engine 有空 slot
    while not engine_submission_ready(state, engine):
        wait_for_inflight_completion(state, engine)

    # 主命令提交（合并批次中的其他命令作为 secondary）
    status = state.merging_list[0].submit()  # → ioctl(SUBMIT_V3)

    # 转移合并列表到 inflight 列表
    state.inflight_list.extend(state.merging_list)
    state.inflight_count[engine] += 1
    state.merging_list.clear()
    notify_wait_thread()
```

### 3.4 验证方式

**模拟器输入**: Qwen-7B 推理的 muLaunchKernel + muMemcpyAsync 调用序列（含 stream ID）  
**模拟器输出**: SUBMIT_V3 ioctl 次数、merge batch size 分布  
**真实值**: `strace -e ioctl` 计数 `MTGPU_JOB_CMD_SUBMIT_V3`  
**验证标准**: 预测值 == 真实值

---

## 四、S3: 流依赖解析模型

### 3.1 源码依据

从 `context.cpp:1845-1907`（`ResolveDependencyAndQueueCommand`）翻译。

### 3.2 依赖规则

```
对当前 stream S 上的新命令 cmd:

if S == default_stream (LEGACY):
    for each non_blocking_stream:
        cmd.add_dependency(last_command_of(stream))   // 依赖所有非阻塞流
    for each blocking_stream:
        last_command_of(stream).add_dependency(cmd)   // 阻塞流依赖默认流

if S is blocking:
    cmd.add_dependency(last_command_of(default_stream))

if S is non_blocking:
    cmd.add_dependency(last_command_of(barrier_stream))

always:
    cmd.add_dependency(last_command_of(S))            // 同流保序

barrier_stream:
    cmd.add_dependency(last_command_of(every_stream))
    every_stream.add_dependency(cmd)
```

### 3.3 模型价值

预测：给定 N 条流的命令交错模式，哪些提交因为是 default stream 上的而被序列化（无法并行）。

---

## 五、S4-S6: 图执行 / Peer 映射 / Ioctl 计数

简要算法规格（需要从源码进一步提取，但核心逻辑已清楚）：

**S4 - Graph Instantiation Model**:
- 输入: graph node 列表 + 是否修改了 params
- 关键决策: `GraphExec::Init()` 的 BFS 拓扑排序 + `CreateExecResource` + `PrepareAllSubmissions` 的重建成本
- 输出: 实例化是否触发了 full rebuild

**S5 - Peer Mapping Model**:
- 状态: `peer_enabled: bool`, `n_allocations: int`
- 输入: `muCtxEnablePeer`, `muMemAlloc`
- 关键决策: 每次 alloc 后 `MapToPeers()` 遍历所有 peer（O(peerCount) ioctl）
- 输出: 额外的 peer ioctl 次数

**S6 - Ioctl 综合计数**:
- 聚合 S1-S5 的所有 ioctl 触发点
- 输出: 按类型分类的 ioctl 预测计数

---

## 六、与 Profiling 数据的关系（关键）

Profiling trace 在三处使用，但**都不是建模输入**：

```
1. 校准常数参数
   - 单次 SUBMIT_V3 ioctl 的基线延迟 (从 strace + 计时)
   - 单次 BO_ALLOC ioctl 的基线延迟
   - 池的 chunk_alloc_size 可能因平台而异（默认 2MB）

2. 生成 API 调用序列（模拟器输入）
   - 从 profiler trace 提取 API 调用的顺序和参数
   - 喂给模拟器 → 模拟器输出 ioctl 预测

3. 验证模拟准确性
   - 模拟器输出 vs strace 真实 ioctl 计数
   - 模拟器输出 vs profiler 的时序特征
```

---

## 七、KR 对齐

| KR | v3 方案如何满足 |
|----|---------------|
| **KR1**: 覆盖 Top90% driver API | 6 个子系统模型覆盖 memory/stream/event/graph/peer 类 API，这些 API 在 trace 中累计耗时占 Top90%。模型输出是 driver 对这些 API 的**内部处理**，不是 API 的延迟回归。 |
| **KR2**: Top50 kernel overlap ≥90% | 模拟器预测的 merge batch 决定了哪些 kernel 在同一批次提交。与实际 profiling 的 kernel 排序对齐，是验证模拟器准确性的方式。**不是模型直接拟合 kernel 延迟。** |
| **KR3**: 分析报告 | 报告包含：模拟器架构文档 + 3 模型的模拟结果 vs profiling 对比 + 识别出的瓶颈机制（碎片化、inflight 阻塞、不必要的 peer ioctl 等） + 优化建议 |

---

## 八、里程碑

```
Week 1: S1(内存分配器) + S2(命令提交) 模拟器实现
        验证: strace ioctl 计数 vs 模拟器预测，目标 100% 匹配
Week 2: S3(流依赖) + S5(Peer映射) + S6(ioctl综合) 实现
        采集 3 模型 profiler trace → 生成 API 调用序列
Week 3: 3 模型模拟运行 → 输出 ioctl 预测 → strace 验证
        校准常数参数
Week 4: S4(图执行) 实现
        完整 6 子系统联调 → KR2 验证
Week 5: 瓶颈分析 → 优化建议 → 3 份报告（KR3）
Week 6: OKR2 行为模式提取 + CTS 用例开发
```
