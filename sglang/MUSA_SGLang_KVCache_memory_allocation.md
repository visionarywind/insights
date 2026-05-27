# MUSA SGLang KV Cache 显存申请技术洞察

分析对象：远程仓库 `/home/shanfeng/workspace/sglang`。

关注问题：MUSA SGLang 的 KV cache 显存是直接调用底层 MUSA API 申请，还是通过 PyTorch/torch_musa 的内存分配体系申请。

## 结论

KV cache 的主体显存不是由 SGLang 直接调用 `musaMalloc` 或 `muMemAlloc` 申请。

从 SGLang 源码看，KV cache 显存由 Python 层创建 `torch.Tensor` 完成，主要入口是：

```python
torch.zeros(..., device=self.device)
torch.empty(..., device=device)
```

在 MUSA 模式下，`self.device` 或 `device` 为 `"musa"`。因此，这些分配会进入 PyTorch 的 MUSA 后端，也就是 torch_musa/torchada 对 PyTorch Tensor 分配路径的实现。SGLang 自己的 `TokenToKVPoolAllocator` 和 `PagedTokenToKVPoolAllocator` 只管理 KV cache 内部的 token/page 下标，不负责申请大块显存。

默认路径可以概括为：

```text
ModelRunner.init_memory_pool()
  -> 计算可用显存和 KV cache token 容量
  -> 创建 ReqToTokenPool
  -> 创建 TokenToKVPool
  -> torch.zeros / torch.empty(device="musa")
  -> PyTorch MUSA 后端分配 Tensor 显存
  -> SGLang allocator 管理 Tensor 内部的可用 token/page 下标
```

只有在启用 `SGLANG_MOONCAKE_CUSTOM_MEM_POOL` 时，部分 KV cache Tensor 分配会被包在 `torch.cuda.use_mem_pool(self.custom_mem_pool)` 上下文中。即便如此，源码层仍是 PyTorch Tensor 分配，不是 KV cache 代码直接调用底层 MUSA 分配 API。

## 核心概念

### 物理显存分配

物理显存分配是创建真正占用 GPU 显存的 Tensor。例如：

```python
self.k_buffer = [
    torch.zeros((self.size + self.page_size, self.head_num, self.head_dim),
                dtype=self.store_dtype,
                device=self.device)
    for _ in range(self.layer_num)
]
```

这类语句会申请实际显存。

### SGLang KV allocator

SGLang 中的 allocator 名称容易误解。`TokenToKVPoolAllocator` 和 `PagedTokenToKVPoolAllocator` 不是底层显存 allocator。它们维护的是已经申请好的 KV cache Tensor 中哪些 token/page 位置可用。

典型逻辑是：

```python
self.free_pages = torch.arange(1, self.size + 1, dtype=torch.int64, device=self.device)
select_index = self.free_pages[:need_size]
self.free_pages = self.free_pages[need_size:]
return select_index
```

这里返回的是 KV cache 中的位置编号，不是新的显存指针。

## 初始化链路

入口在 `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py`。

### 1. 计算可用于 KV cache 的显存

源码位置：`model_runner_kv_cache_mixin.py:60-75`

```python
post_model_load_memory = get_available_gpu_memory(self.device, self.gpu_id, ...)
rest_memory = post_model_load_memory - pre_model_load_memory * (
    1 - self.mem_fraction_static
)
return int(rest_memory * (1 << 30))
```

作用：

- 模型权重加载完成后，再查询剩余显存。
- 根据 `mem_fraction_static` 计算可给 KV cache 使用的显存。
- 返回字节数，供后续计算最大 token 数。

MUSA 的可用显存查询在 `python/sglang/srt/utils/common.py:618-636`：

```python
elif device == "musa":
    num_gpus = torch.musa.device_count()
    props = torch.musa.get_device_properties(gpu_id)
    free_gpu_memory, total_gpu_memory = torch.musa.mem_get_info()
```

这一步只查询显存容量，不分配 KV cache。

### 2. 计算 KV cache token 容量

源码位置：`python/sglang/srt/model_executor/pool_configurator.py`

普通 MHA/MLA 模型使用 `DefaultPoolConfigurator`。核心公式是：

```text
max_total_num_tokens = available_bytes / bytes_per_token
```

MHA 场景下，每个 token 的 KV cache 成本近似为：

```text
num_kv_heads * (head_dim + v_head_dim) * layer_num * dtype_size
```

MLA 场景下，每个 token 的 KV cache 成本近似为：

```text
(kv_lora_rank + qk_rope_head_dim) * layer_num * dtype_size
```

DeepSeek V4 使用专门的配置逻辑，会把显存拆到 SWA、C4、C128、compress state、indexer 等多个池。

### 3. 创建 memory pool

入口：`model_runner_kv_cache_mixin.py:830-840`

```python
def init_memory_pool(self, pre_model_load_memory: int):
    self.memory_pool_config = self._resolve_memory_pool_config(pre_model_load_memory)
    self._apply_memory_pool_config(self.memory_pool_config)
```

随后进入 `_init_pools()`，源码位置：`model_runner_kv_cache_mixin.py:199-722`。

主要创建三类对象：

| 对象 | 作用 | 是否申请大块 KV 显存 |
|---|---|---|
| `ReqToTokenPool` | request 到 token 位置的映射表 | 申请映射表 Tensor |
| `MHATokenToKVPool` / `MLATokenToKVPool` / `DeepSeekV4TokenToKVPool` | KV cache 数据池 | 申请 KV cache 主体 Tensor |
| `TokenToKVPoolAllocator` / `PagedTokenToKVPoolAllocator` | 管理可用 token/page 下标 | 不申请 KV cache 主体显存 |

## 普通 MHA KV cache 分配

源码位置：`python/sglang/srt/mem_cache/memory_pool.py:799-853`

`MHATokenToKVPool.__init__()` 调用 `_create_buffers()`。

源码位置：`memory_pool.py:906-930`

```python
self.k_buffer = [
    torch.zeros(
        (self.size + self.page_size, self.head_num, self.head_dim),
        dtype=self.store_dtype,
        device=self.device,
    )
    for _ in range(self.layer_num)
]
self.v_buffer = [
    torch.zeros(
        (self.size + self.page_size, self.head_num, self.v_head_dim),
        dtype=self.store_dtype,
        device=self.device,
    )
    for _ in range(self.layer_num)
]
```

含义：

- 每一层创建一个 K buffer。
- 每一层创建一个 V buffer。
- shape 中的 `self.size + self.page_size` 包含 padding 区域。
- `device=self.device` 在 MUSA 模式下是 `"musa"`。

所以普通 MHA KV cache 的主体显存来自 `torch.zeros(..., device="musa")`。

## 普通 MLA KV cache 分配

源码位置：`memory_pool.py:1517-1569`

`MLATokenToKVPool` 不单独保存完整的 K/V，而是把 MLA 所需的 latent KV 存入一个 `kv_buffer`。

源码位置：`memory_pool.py:1571-1586`

```python
self.kv_buffer = [
    torch.zeros(
        (self.size + self.page_size, 1, self.kv_cache_dim),
        dtype=self.store_dtype,
        device=self.device,
    )
    for _ in range(self.layer_num)
]
```

含义：

- 每层一个 `kv_buffer`。
- `kv_cache_dim = kv_lora_rank + qk_rope_head_dim`，NSA/FP8 场景会有额外处理。
- 显存仍然通过 `torch.zeros(..., device="musa")` 申请。

## DeepSeek V4 KV cache 分配

DeepSeek V4 的 KV cache 不是单一池，而是组合池。

源码位置：`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py:351-467`

`DeepSeekV4TokenToKVPool` 内部创建：

```text
swa_kv_pool
c4_kv_pool
c128_kv_pool
c4_indexer_kv_pool
compress_state_pools
indexer_compress_state_pools
```

### 1. SWA / C4 / C128 KV 数据池

源码位置：`deepseek_v4_memory_pool.py:419-453`

```python
self.swa_kv_pool = DeepSeekV4SingleKVPool(...)
self.c4_kv_pool = DeepSeekV4SingleKVPool(...)
self.c128_kv_pool = DeepSeekV4SingleKVPool(...)
```

`DeepSeekV4SingleKVPool` 的实际分配在 `deepseek_v4_memory_pool.py:77-117`：

```python
self.kv_buffer = [
    self.create_buffer(
        num_pages=(self.size + self.page_size + 1) // self.page_size,
    )
    for _ in range(self.layer_num)
]
```

`create_buffer()` 内部：

```python
return torch.zeros(
    num_pages,
    self.bytes_per_page_padded,
    dtype=self.store_dtype,
    device=self.device,
)
```

DeepSeek V4 这里使用 page-packed 布局：

```text
shape = [num_pages, bytes_per_page_padded]
dtype = uint8
```

每个 token 的压缩布局由源码中的断言描述：

```python
assert bytes_per_token == 448 + 64 * 2 + 8
```

也就是：

```text
448 bytes: qk_nope_head_dim FP8
128 bytes: qk_rope_head_dim BF16, 64 * 2
8 bytes: FP8 scale 和 padding
合计 584 bytes/token
```

然后按 page 对齐到 `bytes_per_page_padded`。

### 2. DeepSeek V4 indexer KV 池

源码位置：`deepseek_v4_memory_pool.py:455-463`

```python
self.c4_indexer_kv_pool = DeepSeekV4IndexerPool(...)
```

实际分配在 `deepseek_v4_memory_pool.py:272-290`：

```python
self.index_k_with_scale_buffer = [
    torch.zeros(
        (self.size + self.page_size + 1) // self.page_size,
        page_bytes,
        dtype=self.index_k_with_scale_buffer_dtype,
        device=self.device,
    )
    for _ in range(self.layer_num)
]
```

这里保存 indexer 用的 K 和 scale，也是 `torch.zeros(..., device="musa")`。

### 3. DeepSeek V4 compress state 池

源码位置：`deepseek_v4_memory_pool.py:535-572`

```python
compress_state_pool = CompressStatePool(...)
indexer_compress_state_pool = CompressStatePool(...)
```

实际分配在 `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py:67-79`：

```python
self.kv_score_buffer = KVAndScore(
    torch.empty(
        (self._size, last_dim),
        dtype=dtype,
        device=device,
    )
)
```

这部分也是 PyTorch Tensor 分配。不同点是使用 `torch.empty`，不会像 `torch.zeros` 一样初始化整块内存为 0。

## ReqToTokenPool 分配

`ReqToTokenPool` 不是 KV 数据本体，但它是运行时必须使用的映射表。

源码位置：`python/sglang/srt/mem_cache/memory_pool.py:129-153`

```python
self.req_to_token = torch.zeros(
    (self._alloc_size, max_context_len),
    dtype=torch.int32,
    device=device,
)
```

含义：

- 行表示 request slot。
- 列表示该 request 的 token 位置。
- 内容是 token 对应的 KV cache 下标。

它也通过 PyTorch Tensor 分配，不直接调用底层 MUSA API。

## SGLang allocator 只管理下标

源码位置：`python/sglang/srt/mem_cache/allocator.py`

### TokenToKVPoolAllocator

源码位置：`allocator.py:121-157`

```python
self.free_pages = torch.arange(
    1, self.size + 1, dtype=torch.int64, device=self.device
)

select_index = self.free_pages[:need_size]
self.free_pages = self.free_pages[need_size:]
return select_index
```

这里的 `select_index` 是 token 位置。例如返回 `[1, 2, 3]`，表示新 token 写入 KV cache 的第 1、2、3 个槽位。

### PagedTokenToKVPoolAllocator

源码位置：`allocator.py:362-521`

```python
out_pages = self.free_pages[:num_pages]
self.free_pages = self.free_pages[num_pages:]

out_indices = (
    out_pages[:, None] * self.page_size
    + torch.arange(self.page_size, device=self.device)
).reshape(-1)
```

这里先分配 page 编号，再展开成 token 编号。例如：

```text
page_size = 4
out_pages = [2, 3]
out_indices = [8, 9, 10, 11, 12, 13, 14, 15]
```

它管理的是已经存在的 KV cache Tensor 内部位置，不是新的显存块。

## MUSA store cache 路径

DeepSeek V4 MUSA 后端有专门的 fused store cache 实现。

入口：`python/sglang/jit_kernel/deepseek_v4.py:728-748`

```python
def fused_store_cache(input, cache, indices, *, page_size, type):
    if input.device.type == "musa":
        from sglang.srt.hardware_backend.layers.deepseek_v4_musa_ops import (
            fused_store_cache_musa,
        )
        fused_store_cache_musa(input, cache, indices, page_size=page_size, type=type)
        return
```

MUSA 实现：`python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/ops/cache_ops.py:1170-1195`

```python
def fused_store_cache_musa(input, cache, indices, *, page_size, type):
    ...
```

这条路径的作用是把当前 forward 产生的 K/scale/rope 等数据写入已经分配好的 `cache` Tensor。

它不是显存申请路径。参数 `cache` 已经是 `DeepSeekV4SingleKVPool` 或 `DeepSeekV4IndexerPool` 中预先创建好的 Tensor。

## custom mem pool 路径

源码位置：`python/sglang/srt/mem_cache/utils.py:332-358`

```python
enable_custom_mem_pool = (
    True if envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get() is not None else False
)

if enable_custom_mem_pool:
    return init_mooncake_custom_mem_pool(device)
else:
    return False, None, None
```

KV cache 分配点会判断是否启用 custom pool：

```python
with (
    torch.cuda.use_mem_pool(self.custom_mem_pool)
    if self.custom_mem_pool
    else nullcontext()
):
    torch.zeros(..., device=self.device)
```

结论：

- 默认不启用 custom mem pool。
- 启用后仍然是 PyTorch Tensor 分配，只是切换到指定 PyTorch MemPool。
- 这不是 KV cache 代码直接调用 `musaMalloc`。

需要注意：当前 `utils/common.py` 中 `is_musa()` 定义了两次，后一个 `torchada` 版本覆盖前一个 `torch_musa` monkey patch 版本。是否能在 MUSA 环境下稳定使用 `torch.cuda.use_mem_pool(...)`，需要在实际运行环境中验证。

## 直接底层 API 搜索结果

在 KV cache 相关路径中，没有发现 `musaMalloc` 或 `muMemAlloc` 用于申请 KV cache 主体显存。

仓库中能搜到的直接 `musaMalloc` 位于：

```text
python/sglang/jit_kernel/include/sgl_kernel/distributed/custom_all_reduce.cuh:103
```

该文件属于 custom all-reduce 通信 workspace，不属于 KV cache 分配路径。

## 端到端流程

### 普通 MHA 模型

```text
ModelRunner.init_memory_pool()
  -> _resolve_memory_pool_config()
     -> get_available_gpu_memory(device="musa")
     -> DefaultPoolConfigurator.calculate_pool_sizes()
  -> _apply_memory_pool_config()
  -> _init_pools()
     -> ReqToTokenPool(...)
        -> torch.zeros(req_to_token, device="musa")
     -> MHATokenToKVPool(...)
        -> torch.zeros(k_buffer[layer], device="musa")
        -> torch.zeros(v_buffer[layer], device="musa")
     -> TokenToKVPoolAllocator 或 PagedTokenToKVPoolAllocator
        -> torch.arange/free_pages 管理可用下标
```

### 普通 MLA 模型

```text
ModelRunner.init_memory_pool()
  -> _init_pools()
     -> ReqToTokenPool(...)
     -> MLATokenToKVPool(...)
        -> torch.zeros(kv_buffer[layer], device="musa")
     -> TokenToKVPoolAllocator 或 PagedTokenToKVPoolAllocator
```

### DeepSeek V4

```text
ModelRunner.init_memory_pool()
  -> _init_pools()
     -> DeepSeekV4TokenToKVPool(...)
        -> DeepSeekV4SingleKVPool(swa)
           -> torch.zeros(kv_buffer[layer], device="musa")
        -> DeepSeekV4SingleKVPool(c4)
           -> torch.zeros(kv_buffer[layer], device="musa")
        -> DeepSeekV4SingleKVPool(c128)
           -> torch.zeros(kv_buffer[layer], device="musa")
        -> DeepSeekV4IndexerPool(c4_indexer)
           -> torch.zeros(index_k_with_scale_buffer[layer], device="musa")
        -> CompressStatePool(...)
           -> torch.empty(kv_score_buffer, device="musa")
     -> PagedTokenToKVPoolAllocator / SWA allocator
        -> 管理 page/token 下标
```

## 判断标准

### 可以确定的结论

- SGLang KV cache 主体显存通过 `torch.zeros` / `torch.empty` 创建 Tensor。
- MUSA 设备下，Tensor 的 `device` 是 `"musa"`。
- SGLang KV allocator 管理的是 Tensor 内部下标，不是底层显存指针。
- DeepSeek V4 MUSA fused store cache 是写入预分配 Tensor 的 kernel 路径，不是显存申请路径。
- 当前源码没有在 KV cache 路径中直接调用 `musaMalloc` 或 `muMemAlloc`。

### 需要在 torch_musa/MUSA Runtime 侧继续确认的内容

从 SGLang 源码无法直接证明 torch_musa 内部最终使用的是 caching allocator、stream-ordered allocator，还是直接 runtime allocation。

如果要继续下钻，需要在 torch_musa 或 MUSA Runtime 侧追踪：

```text
torch.zeros(device="musa")
  -> aten empty/zero
  -> torch_musa allocator
  -> MUSA Runtime memory API
  -> driver memory allocation
```

建议插桩点：

- torch_musa Tensor allocator 的 `allocate` / `raw_alloc` 等入口。
- MUSA Runtime 的 `musaMalloc`、`musaMallocAsync`、`muMemAlloc`、`muMemAllocAsync`。
- driver UMD 的 memory create / map / free 路径。

## 性能分析关注点

KV cache 初始化阶段的主要成本来自大 Tensor 分配和初始化：

- `torch.zeros` 会申请显存并清零，初始化成本可能明显高于 `torch.empty`。
- MHA 会按层创建 K/V 两组 Tensor。
- MLA 会按层创建单组 latent KV Tensor。
- DeepSeek V4 会创建 SWA、C4、C128、indexer、compress state 多类 Tensor，初始化路径更分散。

运行阶段的主要成本不是显存申请，而是：

- allocator 分配 token/page 下标；
- attention 计算读取 KV cache；
- fused store cache 把当前 token 的 K/V 或压缩表示写入预分配 cache；
- prefix cache、radix cache、SWA、压缩状态带来的下标转换和复用逻辑。

## 最终结论

MUSA SGLang 的 KV cache 不是直接在 SGLang 层调用底层 MUSA API 申请显存。

它使用 PyTorch Tensor API 创建 `"musa"` 设备上的 Tensor。默认情况下，这会进入 PyTorch/torch_musa 的默认内存分配路径。SGLang 自己实现的 KV allocator 只负责管理预分配 Tensor 内部的 token/page 位置。DeepSeek V4 的 MUSA 自定义 kernel 负责写入 KV cache，不负责申请 KV cache 显存。
