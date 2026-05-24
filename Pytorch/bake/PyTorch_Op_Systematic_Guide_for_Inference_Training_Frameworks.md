# PyTorch Op 系统技术洞察：功能、用法、注意事项与推理/训练框架实践

## 背景

本文面向大模型推理/训练框架中的 PyTorch op 使用，结合 SGLang DeepSeek V4/MUSA 适配中实际出现的 op，系统梳理每类 op 的功能、用法、注意事项、上层框架中的典型角色，以及相关算子。重点不是复述 API，而是解释这些 op 为什么会出现在模型执行、CUDA Graph、KV cache、MoE、量化和 metadata 构造路径中。

## 总览

| 类别 | 代表 op | 框架中的主要作用 |
| --- | --- | --- |
| Tensor 创建 | `empty`、`zeros`、`ones`、`full`、`new_empty` | 预分配 graph buffer、KV cache、临时输出 |
| 复制与原地更新 | `copy_`、`_foreach_copy_`、`fill_`、`zero_` | CUDA Graph replay 前灌入真实 batch |
| 形状变换 | `view`、`reshape`、`flatten`、`unsqueeze`、`squeeze` | 在 attention/MoE/HC/logits 间重排逻辑维度 |
| 内存布局 | `contiguous`、`stride`、`is_contiguous` | 满足 fused kernel、TileLang、通信算子要求 |
| 索引与切分 | slice、advanced indexing、`gather`、`tensor_split`、`where` | page table、SWA window、TP/DP/CP 分片 |
| 序列构造 | `arange`、`repeat_interleave`、`expand` | position、causal seq lens、batch index 重复 |
| 填充与拼接 | `pad`、`cat`、`stack` | CUDA Graph 固定形状、all-gather 后拼接 |
| 数学归约 | `sum`、`mean`、`amax`、`abs`、`square`、`rsqrt` | norm、scale、量化统计、checksum |
| 激活与限制 | `sigmoid`、`clamp`、`relu`、SwiGLU 组合 | gating、量化防溢出、MoE FFN |
| 线性代数 | `F.linear`、`einsum`、`matmul` | projection、HC head、fallback GEMM |
| TopK/路由 | `topk`、`sort`、`argmax` | MoE expert routing、sparse attention 选块 |
| dtype/device | `.to`、`.float`、`.bfloat16`、`.cpu` | 混合精度、metadata dtype、host/device 搬运 |
| 标量化 | `.item`、`.tolist` | Python 控制流、日志、host planner |

## Tensor 创建类

### `torch.empty`

功能：分配指定 shape/dtype/device 的 tensor，但不初始化内容。

用法：

```python
out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=device)
```

注意事项：

- 内容是未定义值，只有后续 kernel 会完整写入时才安全。
- 在 CUDA Graph capture/replay 中，`empty` 生成的 tensor 地址会被 graph 记录；capture 后 replay 不能随意换对象地址。
- 推理框架常用它预分配 logits buffer、KV cache page、temporary partial buffer，以减少 allocator 开销。

相关 op：

- `new_empty`：继承已有 tensor 的 dtype/device。
- `empty_like`：继承 shape/dtype/device。
- `zeros`：需要清零时使用，但有额外写内存成本。

### `torch.zeros` / `torch.ones` / `torch.full`

功能：分配并初始化 tensor。

用法：

```python
seq_lens = torch.full((max_bs,), fill_value=1, dtype=torch.int32, device=device)
mask = torch.ones((batch, seqlen), dtype=torch.bool, device=device)
out_loc = torch.zeros((num_tokens,), dtype=torch.int64, device=device)
```

注意事项：

- 初始化会产生额外 kernel 或 memset。
- CUDA Graph padding 场景常用 `zeros/full` 构造固定 shape 的 dummy 输入。
- `full` 的 `fill_value` 如果来自 GPU tensor 的 `.item()`，会触发 GPU 到 CPU 同步。

框架实践：

- 推理 decode graph 中，固定 `seq_lens`、`positions`、`out_cache_loc` buffer。
- idle batch 或 padding batch 中，用 `ones/zeros` 构造合法 dummy 请求，避免 graph 分支变化。

## 复制与原地更新类

### `copy_`

功能：把源 tensor 数据复制到目标 tensor，目标对象地址不变。

用法：

```python
graph_input[:raw_bs].copy_(real_input)
metadata_buffer.copy_(new_metadata)
```

注意事项：

- dtype/device/shape 需要兼容。
- GPU 到 GPU copy 是异步入队；CPU/GPU copy 是否异步取决于 pinned memory、`non_blocking` 和 stream。
- 对 CUDA Graph 很关键：replay 前通过 `copy_` 更新内容，同时保持 graph 捕获时的 tensor 地址。

框架实践：

- SGLang `DecodeInputBuffers.populate_from_forward_batch` 用 `copy_` 把真实 batch 写入 capture buffer。
- DSV4 metadata replay 用 `copy_` 更新 page table、seq_lens、topk metadata。

相关 op：

- `torch._foreach_copy_`：批量复制多个 tensor，减少 Python 循环和调度开销。
- `fill_`、`zero_`：原地填充值。
- `masked_fill_`：按 mask 原地写入。

### `torch._foreach_copy_`

功能：批量执行多个 `copy_`。

注意事项：

- 通常需要按 dtype/device 分组。
- 属于较底层优化 API，不如普通 `copy_` 通用。
- 适合 graph replay 前批量更新 input buffer。

## 形状变换类

### `view`

功能：在不复制数据的前提下改变 tensor 视图形状。

用法：

```python
x = x.view(num_tokens, num_heads, head_dim)
```

注意事项：

- 要求原 tensor stride 与目标 shape 兼容，通常要求 contiguous。
- 不改变底层存储。
- 如果失败，不应盲目改成 `reshape`，因为 `reshape` 可能隐式复制，影响性能和 graph 地址。

相关 op：

- `reshape`：可返回 view，也可复制。
- `flatten`：合并连续维度。
- `unflatten`：拆分维度。

### `reshape`

功能：改变形状，必要时复制。

注意事项：

- 更灵活，但可能产生新 tensor。
- 在性能敏感路径中，应确认是否发生拷贝。
- 推理框架中用于量化前把 `[T, G, D]` 合并为 `[T*G, D]`。

### `flatten`

功能：把一段维度合并。

用法：

```python
x_flat = x.flatten(1)  # [T, hc_mult, H] -> [T, hc_mult*H]
```

框架实践：

- HC/MHC pre 里把多路 residual 合成 GEMM 输入。
- logits 或 hidden states 进入 linear 层前规整维度。

### `unsqueeze` / `squeeze`

功能：增加或删除 size=1 的维度。

用法：

```python
weight = weight.unsqueeze(1)
post = post.squeeze(-1)
```

注意事项：

- 通常是 view，不复制。
- 常用于 broadcasting。
- `squeeze()` 不指定维度可能误删 batch 维，建议指定 `dim`。

## 内存布局类

### `contiguous`

功能：返回内存连续的 tensor；如果已连续则返回自身，否则复制。

用法：

```python
x = x.tensor_split(tp_size)[rank].contiguous()
```

注意事项：

- 可能发生真实拷贝。
- fused kernel、TileLang、custom op 通常要求 contiguous，以便使用线性地址计算和 vectorized load/store。
- 在 graph replay 中，隐式 contiguous 分配可能增加 allocator 压力。

相关属性：

- `tensor.stride()`
- `tensor.is_contiguous()`
- `tensor.storage_offset()`

框架实践：

- TP/CP 切分后调用 `contiguous`，保证后续 MoE/GEMM 输入 layout 合法。
- MUSA backend 明确检查 contiguous，避免 kernel 误读 stride。

## 索引与切分类

### slice / advanced indexing

功能：按范围或 index tensor 取子集。

用法：

```python
page_table = req_to_token[req_pool_indices, :max_seq_len:page_size]
local = hidden_states[start:end]
```

注意事项：

- 普通 slice 多数是 view。
- advanced indexing 通常生成新 tensor。
- index tensor dtype 通常要求 `int64`，但 custom kernel 更偏好 `int32`，常见 `.to(torch.int32)`。

框架实践：

- KV page table 构造。
- batch/TP/CP/DP token 分片。
- request pool 到 token pool 的映射。

### `torch.gather`

功能：按 index 从指定维度收集数据。

用法：

```python
physical_pages = torch.gather(page_table, dim=1, index=page_idx.long())
```

注意事项：

- `index` 通常必须是 `int64`。
- 输出 shape 与 index shape 一致。
- 大量随机 gather 访存不连续，容易带宽受限。

相关 op：

- `index_select`
- `scatter_`
- `take_along_dim`

### `tensor_split`

功能：把 tensor 按维度切成多个块。

用法：

```python
chunks = hidden_states.tensor_split(tp_size)
local = chunks[rank].contiguous()
```

框架实践：

- TP attention/MoE scatter。
- expert/token 分组。

注意事项：

- 返回的块可能不是 contiguous。
- 后续 collective 或 custom kernel 前常需要 `contiguous`。

### `masked_fill_`

功能：按 bool mask 原地填充值。

用法：

```python
offsets.masked_fill_(invalid_mask, 0)
raw_indices.masked_fill_(invalid_mask, -1)
```

框架实践：

- attention mask。
- SWA window 非法位置处理。
- topk padding 位置填 `-1`。

注意事项：

- 原地修改会影响共享底层存储的其他 view。
- mask shape 需可 broadcast。

## 序列构造类

### `torch.arange`

功能：构造等差序列。

用法：

```python
positions = torch.arange(max_seq_len, device=device)
torch.arange(start, end, out=out)
```

注意事项：

- 指定 `device`，避免先在 CPU 创建再搬到 GPU。
- `out=` 可复用预分配 buffer，减少 allocator 开销。
- CUDA Graph 中 shape 必须固定。

框架实践：

- position ids。
- causal seq_lens。
- page/window offsets。

### `repeat_interleave`

功能：按元素重复。

用法：

```python
idx_to_req = torch.arange(bs, device=device).repeat_interleave(qo_len)
```

框架实践：

- prefill 中把 request id 展开到 token 级。
- MoE routing 中构造 token/expert 映射。

注意事项：

- 可能分配新 tensor。
- 重复次数动态时不利于 graph capture。

### `expand`

功能：通过 stride=0 的 view 扩展维度，不复制数据。

用法：

```python
batch_indices = torch.arange(bs, device=device).unsqueeze(1).expand(-1, topk)
```

注意事项：

- 返回 view，不能随意原地写。
- 后续若要求 contiguous，可能需要复制。

## 拼接与填充类

### `torch.cat`

功能：沿指定维度拼接多个 tensor。

用法：

```python
hidden_states = torch.cat(gathered, dim=0)
```

注意事项：

- 必然分配新 tensor。
- all-gather 后常用 cat 合并各 rank 结果。
- 热路径中频繁 cat 可能造成 allocator 和带宽压力。

相关 op：

- `stack`：新增维度后拼接。
- `concat`：`cat` 别名。

### `torch.stack`

功能：新增一个维度后拼接。

用法：

```python
x = torch.stack([a, b], dim=0)
```

注意事项：

- 所有输入 shape 必须一致。
- 常用于构造小型 metadata 或 batch 维。

### `torch.nn.functional.pad`

功能：对 tensor 边界填充。

用法：

```python
padded = F.pad(out_cache_loc, pad=(0, bs - raw_bs), value=0)
```

框架实践：

- CUDA Graph replay 把真实 batch padding 到 capture batch size。
- attention page indices 对齐到 64/128 等硬件友好长度。

注意事项：

- 通常分配新 tensor。
- `value` 必须是 Python scalar；若通过 GPU tensor `.item()` 获得，会同步。

## 数学归约类

### `sum` / `mean`

功能：沿维度求和/均值。

用法：

```python
mean_sq = x.square().mean(-1, keepdim=True)
y = weighted.sum(dim=1)
```

框架实践：

- RMSNorm 统计。
- HC/MHC residual 混合。
- 量化误差和 debug checksum。

注意事项：

- 大维度归约容易成为独立 kernel。
- 多个小归约可融合到 custom kernel。
- `.sum().item()` 会把结果搬到 CPU 并同步。

### `square` / `rsqrt`

功能：

- `square`：逐元素平方。
- `rsqrt`：计算 `1 / sqrt(x)`。

用法：

```python
scale = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
```

框架实践：

- RMSNorm、HC pre norm。
- fused norm kernel 的 PyTorch fallback。

注意事项：

- 单独执行时会产生多个 kernel launch。
- 训练中要考虑数值稳定性和 eps。

### `abs` / `amax`

功能：绝对值与最大值归约。

用法：

```python
scale = x.float().reshape(M, -1, group).abs().amax(dim=-1) / 448.0
```

框架实践：

- FP8/int8 per-token/per-group quant scale。

注意事项：

- scale 通常需要 clamp 下限，避免除零或过小 scale。
- 量化 scale layout 与硬件 kernel 强相关。

## 激活与限制类

### `sigmoid`

功能：计算 `1 / (1 + exp(-x))`。

框架实践：

- gating。
- SwiGLU 中 `silu(x) = x * sigmoid(x)`。
- HC head 混合权重。

注意事项：

- 单独 sigmoid + multiply 可融合。
- 大模型推理中通常用 fused activation kernel。

### `clamp`

功能：限制数值上下界。

用法：

```python
x = torch.clamp(x, min=-limit, max=limit)
seq_lens = torch.clamp(seq_lens, max=SWA_WINDOW)
```

框架实践：

- SwiGLU clamp。
- attention topk length 裁剪。
- quant scale 下限。

注意事项：

- in-place `clamp_` 会修改原 tensor。
- clamp 常与 quant/activation 融合。

### `relu`

功能：负数置零。

框架实践：

- 一些 fallback attention score 或 MQA logits reference 实现。

相关 op：

- `gelu`
- `silu`
- `softmax`

## 线性代数类

### `torch.nn.functional.linear`

功能：执行 `y = x @ weight.T + bias`。

用法：

```python
mixes = F.linear(x_flat, hc_fn)
```

框架实践：

- projection fallback。
- HC head / MHC pre fallback。
- 小型 reference implementation。

注意事项：

- weight shape 是 `[out_features, in_features]`。
- 对小 batch 大 hidden，GEMM 可能需要 split-K 或专用 kernel。
- 推理中常被 Tensor Parallel Linear、quantized linear、DeepGEMM、CUTLASS/TileLang kernel 替代。

相关 op：

- `matmul`
- `mm`
- `bmm`
- `einsum`
- `nn.Linear`

### `torch.einsum`

功能：用 Einstein notation 表达张量乘法/归约。

用法：

```python
o = torch.einsum("tgd,grd->tgr", o, wo_a)
```

框架实践：

- grouped projection。
- 小规模或表达复杂的 reference path。

注意事项：

- 表达力强，但性能不一定等于最优 GEMM。
- 生产路径常替换为 batched GEMM、custom kernel、DeepGEMM。

## TopK 与路由类

### `torch.topk`

功能：返回前 k 大/小的值和索引。

用法：

```python
values, indices = torch.topk(scores, k=topk, dim=-1)
```

框架实践：

- MoE expert routing。
- sparse attention 选 top-k page/token。
- speculative decoding 候选 token。

注意事项：

- `sorted=False` 可减少排序开销。
- 对固定 topk 和固定 shape，custom kernel 通常更快。
- topk 输出索引用于 gather/scatter，需要注意 dtype 和 padding。

相关 op：

- `sort`
- `argsort`
- `argmax`
- custom radix topk / hash topk。

### `torch.where`

功能：按条件选择两个 tensor/scalar。

用法：

```python
raw_indices = torch.where(valid_mask, raw_indices, torch.tensor(-1, device=device))
```

注意事项：

- 条件和分支需可 broadcast。
- `torch.tensor(-1, device=device)` 在热路径中会分配小 tensor，最好预创建或使用 kernel 融合。

框架实践：

- topk padding。
- mask-based index 修正。
- routing fallback。

## dtype 与 device 转换类

### `.to`

功能：转换 dtype 或 device。

用法：

```python
seq_lens = seq_lens.to(torch.int32)
x = x.to(device, non_blocking=True)
```

注意事项：

- dtype 转换会产生新 tensor，除非 dtype 已相同。
- device 转换可能触发 H2D/D2H copy。
- `non_blocking=True` 只有在 pinned memory 或合适 stream 条件下才真正异步。

框架实践：

- Python metadata 转 GPU tensor。
- page table/indices 转 `int32` 给 custom kernel。
- FP32/BF16/FP8 混合精度转换。

### `.float()` / `.bfloat16()`

功能：转换到 FP32/BF16。

框架实践：

- norm/scale 统计用 FP32。
- hidden/output 保持 BF16。
- FP8 dequant 后常转 BF16。

注意事项：

- 转换是实拷贝。
- FP32 提升精度但增加带宽。

### `.cpu()`

功能：把 tensor 拷贝到 CPU。

注意事项：

- 对 GPU tensor 调用 `.cpu()` 通常需要等待相关 GPU 工作完成。
- 推理热路径中应避免。
- 如果必须 host planning，优先使用小 metadata、pinned memory、异步 copy 和明确 stream 同步点。

框架实践：

- correctness dump。
- tokenizer/output 后处理。
- host-side compress plan。
- profiler/debug。

## 标量化类

### `.item()`

功能：把单元素 tensor 转为 Python scalar。

用法：

```python
max_seq_len = seq_lens_cpu.max().item()
```

注意事项：

- GPU tensor 上调用会同步。
- CPU mirror tensor 上调用通常安全。
- 不应在 decode 每步热路径对 GPU tensor 调用。

框架实践：

- 选择 CUDA Graph bucket。
- 构造 Python 控制流参数。
- 生成日志或断言。

### `.tolist()`

功能：把 tensor 转为 Python list。

注意事项：

- GPU tensor 上调用会 D2H 同步。
- list 适合 Python loop、host planner、配置，不适合 GPU 热路径。

框架实践：

- `seq_lens_cpu.tolist()` 传给 CP metadata 或 prefill planner。
- 输出 token id 后处理。

## 上层推理框架中的组合模式

### CUDA Graph replay 模式

典型组合：

```python
fixed_input[:n].copy_(real_input)
fixed_seq_lens[:bs].copy_(real_seq_lens)
metadata.copy_(new_metadata)
graph.replay()
```

关键原则：

- graph 捕获后 tensor 地址必须稳定。
- replay 前只更新内容，不替换对象。
- 避免 replay 热路径中 allocator、`.item()`、`.cpu()`。
- padding 到 capture batch size，保证 shape 固定。

相关 op：

- `empty/zeros/full`
- `copy_/_foreach_copy_`
- `pad`
- `slice`
- `fill_/zero_`

### KV Cache / Page Table 模式

典型组合：

```python
page_table = req_to_token[req_pool_indices, :max_seq_len:page_size]
page_table = (page_table // page_size).to(torch.int32)
```

关键原则：

- page table 是小 metadata，但会驱动大 attention kernel。
- dtype 通常转 `int32`，减少带宽并匹配 custom kernel。
- padding value 要避免 GPU scalar `.item()`。

相关 op：

- advanced indexing
- `arange`
- `masked_fill_`
- `clamp`
- `to(torch.int32)`

### MoE Routing / TopK 模式

典型组合：

```python
weights, ids = torch.topk(router_logits, k=topk, dim=-1, sorted=False)
```

推理框架中常替换为：

- fused topk
- hash topk
- radix topk
- fused routing + dispatch

关键原则：

- topk 后通常紧接 gather/scatter、all-to-all 或 grouped GEMM。
- padding token 的 expert id 应被 mask 掉。
- routing indices dtype/layout 直接影响 MoE kernel。

### Quantization 模式

典型组合：

```python
scale = x.float().reshape(M, -1, group).abs().amax(dim=-1).clamp(min=1e-4) / max_val
q = quantize(x, scale)
```

关键原则：

- scale 计算必须数值稳定。
- scale layout 与硬件 kernel 强绑定。
- FP8 E4M3、UE8M0、transposed scale、swizzled layout 不能混用。

相关 op：

- `float`
- `reshape`
- `abs`
- `amax`
- `clamp`
- custom quant kernel

### HC/MHC 模式

典型 PyTorch fallback：

```python
x_flat = x.flatten(1).float()
rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + eps)
mixes = F.linear(x_flat, weight) * rsqrt
pre = torch.sigmoid(mixes * scale + base) + eps
y = (pre.unsqueeze(-1) * x.view(shape)).sum(dim=1)
```

关键原则：

- 语义正确但 kernel 碎。
- 小 batch 大 hidden 时 split-K/fused kernel 更重要。
- MUSA 适配应优先下沉到 TileLang/native op，而不是依赖 torch fallback。

## 训练框架中的差异

训练比推理多出 autograd、optimizer、activation checkpoint、梯度通信等约束：

- `copy_`、`masked_fill_`、`zero_` 等原地 op 可能破坏 autograd 版本计数，需要确认是否在 `torch.no_grad()` 或 inference-only 路径。
- `view` 产生的 view 与原 tensor 共享存储，原地修改可能影响 backward。
- `contiguous` 和 `reshape` 的隐式 copy 会增加激活内存。
- `.item()` 会让动态图出现 Python 控制流，影响 `torch.compile` 和分布式同步。
- `topk`、`where`、`gather` 的 backward 可能稀疏且不连续，训练中更关注梯度正确性；推理中更关注延迟。

推理框架通常在 `@torch.no_grad()` 下运行，更敢于使用原地 op 和预分配 buffer；训练框架则要优先维护 autograd 正确性。

## 逐 Op 用法补充与 MUSA 验证

本节把上文出现的 op 按“功能、典型写法、注意事项、上层框架用法、相关算子”展开。验证脚本位于 `Insights/pytorch_op_musa_validation.py`，覆盖小 tensor correctness、CPU 对照、同步边界和基础 MUSA Graph capture。

### 输入输出示例速查

下面示例默认省略 `device=device`，在 MUSA 上只需把输入 tensor 放到 `musa:0`。`empty` 类 op 的内容未定义，因此只展示 shape/dtype。

#### 创建、复制与原地更新

| op | 输入 | 执行 | 结果 |
| --- | --- | --- | --- |
| `empty` | shape `(2, 3)` | `torch.empty((2, 3), dtype=torch.float32)` | shape 为 `(2, 3)`，内容未定义 |
| `new_empty` | `x.shape=(2, 3), x.dtype=float32` | `x.new_empty((3, 2))` | shape 为 `(3, 2)`，dtype/device 继承 `x` |
| `empty_like` | `x=[[1,2],[3,4]]` | `torch.empty_like(x)` | shape/dtype/device 与 `x` 相同，内容未定义 |
| `zeros` | shape `(2, 3)` | `torch.zeros((2, 3))` | `[[0,0,0],[0,0,0]]` |
| `ones` | shape `(2, 3)` | `torch.ones((2, 3))` | `[[1,1,1],[1,1,1]]` |
| `full` | shape `(2, 3)`, value `7` | `torch.full((2, 3), 7)` | `[[7,7,7],[7,7,7]]` |
| `copy_` | `dst=[0,0,0]`, `src=[1,2,3]` | `dst.copy_(src)` | `dst=[1,2,3]`，对象地址不变 |
| `_foreach_copy_` | `dst0=[0,0]`, `dst1=[0,0]`, `src0=[1,1]`, `src1=[2,2]` | `torch._foreach_copy_([dst0,dst1],[src0,src1])` | `dst0=[1,1]`, `dst1=[2,2]` |
| `fill_` | `x=[0,0,0]` | `x.fill_(3)` | `x=[3,3,3]` |
| `zero_` | `x=[3,3,3]` | `x.zero_()` | `x=[0,0,0]` |
| `masked_fill_` | `x=[1,2,3,4]`, `mask=[T,F,T,F]` | `x.masked_fill_(mask, -1)` | `x=[-1,2,-1,4]` |

#### 形状、布局与广播

| op | 输入 | 执行 | 结果 |
| --- | --- | --- | --- |
| `view` | `x=torch.arange(6)` | `x.view(2, 3)` | `[[0,1,2],[3,4,5]]`，通常不复制 |
| `reshape` | `x=torch.arange(6)` | `x.reshape(3, 2)` | `[[0,1],[2,3],[4,5]]`，必要时复制 |
| `flatten` | `x.shape=(2,3,4)` | `x.flatten(1)` | shape `(2, 12)` |
| `unsqueeze` | `x.shape=(3,)` | `x.unsqueeze(0)` | shape `(1, 3)` |
| `squeeze` | `x.shape=(1,3,1)` | `x.squeeze(-1)` | shape `(1, 3)` |
| `expand` | `x=[[0],[1],[2]]` | `x.expand(-1, 4)` | `[[0,0,0,0],[1,1,1,1],[2,2,2,2]]`，stride=0 view |
| `contiguous` | `x=torch.arange(6).view(2,3).t()` | `x.contiguous()` | 数值不变，layout 变为连续 |
| `stride` | `x=torch.arange(6).view(2,3)` | `x.stride()` | `(3, 1)` |
| `is_contiguous` | `x.t()` | `x.t().is_contiguous()` | `False` |
| `storage_offset` | `x=torch.arange(6)[2:]` | `x.storage_offset()` | `2` |

#### 索引、切分与映射

| op | 输入 | 执行 | 结果 |
| --- | --- | --- | --- |
| slice | `x=[[0,1,2],[3,4,5],[6,7,8]]` | `x[1:, :2]` | `[[3,4],[6,7]]` |
| advanced indexing | `x=[[0,1,2],[3,4,5],[6,7,8]]`, `rows=[0,2]`, `cols=[1,2]` | `x[rows, cols]` | `[1,8]` |
| `gather` | `x=[[10,11,12],[20,21,22]]`, `idx=[[2,0],[1,1]]` | `torch.gather(x, 1, idx)` | `[[12,10],[21,21]]` |
| `take_along_dim` | 同上 | `torch.take_along_dim(x, idx, dim=1)` | `[[12,10],[21,21]]` |
| `index_select` | `x=[[0,1],[2,3],[4,5]]`, `idx=[2,0]` | `torch.index_select(x, 0, idx)` | `[[4,5],[0,1]]` |
| `scatter_` | `out=zeros(2,3)`, `idx=[[0,2],[0,1]]`, `src=[[5,6],[7,8]]` | `out.scatter_(1, idx, src)` | `[[5,0,6],[7,8,0]]` |
| `tensor_split` | `x=[0,1,2,3,4,5]` | `x.tensor_split(3)` | `[tensor([0,1]), tensor([2,3]), tensor([4,5])]` |

#### 序列、拼接、填充与条件选择

| op | 输入 | 执行 | 结果 |
| --- | --- | --- | --- |
| `arange` | start `3`, end `7` | `torch.arange(3, 7)` | `[3,4,5,6]` |
| `arange(out=...)` | `out=torch.empty(4)` | `torch.arange(3, 7, out=out)` | `out=[3,4,5,6]`，复用 `out` |
| `repeat_interleave` | `x=[0,1,2]`, repeat `2` | `x.repeat_interleave(2)` | `[0,0,1,1,2,2]` |
| `repeat_interleave` 动态次数 | `x=[0,1,2]`, repeats `[1,2,1]` | `torch.repeat_interleave(x, repeats)` | `[0,1,1,2]` |
| `cat` | `a=[1,2]`, `b=[3,4]` | `torch.cat([a,b], dim=0)` | `[1,2,3,4]` |
| `stack` | `a=[1,2]`, `b=[3,4]` | `torch.stack([a,b], dim=0)` | `[[1,2],[3,4]]` |
| `F.pad` | `x=[[1,2,3]]` | `F.pad(x, (1,2), value=0)` | `[[0,1,2,3,0,0]]` |
| `where` | `cond=[T,F,T]`, `a=[1,1,1]`, `b=[2,2,2]` | `torch.where(cond, a, b)` | `[1,2,1]` |

#### 数学、归约与激活

| op | 输入 | 执行 | 结果 |
| --- | --- | --- | --- |
| `sum` | `x=[[1,2,3],[4,5,6]]` | `x.sum(dim=1)` | `[6,15]` |
| `mean` | `x=[[1,2,3],[4,5,6]]` | `x.mean(dim=0)` | `[2.5,3.5,4.5]` |
| `amax` | `x=[[-1,3],[5,2]]` | `x.amax(dim=1)` | `[3,5]` |
| `min` / `max` | `x=[[3,1],[2,5]]` | `x.min(dim=1).values`, `x.max(dim=1).values` | `[1,2]`, `[3,5]` |
| `abs` | `x=[-2,0,3]` | `x.abs()` | `[2,0,3]` |
| `square` | `x=[-2,3]` | `x.square()` | `[4,9]` |
| `rsqrt` | `x=[4,16]` | `torch.rsqrt(x)` | `[0.5,0.25]` |
| `sigmoid` | `x=[0]` | `torch.sigmoid(x)` | `[0.5]` |
| `silu` | `x=[0,1]` | `F.silu(x)` | `[0,0.7311]` 近似 |
| `gelu` | `x=[0,1]` | `F.gelu(x)` | `[0,0.8413]` 近似 |
| `relu` | `x=[-1,0,2]` | `F.relu(x)` | `[0,0,2]` |
| `softmax` | `x=[1,2,3]` | `F.softmax(x, dim=0)` | `[0.0900,0.2447,0.6652]` 近似 |
| `clamp` | `x=[-2,0,3]` | `torch.clamp(x, min=-1, max=1)` | `[-1,0,1]` |

#### 线性代数、排序与路由

| op | 输入 | 执行 | 结果 |
| --- | --- | --- | --- |
| `F.linear` | `x=[[1,2]]`, `weight=[[1,0],[0,1],[1,1]]`, `bias=[0,0,1]` | `F.linear(x, weight, bias)` | `[[1,2,4]]` |
| `matmul` | `a.shape=(2,3)`, `b.shape=(3,4)` | `torch.matmul(a,b)` | shape `(2,4)` |
| `mm` | `a=[[1,2]]`, `b=[[3],[4]]` | `torch.mm(a,b)` | `[[11]]` |
| `bmm` | `a.shape=(2,3,4)`, `b.shape=(2,4,5)` | `torch.bmm(a,b)` | shape `(2,3,5)` |
| `einsum` | `a.shape=(2,3,4)`, `b.shape=(2,4,5)` | `torch.einsum("bik,bkj->bij", a, b)` | shape `(2,3,5)`，等价 batched matmul |
| `topk` | `x=[0.1,3.0,2.0,-1.0]` | `torch.topk(x, k=2)` | values `[3.0,2.0]`, indices `[1,2]` |
| `sort` | `x=[3,1,2]` | `torch.sort(x)` | values `[1,2,3]`, indices `[1,2,0]` |
| `argsort` | `x=[3,1,2]` | `torch.argsort(x)` | `[1,2,0]` |
| `argmax` | `x=[3,1,5,2]` | `torch.argmax(x)` | `2` |

#### dtype、CPU 边界与同步

| op/API | 输入 | 执行 | 结果 |
| --- | --- | --- | --- |
| `.to(dtype)` | `x=[1.2,2.8] float32` | `x.to(torch.int32)` | `[1,2] int32` |
| `.to(device)` | `x` on CPU | `x.to("musa:0")` | 数值相同，device 变为 MUSA |
| `.float()` | `x=[1,2] int32` | `x.float()` | `[1.0,2.0] float32` |
| `.bfloat16()` | `x=[1.0,2.0] float32` | `x.bfloat16()` | BF16 tensor，数值近似相同 |
| `.cpu()` | `x` on MUSA | `x.cpu()` | 数值相同，device 变为 CPU，会形成同步边界 |
| `.numpy()` | CPU tensor `x=[1,2]` | `x.numpy()` | NumPy array `[1,2]`，只适用于 CPU tensor |
| `.item()` | `x=tensor([7])` | `x.item()` | Python scalar `7`，device tensor 上会同步 |
| `.tolist()` | `x=tensor([1,2,3])` | `x.tolist()` | Python list `[1,2,3]`，device tensor 上会同步 |
| `synchronize` | device 队列已有 kernel | `torch.musa.synchronize()` | CPU 等待 MUSA 队列完成 |
| `MUSAGraph.replay` | capture 中记录 `out = inp * 2 + 1`，replay 前 `inp.fill_(3)` | `graph.replay()` | `out` 变为全 `7` |

### 创建与初始化

#### `empty` / `new_empty` / `empty_like`

- 功能：只分配存储，不初始化内容。`new_empty` 继承输入 tensor 的 dtype/device，`empty_like` 继承 shape/dtype/device。
- 用法：`buf = torch.empty((bs, hidden), dtype=torch.bfloat16, device=device)`；`tmp = x.new_empty((n, k))`。
- 注意：读取未写满的 `empty` 是未定义行为；graph capture 后对象地址会进入 graph，不能 replay 时换 tensor。
- 框架用法：SGLang decode graph、KV cache 和 logits buffer 预分配，减少 allocator 干扰。
- 相关算子：`zeros`、`full`、`empty_strided`。

#### `zeros` / `ones` / `full`

- 功能：分配并初始化为 0、1 或指定标量。
- 用法：`torch.full((max_bs,), 1, dtype=torch.int32, device=device)`。
- 注意：初始化会多一次写内存；`fill_value` 应使用 Python scalar，避免从 GPU tensor `.item()` 产生同步。
- 框架用法：构造 graph padding、dummy request、seq_lens、attention mask 和 routing 默认值。
- 相关算子：`zero_`、`fill_`、`zeros_like`、`ones_like`、`full_like`。

### 原地复制与更新

#### `copy_`

- 功能：把源 tensor 内容复制到目标 tensor，目标对象和地址不变。
- 用法：`static_input[:n].copy_(real_input)`；`metadata_buf.copy_(metadata)`。
- 注意：shape/dtype/device 要兼容；GPU copy 通常异步入队，D2H/H2D 是否异步取决于 pinned memory 和 stream。
- 框架用法：CUDA/MUSA Graph replay 前把真实 batch 写入捕获时的静态 buffer。
- 相关算子：`to`、`clone`、`_foreach_copy_`、`copy.deepcopy` 不是 tensor device copy。

#### `_foreach_copy_`

- 功能：批量执行多组 `copy_`，降低 Python 循环和调度开销。
- 用法：`torch._foreach_copy_([dst0, dst1], [src0, src1])`。
- 注意：需要按 dtype/device 分组；不同后端支持程度可能不同，建议保留普通 `copy_` fallback。
- 框架用法：graph replay 前批量更新 page table、seq_lens、positions 等 metadata buffer。
- 相关算子：`_foreach_add_`、`_foreach_mul_`、`copy_`。

#### `fill_` / `zero_` / `masked_fill_`

- 功能：原地填充值。`zero_` 是填 0 快捷形式，`masked_fill_` 只更新 mask 为 true 的位置。
- 用法：`buf.zero_()`；`ids.masked_fill_(invalid, -1)`。
- 注意：原地 op 会影响共享 storage 的 view；训练中可能触发 autograd version counter 错误。
- 框架用法：清空临时 metadata、处理 padding token、屏蔽非法 expert/page id。
- 相关算子：`where`、`scatter_`、`index_fill_`。

### 形状与布局

#### `view` / `reshape` / `flatten`

- 功能：改变逻辑 shape。`view` 必须 stride 兼容，`reshape` 必要时会复制，`flatten` 合并连续维。
- 用法：`q = q.view(T, H, D)`；`x = x.reshape(M, -1, group)`；`x_flat = x.flatten(1)`。
- 注意：热路径不要用 `reshape` 掩盖隐式 copy；需要稳定地址时应显式 `contiguous()` 后再 `view`。
- 框架用法：attention QKV head 维度整理、MoE token/group 展平、量化 scale 分组。
- 相关算子：`unflatten`、`permute`、`transpose`、`as_strided`。

#### `unsqueeze` / `squeeze` / `expand`

- 功能：增加、删除 size=1 维度，或用 stride=0 做 broadcast view。
- 用法：`w = w.unsqueeze(1)`；`idx = torch.arange(bs, device=device).unsqueeze(1).expand(-1, topk)`。
- 注意：`squeeze()` 不指定 dim 可能误删 batch 维；`expand` 返回 view，不适合原地写。
- 框架用法：broadcast scale、构造 `[bs, topk]` request index、RMSNorm keepdim 结果广播。
- 相关算子：`repeat`、`repeat_interleave`、`broadcast_to`。

#### `contiguous` / `stride` / `is_contiguous` / `storage_offset`

- 功能：检查或转换内存布局。`contiguous` 在必要时复制为连续存储。
- 用法：`local = hidden.tensor_split(tp_size)[rank].contiguous()`。
- 注意：`contiguous` 是潜在分配点；custom kernel 要明确要求 NCHW/NHWC 或行主序布局。
- 框架用法：TP/CP 切分后喂给 fused kernel、TileLang、通信和 grouped GEMM。
- 相关算子：`permute`、`transpose`、`channels_last`、`clone`。

### 索引、切分与映射

#### slice / advanced indexing

- 功能：按范围或 index tensor 取子集。普通 slice 多为 view，advanced indexing 通常分配新 tensor。
- 用法：`page = req_to_token[req_idx, :max_len:page_size]`；`local = x[start:end]`。
- 注意：index tensor 常要求 `int64`，但 custom kernel 往往偏好 `int32`，需要显式转换。
- 框架用法：KV page table、request/token 映射、TP/DP/CP 分片。
- 相关算子：`index_select`、`gather`、`take_along_dim`。

#### `gather` / `take_along_dim` / `index_select`

- 功能：按 index 从指定维取数。`gather` 输出 shape 与 index 一致，`index_select` 只沿单维选择。
- 用法：`y = torch.gather(x, dim=1, index=idx.long())`；`rows = torch.index_select(x, 0, req_idx)`。
- 注意：随机 gather 访存不连续，容易带宽受限；index dtype 和 device 必须正确。
- 框架用法：根据 routing ids 取 expert 权重、根据 page id 取 KV block、根据 token ids 取 hidden/logits。
- 相关算子：`scatter_`、`embedding`、`take`。

#### `scatter_`

- 功能：按 index 把 src 写到目标 tensor 指定位置。
- 用法：`out.scatter_(dim=1, index=idx, src=values)`。
- 注意：重复 index 的写入顺序可能不确定；训练中重复 scatter 的梯度语义要谨慎。
- 框架用法：MoE dispatch/combine、routing mask 回填、token 重新排序。
- 相关算子：`scatter_add_`、`index_put_`、`index_add_`。

#### `tensor_split`

- 功能：沿指定维度切分 tensor，返回多个 view 或小块。
- 用法：`chunks = x.tensor_split(tp_size, dim=0)`。
- 注意：块大小可能不均匀；返回结果后续不一定满足 custom kernel 的 contiguous 要求。
- 框架用法：Tensor Parallel、Context Parallel、expert/token 分块。
- 相关算子：`chunk`、`split`、`narrow`。

### 序列、拼接与条件选择

#### `arange`

- 功能：生成等差序列，常用于 position、offset 和 index。
- 用法：`pos = torch.arange(start, end, device=device, dtype=torch.int32)`；`torch.arange(3, 7, out=buf)`。
- 注意：指定 device，避免 CPU 创建后搬运；`out=` 可复用 buffer，但 shape 要匹配。
- 框架用法：position ids、block offsets、batch index、page offsets。
- 相关算子：`linspace`、`range` 已弃用、`cumsum`。

#### `repeat_interleave`

- 功能：按元素重复，支持固定次数或每个元素不同重复次数。
- 用法：`req_ids = torch.arange(bs, device=device).repeat_interleave(qo_lens)`。
- 注意：动态重复次数会产生动态 shape，不利于 graph capture；结果通常是新 tensor。
- 框架用法：prefill request id 展开到 token 级、MoE token/expert 映射。
- 相关算子：`repeat`、`expand`、`tile`。

#### `cat` / `stack` / `F.pad`

- 功能：`cat` 沿已有维拼接，`stack` 新增一维再拼接，`pad` 在边界填充。
- 用法：`x = torch.cat(parts, dim=0)`；`x = torch.stack([a, b], dim=0)`；`x = F.pad(ids, (0, pad_n), value=0)`。
- 注意：三者通常都会分配新 tensor；热路径频繁使用会引入 allocator 和带宽压力。
- 框架用法：all-gather 后合并 hidden、graph batch padding、metadata 对齐到固定 bucket。
- 相关算子：`concat`、`vstack`、`hstack`。

#### `where`

- 功能：按 bool 条件在两个输入间逐元素选择。
- 用法：`ids = torch.where(valid, ids, torch.full_like(ids, -1))`。
- 注意：分支要可 broadcast；热路径避免临时 `torch.tensor(-1, device=device)` 小分配。
- 框架用法：修正非法 topk/padding id、mask attention score、选择 fallback routing。
- 相关算子：`masked_fill`、`select_scatter`、`clamp`。

### 数学、归约与激活

#### `sum` / `mean` / `amax` / `min` / `max` / `abs` / `square` / `rsqrt`

- 功能：归约、绝对值、平方和倒平方根。
- 用法：`scale = x.float().reshape(M, -1, G).abs().amax(-1).clamp(min=eps)`；`rstd = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)`。
- 注意：小 op 链会产生多个 kernel launch；`.sum().item()` 会同步到 CPU。
- 框架用法：RMSNorm fallback、FP8/int8 scale 统计、debug checksum、量化误差分析。
- 相关算子：`var_mean`、`norm`、`std`、`reciprocal`、`sqrt`。

#### `sigmoid` / `silu` / `gelu` / `relu` / `softmax` / `clamp`

- 功能：常见激活、归一化概率和数值裁剪。`silu(x)=x*sigmoid(x)`，`softmax` 常用于概率分布。
- 用法：`gate = F.silu(x)`；`prob = F.softmax(logits, dim=-1)`；`x = torch.clamp(x, min=-limit, max=limit)`。
- 注意：推理热路径通常融合激活和乘法；`softmax` 要注意 dtype 精度和 mask 后的 `-inf`。
- 框架用法：SwiGLU FFN、MoE router logits、采样概率、量化 scale 下限、attention reference。
- 相关算子：`log_softmax`、`sigmoid_`、`clamp_`、`tanh`。

### 线性代数与路由

#### `F.linear` / `matmul` / `mm` / `bmm` / `einsum`

- 功能：矩阵乘。`F.linear(x, w, b)` 等价于 `x @ w.T + b`；`mm` 是 2D GEMM，`bmm` 是 batch GEMM，`einsum` 用下标表达复杂乘法/归约。
- 用法：`y = F.linear(x, weight, bias)`；`y = torch.bmm(q, k.transpose(1, 2))`；`y = torch.einsum("tgd,grd->tgr", x, w)`。
- 注意：weight layout、dtype、transpose 和 contiguous 直接影响 GEMM 性能；`einsum` 语义清晰但未必是最高性能实现。
- 框架用法：QKV/O projection、HC/MHC fallback、MoE grouped projection、attention score reference。
- 相关算子：`addmm`、`baddbmm`、`scaled_mm`、DeepGEMM、quantized linear。

#### `topk` / `sort` / `argsort` / `argmax`

- 功能：排序、选择前 k 或最大值索引。
- 用法：`vals, ids = torch.topk(scores, k=topk, dim=-1, sorted=False)`；`order = torch.argsort(scores, dim=-1)`。
- 注意：`sorted=False` 通常更快；相等值的稳定性和 tie-break 不应作为跨后端一致性假设。
- 框架用法：MoE expert routing、sparse attention page 选择、采样候选、debug reference。
- 相关算子：custom fused topk、hash topk、radix select、`multinomial`。

### dtype、CPU 边界与同步

#### `.to` / `.float()` / `.bfloat16()`

- 功能：转换 dtype 或 device，快捷方法是 `.float()`、`.bfloat16()`。
- 用法：`idx = idx.to(torch.int32)`；`x = x.float()`；`x = x.to(device, non_blocking=True)`。
- 注意：dtype 变化会复制；H2D/D2H copy 的 `non_blocking=True` 需要 pinned memory 才有意义。
- 框架用法：metadata 转 int32 给 custom kernel、norm/scale 升 FP32、hidden/output 保持 BF16。
- 相关算子：`type_as`、`half`、`long`、`pin_memory`。

#### `.cpu()` / `.numpy()` / `.item()` / `.tolist()`

- 功能：把 device tensor 搬到 CPU，或转换为 Python/NumPy 对象。
- 用法：`lens = seq_lens_cpu.tolist()`；`max_len = int(seq_lens_cpu.max().item())`。
- 注意：对 GPU/MUSA tensor 使用会触发同步；`.numpy()` 只能用于 CPU tensor，通常先 `.detach().cpu()`。
- 框架用法：host planner、日志、bucket 选择、输出 token 后处理。热路径应使用 CPU mirror metadata，避免每步 D2H。
- 相关算子：`detach`、`clone`、`record_stream`。

#### `synchronize` / `CUDAGraph` / `MUSAGraph`

- 功能：`synchronize` 等待 device 队列完成；graph API 捕获一段固定形状、固定地址的 device 工作并重复 replay。
- 用法：`torch.musa.synchronize()`；`g = torch.musa.MUSAGraph(); g.capture_begin(); ...; g.capture_end(); g.replay()`。
- 注意：MUSA 验证中发现 graph capture 必须在非默认 stream 上进行；capture 后 replay 可在默认 stream。capture 内避免 allocator、CPU 同步、动态 shape 和不支持的 op。
- 框架用法：SGLang decode 阶段固定 batch bucket replay；`copy_` 更新静态输入，`graph.replay()` 执行 attention、MLP、采样前的稳定子图。
- 相关算子/API：`Stream`、`current_stream`、`wait_stream`、`graph_pool_handle`、CUDA Graph。

### MUSA 验证结果

验证时间：2026-05-21。环境：远端 `shanfeng@10.18.32.25`，Docker 实例 `mochi-sglang`，SGLang 虚拟环境 `/root/.virtualenvs/sglang-0.5.6`，`torch=2.9.0`，`device=musa:0`。

执行命令：

```bash
ssh shanfeng@10.18.32.25 "docker cp /tmp/pytorch_op_musa_validation.py mochi-sglang:/tmp/pytorch_op_musa_validation.py && docker exec mochi-sglang bash -lc 'RUN_LD=/workspace/scripts/musa_toolkits_5.1.0/lib:/workspace/scripts/mccl/lib:/data/xiyin/pkgs/1015-daily/mudnn/lib:/usr/local/musa/lib:/usr/lib/x86_64-linux-gnu; source /root/.virtualenvs/sglang-0.5.6/bin/activate; cd /sgl-workspace/sglang; env MUSA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=\$RUN_LD python /tmp/pytorch_op_musa_validation.py'"
```

结果摘要：

```text
torch=2.9.0 device=musa:0 count=1
PASS empty/new_empty/empty_like
PASS zeros/ones/full
PASS copy_/foreach_copy_/fill_/zero_
PASS view/reshape/flatten/unsqueeze/squeeze
PASS contiguous/stride/is_contiguous/storage_offset
PASS slice/advanced_indexing/gather/index_select/scatter/take_along_dim
PASS tensor_split/masked_fill_/where
PASS arange/repeat_interleave/expand
PASS pad/cat/stack
PASS sum/mean/amax/abs/square/rsqrt
PASS sigmoid/clamp/relu/silu/gelu/softmax
PASS F.linear/matmul/mm/bmm/einsum
PASS topk/sort/argsort/argmax/min/max
PASS to/float/bfloat16/cpu/item/tolist
PASS synchronize/cpu/numpy
PASS cudagraph_basic
SUMMARY passed=16 total=16
```

针对上方“输入输出示例速查”的逐例验证脚本为 `Insights/pytorch_op_musa_examples_validation.py`。该脚本把表格中的输入、执行语句和期望输出逐条写成断言，并在同一 MUSA 环境中通过：

```text
torch=2.9.0 device=musa:0 count=1
PASS create_copy_update_examples
PASS shape_layout_broadcast_examples
PASS index_split_mapping_examples
PASS sequence_join_condition_examples
PASS math_reduction_activation_examples
PASS linalg_sort_route_examples
PASS dtype_cpu_sync_graph_examples
SUMMARY passed=7 total=7
```

验证洞察：

1. 常见 PyTorch shape/index/math/linear/topk op 在当前 torch_musa 环境下小规模 correctness 通过。
2. `.cpu()`、`.item()`、`.tolist()` 能正常工作，但它们是明确同步边界，不能放在 decode 热路径。
3. MUSA Graph 与 CUDA Graph 的核心约束一致：固定地址、固定 shape、capture 内避免 CPU op；额外注意 capture 需在非默认 stream 上进行。
4. 该脚本验证的是功能正确性，不代表生产 shape、并发 stream、allocator 压力和 fused kernel 性能已经充分验证。

## 实践检查清单

1. `empty` 后是否保证完整写入。
2. `reshape` 是否可能隐式复制，是否应改成 `view + contiguous` 显式表达。
3. 热路径是否存在 GPU tensor 的 `.item()`、`.tolist()`、`.cpu()`。
4. custom kernel 输入是否满足 dtype、device、contiguous、stride 要求。
5. CUDA Graph replay 是否只 `copy_` 内容而不替换 tensor 对象。
6. padding 是否引入 GPU scalar 到 Python scalar 的同步。
7. quant scale layout 是否与 kernel 预期一致。
8. fallback torch op 是否只用于 debug/correctness，而非生产热路径。
9. training 路径是否避免不安全原地 op。
10. MUSA/CUDA/HIP/NPU backend 是否有各自 graph-safe 的等价算子。

## 结论

PyTorch op 在上层推理/训练框架中承担的是“表达语义”和“组织数据”的角色；真正高性能路径通常会把热点组合下沉为 fused kernel、Triton、TileLang、DeepGEMM、CUTLASS 或硬件后端 native op。判断一个 op 是否有风险，不应只看 API 名称，而要看它是否分配新 tensor、是否改变内存布局、是否触发 host-device 同步、是否破坏 CUDA Graph 地址稳定、是否兼容 autograd。

在 DeepSeek V4/MUSA 适配中，普通 shape op 不是主要瓶颈；关键风险集中在 `F.linear/sum/sigmoid/rsqrt` 这类碎计算组合、`topk/gather/scatter` 这类 routing/indexing 组合、FP8 quant/cache store layout，以及 `.item/.tolist/.cpu` 引入的同步。成熟框架的优化方向是：保留 PyTorch op 作为清晰 fallback，同时为热路径建立 graph-safe、layout-stable、backend-native 的融合算子。
