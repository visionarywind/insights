# DeepSeek V4-Pro 与 SGLang 实践：源码、Graph 与 MUSA 验证

> 本篇来自原始长文，保留 DeepSeek V4-Pro 和 SGLang 源码分析、Graph 详细用例、MUSA 最小验证输出和回顾总结。OP 基础与 Transformer 主线分别见第 1 篇和第 2 篇。

## 1. DeepSeek V4-Pro 源码中的 OP 组织

DeepSeek V4-Pro 在 Transformer OP 基础上叠加了更多约束：量化 Linear 引入 FP8/FP4、activation scale 和 packed weight；Compressed Attention 引入 sparse index、compressed KV 和 `topk/where/cat`；MoE 引入 router、top-k expert、dispatch/combine。

HC/MHC 用 reference 计算序列表达可验证语义。

正文只保留必要片段，完整源码摘录放在附录 B。

阅读 DeepSeek V4-Pro 源码时，先按模块识别对应的执行行为，再读具体代码。量化 Linear 主要对应 GEMM 路径选择，Attention 主要对应 layout、sparse index 和 KV cache，MoE 主要对应动态路由和多个 expert 计算，Graph 主要对应固定地址 replay。

```mermaid
flowchart TD
    A["DeepSeek V4-Pro forward"] --> B["量化 Linear"]
    A --> C["Compressed Attention"]
    A --> D["MoE"]
    A --> E["HC / MHC reference"]
    A --> F["Graph replay 边界"]

    B --> B1["关注: dtype, packed weight, scale layout, GEMM kernel"]
    C --> C1["关注: Q/KV layout, RoPE, sparse index, KV cache 写入"]
    D --> D1["关注: top-k, dispatch, grouped GEMM, combine, CPU 回读"]
    E --> E1["关注: 多个逐元素 OP, reduction, 中间 tensor"]
    F --> F1["关注: 固定 shape, 固定地址, stream/event, capture 兼容性"]

    B1 --> G["Profiler / 后端执行证据"]
    C1 --> G
    D1 --> G
    E1 --> G
    F1 --> G
```

### 1.1 量化 Linear：dtype 分派、activation scale 与 packed weight

DeepSeek V4-Pro 的 `linear` 根据权重 dtype 选择执行路径：FP4/FP8 权重先对 activation 做 `act_quant`，再进入量化 GEMM；普通权重才走 PyTorch reference。

```python
def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    # 该实现只覆盖无 bias 路径，量化 GEMM 接口不接收 bias。
    assert bias is None

    # FP4/FP8 权重先量化 activation，再调用对应 GEMM wrapper。
    if weight.dtype == torch.float4_e2m1fn_x2:
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp4_gemm(x, s, weight, weight.scale, scale_dtype)
    elif weight.dtype == torch.float8_e4m3fn:
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp8_gemm(x, s, weight, weight.scale, scale_dtype)
    else:
        # 普通 dtype 使用 PyTorch dense linear 作为 reference/fallback。
        return F.linear(x, weight)
```

| 位置 | 主要 OP | 组织方式 | 注意事项 |
|------|---------|----------|----------|
| 参数创建 | `torch.empty` | 创建 packed FP4/FP8 weight 和 scale tensor | `empty` 后必须由权重加载完整写入 |
| activation 量化 | `act_quant`、`to(dtype)` | BF16/FP32 activation -> quant activation + scale | block size、scale dtype 和 layout 是接口约束 |
| GEMM | `fp4_gemm/fp8_gemm` | quant activation + quant weight + scale -> output | packed layout、scale shape 和输出 dtype 要一致 |
| reference | `F.linear` | 普通 dtype 或 fallback 路径 | 可以验证语义，但热点路径要确认是否进入预期执行路径 |

### 1.2 Compressed Attention：layout、RoPE、top-k index 与 sparse kernel

Attention 模块先生成 Q/KV，再做 head layout、RoPE、Q norm、KV quant 和 sparse/compressed index，最后调用 sparse attention kernel。这些 PyTorch OP 主要用于组织输入，而不是单独完成完整 attention 计算。

```python
# Q 路径：projection 后整理成 head layout，并只对 RoPE 维旋转。
q = self.wq_b(qr)
q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
apply_rotary_emb(q[..., -rd:], freqs_cis)
q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)

# KV 路径：norm、RoPE 和非 RoPE 维原地量化。
kv = self.wkv(x)
kv = self.kv_norm(kv)
apply_rotary_emb(kv[..., -rd:], freqs_cis)
act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)

# sparse attention index 由窗口 index 和压缩 index 拼成。
topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
if self.compress_ratio:
    compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
    topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
topk_idxs = topk_idxs.int()

# attention kernel 直接消费 Q、KV、sink 和 sparse index。
o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
```

| 模块 | 主要 OP | 组织方式 | 注意事项 |
|------|---------|----------|----------|
| Q layout | `unflatten/view` | projection 输出 -> `[batch, seq, heads, head_dim]` | stride 不兼容会影响后续 kernel |
| Q norm | `square`、`mean`、`rsqrt`、`mul_` | 对 Q 做 RMS-like normalization | 多个小 OP 可用于 reference，热点路径关注融合 |
| KV 路径 | `kv_norm`、`act_quant(..., inplace=True)` | KV norm 后对非 RoPE 维量化 | 原地量化要确认后续不需要原 BF16 值 |
| sparse index | `topk`、`where`、`cat`、`int` | window index + compressed index -> sparse index | `k`、sentinel `-1`、index dtype 要与 kernel 一致 |
| attention kernel | `sparse_attn` | 读取 Q/KV/topk index 输出 attention | layout、padding head、index 越界和 graph capture 都会影响稳定性 |

### 1.3 MoE：Router、Expert Dispatch 与 Combine

MoE 模块先用 gate 产生 `weights` 和 `indices`，再按 expert id 分发 token。reference 写法便于观察 OP 语义：`topk/gather` 生成固定 top-k expert，`where/indexing` 找到属于某个 expert 的 token，`sum/+=` 完成 combine。

```python
# gate 输出每个 token 的 top-k expert 权重和 expert id。
weights, indices = self.gate(x, input_ids.flatten())

# 使用 FP32 输出 buffer 累加各 expert 的结果。
y = torch.zeros_like(x, dtype=torch.float32)

# reference 路径把 expert 计数回读到 CPU，用于跳过空 expert。
counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
for i in range(self.experts_start_idx, self.experts_end_idx):
    if counts[i] == 0:
        continue
    # 找出路由到当前 expert 的 token，并按对应 top-k 权重执行 expert。
    idx, top = torch.where(indices == i)
    y[idx] += self.experts[i](x[idx], weights[idx, top, None])

# shared expert 输出与 routed expert 输出合并。
y += self.shared_experts(x)
```

| 阶段 | 主要 OP | 组织方式 | 注意事项 |
|------|---------|----------|----------|
| Router | `linear`、`softmax/sigmoid/softplus` | hidden -> expert score | score 函数和归一化要匹配模型配置 |
| Top-k | `topk`、`gather` | expert score -> top-k id/weight | top-k shape 固定，便于后续 kernel 和 metadata |
| 统计 | `bincount(...).tolist()` | 统计每个 expert token 数 | GPU tensor 回 CPU 会同步；适合 reference，不适合热点路径 |
| Dispatch | `where`、advanced indexing | 按 expert id 选 token | 输出 token 数动态，会影响 Graph 和 workspace |
| Combine | `+=`、broadcast、`sum` | expert output 按权重回写 token | 重复 index、dtype 和累加顺序会影响数值 |

### 1.4 HC/MHC：Reference 计算序列如何表达数学语义

HC/MHC 模块使用 PyTorch 基础 OP 表达 reference 语义：先 flatten hidden，转成 float，做 RMS-like normalization，再用 `F.linear` 生成 mixing 参数，最后通过 broadcast 和 `sum` 合成输出。

该 reference 实现适合语义验证，也会产生多次 kernel launch 和中间 tensor 读写。

```python
# 合并 HC copy 维后的 hidden 用于计算 mixing 参数。
x = x.flatten(2).float()

# RMS-like scale 控制 mixing 参数幅度。
rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
mixes = F.linear(x, hc_fn) * rsqrt

# split/sinkhorn 生成 pre、post 和 comb 三组 HC 参数。
pre, post, comb = hc_split_sinkhorn(
    mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
)

# pre 权重作用在 HC copy 维上，sum 后回到普通 hidden。
y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
```

| 阶段 | 主要 OP | 组织方式 | 注意事项 |
|------|---------|----------|----------|
| 展平 | `flatten`、`view` | 多维 hidden -> 线性输入 | `view` 需要 layout 兼容 |
| 归一化 | `float`、`square`、`mean`、`rsqrt` | 计算 per-token scale | cast 和 reduction 维度要明确 |
| mixing | `F.linear`、`mul` | hidden -> pre/post/comb 参数 | weight shape 和输出切分要匹配 |
| 合成 | `unsqueeze`、broadcast、`sum` | mixing 参数和 hidden 合成输出 | 多个 broadcast/归约 OP 会产生中间读写 |

### 1.5 权重转换与 Packed Layout

量化模型的运行时性能不只取决于 forward 里的 OP，也取决于加载阶段如何把 checkpoint 权重转换成运行时 layout。DeepSeek V4-Pro 的转换逻辑使用 `view(uint8)`、bit op、`stack`、`flatten`、`transpose` 等 OP 把 packed FP4 数据展开并重排。

```python
# packed FP4 先按 uint8 读取底层字节。
x = x.view(torch.uint8)

# 每个字节拆成 low/high 两个 nibble。
low = x & 0x0F
high = (x >> 4) & 0x0F

# 查表恢复两个 FP4 逻辑值，并展平成连续元素。
x = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(2)

# 重排成运行时 GEMM 需要的 block/tile layout。
x = x.view(bOut, fp8_block_size, bIn, fp8_block_size).transpose(1, 2)
```

| 主要 OP | 组织方式 | 注意事项 |
|---------|----------|----------|
| `view(torch.uint8)` | 按字节解释 packed 数据 | 只改变解释方式，不等于数值转换 |
| bit op | 拆 low/high nibble | 要明确 endian、pack 顺序和 table 映射 |
| `stack/flatten` | 把两个半字节恢复成连续元素 | shape 变化必须和 block size 对齐 |
| `transpose/view` | 转成运行时 GEMM 需要的 tile layout | stride 和 contiguous 影响加载后 kernel 输入 |

## 2. MUSA 上的模块最小用例与验证输出

| 场景 | 章节 | 关注的主要 OP |
|------|------|--------------|
| 验证 graph replay 或 buffer 复用行为 | §2.1 | `copy_`, `fill_`, `zero_`, `MUSAGraph.replay()` |
| 验证 prefill metadata（positions 展开、cache 起始位置） | §2.2 | `arange`, `tolist()`, `repeat_interleave` |
| 验证 decode graph bucket 固定地址 + padding 正确性 | §2.3 | `MUSAGraph.replay()`, `copy_`, padding |
| 调试 Paged KV cache 写入 / page slot 映射 | §2.4 | advanced indexing, `div`, `%` |
| 验证 MoE routing 权重与 expert combine 路径 | §2.5 | `topk`, `softmax`, `unsqueeze`, `sum` |
| 验证 sampling 后处理不把完整 logits 或大概率表拉回 CPU | §2.6 | `.cpu()`, `.tolist()`, `softmax` |

面对长代码示例时，先按要验证的行为选择小节。每个用例只回答一个问题：这段上层 OP 在 MUSA 上是否按预期产生固定地址、同步、copy、layout 或路由行为。

为减少阅读干扰，样例代码只保留输入构造、目标 OP 和必要输出，不再在每个代码块内重复放置统一格式化函数。运行结果保留为整理后的 `shape/dtype/device/value` 格式，便于核对。

```mermaid
flowchart TD
    A["要验证的行为"] --> B{"是否与 Graph 固定地址有关"}
    B -->|"是"| C["看 2.1 和 2.3: copy_, fill_, MUSAGraph.replay"]
    B -->|"否"| D{"是否与 metadata 构造有关"}
    D -->|"是"| E["看 2.2: positions, seq_lens, cache 起点"]
    D -->|"否"| F{"是否与 KV cache 写入有关"}
    F -->|"是"| G["看 2.4: slot 到 page / offset"]
    F -->|"否"| H{"是否与 MoE 或 sampling 有关"}
    H -->|"MoE"| I["看 2.5: top-k, expert 输出, combine"]
    H -->|"sampling"| J["看 2.6: top-k, softmax, 最终 token 回读"]
    H -->|"其他"| K["回到第 1/2 篇附录 A 查具体 OP 用例"]
```

### 2.1 主要 OP 序列的 MUSA 最小用例

本用例不依赖 SGLang 运行时对象，直接复现典型 OP 组合与执行边界：graph replay buffer 批量拷贝、metadata 原地 `copy_`、MUSA graph 固定地址 replay、HC pre/post reference 链，以及 SwiGLU clamp 路径。

```python
import torch
import torch.nn.functional as F
from dataclasses import dataclass


def grouped_foreach_copy(dsts, srcs):
    # 按 dtype 分组后批量 copy，模拟 replay 前多个固定 buffer 的集中更新。
    groups = {}
    for dst, src in zip(dsts, srcs):
        groups.setdefault((dst.dtype, src.dtype), ([], []))
        groups[(dst.dtype, src.dtype)][0].append(dst)
        groups[(dst.dtype, src.dtype)][1].append(src)
    for group_dsts, group_srcs in groups.values():
        torch._foreach_copy_(group_dsts, group_srcs)


@dataclass
class RawDecodeMetadata:
    # 只保留 decode replay 需要的三个 metadata tensor。
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor

    def copy_(self, other):
        # 原地复制字段内容，保持 capture 时绑定的 tensor 对象不变。
        self.req_pool_indices.copy_(other.req_pool_indices)
        self.seq_lens.copy_(other.seq_lens)
        self.out_cache_loc.copy_(other.out_cache_loc)


def hc_pre_reference(x, hc_fn, eps=1e-5):
    # HC pre reference：展平多 copy hidden，生成 pre/post/comb 参数。
    shape, dtype = x.size(), x.dtype
    x_flat = x.flatten(1).float()
    rstd = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + eps)
    mixes = (F.linear(x_flat, hc_fn) * rstd).unsqueeze(1)
    pre = torch.sigmoid(mixes[:, :, : shape[1]])
    post = torch.sigmoid(mixes[:, 0, shape[1] : 2 * shape[1]])
    comb_raw = mixes[:, 0, 2 * shape[1] : 2 * shape[1] + shape[1] * shape[1]]
    comb = comb_raw.view(shape[0], shape[1], shape[1]).softmax(dim=1)
    y = (pre.squeeze(1).unsqueeze(-1) * x_flat.view(shape)).sum(dim=1)
    return y.to(dtype), post.to(dtype), comb.to(dtype)


def hc_post_reference(x, residual, post, comb):
    # HC post reference：把当前输出和 residual 按 post/comb 合成回多 copy 状态。
    return (
        post.unsqueeze(-1) * x.unsqueeze(1)
        + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
    ).type_as(x)


device = torch.device("musa:0")
# 1. replay buffer：模拟 DecodeInputBuffers.populate_from_forward_batch。
input_ids = torch.zeros(4, dtype=torch.int64, device=device)
req_pool_indices = torch.zeros(2, dtype=torch.int64, device=device)
seq_lens = torch.empty(2, dtype=torch.int32, device=device)
out_cache_loc = torch.empty(4, dtype=torch.int64, device=device)
seq_lens.fill_(1)
out_cache_loc.zero_()
grouped_foreach_copy(
    [input_ids[:3], req_pool_indices[:2], seq_lens[:2], out_cache_loc[:3]],
    [
        torch.tensor([11, 12, 13], device=device),
        torch.tensor([7, 8], device=device),
        torch.tensor([5, 6], dtype=torch.int32, device=device),
        torch.tensor([100, 101, 102], device=device),
    ],
)

# 2. metadata copy_：模拟 DSV4RawDecodeMetadata.copy_。
captured = RawDecodeMetadata(
    req_pool_indices=torch.zeros(2, dtype=torch.int64, device=device),
    seq_lens=torch.zeros(2, dtype=torch.int32, device=device),
    out_cache_loc=torch.zeros(4, dtype=torch.int64, device=device),
)
temp = RawDecodeMetadata(req_pool_indices, seq_lens, out_cache_loc)
captured.copy_(temp)

# 3. graph replay：capture 后只改输入内容，不替换 inp/graph_out 对象。
inp = torch.ones((2, 2), device=device)
graph_out = torch.empty_like(inp)
stream = torch.musa.Stream()
stream.wait_stream(torch.musa.current_stream())
with torch.musa.stream(stream):
    for _ in range(3):
        graph_out.copy_(inp * 2 + 1)
torch.musa.current_stream().wait_stream(stream)
graph = torch.musa.MUSAGraph()
with torch.musa.stream(stream):
    graph.capture_begin()
    graph_out.copy_(inp * 2 + 1)
    graph.capture_end()
inp.fill_(4)
graph.replay()
torch.musa.synchronize()

# 4. HC pre/post fallback：验证 flatten/norm/linear/broadcast/sum 的 reference 语义。
x = torch.arange(12, dtype=torch.float32, device=device).reshape(2, 2, 3)
hc_fn = torch.arange(48, dtype=torch.float32, device=device).reshape(8, 6) / 10.0
hc_y, post, comb = hc_pre_reference(x, hc_fn)
hc_post = hc_post_reference(hc_y, x, post, comb)

# 5. SwiGLU clamp：复现 fused_moe.py 中 gate/up 限幅后的乘法路径。
swiglu_in = torch.tensor([[1.0, -2.0, 20.0, -20.0]], device=device)
gate, up = swiglu_in.chunk(2, dim=-1)
gate = F.silu(gate).clamp(max=10.0)
up = up.clamp(min=-10.0, max=10.0)
swiglu_out = gate * up

print("input_ids =", input_ids)
print("captured.req_pool_indices =", captured.req_pool_indices)
print("captured.seq_lens =", captured.seq_lens)
print("captured.out_cache_loc =", captured.out_cache_loc)
print("graph_out =", graph_out)
print("hc_y =", hc_y)
print("post =", post)
print("comb =", comb)
print("hc_post =", hc_post)
print("swiglu_out =", swiglu_out)
```

MUSA 运行结果（整理后）：

```text
input_ids shape=(4,), dtype=int64, device=musa:0, value=[11, 12, 13, 0]
captured.req_pool_indices shape=(2,), dtype=int64, device=musa:0, value=[7, 8]
captured.seq_lens shape=(2,), dtype=int32, device=musa:0, value=[5, 6]
captured.out_cache_loc shape=(4,), dtype=int64, device=musa:0, value=[100, 101, 102, 0]
graph_out shape=(2, 2), dtype=float32, device=musa:0, value=[[9.0, 9.0], [9.0, 9.0]]
hc_y shape=(2, 3), dtype=float32, device=musa:0, value=[[2.9752485752105713, 4.827154636383057, 6.679060459136963], [14.002138137817383, 15.838565826416016, 17.674991607666016]]
post shape=(2, 2), dtype=float32, device=musa:0, value=[[0.9995744824409485, 0.9999781847000122], [0.999838650226593, 0.999995231628418]]
comb shape=(2, 2, 2), dtype=float32, device=musa:0, value=[[[0.002611535834148526, 0.0026115409564226866], [0.9973884224891663, 0.9973884224891663]], [[0.0008589610224589705, 0.0008589610224589705], [0.9991409778594971, 0.9991409778594971]]]
hc_post shape=(2, 2, 3), dtype=float32, device=musa:0, value=[[[5.9661478996276855, 8.817265510559082, 11.668384552001953], [5.967349052429199, 8.819214820861816, 11.671079635620117]], [[22.99730110168457, 25.833431243896484, 28.66956329345703], [22.999492645263672, 25.835912704467773, 28.67232894897461]]]
swiglu_out shape=(1, 2), dtype=float32, device=musa:0, value=[[7.310585975646973, 2.3840584754943848]]
```

验证结论：该用例执行通过。输出显示 `captured.*` 字段通过 `copy_` 写入固定 metadata 对象，`graph_out` 在 replay 后从输入 `4` 得到 `9`，HC pre/post 链输出符合预期，SwiGLU clamp 路径输出 `[[7.310585975646973, 2.3840584754943848]]`。

### 2.2 Prefill Metadata 构造最小用例

Prefill 阶段会把 CPU scheduler 的 `seq_lens`、`extend_lens` 和 KV cache 起始位置转换成 DEVICE侧的 `positions`、`req_pool_indices`、`out_cache_loc`。涉及 OP 包括 CPU侧元数据副本、`.tolist()`、`arange`、`cat`、`repeat_interleave`、indexing 和加法。

```python
import torch


device = torch.device("musa:0")
# CPU 侧保存请求长度，scheduler 可直接读取这些标量。
seq_lens_cpu = torch.tensor([3, 2], dtype=torch.int32, device="cpu")
extend_lens_cpu = torch.tensor([2, 1], dtype=torch.int32, device="cpu")
start_pos_cpu = seq_lens_cpu - extend_lens_cpu

# 把每个请求的增量 token 展开成 DEVICE 侧 position 序列。
positions = torch.cat([
    torch.arange(int(s), int(s + e), dtype=torch.int64, device=device)
    for s, e in zip(start_pos_cpu.tolist(), extend_lens_cpu.tolist())
])

# 为每个增量 token 生成所属 request 的 index。
req_pool_indices = torch.repeat_interleave(
    torch.arange(2, dtype=torch.int64, device=device),
    extend_lens_cpu.to(device=device, dtype=torch.int64),
)

# 按 request 的 cache 起点和本地 offset 计算写入位置。
base_cache_loc = torch.tensor([10, 20], dtype=torch.int64, device=device)
local_offsets = torch.cat([
    torch.arange(int(e), dtype=torch.int64, device=device)
    for e in extend_lens_cpu.tolist()
])
out_cache_loc = base_cache_loc[req_pool_indices] + local_offsets

print("positions =", positions)
print("req_pool_indices =", req_pool_indices)
print("out_cache_loc =", out_cache_loc)
```

MUSA 运行结果（整理后）：

```text
positions shape=(3,), dtype=int64, device=musa:0, value=[1, 2, 1]
req_pool_indices shape=(3,), dtype=int64, device=musa:0, value=[0, 0, 1]
out_cache_loc shape=(3,), dtype=int64, device=musa:0, value=[10, 11, 20]
```

验证结论：该用例把两个请求的增量 token 展开成 3 个 prefill token。第 0 个请求写入 cache 位置 `10,11`，第 1 个请求写入 cache 位置 `20`。

### 2.3 Decode Graph Replay 最小用例

Decode 阶段每步只新增少量 token，适合把固定 batch bucket capture 成 graph。replay 前通过 `copy_` 更新 `input_ids` 和 `positions`，graph 内执行固定的 `stack -> matmul -> copy_` 路径。

```python
import torch


device = torch.device("musa:0")
# 固定 bucket 大小为 4，真实 batch 可小于该大小。
input_ids = torch.zeros(4, dtype=torch.float32, device=device)
positions = torch.zeros(4, dtype=torch.float32, device=device)
logits = torch.empty((4, 3), dtype=torch.float32, device=device)
weight = torch.tensor([[0.1, 0.2, 0.3], [1.0, 1.5, 2.0]], device=device)

# warmup 初始化 kernel/allocator 状态，降低 capture 时的额外干扰。
stream = torch.musa.Stream()
stream.wait_stream(torch.musa.current_stream())
with torch.musa.stream(stream):
    for _ in range(3):
        hidden = torch.stack([input_ids, positions], dim=1)
        logits.copy_(hidden @ weight)
torch.musa.current_stream().wait_stream(stream)

# capture 固定的 stack -> matmul -> copy_ 执行路径。
graph = torch.musa.MUSAGraph()
with torch.musa.stream(stream):
    graph.capture_begin()
    hidden = torch.stack([input_ids, positions], dim=1)
    logits.copy_(hidden @ weight)
    graph.capture_end()

# replay 前只更新固定 buffer 内容；后两个位置模拟 padding。
input_ids.copy_(torch.tensor([5, 6, 0, 0], dtype=torch.float32, device=device))
positions.copy_(torch.tensor([9, 10, 0, 0], dtype=torch.float32, device=device))
graph.replay()
torch.musa.synchronize()

print("input_ids =", input_ids)
print("positions =", positions)
print("logits =", logits)
```

MUSA 运行结果（整理后）：

```text
input_ids shape=(4,), dtype=float32, device=musa:0, value=[5.0, 6.0, 0.0, 0.0]
positions shape=(4,), dtype=float32, device=musa:0, value=[9.0, 10.0, 0.0, 0.0]
logits shape=(4, 3), dtype=float32, device=musa:0, value=[[9.5, 14.5, 19.5], [10.600000381469727, 16.200000762939453, 21.799999237060547], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
```

验证结论：graph capture 后没有替换 `input_ids`、`positions` 和 `logits` 对象，只更新内容。真实 batch 为 2，bucket 大小为 4，padding 位置输出保持 0。

### 2.4 KV Cache 写入最小用例

Paged KV cache 根据 `slot_mapping` 将新 token 的 K/V 写入 page 和 page offset。该用例用一个 K cache 展示 `div`、取模、advanced indexing 和原地写入。

```python
import torch


device = torch.device("musa:0")
# cache shape: [page, page_offset, kv_head, head_dim]。
kv_cache = torch.zeros((3, 2, 1, 2), dtype=torch.float32, device=device)
slot_mapping = torch.tensor([1, 4, 5], dtype=torch.int64, device=device)
page_size = 2

# slot id 拆成 page index 和 page 内 offset。
page_idx = torch.div(slot_mapping, page_size, rounding_mode="floor")
page_offset = slot_mapping % page_size

# 新 token 的 K 值按 page/offset 写入 cache。
new_k = torch.tensor([[[1.0, 1.5]], [[2.0, 2.5]], [[3.0, 3.5]]], device=device)
kv_cache[page_idx, page_offset] = new_k
selected = kv_cache[page_idx, page_offset]

print("page_idx =", page_idx)
print("page_offset =", page_offset)
print("selected =", selected)
print("kv_cache =", kv_cache)
```

MUSA 运行结果（整理后）：

```text
page_idx shape=(3,), dtype=int64, device=musa:0, value=[0, 2, 2]
page_offset shape=(3,), dtype=int64, device=musa:0, value=[1, 0, 1]
selected shape=(3, 1, 2), dtype=float32, device=musa:0, value=[[[1.0, 1.5]], [[2.0, 2.5]], [[3.0, 3.5]]]
kv_cache shape=(3, 2, 1, 2), dtype=float32, device=musa:0, value=[[[[0.0, 0.0]], [[1.0, 1.5]]], [[[0.0, 0.0]], [[0.0, 0.0]]], [[[2.0, 2.5]], [[3.0, 3.5]]]]
```

验证结论：`slot_mapping=[1,4,5]` 被拆成 page `[0,2,2]` 和 offset `[1,0,1]`，新 token 的 cache 写入后可按相同索引读回。

### 2.5 MoE 路由与 Combine 最小用例

场景：MoE 推理先用 router logits 选 top-k expert，再按路由权重合并 expert 输出。该用例用 `softmax`、`topk`、`unsqueeze`、broadcast multiply 和 `sum` 表达最小路由计算。

```python
import torch


device = torch.device("musa:0")
hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device)
router_logits = torch.tensor([[1.0, 3.0, 0.0], [2.0, 0.5, 1.5], [0.0, 1.0, 4.0]], device=device)

# router logits 转成概率后，为每个 token 选择两个 expert。
probs = torch.softmax(router_logits, dim=-1)
topk_vals, topk_ids = torch.topk(probs, k=2, dim=-1)

# 用 expert id 构造可验证的 expert 输出，再按 top-k 权重合并。
expert_scale = (topk_ids.to(torch.float32) + 1.0).unsqueeze(-1)
expert_out = hidden.unsqueeze(1) * expert_scale
combined = (expert_out * topk_vals.unsqueeze(-1)).sum(dim=1)

print("topk_ids =", topk_ids)
print("topk_vals =", topk_vals)
print("combined =", combined)
```

MUSA 运行结果（整理后）：

```text
topk_ids shape=(3, 2), dtype=int64, device=musa:0, value=[[1, 0], [0, 2], [2, 1]]
topk_vals shape=(3, 2), dtype=float32, device=musa:0, value=[[0.8437947034835815, 0.11419519037008286], [0.546549379825592, 0.3314989507198334], [0.9362395405769348, 0.04661262407898903]]
combined shape=(3, 2), dtype=float32, device=musa:0, value=[[1.801784634590149, 3.603569269180298], [4.623138427734375, 6.1641845703125], [14.509719848632812, 17.411663055419922]]
```

验证结论：该用例保留 MoE 上层语义：router 产生 expert id 和权重，expert 输出按 `topk_vals` 加权合并。SGLang DeepSeek V4 的在线推理热点路径应由 fused kernel 执行 expert GEMM、dispatch 和 combine。

### 2.6 Sampling 后处理最小用例

Logits 输出后执行 temperature、top-k、softmax 和 next token 选择。该用例保留 DEVICE侧的 top-k 和概率计算，只把最终 `next_token` 作为必要结果拷回 CPU。

```python
import torch


device = torch.device("musa:0")
logits = torch.tensor([[1.0, 3.0, 2.0, -1.0], [0.5, 0.0, 4.0, 1.0]], device=device)
temperature = 0.5

# temperature 缩放和 top-k 选择保留在 MUSA tensor 上。
scaled = logits / temperature
topk_vals, topk_ids = torch.topk(scaled, k=2, dim=-1)
probs = torch.softmax(topk_vals, dim=-1)
next_token = topk_ids[:, 0]

# CPU 只接收最终 token id。
next_token_cpu = next_token.cpu().tolist()

print("topk_ids =", topk_ids)
print("topk_vals =", topk_vals)
print("probs =", probs)
print("next_token_cpu", next_token_cpu)
```

MUSA 运行结果（整理后）：

```text
topk_ids shape=(2, 2), dtype=int64, device=musa:0, value=[[1, 2], [2, 3]]
topk_vals shape=(2, 2), dtype=float32, device=musa:0, value=[[6.0, 4.0], [8.0, 2.0]]
probs shape=(2, 2), dtype=float32, device=musa:0, value=[[0.8807970285415649, 0.11920291185379028], [0.9975274205207825, 0.0024726237170398235]]
next_token_cpu [1, 2]
```

验证结论：sampling 计算主要保留在 MUSA tensor 上，CPU-DEVICE 边界只发生在最终 token 回传处。在线服务应避免频繁将完整 logits 或大概率表 `.cpu()`。

### 2.7 Graph 深入用法与 MUSA 示例

本节展开 §1.5 的 Graph 生命周期，覆盖 SGLang DeepSeek V4 的 graph 用法、三类执行方式对比（普通 Graph / Piecewise Graph / torch.compile）以及 Graph 相关 OP 的使用约束，最后总结 DSV4 graph 的三个设计约束。

SGLang DeepSeek V4 的 graph 流程遵循“动态决策在 Graph 外，固定 tensor 计算在 Graph 内”：CPU scheduler 处理变长请求和 KV 管理，DEVICE侧 graph 只读取已经 padding/bucket 化的固定 buffer。

```mermaid
flowchart LR
    subgraph CPU_Control["Graph 外 CPU侧"]
        A["request queue, tokenizer, detokenizer"]
        B["scheduler 选择 prefill 或 decode"]
        C["选择 graph bucket: bs, seq, page"]
        D["KV block 和 page 分配"]
        E["构造 CPU侧元数据副本"]
        A --> B --> C --> D --> E
    end

    subgraph H2D_Prepare["Replay 前 固定 buffer 更新"]
        F["padding 与填充槽位"]
        G["copy_ input_ids 和 positions"]
        H["copy_ seq_lens 和 req_pool_indices"]
        I["copy_ page table 和 out_cache_loc"]
        F --> G --> H --> I
    end

    subgraph DEVICE_Graph["Graph 内 固定执行路径"]
        J["embedding 和 qkv projection"]
        K["attention, paged KV, MLA"]
        L["MLP or MoE"]
        M["logits"]
        J --> K --> L --> M
    end

    subgraph CPU_Post["Graph 后必要结果回传"]
        N["sampling top token 和 logprob"]
        O["更新 request state"]
        P["输出协议和 logging"]
        N --> O --> P
    end

    E --> F
    I --> J
    M --> N
```

图中包含三类边界：`dict/list/deque/bisect`、prefix cache、KV allocator、tokenizer 和协议处理属于 Graph 外；`copy_/_foreach_copy_` 是 Graph 外到 Graph 内的入口；attention、MLP/MoE、logits 等固定执行路径适合 capture。

动态 shape OP、`.item()`、完整 tensor `.cpu()` 或 Python 容器进入 Graph 内，会破坏 Graph 外/内边界。

#### SGLang DeepSeek V4 中的 Graph 用法

Graph 录制固定的 DEVICE侧执行流程。第一次 capture 时，SGLang 固定 kernel 顺序、tensor 地址和 stream 依赖；后续 replay 时，Python 不再在 decode 单步中重复调度，而是把新请求的数据写入既有 buffer，再重放同一段执行流程。

该机制降低了小 batch、多 kernel decode 场景中的单步调度开销。

SGLang graph 服务在线推理 decode。请求进入 scheduler 后，CPU侧先确定本轮 batch、seq_len、KV cache 位置和 graph bucket。

例如真实 batch 为 3 时，scheduler 会选择已 capture 的 `bs=4` bucket。

进入 replay 前，SGLang 不重新创建 `input_ids`、`positions`、`seq_lens`、`out_cache_loc` 等 tensor，而是把真实内容 `copy_` 到 capture 时留下的固定 buffer。batch 不足的槽位用 padding 或填充请求补齐。

随后 attention metadata、KV page 信息和 logits buffer 都保持对象地址不变，最后调用 `graph.replay()`。

DSV4 attention metadata 也遵循同一模式：普通 tensor 字段原地 `copy_`，FlashMLA 这类特殊 metadata 按实现约定更新引用，把动态请求转换成固定形状的 tensor 输入。

一轮 SGLang decode 的流程为：CPU scheduler 选择请求和 bucket；CPU侧计算 padding、page table、seq_lens 等 metadata；DEVICE侧固定 buffer 接收 `copy_`。

随后 graph replay 执行 embedding、attention、MLP/MoE、logits 等固定路径；replay 结束后只回传最终 token、必要 logprob 或统计值给采样和调度。

`.item()`、`.tolist()`、新 tensor 分配和动态 Python 分支应避开 graph 延迟敏感路径。它们会破坏 capture 固定性，或把异步执行变成 CPU侧等待 DEVICE侧。

进入 graph 的 OP 应避免改变 tensor 地址、产生动态 shape、触发 CPU 同步、依赖 Python 分支，或调用不支持 capture 的 backend。

#### 普通 Graph、Piecewise Graph 与 torch.compile

普通 Graph、Piecewise Graph 和 `torch.compile` 对应三类不同需求。普通 Graph 避免同一段 DEVICE侧执行每步都被 Python 重新调度。

Piecewise Graph 处理整段 forward 较动态、但局部片段的 shape、地址和控制流固定的场景。

`torch.compile` 让编译器融合、重排或生成更高效的执行图。三者可以组合使用，但约束不同。

典型配合方式是：普通 Graph 捕获整段固定执行路径；Piecewise Graph 捕获多个局部固定片段；`torch.compile` 将 eager 片段优化成 callable 后，Graph 在固定 shape 下 capture 该 callable 的执行。

```mermaid
flowchart TD
    A["PyTorch eager 代码"] --> B{"路径是否固定"}
    B -->|"固定 shape, 固定地址, 固定控制流"| C["普通 Graph capture 整段 forward"]
    C --> D["graph.replay 重放整段路径"]

    B -->|"局部固定, 局部动态"| E["拆分为 piece"]
    E --> F["piece A: attention 或 norm"]
    E --> G["piece B: MLP, MoE 或 logits"]
    F --> H["Graph A capture 和 replay"]
    G --> I["Graph B capture 和 replay"]
    H --> J["eager 调度连接 piece 边界"]
    I --> J

    B -->|"OP 组合可优化但不满足 replay 约束"| K["torch.compile"]
    K --> L["compiled callable"]
    L --> M{"compiled callable 是否具备 Graph capture/replay 兼容性"}
    M -->|"是"| N["Graph capture compiled callable"]
    M -->|"否"| O["保留 compiled 或 eager 执行"]
```

推进时先保证 eager 语义正确，再评估 compile 或融合，最后检查是否满足 Graph 的固定 shape、固定地址和后端 Graph capture/replay 兼容性要求。

过早追求 capture 会把动态请求、allocator、CPU 同步和 backend fallback 同时纳入问题定位范围，削弱具体 OP 分析的针对性。

普通 Graph 捕获一整段 shape、地址和控制流固定的执行路径。以 decode 为例，capture 前 SGLang 先选定 batch bucket，预分配 `input_ids`、`positions`、`seq_lens`、KV metadata、logits buffer 和中间临时 buffer。

warmup 用于让 allocator、kernel cache 和后端状态完成初始化。

capture 中，Graph 记录 kernel 顺序、tensor 地址和 stream 依赖。replay 前，SGLang 只用 `copy_/_foreach_copy_` 更新固定 buffer 的内容。

replay 时，Graph 按录制好的路径执行，不再重新走 Python 调度。

replay 后，采样、请求队列更新、日志和协议输出回到 CPU侧。

普通 Graph 覆盖整段 decode 路径，减少 Python 调度和 launch 开销的空间最大，约束也最严格：shape 要固定，tensor 地址要固定，执行路径要固定，capture 内所有后端 OP 都要支持 Graph capture/replay。

中间只要出现动态 shape、数据相关 Python 分支、临时 tensor 大量分配、`.item()`/`.tolist()` 这类 CPU 同步，或者某个 fused kernel 不支持 capture，整段 capture 就会失败或 replay 结果不可预期。

Piecewise Graph 将整段 forward 拆成多个可 capture 片段。例如 attention block、MLP/MoE block、RMSNorm、logits projection 或某个已经编译好的 runnable，只要输入 shape、地址和 stream 依赖固定，就可以单独 capture。

piece 之间仍由普通 eager 或 SGLang 调度连接，边界处传递 tensor。它减少的是局部固定片段的调度开销，适合绕开局部动态逻辑或不支持 capture 的后端 OP。

Piecewise Graph 会引入更多边界。每个 piece 都要管理自己的输入地址、输出地址、stream 顺序和生命周期；piece 之间如果需要重新分配 tensor、做 CPU 决策或等待通信，减少调度开销的效果会被削弱。

DeepSeek V4 的 attention、HC/MHC、MoE、fallback 和融合 kernel 路径混在一起时，可以把支持 Graph capture/replay 的片段单独 capture，不能 capture 的片段留在 graph 外。

`torch.compile` 和 Graph 的分工是：`torch.compile` 优化要执行的 OP 图，Graph 固定这段图在某个 shape 和地址下的重复执行。`torch.compile` 会把一段 PyTorch eager 代码变成优化后的 callable，可能做 OP 融合、图级优化或调用后端生成 kernel。

Graph capture 再把该 callable 在固定 shape 下的一次执行录下来。

常见做法是先 compile 出局部计算片段，再对该片段做 Graph capture/replay。

`torch.compile` 本身不保证适合 Graph capture/replay。dynamic shape、数据相关分支、Python 容器操作、fallback 到 eager、CPU sync、backend 不支持，都会导致 graph break 或让编译后的片段不能被 capture。

compile 成功只说明这段 PyTorch OP 可以被编译执行。是否能进一步进入 Graph replay，还要看编译后调用的 kernel、通信和同步 API 是否支持 capture。

在 SGLang DeepSeek V4 实现中，通常优先用普通 Graph 优化 decode bucket。当 attention/MoE/HC 路径过于复杂时，再用 Piecewise Graph 或 `torch.compile` 处理局部 shape/dtype 固定的片段。

CPU scheduler 处理动态请求和 KV 管理，Graph 读取固定 input buffer、metadata buffer 和 logits buffer。

按三类场景选择执行方式。路径 shape 固定、地址固定、控制流固定，且后端具备 Graph capture/replay 兼容性时，优先考虑普通 Graph。

大部分路径可 capture、少量局部逻辑动态时，考虑 Piecewise Graph。

PyTorch OP 组合明显、适合编译优化，但地址和 replay 约束尚不满足时，先考虑 `torch.compile`。

请求调度、tokenizer、I/O、日志、Python 容器、`.item()`、`.tolist()`、动态分配和分布式控制流，应保留在 Graph 外。

##### 普通 Graph 使用样例

示例说明：先创建固定输入 `x`、权重 `w` 和输出 `out`；warmup 后 capture `F.silu(x @ w)`；replay 前只用 `x.copy_()` 更新输入内容。该样例对应 decode bucket 中"输入内容变，tensor 地址不变"的模式。

```python
import torch
import torch.nn.functional as F


device = torch.device("musa:0")
# 固定输入、权重和输出 buffer，capture 后不替换对象。
x = torch.zeros((2, 3), dtype=torch.float32, device=device)
w = torch.tensor([[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]], device=device)
out = torch.empty((2, 2), dtype=torch.float32, device=device)

# warmup 让 allocator/kernel cache 在 capture 前完成初始化。
stream = torch.musa.Stream()
stream.wait_stream(torch.musa.current_stream())
with torch.musa.stream(stream):
    for _ in range(3):
        out.copy_(F.silu(x @ w))
torch.musa.current_stream().wait_stream(stream)

# capture 固定的 matmul + SiLU + copy_ 路径。
graph = torch.musa.MUSAGraph()
with torch.musa.stream(stream):
    graph.capture_begin()
    out.copy_(F.silu(x @ w))
    graph.capture_end()

# replay 前只更新 x 的内容，地址保持不变。
x.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device))
graph.replay()
torch.musa.synchronize()

print("x =", x)
print("out =", out)
```

MUSA 运行结果（整理后）：

```text
x shape=(2, 3), dtype=float32, device=musa:0, value=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
out shape=(2, 2), dtype=float32, device=musa:0, value=[[13.999988555908203, 6.993622779846191], [32.0, 15.999998092651367]]
```

##### Piecewise Graph 使用样例

示例说明：把模型拆成两个 shape 和地址固定的片段 `PieceA` 和 `PieceB`，分别 capture 两张 graph。replay 时先执行 `graph_a` 写入固定中间 buffer `mid`，再执行 `graph_b` 读取 `mid` 并写入 `out`。

该样例对应 attention、MLP/MoE 等片段分别 graph 化的模式。

```python
import torch
import torch.nn.functional as F


class PieceA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([[1.0, -1.0], [0.5, 2.0]], dtype=torch.float32)
        )

    def forward(self, x):
        # 第一个可 capture 片段：matmul 后接 ReLU。
        return F.relu(x @ self.weight)


class PieceB(torch.nn.Module):
    def forward(self, x):
        # 第二个可 capture 片段：读取固定 mid buffer 后做 elementwise add。
        return x + 1.0


device = torch.device("musa:0")
piece_a = PieceA().to(device)
piece_b = PieceB().to(device)

# x/mid/out 都是固定 buffer，分别连接两个 graph piece。
x = torch.zeros((2, 2), dtype=torch.float32, device=device)
mid = torch.empty((2, 2), dtype=torch.float32, device=device)
out = torch.empty((2, 2), dtype=torch.float32, device=device)

# warmup 同时覆盖两个 piece，避免 capture 包含初始化行为。
stream = torch.musa.Stream()
stream.wait_stream(torch.musa.current_stream())
with torch.musa.stream(stream):
    for _ in range(3):
        mid.copy_(piece_a(x))
        out.copy_(piece_b(mid))
torch.musa.current_stream().wait_stream(stream)

# 分别 capture 两个局部固定片段。
graph_a = torch.musa.MUSAGraph()
graph_b = torch.musa.MUSAGraph()
with torch.musa.stream(stream):
    graph_a.capture_begin()
    mid.copy_(piece_a(x))
    graph_a.capture_end()
with torch.musa.stream(stream):
    graph_b.capture_begin()
    out.copy_(piece_b(mid))
    graph_b.capture_end()

# replay 顺序必须保持 piece 间的数据依赖：A 写 mid，B 读 mid。
x.copy_(torch.tensor([[2.0, 1.0], [3.0, 4.0]], device=device))
graph_a.replay()
graph_b.replay()
torch.musa.synchronize()

print("x =", x)
print("mid =", mid)
print("out =", out)
```

MUSA 运行结果（整理后）：

```text
x shape=(2, 2), dtype=float32, device=musa:0, value=[[2.0, 1.0], [3.0, 4.0]]
mid shape=(2, 2), dtype=float32, device=musa:0, value=[[2.5, 0.0], [5.0, 5.0]]
out shape=(2, 2), dtype=float32, device=musa:0, value=[[3.5, 1.0], [6.0, 6.0]]
```

##### torch.compile 使用样例

示例说明：把一段 PyTorch eager 函数交给 `torch.compile`，让编译器处理 `matmul -> gelu -> sigmoid -> add` 这类结构固定的 OP 组合。该样例不做 graph capture，只展示 compile 的基本调用方式。

工程中可在 compile 后再评估该 callable 是否满足 Graph capture 条件。

```python
import torch
import torch.nn.functional as F


device = torch.device("musa:0")


@torch.compile
def compiled_block(x, weight):
    # 编译器可优化这段固定 OP 组合；本例不进入 graph capture。
    y = x @ weight
    return F.gelu(y) + torch.sigmoid(y)


x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
weight = torch.tensor([[1.0, -1.0], [0.5, 2.0]], device=device)
out = compiled_block(x, weight)
torch.musa.synchronize()

print("out =", out)
```

MUSA 运行结果（整理后）：

```text
out shape=(2, 2), dtype=float32, device=musa:0, value=[[2.835296630859375, 3.9485244750976562], [5.9933061599731445, 5.9933061599731445]]
```

注意：该用例在 MUSA 上执行通过；运行时 Inductor 对 MUSA matmul template 使用 fallback heuristic。结论是 `torch.compile` 调用链可执行，但 compile 成功不等于已经命中特定高性能 fused template，也不等于该片段适合 Graph capture。

#### Graph OP 使用注意事项

CUDA/MUSA Graph 都要求固定地址、固定 shape，并且 capture 内 OP 集具备 Graph capture/replay 兼容性。排查 Graph OP 时，检查 replay 前是否只更新固定 buffer 内容。

capture 内还需要检查 dynamic shape、CPU-DEVICE 同步、allocator、新 tensor 替换，以及不支持 capture 的 kernel/通信 API。

工程上应在进入 replay 前完成动态 Python 控制流、CPU 容器处理、`.item()` 决策和 allocator 相关操作。进入 graph replay 前，所有输入都应写入固定 tensor buffer；replay 内只保留确定的 tensor OP、custom kernel 和必要的局部 stream 依赖。

#### SGLang DeepSeek V4 Graph 设计要点

SGLang DSV4 的 graph 设计包含三个约束：

1. shape 固定：不同 batch/token 场景进入不同 bucket，例如 decode/idle、target verify、draft extend。
2. 地址固定：capture 后 replay 避免替换 tensor，通过 `copy_` 更新内容。
3. 同步最小化：replay 路径避免 `.item()`、`.tolist()`、`.cpu()`、allocator 和动态 Python 控制流。

典型 replay 模式：

```python
# replay 前把真实输入写入固定 buffer 的前缀。
fixed_input[:n].copy_(real_input)
fixed_seq_lens[:bs].copy_(real_seq_lens)

# metadata 对象保持固定，只更新字段内容。
metadata.copy_(new_metadata)

# replay 读取 capture 时绑定的 tensor 地址。
graph.replay()
```

DSV4 metadata 通过 `copy_metadata` 更新：tensor 字段用 `dst.copy_(src)`，特殊 FlashMLA metadata 可 assign。该更新方式既能表达动态请求，又不会替换 graph capture 绑定过的 tensor 对象地址。


## 3. Graph OP 详细用例与 MUSA 输出

本节保留 Graph 相关 OP 的生命周期说明、固定地址 replay 模式、MUSA 最小用例和注意事项。

为减少阅读干扰，样例代码只保留输入构造、目标 OP 和必要输出，不再在每个代码块内重复放置统一格式化函数。运行结果保留为整理后的 `shape/dtype/device/value` 格式，便于核对。

### A.5 Graph OP

Graph API 和配套 OP 包括 `torch.musa.MUSAGraph`、`capture_begin/capture_end`、`graph.replay()`、`torch.musa.Stream`、`stream.wait_stream`，以及 graph 前后配套使用的 `copy_`、`_foreach_copy_`、`fill_`、`zero_`、`pad`、`cat`、`stack`、`arange`、`gather/scatter_`。

它们共同完成固定地址 replay、bucket padding、metadata 更新和局部 stream 同步。

SGLang DeepSeek V4 的 decode 阶段按 batch bucket capture graph，replay 前用 `copy_` 更新固定输入，replay 内执行 attention、MLP/MoE 和 logits 路径。

piecewise graph 可只 capture attention block、MLP/MoE block 或 compiled callable；stream/event 保证 capture stream、compute stream 和通信 stream 的顺序。

使用 Graph 时，应确认 capture 内所有 kernel、通信和同步 API 具备 Graph capture/replay 兼容性。

CUDA/MUSA Graph 会把固定 shape、固定地址、固定执行路径的 decode 热点路径捕获下来，replay 时减少 Python 调度和 kernel launch 开销。它表示由 PyTorch tensor OP、stream/event 同步和 graph API 共同组成的执行模式，而非单个数学算子。

#### Graph 生命周期

典型流程是预分配 input buffer 和 metadata buffer，warmup 一次目标 batch size，在非默认 stream 上 capture。之后每轮请求只把真实输入 `copy_` 到固定 buffer，再调用 `graph.replay()`。

capture 记录的是 tensor 地址、kernel 序列和 stream 依赖；replay 阶段不能替换 capture 时使用的 tensor 对象。

SGLang 的 decode 路径会按 batch bucket 捕获多张 graph，例如 `bs=1/2/4/8/...`。真实 batch 小于 capture batch 时，通过 padding、填充槽位和 metadata mask 保持 replay shape 不变。

API 配合关系如下。左侧是一次性初始化和 capture，右侧是每轮 decode replay；capture 之后不再替换 tensor 对象，只更新固定 buffer 内容。

```mermaid
flowchart TD
    A["CPU scheduler 选择 capture batch size"] --> B["torch.empty 和 new_empty 预分配 input buffer"]
    B --> C["预分配 metadata, logits, KV 临时 buffer"]
    C --> D["torch.musa.Stream 创建非默认 stream"]
    D --> E["stream.wait_stream current_stream"]
    E --> F["warmup 执行相同 forward 多次"]
    F --> G["graph = torch.musa.MUSAGraph"]
    G --> H["with torch.musa.stream stream"]
    H --> I["graph.capture_begin"]
    I --> J["执行固定 shape forward: attention, MLP, logits"]
    J --> K["graph.capture_end"]
    K --> L["保存 graph 与固定 buffer 对象"]

    M["每轮 decode 请求进入"] --> N["CPU scheduler 选 graph bucket"]
    N --> O["padding 与填充槽位整理成固定 shape"]
    O --> P["copy_ 和 foreach_copy_ 写 input_ids, positions, seq_lens"]
    P --> Q["metadata.copy_ 写 page table 和 out_cache_loc"]
    Q --> R["graph.replay"]
    R --> S["必要时 stream 或 event 等待"]
    S --> T["采样只回传 next token 和必要 logprob"]
    T --> U["CPU 更新请求队列和 KV 状态"]

    L -->|"复用固定地址"| R
```

图中的 API 分工是：`empty/new_empty` 创建固定对象，`Stream/wait_stream` 建立 capture 前 stream 顺序，`capture_begin/capture_end` 录制固定执行路径。

`copy_/_foreach_copy_` 在 replay 前更新内容，`graph.replay()` 触发重放，`.cpu/.item/.tolist` 只保留在 replay 后的必要结果回传中。

#### Graph 中主要的 PyTorch OP

`empty/new_empty/empty_like` 用于创建固定地址的输入、输出和临时 buffer。capture 后这些 tensor 的对象身份需要保持不变，通过复用来满足 replay 约束。

`copy_/_foreach_copy_` 用于在 replay 前更新输入。它把真实 `input_ids`、`positions`、`seq_lens`、`req_pool_indices`、`out_cache_loc` 写入 capture 时的静态 buffer，满足“内容动态、地址固定”的约束。

`fill_/zero_/masked_fill_` 用于清理 padding 区、填充请求、mask 和输出缓存。它们通常出现在 replay 前准备阶段，避免旧 batch 的残留数据影响新 batch。

`pad/cat/stack/arange/repeat_interleave/expand/gather/scatter_/index_select` 用于把动态请求整理成固定 shape 的 page table、position id、MoE dispatch/combine index 和 token mapping。使用这些 OP 时，需要控制输出 shape、dtype 和 contiguous 约束。

`view/reshape/flatten/unsqueeze/squeeze/contiguous` 用于 graph 内外的 layout 组织。graph 延迟敏感路径中要避免由 `reshape/contiguous` 隐式引入不可控分配；custom kernel 前应显式保证 layout。

`.item()/.tolist()/.cpu()/.numpy()` 属于 CPU-DEVICE 边界 OP，会触发 CPU-DEVICE 同步。它们建议仅用于最终 token、少量 logprob/statistics 的状态上报或 scheduler 决策，避免放入 capture/replay 路径。

SGLang DeepSeek V4 中 Graph 的用法、普通 Graph / Piecewise Graph / torch.compile 的深入比较（含 3 段 MUSA 可执行代码）以及 Graph 相关 OP 注意事项，见 §2.7。本节仅保留 `MUSAGraph.replay` 最小用例。

#### `MUSAGraph.replay`

功能：重放固定地址、固定 shape 的 graph。  
用例：SGLang decode graph bucket replay。

```python
import torch
inp = torch.ones((2, 2), device="musa:0")
out = torch.empty_like(inp)
stream = torch.musa.Stream()
stream.wait_stream(torch.musa.current_stream())
with torch.musa.stream(stream):
    for _ in range(3):
        out.copy_(inp * 2 + 1)
torch.musa.current_stream().wait_stream(stream)

graph = torch.musa.MUSAGraph()
with torch.musa.stream(stream):
    graph.capture_begin()
    out.copy_(inp * 2 + 1)
    graph.capture_end()

inp.fill_(3)
graph.replay()
torch.musa.synchronize()

print("inp =", inp)
print("out =", out)
```

输入：capture 中记录 `out = inp * 2 + 1`，replay 前 `inp` 改为全 `3`。  
MUSA 运行结果（整理后）：

```text
inp = Tensor(shape=(2, 2), dtype=float32, device=musa:0, value=[[3.0, 3.0], [3.0, 3.0]])
out = Tensor(shape=(2, 2), dtype=float32, device=musa:0, value=[[7.0, 7.0], [7.0, 7.0]])
```

注意：MUSA graph capture 需要在非默认 stream 上进行，capture 后可在默认 stream replay。

#### 已验证的 Graph 用例位置

`MUSAGraph.replay` 小节提供最小 capture/replay 用例；第 2 篇 A.3.2 提供 stream/event 与 graph replay 组合用例；§2.7 比较 SGLang DeepSeek V4 Graph、普通 Graph、Piecewise Graph 和 `torch.compile`。

§2.1 将 buffer copy、metadata copy、graph replay、HC fallback 和 SwiGLU clamp 流程抽成 MUSA 最小用例。这些用例共同说明固定地址 replay 的基本模式。

## 4. DeepSeek V4-Pro 源码注释与阅读要点

本节使用 DeepSeek V4-Pro `inference/` 下的主要函数。源码片段采用注释版格式：说明直接写在相关代码旁边，便于按执行顺序阅读输入、layout、dtype、metadata 和热点计算。

源码阅读流程如下。图中表达模块依赖和主要执行关系，不表示所有分支都会在一次 forward 中同时发生。

```mermaid
flowchart TD
    A["输入 token / hidden states"] --> B["B1 Embedding 与 Linear 分派"]
    B --> C["B2 act_quant 与 FP8/FP4 GEMM"]
    C --> D["B3 RoPE 与 window top-k index"]
    D --> E["B4 Attention forward"]
    E --> F["B5 Gate / Expert / MoE forward"]
    F --> G["B6 HC/MHC pre-post block"]

    H["B7 checkpoint 权重转换"] --> B
    H --> C

    E1["Q 路径: q_norm, RoPE"] --> E
    E2["KV 路径: kv_norm, quant, cache"] --> E
    E3["sparse index: window / compressed top-k"] --> E
    E4["output projection: view, einsum, wo_b"] --> E
```

### B.1 Embedding、Linear 分派与量化权重布局

源码位置：`inference/model.py`。

#### B1-01 Embedding forward

```python
# [B1-01]
def forward(self, x: torch.Tensor) -> torch.Tensor:
    if world_size > 1:
        # TP 场景下判断 token 是否属于当前 vocab shard，输出 bool mask。
        mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)

        # 全局 token id 转成当前 shard 内的局部 id。
        x = x - self.vocab_start_idx

        # 非本 shard token 临时置 0，避免 embedding 查表越界。
        x[mask] = 0

    # embedding lookup：输入 id 的 dtype/device 必须与查表路径匹配。
    y = F.embedding(x, self.weight)

    if world_size > 1:
        # 清零非本 shard 的 embedding 结果，准备跨 rank 汇总。
        y[mask] = 0

        # 合并各 rank 的 embedding 结果，是分布式同步点。
        dist.all_reduce(y)
    return y
```

#### B1-02 Linear dtype 分派

```python
# [B1-02]
def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    # 该路径不处理 bias，避免量化 GEMM 接口混入额外分支。
    assert bias is None

    # FP4 权重走量化分支，不走普通 F.linear。
    if weight.dtype == torch.float4_e2m1fn_x2:
        # 对 activation 分块量化，并生成 activation scale。
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)

        # 量化 GEMM 同时读取 activation、activation scale、weight 和 weight scale。
        return fp4_gemm(x, s, weight, weight.scale, scale_dtype)

    # FP8 权重同样需要先量化 activation，再进入 FP8 GEMM。
    elif weight.dtype == torch.float8_e4m3fn:
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp8_gemm(x, s, weight, weight.scale, scale_dtype)

    # 普通 dtype 的 dense linear 路径，不涉及 FP4/FP8 scale 约束。
    else:
        return F.linear(x, weight)
```

#### B1-03 量化权重与 scale 初始化

```python
# [B1-03]
if dtype == torch.float4_e2m1fn_x2:
    # FP4 两个值打包到一个存储单元，逻辑 K 维仍是 in_features。
    self.weight = nn.Parameter(torch.empty(out_features, in_features // 2, dtype=torch.float4_e2m1fn_x2))

    # FP4 scale 沿 K 维按 block 管理，shape 必须与 GEMM kernel 一致。
    scale_out_features = out_features
    scale_in_features = in_features // fp4_block_size

    # scale 也是低精度格式，后续计算要按该 dtype 解释。
    self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float8_e8m0fnu))

elif dtype == torch.float8_e4m3fn:
    # FP8 weight 不做 K/2 打包。
    self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))

    # FP8 scale shape 按 block 上取整。
    scale_out_features = (out_features + block_size - 1) // block_size
    scale_in_features = (in_features + block_size - 1) // block_size
    self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float8_e8m0fnu))

else:
    # 普通 dtype 只创建 weight，不注册 scale 参数；forward 会落到 F.linear。
    self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
    self.register_parameter("scale", None)
```

### B.2 `act_quant`、`fp8_gemm` 与 `fp4_gemm`

源码位置：`inference/kernel.py`。

#### B2-01 Activation quant

```python
# [B2-01]
def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    # 量化沿最后一维分块，最后一维通常是 hidden 或 GEMM K。
    N = x.size(-1)

    # block quant 要求 K 维可整除，否则 scale shape 无法对齐。
    assert N % block_size == 0

    tl_dtype = FE8M0 if scale_dtype == torch.float8_e8m0fnu else FP32

    # kernel 需要连续输入；上游非连续时会产生真实 copy。
    z = x.contiguous()

    # inplace=False 输出 FP8；inplace=True 创建同 dtype 临时输出。
    y = torch.empty_like(z) if inplace else torch.empty_like(z, dtype=torch.float8_e4m3fn)

    # activation scale shape 为“原 shape 去掉最后一维 + N/block_size”。
    s = z.new_empty(*z.size()[:-1], N // block_size, dtype=scale_dtype)

    kernel = act_quant_kernel(N, block_size, scale_dtype=tl_dtype, round_scale=scale_fmt is not None, inplace=inplace)

    # 将任意前缀维折叠成二维 GEMM-like 输入。
    kernel(z.view(-1, N), y.view(-1, N), s.view(-1, N // block_size))

    if inplace:
        # inplace 模式写回原 tensor，后续不能再依赖原始 activation 值。
        x.copy_(y)
        return x
    return y, s
```

#### B2-02 FP8 GEMM

```python
# [B2-02]
def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
    # 明确要求 activation、weight 和 scale 都是连续 layout。
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()

    tl_dtype = FE8M0 if scale_dtype == torch.float8_e8m0fnu else FP32

    # activation 折叠成 [M, K]；weight 逻辑 shape 为 [N, K]。
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)

    # 输出继承 activation 前缀维，最后一维变成 out features。
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())

    kernel = fp8_gemm_kernel(N, K, scale_dtype=tl_dtype)

    # kernel 接口使用二维矩阵，依赖前面的连续性检查。
    kernel(a.view(M, K), b, c.view(M, N), a_s.view(M, -1), b_s)
    return c
```

#### B2-03 FP4 GEMM

```python
# [B2-03]
def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32):
    # FP4 wrapper 与 FP8 wrapper 同样要求输入和 scale 连续。
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()

    tl_dtype = FE8M0 if scale_dtype == torch.float8_e8m0fnu else FP32

    # activation 仍按量化后的 FP8 输入；packed FP4 只影响 weight 的 K 维存储。
    K = a.size(-1)
    M = a.numel() // K

    # b.size(0) 仍表示 out features。
    N = b.size(0)

    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    kernel = fp4_gemm_kernel(N, K, scale_dtype=tl_dtype)
    kernel(a.view(M, K), b, c.view(M, N), a_s.view(M, -1), b_s)
    return c
```

### B.3 RoPE、Window Top-k 与 Compressed Top-k

源码位置：`inference/model.py`。

#### B3-01 RoPE 原地旋转

```python
# [B3-01]
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    # 保存外部传入的 tensor 引用，最后通过 copy_ 写回。
    y = x

    # 转 FP32 后把最后一维两两配对，为 complex view 做准备。
    # view_as_complex 要求最后一维长度为 2 且 stride 合法。
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))

    if inverse:
        # inverse RoPE 使用共轭实现反向旋转。
        freqs_cis = freqs_cis.conj()

    # 根据 Q/K 的维度补 batch/head 维，服务 broadcast。
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))

    # 复数乘法完成旋转，再还原成 real tensor。
    x = torch.view_as_real(x * freqs_cis).flatten(-2)

    # 保持外部 tensor 对象不变，只更新 RoPE 维内容。
    y.copy_(x)
    return y
```

#### B3-02 Window top-k index

```python
# [B3-02]
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    if start_pos >= window_size - 1:
        start_pos %= window_size

        # decode 环形窗口：把尾部和头部拼成窗口 index。
        matrix = torch.cat([torch.arange(start_pos + 1, window_size), torch.arange(0, start_pos + 1)], dim=0)

    elif start_pos > 0:
        # 用 -1 表达无效 index，后续 sparse attention 需要识别该 sentinel。
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)

    else:
        # prefill：生成 [seqlen, window] 的 base 矩阵。
        base = torch.arange(seqlen).unsqueeze(1)

        # clamp/where 修正越界位置，把未来 token 位置置为 -1。
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)

    # expand 生成 batch 维，是 broadcast view，不应对 expanded view 原地写。
    return matrix.unsqueeze(0).expand(bsz, -1, -1)
```

### B.4 Attention Forward：Q/KV、Sparse Index、Cache 与 Output Projection

源码位置：`inference/model.py`。

#### B4-01 输入 metadata 与 Q 路径

```python
# [B4-01]
def forward(self, x: torch.Tensor, start_pos: int):
    # 读取 bsz/seqlen，后续 cache、top-k index 和输出 shape 都依赖它。
    bsz, seqlen, _ = x.size()

    # 取本次 token 对应的 RoPE 频率。
    freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]

    # 保存 window size、compressed ratio 和 RoPE 维度。
    win = self.window_size
    ratio = self.compress_ratio
    rd = self.rope_head_dim

    # 低秩 Q 投影后归一化。
    qr = q = self.q_norm(self.wq_a(x))

    # Q 投影输出整理成 [bsz, seqlen, heads, head_dim]。
    q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))

    # 对 Q 做 RMS-like 缩放，用多个小 OP 表达 reference 语义。
    q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)

    # 只旋转 Q 的 RoPE 维，并通过 copy_ 写回 slice。
    apply_rotary_emb(q[..., -rd:], freqs_cis)
```

#### B4-02 KV 路径、量化与 sparse index

```python
# [B4-02]
# KV 走单独低维路径，输出包含 RoPE 维和非 RoPE 维。
kv = self.wkv(x)
kv = self.kv_norm(kv)

# 只旋转 KV 的 RoPE 维。
apply_rotary_emb(kv[..., -rd:], freqs_cis)

# 非 RoPE 维原地量化，RoPE 维保持较高精度。
act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)

# 构造 sliding window index。
topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)

if self.compress_ratio:
    # prefill 和 decode 使用不同 offset，影响 compressed index 范围。
    offset = kv.size(1) if start_pos == 0 else win

    # indexer 存在时走模型路径，否则走默认 compressed top-k 生成函数。
    if self.indexer is not None:
        compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
    else:
        compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)

    # 合并 window index 和 compressed index。
    topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)

# attention kernel 使用 int32 index。
topk_idxs = topk_idxs.int()
```

#### B4-03 KV cache 写入与 sparse attention

```python
# [B4-03]
if start_pos == 0:
    if seqlen <= win:
        # prefill 短序列直接写窗口 cache。
        self.kv_cache[:bsz, :seqlen] = kv
    else:
        # 长序列进入环形窗口时按 cutoff 拆分写入。
        cutoff = seqlen % win
        self.kv_cache[:bsz, cutoff: win], self.kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1)

    if self.compress_ratio:
        # 生成压缩 KV，用于 compressed attention。
        if (kv_compress := self.compressor(x, start_pos)) is not None:
            # prefill 时把原始 KV 和压缩 KV 合并给 sparse attention。
            kv = torch.cat([kv, kv_compress], dim=1)

    # sparse attention 读取 Q、KV、attn sink 和 top-k index。
    o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
else:
    # decode 单 token 写 cache 时去掉 seq 维。
    self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
    if self.compress_ratio:
        self.compressor(x, start_pos)
    o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)

# 对输出 RoPE 维做反旋转。
apply_rotary_emb(o[..., -rd:], freqs_cis, True)
```

#### B4-04 Output projection

```python
# [B4-04]
# 按 group 整理输出，准备低秩 O projection。
o = o.view(bsz, seqlen, self.n_local_groups, -1)

# O projection weight 整理成 group/rank 维。
wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)

# group-wise low-rank projection。
o = torch.einsum("bsgd,grd->bsgr", o, wo_a)

# 合并 group/rank 维后投回 hidden dim。
x = self.wo_b(o.flatten(2))
return x
```

### B.5 Gate、Expert 与 MoE Forward

源码位置：`inference/model.py`。

#### B5-01 Gate forward

```python
# [B5-01]
def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None):
    # router score 用 FP32 计算，提升数值稳定性，同时引入 cast。
    # Gate weight dtype 仍会决定内部是普通 linear 还是量化 GEMM。
    scores = linear(x.float(), self.weight.float())

    # 不同 score 函数对应不同路由权重语义。
    if self.score_func == "softmax":
        scores = scores.softmax(dim=-1)
    elif self.score_func == "sigmoid":
        scores = scores.sigmoid()
    else:
        scores = F.softplus(scores).sqrt()

    # original_scores 保留 bias 前的分数。
    original_scores = scores

    # bias 影响 top-k 选择。
    if self.bias is not None:
        scores = scores + self.bias

    if self.hash:
        # hash routing 根据 token id 直接查 expert id。
        indices = self.tid2eid[input_ids]
    else:
        # top-k 输出 shape 为 [tokens, topk]。
        indices = scores.topk(self.topk, dim=-1)[1]

    # 按 top-k expert id 取原始权重。
    weights = original_scores.gather(1, indices)

    if self.score_func != "softmax":
        # keepdim=True 保持 broadcast shape。
        weights /= weights.sum(dim=-1, keepdim=True)

    # 对最终路由权重做缩放。
    weights *= self.route_scale
    return weights, indices
```

#### B5-02 Expert forward

```python
# [B5-02]
def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None):
    # 保存输入 dtype，最后投影前恢复。
    dtype = x.dtype

    # expert 内 gate/up 两路投影，并转 FP32 计算激活。
    gate = self.w1(x).float()
    up = self.w3(x).float()

    if self.swiglu_limit > 0:
        # DeepSeek V4 SwiGLU 限幅，控制 gate/up 数值范围。
        up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
        gate = torch.clamp(gate, max=self.swiglu_limit)

    # SwiGLU 激活与逐元素乘。
    x = F.silu(gate) * up

    if weights is not None:
        # 应用 router 权重，依赖 broadcast。
        x = weights * x

    # w2 输入转回原 dtype，避免 FP32 向后扩散。
    return self.w2(x.to(dtype))
```

#### B5-03 MoE forward

```python
# [B5-03]
def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    # 保存原始 shape，最后恢复 batch/seq 结构。
    shape = x.size()

    # 合并 batch/seq，把 token 维展平。
    x = x.view(-1, self.dim)

    # input_ids.flatten() 与展平后的 token 对齐。
    weights, indices = self.gate(x, input_ids.flatten())

    # combine 输出 buffer，累加使用 FP32。
    y = torch.zeros_like(x, dtype=torch.float32)

    # 统计每个 expert token 数并回读到 CPU，会形成 CPU-DEVICE 边界。
    counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()

    for i in range(self.experts_start_idx, self.experts_end_idx):
        if counts[i] == 0:
            continue

        expert = self.experts[i]

        # 当前 expert 的动态长度 token index，适合 reference，不适合 Graph 内部。
        idx, top = torch.where(indices == i)

        # advanced indexing 取 token 和权重，再按 token index 累加 expert 结果。
        y[idx] += expert(x[idx], weights[idx, top, None])

    if world_size > 1:
        # TP rank 间合并本地 expert 输出，是分布式同步点。
        dist.all_reduce(y)

    # 加入 shared expert 结果。
    y += self.shared_experts(x)

    # 恢复 dtype 和原始 shape。
    return y.type_as(x).view(shape)
```

### B.6 HC/MHC：Pre、Post 与 Block Forward

源码位置：`inference/model.py`。

#### B6-01 HC pre

```python
# [B6-01]
def hc_pre(self, x, hc_fn, hc_scale, hc_base):
    # 保存原 shape/dtype，后面恢复输出。
    shape, dtype = x.size(), x.dtype

    # 合并 HC copy 维和 hidden 维，计算使用 FP32。
    x = x.flatten(2).float()

    # 计算 RMS-like normalization scale。
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)

    # 生成 HC mixing 参数，并按归一化 scale 缩放。
    mixes = F.linear(x, hc_fn) * rsqrt

    # 把 mixing 参数拆成 pre/post/comb。
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)

    # pre 权重作用到每个 HC copy，再沿 copy 维压缩成普通 hidden。
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)

    # 输出恢复到输入 dtype。
    return y.to(dtype), post, comb
```

#### B6-02 HC post

```python
# [B6-02]
def hc_post(self, x, residual, post, comb):
    # post 把 attention/FFN 输出扩展回 HC copy 维。
    # comb 对 residual 的 HC copy 做加权组合，并沿 copy 维求和。
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)

    # 输出 dtype 与当前 hidden 对齐。
    return y.type_as(x)
```

#### B6-03 Block forward

```python
# [B6-03]
def forward(self, x, start_pos, input_ids):
    # attention 子层由 HC pre/post 包裹，不直接执行普通 residual add。
    residual = x
    x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
    x = self.attn_norm(x)
    x = self.attn(x, start_pos)
    x = self.hc_post(x, residual, post, comb)

    # FFN/MoE 子层同样通过 HC 混合回多 copy 状态。
    residual = x
    x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
    x = self.ffn_norm(x)
    x = self.ffn(x, input_ids)
    x = self.hc_post(x, residual, post, comb)
    return x
```

### B.7 权重转换：FP8 反量化与 FP4 视图转换

源码位置：`inference/convert.py`。

#### B7-01 FP8 权重反量化

```python
# [B7-01]
# 读取 checkpoint 中的 packed/blocked weight。
weight = state_dicts[i][name]

# 取出对应 scale，并从 state_dict 中移除，避免重复保存。
scale = state_dicts[i].pop(name.replace("weight", "scale"))

# 按 block 还原 out 维和 in/K 维，再用 scale 反量化每个 block。
weight = weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float() * scale[:, None, :, None].float()

# 恢复普通矩阵 layout，并保存为 BF16。
state_dicts[i][name] = weight.flatten(2, 3).flatten(0, 1).bfloat16()
```

#### B7-02 FP4/FP8 格式转换

```python
# [B7-02]
if expert_dtype == "fp8":
    # 定位同一权重对应的 scale。
    scale_name = name.replace("weight", "scale")

    # 取出 weight 和 scale，后面写回转换后的结果。
    weight = state_dicts[i].pop(name)
    scale = state_dicts[i].pop(scale_name)

    # FP4/FP8 格式转换同时更新 weight 和 scale。
    state_dicts[i][name], state_dicts[i][scale_name] = cast_e2m1fn_to_e4m3fn(weight, scale)
else:
    # 重新解释底层 packed 数据为 FP4 dtype，不是数值计算转换。
    state_dicts[i][name] = state_dicts[i][name].view(torch.float4_e2m1fn_x2)
```

## 5. DeepSeek V4 在 SGLang 中的源码注释与阅读要点

本节聚焦 SGLang DeepSeek V4 的运行时源码。和附录 B 的 reference 模型不同，SGLang 代码更强调在线推理：多 stream prepare、metadata 原地更新、Graph replay、HC/MHC 可替换路径，以及 MoE SwiGLU 的执行路径选择。

### C.1 Attention Prepare：多 Stream、Q/KV 计算、RoPE 与 Cache Store

源码位置：`python/sglang/srt/models/deepseek_v4.py`。

```python
# [C1-01]
# 当前 stream 承载主计算路径，三条辅助 stream 分别处理 KV、compressor 和 indexer。
current_stream = torch.cuda.current_stream()
stream_kv = self.alt_streams[0]
stream_compressor = self.alt_streams[1]
stream_indexer = self.alt_streams[2]

# 辅助 stream 先等待主 stream，确保输入 x/qkv_a 等已准备好。
stream_kv.wait_stream(current_stream)
stream_compressor.wait_stream(current_stream)
stream_indexer.wait_stream(current_stream)

# 主 stream 计算 Q LoRA 中间态，并记录 ready event 供 indexer 使用。
q_lora = self._compute_q_a(x, qkv_a=qkv_a)
q_lora_ready = current_stream.record_event()

if self.indexer is not None:
    # indexer 使用独立 stream 生成 sparse/compressed attention metadata。
    with torch.cuda.stream(stream_indexer):
        self.indexer(x=x, q_lora=q_lora, forward_batch=forward_batch, enable_multi_stream=True, q_lora_ready=q_lora_ready)

# KV 路径放到 stream_kv，与 Q-B 或 indexer 路径并行。
with torch.cuda.stream(stream_kv):
    if qkv_a_ready is not None:
        # fuse_wqa_wkv 场景下，KV 计算需要等待 QKV-A 输出 ready。
        stream_kv.wait_event(qkv_a_ready)
    kv = self._compute_kv(x, positions, qkv_a=qkv_a)
    if self.overlap_store_cache:
        # KV 生成后立即写 cache，减少主 stream 上的等待时间。
        attn_backend.store_cache(layer_id=self.layer_id, swa_k=kv, forward_batch=forward_batch)

if self.compressor is not None:
    # compressor 独立 stream 处理压缩 KV 相关计算。
    with torch.cuda.stream(stream_compressor):
        attn_backend.forward_core_compressor(x, forward_batch, self.layer_id, self.compressor)

# 主 stream 继续计算 Q-B，输出 attention kernel 需要的 Q。
q = self._compute_q_b(q_lora, positions)
if q_out is not None:
    # 写入外部预分配 buffer，保持 graph/piecewise 场景中的对象地址稳定。
    q_out.copy_(q)

# 返回前 join 三条辅助 stream，保证 q/kv/index/compressor 状态均可用。
current_stream.wait_stream(stream_kv)
current_stream.wait_stream(stream_compressor)
current_stream.wait_stream(stream_indexer)
return q, kv
```

阅读要点：

- 这段代码把 Q、KV、compressor 和 indexer 拆到多条 stream 上执行，最后回到主 stream 汇合。
- 排查时重点看 `wait_stream`、`wait_event` 和 `record_event` 的依赖范围，判断并行是否被过度等待削弱。
- `store_cache` 是 KV cache 写入入口，依赖 `forward_batch` 中的 page、slot 和 sequence metadata。
- `q_out.copy_(q)` 用于写预分配输出 buffer，Graph 或 piecewise graph 场景中要确认它不会替换 tensor 对象。

### C.2 Attention Prepare：普通路径中的 Q/KV Layout 与 RoPE

源码位置：`python/sglang/srt/models/deepseek_v4.py`。

```python
# [C2-01]
if self.fuse_wqa_wkv:
    # 融合路径一次 projection 得到 Q LoRA 和 KV 原始表示。
    qkv_a, _ = self.wqkv_a(x)
    q = qkv_a[..., : self.q_lora_rank]
    kv = qkv_a[..., self.q_lora_rank :]
    # 切分完成后释放中间引用，降低峰值显存占用。
    del qkv_a
else:
    # 非融合路径分别计算 KV 和 Q-A。
    kv, _ = self.wkv(x)
    q, _ = self.wq_a(x)

# Q-A 先归一化，q_lora 保留给 indexer 使用。
q = self.q_norm(q)
q_lora = q

# Q-B 投影后整理成 attention head layout。
q, _ = self.wq_b(q)
q = q.view(-1, self.n_local_heads, self.head_dim)

# 根据配置选择 JIT norm 或 Triton norm。
if self.use_jit_norm:
    q = rmsnorm_self(q, self.eps)
else:
    q = rms_normalize_triton(q, self.eps)

# KV 路径归一化后，Q/KV 的 RoPE 维一起进入 fused_rope。
kv = self.kv_norm(kv)
fused_rope(
    q[..., -self.qk_rope_head_dim :],
    kv[..., -self.qk_rope_head_dim :].unsqueeze(1),
    self.freqs_cis,
    positions=positions,
)

if self.nsa_enable_prefill_cp and nsa_use_prefill_cp(forward_batch):
    # CP prefill 需要 all-gather 并重新排列 KV，输入要求连续 layout。
    kv = cp_all_gather_rerange_output(kv.contiguous(), self.cp_size, forward_batch, torch.cuda.current_stream())

if self.overlap_store_cache:
    # 允许 cache store 与后续计算重叠。
    attn_backend.store_cache(layer_id=self.layer_id, swa_k=kv, forward_batch=forward_batch)

if self.indexer is not None:
    # 准备 sparse/compressed attention index。
    self.indexer(x=x, q_lora=q_lora, forward_batch=forward_batch)
if self.compressor is not None:
    # 准备 compressed KV 路径。
    attn_backend.forward_core_compressor(x, forward_batch, self.layer_id, self.compressor)

if q_out is not None:
    # 外部需要固定输出 buffer 时使用 copy_ 写入。
    q_out.copy_(q)
return q, kv
```

阅读要点：

- 这段代码先选择 fused 或非 fused 的 Q/KV projection，再整理 Q head layout，并对 Q/KV 的 RoPE 维调用 fused RoPE。
- `q.view(-1, heads, head_dim)` 依赖 stride 兼容；如果上游 layout 不满足要求，后面可能出现额外 copy 或 kernel 输入不匹配。
- `kv.contiguous()` 只在 CP prefill 条件下出现，可通过 profiler 判断是否产生真实 copy。
- `store_cache`、`indexer` 和 `compressor` 是 attention backend 的关键入口，分别对应 KV 写入、sparse index 构造和 compressed KV 路径。

### C.3 Metadata Copy 与 Graph Metadata Replay

源码位置：`python/sglang/srt/layers/attention/deepseek_v4_backend.py`。

```python
# [C3-01]
def copy_(self, other: DSV4AttnMetadata) -> None:
    # 结构性字段必须与 capture 时一致，普通 tensor 字段用原地 copy 更新。
    copy_metadata(
        src=other,
        dst=self,
        check_eq_fields=["c4_sparse_topk", "page_size", "cuda_int32_kwargs"],
        copy_fields=[
            "raw_out_loc", "seq_lens_casual", "positions_casual", "c4_out_loc",
            "c128_out_loc", "page_table", "swa_page_indices", "swa_topk_lengths",
            "c128_page_indices", "c128_topk_lengths_clamp1", "c4_topk_lengths_raw",
            "c4_topk_lengths_clamp1", "c4_sparse_topk_lengths", "c4_sparse_page_indices",
        ],
        assign_fields=["c1_flashmla_metadata", "c4_flashmla_metadata", "c128_flashmla_metadata"],
    )

# [C3-02]
def copy_(self, other: DSV4RawDecodeMetadata):
    # replay 前更新 DEVICE侧 request index、seq lens 和 cache 写入位置。
    self.req_pool_indices.copy_(other.req_pool_indices)
    self.seq_lens.copy_(other.seq_lens)
    self.out_cache_loc.copy_(other.out_cache_loc)

# [C3-03]
def replay_cuda_graph_metadata_from(self, bs, temp_metadata, bucket):
    # 按 bucket/bs 找到 capture 时创建的固定 metadata 对象。
    chosen_metadata = self.cuda_graph_metadata_of_bucket_and_bs[bucket][bs]
    # 把本轮动态请求的 metadata 写入固定对象。
    chosen_metadata.copy_(temp_metadata)
    # forward 读取固定 metadata，满足 graph replay 的地址约束。
    self.forward_metadata = chosen_metadata
```

阅读要点：

- Graph replay 使用固定 metadata 对象，普通 tensor 字段通过 `copy_` 更新内容，不替换对象。
- `check_eq_fields` 表示 replay 时不能改变的结构性约束，例如 page size、top-k 配置和 int32 参数集合。
- FlashMLA metadata 属于特殊对象，代码中通过 `assign_fields` 按实现约定处理，排查时要单独确认生命周期和 capture 兼容性。
- `replay_cuda_graph_metadata_from` 的核心是把本轮动态请求写入已 capture 的 bucket metadata。

### C.4 Forward Metadata 初始化与 Replay 准备

源码位置：`python/sglang/srt/layers/attention/deepseek_v4_backend.py`。

```python
# [C4-01]
# DEVICE侧 metadata 使用 int32，CPU侧副本用于调度判断。
req_pool_indices = forward_batch.req_pool_indices
seq_lens = forward_batch.seq_lens.to(torch.int32)
seq_lens_cpu = forward_batch.seq_lens_cpu
max_seq_len = int(seq_lens_cpu.max().item())

if forward_batch.forward_mode.is_decode_or_idle():
    # decode/idle 使用固定长度 metadata，服务单步 replay。
    metadata = self.init_forward_metadata_decode(
        max_seq_len=max_seq_len,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        out_cache_loc=forward_batch.out_cache_loc,
    )
elif forward_batch.forward_mode.is_prefill(include_draft_extend_v2=True):
    # prefill 需要额外的 extend length 和 token 数，用于构造变长 attention metadata。
    extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
    extend_seq_lens = forward_batch.extend_seq_lens
    metadata = self.init_forward_metadata_prefill(
        max_seq_len=max_seq_len,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu.tolist(),
        out_cache_loc=forward_batch.out_cache_loc,
        num_tokens=sum(extend_seq_lens_cpu),
        extend_seq_lens=extend_seq_lens,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        need_compress=not is_draft,
    )
self.forward_metadata = metadata

# [C4-02]
# replay bucket 只保留真实 batch 前缀，剩余槽位后面 padding。
seq_lens = seq_lens[:bs]
seq_lens_cpu = seq_lens_cpu[:bs]
req_pool_indices = req_pool_indices[:bs]
actual_max_seq_len = seq_lens_cpu.max().item()
chosen_max_seq_len = self.MAX_SEQ_LEN_FOR_CAPTURE
assert actual_max_seq_len <= chosen_max_seq_len

# out_cache_loc 补齐到 capture bucket 的固定长度。
out_cache_loc_padded = torch.nn.functional.pad(out_cache_loc, pad=(0, bs - len(out_cache_loc)), mode="constant", value=0)

# 用固定 max_seq_len 重新构造 replay metadata，再写入固定对象。
temp_metadata = self.init_forward_metadata_decode(
    max_seq_len=chosen_max_seq_len,
    req_pool_indices=req_pool_indices,
    seq_lens=seq_lens,
    out_cache_loc=out_cache_loc_padded,
)
self.replay_cuda_graph_metadata_from(bs=bs, temp_metadata=temp_metadata, bucket=bucket)
```

阅读要点：

- metadata 初始化在 Graph 外完成，decode、idle 和 prefill 根据 forward mode 走不同构造路径。
- `seq_lens_cpu.max().item()` 读取的是 CPU metadata，不是 DEVICE tensor；这里不会引入 DEVICE 等待。
- replay 准备阶段只保留真实 batch 前缀，并把 `out_cache_loc` padding 到 capture bucket 需要的固定长度。
- `chosen_max_seq_len` 是 Graph bucket 的固定上限，真实请求不能超过该上限。

### C.5 Decode Graph Buffer：CPU/GPU Buffer 的分开更新

源码位置：`python/sglang/srt/model_executor/cuda_graph_runner.py`。

```python
# [C5-01]
# CPU seq_lens buffer 保持在 CPU，供 scheduler 和 bucket 逻辑直接读取。
seq_lens_cpu = torch.full(
    (max_bs,),
    seq_len_fill_value,
    dtype=torch.int32,
    device="cpu",
)

# [C5-02]
# GPU tensor 字段按组批量 copy，减少 Python 循环触发的调度开销。
_grouped_foreach_copy_(dsts, srcs)

if forward_batch.seq_lens_cpu is not None:
    if bs != raw_bs:
        # bucket 大于真实 batch 时，先清理 CPU padding 槽位。
        self.seq_lens_cpu.fill_(seq_len_fill_value)
    # 只覆盖真实 batch 前缀，保持固定长度 CPU buffer。
    self.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)
```

阅读要点：

- decode graph buffer 分成 CPU metadata buffer 和 GPU tensor buffer，两者不能混在同一组 foreach copy 中处理。
- `_grouped_foreach_copy_` 用于批量更新 GPU 固定 buffer，减少 Python 循环调度开销。
- `seq_lens_cpu` 保持在 CPU，scheduler 和 bucket 选择可以直接读取，不需要从 DEVICE 回读。
- bucket 大于真实 batch 时，CPU 和 GPU 侧 padding 槽位都要有稳定填充值。

### C.6 HC/MHC：Torch Reference、TileLang/DeepGEMM 路径与 Post Combine

源码位置：`python/sglang/srt/models/deepseek_v4.py`。

```python
# [C6-01]
def hc_pre_torch_impl(x, hc_fn):
    # Torch reference 路径：展平 HC copy/hidden 维并转 FP32。
    x_flat = x.flatten(1).float()
    # 计算 RMS-like scale。
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.rms_norm_eps)
    # linear 生成 mixing 参数，再补 piece 维。
    mixes = (F.linear(x_flat, hc_fn) * rsqrt).unsqueeze(1)
    return x_flat, mixes

shape, dtype = x.size(), x.dtype
if x.shape[0] == 0:
    # 空 batch 直接返回空 tensor，避免后续 kernel 处理 0 token。
    y = torch.empty((0, shape[-1]), dtype=dtype, device=x.device)
    post = torch.empty((0, self.hc_mult), dtype=dtype, device=x.device)
    comb = torch.empty((0, self.hc_mult, self.hc_mult), dtype=dtype, device=x.device)
    return y, post, comb, False

if envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get():
    # TileLang/MHC 融合路径同时完成 norm、mixing 和 pre 输出。
    post, comb, y = mhc_pre(
        residual=x,
        fn=hc_fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=self.rms_norm_eps,
        hc_pre_eps=self.hc_eps,
        hc_sinkhorn_eps=self.hc_eps,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=self.hc_sinkhorn_iters,
        **norm_kwargs,
    )
    return y, post.squeeze(-1), comb, fuse_norm

if envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():
    # DeepGEMM 路径使用 BF16 输入和 FP32 weight/workspace。
    x_flat = x.flatten(1).bfloat16()
    m, k = x_flat.shape
    mix_hc = hc_fn.size(0)
    d_out = torch.empty((m, mix_hc), dtype=torch.float, device=x.device)
    s_out = torch.empty((m,), dtype=torch.float, device=x.device)
    deep_gemm.tf32_hc_prenorm_gemm(x_flat, hc_fn.float().contiguous(), d_out, s_out, num_splits=None)
    # s_out 保存平方和，除以 K 后计算 RMS-like scale。
    rsqrt = torch.rsqrt(s_out / k + self.rms_norm_eps)
    mixes = (d_out * rsqrt.unsqueeze(1)).unsqueeze(1)
else:
    # fallback 使用纯 PyTorch reference。
    x_flat, mixes = hc_pre_torch_impl(x, hc_fn)

# split/sinkhorn 生成 pre/post/comb，pre 用于压缩 HC copy 维。
pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
y = (pre.squeeze(1).unsqueeze(-1) * x_flat.view(shape)).sum(dim=1)
return y.to(dtype), post.squeeze(1), comb.squeeze(1), False

# [C6-02]
def hc_post_torch_impl(x, residual, post, comb):
    # post 作用于当前子层输出，comb 作用于 residual 的 HC copy 维。
    return (
        post.unsqueeze(-1) * x.unsqueeze(1)
        + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
    ).type_as(x)
```

阅读要点：

- HC pre 有三条路径：TileLang/MHC 融合路径、DeepGEMM 路径和 PyTorch reference fallback。
- PyTorch reference 由 `flatten -> float -> square/mean/rsqrt -> linear -> split/sinkhorn -> sum` 组成，语义清楚但 kernel 数量多。
- DeepGEMM 路径要求 BF16 输入、FP32 weight/workspace，并显式调用 `contiguous()` 满足 kernel layout。
- HC post 用 broadcast 和 reduction 把当前子层输出、residual copy 和 mixing 参数合成回多 copy hidden。

### C.7 MoE SwiGLU Clamp：Reference、原地裁剪与 SwishGLU 路径

源码位置：`python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py`。

```python
# [C7-01]
def _swiglu_silu_clamp_mul(x, gemm1_limit):
    # GEMM1 输出按最后一维拆成 gate/up 两支。
    gate, up = x.chunk(2, dim=-1)
    # gate 先做 SiLU，再限制上界。
    gate = F.silu(gate)
    gate = gate.clamp(min=None, max=gemm1_limit)
    # up 分支限制上下界。
    up = up.clamp(min=-gemm1_limit, max=gemm1_limit)
    # SwiGLU 输出为 gate 与 up 的逐元素乘。
    return gate * up

# [C7-02]
if swiglu_limit is not None:
    # DeepSeek V4 当前限制值固定为 10，GEMM1 输出 shape 必须为 [tokens, N]。
    assert swiglu_limit == 10
    assert intermediate_cache1.shape == (total_tokens, N)
    swiglu_limit_for_triton = None
    swiglu_limit_for_silu_and_mul_clamp = None

    if envs.SGLANG_OPT_SWIGLU_CLAMP_FUSION.get():
        # 根据 expert 是否过滤，选择把 clamp 下沉到 Triton 或 activation kernel。
        if filter_expert:
            swiglu_limit_for_triton = swiglu_limit
        else:
            swiglu_limit_for_silu_and_mul_clamp = swiglu_limit
    else:
        # 非融合路径直接在 GEMM1 cache 上原地裁剪 gate/up。
        half = N // 2
        intermediate_cache1[:, :half].clamp_(max=swiglu_limit)
        intermediate_cache1[:, half:].clamp_(min=-swiglu_limit, max=swiglu_limit)

    if not filter_expert:
        if swiglu_limit_for_silu_and_mul_clamp is not None:
            # fused activation kernel 一次完成 clamp、SiLU 和 mul。
            silu_and_mul_clamp(intermediate_cache1.view(-1, N), intermediate_cache2, swiglu_limit_for_silu_and_mul_clamp)
        else:
            if _is_musa:
                # MUSA 路径调用 SwishGLU module 表达同一语义。
                intermediate_cache2 = torch.nn.SwishGLU()(intermediate_cache1.view(-1, N))
            else:
                # 非 MUSA 分支使用已有 fused silu_and_mul kernel。
                silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
```

阅读要点：

- 这段代码处理 DeepSeek V4 的 SwiGLU 限幅：GEMM1 输出拆成 gate/up，gate 走 SiLU，两支都要按限制值裁剪。
- 非融合路径会对 `intermediate_cache1` 做原地 `clamp_`，后续不能再依赖裁剪前的 gate/up 值。
- 融合路径把 clamp、SiLU 和 multiply 下沉到专用 kernel，排查时需要确认是否命中目标 kernel。
- MUSA 分支使用 `torch.nn.SwishGLU()` 表达同一语义，排查时要看它实际展开成哪些 kernel。

## 6. 回顾总结

本文基于 Transformer 和 DeepSeek V4-Pro 的真实结构，分析 PyTorch OP 在计算、layout、metadata、同步和 Graph replay 中分别承担什么角色。

1. **先看 OP 类型**：GPU OP 负责张量计算和 layout，CPU OP 负责调度与 metadata，Sync OP 改变执行时序，Dynamic Shape OP 带来 shape 不确定性，Graph OP 依赖固定地址和固定执行序列。
2. **性能问题常来自放置位置不合适**：`contiguous()`、`.item()`、`masked_select`、`bincount(...).tolist()`、动态 `cat/split` 单独看都合理，但放进 decode 热点路径、Graph replay 内部或大 tensor 处理路径里，就可能成为瓶颈。
3. **Transformer 按模块组织 OP**：Embedding/Position、Attention/KV cache、RMSNorm/MLP、MoE、Logits/Sampling 都有稳定的 OP 组合方式。先看模块输入输出，再看每个 OP 是否改变 layout、分配内存、触发同步或制造动态 shape。
4. **DeepSeek V4-Pro 放大了 OP 约束**：FP8/FP4 Linear 要同时看 activation scale、weight scale、packed layout 和 GEMM 路径；Compressed Attention 要同时看 Q/KV layout、top-k index 和 sparse kernel；MoE/HC/MHC 要区分 reference 计算序列和热点执行路径。
5. **MUSA 用例用于验证语义和边界**：最小例子展示了 `copy_`、Graph replay、KV indexing、MoE combine、sampling 回传等 OP 的输入输出。验证时除了能否运行，还要检查输出 shape、dtype、device、数值和同步边界是否符合预期。

后续分析任意 PyTorch OP 时，按以下顺序检查：功能语义和输入输出、layout/dtype、分配或同步行为、所在模块角色。

模块角色包括 reference、metadata 准备、Graph replay 前更新和热点张量计算。
