# PyTorch 常见 OP 用法与 DeepSeekV4 源码分析

内容分两部分：

1. 常见 PyTorch OP 的基本用法、输入输出和内存行为。
2. 这些 OP 在 DeepSeekV4 源码中的具体位置和作用。

DeepSeekV4 源码文件：

```text
repos/transformers/src/transformers/models/deepseek_v4/modular_deepseek_v4.py
```

配套最小示例脚本：

```bash
cd /home/mtuser/workspace
python insights/Pytorch/pytorch_op_common_usage_deepseekv4_examples.py
```

脚本只依赖 CPU PyTorch，不导入 Transformers，不运行真实 DeepSeekV4 模型。脚本用小尺寸 tensor 演示下列 OP，重点核对 shape、value、contiguous 状态和执行流程。

运行输出摘要：

```text
embedding / indexing:
  input_ids shape=(4,), value=tensor([1, 3, 2, 4])
  hidden    shape=(4, 2), value=tensor([[1., 0.], [1., 1.], [0., 1.], [2., 1.]])

view / transpose / contiguous:
  q.transpose(1, 2) shape=(2, 4, 2)
  transpose is_contiguous=False
  after contiguous is_contiguous=True

linear / matmul / bmm:
  F.linear(hidden, weight) shape=(4, 3)
  hidden @ hidden.T        shape=(4, 4)
  torch.bmm(...)           shape=(2, 2, 1)

compression:
  compressed entries shape=(2, 2), value=tensor([[0.881, 0.881], [1.269, 1.000]])

mask:
  block_bias shape=(4, 2), value=tensor([[-inf, -inf], [0., -inf], [0., -inf], [0., 0.]])

MoE routing:
  topk_indices        shape=(3, 2), value=tensor([[1, 2], [0, 2], [2, 1]])
  MoE combined output shape=(3, 2), value=tensor([[2.182, 0.000], [0.000, 1.667], [2.583, 2.583]])
```

## 1. OP 是什么

PyTorch OP 是一次张量操作。有些 OP 只改 tensor metadata，有些 OP 会执行计算、分配内存、复制数据或触发设备同步。

同一个 API，输入不同，底层行为也不同。读源码时重点看这几项：

1. `shape`
2. `dtype`
3. `device`
4. `stride`
5. 是否连续
6. 是否分配新 tensor
7. 是否处于 decode 热点路径

| API | 常见行为 | 检查点 |
| --- | --- | --- |
| `view` | 只改 shape metadata | 输入 stride 必须兼容目标 shape |
| `reshape` | 返回 view 或复制数据 | stride 不兼容时会复制 |
| `transpose` | 交换维度，返回非连续 view | 后续 OP 是否接受非连续 layout |
| `contiguous` | 生成连续内存 | 非连续输入会产生真实 copy |
| `F.linear` | 线性投影或 GEMM | Q/K/V、MLP、LM head、expert GEMM |
| `softmax` | 归一化概率 | attention、压缩 gate、MoE routing |
| `topk` | 选最大 k 个值 | MoE 选专家、sampling、CSA indexer |
| `gather/scatter/index_add_` | 按 index 读写或累加 | MoE dispatch/combine、mask、cache |
| `.item/.cpu/.tolist` | CPU-DEVICE 边界 | DEVICE tensor 上使用会触发同步 |

## 2. 常见 OP 用法

### 2.1 查表与索引：`embedding`、advanced indexing

token id 是整数。embedding 查表把 token id 转成 hidden states。

```python
import torch

input_ids = torch.tensor([1, 3, 2, 4])
embedding_table = torch.tensor([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, 1.0],
])

# 每个 token id 选择 embedding_table 中的一行。
hidden = embedding_table[input_ids]
```

输出：

```text
input_ids shape=(4,), value=tensor([1, 3, 2, 4])
hidden    shape=(4, 2), value=tensor([[1., 0.],
                                      [1., 1.],
                                      [0., 1.],
                                      [2., 1.]])
```

DeepSeekV4 对应代码：

```text
DeepseekV4Model.forward
inputs_embeds = self.embed_tokens(input_ids)
```

`input_ids` 只负责查表。attention、mHC、MoE 处理的是查表后的 `hidden_states`。

### 2.2 Shape 与 Layout：`view`、`transpose`、`contiguous`

`view` 改 shape。`transpose` 交换维度，多数情况下不复制数据，但会改变 stride。`contiguous` 把非连续 layout 复制成连续内存。

```python
import torch

# 原始 shape: [batch=2, seq=2, hidden=4]。
q = torch.arange(1, 17, dtype=torch.float32).view(2, 2, 4)

# 交换 seq 与 hidden 维度；该操作通常只改变 stride。
q_t = q.transpose(1, 2)

# 将非连续 view 复制成连续内存。
q_contiguous = q_t.contiguous()
```

输出：

```text
q.transpose(1, 2) shape=(2, 4, 2)
transpose is_contiguous=False
after contiguous is_contiguous=True
```

DeepSeekV4 对应代码：

```text
DeepseekV4Attention.forward:
q_b_proj(...).view(...).transpose(1, 2)
kv_proj(...).view(...).transpose(1, 2)

DeepseekV4GroupedLinear.forward:
x.reshape(...).transpose(...)
weight.view(...).transpose(...)
```

Transformer 源码中大量 shape 变化服务于 head 维度组织。先写出每一步 shape，再看是否引入非连续 layout 或额外 copy。

### 2.3 线性代数：`F.linear`、`matmul`、`bmm`

线性投影和矩阵乘是 Transformer 的主要计算来源。Q/K/V 投影、attention score、MLP、LM head、expert 计算都依赖这类 OP。

```python
import torch
import torch.nn.functional as F

hidden = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 1.0]])
weight = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])

# F.linear 对最后一维做线性投影。
linear_out = F.linear(hidden, weight)

# matmul 计算两组向量之间的相似度。
attn_scores = hidden @ hidden.T

# bmm 对 batch 中的每组矩阵分别做矩阵乘。
grouped_x = torch.arange(1, 9, dtype=torch.float32).view(2, 2, 2)
grouped_w = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
bmm_out = torch.bmm(grouped_x, grouped_w)
```

输出：

```text
F.linear(hidden, weight) shape=(4, 3)
hidden @ hidden.T        shape=(4, 4)
torch.bmm(grouped_x, w)  shape=(2, 2, 1)
```

DeepSeekV4 对应代码：

```text
DeepseekV4Attention:
q_a_proj, q_b_proj, kv_proj, o_b_proj

DeepseekV4Indexer:
torch.matmul(q, compressed_kv.T)

DeepseekV4GroupedLinear:
torch.bmm(x, w)

DeepseekV4Experts:
F.linear(hidden_states[token_idx], expert_weight)
```

检查重点：shape 是否对齐，dtype 是否匹配，layout 是否满足 GEMM 路径，MoE token 分发后的 batch size 是否合理。

### 2.4 Elementwise 与 Reduction：`square`、`mean`、`rsqrt`、`mul`

RMSNorm 拆成四类 OP：逐元素平方、均值、倒数平方根、逐元素乘法。

```python
import torch

hidden = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 1.0]])

# RMSNorm 的 reference 表达：先计算均方，再用 rsqrt 得到缩放因子。
rms = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-6)
```

DeepSeekV4 对应代码：

```text
DeepseekV4RMSNorm
DeepseekV4UnweightedRMSNorm
q_a_norm / q_b_norm / kv_norm / input_layernorm / post_attention_layernorm
```

reference 写法适合理解和验证。高性能实现通常把 norm 相关 OP 融合到专用 kernel。

### 2.5 RoPE：`repeat_interleave`、slice、`cat`

DeepSeekV4 的 RoPE 只旋转 head 的最后一段维度。前面的 `nope` 维度保持不变。

```python
import torch

rope_input = torch.tensor([[10.0, 20.0, 1.0, 2.0]])
cos = torch.tensor([[0.8]])
sin = torch.tensor([[0.6]])

def rotate_half(x):
    left = x[..., 0::2]
    right = x[..., 1::2]
    return torch.stack((-right, left), dim=-1).flatten(-2)

# cos/sin 扩展到 rope 维度；前两维 nope 不参与旋转。
cos_full = cos.repeat_interleave(2, dim=-1)
sin_full = sin.repeat_interleave(2, dim=-1)
nope, rope = rope_input[..., :-2], rope_input[..., -2:]
rotated_rope = rope * cos_full + rotate_half(rope) * sin_full
rope_out = torch.cat([nope, rotated_rope], dim=-1)
```

DeepSeekV4 对应代码：

```text
apply_rotary_pos_emb:
cos.repeat_interleave(...)
sin.repeat_interleave(...)
nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
torch.cat([nope, rotated], dim=-1)
```

张量结构：

```text
head = [nope | rope]
```

`rope` 部分参与旋转，`nope` 部分直接保留，最后用 `cat` 拼回完整 head。

### 2.6 压缩：`view`、`softmax`、加权 `sum`

压缩模块把一段 token 的 KV 表示聚合成一个 compressed entry。

```python
import torch

tokens = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
gate = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0], [0.0, 2.0]])

# 每 2 个 token 组成一个窗口。
windows = tokens.view(2, 2, 2)
gate_windows = gate.view(2, 2, 2)

# softmax 得到窗口内部权重，再进行加权求和。
weights = gate_windows.softmax(dim=1)
compressed = (windows * weights).sum(dim=1)
```

DeepSeekV4 对应代码：

```text
DeepseekV4HCACompressor.forward:
chunk_kv.view(batch, n_windows, compress_rate, -1)
chunk_gate.softmax(dim=2)
(chunk_kv * weights).sum(dim=2)

DeepseekV4CSACompressor.forward:
new_kv / new_gate 构造 Ca/Cb overlap
softmax + weighted sum 得到 compressed entry
```

HCA 使用较高压缩率生成长程摘要。CSA 使用重叠窗口生成摘要，并交给 indexer 选择相关 entries。

### 2.7 Mask：`arange`、比较、`masked_fill`

attention mask 用 `0` 表示可见位置，用 `-inf` 阻断不可见位置。

```python
import torch

seq_len = 4
compressed_len = 2
compress_rate = 2
position_ids = torch.arange(seq_len)
entry_indices = torch.arange(compressed_len)
causal_threshold = (position_ids + 1) // compress_rate
block_bias = torch.zeros(seq_len, compressed_len)

# entry index 大于等于当前 token 可见阈值时，将对应位置置为 -inf。
block_bias = block_bias.masked_fill(
    entry_indices.view(1, -1) >= causal_threshold.view(-1, 1),
    float("-inf"),
)
```

DeepSeekV4 对应代码：

```text
DeepseekV4HCACompressor.forward:
entry_indices = torch.arange(compressed_len)
causal_threshold = (position_ids + 1) // compress_rate
block_bias.masked_fill(..., -inf)

DeepseekV4CSACompressor.forward:
block_bias.scatter_(...)
```

mask 不参与训练。它保证因果性，并限制 compressed branch 只能访问合法摘要。

### 2.8 MoE 路由：`topk`、`gather`、`one_hot`、`where`、`index_add_`

MoE 的流程是：router 打分，选择专家，专家计算，按 token 原位置合并结果。

```python
import torch
import torch.nn.functional as F

scores = torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.3, 0.4], [0.2, 0.5, 0.7]])

# 每个 token 选择得分最高的 2 个专家。
topk_values, topk_indices = scores.topk(k=2, dim=-1)
weights = topk_values / topk_values.sum(dim=-1, keepdim=True)

# one_hot 生成 expert -> token 的选择关系。
mask = F.one_hot(topk_indices, num_classes=3).permute(2, 1, 0)
final = torch.zeros(3, 2)
```

DeepSeekV4 对应代码：

```text
DeepseekV4TopKRouter.forward:
logits = F.linear(flat, self.weight)
scores = self.score_fn(logits)
indices = torch.topk(scores + bias, top_k).indices
weights = scores.gather(1, indices)

DeepseekV4HashRouter.forward:
indices = self.tid2eid[input_ids.reshape(-1)]
weights = scores.gather(1, indices)

DeepseekV4Experts.forward:
F.one_hot(...).permute(...)
torch.where(...)
final.index_add_(...)
```

`topk` 和 `gather` 决定 token 进入哪些专家。`where` 找出每个专家对应的 token。`index_add_` 把专家输出加回原 token 位置。

## 3. DeepSeekV4 源码主线

### 3.1 `DeepseekV4Model.forward`

| 代码动作 | OP 类型 | 作用 |
| --- | --- | --- |
| `self.embed_tokens(input_ids)` | indexing/embedding | token id 转 hidden states |
| `torch.arange(...)` | 创建/序列 | 生成 position ids |
| `create_sliding_window_causal_mask(...)` | mask | 生成滑窗 causal mask |
| `inputs_embeds.unsqueeze(2).expand(...).contiguous()` | shape/layout/copy | 扩展为 `hc_mult` 条残差流 |
| `self.rotary_emb(...)` | RoPE | 预计算 main/compress 两套 cos/sin |
| `for layer in self.layers` | 控制流 | 逐层执行 DecoderLayer |

执行流程：

```text
input_ids -> embedding -> 多残差流扩展 -> decoder layers -> 末层 hidden states
```

### 3.2 `DeepseekV4DecoderLayer.forward`

| 代码动作 | OP 类型 | 作用 |
| --- | --- | --- |
| `self.attn_hc(hidden_states)` | linear/reduction/matmul | 将多条残差流合成为 attention 输入 |
| `self.input_layernorm(collapsed)` | elementwise/reduction | attention 前归一化 |
| `self.self_attn(...)` | mixed | 计算局部与压缩上下文 |
| `torch.matmul(comb.T, hidden_states)` | matmul | mHC 混合残差流 |
| `self.ffn_hc(hidden_states)` | linear/reduction/matmul | 将多条残差流合成为 MoE 输入 |
| `self.mlp(...)` | topk/GEMM/index_add | 执行专家路由与专家计算 |

这一层包含两次 mHC：一次服务 attention，一次服务 MoE。mHC 决定多条残差流如何合并，以及子层输出如何写回。

### 3.3 `DeepseekV4Attention.forward`

| 阶段 | 关键 OP | 作用 |
| --- | --- | --- |
| Q 投影 | `Linear -> view -> transpose -> norm -> RoPE` | 生成 query heads |
| KV 投影 | `Linear -> view -> transpose -> norm -> RoPE` | 生成单 KV head，K 与 V 共享来源 |
| cache 更新 | `past_key_values.update` | 保存滑窗 KV |
| 压缩分支 | HCA/CSA compressor | 生成 compressed KV |
| KV 拼接 | `torch.cat([kv, compressed_kv], dim=2)` | 拼接本地 KV 与长程摘要 KV |
| mask 拼接 | `torch.cat([attention_mask, block_bias], dim=-1)` | 约束 compressed KV 可见性 |
| attention | `eager_attention_forward` | 执行 attention 计算 |
| 输出投影 | grouped `bmm` + `Linear` | 将多头输出映射回 hidden size |

attention 的输入由两部分组成：

```text
局部滑窗 KV + 压缩长程 KV
```

对应的 mask 也分成两部分：

```text
滑窗 causal mask + compressed branch block_bias
```

### 3.4 `DeepseekV4HCACompressor`

| OP | 作用 |
| --- | --- |
| `kv_proj/gate_proj` | 为每个 token 生成候选 KV 与 gate |
| `view(batch, n_windows, compress_rate, -1)` | 按固定窗口分组 |
| `softmax(dim=2)` | 得到窗口内部权重 |
| `(chunk_kv * weights).sum(dim=2)` | 将窗口压缩为 compressed entry |
| `torch.arange` | 生成 compressed entry 的位置 |
| RoPE | 为 compressed entry 添加位置信息 |
| `masked_fill(-inf)` | 构造 compressed branch causal mask |

HCA 将多个 token 压缩为一个长程摘要。后续 token 只能访问已经完成、满足因果约束的摘要。

### 3.5 `DeepseekV4CSACompressor` 与 `DeepseekV4Indexer`

CSA 做两件事：构造重叠压缩窗口，按 query 选择 top-k compressed entries。

| 模块 | OP | 作用 |
| --- | --- | --- |
| CSA compressor | `view / new_zeros / new_full / slice assign` | 构造 Ca/Cb overlap 窗口 |
| CSA compressor | `softmax + weighted sum` | 生成 compressed KV |
| Indexer | `q_b_proj / weights_proj` | 生成 indexer query 和 per-head 权重 |
| Indexer | `matmul(q, compressed_kv.T)` | query 与 compressed key 打分 |
| Indexer | `relu / sum / topk` | 选择 top-k compressed entries |
| CSA mask | `where / scatter_` | 只允许访问 indexer 选中的 entries |

### 3.6 `DeepseekV4HyperConnection`

| OP | 作用 |
| --- | --- |
| `flatten(start_dim=2)` | 将 `[B,S,hc,D]` 展为 `[B,S,hc*D]` |
| `F.linear(flat, fn)` | 生成 pre、post、comb logits |
| `split` | 拆分 pre、post、comb |
| `sigmoid / softmax` | 生成混合权重 |
| 循环归一化 | Sinkhorn 风格归一化，使 comb 接近双随机矩阵 |
| `(pre * hidden_streams).sum(dim=2)` | 多条残差流合成单条子层输入 |
| `torch.matmul(comb.T, hidden_streams)` | 多条残差流之间混合 |

mHC 负责两类混合：子层输入前的残差流合并，子层输出后的残差流回写。

### 3.7 `DeepseekV4SparseMoeBlock`

| 模块 | OP | 作用 |
| --- | --- | --- |
| TopKRouter | `F.linear` | router 打分 |
| TopKRouter | `topk` | 选择专家 |
| TopKRouter | `gather` | 取被选专家权重 |
| HashRouter | `tid2eid[input_ids]` | 按 token id 查表选专家 |
| Experts | `one_hot / where` | 找出每个 expert 对应的 token |
| Experts | `F.linear` | expert gate/up/down 计算 |
| Experts | `clamp / silu / mul` | SwiGLU 激活 |
| Experts | `index_add_` | 将 expert 输出累加回 token 位置 |

MoE 主线：

```text
router 打分 -> topk 选专家 -> expert 计算 -> index_add_ 合并
```

### 3.8 `DeepseekV4ForCausalLM.forward`

| OP | 作用 |
| --- | --- |
| `self.model(...)` | 获取最后 hidden states |
| slice `hidden_states[:, slice_indices, :]` | 只保留需要计算 logits 的 token |
| `lm_head` / `Linear` | hidden states 转 vocab logits |
| loss function | `labels` 非空时计算语言模型 loss |
| load balancing loss | `output_router_logits` 时计算 aux loss |

推理通常只用最后一个 token 的 logits。`logits_to_keep` 用于减少不必要的 vocab projection。

## 4. DeepSeekV4 源码分析步骤

分析 DeepSeekV4 源码时，先确认张量形状和 OP 类型，再对应到模型模块。

| 源码模块 | 主要 OP | 作用 |
| --- | --- | --- |
| `DeepseekV4Model.forward` | embedding、arange、mask、expand、contiguous | 准备输入、position 和多残差流 |
| `apply_rotary_pos_emb` | repeat_interleave、slice、rotate、cat | 旋转 head 的 RoPE 子空间 |
| `DeepseekV4Attention` | linear、view、transpose、cat、pad、attention、bmm | 组合滑窗 KV 与压缩 KV |
| `DeepseekV4HCACompressor` | linear、view、softmax、sum、masked_fill | 将多 token 压缩为长程摘要 |
| `DeepseekV4CSACompressor` | slice assign、softmax、sum、scatter | 构造重叠窗口摘要和访问 mask |
| `DeepseekV4Indexer` | linear、matmul、relu、topk、where | 为 CSA 选择相关摘要 |
| `DeepseekV4HyperConnection` | flatten、linear、split、sigmoid、softmax、matmul | 混合多条残差流 |
| `DeepseekV4SparseMoeBlock` | topk、gather、one_hot、where、linear、index_add | token 选专家并合并专家输出 |
| `DeepseekV4ForCausalLM` | linear、slice、loss | hidden states 转 logits |

单行代码拆解示例：

```python
q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
```

拆解结果：

```text
Linear   : [B,S,q_lora_rank] -> [B,S,num_heads*head_dim]
view     : [B,S,num_heads,head_dim]
transpose: [B,num_heads,S,head_dim]
```

检查顺序：

1. 写出输入输出 shape。
2. 标注 OP 类型：layout、GEMM、elementwise、reduction、indexing、mask、routing。
3. 检查 layout：是否连续，是否调用 `contiguous()`。
4. 检查内存行为：是否复制，是否分配临时 tensor。
5. 检查设备边界：是否出现 `.item()`、`.cpu()`、`.tolist()`。
6. 映射到模型模块：attention、compression、mHC、MoE、logits。

## 5. 结论

DeepSeekV4 源码能拆成一组普通 PyTorch OP：

```text
源码 API
  -> OP 行为：计算 / layout / 索引 / 更新 / 同步
  -> 模型语义：attention / compression / mHC / MoE / logits
```

先看 OP 行为，再看模型语义。这种方法能把复杂模块拆成可验证的张量操作，也能继续向下分析 kernel 命中、layout copy、dynamic shape、Graph replay 和 MUSA 后端行为。
