# PyTorch 常见 OP 与 Transformer 源码分析：Llama 到 DeepSeek V4

PyTorch 模型源码可以拆成一组张量 OP。阅读 Transformer 源码时，先确认 OP 对 `shape / dtype / device / stride` 的影响，再判断它是否触发计算、内存分配、数据复制、动态 shape 或同步等待。完成这一步后，再把 OP 映射回 Attention、MLP、MoE、KV cache 和 Graph replay。

本文按三层展开：

1. PyTorch 常见 OP 的行为、用法和最小例子。
2. 以 Transformers 中的 Llama 源码拆解经典 decoder-only 模型。
3. 以 DeepSeek V4 源码说明压缩注意力、多残差流和 MoE 的进阶结构。

参考源码：

```text
repos/transformers/src/transformers/models/llama/modeling_llama.py
repos/transformers/src/transformers/models/deepseek_v4/modular_deepseek_v4.py
repos/transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py
```

`modular_deepseek_v4.py` 是模块化维护入口，`modeling_deepseek_v4.py` 是生成后的完整运行文件。源码分析时优先阅读模块化文件，定位完整调用链时再对照生成文件。

## 1. PyTorch OP 的运行行为

PyTorch API 名称只能说明上层语义，不能直接说明底层成本。源码分析时应直接检查输入布局、输出布局、设备边界和执行位置。

| 行为类别 | 常见 OP | 底层表现 | 主要检查点 |
|---|---|---|---|
| Metadata 更新 | `view`、`unsqueeze`、`squeeze`、合法 `reshape` | 不提交 kernel，只改变 shape/stride 描述 | stride 是否兼容；后续 kernel 是否接受非连续输入 |
| 数据复制 | `contiguous`、stride 不兼容的 `reshape`、`to(device)` | D2D、H2D 或 D2H copy | 是否处于 decode 单步或 Graph replay 高频路径 |
| Tensor 创建 | `empty`、`zeros`、`full`、`new_empty`、`cat`、`stack` | allocator 活跃，可能伴随 memset 或 copy | 地址是否稳定；是否能预分配并复用 |
| 计算提交 | `linear`、`matmul`、`softmax`、`topk`、`index_add_` | GEMM、归约、排序、scatter 或 fused kernel | dtype、layout、shape 是否命中目标 kernel |
| 动态 shape | `nonzero`、`unique`、`masked_select`、boolean indexing | 输出长度依赖数据内容 | 是否影响 Graph replay、workspace 和通信 bucket |
| 同步边界 | `.item()`、`.tolist()`、`.cpu()`、`synchronize`、`work.wait()` | CPU、stream 或 rank 等待 | 是否打断异步执行或通信计算重叠 |
| Graph 执行 | `MUSAGraph.replay`、CUDA Graph replay | 重放固定地址、固定 shape 和固定 kernel 序列 | capture 内是否存在分配、CPU 回读或不支持的 backend |

性能问题通常不是由某个 API 名称单独决定，而是由具体输入状态触发。典型例子包括：

- `reshape` 在 stride 兼容时返回 view，在 stride 不兼容时可能复制数据。
- `contiguous()` 对连续 tensor 通常不复制，对非连续 tensor 会创建连续副本。
- DEVICE tensor 的 `.item()` 必须等待设备端结果可见，再把标量读回 CPU。
- `unique / nonzero / masked_select` 的输出长度随数据变化，不应直接放入固定 shape 的 Graph replay。

## 2. 常见 OP 用法

### 2.1 Shape 与 Layout

常见 OP：

```text
view / reshape / flatten / unsqueeze / squeeze / expand / transpose / permute / contiguous
```

这些 OP 用于组织 Q/K/V head、RoPE 输入、KV cache layout 和 MoE expert buffer。分析时先写出 shape 和 stride，再判断是否发生实际复制。

最小例子：

```python
import torch

# x 是连续 tensor，shape=[2,3,4]，stride=(12,4,1)。
x = torch.arange(24).view(2, 3, 4)

# transpose 交换 seq 与 hidden 维度。该操作通常只改 metadata。
y = x.transpose(1, 2)

# contiguous 将非连续 view 复制为连续内存。
z = y.contiguous()

print("x", tuple(x.shape), x.stride(), x.is_contiguous())
print("y", tuple(y.shape), y.stride(), y.is_contiguous())
print("z", tuple(z.shape), z.stride(), z.is_contiguous())
```

输出：

```text
x (2, 3, 4) (12, 4, 1) True
y (2, 4, 3) (12, 1, 4) False
z (2, 4, 3) (12, 3, 1) True
```

源码检查：

```text
LlamaAttention.forward:
query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

DeepseekV4Attention.forward:
q_b_proj(...).view(...).transpose(1, 2)
kv_proj(...).view(...).transpose(1, 2)
```

检查结论：

- `linear -> view -> transpose` 是 attention head layout 的常见路径。
- `transpose` 后的 tensor 通常非连续；后续 kernel 若要求连续输入，需要显式处理。
- `expand` 可能产生 zero-stride view，不应用作原地写入目标。

### 2.2 Tensor 创建与原地更新

常见 OP：

```text
empty / new_empty / empty_like / zeros / ones / full
copy_ / fill_ / zero_ / masked_fill_ / clamp_
```

这些 OP 常见于 KV cache、Graph input buffer、临时 workspace、padding 清理和 logits mask。

Graph replay 的固定 buffer 例子：

```python
import torch

# Graph replay 记录的是固定 tensor 对象和底层地址。
# 这里用 CPU tensor 模拟固定 buffer 的内容更新语义。
graph_input = torch.zeros(2)

# 第一次执行读取 graph_input 的当前内容。
out0 = graph_input * 3 + 2

# replay 前只更新已有 buffer 内容，不替换 graph_input 对象。
graph_input.copy_(torch.tensor([4.0, 5.0]))
out1 = graph_input * 3 + 2

print("out0 =", out0)
print("out1 =", out1)
```

输出：

```text
out0 = tensor([2., 2.])
out1 = tensor([14., 17.])
```

工程约束：

- Graph capture 前创建 input、output 和 metadata tensor。
- replay 前使用 `copy_ / fill_ / zero_` 更新内容。
- 不在 replay 路径中替换已 capture 的 tensor 对象。
- padding 槽位必须清理，避免上轮残留影响 mask、cache 或 logits。

### 2.3 数学、归约与激活

常见 OP：

```text
sum / mean / max / min / pow / square / rsqrt
sigmoid / silu / gelu / relu / softmax / log_softmax / clamp
```

它们构成 RMSNorm、SwiGLU、attention softmax、MoE gate 和 sampling。

RMSNorm 最小例子：

```python
import torch

h = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

# RMSNorm 在 hidden 维度计算均方。
variance = h.pow(2).mean(-1, keepdim=True)

# rsqrt(x) 等价于 1 / sqrt(x)。
out = h * torch.rsqrt(variance + 1e-6)

print("variance =", variance)
print("out =", out)
```

输出：

```text
variance = tensor([[7.5000]])
out = tensor([[0.3651, 0.7303, 1.0954, 1.4606]])
```

Llama RMSNorm 的源码结构：

```python
# 使用 float32 计算 variance，降低归一化误差。
hidden_states = hidden_states.to(torch.float32)

# 在 hidden 维度计算均方。
variance = hidden_states.pow(2).mean(-1, keepdim=True)

# 按 RMS 缩放 hidden states。
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

# 乘可学习权重，并转回输入 dtype。
return self.weight * hidden_states.to(input_dtype)
```

性能检查：

- reference RMSNorm 可能包含多个 kernel：`pow -> mean -> rsqrt -> mul`。
- 在线热点路径通常使用 fused RMSNorm kernel。
- `softmax(dtype=torch.float32)` 有助于数值稳定，但会引入 dtype 转换。

### 2.4 线性代数与 Attention

常见 OP：

```text
F.linear / matmul / mm / bmm / einsum
softmax / scaled_dot_product_attention
```

`F.linear(x, weight, bias)` 等价于：

```text
x @ weight.T + bias
```

Transformer 中的主要用途：

- Q/K/V projection。
- Attention score 与 value 聚合。
- Attention output projection。
- MLP gate/up/down projection。
- MoE router、expert GEMM 和 LM head。

Attention 最小例子：

```python
import torch

# query/key/value shape=[B=1, H=1, S=2, D=2]。
query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
key = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])
value = torch.tensor([[[[2.0, 0.0], [0.0, 3.0]]]])

# q @ k^T 得到每个 query 对每个 key 的注意力分数。
scores = torch.matmul(query, key.transpose(2, 3)) * (2 ** -0.5)

# softmax 将分数转换为概率。
probs = torch.softmax(scores, dim=-1)

# 概率加权 value，得到 attention 输出。
out = torch.matmul(probs, value)

print("scores =", scores)
print("probs =", probs)
print("out =", out)
```

输出：

```text
scores = tensor([[[[0.7071, 0.7071],
                   [0.0000, 0.7071]]]])
probs = tensor([[[[0.5000, 0.5000],
                  [0.3302, 0.6698]]]])
out = tensor([[[[1.0000, 1.5000],
                [0.6605, 2.0095]]]])
```

性能检查：

- `matmul -> softmax -> matmul` 是 reference attention 结构，score tensor 可能较大。
- 实际推理通常使用 SDPA、FlashAttention 或后端自定义 fused attention。
- dtype、mask layout、KV cache layout 和 sequence length 会决定后端选择。

### 2.5 索引、路由与动态 Shape

常见 OP：

```text
gather / scatter_ / index_select / index_add_ / where
topk / unique / nonzero / masked_select / boolean indexing
```

它们常见于 MoE routing、KV page 选择、sampling、mask 构造和统计。

MoE top-k 最小例子：

```python
import torch

# 两个 token，三个 expert。
scores = torch.tensor([
    [0.1, 0.7, 0.2],
    [0.6, 0.1, 0.3],
])

# 每个 token 选择分数最高的两个 expert。
values, indices = torch.topk(scores, 2, dim=-1)

# 对被选 expert 的权重归一化。
weights = values / values.sum(-1, keepdim=True)

print("indices =", indices)
print("weights =", weights)
```

输出：

```text
indices = tensor([[1, 2],
                  [0, 2]])
weights = tensor([[0.7778, 0.2222],
                  [0.6667, 0.3333]])
```

动态 shape 风险：

| OP | 风险 | 稳定化方式 |
|---|---|---|
| `nonzero` | 输出长度等于满足条件的元素数 | fixed mask + padding |
| `unique` | 输出长度等于唯一值个数 | histogram 或 `bincount(minlength=...)` |
| `masked_select` | 输出长度等于 `True` 个数 | 保持原 shape，用 mask 控制有效位 |
| boolean indexing | 输出长度依赖数据内容 | padded index + sentinel |
| `where(mask)` 单参数形式 | 返回坐标长度依赖 mask | 双参数形式保持原 shape |

以 `torch_musa` 当前实现为例，`torch.unique(musa_tensor)` 会触发当前 stream 同步。普通 dtype 路径需要从设备端回读输出长度，bool 路径内部使用 `memcpy_and_sync` 并调用 `musaStreamSynchronize(stream)`。在线推理路径应避免用设备端动态长度结果驱动 Python 分支。

### 2.6 CPU 边界、dtype 与后端路径

常见 CPU 边界：

```text
.item() / .tolist() / .cpu() / .numpy()
torch.cuda.synchronize() / torch.musa.synchronize()
stream.wait_event() / distributed work.wait()
```

使用原则：

- benchmark、错误定位、进程退出前可以使用显式同步。
- decode 热点路径优先使用 stream/event 局部依赖。
- scheduler 需要的长度、bucket、request 状态，优先维护 CPU 元数据副本。
- 最终 token、少量 logprob 或少量统计允许回读；完整 logits、hidden states、KV metadata 不应频繁回 CPU。

dtype 和 device 相关 OP：

```text
to / float / half / bfloat16 / int / long
```

检查项：

- Norm 内部是否只在局部升精度，输出是否 cast 回模型 dtype。
- index dtype 是否满足后端 kernel 要求。
- `to(device)` 是否引入 H2D、D2H 或跨设备 copy。
- 量化路径中 activation、weight、scale 的 dtype、shape、stride 是否匹配目标 kernel。

## 3. Llama Decoder 源码拆解

Llama 是标准 decoder-only 模型。主执行链路：

```text
input_ids
  -> embedding
  -> position_ids / causal_mask / rotary embedding
  -> N 个 LlamaDecoderLayer
       -> RMSNorm
       -> Self Attention
       -> residual add
       -> RMSNorm
       -> MLP
       -> residual add
  -> final RMSNorm
  -> lm_head
  -> logits
```

prefill 和 decode 的热点不同：

| 阶段 | 输入特点 | Attention 行为 | 性能重点 |
|---|---|---|---|
| prefill | 一次输入多个 prompt token | 当前 token 之间做 causal attention，并写入 KV cache | GEMM、attention 带宽、KV cache 初始化 |
| decode | 每步新增一个 token | 新 query 读取历史 KV cache，追加当前 K/V | kernel launch、Graph replay、KV cache 访问、CPU 同步边界 |

### 3.1 `LlamaModel.forward`

源码位置：

```text
modeling_llama.py::LlamaModel.forward
```

关键路径：

```python
# input_ids 是 token id，embedding 后变成 hidden states。
inputs_embeds = self.embed_tokens(input_ids)

# position_ids 表示 token 在序列中的位置。
position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
position_ids = position_ids.unsqueeze(0)

# causal_mask 禁止当前位置访问未来 token。
causal_mask = create_causal_mask(...)

# RoPE 的 cos/sin 在模型层统一生成，并传给每一层 attention。
position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

for decoder_layer in self.layers:
    hidden_states = decoder_layer(
        hidden_states,
        attention_mask=causal_mask,
        position_embeddings=position_embeddings,
        past_key_values=past_key_values,
    )

hidden_states = self.norm(hidden_states)
```

对应 OP：

| 源码动作 | 主要 OP | 行为 |
|---|---|---|
| token 查表 | `nn.Embedding` | 按 id 读取 embedding 表 |
| 生成位置 | `arange`、`unsqueeze` | 创建 position tensor |
| 构造 mask | `full`、`triu`、`masked_fill`、加法 | 形成 causal bias |
| RoPE | `cos`、`sin`、slice、cat | 生成旋转位置编码 |
| decoder 循环 | Python loop + module call | 每层重复 attention 和 MLP |

Causal mask 最小例子：

```python
import torch

# scores 表示 3 个 query token 对 3 个 key token 的原始注意力分数。
scores = torch.tensor([
    [0.2, 0.1, 0.4],
    [0.3, 0.5, 0.0],
    [0.2, 0.4, 0.6],
])

# 上三角位置是未来 token，填入 -inf 后 softmax 概率为 0。
mask = torch.triu(torch.full((3, 3), float("-inf")), diagonal=1)
masked_scores = scores + mask
probs = torch.softmax(masked_scores, dim=-1)

print("masked_scores =", masked_scores)
print("probs =", probs)
```

输出：

```text
masked_scores = tensor([[0.2000,   -inf,   -inf],
                        [0.3000, 0.5000,   -inf],
                        [0.2000, 0.4000, 0.6000]])
probs = tensor([[1.0000, 0.0000, 0.0000],
                [0.4502, 0.5498, 0.0000],
                [0.2693, 0.3289, 0.4018]])
```

### 3.2 `LlamaDecoderLayer.forward`

源码位置：

```text
modeling_llama.py::LlamaDecoderLayer.forward
```

Llama decoder layer 是 pre-norm block：

```text
x -> RMSNorm -> Attention -> residual add
  -> RMSNorm -> MLP       -> residual add
```

关键路径：

```python
# 保存 attention 前的输入，用于 residual add。
residual = hidden_states

# 先归一化，再进入 self-attention。
hidden_states = self.input_layernorm(hidden_states)
hidden_states, _ = self.self_attn(...)
hidden_states = residual + hidden_states

# 保存 MLP 前的输入，用于第二条 residual add。
residual = hidden_states

# 先归一化，再进入 MLP。
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states
```

对应 OP：

| 阶段 | 主要 OP | 作用 |
|---|---|---|
| RMSNorm | `to(float32)`、`pow`、`mean`、`rsqrt`、`mul` | 对 hidden 维做归一化 |
| Attention | `linear`、`view`、`transpose`、RoPE、attention kernel | 建模 token 间依赖 |
| Residual | `add` | 保留原始信息，稳定深层网络 |
| MLP | `linear`、`silu`、`mul`、`linear` | token 内非线性变换 |

### 3.3 `LlamaAttention.forward`

源码位置：

```text
modeling_llama.py::LlamaAttention.forward
```

关键路径：

```python
input_shape = hidden_states.shape[:-1]
hidden_shape = (*input_shape, -1, self.head_dim)

# Q/K/V projection，并整理成多头布局。
query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

# RoPE 只作用于 Q/K。
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

# decode 时更新 KV cache。
if past_key_values is not None:
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

# 后端可选择 eager、SDPA、FlashAttention 或框架注册实现。
attn_output, attn_weights = attention_interface(...)

# 回到 [batch, seq, hidden]，再做 output projection。
attn_output = attn_output.reshape(*input_shape, -1).contiguous()
attn_output = self.o_proj(attn_output)
```

Q projection 的 shape 例子：

```python
import torch
import torch.nn.functional as F

# hidden shape=[B=1, S=2, D=4]。
hidden = torch.tensor([[
    [1.0, 0.0, 2.0, 1.0],
    [0.0, 1.0, 1.0, 2.0],
]])

# 使用单位矩阵作为 projection weight，便于核对 shape。
q_weight = torch.eye(4)

# projection 后 view 成 [B,S,H,Dh]，再 transpose 成 [B,H,S,Dh]。
q = F.linear(hidden, q_weight).view(1, 2, 2, 2).transpose(1, 2)

print("q shape =", tuple(q.shape))
print(q)
```

输出：

```text
q shape = (1, 2, 2, 2)
tensor([[[[1., 0.],
          [0., 1.]],

         [[2., 1.],
          [1., 2.]]]])
```

shape 转换：

```text
[B,S,D]
  -> linear
  -> [B,S,H*Dh]
  -> view
  -> [B,S,H,Dh]
  -> transpose
  -> [B,H,S,Dh]
```

### 3.4 `LlamaMLP.forward`

源码位置：

```text
modeling_llama.py::LlamaMLP.forward
```

关键路径：

```python
# gate_proj 产生门控分支，up_proj 产生内容分支。
gate = self.gate_proj(x)
up = self.up_proj(x)

# SwiGLU: 对 gate 做 SiLU 后与 up 分支逐元素相乘。
hidden = self.act_fn(gate) * up

# down_proj 投回 hidden size。
out = self.down_proj(hidden)
```

SwiGLU 最小例子：

```python
import torch
import torch.nn.functional as F

gate = torch.tensor([[1.0, -1.0]])
up = torch.tensor([[2.0, 3.0]])

# SiLU(gate) = gate * sigmoid(gate)。
mix = F.silu(gate) * up

print("mix =", mix)
```

输出：

```text
mix = tensor([[ 1.4621, -0.8068]])
```

### 3.5 `LlamaForCausalLM.forward`

源码位置：

```text
modeling_llama.py::LlamaForCausalLM.forward
```

关键路径：

```python
# base decoder 输出每个 token 的 hidden states。
outputs = self.model(...)
hidden_states = outputs[0]

# decode 通常只需要最后一个或最后几个 token 的 logits。
hidden_states = hidden_states[:, slice_indices, :]

# lm_head 将 hidden states 投影到 vocabulary 维度。
logits = self.lm_head(hidden_states)
```

检查重点：

- prefill 通常需要多个 token 的 logits。
- decode 通常只需要最后一个 token 的 logits。
- `logits_to_keep` 可以减少不必要的 vocab projection。

## 4. DeepSeek V4 源码进阶

DeepSeek V4 在经典 decoder 结构上增加三类关键机制：

| 机制 | 代表模块 | 主要 OP | 作用 |
|---|---|---|---|
| 多残差流 | `DeepseekV4HyperConnection`、`DeepseekV4HyperHead` | `flatten`、`linear`、`split`、`sigmoid`、`softmax`、`matmul` | 多条 residual stream 的折叠、混合和写回 |
| 压缩注意力 | `DeepseekV4HCACompressor`、`DeepseekV4CSACompressor`、`DeepseekV4Indexer` | `view`、`softmax`、`sum`、`matmul`、`topk`、`scatter_` | 将历史 token 压缩为 compressed KV，并按 query 选择摘要 |
| MoE | `DeepseekV4SparseMoeBlock`、router、experts | `topk`、`gather`、`one_hot`、`where`、`linear`、`index_add_` | token 选择专家，专家计算后合并回 token 位置 |

### 4.1 `DeepseekV4Model.forward`

主线：

```text
input_ids
  -> embed_tokens
  -> hidden_states.unsqueeze(2).expand(...).contiguous()
  -> rotary_emb 生成 main/compress 两套位置编码
  -> N 个 DeepseekV4DecoderLayer
  -> HyperHead 折叠多残差流
```

关键点：

- Llama 中的 hidden states 是 `[B, S, D]`。
- DeepSeek V4 引入 `hc_mult` 后，layer 内部使用 `[B, S, hc_mult, D]`。
- 进入 attention 或 MoE 前，需要由 mHC 折叠为普通 `[B, S, D]`。
- 输出后再按 mHC 权重写回多条 residual stream。

### 4.2 `DeepseekV4HyperConnection`

mHC 负责两类操作：

1. 子层输入前，把多条 residual stream 折叠成一条 hidden。
2. 子层输出后，把 attention 或 MoE 输出写回多条 residual stream，并混合旧 stream。

最小例子：

```python
import torch

# 一个 token，两条 residual stream，每条 hidden_dim=2。
streams = torch.tensor([
    [1.0, 0.0],
    [0.0, 2.0],
])

# pre 是 mHC 生成的折叠权重。这里手工指定，便于核对计算。
pre = torch.tensor([0.25, 0.75])

# collapsed 是送入 attention 或 MoE 的普通 hidden。
collapsed = (pre[:, None] * streams).sum(dim=0)

print("collapsed =", collapsed)
```

输出：

```text
collapsed = tensor([0.2500, 1.5000])
```

写回例子：

```python
import torch

old_streams = torch.tensor([
    [1.0, 0.0],
    [0.0, 2.0],
])

# 子层输出，例如 attention 输出。
sub_layer_out = torch.tensor([0.5, 1.0])

# post 控制子层输出写入每条 stream 的比例。
post = torch.tensor([1.0, 0.5])

# comb 控制旧 residual stream 之间的混合。
comb = torch.tensor([
    [0.8, 0.2],
    [0.1, 0.9],
])

mixed_old = comb.T @ old_streams
new_streams = mixed_old + post[:, None] * sub_layer_out

print("mixed_old =", mixed_old)
print("new_streams =", new_streams)
```

输出：

```text
mixed_old = tensor([[0.8000, 0.2000],
                    [0.1000, 1.8000]])
new_streams = tensor([[1.3000, 1.2000],
                      [0.3500, 2.3000]])
```

源码映射：

```text
DeepseekV4DecoderLayer.forward:
attn_hc -> input_layernorm -> self_attn -> residual stream writeback
ffn_hc  -> post_attention_layernorm -> mlp -> residual stream writeback
```

### 4.3 `DeepseekV4Attention`

DeepSeek V4 attention 的输入由两部分组成：

```text
局部滑窗 KV + 压缩长程 KV
```

对应 mask 也分为两部分：

```text
滑窗 causal mask + compressed branch block_bias
```

关键路径：

```text
Q: q_a_proj -> q_a_norm -> q_b_proj -> view -> transpose -> q_b_norm -> RoPE
KV: kv_proj -> view -> transpose -> kv_norm -> RoPE
Compressed KV: HCA 或 CSA compressor
Concat: cat([kv, compressed_kv], dim=2)
Mask: cat([attention_mask, block_bias], dim=-1)
Attention: attention_interface(...)
Output: grouped bmm / grouped linear -> output projection
```

检查重点：

- Q/K/V 的 head layout 是否满足 attention backend。
- compressed KV 是否与原 KV 在 sequence 维拼接。
- `block_bias` 是否只允许访问因果合法的 compressed entries。
- `cat` 会创建新 tensor，热点路径中应关注内存带宽和 allocator 行为。

### 4.4 `DeepseekV4HCACompressor`

HCA 将连续 token 按固定窗口压缩成 compressed KV。核心操作是：

```text
window 切分 -> gate softmax -> weighted sum -> compressed entry
```

最小例子：

```python
import torch

# 4 个 token，每个 token 的 KV 维度为 2。
tokens = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, 1.0],
])

# 每个 token 对每个维度都有一个 gate logit。
gate = torch.tensor([
    [2.0, 0.0],
    [0.0, 2.0],
    [1.0, 1.0],
    [0.0, 2.0],
])

# 每 2 个 token 压缩成一个 entry。
windows = tokens.view(2, 2, 2)
gate_windows = gate.view(2, 2, 2)

# 在窗口内部做 softmax，再按维度加权求和。
weights = gate_windows.softmax(dim=1)
compressed = (windows * weights).sum(dim=1)

print("weights =", weights)
print("compressed =", compressed)
```

输出：

```text
weights = tensor([[[0.8808, 0.1192],
                   [0.1192, 0.8808]],

                  [[0.7311, 0.2689],
                   [0.2689, 0.7311]]])
compressed = tensor([[0.8808, 0.8808],
                     [1.2689, 1.0000]])
```

计算含义：

- `tokens 0-1` 被压缩成 `compressed[0]`。
- `tokens 2-3` 被压缩成 `compressed[1]`。
- 后续 attention 不直接访问全部历史 token 时，可以访问这些 compressed entries 作为长程摘要。

HCA causal mask 最小例子：

```python
import torch

seq_len = 4
compressed_len = 2
compress_rate = 2

position_ids = torch.arange(seq_len)
entry_indices = torch.arange(compressed_len)

# causal_threshold 表示每个 query token 可见的 compressed entry 数量。
causal_threshold = (position_ids + 1) // compress_rate

block_bias = torch.zeros(seq_len, compressed_len)
block_bias = block_bias.masked_fill(
    entry_indices.view(1, -1) >= causal_threshold.view(-1, 1),
    float("-inf"),
)

print("causal_threshold =", causal_threshold)
print("block_bias =", block_bias)
```

输出：

```text
causal_threshold = tensor([0, 1, 1, 2])
block_bias = tensor([[-inf, -inf],
                     [0., -inf],
                     [0., -inf],
                     [0., 0.]])
```

作用：

- token0 没有完整历史窗口可访问。
- token1 和 token2 可访问 `tokens 0-1` 压缩得到的 entry0。
- token3 可访问 entry0 和 `tokens 2-3` 压缩得到的 entry1。

### 4.5 `DeepseekV4CSACompressor` 与 `DeepseekV4Indexer`

CSA 在压缩后增加 indexer。indexer 根据当前 query 选择 top-k compressed entries，避免每个 query 访问所有压缩摘要。

最小例子：

```python
import torch

# 两个 query，三个 compressed entries，维度为 2。
q = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
])
compressed_k = torch.tensor([
    [1.0, 0.0],
    [0.5, 0.5],
    [0.0, 1.0],
])

# query 与 compressed key 打分。
scores = torch.matmul(q, compressed_k.T).relu()

# 每个 query 选择两个 compressed entries。
values, indices = torch.topk(scores, k=2, dim=-1)

print("scores =", scores)
print("indices =", indices)
```

输出：

```text
scores = tensor([[1.0000, 0.5000, 0.0000],
                 [0.0000, 0.5000, 1.0000]])
indices = tensor([[0, 1],
                  [2, 1]])
```

源码映射：

```text
DeepseekV4Indexer:
q_b_proj / weights_proj
matmul(q, compressed_kv.T)
relu / sum / topk

DeepseekV4CSACompressor:
scatter_ 构造 compressed branch mask
```

### 4.6 `DeepseekV4SparseMoeBlock`

MoE 主线：

```text
hidden_states
  -> router 打分
  -> topk 选 expert
  -> gather 取被选 expert 权重
  -> expert 计算
  -> index_add_ 合并回 token 位置
  -> shared expert 分支相加
```

最小例子：

```python
import torch

# 三个 token，三个 expert 的 router score。
scores = torch.tensor([
    [0.1, 0.7, 0.2],
    [0.6, 0.1, 0.3],
    [0.2, 0.4, 0.8],
])

topk_values, topk_indices = scores.topk(k=2, dim=-1)
weights = topk_values / topk_values.sum(dim=-1, keepdim=True)

# 示例直接给出 expert 对 token 的输出，便于说明 combine。
expert_outputs = torch.tensor([
    [[0.0, 2.0], [1.0, 0.0]],  # token0 对 expert1/expert2 的输出
    [[2.0, 0.0], [0.0, 1.0]],  # token1 对 expert0/expert2 的输出
    [[1.0, 1.0], [0.0, 2.0]],  # token2 对 expert2/expert1 的输出
])

combined = (expert_outputs * weights.unsqueeze(-1)).sum(dim=1)

print("topk_indices =", topk_indices)
print("weights =", weights)
print("combined =", combined)
```

输出：

```text
topk_indices = tensor([[1, 2],
                       [0, 2],
                       [2, 1]])
weights = tensor([[0.7778, 0.2222],
                  [0.6667, 0.3333],
                  [0.6667, 0.3333]])
combined = tensor([[0.2222, 1.5556],
                   [1.3333, 0.3333],
                   [0.6667, 1.3333]])
```

源码映射：

```text
DeepseekV4TopKRouter:
F.linear -> score_fn -> topk -> gather -> normalize

DeepseekV4HashRouter:
tid2eid[input_ids.reshape(-1)] -> gather

DeepseekV4Experts:
one_hot -> where -> per-expert F.linear -> index_add_
```

性能检查：

- `topk` 固定 k 有利于保持后续 shape 稳定。
- `where` 会产生数据相关 token 列表，reference 实现适合验证，不应直接作为高性能 MoE dispatch。
- per-expert Python loop 会增加 launch 数和调度开销。
- 高性能实现通常使用 fused dispatch、grouped GEMM 和 fused combine。

### 4.7 `DeepseekV4ForCausalLM.forward`

主线：

```text
self.model(...)
  -> hidden states
  -> slice logits_to_keep
  -> lm_head
  -> logits
  -> optional loss / aux loss
```

检查重点：

- 推理通常只需要最后 token 的 logits。
- `logits_to_keep` 能减少不必要的 vocab projection。
- 训练路径需要 labels、loss 和 router aux loss；推理路径应避免无关分支进入热点。

## 5. OP 与性能排查

Profiler 现象应回到源码中的具体 OP，而不是停留在模块名。

| 现象 | 源码检查 | 机制 |
|---|---|---|
| 出现额外 memcpy | `reshape / transpose / permute / contiguous / cat / stack / pad / to(device)` | layout 转换或 tensor 拼接触发数据复制 |
| 出现 stream synchronize | `.item() / .tolist() / .cpu() / synchronize / unique / nonzero / masked_select` | CPU 需要等待设备结果或输出长度 |
| allocator 抖动 | `empty / zeros / new_full / cat / dynamic output OPs` | 高频路径分配临时 tensor |
| kernel 数量过多 | reference RMSNorm、SwiGLU、attention、MoE routing | 多个逐元素或归约 OP 分别 launch |
| kernel 名称不符合预期 | `linear / attention / quantized GEMM` 的 dtype、layout、scale、shape | 后端 fallback 到普通路径 |
| Graph replay 失败 | dynamic shape、地址变化、capture 内 allocator、CPU 回读、不支持 backend | 不满足固定 shape 和固定地址约束 |

处理方式：

- 对 layout copy，固定输入 layout 或把 `contiguous()` 前移到低频路径。
- 对 CPU 回读，维护 CPU metadata mirror，避免 DEVICE tensor 驱动 Python 分支。
- 对 allocator 抖动，按 bucket 预分配 workspace，并使用 `copy_ / fill_ / zero_` 更新内容。
- 对 kernel 数量过多，使用 fused norm、fused activation、fused attention、grouped GEMM 或后端专用 MoE kernel。
- 对 Graph replay，固定 shape、固定地址、固定 metadata，并把动态整理放在 Graph 外。

## 6. 源码分析步骤

分析一段 Transformer 源码时，按以下顺序执行：

1. 写出输入和输出 shape。
2. 标注 OP 类型：layout、GEMM、elementwise、reduction、indexing、mask、routing、sync、graph。
3. 检查 `dtype / device / stride / is_contiguous()`。
4. 判断是否分配新 tensor 或触发实际 copy。
5. 判断是否出现 `.item()`、`.tolist()`、`.cpu()`、`synchronize()` 或动态 shape OP。
6. 对应到模型模块：embedding、position、attention、KV cache、MLP、MoE、logits。
7. 对应到 profiler：kernel 名称、memcpy、memset、allocator、stream wait、graph replay。

单行代码拆解示例：

```python
q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
```

拆解结果：

```text
Linear    : [B,S,q_lora_rank] -> [B,S,num_heads*head_dim]
view      : [B,S,num_heads,head_dim]
transpose : [B,num_heads,S,head_dim]
```

该分析能直接回答三个问题：

- 数学语义：生成 attention query。
- 张量变化：从普通 hidden layout 变成多头 attention layout。
- 性能检查：`transpose` 后是否非连续，后续 attention backend 是否支持该 layout。

## 7. 结论

PyTorch 常见 OP 是阅读 Transformer 源码的基础单位。Llama decoder 可以拆成 embedding、position/mask、RMSNorm、attention、MLP、residual 和 lm_head。DeepSeek V4 在该结构上增加 mHC、多种压缩注意力和 MoE，但核心仍然由 `linear / view / transpose / softmax / matmul / topk / gather / scatter / index_add_` 等 OP 组合完成。

源码分析应先判断 OP 的运行行为，再解释模型语义，最后对照 profiler 信号确认底层执行路径。
