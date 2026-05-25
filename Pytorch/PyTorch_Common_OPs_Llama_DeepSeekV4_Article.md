# PyTorch 常见 OP 与 Decoder 源码阅读：从 Llama 到 DeepSeek V4

PyTorch 模型源码由一组 OP 组成。阅读模型代码时，应先判断 OP 的执行行为，再理解模型语义。该顺序对应三个检查维度：

- 张量的 `shape / dtype / device / stride` 如何变化。
- 代码是否引入实际计算、内存分配、数据复制或同步。
- 这些 OP 在 decoder、attention、MLP、MoE、Graph replay 中承担什么作用。

本文分三段展开：

1. PyTorch 常见 OP 的基本行为。
2. 以 Transformers 中的 Llama 源码拆解经典 decoder-only 模型。
3. 以 DeepSeek V4 源码说明进阶结构：mHC 多残差流、压缩注意力和 MoE。

参考源码：

```text
repos/transformers/src/transformers/models/llama/modeling_llama.py
repos/transformers/src/transformers/models/deepseek_v4/modular_deepseek_v4.py
repos/transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py
```

阅读和修改入口使用 `modular_deepseek_v4.py`；`modeling_deepseek_v4.py` 是生成后的完整文件，运行时导入该文件。

## 1. PyTorch OP 的基本行为

PyTorch OP 同时包含数学语义和运行时行为。一个 OP 可能仅修改 tensor metadata，也可能触发 kernel、分配内存、复制数据或等待设备执行完成。

阅读源码时，应先检查以下项目：

| 检查项 | 说明 |
|---|---|
| `shape` | 决定矩阵乘、attention、cache 和 MoE routing 的维度是否合法。 |
| `dtype` | 决定 kernel 路径、精度、cast 成本和量化路径。 |
| `device` | 决定计算在 CPU、GPU、MUSA 还是跨设备拷贝。 |
| `stride` | 决定 `view` 是否可用，后续 kernel 是否需要连续内存。 |
| `is_contiguous()` | 判断 `contiguous()` 是否会产生实际数据复制。 |
| 输出 shape 是否固定 | 判断是否满足 Graph replay 和固定 buffer 要求。 |
| 是否回读到 CPU | 判断是否可能触发同步。 |

同一个 API 在不同输入布局下可能对应不同执行路径。`reshape` 可能返回 view，也可能复制数据。`contiguous()` 对连续 tensor 通常不产生新副本，对非连续 tensor 会生成连续副本。DEVICE tensor 的 `.item()` 需要等待设备结果并回读到 CPU。

### 1.1 Shape 与 Layout OP

常见 OP：

```text
view / reshape / flatten / unsqueeze / squeeze / expand / transpose / permute / contiguous
```

它们用于整理 tensor 形状和内存布局。Transformer 中的 Q/K/V head 维度、RoPE 输入、KV cache layout、MoE expert buffer 都依赖这些 OP。

最小示例：

```python
import torch

# x 是连续 tensor，shape=[2,3,4]，stride=(12,4,1)。
x = torch.arange(24).view(2, 3, 4)

# transpose 仅交换维度，通常不复制数据，但 stride 会改变。
y = x.transpose(1, 2)

# contiguous 将非连续 layout 复制成连续内存。
z = y.contiguous()

print(x.shape, x.stride(), x.is_contiguous())
print(y.shape, y.stride(), y.is_contiguous())
print(z.shape, z.stride(), z.is_contiguous())
```

输出：

```text
x shape=(2, 3, 4), stride=(12, 4, 1), contiguous=True
y shape=(2, 4, 3), stride=(12, 1, 4), contiguous=False
z shape=(2, 4, 3), stride=(12, 3, 1), contiguous=True
```

性能检查：

- `transpose / permute` 后接 custom kernel 前，应确认 kernel 是否支持非连续输入。
- `reshape` 不应默认按零拷贝处理，stride 不兼容时会复制。
- `contiguous()` 在高频路径中可能成为额外 D2D copy。
- `expand` 可能产生 zero-stride view，不应用于原地写入路径。

### 1.2 创建与原地更新 OP

常见 OP：

```text
empty / new_empty / empty_like / zeros / ones / full
copy_ / fill_ / zero_ / masked_fill_ / clamp_
```

这些 OP 常用于 KV cache、Graph input buffer、临时 workspace、padding 清理和 logits mask。

使用原则：

| 场景 | 实现方式 | 原因 |
|---|---|---|
| 高频临时 buffer | 按 bucket 预分配并复用 | 降低 allocator 开销，保持地址稳定。 |
| Graph replay 输入 | capture 前创建 tensor，replay 前用 `copy_` 更新内容 | Graph 要求固定地址。 |
| padding 槽位 | replay 前 `fill_ / zero_` 清理 | 避免上轮残留数据影响 mask、cache 或 logits。 |
| 激活裁剪 | 可原地更新时使用 `clamp_` | 减少临时 tensor；使用前需确认后续计算不再依赖原值。 |

### 1.3 数学、归约与激活 OP

常见 OP：

```text
sum / mean / max / min / pow / square / rsqrt
sigmoid / silu / gelu / relu / softmax / log_softmax / clamp
```

它们构成 RMSNorm、SwiGLU、attention softmax、MoE gate 和 sampling。

RMSNorm 最小示例：

```python
import torch

h = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

# RMSNorm 的核心是每个 token 在 hidden 维度上求均方。
variance = h.pow(2).mean(-1, keepdim=True)

# rsqrt(x) = 1 / sqrt(x)。
out = h * torch.rsqrt(variance + 1e-6)

print("variance =", variance)
print("out =", out)
```

输出：

```text
variance = tensor([[7.500]])
out      = tensor([[0.365, 0.730, 1.095, 1.461]])
```

Llama 源码中的 RMSNorm：

```python
# 使用 float32 计算 variance，降低归一化的数值误差。
hidden_states = hidden_states.to(torch.float32)

# 在 hidden 维度计算均方。
variance = hidden_states.pow(2).mean(-1, keepdim=True)

# 按 RMS 缩放 hidden states。
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

# 乘可学习权重，并转回输入 dtype。
return self.weight * hidden_states.to(input_dtype)
```

性能检查：

- 多个逐元素 OP 连续出现时，reference 实现可能产生多个 kernel launch。
- RMSNorm、SwiGLU、softmax 在热点路径中通常由专用 kernel 融合执行。
- `softmax(dtype=torch.float32)` 常用于数值稳定，但会引入 dtype 转换。

### 1.4 线性代数与 Attention OP

常见 OP：

```text
F.linear / matmul / mm / bmm / einsum
topk / sort / argsort / argmax
```

`F.linear(x, weight)` 等价于：

```text
x @ weight.T + bias
```

在 decoder 模型中，`linear` 用于：

- Q/K/V projection。
- Attention output projection。
- MLP gate/up/down projection。
- MoE router 和 expert GEMM。
- LM head。

Attention 最小示例：

```python
import torch

# query/key/value shape=[B=1, H=1, S=2, D=2]
query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
key = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])
value = torch.tensor([[[[2.0, 0.0], [0.0, 3.0]]]])

# q @ k^T 得到每个 query 对每个 key 的分数。
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
scores = tensor([[[[0.707, 0.707],
                   [0.000, 0.707]]]])
probs  = tensor([[[[0.500, 0.500],
                   [0.330, 0.670]]]])
out    = tensor([[[[1.000, 1.500],
                   [0.660, 2.009]]]])
```

### 1.5 索引、路由与动态 shape OP

常见 OP：

```text
gather / scatter_ / index_select / index_add_ / where / nonzero
topk / unique / unique_consecutive / masked_select / boolean indexing
```

它们常见于 MoE routing、KV page 选择、sampling、mask 调试和统计。

MoE top-k 最小示例：

```python
import torch

# 两个 token，三个 expert。
scores = torch.tensor([
    [0.1, 0.7, 0.2],
    [0.6, 0.1, 0.3],
])

# 每个 token 选择分数最高的两个 expert。
values, indices = torch.topk(scores, 2, dim=-1)

# 对选中的 expert 权重做归一化。
weights = values / values.sum(-1, keepdim=True)

print("indices =", indices)
print("weights =", weights)
```

输出：

```text
indices = tensor([[1, 2],
                  [0, 2]])
weights = tensor([[0.778, 0.222],
                  [0.667, 0.333]])
```

动态 shape 风险：

| OP | 风险 |
|---|---|
| `nonzero` | 输出长度等于满足条件的元素数。 |
| `unique` | 输出长度等于唯一值个数。 |
| `masked_select` | 输出长度等于 `True` 个数。 |
| boolean indexing | 输出长度依赖数据内容。 |
| `where(mask)` | 返回坐标长度依赖 mask。 |

这些 OP 可用于 eager 调试和离线分析，不应在固定 shape 的 Graph replay 中直接使用。在线推理通常改用 fixed capacity、padding、histogram 或 `bincount(minlength=...)`。

以当前 `torch_musa` 为例，`torch.unique(musa_tensor)` 会触发当前 stream 同步。原因是输出长度 `num_out` 在设备端生成后，主机端需要回读该值并据此 `resize_` 输出 tensor。源码中普通 dtype 路径使用 `length.item<int64_t>()`，bool 路径使用 `memcpy_and_sync`；后者内部调用 `musaStreamSynchronize(stream)`。

### 1.6 CPU 边界与同步 OP

常见同步来源：

```text
torch.cuda.synchronize()
torch.musa.synchronize()
stream.wait_event()
stream.wait_stream()
.item()
.tolist()
.cpu()
.numpy()
distributed work.wait()
```

显式同步可通过 API 识别；隐式同步需要检查 CPU 回读路径。对 DEVICE tensor 调用 `.item()`、`.tolist()`、`.cpu()` 时，CPU 必须等待前序设备任务完成。

使用原则：

- benchmark、错误定位、程序退出前可使用全设备同步。
- decode 热点路径优先使用 stream/event 局部依赖。
- scheduler 需要的长度、bucket、request 状态，优先维护 CPU 元数据副本。
- 最终 token、少量 logprob 或少量统计允许回读；完整 logits、hidden states、KV metadata 不应频繁回 CPU。

## 2. 经典 Decoder：Llama 源码拆解

Llama 是标准 decoder-only 语言模型。执行链路如下：

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

### 2.1 LlamaModel.forward

源码位置：

```text
repos/transformers/src/transformers/models/llama/modeling_llama.py::LlamaModel.forward
```

关键代码：

```python
# input_ids 是 token id，embedding 后变成 hidden states。
inputs_embeds = self.embed_tokens(input_ids)

# position_ids 通常是 [0, 1, 2, ...]，decode 时叠加 past_seen_tokens。
position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
position_ids = position_ids.unsqueeze(0)

# causal_mask 禁止当前位置访问未来 token。
causal_mask = create_causal_mask(...)

# decoder 的主输入从 embedding 开始。
hidden_states = inputs_embeds

# RoPE 的 cos/sin 在模型层统一生成，传给每一层 attention。
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
| token 查表 | `nn.Embedding` | 按 id 读取 embedding 表。 |
| 生成位置 | `arange`、`unsqueeze` | 创建位置 tensor。 |
| 构造 mask | mask 创建、加法、填充 | 形成 causal bias。 |
| RoPE | `matmul`、`cat`、`cos`、`sin` | 生成旋转位置编码。 |
| decoder 循环 | Python loop + module call | 每层重复 attention 和 MLP。 |

### 2.2 LlamaDecoderLayer.forward

源码位置：

```text
modeling_llama.py::LlamaDecoderLayer.forward
```

关键代码：

```python
# 保存 attention 前的输入，用于第一条 residual add。
residual = hidden_states

# pre-norm：先归一化，再进入 self-attention。
hidden_states = self.input_layernorm(hidden_states)
hidden_states, _ = self.self_attn(...)
hidden_states = residual + hidden_states

# 保存 MLP 前的输入，用于第二条 residual add。
residual = hidden_states

# pre-norm：先归一化，再进入 MLP。
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states
```

这是 classic pre-norm decoder block：

```text
x -> RMSNorm -> Attention -> residual add
  -> RMSNorm -> MLP       -> residual add
```

对应 OP：

| 阶段 | 主要 OP | 说明 |
|---|---|---|
| RMSNorm | `to(float32)`、`pow`、`mean`、`rsqrt`、`mul` | 对 hidden 维做归一化。 |
| Attention | `linear`、`view`、`transpose`、RoPE、`matmul`、`softmax`、`matmul` | 计算 token 间依赖。 |
| Residual | `add` | 保留原始信息，稳定深层训练。 |
| MLP | `linear`、activation、`mul`、`linear` | token 内部的非线性变换。 |

### 2.3 LlamaAttention.forward

源码位置：

```text
modeling_llama.py::LlamaAttention.forward
```

关键流程：

```python
input_shape = hidden_states.shape[:-1]
hidden_shape = (*input_shape, -1, self.head_dim)

# Q/K/V projection，并整理成多头布局。
query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

# RoPE 仅作用在 Q/K。
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

# decode 时更新 KV cache。
if past_key_values is not None:
    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

# attention_interface 可选择 eager、SDPA、FlashAttention 等实现。
attn_output, attn_weights = attention_interface(...)

# 回到 [batch, seq, hidden]，再做 output projection。
attn_output = attn_output.reshape(*input_shape, -1).contiguous()
attn_output = self.o_proj(attn_output)
```

Q projection 的最小 shape 例子：

```python
import torch
import torch.nn.functional as F

# hidden shape=[B=1, S=2, D=4]
hidden = torch.tensor([[
    [1.0, 0.0, 2.0, 1.0],
    [0.0, 1.0, 1.0, 2.0],
]])

# 示例使用单位矩阵作为 projection weight。
q_weight = torch.eye(4)

# projection 后 view 成 [B,S,H,head_dim]，再 transpose 成 [B,H,S,head_dim]。
q = F.linear(hidden, q_weight).view(1, 2, 2, 2).transpose(1, 2)

print(q.shape)
print(q)
```

输出：

```text
q shape=(1, 2, 2, 2)
q =
tensor([[[[1., 0.],
          [0., 1.]],

         [[2., 1.],
          [1., 2.]]]])
```

对应关系：

```text
[B,S,D] -> linear -> [B,S,H*Dh] -> view -> [B,S,H,Dh] -> transpose -> [B,H,S,Dh]
```

性能检查：

- `view` 要求 projection 输出的 stride 兼容。
- `transpose` 后是非连续 layout，后续 kernel 是否接受该 layout 取决于 attention backend。
- `attn_output.reshape(...).contiguous()` 可能引入实际数据复制。
- `past_key_values.update` 会改变 decode 中 KV cache 的读写路径。
- `attention_interface` 决定底层 kernel，不应仅依据 Python 函数名判断实际执行路径。

### 2.4 RoPE

源码位置：

```text
modeling_llama.py::LlamaRotaryEmbedding.forward
modeling_llama.py::apply_rotary_pos_emb
```

RoPE 主要 OP：

| 代码动作 | OP |
|---|---|
| 生成频率 | `arange`、幂运算、除法 |
| 位置相乘 | `matmul` |
| 拼接正余弦输入 | `cat` |
| 得到旋转因子 | `cos`、`sin` |
| 应用到 Q/K | `unsqueeze`、`mul`、`add`、`cat` |

核心公式在源码中体现为：

```python
# q 乘 cos 保留原方向，rotate_half(q) 乘 sin 引入旋转方向。
q_embed = (q * cos) + (rotate_half(q) * sin)

# k 使用同一组 cos/sin，保证 attention score 中的位置关系一致。
k_embed = (k * cos) + (rotate_half(k) * sin)
```

`rotate_half` 将最后一维拆成两半：

```text
[x1, x2] -> [-x2, x1]
```

RoPE 的作用是将 position 信息注入 Q/K，使注意力分数携带相对位置信息。

### 2.5 LlamaMLP.forward

源码位置：

```text
modeling_llama.py::LlamaMLP.forward
```

关键代码：

```python
# gate_proj 产生门控分支，up_proj 产生内容分支；两者相乘后由 down_proj 投回 hidden size。
down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

展开后：

```text
gate = gate_proj(x)
up   = up_proj(x)
act  = activation(gate)
mix  = act * up
out  = down_proj(mix)
```

对应 OP：

| 阶段 | OP | 说明 |
|---|---|---|
| gate projection | `linear` | 产生门控分支。 |
| up projection | `linear` | 产生被门控的内容分支。 |
| activation | `silu` 或其他激活 | 非线性。 |
| elementwise multiply | `mul` | 门控分支控制内容分支。 |
| down projection | `linear` | 回到 hidden size。 |

该结构属于 SwiGLU 类 MLP。reference 实现便于校验数学语义；热点路径通常会融合 activation 和 multiply。

### 2.6 LlamaForCausalLM.forward

源码位置：

```text
modeling_llama.py::LlamaForCausalLM.forward
```

关键代码：

```python
# 先运行 base decoder，得到每个 token 的 hidden states。
outputs = self.model(...)
hidden_states = outputs.last_hidden_state

# decode 时通常仅需要最后一个或最后几个 token 的 logits。
slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

# lm_head 将 hidden states 投影到 vocabulary 维度。
logits = self.lm_head(hidden_states[:, slice_indices, :])
```

这里的 `lm_head` 是最后一次 `linear`：

```text
[B,S,D] -> [B,S,vocab_size]
```

decode 阶段若仅生成下一个 token，通常不需要对所有历史 token 计算 logits。`logits_to_keep` 用于减少无用计算和显存写入。

## 3. DeepSeek V4：在经典 Decoder 上增加的结构

DeepSeek V4 仍然是 decoder-only 语言模型，并在 Llama 结构上增加以下模块：

| 模块 | Llama | DeepSeek V4 |
|---|---|---|
| 残差连接 | 单条 hidden state | `hc_mult` 条残差流，mHC 控制折叠和写回。 |
| Attention | 标准 Q/K/V，多头注意力 | 单 KV head、partial RoPE、compressed KV、attention sink。 |
| 长上下文 | 常规 KV cache / sliding window | HCA / CSA 压缩历史信息。 |
| MLP | dense SwiGLU | shared expert + routed experts。 |
| Router | 无 | hash router 或 top-k router。 |

说明：DeepSeek V4 里的 residual stream / hidden stream 是模型张量维度，不是 GPU stream。GPU stream 是 runtime 的执行队列，二者含义不同。

### 3.1 DeepseekV4Model.forward

源码位置：

```text
modular_deepseek_v4.py::DeepseekV4Model.forward
```

关键代码：

```python
# input_ids 先通过 embedding 表转换为普通 hidden states。
inputs_embeds = self.embed_tokens(input_ids)

# 将普通 hidden 扩展成 hc_mult 条残差流。
hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()

# main 和 compress 使用不同的 RoPE 配置，分别服务普通 attention 和压缩路径。
position_embeddings = {
    "main": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main"),
    "compress": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="compress"),
}

# 每个 decoder layer 都接收并返回多条残差流。
for layer in self.layers:
    hidden_states = layer(...)

# 最终使用 HyperHead 将多条残差流折叠回普通 hidden。
hidden_states = self.norm(self.hc_head(hidden_states))
```

Llama 的 hidden shape：

```text
[B, S, D]
```

DeepSeek V4 的 hidden shape：

```text
[B, S, hc_mult, D]
```

如果 `hc_mult=2`，每个 token 有两条 residual stream：

```text
token0 stream0 = [1.0,  0.0]
token0 stream1 = [1.2, -0.2]
```

### 3.2 mHC：多残差流如何进入 Attention / MoE

源码位置：

```text
modular_deepseek_v4.py::DeepseekV4HyperConnection.forward
modular_deepseek_v4.py::DeepseekV4DecoderLayer.forward
```

mHC 包含三个步骤：

1. 计算 `pre`：将多条残差流折叠成一条 hidden，送入 Attention 或 MoE。
2. 计算 `post`：控制子层输出写回每条残差流的比例。
3. 计算 `comb`：控制旧残差流之间如何混合。

核心代码：

```python
# flatten 将 [B, S, hc_mult, D] 合并为 [B, S, hc_mult * D]，用于生成 mHC 参数。
flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())

# 一个 linear 同时生成 pre、post、comb 三组权重。
pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)

# pre 控制多条 residual stream 折叠到单条 hidden 的比例。
pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps

# post 控制子层输出写回每条 residual stream 的比例。
post = 2 * torch.sigmoid(post_w * post_scale + post_b)

# comb 控制旧 residual stream 之间的混合比例。
comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps

# collapsed 是 Attention 或 MoE 接收的普通 hidden。
collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
```

最小示例：

```python
import torch

# 一个 token，两条 residual stream。
stream0 = torch.tensor([1.0, 0.0])
stream1 = torch.tensor([1.2, -0.2])

# pre 是 mHC 生成的折叠权重。示例中手工指定，便于计算。
pre = torch.tensor([0.25, 0.75])

# collapsed 是送入 attention / MoE 的普通 hidden。
collapsed = pre[0] * stream0 + pre[1] * stream1

print(collapsed)
```

输出：

```text
collapsed = tensor([ 1.150, -0.150])
```

DecoderLayer 中的使用方式：

```python
# attn_hc 将多条 residual stream 折叠为 attention 输入，并返回写回权重。
post, comb, collapsed = self.attn_hc(hidden_states)
attn_output, _ = self.self_attn(self.input_layernorm(collapsed), **kwargs)

# attention 输出按 post 写回各 stream，同时用 comb 混合旧 stream。
hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
    comb.to(dtype).transpose(-1, -2), hidden_states
)

# ffn_hc 对 MLP 分支重复同样的折叠和写回流程。
post, comb, collapsed = self.ffn_hc(hidden_states)
mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
```

对应 OP：

| mHC 动作 | OP |
|---|---|
| 多流拼接 | `flatten(start_dim=2)` |
| 生成混合参数 | `F.linear` |
| 切分参数 | `split` |
| 权重约束 | `sigmoid`、`softmax`、`sum`、除法 |
| 折叠多流 | `unsqueeze`、`mul`、`sum` |
| 写回多流 | `matmul`、`transpose`、`add` |

### 3.3 DeepSeek V4 Attention

源码位置：

```text
modular_deepseek_v4.py::DeepseekV4Attention.forward
```

关键差异：

| 点 | 说明 |
|---|---|
| 单 KV head | `num_key_value_heads = 1`，同一个 KV 被所有 query heads 共享。 |
| Q 低秩投影 | `q_a_proj -> RMSNorm -> q_b_proj`。 |
| K=V | `kv_proj` 生成一个 tensor，同时作为 key 和 value。 |
| partial RoPE | RoPE 仅作用在部分 head 维度。 |
| compressed KV | HCA / CSA 会额外生成压缩 KV，并拼接到 KV 轴。 |
| output projection | grouped low-rank projection：`o_a_proj -> flatten -> o_b_proj`。 |

核心代码：

```python
# Q 先经过低秩投影和归一化，再展开为多头布局。
q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
q = self.q_b_norm(q)

# RoPE 作用在 Q 的位置相关维度。
q = apply_rotary_pos_emb(q, cos, sin)

# KV 由同一个投影生成，DeepSeek V4 中 key 和 value 共享该 tensor。
kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
kv = apply_rotary_pos_emb(kv, cos, sin)

# 压缩注意力启用时，将 compressed KV 拼接到原 KV 后面。
if self.compressor is not None:
    compressed_kv, block_bias = self.compressor(...)
    kv = torch.cat([kv, compressed_kv], dim=2)

# q 作为 query，kv 同时作为 key 和 value。
attn_output, attn_weights = attention_interface(self, q, kv, kv, attention_mask, ...)
```

对应 OP：

| 阶段 | OP |
|---|---|
| Q 低秩投影 | `linear`、RMSNorm、`view`、`transpose` |
| KV 投影 | `linear`、RMSNorm、`view`、`transpose` |
| RoPE | `mul`、`add`、`cat` |
| 压缩 KV 拼接 | `cat` |
| mask 扩展 | `cat` 或 `pad` |
| attention | `matmul`、`softmax`、`matmul` 或后端 fused kernel |
| 输出投影 | `reshape`、grouped `linear`、`flatten`、`linear` |

性能检查：

- `torch.cat([kv, compressed_kv], dim=2)` 会生成新的 KV tensor，长上下文下应关注内存和带宽。
- mask 长度必须与 KV 拼接后的长度一致，否则会出现可见性错误。
- `attention_interface` 决定底层 kernel；压缩 KV 后的 KV 长度和 mask 形状会影响能否命中特定 attention 后端。

### 3.4 HCA：将多个 token 压缩成一个 compressed KV

源码位置：

```text
modular_deepseek_v4.py::DeepseekV4HCACompressor.forward
```

HCA 的核心是每 `compress_rate` 个 token 生成一个 compressed KV entry。

核心代码：

```python
# 将连续 token 按 compress_rate 分成多个 window。
chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, -1)

# gate 同样按 window 切分，并叠加位置偏置。
chunk_gate = chunk_gate.view(batch, n_windows, self.compress_rate, -1) + self.position_bias.to(chunk_gate.dtype)

# window 内对 gate 做 softmax，再对 KV 加权求和，得到一个 compressed entry。
compressed = self.kv_norm(
    (chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)).sum(dim=2)
)
```

含义：

```text
同一个 window 内：
  gate -> softmax -> 每个 token 的权重
  kv * weight -> 加权
  sum(dim=2) -> 一个 compressed entry
```

最小示例，两个 token 压成一个 entry：

```python
import torch

# 两个 token 的 KV。
kv = torch.tensor([
    [1.0, 0.0],
    [0.0, 2.0],
])

# gate 每一列独立做 softmax，得到每个 token 对每个维度的权重。
gate = torch.tensor([
    [0.0, 1.0],
    [1.0, 0.0],
])

weights = torch.softmax(gate, dim=0)
compressed = (kv * weights).sum(dim=0)

print("weights =", weights)
print("compressed =", compressed)
```

输出：

```text
weights =
tensor([[0.269, 0.731],
        [0.731, 0.269]])

compressed = tensor([0.269, 0.538])
```

该 compressed entry 会拼接到 attention 的 KV 轴上。后续 query 可以同时访问局部 sliding KV 和历史压缩 KV。

HCA 同时生成 `block_bias`：

```python
# causal_threshold 表示当前 query 最多可见到哪个 compressed entry。
causal_threshold = (position_ids + 1) // self.compress_rate

# 超过可见范围的 compressed entry 被置为 -inf。
block_bias = block_bias.masked_fill(
    entry_indices.view(1, 1, 1, -1) >= causal_threshold.unsqueeze(1).unsqueeze(-1),
    float("-inf"),
)
```

`block_bias` 用于屏蔽当前 query 不具备可见性的 compressed entry。

### 3.5 CSA：压缩后再用 Indexer 选 top-k compressed blocks

源码位置：

```text
modular_deepseek_v4.py::DeepseekV4CSACompressor.forward
modular_deepseek_v4.py::DeepseekV4Indexer.forward
```

CSA 在 HCA 的压缩流程后增加索引选择阶段。该模块先生成 compressed KV，再用 Indexer 为每个 query 选择 top-k compressed entries。

Indexer 核心代码：

```python
# 计算 query 与每个 compressed KV 的相似度。
scores = torch.matmul(q.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))

# relu 保留非负分数，并乘缩放系数。
scores = F.relu(scores) * self.softmax_scale

# weights_proj 为不同 head 或分量生成加权系数。
weights = self.weights_proj(hidden_states).float() * self.weights_scaling

# 汇总得到每个 query 对 compressed entry 的索引分数。
index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)

# 选择 top-k 个 compressed entry。
top_k_indices = index_scores.topk(top_k, dim=-1).indices
```

CSA compressor 使用上述 index 构造 block bias：

```python
# invalid index 用 compressed_len 作为 sentinel。
valid = top_k_indices >= 0
safe_indices = torch.where(valid, top_k_indices, torch.full_like(top_k_indices, compressed_len))

# 默认所有 compressed entry 不可见。
block_bias = compressed_kv.new_full((batch, 1, seq_len, compressed_len + 1), float("-inf"))

# top-k 命中的 entry 写入 0，表示允许 attention 访问。
block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
```

对应 OP：

| 阶段 | OP |
|---|---|
| 压缩 KV | `view`、`new_zeros`、`new_full`、slice 写入、`softmax`、`sum` |
| 计算 query 与 compressed KV 分数 | `matmul`、`transpose`、`relu` |
| 生成 index score | `linear`、`mul`、`sum` |
| 选择 top-k | `topk` |
| 构造可见性 mask | `where`、`full_like`、`new_full`、`scatter_` |

性能检查：

- `topk` 的 `k` 应固定，便于后续 shape 稳定。
- `scatter_` 写 mask 前需确认 index 范围，源码用 sentinel 处理 invalid index。
- `new_zeros / new_full` 在热点路径中可能产生分配，后端优化时应考虑 workspace 复用。

### 3.6 DeepSeek V4 MoE

源码位置：

```text
modular_deepseek_v4.py::DeepseekV4SparseMoeBlock.forward
modular_deepseek_v4.py::DeepseekV4TopKRouter.forward
modular_deepseek_v4.py::DeepseekV4HashRouter.forward
modular_deepseek_v4.py::DeepseekV4Experts.forward
```

SparseMoeBlock 主线：

```python
# 将 [B, S, D] 拉平成 token 维，便于按 token 路由到 expert。
flat = hidden_states.view(-1, hidden_dim)

# hash router 使用 input_ids；top-k router 仅使用 hidden_states。
if self.is_hash:
    _, weights, indices = self.gate(hidden_states, input_ids)
else:
    _, weights, indices = self.gate(hidden_states)

# experts 输出 routed 分支，shared_experts 输出共享分支，二者相加得到 MoE 输出。
routed = self.experts(flat, indices, weights).view(batch, seq_len, hidden_dim)
return routed + self.shared_experts(residual)
```

TopKRouter：

```python
# 将 token 拉平成二维矩阵，送入 router linear。
flat = hidden_states.reshape(-1, self.hidden_dim)
logits = F.linear(flat, self.weight)

# score_fn 通常是 softmax 或 sigmoid。
scores = self.score_fn(logits)

# 每个 token 选择 top-k 个 expert。
indices = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices

# 取出被选中 expert 的权重，并归一化。
weights = scores.gather(1, indices)
weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
```

HashRouter：

```python
# hash router 直接由 token id 查表得到 expert id。
indices = self.tid2eid[input_ids.reshape(-1)].long()

# 从 router score 中取出对应 expert 的权重。
weights = scores.gather(1, indices)
```

Experts reference 实现：

```python
# final 用于累加所有 expert 的输出。
final = torch.zeros_like(hidden_states)

with torch.no_grad():
    # mask shape 经过 permute 后按 expert 维度组织，便于枚举活跃 expert。
    mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)

    # hit 仅保留至少接收一个 token 的 expert。
    hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()

for expert_idx in hit:
    # 提取当前 expert 接收的 token 以及对应的 top-k 槽位。
    top_k_pos, token_idx = torch.where(mask[expert_idx])

    # 当前 expert 执行 gate/up 分支和 down projection。
    current = self._apply_gate(F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]))

    # 按 router 权重缩放 expert 输出。
    current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]

    # 同一个 token 可能来自多个 expert，使用 index_add_ 累加。
    final.index_add_(0, token_idx, current.to(final.dtype))
```

对应 OP：

| 阶段 | OP | 风险 |
|---|---|---|
| router 评分 | `reshape`、`linear`、activation | dtype 和 layout 决定 GEMM 路径。 |
| 选择 expert | `topk` 或 embedding lookup | `topk` 固定 k；hash lookup 依赖 token id。 |
| 取权重 | `gather` | index device/dtype 必须正确。 |
| 识别活跃 expert | `one_hot`、`sum`、`nonzero` | 动态 shape，不应作为 Graph 热路径中的 reference 实现。 |
| 选择 token | `where` | 每个 expert token 数动态。 |
| expert 计算 | `linear`、`chunk`、`clamp`、activation、`mul`、`linear` | 应优先使用 fused/grouped expert kernel。 |
| 合并输出 | `index_add_` | 重复 token 累加，注意 dtype 和写冲突语义。 |

该 reference 实现用于说明 MoE 语义，不作为高性能在线路径。生产推理通常会将 dispatch、expert GEMM、combine 下沉到 fused kernel、grouped GEMM 或专用后端。

## 4. OP 与性能排查

排查性能问题时，应判断代码触发的底层行为，不以 Python 表达是否简短作为判断依据。

### 4.1 出现额外数据复制

优先检查：

```text
reshape / transpose / permute / contiguous / cat / stack / pad / to(device)
```

典型原因：

- stride 不兼容。
- attention 或 GEMM kernel 要求连续输入。
- `cat / stack` 生成新 tensor。
- dtype 或 device 转换引入拷贝。

### 4.2 出现同步等待

优先检查：

```text
.item() / .tolist() / .cpu() / .numpy()
synchronize()
stream.wait_event()
unique / nonzero / masked_select
distributed work.wait()
```

典型原因：

- CPU 读取 DEVICE 结果。
- 动态 shape OP 需要回读输出长度。
- benchmark 或日志代码进入 decode 热点路径。
- collective 过早 wait，破坏通信与计算重叠。

### 4.3 出现 allocator 抖动

优先检查：

```text
empty / zeros / new_full / cat / masked_select / nonzero / unique
```

处理方式：

- 按 batch/seq bucket 预分配 workspace。
- Graph replay 前固定 input/output/metadata tensor。
- 用 padding 和 fixed capacity 稳定 shape。

### 4.4 kernel 数量过多

优先检查：

```text
RMSNorm reference:
  pow -> mean -> rsqrt -> mul

SwiGLU reference:
  linear -> chunk -> clamp -> silu -> mul -> linear

MoE reference:
  one_hot -> nonzero -> where -> per-expert loop -> index_add_
```

处理方式：

- 使用 fused RMSNorm、fused SwiGLU、FlashAttention、grouped GEMM。
- 将多个小 OP 合并到后端 kernel。
- 避免 Python per-expert loop 进入高频路径。

### 4.5 Graph replay 失败或收益低

优先检查：

```text
dynamic shape
tensor 地址变化
capture 内 allocator
capture 内 CPU sync
不支持 graph capture 的 kernel 或通信
```

处理方式：

- 固定 bucket。
- replay 前仅用 `copy_ / fill_ / zero_` 更新固定 buffer。
- 避免 `unique / nonzero / masked_select` 直接决定 graph 内 shape。
- 对不支持 capture 的后端路径显式拆到 graph 外。

## 5. 源码分析步骤

分析顺序：

1. 确认输入输出 shape。
2. 检查 layout：`view / reshape / transpose / contiguous`。
3. 检查核心计算：`linear / matmul / softmax / activation`。
4. 检查索引和动态 shape：`topk / gather / scatter / where / nonzero / unique`。
5. 检查 CPU 边界：`.item / .tolist / .cpu / synchronize`。
6. 汇总模型语义：attention、KV cache、MLP、MoE、压缩注意力、Graph replay。

Llama 用于建立 decoder 的基本框架：

```text
embedding -> decoder layers -> norm -> lm_head
```

每个 decoder layer 是：

```text
RMSNorm -> Attention -> Residual
RMSNorm -> MLP       -> Residual
```

DeepSeek V4 在该框架上扩展 mHC、压缩注意力和 MoE：

```text
embedding
  -> 扩展成多条 residual streams
  -> mHC 折叠成普通 hidden
  -> compressed attention
  -> mHC 写回 residual streams
  -> MoE
  -> mHC 写回 residual streams
  -> HyperHead 折叠回普通 hidden
  -> lm_head
```

完成上述检查后，可将复杂源码拆分为可验证的张量步骤。性能分析应从 profiler 现象回到源码位置：copy 对应 layout 和拼接，sync 对应 CPU 回读和动态 shape，kernel 数量对应 reference OP 序列，Graph 问题对应 shape、地址和同步边界。
