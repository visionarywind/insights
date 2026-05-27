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

### 1.1 从 API 到运行时行为

源码中的 PyTorch API 与底层 kernel 不是一一对应关系。同一条 Python 语句可能只改 metadata，也可能进入 GEMM、copy kernel、allocator、stream wait 或 graph replay。分析时先把 OP 归入执行行为，再判断是否影响热点路径。

| 执行行为 | 典型 OP | 底层表现 | 主要风险 |
|---|---|---|---|
| metadata 更新 | `view`、`unsqueeze`、合法 `reshape` | 不提交 kernel，改变 shape/stride 描述 | 后续 kernel 不接受当前 stride。 |
| 数据复制 | `contiguous`、stride 不兼容的 `reshape`、`to(device)` | D2D、H2D 或 D2H copy | decode 单步中反复复制。 |
| 计算提交 | `linear`、`matmul`、`softmax`、`topk` | GEMM、排序、归约或 fused kernel | dtype/layout 不匹配导致 fallback。 |
| 内存分配 | `empty`、`zeros`、`cat`、`new_full` | allocator 活跃，可能伴随 memset | Graph 地址不稳定，尾延迟抖动。 |
| 同步边界 | `.item()`、`.cpu()`、`synchronize`、`work.wait()` | CPU、stream 或 rank 等待 | 打断异步流水和通信计算重叠。 |
| 动态 shape | `nonzero`、`unique`、`masked_select` | 输出长度依赖数据内容 | Graph replay、workspace 和通信 bucket 不稳定。 |
| Graph 执行 | `MUSAGraph.replay`、CUDA Graph replay | 重放固定地址和固定执行路径 | capture 内存在动态 shape、allocator 或 CPU sync。 |

Driver/Runtime 侧看到的是 kernel launch、memcpy/memset、allocator、stream/event、synchronize、capture/replay 和 fallback。源码分析应把 Python OP、tensor 属性、profiler 事件和后端 kernel 名称对应起来。

### 1.2 Shape 与 Layout OP

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

### 1.3 创建与原地更新 OP

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

固定 buffer 最小示例：

```python
import torch

# capture/replay 场景中，Graph 记录的是固定 tensor 对象和底层地址。
# 这里用 CPU tensor 模拟固定 buffer 的内容更新语义。
graph_input = torch.zeros(2)

# 第一次执行时读取 graph_input 的当前内容。
out0 = graph_input * 3 + 2

# replay 前用 copy_ 改内容，不替换 graph_input 对象。
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

该例子对应 Graph replay 的基本约束：capture 后保留 input/output/metadata tensor 对象，replay 前只更新已有 buffer 内容。若重新创建 tensor 并替换变量，Graph 记录的地址关系不会随 Python 变量一起改变。

### 1.4 数学、归约与激活 OP

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

### 1.5 线性代数与 Attention OP

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

### 1.6 索引、路由与动态 shape OP

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

### 1.7 CPU 边界与同步 OP

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

### 1.8 dtype、device 与后端 fallback

`Tensor.to(dtype)`、`Tensor.to(device)`、`.float()`、`.bfloat16()`、`.long()` 等 OP 会改变 dtype 或 device。它们常见于模型加载、量化前后处理、metadata 整理和 CPU pinned metadata 上传。

| 场景 | 常见 OP | 检查点 |
|---|---|---|
| Norm 内部升精度 | `.to(torch.float32)` | 是否只在局部计算中升精度，输出是否 cast 回模型 dtype。 |
| index dtype 对齐 | `.long()`、`.int()` | 后端 kernel 需要 int32 还是 int64，index 是否越界。 |
| CPU 到 DEVICE metadata | `.to(device, non_blocking=True)`、`copy_` | 是否使用固定 buffer，是否引入 H2D copy。 |
| 量化路径 | `.to(dtype)`、scale tensor | activation、weight、scale 的 dtype、shape、stride 是否匹配 kernel。 |

fallback 的判断不以 Python API 为准，而以执行路径为准。`F.linear` 能够进入高性能 GEMM，也可能进入普通 ATen 路径；attention 能够进入 fused backend，也可能退回 `matmul -> softmax -> matmul` reference。出现性能异常时，应检查 kernel 名称、dtype、layout、scale shape、Graph capture 支持和后端日志。

### 1.9 Graph、Compile 与执行边界

`torch.compile` 和 Graph replay 解决的问题不同：

- `torch.compile` 优化 PyTorch OP 图，可能做融合、重排或生成后端 kernel。
- Graph replay 固定一次执行的 shape、tensor 地址、kernel 序列和 stream 依赖，用于重复执行。
- compile 成功不代表该片段可被 Graph capture，还需检查 dynamic shape、CPU sync、allocator、通信 API 和后端 kernel 是否支持 capture。

Graph replay 用于 decode 中固定 bucket 的高频路径。动态请求应在 Graph 外整理成固定 shape 的 input ids、positions、seq_lens、KV slot、block table 或 MoE metadata，再通过 `copy_ / fill_ / zero_` 写入固定 buffer。

| 不稳定来源 | 稳定化方式 |
|---|---|
| batch size 变化 | bucket 化，未使用槽位 padding。 |
| sequence length 变化 | 固定最大长度或 sliding window。 |
| 动态输出长度 | fixed capacity、sentinel index、固定 top-k。 |
| CPU 读取 DEVICE metadata | 维护 CPU mirror metadata。 |
| capture 内 allocator | capture 前预分配 workspace。 |
| 不支持 capture 的 backend | 拆到 Graph 外，或使用 piecewise graph。 |

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

prefill 和 decode 的执行重点不同：

| 阶段 | 输入特点 | Attention 行为 | 性能重点 |
|---|---|---|---|
| prefill | 一次输入多个 prompt token | 当前 token 之间做 causal attention，并写入 KV cache | GEMM、attention 带宽、KV cache 初始化。 |
| decode | 每步新增一个 token | 新 query 读取历史 KV cache，追加当前 K/V | kernel launch、Graph replay、KV cache 访问、CPU 同步边界。 |

同一段源码在两个阶段的热点不同。prefill 更接近大矩阵计算，decode 更容易被小 kernel、metadata copy、同步等待和 Graph replay 约束放大。

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

Causal mask 最小示例：

```python
import torch

# scores 表示 3 个 query token 对 3 个 key token 的原始注意力分数。
scores = torch.tensor([
    [0.2, 0.1, 0.4],
    [0.3, 0.5, 0.0],
    [0.2, 0.4, 0.6],
])

# 上三角位置是未来 token，填入 -inf 后 softmax 概率变成 0。
mask = torch.triu(torch.full((3, 3), float("-inf")), diagonal=1)
masked_scores = scores + mask
probs = torch.softmax(masked_scores, dim=-1)

print("masked_scores =", masked_scores)
print("probs =", probs)
```

输出：

```text
masked_scores = tensor([[0.200,  -inf,  -inf],
                        [0.300, 0.500,  -inf],
                        [0.200, 0.400, 0.600]])
probs = tensor([[1.000, 0.000, 0.000],
                [0.450, 0.550, 0.000],
                [0.269, 0.329, 0.402]])
```

该例子对应 `create_causal_mask(...)` 的作用：第 0 个 token 仅能访问自己，第 1 个 token 能够访问 token0 和 token1，第 2 个 token 能够访问 token0 到 token2。

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

print(f"q shape={tuple(q.shape)}")
print("q =")
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

GQA / MQA 中的 KV head 扩展：

Llama 的 `repeat_kv` 将较少的 KV head 扩展到 query head 数量。源码中先插入一个维度并用 `expand` 共享数据视图，再通过 `reshape` 得到 `[batch, num_attention_heads, seq, head_dim]`。

```python
import torch

# kv shape=[B=1, KVH=1, S=2, D=2]，只有一个 KV head。
kv = torch.tensor([[[[1.0, 0.0], [2.0, 0.0]]]])

# n_rep=2 表示一个 KV head 被两个 query head 共享。
expanded = kv[:, :, None, :, :].expand(1, 1, 2, 2, 2)
repeated = expanded.reshape(1, 2, 2, 2)

print(repeated)
```

输出：

```text
tensor([[[[1., 0.],
          [2., 0.]],

         [[1., 0.],
          [2., 0.]]]])
```

该例子对应源码中的 `repeat_kv(key, module.num_key_value_groups)` 和 `repeat_kv(value, module.num_key_value_groups)`。`expand` 可能产生 zero-stride view，后续 backend 是否接受该 layout 取决于 attention 实现。

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

RoPE 最小数值例子：

```python
import torch

# 一个二维 query，最后一维拆成两半。
q = torch.tensor([1.0, 2.0])

# 示例手工指定 cos/sin，便于核对公式。
cos = torch.tensor([0.5, 0.5])
sin = torch.tensor([0.866, 0.866])

# rotate_half([1, 2]) = [-2, 1]。
rotated = torch.tensor([-q[1], q[0]])

# q_embed = q * cos + rotate_half(q) * sin。
q_embed = q * cos + rotated * sin

print("rotated =", rotated)
print("q_embed =", q_embed)
```

输出：

```text
rotated = tensor([-2.,  1.])
q_embed = tensor([-1.232,  1.866])
```

该例子对应 `apply_rotary_pos_emb` 中的两项相加。实际模型的 `cos/sin` 由 `position_ids` 和 `inv_freq` 生成，示例使用固定数值仅用于验证计算公式。

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

SwiGLU 最小示例：

```python
import torch

# gate 是门控分支，up 是内容分支。
gate = torch.tensor([[-1.0, 0.5, 2.0]])
up = torch.tensor([[2.0, 4.0, -1.0]])

# SiLU(gate) = gate * sigmoid(gate)。
act = torch.nn.functional.silu(gate)

# mix 对应源码中的 act_fn(gate_proj(x)) * up_proj(x)。
mix = act * up

print("act =", act)
print("mix =", mix)
```

输出：

```text
act = tensor([[-0.269,  0.311,  1.762]])
mix = tensor([[-0.538,  1.245, -1.762]])
```

该例子对应 `LlamaMLP.forward` 中 down projection 之前的门控乘法。实际源码中 `gate` 和 `up` 都来自 `linear`，示例直接给出中间值，便于核对 activation 和 elementwise multiply。

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

当前 Transformers 源码还显式关闭了部分 attention 后端：

```text
DeepseekV4PreTrainedModel._supports_flash_attn = False
DeepseekV4PreTrainedModel._supports_sdpa = False
DeepseekV4PreTrainedModel._supports_flex_attn = False
DeepseekV4PreTrainedModel._can_compile_fullgraph = False
```

源码注释给出的原因包括：V4 `head_dim=512` 超过当前 FlashAttention 2/3/4 的 head dim 限制；SDPA 无法表达 per-head learnable sink；FlexAttention 的 BlockMask 无法在 attention 内部拼接 compressed KV 后动态扩展；压缩器状态依赖动态 cache 层，不满足 fullgraph compile 的静态约束。该信息说明：DeepSeek V4 的 reference 路径主要用于结构分析，生产推理仍需确认后端 kernel、Graph 和 cache 实现是否覆盖这些约束。

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

### 3.2 HyperHead：最终折叠多条残差流

源码位置：

```text
modular_deepseek_v4.py::DeepseekV4HyperHead.forward
```

关键代码：

```python
# 将 [B, S, hc_mult, D] 合并成 [B, S, hc_mult * D]，用于生成最终折叠权重。
flat = self.input_norm(x.flatten(2).float())

# F.linear 生成每条 residual stream 的权重 logit。
mixes = F.linear(flat, self.hc_fn.float())

# sigmoid 将权重约束到正值区间，并加 eps 保持数值稳定。
pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps

# 按 stream 维度加权求和，输出普通 hidden states。
return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)
```

最小示例：

```python
import torch

stream0 = torch.tensor([1.0, 0.0])
stream1 = torch.tensor([1.2, -0.2])

# HyperHead 生成的最终折叠权重，示例中手工指定。
pre = torch.tensor([0.4, 0.6])

hidden = pre[0] * stream0 + pre[1] * stream1

print("hidden =", hidden)
```

输出：

```text
hidden = tensor([ 1.120, -0.120])
```

HyperHead 与 mHC 的差别在于：mHC 位于每个 decoder layer 内部，用于子层输入折叠和输出写回；HyperHead 位于模型末尾，负责把多条 residual stream 折叠回普通 hidden，再交给最终 RMSNorm。

### 3.3 mHC：多残差流如何进入 Attention / MoE

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

print("collapsed =", collapsed)
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

mHC 写回最小示例：

```python
import torch

# 两条旧 residual stream。
hidden_streams = torch.tensor([
    [1.0, 0.0],
    [1.2, -0.2],
])

# 子层输出，例如 attention 输出。
attn_output = torch.tensor([0.5, -0.1])

# post 控制 attn_output 写入每条 stream 的比例。
post = torch.tensor([0.8, 0.2])

# comb 控制旧 stream 的混合。源码中使用 comb.T @ hidden_streams。
comb = torch.tensor([
    [0.7, 0.3],
    [0.4, 0.6],
])

old_part = comb.T @ hidden_streams
new_streams = post.unsqueeze(-1) * attn_output.unsqueeze(0) + old_part

print("old_part =", old_part)
print("new_streams =", new_streams)
```

输出：

```text
old_part = tensor([[ 1.180, -0.080],
                   [ 1.020, -0.120]])
new_streams = tensor([[ 1.580, -0.160],
                      [ 1.120, -0.140]])
```

该例子对应 `DeepseekV4DecoderLayer.forward` 中的写回公式：子层输出按 `post` 写入每条 residual stream，旧 residual streams 按 `comb.transpose(-1, -2)` 混合后保留。

对应 OP：

| mHC 动作 | OP |
|---|---|
| 多流拼接 | `flatten(start_dim=2)` |
| 生成混合参数 | `F.linear` |
| 切分参数 | `split` |
| 权重约束 | `sigmoid`、`softmax`、`sum`、除法 |
| 折叠多流 | `unsqueeze`、`mul`、`sum` |
| 写回多流 | `matmul`、`transpose`、`add` |

### 3.4 DeepSeek V4 Attention

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

# K=V 且 KV 已应用 RoPE，输出的 RoPE slice 需要用 -sin 做反向旋转。
attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)

# grouped output projection：先按 o_groups 分组，再投影回 hidden size。
grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
grouped = self.o_a_proj(grouped).flatten(2)
output = self.o_b_proj(grouped)
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
- attention sink 通过 `s_aux=self.sinks` 传入 backend。后端若不支持该项，会影响是否可替换为 SDPA 或 FlashAttention。

### 3.5 HCA：将多个 token 压缩成一个 compressed KV

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

`block_bias` 可见性最小示例：

```python
import torch

compress_rate = 2

# 假设已有两个 compressed entry：
# entry0 来自 tokens 0-1，entry1 来自 tokens 2-3。
position_ids = torch.arange(5).unsqueeze(0)
entry_indices = torch.arange(2)

# causal_threshold 表示每个 query 允许访问的 compressed entry 数量。
causal_threshold = (position_ids + 1) // compress_rate

block_bias = torch.zeros((1, 1, 5, 2))
block_bias = block_bias.masked_fill(
    entry_indices.view(1, 1, 1, -1) >= causal_threshold.unsqueeze(1).unsqueeze(-1),
    float("-inf"),
)

print("causal_threshold =", causal_threshold)
print("block_bias =", block_bias[0, 0])
```

输出：

```text
causal_threshold = tensor([[0, 1, 1, 2, 2]])
block_bias = tensor([[-inf, -inf],
                     [0., -inf],
                     [0., -inf],
                     [0., 0.],
                     [0., 0.]])
```

含义如下：

| query token | `causal_threshold` | 可访问 compressed entry |
|---|---:|---|
| token0 | 0 | 无 |
| token1 | 1 | entry0，也就是 tokens 0-1 的压缩结果 |
| token2 | 1 | entry0 |
| token3 | 2 | entry0、entry1 |
| token4 | 2 | entry0、entry1 |

说明：`entry0` 表示 window 0 的压缩结果，来自 tokens 0-1；`entry1` 表示 window 1 的压缩结果，来自 tokens 2-3。`block_bias` 不保存压缩内容本身，只控制当前 query 是否允许访问这些 compressed entry。

### 3.6 CSA：压缩后再用 Indexer 选 top-k compressed blocks

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

CSA top-k 最小示例：

```python
import torch

# 两个 query，各自与三个 compressed entry 计算相似度。
q = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
])
compressed_kv = torch.tensor([
    [1.0, 0.2],
    [0.2, 1.0],
    [0.8, 0.8],
])

# 对应源码中的 q @ compressed_kv.T，再经过 relu。
scores = torch.relu(q @ compressed_kv.T)

# 每个 query 选择两个 compressed entry。
top_k_indices = torch.topk(scores, 2, dim=-1).indices

print("scores =", scores)
print("top_k_indices =", top_k_indices)
```

输出：

```text
scores = tensor([[1.000, 0.200, 0.800],
                 [0.200, 1.000, 0.800]])
top_k_indices = tensor([[0, 2],
                        [1, 2]])
```

该例子对应 `DeepseekV4Indexer.forward` 的核心选择逻辑。第一个 query 更接近 entry0 和 entry2，第二个 query 更接近 entry1 和 entry2。后续 `scatter_` 会把这些位置写成 `0.0`，未命中的 compressed entry 保持 `-inf`。

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

### 3.7 DeepSeek V4 MoE

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
| 合并输出 | `index_add_` | 重复 token 累加，关注 dtype 和写冲突语义。 |

该 reference 实现用于说明 MoE 语义，不作为高性能在线路径。生产推理通常会将 dispatch、expert GEMM、combine 下沉到 fused kernel、grouped GEMM 或专用后端。

MoE combine 最小示例：

```python
import torch

# 两个 token，每个 token 选择两个 expert。
indices = torch.tensor([
    [0, 2],
    [1, 2],
])
weights = torch.tensor([
    [0.6, 0.4],
    [0.7, 0.3],
])

# 示例直接给出各 expert 对各 token 的输出。
expert_outputs = {
    (0, 0): torch.tensor([1.0, 0.0]),
    (0, 2): torch.tensor([0.5, 1.0]),
    (1, 1): torch.tensor([0.0, 2.0]),
    (1, 2): torch.tensor([1.0, 1.0]),
}

final = torch.zeros(2, 2)
for token_idx in range(2):
    for top_k_pos in range(2):
        expert_idx = int(indices[token_idx, top_k_pos])
        final[token_idx] += weights[token_idx, top_k_pos] * expert_outputs[(token_idx, expert_idx)]

print("final =", final)
```

输出：

```text
final = tensor([[0.800, 0.400],
                [0.300, 1.700]])
```

计算过程：

```text
token0 = 0.6 * expert0([1.0, 0.0]) + 0.4 * expert2([0.5, 1.0])
       = [0.8, 0.4]

token1 = 0.7 * expert1([0.0, 2.0]) + 0.3 * expert2([1.0, 1.0])
       = [0.3, 1.7]
```

该例子对应 `DeepseekV4Experts.forward` 中的三步：`where` 提取当前 expert 接收的 token，expert MLP 生成 `current`，再用 `index_add_` 将多个 expert 的贡献累加回 token 维度。

## 4. OP 与性能排查

排查性能问题时，应判断代码触发的底层行为，不以 Python 表达是否简短作为判断依据。

### 4.1 从 profiler 信号回到源码

Profiler 中的事件应映射回源码 OP 和 tensor 属性。常见对应关系如下：

| Profiler 信号 | 源码检查点 | 典型原因 |
|---|---|---|
| copy kernel、`memcpy`、`CopyTranspose` | `contiguous`、`to(device)`、stride 不兼容 `reshape` | layout 修正、device 转换、D2D/H2D/D2H copy。 |
| `musaStreamSynchronize`、host wait | `.item()`、`.cpu()`、`.tolist()`、`synchronize`、dynamic shape 输出长度回读 | CPU 等待 device 或 stream。 |
| `musaGraphLaunch` | `MUSAGraph.replay()` | Graph replay 生效，需继续检查 replay 前 copy 和固定 buffer。 |
| allocator 活跃、memset 多 | `empty`、`zeros`、`new_full`、`cat`、动态输出 OP | capture 内分配、临时 tensor 过多、padding 清理频繁。 |
| 小 kernel 数量多 | reference RMSNorm、SwiGLU、MoE loop、fallback attention | 未进入 fused kernel 或后端路径。 |
| kernel 名称不符合预期 | `linear`、attention backend、expert backend、quantized GEMM | dtype、layout、scale shape 或 capture 支持不满足后端契约。 |

MUSA trace 中可见的典型事件包括：`aten::contiguous` 对应 layout 修正，`aten::copy_` 可对应 replay buffer 更新或 H2D/D2D copy，`aten::item` 可对应 DEVICE scalar 回读，`MUSAGraph.replay()` 可对应 `musaGraphLaunch`。这些事件用于定位执行边界；稳定性能结论仍需固定输入规模、warmup、多轮统计和底层 runtime/API trace。

### 4.2 出现额外数据复制

优先检查：

```text
reshape / transpose / permute / contiguous / cat / stack / pad / to(device)
```

典型原因：

- stride 不兼容。
- attention 或 GEMM kernel 要求连续输入。
- `cat / stack` 生成新 tensor。
- dtype 或 device 转换引入拷贝。

### 4.3 出现同步等待

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

### 4.4 出现 allocator 抖动

优先检查：

```text
empty / zeros / new_full / cat / masked_select / nonzero / unique
```

处理方式：

- 按 batch/seq bucket 预分配 workspace。
- Graph replay 前固定 input/output/metadata tensor。
- 用 padding 和 fixed capacity 稳定 shape。

### 4.5 kernel 数量过多

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

### 4.6 Graph replay 失败或收益低

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

### 4.7 上层 OP 与 Runtime/Driver 的分工

排查顺序应先确认源码层约束，再进入 Runtime/Driver：

1. 确认 shape、dtype、device、stride、contiguous 状态。
2. 确认是否存在额外 copy、动态 shape、CPU 回读、频繁分配。
3. 确认 attention、GEMM、MoE、Graph 是否命中预期后端。
4. 若 OP 组织正确但目标 kernel 或 runtime API 异常，再进入 Runtime/Driver 分析。

提交 Runtime/Driver 分析时，至少提供以下信息：

| 信息 | 用途 |
|---|---|
| 最小复现脚本 | 固定问题输入和执行路径。 |
| shape / dtype / device / stride / contiguous | 判断 kernel 契约是否满足。 |
| stream / event / Graph 使用方式 | 判断同步和 capture 边界。 |
| profiler trace | 对齐 kernel、copy、sync、graph launch。 |
| 期望后端路径 | 判断是否 fallback。 |
| 是否存在 `.item()`、`.cpu()`、dynamic shape | 判断是否为上层同步或 shape 问题。 |

对于 DeepSeek V4 这类路径复杂的模型，生产推理通常将 PyTorch reference 语义下沉到后端实现：attention 进入 fused attention 或 paged attention，MLP/SwiGLU 进入 fused activation，MoE dispatch/expert/combine 进入 grouped GEMM 或专用 kernel，decode bucket 进入 Graph replay。源码中的 reference OP 用于解释语义和构造校验用例，不应直接等同于最终在线路径。

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

## 6. 工程附录：OP 用例、MUSA 验证与在线推理实现

前文基于 Transformers 中的 Llama 和 DeepSeek V4 源码建立分析主线。工程落地还需要补充两类信息：

1. OP 在 MUSA 环境中的可验证行为，包括固定 buffer、Graph replay、CPU-DEVICE 边界、Dynamic Shape 和 MoE combine。
2. 在线推理框架中的实现差异，包括 metadata 原地更新、多 stream 编排、KV cache 写入、HC/MHC 融合路径和 MoE SwiGLU 执行路径。

本节将这些内容整理为独立附录。附录中的 DeepSeek V4-Pro 和 SGLang 片段用于说明工程实现中的常见结构，不替代前文对 Transformers `modular_deepseek_v4.py` 的源码分析。

### 6.1 常见 OP 用例索引

逐个 OP 分析时，应先确定 OP 类别，再判断输入输出、layout、同步边界和 Graph 约束。下表给出工程排查中最常用的分类索引。

| OP 类别 | 代表 OP | 典型场景 | 重点检查项 |
|---|---|---|---|
| 创建初始化 | `empty / new_empty / empty_like / zeros / ones / full` | graph buffer、KV cache、logits、padding、临时 workspace | `empty` 后是否完整写入；Graph capture 后是否保持对象地址稳定。 |
| 原地更新 | `copy_ / _foreach_copy_ / fill_ / zero_ / masked_fill_` | metadata 写回、replay 前准备、padding 清理、mask 更新 | 是否覆盖有效范围；是否误替换 tensor 对象。 |
| Shape/Layout | `view / reshape / flatten / unsqueeze / squeeze / expand / permute / transpose / contiguous` | attention head layout、MoE buffer、KV cache layout、custom kernel 输入 | stride 是否兼容；`contiguous()` 是否引入 D2D copy；`expand` 是否产生 zero-stride view。 |
| 索引映射 | slice、advanced indexing、`gather / take_along_dim / index_select / scatter_ / tensor_split` | KV page/slot、MoE dispatch/combine、chunked prefill 切分 | index dtype/device 是否匹配；是否越界；重复 index 的累加语义是否明确。 |
| 序列组合 | `arange / repeat_interleave / cat / stack / pad / where` | positions 展开、bucket padding、batch 拼接、logits mask | 输出 shape 是否固定；`cat/stack` 是否分配新 tensor。 |
| 数学激活 | `sum / mean / max / min / clamp / square / rsqrt / sigmoid / gelu / silu / relu / softmax` | RMSNorm、SwiGLU、sampling、fallback reference | 是否拆成多个小 kernel；是否需要 fused kernel。 |
| 线性代数与路由 | `linear / matmul / mm / bmm / einsum / topk / sort / argsort / argmax` | GEMM、attention reference、MoE routing、sampling top-k | 是否命中目标 GEMM/attention backend；top-k 的 `k` 是否固定。 |
| dtype/device | `to / float / long / bfloat16` | metadata dtype、BF16/FP8/FP4、CPU/MUSA 边界 | 是否反复 cast；是否引入 H2D/D2H copy。 |
| CPU OP | Python 容器、CPU tensor、`.cpu / .tolist / .item`、tokenizer/I/O | scheduler、metadata 副本、日志、协议边界 | 是否作用在 DEVICE tensor；是否进入 decode 热点路径。 |
| Sync OP | `synchronize`、stream/event、隐式 sync、collective wait | benchmark、graph 边界、通信并行执行、CPU-DEVICE 等待 | wait 范围是否过大；是否破坏通信计算重叠。 |
| Dynamic Shape | `nonzero / unique / masked_select` | 调试、统计、CPU 规划逻辑 | 输出长度是否依赖数据；是否影响 Graph replay。 |
| Graph OP | `MUSAGraph`、capture/replay、fixed buffer update | decode graph、piecewise graph、固定地址 replay | capture 内是否存在 dynamic shape、allocator、CPU sync 或不支持 capture 的 backend。 |

### 6.2 MUSA 验证用例的组织方式

MUSA 用例不等同于完整模型 benchmark。它们用于确认局部语义和执行边界：固定地址是否稳定、`copy_` 是否只更新内容、Dynamic Shape 是否产生变长输出、CPU 回读是否形成同步边界、MoE combine 是否按权重累加。

| 用例 | 关键 OP | 验证内容 |
|---|---|---|
| replay buffer 更新 | `copy_ / _foreach_copy_` | replay 前更新固定 buffer，不替换 tensor 对象。 |
| metadata copy | `copy_ / fill_ / zero_` | tensor 字段原地更新，padding 槽位清理。 |
| Graph replay | `MUSAGraph.replay()` | 固定 shape、固定地址、固定执行路径是否可重放。 |
| Prefill metadata | `arange / repeat_interleave / pad / to(device)` | positions、request index、cache 写入位置是否可由 CPU 规划后上传。 |
| KV cache 写入 | advanced indexing、`div`、`remainder` | slot id 到 page/offset 的映射是否正确。 |
| MoE combine | `topk / softmax / gather / index_add_ / sum` | top-k expert 输出是否按 router 权重累加。 |
| Sampling 后处理 | `softmax / topk / argmax / cpu` | 大部分计算保留在 DEVICE，仅回传最终 token 或少量统计。 |
| Dynamic Shape | `nonzero / unique / masked_select` | 输出长度是否随输入内容变化。 |

### 6.3 MUSA Graph 最小用例

Graph replay 的核心约束是：capture 后不替换输入、输出和 metadata tensor 对象；replay 前只更新固定 buffer 内容。

```python
import torch
import torch.nn.functional as F

device = torch.device("musa:0")

# x、w、out 都是固定对象。capture 后不替换这些 tensor。
x = torch.zeros((2, 3), dtype=torch.float32, device=device)
w = torch.tensor(
    [[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]],
    dtype=torch.float32,
    device=device,
)
out = torch.empty((2, 2), dtype=torch.float32, device=device)

# warmup 让 allocator 和 kernel cache 在 capture 前稳定。
stream = torch.musa.Stream()
stream.wait_stream(torch.musa.current_stream())
with torch.musa.stream(stream):
    for _ in range(3):
        out.copy_(F.silu(x @ w))
torch.musa.current_stream().wait_stream(stream)

# capture 固定的 matmul -> SiLU -> copy_ 路径。
graph = torch.musa.MUSAGraph()
with torch.musa.stream(stream):
    graph.capture_begin()
    out.copy_(F.silu(x @ w))
    graph.capture_end()

# replay 前只更新 x 的内容，x 的对象和地址保持不变。
x.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device))
graph.replay()
torch.musa.synchronize()

print("x =", x.cpu().tolist())
print("out =", out.cpu().tolist())
```

输出示例：

```text
x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
out = [[13.999988555908203, 6.993622779846191],
       [32.0, 15.999998092651367]]
```

该用例验证三件事：

- `graph.replay()` 读取 capture 时绑定的 tensor 地址。
- `x.copy_(...)` 改变输入内容，不改变 graph 绑定关系。
- 输出回读发生在 replay 后，是明确的 CPU-DEVICE 边界。

### 6.4 Piecewise Graph 与 `torch.compile`

普通 Graph 捕获整段固定路径，收益大，约束也最强。Piecewise Graph 将 attention、MLP/MoE、RMSNorm、logits projection 等局部稳定片段分别 capture，再由 eager 调度连接片段边界。`torch.compile` 优化的是 PyTorch OP 图，Graph replay 固定的是某个 shape 和地址下的重复执行。

| 执行方式 | 适用条件 | 主要收益 | 主要风险 |
|---|---|---|---|
| 普通 Graph | shape 固定、地址固定、控制流固定、backend graph-safe | 减少整段 decode 的 Python 调度和 launch 开销 | 任一 dynamic shape、CPU sync、allocator 或不支持 capture 的 backend 都可能破坏 capture。 |
| Piecewise Graph | 大部分路径稳定，局部逻辑动态 | 固化 attention、MLP/MoE、logits 等局部片段 | piece 边界增多，需要管理输入输出地址、stream 顺序和生命周期。 |
| `torch.compile` | OP 组合稳定，可由编译器融合或重排 | 生成优化 callable 或后端 kernel | compile 成功不代表可 Graph capture；仍需检查 dynamic shape、sync 和 backend 支持。 |

工程选择顺序如下：

1. eager 路径先保证语义正确。
2. 对 shape/dtype/layout 稳定的局部计算使用 fused kernel 或 `torch.compile`。
3. 对固定 bucket 的 decode 路径使用 Graph capture/replay。
4. 对局部不稳定的路径使用 Piecewise Graph。
5. tokenizer、I/O、request queue、日志、CPU metadata、`.item()`、`.tolist()` 和动态分配保留在 Graph 外。

### 6.5 DeepSeek V4-Pro 推理实现中的量化 Linear

工程实现中的 DeepSeek V4-Pro 可能包含 FP4/FP8 权重、activation scale 和 packed layout。该路径与前文 Transformers reference 源码处在不同实现层级：reference 代码解释模型结构，量化推理实现还需要满足 kernel 接口约束。

```python
def linear(x, weight, bias=None):
    # 量化 GEMM 路径通常不混入 bias。
    assert bias is None

    if weight.dtype == torch.float4_e2m1fn_x2:
        # FP4 权重路径：activation 先分块量化，再进入 FP4 GEMM。
        x_quant, x_scale = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp4_gemm(x_quant, x_scale, weight, weight.scale, scale_dtype)

    if weight.dtype == torch.float8_e4m3fn:
        # FP8 权重路径：activation 和 weight scale 必须满足 kernel layout。
        x_quant, x_scale = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp8_gemm(x_quant, x_scale, weight, weight.scale, scale_dtype)

    # 普通 dtype 路径用于 dense linear 或 fallback。
    return F.linear(x, weight)
```

检查项：

| 项目 | 检查内容 |
|---|---|
| activation quant | 最后一维是否能按 block size 整除；scale shape 是否与 GEMM K 维一致。 |
| weight dtype | FP4、FP8、BF16/FP16 是否进入预期路径。 |
| packed layout | FP4 是否按两个值打包；runtime kernel 是否按相同格式解释。 |
| scale tensor | dtype、shape、stride、contiguous 状态是否满足 kernel。 |
| fallback | 普通 `F.linear` 是否只作为 reference 或受控 fallback。 |

### 6.6 `act_quant`、FP8 GEMM 与 FP4 GEMM 的约束

量化 GEMM 的关键不是单个 `linear` API，而是 activation、weight、scale 三类输入同时满足约束。

```python
def act_quant(x, block_size=128, scale_dtype=torch.float32, inplace=False):
    # 量化沿最后一维分块，通常对应 GEMM 的 K 维。
    k_dim = x.size(-1)
    assert k_dim % block_size == 0

    # kernel 通常要求连续输入；非连续输入需要先 materialize。
    x_contiguous = x.contiguous()

    # 输出量化 activation，同时生成每个 block 的 scale。
    y = torch.empty_like(x_contiguous) if inplace else torch.empty_like(x_contiguous, dtype=torch.float8_e4m3fn)
    scale = x_contiguous.new_empty(*x_contiguous.size()[:-1], k_dim // block_size, dtype=scale_dtype)

    # 实际实现会调用后端量化 kernel。
    # kernel(x_contiguous.view(-1, k_dim), y.view(-1, k_dim), scale.view(-1, k_dim // block_size))

    if inplace:
        x.copy_(y)
        return x
    return y, scale
```

性能和正确性风险：

- `x.contiguous()` 可能引入 D2D copy。
- inplace quant 会覆盖原 activation，后续路径不得再依赖原值。
- FP4 weight 的存储 K 维可能是逻辑 K 维的一半，scale 仍按逻辑 block 管理。
- GEMM wrapper 通常要求 activation、weight、activation scale、weight scale 均为 contiguous。

### 6.7 DeepSeek V4-Pro Attention 的工程路径

工程推理实现中的 compressed attention 通常拆成 Q 路径、KV 路径、sparse index 和 attention kernel 四部分。

```python
# Q 路径：低秩 projection 后整理成 head layout。
q_residual = q_norm(wq_a(x))
q = wq_b(q_residual)
q = q.unflatten(-1, (num_heads, head_dim))

# RoPE 只作用在指定维度；Q norm 可用多个 OP 表达 reference 语义。
apply_rotary_emb(q[..., -rope_dim:], freqs_cis)
q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)

# KV 路径：norm、RoPE、非 RoPE 维量化。
kv = kv_norm(wkv(x))
apply_rotary_emb(kv[..., -rope_dim:], freqs_cis)
act_quant(kv[..., :-rope_dim], block_size=64, inplace=True)

# sparse index 合并 sliding window index 和 compressed index。
window_index = get_window_topk_idxs(window_size, batch, seq_len, start_pos)
compressed_index = indexer(x, q_residual, start_pos, offset)
topk_index = torch.cat([window_index, compressed_index], dim=-1).int()

# 后端 attention kernel 直接消费 Q、KV、sink 和 sparse index。
out = sparse_attn(q, kv, attention_sink, topk_index, softmax_scale)
```

检查项：

| 模块 | 主要 OP | 检查项 |
|---|---|---|
| Q layout | `unflatten / view` | head layout 是否符合 attention kernel；stride 是否兼容。 |
| Q norm | `square / mean / rsqrt / mul` | 是否被 fused；是否产生多个小 kernel。 |
| KV quant | `act_quant(..., inplace=True)` | 非 RoPE 维是否可覆盖；scale 是否匹配。 |
| sparse index | `topk / where / cat / int` | invalid index 是否使用 sentinel；index dtype 是否为 kernel 需要的类型。 |
| attention kernel | custom sparse attention | Q/KV layout、sink、top-k index、mask 和 Graph capture 是否同时满足。 |

### 6.8 DeepSeek V4-Pro MoE：Router、Expert 与 Combine

MoE 工程路径包含 router、top-k、dispatch、expert 计算和 combine。reference 代码通常清晰，但会包含动态 token 数、Python loop 和 CPU 回读。

```python
# router score 使用 FP32 计算，提升数值稳定性。
scores = linear(hidden.float(), router_weight.float())
scores = scores.softmax(dim=-1)

# 每个 token 选择 top-k expert。
indices = scores.topk(top_k, dim=-1).indices
weights = scores.gather(1, indices)
weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)

# reference 路径统计每个 expert 的 token 数。
# 若 indices 在 DEVICE 上，tolist() 会形成 CPU-DEVICE 同步。
counts = torch.bincount(indices.flatten(), minlength=num_experts).tolist()

final = torch.zeros_like(hidden, dtype=torch.float32)
for expert_id in range(num_experts):
    if counts[expert_id] == 0:
        continue
    token_idx, top_k_pos = torch.where(indices == expert_id)
    expert_out = experts[expert_id](hidden[token_idx])
    final[token_idx] += expert_out * weights[token_idx, top_k_pos, None]
```

热点路径通常不采用上述 reference 组织方式。更稳定的工程实现会使用 fixed capacity、padded metadata、grouped GEMM、fused dispatch/combine 或 all-to-all 专用路径，减少动态 shape、CPU 回读和 per-expert Python loop。

### 6.9 HC/MHC 的 reference 路径与融合路径

HC/MHC 的 reference 计算通常由 `flatten -> float -> square/mean/rsqrt -> linear -> split/sinkhorn -> sum` 构成。该路径便于验证数学语义，但 kernel 数量和中间 tensor 较多。

```python
def hc_pre_reference(x, hc_fn, hc_scale, hc_base):
    # x shape=[tokens, hc_mult, hidden]。
    original_shape = x.shape
    original_dtype = x.dtype

    # 合并 hc_mult 和 hidden，使用 FP32 计算 mixing 参数。
    x_flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(x_flat, hc_fn) * rsqrt

    # split/sinkhorn 生成 pre、post、comb。
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, hc_eps)

    # pre 沿 HC copy 维加权求和，输出普通 hidden。
    y = (pre.unsqueeze(-1) * x_flat.view(original_shape)).sum(dim=1)
    return y.to(original_dtype), post, comb
```

在线推理中常见三类路径：

| 路径 | 特点 | 检查项 |
|---|---|---|
| PyTorch reference | 语义清晰，便于对齐 | kernel 数量、dtype cast、中间 tensor。 |
| TileLang/MHC fused | norm、mixing、pre 输出融合 | kernel 是否 graph-safe；输入 layout 是否满足。 |
| DeepGEMM prenorm | BF16 输入、FP32 weight/workspace | `contiguous()`、workspace 生命周期、输出 scale 语义。 |

HC post 的核心公式是：

```text
new_streams = post * sublayer_output + comb * residual_streams
```

实现中通常通过 broadcast、`unsqueeze`、`sum` 或 `matmul` 完成。排查时要确认 `comb` 的转置方向、dtype cast 和 residual stream 维度。

### 6.10 权重转换与 Packed Layout

量化权重的加载阶段会决定运行时 GEMM 是否能直接消费 checkpoint 数据。常见转换包括 FP8 反量化、FP4 packed view、scale 提取和 tile layout 重排。

```python
# FP8 权重反量化示意。
weight = state_dict[name]
scale = state_dict.pop(name.replace("weight", "scale"))

# 按 block 恢复 out/in 维，再用 scale 反量化。
weight = weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float()
weight = weight * scale[:, None, :, None].float()

# 恢复普通矩阵 layout，并保存为 BF16。
state_dict[name] = weight.flatten(2, 3).flatten(0, 1).bfloat16()
```

```python
# FP4 packed 数据解析示意。
packed = packed.view(torch.uint8)
low = packed & 0x0F
high = (packed >> 4) & 0x0F

# 查表恢复两个 FP4 逻辑值，再展平成连续元素。
values = torch.stack([fp4_table[low.long()], fp4_table[high.long()]], dim=-1)
values = values.flatten(2)
```

检查项：

- packed 数据的 endian、low/high nibble 顺序是否与 kernel 一致。
- `view(torch.uint8)` 是重新解释底层存储，不是数值计算转换。
- scale 是否与 weight block 维度一致。
- 转换后的 weight 是否满足 runtime GEMM 的 tile layout 和 contiguous 要求。

### 6.11 SGLang DeepSeek V4：多 stream Attention Prepare

在线推理实现会把 Q、KV、compressor 和 indexer 拆到多条 stream 上执行，减少主 stream 等待。该结构和 Transformers reference forward 的单路径执行不同。

```python
# 当前 stream 负责主计算路径。
current_stream = torch.cuda.current_stream()

# 辅助 stream 分别服务 KV、compressor 和 indexer。
stream_kv = alt_streams[0]
stream_compressor = alt_streams[1]
stream_indexer = alt_streams[2]

# 辅助 stream 等待主 stream，确保输入 tensor 已就绪。
stream_kv.wait_stream(current_stream)
stream_compressor.wait_stream(current_stream)
stream_indexer.wait_stream(current_stream)

# 主 stream 计算 Q LoRA 中间态，并记录 event。
q_lora = compute_q_a(x)
q_lora_ready = current_stream.record_event()

# indexer 可在独立 stream 上生成 sparse/compressed attention metadata。
with torch.cuda.stream(stream_indexer):
    stream_indexer.wait_event(q_lora_ready)
    indexer(x=x, q_lora=q_lora, forward_batch=forward_batch)

# KV 路径放在独立 stream，与 Q-B 或 indexer 并行。
with torch.cuda.stream(stream_kv):
    kv = compute_kv(x, positions)
    if overlap_store_cache:
        attn_backend.store_cache(layer_id=layer_id, swa_k=kv, forward_batch=forward_batch)

# compressor 路径独立执行。
with torch.cuda.stream(stream_compressor):
    attn_backend.forward_core_compressor(x, forward_batch, layer_id, compressor)

# 主 stream 继续计算 Q-B。
q = compute_q_b(q_lora, positions)

# 返回前 join 三条辅助 stream，保证 q/kv/index/compressor 状态可用。
current_stream.wait_stream(stream_kv)
current_stream.wait_stream(stream_compressor)
current_stream.wait_stream(stream_indexer)
```

排查重点：

- `wait_stream` 和 `wait_event` 的范围是否过大。
- `store_cache` 是否能和后续计算重叠。
- indexer 和 compressor 是否依赖同一份 metadata。
- Graph 或 piecewise graph 场景中，`q_out.copy_(q)` 是否写入固定输出 buffer。

### 6.12 SGLang DeepSeek V4：metadata 原地更新与 Graph replay

Graph replay 使用固定 metadata 对象。动态请求只更新字段内容，不替换 capture 时绑定的对象。

```python
def copy_attention_metadata(dst, src):
    # 结构性字段必须保持与 capture 时一致。
    assert dst.page_size == src.page_size
    assert dst.c4_sparse_topk == src.c4_sparse_topk

    # tensor 字段原地 copy，保持 dst 对象地址不变。
    dst.req_pool_indices.copy_(src.req_pool_indices)
    dst.seq_lens.copy_(src.seq_lens)
    dst.out_cache_loc.copy_(src.out_cache_loc)
    dst.page_table.copy_(src.page_table)

    # 特殊 backend metadata 按实现约定更新对象指针或内部字段。
    dst.flashmla_metadata = src.flashmla_metadata
```

replay 准备流程：

```python
# 选择已经 capture 的 bucket metadata。
chosen_metadata = graph_metadata_of_bucket[bucket][batch_size]

# temp_metadata 由本轮动态请求构造，但不会直接进入 graph。
copy_attention_metadata(chosen_metadata, temp_metadata)

# forward 读取固定 metadata 对象。
forward_metadata = chosen_metadata
graph.replay()
```

该设计把动态请求收敛到 Graph 外：request 数量、seq lens、page table、out cache location 在 CPU scheduler 中确定，再写入固定 tensor。Graph 内只读取固定 shape、固定地址的 metadata。

### 6.13 SGLang DeepSeek V4：CPU/GPU buffer 分开更新

decode graph buffer 通常分成 CPU metadata buffer 和 GPU tensor buffer。CPU buffer 服务 scheduler 和 bucket 选择，GPU buffer 服务 Graph replay。

```python
# CPU seq_lens buffer 保持在 CPU，scheduler 可直接读取。
seq_lens_cpu = torch.full(
    (max_batch_size,),
    seq_len_fill_value,
    dtype=torch.int32,
    device="cpu",
)

# GPU tensor 字段按 dtype 分组批量 copy，减少 Python 循环调度。
grouped_foreach_copy(device_dsts, device_srcs)

if batch_size != raw_batch_size:
    # bucket 大于实际 batch 时，先清理 padding 槽位。
    seq_lens_cpu.fill_(seq_len_fill_value)

# 仅覆盖实际 batch 前缀，保持固定长度。
seq_lens_cpu[:raw_batch_size].copy_(forward_batch.seq_lens_cpu)
```

检查项：

- CPU metadata 不应通过 DEVICE tensor `.item()` 高频回读。
- GPU tensor buffer 使用 `copy_ / _foreach_copy_` 更新内容。
- bucket padding 槽位必须有稳定填充值。
- CPU/GPU buffer 的生命周期应覆盖异步 copy 和 graph replay。

### 6.14 SGLang DeepSeek V4：HC/MHC 路径选择

在线推理中，HC/MHC 通常存在 reference、TileLang 和 DeepGEMM 多条路径。排查时不应仅验证 reference 语义，还要确认实际命中的路径。

| 路径 | 执行方式 | 适用检查 |
|---|---|---|
| Torch reference | `flatten -> float -> square/mean/rsqrt -> linear -> split/sinkhorn -> sum` | 语义对齐、最小复现、fallback 定位。 |
| TileLang/MHC fused | 单个或少量 kernel 完成 norm、mixing、pre 输出 | kernel 是否命中；Graph capture 是否支持；layout 是否稳定。 |
| DeepGEMM HC prenorm | BF16 输入 + FP32 weight/workspace | `contiguous()`、workspace、scale 语义和 dtype cast。 |

HC post 仍需关注 broadcast 和 combine：

```python
def hc_post_reference(x, residual, post, comb):
    # post 作用于当前子层输出。
    output_part = post.unsqueeze(-1) * x.unsqueeze(1)

    # comb 作用于 residual 的 HC copy 维。
    residual_part = (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)

    return (output_part + residual_part).type_as(x)
```

### 6.15 SGLang DeepSeek V4：MoE SwiGLU Clamp

DeepSeek V4 的 MoE expert 中常见 gate/up 限幅。GEMM1 输出按最后一维拆成 gate 和 up 两支，gate 走 SiLU，两支都按限制值裁剪，再逐元素相乘。

```python
def swiglu_silu_clamp_mul(x, limit):
    # GEMM1 输出按最后一维拆成 gate/up。
    gate, up = x.chunk(2, dim=-1)

    # gate 分支执行 SiLU，并限制上界。
    gate = torch.nn.functional.silu(gate)
    gate = gate.clamp(max=limit)

    # up 分支限制上下界。
    up = up.clamp(min=-limit, max=limit)

    # SwiGLU 输出。
    return gate * up
```

执行路径分为两类：

- 非融合路径：对 GEMM1 cache 原地 `clamp_`，再调用 activation/multiply。
- 融合路径：将 clamp、SiLU 和 multiply 下沉到专用 kernel。

MUSA 适配时应检查实际路径是否命中 `SwishGLU` 或对应 fused kernel。若 profiler 中看到多个小 elementwise kernel，应回到 OP 组织和后端 dispatch 检查是否 fallback。

### 6.16 附录检查清单

工程分析最终落到以下检查项：

1. **功能语义**：输入、输出、shape 和数值是否与 reference 对齐。
2. **layout 契约**：stride、contiguous、storage offset 是否满足后端 kernel。
3. **dtype 契约**：activation、weight、scale、index 和 metadata dtype 是否匹配。
4. **内存行为**：是否存在额外 copy、allocator、memset 或未清理 padding。
5. **同步边界**：是否存在 `.item()`、`.tolist()`、`.cpu()`、`synchronize()` 或过早 collective wait。
6. **Graph 约束**：shape、地址、控制流和 backend 是否支持 capture/replay。
7. **后端路径**：是否命中 fused attention、quantized GEMM、grouped GEMM、TileLang、DeepGEMM 或 MUSA native kernel。
8. **Profiler 证据**：kernel 名称、copy、sync、graph launch 和 CPU wait 是否与预期一致。
