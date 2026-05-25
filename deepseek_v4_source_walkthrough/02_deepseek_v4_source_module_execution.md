# DeepSeekV4 源码级执行分析

源码文件：

- `repos/transformers/src/transformers/models/deepseek_v4/configuration_deepseek_v4.py`
- `repos/transformers/src/transformers/models/deepseek_v4/modular_deepseek_v4.py`
- `repos/transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py`

`modular_deepseek_v4.py` 是模块化源码，结构更清晰。`modeling_deepseek_v4.py` 是自动生成后的运行文件，实际导入模型时使用。`DeepseekV4ForCausalLM.forward` 只在生成后的 `modeling_deepseek_v4.py` 中展开；`modular_deepseek_v4.py` 里该类继承 `MixtralForCausalLM` 并保持 `pass`。

本文按真实调用顺序拆解源码。每节包含源码片段、执行解释和最小示例。

## 1. 源码调用链

一次 forward 的主链路如下：

```text
DeepseekV4ForCausalLM.forward
  -> DeepseekV4Model.forward
    -> embed_tokens
    -> create_sliding_window_causal_mask
    -> expand to hc_mult streams
    -> build RoPE: main / compress
    -> for each DeepseekV4DecoderLayer.forward
      -> DeepseekV4HyperConnection.forward
      -> DeepseekV4Attention.forward
        -> Q projection
        -> shared KV projection
        -> sliding cache update
        -> optional HCA / CSA compressor
        -> attention backend
        -> grouped output projection
      -> DeepseekV4HyperConnection.forward
      -> DeepseekV4SparseMoeBlock.forward
        -> router
        -> routed experts
        -> shared expert
    -> DeepseekV4HyperHead.forward
    -> final RMSNorm
  -> lm_head
  -> optional loss and router aux loss
```

关键形状：

| 阶段 | 形状 |
| --- | --- |
| `input_ids` | `[B, S]` |
| embedding 后 | `[B, S, D]` |
| mHC 多残差流 | `[B, S, hc_mult, D]` |
| attention / MoE 子层输入 | `[B, S, D]` |
| Q | `[B, num_heads, S, head_dim]` |
| shared KV | `[B, 1, KV_len, head_dim]` |
| compressed KV | `[B, 1, compressed_len, head_dim]` |
| logits | `[B, kept_tokens, vocab_size]` |

最小示例：

```text
B = 1
S = 4
D = 2
hc_mult = 2

input_ids       : [1, 4]
inputs_embeds   : [1, 4, 2]
hidden_streams  : [1, 4, 2, 2]
collapsed input : [1, 4, 2]
final hidden    : [1, 4, 2]
logits          : [1, 4, vocab_size]
```

配套模拟脚本：

```bash
cd /home/mtuser/workspace
python insights/deepseek_v4_source_walkthrough/run_deepseek_v4_tiny_trace.py
```

该脚本不依赖 PyTorch、Transformers、模型权重或 GPU。它模拟源码的执行结构，用于核对模块顺序、形状变化和关键中间结果。

## 2. 配置入口：`DeepseekV4Config.__post_init__`

源码位置：`configuration_deepseek_v4.py:244`

核心源码：

```python
legacy_compress_ratios = kwargs.pop("compress_ratios", None)
legacy_compress_rate_csa = kwargs.pop("compress_rate_csa", None)
legacy_compress_rate_hca = kwargs.pop("compress_rate_hca", None)
legacy_num_hash_layers = kwargs.pop("num_hash_layers", None)
legacy_qk_rope_head_dim = kwargs.pop("qk_rope_head_dim", None)

if self.compress_rates is None:
    self.compress_rates = dict(self.default_compress_rates)
if legacy_compress_rate_csa is not None:
    self.compress_rates["compressed_sparse_attention"] = legacy_compress_rate_csa
if legacy_compress_rate_hca is not None:
    self.compress_rates["heavily_compressed_attention"] = legacy_compress_rate_hca

if self.layer_types is None and legacy_compress_ratios is not None:
    self.layer_types = [_COMPRESS_RATIO_TO_LAYER_TYPE[r] for r in legacy_compress_ratios]
if self.layer_types is None:
    interleave = [
        "compressed_sparse_attention" if i % 2 else "heavily_compressed_attention"
        for i in range(max(n - 2, 0))
    ]
    self.layer_types = ["heavily_compressed_attention"] * min(n, 2) + interleave
self.layer_types = list(self.layer_types[:n])
```

执行解释：

- `compress_rates` 决定 HCA / CSA 每多少个 token 压缩成一个 compressed KV。
- `layer_types` 决定每层 attention 使用滑窗、HCA 还是 CSA。
- 旧 checkpoint 中的 `compress_ratios` 会被转换成新的 `layer_types`。
- 默认层类型为前两层 HCA，后续 HCA / CSA 交替。

最小示例：

```python
# 假设 num_hidden_layers = 4，且未显式传 layer_types。
n = 4
interleave = [
    "heavily_compressed_attention",   # i = 0
    "compressed_sparse_attention",    # i = 1
]
layer_types = [
    "heavily_compressed_attention",
    "heavily_compressed_attention",
    "heavily_compressed_attention",
    "compressed_sparse_attention",
]
```

MoE 类型也在同一个函数中生成：

```python
if self.mlp_layer_types is None:
    n_hash = legacy_num_hash_layers if legacy_num_hash_layers is not None else self.default_num_hash_layers
    self.mlp_layer_types = ["hash_moe"] * min(n, n_hash) + ["moe"] * max(0, n - n_hash)
self.mlp_layer_types = list(self.mlp_layer_types[:n])
```

执行解释：

- 前 `n_hash` 层使用 `hash_moe`，专家由 token id 查表得到。
- 后续层使用 `moe`，专家由 TopK router 动态选择。

RoPE 参数也在这里拆成两套：

```python
main = {
    "rope_type": "default",
    "rope_theta": self.rope_theta,
    "partial_rotary_factor": self.partial_rotary_factor,
}
compress = {
    **yarn,
    "rope_theta": self.compress_rope_theta,
    "partial_rotary_factor": self.partial_rotary_factor,
    "attention_factor": 1.0,
}
self.rope_parameters = {"main": main, "compress": compress}
```

执行解释：

- `main` 给纯滑窗 attention 使用。
- `compress` 给 HCA、CSA 和 Indexer 使用。
- 两套 RoPE 的 theta 不同，压缩分支使用更大的 `compress_rope_theta`。

## 3. 语言模型入口：`DeepseekV4ForCausalLM.forward`

源码位置：`modeling_deepseek_v4.py:1417`

核心源码：

```python
output_router_logits = (
    output_router_logits if output_router_logits is not None else self.config.output_router_logits
)

outputs: MoeModelOutputWithPast = self.model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_values=past_key_values,
    inputs_embeds=inputs_embeds,
    use_cache=use_cache,
    output_router_logits=output_router_logits,
    **kwargs,
)

hidden_states = outputs.last_hidden_state
slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
logits = self.lm_head(hidden_states[:, slice_indices, :])
```

执行解释：

- `self.model(...)` 负责 Transformer 主体。
- `outputs.last_hidden_state` 是最终 hidden states，形状为 `[B, S, D]`。
- `lm_head` 把 hidden states 投影到词表维度。
- `logits_to_keep=0` 表示保留全部 token。
- `logits_to_keep=1` 表示只计算最后一个 token 的 logits，常用于自回归推理。

最小示例：

```python
# hidden_states.shape = [1, 4, 2]
# vocab_size = 3
# logits_to_keep = 1

slice_indices = slice(-1, None)
selected_hidden = hidden_states[:, slice_indices, :]  # [1, 1, 2]
logits = lm_head(selected_hidden)                     # [1, 1, 3]
```

loss 和 router auxiliary loss：

```python
if labels is not None:
    loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

if output_router_logits:
    aux_loss = load_balancing_loss_func(
        outputs.router_logits,
        self.num_experts,
        self.num_experts_per_tok,
        attention_mask,
    )
    if labels is not None:
        loss += self.router_aux_loss_coef * aux_loss.to(loss.device)
```

执行解释：

- `labels` 存在时计算语言模型 loss。
- `output_router_logits=True` 时计算 MoE 负载均衡辅助 loss。
- 辅助 loss 只在训练或分析 MoE 路由时有意义。

## 4. 模型主体：`DeepseekV4Model.forward`

源码位置：`modular_deepseek_v4.py:1112`

核心源码：

```python
if (input_ids is None) ^ (inputs_embeds is not None):
    raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

return_cache = past_key_values if use_cache else None
if past_key_values is None:
    past_key_values = DynamicCache(config=self.config)
if inputs_embeds is None:
    inputs_embeds = self.embed_tokens(input_ids)
```

执行解释：

- `input_ids` 和 `inputs_embeds` 必须二选一。
- 未传 `past_key_values` 时，内部创建 `DynamicCache`。
- 传 `input_ids` 时，先通过 `embed_tokens` 变成 `[B, S, D]`。

位置和 mask：

```python
if position_ids is None:
    past_seen = past_key_values.get_seq_length()
    position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
    position_ids = position_ids.unsqueeze(0)

if isinstance(attention_mask, dict):
    causal_mask = next(iter(attention_mask.values()))
else:
    causal_mask = create_sliding_window_causal_mask(
        config=self.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
```

执行解释：

- `position_ids` 从 cache 已有长度开始计数。
- mask 是滑窗 causal mask，所有层先共用这份滑窗 mask。
- HCA / CSA 后续会在 attention 内部拼接 compressed KV，并把对应 mask 拼到滑窗 mask 后面。

mHC 扩展和 RoPE 预计算：

```python
hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
position_embeddings = {
    "main": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main"),
    "compress": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="compress"),
}
```

执行解释：

- `unsqueeze(2)` 在 token 维和 hidden 维之间插入残差流维度。
- `expand(..., hc_mult, ...)` 把每个 token 的 embedding 展成 `hc_mult` 条残差流。
- `contiguous()` 生成连续内存，便于后续矩阵计算。
- `main` RoPE 给滑窗层使用。
- `compress` RoPE 给 HCA、CSA、Indexer 使用。

最小示例：

```python
# inputs_embeds.shape = [1, 4, 2]
# hc_mult = 2

hidden_states = inputs_embeds.unsqueeze(2)
# [1, 4, 1, 2]

hidden_states = hidden_states.expand(-1, -1, 2, -1)
# [1, 4, 2, 2]
```

层循环和最终输出：

```python
for layer in self.layers:
    hidden_states = layer(
        hidden_states,
        position_embeddings=position_embeddings,
        position_ids=position_ids,
        attention_mask=causal_mask,
        input_ids=input_ids,
        past_key_values=past_key_values,
        **kwargs,
    )

hidden_states = self.norm(self.hc_head(hidden_states))
return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=return_cache)
```

执行解释：

- 每层输入和输出都是 `[B, S, hc_mult, D]`。
- 所有 decoder layer 结束后，`hc_head` 把多残差流折叠成 `[B, S, D]`。
- 最后执行 RMSNorm。

### 4.1 归一化模块：`DeepseekV4RMSNorm` 和 `DeepseekV4UnweightedRMSNorm`

源码位置：

- `modeling_deepseek_v4.py:46`
- `modeling_deepseek_v4.py:66`
- `modular_deepseek_v4.py:66`
- `modular_deepseek_v4.py:70`

`modular_deepseek_v4.py` 中 `DeepseekV4RMSNorm` 继承 `DeepseekV3RMSNorm`。生成后的 `modeling_deepseek_v4.py` 展开了完整实现。

核心源码：

```python
class DeepseekV4RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
```

执行解释：

- RMSNorm 只按最后一维计算均方值，不减均值。
- `to(torch.float32)` 保证归一化计算稳定。
- `self.weight` 是可学习缩放参数。
- 该模块用于 decoder layer 的 `input_layernorm`、`post_attention_layernorm` 和模型末尾的 `norm`。

最小示例：

```python
x = [3.0, 4.0]
eps = 0.0
weight = [1.0, 1.0]

variance = (3.0**2 + 4.0**2) / 2
# variance = 12.5

scale = 1 / sqrt(12.5)
# scale = 0.2828427

output = [3.0 * scale, 4.0 * scale]
# output = [0.8485, 1.1314]
```

`DeepseekV4UnweightedRMSNorm` 没有可学习 `weight`：

```python
class DeepseekV4UnweightedRMSNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)
```

执行解释：

- 该模块只做数值归一化，不改变可学习尺度。
- 它用于 mHC 的输入归一化和 attention 中的 `q_b_norm`。

最小示例：

```python
x = [6.0, 8.0]
variance = (36.0 + 64.0) / 2
# variance = 50.0

output = [6.0 / sqrt(50.0), 8.0 / sqrt(50.0)]
# output = [0.8485, 1.1314]
```

### 4.2 模型基类：`DeepseekV4PreTrainedModel`

源码位置：`modeling_deepseek_v4.py:1143`

该类没有自己的 forward。它定义模型能力、输出捕获、初始化规则和生成限制。

关键源码：

```python
_supports_flash_attn = False
_supports_sdpa = False
_supports_flex_attn = False
_can_compile_fullgraph = False
_supports_attention_backend = True
_can_record_outputs = {
    "router_logits": OutputRecorder(DeepseekV4TopKRouter, index=0),
    "hidden_states": DeepseekV4DecoderLayer,
    "attentions": DeepseekV4Attention,
}
_is_stateful = True
```

执行解释：

- FlashAttention、SDPA、FlexAttention 被关闭，实际 attention 走 eager backend。
- compressor 会动态追加 compressed KV，非 eager backend 难以直接复用固定长度 mask。
- `_is_stateful=True` 表示生成时 cache 不能任意回滚。
- `_can_record_outputs` 允许框架捕获 router logits、hidden states 和 attentions。

初始化源码：

```python
if isinstance(module, (DeepseekV4TopKRouter, DeepseekV4HashRouter)):
    init.normal_(module.weight, mean=0.0, std=std)
    if isinstance(module, DeepseekV4TopKRouter):
        init.zeros_(module.e_score_correction_bias)
    if isinstance(module, DeepseekV4HashRouter):
        init.zeros_(module.tid2eid)
elif isinstance(module, DeepseekV4Attention):
    init.zeros_(module.sinks)
elif isinstance(module, DeepseekV4HyperConnection):
    init.normal_(module.fn, mean=0.0, std=std)
    init.zeros_(module.base)
    init.ones_(module.scale)
elif isinstance(module, (DeepseekV4HCACompressor, DeepseekV4CSACompressor, DeepseekV4Indexer)):
    init.zeros_(module.position_bias)
```

执行解释：

- router 权重按正态分布初始化。
- TopK router 的校正 bias 初始为 0。
- Hash router 的 `tid2eid` 初始为 0，真实映射来自 checkpoint。
- attention sink 初始为 0。
- mHC 的 `base=0`、`scale=1`。
- compressor 的 `position_bias=0`。

最小示例：

```python
# 初始化后的 HashRouter 行为。
tid2eid = [
    [0, 0],  # token id 0
    [0, 0],  # token id 1
]

input_ids = [1]
selected_experts = tid2eid[1]
# selected_experts = [0, 0]

# 加载 checkpoint 后，tid2eid 会被真实专家表覆盖。
```

## 5. Decoder 层：`DeepseekV4DecoderLayer.forward`

源码位置：`modular_deepseek_v4.py:982`

核心源码：

```python
dtype = hidden_states.dtype

post, comb, collapsed = self.attn_hc(hidden_states)
attn_output, _ = self.self_attn(self.input_layernorm(collapsed), **kwargs)
hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
    comb.to(dtype).transpose(-1, -2), hidden_states
)

post, comb, collapsed = self.ffn_hc(hidden_states)
mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
return post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(
    comb.to(dtype).transpose(-1, -2), hidden_states
)
```

执行解释：

- `attn_hc` 把多残差流折叠成 attention 输入 `collapsed`。
- `self_attn(...)` 只接收 `[B, S, D]`。
- attention 输出通过 `post` 写入各条残差流。
- 旧残差流通过 `comb.T @ hidden_states` 混合后保留。
- `ffn_hc` 对 MoE 子层重复同样逻辑。
- `self.mlp(...)` 是 MoE block。

这一层不是普通残差写法：

```text
x = x + attention(x)
x = x + mlp(x)
```

真实源码采用：

```text
hidden_streams
  -> mHC collapse
  -> sublayer
  -> post * sublayer_output + comb.T @ old_streams
```

最小示例：

```python
# 单个 token，hc_mult = 2，hidden_dim = 1
hidden_streams = [[1.0], [2.0]]
attn_output = [10.0]

# mHC 给出的权重
post = [0.3, 0.7]
comb = [
    [0.8, 0.2],
    [0.2, 0.8],
]

# 写回第 0 条流：
# post[0] * 10 + comb[0,0] * 1 + comb[1,0] * 2 = 3 + 0.8 + 0.4 = 4.2

# 写回第 1 条流：
# post[1] * 10 + comb[0,1] * 1 + comb[1,1] * 2 = 7 + 0.2 + 1.6 = 8.8
```

## 6. mHC 折叠与回写：`DeepseekV4HyperConnection.forward`

源码位置：`modular_deepseek_v4.py:805`

核心源码：

```python
hc = self.hc_mult
flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
pre_scale, post_scale, comb_scale = self.scale.unbind(0)

pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
post = 2 * torch.sigmoid(post_w * post_scale + post_b)
comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
for _ in range(self.hc_sinkhorn_iters - 1):
    comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
    comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)

collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
return post, comb, collapsed
```

执行解释：

- `flatten(start_dim=2)` 把 `[hc_mult, D]` 合并为 `hc_mult * D`。
- `F.linear(...).split(...)` 一次生成三组权重：`pre`、`post`、`comb`。
- `pre` 控制多残差流如何合成一条子层输入。
- `post` 控制子层输出写回每条残差流的比例。
- `comb` 控制旧残差流之间如何混合。
- Sinkhorn 归一化让 `comb` 的行列和更稳定，避免残差流无限放大。

最小示例：

```python
# hidden_streams.shape = [B=1, S=1, hc_mult=2, D=2]
stream0 = [1.0, 0.0]
stream1 = [3.0, 2.0]

# 假设源码计算出的 pre 为：
pre = [0.25, 0.75]

collapsed = 0.25 * stream0 + 0.75 * stream1
# collapsed = [2.5, 1.5]
```

最终折叠使用 `DeepseekV4HyperHead.forward`。

源码位置：`modular_deepseek_v4.py:848`

```python
flat = self.input_norm(x.flatten(2).float())
mixes = F.linear(flat, self.hc_fn.float())
pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)
```

执行解释：

- `HyperHead` 只负责最终折叠。
- 输入 `[B, S, hc_mult, D]`。
- 输出 `[B, S, D]`。

## 7. RoPE：`DeepseekV4RotaryEmbedding` 和 `apply_rotary_pos_emb`

源码位置：

- `modular_deepseek_v4.py:79`
- `modular_deepseek_v4.py:116`
- `modular_deepseek_v4.py:46`

`DeepseekV4RotaryEmbedding` 负责生成 `cos` 和 `sin`。`apply_rotary_pos_emb` 负责把 `cos` 和 `sin` 应用到 Q、KV 或 attention output。

RoPE 生成核心源码：

```python
inv_freq = getattr(self, f"{layer_type}_inv_freq")
attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
position_ids_expanded = position_ids[:, None, :].float()
with maybe_autocast(device_type=device_type, enabled=False):
    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    cos = freqs.cos() * attention_scaling
    sin = freqs.sin() * attention_scaling
return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

执行解释：

- `layer_type` 只能取 `main` 或 `compress`。
- `main_inv_freq` 来自主 attention 的 RoPE 参数。
- `compress_inv_freq` 来自压缩分支的 RoPE 参数。
- `inv_freq @ position_ids` 得到每个位置的旋转角。
- 返回的 `cos` / `sin` 形状为 `[B, S, rope_dim / 2]`。

最小示例：

```python
position_ids = [[0, 1]]
inv_freq = [1.0]

freqs = [
    [0 * 1.0],
    [1 * 1.0],
]

cos = [[cos(0.0)], [cos(1.0)]]
sin = [[sin(0.0)], [sin(1.0)]]
```

核心源码：

```python
cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
rope_dim = cos.shape[-1]
nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
rotated = ((rope.float() * cos) + (rotate_half(rope).float() * sin)).to(x.dtype)
return torch.cat([nope, rotated], dim=-1)
```

执行解释：

- `cos` 和 `sin` 原本只覆盖 RoPE pair 数量，需要 `repeat_interleave(2)` 扩成完整 RoPE 维度。
- head 被拆成两段：前面的 `nope` 不旋转，最后的 `rope` 旋转。
- 返回时把 `nope` 和旋转后的 `rope` 拼回原维度。

最小示例：

```python
# head_dim = 4
# rope_dim = 2
head = [10.0, 20.0, 1.0, 2.0]

nope = [10.0, 20.0]  # 不参与 RoPE
rope = [1.0, 2.0]    # 参与 RoPE
```

注意：DeepSeekV4 的 attention 中 K 和 V 使用同一个 `kv`。源码会对 `kv` 应用 RoPE，因此 attention 输出后还会对输出的 RoPE 部分执行反向旋转。

## 8. Attention 主体：`DeepseekV4Attention.forward`

源码位置：`modular_deepseek_v4.py:636`

初始化关键源码：

```python
self.layer_type = config.layer_types[layer_idx]
self.rope_layer_type = "main" if self.layer_type == "sliding_attention" else "compress"
self.num_heads = config.num_attention_heads
self.num_key_value_groups = config.num_attention_heads
self.head_dim = config.head_dim

self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
self.q_a_norm = DeepseekV4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
self.q_b_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
self.compressor = (
    COMPRESSOR_CLASSES[self.layer_type](config) if self.layer_type != "sliding_attention" else None
)
```

执行解释：

- `sliding_attention` 使用 `main` RoPE。
- HCA / CSA 使用 `compress` RoPE。
- Q 有多个 head。
- KV 只有一个 head，之后广播给所有 Q head。
- 非滑窗层会创建 compressor。

forward 核心源码：

```python
cos, sin = position_embeddings[self.rope_layer_type]

q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
q = self.q_b_norm(q)
q = apply_rotary_pos_emb(q, cos, sin)

kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
kv = apply_rotary_pos_emb(kv, cos, sin)
```

执行解释：

- `q_a_proj` 先把 hidden 映射到 LoRA rank 空间。
- `q_a_norm` 做 RMSNorm。
- `q_b_proj` 生成所有 attention heads 的 Q。
- `kv_proj` 只生成一个 KV head。
- Q 和 KV 都应用同一套 RoPE。

最小形状示例：

```text
B = 1
S = 4
D = 2
num_heads = 2
head_dim = 3

q_b_proj output : [1, 4, 6]
q view          : [1, 4, 2, 3]
q transpose     : [1, 2, 4, 3]

kv_proj output  : [1, 4, 3]
kv view         : [1, 4, 1, 3]
kv transpose    : [1, 1, 4, 3]
```

cache 和 compressor：

```python
if past_key_values is not None:
    kv = past_key_values.update(kv, kv, self.layer_idx)[0]

block_bias = None
if self.compressor is not None:
    compressed_kv, block_bias = self.compressor(
        hidden_states, q_residual, position_ids, past_key_values, self.layer_idx
    )
    kv = torch.cat([kv, compressed_kv], dim=2)
```

执行解释：

- 滑窗 KV 先进入 `past_key_values.update(...)`。
- HCA / CSA 额外生成 `compressed_kv`。
- `compressed_kv` 拼到普通 KV 后面。

最小示例：

```text
sliding kv length     = 4
compressed kv length  = 2

before cat: kv.shape = [1, 1, 4, head_dim]
after cat : kv.shape = [1, 1, 6, head_dim]
```

mask 扩展：

```python
if isinstance(attention_mask, torch.Tensor) and kv.shape[2] > attention_mask.shape[-1]:
    if block_bias is not None:
        attention_mask = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)
    else:
        attention_mask = F.pad(attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]), value=0.0)
```

执行解释：

- 原始 `attention_mask` 只覆盖滑窗 KV。
- 拼接 compressed KV 后，mask 的 key 维度必须同步扩展。
- HCA / CSA 返回的 `block_bias` 决定 query 是否能访问 compressed entries。

attention 和输出投影：

```python
attn_output, attn_weights = attention_interface(
    self,
    q,
    kv,
    kv,
    attention_mask,
    dropout=0.0 if not self.training else self.attention_dropout,
    scaling=self.scaling,
    sliding_window=self.sliding_window,
    s_aux=self.sinks,
    **kwargs,
)

attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)

grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
grouped = self.o_a_proj(grouped).flatten(2)
output = self.o_b_proj(grouped)
return output, attn_weights
```

执行解释：

- `q, kv, kv` 表示 K 和 V 使用同一个张量。
- `s_aux=self.sinks` 是每个 head 的 attention sink 参数。
- 输出先用 `-sin` 做反向 RoPE。
- 输出再按 `o_groups` 分组投影，最后用 `o_b_proj` 回到 `hidden_size`。

### 8.1 分组输出投影：`DeepseekV4GroupedLinear`

源码位置：`modular_deepseek_v4.py:266`

该模块是 attention 输出投影的第一段，对应 `self.o_a_proj(grouped)`。

核心源码：

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    input_shape = x.shape[:-2]
    hidden_dim = x.shape[-1]
    w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
    x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
    y = torch.bmm(x, w).transpose(0, 1)
    return y.reshape(*input_shape, self.n_groups, -1)
```

执行解释：

- attention 输出先 reshape 成 `[..., o_groups, group_hidden]`。
- `self.weight` 被 reshape 成每组独立使用的权重。
- `torch.bmm` 对每个 group 执行批量矩阵乘。
- 返回形状仍保留 group 维度。
- 后续 `flatten(2)` 和 `o_b_proj` 把分组结果混回 `hidden_size`。

最小示例：

```python
# 输入有 2 个 group，每组 2 维。
x = [
    [[1.0, 2.0], [10.0, 20.0]]
]
# x.shape = [tokens=1, groups=2, group_hidden=2]

# group0 权重把 [a, b] 映射成 [a + b]
# group1 权重把 [a, b] 映射成 [0.1a + 0.1b]

group0_out = 1.0 + 2.0
# 3.0

group1_out = 0.1 * 10.0 + 0.1 * 20.0
# 3.0

o_a_proj_output = [[[3.0], [3.0]]]
# shape = [1, 2, 1]

flattened = [3.0, 3.0]
# 再交给 o_b_proj 输出 hidden_size。
```

## 9. HCA 压缩：`DeepseekV4HCACompressor.forward`

源码位置：`modular_deepseek_v4.py:330`

核心源码：

```python
kv = self.kv_proj(hidden_states)
gate = self.gate_proj(hidden_states)

if cache_layer is None:
    usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
    chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
else:
    chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("compressor", kv, gate)
```

执行解释：

- `kv_proj` 生成每个 token 的候选 compressed KV。
- `gate_proj` 生成窗口内加权求和的权重。
- `usable` 只保留完整窗口。
- 不完整窗口在 cache 中等待后续 token。

窗口压缩：

```python
if chunk_kv.shape[1] > 0:
    n_windows = chunk_kv.shape[1] // self.compress_rate
    chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, -1)
    chunk_gate = chunk_gate.view(batch, n_windows, self.compress_rate, -1) + self.position_bias.to(
        chunk_gate.dtype
    )
    compressed = self.kv_norm(
        (chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)).sum(dim=2)
    )
```

执行解释：

- 每 `compress_rate` 个 token 组成一个窗口。
- `softmax(dim=2)` 在窗口内部计算权重。
- 加权求和后得到一个 compressed entry。
- 每个窗口输出一个 compressed KV。

最小示例：

```python
compress_rate = 2
token_kv = [10.0, 20.0, 30.0, 40.0]

window0 = [10.0, 20.0]  # 生成 entry0
window1 = [30.0, 40.0]  # 生成 entry1

# 假设 window0 的 gate softmax = [0.25, 0.75]
entry0 = 0.25 * 10.0 + 0.75 * 20.0
# entry0 = 17.5
```

位置编码和 block bias：

```python
positions = torch.arange(n_windows, device=compressed.device)
positions = (positions * self.compress_rate + first_window_position).unsqueeze(0).expand(batch, -1)
cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)

entry_indices = torch.arange(compressed_len, device=compressed_kv.device)
causal_threshold = (position_ids + 1) // self.compress_rate
block_bias = compressed_kv.new_zeros((batch, 1, seq_len, compressed_len))
block_bias = block_bias.masked_fill(
    entry_indices.view(1, 1, 1, -1) >= causal_threshold.unsqueeze(1).unsqueeze(-1),
    float("-inf"),
)
```

执行解释：

- 每个 compressed entry 使用窗口起点位置做 RoPE。
- `causal_threshold` 控制 query 能看到哪些 compressed entries。
- 不允许访问的 compressed entry 在 `block_bias` 中写成 `-inf`。

最小因果示例：

```text
compress_rate = 2
compressed entries:
entry0 = token0, token1 的压缩结果
entry1 = token2, token3 的压缩结果

query position 0: threshold = (0 + 1) // 2 = 0，可见 entry 数量 0
query position 1: threshold = (1 + 1) // 2 = 1，可见 entry0
query position 2: threshold = (2 + 1) // 2 = 1，可见 entry0
query position 3: threshold = (3 + 1) // 2 = 2，可见 entry0 和 entry1
```

## 10. 压缩 cache：`DeepseekV4HCACache` 和 `DeepseekV4CSACache`

源码位置：`modular_deepseek_v4.py:181`

滑窗 KV cache 核心源码：

```python
def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
    if not self.is_initialized:
        self.lazy_initialization(key_states, value_states)
        self.values = self.keys
    self.cumulative_length += key_states.shape[-2]
    full = torch.cat([self.keys, key_states], dim=-2)
    self.keys = full[:, :, -self.sliding_window + 1 :, :]
    self.values = self.keys
    return full, full
```

执行解释：

- DeepSeekV4 使用 shared KV，K 和 V 指向同一份数据。
- `full` 是历史 KV 与当前 KV 的拼接结果。
- `self.keys` 只保留滑窗范围内的 KV。
- attention forward 使用返回的 `full`，同时 cache 内部保留下一次调用需要的滑窗尾部。

最小示例：

```text
sliding_window = 4

cache 中已有 keys:
[token0, token1, token2]

本次新增:
[token3, token4]

full:
[token0, token1, token2, token3, token4]

self.keys 保留最后 sliding_window - 1 个:
[token2, token3, token4]
```

HCA cache 核心源码：

```python
first_window_position = self.entry_count[name] * self.compress_rate
buffered_kv, buffered_gate = self.buffer_kv[name], self.buffer_gate[name]
if buffered_kv is not None and buffered_kv.shape[1]:
    kv = torch.cat([buffered_kv, kv], dim=1)
    gate = torch.cat([buffered_gate, gate], dim=1)

usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
self.buffer_kv[name], self.buffer_gate[name] = kv[:, usable:], gate[:, usable:]
return kv[:, :usable], gate[:, :usable], first_window_position
```

执行解释：

- cache 保存不足一个压缩窗口的 token。
- 新 token 到来后与旧 buffer 拼接。
- `usable` 是能形成完整窗口的 token 数。
- 剩余 token 继续留在 buffer 中。

最小示例：

```text
compress_rate = 4

第 1 次 decode: 1 个 token -> usable = 0，buffer 保留 1 个 token
第 2 次 decode: 1 个 token -> usable = 0，buffer 保留 2 个 token
第 3 次 decode: 1 个 token -> usable = 0，buffer 保留 3 个 token
第 4 次 decode: 1 个 token -> usable = 4，输出 1 个完整窗口，buffer 清空
```

追加 compressed entries：

```python
if self.compressed_kv[name] is None:
    self.compressed_kv[name] = compressed
elif compressed.shape[1] > 0:
    self.compressed_kv[name] = torch.cat([self.compressed_kv[name], compressed], dim=1)
self.entry_count[name] += compressed.shape[1]
return self.compressed_kv[name]
```

执行解释：

- 新生成的 compressed entries 会追加到历史 entries 后面。
- attention 看到的是当前层已经生成的完整 compressed KV 历史。

CSA 额外保存 overlap state：

```python
prior_kv, prior_gate = self.overlap_kv[name], self.overlap_gate[name]
self.overlap_kv[name] = chunk_kv[:, -1, :, :head_dim].clone()
self.overlap_gate[name] = chunk_gate[:, -1, :, :head_dim].clone()
return prior_kv, prior_gate
```

执行解释：

- CSA 每个 token 生成两段：`Ca` 和 `Cb`。
- 当前窗口的 `Ca` 会给下一个窗口使用。
- cache 保存最后一个完整窗口的 `Ca`，供下一次 forward 继续压缩。

## 11. CSA 压缩：`DeepseekV4CSACompressor.forward`

源码位置：`modular_deepseek_v4.py:549`

初始化关键源码：

```python
self.compress_rate = config.compress_rates["compressed_sparse_attention"]
self.head_dim = config.head_dim
self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim))
self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
self.rotary_emb = DeepseekV4RotaryEmbedding(config)
self.indexer = DeepseekV4Indexer(config)
```

执行解释：

- CSA 的 `kv_proj` 和 `gate_proj` 输出 `2 * head_dim`。
- 前半段是 `Ca`，后半段是 `Cb`。
- `indexer` 负责为每个 query 选择 compressed entries。

forward 中的 Ca / Cb 布局：

```python
n_windows = chunk_kv.shape[1] // self.compress_rate
ratio = self.compress_rate
chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias.to(chunk_gate.dtype)

new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim:]
new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim:]
if n_windows > 1:
    new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
    new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
```

执行解释：

- `new_kv` 的后半段放当前窗口的 `Cb`。
- `new_kv` 的前半段放上一个窗口的 `Ca`。
- 第一个窗口如果没有历史 `Ca`，前半段 gate 是 `-inf`，softmax 后权重为 0。

最小示例：

```text
compress_rate = 2

窗口 0:
token0: Ca0, Cb0
token1: Ca1, Cb1

窗口 1:
token2: Ca2, Cb2
token3: Ca3, Cb3

CSA entry0 使用:
Cb0, Cb1

CSA entry1 使用:
Ca0, Ca1, Cb2, Cb3
```

压缩和 RoPE：

```python
compressed = self.kv_norm(
    (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2)
)
positions = torch.arange(n_windows, device=compressed.device)
positions = positions * self.compress_rate + first_window_position
positions = positions.unsqueeze(0).expand(batch, -1)
cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
```

执行解释：

- CSA 的压缩仍然是窗口内加权求和。
- 与 HCA 不同，CSA 的窗口宽度是 `2 * compress_rate`，并且有跨窗口重叠。

Indexer 选择和 block bias：

```python
top_k_indices = self.indexer(hidden_states, q_residual, position_ids, past_key_values, layer_idx)
compressed_len = compressed_kv.shape[2]
valid = top_k_indices >= 0
safe_indices = torch.where(valid, top_k_indices, torch.full_like(top_k_indices, compressed_len))
block_bias = compressed_kv.new_full((batch, 1, seq_len, compressed_len + 1), float("-inf"))
block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
return compressed_kv, block_bias[..., :compressed_len]
```

执行解释：

- `indexer` 返回每个 query 允许访问的 compressed entry 下标。
- `-1` 表示该位置无效。
- `scatter_` 把被选中的位置写成 `0.0`，未选中的位置保持 `-inf`。
- attention 拼接 mask 后，只能看被 indexer 选中的 compressed entries。

## 12. CSA Indexer：`DeepseekV4Indexer.forward`

源码位置：`modular_deepseek_v4.py:431`

压缩 indexer keys：

```python
kv = self.kv_proj(hidden_states)
gate = self.gate_proj(hidden_states)

if cache_layer is None:
    usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
    chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
else:
    chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("indexer", kv, gate)
```

执行解释：

- Indexer 有自己的 `kv_proj` 和 `gate_proj`。
- 它的压缩逻辑与 CSA compressor 对齐，但使用 `index_head_dim`。
- cache 中 `"indexer"` 和 `"compressor"` 是两套独立状态。

生成 query 并打分：

```python
cos_q, sin_q = self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=self.rope_layer_type)
q = self.q_b_proj(q_residual).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
q = apply_rotary_pos_emb(q, cos_q, sin_q).transpose(1, 2)

scores = torch.matmul(q.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))
scores = F.relu(scores) * self.softmax_scale
weights = self.weights_proj(hidden_states).float() * self.weights_scaling
index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)
```

执行解释：

- query 来自 attention 中间结果 `q_residual`。
- `scores` 是 query 与 indexer compressed keys 的相似度。
- `ReLU` 会过滤负相关分数。
- `weights_proj` 给不同 indexer head 分配权重。
- 对 head 维度求和后得到 `[B, S, compressed_len]`。

最小示例：

```python
# 一个 query 对两个 compressed entries 的分数：
scores = [2.0, 5.0]
weights = [1.0]

index_scores = [2.0, 5.0]
top1 = [1]  # 选择 entry1
```

因果约束和 TopK：

```python
compressed_len = compressed_kv.shape[1]
top_k = min(self.index_topk, compressed_len)

if compressed_len > 0:
    causal_threshold = (position_ids + 1) // self.compress_rate
    entry_indices = torch.arange(compressed_len, device=index_scores.device)
    future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)
    index_scores = index_scores.masked_fill(future_mask, float("-inf"))
    top_k_indices = index_scores.topk(top_k, dim=-1).indices
    invalid = top_k_indices >= causal_threshold.unsqueeze(-1)
    return torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)
```

执行解释：

- Indexer 不能选择未来 compressed entries。
- `future_mask` 把未来 entry 的分数设为 `-inf`。
- `topk` 后仍然不合法的位置会替换为 `-1`。
- CSA compressor 接收到 `-1` 后不会允许对应位置进入 attention。

## 13. MoE：`DeepseekV4SparseMoeBlock`

源码位置：`modular_deepseek_v4.py:938`

MoE block 核心源码：

```python
self.is_hash = config.mlp_layer_types[layer_idx] == "hash_moe"
self.gate = DeepseekV4HashRouter(config) if self.is_hash else DeepseekV4TopKRouter(config)
self.experts = DeepseekV4Experts(config)
self.shared_experts = DeepseekV4MLP(config)

def forward(self, hidden_states, input_ids=None):
    batch, seq_len, hidden_dim = hidden_states.shape
    residual = hidden_states
    flat = hidden_states.view(-1, hidden_dim)
    if self.is_hash:
        _, weights, indices = self.gate(hidden_states, input_ids)
    else:
        _, weights, indices = self.gate(hidden_states)
    routed = self.experts(flat, indices, weights).view(batch, seq_len, hidden_dim)
    return routed + self.shared_experts(residual)
```

执行解释：

- `hash_moe` 使用 `DeepseekV4HashRouter`。
- `moe` 使用 `DeepseekV4TopKRouter`。
- routed experts 只处理被路由到的 token。
- shared expert 对所有 token 都执行。
- 最终输出是 `routed + shared_expert`。

Shared expert 使用 `DeepseekV4MLP`：

```python
class DeepseekV4MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
```

执行解释：

- `DeepseekV4MLP` 是共享 MLP，所有 token 都会经过。
- `gate_proj` 和 `up_proj` 共同组成 SwiGLU。
- `down_proj` 把中间维度投回 `hidden_size`。

最小示例：

```python
# 用标量模拟一维 MLP。
x = 2.0
gate_proj_x = 1.5
up_proj_x = 3.0

# 假设 silu(1.5) = 1.226
hidden = 1.226 * 3.0
# hidden = 3.678

# 假设 down_proj 是乘 0.5。
shared_output = 3.678 * 0.5
# shared_output = 1.839
```

TopK router 源码：

```python
flat = hidden_states.reshape(-1, self.hidden_dim)
logits = F.linear(flat, self.weight)
scores = self.score_fn(logits)
indices = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices
weights = scores.gather(1, indices)
weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
return logits, weights * self.routed_scaling_factor, indices
```

执行解释：

- `logits` 是每个 token 对每个 expert 的原始分数。
- `score_fn` 把 logits 转成可比较的 scores。
- `topk` 选择分数最高的专家。
- `gather` 取出被选专家的权重。
- 权重归一化后乘 `routed_scaling_factor`。

TopK router 最小示例：

```python
# 一个 token，3 个 expert，top_k = 2。
scores = [0.2, 0.9, 0.4]

indices = topk(scores, k=2)
# indices = [1, 2]

selected_scores = [0.9, 0.4]
weights = [
    0.9 / (0.9 + 0.4),
    0.4 / (0.9 + 0.4),
]
# weights = [0.6923, 0.3077]
```

Hash router 源码：

```python
flat = hidden_states.reshape(-1, self.hidden_dim)
logits = F.linear(flat, self.weight)
scores = self.score_fn(logits)
indices = self.tid2eid[input_ids.reshape(-1)].long()
weights = scores.gather(1, indices)
weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
return logits, weights * self.routed_scaling_factor, indices
```

执行解释：

- Hash router 不用 `topk` 选专家。
- 它通过 `tid2eid[input_ids]` 直接查出专家编号。
- checkpoint 会提供真实的 `tid2eid`，初始化时只是全 0。

Hash router 最小示例：

```python
input_ids = [7]
tid2eid = {
    7: [3, 12],
}

indices = tid2eid[7]
# indices = [3, 12]

# router 仍然会计算 logits 和 scores，
# 但专家编号来自 tid2eid，而不是 topk(scores)。
scores_for_selected = [0.6, 0.2]
weights = [
    0.6 / (0.6 + 0.2),
    0.2 / (0.6 + 0.2),
]
# weights = [0.75, 0.25]
```

Expert 执行源码：

```python
final = torch.zeros_like(hidden_states)
mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()

for expert_idx in hit:
    expert_idx = expert_idx[0]
    top_k_pos, token_idx = torch.where(mask[expert_idx])
    current = self._apply_gate(F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]))
    current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]
    final.index_add_(0, token_idx, current.to(final.dtype))
return final
```

执行解释：

- `one_hot` 生成 expert 到 token 的分配矩阵。
- `hit` 只保留实际收到 token 的专家。
- 每个专家只计算分配给自己的 token。
- `index_add_` 把专家输出加回原 token 位置。

Experts 分发最小示例：

```python
# 2 个 token，每个 token 选择 2 个 expert。
top_k_index = [
    [0, 2],  # token0
    [1, 2],  # token1
]

# expert0 只处理 token0。
# expert1 只处理 token1。
# expert2 同时处理 token0 和 token1。

expert2_token_indices = [0, 1]
expert2_outputs = [
    [4.0, 3.0],
    [2.0, 5.0],
]

# index_add_ 会把 expert2_outputs 加回 final[0] 和 final[1]。
```

Expert 内部激活：

```python
gate, up = gate_up.chunk(2, dim=-1)
gate = gate.clamp(max=self.limit)
up = up.clamp(min=-self.limit, max=self.limit)
return self.act_fn(gate) * up
```

执行解释：

- 专家内部使用 SwiGLU 结构。
- `gate` 和 `up` 都做 clamp，限制数值范围。
- 输出是 `silu(gate) * up`。

最小 MoE 示例：

```python
# token0 选择 expert0 和 expert2
indices = [0, 2]
weights = [0.7, 0.3]

expert0_output = [10.0, 1.0]
expert2_output = [4.0, 3.0]
shared_output = [0.5, 0.5]

routed = 0.7 * expert0_output + 0.3 * expert2_output
# routed = [8.2, 1.6]

final = routed + shared_output
# final = [8.7, 2.1]
```

### 13.1 Router 辅助损失：`load_balancing_loss_func`

源码位置：`modeling_deepseek_v4.py:1313`

该函数只在 `output_router_logits=True` 时使用。它用于约束 token 不要长期集中到少数专家。

核心源码：

```python
if gate_logits is None or not isinstance(gate_logits, tuple):
    return 0

concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)
routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
_, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
router_prob_per_expert = torch.mean(routing_weights, dim=0)
overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
return overall_loss * num_experts
```

执行解释：

- `gate_logits` 是多个 MoE 层的 router logits。
- `softmax` 得到每个 token 分配到各 expert 的概率。
- `topk` 得到实际选择的 expert。
- `tokens_per_expert` 统计每个 expert 实际收到多少 token。
- `router_prob_per_expert` 统计 router 对每个 expert 的平均概率。
- 两者相乘后求和，得到负载均衡辅助损失。

最小示例：

```python
# 2 个 token，2 个 expert，top_k = 1。
routing_weights = [
    [0.9, 0.1],
    [0.8, 0.2],
]

selected_experts = [0, 0]

tokens_per_expert = [1.0, 0.0]
router_prob_per_expert = [
    (0.9 + 0.8) / 2,
    (0.1 + 0.2) / 2,
]
# router_prob_per_expert = [0.85, 0.15]

overall_loss = 1.0 * 0.85 + 0.0 * 0.15
# overall_loss = 0.85

aux_loss = overall_loss * num_experts
# aux_loss = 1.7
```

该例中两个 token 都选择 expert0，辅助损失会反映专家负载不均衡。

## 14. 完整执行示例对照

使用一个 3 层配置描述源码执行：

```python
layer_types = [
    "sliding_attention",
    "heavily_compressed_attention",
    "compressed_sparse_attention",
]
mlp_layer_types = [
    "moe",
    "moe",
    "moe",
]
hc_mult = 2
```

执行过程：

```text
input_ids
  -> DeepseekV4Model.forward
     -> embed_tokens
     -> hidden_states = inputs_embeds.unsqueeze(2).expand(...)
     -> position_embeddings["main"]
     -> position_embeddings["compress"]

Layer 0: sliding_attention
  -> attn_hc: [B,S,2,D] -> [B,S,D]
  -> Attention: only sliding KV
  -> ffn_hc: [B,S,2,D] -> [B,S,D]
  -> MoE

Layer 1: heavily_compressed_attention
  -> attn_hc
  -> Attention: sliding KV + HCA compressed KV
  -> HCA block_bias
  -> ffn_hc
  -> MoE

Layer 2: compressed_sparse_attention
  -> attn_hc
  -> Attention: sliding KV + CSA compressed KV
  -> Indexer selects compressed entries
  -> CSA block_bias
  -> ffn_hc
  -> MoE

DeepseekV4HyperHead
  -> RMSNorm
  -> DeepseekV4ForCausalLM.lm_head
  -> logits
```

运行对应模拟：

```bash
cd /home/mtuser/workspace
python insights/deepseek_v4_source_walkthrough/run_deepseek_v4_tiny_trace.py
```

该脚本用固定小数值模拟以下源码行为：

- mHC 的折叠和回写。
- 滑窗 attention 的局部可见范围。
- HCA 的窗口压缩和因果 `block_bias`。
- CSA 的 Ca / Cb 重叠压缩。
- Indexer 的 top-k compressed entry 选择。
- MoE router、routed experts 和 shared expert。
- `lm_head` 输出 logits。

## 15. 其他源码项

以下源码项不直接处理 hidden states，但会影响配置校验、模块选择或框架集成。

### 15.1 `validate_layer_type`

源码位置：`configuration_deepseek_v4.py:223`

作用：

- 检查 `layer_types` 长度是否等于 `num_hidden_layers`。
- 检查 `layer_types` 是否只包含 DeepSeekV4 支持的 attention 类型。
- 检查 `mlp_layer_types` 是否只包含 `hash_moe` 或 `moe`。

最小示例：

```python
num_hidden_layers = 2
layer_types = ["heavily_compressed_attention"]

# len(layer_types) != num_hidden_layers
# 配置初始化时应抛出 ValueError。
```

### 15.2 `validate_rope`

源码位置：`configuration_deepseek_v4.py:198`

作用：

- DeepSeekV4 的 RoPE 参数按 `main` 和 `compress` 两组保存。
- 该函数逐组调用对应的 RoPE 校验逻辑。
- 它不生成 `cos` / `sin`，只检查配置是否合法。

最小示例：

```python
rope_parameters = {
    "main": {"rope_type": "default", "rope_theta": 10000.0},
    "compress": {"rope_type": "default", "rope_theta": 160000.0},
}

# validate_rope 会分别检查 main 和 compress。
```

### 15.3 `COMPRESSOR_CLASSES`

源码位置：`modular_deepseek_v4.py:629`

作用：

- 根据 `layer_type` 选择 compressor。
- `sliding_attention` 不创建 compressor。
- HCA 层创建 `DeepseekV4HCACompressor`。
- CSA 层创建 `DeepseekV4CSACompressor`。

最小示例：

```python
COMPRESSOR_CLASSES = {
    "sliding_attention": None,
    "compressed_sparse_attention": DeepseekV4CSACompressor,
    "heavily_compressed_attention": DeepseekV4HCACompressor,
}

layer_type = "compressed_sparse_attention"
compressor_cls = COMPRESSOR_CLASSES[layer_type]
# compressor_cls = DeepseekV4CSACompressor
```

### 15.4 并行计划和导出列表

源码位置：

- `configuration_deepseek_v4.py:101`
- `configuration_deepseek_v4.py:109`
- `configuration_deepseek_v4.py:114`
- `modeling_deepseek_v4.py:1500`

作用：

- `attribute_map` 把通用字段名映射到 DeepSeekV4 的字段名。
- `base_model_pp_plan`、`base_model_ep_plan`、`base_model_fsdp_plan` 是框架并行和切分计划。
- `__all__` 控制模块导出名称。

最小示例：

```python
attribute_map = {
    "num_local_experts": "n_routed_experts",
    "intermediate_size": "moe_intermediate_size",
}

# 外部代码读取 config.num_local_experts 时，
# 实际对应 DeepSeekV4 配置中的 n_routed_experts。
```

这些项不改变单次 forward 的张量形状，因此不放入主执行链路。

## 16. 源码阅读顺序

建议按下面顺序阅读：

1. `configuration_deepseek_v4.py:244`，确认配置如何生成 `layer_types`、`mlp_layer_types`、`rope_parameters`。
2. `modeling_deepseek_v4.py:1417`，确认 CausalLM 如何调用模型主体和 `lm_head`。
3. `modular_deepseek_v4.py:1112`，确认 embedding、mask、mHC 扩展、RoPE 和层循环。
4. `modular_deepseek_v4.py:982`，确认单层 decoder 的 attention 和 MoE 回写公式。
5. `modular_deepseek_v4.py:805`，确认 mHC 的 `pre`、`post`、`comb`。
6. `modular_deepseek_v4.py:682`，确认 Q、KV、compressor、mask 和 grouped output projection。
7. `modular_deepseek_v4.py:330`，确认 HCA 压缩。
8. `modular_deepseek_v4.py:549`，确认 CSA 压缩。
9. `modular_deepseek_v4.py:431`，确认 Indexer 选择 compressed entries。
10. `modular_deepseek_v4.py:938`，确认 MoE 路由和专家执行。

## 17. 关键结论

DeepSeekV4 的源码主线可以拆成四组机制：

| 机制 | 源码模块 | 作用 |
| --- | --- | --- |
| 配置和初始化 | `DeepseekV4Config`、`DeepseekV4PreTrainedModel` | 生成层类型、MoE 类型、RoPE 参数，并设置初始化和框架能力 |
| 归一化和 RoPE | `DeepseekV4RMSNorm`、`DeepseekV4UnweightedRMSNorm`、`DeepseekV4RotaryEmbedding`、`apply_rotary_pos_emb` | 控制数值尺度和位置信息注入 |
| mHC | `DeepseekV4HyperConnection`、`DeepseekV4HyperHead` | 在多条残差流之间学习折叠、回写和混合 |
| 混合 attention | `DeepseekV4Attention`、`DeepseekV4GroupedLinear` | 滑窗 KV 与可选 compressed KV 拼接后执行 attention，并使用分组输出投影 |
| 长程压缩 | `DeepseekV4HCACache`、`DeepseekV4CSACache`、`DeepseekV4HCACompressor`、`DeepseekV4CSACompressor`、`DeepseekV4Indexer` | 用 compressed KV 表示长上下文，并控制 query 的访问范围 |
| MoE | `DeepseekV4SparseMoeBlock`、`DeepseekV4TopKRouter`、`DeepseekV4HashRouter`、`DeepseekV4Experts`、`DeepseekV4MLP`、`load_balancing_loss_func` | 每个 token 进入少量 routed experts，同时保留 shared expert 和 router 辅助损失 |

完整输出路径：

```text
token id
  -> embedding
  -> mHC streams
  -> decoder layers
  -> HyperHead
  -> RMSNorm
  -> lm_head
  -> logits
```
