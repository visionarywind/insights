# 2. LLM 常见模块源码分析：以 Llama 为例

一个 decoder-only LLM 由结构相同的 layer 堆叠而成。本章以 HuggingFace Transformers 中 Llama 的完整源码为例，提取核心 forward 代码并按执行链路逐模块拆解。

参考源码：

```text
transformers/src/transformers/models/llama/modeling_llama.py
```

## 2.1 RMSNorm

Llama 使用 RMSNorm 替代 LayerNorm，去掉均值的减法，只做 RMS 缩放，计算量更小。

源码位置 `modeling_llama.py:62-67`：

```python
def forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)                    # 升精度，减小归约误差
    variance = hidden_states.pow(2).mean(-1, keepdim=True)            # 计算均方
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)  # 归一化
    return self.weight * hidden_states.to(input_dtype)                # 乘可学习权重，cast 回原精度
```

执行链路：`float32 → pow(2) → mean → rsqrt → mul → to(input_dtype) * weight`。

| 步骤 | OP | 说明 |
|------|-----|------|
| 升精度 | `.to(float32)` | 避免半精度归约的数值误差 |
| 均方 | `pow(2)` + `mean(-1, keepdim=True)` | `keepdim=True` 保持 dim 数，便于 broadcast |
| 归一化 | `rsqrt(variance + eps)` + `mul` | `1/sqrt(x)` |
| 缩放+降精度 | `* weight` + `.to(input_dtype)` | 可学习 affine 参数 |

reference 实现含 4 个独立的逐元素/归约 kernel（pow → mean → rsqrt → mul），在线热点路径通常使用 fused RMSNorm kernel 一次完成。

## 2.2 RoPE（旋转位置编码）

RoPE 通过旋转矩阵将位置信息注入 Q 和 K。Llama 在模型顶层生成 cos/sin，传给每一层 attention。

源码位置 `modeling_llama.py:124-135`（`LlamaRotaryEmbedding.forward`）：

```python
def forward(self, x, position_ids):
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(...).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()
    # freqs: [B, seq_len, head_dim/2]
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)          # 复制一份，凑满 head_dim
    cos = emb.cos() * self.attention_scaling
    sin = emb.sin() * self.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

应用到 Q/K 源码位置 `modeling_llama.py:146-168`：

```python
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)               # 补 head 维: [B,1,S,Dh]
    sin = sin.unsqueeze(unsqueeze_dim)
    # 复数乘法：将相邻两维视为复数的实部和虚部
    q_embed = (q * cos) + (rotate_half(q) * sin)     # rotate_half: (x1,x2) → (-x2,x1)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

涉及的 OP：频率生成 `matmul` + `transpose` + `cat` + `cos` + `sin`；旋转 `unsqueeze` + `mul` + `add`。cos/sin 在 float32 下计算后 cast 回模型 dtype。

## 2.3 Attention

`LlamaAttention` 是标准的 Multi-Head Attention，支持 GQA（Query 头数 > KV 头数）。

源码位置 `modeling_llama.py:251-289`，核心 forward：

```python
def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs):
    input_shape = hidden_states.shape[:-1]                         # [B, S]
    hidden_shape = (*input_shape, -1, self.head_dim)               # [B, S, H, Dh]

    # 1. Q/K/V projection + head layout
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states   = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # 2. RoPE 旋转 Q 和 K
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # 3. KV cache 更新（decode 时追加当前 K/V）
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    # 4. attention 计算（eager / SDPA / FlashAttention）
    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling, **kwargs,
    )

    # 5. 输出恢复 layout + 投影
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights
```

shape 转换流程（以 B=1, S=3, H=8, Dh=64 为例）：

```text
hidden [1, 3, 512]
  → q_proj     [1, 3, 512]        F.linear: D → H*Dh
  → view       [1, 3, 8, 64]      metadata only
  → transpose  [1, 8, 3, 64]      metadata only，输出非连续
  → RoPE       [1, 8, 3, 64]      unsqueeze + mul + add
  → attention  [1, 8, 3, 64]      matmul + softmax + matmul
  → reshape    [1, 3, 512]        metadata
  → contiguous                     D2D copy（transpose 后非连续）
  → o_proj     [1, 3, 512]        F.linear
```

涉及的 OP：

| 阶段 | OP | shape 变化 | 行为 |
|------|-----|-----------|------|
| Q/K/V 投影 | `F.linear`(×3) | `[B,S,D]` → `[B,S,H*Dh]` | GEMM |
| head layout | `view` + `transpose` | `[B,S,H,Dh]` → `[B,H,S,Dh]` | 零拷贝 |
| RoPE | `unsqueeze` + `mul` + `add` | — | elementwise |
| KV cache | cache.update | 追加 | copy |
| attention | matmul / SDPA | `[B,H,S,Dh]` → `[B,H,S,Dh]` | GEMM + softmax |
| output layout | `reshape` + `contiguous` | `[B,H,S,Dh]` → `[B,S,D]` | contiguous 可能 D2D copy |
| output proj | `F.linear` | `[B,S,D]` → `[B,S,D]` | GEMM |

### GQA：repeat_kv

当 KV head 数少于 Q head 数时，需要将 KV 扩展到 Q 的头数。源码位置 `modeling_llama.py:187-196`：

```python
def repeat_kv(hidden_states, n_rep):
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
```

`expand` + `reshape` 组合产生 zero-stride view，不复制内存。

### 参考实现：eager_attention_forward

源码位置 `modeling_llama.py:199-221`：

```python
def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)         # GQA 扩展
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling   # Q @ K^T
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask                # causal mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)          # softmax(QK^T) @ V
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
```

关键点：softmax 在 float32 下计算后 cast 回原始 dtype，提升数值稳定性。eager 路径会 materialize 完整的 `[B,H,S,S]` score tensor，S 大时显存和带宽开销显著。实际推理使用 SDPA 或 FlashAttention 避免。

## 2.4 MLP（SwiGLU）

Llama MLP 使用 SwiGLU 激活：gate 分支过 SiLU 后与 up 分支逐元素乘，再 down 投影回 hidden size。

源码位置 `modeling_llama.py:171-184`：

```python
class LlamaMLP(nn.Module):
    def __init__(self, config):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=config.mlp_bias)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]   # SiLU

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
```

一行 forward 完成三个线性投影 + 一个激活：

```text
x [B, S, D]
  → gate_proj: D → I          gate 分支
  → up_proj:   D → I          up 分支
  → SiLU(gate) * up           SwiGLU: 逐元素乘
  → down_proj: I → D          降维
```

涉及的 OP：`F.linear`(×3) + `silu` + `mul`。production 优化通常将 gate_proj 和 up_proj 合并为一次 `F.linear` 后 `chunk` 拆分，减少 GEMM launch 数。

## 2.5 Decoder Layer

`LlamaDecoderLayer` 是 pre-norm block：子层前做 RMSNorm，再进入 attention/MLP，最后残差 add。

源码位置 `modeling_llama.py:303-332`：

```python
def forward(self, hidden_states, attention_mask, position_ids, past_key_values,
            use_cache, position_embeddings, **kwargs):
    # ── Attention 子层 ──
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)            # pre-norm
    hidden_states, _ = self.self_attn(
        hidden_states, attention_mask=attention_mask,
        position_embeddings=position_embeddings,
        past_key_values=past_key_values, **kwargs,
    )
    hidden_states = residual + hidden_states                       # residual add

    # ── MLP 子层 ──
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)   # pre-norm
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states                       # residual add
    return hidden_states
```

执行链路：`RMSNorm → Attention → residual → RMSNorm → MLP → residual`。

prefill 和 decode 的热点不同：

| 阶段 | Q | K/V | 性能重点 |
|------|---|---|----------|
| prefill | 多个 token 位置 | 多个 token 位置 | GEMM 带宽、attention 计算 |
| decode | 1 个 token | 全部历史 | kernel launch 数、KV cache 访问 |

decode 时 KV cache 复用历史 K/V，只计算新 token 的 K/V 并追加到 cache。

## 2.6 完整模型

### LlamaModel.forward

源码位置 `modeling_llama.py:375-425`：

```python
def forward(self, input_ids, attention_mask, position_ids, past_key_values,
            inputs_embeds, use_cache, **kwargs):
    # token id → embedding
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)             # [B, S] → [B, S, D]

    # 生成 position_ids
    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], ...) + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)

    # causal mask
    causal_mask = create_causal_mask(...)

    # RoPE cos/sin 在模型层统一生成，传入每一层
    position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

    # N 层 decoder layer
    for decoder_layer in self.layers:
        hidden_states = decoder_layer(hidden_states, attention_mask=causal_mask,
                                       position_embeddings=position_embeddings, ...)

    # final RMSNorm
    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)
```

### LlamaForCausalLM.forward

源码位置 `modeling_llama.py:445-499`：

```python
def forward(self, input_ids, attention_mask, ..., labels, logits_to_keep=0, **kwargs):
    outputs = self.model(...)                                   # LlamaModel
    hidden_states = outputs.last_hidden_state                   # [B, S, D]

    # decode 通常只需要最后 token 的 logits
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])  # [B, S, D] → [B, S, V]
    return CausalLMOutputWithPast(logits=logits, ...)
```

`logits_to_keep` 优化：decode 时只需要最后一个 token 的 logits，`logits_to_keep=1` 只对最后一个位置做 lm_head projection，减少 vocab 维度（V 通常 32k-128k）的 GEMM 计算量。

## 2.7 完整执行链路

以 prefill（S=3）单层为例：

```text
input_ids [1, 3]
  → embed_tokens           [1, 3, 4096]      F.embedding
  → position_ids           [1, 3]            arange + unsqueeze
  → causal_mask            [1,1,3,3]         full + triu
  → rotary_emb.cos/sin     [1,3,64]          matmul + cos + sin + cat
  ┌─ layer 0 ─────────────────────────────────────────────────
  │ input_layernorm        [1, 3, 4096]      to(float32) + pow + mean + rsqrt + mul
  │ q/k/v_proj             [1,8,3,64]        F.linear(×3) + view + transpose
  │ RoPE                   [1,8,3,64]        unsqueeze + mul + add
  │ attention              [1,8,3,64]        matmul + softmax + matmul (或 FlashAttention)
  │ o_proj                 [1, 3, 4096]      reshape + contiguous + F.linear
  │ residual add           [1, 3, 4096]      +
  │ post_attn_norm         [1, 3, 4096]      RMSNorm
  │ gate/up/down           [1, 3, 4096]      F.linear(×3) + silu + mul
  │ residual add           [1, 3, 4096]      +
  └───────────────────────────────────────────────────────────
  ... × 32 layers
  → final_norm             [1, 3, 4096]      RMSNorm
  → lm_head                [1, 3, 128256]    F.linear (decode 时只取最后 token)
```

## 2.8 总结：Llama 的 OP 全景

| 模块 | 核心 OP | 类型 |
|------|---------|------|
| Embedding | `F.embedding` | 查表 |
| Position | `arange` + `unsqueeze` | metadata |
| Causal Mask | `full` + `triu` | metadata |
| RoPE | `unsqueeze` + `mul` + `add` | elementwise |
| RMSNorm | `float()` + `pow` + `mean` + `rsqrt` + `mul` + `to(dtype)` | reduction + elementwise |
| Q/K/V Proj | `F.linear`(×3) | GEMM |
| Head Layout | `view` + `transpose` | metadata |
| Attention | `matmul`(×2) + `softmax` | GEMM + reduction |
| Output Layout | `reshape` + `contiguous` + `F.linear` | copy + GEMM |
| SwiGLU | `F.linear`(×3) + `silu` + `mul` | GEMM + elementwise |
| Residual | `+` | elementwise |
| LM Head | `F.linear` | GEMM |

源码分析的通用方法：先识别模块的输入输出 shape，再标注每个 OP 的类型（layout / GEMM / elementwise / reduction），最后对照 profiler 确认底层执行路径。prefill 的性能重点在 GEMM 和 attention 带宽，decode 的性能重点在 kernel launch 数、KV cache 访问和同步边界。
