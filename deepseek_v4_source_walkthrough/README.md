# DeepSeek-V4 入门源码解读

如果这一版还是看不懂，先读同目录下的 [`00_start_here.md`](00_start_here.md)。那份是零基础版，只解释“每一步在干什么”，不解释公式和论文名词。

如果还需要看具体数字，读 [`01_minimal_numeric_example.md`](01_minimal_numeric_example.md)，里面有一个最小可执行脚本和完整运行结果。

同目录还有一个不依赖 PyTorch/Transformers 的玩具脚本：

```bash
cd /home/mtuser/workspace
python insights/deepseek_v4_source_walkthrough/toy_flow_no_torch.py
```

它只打印形状和流程，适合先建立直觉。

这份说明面向第一次读 Transformer 源码的读者。目标不是推公式，而是回答三个问题：

1. 一串 token 输入模型后，数据按什么顺序流动？
2. DeepSeek-V4 比普通 decoder-only Transformer 多了哪些关键模块？
3. 用很小的配置跑一次 forward 时，每个模块会看到什么形状的张量？

## 先记住三句话

DeepSeek-V4 仍然是 decoder-only 语言模型：输入 token，输出每个位置预测下一个 token 的 logits。

它最特别的三块是：

- **混合注意力**：每层都有滑动窗口注意力，部分层还会拼上压缩后的长程 KV。
- **mHC 残差流**：不是一条残差线，而是 `hc_mult` 条并行残差流。
- **MoE 前馈层**：每个 token 只走少数几个专家，同时还有一个共享 MLP。

可以先把它想成下面这条流水线：

```text
input_ids
  -> embedding
  -> 扩成 hc_mult 条残差流
  -> 多个 DecoderLayer
       -> mHC 折叠多流，送入 Attention
       -> Attention 做滑窗 + 可选压缩长程注意力
       -> mHC 把 Attention 输出混回多流
       -> mHC 再折叠多流，送入 MoE
       -> MoE 选择专家 + 共享 MLP
       -> mHC 把 MoE 输出混回多流
  -> HyperHead 折叠多流
  -> RMSNorm
  -> lm_head
  -> logits
```

## 源码入口

主要看手写源码，不要改自动生成文件：

- [`configuration_deepseek_v4.py`](../../repos/transformers/src/transformers/models/deepseek_v4/configuration_deepseek_v4.py)：配置字段和兼容旧 checkpoint 的逻辑。
- [`modular_deepseek_v4.py`](../../repos/transformers/src/transformers/models/deepseek_v4/modular_deepseek_v4.py)：真正应该阅读和修改的源码。
- [`modeling_deepseek_v4.py`](../../repos/transformers/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py)：由 modular 文件自动生成，运行时导入这个文件，但不要手改。
- [`test_modeling_deepseek_v4.py`](../../repos/transformers/tests/models/deepseek_v4/test_modeling_deepseek_v4.py)：tiny config 测试和真实 checkpoint 慢测。

## 一条输入怎么走完整个模型

假设输入是 8 个 token：

```text
input_ids.shape = [batch=1, seq=8]
```

### 1. CausalLM 外壳

`DeepseekV4ForCausalLM.forward` 负责语言模型头：

```text
DeepseekV4ForCausalLM
  -> self.model(...) 得到 hidden_states
  -> self.lm_head(hidden_states) 得到 logits
  -> 如果 labels 不为空，再算 loss
```

源码位置：`DeepseekV4ForCausalLM` 在 `modeling_deepseek_v4.py`，生成文件里包含 `lm_head`、loss 和辅助 router loss。

### 2. 基础模型 forward

`DeepseekV4Model.forward` 做几件事：

- 检查 `input_ids` 和 `inputs_embeds`：只能传一个。
- 如果没有 `past_key_values`，创建 `DynamicCache(config=self.config)`。
- 把 token id 变成 embedding：

```text
input_ids       [1, 8]
inputs_embeds   [1, 8, hidden_size]
```

- 构造滑动窗口 causal mask。
- 预先算两套 RoPE：

```text
main     -> 给纯滑窗层使用，rope_theta 默认 10000
compress -> 给 CSA/HCA 压缩分支使用，compress_rope_theta 默认 160000
```

- 把一条 hidden state 扩成多条 mHC 残差流：

```text
inputs_embeds   [1, 8, D]
hidden_streams  [1, 8, hc_mult, D]
```

### 3. 每个 DecoderLayer

每层结构可以看成：

```text
hidden_streams
  -> attn_hc(hidden_streams)
       返回 post, comb, collapsed
  -> self_attn(layernorm(collapsed))
  -> 用 post 和 comb 把 attn_output 混回 hidden_streams
  -> ffn_hc(hidden_streams)
       返回 post, comb, collapsed
  -> mlp(layernorm(collapsed))
  -> 用 post 和 comb 把 mlp_output 混回 hidden_streams
```

这里的 `collapsed` 是给子模块用的一条普通 hidden state：

```text
collapsed.shape = [B, S, D]
```

而 `hidden_streams` 一直保持多流：

```text
hidden_streams.shape = [B, S, hc_mult, D]
```

## 关键模块通俗注释

### 配置：DeepseekV4Config

配置决定每一层是什么类型。最重要的是两个列表：

```python
layer_types = [
    "sliding_attention",
    "heavily_compressed_attention",
    "compressed_sparse_attention",
]

mlp_layer_types = [
    "hash_moe",
    "moe",
    "moe",
]
```

`layer_types[i]` 决定第 `i` 层 attention：

- `sliding_attention`：只看局部窗口。
- `heavily_compressed_attention`，简称 HCA：看局部窗口 + 高压缩长程 KV。
- `compressed_sparse_attention`，简称 CSA：看局部窗口 + 低压缩长程 KV + indexer 选 TopK。

`mlp_layer_types[i]` 决定第 `i` 层 MoE 路由：

- `hash_moe`：根据 token id 查表选专家。
- `moe`：根据 learned gate 的分数动态选专家。

配置里还会兼容旧 checkpoint 字段。例如旧字段 `compress_ratios = [128, 4, 128]` 会被转成：

```python
["heavily_compressed_attention", "compressed_sparse_attention", "heavily_compressed_attention"]
```

### RoPE：只旋转最后一小段

普通实现常常对整个 head 做 RoPE。DeepSeek-V4 只对 head 的最后一段做 RoPE：

```text
head = [nope 部分 | rope 部分]
```

`apply_rotary_pos_emb` 做的事是：

1. `cos/sin` 本来只有 pair 数量的一半，先 `repeat_interleave(2)` 扩成完整 rope 维度。
2. 拆开 head：前面 `nope` 不动，后面 `rope` 旋转。
3. 拼回原形状。

注意力输出后还会用 `-sin` 做一次反向旋转。原因是 V4 里 K=V，V 也被旋转了；输出前反向旋转能让结果更像“只和相对距离有关”。

### Cache：为什么 V4 需要自定义缓存

普通生成只缓存每层的 K/V。V4 除了滑动窗口 K/V，还要缓存压缩分支的状态：

- 当前窗口里还不够压缩的 token。
- 已经压好的 compressed KV。
- CSA 的 overlap 状态。
- CSA indexer 自己的 compressed KV。

所以 V4 定义了两个 cache layer：

- `DeepseekV4HCACache`：给 HCA 用。
- `DeepseekV4CSACache`：给 CSA 用。

`DynamicCache(config=...)` 会根据 `config.layer_types` 自动创建对应 cache。

### HCA：把很多 token 压成一个长程记忆

HCA 的逻辑很像“每 128 个 token 写一条摘要”。tiny 示例里可以把 128 改成 4，方便观察。

流程：

```text
hidden_states [B, S, D]
  -> kv_proj   得到每个 token 的候选 KV
  -> gate_proj 得到每个 token 的权重
  -> 每 compress_rate 个 token 分一组
  -> 对 gate 做 softmax
  -> 加权求和，得到 1 个 compressed KV
  -> 拼到普通滑窗 KV 后面
```

HCA 没有 indexer。只要某个压缩窗口已经闭合，后面的 query 就可以看到它。

### CSA：压缩后还要让 indexer 挑重点

CSA 比 HCA 更细。默认每 4 个 token 压一次，但是不会让每个 query 看所有 compressed KV，而是用 `DeepseekV4Indexer` 挑出最相关的 `index_topk` 个。

CSA 的压缩有两个序列：

```text
Ca: 给下一个窗口用
Cb: 给当前窗口用
```

第 `w` 个 compressed KV 来自：

```text
前一个窗口的 Ca + 当前窗口的 Cb
```

这就是代码里 overlap state 的来源。跨 forward 调用时，上一段输入最后一个窗口的 Ca 要保存下来，下一段继续用。

Indexer 的流程：

```text
query q
  -> 和 indexer 的 compressed key 做相似度
  -> ReLU
  -> 乘每个 index head 的权重
  -> 对 compressed entries 求 topk
  -> 得到 top_k_indices
```

CSA compressor 再把这些 index 变成 `block_bias` mask：

```text
被选中的 compressed KV -> 0
没选中的 compressed KV -> -inf
```

这样核心 attention 只会看 indexer 挑出来的 compressed entries。

### Attention：滑窗 KV + 压缩 KV 拼在一起

`DeepseekV4Attention.forward` 可以按这几步读：

1. 计算 Q：

```text
hidden_states -> q_a_proj -> q_a_norm -> q_b_proj -> q_b_norm -> RoPE
```

2. 计算共享 KV：

```text
hidden_states -> kv_proj -> kv_norm -> RoPE
```

V4 是 shared K=V MQA，所以只有一个 KV head，后面广播给所有 query heads。

3. 更新滑动窗口 cache。
4. 如果本层是 CSA/HCA，调用 compressor 得到：

```text
compressed_kv [B, 1, compressed_len, head_dim]
block_bias    [B, 1, S, compressed_len]
```

5. 拼接 KV：

```text
kv = cat([sliding_kv, compressed_kv], dim=kv_seq_axis)
```

6. 拼接 mask：

```text
attention_mask = cat([sliding_mask, block_bias], dim=key_axis)
```

7. 调 eager attention。
8. 对 attention 输出的 rope 部分做反向旋转。
9. 用 grouped output projection 降低输出投影开销：

```text
attn_output
  -> 按 o_groups 分组
  -> o_a_proj 每组独立投影
  -> flatten
  -> o_b_proj 混回 hidden_size
```

### mHC：不是简单残差相加

普通 Transformer 常见写法：

```text
x = x + Attention(LN(x))
x = x + MLP(LN(x))
```

V4 的 mHC 更像：

```text
多条残差流 hidden_streams
  -> 算 pre：怎么把多条流合成一条给子层
  -> 算 post：子层输出放回每条流的比例
  -> 算 comb：多条旧残差流之间怎么互相混合
```

其中 `comb` 会经过 Sinkhorn 归一化，让矩阵接近“双随机矩阵”。小白可以先把它理解成：模型学习了一种更稳的残差混合方式，而不是简单地 `x + y`。

### MoE：每个 token 只走少数专家

MoE block 里有两部分：

```text
routed experts + shared expert
```

对每个 token：

1. router 选出 `num_experts_per_tok` 个专家。
2. 这些专家分别处理该 token。
3. 按 router 权重加权求和。
4. 再加上共享 MLP 的输出。

`hash_moe` 和 `moe` 的区别只在“选哪些专家”：

- `hash_moe`：查 `tid2eid[input_ids]`，专家选择固定。
- `moe`：用 `TopKRouter` 动态计算。

专家内部是 SwiGLU：

```text
gate, up = gate_up.chunk(2)
gate = gate.clamp(max=swiglu_limit)
up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
output = silu(gate) * up
```

## 完整流程模拟

本目录提供了一个完整流程模拟脚本：

```bash
cd /home/mtuser/workspace
python insights/deepseek_v4_source_walkthrough/run_deepseek_v4_tiny_trace.py
```

这个脚本不依赖 Transformers、PyTorch、DeepSeek 权重或 GPU。它使用 4 个 token、2 维向量和 3 层简化模块，完整模拟一次 DeepSeekV4 风格的执行流程。

```text
input_ids -> embedding -> mHC residual streams
          -> sliding attention
          -> HCA compressed attention
          -> CSA compressed attention + indexer
          -> MoE router + experts
          -> lm_head logits
```

三层分别演示：

```text
layer 0: sliding_attention
layer 1: heavily_compressed_attention
layer 2: compressed_sparse_attention
```

预期输出会包含以下关键步骤：

```text
Step 1. Embedding: token id -> hidden vector
Step 2. mHC expands hidden states into residual streams
Layer 0: sliding_attention
Layer 1: heavily_compressed_attention
Layer 2: compressed_sparse_attention
Final hidden states after mHC head
lm_head logits
```

这些数字的含义：

- `[4, 2]`：4 个 token，每个 token 用 2 维向量表示。
- `[tokens=4, streams=2, dim=2]`：mHC 为每个 token 保留 2 条残差流。
- `HCA compressed entries`：按窗口把多个 token 压成摘要。
- `CSA compressed entries`：先压缩，再由 indexer 选择相关摘要。
- `lm_head logits`：把最后 hidden state 转成词表分数。

## 推荐阅读顺序

第一次读源码时，不要从文件顶部一路读到底。按下面顺序更容易：

1. 读 `DeepseekV4Model.forward`：建立整体流水线。
2. 读 `DeepseekV4DecoderLayer.forward`：理解一层里 attention 和 MoE 怎么接。
3. 读 `DeepseekV4HyperConnection`：理解为什么 hidden state 是 4D。
4. 读 `DeepseekV4Attention.forward`：看 Q/KV、compressor、mask、output projection。
5. 读 `DeepseekV4HCACompressor`：先理解简单压缩。
6. 读 `DeepseekV4CSACompressor` 和 `DeepseekV4Indexer`：再理解重叠窗口和 TopK 选择。
7. 读 `DeepseekV4SparseMoeBlock`、`TopKRouter`、`HashRouter`：理解专家路由。
8. 最后读 `DeepseekV4Config.__post_init__`：理解 checkpoint 字段为什么会变形。

## 常见困惑

### 为什么 `modeling_deepseek_v4.py` 也有完整源码？

它是由 `modular_deepseek_v4.py` 自动生成的。实际导入模型时用它，但开发时应该改 modular 文件，否则下次生成会覆盖手工修改。

### 为什么不用 FlashAttention？

源码里关闭了 FlashAttention、SDPA 和 FlexAttention。主要原因是 V4 的 `head_dim=512` 超过常见 FlashAttention 限制，SDPA 不支持 V4 的 attention sink，FlexAttention 不适合 compressor 在 attention 内部动态拼接 KV 的模式。

### 为什么左 padding 不兼容？

压缩分支先按固定窗口把 token 池化，再应用 attention mask。左 padding 会改变窗口边界，导致 pad token 被压进 compressed KV，所以 logits 会和不 padding 的输入不同。

### tiny 示例能说明真实模型效果吗？

不能。tiny 示例是随机权重，只用于观察执行流程和张量形状。真实生成质量要加载 `deepseek-ai/DeepSeek-V4-Flash` 这类 checkpoint。
