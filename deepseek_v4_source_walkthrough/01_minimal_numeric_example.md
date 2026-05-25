# 最小可执行数字例子

如果你看“形状流”还是不理解，就看这个文件。这里不用 PyTorch，也不导入 Transformers。它只是用 4 个 token 和 2 维向量，把 DeepSeek-V4 的关键流程算一遍。

先说明：这不是 DeepSeek-V4 真模型。它只是一个玩具程序，用来理解源码里的模块为什么这样连接。

## 运行命令

```bash
cd /home/mtuser/workspace
python insights/deepseek_v4_source_walkthrough/minimal_numeric_example.py
```

## 这个例子在干什么

输入只有 4 个 token：

```text
[1, 2, 3, 4]
```

每个 token 变成一个 2 维向量：

```text
1 -> [1, 0]
2 -> [0, 1]
3 -> [1, 1]
4 -> [2, 1]
```

然后走 6 步：

1. `Embedding`：token id 变向量。
2. `mHC`：说明 V4 有多条残差流，先合成一条给子模块用。
3. `Compression`：每 2 个 token 压成 1 个摘要。
4. `Attention`：当前位置看附近 token，也看已经压好的摘要。
5. `MoE`：每个 token 选一个专家，再加共享专家。
6. `lm_head`：把最后向量变成下一个 token 的分数。

## 运行结果

下面是脚本的完整输出：

```text
Step 0. Input token ids
  input_ids = [1, 2, 3, 4]

Step 1. Embedding: token id -> 2-number vector
  pos 1: [1, 0]
  pos 2: [0, 1]
  pos 3: [1, 1]
  pos 4: [2, 1]

Step 2. mHC idea: V4 keeps multiple residual streams
  toy setting: hc_mult = 2
  stream A and stream B are two drafts of the same sequence
  before attention/MoE, we collapse them back to one vector per token

Step 3. Compression: every 2 tokens become 1 summary
  tokens 1-2 -> summary 1 = [0.5, 0.5]
  tokens 3-4 -> summary 2 = [1.5, 1]

Step 4. Attention: local nearby tokens + visible compressed summaries
  pos 1: local=[1, 0], compressed=none -> attention_out=[1, 0]
  pos 2: local=[0.5, 0.5], compressed=[0.5, 0.5] -> attention_out=[0.5, 0.5]
  pos 3: local=[0.5, 1], compressed=[0.5, 0.5] -> attention_out=[0.5, 0.75]
  pos 4: local=[1.5, 1], compressed=[1, 0.75] -> attention_out=[1.25, 0.875]

Step 5. MoE: each token chooses one expert, plus one shared expert
  expert 0 doubles feature 0: [x, y] -> [2x, y]
  expert 1 doubles feature 1: [x, y] -> [x, 2y]
  shared expert adds half of the input: [x, y] -> [0.5x, 0.5y]
  token 1: choose expert 0, expert_out=[2, 0], shared=[0.5, 0] -> moe_out=[2.5, 0]
  token 2: choose expert 1, expert_out=[0.5, 1], shared=[0.25, 0.25] -> moe_out=[0.75, 1.25]
  token 3: choose expert 0, expert_out=[1, 0.75], shared=[0.25, 0.375] -> moe_out=[1.25, 1.12]
  token 4: choose expert 1, expert_out=[1.25, 1.75], shared=[0.625, 0.438] -> moe_out=[1.88, 2.19]

Step 6. lm_head: turn each final vector into next-token scores
  toy vocab has 3 possible next tokens: A, B, C
  score(A)=x, score(B)=y, score(C)=0.3*x+0.7*y
  pos 1: final=[2.5, 0], logits=[2.5, 0, 0.75] -> predict A
  pos 2: final=[0.75, 1.25], logits=[0.75, 1.25, 1.1] -> predict B
  pos 3: final=[1.25, 1.12], logits=[1.25, 1.12, 1.16] -> predict A
  pos 4: final=[1.88, 2.19], logits=[1.88, 2.19, 2.09] -> predict B

What this example maps to in the real source:
  Step 1 -> embed_tokens
  Step 2 -> DeepseekV4HyperConnection / DeepseekV4HyperHead
  Step 3 -> DeepseekV4HCACompressor or DeepseekV4CSACompressor
  Step 4 -> DeepseekV4Attention
  Step 5 -> DeepseekV4SparseMoeBlock
  Step 6 -> DeepseekV4ForCausalLM.lm_head
```

## 一步一步解释

### Step 1：Embedding 是查表

输入是：

```text
[1, 2, 3, 4]
```

模型不能直接理解 `1`、`2`、`3`、`4`，所以查 embedding 表：

```text
1 -> [1, 0]
2 -> [0, 1]
3 -> [1, 1]
4 -> [2, 1]
```

真实源码对应：

```text
self.embed_tokens(input_ids)
```

### Step 2：mHC 是“多份草稿”

真实 V4 不是只有一份 hidden state，而是有 `hc_mult` 份。

你可以理解成每个 token 有多份草稿：

```text
token 1 -> 草稿 A + 草稿 B
token 2 -> 草稿 A + 草稿 B
```

Attention 和 MoE 工作前，mHC 先把多份草稿合成一份。工作后，再把结果混回多份草稿。

真实源码对应：

```text
DeepseekV4HyperConnection
```

### Step 3：Compression 是“做摘要”

例子里每 2 个 token 压成一个摘要：

```text
token 1 [1, 0] 和 token 2 [0, 1]
平均后得到 summary 1 = [0.5, 0.5]
```

```text
token 3 [1, 1] 和 token 4 [2, 1]
平均后得到 summary 2 = [1.5, 1]
```

真实 V4 不是简单平均，而是 learned projection + gate softmax 加权求和。但目的类似：把一段 token 变成更少的摘要。

真实源码对应：

```text
DeepseekV4HCACompressor
DeepseekV4CSACompressor
```

### Step 4：Attention 是“看近处 + 看摘要”

例如位置 4：

近处窗口看 token 3 和 token 4：

```text
local = average([1, 1], [2, 1]) = [1.5, 1]
```

它还可以看两个摘要：

```text
summary 1 = [0.5, 0.5]
summary 2 = [1.5, 1]
compressed = average(summary 1, summary 2) = [1, 0.75]
```

最后把 local 和 compressed 合起来：

```text
attention_out = average([1.5, 1], [1, 0.75]) = [1.25, 0.875]
```

真实 V4 的 attention 不是平均，而是 QK 点积 + softmax。但直觉一样：当前 token 从上下文拿信息。

真实源码对应：

```text
DeepseekV4Attention.forward
```

### Step 5：MoE 是“找专家”

例子里有两个专家：

```text
expert 0: 强化第 1 个特征
expert 1: 强化第 2 个特征
```

token 4 是偶数，所以选 expert 1：

```text
attention_out = [1.25, 0.875]
expert 1 output = [1.25, 1.75]
shared output = [0.625, 0.438]
moe_out = [1.875, 2.188]
```

真实 V4 的专家是大矩阵，不是这么简单。但目的一样：不同 token 分给不同专家处理。

真实源码对应：

```text
DeepseekV4SparseMoeBlock
DeepseekV4TopKRouter
DeepseekV4HashRouter
DeepseekV4Experts
```

### Step 6：lm_head 是“给词表打分”

最后每个位置都有一个向量，比如位置 4：

```text
[1.875, 2.188]
```

玩具词表只有 3 个候选：

```text
A, B, C
```

脚本规定：

```text
score(A)=x
score(B)=y
score(C)=0.3*x+0.7*y
```

所以位置 4：

```text
A = 1.875
B = 2.188
C = 2.094
```

最高是 B，所以预测 B。

真实源码对应：

```text
self.lm_head(hidden_states)
```

## 读完这个后再看源码

你现在只需要记住这张表：

| 玩具例子步骤 | 真实源码模块 |
| --- | --- |
| token id 查表变向量 | `embed_tokens` |
| 多份草稿合成/混回 | `DeepseekV4HyperConnection` |
| 每段 token 做摘要 | `DeepseekV4HCACompressor` / `DeepseekV4CSACompressor` |
| 看近处和摘要 | `DeepseekV4Attention` |
| 选专家处理 | `DeepseekV4SparseMoeBlock` |
| 给下一个 token 打分 | `lm_head` |
