# DeepSeek-V4 零基础先看版

这份只讲最基本的执行流程。先不要管公式，也不要管论文名词。

## 你先把模型想成一个文本工厂

输入一句话：

```text
我 喜欢 深度 学习
```

模型不会直接看汉字。它先把每个词或 token 变成数字编号：

```text
[10, 25, 87, 66]
```

然后每个编号会变成一串小数，也就是向量：

```text
10 -> [0.1, -0.3, 0.8, ...]
25 -> [0.4,  0.2, 0.5, ...]
```

源码里叫：

```text
input_ids -> embed_tokens -> hidden_states
```

你先记住：

```text
token 编号 -> 向量 -> 一层一层加工 -> 预测下一个 token
```

## 一个最小例子

假设输入 8 个 token：

```text
[1, 2, 3, 4, 5, 6, 7, 8]
```

为了简单，假设每个 token 变成 16 个数字：

```text
input_ids      = 1 行 8 个 token
hidden_states  = 1 行 8 个 token，每个 token 16 个数字
```

写成形状就是：

```text
input_ids.shape     = [1, 8]
hidden_states.shape = [1, 8, 16]
```

这里的三个数字分别是：

```text
1  = batch，一次处理几句话
8  = seq，一句话里有几个 token
16 = hidden_size，每个 token 用多少个数字表示
```

## DeepSeek-V4 每层主要做两件事

每一层大体是：

```text
Attention
MoE
```

Attention 负责：

```text
当前 token 应该参考前面哪些 token？
```

MoE 负责：

```text
找几个专家来处理当前 token。
```

所以一层可以想成：

```text
输入向量
  -> Attention：看上下文
  -> MoE：找专家加工
  -> 输出向量
```

多层叠起来就是：

```text
第 0 层加工一次
第 1 层再加工一次
第 2 层再加工一次
...
最后预测下一个 token
```

## V4 比普通 Transformer 多了三个东西

### 1. 压缩注意力

普通注意力会尽量看很多历史 token，但长文本太贵。

V4 的做法是：

```text
近处 token：保留原样看
远处 token：先压缩成摘要再看
```

举例，8 个 token：

```text
[1, 2, 3, 4, 5, 6, 7, 8]
```

如果每 4 个 token 压成一个摘要：

```text
[1, 2, 3, 4] -> 摘要 A
[5, 6, 7, 8] -> 摘要 B
```

模型后面看长距离内容时，不一定看全部 8 个 token，而是看：

```text
局部 token + 摘要 A + 摘要 B
```

源码里：

```text
HCA = heavily_compressed_attention，高压缩
CSA = compressed_sparse_attention，低压缩再挑重点
```

先理解 HCA 就够了：很多 token 压成少数几个摘要。

### 2. mHC 残差流

普通 Transformer 每个 token 只有一份 hidden state：

```text
[B, S, D]
```

V4 会复制成多份并行流：

```text
[B, S, hc_mult, D]
```

如果 `hc_mult=2`，就像每个 token 同时带着两份草稿：

```text
token 1: 草稿 A + 草稿 B
token 2: 草稿 A + 草稿 B
...
```

每层先把多份草稿合成一份给 Attention 或 MoE，用完后再混回多份草稿。

源码里：

```text
DeepseekV4HyperConnection = 负责多份草稿怎么合并、怎么混回去
DeepseekV4HyperHead       = 最后把多份草稿合成一份
```

### 3. MoE 专家

普通 MLP 是所有 token 都走同一个前馈网络。

MoE 是准备很多专家，但每个 token 只选几个：

```text
token 1 -> 专家 0 + 专家 3
token 2 -> 专家 1 + 专家 2
token 3 -> 专家 0 + 专家 2
```

这样总专家很多，但每个 token 实际只跑少数几个。

源码里：

```text
DeepseekV4TopKRouter  = 动态算每个 token 该选哪些专家
DeepseekV4HashRouter  = 根据 token id 查表选专家
DeepseekV4Experts     = 真正的专家计算
```

## 最简单的完整流程

看这条就够：

```text
input_ids
  -> embed_tokens，把 token 编号变成向量
  -> 复制成 hc_mult 份残差流
  -> 第 0 层
       -> mHC 合成一份
       -> Attention 看上下文
       -> mHC 混回多份
       -> mHC 再合成一份
       -> MoE 找专家加工
       -> mHC 混回多份
  -> 第 1 层
       -> 重复上面过程
  -> 第 2 层
       -> 重复上面过程
  -> HyperHead，把多份残差流合成一份
  -> lm_head，预测下一个 token
```

## 你读源码时只盯这几个类

先不要全读。按这个顺序：

1. `DeepseekV4Model.forward`
2. `DeepseekV4DecoderLayer.forward`
3. `DeepseekV4Attention.forward`
4. `DeepseekV4HCACompressor.forward`
5. `DeepseekV4SparseMoeBlock.forward`

如果这 5 个能看懂 60%，再回头看 CSA、Indexer、RoPE、Cache。

## 先跑无依赖玩具脚本

这个脚本不跑真实模型，只模拟形状变化：

```bash
cd /home/mtuser/workspace
python insights/deepseek_v4_source_walkthrough/toy_flow_no_torch.py
```

你会看到类似：

```text
input_ids                 [1, 8]
embed_tokens              [1, 8, 16]
expand_hc_streams         [1, 8, 2, 16]
layer 0 sliding_attention
layer 1 HCA compress 8 tokens -> 2 compressed entries
layer 2 CSA compress 8 tokens -> 4 entries, each query picks top 2
lm_head                   [1, 8, 32]
```

先能看懂这个输出，再看详细版 README。

如果你需要看到每一步的具体数字，继续看 [`01_minimal_numeric_example.md`](01_minimal_numeric_example.md)，并运行：

```bash
cd /home/mtuser/workspace
python insights/deepseek_v4_source_walkthrough/minimal_numeric_example.py
```
