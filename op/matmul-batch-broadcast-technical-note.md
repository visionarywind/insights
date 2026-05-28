# PyTorch matmul 多维矩阵乘法与批量广播

`torch.matmul` 用于向量点积、矩阵乘法和批量矩阵乘法。阅读 Transformer、Llama、DeepSeek V4 等模型源码时，`matmul` 主要出现在 attention score、attention value 聚合、投影计算和部分自定义 fallback 路径中。

理解 `matmul` 需要先分清两类维度：

| 维度类型 | 位置 | 作用 |
|---|---|---|
| 矩阵维度 | 最后两维 | 真正参与矩阵乘法。 |
| 批量维度 | 最后两维之前的所有维度 | 表示有多少组矩阵要分别相乘。 |

## 基本规则

对两个高维 tensor：

```text
a: [..., M, K]
b: [..., K, N]
out: [..., M, N]
```

规则如下：

1. `a` 的最后一维必须等于 `b` 的倒数第二维，也就是 `K` 必须一致。
2. `a` 的最后两维 `[M, K]` 和 `b` 的最后两维 `[K, N]` 执行矩阵乘法。
3. 最后两维之前的维度是批量维度。
4. 批量维度按 PyTorch 广播规则对齐。

示例：

```text
[2, 3] @ [3, 4] -> [2, 4]
```

这里没有批量维度，只是普通矩阵乘法。

```text
[5, 2, 3] @ [5, 3, 4] -> [5, 2, 4]
```

这里 `5` 是批量维度，表示执行 5 次 `[2, 3] @ [3, 4]`。

## 批量广播规则

批量广播只作用在最后两维之前的维度。

广播规则：

1. 从右往左对齐批量维度。
2. 两个维度相同，可以广播。
3. 其中一个维度是 `1`，可以扩展成另一个维度。
4. 其中一个 tensor 缺少该维度，相当于补 `1`。
5. 两个维度不同且都不为 `1`，不能广播。

示例：

| `a` 批量维度 | `b` 批量维度 | 结果批量维度 | 是否合法 |
|---|---|---|---|
| `[2, 1]` | `[1, 3]` | `[2, 3]` | 合法 |
| `[2, 3]` | `[1, 3]` | `[2, 3]` | 合法 |
| `[2, 1, 4]` | `[3, 4]` | `[2, 3, 4]` | 合法 |
| `[2, 3]` | `[4, 3]` | 无 | 不合法 |

广播不是一定会复制数据。PyTorch 通常通过 stride 视图复用数据，只有后续算子或布局要求不满足时才可能产生实际复制。

## 最小完整例子：两个输入都带批量维度

这个例子中，`a` 和 `b` 都有批量维度。

```python
import torch

# a.shape = [2, 1, 2, 2]
# 最后两维 [2, 2] 是矩阵。
# 前两维 [2, 1] 是批量维度。
a = torch.tensor([
    [
        [[1., 2.],
         [3., 4.]]
    ],
    [
        [[10., 20.],
         [30., 40.]]
    ]
])

# b.shape = [1, 3, 2, 1]
# 最后两维 [2, 1] 是矩阵。
# 前两维 [1, 3] 是批量维度。
b = torch.tensor([
    [
        [[1.],
         [2.]],

        [[3.],
         [4.]],

        [[5.],
         [6.]]
    ]
])

# matmul 的最后两维执行矩阵乘法：
# [2, 2] @ [2, 1] -> [2, 1]
#
# matmul 的批量维度执行广播：
# a 的批量维度 [2, 1]
# b 的批量维度 [1, 3]
# 广播结果是 [2, 3]
#
# 所以 out.shape = [2, 3, 2, 1]
out = torch.matmul(a, b)

print("a.shape =", tuple(a.shape))
print("b.shape =", tuple(b.shape))
print("out.shape =", tuple(out.shape))
print(out)
```

输出：

```text
a.shape = (2, 1, 2, 2)
b.shape = (1, 3, 2, 1)
out.shape = (2, 3, 2, 1)
tensor([[[[  5.],
          [ 11.]],

         [[ 11.],
          [ 25.]],

         [[ 17.],
          [ 39.]]],


        [[[ 50.],
          [110.]],

         [[110.],
          [250.]],

         [[170.],
          [390.]]]])
```

### 第一步：拆分矩阵维度和批量维度

`a` 的 shape：

```text
a.shape = [2, 1, 2, 2]
           |  |  |  |
           |  |  M  K
           |  batch
           batch
```

`b` 的 shape：

```text
b.shape = [1, 3, 2, 1]
           |  |  |  |
           |  |  K  N
           |  batch
           batch
```

最后两维：

```text
a 的矩阵维度: [2, 2]
b 的矩阵维度: [2, 1]
```

矩阵乘法结果：

```text
[2, 2] @ [2, 1] -> [2, 1]
```

### 第二步：计算批量维度广播

批量维度：

```text
a batch = [2, 1]
b batch = [1, 3]
```

逐位对齐：

```text
第 1 个 batch 维: 2 和 1 -> 2
第 2 个 batch 维: 1 和 3 -> 3
```

广播后的批量维度：

```text
[2, 3]
```

最终输出：

```text
out.shape = [2, 3, 2, 1]
```

### 第三步：展开每一次矩阵乘法

广播后实际等价于 6 次矩阵乘法：

```text
out[0, 0] = a[0, 0] @ b[0, 0]
out[0, 1] = a[0, 0] @ b[0, 1]
out[0, 2] = a[0, 0] @ b[0, 2]

out[1, 0] = a[1, 0] @ b[0, 0]
out[1, 1] = a[1, 0] @ b[0, 1]
out[1, 2] = a[1, 0] @ b[0, 2]
```

`a` 的第二个 batch 维是 `1`，所以 `a[0, 0]` 会复用于 `out[0, 0]`、`out[0, 1]`、`out[0, 2]`。

`b` 的第一个 batch 维是 `1`，所以 `b[0, 0]`、`b[0, 1]`、`b[0, 2]` 会复用于第 0 组和第 1 组输出。

### 第四步：计算一个具体位置

计算：

```text
out[0, 1] = a[0, 0] @ b[0, 1]
```

输入：

```text
a[0, 0] =
[[1, 2],
 [3, 4]]

b[0, 1] =
[[3],
 [4]]
```

矩阵乘法：

```text
第一行: 1 * 3 + 2 * 4 = 11
第二行: 3 * 3 + 4 * 4 = 25
```

结果：

```text
out[0, 1] =
[[11],
 [25]]
```

## 维度不合法的例子

最后两维的矩阵乘法维度必须匹配。

```python
import torch

# a 的最后两维是 [2, 3]
a = torch.randn(4, 2, 3)

# b 的最后两维是 [4, 5]
# 这里 3 != 4，所以不能执行 [2, 3] @ [4, 5]。
b = torch.randn(4, 4, 5)

out = torch.matmul(a, b)
```

会报错，因为矩阵乘法要求：

```text
[M, K] @ [K, N]
```

这里 `K` 不一致。

批量维度也必须能广播。

```python
import torch

# a batch = [2]
a = torch.randn(2, 2, 3)

# b batch = [4]
b = torch.randn(4, 3, 5)

# 2 和 4 不能广播，执行会报错。
out = torch.matmul(a, b)
```

## `matmul`、`mm`、`bmm` 的区别

| API | 支持输入 | 是否支持批量广播 | 典型用途 |
|---|---|---|---|
| `torch.mm` | 只能是 2D 和 2D | 不支持 | 单个矩阵乘法。 |
| `torch.bmm` | 只能是 3D 和 3D | 不支持 | 固定批量矩阵乘法。 |
| `torch.matmul` | 1D、2D、N-D | 支持 | 通用矩阵乘法，Transformer 中最常见。 |

示例：

```python
import torch

# mm: 只能处理二维矩阵。
x = torch.randn(2, 3)
w = torch.randn(3, 4)
y = torch.mm(x, w)
print(tuple(y.shape))  # (2, 4)

# bmm: 只能处理三维批量矩阵乘法。
x = torch.randn(5, 2, 3)
w = torch.randn(5, 3, 4)
y = torch.bmm(x, w)
print(tuple(y.shape))  # (5, 2, 4)

# matmul: 支持更多维度，并支持 batch 广播。
x = torch.randn(2, 1, 2, 3)
w = torch.randn(1, 4, 3, 5)
y = torch.matmul(x, w)
print(tuple(y.shape))  # (2, 4, 2, 5)
```

## Transformer attention 中的 `matmul`

Decoder 模型中的 attention 通常有以下张量：

```text
q.shape = [B, H, S, D]
k.shape = [B, H, S, D]
v.shape = [B, H, S, D]
```

含义：

| 维度 | 含义 |
|---|---|
| `B` | batch size |
| `H` | attention head 数量 |
| `S` | token 序列长度 |
| `D` | 每个 head 的 hidden size |

### 计算 attention score

源码常见写法：

```python
scores = torch.matmul(q, k.transpose(-2, -1))
```

`k.transpose(-2, -1)` 的含义是交换 `k` 的最后两维：

```text
k.shape                      = [B, H, S, D]
k.transpose(-2, -1).shape    = [B, H, D, S]
```

然后执行：

```text
q:   [B, H, S, D]
k^T: [B, H, D, S]
out: [B, H, S, S]
```

最后两维执行矩阵乘法：

```text
[S, D] @ [D, S] -> [S, S]
```

前两维 `[B, H]` 是批量维度，表示每个 batch、每个 head 都独立计算一张 attention score 矩阵。

最小代码：

```python
import torch

B, H, S, D = 2, 4, 3, 8

# q、k 分别表示 query 和 key。
q = torch.randn(B, H, S, D)
k = torch.randn(B, H, S, D)

# transpose(-2, -1) 将 k 的 [S, D] 变成 [D, S]。
# matmul 在每个 batch、每个 head 内计算 [S, D] @ [D, S]。
scores = torch.matmul(q, k.transpose(-2, -1))

print("scores.shape =", tuple(scores.shape))
```

输出：

```text
scores.shape = (2, 4, 3, 3)
```

### 计算 attention 输出

softmax 后的 attention 权重：

```text
probs.shape = [B, H, S, S]
v.shape     = [B, H, S, D]
```

聚合 value：

```python
out = torch.matmul(probs, v)
```

shape 变化：

```text
probs: [B, H, S, S]
v:     [B, H, S, D]
out:   [B, H, S, D]
```

最后两维：

```text
[S, S] @ [S, D] -> [S, D]
```

含义是每个 query token 根据 attention 权重，对所有 value token 做加权求和。

## 运行时技术洞察

### 1. `matmul` 是计算 OP

`matmul` 通常会提交矩阵乘法 kernel。CPU 后端可能进入 BLAS 路径，CUDA/MUSA 后端通常进入 GEMM 或 batched GEMM 路径。

在模型性能分析中，`matmul` 通常对应以下热点：

```text
linear projection
attention score
attention value aggregation
MLP up/down/gate projection
MoE expert projection
```

### 2. 高维 `matmul` 通常会映射到批量 GEMM

当输入是：

```text
[B, H, S, D] @ [B, H, D, S]
```

逻辑上是 `B * H` 次矩阵乘法。后端可能使用 batched GEMM、strided batched GEMM 或更专门的 kernel。

如果模型使用 fused attention，例如 FlashAttention 或后端自研 fused kernel，源码中可能看不到显式的 `matmul + softmax + matmul` 三段式实现，但数学语义仍然对应：

```text
scores = q @ k^T
probs = softmax(scores)
out = probs @ v
```

### 3. `transpose` 不等于复制

`k.transpose(-2, -1)` 通常只修改 shape 和 stride，不直接复制数据。

示例：

```python
import torch

k = torch.randn(2, 4, 3, 8)
kt = k.transpose(-2, -1)

print("k.shape =", tuple(k.shape), "stride =", k.stride(), "contiguous =", k.is_contiguous())
print("kt.shape =", tuple(kt.shape), "stride =", kt.stride(), "contiguous =", kt.is_contiguous())
```

典型输出：

```text
k.shape = (2, 4, 3, 8) stride = (96, 24, 8, 1) contiguous = True
kt.shape = (2, 4, 8, 3) stride = (96, 24, 1, 8) contiguous = False
```

`kt` 是非连续 view。后端 `matmul` 是否能直接处理该 layout，取决于具体 kernel 和后端实现。如果后端要求连续内存，可能出现额外 copy 或 fallback。

### 4. 广播可能引入隐式 expand

批量广播通常不复制数据，而是通过 view 复用输入。

例如：

```text
a batch = [2, 1]
b batch = [1, 3]
out batch = [2, 3]
```

逻辑上 `a` 和 `b` 都被扩展到 `[2, 3]`。这一步通常是 metadata 层面的扩展。

需要注意：

- broadcast 后的 tensor 可能包含 zero stride。
- 如果后续算子要求连续 layout，可能触发 `contiguous()`。
- 如果自定义 kernel 不支持 broadcast stride，需要显式处理输入布局。

### 5. 小矩阵和大矩阵的瓶颈不同

小矩阵场景中，kernel launch、调度开销、batch 数量和 layout 转换可能占比较高。

大矩阵场景中，主要瓶颈通常转向：

```text
GEMM 算力利用率
HBM 带宽
输入输出 layout
dtype / accumulation 精度
```

Decoder 推理中常见两类形态：

| 阶段 | 常见 shape | 主要特点 |
|---|---|---|
| prefill | `S` 较大 | attention score 矩阵大，GEMM 和 attention kernel 是主热点。 |
| decode | `S_q = 1`，KV cache 长度逐步增长 | 单步矩阵较小，launch、cache 访问、调度和 graph replay 更关键。 |

### 6. dtype 会影响后端路径

`float32`、`float16`、`bfloat16`、`int8`、`fp8` 可能进入不同 kernel 路径。

需要重点检查：

```text
输入 dtype
权重 dtype
accumulation dtype
是否启用 TF32
是否触发 cast
```

例如 PyTorch CUDA 后端中，`float32` 矩阵乘法可能受到 TF32 设置影响。MUSA 后端也可能根据 dtype 和 shape 选择不同实现路径。分析性能时不能只看 API 名称，还需要记录 dtype、shape、stride 和后端 kernel。

## 源码阅读检查表

阅读模型源码中的 `matmul` 时，按以下顺序检查：

| 检查项 | 需要确认的问题 |
|---|---|
| 输入 shape | 最后两维是否满足 `[M, K] @ [K, N]`。 |
| 批量维度 | 前置维度是否相同，是否发生广播。 |
| 输出 shape | 输出是否符合后续 reshape、transpose 或 residual add。 |
| stride | 输入是否连续，是否来自 `transpose / permute / expand`。 |
| dtype | 是否发生隐式 cast，是否影响 kernel 路径。 |
| device | 输入是否都在同一设备，是否存在 CPU/GPU/MUSA 混用。 |
| 动态 shape | batch、sequence、head、expert 数是否变化。 |
| 后端 kernel | profiler 中对应 GEMM、batched GEMM、fused attention 还是 fallback。 |

## 常见错误

### 错误 1：把所有维度都当成矩阵乘法维度

错误理解：

```text
[B, H, S, D] 的四个维度都参与矩阵乘法。
```

正确理解：

```text
只有最后两维参与矩阵乘法。
B 和 H 是批量维度。
```

### 错误 2：认为广播一定复制数据

广播通常是 view 级别的复用，不一定分配新内存。是否复制取决于后续算子和 layout 要求。

### 错误 3：忽略 `transpose` 后的 stride

`transpose` 后的 tensor 通常非连续。高性能路径中需要确认后端 kernel 是否支持该 stride。

### 错误 4：只看 shape，不看 dtype 和 device

shape 正确只能说明数学维度合法。实际性能还受 dtype、device、stride、后端 kernel 和 allocator 行为影响。

## 总结

`torch.matmul` 的核心规则是：

```text
最后两维做矩阵乘法。
前面的维度做批量广播。
```

多维 `matmul` 不需要把所有维度都展开理解。先切分：

```text
批量维度 + 矩阵维度
```

再分别判断：

```text
批量维度是否能广播。
矩阵维度是否满足 [M, K] @ [K, N]。
```

在 Transformer 中，attention score 的典型形式是：

```text
[B, H, S, D] @ [B, H, D, S] -> [B, H, S, S]
```

这里 `[B, H]` 是批量维度，`[S, D] @ [D, S]` 才是每个 head 内部真正执行的矩阵乘法。
