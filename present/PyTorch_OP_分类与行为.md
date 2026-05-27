# PyTorch OP 分类与行为

如果说自然语言的最小意义单元是词语，那么AI世界的最小功能单元就是算子（Operator）  
一个OP就是对张量的一次定义明确的操作：可以是简单的加减乘除，也可以是复杂的卷积、注意力计算  
无数这样的“动词”与“名词”组合起来，才写就了千亿参数的模型“文章”

理解算子是理解模型行为、优化性能乃至设计新架构的起点

## PyTorch OP 分类

### View OP：Shape 与 Stride

`view` / `reshape` / `unsqueeze` / `squeeze` / `transpose` / `permute` / `expand` 等 OP 只改变 tensor 的 shape 和 stride 描述，不移动底层数据。

`view` 仅在 stride 兼容时成功，否则直接报错。例如 `x.transpose(1, 2).view(...)` 通常失败，因为 transpose 改变了内存排布。`reshape` 的行为更宽松：stride 兼容时返回 view，不兼容时静默复制数据。

**零拷贝是默认行为，但不是保证，源码中一行 shape 调整，可能在运行时变成一次 D2D copy**

验证 stride 与 contiguity：

```python
import torch

# 连续 tensor，shape=[2,3,4]，stride=(12,4,1)
x = torch.arange(24).view(2, 3, 4)

# transpose 交换维度，只改 metadata
y = x.transpose(1, 2)

# contiguous 将非连续 view 复制为连续内存
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


---

#### `contiguous`

功能：返回内存连续 tensor；已连续则返回自身（零拷贝），否则复制为新连续内存。  
用例：MUSA/TileLang/fused kernel 前满足内存布局要求。

```python
import torch
device = torch.device("musa:0")
# t() 等价于 transpose(0, 1)，只改 stride，得到非连续 view。
x = torch.arange(6, device="musa:0").view(2, 3).t()
# contiguous 将非连续 view 复制为连续 tensor。
y = x.contiguous()

print("x contiguous:", x.is_contiguous(), "stride:", x.stride())
print("y contiguous:", y.is_contiguous(), "stride:", y.stride())
print("y =", y.cpu().tolist())
```

输入：`x` 为 `[[0,3],[1,4],[2,5]]`，transpose 后非连续。  
MUSA 运行结果（MUSA stdout）：
```text
x contiguous: False stride: (1, 3)
y contiguous: True stride: (2, 1)
y = [[0, 3], [1, 4], [2, 5]]
```

注意：延迟敏感路径应确认是否需要 `contiguous()`；若下游 fused kernel 支持非连续 layout，可移除这行 copy。

#### `reshape`

功能：改变 shape，必要时复制。  
用例：量化分组、fallback 参考实现中规整 `[T,G,D]` 等逻辑维度。

```python
import torch
device = torch.device("musa:0")
x = torch.arange(6, device="musa:0")
# 连续 tensor 的 reshape 通常只改 shape/stride 描述。
y = x.reshape(3, 2)

print("x =", x.cpu().tolist())
print("y shape:", tuple(y.shape))
print("y =", y.cpu().tolist())
```

输入：`x=[0,1,2,3,4,5]`。  
MUSA 运行结果（MUSA stdout）：
```text
x = [0, 1, 2, 3, 4, 5]
y shape: (3, 2)
y = [[0, 1], [2, 3], [4, 5]]
```

注意：延迟敏感路径避免用 `reshape` 掩盖隐式 copy；需要固定地址时显式处理 layout。能用 `view` 时优先 `view`。

#### `expand`

功能：通过 zero-stride 扩展维度，不复制数据。  
用例：RoPE 频率对齐、broadcast scale/mask。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([[1.0, 2.0]], device="musa:0")
# expand 不复制数据，扩展出的 batch 维 stride 为 0。
y = x.expand(3, 2)

print("x stride:", x.stride(), "y stride:", y.stride())
print("y =", y.cpu().tolist())
```

输入：`x=[[1,2]]`。  
MUSA 运行结果（MUSA stdout）：
```text
x stride: (2, 1) y stride: (0, 1)
y = [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
```

注意：expanded view 是 zero-stride（stride 含 0），多个位置指向同一内存。**不应用作原地写入目标**——写一处会影响所有"扩展"位置。

#### `view`

功能：返回 tensor 的 view，共享底层数据；stride 不兼容时报错。  
用例：线性层输出 shape 重组。

```python
import torch
device = torch.device("musa:0")
x = torch.arange(6, device="musa:0")
# view 要求原 tensor 的 stride 与目标 shape 兼容。
y = x.view(2, 3)
print("y shape:", tuple(y.shape), "stride:", y.stride())
print("y =", y.cpu().tolist())
```

输入：`x=[0,1,2,3,4,5]`。  
MUSA 运行结果（MUSA stdout）：
```text
y shape: (2, 3) stride: (3, 1)
y = [[0, 1, 2], [3, 4, 5]]
```

注意：stride 不兼容时直接报错而非静默复制。无法满足 shape 需求时用 `reshape` 或 `contiguous()` + `view`。

#### `unsqueeze`

功能：在指定位置插入 size=1 的新维度。  
用例：broadcast 前对齐维度。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([1.0, 2.0, 3.0], device="musa:0")
# 在 dim=0 插入一个 size=1 的 batch 维。
y = x.unsqueeze(0)
print("y shape:", tuple(y.shape))
print("y =", y.cpu().tolist())
```

输入：`x=[1,2,3]`，shape `(3,)`。  
MUSA 运行结果（MUSA stdout）：
```text
y shape: (1, 3)
y = [[1.0, 2.0, 3.0]]
```

注意：unsqueeze 只改 metadata 零拷贝；`unsqueeze(-1)` 得到 `(3,1)`。

#### `squeeze`

功能：移除所有 size=1 的维度，或指定特定 dim 移除。  
用例：移除 batch 或 head 维度。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([[[1.0], [2.0], [3.0]]], device="musa:0")
# 只移除最后一维，避免误删 batch 维。
y = x.squeeze(-1)
print("x shape:", tuple(x.shape), "y shape:", tuple(y.shape))
print("y =", y.cpu().tolist())
```

输入：`x` shape `(1,3,1)`。  
MUSA 运行结果（MUSA stdout）：
```text
x shape: (1, 3, 1) y shape: (1, 3)
y = [[1.0, 2.0, 3.0]]
```

注意：避免使用无参数的 `squeeze()`，可能意外删除 batch 维度。应始终指定具体 dim。

#### `transpose`

功能：交换两个维度，只改 metadata 不复制数据。  
用例：attention 中 `seq_len` 和 `head_dim` 的 layout 调整。

```python
import torch
device = torch.device("musa:0")
x = torch.arange(6, device="musa:0").view(2, 3)
# 交换第 0 维和第 1 维，底层数据不复制。
y = x.transpose(0, 1)
print("x shape:", tuple(x.shape), "y shape:", tuple(y.shape))
print("y =", y.cpu().tolist())
```

输入：`x=[[0,1,2],[3,4,5]]`。  
MUSA 运行结果（MUSA stdout）：
```text
x shape: (2, 3) y shape: (3, 2)
y = [[0, 3], [1, 4], [2, 5]]
```

注意：transpose 后 tensor 通常非连续，后续 `view()` 会失败；需要 `contiguous()` 再 `view`。

#### `flatten`

功能：将指定 dim 范围展平为单个维度。  
用例：多头注意力计算前融合 head 维和 batch 维。

```python
import torch
device = torch.device("musa:0")
x = torch.arange(24, dtype=torch.float32, device="musa:0").reshape(2, 3, 4)
# 从 dim=1 开始展平：[2,3,4] -> [2,12]。
y = x.flatten(1)
print("x shape:", tuple(x.shape), "y shape:", tuple(y.shape))
print("y =", y.cpu().tolist())
```

输入：`x` shape `(2,3,4)`。  
MUSA 运行结果（MUSA stdout）：
```text
x shape: (2, 3, 4) y shape: (2, 12)
y = [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]]
```

注意：`flatten(0, -1)` 等价于 `reshape(-1)`。stride 兼容时零拷贝，否则复制。

### Inplace OP：固定 Buffer 原地更新

`copy_` / `fill_` / `zero_` / `clamp_` / `masked_fill_` / `_foreach_copy_` 等以 `_` 结尾的 OP 直接修改输入 tensor 的内容，不分配新内存。这类 OP 的工程价值在于保持 tensor 对象地址不变。高频推理路径应复用固定 buffer，减少重复分配。

预分配固定 buffer，后续通过 `copy_` 把本轮真实数据写入这些 buffer。如果创建新 tensor 替换旧对象，之前绑定的地址就会失效。

常见写法：`input_ids.copy_(real_ids)`、`seq_lens.fill_(1)`、`out_cache_loc.zero_()`。padding 槽位必须清理，否则上轮残留值会污染 attention mask 或 logits。

---

#### `copy_`

功能：把源 tensor 内容复制到目标 tensor，目标对象地址不变。  
用例：固定 buffer 更新、metadata replay 中原地更新 tensor 字段。

```python
import torch
device = torch.device("musa:0")
dst = torch.tensor([0, 0, 0], device="musa:0")
src = torch.tensor([1, 2, 3], device="musa:0")
# copy_ 写入 dst，但 dst 这个 tensor 对象不变。
print("dst before =", dst.cpu().tolist())
dst.copy_(src)

print("dst after =", dst.cpu().tolist())
print("src =", src.cpu().tolist())
```

输入：`dst=[0,0,0]`，`src=[1,2,3]`。  
MUSA 运行结果（MUSA stdout）：
```text
dst before = [0, 0, 0]
dst after = [1, 2, 3]
src = [1, 2, 3]
```

注意：推理 `no_grad` 路径可使用原地更新；`torch._foreach_copy_` 可批量执行多个 `copy_`，减少 Python 循环和调度开销。

#### `fill_`

功能：原地填充指定标量。  
用例：复用固定 buffer 前写默认值、填 padding 槽位。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([0, 0, 0], device="musa:0")
# fill_ 原地覆盖每个元素。
x.fill_(3)

print("x =", x.cpu().tolist())
```

输入：`x=[0,0,0]`。  
MUSA 运行结果（MUSA stdout）：
```text
x = [3, 3, 3]
```

#### `masked_fill_`

功能：按 bool mask 原地填充值。  
用例：attention mask 处理、logits mask、topk padding id 置 sentinel。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([1, 2, 3, 4], device="musa:0")
mask = torch.tensor([True, False, True, False], device="musa:0")
# mask=True 的位置写入 -1，mask=False 的位置保留原值。
x.masked_fill_(mask, -1)

print("mask =", mask.cpu().tolist())
print("x =", x.cpu().tolist())
```

输入：`x=[1,2,3,4]`，`mask=[True,False,True,False]`。  
MUSA 运行结果（MUSA stdout）：
```text
mask = [True, False, True, False]
x = [-1, 2, -1, 4]
```

注意：推理 `no_grad` 路径可使用原地更新；mask broadcast shape 必须正确。

#### `zero_`

功能：原地将 tensor 所有元素置零。  
用例：attention mask、KV cache padding 槽位清理。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([3, 3, 3], device="musa:0")
# zero_ 原地清零，常用于复用 buffer 前清理残留值。
print("x before =", x.cpu().tolist())
x.zero_()
print("x after =", x.cpu().tolist())
```

输入：`x=[3,3,3]`。  
MUSA 运行结果（MUSA stdout）：
```text
x before = [3, 3, 3]
x after = [0, 0, 0]
```

注意：复用预分配 buffer 前用 `zero_()` 或 `fill_()` 清理残留值。

#### `clamp_`

功能：原地将元素裁切到 `[min, max]` 范围。  
用例：logits 数值稳定、MoE routing score 裁切。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([1.0, -2.0, 20.0], device="musa:0")
# 小于 min 的值写成 min，大于 max 的值写成 max。
x.clamp_(min=0, max=10)
print("x =", x.cpu().tolist())
```

输入：`x=[1,-2,20]`。  
MUSA 运行结果（MUSA stdout）：
```text
x = [1.0, 0.0, 10.0]
```

注意：clamp 的边界可以是 scalar 或 tensor；替换 `min`/`max` 之一即可做单侧裁切。

#### `_foreach_copy_`

功能：批量执行多个 `copy_`，减少 Python 循环开销。  
用例：批量更新多个预分配 buffer。

```python
import torch
device = torch.device("musa:0")
d0 = torch.zeros(2, dtype=torch.int64, device="musa:0")
d1 = torch.zeros(2, dtype=torch.int64, device="musa:0")
src0 = torch.tensor([1, 1], device="musa:0")
src1 = torch.tensor([2, 2], device="musa:0")
# 一次接口调用更新多个目标 tensor，减少 Python 逐个调用 copy_。
torch._foreach_copy_([d0, d1], [src0, src1])
print("d0 =", d0.cpu().tolist())
print("d1 =", d1.cpu().tolist())
```

输入：`d0=[0,0]`，`d1=[0,0]`，`src0=[1,1]`，`src1=[2,2]`。  
MUSA 运行结果（MUSA stdout）：
```text
d0 = [1, 1]
d1 = [2, 2]
```

注意：`_foreach_copy_` 是 PyTorch foreach 机制的一部分，kernel launch 合并比 Python 循环逐个 `copy_` 高效。

### Dynamic Shape OP：动态输出长度

`nonzero` / `unique` / `masked_select` / boolean indexing 的输出长度依赖输入数据值，不只由 shape 决定。例如 `x[x > 0]` 返回多少个元素，取决于运行时 x 中有多少正数。固定 shape 推理流水线通常需要预分配 buffer 和 workspace，因此不适合直接接收这类动态输出。

---

#### `nonzero`

功能：返回非零元素坐标，输出 shape 为 `[N, ndim]`，N 随数据变化。  
用例：找有效 token、稀疏 mask 调试。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([[0, 1, 0], [2, 0, 3]], device="musa:0")
# 输出行数等于非零元素个数，运行时才确定。
idx = torch.nonzero(x)

print("idx shape:", tuple(idx.shape))
print("idx =", idx.cpu().tolist())
```

输入：`x=[[0,1,0],[2,0,3]]`。  
MUSA 运行结果（MUSA stdout）：
```text
idx shape: (3, 2)
idx = [[0, 1], [1, 0], [1, 2]]
```

注意：输出动态长度，不应用于固定 shape 流水线。替代：fixed mask + padding 保持 shape 不变。

#### `unique`

功能：返回唯一值及其索引，输出长度等于唯一值个数。  
用例：expert/block/request 分布统计。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([1, 3, 2, 3, 1, 2], device="musa:0")
# unique 输出长度等于唯一值个数。
u = torch.unique(x)

print("unique count:", u.numel())
print("u =", u.cpu().tolist())
```

输入：`x=[1,3,2,3,1,2]`。  
MUSA 运行结果（MUSA stdout）：
```text
unique count: 3
u = [1, 2, 3]
```

注意：MUSA 实现会触发 stream 同步回读输出长度。在线路径优先用 `bincount(minlength=N)` 做固定容量直方图。

#### `masked_select`

功能：按 mask 选取元素，返回一维 tensor，长度等于 mask 中 `True` 的个数。  
用例：抽取有效 token/logits。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([1, 2, 3, 4], device="musa:0")
mask = torch.tensor([True, False, True, False], device="musa:0")
# 输出长度等于 mask 中 True 的个数。
y = torch.masked_select(x, mask)

print("y shape:", tuple(y.shape))
print("y =", y.cpu().tolist())
```

输入：`x=[1,2,3,4]`，`mask=[True,False,True,False]`。  
MUSA 运行结果（MUSA stdout）：
```text
y shape: (2,)
y = [1, 3]
```

注意：输出动态长度。延迟敏感路径用 `where(condition, x, fill)` 保持原 shape。

#### `where`

功能：双参数 `where(cond, x, y)` 按条件逐元素选择（固定 shape）；单参数 `where(mask)` 返回坐标（动态长度）。  
用例：MoE token 查找、mask 条件选择。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([1, 2, 3], device="musa:0")
# 三参数 where 保持输入 shape 不变。
y = torch.where(x > 1, x, torch.zeros_like(x))

print("y =", y.cpu().tolist())
```

输入：`x=[1,2,3]`。  
MUSA 运行结果（MUSA stdout）：
```text
y = [0, 2, 3]
```

#### `topk`

功能：沿指定 dim 选择最大 k 个值。  
用例：MoE routing、CSA indexer top-k 选择、sampling。

```python
import torch
device = torch.device("musa:0")
scores = torch.tensor([[0.1, 0.7, 0.2], [0.6, 0.1, 0.3]], device="musa:0")
# 每行取最大的 2 个值，indices 是这些值在原行中的位置。
values, indices = torch.topk(scores, 2, dim=-1)
weights = values / values.sum(-1, keepdim=True)

print("indices =", indices.cpu().tolist())
print("values =", [[round(v, 4) for v in row] for row in values.cpu().tolist()])
print("weights =", [[round(v, 4) for v in row] for row in weights.cpu().tolist()])
```

输入：`scores=[[0.1,0.7,0.2],[0.6,0.1,0.3]]`。  
MUSA 运行结果（MUSA stdout）：
```text
indices = [[1, 2], [0, 2]]
values = [[0.7, 0.2], [0.6, 0.3]]
weights = [[0.7778, 0.2222], [0.6667, 0.3333]]
```

注意：固定 k 有利于后续 shape 稳定；top-k shape 和排序稳定性会传导到后续 kernel。

#### `gather`

功能：根据索引从指定 dim 收集元素。  
用例：embedding lookup、token mapping。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([[10, 11, 12], [20, 21, 22]], device="musa:0")
idx = torch.tensor([[2, 0], [1, 1]], device="musa:0")
# 沿 dim=1 按 idx 取值，输出 shape 与 idx 一致。
y = torch.gather(x, 1, idx)
print("idx =", idx.cpu().tolist())
print("y =", y.cpu().tolist())
```

输入：`x=[[10,11,12],[20,21,22]]`，`idx=[[2,0],[1,1]]`。  
MUSA 运行结果（MUSA stdout）：
```text
idx = [[2, 0], [1, 1]]
y = [[12, 10], [21, 21]]
```

注意：输出 shape 与 index tensor 一致。index 越界可能导致未定义行为或报错。

#### `scatter_`

功能：根据索引将 src 的值写入 tensor（原地）。  
用例：logits 去乱序、反向 lookup 映射。

```python
import torch
device = torch.device("musa:0")
x = torch.zeros(3, 5, dtype=torch.int64, device="musa:0")
idx = torch.tensor([[0, 1], [1, 2]], device="musa:0")
# 沿 dim=1 将标量 1 写入 idx 指定的位置。
x.scatter_(1, idx, 1)
print("idx =", idx.cpu().tolist())
print("x =", x.cpu().tolist())
```

输入：`x` shape `(3,5)` zeros，`idx=[[0,1],[1,2]]`。  
MUSA 运行结果（MUSA stdout）：
```text
idx = [[0, 1], [1, 2]]
x = [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0]]
```

注意：index 重复时行为取决于 `reduce` 参数（`'sum'` 累加 vs 默认覆盖）。scatter 的 index 越界会报错。

#### `index_select`

功能：沿指定 dim 按 1D index 选取切片。  
用例：稀疏 token 选取、KV block 选取。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([[1, 2], [3, 4], [5, 6]], device="musa:0")
idx = torch.tensor([0, 2], device="musa:0")
# 沿 dim=0 选择第 0 行和第 2 行。
y = torch.index_select(x, 0, idx)
print("idx =", idx.cpu().tolist())
print("y =", y.cpu().tolist())
```

输入：`x=[[1,2],[3,4],[5,6]]`，`idx=[0,2]`。  
MUSA 运行结果（MUSA stdout）：
```text
idx = [0, 2]
y = [[1, 2], [5, 6]]
```

注意：`index_select` 输出 shape 固定（取决于 index 长度，不依赖数据值），比 bool masking 更容易接入固定 shape 流水线。

#### `index_add_`

功能：按 index 将 src 累加到目标 tensor（原地）。  
用例：MoE token 回写、梯度累积。

```python
import torch
device = torch.device("musa:0")
x = torch.zeros(2, 2, dtype=torch.int64, device="musa:0")
src = torch.tensor([[1, 0], [0, 2], [3, 4]], device="musa:0")
idx = torch.tensor([0, 1, 0], device="musa:0")
# idx 中两个 0 会把 src 的第 0 行和第 2 行都累加到 x[0]。
x.index_add_(0, idx, src)
print("idx =", idx.cpu().tolist())
print("x =", x.cpu().tolist())
```

输入：`x=[[0,0],[0,0]]`，`src=[[1,0],[0,2],[3,4]]`，`idx=[0,1,0]`。  
MUSA 运行结果（MUSA stdout）：
```text
idx = [0, 1, 0]
x = [[4, 4], [0, 2]]
```

注意：index 相同的行会累加（不是覆盖）。与 `scatter_` 不同，`index_add_` 是累加语义。index 越界会报错。

### Sync OP：同步边界

Sync OP 控制 CPU 与 DEVICE 之间的时序，分为显式同步和隐式同步两类。

**显式同步**：`synchronize()`、`Stream`、`Event.record()`、`wait_event()`。`synchronize()` 阻塞 CPU 直到 DEVICE 上所有已提交操作完成，适用于 benchmark 计时和错误定位，但放入 decode 热点路径会打断 CPU/DEVICE 异步流水。局部依赖应使用 stream/event 管理，精确控制等待范围。

**隐式同步**：`.item()`、`.tolist()`、`.cpu()`、`.numpy()`。这些 API 的调用形式是取值或类型转换，但作用在 DEVICE tensor 上时，CPU 必须先等待 GPU 完成所有前序操作，再发起 D2H 拷贝。

原则：CPU侧维护一份元数据副本（`seq_lens_cpu`），避免从 DEVICE tensor 读取元数据驱动 Python 分支。最终 token 和少量 logprob 可以回读；完整 logits、hidden states、KV metadata 不应频繁回 CPU。

---

#### `torch.musa.synchronize`

功能：阻塞 CPU 直到 MUSA 设备上所有 stream 操作完成。  
用例：benchmark 计时、错误定位、程序退出前等待。

```python
import torch
device = torch.device("musa:0")
x = torch.arange(4, dtype=torch.float32, device="musa:0")
y = x * 2
# synchronize 等待前面的 device 计算完成。
torch.musa.synchronize()

print("y =", y.cpu().tolist())
print("y mean =", y.mean().cpu().item())
```

输入：`x=[0,1,2,3]`。  
MUSA 运行结果（MUSA stdout）：
```text
y = [0.0, 2.0, 4.0, 6.0]
y mean = 3.0
```

注意：**不应放入 decode 热点路径**。该调用会打断 CPU/DEVICE 并行流水。benchmark 或调试时才显式同步。

#### `.item()`

功能：将 DEVICE tensor 的标量值读回 CPU。  
用例：获取最终 token id、loss scalar。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([3.14], device="musa:0")
# item 会把 device scalar 读回 Python 标量。
val = x.item()

print("val =", val)
```

输入：`x=[3.14]`。  
MUSA 运行结果（MUSA stdout）：
```text
val = 3.140000104904175
```

注意：底层触发 D2H copy + stream 同步。高频调用（如 per-token 回读）会把异步执行变成 CPU 等待。

#### `.cpu()`

功能：将 DEVICE tensor 复制到 CPU。  
用例：最终输出、少量 logprob、离线分析。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([1, 2, 3], device="musa:0")
# cpu() 将 device tensor 拷贝到 host 侧。
y = x.cpu()

print("x device:", x.device, "y device:", y.device)
print("y =", y.tolist())
```

输入：`x=[1,2,3]`。  
MUSA 运行结果（MUSA stdout）：
```text
x device: musa:0 y device: cpu
y = [1, 2, 3]
```

注意：触发 D2H copy + 隐式同步。仅用于最终结果或少量统计，不应频繁拉回完整 logits、hidden states 或 KV metadata。

#### `to(dtype)`

功能：精度转换，可能触发 kernel 和数据复制。  
用例：Norm 内部升精度、量化前后 cast。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([1.5, 2.7], device="musa:0")
# float32 转 int32，小数部分按 PyTorch cast 规则截断。
y = x.to(torch.int32)

print("x dtype:", x.dtype, "y dtype:", y.dtype)
print("y =", y.cpu().tolist())
```

输入：`x=[1.5, 2.7]`。  
MUSA 运行结果（MUSA stdout）：
```text
x dtype: torch.float32 y dtype: torch.int32
y = [1, 2]
```

注意：在模块边界集中转换，避免 OP 间反复 cast；`to(device)` 可能引入 H2D/D2H copy。

#### `.numpy()`

功能：CPU tensor 转 NumPy array。对 DEVICE tensor 调用需先 `.cpu()`，触发隐式同步 + D2H。  
用例：调试、离线分析。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([1, 2, 3], device="musa:0")
# 对 device tensor 转 numpy 前必须先拷贝到 CPU。
y = x.cpu().numpy()  # 先 D2H 到 CPU，再转 numpy

print("x device:", x.device, "y type:", type(y).__name__)
print("y =", y.tolist())
```

输入：MUSA tensor `[1,2,3]`。  
MUSA 运行结果（MUSA stdout）：
```text
x device: musa:0 y type: ndarray
y = [1, 2, 3]
```

注意：`.numpy()` 只能作用于 CPU tensor，对 DEVICE tensor 调用会报错——需先 `.cpu()`，而 `.cpu()` 自带隐式同步。延迟敏感路径避免用 DEVICE tensor 频繁 `.cpu().numpy()`。

#### `.tolist()`

功能：tensor 转 Python list。对 DEVICE tensor 调用触发 D2H 隐式同步。  
用例：CPU侧规划逻辑、输出 token 后处理、batch size 读取。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([1, 2, 3], device="musa:0")
# tolist 会生成 Python list；对 device tensor 会先发生 D2H 回读。
v = x.tolist()

print("v =", v, type(v))
```

输入：MUSA tensor `[1,2,3]`。  
MUSA 运行结果（MUSA stdout）：
```text
v = [1, 2, 3] <class 'list'>
```

注意：对 GPU tensor 调用会 D2H 同步。`seq_lens` 等调度元数据应维护 CPU侧副本，避免从 DEVICE tensor 频繁 `.tolist()`。

### GPU OP：设备端计算与内存行为

GPU OP 处理 DEVICE tensor 上的分配、layout、索引、数学计算、线性代数、路由和 dtype/device 转换。Transformer forward、KV cache、MoE 和 sampling 的主要计算都落在这一类。

**创建与初始化**

| OP | 用途 | 场景 | 注意 |
|----|------|------|------|
| `empty` | 预分配 input buffer、KV cache、logits、workspace | decode 固定 batch buffer、KV cache page、临时 GEMM workspace | 不初始化，后续 kernel 必须完整写入；不应替换预分配好的 tensor 对象 |
| `zeros` | mask 初始化、固定 shape 预分配 | attention mask、router mask、padding buffer | 语义上等价于分配后置零；后端通常以 memset 或 fill kernel 实现 |
| `ones` | attention mask additive、scale 初始化 | scale tensor、全 1 mask、测试输入构造 | 与 `full(..., 1)` 等价，语义明确 |
| `full` | padding token id、fill value、sentinel 值 | logits mask 填 `-inf`、padding token、invalid block 标记 | fill value 类型需与 dtype 兼容 |
| `new_empty` / `empty_like` | 按已有 tensor 的 dtype/device 创建临时 buffer | 按 hidden states 创建中间输出、按 logits 创建采样 buffer | 只继承属性，不继承内容 |

**原地更新**

| OP | 用途 | 场景 | 注意 |
|----|------|------|------|
| `copy_` | 固定 buffer 更新、metadata 写回 | graph replay 前写 input_ids、positions、seq_lens | 目标地址不变，inplace op 覆盖输入 |
| `_foreach_copy_` | 批量 buffer 更新 | 一次更新多个静态输入 buffer 或 metadata tensor | 减少 Python loop 和调度开销 |
| `fill_` | padding 槽位填充 | batch padding、block table 默认值、mask 默认值 | 原地写，覆盖原内容 |
| `zero_` | 清空 metadata、padding 清零 | 复用 logits buffer、清理 token mask、清空计数器 | 置零场景优先使用 `zero_` |
| `clamp_` / `masked_fill_` | 激活裁剪、attention/logits mask | logits 禁用 token、attention causal mask、router score 裁剪 | mask broadcast shape 必须正确 |

**Shape/Layout**

| OP | 用途 | 场景 | 注意 |
|----|------|------|------|
| `view` | head layout 整理，stride 兼容时零拷贝 | Q/K/V projection 后 `[B,S,H,Dh]` 重组 | stride 不兼容会报错 |
| `reshape` | 通用 shape 调整 | attention 输出恢复 `[B,S,D]`、MoE expert 维度规整 | stride 不兼容时静默 copy |
| `transpose / permute` | 交换维度 | attention 中 `[B,S,H,Dh]` 到 `[B,H,S,Dh]` | 输出通常非连续 |
| `contiguous` | 生成连续内存布局 | fused kernel 输入、transpose 后接 `view` 或 GEMM | 已连续则零拷贝，否则 D2D copy |
| `expand` | broadcast，zero-stride view | RoPE cos/sin 对齐 head 维、GQA repeat_kv | 不应用于原地写入 |
| `unsqueeze / squeeze` | 插入/删除 size=1 维度 | RoPE 补 head 维、mask 对齐 batch/head 维 | 零拷贝；squeeze 无参可能误删 batch 维 |
| `flatten` | 合并连续维度 | norm 前展平 hidden、batch/head 维合并 | stride 不兼容时可能复制 |

**索引映射**

| OP | 用途 | 场景 | 注意 |
|----|------|------|------|
| `gather` | MoE router 权重提取、top-k logits | 按 top-k index 取 expert score、按 token index 取 logits | index shape 与输入可广播 |
| `scatter_` | 构造 mask、写 selected block | token 到 expert slot 写入、block table 反向映射 | 重复 index 语义需明确（覆盖 vs 累加） |
| `index_select` | token dispatch、KV block 选择 | 选有效 token、选 KV page、按 request 顺序重排 | 输出 shape 跟 index 长度相关 |
| `index_add_` | MoE expert combine | expert 输出按 token index 累加回原序 | 重复 index 会累加 |

**序列组合**

| OP | 用途 | 场景 | 注意 |
|----|------|------|------|
| `arange` | position ids、request index | 生成 position_ids、batch request offset、block id | 半精度设备上注意 dtype |
| `cat` | RoPE/nope 拼接、local+compressed KV | DeepSeek MLA 中 q_nope/q_pe 拼接、KV 特征拼接 | 创建新 tensor，分配新内存 |
| `stack` | 新增维度组合 tensor | 多路统计值合并、多个小 tensor 组成 batch | 与 cat 不同：新增维度而非沿已有维 |
| `split / chunk` | gate/up 拆分、分组处理 | MLP gate/up 合并投影后拆分、QKV fused projection 拆分 | 返回 view，不复制数据 |
| `pad` | bucket padding、sequence padding | 请求对齐 bucket、固定 shape graph replay | 创建新 tensor |

**数学与激活**

| OP | 用途 | 场景 | 注意 |
|----|------|------|------|
| `sum / mean` | RMSNorm reduction、loss | RMSNorm hidden dim 归约、router 分数归一化 | `keepdim=True` 保持维度便于 broadcast |
| `square` | RMSNorm 平方和 | RMSNorm 中计算 hidden state 均方 | 等价于 `x*x`，语义明确 |
| `rsqrt` | RMSNorm reciprocal std | RMSNorm 计算 `1 / sqrt(variance + eps)` | `rsqrt(x) = 1/sqrt(x)` |
| `silu` | LLaMA/DeepSeek MLP 激活 | SwiGLU 中 gate 分支激活 | `x * sigmoid(x)`，fused kernel 可合并 gate projection |
| `softmax` | attention weight、router 分数 | attention score 归一化、MoE router top-k 前后处理 | `dtype=float32` 提升数值稳定但引入 cast |
| `gelu / relu` | BERT/经典 Transformer 激活 | BERT MLP、传统 FFN 激活 | 高频路径常融合 |

**线性代数与路由**

| OP | 用途 | 场景 | 注意 |
|----|------|------|------|
| `F.linear` | QKV/O projection、MLP、router、LM head | attention 投影、MLP gate/up/down、vocab projection | `x @ weight.T + bias`；dtype/layout/scale 决定 GEMM 路径 |
| `matmul` | attention score、value 聚合 | `Q @ K^T`、`softmax(QK^T) @ V` | batch/head 维必须对齐 |
| `bmm` | grouped projection、grouped expert | 多 batch 小矩阵乘、按 expert 分组后的 batched GEMM | batch dim 必须完全匹配 |
| `einsum` | 低秩 wo_a 投影 | MLA/低秩投影、分组专家权重组合 | 表达灵活，但后端未必走最优 kernel |
| `topk` | MoE routing、sampling | router 选 top-k expert、logits top-k sampling | 固定 k 有利于 shape 稳定 |
| `argmax` | greedy decode、routing 决策 | greedy token 选择、单 expert 路由 | ties 取第一个 |

**dtype/device**

| OP | 用途 | 场景 | 注意 |
|----|------|------|------|
| `float / half / bfloat16` | 精度转换 | RMSNorm 升精度、attention softmax 升精度、量化前后 cast | 触发 kernel，非零开销；模块边界集中转换 |
| `to(device)` | CPU↔DEVICE 移动 | metadata 上传、最终 logits/token 回读、离线调试 | H2D/D2H copy + 可能隐式同步 |

---

`F.linear` 和 `matmul` 是 Transformer 中最常见的计算 OP。下面用最小输入说明两者的计算规则。

#### `F.linear`

功能：执行 `y = x @ weight.T + bias`。  
用例：QKV/O projection、MLP gate/up/down、router、LM head。

```python
import torch
import torch.nn.functional as F
device = torch.device("musa:0")
x = torch.tensor([[1.0, 2.0]], device="musa:0")
weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], device="musa:0")
bias = torch.tensor([0.0, 0.0, 1.0], device="musa:0")
# F.linear(x, weight, bias) 等价于 x @ weight.T + bias。
y = F.linear(x, weight, bias)

print("x =", x.cpu().tolist())
print("y =", y.cpu().tolist())
```

输入：`x=[[1,2]]`，`weight=[[1,0],[0,1],[1,1]]`，`bias=[0,0,1]`。  
MUSA 运行结果（MUSA stdout）：
```text
x = [[1.0, 2.0]]
y = [[1.0, 2.0, 4.0]]
```

注意：量化模型还要检查 activation scale、weight scale 和 packed layout。

#### `matmul`

功能：矩阵乘，对最后两个维度做乘法。  
用例：attention score（`q @ k.T`）、value 聚合（`probs @ v`）。

```python
import torch
device = torch.device("musa:0")
q = torch.tensor([[[[1.0, 0.0, 2.0],
                    [0.0, 1.0, 1.0]]]], device="musa:0")  # [B=1,H=1,Sq=2,D=3]
k = torch.tensor([[[[1.0, 2.0, 0.0],
                    [0.0, 1.0, 3.0]]]], device="musa:0")  # [B=1,H=1,Sk=2,D=3]

# attention score = Q @ K^T / sqrt(D)
# 未缩放点积：
# q0·k0 = 1*1 + 0*2 + 2*0 = 1
# q0·k1 = 1*0 + 0*1 + 2*3 = 6
# q1·k0 = 0*1 + 1*2 + 1*0 = 2
# q1·k1 = 0*0 + 1*1 + 1*3 = 4
scale = q.shape[-1] ** -0.5
scores = torch.matmul(q, k.transpose(-2, -1)) * scale

print("scores shape:", tuple(scores.shape))
print("scores =", [[round(v, 4) for v in row] for row in scores[0, 0].cpu().tolist()])
```

输入：Q `[1,1,2,3]`，K `[1,1,2,3]`。`k.transpose(-2, -1)` 后 shape 为 `[1,1,3,2]`，`matmul` 输出 attention score shape `[1,1,2,2]`。  
MUSA 运行结果（MUSA stdout）：
```text
scores shape: (1, 1, 2, 2)
scores = [[0.5774, 3.4641], [1.1547, 2.3094]]
```

注意：朴素 attention 会显式生成 score tensor；推理路径通常使用 FlashAttention 或 fused kernel。

### CPU OP：调度与元数据

CPU OP 负责调度和状态管理，不直接处理大张量计算。在线推理中，CPU侧负责 request queue、prefix cache、bucket 选择、KV block 管理、metadata 构造和协议输出；DEVICE侧负责 attention、MLP/MoE、logits 和 sampling 的张量计算。

| 类别 | 代表 OP | 场景 | 注意 |
|------|---------|------|------|
| Python 容器 | `list`、`dict`、`len`、`range` | scheduler、request state、block table | 不应驱动 decode 单步里的 DEVICE tensor 分支 |
| CPU tensor | `torch.tensor(..., device="cpu")` | `seq_lens_cpu`、batch size | CPU侧副本需与 DEVICE metadata 同步 |
| CPU-DEVICE 转换 | `.cpu()`、`.numpy()`、`.tolist()` | 日志、调试、最终 token | 对 GPU tensor 触发隐式同步 + D2H |
| 标量读取 | `.item()` | 最终 token id、loss scalar | 高频调用把异步执行变成 CPU 等待 |
| H2D metadata | `to(device)`、`torch.as_tensor(...)` | seq_lens、positions 上传 | 复用固定 buffer，避免零散小 tensor |

## 附录 A：Llama Decoder 源码分析

decoder-only LLM 由多层相同 block 堆叠而成。本附录以 HuggingFace Transformers 中 Llama 的源码为例，提取 forward 主路径并按执行链路拆解。

参考源码：

```text
transformers/src/transformers/models/llama/modeling_llama.py
```

### A.1 RMSNorm

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

朴素实现包含 4 个独立的逐元素/归约 kernel（pow → mean → rsqrt → mul）。在线热点路径通常使用 fused RMSNorm kernel 一次完成。

### A.2 RoPE（旋转位置编码）

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

OP 组成：频率生成使用 `matmul` + `transpose` + `cat` + `cos` + `sin`；旋转使用 `unsqueeze` + `mul` + `add`。cos/sin 在 float32 下计算后 cast 回模型 dtype。

### A.3 Attention

`LlamaAttention` 是标准的 Multi-Head Attention，支持 GQA（Query 头数 > KV 头数）。

源码位置 `modeling_llama.py:251-289`，forward 主路径：

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

OP 组成：

| 阶段 | OP | shape 变化 | 行为 |
|------|-----|-----------|------|
| Q/K/V 投影 | `F.linear`(×3) | `[B,S,D]` → `[B,S,H*Dh]` | GEMM |
| head layout | `view` + `transpose` | `[B,S,H,Dh]` → `[B,H,S,Dh]` | 零拷贝 |
| RoPE | `unsqueeze` + `mul` + `add` | — | elementwise |
| KV cache | cache.update | 追加 | copy |
| attention | matmul / SDPA | `[B,H,S,Dh]` → `[B,H,S,Dh]` | GEMM + softmax |
| output layout | `reshape` + `contiguous` | `[B,H,S,Dh]` → `[B,S,D]` | contiguous 可能 D2D copy |
| output proj | `F.linear` | `[B,S,D]` → `[B,S,D]` | GEMM |

#### GQA：repeat_kv

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

#### 参考实现：eager_attention_forward

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

softmax 在 float32 下计算后 cast 回原始 dtype，用于降低归约误差。eager 路径会 materialize 完整的 `[B,H,S,S]` score tensor，长序列下显存和带宽开销很高。推理路径通常使用 SDPA 或 FlashAttention。

### A.4 MLP（SwiGLU）

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

OP 组成：`F.linear`(×3) + `silu` + `mul`。生产实现通常将 gate_proj 和 up_proj 合并为一次 `F.linear`，再通过 `chunk` 拆分，减少 GEMM launch 数。

### A.5 Decoder Layer

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

### A.6 模型入口

#### LlamaModel.forward

源码位置 `modeling_llama.py:375-425`：

```python
def forward(self, input_ids, attention_mask=None, position_ids=None,
            past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
    # token id → embedding
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)             # [B, S] → [B, S, D]
    hidden_states = inputs_embeds

    # 生成 position_ids
    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)

    # causal mask
    causal_mask = create_causal_mask(attention_mask, inputs_embeds, position_ids, past_key_values)

    # RoPE cos/sin 在模型层统一生成，传入每一层
    position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

    # N 层 decoder layer
    for decoder_layer in self.layers:
        hidden_states = decoder_layer(hidden_states, attention_mask=causal_mask,
                                       position_embeddings=position_embeddings,
                                       past_key_values=past_key_values, use_cache=use_cache,
                                       **kwargs)

    # final RMSNorm
    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)
```

#### LlamaForCausalLM.forward

源码位置 `modeling_llama.py:445-499`：

```python
def forward(self, input_ids=None, attention_mask=None, labels=None, logits_to_keep=0, **kwargs):
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)  # LlamaModel
    hidden_states = outputs.last_hidden_state                   # [B, S, D]

    # decode 通常只需要最后 token 的 logits
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])  # [B, S, D] → [B, S, V]
    return CausalLMOutputWithPast(logits=logits, past_key_values=outputs.past_key_values)
```

`logits_to_keep` 优化：decode 时只需要最后一个 token 的 logits，`logits_to_keep=1` 只对最后一个位置做 lm_head projection，减少 vocab 维度（V 通常 32k-128k）的 GEMM 计算量。

### A.7 执行链路

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

### A.8 总结：Llama 的 OP 全景

| 模块 | 主要 OP | 类型 |
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
