# PyTorch OP 分类与行为

根据执行的设备类型，PyTorch OP 分为 GPU OP 和 CPU OP。此外还有几类行为特殊的 OP——view op、inplace op、dynamic shape op 和 sync op——它们不按设备分类，但各自有独立的底层行为和工程约束。

下文按分类展开，每个算子附可运行用例和 MUSA 执行输出。

## 1.1 view op

`view` / `reshape` / `unsqueeze` / `squeeze` / `transpose` / `permute` / `expand` 等 OP 只改变 tensor 的 shape 和 stride 描述，不移动底层数据。**零拷贝是默认行为，但不是保证**。

`view` 仅在 stride 兼容时成功，否则直接报错。例如 `x.transpose(1, 2).view(...)` 通常失败，因为 transpose 改变了内存排布。`reshape` 更宽容——stride 兼容时返回 view，不兼容时静默复制数据。这正是 `reshape` 可能成为性能陷阱的原因：代码看起来只是一行 shape 调整，实际可能触发一次 D2D copy。

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

源码中 `linear -> view -> transpose` 是 attention head layout 的常见路径：`LlamaAttention.forward` 和 `DeepseekV4Attention.forward` 都使用 `view(...).transpose(1, 2)`。transpose 后的 tensor 通常非连续，后续 kernel 若要求连续输入需要显式处理。

---

##### `contiguous`

功能：返回内存连续 tensor；已连续则返回自身（零拷贝），否则复制为新连续内存。  
用例：MUSA/TileLang/fused kernel 前满足内存布局要求。

```python
import torch
device = torch.device("musa:0")
x = torch.arange(6, device="musa:0").view(2, 3).t()
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

##### `reshape`

功能：改变 shape，必要时复制。  
用例：量化分组、fallback reference 中规整 `[T,G,D]` 等逻辑维度。

```python
import torch
device = torch.device("musa:0")
x = torch.arange(6, device="musa:0")
y = x.reshape(3, 2)

print("x =", x.cpu().tolist())
print("y =", y.cpu().tolist())
```

输入：`x=[0,1,2,3,4,5]`。  
MUSA 运行结果（MUSA stdout）：
```text
x = [0, 1, 2, 3, 4, 5]
y = [[0, 1], [2, 3], [4, 5]]
```

注意：延迟敏感路径避免用 `reshape` 掩盖隐式 copy；需要固定地址时显式处理 layout。能用 `view` 时优先 `view`。

##### `expand`

功能：通过 zero-stride 扩展维度，不复制数据。  
用例：RoPE 频率对齐、broadcast scale/mask。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([[1.0, 2.0]], device="musa:0")
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

##### `view`

功能：返回 tensor 的 view，共享底层数据；stride 不兼容时报错。  
用例：线性层输出 shape 重组。

```python
import torch
device = torch.device("musa:0")
x = torch.arange(6, device="musa:0")
y = x.view(2, 3)
print("y =", y.cpu().tolist())
```

输入：`x=[0,1,2,3,4,5]`。  
MUSA 运行结果（MUSA stdout）：
```text
y = [[0, 1, 2], [3, 4, 5]]
```

注意：stride 不兼容时直接报错而非静默复制。无法满足 shape 需求时用 `reshape` 或 `contiguous()` + `view`。

##### `unsqueeze`

功能：在指定位置插入 size=1 的新维度。  
用例：broadcast 前对齐维度。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([1.0, 2.0, 3.0], device="musa:0")
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

##### `squeeze`

功能：移除所有 size=1 的维度，或指定特定 dim 移除。  
用例：移除 batch 或 head 维度。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([[[1.0], [2.0], [3.0]]], device="musa:0")
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

##### `transpose`

功能：交换两个维度，只改 metadata 不复制数据。  
用例：attention 中 `seq_len` 和 `head_dim` 的 layout 调整。

```python
import torch
device = torch.device("musa:0")
x = torch.arange(6, device="musa:0").view(2, 3)
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

##### `flatten`

功能：将指定 dim 范围展平为单个维度。  
用例：多头注意力计算前融合 head 维和 batch 维。

```python
import torch
device = torch.device("musa:0")
x = torch.arange(24, dtype=torch.float32, device="musa:0").reshape(2, 3, 4)
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

## 1.2 inplace op

`copy_` / `fill_` / `zero_` / `clamp_` / `masked_fill_` / `_foreach_copy_` 等以 `_` 结尾的 OP 直接修改输入 tensor 的内容，不分配新内存。**核心价值是保持 tensor 对象地址不变**——在高频推理路径中，复用固定 buffer 比反复分配更高效。

预分配固定 buffer，后续通过 `copy_` 把本轮真实数据写入这些 buffer。如果创建新 tensor 替换旧对象，之前绑定的地址就会失效。

典型用法：`input_ids.copy_(real_ids)`、`seq_lens.fill_(1)`、`out_cache_loc.zero_()`。padding 槽位必须清理，否则上轮残留值会污染 attention mask 或 logits。

---

##### `copy_`

功能：把源 tensor 内容复制到目标 tensor，目标对象地址不变。  
用例：固定 buffer 更新、metadata replay 中原地更新 tensor 字段。

```python
import torch
device = torch.device("musa:0")
dst = torch.tensor([0, 0, 0], device="musa:0")
src = torch.tensor([1, 2, 3], device="musa:0")
dst.copy_(src)

print("dst =", dst.cpu().tolist())
print("src =", src.cpu().tolist())
```

输入：`dst=[0,0,0]`，`src=[1,2,3]`。  
MUSA 运行结果（MUSA stdout）：
```text
dst = [1, 2, 3]
src = [1, 2, 3]
```

注意：推理 `no_grad` 路径更适合原地更新；`torch._foreach_copy_` 可批量执行多个 `copy_`，减少 Python 循环和调度开销。

##### `fill_`

功能：原地填充指定标量。  
用例：复用固定 buffer 前写默认值、填 padding 槽位。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([0, 0, 0], device="musa:0")
x.fill_(3)

print("x =", x.cpu().tolist())
```

输入：`x=[0,0,0]`。  
MUSA 运行结果（MUSA stdout）：
```text
x = [3, 3, 3]
```

##### `masked_fill_`

功能：按 bool mask 原地填充值。  
用例：attention mask 处理、logits mask、topk padding id 置 sentinel。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([1, 2, 3, 4], device="musa:0")
mask = torch.tensor([True, False, True, False], device="musa:0")
x.masked_fill_(mask, -1)

print("x =", x.cpu().tolist())
```

输入：`x=[1,2,3,4]`，`mask=[True,False,True,False]`。  
MUSA 运行结果（MUSA stdout）：
```text
x = [-1, 2, -1, 4]
```

注意：推理 `no_grad` 路径更适合原地更新；mask broadcast shape 必须正确。

##### `zero_`

功能：原地将 tensor 所有元素置零。  
用例：attention mask、KV cache padding 槽位清理。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([3, 3, 3], device="musa:0")
x.zero_()
print("x =", x.cpu().tolist())
```

输入：`x=[3,3,3]`。  
MUSA 运行结果（MUSA stdout）：
```text
x = [0, 0, 0]
```

注意：复用预分配 buffer 前用 `zero_()` 或 `fill_()` 清理残留值。

##### `clamp_`

功能：原地将元素裁切到 `[min, max]` 范围。  
用例：logits 数值稳定、MoE routing score 裁切。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([1.0, -2.0, 20.0], device="musa:0")
x.clamp_(min=0, max=10)
print("x =", x.cpu().tolist())
```

输入：`x=[1,-2,20]`。  
MUSA 运行结果（MUSA stdout）：
```text
x = [1.0, 0.0, 10.0]
```

注意：clamp 的边界可以是 scalar 或 tensor；替换 `min`/`max` 之一即可做单侧裁切。

##### `_foreach_copy_`

功能：批量执行多个 `copy_`，减少 Python 循环开销。  
用例：批量更新多个预分配 buffer。

```python
import torch
device = torch.device("musa:0")
d0 = torch.zeros(2, device="musa:0")
d1 = torch.zeros(2, device="musa:0")
src0 = torch.tensor([1, 1], device="musa:0")
src1 = torch.tensor([2, 2], device="musa:0")
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

注意：`_foreach_copy_` 是 PyTorch foreach 机制的一部分，kernel launch 合并比 Python 循环逐个 `copy_` 高效。下划线前缀表示内部 API，无 stable 接口保证。

## 1.3 dynamic shape op

`nonzero` / `unique` / `masked_select` / boolean indexing 的输出长度依赖输入数据的实际值，而不仅仅是 shape。例如 `x[x > 0]` 返回多少个元素，取决于运行时 x 中有多少正数。这一特性在 eager 模式下很自然，但与固定 shape 的推理流水线直接冲突——输出 shape 的动态变化会让预分配的 buffer 和 workspace 失效。

---

##### `nonzero`

功能：返回非零元素坐标，输出 shape 为 `[N, ndim]`，N 随数据变化。  
用例：找有效 token、稀疏 mask 调试。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([[0, 1, 0], [2, 0, 3]], device="musa:0")
idx = torch.nonzero(x)

print("idx =", idx.cpu().tolist())
```

输入：`x=[[0,1,0],[2,0,3]]`。  
MUSA 运行结果（MUSA stdout）：
```text
idx = [[0, 1], [1, 0], [1, 2]]
```

注意：输出动态长度，不应用于固定 shape 流水线。替代：fixed mask + padding 保持 shape 不变。

##### `unique`

功能：返回唯一值及其索引，输出长度等于唯一值个数。  
用例：expert/block/request 分布统计。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([1, 3, 2, 3, 1, 2], device="musa:0")
u = torch.unique(x)

print("u =", u.cpu().tolist())
```

输入：`x=[1,3,2,3,1,2]`。  
MUSA 运行结果（MUSA stdout）：
```text
u = [1, 2, 3]
```

注意：MUSA 实现会触发 stream 同步回读输出长度。在线路径优先用 `bincount(minlength=N)` 做固定容量直方图。

##### `masked_select`

功能：按 mask 选取元素，返回一维 tensor，长度等于 mask 中 `True` 的个数。  
用例：抽取有效 token/logits。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([1, 2, 3, 4], device="musa:0")
mask = torch.tensor([True, False, True, False], device="musa:0")
y = torch.masked_select(x, mask)

print("y =", y.cpu().tolist())
```

输入：`x=[1,2,3,4]`，`mask=[True,False,True,False]`。  
MUSA 运行结果（MUSA stdout）：
```text
y = [1, 3]
```

注意：输出动态长度。延迟敏感路径用 `where(condition, x, fill)` 保持原 shape。

##### `where`

功能：双参数 `where(cond, x, y)` 按条件逐元素选择（固定 shape）；单参数 `where(mask)` 返回坐标（动态长度）。  
用例：MoE token 查找、mask 条件选择。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([1, 2, 3], device="musa:0")
y = torch.where(x > 1, x, torch.zeros_like(x))

print("y =", y.cpu().tolist())
```

输入：`x=[1,2,3]`。  
MUSA 运行结果（MUSA stdout）：
```text
y = [0, 2, 3]
```

##### `topk`

功能：沿指定 dim 选择最大 k 个值。  
用例：MoE routing、CSA indexer top-k 选择、sampling。

```python
import torch
device = torch.device("musa:0")
scores = torch.tensor([[0.1, 0.7, 0.2], [0.6, 0.1, 0.3]], device="musa:0")
values, indices = torch.topk(scores, 2, dim=-1)
weights = values / values.sum(-1, keepdim=True)

print("indices =", indices.cpu().tolist())
print("weights =", weights.cpu().tolist())
```

输入：`scores=[[0.1,0.7,0.2],[0.6,0.1,0.3]]`。  
MUSA 运行结果（MUSA stdout）：
```text
indices = [[1, 2], [0, 2]]
weights = [[0.7778, 0.2222], [0.6667, 0.3333]]
```

注意：固定 k 有利于后续 shape 稳定；top-k shape 和排序稳定性会传导到后续 kernel。

##### `gather`

功能：根据索引从指定 dim 收集元素。  
用例：embedding lookup、token mapping。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([[10, 11, 12], [20, 21, 22]], device="musa:0")
idx = torch.tensor([[2, 0], [1, 1]], device="musa:0")
y = torch.gather(x, 1, idx)
print("y =", y.cpu().tolist())
```

输入：`x=[[10,11,12],[20,21,22]]`，`idx=[[2,0],[1,1]]`。  
MUSA 运行结果（MUSA stdout）：
```text
y = [[12, 10], [21, 21]]
```

注意：输出 shape 与 index tensor 一致。index 越界可能导致未定义行为或报错。

##### `scatter_`

功能：根据索引将 src 的值写入 tensor（原地）。  
用例：logits 去乱序、反向 lookup 映射。

```python
import torch
device = torch.device("musa:0")
x = torch.zeros(3, 5, device="musa:0")
idx = torch.tensor([[0, 1], [1, 2]], device="musa:0")
x.scatter_(1, idx, 1)
print("x =", x.cpu().tolist())
```

输入：`x` shape `(3,5)` zeros，`idx=[[0,1],[1,2]]`。  
MUSA 运行结果（MUSA stdout）：
```text
x = [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0]]
```

注意：index 重复时行为取决于 `reduce` 参数（`'sum'` 累加 vs 默认覆盖）。scatter 的 index 越界会报错。

##### `index_select`

功能：沿指定 dim 按 1D index 选取切片。  
用例：稀疏 token 选取、KV block 选取。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([[1, 2], [3, 4], [5, 6]], device="musa:0")
idx = torch.tensor([0, 2], device="musa:0")
y = torch.index_select(x, 0, idx)
print("y =", y.cpu().tolist())
```

输入：`x=[[1,2],[3,4],[5,6]]`，`idx=[0,2]`。  
MUSA 运行结果（MUSA stdout）：
```text
y = [[1, 2], [5, 6]]
```

注意：`index_select` 输出 shape 固定（取决于 index 长度，不依赖数据值），比 bool masking 更适合固定 shape 流水线。

##### `index_add_`

功能：按 index 将 src 累加到目标 tensor（原地）。  
用例：MoE token 回写、梯度累积。

```python
import torch
device = torch.device("musa:0")
x = torch.zeros(2, 2, device="musa:0")
src = torch.tensor([[1, 0], [0, 2], [3, 4]], device="musa:0")
idx = torch.tensor([0, 1, 0], device="musa:0")
x.index_add_(0, idx, src)
print("x =", x.cpu().tolist())
```

输入：`x=[[0,0],[0,0]]`，`src=[[1,0],[0,2],[3,4]]`，`idx=[0,1,0]`。  
MUSA 运行结果（MUSA stdout）：
```text
x = [[4, 4], [0, 2]]
```

注意：index 相同的行会累加（不是覆盖）。与 `scatter_` 不同，`index_add_` 是累加语义。index 越界会报错。

## 1.4 sync op

sync op 控制 CPU 与 DEVICE 之间的时序。分为显式和隐式两类。

**显式同步**：`synchronize()`、`Stream`、`Event.record()`、`wait_event()`。`synchronize()` 阻塞 CPU 直到 DEVICE 上所有已提交操作完成，适用于 benchmark 计时和错误定位，但放入 decode 热点路径会打断 CPU/DEVICE 异步流水。局部依赖应使用 stream/event 管理，精确控制等待范围。

**隐式同步**：`.item()`、`.tolist()`、`.cpu()`、`.numpy()`。这些 API 看起来只是取值或类型转换，但作用在 DEVICE tensor 上时，CPU 必须先等待 GPU 完成所有前序操作，再发起 D2H 拷贝。

工程原则：CPU侧维护一份元数据副本（`seq_lens_cpu`），避免从 DEVICE tensor 读取元数据驱动 Python 分支。最终 token 和少量 logprob 允许回读；完整 logits、hidden states、KV metadata 不应频繁回 CPU。

---

##### `torch.musa.synchronize`

功能：阻塞 CPU 直到 MUSA 设备上所有 stream 操作完成。  
用例：benchmark 计时、错误定位、程序退出前等待。

```python
import torch
device = torch.device("musa:0")
x = torch.randn(1000, device="musa:0")
y = x * 2
torch.musa.synchronize()  # 等待 y 的计算完成

print("y mean =", y.mean().cpu().item())
```

输入：`x` 为 1000 个随机数。  
注意：**不要放入 decode 热点路径**——这会打断 CPU/DEVICE 并行流水。benchmark 或调试时才显式同步。

##### `.item()`

功能：将 DEVICE tensor 的标量值读回 CPU。  
用例：获取最终 token id、loss scalar。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([3.14], device="musa:0")
val = x.item()

print("val =", val)
```

输入：`x=[3.14]`。  
MUSA 运行结果（MUSA stdout）：
```text
val = 3.140000104904175
```

注意：底层触发 D2H copy + stream 同步。高频调用（如 per-token 回读）会把异步执行变成 CPU 等待。

##### `.cpu()`

功能：将 DEVICE tensor 复制到 CPU。  
用例：最终输出、少量 logprob、离线分析。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([1, 2, 3], device="musa:0")
y = x.cpu()

print("y =", y.tolist())
```

输入：`x=[1,2,3]`。  
注意：触发 D2D copy + 隐式同步。仅用于最终结果或少量统计，不应频繁拉回完整 logits、hidden states 或 KV metadata。

##### `to(dtype)`

功能：精度转换，可能触发 kernel 和数据复制。  
用例：Norm 内部升精度、量化前后 cast。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([1.5, 2.7], device="musa:0")
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

##### `.numpy()`

功能：CPU tensor 转 NumPy array。对 DEVICE tensor 调用需先 `.cpu()`，触发隐式同步 + D2H。  
用例：调试、离线分析。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([1, 2, 3], device="musa:0")
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

##### `.tolist()`

功能：tensor 转 Python list。对 DEVICE tensor 调用触发 D2H 隐式同步。  
用例：CPU侧规划逻辑、输出 token 后处理、batch size 读取。

```python
import torch
device = torch.device("musa:0")
x = torch.tensor([1, 2, 3], device="musa:0")
v = x.tolist()

print("v =", v, type(v))
```

输入：MUSA tensor `[1,2,3]`。  
MUSA 运行结果（MUSA stdout）：
```text
v = [1, 2, 3] <class 'list'>
```

注意：对 GPU tensor 调用会 D2H 同步。`seq_lens` 等调度元数据应维护 CPU侧副本，避免从 DEVICE tensor 频繁 `.tolist()`。

## 1.5 GPU OP

GPU OP 作用于 DEVICE tensor，覆盖张量创建、layout 整理、索引映射、数学计算、线性代数、路由和 dtype/device 转换，是 Transformer forward、KV cache、MoE、sampling 的主体。

**创建与初始化**

| OP | 用途 | 注意 |
|----|------|------|
| `empty` | 预分配 input buffer、KV cache、logits、workspace | 不初始化，后续 kernel 必须完整写入；不要替换预分配好的 tensor 对象 |
| `zeros` | mask 初始化、固定 shape 预分配 | 等价于 `empty().zero_()`，eager 模式下单次 launch |
| `ones` | attention mask additive、scale 初始化 | 与 `full(..., 1)` 等价，语义更清晰 |
| `full` | padding token id、fill value、sentinel 值 | fill value 类型需与 dtype 兼容 |
| `new_empty` / `empty_like` | 按已有 tensor 的 dtype/device 创建临时 buffer | 只继承属性，不继承内容 |

**原地更新**

| OP | 用途 | 注意 |
|----|------|------|
| `copy_` | 固定 buffer 更新、metadata 写回 | 目标地址不变，inplace op 覆盖输入 |
| `_foreach_copy_` | 批量 buffer 更新 | 减少 Python loop 和调度开销 |
| `fill_` | padding 槽位填充 | 原地写，覆盖原内容 |
| `zero_` | 清空 metadata、padding 清零 | `zero_` 比 `fill_(0)` 更高效 |
| `clamp_` / `masked_fill_` | 激活裁剪、attention/logits mask | mask broadcast shape 必须正确 |

**Shape/Layout**

| OP | 用途 | 注意 |
|----|------|------|
| `view` | head layout 整理，stride 兼容时零拷贝 | stride 不兼容会报错 |
| `reshape` | 通用 shape 调整 | stride 不兼容时静默 copy |
| `transpose / permute` | 交换维度 | 输出通常非连续 |
| `contiguous` | 生成连续内存布局 | 已连续则零拷贝，否则 D2D copy |
| `expand` | broadcast，zero-stride view | 不应用于原地写入 |
| `unsqueeze / squeeze` | 插入/删除 size=1 维度 | 零拷贝；squeeze 无参可能误删 batch 维 |
| `flatten` | 合并连续维度 | HC fallback norm 前展平 hidden |

**索引映射**

| OP | 用途 | 注意 |
|----|------|------|
| `gather` | MoE router 权重提取、top-k logits | index shape 与输入可广播 |
| `scatter_` | 构造 mask、写 selected block | 重复 index 语义需明确（覆盖 vs 累加） |
| `index_select` | token dispatch、KV block 选择 | 输出 shape 跟 index 长度相关 |
| `index_add_` | MoE expert combine | 重复 index 会累加 |

**序列组合**

| OP | 用途 | 注意 |
|----|------|------|
| `arange` | position ids、request index | 半精度设备上注意 dtype |
| `cat` | RoPE/nope 拼接、local+compressed KV | 创建新 tensor，分配新内存 |
| `stack` | 新增维度组合 tensor | 与 cat 不同：新增维度而非沿已有维 |
| `split / chunk` | gate/up 拆分、分组处理 | 返回 view，不复制数据 |
| `pad` | bucket padding、sequence padding | 创建新 tensor |

**数学与激活**

| OP | 用途 | 注意 |
|----|------|------|
| `sum / mean` | RMSNorm reduction、loss | `keepdim=True` 保持维度便于 broadcast |
| `square` | RMSNorm 平方和 | 等价于 `x*x`，语义更清晰 |
| `rsqrt` | RMSNorm reciprocal std | `rsqrt(x) = 1/sqrt(x)` |
| `silu` | LLaMA/DeepSeek MLP 激活 | `x * sigmoid(x)`，fused kernel 可合并 gate projection |
| `softmax` | attention weight、router 分数 | `dtype=float32` 提升数值稳定但引入 cast |
| `gelu / relu` | BERT/经典 Transformer 激活 | 高频路径常融合 |

**线性代数与路由**

| OP | 用途 | 注意 |
|----|------|------|
| `F.linear` | QKV/O projection、MLP、router、LM head | `x @ weight.T + bias`；dtype/layout/scale 决定 GEMM 路径 |
| `matmul` | attention score、value 聚合 | batch/head 维必须对齐 |
| `bmm` | grouped projection、grouped expert | batch dim 必须完全匹配 |
| `einsum` | 低秩 wo_a 投影 | `"bsgd,grd->bsgr"` 灵活表达 |
| `topk` | MoE routing、sampling | 固定 k 有利于 shape 稳定 |
| `argmax` | greedy decode、routing 决策 | ties 取第一个 |

**dtype/device**

| OP | 用途 | 注意 |
|----|------|------|
| `float / half / bfloat16` | 精度转换 | 触发 kernel，非零开销；模块边界集中转换 |
| `to(device)` | CPU↔DEVICE 移动 | H2D/D2H copy + 可能隐式同步 |

---

`F.linear` 和 `matmul` 是 Transformer 中最核心的两个计算 OP，下面给出完整用例：

##### `F.linear`

功能：执行 `y = x @ weight.T + bias`。  
用例：QKV/O projection、MLP gate/up/down、router、LM head。

```python
import torch
import torch.nn.functional as F
device = torch.device("musa:0")
x = torch.tensor([[1.0, 2.0]], device="musa:0")
weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], device="musa:0")
bias = torch.tensor([0.0, 0.0, 1.0], device="musa:0")
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

##### `matmul`

功能：矩阵乘，对最后两个维度做乘法。  
用例：attention score（`q @ k.T`）、value 聚合（`probs @ v`）。

```python
import torch
device = torch.device("musa:0")
q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], device="musa:0")
k = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]], device="musa:0")
scores = torch.matmul(q, k.transpose(2, 3)) * (2 ** -0.5)

print("scores =", scores.cpu().tolist())
```

输入：Q `[1,1,2,2]`，K `[1,1,2,2]`。  
MUSA 运行结果（MUSA stdout）：
```text
scores = [[[[0.7071, 0.7071], [0.0, 0.7071]]]]
```

注意：reference attention 中 score tensor 可能较大；实际推理用 FlashAttention 或 fused kernel。

## 1.6 CPU OP

CPU OP 服务调度和状态管理，不直接承担大张量计算。在线推理中，CPU侧负责 request queue、prefix cache、bucket 选择、KV block 管理、metadata 构造和协议输出；DEVICE侧负责 attention、MLP/MoE、logits 和 sampling 的张量计算。

| 类别 | 典型 OP | 场景 | 注意 |
|------|---------|------|------|
| Python 容器 | `list`、`dict`、`len`、`range` | scheduler、request state、block table | 不要驱动 decode 单步里的 DEVICE tensor 分支 |
| CPU tensor | `torch.tensor(..., device="cpu")` | `seq_lens_cpu`、batch size | CPU侧副本需与 DEVICE metadata 同步 |
| CPU-DEVICE 转换 | `.cpu()`、`.numpy()`、`.tolist()` | 日志、调试、最终 token | 对 GPU tensor 触发隐式同步 + D2H |
| 标量读取 | `.item()` | 最终 token id、loss scalar | 高频调用把异步执行变成 CPU 等待 |
| H2D metadata | `to(device)`、`torch.as_tensor(...)` | seq_lens、positions 上传 | 复用固定 buffer，避免零散小 tensor |

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
