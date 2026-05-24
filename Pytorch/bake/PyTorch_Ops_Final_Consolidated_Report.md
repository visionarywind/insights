# PyTorch Op 完整技术洞察与 MUSA 验证报告

**验证环境**: `shanfeng@10.18.32.25` / Docker `mochi-sglang` / PyTorch 2.9.0 / MUSA 8 GPU / SGLang 0.5.6
**覆盖框架**: SGLang DeepSeek V4 / vLLM / Megatron-LM
**验证日期**: 2026-05-21 | **17/17 用例 PASS**

---

## 目录

1. [MUSA 执行验证结果](#1-musa-执行验证结果)
2. [Op 分类详细分析](#2-op-分类详细分析)
3. [SGLang DeepSeek V4 实战](#3-sglang-deepseek-v4-实战)
4. [CUDA/MUSA Graph 分析](#4-cudamusa-graph-分析)
5. [三框架 Op 策略对比](#5-三框架-op-策略对比)
6. [MUSA 适配差异](#6-musa-适配差异)
7. [性能洞察与优化原则](#7-性能洞察与优化原则)

---

## 1. MUSA 执行验证结果

MUSA (Moore Threads GPU) 设备上 PyTorch 2.9.0 的 17 个 op 类别全部通过验证：

| # | 测试 | 关键 Op | 结果 |
|---|------|---------|------|
| 1a | Tensor 创建 | `empty`, `new_empty`, `empty_like` | ✅ |
| 1b | 初始化分配 | `zeros`, `ones`, `full` | ✅ |
| 2a | 原地复制 | `copy_` | ✅ |
| 2b | 原地填充 | `fill_`, `zero_`, `masked_fill_` | ✅ |
| 2c | 批量复制 | `torch._foreach_copy_` | ✅ |
| 3a | 形状变换 | `view`, `reshape`, `flatten`, `unsqueeze`, `squeeze` | ✅ |
| 3b | 内存布局 | `contiguous`, `stride`, `is_contiguous` | ✅ |
| 4a | 索引切分 | slice, advanced indexing, `gather`, `scatter_`, `index_select`, `tensor_split` | ✅ |
| 4b | 条件选择 | `torch.where` | ✅ |
| 5a | 序列构造 | `arange`, `repeat_interleave`, `expand` | ✅ |
| 6a | 拼接填充 | `F.pad`, `cat`, `stack` | ✅ |
| 7a | 数学归约 | `sum`, `mean`, `amax`, `abs`, `square`, `rsqrt` | ✅ |
| 8a | 激活函数 | `sigmoid`, `silu`, `gelu`, `relu`, `softmax`, `clamp` | ✅ |
| 9a | 线性代数 | `F.linear`, `matmul`, `mm`, `bmm`, `einsum` | ✅ |
| 10a | 路由排序 | `topk`, `sort`, `argsort`, `argmax` | ✅ |
| 11a | 精度转换 | `.to`, `.float`, `.bfloat16`, `.cpu`, `.item`, `.tolist` | ✅ |
| 12a | Graph | `MUSAGraph` capture/replay | ✅ |

---

## 2. Op 分类详细分析

### 2.1 Tensor 创建

**功能**: 在 GPU 上分配 tensor 内存。

**用法**:

```python
out = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device="musa:0")
buf = torch.zeros((batch, seqlen), dtype=torch.int32, device="musa:0")
mask = torch.ones((batch,), dtype=torch.bool, device="musa:0")
fill = torch.full((2,3), 7, device="musa:0")
inherited = x.new_empty((4,))           # 继承 dtype/device
clone_cfg = torch.empty_like(existing)  # 继承 shape/dtype/device
```

**MUSA 实际执行**:

```
输入: torch.empty((2,3), float32)
输出: shape=(2,3) dtype=float32[未初始化]
      [[3.0,3.0,3.0],[3.0,3.0,3.0]]  # empty 不保证值

输入: torch.zeros((3,2), int64)
输出: shape=(3,2) dtype=int64
      [[0,0],[0,0],[0,0]]

输入: torch.ones((3,2), int64)
输出: shape=(3,2) dtype=int64
      [[1,1],[1,1],[1,1]]

输入: torch.full((3,2), 7)
输出: shape=(3,2) dtype=int64
      [[7,7],[7,7],[7,7]]
```

**框架使用**:
- SGLang: `q_padded = x.new_empty(x.shape[0], n_heads, head_dim)` — 预分配 MQA Q buffer
- vLLM: PagedAttention KV cache block 预分配
- Megatron: TP 通信 buffer 预分配

**注意事项**:
- `empty` 内容为未定义值，后续 kernel 必须完整写入
- CUDA Graph capture 记录 `empty` 返回的地址；replay 不能换对象
- `full(value=tensor.item())` → GPU 标量化，引入同步

---

### 2.2 复制与原地更新

**功能**: 原地修改 tensor 内容（地址不变），CUDA Graph 的核心依赖。

**MUSA 实际执行**:

```
[copy_] — CUDA Graph replay 核心
输入: a=zeros(2,3), b=ones(2,3)
    a.copy_(b)
输出: a = [[1,1,1],[1,1,1]]  (地址不变)

[fill_ / zero_ / masked_fill_]
输入: a=[0,0,0]
    a.fill_(3.0)   → [3.0,3.0,3.0]
    a.zero_()      → [0.0,0.0,0.0]
输入: x=[0,1,2,3,4,5]
    mask = (x%2==0)    # [T,F,T,F,T,F]
    x.masked_fill_(mask, -1)  → [-1,1,-1,3,-1,5]

[_foreach_copy_] — 批量复制
输入: dst0,dst1 = zeros(3), src0=[1,2,3], src1=[4,5,6]
    torch._foreach_copy_([dst0,dst1], [src0,src1])
输出: dst0=[1,2,3], dst1=[4,5,6]
```

**DeepSeek V4 用法**:

```python
# CUDA/MUSA Graph replay 前更新 buffer 内容
graph_input[:raw_bs].copy_(real_input)        # 真实数据 → 固定地址 buffer
metadata_buffer.copy_(new_metadata)            # metadata 原地更新
torch._foreach_copy_(all_graph_bufs, all_real) # 批量更新
```

---

### 2.3 形状变换与内存布局

**MUSA 实际执行**:

```
[view/reshape/flatten/unsqueeze/squeeze]
输入: x = arange(12) = [0,1,...,11]
    x.view(3,4)      → shape=(3,4) [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
    x.reshape(4,3)   → shape=(4,3) [[0,1,2],[3,4,5],[6,7,8],[9,10,11]]
    x.view(2,3,2).flatten(1) → shape=(2,6)
    x[:3].unsqueeze(0)       → shape=(1,3)
    x[:3].unsqueeze(0).squeeze(0) → shape=(3,)

[contiguous/stride]
输入: x = arange(6).view(2,3)
    x.t()  → shape=(3,2) stride=(1,3) is_contiguous=False  # 转置后不连续
    x.t().contiguous() → shape=(3,2) stride=(2,1) is_contiguous=True
    值: [[0,3],[1,4],[2,5]]
```

**框架路径**:

```
MQA 布局: [T, G, D] → view → [B, H, D] → attention → view → [T, G, D]
HC 流展开: [B, S, hc_mult, D] → flatten → [B*S, hc_mult*D]
Fused kernel 输入: 调用 contiguous() 保证 stride 连续
```

**注意**: MUSA kernel 普遍要求 contiguous 输入 → `contiguous()` 调用频率高于 CUDA 版。

---

### 2.4 索引与切分

**功能**: Page Table 查表、MoE token dispatch/combine、TP 分片。

**MUSA 实际执行**:

```
[slice / advanced indexing]
输入: x = arange(20).view(4,5)
    x[1:3, 2:]  → [[7,8,9],[12,13,14]]
    x[[0,2,3],[1,3,4]] → [1,13,19]  # 对角线索引

[gather]
输入: x = arange(20).view(4,5), idx = [[0,2],[1,3],[2,4],[0,1]]
    torch.gather(x, 1, idx) → [[0,2],[6,8],[12,14],[15,16]]

[scatter_] — MoE dispatch 核心
输入: out=zeros(4,5), src=ones(4,2), idx=[[0,2],[1,3],[2,4],[0,1]]
    out.scatter_(1, idx, src)
输出: [[1,0,1,0,0], [0,1,0,1,0], [0,0,1,0,1], [1,1,0,0,0]]

[index_select]
输入: x, dim=0, index=[2,0]
输出: 第2行+第0行 → [[10..14],[0..4]]

[tensor_split]
输入: x=arange(20).view(4,5), tensor_split(2, dim=1)
    chunk[0] shape=(4,3), chunk[1] shape=(4,2)

[torch.where]
输入: x=[0,1,2,3,4,5], cond=(x%2==0)
    torch.where(cond, zeros_like(x), x) → [0,1,0,3,0,5]
```

**DeepSeek V4 Page Table 构造**:

```python
page_table = req_to_token[req_pool_indices_repeated, :max_seq_len:page_size]
page_table = page_table // page_size
page_table = page_table.to(torch.int32)
```

---

### 2.5 序列构造

**MUSA 实际执行**:

```
[arange] — position IDs 生成
输入: torch.arange(3, 7)
输出: [3, 4, 5, 6]

[repeat_interleave] — request ids 展开到 token 级
输入: arange(3).repeat_interleave(2)
输出: [0, 0, 1, 1, 2, 2]
输入: arange(3), repeats=[1,2,1]
输出: [0, 1, 1, 2]
```

---

### 2.6 拼接与填充

**MUSA 实际执行**:

```
[F.pad] — CUDA Graph 固定形状
输入: x=[[0,1,2],[3,4,5]], F.pad(x, (1,2), value=-1)
输出: [[-1,0,1,2,-1,-1], [-1,3,4,5,-1,-1]]

[cat] — TP all-gather 拼接
输入: cat([x,x], dim=0)
输出: shape=(4,3) [[0..2],[3..5],[0..2],[3..5]]

[stack] — 多 rank metadata 堆叠
输入: stack([x,x], dim=0)
输出: shape=(2,2,3) [[[0,1,2],[3,4,5]], [[0,1,2],[3,4,5]]]
```

---

### 2.7 数学归约 — RMSNorm 核心

**MUSA 实际执行**:

```
输入: x = linspace(-3,3,12).view(3,4)
      [[-3.0, -2.45, -1.91, -1.36],
       [-0.82, -0.27, 0.27, 0.82],
       [1.36, 1.91, 2.45, 3.0]]

sum(dim=1)    → [-8.73, 0.0, 8.73]
mean(dim=0)   → [-0.82, -0.27, 0.27, 0.82]
abs().amax(dim=1) → [3.0, 0.82, 3.0]
square()      → [[9.0, 6.02, 3.64, 1.86], [0.67, 0.07, 0.07, 0.67], ...]
rsqrt(abs+0.5) → [[0.53, 0.58, 0.64, 0.73], [0.87, 1.14, 1.14, 0.87], ...]
```

**RMSNorm 链**（生产路径应融合）:

```python
# PyTorch 手写 — 3 kernel launch
x_norm = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)
return x_norm * weight

# Fused — 1 kernel launch
return torch.ops.vllm.rms_norm(x, weight, eps)
```

---

### 2.8 激活函数

**MUSA 实际执行**:

```
输入: x = linspace(-3,3,6) = [-3.0, -1.8, -0.6, 0.6, 1.8, 3.0]

sigmoid(x) → [0.047, 0.142, 0.354, 0.646, 0.858, 0.953]
silu(x)    → [-0.142, -0.255, -0.213, 0.387, 1.545, 2.858]
gelu(x)    → [-0.004, -0.065, -0.165, 0.435, 1.735, 2.996]
relu(x)    → [0.0, 0.0, 0.0, 0.6, 1.8, 3.0]
softmax(x,dim=0) → [0.002, 0.006, 0.019, 0.063, 0.211, 0.699]
clamp(x,-1,1)    → [-1.0, -1.0, -0.6, 0.6, 1.0, 1.0]
```

**SwiGLU** (LLM 核心激活):

```python
# PyTorch 手写 (2 kernel)
gate, up = x.chunk(2, dim=-1)
output = F.silu(gate) * up

# 生产 fused (1 kernel)
torch.ops.sgl_kernel.silu_and_mul(x, out)
# DeepSeek V4 额外: gate.clamp(max=10.0), up.clamp(-10.0, 10.0)
```

---

### 2.9 线性代数

**MUSA 实际执行**:

```
[F.linear] — Q/K/V/O 投影
输入: x=[[0,1,2],[3,4,5]], W=I₃+bias_row, bias=[1,0,0,2]
输出: F.linear(x,W,bias) = [[1,1,2,5],[4,4,5,14]]

[mm] — 2D 矩阵乘
输入: a=[[1,2],[3,4]], b=[[5,6],[7,8]]
输出: mm(a,b) = [[19,22],[43,50]]

[bmm] — Batch 矩阵乘 (每 head 独立)
输入: ba=[[[0,1],[2,3]],[[4,5],[6,7]]], bb=[[[1,2],[3,4]],[[5,6],[7,8]]]
输出: [[[3,4],[11,16]],[[55,64],[79,92]]]

[einsum] — MQA grouped output
输入: e0=[[0,1,2],[3,4,5]], e1=I₃
输出: einsum('ij,jk->ik', e0, e1) = [[0,1,2],[3,4,5]]
```

**生产替换**:

```
F.linear → deep_gemm.fp8_gemm_nt     (Hopper/Blackwell)
         → sgl_kernel.musa_fused_gemv (MUSA)
         → MuDNN matmul               (MUSA dense)

einsum   → deep_gemm.m_grouped_fp8_gemm_nt_contiguous
```

---

### 2.10 TopK / 排序 / 路由

**MUSA 实际执行**:

```
[topk] — MoE expert 选择
输入: scores=[0.1, 3.0, 2.0, -1.0]
    topk(scores, k=2)
输出: values=[3.0, 2.0], indices=[1, 2]

[sort / argsort]
输入: [3,1,2]
    sort.values=[1,2,3], sort.indices=[1,2,0]
    argsort=[1,2,0]

[argmax] — Greedy 解码
输入: [3,1,5,2]
输出: argmax=2 (值为5的元素索引)
```

**DeepSeek V4 MoE 路由**:

```python
router_logits = F.linear(hidden_states, W_gate)           # [B,S,n_experts]
affinity = torch.sqrt(F.softplus(router_logits))           # SqrtSoftplus
weights, indices = torch.topk(affinity, k=6, dim=-1)      # 选 top-6
weights = F.softmax(weights, dim=-1)                       # renormalize
```

---

### 2.11 精度转换与 CPU 同步

**MUSA 实际执行**:

```
输入: x = [1.5, 2.8, 3.1]

x.to(int32)    → [1, 2, 3]
x.float()      → dtype=float32
x.bfloat16()   → dtype=bfloat16
x.cpu()        → device=cpu, [1.5, 2.8, 3.1]  (同步边界!)
x[0].item()    → 1.5  (GPU标量→Python float, 同步!)
x.tolist()     → [1.5, 2.8, 3.1]  (GPU tensor→Python list, 同步!)
```

**⚠️ 性能陷阱**: `.item()`, `.tolist()`, `.cpu()` 在 GPU tensor 上调用会打断异步流水线。SGLang 用 `seq_lens_cpu` (CPU mirror) 规避。

---

### 2.12 MUSA Graph

**MUSA 实际执行**:

```
[MUSAGraph capture/replay]
1. Warmup: static_out.copy_(static_in * 2 + 1)  # 预热
2. Capture: graph.capture_begin() → copy_ → capture_end()
3. Replay:  static_in.fill_(5.0) → graph.replay()
   输入: static_in = [[5,5,5],[5,5,5],[5,5,5]]
   输出: static_out = [[11,11,11],[11,11,11],[11,11,11]]  (5*2+1=11 ✅)
```

**关键约束**:
- Graph 内 tensor 地址不可变 → `empty` + `copy_` 模式
- 禁止 `.item()`, `.tolist()`, `.cpu()`, `print`
- 禁止 dynamic shape 操作 (cat/stack 可能分配新 tensor)
- 禁止 `.unique()`, `.nonzero()` (动态输出 shape)

---

## 3. SGLang DeepSeek V4 实战

### 3.1 模型架构与 Op 映射

DeepSeek V4 的 DecoderLayer 由三个子模块组成：

```
Input → HC Pre → Attention(CSA/HCA) → HC Post → MoE(Hash/TopK) → Output
```

**各模块的 PyTorch Op 链**:

| 模块 | PyTorch fallback op 链 | 生产 fused kernel |
|------|----------------------|-------------------|
| HC Pre | `flatten→float→square→mean→rsqrt→F.linear→sigmoid→unsqueeze→sum` | TileLang single kernel |
| HC Post | `unsqueeze→broadcast mul→sum(dim=1)` | TileLang single kernel |
| Attention (CSA) | `qkv proj→kv state→softmax pool→RoPE→indexer→sparse attn→wo_a einsum` | Compressor+Indexer+SparseAttn fused |
| MoE routing | `F.linear→softplus→sqrt→topk→softmax` | Fused gating topk |
| MoE FFN | `permute→gate_gemm→up_gemm→silu*up→clamp→down_gemm→unpermute` | MegaMoE single kernel |
| Metadata | `empty→arange→indexing→masked_fill_→to(int32)→copy_` | Triton metadata kernel |

### 3.2 MoE 路由完整路径

```python
# Step 1: Router 投影
router_logits = F.linear(hidden_states, W_gate)           # [B,S,256_experts]

# Step 2: SqrtSoftplus (V4 专属，替代 V3 的 sigmoid)
affinity = torch.sqrt(F.softplus(router_logits))

# Step 3: Top-6 专家选择
topk_weights, topk_indices = torch.topk(affinity, k=6, dim=-1)

# Step 4: Renormalize (auxiliary-loss-free)
topk_weights = F.softmax(topk_weights, dim=-1)

# Step 5-7: Token dispatch → Expert FFN → Combine (MegaMoE fused)
# 单 kernel: EP dispatch + L1 GEMM + SwiGLU + L2 GEMM + EP combine
```

### 3.3 HC fallback vs Fused

```python
# PyTorch fallback (~6 kernel launch)
residual_flat = residual.flatten(1).float()                     # kernel 1
norm = residual_flat * torch.rsqrt(
    residual_flat.square().mean(-1, keepdim=True) + eps)        # kernel 2+3
mix = F.linear(norm, fc_hc_fn)                                  # kernel 4
mix = torch.sigmoid(mix)                                        # kernel 5
layer_input = (mix * streams).sum(dim=1)                        # kernel 6

# MUSA TileLang fused (1 kernel launch)
mhc_pre_big_fuse(residual, fn, mhc_scale, mhc_base, eps, ...)
```

---

## 4. CUDA/MUSA Graph 分析

### 4.1 Graph 生命周期

```
1. Init: 预分配固定 shape buffer (empty/zeros)
2. Warmup: run_once() ×2 走通路径
3. Capture: graph.capture_begin() → forward → capture_end()
   记录 tensor 地址 + kernel launch 序列
4. Replay: buffer.copy_(real_data) → graph.replay()
   每步 decode 热路径 (消除 launch overhead)
```

### 4.2 SGLang 的 Graph 约束

```python
# Capture 前 (warmup)
seq_lens_sum = seq_lens.sum().item()   # GPU→CPU 同步，只在 warmup 做
attn_backend.init_forward_metadata_capture_cuda_graph(...)

# Replay 热路径 (不能有同步)
graph_input[:raw_bs].copy_(real_input)    # 用 copy_ 更新 buffer
graph.replay()                             # 纯 GPU 执行
# 禁止: item(), tolist(), cpu(), print, unique()
```

---

## 5. 三框架 Op 策略对比

| Op 类别 | PyTorch 原生 | SGLang (生产) | vLLM (生产) | Megatron (训练) |
|---------|-------------|--------------|------------|----------------|
| GEMM dense | `F.linear` | DeepGEMM fp8_gemm / MuDNN | CUTLASS scaled_mm | TE linear |
| GEMM MoE | 多次 `F.linear` | **MegaMoE** (fused) | fused_moe (Triton) | TE MoE |
| Attention | SDPA | FlashMLA / FlashAttn | FlashInfer / FA2 | FA3 + TE |
| RMSNorm | square+mean+rsqrt | tilelang pre_norm | vllm.rms_norm | TE rmsnorm |
| SwiGLU | silu(gate)*up | silu_and_mul_musa | silu_and_mul | TE swiglu |
| RoPE | sin/cos + rotate | partial rotary | fused_rotary | fused_rotary |
| TopK | torch.topk | flashinfer topk | vllm sample | (训练不用) |
| FP8 Quant | .to(float8) | block quant + GEMM | scaled_mm | TE delayed scaling |
| Graph | MUSAGraph/CUDAGraph | ✅ decode | ✅ decode | ❌ |

---

## 6. MUSA 适配差异

| 模块 | CUDA 上游 | MUSA 适配 | 原因 |
|------|----------|----------|------|
| GEMM | DeepGEMM FP8/FP4 GEMM | MuDNN matmul / musa_fused_gemv | MUSA 无 Tensor Core → 只能用 GEMV |
| HC pre | torch fallback / DeepGEMM | TileLang fused kernel | CUDA fused kernel 不能直接跑 |
| SwiGLU | JIT CUDA kernel | silu_and_mul_musa | TileLang MUSA-target |
| FlashMLA store | JIT CUDA pack/store | 18+ TileLang kernel 变体 | MUSA memory layout 不同 |
| Graph | torch.cuda.CUDAGraph | torch.musa.MUSAGraph | API 兼容，独立命名 |
| Stream | cudaStream_t | musaStream_t | 完全替换 |
| Device str | "cuda" | "musa" | 所有 device 字面量 |
| Arch detect | __CUDA_ARCH__ | __MUSA_ARCH__ (220/310) | 不同 warp 大小 |

---

## 7. 性能洞察与优化原则

### 7.1 瓶颈分布 (DeepSeek V4 decode)

```
Kernel 类型              GPU 占比     瓶颈类型
─────────────────────────────────────────────
bf16 element-wise copy     45%       Memory BW
bf16 element-wise mul      17%       Memory BW
bf16 clamp/mul             14%       Memory BW
FP8 GEMM                    4%       Compute
Fused MoE                 0.8%       Compute
其他                        19%       -
```

**结论**: 77% 时间在 memory-bound 操作上。优化方向应为减少 kernel launch 数 + 融合 element-wise op，而非加速 GEMM。

### 7.2 核心原则

1. **GEMM 从不裸用 PyTorch** — 生产路径 100% 替换
2. **融合优于精度损失** — 减少 launch > 微调 FLOP 利用率
3. **CPU 同步是杀手** — `.item()/.tolist()/.cpu()` 打断流水线
4. **Graph 要求地址稳定** — `empty` + `copy_` 模式
5. **PyTorch 做胶水** — 管理 tensor 形态，计算下沉 fused kernel

### 7.3 MUSA 特殊考量

- MUSA arch < 300 不支持 FP8
- 所有 MUSA kernel 要求 contiguous 输入
- GEMV 而非 GEMM → prefill 性能差距显著
- TileLang 是主要的 MUSA kernel 编写语言
- `mate.moe_fused_gate` 提供 MUSA 加速的 MoE gate 算子

---

*验证环境: shanfeng@10.18.32.25 / Docker mochi-sglang / PyTorch 2.9.0 / MUSA 8卡*
*17/17 用例全部通过*
