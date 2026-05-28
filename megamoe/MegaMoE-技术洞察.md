# MegaMoE 技术洞察

> SGLang 中基于 DeepGEMM 的融合 MoE 推理路径。面向 DeepSeek V2/V4 等大语言模型，将 MoE 的 token dispatch、per-expert GEMM、activation 和 combine 融合为单一 kernel 调用，消除传统逐 expert 循环中的 kernel launch 开销和中间显存往返。

---

## 1. 问题背景

### 1.1 传统 MoE 推理的执行模式

在标准的 MoE（Mixture of Experts）推理中，每个 Transformer 层的 MoE block 执行以下步骤：

```text
hidden_states: [num_tokens, hidden]      # 例如 [256, 7168]

1. Router（门控网络）
   router_logits = gate(hidden_states)    # [256, 256]  256 个 routed experts
   topk_ids, topk_weights = topk(router_logits, k=8)  # 每个 token 选 8 个专家

2. Token Dispatch（token 分发）
   for each expert in 256:
       收集被路由到这个 expert 的 token 子集
       # 稀疏矩阵乘法：只有 ~8/256 的 token 被每个 expert 处理

3. Per-Expert GEMM（逐专家计算）
   for each expert:
       x_i = hidden_states[selected_tokens]     # [tokens_for_expert_i, hidden]
       gate = x_i @ W_gate.T                     # [tokens_for_expert_i, intermediate]
       up   = x_i @ W_up.T                       # [tokens_for_expert_i, intermediate]
       y_i  = silu(gate) * up                    # SwiGLU activation
       out_i = y_i @ W_down.T                    # [tokens_for_expert_i, hidden]
       out_i *= topk_weights                      # 路由权重缩放

4. Combine（结果合并）
   final = scatter_add(out_i for all experts)    # [num_tokens, hidden]
   final += shared_expert(hidden_states)          # 共享专家
```

**问题**：当 `num_experts` 很大（256）且每个 expert 处理的 token 很少（如 decode 阶段 batch=1 时只有 8 个 token 被选中），传统逐 expert 循环会遭遇严重的 kernel launch 瓶颈——每层要发起 256×3=768 次小 GEMM kernel，每次只处理几个 token，GPU 利用率极低。

### 1.2 MegaMoE 的核心思路

MegaMoE 将上述步骤 2-4 融合为**两次 kernel 调用**：

1. **Pre-Dispatch Kernel**（SGLang JIT）：将 bf16 hidden_states 按 per-token-group 量化为 FP8，拷贝 topk_ids/topk_weights 到 DeepGEMM 所需的 SymmBuffer 布局
2. **Fused Mega Kernel**（DeepGEMM）：在一次 kernel 内完成 all-to-all dispatch + L1 GEMM (gate+up) + SwiGLU + L2 GEMM (down) + combine

这样将 O(num_experts) 次 kernel launch 降为 O(1) 次，同时利用 FP8 量化减少显存带宽压力。

---

## 2. 架构全景

```text
                    ┌──────────────────────────────────────────────┐
                    │            MegaMoE Forward Path              │
                    └──────────────────────────────────────────────┘

hidden_states [T, H] bf16          topk_ids [T, K] int32       topk_weights [T, K] float
         │                                  │                          │
         ▼                                  ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1: mega_moe_pre_dispatch (SGLang JIT Kernel)                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ • Per-token group quantization: bf16 → fp8_e4m3 (group_size=32)     │   │
│  │ • Compute UE8M0 scale factors per group                             │   │
│  │ • Copy topk_ids / topk_weights to SymmBuffer layout                 │   │
│  │ • Pad unused slots with (-1, 0)                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│     buf.x [P, H] fp8              buf.x_sf [P, G/4] int32                   │
│     buf.topk_idx [P, K] int64     buf.topk_weights [P, K] float             │
└────────────────────────────────────┴────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 2: deep_gemm.fp8_fp4_mega_moe (DeepGEMM Fused Kernel)                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Internally:                                                         │   │
│  │  ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────┐   │   │
│  │  │ All-to-All      │ → │ L1 GEMM          │ → │ SwiGLU           │   │   │
│  │  │ Dispatch        │   │ (gate_proj +     │   │ Activation        │   │   │
│  │  │ (token → expert)│   │  up_proj fused)  │   │ silu(gate) * up   │   │   │
│  │  └─────────────────┘   └──────────────────┘   └────────┬─────────┘   │   │
│  │                                                        ▼             │   │
│  │                   ┌──────────────────┐   ┌──────────────────┐        │   │
│  │                   │ Combine          │ ← │ L2 GEMM          │        │   │
│  │                   │ (scatter back    │   │ (down_proj)      │        │   │
│  │                   │  to tokens)      │   │                  │        │   │
│  │                   └──────────────────┘   └──────────────────┘        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Input:                                                                     │
│    buf (SymmBuffer) + mega_l1_weights + mega_l2_weights                     │
│    recipe=(1,1,32)  activation="swiglu"  fast_math=True                     │
│                                                                             │
│  Output:                                                                     │
│    y [T, H] bf16                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                          y *= routed_scaling_factor
                          y += shared_expert_output
                                     │
                                     ▼
                          final_hidden_states [T, H]
```

---

## 3. 核心组件详解

### 3.1 Pre-Dispatch Kernel（SGLang JIT）

源码位置：`repos/sglang/python/sglang/jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh`

这是一个 CUDA/MUSA kernel，完成三项工作：

#### 3.1.1 Per-Token-Group FP8 量化

```c
// 关键参数
constexpr uint32_t kGroupSize = 32;   // 量化组大小
constexpr uint32_t kVecElems = 8;     // 每线程处理 8 个 bf16（16B 对齐加载）
constexpr uint32_t kThreadsPerGroup = kGroupSize / kVecElems;  // = 4 threads/group

// 每线程处理逻辑（一个 CTA 处理一个 token 的一行）
for each group (32 elements):
    4 threads 协作：
      1. 加载 8 个 bf16 值（16B aligned vector load）
      2. 计算 group 内 absmax → scale
      3. absmax / FP8_E4M3_MAX → raw_scale
      4. cast_to_ue8m0(raw_scale) → UE8M0 指数字节
      5. 量化：val * inv_scale → pack_fp8
      6. 写入 buf_x（fp8）和 buf_x_sf（UE8M0 scale）
```

**UE8M0 格式**：UE8M0 是 FP8 的 scale 存储格式——只存储 8-bit 指数部分（无尾数）。`raw_scale` 是一个 float32，`cast_to_ue8m0` 提取其指数并打包为 uint8。这比存 float32 scale 节省 4× 空间。

**量化布局**：
```
buf_x:    [P, H] fp8_e4m3          — 量化后的激活值
buf_x_sf: [P, H/32] uint8 packed   — 每 32 个元素一个 UE8M0 scale
          存储为 [P, H/128] int32  — 4 个 UE8M0 打包为 1 个 int32
```

#### 3.1.2 TopK 数据拷贝与 Padding

```c
// 每个 CTA 的前 top_k 个线程负责拷贝
if (tid < params.top_k) {
    buf.topk_idx[token_id * top_k + tid] = topk_idx[token_id * top_k + tid];
    buf.topk_weights[token_id * top_k + tid] = topk_weights[token_id * top_k + tid];
}

// 剩余 CTA 填充 padding 区域（num_tokens → padded_max）
// padded_max 由 CUDA Graph 静态 shape 决定
for slot in [num_tokens * top_k, padded_max * top_k):
    buf.topk_idx[slot] = -1;
    buf.topk_weights[slot] = 0.0f;
```

Padding 的必要性：SGLang 使用 CUDA Graph 加速 decode，graph 捕获时需要固定 shape。`padded_max` 是 CUDA Graph 的最大 batch size（`cuda_graph_max_bs`），实际 token 数可能小于此值，因此需要 padding 并用 `-1` 标记无效 slot。

#### 3.1.3 线程组织

```c
// grid = num_tokens + num_pad_blocks
// block = hidden / 8 线程（每线程处理 16B）
// 例如 hidden=7168 → block=896 线程
// 约束：block ≤ 1024，hidden % 8 == 0
```

### 3.2 DeepGEMM Fused Mega Kernel

源码位置：`deep_gemm` 包（外部依赖，不在 workspace 内）

调用接口：
```python
deep_gemm.fp8_fp4_mega_moe(
    y,                              # output: [T, H] bf16
    moe.experts.mega_l1_weights,    # (w13_weight, w13_scale) — gate+up fused
    moe.experts.mega_l2_weights,    # (w2_weight, w2_scale)   — down_proj
    buf,                            # SymmBuffer: x, x_sf, topk_idx, topk_weights
    recipe=(1, 1, 32),              # (M_block, N_block, K_block) tiling
    activation="swiglu",            # silu(gate) * up
    activation_clamp=swiglu_limit,  # DeepSeekV4 的 swiglu clamp 上限
    fast_math=True,                 # 允许 TF32/FP8 近似
)
```

**DeepGEMM 内部做了什么**（基于源码结构推断）：

```
1. All-to-All Dispatch（token → expert 映射）
   - 读取 buf.topk_idx，建立 token_id → expert_id 的映射
   - 按 expert 分组 token，构造每个 expert 的输入子矩阵
   - 利用 NVSwitch/IB 进行跨 GPU 的 expert-parallel 通信

2. L1 GEMM（gate_proj + up_proj 融合）
   - mega_l1_weights 是 w13_weight: [num_experts, 2*intermediate, hidden]
   - 前半 [intermediate, hidden] 是 gate_proj
   - 后半 [intermediate, hidden] 是 up_proj
   - 单次 GEMM 同时计算 gate 和 up 投影

3. SwiGLU Activation
   - gate, up = chunk(output_L1, 2, dim=-1)
   - output = silu(gate) * up
   - DeepSeekV4 特有：gate = clamp(gate, max=swiglu_limit)

4. L2 GEMM（down_proj）
   - mega_l2_weights: [num_experts, hidden, intermediate]
   - 对 SwiGLU 输出做降维投影

5. Combine（scatter back）
   - 将各 expert 的输出按 token_id scatter 回原始位置
   - 乘以 topk_weights（路由权重）
   - 同一 token 被多个 expert 处理的结果相加
```

### 3.3 权重布局变换（Weight Layout Transform）

源码位置：`repos/sglang/python/sglang/srt/layers/moe/mega_moe.py` 的 `build_mega_moe_experts_weights()`

DeepGEMM 的 mega kernel 要求权重按特定布局存储，需要在模型加载时做一次预处理：

```python
# 原始权重（标准 MoE 格式）
w13: [num_experts, 2*intermediate, hidden]   # gate + up 拼接
w2:  [num_experts, hidden, intermediate]      # down

# 步骤 1：Scale 格式转换
w13_sf = transform_sf_into_required_layout(
    w13_sf_fp32, mn=2*intermediate, k=hidden,
    recipe=(1, 32), num_groups=num_experts
)
# float32 scale → UE8M0 格式 + 按 (1,32) block 重排

# 步骤 2：权重+Scale 交织（Memory Optimization）
if SGLANG_OPT_FIX_MEGA_MOE_MEMORY:
    # 方案 A：原位修改，共享 deep-ep 的权重 buffer（省显存）
    w13_interleaved, w13_sf_interleaved = _interleave_l1_weights((w13, w13_sf))
    w13_sf_utccp = _transpose_sf_for_utccp(w13_sf_interleaved)
    experts.mega_l1_weights = (experts.w13_weight.data, w13_sf_utccp)
else:
    # 方案 B：独立副本（默认，多占一份显存）
    l1_pair, l2_pair = transform_weights_for_mega_moe((w13, w13_sf), (w2, w2_sf))
    experts.mega_l1_weights = l1_pair
    experts.mega_l2_weights = l2_pair
```

**UTCCP**（Unified Tensor Core Compute Pipeline）是 DeepGEMM 内部的 Tensor Core 数据通路格式，要求 scale 矩阵按特定 tile 顺序排列以匹配 weight 的 swizzle 模式。

### 3.4 SymmBuffer

源码位置：`deep_gemm.get_symm_buffer_for_mega_moe()`

SymmBuffer 是 DeepGEMM 为 mega kernel 预分配的对称缓冲区：

```python
buf = deep_gemm.get_symm_buffer_for_mega_moe(
    ep_group,                    # expert-parallel 通信组
    num_experts=256,             # routed experts 数量
    num_max_tokens_per_rank=256, # 每 rank 最大 token 数（cuda_graph_max_bs）
    num_topk=8,                  # 每 token 选择的 expert 数
    hidden=7168,                 # hidden size
    intermediate_hidden=2048,    # FFN intermediate size
    use_fp8_dispatch=True,       # 使用 FP8 量化 dispatch
    activation="swiglu",         # 激活函数类型
)

# buf 包含以下缓冲区（均预分配，CUDA Graph 友好）：
buf.x:            [padded_max, hidden] fp8          # 量化后的输入
buf.x_sf:         [padded_max, hidden/128] int32    # UE8M0 scale（4个/ int32）
buf.topk_idx:     [padded_max, top_k] int64         # expert 索引
buf.topk_weights: [padded_max, top_k] float         # 路由权重
```

缓冲区按 `(ep_group, num_max_tokens_per_rank, num_experts, num_topk, hidden, intermediate_hidden)` 为 key 缓存，同形状的层共享。

---

## 4. 完整示例

### 4.1 DeepSeekV4 MoE 层参数

```text
模型: DeepSeekV4
hidden_size:          7168
moe_intermediate_size: 2048
n_routed_experts:     256
num_experts_per_tok:  8       (TopK=8)
num_fused_shared_experts: 1   (共享专家融合到 MoE kernel)
routed_scaling_factor: 2.5

EP (Expert Parallel) size: 16  (跨 16 GPU)
每 GPU 负责: 256/16 = 16 个 routed experts
```

### 4.2 Decode 阶段执行（batch=1, 单 token）

```text
输入:
  hidden_states: [1, 7168] bf16

Step 1: Router
  router_logits = gate(hidden_states)           → [1, 256]
  topk_ids, topk_weights = topk(logits, k=8)    → [1, 8], [1, 8]
  # 选出 8 个 expert，例如: [12, 45, 78, 103, 156, 189, 210, 243]

Step 2: Pre-Dispatch
  mega_moe_pre_dispatch(
      hidden_states [1, 7168] bf16
      → buf.x [256, 7168] fp8          # padded_max=256 (CUDA Graph 固定)
      → buf.x_sf [256, 56] int32       # 7168/32=224 groups → 224/4=56 int32
      → buf.topk_idx [256, 8] int64    # 第 0 行有效，其余 -1
      → buf.topk_weights [256, 8] float
  )
  # 注意：只有第 0 行有数据，第 1-255 行是 padding (-1, 0)

Step 3: Fused Mega Kernel
  y = deep_gemm.fp8_fp4_mega_moe(
      buf,
      mega_l1_weights,   # [16, 4096, 7168] — 每 GPU 16 experts，gate+up=2*2048
      mega_l2_weights,   # [16, 7168, 2048]
      recipe=(1,1,32)
  )
  → y [1, 7168] bf16

Step 4: Post-processing
  y *= 2.5                          # routed_scaling_factor
  y += shared_expert_output         # 如果 SBO overlap 未启用，已在前面计算
```

### 4.3 Prefill 阶段执行（batch=128, 128 token）

```text
输入:
  hidden_states: [128, 7168] bf16

Step 1: Router
  router_logits: [128, 256]
  topk_ids:      [128, 8]    # 每 token 选 8 个 expert
  topk_weights:  [128, 8]

Step 2: Pre-Dispatch
  # 128 个 CTA 并行处理 128 个 token
  # 每个 CTA: hidden/8 = 896 threads
  # 量化 128×7168 bf16 → 128×7168 fp8
  # + 128-255 行 padding

Step 3: Fused Mega Kernel
  # DeepGEMM 内部：
  #   token 0:  expert [12, 45, 78, 103, 156, 189, 210, 243]
  #   token 1:  expert [3, 45, 67, 112, 156, 200, 210, 255]
  #   ...
  #   expert 12 收到 tokens [0, 15, 47, 89, ...]（来自不同 GPU）
  #   all-to-all 通信后 → 本地 16 experts 各自计算
  #   结果 scatter 回 [128, 7168]
```

---

## 5. 关键优化技术

### 5.1 FP8 量化（Per-Token-Group）

```
精度分析：
  bf16 输入 → group_size=32 → per-group absmax → UE8M0 scale → fp8_e4m3

量化误差：
  每个 32-element group 独立 scale，最大量化误差 ≤ absmax/256
  对于 LLM 推理，group_size=32 的 FP8 精度损失 < 0.1% perplexity

带宽节省：
  bf16 (2B/elem) → fp8 (1B/elem) = 2× 带宽节省
  scale 开销：UE8M0 (1B/32elem) = 3.1% 额外带宽
  净节省：≈ 48% 显存带宽
```

### 5.2 SBO Overlap（Shared-expert / Broadcast Overlap）

```python
# mega_moe.py: forward_mega_moe()
sbo_overlap_flag = (
    moe.alt_stream is not None
    and moe.num_fused_shared_experts == 0  # shared expert 未融合到 mega kernel
    and num_tokens > 0
    and get_is_capture_mode()              # CUDA Graph 模式
)

if sbo_overlap_flag:
    # shared expert 在 alt_stream 上与 mega kernel 并行执行
    moe.alt_stream.wait_stream(current_stream)
    shared_output = moe._forward_shared_experts(hidden_states)
    with torch.cuda.stream(moe.alt_stream):
        y = _run_mega_routed(...)
    current_stream.wait_stream(moe.alt_stream)
```

当 shared expert 未融合到 mega kernel 时（DeepEP 路径），shared expert 计算可以与 routed mega kernel 在不同 CUDA stream 上并行，隐藏 shared expert 的延迟。

### 5.3 CUDA Graph 友好设计

MegaMoE 的所有 buffer 在初始化时预分配，shape 固定为 `padded_max`（= `cuda_graph_max_bs`）。运行时只需填充有效数据（`num_tokens ≤ padded_max`），padding 区域由 pre-dispatch kernel 自动填充 `-1`。这使得整个 forward 路径可以被 CUDA Graph 捕获，消除 decode 阶段的 kernel launch 开销。

### 5.4 权重内存优化（SGLANG_OPT_FIX_MEGA_MOE_MEMORY）

默认情况下，`transform_weights_for_mega_moe()` 创建权重的独立副本——一份用于标准 MoE 路径，一份用于 MegaMoE。开启 `SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1` 后，两个路径共享同一份权重 buffer（通过 `_interleave_l1_weights` 原位修改），节省约 50% 的 expert 权重显存。

---

## 6. 触发条件

MegaMoE 的启用由 `should_use_mega_moe()` 控制：

```python
def should_use_mega_moe(moe, hidden_states) -> bool:
    # 条件 1: 环境变量显式启用
    if not envs.SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE:
        return False

    # 条件 2: 权重已完成 mega 布局转换
    if not moe.experts._mega_moe_weights_built:
        return False

    # 条件 3: CUDA Graph 模式 → 直接启用（decode 阶段）
    if get_is_capture_mode():
        return True

    # 条件 4: 非 CUDA Graph 模式 → token 数不超过上限
    max_tokens_per_rank = max(global_num_tokens)
    cap = SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK
    return max_tokens_per_rank <= cap
```

**适用场景**：
- ✅ Decode 阶段（CUDA Graph 模式，batch ≤ `cuda_graph_max_bs`）
- ✅ 小 batch prefill（token 数 ≤ `NUM_MAX_TOKENS_PER_RANK`）
- ❌ 大 batch prefill（token 数超限时回退到标准 MoE 路径）

---

## 7. 对 MUSA 性能建模的启示

### 7.1 MegaMoE 的成本分解

从性能建模角度看，MegaMoE 的一次 forward 包含以下成本项：

| 阶段 | 操作 | 成本特征 | 建模方式 |
|------|------|---------|---------|
| Router | `hidden @ W_gate.T` + topk | 小 GEMM (H×E)，计算密集 | `musa_benchmarks` 小 GEMM 校准 |
| Pre-Dispatch | bf16→fp8 量化 + memcpy | 访存密集，O(T×H) | 内存带宽模型 + `memcpy` benchmark |
| All-to-All | NVSwitch/IB 通信 | 网络延迟 + 带宽 | EP group size、token 分布建模 |
| L1 GEMM | `x_i @ mega_l1.T` | 大 GEMM，计算密集 | occupancy 模型 + GEMM throughput 校准 |
| SwiGLU | `silu(gate) * up` | Element-wise，访存密集 | 内存带宽模型 |
| L2 GEMM | `act @ mega_l2.T` | 大 GEMM，计算密集 | 同 L1 |
| Combine | Scatter-add + weight mul | 访存密集，O(T×H) | 内存带宽模型 |
| Post | `*= scale` + `+= shared` | Element-wise | 可忽略 |

### 7.2 与传统 MoE 的关键差异

| 维度 | 传统 MoE | MegaMoE |
|------|---------|---------|
| Kernel Launch 次数 | O(num_experts × 3) ≈ 768 次 | O(1) = 2 次 |
| 中间数据格式 | bf16（全精度） | fp8（量化，省 48% 带宽） |
| 显存往返 | 每 expert 一次 global memory 读写 | 融合 kernel 内寄存器/L1 复用 |
| 跨 GPU 通信 | 显式 all-to-all | DeepGEMM 内部融合 |
| CUDA Graph 兼容 | 需逐 expert 捕获（复杂） | 天然兼容（固定 shape buffer） |

### 7.3 MUSA 适配要点

MegaMoE 的 pre-dispatch kernel 已有 MUSA 移植（`#include <musa_fp8.h>` 替代 `<cuda_fp8.h>`，使用 `kDLCUDA` 设备选项），核心的 DeepGEMM `fp8_fp4_mega_moe` 是外部依赖，需要：
1. 确认 DeepGEMM 是否已适配 MUSA（关注 `DG_USE_FP4_ACTS`、`DG_USE_MXF4_KIND` 等 FP4/MXF4 特性）
2. 确认 SymmBuffer 的 UE8M0 scale 布局在 MUSA 上的兼容性
3. 关注 MUSA 的 Tensor Core FP8 吞吐是否匹配 mega kernel 的 `recipe=(1,1,32)` tiling

---

## 8. 相关文件索引

| 文件 | 内容 |
|------|------|
| `repos/sglang/python/sglang/srt/layers/moe/mega_moe.py` | MegaMoE forward 主逻辑、权重构建、SymmBuffer 管理 |
| `repos/sglang/python/sglang/jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh` | Pre-dispatch CUDA/MUSA kernel（bf16→fp8 量化） |
| `repos/sglang/python/sglang/srt/models/deepseek_v2.py` | DeepSeekV2MoE 类，MegaMoE 调用点（L654-657） |
| `repos/sglang/python/sglang/jit_kernel/deepseek_v4.py` | JIT kernel Python 绑定 |
| `repos/github/sglang/python/sglang/srt/layers/moe/mega_moe.py` | 上游版 MegaMoE（含 FP4/MXF4 支持） |
| `repos/github/sglang/python/sglang/jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh` | 上游版 pre-dispatch kernel（CUDA） |
| `insights/deepseek_v4_source_walkthrough/02_deepseek_v4_source_module_execution.md` | DeepSeekV4 模型执行详解（含 MoE 路由机制） |
