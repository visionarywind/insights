# PyTorch 常用 Op 技术洞察 — 以 DeepSeek V4 SGLang 推理为例

**更新**: 2026-05-21
**范围**: SGLang `deepseek_v4.py`、`deepseek_v4_backend.py`、DeepGEMM/MegaMoE/FlashMLA kernel 生态，HuggingFace Transformers V5.8 DeepSeek-V4 实现。

---

## 1. DeepSeek V4 架构总览

DeepSeek V4 是 2026 年 4 月发布的新一代 MoE 语言模型，两个变体：

| 变体 | 总参数 | 激活参数 | 上下文 |
|------|--------|----------|--------|
| **V4-Flash** | 284B | 13B (top-6 experts) | 1M token |
| **V4-Pro** | 1.6T | 49B (top-6 experts) | 1M token |

### 核心架构创新

```
Token → Embedding → [DecoderLayer × 61] → LM Head → Logits
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       CSA/HCA      Hash-MoE/      mHC
      Attention     Top-k MoE     (hyper-connection)
```

**DecoderLayer 三大子模块**：

1. **Hybrid Attention** — CSA (Compressed Sparse, m=4) + HCA (Heavily Compressed, m′=128) 交替排列，每层附带 SWA (Sliding Window, n=128) 和 Lightning Indexer。V4-Pro 前 2 层纯 HCA bootstrap，后续 CSA/HCA 交替。
2. **Mixture-of-Experts** — 前 3 层 `hash_moe`（静态 token_id → expert_id 查表），其余 `moe`（`Sqrt(Softplus(·))` 路由 top-6，共享 expert 并行）。Expert 激活使用 **clamped SwiGLU** (`gate.clamp(max=swiglu_limit)`, `up.clamp(min=-swiglu_limit, max=swiglu_limit)`)。
3. **mHC (Manifold-Constrained Hyper-Connections)** — 替代残差连接，`hc_mult=4` 并行流经 `(pre, post, comb)` 三元组混合矩阵（doubly-stochastic via Sinkhorn-Knopp），保证信号非扩张传播。

**精度方案**：FP4 MoE experts + FP8 attention/dense，BF16 accumulation。

---

## 2. PyTorch Op 分类与 DeepSeek V4 使用矩阵

### 2.1 创建 Tensor

| Op | 用途 | 热路径频率 |
|----|------|-----------|
| `torch.empty/zeros/ones` | 预分配 KV cache (CSA/HCA compressor buffer, SWA cache, page table)、attention metadata (cu_seq_lens, block_tables)、MoE token dispatch buffer | **每 forward 数十次** |
| `new_empty/new_zeros` | 继承已有 tensor 的 device/dtype 创建输出 buffer（`q_padded = x.new_empty(...)` 分配 MQA Q buffer） | 每 layer |
| `torch.full` | 初始化 mask、padding value | 低频 |
| `torch.arange(..., out=out)` | 直接写入 GPU tensor 生成序列索引，避免 Python 循环 | prefill 高频 |

**关键模式**：`empty/new_empty` 只分配不初始化，后续 fused kernel 完整覆写 → 避免双写开销。

### 2.2 Shape/View/布局变换

| Op | DeepSeek V4 中的典型用法 |
|----|--------------------------|
| `view` / `reshape` | MQA head 分组间切换：`[T, G, D]` ↔ `[B, H, D]`；mHC hc_mult 维度展开/折叠：`[B, S, hc_mult, D]`；MoE gate/up weight 拆解 |
| `flatten` | HC fallback torch 路径中 `flatten(1)` 做 RMS-like 归一化前降维 |
| `unsqueeze` / `squeeze` | broadcast 操作前的维度对齐（RoPE、attention mask） |
| `contiguous` | fused kernel 输入前保证 stride 连续性，chunked prefill 分块后整理内存 |
| `permute` / `transpose` | MoE token 重排（expert-contiguous layout）；attention 头维度转置 |

**注意**：过多的 `view/contiguous` 调用在 AMD/NPU 平台上会产生大量 `direct_copy_kernel` 时间（已知占据 decode ~45% GPU 时间），需要融合消除。

### 2.3 数学计算

| Op | 语义 | DeepSeek V4 使用场景 |
|----|------|---------------------|
| `torch.square()` + `.mean()` + `torch.rsqrt()` | RMSNorm 链 | HC pre-norm fallback；q/k norm |
| `torch.sum(dim=...)` | 降维求和 | mHC post 的加权混合：`sum(dim=1)` across `hc_mult` |
| `F.softplus()` + `torch.sqrt()` | **SqrtSoftplus 路由** | MoE gate 计算 expert affinity `sqrt(softplus(x @ W_gate))` |
| `torch.sigmoid()` | 旧版 MoE gate（V3） | Hash-MoE 层保留 sigmoid 兼容 |
| `torch.clamp(min=, max=)` | **SwiGLU clamping** | gate: `clamp(max=10.0)`; up: `clamp(-10.0, 10.0)` |
| `torch.relu()` | Lightning Indexer activation | 压缩 token score 过滤 |
| `torch.topk(k=)` | **MoE 专家选择**、**Lightning Indexer 压缩块选择** | `topk(scores, k=6)` 选 expert; `topk(scores, k=512)` 选压缩块 |

### 2.4 线性代数

| Op | DeepSeek V4 用途 | 后端 |
|----|-----------------|------|
| `F.linear(input, weight)` | **通用投影**：gate/up/down projection、router projection `x @ W_gate.T`、HC head/combine dense | PyTorch fallback → DeepGEMM fp8 gemm 替代 |
| `torch.einsum("tgd,grd->tgr", ...)` | `wo_a` grouped output projection（MQA grouped matmul） | PyTorch → DeepGEMM grouped gemm |
| `torch.bmm` | batch matmul：attention score 计算、CSA compressor grouped KV | Triton fused kernel 替代 |
| `torch.matmul` / `@` | 通用矩阵乘 | DeepGEMM/TileLang fused kernel 替代 |
| `torch.addmm` | `C + A @ B` 形式的 bias 加 GEMM | DeepGEMM `D = C + A @ B` 融合 |

### 2.5 索引/切分/掩码

| Op | 用途 |
|----|------|
| `tensor_split` / 切片 | TP/CP token 按 rank 切分；expert weight 按 EP rank 分片 |
| `advanced indexing` | MoE token dispatch：`tokens[expert == e]` gather to expert；CSA compressor overlap 窗口索引 |
| `masked_fill_()` | SWA window index 过滤非法偏移；attention causal mask |
| `gather` / `scatter_` | MoE token 分发与收集；KV cache 更新 |
| `index_select` | page table 查表构造 attention block |
| `// page_size` 运算符 | page table 中 token index → page index 转换 |

### 2.6 拼接/填充

| Op | 用途 |
|----|------|
| `torch.cat` | TP/EP 结果拼接：`all_gather` 后 `cat`；CSA compressor kv_state 更新追加 |
| `F.pad` | CUDA graph replay 固定形状 padding；sequence padding in chunked prefill |
| `torch.stack` | 多 rank metadata 堆叠 |

### 2.7 设备/Dtype 转换

| Op | 用途 |
|----|------|
| `.to(dtype)` | metadata int32 ↔ int64；中间结果 bf16 ↔ fp32 |
| `.to(device)` | CPU pinned memory → GPU（metadata 上传） |
| `.float()` / `.bfloat16()` | HC fallback 精度提升；output projection 回 bf16 |
| `.contiguous()` | fused kernel 输入要求 |

### 2.8 元素级 / Low-level

| Op | 用途 |
|----|------|
| `torch.mul` / `*` | gate * up (SwiGLU 的 element-wise 部分)；scaling factor 应用 |
| `torch.add` / `+` | 残差加 (被 mHC 替代后极少)；bias 加 |
| `torch.div` / `/` | normalization；scaling |
| `copy_()` | Symmetric buffer 数据拷贝（MegaMoE 输入）、KV cache 更新 |
| `F.silu(gate) * up` | **SwiGLU activation** (fused in MegaMoE kernel) |

---

## 3. DeepSeek V4 推理全链路 Op 流

### 3.1 单层 Decode 路径 (简化)

```
Input hidden_states [B, S, hc_mult, D]

┌── HC Pre ──────────────────────────────────────────┐
│ 1. RMSNorm (torch.square + mean + rsqrt)  ← fused   │
│ 2. FC_hc_fn (F.linear)                     ← DeepGEMM│
│ 3. Sinkhorn projection                    ← TileLang │
│ 4. MulSum: stream mixing across hc_mult   ← fused   │
└─────────────────────────────────────────────────────┘

┌── Attention (CSA or HCA) ──────────────────────────┐
│ 5. q_proj, k_proj, v_proj, q_norm, k_norm           │
│ 6. Compressor: kv_state update, softmax pool, RoPE  │
│ 7. Lightning Indexer: score matmul, ReLU, TopK      │
│ 8. Sparse Attention: Q @ compressed KV (top-k sel)  │
│ 9. Sliding Window Attention: Q @ recent KV          │
│10. wo_a grouped projection (einsum / DeepGEMM)      │
└─────────────────────────────────────────────────────┘

┌── HC Post ─────────────────────────────────────────┐
│11. FC_hc_fn (F.linear)                      ← DeepGEMM│
│12. MulSum: stream mixing across hc_mult     ← fused │
└─────────────────────────────────────────────────────┘

┌── MoE (Hash or Top-k) ─────────────────────────────┐
│13. Router: fc_router + sqrt(softplus) + topk       │
│14. Token Permute → expert-contiguous layout        │
│15. Gate GEMM: x @ W_gate.T  ┐                      │
│16. Up GEMM:   x @ W_up.T    ├→ SwiGLU fused       │
│17. Down GEMM: act @ W_down.T┘  (MegaMoE fuses all) │
│18. Shared Expert: SwiGLU (parallel)                │
│19. Token Unpermute + Weighted Combine              │
└─────────────────────────────────────────────────────┘

Output hidden_states [B, S, hc_mult, D]
```

**关键融合点**：
- HC Pre/Post: `RMSNorm + FC + Sinkhorn + MulSum` → 单一 tilelang kernel
- Attention: `Compressor + Indexer + SparseAttention` → fused kernel 链
- MoE: `EP Dispatch + Gate GEMM + Up GEMM + SwiGLU + Down GEMM + EP Combine` → MegaMoE 单一 kernel

---

## 4. MoE 路由与计算 Op 深度分析

### 4.1 MoE 路由（Gating）

**V4 路由公式**（替代 V3 的 Sigmoid）：

```
affinity = sqrt(softplus(hidden @ W_gate.T + bias))
selected_experts = topk(affinity, k=6)
weights = softmax(selected_affinity)  # renormalize
```

**PyTorch 等效实现**：
```python
# 步骤 1: 路由投影 (F.linear, 用 deepgemm fp8 gemm 加速)
router_logits = F.linear(hidden_states, W_gate)  # [B, S, N_experts]

# 步骤 2: SqrtSoftplus 激活
affinity = torch.sqrt(F.softplus(router_logits))  # square + mean + rsqrt 链

# 步骤 3: top-k 选择
topk_weights, topk_indices = torch.topk(affinity, k=6, dim=-1)

# 步骤 4: renormalize
topk_weights = F.softmax(topk_weights, dim=-1)
```

**两种 MoE 层类型**：

| 层类型 | 路由方式 | 专家权重来源 | 涉及 Op |
|--------|---------|-------------|---------|
| `hash_moe` (前 3 层) | 静态 `token_id → expert_id` 查表 | 预计算 `tid2eid` mapping | `index_select`, gather |
| `moe` (其余层) | SqrtSoftplus + TopK 动态路由 | 可学习 gate weight + `e_score_correction_bias` | `F.linear`, `softplus`, `sqrt`, `topk`, `softmax` |

**辅助损失免除策略**：使用 `e_score_correction_bias` buffer，在 top-k argmax 上 bias 而不经过梯度流。

### 4.2 MoE 计算（Expert FFN）

```
传统 PyTorch 实现（~7 个 kernel launch）：

Permute          → torch.gather / scatter    (kernel 1)
Gate GEMM        → F.linear / @              (kernel 2)
Up GEMM          → F.linear / @              (kernel 3)
SwiGLU           → silu(gate) * up           (kernel 4 - element-wise)
Down GEMM        → F.linear / @              (kernel 5)
Unpermute        → scatter / index_add       (kernel 6)
Scale & Combine  → mul + add                 (kernel 7)
```

**SGLang 生产实现 → MegaMoE 单一 kernel launch**：

| Fused 组件 | 原 PyTorch Op | MegaMoE 实现方式 |
|-----------|--------------|-----------------|
| EP Dispatch | `all_to_all` / DeepEP | NVLink symmetric buffer + one-sided put |
| Gate+Up GEMM | 2× `F.linear` | 单次 UMMA: FP8 activations × FP4 interleaved gate/up weights |
| SwiGLU | `silu(gate) * up` + clamp | 在寄存器中执行；TMEM 中间结果直接 requant → FP8 |
| Down GEMM | `F.linear` | UMMA: requantized FP8 × FP4 down weights |
| EP Combine | `all_to_all` / DeepEP | Symmetric buffer write-back |

**Warp 特化架构（SM100）**：
- **Dispatch Warps**: 通过 NVLink 搬运 token 数据到 SymBuffer
- **MMA Warps**: 执行 Linear1/Linear2 的 UMMA 指令
- **Epilogue Warps**: SwiGLU + FP8 requant + EP combine

**精度变体**：

| 方案 | 激活 | 权重 | 吞吐(Pro EP8 bs=512) |
|------|------|------|---------------------|
| W4A8 (default) | FP8 | FP4 | ~370 µs |
| W4A4 MegaMoE | FP4 (MXF4) | FP4 | 1.54× faster, negligible accuracy loss |

### 4.3 Expert 并行 (EP) 通信 Op

```
EP8 配置下 DeepSeek V4-Pro 单层 MoE 的数据流：

Rank 0 (experts 0-47)     Rank 1 (experts 48-95)    ... Rank 7
    │                          │
    ├─ tokens for expert 3 →──┼──→ NVLink one-sided put
    ├─ tokens for expert 50 ←─┤
    │                          │
    ├─ Linear1 (FP8×FP4 GEMM)   ├─ Linear1
    ├─ SwiGLU + requant         ├─ SwiGLU + requant
    ├─ Linear2 (FP8×FP4 GEMM)   ├─ Linear2
    │                          │
    ├─ output for token 0 ←────┤
    ├─ output for token 1 →────┼──→ NVLink write-back
```

---

## 5. Attention Op 体系

### 5.1 CSA (Compressed Sparse Attention) — m=4 压缩

```
输入 hidden_states [B, S, D]

┌── Compressor ──────────────────────────────────────┐
│ 1. q/k/v linear proj (DeepGEMM fp8 gemm)           │
│ 2. q_norm, k_norm (RMSNorm: square+mean+rsqrt)      │
│ 3. kv_state update (滑动窗口追加, memory copy)       │
│ 4. Softmax Pool: 每 m=4 token 压缩 1 个 KV entry    │
│    → softmax(score) @ KV within window             │
│ 5. RoPE apply to compressed KV                     │
│ 6. InvRoPE on Q（部分维度）                          │
└─────────────────────────────────────────────────────┘

┌── Lightning Indexer ───────────────────────────────┐
│ 7. Score matmul: Q @ compressed KV (bmm)           │
│ 8. ReLU + ReduceSum                                │
│ 9. TopK(k=512): 选择高相关压缩块                    │
└─────────────────────────────────────────────────────┘

┌── Sparse Attention ────────────────────────────────┐
│10. Q @ selected top-k compressed KV (sparse bmm)    │
│11. Q @ sliding window KV (recent 128 tokens)        │
│12. Attention sink: 可学习 logit bias                │
│13. wo_a grouped output: einsum/gemm per group       │
└─────────────────────────────────────────────────────┘
```

**CSA 涉及的 PyTorch Op + 融合替代**：

| 原始 Op | 融合后 |
|---------|--------|
| `q_proj, kv_proj` (2× `F.linear`) | **fused_qkv_a_proj**: 单次 DeepGEMM gemm |
| `kv_state update` (copy_, cat) | **Fused Compressor** |
| `softmax pool` (softmax + bmm) | **Triton fused softmax pool kernel** |
| `RoPE apply` (sin/cos multiply) | **Inplace Partial Rotary Mul** |
| `score matmul` (bmm) + `ReLU` + `ReduceSum` + `TopK` | **Fused Lightning Indexer** |
| `sparse bmm` + `SWA bmm` + `sink` | **SparseAttentionSharedKV** |

### 5.2 HCA (Heavily Compressed Attention) — m′=128 压缩

- 更大的压缩窗口（128 token → 1 entry），**无 Lightning Indexer**
- 全部压缩 KV 参与 dense attention
- 重点 Op：Compressor softmax pool（大窗口）→ dense attention score matmul → wo_a grouped projection

### 5.3 MQA (Multi-Query Attention) 相关 Op

```
q_padded [total_heads, head_dim]  # TP-sharded padding
KV cache (compressed + SWA)
→ mqa_logits = q @ KV.T  (DeepGEMM MQA logits kernel)
→ attention output
→ wo_a: einsum("tgd,grd->tgr", output, w_o_a)
```

**FlashMLA**：MLA decode 专用 sparse attention kernel。SM120 无 tmem → Triton FlashMLA fallback (3.1-5.4× vs FlashInfer)。

### 5.4 RoPE (Rotary Position Embedding)

涉及 Op：`sin/cos 预计算`、`torch.split`（部分维度切片）、`element-wise mul+add`（旋转）、**inplace 更新**。

SGLang 融合方向：**Inplace Partial Rotary Mul** — 单 kernel 完成 slice + rope + inplace write。

---

## 6. mHC (Manifold-Constrained Hyper-Connections) Op 体系

### 6.1 HC Pre

```
输入: hidden_states [B, S, hc_mult=4, D]

PyTorch fallback:
    x = flatten(1).float()          # [B*S, 4*D]
    x = square().mean().rsqrt()      # RMS
    x = F.linear(x, fc_hc_fn)       # 混合矩阵
    x = unsqueeze(-2)               # 恢复 hc_mult 维度
    x = sum(dim=1, keepdim=True)    # 流归并

生产融合 (tilelang kernel):
    单一 kernel: RMSNorm + FC + Sinkhorn + MulSum
```

### 6.2 HC Post

```
输入: hidden_states + attention_output

PyTorch fallback:
    → fc_hc_fn dense (F.linear)
    → MulSum (element-wise weighted sum across streams)

生产融合 (tilelang kernel):
    单 kernel: FC + MulSum + (可选 RMSNorm+FP8 Quant)
```

### 6.3 Sinkhorn 投影

mHC 的核心数学保证：`comb` 矩阵通过 **Sinkhorn-Knopp 迭代** 投影到 Birkhoff polytope（doubly-stochastic，每行每列和为 1），保证 spectral norm ≤ 1（非扩张）。

```
comb_matrix = sinkhorn(raw_matrix, iters=hc_sinkhorn_iters)  # 纯矩阵运算
```

**Op 链**：`F.linear` (raw) → 迭代 `row_norm + col_norm` (element-wise div) → doubly-stochastic matrix。

---

## 7. 融合 Kernel 生态

### 7.1 DeepGEMM — Hopper/Blackwell 通用 GEMM 库

| 功能 | 覆盖的 PyTorch Op | 精度 |
|------|------------------|------|
| `fp8_gemm_nt` (dense) | `F.linear`, `torch.matmul` | FP8 |
| `m_grouped_fp8_gemm_nt_contiguous` | MoE grouped GEMM (prefill: 各 expert 不同 token 数) | FP8 |
| `m_grouped_fp8_gemm_nt_masked` | MoE grouped GEMM (decode: CUDA graph, mask 标记有效 token) | FP8 |
| `fp8_fp4_gemm_nt` (SM100) | `F.linear` (FP4 weight, FP8 act) | FP8×FP4 |
| `fp8_fp4_mqa_logits` | `q @ KV.T` | FP8/FP4 |
| `fp8_fp4_mega_moe` | **MoE 全链路融合** | FP8×FP4 |
| `hc_prenorm_gemm` | RMSNorm + FC (HC Pre) | TF32 |
| `hc_post_gemm` | FC + MulSum (HC Post) | TF32 |

### 7.2 Fused Kernel 列表（来自 SGLang NPU Day-0 PR）

| Fused Kernel | 融合的 Op | 开关 |
|-------------|----------|------|
| **Compressor** | qkv linear + kv_state update + score_state update + softmax pool + compressed_kv | `USE_FUSED_COMPRESSOR=1` |
| **Lightning Indexer** | Score bmm + ReLU + ReduceSum + TopK (pipelined) | `LI_KV_DTYPE_INT8=1` |
| **SparseAttentionSharedKV** | Sparse attention + Window attention (decode + prefill) | `USE_PA_DECODE=1`, `USE_PA_PREFILL=1` |
| **HC Pre** | RMSNorm + FC + Sinkhorn + MulSum | `USE_FUSED_HC_PRE_ASCENDC=1` |
| **HC Post** | FC + MulSum + (opt) RMSNorm+FP8Quant | `USE_FUSED_HC_POST_ASCENDC=1` |
| **MoE Gating TopK** | SqrtSoftplus + Hash/TopK expert selection | `USE_NPU_MOE_GATING_TOP_K=1` |
| **Transpose BatchMatmul** | Batch matmul + transpose | `USE_FUSED_TRANSPOSE_BATCHMATMUL=1` |
| **Inplace Partial Rotary Mul** | Slice + partial RoPE + inplace update | `USE_ROPE_PARTIAL_IN_PLACE_ASCENDC=1` |
| **Fused SiLU+clamp+FP8 quant** | SwiGLU activation + clamping + FP8 requant | fused in MegaMoE / sgl-kernel |
| **W4A4 MegaMoE** | EP dispatch + L1 GEMM + SwiGLU + requant + L2 GEMM + EP combine | `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1` |

### 7.3 MoE Backend 矩阵

| Backend | 适用硬件 | 精度 | 支持并行 |
|---------|---------|------|---------|
| `marlin` | Hopper (H100/H200) | W4(MXFP4)A16 | TP only |
| `flashinfer_mxfp4` | Hopper | W4(MXFP4)A16 | TP only |
| `flashinfer_cutedsl` | Blackwell | W4(MXFP4)A16 | EP+TP |
| `flashinfer_trtllm` | Hopper/Blackwell | FP8/FP4 | EP+TP |
| `megamoe` (DeepGEMM) | Blackwell (SM100) | W4A8 / W4A4 | EP+TP |

---

## 8. 通信与同步 Op

### 8.1 GPU-GPU 通信 Op

| 通信原语 | 用途 | 后端 |
|---------|------|------|
| **TP all-reduce / all-gather** | Tensor 并行结果聚合 | NCCL / TRT-LLM allreduce fusion |
| **EP all-to-all (token dispatch)** | Expert 并行 token 分发与收集 | DeepEP → MegaMoE SymBuffer (NVLink 直连) |
| **DP all-reduce** | Data 并行 attention 聚合 | NCCL |
| **Context Parallel (CP)** | 长序列分片 attention | ring attention / Ulysses |

**DeepEP → MegaMoE 演进**：
- DeepEP: `all_to_all` 语义，通过 NVLink 交换 token
- MegaMoE: 取消独立通信 kernel，dispatch/combine 融合进 GEMM kernel 内部，通过 SymBuffer + NVLink barrier 实现通信-计算 overlap

### 8.2 CPU-GPU 同步热点 (Sync & CPU Op)

SGLang DeepSeek V4 中会引入 host sync 的关键调用：

| 调用 | 位置 | 风险 | 说明 |
|------|------|------|------|
| `torch.cuda.synchronize()` | `set_embed_and_head` | 阻塞全部 stream | embedding/head 替换后保证内存一致 |
| `.tolist()` | metadata 构造 | GPU→CPU 拷贝 + sync | 获取 sequence length list 用于 Python 循环 |
| `.item()` | padding value, max_seq_len | 标量同步 | GPU scalar → Python int |
| `.cpu()` | compressor plan 生成 | D2H + sync | online c128 host-side plan |
| `torch.tensor(..., pin_memory=True).to(device, non_blocking=True)` | metadata 上传 | 异步但 stream 依赖 | Python list → GPU (pinned) |

**规避策略（SGLang 已实现）**：
1. `seq_lens_cpu` 预计算并保持 CPU mirror → 避免 GPU `tolist()`
2. CUDA graph capture/replay 路径复用固定 metadata tensor → 避免 per-step 构造
3. `SGLANG_PREP_IN_CUDA_GRAPH` 将 metadata 准备也纳入 graph
4. `non_blocking=True` + 显式 stream 管理 → 避免默认流隐式同步

**SM120 CUDA Graph 兼容性问题**：SM120 原 fallback 路径中 `.item()`, `.unique()`, `.nonzero()` 会 break graph → PR #24692 用 vectorized gather + `masked_fill` 替代。

---

## 9. 性能分析与瓶颈

### 9.1 已知热点（AMD MI300X 平台 profile）

来自 SGLang PR #23608 的 decode 阶段 5 步 profile：

| 排名 | Kernel | 耗时 (TP=4, ms) | 调用次数 | GPU% |
|-----|--------|---------------|---------|------|
| 1 | `direct_copy_kernel` | 513.2 | 5,880 | ~45% |
| 2 | `direct_copy_kernel` (nocast) | 191.5 | 3,365 | ~17% |
| 3 | `BinaryFunctor` (bf16 mul/clamp) | 158.8 | 2,940 | ~14% |
| 4 | `_gemm_a8w8_blockscale_kernel` | 46.5 | 860 | ~4% |
| 5 | `fused_moe_kernel` | 9.5 | 430 | ~0.8% |

**结论**：Top 3 kernel = ~77% GPU 时间全在 bf16 element-wise copy/mul，每 token 1,849 次 bf16 拷贝 + 588 次 bf16 乘法。真正的计算型 kernel（GEMMs + MoE + softmax）仅占 ~6%。**瓶颈在 memory bandwidth，不是 compute**。

### 9.2 MoE 性能（Blackwell B200, EP8）

| 模型 | Batch Size | 耗时 (µs) | Compute (TFLOPS) | Memory BW (GB/s) | NVLink BW (GB/s) |
|------|-----------|-----------|-----------------|------------------|-------------------|
| V4-Flash | 1 token | 37.3 | 4 | 3192 | 1 |
| V4-Flash | 512 token | 88.1 | 534 | 4560 | 126 |
| V4-Flash | 8192 token | 823.3 | 914 | 1056 | 289 |
| V4-Pro | 1 token | 108.1 | 7 | 1758 | 1 |
| V4-Pro | 512 token | 369.6 | 1098 | 4619 | 182 |
| V4-Pro | 32768 token | 10655.2 | 2438 | 692 | 417 |

**MegaMoE vs Legacy speedup**: bs=1 时 1.61×; bs=512 时 1.54×; bs=8192 时 1.50×。

---

## 10. Op 融合优化路线图

### P0 已实现
- [x] MegaMoE: EP dispatch + L1 + SwiGLU + L2 + EP combine → 单一 kernel
- [x] HC Pre/Post: RMSNorm + FC + Sinkhorn + MulSum → tilelang kernel
- [x] SwiGLU clamp + FP8 requant fusion
- [x] fused_qkv_a_proj (DeepGEMM gemm 合并)
- [x] Fused SiLU+clamp+FP8 quant kernel

### P1 进行中/建议
- [ ] RMSNorm + RoPE fusion (for q)
- [ ] q_norm + k_norm fusion
- [ ] Compressor: kv-update + ape-Add + score-update 融合
- [ ] Compression: Softmax + MulSum + RoPE (+ CacheUpdate) 融合
- [ ] MoE Gating: softplus + sqrt + biasAdd + Top6 + Gather + Norm + Mul 融合
- [ ] FC_hc_fn 小 GEMM 专用 kernel (tiny gemmN 优化)
- [ ] 消除 MoE 和 mHC_post 之间的 "Fill" 和 "memcpy"

### P2 长期
- [ ] `direct_copy_kernel` 消除：slice-and-copy ladder → fused Triton copy in MHC path
- [ ] HCA Compressor CUDA graph 兼容（decode 阶段）
- [ ] MQA 直接读 compressed KV cache + SWA cache（避免 copy/concat）
- [ ] InvRoPE 单 kernel
- [ ] PP (Pipeline Parallelism) + PD (Prefill-Decode Disaggregation) 全链路

---

## 11. 设计原则总结

### 分层架构

```
┌─────────────────────────────────────────────┐
│  Python 层 (deepseek_v4.py)                  │
│  → Tensor 形状管理、metadata 构造、调度逻辑    │
│  → Op: empty, view, cat, arange, indexing    │
├─────────────────────────────────────────────┤
│  PyTorch Fallback 层                          │
│  → 正确性验证 + 新硬件启动兼容                 │
│  → Op: F.linear, einsum, softmax, rsqrt     │
├─────────────────────────────────────────────┤
│  Fused Kernel 层 (Triton/TileLang/DeepGEMM)  │
│  → 生产路径，op 融合 + 精度优化                │
│  → Kernel: fp8_gemm, m_grouped_gemm, HC    │
├─────────────────────────────────────────────┤
│  Mega Kernel 层 (SM100 CUDA)                  │
│  → 跨算子、跨 GPU 的端到端融合                │
│  → Kernel: MegaMoE (dispatch+GEMM+combine)  │
└─────────────────────────────────────────────┘
```

### 核心原则

1. **大计算走 fused kernel，PyTorch 做胶水** — GEMM 类 op 不应出现在热路径 PyTorch
2. **CPU 边界最小化** — `.item()`, `.tolist()`, `.cpu()` 是性能杀手，优先 CPU mirror + tensor 化 metadata
3. **融合优于精度损失** — 减少 kernel launch 数 > 微调单个 kernel 的 FLOP 利用率
4. **PyTorch fallback 是必需的** — 正确性 fallback + 新硬件 bring-up 的基础（如 SM120、AMD、NPU）
5. **内存带宽是瓶颈** — DeepSeek V4 decode 阶段的 element-wise copy/mul 占 ~77% GPU 时间，真正的 GEMM compute 仅 ~6%

---

## 参考资料

- [DeepSeek-V4 Technical Report](https://arxiv.org/abs/2504) — CSA/HCA, mHC, Hash-MoE
- [DeepSeek-V4 HuggingFace](https://huggingface.co/deepseek-ai) — Model config & inference code
- [SGLang DeepSeek-V4 Cookbook](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4) — Serving recipes
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) — FP8/FP4 GEMM library with MegaMoE
- [FlashMLA](https://github.com/deepseek-ai/FlashMLA) — MLA sparse decode kernel
- [SGLang PR #23882](https://github.com/sgl-project/sglang/pull/23882) — Day-0 DeepSeek V4 support
- [SGLang PR #24047](https://github.com/sgl-project/sglang/pull/24047) — SM120 support
- [SGLang DSV4 Tracker #23666](https://github.com/sgl-project/sglang/issues/23666) — Performance optimization checklist
- [SGLang PR #23598](https://github.com/sgl-project/sglang/issues/23598) — NPU Day-0 with fused kernel list
