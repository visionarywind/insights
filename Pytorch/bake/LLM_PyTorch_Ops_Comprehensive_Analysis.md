[TOC]

---

# LLM 常用 PyTorch Op 技术洞察 — 场景、替代方案与框架实践

**分析日期**: 2026-05-21
**覆盖框架**: SGLang、vLLM、Megatron-LM
**范围**: LLM 推理 & 训练中的核心 PyTorch op 生态

---

## 1. 总览：LLM 计算栈中的 PyTorch Op 分层

```
┌──────────────────────────────────────────────────────────┐
│  Layer 4: 框架组合层 (SGLang/vLLM/Megatron)               │
│  → 将 op 编排为 forward graph，管理调度 & 并行            │
├──────────────────────────────────────────────────────────┤
│  Layer 3: Fused Kernel 层 (FlashAttention, DeepGEMM,      │
│           Apex, TransformerEngine, xFormers)              │
│  → 将多个 PyTorch op 融合为单一 CUDA/Triton kernel       │
├──────────────────────────────────────────────────────────┤
│  Layer 2: PyTorch 原生 op 层                              │
│  → torch.matmul, F.linear, F.layer_norm, F.softmax ...  │
├──────────────────────────────────────────────────────────┤
│  Layer 1: PyTorch 基础 op 层                              │
│  → view, empty, cat, index_select, arange, copy_ ...    │
└──────────────────────────────────────────────────────────┘
```

**核心设计原则**: 大计算走 Layer 3 fused kernel；PyTorch 原生 op 主要用作正确性 baseline 和 fallback；Layer 1 基础 op 做 tensor 生命周期管理和形状变换。

---

## 2. 按功能分类的 Op 深度分析

### 2.1 线性代数 Op（GEMM 类）

这是 LLM 中**计算量最大**、**优化最激进**的 op 类别。

| Op | 场景 | 解决的问题 | 替代方案 | SGLang | vLLM | Megatron |
|----|------|----------|---------|--------|------|----------|
| `F.linear(x, W)` | Q/K/V/O 投影、FFN gate/up/down、router | 通用矩阵乘，支持 bias、自动 batching | DeepGEMM FP8、TRT-LLM gemm | DeepGEMM / TileLang / MuDNN (MUSA) | CUTLASS / FlashInfer TRT-LLM / Marlin | TE linear / fused attention |
| `torch.matmul(a, b)` | Attention score: `Q @ K^T` | 通用 2D 矩阵乘 | FlashAttention (内嵌) | FlashMLA / DeepGEMM MQA logits | FlashInfer / CuteDSL | TE dot_product_attention |
| `torch.bmm(a, b)` | Batch attention: 每 head 独立计算 | 3D batch 矩阵乘，比 matmul + reshape 快 | FlashAttention 批量处理 | FlashMLA batched | FlashAttention-v2 batched | TE multi-head attention |
| `torch.einsum("tgd,grd->tgr", ...)` | MQA grouped output projection | 自定义维度收缩，可读性强但慢 | DeepGEMM grouped gemm | DeepGEMM m_grouped_gemm | vLLM grouped_gemm | TE fused |
| `torch.addmm(C, A, B)` | `bias + A @ B` 融合，减少一次 add launch | 减少 kernel launch | DeepGEMM `D = C + A @ B` 内嵌 | DeepGEMM fp8_gemm | CUTLASS epilogue | TE gemm + bias |

**框架使用模式**:

```
SGLang:
  F.linear → deep_gemm.fp8_gemm_nt (Hopper/Blackwell)
          → sgl_kernel.musa_fused_gemv (MUSA)
          → tilelang hc_pre_norm_fn_fwd_mul (TileLang fallback)
  torch.einsum → deep_gemm.m_grouped_fp8_gemm_nt_contiguous

vLLM:
  F.linear → vllm.cutlass_scaled_mm (FP8)
          → marlin_gemm (W4A16)
          → flashinfer_trtllm (W4A8 MoE)
          → torch.compile + inductor (fallback fusion)

Megatron:
  F.linear → TE linear (FP8, with delayed scaling)
          → column parallel linear (TP sharding: f → f/tp)
          → row parallel linear (TP gathering: f/tp → f)
```

---

### 2.2 Attention Op

Attention 是 LLM 中**访存密集度最高**的 op，决定了 KV cache 大小和 decode 延迟。

| Op | 场景 | 解决的问题 | 替代方案 | 框架实现 |
|----|------|----------|---------|---------|
| `F.scaled_dot_product_attention` | 标准 attention：QK^T → softmax → PV | PyTorch 2.0+ 官方 SDPA，自动选最优 backend | FlashAttention-2/3, xFormers | vLLM/SGLang 都绕过了它，直接用 FlashInfer/FlashMLA |
| `F.softmax(scores, dim=-1)` | Attention score 归一化 | 传统 softmax（数值不稳定，需 max-subtract） | **Online softmax**（FlashAttention 内嵌） | FlashAttention 和 FlashMLA 内嵌实现 |
| `scores @ V` (bmm/matmul) | Attention 输出的 value 投影 | 大矩阵乘，O(S^2 × D) | FlashAttention 分块 tiling（O(S×D) 访存） | vLLM paged_attention v1/v2 |
| `Q @ K^T` (bmm/matmul) | Attention score 计算 | O(S^2) 计算 + 访存 | FlashAttention MQA/GQA 优化 | FlashMLA (DeepSeek) / FlashInfer |
| KV cache 读写 | KV state 存储/更新 | PageAttention 管理物理页 | **PagedAttention** (vLLM 原创) | SGLang RadixAttention, vLLM PagedAttention v2 |

**关键替代技术 — FlashAttention 家族进化**:

| 版本 | 优化技术 | 解决的问题 |
|------|---------|----------|
| FlashAttention-1 | Tiling + online softmax + recomputation | 减少 HBM 读写，O(N²) → O(N²d²/M) |
| FlashAttention-2 | 减少 non-matmul FLOPs, parallel over seqlen | 接近理论峰值利用率 |
| FlashAttention-3 (Hopper) | Async wgmma, TMA, FP8 | Hopper 硬件特性 |
| FlashMLA (DeepSeek) | MLA sparse decode, compressed KV | DeepSeek MLA 特化 |
| FlashInfer | 统一 prefill/decode 接口，block sparse | 减少算子碎片化 |

**SGLang 的特殊路径**:
```
DeepSeek V4 Attention = CSA (压缩稀疏) + HCA (重度压缩) + SWA (滑动窗口)
  ├── Compressor: kv_state update + softmax pool + RoPE → fused kernel
  ├── Lightning Indexer: score matmul + ReLU + TopK → fused kernel  
  └── SparseAttentionSharedKV: sparse + window attention → fused kernel
```

---

### 2.3 归一化 Op

| Op | 场景 | 解决的问题 | 替代方案 | SGLang | vLLM | Megatron |
|----|------|----------|---------|--------|------|----------|
| `F.layer_norm(x)` | Pre-norm / post-norm（GPT/LLaMA 风格） | 训练稳定，但需要计算 mean+var | **RMSNorm** (更高效) | fused_rms_norm | fused_rms_norm (vllm custom op) | TE fused layer_norm |
| RMSNorm: `x * rsqrt(mean(x²) + ε)` | LLaMA/DeepSeek/Mistral 等主流模型 | 比 LayerNorm 少算 mean，快 ~15% | **Fused RMSNorm** (单 kernel) | TileLang fused HC pre-norm | vllm.fused_rms_norm | TE rmsnorm_fwd |
| `F.group_norm(x)` | ViT / 多模态 | 跨 channel 归一化 | 较少用于纯 LLM | - | vllm multimodal | Megatron VLM 路径 |
| `F.batch_norm` | 传统 CV 模型 | 训练时 batch 统计 | 在 LLM 中基本不用 | - | - | - |

**RMSNorm 的替代演变**:

```
PyTorch 手动实现 (3 kernel launch):
  x_sq = x.pow(2).mean(dim=-1, keepdim=True)   # kernel 1
  x_norm = x * torch.rsqrt(x_sq + eps)          # kernel 2
  return x_norm * weight                         # kernel 3

↓

Fused RMSNorm (1 kernel launch):
  return torch.ops.vllm.rms_norm(x, weight, eps)
  // 或 torch.ops.sgl_kernel.rmsnorm(x, weight, eps)
```

**SGLang DeepSeek V4 的 HC Pre Norm**:
```
HC Pre = RMSNorm + FC_hc_fn + Sinkhorn + MulSum → tilelang 单 kernel
  → 将 4 个独立 PyTorch op 融合为 1 个 GPU kernel
```

---

### 2.4 激活函数 Op

| Op | 场景 | 解决的问题 | 替代方案 | 框架实现 |
|----|------|----------|---------|---------|
| `F.silu(x)` | Swish/SiLU 激活（LLaMA, DeepSeek, Mistral） | 平滑非线性，训练效果好 | **SwiGLU: silu(gate) * up** | fused_silu_and_mul |
| `F.gelu(x)` | BERT/GPT-2 时代激活函数 | 平滑 ReLU 替代 | 已基本被 SiLU 取代 | TE gelu (legacy) |
| `F.relu(x)` | Lightning Indexer, GLU 变体 | 稀疏激活 | clamp, leaky_relu | SGLang indexer 使用 |
| **SwiGLU**: `F.silu(gate) * up` | FFN/Expert 的主激活 | silu + gate*up 融合 | **Fused SwiGLU** (单 kernel + FP8 quant) | 所有3个框架都有 |
| `torch.clamp(gate, max=swiglu_limit)` | **DeepSeek V4 SwiGLU clamping** | 防止训练/推理中激活值爆炸 | Fused in MegaMoE | SGLang fused_kernel |
| `F.softplus(x)` | DeepSeek V4 MoE gate (sqrt softplus) | 比 sigmoid 更平滑的 MoE 路由 | sigmoid (V3), softmax | SGLang fused gate-topk |
| `torch.sqrt(x)` | DeepSeek V4 SqrtSoftplus | 与 softplus 组成 V4 路由 | - | SGLang fused MoE gating |

**SwiGLU 融合路径**:

```
PyTorch 手写 (2 kernel launch):
  gate, up = x.chunk(2, dim=-1)   # 可能引入 copy
  output = F.silu(gate) * up       # kernel 1 (silu) + kernel 2 (mul)

↓

Fused SwiGLU (1 kernel launch, shared memory):
  torch.ops.sgl_kernel.silu_and_mul(x, output)
  # gate 和 up 在同一 shared memory tile 中完成 silu + mul

↓

MegaMoE SwiGLU (0 extra kernel):
  # SwiGLU 内嵌在 MegaMoE mega-kernel 的 epilogue 阶段
  # 直接用 TMEM 中间结果做 silu + gate*up + FP8 requant
```

---

### 2.5 位置编码 Op（RoPE 等）

| Op | 场景 | 解决的问题 | 替代方案 | 框架实现 |
|----|------|----------|---------|---------|
| `cos = torch.cos(freqs)`<br>`sin = torch.sin(freqs)` | RoPE 频率预计算 | 1次预计算，后续查表 | 可缓存，避免重复计算 | 所有框架都预计算 |
| `q_rot = q*cos - rotate_half(q)*sin` | RoPE 应用到 Q/K | 位置感知的注意力 | **Fused RoPE** (单 kernel) | SGLang: inplace partial rotary |
| ALiBi (bias 加法) | Attention 位置 bias | 无需显式编码的简单方法 | 已被 RoPE 取代 | 少数 legacy 模型 |
| `apply_rotary_emb(x, cos, sin)` | HuggingFace 标准接口 | 接口统一，但慢 | Fused kernel | vLLM/Megatron fused_rotary |

**RoPE 融合的三种层级**:

```
Level 0 (PyTorch 手写, ~4 kernel launch):
  sin, cos = precompute()
  x_rot = x * cos
  x_pass = rotate_half(x) * sin
  x = torch.cat([x_rot - x_pass, x_pass + x_rot], ...)

Level 1 (Fused RoPE kernel, 1 kernel launch):
  torch.ops.vllm.rotary_embedding(positions, query, key, cos_sin_cache)
  # 在单个 CUDA kernel 中完成所有旋转计算

Level 2 (Inplace Partial RoPE, DeepSeek V4):
  # 只对 Q/K 的部分维度做 RoPE，其余维度 pass through
  # 单 kernel: slice + rotate + inplace write
```

---

### 2.6 通信 Op（分布式并行）

这是**多 GPU 训练/推理**的核心，PyTorch 提供了 NCCL 后端。

| Op | 场景 | 解决的问题 | 替代方案 | SGLang | vLLM | Megatron |
|----|------|----------|---------|--------|------|----------|
| `torch.distributed.all_reduce(tensor)` | TP/DP 归约 | 多 GPU 结果求和/平均 | NCCL, NVSwitch SHARP | DeepEP (token dispatch) | NCCL / TRT-LLM fusion | NCCL (TP/DP/PP 都依赖) |
| `torch.distributed.all_gather(tensor)` | TP forward 收集 | 拼接多个 rank 的输出 | NCCL | SGLang TP all-gather | vLLM TP gather | Megatron TP gather |
| `torch.distributed.reduce_scatter(tensor)` | TP backward 分发 | 先 reduce 后 scatter | NCCL | - | - | Megatron TP backward |
| `torch.distributed.all_to_all(tensor)` | **EP MoE token dispatch** | 按 expert 分发 token | DeepEP NVLink 直连 | MegaMoE SymBuffer (取消独立通信) | DeepEP | Megatron MoE all-to-all |
| `torch.distributed.broadcast(tensor)` | PP 首个 stage 广播 | pipeline 启动 | NCCL P2P | - | vLLM PP | Megatron PP |
| `torch.distributed.send/recv` | PP 层间 P2P 传递 | Pipeline 通信 | NCCL P2P | - | vLLM PP | Megatron PP |

**通信-计算 Overlap 模式**:

```
传统模式 (2 个独立阶段):
  Compute → Sync → AllReduce → Compute
  50% GPU 空闲

Overlap 模式 (SGLang DeepEP / MegaMoE):
  Compute chunk 1 ──→ AllReduce chunk 1 ──→
  Compute chunk 2 ──→ AllReduce chunk 2 ──→
  通过分桶 + 异步启动，GPU 利用率接近 100%
```

**SGLang MegaMoE 的极致融合**:
```
传统 MoE:
  all_to_all (dispatch) → GEMM → all_to_all (combine)
  3 个独立操作，2 次通信同步

MegaMoE (SM100):
  单 kernel: EP dispatch + Linear1 + SwiGLU + Linear2 + EP combine
  SymBuffer + NVLink barrier → 通信与计算在同一 kernel 内 overlap
```

---

### 2.7 量化 Op

| Op | 场景 | 解决的问题 | 替代方案 |
|----|------|----------|---------|
| `x.to(torch.float8_e4m3fn)` | FP8 量化推理 | 直接 cast（需 block scale） | **Fused quant + GEMM** |
| `torch._scaled_mm(a, b, scale_a, scale_b)` | PyTorch 2.4+ FP8 GEMM | 内置 FP8 支持 | CUTLASS scaled_mm |
| `Dequant: W_int4 * scale + zp` | W4A16/W4A8 推理 | INT4 weight 解量化为 FP16/BF16 | **Fused dequant + GEMM** (Marlin) |
| AWQ: `weight * scale_per_channel` | W4A16 通道级量化 | 细粒度 scale 保持精度 | AWQ kernel |
| GPTQ: group-wise quantization | W4A16 分组量化 | 平衡精度与速度 | GPTQ Marlin kernel |
| MXFP4 (Microscaling FP4) | DeepSeek V4 原生 FP4 | E2M1 浮点格式 + shared exponent | DeepGEMM W4A4 MegaMoE |

**量化 op 的框架实现对比**:

```
vLLM:
  AWQ: torch.ops.vllm.awq_gemm (fused dequant + gemm)
  GPTQ: torch.ops.vllm.gptq_gemm (group-wise)
  FP8: torch.ops.vllm.cutlass_scaled_mm
  MXFP4: flashinfer_mxfp4 / flashinfer_cutedsl

SGLang:
  FP4: Marlin (Hopper) / FlashInfer MXFP4 (Hopper) / DeepGEMM (Blackwell)
  W4A4: MegaMoE (DeepGEMM SM100)
  FP8: deep_gemm.fp8_gemm_nt

Megatron:
  FP8: TransformerEngine (TE) linear layers + delayed scaling
  BF16: 原生 autocast + TE fused attention
```

---

### 2.8 采样 Op

| Op | 场景 | 解决的问题 | 替代方案 |
|----|------|----------|---------|
| `torch.topk(logits, k)` | Top-K 采样 | 选 top-k token | **Fused topk + softmax** |
| `torch.multinomial(probs, 1)` | 随机采样 | CPU 端采样慢 | GPU 端采样 kernel |
| `torch.argmax(logits)` | Greedy 解码 | 选最大概率 token | 与 topk 共享 kernel |
| `logits / temperature` | 温度缩放 | 控制输出随机性 | **Fused temperature + topk** |

**框架实践**:

```
vLLM:
  torch.ops.vllm.sample(logits)  # 融合: temperature + topk + top-p + multinomial

SGLang:
  flashinfer.sampling.top_k_top_p_sampling_from_probs
  → MUSA: musa_top_k_top_p_sampling (MUSA native port)

Megatron:
  # 训练时不需要采样；推理时用 vLLM/SGLang 或 HF generate
```

---

### 2.9 Tensor 生命周期管理 Op

这些是**最基础但调用最频繁**的 PyTorch op。虽然单个 op 耗时很小，但数量众多，合计开销不可忽略。

| Op | 场景 | 框架实践 |
|----|------|---------|
| `torch.empty/zeros` | 预分配输出 buffer、KV cache | SGLang: `q_padded = x.new_empty(...)` |
| `view/reshape` | MQA head 维度变换、mHC hc_mult 展开 | 所有框架高频使用 |
| `cat/stack` | TP all-gather 后拼接、chunked prefill 合并 | SGLang: TP 路径 cat |
| `index_select / [:, indices]` | Page table 查表、MoE expert gather | vLLM PagedAttention 核心操作 |
| `permute/transpose` | Attention head 维度重排 | FlashAttention 输入前 |
| `contiguous()` | fused kernel 输入前保证 stride | 所有框架 fuse kernel 前必调 |
| `to(dtype/device)` | 精度转换、设备迁移 | FP16 ↔ FP32 中间精度 |
| `copy_()` | SymBuffer 输入填充、KV cache 更新 | MegaMoE、vLLM block copy |
| `arange(...)` | 生成序列索引、position id | 所有框架 prefill 阶段 |
| `masked_fill_()` | Attention mask、SWA window 过滤 | vLLM/SGLang causal mask |

---

## 3. 三大框架的 Op 使用策略对比

### 3.1 核心哲学

| 维度 | SGLang | vLLM | Megatron-LM |
|------|--------|------|-------------|
| **定位** | 推理 serving | 推理 serving | 训练 |
| **Op 替换激进程度** | 🔴 非常激进（DeepGEMM + MegaMoE + TileLang） | 🟡 中等激进（CUTLASS + FlashInfer + TRT-LLM） | 🟢 稳健（TE + NCCL + Apex） |
| **Kernel 生态** | 自研 DeepGEMM, FlashMLA, sgl-kernel | FlashInfer, Marlin, CUTLASS, TRT-LLM | TransformerEngine, Apex, custom CUDA |
| **支持的硬件** | NVIDIA (Hopper/Blackwell) + MUSA + AMD + NPU | NVIDIA (全系列) + AMD + Intel GPU + TPU | NVIDIA (全系列) |
| **精度方案** | FP4 MoE + FP8 attention + BF16 | FP8 KV cache + W4A16 MoE + BF16 | FP8 training + BF16 + TE |
| **并行策略** | TP + EP + DP + CP + PD disagg | TP + EP + PP + DP | TP + PP + DP + SP |

### 3.2 Op 替换矩阵

```
                     PyTorch 原生        SGLang              vLLM               Megatron
──────────────────────────────────────────────────────────────────────────────────────────
GEMM (dense)         F.linear            deep_gemm.fp8       vllm.cutlass_      TE linear
                                          _gemm_nt            scaled_mm           (FP8/BF16)

GEMM (MoE)           F.linear × N_exp    MegaMoE (fused      fused_moe           TE MoE
                                          dispatch+GEMM)      (Triton)            (all-to-all)

Attention            SDPA                FlashMLA/           FlashInfer/         FlashAttention-3
                                          FlashAttention       FlashAttn-v2        + TE dot_product

RMSNorm              pow+mean+rsqrt      fused kernel        vllm.rms_norm       TE rmsnorm
                                          (TileLang)          (custom op)         (fused)

RoPE                 sin/cos + rotate    Inplace partial     fused_rotary        fused_rotary
                                          (1 kernel)          (1 kernel)          (1 kernel)

SwiGLU               silu(gate)*up       fused silu+mul      fused silu+mul      TE swiglu
                                          (1 kernel)          (Triton)            (fused)

MoE token dispatch   all_to_all          MegaMoE SymBuffer   DeepEP all-to-all   all_to_all
                                          (取消独立通信)       (NVLink)            (NCCL)

FP8 quant            .to(float8)         block quant+GEMM    scaled_mm           TE delayed
                                          (内嵌 DeepGEMM)     (CUTLASS)           scaling

Top-K sampling       torch.topk          flashinfer          vllm fused          (训练不用)
                        + multinomial     topk_top_p           sample
```

---

## 4. Fused Kernel 生态系统

### 4.1 主流融合 Kernel 库对比

| 融合库 | 覆盖范围 | 框架集成 | 硬件支持 |
|--------|---------|---------|---------|
| **FlashAttention** | Attention (prefill + decode) | 所有框架 | NVIDIA (SM80+) + AMD |
| **FlashInfer** | Attention + sampling + GEMM | SGLang, vLLM | NVIDIA (SM80+) |
| **DeepGEMM** | FP8/FP4 GEMM + MoE + HC + MQA | SGLang | NVIDIA Hopper/Blackwell |
| **TransformerEngine (TE)** | FP8 linear + attention + norm | Megatron, vLLM | NVIDIA Hopper+ |
| **CUTLASS** | GEMM template library | vLLM (scaled_mm), FlashInfer | NVIDIA |
| **Marlin** | W4A16 MoE GEMM | vLLM, SGLang | NVIDIA (SM80+) |
| **xFormers** | Attention + fused ops | HF, vLLM (legacy) | NVIDIA + AMD |
| **Apex** | Fused optimizers + norm | Megatron (legacy) | NVIDIA |
| **TileLang** | Cross-hardware kernel DSL | SGLang (MUSA) | MUSA + NVIDIA |
| **Triton** | Cross-vendor kernel language | vLLM, SGLang (fallback) | NVIDIA + AMD + Intel |

### 4.2 融合的典型模式

| 模式 | 融合前 (离散 PyTorch op) | 融合后 |
|------|------------------------|--------|
| **GEMM + Activation** | `linear(x) → silu(gate) * up` | MegaMoE / fused_moe (1 kernel) |
| **Norm + GEMM** | `rms_norm(x) → linear(x)` | DeepGEMM hc_prenorm_gemm |
| **GEMM + Bias + Act + Quant** | `linear → bias → silu → clamp → to(fp8)` | tile_kernels swiglu_forward_and_per_token_cast |
| **Attention (全链路)** | `qk^T → scale → mask → softmax → pv` | FlashAttention (1 kernel, O(N²d) block-fused) |
| **MoE (全链路)** | `router → permute → gate_gemm → up_gemm → swiglu → down_gemm → unpermute` | MegaMoE (1 kernel, NVLink overlap) |
| **Sampling (全链路)** | `logits/temp → softmax → topk → topp → multinomial` | flashinfer top_k_top_p (1 kernel) |

---

## 5. 性能特征分析

### 5.1 Op 类型的性能瓶颈

```
Op 类型          主要瓶颈         优化方向               典型加速
──────────────────────────────────────────────────────────────
GEMM (dense)     Compute bound    Tensor Core (FP8/FP4)   2-4x
GEMM (MoE)       Communication    Expert dispatch overlap 1.5-2x
Attention        Memory bound     Tiling + flash-attn      5-10x
RMSNorm          Memory bound     Fusion with next op      2-3x
RoPE             Memory bound     Fusion + inplace         2-3x
SwiGLU           Memory bound     Fusion + in-register     2-3x
Top-K sampling   Compute bound    Fused temp+softmax+topk  2-4x
AllReduce        Communication    Bucket + overlap         1.5-2x
All-to-All       Communication    NVLink direct (DeepEP)   2-4x
```

### 5.2 SGLang DeepSeek V4 实测瓶颈分布

来自 AMD MI300X 平台 decode 阶段 profile：

```
Kernel                        GPU Time%  瓶颈类型
────────────────────────────────────────────────────
bf16 element-wise copy           45%    Memory BW
bf16 element-wise mul            17%    Memory BW
bf16 BinaryFunctor (mul/clamp)   14%    Memory BW
FP8 GEMM (blockscale)             4%    Compute
Fused MoE kernel                0.8%    Compute
```

**关键洞察**: Decode 阶段 **77% GPU 时间在 memory-bound 操作**（copy/clamp/mul），真正的矩阵乘计算仅占 ~6%。优化的核心方向不是加速 GEMM，而是**减少 element-wise op 的 kernel launch 和 HBM 访问**。

---

## 6. PyTorch Compile (torch.compile) 在 LLM 中的作用

### 6.1 什么时候有用

```
场景                            效果
────────────────────────────────────────
小 token 数 decode (bs=1-8)     ⚡ 显著加速 (融合 element-wise op)
大 token 数 prefill (bs>512)    🟡 中等效果 (GEMM 本已高效)
含大量 view/reshape 路径        ⚡ 显著加速 (消除中间 tensor)
CUDA graph 配合                  ✅ 互补 (graph 消除 launch overhead，compile 融合 op)
```

### 6.2 vLLM 的 torch.compile 集成

vLLM v1 支持 `--compilation-config` 参数使用 `torch.compile`：
- `inductor` backend：自动融合 element-wise op
- 与 CUDA graph capture 配合：graph 负责固定 shape 的 kernel dispatch，compile 负责 op 融合
- MoE 路径有专门的 `Fusion torch.compile passes` 文档

### 6.3 SGLang 的权衡

SGLang 更倾向于**手工融合**（DeepGEMM/TileLang）而非依赖 `torch.compile`：
- 手工融合可以实现跨 op 通信 overlap（MegaMoE NVLink overlap）
- 手工融合可以针对特定 shape 做极致优化
- `torch.compile` 更适合快速原型和兼容性保证

---

## 7. 未来趋势

| 趋势 | 影响 | 涉及 op |
|------|------|---------|
| **FP4 普及** (DeepSeek V4, Blackwell) | GEMM 密度翻倍 | `F.linear` → 直接 FP4 GEMM |
| **Mega-kernel 化** | 多 op 融合为单 kernel | SwiGLU+GEMM, MoE 全链路 |
| **torch.compile 成熟** | 自动融合替代手工 kernel | 各种 element-wise op |
| **torch.export 标准化** | 图导出跨框架部署 | 所有 op |
| **硬件多样化** (MUSA, AMD, NPU) | PyTorch op 作为统一接口 | backend 多态 |
| **Decode-only 优化** | 小 batch 场景的 memory-bound 优化 | Attention decode, element-wise |
| **异步通信原语** | 完全隐藏通信开销 | allreduce, all-to-all |
| **FP8/FP4 KV cache** | KV cache 内存减半 | Attention PE (位置编码) 精度友好 |

---

## 8. 总结：LLM 框架的 Op 使用原则

1. **GEMM 类 op 从不裸用 PyTorch** — 生产路径 100% 替换为 fused kernel（DeepGEMM/CUTLASS/TE/Marlin）
2. **Attention 全部走 FlashAttention 体系** — 标准 SDPA 只在 fallback 场景使用
3. **Norm + Activation + RoPE 走融合路径** — 手工或 compile 融合，减少 memory-bound 瓶颈
4. **PyTorch 的角色是 tensor 生命周期管理 + 正确性 baseline** — 创建/形状变换/索引/dispatch 用 PyTorch，计算用 fused kernel
5. **通信 op 追求 overlap** — 理想情况下通信时间完全被计算掩盖
6. **量化 op 追求 fused dequant + GEMM** — 不在显存中展开量化 weight

---

*本文档覆盖 SGLang (含 DeepSeek V4 + MUSA)、vLLM、Megatron-LM 三个主流 LLM 框架的 PyTorch op 使用实践。*
