# MUSA 适配 SGLang DeepSeek V4 差异项分析

**分析日期**: 2026-05-21
**上游基线**: `/home/mtuser/workspace/Github/sglang/` (upstream SGLang)
**MUSA 适配版**: `/home/mtuser/workspace/sglang/` (MUSA port)

---

## 1. 总体差异概览

MUSA (Moore Threads GPU) 适配版 SGLang 是一个**系统性 GPU 后端移植**，涵盖从 C++ kernel 到 Python 框架层的完整适配。差异分布在 4 个层次：

| 层次 | 差异规模 | 适配策略 |
|------|---------|---------|
| **C++ MUSA Kernel** (`sgl-kernel/csrc/musa/`) | 7 个文件, ~2,200 行 | MUSA 原生 kernel 重写 |
| **Python 硬件后端** (`hardware_backend/musa/`) | 7 个文件 | 新硬件 backend 注册 |
| **DeepSeek V4 MUSA Ops** (`hardware_backend/layers/deepseek_v4_musa/`) | 15 个文件, ~3,000 行 | TileLang + tile_kernels 适配 |
| **SGLang 核心 inline 改动** | ~6 个核心文件, ~300 行新增 | 条件编译 + MUSA 路径注入 |

---

## 2. 层 1: C++ MUSA Kernel 适配 (`sgl-kernel/csrc/musa/`)

MUSA 适配版在 `sgl-kernel` 中新增了独立的 MUSA kernel 源文件目录，包含 7 个文件。

### 2.1 文件清单

| 文件 | 行数 | 功能 | CUDA 对应 |
|------|------|------|----------|
| `moe_gemv_swiglu.mu` | 825 | MoE GEMV + SwiGLU 融合 kernel (BF16/FP16/FP8/W4A16) | `fused_moe_triton/`, `moe_runner/` |
| `top_k_top_p_sampling.mu` | 365 | Top-K/Top-P 采样 kernel | flashinfer `sampling.cuh` |
| `pos_encoding_contiguous.mu` | 247 | Rotary Position Embedding (GPT-NeoX + GPT-J) | `rotary_embedding.py` |
| `ternary.mu` | 106 | Fused multiply-add 元素级 kernel | torch element-wise ops |
| `matmul_mudnn.cpp` | 53 | W8A8 scaled matmul via MuDNN | `fp8_kernel.py` / cutlass |
| `common.muh` | 27 | MUSA 通用宏和同步工具 | CUDA `__syncthreads()` |
| `dtype.muh` | 516 | MUSA 类型系统、向量类型定义 | CUDA vector types |

### 2.2 核心技术差异

#### 2.2.1 API 命名空间替换

```cpp
// CUDA  →  MUSA
#include <cuda_runtime.h>     →  #include <musa_runtime.h>
#include <cuda_fp16.h>        →  #include <musa_fp16.h>
#include <cuda_bf16.h>        →  #include <musa_bf16.h>
at::cuda::OptionalCUDAGuard   →  at::musa::OptionalMUSAGuard
cudaStream_t                  →  musaStream_t
cudaGetDeviceProperties()     →  musaGetDeviceProperties()
__syncthreads()               →  __SYNCTHREADS_LM (arch-dependent)
```

#### 2.2.2 架构差异 (`__MUSA_ARCH__`)

MUSA 根据架构版本分叉行为：

```cpp
#if defined(__MUSA_ARCH__) && __MUSA_ARCH__ == 310
#define ThreadNumPerWarp 32      // MUSA 310: 32 threads/warp
#define __SYNCTHREADS_LM __syncthreads_lm()  // 轻量级同步
#else
#define ThreadNumPerWarp 128     // 旧架构: 128 threads/warp
#define __SYNCTHREADS_LM __syncthreads()
#endif
```

这意味着 MUSA 310 架构的 warp 大小与 NVIDIA 不同（32 vs 128），kernel block size 需重新调优。

#### 2.2.3 MoE GEMV Kernel 设计 (`moe_gemv_swiglu.mu`)

这是适配版中**最复杂**的 kernel（825 行），实现了完整的 MoE GEMV + SwiGLU + RMSNorm 融合：

**特性矩阵**：
- 数据精度：BF16, FP16, FP8 (E4M3), W4A16 (INT4 weight)
- Scale 精度：float, bfloat16, float16
- 激活函数：SwiGLU (clamped), RMSNorm
- Block 配置：`{8,16}`, `{16,8}`, `{32,4}`, `{4,32}`, `{32,1}`, `{128,1}` (auto-tuned)

**与 CUDA 版的差异**：

| 特性 | CUDA (DeepGEMM/Triton) | MUSA |
|------|----------------------|------|
| GEMM 类型 | 完整的 group GEMM | **简化 GEMV** (per-token per-expert) |
| 指令集 | UMMA (SM100), MMA (SM90) | MUSA SIMT 向量指令 |
| FP8 dequant | 硬件 `CVT` 指令 | 软件 `CVT.RTE.DENORM.BST4` asm |
| 量化 | per-group (128) scale | per-group (128/64) scale |
| 并行策略 | warp specialization (3 roles) | grid-stride loop + shared memory reduce |

**关键 FP8 转换 asm**：
```cpp
// MUSA FP8 → FP16 转换使用内联汇编
asm("CVT.RTE.DENORM.BST4 %0.f16, %1.e4m3;" : "=R"(halfv4):"R"(fp8_data));
```

#### 2.2.4 Top-K/Top-P Sampling Kernel

基于 flashinfer 的 sampling 代码，做 MUSA 移植：

- 复用 flashinfer 的 `BlockReduce`, `BlockScan`, `SamplingTempStorage` 数据结构
- 替换 CUDA RNG (`curand`) → MUSA RNG (`murand_init`, `murand_uniform`)
- 替换 `cudaLaunchKernel` → `musaLaunchKernel`
- 保持 flashinfer 的 `DISPATCH_COMPUTE_CAP_NUM_THREADS` dispatch 模式

```cpp
// CUDA
curandStatePhilox4_32_10_t state;
curand_init(philox_seed, bx, philox_offset, &state);

// MUSA
murandStatePhilox4_32_10_t state;
murand_init(philox_seed, bx, philox_offset, &state);
```

#### 2.2.5 MuDNN Matmul (W8A8)

MUSA 使用 Moore Threads 的 **MuDNN** 库（类比 cuDNN/cuBLAS）做 W8A8 scaled matmul：

```cpp
::musa::dnn::BatchMatMul op;
::musa::dnn::MatMulLtParam param;
param.SetScale(a_scales_mu, b_scales_mu, ...);  // per-128 group scale
op.SetTranspose(false, true);                     // NT layout
op.RunLt(handle, out_mu, a_mu, b_mu, ...);
```

**差异**：CUDA 版使用 CUTLASS/DeepGEMM 做 W8A8 GEMM；MUSA 版调用 MuDNN。

---

## 3. 层 2: Python 硬件后端 (`hardware_backend/musa/`)

### 3.1 后端注册

MUSA 通过独立的 `hardware_backend/musa/` 目录注册为**一等 GPU 后端**：

```
hardware_backend/musa/
├── __init__.py                           # "MUSA (Moore Threads GPU) hardware backend"
├── attention/
│   ├── __init__.py
│   ├── flashattention_backend.py         # MUSA FlashAttention 实现
│   └── flash_attention.py               # FlashAttention 核心逻辑
├── kernels/
│   └── topk.py                           # MUSA topk kernel wrapper
└── layers/utils/
    ├── __init__.py
    └── cp_utils.py                        # Context Parallel 工具
```

**与 CUDA 版的架构差异**：

| 维度 | CUDA | MUSA |
|------|------|------|
| FlashAttention 后端 | `flashinfer` / `flash_attn` / `triton` | MUSA 自实现 `flashattention_backend.py` |
| GEMM 后端 | `deep_gemm` (Hopper/Blackwell) | `sgl_kernel.musa_*` + `MuDNN` |
| TopK | Triton kernel | MUSA native kernel (`musa_topk`) |

### 3.2 Device 字符串

MUSA 使用了独立的 device type string：

```python
# CUDA 路径
tensor = torch.empty(..., device="cuda")

# MUSA 路径
tensor = torch.empty(..., device="musa")
```

这个差异导致所有涉及 `device` 字符串的 hard-coded 路径都需要针对 MUSA 做条件判断。

---

## 4. 层 3: DeepSeek V4 MUSA 专用 Ops

这是 MUSA 适配版**增量最大**的部分（15 个文件，~3,000 行）。通过 Python 的模块 forwarding 机制，在运行时透明替换 DeepSeek V4 的核心 op。

### 4.1 文件组织

```
hardware_backend/layers/deepseek_v4_musa/
├── __init__.py
├── _forwarding.py              # Module forwarding 替换机制
├── common.py                   # 公共导出
├── ops/
│   ├── __init__.py
│   ├── mhc_ops.py              # mHC pre/post (252 行)
│   ├── mhc_prenorm_ops.py      # mHC pre-normalization (DeepGEMM 替代)
│   ├── hc_head_ops.py           # HC head operations
│   ├── swiglu_quant_ops.py     # SwiGLU + FP8 quant (520 行)
│   ├── cache_ops.py            # FlashMLA/Indexer KV cache pack/store (1274 行)
│   └── ops_common.py           # 公共工具
└── kernels/
    ├── __init__.py
    ├── kernel_common.py        # TileLang JIT + MUSA pass configs
    ├── mhc_kernels.py           # mHC TileLang kernel factories
    ├── hc_head_kernels.py       # HC head kernel factories
    └── cache_kernels.py         # FlashMLA/Indexer cache kernel factories
```

### 4.2 核心技术决策

#### 4.2.1 TileLang 作为 kernel 编写语言

CUDA 版使用 **DeepGEMM** + **Triton** + **CUDA C++** 编写 kernel。MUSA 版选择了 **TileLang** 作为主要 kernel 编写语言：

```python
import tilelang
import tilelang.language as T

@_tilelang_jit(tilelang, "dsv4_moe_silu_and_mul_h...",
               pass_configs=_tilelang_musa_aggressive_pass_configs(...))
def silu_and_mul_kernel(
    x: T.Tensor[(num_rows, hidden2), input_dtype],
    out: T.Tensor[(num_rows, hidden), input_dtype],
    swiglu_limit: T.float32,
):
    with T.Kernel(T.ceildiv(num_rows * hidden, threads), threads=threads):
        ...
```

**为什么用 TileLang**：MUSA 没有 NVIDIA 的 Tensor Core（TMEM/UMMA），无法直接使用 DeepGEMM。TileLang 提供跨硬件的 kernel 描述能力，可以编译到 MUSA 的指令集。

#### 4.2.2 mHC Pre 双路径设计

`mhc_pre_big_fuse` 使用 **try-catch 双路径**：

1. **优先路径**: `mhc_prenorm_gemm_sqrsum` — 通过 DeepGEMM-like API 做 split-K GEMM + RMSNorm
2. **Fallback 路径**: `_tilelang_mhc_pre_norm_fn_fwd_mul_kernel` — 纯 TileLang GEMM (TF32 精度)

```python
def _try_prenorm_backend(residual_flat, fn):
    from .mhc_prenorm_ops import mhc_prenorm_gemm_sqrsum
    d_out, s_out = mhc_prenorm_gemm_sqrsum(
        residual_flat.view(...), fn, split_k=split_k, return_partials=True)
    return True, d_out, s_out
```

#### 4.2.3 FlashMLA Cache 多 kernel 调度

MUSA 版实现了**行业领先的 FlashMLA KV cache 存储 kernel 选择器**，根据 token 数量、page_size、dtype 自动选择最优 kernel：

| Kernel 系列 | 适用场景 | 实现方式 |
|------------|---------|---------|
| **decode_x4** | decode 小 batch (≤128 tokens) | 4 token/warp block-per-token |
| **warp_col** / **warp_col_split** | 中等 batch (32K-128K tokens) | warp column store |
| **warp_col_fused** | 大 batch (≥131K tokens) | fused Nope+RoPE column |
| **c128** / **c128_contig** / **c128_pair** | page_size=2 (C128 extra cache) | page-aligned contiguous store |
| **vector** / **vector_x4_remap** / **vector_x4_tile_fused** | 大 batch 高性能 | 4-element vectorized store with remap/tile fusion |

**自动调度逻辑** (简化)：
```
if tokens ≤ 128 and contiguous and bf16/fp32:
    → decode_x4
elif page_size == 2 and tokens ≥ min(c128_threshold):
    → c128_pair / c128_contig / c128
elif tokens ≥ min(x4_remap_threshold):
    → vector_x4_remap / vector_x4_tile_fused
elif tokens ≥ min(vector_threshold):
    → vector / warp_col_fused / warp_col_split / warp_col
else:
    → row (default)
```

这个多级调度是**MUSA 版独有的优化**，CUDA 版没有这种细粒度的 kernel 选择。

#### 4.2.4 SwiGLU + Quant 多层 fallback

`sigu_and_mul_contig_post_quant_musa` 实现了 4 级 fallback：

```
1. _try_tile_swiglu_per_token_cast_prealloc_musa  → tile_kernels prealloc 路径
2. _try_tile_swiglu_per_token_cast_musa           → tile_kernels 通用路径
3. _tile_swiglu_forward                           → tile_kernels torch API
4. PyTorch fallback: F.silu(gate) * up + manual FP8 quant
```

MUSA 依赖 `tile_kernels` 作为第三方优化库（类似 CUDA 生态的 CUTLASS/FlashInfer）。

---

## 5. 层 4: SGLang 核心 inline 改动

### 5.1 改动文件清单

| 文件 | 改动类型 | 行数变化 |
|------|---------|---------|
| `layers/moe/topk.py` | `_is_musa` 标志 + MUSA Triton kernel + mate 集成 | ~300 行新增 |
| `layers/moe/token_dispatcher/standard.py` | device string `"musa"` | ~5 行 |
| `layers/moe/moe_runner/triton_utils/fused_moe.py` | MUSA SwiGLU + sum_reduce 路径 | ~30 行 |
| `layers/moe/ep_moe/kernels.py` | MUSA 支持 | 若干行 |
| `layers/moe/ep_moe/layer.py` | MUSA 支持 | 若干行 |
| `layers/quantization/fp8*.py` | MuDNN matmul 路径 | 若干行 |
| `platforms/device_mixin.py` | MUSA device 检测 | 若干行 |

### 5.2 核心改动模式

#### 模式 1: `_is_musa` 标志注入

```python
# 原 CUDA
if _is_cuda:
    ...

# MUSA 版
if _is_cuda or _is_musa:
    ...
```

应用到：topk、MoE runner、fused MoE 等核心路径。

#### 模式 2: device 字符串替换

```python
# 原
tensor = torch.empty(..., device="cuda")

# MUSA 版
tensor = torch.empty(..., device="cuda" if not _is_musa else "musa")
```

#### 模式 3: 特定 op 的 MUSA 替代

```python
# 原 CUDA
silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)

# MUSA 版
if _is_musa:
    intermediate_cache2 = torch.nn.SwishGLU()(intermediate_cache1.view(-1, N))
else:
    silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
```

#### 模式 4: MUSA Triton kernel

在 `topk.py` 中新增了 ~280 行 MUSA 专用的 Triton topk-softmax kernel：

```python
@triton.autotune(configs=[...], key=["num_tokens", "num_experts"])
@triton.jit
def topk_softmax_triton_kernel(
    gating_output_ptr, selected_expert_ptr, moe_weights_ptr,
    renormalize_flag, num_experts, num_tokens, K, BLOCK_WIDTH_SIZE_UP):
    ...
```

同时集成了 `mate.moe_fused_gate`（MUSA MoE 加速库）：

```python
if _is_musa:
    from mate import moe_fused_gate as mate_moe_fused_gate
```

#### 模式 5: MTT_S5000 设备配置

在 Triton autotune 配置中新增 8 个 MTT_S5000 专用 config：

```
layers/moe/moe_runner/triton_utils/configs/triton_3_2_0/
├── E=10,N=1536,device_name=MTT_S5000.json
├── E=128,N=192,device_name=MTT_S5000.json
├── E=128,N=384,device_name=MTT_S5000.json
├── E=160,N=192,device_name=MTT_S5000.json
├── E=32,N=1408,device_name=MTT_S5000.json
├── E=512,N=64,device_name=MTT_S5000.json
├── E=5,N=1536,device_name=MTT_S5000.json
└── E=64,N=768,device_name=MTT_S5000.json
```

---

## 6. 适配策略总结

### 6.1 架构模式

```
                       ┌─────────────────────┐
                       │   deepseek_v4.py     │ ← 模型定义 (不变)
                       │   (model forward)    │
                       └──────────┬──────────┘
                                  │
                  ┌───────────────┼───────────────┐
                  ▼               ▼               ▼
           ┌──────────┐   ┌──────────┐   ┌──────────┐
           │  mHC ops │   │ Attn ops │   │  MoE ops │
           │mhc_ops.py│   │cache_ops │   │swiglu_ops│
           └────┬─────┘   └────┬─────┘   └────┬─────┘
                │              │              │
     ┌──────────┼──────────────┼──────────────┼──────────┐
     │          ▼              ▼              ▼          │
     │   TileLang kernels   sgl-kernel    tile_kernels   │ ← MUSA 加速库层
     │   (tilelang JIT)    (musa C++)   (third-party)    │
     └──────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  MUSA Runtime   │ ← MUSA 运行时
                    │  + MuDNN        │
                    └─────────────────┘
```

### 6.2 关键设计原则

| 原则 | 实现方式 |
|------|---------|
| **最小侵入** | 核心模型代码不动，通过 `_is_musa` flag + module forwarding 注入 |
| **渐进式性能** | 多层 fallback：MUSA native → TileLang → tile_kernels → PyTorch |
| **条件编译** | C++ 层 `#if __MUSA_ARCH__` 区分架构版本 |
| **Python 透明替换** | `_ForwardingModule` 机制在运行时替换 module 属性 |
| **自调优** | FlashMLA cache 根据 token 数/page_size 自动选 kernel；MoE block size score 排序 |

### 6.3 未完成/待完善项

从代码中的 TODO 注释和 `NotImplementedError` 来看：

| 待完善特性 | 位置 | 原因 |
|-----------|------|------|
| **W4A4 MegaMoE** | swiglu_quant_ops | MUSA 无 TMEM/SM100 特性 |
| **UE8M0 scale layout** | swiglu_quant_ops, cache_ops | MUSA 对齐 SM90 target |
| **Swizzled layout** | swiglu_quant_ops | 需要 MUSA 原生 swizzle 支持 |
| **Transposed FP8 scale** | swiglu_quant_ops | 需要 MUSA 硬件支持 |
| **Native masked swizzle** | cache_ops | 待实现 |
| **DeepGEMM MegaMoE** | mhc_ops, mhc_prenorm | MUSA 无 SM100 UMMA 指令 |

---

## 7. 与上游 CUDA 版的差异量化

| 维度 | CUDA 版 | MUSA 版 | 差异幅度 |
|------|---------|---------|---------|
| **C++ kernel 实现** | DeepGEMM (~15K C++) + FlashInfer + CUTLASS | MUSA native kernels (~2.2K C++) + MuDNN | **重写** (不同指令集) |
| **Kernel 编写语言** | CUDA C++ / Triton / CUTLASS CuTe | TileLang / MUSA C++ / Triton MUSA fork | 语言层面不同 |
| **Attention backend** | flashinfer / flash_attn / triton | MUSA flashattention_backend | **重写** |
| **GEMM 后端** | DeepGEMM (FP8/FP4) | MuDNN + sgl_kernel.musa GEMV | 能力不同 (GEMV vs GEMM) |
| **MoE 性能关键路径** | Triton fused MoE / MegaMoE | MUSA GEMV + TileLang SwiGLU | **重写** (无 tensor core) |
| **Python 框架层** | ~0 额外代码 | ~3,000 行新 Python | +3% |
| **SGLang 核心改动** | 基准 | ~300 行条件分支 | +0.5% |
| **FlashMLA cache** | 简单 kernel 选择 | 10+ kernel 变体 + auto dispatch | MUSA 更复杂 |

---

## 8. 风险评估

### 8.1 技术风险

| 风险项 | 等级 | 说明 |
|--------|------|------|
| **维护同步** | 🔴 高 | MUSA 需持续跟踪 upstream SGLang 的 kernel 变更和架构演进 |
| **性能差距** | 🟡 中 | MUSA 无 Tensor Core，decode GEMM 只能用 GEMV，预计 prefill 性能差距较大 |
| **FP8 精度** | 🟡 中 | MUSA arch < 300 不支持 FP8；arch 310 支持但 kernel 尚未充分验证 |
| **tile_kernels 依赖** | 🟡 中 | 外部 `tile_kernels` 库的可用性和稳定性 |
| **Merge Conflict** | 🔴 高 | inline 改动分散在 6+ 个核心文件，与 upstream 同步时易冲突 |

### 8.2 架构债务

1. **Inline `_is_musa` 分支**：分散在 6+ 个文件中，存在条件分支爆炸风险
2. **双重 GEMM 生态**：MuDNN (dense) + MUSA GEMV (MoE) 两个 GEMM 路径
3. **TileLang 版本锁定**：TileLang 生成的 MUSA 代码依赖特定编译器版本
4. **kernel 变体爆炸**：FlashMLA cache 有 18+ 个 kernel 变体需各自维护

---

## 9. 优化建议

### 9.1 短期（降低维护成本）

1. **统一 MUSA 路径入口**：将分散的 `_is_musa` 分支集中到一个 `musa_backend` 调度器
2. **减少 FlashMLA kernel 变体**：基于实际性能数据精简到 4-5 个覆盖主力场景
3. **补全 FP8 inference 验证**：在 MUSA arch 310 上完成端到端 FP8 精度测试

### 9.2 中期（性能提升）

1. **GEMV → GEMM 升级**：利用 MuDNN 的 grouped GEMM 替代当前的 per-token GEMV
2. **MoE Gate fuse kernel**：将 `softplus + sqrt + biasAdd + TopK` 融合为 MUSA native kernel（参考 CUDA Triton kernel）
3. **Attention compress kernel**：为 CSA/HCA compressor 编写 MUSA fused kernel

### 9.3 长期（架构优化）

1. **统一 kernel 描述层**：对齐 TileLang 生态，减少对 MuDNN/tile_kernels/sgl-kernel 的多重依赖
2. **MUSA 上支持 GEMM tensor core 等效指令**：如硬件支持类似 MMA 的指令，实现 group GEMM
3. **建立 MUSA-SGLang CI 体系**：利用 `scripts/ci/musa/` 基础，建立 nightly regression

---

## 10. 结论

MUSA 适配版 SGLang 是一个**工程规模显著**（~5,500 行新增代码）的系统性 GPU 后端移植。其核心策略是：

- **不修改模型定义**（`deepseek_v4.py` 不变）
- **在 kernel 层完全重写**（MUSA 没有 CUDA tensor core，必须用 GEMV + TileLang 替代 GEMM）
- **在框架层条件注入**（`_is_musa` flag + module forwarding）

最大的性能挑战是 MUSA 硬件缺少类似 NVIDIA Tensor Core 的矩阵乘加速单元，导致 MoE 的 GEMM 操作只能用 GEMV 实现。FlashMLA cache 存储路径的 MUSA 版反而展现了超越 CUDA 版的 kernel 调度精细度（18+ 变体自动选择），说明 MUSA 团队在 memory-bound 操作上有独特的优化经验。
