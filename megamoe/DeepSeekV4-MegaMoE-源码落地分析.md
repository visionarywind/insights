# DeepSeekV4 MegaMoE 源码级落地分析

> 以 DeepSeekV4 源码为例，从模型初始化、权重加载、运行时 forward 三个维度，完整追踪 MegaMoE 融合推理路径的落地实现。

---

## 1. 模型架构概览

### 1.1 类层次结构

```
DeepseekV4ForCausalLM                          # 顶层模型 (deepseek_v4.py:1067)
  └─ DeepseekV4Model                           # Transformer 主体 (deepseek_v4.py:929)
       └─ DeepseekV4DecoderLayer               # 解码器层 (deepseek_v4.py:597)
            ├─ self.self_attn                  # MLA 注意力
            ├─ self.hc_pre / self.hc_post      # Head Change (V4 特有)
            └─ self.mlp = DeepseekV2MoE        # MoE 层 (复用 V2 的 MoE 封装)
                 ├─ self.gate = MoEGate        # Router 门控网络
                 ├─ self.topk = TopK           # Expert 选择 (sqrtsoftplus, ungrouped)
                 ├─ self.experts               # MoE 计算层 (FusedMoE 或 DeepEPMoE)
                 │    ├─ w13_weight            # gate+up 融合权重 [E, 2I, H]
                 │    ├─ w2_weight             # down 投影权重 [E, H, I]
                 │    ├─ mega_l1_weights       # MegaMoE 专用 L1 (weight, scale_utccp)
                 │    └─ mega_l2_weights       # MegaMoE 专用 L2 (weight, scale_utccp)
                 └─ self.shared_experts        # 共享专家 (V4 不融合，独立执行)
```

**关键设计决策**：DeepSeekV4 复用 `deepseek_v2.DeepseekV2MoE` 作为其 MoE block。V2/V3/V4 的 MoE 结构高度相似（都是 256 routed experts + shared expert + SwiGLU），V4 的主要差异在于 `sqrtsoftplus` 门控函数、swiglu clamping、以及 Multi-Token Prediction (MTP) 的 Head Change 机制。

### 1.2 关键配置参数

```python
# deepseek_v4.py:1054-1064
def determine_num_fused_shared_experts(self):
    self.num_fused_shared_experts = 0
    get_global_server_args().disable_shared_experts_fusion = True
    # "DeepSeek V4 requires different clamping for shared and routed experts"
```

V4 的 shared expert **始终不融合**到 MoE kernel 中，因为 routed 和 shared expert 使用不同的 swiglu clamping 值。这影响了 MegaMoE 的 SBO (Single Batch Overlap) 策略——shared expert 可以在单独的 CUDA stream 上与 MegaMoE routed kernel 并行执行。

### 1.3 模型参数一览

| 参数 | 值 | 说明 |
|------|-----|------|
| `hidden_size` | 7168 | 隐藏层维度 |
| `moe_intermediate_size` | 2048 | FFN 中间维度 |
| `n_routed_experts` | 256 | 路由专家总数 |
| `num_experts_per_tok` | 6 (V4) / 8 (V3) | 每 token 激活的 expert 数 |
| `n_shared_experts` | 1 | 共享专家数 |
| `num_fused_shared_experts` | 0 (V4 强制) | 融合到 MoE 的共享专家数 |
| `routed_scaling_factor` | 2.5 | 路由输出缩放 |
| `scoring_func` | `"sqrtsoftplus"` | V4 特有门控函数 |
| `topk_group` | 1 (ungrouped) | 不使用 expert 分组 |
| EP size | 16 | Expert Parallel 度，每 GPU 16 experts |

---

## 2. 初始化阶段

### 2.1 模型构建

```python
# deepseek_v4.py:597-630 — DeepseekV4DecoderLayer.__init__()
class DeepseekV4DecoderLayer(nn.Module):
    def __init__(self, config, ...):
        # 第 621 行：MoE block 实例化
        self.mlp = deepseek_v2.DeepseekV2MoE(
            config, quant_config, layer_id=layer_id,
            prefix=add_prefix("mlp", prefix),
            enable_mtp_hint=enable_mtp_hint,
            is_deepseek_v4=True,  # ← V4 标记
        )
```

`DeepseekV2MoE.__init__()` 在 `deepseek_v2.py:386-550` 中完成：

1. **选择 MoE 实现类**：`get_moe_impl_class()` → FusedMoE 或 DeepEPMoE
2. **创建 experts 层**：`moe_impl_cls(num_experts=256, top_k=6, hidden_size=7168, intermediate_size=2048, ...)`
3. **创建 Router**：`MoEGate(scoring_func="sqrtsoftplus", topk_group=1, ...)` — V4 特有门控
4. **创建 TopK**：`TopK(...)` — 无分组的 sqrtsoftplus TopK
5. **创建 Shared Expert**：`DeepseekV2MLP(hidden_size=7168, intermediate_size=2048, ...)`

### 2.2 Expert 权重创建

权重的实际 tensor 由量化方法创建。对于 DeepSeekV4 的 FP4 原生格式（`fp8.py:853-960`）：

```
FP4 packed 权重（int8 存储，每字节两个 fp4 值）：
  w13_weight: [256, 4096, 3584]  # 2*2048 × 7168/2（FP4 压缩）
  w2_weight:  [256, 7168, 1024]  # 7168 × 2048/2（FP4 压缩）

Block-wise scale（float32）：
  w13_weight_scale_inv: [256, 128, 224]  # 2*2048/32 × 7168/32
  w2_weight_scale_inv:  [256, 224, 64]   # 7168/32 × 2048/32
```

此时 `experts._mega_moe_weights_built = False`，MegaMoE 专用权重尚未构建。

---

## 3. 权重加载阶段 — MegaMoE 权重变换

### 3.1 触发时机

MegaMoE 权重构建发生在模型权重加载完成后的 `process_weights_after_loading()` 阶段：

```python
# fp8.py:1185-1200 — Fp8MoEMethod.process_weights_after_loading()
if (
    envs.SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE.get()    # 环境变量启用
    and self.is_fp4_expert                          # FP4 专家权重
    and hasattr(layer, "w13_weight")                # 权重已加载
):
    from sglang.srt.layers.moe.mega_moe import build_mega_moe_experts_weights
    build_mega_moe_experts_weights(layer)           # 构建 mega 布局
```

### 3.2 权重布局变换

`build_mega_moe_experts_weights()` 在 `mega_moe.py:227-330`：

```python
def build_mega_moe_experts_weights(experts: nn.Module):
    # 提取原始权重
    w13_weight = experts.w13_weight        # [256, 4096, 3584]  int8 (FP4 packed)
    w13_sf = experts.w13_weight_scale_inv   # [256, 128, 224]   float32
    w2_weight = experts.w2_weight          # [256, 7168, 1024]  int8
    w2_sf = experts.w2_weight_scale_inv    # [256, 224, 64]     float32

    # Step 1: Scale 格式转换 (float32 → UE8M0 + block 重排)
    w13_sf_f32 = 1.0 / w13_sf  # scale_inv → scale
    w13_sf_ue8m0 = deep_gemm.transform_sf_into_required_layout(
        w13_sf_f32, mn=4096, k=7168, recipe=(1, 32), num_groups=256
    )

    # Step 2: 权重+Scale 交织 (两个方案)
    if envs.SGLANG_OPT_FIX_MEGA_MOE_MEMORY.get():
        # 方案 A：原位修改，共享 deep-ep 的权重 buffer（省显存）
        w13_interleaved, w13_sf_interleaved = _interleave_l1_weights(...)
        w13_sf_utccp = _transpose_sf_for_utccp(w13_sf_interleaved)
        experts.mega_l1_weights = (experts.w13_weight.data, w13_sf_utccp)
    else:
        # 方案 B：独立副本（默认，多占一份显存）
        l1_pair, l2_pair = deep_gemm.transform_weights_for_mega_moe(...)
        experts.mega_l1_weights = l1_pair
        experts.mega_l2_weights = l2_pair

    experts._mega_moe_weights_built = True
```

**UTCCP (Unified Tensor Core Compute Pipeline)** 是 DeepGEMM 内部的 Tensor Core 数据通路格式。它要求 scale 矩阵按特定 tile 顺序排列（与 weight 的 swizzle 模式匹配），使得 Tensor Core 在加载 weight block 时能同时获取对应的 scale 值，避免额外的 global memory 访问。

### 3.3 其他量化后端的感知

MegaMoE 权重构建完成后，其他量化后端（Marlin、FlashInfer TRT-LLM）会检测 `_mega_moe_weights_built` 标记并跳过自己的权重变换：

```python
# mxfp4_marlin_moe.py:63
if getattr(layer, "_mega_moe_weights_built", False):
    return  # MegaMoE 已处理，跳过 Marlin 的 repack

# mxfp4_flashinfer_trtllm_moe.py:218
if getattr(layer, "_mega_moe_weights_built", False):
    return  # 同上
```

---

## 4. 运行时 Forward 阶段

### 4.1 总体调用链

```
DeepseekV4ForCausalLM.forward()                        # deepseek_v4.py:1067
  └─ DeepseekV4Model.forward()                         # deepseek_v4.py:929
       └─ for layer in layers:
            DeepseekV4DecoderLayer.forward()            # deepseek_v4.py:771
              ├─ self_attn(hidden_states)               # MLA attention
              ├─ hc_post(attn_output)                   # Head Change
              └─ self.mlp(hidden_states, ...)           # 第 839 行 → MoE
                   └─ DeepseekV2MoE.forward()           # deepseek_v2.py:644
                        └─ forward_mega_moe()           # mega_moe.py:90 (MegaMoE 路径)
```

### 4.2 MoE 路径选择（关键分叉点）

```python
# deepseek_v2.py:644-700 — DeepseekV2MoE.forward()
def forward(self, hidden_states, forward_batch, ...):
    from sglang.srt.layers.moe.mega_moe import forward_mega_moe, should_use_mega_moe

    if should_use_mega_moe(self, hidden_states):
        return forward_mega_moe(self, hidden_states, forward_batch, ...)

    if not self.use_deepep:
        return self.forward_normal(...)         # 非 EP 路径
    else:
        return self.forward_deepep(...)          # DeepEP all-to-all 路径
```

### 4.3 MegaMoE 门控条件

```python
# mega_moe.py:73-85 — should_use_mega_moe()
def should_use_mega_moe(moe, hidden_states) -> bool:
    if not envs.SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE.get():
        return False
    if not moe.experts._mega_moe_weights_built:
        return False
    if get_is_capture_mode():
        return True
    max_tokens_per_rank = max(global_num_tokens)
    cap = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.get()
    return max_tokens_per_rank <= cap
```

**门控逻辑解读**：

| 阶段 | 条件 | 结果 |
|------|------|------|
| Decode (CUDA Graph) | `capture_mode=True` | 始终启用 |
| 小 batch prefill | `tokens <= 1024` (默认) | 启用 |
| 大 batch prefill | `tokens > 1024` | 回退标准 MoE |
| 环境变量未设置 | `USE_MEGA_MOE=False` | 回退 |
| 权重未构建 | `_mega_moe_weights_built=False` | 回退 |

### 4.4 forward_mega_moe() — 主流程

```python
# mega_moe.py:90-125
def forward_mega_moe(moe, hidden_states, forward_batch, ...):
    # Step 1: Shared Expert (可选 SBO overlap)
    sbo_overlap_flag = (
        moe.alt_stream is not None
        and moe.num_fused_shared_experts == 0  # V4 始终为 True
        and num_tokens > 0
        and get_is_capture_mode()
    )

    if sbo_overlap_flag:
        # Shared expert 在 alt_stream 上与 MegaMoE 并行
        moe.alt_stream.wait_stream(torch.cuda.current_stream())
        shared_output = moe._forward_shared_experts(hidden_states)
    else:
        shared_output = moe._forward_shared_experts(hidden_states)

    # Step 2: MegaMoE Routed Expert (核心计算)
    y = _run_mega_routed(moe, hidden_states, forward_batch, ...)

    if sbo_overlap_flag:
        torch.cuda.current_stream().wait_stream(moe.alt_stream)

    # Step 3: 合并 Shared Expert 输出
    if shared_output is not None:
        y += shared_output
    return y
```

**SBO Overlap 策略**：当 V4 的 shared expert 未融合时，shared expert 的 MLP 计算可以与 MegaMoE routed kernel 在不同的 CUDA stream 上并行执行，隐藏 shared expert 的延迟。

### 4.5 _run_mega_routed() — 核心计算管线

```python
# mega_moe.py:127-235
def _run_mega_routed(moe, hidden_states, forward_batch, ...):
    # ===== 阶段 1: Router =====
    router_logits = moe.gate(hidden_states, forward_batch=forward_batch)
    # → [num_tokens, 256]

    topk_output = moe.topk(hidden_states, router_logits, forward_batch, ...)
    topk_ids = topk_output.topk_ids          # [num_tokens, 6] int32
    topk_weights = topk_output.topk_weights  # [num_tokens, 6] float32

    # ===== 阶段 2: SymmBuffer 准备 =====
    ep_group = get_moe_ep_group().device_group
    buf = _get_mega_moe_symm_buffer(
        ep_group, num_experts=256,
        num_max_tokens_per_rank=cuda_graph_max_bs,  # 通常 256
        num_topk=6, hidden=7168, intermediate_hidden=2048,
    )
    # buf 结构：
    #   buf.x:            [256, 7168]  fp8_e4m3
    #   buf.x_sf:         [256, 56]    int32 (UE8M0 packed)
    #   buf.topk_idx:     [256, 6]     int64
    #   buf.topk_weights: [256, 6]     float32

    # ===== 阶段 3: Pre-Dispatch Kernel =====
    mega_moe_pre_dispatch(
        hidden_states, topk_ids_in, topk_weights_in,
        buf.x, buf.x_sf, buf.topk_idx, buf.topk_weights,
        quant_group_size=32,
    )

    # ===== 阶段 4: Fused Mega Kernel =====
    swiglu_limit = moe.topk.moe_runner_config.activation_clamp
    deep_gemm.fp8_fp4_mega_moe(
        y,
        moe.experts.mega_l1_weights,    # (w13_weight, w13_sf_utccp)
        moe.experts.mega_l2_weights,    # (w2_weight, w2_sf_utccp)
        buf,
        recipe=(1, 1, 32), activation="swiglu",
        activation_clamp=swiglu_limit, fast_math=True,
    )

    # ===== 阶段 5: 路由缩放 =====
    if moe.routed_scaling_factor is not None:
        y *= moe.routed_scaling_factor   # × 2.5
    return y
```

---

## 5. Pre-Dispatch Kernel 详解

### 5.1 JIT 编译链

```
mega_moe_pre_dispatch()                          # Python 入口 (deepseek_v4.py:863)
  └─ _jit_mega_moe_pre_dispatch_module(32)       # JIT 模块工厂 (line 280)
       ├─ make_cpp_args(32, is_arch_support_pdl())
       └─ load_jit(name, *args,
            cuda_files=["deepseek_v4/mega_moe_pre_dispatch.cuh"],
            cuda_wrappers=[("run", "MegaMoEPreDispatchKernel<...>::run")])
            └─ tvm_ffi.cpp.load(source, ...)     # TVM JIT 编译
```

模板参数：`quant_group_size = 32`，`kUsePDL = is_arch_support_pdl()`（sm_90+ 启用）

### 5.2 线程组织

```cpp
// mega_moe_pre_dispatch.cuh:190-215
const auto num_threads = hidden / 8;    // 7168/8 = 896 threads/block
const uint32_t num_pad_blocks = (pad_slots + num_threads - 1) / num_threads;
const auto num_total_blocks = num_tokens + num_pad_blocks;

LaunchKernel(num_total_blocks, num_threads, device)(kernel, params);
```

- **Block dim**：896（每线程处理 8 个 bf16 = 16B 对齐加载）
- **Grid dim**：`num_tokens + num_pad_blocks`
- 每个 CTA 处理一个 token 的整行

### 5.3 核函数逻辑（两条路径）

**路径 A：有效 token（bid < num_tokens）**— 7 个步骤：

1. **向量化加载**：`AlignedVector<bf16x2_t, 4>` 加载每线程 8 个 bf16（16B）
2. **bf16 → fp32 转换 + 局部 absmax**：4 个 `bf16x2_t` → 8 个 fp32，计算 `local_max`
3. **Warp-level absmax reduce**：`warp::reduce_max<4>(local_max)` — 4 线程协作
4. **UE8M0 scale 计算**：`raw_scale = absmax / 448.0` → `cast_to_ue8m0(raw_scale)` → `inv_scale`
5. **量化**：`val × inv_scale → clamp → pack_fp8` → `fp8_e4m3`
6. **写入 buf_x + buf_x_sf**：所有线程写 fp8 值，每组 1 线程写 UE8M0 scale
7. **拷贝 topk 数据**：前 `top_k` 个线程拷贝 `topk_idx` 和 `topk_weights`

**路径 B：Padding CTA（bid ≥ num_tokens）**— 填充 `-1` 和 `0.0f`

### 5.4 数据格式转换

```
输入                          →  输出 (SymmBuffer)
hidden_states [T, H] bf16     →  buf.x       [P, H]    fp8_e4m3
                               →  buf.x_sf    [P, H/128] int32 (4×UE8M0)
topk_ids      [T, K] int32    →  buf.topk_idx [P, K]    int64 (widened!)
topk_weights  [T, K] float32  →  buf.topk_weights [P, K] float32
padding                        →  buf.topk_idx 填充 -1, weights 填充 0.0

T = num_tokens, P = padded_max (CUDA Graph 固定), K = top_k
```

关键细节：`topk_idx` 从 `int32` 拓宽为 `int64`，DeepGEMM 的 SymmBuffer 接口统一要求 int64。

---

## 6. DeepGEMM Fused Mega Kernel

### 6.1 调用接口

```python
deep_gemm.fp8_fp4_mega_moe(
    y,                              # output: [T, 7168] bf16
    moe.experts.mega_l1_weights,    # (w13_weight, w13_sf_utccp)
    moe.experts.mega_l2_weights,    # (w2_weight, w2_sf_utccp)
    buf,                            # SymmBuffer
    recipe=(1, 1, 32), activation="swiglu",
    activation_clamp=swiglu_limit, fast_math=True,
)
```

### 6.2 内部执行流程（推断）

```
deep_gemm.fp8_fp4_mega_moe() 单次 kernel 调用内完成：

1. READ buf.topk_idx → token→expert 映射表
2. All-to-All Dispatch: 按 expert 分组 token，跨 EP ranks 通信
3. L1 GEMM: X_local @ mega_l1_weights.T = [tokens_for_expert, 2*2048]
   - 前半 2048: gate 投影，后半 2048: up 投影
   - 输入 FP8, 权重 FP4, 累加器 FP32
4. SwiGLU Activation: gate, up = split → silu(clamp(gate)) * up
5. L2 GEMM: act @ mega_l2_weights.T = [tokens_for_expert, 7168]
6. Combine: scatter 回 token 位置 × topk_weights，累加同 token 多 expert 结果
7. WRITE y [T, 7168] bf16
```

### 6.3 与传统路径的性能对比

| 维度 | 传统 MoE (FusedMoE) | MegaMoE |
|------|---------------------|---------|
| Kernel Launch | 768 次 (256×3) | 2 次 |
| 中间数据精度 | bf16 | fp8 (省 48% 带宽) |
| 显存往返 | 每 expert 一次 global | 融合 kernel 内复用 |
| 跨 GPU 通信 | 显式 all-to-all (DeepEP) | kernel 内融合 |
| CUDA Graph | 需逐 expert 捕获 | 天然兼容 |

---

## 7. MUSA 适配分析

### 7.1 已有适配

Pre-dispatch kernel 已兼容 MUSA：

```cpp
// utils.cuh:25-46 — 设备头文件选择
#ifndef USE_ROCM
  #include <musa_bf16.h>    // MUSA bf16
  #include <musa_fp8.h>     // MUSA fp8_e4m3
  #include <musa_runtime.h>
  using deviceStream_t = MusaStream*;
  #define LaunchKernel(...) musaLaunchKernelEx(...)
#endif

// fp8_utils.cuh:8 — FP8 工具函数
#include <musa_fp8.h>  // cast_to_ue8m0, pack_fp8

// type.cuh — 类型注册
template<> struct dtype_trait<fp8_e4m3_t> { ... };
template<> struct dtype_trait<bf16_t> { ... };
```

### 7.2 待适配项

核心的 `deep_gemm.fp8_fp4_mega_moe()` 是外部 DeepGEMM 包：

1. **FP8 Tensor Core 支持**：MUSA TC FP8 吞吐是否匹配 `recipe=(1,1,32)` tiling
2. **UE8M0 scale 布局**：SymmBuffer 格式兼容性
3. **FP4/MXF4 支持**：V4 使用 FP4 原生权重
4. **All-to-All 通信**：Mega kernel 内部跨 GPU 通信依赖 NVSwitch/NVLink

### 7.3 deepseek_v4_musa 后端

```
repos/sglang/python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/
```

该目录有 swiglu quant、cache store、MHC ops 的 MUSA 重写，但**没有** `mega_moe_pre_dispatch` 重写——说明 CUDA JIT kernel 可直接在 MUSA 上运行。

---

## 8. 环境变量与配置

| 变量 | 默认值 | 作用 |
|------|--------|------|
| `SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE` | `False` | MegaMoE 总开关 |
| `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK` | `1024` | prefill token 阈值 |
| `SGLANG_OPT_FIX_MEGA_MOE_MEMORY` | `False` | 权重内存共享 |

测试配置（`test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py:34`）：

```python
_MEGAMOE_ENV = {
    "SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE": "1",
    "SGLANG_OPT_FIX_MEGA_MOE_MEMORY": "1",
    "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "4096",
    "SGLANG_OPT_FIX_NEXTN_MEGA_MOE": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "0",
}
```

---

## 9. 完整数据流总结

```
═══════════════  初始化（模型加载时）  ═══════════════
config → DeepseekV4DecoderLayer.__init__()
  → DeepseekV2MoE(gate, topk, experts=DeepEPMoE, shared_experts)
  → Fp8MoEMethod.create_weights() → w13_weight, w2_weight (FP4 packed)

═══════════════  权重加载后  ═══════════════
process_weights_after_loading()
  → build_mega_moe_experts_weights()
     → deep_gemm.transform_sf_into_required_layout()  # scale 重排
     → deep_gemm.transform_weights_for_mega_moe()     # weight+scale 交织
     → experts.mega_l1_weights, experts.mega_l2_weights
     → experts._mega_moe_weights_built = True

═══════════════  运行时 Forward  ═══════════════
hidden_states [T, 7168] bf16
  → Router: gate(hidden_states) → [T, 256]
  → TopK: topk(logits) → topk_ids [T,6], topk_weights [T,6]
  → should_use_mega_moe()? → YES
     → forward_mega_moe()
        ├─ _forward_shared_experts() [alt_stream, 可选并行]
        └─ _run_mega_routed()
           ├─ SymmBuffer: deep_gemm.get_symm_buffer_for_mega_moe()
           ├─ mega_moe_pre_dispatch() [JIT kernel]
           │     bf16 → FP8 quantize (group_size=32, UE8M0 scale)
           │     copy topk data → SymmBuffer
           │     pad unused slots
           └─ deep_gemm.fp8_fp4_mega_moe() [fused kernel]
                 all-to-all → L1 GEMM → SwiGLU → L2 GEMM → combine
           → y *= 2.5
        └─ y += shared_output
  → final_hidden_states [T, 7168] bf16

═══════════════  回退路径（MegaMoE 不可用时）  ═══════════════
  → forward_normal() 或 forward_deepep()
     → FusedMoE.forward_impl()
        → dispatcher.dispatch() → run_moe_core() → dispatcher.combine()
```

---

## 10. 相关文件索引

| 文件 | 行号 | 内容 |
|------|------|------|
| `repos/sglang/python/sglang/srt/models/deepseek_v4.py` | 597-630 | DecoderLayer 构建，MoE 实例化 |
| `repos/sglang/python/sglang/srt/models/deepseek_v4.py` | 771-839 | DecoderLayer forward，MoE 调用点 |
| `repos/sglang/python/sglang/srt/models/deepseek_v4.py` | 1054-1064 | `num_fused_shared_experts=0` 强制设置 |
| `repos/sglang/python/sglang/srt/models/deepseek_v2.py` | 386-550 | `DeepseekV2MoE.__init__()` |
| `repos/sglang/python/sglang/srt/models/deepseek_v2.py` | 644-700 | `forward()` — MegaMoE 分叉点 |
| `repos/sglang/python/sglang/srt/layers/moe/mega_moe.py` | 36-70 | SymmBuffer 缓存 |
| `repos/sglang/python/sglang/srt/layers/moe/mega_moe.py` | 73-85 | `should_use_mega_moe()` 门控 |
| `repos/sglang/python/sglang/srt/layers/moe/mega_moe.py` | 90-125 | `forward_mega_moe()` 主流程 |
| `repos/sglang/python/sglang/srt/layers/moe/mega_moe.py` | 127-235 | `_run_mega_routed()` 核心管线 |
| `repos/sglang/python/sglang/srt/layers/moe/mega_moe.py` | 227-330 | `build_mega_moe_experts_weights()` |
| `repos/sglang/python/sglang/srt/layers/quantization/fp8.py` | 853-960 | `create_weights()` — FP4 权重创建 |
| `repos/sglang/python/sglang/srt/layers/quantization/fp8.py` | 1185-1200 | `process_weights_after_loading()` — MegaMoE 触发 |
| `repos/sglang/python/sglang/jit_kernel/deepseek_v4.py` | 280-287 | `_jit_mega_moe_pre_dispatch_module()` |
| `repos/sglang/python/sglang/jit_kernel/deepseek_v4.py` | 863-882 | `mega_moe_pre_dispatch()` Python 入口 |
| `repos/sglang/python/sglang/jit_kernel/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh` | 1-219 | CUDA/MUSA kernel 完整源码 |
| `repos/sglang/python/sglang/jit_kernel/include/sgl_kernel/utils.cuh` | 25-46 | MUSA 头文件选择 |
| `repos/sglang/python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/fp8_utils.cuh` | 1-60 | FP8/UE8M0 工具函数 |
| `repos/sglang/python/sglang/srt/environ.py` | 615-617 | MegaMoE 环境变量定义 |
| `repos/sglang/test/registered/dsv4/test_deepseek_v4_flash_fp4_b200.py` | 34-40 | MegaMoE 测试配置 |
