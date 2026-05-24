# SGLang DeepSeek V4 MUSA 适配与开源版差异洞察

## 对比范围

本次对比使用：

- MUSA 适配版：`/home/mtuser/workspace/sglang`，分支 `deepseek_v4_0511_mhc`
- 开源参考版：`/home/mtuser/workspace/Github/sglang`，分支 `main`

DeepSeek V4 相关差异集中在：

- `python/sglang/srt/models/deepseek_v4.py`
- `python/sglang/jit_kernel/deepseek_v4.py`
- `python/sglang/jit_kernel/csrc/deepseek_v4/*.cuh`
- MUSA 版新增 `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/`

相反，`layers/attention/deepseek_v4_backend.py`、`layers/attention/dsv4/`、`mem_cache/deepseek_v4_*` 与开源参考基本一致。这说明核心 DSV4 metadata、CUDA Graph replay metadata、KV pool 结构没有被大改，MUSA 适配主要发生在算子可用性、kernel 编译、HC/MoE/cache store 热点路径上。

## 差异总览

| 模块 | 开源版做法 | MUSA 版变化 | 根因 |
| --- | --- | --- | --- |
| 平台判断 | `DeepSeekV4` 只识别 CUDA/HIP/NPU | 引入 `_is_musa`，MUSA 也创建 alt streams | MUSA 需要走 CUDA-like stream/graph 多流路径 |
| HC pre | torch fallback 带 `@compile_in_capture_mode` | MUSA 版注释掉该装饰器 | MUSA torch.compile/capture 对该 fallback 路径不够稳定或收益不足 |
| MHC pre norm | TileLang path 默认可融合 RMSNorm | MUSA backend 为 `sglang_tilelang/tilekernels` 时禁止 norm 融合 | MUSA MHC pre backend 返回 pre-norm layer input，融合契约不同 |
| HC head | 默认 fused_hc_head 或 torch fallback | 增加 `try_hc_head_tilelang`、`try_hc_head_linear_tilelang` | PyTorch `F.linear + sigmoid + sum` 对大 hidden/small batch 太慢 |
| SwiGLU/quant | JIT CUDA wrapper | MUSA device tensor 分派到 `silu_and_mul_*_musa` | CUDA JIT kernel/FP8 layout 不直接适配 MUSA |
| cache store | JIT `fused_store_cache` | MUSA tensor 分派到 `fused_store_cache_musa` | FlashMLA/indexer KV pack-store 需要 MUSA TileLang/native op |
| JIT C++ | CUDA/NVCC/C++20 风格 | 大量降级/替换为 MUSA 可编译写法 | MUSA clang/runtime 与 CUDA 编译特性不完全等价 |

## PyTorch 常见 Op 差异

### 1. `F.linear`、`einsum` 与 HC/MHC

开源 DeepSeek V4 中 HC 路径的 fallback 主要由 PyTorch op 组成：

- `x.flatten(1).float()`
- `x.square().mean(...), torch.rsqrt(...)`
- `F.linear(x_flat, hc_fn)`
- `torch.sigmoid(...)`
- `sum(dim=1)`

这些 op 语义清晰，但对 DeepSeek V4 的 HC 结构并不理想：hidden 很大、decode batch 小、op 粒度碎，容易出现 kernel launch 多、访存重复、图捕获开销高的问题。MUSA 版新增 HC/MHC TileLang 后端，把 `linear/sqrsum/rsqrt/mix/sinkhorn/post` 拆成或融合成 MUSA kernel，并通过环境变量控制 split-K、threads、hidden block、layout。

根因：开源版默认依赖 CUDA 生态里的 fused kernel、DeepGEMM 或 torch.compile 补性能；MUSA 上这些路径不可直接复用，必须把 PyTorch fallback 中的常见 op 下沉到 MUSA TileLang/native kernel，否则 decode 小 batch 会被 launch overhead 和内存带宽放大。

### 2. `empty`、`copy_`、`contiguous`

MUSA 新增算子普遍要求输入输出 contiguous，并大量使用 `torch.empty` 预分配：

- HC head split-K partial/output buffer
- MHC pre/post intermediate buffer
- FlashMLA KV pack-store buffer
- SwiGLU quant output/scale

这与开源版的 `new_empty`、`copy_` 使用方向一致，但 MUSA 版更强调 shape/dtype/stride 检查，很多路径 fail-closed，不默认退回 torch fallback。

根因：MUSA kernel 对 layout 更敏感，且 graph replay 要求地址稳定；隐式 fallback 虽然方便，但可能引入 CPU/GPU sync、额外 copy 或未覆盖 FP8 layout，影响正确性和性能定位。

### 3. `item/tolist/cpu`

核心 DSV4 backend 在两份源码中基本一致，因此 `.item()`、`.tolist()`、`.cpu()` 的风险点没有本质变化：

- `seq_lens_cpu.max().item()` 通常安全，因为 `seq_lens_cpu` 是 CPU mirror。
- `seq_lens.tolist()` 若来源是 GPU，会触发同步。
- `F.pad(value=req_pool_indices_repeated[-1].item())` 若 tensor 在 GPU，会产生单标量 D2H sync。
- compressor online plan 中 `.cpu()` 是明确的 host planner 路径。

MUSA 适配的重点不是改掉这些 metadata CPU op，而是避免新增算子路径再引入额外同步。因此 MUSA backend 倾向 prealloc、inplace `copy_`、固定 layout、显式不支持未适配 scale layout。

## CUDA Graph 相关差异

开源版和 MUSA 版的 `deepseek_v4_backend.py` 基本一致，仍使用：

- `_GraphBucket` 区分 decode/idle、target verify、draft extend
- capture 时保存固定 batch size metadata
- replay 时构造临时 metadata，再 `copy_` 到已捕获对象
- `SGLANG_PREP_IN_CUDA_GRAPH` 下 raw metadata 延迟升级

MUSA 版在模型层增加了 `_is_musa` 后创建 `alt_streams`，使 MUSA 也能进入 multi-stream overlap 逻辑；同时对 HC pre 的 `compile_in_capture_mode` 更保守。这反映出两层约束：

1. CUDA Graph 的对象地址稳定机制可以复用，MUSA 不需要重写 DSV4 metadata 体系。
2. 被 capture 的具体 op 必须是 MUSA graph-safe 的；不稳定的 torch.compile/fallback 路径需要绕开或替换为 TileLang/native kernel。

## JIT C++/CUDA Kernel 适配差异

MUSA 版 `jit_kernel/csrc/deepseek_v4/*.cuh` 的 diff 主要不是算法重写，而是编译与运行时兼容性修补：

- 删除部分 `__grid_constant__` kernel 参数修饰。
- 避免 `using enum`，改为显式 `ForwardMode::...`。
- 将 `std::bit_cast` 改为 `reinterpret_cast`，或减少 C++20 依赖。
- 将 `std::has_single_bit/std::countr_zero` 替换为位运算和 `__builtin_ctz`。
- `cudaFuncSetAttribute` 替换为 `musaFuncSetAttribute`。
- 引入 `<musa_fp8.h>` 处理 MUSA FP8 类型/转换。
- `TensorMatcher` 显式包 `host::details::SizeRef/DeviceRef`，避免模板推导/重载差异。
- `MaskKernel` 简化为只接受 device 上的 `num_token_non_padded`，移除 CPU scalar 分支。

根因：开源 CUDA kernel 同时依赖 NVIDIA runtime API、NVCC 支持的 device C++ 特性、C++20 标准库能力以及 CUDA tensor matcher 的隐式转换。MUSA 编译链/运行时 API 与 CUDA 类似但不完全兼容，所以适配层必须降低语言特性要求、替换 runtime API，并收紧输入设备约束。

## 新增 MUSA 后端的职责

`hardware_backend/layers/deepseek_v4_musa/` 新增了三类核心能力：

1. `cache_ops.py`：FlashMLA/indexer cache pack-store，覆盖 decode、vector、c128、warp-col 等多种实现，并提供大量 env knob。
2. `swiglu_quant_ops.py`：SwiGLU、masked/contiguous post quant、TileKernels quant fallback，明确拒绝 UE8M0、transposed scale、部分 swizzle 未实现路径。
3. `mhc_ops.py`、`hc_head_ops.py`：MHC pre/post、HC head split-K/fused path，用 TileLang 取代 PyTorch fallback 或 CUDA-only fused op。

这说明适配不是简单把 `cuda` 字符串换成 `musa`，而是要重建 DeepSeek V4 的性能关键算子集合。

## 根因洞察

1. **DeepSeek V4 的瓶颈不是普通 PyTorch view/reshape，而是小 batch 大 hidden 的碎 op 和 FP8/cache layout。**  
   `flatten/view/unsqueeze` 本身成本低，真正拖慢的是 `F.linear + norm + sigmoid + sum` 拆成多个 kernel，以及 KV pack-store/SwiGLU quant 这类布局敏感操作。

2. **开源版默认站在 NVIDIA CUDA 生态上，MUSA 需要补齐“等价高性能算子”。**  
   DeepGEMM、CUDA JIT、FP8、dynamic shared memory、PDL、CUDA Graph 都隐含 NVIDIA 语义。MUSA 需要 TileLang/native op 替代，而不是依赖 torch fallback。

3. **CUDA Graph 体系可以复用，但 graph-safe 的 op 集合不同。**  
   DSV4 metadata replay 的 `copy_` 机制没有变，说明抽象层设计合理；但 HC pre compile、SwiGLU quant、cache store 这些具体 op 必须按 MUSA graph capture 能力重新筛选。

4. **适配中很多“限制”是为了防止静默错误。**  
   MUSA backend 对 unsupported UE8M0、transposed scale、swizzle、非 contiguous 输入直接报错，是工程上正确的 fail-closed 策略。否则 fallback 可能引入隐式同步或生成不同 FP8 layout。

5. **C++ kernel diff 体现的是工具链差异，不是算法差异。**  
   `__grid_constant__`、`std::bit_cast`、`std::has_single_bit`、`cudaFuncSetAttribute` 等改动说明 MUSA 适配需要维护一层“CUDA 方言降级”。这类改动应尽量集中封装，否则后续跟进 upstream 会有持续 merge 成本。

## 建议

1. 将 `deepseek_v4.py` 中的 MUSA 分支继续保持薄层分派，核心实现放在 `hardware_backend/layers/deepseek_v4_musa/`，减少 upstream merge 冲突。
2. 给 JIT C++ 建一个小型兼容头，例如统一封装 `FuncSetAttribute`、power-of-two 检查、bit cast、TensorMatcher SizeRef，避免每个 `.cuh` 手工改。
3. 对 `.item/.tolist/.cpu` 建 profiler checklist，区分 CPU mirror 安全路径和 GPU scalar 同步路径。
4. 对 MUSA TileLang op 建立 fallback 策略矩阵：哪些可 torch fallback，哪些必须 fail-closed，哪些只允许 debug fallback。
5. CUDA Graph 验证重点放在 MUSA op 的 graph capture/replay 安全性，而不是 DSV4 metadata 框架本身。
