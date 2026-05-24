# SGLang DeepSeek V4 PyTorch 常见 Op 与 CUDA Graph 分析

## 源码范围

本文基于 `/home/mtuser/workspace/sglang`，重点覆盖 DeepSeek V4 相关源码：

- `python/sglang/srt/models/deepseek_v4.py`：模型主体，MQA、HC、MoE、logits 入口。
- `python/sglang/srt/layers/attention/deepseek_v4_backend.py`：DSV4 attention backend、metadata、CUDA Graph capture/replay 适配。
- `python/sglang/srt/layers/attention/dsv4/`：compressor、indexer、metadata、metadata Triton kernel、KV quant/store。
- `python/sglang/jit_kernel/deepseek_v4.py` 与 `jit_kernel/csrc/deepseek_v4/`：topk、compress、RoPE、RMSNorm、store、paged metadata 等 JIT/CUDA kernel wrapper。
- `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`、`deepseek_v4_compress_state.py`：SWA/full/compressed KV cache 和 compress state buffer。
- `python/sglang/srt/model_executor/cuda_graph_runner.py`、`model_runner.py`、`compilation/cuda_piecewise_backend.py`：普通 CUDA Graph、piecewise CUDA Graph、torch.compile capture 管理。
- `python/sglang/srt/hardware_backend/layers/deepseek_v4_musa/`：MUSA 侧 HC/cache/swiglu quant operator facade 与 kernel 封装。

## PyTorch 常见 Op 在 DSV4 中的角色

| Op 类别 | 常见调用 | 主要用途 | 性能关注点 |
| --- | --- | --- | --- |
| 分配 | `torch.empty/zeros/ones/full`、`new_empty/new_zeros` | graph 输入 buffer、KV cache、metadata、临时输出 | `empty` 不初始化，必须保证后续完整写入 |
| 复制 | `copy_`、`torch._foreach_copy_` | replay 前把真实 batch 写入固定 graph buffer | replay 热路径核心，减少 Python 循环和 launch |
| 形状变换 | `view/reshape/flatten/unsqueeze/squeeze/contiguous` | MQA head/group 布局、HC residual、多路 TP/CP 切分 | `contiguous` 可能触发拷贝 |
| 索引 | slice、advanced indexing、`gather`、`masked_fill_`、`tensor_split` | page table、SWA window、TP/CP token 分片、padding mask | advanced indexing 会生成新 tensor |
| 序列生成 | `arange`、`repeat_interleave`、`expand` | prefill causal seq_lens、req index repeated、SWA offsets | graph 内需固定 shape |
| 数学 | `rsqrt/square/mean/sum/sigmoid/clamp` | HC pre/post、RMS-like norm、SWA topk length clamp | 多个小 op 可被 fused kernel 替代 |
| 线性代数 | `F.linear`、`torch.einsum` | HC fallback、`wo_a` grouped projection、indexer torch reference | 生产路径多用 DeepGEMM/TileLang/JIT |
| 设备转换 | `.to(dtype/device)`、`.cpu()`、`pin_memory`、`non_blocking` | dtype 对齐、CPU plan、host-device metadata 搬运 | `.cpu()` 和标量化会同步 |

## 模型主路径

`MQALayer.forward` 中，TP 场景用 `x.new_empty(x.shape[0], n_heads, head_dim)` 预分配全局 Q buffer，再通过 `q_out = q_padded[:, tp_slice, :]` 写入本 rank 片段。attention 输出后执行 `fused_rope(..., inverse=True)`，再 `view(T, local_groups, -1)` 转 grouped layout。`wo_a` 可走两条路径：默认 `torch.einsum("tgd,grd->tgr", o, wo_a)`，FP8 优化路径走 `sglang_per_token_group_quant_fp8 + deep_gemm.fp8_einsum`。

HC 路径中，`hc_pre` 的 torch fallback 使用 `flatten(1).float()`、`square().mean()`、`torch.rsqrt()`、`F.linear()` 和 `unsqueeze()` 生成 mixture；`hc_post` 使用 `unsqueeze`、broadcast multiply、`sum(dim=1)` 合成 residual。当前源码中 `hc_post_torch_impl` 带 `@compile_in_capture_mode`，在 CUDA Graph capture mode 下会触发 `torch.compile`；`hc_pre_torch_impl` 的装饰器被注释，说明这份分支对 pre 侧 capture/compile 兼容性更谨慎。

MoE/TP/CP 相关数据移动大量使用 `tensor_split`、`contiguous`、`torch.empty_like`、`torch.cat`。这些不是显式同步点，但会影响 graph replay 中的内存地址稳定性和临时 buffer 生命周期。

## Attention Metadata 与 DSV4 子模块

`deepseek_v4_backend.py` 的 metadata 负责把 request 级输入转成 attention kernel 可消费的结构：

- `expand_prefill_casually`：用 `torch.empty` 分配 `seq_lens_casual` 和 `idx_to_req_repeated`，循环内用 `torch.arange(..., out=out)` 写 GPU buffer。
- `make_core_attn_metadata`：通过 `req_to_token[req_pool_indices_repeated, :max_seq_len:page_size]` 生成 page table，`// page_size` 后 `.to(torch.int32)`。
- `get_swa_page_indices`：构造 `offsets = pos.unsqueeze(1) - arange(SWA_WINDOW)`，用 `masked_fill_` 清非法 offset，再映射 full KV loc 到 SWA loc。
- `DSV4AttnMetadata.init_compression_metadata`：调用 `dsv4/metadata_kernel.py` 的 Triton kernel，一次性生成 c4/c128 out loc、positions、seq_lens 和 c128 page indices。

`dsv4/compressor.py` 负责生成 compress metadata。prefill 路径先用 `triton_create_paged_compress_data` 在 GPU 上生成 `write_loc/extra_data`，再调用 `CompressorPrefillPlan.generate`。当已有 CPU list 时，会构造 `torch.tensor(seq_lens_cpu)` 和 `torch.tensor(extend_lens_cpu)` 走 CPU/host 规划；CUDA Graph replay 路径可传 `seq_lens=None` 的 raw metadata，延迟到 graph 内 lazy upgrade。

`dsv4/indexer.py` 包含 torch reference topk/indexer：`torch.topk`、`torch.where`、`torch.gather`、`torch.cat`、`torch.tensor(-1, device=...)`。这类实现更适合 fallback 或验证；实际高性能路径由 `jit_kernel/deepseek_v4.py` 的 `topk_transform_512/_v2` 和 C++/CUDA wrapper 承担。

## 普通 CUDA Graph 使用

普通 CUDA Graph 由 `ModelRunner.init_cuda_graphs()` 创建，非 CPU 且未 `--disable-cuda-graph` 时使用 `CudaGraphRunner`。初始化时 `_dummy_run()` 先构造固定形状 `DecodeInputBuffers`，包括 GPU 上的 `input_ids/seq_lens/out_cache_loc/positions/next_token_logits_buffer`，以及真正位于 CPU 的 `seq_lens_cpu`。

`CudaGraphRunner.capture_one_batch_size` 对每个 capture batch size 做：

1. 从固定 buffer 切片形成 graph input，例如 `input_ids[:num_tokens]`、`seq_lens[:bs]`。
2. 构造 capture 用 `ForwardBatch`，其 `seq_lens_sum=seq_lens.sum().item()` 发生在 capture 前 warmup，不在 replay 热路径。
3. 调用 `attn_backend.init_forward_metadata_capture_cuda_graph(...)`，DSV4 backend 为不同 `_GraphBucket` 保存 capture-time metadata。
4. `run_once()` warmup 两次，调用 `attn_backend.on_after_cuda_graph_warmup()` 重建 FlashMLA metadata，并在 `SGLANG_PREP_IN_CUDA_GRAPH=1` 时把 full metadata 恢复为 raw metadata，确保真正 capture 时在 graph 内升级。
5. 用 `torch.cuda.CUDAGraph()` 和 `torch.cuda.graph(graph, pool=..., stream=...)` 捕获。MUSA 分支注释说明这里避免使用 `cuda_graph=` keyword，改为位置参数调用。

Replay 前，`DecodeInputBuffers.populate_from_forward_batch` 用 `torch._foreach_copy_` 批量复制 GPU tensor，并单独复制 CPU `seq_lens_cpu`。随后 `attn_backend.init_forward_metadata_replay_cuda_graph(...)` 按真实 batch 构造临时 metadata，再用 `chosen_metadata.copy_(temp_metadata)` 原地更新 capture 时对象，保证 graph 内 tensor 地址稳定。最后 `self.graphs[graph_key].replay()` 执行。

## DeepSeek V4 CUDA Graph Metadata 设计

DSV4 backend 把 graph metadata 按 `_GraphBucket` 和 `bs` 保存：

- `DECODE_OR_IDLE`：decode 和 idle 共用 bucket；idle replay 时构造全 1 的 `seq_lens`、全 0 的 req/out loc。
- `TARGET_VERIFY`：speculative verify 使用 `speculative_num_draft_tokens * bs` 的固定 token 数。
- `DRAFT_EXTEND`：使用 `draft_extend_num_tokens_per_bs` 固定每 batch token 数。

关键点是 `copy_` 语义：`dsv4/metadata.py::copy_metadata` 对 tensor 字段调用 `dst.copy_(src)`，对 FlashMLA metadata 等特殊字段允许 assign。这样 replay 可以更新数据内容而不换对象地址，符合 CUDA Graph 对输入地址稳定的要求。

## Piecewise CUDA Graph

`model_runner.py::init_piecewise_cuda_graphs` 在 `piecewise_cuda_graph_tokens` 配置存在时启用。它遍历语言模型 layers，DeepSeek V4 通过 `layer.self_attn.attn_mqa` 识别 attention layer，通过 `layer.mlp.experts` 识别 MoE。普通 `PiecewiseCudaGraphRunner` 或实验性的 `BreakableCudaGraphRunner` 捕获模型片段。

`compilation/cuda_piecewise_backend.py` 结合 torch.compile 和 CUDA Graph：第一次走 general compiled graph；指定 shape 先 warmup 一次，然后用 `torch.cuda.CUDAGraph()` 捕获 `entry.runnable(*args)`；replay 时校验 tensor data_ptr（debug mode）并调用 `entry.cudagraph.replay()`。piecewise 的收益是减少整图 capture 的动态约束，但要求每个 piece 的输入地址和 shape 稳定。

## 同步与 CPU Op 风险点

| 调用 | 位置/场景 | 风险 | 说明 |
| --- | --- | --- | --- |
| `torch.cuda.synchronize()` / `device_module.synchronize()` | graph warmup、模型 set embed/head、benchmark | 显式全设备等待 | capture 前用于稳定状态；热路径应避免 |
| `.item()` | `seq_lens_cpu.max().item()`、`seq_lens.sum().item()`、`req_pool_indices_repeated[-1].item()` | 若来源是 GPU tensor 会 D2H 同步 | `seq_lens_cpu` 通常是 CPU tensor；padding value 从 GPU 标量取值需警惕 |
| `.tolist()` | `seq_lens.tolist()`、`seq_lens_cpu.tolist()`、CP metadata | GPU tensor 上会同步 | DSV4 大多通过 CPU mirror `seq_lens_cpu` 规避 |
| `.cpu()` | compressor online plan、state capturer、测试/benchmark | D2H 拷贝并可能同步 | 若必须 host planning，应用 pinned memory/异步 copy 降低影响 |
| `torch.tensor(list).to(device, non_blocking=True)` | `_move_to_device`、metadata 上传 | pinned 时可异步 | 后续 GPU 消费仍受 stream dependency 约束 |
| `F.pad(value=tensor.item())` | prefill padding req id | 单标量同步风险 | 可考虑维护 CPU mirror 或 GPU kernel padding |

## 结论

DeepSeek V4 的 PyTorch op 主要承担三类工作：固定形状 buffer 管理、metadata 构造、fallback/可编译表达式。真正大计算集中在 Triton、TileLang、DeepGEMM、JIT CUDA/MUSA kernel。CUDA Graph 的核心约束是 shape 和地址稳定，因此 SGLang 使用预分配 input buffers、`copy_` 原地更新、metadata `copy_` replay、capture bucket 和 raw/full metadata lazy upgrade 来适配 DSV4 的动态请求。

性能分析时应优先看三类点：第一，replay 前的 GPU/CPU copy 是否过多；第二，`item/tolist/cpu` 是否落在 decode 每步热路径；第三，metadata 构造是否仍由多个小 PyTorch op 完成而没有融合到 Triton/JIT kernel。普通 `view/empty/clamp` 本身通常不是瓶颈，真正需要警惕的是会打断异步执行的标量化和 host-device 迁移。
