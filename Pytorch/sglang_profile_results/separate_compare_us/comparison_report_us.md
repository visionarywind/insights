# SGLang Profile 与 API Profile 对比（us 单位）

## 数据位置

新一轮验证结果保存在当前目录，原始数据与派生数据分开存放：

- `sglang_profile_only/`：SGLang 内置 profiler 原始输出。
- `sglang_profile_only/traces/`：prefill/decode Chrome trace，格式为 `.trace.json.gz`。
- `api_activity_profile_console/console.log`：API-side activity profiler 的完整控制台输出。
- `derived/`：解析后的 TSV/JSON 文件，所有耗时统一使用 `us`。
- `raw_timeline/`：从远端恢复的原始 trace 与按 Chrome trace `ts` 排序的 GPU 时间线。

## 实验配置

两轮实验使用同一配置：Qwen3-8B、batch=1、input_len=16、output_len=8、dtype=float16、device=musa、关闭 cudagraph。

## SGLang Profile 统计

Chrome trace 的 `dur` 字段原始单位就是 `us`，解析时不做 ms 转换。

- prefill kernel：628 次，total 16726.441 us。
- decode kernel：555 次，total 16343.556 us。
- prefill + decode 合计 kernel：1183 次，total 33069.997 us。
- gpu memcpy：13 次，total 33.881 us。
- privateuse1 runtime：949 次，total 8316.720 us。
- privateuse1 driver：258 次，total 1305.986 us。

SGLang trace 中 GPU/API 热点：

- `musa_asm_hhhhssgemm_nt_tce_128_32x128B128_epilogue`：144 次，total 11415.698 us，avg 79.276 us，max 150.805 us。
- `void musa::dnn::(anonymous namespace)::batch_gemv_row_continuous_kernel<__half, (musa::dnn::AlignAttr)1, 128, 64, 1024, false, true, 128>(__half*, __half const*, __half const*, __half const*, __half const*, int, int, int, int, long, int, long, int, long, int, long, float, float, float)`：110 次，total 9695.966 us，avg 88.145 us，max 847.668 us。
- `musaLaunchKernel`：781 次，total 4169.231 us，avg 5.338 us，max 23.703 us。
- `void musa::dnn::(anonymous namespace)::batch_gemv_row_continuous_kernel<__half, (musa::dnn::AlignAttr)1, 256, 32, 1024, false, true, 128>(__half*, __half const*, __half const*, __half const*, __half const*, int, int, int, int, long, int, long, int, long, int, long, float, float, float)`：36 次，total 3557.635 us，avg 98.823 us，max 100.883 us。
- `musaMemcpyAsync`：13 次，total 1679.425 us，avg 129.187 us，max 242.729 us。
- `musaDeviceSynchronize`：5 次，total 1433.958 us，avg 286.792 us，max 708.024 us。
- `void LayerNormGlobalKernelVlen<__half, float, 1024, 1, 8>(__half*, __half*, __half const*, unsigned long, unsigned long, float)`：144 次，total 1397.052 us，avg 9.702 us，max 11.360 us。
- `muLaunchKernel`：258 次，total 1305.986 us，avg 5.062 us，max 10.944 us。

## API-side Activity Profile 统计

API activity 输出中的 `total_ms` 已转换为 `total_us = total_ms * 1000`，`avg_us` 和 `max_us` 保持原始单位。

API latency top 项：

- `musaLaunchKernel_v7000`：4674 次，total 7633483.000 us，avg 1633.180 us，max 2379837.587 us。
- `musaMemcpyAsync_v3020`：430 次，total 1085954.000 us，avg 2525.474 us，max 64975.295 us。
- `muMemSetAccess`：224 次，total 417727.000 us，avg 1864.851 us，max 6976.242 us。
- `musaStreamCreateWithPriority_v5050`：96 次，total 118159.000 us，avg 1230.821 us，max 2329.670 us。
- `musaLaunchKernelExC_v11060`：2321 次，total 101415.000 us，avg 43.695 us，max 47893.158 us。
- `muMemCreate`：3405 次，total 23059.000 us，avg 6.772 us，max 312.106 us。
- `musaDeviceSynchronize_v3020`：37 次，total 12876.000 us，avg 347.993 us，max 852.742 us。
- `muLaunchKernel`：2280 次，total 9434.000 us，avg 4.138 us，max 39.081 us。

Kernel device time top 项：

- `musa_asm_hhhhssgemm_nt_tce_128_32x128B128_epilogue_stage3_btmenc`：592 次，total 93686.000 us，avg 158.254 us，max 836.428 us。
- `musa_asm_hhhhssgemm_nt_tce_256_32x128B128_epilogue_stage3_btmenc`：1728 次，total 83429.000 us，avg 48.281 us，max 77.922 us。
- `_ZN4musa3dnn12_GLOBAL__N_110KernelFillI6__halfLb0ELb0ELb0ELNS1_13BroadcastModeE0ENS0_12D`：219 次，total 40939.000 us，avg 186.936 us，max 588.180 us。
- `_fwd_grouped_kernel_stage1`：504 次，total 10924.000 us，avg 21.674 us，max 30.201 us。
- `_Z25LayerNormGlobalKernelVlenI6__halffLi1024ELi1ELi8EEvPT_S2_PKS1_mmT0_`：1152 次，total 8167.000 us，avg 7.089 us，max 9.721 us。
- `_ZN4musa3dnn25LayerNormGlobalKernelVlenI6__halffS2_fLi64ELi1ELi8ELNS0_8NormTypeE1ELb0EEE`：1080 次，total 6153.000 us，avg 5.697 us，max 7.360 us。
- `_ZN2at6native12_GLOBAL__N_121IndexVectorizedKernelIZNS1_14GPUIndexKernelIN3c104HalfELi1E`：1008 次，total 5683.000 us，avg 5.638 us，max 6.361 us。
- `triton_poi_fused_add_cat_index_select_mul_split_sub_unsqueeze_view_0`：576 次，total 4367.000 us，avg 7.581 us，max 12.560 us。

API-side 汇总：

- API latency 表合计：173555 次，total 9423276.000 us。
- Kernel device time 表合计：9275 次，total 268799.000 us。

## 对比结论

- SGLang profiler 的优势是能看到 PyTorch/SGLang 层的 `aten`、compiled region、MUSA runtime/driver、kernel 时间轴关系。
- API-side activity profiler 的优势是能按 MUSA API 和底层 kernel 聚合整进程热点。
- 两边统计窗口不同：SGLang trace 覆盖 profiler 标记的 prefill/decode 区间；API-side activity 覆盖更大的进程范围，包括 warmup、benchmark 以及首次触发开销。
- 因此不要直接相减总耗时，应对齐热点方向：GEMM、GEMV/attention、LayerNorm、Triton fused kernel 在两边都能看到，是主要分析对象。

## 校验结果

- SGLang profiler exit code：0。
- API activity profiler 控制台输出包含 `EXIT=0`：True。
- SGLang trace 文件按执行顺序为：sglang_profile_only_compare_batch1_input16_output8_prefill.trace.json.gz, sglang_profile_only_compare_batch1_input16_output8_decode.trace.json.gz。
- 结构化解析结果：`derived/sglang_trace_category_summary_us.tsv`、`derived/sglang_trace_gpu_api_summary_us.tsv`、`derived/api_activity_summary_us.tsv`。
- 原始时间线：`raw_timeline/raw_gpu_timeline_us.tsv`，按 `ts` 从早到晚排序，第一条 GPU 相关事件为 prefill，最后一条为 decode。
