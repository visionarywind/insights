# Profile 对比实验方法（us 单位）

## 1. 采集目标

同一模型配置下分别采集两类数据：

- SGLang profiler：观察 PyTorch/SGLang 层 op、compiled region、runtime/driver API、kernel 的时间轴关系。
- API-side activity profiler：观察 MUSA runtime/driver API latency 和底层 kernel device time。

两类数据的窗口不同，不直接比较总耗时；重点比较热点 kernel、API 类型和耗时方向。

## 2. 统一配置

两轮实验固定以下配置：Qwen3-8B、batch=1、input_len=16、output_len=8、dtype=float16、device=musa、关闭 cudagraph。

## 3. SGLang Profiler 数据转换

SGLang profiler 产物是 `.trace.json.gz`。处理步骤：

1. 用 `gzip.open(path, "rt")` 解压。
2. 读取 JSON 中的 `traceEvents`。
3. 读取每个事件的 `cat`、`name`、`dur`。
4. Chrome trace 的 `dur` 原始单位是 `us`，直接累加，不转换为 ms。
5. 按 `cat` 汇总 `kernel`、`gpu_memcpy`、`privateuse1_runtime`、`privateuse1_driver`。
6. 按 `name` 汇总 kernel/API 的调用次数、total_us、avg_us、max_us。

输出文件：

- `derived/sglang_trace_category_summary_us.tsv`
- `derived/sglang_trace_gpu_api_summary_us.tsv`

## 4. API-side Activity 数据转换

API-side activity profiler 的有效数据来自 `api_activity_profile_console/console.log`。文本中有两类关键表：

- `API latency by muXXX/musaXXX name`
- `Kernel/device time by kernel symbol`

每行格式为：

```text
<name> <calls> <total_ms> <avg_us> <max_us>
```

处理步骤：

1. 按 section 标题定位当前表。
2. 跳过表头、空行和分隔行。
3. 使用 `rsplit(None, 4)` 从右侧拆分出 4 个数值字段。
4. 将 `calls` 转成整数。
5. 将 `total_ms` 转成 `total_us = total_ms * 1000`。
6. `avg_us` 和 `max_us` 已经是 `us`，不再转换。
7. 按 `total_us` 排序得到 top API 和 top kernel。

输出文件：

- `derived/api_activity_summary_us.tsv`

## 5. 写入文件策略

保留三类文件：

- 原始数据：trace、stdout、stderr、console.log。
- 派生数据：TSV/JSON，统一 us 单位，便于脚本二次处理。
- 结论报告：Markdown，直接面向阅读和复查。

本轮输出：

- `comparison_report_us.md`：对比结论。
- `comparison_methodology_us.md`：实验方法和数据转换说明。
- `derived/validation_summary.json`：运行与解析校验摘要。

## 6. 对比原则

- SGLang `kernel` 对齐 API-side `Kernel/device time`，用于判断设备侧热点是否一致。
- SGLang `privateuse1_runtime/privateuse1_driver` 对齐 API-side API latency，用于判断 runtime/driver 调用类型。
- `aten::linear`、`Torch-Compiled Region`、`sgl_kernel::fused_add_rmsnorm` 属于框架视角，只能在 SGLang trace 中看调用链。
- 不直接比较两边总耗时，因为采集窗口不同。
## 7. 原始时间线保留

为了避免文件名排序造成误读，额外保留一份按 Chrome trace `ts` 从早到晚排序的 GPU 相关原始时间线：

- `raw_timeline/raw_gpu_timeline_us.tsv`
- `raw_timeline/raw_gpu_timeline_head.md`

时间线字段包括 `stage`、`ts_us`、`relative_ts_us`、`dur_us`、`cat`、`name`、`pid`、`tid`、`external_id`、`correlation`。该文件以真实时间戳排序，开头是 prefill，随后才是 decode。

