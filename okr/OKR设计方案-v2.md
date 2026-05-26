# MUSA Driver 整体性能建模 — 方案 v2

> 2026-05-26 | 重新分析 OKR 后的修正方案

---

## 零、关键理解纠正

**OKR1 不是要你写一个 GPU 模拟器，也不是 API 延迟的线性回归。**

```
OKR1 的"性能模型" = 一个分析框架，给定模型负载参数，能预测：
  1. 各 API 的调用次数和单次耗时
  2. 各 kernel 的执行时间分布
  3. 总耗时和瓶颈在哪里
并且预测结果与真实 profiling 数据对齐 (KR2: Top-K overlap ≥90%)
```

**与之前方案的根本区别：**

| 之前的错误理解 | 正确理解 |
|-------------|---------|
| 采集 trace → 拟合 latency=f(size) → 预测新 batch_size 的耗时 | 采集 trace → **分析 API/参数/耗时 的三元关系** → 建立参数化性能公式 → **验证 vs 真实数据** |
| 每个 API 独立建模 | 以模型为维度建模，覆盖 Top90% 累计耗时的 API |
| 输出是预测值 | 输出是 **分析报告**（热点 + 瓶颈 + 优化建议）|

---

## 一、整体框架

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MUSA Driver 性能建模框架                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Phase 1: 数据采集                                                       │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  3个模型 × (推理/训练) × profiler 注入                         │      │
│  │  输出: API trace (811 driver + 519 runtime CBIDs)             │      │
│  │        Kernel trace (名称 + grid/block + GPU ticks)            │      │
│  │        模型元数据 (层数, hidden_dim, num_heads, vocab_size...) │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                              │                                           │
│                              ▼                                           │
│  Phase 2: 热点分析                                                       │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  2.1 累积耗时排序 → 识别 Top90% API                           │      │
│  │  2.2 按模型阶段拆分 (Prefill / Decode / Warmup / Load)       │      │
│  │  2.3 API 调用频次 vs 模型参数的关系                           │      │
│  │  2.4 Kernel 热力图 (哪些 kernel 占 GPU 时间最多)               │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                              │                                           │
│                              ▼                                           │
│  Phase 3: 性能建模                                                       │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  对 Top90% API 建立:                                          │      │
│  │    - 调用次数预测: N_api = f(batch_size, seq_len, n_layers)  │      │
│  │    - 单次耗时预测: T_api = g(input_size, grid, stream)       │      │
│  │    - 阶段耗时 = Σ (N_api × T_api) per phase                  │      │
│  │                                                               │      │
│  │  对 Top50 Kernel 建立:                                        │      │
│  │    - 执行时间分布: p50/p90/p99                               │      │
│  │    - 按 kernel 类型的加速比分析                                │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                              │                                           │
│                              ▼                                           │
│  Phase 4: 验证 + 报告                                                    │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  KR2: 模型预测 vs profiling 实测 → Top50 kernel overlap ≥90% │      │
│  │  KR3: 3份报告 = API 热点 + Kernel 热点 + 优化建议              │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 二、3 个模型选型 & 采集维度

| # | 模型 | 类型 | 框架 | 采集维度 | 关键观察点 |
|---|------|------|------|---------|-----------|
| **M1** | Qwen2.5-7B | Dense LLM 推理 | SGLang | batch=1/4/8, seq_in=256/512/1024, seq_out=64/128 | KV cache 增长对 memcpy 的影响，decode 阶段的 kernel 模式 |
| **M2** | DeepSeek-V2-Lite | MoE 推理 | SGLang | batch=1/4, seq_in=256/512, top-k=2/4 | expert routing 的 event 同步模式，all-to-all 通信 |
| **M3** | LLaMA-3-8B | Dense 训练 | PyTorch MUSA FSDP | batch=4/8, seq=2048, grad_accum=1/4 | backward 的 graph replay + all-reduce 通信 |

每个模型至少采集 **4 个配置点**（不同 batch/seq），用于建模时区分调用次数 vs 单次耗时的变化。

---

## 三、Phase 2: 热点分析 — 从 trace 到 Top90%

### 3.1 API 累积耗时排序

```python
# 从 profiler trace 计算
api_stats = defaultdict(lambda: {"total": 0, "count": 0, "self": 0})

for event in trace:
    api_stats[event.name]["count"] += 1
    api_stats[event.name]["total"] += event.duration
    api_stats[event.name]["self"] += event.self_time   # 扣除子调用

# 按 total 降序排列
ranked = sorted(api_stats.items(), key=lambda x: x[1]["total"], reverse=True)

# 取累计占比 90%
cumulative = 0
top90_apis = []
for name, stats in ranked:
    cumulative += stats["total"]
    top90_apis.append(name)
    if cumulative / total_trace_time >= 0.90:
        break
```

**对 Qwen-7B 推理的预期结果（基于 test4 已有数据）：**

| 排名 | API | 耗时占比 | 调用次数 | 类型 |
|------|-----|---------|---------|------|
| 1 | muModuleLoadData | ~50% | 23 | 一次性(Load) |
| 2 | muMemcpyHtoDAsync | ~6% | 429 | Per-prefill |
| 3 | muLaunchKernel | ~1.3% | 53,387 | Per-token |
| 4 | muStreamCreateWithPriority | ~0.7% | 96 | 初始化 |
| 5 | muCtxGetDevice | ~0.5% | 550,877 | 高频低延迟 |
| ... | ... | ... | ... | ... |

### 3.2 按阶段拆分

trace 中插入阶段标记（Prefill / Decode / Warmup），拆分后分别分析：

```
Prefill 阶段:
  API 热点: muMemcpyHtoDAsync (输入拷贝), muLaunchKernel (attention + MLP)
  特征: 调用次数少, 单次数据量大, kernel launch 密集

Decode 阶段:
  API 热点: muLaunchKernel (per-token attention + MLP), muCtxGetDevice (高频)
  特征: 调用次数多, 单次数据量小, latency 敏感

Warmup 阶段:
  API 热点: muModuleLoadData, muMemAlloc, muStreamCreate
  特征: 一次性开销, 与模型大小直接相关
```

---

## 四、Phase 3: 性能模型构建

### 4.1 调用次数模型: N_api = f(模型参数)

对每个 Top90% API，从多个配置点（不同 batch/seq）的 trace 中回归出调用次数与模型参数的关系：

```python
# 示例: muLaunchKernel 的调用次数
# 观测数据:
#   batch=1, seq=256, n_layers=32: N_launch = 53,387
#   batch=1, seq=512, n_layers=32: N_launch = 54,200
#   batch=4, seq=256, n_layers=32: N_launch = 53,800

# 模型: 
#   N_launch ≈ n_prefill_launches + n_decode_launches × output_len
#   其中 n_prefill_launches ≈ n_layers × 2 (attention + MLP per layer)
#        n_decode_launches  ≈ n_layers × n_kernels_per_token
```

| API | 预测公式 | 关键参数 |
|-----|---------|---------|
| muLaunchKernel | `n_layers × (K_prefill(batch,seq) + K_decode × out_len)` | n_layers, batch, seq_len, out_len |
| muMemcpyHtoDAsync | `n_prefill_copies × batch` | batch, seq_len |
| muCtxGetDevice | `n_kernel_launches × C` (几乎1:1) | — |
| muStreamCreateWithPriority | `n_streams` (常量) | — |
| muModuleLoadData | `n_modules` (常量, 但每次 load 的数据量 = f(model_size)) | model_size |

### 4.2 单次耗时模型: T_api = g(input_params)

```python
# muMemcpyHtoDAsync: T ≈ α + β × size
# 观测: size=4KB → 2.5ms, size=64KB → 4.2ms, size=1MB → 12ms
# 模型: T_memcpy = 2.3 + 0.01 × size(KB)   [ms]

# muLaunchKernel: T ≈ α + β × grid.x × block.x
# 观测: grid(128,1,1)×block(256,1,1) → 4.2µs
#       grid(1024,1,1)×block(128,1,1) → 6.2µs
# 模型: T_launch = 1.5 + 0.002 × grid.x × block.x   [µs]
```

关键是：这个"模型"**不需要精确**，它只要能区分"调用次数驱动"和"规模驱动"两类 API，就能用于瓶颈分析。

### 4.3 阶段耗时 = Σ(N_api × T_api)

```python
def predict_phase_time(phase, params):
    total = 0
    for api in top90_apis[phase]:
        N = predict_call_count(api, params)
        T = predict_single_time(api, params)
        total += N * T
    return total

# Prefill:    T_prefill  = Σ(prefill APIs, batch, seq_len)
# Decode:     T_decode   = Σ(decode APIs, batch, out_len) × output_tokens
# One-time:   T_onetime  = Σ(load APIs, model_size)
```

---

## 五、KR2: 对齐验证

```python
def validate_kr2(model_predictions, profiling_data):
    """
    模型预测的 Top50 kernel vs 实际 profiling 的 Top50 kernel
    
    overlap = |model_top50 ∩ prof_top50| / 50
    
    目标: overlap ≥ 90% (即 45/50 个 kernel 一致)
    """
    model_kernels  = sorted(model_predictions.kernels, 
                           key=lambda k: k.predicted_time, reverse=True)[:50]
    prof_kernels   = sorted(profiling_data.kernels,
                           key=lambda k: k.actual_time, reverse=True)[:50]
    
    model_names = {k.name for k in model_kernels}
    prof_names  = {k.name for k in prof_kernels}
    
    overlap_count = len(model_names & prof_names)
    overlap_rate  = overlap_count / 50
    
    # 排位差异分析
    for k_name in model_names & prof_names:
        model_rank = rank_in(k_name, model_kernels)
        prof_rank  = rank_in(k_name, prof_kernels)
        rank_diff  = abs(model_rank - prof_rank)
        if rank_diff > 10:
            log_warning(f"{k_name}: 排名偏差 {rank_diff}")
    
    return overlap_rate >= 0.90
```

**什么情况下 overlap 不到 90%？**
- 模型没覆盖某些 kernel（模型参数化不完整）
- 某个 kernel 的调用次数模型完全预测错了
- profiler 的 kernel name 解析失败（未命名 kernel 被归到 `unknown`）

---

## 六、KR3: 报告模板

每份报告（3个模型 × 1份 = 3份）包含：

### 6.1 API 热点

| 排名 | API | 总耗时 | 占比 | 调用次数 | 单次耗时 | 调用次数敏感参数 | 耗时敏感参数 |
|------|-----|--------|------|---------|---------|---------------|------------|
| 1 | muModuleLoadData | 8.9s | 50% | 23 | 387ms | n_modules | model_size |
| 2 | muMemcpyHtoDAsync | 1.1s | 6% | 429 | 2.5ms | batch × seq_len | size |
| 3 | muLaunchKernel | 223ms | 1.3% | 53,387 | 4.2µs | n_layers × out_len | grid × block |
| 4 | muStreamCreate | 129ms | 0.7% | 96 | 1.3ms | n_streams | — |
| 5 | muCtxGetDevice | 91ms | 0.5% | 550,877 | 0.17µs | ∝kernel_launches | — |

### 6.2 Kernel 热点

| 排名 | Kernel | GPU 总耗时 | 调用次数 | 平均耗时 | 所属阶段 | 所属 op |
|------|--------|-----------|---------|---------|---------|--------|
| 1 | flash_attn_fwd | 120ms | 12,800 | 9.4µs | Decode | Attention |
| 2 | gemm_fp16 | 95ms | 6,400 | 14.8µs | Prefill+Decode | MLP |
| 3 | rms_norm | 45ms | 19,200 | 2.3µs | Decode | LayerNorm |
| 4 | silu_mul | 38ms | 6,400 | 5.9µs | Decode | MLP |
| 5 | rotary_emb | 22ms | 12,800 | 1.7µs | Decode | Attention |

### 6.3 优化建议

基于热点分析，给出量化优化建议：

```
1. muModuleLoadData (50%耗时, 23次调用):
   问题: 每次加载 387ms, 23个分片顺序加载
   建议: 并行加载多个 module 分片, 预期减少 60% 加载时间 (→ 3.5s)

2. muCtxGetDevice (0.5%耗时, 550K 调用):
   问题: 每次 0.17µs, 但 550K 次累计 91ms
   建议: 在 kernel launch 路径中缓存 device handle, 避免每次 TLS 查询

3. muMemcpyHtoDAsync (6%耗时, 429 调用):
   问题: 单次 2.5ms, 429 次累计 1.1s
   建议: 对 prefilling 阶段使用更大的连续 buffer 减少 memcpy 次数
```

---

## 七、OKR 2: 行为分析 & CTS（简述）

与 OKR1 并行推进，两个 OKR 共享采集阶段的数据：

```
Phase 1 (共享): 采集 3个模型的 profiler trace
       │
       ├──→ OKR1: 热点分析 → 性能模型 → 报告
       │
       └──→ OKR2: 行为模式提取 → CTS 用例 → 基线 → 版本差异报告

OKR2 行为特征:
  F1: Attention Pattern — LaunchKernel + MemcpyAsync + Event + StreamWait
  F2: MLP Pattern — LaunchKernel + MemAllocAsync + MemFreeAsync  
  F3: KV Cache — MemcpyDtoDAsync + MemAllocAsync + StreamSync
  F4: Expert Routing — LaunchKernel(topk) + Allocate + Event + AllToAll
  F5: Graph Execution — GraphInstantiate + GraphLaunch + StreamCapture
  F6: Memory Pool — MemPoolCreate + MemAllocFromPoolAsync + TrimTo

每个特征编码为 CTS C++ 用例 → CSV 注册 → CI 自动跑
```

---

## 八、里程碑

```
Week 1: Profiler 增强 (参数捕获) + 3模型采集环境搭建
Week 2: M1 (Qwen) 4配置点采集 → 热点分析 → Top90% API → 初始模型
Week 3: M2 (MoE) + M3 (训练) 采集 → 热点分析
Week 4: 三模型性能模型构建 → KR2 对齐验证
Week 5: 行为特征提取 → CTS 用例开发 + CSV 注册
Week 6: 分析报告 (API+kernel 热点 + 优化建议) + 行为基线 + 版本差异框架
```
