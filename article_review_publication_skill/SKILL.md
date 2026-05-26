---
name: article-review-publication
description: Use this skill when reviewing or rewriting Chinese technical articles for publication quality. It focuses on removing colloquial wording, vague section names, circular explanations, and unsupported claims; improving source-code analysis, code comments, examples, and final validation.
---

# Article Review For Publication

## When To Use

Use this skill for Chinese technical documents, especially articles that explain PyTorch OPs, model source code, runtime behavior, performance analysis, GPU/MUSA synchronization, graph replay, MoE, attention, or driver/API modeling.

Do not use it for casual notes unless the user asks for publication quality, professional wording, review, audit, cleanup, or removal of colloquial and unclear expressions.

## Output Standard

The document must be:

- Professional: use precise engineering terms and avoid casual wording.
- Direct: explain the behavior first, then provide source evidence or an example.
- Source-based: tie conclusions to code paths, APIs, tensor shapes, runtime behavior, or profiler signals.
- Reproducible: key examples should be executable or numerically checkable.
- Publishable: no process notes such as "document goal", no self-referential wording, no filler.

## Review Workflow

1. Identify the document scope and target audience.
2. Check title and section names.
3. Remove colloquial, vague, and circular expressions.
4. Verify each technical claim against source code, API behavior, examples, or measured output.
5. Ensure each code block has comments that explain execution meaning, not syntax trivia.
6. Ensure each key module has a minimal example or calculation path.
7. Check whether performance conclusions name the concrete trigger: copy, sync, allocation, dynamic shape, kernel count, dtype cast, layout conversion, or graph boundary.
8. Run a final textual scan for banned patterns and a Markdown fence check.

## Section Naming Rules

Use direct professional names:

- `PyTorch OP 的基本行为`
- `Llama Decoder 源码拆解`
- `DeepSeek V4 Attention`
- `DeepSeek V4 MoE`
- `OP 与性能排查`
- `源码分析步骤`

Avoid vague names:

- `OP 视角`
- `从 OP 看源码`
- `源码阅读方法`
- `文档目标`
- `整体介绍`
- `一些问题`

## Language Rules

Replace casual or vague wording with direct technical wording.

| Avoid | Prefer |
|---|---|
| 看一下 / 看懂 / 看出来 | 检查 / 识别 / 验证 |
| 这个 / 它 | 具体模块名、变量名或 API 名称 |
| 真实 copy | 实际数据复制 |
| 打分 | 评分 |
| 拼到 | 拼接到 |
| 放进 | 放入 / 用于 |
| 不适合 | 不应用于 / 不满足要求 |
| 不能仅依据 | 不应仅依据 |
| 这就是 | 该结构属于 / 该操作表示 |
| 很复杂 / 更复杂 | 增加某个阶段 / 引入某个模块 |
| 只要 / 最好 / 比较 | remove or replace with a concrete condition |

Avoid circular sentences:

- Bad: `同一个 API 名称不等于固定的底层行为。源码分析时需要同时关注...`
- Better: `同一个 API 在不同输入布局下可能对应不同执行路径。检查 shape、dtype、device、stride、CPU 回读和动态 shape。`

## Technical Claim Rules

Every important claim should answer three questions:

- What API or source line triggers the behavior?
- What tensor or runtime state changes?
- What performance or correctness effect follows?

Examples:

- `reshape` may return a view or copy data. Check stride compatibility.
- `contiguous()` copies only when the input is non-contiguous.
- DEVICE tensor `.item()` waits for device execution and returns a CPU scalar.
- `unique / nonzero / masked_select` produce data-dependent output length and can break fixed-shape graph replay.
- `cat / stack / pad` allocate new tensors unless handled by a fused backend or preallocated buffer.

## Code Block Rules

Code comments must explain execution intent and data movement:

```python
# 将 [B, S, D] 拉平成 token 维，便于按 token 路由到 expert。
flat = hidden_states.view(-1, hidden_dim)

# 每个 token 选择 top-k 个 expert。
indices = torch.topk(scores, top_k, dim=-1).indices

# 从 router score 中取出被选中 expert 的权重。
weights = scores.gather(1, indices)
```

Do not add comments that restate syntax:

```python
# call function
out = fn(x)
```

## Example Rules

Use small examples with fixed numbers and expected output.

Good examples:

- `transpose -> contiguous`: show shape, stride, and contiguity.
- RMSNorm: show variance and normalized output.
- Attention: show `q @ k.T`, softmax probability, and weighted value output.
- MoE top-k: show selected expert ids and normalized weights.
- HCA compression: show token window, gate softmax, weighted sum, and compressed entry.
- mHC collapse: show residual streams, `pre` weights, and collapsed hidden.

Each example should include:

- input tensor values
- operation sequence
- output values
- short mapping back to source code

## Source Analysis Rules

For model source analysis, describe the execution chain before discussing performance:

```text
input_ids
  -> embedding
  -> position_ids / causal_mask / rotary embedding
  -> decoder layers
  -> final norm
  -> lm_head
  -> logits
```

For each module, include:

- source file and function name
- key code path
- tensor shape transition
- main OPs
- minimal example or concrete calculation
- performance check points

## Performance Review Rules

Map profiler symptoms to source-level causes.

| Symptom | Source-level checks |
|---|---|
| Extra copy | `reshape / transpose / permute / contiguous / cat / stack / pad / to(device)` |
| Sync wait | `.item() / .tolist() / .cpu() / synchronize / unique / nonzero / masked_select` |
| Allocator jitter | `empty / zeros / new_full / cat / dynamic output OPs` |
| Too many kernels | reference RMSNorm, SwiGLU, attention, MoE routing, per-expert loops |
| Graph replay issue | dynamic shape, address change, allocator inside capture, CPU sync, unsupported backend |

Do not state "performance may be affected" alone. Name the mechanism:

- layout conversion causes D2D copy
- CPU scalar read causes stream synchronization
- dynamic shape requires output-length handling
- per-expert Python loop increases launch count and dispatch overhead
- `cat` creates a new tensor and increases memory bandwidth pressure

## Final Validation

Run text scans for banned or weak wording:

```bash
rg -n "这样|看清|看懂|看出来|啥|怎么|小白|绕口|不专业|文档目标|OP视角|真实 copy|这就是|不适合|拼到|要跟|做三件事" <file>
```

Check Markdown fences:

```bash
python3 - <<'PY'
from pathlib import Path
p = Path("<file>")
text = p.read_text(encoding="utf-8")
print("lines", len(text.splitlines()))
print("fences", text.count("```"), "balanced", text.count("```") % 2 == 0)
PY
```

The final response should state:

- file path changed
- what review standards were applied
- whether Markdown fences are balanced
