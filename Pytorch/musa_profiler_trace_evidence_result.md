# MUSA Profiler/API Trace 证据

验证脚本：`Pytorch/musa_profiler_trace_evidence.py`

验证时间：2026-05-22 15:20 左右

验证环境：

- 验证机器：MUSA 服务器
- Python 环境：SGLang 验证环境
- 设备：`musa:0`
- MUSA 设备名：`MTT S5000`
- PyTorch 版本：`2.9.0`
- Profiler activity：`CPU`、`MUSA`（当前环境输出名显示为 `PrivateUse1`）

## 验证目标

本次 profiler 用例覆盖四类行为：

1. `transpose -> contiguous -> empty_like -> copy_ -> square -> sum`，观察 layout 修正、copy、kernel launch 和 MUSA memory 使用。
2. `.item()`，观察 GPU scalar tensor 回读相关的 CPU-DEVICE 边界。
3. `nonzero + where`，观察 dynamic shape OP 和固定 shape mask 的执行差异。
4. `copy_ + MUSAGraph.replay()`，观察 graph replay 是否落到 `musaGraphLaunch`。

## 关键输出

```text
ENV
{
  "torch_version": "2.9.0",
  "device": "musa:0",
  "device_name": "MTT S5000",
  "activities": [
    "CPU",
    "PrivateUse1"
  ],
  "result": {
    "scalar": 6004765322379264.0,
    "nonzero_shape": [
      3,
      1
    ],
    "fixed_mask_shape": [
      6
    ]
  },
  "graph_weight_shape": [
    3,
    2
  ],
  "graph_out": [
    [
      13.999988555908203,
      6.993622779846191
    ],
    [
      32.0,
      15.999998092651367
    ]
  ]
}
```

## Profiler 摘要

| 事件 | 调用次数 | CPU/MUSA 观测 | 说明 |
|------|----------|---------------|------|
| `musaLaunchKernel` | 13 | Self CPU `117.145us` | 未融合的逐元素/归约 OP 序列最终会形成多次 kernel launch |
| `musaMemcpyAsync` | 4 | Self CPU `128.701us` | `copy_`、tensor 构造或边界转换会落到异步 copy API |
| `musaDeviceSynchronize` | 3 | Self CPU `156.589us` | 用例中的显式 `torch.musa.synchronize()` 形成全设备等待 |
| `musaStreamSynchronize` | 3 | Self CPU `71.209us` | profiler/边界同步中能看到 stream 等待 |
| `musaGraphLaunch` | 1 | Self CPU `8.327us` | `MUSAGraph.replay()` 落到 graph launch |
| `aten::copy_` | 5 | CPU total `235.619us`，MUSA total `16.257us` | replay buffer 更新、D2D/H2D copy 都需要继续看底层 copy 类型 |
| `aten::contiguous` | 1 | CPU total `150.922us`，MUSA total `7.720us` | 非 contiguous layout 修正触发 CopyTranspose kernel |
| `aten::item` | 1 | CPU total `113.432us`，MUSA total `2.880us` | GPU scalar tensor 回读形成 CPU-DEVICE 边界 |
| `Memcpy DtoH (Device -> Pageable)` | 1 | MUSA total `3.128us` | profiler 中能直接看到 D2H copy 事件 |
| `aten::nonzero` | 1 | CPU total `128.734us`，MUSA total `13.888us` | dynamic shape OP 伴随 scan/nonzero 相关 kernel |

关键 GPU kernel 摘录：

```text
void musa::dnn::(anonymous namespace)::CopyTransposeKernelAlign...
  Self MUSA 7.720us, Count 1

void musa::dnn::ScanPartSumKernel...
  Self MUSA 5.920us, Count 1

void musa::dnn::(anonymous namespace)::AddBaseSumFusedNonzero...
  Self MUSA 4.840us, Count 1
```

## 结论

这段 trace 说明，文章中讨论的 PyTorch OP 可以在 profiler 中对应到 runtime/API 行为：

- `contiguous()` 不是简单的 Python 层 layout 标记；当输入非连续时，会出现 `CopyTransposeKernel`。
- `copy_` 会在 profiler 中体现为 `aten::copy_`、`aten::_copy_from` 和 `musaMemcpyAsync` 等事件。
- `.item()` 会触发 scalar 回读路径，trace 中能看到 `aten::item`、`aten::_local_scalar_dense` 和 D2H 相关事件。
- `MUSAGraph.replay()` 可以对应到 `musaGraphLaunch`。
- `nonzero` 这类 dynamic shape OP 会出现 scan/nonzero 相关 kernel，不适合作为 fixed-shape graph replay 内部的变长输出来源。

## 注意事项

本次 profiler 结果用于说明 PyTorch OP 会对应到哪些 runtime API、kernel、copy 和同步事件。例如 `contiguous()` 可以对应到 `CopyTransposeKernel`，`MUSAGraph.replay()` 可以对应到 `musaGraphLaunch`。这些事件和调用次数是单次 trace 中观察到的链路证据，不用于推导稳定耗时、吞吐或性能优劣。

本次 profiler 输出包含提示：`Some activities may have incomplete timestamps because not all HW events are returned from mt-perf before time limit!`。若要形成性能结论，需要增加 warmup、多轮统计、固定输入规模、关闭额外同步，并结合更底层的 runtime API trace 或硬件计数器。
