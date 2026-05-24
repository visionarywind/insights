# PyTorch OP 文档用例 MUSA 验证结果

- Total: 81
- PASS: 79
- FAIL: 2
- TIMEOUT: 0

| ID | Status | Line | Heading | Elapsed ms |
|----|--------|------|---------|------------|
| 001 | FAIL | 716 | 4.1 源码 OP 链抽取出的 MUSA 最小用例 | 17463.266 |
| 002 | PASS | 863 | 4.2 Prefill Metadata 构造最小用例 | 17271.649 |
| 003 | PASS | 911 | 4.3 Decode Graph Replay 最小用例 | 7765.463 |
| 004 | PASS | 966 | 4.4 KV Cache 写入最小用例 | 7695.435 |
| 005 | PASS | 1007 | 4.5 MoE 路由与 Combine 最小用例 | 15557.796 |
| 006 | PASS | 1045 | 4.6 Sampling 后处理最小用例 | 10111.713 |
| 007 | PASS | 1187 | 普通 Graph 使用样例 | 7608.979 |
| 008 | PASS | 1232 | Piecewise Graph 使用样例 | 7678.272 |
| 009 | PASS | 1304 | torch.compile 使用样例 | 5323.796 |
| 010 | PASS | 1458 | `torch.empty` | 4525.215 |
| 011 | PASS | 1494 | `Tensor.new_empty` | 4337.479 |
| 012 | PASS | 1531 | `torch.empty_like` | 4475.693 |
| 013 | PASS | 1568 | `torch.zeros` | 6226.387 |
| 014 | PASS | 1602 | `torch.ones` | 6256.728 |
| 015 | PASS | 1636 | `torch.full` | 6213.577 |
| 016 | PASS | 1678 | `Tensor.copy_` | 5931.512 |
| 017 | PASS | 1716 | `torch._foreach_copy_` | 4515.609 |
| 018 | PASS | 1760 | `Tensor.fill_` | 6069.382 |
| 019 | PASS | 1795 | `Tensor.zero_` | 6127.662 |
| 020 | PASS | 1830 | `Tensor.masked_fill_` | 6097.437 |
| 021 | PASS | 1876 | `Tensor.view` | 4501.592 |
| 022 | PASS | 1913 | `Tensor.reshape` | 4496.372 |
| 023 | PASS | 1952 | `Tensor.flatten` | 4314.624 |
| 024 | PASS | 1989 | `Tensor.unsqueeze` | 4473.625 |
| 025 | PASS | 2026 | `Tensor.squeeze` | 4307.695 |
| 026 | PASS | 2065 | `Tensor.expand` | 6141.598 |
| 027 | PASS | 2104 | `Tensor.contiguous` | 6137.462 |
| 028 | PASS | 2141 | `Tensor.stride` | 4474.159 |
| 029 | PASS | 2178 | `Tensor.is_contiguous` | 4500.93 |
| 030 | PASS | 2215 | `Tensor.storage_offset` | 4355.647 |
| 031 | PASS | 2258 | Slice | 6136.088 |
| 032 | PASS | 2295 | Advanced indexing | 4566.646 |
| 033 | PASS | 2338 | `torch.gather` | 4582.668 |
| 034 | PASS | 2378 | `torch.take_along_dim` | 4631.119 |
| 035 | PASS | 2418 | `torch.index_select` | 4484.827 |
| 036 | PASS | 2458 | `Tensor.scatter_` | 8616.76 |
| 037 | FAIL | 2499 | `Tensor.tensor_split` | 4594.024 |
| 038 | PASS | 2539 | `torch.arange` | 4567.706 |
| 039 | PASS | 2573 | `torch.arange(..., out=out)` | 4508.578 |
| 040 | PASS | 2608 | `repeat_interleave` | 14665.039 |
| 041 | PASS | 2651 | `torch.cat` | 6019.834 |
| 042 | PASS | 2691 | `torch.stack` | 6180.36 |
| 043 | PASS | 2731 | `torch.nn.functional.pad` | 7566.778 |
| 044 | PASS | 2770 | `torch.where` | 6142.977 |
| 045 | PASS | 2813 | `sum` | 9107.236 |
| 046 | PASS | 2850 | `mean` | 9232.003 |
| 047 | PASS | 2887 | `amax` | 9237.702 |
| 048 | PASS | 2924 | `min` / `max` | 6145.843 |
| 049 | PASS | 2964 | `abs` | 5754.91 |
| 050 | PASS | 2998 | `square` | 5903.8 |
| 051 | PASS | 3032 | `rsqrt` | 5897.774 |
| 052 | PASS | 3066 | `sigmoid` | 5965.105 |
| 053 | PASS | 3100 | `silu` | 5937.025 |
| 054 | PASS | 3134 | `gelu` | 5939.847 |
| 055 | PASS | 3168 | `relu` | 5895.474 |
| 056 | PASS | 3202 | `softmax` | 6924.464 |
| 057 | PASS | 3236 | `clamp` | 5902.955 |
| 058 | PASS | 3276 | `torch.nn.functional.linear` | 4519.537 |
| 059 | PASS | 3319 | `matmul` | 4514.555 |
| 060 | PASS | 3359 | `mm` | 4352.347 |
| 061 | PASS | 3399 | `bmm` | 6095.635 |
| 062 | PASS | 3439 | `einsum` | 6147.657 |
| 063 | PASS | 3479 | `topk` | 6958.158 |
| 064 | PASS | 3518 | `sort` | 6995.249 |
| 065 | PASS | 3554 | `argsort` | 6979.315 |
| 066 | PASS | 3588 | `argmax` | 6108.961 |
| 067 | PASS | 3628 | `Tensor.to(dtype)` | 5938.791 |
| 068 | PASS | 3665 | `Tensor.to(device)` | 4486.646 |
| 069 | PASS | 3702 | `Tensor.float` | 5730.487 |
| 070 | PASS | 3739 | `Tensor.bfloat16` | 5875.976 |
| 071 | PASS | 3796 | `Tensor.cpu` | 4508.096 |
| 072 | PASS | 3835 | `Tensor.numpy` | 4417.68 |
| 073 | PASS | 3872 | `Tensor.item` | 4330.072 |
| 074 | PASS | 3911 | `Tensor.tolist` | 4520.798 |
| 075 | PASS | 3949 | A.2.2 CPU OP MUSA 环境用例 | 4571.084 |
| 076 | PASS | 4045 | `torch.musa.synchronize` | 5770.158 |
| 077 | PASS | 4079 | A.3.2 Sync OP MUSA 环境用例 | 7828.473 |
| 078 | PASS | 4155 | `torch.nonzero` | 8596.903 |
| 079 | PASS | 4191 | `torch.unique` | 7163.564 |
| 080 | PASS | 4226 | `torch.masked_select` | 10164.825 |
| 081 | PASS | 4321 | `MUSAGraph.replay` | 7786.687 |

## Failures

### 001 line 716 4.1 源码 OP 链抽取出的 MUSA 最小用例

Expected:
```text
input_ids shape=(4,), dtype=int64, device=musa:0, value=[11, 12, 13, 0]
captured.req_pool_indices shape=(2,), dtype=int64, device=musa:0, value=[7, 8]
captured.seq_lens shape=(2,), dtype=int32, device=musa:0, value=[5, 6]
captured.out_cache_loc shape=(4,), dtype=int64, device=musa:0, value=[100, 101, 102, 0]
graph_out shape=(2, 2), dtype=float32, device=musa:0, value=[[9.0, 9.0], [9.0, 9.0]]
hc_y shape=(2, 3), dtype=float32, device=musa:0, value=[[2.9752485752105713, 4.827154636383057, 6.679060459136963], [14.002138137817383, 15.838565826416016, 17.674991607666016]]
post shape=(2, 2), dtype=float32, device=musa:0, value=[[0.9995744824409485, 0.9999781847000122], [0.999838650226593, 0.999995231628418]]
comb shape=(2, 2, 2), dtype=float32, device=musa:0, value=[[[0.002611535834148526, 0.0026115409564226866], [0.9973884224891663, 0.9973884224891663]], [[0.0008589610224589705, 0.0008589592762291431], [0.9991409778594971, 0.9991409778594971]]]
hc_post shape=(2, 2, 3), dtype=float32, device=musa:0, value=[[[5.9661478996276855, 8.817265510559082, 11.668384552001953], [5.967349052429199, 8.819214820861816, 11.671079635620117]], [[22.99730110168457, 25.833431243896484, 28.66956329345703], [22.999492645263672, 25.835912704467773, 28.67232894897461]]]
swiglu_out shape=(1, 2), dtype=float32, device=musa:0, value=[[7.310585975646973, 2.3840584754943848]]
```
Actual stdout:
```text
input_ids shape=(4,), dtype=int64, device=musa:0, value=[11, 12, 13, 0]
captured.req_pool_indices shape=(2,), dtype=int64, device=musa:0, value=[7, 8]
captured.seq_lens shape=(2,), dtype=int32, device=musa:0, value=[5, 6]
captured.out_cache_loc shape=(4,), dtype=int64, device=musa:0, value=[100, 101, 102, 0]
graph_out shape=(2, 2), dtype=float32, device=musa:0, value=[[9.0, 9.0], [9.0, 9.0]]
hc_y shape=(2, 3), dtype=float32, device=musa:0, value=[[2.9752485752105713, 4.827154636383057, 6.679060459136963], [14.002138137817383, 15.838565826416016, 17.674991607666016]]
post shape=(2, 2), dtype=float32, device=musa:0, value=[[0.9995744824409485, 0.9999781847000122], [0.999838650226593, 0.999995231628418]]
comb shape=(2, 2, 2), dtype=float32, device=musa:0, value=[[[0.002611535834148526, 0.0026115409564226866], [0.9973884224891663, 0.9973884224891663]], [[0.0008589610224589705, 0.0008589610224589705], [0.9991409778594971, 0.9991409778594971]]]
hc_post shape=(2, 2, 3), dtype=float32, device=musa:0, value=[[[5.9661478996276855, 8.817265510559082, 11.668384552001953], [5.967349052429199, 8.819214820861816, 11.671079635620117]], [[22.99730110168457, 25.833431243896484, 28.66956329345703], [22.999492645263672, 25.835912704467773, 28.67232894897461]]]
swiglu_out shape=(1, 2), dtype=float32, device=musa:0, value=[[7.310585975646973, 2.3840584754943848]]
```
Stderr:
```text

```

### 037 line 2499 `Tensor.tensor_split`

Expected:
```text
chunks = [Tensor(shape=(2,), dtype=int64, device=musa:0, value=[0, 1]), Tensor(shape=(2,), dtype=int64, device=musa:0, value=[2, 3]), Tensor(shape=(2,), dtype=int64, device=musa:0, value=[4, 5])]
```
Actual stdout:
```text
chunks = (tensor([0, 1], device='musa:0'), tensor([2, 3], device='musa:0'), tensor([4, 5], device='musa:0'))
```
Stderr:
```text

```

