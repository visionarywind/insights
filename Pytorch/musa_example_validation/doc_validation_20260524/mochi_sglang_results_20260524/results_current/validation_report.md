# PyTorch OP 文档用例 MUSA 验证结果

- Total: 81
- PASS: 81
- FAIL: 0
- TIMEOUT: 0

| ID | Status | Line | Heading | Elapsed ms |
|----|--------|------|---------|------------|
| 001 | PASS | 629 | 5.1 主要 OP 序列的 MUSA 最小用例 | 17414.06 |
| 002 | PASS | 782 | 5.2 Prefill Metadata 构造最小用例 | 17428.993 |
| 003 | PASS | 838 | 5.3 Decode Graph Replay 最小用例 | 7732.821 |
| 004 | PASS | 898 | 5.4 KV Cache 写入最小用例 | 7620.062 |
| 005 | PASS | 945 | 5.5 MoE 路由与 Combine 最小用例 | 15526.396 |
| 006 | PASS | 988 | 5.6 Sampling 后处理最小用例 | 10097.039 |
| 007 | PASS | 1185 | 普通 Graph 使用样例 | 7732.15 |
| 008 | PASS | 1238 | Piecewise Graph 使用样例 | 7642.009 |
| 009 | PASS | 1321 | torch.compile 使用样例 | 5520.094 |
| 010 | PASS | 1439 | `torch.empty` | 4553.621 |
| 011 | PASS | 1477 | `Tensor.new_empty` | 4534.456 |
| 012 | PASS | 1514 | `torch.empty_like` | 4510.212 |
| 013 | PASS | 1551 | `torch.zeros` | 6244.655 |
| 014 | PASS | 1585 | `torch.ones` | 6247.6 |
| 015 | PASS | 1619 | `torch.full` | 6114.609 |
| 016 | PASS | 1663 | `Tensor.copy_` | 5923.117 |
| 017 | PASS | 1701 | `torch._foreach_copy_` | 4550.448 |
| 018 | PASS | 1745 | `Tensor.fill_` | 6167.828 |
| 019 | PASS | 1780 | `Tensor.zero_` | 6136.157 |
| 020 | PASS | 1815 | `Tensor.masked_fill_` | 6163.019 |
| 021 | PASS | 1867 | `Tensor.view` | 4523.243 |
| 022 | PASS | 1904 | `Tensor.reshape` | 4474.749 |
| 023 | PASS | 1943 | `Tensor.flatten` | 4515.004 |
| 024 | PASS | 1980 | `Tensor.unsqueeze` | 4468.549 |
| 025 | PASS | 2017 | `Tensor.squeeze` | 4478.07 |
| 026 | PASS | 2056 | `Tensor.expand` | 6110.41 |
| 027 | PASS | 2095 | `Tensor.contiguous` | 6133.255 |
| 028 | PASS | 2132 | `Tensor.stride` | 4500.147 |
| 029 | PASS | 2169 | `Tensor.is_contiguous` | 4455.632 |
| 030 | PASS | 2206 | `Tensor.storage_offset` | 4502.5 |
| 031 | PASS | 2253 | Slice | 6187.563 |
| 032 | PASS | 2290 | Advanced indexing | 4525.321 |
| 033 | PASS | 2333 | `torch.gather` | 4434.899 |
| 034 | PASS | 2373 | `torch.take_along_dim` | 4417.837 |
| 035 | PASS | 2413 | `torch.index_select` | 4477.478 |
| 036 | PASS | 2453 | `Tensor.scatter_` | 8620.493 |
| 037 | PASS | 2494 | `Tensor.tensor_split` | 4475.034 |
| 038 | PASS | 2540 | `torch.arange` | 4541.149 |
| 039 | PASS | 2574 | `torch.arange(..., out=out)` | 4543.717 |
| 040 | PASS | 2609 | `repeat_interleave` | 14957.69 |
| 041 | PASS | 2652 | `torch.cat` | 6161.037 |
| 042 | PASS | 2692 | `torch.stack` | 6141.005 |
| 043 | PASS | 2732 | `torch.nn.functional.pad` | 7562.306 |
| 044 | PASS | 2771 | `torch.where` | 6171.654 |
| 045 | PASS | 2816 | `sum` | 8876.485 |
| 046 | PASS | 2853 | `mean` | 9270.021 |
| 047 | PASS | 2890 | `amax` | 9255.58 |
| 048 | PASS | 2927 | `min` / `max` | 6142.873 |
| 049 | PASS | 2967 | `abs` | 5769.225 |
| 050 | PASS | 3001 | `square` | 5888.639 |
| 051 | PASS | 3035 | `rsqrt` | 5932.936 |
| 052 | PASS | 3069 | `sigmoid` | 5819.363 |
| 053 | PASS | 3103 | `silu` | 5986.769 |
| 054 | PASS | 3137 | `gelu` | 5937.72 |
| 055 | PASS | 3171 | `relu` | 5943.219 |
| 056 | PASS | 3205 | `softmax` | 6939.513 |
| 057 | PASS | 3239 | `clamp` | 5942.318 |
| 058 | PASS | 3283 | `torch.nn.functional.linear` | 4478.0 |
| 059 | PASS | 3326 | `matmul` | 4488.732 |
| 060 | PASS | 3366 | `mm` | 4556.126 |
| 061 | PASS | 3406 | `bmm` | 6161.129 |
| 062 | PASS | 3446 | `einsum` | 6134.299 |
| 063 | PASS | 3486 | `topk` | 6978.831 |
| 064 | PASS | 3525 | `sort` | 6992.182 |
| 065 | PASS | 3561 | `argsort` | 6981.205 |
| 066 | PASS | 3595 | `argmax` | 6152.78 |
| 067 | PASS | 3637 | `Tensor.to(dtype)` | 5879.747 |
| 068 | PASS | 3674 | `Tensor.to(device)` | 4501.227 |
| 069 | PASS | 3711 | `Tensor.float` | 5906.264 |
| 070 | PASS | 3748 | `Tensor.bfloat16` | 5818.91 |
| 071 | PASS | 3813 | `Tensor.cpu` | 4507.323 |
| 072 | PASS | 3852 | `Tensor.numpy` | 4234.032 |
| 073 | PASS | 3889 | `Tensor.item` | 4529.004 |
| 074 | PASS | 3928 | `Tensor.tolist` | 4515.3 |
| 075 | PASS | 3966 | A.2.2 CPU OP MUSA 环境用例 | 4957.985 |
| 076 | PASS | 4066 | `torch.musa.synchronize` | 5756.767 |
| 077 | PASS | 4100 | A.3.2 Sync OP MUSA 环境用例 | 7779.347 |
| 078 | PASS | 4182 | `torch.nonzero` | 8614.386 |
| 079 | PASS | 4219 | `torch.unique` | 6970.246 |
| 080 | PASS | 4255 | `torch.masked_select` | 9962.585 |
| 081 | PASS | 4361 | `MUSAGraph.replay` | 7608.958 |
