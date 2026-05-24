# 当前文章用例与远端 MUSA 实测输出核对

- Total: 81
- MATCH: 80
- MISMATCH: 1
- MISSING: 0

| ID | Status | Line | Heading | Remote stdout |
|----|--------|------|---------|---------------|
| 001 | MATCH | 629 | 5.1 主要 OP 序列的 MUSA 最小用例 | remote_rerun_results/001.stdout.txt |
| 002 | MATCH | 782 | 5.2 Prefill Metadata 构造最小用例 | remote_results/002.stdout.txt |
| 003 | MATCH | 838 | 5.3 Decode Graph Replay 最小用例 | remote_results/003.stdout.txt |
| 004 | MATCH | 898 | 5.4 KV Cache 写入最小用例 | remote_results/004.stdout.txt |
| 005 | MATCH | 945 | 5.5 MoE 路由与 Combine 最小用例 | remote_results/005.stdout.txt |
| 006 | MATCH | 988 | 5.6 Sampling 后处理最小用例 | remote_results/006.stdout.txt |
| 007 | MATCH | 1185 | 普通 Graph 使用样例 | remote_results/007.stdout.txt |
| 008 | MATCH | 1238 | Piecewise Graph 使用样例 | remote_results/008.stdout.txt |
| 009 | MATCH | 1321 | torch.compile 使用样例 | remote_results/009.stdout.txt |
| 010 | MATCH | 1439 | `torch.empty` | remote_results/010.stdout.txt |
| 011 | MATCH | 1477 | `Tensor.new_empty` | remote_results/011.stdout.txt |
| 012 | MATCH | 1514 | `torch.empty_like` | remote_results/012.stdout.txt |
| 013 | MATCH | 1551 | `torch.zeros` | remote_results/013.stdout.txt |
| 014 | MATCH | 1585 | `torch.ones` | remote_results/014.stdout.txt |
| 015 | MATCH | 1619 | `torch.full` | remote_results/015.stdout.txt |
| 016 | MATCH | 1663 | `Tensor.copy_` | remote_results/016.stdout.txt |
| 017 | MATCH | 1701 | `torch._foreach_copy_` | remote_results/017.stdout.txt |
| 018 | MATCH | 1745 | `Tensor.fill_` | remote_results/018.stdout.txt |
| 019 | MATCH | 1780 | `Tensor.zero_` | remote_results/019.stdout.txt |
| 020 | MATCH | 1815 | `Tensor.masked_fill_` | remote_results/020.stdout.txt |
| 021 | MATCH | 1867 | `Tensor.view` | remote_results/021.stdout.txt |
| 022 | MATCH | 1904 | `Tensor.reshape` | remote_results/022.stdout.txt |
| 023 | MATCH | 1943 | `Tensor.flatten` | remote_results/023.stdout.txt |
| 024 | MATCH | 1980 | `Tensor.unsqueeze` | remote_results/024.stdout.txt |
| 025 | MATCH | 2017 | `Tensor.squeeze` | remote_results/025.stdout.txt |
| 026 | MATCH | 2056 | `Tensor.expand` | remote_results/026.stdout.txt |
| 027 | MATCH | 2095 | `Tensor.contiguous` | remote_results/027.stdout.txt |
| 028 | MATCH | 2132 | `Tensor.stride` | remote_results/028.stdout.txt |
| 029 | MATCH | 2169 | `Tensor.is_contiguous` | remote_results/029.stdout.txt |
| 030 | MATCH | 2206 | `Tensor.storage_offset` | remote_results/030.stdout.txt |
| 031 | MATCH | 2253 | Slice | remote_results/031.stdout.txt |
| 032 | MATCH | 2290 | Advanced indexing | remote_results/032.stdout.txt |
| 033 | MATCH | 2333 | `torch.gather` | remote_results/033.stdout.txt |
| 034 | MATCH | 2373 | `torch.take_along_dim` | remote_results/034.stdout.txt |
| 035 | MATCH | 2413 | `torch.index_select` | remote_results/035.stdout.txt |
| 036 | MATCH | 2453 | `Tensor.scatter_` | remote_results/036.stdout.txt |
| 037 | MISMATCH | 2494 | `Tensor.tensor_split` | remote_rerun_results/037.stdout.txt |
| 038 | MATCH | 2540 | `torch.arange` | remote_results/038.stdout.txt |
| 039 | MATCH | 2574 | `torch.arange(..., out=out)` | remote_results/039.stdout.txt |
| 040 | MATCH | 2609 | `repeat_interleave` | remote_results/040.stdout.txt |
| 041 | MATCH | 2652 | `torch.cat` | remote_results/041.stdout.txt |
| 042 | MATCH | 2692 | `torch.stack` | remote_results/042.stdout.txt |
| 043 | MATCH | 2732 | `torch.nn.functional.pad` | remote_results/043.stdout.txt |
| 044 | MATCH | 2771 | `torch.where` | remote_results/044.stdout.txt |
| 045 | MATCH | 2816 | `sum` | remote_results/045.stdout.txt |
| 046 | MATCH | 2853 | `mean` | remote_results/046.stdout.txt |
| 047 | MATCH | 2890 | `amax` | remote_results/047.stdout.txt |
| 048 | MATCH | 2927 | `min` / `max` | remote_results/048.stdout.txt |
| 049 | MATCH | 2967 | `abs` | remote_results/049.stdout.txt |
| 050 | MATCH | 3001 | `square` | remote_results/050.stdout.txt |
| 051 | MATCH | 3035 | `rsqrt` | remote_results/051.stdout.txt |
| 052 | MATCH | 3069 | `sigmoid` | remote_results/052.stdout.txt |
| 053 | MATCH | 3103 | `silu` | remote_results/053.stdout.txt |
| 054 | MATCH | 3137 | `gelu` | remote_results/054.stdout.txt |
| 055 | MATCH | 3171 | `relu` | remote_results/055.stdout.txt |
| 056 | MATCH | 3205 | `softmax` | remote_results/056.stdout.txt |
| 057 | MATCH | 3239 | `clamp` | remote_results/057.stdout.txt |
| 058 | MATCH | 3283 | `torch.nn.functional.linear` | remote_results/058.stdout.txt |
| 059 | MATCH | 3326 | `matmul` | remote_results/059.stdout.txt |
| 060 | MATCH | 3366 | `mm` | remote_results/060.stdout.txt |
| 061 | MATCH | 3406 | `bmm` | remote_results/061.stdout.txt |
| 062 | MATCH | 3446 | `einsum` | remote_results/062.stdout.txt |
| 063 | MATCH | 3486 | `topk` | remote_results/063.stdout.txt |
| 064 | MATCH | 3525 | `sort` | remote_results/064.stdout.txt |
| 065 | MATCH | 3561 | `argsort` | remote_results/065.stdout.txt |
| 066 | MATCH | 3595 | `argmax` | remote_results/066.stdout.txt |
| 067 | MATCH | 3637 | `Tensor.to(dtype)` | remote_results/067.stdout.txt |
| 068 | MATCH | 3674 | `Tensor.to(device)` | remote_results/068.stdout.txt |
| 069 | MATCH | 3711 | `Tensor.float` | remote_results/069.stdout.txt |
| 070 | MATCH | 3748 | `Tensor.bfloat16` | remote_results/070.stdout.txt |
| 071 | MATCH | 3813 | `Tensor.cpu` | remote_results/071.stdout.txt |
| 072 | MATCH | 3852 | `Tensor.numpy` | remote_results/072.stdout.txt |
| 073 | MATCH | 3889 | `Tensor.item` | remote_results/073.stdout.txt |
| 074 | MATCH | 3928 | `Tensor.tolist` | remote_results/074.stdout.txt |
| 075 | MATCH | 3966 | A.2.2 CPU OP MUSA 环境用例 | remote_results/075.stdout.txt |
| 076 | MATCH | 4066 | `torch.musa.synchronize` | remote_results/076.stdout.txt |
| 077 | MATCH | 4100 | A.3.2 Sync OP MUSA 环境用例 | remote_results/077.stdout.txt |
| 078 | MATCH | 4182 | `torch.nonzero` | remote_results/078.stdout.txt |
| 079 | MATCH | 4219 | `torch.unique` | remote_results/079.stdout.txt |
| 080 | MATCH | 4255 | `torch.masked_select` | remote_results/080.stdout.txt |
| 081 | MATCH | 4361 | `MUSAGraph.replay` | remote_results/081.stdout.txt |
