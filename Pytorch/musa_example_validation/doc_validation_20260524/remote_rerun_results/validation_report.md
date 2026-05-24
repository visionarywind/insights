# PyTorch OP 文档用例 MUSA 验证结果

- Total: 2
- PASS: 1
- FAIL: 1
- TIMEOUT: 0

| ID | Status | Line | Heading | Elapsed ms |
|----|--------|------|---------|------------|
| 001 | PASS | 716 | 4.1 源码 OP 链抽取出的 MUSA 最小用例 | 17245.245 |
| 037 | FAIL | 2501 | `Tensor.tensor_split` | 4520.121 |

## Failures

### 037 line 2501 `Tensor.tensor_split`

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

