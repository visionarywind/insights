# MUSA Driver/Runtime 视角 OP 验证结果

验证脚本：`Pytorch/musa_driver_runtime_validation.py`

验证时间：2026-05-22 15:00 左右

验证环境：

- 验证机器：MUSA 服务器
- Python 环境：SGLang 验证环境
- 设备：`musa:0`
- MUSA 设备名：`MTT S5000`
- PyTorch 版本：`2.9.0`

## 验证目标

本次验证面向文章中的 GPU driver/runtime 视角，重点确认四类行为：

1. `view`、`contiguous`、`copy_` 对 storage、stride 和固定 buffer 地址的影响。
2. `MUSAGraph` capture/replay 是否可以在固定 tensor 地址下复用执行路径。
3. `.item()`、`.cpu()` 这类 CPU-DEVICE 边界 API 是否可以作为同步边界用例被观测。
4. `nonzero` 这类 dynamic shape OP 是否会产生随输入内容变化的输出 shape。

## 验证结论

| 用例 | 结论 |
|------|------|
| layout 与 copy | `view` 与原 tensor 共享 storage；`transpose` 生成非 contiguous view；`contiguous()` 生成新 storage；`copy_` 写入固定 buffer 后 data pointer 保持不变 |
| Graph replay | `x` 和 `out` 的 data pointer 在 replay 前后保持稳定；修改 `x` 内容后 replay 输出随之变化，符合固定地址、内容更新的 replay 模式 |
| 同步边界 | `.item()` 与 `.cpu().tolist()` 可以作为必要结果回传场景的观测用例；单次耗时只用于确认 CPU-DEVICE 边界存在，不作为性能结论 |
| Dynamic shape | 两个 mask 的 `nonzero` 输出 shape 分别为 `[3, 1]` 和 `[1, 1]`；`where` 生成的 fixed mask shape 均为 `[6]`，更适合固定 buffer 和 graph replay |

## 原始输出

```text
CASE environment
{
  "torch_version": "2.9.0",
  "device": "musa:0",
  "device_count": 1,
  "current_device": 0,
  "device_name": "MTT S5000"
}
CASE layout_and_copy
{
  "base": {
    "shape": [
      3,
      4
    ],
    "dtype": "float32",
    "device": "musa:0",
    "stride": [
      4,
      1
    ],
    "is_contiguous": true,
    "data_ptr": 1099723440128,
    "value": [
      [
        0.0,
        1.0,
        2.0,
        3.0
      ],
      [
        4.0,
        5.0,
        6.0,
        7.0
      ],
      [
        8.0,
        9.0,
        10.0,
        11.0
      ]
    ]
  },
  "viewed": {
    "shape": [
      2,
      6
    ],
    "stride": [
      6,
      1
    ],
    "shares_storage_with_base": true
  },
  "transposed": {
    "shape": [
      4,
      3
    ],
    "stride": [
      1,
      4
    ],
    "is_contiguous": false
  },
  "contiguous": {
    "shape": [
      4,
      3
    ],
    "stride": [
      3,
      1
    ],
    "is_contiguous": true,
    "new_storage_from_base": true
  },
  "copy_": {
    "fixed_ptr_before": 1099723441152,
    "fixed_ptr_after": 1099723441152,
    "fixed_ptr_stable": true,
    "fixed_value": [
      [
        0.0,
        1.0,
        2.0,
        3.0
      ],
      [
        4.0,
        5.0,
        6.0,
        7.0
      ],
      [
        8.0,
        9.0,
        10.0,
        11.0
      ]
    ]
  }
}
CASE graph_replay_fixed_address
{
  "x_ptr_stable": true,
  "out_ptr_stable": true,
  "first_replay_out": [
    [
      13.999988555908203,
      6.993622779846191
    ],
    [
      32.0,
      15.999998092651367
    ]
  ],
  "second_replay_out": [
    [
      3.9280550479888916,
      1.7615940570831299
    ],
    [
      7.997317314147949,
      3.9280550479888916
    ]
  ]
}
CASE sync_boundaries
{
  "item_value": 140.0,
  "item_elapsed_ms_single_run": 1.046292,
  "cpu_list": [
    0.0,
    1.0,
    2.0,
    3.0
  ],
  "cpu_elapsed_ms_single_run": 0.186093,
  "note": "single-run timing is only a sync-boundary smoke test, not a benchmark"
}
CASE dynamic_shape
{
  "nonzero_shape_a": [
    3,
    1
  ],
  "nonzero_shape_b": [
    1,
    1
  ],
  "nonzero_value_a": [
    [
      0
    ],
    [
      2
    ],
    [
      5
    ]
  ],
  "nonzero_value_b": [
    [
      1
    ]
  ],
  "fixed_mask_shape_a": [
    6
  ],
  "fixed_mask_shape_b": [
    6
  ],
  "fixed_mask_value_a": [
    1,
    0,
    1,
    0,
    0,
    1
  ],
  "fixed_mask_value_b": [
    0,
    1,
    0,
    0,
    0,
    0
  ]
}
```

## 注意事项

本次验证是功能和边界检查。`.item()` 和 `.cpu()` 的单次耗时受环境、队列状态和同步位置影响，只用于说明这类 API 会形成 CPU-DEVICE 边界，不用于比较性能优劣。若需要形成性能结论，应补充多轮统计、固定输入规模、runtime/API trace、kernel launch 统计、H2D/D2H/D2D copy 统计和 Graph capture 前后对比。
