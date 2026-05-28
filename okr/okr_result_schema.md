# OKR Result Schema

## 1. 目标

OKR result 用于把白盒建模结果变成可验收、可回归的结构化产物。

## 2. JSON 字段

```json
{
  "okr": "driver_api_whitebox_modeling",
  "sdk_version": "",
  "driver_commit": "",
  "runtime_commit": "",
  "mupti_commit": "",
  "workload": "",
  "device": "",
  "metrics": [
    {
      "name": "relation_recall_launch",
      "target": 0.95,
      "actual": 0.0,
      "pass": false,
      "reason": ""
    }
  ],
  "reports": {
    "event_quality": "event_quality_report.md",
    "relation_recall": "relation_recall_report.md",
    "overhead": "overhead_report.md",
    "cost_breakdown": "api_cost_breakdown.csv"
  }
}
```

## 3. M1 必须指标

| 指标 | 目标 |
| --- | --- |
| `relation_recall_launch` | >= 0.95 |
| `unknown_cost_ratio` | <= 0.15 |
| `trace_off_overhead` | <= 0.001 或噪声内 |
| `api_only_overhead` | <= 0.01 |
| `internal_targeted_overhead` | <= 0.03 |
| `dropped_records` | 默认 buffer 下为 0，压力测试必须报告 |
| `flush_errors` | 0 |
