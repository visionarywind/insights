# Relation Schema

## 1. 目标

Relation schema 用于重建 Runtime API -> Driver API -> command -> submission -> kernel activity，以及 sync/memory 的等待和对象关系。

## 2. 通用字段

| 字段 | 含义 |
| --- | --- |
| `src_type` | runtime_api / driver_api / command / submission / activity / allocation / stream |
| `src_id` | 源对象 ID |
| `dst_type` | 目标对象类型 |
| `dst_id` | 目标对象 ID |
| `relation_type` | call / create / submit / execute / wait / allocate / free |
| `correlation_id` | MUPTI correlation id |
| `timestamp_ns` | 关系产生时间 |
| `confidence` | 0-1，关系置信度 |
| `source` | runtime / driver / mupti / msight-compute / msight-system |

## 3. M1 必须关系

| 关系 | 召回要求 |
| --- | --- |
| Runtime API -> Driver API | 100%，缺失必须解释 |
| Driver API -> command | launch 路径 >= 95% |
| command -> submission | launch 路径 >= 95% |
| submission -> kernel activity | launch 路径 >= 95% |
| stream sync -> waited command/activity | 能说明等待对象和等待时间 |
| alloc/free -> allocation object | 能关联 ptr、size、pool 行为 |

## 4. 输出

M1 输出 `relations.csv` 或 `relations.parquet`。字段必须支持 join `api_events`、`model_events`、`activity_events`。
