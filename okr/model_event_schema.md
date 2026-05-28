# ModelEvent Schema

## 1. 目标

ModelEvent 用于补齐 MUPTI activity 无法表达的 Runtime、Driver、Core、HAL/M3D 内部软件阶段成本，使 OKR 能输出 API 成本分项和源码归因。

M1 先使用 private hook，不承诺 public ABI。

## 2. Event header

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `version` | u16 | schema 版本，M1 为 1 |
| `event_type` | enum | `span_begin` / `span_end` / `instant` / `counter` / `relation` |
| `domain` | enum | Runtime、Driver、Memory、Stream、Command、Submit、Sync、HAL、M3D、Relation |
| `event_id` | u32 | 事件 ID |
| `timestamp_ns` | u64 | 单调时间戳 |
| `pid` | u32 | process id |
| `tid` | u32 | thread id |
| `correlation_id` | u64 | Runtime/Driver/MUPTI correlation |
| `span_id` | u64 | span ID |
| `parent_id` | u64 | 父 span ID |
| `context_id` | u64 | context handle/id |
| `stream_id` | u64 | stream handle/id |
| `command_id` | u64 | command object id |
| `submission_id` | u64 | submit/kick id |
| `activity_id` | u64 | MUPTI activity id |
| `status` | s32 | API 或内部阶段状态 |
| `payload_size` | u32 | payload 字节数 |
| `payload` | bytes | event_id 对应的固定 payload |

## 3. M1 domains

| Domain | 用途 |
| --- | --- |
| `MODEL_RUNTIME` | Runtime wrapper、Runtime->Driver relation、module/function cache |
| `MODEL_DRIVER` | Driver API wrapper、handle/context lookup |
| `MODEL_MEMORY` | allocation/free、pool hit/miss/grow/trim |
| `MODEL_STREAM` | stream lookup、queue、wait |
| `MODEL_COMMAND` | command create/build/merge/wait |
| `MODEL_SUBMIT` | submit to queue、HAL/M3D submit、DRM ioctl 边界 |
| `MODEL_SYNC` | sync wait object、wait reason、engine error query |
| `MODEL_RELATION` | API、command、submission、activity 关系 |

## 4. M1 event ids

| Event | Type | Required payload |
| --- | --- | --- |
| `RUNTIME_API_SPAN` | span | api_id、api_name、status |
| `RUNTIME_DRIVER_RELATION` | relation | runtime_correlation_id、driver_correlation_id |
| `DRIVER_API_SPAN` | span | cbid、api_name、status |
| `COMMAND_CREATE` | instant | command_id、command_type、stream_id |
| `DEPENDENCY_RESOLVE_SPAN` | span | dependency_count、wait_count |
| `STREAM_QUEUE_COMMAND_SPAN` | span | queue_depth、async_count |
| `COMMAND_BUILD_SPAN` | span | command_type、dependency_count、status |
| `COMMAND_MERGE_DECISION` | instant/counter | merge_enabled、merge_size、reason |
| `COMMAND_SUBMIT_SPAN` | span | command_id、engine、wait_count、signal_count |
| `SUBMISSION_RELATION` | relation | command_id、submission_id |
| `KERNEL_ACTIVITY_RELATION` | relation | submission_id、activity_id、correlation_id |
| `MEM_POOL_ALLOC_SPAN` | span | ptr、size、pool_id、hit_or_miss |
| `MEM_POOL_GROW_SPAN` | span | requested_size、grow_size、status |
| `MEM_POOL_FREE_SPAN` | span | ptr、size、pool_id、merge |
| `STREAM_WAIT_FINISH_SPAN` | span | stream_id、waited_command_id、wait_reason |
| `COMMAND_WAIT_SPAN` | span | command_id、activity_id、wait_status |
| `SYNC_WAIT_REASON` | instant | reason、waited_object_type、waited_object_id |

## 5. 质量要求

- trace off：只允许 ready flag 分支，开销 <= 0.1% 或在 benchmark 噪声内。
- API-only：开销 <= 1%。
- internal targeted：开销 <= 3%。
- 所有 collector 输出必须包含 dropped record、buffer overflow、flush error。
