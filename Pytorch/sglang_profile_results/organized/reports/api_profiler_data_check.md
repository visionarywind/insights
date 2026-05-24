# API Profiler Data Check

## Finding

The current sparse-looking API profiler result comes from the summary file, not from the timeline capture.

The checked SGLang API profiler timeline contains:

- `72818` timeline events
- `63543` driver API events
- `9275` kernel events

Files:

- `organized/api_profiler_timeline_check/gpu_activity_timeline.tsv`
- `organized/api_profiler_timeline_check/sglang_api_timeline_us.tsv`
- `organized/api_profiler_timeline_check/sglang_api_summary_us.tsv`

## Why `gpu_activity_profile.txt` Looked Small

The profile summary file in the SGLang timeline check was only `589` bytes after sync. It contained an empty table even though the run printed a non-empty summary during execution.

This indicates a multi-process overwrite issue: a later process with no collected activity can run the profiler `atexit` handler and rewrite `MUPTI_ACTIVITY_PROFILE_OUTPUT` with an empty report.

The timeline file did not lose data because the profiler already skips timeline dump when no events are present.

## Fix Applied

`musa_mupti_activity_profiler.cpp` now skips the summary report entirely when all collected maps are empty:

- `g_apiStats`
- `g_launchStats`
- `g_kernelStats`
- `g_timelineEvents`

This prevents empty processes from overwriting a populated summary file.

## Remaining Difference

The new timeline run currently contains driver API and kernel activity, but not runtime API rows such as `musaLaunchKernel_v7000`.

Observed SGLang timeline distribution:

```text
api/driver              63543
kernel/concurrent_kernel 9275
```

So the timeline is not small, but its API side is driver-domain only in this run. The older console-derived summary still contains runtime API rows and is preserved in `organized/derived/api_activity_summary_us.tsv`.

For detailed time-order analysis, use the timeline TSV rather than `gpu_activity_profile.txt`.
