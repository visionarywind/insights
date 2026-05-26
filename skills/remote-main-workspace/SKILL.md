---
name: remote-main-workspace
description: Use this skill when MUSA, MUPTI, MUSA-Runtime, linux-ddk/musa, or insights work should treat shanfeng@10.18.32.25:/home/shanfeng/workspace as the authoritative workspace. It defines remote-first reading, editing, patching, build, validation, and documentation rules so local/remote synchronization is avoided.
---

# Remote Main Workspace

## Scope

Use this skill when the user wants remote repositories to be the main workspace for MUSA-related work:

- Inspecting or editing `MUSA-Runtime`, `linux-ddk/musa`, `MUPTI`, or `insights`.
- Adding logs, instrumentation, technical notes, or validation documents.
- Building or running checks in `mochi-sglang` or `mochi-build`.
- Avoiding local copies as the source of truth.

Do not treat `/home/mtuser/workspace` as authoritative for these tasks unless the user explicitly asks for a local copy.

## Authoritative Paths

Remote host:

```text
shanfeng@10.18.32.25
```

Remote workspace:

```text
/home/shanfeng/workspace
```

Primary repositories:

```text
/home/shanfeng/workspace/MUSA-Runtime
/home/shanfeng/workspace/linux-ddk/musa
/home/shanfeng/workspace/MUPTI
/home/shanfeng/workspace/insights
```

Container paths:

```text
mochi-sglang: /workspace/workspace
mochi-build:  /root/linux-ddk
```

## Default Rule

For matching tasks, operate remote-first:

1. Read files from the remote workspace with `ssh`.
2. Edit files in the remote workspace with patch application.
3. Build and validate in the remote containers.
4. Write final documentation under remote `/home/shanfeng/workspace/insights`.
5. Only mirror files to local when the user asks for a local copy.

## Command Pattern

Use non-interactive SSH by default:

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 shanfeng@10.18.32.25 "<command>"
```

Run repository commands on the remote host:

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 shanfeng@10.18.32.25 \
  "cd /home/shanfeng/workspace/MUSA-Runtime && git status --short"
```

Run runtime/demo checks in `mochi-sglang`:

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 shanfeng@10.18.32.25 \
  "docker exec -w /workspace/workspace mochi-sglang bash -lc '<command>'"
```

Run DDK builds in `mochi-build`:

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 shanfeng@10.18.32.25 \
  "docker exec mochi-build bash -lc 'cd /root/linux-ddk && <command>'"
```

For detailed MUSA environment checks, use the `musa-env` skill. For DDK build details, use the `musa-build` skill.

## Remote Editing Workflow

Use patch-based editing for remote files.

Recommended workflow:

1. Inspect the remote file with `sed`, `nl`, `rg`, or `git diff`.
2. Create a focused patch locally under `/tmp`.
3. Apply it on the remote repository with `git apply --check`.
4. Apply it with `git apply`.
5. Verify with `git diff --stat`, targeted file reads, build, or runtime checks.

Patch application pattern:

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 shanfeng@10.18.32.25 \
  "cd /home/shanfeng/workspace/<repo> && git apply --check -" < /tmp/<change>.patch

ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 shanfeng@10.18.32.25 \
  "cd /home/shanfeng/workspace/<repo> && git apply -" < /tmp/<change>.patch
```

For files under remote `/home/shanfeng/workspace/insights`, use `/home/shanfeng/workspace` as the patch root.

Do not use local-only edits as final output for remote-workspace tasks.

## Remote Documentation Rule

When the user asks to write analysis, records, technical insights, plans, or skills, write them to remote:

```text
/home/shanfeng/workspace/insights
```

Use local files only as temporary patch sources. The final answer should report the remote path.

## Validation Checklist

Before finalizing a remote-workspace task, collect the relevant proof:

```text
remote git status --short for touched repositories
build command and result
test/demo command and return code
important output lines
remote document path, if documentation was written
paths that were instrumented or modified
known paths not covered by the validation
```

For MUPTI call-flow work, prefer this proof shape:

```text
MUPTI_CALLFLOW_DEBUG=1
MUSART_CALLFLOW_DEBUG=1
MUSA_DRIVER_CALLFLOW_DEBUG=1
demo return code
log path
log line count
functional output
representative Runtime -> Driver -> Core -> Command -> HAL evidence
```

## Hygiene

- Keep `/home/shanfeng/workspace` as the single source of truth.
- Do not overwrite user changes on the remote host.
- Check remote `git status --short` before broad edits.
- Keep patches scoped to the requested component.
- Keep temporary binaries and logs under `/tmp` unless persistence is required.
- Do not commit generated binaries, `.elf` files, profiler traces, or backup documents unless explicitly requested.
- State when a path was instrumented but not exercised by the validation workload.
