---
name: musa-env
description: Use this skill when a task needs a MUSA environment for compilation or validation. It uses the remote sglang Docker environment on shanfeng@10.18.32.25, covering the verified mochi-sglang container, MUSA SDK checks, mcc compilation, MUPTI demo execution, PyTorch/MUSA sanity checks, and common environment failures.
---

# MUSA-env

## Scope

Use this skill when a task needs a working MUSA environment for compilation or validation:

- Run MUSA Runtime, MUPTI, PyTorch/MUSA, or SGLang-related checks.
- Compile standalone `.cu` examples with `mcc`.
- Compile and run `insights/mupti/demo.cu`.
- Verify SDK headers, runtime libraries, MUPTI libraries, and Python packages.

Do not use this skill for `linux-ddk` source builds. Use the `musa-build` skill for DDK builds in `mochi-build`.

## Verified Environment

The user may refer to this as the remote `docker-sglang` environment. The verified Docker container name is:

```text
Remote host: shanfeng@10.18.32.25
Docker container: mochi-sglang
Container workspace: /workspace/workspace
Default working directory: /workspace
MUSA SDK symlink: /usr/local/musa -> musa-4.3.5
MUSA compiler: /usr/local/musa/bin/mcc
Observed mcc version: 5.1.0
Python: /usr/bin/python3
Observed torch version: 2.9.0
torch.musa available: yes
```

If the user says `docker-sglang`, first confirm the actual container name:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 shanfeng@10.18.32.25 "docker ps --format '{{.Names}}'"
```

## Basic Access

Interactive access:

```bash
ssh shanfeng@10.18.32.25
docker exec -it mochi-sglang bash
```

Non-interactive command pattern:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 shanfeng@10.18.32.25 "docker exec -w /workspace/workspace mochi-sglang bash -lc '<command>'"
```

Use `/workspace/workspace` for repository files mirrored from the workspace. Use `/tmp` for temporary binaries and generated outputs.

## Environment Checks

Run these checks before compiling or debugging an environment issue:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 shanfeng@10.18.32.25 "docker exec mochi-sglang bash -lc 'hostname; pwd; command -v mcc || true; mcc --version 2>/dev/null | head -5 || true'"
```

Check SDK headers and libraries. Use `find -L` because `/usr/local/musa` is a symlink:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 shanfeng@10.18.32.25 "docker exec mochi-sglang bash -lc 'find -L /usr/local/musa -maxdepth 3 -type f \( -name mupti.h -o -name musa_runtime.h -o -name \"libmupti.so*\" -o -name \"libmusart.so*\" \) 2>/dev/null | sort'"
```

Expected key files:

```text
/usr/local/musa/include/mupti.h
/usr/local/musa/include/musa_runtime.h
/usr/local/musa/lib/libmupti.so
/usr/local/musa/lib/libmusart.so
```

Check PyTorch/MUSA:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 shanfeng@10.18.32.25 "docker exec mochi-sglang bash -lc 'python3 - <<\"PY\"
import torch
print(\"torch\", torch.__version__)
print(\"has_musa_attr\", hasattr(torch, \"musa\"))
PY'"
```

## Compile a MUSA `.cu` File

Default compile command:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 shanfeng@10.18.32.25 "docker exec -w /workspace/workspace mochi-sglang bash -lc 'mcc -std=c++17 -o /tmp/<binary_name> <source.cu> -mtgpu -lmusart -pthread'"
```

For code that uses MUPTI, link `libmupti`:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 shanfeng@10.18.32.25 "docker exec -w /workspace/workspace mochi-sglang bash -lc 'mcc -std=c++17 -o /tmp/<binary_name> <source.cu> -mtgpu -lmusart -lmupti -pthread'"
```

If headers or libraries are not found, add explicit paths:

```bash
mcc -std=c++17 -o /tmp/<binary_name> <source.cu> \
  -I/usr/local/musa/include \
  -L/usr/local/musa/lib \
  -mtgpu -lmusart -lmupti -pthread
```

Check linked libraries:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 shanfeng@10.18.32.25 "docker exec mochi-sglang bash -lc 'ldd /tmp/<binary_name> | egrep \"mupti|musart|musa|not found\" || true'"
```

## MUPTI Demo

Compile the repository demo:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 shanfeng@10.18.32.25 "docker exec -w /workspace/workspace/insights/mupti mochi-sglang bash -lc 'mcc -std=c++17 -o /tmp/mupti_demo_axpy_trace demo.cu -mtgpu -lmusart -lmupti -pthread'"
```

Run the demo:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 shanfeng@10.18.32.25 "docker exec -w /workspace/workspace/insights/mupti mochi-sglang bash -lc '/tmp/mupti_demo_axpy_trace'"
```

Expected functional output:

```text
y[0] = 2
y[1] = 4
y[2] = 6
y[3] = 8
```

Expected trace categories include:

```text
DRIVER API
RUNTIME API
MEMCPY
CONC KERNEL
SYNC
```

## Common Failures

| Symptom | Check | Action |
|---|---|---|
| `docker-sglang` is not found | `docker ps --format '{{.Names}}'` on the remote host | Use the verified container name `mochi-sglang`, or confirm the renamed container |
| `mcc: command not found` | `command -v mcc` inside `mochi-sglang` | Source or install the MUSA SDK environment |
| `mupti.h` not found | `find -L /usr/local/musa -name mupti.h` | Add `-I/usr/local/musa/include` or use the correct SDK |
| `cannot find -lmupti` | `find -L /usr/local/musa -name 'libmupti.so*'` | Add `-L/usr/local/musa/lib` or use the correct SDK |
| Runtime library not found | `ldd /tmp/<binary_name>` | Add `/usr/local/musa/lib` to `LD_LIBRARY_PATH` |
| No MUPTI kernel records | Enabled activity kinds and workload launch | Confirm the program enables kernel activity and launches a real kernel |
| Python cannot use MUSA | `import torch; hasattr(torch, "musa")` | Use `mochi-sglang`; do not run PyTorch/MUSA checks locally |

## Hygiene

- Do not compile MUSA workloads in the local Codex workspace.
- Do not commit binaries, profiler traces, `.elf` files, or temporary outputs.
- Keep generated binaries under `/tmp` unless the user requests a persistent artifact.
- Report the exact remote host, container, working directory, command, and error log when a command fails.
