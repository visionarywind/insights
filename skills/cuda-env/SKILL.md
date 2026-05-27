---
name: cuda-env
description: Use this skill when a task needs a CUDA environment for compilation or validation on the remote host shanfeng@172.31.8.45, including nvcc checks, CUDA header/library checks, CUDA benchmark builds, CUDA runtime sanity tests, and CUDA-side validation for cross-backend MUSA/CUDA code.
---

# CUDA Env

## Scope

Use this skill when the task requires a CUDA-capable environment instead of the local Codex workspace:

- Compile CUDA `.cu` code with `nvcc`.
- Validate CUDA Runtime or CUDA Driver API examples.
- Build `musa_benchmarks` with `ENABLE_NVCC=ON`.
- Verify CUDA headers, runtime libraries, driver libraries, and GPU visibility.
- Compare CUDA behavior with a MUSA implementation.

Do not use this skill for MUSA-only validation. Use `musa-env` for MUSA runtime checks and `musa-build` for DDK builds.

## Remote Host

Authoritative CUDA remote:

```text
Host: shanfeng@172.31.8.45
Default working directory: /home/shanfeng
Observed hostname: mt-System-Product-Name
Observed CUDA compiler: /usr/bin/nvcc
Observed CUDA version: 12.8
Observed GPU: NVIDIA GeForce RTX 3060
```

Use non-interactive SSH by default:

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 shanfeng@172.31.8.45 "<command>"
```

For repository work, start from the remote workspace:

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 shanfeng@172.31.8.45 \
  "cd /home/shanfeng && ls"
```

Do not assume `/home/shanfeng/workspace` exists on this host. Locate or create the task-specific repository
directory before running repository builds.

## Environment Checks

Run these checks before compiling or diagnosing CUDA failures:

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 shanfeng@172.31.8.45 \
  "hostname; command -v nvcc || true; nvcc --version 2>/dev/null || true; nvidia-smi 2>/dev/null | head -20 || true"
```

Check CUDA headers and libraries:

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 shanfeng@172.31.8.45 \
  "find -L /usr/local/cuda -maxdepth 4 -type f \( -name cuda.h -o -name cuda_runtime.h -o -name 'libcuda.so*' -o -name 'libcudart.so*' \) 2>/dev/null | sort"
```

Expected key files:

```text
/usr/local/cuda/include/cuda.h
/usr/local/cuda/include/cuda_runtime.h
/usr/local/cuda/lib64/libcudart.so
```

`libcuda.so` may be provided by the driver path instead of `/usr/local/cuda/lib64`.

## Compile a Standalone CUDA File

Default command:

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 shanfeng@172.31.8.45 \
  "cd /home/shanfeng && nvcc -std=c++17 <source.cu> -o /tmp/<binary_name>"
```

For CUDA Driver API code, link the driver library explicitly:

```bash
nvcc -std=c++17 <source.cu> -o /tmp/<binary_name> -lcuda
```

For code that needs explicit include/library paths:

```bash
nvcc -std=c++17 <source.cu> -o /tmp/<binary_name> \
  -I/usr/local/cuda/include \
  -L/usr/local/cuda/lib64 \
  -lcudart -lcuda
```

Run the binary with a clean library path:

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 shanfeng@172.31.8.45 \
  "LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH /tmp/<binary_name>"
```

## Build musa_benchmarks with CUDA

Use this when validating a cross-backend benchmark such as `greenContextIsolation`.

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 shanfeng@172.31.8.45 \
  "cd <path-to-musa_benchmarks> && mkdir -p build_cuda && cd build_cuda && cmake .. -DENABLE_NVCC=ON && make greenContextIsolation -j"
```

Run the target:

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 shanfeng@172.31.8.45 \
  "cd <path-to-musa_benchmarks>/build_cuda/schedule && LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH ./greenContextIsolation -l"
```

If the case uses CUDA Green Context APIs, CUDA 12.4 or newer headers are expected. Older CUDA headers may build only an unsupported marker path if the source implements that fallback.

## Remote Editing Workflow

For remote repository edits:

1. Inspect the file with `ssh` and `sed`, `nl`, `rg`, or `git diff`.
2. Create a focused local patch under `/tmp`.
3. Apply remotely with `git apply --check`.
4. Apply remotely with `git apply`.
5. Validate with a targeted CUDA build or runtime command.

Patch pattern:

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 shanfeng@172.31.8.45 \
  "cd <path-to-repo> && git apply --check -" < /tmp/<change>.patch

ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 shanfeng@172.31.8.45 \
  "cd <path-to-repo> && git apply -" < /tmp/<change>.patch
```

## Common Failures

| Symptom | Check | Action |
|---|---|---|
| `nvcc: command not found` | `command -v nvcc` | Source CUDA environment or use the correct CUDA install path |
| `cuda.h` not found | `find -L /usr/local/cuda -name cuda.h` | Add `-I/usr/local/cuda/include` or select the correct toolkit |
| `cannot find -lcuda` | `ldconfig -p | grep libcuda` | Add the driver library path or use the host driver environment |
| `cannot find -lcudart` | `find -L /usr/local/cuda -name 'libcudart.so*'` | Add `-L/usr/local/cuda/lib64` |
| Runtime library not found | `ldd /tmp/<binary_name>` | Set `LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH` |
| No GPU visible | `nvidia-smi` | Confirm the remote host has driver access and the session is on the correct machine |
| Green Context type missing | `grep -n 'CUgreenCtx' /usr/local/cuda/include/cuda.h` | Use CUDA 12.4+ headers or keep the unsupported fallback enabled |

## Hygiene

- Do not compile CUDA workloads in the local Codex workspace when this skill applies.
- Keep temporary binaries under `/tmp` unless the user requests a persistent artifact.
- Do not commit binaries, profiler traces, generated CMake outputs, or backup files.
- Report the remote host, repository path, exact command, return code, and key output lines after validation.
- If CUDA validation cannot run, state whether the blocker is SSH access, missing `nvcc`, missing headers, missing GPU visibility, or build failure.
