---
name: musa-build
description: Use this skill when building or validating MUSA linux-ddk on the remote host shanfeng@10.18.32.25, or when compiling MUPTI demos in the mochi-sglang Docker container. It covers the mochi-build DDK build flow, the mochi-sglang MUPTI demo compile flow, environment checks, and common failure checks.
---

# MUSA Build

## Scope

Use this skill for tasks that involve:

- Building `linux-ddk` in the MUSA build Docker environment.
- Checking whether the remote/container environment has a MUSA SDK compiler, runtime libraries, and MUPTI libraries.
- Compiling standalone MUSA/MUPTI `.cu` examples in the correct remote Docker container.
- Explaining or debugging DDK build failures in `mochi-build` and MUPTI demo failures in `mochi-sglang`.

Do not assume the local Codex workspace has a MUSA device or MUSA compiler. Check the selected remote container before building.

## Standard Environment

Known DDK build environment:

```text
Remote host: shanfeng@10.18.32.25
DDK build container: mochi-build
DDK path: /root/linux-ddk
Standard DDK build command: ./ddk_build.sh -a 0 -m 1
```

Known MUPTI demo environment:

```text
Remote host: shanfeng@10.18.32.25
MUPTI demo container: mochi-sglang
Workspace path in container: /workspace/workspace
Demo source path: /workspace/workspace/insights/mupti/demo.cu
Compiler: /usr/local/musa/bin/mcc
Default SDK path: /usr/local/musa
```

MUSA builds and MUPTI demo compilation are executed on the remote host, not in the local Codex workspace.

Connect to the remote host:

```bash
ssh shanfeng@10.18.32.25
```

Then enter the build container:

```bash
docker exec -it mochi-build bash
```

Enter the MUPTI demo container:

```bash
docker exec -it mochi-sglang bash
```

For non-interactive execution from the local workspace:

```bash
ssh shanfeng@10.18.32.25 "docker exec mochi-build bash -lc 'cd /root/linux-ddk && ./ddk_build.sh -a 0 -m 1'"
```

Before running a build, confirm the remote/container execution context:

```bash
pwd
ls /root/linux-ddk
ls -l /root/linux-ddk/ddk_build.sh
command -v mcc || true
test -d /usr/local/musa && ls /usr/local/musa || true
find /root/linux-ddk -maxdepth 5 -type f -name 'libmusa.so*' | head
```

## Build linux-ddk

Default build:

```bash
ssh shanfeng@10.18.32.25
docker exec -it mochi-build bash
cd /root/linux-ddk
./ddk_build.sh -a 0 -m 1
```

Non-interactive build:

```bash
ssh shanfeng@10.18.32.25 "docker exec mochi-build bash -lc 'cd /root/linux-ddk && ./ddk_build.sh -a 0 -m 1'"
```

Recommended checks after the build:

```bash
echo $?
find /root/linux-ddk -maxdepth 3 -type f -name 'libmusa.so*' | head
find /root/linux-ddk -maxdepth 4 -type f -name 'libmupti.so*' | head
```

If the task asks for a specific component, inspect `./ddk_build.sh -h` before changing flags:

```bash
ssh shanfeng@10.18.32.25 "docker exec mochi-build bash -lc 'cd /root/linux-ddk && ./ddk_build.sh -h'"
```

## Compile a MUPTI Demo

MUPTI `.cu` demos are compiled in `mochi-sglang`, not `mochi-build`. Check the compiler, headers, and libraries first:

```bash
ssh shanfeng@10.18.32.25 "docker exec mochi-sglang bash -lc 'command -v mcc || true'"
ssh shanfeng@10.18.32.25 "docker exec mochi-sglang bash -lc 'find /usr/local /workspace /data -type f -name mupti.h 2>/dev/null | head'"
ssh shanfeng@10.18.32.25 "docker exec mochi-sglang bash -lc 'find /usr/local /workspace /data -type f -name musa_runtime.h 2>/dev/null | head'"
ssh shanfeng@10.18.32.25 "docker exec mochi-sglang bash -lc 'find /usr/local /workspace /data -type f -name \"libmupti.so*\" 2>/dev/null | head'"
ssh shanfeng@10.18.32.25 "docker exec mochi-sglang bash -lc 'find /usr/local /workspace /data -type f -name \"libmusart.so*\" 2>/dev/null | head'"
```

Expected default paths in `mochi-sglang`:

```text
/usr/local/musa/bin/mcc
/usr/local/musa/include/mupti.h
/usr/local/musa/include/musa_runtime.h
/usr/local/musa/lib/libmupti.so.1.0
/usr/local/musa/lib/libmusart.so.4
```

Compile `insights/mupti/demo.cu`:

```bash
ssh shanfeng@10.18.32.25 "docker exec -w /workspace/workspace/insights/mupti mochi-sglang bash -lc 'mcc -std=c++17 -o /tmp/mupti_demo_axpy_trace demo.cu -mtgpu -lmusart -lmupti -pthread'"
```

Check linked runtime libraries:

```bash
ssh shanfeng@10.18.32.25 "docker exec mochi-sglang bash -lc 'ldd /tmp/mupti_demo_axpy_trace | egrep \"mupti|musart|musa|not found\" || true'"
```

Run the demo:

```bash
ssh shanfeng@10.18.32.25 "docker exec -w /workspace/workspace/insights/mupti mochi-sglang bash -lc '/tmp/mupti_demo_axpy_trace'"
```

For `insights/mupti/demo.cu`, expected functional output includes:

```text
y[0] = 2
y[1] = 4
y[2] = 6
y[3] = 8
```

The program should also print MUPTI activity records such as `RUNTIME API`, `DRIVER API`, `MEMCPY`, `CONC KERNEL`, and possibly `SYNC`.
