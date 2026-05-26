# muMemPrefetchAsync — 异步内存预取

> 源码文件：`musa/src/driver/mu_memory.cpp` (预取 API), `musa/src/musa/core/stream.cpp` (预取命令)

## 1. 功能概述

`muMemPrefetchAsync` 将指定设备内存区域的**数据预取**到指定的目标设备上，使得目标设备在后续访问该区域时可以更快地获得数据。

这是一个流有序的异步操作，预取命令会在 stream 中排队执行。

## 2. 调用链

```
muapiMemPrefetchAsync(ptr, count, dstDevice, hStream)
  │
  +-- InitPlatform()
  │
  +-- 参数校验:
  │     ptr != 0, count != 0, dstDevice 有效
  │
  +-- TlsCtxTop() → Context*
  │
  +-- GetMemoryByDevicePointer(ptr, &offset) → pMemory
  │     (通过 MemoryTracker 全局查找)
  │
  +-- pMemory->GetProps() & DeviceMapped 校验
  │     (仅已映射到设备的 memory 才可预取)
  │
  +-- 创建预取命令 → Stream::CmdPrefetchAsync() / PrefetchCommand
  │     (将预取操作编码为 stream 中的命令)
  │
  +-- EncodePrefetch(目标设备, 偏移, 大小)
        在硬件命令中编码预取语义
```

## 3. 设计要点

- **目的**: 减少 GPU 访问其他设备内存时的延迟
- **流有序**: 预取命令与 stream 中的其他操作有序执行
- **NUMA 优化**: 将内存预取到目标设备所在的 NUMA 节点
- **page fault 迁移**: 对于 managed 内存, 预取会触发 page migration

## 4. 相关源码位置

| 文件 | 说明 |
|------|------|
| `musa/src/driver/mu_memory.cpp` | API 入口 |
| `musa/src/musa/core/stream.cpp` | 预取命令实现 |
| `musa/src/musa/core/command/prefetchCommand.h` | 预取命令类 |