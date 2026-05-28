# muMemAllocAsync — 异步内存分配

## 功能

在指定 stream 上按 stream 顺序异步分配设备内存。与同步 `muMemAlloc` 不同，分配操作插入到 stream 的命令队列中，在 stream 中所有前序命令完成后才执行分配。对应 CUDA `cudaMallocAsync`。

## 完整调用链

```
用户代码: muMemAllocAsync(&dptr, 4096, hStream)
  │
  ├─ 1. mu_wrappers_generated.cpp — MUPTI 插桩
  │
  ├─ 2. mu_memory.cpp:303 — muapiMemAllocAsync
  │    ├─ InitPlatform()
  │    ├─ 校验: dptr? nullptr? bytesize==0?
  │    ├─ TlsCtxTop() → Context
  │    ├─ InfoStream(ctx, hStream) → Stream*
  │    └─ pStream->CmdMemAlloc(memAllocParam, false)
  │         memAllocParam = {size=4096, virtAddress(输出)}
  │
  ├─ 3. stream.cpp:570 — Stream::CmdMemAlloc
  │    ├─ [Capture active]   → CaptureMemAlloc (记录到 graph)
  │    ├─ [Capture invalid]  → 错误
  │    └─ [正常]             → AsyncMemAlloc(param, false)
  │
  ├─ 4. stream.cpp — AsyncMemAlloc (内部)
  │    ├─ 通过 MemoryPool 分配内存
  │    │   └─ 从 stream 关联的 pool (默认池)
  │    ├─ 分配结果写入 param.virtAddress
  │    └─ command 排队 (stream ordered)
  │
  └─ [Stream Submit Thread] 按序执行分配
       └─ 真正创建 Memory + Allocate GpuMemory
```

## 时序图

```
应用层            Wrapper          Driver(mu_mem)     Context          Stream           MemoryPool        HAL/KMD
  │                │                │                  │               │                │                 │
  │ muMemAllocAsync│                │                  │               │                │                 │
  │───────────────>│                │                  │               │                │                 │
  │                │                │                  │               │                │                 │
  │                │ muapiMemAlloc  │                  │               │                │                 │
  │                │ Async          │                  │               │                │                 │
  │                │───────────────>│                  │               │                │                 │
  │                │                │                  │               │                │                 │
  │                │                │ InitPlatform()   │               │                │                 │
  │                │                │                  │               │                │                 │
  │                │                │ 校验: dptr? size? │               │                │                 │
  │                │                │                  │               │                │                 │
  │                │                │ TlsCtxTop()      │               │                │                 │
  │                │                │─────────────────>│               │                │                 │
  │                │                │<── ctx* ─────────│               │                │                 │
  │                │                │                  │               │                │                 │
  │                │                │ InfoStream(ctx,  │               │                │                 │
  │                │                │   hStream)       │               │                │                 │
  │                │                │─────────────────>│──────────────>│                │                 │
  │                │                │<── Stream* ──────│<── Stream* ───│                │                 │
  │                │                │                  │               │                │                 │
  │                │                │ CmdMemAlloc      │               │                │                 │
  │                │                │ (param,false)    │               │                │                 │
  │                │                │─────────────────>│──────────────>│                │                 │
  │                │                │                  │               │                │                 │
  │                │                │                  │               │ AsyncMemAlloc   │                 │
  │                │                │                  │               │────────────────>│                 │
  │                │                │                  │               │                │                 │
  │                │                │                  │               │ [从 pool 分配]   │                 │
  │                │                │                  │               │ RequestMemory   │                 │
  │                │                │                  │               │ FromPool()      │                 │
  │                │                │                  │               │────────────────>│                 │
  │                │                │                  │               │                │ (具体实现取决于  │
  │                │                │                  │               │                │  池类型)         │                 │
  │                │                │                  │               │<── virtAddr ────│                 │
  │                │                │                  │               │                │                 │
  │                │                │                  │               │ 返回 virtAddr   │                 │
  │                │                │                  │               │ 到 param        │                 │
  │                │                │                  │               │                │                 │
  │                │                │<── OK + va ──────│<── OK ───────│                │                 │
  │                │                │                  │               │                │                 │
  │                │                │ *dptr = param.   │               │                │                 │
  │                │                │   virtAddress    │               │                │                 │
  │                │                │                  │               │                │                 │
  │                │<── OK ─────────│                  │               │                │                 │
  │<── dptr ───────│                │                  │               │                │                 │
```

## 同步 alloc vs 异步 alloc

```
muMemAlloc:                                    muMemAllocAsync:
                                                        
  Context::CreateMemory                          Stream::CmdMemAlloc
    → 立即分配 GPU 内存                            → Stream::AsyncMemAlloc
    → 立即返回 dptr                                 → 从 pool 分配 (可能立即也可能排队)
                                                    → 立即返回 dptr
                                                    → 实际内存可用在 stream fence 后

  stream 无关                                      stream 有序
  不参与 command 排队                              参与 stream 的 command 等待链
  调用完成后 dptr 立即可用                           dptr 在 stream 前序命令完成后才可用
  只用 context                                     需要 stream
```

## 关键设计要点

1. **Stream-ordered 分配**: 分配操作作为 stream 中的命令按序执行，保证与 stream 中其他操作的顺序
2. **通过内存池管理**: 异步分配通常与 `MemoryPool` 配合，通过池化管理来减少 KMD 调用
3. **虚拟地址提前返回**: 尽管内存可能尚未分配完成，但 `muMemAllocAsync` 预先返回虚拟地址，实际分配在 stream 中延迟执行
4. **配合 muMemFreeAsync**: 成对使用，在 stream 中安全地按序分配和释放内存
