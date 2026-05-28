# MUSA Kernel Profiler

基于 MUPTI Driver Hooks 的 GPU kernel 耗时分析工具。

## 架构

```
用户程序 (musaLaunchKernel)
    │
    ▼
libmusa_driver.so ── MUPTI tracepoints ──→ libmusaKernelProfiler.so (本库)
    │                                              │
    │ RegisterKernelV2(DispatchCommand)             │ 创建 KernelRecord
    │ MarkCommandBeginEnd(Context, Command)          │ 读 GPU 时间戳 + 输出报告
    │
    ▼
    输出: stdout (每个 kernel 一行)
```

## 构建

```bash
cd doc/profiler
make
# 生成 libmusaKernelProfiler.so
```

## 使用

```bash
# 设置注入路径
export MUSA_INJECTION64_PATH=/path/to/doc/profiler/libmusaKernelProfiler.so

# 运行你的 MUSA 应用
./your_musa_app

# 输出示例:
# [PROFILER] 2026-05-19 14:32:01.123 | vectorAdd                          | Grid(128,1,1) Block(256,1,1) DynamicSmem:0
# [PROFILER]   CPU调度:    12.34 us | GPU队列:     5.67 us | GPU执行:   123.45 us | 端到端:   141.46 us
```

## Hook 回调与线程模型

| Hook | 调用线程 | 触发时机 |
|------|----------|----------|
| `OnRegisterKernelV2` | Submit 线程 | DispatchCommand 构造函数 |
| `OnMarkCommandBeginEnd` | Wait 线程 | GPU 完成后 ReleaseResources() |
| `OnMarkKernelQueued` | Enqueue 线程 | (旧平台) 命令入队 |
| `OnMarkKernelSubmitted` | Submit 线程 | (旧平台) 命令提交到 GPU |

## 数据流

```
RegisterKernelV2
  → 读取: kernel 名称, grid/block, correlationId
  → 创建 KernelRecord → 存入 g_Records[correlationId]
  → 返回 record 指针 (作为 MUpti::Context)

  ... kernel 在 GPU 上执行 ...

MarkCommandBeginEnd
  → 取回 record (从 MUpti::Context* cast)
  → 读取: queued_time, submit_time, gpu_begin, gpu_end
  → 计算: CPU调度时间, GPU队列等待, GPU执行时间, 端到端延迟
  → 输出报告 → 删除 record
```
