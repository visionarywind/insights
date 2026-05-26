# MUPTI 调用流程日志插桩记录

## 1. 背景

本次任务是在远程主机的 MUPTI 仓库中增加调用流程日志，用于调试 MUPTI 从用户 API 到 driver/runtime hook、activity 记录生成、buffer flush 的完整执行路径。

远程环境：

```text
远程主机：shanfeng@10.18.32.25
仓库根目录：/home/shanfeng/workspace
MUPTI 仓库：/home/shanfeng/workspace/MUPTI
linux-ddk 仓库：/home/shanfeng/workspace/linux-ddk
编译容器：mochi-sglang
容器内 MUPTI 路径：/workspace/workspace/MUPTI
```

## 2. 日志方案

新增环境变量开关：

```bash
MUPTI_CALLFLOW_DEBUG=1
```

默认不开启日志。不开启时只保留一次环境变量判断，不输出日志，不改变原有执行逻辑。

新增日志宏：

```cpp
MUPTI_CALLFLOW_LOG(...)
```

实现位置：

```text
src/utils/log.h
src/utils/log.cpp
```

## 3. 插桩范围

### 3.1 MUPTI Activity API

文件：

```text
src/api/activity.cpp
```

覆盖接口：

```text
muptiActivityEnable
muptiActivityDisable
muptiActivityRegisterCallbacks
muptiActivityFlushAll
muptiFinalize
```

用于观察：

```text
Activity kind 开启和关闭
buffer callback 注册
flush 是否强制执行
finalize 时 hook disable 和内部状态清理
```

### 3.2 MUPTI Callback API

文件：

```text
src/api/callback.cpp
```

覆盖接口：

```text
muptiSubscribe
muptiUnsubscribe
muptiEnableCallback
muptiEnableDomain
muptiEnableAllDomains
```

用于观察：

```text
subscriber 注册
callback domain/cbid 开关
是否转发到 driver ToolsCallback
```

### 3.3 初始化与 hook 注入

文件：

```text
src/core/init.cpp
src/injection/injection.cpp
```

覆盖流程：

```text
MUpti::init
inject_musa
get_export_table_from_musa_driver
get_export_table_from_musa_runtime
driver hook enable
runtime hook enable
driver accessors import
```

用于观察：

```text
libmusa.so / libmusart.so 是否加载成功
muGetExportTable / musaGetExportTable 是否成功
driver/runtime hook 是否启用
当前 driver 是否支持 ToolsCallback
```

### 3.4 Activity buffer

文件：

```text
src/core/buffer.cpp
```

覆盖流程：

```text
ActivityBufferManager::request_buffer
ActivityBufferManager::flush_buffers
ActivityBufferManager 析构 flush
```

用于观察：

```text
用户 buffer 申请
buffer 容量
valid_size
flush buffer 数量
force flush 等待 kernel/memop 完成的状态
```

### 3.5 Runtime/Driver API hook 路径

文件：

```text
src/core/process_callback.cpp
src/core/core.cpp
```

覆盖路径：

```text
ProcessDriverCallback
EnterRuntimeApi / ExitRuntimeApi
EnterDriverApi / ExitDriverApi
RegisterKernel
MarkKernelQueued
MarkKernelSubmitted
AssignKernelToKick
```

用于观察：

```text
runtime API 进入和退出
driver API 进入和退出
API cbid
是否记录 activity
是否触发用户 callback
kernel correlationId
kernel queued/submitted 时间点
correlationId 到 kickId 的关联
```

## 4. 构建问题和修复

### 4.1 问题现象

首次编译 MUPTI 时失败：

```text
generated_musa_runtime_api_meta.h:1249:5:
error: 'musaLogsCallback_t' does not name a type

generated_musa_runtime_api_meta.h:1251:5:
error: 'musaLogsCallbackHandle' does not name a type

generated_musa_runtime_api_meta.h:1259:5:
error: 'musaLogIterator' does not name a type
```

### 4.2 直接原因

`generated_musa_runtime_api_meta.h` 使用了 runtime log API 参数类型：

```text
musaLogsCallback_t
musaLogsCallbackHandle
musaLogIterator
```

但编译时这些类型没有在 include 链中提前定义。

### 4.3 根因

`musa.h` 或 `perf.h` 会间接包含 SDK 路径下的：

```text
/usr/local/musa/include/driver_types.h
```

该头文件先定义了 `__DRIVER_TYPES_H__` include guard。随后 MUPTI 自带的：

```text
/workspace/workspace/MUPTI/musa_shared_include/driver_types.h
```

因为 include guard 已经存在，被直接跳过。结果是 MUPTI 自带的新类型没有进入编译单元。

### 4.4 修复方式

在 MUPTI public header 中，让 MUPTI 自带的 `driver_types.h` 先于 `musa.h` 被包含：

```text
include/mupti.h
include/mupti_activity.h
include/mupti_callbacks.h
include/mupti_events.h
include/mupti_metrics.h
include/mupti_pcsampling.h
include/mupti_profiler_target.h
```

同时在内部 buffer 头文件中，让 `driver_types.h` 先于 `perf.h` 被包含：

```text
src/core/buffer.h
```

补充缺失 typedef：

```text
musa_shared_include/driver_types.h
```

新增：

```cpp
typedef void (MUSART_CB *musaLogsCallback_t)(
    void *data,
    musaLogLevel logLevel,
    char *message,
    size_t length);
```

## 5. 修改文件

MUPTI 主仓库修改：

```text
include/mupti.h
include/mupti_activity.h
include/mupti_callbacks.h
include/mupti_events.h
include/mupti_metrics.h
include/mupti_pcsampling.h
include/mupti_profiler_target.h
src/api/activity.cpp
src/api/callback.cpp
src/core/buffer.cpp
src/core/buffer.h
src/core/core.cpp
src/core/init.cpp
src/core/process_callback.cpp
src/injection/injection.cpp
src/utils/log.cpp
src/utils/log.h
```

MUPTI 子仓库或子模块修改：

```text
musa_shared_include/driver_types.h
```

## 6. 编译验证

编译命令：

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 \
  shanfeng@10.18.32.25 \
  "docker exec -w /workspace/workspace/MUPTI mochi-sglang bash -lc 'cmake --build build --target mupti -j 8'"
```

编译结果：

```text
[100%] Built target mupti
```

生成产物：

```text
/workspace/workspace/MUPTI/build/src/libmupti.so
/workspace/workspace/MUPTI/build/src/libmupti.so.1
/workspace/workspace/MUPTI/build/src/libmupti.so.1.3.0
```

## 7. Demo 验证

使用新编译的 `libmupti.so` 编译 demo：

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 \
  shanfeng@10.18.32.25 \
  "docker exec -w /workspace/workspace/insights/mupti mochi-sglang bash -lc \
  'mcc -std=c++17 -o /tmp/mupti_demo_callflow demo.cu \
   -I/workspace/workspace/MUPTI/include \
   -I/workspace/workspace/MUPTI/musa_shared_include \
   -I/workspace/workspace/MUPTI/musa_shared_include/mupti \
   -L/workspace/workspace/MUPTI/build/src \
   -Wl,-rpath,/workspace/workspace/MUPTI/build/src \
   -mtgpu -lmusart -lmupti -pthread'"
```

确认链接到新库：

```text
libmupti.so.1 => /workspace/workspace/MUPTI/build/src/libmupti.so.1
```

运行 demo：

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 \
  shanfeng@10.18.32.25 \
  "docker exec -w /workspace/workspace/insights/mupti mochi-sglang bash -lc \
  'MUPTI_CALLFLOW_DEBUG=1 \
   LD_LIBRARY_PATH=/workspace/workspace/MUPTI/build/src:/usr/local/musa/lib \
   timeout 20s /tmp/mupti_demo_callflow'"
```

功能输出正常：

```text
y[0] = 2
y[1] = 4
y[2] = 6
y[3] = 8
```

## 8. 关键日志结果

### 8.1 初始化和 hook 注入

日志显示：

```text
muptiActivityEnable: enter kind=DRIVER(4)
init: first initialization, injecting MUSA hooks
inject_musa: enter
get_export_table_from_musa_driver: loading libmusa.so.1
get_export_table_from_musa_driver: requesting MUpti driver export table
get_export_table_from_musa_driver: got MUpti driver export table
get_export_table_from_musa_driver: requesting Tools export table
get_export_table_from_musa_driver: Tools callback is not supported by driver
inject_musa: enabling driver hooks
inject_musa: importing driver accessors
get_export_table_from_musa_runtime: loading libmusart.so
get_export_table_from_musa_runtime: requesting MUpti runtime export table
get_export_table_from_musa_runtime: runtime export table
inject_musa: enabling runtime hooks
inject_musa: done
init: MUSA hook injection succeeded
```

结论：

```text
MUPTI 成功加载 driver/runtime export table。
driver hook 和 runtime hook 均已启用。
当前 driver 不支持 ToolsCallback 路径，本次走的是传统 MUpti hook 路径。
```

### 8.2 Activity 开启

日志显示：

```text
muptiActivityEnable: enter kind=DRIVER(4)
muptiActivityEnable: enter kind=RUNTIME(5)
muptiActivityEnable: enter kind=MEMCPY(1)
muptiActivityEnable: enter kind=MEMSET(2)
muptiActivityEnable: enter kind=KERNEL(3)
muptiActivityEnable: enter kind=CONCURRENT_KERNEL(10)
muptiActivityEnable: enter kind=SYNCHRONIZATION(38)
```

结论：

```text
demo 开启了 driver、runtime、memcpy、memset、kernel、concurrent kernel、synchronization activity。
MEMCPY、MEMSET、KERNEL、CONCURRENT_KERNEL 会触发 device monitor open。
```

### 8.3 Runtime 和 Driver API

日志显示：

```text
EnterRuntimeApi: cbid=20, isInvocation=0, activity=1, callback=0
ExitRuntimeApi: ctx=..., status=0
EnterRuntimeApi: cbid=31, isInvocation=0, activity=1, callback=0
ExitRuntimeApi: ctx=..., status=0
EnterRuntimeApi: cbid=211, isInvocation=0, activity=1, callback=0
ExitRuntimeApi: ctx=..., status=0
```

结论：

```text
runtime activity 已启用。
用户 callback 未启用，因此 callback=0。
cbid=20 对应 musaMalloc。
cbid=31 对应 musaMemcpy。
cbid=211 对应 musaLaunchKernel。
```

### 8.4 Kernel 记录

日志显示：

```text
MarkKernelQueued: correlation=38
MarkKernelSubmitted: correlation=38
AssignKernelToKick: correlation=38, kick=3
```

结论：

```text
kernel launch 后，MUPTI 能记录 queued、submitted 和 kick 映射。
activity 输出中能看到 CONC KERNEL 记录，并带有 correlation、queued、submitted、streamId、grid/block 信息。
```

### 8.5 Buffer flush

日志显示：

```text
ActivityBufferManager::request_buffer: user buffer raw=..., byte_cap=2568, elem_cap=0
ActivityBufferManager::flush_buffers: collected buffers=1
ActivityBufferManager::flush_buffers: complete buffer=..., byte_cap=2568, valid_size=2424
```

结论：

```text
MUPTI 从用户 callback 获取 buffer。
activity record 写入 buffer。
flush 时调用用户 complete callback 输出记录。
```

### 8.6 Finalize

日志显示：

```text
muptiActivityDisable: enter kind=DRIVER(4)
muptiActivityDisable: enter kind=RUNTIME(5)
muptiActivityDisable: enter kind=MEMCPY(1)
muptiActivityDisable: enter kind=MEMSET(2)
muptiActivityDisable: enter kind=KERNEL(3)
muptiActivityDisable: enter kind=CONCURRENT_KERNEL(10)
muptiActivityDisable: enter kind=SYNCHRONIZATION(38)
muptiFinalize: enter
muptiFinalize: disable driver hooks
muptiFinalize: disable runtime hooks
muptiFinalize: done
ActivityBufferManager: destructor enter, callback_registered=1
```

结论：

```text
demo 结束时 activity 被逐项关闭。
finalize 会关闭 driver/runtime hook。
ActivityBufferManager 析构时会做最终 force flush。
```

## 9. 最终结论

本次插桩已经可以支持 MUPTI 调用流程 debug。

当前能观察到的主链路为：

```text
muptiActivityEnable
  -> MUpti::init
  -> inject_musa
  -> get driver export table
  -> enable driver hooks
  -> import driver accessors
  -> get runtime export table
  -> enable runtime hooks

muptiActivityRegisterCallbacks
  -> ActivityBufferManager 注册 request/complete callback

MUSA workload
  -> EnterRuntimeApi / ExitRuntimeApi
  -> EnterDriverApi / ExitDriverApi
  -> RegisterKernel / MarkKernelQueued / MarkKernelSubmitted
  -> AssignKernelToKick
  -> activity record 写入 buffer

muptiActivityFlushAll
  -> ActivityBufferManager::flush_buffers
  -> 用户 complete callback 输出 activity records

muptiFinalize
  -> disable driver hooks
  -> disable runtime hooks
  -> 清理 MUPTI 内部状态
```

验证结果：

```text
MUPTI 编译通过。
demo 编译通过。
demo 运行通过。
MUPTI_CALLFLOW_DEBUG=1 能输出完整调用流程日志。
demo 功能输出正确。
```
