# MUSA Runtime、Driver 与 MUPTI 端到端调用流程日志验证

## 1. 验证环境

远程主机：

```text
shanfeng@10.18.32.25
```

远程仓库：

```text
/home/shanfeng/workspace/MUPTI
/home/shanfeng/workspace/MUSA-Runtime
/home/shanfeng/workspace/linux-ddk/musa
/home/shanfeng/workspace/insights/mupti/demo.cu
```

容器：

```text
mochi-sglang：编译和运行 MUPTI demo、MUSA-Runtime
mochi-build：编译 linux-ddk/musa driver
```

本次验证使用 `insights/mupti/demo.cu`。该 demo 的用户态调用顺序为：

```text
initTrace
musaMalloc(device_x)
musaMalloc(device_y)
musaMemcpy(H2D)
musaStreamCreate x 10
musaLaunchKernel x 10
musaDeviceSynchronize
musaMemcpy(D2H)
musaFree(device_x)
musaFree(device_y)
musaDeviceReset
finiTrace
```

## 2. 日志开关

新增日志均由环境变量控制，默认关闭。

```bash
MUPTI_CALLFLOW_DEBUG=1
MUSART_CALLFLOW_DEBUG=1
MUSA_DRIVER_CALLFLOW_DEBUG=1
```

关闭时只执行一次环境变量读取和布尔判断，不输出日志，不改变原有调用顺序。Runtime 和 Driver 的日志宏在进入格式化输出前先检查开关，避免默认路径产生字符串格式化开销。

日志前缀：

```text
[MUPTI_CALLFLOW]        MUPTI 初始化、hook、activity、callback、buffer flush
[MUSART_CALLFLOW]       MUSA-Runtime API wrapper 和 runtime API 内部路径
[MUSA_DRIVER_CALLFLOW]  Driver API、Core Context/Stream、Command、HAL queue 路径
```

## 3. 插桩范围

### 3.1 MUSA-Runtime

远程仓库：`/home/shanfeng/workspace/MUSA-Runtime`

```text
src/internal.h
src/musa_memory.cpp
src/musa_module.cpp
src/musa_stream.cpp
src/musa_device.cpp
```

覆盖路径：

```text
ApiInvocationGuard enter/exit
musaapiMalloc
musaapiFree
musaapiMemcpy
musaapiLaunchKernel
musaapiStreamCreate
musaapiDeviceSynchronize
musaapiDeviceReset
```

Runtime 日志用于确认：

```text
用户调用进入了哪个 runtime API
runtime API 选择了哪个 driver API
musaMemcpy 的方向分类结果
musaLaunchKernel 的函数解析和 driver launch 参数
runtime API 的返回状态
```

### 3.2 linux-ddk/musa Driver 与 Core

远程仓库：`/home/shanfeng/workspace/linux-ddk/musa`

```text
src/driver/internal.h
src/driver/mu_memory.cpp
src/driver/mu_module.cpp
src/driver/mu_stream.cpp
src/driver/mu_context.cpp
src/driver/mu_device.cpp
src/musa/core/context.cpp
src/musa/core/stream.cpp
src/musa/core/command/command.cpp
src/musa/core/command/SyncMemcpyCommand.cpp
src/musa/core/command/AsyncMemcpyCommand.cpp
src/musa/core/command/dispatchCommand.cpp
```

覆盖路径：

```text
Driver ApiInvocationGuard enter/exit
muMemAlloc_v2 / muMemFree_v2
muMemcpyHtoD_v2 / muMemcpyDtoH_v2 / muMemcpyDtoD_v2 / muMemcpy / muMemcpyAsync
muStreamCreate / muStreamCreateWithPriority / muStreamGetDevice
muLaunchKernel
muCtxSynchronize
muDeviceGetProperties
muDevicePrimaryCtxRetain / muDevicePrimaryCtxReset_v2
Context::CreateMemory / DestroyMemory
Context::GeneralMemcpy
Context::CreateStream
Context::GeneralLaunchKernel
Context::Synchronize
Stream::CmdCopyMemory
Stream::CmdLaunchKernel
Stream::AsyncSubmit
Stream::WaitFinish
SyncMemcpyCommand::Submit
AsyncMemcpyCommand::Submit
DispatchCommand::Submit
Command::SubmitToQueue
IQueue::Submit
```

Driver 与 Core 日志用于确认：

```text
driver API 是否进入 core 层
core 层选择了同步 memcpy command 还是 kernel dispatch command
command 是否进入 stream 的异步提交路径
command 是否提交到 HAL queue
同步和释放是否等待已有 stream command 完成
```

### 3.3 MUPTI

MUPTI 的日志已覆盖：

```text
muptiActivityEnable / Disable
muptiActivityRegisterCallbacks
muptiActivityFlushAll
muptiFinalize
inject_musa
driver export table 获取
runtime export table 获取
driver hook enable
runtime hook enable
ProcessDriverCallback
EnterRuntimeApi / ExitRuntimeApi
EnterDriverApi / ExitDriverApi
kernel queued / submitted / activity record
ActivityBufferManager request / flush
```

MUPTI 日志用于确认：

```text
MUPTI 是否通过 export table 注入 driver/runtime hook
runtime 和 driver API 是否被 callback 捕获
activity record 是否写入并在 flush 时输出
kernel correlationId 是否贯穿 runtime API 与 kernel activity
```

## 4. 构建与运行

### 4.1 构建 MUSA-Runtime

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 shanfeng@10.18.32.25 \
  "docker exec mochi-sglang bash -lc 'ln -sfn /workspace/workspace/MUSA-Runtime /workspace/MUSA-Runtime && cd /workspace/MUSA-Runtime && cmake --build build --target musart -j 8'"
```

结果：

```text
[100%] Built target musart
```

### 4.2 构建 linux-ddk/musa Driver

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 shanfeng@10.18.32.25 \
  "docker exec mochi-build bash -lc 'cd /root/linux-ddk/musa && cmake -S . -B build && cmake --build build --target musa_dynamic -j 8'"
```

结果：

```text
[100%] Built target musa_dynamic
```

构建中存在既有 ODR warning，但目标库构建成功。

### 4.3 编译 Demo

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 shanfeng@10.18.32.25 \
  "docker exec -w /workspace/workspace/insights/mupti mochi-sglang bash -lc 'mcc -std=c++17 -o /tmp/mupti_demo_full_callflow demo.cu -I/workspace/workspace/MUPTI/include -I/workspace/workspace/MUPTI/musa_shared_include -I/workspace/workspace/MUPTI/musa_shared_include/mupti -L/workspace/workspace/MUPTI/build/src -L/workspace/MUSA-Runtime/build/lib -L/workspace/workspace/linux-ddk/musa/build/lib -Wl,-rpath,/workspace/workspace/MUPTI/build/src -Wl,-rpath,/workspace/MUSA-Runtime/build/lib -Wl,-rpath,/workspace/workspace/linux-ddk/musa/build/lib -mtgpu -lmusart -lmupti -pthread'"
```

链接到的关键库：

```text
/workspace/MUSA-Runtime/build/lib/libmusart.so.5
/workspace/workspace/MUPTI/build/src/libmupti.so.1
```

### 4.4 运行 Demo

```bash
ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=10 shanfeng@10.18.32.25 \
  "docker exec -w /workspace/workspace/insights/mupti mochi-sglang bash -lc 'MUPTI_CALLFLOW_DEBUG=1 MUSART_CALLFLOW_DEBUG=1 MUSA_DRIVER_CALLFLOW_DEBUG=1 LD_LIBRARY_PATH=/workspace/workspace/MUPTI/build/src:/workspace/MUSA-Runtime/build/lib:/workspace/workspace/linux-ddk/musa/build/lib:/usr/local/musa/lib timeout 40s /tmp/mupti_demo_full_callflow > /tmp/mupti_demo_full_callflow_rerun.log 2>&1; rc=\$?; echo rc=\$rc; wc -l /tmp/mupti_demo_full_callflow_rerun.log; grep -n -F -e \"y[0] =\" -e \"y[1] =\" -e \"y[2] =\" -e \"y[3] =\" /tmp/mupti_demo_full_callflow_rerun.log'"
```

结果：

```text
rc=0
2360 /tmp/mupti_demo_full_callflow_rerun.log
1820:y[0] = 2
1821:y[1] = 4
1822:y[2] = 6
1823:y[3] = 8
```

结果说明：kernel 执行了 `y = 2.0 * x`，输入为 `[1, 2, 3, 4]`，输出为 `[2, 4, 6, 8]`。Runtime、Driver、Core、Command 和 MUPTI activity 路径均产生了日志。

## 5. 端到端分层总览

本次验证到的完整层次如下：

```text
用户代码 demo.cu
  -> MUPTI Activity API
  -> MUPTI 注入 driver/runtime hooks
  -> MUSA-Runtime wrapper
  -> MUSA-Runtime musaapi* 实现
  -> Driver wrapper
  -> Driver muapi* 实现
  -> Core Context
  -> Core Stream
  -> Command 对象
  -> CopyManager 或 DispatchCommand
  -> Command::SubmitToQueue
  -> HAL IQueue::Submit
  -> MUPTI activity buffer
  -> 用户 bufferCompleted 回调输出 activity records
```

日志来自多个线程。`demo.cu` 中有周期性 flush 线程，Core stream 也存在异步提交，因此部分日志行会交织出现。分析调用顺序时，以同一 API 的 enter/exit、correlation、command 指针、stream 指针和 activity record 为准。

## 6. MUPTI 初始化流程

用户代码调用：

```cpp
initTrace();
```

`initTrace` 依次启用 Driver、Runtime、Memcpy、Memset、Kernel、Concurrent Kernel 和 Synchronization activity，并注册 activity buffer callbacks。

实际调用链：

```text
demo.cu initTrace
  -> muptiActivityEnable(MUPTI_ACTIVITY_KIND_DRIVER)
  -> MUPTI init
  -> inject_musa
  -> dlopen libmusa.so.1
  -> 获取 driver MUpti export table
  -> 获取 Tools export table
  -> 启用 driver hooks
  -> 导入 driver accessors
  -> dlopen libmusart.so
  -> 获取 runtime MUpti export table
  -> 启用 runtime hooks
  -> muptiActivityRegisterCallbacks(bufferRequested, bufferCompleted)
```

日志证据：

```text
178:[MUPTI_CALLFLOW] muptiActivityEnable: enter kind=DRIVER(4)
179:[MUPTI_CALLFLOW] init: first initialization, injecting MUSA hooks
180:[MUPTI_CALLFLOW] inject_musa: enter
181:[MUPTI_CALLFLOW] get_export_table_from_musa_driver: loading libmusa.so.1
182:[MUPTI_CALLFLOW] get_export_table_from_musa_driver: requesting MUpti driver export table
184:[MUPTI_CALLFLOW] get_export_table_from_musa_driver: requesting Tools export table
185:[MUPTI_CALLFLOW] get_export_table_from_musa_driver: Tools callback subscribed handle=0x1
186:[MUPTI_CALLFLOW] inject_musa: enabling driver hooks
188:[MUPTI_CALLFLOW] get_export_table_from_musa_runtime: loading libmusart.so
189:[MUPTI_CALLFLOW] get_export_table_from_musa_runtime: requesting MUpti runtime export table
191:[MUPTI_CALLFLOW] inject_musa: enabling runtime hooks
227:[MUPTI_CALLFLOW] muptiActivityRegisterCallbacks: request_cb=0x4039c0, complete_cb=0x403a20
```

结论：MUPTI 不是在用户代码每次调用后被动解析日志，而是在初始化阶段通过 driver/runtime export table 注入 hook。后续 Runtime API 和 Driver API 进入时，MUPTI 能收到 enter/exit callback，并生成 activity record。

## 7. `musaMalloc` 调用流程

用户代码：

```cpp
CHECK_MUSA_ERROR(musaMalloc(&device_x, kDataLen * sizeof(float)));
```

实际调用链：

```text
demo.cu musaMalloc
  -> MUSA-Runtime ApiInvocationGuard enter
  -> musaapiMalloc
  -> muMemAlloc_v2
  -> Driver ApiInvocationGuard enter
  -> muapiMemAlloc_v2
  -> 获取当前 Context
  -> IContext::CreateMemory
  -> Context::CreateMemory
  -> Memory::Init
  -> 返回 device pointer
  -> Driver ApiInvocationGuard exit
  -> Runtime ApiInvocationGuard exit
  -> MUPTI runtime/driver activity record
```

日志证据：

```text
175:[MUSART_CALLFLOW] wrapper enter api=musaMalloc_v3020 cbid=20 correlation=8 nested=0
176:[MUSART_CALLFLOW] musaapiMalloc enter devPtr_slot=0x7ffcd6063fa8 size=16
249:[MUSART_CALLFLOW] musaapiMalloc call driver=muMemAlloc_v2 size=16
251:[MUSA_DRIVER_CALLFLOW] muapiMemAlloc_v2 enter dptr_slot=0x7ffcd6063fa8 bytesize=16
252:[MUSA_DRIVER_CALLFLOW] muapiMemAlloc_v2 context=0x12efce0
253:[MUSA_DRIVER_CALLFLOW] muapiMemAlloc_v2 call core=IContext::CreateMemory context=0x12efce0 size=16 flags=0x4009
254:[MUSA_DRIVER_CALLFLOW] core Context::CreateMemory enter ctx=0x12efce0 memory_slot=0x7ffcd60638d8 type=0
255:[MUSA_DRIVER_CALLFLOW] core Context::CreateMemory call core=Memory::Init memory=0x28ac950
256:[MUSA_DRIVER_CALLFLOW] core Context::CreateMemory exit memory=0x28ac950 status=0(MUSA_SUCCESS)
257:[MUSA_DRIVER_CALLFLOW] muapiMemAlloc_v2 exit dptr=0x10062000000 status=0(MUSA_SUCCESS)
259:[MUSART_CALLFLOW] musaapiMalloc exit devPtr=0x10062000000 status=0(musaSuccess)
```

关键点：

```text
Runtime 层负责接收 musaMalloc，并转到 driver API。
Driver 层负责找到当前 Context，并把分配请求交给 Core Context。
Core Context 创建 Memory 对象，Memory::Init 完成实际内存对象初始化。
返回值 0x10062000000 是 device_x 的设备地址。
```

## 8. H2D `musaMemcpy` 调用流程

用户代码：

```cpp
CHECK_MUSA_ERROR(musaMemcpy(device_x, host_x, kDataLen * sizeof(float), musaMemcpyHostToDevice));
```

实际调用链：

```text
demo.cu musaMemcpy(H2D)
  -> MUSA-Runtime ApiInvocationGuard enter
  -> musaapiMemcpy
  -> Runtime 根据 kind 和指针跟踪结果判断方向
  -> muMemcpyHtoD_v2
  -> Driver ApiInvocationGuard enter
  -> muapiMemcpyHtoD_v2
  -> Context::GeneralMemcpy(sync=1)
  -> 解析默认 stream
  -> Context::CreateMemcpyNode
  -> Stream::CmdCopyMemory
  -> 创建 SyncMemcpyCommand
  -> Context::ResolveDependencyAndQueueCommand
  -> Stream::AsyncSubmit
  -> SyncMemcpyCommand::Submit
  -> CopyManager::MemcpyH2D
  -> Runtime/Driver exit
  -> MUPTI memcpy activity record
```

日志证据：

```text
316:[MUSART_CALLFLOW] wrapper enter api=musaMemcpy_v3020 cbid=31 correlation=10 nested=0
317:[MUSART_CALLFLOW] musaapiMemcpy enter dst=0x10062000000 src=0x7ffcd6063fc0 count=16 kind=musaMemcpyHostToDevice(1)
334:[MUSART_CALLFLOW] musaapiMemcpy classify dstTracked=1 srcTracked=0
336:[MUSA_DRIVER_CALLFLOW] muapiMemcpyHtoD_v2 enter dst=0x10062000000 src=0x7ffcd6063fc0 bytes=16
337:[MUSA_DRIVER_CALLFLOW] muapiMemcpyHtoD_v2 call core=Context::GeneralMemcpy context=0x12efce0 stream=(nil) sync=1 bytes=16
338:[MUSA_DRIVER_CALLFLOW] core Context::GeneralMemcpy enter ctx=0x12efce0 stream=(nil) bytes=16 height=1 depth=1 wait=1
339:[MUSA_DRIVER_CALLFLOW] core Context::GeneralMemcpy resolved stream=0x13a4280
340:[MUSA_DRIVER_CALLFLOW] core Context::GeneralMemcpy call core=Context::CreateMemcpyNode ctx=0x12efce0 wait=1
342:[MUSA_DRIVER_CALLFLOW] core Context::GeneralMemcpy call core=Stream::CmdCopyMemory stream=0x13a4280 node=0x28d3460 wait=1
343:[MUSA_DRIVER_CALLFLOW] core Stream::CmdCopyMemory enter stream=0x13a4280 graph_node=0x28d3460 blocking=1
344:[MUSA_DRIVER_CALLFLOW] core Stream::CmdCopyMemory create command=SyncMemcpyCommand command=0x28d36a0
349:[MUSA_DRIVER_CALLFLOW] core Stream::AsyncSubmit call command=Command::Submit stream=0x13a4280 command=0x28d36a0 type=3 engine=7
350:[MUSA_DRIVER_CALLFLOW] core SyncMemcpyCommand::Submit enter command=0x28d36a0 direction=2
351:[MUSA_DRIVER_CALLFLOW] core SyncMemcpyCommand::Submit call copyManager=MemcpyH2D command=0x28d36a0
352:[MUSA_DRIVER_CALLFLOW] core SyncMemcpyCommand::Submit exit command=0x28d36a0 status=0(MUSA_SUCCESS)
356:[MUSA_DRIVER_CALLFLOW] muapiMemcpyHtoD_v2 exit status=0(MUSA_SUCCESS)
```

关键点：

```text
dstTracked=1 表示目标地址是 runtime 已跟踪的设备地址。
srcTracked=0 表示源地址是 host 地址。
kind=musaMemcpyHostToDevice 使 runtime 选择 muMemcpyHtoD_v2。
sync=1 和 blocking=1 表示该 H2D 拷贝走同步 memcpy command 路径。
CopyManager::MemcpyH2D 是本次 H2D 数据搬运路径的核心执行点。
```

## 9. `musaStreamCreate` 调用流程

用户代码：

```cpp
for (int i = 0; i < 10; i++) {
    CHECK_MUSA_ERROR(musaStreamCreate(&(streams[i])));
}
```

实际调用链：

```text
demo.cu musaStreamCreate
  -> MUSA-Runtime ApiInvocationGuard enter
  -> musaapiStreamCreate
  -> muStreamCreate
  -> Driver ApiInvocationGuard enter
  -> muapiStreamCreate
  -> muapiStreamCreateWithPriority(flags=0, priority=0)
  -> 获取当前 Context
  -> IContext::CreateStream
  -> Context::CreateStream
  -> Stream::Init
  -> 返回 stream handle
  -> Runtime/Driver exit
  -> MUPTI runtime/driver activity record
```

日志证据：

```text
361:[MUSART_CALLFLOW] musaapiStreamCreate enter stream_slot=0x7ffcd6063f30
402:[MUSART_CALLFLOW] musaapiStreamCreate call driver=muStreamCreate flags=0
404:[MUSA_DRIVER_CALLFLOW] muapiStreamCreate enter stream_slot=0x7ffcd6063f30 flags=0
405:[MUSA_DRIVER_CALLFLOW] muapiStreamCreateWithPriority enter stream_slot=0x7ffcd6063f30 flags=0 priority=0
406:[MUSA_DRIVER_CALLFLOW] muapiStreamCreateWithPriority context=0x12efce0
407:[MUSA_DRIVER_CALLFLOW] muapiStreamCreateWithPriority call core=IContext::CreateStream context=0x12efce0 flags=0 priority=0
408:[MUSA_DRIVER_CALLFLOW] core Context::CreateStream enter ctx=0x12efce0 stream_slot=0x7ffcd6063898 flags=0 priority=0
409:[MUSA_DRIVER_CALLFLOW] core Context::CreateStream call core=Stream::Init stream=0x29b1150
410:[MUSA_DRIVER_CALLFLOW] core Stream::Init enter stream=0x29b1150 parent_ctx=0x12efce0
412:[MUSA_DRIVER_CALLFLOW] core Context::CreateStream exit stream=0x29b1150 status=0(MUSA_SUCCESS)
413:[MUSA_DRIVER_CALLFLOW] muapiStreamCreateWithPriority exit stream=0x29b1150 status=0(MUSA_SUCCESS)
```

关键点：

```text
用户传入的是 stream handle 地址。
Runtime 直接调用 driver 的 muStreamCreate。
Driver 内部统一转到 muStreamCreateWithPriority，默认 flags=0、priority=0。
Core Context 创建 Stream 对象，并调用 Stream::Init 完成初始化。
demo 创建 10 个 stream，后续 10 次 kernel launch 分别使用这些 stream。
```

## 10. `musaLaunchKernel` 调用流程

用户代码：

```cpp
void* args[] = { &device_x, &device_y, &a };
for (int i = 0; i < 10; i++) {
    CHECK_MUSA_ERROR(musaLaunchKernel((void*)axpy, dim3(1), dim3(kDataLen), args, 0, streams[i]));
}
```

单次 kernel launch 的实际调用链：

```text
demo.cu musaLaunchKernel
  -> MUSA-Runtime ApiInvocationGuard enter
  -> musaapiLaunchKernel
  -> muStreamGetDevice
  -> Runtime 解析 host function 对应的 device function
  -> muDeviceGetProperties
  -> muLaunchKernel
  -> Driver ApiInvocationGuard enter
  -> muapiLaunchKernel
  -> KernelReplayHandler
  -> Context::GeneralLaunchKernel
  -> 解析 stream
  -> Context::CreateKernelNode
  -> Stream::CmdLaunchKernel
  -> 创建 DispatchCommand
  -> Context::ResolveDependencyAndQueueCommand
  -> Stream::AsyncSubmit
  -> DispatchCommand::Submit
  -> Command::SubmitToQueue
  -> IQueue::Submit
  -> Runtime/Driver exit
  -> MUPTI kernel queued/submitted/activity record
```

Runtime 到 Core 的日志证据：

```text
835:[MUSART_CALLFLOW] musaapiLaunchKernel enter func=0x403430 grid={1,1,1} block={4,1,1} sharedMem=0 stream=0x29b1150 args=0x7ffcd6063f10
840:[MUSART_CALLFLOW] musaapiLaunchKernel call driver=muStreamGetDevice stream=0x29b1150
900:[MUSART_CALLFLOW] musaapiLaunchKernel call driver=muDeviceGetProperties dev=0
907:[MUSART_CALLFLOW] musaapiLaunchKernel call driver=muLaunchKernel function=0x2e92770 grid={1,1,1} block={4,1,1} sharedMem=0 stream=0x29b1150
909:[MUSA_DRIVER_CALLFLOW] muapiLaunchKernel enter function=0x2e92770 grid={1,1,1} block={4,1,1} sharedMem=0 stream=0x29b1150 kernelParams=0x7ffcd6063f10 extra=(nil)
912:[MUSA_DRIVER_CALLFLOW] muapiLaunchKernel call core=Context::GeneralLaunchKernel context=0x12efce0 stream=0x29b1150 launchBlocking=0
913:[MUSA_DRIVER_CALLFLOW] core Context::GeneralLaunchKernel enter ctx=0x12efce0 stream=0x29b1150 func=0x2e92770 grid={1,1,1} block={4,1,1} wait=0
914:[MUSA_DRIVER_CALLFLOW] core Context::GeneralLaunchKernel resolved stream=0x29b1150
915:[MUSA_DRIVER_CALLFLOW] core Context::GeneralLaunchKernel call core=Context::CreateKernelNode ctx=0x12efce0
916:[MUSA_DRIVER_CALLFLOW] core Context::GeneralLaunchKernel created kernel_node=0x2e98f90 status=0(MUSA_SUCCESS)
917:[MUSA_DRIVER_CALLFLOW] core Context::GeneralLaunchKernel call core=Stream::CmdLaunchKernel stream=0x29b1150 node=0x2e98f90 wait=0
918:[MUSA_DRIVER_CALLFLOW] core Stream::CmdLaunchKernel enter stream=0x29b1150 graph_node=0x2e98f90 blocking=0
919:[MUSA_DRIVER_CALLFLOW] core Stream::CmdLaunchKernel create command=DispatchCommand command=0x2e99150
920:[MUSA_DRIVER_CALLFLOW] core Stream::CmdLaunchKernel call core=Context::ResolveDependencyAndQueueCommand command=0x2e99150 blocking=0
```

Command 到 HAL queue 的日志证据：

```text
1549:[MUSA_DRIVER_CALLFLOW] core Stream::AsyncSubmit call command=Command::Submit stream=0x29b1150 command=0x2e99150 type=0 engine=0
1550:[MUSA_DRIVER_CALLFLOW] core DispatchCommand::Submit enter command=0x2e99150 stream=0x29b1150 engine=0
1558:[MUSA_DRIVER_CALLFLOW] core Command::SubmitToQueue enter command=0x2e99150 type=0 engine=0 queue=0x28b6d80
1569:[MUSA_DRIVER_CALLFLOW] core Command::SubmitToQueue call hal=IQueue::Submit command=0x2e99150 submission_id=33 waits=0 signals=1
```

MUPTI activity 证据：

```text
2061:CONC KERNEL "_Z4axpyPfS_f" [ ... ] device 0, context 1, stream 2, correlation 21, queued: ..., submitted: ...
2063:RUNTIME API musaLaunchKernel_v7000 [ ... ] cbid=211, process 65575, thread 65575, correlation 21, status: 0
```

关键点：

```text
Runtime 先通过 stream 获取 device，再解析 host function 到 device function。
Driver 的 muapiLaunchKernel 不直接提交硬件队列，而是进入 Context::GeneralLaunchKernel。
Core Context 创建 kernel node，Stream::CmdLaunchKernel 把 kernel node 包装为 DispatchCommand。
DispatchCommand 通过 Command::SubmitToQueue 进入 HAL IQueue::Submit。
MUPTI 使用 correlationId 关联 Runtime API activity 与 Kernel activity。
```

## 11. `musaDeviceSynchronize` 调用流程

用户代码：

```cpp
CHECK_MUSA_ERROR(musaDeviceSynchronize());
```

实际调用链：

```text
demo.cu musaDeviceSynchronize
  -> MUSA-Runtime ApiInvocationGuard enter
  -> musaapiDeviceSynchronize
  -> muCtxSynchronize
  -> Driver ApiInvocationGuard enter
  -> muapiCtxSynchronize
  -> 获取当前 Context
  -> Context::Synchronize
  -> Context::LockedWait
  -> 遍历相关 Stream
  -> Stream::WaitFinish
  -> 等待每个 stream 的 last_command 完成
  -> Runtime/Driver exit
  -> MUPTI synchronization/runtime/driver activity record
```

日志证据：

```text
1590:[MUSART_CALLFLOW] musaapiDeviceSynchronize enter
1595:[MUSART_CALLFLOW] musaapiDeviceSynchronize call driver=muCtxSynchronize
1597:[MUSA_DRIVER_CALLFLOW] muapiCtxSynchronize enter
1598:[MUSA_DRIVER_CALLFLOW] muapiCtxSynchronize context=0x12efce0
1599:[MUSA_DRIVER_CALLFLOW] muapiCtxSynchronize call core=Context::Synchronize context=0x12efce0
1600:[MUSA_DRIVER_CALLFLOW] core Context::Synchronize enter ctx=0x12efce0
1601:[MUSA_DRIVER_CALLFLOW] core Context::Synchronize call core=Context::LockedWait ctx=0x12efce0
1602:[MUSA_DRIVER_CALLFLOW] core Stream::WaitFinish enter stream=0x2ca2750
1603:[MUSA_DRIVER_CALLFLOW] core Stream::WaitFinish last_command=0x2ea1320
1736:[MUSA_DRIVER_CALLFLOW] core Context::Synchronize exit status=0(MUSA_SUCCESS)
1737:[MUSA_DRIVER_CALLFLOW] muapiCtxSynchronize exit status=0(MUSA_SUCCESS)
1739:[MUSART_CALLFLOW] musaapiDeviceSynchronize exit status=0(musaSuccess)
```

关键点：

```text
device synchronize 在 driver/core 层不是单个空等待。
它进入 Context::Synchronize，并等待多个 stream 的 last_command。
本 demo 前面创建了 10 个 stream 并提交了 10 个 kernel，因此同步阶段会看到多个 Stream::WaitFinish。
```

## 12. D2H `musaMemcpy` 调用流程

用户代码：

```cpp
CHECK_MUSA_ERROR(musaMemcpy(host_y, device_y, kDataLen * sizeof(float), musaMemcpyDeviceToHost));
```

实际调用链：

```text
demo.cu musaMemcpy(D2H)
  -> MUSA-Runtime ApiInvocationGuard enter
  -> musaapiMemcpy
  -> Runtime 根据 kind 和指针跟踪结果判断方向
  -> muMemcpyDtoH_v2
  -> Driver ApiInvocationGuard enter
  -> muapiMemcpyDtoH_v2
  -> Context::GeneralMemcpy(sync=1)
  -> 解析默认 stream
  -> Context::CreateMemcpyNode
  -> Stream::CmdCopyMemory
  -> 创建 SyncMemcpyCommand
  -> Stream::AsyncSubmit
  -> SyncMemcpyCommand::Submit
  -> CopyManager::MemcpyD2H
  -> Runtime/Driver exit
  -> MUPTI memcpy activity record
```

日志证据：

```text
1741:[MUSART_CALLFLOW] wrapper enter api=musaMemcpy_v3020 cbid=31 correlation=32 nested=0
1759:[MUSART_CALLFLOW] musaapiMemcpy classify dstTracked=0 srcTracked=1
1761:[MUSA_DRIVER_CALLFLOW] muapiMemcpyDtoH_v2 enter dst=0x7ffcd6063fb0 src=0x10062000080 bytes=16
1762:[MUSA_DRIVER_CALLFLOW] muapiMemcpyDtoH_v2 call core=Context::GeneralMemcpy context=0x12efce0 stream=(nil) sync=1 bytes=16
1763:[MUSA_DRIVER_CALLFLOW] core Context::GeneralMemcpy enter ctx=0x12efce0 stream=(nil) bytes=16 height=1 depth=1 wait=1
1767:[MUSA_DRIVER_CALLFLOW] core Context::GeneralMemcpy call core=Stream::CmdCopyMemory stream=0x13a4280 node=0x2ea2990 wait=1
1769:[MUSA_DRIVER_CALLFLOW] core Stream::CmdCopyMemory create command=SyncMemcpyCommand command=0x2ea2bd0
1774:[MUSA_DRIVER_CALLFLOW] core Stream::AsyncSubmit call command=Command::Submit stream=0x13a4280 command=0x2ea2bd0 type=3 engine=7
1776:[MUSA_DRIVER_CALLFLOW] core SyncMemcpyCommand::Submit call copyManager=MemcpyD2H command=0x2ea2bd0
1781:[MUSA_DRIVER_CALLFLOW] muapiMemcpyDtoH_v2 exit status=0(MUSA_SUCCESS)
```

关键点：

```text
dstTracked=0 表示目标是 host 地址。
srcTracked=1 表示源地址是设备地址。
Runtime 选择 muMemcpyDtoH_v2。
D2H 和 H2D 共享 Context::GeneralMemcpy、Stream::CmdCopyMemory、SyncMemcpyCommand 路径。
最终数据搬运方向由 CopyManager::MemcpyD2H 决定。
```

## 13. `musaFree` 调用流程

用户代码：

```cpp
CHECK_MUSA_ERROR(musaFree(device_x));
CHECK_MUSA_ERROR(musaFree(device_y));
```

实际调用链：

```text
demo.cu musaFree
  -> MUSA-Runtime ApiInvocationGuard enter
  -> musaapiFree
  -> muMemFree_v2
  -> Driver ApiInvocationGuard enter
  -> muapiMemFree_v2
  -> Platform::GetMemoryByDevicePointer
  -> 等待相关 stream 完成
  -> IContext::DestroyMemory
  -> Context::DestroyMemory
  -> Runtime/Driver exit
  -> MUPTI runtime/driver activity record
```

日志证据：

```text
1786:[MUSART_CALLFLOW] musaapiFree enter devPtr=0x10062000000
1831:[MUSART_CALLFLOW] musaapiFree call driver=muMemFree_v2 devPtr=0x10062000000
1833:[MUSA_DRIVER_CALLFLOW] muapiMemFree_v2 enter dptr=0x10062000000
1834:[MUSA_DRIVER_CALLFLOW] muapiMemFree_v2 call core=Platform::GetMemoryByDevicePointer dptr=0x10062000000
1835:[MUSA_DRIVER_CALLFLOW] core Stream::WaitFinish enter stream=0x2ca2750
1871:[MUSA_DRIVER_CALLFLOW] muapiMemFree_v2 call core=IContext::DestroyMemory memory=0x28ac950
1872:[MUSA_DRIVER_CALLFLOW] core Context::DestroyMemory enter memory=0x28ac950
1873:[MUSA_DRIVER_CALLFLOW] core Context::DestroyMemory exit status=0(MUSA_SUCCESS)
1874:[MUSA_DRIVER_CALLFLOW] muapiMemFree_v2 exit status=0(MUSA_SUCCESS)
1876:[MUSART_CALLFLOW] musaapiFree exit devPtr=0x10062000000 status=0(musaSuccess)
```

关键点：

```text
释放设备内存前，driver/core 会定位 device pointer 对应的 Memory 对象。
释放前会等待相关 stream 的未完成 command。
内存对象销毁发生在 Context::DestroyMemory。
```

## 14. `musaDeviceReset` 调用流程

用户代码：

```cpp
CHECK_MUSA_ERROR(musaDeviceReset());
```

实际调用链：

```text
demo.cu musaDeviceReset
  -> MUSA-Runtime ApiInvocationGuard enter
  -> musaapiDeviceReset
  -> muCtxGetDevice
  -> muDevicePrimaryCtxRetain
  -> muDevicePrimaryCtxReset_v2
  -> Driver ApiInvocationGuard enter
  -> muapiDevicePrimaryCtxReset_v2
  -> IContext::Reset
  -> 等待 stream 完成
  -> 释放或重置 primary context 关联资源
  -> Runtime/Driver exit
  -> MUPTI runtime/driver activity record
```

日志证据：

```text
1971:[MUSART_CALLFLOW] musaapiDeviceReset enter
1976:[MUSART_CALLFLOW] musaapiDeviceReset call driver=muCtxGetDevice
1982:[MUSART_CALLFLOW] musaapiDeviceReset call driver=muDevicePrimaryCtxRetain dev=0
1988:[MUSART_CALLFLOW] musaapiDeviceReset call driver=muDevicePrimaryCtxReset_v2 dev=0
1990:[MUSA_DRIVER_CALLFLOW] muapiDevicePrimaryCtxReset_v2 enter dev=0
1991:[MUSA_DRIVER_CALLFLOW] muapiDevicePrimaryCtxReset_v2 call core=IContext::Reset primary_ctx=0x12efce0
1992:[MUSA_DRIVER_CALLFLOW] core Stream::WaitFinish enter stream=0x2ca2750
```

关键点：

```text
musaDeviceReset 不是只清理 runtime 侧状态。
Runtime 会先确认 device 和 primary context，然后调用 driver 的 primary context reset。
Driver/Core reset 前会等待 stream 上已有 command 完成。
```

## 15. MUPTI Activity 输出与关联方式

MUPTI 的 activity 输出来自 `bufferCompleted` 回调。demo 中周期性 flush 线程会调用：

```cpp
muptiActivityFlushAll(0);
```

结束时调用：

```cpp
muptiActivityFlushAll(1);
```

日志证据：

```text
2296:[MUPTI_CALLFLOW] muptiActivityFlushAll: enter flag=1, force=1
2306:[MUPTI_CALLFLOW] muptiActivityFlushAll: done result=success
```

Kernel activity 与 Runtime API 的关联通过 `correlationId` 完成。例如同一次 launch：

```text
CONC KERNEL "_Z4axpyPfS_f" ... correlation 21, queued: ..., submitted: ...
RUNTIME API musaLaunchKernel_v7000 ... correlation 21, status: 0
```

含义：

```text
Runtime wrapper 分配 correlationId。
MUPTI enter/exit callback 记录 API activity。
Core/driver 在 kernel queued/submitted 时更新 kernel activity 的时间点。
flush 时 activity buffer 输出 runtime API、driver API、memcpy、kernel、sync 等记录。
```

## 16. 本次已验证路径

已通过日志和 demo 输出验证：

```text
MUPTI activity enable
MUPTI driver/runtime hook 注入
MUPTI activity buffer callback 注册和 flush
musaMalloc -> muMemAlloc_v2 -> Context::CreateMemory -> Memory::Init
musaMemcpy H2D -> muMemcpyHtoD_v2 -> Context::GeneralMemcpy -> SyncMemcpyCommand -> CopyManager::MemcpyH2D
musaStreamCreate -> muStreamCreateWithPriority -> Context::CreateStream -> Stream::Init
musaLaunchKernel -> muLaunchKernel -> Context::GeneralLaunchKernel -> Stream::CmdLaunchKernel -> DispatchCommand -> IQueue::Submit
musaDeviceSynchronize -> muCtxSynchronize -> Context::Synchronize -> Stream::WaitFinish
musaMemcpy D2H -> muMemcpyDtoH_v2 -> SyncMemcpyCommand -> CopyManager::MemcpyD2H
musaFree -> muMemFree_v2 -> Context::DestroyMemory
musaDeviceReset -> muDevicePrimaryCtxReset_v2 -> IContext::Reset
```

本次验证结果：

```text
demo 返回码：0
日志行数：2360
功能输出：2, 4, 6, 8
```

## 17. 已插桩但本次 demo 未覆盖的路径

以下路径已经增加日志，但 `demo.cu` 没有直接触发：

```text
muMemcpyHtoDAsync_v2
muMemcpyDtoHAsync_v2
muMemcpyAsync
AsyncMemcpyCommand::Submit
muMemcpyDtoD_v2
muMemcpy 通用路径
memset activity 的实际 memset record
显式 musaStreamSynchronize 路径
```

后续如需覆盖这些路径，应增加独立 demo：

```text
musaMemcpyAsync H2D/D2H
musaMemcpy DeviceToDevice
musaMemset
musaStreamSynchronize
多 stream + event wait
graph capture/replay
```

## 18. 结论

本次日志验证已经把 `demo.cu` 的主要执行路径从用户态 API 追踪到 Runtime、Driver、Core、Command 和 HAL queue submit。MUPTI hook 注入、Runtime API activity、Driver API activity、memcpy activity、kernel activity 和 synchronization activity 均有日志证据。

当前文档中的每条链路均基于 `/tmp/mupti_demo_full_callflow_rerun.log` 的实际输出。异步提交和周期性 flush 会造成日志交织，但不影响按 API correlation、stream、command 和 activity record 还原调用流程。
