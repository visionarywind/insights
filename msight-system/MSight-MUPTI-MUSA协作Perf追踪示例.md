# MSight System / MUPTI / musa / MUSA-Runtime 协作追踪示例

分析范围：

```text
msight-system
MUPTI
musa
MUSA-Runtime
```

本文用一个最小 workload 说明四个仓库如何协作完成 perf 追踪和分析。示例中的时间戳、correlation id、submission id 是说明用样例值，不代表真实运行结果。

## 示例 workload

示例程序执行四类操作：

```cpp
// 示例代码只用于说明追踪链路，省略错误检查。
__global__ void add_one(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += 1.0f;
    }
}

int main() {
    const int n = 1024;
    const size_t bytes = n * sizeof(float);

    musaStream_t stream;
    float *h = nullptr;
    float *d = nullptr;

    h = static_cast<float *>(malloc(bytes));
    musaStreamCreate(&stream);
    musaMalloc(&d, bytes);

    // 1. Host 到 Device 拷贝。
    musaMemcpyAsync(d, h, bytes, musaMemcpyHostToDevice, stream);

    // 2. 启动一个 kernel。
    add_one<<<4, 256, 0, stream>>>(d, n);

    // 3. 等待 stream 上前面的 memcpy 和 kernel 完成。
    musaStreamSynchronize(stream);

    // 4. 释放资源。
    musaFree(d);
    musaStreamDestroy(stream);
    free(h);
}
```

这个例子覆盖：

```text
Runtime API
Driver API
Memcpy activity
Kernel activity
Stream synchronize activity
Command / submission / kernel 关联
MSight report 落盘和分析
```

## 四个仓库的分工

| 仓库 | 在本例中的职责 |
| --- | --- |
| `MUSA-Runtime` | 接收应用调用的 `musaMemcpyAsync`、kernel launch、`musaStreamSynchronize` 等 Runtime API，并转发到 Driver。 |
| `musa` | 执行 Driver API，完成 stream、memory、module/function、command、submission、sync 等内部逻辑。 |
| `MUPTI` | 订阅 Runtime/Driver API callback，启用 memcpy/kernel/sync activity，收集 correlation 和 submission 关系。 |
| `msight-system` | 启动目标程序，注入 `mupti-injection`，接收 MUPTI 数据，写入 `.msys-rep`，再通过 `stats/analyze/GUI` 分析。 |

## 启动追踪

用户执行：

```bash
msys profile --trace=musa --output add_one.msys-rep ./add_one
```

`msight-system` 先做三件事。

第一，CLI 解析命令：

```text
msys_cli/ms_cmd_profile_handler.cpp
  -> 解析 trace=musa
  -> 构造 MSCaptureConfig
  -> 构造 MSProcessConfig
  -> 构造 MSTraceOptionConfig
```

第二，controller 准备采集：

```text
MSProfilingController::setCaptureInfo()
  -> 创建临时 report
  -> 创建 MSCaptureController
  -> 创建 MSMuptiCaptureComponent
```

第三，controller 准备目标进程环境：

```text
MSProfilingController::setProcessInfo()
  -> ConfigEnvHandler::getEnvironments()
  -> 设置共享内存 key
  -> 设置 MUPTI 相关环境变量
  -> 设置 mupti-injection 路径
```

关键环境变量包括：

```text
MSYS_HOME_DIR
INJECTION_LOG_DIR
MUPTI_FLUSH_PERIOD
MUPTI_GRAPH_MODE
共享内存 key
```

目标进程启动后，`mupti-injection` 被加载进目标进程。

## MUPTI 注入初始化

目标进程内的执行流程：

```text
mupti-injection
  -> MsysInjectionHandler::onInit()
  -> GlobalData::injectMUpti()
  -> MUptiApi::init()
  -> 动态加载 MUPTI 符号
  -> muptiActivityRegisterCallbacks()
  -> enableActivities()
  -> startPeriodFlush()
```

本例至少需要启用这些 activity：

```text
MUPTI_ACTIVITY_KIND_RUNTIME
MUPTI_ACTIVITY_KIND_DRIVER
MUPTI_ACTIVITY_KIND_MEMCPY
MUPTI_ACTIVITY_KIND_KERNEL
MUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL
MUPTI_ACTIVITY_KIND_SYNCHRONIZATION
```

同时，MUPTI 通过 Driver 暴露的 Tools callback 和 hook 表接入 Runtime/Driver：

```text
Tools Callback:
  Runtime API enter/exit
  Driver API enter/exit

Driver Hook / Tracepoint:
  memcpy
  kernel
  command begin/end
  submission relation
  stream synchronize
```

## 事件链路示例

下面按程序执行顺序说明事件如何产生。

### 1. `musaMemcpyAsync`

应用调用：

```cpp
musaMemcpyAsync(d, h, bytes, musaMemcpyHostToDevice, stream);
```

典型调用链：

```text
应用
  -> MUSA-Runtime: musaMemcpyAsync
  -> musa Driver: muMemcpyHtoDAsync_v2 或等价 Driver API
  -> Driver memory path
  -> Driver stream path
  -> 创建 memcpy command
  -> command 入队
  -> submission
  -> GPU copy engine 执行 memcpy
```

追踪事件示例：

| 时间 ns | 事件 | 来源仓库 | 关键字段 |
| --- | --- | --- | --- |
| 1000 | Runtime API enter: `musaMemcpyAsync` | `MUSA-Runtime` + `MUPTI` | `correlation_id=101` |
| 1080 | Driver API enter: `muMemcpyHtoDAsync_v2` | `musa` + `MUPTI` | `correlation_id=101` |
| 1160 | Memcpy command created | `musa` hook | `stream_id=7` |
| 1220 | Submission relation | `musa` hook | `correlation_id=101, submission_id=9001` |
| 1280 | Driver API exit | `musa` + `MUPTI` | `return=SUCCESS` |
| 1300 | Runtime API exit | `MUSA-Runtime` + `MUPTI` | `return=SUCCESS` |
| 1500-2100 | GPU memcpy activity: `Memcpy HtoD` | `musa` + `MUPTI` | `bytes=4096, stream_id=7, submission_id=9001` |

这一段说明三类耗时：

```text
Runtime API 自身耗时：1300 - 1000 = 300 ns
Driver API 自身耗时：1280 - 1080 = 200 ns
GPU memcpy 执行耗时：2100 - 1500 = 600 ns
```

分析时不能把 `musaMemcpyAsync` 的 CPU API 时间直接当作数据拷贝时间。异步 API 返回时，GPU memcpy 可能还没有完成。

### 2. Kernel launch

应用调用：

```cpp
add_one<<<4, 256, 0, stream>>>(d, n);
```

典型调用链：

```text
应用 kernel launch 语法
  -> MUSA-Runtime: kernel launch wrapper
  -> musa Driver: muLaunchKernel 或等价 Driver API
  -> 查找 module/function
  -> 创建 DispatchCommand
  -> 写 command buffer
  -> command 入队
  -> submission
  -> GPU compute engine 执行 kernel
```

追踪事件示例：

| 时间 ns | 事件 | 来源仓库 | 关键字段 |
| --- | --- | --- | --- |
| 2200 | Runtime API enter: kernel launch | `MUSA-Runtime` + `MUPTI` | `correlation_id=102` |
| 2280 | Driver API enter: `muLaunchKernel` | `musa` + `MUPTI` | `correlation_id=102` |
| 2360 | RegisterKernel / RegisterKernelV2 | `musa` hook | `kernel_name=add_one, correlation_id=102` |
| 2440 | MarkKernelQueued | `musa` hook | `stream_id=7` |
| 2500 | AssignKernelToKick | `musa` hook | `correlation_id=102, job_ref=5001` |
| 2550 | AssignSubmissionToCorrelation | `musa` hook | `correlation_id=102, submission_id=9002` |
| 2600 | Driver API exit | `musa` + `MUPTI` | `return=SUCCESS` |
| 2620 | Runtime API exit | `MUSA-Runtime` + `MUPTI` | `return=SUCCESS` |
| 3000-5200 | GPU kernel activity: `add_one` | `musa` + `MUPTI` | `grid=4, block=256, stream_id=7, submission_id=9002` |

这一段可以拆出：

```text
kernel launch CPU 成本：2620 - 2200 = 420 ns
Driver launch 成本：2600 - 2280 = 320 ns
queue 到执行等待：3000 - 2440 = 560 ns
kernel GPU 执行时间：5200 - 3000 = 2200 ns
```

如果目标是分析 kernel launch 开销，要看 Runtime/Driver API 时间和 command/submission 事件；如果目标是分析 kernel 本身性能，要看 GPU kernel activity。

### 3. `musaStreamSynchronize`

应用调用：

```cpp
musaStreamSynchronize(stream);
```

典型调用链：

```text
应用
  -> MUSA-Runtime: musaStreamSynchronize
  -> musa Driver: muStreamSynchronize
  -> stream lookup
  -> 等待 stream 上已经提交的 memcpy/kernel 完成
  -> 返回
```

追踪事件示例：

| 时间 ns | 事件 | 来源仓库 | 关键字段 |
| --- | --- | --- | --- |
| 2700 | Runtime API enter: `musaStreamSynchronize` | `MUSA-Runtime` + `MUPTI` | `correlation_id=103` |
| 2760 | Driver API enter: `muStreamSynchronize` | `musa` + `MUPTI` | `correlation_id=103` |
| 2800 | RegisterStreamSynchronize | `musa` hook | `stream_id=7` |
| 2820 | StartStreamSynchronize | `musa` hook | `stream_id=7` |
| 5200 | StopStreamSynchronize | `musa` hook | `stream_id=7` |
| 5240 | Driver API exit | `musa` + `MUPTI` | `return=SUCCESS` |
| 5260 | Runtime API exit | `MUSA-Runtime` + `MUPTI` | `return=SUCCESS` |

这一段说明：

```text
Stream synchronize API 总耗时：5260 - 2700 = 2560 ns
真正等待 GPU 的时间：5200 - 2820 = 2380 ns
```

这里的同步耗时主要不是 CPU 计算，而是在等待前面的 HtoD memcpy 和 kernel 完成。

`msys analyze -r musa_api_sync` 会重点识别这类同步 API。

## MUPTI 如何把事件送给 MSight

MUPTI activity buffer 完成后，`mupti-injection` 会遍历 activity record：

```text
bufferCompleted()
  -> muptiActivityGetNextRecord()
  -> printActivity(record)
  -> 转成 MuptiKernelData / MuptiMemcpyData / MuptiApiData / MuptiSyncData
  -> GlobalData::sendEvent()
  -> 共享内存
```

示例中的几类记录会被编码为：

| 事件 | MSight 传输结构 |
| --- | --- |
| Runtime API enter/exit 聚合结果 | `MuptiApiData`，type=`RUNTIME` |
| Driver API enter/exit 聚合结果 | `MuptiApiData`，type=`DRIVER` |
| HtoD memcpy | `MuptiMemcpyData`，type=`MEMCPY` |
| kernel 执行 | `MuptiKernelData`，type=`KERNEL` |
| stream synchronize | `MuptiSyncData`，type=`SYNCHRONIZE` |

每条传输数据都有统一头：

```text
MuptiTransferHeader:
  magic = 0xfaf345b
  type
  len
```

## MSight 如何写入报告

`msight-system` 主进程中的接收流程：

```text
MSMuptiCaptureComponent
  -> processMuptiData()
  -> 写入 mupti_temp_event
  -> triggerStop()
  -> convertEvent()
  -> 转成 MSPerfEvent
  -> MSReportWriter::addPerfEvents()
  -> MSReportWriter::saveEvent()
  -> 写入 .msys-rep 的 all_events 表
```

`.msys-rep` 中至少能看到这些信息：

```text
event name
event type
start timestamp
end timestamp
pid
tid
gpu_id
stream/context/device 相关信息
start_data / end_data
```

同时，`MSReportWriter` 会保存：

```text
event_names
event_version
process_info
target_info
cpu_info
gpu_info
stdout/stderr/log
```

## 最终分析如何产生

采集完成后，可以执行：

```bash
msys stats -r musa_api_sum add_one.msys-rep
msys stats -r musa_gpu_kern_sum add_one.msys-rep
msys analyze -r musa_api_sync add_one.msys-rep
msys analyze -r musa_gpu_gaps add_one.msys-rep
```

示例分析结果的含义：

| 命令 | 读取的数据 | 能回答的问题 |
| --- | --- | --- |
| `musa_api_sum` | Runtime API + Driver API event | 哪些 API CPU 侧累计耗时最高。 |
| `musa_gpu_kern_sum` | Kernel activity event | 哪些 kernel GPU 执行耗时最高。 |
| `musa_api_sync` | Runtime sync API event | 是否存在 `musaStreamSynchronize` / `musaDeviceSynchronize` 等主机阻塞。 |
| `musa_gpu_gaps` | GPU hardware activity event | GPU 是否存在空闲窗口，空闲前后分别是什么事件。 |

对于本例，可能得到以下判断：

```text
1. musaMemcpyAsync 的 API 时间很短，但 HtoD memcpy activity 在 GPU 上持续了一段时间。
2. kernel launch API 很快返回，kernel 真正执行发生在后续 GPU timeline。
3. musaStreamSynchronize 的耗时较长，因为它等待前面的 memcpy 和 kernel 完成。
4. 如果 memcpy 结束到 kernel 开始之间有明显空洞，需要继续检查 stream 依赖、command enqueue、submission 或 CPU 侧调度。
```

## 一条完整关联链

本例中 kernel launch 的完整关联链可以表示为：

```text
Runtime API: kernel launch
  correlation_id = 102

Driver API: muLaunchKernel
  correlation_id = 102

Driver command:
  command_type = DispatchCommand
  stream_id = 7
  correlation_id = 102

Submission:
  submission_id = 9002
  job_ref = 5001
  correlation_id = 102

GPU activity:
  kernel_name = add_one
  stream_id = 7
  submission_id = 9002
  start = 3000
  end = 5200
```

这条链说明四个仓库如何协作：

```text
MUSA-Runtime 产生 Runtime API 视角
musa 产生 Driver / command / submission / GPU activity 视角
MUPTI 把这些视角按 correlation_id 和 submission_id 串起来
msight-system 把结果保存为 report 并做统计、专家分析和 GUI 展示
```

## 对白盒性能建模的启发

这个例子也说明了当前 tracing 的边界。

已经能看到：

```text
Runtime API 总耗时
Driver API 总耗时
Memcpy / kernel / sync activity
API -> submission -> kernel 关系
```

还不能完全解释：

```text
Runtime wrapper 内部具体阶段耗时
Driver 参数检查耗时
stream lookup 耗时
memory pool lookup 耗时
command build 耗时
command merge 决策
queue wait 原因
ioctl / doorbell 成本
```

因此，如果要做真正的软件性能建模，需要在 `MUSA-Runtime` 和 `musa` 内部继续增加阶段级埋点，并通过 MUPTI 或独立 collector 输出。MSight/MUPTI 的现有链路负责把这些事件统一采集、关联和分析。

## 对应源码索引

| 环节 | 关键源码 |
| --- | --- |
| CLI profile 解析 | `msight-system/msys_cli/ms_cmd_profile_handler.cpp` |
| profile 调度 | `msight-system/msys_cli/ms_profiling_timer_ctrl.cpp` |
| controller 总控 | `msight-system/sub_module/controller/ms_profiling_controller.cpp` |
| 目标进程启动 | `msight-system/sub_module/controller/ms_process_controller.cpp` |
| capture component 管理 | `msight-system/sub_module/controller/ms_capture_controller.cpp` |
| MUPTI 主进程接收 | `msight-system/sub_module/capture_components/linux/ms_mupti_capture_component.cpp` |
| report 写入 | `msight-system/sub_module/controller/ms_report_writer.cpp` |
| MUPTI 注入入口 | `msight-system/sub_module/injection/mupti_injection.cpp` |
| MUPTI activity 转换 | `msight-system/sub_module/injection/mupti_injection_global_data.cpp` |
| MUPTI 数据结构 | `msight-system/sub_module/injection/mupti_data_def.h` |
| Runtime MUPTI hook | `MUSA-Runtime/src/mupti/hooks.h`、`MUSA-Runtime/src/mupti/hooks.cpp` |
| Runtime API wrapper | `MUSA-Runtime/src/internal.h`、`MUSA-Runtime/src/musa_entry.cpp` |
| Driver API wrapper | `musa/src/driver/mu_wrappers_generated.cpp` |
| Driver callback | `musa/src/driver/callback.cpp` |
| Driver MUPTI hook | `musa/src/driver/mupti/hooks.h`、`musa/src/driver/mupti/hooks.cpp` |
| Driver tracepoint | `musa/src/driver/mupti/tracepoints.h` |
| Driver export table | `musa/src/driver/mu_entry.cpp`、`musa/src/musa_shared_include/export_table.h` |

