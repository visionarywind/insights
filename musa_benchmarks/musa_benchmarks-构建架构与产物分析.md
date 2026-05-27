# musa_benchmarks 构建架构与产物分析

## 结论

`musa_benchmarks` 是一个 CMake 管理的 benchmark 集合，不是单一库工程。它的构建结果由三类产物组成：

1. 公共静态库：`benchmark_common`。
2. 多个独立 benchmark 可执行文件：每个 case 通常对应一个 executable。
3. ELF/PTX/object 辅助文件：用于 kernel launch、module load、graph、hotpot kernel 等测试。

最终使用方式不是运行一个统一程序，而是运行一组可执行文件，并通过脚本批量收集 CSV、计算评分和对比 baseline。

## 目录结构

```text
musa_benchmarks/
  CMakeLists.txt
  install.sh
  common/
  memory/
  schedule/
  multicards/
  resource/
  musaOnly/
  hotPotKernels/
  scripts/
  baseline/
```

| 目录 | 作用 |
|---|---|
| `common/` | 公共 benchmark 框架、计时、打印、UDM 统计、结果表 |
| `memory/` | malloc/free、memcpy、memset、host register、GPU read/write 等 memory benchmark |
| `schedule/` | kernel launch、graph、sync、stream concurrency、dependency 等调度 benchmark |
| `multicards/` | 多卡 P2P copy、多卡 kernel launch 等 benchmark |
| `resource/` | stream/event 创建、销毁、管理开销 |
| `musaOnly/` | MUSA 专属 benchmark，只在 MCC 模式下构建 |
| `hotPotKernels/` | MUSA 专属手写汇编或热点 kernel，只在 MCC 模式下构建 |
| `scripts/` | 批量运行、评分、可视化、CUDA/MUSA 转换脚本 |
| `baseline/` | CUDA、ROCm、MUSA 平台的历史基线 CSV |

## 构建入口

构建入口是：

```bash
cd musa_benchmarks
./install.sh -m
```

常用参数：

| 参数 | 含义 |
|---|---|
| `-m` | 启用 MUSA MCC 编译 |
| `-n` | 启用 CUDA NVCC 编译 |
| `-R` | 删除 `build/` 内容后重新 cmake 和 make |
| `-j N` | 设置 make 并行数，默认 12 |
| `-d` | Debug 构建 |
| `-h` | 打印帮助 |

需要注意：如果 `build/` 已经存在，直接切换 `-m` 或 `-n` 不会重新执行 cmake 配置。切换编译模式时应使用：

```bash
./install.sh -R -m
./install.sh -R -n
```

## 顶层 CMake 架构

顶层 `CMakeLists.txt` 负责：

1. 设置 C++17。
2. 定义 `ENABLE_MCC`、`ENABLE_NVCC`、`ENABLE_DEBUG`。
3. 选择 MUSA、CUDA 或默认构建模式。
4. 检测 MUSA GPU 架构并设置 `COMPILE_ARCH`。
5. 添加各 benchmark 子目录。

子目录加载顺序：

```cmake
add_subdirectory(multicards)
add_subdirectory(schedule)
add_subdirectory(memory)
add_subdirectory(common)
add_subdirectory(resource)
add_subdirectory(hotPotKernels)

if (ENABLE_MCC)
    add_subdirectory(musaOnly)
endif()
```

这里 `common` 虽然后添加，但 CMake 允许后续解析目标依赖。各 benchmark executable 链接 `benchmark_common`。

## 构建模式

| 模式 | 开关 | 编译器 | 编译内容 | 链接库 |
|---|---|---|---|---|
| 默认模式 | 无 | g++ | 只编译 `.cpp` case | `-lmusa -lmusart -lpthread -lcpuid` |
| MUSA 模式 | `-m` / `ENABLE_MCC=ON` | `/usr/local/musa/bin/clang++` | `.cpp` + `.cu`，`.cu` 按 MUSA 编译 | `-lmusa -lmusart -lpthread -lcpuid` |
| CUDA 模式 | `-n` / `ENABLE_NVCC=ON` | `nvcc` | `.cpp` + `.cu`，按 CUDA 编译 | `-lcuda -lcudart -lpthread -lcpuid` |

MUSA 模式下，case 的编译参数一般为：

```text
-x musa -mtgpu --cuda-gpu-arch=<mp_xx>
```

CUDA 模式下，CMake 设置：

```text
CMAKE_CUDA_ARCHITECTURES = 70 75 80 86
```

## MUSA 架构检测

MUSA 模式下，如果环境变量 `MTGPU_ARCH` 未定义，顶层 CMake 会执行：

```bash
lspci -n | grep 1ed5
```

然后根据 PCI device id 映射到 MUSA 架构：

| PCI ID 示例 | 架构 |
|---|---|
| `0111`、`0106`、`0105` 等 | `mp_10` |
| `0201`、`0200`、`0222` 等 | `mp_21` |
| `0301`、`0300` 等 | `mp_22` |
| `0400` | `mp_31` |
| `0500` | `mp_32` |
| `0600` | `mp_41` |
| `0610` | `mp_42` |
| `0700` | `mp_43` |

如果检测成功，后续编译会使用：

```text
--cuda-gpu-arch=${COMPILE_ARCH}
```

也可以手动设置：

```bash
export MTGPU_ARCH=mp_31
./install.sh -R -m
```

## 公共库产物

`common/CMakeLists.txt` 将 `common/src/*.cpp` 和 `common/include/*.h` 编译成静态库：

```text
build/common/libbenchmark_common.a
```

这个库提供：

| 模块 | 作用 |
|---|---|
| Celero framework | benchmark 注册、运行、参数循环 |
| TestFixture | case 生命周期接口 |
| Timer / CPerfCounter | CPU/GPU 时间统计辅助 |
| UserDefinedMeasurement | 自定义指标统计 |
| ResultTable | CSV 输出 |
| Print / Console | 控制台表格输出 |
| BasicInfos | 环境信息输出 |

所有 benchmark 可执行文件都会链接该静态库。

## benchmark 可执行文件产物

`memory/`、`schedule/`、`multicards/`、`resource/`、`musaOnly/` 的 CMake 逻辑基本一致：

```cmake
file(GLOB_RECURSE SOURCES *.cpp)
file(GLOB_RECURSE SOURCES_CU *.cu)

if(ENABLE_MCC OR ENABLE_NVCC)
    list(APPEND SOURCES ${SOURCES_CU})
endif()

foreach(case_path ${SOURCES})
    string(REGEX REPLACE ".+/(.+)\\..*" "\\1" test_case ${case_path})
    add_executable(${test_case} ${case_path})
    target_link_libraries(${test_case} benchmark_common ...)
endforeach()
```

含义：

1. 每个 `.cpp` 或 `.cu` case 文件生成一个同名 executable。
2. 默认模式只编译 `.cpp`。
3. MUSA/CUDA 模式会额外编译 `.cu`。
4. 每个 executable 都会注册为 CTest case。

示例产物：

```text
build/memory/Copy1DAlignedRate
build/memory/MallocRateFrom1Bto4GB
build/schedule/kernelLaunchLatencyMode
build/schedule/efficiencyOfSync
build/schedule/efficiencyOfGraph
build/multicards/CopyP2PRate
build/resource/eventManage
```

MUSA 模式额外产物：

```text
build/musaOnly/kernelLaunchApiLatency
build/musaOnly/kernelLaunchThroughputModeWithoutMerge
```

如果当前架构是 `mp_31`，`musaOnly/ph1/*.cu` 也会加入构建。

## schedule ELF/PTX 辅助文件

`schedule/CMakeLists.txt` 会复制预编译 kernel 文件到 `build/schedule/`。

MUSA/default 模式复制：

```text
EmptyKernel.elf
CopyKernel.elf
VectorAdd.elf
cdm_parallelism.elf
basic_funcs.elf
basic_funcs_template.elf
```

CUDA 模式复制：

```text
EmptyKernel.ptx
CopyKernel.ptx
VectorAdd.ptx
cdm_parallelism.ptx
basic_funcs.ptx
basic_funcs_template.ptx
```

这些文件用于：

| 文件 | 用途 |
|---|---|
| `EmptyKernel` | kernel launch latency / throughput |
| `CopyKernel` | copy kernel 或调度测试 |
| `VectorAdd` | 基础计算和 launch 测试 |
| `cdm_parallelism` | CDM 并发测试 |
| `basic_funcs` | module load、function lookup、graph 等测试 |

`multicards` 和 `musaOnly` 也会复制 `EmptyKernel.elf/ptx`，用于多卡或 MUSA 专属 launch 测试。

## musaOnly bigModule 产物

`musaOnly/bigModule` 是 `kernelLaunchApiLatency` 的特殊依赖，不是普通 benchmark case。

构建流程：

1. 执行 `generateKernels.py` 生成大量 kernel 源文件。
2. 用 `/usr/local/musa/bin/mcc` 分别生成 device object 和 host object。
3. 用 `lld` 将 device object 链接成统一 `device.o`。
4. `kernelLaunchApiLatency` 依赖这些 object，测试大模块加载和函数查找开销。

主要产物：

```text
build/musaOnly/bigModule/elf_path/*.o
build/musaOnly/bigModule/elf_path/*_host.o
build/musaOnly/bigModule/elf_path/device.o
```

对应性能含义：

| 产物 | 用途 |
|---|---|
| `*_host.o` | host 侧符号和 wrapper |
| `device.o` | 多 kernel device binary |
| `kernelLaunchApiLatency` | 测量 lazy module load、first function get、steady-state launch |

## hotPotKernels 产物

`hotPotKernels` 只在 MUSA 模式下启用。

构建流程：

1. 执行 `asms/build_asm.sh`。
2. 执行 `link_asms.sh`。
3. 搜索当前目录下的 `.o` 文件。
4. 每个 `.o` 生成一个 executable。

链接库：

```text
benchmark_common
libmusa
libmusart
pthread
cpuid
openblas
```

这部分用于手写汇编或热点 kernel 性能测试，和常规 Runtime/Driver API 测试不同。

## 批量运行产物

单个 case 可直接执行：

```bash
./build/memory/Copy1DAlignedRate -t result/memoryOps/Copy1DAlignedRate.csv
```

常用参数：

| 参数 | 作用 |
|---|---|
| `-t <csv>` | 将结果写入 CSV |
| `-b` | 输出基础环境信息 |
| `-l` | 列出可用 benchmark group |
| `-g <group>` | 只运行指定 group |
| `-h` | 帮助 |

批量运行脚本：

```bash
cd musa_benchmarks/scripts
python3 autorun.py --suits memoryOp graphAndSchedule mulStreams
```

脚本会在以下目录查找 executable：

```text
build/memory
build/schedule
build/resource
build/multicards
build/musaOnly
```

运行结果写入：

```text
result/memoryOps/*.csv
result/graphAndSchedule/*.csv
result/mulStreams/*.csv
result/resource/*.csv
result/multicards/*.csv
result/musaOnly/*.csv
```

## 评分产物

评分配置来自：

```text
TestSuitConfig.json
```

当前配置覆盖：

| suite | baseline | baseScore |
|---|---|---|
| `graphAndSchedule` | `cudaA100` | 1000 |
| `memoryOps` | `cudaA100` | 1000 |
| `mulStreams` | `cudaA100` | 1000 |
| `musaOnly` | `musaPh1` | 1000 |

评分脚本：

```bash
cd musa_benchmarks/scripts
python3 calculateScoreOfSuit.py
```

输入：

```text
baseline/<platform>/<suite>/<case>.csv
result/<suite>/<case>.csv
TestSuitConfig.json
```

输出：

```text
score/<suite>/*.csv
score/<suite>.csv
score/totalScore.csv
Histogram.svg
```

评分脚本只使用 CSV 表头中以 `*` 开头的列作为 focus column。公式为：

```text
score_ratio = 1 + log10(test / baseline)
```

每个 focus column、每个数据行平均分配该 case 的基础分。

## 产物清单

| 产物 | 路径 | 说明 |
|---|---|---|
| 公共静态库 | `build/common/libbenchmark_common.a` | benchmark 框架和公共工具 |
| memory executable | `build/memory/<case>` | memory suite 每个 case 一个可执行文件 |
| schedule executable | `build/schedule/<case>` | launch、graph、sync、stream suite |
| multicards executable | `build/multicards/<case>` | 多卡测试 |
| resource executable | `build/resource/<case>` | stream/event 资源测试 |
| musaOnly executable | `build/musaOnly/<case>` | MUSA 专属测试 |
| hotPot executable | `build/hotPotKernels/qy2Only/<case>` | 手写汇编或热点 kernel 测试 |
| ELF/PTX | `build/schedule/*.elf` 或 `*.ptx` | module load、launch、graph 测试依赖 |
| bigModule object | `build/musaOnly/bigModule/elf_path/*.o` | 大模块测试依赖 |
| 运行结果 | `result/<suite>/*.csv` | benchmark 输出 |
| 评分结果 | `score/*.csv`、`Histogram.svg` | 与 baseline 对比后的评分 |

## 对 OKR 性能建模的意义

`musa_benchmarks` 可以作为 OKR 性能建模的校准和验收工具：

| OKR 建模项 | 对应用例 |
|---|---|
| kernel launch 成本 | `kernelLaunchLatencyMode`、`kernelLaunchThroughputMode` |
| command merge 成本 | `kernelLaunchThroughputModeWithoutMerge` |
| module/function cache | `moduleLoadAndGetFunction`、`kernelLaunchApiLatency` |
| memory pool / allocation | `MallocRateFrom1Bto4GB`、`ReuseRate` |
| memcpy bandwidth | `Copy1DAlignedRate`、`CopyPinnedRate`、`CopyRegisteredRate` |
| sync wait | `efficiencyOfSync` |
| stream overlap | `streamConcurrencyCompute`、`streamConcurrencyMemcpy` |
| graph launch | `efficiencyOfGraph`、`efficiencyOfGraphLaunch`、`graphLaunchThroughputMode` |
| multi-card | `CopyP2PRate`、`kernelLaunchMulCards` |

建议新增两类 case：

| 新 case | 作用 |
|---|---|
| `traceOffOnOverhead` | 对比 baseline、trace off、trace on level 1/2，验收插桩开销 |
| `modelEventSignature` | 断言关键 ModelEvent 顺序完整，作为 CTS 行为签名 |
