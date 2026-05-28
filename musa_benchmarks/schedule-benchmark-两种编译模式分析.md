# musa_benchmarks schedule/ 目录：两种编译与启动模式分析

## 1. 概述

`musa_benchmarks/schedule/` 目录下有 17 个 benchmark 源文件，但编译和 kernel 启动方式分为截然不同的两种模式。这种分化不是随意为之，而是由"测什么"决定的。

| 模式 | 源文件扩展名 | 数量 | kernel 来源 | 启动 API | 测量目标 |
|------|------------|------|------------|---------|---------|
| 内联编译 | `.cu` | 9 | 同文件中 `__global__` 定义 | Runtime API (`<<<>>>` 或 `musaLaunchKernel`) | kernel 执行行为 |
| 预编译分离 | `.cpp` | 8 | 独立 `.elf` 文件 | Driver API (`muModuleLoad` → `muLaunchKernel`) | 运行时基础设施开销 |

## 2. 模式一：`.cu` 内联编译 — 测"跑什么"

### 2.1 源文件列表

```
efficiencyOfGraph.cu          efficiencyOfGraphLaunch.cu    efficiencyOfSync.cu
graphLaunchThroughputMode.cu  kernelGap.cu                 kernelLaunchWithDependency.cu
kernelLaunchWithLlcPersistent.cu  kernelPostprocessBubbleOfHW.cu  parallelismOfDifferentCommands.cu
```

### 2.2 代码特征

Kernel 函数和 host 代码在同一源文件内，kernel 携带**有意义的计算**：

```cpp
// efficiencyOfSync.cu — kernel 定义就在文件中
__global__ void delay(volatile int* flag, unsigned long long timeout_clocks = 100000000) {
    long long int start_clock, sample_clock;
    start_clock = clock64();
    while (*flag) {
        sample_clock = clock64();
        if (sample_clock - start_clock > timeout_clocks) {
            break;
        }
    }
}

__global__ void emptyKernel() {}

// 启动：直接调用函数名，编译后符号直接可见
delay<<<blocks, threads>>>(flag, timeout);
emptyKernel<<<1, 1, 0, stream>>>();
```

```cpp
// parallelismOfDifferentCommands.cu — 不同指令并行性
__global__ void compute_bound_kernel(float *a, float *b, float *out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = tid; idx < N; idx += total_threads) {
        float x = 1.0f + 0.0001f * idx;
        #pragma unroll 100
        for (int i = 0; i < 100; ++i) {
            x = x * 1.000001f + 0.00001f;
        }
        out[idx] = x;
    }
}
__global__ void memory_bound_kernel(float *a, float *b, float *out, int N) { ... }
```

### 2.3 编译路径

CMakeLists.txt 中，`.cu` 文件被当作 GPU 源码一起编译：

```cmake
if(ENABLE_MCC)
    set_source_files_properties(${case_path} PROPERTIES LANGUAGE CXX)
    target_compile_options(${test_case} PRIVATE -x musa -mtgpu --cuda-gpu-arch=${COMPILE_ARCH})
elseif(ENABLE_NVCC)
    set_property(TARGET ${test_case} PROPERTY CUDA_STANDARD 17)
endif()
```

MCC 模式下设置 `-x musa -mtgpu`，NVCC 模式下设置 `CUDA_STANDARD 17`。host 和 device 代码一起编译成单个可执行文件，kernel 函数符号直接链接。

### 2.4 测量目标

这类 benchmark 关注的是 **kernel 执行本身的行为**：

| Benchmark | 测量内容 |
|-----------|---------|
| `efficiencyOfGraph` | MUSA Graph 中 kernel 依赖/回调执行效率 |
| `efficiencyOfGraphLaunch` | Graph Launch 的效率 |
| `efficiencyOfSync` | 同步操作的吞吐量 |
| `graphLaunchThroughputMode` | 通过 Graph 方式 launch 的吞吐 |
| `kernelGap` | kernel 之间的间隔（gap）分析 |
| `kernelLaunchWithDependency` | 有依赖关系时的 kernel launch |
| `kernelLaunchWithLlcPersistent` | LLC persistent 数据对 launch 的影响 |
| `kernelPostprocessBubbleOfHW` | 硬件后处理气泡分析 |
| `parallelismOfDifferentCommands` | compute/memory bound 指令间的并行性 |

Kernel 函数不是空壳 — `delay` 用 `clock64()` 做忙等待，`compute_bound_kernel` 有 100 次循环展开，`memory_bound_kernel` 做访存。因为测的是**执行行为**，kernel 必须做真实工作。

## 3. 模式二：`.cpp` + ELF — 测"启动本身"

### 3.1 源文件列表

```
kernelLaunchThroughputMode.cpp        kernelLaunchLatencyMode.cpp
kernelLaunchMulStreamTimerByCpu.cpp   kernelLaunchMulStreamTimerByEvent.cpp
moduleLoadAndGetFunction.cpp          streamConcurrencyCompute.cpp
efficiencyOfDependency.cpp            kernelLaunchThroughputModeWithoutMerge.cpp (musaOnly/)
```

### 3.2 代码特征

Host 代码是**纯 C++**，不含任何 `__global__` 定义。Kernel 通过 Driver API 从外部 `.elf` 文件加载：

```cpp
// kernelLaunchThroughputMode.cpp — 纯 C++，没有 __global__

#ifdef TEST_ON_NVIDIA
    std::string codeObjectFile = std::string("./EmptyKernel.ptx");
#else
    std::string codeObjectFile = std::string("./EmptyKernel.elf");
#endif
    std::string functionName = std::string("_Z11EmptyKernelv");  // C++ mangled name

// setUp() 中加载
MUmodule module;
MUfunction function;
checkMuErrors(muModuleLoad(&module, codeObjectFile.c_str()));
checkMuErrors(muModuleGetFunction(&function, module, functionName.c_str()));

// benchmark 循环中启动
for (int i = 0; i < m_count; ++i) {
    checkMuErrors(muLaunchKernel(function, 1, 1, 1,  /* grid */
        1, 1, 1,                                      /* block */
        0, streamIdx, nullptr, nullptr));
}
```

关键特征：
- **`_Z11EmptyKernelv`**：这是 C++ mangled name。Itanium C++ ABI 规则下，`EmptyKernel()` → `_Z11EmptyKernelv`（`Z`=C++ 符号，`11`=名称长度，`v`=void 返回值）。必须用 mangled name 查找，因为 ELF 中存储的就是它。
- **三步调用**：`muModuleLoad` → `muModuleGetFunction` → `muLaunchKernel`，全程 Driver API。
- **kernel 极简**：`EmptyKernel.elf` 里是空函数体 `{}`，`VectorAdd.elf` 是简单的 `c[i] = a[i] + b[i]`。kernel 执行时间必须远小于 launch 开销，否则测量被污染。

### 3.3 Kernel 来源：独立预编译

Kernel 在 `schedule/elf/` 目录下独立编译，与 host 代码完全解耦：

```bash
# gen.sh
mcc EmptyKernel.cu -o EmptyKernel.elf --cuda-device-only -mtgpu --offload-arch=mp_31
mcc CopyKernel.cu   -o CopyKernel.elf   --cuda-device-only -mtgpu --offload-arch=mp_31
mcc VectorAdd.cu    -o VectorAdd.elf    --cuda-device-only -mtgpu --offload-arch=mp_31
mcc basic_funcs.cu  -o basic_funcs.elf  --cuda-device-only -mtgpu --offload-arch=mp_31
mcc basic_funcs_template.cu -o basic_funcs_template.elf --cuda-device-only -mtgpu --offload-arch=mp_31
mcc cdm_parallelism.cu -o cdm_parallelism.elf --cuda-device-only -mtgpu --offload-arch=mp_31
```

`--cuda-device-only` 是关键：只生成 device code（ELF），不链接 host。生成产物通过 CMake 的 `configure_file(COPYONLY)` 复制到构建输出目录，host 程序运行时从文件系统加载。

### 3.4 测量目标

这类 benchmark 关注的是 **MUSA 运行时基础设施的开销**：

| Benchmark | 使用的 ELF | 测量目标 |
|-----------|-----------|---------|
| `kernelLaunchThroughputMode` | `EmptyKernel.elf` | 单 stream 内连续 launch 的吞吐量 |
| `kernelLaunchLatencyMode` | `EmptyKernel.elf` | 单次 launch 的端到端延迟 |
| `kernelLaunchMulStreamTimerByCpu` | `EmptyKernel.elf` | 多 stream 并发 launch，CPU 计时 |
| `kernelLaunchMulStreamTimerByEvent` | `EmptyKernel.elf` | 多 stream 并发 launch，Event 计时 |
| `moduleLoadAndGetFunction` | `EmptyKernel.elf`、`basic_funcs.elf`、`basic_funcs_template.elf` | `muModuleLoad` 和 `muModuleGetFunction` 耗时（小/中/大符号表） |
| `streamConcurrencyCompute` | `cdm_parallelism.elf` | 多 stream 并发执行（CDM 并行性） |
| `efficiencyOfDependency` | `VectorAdd.elf` | kernel 与 device-to-device copy 的依赖效率 |
| `kernelLaunchThroughputModeWithoutMerge` (musaOnly) | `EmptyKernel.elf` | 不合并的 launch 吞吐量 |

以 `kernelLaunchThroughputMode` 为例，它的测量核心是：
```cpp
muEventRecord(start, stream);
for (int i = 0; i < m_count; ++i) {
    muLaunchKernel(function, 1,1,1, 1,1,1, 0, stream, nullptr, nullptr);
}
muEventRecord(stop, stream);
muEventSynchronize(stop);
muEventElapsedTime(&milliseconds, start, stop);
```

100 万个空 kernel launch，kernel 本身执行时间为零。测出的就是 `muLaunchKernel` 调用链的纯开销：命令队列写入、调度、硬件 dispatch。

## 4. 为什么这样分？— 设计意图

### 4.1 消除噪声

测 launch 吞吐量时，如果 kernel 自身有执行时间，就会淹没 launch 开销。`.cu` 模式下即使写 `emptyKernel(){}`，编译器仍然会生成完整的 kernel 启动序列（参数打包、运行时封装等），无法完全排除 Runtime API 的隐式初始化。

用 `.cpp` + ELF 模式，可以：
- 用 Driver API 精确控制每个步骤，排除 Runtime API 的 lazy initialization
- 用空 kernel ELF 让 device 端执行时间趋近于零
- 将 `muModuleLoad` 放在 warmup 之外，测量纯 launch 循环

### 4.2 测量 module load 开销

`moduleLoadAndGetFunction` benchmark 明确测量 `muModuleLoad` 和 `muModuleGetFunction` 本身的耗时。这只有在 kernel 完全独立于 host 编译时才能做 — `.cu` 模式下 module 在程序启动时就被隐式加载了。

它还对比了三种规模的 ELF：
- `EmptyKernel.elf`：1 个函数，最小符号表
- `basic_funcs.elf`：7 个函数，中型符号表
- `basic_funcs_template.elf`：60+ 个模板实例化函数，大型符号表

这是为了测量符号查找开销随符号表规模的变化。

### 4.3 跨平台复用

同一份 `.cpp` 代码通过 `#ifdef TEST_ON_NVIDIA` 切换 code object 文件：

```cpp
#ifdef TEST_ON_NVIDIA
    std::string codeObjectFile = std::string("./EmptyKernel.ptx");  // CUDA PTX
#else
    std::string codeObjectFile = std::string("./EmptyKernel.elf");  // MUSA ELF
#endif
```

Host benchmark 逻辑完全不变，只换后端编译产物。这在 `.cu` 模式下做不到 — CUDA 和 MUSA 的 `__global__` 语法虽然兼容，但编译器和运行时行为不同，混在一起无法精确控制变量。

## 5. CMakeLists.txt 中的编译路径对比

```cmake
# 对所有 .cpp 和 .cu 文件统一处理
file(GLOB KERNEL_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB KERNEL_TEST_SOURCES_CU ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

if(ENABLE_MCC OR ENABLE_NVCC)
    list(APPEND KERNEL_TEST_SOURCES ${KERNEL_TEST_SOURCES_CU})  # .cu 只在 GPU 编译器可用时加入
endif()

foreach(case_path ${KERNEL_TEST_SOURCES})
    # .cpp 文件：
    #   - ENABLE_MCC=ON  → 设置 LANGUAGE CXX，加 -x musa -mtgpu（即使用 MUSA C++ 编译器）
    #   - ENABLE_NVCC=ON → 设置 CUDA_STANDARD 17（即使用 NVCC）
    #   - 都 OFF        → 普通 C++ 编译（但此时 .cu 不会被加入列表）
    # .cu 文件：
    #   - ENABLE_MCC=ON  → 同 .cpp，-x musa -mtgpu
    #   - ENABLE_NVCC=ON → CUDA_STANDARD 17
    #   - 都 OFF        → 不编译（已在上面被排除）
```

`.cpp` 文件在 MCC 模式下仍然用 `-x musa -mtgpu` 编译，因为 Driver API 头文件（`musa.h`）可能依赖 MUSA 编译器扩展。但代码本身不含 `__global__`，所以编译器不会为它生成 device code。

## 6. 汇总对比表

| 维度 | `.cu`（内联） | `.cpp` + ELF（预编译） |
|------|-------------|---------------------|
| **文件数** | 9 | 8 |
| **kernel 定义位置** | 同文件 `__global__` | 独立 `elf/*.cu` → `mcc --cuda-device-only` |
| **kernel 内容** | 有意义（delay、compute/memory bound、多 kernel 依赖） | 故意极简（空函数体、简单向量加） |
| **host 编译** | 与 device 一起编译 | 纯 C++ 编译（但通过 `-x musa` 使用 MUSA 头文件） |
| **kernel 加载** | 编译时链接，符号直接可见 | 运行时 `muModuleLoad` 从文件系统加载 |
| **启动 API** | Runtime API（`<<<>>>` 或 `musaLaunchKernel`） | Driver API（`muModuleGetFunction` → `muLaunchKernel`） |
| **函数名** | 源码名（`emptyKernel`） | C++ mangled name（`_Z11EmptyKernelv`） |
| **测量目标** | kernel 执行行为（graph 效率、sync 开销、指令并行性） | 运行时基础设施开销（launch 吞吐/延迟、module load、多流并发） |
| **跨平台** | 依赖编译器兼容性 | `#ifdef TEST_ON_NVIDIA` 切换 `.elf` / `.ptx` |
| **典型用例** | `efficiencyOfGraph`、`kernelGap`、`parallelismOfDifferentCommands` | `kernelLaunchThroughputMode`、`moduleLoadAndGetFunction` |

## 7. 设计启示

这套双模设计反映了一个清晰的基准测试哲学：

1. **隔离变量**：测 kernel 执行就把 kernel 写得真实；测 runtime 就把 kernel 清零。二者不混淆。
2. **分层测试**：Runtime API 有隐式开销（lazy init、context 管理），Driver API 更底层。用 Driver API 测出的 launch 开销才是硬件的真实调度延迟。
3. **编译时 vs 运行时**：`.cu` 把一切交给编译时，`.cpp`+ELF 把 kernel 加载推到运行时 — 两种路径各有适用场景，benchmark 框架同时覆盖。
