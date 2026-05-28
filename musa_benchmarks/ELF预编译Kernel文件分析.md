# ELF 预编译 Kernel 文件分析

> musa_benchmarks 中 `schedule/elf/` 目录下的 `.elf` 文件是预编译的 MUSA GPU kernel 二进制，用于精确测量 Driver API 层面的开销（module load、function lookup、kernel launch），而非 kernel 内部的计算性能。

---

## 1. 定位与角色

这些 ELF 文件是 **GPU device code**（不是宿主可执行文件），由 `mcc`（MUSA C/C++ Compiler）从 `.cu` 源码以 `--cuda-device-only` 模式编译生成。它们在 benchmark 运行时通过 MUSA Driver API 动态加载：

```cpp
muModuleLoad(&module, "./EmptyKernel.elf");
muModuleGetFunction(&function, module, "_Z11EmptyKernelv");
muLaunchKernel(function, grid, block, args, ...);
```

## 2. 为什么是预编译 ELF 而不是 JIT？

| 方式 | 问题 |
|------|------|
| JIT 编译（运行时从源码编译） | 编译时间波动会污染 launch/latency 测量，无法区分"编译慢"还是"launch 慢" |
| 内联 kernel（`__global__` 写在 .cu 里一起编译） | 无法独立测量 `muModuleLoad` 的文件加载 + 驱动解析开销 |
| **预编译 ELF** | 消除编译波动，精确测量 Driver API 各环节的纯开销 |

核心目的：**隔离测量**。用空 kernel 测 launch 延迟时，要确保测到的是 command 打包 + CCB 写入 + kick off 的时间，而不是编译器优化掉了什么。

## 3. 文件清单与用途

```
musa_benchmarks/schedule/elf/
├── EmptyKernel.cu / .elf / .ptx          ← 空函数体 kernel
├── CopyKernel.cu / .elf / .ptx           ← 逐元素 float 拷贝
├── VectorAdd.cu / .elf / .ptx            ← 向量加法
├── basic_funcs.cu / .elf / .ptx          ← 8 个经典 kernel
├── basic_funcs_template.cu / .elf / .ptx ← 8 个 kernel × 多类型模板（60+ 符号）
├── cdm_parallelism.cu / .elf / .ptx      ← CDM 并发 kernel
└── gen.sh                                 ← 生成脚本
```

| ELF | 包含的 kernel | 对应 benchmark | 测量目标 |
|-----|-------------|----------------|---------|
| `EmptyKernel.elf` | `EmptyKernel()` — 空函数体 | `kernelLaunchThroughputMode`、`kernelLaunchLatencyMode`、多流/多卡 launch | **纯 kernel launch 开销**：空 kernel 执行时间为零，测量值 = command 打包 + CCB 写入 + kick off |
| `CopyKernel.elf` | `CopyKernel(float*, float*, int)` — 逐元素拷贝 | `kernelLaunchMulStreamTimerByEvent`、`streamConcurrencyCompute` | **多流并发 + 依赖测试**：轻量 kernel，计算量可控，用来测 stream overlap 效率 |
| `VectorAdd.elf` | `VectorAdd(int*, int*)` — 向量加法 | `efficiencyOfDependency` 等 | **基础计算 kernel** 的 launch 和 graph 测试 |
| `basic_funcs.elf` | 8 个 kernel：vectorAddition, matrixMultiplication, parallelReduction, elementWiseMultiplication, parallelScan, histogramCalculation, matrixTranspose | `moduleLoadAndGetFunction` | **module load 性能**：测加载包含多个 kernel 的模块的文件 I/O + 驱动解析时间 |
| `basic_funcs_template.elf` | 同上 8 个 kernel 的 60+ 个模板实例化（int/float/double/long/short/char/unsigned 等组合） | `moduleLoadAndGetFunction` | **大符号表 function lookup 性能**：测 `muModuleGetFunction` 在大量 mangled 符号中查找的开销 |
| `cdm_parallelism.elf` | CDM 并发 kernel | CDM 并发测试 | CDM 并行度 |

## 4. 关键 kernel 源码

### EmptyKernel — 测量 kernel launch 开销的"探针"

```c
// EmptyKernel.cu
__global__ void EmptyKernel() {
}
```

空函数体。GPU 硬件执行它只需一个 cycle。benchmark 用它反复调用 `muLaunchKernel`，测量的是 **Driver 层的纯 overhead**：
- 参数打包（grid dim、block dim、shared mem、args）
- CCB（Command Control Block）写入
- kick off 到 GPU 调度器的延迟

### CopyKernel — 多流并发测试的"负载"

```c
// CopyKernel.cu
extern "C" __global__ void CopyKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}
```

简单逐元素拷贝，计算量 = 1 次 global memory read + 1 次 write。在 stream concurrency 测试中，多个 stream 各 launch 一个 CopyKernel，通过 GPU event 计时来验证各 stream 是否真正并行执行。

### basic_funcs_template — 大符号表查找测试

`basic_funcs_template.cu` 将 8 个 kernel 用多种数据类型（int/float/double/long/short/char/unsigned）做模板实例化，生成 60+ 个 mangled 符号：

```
_Z20matrixMultiplicationIdEvPKT_S2_PS0_iii   // matrixMultiplication<double>
_Z20matrixMultiplicationIfEvPKT_S2_PS0_iii   // matrixMultiplication<float>
_Z20matrixMultiplicationIiEvPKT_S2_PS0_iii   // matrixMultiplication<int>
_Z17parallelReductionIdEvPKT_PS0_i            // parallelReduction<double>
...
```

benchmark 逐一调用 `muModuleGetFunction(module, mangled_name)` 查找每个符号，测量大符号表中的符号解析开销。这对评估"模型加载时查找数百个 kernel function"的场景（如 PyTorch 的 `torch.compile` 生成的 fusion kernel）很有参考价值。

## 5. 生成流程

```bash
# gen.sh — 由 mcc 编译 .cu → .elf（device-only，多架构）
mcc EmptyKernel.cu -o EmptyKernel.elf --cuda-device-only -mtgpu \
    --offload-arch=mp_21 --offload-arch=mp_22 --offload-arch=mp_31
```

关键参数：
- `--cuda-device-only`：只生成 GPU device code，不生成 host 端 wrapper（因为 host 端通过 Driver API 加载，不需要编译时链接）
- `-mtgpu`：MUSA GPU 目标
- `--offload-arch=mp_21 --offload-arch=mp_22 --offload-arch=mp_31`：生成多架构 fat binary，运行时驱动根据实际 GPU 架构选择匹配的 binary

CUDA 平台对应生成 `.ptx`（PTX 中间表示，由 CUDA driver JIT 编译为 SASS）。

## 6. 构建集成

`schedule/CMakeLists.txt` 在构建时将 `elf/` 目录下的 ELF/PTX 文件**直接复制到 `build/schedule/`**：

```cmake
# MUSA/default 模式
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/elf/EmptyKernel.elf
          ${CMAKE_CURRENT_SOURCE_DIR}/elf/CopyKernel.elf
          ...
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

# CUDA 模式
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/elf/EmptyKernel.ptx
          ${CMAKE_CURRENT_SOURCE_DIR}/elf/CopyKernel.ptx
          ...
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
```

这样 benchmark 可执行文件运行时，通过相对路径 `"./EmptyKernel.elf"` 就能加载。平台选择在源码中通过条件编译处理：

```cpp
#ifdef TEST_ON_NVIDIA
    std::string codeObjectFile = std::string("./EmptyKernel.ptx");  // CUDA
#else
    std::string codeObjectFile = std::string("./EmptyKernel.elf");  // MUSA
#endif
```

## 7. 对应 Benchmark 汇总

| Benchmark | 使用的 ELF | 测量内容 |
|-----------|-----------|---------|
| `kernelLaunchThroughputMode` | `EmptyKernel.elf` | 异步批量 launch 的吞吐量（launches/sec） |
| `kernelLaunchLatencyMode` | `EmptyKernel.elf` | 单次 launch + sync 的端到端延迟 |
| `kernelLaunchMulStreamTimerByEvent` | `CopyKernel.elf` | 多流并发时 GPU event 计时精度 |
| `kernelLaunchMulStreamTimerByCpu` | `CopyKernel.elf` | 多流并发时 CPU timer 计时精度 |
| `streamConcurrencyCompute` | `CopyKernel.elf` | 多流计算并发效率 |
| `moduleLoadAndGetFunction` | `EmptyKernel.elf`、`basic_funcs.elf`、`basic_funcs_template.elf` | module 文件加载时间、function 查找吞吐量 |
| `efficiencyOfDependency` | `VectorAdd.elf` 等 | stream 间依赖和同步效率 |
