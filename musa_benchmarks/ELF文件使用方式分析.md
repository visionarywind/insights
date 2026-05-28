# ELF 文件使用方式分析

> musa_benchmarks 中 ELF 文件的使用链路：从预编译生成 → CMake 部署 → 运行时 Driver API 加载 → benchmark 测量。

---

## 1. 三步使用模式

所有使用 ELF 的 benchmark 遵循同一套 Driver API 模式：

```cpp
// ① 加载模块（从文件系统读取 ELF → 驱动解析 → 加载到 GPU）
muModuleLoad(&module, "./EmptyKernel.elf");

// ② 查找函数（在模块符号表中按 mangled name 查找 kernel）
muModuleGetFunction(&function, module, "_Z11EmptyKernelv");

// ③ 启动 kernel（反复调用以测量 launch 开销）
muLaunchKernel(function, gridDim, blockDim, args, sharedMem, stream);
```

---

## 2. 具体用例

### 2.1 测 launch 吞吐量 — `kernelLaunchThroughputMode`

**文件**：`schedule/kernelLaunchThroughputMode.cpp`

ELF 路径在 fixture 成员中硬编码（第 64-68 行）：

```cpp
#ifdef TEST_ON_NVIDIA
    std::string codeObjectFile = std::string("./EmptyKernel.ptx");
#else
    std::string codeObjectFile = std::string("./EmptyKernel.elf");  // MUSA 走 ELF
#endif
    std::string functionName = std::string("_Z11EmptyKernelv");     // mangled 符号名
```

运行时（`BASELINE_F` 宏内）：

```cpp
// 加载模块
MUmodule module;
muModuleLoad(&module, codeObjectFile.c_str());

// 获取函数
MUfunction function;
muModuleGetFunction(&function, module, functionName.c_str());

// 反复 launch N 次（N = 1, 10, 100, 1000 作为实验参数）
for (int i = 0; i < m_count; ++i) {
    muLaunchKernel(function, 1, 1, 1, 1, 1, 0, stream, args, 0);
}
// 用 GPU event 计时 → 计算吞吐量 (launches/sec)
```

`EmptyKernel` 是空函数体，GPU 执行它几乎零时间，所以测量到的全部是 **Driver 层的 command 打包 + CCB 写入 + kick off 开销**。

### 2.2 测 launch 延迟 — `kernelLaunchLatencyMode`

**文件**：`schedule/kernelLaunchLatencyMode.cpp`

同样的 ELF 路径（第 63-67 行），但测试方式不同——每次 launch 后立即 `musaDeviceSynchronize()`，测量的是**端到端同步延迟**（launch + GPU 执行 + sync 回 CPU）。

### 2.3 测 module load 性能 — `moduleLoadAndGetFunction`

**文件**：`schedule/moduleLoadAndGetFunction.cpp`

这个 case 使用**三个 ELF 文件**，作为三个实验参数组（第 27 行）：

```cpp
std::vector<std::string> elfFileNames = {
    "EmptyKernel",         // 1 个 kernel — 最小模块
    "basic_funcs",         // 8 个 kernel — 中等模块
    "basic_funcs_template" // 60+ 个 kernel 模板实例化 — 大模块
};
```

平台后缀在顶层通过条件编译决定（第 21-25 行）：

```cpp
#ifdef TEST_ON_NVIDIA
const std::string fileSuffix = ".ptx";
#else
const std::string fileSuffix = ".elf";   // MUSA 统一用 .elf
#endif
```

实际使用时拼接（第 107 行）：

```cpp
codeObjectFile = elfFileNames[experimentValue.User_Data[0]] + fileSuffix;
// → "./EmptyKernel.elf" 或 "./basic_funcs.elf" 或 "./basic_funcs_template.elf"
```

测试分两个维度：

| Benchmark | 测量内容 | 关键 API |
|-----------|---------|---------|
| `module_load` | 文件 I/O + 驱动解析 + 加载到 GPU 的时间 | `muModuleLoad(&module, codeObjectFile.c_str())` |
| `getFunc` | 在已加载的模块中查找所有 kernel 符号的总时间 | 循环 `muModuleGetFunction(&function, module, kernelName)` |

`basic_funcs_template.elf` 有 60+ 个 mangled 符号（如 `_Z20matrixMultiplicationIdEvPKT_S2_PS0_iii`），用它测 `muModuleGetFunction` 可以评估大符号表查找的开销——这对应 PyTorch `torch.compile` 生成大量 fusion kernel 后加载模型的场景。

### 2.4 测多流并发 — 使用 CopyKernel.elf

`schedule/kernelLaunchMulStreamTimerByEvent.cpp` 和 `streamConcurrencyCompute.cpp` 等用例使用 `CopyKernel.elf`：

```cpp
codeObjectFile = std::string("./CopyKernel.elf");
functionName = std::string("CopyKernel");  // extern "C" 无 mangling
```

测试模式：创建多个 stream，每个 stream 上 launch 一个 `CopyKernel`，通过 GPU event 交错计时来验证不同 stream 是否真正并行执行。

---

## 3. 构建时如何部署到运行目录

`schedule/CMakeLists.txt` 在 cmake 配置阶段直接将 `elf/` 目录下的文件**复制到构建输出目录**：

```cmake
# MUSA/default 模式 — 复制 .elf
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/elf/EmptyKernel.elf
          ${CMAKE_CURRENT_SOURCE_DIR}/elf/CopyKernel.elf
          ${CMAKE_CURRENT_SOURCE_DIR}/elf/VectorAdd.elf
          ${CMAKE_CURRENT_SOURCE_DIR}/elf/cdm_parallelism.elf
          ${CMAKE_CURRENT_SOURCE_DIR}/elf/basic_funcs.elf
          ${CMAKE_CURRENT_SOURCE_DIR}/elf/basic_funcs_template.elf
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)

# CUDA 模式 — 复制 .ptx
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/elf/EmptyKernel.ptx
          ${CMAKE_CURRENT_SOURCE_DIR}/elf/CopyKernel.ptx
          ...
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)
```

这样构建后的目录结构是：

```
build/schedule/
├── kernelLaunchThroughputMode   ← benchmark 可执行文件
├── kernelLaunchLatencyMode
├── moduleLoadAndGetFunction
├── EmptyKernel.elf              ← 运行时依赖（相对路径 "./" 加载）
├── CopyKernel.elf
├── VectorAdd.elf
├── basic_funcs.elf
├── basic_funcs_template.elf
└── cdm_parallelism.elf
```

---

## 4. 完整流程图

```
┌─ 源码阶段 ────────────────────────────────────────────────┐
│  elf/EmptyKernel.cu                                        │
│    ↓ mcc --cuda-device-only -mtgpu --offload-arch=mp_31    │
│  elf/EmptyKernel.elf    ← 预编译的 GPU device binary       │
└───────────────────────────────────────────────────────────┘
                           ↓
┌─ CMake 构建阶段 ──────────────────────────────────────────┐
│  schedule/CMakeLists.txt                                   │
│    file(COPY elf/EmptyKernel.elf DESTINATION build/schedule/)│
│    add_executable(kernelLaunchThroughputMode ...)           │
└───────────────────────────────────────────────────────────┘
                           ↓
┌─ 运行时 ──────────────────────────────────────────────────┐
│  ./build/schedule/kernelLaunchThroughputMode               │
│    │                                                       │
│    ├─ muInit(0)                                            │
│    ├─ muCtxCreate(&ctx, 0, device)                         │
│    ├─ muModuleLoad(&mod, "./EmptyKernel.elf")  ← 加载 ELF  │
│    ├─ muModuleGetFunction(&fn, mod, "_Z11EmptyKernelv")    │
│    │                                                       │
│    └─ 循环 N 次:                                           │
│         muLaunchKernel(fn, 1,1,1, 1,1,1, 0, stream, ...)  │
│         ↑ 用 GPU event 计时                                │
│                                                             │
│    输出: CSV (launches/sec, per-launch latency)            │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 关键设计要点

### 5.1 预编译 vs JIT

ELF 文件是预编译的，不是 cmake 构建产物。这保证了：

- **消除 JIT 编译波动**：benchmark 测的是 Driver API 开销，不是编译器性能
- **测试可重复**：同样的 ELF 在不同驱动版本间产生一致的测量结果
- **文件大小可控**：`EmptyKernel.elf` 极小（~几 KB），`basic_funcs_template.elf` 较大（60+ 模板实例化），可以分别测量不同规模模块的加载开销

### 5.2 平台适配

通过条件编译 `#ifdef TEST_ON_NVIDIA` 切换 ELF/PTX 后缀，同一份 benchmark 源码同时支持 MUSA 和 CUDA 平台。

### 5.3 符号名

kernel 函数在 ELF 中以 **C++ mangled name** 存储（如 `_Z11EmptyKernelv`），`muModuleGetFunction` 需要用完整的 mangled name 查找。`CopyKernel` 例外——它用 `extern "C"` 声明，符号名就是 `CopyKernel`。

### 5.4 ELF 不是编译产物

ELF 文件不参与 cmake 的 `add_executable` 或 `add_library`，而是通过 `file(COPY)` 作为**测试数据**部署。它们由 `elf/gen.sh` 独立生成，通常只在 kernel 源码变更时才需要重新生成。

---

## 6. 各 Benchmark 使用的 ELF 汇总

| Benchmark | 使用的 ELF | 测量目标 |
|-----------|-----------|---------|
| `kernelLaunchThroughputMode` | `EmptyKernel.elf` | 异步批量 launch 吞吐量 |
| `kernelLaunchLatencyMode` | `EmptyKernel.elf` | 单次 launch + sync 延迟 |
| `kernelLaunchMulStreamTimerByEvent` | `CopyKernel.elf` | 多流 GPU event 计时 |
| `kernelLaunchMulStreamTimerByCpu` | `CopyKernel.elf` | 多流 CPU timer 计时 |
| `streamConcurrencyCompute` | `CopyKernel.elf` | 多流计算并发效率 |
| `efficiencyOfDependency` | `VectorAdd.elf` | stream 间依赖效率 |
| `moduleLoadAndGetFunction` | `EmptyKernel.elf` + `basic_funcs.elf` + `basic_funcs_template.elf` | module load 延迟 + function lookup 吞吐 |
