# musa_benchmarks 技术洞察

> GPU 驱动性能基准测试框架。支持 MUSA（Moore Threads）和 NVIDIA CUDA 双平台，覆盖内存带宽、调度延迟、多流并发、图执行、MUSA 专属特性等场景。
>
> 基于 Celero 微基准框架，通过 Experiment/Sample/Iteration 三层循环 + User Defined Measurements（UDM）统计模型，实现可量化、可对比、可评分的驱动性能回归测试。

---

## 1. 项目定位

| 维度 | 说明 |
|------|------|
| **目标** | 对 MUSA Driver 进行全面的性能回归测试，覆盖尽可能多的场景 |
| **对标** | NVIDIA CUDA（A100/B200/H100）+ AMD ROCm（W7900） |
| **用途** | 驱动版本间性能对比、回归检测、MUSA vs CUDA 差距量化、提交前门禁 |
| **评分** | 对数评分制：`score = 1 + log10(test / baseline)`，跨 suite 可加总 |

---

## 2. 目录结构

```
musa_benchmarks/
├── common/                   # 共享框架（Celero + 计时 + 日志 + Helper）
│   ├── include/              # 33 个头文件（TestFixture, Timer, UDM, Celero...）
│   └── src/                  # 框架实现
├── memory/                   # 内存操作（带宽、拷贝、分配、清零）─ 13 个 case
├── schedule/                 # 调度 + 图执行（内核启动、模块加载、同步效率）─ 8 个 case
├── musaOnly/                 # MUSA 专属特性（MCCL, AI-CE, Atomic, 不合并）─ 6 个 case
├── multicards/               # 多卡 P2P（4 个 case）
├── resource/                 # 资源管理（event, stream）─ 2 个 case
├── hotPotKernels/             # 手写汇编内核（sgemm_core8x8，QY2 only）
├── scripts/                  # 自动化脚本
│   ├── autorun.py            # 批量运行 + CSV 收集
│   ├── calculateScoreOfSuit.py # 评分计算
│   └── ...
├── baseline/                 # 参考基线数据
│   ├── cudaA100/ cudaB200/ cudaH100/  # NVIDIA 平台
│   ├── musaPh1/ musaS5000.mst0626/    # MUSA 平台
│   └── rocm_amdW7900/                 # AMD 平台
├── TestSuitConfig.json       # 测试套件配置（权重、基线选择）
├── install.sh                # 一键安装（-m MUSA / -n CUDA / -R 重建）
└── CMakeLists.txt            # 构建系统
```

---

## 3. 基准测试循环

### 3.1 三层循环模型

每个 benchmark 遵循固定的执行模式：

```
for (Each Experiment)         ← 实验参数空间（如 copySize: 4B → 4GB）
    for (Each Sample)          ← 采样次数（通常 1-3 次）
        setUp()                ←  准备资源（malloc, 初始化数据）
        for (Each Iteration)   ←  迭代次数（通常 3-10 次，取中位/均值）
            onExperimentStart()
            run()             ←  实际测试代码
            onExperimentEnd()  ←  仅在此循环内可调用 UDM.addValue()
        tearDown()            ←  释放资源（free, 同步）
```

**所有 UDM 统计（`addValue`）必须在 Iteration 循环内调用。** 这保证了统计的样本是独立、可重复的测量值。

### 3.2 TestFixture 基类

**文件**：`common/include/TestFixture.h`

```cpp
class TestFixture {
public:
    // 定义参数空间 — 每个 Experiment 对应一个参数组合
    virtual std::vector<ExperimentValue> getExperimentValues() const;

    // 资源生命周期
    virtual void setUp(const ExperimentValue& x);      // 分配资源
    virtual void tearDown();                           // 释放资源

    // 迭代边界
    virtual void onExperimentStart(const ExperimentValue& x);
    virtual void onExperimentEnd();

    // 注册自定义指标（UDM）
    virtual std::vector<std::shared_ptr<UserDefinedMeasurement>>
        getUserDefinedMeasurements() const;
};
```

### 3.3 Celero 宏：BASELINE_F

```cpp
// 注册一个测试函数
BASELINE_F(GroupName, TestName, FixtureClass, Samples, Iterations) {
    // 测试代码
    // 使用 this->uband_xxx->addValue(value) 记录测量值
    // 使用 CPerfCounter timer 进行 CPU 端计时
}
```

### 3.4 User Defined Measurements（UDM）

**文件**：`common/include/UserDefinedMeasurements.h`

预定义的 UDM 类型（自动统计 mean/min/max/stddev/variance）：

| UDM 类型 | 基础模板 | 用途 |
|----------|---------|------|
| `UDMBandWidth` | `double` | 带宽 `B/s` |
| `UDMThroughPut` | `double` | 吞吐量 `ops/s` |
| `UDMTFLOPS` | `double` | 计算性能 `TF/s` |
| `UDMGPUTime` | `double` | GPU 耗时 `ms` |
| `UDMCPUTime` | `double` | CPU 耗时 `ms` |
| `UDMCount` | `int` | 计数值 |
| `UDMRatio` | `double` | 比率 |
| `UDMUseage` | `int` | 使用率 |

**UDM 命名约定**：以 `*` 开头的列名被评分系统识别为 "focus column"，参与分数计算。非 `*` 开头的列为辅助信息。

---

## 4. 测试套件全景

### 4.1 套件总览

| Suite | 用例数 | 基线对标 | 评分 |
|-------|--------|---------|------|
| **memoryOps** | 13 | cudaA100 | 1000 |
| **graphAndSchedule** | 7 | cudaA100 | 1000 |
| **mulStreams** | 5 | cudaA100 | 1000 |
| **musaOnly** | 6 | musaPh1 | 1000 |
| **resourceManage** | 2 | — | — |
| **multicards** | 3 | — | — |

### 4.2 memoryOps（13 个 case）

| Case | 测试内容 | 参数范围 |
|------|----------|---------|
| `Copy1DAlignedRate` | H2D/D2H 带宽（对齐） | 4B → 4GB |
| `Copy1DUnAlignedRate` | H2D/D2H 带宽（非对齐） | 4B → 4GB |
| `Copy2DRate` | 2D 拷贝带宽 | 多种尺寸 |
| `Copy3DRate` | 3D 拷贝带宽 | 多种尺寸 |
| `Copy3DArrayRate` | 3D Array 拷贝 | 多种尺寸 |
| `CopyPinnedRate` | Pinned memory 拷贝带宽 | 4B → 4GB |
| `CopyRegisteredRate` | Registered memory 拷贝 | 4B → 4GB |
| `SetRate` | Memset 清零速度 | 1B → 1GB |
| `MallocRateFrom1Bto4GB` | 内存分配速度 | 1B → 4GB |
| `GpuReadAndWriteRate` | GPU kernel 内读写带宽 | 多种尺寸 |
| `CpuReadAndWriteRate` | CPU 端读写带宽 | 多种尺寸 |
| `HostRegisterAndUnRegister` | Host 内存注册/解注册延迟 | — |
| `IpcOpenMemHandle` | IPC 共享内存句柄打开速度 | — |

### 4.3 graphAndSchedule（7 个 case）

| Case | 测试内容 |
|------|----------|
| `kernelLaunchThroughputMode` | 内核启动吞吐量（异步，统计平均） |
| `kernelLaunchLatencyMode` | 内核启动延迟（同步等待） |
| `moduleLoadAndGetFunction` | 模块加载 + 函数获取速度 |
| `efficiencyOfGraph` | CUDA Graph 加速比（graph vs 普通 launch） |
| `efficiencyOfGraphLaunch` | Graph launch 延迟 |
| `graphLaunchThroughputMode` | Graph launch 吞吐量 |
| `efficiencyOfSync` | 同步机制效率（event/stream/context sync） |

### 4.4 mulStreams（5 个 case）

| Case | 测试内容 |
|------|----------|
| `kernelLaunchMulStreamTimerByEvent` | 多流内核启动（GPU event 计时） |
| `kernelLaunchMulStreamTimerByCpu` | 多流内核启动（CPU 计时） |
| `parallelismOfDifferentCommands` | 不同类型命令的并行度（kernel+memcpy） |
| `streamConcurrencyCompute` | 计算流并发效率 |
| `streamConcurrencyMemcpy` | 拷贝流并发效率 |

### 4.5 musaOnly（6 个 case，仅 MUSA 平台）

| Case | 测试内容 |
|------|----------|
| `efficiencyOfMcclCeGraph` | MCCL + CE + Graph 组合效率 |
| `efficiencyOfAiCeCommGraph` | AI-CE 通信 + Graph 组合 |
| `memoryAtomicLaunchThroughputMode` | MUSA Atomic 操作吞吐量 |
| `memoryAtomicLaunchLatencyMode` | MUSA Atomic 操作延迟 |
| `kernelLaunchThroughputModeWithoutMerge` | 禁用 kernel merge 时的吞吐量 |
| `kernelLaunchApiLatency` | API 级别内核启动延迟 |

### 4.6 multicards（3 个 case）

| Case | 测试内容 |
|------|----------|
| `CopyP2PRate` | Peer-to-Peer 拷贝带宽 |
| `CopySetiPj2PkRate` | 多卡间 Set + Copy 组合 |
| `kernelLaunchMulCards` | 多卡同时内核启动 |

### 4.7 resourceManage（2 个 case）

| Case | 测试内容 |
|------|----------|
| `eventManage` | Event 创建/销毁/同步开销 |
| `streamManage` | Stream 创建/销毁开销 |

---

## 5. 构建系统

### 5.1 双平台支持

```cmake
# MUSA 模式: clang++ (MCC)
./install.sh -m
  → ENABLE_MCC=ON → CMAKE_CXX_COMPILER=/usr/local/musa/bin/clang++
  → include: /usr/local/musa/include
  → lib:     /usr/local/musa/lib

# CUDA 模式: nvcc
./install.sh -n
  → ENABLE_NVCC=ON → CMAKE_CUDA_COMPILER=nvcc
  → CUDA_ARCH: 70 75 80 86
  → include: /usr/local/cuda/include
  → lib:     /usr/local/cuda/lib64

# 纯 C++ 模式（不使用 GPU 编译器）
./install.sh  # 不带 -m 也不带 -n
  → 默认 g++，仅构建不依赖 GPU 的用例
```

### 5.2 GPU 架构自动检测

构建时通过 `lspci -n | grep 1ed5` 检测 PCI 设备 ID，映射到 MUSA 架构：

```
PCI ID → 架构映射（CMakeLists.txt:66-92）:
  0111 → mp_10 (QY1)
  0201 → mp_21 (QY2)
  0400 → mp_31 (PH1)
  0500 → mp_32 (PH1S)
  0600 → mp_41 (HG)
  0610 → mp_42 (LS)
  0700 → mp_43 (HS)
```

### 5.3 条件编译

- `musaOnly/` 仅在 `ENABLE_MCC=ON` 时编译（依赖 MUSA 专属 API）
- `hotPotKernels/` 包含手写汇编（QY2 only），通过 `-DMTGPU_ARCH` 控制
- 通过 `#ifdef TEST_ON_NVIDIA` 宏区分 CUDA/MUSA 代码路径

---

## 6. 评分系统

### 6.1 配置驱动

**文件**：`TestSuitConfig.json`

```json
{
    "suitName": "graphAndSchedule",
    "baseline": "cudaA100",      // 对比基线
    "baseScore": 1000,            // 基准分数
    "testCases": [
        { "caseName": "kernelLaunchThroughputMode", "scoreWeight": 1 },
        { "caseName": "kernelLaunchLatencyMode", "scoreWeight": 1 },
        // ...
    ]
}
```

### 6.2 评分算法

```python
# 核心公式（calculateScoreOfSuit.py:121）
score_ratio = 1 + log10(test / baseline)
#   test > baseline → ratio > 1 → 加分
#   test = baseline → ratio = 1 → 持平
#   test < baseline → ratio < 1 → 扣分

# 每个数据点均匀分配分数
score_per_element = baseScore / (focus_columns * rows)

# 总分为所有数据点的加权和
total_score = Σ(score_ratio × score_per_element)
```

### 6.3 评分特性

- **对数尺度**：大数值差异不会被过度放大（带宽 10x 差距 ≈ 2 分 vs 1000 分）
- **列级评分**：以 `*` 开头的 CSV 列参与评分，如 `*AH2D`、`*D2AH`
- **数据完整性检查**：行数/列数必须与基线完全匹配，否则该 case 跳过
- **权重可配置**：在 `TestSuitConfig.json` 中调整每个 case 的 `scoreWeight`

---

## 7. 输出格式

### 7.1 CSV 格式

每个 benchmark 生成一个 CSV 文件，格式如下：

```
Value,Iterations,*AH2D,*D2AH,err
4,1,5240.5,5180.3,0
8,1,10480.2,10360.1,0
16,1,20900.8,20720.5,0
...
```

- 第一列 `Value`：Experiment 参数值（如 copySize/streamCount）
- 以 `*` 开头的列：评分焦点列
- 其他列：辅助信息（不计分）

### 7.2 结果目录结构

```
result/
├── memoryOps/
│   ├── Copy1DAlignedRate.csv
│   ├── Copy2DRate.csv
│   └── ...
├── graphAndSchedule/
│   ├── kernelLaunchThroughputMode.csv
│   └── ...
└── ...

score/
├── memoryOps.csv           # 每个 case 的得分明细
├── graphAndSchedule.csv
├── totalScore.csv           # 各 suite 汇总 + 总分
└── Histogram.svg            # 评分分布直方图
```

---

## 8. 代码示例：Copy1DAlignedRate

**文件**：`memory/Copy1DAlignedRate.cpp`

```cpp
// ① 定义 Fixture：包含参数空间、资源、UDM
class CopyFixture : public TestFixture {
    // 参数空间：4B, 8B, 16B, ..., 4GB（2^2 到 2^32）
    std::vector<ExperimentValue> getExperimentValues() const override {
        for (int i = 2; i <= 32; ++i) {
            ev.Value = 1LL << i;
            problemSpace.push_back(ev);
        }
        return problemSpace;
    }

    void setUp(const ExperimentValue& x) override {
        this->copySize = x.Value;  // 记录当前参数
    }

    // 注册自定义指标（评分时以 * 开头的列）
    std::vector<std::shared_ptr<UserDefinedMeasurement>>
        getUserDefinedMeasurements() const override {
        return { this->uband_ah2d,   // *AH2D → 参与评分
                 this->uband_d2ah,   // *D2AH → 参与评分
                 this->ucont_error }; // err   → 辅助信息
    }

    std::shared_ptr<UDMBandWidth> uband_ah2d{new UDMBandWidth("*AH2D")};
    std::shared_ptr<UDMBandWidth> uband_d2ah{new UDMBandWidth("*D2AH")};
    std::shared_ptr<UDMCount> ucont_error{new UDMCount("err")};
};

// ② 注册测试函数
BASELINE_F(musaCopy, copyRateAligned, CopyFixture, 1, 3) {
    // step 1: malloc + 初始化数据
    musaMalloc(&d_B, copySize);
    int* hA = aligned_alloc(128, copySize);

    // step 2: 计时 → H2D copy
    timer.Restart();
    musaMemcpyAsync(d_B, hA, copySize, musaMemcpyHostToDevice, nullptr);
    musaDeviceSynchronize();
    timer.Stop();

    // step 3: 记录 UDM（带宽 = 字节数 / 秒）
    double bw = copySize / (timer.GetElapsedSeconds());
    this->uband_ah2d->addValue(bw);  // ★ 统计到 *AH2D 列 ★

    // step 4: 结果校验
    errorCnt = CheckResult(hB, hA, numElements);

    // step 5: 释放资源
    free(hA);
    musaFree(d_B);
}
```

---

## 9. Celero 框架集成

**文件**：`common/include/Celero.h`

musa_benchmarks 基于 Celero（v2.x）微基准框架，进行了定制化修改：

| Celero 原始概念 | musa_benchmarks 使用 |
|----------------|---------------------|
| `CELERO_MAIN` | `Run(argc, argv)` — 自定义 main 入口 |
| `BASELINE` / `BENCHMARK` | `BASELINE_F(Group, Name, Fixture, Samples, Iters)` |
| Celero 内置 timer | `CPerfCounter` （musa 定制 CPU 计时器） |
| Celero baseline 概念 | 基础用例（如 CPU 版本）vs 对比用例 |
| Factory 注册 | 命令行参数 `-t result.csv` 控制输出 |

---

## 10. 测试执行流程

```bash
# ① 安装依赖 + 构建
sudo apt install libcpuid-dev libeigen3-dev hwloc libblas-dev libopenblas-dev
./install.sh -m          # MUSA 平台

# ② 运行全部测试
cd scripts/
python3 autorun.py --suits memoryOp mulStreams graphAndSchedule

# ③ 单独运行某个 case
cd build/memory/
./Copy1DAlignedRate -t result.csv    # -t 指定 CSV 输出

# ④ 计算评分
python3 calculateScoreOfSuit.py \
    --base ../baseline/      \    # 基线目录
    --test ../result/        \    # 测试结果目录
    --score ../score/        \    # 评分输出
    --config ../TestSuitConfig.json

# ⑤ 查看结果
cat ../score/totalScore.csv
```

---

## 11. 基线数据库

| 基线 | 平台 | 用途 |
|------|------|------|
| `baseline/cudaA100/` | NVIDIA A100 | 主要对标基线（graphAndSchedule/memoryOps/mulStreams） |
| `baseline/cudaB200/` | NVIDIA B200 | 更高性能对标 |
| `baseline/cudaH100/` | NVIDIA H100 | 主流数据中心 GPU 对标 |
| `baseline/musaPh1/` | MUSA PH1 (S4000) | MUSA 专属特性的自比基线 |
| `baseline/musaS5000.mst0626/` | MUSA S5000 (PH1S) | MUSA 下一代自比基线 |
| `baseline/rocm_amdW7900/` | AMD W7900 | AMD ROCm 平台对标 |

### 基线文件格式

```
# memoryOps/Copy1DAlignedRate.csv
Value,Iterations,*AH2D,*D2AH,err
4,1,5240.5,5180.3,0     ← 4B 拷贝带宽 ~5.2 GB/s
8,1,10480.2,10360.1,0
16,1,20900.8,20720.5,0
...
4294967296,1,500000.0,490000.0,0  ← 4GB 拷贝带宽 ~500 GB/s
```

---

## 12. 设计特点与局限

### 优点
- **双平台统一**：同一套代码通过条件编译同时支持 MUSA 和 CUDA
- **可量化对比**：对数评分使跨平台、跨版本的大范围性能差异可比较
- **基线驱动**：参考数据已预制（A100/H100/W7900/S5000），开箱即用
- **自动化完整**：从构建 → 运行 → 收集 → 评分 → 可视化全流程脚本化
- **参数化测试**：每个 case 通过 `getExperimentValues()` 声明参数空间，框架自动遍历

### 局限
- **单 GPU 为主**：大部分 case 只测单卡，多卡覆盖较少（仅 3 个 case）
- **无实时监控**：仅通过 CSV 记录最终统计值，缺少 per-iteration 的时序数据
- **基线维护成本**：每个基线需要完整运行一次所有 case，新增 GPU 或 SDK 版本需要更新基线
- **无 Driver 内部 hook**：依赖公共 API（musaMemcpy/musaLaunchKernel），无法测量驱动内部行为（如 CCB、kick、pool hit/miss）
- **精度受限于 CPU 计时**：H2D/D2H 带宽用 CPU timer 计时，包含了 kernel launch + sync 的开销
- **hotPotKernels 仅 QY2**：手写 sgemm 汇编仅针对 QY2 架构，无法在其他平台运行
