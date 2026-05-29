# MUSA Workspace 技术洞察报告

> 本目录汇集对 workspace 内各项目的源码级技术分析报告，按主题领域归档。
> 分析日期：2026-05-25
性能建模的核心目标，是在系统正式部署到生产环境之前，通过低成本的模型化手段，对系统的真实运行表现进行精准预测，将性能风险前置化解
而针对Driver性能建模的核心目标是在版本发布前对driver的性能进行量化评估和精准定位性能瓶颈
	- 量化评估性能表现：在版本发布前，通过Perf手段发现系统在不同模型下的关键性能指标，对dirver的版本进行性能看护
	- 精准定位性能瓶颈：通过对Driver内部事件的采集，精准识别出影响整体性能的核心约束环节

整体思路: 通过MUPTI埋点,将Driver串成可解释的白盒性能模型

RoadMap:	先跑通链路,然后选择单个模型实现原型,最后覆盖更多API和模型
M0:	针对内存API跑通链路,实现demo达到解释:API后经过了哪些链路,每个链路耗时是什么样
M1:	确定模型,达到针对模型本身的TOP90 API覆盖
M2:	抽象主流模型的行为模式,对抽象的行为验证性能建模
M3:	使用msight/profiling对性能建模进行对齐,达到:模型耗时在dirver侧的统计
M4:	沉淀到cts中,并对性能建模进行看护


---

## 目录结构

```
insights/
├── README.md                                # 本文件 — 总索引
├── musa-runtime/                            # MUSA-Runtime 运行时库分析
│   ├── 技术洞察报告.md                       # ~400 行，API 面、架构层次、GPU 支持、构建系统
│   ├── 能力边界分析.md                       # ~330 行，337 导出符号 vs 430 声明 vs 560 mu* 驱动 API
│   └── 架构图.md                            # 7 层架构：include → wrapper → internal → domain →  ExportTable → Driver
└── musa/                                    # MUSA Driver 内存分配器分析
    ├── 技术洞察报告.md                       # ~313 行，三层分配器、Sub-Allocation、KMD 路径
    ├── 能力边界分析.md                       # ~320 行，musa 文档覆盖 vs MUSA Driver 全 API 面
    └── 架构图.md                            # 五层架构：API → Core → HAL → M3D → KMD，memMgr+MemoryPool Sub-Alloc
├── msight-compute/                           # MUSA GPU Profiler 工具链分析
│   └── 技术洞察报告.md                       # ~200 行，模块架构、数据流、MUPTI 集成、QLCS 对比
└── musa_benchmarks/                          # GPU 性能基准测试框架 + Green Context 用例
    ├── musa_benchmarks-技术洞察.md            # Celero 框架、5 个 suite、评分系统、基线数据库
    ├── musa_benchmarks-构建架构与产物分析.md   # CMake 构建、双平台编译、产物清单
    └── green-context/
        ├── 代码解读.md                        # gree_context_test.mu 逐行解读
        └── GreenContext-性能看护用例设计.md    # 4 个回归 case 设计 + 上库 8 步流程
```

---

## 报告索引

### 1. MUSA-Runtime 计算运行时库

| 条目 | 说明 |
|------|------|
| **位置** | `musa-runtime/技术洞察报告.md` |
| **分析对象** | `/home/shanfeng/workspace/MUSA-Runtime/` |
| **版本** | 5.1.0（develop），commit a6aa4733 |
| **内容覆盖** | 项目结构全景、API 接口面（340+ 导出符号）、架构层次（API/Core/HAL/Driver）、GPU 架构支持（mp_20/30/31/32/35/60）、构建系统（CMake/Docker/CI）、依赖图谱、符号导出控制 |
| **关键词** | MUSA, Runtime, CUDA 兼容, musart, 设备管理, Stream, Graph, IPC, 纹理, 符号导出, GPU 架构 |
| **能力边界** | [能力边界分析](musa-runtime/能力边界分析.md)：337 导出符号 vs 430 声明 vs 560 mu* 驱动 API 全景比对；API 按域分类（Device/Memory/Stream/Event/Graph 等 16 域）；运行时 vs 驱动职责边界；桩函数清单（Profiler、Clang 桥）；平台架构支持矩阵；25 个 UT 模块覆盖热力图；与 CUDA Runtime 12.x 差异表 |
| **架构设计** | [架构图](musa-runtime/架构图.md)：7 层软件架构（include → generated wrapper → internal API → domain impl → ExportTable → Driver → KMD），ExportTable 14 子表、初始化状态机、TLS 设计、符号导出控制 |

### 2. MUSA GPU Driver

| 条目 | 说明 |
|------|------|
| **位置** | `musa/`（3 份文档） |
| **分析对象** | `/home/shanfeng/workspace/linux-ddk/musa/`（~315K lines，112K C++ 核心源码） |
| **内容覆盖** | [技术洞察报告](musa/技术洞察报告.md)：四层架构（API→Core→HAL→M3D）、560 muapi* 全领域分布（Memory/Graph/Stream/Context/Module 等 24 域）、三层内存分配器、9 种分配路径、API→ioctl 完整调用链、8 个关键设计模式、源码目录速查 |
| | [能力边界分析](musa/能力边界分析.md)：基于 560 个 muapi* 函数的完整 API 面比对，量化 musa 文档的横向覆盖（内存 ~29% vs 非内存 ~71%）和纵向深度（Driver→Core→HAL→KMD），含代码覆盖热力图、逐文档深度评级、使用者指南 |
| | [架构图](musa/架构图.md)：五层整体架构（API → Core → HAL → M3D → KMD），memMgr 池管理、MemoryPool 子分配算法、10 种内存类型速查、muMemAlloc 7 步完整调用链 |
| **关键词** | 内存分配器, Sub-Allocation, MemoryPool, KMD, ioctl, Peer 映射, DRM, WDDM2, 碎片管理, 惰性释放, 能力边界, API 覆盖率, 调用栈深度 |

### 3. msight-compute GPU Profiler 工具链

| 条目 | 说明 |
|------|------|
| **位置** | `msight-compute/技术洞察报告.md` |
| **分析对象** | `/home/shanfeng/workspace/msight-compute/` |
| **版本** | 1.3.0，C++17/CMake/Qt6 |
| **内容覆盖** | 项目结构全景、9 个核心模块（profiler/capture/injection/report/mt_rules/mupti_api）、两种 Profiling 模式（Direct/Injection）、MUPTI API 动态加载封装、protobuf 数据模型、CLI+GUI 双入口、与 musa/MUSA-Runtime/MUPTI 的分层协作关系 |
| **关键词** | Nsight Compute 对标, MUPTI, kernel profiling, roofline, timeline, PC sampling, app replay, .mcu-rep |

### 4. musa_benchmarks GPU 性能基准测试

| 条目 | 说明 |
|------|------|
| **位置** | `musa_benchmarks/`（2 份技术文档 + green-context 分析） |
| **分析对象** | `/workspace/workspace/musa_benchmarks/` |
| **内容覆盖** | [技术洞察](musa_benchmarks/musa_benchmarks-技术洞察.md)：Celero 框架三层循环、6 个测试 suite（memoryOps/graphAndSchedule/mulStreams/musaOnly/resourceManage/multicards）、UDM 统计模型、对数评分体系 (`score = 1 + log10(test/baseline)`)、双平台编译（MUSA+CUDA）、基线数据库（A100/H100/W7900/S5000） |
| | [构建架构与产物分析](musa_benchmarks/musa_benchmarks-构建架构与产物分析.md)：CMake 全局架构、MCC/NVCC/G++ 三种编译模式、GPU 架构自动检测（lspci→mp_xx）、公共库 `benchmark_common`、ELF/PTX 辅助文件、bigModule 产物、批量运行与评分脚本流水线 |
| | [Green Context 代码解读](musa_benchmarks/green-context/代码解读.md)：`gree_context_test.mu` 669 行逐段分析，两个测试（SM Provisioning + 隔离性能验证），MP 独占（192 KiB smem）、host-GPU 同步、DCE 防护、三选一 PASS 判定 |
| | [Green Context 性能看护设计](musa_benchmarks/green-context/GreenContext-性能看护用例设计.md)：4 个回归 case（创建开销/分区精确性/launch 延迟/隔离性能），完整的 Celero 代码模板、8 步上库流程（创建→autorun.py→TestSuitConfig→基线→验证→提交） |
| **关键词** | Celero, benchmark, 性能回归, 评分系统, Green Context, SM 分区, UDM, CSV baseline |

---

## 后续扩展

本目录将持续扩展。计划中的洞察方向：

- **Kernel Mode Driver (linux-ddk/gr-kmd/)** — 内核侧 GPU 内存管理、调度器
- **SGLang MUSA 后端** — LLM 推理引擎的 MUSA 适配层 (sglang/sgl-kernel/)
- **Conformance Test Suite (musa_cts/)** — 测试架构与覆盖分析
- **msight-compute 深度分析** — MUPTI callback 时序、App Replay 机制、metrics YAML 规则引擎
