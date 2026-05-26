# MUSA Workspace 技术洞察报告

> 本目录汇集对 workspace 内各项目的源码级技术分析报告，按主题领域归档。
> 分析日期：2026-05-25

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

---

## 后续扩展

本目录将持续扩展。计划中的洞察方向：

- **MUSA Driver (linux-ddk/musa/)** — Driver API 入口、HAL 层、m3d 实现
- **Kernel Mode Driver (linux-ddk/gr-kmd/)** — 内核侧 GPU 内存管理、调度器
- **SGLang MUSA 后端** — LLM 推理引擎的 MUSA 适配层 (sglang/sgl-kernel/)
- **Conformance Test Suite (musa_cts/)** — 测试架构与覆盖分析
