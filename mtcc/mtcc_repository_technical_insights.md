# MTCC 仓库技术洞察

## 1. 仓库定位

远程仓库：

```text
/home/shanfeng/workspace/mtcc
```

当前 HEAD：

```text
43dd983439 (HEAD -> master, tag: 20260528_master, origin/master, origin/HEAD)
Merge branch 'pr/weather/memorymodel' into 'master'
```

从目录结构和源码入口看，`mtcc` 不是一个单独的 `mcc` driver 仓库，而是一个完整 GPU 编译器栈仓库，包含：

```text
llvmsrc/        LLVM/Clang fork，包含 Clang driver、frontend、LLVM IR、MTGPU backend、lld、mlir 等
libdevice/      device math/libdevice 实现
libmusacxx/     MUSA C++/CUDA-compatible device headers 和 C++ library headers
libmtrtc/       runtime compilation API，类似 NVRTC 的 MTRTC
libgfxc/        图形/Shader 编译相关组件，覆盖 Vulkan/OpenGL/D3D/SPIR-V/RT 等方向
libclc/         OpenCL/C language library 相关组件
libcxx/         C++ standard library 相关内容
utils/          install/use-local/clang-tidy 等工具脚本
mtcc_build.py   主构建脚本
```

技术上它承担三类任务：

```text
1. MUSA CUDA-like 编程模型编译：.cu / MUSA offload / mcc / device code / fatbin
2. MTGPU 后端代码生成：IR lowering、DAG ISel、寄存器分配、调度、fence、hazard、ELF/MC
3. 图形/运行时编译生态：libgfxc、MTRTC、shader pipeline、runtime compilation
```

---

## 2. 顶层构建模型

主构建入口是：

```text
mtcc_build.py
```

关键点：

```text
mtcc_build.py 会配置 LLVM_TARGETS_TO_BUILD。
```

已观察到目标组合包含：

```text
X86;MTGPU
AArch64;MTGPU
X86;NVPTX;MTGPU
AArch64;NVPTX;AMDGPU;MTGPU
X86;NVPTX;AMDGPU;MTGPU
```

说明该仓库不是只服务 MTGPU，也保留了 CUDA/NVPTX、AMDGPU、host targets 的构建能力，方便兼容、对照和工具链复用。

构建脚本还会打开：

```text
-DMLIR_ENABLE_MUSA_RUNNER=1
```

并且默认安装路径指向：

```text
/usr/local/musa
```

这解释了为什么 musa_cts 实际运行时会使用：

```text
/usr/local/musa-5.1.0/lib/clang/20/include
/usr/local/musa/include
/usr/local/musa/lib
```

也解释了之前 `musa_cts` 问题中出现的一个重要现象：

```text
mtcc master 源码中已经有的 header 修复，不一定已经出现在当前 /usr/local/musa 安装版中。
```

---

## 3. MUSA 编译链路

从 Clang driver 入口看，MUSA 编译链路核心在：

```text
llvmsrc/clang/lib/Driver/Driver.cpp
llvmsrc/clang/lib/Driver/ToolChains/Musa.cpp
llvmsrc/clang/lib/Driver/ToolChains/Clang.cpp
```

### 3.1 driver 创建 MUSA offload toolchain

`Driver.cpp` 中有 `OFK_MUSA` 相关逻辑，会为 MUSA device compilation 创建：

```text
toolchains::MusaToolChain
```

典型链路：

```text
mcc input .cu
  -> Clang driver 识别 MUSA/CUDA-like 输入
    -> 构造 host action
    -> 构造 MUSA device action
    -> device action 使用 MTGPU triple
    -> host/device 通过 offload action 组合
```

### 3.2 Musa.cpp 负责 MUSA 安装探测和 fatbin

关键文件：

```text
llvmsrc/clang/lib/Driver/ToolChains/Musa.cpp
```

其中有 MUSA 安装路径探测：

```text
MUSAInstallCandidates
MUSA_PATH
MUSA_PATH_V5_1
/usr/local/musa
```

也有强制注入 runtime wrapper：

```cpp
CC1Args.push_back("-include");
CC1Args.push_back("__clang_musa_runtime_wrapper.h");
```

这意味着所有 MUSA device 编译都高度依赖 Clang resource dir 中的 MUSA wrapper headers。

MUSA fatbin 由 `clang-offload-bundler` 生成，相关函数：

```text
constructMUSAFatbinCommand
constructGenerateObjectFileFromMUSAFatBinary
```

整体链路可以抽象为：

```text
.cu
  -> host cc1
  -> device cc1, triple mtgpu-mt-musa / mthg-mt-musa
  -> MTGPU backend object
  -> clang-offload-bundler 生成 .musafb
  -> 生成 host object 中嵌入 MUSA fatbin
  -> host linker 链接 libmusart
```

### 3.3 RDC 正在成为重点能力

近期提交中有：

```text
[SW-79060] Implement MUSA automatic RDC under -rdc=true/-dc
```

driver 中也可以看到：

```text
-fgpu-rdc
-rdc=true
-dc
RelocatableDeviceCode
```

这说明 MTCC 正在补齐 CUDA 兼容的 relocatable device code 能力。

RDC 是高风险区域，因为它会影响：

```text
device symbol visibility
libdevice internalization
device linking
fatbin generation
host object embedding
跨 TU device function resolution
```

之前 `compatible_signbitl` 这类 device wrapper 缺失问题，在 RDC 场景下会更容易暴露，因为 device function 的声明、定义、internalize 和链接边界更复杂。

---

## 4. Clang resource headers 是 CUDA 兼容层的第一关键面

关键目录：

```text
llvmsrc/clang/lib/Headers
```

MUSA 相关 headers：

```text
__clang_musa_runtime_wrapper.h
__clang_musa_device_functions.h
__clang_musa_intrinsics.h
__clang_musa_libdevice_declares.h
__clang_musa_math.h
__clang_musa_cmath.h
__clang_musa_builtin_vars.h
__clang_musa_builtin_forward_declares.h
__clang_musa_texture_intrinsics.h
__clang_musa_runtime_functions.h
__clang_musa_additional_math.h
quadmath.h
```

这些文件的角色是：

```text
CUDA/MUSA 用户 API 名字
  -> header inline wrapper
    -> __musa_* builtin / __mt_* libdevice function
      -> LLVM intrinsic 或 libdevice call
        -> MTGPU backend lowering
```

### 4.1 atomicAdd 体现了地址空间分流模型

在 `__clang_musa_device_functions.h` 中，atomic wrapper 会根据地址空间判断：

```cpp
if (__musa_isspacep_shared(__p))
  return __musa_shared_atom_add_gen_i(... __musa_ptr_gen_to_shared(...));
else
  return __musa_global_atom_add_gen_i(... __musa_ptr_gen_to_global(...));
```

这体现 MTCC 的一个核心设计：

```text
用户看到 generic pointer；
wrapper 通过 isspacep 判断实际 address space；
再通过 ptr_gen_to_* 转换成目标 address space；
最后调用 global/shared 对应 intrinsic。
```

这也解释了为什么之前 `grid_constant` 问题不能只看 `__grid_constant__` attribute：

```text
attribute 只解决 kernel 参数属于 grid_constant；
用户可见的 __isGridConstant / __cvta_* 仍需要 wrapper + intrinsic + backend lowering。
```

### 4.2 signbitl/quadmath 修复已经在源码中

源码 master 中可以看到：

```text
llvmsrc/clang/lib/Headers/__clang_musa_device_functions.h
  __signbitd(double)
  __signbit(double)
  __signbitf(float)
  __signbitl(long double)
```

也可以看到：

```text
llvmsrc/clang/lib/Headers/quadmath.h
llvmsrc/clang/lib/Headers/CMakeLists.txt: quadmath.h
```

这说明 `compatible_signbitl` 的编译问题，在源码 master 中已有明显修复痕迹。当前 `/usr/local/musa` 安装版缺这些 header，更像是安装包版本落后或未同步。

### 4.3 grid_constant 仍是不完整兼容面

源码 master 中可以看到：

```text
llvmsrc/clang/include/clang/Basic/Attr.td
  MTGPUGridConstant

llvmsrc/clang/lib/CodeGen/TargetInfo.cpp
  grid_constant metadata

llvmsrc/llvm/lib/Target/MTGPU/MTGPULowerArgs.cpp
  isParamGridConstant / LowerArgs support
```

但是没有找到完整用户可见 API：

```text
__isGridConstant
__cvta_grid_constant_to_generic
__cvta_generic_to_grid_constant
```

`libmusacxx/include/musa/__memory/address_space.h` 中还能看到注释式占位：

```text
no functional __isGridConstant()
```

这说明源码 master 的状态是：

```text
有 grid_constant 参数属性和后端参数 lowering；
缺 CUDA 12.8 grid_constant address conversion/predicate API。
```

---

## 5. MTGPU backend 架构

关键目录：

```text
llvmsrc/llvm/lib/Target/MTGPU
```

其下有两套明显后端体系：

```text
MTGPU/          通用或一代 MTGPU backend
MTGPU/HG/       HG 系列 backend
```

CMake 中分别 tablegen：

```text
MTGPU.td
HG/HG.td
```

生成：

```text
AsmMatcher
AsmWriter
DAGISel
DisassemblerTables
InstrInfo
RegisterInfo
SubtargetInfo
CallingConv
MCCodeEmitter
SearchableTables
```

这表明 MTGPU backend 是完整 LLVM target backend，而不是简单 PTX 翻译层。

### 5.1 backend pipeline 很长，覆盖从 IR 到机器码的完整流程

`MTGPUTargetMachine.cpp` 和 `HGTargetMachine.cpp` 中 pass pipeline 很密集，典型阶段包括：

```text
IR-level:
  UnsupportedOpToFuncCall
  UnifyMetadata
  Internalize
  AlwaysInline
  PrintfRuntime
  ShuffleOpt
  PromoteAlloca
  LowerAlloca
  InferAddressSpaces
  LowerArgs
  PTXLowering
  LowerIntrinsics
  LowerSharedMemory
  CodeGenPrepare
  AtomicOptimizer
  AnnotateKernelFeatures
  StructurizeCFG
  AnnotateUniformValues
  AnnotateControlFlow

ISel / MI-level:
  DAG ISel
  PseudoCustomElimination
  EarlyIfConversion / IfConversion
  LoadStoreOptimizer
  FoldOperands
  AtomicLegalizer
  Register allocation
  Slot register spilling
  PostRA hazard recognizer
  MemoryLegalizer
  FinalInstrProcess
  FenceSetting
  TCESyncFenceSetting
```

这说明 MTCC 的性能和正确性风险主要集中在：

```text
address space inference/lowering
kernel argument lowering
intrinsic lowering
memory model/fence setting
atomic lowering/legalization
register allocation and spilling
control-flow structurization
hazard recognition
instruction scheduling
```

### 5.2 MTGPU 和 HG 两套后端可能存在功能漂移风险

`MTGPU` 与 `HG` 目录下有大量同名/近似 pass：

```text
MTGPULowerArgs vs HGLowerArgs
MTGPULowerIntrinsics vs HGLowerIntrinsics
MTGPUPrintfRuntime vs HGPrintfRuntime
MTGPUShuffleOpt vs HGShuffleOpt
MTGPUCodeGenPrepare vs HGCodeGenPrepare
MTGPUMemoryLegalizer vs HGMemoryLegalizer
MTGPUFenceSetting vs HGFenceSetting
```

这类双后端结构的主要风险是：

```text
一个 feature 在 MTGPU 侧修了，HG 侧没修；
一个 CTS case 在 mp_31 过了，另一个 arch 失败；
header wrapper 暴露了统一 API，但 backend lowering 在不同 subtarget 上不一致。
```

因此补 CUDA-compatible API 时，不能只改 header，也不能只验证一个架构。

---

## 6. libdevice 的角色

目录：

```text
libdevice/
```

以及 Clang header 声明：

```text
llvmsrc/clang/lib/Headers/__clang_musa_libdevice_declares.h
```

从 `__signbitd` 例子看，链路是：

```cpp
__DEVICE__ bool __signbitd(double __a) { return __mt_signbit_f64(__a); }
```

其中：

```text
__mt_signbit_f64
__mt_signbit_f32
```

由 libdevice declare 暴露。

技术洞察：

```text
数学函数兼容问题通常不是单点问题。
```

至少要检查三层：

```text
1. 用户 API wrapper 是否存在：__signbitl / sin / cos / fp128 等
2. libdevice declare 是否存在：__mt_xxx
3. backend 是否能 lower 或 link 到对应 device implementation
```

之前 `compatible_signbitl` 就是典型：

```text
__mt_signbit_f64 底层已有；
__signbitd wrapper 已有；
__signbitl wrapper 安装版缺失；
源码 master 已补。
```

---

## 7. libmusacxx 的角色

目录：

```text
libmusacxx/include
```

它提供更高层的 MUSA/CUDA C++ headers，例如：

```text
musa_fp16.h / hpp
musa_bf16.h / hpp
musa_fp8 / fp6 / fp4
mma.h
sqmma.h
cooperative_groups.h
texture/surface headers
musa/__memory/address_space.h
musa/std/__cccl/*
```

从 `musa/__memory/address_space.h` 可以看到，libmusacxx 已经抽象出：

```text
global
shared
constant
local
grid_constant
cluster_shared
```

但 `grid_constant` 分支有：

```text
no functional __isGridConstant()
```

这说明 libmusacxx 对 CUDA 新语义的适配依赖底层 clang resource header 和 backend 完整支持。

技术判断：

```text
libmusacxx 更像 API/模板层；
真正 builtin 支持是否完整，要看 __clang_musa_* headers + LLVM intrinsic + backend lowering。
```

---

## 8. MTRTC 技术洞察

目录：

```text
libmtrtc
```

它提供类似 NVRTC 的 runtime compilation API：

```text
mtrtcCreateProgram
mtrtcCompileProgram
mtrtcAddNameExpression
mtrtcGetLoweredName
mtrtcGetFatBinSize
mtrtcGetFatBin
mtrtcGetProgramLogSize
mtrtcGetProgramLog
```

内部类包括：

```text
MTRTCProgram
MTRTCCompiler
MTRTCOptionParser
MTRTCCompileWorkspace
MTRTCToolchain
```

默认 arch：

```text
mp_31
```

输出类型包括：

```text
Fatbin
Relocatable
Executable
```

这说明 MTCC 不只是离线 mcc 编译器，还在补 runtime compilation 生态。

风险点：

```text
mcc 离线编译通过，不代表 MTRTC 通过；
MTRTC 需要复用同一套 headers、libdevice、arch mapping、fatbin/link 流程；
name expression 和 lowered name 还会引入 C++ mangling/模板实例化问题。
```

所以后续任何 CUDA-compatible API 修复，建议同时补：

```text
mcc CTS 验证
MTRTC compile 验证
如果涉及模板/overload，再补 lowered-name 验证
```

---

## 9. libgfxc 技术洞察

目录：

```text
libgfxc
```

从目录和近期提交看，libgfxc 覆盖：

```text
Vulkan
OpenGL / GLES
D3D
SPIR-V lowering
ray query / ray tracing
fragment output
texture/image
shader cache
PFO / scheduling / backend shared logic
```

近期提交中大量内容来自图形/RT：

```text
support 10-bits texture
GLES dual-source output
ray query / callable / any-hit attr
SPIR-V workgroup size
shader cache hash
Dota2 crash
```

技术判断：

```text
MTCC 仓库的 MTGPU backend 同时服务 compute 和 graphics。
```

这会带来一个重要影响：

```text
backend pass 的修改可能同时影响 MUSA compute CTS 和图形 shader 编译。
```

例如：

```text
memory model
fence setting
atomic lowering
control-flow divergence
register allocation
load/store scheduling
```

这些改动既可能修 compute benchmark，也可能影响 Vulkan/OpenGL shader。

---

## 10. 近期热点方向

从最近 30 天关键提交看，热点包括：

```text
memory model / fence / LSU scope
atomicAdd lowering，含 half/bfloat/vector atomicAdd
fast-math
SQMMA / matrix / fp4/fp8/fp128
MUSA automatic RDC
MTRTC APIs
texture / image / 10-bit texture
Vulkan/OpenGL/GLES/RT/rayquery/callable
Triton linking timeout / compatibility
shader cache / SPIR-V lowering
register / scheduling / hazard / wait bar mask
```

这说明当前仓库处于高频开发状态，主要战线是：

```text
1. CUDA-compatible compute API 补齐
2. 新硬件/新 ISA 能力使能
3. 图形 shader 编译正确性
4. device linking / runtime compilation 生态
5. 后端性能和稳定性
```

---

## 11. 与 musa_cts 两个失败问题的关联洞察

### 11.1 compatible_signbitl

源码 master 已有：

```text
quadmath.h
__signbitl(long double)
__signbit(double)
```

当前安装版缺失，说明这是：

```text
源码 master 与 /usr/local/musa 安装版不同步的问题。
```

后续处理优先级：

```text
1. 用 mtcc master 构建并安装/指定 resource dir 后复测。
2. 如果编译通过，再确认 NaN golden 语义。
3. 如仍有 overload 问题，补 __signbitl(double) wrapper。
```

### 11.2 grid_constant

源码 master 已有：

```text
__grid_constant__ attribute
CodeGen metadata
LowerArgs support
```

但缺：

```text
__isGridConstant
__cvta_grid_constant_to_generic
__cvta_generic_to_grid_constant
```

所以这是：

```text
源码 master 中仍存在的 CUDA 12.8 API surface 缺口。
```

后续应补完整链路：

```text
header wrapper
  -> builtin / intrinsic
    -> IR intrinsic
      -> MTGPU/HG backend lowering
        -> CTS + MTRTC + 多架构验证
```

---

## 12. 主要技术风险

### 12.1 Header/API surface 与 backend 能力不同步

典型表现：

```text
header 暴露了 API，但 backend 不能 lower；
backend 有能力，但 header 没暴露 CUDA-compatible 名字；
源码 master 修了，但安装版没同步。
```

`grid_constant` 和 `compatible_signbitl` 分别对应后两种情况。

### 12.2 双后端漂移

`MTGPU` 和 `HG` 两套 backend 都有大量相似 pass。

风险：

```text
feature 在一套 backend 通过，另一套不通过。
```

建议：

```text
重要 API 修复至少验证 mp_31 以及目标产品线对应 arch。
```

### 12.3 CUDA 兼容行为依赖 golden，语义边界要反查 CUDA

例如：

```text
positive quiet NaN 的 __signbitl 返回值
```

不能只靠推理或 MUSA 当前行为判断，需要使用目标 CUDA 版本复核 golden。

### 12.4 公共 CTS header 容易放大依赖

`compatible_signbitl` 本身不是 fp128 case，但因为公共头 include `quadmath.h` 并启用 `ENABLE_FLOAT128`，导致普通 math case 被 quadmath/fp128 依赖拖累。

这类问题容易误导根因分析。

### 12.5 RDC/MTRTC 会扩大编译边界复杂度

RDC 和 MTRTC 会引入：

```text
多 translation unit
device link
fatbin
symbol internalization
name expression
runtime option parsing
```

因此修复不能只覆盖单 TU mcc 编译。

---

## 13. 建议的技术看护清单

### 13.1 CUDA-compatible API 补齐看护

建立表格追踪：

```text
CUDA API 名字
MUSA header wrapper 是否存在
libdevice declare 是否存在
LLVM intrinsic 是否存在
MTGPU lowering 是否存在
HG lowering 是否存在
mcc CTS 是否通过
MTRTC 是否通过
CUDA golden 是否复核
```

优先覆盖：

```text
address-space conversion
math fp64/fp128/long double
atomic vector/half/bfloat
cooperative groups
texture/surface
mma/sqmma
```

### 13.2 安装版与源码版一致性看护

对 `/usr/local/musa` 安装版和 `mtcc master` 做 header diff，至少关注：

```text
__clang_musa_device_functions.h
__clang_musa_runtime_wrapper.h
__clang_musa_intrinsics.h
__clang_musa_libdevice_declares.h
__clang_musa_math.h
__clang_musa_cmath.h
quadmath.h
libmusacxx/include/musa/__memory/address_space.h
```

### 13.3 后端多架构验证

对涉及 lowering 的修复，建议验证：

```text
MTGPU backend
HG backend
mp_31 及目标产品 arch
RDC on/off
O2/O3/fast-math 组合
```

### 13.4 MTRTC 回归

对新增 CUDA-compatible API，至少增加：

```text
mtrtcCompileProgram
mtrtcGetFatBin
mtrtcAddNameExpression / mtrtcGetLoweredName，如果涉及模板或 overload
```

---

## 14. 总结

MTCC 仓库是完整 GPU 编译器栈，不是单一 mcc driver。它的核心结构是：

```text
Clang driver / MUSA offload
  -> Clang MUSA resource headers
    -> LLVM IR intrinsic / libdevice
      -> MTGPU/HG backend lowering
        -> object / fatbin / host embedding
          -> libmusart runtime execution
```

当前最值得关注的技术洞察是：

```text
1. Clang resource headers 是 CUDA 兼容问题的第一入口。
2. MTGPU/HG backend 是 correctness 和性能风险核心。
3. libdevice、libmusacxx、MTRTC 与 mcc driver 共享同一套能力边界。
4. 源码 master 与 /usr/local/musa 安装版可能不同步，分析 CTS 失败必须区分二者。
5. grid_constant 是源码层仍缺 API surface；compatible_signbitl 是安装版未同步源码修复。
6. 后续修复要按 header -> intrinsic/libdevice -> backend -> mcc CTS -> MTRTC -> 多架构验证的链路闭环。
```
