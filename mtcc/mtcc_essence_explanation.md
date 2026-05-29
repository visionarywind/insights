# MTCC 本质解释

## 1. 一句话定义

MTCC 本质上是 **摩尔线程 GPU 的完整编译器工具链**，不是单纯一个 `mcc` 命令。

可以概括为：

```text
MTCC = Clang/LLVM fork + MUSA CUDA 兼容前端 + MTGPU 后端 + device library + runtime compilation + 图形 shader 编译栈
```

从用户视角看，最常见入口是：

```bash
mcc xxx.cu -mtgpu --cuda-gpu-arch=mp_31
```

所以直觉上可以把 `mcc` 理解为 MUSA 版本的 `nvcc`。

但更准确地说：

```text
mcc 是 MTCC 对外暴露的编译驱动；
MTCC 是 mcc 背后的整套编译系统。
```

---

## 2. 从编译链路看 MTCC

一个 `.cu` 文件经过 MTCC，大致流程是：

```text
用户 CUDA/MUSA C++ 代码
  |
  v
mcc driver
  |
  +--> host code 交给 x86/gcc/clang 编译
  |
  +--> device code 交给 Clang MUSA frontend
         |
         v
      Clang 解析 __global__ / __device__ / __shared__ / __grid_constant__
         |
         v
      生成 LLVM IR
         |
         v
      调用 MUSA headers / libdevice / intrinsic
         |
         v
      MTGPU backend lowering
         |
         v
      生成 MTGPU device object
         |
         v
      打包成 MUSA fatbin
         |
         v
      host object 嵌入 device fatbin
         |
         v
      链接 libmusart
         |
         v
      最终可执行文件
```

所以 MTCC 的本质是：

```text
把 CUDA-like / MUSA C++ 程序编译成摩尔线程 GPU 可执行代码的全链路工具链。
```

---

## 3. 从源码结构看 MTCC

远程仓库：

```text
/home/shanfeng/workspace/mtcc
```

核心目录包括：

```text
llvmsrc/
libdevice/
libmusacxx/
libmtrtc/
libgfxc/
libclc/
libcxx/
utils/
mtcc_build.py
```

这些目录分别承担不同职责。

---

## 4. llvmsrc：MTCC 主体

```text
llvmsrc/
```

这是 MTCC 的主体，里面是 LLVM/Clang fork。

它包含：

```text
C/C++ frontend
MUSA/CUDA-like frontend
LLVM IR
优化 pass
MTGPU backend
lld
clang tools
MLIR 等 LLVM 子项目
```

也就是说，MTCC 不是从零写一个编译器，而是在 LLVM/Clang 基础上扩展 MUSA/MTGPU 能力。

---

## 5. mcc driver 与 MUSA toolchain

关键文件：

```text
llvmsrc/clang/lib/Driver/ToolChains/Musa.cpp
llvmsrc/clang/lib/Driver/Driver.cpp
llvmsrc/clang/lib/Driver/ToolChains/Clang.cpp
```

这些文件负责 `mcc` 的 driver 行为。

主要职责：

```text
识别 .cu / MUSA 编译输入
查找 /usr/local/musa 安装路径
添加 MUSA include/lib path
注入 __clang_musa_runtime_wrapper.h
处理 --cuda-gpu-arch=mp_31
创建 host/device offload action
处理 device compilation
处理 fatbin 打包
处理 -rdc=true / -dc
最终链接 libmusart
```

抽象流程：

```text
mcc
  -> Clang driver
    -> host toolchain
    -> MUSA device toolchain
      -> MTGPU backend
    -> MUSA fatbin
    -> host executable
```

---

## 6. Clang MUSA resource headers：CUDA 兼容入口

关键目录：

```text
llvmsrc/clang/lib/Headers/
```

MUSA 相关 headers 包括：

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

这些文件是 CUDA 兼容 API 的第一入口。

典型链路是：

```text
CUDA 用户 API 名字
  -> __clang_musa_* header inline wrapper
    -> __musa_* builtin / __mt_* libdevice function
      -> LLVM intrinsic 或 libdevice call
        -> MTGPU backend lowering
```

例如：

```text
atomicAdd
  -> 判断 pointer address space
    -> __musa_isspacep_shared / __musa_isspacep_global
      -> __musa_ptr_gen_to_shared / __musa_ptr_gen_to_global
        -> shared/global atomic intrinsic
```

这也是为什么很多 CTS 问题首先要看这些 headers。

---

## 7. MTGPU backend：MTCC 的核心价值

关键目录：

```text
llvmsrc/llvm/lib/Target/MTGPU
```

这是 MTCC 最核心的部分之一，负责把 LLVM IR 降到摩尔线程 GPU 指令和 object。

主要职责：

```text
LLVM IR -> MTGPU 指令
地址空间 lowering
kernel 参数 lowering
shared/local/global/constant 处理
atomic lowering
intrinsic lowering
寄存器分配
指令选择
调度
fence / memory model
hazard 处理
ELF/object 生成
```

仓库里还能看到两套后端体系：

```text
llvmsrc/llvm/lib/Target/MTGPU
llvmsrc/llvm/lib/Target/MTGPU/HG
```

说明不同硬件代际或产品线可能有不同 lowering、ISA、调度和 hazard 逻辑。

因此，一个 CUDA-compatible API 是否真正可用，不能只看 header 是否声明，还要看：

```text
MTGPU backend 是否能 lower
HG backend 是否也能 lower
目标 arch 是否支持
```

---

## 8. libdevice：device 数学和底层函数库

目录：

```text
libdevice/
```

以及声明入口：

```text
llvmsrc/clang/lib/Headers/__clang_musa_libdevice_declares.h
```

它负责提供 device 端数学函数和底层 helper。

例如：

```text
__mt_signbit_f64
__mt_signbit_f32
sin/cos/exp/sqrt 等 device math
```

以 `signbit` 为例，链路是：

```text
用户调用 __signbitd / __signbitl
  -> __clang_musa_device_functions.h wrapper
    -> __mt_signbit_f64
      -> libdevice / backend 实现
```

所以数学 API 兼容问题通常要同时检查：

```text
用户 API wrapper 是否存在
libdevice declare 是否存在
底层实现是否存在
backend 是否能 lower/link
```

---

## 9. libmusacxx：MUSA/CUDA C++ 高层 headers

目录：

```text
libmusacxx/include
```

它提供更高层的 MUSA/CUDA C++ header 能力，例如：

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

它更像 CUDA C++ library / CCCL 兼容层。

例如 `musa/__memory/address_space.h` 中可以看到：

```text
global
shared
constant
local
grid_constant
cluster_shared
```

但是如果底层 `__clang_musa_*` headers 或 backend 没有实现对应 builtin，那么高层 libmusacxx 只能预留接口，不能真正工作。

---

## 10. libmtrtc：运行时编译库

目录：

```text
libmtrtc/
```

它类似 NVIDIA NVRTC，提供运行时编译能力。

典型 API：

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

也就是说，MTCC 不只支持：

```bash
mcc xxx.cu
```

还支持程序运行时把字符串源码编译成 MUSA fatbin。

因此新增 CUDA-compatible API 时，最好同时验证：

```text
离线 mcc 编译
MTRTC runtime compilation
```

---

## 11. libgfxc：图形 shader 编译栈

目录：

```text
libgfxc/
```

它覆盖图形和 shader 编译相关能力：

```text
Vulkan
OpenGL / GLES
D3D
SPIR-V lowering
ray tracing / ray query
texture/image
shader cache
图形 pipeline 优化
```

这说明 MTCC 仓库不只服务 MUSA compute，也服务图形编译栈。

后端 pass 的改动可能同时影响：

```text
MUSA compute CTS
Vulkan/OpenGL shader
ray tracing shader
图形驱动编译路径
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

这些都是 compute 和 graphics 可能共用的底层能力。

---

## 12. 与 NVIDIA 工具链类比

可以粗略类比为：

```text
NVIDIA:
  nvcc
  clang CUDA frontend
  NVVM IR
  libdevice
  ptxas / SASS backend
  fatbin
  nvrtc

摩尔线程:
  mcc
  clang MUSA frontend
  LLVM IR
  MUSA libdevice
  MTGPU backend
  MUSA fatbin
  mtrtc
```

所以：

```text
MTCC 对标的是 NVIDIA CUDA 编译工具链，而不是单个编译器二进制。
```

---

## 13. 为什么 musa_cts 问题会落到 MTCC

musa_cts 测的是：

```text
CUDA 兼容语法/API 在 MUSA 上能不能编译、运行、结果一致。
```

这类问题天然跨多个层：

```text
case .cu
  -> MUSA runtime headers
    -> __clang_musa_* resource headers
      -> libdevice declare
        -> LLVM intrinsic
          -> MTGPU backend lowering
            -> runtime execution
              -> CUDA golden 对比
```

### 13.1 compatible_signbitl

本质是 CUDA-compatible math API 兼容问题。

涉及：

```text
__clang_musa_device_functions.h
quadmath.h
libdevice __mt_signbit_f64
CTS common headers
```

当前看到的关键点是：

```text
源码 master 中已有 __signbitl / quadmath.h 修复痕迹；
安装版可能没有同步，或者 5.1.0 的 quadmath.h 没有正确隔离 device target 不支持 __float128 的场景。
```

### 13.2 grid_constant

本质是 CUDA 12.8 address-space builtin API surface 不完整。

涉及：

```text
__grid_constant__ attribute
Clang Sema
CodeGen metadata
MTGPU LowerArgs
__isGridConstant wrapper
__cvta_grid_constant_to_generic wrapper
backend lowering
```

源码中已经有一部分：

```text
__grid_constant__ 属性和后端参数 lowering
```

但缺用户可见 API：

```text
__isGridConstant
__cvta_grid_constant_to_generic
__cvta_generic_to_grid_constant
```

所以 CTS 编译不过。

---

## 14. 最终总结

MTCC 的本质可以分三层理解：

```text
对用户：
  mcc 是 MUSA 版本的 CUDA-like 编译器入口。

对编译链路：
  MTCC 把 host/device 代码拆分、编译、lowering、打包 fatbin、链接成可执行文件。

对源码架构：
  MTCC 是 Clang/LLVM fork + MTGPU backend + MUSA CUDA-compatible headers + libdevice + libmusacxx + MTRTC + libgfxc 的完整 GPU 编译器栈。
```

最短总结：

```text
mcc 是命令；
MTCC 是完整编译器栈；
MTGPU backend 是核心；
__clang_musa_* headers 是 CUDA 兼容入口；
libdevice/libmusacxx/MTRTC/libgfxc 是配套生态。
```
