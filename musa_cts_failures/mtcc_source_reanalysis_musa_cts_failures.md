# 基于 MTCC 源码复核的 musa_cts 两个失败用例分析

## 1. 背景

本次复核对象是远程 MTCC 源码仓库：

```text
/home/shanfeng/workspace/mtcc
```

复核目标是重新分析 musa_cts 中两个失败 case：

```bash
TEST_TYPE=dailyM3d pytest . -v -k "test_musa_mtcc_cuda_builtin_address_space_conversion[__cvta_grid_constant_to_generic]"
```

以及：

```bash
TEST_TYPE=dailyM3d pytest . -v -k "test_musa_mtcc_cuda_compatible_basic_math_api[compatible_signbitl]"
```

这次重点不是只看当前 `/usr/local/musa` 安装版，而是对比：

```text
当前 CTS 实际使用的安装版 MTCC/MUSA headers
vs
/home/shanfeng/workspace/mtcc 源码 master 中已有实现
```

最终结论是：

```text
grid_constant 问题：mtcc master 仍缺 CUDA 12.8 用户可见 builtin API wrapper。
compatible_signbitl 问题：mtcc master 已经包含关键修复，但当前 /usr/local/musa 安装版没有同步这些 header。
```

---

## 2. musa_cts MTCC case 的整体协作流程

一个 MTCC case 的运行链路如下：

```text
pytest
  -> test_musa_mtcc.py
    -> 读取 musa_mtcc_test_cfg.csv
      -> 根据 TEST_TYPE 和 -k 选择 case
        -> 进入 case 源码目录
          -> 调用 mcc 编译 .cu
            -> host compilation
            -> device compilation
            -> include CTS common headers
            -> include MUSA runtime headers
            -> include clang MUSA resource headers / device wrapper
            -> LLVM IR / MTGPU backend lowering
            -> host link
          -> 生成 executable
            -> 运行 gtest
              -> 读取 golden
                -> 比较 MUSA 输出和 CUDA golden
```

pytest 看到的：

```text
AssertionError: run fail!
```

只是外层结果。真实根因需要看：

```text
compile log
run log
```

---

## 3. 相关组件职责

| 组件 | 职责 | 这两个问题中的作用 |
| --- | --- | --- |
| `pytest/test_mtcc/test_musa_mtcc.py` | 选择 case、拼接 mcc 编译命令、运行二进制、收集日志 | 不是根因，只负责调度和报告失败 |
| `musa_mtcc_test_cfg.csv` | 定义 case 名、源码目录、编译参数、架构参数 | 提供 `-mtgpu -O2 -lmusart` 等参数 |
| `musa_cts/mtcc/common/*.h` | 公共数据初始化、边界值、golden 比较、打印 | `compatible_signbitl` 会间接 include `quadmath.h` |
| `case .cu` | 具体测试 CUDA 兼容 API | 分别调用 grid_constant builtin 和 `__signbitl` |
| `mcc` | MTCC 编译驱动 | 组织 host/device 编译和链接 |
| clang MUSA resource headers | CUDA/MUSA device API wrapper | `__signbitl`、`quadmath.h`、address-space builtin 都在这一层相关 |
| MTCC frontend/Sema/CodeGen | 识别语言属性、生成 metadata/IR | 已支持 `__grid_constant__` 属性 |
| MTGPU backend | 后端 lowering 和代码生成 | 已有 grid_constant 参数 lowering 的基础逻辑 |
| `libmusart` | MUSA runtime | 不是这两个 case 的直接根因 |
| gtest/golden | 执行测试并比较结果 | `compatible_signbitl` 还存在 NaN golden 语义需复核 |

---

# 4. 问题一：grid_constant address-space conversion

失败 case：

```text
test_musa_mtcc_cuda_builtin_address_space_conversion[__cvta_grid_constant_to_generic]
```

相关源码使用了：

```cpp
__grid_constant__
__isGridConstant(...)
__cvta_grid_constant_to_generic(...)
__cvta_generic_to_grid_constant(...)
```

## 4.1 当前安装版失败点

当前 `/usr/local/musa` 安装版中可以看到：

```cpp
#define __grid_constant__ __location__(grid_constant)
```

说明安装版已经认识 `__grid_constant__` 这个参数属性。

但是安装版缺少以下 CUDA 12.8 用户可见 API：

```cpp
__isGridConstant
__cvta_grid_constant_to_generic
__cvta_generic_to_grid_constant
```

所以编译报错类似：

```text
use of undeclared identifier '__isGridConstant'
use of undeclared identifier '__cvta_grid_constant_to_generic'
```

最早失败链路是：

```text
case .cu
  -> 调用 CUDA 12.8 grid_constant builtin
    -> MUSA/MTCC header 中没有声明这些名字
      -> clang 名字解析失败
        -> 编译终止
```

---

## 4.2 mtcc 源码中已有的 grid_constant 支持

在 `/home/shanfeng/workspace/mtcc` 源码中，已经有 `__grid_constant__` 参数属性的基础支持。

### 4.2.1 clang attribute 定义

源码位置：

```text
llvmsrc/clang/include/clang/Basic/Attr.td
```

可以看到类似定义：

```cpp
def MTGPUGridConstant : InheritableAttr {
  let Spellings = [GNU<"grid_constant">, Declspec<"__grid_constant__">];
  let Subjects = SubjectList<[ParmVar]>;
  let LangOpts = [MUSA];
}
```

含义：

```text
MTCC clang frontend 能识别 __grid_constant__ / grid_constant attribute。
```

### 4.2.2 Sema 接收属性

源码位置：

```text
llvmsrc/clang/lib/Sema/SemaDeclAttr.cpp
```

有 handler 将属性挂到参数声明上：

```cpp
static void handleMTGPUGridConstantAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  if (D->isInvalidDecl())
    return;
  D->addAttr(::new (S.Context) MTGPUGridConstantAttr(S.Context, AL));
}
```

含义：

```text
语义分析阶段不会拒绝该属性，而是把它保存在 kernel 参数上。
```

### 4.2.3 CodeGen 生成 metadata

源码位置：

```text
llvmsrc/clang/lib/CodeGen/TargetInfo.cpp
```

相关逻辑会收集带有 `MTGPUGridConstantAttr` 的 kernel 参数，并写入 `musa.annotations` metadata：

```cpp
if (IV.value()->hasAttr<MTGPUGridConstantAttr>())
  GCI.push_back(IV.index());
```

并生成：

```text
grid_constant metadata
```

含义：

```text
frontend/codegen 已经能把“哪个 kernel 参数是 grid_constant”传给后端。
```

### 4.2.4 MTGPU backend LowerArgs 支持

源码位置：

```text
llvmsrc/llvm/lib/Target/MTGPU/MTGPULowerArgs.cpp
```

可以看到对 `isParamGridConstant(*Arg)` 的特殊处理。

含义：

```text
后端知道 grid_constant 参数，并能在参数 lowering 阶段把它当作 parameter/constant address space 处理。
```

---

## 4.3 mtcc 源码中仍缺的部分

虽然 mtcc master 已有：

```text
__grid_constant__ attribute
Sema attribute handling
CodeGen metadata
MTGPU LowerArgs support
```

但是没有找到以下用户可见 CUDA API 的实现入口：

```cpp
__isGridConstant
__cvta_grid_constant_to_generic
__cvta_generic_to_grid_constant
```

也就是说，源码中已有的是：

```text
语言属性和后端参数处理能力
```

缺的是：

```text
CUDA 12.8 address-space builtin API surface
```

完整链路应该是：

```text
用户代码调用：
  __isGridConstant
  __cvta_grid_constant_to_generic
  __cvta_generic_to_grid_constant

clang MUSA wrapper 声明这些函数
  -> 映射到 builtin 或 intrinsic
    -> 生成 LLVM IR intrinsic
      -> MTGPU backend lowering
        -> 生成正确 device code
```

当前 mtcc master 只完成了：

```text
__grid_constant__ 参数属性
  -> clang Sema
    -> CodeGen metadata
      -> backend LowerArgs
```

还没有完成：

```text
__isGridConstant / __cvta_*grid_constant* 函数入口
```

---

## 4.4 grid_constant 的最终归因

```text
出问题组件：MTCC clang MUSA CUDA-compatible builtin wrapper/API 层。
```

不是：

```text
pytest
musa_cts common header
gtest
libmusart
```

也不是完全没有 grid_constant 基础能力，因为 mtcc master 已经支持 `__grid_constant__` 参数属性和后端参数 lowering。

准确说法是：

```text
MTCC 已支持 grid_constant 参数属性，但还缺 CUDA 12.8 用户可见 address-space conversion/predicate builtin API。
```

---

# 5. 问题二：compatible_signbitl

失败 case：

```text
test_musa_mtcc_cuda_compatible_basic_math_api[compatible_signbitl]
```

case 核心代码：

```cpp
__global__ void kernel_signbitl(const double* inPut, int* outPut)
{
    outPut[tid] = __signbitl(inPut[tid]);
}
```

注意：

```text
输入类型是 double，但调用的是 __signbitl。
```

---

## 5.1 当前安装版第一层失败：quadmath.h not found

原始 compile log：

```text
../../common/init_data.h:10:10: fatal error: 'quadmath.h' file not found
#include <quadmath.h>
```

对应链路：

```text
compatible_signbitl.cu
  -> include ../../common/init_data.h
    -> init_data.h include <quadmath.h>
      -> 当前 mcc include path 找不到 quadmath.h
        -> 编译终止
```

这一步还没有真正走到 `__signbitl` 本身。

当前安装版中缺少：

```text
/usr/local/musa-5.1.0/lib/clang/20/include/quadmath.h
/usr/local/musa/lib/clang/20/include/quadmath.h
```

所以当前安装版需要依赖系统 GCC 私有路径：

```text
/usr/lib/gcc/x86_64-linux-gnu/11/include/quadmath.h
/usr/lib/gcc/x86_64-linux-gnu/12/include/quadmath.h
```

但 mcc 默认编译命令没有带这些路径。

---

## 5.2 mtcc 源码中已有 quadmath.h wrapper

在 mtcc master 中已经有：

```text
llvmsrc/clang/lib/Headers/quadmath.h
```

并且 CMake header 列表中已经包含：

```text
llvmsrc/clang/lib/Headers/CMakeLists.txt: quadmath.h
```

这个 `quadmath.h` 是一个轻量兼容 wrapper，用于解决 MUSA/CUDA style device compilation 下不可靠依赖系统 `libquadmath` 的问题。

它提供了 CTS 常用的 fp128 常量和 helper，例如：

```text
FLT128_MAX
FLT128_MIN
FLT128_EPSILON
FLT128_DENORM_MIN
HUGE_VALQ
M_PIq
nextafterq
signbitq
isnanq
isinfq
```

因此：

```text
quadmath.h not found 这个问题，在 mtcc master 源码中大概率已经修复。
```

当前 CTS 报错说明：

```text
实际运行 CTS 的 /usr/local/musa 安装版没有同步 mtcc master 中的 quadmath.h。
```

---

## 5.3 当前安装版第二层失败：device 端缺少 __signbitl

当手动补 include path 后，会继续暴露 `__signbitl` 问题：

```text
candidate function not viable: call to __host__ function from __global__ function
```

原因是当前安装版 device wrapper 中只有：

```cpp
__DEVICE__ bool __signbitd(double __a) { return __mt_signbit_f64(__a); }
__DEVICE__ bool __signbitf(float __a) { return __mt_signbit_f32(__a); }
```

缺少：

```cpp
__signbitl
__signbit
```

所以 device kernel 中调用：

```cpp
__signbitl(inPut[tid])
```

会解析到 host glibc/math 声明，最终报：

```text
从 __global__ / device 函数调用 host 函数非法
```

失败链路是：

```text
kernel device code
  -> 调用 __signbitl
    -> clang MUSA device wrapper 没有 device __signbitl
      -> 名字解析落到 host math declaration
        -> device 调 host 函数非法
          -> 编译失败
```

---

## 5.4 mtcc 源码中已有 __signbitl device wrapper

在 mtcc master 中，源码位置：

```text
llvmsrc/clang/lib/Headers/__clang_musa_device_functions.h
```

已经有类似实现：

```cpp
__DEVICE__ bool __signbitd(double __a) { return __mt_signbit_f64(__a); }
__DEVICE__ bool __signbit(double __a) { return __mt_signbit_f64(__a); }
__DEVICE__ bool __signbitf(float __a) { return __mt_signbit_f32(__a); }
__DEVICE__ bool __signbitl(long double __a) {
  return __signbitd(static_cast<double>(__a));
}
```

同时底层 libdevice declare 中已有：

```cpp
__DEVICE__ bool __mt_signbit_f64(double __a);
__DEVICE__ bool __mt_signbit_f32(float __a);
```

所以 `__signbitl` 对应的 device wrapper 在 mtcc master 中已经补上。

当前安装版缺少这些内容，说明：

```text
compatible_signbitl 当前失败主要是安装版 headers 旧，未同步 mtcc master 的修复。
```

---

## 5.5 compatible_signbitl 的剩余语义风险

之前用 overlay 临时映射：

```cpp
static __device__ __forceinline__ int __signbitl(double x) {
    return __signbitd(x);
}
```

可以让编译和大部分运行通过，但还剩一个边界值失败：

```text
input = 0x7FF8000000000000
黄金值 = 1
MUSA 输出 = 0
```

也就是 positive quiet NaN 的 signbit 行为和 golden 不一致。

因此即使用 mtcc master 的 `__signbitl(long double)` 修复了编译，还需要进一步验证：

```text
CUDA 12.8 对 0x7FF8000000000000 输入下 __signbitl 的真实返回值。
```

判断逻辑：

```text
如果 CUDA 12.8 返回 1：
  MUSA __signbitl 不能简单转成 __signbitd，需要特殊对齐 CUDA 语义。

如果 CUDA 12.8 返回 0：
  当前 CTS golden 数据可能有问题，需要重新生成或修正。
```

---

# 6. 两个问题的源码视角最终对比

## 6.1 grid_constant

mtcc master 状态：

```text
已有：
  __grid_constant__ attribute
  Sema attribute handling
  CodeGen metadata
  MTGPU LowerArgs support

缺少：
  __isGridConstant
  __cvta_grid_constant_to_generic
  __cvta_generic_to_grid_constant
```

结论：

```text
grid_constant 仍是 mtcc master 源码层面的 CUDA-compatible API 缺口。
```

准确归因：

```text
MTCC clang MUSA CUDA-compatible builtin wrapper/API 层缺失。
```

---

## 6.2 compatible_signbitl

mtcc master 状态：

```text
已有：
  llvmsrc/clang/lib/Headers/quadmath.h
  __signbitd
  __signbit
  __signbitf
  __signbitl(long double)
```

当前 `/usr/local/musa` 安装版状态：

```text
缺少：
  clang resource dir 下的 quadmath.h
  device wrapper 中的 __signbitl
  device wrapper 中的 __signbit
```

结论：

```text
compatible_signbitl 当前失败主要是安装版 MTCC/MUSA headers 没有同步 mtcc master 的修复。
```

准确归因：

```text
当前 /usr/local/musa-5.1.0 的 clang resource headers 版本落后于 mtcc master。
```

---

# 7. 推荐修复路径

## 7.1 grid_constant 修复路径

需要补完整 CUDA 12.8 用户可见 API 链路：

```text
Header / wrapper:
  __isGridConstant
  __cvta_grid_constant_to_generic
  __cvta_generic_to_grid_constant

Frontend / builtin:
  将 wrapper 映射到 builtin 或 LLVM intrinsic

LLVM IR:
  定义或复用 param/grid_constant address-space intrinsic

MTGPU backend:
  将 intrinsic lowering 到已有 parameter/grid_constant address-space 机制
```

短期只为了让当前 CTS case 通过，可以提供 header-only fallback：

```cpp
static __host__ __device__ __forceinline__ unsigned int __isGridConstant(const void *ptr) {
    return ptr != nullptr;
}

static __host__ __device__ __forceinline__ size_t __cvta_generic_to_grid_constant(const void *ptr) {
    return reinterpret_cast<size_t>(ptr);
}

static __host__ __device__ __forceinline__ void *__cvta_grid_constant_to_generic(size_t rawbits) {
    return reinterpret_cast<void *>(rawbits);
}
```

但这只是兼容当前 CTS offset 类检查，不是完整语义实现。

完整实现仍需要接到 clang wrapper + intrinsic + backend lowering。

---

## 7.2 compatible_signbitl 修复路径

优先动作：

```text
用 /home/shanfeng/workspace/mtcc master 构建出的 mcc/headers 重新跑 compatible_signbitl。
```

因为 master 已经包含：

```text
quadmath.h wrapper
__signbitl device wrapper
```

如果只修安装版，可以同步以下文件能力：

```text
llvmsrc/clang/lib/Headers/quadmath.h
llvmsrc/clang/lib/Headers/__clang_musa_device_functions.h 中的 __signbit / __signbitl
```

同时建议确认是否需要额外提供 double overload：

```cpp
__DEVICE__ bool __signbitl(double __a) {
  return __signbitd(__a);
}
```

原因是当前 CTS 实际传入的是 `double`，而不是 `long double`。

最后还需要单独确认 NaN 语义：

```text
0x7FF8000000000000 的 CUDA 12.8 __signbitl 返回值
```

---

# 8. 最终结论

这两个问题不是同一类问题。

```text
grid_constant：
  mtcc master 已有 __grid_constant__ 参数属性和后端参数 lowering，
  但仍缺 CUDA 12.8 用户可见 API：
    __isGridConstant
    __cvta_grid_constant_to_generic
    __cvta_generic_to_grid_constant
  所以这是源码 master 中仍存在的 CUDA-compatible API surface 缺口。
```

```text
compatible_signbitl：
  mtcc master 已经有 quadmath.h wrapper 和 __signbitl device wrapper，
  但当前 CTS 使用的 /usr/local/musa-5.1.0 安装版没有同步这些 header，
  所以当前失败主要是安装版与 mtcc master 不一致。
  编译问题修复后，还需要复核 positive quiet NaN 的 golden 语义。
```

一句话总结：

```text
grid_constant 要继续在 mtcc 源码里补 CUDA 兼容 API；
compatible_signbitl 要先把 mtcc master 中已有的 header 修复同步到当前安装版，再复核 NaN 语义。
```
