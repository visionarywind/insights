# compatible_signbitl 根因分析与组件依赖拆解

## 1. 问题现象

远程执行命令：

```bash
cd /home/shanfeng/workspace/musa_cts/pytest
TEST_TYPE=dailyM3d pytest . -v -k "test_musa_mtcc_cuda_compatible_basic_math_api[compatible_signbitl]"
```

pytest 只选中 1 个 case，最终失败：

```text
test_mtcc/test_musa_mtcc.py::test_musa_mtcc_cuda_compatible_basic_math_api[compatible_signbitl] FAILED
AssertionError: run fail!
```

pytest 层的 `assert result` 只是外层结果，不是直接根因。直接原因需要看 compile/run log。

## 2. pytest 实际做了什么

`pytest/test_mtcc/test_musa_mtcc.py` 根据 `musa_mtcc_test_cfg.csv` 找到 case：

```text
compatible_signbitl, mtcc_cuda_compatible_basic_math_api, cuda_compatible/basic_math_api, compatible_signbitl, -mtgpu -O2 -lmusart
```

然后进入源码目录，执行类似命令：

```bash
mcc compatible_signbitl.cu \
  -I/home/shanfeng/workspace/musa_cts/googletest/googletest/include/ \
  -L/home/shanfeng/workspace/musa_cts/build/gtest/lib/ \
  -lgtest \
  -o compatible_signbitl \
  -mtgpu -O2 -lmusart \
  --cuda-gpu-arch=mp_31
```

如果编译失败，就不会生成 `compatible_signbitl` 二进制，后面的 run 阶段会继续失败：

```text
chmod: cannot access 'compatible_signbitl': No such file or directory
```

因此需要先看 compile log。

## 3. 第一处失败：公共 CTS 头文件找不到 quadmath.h

原始 compile log：

```text
In file included from compatible_signbitl.cu:6:
./../../common/init_data.h:10:10: fatal error: 'quadmath.h' file not found
   10 | #include <quadmath.h>
      |          ^~~~~~~~~~~~
1 error generated when compiling for mp_31.
```

对应源码：

```cpp
// mtcc/common/init_data.h
#include <quadmath.h>
```

远程机器上 `quadmath.h` 实际存在：

```text
/usr/lib/gcc/x86_64-linux-gnu/11/include/quadmath.h
/usr/lib/gcc/x86_64-linux-gnu/12/include/quadmath.h
```

说明第一处问题不是系统缺包，而是当前 `mcc` 编译命令的 include search path 没有覆盖 GCC 私有 include 目录。

这一层失败发生在：

```text
pytest -> mcc 编译 host/device TU -> include mtcc/common/init_data.h -> 找 quadmath.h -> 找不到 -> 编译终止
```

此时还没有走到 `__signbitl` 数学 API 本身。

## 4. 第二处失败：补 include path 后暴露 fp128/quadmath 依赖

验证时增加：

```bash
-I/usr/lib/gcc/x86_64-linux-gnu/11/include
```

编译继续向前，但出现：

```text
./../../common/init_boundary_values.hpp:91:9: error: use of undeclared identifier '__builtin_huge_valq'
HUGE_VALQ
```

以及后续链接阶段：

```text
undefined reference to `nextafterq'
```

原因是公共头中有这条逻辑：

```cpp
// mtcc/common/helper_print.hpp
#if (defined(__SIZEOF_FLOAT128__) || defined(__FLOAT128__))
#define ENABLE_FLOAT128 1
#endif
```

一旦 clang/mcc 定义了 `__SIZEOF_FLOAT128__`，CTS 公共头就会启用 `__float128` 支持路径。随后：

```cpp
// mtcc/common/init_boundary_values.hpp
#ifdef ENABLE_FLOAT128
std::vector<__float128> get_boundary_values<__float128>() {
    return {
        HUGE_VALQ,
        FLT128_DENORM_MIN,
        nextafterq(...),
        ...
    };
}
#endif
```

这会引入两类依赖：

1. 编译期需要 `quadmath.h` 和 GCC/Clang 对 `__float128` builtin 的支持。
2. 链接期需要 `libquadmath`，否则 `nextafterq` 等符号未解析。

因此即使 `compatible_signbitl.cu` 本身不是 fp128 case，也会因为公共头无条件/间接启用 fp128 路径而依赖 quadmath。

这一步的问题发生在：

```text
公共 CTS helper 组件 -> ENABLE_FLOAT128 自动打开 -> init_boundary_values.hpp 解析 fp128 边界值代码 -> 需要 quadmath builtin/libquadmath -> 编译或链接失败
```

## 5. 第三处失败：MUSA device 端缺少 __signbitl 兼容实现

源码核心：

```cpp
// mtcc/cuda_compatible/basic_math_api/compatible_signbitl.cu
__global__ void kernel_signbitl(const double* inPut, int* outPut)
{
    ...
    outPut[tid] = __signbitl(inPut[tid]);
}
```

注意这里输入类型是 `double*`，但调用的是 CUDA 兼容 API `__signbitl`。

MUSA 当前 device wrapper 中有：

```cpp
// /usr/local/musa-5.1.0/lib/clang/20/include/__clang_musa_device_functions.h
__DEVICE__ bool __signbitd(double __a) { return __mt_signbit_f64(__a); }
__DEVICE__ bool __signbitf(float __a) { return __mt_signbit_f32(__a); }
```

但没有 device 端 `__signbitl`。

MUSA runtime 头里虽然能看到 `__signbitl(long double)` 声明或 host fallback 风格实现，但 clang device wrapper 没有把 CUDA 兼容的 `__signbitl` 映射到 MUSA device builtin。因此编译时 `__signbitl` 会解析到 glibc/math host 声明，报错：

```text
compatible_signbitl.cu:29:19: error: no matching function for call to '__signbitl'
/usr/include/.../mathcalls-helper-functions.h: note: candidate function not viable: call to __host__ function from __global__ function
```

这一层失败发生在：

```text
kernel device code -> 调用 __signbitl -> MUSA clang device wrapper 没有 device __signbitl -> 名字解析落到 host math 声明 -> device 调 host 函数非法 -> 编译失败
```

## 6. 临时验证过程

为了把问题逐层验证，临时注入 overlay：

```cpp
#pragma once
#ifndef __builtin_huge_valq
#define __builtin_huge_valq() ((__float128)1.0Q / (__float128)0.0Q)
#endif
#include <musa_runtime.h>
static __device__ __forceinline__ int __signbitl(double x) {
    return __signbitd(x);
}
```

执行：

```bash
TEST_TYPE=dailyM3d \
COMPILE_EXTRA_ARGS='-I/usr/lib/gcc/x86_64-linux-gnu/11/include -include /tmp/signbitl_compat_overlay.h -L/usr/local/musa/lib -lquadmath' \
pytest . -v -k 'test_musa_mtcc_cuda_compatible_basic_math_api[compatible_signbitl]'
```

验证结果：

1. `quadmath.h` 缺失问题被绕过。
2. `__builtin_huge_valq` 问题被绕过。
3. `nextafterq` 链接问题通过 `-lquadmath` 解决。
4. `__signbitl` device 编译问题通过映射到 `__signbitd` 解决。
5. 二进制成功生成并运行。
6. 36 个 gtest 子用例中 35 个通过，剩余 1 个边界值失败。

剩余运行期失败：

```text
Test Failed! print value: golden value = 1, musa value = 0
[DEBUG] Mismatch at index 0, input = 0x7FF8000000000000
```

即 positive quiet NaN 边界值上，golden 期望 `1`，当前 MUSA `__signbitd` 路径返回 `0`。

这说明简单把 `__signbitl(double)` 映射到 `__signbitd(double)` 可以打通编译和大部分运行，但还不一定完全符合当前 CTS golden 的 NaN 语义。

## 7. 组件依赖链路

完整链路如下：

```text
pytest 测试框架
  -> test_musa_mtcc.py
    -> 读取 musa_mtcc_test_cfg.csv
      -> 定位 compatible_signbitl.cu
        -> mcc 编译驱动
          -> host/device 编译
            -> include CTS common headers
              -> init_data.h
                -> quadmath.h
              -> init_boundary_values.hpp
                -> ENABLE_FLOAT128 / nextafterq / HUGE_VALQ
            -> include MUSA runtime headers
            -> include clang MUSA device wrapper
              -> __signbitd / __signbitf 存在
              -> __signbitl 缺失
          -> host 链接
            -> libgtest
            -> libmusart
            -> libquadmath，如果启用 fp128 路径
        -> 生成 compatible_signbitl 二进制
          -> 运行 gtest
            -> 读取 golden/test_signbitl/*.dat
            -> 比较 MUSA 输出和 CUDA golden
```

对应组件职责：

| 组件 | 职责 | 当前问题 |
| --- | --- | --- |
| pytest/test_musa_mtcc.py | 选择 case、拼 mcc 命令、收集 compile/run log | 本身不是根因，只是报告 `assert result` |
| musa_mtcc_test_cfg.csv | 定义 case 路径、编译参数、架构参数 | `compatible_signbitl` 只给了 `-mtgpu -O2 -lmusart`，没有 quadmath include/link 参数 |
| compatible_signbitl.cu | 测试 CUDA 兼容 `__signbitl` | 在 device kernel 中调用 `__signbitl(double)` |
| mtcc/common/init_data.h | 初始化数据公共头 | 无条件 include `<quadmath.h>`，导致非 fp128 case 也依赖 GCC 私有头 |
| mtcc/common/helper_print.hpp | 根据编译器宏打开 `ENABLE_FLOAT128` | 只要看到 `__SIZEOF_FLOAT128__` 就启用 fp128，可能过度打开 |
| mtcc/common/init_boundary_values.hpp | 生成边界值集合 | fp128 分支使用 `HUGE_VALQ/nextafterq`，需要 quadmath builtin 和 libquadmath |
| mcc 编译驱动 | 组织 host/device 编译和链接 | 默认 include path 未覆盖 `/usr/lib/gcc/.../include/quadmath.h`；默认链接未带 `-lquadmath` |
| clang MUSA device wrapper | 提供 device 端 CUDA/MUSA builtin 映射 | 有 `__signbitd/__signbitf`，缺 `__signbitl` |
| MUSA backend/libdevice | 实现底层 signbit 指令/函数 | `__signbitd` 已能落到 `__mt_signbit_f64`，但 `__signbitl` 没有 wrapper 入口 |
| golden 数据 | CUDA 期望结果 | NaN `0x7FF8000000000000` 期望为 1，需确认是否与 CUDA 12.8 实际一致 |

## 8. 到底是哪一步出问题

按最早失败点看：

```text
第一步出问题：mcc 编译 compatible_signbitl.cu 时 include common/init_data.h，找不到 quadmath.h。
```

按完整兼容目标看，有三步问题：

### 8.1 编译环境/include path 问题

```text
mcc 默认 include path 没找到 GCC 私有 quadmath.h。
```

这是当前 pytest 首次失败的直接原因。

### 8.2 CTS 公共头设计问题

```text
compatible_signbitl 本身不是 fp128 case，但公共头无条件包含 quadmath，并且自动启用 ENABLE_FLOAT128，导致它也依赖 quadmath 编译和链接。
```

这是问题被放大的原因。

### 8.3 MUSA CUDA 兼容 wrapper 问题

```text
device 端没有 __signbitl，导致 kernel 内调用 __signbitl 解析到 host math 函数。
```

这是数学 API 兼容层真正需要补的点。

### 8.4 NaN 语义问题

```text
临时映射 __signbitl -> __signbitd 后，positive quiet NaN 的输出和 golden 不一致。
```

这是功能语义是否完全兼容 CUDA 12.8 的剩余问题，需要单独用 CUDA 12.8 复核 golden。

## 9. 修复建议

### 9.1 CTS 侧短期修复

避免非 fp128 case 被 quadmath 拖累：

1. `init_data.h` 不要无条件 include `quadmath.h`。
2. `init_boundary_values.hpp` 只有在真正需要 fp128 case 时才 include `quadmath.h`。
3. fp128 case 的配置中显式带上：

```bash
-I/usr/lib/gcc/x86_64-linux-gnu/11/include -lquadmath
```

或者通过 CTS 编译配置统一给 fp128 suite 加这些参数。

### 9.2 mcc 驱动侧修复

如果目标是让 `mcc` 像系统 C++ 编译器一样支持 host 侧 `quadmath.h`，则需要：

1. host include search path 自动包含当前 GCC 的 private include 目录。
2. 当用户显式使用 quadmath API 时，可以正确链接 `libquadmath`，或者文档要求用户显式加 `-lquadmath`。

但不建议让所有普通 CUDA compatible math case 默认链接 `libquadmath`，否则会扩大依赖。

### 9.3 MUSA CUDA 兼容层修复

在 clang MUSA device wrapper 或对应 runtime device header 中补 `__signbitl`：

```cpp
__DEVICE__ int __signbitl(long double x) {
    return __signbitd(static_cast<double>(x));
}
```

考虑到当前 CTS 实际传入的是 `double`，还要确认是否需要补：

```cpp
__DEVICE__ int __signbitl(double x) {
    return __signbitd(x);
}
```

或通过 overload / builtin 声明避免解析到 glibc host 函数。

### 9.4 NaN 行为确认

需要用 CUDA 12.8 对以下输入单独确认：

```text
0x7FF8000000000000
```

确认 CUDA 的 `__signbitl` 返回值：

- 如果 CUDA 返回 `1`，MUSA 的 `__signbitl` 兼容实现需要特殊处理 NaN，与 `__signbitd` 不能简单共用。
- 如果 CUDA 返回 `0`，则当前 CTS golden 数据有问题，需要重新生成或修正。

## 10. 当前结论

`compatible_signbitl` 失败不是单纯一个 `quadmath.h` 缺失问题。

分层结论：

1. 最早失败点：`init_data.h` include `<quadmath.h>`，而 `mcc` 默认 include path 找不到 GCC 私有头。
2. 公共头问题：非 fp128 case 被 `ENABLE_FLOAT128` 和 `quadmath` 路径污染，导致需要额外 include/link。
3. 兼容层问题：MUSA clang device wrapper 缺少 `__signbitl` device 入口。
4. 语义问题：临时映射到 `__signbitd` 后，只剩 positive quiet NaN 与 golden 不一致，需要 CUDA 12.8 复核。

因此完整修复应同时处理 CTS common header 的 quadmath 依赖隔离、mcc host include/link 能力，以及 MUSA CUDA 兼容层的 `__signbitl` device wrapper/语义对齐。
