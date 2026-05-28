# musa_cts: grid_constant address-space conversion 用例失败分析

## 1. 问题背景

远程服务器路径：

```text
/home/shanfeng/workspace/musa_cts/pytest
```

复现命令：

```bash
cd /home/shanfeng/workspace/musa_cts/pytest
TEST_TYPE=dailyM3d pytest . -v -k "test_musa_mtcc_cuda_builtin_address_space_conversion[__cvta_grid_constant_to_generic]"
```

失败用例：

```text
test_mtcc/test_musa_mtcc.py::test_musa_mtcc_cuda_builtin_address_space_conversion[__cvta_grid_constant_to_generic]
```

测试环境关键信息：

```text
test device type: s5000
MUSA_INSTALL_PATH: /usr/local/musa
/usr/local/musa -> musa-5.1.0
mcc version 3.1.0
clang version 20.1.8
--cuda-gpu-arch=mp_31
TEST_TYPE=dailyM3d
```

## 2. 错误是如何发现的

pytest 最外层失败只显示 `assert result` 失败：

```text
AssertionError: run fail!
assert False
```

这只能说明 `run_case(...)` 返回失败，不能直接说明根因。

真正的错误入口来自 pytest 的 `Captured stdout call`：

```text
mcc __cvta_grid_constant_to_generic.cu ... 1> log/musa_mtcc_test_compile___cvta_grid_constant_to_generic.txt 2>&1
process run error, ret code: 1

chmod u+x __cvta_grid_constant_to_generic
process run error, ret code: 127
```

同时 stderr 里有：

```text
chmod: cannot access '__cvta_grid_constant_to_generic': No such file or directory
```

因此第一层判断是：

```text
mcc 编译失败，二进制没有生成；后续运行失败 ret code 127 是连带错误。
```

随后查看 pytest 生成的编译日志：

```text
/home/shanfeng/workspace/musa_cts/pytest/log/musa_mtcc_test_compile___cvta_grid_constant_to_generic.txt
```

日志给出直接编译错误：

```text
__cvta_grid_constant_to_generic.cu:24:58: error: use of undeclared identifier '__isGridConstant'; did you mean '__isConstant'?
__cvta_grid_constant_to_generic.cu:28:26: error: use of undeclared identifier '__cvta_grid_constant_to_generic'; did you mean '__cvta_constant_to_generic'?
__cvta_grid_constant_to_generic.cu:29:26: error: use of undeclared identifier '__cvta_grid_constant_to_generic'; did you mean '__cvta_constant_to_generic'?
3 errors generated when compiling for mp_31.
```

错误发现链路：

```text
pytest assert result failed
  -> Captured stdout 显示 mcc ret code 1
  -> 二进制不存在，运行 ret code 127 是连带错误
  -> 打开 compile log
  -> 看到 __isGridConstant / __cvta_grid_constant_to_generic 未声明
```

## 3. 失败源码

源码位置：

```text
/home/shanfeng/workspace/musa_cts/mtcc/cuda_builtin/address_space_conversion/__cvta_grid_constant_to_generic.cu
```

关键代码：

```cpp
__global__ void gpu_kernel(size_t *det, size_t *src, const __grid_constant__ GridConstData grid_const_data)
{
    if (i < 5)
    {
        printf("isGridConstant(&grid_const_data): %d\n", __isGridConstant(&grid_const_data));
    }

    void *generic_ptr1 = __cvta_grid_constant_to_generic((size_t)&grid_const_data.val1);
    void *generic_ptr2 = __cvta_grid_constant_to_generic((size_t)&grid_const_data.val2);

    src[i] = (size_t)&grid_const_data.val2 - (size_t)&grid_const_data.val1;
    det[i] = (size_t)generic_ptr2 - (size_t)generic_ptr1;
}
```

该 case 验证的是 CUDA 12.8 兼容方向的 `grid_constant` address-space conversion：

```text
__grid_constant__ kernel 参数
  -> __isGridConstant predicate
  -> __cvta_grid_constant_to_generic conversion
  -> 比较转换前后结构体成员地址偏移是否一致
```

## 4. CTS 配置为什么会跑到这个 case

配置位置：

```text
/home/shanfeng/workspace/musa_cts/pytest/test_mtcc/config/musa_mtcc_test_cfg.csv
```

相关行：

```text
2308,__cvta_generic_to_grid_constant,mtcc_cuda_builtin_address_space_conversion,cuda_builtin/address_space_conversion,__cvta_generic_to_grid_constant,,-mtgpu -O2 -lmusart,,,m-11000,p-1111,30,weekly-release-dailyM3d-SW-70176
2309,__cvta_grid_constant_to_generic,mtcc_cuda_builtin_address_space_conversion,cuda_builtin/address_space_conversion,__cvta_grid_constant_to_generic,,-mtgpu -O2 -lmusart,,,m-11000,p-1111,30,weekly-release-dailyM3d-SW-70176
```

pytest 装载逻辑在：

```text
/home/shanfeng/workspace/musa_cts/pytest/test_mtcc/test_musa_mtcc.py
```

核心筛选逻辑：

```python
test_type = os.getenv("TEST_TYPE")
...
if test_type is None or test_type in eachline["type"]:
    ...
    if common_fun.is_supported_arch(dc_tmp["test_target"], SysInfo.mtgpu_arch):
        g_test_suites[dc_tmp["suite"]][dc_tmp["case"]] = dc_tmp
```

因为该 case 的 `type` 包含 `dailyM3d`，所以执行 `TEST_TYPE=dailyM3d pytest ...` 会把该 case 收进测试集合。

## 5. 根因分析

当前 `/usr/local/musa-5.1.0` 里可以看到 `__grid_constant__` 修饰符本身存在：

```text
/usr/local/musa/include/crt/host_defines.h
```

```cpp
#define __grid_constant__ \
        __location__(grid_constant)
```

但是 address-space predicate/conversion builtin 没有暴露。

`/usr/local/musa/include/mp_ext_20_intrinsics.h` 只声明了普通 address space：

```cpp
__isGlobal
__isShared
__isConstant
__isLocal

__cvta_generic_to_global
__cvta_generic_to_shared
__cvta_generic_to_constant
__cvta_generic_to_local

__cvta_global_to_generic
__cvta_shared_to_generic
__cvta_constant_to_generic
__cvta_local_to_generic
```

没有声明：

```cpp
__isGridConstant
__cvta_generic_to_grid_constant
__cvta_grid_constant_to_generic
```

`/usr/local/musa/include/mp_ext_20_intrinsics.hpp` 也只定义了普通 address-space wrapper，没有 grid_constant wrapper。

同时，`/usr/local/musa/include/musa/__memory/address_space.h` 中 `grid_constant` 分支仍是未实现状态：

```cpp
case address_space::grid_constant:
  // FIXME: cuda compatible
  // MT_IF_ELSE_TARGET(MT_PROVIDES_SM_70, (return static_cast<bool>(::__isGridConstant(__ptr));), (return false;))
  return false;
```

因此当前状态是：

```text
__grid_constant__ 语法属性存在，
但 grid_constant address-space predicate/cvta builtin 没有在 MUSA Runtime headers / clang wrapper / device builtin 链路中实现。
```

这会导致用例在编译期直接失败。

## 6. 与运行期/M3D 的关系

该问题不是 kernel 执行失败，也不是 M3D runtime 行为错误。

原因：

```text
mcc 编译阶段已经报 undeclared identifier；
二进制没有生成；
用例没有进入 device 执行阶段。
```

因此当前失败点在：

```text
CTS 配置 / MTCC CUDA compatibility header builtin coverage
```

而不是：

```text
MUSA Driver
MUSA Runtime kernel launch
M3D command submission
GPU 执行结果
```

## 7. 关联现象

同组 `__cvta_generic_to_grid_constant` 也会因为相同原因失败：

```text
error: use of undeclared identifier '__isGridConstant'
error: use of undeclared identifier '__cvta_grid_constant_to_generic'
error: use of undeclared identifier '__cvta_generic_to_grid_constant'
```

另外检查相邻普通 address conversion case 时，看到 `-lmusart` 链接路径问题：

```text
/usr/bin/ld: cannot find -lmusart: No such file or directory
```

这是另一个环境/链接路径问题；但它不是本次 `__cvta_grid_constant_to_generic` 的首要根因，因为目标 case 在进入链接前已经因为 builtin 未声明失败。

## 8. 建议处理

### 8.1 如果当前 dailyM3d 不要求 SW-70176 完成

临时处理建议：从 `dailyM3d` 标签中移除以下两个 case，或在 pytest 中做 xfail/skip：

```text
__cvta_generic_to_grid_constant
__cvta_grid_constant_to_generic
```

原因：当前安装包不具备对应 builtin，纳入 dailyM3d 后必然编译失败。

### 8.2 如果目标是补齐 CUDA 12.8 兼容

需要在 MTCC/MUSA Runtime 侧补齐完整链路：

```text
mp_ext_20_intrinsics.h/.hpp:
  声明并定义 __isGridConstant
  声明并定义 __cvta_generic_to_grid_constant
  声明并定义 __cvta_grid_constant_to_generic

__clang_musa_runtime_wrapper.h:
  像 __isGlobal/__isConstant 一样暴露 grid_constant wrapper

compiler/libdevice/backend:
  支持 grid_constant/param address-space predicate 和 cvta lowering

musa/__memory/address_space.h:
  grid_constant 分支不能继续 return false
```

实现后应至少验证：

```bash
cd /home/shanfeng/workspace/musa_cts/pytest
TEST_TYPE=dailyM3d pytest . -v -k "test_musa_mtcc_cuda_builtin_address_space_conversion[__cvta_generic_to_grid_constant]"
TEST_TYPE=dailyM3d pytest . -v -k "test_musa_mtcc_cuda_builtin_address_space_conversion[__cvta_grid_constant_to_generic]"
```

## 9. 一句话结论

`__cvta_grid_constant_to_generic` case 失败的直接原因是 `mcc` 编译期找不到 `__isGridConstant` 和 `__cvta_grid_constant_to_generic`；根因是当前 `/usr/local/musa-5.1.0` 只支持 `global/shared/constant/local` address-space builtin，没有实现 CUDA 12.8 `grid_constant` address-space predicate/conversion builtin，但 CTS 已把该 case 放进 `dailyM3d`。
