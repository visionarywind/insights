# musa_cts: grid_constant 方案 A 验证记录

## 1. 验证目标

针对失败用例：

```text
test_musa_mtcc_cuda_builtin_address_space_conversion[__cvta_grid_constant_to_generic]
```

先尝试“方案 A”：使用最小 header-only fallback，为当前 MUSA/MTCC 缺失的 CUDA 12.8 grid_constant address-space builtin 提供临时兼容定义，验证是否能 unblock CTS。

目标 builtin：

```cpp
__isGridConstant(const void *)
__cvta_generic_to_grid_constant(const void *)
__cvta_grid_constant_to_generic(size_t)
```

## 2. 为什么没有直接修改 `/usr/local/musa`

远程安装包头文件是 root-owned：

```text
-rw-r--r-- 1 root root /usr/local/musa/include/mp_ext_20_intrinsics.h
-rw-r--r-- 1 root root /usr/local/musa/include/mp_ext_20_intrinsics.hpp
```

当前用户没有免密 sudo：

```text
sudo: a password is required
```

因此本次验证没有直接修改 `/usr/local/musa`，而是使用 pytest 支持的 `COMPILE_EXTRA_ARGS` 注入 overlay header。

## 3. Overlay header

临时文件：

```text
/tmp/musa_grid_constant_compat_overlay.h
```

内容：

```cpp
#pragma once
#include <musa_runtime.h>

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

### 3.1 第一次尝试失败原因

第一次 overlay 只在 `__MUSA_ARCH__` 下定义：

```cpp
#if defined(__MUSA_ARCH__)
...
#endif
```

失败日志显示错误发生在 host 编译 pass：

```text
3 errors generated when compiling for host.
```

所以 host pass 也需要看到这些符号。最终改为：

```cpp
static __host__ __device__ __forceinline__ ...
```

## 4. pytest 注入方式

pytest 入口：

```text
/home/shanfeng/workspace/musa_cts/pytest/test_mtcc/test_musa_mtcc.py
```

该文件读取：

```python
g_compile_extra_args = os.getenv("COMPILE_EXTRA_ARGS", "")
```

并把它拼进 `mcc` 编译命令：

```python
dc_tmp["compile_args"] = "{} {}".format(eachline["compile_args"], g_compile_extra_args)
```

因此验证时使用：

```bash
COMPILE_EXTRA_ARGS='-include /tmp/musa_grid_constant_compat_overlay.h -L/usr/local/musa/lib'
```

其中：

```text
-include /tmp/musa_grid_constant_compat_overlay.h
  注入 grid_constant fallback builtin

-L/usr/local/musa/lib
  避免同组普通 address conversion case 触发 -lmusart 链接路径问题
```

## 5. 单点验证：__cvta_grid_constant_to_generic

命令：

```bash
cd /home/shanfeng/workspace/musa_cts/pytest
COMPILE_EXTRA_ARGS='-include /tmp/musa_grid_constant_compat_overlay.h -L/usr/local/musa/lib' \
TEST_TYPE=dailyM3d pytest . -v -k \
"test_musa_mtcc_cuda_builtin_address_space_conversion[__cvta_grid_constant_to_generic]"
```

结果：

```text
test_mtcc/test_musa_mtcc.py::test_musa_mtcc_cuda_builtin_address_space_conversion[__cvta_grid_constant_to_generic] PASSED [100%]

1 passed, 7162 deselected in 10.51s
```

结论：

```text
方案 A 可以让原始失败 case 从编译失败变为通过。
```

## 6. 单点验证：__cvta_generic_to_grid_constant

命令：

```bash
cd /home/shanfeng/workspace/musa_cts/pytest
COMPILE_EXTRA_ARGS='-include /tmp/musa_grid_constant_compat_overlay.h -L/usr/local/musa/lib' \
TEST_TYPE=dailyM3d pytest . -v -k \
"test_musa_mtcc_cuda_builtin_address_space_conversion[__cvta_generic_to_grid_constant]"
```

结果：

```text
test_mtcc/test_musa_mtcc.py::test_musa_mtcc_cuda_builtin_address_space_conversion[__cvta_generic_to_grid_constant] PASSED [100%]

1 passed, 7162 deselected in 10.45s
```

结论：

```text
同组反向 grid_constant conversion case 也可以通过。
```

## 7. 整组 address-space conversion 验证

命令：

```bash
cd /home/shanfeng/workspace/musa_cts/pytest
COMPILE_EXTRA_ARGS='-include /tmp/musa_grid_constant_compat_overlay.h -L/usr/local/musa/lib' \
TEST_TYPE=dailyM3d pytest . -v -k \
"test_musa_mtcc_cuda_builtin_address_space_conversion"
```

结果摘要：

```text
10 selected
8 passed, 2 failed, 7153 deselected
```

通过项：

```text
__cvta_generic_to_shared PASSED
__cvta_generic_to_constant PASSED
__cvta_generic_to_global PASSED
__cvta_shared_to_generic PASSED
__cvta_constant_to_generic PASSED
__cvta_global_to_generic PASSED
__cvta_generic_to_grid_constant PASSED
__cvta_grid_constant_to_generic PASSED
```

失败项：

```text
__cvta_generic_to_local FAILED
__cvta_local_to_generic FAILED
```

失败日志：

```text
fatal error: error in backend: Cannot select: intrinsic %llvm.musa.isspacep.local
mcc: error: clang frontend command failed with exit code 70
```

结论：

```text
剩余失败是 local address-space backend selection 问题，和本次 grid_constant fallback 不是同一个根因。
```

## 8. 验证结论

方案 A 验证通过：

```text
对当前两个 grid_constant CTS case，最小 header-only identity fallback 可行。
```

已经通过的目标 case：

```text
__cvta_grid_constant_to_generic
__cvta_generic_to_grid_constant
```

需要注意：当前 overlay 是临时验证方式，没有写入正式安装包。

## 9. 后续落地建议

如果目标是让 dailyM3d 先恢复：

```text
1. 将 overlay 中的三个 builtin 正式加入 MUSA Runtime headers。
2. 确保 host/device 编译 pass 都能看到声明。
3. 确保 mcc 编译命令能找到 libmusart，必要时补 -L/usr/local/musa/lib 或修 mcc driver 默认 lib path。
```

如果目标是完整 CUDA 12.8 兼容：

```text
1. 不应长期依赖 identity fallback。
2. 需要在 MTCC backend 中补 grid_constant/param address-space predicate 和 cvta lowering。
3. 需要让 musa/__memory/address_space.h 的 grid_constant 分支从 return false 改为调用 __isGridConstant。
```

## 10. 一句话总结

本次验证证明：`__isGridConstant`、`__cvta_generic_to_grid_constant`、`__cvta_grid_constant_to_generic` 的 header-only identity fallback 能让两个 grid_constant CTS case 在 dailyM3d 下通过；整组中剩余 local conversion 失败是独立的 `%llvm.musa.isspacep.local` 后端 selection 问题。
