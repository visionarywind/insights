# musa_cts: grid_constant 失败全链路组件拆解与问题归属

## 1. 整体执行链路

执行命令：

```bash
cd /home/shanfeng/workspace/musa_cts/pytest
TEST_TYPE=dailyM3d pytest . -v -k \
"test_musa_mtcc_cuda_builtin_address_space_conversion[__cvta_grid_constant_to_generic]"
```

整体链路：

```text
pytest
  -> test_musa_mtcc.py
    -> 读取 musa_mtcc_test_cfg.csv
      -> 根据 TEST_TYPE=dailyM3d 选择 case
        -> 找到 __cvta_grid_constant_to_generic.cu
          -> 拼 mcc 编译命令
            -> mcc 编译 host/device 代码
              -> include musa_runtime.h
                -> include MUSA Runtime intrinsic headers
                -> include clang MUSA runtime wrapper
              -> 生成 device code
              -> device backend lowering
              -> host link
          -> 生成可执行文件
          -> 运行可执行文件
          -> gtest 判断结果
```

这次原始失败发生在：

```text
mcc 编译阶段
```

没有进入：

```text
二进制运行
kernel launch
MUSA Runtime 执行
Driver
M3D
GPU 执行
```

## 2. pytest 层做了什么

相关文件：

```text
/home/shanfeng/workspace/musa_cts/pytest/test_mtcc/test_musa_mtcc.py
```

测试函数逻辑：

```python
def test_musa_mtcc_cuda_builtin_address_space_conversion(test_case):
    result, run_log_path, compile_log_path, src_path = run_case(...)
    assert result, "run fail!"
```

pytest 本身不判断 `__cvta_grid_constant_to_generic` 的语义，只负责：

```text
编译 case
运行 case
收集 compile log / run log
最终 assert result
```

所以 pytest 报：

```text
AssertionError: run fail!
```

只是外层结果，不是根因。

## 3. CTS 配置层做了什么

配置文件：

```text
/home/shanfeng/workspace/musa_cts/pytest/test_mtcc/config/musa_mtcc_test_cfg.csv
```

相关配置：

```text
__cvta_grid_constant_to_generic,...,weekly-release-dailyM3d-SW-70176
```

因为包含 `dailyM3d`，所以这个 case 被选中。

结论：

```text
CTS 配置认为该 case 应属于 dailyM3d。
```

但 CTS 配置本身不实现 builtin，它只是把 case 放进测试集合。

如果当前 MUSA/MTCC 版本不要求支持 SW-70176，那么 CTS 配置存在“准入过早”的问题；但它不是底层技术根因。

## 4. 测试源码层做了什么

失败源码：

```text
/home/shanfeng/workspace/musa_cts/mtcc/cuda_builtin/address_space_conversion/__cvta_grid_constant_to_generic.cu
```

关键代码：

```cpp
__global__ void gpu_kernel(
    size_t *det,
    size_t *src,
    const __grid_constant__ GridConstData grid_const_data)
{
    printf("isGridConstant(&grid_const_data): %d\n",
           __isGridConstant(&grid_const_data));

    void *generic_ptr1 =
        __cvta_grid_constant_to_generic((size_t)&grid_const_data.val1);

    void *generic_ptr2 =
        __cvta_grid_constant_to_generic((size_t)&grid_const_data.val2);

    src[i] = (size_t)&grid_const_data.val2 - (size_t)&grid_const_data.val1;
    det[i] = (size_t)generic_ptr2 - (size_t)generic_ptr1;
}
```

该 case 测试 CUDA 12.8 兼容能力：

```text
__grid_constant__ kernel 参数
  -> __isGridConstant 判断是否是 grid_constant 地址空间
  -> __cvta_grid_constant_to_generic 做 grid_constant -> generic 地址转换
  -> 比较转换前后的成员地址偏移
```

该 case 依赖：

```cpp
__grid_constant__
__isGridConstant
__cvta_grid_constant_to_generic
```

## 5. mcc 编译层发生了什么

pytest 拼出的实际编译命令类似：

```bash
mcc __cvta_grid_constant_to_generic.cu \
  -I/home/shanfeng/workspace/musa_cts/googletest/googletest/include/ \
  -L/home/shanfeng/workspace/musa_cts/build/gtest/lib/ \
  -lgtest \
  -o __cvta_grid_constant_to_generic \
  -mtgpu -O2 -lmusart \
  --cuda-gpu-arch=mp_31
```

编译日志：

```text
error: use of undeclared identifier '__isGridConstant'
error: use of undeclared identifier '__cvta_grid_constant_to_generic'
```

这说明：

```text
mcc 前端在编译源码时，找不到这些函数/内建函数的声明。
```

这不是链接错误，也不是 backend lowering 错误。

原始失败点是：

```text
C/C++ 前端语义阶段：identifier 未声明
```

## 6. MUSA Runtime header 层的问题

当前 `/usr/local/musa-5.1.0` 里有 `__grid_constant__`：

```cpp
#define __grid_constant__ __location__(grid_constant)
```

这说明：

```text
grid_constant 这个参数修饰符语法是存在的。
```

但 intrinsic header 只提供了：

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

没有：

```cpp
__isGridConstant
__cvta_generic_to_grid_constant
__cvta_grid_constant_to_generic
```

直接结论：

```text
MUSA Runtime CUDA compatibility headers 没有暴露 grid_constant address-space builtin。
```

这是当前 case 编译失败的直接组件问题。

## 7. clang MUSA runtime wrapper 层的问题

相关文件：

```text
/usr/local/musa-5.1.0/lib/clang/20/include/__clang_musa_runtime_wrapper.h
```

现有 wrapper：

```cpp
__isGlobal    -> __musa_isspacep_global
__isShared    -> __musa_isspacep_shared
__isConstant  -> __musa_isspacep_const
__isLocal     -> __musa_isspacep_local
```

缺少：

```cpp
__isGridConstant -> __musa_isspacep_param
```

或者等价实现。

结论：

```text
clang wrapper 层也没有接上 grid_constant/param address-space predicate。
```

正式兼容时该组件也必须补。

## 8. musa C++ address_space 层的问题

相关文件：

```text
/usr/local/musa/include/musa/__memory/address_space.h
```

该文件已经有：

```cpp
address_space::grid_constant
```

但实现仍是：

```cpp
case address_space::grid_constant:
  // FIXME: cuda compatible
  return false;
```

这说明：

```text
MUSA headers 已经知道 CUDA 12.8 有 grid_constant address space，
但当前 predicate/cvta 功能层没有实现完成。
```

当前状态是：

```text
枚举/语法层知道 grid_constant，
但功能层没有实现。
```

## 9. backend 层是否是原始失败点

对原始 `__cvta_grid_constant_to_generic` case 来说，还没有走到 backend。

原因：

```text
原始错误是 undeclared identifier，前端解析源码时就失败。
```

因此不能把原始失败直接归因于 backend。

但是，如果只补声明，不补 backend，后续可能继续暴露 backend 问题。

方案 A overlay 能通过，是因为它使用 header-only identity fallback：

```cpp
__isGridConstant(ptr) -> ptr != nullptr
__cvta_generic_to_grid_constant(ptr) -> reinterpret_cast<size_t>(ptr)
__cvta_grid_constant_to_generic(rawbits) -> reinterpret_cast<void *>(rawbits)
```

该 fallback 没有引入新的 backend intrinsic，因此绕过了后端 lowering 问题。

这证明：

```text
当前两个 grid_constant CTS case 的最小阻塞点是 header/builtin 缺失。
```

但这不代表完整 CUDA 语义已经实现。

完整实现仍然需要 backend 支持：

```text
grid_constant / param address-space predicate
generic <-> param address conversion
__grid_constant__ 参数 metadata 保留
```

## 10. 运行期/M3D/Driver 是否参与失败

没有证据表明是这些组件的问题。

因为原始失败是：

```text
mcc 编译失败
二进制没有生成
```

所以没有进入：

```text
musart
MUSA Driver
M3D
HAL
GPU kernel execution
```

因此这次问题不应归因于：

```text
MUSA Runtime 执行期
Driver
M3D
GPU
```

## 11. 方案 A 验证说明了什么

使用 overlay header 注入：

```bash
COMPILE_EXTRA_ARGS='-include /tmp/musa_grid_constant_compat_overlay.h -L/usr/local/musa/lib'
```

两个 case 通过：

```text
__cvta_grid_constant_to_generic PASSED
__cvta_generic_to_grid_constant PASSED
```

这说明：

```text
当前失败的最小阻塞点确实是 header/builtin 缺失。
```

如果是 M3D、driver 或 kernel 执行问题，overlay 不会让它通过。

验证结果反向证明：

```text
原始目标 case 的主要问题在 MTCC/MUSA CUDA compatibility builtin 暴露层。
```

## 12. 整组验证中的另一个问题

整组 address-space conversion 结果：

```text
8 passed, 2 failed
```

失败项：

```text
__cvta_generic_to_local
__cvta_local_to_generic
```

错误：

```text
fatal error: error in backend: Cannot select: intrinsic %llvm.musa.isspacep.local
```

该错误发生在 backend instruction selection 阶段，说明：

```text
local address-space predicate 对应的 LLVM intrinsic 没有后端 selection 支持。
```

这属于：

```text
MTCC backend lowering / instruction selection 问题
```

它和 grid_constant case 的：

```text
header/builtin 未声明问题
```

不是同一个根因。

## 13. 组件责任拆分

### 13.1 pytest / CTS 框架

作用：

```text
选择 case
拼 mcc 命令
收集 compile/run log
assert 测试结果
```

问题归属：

```text
不是底层技术根因。
```

但它把未完全支持的 case 纳入了 `dailyM3d`。如果当前版本不要求支持 SW-70176，则 CTS 配置有“准入过早”的问题。

### 13.2 CTS 测试源码

作用：

```text
验证 CUDA 12.8 grid_constant address-space conversion。
```

问题归属：

```text
测试源码本身合理，前提是产品宣称支持这些 CUDA builtin。
```

如果当前 MUSA 版本没有支持，那测试目标超前，不是源码逻辑错误。

### 13.3 MUSA Runtime headers

作用：

```text
暴露 CUDA-compatible builtin 声明和 inline wrapper。
```

问题归属：

```text
缺少 __isGridConstant
缺少 __cvta_generic_to_grid_constant
缺少 __cvta_grid_constant_to_generic
```

这是原始失败的直接问题组件。

### 13.4 clang MUSA runtime wrapper

作用：

```text
把 CUDA-style builtin 映射到 MUSA/LLVM intrinsic。
```

问题归属：

```text
没有 grid_constant / param predicate wrapper。
```

这是正式兼容时必须补的组件。

### 13.5 MTCC frontend / Sema

作用：

```text
识别 __grid_constant__
编译 CUDA/MUSA 语法
```

当前状态：

```text
__grid_constant__ 语法已经存在。
```

所以 frontend attribute 不是这次的主要问题。

### 13.6 MTCC backend / libdevice

作用：

```text
实现 address-space predicate/cvta 的真实语义。
```

对本次 grid_constant 原始失败：

```text
还没走到这里。
```

对完整兼容：

```text
必须补。
```

对整组 local 失败：

```text
backend 已经明确有问题：Cannot select intrinsic %llvm.musa.isspacep.local。
```

### 13.7 MUSA Runtime / Driver / M3D

作用：

```text
运行 kernel
提交命令
执行 GPU 任务
```

问题归属：

```text
没有参与到原始失败路径。
```

所以不是它们的问题。

## 14. 最终判断：哪个组件出了问题

只针对最初失败 case：

```text
test_musa_mtcc_cuda_builtin_address_space_conversion[__cvta_grid_constant_to_generic]
```

最准确结论：

```text
问题组件：MUSA Runtime / MTCC CUDA compatibility builtin header 层。
```

具体问题：

```text
当前 /usr/local/musa-5.1.0 暴露了 __grid_constant__ 修饰符，
但没有暴露 __isGridConstant / __cvta_grid_constant_to_generic / __cvta_generic_to_grid_constant。
```

更完整地说：

```text
CUDA 12.8 grid_constant address-space conversion 兼容链路没有实现完整。
当前首先卡在 header 声明缺失；
正式支持还需要 clang wrapper 和 MTCC backend/libdevice 支持 param/grid_constant address-space predicate/cvta lowering。
```

不是：

```text
pytest 框架问题
MUSA Driver 问题
M3D 问题
GPU 执行问题
```

## 15. 一句话总结

这个失败是 CUDA 12.8 `grid_constant` address-space builtin 兼容能力缺失暴露出来的问题：CTS 把 `__cvta_grid_constant_to_generic` 纳入 dailyM3d，但当前 MUSA/MTCC 只实现了 `global/shared/constant/local` 的 address-space builtin，没有实现 `grid_constant/param` 的 predicate 和 cvta；原始失败点在 MUSA Runtime/MTCC CUDA compatibility header 层缺声明，完整修复还需要 clang wrapper + MTCC backend/libdevice lowering。
