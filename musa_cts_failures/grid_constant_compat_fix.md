# musa_cts: grid_constant address-space conversion 兼容修复建议

## 1. 修复目标

当前失败 case：

```text
test_musa_mtcc_cuda_builtin_address_space_conversion[__cvta_grid_constant_to_generic]
```

直接错误是 `mcc` 编译期找不到以下 CUDA 12.8 builtin：

```cpp
__isGridConstant(const void *)
__cvta_generic_to_grid_constant(const void *)
__cvta_grid_constant_to_generic(size_t)
```

要实现兼容，不能只修改 `musa_cts`，而应补齐 MUSA/MTCC 对 CUDA 12.8 `grid_constant` address-space predicate/conversion 的支持链路：

```text
MUSA Runtime headers
  -> clang MUSA runtime wrapper
  -> device builtin / libdevice / compiler lowering
  -> musa/__memory/address_space.h
  -> CTS 验证
```

## 2. Header 层最小修复

### 2.1 `mp_ext_20_intrinsics.h`

当前 `/usr/local/musa/include/mp_ext_20_intrinsics.h` 只声明了普通 address-space intrinsic：

```cpp
__MP_EXT_20_INTRINSICS_DECL__ unsigned int __isGlobal(const void *ptr) __DEF_IF_HOST
__MP_EXT_20_INTRINSICS_DECL__ unsigned int __isShared(const void *ptr) __DEF_IF_HOST
__MP_EXT_20_INTRINSICS_DECL__ unsigned int __isConstant(const void *ptr) __DEF_IF_HOST
__MP_EXT_20_INTRINSICS_DECL__ unsigned int __isLocal(const void *ptr) __DEF_IF_HOST

__MP_EXT_20_INTRINSICS_DECL__ size_t __cvta_generic_to_global(const void *ptr) __DEF_IF_HOST
__MP_EXT_20_INTRINSICS_DECL__ size_t __cvta_generic_to_shared(const void *ptr) __DEF_IF_HOST
__MP_EXT_20_INTRINSICS_DECL__ size_t __cvta_generic_to_constant(const void *ptr) __DEF_IF_HOST
__MP_EXT_20_INTRINSICS_DECL__ size_t __cvta_generic_to_local(const void *ptr) __DEF_IF_HOST

__MP_EXT_20_INTRINSICS_DECL__ void * __cvta_global_to_generic(size_t rawbits) __DEF_IF_HOST
__MP_EXT_20_INTRINSICS_DECL__ void * __cvta_shared_to_generic(size_t rawbits) __DEF_IF_HOST
__MP_EXT_20_INTRINSICS_DECL__ void * __cvta_constant_to_generic(size_t rawbits) __DEF_IF_HOST
__MP_EXT_20_INTRINSICS_DECL__ void * __cvta_local_to_generic(size_t rawbits) __DEF_IF_HOST
```

需要在同一区域补充：

```cpp
__MP_EXT_20_INTRINSICS_DECL__ unsigned int __isGridConstant(const void *ptr) __DEF_IF_HOST

__MP_EXT_20_INTRINSICS_DECL__ size_t __cvta_generic_to_grid_constant(const void *ptr) __DEF_IF_HOST
__MP_EXT_20_INTRINSICS_DECL__ void * __cvta_grid_constant_to_generic(size_t rawbits) __DEF_IF_HOST
```

### 2.2 `mp_ext_20_intrinsics.hpp`

当前 `/usr/local/musa/include/mp_ext_20_intrinsics.hpp` 只定义了 `global/shared/constant/local` wrapper。

需要补充 device symbol 声明和 wrapper：

```cpp
extern "C" {
  __device__ unsigned __mt_isGridConstant_impl(const void *);
  __device__ size_t __mt_cvta_generic_to_grid_constant_impl(const void *);
  __device__ void * __mt_cvta_grid_constant_to_generic_impl(size_t);
}

__MP_EXT_20_INTRINSICS_DECL__ unsigned int __isGridConstant(const void *ptr)
{
  return __mt_isGridConstant_impl(ptr);
}

__MP_EXT_20_INTRINSICS_DECL__ size_t __cvta_generic_to_grid_constant(const void *p)
{
  return __mt_cvta_generic_to_grid_constant_impl(p);
}

__MP_EXT_20_INTRINSICS_DECL__ void * __cvta_grid_constant_to_generic(size_t rawbits)
{
  return __mt_cvta_grid_constant_to_generic_impl(rawbits);
}
```

注意：只补 `.h/.hpp` 不够。如果没有底层 device symbol，下一步会从“未声明”变成 device link 或 lowering 失败。

## 3. clang MUSA runtime wrapper 修复

当前 `/usr/local/musa-5.1.0/lib/clang/20/include/__clang_musa_runtime_wrapper.h` 有：

```cpp
__DEVICE__ unsigned int __isGlobal(const void *p) {
  return __musa_isspacep_global(p);
}
__DEVICE__ unsigned int __isShared(const void *p) {
  return __musa_isspacep_shared(p);
}
__DEVICE__ unsigned int __isConstant(const void *p) {
  return __musa_isspacep_const(p);
}
__DEVICE__ unsigned int __isLocal(const void *p) {
  return __musa_isspacep_local(p);
}
```

需要增加 grid_constant predicate wrapper，语义上应对应 kernel parameter / param address space：

```cpp
__DEVICE__ unsigned int __isGridConstant(const void *p) {
  return __musa_isspacep_param(p);
}
```

这里的关键前提是 MTCC 是否已有 `__musa_isspacep_param` 或等价 builtin。

- 如果已有：直接接入 wrapper。
- 如果没有：需要在 MTCC builtin 表、Sema/CodeGen/lowering 链路中新增。

## 4. device builtin / libdevice / backend 修复

需要实现以下 device 层入口：

```cpp
__mt_isGridConstant_impl
__mt_cvta_generic_to_grid_constant_impl
__mt_cvta_grid_constant_to_generic_impl
```

语义应对应 CUDA/PTX 中的 parameter state space：

```text
__isGridConstant(ptr)
  -> 判断 ptr 是否属于 kernel parameter / grid_constant address space

__cvta_generic_to_grid_constant(ptr)
  -> generic pointer 转 grid_constant/param raw address

__cvta_grid_constant_to_generic(rawbits)
  -> grid_constant/param raw address 转 generic pointer
```

如果 MUSA IR/ISA 已有 address-space 区分，正确做法是 lowering 到对应 address-space cast / cvta op。

如果当前后端还没有 param address-space cvta，可以考虑先做受控 fallback：

```cpp
__mt_isGridConstant_impl(ptr) -> return 0 或基于 metadata 判断
__mt_cvta_generic_to_grid_constant_impl(ptr) -> reinterpret_cast<size_t>(ptr)
__mt_cvta_grid_constant_to_generic_impl(rawbits) -> reinterpret_cast<void *>(rawbits)
```

这个 fallback 可能让现有 CTS 中的地址偏移类验证先通过，但不等价于完整 CUDA 语义，尤其 `__isGridConstant` 的准确性不完整。

## 5. `musa/__memory/address_space.h` 修复

当前 `/usr/local/musa/include/musa/__memory/address_space.h` 中 `grid_constant` 分支是未实现状态：

```cpp
case address_space::grid_constant:
  // FIXME: cuda compatible
  // MT_IF_ELSE_TARGET(MT_PROVIDES_SM_70, (return static_cast<bool>(::__isGridConstant(__ptr));), (return false;))
  return false;
```

当 `__isGridConstant` 可用后，应改为：

```cpp
case address_space::grid_constant:
  return static_cast<bool>(::__isGridConstant(__ptr));
```

否则 C++ 标准库侧接口仍会永远返回 false：

```cpp
musa::device::is_address_from(ptr, address_space::grid_constant)
```

## 6. 推荐分阶段方案

### 阶段 A：先 unblock dailyM3d

目标：让当前 CUDA 12.8 CTS grid_constant conversion case 先编译通过并进入运行阶段。

建议改动：

```text
MUSA-Runtime/include/mp_ext_20_intrinsics.h
MUSA-Runtime/include/mp_ext_20_intrinsics.hpp
clang MUSA runtime wrapper
libdevice/device builtin
```

可以先用 identity cvta fallback：

```text
generic -> grid_constant: reinterpret raw pointer
grid_constant -> generic: reinterpret raw address
```

优点：

```text
改动小，能快速确认 CTS 是否还存在后续 runtime/semantic 问题。
```

风险：

```text
__isGridConstant 可能不完整，不能代表完整 CUDA 12.8 语义。
```

### 阶段 B：补完整 CUDA 12.8 兼容语义

目标：完整支持 grid_constant / param address-space predicate 和 conversion。

需要 MTCC/backend 支持：

```text
isspacep.param 或等价 predicate
cvta.param 或 generic <-> param conversion
__grid_constant__ kernel 参数地址空间 metadata 保留
optimizer 不错误折叠 grid_constant pointer
```

这一步才是最终兼容方案。

## 7. 验证命令

修复后先单点验证：

```bash
cd /home/shanfeng/workspace/musa_cts/pytest

TEST_TYPE=dailyM3d pytest . -v -k \
"test_musa_mtcc_cuda_builtin_address_space_conversion[__cvta_grid_constant_to_generic]"

TEST_TYPE=dailyM3d pytest . -v -k \
"test_musa_mtcc_cuda_builtin_address_space_conversion[__cvta_generic_to_grid_constant]"
```

再验证同组 address conversion：

```bash
TEST_TYPE=dailyM3d pytest . -v -k \
"test_musa_mtcc_cuda_builtin_address_space_conversion"
```

另外需要注意：之前检查相邻普通 address conversion case 时看到过独立链接问题：

```text
/usr/bin/ld: cannot find -lmusart: No such file or directory
```

这个是链接路径问题，和 grid_constant builtin 缺失不是同一个根因。实现兼容后如果继续卡在 `-lmusart`，还需要确保编译命令带上 MUSA runtime library path：

```bash
-L/usr/local/musa/lib
```

或修复 `mcc` driver 自动注入 runtime lib path 的逻辑。

## 8. 最推荐判断

如果目标是“让当前 dailyM3d 先绿”：

```text
先补 header + wrapper + libdevice fallback，同时修 -lmusart 链接路径。
```

如果目标是“真正 CUDA 12.8 兼容”：

```text
必须在 MTCC backend 增加 grid_constant/param address-space predicate 和 cvta lowering，不能只靠 header alias。
```
