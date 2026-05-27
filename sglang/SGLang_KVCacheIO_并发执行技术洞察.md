# SGLang KVCacheIO 并发执行技术洞察

分析对象：远程仓库 `/home/shanfeng/workspace/sglang`。

关注问题：`sgl_kernel.kvcacheio` 是否可以并发执行，以及在什么条件下可以与其他 KV cache 搬运或计算重叠。

## 结论

`kvcacheio` 可以并发执行，但不会自动并发。

默认情况下，`kvcacheio` kernel 会提交到当前 PyTorch stream。同一个 stream 内的多个 `transfer_kv_*` 调用按提交顺序执行，不会并发。只有上层显式使用不同 stream，并且不同任务之间不存在数据冲突时，多个 `kvcacheio` kernel 才有机会并发执行，或者与 attention、MLP 等计算 kernel 重叠。

需要区分两类并发：

| 类型 | 是否支持 | 说明 |
|---|---|---|
| 单个 `kvcacheio` kernel 内部并行 | 支持 | kernel 内部按 warp 并行搬运 KV item |
| 多个 `kvcacheio` op 之间并发 | 条件支持 | 需要不同 stream，且 src/dst 区域无冲突 |

`kernel` 路径适合做并发和 overlap。`direct` 路径包含 CPU 索引处理和 PyTorch `copy_` 循环，不适合作为高并发主路径。

## 源码入口

Python wrapper：

```text
sgl-kernel/python/sgl_kernel/kvcacheio.py
```

CUDA/MUSA kernel 实现：

```text
sgl-kernel/csrc/kvcacheio/transfer.cu
```

CUDA 注册入口：

```text
sgl-kernel/csrc/common_extension.cc
```

MUSA 注册入口：

```text
sgl-kernel/csrc/common_extension_musa.cc
```

SGLang 主要调用位置：

```text
python/sglang/srt/mem_cache/memory_pool_host.py
```

## 单个 kernel 内部如何并行

`transfer.cu` 中的核心拷贝函数是：

```cpp
__device__ __forceinline__ void
transfer_item_warp(int32_t lane_id, const void* src_addr, void* dst_addr, int64_t item_size_bytes) {
  const uint64_t* __restrict__ src = static_cast<const uint64_t*>(src_addr);
  uint64_t* __restrict__ dst = static_cast<uint64_t*>(dst_addr);
  const int total_chunks = item_size_bytes / sizeof(uint64_t);

  for (int j = lane_id; j < total_chunks; j += WARP_SIZE) {
    ...
  }
}
```

含义：

```text
一个 KV item 被拆成多个 64-bit chunk。
一个 warp 负责搬运一个 item。
warp 内不同 lane 分摊 chunk。
```

例如 `WARP_SIZE = 32` 时：

```text
lane0  复制 chunk 0, 32, 64 ...
lane1  复制 chunk 1, 33, 65 ...
lane2  复制 chunk 2, 34, 66 ...
...
lane31 复制 chunk 31, 63, 95 ...
```

因此，单次 `transfer_kv_per_layer` 或 `transfer_kv_all_layer` 本身已经是 GPU 并行搬运。

CUDA 路径使用：

```cpp
ld.global.nc.b64
st.global.cg.b64
```

ROCm/MUSA 路径使用：

```cpp
__builtin_nontemporal_load
__builtin_nontemporal_store
```

这说明 `kvcacheio` 的主要瓶颈通常是内存带宽和访问模式，而不是计算吞吐。

## 多个 kvcacheio 调用的 stream 语义

`transfer_kv_launcher` 中的 kernel 发射逻辑如下：

```cpp
cudaStream_t torch_current_stream = at::cuda::getCurrentCUDAStream();

transfer_kernel_impl<...><<<grid_dim, threads_per_block, 0, torch_current_stream>>>(
    ...
);
```

关键点：

```text
kvcacheio 不自己创建 stream。
kvcacheio 使用调用时的当前 PyTorch stream。
是否并发由上层 stream 调度决定。
```

因此：

```text
同一个 stream：
  transfer_kv_1
  transfer_kv_2
  transfer_kv_3

执行顺序：
  transfer_kv_1 -> transfer_kv_2 -> transfer_kv_3
```

不同 stream：

```text
stream0: transfer_kv_1
stream1: transfer_kv_2
```

如果硬件资源允许，并且两次 transfer 的读写区域没有冲突，就可以并发执行或部分重叠。

## Python 侧并发提交示例

示例代码：

```python
import torch
from sgl_kernel.kvcacheio import transfer_kv_per_layer

s0 = torch.cuda.Stream()
s1 = torch.cuda.Stream()

with torch.cuda.stream(s0):
    transfer_kv_per_layer(
        src_k_0,
        dst_k_0,
        src_v_0,
        dst_v_0,
        src_indices_0,
        dst_indices_0,
        item_size_0,
    )

with torch.cuda.stream(s1):
    transfer_kv_per_layer(
        src_k_1,
        dst_k_1,
        src_v_1,
        dst_v_1,
        src_indices_1,
        dst_indices_1,
        item_size_1,
    )

# 后续如果要在当前 stream 读取 dst_k_0/dst_v_0/dst_k_1/dst_v_1，
# 必须等待 transfer 所在 stream 完成。
torch.cuda.current_stream().wait_stream(s0)
torch.cuda.current_stream().wait_stream(s1)
```

这个示例只说明提交方式。实际能否重叠，还取决于硬件资源、访存带宽、copy 方向、cache 位置和依赖关系。

## 正确并发的必要条件

并发执行需要同时满足以下条件：

| 条件 | 说明 |
|---|---|
| 使用不同 stream | 同一个 stream 内按顺序执行 |
| 源数据稳定 | kernel 执行期间不能修改 `src_k/src_v/src_indices` |
| 目标区域无冲突 | 不同 transfer 不能写同一个 KV cache slot |
| 无读写冲突 | 一个 transfer 写入的区域不能被另一个 kernel 同时读取，除非有明确同步 |
| 后续计算等待完成 | attention 使用 KV 前必须等待 transfer stream |
| 硬件支持 overlap | 资源不足或带宽饱和时，并发可能退化为排队或低收益重叠 |

最关键的是目标 slot 冲突。`dst_indices` 表示写入的 KV cache 位置。如果两个 transfer 同时写入同一个 `dst_indices`，结果不可靠。

错误示例：

```text
transfer A:
  dst_indices = [100, 101, 102]

transfer B:
  dst_indices = [102, 103, 104]
```

`102` 被两个 kernel 同时写入，属于写写冲突。

正确示例：

```text
transfer A:
  dst_indices = [100, 101, 102]

transfer B:
  dst_indices = [200, 201, 202]
```

两个 transfer 写入不同 KV cache slot，具备并发条件。

## 与 attention 计算重叠的条件

`kvcacheio` 的常见使用场景是 host KV cache 加载到 device KV cache，然后 attention 读取这些 KV。

不能直接这样做：

```text
stream_copy: transfer_kv 写入 dst cache
stream_compute: attention 立即读取 dst cache
```

如果没有同步，attention 可能读到尚未写完的 KV。

正确方式是让 compute stream 等待 copy stream：

```python
with torch.cuda.stream(copy_stream):
    transfer_kv_per_layer(...)

compute_stream.wait_stream(copy_stream)

with torch.cuda.stream(compute_stream):
    run_attention(...)
```

可重叠的场景通常是：

```text
copy stream 搬运 request B 的 KV
compute stream 计算 request A 的 attention/MLP
```

这两个任务操作不同 request 的 KV cache 区域，且没有直接依赖。

不可重叠的场景是：

```text
copy stream 搬运 request A 当前 step 需要的 KV
compute stream 立即计算 request A 当前 step attention
```

这种情况下 compute 必须等待 copy 完成。

## kernel 路径与 direct 路径差异

`kvcacheio.py` 暴露两类接口。

kernel 路径：

```text
transfer_kv_per_layer
transfer_kv_all_layer
transfer_kv_per_layer_pf_lf
transfer_kv_all_layer_lf_pf
transfer_kv_per_layer_mla
transfer_kv_all_layer_mla
```

特点：

```text
直接发射 CUDA/MUSA kernel。
使用当前 PyTorch stream。
更适合并发调度。
适合大量离散 token/page 搬运。
```

direct 路径：

```text
transfer_kv_direct
transfer_kv_per_layer_direct_pf_lf
transfer_kv_all_layer_direct_lf_pf
```

源码中 `transfer_kv_direct` 会执行：

```cpp
auto src_indices_cpu = src_indices.cpu();
auto dst_indices_cpu = dst_indices.cpu();
```

然后在 CPU 侧遍历连续区间，调用：

```cpp
copy_(..., /* non_blocking= */ true)
```

这条路径的特点：

```text
索引处理在 CPU 侧进行。
如果 src_indices/dst_indices 原本在 GPU 上，.cpu() 可能引入等待。
多个 page range 的 copy 提交由 CPU 循环组织。
不适合作为高并发搬运主路径。
```

因此，高并发或 overlap 场景优先使用 kernel 路径。

## 在 SGLang 中的典型调用方向

`memory_pool_host.py` 中主要有两个方向。

### Host 到 Device

入口：

```text
load_to_device_per_layer()
```

典型用途：

```text
host cache 命中后，把指定 token/page 的 KV 加载回 GPU cache。
```

常见接口：

```text
transfer_kv_per_layer
transfer_kv_per_layer_pf_lf
transfer_kv_per_layer_ph_lf
transfer_kv_per_layer_mla
```

执行方向：

```text
host_indices -> device_indices
```

### Device 到 Host

入口：

```text
backup_from_device_all_layer()
```

典型用途：

```text
GPU cache 压力较大时，把已生成或可复用的 KV 备份到 host cache。
```

常见接口：

```text
transfer_kv_all_layer
transfer_kv_all_layer_lf_pf
transfer_kv_all_layer_lf_ph
transfer_kv_all_layer_mla
transfer_kv_all_layer_mla_lf_pf
```

执行方向：

```text
device_indices -> host_indices
```

## 对 MUSA 的含义

MUSA 后端在 `common_extension_musa.cc` 中注册了同一组 `transfer_kv_*` API，dispatch key 是 `torch::kMUSA`。

这说明 MUSA 版本的并发语义与 CUDA 版本保持一致：

```text
Python 调用 torch.ops.sgl_kernel.transfer_kv_xxx
  -> MUSA dispatch
  -> kvcacheio transfer kernel
  -> 使用当前 PyTorch/MUSA stream
```

因此在 MUSA 上分析并发问题时，应重点检查：

| 检查项 | 说明 |
|---|---|
| 调用时所在 stream | 是否所有 transfer 都落在默认 stream |
| 上层是否显式创建 stream | 没有多 stream 就不会有 op 级并发 |
| dst_indices 是否冲突 | 冲突会导致结果不可靠 |
| attention 是否等待 transfer | 未同步会读到未完成写入的 KV |
| direct 路径是否触发 CPU 等待 | `.cpu()` 可能破坏异步调度 |

## 性能分析建议

分析 `kvcacheio` 并发效果时，不应只看单个 op 的耗时，需要同时看 stream 时间线。

建议观察：

```text
1. transfer_kv kernel 是否分布在多个 stream。
2. transfer_kv 与 attention/MLP 是否有时间重叠。
3. direct 路径是否出现 CPU 等待或隐式同步。
4. H2D / D2H 搬运是否被内存带宽限制。
5. 多个 transfer 并发后总耗时是否下降，而不是单个 kernel 变慢。
```

可用的验证方式：

```text
1. 在 Python 侧分别使用单 stream 和双 stream 提交 transfer。
2. 使用 profiler 查看 kernel 时间线。
3. 检查两个 transfer 的 dst_indices 是否完全不重叠。
4. 在 attention 前后加 stream wait，验证结果正确性。
5. 对比 kernel 路径和 direct 路径的耗时和同步行为。
```

## 判断标准

可以认为 `kvcacheio` 并发方案成立，需要满足：

```text
功能正确：
  并发前后 KV cache 内容一致。

无数据冲突：
  不同 transfer 的目标 KV slot 不重叠。

同步明确：
  后续 attention 读取 KV 前已经等待 transfer 完成。

性能有效：
  profiler 中能看到 stream overlap，端到端耗时下降。

路径合理：
  高并发场景优先使用 kernel 路径，避免 direct 路径的 CPU 索引处理成为瓶颈。
```

## 总结

`kvcacheio` 的单个 kernel 内部已经具备并行搬运能力。多个 `kvcacheio` 调用是否并发，取决于上层是否使用多个 stream，以及不同任务之间是否存在 KV cache 读写冲突。

默认调用只会使用当前 PyTorch stream；如果所有调用都在默认 stream 上，执行顺序就是串行。要实现 KV cache 搬运与计算重叠，需要显式设计 copy stream、compute stream、事件等待和 KV slot 分配规则。

`kernel` 路径适合并发调度。`direct` 路径包含 CPU 侧索引处理和 `copy_` 循环，更适合作为连续 page 搬运的简单路径，不适合作为高并发主路径。
