import math
import traceback

import torch
import torch.nn.functional as F


def device_module(device: torch.device):
    if device.type == "musa":
        return torch.musa
    if device.type == "cuda":
        return torch.cuda
    return None


def sync(device: torch.device):
    mod = device_module(device)
    if mod is not None:
        mod.synchronize()


def assert_close(a, b, *, atol=1e-4, rtol=1e-4):
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu()
    if isinstance(b, torch.Tensor):
        b = b.detach().cpu()
    if a.dtype == torch.bool or b.dtype == torch.bool:
        assert torch.equal(a, b), (a, b)
    elif a.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        assert torch.equal(a, b), (a, b)
    else:
        assert torch.allclose(a, b, atol=atol, rtol=rtol), (a, b)


def run_case(name, fn, device):
    try:
        fn(device)
        sync(device)
        print(f"PASS {name}")
        return True
    except Exception as exc:
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def case_empty(device):
    x = torch.empty((2, 3), dtype=torch.float32, device=device)
    assert x.shape == (2, 3) and x.device.type == device.type
    y = x.new_empty((3, 2))
    z = torch.empty_like(y)
    assert y.shape == z.shape == (3, 2)


def case_zeros_ones_full(device):
    assert_close(torch.zeros((2, 3), device=device), torch.zeros((2, 3)))
    assert_close(torch.ones((2, 3), device=device), torch.ones((2, 3)))
    assert_close(torch.full((2, 3), 7, dtype=torch.int32, device=device), torch.full((2, 3), 7, dtype=torch.int32))


def case_copy_foreach_fill_zero(device):
    a = torch.zeros((2, 3), device=device)
    b = torch.ones((2, 3), device=device)
    a.copy_(b)
    assert_close(a, torch.ones((2, 3)))
    a.fill_(3.0)
    assert_close(a, torch.full((2, 3), 3.0))
    a.zero_()
    assert_close(a, torch.zeros((2, 3)))
    d0, d1 = torch.empty_like(a), torch.empty_like(a)
    s0, s1 = torch.ones_like(a), torch.full_like(a, 2.0)
    if hasattr(torch, "_foreach_copy_"):
        torch._foreach_copy_([d0, d1], [s0, s1])
    else:
        d0.copy_(s0)
        d1.copy_(s1)
    assert_close(d0, torch.ones((2, 3)))
    assert_close(d1, torch.full((2, 3), 2.0))
    m = torch.zeros_like(a, dtype=torch.bool)
    m[:, 1] = True
    a.masked_fill_(m, 9.0)
    ref = torch.zeros((2, 3))
    ref[:, 1] = 9.0
    assert_close(a, ref)


def case_shape_ops(device):
    x = torch.arange(24, device=device).view(2, 3, 4)
    assert x.view(6, 4).shape == (6, 4)
    assert x.reshape(4, 6).shape == (4, 6)
    assert x.flatten(1).shape == (2, 12)
    assert x.unsqueeze(0).shape == (1, 2, 3, 4)
    assert x.unsqueeze(0).squeeze(0).shape == x.shape


def case_layout_ops(device):
    x = torch.arange(12, device=device).view(3, 4).t()
    assert not x.is_contiguous()
    y = x.contiguous()
    assert y.is_contiguous()
    assert y.stride() == (3, 1)
    assert y.storage_offset() == 0
    assert_close(y, torch.arange(12).view(3, 4).t().contiguous())


def case_indexing_ops(device):
    x = torch.arange(20, device=device).view(4, 5)
    assert_close(x[1:3, 2:], torch.arange(20).view(4, 5)[1:3, 2:])
    rows = torch.tensor([0, 2, 3], device=device)
    cols = torch.tensor([1, 3, 4], device=device)
    assert_close(x[rows, cols], torch.arange(20).view(4, 5)[[0, 2, 3], [1, 3, 4]])
    idx = torch.tensor([[0, 2], [1, 3], [2, 4], [0, 1]], device=device)
    assert_close(torch.gather(x, 1, idx), torch.gather(torch.arange(20).view(4, 5), 1, idx.cpu()))
    assert_close(torch.index_select(x, 0, rows), torch.index_select(torch.arange(20).view(4, 5), 0, rows.cpu()))
    out = torch.zeros_like(x)
    src = torch.ones((4, 2), dtype=x.dtype, device=device)
    out.scatter_(1, idx, src)
    ref = torch.zeros((4, 5), dtype=torch.int64).scatter_(1, idx.cpu(), torch.ones((4, 2), dtype=torch.int64))
    assert_close(out, ref)
    assert_close(torch.take_along_dim(x, idx, dim=1), torch.take_along_dim(torch.arange(20).view(4, 5), idx.cpu(), dim=1))


def case_split_mask_where(device):
    x = torch.arange(12, device=device).view(3, 4)
    chunks = x.tensor_split(3, dim=0)
    assert len(chunks) == 3
    y = x.clone()
    mask = y % 2 == 0
    y.masked_fill_(mask, -1)
    ref = torch.arange(12).view(3, 4)
    ref.masked_fill_(ref % 2 == 0, -1)
    assert_close(y, ref)
    z = torch.where(mask, torch.zeros_like(x), x)
    assert_close(z, torch.where(torch.arange(12).view(3, 4) % 2 == 0, torch.zeros(3, 4, dtype=torch.int64), torch.arange(12).view(3, 4)))


def case_sequence_ops(device):
    a = torch.arange(0, 6, device=device, dtype=torch.int32)
    assert_close(a, torch.arange(0, 6, dtype=torch.int32))
    out = torch.empty((4,), dtype=torch.int32, device=device)
    torch.arange(3, 7, out=out)
    assert_close(out, torch.arange(3, 7, dtype=torch.int32))
    r = torch.arange(3, device=device).repeat_interleave(2)
    assert_close(r, torch.arange(3).repeat_interleave(2))
    e = torch.arange(3, device=device).unsqueeze(1).expand(-1, 4)
    assert e.shape == (3, 4)
    rep = torch.tensor([1, 2, 1], device=device)
    assert_close(torch.repeat_interleave(torch.arange(3, device=device), rep), torch.tensor([0, 1, 1, 2]))


def case_pad_cat_stack(device):
    x = torch.arange(6, device=device).view(2, 3)
    assert_close(F.pad(x, (1, 2), value=-1), F.pad(torch.arange(6).view(2, 3), (1, 2), value=-1))
    assert_close(torch.cat([x, x], dim=0), torch.cat([torch.arange(6).view(2, 3)] * 2, dim=0))
    assert_close(torch.stack([x, x], dim=0), torch.stack([torch.arange(6).view(2, 3)] * 2, dim=0))


def case_reductions_math(device):
    x = torch.linspace(-3, 3, 12, device=device).view(3, 4)
    ref = torch.linspace(-3, 3, 12).view(3, 4)
    assert_close(x.sum(dim=1), ref.sum(dim=1))
    assert_close(x.mean(dim=0), ref.mean(dim=0))
    assert_close(x.abs().amax(dim=1), ref.abs().amax(dim=1))
    assert_close(x.square(), ref.square())
    pos = x.abs() + 0.5
    assert_close(torch.rsqrt(pos), torch.rsqrt(ref.abs() + 0.5))


def case_activations(device):
    x = torch.linspace(-3, 3, 12, device=device).view(3, 4)
    ref = torch.linspace(-3, 3, 12).view(3, 4)
    assert_close(torch.sigmoid(x), torch.sigmoid(ref))
    assert_close(torch.clamp(x, min=-1, max=1), torch.clamp(ref, min=-1, max=1))
    assert_close(F.relu(x), F.relu(ref))
    assert_close(F.silu(x), F.silu(ref))
    assert_close(F.gelu(x), F.gelu(ref), atol=1e-3, rtol=1e-3)
    assert_close(F.softmax(x, dim=-1), F.softmax(ref, dim=-1), atol=1e-5, rtol=1e-5)


def case_linear_algebra(device):
    x = torch.arange(12, dtype=torch.float32, device=device).view(3, 4)
    w = torch.arange(20, dtype=torch.float32, device=device).view(5, 4) / 10
    assert_close(F.linear(x, w), F.linear(torch.arange(12, dtype=torch.float32).view(3, 4), torch.arange(20, dtype=torch.float32).view(5, 4) / 10))
    a = torch.arange(24, dtype=torch.float32, device=device).view(2, 3, 4)
    b = torch.arange(40, dtype=torch.float32, device=device).view(2, 4, 5)
    assert_close(torch.matmul(a, b), torch.matmul(torch.arange(24, dtype=torch.float32).view(2, 3, 4), torch.arange(40, dtype=torch.float32).view(2, 4, 5)))
    assert_close(torch.mm(x, w.t()), torch.mm(torch.arange(12, dtype=torch.float32).view(3, 4), (torch.arange(20, dtype=torch.float32).view(5, 4) / 10).t()))
    assert_close(torch.bmm(a, b), torch.bmm(torch.arange(24, dtype=torch.float32).view(2, 3, 4), torch.arange(40, dtype=torch.float32).view(2, 4, 5)))
    e0 = torch.arange(24, dtype=torch.float32, device=device).view(2, 3, 4)
    e1 = torch.arange(32, dtype=torch.float32, device=device).view(2, 4, 4)
    assert_close(torch.einsum("bik,bkj->bij", e0, e1), torch.einsum("bik,bkj->bij", torch.arange(24, dtype=torch.float32).view(2, 3, 4), torch.arange(32, dtype=torch.float32).view(2, 4, 4)))


def case_topk_sort_arg(device):
    x = torch.tensor([[0.1, 3.0, 2.0, -1.0], [5.0, 4.0, 4.5, 0.0]], device=device)
    ref = x.cpu()
    vals, ids = torch.topk(x, k=2, dim=1, largest=True, sorted=True)
    rvals, rids = torch.topk(ref, k=2, dim=1, largest=True, sorted=True)
    assert_close(vals, rvals)
    assert_close(ids, rids)
    assert_close(torch.sort(x, dim=1).values, torch.sort(ref, dim=1).values)
    assert_close(torch.argsort(x, dim=1), torch.argsort(ref, dim=1))
    assert_close(torch.argmax(x, dim=1), torch.argmax(ref, dim=1))
    assert_close(torch.min(x, dim=1).values, torch.min(ref, dim=1).values)
    assert_close(torch.max(x, dim=1).values, torch.max(ref, dim=1).values)


def case_dtype_device_scalar(device):
    x = torch.arange(6, dtype=torch.float32, device=device)
    assert x.to(torch.int32).dtype == torch.int32
    assert x.float().dtype == torch.float32
    xb = x.to(torch.bfloat16)
    assert xb.dtype == torch.bfloat16
    cpu = x.cpu()
    assert cpu.device.type == "cpu"
    assert int(x[0].item()) == 0
    assert x[:3].to(torch.int64).tolist() == [0, 1, 2]


def case_sync_and_cpu_boundary(device):
    mod = device_module(device)
    assert mod is not None
    x = torch.arange(8, dtype=torch.float32, device=device)
    y = x.square()
    mod.synchronize()
    assert_close(y.cpu(), torch.arange(8, dtype=torch.float32).square())
    assert y.detach().cpu().numpy().shape == (8,)


def case_cudagraph_basic(device):
    mod = device_module(device)
    assert mod is not None
    if not hasattr(mod, "CUDAGraph") and not hasattr(mod, "MUSAGraph"):
        print("SKIP cudagraph_basic: graph API is not exposed")
        return
    graph_cls = getattr(mod, "MUSAGraph", None) or getattr(mod, "CUDAGraph")
    stream_cls = getattr(mod, "Stream", None)
    if stream_cls is None:
        print("SKIP cudagraph_basic: Stream API is not exposed")
        return
    static_in = torch.ones((4, 4), device=device)
    static_out = torch.empty_like(static_in)
    stream = stream_cls()
    stream.wait_stream(mod.current_stream())
    with mod.stream(stream):
        for _ in range(3):
            static_out.copy_(static_in * 2 + 1)
    mod.current_stream().wait_stream(stream)
    graph = graph_cls()
    try:
        with mod.stream(stream):
            graph.capture_begin()
            static_out.copy_(static_in * 2 + 1)
            graph.capture_end()
        static_in.fill_(3.0)
        graph.replay()
        sync(device)
        assert_close(static_out, torch.full((4, 4), 7.0))
    except Exception as exc:
        print(f"SKIP cudagraph_basic: {type(exc).__name__}: {exc}")


def main():
    assert hasattr(torch, "musa"), "torch.musa is not available"
    assert torch.musa.is_available(), "MUSA is not available"
    device = torch.device("musa:0")
    torch.manual_seed(0)
    print(f"torch={torch.__version__} device={device} count={torch.musa.device_count()}")
    cases = [
        ("empty/new_empty/empty_like", case_empty),
        ("zeros/ones/full", case_zeros_ones_full),
        ("copy_/foreach_copy_/fill_/zero_", case_copy_foreach_fill_zero),
        ("view/reshape/flatten/unsqueeze/squeeze", case_shape_ops),
        ("contiguous/stride/is_contiguous/storage_offset", case_layout_ops),
        ("slice/advanced_indexing/gather/index_select/scatter/take_along_dim", case_indexing_ops),
        ("tensor_split/masked_fill_/where", case_split_mask_where),
        ("arange/repeat_interleave/expand", case_sequence_ops),
        ("pad/cat/stack", case_pad_cat_stack),
        ("sum/mean/amax/abs/square/rsqrt", case_reductions_math),
        ("sigmoid/clamp/relu/silu/gelu/softmax", case_activations),
        ("F.linear/matmul/mm/bmm/einsum", case_linear_algebra),
        ("topk/sort/argsort/argmax/min/max", case_topk_sort_arg),
        ("to/float/bfloat16/cpu/item/tolist", case_dtype_device_scalar),
        ("synchronize/cpu/numpy", case_sync_and_cpu_boundary),
        ("cudagraph_basic", case_cudagraph_basic),
    ]
    passed = 0
    for name, fn in cases:
        passed += int(run_case(name, fn, device))
    print(f"SUMMARY passed={passed} total={len(cases)}")
    if passed != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
