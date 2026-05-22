import math
import traceback

import torch
import torch.nn.functional as F


def sync():
    torch.musa.synchronize()


def cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return x


def assert_equal(actual, expected):
    actual = cpu(actual)
    expected = cpu(expected)
    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def assert_close(actual, expected, atol=1e-4, rtol=1e-4):
    actual = cpu(actual)
    expected = cpu(expected)
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol), (actual, expected)


def run(name, fn):
    try:
        fn()
        sync()
        print(f"PASS {name}")
        return True
    except Exception as exc:
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def test_create_copy_update(device):
    x = torch.empty((2, 3), dtype=torch.float32, device=device)
    assert x.shape == (2, 3) and x.dtype == torch.float32
    y = x.new_empty((3, 2))
    assert y.shape == (3, 2) and y.dtype == x.dtype and y.device == x.device
    z = torch.empty_like(torch.tensor([[1, 2], [3, 4]], device=device))
    assert z.shape == (2, 2) and z.dtype == torch.int64 and z.device.type == "musa"

    assert_equal(torch.zeros((2, 3), dtype=torch.int64, device=device), torch.tensor([[0, 0, 0], [0, 0, 0]]))
    assert_equal(torch.ones((2, 3), dtype=torch.int64, device=device), torch.tensor([[1, 1, 1], [1, 1, 1]]))
    assert_equal(torch.full((2, 3), 7, device=device), torch.tensor([[7, 7, 7], [7, 7, 7]]))

    dst = torch.tensor([0, 0, 0], device=device)
    src = torch.tensor([1, 2, 3], device=device)
    dst.copy_(src)
    assert_equal(dst, torch.tensor([1, 2, 3]))

    dst0, dst1 = torch.tensor([0, 0], device=device), torch.tensor([0, 0], device=device)
    src0, src1 = torch.tensor([1, 1], device=device), torch.tensor([2, 2], device=device)
    torch._foreach_copy_([dst0, dst1], [src0, src1])
    assert_equal(dst0, torch.tensor([1, 1]))
    assert_equal(dst1, torch.tensor([2, 2]))

    a = torch.tensor([0, 0, 0], device=device)
    a.fill_(3)
    assert_equal(a, torch.tensor([3, 3, 3]))
    a.zero_()
    assert_equal(a, torch.tensor([0, 0, 0]))
    b = torch.tensor([1, 2, 3, 4], device=device)
    mask = torch.tensor([True, False, True, False], device=device)
    b.masked_fill_(mask, -1)
    assert_equal(b, torch.tensor([-1, 2, -1, 4]))


def test_shape_layout_broadcast(device):
    x = torch.arange(6, device=device)
    assert_equal(x.view(2, 3), torch.tensor([[0, 1, 2], [3, 4, 5]]))
    assert_equal(x.reshape(3, 2), torch.tensor([[0, 1], [2, 3], [4, 5]]))
    assert torch.empty((2, 3, 4), device=device).flatten(1).shape == (2, 12)
    assert torch.empty((3,), device=device).unsqueeze(0).shape == (1, 3)
    assert torch.empty((1, 3, 1), device=device).squeeze(-1).shape == (1, 3)
    base = torch.tensor([[0], [1], [2]], device=device)
    assert_equal(base.expand(-1, 4), torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]))

    t = torch.arange(6, device=device).view(2, 3).t()
    c = t.contiguous()
    assert_equal(c, torch.tensor([[0, 3], [1, 4], [2, 5]]))
    assert torch.arange(6, device=device).view(2, 3).stride() == (3, 1)
    assert not torch.arange(6, device=device).view(2, 3).t().is_contiguous()
    assert torch.arange(6, device=device)[2:].storage_offset() == 2


def test_index_split_mapping(device):
    x = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], device=device)
    assert_equal(x[1:, :2], torch.tensor([[3, 4], [6, 7]]))
    rows = torch.tensor([0, 2], device=device)
    cols = torch.tensor([1, 2], device=device)
    assert_equal(x[rows, cols], torch.tensor([1, 8]))

    g = torch.tensor([[10, 11, 12], [20, 21, 22]], device=device)
    idx = torch.tensor([[2, 0], [1, 1]], device=device)
    assert_equal(torch.gather(g, 1, idx), torch.tensor([[12, 10], [21, 21]]))
    assert_equal(torch.take_along_dim(g, idx, dim=1), torch.tensor([[12, 10], [21, 21]]))

    s = torch.tensor([[0, 1], [2, 3], [4, 5]], device=device)
    assert_equal(torch.index_select(s, 0, torch.tensor([2, 0], device=device)), torch.tensor([[4, 5], [0, 1]]))

    out = torch.zeros((2, 3), dtype=torch.int64, device=device)
    scatter_idx = torch.tensor([[0, 2], [0, 1]], device=device)
    scatter_src = torch.tensor([[5, 6], [7, 8]], device=device)
    out.scatter_(1, scatter_idx, scatter_src)
    assert_equal(out, torch.tensor([[5, 0, 6], [7, 8, 0]]))

    chunks = torch.arange(6, device=device).tensor_split(3)
    assert len(chunks) == 3
    assert_equal(chunks[0], torch.tensor([0, 1]))
    assert_equal(chunks[1], torch.tensor([2, 3]))
    assert_equal(chunks[2], torch.tensor([4, 5]))


def test_sequence_join_condition(device):
    assert_equal(torch.arange(3, 7, device=device), torch.tensor([3, 4, 5, 6]))
    out = torch.empty((4,), dtype=torch.int64, device=device)
    torch.arange(3, 7, out=out)
    assert_equal(out, torch.tensor([3, 4, 5, 6]))
    assert_equal(torch.tensor([0, 1, 2], device=device).repeat_interleave(2), torch.tensor([0, 0, 1, 1, 2, 2]))
    repeats = torch.tensor([1, 2, 1], device=device)
    assert_equal(torch.repeat_interleave(torch.tensor([0, 1, 2], device=device), repeats), torch.tensor([0, 1, 1, 2]))
    a = torch.tensor([1, 2], device=device)
    b = torch.tensor([3, 4], device=device)
    assert_equal(torch.cat([a, b], dim=0), torch.tensor([1, 2, 3, 4]))
    assert_equal(torch.stack([a, b], dim=0), torch.tensor([[1, 2], [3, 4]]))
    assert_equal(F.pad(torch.tensor([[1, 2, 3]], device=device), (1, 2), value=0), torch.tensor([[0, 1, 2, 3, 0, 0]]))
    cond = torch.tensor([True, False, True], device=device)
    assert_equal(torch.where(cond, torch.tensor([1, 1, 1], device=device), torch.tensor([2, 2, 2], device=device)), torch.tensor([1, 2, 1]))


def test_math_reduction_activation(device):
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device)
    assert_close(x.sum(dim=1), torch.tensor([6.0, 15.0]))
    assert_close(x.mean(dim=0), torch.tensor([2.5, 3.5, 4.5]))
    assert_close(torch.tensor([[-1, 3], [5, 2]], dtype=torch.float32, device=device).amax(dim=1), torch.tensor([3.0, 5.0]))
    y = torch.tensor([[3, 1], [2, 5]], dtype=torch.float32, device=device)
    assert_close(y.min(dim=1).values, torch.tensor([1.0, 2.0]))
    assert_close(y.max(dim=1).values, torch.tensor([3.0, 5.0]))
    assert_close(torch.tensor([-2, 0, 3], dtype=torch.float32, device=device).abs(), torch.tensor([2.0, 0.0, 3.0]))
    assert_close(torch.tensor([-2, 3], dtype=torch.float32, device=device).square(), torch.tensor([4.0, 9.0]))
    assert_close(torch.rsqrt(torch.tensor([4, 16], dtype=torch.float32, device=device)), torch.tensor([0.5, 0.25]))
    assert_close(torch.sigmoid(torch.tensor([0.0], device=device)), torch.tensor([0.5]))
    assert_close(F.silu(torch.tensor([0.0, 1.0], device=device)), torch.tensor([0.0, 0.7310586]))
    assert_close(F.gelu(torch.tensor([0.0, 1.0], device=device)), torch.tensor([0.0, 0.8413447]), atol=1e-3, rtol=1e-3)
    assert_close(F.relu(torch.tensor([-1.0, 0.0, 2.0], device=device)), torch.tensor([0.0, 0.0, 2.0]))
    assert_close(F.softmax(torch.tensor([1.0, 2.0, 3.0], device=device), dim=0), torch.tensor([0.0900306, 0.2447285, 0.6652409]))
    assert_close(torch.clamp(torch.tensor([-2.0, 0.0, 3.0], device=device), min=-1, max=1), torch.tensor([-1.0, 0.0, 1.0]))


def test_linalg_sort_route(device):
    x = torch.tensor([[1.0, 2.0]], device=device)
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], device=device)
    bias = torch.tensor([0.0, 0.0, 1.0], device=device)
    assert_close(F.linear(x, weight, bias), torch.tensor([[1.0, 2.0, 4.0]]))

    assert torch.matmul(torch.empty((2, 3), device=device), torch.empty((3, 4), device=device)).shape == (2, 4)
    assert_close(torch.mm(torch.tensor([[1.0, 2.0]], device=device), torch.tensor([[3.0], [4.0]], device=device)), torch.tensor([[11.0]]))
    assert torch.bmm(torch.empty((2, 3, 4), device=device), torch.empty((2, 4, 5), device=device)).shape == (2, 3, 5)
    assert torch.einsum("bik,bkj->bij", torch.empty((2, 3, 4), device=device), torch.empty((2, 4, 5), device=device)).shape == (2, 3, 5)

    scores = torch.tensor([0.1, 3.0, 2.0, -1.0], device=device)
    vals, ids = torch.topk(scores, k=2)
    assert_close(vals, torch.tensor([3.0, 2.0]))
    assert_equal(ids, torch.tensor([1, 2]))
    sorted_vals, sorted_ids = torch.sort(torch.tensor([3, 1, 2], device=device))
    assert_equal(sorted_vals, torch.tensor([1, 2, 3]))
    assert_equal(sorted_ids, torch.tensor([1, 2, 0]))
    assert_equal(torch.argsort(torch.tensor([3, 1, 2], device=device)), torch.tensor([1, 2, 0]))
    assert_equal(torch.argmax(torch.tensor([3, 1, 5, 2], device=device)), torch.tensor(2))


def test_dtype_cpu_sync_graph(device):
    x = torch.tensor([1.2, 2.8], device=device)
    assert_equal(x.to(torch.int32), torch.tensor([1, 2], dtype=torch.int32))
    assert torch.tensor([1, 2], dtype=torch.int32, device=device).float().dtype == torch.float32
    assert torch.tensor([1.0, 2.0], device=device).bfloat16().dtype == torch.bfloat16
    cpu_x = torch.tensor([1, 2], device=device).cpu()
    assert cpu_x.device.type == "cpu"
    assert_equal(torch.tensor([1, 2]).numpy().tolist(), [1, 2])
    assert torch.tensor([7], device=device).item() == 7
    assert torch.tensor([1, 2, 3], device=device).tolist() == [1, 2, 3]
    torch.musa.synchronize()

    graph_cls = getattr(torch.musa, "MUSAGraph", None) or getattr(torch.musa, "CUDAGraph")
    inp = torch.ones((2, 2), device=device)
    out = torch.empty_like(inp)
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        for _ in range(3):
            out.copy_(inp * 2 + 1)
    torch.musa.current_stream().wait_stream(stream)
    graph = graph_cls()
    with torch.musa.stream(stream):
        graph.capture_begin()
        out.copy_(inp * 2 + 1)
        graph.capture_end()
    inp.fill_(3)
    graph.replay()
    sync()
    assert_close(out, torch.full((2, 2), 7.0))


def main():
    assert hasattr(torch, "musa"), "torch.musa is not available"
    assert torch.musa.is_available(), "MUSA is not available"
    device = torch.device("musa:0")
    print(f"torch={torch.__version__} device={device} count={torch.musa.device_count()}")
    tests = [
        ("create_copy_update_examples", lambda: test_create_copy_update(device)),
        ("shape_layout_broadcast_examples", lambda: test_shape_layout_broadcast(device)),
        ("index_split_mapping_examples", lambda: test_index_split_mapping(device)),
        ("sequence_join_condition_examples", lambda: test_sequence_join_condition(device)),
        ("math_reduction_activation_examples", lambda: test_math_reduction_activation(device)),
        ("linalg_sort_route_examples", lambda: test_linalg_sort_route(device)),
        ("dtype_cpu_sync_graph_examples", lambda: test_dtype_cpu_sync_graph(device)),
    ]
    passed = 0
    for name, fn in tests:
        passed += int(run(name, fn))
    print(f"SUMMARY passed={passed} total={len(tests)}")
    if passed != len(tests):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
