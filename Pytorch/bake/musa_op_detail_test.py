#!/usr/bin/env python3
"""MUSA Pytorch Op Individual Test — with detailed input/output logging."""
import traceback
import torch
import torch.nn.functional as F

PASSED, FAILED = 0, 0
DEVICE = torch.device("musa:0")

def sync():
    torch.musa.synchronize()

def cpu(x):
    return x.detach().cpu() if isinstance(x, torch.Tensor) else x

def run(name, fn):
    global PASSED, FAILED
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        fn()
        sync()
        print(f"RESULT: PASS")
        PASSED += 1
    except Exception as e:
        print(f"RESULT: FAIL — {type(e).__name__}: {e}")
        traceback.print_exc()
        FAILED += 1

def show(desc, tensor):
    if isinstance(tensor, torch.Tensor):
        t = cpu(tensor)
        print(f"  {desc}: shape={tuple(t.shape)} dtype={t.dtype} device=cpu")
        if t.numel() <= 24:
            print(f"    values={t.tolist()}")
        elif t.numel() <= 128:
            print(f"    values={t.flatten().tolist()}")
    elif isinstance(tensor, (list, tuple)):
        if len(tensor) <= 8:
            for i, t in enumerate(tensor):
                if isinstance(t, torch.Tensor):
                    print(f"  {desc}[{i}]: shape={tuple(t.shape)} device=cpu")
                    if t.numel() <= 12:
                        print(f"    values={t.flatten().tolist()}")
        else:
            print(f"  {desc}: {len(tensor)} elements")
    else:
        print(f"  {desc}: {tensor}")

# ═══════════════════════════════════════════════════════════════
# 1. TENSOR CREATION
# ═══════════════════════════════════════════════════════════════

def test_empty():
    print("  [torch.empty] 分配未初始化的tensor")
    x = torch.empty((2, 3), dtype=torch.float32, device=DEVICE)
    show("empty(2x3, float32)", x)
    y = x.new_empty((4,))
    show("new_empty((4,))", y)
    z = torch.empty_like(torch.tensor([[1,2],[3,4]], device=DEVICE))
    show("empty_like(int64 2x2)", z)

def test_zeros_ones_full():
    print("  [torch.zeros/ones/full] 分配并初始化")
    z = torch.zeros((3, 2), dtype=torch.int64, device=DEVICE)
    show("zeros(3x2, int64)", z)
    o = torch.ones((3, 2), dtype=torch.int64, device=DEVICE)
    show("ones(3x2, int64)", o)
    f = torch.full((3, 2), 7, device=DEVICE)
    show("full(3x2, 7)", f)

# ═══════════════════════════════════════════════════════════════
# 2. COPY & INPLACE
# ═══════════════════════════════════════════════════════════════

def test_copy():
    print("  [copy_] 原地复制(地址不变)，CUDA Graph replay 核心")
    a = torch.zeros((2, 3), device=DEVICE)
    b = torch.ones((2, 3), device=DEVICE)
    show("before copy_: a", a)
    a.copy_(b)
    show("after a.copy_(b)", a)

def test_fill_ops():
    print("  [fill_ / zero_ / masked_fill_] 原地填充")
    a = torch.tensor([0., 0., 0.], device=DEVICE)
    a.fill_(3.0)
    show("fill_(3.0)", a)
    a.zero_()
    show("zero_()", a)
    x = torch.arange(6, device=DEVICE)
    m = x % 2 == 0
    x.masked_fill_(m, -1)
    show("masked_fill_(even→-1)", x)

def test_foreach_copy():
    print("  [torch._foreach_copy_] 批量原地复制，减少Python循环")
    d0, d1 = torch.zeros(3, device=DEVICE), torch.zeros(3, device=DEVICE)
    s0, s1 = torch.tensor([1,2,3], device=DEVICE), torch.tensor([4,5,6], device=DEVICE)
    if hasattr(torch, "_foreach_copy_"):
        torch._foreach_copy_([d0, d1], [s0, s1])
    else:
        d0.copy_(s0); d1.copy_(s1)
    show("dst0 after batch copy", d0)
    show("dst1 after batch copy", d1)

# ═══════════════════════════════════════════════════════════════
# 3. SHAPE & LAYOUT
# ═══════════════════════════════════════════════════════════════

def test_view_reshape():
    print("  [view/reshape/flatten/unsqueeze/squeeze]")
    x = torch.arange(12, device=DEVICE)
    show("x=arange(12)", x)
    show("x.view(3,4)", x.view(3, 4))
    show("x.reshape(4,3)", x.reshape(4, 3))
    show("x.view(2,3,2).flatten(1)", x.view(2,3,2).flatten(1))
    show("x[:3].unsqueeze(0)", x[:3].unsqueeze(0))
    show("x[:3].unsqueeze(0).squeeze(0)", x[:3].unsqueeze(0).squeeze(0))

def test_contiguous():
    print("  [contiguous/stride] 转置后需contiguous才能进fused kernel")
    x = torch.arange(6, device=DEVICE).view(2, 3)
    xt = x.t()
    print(f"  x.view(2,3).t(): shape={tuple(xt.shape)} stride={xt.stride()} is_contiguous={xt.is_contiguous()}")
    xc = xt.contiguous()
    print(f"  contiguous(): shape={tuple(xc.shape)} stride={xc.stride()} is_contiguous={xc.is_contiguous()}")
    show("contiguous values", xc)

# ═══════════════════════════════════════════════════════════════
# 4. INDEXING
# ═══════════════════════════════════════════════════════════════

def test_indexing():
    print("  [slice / advanced indexing / gather / scatter] — PageTable & MoE dispatch 核心")
    x = torch.arange(20, device=DEVICE).view(4, 5)
    show("x=arange(20).view(4,5)", x)
    show("x[1:3, 2:] (slice)", x[1:3, 2:])
    rows = torch.tensor([0, 2, 3], device=DEVICE)
    cols = torch.tensor([1, 3, 4], device=DEVICE)
    show("x[[0,2,3],[1,3,4]] (advanced)", x[rows, cols])

    # gather
    idx = torch.tensor([[0, 2], [1, 3], [2, 4], [0, 1]], device=DEVICE)
    show("gather(x, dim=1, idx)", torch.gather(x, 1, idx))

    # scatter
    out = torch.zeros((4, 5), dtype=torch.int64, device=DEVICE)
    src = torch.ones((4, 2), dtype=torch.int64, device=DEVICE)
    out.scatter_(1, idx, src)
    show("scatter_(dim=1, idx, ones) → out", out)

    # index_select
    show("index_select(x, dim=0, [2,0])", torch.index_select(x, 0, torch.tensor([2,0], device=DEVICE)))

    # tensor_split
    chunks = x.tensor_split(2, dim=1)
    show("tensor_split(2, dim=1)", chunks)

def test_where():
    print("  [torch.where] 条件选择")
    x = torch.arange(6, device=DEVICE)
    cond = x % 2 == 0
    show("x", x)
    show("where(even, zeros, x)", torch.where(cond, torch.zeros_like(x), x))

# ═══════════════════════════════════════════════════════════════
# 5. SEQUENCE OPS
# ═══════════════════════════════════════════════════════════════

def test_arange_expand():
    print("  [arange / repeat_interleave / expand] — position ids & batch 展开")
    a = torch.arange(3, 7, device=DEVICE)
    show("arange(3,7)", a)
    r = torch.arange(3, device=DEVICE).repeat_interleave(2)
    show("arange(3).repeat_interleave(2)", r)
    reps = torch.tensor([1, 2, 1], device=DEVICE)
    show("repeat_interleave(arange(3), [1,2,1])", torch.repeat_interleave(torch.arange(3, device=DEVICE), reps))

# ═══════════════════════════════════════════════════════════════
# 6. PAD / CAT / STACK
# ═══════════════════════════════════════════════════════════════

def test_pad_cat_stack():
    print("  [F.pad / cat / stack] — CUDA Graph padding & TP all-gather 拼接")
    x = torch.arange(6, device=DEVICE).view(2, 3)
    show("x=arange(6).view(2,3)", x)
    show("F.pad(x, (1,2), value=-1)", F.pad(x, (1, 2), value=-1))
    show("cat([x,x], dim=0)", torch.cat([x, x], dim=0))
    show("stack([x,x], dim=0)", torch.stack([x, x], dim=0))

# ═══════════════════════════════════════════════════════════════
# 7. MATH REDUCTIONS
# ═══════════════════════════════════════════════════════════════

def test_math():
    print("  [sum/mean/amax/abs/square/rsqrt] — RMSNorm & quantization scale")
    x = torch.linspace(-3, 3, 12, device=DEVICE).view(3, 4)
    show("x=linspace(-3,3,12).view(3,4)", x)
    show("sum(dim=1)", x.sum(dim=1))
    show("mean(dim=0)", x.mean(dim=0))
    show("abs().amax(dim=1)", x.abs().amax(dim=1))
    show("square()", x.square())
    pos = x.abs() + 0.5
    show("rsqrt(abs(x)+0.5)", torch.rsqrt(pos))

# ═══════════════════════════════════════════════════════════════
# 8. ACTIVATIONS
# ═══════════════════════════════════════════════════════════════

def test_activations():
    print("  [sigmoid/silu/gelu/relu/softmax/clamp] — gating & attention")
    x = torch.linspace(-3, 3, 6, device=DEVICE)
    show("x=linspace(-3,3,6)", x)
    show("sigmoid(x)", torch.sigmoid(x))
    show("silu(x)", F.silu(x))
    show("gelu(x)", F.gelu(x))
    show("relu(x)", F.relu(x))
    show("softmax(x, dim=0)", F.softmax(x, dim=0))
    show("clamp(x, -1, 1)", torch.clamp(x, -1, 1))

# ═══════════════════════════════════════════════════════════════
# 9. LINEAR ALGEBRA
# ═══════════════════════════════════════════════════════════════

def test_linalg():
    print("  [F.linear / matmul / mm / bmm / einsum] — projection & attention")
    # F.linear
    x = torch.arange(6, dtype=torch.float32, device=DEVICE).view(2, 3)
    w = torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,1.,1.]], device=DEVICE)
    b = torch.tensor([1., 0., 0., 2.], device=DEVICE)
    show("x", x)
    show("F.linear(x, w, bias)", F.linear(x, w, b))

    # mm
    a = torch.tensor([[1.,2.],[3.,4.]], device=DEVICE)
    bm = torch.tensor([[5.,6.],[7.,8.]], device=DEVICE)
    show("mm(a, b)", torch.mm(a, bm))

    # bmm
    ba = torch.arange(8, dtype=torch.float32, device=DEVICE).view(2, 2, 2)
    bb = torch.arange(8, dtype=torch.float32, device=DEVICE).view(2, 2, 2) + 1
    show("ba", ba)
    show("bb", bb)
    show("bmm(ba, bb)", torch.bmm(ba, bb))

    # einsum
    e0 = torch.arange(6, dtype=torch.float32, device=DEVICE).view(2, 3)
    e1 = torch.eye(3, device=DEVICE)
    show("einsum('ij,jk->ik', e0, I)", torch.einsum("ij,jk->ik", e0, e1))

# ═══════════════════════════════════════════════════════════════
# 10. TOPK / SORT / ROUTING
# ═══════════════════════════════════════════════════════════════

def test_topk_sort():
    print("  [topk / sort / argsort / argmax / min / max] — MoE routing & sampling")
    scores = torch.tensor([0.1, 3.0, 2.0, -1.0], device=DEVICE)
    show("scores", scores)
    vals, ids = torch.topk(scores, k=2)
    show("topk(scores, k=2).values", vals)
    show("topk(scores, k=2).indices", ids)

    x = torch.tensor([3., 1., 2.], device=DEVICE)
    sv, si = torch.sort(x)
    show("sort([3,1,2]).values", sv)
    show("sort([3,1,2]).indices", si)
    show("argsort([3,1,2])", torch.argsort(x))
    show("argmax([3,1,5,2])", torch.argmax(torch.tensor([3.,1.,5.,2.], device=DEVICE)))

# ═══════════════════════════════════════════════════════════════
# 11. DTYPE / DEVICE / SCALAR
# ═══════════════════════════════════════════════════════════════

def test_dtype_device():
    print("  [to/float/bfloat16/cpu/item/tolist] — 精度转换 & 同步边界")
    x = torch.tensor([1.5, 2.8, 3.1], device=DEVICE)
    show("x", x)
    show("x.to(int32)", x.to(torch.int32))
    print(f"  x.float().dtype = {x.float().dtype}")
    print(f"  x.bfloat16().dtype = {x.to(torch.bfloat16).dtype}")
    show("x.cpu()", x.cpu())
    print(f"  x[0].item() = {x[0].item()}")
    show("x.tolist()", x.tolist())

# ═══════════════════════════════════════════════════════════════
# 12. SYNC & GRAPH
# ═══════════════════════════════════════════════════════════════

def test_sync_graph():
    print("  [MUSA Graph capture/replay] — decode 热路径加速")
    graph_cls = getattr(torch.musa, "MUSAGraph", None)
    if graph_cls is None:
        print("  SKIP: MUSAGraph not available")
        return

    static_in = torch.ones((3, 3), device=DEVICE)
    static_out = torch.empty_like(static_in)
    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())

    # warmup
    with torch.musa.stream(stream):
        for _ in range(2):
            static_out.copy_(static_in * 2 + 1)
    torch.musa.current_stream().wait_stream(stream)

    # capture
    graph = graph_cls()
    with torch.musa.stream(stream):
        graph.capture_begin()
        static_out.copy_(static_in * 2 + 1)
        graph.capture_end()

    # replay
    static_in.fill_(5.0)
    show("static_in (replay input)", static_in)
    graph.replay()
    sync()
    show("static_out (replay output: 5*2+1=11)", static_out)

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    assert torch.musa.is_available(), "MUSA not available"
    print(f"ENV: torch={torch.__version__} device={DEVICE} count={torch.musa.device_count()}")

    tests = [
        # 1. Creation
        ("1a empty/new_empty/empty_like", test_empty),
        ("1b zeros/ones/full", test_zeros_ones_full),
        # 2. Copy & Inplace
        ("2a copy_ (CUDA Graph core)", test_copy),
        ("2b fill_/zero_/masked_fill_", test_fill_ops),
        ("2c _foreach_copy_ (batch copy)", test_foreach_copy),
        # 3. Shape & Layout
        ("3a view/reshape/flatten/unsqueeze/squeeze", test_view_reshape),
        ("3b contiguous/stride", test_contiguous),
        # 4. Indexing
        ("4a slice/adv_index/gather/scatter/index_select/split", test_indexing),
        ("4b torch.where", test_where),
        # 5. Sequence
        ("5a arange/repeat_interleave/expand", test_arange_expand),
        # 6. Pad/Cat/Stack
        ("6a F.pad/cat/stack", test_pad_cat_stack),
        # 7. Math
        ("7a sum/mean/amax/abs/square/rsqrt", test_math),
        # 8. Activations
        ("8a sigmoid/silu/gelu/relu/softmax/clamp", test_activations),
        # 9. Linalg
        ("9a F.linear/matmul/mm/bmm/einsum", test_linalg),
        # 10. TopK
        ("10a topk/sort/argsort/argmax/min/max", test_topk_sort),
        # 11. Dtype/Device
        ("11a to/float/bfloat16/cpu/item/tolist", test_dtype_device),
        # 12. Graph
        ("12a MUSAGraph capture/replay", test_sync_graph),
    ]

    for name, fn in tests:
        run(name, fn)

    print(f"\n{'='*60}")
    print(f"SUMMARY: passed={PASSED} failed={FAILED} total={PASSED+FAILED}")
    print(f"{'='*60}")
    if FAILED > 0:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
