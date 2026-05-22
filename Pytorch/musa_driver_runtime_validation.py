import json
import time

import torch
import torch.nn.functional as F


def sync():
    torch.musa.synchronize()


def tensor_summary(t):
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "device": str(t.device),
        "stride": list(t.stride()),
        "is_contiguous": t.is_contiguous(),
        "data_ptr": int(t.data_ptr()),
        "value": t.detach().cpu().tolist(),
    }


def print_case(name, data):
    print(f"CASE {name}")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def case_layout_and_copy(device):
    base = torch.arange(12, dtype=torch.float32, device=device).reshape(3, 4)
    viewed = base.view(2, 6)
    transposed = base.t()
    contiguous = transposed.contiguous()
    fixed = torch.empty_like(base)
    before_ptr = int(fixed.data_ptr())
    fixed.copy_(base)
    sync()
    after_ptr = int(fixed.data_ptr())

    return {
        "base": tensor_summary(base),
        "viewed": {
            "shape": list(viewed.shape),
            "stride": list(viewed.stride()),
            "shares_storage_with_base": int(viewed.data_ptr()) == int(base.data_ptr()),
        },
        "transposed": {
            "shape": list(transposed.shape),
            "stride": list(transposed.stride()),
            "is_contiguous": transposed.is_contiguous(),
        },
        "contiguous": {
            "shape": list(contiguous.shape),
            "stride": list(contiguous.stride()),
            "is_contiguous": contiguous.is_contiguous(),
            "new_storage_from_base": int(contiguous.data_ptr()) != int(base.data_ptr()),
        },
        "copy_": {
            "fixed_ptr_before": before_ptr,
            "fixed_ptr_after": after_ptr,
            "fixed_ptr_stable": before_ptr == after_ptr,
            "fixed_value": fixed.detach().cpu().tolist(),
        },
    }


def case_graph_replay(device):
    x = torch.zeros((2, 3), dtype=torch.float32, device=device)
    w = torch.tensor(
        [[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]],
        dtype=torch.float32,
        device=device,
    )
    out = torch.empty((2, 2), dtype=torch.float32, device=device)
    x_ptr = int(x.data_ptr())
    out_ptr = int(out.data_ptr())

    stream = torch.musa.Stream()
    stream.wait_stream(torch.musa.current_stream())
    with torch.musa.stream(stream):
        for _ in range(3):
            out.copy_(F.silu(x @ w))
    torch.musa.current_stream().wait_stream(stream)
    sync()

    graph = torch.musa.MUSAGraph()
    with torch.musa.stream(stream):
        graph.capture_begin()
        out.copy_(F.silu(x @ w))
        graph.capture_end()

    x.copy_(
        torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float32,
            device=device,
        )
    )
    graph.replay()
    sync()
    first = out.detach().cpu().tolist()

    x.copy_(
        torch.tensor(
            [[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]],
            dtype=torch.float32,
            device=device,
        )
    )
    graph.replay()
    sync()
    second = out.detach().cpu().tolist()

    return {
        "x_ptr_stable": x_ptr == int(x.data_ptr()),
        "out_ptr_stable": out_ptr == int(out.data_ptr()),
        "first_replay_out": first,
        "second_replay_out": second,
    }


def case_sync_boundaries(device):
    x = torch.arange(8, dtype=torch.float32, device=device)
    y = x.square().sum()
    sync()

    item_start = time.perf_counter()
    scalar = float(y.item())
    item_ms = (time.perf_counter() - item_start) * 1000

    cpu_start = time.perf_counter()
    small_list = x[:4].cpu().tolist()
    cpu_ms = (time.perf_counter() - cpu_start) * 1000

    return {
        "item_value": scalar,
        "item_elapsed_ms_single_run": round(item_ms, 6),
        "cpu_list": small_list,
        "cpu_elapsed_ms_single_run": round(cpu_ms, 6),
        "note": "single-run timing is only a sync-boundary smoke test, not a benchmark",
    }


def case_dynamic_shape(device):
    mask_a = torch.tensor([True, False, True, False, False, True], device=device)
    mask_b = torch.tensor([False, True, False, False, False, False], device=device)
    idx_a = torch.nonzero(mask_a, as_tuple=False)
    idx_b = torch.nonzero(mask_b, as_tuple=False)
    fixed_a = torch.where(mask_a, torch.ones_like(mask_a, dtype=torch.int32), torch.zeros_like(mask_a, dtype=torch.int32))
    fixed_b = torch.where(mask_b, torch.ones_like(mask_b, dtype=torch.int32), torch.zeros_like(mask_b, dtype=torch.int32))
    sync()

    return {
        "nonzero_shape_a": list(idx_a.shape),
        "nonzero_shape_b": list(idx_b.shape),
        "nonzero_value_a": idx_a.cpu().tolist(),
        "nonzero_value_b": idx_b.cpu().tolist(),
        "fixed_mask_shape_a": list(fixed_a.shape),
        "fixed_mask_shape_b": list(fixed_b.shape),
        "fixed_mask_value_a": fixed_a.cpu().tolist(),
        "fixed_mask_value_b": fixed_b.cpu().tolist(),
    }


def main():
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("torch.musa is not available")

    device = torch.device("musa:0")
    print_case(
        "environment",
        {
            "torch_version": torch.__version__,
            "device": str(device),
            "device_count": torch.musa.device_count(),
            "current_device": torch.musa.current_device(),
            "device_name": torch.musa.get_device_name(0),
        },
    )
    print_case("layout_and_copy", case_layout_and_copy(device))
    print_case("graph_replay_fixed_address", case_graph_replay(device))
    print_case("sync_boundaries", case_sync_boundaries(device))
    print_case("dynamic_shape", case_dynamic_shape(device))


if __name__ == "__main__":
    main()
