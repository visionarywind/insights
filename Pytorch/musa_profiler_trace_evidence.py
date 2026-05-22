import json

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function


def sync():
    torch.musa.synchronize()


def event_value(event, names):
    for name in names:
        if hasattr(event, name):
            return getattr(event, name)
    return 0


def summarize_events(prof):
    keywords = [
        "contiguous",
        "copy",
        "item",
        "local_scalar",
        "nonzero",
        "where",
        "empty",
        "mul",
        "sum",
        "graph",
        "case_",
    ]
    rows = []
    for event in prof.key_averages():
        key = event.key
        if not any(word in key.lower() for word in keywords):
            continue
        rows.append(
            {
                "key": key,
                "count": int(event.count),
                "self_cpu_us": round(float(event.self_cpu_time_total), 3),
                "cpu_total_us": round(float(event.cpu_time_total), 3),
                "self_device_us": round(
                    float(
                        event_value(
                            event,
                            [
                                "self_musa_time_total",
                                "self_device_time_total",
                                "self_cuda_time_total",
                            ],
                        )
                    ),
                    3,
                ),
                "device_total_us": round(
                    float(
                        event_value(
                            event,
                            [
                                "musa_time_total",
                                "device_time_total",
                                "cuda_time_total",
                            ],
                        )
                    ),
                    3,
                ),
            }
        )
    rows.sort(key=lambda x: (x["self_device_us"], x["self_cpu_us"]), reverse=True)
    return rows


def build_graph_case(device):
    x = torch.zeros((2, 3), dtype=torch.float32, device=device)
    w = torch.tensor(
        [[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]],
        dtype=torch.float32,
        device=device,
    )
    out = torch.empty((2, 2), dtype=torch.float32, device=device)

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
    sync()
    return graph, x, w, out


def run_workload(device, graph, graph_x):
    with record_function("case_layout_contiguous_copy"):
        base = torch.arange(512 * 512, dtype=torch.float32, device=device).reshape(512, 512)
        transposed = base.t()
        contiguous = transposed.contiguous()
        fixed = torch.empty_like(contiguous)
        fixed.copy_(contiguous)
        reduced = fixed.square().sum()

    with record_function("case_scalar_item"):
        scalar = float(reduced.item())

    with record_function("case_dynamic_nonzero_where"):
        mask = torch.tensor([True, False, True, False, False, True], device=device)
        idx = torch.nonzero(mask, as_tuple=False)
        fixed_mask = torch.where(
            mask,
            torch.ones_like(mask, dtype=torch.int32),
            torch.zeros_like(mask, dtype=torch.int32),
        )

    with record_function("case_graph_copy_replay"):
        graph_x.copy_(
            torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=torch.float32,
                device=device,
            )
        )
        graph.replay()
        sync()

    return {
        "scalar": scalar,
        "nonzero_shape": list(idx.shape),
        "fixed_mask_shape": list(fixed_mask.shape),
    }


def main():
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        raise RuntimeError("torch.musa is not available")

    device = torch.device("musa:0")
    activities = [ProfilerActivity.CPU]
    if hasattr(ProfilerActivity, "MUSA"):
        activities.append(ProfilerActivity.MUSA)

    graph, graph_x, graph_w, graph_out = build_graph_case(device)
    run_workload(device, graph, graph_x)
    sync()

    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        result = run_workload(device, graph, graph_x)
        sync()

    print("ENV")
    print(
        json.dumps(
            {
                "torch_version": torch.__version__,
                "device": str(device),
                "device_name": torch.musa.get_device_name(0),
                "activities": [str(activity).split(".")[-1] for activity in activities],
                "result": result,
                "graph_weight_shape": list(graph_w.shape),
                "graph_out": graph_out.detach().cpu().tolist(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("KEY_AVERAGES_BY_CPU")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    print("KEY_AVERAGES_BY_MUSA")
    try:
        print(prof.key_averages().table(sort_by="self_musa_time_total", row_limit=20))
    except Exception as exc:
        print(f"self_musa_time_total table unavailable: {type(exc).__name__}: {exc}")
        print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=20))
    print("INTERESTING_EVENTS_JSON")
    print(json.dumps(summarize_events(prof), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
