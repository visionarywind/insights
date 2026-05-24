import torch
import torch.nn.functional as F
device = torch.device("musa:0")
x = torch.tensor([[0,1,2],[3,4,5],[6,7,8]], device="musa:0")
rows = torch.tensor([0, 2], device="musa:0")
cols = torch.tensor([1, 2], device="musa:0")
y = x[rows, cols]

def _fmt_tensor(t, with_value=True):
    s = f"Tensor(shape={tuple(t.shape)}, dtype={str(t.dtype).replace('torch.', '')}, device={t.device}"
    if with_value:
        s += f", value={t.detach().cpu().tolist()}"
    return s + ")"

def _fmt_value(v, with_value=True):
    if isinstance(v, torch.Tensor):
        return _fmt_tensor(v, with_value)
    if hasattr(v, "shape") and hasattr(v, "dtype") and hasattr(v, "tolist"):
        return f"ndarray(shape={v.shape}, dtype={v.dtype}, value={v.tolist()})"
    return v

print("x =", _fmt_value(x, True))
print("rows =", _fmt_value(rows, True))
print("cols =", _fmt_value(cols, True))
print("y =", _fmt_value(y, True))
