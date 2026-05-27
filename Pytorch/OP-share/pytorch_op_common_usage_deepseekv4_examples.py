"""CPU-only PyTorch OP examples mapped to DeepSeekV4 source patterns.

This script supports the accompanying technical note. It does not import
Transformers and does not run the real DeepSeekV4 model. It uses small tensors
to demonstrate the common PyTorch OP patterns that appear in DeepSeekV4:

1. embedding / indexing
2. view / transpose / contiguous
3. linear / matmul / bmm
4. RMSNorm-style elementwise + reduction
5. RoPE-style split / rotate / concat
6. compression by view + softmax + weighted sum
7. mask construction by arange + masked_fill
8. MoE routing by topk / gather / one_hot / index_add_
"""

import torch
import torch.nn.functional as F


def show(name, tensor):
    print(f"{name:<28} shape={tuple(tensor.shape)}, value={tensor}")


def section(title):
    print(f"\n=== {title} ===")


def rotate_half(x):
    """Rotate adjacent feature pairs, matching the core RoPE helper behavior."""
    left = x[..., 0::2]
    right = x[..., 1::2]
    return torch.stack((-right, left), dim=-1).flatten(-2)


def main():
    torch.set_printoptions(precision=3, sci_mode=False)

    section("1. embedding / indexing")
    # Embedding lookup maps integer token ids to dense hidden vectors.
    input_ids = torch.tensor([1, 3, 2, 4])
    embedding_table = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    hidden = embedding_table[input_ids]
    show("input_ids", input_ids)
    show("hidden", hidden)

    section("2. view / transpose / contiguous")
    # view changes logical shape when the original stride is compatible.
    q = torch.arange(1, 17, dtype=torch.float32).view(2, 2, 4)
    # transpose returns a view with changed stride; it is usually non-contiguous.
    q_t = q.transpose(1, 2)
    # contiguous materializes the transposed view into continuous storage.
    q_contiguous = q_t.contiguous()
    show("q", q)
    show("q.transpose(1, 2)", q_t)
    print(f"transpose is_contiguous={q_t.is_contiguous()}")
    print(f"after contiguous is_contiguous={q_contiguous.is_contiguous()}")

    section("3. linear / matmul / bmm")
    # F.linear projects the last dimension and is the base form of Q/K/V, MLP,
    # LM head, and expert projections.
    weight = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    linear_out = F.linear(hidden, weight)
    # matmul can express attention scores or similarity scores.
    attn_scores = hidden @ hidden.T
    # bmm runs one matrix multiply per batch item; grouped projections use this pattern.
    grouped_x = torch.arange(1, 9, dtype=torch.float32).view(2, 2, 2)
    grouped_w = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]])
    bmm_out = torch.bmm(grouped_x, grouped_w)
    show("F.linear(hidden, weight)", linear_out)
    show("hidden @ hidden.T", attn_scores)
    show("torch.bmm(grouped_x, w)", bmm_out)

    section("4. RMSNorm-style ops")
    # RMSNorm decomposes into elementwise square, reduction mean, rsqrt, and multiply.
    rms = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + 1e-6)
    show("hidden.square()", hidden.square())
    show("RMSNorm output", rms)

    section("5. RoPE-style split / rotate / concat")
    # DeepSeekV4 applies RoPE only to the tail slice of each head.
    rope_input = torch.tensor([[10.0, 20.0, 1.0, 2.0]])
    cos = torch.tensor([[0.8]])
    sin = torch.tensor([[0.6]])
    cos_full = cos.repeat_interleave(2, dim=-1)
    sin_full = sin.repeat_interleave(2, dim=-1)
    nope, rope = rope_input[..., :-2], rope_input[..., -2:]
    rotated_rope = rope * cos_full + rotate_half(rope) * sin_full
    rope_out = torch.cat([nope, rotated_rope], dim=-1)
    show("rope_input [nope|rope]", rope_input)
    show("rotated rope slice", rotated_rope)
    show("concat output", rope_out)

    section("6. compression: view + softmax + weighted sum")
    # Compression groups tokens into windows and computes one weighted summary per window.
    tokens = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    gate = torch.tensor(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
            [0.0, 2.0],
        ]
    )
    windows = tokens.view(2, 2, 2)
    gate_windows = gate.view(2, 2, 2)
    # softmax produces per-window weights; weighted sum produces compressed entries.
    weights = gate_windows.softmax(dim=1)
    compressed = (windows * weights).sum(dim=1)
    show("windows", windows)
    show("softmax weights", weights)
    show("compressed entries", compressed)

    section("7. mask: arange + masked_fill")
    # A block bias uses 0 for visible compressed entries and -inf for blocked entries.
    seq_len = 4
    compressed_len = 2
    compress_rate = 2
    position_ids = torch.arange(seq_len)
    entry_indices = torch.arange(compressed_len)
    causal_threshold = (position_ids + 1) // compress_rate
    block_bias = torch.zeros(seq_len, compressed_len)
    # Each query position can only see completed compressed entries.
    block_bias = block_bias.masked_fill(entry_indices.view(1, -1) >= causal_threshold.view(-1, 1), float("-inf"))
    show("causal_threshold", causal_threshold)
    show("block_bias", block_bias)

    section("8. MoE routing: topk / gather / one_hot / index_add_")
    # Router scores select the top-k experts for each token.
    scores = torch.tensor(
        [
            [0.1, 0.9, 0.2],
            [0.8, 0.3, 0.4],
            [0.2, 0.5, 0.7],
        ]
    )
    topk_values, topk_indices = scores.topk(k=2, dim=-1)
    # Normalize the selected expert weights per token.
    weights = topk_values / topk_values.sum(dim=-1, keepdim=True)
    token_hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    expert_scale = torch.tensor([1.0, 2.0, 3.0])
    # one_hot + permute converts token -> top-k experts into expert -> token lookup.
    mask = F.one_hot(topk_indices, num_classes=3).permute(2, 1, 0)
    final = torch.zeros_like(token_hidden)
    for expert_id in range(3):
        topk_pos, token_idx = torch.where(mask[expert_id])
        if token_idx.numel() == 0:
            continue
        # This multiplication stands in for the selected expert's projection stack.
        current = token_hidden[token_idx] * expert_scale[expert_id]
        current = current * weights[token_idx, topk_pos].unsqueeze(-1)
        # index_add_ combines expert outputs back to the original token positions.
        final.index_add_(0, token_idx, current)
    show("topk_indices", topk_indices)
    show("topk weights", weights)
    show("MoE combined output", final)


if __name__ == "__main__":
    main()
