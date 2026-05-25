"""No-dependency DeepSeek-V4 toy flow.

This is not the real model. It only prints the beginner-level data flow and
shape changes, so you can understand the big picture before reading PyTorch
source code.

Run:
    python insights/deepseek_v4_source_walkthrough/toy_flow_no_torch.py
"""


class FakeTensor:
    def __init__(self, name, shape):
        self.name = name
        self.shape = tuple(shape)

    def show(self, note=""):
        suffix = f"  # {note}" if note else ""
        print(f"{self.name:<28} {list(self.shape)}{suffix}")


def embedding(input_ids, hidden_size):
    batch, seq_len = input_ids.shape
    return FakeTensor("embed_tokens", [batch, seq_len, hidden_size])


def expand_hc_streams(hidden_states, hc_mult):
    batch, seq_len, hidden_size = hidden_states.shape
    return FakeTensor("expand_hc_streams", [batch, seq_len, hc_mult, hidden_size])


def mhc_collapse(hidden_streams):
    batch, seq_len, _hc_mult, hidden_size = hidden_streams.shape
    return FakeTensor("mHC collapse", [batch, seq_len, hidden_size])


def mhc_mix_back(one_stream, hc_mult):
    batch, seq_len, hidden_size = one_stream.shape
    return FakeTensor("mHC mix back", [batch, seq_len, hc_mult, hidden_size])


def attention(hidden_states, layer_type, compress_rate=None, index_topk=None):
    batch, seq_len, hidden_size = hidden_states.shape
    print(f"  attention type: {layer_type}")
    print("  local branch: keep nearby tokens with sliding window")

    if layer_type == "sliding_attention":
        print("  long branch : none")
    elif layer_type == "heavily_compressed_attention":
        entries = seq_len // compress_rate
        print(f"  HCA branch  : compress every {compress_rate} tokens -> {entries} compressed entries")
    elif layer_type == "compressed_sparse_attention":
        entries = seq_len // compress_rate
        print(f"  CSA branch  : compress every {compress_rate} tokens -> {entries} compressed entries")
        print(f"  indexer     : each query picks top {index_topk} compressed entries")

    return FakeTensor("attention output", [batch, seq_len, hidden_size])


def moe(hidden_states, num_experts, experts_per_token):
    batch, seq_len, hidden_size = hidden_states.shape
    print(f"  router      : {seq_len} tokens choose {experts_per_token} of {num_experts} experts")
    print("  experts     : selected experts process tokens")
    print("  shared mlp  : every token also goes through one shared MLP")
    return FakeTensor("moe output", [batch, seq_len, hidden_size])


def lm_head(hidden_states, vocab_size):
    batch, seq_len, _hidden_size = hidden_states.shape
    return FakeTensor("lm_head logits", [batch, seq_len, vocab_size])


def run_layer(layer_idx, hidden_streams, layer_type, hc_mult, num_experts, experts_per_token):
    print(f"\nlayer {layer_idx}: {layer_type}")

    collapsed = mhc_collapse(hidden_streams)
    collapsed.show("multi-stream hidden states become one stream for attention")
    if layer_type == "heavily_compressed_attention":
        attn_out = attention(collapsed, layer_type, compress_rate=4)
    elif layer_type == "compressed_sparse_attention":
        attn_out = attention(collapsed, layer_type, compress_rate=2, index_topk=2)
    else:
        attn_out = attention(collapsed, layer_type)
    attn_out.show()

    hidden_streams = mhc_mix_back(attn_out, hc_mult)
    hidden_streams.show("attention output returns to hc_mult streams")

    collapsed = mhc_collapse(hidden_streams)
    collapsed.show("multi-stream hidden states become one stream for MoE")
    moe_out = moe(collapsed, num_experts=num_experts, experts_per_token=experts_per_token)
    moe_out.show()

    hidden_streams = mhc_mix_back(moe_out, hc_mult)
    hidden_streams.show("MoE output returns to hc_mult streams")
    return hidden_streams


def main():
    batch = 1
    seq_len = 8
    hidden_size = 16
    hc_mult = 2
    vocab_size = 32
    num_experts = 4
    experts_per_token = 2

    print("DeepSeek-V4 toy flow, no torch, no GPU\n")

    input_ids = FakeTensor("input_ids", [batch, seq_len])
    input_ids.show("8 token ids")

    hidden_states = embedding(input_ids, hidden_size)
    hidden_states.show("each token becomes a 16-number vector")

    hidden_streams = expand_hc_streams(hidden_states, hc_mult)
    hidden_streams.show("V4 keeps 2 residual streams per token")

    layer_types = [
        "sliding_attention",
        "heavily_compressed_attention",
        "compressed_sparse_attention",
    ]
    for layer_idx, layer_type in enumerate(layer_types):
        hidden_streams = run_layer(
            layer_idx,
            hidden_streams,
            layer_type,
            hc_mult=hc_mult,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
        )

    print("\nfinal")
    final_hidden = mhc_collapse(hidden_streams)
    final_hidden.name = "HyperHead output"
    final_hidden.show("collapse hc_mult streams back to one hidden state")

    logits = lm_head(final_hidden, vocab_size)
    logits.show("for every token position, score every vocab id")


if __name__ == "__main__":
    main()
