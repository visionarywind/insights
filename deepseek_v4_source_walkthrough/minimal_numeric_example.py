"""A tiny numeric example for understanding DeepSeek-V4-style flow.

This is not the real DeepSeek-V4 model. It is a four-token, two-dimensional
toy program that keeps only the ideas:

1. token id -> embedding vector
2. attention = local window + compressed summary
3. MoE = router chooses one expert + shared expert
4. lm_head = turn final vectors into next-token scores

Run:
    python insights/deepseek_v4_source_walkthrough/minimal_numeric_example.py
"""


def fmt_vec(vector):
    return "[" + ", ".join(f"{value:.3g}" for value in vector) + "]"


def avg(vectors):
    size = len(vectors)
    return [sum(vector[i] for vector in vectors) / size for i in range(len(vectors[0]))]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def scale(vector, factor):
    return [x * factor for x in vector]


def print_vectors(title, vectors):
    print(title)
    for i, vector in enumerate(vectors, start=1):
        print(f"  pos {i}: {fmt_vec(vector)}")
    print()


def main():
    input_ids = [1, 2, 3, 4]
    embedding_table = {
        1: [1.0, 0.0],
        2: [0.0, 1.0],
        3: [1.0, 1.0],
        4: [2.0, 1.0],
    }

    print("Step 0. Input token ids")
    print(f"  input_ids = {input_ids}\n")

    hidden = [embedding_table[token_id] for token_id in input_ids]
    print_vectors("Step 1. Embedding: token id -> 2-number vector", hidden)

    print("Step 2. mHC idea: V4 keeps multiple residual streams")
    print("  toy setting: hc_mult = 2")
    print("  stream A and stream B are two drafts of the same sequence")
    print("  before attention/MoE, we collapse them back to one vector per token\n")

    compress_rate = 2
    compressed = []
    print("Step 3. Compression: every 2 tokens become 1 summary")
    for start in range(0, len(hidden), compress_rate):
        block = hidden[start : start + compress_rate]
        summary = avg(block)
        compressed.append(summary)
        print(f"  tokens {start + 1}-{start + len(block)} -> summary {len(compressed)} = {fmt_vec(summary)}")
    print()

    print("Step 4. Attention: local nearby tokens + visible compressed summaries")
    attention_out = []
    for pos in range(len(hidden)):
        local_start = max(0, pos - 1)
        local_vectors = hidden[local_start : pos + 1]
        local = avg(local_vectors)

        visible_summary_count = (pos + 1) // compress_rate
        visible_summaries = compressed[:visible_summary_count]
        if visible_summaries:
            long_range = avg(visible_summaries)
            out = avg([local, long_range])
            long_text = fmt_vec(long_range)
        else:
            out = local
            long_text = "none"

        attention_out.append(out)
        print(
            f"  pos {pos + 1}: local={fmt_vec(local)}, "
            f"compressed={long_text} -> attention_out={fmt_vec(out)}"
        )
    print()

    print("Step 5. MoE: each token chooses one expert, plus one shared expert")
    print("  expert 0 doubles feature 0: [x, y] -> [2x, y]")
    print("  expert 1 doubles feature 1: [x, y] -> [x, 2y]")
    print("  shared expert adds half of the input: [x, y] -> [0.5x, 0.5y]")
    moe_out = []
    for token_id, vector in zip(input_ids, attention_out):
        if token_id % 2 == 1:
            expert_id = 0
            expert_out = [2 * vector[0], vector[1]]
        else:
            expert_id = 1
            expert_out = [vector[0], 2 * vector[1]]

        shared_out = scale(vector, 0.5)
        out = add(expert_out, shared_out)
        moe_out.append(out)
        print(
            f"  token {token_id}: choose expert {expert_id}, "
            f"expert_out={fmt_vec(expert_out)}, shared={fmt_vec(shared_out)} -> moe_out={fmt_vec(out)}"
        )
    print()

    print("Step 6. lm_head: turn each final vector into next-token scores")
    print("  toy vocab has 3 possible next tokens: A, B, C")
    print("  score(A)=x, score(B)=y, score(C)=0.3*x+0.7*y")
    labels = ["A", "B", "C"]
    for pos, vector in enumerate(moe_out, start=1):
        x, y = vector
        logits = [x, y, 0.3 * x + 0.7 * y]
        best_idx = max(range(len(logits)), key=lambda idx: logits[idx])
        print(
            f"  pos {pos}: final={fmt_vec(vector)}, "
            f"logits={fmt_vec(logits)} -> predict {labels[best_idx]}"
        )

    print("\nWhat this example maps to in the real source:")
    print("  Step 1 -> embed_tokens")
    print("  Step 2 -> DeepseekV4HyperConnection / DeepseekV4HyperHead")
    print("  Step 3 -> DeepseekV4HCACompressor or DeepseekV4CSACompressor")
    print("  Step 4 -> DeepseekV4Attention")
    print("  Step 5 -> DeepseekV4SparseMoeBlock")
    print("  Step 6 -> DeepseekV4ForCausalLM.lm_head")


if __name__ == "__main__":
    main()
