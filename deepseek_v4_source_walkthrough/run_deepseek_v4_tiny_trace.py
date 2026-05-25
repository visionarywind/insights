"""DeepSeekV4 完整流程模拟用例。

用例设计：
    这个脚本不导入 Transformers、PyTorch、模型权重或 GPU 后端。
    它使用 4 个 token、2 维 hidden vector 和 3 层简化模块，完整模拟一次
    DeepSeekV4 风格的前向流程。

输出内容：
    脚本会打印每一步的输入、规则、shape、中间值和输出。重点展示：
    1. token id 如何查 embedding 表
    2. mHC 如何展开、折叠和回写多残差流
    3. sliding attention 如何只看局部窗口
    4. HCA 如何压缩窗口并按因果规则访问摘要
    5. CSA 如何压缩窗口并用 indexer 选择摘要
    6. MoE 如何 router 打分、选专家、加权合并、叠加 shared expert
    7. lm_head 如何把最终 hidden vector 转成词表分数

源码模块对应关系：
    embed_tokens        -> DeepseekV4Model.embed_tokens
    mHC collapse/mix    -> DeepseekV4HyperConnection / DeepseekV4HyperHead
    sliding attention   -> sliding_attention layer
    HCA compression     -> DeepseekV4HCACompressor
    CSA + indexer       -> DeepseekV4CSACompressor / DeepseekV4Indexer
    MoE router/experts  -> DeepseekV4SparseMoeBlock
    lm_head             -> DeepseekV4ForCausalLM.lm_head

执行方式：
    cd /home/mtuser/workspace
    python insights/deepseek_v4_source_walkthrough/run_deepseek_v4_tiny_trace.py
"""


def fmt_num(value):
    return f"{value:.3f}".rstrip("0").rstrip(".")


def fmt_vec(vector):
    return "[" + ", ".join(fmt_num(value) for value in vector) + "]"


def fmt_vectors(vectors):
    return "[" + ", ".join(fmt_vec(vector) for vector in vectors) + "]"


def avg_component_details(vectors, result):
    """Return per-dimension arithmetic for an average over vectors."""
    details = []
    size = len(vectors)
    for dim_idx in range(len(result)):
        terms = " + ".join(fmt_num(vector[dim_idx]) for vector in vectors)
        details.append(f"dim{dim_idx}: ({terms}) / {size} = {fmt_num(result[dim_idx])}")
    return details


def print_avg_calculation(indent, title, vectors, result):
    """Print the arithmetic behind an average operation."""
    print(f"{indent}{title}")
    for line in avg_component_details(vectors, result):
        print(f"{indent}  {line}")


def weighted_sum_component_details(items, result):
    """Return per-dimension arithmetic for sum(weight * vector)."""
    details = []
    for dim_idx in range(len(result)):
        terms = " + ".join(f"{fmt_num(weight)}*{fmt_num(vector[dim_idx])}" for vector, weight in items)
        details.append(f"dim{dim_idx}: {terms} = {fmt_num(result[dim_idx])}")
    return details


def dot_detail(a, b, result):
    """Return arithmetic for a dot product."""
    terms = " + ".join(f"{fmt_num(x)}*{fmt_num(y)}" for x, y in zip(a, b))
    return f"{terms} = {fmt_num(result)}"


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def scale(vector, factor):
    return [value * factor for value in vector]


def avg(vectors):
    size = len(vectors)
    return [sum(vector[i] for vector in vectors) / size for i in range(len(vectors[0]))]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def weighted_sum(items):
    result = [0.0 for _ in items[0][0]]
    for vector, weight in items:
        result = add(result, scale(vector, weight))
    return result


def section(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def substep(title):
    print(f"\n-- {title}")


def show_rule(title, lines):
    print(f"  {title}")
    for line in lines:
        print(f"    {line}")


def show_sequence(title, vectors):
    print(title)
    print(f"  shape = [tokens={len(vectors)}, dim={len(vectors[0])}]")
    for index, vector in enumerate(vectors):
        print(f"  token {index}: {fmt_vec(vector)}")
    print()


def show_streams(title, streams):
    print(title)
    print(f"  shape = [tokens={len(streams)}, streams={len(streams[0])}, dim={len(streams[0][0])}]")
    for token_idx, token_streams in enumerate(streams):
        stream_text = ", ".join(f"s{stream_idx}={fmt_vec(vector)}" for stream_idx, vector in enumerate(token_streams))
        print(f"  token {token_idx}: {stream_text}")
    print()


def print_source_mapping():
    section("Source mapping")
    print("  embed_tokens        -> DeepseekV4Model.embed_tokens")
    print("  mHC collapse/mix    -> DeepseekV4HyperConnection / DeepseekV4HyperHead")
    print("  sliding attention   -> sliding_attention layer")
    print("  HCA compression     -> DeepseekV4HCACompressor")
    print("  CSA + indexer       -> DeepseekV4CSACompressor / DeepseekV4Indexer")
    print("  MoE router/experts  -> DeepseekV4SparseMoeBlock")
    print("  lm_head             -> DeepseekV4ForCausalLM.lm_head")


def embedding_table():
    return {
        1: [1.0, 0.0],
        2: [0.0, 1.0],
        3: [1.0, 1.0],
        4: [2.0, 1.0],
    }


def embed_tokens(input_ids):
    """模拟 embedding 查表：把离散 token id 转成连续 hidden vector。"""
    table = embedding_table()

    substep("Step 1. Embedding lookup")
    show_rule(
        "Rule",
        [
            "input_ids 是整数 token id。",
            "embedding_table[token_id] 返回该 token 的 hidden vector。",
            "真实源码位置：DeepseekV4Model.forward -> self.embed_tokens(input_ids)。",
        ],
    )

    print("  Embedding table:")
    for token_id, vector in table.items():
        print(f"    id {token_id} -> {fmt_vec(vector)}")

    hidden_states = []
    print("  Lookup result:")
    for position, token_id in enumerate(input_ids):
        vector = table[token_id]
        hidden_states.append(vector)
        print(f"    position {position}: token_id={token_id} -> hidden={fmt_vec(vector)}")
    print()
    return hidden_states


def expand_mhc_streams(hidden_states):
    """模拟 mHC 输入形态：每个 token 保留两条残差流。"""
    substep("Step 2. mHC stream expansion")
    show_rule(
        "Rule",
        [
            "每个 token 保留 2 条残差流。",
            "stream0 = hidden。",
            "stream1 = hidden + [0.1, -0.1]。",
            "真实源码中 hc_mult 控制残差流数量。",
        ],
    )

    streams = []
    for token_idx, vector in enumerate(hidden_states):
        stream_0 = vector
        stream_1 = add(vector, [0.1, -0.1])
        streams.append([stream_0, stream_1])
        print(
            f"  token {token_idx}: hidden={fmt_vec(vector)} -> "
            f"stream0={fmt_vec(stream_0)}, stream1={fmt_vec(stream_1)}"
        )
    print()
    return streams


def mhc_collapse(streams):
    """多条残差流合成一个子层输入。"""
    return [avg(token_streams) for token_streams in streams]


def trace_mhc_collapse(title, streams):
    """打印 mHC collapse 的每个 token 计算过程。"""
    substep(title)
    show_rule(
        "Rule",
        [
            "每个 token 的多条残差流先合成一个向量。",
            "本用例用平均值模拟真实 mHC 的 learned mixing。",
        ],
    )

    collapsed = []
    for token_idx, token_streams in enumerate(streams):
        vector = avg(token_streams)
        collapsed.append(vector)
        print(f"  token {token_idx}: avg({fmt_vectors(token_streams)}) = {fmt_vec(vector)}")
        print_avg_calculation("    ", "component calculation:", token_streams, vector)
    print()
    return collapsed


def mhc_mix_back(old_streams, sublayer_output):
    """子层输出回写到多条残差流。"""
    mixed = []
    for token_streams, output in zip(old_streams, sublayer_output):
        stream_0 = add(scale(token_streams[0], 0.7), scale(output, 0.3))
        stream_1 = add(scale(token_streams[1], 0.4), scale(output, 0.6))
        mixed.append([stream_0, stream_1])
    return mixed


def trace_mhc_mix_back(title, old_streams, sublayer_output):
    """打印 mHC mix back 的每个 token 计算过程。"""
    substep(title)
    show_rule(
        "Rule",
        [
            "子层输出要写回多条残差流。",
            "stream0_new = 0.7 * stream0_old + 0.3 * sublayer_output。",
            "stream1_new = 0.4 * stream1_old + 0.6 * sublayer_output。",
        ],
    )

    mixed = mhc_mix_back(old_streams, sublayer_output)
    for token_idx, (old, output, new) in enumerate(zip(old_streams, sublayer_output, mixed)):
        print(f"  token {token_idx}:")
        print(f"    old stream0={fmt_vec(old[0])}, old stream1={fmt_vec(old[1])}")
        print(f"    sublayer_output={fmt_vec(output)}")
        print(f"    new stream0={fmt_vec(new[0])}")
        for dim_idx, value in enumerate(new[0]):
            print(
                f"      dim{dim_idx}: 0.7*{fmt_num(old[0][dim_idx])} + "
                f"0.3*{fmt_num(output[dim_idx])} = {fmt_num(value)}"
            )
        print(f"    new stream1={fmt_vec(new[1])}")
        for dim_idx, value in enumerate(new[1]):
            print(
                f"      dim{dim_idx}: 0.4*{fmt_num(old[1][dim_idx])} + "
                f"0.6*{fmt_num(output[dim_idx])} = {fmt_num(value)}"
            )
    print()
    return mixed


def local_context(hidden_states, position, window_size=2):
    """模拟 sliding window attention 的局部上下文。"""
    start = max(0, position - window_size + 1)
    return start, hidden_states[start : position + 1], avg(hidden_states[start : position + 1])


def compress_windows(hidden_states, compress_rate, name):
    """固定窗口内的多个 token 聚合为一个摘要。"""
    print(f"    {name} compression rule: every {compress_rate} tokens -> 1 compressed entry")
    compressed = []
    for start in range(0, len(hidden_states), compress_rate):
        end = min(start + compress_rate, len(hidden_states))
        window = hidden_states[start:end]
        summary = avg(window)
        compressed.append(summary)
        print(f"      tokens {start}-{end - 1}: avg({fmt_vectors(window)}) = entry{len(compressed) - 1} {fmt_vec(summary)}")
        print_avg_calculation("        ", "component calculation:", window, summary)
    return compressed


def sliding_attention(hidden_states):
    """第 0 层：只使用局部滑窗上下文。"""
    show_rule(
        "Sliding attention",
        [
            "每个 token 只看自己和前一个 token。",
            "本用例用窗口平均值模拟 attention 输出。",
            "真实模型中这里是 QK^T -> softmax -> weighted sum(V)。",
        ],
    )

    output = []
    for position in range(len(hidden_states)):
        start, window, local = local_context(hidden_states, position)
        output.append(local)
        print(
            f"    token {position}: window=tokens {start}-{position}, "
            f"values={fmt_vectors(window)} -> local_avg={fmt_vec(local)} -> out={fmt_vec(local)}"
        )
        print_avg_calculation("      ", "local_avg calculation:", window, local)
    return output


def hca_attention(hidden_states, compress_rate=2):
    """第 1 层：局部上下文 + 已完成的 HCA 压缩摘要。"""
    show_rule(
        "HCA attention",
        [
            "先按窗口生成 compressed entries。",
            "token 只能访问已经完成的 compressed entries。",
            "本用例用 avg(local_context, visible_compressed_context) 模拟输出。",
        ],
    )
    compressed = compress_windows(hidden_states, compress_rate, "HCA")

    output = []
    for position in range(len(hidden_states)):
        start, window, local = local_context(hidden_states, position)
        visible_count = (position + 1) // compress_rate
        visible = compressed[:visible_count]

        if visible:
            long_context = avg(visible)
            out = avg([local, long_context])
            print(f"    token {position}:")
            print(f"      local window=tokens {start}-{position}, values={fmt_vectors(window)}, local={fmt_vec(local)}")
            print_avg_calculation("        ", "local calculation:", window, local)
            print(f"      visible HCA entries=entry0-entry{visible_count - 1}, context={fmt_vec(long_context)}")
            print_avg_calculation("        ", "compressed context calculation:", visible, long_context)
            print(f"      out=avg(local, context)={fmt_vec(out)}")
            print_avg_calculation("        ", "output calculation:", [local, long_context], out)
        else:
            out = local
            print(f"    token {position}:")
            print(f"      local window=tokens {start}-{position}, values={fmt_vectors(window)}, local={fmt_vec(local)}")
            print_avg_calculation("        ", "local calculation:", window, local)
            print("      visible HCA entries=none")
            print(f"      out=local={fmt_vec(out)}")
        output.append(out)
    return output


def csa_attention(hidden_states, compress_rate=2):
    """第 2 层：局部上下文 + CSA 压缩摘要 + indexer 选择。"""
    show_rule(
        "CSA attention + indexer",
        [
            "先按窗口生成 compressed entries。",
            "indexer 用 query 和 entry 做 dot score。",
            "本用例选择得分最高的 compressed entry。",
            "输出用 avg(local_context, selected_entry) 模拟。",
        ],
    )
    compressed = compress_windows(hidden_states, compress_rate, "CSA")

    output = []
    for position, query in enumerate(hidden_states):
        start, window, local = local_context(hidden_states, position)
        visible_count = (position + 1) // compress_rate
        visible = compressed[:visible_count]

        print(f"    token {position}:")
        print(f"      query={fmt_vec(query)}")
        print(f"      local window=tokens {start}-{position}, values={fmt_vectors(window)}, local={fmt_vec(local)}")
        print_avg_calculation("        ", "local calculation:", window, local)

        if not visible:
            output.append(local)
            print("      visible CSA entries=none")
            print(f"      out=local={fmt_vec(local)}")
            continue

        scores = [dot(query, entry) for entry in visible]
        selected_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        selected_entry = visible[selected_idx]
        out = avg([local, selected_entry])

        for entry_idx, (entry, score) in enumerate(zip(visible, scores)):
            print(f"      indexer score: dot(query, entry{entry_idx}={fmt_vec(entry)}) = {fmt_num(score)}")
            print(f"        calculation: {dot_detail(query, entry, score)}")
        print(f"      selected entry={selected_idx}, selected_value={fmt_vec(selected_entry)}")
        print(f"      out=avg(local, selected_entry)={fmt_vec(out)}")
        print_avg_calculation("        ", "output calculation:", [local, selected_entry], out)
        output.append(out)
    return output


def run_attention(hidden_states, layer_type):
    """根据层类型选择对应的简化 attention 流程。"""
    substep(f"Attention: {layer_type}")
    if layer_type == "sliding_attention":
        return sliding_attention(hidden_states)
    if layer_type == "heavily_compressed_attention":
        return hca_attention(hidden_states)
    if layer_type == "compressed_sparse_attention":
        return csa_attention(hidden_states)
    raise ValueError(f"unknown layer_type: {layer_type}")


def expert_output(expert_id, vector):
    """模拟单个 expert 的线性变换。"""
    x, y = vector
    if expert_id == 0:
        return [1.4 * x, 0.7 * y]
    if expert_id == 1:
        return [0.7 * x, 1.4 * y]
    return [1.1 * x, 1.1 * y]


def expert_rule(expert_id):
    if expert_id == 0:
        return "expert0: [x, y] -> [1.4x, 0.7y]"
    if expert_id == 1:
        return "expert1: [x, y] -> [0.7x, 1.4y]"
    return "expert2: [x, y] -> [1.1x, 1.1y]"


def moe_block(hidden_states):
    """模拟 MoE：router 选两个 expert，再与 shared expert 输出相加。"""
    substep("MoE router + experts")
    show_rule(
        "MoE rule",
        [
            "router_scores = [x, y, 0.5 * (x + y)]。",
            "每个 token 选择得分最高的 2 个 expert。",
            "selected expert 输出按 router weight 加权求和。",
            "shared expert = 0.2 * input。",
            "MoE output = routed expert output + shared expert output。",
        ],
    )
    print(f"    {expert_rule(0)}")
    print(f"    {expert_rule(1)}")
    print(f"    {expert_rule(2)}")

    output = []
    for token_idx, vector in enumerate(hidden_states):
        x, y = vector
        router_scores = [x, y, 0.5 * (x + y)]
        selected = sorted(range(len(router_scores)), key=lambda idx: router_scores[idx], reverse=True)[:2]
        selected_scores = [router_scores[idx] for idx in selected]
        score_sum = sum(selected_scores)
        weights = [score / score_sum for score in selected_scores]

        expert_items = []
        print(f"    token {token_idx}: input={fmt_vec(vector)}")
        print(f"      router_scores={fmt_vec(router_scores)}")
        print(f"        expert0 score = x = {fmt_num(x)}")
        print(f"        expert1 score = y = {fmt_num(y)}")
        print(
            f"        expert2 score = 0.5 * (x + y) = "
            f"0.5 * ({fmt_num(x)} + {fmt_num(y)}) = {fmt_num(router_scores[2])}"
        )
        print(f"      selected experts={selected}, selected_scores={fmt_vec(selected_scores)}")
        print(f"      selected score sum={fmt_num(score_sum)}")
        for expert_id, weight in zip(selected, weights):
            expert_vec = expert_output(expert_id, vector)
            expert_items.append((expert_vec, weight))
            print(f"      selected expert{expert_id}: weight={fmt_num(weight)}, output={fmt_vec(expert_vec)}")
            print(
                f"        weight calculation: score / score_sum = "
                f"{fmt_num(router_scores[expert_id])} / {fmt_num(score_sum)} = {fmt_num(weight)}"
            )
            if expert_id == 0:
                print(f"        output dim0: 1.4 * x = 1.4 * {fmt_num(x)} = {fmt_num(expert_vec[0])}")
                print(f"        output dim1: 0.7 * y = 0.7 * {fmt_num(y)} = {fmt_num(expert_vec[1])}")
            elif expert_id == 1:
                print(f"        output dim0: 0.7 * x = 0.7 * {fmt_num(x)} = {fmt_num(expert_vec[0])}")
                print(f"        output dim1: 1.4 * y = 1.4 * {fmt_num(y)} = {fmt_num(expert_vec[1])}")
            else:
                print(f"        output dim0: 1.1 * x = 1.1 * {fmt_num(x)} = {fmt_num(expert_vec[0])}")
                print(f"        output dim1: 1.1 * y = 1.1 * {fmt_num(y)} = {fmt_num(expert_vec[1])}")

        routed = weighted_sum(expert_items)
        shared = scale(vector, 0.2)
        combined = add(routed, shared)
        output.append(combined)

        print(f"      routed weighted sum={fmt_vec(routed)}")
        for line in weighted_sum_component_details(expert_items, routed):
            print(f"        {line}")
        print(f"      shared expert output={fmt_vec(shared)}")
        for dim_idx, value in enumerate(shared):
            print(f"        dim{dim_idx}: 0.2*{fmt_num(vector[dim_idx])} = {fmt_num(value)}")
        print(f"      MoE output={fmt_vec(combined)}")
        for dim_idx, value in enumerate(combined):
            print(f"        dim{dim_idx}: {fmt_num(routed[dim_idx])} + {fmt_num(shared[dim_idx])} = {fmt_num(value)}")
    return output


def lm_head(hidden_states):
    """模拟 lm_head：把 hidden vector 投影成 3 个词表分数。"""
    substep("lm_head projection")
    show_rule(
        "lm_head rule",
        [
            "toy vocab = [A, B, C]。",
            "score(A)=x。",
            "score(B)=y。",
            "score(C)=0.3*x + 0.7*y。",
        ],
    )

    logits = []
    for token_idx, vector in enumerate(hidden_states):
        x, y = vector
        scores = [x, y, 0.3 * x + 0.7 * y]
        logits.append(scores)
        print(f"  token {token_idx}: hidden={fmt_vec(vector)} -> logits={fmt_vec(scores)}")
        print(f"    score(A) = x = {fmt_num(x)}")
        print(f"    score(B) = y = {fmt_num(y)}")
        print(
            f"    score(C) = 0.3*x + 0.7*y = "
            f"0.3*{fmt_num(x)} + 0.7*{fmt_num(y)} = {fmt_num(scores[2])}"
        )
    return logits


def run_layer(layer_idx, layer_type, streams):
    # 每层都按 DeepSeekV4 decoder layer 的主线执行：
    # mHC collapse -> attention -> mHC mix back -> mHC collapse -> MoE -> mHC mix back。
    section(f"Layer {layer_idx}: {layer_type}")
    show_rule(
        "Layer execution order",
        [
            "1. mHC collapse before attention",
            "2. attention",
            "3. mHC mix back after attention",
            "4. mHC collapse before MoE",
            "5. MoE router + experts",
            "6. mHC mix back after MoE",
        ],
    )

    collapsed = trace_mhc_collapse("mHC collapse before attention", streams)
    show_sequence("  attention input", collapsed)

    attn_out = run_attention(collapsed, layer_type)
    show_sequence("  attention output", attn_out)

    streams = trace_mhc_mix_back("mHC mix back after attention", streams, attn_out)
    show_streams("  streams after attention", streams)

    collapsed = trace_mhc_collapse("mHC collapse before MoE", streams)
    show_sequence("  MoE input", collapsed)

    moe_out = moe_block(collapsed)
    show_sequence("  MoE output", moe_out)

    streams = trace_mhc_mix_back("mHC mix back after MoE", streams, moe_out)
    show_streams("  streams after MoE", streams)
    return streams


def main():
    # 用例输入：4 个 token id。每个 token 会被映射成 2 维 hidden vector。
    input_ids = [1, 2, 3, 4]

    # 三层分别覆盖 DeepSeekV4 中最关键的三类 attention 路径。
    layer_types = [
        "sliding_attention",
        "heavily_compressed_attention",
        "compressed_sparse_attention",
    ]

    section("DeepSeekV4 complete process simulation")
    print("  This is a small numeric simulation, not the real model.")
    print("  No Transformers, no PyTorch, no model weights, no GPU.\n")
    print("  Example scale:")
    print(f"    input tokens = {len(input_ids)}")
    print("    hidden dim   = 2")
    print("    mHC streams  = 2")
    print(f"    layers       = {len(layer_types)}")

    print_source_mapping()

    section("Step 0. Input")
    print(f"  input_ids = {input_ids}")
    for position, token_id in enumerate(input_ids):
        print(f"  position {position}: token_id={token_id}")

    # Step 1 对应 DeepseekV4Model.forward 中的 self.embed_tokens(input_ids)。
    hidden_states = embed_tokens(input_ids)
    show_sequence("Embedding output", hidden_states)

    # Step 2 对应 DeepSeekV4 的 mHC 多残差流输入形态。
    streams = expand_mhc_streams(hidden_states)
    show_streams("mHC stream output", streams)

    # Step 3 依次执行 3 个 decoder layer，每层包含 attention 和 MoE。
    for layer_idx, layer_type in enumerate(layer_types):
        streams = run_layer(layer_idx, layer_type, streams)

    # Step 4 对应 DeepseekV4HyperHead：把多条残差流折叠回最终 hidden states。
    section("Final mHC head")
    final_hidden = trace_mhc_collapse("mHC head collapse", streams)
    show_sequence("Final hidden states", final_hidden)

    # Step 5 对应 DeepseekV4ForCausalLM.lm_head。
    section("Final lm_head")
    logits = lm_head(final_hidden)
    labels = ["A", "B", "C"]
    print("\nPrediction result")
    print(f"  shape = [tokens={len(logits)}, vocab={len(logits[0])}]")
    for token_idx, scores in enumerate(logits):
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])
        print(f"  token {token_idx}: logits={fmt_vec(scores)} -> predict {labels[best_idx]}")


if __name__ == "__main__":
    main()
